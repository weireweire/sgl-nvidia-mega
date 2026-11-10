# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""One rank's SM103 kernels for the routed step, all built and compiled at build:
the Lamport mailbox and its consumer, the publishing wo_b GEMM per row count, and a
routed-expert GEMM pair per MoE layer. Nothing here allocates collective memory or
compiles inside a step."""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.distributed as dist

from sgl_nvidia_mega.deepseek_v41.sm103 import all_reduce_mhc
from sgl_nvidia_mega.deepseek_v41.sm103.moe.routed_expert_gemm import RoutedExpertGemm
from sgl_nvidia_mega.deepseek_v41.sm103.wo_b_publish import PublishingMxfp8Linear

logger = logging.getLogger(__name__)

# rank:site:after — that rank skips its publish at the given site after that many
# calls, so its peers' waits must end at their deadline (eager steps only: a
# captured graph replays without this counter)
FAULT_INJECT = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_FAULT_INJECT", "")


def _fault_inject(rank: int, site: str) -> Optional[int]:
    if not FAULT_INJECT:
        return None
    r, s, after = FAULT_INJECT.split(":")
    return int(after) if int(r) == rank and s == site else None


class Runner:
    def __init__(self, *, device: torch.device, rank: int):
        self.device, self.rank = device, rank
        self.hop: Optional[all_reduce_mhc.AllReduceMHC] = None
        self.hop_wide: Optional[all_reduce_mhc.AllReduceMHC] = None
        self.publisher: Optional[PublishingMxfp8Linear] = None
        self.moe: dict[int, RoutedExpertGemm] = {}
        self._inject_after = _fault_inject(rank, "hop_a")
        self._hop_calls = 0

    def build_hop(
        self,
        *,
        group: dist.ProcessGroup,
        hidden: int,
        hc_mult: int,
        top_k: int,
        hop_a_rows: int,
        deadline_s: float,
        hop_b_wide_rows: int = 0,
        hop_b: bool = False,
        early_trigger: bool = False,
        hop_b_start_trigger: bool = False,
        row_tile_layer_input: bool = False,
        wide_row_tile_layer_input: bool = False,
        publish_trigger_max_rows: int = 0,
        mbarrier_exchange: bool = False,
        final_collapse: bool = False,
    ) -> None:
        """The Lamport mailbox (collective over ``group``) and the publishing GEMM;
        with ``hop_b`` also the MoE finalize's quantized and streams-only consumers."""
        # before the consumer compiles: the deadline is a constant of its kernel
        all_reduce_mhc.set_wait_deadline(deadline_s)
        self.hop = all_reduce_mhc.AllReduceMHC(
            hidden_size=hidden,
            hc_mult=hc_mult,
            max_num_tokens=hop_a_rows,
            top_k=top_k,
            device=self.device,
            group=group,
            quantize_layer_input=hop_b,
            residual_only=hop_b,
            early_trigger=early_trigger,
            hop_b_start_trigger=hop_b_start_trigger,
            row_tile_layer_input=row_tile_layer_input,
            publish_trigger_max_rows=publish_trigger_max_rows,
            mbarrier_exchange=mbarrier_exchange,
            final_collapse=final_collapse,
        )
        self.publisher = PublishingMxfp8Linear(self.hop)
        # slot hop_b_wide: hop B past hop A's widths on a mailbox of its own, so hop A's
        # widths keep addressing the small one
        self.hop_wide = (
            all_reduce_mhc.AllReduceMHC(
                hidden_size=hidden,
                hc_mult=hc_mult,
                max_num_tokens=hop_b_wide_rows,
                top_k=top_k,
                device=self.device,
                group=group,
                quantize_layer_input=True,
                residual_only=True,
                early_trigger=early_trigger,
                hop_b_start_trigger=hop_b_start_trigger,
                row_tile_layer_input=wide_row_tile_layer_input,
                publish_trigger_max_rows=publish_trigger_max_rows,
                mbarrier_exchange=mbarrier_exchange,
                final_collapse=final_collapse,
            )
            if hop_b and hop_b_wide_rows > hop_a_rows
            else None
        )

    def hop_for(self, rows: int):
        """The mailbox hop B of ``rows`` rows runs on."""
        if self.hop_wide is not None and rows > self.hop.capacity:
            return self.hop_wide
        return self.hop

    def mailbox_bytes(self) -> int:
        return sum(
            h._mailbox.numel() * h._mailbox.element_size()
            for h in (self.hop, self.hop_wide)
            if h is not None
        )

    def precompile_publish(self, wo_b, rows: list[int]) -> list[int]:
        """Compile the publishing GEMM for each row count it takes; returns them."""
        taken = [m for m in rows if self.publisher.supports(wo_b, m)]
        for m in taken:
            self.publisher.precompile(wo_b, m)
        return taken

    def add_moe(self, layer_id: int, *, w13, w13_scale, w2, w2_scale, clamp, top_k,
                max_rows: int, gate_topk: bool) -> None:  # fmt: skip
        self.moe[layer_id] = RoutedExpertGemm(
            w13,
            w13_scale,
            w2,
            w2_scale,
            clamp,
            top_k=top_k,
            max_rows=max_rows,
            gate_topk=gate_topk,
        )

    def publish_wo_b(self, wo_b, o) -> Optional[torch.Tensor]:
        """wo_b's partial rows for ``o`` (BF16, or an MXFP8 pair), also stored into
        the mailbox; None when the publishing GEMM cannot take them."""
        prepared = self.publisher.prepare(wo_b, o)
        if prepared is None:
            return None
        self._hop_calls += 1
        if self._inject_after is not None and self._hop_calls > self._inject_after:
            # injected fault: leave this rank's mailbox slot empty
            rows = prepared[0].shape[0]
            out = torch.empty(rows, wo_b.weight.shape[0], dtype=torch.bfloat16,
                              device=self.device)  # fmt: skip
            self.hop.mark_published(out)
            logger.warning(
                "NVIDIA megakernel: injected hop_a fault on rank %d", self.rank
            )
            return out
        return self.publisher(wo_b, prepared)

    def check_consumed(self) -> None:
        """Raise unless the last published rows were reduced: a publish whose
        consumer never ran leaves the mailbox generation it wrote dirty."""
        if self.hop._published is not None:
            self.hop._published = None
            raise RuntimeError(
                "NVIDIA megakernel: published wo_b rows were not reduced by the hop; "
                "the attention post did not take the fused reduction"
            )

    def hop_a(self, partial, residual, post, comb, pre, norm_weight, norm_eps):
        """The Lamport consumer of the published rows: (updated streams, normalized
        next input)."""
        return self.hop(partial, residual, post, comb, pre, norm_weight, norm_eps)
