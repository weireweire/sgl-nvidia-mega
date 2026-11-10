# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hop A: wo_b publishes its rows into a Lamport mailbox on every rank, and the
attention post collects them with the all-reduce + mHC post + FFN norm fused.
Hop B reuses the mailbox for the MoE post (the Lamport MoE finalize)."""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sgl_nvidia_mega.deepseek_v41 import plan as slot_plan

logger = logging.getLogger(__name__)


class HopA:
    """One rank's mailbox, publishing wo_b GEMM and Lamport consumer."""

    def __init__(self, runner, rows: tuple[int, ...], hop_b_rows: tuple[int, ...] = ()):
        self._runner = runner
        self.rows = rows  # the widths the publishing GEMM was compiled for
        self.hop_b_rows = hop_b_rows  # the widths the MoE post runs on the mailbox

    def publish_wo_b(self, wo_b, o) -> Optional[torch.Tensor]:
        """wo_b's TP-partial rows for ``o`` (BF16 [rows, heads * dim], or its MXFP8
        pair), also stored into the mailbox; None when the GEMM cannot take them."""
        return self._runner.publish_wo_b(wo_b, o)

    def published(self, partial: torch.Tensor) -> bool:
        return self._runner.hop._published is partial

    def collect(self, partial, residual, post_mix, comb, pre, norm_weight, eps):
        """(updated streams, normed FFN input) of the published ``partial``."""
        return self._runner.hop_a(
            partial, residual, post_mix, comb, pre, norm_weight, eps
        )

    def finalize(self, *args, rows: int, mxfp8_out=None, collapse: bool = False):
        """Hop B: ``AllReduceMHC.finalize`` of a deferred MoE output of ``rows`` rows, on
        the hop A mailbox or, past its capacity, slot hop_b_wide's."""
        return self._runner.hop_for(rows).finalize(
            *args, mxfp8_out=mxfp8_out, collapse=collapse
        )

    def check_consumed(self) -> None:
        """Raise on the host if a publish was not collected: it would leave the
        mailbox generation it wrote dirty."""
        self._runner.check_consumed()


def build(
    model,
    ctx,
    group,
    *,
    hop_b: bool = False,
    early_trigger: bool = False,
    hop_b_start_trigger: bool = False,
    batch2: bool = False,
    hop_b_wide: bool = False,
    hop_b_cap16: bool = False,
    hop_b_quad_trigger: bool = False,
    mbarrier_exchange: bool = False,
    final_collapse: bool = False,
) -> tuple[Optional[HopA], str]:
    """(the hop, "") or (None, why not). Collective over the TP group: a failure on
    any rank raises on every rank; an unsupported wo_b leaves the hop off. ``hop_b``
    also compiles the MoE finalize's consumers; ``early_trigger`` launches the
    consumer's dependents right after its poll, ``hop_b_start_trigger`` the MoE
    finalize's at its start; ``batch2`` (slot hop_a_batch2) adds 9-16 rows; ``hop_b_wide``
    runs hop B at every width up to plan.HOP_B_MAX_ROWS, not only hop A's;
    ``mbarrier_exchange`` (slot hop_mbar) and ``final_collapse`` (slot
    hop_final_collapse, with ``hop_b``) are AllReduceMHC's, on every mailbox."""
    from sgl_nvidia_mega.deepseek_v41.sm103.runner import Runner

    m = model.model
    cfg = model.config
    wo_b = m.layers[m.start_layer].self_attn.wo_b
    lay = slot_plan.hop_layout(
        ctx.widths,
        hop_b=hop_b,
        batch2=batch2,
        hop_b_wide=hop_b_wide,
        cap16=hop_b_cap16,
    )
    rows, hop_b_rows = lay.hop_a_rows, lay.hop_b_rows
    if not rows:
        return None, f"no width up to {lay.capacity} rows"
    runner = Runner(device=wo_b.weight.device, rank=group.rank_in_group)
    error, taken = "", ()
    try:
        runner.build_hop(
            group=group.device_group,
            hidden=cfg.hidden_size,
            hc_mult=cfg.hc_mult,
            top_k=cfg.num_experts_per_tok,
            hop_a_rows=lay.capacity,
            hop_b_wide_rows=lay.wide_capacity,
            deadline_s=ctx.deadline_s,
            hop_b=hop_b,
            early_trigger=early_trigger,
            hop_b_start_trigger=hop_b_start_trigger,
            row_tile_layer_input=lay.row_tiles,
            wide_row_tile_layer_input=lay.wide_row_tiles,
            # the batch-1 graph only (measured faster there); wider rows keep the end trigger
            publish_trigger_max_rows=(
                slot_plan.HOP_A_BATCH1_ROWS if hop_b_quad_trigger else 0
            ),
            mbarrier_exchange=mbarrier_exchange,
            final_collapse=hop_b and final_collapse,
        )
        taken = tuple(runner.precompile_publish(wo_b, list(rows)))
        if hop_b and hop_b_quad_trigger:
            logger.info(
                "NVIDIA megakernel: hop B finalize-publish triggers its consumer "
                "after its wait at 1-%d rows",
                slot_plan.HOP_A_BATCH1_ROWS,
            )
    except Exception as e:  # gathered, then raised on every rank
        error = f"{type(e).__name__}: {e}"
    gathered = [None] * group.world_size
    torch.distributed.all_gather_object(gathered, (error, taken), group=group.cpu_group)
    failed = [f"rank {r}: {e}" for r, (e, _) in enumerate(gathered) if e]
    if failed:
        raise RuntimeError(
            "NVIDIA megakernel: the hop-A build failed: " + "; ".join(failed)
        )
    if any(t != taken for _, t in gathered):
        raise RuntimeError(
            f"NVIDIA megakernel: the ranks disagree on hop A: {gathered}"
        )
    if not taken:
        return None, "wo_b is not an MXFP8 linear the low-latency GEMM takes"
    return HopA(runner, taken, hop_b_rows), ""
