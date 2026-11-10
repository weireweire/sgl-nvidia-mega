# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot draft_hop: the DSpark draft stages' posts on the plugin's Lamport all-reduce +
mHC where SGLang has no fused post (attention TP8, MoE TP4 x EP2).

At TP2 / TP4 slot draft_post runs SGLang's fused all-reduce + mHC kernels on the
stages. SGLang builds those for TP2 / TP4 only (``deepseek_v4_mhc._FUSED_MHC_TP_SIZES``),
so at TP8 its draft posts run a plain all-reduce, the MoE finalize and the unfused mHC
post. This hop takes both posts there instead: the attention post publishes the wo_b
rows and collects them (hop A without the publishing GEMM), the MoE post finalizes into
the same mailbox (hop B). The draft's mailbox is its own, beside the target's.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/deepseek_v4_mhc.py": ("fused_mhc_tp_ok",),
    # the module alias SGLang's DSV4 model imports too
    "srt/models/deepseek_v4.py": ("mhc", "hc_cfg", "self_attn", "layer_id"),
    "srt/runtime_context.py": ("get_parallel", '"tp_size"', '"moe_ep_size"'),
}

# The layout whose draft posts SGLang leaves unfused and this hop serves
LAYOUT = (8, 2)


class DraftHop:
    """One rank's draft mailbox (``AllReduceMHC``): the attention post publishes and
    collects in one call, the MoE post finalizes; both at 1-``capacity`` rows."""

    def __init__(self, hop):
        self._hop = hop
        self.capacity = hop.capacity

    def collect(self, partial, residual, post_mix, comb, pre, norm_weight, eps):
        """(updated streams, normed FFN input) of the TP-partial ``partial``."""
        return self._hop(partial, residual, post_mix, comb, pre, norm_weight, eps)

    def finalize(self, *args, rows: int, mxfp8_out=None):
        """Hop B's call (``hop_a.HopA.finalize``) on the draft's mailbox."""
        del rows  # one mailbox at every width
        return self._hop.finalize(*args, mxfp8_out=mxfp8_out)


def sglang_fuses() -> bool:
    """SGLang builds its fused all-reduce + mHC posts at this TP size."""
    from sglang.srt.models import deepseek_v4_mhc as mhc
    from sglang.srt.runtime_context import get_parallel

    return mhc.fused_mhc_tp_ok(get_parallel().tp_size)


def layout_reason() -> str:
    """Why the draft's posts are not this hop's at this layout, or ""."""
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    tp = parallel.tp_size
    if sglang_fuses():
        return f"SGLang's fused all-reduce + mHC posts serve at TP{tp}"
    if (tp, parallel.moe_ep_size) != LAYOUT:
        return f"no fused post at (tp, ep) = ({tp}, {parallel.moe_ep_size})"
    return ""


def build(
    stages, *, deadline_s: float, max_rows: int
) -> tuple[Optional[DraftHop], str]:
    """(the hop, "") or (None, why not). Collective over the TP group at (8, 2): a
    failure on any rank raises on every rank."""
    from sglang.srt.runtime_context import get_parallel

    from sgl_nvidia_mega.deepseek_v41.boundary.boundaries import _fusable

    if why := layout_reason():
        return None, why
    for stage in stages:
        for kind in ("attn", "ffn"):
            if not _fusable(stage, kind, True):
                return None, (
                    f"stage {stage.layer_id}: SGLang's {kind} fuse gate (less its TP "
                    "size) does not hold"
                )
    top_k = {stage.mlp.experts.top_k for stage in stages}
    if len(top_k) != 1:
        return None, f"the stages route different top-k {sorted(top_k)}"
    group = get_parallel().tp_group
    error, hop = "", None
    try:
        hop = _mailbox(stages[0], top_k.pop(), group, deadline_s, max_rows)
    except Exception as e:  # gathered, then raised on every rank
        error = f"{type(e).__name__}: {e}"
    errors = [None] * group.world_size
    torch.distributed.all_gather_object(errors, error, group=group.cpu_group)
    failed = [f"rank {r}: {e}" for r, e in enumerate(errors) if e]
    if failed:
        raise RuntimeError(
            "NVIDIA megakernel: the draft hop build failed: " + "; ".join(failed)
        )
    return DraftHop(hop), ""


def _mailbox(stage, top_k: int, group, deadline_s: float, max_rows: int):
    from sgl_nvidia_mega.deepseek_v41.sm103 import all_reduce_mhc

    # before the consumer compiles: the deadline is a constant of its kernel
    all_reduce_mhc.set_wait_deadline(deadline_s)
    cfg = stage.hc_cfg
    return all_reduce_mhc.AllReduceMHC(
        hidden_size=cfg.hidden,
        hc_mult=cfg.mult,
        max_num_tokens=max_rows,
        top_k=top_k,
        device=stage.self_attn.wo_b.weight.device,
        group=group.device_group,
        quantize_layer_input=True,
        residual_only=True,
    )
