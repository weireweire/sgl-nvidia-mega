# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The draft's slots as built: each requested draft slot, built (where) or declined
(why), for the startup summary. A requested slot whose build serves nothing would
otherwise leave the draft on SGLang's path without a word."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py": ("can_fuse_all_reduce",),
    "srt/models/deepseek_v4.py": ("self_attn", "layer_id", "hc_cfg"),
}

BUILT, DECLINED, UNKNOWN = "built", "declined", "unknown"
NOT_NEEDED = "not needed"  # another path serves what the slot would


def states(built, slots) -> dict[str, tuple[str, str]]:
    """slot -> (BUILT or DECLINED, where or why) for each requested draft slot."""
    from sgl_nvidia_mega.deepseek_v41.runtime import DRAFT_SLOTS

    requested = sorted(DRAFT_SLOTS & set(slots))
    out = {name: _proposal(built, name) for name in requested if name in _PROPOSAL}
    if "draft" not in slots:
        return out
    stages = list(built._view.model.layers)
    for name in requested:
        if name in _PROPOSAL:
            continue
        try:
            out[name] = _forward(built, stages, slots, name)
        except Exception as e:  # a report, never a reason to stop the build
            out[name] = UNKNOWN, f"{type(e).__name__}: {e}"
    return out


def summary(table: dict[str, tuple[str, str]]) -> str:
    if not table:
        return "none requested"
    return "; ".join(f"{name} {state} ({why})" for name, (state, why) in table.items())


def warn_declined(table: dict[str, tuple[str, str]]) -> None:
    for name, (state, why) in table.items():
        if state in (DECLINED, UNKNOWN):
            logger.warning(
                "NVIDIA megakernel: draft slot %s requested, %s: %s", name, state, why
            )


_PROPOSAL = ("markov", "draft_lm_head")


def _proposal(built, name):
    if name == "markov":
        if built.draft_sampler is None:
            return DECLINED, "SGLang's proposal"
        return BUILT, "Markov chain in one kernel, built with the proposal's buffers"
    if not built.low_m:
        return DECLINED, "LM head SGLang's"
    return BUILT, "low-M LM head, built with the proposal's buffers"


def _forward(built, stages, slots, name):
    if name == "draft":
        return BUILT, f"widths {list(built.widths)}"
    if name in ("draft_post", "draft_hop", "draft_seam", "draft_prenorm"):
        return _boundary(built, stages, name)
    if name in ("draft_moe", "draft_gate"):
        return _moe(built, name)
    if name == "draft_tail":
        return BUILT, "BF16 streams into hc_combine"
    if name == "draft_embed":
        return BUILT, "the target's whole table, bound at the first forward"
    return _attention(built, stages, slots, name)


def _boundary(built, stages, name):
    bound = built._bound["boundary"]
    hop = bound.draft_hop
    if name == "draft_prenorm":
        return (BUILT, "DeepGEMM TF32") if bound.prenorm else (DECLINED, "not served")
    if name == "draft_hop":
        if hop is None:
            from sgl_nvidia_mega.deepseek_v41.boundary import draft_hop

            if draft_hop.sglang_fuses():
                return NOT_NEEDED, built._hop_reason
            return DECLINED, built._hop_reason
        return BUILT, f"Lamport all-reduce + mHC at 1-{hop.capacity} rows"
    if name == "draft_seam":
        if not bound.cross_layer:
            return DECLINED, "needs draft_post"
        if not any(bound._attn_mxfp8.values()):
            return (
                BUILT,
                "the next input stays streams only: the draft attention takes no MXFP8",
            )
        return BUILT, "the MoE post quantizes the next stage's input"
    # draft_post: who serves the posts at the narrowest width
    if hop is not None:
        return BUILT, f"the plugin's draft hop at 1-{hop.capacity} rows"
    rows = built.widths[0]
    served = [s.layer_id for s in stages if _sglang_fuses(bound, s, rows)]
    if len(served) == len(stages):
        return BUILT, f"SGLang's fused all-reduce + mHC posts from {rows} rows"
    return DECLINED, (
        f"stages {[s.layer_id for s in stages if s.layer_id not in served]} at "
        f"{rows} rows run SGLang's plain all-reduce + unfused mHC post "
        f"(draft_hop: {built._hop_reason})"
    )


def _sglang_fuses(bound, stage, rows) -> bool:
    from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
        can_fuse_all_reduce,
    )

    from sgl_nvidia_mega.deepseek_v41.boundary.boundaries import _fusable

    return all(
        _fusable(stage, kind, bound._fuses(stage)) for kind in ("attn", "ffn")
    ) and can_fuse_all_reduce(rows, stage.hc_cfg.hidden)


def _moe(built, name):
    moe = built._bound["moe"]
    slots = getattr(moe, "slots", {}) or {}
    if not slots:
        return DECLINED, getattr(moe, "reason", "") or "no stage built"
    if name == "draft_moe":
        return BUILT, f"routed pair in stages {sorted(slots)}"
    gated = sorted(i for i, s in slots.items() if s.gate)
    if not gated:
        return DECLINED, "the fast top-k applies to no stage"
    return BUILT, f"fast top-k in stages {gated}"


# slot -> the Kernels field it sets
_ATTN_FIELDS = {
    "draft_qfront": "qfront",
    "draft_wo_a": "wo_a",
    "draft_swapab": "swapab",
    "draft_qkv_a": "qkv_a",
    "draft_wo_b_ll": "wo_b_ll",
}


def _attention(built, stages, slots, name):
    ported = built._attn
    if not ported:
        if built._binding is None:
            return DECLINED, "SGLang offers no binding for the draft forward"
        return DECLINED, "no stage ported (unified KV, or SGLang's slow draft kernels)"
    if name == "draft_attn":
        return BUILT, f"stages {sorted(ported)}"
    if name == "draft_qkv_a_ll":
        on = sorted(i for i, k in ported.items() if k.qkv_a is not None and k.qkv_a.ll)
    else:
        field = _ATTN_FIELDS[name]
        on = sorted(i for i, k in ported.items() if getattr(k, field))
    if on:
        return BUILT, f"stages {on}"
    return DECLINED, _attention_why(stages, slots, name)


def _attention_why(stages, slots, name) -> str:
    attn = stages[0].self_attn
    if name == "draft_swapab":
        return f"swap-AB takes no {attn.n_local_heads} local heads" + (
            "" if "swapab_h8" in slots else " (slot swapab_h8 off)"
        )
    if name == "draft_wo_b_ll" and "draft_wo_a" in slots:
        return "wo_b is not an MXFP8 linear the low-latency GEMM takes, or wo_a unfused"
    if name in ("draft_wo_a", "draft_wo_b_ll"):
        return f"the fused wo_a takes no {attn.n_local_groups} local groups" + (
            "" if "wo_a_g1" in slots else " (slot wo_a_g1 off)"
        )
    if name in ("draft_qkv_a", "draft_qkv_a_ll"):
        from sgl_nvidia_mega.deepseek_v41.draft_attention import qkv_a_fusable

        return (
            qkv_a_fusable(attn.wq_a, attn.wkv) or "the low-latency GEMM is unsupported"
        )
    return "the query front applies to no stage (head norm, or wq_b off the LL GEMM)"
