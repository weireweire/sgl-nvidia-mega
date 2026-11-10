# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The attention step over an ``MQALayer``'s weights, SGLang's attention backend
and its caches: projections, KV / compressor / indexer stores, sparse decode,
inverse RoPE + ``wo_a``, ``wo_b`` (``step.py``).

``step`` returns the attention output in the form ``boundary.post(layer, "attn")``
takes: SGLang's ``AttnOutput``, ``wo_b``'s all-reduce deferred to the post. A layer
the step does not cover runs SGLang's own attention, and build logs why.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import torch

from sgl_nvidia_mega.deepseek_v41.attention import chain, indexer, sched_ahead
from sgl_nvidia_mega.deepseek_v41.attention.step import AttentionStep, disable_reason

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/attention/dsv4/dsv41_sparse.py": ("token_req_indices",),
    "srt/layers/attention/deepseek_v4_backend.py": (
        "low_ratio_req_indices",
        "low_ratio_pos_i64",
    ),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "ForwardView",
        "metadata",
    ),
    "srt/models/deepseek_v4.py": (
        "layers",
        "start_layer",
        "end_layer",
        "self_attn",
        "compress_ratio",
        "maybe_use_decode_attn_tp",
        "_run_boundary_attention",
        "layer_id",
    ),
}


def build(model, ctx, group) -> dict[int, AttentionStep]:
    """Each covered layer's ``AttentionStep`` by layer id; the result is
    ``st.bound["attention"]``."""
    from sgl_nvidia_mega.deepseek_v41.runtime import SLOTS

    m = model.model
    chain_on = "attn" in SLOTS
    qkv_a_ll = chain_on and "attn_qkv_a" in SLOTS
    qkv_a_bf16 = qkv_a_ll and "attn_qkv_a_bf16" in SLOTS
    steps, skipped = {}, []
    kinds: Counter = Counter()
    # slot attn_indexer: one plugin indexer for every layer it takes
    step_indexer = (
        indexer.StepIndexer(model.config, fused="attn_indexer_fused" in SLOTS)
        if chain_on and "attn_indexer" in SLOTS
        else None
    )
    # slots attn_indexer_narrow / _fused: also the swap-AB widths' query ops
    narrow_indexer = SLOTS & {"attn_indexer_narrow", "attn_indexer_fused"}
    for i in range(m.start_layer, m.end_layer):
        attn = m.layers[i].self_attn
        reason = disable_reason(attn)
        if reason:
            skipped.append(reason)
            continue
        steps[attn.layer_id] = step_ = AttentionStep(
            attn,
            chain_ops=chain_on,
            qkv_a_ll=qkv_a_ll,
            qkv_a_bf16=qkv_a_bf16,
            index_q_mxfp8_in=chain_on and "index_q_mxfp8_in" in SLOTS,
            kv_before_wait="kv_before_wait" in SLOTS,
        )
        if chain_on:
            ops0 = step_.chain[0]
            if (
                step_indexer is not None
                and ops0.wq_b_ll
                and not indexer.reason(attn, model.config)
            ):
                ops0.indexer = step_indexer
                if narrow_indexer:
                    step_.chain[1].indexer = step_indexer
            chain.precompile(attn, ctx.widths, qkv_a_ll=qkv_a_ll)
            kinds[_chain_kind(attn, step_)] += 1
    logger.info(
        "NVIDIA megakernel: attention step in %d layers (%s)%s",
        len(steps),
        "; ".join(f"{n} x {k}" for k, n in sorted(kinds.items())) or "stock",
        f"; SGLang's in the rest ({'; '.join(skipped)})" if skipped else "",
    )
    return steps


def _chain_kind(attn, step_: AttentionStep) -> str:
    """One layer's chain decisions, as build logs them."""
    parts = [f"ratio {attn.compress_ratio}"]
    if step_.chain[0].qkv_a_ll:
        parts.append(
            "wqkv_a low-latency 1-8"
            + (" (BF16 input too)" if step_.chain[0].qkv_a_bf16 else "")
        )
    if step_.chain[0].index_q_mxfp8_in:
        parts.append("indexer wq_b on the chain's q_lora quant 1-8")
    if step_.chain[0].indexer is not None:
        parts.append(
            "indexer top-k in the plugin 1-64 (query on stream 1)"
            + (", swap-AB widths too" if step_.chain[1].indexer is not None else "")
            + (", fused consumers 1-8" if step_.chain[0].indexer.fused else "")
        )
    parts.append("wq_b low-latency 1-48" if step_.chain[0].wq_b_ll else "wq_b SGLang's")
    swapab = "swap-AB 1-24" if step_.swapab else "no swap-AB"
    if step_.swapab and step_.kv_before_wait and step_.low_ratio:
        swapab += " (KV before the PDL wait)"
    parts.append(swapab)
    parts.append("fused wo_a 1-48" if attn.use_fused_wo_a else "wo_a SGLang's")
    return ", ".join(parts)


def head(st) -> None:
    """The step head's attention work, before the embedding: the FlashMLA schedules
    forked onto side streams (slot "sched_ahead") and the low-ratio layers' (request,
    position) rows built once (slot "req_pos_once")."""
    from sgl_nvidia_mega.deepseek_v41.runtime import SLOTS

    view, rows = st.forward_view(), int(st.positions.shape[0])
    if "req_pos_once" in SLOTS:
        _req_pos_once(view.metadata, st.fb, rows)
    if "sched_ahead" in SLOTS:
        st.flashmla_sched = sched_ahead.start(
            view, st.fb, rows, lambda ratio: _runs_flashmla(st, view, rows, ratio)
        )


def _runs_flashmla(st, view, rows: int, ratio: int) -> bool:
    """Whether ratio ``ratio``'s layers run FlashMLA at ``rows`` rows: neither the
    chain's swap-AB nor SGLang's takes the ratio's first layer."""
    m = st.model
    attn = next(
        (
            m.layers[i].self_attn
            for i in range(m.start_layer, m.end_layer)
            if m.layers[i].self_attn.compress_ratio == ratio
        ),
        None,
    )
    if attn is None:
        return False
    step_ = st.bound["attention"].get(attn.layer_id)
    if step_ is not None and step_.takes_swapab(rows, st.fb, view):
        return False
    return not sched_ahead.sglang_takes_swapab(attn, view, rows)


def _req_pos_once(meta, fb, rows: int) -> None:
    """SGLang's low-ratio layers take these rows when set, instead of expanding them
    in every layer's index-query and sources streams (the decode step's own hoist)."""
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import token_req_indices

    core = getattr(meta, "core_attn_metadata", None)
    if not fb.forward_mode.is_target_verify() or not getattr(core, "low_ratios", ()):
        return
    meta.low_ratio_req_indices = token_req_indices(fb, num_tokens=rows)
    meta.low_ratio_pos_i64 = fb.positions.to(torch.int64)


def step(layer, x, st) -> Any:
    """``layer``'s attention on the normed input ``x`` [rows, dim]; reads
    ``st.x_quant``, ``st.positions``, ``st.fb``."""
    if st.flashmla_sched:
        # the first attention layer: every schedule built ahead joins here
        sched_ahead.join(st.flashmla_sched)
        st.flashmla_sched = []
    attention = st.bound["attention"].get(layer.self_attn.layer_id)
    if attention is None:
        with layer.self_attn.maybe_use_decode_attn_tp(st.fb):
            return layer._run_boundary_attention(x, st.positions, st.fb, st.x_quant)
    # hop A: wo_b publishes into the mailbox the attention post collects
    publish = st.bound["boundary"].hop_a.publish_wo_b if st.slots.hop_a else None
    return attention.run(
        x, st.positions, st.fb, st.x_quant, publish=publish, view=st.forward_view()
    )
