# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The mHC between sublayers (owner: boundary lane).

``pre`` turns the residual streams into a sublayer's input (mHC pre: combine +
norm); ``post`` reduces a sublayer's output into the streams (all-reduce + mHC
post) and, after attention, returns the MoE input (it folds the FFN's pre, as
SGLang does). ``start`` / ``final`` open and close the streams around the layers;
``materialize`` / ``set_residual`` serve the readers between layers (Engram, the
draft's capture).

Today every function runs SGLang's stock decode boundary (``stock.py``): SGLang's
own mHC functions in its stage boundary's order, without its stage objects. The
MoE post runs right after the MoE, so the streams are always materialized between
layers.
"""

from __future__ import annotations

import msgspec
import torch

from sgl_nvidia_mega.deepseek_v41.boundary.boundaries import Boundaries, BoundaryState

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/deepseek_v4.py": ("layers", "start_layer", "end_layer"),
    "srt/models/deepseek_v4_mhc.py": ("materialized", "residual", "pre"),
    "srt/runtime_context.py": (
        "get_forward",
        "scoped",
        '"mlp_reduce_scatter"',
        '"defer_moe_finalize"',
    ),
}


class MoeOperands(msgspec.Struct, frozen=True):
    """What SGLang's MoE transport hands the MoE in this configuration (no DP
    gather, no attention-TP scatter, no context parallelism)."""

    input_ids: object
    input_ids_global: object
    do_shared_local: bool
    return_moe_output: bool  # the MoE leaves its finalize to the post


def build(model, ctx, group) -> Boundaries:
    """Bind weights, allocate buffers, compile; the result is ``st.bound["boundary"]``.
    Its ``hop_a`` / ``hop_a_reason`` / ``hop_a.rows`` say where hop A serves."""
    from sgl_nvidia_mega.deepseek_v41 import runtime
    from sgl_nvidia_mega.deepseek_v41.boundary import hop_a, prenorm

    m = model.model
    hop, reason = None, "stock (slot off)"
    if "hop_a" in runtime.SLOTS:
        hop, reason = hop_a.build(
            model,
            ctx,
            group,
            hop_b="hop_b" in runtime.SLOTS,
            early_trigger="early_trigger" in runtime.SLOTS,
            hop_b_start_trigger="hop_b_start_trigger" in runtime.SLOTS,
            batch2="hop_a_batch2" in runtime.SLOTS,
            hop_b_wide="hop_b_wide" in runtime.SLOTS,
            hop_b_cap16="hop_b_cap16" in runtime.SLOTS,
            hop_b_quad_trigger="hop_b_quad_trigger" in runtime.SLOTS,
            mbarrier_exchange="hop_mbar" in runtime.SLOTS,
            final_collapse="hop_final_collapse" in runtime.SLOTS,
        )
    two_shot, two_shot_reason = None, "stock (slot off)"
    if "two_shot" in runtime.SLOTS:
        two_shot, two_shot_reason = _build_two_shot(model, ctx, group)
    # a load-time fact of the platform and config, the same on every rank
    use_prenorm = "hc_prenorm" in runtime.SLOTS and not prenorm.reason()
    return Boundaries(
        [m.layers[i] for i in range(m.start_layer, m.end_layer)],
        hop_a=hop,
        hop_a_reason=reason,
        two_shot=two_shot,
        two_shot_reason=two_shot_reason,
        hop_b="hop_b" in runtime.SLOTS,
        prenorm=use_prenorm,
        final_collapse="hop_final_collapse" in runtime.SLOTS,
    )


def _build_two_shot(model, ctx, group):
    """(the two-shot hops, "") or (None, why not); a failure on any rank raises
    on every rank."""
    from sgl_nvidia_mega.deepseek_v41.sm103.two_shot_mhc import MAX_ROWS, TwoShotMHC

    cfg, m = model.config, model.model
    error, hop = "", None
    try:
        hop = TwoShotMHC(
            hidden_size=cfg.hidden_size,
            hc_mult=cfg.hc_mult,
            max_num_tokens=MAX_ROWS,
            top_k=cfg.num_experts_per_tok,
            device=m.layers[m.start_layer].self_attn.wo_b.weight.device,
            group=group.device_group,
            deadline_s=ctx.deadline_s,
        )
    except Exception as e:  # gathered, then raised on every rank
        error = f"{type(e).__name__}: {e}"
    errors = [None] * group.world_size
    torch.distributed.all_gather_object(errors, error, group=group.cpu_group)
    failed = [f"rank {r}: {e}" for r, e in enumerate(errors) if e]
    if failed:
        raise RuntimeError(
            "NVIDIA megakernel: the two-shot build failed: " + "; ".join(failed)
        )
    return hop, ""


def _state(st) -> BoundaryState:
    return st.scratch["boundary.state"]


def start(st, x) -> None:
    """Open the residual streams from the broadcast embedding ``x`` [rows, hc, dim]."""
    bs = st.bound["boundary"].start(x)
    st.scratch["boundary.state"] = bs
    st.state = bs.hc


def materialize(layer, st):
    """The streams as an ``HcState`` before ``layer`` (for Engram and the capture)."""
    st.state = _state(st).hc
    return st.state


def set_residual(st, residual) -> None:
    """Replace the materialized streams' residual (Engram's output)."""
    bs = _state(st)
    st.bound["boundary"].with_residual(bs, residual)
    st.state = bs.hc


def pre(layer, sublayer: str, st, aux=None):
    """The input of ``layer``'s ``sublayer`` ("attn"); sets ``st.x_quant``. With
    ``aux``, appends the streams' mean (a DSpark capture layer)."""
    assert sublayer == "attn"
    x, st.x_quant = st.bound["boundary"].pre(layer, "attn", _state(st), st.fb, aux)
    return x


def post(layer, sublayer: str, out, st):
    """Reduce ``out`` of ``layer``'s ``sublayer`` into the streams. After "attn"
    returns the MoE input (and sets ``st.transport``); after "ffn" returns None."""
    boundaries, bs = st.bound["boundary"], _state(st)
    boundaries.post(layer, sublayer, out, bs)
    if boundaries.hop_a is not None and sublayer == "attn":
        boundaries.hop_a.check_consumed()
    if sublayer == "ffn":
        st.state = bs.hc
        return None
    x, st.x_quant = boundaries.pre(layer, "ffn", bs, st.fb)
    st.transport = MoeOperands(
        st.input_ids,
        st.input_ids_global,
        do_shared_local=False,
        return_moe_output=boundaries.defers_sum(layer, "ffn", x.shape[0]),
    )
    # the scope SGLang's MoE transport opens around the MoE
    from sglang.srt.runtime_context import get_forward

    st.exit.enter_context(
        get_forward().scoped(mlp_reduce_scatter=False, defer_moe_finalize=False)
    )
    return x


def final(st):
    """Close the streams after the last layer: (residual [rows, hc, dim], pre)."""
    m = st.model
    bs = _state(st)
    st.bound["boundary"].join_aux(bs)
    state = bs.hc.materialized(m.layers[m.end_layer - 1].hc_cfg)
    return state.residual, state.pre


def final_collapse(st):
    """The head's mHC collapse the last layer's hop B wrote (slot hop_final_collapse),
    or None: the head then combines the final streams itself."""
    return _state(st).final_collapse


def head(st, residual, collapsed):
    """``logits.head`` on the collapse hop B wrote: (hidden, pre_hc_head), SGLang's
    final RMSNorm on the same BF16 rows its mhc_combine would give."""
    return st.model.norm(collapsed), residual.flatten(1)


def clear(st) -> None:
    """Drop the step's stream state (also on an error)."""
    st.scratch.pop("boundary.state", None)
