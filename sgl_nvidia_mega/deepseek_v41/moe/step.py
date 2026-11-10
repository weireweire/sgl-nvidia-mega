# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/deepseek_v2.py (DeepseekV2MoE.forward and its dual-stream path).
"""The MoE step of a covered decode step, over SGLang's loaded ``DeepseekV2MoE`` block.

The stock step issues what ``DeepseekV2MoE.forward`` issues for the covered
configuration (CUDA, TP4, or TP8 with MoE TP4 x EP2; no attention / context
parallelism, no EPLB, no fused shared experts), calling the block's own gate, top-k, routed experts and shared
expert: at CUDA graph capture the dual-stream order (routed on the main stream, shared
expert on the alternate stream), otherwise the eager order. Its output is what the
decoder layer hands the FFN post: the unmerged pieces (``MoEOutput``), or the deferred
finalize when the forward context defers it.
"""

from __future__ import annotations

import logging

import torch

from sgl_nvidia_mega.deepseek_v41 import sglang_ops

logger = logging.getLogger(__name__)

# The SGLang names this module reads, by file under the sglang package (``f(p)``: a
# parameter); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/models/deepseek_v2.py": (
        "MoEOutput",
        "alt_stream",
        "routed_quant_stream",
        "num_fused_shared_experts",
        "gate",
        "topk",
        "experts",
        "shared_experts",
        "is_hash",
        "_shared_expert_tp1",
        "_deferred_finalize_max_tokens",
        "routed_scaling_factor",
        "_maybe_quant_moe_input_once",
        "_should_quant_routed_input_mxfp8",
        "_forward_shared_experts",
        "_forward_shared_experts(pre_quant_input)",
    ),
    "srt/models/deepseek_v4.py": ("mlp",),
    "srt/models/deepseek_common/moe_transport.py": (
        "input_ids_global",
        "do_shared_local",
    ),
    "srt/models/deepseek_v4_boundary.py": ("return_moe_output",),
    "srt/layers/moe/topk.py": ("BypassedTopKOutput", "TopKOutputFormat", "format"),
    "srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py": (
        "Mxfp4FlashinferTrtllmMoEMethod",
        "Mxfp8RoutedInputPreQuant",
        "routed_hidden_size",
        "quantize_routed_input",
    ),
    "srt/layers/moe/fused_moe_triton/layer.py": (
        "supports_deferred_finalize",
        "forward_deferred_finalize",
        "forward_deferred_finalize(pre_quant_input)",
        "forward_impl",
        "moe_runner_config",
    ),
    "srt/layers/moe/utils.py": ("sum_post_experts_output",),
    "srt/layers/layer_boundary/fusions/moe_finalize.py|srt/layers/layer_boundary/fusions/cutedsl.py": (
        "MoeDeferredFinalize",
        "from_flashinfer",
    ),
    "srt/multimodal/dsv41/vl_routing.py": ("vision_topk",),
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
    "srt/model_executor/runner_backend_utils/breakable_cuda_graph/context.py": (
        "is_in_breakable_cuda_graph",
    ),
    "srt/runtime_context.py": (
        "get_forward",
        '"flashinfer_trtllm_bypass"',
        '"defer_moe_finalize"',
    ),
}


def dual_stream(mlp: torch.nn.Module, rows: int) -> bool:
    """SGLang's branch for a decode MoE: the dual-stream order at CUDA graph capture."""
    from sglang.srt.model_executor.runner import get_is_capture_mode
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
        is_in_breakable_cuda_graph,
    )

    return (
        mlp.alt_stream is not None
        and mlp.num_fused_shared_experts == 0
        and rows > 0
        and get_is_capture_mode()
        and not is_in_breakable_cuda_graph()
    )


def step(layer: torch.nn.Module, m_in: torch.Tensor, st) -> object:
    """The FFN of ``layer`` on its input rows ``m_in``, with the operands of the
    layer's MoE transport (``st.transport``, set by ``boundary.post(layer, "attn")``):
    its token ids, whether the shared expert ran on local rows, and whether the sum
    is left to the FFN post (``return_moe_output``)."""
    mlp, t = layer.mlp, st.transport
    num_token_non_padded = st.fb.moe_num_token_non_padded()
    use_vision_topk = mlp.gate.e_score_correction_bias_vl is not None
    bound = st.bound.get("moe") if st.slots.routed else None
    slot = bound.slots.get(layer.layer_id) if bound is not None else None
    # the routed pair runs in the dual-stream order in every mode
    if slot is not None or dual_stream(mlp, m_in.shape[0]):
        return _dual_stream(
            mlp,
            m_in,
            t.input_ids_global,
            num_token_non_padded,
            use_vision_topk,
            bool(t.return_moe_output),
            slot=slot,
            fast_gate=slot is not None and slot.gate and st.slots.fast_gate,
        )
    return _eager(
        mlp,
        m_in,
        t.input_ids_global,
        num_token_non_padded,
        use_vision_topk,
        bool(t.do_shared_local),
    )


# engagement, from the dispatch itself: the widths the fast top-k served (logging only)
_FAST_GATE_SERVED: set = set()


def _log_fast_gate(rows: int, fused: bool = False) -> None:
    if (rows, fused) not in _FAST_GATE_SERVED:
        _FAST_GATE_SERVED.add((rows, fused))
        logger.info(
            "NVIDIA megakernel: fast gate top-k serves %d rows%s",
            rows,
            " with the route record in one launch (gate_route)" if fused else "",
        )


# engagement of slot shared_ll, as above
_SHARED_LL_SERVED: set = set()


def _log_shared_ll(rows: int) -> None:
    if rows not in _SHARED_LL_SERVED:
        _SHARED_LL_SERVED.add(rows)
        logger.info(
            "NVIDIA megakernel: shared expert gate_up on the low-latency GEMM "
            "serves %d rows",
            rows,
        )


def _route(mlp, x, input_ids_global, num_token_non_padded, use_vision_topk):
    """The router logits and top-k, as SGLang computes them on CUDA."""
    from sglang.srt.multimodal.dsv41.vl_routing import vision_topk

    router_logits = mlp.gate(x, None)
    if use_vision_topk:
        return vision_topk(
            mlp,
            router_logits,
            input_ids_global,
            num_token_non_padded=num_token_non_padded,
        )
    kwargs = {"input_ids": input_ids_global} if getattr(mlp, "is_hash", False) else {}
    return mlp.topk(
        x,
        router_logits,
        num_token_non_padded=num_token_non_padded,
        expert_location_dispatch_info=None,
        **kwargs,
    )


def _pieces(mlp, routed, shared):
    from sglang.srt.models.deepseek_v2 import MoEOutput

    return MoEOutput(
        routed=routed,
        shared=shared,
        experts=mlp.experts,
        routed_scaling_factor=mlp.routed_scaling_factor,
        shared_is_replicated=mlp._shared_expert_tp1,
    )


def _eager(
    mlp, x, input_ids_global, num_token_non_padded, use_vision_topk, skip_shared
):
    """``DeepseekV2MoE.forward_normal`` with ``leave_output_sum``."""
    shared = None
    if x.shape[0] > 0:
        defer_shared = not mlp.experts.moe_runner_config.inplace
        pre_quant_input = None if skip_shared else mlp._maybe_quant_moe_input_once(x)
        if not defer_shared and not skip_shared:
            shared = mlp._forward_shared_experts(
                x, None, pre_quant_input=pre_quant_input
            )
        topk_output = _route(
            mlp, x, input_ids_global, num_token_non_padded, use_vision_topk
        )
    else:
        defer_shared, pre_quant_input = False, None
        topk_output = mlp.topk.empty_topk_output(x.device, layer_id=mlp.layer_id)
    if pre_quant_input is not None:
        routed = mlp.experts(x, topk_output, pre_quant_input=pre_quant_input)
    else:
        routed = mlp.experts(x, topk_output)
    if defer_shared and x.shape[0] > 0 and not skip_shared:
        shared = mlp._forward_shared_experts(x, None, pre_quant_input=pre_quant_input)
    return _pieces(mlp, routed, shared)


def _dual_stream(
    mlp,
    x,
    input_ids_global,
    num_token_non_padded,
    use_vision_topk,
    defer,
    *,
    slot=None,
    fast_gate=False,
):
    """``DeepseekV2MoE.forward_normal_dual_stream`` with ``leave_output_sum``: the
    router and routed experts on the main stream, the routed MXFP8 input on its own
    stream, the shared expert on the alternate stream after the routed branch. With
    ``slot`` the routed experts run on the GEMM pair (a deferred finalize), and with
    ``fast_gate`` its top-k takes SGLang's router logits; with ``slot.shared`` the
    shared expert runs on the low-latency GEMM, from the same fork."""
    from sglang.srt.layers.moe.topk import BypassedTopKOutput, TopKOutputFormat
    from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
        Mxfp4FlashinferTrtllmMoEMethod,
        Mxfp8RoutedInputPreQuant,
        routed_hidden_size,
    )
    from sglang.srt.runtime_context import get_forward

    fwd = get_forward()
    bypass = fwd.flashinfer_trtllm_bypass
    current = torch.cuda.current_stream()
    pre_quant_input = None if bypass else mlp._maybe_quant_moe_input_once(x)
    mlp.alt_stream.wait_stream(current)
    quant_routed = (
        not bypass
        and pre_quant_input is None
        and mlp._should_quant_routed_input_mxfp8(x)
    )
    if quant_routed:
        mlp.routed_quant_stream.wait_stream(current)
    has_shared = x.shape[0] > 0 and mlp.num_fused_shared_experts == 0
    # slot shared_ll: not with SGLang's fp8 quant-once input, which its shared expert
    # would read (never on MXFP4 experts)
    shared_ll = (
        slot.shared
        if slot is not None and has_shared and pre_quant_input is None
        else None
    )

    if fast_gate:
        _log_fast_gate(
            x.shape[0],
            slot.gemm.serves_gate_route(
                x.shape[0], input_ids_global if use_vision_topk else None
            ),
        )
        router_logits = mlp.gate(x, None)
        topk_output = slot.select(
            router_logits, input_ids_global, num_token_non_padded, use_vision_topk
        )
    elif bypass:
        topk_output = BypassedTopKOutput(
            hidden_states=x,
            router_logits=mlp.gate(x, None),
            topk_config=mlp.topk.topk_config,
        )
    else:
        topk_output = _route(
            mlp, x, input_ids_global, num_token_non_padded, use_vision_topk
        )
    routed_pre_quant = pre_quant_input
    if quant_routed:
        with torch.cuda.stream(mlp.routed_quant_stream):
            x_q, x_sf = mlp.experts.quant_method.quantize_routed_input(
                x, routed_hidden_size(mlp.experts)
            )
            ready = mlp.routed_quant_stream.record_event()
        routed_pre_quant = Mxfp8RoutedInputPreQuant(x_q, x_sf, ready)
    deferred = (
        slot is not None
        or (
            has_shared
            and not mlp._shared_expert_tp1
            and topk_output.format == TopKOutputFormat.BYPASSED
            and mlp.experts.supports_deferred_finalize
            and (
                mlp._deferred_finalize_max_tokens <= 0
                or x.shape[0] <= mlp._deferred_finalize_max_tokens
            )
        )
        or (
            defer
            and isinstance(mlp.experts.quant_method, Mxfp4FlashinferTrtllmMoEMethod)
        )
    )
    if slot is not None:
        routed = slot.run(x, topk_output, routed_pre_quant)
    elif deferred:
        routed = mlp.experts.forward_deferred_finalize(
            x, topk_output, pre_quant_input=routed_pre_quant
        )
    elif bypass:
        routed = mlp.experts.forward_impl(x, topk_output)
    elif routed_pre_quant is not None:
        routed = mlp.experts(x, topk_output, pre_quant_input=routed_pre_quant)
    else:
        routed = mlp.experts(x, topk_output)

    with torch.cuda.stream(mlp.alt_stream):
        if shared_ll is not None:
            _log_shared_ll(x.shape[0])
            shared = shared_ll(x)
        else:
            shared = mlp._forward_shared_experts(
                x, None, pre_quant_input=pre_quant_input
            )
    current.wait_stream(mlp.alt_stream)

    if deferred and fwd.defer_moe_finalize:
        from sglang.srt.layers.moe.utils import sum_post_experts_output

        return sglang_ops.moe_deferred_finalize_from_flashinfer(
            routed,
            gated_shared_output=shared,
            m=x.shape[0],
            reduce=sum_post_experts_output,
        )
    return _pieces(mlp, routed, shared)
