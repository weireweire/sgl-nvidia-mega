# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/deepseek_v4.py (MQALayer.forward).
"""The attention of a covered decode step: ``MQALayer.forward(defer_all_reduce=True)``
on CUDA for DeepSeek-V4.1, over the layer's loaded weights and SGLang's attention
backend and caches.

The stock step runs SGLang's own ops in SGLang's order, so it is bitwise to
SGLang's decode attention; the plugin's kernels replace single ops in it."""

from __future__ import annotations

from typing import Any

import torch

from sgl_nvidia_mega.deepseek_v41.attention import chain, ops, prepare

# The SGLang names this package reads, by file under the sglang package (``f(p)``: a
# parameter); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/models/deepseek_v4.py": (
        "MQALayer",
        "is_dsv41",
        "use_mega_attention",
        "q_head_norm",
        "fuse_wqa_wkv",
        "wqkv_a",
        "wq_a",
        "wkv",
        "wq_b",
        "q_norm",
        "kv_norm",
        "wo_a",
        "wo_b",
        "wo_a_fp8",
        "use_fused_wo_a",
        "use_flashinfer_mxfp8_wo_b",
        "use_npu_arch35_mxfp8_wo_a",
        "alt_streams",
        "compressor",
        "indexer",
        "compress_ratio",
        "layer_id",
        "q_lora_rank",
        "n_local_heads",
        "n_local_groups",
        "o_lora_rank",
        "head_dim",
        "qk_rope_head_dim",
        "freqs_cis",
        "eps",
        "attn_mqa",
        "attn_tp_size",
        "_multi_stream_bs_limit",
        "_kernel_num_heads",
        "_local_attn_sink",
        "maybe_use_decode_attn_tp",
        "_FP8_WO_A_UE8M0",
        "_FUSED_WO_A_MAX_TOKENS",
        "_apply_wo_a_bf16_matmul",
        "_fp8_wo_a_einsum",
        "_wo_a_fp8_mxscale",
        "_wo_a_fp8_mxscale_fused_invrope",
    ),
    "srt/models/deepseek_v4_mhc.py": ("AttnOutput",),
    "srt/layers/layernorm.py": ("weight", "variance_epsilon"),
    "kernels/ops/attention/dsv4/__init__.py": (
        "fused_norm_rope_inplace",
        "fused_q_norm_rope",
        "fused_rope_inplace",
    ),
    "kernels/ops/attention/dsv4/wo_a.py": ("fused_rope_wo_a_bf16",),
    "kernels/ops/quantization/fp8_kernel.py": ("sglang_per_token_group_quant_fp8",),
    "srt/layers/quantization/fp8_utils.py": (
        "Mxfp8DenseGemmBackend",
        "FLASHINFER_CUTEDSL",
    ),
    "srt/layers/quantization/fp8.py": (
        "mxfp8_dense_backend",
        "use_mxfp8",
        "block_fp8_mxfp8_ready",
    ),
    "srt/layers/quantization/mxfp8_input.py": ("Mxfp8SwizzledInput",),
    "srt/layers/moe/utils.py": ("should_skip_mlp_all_reduce",),
    # the metadata field the step writes (a tolerated write of the binding), and the
    # keywords the step passes through ForwardView.low_ratio_sources to SGLang's own
    "srt/layers/attention/deepseek_v4_backend.py": (
        "q_pad_buffer",
        "forward_low_ratio_sources(run_indexer)",
        "forward_low_ratio_sources(run_compressor)",
    ),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "ForwardView",
        "metadata",
        "kv_pool",
        "attention",
        "store_cache",
        "swa_out_cache_loc",
        "low_ratio_sources",
        "core_compressor",
        "uniform_fp8",
        "set_swa_key_buffer_radix_fused_norm_rope",
    ),
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
    "srt/model_executor/runner_backend_utils/breakable_cuda_graph/context.py": (
        "is_in_breakable_cuda_graph",
    ),
    "srt/batch_invariant_ops/*.py": ("is_batch_invariant_mode_enabled",),
    "srt/runtime_context.py": ("get_exec", "get_forward", '"sp_active"'),
    "srt/environ.py": (
        "envs",
        "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP",
        "SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE",
    ),
}


def disable_reason(attn) -> str:
    """Why the stock step cannot stand in for ``attn``'s forward, or ""."""
    from sglang.srt.models.deepseek_v4 import (
        MQALayer,
        _wo_a_fp8_mxscale,
        _wo_a_fp8_mxscale_fused_invrope,
    )

    from sgl_nvidia_mega.deepseek_v41 import sglang_ops

    if not isinstance(attn, MQALayer):
        return f"layer {attn.layer_id}: {type(attn).__name__}, not MQALayer"
    if not attn.is_dsv41 or attn.use_mega_attention:
        return f"layer {attn.layer_id}: not the DSV4.1 FlashMLA attention"
    if attn.compress_ratio not in (0, 1, 2):
        return f"layer {attn.layer_id}: compress ratio {attn.compress_ratio}"
    if _wo_a_fp8_mxscale is not None or _wo_a_fp8_mxscale_fused_invrope is not None:
        return "the ROCm wo_a kernels are loaded"
    if attn.use_npu_arch35_mxfp8_wo_a:
        return f"layer {attn.layer_id}: the NPU wo_a"
    if sglang_ops.unified_kv_triton():
        return "the unified KV cache"
    if not hasattr(attn.wo_a, "weight"):
        return f"layer {attn.layer_id}: wo_a has no dense weight"
    return ""


class AttentionStep:
    """One layer's attention in a covered step, over the layer's own modules."""

    def __init__(
        self,
        attn,
        *,
        chain_ops: bool = False,
        qkv_a_ll: bool = False,
        qkv_a_bf16: bool = False,
        index_q_mxfp8_in: bool = False,
        kv_before_wait: bool = False,
        index_after_wait: bool = False,
        swapab_h8: bool = False,
        wo_a_g1: bool = False,
        q_norm_gemm: bool = False,
        index_q_first: bool = False,
        index_weights_bf16: bool = False,
    ):
        self.attn = attn
        # swap-AB gathers the KV before its PDL wait on the low-ratio prepare (slot
        # "kv_before_wait")
        self.kv_before_wait = chain_ops and kv_before_wait
        # slot attn_index_after_wait, an indexer layer: swap-AB reads the layer's own
        # selection after its PDL wait unless the plugin's fused consumer wrote it
        self.index_after_wait = (
            self.kv_before_wait and index_after_wait and attn.indexer is not None
        )
        self.low_ratio = attn.compress_ratio in (1, 2)
        # the attention chain's query ops, padded / roped query and unroped 16 heads
        self.chain = None
        if chain_ops:
            self.chain = tuple(
                chain.ChainOps(
                    attn,
                    rope_in_attention=rope,
                    qkv_a_ll=qkv_a_ll,
                    qkv_a_bf16=qkv_a_bf16,
                    index_q_mxfp8_in=index_q_mxfp8_in,
                    q_norm_gemm=q_norm_gemm,
                    index_q_first=index_q_first,
                    index_weights_bf16=index_weights_bf16,
                )
                for rope in (False, True)
            )
        # slot swapab_h8: swap-AB on attention TP8's 8 local heads
        self.swapab = chain_ops and chain.swapab_applies(attn, heads8=swapab_h8)
        # slot wo_a_g1: the fused wo_a on attention TP8's 1 local group
        self.fused_wo_a = chain_ops and chain.fused_wo_a_applies(attn, wo_a_g1)
        # fp32 [positions, 64]: cos, sin interleaved, as the swap-AB query RoPE reads them
        self.freqs = (
            torch.view_as_real(attn.freqs_cis).flatten(-2).contiguous()
            if self.swapab
            else None
        )

    def _prepare_path(self, rows: int, fb):
        """The prepare ``MQALayer.forward`` picks for this step on CUDA."""
        from sglang.srt.environ import envs
        from sglang.srt.layers.quantization.fp8_utils import Mxfp8DenseGemmBackend
        from sglang.srt.model_executor.runner import get_is_capture_mode
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
            is_in_breakable_cuda_graph,
        )

        attn = self.attn
        if attn.alt_streams is None:
            return prepare.eager
        if (
            not self.low_ratio
            and envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            and get_is_capture_mode()
            and (is_in_breakable_cuda_graph() or rows <= attn._multi_stream_bs_limit)
        ):
            return prepare.multi_stream
        mode = fb.forward_mode
        if self.low_ratio and (
            mode.is_decode()
            or (
                mode.is_target_verify()
                and getattr(attn.wq_b.quant_method, "mxfp8_dense_backend", None)
                == Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL
            )
        ):
            return prepare.low_ratio
        return prepare.eager

    def _query_buffer(self, x, view):
        """The query buffer as ``MQALayer.forward`` lays it out: the local heads, or
        the zeroed 64-head FlashMLA pad shared through the step's metadata."""
        attn = self.attn
        heads = attn._kernel_num_heads(x.shape[0])
        if heads == attn.n_local_heads:
            return heads, None, None, slice(None)
        want = (x.shape[0], heads, attn.head_dim)
        meta = view.metadata
        q_padded = getattr(meta, "q_pad_buffer", None)
        if (
            q_padded is None
            or tuple(q_padded.shape) != want
            or q_padded.dtype != x.dtype
        ):
            q_padded = x.new_zeros(*want)
            if meta is not None:
                try:
                    meta.q_pad_buffer = q_padded
                except (AttributeError, TypeError):
                    pass
        tp_slice = slice(0, attn.n_local_heads)
        return heads, q_padded, q_padded[:, tp_slice, :], tp_slice

    def takes_swapab(self, rows: int, fb, view) -> bool:
        """Whether this step's attention runs the chain's swap-AB at ``rows`` rows."""
        return (
            self.chain is not None
            and rows <= chain.NORM_QUANT_MAX_ROWS
            and self.swapab
            and chain.swapab_takes(rows, fb, view)
        )

    def run(self, x, positions, fb, x_quant=None, publish=None, *, view) -> Any:
        """The layer's TP-partial attention output (``mhc.AttnOutput``) for input
        ``x`` (``x_quant``: the same input as MXFP8); ``publish``: see ``ops.wo_b``."""
        attn = self.attn
        fused = self.fused_wo_a
        assert not view.kv_pool.uniform_fp8, "the uniform-FP8 pool"
        rows = x.shape[0]
        use_chain = self.chain is not None and rows <= chain.NORM_QUANT_MAX_ROWS
        swapab = self.takes_swapab(rows, fb, view)
        with attn.maybe_use_decode_attn_tp(fb):
            path = self._prepare_path(rows, fb)
            if swapab:
                # the attention ropes the 16-head query; no 64-head pad
                q = path(attn, x, positions, fb, view, None, x_quant, self.chain[1])
                # the low-ratio prepare joins its KV / index streams before the query,
                # so the attention may gather the KV before its wait. Under capture
                # those joins are programmatic edges: a stock top-k triggers before it
                # writes the selection, which slot attn_index_after_wait then reads
                # after the wait (the fused consumer fences it before its trigger)
                early = self.kv_before_wait and path is prepare.low_ratio
                after_wait = early and self.index_after_wait
                if after_wait and self.chain[1].indexer is not None:
                    after_wait = not self.chain[1].indexer.wrote_fused(attn.layer_id)
                o = chain.sparse_decode(
                    attn,
                    view,
                    q,
                    positions,
                    self.freqs,
                    kv_before_wait=early,
                    extra_after_wait=after_wait,
                )
                tiles = _publish_row_tiles(rows, publish)
                return ops.wo_b(
                    attn,
                    chain.wo_a(attn, o, positions, fb, tiles, fused=fused),
                    publish,
                )
            heads, q_padded, q_out, tp_slice = self._query_buffer(x, view)
            attn_sink = attn._local_attn_sink(heads)
            q_ops = self.chain[0] if use_chain else ops
            q = path(attn, x, positions, fb, view, q_out, x_quant, q_ops)
            o = view.attention(
                q=q_padded if q_padded is not None else q,
                k=q,
                v=q,
                layer=attn.attn_mqa,
                forward_batch=fb,
                compress_ratio=attn.compress_ratio,
                attn_sink=attn_sink,
                save_kv_cache=False,
            )
            o = o[:, tp_slice, :]
            if use_chain:
                tiles = _publish_row_tiles(rows, publish)
                return ops.wo_b(
                    attn,
                    chain.wo_a(attn, o, positions, fb, tiles, fused=fused),
                    publish,
                )
            return ops.wo_b(attn, ops.wo_a(attn, o, positions, fb), publish)


def _publish_row_tiles(rows: int, publish) -> bool:
    """Whether wo_a writes row-tile scales: hop A's publishing wo_b reads them past 8
    rows (SGLang's wo_b, its fallback, cannot: ``ops.wo_b`` refuses them)."""
    from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll

    return publish is not None and rows > mxfp8_ll.MAX_ROWS
