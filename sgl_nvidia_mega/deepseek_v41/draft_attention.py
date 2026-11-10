# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/deepseek_v4_dspark.py (DSparkAttention.forward).
"""The DSpark draft stage's attention on the megakernel's kernels (slot "draft_attn").

The stock arms call SGLang's own draft methods in SGLang's order, so the step is bitwise
to ``DSparkAttention.forward``. Each further slot puts one of the target chain's kernels
in place of a stock op, on the rows the target serves with it:
- draft_qfront: the q-LoRA RMSNorm + MXFP8 quant and the low-latency wq_b (1-48 rows);
- draft_wo_a: the fused inverse RoPE + wo_a + MXFP8 for wo_b (1-48 rows);
- draft_swapab: swap-AB attention on the 16 local heads (8 at TP8 with slot swapab_h8),
  roping q as it loads (1-24);
- draft_qkv_a: wq_a and wkv as one MXFP8 GEMM on SGLang's dispatch; draft_qkv_a_ll:
  that GEMM on the low-latency kernel at 1-8 rows;
- draft_wo_b_ll: wo_b on the low-latency kernel at 1-8 rows, fed by draft_wo_a.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from sgl_nvidia_mega.deepseek_v41.attention import chain
from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll
from sgl_nvidia_mega.deepseek_v41.sm103.attention import wo_a as wo_a_kernel

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/deepseek_v4_dspark.py": (
        "_PAD_NUM_HEADS",
        "_compute_q",
        "kv_proj_only",
        "freqs_cis",
        "alt_streams",
        "_multi_stream_bs_limit",
        "_use_fast_kernel",
        "attn",
    ),
    "srt/models/deepseek_v4.py": (
        "_apply_wo_a_bf16_matmul",
        "_fused_wo_a_arch_supported",
        "is_dsv41",
        "attn_tp_size",
        "wo_b",
        "wq_a",
        "wkv",
        "rope_head_dim",
        "use_flashinfer_mxfp8_wo_b",
        # the draft window store's arguments (DSparkAttention._store_block_kv)
        "kv_norm",
        "eps",
        "layer_id",
        # the module alias SGLang's DSV4 model imports too
        "mhc",
    ),
    "srt/models/deepseek_v4_mhc.py": ("AttnOutput",),
    "srt/layers/quantization/mxfp8_input.py": ("Mxfp8SwizzledInput",),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "kv_pool",
        "swa_out_cache_loc",
        "attention",
        "uniform_fp8",
        "set_swa_key_buffer_radix_fused_norm_rope",
    ),
    "srt/runtime_context.py": ("get_forward", '"sp_active"'),
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
    "srt/layers/linear.py": ("quant_method",),
    "srt/layers/quantization/fp8.py": (
        "weight_scale_inv_swizzled",
        "mxfp8_dense_backend",
    ),
    "srt/layers/quantization/fp8_utils.py": (
        "is_flashinfer_cutedsl",
        "is_flashinfer_cutlass",
    ),
    "kernels/ops/attention/dsv4/__init__.py": (
        "fused_q_norm_rope",
        "fused_rope_inplace",
    ),
}

_SCALE_ROWS = 128  # the swizzled UE8M0 scales are tiled 128 weight rows at a time


class FusedQkvA:
    """wq_a and wkv as one GEMM over [wq_a; wkv]: a shallow clone of wq_a holding the
    concatenated weight and scales, so SGLang's own MXFP8 dispatch, input quantization
    and tactic cache apply to it. Returns [rows, q_lora_rank + head_dim]. A copy made
    at build: a weight update after load does not reach it."""

    def __init__(self, wq_a, wkv, *, ll: bool = False):
        fused = torch.nn.Module.__new__(type(wq_a))
        fused.__dict__.update(wq_a.__dict__)
        fused._parameters = dict(wq_a._parameters)
        fused._buffers = dict(wq_a._buffers)
        fused._modules = dict(wq_a._modules)
        weight = torch.cat([wq_a.weight.data, wkv.weight.data])
        fused._parameters["weight"] = torch.nn.Parameter(weight, requires_grad=False)
        # the scales tile per 128 weight rows, so the two linears' tiles concatenate
        scales = (wq_a.weight_scale_inv_swizzled, wkv.weight_scale_inv_swizzled)
        shape = (-1,) if scales[0].dim() == 1 else (-1, scales[0].shape[-1])
        # a registered parameter on wq_a: the clone holds the concatenation as a plain
        # attribute, which the MXFP8 apply reads the same way
        fused._parameters.pop("weight_scale_inv_swizzled", None)
        fused.weight_scale_inv_swizzled = torch.cat(
            [s.reshape(-1) for s in scales]
        ).view(*shape)
        fused.output_size_per_partition = weight.shape[0]
        self.fused, self.split = fused, wq_a.weight.shape[0]
        # at 1-8 rows: FlashInfer's MXFP8 quantize + the low-latency GEMM (A03's tactics,
        # the target wqkv_a's shape) in place of SGLang's dense dispatch
        self.ll = ll and mxfp8_ll.supported() and weight.is_contiguous()

    def __call__(self, x):
        if self.ll and 0 < x.shape[0] <= mxfp8_ll.MAX_ROWS:
            q, sf = mxfp8_ll.quantize(x)
            f = self.fused
            return mxfp8_ll.linear(
                q, sf.reshape(-1), f.weight, f.weight_scale_inv_swizzled
            )
        return self.fused.quant_method.apply(self.fused, x, None)

    def kv(self, qkv_a):
        from sgl_nvidia_mega.deepseek_v41 import sglang_ops

        kv = qkv_a[:, self.split :]
        # the unified-KV store norms + ropes kv in place and scatters contiguous rows only
        return kv.contiguous() if sglang_ops.unified_kv_triton() else kv


def qkv_a_fusable(wq_a, wkv) -> str:
    """Why wq_a and wkv cannot be one GEMM on SGLang's MXFP8 dispatch, or ""."""
    for name, linear in (("wq_a", wq_a), ("wkv", wkv)):
        if (
            getattr(linear, "weight_scale_inv_swizzled", None) is None
            or linear.weight.dtype != torch.float8_e4m3fn
        ):
            return f"{name} is not an MXFP8 linear with swizzled scales"
        if linear.weight.shape[0] % _SCALE_ROWS or linear.bias is not None:
            return f"{name} {tuple(linear.weight.shape)} does not tile by {_SCALE_ROWS}"
    if type(wq_a.quant_method) is not type(wkv.quant_method):
        return "wq_a and wkv use different quant methods"
    if wq_a.weight.shape[1] != wkv.weight.shape[1]:
        return "wq_a and wkv read different inputs"
    backend = getattr(wq_a.quant_method, "mxfp8_dense_backend", None)
    if backend is None or not (
        backend.is_flashinfer_cutedsl() or backend.is_flashinfer_cutlass()
    ):
        # the backends whose MXFP8 apply reads only the swizzled scales
        return f"MXFP8 dense backend {backend} reads other scales"
    return ""


class Kernels(NamedTuple):
    """The slots' kernels a stage's attention takes; fixed at build."""

    qfront: bool = False
    wo_a: bool = False
    swapab: bool = False
    qkv_a: Optional[FusedQkvA] = None
    wo_b_ll: bool = False
    # swap-AB gathers the KV before its PDL wait where the store joined first (slot
    # kv_before_wait)
    kv_before_wait: bool = False


def wo_a_applies(attn, one_group: bool = False) -> bool:
    """The draft's wo_a fits the fused kernel (the target's ``use_fused_wo_a`` shape;
    with ``one_group``, attention TP8's 1 local group too) and wo_b takes its MXFP8
    rows."""
    from sglang.srt.models.deepseek_v4 import _fused_wo_a_arch_supported

    groups = wo_a_kernel.GROUP_COUNTS if one_group else (wo_a_kernel.GROUPS,)
    return (
        attn.is_dsv41
        and _fused_wo_a_arch_supported()
        and attn.wo_a.weight.dtype == torch.bfloat16
        and attn.wo_a.weight.shape == (attn.n_local_groups * attn.o_lora_rank, 4096)
        and attn.n_local_groups in groups
        and attn.o_lora_rank == wo_a_kernel.RANK
        and attn.use_flashinfer_mxfp8_wo_b
    )


def build(stages, slots: frozenset[str]) -> dict[int, Kernels]:
    """Each ported stage's kernels by layer id; a stage left out keeps SGLang's call
    (slot draft_attn off, or SGLang's draft without its fast kernels)."""
    from sgl_nvidia_mega.deepseek_v41 import sglang_ops

    # unified KV writes the draft window through a ring the binding does not offer
    if "draft_attn" not in slots or sglang_ops.unified_kv_triton():
        return {}
    kernels = {}
    for stage in stages:
        attn = stage.self_attn
        if not attn._use_fast_kernel:
            continue
        fusable = "draft_qkv_a" in slots and not qkv_a_fusable(attn.wq_a, attn.wkv)
        wo_a = "draft_wo_a" in slots and wo_a_applies(attn, "wo_a_g1" in slots)
        kernels[stage.layer_id] = Kernels(
            # the chain's query kernels have no per-head norm
            qfront="draft_qfront" in slots
            and not attn.q_head_norm
            and chain.wq_b_ll_applies(attn),
            wo_a=wo_a,
            swapab="draft_swapab" in slots
            and chain.swapab_applies(attn, heads8="swapab_h8" in slots),
            kv_before_wait="draft_swapab" in slots and "kv_before_wait" in slots,
            qkv_a=(
                FusedQkvA(attn.wq_a, attn.wkv, ll="draft_qkv_a_ll" in slots)
                if fusable
                else None
            ),
            # wo_b reads the fused wo_a's MXFP8 rows (FlashInfer's 128x4 scales)
            wo_b_ll="draft_wo_b_ll" in slots
            and wo_a
            and attn.wo_b.bias is None
            and chain._ll_linear_applies(attn.wo_b),
        )
    return kernels


def precompile(stages, kernels: dict[int, Kernels], widths) -> None:
    """Compile the slots' kernels that do not depend on the KV pool, before capture."""
    from sgl_nvidia_mega.deepseek_v41.sm103.attention.q_norm_quant import (
        rmsnorm_mxfp8,
    )

    rows = [w for w in widths if w <= chain.NORM_QUANT_MAX_ROWS]
    ll_rows = [w for w in widths if w <= mxfp8_ll.MAX_ROWS]
    for stage in stages:
        k, attn = kernels.get(stage.layer_id), stage.self_attn
        if k is None or not rows:
            continue
        if k.qkv_a is not None and k.qkv_a.ll and ll_rows:
            mxfp8_ll.precompile(k.qkv_a.fused.weight, ll_rows)
        if k.wo_b_ll and ll_rows:
            mxfp8_ll.precompile(attn.wo_b.weight, ll_rows)
        if k.qfront:
            mxfp8_ll.precompile(attn.wq_b.weight, rows)
            q = attn.q_lora_rank
            # the norm compiles its input's row stride: a view of the fused output
            width = q + (attn.head_dim if k.qkv_a is not None else 0)
            x = torch.zeros(
                max(rows), width, dtype=torch.bfloat16, device=attn.q_norm.weight.device
            )
            for r in rows:
                rmsnorm_mxfp8(
                    x[:r, :q],
                    attn.q_norm.weight,
                    attn.q_norm.variance_epsilon,
                    pdl=True,
                    row_tiles=r > mxfp8_ll.MAX_ROWS,
                )
        if k.wo_a:
            wo_a_kernel.precompile(
                min(max(rows), chain.WO_A_MAX_ROWS), groups=attn.n_local_groups
            )


def forward(attn, x, positions, fb, *, kernels: Kernels, view):
    """``DSparkAttention.forward(..., defer_all_reduce=True)`` with ``kernels``; ``view``:
    the draft forward's ``ForwardView`` (its KV pool, metadata and attention)."""
    from sglang.srt.model_executor.runner import get_is_capture_mode
    from sglang.srt.models.deepseek_v4_dspark import _PAD_NUM_HEADS

    if fb.forward_mode.is_idle():
        return torch.zeros_like(x)
    pool = view.kv_pool
    rows = x.shape[0]
    swapab = kernels.swapab and _swapab_takes(rows, fb, view)
    q_padded = q_out = None
    if (
        pool.uniform_fp8
        and not attn.q_head_norm
        and attn.n_local_heads in (8, 16, 32, 64, 128)
    ):
        q_out = torch.empty(
            (rows, attn.n_local_heads, attn.head_dim),
            dtype=torch.float8_e4m3fn,
            device=x.device,
        )
    elif attn.n_local_heads < _PAD_NUM_HEADS and not swapab:
        # swap-AB reads only the real heads, so its unroped q stays unpadded
        q_padded = x.new_empty(rows, _PAD_NUM_HEADS, attn.head_dim)
        q_out = q_padded[:, : attn.n_local_heads, :]

    def kv_and_store():
        kv = attn.kv_proj_only(x) if qkv_a is None else kernels.qkv_a.kv(qkv_a)
        # DSparkAttention._store_block_kv (the SWA window; unified-KV declines at build)
        pool.set_swa_key_buffer_radix_fused_norm_rope(
            layer_id=attn.layer_id,
            swa_loc=view.swa_out_cache_loc(fb),
            kv=kv,
            kv_weight=attn.kv_norm.weight.data,
            eps=attn.eps,
            freqs_cis=attn.freqs_cis,
            positions=positions,
        )
        return kv

    # the fused GEMM runs before the fork; the side stream takes its kv columns
    qkv_a = kernels.qkv_a(x) if kernels.qkv_a is not None else None
    # the KV store joined the attention's stream (a full graph edge), so swap-AB may
    # gather the KV before its PDL wait
    store_joined = False
    if (
        attn.alt_streams is not None
        and get_is_capture_mode()
        and rows <= attn._multi_stream_bs_limit
    ):
        current, side = torch.cuda.current_stream(), attn.alt_streams[0]
        side.wait_stream(current)
        with torch.cuda.stream(side):
            kv = kv_and_store()
        q = _compute_q(attn, x, positions, q_out, qkv_a, kernels.qfront, swapab)
        current.wait_stream(side)
        store_joined = True
    else:
        kv = kv_and_store()
        q = _compute_q(attn, x, positions, q_out, qkv_a, kernels.qfront, swapab)

    if swapab:
        freqs = torch.view_as_real(attn.freqs_cis).flatten(-2)
        o = chain.sparse_decode(
            attn,
            view,
            q,
            positions,
            freqs,
            kv_before_wait=kernels.kv_before_wait and store_joined,
            model="draft",
        )
    else:
        o = view.attention(
            q=q_padded if q_padded is not None else q,
            k=kv,
            v=kv,
            layer=attn.attn,
            forward_batch=fb,
            compress_ratio=0,
            attn_sink=attn._local_attn_sink(),
            save_kv_cache=False,
        )
    if o.shape[1] != attn.n_local_heads:
        o = o[:, : attn.n_local_heads, :]
    return _wo_b(attn, _wo_a(attn, o, positions, fb, kernels.wo_a), kernels.wo_b_ll)


def _swapab_takes(rows, fb, view) -> bool:
    """The target's swap-AB step conditions, on the draft's own window cache."""
    return (
        not view.kv_pool.uniform_fp8  # an FP8 q never takes swap-AB
        and chain.swapab_takes(rows, fb, view)
    )


def _compute_q(attn, x, positions, q_out, qkv_a, qfront, swapab):
    """``DSparkAttention._compute_q`` (fast-kernel arm), on the fused GEMM's q columns,
    the chain's query kernels, or unroped for swap-AB."""
    from sglang.kernels.ops.attention.dsv4 import fused_q_norm_rope, fused_rope_inplace

    if not (qfront or swapab or qkv_a is not None):
        return attn._compute_q(x, positions, q_out=q_out)
    q = qkv_a[:, : attn.q_lora_rank] if qkv_a is not None else attn.wq_a(x)[0]
    if qfront:
        ops = chain.ChainOps(attn, rope_in_attention=swapab)
        _, q = ops.normalize_q_lora(attn, q)
        return ops.compute_q_b(attn, q, positions, q_out)
    q, _ = attn.wq_b(attn.q_norm(q))
    q = q.view(-1, attn.n_local_heads, attn.head_dim)
    if swapab:
        return q
    if attn.q_head_norm:
        if q_out is None:
            q_out = torch.empty_like(q)
        fused_q_norm_rope(q, q_out, attn.eps, attn.freqs_cis, positions)
        return q_out
    if q_out is not None and q_out.dtype == torch.float8_e4m3fn:
        fused_q_norm_rope(q, q_out, None, attn.freqs_cis, positions)
        return q_out
    fused_rope_inplace(
        q[..., -attn.rope_head_dim :], None, attn.freqs_cis, positions=positions
    )
    if q_out is None:
        return q
    q_out.copy_(q)
    return q_out


def _wo_a(attn, o, positions, fb, fused: bool):
    """The inverse RoPE + wo_a: fused into wo_b's MXFP8 rows, or SGLang's draft arm."""
    from sglang.kernels.ops.attention.dsv4 import fused_rope_inplace
    from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput
    from sglang.srt.models.deepseek_v4 import _apply_wo_a_bf16_matmul
    from sglang.srt.runtime_context import get_forward

    # the decode attention TP context can change the local groups after build; the
    # target takes the MXFP8 rows only without sequence parallelism
    if (
        fused
        and attn.n_local_groups == attn.wo_a.weight.shape[0] // attn.o_lora_rank
        and not get_forward().sp_active
        and 0 < o.shape[0] <= chain.WO_A_MAX_ROWS
    ):
        codes, scales = wo_a_kernel.fused_rope_wo_a(
            o.view(o.shape[0], attn.n_local_groups, -1),
            attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1),
            torch.view_as_real(attn.freqs_cis).flatten(1),
            positions,
            early_trigger=True,
        )
        return Mxfp8SwizzledInput(codes, scales)
    rd = attn.rope_head_dim
    fused_rope_inplace(
        o[..., -rd:], None, attn.freqs_cis, positions=positions, inverse=True
    )
    o = o.view(o.shape[0], attn.n_local_groups, -1)
    mode = fb.forward_mode
    o = _apply_wo_a_bf16_matmul(
        o,
        attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1),
        is_decode=mode.is_decode(),
        is_target_verify=mode.is_target_verify(),
        fast_path=attn.is_dsv41,
    )
    return o.reshape(o.shape[0], o.shape[1] * o.shape[2])


def _wo_b(attn, o, ll: bool = False):
    """wo_b with the TP sum left to the stage's attention post, as SGLang's draft; with
    slot draft_wo_b_ll, the fused wo_a's MXFP8 rows on the low-latency GEMM at 1-8
    rows (the target wo_b's shape and tactics)."""
    from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput
    from sglang.srt.models import deepseek_v4_mhc as mhc

    if (
        ll
        and isinstance(o, Mxfp8SwizzledInput)
        and 0 < o.data.shape[0] <= mxfp8_ll.MAX_ROWS
    ):
        w = attn.wo_b
        out = mxfp8_ll.linear(
            o.data, o.scales.reshape(-1), w.weight, w.weight_scale_inv_swizzled
        )
    else:
        out, _ = attn.wo_b(o, skip_all_reduce=True)
    return mhc.AttnOutput(out) if attn.attn_tp_size > 1 else out
