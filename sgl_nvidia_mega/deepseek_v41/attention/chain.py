# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The attention chain at 1-48 rows: the stock step with the megakernel's kernels in
place of single ops, each launched as a programmatic dependent (PDL) of the last.

- q-LoRA RMSNorm + MXFP8 quant (1-48 rows), then wq_b on the low-latency MXFP8 GEMM;
- at 1-24 rows the query keeps the 16 local heads (8 at attention TP8, slot
  swapab_h8), unroped, and the swap-AB sparse attention ropes it as it loads it; its
  narrow combine triggers wo_a at entry;
- the fused inverse RoPE + wo_a + MXFP8, which triggers wo_b at entry (1-48 rows; on
  TP8's one local group with slot wo_a_g1).
At 25-48 rows the attention itself is SGLang's, on the 64-head query."""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch

from sgl_nvidia_mega.deepseek_v41 import plan as slot_plan
from sgl_nvidia_mega.deepseek_v41.attention import ops
from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll
from sgl_nvidia_mega.deepseek_v41.sm103.attention import swapab
from sgl_nvidia_mega.deepseek_v41.sm103.attention import wo_a as wo_a_kernel
from sgl_nvidia_mega.deepseek_v41.sm103.attention.q_norm_quant import rmsnorm_mxfp8

logger = logging.getLogger(__name__)

NORM_QUANT_MAX_ROWS = 48
SWAPAB_MAX_ROWS = 24  # swap-AB served slower than FlashMLA at 48 rows
WO_A_MAX_ROWS = wo_a_kernel.MAX_ROWS

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/quantization/fp8.py": (
        "Fp8LinearMethod",
        "use_mxfp8",
        "block_fp8_as_mxfp8",
        "block_fp8_mxfp8_ready",
        "mxfp8_dense_backend",
        "weight_scale_inv_swizzled",
    ),
    "srt/models/deepseek_v4.py": (
        "wq_b",
        "q_norm",
        "wo_a",
        "freqs_cis",
        "n_local_heads",
        "n_local_groups",
        "o_lora_rank",
        "head_dim",
        "q_lora_rank",
        "layer_id",
        "compress_ratio",
        "use_fused_wo_a",
        "use_flashinfer_mxfp8_wo_b",
        "_fused_wo_a_arch_supported",
        "is_dsv41",
        "wo_a_fp8",
        "use_npu_arch35_mxfp8_wo_a",
        "_local_attn_sink",
        "q_head_norm",
        "softmax_scale",
        "wqkv_a",
        "fuse_wqa_wkv",
        "indexer",
        "alt_streams",
    ),
    "srt/layers/layernorm.py": ("weight", "variance_epsilon"),
    "kernels/ops/attention/dsv4/__init__.py": ("fused_q_norm_rope",),
    "srt/layers/quantization/mxfp8_input.py": ("Mxfp8SwizzledInput",),
    "srt/layers/attention/deepseek_v4_backend.py": (
        "DSV4AttnMetadata",
        "core_attn_metadata",
        "swa_page_indices",
        "swa_topk_lengths",
        "sparse_page_indices",
        "sparse_topk_lengths",
        "_pad_tensor_to_size",
    ),
    "srt/layers/attention/dsv4/dsv41_sparse.py": ("wq_b",),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "ForwardView",
        "metadata",
        "kv_pool",
    ),
    "srt/mem_cache/deepseek_v4_memory_pool.py": (
        "request_window",
        "get_swa_key_buffer_radix",
        "get_swa_key_page_size",
        "get_swa_key_bytes_per_token",
        "get_extra_key_buffer",
        "get_extra_key_page_size",
        "get_extra_key_bytes_per_token",
    ),
    "srt/runtime_context.py": ("get_forward", '"sp_active"'),
    "srt/environ.py": ("envs", "SGLANG_DSV41_FUSED_WO_A"),
}


def _ll_linear_applies(linear) -> bool:
    return (
        mxfp8_ll.supported()
        and getattr(linear, "weight_scale_inv_swizzled", None) is not None
        and linear.weight.dtype == torch.float8_e4m3fn
        and linear.weight.is_contiguous()
    )


def swapab_applies(attn, heads8: bool = False) -> bool:
    """The layer's query fits the swap-AB kernel: 16 local heads of 512 (or, with
    ``heads8``, attention TP8's 8), no head norm."""
    heads = (8, swapab.NUM_HEADS) if heads8 else (swapab.NUM_HEADS,)
    return (
        not attn.q_head_norm
        and attn.n_local_heads in heads
        and attn.head_dim == swapab.HEAD_DIM
        and attn.softmax_scale == swapab.SOFTMAX_SCALE
    )


def fused_wo_a_applies(attn, one_group: bool = False) -> bool:
    """The layer's wo_a runs the fused kernel: where SGLang's step fuses it (2 local
    groups) or, with ``one_group``, on attention TP8's 1 local group under SGLang's
    other conditions (SGLang's kernel takes 2 or 4 groups only and runs the einsum)."""
    if attn.use_fused_wo_a or not one_group:
        return attn.use_fused_wo_a
    from sglang.srt.environ import envs
    from sglang.srt.models.deepseek_v4 import _fused_wo_a_arch_supported

    return (
        attn.is_dsv41
        and envs.SGLANG_DSV41_FUSED_WO_A.get()
        and _fused_wo_a_arch_supported()
        and not attn.wo_a_fp8
        and not attn.use_npu_arch35_mxfp8_wo_a
        and attn.wo_a.weight.dtype == torch.bfloat16
        and attn.n_local_groups == 1
        and attn.o_lora_rank == wo_a_kernel.RANK
        and tuple(attn.wo_a.weight.shape) == (wo_a_kernel.RANK, wo_a_kernel.GROUP_K)
    )


def swapab_takes(rows: int, fb, view) -> bool:
    """This step's swap-AB conditions, from host state every rank shares: a decode or
    verify step of 1-24 rows on the paged V4-layout window cache."""
    from sglang.srt.layers.attention.deepseek_v4_backend import DSV4AttnMetadata

    mode = fb.forward_mode
    pool = view.kv_pool
    return (
        0 < rows <= SWAPAB_MAX_ROWS
        and (mode.is_decode() or mode.is_target_verify())
        and isinstance(view.metadata.core_attn_metadata, DSV4AttnMetadata)
        and pool.request_window is None
        and pool.get_swa_key_bytes_per_token() == swapab.LAYOUT.bytes_per_token
    )


class _Engagement:
    """Which layers' wqkv_a ran on the low-latency GEMM, logged once per row count when
    the first layer repeats: the boundary hands it MXFP8 rows only where its norm fuses.
    Host side only; graph replays never reach it."""

    def __init__(self):
        self._seen: dict[int, dict[int, bool]] = {}
        self._logged: set[int] = set()

    def note(self, layer_id: int, rows: int, ll: bool) -> None:
        if rows in self._logged:
            return
        seen = self._seen.setdefault(rows, {})
        if layer_id not in seen:
            seen[layer_id] = ll
            return
        self._logged.add(rows)
        fell_back = sorted(i for i, on in seen.items() if not on)
        logger.info(
            "NVIDIA megakernel: wqkv_a on the low-latency GEMM at %d rows in %d of %d "
            "layers (SGLang's in %s: no MXFP8 input, slot attn_qkv_a_bf16 off)",
            rows,
            len(seen) - len(fell_back),
            len(seen),
            fell_back or "none",
        )


QKV_A_ENGAGEMENT = _Engagement()


class Mxfp8Rows(NamedTuple):
    """wq_b's input from the chain's norm: e4m3 codes [rows, K] and UE8M0 scales, in
    FlashInfer's 128x4 swizzle up to 8 rows and per-8-row tiles beyond, the two
    layouts the low-latency GEMM reads. Only ``ChainOps.compute_q_b`` consumes it."""

    quant: torch.Tensor
    scale: torch.Tensor


def qkv_a_ll_applies(attn) -> bool:
    """wqkv_a runs on the low-latency GEMM at 1-8 rows, on the boundary's MXFP8 input."""
    return (
        attn.fuse_wqa_wkv
        and attn.wqkv_a.bias is None
        and _ll_linear_applies(attn.wqkv_a)
    )


def index_q_mxfp8_applies(attn) -> bool:
    """The layer's indexer wq_b takes a pre-quantized ``Mxfp8SwizzledInput`` of q_lora,
    as SGLang's MXFP8 linear decides it (an MXFP8 view on FlashInfer's CUTLASS / CuTe
    DSL backend), so it can read the chain's quant instead of quantizing again."""
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    indexer = getattr(attn, "indexer", None)
    if indexer is None:
        return False
    linear, method = indexer.wq_b, indexer.wq_b.quant_method
    if not isinstance(method, Fp8LinearMethod):
        return False
    backend = getattr(method, "mxfp8_dense_backend", None)
    return (
        (
            getattr(method, "use_mxfp8", False)
            or (
                getattr(method, "block_fp8_as_mxfp8", False)
                and getattr(linear, "block_fp8_mxfp8_ready", False)
            )
        )
        and backend is not None
        and (backend.is_flashinfer_cutlass() or backend.is_flashinfer_cutedsl())
        and linear.weight.shape[-1] == attn.q_lora_rank
    )


def wq_b_consumer(attn):
    """The low-latency GEMM's tactic table for this wq_b: attention TP8's 8 heads have
    their own (``mxfp8_ll.CONSUMER_TACTICS``)."""
    return mxfp8_ll.WQ_B_TP8 if attn.n_local_heads == 8 else None


def wq_b_ll_applies(attn) -> bool:
    """wq_b runs on the low-latency GEMM, fed by the chain's PDL norm + quant."""
    return _ll_linear_applies(attn.wq_b) and attn.q_norm.weight.dtype == torch.bfloat16


class ChainOps:
    """``ops`` with the chain's query kernels; the prepare paths call either. Which
    kernels run is fixed at build (``wq_b_ll``, ``rope_in_attention``)."""

    def __init__(
        self,
        attn,
        *,
        rope_in_attention: bool,
        qkv_a_ll: bool = False,
        qkv_a_bf16: bool = False,
        index_q_mxfp8_in: bool = False,
    ):
        self.rope_in_attention = rope_in_attention
        self.wq_b_ll = wq_b_ll_applies(attn)
        # slot attn_indexer: the process's plugin indexer (attention/indexer.py), set by
        # the attention build for the layers it takes
        self.indexer = None
        self.qkv_a_ll = qkv_a_ll and qkv_a_ll_applies(attn)
        # slot attn_qkv_a_bf16: BF16 input (no MXFP8 rows from the boundary) is
        # quantized for the low-latency GEMM too, with FlashInfer's MXFP8 quantize
        self.qkv_a_bf16 = self.qkv_a_ll and qkv_a_bf16
        # slot "index_q_mxfp8_in": the indexer wq_b reads the chain's q_lora quant
        self.index_q_mxfp8_in = (
            index_q_mxfp8_in and self.wq_b_ll and index_q_mxfp8_applies(attn)
        )
        self.kv_to_cache = ops.kv_to_cache

    def index_q_input(self, attn, q_lora, q_for_wqb):
        """The indexer wq_b's input: at 1-8 rows the chain's 128x4 MXFP8 q_lora rows,
        which SGLang's linear takes without quantizing again; q_lora otherwise (the
        row-tile scales past 8 rows are not a layout it reads)."""
        from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

        if (
            self.index_q_mxfp8_in
            and isinstance(q_for_wqb, Mxfp8Rows)
            and q_lora.shape[0] <= mxfp8_ll.MAX_ROWS
        ):
            return Mxfp8SwizzledInput(q_for_wqb.quant, q_for_wqb.scale)
        return q_lora

    def qkv_a(self, attn, x):
        """wqkv_a on the low-latency GEMM for the boundary's MXFP8 rows (1-8 rows, the
        plain 128x4 scales it reads) or, with ``qkv_a_bf16``, for BF16 rows quantized
        here; SGLang's call otherwise."""
        from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

        # hop B's row-tile rows past 8 rows: the wide low-latency GEMM (only for a chain
        # with qkv_a_ll, which the boundary checks before writing them)
        if self.qkv_a_ll and isinstance(x, mxfp8_ll.RowTileMxfp8):
            QKV_A_ENGAGEMENT.note(attn.layer_id, x.quant.shape[0], True)
            w = attn.wqkv_a
            return mxfp8_ll.linear(
                x.quant, x.scale, w.weight, w.weight_scale_inv_swizzled
            )
        # SGLang's forward may hand a plain (codes, scales) tuple: still SGLang's call
        rows = x[0].shape[0] if isinstance(x, tuple) else x.shape[0]
        fits = self.qkv_a_ll and 0 < rows <= mxfp8_ll.MAX_ROWS
        if fits and isinstance(x, Mxfp8SwizzledInput):
            codes, scales = x.data, x.scales
        elif (
            fits
            and self.qkv_a_bf16
            and isinstance(x, torch.Tensor)
            and x.dtype == torch.bfloat16
            # FlashInfer's quantize reads contiguous rows
            and x.is_contiguous()
        ):
            codes, scales = mxfp8_ll.quantize(x)
        else:
            codes = None
        if self.qkv_a_ll and rows <= mxfp8_ll.MAX_ROWS:
            QKV_A_ENGAGEMENT.note(attn.layer_id, rows, codes is not None)
        if codes is None:
            return ops.qkv_a(attn, x)
        w = attn.wqkv_a
        return mxfp8_ll.linear(
            codes, scales.reshape(-1), w.weight, w.weight_scale_inv_swizzled
        )

    def normalize_q_lora(self, attn, q):
        if not self.wq_b_ll:
            return ops.normalize_q_lora(attn, q)
        y, quant, scale = rmsnorm_mxfp8(
            q,
            attn.q_norm.weight,
            attn.q_norm.variance_epsilon,
            pdl=True,
            row_tiles=q.shape[0] > mxfp8_ll.MAX_ROWS,
        )
        return y, Mxfp8Rows(quant, scale)

    def compute_q_a(self, attn, x, qkv_a):
        q = qkv_a[..., : attn.q_lora_rank] if qkv_a is not None else attn.wq_a(x)[0]
        return self.normalize_q_lora(attn, q)

    def compute_q_b(self, attn, q, positions, q_out):
        from sglang.kernels.ops.attention.dsv4 import fused_q_norm_rope

        if self.wq_b_ll:
            out = mxfp8_ll.linear(
                q.quant,
                q.scale,
                attn.wq_b.weight,
                attn.wq_b.weight_scale_inv_swizzled,
                consumer=wq_b_consumer(attn),
            )
        elif self.rope_in_attention:
            out = attn.wq_b(q)[0]
        else:
            return ops.compute_q_b(attn, q, positions, q_out)
        out = out.view(-1, attn.n_local_heads, attn.head_dim)
        if self.rope_in_attention:
            return out
        if q_out is None:
            q_out = torch.empty_like(out)
        fused_q_norm_rope(out, q_out, None, attn.freqs_cis, positions)
        return q_out


def resolve_caches(attn, view, rows: int):
    """The window and compressed KV-cache views and their sparse indices, as the
    backend's ``_forward_attention`` resolves them for a decode or verify step."""
    from sglang.srt.layers.attention.deepseek_v4_backend import _pad_tensor_to_size

    meta = view.metadata.core_attn_metadata
    pool = view.kv_pool
    layer_id, ratio = attn.layer_id, attn.compress_ratio
    swa = pool.get_swa_key_buffer_radix(layer_id)
    page, width = pool.get_swa_key_page_size(), pool.get_swa_key_bytes_per_token()
    swa = swa[:, : page * width].view(swa.shape[0], page, 1, width)
    extra, extra_indices, extra_lengths = None, None, None
    if ratio != 0:
        extra = pool.get_extra_key_buffer(layer_id)
        page = pool.get_extra_key_page_size(layer_id)
        width = pool.get_extra_key_bytes_per_token(layer_id)
        extra = extra[:, : page * width].view(extra.shape[0], page, 1, width)
        extra_indices = meta.sparse_page_indices(ratio)
        extra_lengths = meta.sparse_topk_lengths(ratio)

    def match(x, value):
        if x is None or x.shape[0] == rows:
            return x
        if x.shape[0] > rows:
            return x[:rows]
        return _pad_tensor_to_size(x, rows, value=value)

    swa_indices = match(meta.swa_page_indices, 0)
    if swa_indices.ndim == 2:
        swa_indices = swa_indices.unsqueeze(1)
    extra_indices = match(extra_indices, -1)
    if extra_indices is not None and extra_indices.ndim == 2:
        extra_indices = extra_indices.unsqueeze(1)
    return (
        swa,
        swa_indices,
        match(meta.swa_topk_lengths, 1),
        extra,
        extra_indices,
        match(extra_lengths, 1),
    )


# slot kv_before_wait's engagement: swap-AB calls captured with the KV gathered before
# the PDL wait, by (model, rows); the first of each is logged
kv_before_wait_captures: dict[tuple[str, int], int] = {}


def _note_kv_before_wait(model: str, rows: int, layer_id: int) -> None:
    if not torch.cuda.is_current_stream_capturing():
        return
    key = (model, rows)
    kv_before_wait_captures[key] = kv_before_wait_captures.get(key, 0) + 1
    if kv_before_wait_captures[key] == 1:
        logger.info(
            "NVIDIA megakernel: swap-AB gathers the KV before its PDL wait: %s at %d "
            "rows (first layer %d)",
            model,
            rows,
            layer_id,
        )


def sparse_decode(
    attn,
    view,
    q,
    positions,
    freqs,
    *,
    kv_before_wait=False,
    extra_after_wait=False,
    model="target",
):
    """Swap-AB on the 16 (or 8) local heads in place of the backend's swap-AB /
    FlashMLA choice; ropes ``q`` (unroped, [rows, heads, 512]) as it loads it. ``kv_before_wait``:
    gather the KV before the PDL wait, only where every KV and index writer joined
    the step's stream before it (the low-ratio prepare); ``extra_after_wait``: the
    sparse indices only after it (the layer's own selection is the predecessor)."""
    swa, swa_indices, swa_lengths, extra, extra_indices, extra_lengths = resolve_caches(
        attn, view, q.shape[0]
    )
    if kv_before_wait:
        _note_kv_before_wait(model, q.shape[0], attn.layer_id)
    return swapab.swapab_attention(
        q.unsqueeze(1),
        swa,
        swa_indices,
        swa_lengths,
        attn._local_attn_sink(attn.n_local_heads),
        extra,
        extra_indices,
        extra_lengths,
        max_rows=SWAPAB_MAX_ROWS,
        pdl=True,
        narrow_combine=True,
        q_rope=(freqs, positions),
        kv_before_wait=kv_before_wait,
        combine_entry_trigger=True,
        extra_after_wait=extra_after_wait,
    )


def wo_a(attn, o, positions, fb, row_tiles: bool = False, *, fused=None):
    """The fused inverse RoPE + wo_a + MXFP8 with the early trigger, where SGLang's
    own step fuses them (``fused``: the build's ``fused_wo_a_applies``, default
    SGLang's choice); SGLang's arm otherwise. ``row_tiles`` (past 8 rows, for hop A's
    publishing wo_b only) writes the scales per 8-row tile: a ``RowTileMxfp8``."""
    from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput
    from sglang.srt.runtime_context import get_forward

    if fused is None:
        fused = attn.use_fused_wo_a
    if not (
        fused
        and attn.use_flashinfer_mxfp8_wo_b
        and not get_forward().sp_active
        and 0 < o.shape[0] <= WO_A_MAX_ROWS
    ):
        return ops.wo_a(attn, o, positions, fb)
    o = o.view(o.shape[0], attn.n_local_groups, -1)
    codes, scales = wo_a_kernel.fused_rope_wo_a(
        o,
        attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1),
        torch.view_as_real(attn.freqs_cis).flatten(1),
        positions,
        early_trigger=True,
        row_tile_scales=row_tiles,
    )
    if row_tiles:
        return mxfp8_ll.RowTileMxfp8(codes, scales)
    return Mxfp8SwizzledInput(codes, scales)


def precompile(
    attn, widths, *, qkv_a_ll: bool = False, wo_a_one_group: bool = False
) -> None:
    """Compile the chain's kernels that do not depend on the KV pool, for these widths:
    the low-latency wq_b GEMM, the fused wo_a, and the norm + quant (its row count and
    input row stride are compile-time constants). Called at build, before capture.
    The swap-AB kernels specialize on the pool's page geometry and the step's index
    widths, which exist only after KV sizing: they compile in SGLang's warm-up runs
    before graph capture."""
    from sgl_nvidia_mega.deepseek_v41.runtime import SLOTS

    if qkv_a_ll and qkv_a_ll_applies(attn):
        ll_rows = [w for w in widths if w <= mxfp8_ll.MAX_ROWS]
        if "hop_b_wide" in SLOTS:
            # hop B's row-tile rows past 8 rows (plan.HOP_B_MAX_ROWS)
            ll_rows += [
                w for w in widths if mxfp8_ll.MAX_ROWS < w <= slot_plan.HOP_B_MAX_ROWS
            ]
        if ll_rows:
            mxfp8_ll.precompile(attn.wqkv_a.weight, ll_rows)
    rows = [w for w in widths if w <= NORM_QUANT_MAX_ROWS]
    if not rows:
        return
    if wq_b_ll_applies(attn):
        mxfp8_ll.precompile(attn.wq_b.weight, rows, wq_b_consumer(attn))
        k = attn.q_lora_rank
        # q_lora as the prepare paths slice it: a view of wqkv_a's output when fused
        width = attn.wqkv_a.weight.shape[0] if attn.fuse_wqa_wkv else k
        x = torch.zeros(
            max(rows), width, dtype=torch.bfloat16, device=attn.q_norm.weight.device
        )
        for r in rows:
            rmsnorm_mxfp8(
                x[:r, :k],
                attn.q_norm.weight,
                attn.q_norm.variance_epsilon,
                pdl=True,
                row_tiles=r > mxfp8_ll.MAX_ROWS,
            )
    if fused_wo_a_applies(attn, wo_a_one_group):
        groups = attn.n_local_groups
        wo_a_kernel.precompile(min(max(rows), WO_A_MAX_ROWS), groups=groups)
        # hop A's publishing wo_b past 8 rows reads row-tile scales
        tiled = [w for w in rows if mxfp8_ll.MAX_ROWS < w <= slot_plan.HOP_A_MAX_ROWS]
        if tiled and {"hop_a", "hop_a_batch2"} <= SLOTS:
            wo_a_kernel.precompile(max(tiled), row_tile_scales=True, groups=groups)
