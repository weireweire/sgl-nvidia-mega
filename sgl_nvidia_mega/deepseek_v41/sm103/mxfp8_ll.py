# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FlashInfer's ``cutedsl_low_latency`` MXFP8 GEMM as the publishing wo_b uses it:
whether this GPU has it, and the GB300 tactics pinned for DeepSeek-V4.1-Flash's TP4
attention projections. :func:`linear` launches the compiled kernel directly for wqkv_a
and wq_b at 1-64 rows, past the 8 rows FlashInfer's runner takes."""

from __future__ import annotations

import contextlib
import importlib
import logging
from functools import lru_cache
from typing import NamedTuple, Optional

import torch

logger = logging.getLogger(__name__)

BACKEND = "cutedsl_low_latency"
# FlashInfer's cutedsl_low_latency mm_mxfp8 takes at most 8 rows and K % 128 == 0.
MAX_ROWS = 8
# (cta_k, ab_stages, sfb_tmem_stages, split_k) per (SM, N, K) for 5-8 rows: the TP4 wqkv_a,
# wq_b, wo_b of DSV4.1-Flash on GB300, picked jointly in a 40-layer chain; the indexer wq_b.
PINNED_TACTICS: dict[tuple[int, int, int], tuple[int, int, int, int]] = {
    (103, 1792, 5120): (128, 10, 1, 4),
    (103, 8192, 1280): (256, 6, 1, 1),
    (103, 5120, 2048): (256, 4, 1, 2),
    # the indexer wq_b (on the low-latency GEMM with slot index_q_mxfp8_in): best of 80
    # at 6 and 8 rows, 2.3 us
    (103, 4096, 1280): (128, 4, 1, 4),
}
_GEMM_MODULE = "flashinfer.gemm.kernels.cute_dsl.low_latency_blockscaled_gemm"


@lru_cache(maxsize=1)
def supported() -> bool:
    try:
        import torch
        from flashinfer import mm_mxfp8

        major, minor = torch.cuda.get_device_capability()
        return bool(mm_mxfp8.is_backend_supported(BACKEND, major * 10 + minor))
    except Exception:
        return False


def quantize(x):
    """BF16 rows as MXFP8 with swizzled 128x4 UE8M0 scales, as SGLang's MXFP8 linears
    quantize their input."""
    from flashinfer import SfLayout, mxfp8_quantize

    return mxfp8_quantize(
        x,
        is_sf_swizzled_layout=True,
        alignment=32,
        backend="cute-dsl",
        sf_swizzle_layout=SfLayout.layout_128x4,
    )


def pinned_tactic(
    *, sm_version: int, n: int, k: int, rows: int
) -> Optional[tuple[int, int, int, int]]:
    # 5-8 rows is FlashInfer's tuning bucket 8 (buckets round up to a power of two).
    if not MAX_ROWS // 2 < rows <= MAX_ROWS:
        return None
    return PINNED_TACTICS.get((sm_version, n, k))


@contextlib.contextmanager
def pinned_tactics(sm_version: int):
    """While tuning, FlashInfer's low-latency tactic list is the pinned tactic alone where
    one applies, so the autotuner caches it under its usual key on every rank."""
    ll_gemm = importlib.import_module(_GEMM_MODULE)
    enumerate_tactics = ll_gemm.autotune_tactics

    def autotune_tactics(problem_sizes_mnkl, *args, **kwargs):
        tactics = enumerate_tactics(problem_sizes_mnkl, *args, **kwargs)
        # The runner computes out.T, so the problem is (N, M, K, L).
        n, rows, k, _ = problem_sizes_mnkl
        pinned = pinned_tactic(sm_version=sm_version, n=n, k=k, rows=rows)
        if pinned is None:
            return tactics
        if pinned not in tactics:
            logger.warning(
                f"NVIDIA megakernel: pinned low-latency MXFP8 tactic {pinned} is not "
                f"offered for N={n} K={k}; autotuning that shape instead."
            )
            return tactics
        return [pinned]

    ll_gemm.autotune_tactics = autotune_tactics
    try:
        yield
    finally:
        ll_gemm.autotune_tactics = enumerate_tactics


# --- wqkv_a / wq_b launched directly, 1-64 rows ------------------------------

Tactic = tuple[int, int, int, int]
# Past 8 rows the kernel runs one CTA per 8-row tile and reads each tile's activation
# scales from a 128-row atom of its own (q_norm_quant's ``row_tiles`` layout); at 1-8
# rows that layout is FlashInfer's plain 128x4 swizzle. FlashInfer's runner refuses
# more than 8 rows (its valid tactics are empty there); the kernel does not.
WIDE_MAX_ROWS = 64


class RowTileMxfp8(NamedTuple):
    """MXFP8 rows [M, K] whose UE8M0 scales are in the row-tile layout above, as a
    producer writes them for this kernel past 8 rows (SGLang's linears cannot read it)."""

    quant: torch.Tensor
    scale: torch.Tensor


# Tactics per (SM, N, K, row bucket), ahead of PINNED_TACTICS (bucket 8); a bucket is
# the row count rounded up to a power of two.
# 1 / 2 / 4: the GB300 table chain-picked for SGLang's autotune
# (flashinfer_autotune_configs/NVIDIA_GB300.json, SGLang 26fd68fcd34, FlashInfer 0.7.2,
# whose low-latency kernel source equals 0.7.0.post1's). BEST tunes these at boot.
# 16 / 32 / 64: BEST's row-tile pins (best of 92 / 80 at 12 / 24 / 48 rows, 16 weights
# streaming from DRAM).
# wqkv_a's 16 / 32 / 64 rows are unused while slot attn_qkv_a serves 1-8 rows only.
BUCKET_TACTICS: dict[tuple[int, int, int, int], Tactic] = {
    (103, 1792, 5120, 1): (128, 8, 1, 4),
    (103, 1792, 5120, 2): (128, 9, 1, 2),
    (103, 1792, 5120, 4): (256, 1, 1, 8),
    (103, 1792, 5120, 16): (128, 10, 1, 4),
    (103, 1792, 5120, 32): (128, 11, 1, 2),
    (103, 1792, 5120, 64): (128, 6, 1, 2),
    (103, 8192, 1280, 1): (128, 8, 1, 1),
    (103, 8192, 1280, 2): (128, 8, 1, 1),
    (103, 8192, 1280, 4): (128, 6, 1, 1),
    (103, 8192, 1280, 16): (128, 6, 1, 1),
    (103, 8192, 1280, 32): (128, 5, 1, 1),
    (103, 8192, 1280, 64): (128, 4, 1, 1),
    # the shared expert's gate_up (slot shared_ll): best of 84-92 at 6 / 12 / 24 / 48
    # rows in a cold 40-layer chain; 1 / 2 / 4 take FlashInfer's default
    (103, 1152, 5120, 8): (128, 8, 1, 8),
    (103, 1152, 5120, 16): (128, 12, 1, 4),
    (103, 1152, 5120, 32): (128, 8, 1, 4),
    (103, 1152, 5120, 64): (128, 9, 1, 2),
}
# Attention TP8's wq_b (N 4096, K 1280): the indexer wq_b's shape, whose bucket-8 pin
# above it keeps (TP4 serves it); the chain asks for these by consumer instead. Best of
# 80 per row count in a 40-layer chain on GB300 (2.21 vs 2.51 us at 6 rows, 2.22 vs
# 2.69 at 1-4, 2.53 vs 2.77 at 12-16; benchmark/sweep_deepseek_v41_tactics.py).
WQ_B_TP8 = "wq_b_tp8"
CONSUMER_TACTICS: dict[tuple[str, int, int, int, int], Tactic] = {
    (WQ_B_TP8, 103, 4096, 1280, 1): (128, 10, 1, 2),
    (WQ_B_TP8, 103, 4096, 1280, 2): (128, 10, 1, 2),
    (WQ_B_TP8, 103, 4096, 1280, 4): (128, 10, 1, 2),
    (WQ_B_TP8, 103, 4096, 1280, 8): (128, 10, 1, 2),
    (WQ_B_TP8, 103, 4096, 1280, 16): (256, 6, 1, 1),
}
_COMPILED: dict[tuple[int, Tactic], object] = {}


def row_bucket(rows: int) -> int:
    return 1 << (rows - 1).bit_length()


def scale_bytes(rows: int, k: int) -> int:
    """Activation scale bytes the kernel reads: one 128-row x (K / 32) atom per 8 rows."""
    return -(-rows // 8) * (k // 128) * 512


def offered_tactics(n: int, k: int, rows: int) -> list[Tactic]:
    """FlashInfer's tactic list for an MXFP8 (N, K) problem, in its default order."""
    import cutlass

    ll_gemm = importlib.import_module(_GEMM_MODULE)
    e4m3, e8m0 = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU
    return [
        tuple(t)
        for t in ll_gemm.autotune_tactics(
            (n, rows, k, 1), e4m3, e4m3, e8m0, 32, cutlass.BFloat16
        )
    ]


@lru_cache(maxsize=None)
def select_tactic(
    sm_version: int, n: int, k: int, bucket: int, consumer: Optional[str] = None
) -> Tactic:
    """The pinned tactic of this row bucket if FlashInfer offers it, else FlashInfer's
    default (its first tactic): fixed per bucket, never tuned per process. A
    ``consumer`` (``CONSUMER_TACTICS``) takes its own pin first."""
    pinned = CONSUMER_TACTICS.get((consumer, sm_version, n, k, bucket))
    if pinned is None:
        pinned = BUCKET_TACTICS.get((sm_version, n, k, bucket))
    if pinned is None and bucket == MAX_ROWS:
        pinned = PINNED_TACTICS.get((sm_version, n, k))
    offered = offered_tactics(n, k, min(bucket, MAX_ROWS))
    if pinned in offered:
        return pinned
    if pinned is not None:
        logger.warning(f"Low-latency MXFP8 tactic {pinned} not offered for N={n} K={k}")
    return offered[0]


def _compiled(tactic: Tactic, device: torch.device):
    key = (device.index, tactic)
    if key not in _COMPILED:
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_fake_stream

        ll_gemm = importlib.import_module(_GEMM_MODULE)
        cta_k, ab_stages, sfb_tmem_stages, split_k = tactic
        gemm = ll_gemm.LowLatencyBlockscaledGemmKernel(
            acc_dtype=cutlass.Float32,
            mma_tiler_mnk=(128, 8, cta_k),
            num_ab_stage=ab_stages,
            num_sfb_tmem_stage=sfb_tmem_stages,
            sf_vec_size=32,
            use_pdl=True,
            split_k=split_k,
        )
        e4m3, e8m0 = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU
        with torch.accelerator.device_index(device.index):
            _COMPILED[key] = cute.compile(
                gemm,
                _ptr(e4m3, 16, 16),
                _ptr(e8m0, 32, 32),
                _ptr(e4m3, 16, 16),
                _ptr(e8m0, 16, 16),
                _ptr(cutlass.BFloat16, 16, 16),
                None,
                None,
                tuple(cutlass.Int32(v) for v in (128, 8, cta_k, 1)),
                make_fake_stream(),
            )
    return _COMPILED[key]


def _ptr(dtype, address: int, align: int):
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)


def precompile(
    weight: torch.Tensor, rows: list[int], consumer: Optional[str] = None
) -> None:
    """Compile the kernels these row counts select; call before CUDA graph capture."""
    n, k = weight.shape
    major, minor = torch.cuda.get_device_capability(weight.device)
    for r in rows:
        tactic = select_tactic(major * 10 + minor, n, k, row_bucket(r), consumer)
        _compiled(tactic, weight.device)


def linear(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    tactic: Optional[Tactic] = None,
    consumer: Optional[str] = None,
) -> torch.Tensor:
    """``a @ weight.T`` in BF16 for MXFP8 rows ``a`` [M, K] (1 <= M <= 64) with their
    UE8M0 scales (FlashInfer's 128x4 swizzle up to 8 rows, the row-tile layout above),
    and an MXFP8 weight [N, K] with its swizzled scales as SGLang loads them
    (``weight_scale_inv_swizzled``). The tactic is the row bucket's (:func:`select_tactic`)."""
    from cutlass import BFloat16, Float8E4M3FN, Float8E8M0FNU, Int32

    from sgl_nvidia_mega.common.sync import current_cu_stream

    rows, k = a.shape
    n = weight.shape[0]
    assert 0 < rows <= WIDE_MAX_ROWS and k % 128 == 0 and weight.shape[1] == k
    assert a.is_contiguous() and weight.is_contiguous()
    assert a_scale.numel() >= scale_bytes(rows, k)
    if tactic is None:
        major, minor = torch.cuda.get_device_capability(a.device)
        tactic = select_tactic(major * 10 + minor, n, k, row_bucket(rows), consumer)
    if out is None:
        out = torch.empty((rows, n), dtype=torch.bfloat16, device=a.device)
    assert out.shape == (rows, n) and out.is_contiguous()
    _compiled(tuple(tactic), a.device)(
        _ptr(Float8E4M3FN, weight.data_ptr(), 16),
        _ptr(Float8E8M0FNU, weight_scale.data_ptr(), 32),
        _ptr(Float8E4M3FN, a.data_ptr(), 16),
        _ptr(Float8E8M0FNU, a_scale.data_ptr(), 16),
        _ptr(BFloat16, out.data_ptr(), 16),
        None,
        None,
        (Int32(n), Int32(rows), Int32(k), Int32(1)),
        current_cu_stream(),
    )
    return out
