# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang's python/sglang/kernels/ops/attention/dsv4/decode_attention_sm100.py (Apache-2.0).
"""SM100 small-batch paged attention with the heads on the MMA N dimension; the
caller applies the output's inverse RoPE (the fused wo_a does)."""

import torch
import triton
import triton.language as tl

from .kv_layout import KVLayout

LAYOUT = KVLayout.V4
MAX_BATCH = 8
# Each (row, KV tile) is its own CTA, so rows past MAX_BATCH are a caller's dispatch choice.
WIDE_MAX_BATCH = 64
NUM_HEADS = 16  # the MMA's head columns
# Local query heads served: 16 at attention TP4, 8 at TP8 (padded to 16 in the MMA)
HEAD_COUNTS = (8, 16)
HEAD_DIM = 512
# With pdl, trigger the next kernel (wo_a_fused streams its weights before its wait) right
# after the wait, not after the loads: faster in 7 of 9 GB300 micro-bench cases at 6-24 rows.
COMBINE_EARLY_TRIGGER = True
SOFTMAX_SCALE = HEAD_DIM**-0.5


@triton.jit
def _combine(
    PART,
    MAX,
    SUM,
    SINK,
    OUT,
    NT: tl.constexpr,
    ST: tl.constexpr,
    H: tl.constexpr,
    BD: tl.constexpr,
    USE_GDC: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr,
    ENTRY_TRIGGER: tl.constexpr = False,
):
    b, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t, d = tl.arange(0, ST), tile * BD + tl.arange(0, BD)
    if USE_GDC:
        if ENTRY_TRIGGER:
            # The consumer reads OUT only after its own grid wait; launching it now lets
            # it stream its weights while the partial kernel still runs.
            tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
        if EARLY_TRIGGER and not ENTRY_TRIGGER:
            tl.extra.cuda.gdc_launch_dependents()
    den = tl.load(SUM + (b * NT + t) * H + h, t < NT, 0)
    mx = tl.load(MAX + (b * NT + t) * H + h, t < NT, 0)
    mx = tl.where(den > 0, mx, -float("inf"))
    sink = tl.load(SINK + h)
    m = tl.maximum(tl.max(mx, 0), sink)
    m = tl.where(tl.abs(m) == float("inf"), 0.0, m)
    factor = tl.exp(mx - m)
    denominator = tl.sum(den * factor, 0) + tl.exp(sink - m)
    vals = tl.load(
        PART + ((b * NT + t[:, None]) * H + h) * 512 + d[None, :],
        t[:, None] < NT,
        0,
    )
    if USE_GDC and not EARLY_TRIGGER and not ENTRY_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()
    out = tl.sum(vals * factor[:, None], 0) / denominator
    out = tl.where((denominator > 0) & (sink != float("inf")), out, 0.0)
    out = out.to(OUT.dtype.element_ty)
    tl.store(OUT + (b * H + h) * 512 + d, out)


def _q_rope_kwargs(q_rope, b):
    freqs, positions = q_rope
    assert (
        freqs.dtype == torch.float32 and freqs.shape[-1] == 64 and freqs.is_contiguous()
    )
    assert positions.shape == (b,)
    return {"FREQS": freqs, "POS": positions, "ROPE_Q": True}


def swapab_attention(
    q,
    kv,
    indices,
    lengths,
    sink,
    extra_kv=None,
    extra_indices=None,
    extra_lengths=None,
    *,
    max_rows=MAX_BATCH,
    pdl=False,
    narrow_combine=False,
    q_rope=None,
    kv_before_wait=False,
    combine_entry_trigger=False,
):
    """V4-layout attention on 16 (TP4) or 8 (TP8) heads; `extra_*` is a second slot range appended
    to each request's keys, and the attention sink is folded in exactly once.
    `pdl` launches both kernels as programmatic dependents; `narrow_combine` runs the
    combine as one-warp programs over 128 columns. `q_rope` = (fp32 [pos, 64] freqs,
    positions [b]) applies the query RoPE as Q loads. `kv_before_wait` gathers the KV
    rows before the PDL wait: the caller guarantees that only Q is still in flight.
    `combine_entry_trigger` launches the combine's dependent at the combine's entry."""
    from .swapab_gluon import partial_gluon  # Gluon imports only where it runs

    block = 64
    b, h, d = q.shape[0], q.shape[-2], q.shape[-1]
    assert q.ndim in (3, 4) and (q.ndim == 3 or q.shape[1] == 1)
    assert 0 < b <= min(max_rows, WIDE_MAX_BATCH) and h in HEAD_COUNTS and d == HEAD_DIM
    assert q.dtype == torch.bfloat16 and q.stride(-1) == 1
    assert kv.shape[-1] == LAYOUT.bytes_per_token
    assert kv.dtype in (torch.uint8, torch.float8_e4m3fn)
    assert indices.stride(-1) == 1 and lengths.is_contiguous()
    assert sink.stride(0) == 1 and sink.numel() >= h
    nk = indices.shape[-1]
    ne = 0 if extra_indices is None else extra_indices.shape[-1]
    assert block in (32, 64, 128) and nk > 0
    if extra_kv is None:
        assert extra_indices is None and extra_lengths is None
        extra_kv, extra_indices, extra_lengths = kv, indices, lengths
    else:
        assert extra_kv.shape[-1] == LAYOUT.bytes_per_token
        assert extra_kv.dtype in (torch.uint8, torch.float8_e4m3fn)
        assert extra_indices.stride(-1) == 1 and extra_lengths.is_contiguous()
    kv, extra_kv = kv.view(torch.uint8), extra_kv.view(torch.uint8)
    kt = triton.cdiv(nk, block)
    nt = kt + triton.cdiv(ne, block)
    partial = torch.empty((b, nt, h, 512), dtype=torch.float32, device=q.device)
    maximum = torch.empty((b, nt, h), dtype=torch.float32, device=q.device)
    sums = torch.empty_like(maximum)
    out = torch.empty((b, h, 512), dtype=q.dtype, device=q.device)
    partial_gluon[(b, nt)](
        q,
        kv,
        extra_kv,
        indices,
        extra_indices,
        lengths,
        extra_lengths,
        partial,
        maximum,
        sums,
        QS=q.stride(0),
        QH=q.stride(-2),
        IS=indices.stride(0),
        EIS=extra_indices.stride(0),
        KP=kv.shape[1],
        KS=kv.stride(0),
        EP=extra_kv.shape[1],
        ES=extra_kv.stride(0),
        NK=nk,
        NE=ne,
        NT=nt,
        KT=kt,
        BT=block,
        H=NUM_HEADS,
        SCALE=SOFTMAX_SCALE,
        KTOKENS=kv.shape[0] * kv.shape[1],
        ETOKENS=extra_kv.shape[0] * extra_kv.shape[1],
        COMPENSATE=True,
        SWAP_AB=True,
        DATA_BYTES=LAYOUT.data_bytes,
        SCALE_BYTES=LAYOUT.scale_bytes,
        TILE=LAYOUT.tile_size,
        num_warps=4,
        **({"USE_GDC": True, "launch_pdl": True} if pdl else {}),
        **(_q_rope_kwargs(q_rope, b) if q_rope is not None else {}),
        **({"KV_BEFORE_WAIT": True} if pdl and kv_before_wait else {}),
        **({"HQ": h} if h != NUM_HEADS else {}),
    )
    bd = 64 if ne else 512
    combine_warps = 4
    if narrow_combine and ne:
        # GB300, 10 KV tiles: 2.1 vs 3.1 us at 12 rows, 2.5 vs 4.6 at 24 (BD 64 / 4 warps).
        bd, combine_warps = 128, 1
    _combine[(b, h, triton.cdiv(512, bd))](
        partial,
        maximum,
        sums,
        sink,
        out,
        NT=nt,
        ST=triton.next_power_of_2(nt),
        H=h,
        BD=bd,
        USE_GDC=pdl,
        EARLY_TRIGGER=COMBINE_EARLY_TRIGGER,
        **({"ENTRY_TRIGGER": True} if pdl and combine_entry_trigger else {}),
        num_warps=combine_warps,
        **({"launch_pdl": True} if pdl else {}),
    )
    return out
