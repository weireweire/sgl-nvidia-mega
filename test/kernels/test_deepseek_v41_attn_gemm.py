# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DeepSeek-V4.1 attention kernels on one SM100-family GPU (GB300), against SGLang's.

- wqkv_a (N 1792, K 5120) on FlashInfer's low-latency MXFP8 GEMM against SGLang's MXFP8
  linear (``mm_mxfp8`` ``cute-dsl``), through SGLang's KV store kernel into both window
  cache layouts: the KV bytes that change, per byte and dequantized, within one step of
  each block's coarsest quantum (synthetic weights, and the checkpoint's layers when
  ``SGL_NVIDIA_MEGA_DSV41_CKPT`` names it; ``KV_TOLERANCE`` lines with ``pytest -s``).
- The draft's low-latency wo_b on the fused wo_a's MXFP8 rows against SGLang's MXFP8
  linear at 1-8 rows (slot draft_wo_b_ll).
- The fused inverse RoPE + wo_a + MXFP8 kernel (src/wo_a_fused.cu) against SGLang's JIT
  build of the same device code, bit for bit, at representative row counts, int32 and
  int64 positions, the 64-head row stride and a packed one.
- The fused wo_a on attention TP8's one local group (slot wo_a_g1): bit for bit the
  first group of the two-group kernel (which is SGLang's) at every row tile, both scale
  layouts, int32 / int64 positions, the 8-head output viewed packed and padded; and
  within one MXFP8 step of SGLang's TP8 arm (inverse RoPE + BF16 einsum, quantized as
  wo_b's input).
- Swap-AB sparse attention with the query RoPE inside it and the KV-before-wait
  low-ratio path (PDL, narrow combine, combine entry trigger): bit for bit against
  SGLang's q RoPE kernel followed by the plain launch, eager and replayed. At attention
  TP8's 8 local heads (slot swapab_h8): bit for bit the 16-head kernel's first 8 heads.
- The FlashMLA schedules the step head builds ahead (slot "sched_ahead"): bit for bit
  SGLang's lazy per-layer schedule (``_maybe_precompute_flashmla_sched_meta``) for every
  ratio, at a small and a large verify width, with padded lengths, eager and replayed.
- wq_b on poisoned memory is in test_deepseek_v41_0_poisoned_scales.py.
"""

import inspect
import json
import math
import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.attention import sched_ahead
from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll
from sgl_nvidia_mega.deepseek_v41.sm103.attention import swapab, wo_a

try:
    from sglang.srt.layers.attention import deepseek_v4_backend as backend
except ImportError:
    backend = None

# -------------------------------------------------------- low-latency MXFP8 projections

SHAPES = {
    "wqkv_a": (1792, 5120),
    "wq_b": (8192, 1280),
    "wo_b": (5120, 2048),
    # the indexer's wq_b (slot index_q_mxfp8_in), fed the same q-LoRA MXFP8 rows as wq_b
    "index_wq_b": (4096, 1280),
}
Q_LORA, HEAD_DIM, ROPE_DIM = 1280, 512, 64
EPS = 1e-20  # text_config.rms_norm_eps (kv_norm)
PAGE = 256  # SGLang's SWA page on SM100
MAX_POS = 1 << 16
KV_ROWS = (1, 5, 6, 8, 12, 24, 48)
SEEDS = 24
CKPT_ENV = "SGL_NVIDIA_MEGA_DSV41_CKPT"
CKPT_LAYERS = (0, 13, 26, 39)
# The BF16 outputs differ only by FP32 summation order (split-K, K tile)
GEMM_NORM_REL = 5e-4


def _ll_gemm_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import cutlass  # noqa: F401
        import flashinfer  # noqa: F401
    except ImportError:
        return False
    return mxfp8_ll.supported()


needs_ll_gemm = pytest.mark.skipif(
    not _ll_gemm_ready(), reason="needs an SM100-family GPU"
)


def _synthetic_weight(n, k, gen):
    w = torch.randn(n, k, device="cuda", generator=gen) / math.sqrt(k)
    w_q, w_sf = mxfp8_ll.quantize(w.bfloat16())
    return w_q.contiguous(), w_sf.reshape(-1).contiguous()


def _activations(rows, k, gen, norm_weight=None):
    """Rows as a pre-norm feeds them to a projection: heavy-tailed channels (a few
    massive ones), RMS-normalized, times the norm weight; BF16."""
    gain = torch.exp(0.75 * torch.randn(k, device="cuda", generator=gen))
    gain[torch.randperm(k, device="cuda", generator=gen)[:8]] *= 30.0
    h = torch.randn(rows, k, device="cuda", generator=gen) * gain
    y = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
    if norm_weight is not None:
        y = y * norm_weight.float()
    return y.bfloat16()


def _cute_dsl(a, a_sf, w, w_sf):
    """SGLang's MXFP8 linear for block-FP8-as-MXFP8 weights (fp8_utils
    ``flashinfer_mxfp8_blockscaled_linear(backend="cute-dsl", pin_tactic=True)``)."""
    from flashinfer import mm_mxfp8
    from flashinfer.autotuner import autotune

    with autotune(False, skip_ops={"mxfp8_gemm"}):
        return mm_mxfp8(
            a,
            w.t(),
            a_sf.reshape(-1),
            w_sf,
            out_dtype=torch.bfloat16,
            use_8x4_sf_layout=False,
            backend="cute-dsl",
        )


def quantize_row_tiles(x):
    """FlashInfer's swizzled ``mxfp8_quantize`` of each 8-row tile on its own: codes
    [M, K] and the row-tile scale layout (one 128-row atom per tile, rows at its top)."""
    tiles = [mxfp8_ll.quantize(x[t : t + 8]) for t in range(0, x.shape[0], 8)]
    codes = torch.cat([q for q, _ in tiles])
    return codes, torch.cat([s.reshape(-1) for _, s in tiles])


def _ll(x, w, w_sf, tactic=None):
    """The plugin's arm: MXFP8 rows as the layer input arrives (plain swizzled scales
    up to 8 rows, row-tile scales past 8)."""
    if x.shape[0] <= mxfp8_ll.MAX_ROWS:
        a, a_sf = mxfp8_ll.quantize(x)
    else:
        a, a_sf = quantize_row_tiles(x)
    return mxfp8_ll.linear(a, a_sf.reshape(-1), w, w_sf, tactic=tactic), a


def _rel(out, ref):
    d = (out.float() - ref.float()).abs()
    return (d.norm() / ref.float().norm()).item(), d.max().item()


@needs_ll_gemm
def test_draft_wo_b_reads_the_fused_wo_a_rows():
    """Slot draft_wo_b_ll: the low-latency wo_b (N 5120, K 2048) on the MXFP8 rows the
    fused inverse RoPE + wo_a writes, against SGLang's MXFP8 linear on the same rows,
    at the draft's 1-8 rows; eager equals a CUDA graph replay."""
    from sgl_nvidia_mega.deepseek_v41.sm103.attention import wo_a as wo_a_kernel

    gen = torch.Generator(device="cuda").manual_seed(29)
    w, w_sf = _synthetic_weight(5120, 2048, gen)
    wo_a = (torch.randn(2, 1024, 4096, device="cuda", generator=gen) / 64).bfloat16()
    freqs = torch.view_as_real(_freqs_cis()).flatten(1).contiguous()
    worst = 0.0
    for rows in (1, 2, 5, 6, 8):
        o = torch.randn(rows, 2, 4096, device="cuda", generator=gen).bfloat16()
        positions = torch.randint(0, MAX_POS, (rows,), device="cuda", generator=gen)
        codes, scales = wo_a_kernel.fused_rope_wo_a(o, wo_a, freqs, positions)
        out = mxfp8_ll.linear(codes, scales.reshape(-1), w, w_sf)
        ref = _cute_dsl(codes, scales, w, w_sf)
        rel, max_abs = _rel(out, ref)
        worst = max(worst, rel)
        print(
            f"GEMM_REL draft_wo_b.fused_wo_a rows={rows} norm_rel={rel:.3e} max_abs={max_abs:.3e}"
        )
        assert rel <= GEMM_NORM_REL, (rows, rel)
    print(f"GEMM_REL_WORST draft_wo_b {worst:.3e}")


@needs_ll_gemm
def test_tp8_wq_b_takes_its_own_tactics_and_the_indexer_keeps_its_pin():
    """Attention TP8's wq_b (N 4096, K 1280) has the indexer wq_b's shape: its tactics
    are asked for by consumer, so the indexer's bucket-8 pin (TP4) does not move; on
    them the GEMM is SGLang's MXFP8 linear up to summation order, rerun bitwise."""
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    n, k = SHAPES["index_wq_b"]
    assert mxfp8_ll.select_tactic(sm, n, k, 8) == mxfp8_ll.PINNED_TACTICS[sm, n, k]
    gen = torch.Generator(device="cuda").manual_seed(41)
    w, w_sf = _synthetic_weight(n, k, gen)
    for rows in (1, 2, 5, 6, 8, 12, 16):
        bucket = mxfp8_ll.row_bucket(rows)
        tactic = mxfp8_ll.select_tactic(sm, n, k, bucket, mxfp8_ll.WQ_B_TP8)
        pinned = mxfp8_ll.CONSUMER_TACTICS.get((mxfp8_ll.WQ_B_TP8, sm, n, k, bucket))
        assert pinned is None or tactic == pinned, (rows, tactic, pinned)
        x = _activations(rows, k, gen)
        out, a = _ll(x, w, w_sf, tactic=tactic)
        assert torch.equal(_ll(x, w, w_sf, tactic=tactic)[0], out)
        a8, a8_sf = mxfp8_ll.quantize(x)
        rel, _ = _rel(out, _cute_dsl(a8, a8_sf, w, w_sf))
        print(f"GEMM_REL wq_b_tp8 rows={rows} tactic={tactic} norm_rel={rel:.3e}")
        assert rel <= GEMM_NORM_REL, (rows, rel)


# --- KV tolerance --------------------------------------------------------------------


def _ckpt_tensor(root, index, name):
    from safetensors import safe_open

    with safe_open(os.path.join(root, index[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def _ue8m0_bytes(scale):
    if scale.dtype == torch.float32:
        return ((scale.view(torch.int32) >> 23) & 255).to(torch.uint8)
    return scale.view(torch.uint8)


def _kv_layers(source):
    """(label, wqkv_a weight, swizzled scales, attn_norm weight, kv_norm weight) per layer:
    wq_a and wkv stacked as SGLang fuses them, 32x32 UE8M0 blocks expanded to rows."""
    from flashinfer import block_scale_interleave

    if source == "synthetic":
        gen = torch.Generator(device="cuda").manual_seed(17)
        w, w_sf = _synthetic_weight(*SHAPES["wqkv_a"], gen)
        attn_norm = torch.exp(0.3 * torch.randn(5120, device="cuda", generator=gen))
        kv_norm = torch.exp(0.3 * torch.randn(HEAD_DIM, device="cuda", generator=gen))
        return [("synthetic", w, w_sf, attn_norm.bfloat16(), kv_norm.bfloat16())]
    root = os.environ.get(CKPT_ENV)
    if not root:
        pytest.skip(f"{CKPT_ENV} is not set")
    with open(os.path.join(root, "model.safetensors.index.json")) as f:
        index = json.load(f)["weight_map"]
    layers = []
    for layer in CKPT_LAYERS:
        p = f"layers.{layer}."
        get = lambda name: _ckpt_tensor(root, index, p + name)  # noqa: E731
        w = torch.cat(
            [get(f"attn.{m}.weight").view(torch.uint8) for m in ("wq_a", "wkv")]
        )
        scale = torch.cat(
            [_ue8m0_bytes(get("attn.wq_a.scale")), _ue8m0_bytes(get("attn.wkv.scale"))]
        )
        scale = scale.repeat_interleave(32, dim=0)
        assert w.shape == SHAPES["wqkv_a"] and scale.shape == (1792, 5120 // 32)
        w_sf = block_scale_interleave(scale.cuda().contiguous()).reshape(-1)
        layers.append(
            (
                f"layer{layer}",
                w.cuda().view(torch.float8_e4m3fn).contiguous(),
                w_sf.contiguous(),
                get("attn_norm.weight").cuda().bfloat16(),
                get("attn.kv_norm.weight").cuda().bfloat16(),
            )
        )
    return layers


def _freqs_cis():
    inv = 10000.0 ** (-torch.arange(0, ROPE_DIM, 2, device="cuda").float() / ROPE_DIM)
    angles = torch.arange(MAX_POS, device="cuda").float()[:, None] * inv[None, :]
    return torch.polar(torch.ones_like(angles), angles)


def _store(kv, kv_norm, freqs_cis, positions, loc, pages, layout):
    """SGLang's KV store kernel, as set_swa_key_buffer_radix_fused_norm_rope calls it."""
    from sglang.kernels.ops.attention.dsv4 import fused_k_norm_rope_flashmla

    cache = torch.zeros(
        pages, layout.page_bytes(PAGE), dtype=torch.uint8, device="cuda"
    )
    fused_k_norm_rope_flashmla(
        kv=kv,
        kv_weight=kv_norm,
        eps=EPS,
        freqs_cis=freqs_cis,
        positions=positions,
        out_loc=loc,
        kvcache=cache,
        page_size=PAGE,
        layout=layout.value,
    )
    return cache


def _regions(cache, loc, layout):
    """The tokens at ``loc`` by region: FP8 codes [T, n], the UE8M0 scale of each
    code's block [T, n], BF16 RoPE values [T, 64] or None, and the scale bytes."""
    from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

    page, slot = (loc // PAGE).long(), (loc % PAGE).long()
    width, sb = layout.data_bytes, layout.scale_bytes
    data = cache[:, : PAGE * width].view(-1, PAGE, width)[page, slot]
    off = layout.scale_offset(PAGE)
    scales = cache[:, off : off + PAGE * sb].view(-1, PAGE, sb)[page, slot]
    # V4 (SGLang's default window cache): 448 FP8 nope in 64-value blocks, 64 BF16
    # RoPE values, 7 scales + a pad; V4.1: 512 FP8 in 32-value blocks, 16 scales
    if layout is KVLayout.V4:
        fp8, rope, block, nscales = data[:, :448], data[:, 448:576], 64, 7
        rope = rope.contiguous().view(torch.bfloat16)
    else:
        fp8, rope, block, nscales = data, None, 32, 16
    code_scale = scales[:, :nscales].repeat_interleave(block, dim=1)
    return fp8, code_scale, rope, scales[:, :nscales]


def _ordered(bits, sign_bit):
    """Sign-magnitude codes as integers whose difference counts representable steps."""
    bits = bits.long()
    mag = bits & (sign_bit - 1)
    return torch.where(bits & sign_bit != 0, -mag, mag)


@needs_ll_gemm
@pytest.mark.parametrize("layout_name", ["v4", "v41"])
@pytest.mark.parametrize("source", ["synthetic", "checkpoint"])
def test_kv_bytes_tolerance(source, layout_name):
    """wqkv_a on the low-latency GEMM against SGLang's, through SGLang's KV store in
    each window-cache layout: FP8 codes in e4m3 steps (where both arms share the block
    scale), BF16 RoPE values in ulps, scale bytes exact or not."""
    from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

    layout = KVLayout(layout_name)
    freqs_cis = _freqs_cis()
    pages = -(-max(KV_ROWS) * 4 // PAGE) + 1
    st = dict(tokens=0, cases=0, fp8=0, fp8_moved=0, rope=0, rope_moved=0)
    st.update(scales=0, scales_moved=0, kv_bf16_moved=0)
    max_steps, max_ulp, worst_rel = 0, 0, 0.0
    # peak-relative: a tiny change near zero spans many fine codes / ulps, so each region
    # is also measured against its coarsest quantum (FP8: the block's top-binade step,
    # 32 x the block scale; BF16 rope: the row's peak magnitude)
    max_top_steps, max_rope_rel = 0.0, 0.0
    for label, w, w_sf, attn_norm, kv_norm in _kv_layers(source):
        mxfp8_ll.precompile(w, list(KV_ROWS))
        for seed in range(SEEDS):
            gen = torch.Generator(device="cuda").manual_seed(1000 + seed)
            for rows in KV_ROWS:
                x = _activations(rows, 5120, gen, attn_norm)
                a, a_sf = mxfp8_ll.quantize(x)
                ref = _cute_dsl(a, a_sf, w, w_sf)
                out, _ = _ll(x, w, w_sf)
                # each arm is bitwise across reruns (fixed tactics)
                assert torch.equal(_cute_dsl(a, a_sf, w, w_sf), ref), (label, rows)
                assert torch.equal(_ll(x, w, w_sf)[0], out), (label, seed, rows)
                worst_rel = max(worst_rel, _rel(out, ref)[0])
                positions = torch.randint(
                    0, MAX_POS, (rows,), device="cuda", generator=gen
                )
                loc = torch.randperm(pages * PAGE, device="cuda", generator=gen)[:rows]
                loc = loc.int()
                kv_ref, kv_ll = ref[:, Q_LORA:], out[:, Q_LORA:]
                args = (kv_norm, freqs_cis, positions, loc, pages, layout)
                c_ref, c_ll = _store(kv_ref, *args), _store(kv_ll, *args)
                assert torch.equal(_store(kv_ref, *args), c_ref)
                f_ref, cs_ref, r_ref, s_ref = _regions(c_ref, loc, layout)
                f_ll, cs_ll, r_ll, s_ll = _regions(c_ll, loc, layout)
                same_scale = cs_ref == cs_ll
                steps = (_ordered(f_ref, 0x80) - _ordered(f_ll, 0x80)).abs()
                st["fp8"] += f_ref.numel()
                st["fp8_moved"] += int((f_ref != f_ll).sum())
                shared = steps[same_scale]
                if shared.numel():
                    max_steps = max(max_steps, int(shared.max()))
                unit_ref = torch.exp2(cs_ref.float() - 127.0)
                unit_ll = torch.exp2(cs_ll.float() - 127.0)
                deq_ref = f_ref.view(torch.float8_e4m3fn).float() * unit_ref
                deq_ll = f_ll.view(torch.float8_e4m3fn).float() * unit_ll
                top_step = 32.0 * torch.maximum(unit_ref, unit_ll)
                max_top_steps = max(
                    max_top_steps, ((deq_ll - deq_ref).abs() / top_step).max().item()
                )
                if r_ref is not None:
                    ulp = (
                        _ordered(r_ref.view(torch.int16).long() & 0xFFFF, 0x8000)
                        - _ordered(r_ll.view(torch.int16).long() & 0xFFFF, 0x8000)
                    ).abs()
                    st["rope"] += r_ref.numel()
                    st["rope_moved"] += int((r_ref != r_ll).sum())
                    max_ulp = max(max_ulp, int(ulp.max()))
                    peak = r_ref.float().abs().amax(-1, keepdim=True).clamp_min(1e-30)
                    rel = ((r_ll.float() - r_ref.float()).abs() / peak).max().item()
                    max_rope_rel = max(max_rope_rel, rel)
                st["scales"] += s_ref.numel()
                st["scales_moved"] += int((s_ref != s_ll).sum())
                st["kv_bf16_moved"] += int((kv_ref != kv_ll).sum())
                st["tokens"] += rows
                st["cases"] += 1
    print(
        "KV_TOLERANCE "
        + json.dumps(
            dict(
                layout=layout_name,
                weights=source if source == "synthetic" else list(CKPT_LAYERS),
                seeds=SEEDS,
                rows=KV_ROWS,
                cases=st["cases"],
                tokens=st["tokens"],
                kv_bf16_elements_moved=st["kv_bf16_moved"] / (st["tokens"] * HEAD_DIM),
                fp8_codes_moved=st["fp8_moved"] / st["fp8"],
                fp8_max_e4m3_steps=max_steps,
                fp8_max_diff_in_block_top_steps=max_top_steps,
                rope_bf16_values_moved=(st["rope_moved"] / st["rope"])
                if st["rope"]
                else None,
                rope_max_bf16_ulp=max_ulp if st["rope"] else None,
                rope_max_diff_over_row_peak=max_rope_rel if st["rope"] else None,
                scale_bytes_moved=st["scales_moved"] / st["scales"],
                wqkv_a_worst_norm_rel=worst_rel,
                reruns_bitwise=True,
            )
        )
    )
    # at most one step of the block's coarsest quantum (codes near zero move by more
    # fine steps for the same absolute change)
    assert max_top_steps <= 1.0, max_top_steps


# ---------------------------------------- swap-AB attention and the q-LoRA norm + quant

# The plugin's launch (attention._sparse_decode) without and with the low-ratio extras
PLUGIN = dict(max_rows=24, pdl=True, narrow_combine=True, combine_entry_trigger=True)
# The same arithmetic launched as a plain grid: PDL and the triggers only move waits
PLAIN = dict(max_rows=24, narrow_combine=True)


def _attention_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import flashinfer  # noqa: F401
        from triton.experimental import gluon  # noqa: F401
    except ImportError:
        return False
    return True


needs_attention = pytest.mark.skipif(
    not _attention_ready(), reason="needs an SM100-family GPU"
)


def _make_cache(page, pages=11):
    """A padded V4 cache with poisoned unused bytes, and its dequantized values."""
    stride = page * 584 + 256
    raw = torch.full((pages, stride), 0x7F, dtype=torch.uint8, device="cuda")
    fp8 = torch.randn(pages, page, 448, device="cuda").to(torch.float8_e4m3fn)
    exps = torch.randint(124, 130, (pages, page, 7), device="cuda", dtype=torch.uint8)
    rope = torch.randn(pages, page, 64, device="cuda").to(torch.bfloat16)
    tokens = raw[:, : page * 576].view(pages, page, 576)
    tokens[..., :448].copy_(fp8.view(torch.uint8))
    tokens[..., 448:].copy_(rope.view(torch.uint8))
    raw[:, page * 576 : page * 584].view(pages, page, 8)[..., :7].copy_(exps)
    nope = (
        fp8.float().view(pages, page, 7, 64) * torch.exp2(exps.float() - 127)[..., None]
    )
    values = torch.cat((nope.flatten(-2), rope.float()), -1).reshape(-1, 512)
    return raw.as_strided((pages, page, 1, 584), (stride, 584, 584, 1)), values


def _bitwise(a, b):
    assert torch.equal(a.view(torch.int16), b.view(torch.int16)), (
        f"{(a != b).sum().item()} of {a.numel()} differ, "
        f"max |diff| {(a.float() - b.float()).abs().max().item():.3e}"
    )


# 64 rows (65K rotated values) is what catches a reordered RoPE expression: rounding
# order flips about 1 in 30K values; the plugin serves 1-24.
@needs_attention
@pytest.mark.parametrize("b", [6, 16, 24])
def test_swapab_q_rope_and_kv_before_wait(b):
    try:
        from sglang.kernels.ops.attention.dsv4 import fused_q_norm_rope
    except ImportError:
        pytest.skip("this SGLang has no fused_q_norm_rope")
    kv, _ = _make_cache(64)
    extra, _ = _make_cache(16)
    angles = torch.rand(70000, 32, device="cuda") * 6.3
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    freqs = torch.view_as_real(freqs_cis).flatten(-2)
    torch.manual_seed(77 + b)
    q_raw = (torch.randn(b, 16, 512, device="cuda") * 3).to(torch.bfloat16)
    positions = torch.randint(70000, (b,), device="cuda", dtype=torch.int64)
    ids = torch.randint(0, 64 * 11, (b, 1, 128), device="cuda", dtype=torch.int32)
    lengths = torch.full((b,), 128, device="cuda", dtype=torch.int32)
    ei = torch.randint(0, 16 * 11, (b, 1, 512), device="cuda", dtype=torch.int32)
    el = torch.full((b,), 500, device="cuda", dtype=torch.int32)
    sink = torch.randn(64, device="cuda")
    q_pad = torch.zeros(b, 64, 512, device="cuda", dtype=torch.bfloat16)
    fused_q_norm_rope(q_raw, q_pad[:, :16], None, freqs_cis, positions)
    tail = (kv, ids, lengths, sink, extra, ei, el)
    rows = dict(max_rows=64) if b > 24 else {}
    expected = swapab.swapab_attention(q_pad[:, :16], *tail, **{**PLAIN, **rows})
    _bitwise(
        swapab.swapab_attention(q_pad[:, :16], *tail, **{**PLUGIN, **rows}), expected
    )

    def fused(**kwargs):
        return swapab.swapab_attention(
            q_raw, *tail, **{**PLUGIN, **rows}, q_rope=(freqs, positions), **kwargs
        )

    for kv_before_wait in (False, True):
        _bitwise(fused(kv_before_wait=kv_before_wait), expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused(kv_before_wait=True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = fused(kv_before_wait=True)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    _bitwise(out, expected)


@needs_attention
@pytest.mark.parametrize("b", [1, 6, 16, 24])
def test_swapab_8_heads_are_the_16_head_kernels_first_8(b):
    """TP8's 8 local heads ride in the 16-column MMA: the other heads' queries do not
    reach the first 8 heads' outputs, so 8 heads equal the 16-head launch's first 8,
    for each launch form the plugin uses (plain, PDL + RoPE in the kernel, KV before
    the wait, graph replay), with the query packed [b, 8, 512] or a strided view."""
    kv, _ = _make_cache(64)
    extra, _ = _make_cache(16)
    angles = torch.rand(70000, 32, device="cuda") * 6.3
    freqs = torch.view_as_real(torch.polar(torch.ones_like(angles), angles)).flatten(-2)
    torch.manual_seed(91 + b)
    q16 = (torch.randn(b, 16, 512, device="cuda") * 3).to(torch.bfloat16)
    positions = torch.randint(70000, (b,), device="cuda", dtype=torch.int64)
    ids = torch.randint(0, 64 * 11, (b, 1, 128), device="cuda", dtype=torch.int32)
    lengths = torch.randint(1, 129, (b,), device="cuda", dtype=torch.int32)
    ei = torch.randint(0, 16 * 11, (b, 1, 512), device="cuda", dtype=torch.int32)
    ei[:, :, -7:] = -1  # padded compressed indices
    el = torch.randint(1, 513, (b,), device="cuda", dtype=torch.int32)
    sink = torch.randn(16, device="cuda")
    tail16 = (kv, ids, lengths, sink, extra, ei, el)
    tail8 = (kv, ids, lengths, sink[:8].contiguous(), extra, ei, el)
    for q8 in (q16[:, :8].contiguous(), q16[:, :8]):
        for launch in (
            PLAIN,
            dict(PLUGIN, q_rope=(freqs, positions)),
            dict(PLUGIN, q_rope=(freqs, positions), kv_before_wait=True),
            dict(max_rows=24),
        ):
            want = swapab.swapab_attention(q16, *tail16, **launch)
            got = swapab.swapab_attention(q8, *tail8, **launch)
            assert got.shape == (b, 8, 512)
            _bitwise(got, want[:, :8].contiguous())
    # one-row sparse window, no compressed range (ratio 0 layers)
    want = swapab.swapab_attention(q16, kv, ids, lengths, sink, **PLUGIN)
    _bitwise(
        swapab.swapab_attention(q16[:, :8], kv, ids, lengths, sink[:8], **PLUGIN),
        want[:, :8].contiguous(),
    )
    q8 = q16[:, :8].contiguous()
    launch = dict(PLUGIN, q_rope=(freqs, positions), kv_before_wait=True)
    want = swapab.swapab_attention(q16, *tail16, **launch)[:, :8].contiguous()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        swapab.swapab_attention(q8, *tail8, **launch)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = swapab.swapab_attention(q8, *tail8, **launch)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    _bitwise(out, want)


# --------------------------------------------------------------------------- fused wo_a

WO_A_SEEDS = (0, 1, 2)
MAX_POSITION = 4096
HEADS = 64  # the attention output x views: [T, 64, 512]


def _wo_a_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        from cuda.bindings import nvrtc  # noqa: F401
    except ImportError:
        return False
    return True


needs_wo_a = pytest.mark.skipif(not _wo_a_ready(), reason="needs an SM100-family GPU")


def _wo_a_reference():
    """SGLang's wrapper and the most rows it serves; skips without SGLang."""
    module = pytest.importorskip("sglang.kernels.ops.attention.dsv4.wo_a")
    # 48 rows exist only where SGLang has the shared-inbox tile.
    return module, getattr(module, "WIDE_MAX_M", module.MAX_M)


def _wo_a_inputs(rows: int, seed: int, int64_positions: bool, padded_heads: bool):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if padded_heads:
        attention = torch.randn(
            (rows, HEADS, HEAD_DIM),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        x = attention.view(rows, HEADS * HEAD_DIM // wo_a.GROUP_K, wo_a.GROUP_K)[
            :, : wo_a.GROUPS
        ]
    else:
        x = torch.randn(
            (rows, wo_a.GROUPS, wo_a.GROUP_K),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
    if seed == 1:
        x[0].zero_()  # an all-zero row takes the zero-scale branch
    weight = (
        torch.randn(
            (wo_a.GROUPS, wo_a.RANK, wo_a.GROUP_K),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.02
    ).to(torch.bfloat16)
    angles = (
        torch.rand(
            (MAX_POSITION, wo_a.ROPE_DIM // 2),
            generator=generator,
            device="cuda",
        )
        * 6.2831855
    )
    freqs = torch.stack((angles.cos(), angles.sin()), dim=-1).reshape(
        MAX_POSITION, wo_a.ROPE_DIM
    )
    positions = torch.randint(
        0, MAX_POSITION, (rows,), generator=generator, device="cuda"
    ).to(torch.int64 if int64_positions else torch.int32)
    return x, weight, freqs, positions


def _assert_bitwise(actual, expected) -> None:
    codes, scales = actual
    expected_codes, expected_scales = expected
    assert codes.shape == expected_codes.shape and scales.shape == expected_scales.shape
    assert torch.equal(codes.view(torch.uint8), expected_codes.view(torch.uint8))
    assert torch.equal(scales, expected_scales)


@needs_wo_a
@pytest.mark.parametrize("padded_heads", [True, False])
@pytest.mark.parametrize("int64_positions", [False, True])
@pytest.mark.parametrize("rows", [1, 6, 16, 32])
def test_wo_a_fused_matches_sglang(rows, int64_positions, padded_heads):
    reference, reference_max_rows = _wo_a_reference()
    if rows > reference_max_rows:
        pytest.skip(f"SGLang serves at most {reference_max_rows} rows")
    parameters = inspect.signature(reference.fused_rope_wo_a_bf16).parameters
    for seed in WO_A_SEEDS:
        x, weight, freqs, positions = _wo_a_inputs(
            rows, seed, int64_positions, padded_heads
        )
        expected = reference.fused_rope_wo_a_bf16(x, weight, freqs, positions)
        for early_trigger in (True, False):
            _assert_bitwise(
                wo_a.fused_rope_wo_a(
                    x, weight, freqs, positions, early_trigger=early_trigger
                ),
                expected,
            )
        if "row_tile_scales" in parameters:
            _assert_bitwise(
                wo_a.fused_rope_wo_a(x, weight, freqs, positions, row_tile_scales=True),
                reference.fused_rope_wo_a_bf16(
                    x, weight, freqs, positions, row_tile_scales=True
                ),
            )
    torch.cuda.synchronize()


def _row_tile_scale_index(rows: int, groups: int) -> torch.Tensor:
    """The byte offsets the kernel writes in the row-tile scale layout: blocks g of
    rows [0, ceil(rows / 8) * 8) (the last tile's padding rows are written zero)."""
    atoms = groups * wo_a.RANK // 128
    r = torch.arange(-(-rows // 8) * 8)[:, None]
    g = torch.arange(groups * wo_a.RANK // 32)[None, :]
    return (((r >> 3) * atoms + (g >> 2)) * 512 + (r & 7) * 16 + (g & 3)).flatten()


@needs_wo_a
@pytest.mark.parametrize("row_tile_scales", [False, True])
@pytest.mark.parametrize("padded_heads", [True, False])
@pytest.mark.parametrize("int64_positions", [False, True])
@pytest.mark.parametrize("rows", [1, 6, 8, 12, 16, 32, 48])
def test_wo_a_one_group_is_the_first_of_two(
    rows, int64_positions, padded_heads, row_tile_scales
):
    """TP8's 1 local group: each output column reads only its own group's 4096 inputs
    in the same split-K order, so 1 group equals the 2-group kernel's first 1024 columns
    and their scale blocks (the 2-group kernel is SGLang's, test above)."""
    for seed in WO_A_SEEDS:
        x2, w2, freqs, positions = _wo_a_inputs(
            rows, seed, int64_positions, padded_heads
        )
        if padded_heads:
            # TP8's attention output [T, 8, 512] as a strided slice of a padded buffer
            x1 = x2[:, :1]
        else:
            # TP8's attention output [T, 8, 512], contiguous, viewed [T, 1, 4096]
            x1 = x2[:, 0].contiguous().view(rows, 1, wo_a.GROUP_K)
        w1 = w2[:1].contiguous()
        codes2, scales2 = wo_a.fused_rope_wo_a(
            x2, w2, freqs, positions, row_tile_scales=row_tile_scales
        )
        for early_trigger in (True, False):
            codes1, scales1 = wo_a.fused_rope_wo_a(
                x1,
                w1,
                freqs,
                positions,
                early_trigger=early_trigger,
                row_tile_scales=row_tile_scales,
            )
            assert codes1.shape == (rows, wo_a.RANK)
            assert torch.equal(
                codes1.view(torch.uint8), codes2[:, : wo_a.RANK].view(torch.uint8)
            )
            if row_tile_scales:
                assert scales1.numel() == -(-rows // 8) * wo_a.scale_bytes(1)
                i1 = _row_tile_scale_index(rows, 1).cuda()
                i2 = _row_tile_scale_index(rows, 2).cuda()
                # group 0's blocks are the first 32 of each row's 64
                i2 = i2.view(-1, 64)[:, :32].flatten()
                assert torch.equal(scales1[i1], scales2[i2])
            else:
                assert scales1.numel() == wo_a.scale_bytes(1)
                # the 128x4 swizzle puts blocks 0-31 of all 128 rows first
                assert torch.equal(scales1, scales2[: wo_a.scale_bytes(1)])
    torch.cuda.synchronize()


@needs_wo_a
@pytest.mark.parametrize("rows", [1, 6, 16])
def test_wo_a_one_group_within_one_mxfp8_step_of_sglangs_tp8_arm(rows):
    """SGLang at TP8 does not fuse: inverse RoPE in place, the BF16 einsum, then wo_b
    quantizes its input (FlashInfer's MXFP8 quantize). The fused kernel rounds its FP32
    split-K sum to BF16 the same way, so codes differ only where the two BF16 sums do:
    at most one step of the block's coarsest quantum."""
    try:
        from sglang.kernels.ops.attention.dsv4 import fused_rope_inplace
    except ImportError:
        pytest.skip("this SGLang has no fused_rope_inplace")
    if not _ll_gemm_ready():
        pytest.skip("needs FlashInfer's MXFP8 quantize")
    x2, w2, freqs, positions = _wo_a_inputs(rows, 0, True, False)
    x = x2[:, 0].contiguous().view(rows, 8, HEAD_DIM)
    w = w2[:1].contiguous()
    codes, scales = wo_a.fused_rope_wo_a(x.view(rows, 1, -1), w, freqs, positions)
    freqs_cis = torch.view_as_complex(freqs.view(freqs.shape[0], -1, 2).contiguous())
    o = x.clone()
    fused_rope_inplace(
        o[..., -ROPE_DIM:], None, freqs_cis, positions=positions, inverse=True
    )
    y = torch.einsum("tgd,grd->tgr", o.view(rows, 1, -1), w).reshape(rows, -1)
    ref_codes, ref_scales = mxfp8_ll.quantize(y)

    def dequant(c, sf):
        # 128x4 swizzle: block g of row r at (g // 4) * 512 + (r % 32) * 16 + (r // 32 % 4) * 4 + g % 4
        r = torch.arange(rows, device="cuda")[:, None]
        g = torch.arange(wo_a.RANK // 32, device="cuda")[None, :]
        off = (g // 4) * 512 + (r % 32) * 16 + (r // 32 % 4) * 4 + g % 4
        unit = torch.exp2(sf.reshape(-1)[off].float() - 127.0)
        return c.float().view(rows, -1, 32) * unit[..., None], unit

    got, unit_got = dequant(codes, scales)
    want, unit_want = dequant(ref_codes.view(torch.float8_e4m3fn), ref_scales)
    top = 32.0 * torch.maximum(unit_got, unit_want)[..., None]
    steps = ((got - want).abs() / top).max().item()
    moved = (
        (codes.view(torch.uint8) != ref_codes.view(torch.uint8)).float().mean().item()
    )
    print(
        f"WO_A_TP8_VS_SGLANG rows={rows} codes_moved={moved:.2e} max_top_steps={steps:.3f}"
    )
    assert steps <= 1.0, steps


# ------------------------------------------------------------- FlashMLA schedules ahead

RATIOS = (0, 1, 2, 4, 128)
SWA_WIDTH = 128
EXTRA_WIDTH = {1: 512, 2: 512, 4: 512, 128: 128}


def _sched_ready() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


needs_sched_ahead = pytest.mark.skipif(
    not _sched_ready() or backend is None,
    reason="needs an SM100-family GPU and SGLang's DeepSeek-V4 backend",
)


class _Core:
    """The step's attention metadata as the schedule reads it."""

    def __init__(self, rows: int, length_rows: int, seed: int):
        g = torch.Generator(device="cuda").manual_seed(seed)

        def lengths(width):
            return torch.randint(
                1, width + 1, (length_rows,), generator=g, device="cuda"
            ).to(torch.int32)

        self.low_ratios, self.present_ratios = (1, 2), (1, 2, 4, 128)
        self.swa_page_indices = torch.zeros(
            rows, SWA_WIDTH, dtype=torch.int32, device="cuda"
        )
        self.swa_topk_lengths = lengths(SWA_WIDTH)
        self._extra = {
            r: torch.zeros(rows, w, dtype=torch.int32, device="cuda")
            for r, w in EXTRA_WIDTH.items()
        }
        self._extra_lengths = {r: lengths(w) for r, w in EXTRA_WIDTH.items()}
        self._fm = {r: backend._create_flashmla_metadata() for r in RATIOS}

    def get_flashmla_metadata(self, ratio):
        return self._fm[ratio]

    def sparse_page_indices(self, ratio):
        return self._extra[ratio]

    def sparse_topk_lengths(self, ratio):
        return self._extra_lengths[ratio]


def _lazy(core: _Core, rows: int) -> dict:
    """SGLang's per-layer build, with the layer's own query shape and matching."""
    q = torch.empty(rows, 1, 64, 512, dtype=torch.bfloat16, device="cuda")

    def match(x, value):
        if x is None or x.shape[0] == rows:
            return x
        return (
            x[:rows]
            if x.shape[0] > rows
            else backend._pad_tensor_to_size(x, rows, value=value)
        )

    for r in RATIOS:
        extra = None if r == 0 else match(core.sparse_page_indices(r), -1)
        backend._maybe_precompute_flashmla_sched_meta(
            core.get_flashmla_metadata(r),
            q=q,
            indices=match(core.swa_page_indices, 0).unsqueeze(1),
            topk_length=match(core.swa_topk_lengths, 1),
            extra_indices=None if extra is None else extra.unsqueeze(1),
            extra_topk_length=None if r == 0 else match(core.sparse_topk_lengths(r), 1),
        )
    return {r: core.get_flashmla_metadata(r) for r in RATIOS}


def _view(core: _Core) -> SimpleNamespace:
    """The running forward's view (SGLang's ForwardView) over ``core``."""
    return SimpleNamespace(
        metadata=SimpleNamespace(core_attn_metadata=core),
        trtllm_attn=False,
        kv_pool=SimpleNamespace(
            request_window=None,
            pool=SimpleNamespace(
                get_swa_key_layout=lambda: SimpleNamespace(is_interleaved=False)
            ),
        ),
    )


def _ahead(core: _Core, rows: int, graph: bool) -> dict:
    view = _view(core)
    fb = SimpleNamespace(
        forward_mode=SimpleNamespace(is_target_verify=lambda: True),
        positions=torch.zeros(rows, dtype=torch.int64, device="cuda"),
    )
    with mock.patch.object(sched_ahead, "DSV4AttnMetadata", _Core):
        if not graph:
            sched_ahead.join(sched_ahead.start(view, fb, rows, lambda ratio: True))
        else:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                sched_ahead.join(sched_ahead.start(view, fb, rows, lambda ratio: True))
            # replay into the captured buffers after scribbling over them
            for r in RATIOS:
                core.get_flashmla_metadata(r).tile_scheduler_metadata.fill_(-7)
            g.replay()
            torch.cuda.synchronize()
    return {r: core.get_flashmla_metadata(r) for r in RATIOS}


@needs_sched_ahead
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("rows, length_rows", [(12, 12), (48, 52)])
def test_schedule_ahead_is_sglangs_lazy_schedule(rows, length_rows, graph):
    want = _lazy(_Core(rows, length_rows, seed=rows), rows)
    got = _ahead(_Core(rows, length_rows, seed=rows), rows, graph)
    for r in RATIOS:
        assert torch.equal(
            got[r].tile_scheduler_metadata, want[r].tile_scheduler_metadata
        ), r
        assert torch.equal(got[r].num_splits, want[r].num_splits), r
