# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slots shared_fold / shared_fold_quant on one SM100-family GPU (GB300), bit for bit
against the chain they replace (slot shared_ll): the low-latency MXFP8 gate_up GEMM,
SGLang's ``silu_and_mul_clamp``, then FlashInfer's cute-dsl ``mxfp8_quantize`` (128x4
scales) that ``down_proj`` runs.

- The epilogue's SwiGLU against SGLang's kernel on every BF16 gate value x a set of up
  values (clamp edges, signed zeros, subnormals, infinities, NaN), at two limits.
- The interleaved weight: its GEMM columns are the plain GEMM's, reordered.
- The folded GEMM at 1-48 rows (TP4's and TP8's gate_up, narrower / wider ones), with
  zero, negative and clamped rows, eager and replayed from a CUDA graph, on output
  and scale memory poisoned first (every byte must be written).
"""

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import cutlass  # noqa: F401
        import flashinfer  # noqa: F401
    except ImportError:
        return False
    return mxfp8_ll.supported()


needs_gpu = pytest.mark.skipif(not _ready(), reason="needs an SM100-family GPU")

K = 5120  # hidden
INTER = 576  # the shared expert's TP4 shard (2 x 576 gate_up rows)
INTER_TP8 = 288  # TP8's: half a 128-row tile at the end
WIDTHS = [2 * INTER, 2 * INTER_TP8, 2 * 64, 2 * 1152]
LIMITS = (10.0, 6.9)  # DSV4.1's swiglu_limit; one that BF16 rounds
POISON = (0xFF, 0x7F)


def _sglang_swiglu(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    from sgl_nvidia_mega.deepseek_v41 import sglang_ops

    h = gate_up.new_empty((gate_up.shape[0], gate_up.shape[1] // 2))
    sglang_ops.silu_and_mul_clamp(gate_up, h, limit)
    return h


# ------------------------------------------------------------- the SwiGLU, exhaustive


def _bf16(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).bfloat16()


def _up_values(limit: float) -> torch.Tensor:
    lim = _bf16([limit])
    ulp = lambda t, d: (t.view(torch.int16) + d).view(torch.bfloat16)  # noqa: E731
    edges = torch.cat(
        [
            lim,
            -lim,
            ulp(lim, 1),
            ulp(lim, -1),
            ulp(-lim, 1),
            ulp(-lim, -1),
            _bf16([0.0, -0.0, 1.0, -1.0, 0.5, -3.25, 1e-39, -1e-39, 1e-30, 3e38]),
            _bf16([float("inf"), float("-inf"), float("nan")]),
        ]
    )
    g = torch.Generator().manual_seed(0)
    rand = (torch.randn(64 - edges.numel(), generator=g) * 4).bfloat16()
    return torch.cat([edges, rand])


def _device_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """swiglu_ll.swiglu_clamp_bf16 on FP32 inputs (BF16 values), elementwise."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from sgl_nvidia_mega.deepseek_v41.sm103.swiglu_ll import (
        limit_bits,
        swiglu_clamp_bf16,
    )

    bits = limit_bits(limit)

    @cute.kernel
    def kernel(g: cute.Tensor, u: cute.Tensor, o: cute.Tensor, n: cutlass.Int32):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        i = bid * 256 + tid
        if i < n:
            o[i] = cutlass.Int32(swiglu_clamp_bf16(g[i], u[i], bits))

    @cute.jit
    def run(g: cute.Tensor, u: cute.Tensor, o: cute.Tensor, n: cutlass.Int32):
        kernel(g, u, o, n).launch(grid=[(n + 255) // 256, 1, 1], block=[256, 1, 1])

    n = gate.numel()
    out = torch.empty(n, dtype=torch.int32, device="cuda")
    run(
        from_dlpack(gate.float().cuda()),
        from_dlpack(up.float().cuda()),
        from_dlpack(out),
        n,
    )
    return (out & 0xFFFF).to(torch.int16).view(torch.bfloat16)


@needs_gpu
@pytest.mark.parametrize("limit", LIMITS)
def test_swiglu_matches_sglang_on_every_bf16_gate(limit):
    gates = (
        torch.arange(1 << 16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    )
    ups = _up_values(limit)
    gate = gates.repeat(ups.numel())
    up = ups.repeat_interleave(gates.numel())
    width = 4096
    gate_up = torch.cat([gate.view(-1, width), up.view(-1, width)], dim=1).cuda()
    want = _sglang_swiglu(gate_up, limit).view(-1)
    got = _device_swiglu(gate, up, limit)
    torch.cuda.synchronize()
    differ = int((got.view(torch.int16) != want.view(torch.int16)).sum())
    print(f"SWIGLU limit={limit} cases={gate.numel()} differ={differ}")
    assert differ == 0


# ----------------------------------------------------------------------- the GEMMs


def _weights(n: int, seed: int, edges: bool):
    """An MXFP8 gate_up [n, K] as FlashInfer quantizes it; with ``edges`` some rows
    zero, negated or scaled past the clamp."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, K, device="cuda", generator=g) / K**0.5
    if edges:
        inter = n // 2
        w[0:3] = 0  # gate 0: silu 0
        w[inter + 3 : inter + 6] = 0  # up 0
        w[6:40] *= 60  # gate past the clamp
        w[inter + 40 : inter + 80] *= 60  # up past +-limit
        w[inter + 80 : inter + 96] *= -60
        w[100:132] = 0  # a whole 32-column block of h zero
        w[inter - 64 : inter] *= -1
    w_q, w_sf = mxfp8_ll.quantize(w.bfloat16())
    return w_q.contiguous(), w_sf.reshape(-1).contiguous()


def _rows(rows: int, seed: int):
    from sgl_nvidia_mega.deepseek_v41.sm103.attention.q_norm_quant import (
        mxfp8_row_tiles,
    )

    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, K, device="cuda", generator=g).bfloat16()
    if rows > 2:
        x[1] = 0  # a zero row
        x[2] *= 40
    return mxfp8_row_tiles(x)


def _reference(x_q, x_s, w_q, w_sf, limit, quant):
    gate_up = mxfp8_ll.linear(x_q, x_s, w_q, w_sf)
    h = _sglang_swiglu(gate_up, limit)
    if not quant:
        return h
    q, s = mxfp8_ll.quantize(h)  # FlashInfer's mxfp8_quantize as down_proj calls it
    return q, s.reshape(-1)


@needs_gpu
@pytest.mark.parametrize("n", WIDTHS)
def test_interleaved_weight_reorders_the_gemm_columns(n):
    """On the unfolded gate_up's tactic, which the fold runs; padding columns are 0."""
    from sgl_nvidia_mega.deepseek_v41.sm103.swiglu_ll import (
        gate_up_order,
        interleave_gate_up,
    )

    w_q, w_sf = _weights(n, 1, edges=False)
    wi_q, wi_sf = interleave_gate_up(w_q, w_sf)
    order = gate_up_order(n, "cuda")
    valid = order >= 0
    assert wi_q.shape[0] == order.numel() and wi_q.shape[0] % 128 == 0
    for rows in (1, 6, 48):
        x_q, x_s = _rows(rows, rows)
        tactic = mxfp8_ll.swiglu_tactic(x_q.device, n // 2, K, rows)
        want = mxfp8_ll.linear(x_q, x_s, w_q, w_sf)[:, order[valid]]
        got = mxfp8_ll.linear(x_q, x_s, wi_q, wi_sf, tactic=tactic)
        assert torch.equal(got[:, valid].view(torch.int16), want.view(torch.int16))
        assert not got[:, ~valid].view(torch.int16).any(), rows


def _fold(x_q, x_s, wi_q, wi_sf, limit, inter, buffers, byte):
    from sgl_nvidia_mega.deepseek_v41.sm103.swiglu_ll import limit_bits

    rows = x_q.shape[0]
    if buffers is not None:
        buffers.quant.view(torch.uint8).fill_(byte)
        buffers.scale.fill_(byte)
        return mxfp8_ll.linear_swiglu(
            x_q, x_s, wi_q, wi_sf, limit_bits(limit), inter, buffers=buffers
        )
    out = torch.empty((rows, inter), dtype=torch.bfloat16, device="cuda")
    out.view(torch.uint8).fill_(byte)
    return mxfp8_ll.linear_swiglu(
        x_q, x_s, wi_q, wi_sf, limit_bits(limit), inter, out=out
    )


def _same(got, want) -> bool:
    if isinstance(want, tuple):
        return all(
            torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            for a, b in zip(got, want)
        )
    return torch.equal(got.view(torch.int16), want.view(torch.int16))


@needs_gpu
@pytest.mark.parametrize("quant", [False, True], ids=["swiglu", "quant"])
@pytest.mark.parametrize("n", WIDTHS)
@pytest.mark.parametrize("limit", LIMITS)
def test_fold_is_bitwise_at_every_width(limit, n, quant):
    from sgl_nvidia_mega.deepseek_v41.sm103.swiglu_ll import (
        interleave_gate_up,
        limit_bits,
    )

    w_q, w_sf = _weights(n, 7, edges=True)
    wi_q, wi_sf = interleave_gate_up(w_q, w_sf)
    buffers = mxfp8_ll.swiglu_mxfp8_buffers(0, n // 2, 48) if quant else None
    rows_all = list(range(1, 49))
    mxfp8_ll.precompile_swiglu(wi_q, rows_all, limit_bits(limit), n // 2, buffers)
    failed = []
    for rows in rows_all:
        x_q, x_s = _rows(rows, 100 + rows)
        want = _reference(x_q, x_s, w_q, w_sf, limit, quant)
        for byte in POISON:
            got = _fold(x_q, x_s, wi_q, wi_sf, limit, n // 2, buffers, byte)
            torch.cuda.synchronize()
            if not _same(got, want):
                failed.append((rows, hex(byte)))
    print(f"FOLD limit={limit} n={n} quant={quant} rows=1-48 failed={failed}")
    assert not failed


@needs_gpu
@pytest.mark.parametrize("quant", [False, True], ids=["swiglu", "quant"])
@pytest.mark.parametrize("rows", [6, 12, 48])
@pytest.mark.parametrize("inter", [INTER, INTER_TP8])
def test_fold_replays_bitwise_from_a_graph(inter, rows, quant):
    from sgl_nvidia_mega.deepseek_v41.sm103.swiglu_ll import (
        interleave_gate_up,
        limit_bits,
    )

    limit = LIMITS[0]
    w_q, w_sf = _weights(2 * inter, 3, edges=True)
    wi_q, wi_sf = interleave_gate_up(w_q, w_sf)
    buffers = mxfp8_ll.swiglu_mxfp8_buffers(0, inter, 48) if quant else None
    mxfp8_ll.precompile_swiglu(wi_q, [rows], limit_bits(limit), inter, buffers)
    x_q, x_s = _rows(rows, rows)
    want = _reference(x_q, x_s, w_q, w_sf, limit, quant)
    _fold(x_q, x_s, wi_q, wi_sf, limit, inter, buffers, 0)
    torch.cuda.synchronize()
    for byte in POISON:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = _fold(x_q, x_s, wi_q, wi_sf, limit, inter, buffers, byte)
        graph.replay()
        torch.cuda.synchronize()
        assert _same(got, want), hex(byte)


@needs_gpu
@pytest.mark.parametrize("rows", [1, 6, 9, 48])
@pytest.mark.parametrize("inter", [INTER, INTER_TP8])
def test_down_proj_output_is_bitwise_on_the_folded_quant(inter, rows):
    """down_proj (FlashInfer's cute-dsl MXFP8 linear, as SGLang dispatches it on GB300)
    on the folded quant against its own quant of the SwiGLU rows."""
    from sglang.srt.layers.quantization.fp8_utils import (
        flashinfer_mxfp8_blockscaled_linear,
    )

    from sgl_nvidia_mega.deepseek_v41.sm103.swiglu_ll import (
        interleave_gate_up,
        limit_bits,
    )

    limit = LIMITS[0]
    w_q, w_sf = _weights(2 * inter, 5, edges=True)
    wi_q, wi_sf = interleave_gate_up(w_q, w_sf)
    g = torch.Generator(device="cuda").manual_seed(rows)
    d = torch.randn(K, inter, device="cuda", generator=g) / inter**0.5
    d_q, d_sf = mxfp8_ll.quantize(d.bfloat16())
    d_q, d_sf = d_q.contiguous(), d_sf.reshape(-1).contiguous()
    buffers = mxfp8_ll.swiglu_mxfp8_buffers(0, inter, 48)
    x_q, x_s = _rows(rows, 300 + rows)
    gate_up = mxfp8_ll.linear(x_q, x_s, w_q, w_sf)
    clamped = int((gate_up.float().abs() > limit).sum())
    want = flashinfer_mxfp8_blockscaled_linear(
        _sglang_swiglu(gate_up, limit), d_q, d_sf, backend="cute-dsl"
    )
    q, s = mxfp8_ll.linear_swiglu(
        x_q, x_s, wi_q, wi_sf, limit_bits(limit), inter, buffers=buffers
    )
    got = flashinfer_mxfp8_blockscaled_linear(
        q, d_q, d_sf, input_scale=s, backend="cute-dsl"
    )
    torch.cuda.synchronize()
    print(f"DOWN rows={rows} clamped_gate_up={clamped}")
    assert clamped > 0
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_early_release_is_refused_unless_down_proj_is_the_checked_gemm(monkeypatch):
    """Early release relies on down_proj's GEMM reading nothing before its grid wait:
    FlashInfer's cute-dsl MXFP8 GEMMs (persistent and split-K) at a checked release."""
    from types import SimpleNamespace

    flashinfer = pytest.importorskip("flashinfer")

    from sgl_nvidia_mega.deepseek_v41.moe import shared_ll

    def mlp(cutedsl):
        backend = SimpleNamespace(is_flashinfer_cutedsl=lambda: cutedsl)
        quant_method = SimpleNamespace(mxfp8_dense_backend=backend)
        down = SimpleNamespace(quant_method=quant_method)
        return SimpleNamespace(shared_experts=SimpleNamespace(down_proj=down))

    monkeypatch.setattr(
        flashinfer, "__version__", shared_ll.EARLY_RELEASE_FLASHINFER[0]
    )
    assert shared_ll.why_not_early_release(mlp(True)) == ""
    assert "cute-dsl" in shared_ll.why_not_early_release(mlp(False))
    monkeypatch.setattr(flashinfer, "__version__", "0.0.0")
    assert "unchecked" in shared_ll.why_not_early_release(mlp(True))
