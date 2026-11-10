# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slots attn_q_norm_gemm, attn_index_q_first and attn_index_weights_bf16.

- ``q_norm_gemm.q_norm_linear`` (q_norm + MXFP8 quant inside FlashInfer's low-latency
  GEMM) against ``rmsnorm_mxfp8`` followed by ``mxfp8_ll.linear``, bit for bit at 1-8
  rows: wq_b at attention TP8 (8 heads) and TP4 (16 heads), the index query's wq_b, the
  quant it writes for the next GEMM (codes, the 128x4 scales and their zero padding), on
  memory poisoned beforehand; past 8 rows it declines.
- The index-first hand-off of ``prepare.low_ratio`` in a captured graph: the index wq_b
  on the main stream, the pack after a join with the head-weights stream, wq_b on that
  stream after an event. Every writer is a late writer that releases its PDL dependents
  before it writes; every replay reads the new inputs (each consumer reads after its
  PDL wait, and a kernel ends only after its own wait, so the order is transitive).
- The BF16 head weights the scorers read: ``head_weights * scale`` is the pack kernel's
  FP32 weights cast to BF16, bit for bit.
"""

import math

import pytest
import torch
import triton
import triton.language as tl


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import flashinfer  # noqa: F401
        from sglang.kernels.ops.attention.dsv4 import fp4_indexer_rope  # noqa: F401
    except ImportError:
        return False
    return True


needs_gpu = pytest.mark.skipif(not _ready(), reason="needs an SM100-family GPU, SGLang")

K = 1280  # q_lora_rank
EPS = 1e-6
SPIN_NS = 50_000
INDEX_N = 4096  # the index query's wq_b at attention TP8 (32 heads x 128)


def _weight(n, seed):
    from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll

    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, K, device="cuda", generator=g) / math.sqrt(K)
    q, sf = mxfp8_ll.quantize(w.bfloat16())
    return q.contiguous(), sf.reshape(-1).contiguous()


def _shapes():
    from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll

    # (name, N, consumer): wq_b at 8 and 16 local heads, the index query's wq_b
    return [
        ("wq_b_tp8", 4096, mxfp8_ll.WQ_B_TP8),
        ("wq_b_tp4", 8192, None),
        ("index", INDEX_N, None),
    ]


def _poison():
    # the caching allocator hands these bytes to the next allocations
    junk = torch.full((1 << 24,), 0x7F, dtype=torch.uint8, device="cuda")
    del junk


def _reference(x, nw, w, ws, consumer):
    from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll
    from sgl_nvidia_mega.deepseek_v41.sm103.attention.q_norm_quant import rmsnorm_mxfp8

    _, q8, s8 = rmsnorm_mxfp8(x, nw, EPS, pdl=True)
    return mxfp8_ll.linear(q8, s8, w, ws, consumer=consumer), q8, s8


@needs_gpu
@pytest.mark.parametrize("shape", range(3))
def test_q_norm_linear_is_the_unfused_chain_bit_for_bit(shape):
    from sgl_nvidia_mega.deepseek_v41.sm103.attention import q_norm_gemm as qng

    name, n, consumer = _shapes()[shape]
    w, ws = _weight(n, shape)
    nw = (1 + 0.1 * torch.randn(K, device="cuda")).bfloat16()
    for rows in range(1, 9):
        assert not qng.reason(w, rows, consumer), (name, rows)
        for trial in range(3):
            # a strided view of wqkv_a's output, as the chain passes it
            qkv = torch.randn(rows, 1792, device="cuda").bfloat16() * (0.5 + trial)
            x = qkv[:, :K]
            ref, q8, s8 = _reference(x, nw, w, ws, consumer)
            _poison()
            out, quant = qng.q_norm_linear(
                x, nw, EPS, w, ws, consumer=consumer, write_quant=True
            )
            _poison()
            plain = qng.q_norm_linear(x, nw, EPS, w, ws, consumer=consumer)
            torch.cuda.synchronize()
            what = (name, rows, trial)
            assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), what
            assert torch.equal(plain.view(torch.int16), ref.view(torch.int16)), what
            assert torch.equal(quant.quant.view(torch.uint8), q8.view(torch.uint8)), (
                what
            )
            assert torch.equal(quant.scale, s8), what
    assert qng.reason(w, 24, consumer)


SPIN = None


def _late_copy_kernel():
    global SPIN
    if SPIN is None:

        @triton.jit
        def late_copy(
            SRC, DST, N: tl.constexpr, BLOCK: tl.constexpr, spin_ns: tl.constexpr
        ):
            # release the PDL dependents, then write late
            tl.extra.cuda.gdc_launch_dependents()
            t0 = tl.inline_asm_elementwise(
                "mov.u64 $0, %globaltimer;",
                "=l",
                [],
                dtype=tl.int64,
                is_pure=False,
                pack=1,
            )
            t = t0
            while t - t0 < spin_ns:
                t = tl.inline_asm_elementwise(
                    "mov.u64 $0, %globaltimer;",
                    "=l",
                    [],
                    dtype=tl.int64,
                    is_pure=False,
                    pack=1,
                )
            i = tl.arange(0, BLOCK)
            tl.store(DST + i, tl.load(SRC + i, i < N), i < N)

        SPIN = late_copy
    return SPIN


def _late_write(src, dst):
    """``dst`` <- ``src`` on the current stream by a kernel that triggers first."""
    s, d = src.view(torch.int16).reshape(-1), dst.view(torch.int16).reshape(-1)
    n = s.numel()
    _late_copy_kernel()[(1,)](
        s, d, N=n, BLOCK=triton.next_power_of_2(n), spin_ns=SPIN_NS, launch_pdl=True
    )


class Hand:
    """``prepare.low_ratio``'s index-first order on stand-in inputs at 6 rows."""

    def __init__(self, fused, rows=6, ctx=4096):
        from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll

        self.fused, self.rows = fused, rows
        self.w, self.ws = _weight(4096, 1)
        self.wi, self.wis = _weight(INDEX_N, 2)
        self.consumer = mxfp8_ll.WQ_B_TP8
        self.heads = INDEX_N // 128
        self.nw = (1 + 0.1 * torch.randn(K, device="cuda")).bfloat16()
        angles = torch.rand(ctx, 32, device="cuda") * 6.3
        self.freqs = (
            torch.view_as_real(torch.polar(torch.ones_like(angles), angles))
            .flatten(-2)
            .contiguous()
        )
        self.pos = torch.randint(0, ctx, (rows,), device="cuda")
        self.scale = 128**-0.5 * self.heads**-0.5
        # the inputs the late writers copy in, and their destinations
        self.src_x = torch.empty(rows, K, dtype=torch.bfloat16, device="cuda")
        self.src_hw = torch.empty(rows, self.heads, dtype=torch.bfloat16, device="cuda")
        self.x = torch.zeros_like(self.src_x)
        self.hw = torch.zeros_like(self.src_hw)
        self.side = torch.cuda.Stream()

    def fill(self, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.src_x.copy_(torch.randn(self.rows, K, device="cuda", generator=g))
        self.src_hw.copy_(
            torch.randn(self.rows, self.heads, device="cuda", generator=g)
        )
        self.x.zero_()
        self.hw.zero_()

    def chain(self):
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_rope import (
            index_q_rope_pack_weights,
        )

        from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll
        from sgl_nvidia_mega.deepseek_v41.sm103.attention import q_norm_gemm as qng
        from sgl_nvidia_mega.deepseek_v41.sm103.attention.q_norm_quant import (
            rmsnorm_mxfp8,
        )

        main, side = torch.cuda.current_stream(), self.side
        side.wait_stream(main)
        with torch.cuda.stream(side):
            _late_write(self.src_hw, self.hw)  # the head weights
        _late_write(self.src_x, self.x)  # wqkv_a's q_lora
        if self.fused:
            index_raw, quant = qng.q_norm_linear(
                self.x, self.nw, EPS, self.wi, self.wis, write_quant=True
            )
            q8, s8 = quant.quant, quant.scale
        else:
            _, q8, s8 = rmsnorm_mxfp8(self.x, self.nw, EPS, pdl=True)
            index_raw = mxfp8_ll.linear(q8, s8, self.wi, self.wis)
        quant_ready = main.record_event()
        main.wait_stream(side)
        packed = index_q_rope_pack_weights(
            index_raw.view(self.rows, self.heads, 128),
            self.freqs,
            self.pos,
            self.hw,
            self.scale,
        )
        side.wait_event(quant_ready)
        with torch.cuda.stream(side):
            q = mxfp8_ll.linear(q8, s8, self.w, self.ws, consumer=self.consumer)
        main.wait_stream(side)
        return (q, index_raw) + tuple(packed)


@needs_gpu
@pytest.mark.parametrize("fused", (False, True))
def test_index_first_hand_off_reads_late_writers(fused):
    h = Hand(fused)
    h.fill(0)
    h.chain()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        outs = h.chain()
    torch.cuda.synchronize()
    for seed in range(1, 9):
        h.fill(seed)
        want = [t.clone() for t in h.chain()]
        torch.cuda.synchronize()
        h.fill(seed)
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        for i, (a, b) in enumerate(zip(outs, want)):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (seed, i)


@needs_gpu
def test_bf16_head_weights_are_the_pack_cast():
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_rope import (
        index_q_rope_pack_weights,
    )

    from sgl_nvidia_mega.deepseek_v41.attention.indexer import StepIndexer

    rows, ctx = 6, 4096
    angles = torch.rand(ctx, 32, device="cuda") * 6.3
    freqs = (
        torch.view_as_real(torch.polar(torch.ones_like(angles), angles))
        .flatten(-2)
        .contiguous()
    )
    pos = torch.randint(0, ctx, (rows,), device="cuda")
    for heads in (32, 64):
        scale = 128**-0.5 * heads**-0.5

        class Attn:
            class indexer:
                head_weight_scale = scale

        q = torch.randn(rows, heads, 128, device="cuda").bfloat16()
        hw = (torch.randn(rows, heads, device="cuda") * 4).bfloat16()
        _, _, weights = index_q_rope_pack_weights(q, freqs, pos, hw, scale)
        got = StepIndexer.weights_bf16(Attn, hw)
        assert got.dtype == torch.bfloat16
        want = weights.to(torch.bfloat16)
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), heads
