# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The MXFP8 LM-head stream (sm103/lm_head_mxfp8.py) on SM100-family GPUs (GB300).

- Bit for bit: the MXFP8 kernel equals the same kernel's BF16 build on the dequantized
  weight (same fragments, MMAs and order), at 1-8 rows, on the TP4 and TP8 shards
  (N 32320 / 16160, K 5120: one and two K parts a tile), a ragged N, and on codes that
  cover every E4M3 value (subnormals, +-0, 448) under scales 2^-30 .. 2^7.
- Against SGLang's BF16 projection (``x @ W.T`` in torch) on the BF16 weight: the
  quantization error only, bounded over the logits' peak; and the kernel's own sum
  against torch on the dequantized weight (summation order only).
- The quantizer on the GPU equals the CPU one bit for bit, and its swizzle equals
  FlashInfer's ``mxfp8_quantize`` 128x4 layout (scale positions).
- PDL launch inside a CUDA graph, a strided input and a caller's output buffer.

Run with ``python -m pytest test/kernels/test_deepseek_v41_lm_head_mxfp8.py -s``.
"""

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.sm103 import lm_head_mxfp8 as lh

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)

K = 5120
SHARDS = (32320, 16160, 1000)
# measured on random N(0, 0.02) weights and N(0, 1) rows: the logits' error over their
# peak from the weight's quantization
QUANT_REL = 4e-2


def _weight(n, k=K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(n, k, generator=g, device="cuda") * 0.02).to(torch.bfloat16)


def _rows(rows, k=K, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(rows, k, generator=g, device="cuda").to(torch.bfloat16)


@pytest.fixture(scope="module", params=SHARDS, ids=lambda n: f"n{n}")
def shard(request):
    w = _weight(request.param)
    q = lh.quantize(w)
    return w, q, lh.dequantize(q)


@pytest.mark.parametrize("rows", range(1, lh.MAX_ROWS + 1))
def test_mxfp8_equals_the_bf16_build_on_the_dequantized_weight(shard, rows):
    _, q, dq = shard
    x = _rows(rows)
    ours = lh.matmul(x, q)
    torch.testing.assert_close(ours, lh.matmul_bf16(x, dq), rtol=0, atol=0)
    # the kernel's sum against torch's on the same weight: summation order only
    torch.testing.assert_close(
        ours.float(), (x.float() @ dq.float().T), rtol=2**-7, atol=2e-2
    )


@pytest.mark.parametrize("rows", (5, 6))
def test_quantization_error_against_the_bf16_head(shard, rows):
    w, q, _ = shard
    x = _rows(rows)
    ours = lh.matmul(x, q).float()
    theirs = (x @ w.T).float()
    rel = ((ours - theirs).abs().max() / theirs.abs().max()).item()
    print(f"n={w.shape[0]} rows={rows} rel={rel:.3e}")
    assert rel < QUANT_REL


@pytest.mark.parametrize("log2_scale", (-30, -9, 0, 7))
def test_every_code_dequantizes_exactly(log2_scale):
    """All 254 finite E4M3 codes (no NaN) in every lane position, one scale."""
    n = 256
    finite = torch.tensor(
        [c for c in range(256) if c & 0x7F != 0x7F], dtype=torch.uint8, device="cuda"
    )
    idx = torch.arange(n * K, device="cuda") * 7919 % finite.numel()
    codes = finite[idx].view(n, K).view(torch.float8_e4m3fn)
    scales = torch.full(
        (n, K // lh.BLOCK), 127 + log2_scale, dtype=torch.uint8, device="cuda"
    )
    q = lh.Mxfp8Weight(codes, lh.swizzle(scales))
    dq = lh.dequantize(q)
    assert torch.equal(dq.float(), codes.float() * 2.0**log2_scale), (
        "BF16 holds every code x scale"
    )
    for rows in (1, 6, 8):
        x = _rows(rows)
        torch.testing.assert_close(
            lh.matmul(x, q), lh.matmul_bf16(x, dq), rtol=0, atol=0
        )


def test_the_quantizer_on_gpu_equals_cpu_and_never_saturates():
    w = _weight(640)
    w[3, :32] = 0
    w[5, 64:96] *= 1e4
    gpu = lh.quantize(w)
    cpu = lh.quantize(w.cpu())
    assert torch.equal(gpu.codes.cpu().view(torch.uint8), cpu.codes.view(torch.uint8))
    assert torch.equal(gpu.scales.cpu(), cpu.scales)
    assert gpu.codes.float().abs().max().item() <= lh.E4M3_MAX


def test_the_swizzle_is_flashinfers_128x4_layout():
    """Scale positions: a weight whose every block has its own power of two reads back
    FlashInfer's scale bytes at the same offsets."""
    from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll

    n, k = 256, 512
    e = torch.arange(n * k // lh.BLOCK, device="cuda").view(n, -1) % 40 - 20
    w = torch.ones(n, k // lh.BLOCK, lh.BLOCK, device="cuda") * 2.0 ** e[..., None]
    w = w.view(n, k).to(torch.bfloat16)
    ours = lh.quantize(w)
    _, theirs = mxfp8_ll.quantize(w)
    assert torch.equal(
        ours.scales, theirs.view(torch.uint8).view(-1)[: ours.scales.numel()]
    )


def test_pdl_graph_replay_strided_rows_and_out_buffer():
    w = _weight(32320)
    q = lh.quantize(w)
    lh.precompile(32320, K, w.device)
    base = _rows(6, k=K + 64)
    x = base[:, :K]  # row stride K + 64
    out = torch.empty(6, 32320, dtype=torch.bfloat16, device="cuda")
    want = lh.matmul(x.contiguous(), q)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(3):
            lh.matmul(x, q, out=out)
    out.zero_()
    graph.replay()
    torch.testing.assert_close(out, want, rtol=0, atol=0)
