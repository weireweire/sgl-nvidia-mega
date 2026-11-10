# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The one-launch mHC coefficients (slot ``hc_coeff_fused``) against the pair they
replace, DeepGEMM's ``tf32_hc_prenorm_gemm`` over 16 K slices + SGLang's
``_hc_mix_reduce_sinkhorn_kernel``: pre, post and comb bit for bit, at 1-48 rows and
DeepSeek-V4.1-Flash's shape (hc 4, hidden 5120), on random streams and on streams
with zero rows, -0.0, subnormals, huge and tiny rows, one hot column; eagerly, back
to back on one stream, and replayed from a CUDA graph."""

from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.boundary import prenorm

HC, HIDDEN, ITERS = 4, 5120, 20
K, MIX = HC * HIDDEN, (2 + HC) * HC


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(not _sm100(), reason="needs an SM100-family GPU")


def _hc(gen, eps=1e-6, rms_eps=1e-6):
    cfg = SimpleNamespace(mult=HC, sinkhorn_iters=ITERS, eps=eps, rms_eps=rms_eps)
    fn = torch.randn((MIX, K), device="cuda", generator=gen) * 0.02
    scale = torch.rand(3, device="cuda", generator=gen) * 2 + 0.1
    base = torch.randn(MIX, device="cuda", generator=gen) * 0.5
    return SimpleNamespace(cfg=cfg, fn=fn, scale=scale, base=base)


def _kernel(hc):
    from sgl_nvidia_mega.deepseek_v41.sm103.hc_coeff import HcCoeff

    cfg = hc.cfg
    return HcCoeff(
        hc=HC,
        k=K,
        slices=prenorm.SLICES,
        iters=cfg.sinkhorn_iters,
        eps=cfg.eps,
        rms_eps=cfg.rms_eps,
        max_rows=prenorm.MAX_ROWS,
        device="cuda",
    )


def _streams(rows, gen, kind):
    x = torch.randn((rows, K), device="cuda", generator=gen)
    if kind == "edges":
        bits = x.bfloat16().view(torch.int16)
        bits[torch.rand(x.shape, device="cuda", generator=gen) < 1e-3] = -32768  # -0.0
        bits[torch.rand(x.shape, device="cuda", generator=gen) < 1e-3] = 1  # subnormal
        x = bits.view(torch.bfloat16).float()
        x[0] = 0.0
        if rows > 1:
            x[1] *= 3e4
        if rows > 2:
            x[2] *= 1e-30
        if rows > 3:
            x[3] = 0.0
            x[3, 777] = 5.0
        if rows > 4:
            x[4] *= 1e-20  # square sums in the subnormal range
    return x.bfloat16().contiguous()


def _same(got, want, what):
    for g, w, name in zip(got, want, ("pre", "post", "comb")):
        assert g.shape == w.shape, f"{what} {name}: {g.shape} vs {w.shape}"
        diff = (g.view(torch.int32) != w.view(torch.int32)).sum().item()
        assert diff == 0, f"{what} {name}: {diff} words differ"


def _args(hc):
    return hc.fn, hc.scale.float().contiguous(), hc.base.float().contiguous()


@pytest.mark.parametrize("kind", ["random", "edges"])
@pytest.mark.parametrize("eps", [1e-6, 1e-4])
def test_matches_the_prenorm_pair(kind, eps):
    gen = torch.Generator(device="cuda").manual_seed(11)
    hc = _hc(gen, eps=eps, rms_eps=eps)
    kernel = _kernel(hc)
    for rows in range(1, prenorm.MAX_ROWS + 1):
        x = _streams(rows, gen, kind)
        want = prenorm._pair(hc, x)
        got = kernel(x, *_args(hc))
        torch.cuda.synchronize()
        _same(got, want, f"{kind} rows {rows}")


def test_back_to_back_and_graph_replays():
    gen = torch.Generator(device="cuda").manual_seed(5)
    layers = [_hc(gen) for _ in range(3)]
    kernel = _kernel(layers[0])
    rows = 6
    xs = [_streams(rows, gen, "random") for _ in layers]
    want = [prenorm._pair(hc, x) for hc, x in zip(layers, xs)]
    # many launches queued at once: tickets and partials must not mix calls
    outs = [kernel(x, *_args(hc)) for _ in range(40) for hc, x in zip(layers, xs)]
    torch.cuda.synchronize()
    for i, got in enumerate(outs):
        _same(got, want[i % len(layers)], f"queued call {i}")

    static = [x.clone() for x in xs]
    graph = torch.cuda.CUDAGraph()
    kernel(static[0], *_args(layers[0]))  # warm the maps
    with torch.cuda.graph(graph):
        graph_outs = [kernel(x, *_args(hc)) for hc, x in zip(layers, static)]
    for replay in range(20):
        fresh = [_streams(rows, gen, "random" if replay % 2 else "edges") for _ in xs]
        for s, f in zip(static, fresh):
            s.copy_(f)
        graph.replay()
        torch.cuda.synchronize()
        for hc, f, got in zip(layers, fresh, graph_outs):
            _same(got, prenorm._pair(hc, f), f"replay {replay}")
