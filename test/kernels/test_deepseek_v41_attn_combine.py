# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Swap-AB attention with the combine merged in the kernel (slot fused_combine) on one
SM100-family GPU (GB300): bit for bit the separate narrow combine at 1-24 rows, 16 and
8 heads, the low-ratio range (window + compressed, padded indices, short lengths) and
the window only (ratio 0), every launch form the chain uses, repeated launches on one
counter, graph replay, and a long-context cache; past one CTA per SM the launch keeps
the separate combine. A row whose tiles never all arrive faults within the deadline
(subprocess).
"""

import subprocess
import sys
import textwrap
import time

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.sm103.attention import swapab


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        from triton.experimental import gluon  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _ready(), reason="needs an SM100-family GPU")

ROWS = (1, 2, 3, 5, 6, 8, 12, 14, 16, 24)


def _cache(page, pages):
    stride = page * 584 + 256
    raw = torch.full((pages, stride), 0x7F, dtype=torch.uint8, device="cuda")
    fp8 = torch.randn(pages, page, 448, device="cuda").to(torch.float8_e4m3fn)
    exps = torch.randint(118, 134, (pages, page, 7), device="cuda", dtype=torch.uint8)
    rope = (torch.randn(pages, page, 64, device="cuda") * 4).to(torch.bfloat16)
    tokens = raw[:, : page * 576].view(pages, page, 576)
    tokens[..., :448].copy_(fp8.view(torch.uint8))
    tokens[..., 448:].copy_(rope.view(torch.uint8))
    raw[:, page * 576 : page * 584].view(pages, page, 8)[..., :7].copy_(exps)
    return raw.as_strided((pages, page, 1, 584), (stride, 584, 584, 1))


def _bitwise(a, b):
    assert torch.equal(a.view(torch.int16), b.view(torch.int16)), (
        f"{(a.view(torch.int16) != b.view(torch.int16)).sum().item()} of {a.numel()} "
        f"differ, max |diff| {(a.float() - b.float()).abs().max().item():.3e}"
    )


def _inputs(b, heads, seed, extra_tokens=16 * 11, window=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    kv = _cache(64, 11)
    extra = _cache(16, -(-extra_tokens // 16))
    q = (torch.randn(b, heads, 512, device="cuda", generator=g) * 3).to(torch.bfloat16)
    ids = torch.randint(
        0, 64 * 11, (b, 1, 128), device="cuda", dtype=torch.int32, generator=g
    )
    lengths = torch.randint(1, 129, (b,), device="cuda", dtype=torch.int32, generator=g)
    lengths[0] = 128
    ei = torch.randint(
        0, extra_tokens, (b, 1, 512), device="cuda", dtype=torch.int32, generator=g
    )
    ei[:, :, -7:] = -1
    el = torch.randint(1, 513, (b,), device="cuda", dtype=torch.int32, generator=g)
    el[-1] = 512
    sink = torch.randn(heads, device="cuda", generator=g) * 4
    angles = torch.rand(70000, 32, device="cuda", generator=g) * 6.3
    freqs = torch.view_as_real(torch.polar(torch.ones_like(angles), angles)).flatten(-2)
    positions = torch.randint(
        70000, (b,), device="cuda", dtype=torch.int64, generator=g
    )
    if not window:
        return q, (kv, ids, lengths, sink), freqs, positions
    return q, (kv, ids, lengths, sink, extra, ei, el), freqs, positions


LAUNCHES = {
    "plain": dict(max_rows=24, narrow_combine=True),
    "chain": dict(
        max_rows=24, pdl=True, narrow_combine=True, combine_entry_trigger=True
    ),
}


def _launch(q, tail, freqs, positions, form, counter=None, kv_before_wait=False):
    kw = dict(LAUNCHES[form])
    if form == "chain":
        kw.update(q_rope=(freqs, positions), kv_before_wait=kv_before_wait)
    return swapab.swapab_attention(q, *tail, combine_counter=counter, **kw)


def _counter():
    return torch.zeros(swapab.WIDE_MAX_BATCH, dtype=torch.int32, device="cuda")


@pytest.mark.parametrize("heads", [16, 8])
@pytest.mark.parametrize("b", ROWS)
def test_fused_combine_is_the_narrow_combine(b, heads):
    counter = _counter()
    for seed in range(3):
        for window_only in (False, True):
            q, tail, freqs, positions = _inputs(
                b, heads, 100 * b + seed, window=not window_only
            )
            for form in ("plain", "chain"):
                for kvbw in (False, True) if form == "chain" else (False,):
                    want = _launch(q, tail, freqs, positions, form, None, kvbw)
                    for _ in range(2):  # the counter resets itself
                        got = _launch(q, tail, freqs, positions, form, counter, kvbw)
                        _bitwise(got, want)
    torch.cuda.synchronize()
    assert int(counter.abs().sum()) == 0


@pytest.mark.parametrize("heads", [16, 8])
@pytest.mark.parametrize("b", [1, 6, 12, 24])
def test_fused_combine_graph_replay(b, heads):
    q, tail, freqs, positions = _inputs(b, heads, 7 + b)
    want = _launch(q, tail, freqs, positions, "chain", None, True)
    counters = [_counter() for _ in range(3)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for c in counters:
            _launch(q, tail, freqs, positions, "chain", c, True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        # three layers back to back, each its own counter, as the chain launches them
        outs = [_launch(q, tail, freqs, positions, "chain", c, True) for c in counters]
    for _ in range(20):
        for o in outs:
            o.zero_()
        graph.replay()
        torch.cuda.synchronize()
        for o in outs:
            _bitwise(o, want)


@pytest.mark.parametrize("heads", [16, 8])
def test_fused_combine_long_context(heads):
    """A 250K-token compressed cache: the top-k scattered over 146 MB."""
    b = 6
    q, tail, freqs, positions = _inputs(b, heads, 5, extra_tokens=250_000)
    want = _launch(q, tail, freqs, positions, "chain", None, True)
    _bitwise(_launch(q, tail, freqs, positions, "chain", _counter(), True), want)


def test_fused_combine_keeps_the_separate_combine_past_one_cta_per_sm():
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    assert swapab.fused_combine_fits(14, 10, "cuda") == (140 <= sms)
    assert not swapab.fused_combine_fits(24, 10, "cuda")
    q, tail, freqs, positions = _inputs(24, 16, 3)
    counter = _counter()
    _bitwise(
        _launch(q, tail, freqs, positions, "chain", counter, True),
        _launch(q, tail, freqs, positions, "chain", None, True),
    )


FAULT_WORKER = textwrap.dedent(
    """
    import sys, torch
    sys.path.insert(0, sys.argv[1])
    from test_deepseek_v41_attn_combine import _inputs, _launch
    from sgl_nvidia_mega.deepseek_v41.sm103.attention import swapab
    q, tail, freqs, positions = _inputs(6, 16, 1)
    counter = torch.zeros(64, dtype=torch.int32, device="cuda")
    # 0 - 5: after the row's 10 arrivals the count is 5, as if 5 tiles never arrived
    counter[0] = -5
    out = swapab.swapab_attention(q, *tail, max_rows=24, narrow_combine=True,
                                  combine_counter=counter, wait_deadline_ns=int(float(sys.argv[2]) * 1e9))
    torch.cuda.synchronize()
    print("NO_FAULT")
    """
)


def test_fused_combine_wait_faults_within_its_deadline():
    import os

    deadline = 0.5
    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", FAULT_WORKER, os.path.dirname(__file__), str(deadline)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    elapsed = time.monotonic() - start
    assert "NO_FAULT" not in proc.stdout, proc.stdout
    assert proc.returncode != 0
    assert "CUDA" in proc.stderr or "cuda" in proc.stderr, proc.stderr[-2000:]
    print(
        f"FAULT after {elapsed:.1f} s (deadline {deadline} s): {proc.stderr.strip().splitlines()[-1]}"
    )


@pytest.mark.parametrize("heads", [16, 8])
@pytest.mark.parametrize("b", ROWS)
@pytest.mark.parametrize("stage", [1, 2])
def test_stage_before_wait_is_the_kv_before_wait_launch(b, heads, stage):
    """Slot attn_prewait: the KV tile staged in shared memory and the RoPE row loaded
    before the PDL wait; the same arithmetic, eager and replayed, both combines."""
    for seed in range(2):
        for window_only in (False, True):
            q, tail, freqs, positions = _inputs(
                b, heads, 300 * b + seed, window=not window_only
            )
            kw = dict(LAUNCHES["chain"], q_rope=(freqs, positions), kv_before_wait=True)
            want = swapab.swapab_attention(q, *tail, **kw)
            _bitwise(
                swapab.swapab_attention(q, *tail, stage_before_wait=stage, **kw), want
            )
            _bitwise(
                swapab.swapab_attention(
                    q, *tail, stage_before_wait=stage, combine_counter=_counter(), **kw
                ),
                want,
            )
    q, tail, freqs, positions = _inputs(b, heads, 9, extra_tokens=250_000)
    kw = dict(LAUNCHES["chain"], q_rope=(freqs, positions), kv_before_wait=True)
    want = swapab.swapab_attention(q, *tail, **kw)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        swapab.swapab_attention(q, *tail, stage_before_wait=stage, **kw)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = swapab.swapab_attention(q, *tail, stage_before_wait=stage, **kw)
    for _ in range(5):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        _bitwise(out, want)
