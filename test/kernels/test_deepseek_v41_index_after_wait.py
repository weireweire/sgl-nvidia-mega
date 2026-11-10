# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot attn_index_after_wait and the fused consumer's fenced trigger.

- The premise: in a captured graph a PDL launch after a stream join depends on the
  joined stream's kernel by a programmatic edge, so it may start (and read) before that
  kernel's writes once the kernel triggers; eager, the join is a full dependency.
- swap-AB with ``extra_after_wait`` is bit for bit the current launch, and in a graph
  behind a selection writer that triggers before it writes (as SGLang's top-k kernels
  do) it reads the new selection on every replay; without it, it reads the old one.
- The fused consumer with ``trigger_after_write``: a dependent that reads the selection
  before its own wait sees the written selection.
- The real index chain (benchmark/bench_deepseek_v41_index_chain.py: the plugin indexer
  on SGLang's kernels, the plugin's query chain, swap-AB, wo_a, wo_b): every consumer
  layer's attention output is the same with the slot and the fenced trigger as without,
  over graph replays (a Full layer's stock top-k emits its picks in no fixed order, so
  its attention varies run to run in every arm).
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl

from sgl_nvidia_mega.deepseek_v41.sm103.attention import swapab

HERE = Path(__file__).resolve().parent


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import deep_gemm  # noqa: F401
        from sglang.kernels.ops.attention.dsv4 import candidate_table  # noqa: F401
    except ImportError:
        return False
    return True


needs_gpu = pytest.mark.skipif(not _ready(), reason="needs an SM100-family GPU, SGLang")

SPIN_NS = 100_000


@triton.jit
def _late_copy(SRC, DST, N: tl.constexpr, BLOCK: tl.constexpr, spin_ns: tl.constexpr):
    # SGLang's top-k shape of the hazard: release the dependents, then write
    tl.extra.cuda.gdc_launch_dependents()
    t0 = tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
    )
    t = t0
    while t - t0 < spin_ns:
        t = tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
        )
    i = tl.arange(0, BLOCK)
    tl.store(DST + i, tl.load(SRC + i, i < N), i < N)


@triton.jit
def _read_before_wait(SRC, DST, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(SRC + i, i < N, volatile=True)
    tl.extra.cuda.gdc_wait()
    tl.store(DST + i, v, i < N)


def _late_writer(src, dst, side):
    main = torch.cuda.current_stream()
    side.wait_stream(main)
    with torch.cuda.stream(side):
        n = src.numel()
        _late_copy[(1,)](
            src, dst, N=n, BLOCK=triton.next_power_of_2(n), spin_ns=SPIN_NS
        )
    main.wait_stream(side)


def _graph(fn):
    fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        out = fn()
    torch.cuda.synchronize()
    return g, out


@needs_gpu
def test_a_captured_join_is_a_programmatic_edge():
    src = torch.arange(64, dtype=torch.int32, device="cuda") + 1
    dst = torch.zeros_like(src)
    seen = torch.zeros_like(src)
    side = torch.cuda.Stream()

    def chain():
        _late_writer(src, dst, side)
        _read_before_wait[(1,)](dst, seen, N=64, BLOCK=64, launch_pdl=True)

    dst.zero_()
    chain()
    torch.cuda.synchronize()
    assert torch.equal(seen, src), "eager: the event join orders the write"
    g, _ = _graph(chain)
    stale = 0
    for _ in range(5):
        dst.zero_()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        stale += int(not torch.equal(seen, src))
    assert stale == 5, "captured: the dependent ran before the joined kernel wrote"


def _cache(pages, page=256, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    stride = page * swapab.LAYOUT.bytes_per_token + 256
    raw = torch.randint(
        0, 120, (pages, stride), dtype=torch.uint8, device="cuda", generator=g
    )
    return raw.as_strided(
        (pages, page, 1, swapab.LAYOUT.bytes_per_token),
        (stride, swapab.LAYOUT.bytes_per_token, swapab.LAYOUT.bytes_per_token, 1),
    )


class Attn:
    """One swap-AB call's operands: a 128-slot window and a 512-slot sparse range."""

    def __init__(self, rows, heads, ctx=20000, seed=0):
        torch.manual_seed(seed)
        self.rows, self.heads = rows, heads
        self.q = (torch.randn(rows, 1, heads, 512, device="cuda") * 0.5).bfloat16()
        self.swa = _cache(4, seed=seed)
        self.extra = _cache(-(-ctx // 256), seed=seed + 1)
        base = torch.randint(0, 600, (rows, 1, 1), device="cuda", dtype=torch.int32)
        self.ids = base + torch.arange(128, device="cuda", dtype=torch.int32)
        self.lengths = torch.full((rows,), 128, device="cuda", dtype=torch.int32)
        self.el = torch.randint(400, 513, (rows,), device="cuda", dtype=torch.int32)
        self.sink = torch.randn(heads, device="cuda")
        angles = torch.rand(ctx, 32, device="cuda") * 6.3
        self.freqs = (
            torch.view_as_real(torch.polar(torch.ones_like(angles), angles))
            .flatten(-2)
            .contiguous()
        )
        self.pos = torch.randint(0, ctx, (rows,), device="cuda")
        self.ctx = ctx

    def selection(self, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        return torch.stack(
            [
                torch.randperm(self.ctx, device="cuda", generator=g)[:512]
                for _ in range(self.rows)
            ]
        ).to(torch.int32)[:, None, :]

    def run(self, sel, **kw):
        return swapab.swapab_attention(
            self.q,
            self.swa,
            self.ids,
            self.lengths,
            self.sink,
            self.extra,
            sel,
            self.el,
            max_rows=24,
            narrow_combine=True,
            q_rope=(self.freqs, self.pos),
            combine_entry_trigger=True,
            pdl=True,
            **kw,
        )


@needs_gpu
@pytest.mark.parametrize("heads", (16, 8))
@pytest.mark.parametrize("rows", (1, 2, 6, 8, 12, 24))
def test_extra_after_wait_is_the_current_launch_bit_for_bit(rows, heads):
    a = Attn(rows, heads, seed=rows)
    sel = a.selection(rows)
    want = a.run(sel, kv_before_wait=True)
    got = a.run(sel, kv_before_wait=True, extra_after_wait=True)
    plain = a.run(sel)
    torch.cuda.synchronize()
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    assert torch.equal(got.view(torch.int16), plain.view(torch.int16))


@needs_gpu
@pytest.mark.parametrize("heads", (16, 8))
@pytest.mark.parametrize("rows", (1, 6, 8))
def test_a_late_selection_is_read_after_the_wait(rows, heads):
    a = Attn(rows, heads, seed=7 + rows)
    sel = torch.empty(rows, 1, 512, dtype=torch.int32, device="cuda")
    fresh = torch.empty_like(sel)
    side = torch.cuda.Stream()
    graphs = {}
    for after in (False, True):

        def chain(after=after):
            _late_writer(fresh, sel, side)
            return a.run(sel, kv_before_wait=True, extra_after_wait=after)

        sel.copy_(a.selection(0))
        fresh.copy_(a.selection(1))
        graphs[after] = _graph(chain)
    stale = {False: 0, True: 0}
    for r in range(8):
        new = a.selection(100 + r)
        want = a.run(new)
        for after, (g, out) in graphs.items():
            sel.copy_(a.selection(200 + r))  # the previous layer's selection
            fresh.copy_(new)
            torch.cuda.synchronize()
            g.replay()
            torch.cuda.synchronize()
            stale[after] += int(
                not torch.equal(out.view(torch.int16), want.view(torch.int16))
            )
    assert stale[True] == 0
    # the hazard the slot removes: the window gathered before the wait, the sparse range
    # too when not after it
    assert stale[False] > 0


def _reindex_case():
    sys.path.insert(0, str(HERE))
    try:
        from test_deepseek_v41_reindex_select import Case
    finally:
        sys.path.remove(str(HERE))
    return Case


@needs_gpu
@pytest.mark.parametrize("rows", (1, 6, 8))
def test_the_fused_consumer_writes_before_it_triggers(rows):
    from sgl_nvidia_mega.deepseek_v41.sm103.attention import reindex_select as rs

    Case = _reindex_case()
    case = Case([30000 + 11 * i for i in range(rows)], seed=rows)
    phys, valid_lens, _ = case.publish()
    out = torch.full((rows, 512), -7, dtype=torch.int32, device="cuda")
    seen = torch.zeros_like(out)
    side = torch.cuda.Stream()

    def chain():
        main = torch.cuda.current_stream()
        side.wait_stream(main)
        with torch.cuda.stream(side):
            rs.reindex_select(
                case.q_fp4, case.q_sf, case.weights, case.k_cache, phys, valid_lens, out
            )
        main.wait_stream(side)
        n = out.numel()
        _read_before_wait[(1,)](
            out, seen, N=n, BLOCK=triton.next_power_of_2(n), launch_pdl=True
        )

    g, _ = _graph(chain)
    for r in range(10):
        other = Case([30000 + 11 * i for i in range(rows)], seed=100 + r)
        case.q_fp4.copy_(other.q_fp4)
        case.weights.copy_(other.weights)
        out.fill_(-7)
        seen.zero_()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        assert (out != -7).any()
        assert torch.equal(seen, out)


def _bench():
    path = HERE.parent.parent / "benchmark" / "bench_deepseek_v41_index_chain.py"
    spec = importlib.util.spec_from_file_location("bench_index_chain", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@needs_gpu
@pytest.mark.parametrize("heads", (16, 8))
@pytest.mark.parametrize("ctx", (8192, 65536))
def test_the_index_chain_attention_is_unchanged_over_replays(heads, ctx):
    from sgl_nvidia_mega.deepseek_v41.sm103.attention import reindex_select as rs

    bench = _bench()
    pattern = "R,S,R,F,R,C,R,C,F,C,R".split(",")
    step = bench.Step(pattern, heads, 6, ctx, stock_consumer=False)
    arms = ("old", "base", "stock_safe", "safe")
    try:
        graphs = {arm: step.graph(arm) for arm in arms}
    finally:
        # the bench binds the fused consumer's trigger per arm
        rs.reindex_select = getattr(rs, "_bench_orig", rs.reindex_select)
    # the consumers' (fused, deterministic) selections; a Full layer's stock paged
    # top-k emits its picks in no fixed order, which moves its attention's tiles
    index = [i for i, k in enumerate(pattern) if k == "C"]
    differ = {arm: 0 for arm in arms[1:]}
    for _ in range(12):
        got = {}
        for arm in arms:
            graphs[arm][0].replay()
            torch.cuda.synchronize()
            got[arm] = [graphs[arm][1][i].clone() for i in index]
        for arm in arms[1:]:
            differ[arm] += sum(
                int(not torch.equal(u.view(torch.int16), v.view(torch.int16)))
                for u, v in zip(got["old"], got[arm])
            )
    assert differ == {arm: 0 for arm in arms[1:]}
