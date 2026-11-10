# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DSpark draft's kernels on SM100-family GPUs (GB300).

- The draft's low-M LM head: FlashInfer's cute-dsl ``mm_bf16`` against the BF16
  projection SGLang's draft runs, on the TP4 shard (N 32320, K 5120), at one request's
  rows and the 16-row bound. Only the K summation order differs, so the bound is
  ``sampler.LM_HEAD_LOGITS_REL``.
- The sharded Markov chain (sm103/markov.py) on one rank draws SGLang's
  MarkovSampleStep tokens and keeps its step logits bit for bit (the bias products sum
  exactly): 1-8 rows, greedy, sampled and mixed temperatures, BF16 and FP32 base
  logits, local-shard and full-vocab base logits, three launches and a CUDA graph
  replay with the noise forked onto its side stream. Rows at temperature 0 leave their
  step logits untouched. A missing rank's fault is in test_deepseek_v41_fail_stop.py.
- The same on every rank of a torchrun job (world 2, 4 or 8; world 8 is attention TP8 on
  two GB300 trays).
- Slot markov_lowlat's kernel (converged pollers, acquire loads, the direct exchange to
  4 rows at world 4 and 2 at world 8, the narrow kernel to 8 rows at every world): the
  same SGLang checks, and bit for bit the current kernel's tokens and kept logits over
  many CUDA-graph-replayed draft blocks with Gaussian (not exactly summing) weights at
  1-8 rows, greedy, sampled and mixed temperatures (at world 8, 5-8 rows against the
  current wide kernel).

Run with ``python -m pytest test/kernels/test_deepseek_v41_draft.py -s``; the ranks'
cases with ``torchrun --nproc_per_node 4 -m pytest -k ranks ...`` (world 8:
``torchrun --nnodes 2 --nproc_per_node 4 ...``).
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sgl_nvidia_mega.deepseek_v41 import sampler
from sgl_nvidia_mega.deepseek_v41.sm103 import markov

# ------------------------------------------------------------------------ low-M LM head

needs_lm_head = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)

N, K = 32320, 5120
ROWS = (5, 10, 15, 16)


@pytest.fixture(scope="module")
def head():
    torch.manual_seed(0)
    weight = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
    lm_head = sampler.DraftLmHead(SimpleNamespace(weight=weight), low_m=True)
    assert lm_head.low_m, "FlashInfer cute-dsl mm_bf16 is unavailable"
    lm_head.tune(ROWS)
    return lm_head


def _us(fn, iters=50) -> float:
    """Per-call time from CUDA graph replay, as the draft graph runs it (the CuTe DSL
    host launch path would dominate an eager loop)."""
    for _ in range(3):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / (5 * iters)


@needs_lm_head
@pytest.mark.parametrize("rows", [5, 16])
def test_low_m_logits_are_the_bf16_projection_within_the_bound(head, rows):
    assert head.takes(rows) and not head.takes(sampler.LM_HEAD_MAX_ROWS + 1)
    x = torch.randn(rows, K, device="cuda").to(torch.bfloat16)
    weight = head.lm_head.weight
    want = torch.matmul(x, weight.T).float()
    got = head(x)
    assert got.dtype == torch.bfloat16 and got.shape == (rows, N)
    got = got.float()
    rel = ((got - want).abs().max() / want.abs().max()).item()
    assert rel <= sampler.LM_HEAD_LOGITS_REL, rel
    # greedy drafting reads the argmax; near-ties may flip, nothing else may
    flips = (got.argmax(-1) != want.argmax(-1)).sum().item()
    top2 = want.topk(2, dim=-1).values
    near_tie = ((top2[:, 0] - top2[:, 1]) <= 2 * rel * want.abs().max()).sum().item()
    assert flips <= near_tie
    print(
        f"rows {rows}: rel {rel:.2e}, argmax flips {flips}; "
        f"low-M {_us(lambda: head(x)):.1f} us, "
        f"cuBLAS {_us(lambda: torch.matmul(x, weight.T)):.1f} us"
    )


# ----------------------------------------------------------------- sharded Markov chain

VOCAB = 129_280  # DeepSeek-V4.1
RANK = 256
STEPS = 5
MAX_BS = 8
KEY = 0x1234_5678_9ABC_DEF
TEMPERATURE_SETS = ((0.0,) * 8, (1.0,) * 8, (0.7, 1.3, 1.0, 0.0, 0.5, 2.0, 0.9, 1.1))
# Far above any launch, so only a missing peer reaches it.
DEADLINE_S = 20.0
ARRIVAL_WORDS = slice(128, 136)


def _markov_ready() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() in markov.SUPPORTED_CAPABILITIES
    )


needs_gpu = pytest.mark.skipif(not _markov_ready(), reason="needs an SM100/SM103 GPU")


def _ternary(g: torch.Generator, device, *shape: int) -> torch.Tensor:
    """Sparse {-1, 0, 1} BF16: |bias| <= 256, exact in BF16 in any summation order."""
    values = torch.randint(-1, 2, shape, generator=g)
    values *= torch.rand(shape, generator=g) < 0.25
    return values.to(device, torch.bfloat16)


def _reference(w1, w2, base, anchor, temperatures, offset, cache):
    """SGLang's MarkovSampleStep per step over the full vocab, as the unsharded draft runs it."""
    from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep

    prev, tokens = anchor.long(), []
    for step in range(base.shape[1]):
        bias = F.linear(w1[prev], w2)
        prev = MarkovSampleStep.execute(
            base_logits=base[:, step],
            bias=bias,
            temperatures=temperatures,
            key=KEY,
            offset=offset,
            step=step,
            step_dtype=torch.float32,
            logits_out=None if cache is None else cache[: base.shape[0], step],
            has_sampled_rows=None if cache is not None else False,
        )
        tokens.append(prev)
    return torch.stack(tokens, 1)


def _assert_cache(cache, want_cache, temperatures, rows: int, what) -> None:
    """Rows that keep logits match; temperature-0 rows stay zero."""
    keeps = temperatures[:rows] != 0
    assert torch.equal(cache[:rows][keeps], want_cache[:rows][keeps]), what
    assert not cache[:rows][~keeps].any(), what


def _make(group, rank, world, w1, w2, max_bs=MAX_BS, keeps_logits=True, lowlat=False):
    return markov.ShardedMarkov(
        embed_weight=w1,
        head_weight=w2,
        vocab_size=VOCAB,
        num_steps=STEPS,
        cache_dtype=torch.float32,
        group=group,
        rank=rank,
        world=world,
        max_bs=max_bs,
        deadline_s=DEADLINE_S,
        keeps_logits=keeps_logits,
        lowlat=lowlat,
    )


def _check_equal(device, group, rank: int, world: int, base_dtype, seed: int, lowlat=False):
    g = torch.Generator().manual_seed(seed)
    w1, w2 = _ternary(g, device, VOCAB, RANK), _ternary(g, device, VOCAB, RANK)
    chain = _make(group, rank, world, w1, w2, lowlat=lowlat)
    # A greedy draft's chain has no logit staging; it serves greedy blocks only.
    greedy_chain = _make(group, rank, world, w1, w2, keeps_logits=False, lowlat=lowlat)
    shard = slice(rank * VOCAB // world, (rank + 1) * VOCAB // world)
    offset = torch.tensor([3], dtype=torch.int64, device=device)
    for rows in (1, 2, 4, 8):
        base = (torch.randn(rows, STEPS, VOCAB, generator=g) * 3).to(device, base_dtype)
        anchor = torch.randint(0, VOCAB, (rows,), generator=g).to(device)
        for temps in TEMPERATURE_SETS:
            temperatures = torch.tensor(temps, device=device)
            for keep in (True, False):  # sampled draft vs greedy draft (no logits)
                want_cache = (
                    torch.zeros(MAX_BS, STEPS, VOCAB, device=device) if keep else None
                )
                want = _reference(
                    w1, w2, base, anchor, temperatures[:rows], offset, want_cache
                )
                for sampler in (chain,) if keep else (chain, greedy_chain):
                    for local in (True, False):
                        inputs = base[:, :, shard].contiguous() if local else base
                        for _ in range(3):
                            cache = (
                                torch.zeros(MAX_BS, STEPS, VOCAB, device=device)
                                if keep
                                else None
                            )
                            tokens = torch.full(
                                (MAX_BS, STEPS), -7, dtype=torch.int64, device=device
                            )
                            if keep:
                                sampler.launch_noise(rows=rows, key=KEY, offset=offset)
                            sampler.sample(
                                base_logits=inputs,
                                anchor_tokens=anchor,
                                temperatures=temperatures,
                                logits_cache=cache,
                                draft_tokens=tokens,
                            )
                            torch.cuda.synchronize()
                            what = (rows, temps, keep, local, sampler is chain)
                            assert torch.equal(tokens[:rows], want), what
                            if keep:
                                _assert_cache(
                                    cache, want_cache, temperatures, rows, what
                                )

    _check_forked_graph(device, chain, w1, w2, shard, base_dtype, g)
    assert int(chain.control[ARRIVAL_WORDS].abs().sum()) == 0  # arrivals reset


def _check_forked_graph(device, chain, w1, w2, shard, base_dtype, g):
    """CUDA graph: noise forked onto the side stream, replayed with new offsets."""
    rows = 2
    base = (torch.randn(rows, STEPS, VOCAB, generator=g) * 3).to(device, base_dtype)
    anchor = torch.randint(0, VOCAB, (rows,), generator=g).to(device)
    temperatures = torch.tensor(TEMPERATURE_SETS[2], device=device)
    offset = torch.tensor([3], dtype=torch.int64, device=device)
    cache = torch.zeros(MAX_BS, STEPS, VOCAB, device=device)
    tokens = torch.full((MAX_BS, STEPS), -7, dtype=torch.int64, device=device)
    local = base[:, :, shard].contiguous()

    def forked_sample():
        chain.fork_noise(rows=rows, key=KEY, offset=offset)
        chain.join_noise()
        chain.sample(
            base_logits=local,
            anchor_tokens=anchor,
            temperatures=temperatures,
            logits_cache=cache,
            draft_tokens=tokens,
        )

    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        forked_sample()
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forked_sample()
    for step_offset in (5, 6, 1 << 33):
        offset.fill_(step_offset)
        graph.replay()
        want_cache = torch.zeros_like(cache)
        want = _reference(w1, w2, base, anchor, temperatures[:rows], offset, want_cache)
        torch.cuda.synchronize()
        assert torch.equal(tokens[:rows], want), step_offset
        _assert_cache(cache, want_cache, temperatures, rows, step_offset)


@needs_gpu
@pytest.mark.parametrize("lowlat", [False, True], ids=["current", "lowlat"])
@pytest.mark.parametrize("base_dtype", [torch.bfloat16, torch.float32])
def test_one_rank_matches_markov_sample_step(base_dtype, lowlat):
    _check_equal(torch.device("cuda"), None, 0, 1, base_dtype, seed=1, lowlat=lowlat)


# Blocks per (rows, temperatures) of the replay comparison.
REPLAY_BLOCKS = 300


def _compare_replays(device, group, rank: int, world: int, seed: int) -> dict:
    """Slot markov_lowlat's chain against the current one, both CUDA graphs of one draft
    block (noise forked onto the side stream when a row samples) over the same inputs,
    replayed REPLAY_BLOCKS times per row count and temperature set; mismatch counts."""
    g = torch.Generator().manual_seed(seed)  # the same weights on every rank
    w1 = (torch.randn(VOCAB, RANK, generator=g) * 0.05).to(device, torch.bfloat16)
    w2 = (torch.randn(VOCAB, RANK, generator=g) * 0.05).to(device, torch.bfloat16)
    chains = {lowlat: _make(group, rank, world, w1, w2, lowlat=lowlat) for lowlat in (False, True)}
    vocab_local = VOCAB // world
    shard = slice(rank * vocab_local, (rank + 1) * vocab_local)
    offset = torch.zeros(1, dtype=torch.int64, device=device)
    counts = {"blocks": 0, "tokens": 0, "logits": 0, "distinct": 0}
    for temps in TEMPERATURE_SETS:
        temperatures = torch.tensor(temps, device=device)
        for rows in range(1, MAX_BS + 1):
            sampled = any(t != 0.0 for t in temps[:rows])
            base = torch.zeros(rows, STEPS, vocab_local, dtype=torch.bfloat16, device=device)
            anchor = torch.zeros(rows, dtype=torch.int64, device=device)
            out = {
                lowlat: (
                    torch.full((MAX_BS, STEPS), -7, dtype=torch.int64, device=device),
                    torch.zeros(MAX_BS, STEPS, VOCAB, device=device),
                )
                for lowlat in chains
            }
            graphs = {}
            for lowlat, chain in chains.items():
                tokens, cache = out[lowlat]

                def block(chain=chain, tokens=tokens, cache=cache):
                    if sampled:
                        chain.fork_noise(rows=rows, key=KEY, offset=offset)
                        chain.join_noise()
                    chain.sample(
                        base_logits=base,
                        anchor_tokens=anchor,
                        temperatures=temperatures,
                        logits_cache=cache if sampled else None,
                        draft_tokens=tokens,
                    )

                side = torch.cuda.Stream(device)
                side.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(side):
                    block()
                torch.cuda.current_stream(device).wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    block()
                graphs[lowlat] = graph
            keeps = (temperatures[:rows] != 0).view(rows, 1, 1)
            first = set()
            for i in range(REPLAY_BLOCKS):
                # the same values on every rank (one seed, one device generator algorithm)
                gen = torch.Generator(device=device).manual_seed(seed * 100_003 + i)
                full = torch.randn(rows, STEPS, VOCAB, generator=gen, device=device) * 3
                base.copy_(full[:, :, shard])
                anchor.copy_(torch.randint(0, VOCAB, (rows,), generator=gen, device=device))
                offset.fill_(1000 + i)
                for lowlat in chains:
                    graphs[lowlat].replay()
                (t0, c0), (t1, c1) = out[False], out[True]
                counts["blocks"] += 1
                counts["tokens"] += int((t0[:rows] != t1[:rows]).sum())
                if sampled:
                    counts["logits"] += int(((c0[:rows] != c1[:rows]) & keeps).sum())
                first.add(int(t0[0, 0]))
            counts["distinct"] += len(first)
    torch.cuda.synchronize(device)
    return counts


@needs_gpu
def test_one_rank_lowlat_is_the_current_kernel_over_graph_replays():
    counts = _compare_replays(torch.device("cuda"), None, 0, 1, seed=5)
    assert counts["tokens"] == 0 and counts["logits"] == 0, counts
    assert counts["distinct"] > counts["blocks"] // 4, counts  # the blocks differ


def _pdl_chain(device, pdl: bool, seed: int, blocks: int, temps):
    """``blocks`` draft blocks of one CUDA graph: the low-M LM head (FlashInfer's PDL
    GEMM, the real producer) writes the base logits the chain reads, as the draft
    proposal runs them; tokens and kept logits of every block."""
    g = torch.Generator().manual_seed(seed)
    vocab_local, hidden = VOCAB, 1024
    w1, w2 = _ternary(g, device, VOCAB, RANK), _ternary(g, device, VOCAB, RANK)
    chain = markov.ShardedMarkov(
        embed_weight=w1,
        head_weight=w2,
        vocab_size=VOCAB,
        num_steps=STEPS,
        cache_dtype=torch.float32,
        group=None,
        rank=0,
        world=1,
        max_bs=1,
        deadline_s=DEADLINE_S,
        keeps_logits=True,
        pdl=pdl,
    )
    weight = (torch.randn(vocab_local, hidden, generator=g) * 0.05).to(
        device, torch.bfloat16
    )
    lm_head = sampler.DraftLmHead(SimpleNamespace(weight=weight), low_m=True)
    lm_head.tune((STEPS,))
    x = torch.zeros(STEPS, hidden, dtype=torch.bfloat16, device=device)
    anchor = torch.zeros(1, dtype=torch.int64, device=device)
    temperatures = torch.tensor(temps, device=device)
    offset = torch.tensor([3], dtype=torch.int64, device=device)
    cache = torch.zeros(1, STEPS, VOCAB, device=device)
    tokens = torch.full((1, STEPS), -7, dtype=torch.int64, device=device)

    def block():
        if temps[0] != 0.0:
            chain.fork_noise(rows=1, key=KEY, offset=offset)
        if pdl:
            chain.stage_anchor(anchor)
        base = lm_head(x).view(1, STEPS, -1)
        if temps[0] != 0.0:
            chain.join_noise()
        chain.sample(
            base_logits=base,
            anchor_tokens=anchor,
            temperatures=temperatures,
            logits_cache=cache if temps[0] != 0.0 else None,
            draft_tokens=tokens,
            anchor_staged=pdl,
        )

    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        block()
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        block()
    out_tokens, out_logits = [], []
    for i in range(blocks):
        x.copy_((torch.randn(STEPS, hidden, generator=g) * 2).to(torch.bfloat16))
        anchor.fill_(int(torch.randint(0, VOCAB, (1,), generator=g)))
        offset.fill_(100 + i)
        graph.replay()
        out_tokens.append(tokens.clone())
        out_logits.append(cache[0, :, ::997].clone())
    torch.cuda.synchronize()
    return torch.cat(out_tokens), torch.stack(out_logits)


@needs_lm_head
@pytest.mark.parametrize("temps", [(0.0,), (1.0,)], ids=["greedy", "sampled"])
def test_markov_pdl_behind_the_low_m_lm_head_is_bitwise(temps):
    """Slot markov_pdl: 300 graph-replayed blocks behind the real PDL producer give the
    same tokens and kept logits as the plain launch."""
    device = torch.device("cuda")
    want = _pdl_chain(device, False, seed=11, blocks=300, temps=temps)
    got = _pdl_chain(device, True, seed=11, blocks=300, temps=temps)
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])
    assert len(set(want[0][:, 0].tolist())) > 100  # the blocks differ


_TORCHRUN = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1


@pytest.fixture(scope="module")
def ranks():
    """This torchrun rank's device and CPU group (one process group per module)."""
    import torch.distributed as dist

    world, rank = int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world)
    group = dist.new_group(list(range(world)), backend="gloo")
    yield device, group, rank, world
    dist.barrier(group=group)
    dist.destroy_process_group()


@needs_gpu
@pytest.mark.skipif(not _TORCHRUN, reason="runs on every rank of a torchrun job")
@pytest.mark.parametrize("lowlat", [False, True], ids=["current", "lowlat"])
@pytest.mark.parametrize("base_dtype", [torch.bfloat16, torch.float32])
def test_ranks_match_markov_sample_step(ranks, base_dtype, lowlat):
    """Every rank's sharded chain draws SGLang's tokens and keeps its full-vocab step
    logits; the same seed on every rank, so all ranks agree on the reference."""
    device, group, rank, world = ranks
    assert world in markov.SUPPORTED_WORLDS
    _check_equal(device, group, rank, world, base_dtype, seed=2, lowlat=lowlat)


@needs_gpu
@pytest.mark.skipif(not _TORCHRUN, reason="runs on every rank of a torchrun job")
def test_ranks_lowlat_is_the_current_kernel_over_graph_replays(ranks):
    device, group, rank, world = ranks
    counts = _compare_replays(device, group, rank, world, seed=6)
    print(f"rank {rank} world {world}: {counts}", flush=True)
    assert counts["tokens"] == 0 and counts["logits"] == 0, counts
    assert counts["distinct"] > counts["blocks"] // 4, counts


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
