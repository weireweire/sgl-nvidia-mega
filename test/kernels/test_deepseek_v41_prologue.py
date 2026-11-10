# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The step prologue with slots ``prologue_fork`` / ``engram_side_ar``, against the
serial prologue it replaces.

- One GPU: the embedding written straight into the mHC streams equals SGLang's fused
  lookup of the whole table followed by ``hc_broadcast``, bit for bit (int32 / int64
  ids, 1-48 rows, ids outside the table, -0.0 and NaN payloads in the table).
- Every rank (torchrun): the Engram package's forked prologue (``engram.fork``: the
  hashes, the prefetch's lookups and, with ``engram_side_ar``, both all-reduces and the
  WKV quarters on the Engram stream; ``before_layer`` joins) gives the same rows as the
  serial one (``engram.hashes`` + the main-stream ``reduce`` / ``gather``), from CUDA
  graphs replayed with fresh token ids every replay. The graphs run beside a main
  stream that spins in an NCCL all-reduce and sleeps, with the side stream or one rank
  late in turn, and a reader after the join must always see the step's own rows.
  Stand-in Engram layers (per-rank table shards, SGLang's TP all-reduce, a real MXFP8
  WKV on FlashInfer's GEMM) at DeepSeek-V4.1-Flash's shapes.

Run the second part with
``python test/kernels/test_deepseek_v41_prologue.py --num-gpu 4 -s``.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.runtime import prologue


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


# ------------------------------------------------------------- embedding into streams

VOCAB, HIDDEN, HC = 129280, 5120, 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_the_streams_equal_sglangs_lookup_and_broadcast(dtype):
    from sglang.kernels.ops.embeddings.vocab_parallel_embedding import (
        vocab_parallel_embedding,
    )
    from sglang.kernels.ops.layernorm.mhc import hc_broadcast

    gen = torch.Generator(device="cuda").manual_seed(7)
    table = torch.randn((VOCAB, HIDDEN), device="cuda", generator=gen).bfloat16()
    bits = table.view(torch.int16)
    bits[torch.rand(table.shape, device="cuda", generator=gen) < 1e-3] = -32768  # -0.0
    bits[torch.rand(table.shape, device="cuda", generator=gen) < 1e-4] = 0x7FC1  # NaN
    for rows in range(1, 49):
        ids = torch.randint(0, VOCAB, (rows,), device="cuda", generator=gen)
        ids[0] = VOCAB - 1
        if rows > 2:
            ids[1], ids[2] = VOCAB, -1  # outside the table: zero rows
        ids = ids.to(dtype)
        got = prologue.embed_into_streams(ids, table, HC)
        x = vocab_parallel_embedding(ids, table, 0, VOCAB, 0, VOCAB, VOCAB)
        want = hc_broadcast(x, HC)
        torch.cuda.synchronize()
        assert got.shape == want.shape == (rows, HC, HIDDEN)
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), rows


# ------------------------------------------------------------- Engram prologue fork

N, K, HASH_COLS, LAYERS, SHARD_ROWS = 25600, 6144, 12, 2, 2048
DIM = K // HASH_COLS
REPLAYS = 60  # per scenario


def _dist() -> bool:
    return "RANK" in os.environ and _sm100()


needs_dist = pytest.mark.skipif(
    not _dist(),
    reason="run via torchrun: python test/kernels/test_deepseek_v41_prologue.py "
    "--num-gpu 4 -s",
)


@pytest.fixture(scope="module")
def env():
    from sglang.srt.distributed import parallel_state as ps
    from sglang.test.test_utils import publish_build_topology

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    ps.set_custom_all_reduce(True)
    ps.init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    publish_build_topology(tp_size=world_size, world_rank=rank)
    ps.initialize_model_parallel()
    yield SimpleNamespace(rank=rank, tp=world_size, group=ps.get_tp_group())
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


class _Embed:
    """SGLang's per-rank Engram table: this rank's rows of one hash table, zero for
    the rest, summed over ranks by the TP all-reduce."""

    def __init__(self, shard, row_start):
        self.weight, self.row_start, self.rows, self.dim = (
            shard,
            row_start,
            len(shard),
            DIM,
        )

    def _owned_rows(self, indices):
        local = indices - self.row_start
        owned = (local >= 0) & (local < self.rows)
        rows = self.weight[local.clamp(0, self.rows - 1)]
        return torch.where(owned[..., None], rows, torch.zeros_like(rows))

    def _reduce_owned_rows(self, values):
        from sglang.srt.distributed import tensor_model_parallel_all_reduce

        return tensor_model_parallel_all_reduce(values)


def _engrams(rank, tp):
    from flashinfer import block_scale_interleave

    out = {}
    for h in range(LAYERS):
        gen = torch.Generator(device="cuda").manual_seed(100 + h)
        weight = (torch.randn((N, K), device="cuda", generator=gen) * 0.5).to(
            torch.float8_e4m3fn
        )
        scales = torch.randint(118, 126, (N, K // 32), device="cuda", generator=gen)
        swizzled = block_scale_interleave(scales.to(torch.uint8).contiguous())
        wkv = SimpleNamespace(weight=weight, weight_scale_inv_swizzled=swizzled)
        local = torch.Generator(device="cuda").manual_seed(1000 * h + rank)
        shard = torch.randn(
            (SHARD_ROWS, DIM), device="cuda", generator=local
        ).bfloat16()
        embed = _Embed(shard, rank * SHARD_ROWS)
        out[1 + 20 * h] = SimpleNamespace(embed=embed, wkv=wkv, layer_hash_index=h)
    return out


class _Hasher:
    """Hash ids [rows, layers, hash columns] from the step's token ids alone, as
    SGLang's verify-mode hash reads only the step's inputs (two kernels)."""

    def __init__(self, tp):
        self.mod = tp * SHARD_ROWS

    def __call__(self, input_ids, forward_batch):
        k = torch.arange(LAYERS * HASH_COLS, device=input_ids.device)
        h = input_ids.long()[:, None] * 7919 + k * 104729
        return (h % self.mod).view(-1, LAYERS, HASH_COLS)


def _state(env, prefetch, stream, ids, fork, side_ar):
    from sgl_nvidia_mega.deepseek_v41.engram import Bound

    return SimpleNamespace(
        model=SimpleNamespace(engram_hasher=_Hasher(env.tp)),
        bound={"engram": Bound(prefetch, "", stream if fork else None)},
        input_ids=ids,
        fb=None,
        slots=SimpleNamespace(
            engram_prefetch=True, prologue_fork=fork, engram_side_ar=side_ar
        ),
        scratch={},
        hash_ids=None,
    )


def _step(env, prefetch, stream, ids, out, *, fork, side_ar, late, read_early=False):
    """One prologue up to the first Engram gate's read; the main stream meanwhile runs
    layer 0's stand-in (a sleep and an NCCL all-reduce). ``late``: where to sleep
    ~0.2 ms (the side stream, this rank's main stream, or nowhere). ``read_early``
    (the control): the reader runs before the join."""
    import torch.distributed as dist

    from sgl_nvidia_mega.deepseek_v41 import engram

    st = _state(env, prefetch, stream, ids, fork, side_ar)
    if fork:
        if late == "side":
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                torch.cuda._sleep(200_000)
        engram.fork(st)
    if late == "main":
        torch.cuda._sleep(200_000)
    engram.hashes(st)
    busy = torch.ones(4096, device="cuda")
    dist.all_reduce(busy, group=env.group.device_group)
    engram.before_moe(prefetch.reduce_at, st)
    torch.cuda._sleep(20_000)
    if read_early:
        out.copy_(prefetch.kv)
    engram.before_layer(prefetch.gather_at, st)
    if not read_early:
        out.copy_(prefetch.kv)


def _graph(env, prefetch, stream, ids, out, **kw):
    from sglang.srt.distributed import parallel_state as ps

    for _ in range(2):  # warm up eagerly, as SGLang's capture does
        _step(env, prefetch, stream, ids, out, **kw)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with ps.graph_capture() as ctx:
        with torch.cuda.graph(graph, stream=ctx.stream):
            _step(env, prefetch, stream, ids, out, **kw)
    return graph


@needs_dist
@pytest.mark.parametrize("side_ar", [False, True])
@pytest.mark.parametrize("rows", [1, 6, 8])
def test_the_forked_prologue_gives_the_serial_rows(env, side_ar, rows):
    from sgl_nvidia_mega.deepseek_v41.engram import prefetch as ep

    engrams = _engrams(env.rank, env.tp)
    prefetch = ep.EngramPrefetch(engrams, env.group, min(engrams))
    stream = prefetch.stream
    ids = torch.zeros(rows, dtype=torch.int64, device="cuda")
    shape = (LAYERS, rows, N)
    want = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    got = torch.empty_like(want)
    # the rank sleeps on its own main stream in turn
    delays = ["none", "side"] + [f"rank{r}" for r in range(env.tp)]
    serial = _graph(
        env, prefetch, stream, ids, want, fork=False, side_ar=False, late="none"
    )
    forked = {
        late: _graph(
            env,
            prefetch,
            stream,
            ids,
            got,
            fork=True,
            side_ar=side_ar,
            late="main" if late == f"rank{env.rank}" else late,
        )
        for late in delays
    }
    gen = torch.Generator().manual_seed(rows * 2 + side_ar)
    checked = 0
    for late, graph in forked.items():
        for _ in range(REPLAYS):
            ids.copy_(torch.randint(0, 129280, (rows,), generator=gen))
            got.fill_(float("nan"))
            serial.replay()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(got.view(torch.int16), want.view(torch.int16)), late
            checked += 1
    # an eager serial step on the last ids: the rows are real, not a stale buffer
    eager = torch.empty_like(want)
    _step(env, prefetch, stream, ids, eager, fork=False, side_ar=False, late="none")
    torch.cuda.synchronize()
    assert torch.equal(eager.view(torch.int16), want.view(torch.int16))
    assert checked == len(delays) * REPLAYS


@needs_dist
def test_a_reader_before_the_join_sees_a_stale_step(env):
    """The control: with the side stream late, a reader placed before the join reads
    the previous replay's rows, so the bitwise test above would see a missing join."""
    from sgl_nvidia_mega.deepseek_v41.engram import prefetch as ep

    rows = 6
    engrams = _engrams(env.rank, env.tp)
    prefetch = ep.EngramPrefetch(engrams, env.group, min(engrams))
    ids = torch.zeros(rows, dtype=torch.int64, device="cuda")
    want = torch.empty((LAYERS, rows, N), dtype=torch.bfloat16, device="cuda")
    got = torch.empty_like(want)
    kw = dict(fork=True, side_ar=True, late="side")
    serial = _graph(
        env,
        prefetch,
        prefetch.stream,
        ids,
        want,
        fork=False,
        side_ar=False,
        late="none",
    )
    early = _graph(env, prefetch, prefetch.stream, ids, got, read_early=True, **kw)
    gen = torch.Generator().manual_seed(17)
    stale = 0
    for _ in range(20):
        ids.copy_(torch.randint(0, 129280, (rows,), generator=gen))
        serial.replay()
        early.replay()
        torch.cuda.synchronize()
        stale += not torch.equal(got.view(torch.int16), want.view(torch.int16))
    assert stale >= 15, stale


if __name__ == "__main__":
    from sglang.test.kernels.utils import multigpu_pytest_main

    multigpu_pytest_main(__name__, __file__, num_gpus=(4,), timeout=2400)
