# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DeepSeek-V4.1 target logits gather at TP8 on two GB300 trays (8 ranks, one NVLink
domain): SGLang's ``MultimemAllGatherer``, which the logits processor runs on the LM head's
vocab shard (BF16 [rows, 16160] per rank), against the NCCL all-gather it falls back to.

- The gatherer is on across the two trays (one NVLink clique), not on the NCCL fallback.
- Eager at 1-8 rows and at 48 / 128 rows: bit for bit the NCCL gather.
- From a captured CUDA graph, as the verify graph runs it (no entry barrier, a view into
  the symmetric buffer): 200 replays on fresh logits each, with one rank on the other
  tray late by ~1 ms on every replay, bit for bit the NCCL gather of the same logits.
- Timing of both gathers in a graph at the verify's 6 rows (printed, not asserted).

The gather adds no wait of its own: its exit barrier is SGLang's, as on one tray. Run on
two trays with ``torchrun --nnodes 2 --nproc-per-node 4 ... -m pytest
test/kernels/test_deepseek_v41_logits_gather.py -s``.
"""

import os

import pytest
import torch
import torch.distributed as dist
from sglang.srt.distributed import parallel_state as ps
from sglang.test.test_utils import publish_build_topology

VOCAB, WORLD = 129280, 8
LOCAL = VOCAB // WORLD
EAGER_ROWS = (*range(1, 9), 48, 128)
VERIFY_ROWS = 6
REPLAYS = 200
LATE_RANK = 5  # on the second tray
LATE_CYCLES = 2_000_000  # ~1 ms at GB300 clocks

pytestmark = [
    pytest.mark.skipif(
        "RANK" not in os.environ or os.environ.get("WORLD_SIZE") != str(WORLD),
        reason="run on 8 ranks via torchrun (two trays)",
    ),
    pytest.mark.skipif(
        not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
        reason="needs SM100-family GPUs",
    ),
]


@pytest.fixture(scope="module")
def env():
    from sglang.srt.distributed.device_communicators import triton_symm_mem_ag

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    ps.init_distributed_environment(
        world_size=WORLD,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    publish_build_topology(tp_size=WORLD, nnodes=2, world_rank=rank)
    ps.initialize_model_parallel()
    tp = ps.get_tp_group()
    spans = not all(ps.in_the_same_node_as(tp.cpu_group, source_rank=0))
    # as the logits processor builds it
    gatherer = triton_symm_mem_ag.MultimemAllGatherer(
        max_tokens=triton_symm_mem_ag.recommended_max_tokens(
            include_prefill=False, floor=128
        ),
        skip_entry_sync=True,
    )
    yield rank, tp, gatherer, spans


def _nccl(x: torch.Tensor) -> torch.Tensor:
    from sglang.srt.distributed import tensor_model_parallel_all_gather

    return tensor_model_parallel_all_gather(x, dim=-1)


def _logits(rows: int, seed: int, rank: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed * WORLD + rank)
    # LM-head-like magnitudes; every BF16 bit pattern of that range occurs
    return (torch.randn(rows, LOCAL, generator=gen, device="cuda") * 8).bfloat16()


def _sync(tp) -> None:
    torch.cuda.synchronize()
    dist.barrier(group=tp.cpu_group)


def test_gatherer_is_on_across_the_trays(env):
    rank, tp, gatherer, spans = env
    assert spans, "the 8 ranks are on one node; this test needs two trays"
    with torch.inference_mode():
        gatherer(_logits(1, 0, rank))  # first eager call builds the symmetric buffer
    _sync(tp)
    state = gatherer._state
    assert state is not None and state is not gatherer._UNINIT, "fell back to NCCL"
    assert state.world_size == WORLD and state.symm_mem_hdl.multicast_ptr != 0


@torch.inference_mode()
def test_eager_matches_nccl_bitwise(env):
    rank, tp, gatherer, _ = env
    for rows in EAGER_ROWS:
        x = _logits(rows, rows, rank)
        _sync(tp)  # the entry barrier is skipped; the previous view is no longer read
        ours = gatherer(x).clone()
        theirs = _nccl(x)
        assert ours.shape == (rows, VOCAB)
        assert torch.equal(ours.view(torch.int16), theirs.view(torch.int16)), rows


@torch.inference_mode()
def test_graph_replays_with_a_late_rank_match_nccl_bitwise(env):
    rank, tp, gatherer, _ = env
    x = torch.empty(VERIFY_ROWS, LOCAL, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(VERIFY_ROWS, VOCAB, dtype=torch.bfloat16, device="cuda")
    x.copy_(_logits(VERIFY_ROWS, 1000, rank))
    gatherer(x)  # warm the kernel outside capture
    _sync(tp)
    graph = torch.cuda.CUDAGraph()
    with ps.graph_capture() as ctx, torch.cuda.graph(graph, stream=ctx.stream):
        if rank == LATE_RANK:
            torch.cuda._sleep(LATE_CYCLES)
        out.copy_(gatherer(x))
    _sync(tp)
    mismatches = []
    for step in range(REPLAYS):
        x.copy_(_logits(VERIFY_ROWS, 2000 + step, rank))
        _sync(tp)
        graph.replay()
        _sync(tp)
        theirs = _nccl(x)
        if not torch.equal(out.view(torch.int16), theirs.view(torch.int16)):
            mismatches.append(step)
    flag = torch.tensor([len(mismatches)], dtype=torch.int64)
    dist.all_reduce(flag, group=tp.cpu_group)
    assert not mismatches and flag.item() == 0, (rank, mismatches[:8], flag.item())


@torch.inference_mode()
def test_graph_timing(env):
    rank, tp, gatherer, _ = env
    x = _logits(VERIFY_ROWS, 7, rank)
    times = {}
    for name, fn in (("multimem", gatherer), ("nccl", _nccl)):
        fn(x)
        _sync(tp)
        ones = torch.ones(8, device="cuda", dtype=torch.bfloat16)
        graph, base = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with ps.graph_capture() as ctx:
            with torch.cuda.graph(graph, stream=ctx.stream):
                for _ in range(20):
                    fn(x)
                    # what the verify graph has between two gathers: a cross-rank sync
                    tp.all_reduce(ones)
            with torch.cuda.graph(base, stream=ctx.stream):
                for _ in range(20):
                    tp.all_reduce(ones)
        per = []
        for g in (graph, base):
            g.replay()
            _sync(tp)
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(10):
                g.replay()
            end.record()
            torch.cuda.synchronize()
            per.append(start.elapsed_time(end) * 1e3 / 200)
            _sync(tp)
        times[name] = per[0] - per[1]
    print(
        f"RANK{rank} gather {VERIFY_ROWS}x{LOCAL} bf16 per call (us, less the sync): "
        + ", ".join(f"{k} {v:.1f}" for k, v in times.items()),
        flush=True,
    )
