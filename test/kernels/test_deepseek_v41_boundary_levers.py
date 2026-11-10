# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The opt-in levers of the Lamport all-reduce + mHC consumer (AllReduceMHC) on 4
SM100-family GPUs (GB300) with NVLink multicast:

- ``mbarrier_exchange`` (slot hop_mbar): the RMSNorm's CTA sums cross the cluster by
  st.async onto mbarriers instead of DSMEM stores and a cluster barrier;
- ``final_collapse`` (slot hop_final_collapse): the last layer's hop B also writes the
  head's mHC collapse, bit for bit SGLang's ``mhc_combine`` of the streams it writes
  (and the streams bit for bit the residual-only variant's), so the head runs only
  SGLang's RMSNorm on it.

Each against the same mailbox without it, bit for bit: hop A (with and without the
MXFP8 layer input) and every hop B variant (normed, 128x4 and row-tile MXFP8,
residual-only) interleaved on one 16-row instance at 1-16 rows, back to back with one
rank late in turn and from a replayed CUDA graph, under FlashInfer's and the serving
triggers; and hop B on the 48-row mailbox (slot hop_b_wide) at 1-48 rows.

Run with ``python test/kernels/test_deepseek_v41_boundary_levers.py --num-gpu 4 -s``.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

HIDDEN, HC, TOP_K, EPS = 5120, 4, 6, 1e-6
CAPACITY = 16
ROWS = tuple(range(1, CAPACITY + 1))
LEVERS = {"mbar": dict(mbarrier_exchange=True)}
# the consumers' trigger points: FlashInfer's (end), and what serving builds at batch 1
TRIGGERS = {
    "end": dict(),
    "serving": dict(
        early_trigger=True, hop_b_start_trigger=True, publish_trigger_max_rows=8
    ),
}


def _gpu4() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 10
        and torch.cuda.device_count() >= 4
    )


pytestmark = [
    pytest.mark.skipif(not _gpu4(), reason="needs 4 SM100-family GPUs"),
    pytest.mark.skipif(
        "RANK" not in os.environ,
        reason="run via torchrun: python test/kernels/test_deepseek_v41_boundary_levers.py --num-gpu 4 -s",
    ),
]


def _hop(device, group, capacity=CAPACITY, **kwargs):
    from sgl_nvidia_mega.deepseek_v41.sm103.all_reduce_mhc import AllReduceMHC

    return AllReduceMHC(
        hidden_size=HIDDEN,
        hc_mult=HC,
        max_num_tokens=capacity,
        top_k=TOP_K,
        device=device,
        group=group,
        quantize_layer_input=True,
        residual_only=True,
        row_tile_layer_input=True,
        **kwargs,
    )


def _init_dist():
    from sglang.srt.distributed import parallel_state as ps
    from sglang.test.test_utils import publish_build_topology

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    ps.set_custom_all_reduce(False)
    ps.init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    publish_build_topology(tp_size=world, world_rank=rank)
    ps.initialize_model_parallel()
    tp = ps.get_tp_group()
    return SimpleNamespace(
        rank=rank,
        tp=world,
        device=torch.device(f"cuda:{local_rank}"),
        group=tp.device_group,
        cpu_group=tp.cpu_group,
    )


def _inputs(rows, seed, rank, device):
    """Routing, mixes and norm weight equal on every rank; GEMM2 rows, shared expert
    and hop A's x per rank; ~15% of a rank's routes name an expert it does not hold."""
    common = torch.Generator().manual_seed(seed)
    local = torch.Generator().manual_seed(seed * 31 + rank + 1)
    padded = rows * TOP_K + 8
    perm = torch.randperm(padded, generator=common)[: rows * TOP_K].view(rows, TOP_K)
    held = torch.rand(rows, TOP_K, generator=local) >= 0.15
    tensors = dict(
        idx=torch.where(held, perm, -1).int(),
        weights=torch.rand(rows, TOP_K, generator=common),
        gemm2=torch.randn(padded, HIDDEN, generator=local).bfloat16(),
        shared=(torch.randn(rows, HIDDEN, generator=local) * 0.5).bfloat16(),
        x=torch.randn(rows, HIDDEN, generator=local).bfloat16(),
        residual=torch.randn(rows, HC, HIDDEN, generator=common).bfloat16(),
        post=torch.rand(rows, HC, generator=common),
        comb=torch.randn(rows, HC, HC, generator=common) * 0.1,
        pre=torch.rand(rows, HC, generator=common),
        norm_weight=torch.randn(HIDDEN, generator=common).bfloat16(),
    )
    return SimpleNamespace(rows=rows, **{k: v.to(device) for k, v in tensors.items()})


def _mxfp8_out(rows, device):
    from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll
    from sgl_nvidia_mega.deepseek_v41.sm103.all_reduce_mhc import mxfp8_scale_bytes

    values = torch.empty(rows, HIDDEN, dtype=torch.float8_e4m3fn, device=device)
    if rows <= 8:
        size = mxfp8_scale_bytes(rows, HIDDEN)
        return values, torch.empty(size, dtype=torch.uint8, device=device)
    size = mxfp8_ll.scale_bytes(rows, HIDDEN)
    scales = torch.empty(size, dtype=torch.uint8, device=device)
    return mxfp8_ll.RowTileMxfp8(values, scales)


def _row_tile_scale_offsets(rows, device):
    """The row-tile scales the consumer writes: each row's at its tile's atom, and the
    last tile's rows past ``rows`` (zeroed); the rest of each atom is never read."""
    cols = HIDDEN // 32
    g = torch.arange(cols, device=device)
    r = torch.arange((rows + 7) // 8 * 8, device=device)[:, None]
    return ((r // 8) * (cols // 4) + g // 4) * 512 + (r % 8) * 16 + g % 4


def _run(hop, kind, inp):
    """One boundary of ``kind``: its outputs, MXFP8 buffers last (row-tile scales as
    the bytes the consumer writes)."""
    mix = (inp.residual, inp.post, inp.comb)
    if kind == "residual":
        return hop.finalize(
            inp.gemm2, inp.weights, inp.idx, inp.shared, *mix, None, None, EPS
        )[:1]
    mx = _mxfp8_out(inp.rows, inp.x.device) if kind.endswith("quant") else None
    norm = (inp.pre, inp.norm_weight, EPS)
    if kind.startswith("a"):
        out = hop(inp.x, *mix, *norm, mxfp8_out=mx)
    else:
        out = hop.finalize(
            inp.gemm2, inp.weights, inp.idx, inp.shared, *mix, *norm, mxfp8_out=mx
        )
    if mx is None:
        return tuple(out)
    values, scales = mx
    if inp.rows > 8:
        scales = scales[_row_tile_scale_offsets(inp.rows, scales.device)]
    return (*out, values, scales)


def _same(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _plan(steps):
    """Hop A and every hop B variant at every width, in an order that crosses the
    three generations on every path."""
    plan = []
    for step in range(steps):
        rows = ROWS[(step * 5) % len(ROWS)]
        kinds = ["a", "quant", "residual", "norm", "aquant"]
        if rows > 8:
            kinds.remove("aquant")  # hop A's MXFP8 layer input stops at 128x4 rows
        plan += [(kind, rows, step % 2) for kind in kinds[: 2 + step % len(kinds)]]
    return plan


@pytest.fixture(scope="module")
def env():
    from sglang.srt.distributed import parallel_state as ps

    e = _init_dist()
    with torch.inference_mode():
        e.base = {t: _hop(e.device, e.group, **kw) for t, kw in TRIGGERS.items()}
        e.levers = {
            (lever, t): _hop(e.device, e.group, **kw, **lkw)
            for lever, lkw in LEVERS.items()
            for t, kw in TRIGGERS.items()
        }
    yield e
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


@torch.inference_mode()
@pytest.mark.parametrize("trigger", list(TRIGGERS))
@pytest.mark.parametrize("lever", list(LEVERS))
def test_lever_is_bitwise(env, lever, trigger):
    base, hop = env.base[trigger], env.levers[lever, trigger]
    inputs = {
        (rows, s): _inputs(rows, 700 + 10 * rows + s, env.rank, env.device)
        for rows in ROWS
        for s in (0, 1)
    }
    plan = _plan(24)
    plan = plan[: len(plan) - (len(plan) % 3 == 0)]
    assert len(plan) >= 60 and len(plan) % 3
    assert {rows for _, rows, _ in plan} == set(ROWS)

    def run(h, plan, late=None):
        outs = []
        for kind, rows, s in plan:
            if late == env.rank:
                torch.cuda._sleep(20_000)
            outs.append(_run(h, kind, inputs[rows, s]))
        return outs

    # reference: the mailbox without the lever, one boundary at a time
    reference = []
    for item in plan:
        reference += run(base, [item])
        torch.cuda.synchronize()
        dist.barrier(group=env.cpu_group)

    def compare(outs, label):
        for index, (got, ref) in enumerate(zip(outs, reference[: len(outs)])):
            assert len(got) == len(ref)
            for a, b in zip(got, ref):
                assert _same(a, b), f"{lever}/{trigger} {label}: {index} {plan[index]}"

    for late in (None, *range(env.tp)):
        outs = run(hop, plan, late)
        torch.cuda.synchronize()
        compare(outs, f"back-to-back late={late}")

    # 22 boundaries in one graph (not a multiple of the 3 generations), replayed with
    # poisoned outputs, and eager boundaries between replays moving the generation
    sub = plan[:22]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        captured = run(hop, sub)
    torch.cuda.current_stream().wait_stream(stream)
    for replay in range(20):
        for outs in captured:
            for t in outs:
                t.view(torch.uint8).fill_(0xFF)
        if env.rank == replay % env.tp:
            torch.cuda._sleep(50_000)
        graph.replay()
        torch.cuda.synchronize()
        compare(captured, f"graph replay {replay}")
        if replay % 4 == 3:
            compare(run(hop, plan[: 1 + replay % 3]), "eager between replays")
    torch.cuda.synchronize()
    dist.barrier(group=env.cpu_group)


@torch.inference_mode()
@pytest.mark.parametrize("lever", list(LEVERS))
def test_lever_is_bitwise_on_the_wide_mailbox(env, lever):
    """Hop B on slot hop_b_wide's 48-row mailbox (serving triggers), 1-48 rows."""
    kw = TRIGGERS["serving"]
    base = _hop(env.device, env.group, capacity=48, **kw)
    hop = _hop(env.device, env.group, capacity=48, **kw, **LEVERS[lever])
    for rows in (1, 6, 9, 12, 17, 24, 30, 40, 47, 48):
        inp = _inputs(rows, 800 + rows, env.rank, env.device)
        for kind in ("quant", "residual", "norm", "a"):
            want = _run(base, kind, inp)
            got = _run(hop, kind, inp)
            assert all(_same(a, b) for a, b in zip(got, want)), (lever, kind, rows)
    torch.cuda.synchronize()
    dist.barrier(group=env.cpu_group)


@torch.inference_mode()
@pytest.mark.parametrize("capacity", [16, 48])
def test_final_collapse_is_sglangs_head_combine(env, capacity):
    from sglang.kernels.ops.layernorm.mhc import mhc_combine

    kw = TRIGGERS["serving"]
    hop = _hop(env.device, env.group, capacity=capacity, final_collapse=True, **kw)
    plain = _hop(env.device, env.group, capacity=capacity, **kw)
    mix = lambda inp: (inp.residual, inp.post, inp.comb)  # noqa: E731
    rows_list = range(1, capacity + 1) if capacity == 16 else (17, 24, 33, 48)
    for rows in rows_list:
        inp = _inputs(rows, 900 + rows, env.rank, env.device)
        moe = (inp.gemm2, inp.weights, inp.idx, inp.shared, *mix(inp))
        for _ in range(4):  # cycle the generations, alternating with the plain path
            streams, collapsed = hop.finalize(*moe, inp.pre, None, EPS, collapse=True)
            want = plain.finalize(*moe, None, None, EPS)[0]
            head = mhc_combine(streams.flatten(1).float(), inp.pre, HC, streams.dtype)
            assert _same(streams, want), rows
            assert _same(collapsed, head), rows
            # the plain hop on the same instance stays bitwise too
            again = hop.finalize(*moe, None, None, EPS)[0]
            assert _same(again, want), rows
    torch.cuda.synchronize()
    dist.barrier(group=env.cpu_group)


if __name__ == "__main__":
    from sglang.test.kernels.utils import multigpu_pytest_main

    multigpu_pytest_main(__name__, __file__, num_gpus=(4,), timeout=3600)
