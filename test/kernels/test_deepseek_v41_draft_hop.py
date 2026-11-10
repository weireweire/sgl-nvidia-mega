# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot draft_hop: the DSpark draft step's posts on the plugin's Lamport all-reduce + mHC,
as the draft builds them (``boundary.draft_hop``), at the draft's shapes: 5-45 rows
(gamma 5, batch 1-9), top-k 3 over 128 experts. On 8 SM100-family GPUs (two GB300
trays, TP8) with the routes of MoE TP4 x EP2 (each slot held by the four ranks of one
EP half, -1 on the other four); on 4 with every slot held.

- the attention post (publish + collect, FFN norm folded) and the MoE post (finalize,
  streams only, as the draft attention takes no MXFP8) against SGLang's fused one-shot
  kernels at the same world size (``all_reduce_mhc_post_combine_norm``,
  ``moe_finalize_all_reduce_mhc_post``): within the hop B tolerances, the share of
  equal elements reported; and against a torch all-reduce + mHC post;
- one draft step's six posts (three stages), back to back with each rank late in turn
  and replayed from a CUDA graph, bit for bit with the posts run one at a time.

It registers its own SGLang one-shot plane, so it keeps its own process. Run with
``python test/kernels/test_deepseek_v41_draft_hop.py --num-gpu 4 -s``; at TP8 under a
two-node torchrun (``--nnodes 2 --nproc-per-node 4 -m pytest <file> -s``).
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from sglang.srt.distributed import parallel_state as ps
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import publish_build_topology

from sgl_nvidia_mega.deepseek_v41.spec import NUMERICS

HIDDEN, HC, EPS = 5120, 4, 1e-6
TOP_K, EXPERTS = 3, 128  # the DSpark stages' router (dsv41_gate_topk_e128_k3)
GAMMA, MAX_ROWS = 5, 48  # draft_step.DRAFT_MAX_ROWS
ROWS = tuple(range(GAMMA, MAX_ROWS + 1, GAMMA)) + (1, 6, 8)
STAGES = 3


def _gpus() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 10
        and torch.cuda.device_count() >= 4
    )


pytestmark = [
    pytest.mark.skipif(not _gpus(), reason="needs 4 SM100-family GPUs"),
    pytest.mark.skipif(
        "RANK" not in os.environ,
        reason="run via torchrun: python test/kernels/test_deepseek_v41_draft_hop.py "
        "--num-gpu 4 -s",
    ),
]


@pytest.fixture(scope="module")
def env():
    from sglang.kernels.ops.communication import all_reduce_fusion
    from sglang.srt.distributed.comm_cleanup import register_comm_cleanup
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    from sgl_nvidia_mega.deepseek_v41.boundary import draft_hop

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    ps.set_custom_all_reduce(False)
    ps.init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    publish_build_topology(tp_size=world_size, world_rank=rank)
    ps.initialize_model_parallel()
    tp = ps.get_tp_group()
    device = torch.device(f"cuda:{local_rank}")
    # SGLang's one-shot kernels stage through this communicator's push plane.
    comm = CustomAllReduceV2(
        tp.cpu_group, device, max_pull_size=1 << 20, max_push_size=2 << 20
    )
    assert not comm.disabled and comm.has_multicast
    all_reduce_fusion.register_comm(comm.obj)
    register_comm_cleanup(comm)
    stage = SimpleNamespace(
        hc_cfg=SimpleNamespace(hidden=HIDDEN, mult=HC),
        self_attn=SimpleNamespace(
            wo_b=SimpleNamespace(weight=torch.empty(1, device=device))
        ),
    )
    with torch.inference_mode():
        # the draft's mailbox exactly as draft_hop.build makes it
        hop = draft_hop.DraftHop(
            draft_hop._mailbox(stage, TOP_K, tp, deadline_s=30.0, max_rows=MAX_ROWS)
        )
    yield SimpleNamespace(
        rank=rank,
        tp=world_size,
        device=device,
        group=tp.device_group,
        cpu_group=tp.cpu_group,
        hop=hop,
        ep=2 if world_size == 8 else 1,
    )
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


def _inputs(rows, seed, env):
    """One post's inputs: routing, mixes and norm weight equal on every rank (as in TP),
    GEMM2 rows, the TP-partial shared expert and the attention partial per rank. With
    EP2 the routed slots of the other half's 64 experts are -1."""
    common = torch.Generator().manual_seed(seed)
    local = torch.Generator().manual_seed(seed * 31 + env.rank + 1)
    padded = rows * TOP_K + 8
    perm = torch.randperm(padded, generator=common)[: rows * TOP_K].view(rows, TOP_K)
    experts = torch.randint(0, EXPERTS, (rows, TOP_K), generator=common)
    half = env.rank // (env.tp // env.ep)
    held = experts // (EXPERTS // env.ep) == half
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
    return SimpleNamespace(
        rows=rows, **{k: v.to(env.device) for k, v in tensors.items()}
    )


def _draft(hop, kind, inp):
    """The draft's post of ``kind`` as Boundaries calls the hop."""
    mix = (inp.residual, inp.post, inp.comb)
    if kind == "attn":
        return hop.collect(inp.x, *mix, inp.pre, inp.norm_weight, EPS)
    args = (inp.gemm2, inp.weights, inp.idx, inp.shared, *mix)
    return hop.finalize(*args, None, None, 0.0, rows=inp.rows)[:1]


def _sglang(kind, inp, tp):
    """SGLang's fused post of ``kind`` (what draft_post runs at TP4)."""
    from sglang.kernels.ops.communication import all_reduce_mhc as sgl

    mix = (inp.residual, inp.post, inp.comb)
    if kind == "attn":
        return sgl.all_reduce_mhc_post_combine_norm(
            inp.x, *mix, inp.pre, inp.norm_weight, EPS, world_size=tp
        )[1:]
    routing = (inp.gemm2, inp.idx.flatten(), inp.weights, TOP_K, inp.shared)
    return sgl.moe_finalize_all_reduce_mhc_post(*routing, *mix, world_size=tp)[1:]


def _local(kind, inp):
    if kind == "attn":
        return inp.x
    rows = inp.gemm2.float()[inp.idx.clamp(min=0).long()]
    weights = torch.where(inp.idx >= 0, inp.weights, 0.0)
    routed = (rows * weights.unsqueeze(-1)).sum(1).bfloat16()
    return (routed.float() + inp.shared.float()).bfloat16()


def _torch(kind, inp, group):
    y = _local(kind, inp).float()
    dist.all_reduce(y, group=group)
    y = y.bfloat16().float()
    streams = inp.post.unsqueeze(-1) * y.unsqueeze(1) + (
        inp.comb.unsqueeze(-1) * inp.residual.float().unsqueeze(2)
    ).sum(dim=1)
    streams = streams.bfloat16()
    collapsed = (inp.pre.unsqueeze(-1) * streams.float()).sum(dim=1)
    normed = collapsed * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + EPS)
    return streams, (normed * inp.norm_weight.float()).bfloat16()


def _same(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _ordered(t):
    bits = t.view(torch.int16).int()
    return torch.where(bits < 0, -(bits & 0x7FFF), bits)


def _diff(a, b):
    """(fraction of equal elements, max ULPs where |b| >= 1e-3)."""
    ulps = (_ordered(a) - _ordered(b)).abs()
    big = torch.where(b.float().abs() >= 1e-3, ulps, 0).max().item()
    return (ulps == 0).float().mean().item(), big


@torch.inference_mode()
@pytest.mark.parametrize("kind", ["attn", "moe"])
def test_draft_posts_match_sglangs_fused_kernel(env, kind):
    if env.ep == 2:
        # the EP pair (rank, rank + 4) holds every routed slot exactly once
        inp = _inputs(GAMMA, 1, env)
        held = (inp.idx >= 0).int()
        every = [torch.empty_like(held) for _ in range(env.tp)]
        dist.all_gather(every, held, group=env.group)
        assert torch.equal(held + every[(env.rank + 4) % 8], torch.ones_like(held))
    worst = {}
    for rows in ROWS:
        inp = _inputs(rows, 100 + rows, env)
        got = _draft(env.hop, kind, inp)
        want = _sglang(kind, inp, env.tp)
        torch.cuda.synchronize()
        streams, normed = _torch(kind, inp, env.group)
        torch.testing.assert_close(
            got[0].float(), streams.float(), atol=3e-2, rtol=2e-2
        )
        names = ("streams", "normed")[: len(got)]
        if kind == "attn":
            torch.testing.assert_close(
                got[1].float(), normed.float(), atol=6e-2, rtol=3e-2
            )
        for name, a, b in zip(names, got, want):
            equal, ulps = _diff(a, b)
            print(
                f"DRAFTHOP r{env.rank} tp{env.tp} ep{env.ep} {kind} rows={rows} "
                f"{name}: equal={equal:.6f} max_ulp={ulps}"
            )
            prev = worst.get(name, (1.0, 0))
            worst[name] = (min(prev[0], equal), max(prev[1], ulps))
    print(f"DRAFTHOP r{env.rank} tp{env.tp} {kind} worst={worst}")
    # as hop B against the same kernels: an ULP where the mHC post rounds one product
    # in another order
    assert worst["streams"][1] <= NUMERICS.tolerances["hop_b.residual.ulp"], worst
    assert worst["streams"][0] >= 0.9999, worst
    if "normed" in worst:
        assert worst["normed"][1] <= NUMERICS.tolerances["hop_b.normed.ulp"], worst
        assert worst["normed"][0] >= 0.999, worst


def _step_plan(steps):
    """Draft steps of three stages each (attention post, MoE post), widths cycling."""
    return [
        (kind, ROWS[step % len(ROWS)], stage)
        for step in range(steps)
        for stage in range(STAGES)
        for kind in ("attn", "moe")
    ]


@torch.inference_mode()
def test_draft_steps_are_bitwise_stable_late_and_replayed(env):
    plan = _step_plan(9)
    inputs = {
        (rows, stage): _inputs(rows, 300 + 10 * rows + stage, env)
        for _, rows, stage in plan
    }

    def run(items, late=None):
        outs = []
        for kind, rows, stage in items:
            if late == env.rank:
                torch.cuda._sleep(20_000)
            outs.append(_draft(env.hop, kind, inputs[rows, stage]))
        return outs

    # reference: one post at a time, every rank in step
    reference = []
    for item in plan:
        reference += run([item])
        torch.cuda.synchronize()
        dist.barrier(group=env.cpu_group)

    def compare(outs, label):
        for i, (got, ref) in enumerate(zip(outs, reference)):
            assert all(_same(a, b) for a, b in zip(got, ref)), f"{label}: {plan[i]}"

    for late in (None, *range(env.tp)):
        outs = run(plan, late)
        torch.cuda.synchronize()
        compare(outs, f"back-to-back late={late}")

    # one draft step at 5 rows (batch 1), captured and replayed as the draft graph is
    step = [item for item in plan if item[1] == GAMMA][: 2 * STAGES]
    first = plan.index(step[0])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        captured = run(step)
    torch.cuda.current_stream().wait_stream(stream)
    for replay in range(20):
        if replay % 5 == 4 and env.rank == replay % env.tp:
            torch.cuda._sleep(20_000)
        graph.replay()
        torch.cuda.synchronize()
        for i, got in enumerate(captured):
            ref = reference[first + i]
            assert all(_same(a, b) for a, b in zip(got, ref)), f"replay {replay}: {i}"
        if replay % 4 == 3:
            # eager posts between replays move the generation the graph starts from
            compare(run(plan[:1]), "eager between replays")
    torch.cuda.synchronize()


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,), timeout=2400)
