# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4.1 hop B, the MoE finalize into the Lamport all-reduce + mHC, and hop A on
the same instance, on 4 SM100-family GPUs (GB300) with NVLink multicast, at 1-8 rows; on
8 (two trays, TP8) also with the routes of MoE TP4 x EP2, where each slot is held by the
four ranks of one EP half and dropped (-1) by the other four.

- ``AllReduceMHC.finalize`` with MXFP8 output and residual-only against SGLang's
  one-shot push kernels (streams, normalized row, MXFP8 values and scales), and its
  MXFP8 output against FlashInfer's ``mxfp8_quantize`` of its own normalized row;
- hop A and both finalize variants interleaved on one instance, with late ranks and
  from a replayed CUDA graph, stay bitwise stable and match torch.

It registers its own SGLang one-shot plane, so it keeps its own process. Run with
``python test/kernels/test_deepseek_v41_hop_b.py --num-gpu 4 -s``; at TP8 under a
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

HIDDEN, HC, TOP_K, EPS = 5120, 4, 6, 1e-6
EXPERTS, EP = 384, 2
ROWS = tuple(range(1, 9))


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
        reason="run via torchrun: python test/kernels/test_deepseek_v41_hop_b.py --num-gpu 4 -s",
    ),
]


@pytest.fixture(scope="module")
def env():
    from sglang.kernels.ops.communication import all_reduce_fusion
    from sglang.srt.distributed.comm_cleanup import register_comm_cleanup
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    from sgl_nvidia_mega.deepseek_v41.sm103.all_reduce_mhc import AllReduceMHC

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
    with torch.inference_mode():
        hops = {
            early: AllReduceMHC(
                hidden_size=HIDDEN,
                hc_mult=HC,
                max_num_tokens=16,
                top_k=TOP_K,
                device=device,
                group=tp.device_group,
                quantize_layer_input=True,
                residual_only=True,
                early_trigger=early == "poll",
                hop_b_start_trigger=early in ("start", "quad", "quad8"),
                publish_trigger_max_rows={"quad": 48, "quad8": 8}.get(early, 0),
            )
            for early in ("end", "poll", "start", "quad", "quad8")
        }
        # hop B up to the routed pair's 48 rows, with the row-tile scale variant
        wide = {
            early: AllReduceMHC(
                hidden_size=HIDDEN,
                hc_mult=HC,
                max_num_tokens=48,
                top_k=TOP_K,
                device=device,
                group=tp.device_group,
                quantize_layer_input=True,
                residual_only=True,
                early_trigger=early == "poll",
                hop_b_start_trigger=early in ("start", "quad", "quad8"),
                row_tile_layer_input=True,
                publish_trigger_max_rows={"quad": 48, "quad8": 8}.get(early, 0),
            )
            for early in ("end", "poll", "start", "quad", "quad8")
        }
    yield SimpleNamespace(
        rank=rank,
        tp=world_size,
        device=device,
        group=tp.device_group,
        cpu_group=tp.cpu_group,
        hops=hops,
        wide=wide,
    )
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


def _inputs(rows, seed, rank, device, shared_zero=False, ep_of=None):
    """One hop's inputs: routing, mixes and norm weight equal on every rank (as in
    TP), GEMM2 rows, shared expert and hop A's x per rank; ~15% of a rank's routes
    point at an expert it does not hold, or with ``ep_of`` (rank -> EP rank) the
    routes of the other EP half's experts."""
    common = torch.Generator().manual_seed(seed)
    local = torch.Generator().manual_seed(seed * 31 + rank + 1)
    padded = rows * TOP_K + 8
    perm = torch.randperm(padded, generator=common)[: rows * TOP_K].view(rows, TOP_K)
    held = torch.rand(rows, TOP_K, generator=local) >= 0.15
    if ep_of is not None:
        routed = torch.Generator().manual_seed(seed + 7919)
        experts = torch.randint(0, EXPERTS, (rows, TOP_K), generator=routed)
        held = experts // (EXPERTS // EP) == ep_of(rank)
    shared = torch.randn(rows, HIDDEN, generator=local) * 0.5
    tensors = dict(
        idx=torch.where(held, perm, -1).int(),
        weights=torch.rand(rows, TOP_K, generator=common),
        gemm2=torch.randn(padded, HIDDEN, generator=local).bfloat16(),
        shared=(torch.zeros_like(shared) if shared_zero else shared).bfloat16(),
        x=torch.randn(rows, HIDDEN, generator=local).bfloat16(),
        residual=torch.randn(rows, HC, HIDDEN, generator=common).bfloat16(),
        post=torch.rand(rows, HC, generator=common),
        comb=torch.randn(rows, HC, HC, generator=common) * 0.1,
        pre=torch.rand(rows, HC, generator=common),
        norm_weight=torch.randn(HIDDEN, generator=common).bfloat16(),
    )
    return SimpleNamespace(rows=rows, **{k: v.to(device) for k, v in tensors.items()})


def _mxfp8_buffers(rows, device):
    from sgl_nvidia_mega.deepseek_v41.sm103.all_reduce_mhc import mxfp8_scale_bytes

    return (
        torch.empty(rows, HIDDEN, dtype=torch.float8_e4m3fn, device=device),
        torch.empty(mxfp8_scale_bytes(rows, HIDDEN), dtype=torch.uint8, device=device),
    )


def _run(hop, kind, inp):
    """One hop of ``kind``: its outputs, MXFP8 buffers last."""
    mix = (inp.residual, inp.post, inp.comb)
    if kind == "residual":
        return hop.finalize(
            inp.gemm2, inp.weights, inp.idx, inp.shared, *mix, None, None, EPS
        )[:1]
    mx = _mxfp8_buffers(inp.rows, inp.x.device) if kind.endswith("quant") else None
    norm = (inp.pre, inp.norm_weight, EPS)
    if kind.startswith("a"):
        out = hop(inp.x, *mix, *norm, mxfp8_out=mx)
    else:
        out = hop.finalize(
            inp.gemm2, inp.weights, inp.idx, inp.shared, *mix, *norm, mxfp8_out=mx
        )
    return tuple(out) + (mx or ())


def _sglang(kind, inp, tp):
    from sglang.kernels.ops.communication import all_reduce_mhc as sgl

    routing = (inp.gemm2, inp.idx.flatten(), inp.weights, TOP_K, inp.shared)
    mix = (inp.residual, inp.post, inp.comb)
    if kind == "residual":
        return sgl.moe_finalize_all_reduce_mhc_post(*routing, *mix, world_size=tp)[1:]
    return sgl.moe_finalize_all_reduce_mhc_post_combine_norm_quant(
        *routing, *mix, inp.pre, inp.norm_weight, EPS, world_size=tp
    )[1:]


def _local_finalize(inp):
    rows = inp.gemm2.float()[inp.idx.clamp(min=0).long()]
    weights = torch.where(inp.idx >= 0, inp.weights, 0.0)
    routed = (rows * weights.unsqueeze(-1)).sum(1).bfloat16()
    return (routed.float() + inp.shared.float()).bfloat16()


def _torch_boundary(x, inp, group):
    y = x.float()
    dist.all_reduce(y, group=group)
    y = y.bfloat16().float()
    streams = inp.post.unsqueeze(-1) * y.unsqueeze(1) + (
        inp.comb.unsqueeze(-1) * inp.residual.float().unsqueeze(2)
    ).sum(dim=1)
    streams = streams.bfloat16()
    collapsed = (inp.pre.unsqueeze(-1) * streams.float()).sum(dim=1)
    normed = collapsed * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + EPS)
    return streams, (normed * inp.norm_weight.float()).bfloat16()


def _check_torch(kind, inp, out, group):
    x = inp.x if kind.startswith("a") else _local_finalize(inp)
    streams, normed = _torch_boundary(x, inp, group)
    torch.testing.assert_close(out[0].float(), streams.float(), atol=3e-2, rtol=2e-2)
    if kind != "residual":
        torch.testing.assert_close(out[1].float(), normed.float(), atol=6e-2, rtol=3e-2)


def _same(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _ordered(t):
    """Integers whose distance is the ULP distance (sign-magnitude floats)."""
    if t.dtype == torch.bfloat16:
        bits = t.view(torch.int16).int()
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)
    bits = t.view(torch.uint8).int()
    return torch.where(bits >= 0x80, -(bits & 0x7F), bits)


def _diff(a, b, scales=False):
    """(fraction of equal elements, max ULPs where |b| >= 1e-3 or max scale
    exponent step, max abs)."""
    if scales:
        delta = (a.int() - b.int()).abs()
        return (delta == 0).float().mean().item(), delta.max().item(), 0.0
    ulps = (_ordered(a) - _ordered(b)).abs()
    abs_err = (a.float() - b.float()).abs().max().item()
    ulps_big = torch.where(b.float().abs() >= 1e-3, ulps, 0).max().item()
    return (ulps == 0).float().mean().item(), ulps_big, abs_err


def _ep_of(tp):
    """MoE TP4 x EP2 at TP8: ranks 0-3 hold the first half of the experts."""
    return lambda rank: rank // (tp // EP)


@torch.inference_mode()
@pytest.mark.parametrize("routing", ["random", "ep2"])
@pytest.mark.parametrize("shared_zero", [False, True])
def test_finalize_matches_sglang_one_shot(env, shared_zero, routing):
    from flashinfer import mxfp8_quantize

    if routing == "ep2" and env.tp != 8:
        pytest.skip("MoE TP4 x EP2 is a TP8 layout")
    ep_of = _ep_of(env.tp) if routing == "ep2" else None
    hop = env.hops["end"]
    worst = {}
    for rows in ROWS:
        inp = _inputs(rows, 100 + rows, env.rank, env.device, shared_zero, ep_of)
        if ep_of is not None:
            # the EP pair (rank, rank + 4) holds every slot exactly once
            held = (inp.idx >= 0).int()
            pair = [torch.empty_like(held) for _ in range(env.tp)]
            dist.all_gather(pair, held, group=env.group)
            peer = (env.rank + env.tp // EP) % env.tp
            assert torch.equal(held + pair[peer], torch.ones_like(held))
        streams, normed, values, scales = _run(hop, "quant", inp)
        # The MXFP8 output is FlashInfer's quantize of the BF16 row, bit for bit.
        fi_values, fi_scales = mxfp8_quantize(normed, is_sf_swizzled_layout=True)
        assert _same(values, fi_values), f"MXFP8 values differ: rows={rows}"
        assert _same(scales, fi_scales.reshape(-1)), f"scales differ: rows={rows}"
        # The quantized, unquantized and residual-only variants agree bitwise.
        plain = _run(hop, "norm", inp)
        residual_only = _run(hop, "residual", inp)
        assert _same(plain[0], streams) and _same(plain[1], normed)
        assert _same(residual_only[0], streams)
        _check_torch("quant", inp, (streams, normed), env.group)

        sgl_quant = _sglang("quant", inp, env.tp)
        sgl_streams = _sglang("residual", inp, env.tp)[0]
        torch.cuda.synchronize()
        assert _same(sgl_streams, sgl_quant[0])
        stats = {
            "streams": _diff(streams, sgl_quant[0]),
            "normed": _diff(normed, sgl_quant[1]),
            "values": _diff(values, sgl_quant[2]),
            "scales": _diff(scales, sgl_quant[3], scales=True),
        }
        for name, (equal, ulps, abs_err) in stats.items():
            print(
                f"HOPB r{env.rank} {routing} sglang shared_zero={shared_zero} "
                f"rows={rows} {name}: equal={equal:.6f} max_ulp={ulps} max_abs={abs_err:.4g}"
            )
            prev = worst.get(name, (1.0, 0, 0.0))
            worst[name] = (
                min(prev[0], equal),
                max(prev[1], ulps),
                max(prev[2], abs_err),
            )
    print(f"HOPB r{env.rank} {routing} sglang shared_zero={shared_zero} worst={worst}")
    # The mHC post rounds one product in another order than SGLang's: a few
    # stream values per 10^5 differ by an ULP, which moves their row's RMS.
    assert worst["streams"][0] >= 0.9999, worst
    assert worst["streams"][1] <= NUMERICS.tolerances["hop_b.residual.ulp"], worst
    assert worst["normed"][0] >= 0.999, worst
    assert worst["normed"][1] <= NUMERICS.tolerances["hop_b.normed.ulp"], worst
    assert worst["values"][1] <= NUMERICS.tolerances["hop_b.mxfp8.ulp"], worst
    assert worst["scales"][2] <= NUMERICS.tolerances["hop_b.mxfp8_scales.max_abs"], (
        worst
    )


def _interleaved_plan(steps):
    plan = []
    for step in range(steps):
        rows = ROWS[step % len(ROWS)]
        kinds = ("a", "quant", "residual", "a_quant", "norm")
        plan += [(kind, rows, step % 2) for kind in kinds[: 3 + step % 3]]
    return plan


@torch.inference_mode()
@pytest.mark.parametrize("early", ["end"])
def test_hop_a_and_finalize_interleave_on_one_instance(env, early):
    hop = env.hops[early]
    inputs = {
        (rows, s): _inputs(rows, 300 + 10 * rows + s, env.rank, env.device)
        for rows in ROWS
        for s in (0, 1)
    }
    plan = _interleaved_plan(20)
    assert len(plan) >= 50 and len(plan) % 3

    def run(plan, late=None):
        outs = []
        for kind, rows, s in plan:
            if late == env.rank:
                torch.cuda._sleep(20_000)
            outs.append(_run(hop, kind, inputs[rows, s]))
        return outs

    # Reference: one hop at a time, every rank in step.
    reference = []
    for item in plan:
        reference += run([item])
        torch.cuda.synchronize()
        dist.barrier(group=env.cpu_group)
    for (kind, rows, s), out in zip(plan, reference):
        _check_torch(kind, inputs[rows, s], out, env.group)

    def compare(outs, label):
        for index, (got, ref) in enumerate(zip(outs, reference[: len(outs)])):
            for a, b in zip(got, ref):
                assert _same(a, b), f"{label}: step {index} {plan[index]}"

    for late in (None, *range(env.tp)):
        outs = run(plan, late)
        torch.cuda.synchronize()
        compare(outs, f"back-to-back late={late}")

    # A graph of 22 hops (not a multiple of the 3 generations), replayed.
    sub = plan[:22]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        captured = run(sub)
    torch.cuda.current_stream().wait_stream(stream)
    for replay in range(20):
        graph.replay()
        torch.cuda.synchronize()
        compare(captured, f"graph replay {replay}")
        # Eager hops between replays move the generation the graph starts from.
        if replay % 4 == 3:
            compare(run(plan[:1]), "eager between replays")
    torch.cuda.synchronize()


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,), timeout=2400)


@torch.inference_mode()
@pytest.mark.parametrize("early", ["end", "poll", "start", "quad"])
def test_row_tile_finalize_relays_the_swizzled_scales(env, early):
    """Hop B past 8 rows for wqkv_a's wide low-latency GEMM: the same streams, row and
    MXFP8 values as the 128x4 variant, each row's scales at its tile's atom, and the
    last tile's rows past ``rows`` zero though the scale buffer starts as 0xFF."""
    from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll

    hop = env.wide[early]
    scale_cols = HIDDEN // 32
    for rows in (9, 12, 20, 24, 40, 48):
        inp = _inputs(rows, 300 + rows, env.rank, env.device)
        streams, normed, values, scales = _run(hop, "quant", inp)
        tiled = mxfp8_ll.RowTileMxfp8(
            torch.empty(rows, HIDDEN, dtype=torch.float8_e4m3fn, device=env.device),
            torch.full(
                (mxfp8_ll.scale_bytes(rows, HIDDEN),),
                0xFF,
                dtype=torch.uint8,
                device=env.device,
            ),
        )
        mix = (inp.residual, inp.post, inp.comb)
        rt_streams, rt_normed = hop.finalize(
            inp.gemm2,
            inp.weights,
            inp.idx,
            inp.shared,
            *mix,
            inp.pre,
            inp.norm_weight,
            EPS,
            mxfp8_out=tiled,
        )
        assert _same(rt_streams, streams) and _same(rt_normed, normed), rows
        assert _same(tiled.quant, values), f"values differ: rows={rows}"
        g = torch.arange(scale_cols, device=env.device)
        padded = (rows + 7) // 8 * 8
        r = torch.arange(padded, device=env.device)[:, None]
        at = ((r // 8) * (scale_cols // 4) + g // 4) * 512 + (r % 8) * 16 + g % 4
        swz = (g // 4) * 512 + ((r[:rows] % 32) * 4 + (r[:rows] // 32) % 4) * 4 + g % 4
        assert torch.equal(tiled.scale[at[:rows]], scales[swz]), f"scales: rows={rows}"
        assert not tiled.scale[at[rows:]].any(), f"pad rows: rows={rows}"
    torch.cuda.synchronize()


@torch.inference_mode()
@pytest.mark.parametrize("early", ["end", "poll", "start", "quad"])
def test_wide_mailbox_changes_no_bit_up_to_8_rows(env, early):
    """The 48-row mailbox serves batch 1 exactly as the 16-row one: hop A and both
    finalize variants at 1-8 rows, bitwise."""
    for rows in range(1, 9):
        inp = _inputs(rows, 500 + rows, env.rank, env.device)
        for kind in ("aquant", "norm", "quant", "residual"):
            narrow = _run(env.hops[early], kind, inp)
            wide = _run(env.wide[early], kind, inp)
            assert all(_same(a, b) for a, b in zip(narrow, wide)), (kind, rows)
    torch.cuda.synchronize()


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,), timeout=2400)
