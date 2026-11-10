# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DeepSeek-V4.1 MoE half of a decode step at TP8 (attention TP8, MoE TP4 x EP2) on
two GB300 trays (8 ranks, one NVLink domain), against SGLang's stock MoE at that layout.

Each rank holds its EP half of the experts (ranks 0-3 experts 0-191, ranks 4-7 experts
192-383) at its MoE-TP slice (random weights of its own); the router logits, the rows
and the mixes are the same on every rank, the shared expert's TP partial is the rank's.

- At the routed pair's widths (1-48 rows): the fast gate's ids and weights equal SGLang's
  moe_fused_gate bit for bit, on every rank alike; the pair's GEMM2 rows match
  FlashInfer's routed MoE with SGLang's ``local_expert_offset`` (one BF16 rounding), and
  both drop the other half's slots alike; hop B (the Lamport finalize + all-reduce +
  mHC) of the pair's output against SGLang's one-shot fused finalize of the same
  output (hop B's TP4 tolerances), and the whole chain against SGLang's whole chain
  (FlashInfer's rows into the one-shot kernel) within the torch bound.
- At the two-shot widths (49-384 rows, FlashInfer's routed MoE as SGLang serves them):
  the two-shot MoE post against SGLang's one-shot fused kernel, streams bit for bit, as
  at TP4.

It registers its own SGLang one-shot plane, so it keeps its own process. Run on two trays
with ``torchrun --nnodes 2 --nproc-per-node 4 ... -m pytest
test/kernels/test_deepseek_v41_moe_ep.py -s``.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from sglang.srt.distributed import parallel_state as ps
from sglang.test.test_utils import publish_build_topology

from sgl_nvidia_mega.deepseek_v41.spec import NUMERICS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_deepseek_v41_moe as single  # noqa: E402

from sgl_nvidia_mega.deepseek_v41.sm103.moe import (
    routed_expert_gemm as reg,
)  # noqa: E402

HIDDEN, HC, TOP_K, EPS = 5120, 4, 6, 1e-6
EXPERTS, EP, SCALE = 384, 2, 1.5
PAIR_ROWS = (1, 6, 8, 12, 24, 48)
TWO_SHOT_ROWS = (49, 96, 128, 192, 384)
ONE_SHOT_QUANT_ROWS = 128  # the one-shot quant epilogue addresses one 128-row tile

pytestmark = [
    pytest.mark.skipif(
        "RANK" not in os.environ or os.environ.get("WORLD_SIZE") != "8",
        reason="run on 8 ranks via torchrun (two trays)",
    ),
    pytest.mark.skipif(
        not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
        reason="needs SM100-family GPUs",
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
    from sgl_nvidia_mega.deepseek_v41.sm103.two_shot_mhc import MAX_ROWS, TwoShotMHC

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
    device = torch.device(f"cuda:{local_rank}")
    plane = CustomAllReduceV2(
        tp.cpu_group,
        device,
        max_pull_size=0,
        max_pull_blocks=0,
        max_push_size=4 << 20,
        max_push_blocks=512,
    )
    assert not plane.disabled, "SGLang's one-shot plane does not span the 8 ranks"
    all_reduce_fusion.register_comm(plane.obj)
    register_comm_cleanup(plane)
    ep_rank = rank // (world // EP)
    full = single._shuffled_layer(
        torch.Generator().manual_seed(50 + rank % (world // EP)), EXPERTS // EP
    )
    offset = ep_rank * (EXPERTS // EP)
    with torch.inference_mode():
        hop_b = AllReduceMHC(
            hidden_size=HIDDEN,
            hc_mult=HC,
            max_num_tokens=max(PAIR_ROWS),
            top_k=TOP_K,
            device=device,
            group=tp.device_group,
            quantize_layer_input=True,
            residual_only=True,
        )
        two_shot = TwoShotMHC(
            hidden_size=HIDDEN,
            hc_mult=HC,
            max_num_tokens=MAX_ROWS,
            top_k=TOP_K,
            device=device,
            group=tp.device_group,
            deadline_s=30.0,
        )
        from sgl_nvidia_mega.deepseek_v41.runtime import SLOTS

        # slot gate_route (SGLANG_NVIDIA_MEGAKERNEL_SLOTS): top-k + route in one launch
        pair = reg.RoutedExpertGemm(
            *full,
            top_k=TOP_K,
            max_rows=reg.SERVED_MAX_ROWS,
            gate_topk=True,
            router_experts=EXPERTS,
            expert_offset=offset,
            gate_route="gate_route" in SLOTS,
        )
    yield SimpleNamespace(
        rank=rank,
        world=world,
        device=device,
        group=tp.device_group,
        layer=full,
        offset=offset,
        pair=pair,
        hop_b=hop_b,
        two_shot=two_shot,
    )
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


def _step_inputs(rows, seed, rank, device):
    """The rows, router and mixes alike on every rank; the shared partial per rank."""
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    common = torch.Generator().manual_seed(seed)
    local = torch.Generator().manual_seed(seed * 31 + rank + 1)
    x = torch.randn(rows, HIDDEN, generator=common).bfloat16().to(device)
    x_q, x_s = flashinfer_mxfp8_quantize(x, False, alignment=HIDDEN)
    tensors = dict(
        logits=torch.randn(rows, EXPERTS, generator=common),
        bias=torch.randn(EXPERTS, generator=common) * 0.05,
        shared=(torch.randn(rows, HIDDEN, generator=local) / 8).bfloat16(),
        residual=torch.randn(rows, HC, HIDDEN, generator=common).bfloat16(),
        post=torch.rand(rows, HC, generator=common),
        comb=torch.randn(rows, HC, HC, generator=common) * 0.1,
        pre=torch.rand(rows, HC, generator=common),
        norm_weight=torch.randn(HIDDEN, generator=common).bfloat16(),
    )
    return SimpleNamespace(
        rows=rows,
        x_q=x_q,
        x_s=x_s.view(torch.float8_e4m3fn).reshape(rows, -1),
        num_valid=torch.tensor([rows], dtype=torch.int32, device=device),
        **{k: v.to(device).contiguous() for k, v in tensors.items()},
    )


def _stock_gate(a):
    from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

    w, ids = moe_fused_gate(
        a.logits,
        a.bias,
        topk=TOP_K,
        scoring_func="sqrtsoftplus",
        num_fused_shared_experts=0,
        renormalize=True,
        routed_scaling_factor=SCALE,
        apply_routed_scaling_factor_on_output=True,
        num_token_non_padded=a.num_valid,
        sqrtsoftplus_log1p=True,
    )
    return w, ids.to(torch.int32)


def _stock_routed(env, a, ids, weights):
    """FlashInfer's routed MoE on this rank's experts, as SGLang's EP2 MoE runs it."""
    return single._flashinfer_ep(
        a.x_q, a.x_s, ids, weights, env.layer, num_experts=EXPERTS, offset=env.offset
    )


def _sglang_one_shot(a, gemm2, permuted, weights, world, quant):
    from sglang.kernels.ops.communication import all_reduce_mhc as sgl

    moe = (gemm2, permuted.flatten(), weights, TOP_K, a.shared)
    moe += (a.residual, a.post, a.comb)
    norm = (a.pre, a.norm_weight, EPS)
    if quant:
        return sgl.moe_finalize_all_reduce_mhc_post_combine_norm_quant(
            *moe, *norm, world_size=world
        )[1:]
    return sgl.moe_finalize_all_reduce_mhc_post_combine_norm(
        *moe, *norm, world_size=world
    )[1:]


def _mxfp8_buffers(rows, device, scale_bytes):
    return (
        torch.empty(rows, HIDDEN, dtype=torch.float8_e4m3fn, device=device),
        torch.empty(scale_bytes(rows, HIDDEN), dtype=torch.uint8, device=device),
    )


def _same(a, b):
    return a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def _ulps(a, b):
    ia, ib = (
        a.contiguous().view(torch.int16).int(),
        b.contiguous().view(torch.int16).int(),
    )
    ia = torch.where(ia < 0, -(ia & 0x7FFF), ia)
    ib = torch.where(ib < 0, -(ib & 0x7FFF), ib)
    return (ia - ib).abs()


def _alike_on_every_rank(t, group, world):
    gathered = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(gathered, t.contiguous(), group=group)
    return all(_same(g, gathered[0]) for g in gathered)


def _local(ids, offset):
    return (ids >= offset) & (ids < offset + EXPERTS // EP)


@torch.inference_mode()
def test_pair_widths_gate_route_gemms_hop_b(env):
    from sgl_nvidia_mega.deepseek_v41.sm103.all_reduce_mhc import mxfp8_scale_bytes

    worst = {"streams": 0, "normed": 0, "equal": 1.0}
    for rows in PAIR_ROWS:
        a = _step_inputs(rows, 700 + rows, env.rank, env.device)
        where = f"rank {env.rank} rows={rows}"
        assert env.pair.supports_gate_topk(a.logits, a.bias), where
        if env.pair.gate_route_kernel is not None:
            assert env.pair.serves_gate_route(rows) == (rows <= reg.MAX_ROWS), where
        rows_out, weights, permuted, ids = env.pair.gate_topk_and_gemms(
            a.x_q, a.x_s, a.logits, a.bias, SCALE, a.num_valid
        )
        stock_w, stock_ids = _stock_gate(a)
        want_rows, want_permuted = _stock_routed(env, a, stock_ids, stock_w)
        torch.cuda.synchronize()
        # gate: SGLang's router, bit for bit, the same on all 8 ranks
        assert torch.equal(ids, stock_ids), where
        assert _same(weights, stock_w), where
        assert _alike_on_every_rank(ids, env.group, env.world), where
        assert _alike_on_every_rank(weights, env.group, env.world), where
        # route + GEMMs: the other half's slots dropped alike, kept rows within a rounding
        local = _local(ids, env.offset)
        assert torch.equal(permuted >= 0, local), where
        assert torch.equal(want_permuted >= 0, local), where
        got = rows_out[permuted.clamp(min=0).long()][local].float()
        want = want_rows[want_permuted.clamp(min=0).long()][local].float()
        torch.testing.assert_close(got, want, rtol=2**-7, atol=1e-6)

        # hop B on the pair's output against SGLang's one-shot on the same output
        q, sf = _mxfp8_buffers(rows, env.device, mxfp8_scale_bytes)
        mix = (a.residual, a.post, a.comb)
        streams, normed = env.hop_b.finalize(
            rows_out,
            weights,
            permuted,
            a.shared,
            *mix,
            a.pre,
            a.norm_weight,
            EPS,
            mxfp8_out=(q, sf),
        )
        ref = _sglang_one_shot(a, rows_out, permuted, weights, env.world, quant=True)
        whole = _sglang_one_shot(
            a, want_rows, want_permuted, stock_w, env.world, quant=True
        )
        torch.cuda.synchronize()
        streams_ulps = _ulps(streams, ref[0])
        normed_ulps = _ulps(normed, ref[1])
        big = ref[1].float().abs() >= 1e-3
        worst["streams"] = max(worst["streams"], int(streams_ulps.max()))
        worst["normed"] = max(worst["normed"], int(normed_ulps[big].max()))
        worst["equal"] = min(worst["equal"], (streams_ulps == 0).float().mean().item())
        torch.testing.assert_close(
            streams.float(), whole[0].float(), atol=3e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            normed.float(), whole[1].float(), atol=6e-2, rtol=3e-2
        )
        if env.rank == 0:
            print(
                f"EP2 pair rows={rows} gate_route="
                f"{env.pair.serves_gate_route(rows)}: kept {int(local.sum())}/{local.numel()} slots; "
                f"hop B vs one-shot streams max_ulp={int(streams_ulps.max())}, "
                f"vs SGLang's whole chain max_abs="
                f"{(streams.float() - whole[0].float()).abs().max().item():.3g}",
                flush=True,
            )
    assert worst["equal"] >= 0.9999, worst
    assert worst["streams"] <= NUMERICS.tolerances["hop_b.residual.ulp"], worst
    assert worst["normed"] <= NUMERICS.tolerances["hop_b.normed.ulp"], worst


@torch.inference_mode()
def test_two_shot_widths_against_sglang_one_shot(env):
    for rows in TWO_SHOT_ROWS:
        a = _step_inputs(rows, 900 + rows, env.rank, env.device)
        where = f"rank {env.rank} rows={rows}"
        weights, ids = _stock_gate(a)
        gemm2, permuted = _stock_routed(env, a, ids, weights)
        assert torch.equal(permuted >= 0, _local(ids, env.offset)), where
        moe = (gemm2, permuted.flatten(), weights, TOP_K, a.shared)
        moe += (a.residual, a.post, a.comb)
        quant = rows <= ONE_SHOT_QUANT_ROWS
        mx = (
            _mxfp8_buffers(rows, env.device, lambda m, _: env.two_shot.scale_bytes(m))
            if quant
            else None
        )
        got = env.two_shot.moe_post(
            *moe, a.pre, a.norm_weight, EPS, **({"mxfp8_out": mx} if quant else {})
        )
        one = _sglang_one_shot(a, gemm2, permuted, weights, env.world, quant=quant)
        torch.cuda.synchronize()
        assert _same(got[0], one[0]), f"two-shot streams differ: {where}"
        normed_ulps = int(_ulps(got[1], one[1]).max())
        assert normed_ulps <= NUMERICS.tolerances["two_shot.normed.ulp"], where
        if quant and NUMERICS.tolerances["two_shot.mxfp8.max_abs"] == 0:
            assert _same(mx[0], one[2]) and _same(mx[1], one[3]), where
        if env.rank == 0:
            print(f"EP2 two-shot rows={rows}: streams bitwise", flush=True)
