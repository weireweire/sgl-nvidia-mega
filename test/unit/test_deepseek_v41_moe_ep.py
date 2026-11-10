# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DeepSeek-V4.1 MoE build at TP8 (MoE TP4 x EP2) on CPU: each EP rank's routed pair
takes the router's expert count and its window's offset as SGLang's FlashInfer routed
MoE does, and the shared expert's TP8 shard (gate_up N 576) takes slot shared_ll."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.moe import routed, shared_ll


class _Method:
    _gemm1_clamp_limit_tensor = None


def _layer(layer_id, *, ep_rank, local, experts=384):
    fused = SimpleNamespace(
        quant_method=_Method(),
        num_local_experts=local,
        num_experts=experts,
        moe_ep_rank=ep_rank,
        top_k=6,
        w13_weight=None,
        w13_weight_scale_inv=None,
        w2_weight=None,
        w2_weight_scale_inv=None,
    )
    mlp = SimpleNamespace(
        experts=fused, alt_stream=object(), num_fused_shared_experts=0
    )
    return SimpleNamespace(layer_id=layer_id, mlp=mlp)


def _build(layers):
    made = []

    class Gemm:
        def __init__(self, *args, **kwargs):
            made.append(kwargs)
            self.gate_topk_kernel = None
            self.shared = None

    group = SimpleNamespace(world_size=1, cpu_group=None)

    def gather(out, obj, group=None):
        out[0] = obj

    with (
        mock.patch(
            "sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe."
            "Mxfp4FlashinferTrtllmMoEMethod",
            _Method,
        ),
        mock.patch(
            "sgl_nvidia_mega.deepseek_v41.sm103.moe.routed_expert_gemm."
            "RoutedExpertGemm",
            Gemm,
        ),
        mock.patch.object(torch.distributed, "all_gather_object", gather),
        mock.patch.object(routed, "MoeSlot", lambda mlp, gemm, gate: gemm),
    ):
        bound = routed.build_layers(layers, group, gate_topk=False)
    return bound, made


@pytest.mark.parametrize(
    "ep_rank, local, offset", [(0, 384, 0), (0, 192, 0), (1, 192, 192)]
)
def test_each_ep_rank_binds_its_expert_window(ep_rank, local, offset):
    bound, made = _build([_layer(3, ep_rank=ep_rank, local=local)])
    assert bound.reason == "" and sorted(bound.slots) == [3]
    assert made[0]["router_experts"] == 384
    assert made[0]["expert_offset"] == offset


def test_the_offset_is_sglangs_local_expert_offset():
    # Mxfp4FlashinferTrtllmMoEMethod passes moe_ep_rank * num_local_experts
    experts = SimpleNamespace(moe_ep_rank=1, num_local_experts=64)
    assert routed.expert_offset(experts) == 64


def _mlp(n, k=5120):
    gate_up = SimpleNamespace(
        weight=torch.empty(n, k, dtype=torch.float8_e4m3fn),
        weight_scale_inv_swizzled=torch.empty(1, dtype=torch.uint8),
    )
    shared = SimpleNamespace(
        gate_up_proj=gate_up,
        down_proj=SimpleNamespace(weight=None),
        swiglu_limit=10.0,
        use_fused_clamp_act_mul=False,
        allow_fused_down=False,
    )
    return SimpleNamespace(shared_experts=shared, num_fused_shared_experts=0)


@pytest.mark.parametrize(
    "n, ok",
    [
        (1152, True),  # TP4: 2 x 576
        (576, True),  # TP8: 2 x 288, half of its last 128-row tile
        (4608, True),  # replicated (SGLANG_SHARED_EXPERT_TP1, refused by the spec)
        (544, False),
    ],
)
def test_shared_ll_takes_the_tp8_shard(n, ok):
    with mock.patch.object(shared_ll.mxfp8_ll, "supported", return_value=True):
        why = shared_ll.why_not(_mlp(n))
    assert (why == "") == ok, why


def _boundary_layer(sglang_fuses: bool, *, shared_tp1=False):
    return SimpleNamespace(
        boundary_pre_from_prev=True,
        _can_fuse_attn_mhc=sglang_fuses,
        _can_fuse_ffn_mhc=sglang_fuses,
        post_attention_layernorm=None,
        ffn_hc=None,
        hc_cfg=None,
        self_attn=SimpleNamespace(attn_tp_size=8),
        mlp=SimpleNamespace(_shared_expert_tp1=shared_tp1),
    )


@pytest.mark.parametrize(
    "tp, ep, sglang_fuses, shared_tp1, want",
    [
        (4, 1, True, False, True),  # TP4: SGLang's own gate, unchanged
        (4, 1, False, False, False),
        (8, 2, False, False, True),  # TP8 + EP2: SGLang's gate less its TP list
        (8, 2, False, True, False),  # a replicated shared expert keeps it off
        (8, 1, False, False, False),
    ],
)
def test_the_post_gate_admits_tp8_ep2_only(tp, ep, sglang_fuses, shared_tp1, want):
    from sgl_nvidia_mega.deepseek_v41.boundary import boundaries

    parallel = SimpleNamespace(tp_size=tp, moe_ep_size=ep)
    with (
        mock.patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
        mock.patch(
            "sglang.srt.layers.moe.get_moe_a2a_backend",
            return_value=SimpleNamespace(is_none=lambda: True),
        ),
        mock.patch(
            "sglang.srt.layers.moe.utils.should_skip_post_experts_all_reduce",
            return_value=False,
        ),
        mock.patch(
            "sglang.srt.models.deepseek_v4_mhc.can_fuse_post", return_value=True
        ),
        mock.patch(
            "sglang.srt.models.deepseek_v4_mhc.make_boundary",
            return_value=SimpleNamespace(norm_fusable=True),
        ),
    ):
        layer = _boundary_layer(sglang_fuses, shared_tp1=shared_tp1)
        assert boundaries._fusable(layer, "ffn", True) is want
        if not shared_tp1:
            assert boundaries._fusable(layer, "attn", True) is want
        assert boundaries._fusable(layer, "ffn", False) is False
