# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot draft_hop on CPU: the draft's posts take the plugin's Lamport hop only at
attention TP8 with MoE TP4 x EP2 (where SGLang has no fused post), every other layout
keeps its posts as before, and the startup summary names each requested draft slot,
built or declined with the reason, at every TP."""

import logging
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

pytest.importorskip("sglang.srt.models.deepseek_v4_mhc")

from sgl_nvidia_mega.deepseek_v41 import draft_slots  # noqa: E402
from sgl_nvidia_mega.deepseek_v41.boundary import boundaries, draft_hop  # noqa: E402

PARALLEL = "sglang.srt.runtime_context.get_parallel"


def _parallel(tp, ep, world=None):
    group = SimpleNamespace(world_size=world or tp, device_group=None, cpu_group=None)
    return SimpleNamespace(tp_size=tp, moe_ep_size=ep, tp_group=group)


def _stage(layer_id, top_k=3):
    return SimpleNamespace(
        layer_id=layer_id, mlp=SimpleNamespace(experts=SimpleNamespace(top_k=top_k))
    )


# ------------------------------------------------------------------------------- gate


@pytest.mark.parametrize(
    "tp, ep, served",
    [(4, 1, False), (2, 1, False), (8, 2, True), (8, 1, False), (16, 2, False)],
)
def test_the_hop_serves_only_where_sglang_has_no_fused_post(tp, ep, served):
    with mock.patch(PARALLEL, return_value=_parallel(tp, ep)):
        why = draft_hop.layout_reason()
        fuses = draft_hop.sglang_fuses()
    assert (why == "") == served
    # SGLang's fused posts: TP2 / TP4 (deepseek_v4_mhc._FUSED_MHC_TP_SIZES)
    assert fuses == (tp in (2, 4))
    if fuses:
        assert why == f"SGLang's fused all-reduce + mHC posts serve at TP{tp}"


def _no_collective(*args, **kwargs):
    raise AssertionError("a declined build must not reach the TP group")


def test_tp4_builds_nothing_and_meets_no_rank():
    with (
        mock.patch(PARALLEL, return_value=_parallel(4, 1)),
        mock.patch.object(torch.distributed, "all_gather_object", _no_collective),
        mock.patch.object(draft_hop, "_mailbox", _no_collective),
    ):
        hop, why = draft_hop.build([_stage(0)], deadline_s=1.0, max_rows=48)
    assert hop is None and "TP4" in why


def _gather(errors):
    def gather(out, obj, group=None):
        out[:] = [obj if e is None else e for e in errors]

    return gather


@pytest.fixture
def tp8():
    with mock.patch(PARALLEL, return_value=_parallel(8, 2)):
        yield


def test_tp8_builds_the_draft_mailbox_at_the_draft_top_k(tp8):
    made = []

    def mailbox(stage, top_k, group, deadline_s, max_rows):
        made.append((stage.layer_id, top_k, deadline_s, max_rows))
        return SimpleNamespace(capacity=max_rows)

    with (
        mock.patch.object(boundaries, "_fusable", return_value=True),
        mock.patch.object(draft_hop, "_mailbox", mailbox),
        mock.patch.object(torch.distributed, "all_gather_object", _gather([None] * 8)),
    ):
        hop, why = draft_hop.build(
            [_stage(43), _stage(44), _stage(45)], deadline_s=2.5, max_rows=48
        )
    assert why == "" and hop.capacity == 48
    assert made == [(43, 3, 2.5, 48)]


def test_tp8_declines_a_stage_off_sglangs_fuse_gate(tp8):
    def fusable(stage, kind, fuses):
        return not (stage.layer_id == 44 and kind == "ffn")

    with (
        mock.patch.object(boundaries, "_fusable", fusable),
        mock.patch.object(draft_hop, "_mailbox", _no_collective),
    ):
        hop, why = draft_hop.build(
            [_stage(43), _stage(44)], deadline_s=1.0, max_rows=48
        )
    assert hop is None and why.startswith("stage 44: SGLang's ffn fuse gate")


def test_a_failed_mailbox_on_one_rank_raises_on_every_rank(tp8):
    errors = [None] * 8
    errors[5] = "RuntimeError: NVLink multicast mapping is unavailable"
    with (
        mock.patch.object(boundaries, "_fusable", return_value=True),
        mock.patch.object(draft_hop, "_mailbox", return_value=object()),
        mock.patch.object(torch.distributed, "all_gather_object", _gather(errors)),
        pytest.raises(RuntimeError, match="rank 5: RuntimeError: NVLink multicast"),
    ):
        draft_hop.build([_stage(43)], deadline_s=1.0, max_rows=48)


# ---------------------------------------------------------------------------- posts


class _AttnOutput:
    def __init__(self, partial):
        self.partial = partial


def _mhc(calls):
    return SimpleNamespace(
        AttnOutput=_AttnOutput,
        HcState=lambda *a: ("state",) + a,
        HcNormed=lambda x: ("normed", x),
        is_deferred_finalize=lambda routed: True,
        run_attn_post=lambda *a, **k: calls.append("sglang attn post") or "sglang",
        run_moe_post=lambda *a, **k: calls.append("sglang moe post") or "sglang",
    )


class _Hop:
    capacity = 48

    def __init__(self, calls):
        self.calls = calls

    def collect(self, partial, *rest):
        self.calls.append(("collect", partial.shape[0]))
        return "streams", "normed"

    def finalize(self, *args, rows, mxfp8_out=None):
        self.calls.append(("finalize", rows, args[7:10]))
        return "streams", None


def _boundaries(calls, hop):
    b = boundaries.Boundaries.__new__(boundaries.Boundaries)
    b._mhc = _mhc(calls)
    b.hop_a = b.two_shot = None
    b.hop_b, b.cross_layer, b.prenorm = False, True, False
    b.draft_hop = hop
    b._mhc_fusions = True
    b._served = set()
    b._ffn_next = {
        7: SimpleNamespace(norm=SimpleNamespace(weight="w", variance_epsilon=1e-6))
    }
    b._attn_next = {7: None}
    b._next_layer, b.row_tile_layers = {}, frozenset()
    b._mix_stats = lambda hc, streams, stream: ("pre", "post", "comb")
    return b


def _layer():
    return SimpleNamespace(
        layer_id=7,
        attn_hc="attn_hc",
        ffn_hc="ffn_hc",
        hc_cfg=SimpleNamespace(hidden=5120),
        self_attn=SimpleNamespace(attn_tp_size=8),
        mlp=SimpleNamespace(tp_size=8),
    )


def _moe_out(rows, *, weights=torch.float32, replicated=False):
    from sglang.srt.models.deepseek_v2 import MoEOutput

    routed = SimpleNamespace(
        gemm2_out="gemm2",
        expanded_idx_to_permuted_idx=torch.zeros(rows, 3, dtype=torch.int32),
        expert_weights=torch.zeros(rows, 3, dtype=weights),
        top_k=3,
    )
    return MoEOutput(
        routed=routed,
        shared="shared",
        experts=None,
        routed_scaling_factor=1.0,
        shared_is_replicated=replicated,
    )


def _open(b, rows):
    st = boundaries.BoundaryState(hc=None)
    st.open = boundaries.Open(torch.zeros(rows, 4, 1))
    return st


@pytest.fixture
def fused_gate():
    with (
        mock.patch.object(boundaries, "_fusable", return_value=True),
        mock.patch(
            "sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe."
            "can_fuse_all_reduce",
            return_value=False,
        ),
    ):
        yield


@pytest.mark.parametrize("rows", [5, 10, 45])
def test_both_draft_posts_take_the_hop_at_tp8(fused_gate, rows):
    calls = []
    b = _boundaries(calls, _Hop(calls))
    layer = _layer()
    assert b.defers_sum(layer, "attn", rows) and b.defers_sum(layer, "ffn", rows)
    st = _open(b, rows)
    b.post(layer, "attn", _AttnOutput(torch.zeros(rows, 4)), st)
    assert st.hc == ("state", "streams", "pre", ("normed", "normed"))
    st = _open(b, rows)
    b.post(layer, "ffn", _moe_out(rows), st)
    assert st.hc == ("state", "streams", "pre")
    # the residual-only finalize: no pre, no norm (the draft attention takes no MXFP8)
    assert calls == [("collect", rows), ("finalize", rows, (None, None, 0.0))]


def test_past_the_mailbox_or_off_the_finalize_the_post_stays_sglangs(fused_gate):
    calls = []
    b = _boundaries(calls, _Hop(calls))
    layer = _layer()
    assert not b.defers_sum(layer, "attn", 49)
    st = _open(b, 6)
    b.post(layer, "ffn", _moe_out(6, weights=torch.bfloat16), st)
    st = _open(b, 6)
    b.post(layer, "ffn", _moe_out(6, replicated=True), st)
    assert calls == ["sglang moe post", "sglang moe post"]


def test_without_the_hop_the_posts_dispatch_as_before():
    """TP4: SGLang's fused posts (an AttnOutput into run_attn_post) where its push plane
    takes the rows; the draft hop never enters."""
    calls = []
    b = _boundaries(calls, None)
    layer = _layer()
    with (
        mock.patch.object(boundaries, "_fusable", return_value=True),
        mock.patch(
            "sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe."
            "can_fuse_all_reduce",
            return_value=True,
        ),
    ):
        st = _open(b, 5)
        b.post(layer, "attn", _AttnOutput(torch.zeros(5, 4)), st)
        st = _open(b, 5)
        b.post(layer, "ffn", _moe_out(5), st)
    assert calls == ["sglang attn post", "sglang moe post"]
    assert not any(hop.startswith("unfused") for hop, _ in b._served)


def test_an_unfused_post_says_so_once(caplog):
    calls = []
    b = _boundaries(calls, None)
    layer = _layer()
    reduce = SimpleNamespace(all_reduce=lambda x: x)
    with (
        mock.patch.object(boundaries, "_fusable", return_value=True),
        mock.patch(
            "sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe."
            "can_fuse_all_reduce",
            return_value=False,
        ),
        mock.patch(PARALLEL, return_value=SimpleNamespace(attn_tp_group=reduce)),
        caplog.at_level(logging.WARNING, logger=boundaries.logger.name),
    ):
        for _ in range(3):
            st = _open(b, 5)
            b.post(layer, "attn", _AttnOutput(torch.zeros(5, 4)), st)
            st = _open(b, 5)
            b.post(layer, "ffn", _moe_out(5), st)
    unfused = [r.message for r in caplog.records if "unfused" in r.message]
    assert len(unfused) == 2, unfused
    assert "attention post at 5 rows" in unfused[0] and "MoE post at 5" in unfused[1]


# -------------------------------------------------------------------- startup summary

REQUESTED = frozenset(
    {"markov", "draft_lm_head", "draft", "draft_moe", "draft_gate", "draft_tail",
     "draft_post", "draft_seam", "draft_prenorm"}
)  # fmt: skip


def _built(hop=None, hop_reason="", moe_slots=None):
    from sgl_nvidia_mega.deepseek_v41.moe.routed import Bound

    bound = SimpleNamespace(
        draft_hop=hop,
        prenorm=False,
        cross_layer=True,
        _attn_mxfp8={43: False},
        _fuses=lambda stage: True,
    )
    return SimpleNamespace(
        draft_sampler=object(),
        low_m=True,
        widths=(5, 10),
        _view=SimpleNamespace(model=SimpleNamespace(layers=[_layer()])),
        _bound={
            "boundary": bound,
            "moe": Bound(slots=moe_slots or {}, reason="" if moe_slots else "boom"),
        },
        _hop_reason=hop_reason,
        _attn={},
        _binding=None,
    )


def _states(built, slots, *, tp, sglang_fuses):
    with (
        mock.patch(PARALLEL, return_value=_parallel(tp, 2 if tp == 8 else 1)),
        mock.patch.object(draft_slots, "_sglang_fuses", return_value=sglang_fuses),
    ):
        return draft_slots.states(built, slots)


def test_tp4_summary_names_sglangs_fused_posts():
    slots = REQUESTED | {"draft_hop"}
    table = _states(
        _built(hop_reason="SGLang's fused all-reduce + mHC posts serve at TP4"),
        slots,
        tp=4,
        sglang_fuses=True,
    )
    assert set(table) == slots
    assert table["draft_post"] == (
        draft_slots.BUILT,
        "SGLang's fused all-reduce + mHC posts from 5 rows",
    )
    assert table["draft_hop"][0] == draft_slots.NOT_NEEDED
    assert table["draft_moe"] == (draft_slots.DECLINED, "boom")
    line = draft_slots.summary(table)
    assert all(name in line for name in slots)


def test_tp8_without_the_hop_declines_draft_post_loudly(caplog):
    built = _built(hop_reason="not requested")
    with caplog.at_level(logging.WARNING, logger=draft_slots.logger.name):
        table = _states(built, REQUESTED, tp=8, sglang_fuses=False)
        draft_slots.warn_declined(table)
    state, why = table["draft_post"]
    assert state == draft_slots.DECLINED
    assert "plain all-reduce + unfused mHC post" in why and "not requested" in why
    assert any("draft_post requested, declined" in r.message for r in caplog.records)


def test_tp8_with_the_hop_builds_draft_post_on_it():
    table = _states(
        _built(hop=SimpleNamespace(capacity=48)),
        REQUESTED | {"draft_hop"},
        tp=8,
        sglang_fuses=False,
    )
    assert table["draft_post"] == (
        draft_slots.BUILT,
        "the plugin's draft hop at 1-48 rows",
    )
    assert table["draft_hop"][0] == draft_slots.BUILT
    assert table["draft_seam"][0] == draft_slots.BUILT  # streams only, said so


def test_a_report_error_is_reported_not_raised():
    built = _built()
    built._bound["boundary"] = None
    table = _states(built, REQUESTED, tp=4, sglang_fuses=True)
    assert table["draft_post"][0] == draft_slots.UNKNOWN


def test_the_proposal_only_draft_reports_its_slots():
    from sgl_nvidia_mega.deepseek_v41.draft import DraftProposal

    built = DraftProposal(None, deadline_s=1.0, slots=frozenset({"markov"}))
    assert "draft slots: markov built" in built.summary()
