# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slots target_lm_head_mxfp8 and draft_lm_head_mxfp8 on CPU: the quantizer's rules
(scale, rounding, swizzle) and the wiring of both heads onto one MXFP8 copy. The
kernel itself is in test/kernels/test_deepseek_v41_lm_head_mxfp8.py."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod

from sgl_nvidia_mega.deepseek_v41 import logits, plan, runtime, sampler
from sgl_nvidia_mega.deepseek_v41.sm103 import lm_head_mxfp8 as lh

DIM, VOCAB = 256, 200


def _weight(n=VOCAB, k=DIM, seed=0):
    torch.manual_seed(seed)
    return (torch.randn(n, k) * 0.02).to(torch.bfloat16)


# ------------------------------------------------------------------- the quantizer


def test_each_block_takes_the_smallest_scale_that_fits_448():
    w = _weight()
    w[1, :32] = 0
    w[2, 32:64] = 448.0  # exactly 448 * 2^0
    w[3, 64:96] = 456.0  # just past 448 (449 is 448 in BF16)
    q = lh.quantize(w)
    e = lh.unswizzle(q.scales, *q.shape).to(torch.int64) - 127
    amax = w.float().view(VOCAB, -1, lh.BLOCK).abs().amax(-1)
    nonzero = amax > 0
    assert torch.all(amax[nonzero] <= 448 * torch.exp2(e[nonzero].double()))
    assert torch.all(amax[nonzero] > 448 * torch.exp2(e[nonzero].double() - 1))
    assert e[2, 1] == 0 and e[3, 2] == 1
    assert q.codes.float().abs().max() <= lh.E4M3_MAX
    assert torch.all(q.codes[1, :32].float() == 0)


def test_codes_round_to_the_nearest_e4m3_value():
    w = _weight()
    q = lh.quantize(w)
    e = lh.unswizzle(q.scales, *q.shape).to(torch.float32) - 127
    scaled = torch.ldexp(w.float().view(VOCAB, -1, lh.BLOCK), -e[..., None])
    codes = q.codes.float().view_as(scaled)
    # an E4M3 step at |v|: 2^(floor(log2 |v|) - 3), 2^-9 among the subnormals
    step = torch.exp2((torch.floor(torch.log2(scaled.abs().clamp_min(2**-6))) - 3))
    assert torch.all((scaled - codes).abs() <= step / 2)
    assert torch.equal(scaled.to(torch.float8_e4m3fn).float(), codes)


def test_dequantize_is_exact_and_requantizing_keeps_the_values():
    q = lh.quantize(_weight())
    dq = lh.dequantize(q)
    # a block whose peak rounded down to 224 may take the next smaller scale: other
    # codes, the same values
    assert torch.equal(lh.dequantize(lh.quantize(dq)), dq)
    e = lh.unswizzle(q.scales, *q.shape).to(torch.float32) - 127
    exact = torch.ldexp(q.codes.float().view(VOCAB, -1, lh.BLOCK), e[..., None])
    assert torch.equal(dq.float(), exact.view(VOCAB, DIM))


@pytest.mark.parametrize("n", [128, 200, 384])
def test_the_swizzle_is_128x4_and_unswizzles(n):
    kb = 512 // lh.BLOCK
    scales = torch.randint(0, 255, (-(-n // 128) * 128, kb), dtype=torch.uint8)
    flat = lh.swizzle(scales)
    for r, b in [(0, 0), (31, 3), (32, 0), (127, 15), (n - 1, 5)]:
        offset = (
            ((r // 128) * (kb // 4) + b // 4) * 512 + (r % 32) * 16 + (r // 32 % 4) * 4
        ) + b % 4
        assert flat[offset] == scales[r, b]
    assert torch.equal(lh.unswizzle(flat, n, 512), scales[:n])


def test_the_kernel_shape_rules():
    assert lh.unsupported_reason(_weight()) == ""
    assert "multiple of 128" in lh.unsupported_reason(_weight(k=192))
    assert "BF16" in lh.unsupported_reason(_weight().float())
    big = lh.quantize(_weight() * 1e6)
    assert lh.scale_reason(big) and not lh.scale_reason(lh.quantize(_weight()))
    # one wave of tiles a CTA, whatever the shard: TP4 / TP8 / TP16
    assert lh.split_for(32320, 152) == 2
    assert lh.split_for(16160, 152) == 2
    assert lh.split_for(8080, 152) == 4  # 505 tiles: 4 a CTA
    assert lh.variant(16160, 5120, fp8=True, sms=152) == lh.Variant(5120, True, 2, 4)


def test_one_copy_per_weight():
    w = _weight()
    assert lh.shared(w) is lh.shared(w)
    assert lh.shared(_weight(seed=1)) is not lh.shared(w)


# --------------------------------------------------------------- the slots' wiring


@pytest.fixture
def stream(monkeypatch):
    """The MXFP8 stream on CPU (x @ dequantize(q).T); records each call's rows."""
    calls = []

    def matmul(x, q, *, out=None):
        calls.append(x.shape[0])
        return (x.float() @ lh.dequantize(q).float().T).to(torch.bfloat16)

    monkeypatch.setattr(lh, "matmul", matmul)
    monkeypatch.setattr(lh, "precompile", lambda *a, **k: None)
    monkeypatch.setattr(logits, "low_m_lm_head_supported", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.autotuner",
        SimpleNamespace(autotune=lambda on: nullcontext()),
    )
    lh._SHARED.clear()
    yield calls
    lh._SHARED.clear()


def _model():
    proc = LogitsProcessor.__new__(LogitsProcessor)
    torch.nn.Module.__init__(proc)
    proc.use_fp32_lm_head, proc.rl_on_policy_target = False, None
    head = SimpleNamespace(weight=_weight(), quant_method=UnquantizedEmbeddingMethod())
    return SimpleNamespace(lm_head=head, logits_processor=proc)


def test_the_slots_are_opt_in_and_need_their_low_m_heads(monkeypatch):
    for slot, needs in (
        ("target_lm_head_mxfp8", "target_lm_head"),
        ("draft_lm_head_mxfp8", "draft_lm_head"),
    ):
        assert slot in runtime.OPT_IN_SLOTS and slot not in runtime.SHIPPED_SLOTS
        monkeypatch.setattr(runtime, "SLOTS", frozenset({slot}))
        assert any(f"needs slot {needs!r}" in r for r in runtime.slot_reasons())
    assert plan.MEASUREMENT_ONLY_SLOTS == {"draft_lm_head_mxfp8"}


@pytest.mark.parametrize("rows", [1, 5, 6, 8, 9, 12])
def test_the_target_streams_1_to_8_rows_and_sglangs_matmul_above(stream, rows):
    model = _model()
    w = model.lm_head.weight
    head = logits.TargetLmHead(model.lm_head, mxfp8=lh.shared(w))
    x = torch.randn(rows, DIM).to(torch.bfloat16)
    ours = model.logits_processor._compute_lm_head(x, head)
    theirs = model.logits_processor._compute_lm_head(x, model.lm_head)
    if rows <= 8:
        assert stream == [rows]
        want = (x.float() @ lh.dequantize(lh.shared(w)).float().T).to(torch.bfloat16)
        assert torch.equal(ours, want)
    else:
        assert stream == [] and torch.equal(ours, theirs)


def test_build_streams_with_the_slot_and_checks_against_sglang(stream, monkeypatch):
    model = _model()
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"target_lm_head"}))
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(mm_bf16=lambda x, w, **k: (x.float() @ w.float()).bfloat16()),
    )
    assert logits.build(model, None, None).quant_method.mxfp8 is None
    monkeypatch.setattr(
        runtime, "SLOTS", frozenset({"target_lm_head", "target_lm_head_mxfp8"})
    )
    head = logits.build(model, None, None)
    assert head.quant_method.mxfp8 is lh.shared(model.lm_head.weight)
    assert stream == list(logits.CHECK_ROWS)
    assert runtime.built_slots(
        {},
        {
            "moe": None,
            "attention": {},
            "embedding": None,
            "engram": SimpleNamespace(prefetch=None),
            "logits": head,
        },
        SimpleNamespace(routed=False),
    )["target_lm_head_mxfp8"]


def test_build_refuses_a_stream_outside_the_tolerance(stream, monkeypatch, caplog):
    model = _model()
    monkeypatch.setattr(
        runtime, "SLOTS", frozenset({"target_lm_head", "target_lm_head_mxfp8"})
    )
    monkeypatch.setattr(
        lh,
        "matmul",
        lambda x, q, out=None: torch.zeros(x.shape[0], VOCAB, dtype=torch.bfloat16),
    )
    assert logits.build(model, None, None) is None
    assert "differ by" in caplog.text


def test_the_draft_streams_1_to_8_rows_on_the_targets_copy(stream):
    w = _weight()
    target = logits.TargetLmHead(SimpleNamespace(weight=w), mxfp8=logits.mxfp8_copy(w))
    copy = sampler.draft_mxfp8_copy(
        w, frozenset({"draft_lm_head", "draft_lm_head_mxfp8"})
    )
    assert copy is target.quant_method.mxfp8
    assert sampler.draft_mxfp8_copy(w, frozenset({"draft_lm_head"})) is None
    draft = sampler.DraftLmHead(
        SimpleNamespace(weight=w), low_m=True, pinned=True, mxfp8=copy
    )
    assert [draft.kind(r) for r in (5, 8, 10, 20)] == [
        "MXFP8 stream",
        "MXFP8 stream",
        "low-M",
        "SGLang's",
    ]
    draft(torch.randn(5, DIM).to(torch.bfloat16))
    assert stream == [5]
    # no low-M head, no stream: SGLang's projection at every row count
    off = sampler.DraftLmHead(SimpleNamespace(weight=w), low_m=False, mxfp8=copy)
    assert off.mxfp8 is None and off.kind(5) == "SGLang's"


def test_ranks_that_disagree_on_the_stream_are_reported():
    assert sampler.mismatch_message([(1, True, True), (1, True, True)]) == ""
    message = sampler.mismatch_message([(1, True, True), (1, True, False)])
    assert "rank 1: batch 1-1, low-M LM head True, MXFP8 stream False" in message
