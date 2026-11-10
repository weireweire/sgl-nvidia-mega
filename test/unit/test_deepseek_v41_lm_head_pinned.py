# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot lm_head_pinned_tactic: both low-M LM heads launch one fixed FlashInfer runner
and tactic per row count, never FlashInfer's autotuner, and fall back to FlashInfer's
default when the pinned tactic is not offered."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41 import logits, runtime, sampler
from sgl_nvidia_mega.deepseek_v41.sampler import pinned

N, K = 96, 64


class _Runner:
    def __init__(self, tactics):
        self.tactics, self.calls = tactics, []

    def supports_inputs(self, inputs):
        return True

    def get_valid_tactics(self, inputs, profile):
        return list(self.tactics)

    def __call__(self, *, inputs, tactic):
        a, b, _, _, out, _ = inputs
        self.calls.append((a.shape[0], tactic))
        out.copy_((a.float() @ b.float()).to(out.dtype))


class WarpSplitK(_Runner):
    pass


class Direct(_Runner):
    pass


@pytest.fixture
def fi(monkeypatch):
    """FlashInfer's cute-dsl runner list on CPU; any autotuner use fails the test."""
    runners = [WarpSplitK([(16, 8, 128, 7, 1)]), Direct([(64, 1, 1), (64, 4, 1)])]
    gemm_base = SimpleNamespace(_cute_dsl_bf16_runners=lambda inputs: runners)

    def no_autotune(*a, **k):
        raise AssertionError("the pinned heads must not tune")

    monkeypatch.setitem(
        sys.modules, "flashinfer.gemm", SimpleNamespace(gemm_base=gemm_base)
    )
    monkeypatch.setitem(sys.modules, "flashinfer.gemm.gemm_base", gemm_base)
    monkeypatch.setitem(
        sys.modules, "flashinfer.autotuner", SimpleNamespace(autotune=no_autotune)
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (10, 3)
    )
    monkeypatch.setattr(
        pinned, "PINNED_TACTICS", {(103, N, K, 1): ("Direct", (64, 4, 1))}
    )
    pinned._choice.cache_clear()
    yield runners
    pinned._choice.cache_clear()


def test_pinned_rows_run_their_tactic_and_others_flashinfers_default(fi, caplog):
    warp, direct = fi
    w = torch.randn(N, K).to(torch.bfloat16)
    for rows in (1, 6, 1):
        x = torch.randn(rows, K).to(torch.bfloat16)
        assert torch.equal(
            pinned.pinned_mm(x, w.t()), (x.float() @ w.t().float()).to(torch.bfloat16)
        )
    assert direct.calls == [(1, (64, 4, 1))] * 2 and warp.calls == [(6, -1)]
    pinned.PINNED_TACTICS[(103, N, K, 2)] = ("Direct", (128, 2, 2))
    pinned.pinned_mm(torch.randn(2, K).to(torch.bfloat16), w.t())
    assert warp.calls[-1] == (2, -1) and "not offered at 2 rows" in caplog.text


def test_both_heads_take_the_pinned_path_untuned(fi):
    warp, direct = fi
    w = torch.randn(N, K).to(torch.bfloat16)
    draft = sampler.DraftLmHead(SimpleNamespace(weight=w), low_m=True, pinned=True)
    target = logits.LowMLmHeadMethod(w, pinned=True)
    draft.tune((1, 6))
    target.tune()
    draft(torch.randn(1, K).to(torch.bfloat16))
    target.apply(None, torch.randn(6, K).to(torch.bfloat16))
    assert direct.calls == [(1, (64, 4, 1))] and warp.calls == [(6, -1)]


def test_the_slot_is_opt_in_and_needs_a_low_m_head(monkeypatch):
    assert "lm_head_pinned_tactic" in runtime.OPT_IN_SLOTS
    assert "lm_head_pinned_tactic" not in runtime.SHIPPED_SLOTS
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"lm_head_pinned_tactic"}))
    assert any("lm_head_pinned_tactic" in r for r in runtime.slot_reasons())
    monkeypatch.setattr(
        runtime, "SLOTS", frozenset({"lm_head_pinned_tactic", "target_lm_head"})
    )
    assert not any("lm_head_pinned_tactic" in r for r in runtime.slot_reasons())


def test_a_pin_pruned_from_the_autotune_list_still_runs_if_the_runner_accepts_it(fi):
    # FlashInfer 0.7.3 prunes the warp split-K autotune list; the kernel still runs the pin
    warp, _ = fi
    warp.is_tactic_compatible = lambda inputs, tactic: tactic == (16, 8, 256, 4, 2)
    pinned.PINNED_TACTICS[(103, N, K, 6)] = ("WarpSplitK", (16, 8, 256, 4, 2))
    pinned.PINNED_TACTICS[(103, N, K, 8)] = ("WarpSplitK", (16, 8, 128, 12, 1))
    w = torch.randn(N, K).to(torch.bfloat16)
    for rows in (6, 8):
        pinned.pinned_mm(torch.randn(rows, K).to(torch.bfloat16), w.t())
    assert warp.calls == [(6, (16, 8, 256, 4, 2)), (8, -1)]
