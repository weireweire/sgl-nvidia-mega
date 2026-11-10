# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot attn_prewait: opt-in, needs kv_before_wait, and engages only on 16-head reuse
layers (low ratio, no indexer of their own)."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41 import runtime
from sgl_nvidia_mega.deepseek_v41.attention import chain

# the package's step() shadows the module of the same name
step = importlib.import_module("sgl_nvidia_mega.deepseek_v41.attention.step")


def test_the_slot_is_opt_in_and_needs_kv_before_wait(monkeypatch):
    assert "attn_prewait" in runtime.OPT_IN_SLOTS
    assert "attn_prewait" not in runtime.SHIPPED_SLOTS
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"attn", "attn_prewait"}))
    assert any("attn_prewait" in r for r in runtime.slot_reasons())
    monkeypatch.setattr(
        runtime, "SLOTS", frozenset({"attn", "kv_before_wait", "attn_prewait"})
    )
    assert runtime.slot_reasons() == []


def _attn(ratio, heads, indexer):
    return SimpleNamespace(
        compress_ratio=ratio,
        n_local_heads=heads,
        indexer=object() if indexer else None,
        freqs_cis=torch.zeros(4, 32, dtype=torch.complex64),
    )


@pytest.mark.parametrize(
    "ratio,heads,indexer,slot,kv_before_wait,engaged",
    [
        (1, 16, False, True, True, True),
        (2, 16, False, True, True, True),
        (1, 16, True, True, True, False),  # an index source layer
        (0, 16, False, True, True, False),  # window only
        (1, 8, False, True, True, False),  # attention TP8
        (1, 16, False, False, True, False),
        (1, 16, False, True, False, False),
    ],
)
def test_engages_on_16_head_reuse_layers(
    monkeypatch, ratio, heads, indexer, slot, kv_before_wait, engaged
):
    monkeypatch.setattr(chain, "ChainOps", lambda attn, **kw: SimpleNamespace())
    monkeypatch.setattr(chain, "swapab_applies", lambda attn, heads8=False: True)
    monkeypatch.setattr(chain, "fused_wo_a_applies", lambda attn, one=False: False)
    s = step.AttentionStep(
        _attn(ratio, heads, indexer),
        chain_ops=True,
        kv_before_wait=kv_before_wait,
        attn_prewait=slot,
    )
    assert s.prewait is engaged


def test_widths():
    assert list(step.PREWAIT_ROWS) == [6, 7, 8]
