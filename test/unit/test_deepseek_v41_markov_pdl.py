# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot markov_pdl: opt-in, a draft slot, served with the Markov chain only. The
launch itself is tested bitwise in test/kernels/test_deepseek_v41_draft.py."""

from sgl_nvidia_mega.deepseek_v41 import runtime


def test_the_slot_is_an_opt_in_draft_slot():
    assert "markov_pdl" in runtime.OPT_IN_SLOTS
    assert "markov_pdl" in runtime.DRAFT_SLOTS
    assert "markov_pdl" not in runtime.SHIPPED_SLOTS


def test_the_slot_needs_the_markov_chain(monkeypatch):
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"markov_pdl"}))
    assert any("'markov_pdl' needs slot 'markov'" in r for r in runtime.slot_reasons())
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"markov_pdl", "markov"}))
    assert not any("markov_pdl" in r for r in runtime.slot_reasons())
