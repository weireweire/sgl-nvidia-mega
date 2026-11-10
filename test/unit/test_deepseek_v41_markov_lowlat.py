# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot markov_lowlat: opt-in, a draft slot, served with the Markov chain only, and the
chain's row limits with it. The kernel is tested bitwise in
test/kernels/test_deepseek_v41_draft.py and its waits in test_deepseek_v41_fail_stop.py."""

from sgl_nvidia_mega.deepseek_v41 import runtime
from sgl_nvidia_mega.deepseek_v41.sm103 import markov


def test_the_slot_is_an_opt_in_draft_slot():
    assert "markov_lowlat" in runtime.OPT_IN_SLOTS
    assert "markov_lowlat" in runtime.DRAFT_SLOTS
    assert "markov_lowlat" not in runtime.SHIPPED_SLOTS


def test_the_slot_needs_the_markov_chain(monkeypatch):
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"markov_lowlat"}))
    assert any("'markov_lowlat' needs slot 'markov'" in r for r in runtime.slot_reasons())
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"markov_lowlat", "markov"}))
    assert not any("markov_lowlat" in r for r in runtime.slot_reasons())


def test_row_limits():
    # the current kernel: one lane per (row, rank) pair; direct at 1 row
    assert [markov.narrow_row_limit(world=w, max_bs=64) for w in (1, 2, 4, 8)] == [8, 8, 8, 4]
    assert {markov.direct_row_limit(world=w, lowlat=False) for w in (1, 2, 4, 8)} == {1}
    # lowlat: two passes, so 8 rows at every world; direct to 4 rows, 2 at world 8
    assert [
        markov.narrow_row_limit(world=w, max_bs=64, lowlat=True) for w in (1, 2, 4, 8)
    ] == [8, 8, 8, 8]
    assert [markov.direct_row_limit(world=w, lowlat=True) for w in (1, 2, 4, 8)] == [4, 4, 4, 2]
    assert markov.narrow_row_limit(world=8, max_bs=3, lowlat=True) == 3
