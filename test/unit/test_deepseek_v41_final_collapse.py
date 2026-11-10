# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot hop_final_collapse: only the last layer's hop B writes the head's collapse (an
Engram entry's streams-only seam keeps the plain consumer), and the head then runs
SGLang's final RMSNorm on it in place of the combine."""

from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41 import boundary
from sgl_nvidia_mega.deepseek_v41.boundary import boundaries


def _boundaries(final_collapse):
    b = boundaries.Boundaries.__new__(boundaries.Boundaries)
    b._mhc = SimpleNamespace(HcState=lambda *a: a)
    b.final_collapse, b._last_layer, b._served = final_collapse, 39, set()
    b._triplet = lambda hc, opened: ("pre", "post", "comb")
    calls = []

    def finalize(*args, rows, mxfp8_out=None, collapse=False):
        calls.append((args[-3:], collapse))
        return "streams", ("collapsed" if collapse else None)

    b.hop_a = SimpleNamespace(finalize=finalize)
    return b, calls


def _moe():
    routed = SimpleNamespace(
        gemm2_out="g2",
        expert_weights=torch.zeros(6, 6),
        expanded_idx_to_permuted_idx="idx",
    )
    return SimpleNamespace(routed=routed, shared="shared")


@pytest.mark.parametrize("final_collapse", [True, False])
@pytest.mark.parametrize("layer_id", [39, 12])
def test_only_the_last_layer_writes_the_collapse(final_collapse, layer_id):
    b, calls = _boundaries(final_collapse)
    st = boundaries.BoundaryState(hc=None)
    layer = SimpleNamespace(layer_id=layer_id, ffn_hc="hc")
    opened = SimpleNamespace(streams="streams_in")
    state = b._hop_b_moe(layer, _moe(), opened, None, st)
    collapse = final_collapse and layer_id == 39
    # the collapse consumer takes the pre-mix and no norm weight; the plain one neither
    assert calls == [
        ((("pre", None, 0.0) if collapse else (None, None, 0.0)), collapse)
    ]
    assert state == ("streams", "pre")
    assert st.final_collapse == ("collapsed" if collapse else None)


def test_the_head_norms_the_collapse_as_logits_head_norms_the_combine():
    residual = torch.randn(6, 4, 8)
    collapsed = torch.randn(6, 8)
    st = SimpleNamespace(model=SimpleNamespace(norm=lambda x: x * 2))
    hidden, pre_hc_head = boundary.head(st, residual, collapsed)
    assert torch.equal(hidden, collapsed * 2)
    assert torch.equal(pre_hc_head, residual.flatten(1))


def test_the_slot_is_opt_in_and_needs_hop_b():
    from sgl_nvidia_mega.deepseek_v41 import runtime

    assert "hop_final_collapse" in runtime.OPT_IN_SLOTS
    assert "hop_final_collapse" not in runtime.SHIPPED_SLOTS
    assert "hop_mbar" in runtime.OPT_IN_SLOTS
    assert "hop_mbar" not in runtime.SHIPPED_SLOTS
