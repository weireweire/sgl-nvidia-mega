# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/layers/engram.py (Engram.forward) and python/sglang/srt/models/deepseek_v4.py.
"""The Engram layers: table lookup, WKV GEMM and gate, before the layer's attention.

``hashes`` computes the step's token hashes once; ``apply`` runs one Engram layer
on the materialized streams and writes its output back as the residual. Stock:
``Engram.forward`` (lookup -> WKV -> ``engram_gate``) from its own parts, and the
DeepSeek-V4.1 image-token rows keep their residual, as SGLang's layer loop does.
With the ``engram`` slot (``prefetch``), ``hashes`` also issues every layer's own
rows on a side stream, ``before_moe`` / ``before_layer`` run the prefetch's two
main-stream all-reduces at its layers, and ``apply`` only gates. With slot
``prologue_fork``, ``fork`` runs the hashes and the prefetch's lookups on the Engram
stream from the step start (they read only the step's inputs); with slot
``engram_side_ar`` the prefetch's two all-reduces follow there too, and the main
stream joins its rows at the first Engram layer.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from sgl_nvidia_mega.deepseek_v41 import boundary

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/engram.py": (
        "engram_gate",
        "embed",
        "forward(cp_all_tokens)",
        "wkv",
        "q_weight",
        "k_weight",
        "eps",
        "clamp_value",
        "layer_hash_index",
    ),
    "srt/models/deepseek_v4.py": ("engram_hasher", "engram", "config"),
    "srt/configs/deepseek_v41.py": ("vision_n_layers",),
    "srt/model_executor/forward_batch_info.py": ("contains_mm_inputs",),
}


class Bound(NamedTuple):
    """``st.bound["engram"]``: the prefetch (or None) and why it is off; with slot
    ``prologue_fork`` the stream the hashes fork onto (the prefetch's when built)."""

    prefetch: object
    reason: str
    stream: object = None


def build(model, ctx, group):
    import torch

    from sgl_nvidia_mega.deepseek_v41 import runtime
    from sgl_nvidia_mega.deepseek_v41.engram import prefetch

    if "engram" not in runtime.SLOTS:
        bound = Bound(None, "stock (SGLANG_NVIDIA_MEGAKERNEL_SLOTS)")
    else:
        bound = Bound(*prefetch.build(model, ctx, group))
    if "prologue_fork" in runtime.SLOTS and model.model.engram_hasher is not None:
        stream = bound.prefetch.stream if bound.prefetch else torch.cuda.Stream()
        bound = bound._replace(stream=stream)
    return bound


_FORKED = "engram.forked"  # st.scratch: the hashes were forked at the step start
# st.scratch: the forked hashes' event until the stock apply joins it (the prefetch
# joins its own later events instead)
_HASHED = "engram.hashed"


def fork(st) -> None:
    """Slot prologue_fork, at the step start: ``hashes`` on the Engram stream (with
    the prefetch's lookups, and with slot engram_side_ar its all-reduces, behind them)."""
    import torch

    hasher, side = st.model.engram_hasher, st.bound["engram"].stream
    if hasher is None or side is None:
        return
    main = torch.cuda.current_stream()
    side.wait_stream(main)
    with torch.cuda.stream(side):
        st.hash_ids = hasher(st.input_ids, st.fb)
        if st.slots.engram_prefetch:
            prefetch = st.bound["engram"].prefetch
            if st.slots.engram_side_ar:
                prefetch.run_side(st.hash_ids)
            else:
                prefetch.issue(st.hash_ids)
        else:
            st.scratch[_HASHED] = torch.cuda.Event()
            st.scratch[_HASHED].record(side)
    st.hash_ids.record_stream(main)
    st.scratch[_FORKED] = True


def hashes(st) -> None:
    """The step's token hashes [rows, Engram layers, hash columns], or None; with the
    prefetch, every Engram layer's lookup + WKV issued right after them. A no-op
    after ``fork``."""
    if _FORKED in st.scratch:
        return
    hasher = st.model.engram_hasher
    st.hash_ids = hasher(st.input_ids, st.fb) if hasher is not None else None
    if st.slots.engram_prefetch and st.hash_ids is not None:
        st.bound["engram"].prefetch.issue(st.hash_ids)


def _join_hashes(st) -> None:
    """The main stream waits for the forked hashes before reading them."""
    import torch

    hashed = st.scratch.pop(_HASHED, None)
    if hashed is not None:
        torch.cuda.current_stream().wait_event(hashed)


def before_moe(i: int, st) -> None:
    """Layer ``i`` between its attention and its MoE: the prefetch's lookup reduce."""
    if (
        st.slots.engram_prefetch
        and not st.slots.engram_side_ar
        and st.hash_ids is not None
    ):
        prefetch = st.bound["engram"].prefetch
        if i == prefetch.reduce_at:
            prefetch.reduce()


def before_layer(i: int, st) -> None:
    """Layer ``i`` before its Engram: the prefetch's WKV gather."""
    if st.slots.engram_prefetch and st.hash_ids is not None:
        prefetch = st.bound["engram"].prefetch
        if i == prefetch.gather_at:
            if st.slots.engram_side_ar:
                prefetch.join()
            else:
                prefetch.gather()


def apply(layer, st) -> None:
    """``layer.engram`` on ``st.state.residual`` (materialized by the runtime)."""
    from sglang.srt.layers.engram import engram_gate

    engram, config = layer.engram, st.model.config
    x = st.state.residual
    if st.slots.engram_prefetch:
        out = st.bound["engram"].prefetch.gate(st.layer_id, x)
    else:
        _join_hashes(st)
        ids = st.hash_ids[:, engram.layer_hash_index]
        emb = engram.embed(ids, st.fb, cp_all_tokens=False)
        kv, _ = engram.wkv(emb.flatten(-2))
        out = engram_gate(
            x, kv, engram.q_weight, engram.k_weight, engram.eps, engram.clamp_value
        )
    # Only multimodal placeholders become image_token_id, so a text-only batch (every
    # decode / verify graph among them) skips the full-residual select.
    if (
        config.model_type == "deepseek_v41"
        and config.vision_n_layers > 0
        and st.fb.contains_mm_inputs()
    ):
        image = st.input_ids == config.image_token_id
        out = torch.where(image[:, None, None], x, out)
    boundary.set_residual(st, out)
