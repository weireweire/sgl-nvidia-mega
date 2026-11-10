# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The step prologue with slot ``prologue_fork``: the embedding written straight into
the mHC streams.

Stock, the prologue is one chain on the main stream: the embedding lookup, the mHC
broadcast of its rows into the streams, the Engram hashes, then layer 0's input norm.
With the slot, the Engram package forks the hashes (and the prefetch behind them)
onto its stream at the step start (``engram.fork``), and with the replicated table
(slot ``embed``) one kernel looks each token up and stores its row into every stream,
so the chain is lookup -> norm. Both are copies of the same rows: the streams equal
SGLang's lookup + ``hc_broadcast`` bit for bit.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/ops/layernorm/mhc.py": ("hc_broadcast",),
    "srt/models/deepseek_v4.py": ("embed_tokens", "hc_mult"),
    "srt/layers/vocab_parallel_embedding.py": (
        "num_embeddings",
        "_use_triton_embedding",
        "_use_triton_embedding(input_)",
    ),
    "srt/utils/async_probe.py": ("maybe_detect_oob",),
    "srt/layers/layer_boundary/__init__.py": ("get_attn_tp_context",),
    "srt/layers/layer_boundary/adapters/attention.py": ("input_scattered",),
}

BLOCK_H = 1024


@triton.jit
def _embed_streams_kernel(
    ids_ptr, table_ptr, out_ptr, vocab, H, stride, HC: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    col_mask = cols < H
    token = tl.load(ids_ptr + row).to(tl.int64)
    valid = (token >= 0) & (token < vocab)
    token = tl.where(valid, token, 0)
    vals = tl.load(table_ptr + token * stride + cols, mask=col_mask & valid, other=0.0)
    out = out_ptr + row * (HC * H)
    for k in tl.static_range(HC):
        tl.store(out + k * H + cols, vals, mask=col_mask)


def embed_into_streams(ids: torch.Tensor, table: torch.Tensor, hc: int) -> torch.Tensor:
    """[rows, hc, dim]: every stream the row of ``table`` for each id (zero for an id
    outside the table, as SGLang's fused lookup)."""
    rows, (vocab, dim) = ids.numel(), table.shape
    out = table.new_empty((rows, hc, dim))
    if rows:
        _embed_streams_kernel[(rows, triton.cdiv(dim, BLOCK_H))](
            ids,
            table,
            out,
            vocab,
            dim,
            table.stride(0),
            HC=hc,
            BLOCK=BLOCK_H,
            num_warps=8,
        )
    return out


def streams(st, input_embeds=None) -> torch.Tensor:
    """The streams before layer 0 [rows, hc, dim]: from the replicated table in one
    kernel where it serves, else SGLang's lookup (or ``input_embeds``) + broadcast."""
    from sglang.kernels.ops.layernorm.mhc import hc_broadcast

    from sgl_nvidia_mega.deepseek_v41.runtime import embedding

    m = st.model
    if input_embeds is None and st.slots.embed:
        table = st.bound.get("embedding")
        if table is not None and _takes(m.embed_tokens, st.input_ids):
            return embed_into_streams(st.input_ids, table, m.hc_mult)
    x = embedding.embed(st) if input_embeds is None else input_embeds
    return hc_broadcast(x, m.hc_mult)


def _takes(embed_tokens, input_ids) -> bool:
    """Where SGLang's fused lookup serves the step (as ``embedding._local``)."""
    from sglang.srt.layers.layer_boundary import get_attn_tp_context
    from sglang.srt.utils.async_probe import maybe_detect_oob

    if (
        get_attn_tp_context().input_scattered
        or not input_ids.is_contiguous()
        or not embed_tokens._use_triton_embedding(input_ids)
    ):
        return False
    maybe_detect_oob(
        input_ids, 0, embed_tokens.num_embeddings, "VocabParallelEmbedding input id"
    )
    return True
