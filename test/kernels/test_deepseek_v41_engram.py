# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""A covered step's embedding, Engram prefetch and capture mean on one GPU (GB300 where
noted), against SGLang's.

- The replicated embedding (GB300): ``runtime.embedding``'s lookup on the whole table
  against SGLang's TP4 lookup (each rank's fused shard lookup, then the all-reduce's
  sum) at every routed row count, int32 and int64 token ids. The values are equal; only
  a -0.0 entry reads +0.0 after the all-reduce.
- The Engram prefetch's WKV (GB300): each TP4 rank's quarter of the output columns
  (views of the weight and of its 128-row swizzled scale tiles), zero-padded and summed
  as the main-stream gather sums them, equals SGLang's whole WKV GEMM at representative
  row counts the prefetch serves. The serving build repeats the check on the loaded
  model, with the real all-reduces and its buffers poisoned.
- Slot aux_mean_stats: the DSpark capture mean issued on the mHC stats stream equals
  the main-stream mean, and a main-stream reader after join_aux sees it even when the
  stats stream runs late.
"""

from types import SimpleNamespace

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.boundary.boundaries import (
    Boundaries,
    BoundaryState,
)
from sgl_nvidia_mega.deepseek_v41.engram import prefetch as ep
from sgl_nvidia_mega.deepseek_v41.runtime import embedding

try:
    from sglang.srt.models import deepseek_v4_mhc as mhc
except ImportError:
    mhc = None


# ---------------------------------------------------------------------------- embedding

VOCAB, HIDDEN, TP = 129280, 5120, 4  # DeepSeek-V4.1-Flash's table, sharded at TP4

needs_embedding = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)


class _Embedding:
    num_embeddings = VOCAB

    def _use_triton_embedding(self, input_):
        return True


@pytest.fixture(scope="module")
def table():
    gen = torch.Generator(device="cuda").manual_seed(5)
    full = torch.randn((VOCAB, HIDDEN), device="cuda", generator=gen).bfloat16()
    full[torch.rand((VOCAB, HIDDEN), device="cuda", generator=gen) < 1e-3] = -0.0
    return full


def _sharded(ids, full):
    from sglang.kernels.ops.embeddings.vocab_parallel_embedding import (
        vocab_parallel_embedding,
    )

    part = VOCAB // TP
    outs = [
        vocab_parallel_embedding(
            ids,
            full[r * part : (r + 1) * part],
            r * part,
            (r + 1) * part,
            0,
            VOCAB,
            VOCAB,
        )
        for r in range(TP)
    ]
    return torch.stack(outs).float().sum(0).bfloat16()


@needs_embedding
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_the_whole_table_lookup_equals_the_sharded_one(table, dtype, monkeypatch):
    from sglang.srt.layers import layer_boundary

    context = type("_Context", (), {"input_scattered": False})()
    monkeypatch.setattr(layer_boundary, "get_attn_tp_context", lambda: context)
    embed = _Embedding()
    gen = torch.Generator(device="cuda").manual_seed(11)
    for rows in range(1, 49):
        ids = torch.randint(0, VOCAB, (rows,), device="cuda", generator=gen).to(dtype)
        ids[0] = VOCAB - 1  # the last row of the last shard
        got = embedding._local(embed, table, ids)
        want = _sharded(ids, table)
        torch.cuda.synchronize()
        assert torch.equal(got, want), rows
        assert torch.equal(got.view(torch.int16), table[ids.long()].view(torch.int16))
    context.input_scattered = True  # scattered input: SGLang's own lookup
    assert embedding._local(embed, table, ids) is None


# ---------------------------------------------------------------------- Engram prefetch

TP, N, K, HASH_COLS = 4, 25600, 6144, 12  # DeepSeek-V4.1-Flash's WKV


def _engram_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        return False
    return True


needs_engram = pytest.mark.skipif(
    not _engram_ready(), reason="needs an SM100-family GPU"
)


@pytest.fixture(scope="module")
def wkv():
    from flashinfer import block_scale_interleave

    gen = torch.Generator(device="cuda").manual_seed(3)
    weight = (torch.randn((N, K), device="cuda", generator=gen) * 0.5).to(
        torch.float8_e4m3fn
    )
    scales = torch.randint(118, 126, (N, K // 32), device="cuda", generator=gen)
    swizzled = block_scale_interleave(scales.to(torch.uint8).contiguous())
    return SimpleNamespace(
        weight=weight, weight_scale_inv_swizzled=swizzled.contiguous()
    )


@needs_engram
@pytest.mark.parametrize("rows", [1, 6, 16])
def test_the_column_quarters_sum_to_sglangs_wkv(wkv, rows):
    from sglang.srt.layers.quantization.fp8_utils import (
        flashinfer_mxfp8_blockscaled_linear,
    )

    gen = torch.Generator(device="cuda").manual_seed(rows)
    emb = torch.randn((rows, HASH_COLS, K // HASH_COLS), device="cuda", generator=gen)
    emb = emb.to(torch.bfloat16)
    want = flashinfer_mxfp8_blockscaled_linear(
        input=emb.flatten(-2),
        weight=wkv.weight,
        weight_scale=wkv.weight_scale_inv_swizzled,
        backend="cute-dsl",
        pin_tactic=True,
    )
    engram = SimpleNamespace(wkv=wkv)
    total = torch.zeros((rows, N), dtype=torch.float32, device="cuda")
    for rank in range(TP):
        layer = ep._Layer(engram, rank, TP)
        assert (
            layer.weight.data_ptr() == wkv.weight[rank * N // TP].data_ptr()
        )  # a view
        # the side stream's zero-padded quarter, as gather sums it
        padded = torch.zeros((rows, N), dtype=torch.bfloat16, device="cuda")
        padded[:, layer.cols] = layer.quarter(emb)
        assert layer.cols == slice(rank * N // TP, (rank + 1) * N // TP)
        total += padded.float()
    torch.cuda.synchronize()
    assert torch.equal(total.bfloat16(), want)


# ----------------------------------------------------------------------- aux_mean_stats

needs_aux_mean = pytest.mark.skipif(
    not torch.cuda.is_available() or mhc is None,
    reason="needs a GPU and SGLang's DeepSeek-V4 mHC",
)

ROWS, HC, AUX_HIDDEN = 6, 4, 4096


def _boundaries():
    b = Boundaries.__new__(Boundaries)
    b._served = set()
    return b


def _run(b, streams, side, *, late_ns=0):
    """pre's capture path: fork, the mean on ``side``, then join_aux and a reader."""
    st = BoundaryState(mhc.HcState(streams))
    main = torch.cuda.current_stream()
    side.wait_stream(main)
    if late_ns:
        with torch.cuda.stream(side):
            torch.cuda._sleep(late_ns)
    mean = b._aux_mean(streams, side, st, streams.new_empty(streams[:, 0].shape))
    b.join_aux(st)
    # a main-stream reader of the mean (the draft head's input)
    return mean * 1.0


@needs_aux_mean
@pytest.mark.parametrize("rows", [1, 6, 12, 48])
def test_the_side_stream_mean_is_bitwise_and_joined(rows):
    torch.manual_seed(rows)
    streams = torch.randn(rows, HC, AUX_HIDDEN, device="cuda", dtype=torch.bfloat16)
    side = torch.cuda.Stream()
    b = _boundaries()
    want = streams.mean(dim=1)
    # the stats stream runs ~2 ms late: the reader must still see the mean
    got = _run(b, streams, side, late_ns=2_000_000)
    torch.cuda.synchronize()
    assert torch.equal(got, want)
