# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Engram prefetch (slot ``engram``): every Engram layer's lookup and WKV GEMM off
the critical path, with no collective on the side stream.

An Engram layer's table lookup, MXFP8 quantize and WKV GEMM read only the token
hashes, yet SGLang runs them on that layer's critical path. For a routed step of at
most ``MAX_ROWS`` rows:

- right after the hashes, a side stream looks up this rank's rows of every Engram
  layer's table (rank-local; the per-rank host table zeroes the rows it does not own);
- at ``reduce_at`` (the layer before the first Engram layer, before its MoE) the main
  stream reduces all layers' rows in one all-reduce, SGLang's ``_reduce_owned_rows``;
  the side stream then multiplies each layer's rows by this rank's quarter of the WKV
  output columns (views of the weight and of its 128-row scale tiles) into zeros;
- at ``gather_at`` (the first Engram layer) the main stream sums the zero-padded
  quarters in one all-reduce on SGLang's TP group.

Every cross-GPU wait is on the main stream, in the same order on every rank, as in
SGLang's inline path. With slot ``engram_side_ar`` (``run_side`` / ``join``) both
all-reduces run on the side stream instead, right behind the lookups and the WKV
quarters, still in the same order on every rank; the main stream only waits for the
gathered rows at ``gather_at``. Both spin beside the main stream's Lamport hops:
SGLang's one-shot push kernel (152 CTAs of 128 threads, no shared memory at 6 rows)
and the hops' few CTAs fit on an SM beside any resident kernel, so neither waits for
the other to leave one. Both all-reduces sum one owner's value with zeros and the WKV
runs the same CuTe DSL GEMM, so the rows equal SGLang's inline ones bit for bit;
build checks that at every width it serves, with the new buffers poisoned first.
"""

from __future__ import annotations

import torch

from sgl_nvidia_mega.deepseek_v41 import plan

# The SGLang names this module reads, by file under the sglang package (``f(p)``: a
# parameter, ``"x"``: a string literal); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/layers/engram.py": (
        "engram_gate",
        "embed",
        "wkv",
        "q_weight",
        "k_weight",
        "eps",
        "clamp_value",
        "layer_hash_index",
        "row_start",
        "_owned_rows",
        "_reduce_owned_rows",
        "replication|layout",
        "rows",
        "dim",
    ),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "engram_host_table",
    ),
    "srt/models/deepseek_v4.py": (
        "model",
        "layers",
        "start_layer",
        "end_layer",
        "engram",
    ),
    "srt/layers/quantization/fp8.py": (
        "Fp8LinearMethod",
        "use_mxfp8",
        "block_fp8_as_mxfp8",
        "block_fp8_mxfp8_ready",
        "mxfp8_dense_backend",
        "weight_scale_inv_swizzled",
    ),
    "srt/layers/quantization/fp8_utils.py": (
        "flashinfer_mxfp8_blockscaled_linear",
        "flashinfer_mxfp8_blockscaled_linear(backend)",
        "flashinfer_mxfp8_blockscaled_linear(pin_tactic)",
        '"cute-dsl"',
        "is_flashinfer_cutlass",
        "is_flashinfer_cutedsl",
    ),
    "srt/distributed/parallel_state.py": (
        "all_reduce",
        "world_size",
        "rank_in_group",
        "cpu_group",
    ),
}

MAX_ROWS = plan.ENGRAM_PREFETCH_MAX_ROWS
# The WKV slice of each rank is whole 128-row scale tiles
SCALE_TILE_ROWS = 128


def _per_rank_host_table(table) -> bool:
    """Accept current replication or legacy layout, rejecting unknown tables."""
    if table is None:
        return False
    try:
        replication = table.replication
    except AttributeError:
        try:
            return table.layout == "per_rank"
        except AttributeError:
            return False
    return replication == "none"


def reason(engram, tp_size: int, table) -> str:
    """Why this Engram layer cannot prefetch, or "": the per-rank host ``table`` (the
    binding's; each rank looks up only its own rows) and an MXFP8 WKV with swizzled
    scales."""
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    if not _per_rank_host_table(table):
        return (
            "needs the per-rank host Engram table (SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1"
            ", SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank)"
        )
    wkv = engram.wkv
    method = wkv.quant_method
    if not isinstance(method, Fp8LinearMethod):
        return f"the WKV is on {type(method).__name__}"
    backend = method.mxfp8_dense_backend
    mxfp8 = method.use_mxfp8 or (
        method.block_fp8_as_mxfp8 and getattr(wkv, "block_fp8_mxfp8_ready", False)
    )
    if not (
        mxfp8
        and backend is not None
        and (backend.is_flashinfer_cutlass() or backend.is_flashinfer_cutedsl())
    ):
        return "the WKV is not a FlashInfer MXFP8 GEMM"
    if wkv.weight.size(0) % (tp_size * SCALE_TILE_ROWS):
        return f"{wkv.weight.size(0)} WKV columns do not split into scale tiles"
    return ""


class _Layer:
    """One Engram layer's WKV quarter: this rank's output columns."""

    def __init__(self, engram, rank: int, tp_size: int):
        wkv = engram.wkv
        # detached: CuTe DSL's DLPack import refuses tensors that require grad
        weight = wkv.weight.detach()
        n = weight.size(0)
        cols = n // tp_size
        start = rank * cols
        tiles = wkv.weight_scale_inv_swizzled.reshape(n // SCALE_TILE_ROWS, -1)
        self.engram = engram
        self.weight = weight[start : start + cols]
        self.scale = tiles[
            start // SCALE_TILE_ROWS : (start + cols) // SCALE_TILE_ROWS
        ].reshape(-1)
        self.cols = slice(start, start + cols)
        self.n = n

    def quarter(self, emb: torch.Tensor) -> torch.Tensor:
        """This rank's columns of SGLang's ``wkv(emb)`` [rows, n / tp]."""
        from sglang.srt.layers.quantization.fp8_utils import (
            flashinfer_mxfp8_blockscaled_linear,
        )

        # CUTLASS's MXFP8 GEMM keeps one process-wide workspace; CuTe DSL keeps none
        return flashinfer_mxfp8_blockscaled_linear(
            input=emb.flatten(-2),
            weight=self.weight,
            weight_scale=self.scale,
            backend="cute-dsl",
            pin_tactic=True,
        )


class EngramPrefetch:
    """The side stream, its persistent buffers and the two main-stream points."""

    def __init__(self, engrams: dict, group, first_layer: int):
        rank, tp_size = group.rank_in_group, group.world_size
        self.group = group
        self.stream = torch.cuda.Stream()
        self.layers = {i: _Layer(e, rank, tp_size) for i, e in sorted(engrams.items())}
        self.index = {i: j for j, i in enumerate(self.layers)}
        self.reduce_at, self.gather_at = first_layer - 1, first_layer
        first = next(iter(self.layers.values()))
        self.dim = first.engram.embed.dim
        self.hash_cols = first.engram.wkv.weight.size(1) // self.dim
        device = first.weight.device
        n = len(self.layers)
        # persistent and flat: every width's views start at one address
        self.looked = torch.empty(
            n * MAX_ROWS * self.hash_cols * self.dim,
            dtype=torch.bfloat16,
            device=device,
        )
        self.parts = torch.empty(
            n * MAX_ROWS * first.n, dtype=torch.bfloat16, device=device
        )
        self.looked_up = torch.cuda.Event()
        self.reduced = torch.cuda.Event()
        self.wkv_done = torch.cuda.Event()
        self.kv_ready = torch.cuda.Event()
        self.rows = 0
        self.kv = None  # this step's rows [layers, rows, n], from ``gather``
        self._emb = None  # ``reduce``'s all-reduced rows, read by the side stream

    def _looked(self, rows: int) -> torch.Tensor:
        n = len(self.layers)
        return self.looked[: n * rows * self.hash_cols * self.dim].view(
            n, rows, self.hash_cols, self.dim
        )

    def _parts(self, rows: int) -> torch.Tensor:
        n, width = len(self.layers), next(iter(self.layers.values())).n
        return self.parts[: n * rows * width].view(n, rows, width)

    def issue(self, hash_ids: torch.Tensor) -> None:
        """Every Engram layer's own rows for ``hash_ids`` [rows, layers, hash columns],
        on the side stream (rank-local)."""
        if not 0 < hash_ids.size(0) <= MAX_ROWS:
            raise ValueError(f"the Engram prefetch takes 1-{MAX_ROWS} rows")
        self.rows = hash_ids.size(0)
        looked = self._looked(self.rows)
        self.stream.wait_stream(torch.cuda.current_stream())
        hash_ids.record_stream(self.stream)
        with torch.cuda.stream(self.stream):
            for j, layer in enumerate(self.layers.values()):
                e = layer.engram
                looked[j].copy_(e.embed._owned_rows(hash_ids[:, e.layer_hash_index]))
            self.looked_up.record(self.stream)

    def reduce(self) -> None:
        """Main stream at ``reduce_at``: SGLang's reduction of every layer's rows in one
        all-reduce; then the WKV quarters on the side stream."""
        main = torch.cuda.current_stream()
        main.wait_event(self.looked_up)
        first = next(iter(self.layers.values())).engram
        emb = first.embed._reduce_owned_rows(self._looked(self.rows))
        # An out-of-place all-reduce returns a DLPack tensor, whose record_stream the
        # caching allocator ignores: hold it until ``gather`` joins the side stream.
        emb.record_stream(self.stream)
        self._emb = emb
        self.reduced.record(main)
        parts = self._parts(self.rows)
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.reduced)
            parts.zero_()
            for j, layer in enumerate(self.layers.values()):
                parts[j, :, layer.cols] = layer.quarter(emb[j])
            self.wkv_done.record(self.stream)

    def gather(self) -> None:
        """Main stream at ``gather_at``: the zero-padded quarters summed into every
        layer's rows (this also joins the side stream)."""
        torch.cuda.current_stream().wait_event(self.wkv_done)
        self._emb = None  # the side stream's reads of it are now ordered before main
        self.kv = self.group.all_reduce(self._parts(self.rows))

    def run_side(self, hash_ids: torch.Tensor) -> None:
        """Slot engram_side_ar: ``issue``, ``reduce`` and ``gather`` all on the side
        stream (called on it, after the hashes); ``join`` hands the rows to main."""
        self.issue(hash_ids)
        with torch.cuda.stream(self.stream):
            first = next(iter(self.layers.values())).engram
            emb = first.embed._reduce_owned_rows(self._looked(self.rows))
            self._emb = emb
            parts = self._parts(self.rows)
            parts.zero_()
            for j, layer in enumerate(self.layers.values()):
                parts[j, :, layer.cols] = layer.quarter(emb[j])
            self.kv = self.group.all_reduce(parts)
            self.kv_ready.record(self.stream)

    def join(self) -> None:
        """Main stream at ``gather_at`` (slot engram_side_ar): wait for the rows."""
        main = torch.cuda.current_stream()
        main.wait_event(self.kv_ready)
        self.kv.record_stream(main)
        self._emb = None

    def gate(self, layer_id: int, x: torch.Tensor) -> torch.Tensor:
        """``Engram.forward(x, ...)`` of layer ``layer_id`` from its gathered rows."""
        from sglang.srt.layers.engram import engram_gate

        e = self.layers[layer_id].engram
        kv = self.kv[self.index[layer_id]][: x.size(0)]
        return engram_gate(x, kv, e.q_weight, e.k_weight, e.eps, e.clamp_value)

    def poison(self, byte: int) -> None:
        """Fill the persistent buffers with ``byte`` (build's check)."""
        self.looked.view(torch.uint8).fill_(byte)
        self.parts.view(torch.uint8).fill_(byte)


def _any_error(group, error: str) -> str:
    """Every rank's error, joined ("" when none failed): the same on every rank."""
    errors = [None] * group.world_size
    torch.distributed.all_gather_object(errors, error, group=group.cpu_group)
    return "; ".join(f"rank {r}: {e}" for r, e in enumerate(errors) if e)


def _hash_ids(w: int, cols: int, ranges: list) -> torch.Tensor:
    """[w, hash tables, cols] hash ids; element k of table h falls in rank k % tp's
    rows of h (``ranges[rank][h]`` = (row_start, rows))."""
    tp, tables = len(ranges), max(ranges[0]) + 1
    k = torch.arange(w * cols).view(w, cols)
    owner = k % tp
    out = torch.zeros((w, tables, cols), dtype=torch.int64)
    for h in ranges[0]:
        # a rank that holds no rows of table h lends its turn to the first that does
        first = next(r for r in range(tp) if ranges[r][h][1])
        held = [r if ranges[r][h][1] else first for r in range(tp)]
        start = torch.tensor([ranges[held[r]][h][0] for r in range(tp)])
        rows = torch.tensor([ranges[held[r]][h][1] for r in range(tp)])
        out[:, h] = start[owner] + (k * 7919) % rows[owner]
    return out


def build(model, ctx, group):
    """The Engram prefetch, or None, and the reason it is off. Build runs it at every
    width it takes, with its buffers poisoned (0xFF: a NaN in BF16), and compares each
    layer's rows with SGLang's inline lookup + WKV bit for bit (which also compiles its
    GEMMs before capture)."""
    m = model.model
    engrams = {
        i: m.layers[i].engram
        for i in range(m.start_layer, m.end_layer)
        if m.layers[i].engram is not None
    }
    widths = [w for w in ctx.widths if w <= MAX_ROWS]
    if not engrams or not widths:
        return None, "no Engram layer" if not engrams else "no width it takes"
    first_layer = min(engrams)
    if first_layer == m.start_layer:
        return None, f"layer {first_layer} is the first layer: no layer to reduce in"
    binding = getattr(ctx, "binding", None)
    if binding is None:
        return None, "SGLang offers no megakernel binding (the Engram host tables)"
    reasons = (
        reason(e, group.world_size, binding.engram_host_table(e))
        for e in engrams.values()
    )
    error = _any_error(group, next((r for r in reasons if r), ""))
    if error:
        return None, error
    # the same hash ids on every rank, inside every rank's own rows of each table
    # (SGLang shards the rows on hash-column boundaries, so the ranges differ)
    ranges = [None] * group.world_size
    torch.distributed.all_gather_object(
        ranges,
        {
            e.layer_hash_index: (e.embed.row_start, e.embed.rows)
            for e in engrams.values()
        },
        group=group.cpu_group,
    )
    # rank-local setup only: a failure here is gathered before any collective runs
    try:
        prefetch = EngramPrefetch(engrams, group, first_layer)
        device = prefetch.looked.device
        ids = {w: _hash_ids(w, prefetch.hash_cols, ranges).to(device) for w in widths}
        error = ""
    except Exception as e:  # gathered: the slot is then off on every rank
        error = f"{type(e).__name__}: {e}"
    error = _any_error(group, error)
    if error:
        return None, error
    # every collective below runs on the main stream, in the same order on every
    # rank; a mismatch is recorded, not raised, so no rank leaves them early
    from sgl_nvidia_mega.deepseek_v41 import runtime

    paths = ("main", "side") if "engram_side_ar" in runtime.SLOTS else ("main",)
    mismatch = ""
    for w in widths:
        for path in paths:
            prefetch.poison(0xFF)
            if path == "side":
                prefetch.run_side(ids[w])
                prefetch.join()
            else:
                prefetch.issue(ids[w])
                prefetch.reduce()
                prefetch.gather()
            for i, layer in prefetch.layers.items():
                e = layer.engram
                want, _ = e.wkv(e.embed(ids[w][:, e.layer_hash_index]).flatten(-2))
                got = prefetch.kv[prefetch.index[i]]
                if not mismatch and not torch.equal(got, want):
                    mismatch = f"layer {i} at {w} rows ({path}): not SGLang's rows"
    prefetch.kv = None
    error = _any_error(group, mismatch)
    return (None, error) if error else (prefetch, "")
