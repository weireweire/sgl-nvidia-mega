# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/layers/attention/deepseek_v4_backend.py (_low_ratio_index_topk, the
# decode branch) and python/sglang/srt/layers/attention/dsv4/v41_indexer/
# (sparse_table.SparseTableBackend.publish_decode / consume_decode,
# full_topk.FullTopKIndexer._deep_gemm_decode, scoring._index_q_and_weights).
"""Slot attn_indexer: a covered verify step's DSV4.1 indexer top-k in the plugin's own
forward, with the index query built on the layer's second stream.

SGLang's backend computes the index query (wq_b, RoPE + fp4 pack, head weights) inside
``forward_low_ratio_sources`` on the sources stream, after the main query chain is
launched. Here the query is the indexer wq_b on the low-latency GEMM fed the chain's
q_lora quant, with SGLang's RoPE + pack, on ``alt_streams[1]`` beside the main query
chain; the head weights read only the layer input, so they start on that stream before
q_lora. The scoring and selection are SGLang's kernel ops on the sources stream, as
SGLang's three decode paths run them:

- a candidate source: paged MQA logits, the paged top-k, then the block selection and
  the sparse schedule on a side stream (its table stays in the step's state);
- a candidate consumer: sparse logits over the source's blocks, the sparse top-k;
- any other layer (or a capture variant that skips filtering): the full paged top-k.

A step this path does not take (chunked metadata, no candidate blocks, rows past the
chain) keeps SGLang's ``forward_low_ratio_sources`` for every indexer layer of it, so a
consumer never reads a table the other path published. The same kernels on the same
inputs: each row selects SGLang's index-K slots (``CHECK_ENV`` compares the sorted rows;
the top-k's emission order varies run to run).

Slot attn_indexer_fused: a consumer at 1-8 rows runs the sparse logits and the top-k in
one cluster launch (sm103/attention/reindex_select.py): DeepGEMM's logits bit for bit,
the stock top-k's selection on them (its pick among elements equal to the k-th value
follows warp arrival; the fused kernel's is the in-order one); ``CHECK_ENV`` then
checks the logits and that the selection is a top-k of them."""

from __future__ import annotations

import logging
import os
from typing import NamedTuple, Optional

import torch

from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll
from sgl_nvidia_mega.deepseek_v41.sm103.attention import reindex_select

logger = logging.getLogger(__name__)

# Test only: in every eager step it serves, also run SGLang's forward_low_ratio_sources on
# the same inputs and compare the selected index-K slots (counted, logged)
CHECK_ENV = "SGLANG_NVIDIA_MEGAKERNEL_CHECK_INDEXER"
LOG_EVERY = 500

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/attention/deepseek_v4_backend.py": (
        "c1_indexer_metadata",
        "c2_indexer_metadata",
        "core_metadata",
        "sparse_page_indices",
        "low_ratio_req_indices",
        "low_ratio_pos_i64",
    ),
    "srt/layers/attention/dsv4/metadata.py": (
        "deep_gemm_metadata",
        "compressed_seq_lens",
        "page_table",
        "max_compressed_seq_len",
        "compressed_page_size",
        "topk_metadata",
        "rows_per_request",
    ),
    "srt/layers/attention/dsv4/dsv41_sparse.py": (
        "is_candidate_source",
        "uses_candidates",
        "head_weights_raw",
        "head_weight_scale",
        "index_head_dim",
        "n_local_heads",
        "n_heads",
        "wq_b",
        "token_req_indices",
    ),
    "srt/configs/deepseek_v41.py": ("candidate_topk_blocks",),
    "srt/model_executor/runner_utils/capture_mode.py": (
        "skip_low_ratio_indexer",
        "skip_candidate_filtering",
    ),
    "srt/layers/attention/dsv4/v41_indexer/__init__.py": ("is_sm100_or_newer",),
    "srt/layers/attention/dsv4/v41_indexer/scoring.py": ("get_index_k_cache",),
    "srt/layers/attention/dsv4/indexer.py": (
        "topk_transform_paged_bf16",
        "topk_transform_paged_from_metadata",
    ),
    "srt/layers/deep_gemm_wrapper/configurer.py": ("DEEPGEMM_PAGED_SPARSE_MQA_LOGITS",),
    "kernels/ops/attention/dsv4/fp4_indexer_rope.py": ("index_q_rope_pack_weights",),
    "kernels/ops/attention/dsv4/index_logits.py": (
        "deep_gemm_fp4_paged_mqa_logits",
        "sparse_logits",
    ),
    "kernels/ops/attention/dsv4/topk.py": (
        "topk_transform_paged_v2",
        "topk_transform_sparse",
    ),
    "kernels/ops/attention/dsv4/candidate_blocks.py": (
        "amax_topk_blocks",
        "candidate_row_lens",
    ),
    "kernels/ops/attention/dsv4/candidate_table.py": (
        "build_sparse_indexer_schedule",
        "sort_candidate_blocks",
    ),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "metadata",
        "kv_pool",
        "pool",
    ),
}


class Query(NamedTuple):
    """``_index_q_and_weights`` of a verify step, shaped for the DeepGEMM scorers."""

    q_fp4: torch.Tensor  # [rows, 1, heads, 64]
    q_sf: torch.Tensor  # [rows, 1, heads]
    weights: torch.Tensor  # [rows, heads] fp32


class Table(NamedTuple):
    """The candidate source's selection (SGLang's ``_SparseTable``)."""

    blocks: torch.Tensor
    schedule: torch.Tensor
    phys_blocks: torch.Tensor
    valid_lens: torch.Tensor
    ready: torch.cuda.Event


def reason(attn, config) -> str:
    """Why slot attn_indexer cannot take this layer's indexer, or "" (load-time facts,
    the same on every rank)."""
    from sglang.srt.layers.attention.dsv4.v41_indexer import is_sm100_or_newer
    from sglang.srt.layers.deep_gemm_wrapper.configurer import (
        DEEPGEMM_PAGED_SPARSE_MQA_LOGITS,
    )

    from sgl_nvidia_mega.deepseek_v41.attention import chain

    indexer = getattr(attn, "indexer", None)
    if indexer is None:
        return "no indexer"
    if attn.compress_ratio not in (1, 2):
        return f"compress ratio {attn.compress_ratio}"
    if indexer.n_local_heads != indexer.n_heads:
        return "the indexer heads are sharded (the scorer sums them locally)"
    if not (is_sm100_or_newer() and DEEPGEMM_PAGED_SPARSE_MQA_LOGITS):
        return "SGLang's decode indexer is not the DeepGEMM sparse table here"
    if getattr(config, "candidate_topk_blocks", 0) <= 0:
        return "no candidate blocks (SGLang's dense-block indexer)"
    wq_b = indexer.wq_b
    if not (
        chain._ll_linear_applies(wq_b)
        and wq_b.bias is None
        and wq_b.weight.shape[-1] == attn.q_lora_rank
    ):
        return "the indexer wq_b is not an MXFP8 linear the low-latency GEMM takes"
    if attn.alt_streams is None or len(attn.alt_streams) < 3:
        return "no second stream for the index query"
    return ""


class StepIndexer:
    """One process's plugin indexer: its block-selection stream and its counters."""

    def __init__(self, config, fused: bool = False):
        self.topk_blocks = int(config.candidate_topk_blocks)
        self.stream = torch.cuda.Stream()
        self.check = os.environ.get(CHECK_ENV) == "1"
        # slot attn_indexer_fused: the consumers' fused kernel, built here (not in capture)
        self.fused = fused and self.topk_blocks * 8 == reindex_select.WIDTH
        if self.fused:
            reindex_select.precompile(torch.cuda.current_device())
        self._fused_logged: set[int] = set()
        self._fused_now = False  # the last consumer ran the fused kernel
        self._fused_layer = None  # the layer whose selection the fused kernel wrote last
        self._served_rows: set[int] = set()
        self._declined: dict[tuple, int] = {}
        self.compared = self.differ = 0
        # the forward this step's decision and table belong to: its metadata object, held
        # so that no later forward's metadata can take its id
        self._meta = None
        self._serves = False
        self._table: Optional[Table] = None

    def serves(self, view, rows: int, fb) -> bool:
        """Whether this forward's indexer layers all take the plugin's path: decided
        once per forward, so a consumer never reads the other path's table."""
        meta = view.metadata
        if meta is not self._meta:
            self._meta, self._table, self._fused_layer = meta, None, None
            paged = [
                m
                for m in (
                    getattr(meta, "c1_indexer_metadata", None),
                    getattr(meta, "c2_indexer_metadata", None),
                )
                if m is not None
            ]
            # chunked metadata: SGLang scores row chunks one by one
            chunked = any(isinstance(m.deep_gemm_metadata, list) for m in paged)
            self._serves = (
                fb.forward_mode.is_target_verify()
                and bool(paged)
                and rows <= mxfp8_ll.WIDE_MAX_ROWS
                and not chunked
            )
            if chunked:
                self._log_declined(rows, "chunked metadata")
        return self._serves

    @staticmethod
    def head_weights(attn, x) -> torch.Tensor:
        """The index heads' weights (``head_weights_raw``): they read only the layer
        input, so they run before q_lora is ready."""
        return attn.indexer.head_weights_raw(x)

    def query(self, attn, q_quant, head_weights, positions) -> Query:
        """``_index_q_and_weights`` with the wq_b on the low-latency GEMM (the chain's
        q_lora quant: 128x4 scales to 8 rows, row tiles past)."""
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_rope import (
            index_q_rope_pack_weights,
        )

        indexer = attn.indexer
        wq_b = indexer.wq_b
        rows, heads = head_weights.shape[0], indexer.n_local_heads
        q = mxfp8_ll.linear(
            q_quant.quant, q_quant.scale, wq_b.weight, wq_b.weight_scale_inv_swizzled
        )
        q_fp4, q_sf, weights = index_q_rope_pack_weights(
            q.view(rows, heads, indexer.index_head_dim),
            torch.view_as_real(attn.freqs_cis).flatten(-2),
            positions,
            head_weights,
            indexer.head_weight_scale,
        )
        return Query(q_fp4.view(rows, 1, heads, 64), q_sf.view(rows, 1, heads), weights)

    def topk(self, attn, data: Query, req, view) -> None:
        """``_low_ratio_index_topk`` (decode branch) on the current stream: writes the
        layer's selection into SGLang's ``sparse_page_indices(ratio)``."""
        from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import (
            get_index_k_cache,
        )
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            skip_candidate_filtering,
            skip_low_ratio_indexer,
        )

        ratio = attn.compress_ratio
        if skip_low_ratio_indexer(ratio):
            return
        indexer = attn.indexer
        is_source, is_consumer = indexer.is_candidate_source, indexer.uses_candidates
        if skip_candidate_filtering():
            is_source = is_consumer = False
        meta = view.metadata
        paged = meta.c1_indexer_metadata if ratio == 1 else meta.c2_indexer_metadata
        out = meta.core_metadata.sparse_page_indices(ratio)
        k_cache = get_index_k_cache(
            token_to_kv_pool=view.kv_pool.pool,
            layer_id=attn.layer_id,
            page_size=paged.compressed_page_size,
        )
        if is_consumer:
            # as SGLang's consume_decode asserts a published table
            assert self._table is not None, "an indexer consumer before its source"
            self._fused_now = self.fused_takes(data, k_cache, out)
            if self._fused_now:
                self._consume_fused(data, k_cache, self._table, out)
                self._fused_layer = attn.layer_id
                if self.check and not torch.cuda.is_current_stream_capturing():
                    self._check_fused(data, k_cache, self._table, out)
            else:
                self._consume(data, k_cache, self._table, out)
        else:
            logits = self._paged_logits(data, k_cache, paged)
            if is_source:
                self._table = self._publish(data, logits, paged, req, out)
            else:
                from sglang.srt.layers.attention.dsv4.indexer import (
                    topk_transform_paged_from_metadata,
                )

                topk_transform_paged_from_metadata(logits, paged, out, None)
        self._log_served(data.q_fp4.shape[0])

    @staticmethod
    def _paged_logits(data: Query, k_cache, paged):
        from sglang.kernels.ops.attention.dsv4.index_logits import (
            deep_gemm_fp4_paged_mqa_logits,
        )

        # the schedule plans rows_per_request consecutive rows per request, so the
        # operands are grouped the same way (DeepGEMMDecodeData.paged_logits_args)
        q_fp4, q_sf = data.q_fp4, data.q_sf
        lens, table = paged.compressed_seq_lens, paged.page_table
        n = paged.rows_per_request
        if n > 1:
            heads = q_fp4.shape[2]
            q_fp4 = q_fp4.view(-1, n, heads, 64)
            q_sf = q_sf.view(-1, n, heads)
            lens = lens.view(-1, n)
            table = table[::n]
        return deep_gemm_fp4_paged_mqa_logits(
            (q_fp4, q_sf),
            k_cache,
            data.weights,
            lens,
            table,
            paged.deep_gemm_metadata,
            paged.max_compressed_seq_len,
        )

    def _publish(self, data: Query, logits, paged, req, out) -> Table:
        """``SparseTableBackend.publish_decode`` (unchunked, verify rows)."""
        from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
            amax_topk_blocks,
            candidate_row_lens,
        )
        from sglang.kernels.ops.attention.dsv4.candidate_table import (
            build_sparse_indexer_schedule,
            sort_candidate_blocks,
        )
        from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged_v2
        from sglang.srt.layers.attention.dsv4.indexer import topk_transform_paged_bf16

        seq_lens = paged.compressed_seq_lens.reshape(-1)
        main = torch.cuda.current_stream()
        self.stream.wait_stream(main)
        # the block selection reads the logits after this stream moves on
        logits.record_stream(self.stream)
        if logits.dtype == torch.bfloat16:
            topk_transform_paged_bf16(
                logits, seq_lens, paged.page_table, out, paged.compressed_page_size
            )
        else:
            topk_transform_paged_v2(
                logits,
                seq_lens,
                paged.page_table,
                out,
                paged.compressed_page_size,
                paged.topk_metadata,
            )
        with torch.cuda.stream(self.stream):
            nblocks, valid_lens = candidate_row_lens(seq_lens, self.topk_blocks)
            blocks = amax_topk_blocks(logits, seq_lens, nblocks, self.topk_blocks)
            phys_blocks = sort_candidate_blocks(
                blocks, seq_lens, paged.page_table, paged.compressed_page_size
            )
            rows = data.q_fp4.shape[0]
            schedule = build_sparse_indexer_schedule(
                blocks,
                seq_lens,
                paged.page_table,
                paged.compressed_page_size,
                data.q_fp4.dtype,
                req[:rows].to(torch.int32).contiguous(),
            )
            # the consumers read these on their own streams
            for t in (blocks, schedule, phys_blocks, valid_lens):
                t.record_stream(main)
            ready = torch.cuda.Event()
            ready.record(self.stream)
        return Table(blocks, schedule, phys_blocks, valid_lens, ready)

    def wrote_fused(self, layer_id: int) -> bool:
        """Whether this forward's selection of ``layer_id`` came from the fused consumer,
        which writes it before it lets its dependents launch."""
        return self._fused_layer == layer_id

    def fused_takes(self, data: Query, k_cache, out) -> bool:
        """Whether slot attn_indexer_fused runs this consumer call (shapes known at
        capture, the same on every rank)."""
        if not self.fused:
            return False
        rows = data.q_fp4.shape[0]
        why = reindex_select.reason(
            rows,
            self.topk_blocks * 8,
            out.shape[1],
            k_cache.shape[1],
            data.q_fp4.shape[2],
        )
        if why:
            if rows not in self._fused_logged:
                self._fused_logged.add(rows)
                logger.info(
                    "NVIDIA megakernel: attention indexer fused consumer leaves %d rows "
                    "to the stock kernels (%s)",
                    rows,
                    why,
                )
        elif rows not in self._fused_logged:
            self._fused_logged.add(rows)
            logger.info(
                "NVIDIA megakernel: attention indexer fused consumer serves %d rows",
                rows,
            )
        return not why

    @staticmethod
    def _consume_fused(data: Query, k_cache, table: Table, out, logits=None) -> None:
        torch.cuda.current_stream().wait_event(table.ready)
        # the query (stream 1) and the table (the source's stream) join this stream by
        # events, which a captured graph turns into programmatic edges: the kernel reads
        # them before its PDL wait because their writers finished layers earlier (the
        # table) or end at their exit (the query's RoPE + pack, which never triggers)
        reindex_select.reindex_select(
            data.q_fp4,
            data.q_sf,
            data.weights,
            k_cache,
            table.phys_blocks,
            table.valid_lens,
            out,
            logits,
            inputs_before_wait=True,
        )

    @staticmethod
    def _consume(data: Query, k_cache, table: Table, out) -> None:
        """``SparseTableBackend.consume_decode``."""
        from sglang.kernels.ops.attention.dsv4.index_logits import sparse_logits
        from sglang.kernels.ops.attention.dsv4.topk import topk_transform_sparse

        torch.cuda.current_stream().wait_event(table.ready)
        logits = sparse_logits(
            data.q_fp4,
            data.q_sf,
            k_cache,
            data.weights.to(torch.bfloat16),
            table.schedule,
            table.blocks.shape[1],
        )
        topk_transform_sparse(logits, table.valid_lens, table.phys_blocks, out)

    def _log_declined(self, rows: int, why: str) -> None:
        # counted per (width, reason), logged at the first and every LOG_EVERY: the step
        # keeps SGLang's call for every indexer layer
        key = (rows, why)
        self._declined[key] = n = self._declined.get(key, 0) + 1
        if n == 1 or n % LOG_EVERY == 0:
            logger.info(
                "NVIDIA megakernel: attention indexer leaves %d rows to SGLang (%s) "
                "on %d forwards",
                rows,
                why,
                n,
            )

    def _log_served(self, rows: int) -> None:
        # engagement, from the dispatch itself: once per width on every rank
        if rows not in self._served_rows:
            self._served_rows.add(rows)
            logger.info("NVIDIA megakernel: attention indexer serves %d rows", rows)

    def compare(self, attn, view, run_sglang) -> None:
        """``CHECK_ENV``, eager steps only: SGLang's own call on the same inputs into
        the same buffer, against the plugin's selection."""
        if torch.cuda.is_current_stream_capturing():
            return
        if self._fused_now and attn.indexer.uses_candidates:
            # checked against the stock kernels in topk (_check_fused)
            self._fused_now = False
            return
        out = view.metadata.core_metadata.sparse_page_indices(attn.compress_ratio)
        ours = out.clone()
        run_sglang()
        self.compared += 1
        # the top-k emits its picks in no fixed order: each row's sorted slots (the -1
        # padding sorts first, so the row's count is compared too)
        self.differ += int(
            not torch.equal(ours.sort(dim=-1).values, out.sort(dim=-1).values)
        )
        # the forward goes on with the plugin's selection, so a CHECK run also measures
        # the plugin path's accuracy
        out.copy_(ours)
        if self.differ or self.compared % 200 == 1:
            logger.info(
                "NVIDIA megakernel: attention indexer check: %d of %d layer selections "
                "differ from SGLang's",
                self.differ,
                self.compared,
            )

    def _check_fused(self, data: Query, k_cache, table: Table, out) -> None:
        """``CHECK_ENV`` for a fused consumer: the fused logits are the stock sparse
        logits bit for bit, and the selection holds every valid column above the k-th
        largest logit and only columns at or above it (the stock top-k's equal pick
        follows warp arrival); a short row is the stock top-k's own output."""
        from sglang.kernels.ops.attention.dsv4.index_logits import sparse_logits
        from sglang.kernels.ops.attention.dsv4.topk import topk_transform_sparse

        rows, width, k = out.shape[0], self.topk_blocks * 8, out.shape[1]
        fused_logits = torch.zeros(rows, width, dtype=torch.bfloat16, device=out.device)
        again = torch.empty_like(out)
        self._consume_fused(data, k_cache, table, again, fused_logits)
        stock_logits = sparse_logits(
            data.q_fp4,
            data.q_sf,
            k_cache,
            data.weights.to(torch.bfloat16),
            table.schedule,
            self.topk_blocks,
        )
        stock_out = torch.empty_like(out)
        topk_transform_sparse(
            stock_logits, table.valid_lens, table.phys_blocks, stock_out
        )
        lens = table.valid_lens.tolist()
        cols = torch.arange(width, device=out.device)
        ok = torch.equal(again, out)
        for b, n in enumerate(lens):
            ok &= torch.equal(
                fused_logits[b, :n].view(torch.int16),
                stock_logits[b, :n].view(torch.int16),
            )
            if n <= k:
                ok &= torch.equal(out[b], stock_out[b])
                continue
            slots = (table.phys_blocks[b, cols[:n] // 8].long() << 3) | (cols[:n] & 7)
            order = slots.argsort()
            at = order[torch.searchsorted(slots[order], out[b].long())]
            ok &= torch.equal(slots[at], out[b].long()) and at.unique().numel() == k
            values = stock_logits[b, :n].float()
            kth = values.topk(k).values[-1]
            picked = values[at]
            ok &= bool((picked >= kth).all()) and int((picked > kth).sum()) == int(
                (values > kth).sum()
            )
        self.compared += 1
        self.differ += int(not ok)
        if self.differ or self.compared % 200 == 1:
            logger.info(
                "NVIDIA megakernel: attention indexer check: %d of %d layer selections "
                "differ from SGLang's (fused consumers: logits bit for bit, a top-k)",
                self.differ,
                self.compared,
            )


def request_rows(view, fb, rows: int):
    """The query rows' req_to_token rows, as ``forward_low_ratio_sources`` reads them."""
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import token_req_indices

    meta = view.metadata
    hoisted = getattr(meta, "low_ratio_req_indices", None)
    hoisted_pos = getattr(meta, "low_ratio_pos_i64", None)
    if hoisted is not None and hoisted_pos is not None and hoisted_pos.shape[0] == rows:
        return hoisted, hoisted_pos
    return token_req_indices(fb, num_tokens=rows), None


def positions_for(pos_hoisted: Optional[torch.Tensor], positions):
    return pos_hoisted if pos_hoisted is not None else positions
