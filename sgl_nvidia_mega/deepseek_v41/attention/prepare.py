# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/deepseek_v4.py (MQALayer.forward).
"""The query and cache-write half of SGLang's DeepSeek-V4.1 decode attention on CUDA:
the three prepare paths ``MQALayer`` picks between, with their stream order.

- ``multi_stream`` (``_forward_prepare_multi_stream``): the window-only layers under
  CUDA-graph capture; the KV store on a side stream.
- ``low_ratio`` (``_forward_prepare_low_ratio_multi_stream``): the ratio-1 / -2 layers;
  the compressor and the indexer through the backend on a side stream, the KV store
  on another.
- ``eager`` (``_forward_prepare``): everything else, on one stream.

Each returns the query (written into ``q_out`` when given); every side stream is
joined before it returns. ``q_ops`` runs the query and KV-store ops: ``ops`` (SGLang's
own) or the attention chain's."""

from __future__ import annotations

import torch

from sgl_nvidia_mega.deepseek_v41.attention import indexer, ops


def multi_stream(attn, x, positions, fb, view, q_out, x_quant, q_ops=ops):
    current = torch.cuda.current_stream()
    stream_kv, stream_compressor, stream_indexer = attn.alt_streams[:3]
    stream_kv.wait_stream(current)
    stream_compressor.wait_stream(current)
    stream_indexer.wait_stream(current)

    x_linear = x_quant if x_quant is not None else x
    qkv_a, qkv_a_ready = None, None
    if attn.fuse_wqa_wkv:
        qkv_a = q_ops.qkv_a(attn, x_linear)
        qkv_a_ready = current.record_event()
    # slot attn_q_norm_gemm: no indexer reads q_lora, so wq_b may norm it itself
    if (
        attn.indexer is None
        and getattr(q_ops, "q_norm_gemm", False)
        and q_ops.defers_norm(attn, x.shape[0])
    ):
        q_lora, q_for_wqb = q_ops.compute_q_a(attn, x_linear, qkv_a, defer=True)
    else:
        q_lora, q_for_wqb = q_ops.compute_q_a(attn, x_linear, qkv_a)
    q_lora_ready = current.record_event()
    if attn.indexer is not None:
        with torch.cuda.stream(stream_indexer):
            attn.indexer(
                x=x,
                q_lora=q_lora,
                forward_batch=fb,
                view=view,
                enable_multi_stream=True,
                q_lora_ready=q_lora_ready,
            )
    with torch.cuda.stream(stream_kv):
        if qkv_a_ready is not None:
            stream_kv.wait_event(qkv_a_ready)
        q_ops.kv_to_cache(attn, x_linear, positions, fb, view, qkv_a)
    if attn.compressor is not None:
        with torch.cuda.stream(stream_compressor):
            view.core_compressor(x, fb, attn.layer_id, attn.compressor)
    q = q_ops.compute_q_b(attn, q_for_wqb, positions, q_out)
    current.wait_stream(stream_kv)
    current.wait_stream(stream_compressor)
    current.wait_stream(stream_indexer)
    del qkv_a
    return q


def low_ratio(attn, x, positions, fb, view, q_out, x_quant, q_ops=ops):
    current = torch.cuda.current_stream()
    stream_kv = attn.alt_streams[0]
    stream_sources = attn.alt_streams[-1]
    x_linear = x_quant if x_quant is not None else x

    if attn.compressor is not None:
        stream_sources.wait_stream(current)
    # slot attn_indexer: the index head weights read only x, so the second stream forks
    # here; the index query joins it once q_lora is quantized
    plugin_index = _plugin_index(attn, q_ops, x, fb, view)
    weights_bf16 = None
    if plugin_index is not None:
        stream_query = attn.alt_streams[1]
        stream_query.wait_stream(current)
        with torch.cuda.stream(stream_query):
            head_weights = plugin_index.head_weights(attn, x)
            if getattr(q_ops, "index_weights_bf16", False):
                weights_bf16 = plugin_index.weights_bf16(attn, head_weights)
    qkv_a = None
    if attn.fuse_wqa_wkv:
        qkv_a = q_ops.qkv_a(attn, x_linear)
    if attn.compressor is not None:
        with torch.cuda.stream(stream_sources):
            view.low_ratio_sources(
                layer=attn,
                x=x,
                q_lora=None,
                positions=positions,
                forward_batch=fb,
                run_indexer=False,
            )
    stream_kv.wait_stream(current)
    # slot attn_index_q_first: the index query's wq_b ahead of wq_b on this stream, so
    # it never waits for the SMs wq_b and the attention hold
    first = plugin_index is not None and getattr(q_ops, "index_q_first", False)
    index_wq_b = attn.indexer.wq_b if first else None
    # slot attn_q_norm_gemm: only where no indexer reads q_lora (fused into the index
    # query's wq_b instead, the norm delays the index chain: slower per step)
    defer = (
        attn.indexer is None
        and getattr(q_ops, "q_norm_gemm", False)
        and q_ops.defers_norm(attn, x.shape[0])
    )
    q_lora, q_for_wqb = (
        q_ops.compute_q_a(attn, x_linear, qkv_a, defer=True)
        if defer
        else q_ops.compute_q_a(attn, x_linear, qkv_a)
    )
    # the index query on the second stream, beside the query chain
    if plugin_index is not None:
        req, pos = indexer.request_rows(view, fb, x.shape[0])
        pos = indexer.positions_for(pos, positions)
        if first:
            # the index chain stays on this stream (PDL launches overlap each hop); wq_b
            # moves to the second stream, off it. Each hop reads after its PDL wait.
            index_raw, q_for_wqb = q_ops.index_q_b(
                attn, q_for_wqb, index_wq_b.weight, index_wq_b.weight_scale_inv_swizzled
            )
            quant_ready = current.record_event()
            current.wait_stream(stream_query)  # the head weights
            index_q = plugin_index.pack(
                attn, index_raw, head_weights, pos, weights_bf16
            )
            stream_query.wait_event(quant_ready)
            with torch.cuda.stream(stream_query):
                q = q_ops.compute_q_b(attn, q_for_wqb, positions, q_out)
            q.record_stream(current)
        else:
            stream_query.wait_stream(current)
            with torch.cuda.stream(stream_query):
                index_q = plugin_index.query(
                    attn, q_for_wqb, head_weights, pos, weights_bf16
                )
        for t in index_q:
            if t is not None:
                t.record_stream(stream_sources)
    if attn.indexer is not None:
        stream_sources.wait_stream(current)
    if not first:
        q = q_ops.compute_q_b(attn, q_for_wqb, positions, q_out)

    def sglang_index():
        view.low_ratio_sources(
            layer=attn,
            x=x,
            q_lora=q_ops.index_q_input(attn, q_lora, q_for_wqb),
            positions=positions,
            forward_batch=fb,
            run_compressor=False,
        )

    if attn.indexer is not None:
        # forked above, right after q_lora; recorded here, after the query chain
        with torch.cuda.stream(stream_sources):
            if plugin_index is None:
                sglang_index()
            else:
                if not first:
                    stream_sources.wait_stream(stream_query)
                plugin_index.topk(attn, index_q, req, view)
                if plugin_index.check:
                    plugin_index.compare(attn, view, sglang_index)
    with torch.cuda.stream(stream_kv):
        q_ops.kv_to_cache(attn, x_linear, positions, fb, view, qkv_a)
    current.wait_stream(stream_kv)
    if attn.compressor is not None or attn.indexer is not None:
        current.wait_stream(stream_sources)
    if plugin_index is not None:
        current.wait_stream(stream_query)
    return q


def _plugin_index(attn, q_ops, x, fb, view):
    """The process's plugin indexer when it takes this layer's step, else None: the
    layer is one it took at build, the chain quantizes q_lora (``wq_b_ll``: its norm
    returns ``Mxfp8Rows``), and the step's metadata is one it scores
    (``StepIndexer.serves``)."""
    plugin = getattr(q_ops, "indexer", None)
    if (
        plugin is None
        or attn.indexer is None
        or not getattr(q_ops, "wq_b_ll", False)
        or not plugin.serves(view, x.shape[0], fb)
    ):
        return None
    return plugin


def eager(attn, x, positions, fb, view, q_out, x_quant, q_ops=ops):
    x_linear = x_quant if x_quant is not None else x
    if attn.fuse_wqa_wkv:
        qkv_a = q_ops.qkv_a(attn, x_linear)
        q_lora = qkv_a[..., : attn.q_lora_rank]
    else:
        q_lora, _ = attn.wq_a(x_linear)
        qkv_a = None
    q_lora, q_for_wqb = q_ops.normalize_q_lora(attn, q_lora)
    q = q_ops.compute_q_b(attn, q_for_wqb, positions, q_out)
    q_ops.kv_to_cache(attn, x_linear, positions, fb, view, qkv_a)
    del qkv_a
    if attn.compress_ratio in (1, 2) and (
        attn.compressor is not None or attn.indexer is not None
    ):
        view.low_ratio_sources(
            layer=attn, x=x, q_lora=q_lora, positions=positions, forward_batch=fb
        )
        return q
    if attn.indexer is not None:
        attn.indexer(
            x=x,
            q_lora=q_lora,
            forward_batch=fb,
            view=view,
            q_lora_quant=None,
        )
    if attn.compressor is not None:
        view.core_compressor(x, fb, attn.layer_id, attn.compressor)
    return q
