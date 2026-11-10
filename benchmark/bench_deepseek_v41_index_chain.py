# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Microbenchmark on one GPU: DeepSeek-V4.1's index layers as the plugin chains them at
verify widths, beside reuse layers, over a CUDA graph.

Every layer: wqkv_a (low-latency MXFP8) -> q_norm + MXFP8 -> wq_b (low-latency) ->
swap-AB partial (PDL, KV before the wait) -> narrow combine -> fused inverse RoPE + wo_a
-> wo_b (low-latency). A window-KV store stand-in (a copy) runs on the KV stream.
Index layers add, through the plugin's ``StepIndexer`` (slot attn_indexer):

- the index query (the indexer wq_b on the low-latency GEMM fed the chain's quant, SGLang's RoPE +
  FP4 pack) on the query stream;
- Full layers (``F``): SGLang's paged MQA logits (DeepGEMM) and paged top-k over the
  whole compressed context; the candidate source (``S``) also its block selection and
  sparse schedule on the indexer's side stream;
- consumers (``C``): the fused logits + top-512 over the source's 16,384 candidate
  columns (slot attn_indexer_fused) or SGLang's sparse logits + top-k (``--stock``).

Arms: ``old`` is the shipped chain (the fused consumer lets the attention launch just
before it writes the selection, which the attention reads before its PDL wait);
``base`` the fused consumer fencing its selection before that trigger;
``stock_safe`` also slot attn_index_after_wait (the attention reads a stock top-k's
selection after its wait); ``safe`` every selection after the wait. The compressor is
not modeled (its index-K and compressed-KV caches are static here).

Reports, per layer kind, the median over layers and replays (torch profiler, CUPTI) of
each kernel's stream, start and end from the layer's wqkv_a start, and the layer
period (next layer's wqkv_a start); and us per step from plain graph replays, arms
alternating. ``--check`` compares every layer's attention output across arms bitwise
over the replays.

  python benchmark/bench_deepseek_v41_index_chain.py --heads 16,8 --ctx 65536
"""

import argparse
import json
import math
import os
import statistics
import tempfile
from types import SimpleNamespace

import torch

from sgl_nvidia_mega.deepseek_v41.attention import chain as chain_mod
from sgl_nvidia_mega.deepseek_v41.attention import indexer as indexer_mod
from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll
from sgl_nvidia_mega.deepseek_v41.sm103.attention import swapab, wo_a
from sgl_nvidia_mega.deepseek_v41.sm103.attention.q_norm_quant import rmsnorm_mxfp8

HIDDEN, Q_LORA, KV_DIM = 5120, 1280, 512
QKV_N = Q_LORA + KV_DIM
SWA, TOPK = 128, 512
PAGE = 256  # tokens per page: the window and compressed KV caches
IPAGE = 128  # tokens per index-K page (DeepGEMM's paged logits take 32-128)
INDEX_HEADS, INDEX_DIM = 32, 128
TOPK_BLOCKS = 2048
EPS = 1e-20


def _weight(n, k):
    w = torch.randn(n, k, device="cuda") / math.sqrt(k)
    q, sf = mxfp8_ll.quantize(w.bfloat16())
    return q.contiguous(), sf.reshape(-1).contiguous()


def _kv_cache(pages):
    stride = PAGE * swapab.LAYOUT.bytes_per_token + 256
    raw = torch.randint(0, 120, (pages, stride), dtype=torch.uint8, device="cuda")
    return raw.as_strided(
        (pages, PAGE, 1, swapab.LAYOUT.bytes_per_token),
        (stride, swapab.LAYOUT.bytes_per_token, swapab.LAYOUT.bytes_per_token, 1),
    )


def _index_k_cache(pages, g):
    pool = torch.randint(0, 256, (pages, IPAGE * 68), generator=g, dtype=torch.uint8)
    pool[:, IPAGE * 64 :] = torch.randint(
        118, 131, (pages, IPAGE * 4), generator=g, dtype=torch.uint8
    )
    return pool.cuda().view(pages, IPAGE, 1, 68)


class Layer:
    def __init__(self, kind, heads, rows, pages, layer_id, g):
        self.kind = kind
        groups = heads // 8
        self.wqkv = _weight(QKV_N, HIDDEN)
        self.q_norm = (1 + 0.1 * torch.randn(Q_LORA, device="cuda")).bfloat16()
        self.wq_b = _weight(heads * 512, Q_LORA)
        self.wo_a = (
            torch.randn(groups, wo_a.RANK, wo_a.GROUP_K, device="cuda") * 0.02
        ).bfloat16()
        self.wo_b = _weight(HIDDEN, groups * wo_a.RANK)
        self.swa = _kv_cache(4)
        # the selection's slots index the compressed cache as they index the index-K pool
        self.extra = _kv_cache(-(-pages * IPAGE // PAGE))
        self.ids = (
            torch.randint(0, 3 * PAGE, (rows, 1, 1), device="cuda", dtype=torch.int32)
            + torch.arange(SWA, device="cuda", dtype=torch.int32)
        )
        self.sink = torch.randn(heads, device="cuda")
        self.kv_out = torch.empty(rows, KV_DIM, dtype=torch.bfloat16, device="cuda")
        self.attn = None
        if kind != "R":
            w_idx, s_idx = _weight(INDEX_HEADS * INDEX_DIM, Q_LORA)
            w_head = (torch.randn(INDEX_HEADS, HIDDEN, device="cuda") * 0.02).bfloat16()
            self.attn = SimpleNamespace(
                layer_id=layer_id,
                compress_ratio=1,
                freqs_cis=None,
                indexer=SimpleNamespace(
                    is_candidate_source=kind == "S",
                    uses_candidates=kind == "C",
                    n_local_heads=INDEX_HEADS,
                    index_head_dim=INDEX_DIM,
                    head_weight_scale=0.04,
                    wq_b=SimpleNamespace(weight=w_idx, weight_scale_inv_swizzled=s_idx),
                    head_weights_raw=lambda x, w=w_head: x @ w.t(),
                ),
            )
            self.k_cache = _index_k_cache(pages, g)


class Step:
    def __init__(self, pattern, heads, rows, ctx, stock_consumer):
        from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata
        from sglang.srt.layers.attention.dsv4.v41_indexer import scoring

        torch.manual_seed(0)
        g = torch.Generator().manual_seed(0)
        self.heads, self.rows = heads, rows
        pages = -(-ctx // IPAGE) + 2
        angles = torch.rand(ctx + 64, 32, device="cuda") * 6.3
        self.freqs_cis = torch.polar(torch.ones_like(angles), angles)
        self.freqs = torch.view_as_real(self.freqs_cis).flatten(-2).contiguous()
        self.pos = torch.arange(ctx - rows, ctx, device="cuda")
        self.x = mxfp8_ll.quantize(torch.randn(rows, HIDDEN, device="cuda").bfloat16())
        self.x_bf16 = torch.randn(rows, HIDDEN, device="cuda").bfloat16()
        self.layers = [
            Layer(k, heads, rows, pages, i, g) for i, k in enumerate(pattern)
        ]
        for L in self.layers:
            if L.attn is not None:
                L.attn.freqs_cis = self.freqs_cis
                # SGLang's weights_proj (n32k5120, ~2 us) reads only the layer input and
                # starts with the layer, off the index chain: precomputed here
                L.head_weights = self.indexer_head_weights(L)
        # one request of `rows` verify rows over a scattered page table
        perm = torch.randperm(pages - 1, generator=g)[: pages - 2].to(torch.int32)
        table = perm.view(1, -1).expand(rows, -1).contiguous().cuda()
        lens = torch.arange(ctx - rows + 1, ctx + 1, dtype=torch.int32, device="cuda")
        self.paged = PagedIndexerMetadata(
            page_size=PAGE,
            compressed_page_size=IPAGE,
            page_table=table,
            compressed_seq_lens=lens,
            use_topk_v2=True,
            compress_ratio=1,
            rows_per_request=rows,
        )
        self.sel = torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda")
        self.view = SimpleNamespace(
            metadata=SimpleNamespace(
                c1_indexer_metadata=self.paged,
                c2_indexer_metadata=None,
                core_metadata=SimpleNamespace(sparse_page_indices=lambda r: self.sel),
            ),
            kv_pool=SimpleNamespace(pool=None),
        )
        caches = {L.attn.layer_id: L.k_cache for L in self.layers if L.attn is not None}
        scoring.get_index_k_cache = lambda token_to_kv_pool, layer_id, page_size: caches[
            layer_id
        ]
        self.req = torch.zeros(rows, dtype=torch.int32, device="cuda")
        self.indexer = indexer_mod.StepIndexer(
            SimpleNamespace(candidate_topk_blocks=TOPK_BLOCKS), fused=not stock_consumer
        )
        self.lengths = torch.full((rows,), SWA, device="cuda", dtype=torch.int32)
        self.el = torch.full((rows,), TOPK, device="cuda", dtype=torch.int32)
        self.s_kv, self.s_query, self.s_sources = (torch.cuda.Stream() for _ in range(3))
        self.consumer = mxfp8_ll.WQ_B_TP8 if heads == 8 else None
        L0 = self.layers[0]
        mxfp8_ll.precompile(L0.wq_b[0], [rows], self.consumer)
        mxfp8_ll.precompile(L0.wqkv[0], [rows])
        mxfp8_ll.precompile(L0.wo_b[0], [rows])
        first_index = next(L for L in self.layers if L.attn is not None)
        mxfp8_ll.precompile(first_index.attn.indexer.wq_b.weight, [rows])
        wo_a.precompile(rows, groups=heads // 8)

    def indexer_head_weights(self, L):
        return L.attn.indexer.head_weights_raw(self.x_bf16)

    def run(self, arm):
        import functools

        from sgl_nvidia_mega.deepseek_v41.sm103.attention import reindex_select as rs

        # old: the fused consumer's dependents launch before it writes the selection (as
        # before trigger_after_write)
        if not hasattr(rs, "_bench_orig"):
            rs._bench_orig = rs.reindex_select
        rs.reindex_select = functools.partial(
            rs._bench_orig, trigger_after_write=not arm.startswith("old")
        )
        # base: slot attn_index_after_wait off; stock_safe: on (the stock selections of
        # the Full layers and the source after the wait); safe: every selection after it
        after_wait = arm.startswith("safe")
        stock_after_wait = after_wait or arm.startswith("stock_safe")
        rows, x = self.rows, self.x
        cur = torch.cuda.current_stream()
        self.indexer._meta = None
        self.indexer.serves(self.view, rows, SimpleNamespace(forward_mode=SimpleNamespace(is_target_verify=lambda: True)))
        outs = []
        for L in self.layers:
            index = L.attn is not None
            hw = L.head_weights if index else None
            qkv = mxfp8_ll.linear(x[0], x[1].reshape(-1), *L.wqkv)
            self.s_kv.wait_stream(cur)
            _, q8, s8 = rmsnorm_mxfp8(qkv[:, :Q_LORA], L.q_norm, EPS, pdl=True)
            if index:
                self.s_query.wait_stream(cur)
                with torch.cuda.stream(self.s_query):
                    iq = self.indexer.query(
                        L.attn, chain_mod.Mxfp8Rows(q8, s8), hw, self.pos
                    )
                self.s_sources.wait_stream(cur)
            q = mxfp8_ll.linear(q8, s8, *L.wq_b, consumer=self.consumer)
            if index:
                with torch.cuda.stream(self.s_sources):
                    self.s_sources.wait_stream(self.s_query)
                    self.indexer.topk(L.attn, iq, self.req, self.view)
            with torch.cuda.stream(self.s_kv):
                L.kv_out.copy_(qkv[:, Q_LORA:])
            cur.wait_stream(self.s_kv)
            if index:
                cur.wait_stream(self.s_sources)
                cur.wait_stream(self.s_query)
            o = swapab.swapab_attention(
                q.view(rows, 1, self.heads, 512),
                L.swa,
                L.ids,
                self.lengths,
                L.sink,
                L.extra,
                self.sel.view(rows, 1, TOPK),
                self.el,
                max_rows=24,
                pdl=True,
                narrow_combine=True,
                q_rope=(self.freqs, self.pos),
                kv_before_wait=True,
                combine_entry_trigger=True,
                extra_after_wait=index and (after_wait or (stock_after_wait and L.kind != "C")),
            )
            outs.append(o)
            groups = self.heads // 8
            codes, scales = wo_a.fused_rope_wo_a(
                o.view(rows, groups, -1), L.wo_a, self.freqs, self.pos
            )
            mxfp8_ll.linear(codes, scales.reshape(-1), *L.wo_b)
        return outs

    def graph(self, arm):
        for _ in range(2):
            self.run(arm)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            outs = self.run(arm)
        g.replay()
        torch.cuda.synchronize()
        return g, outs


def _replay_us(g, replays):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    times = []
    for _ in range(replays):
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3)
    return times


SHORT = (
    ("LowLatencyBlockscaledGemm", "llgemm"),
    ("_rmsnorm_mxfp8", "q_norm"),
    ("partial_gluon", "attn"),
    ("_combine", "combine"),
    ("wo_a_fused", "wo_a"),
    ("reindex_select", "reindex_select"),
    ("paged_mqa_logits", "paged_logits"),
    ("sparse_mqa_logits_metadata", "sparse_schedule"),
    ("sparse_mqa_logits", "sparse_logits"),
    ("index_q", "index_q_pack"),
)


def _short(name):
    for key, short in SHORT:
        if key in name:
            return short
    return name[:28]


def profile(step, g, replays):
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as tprofile

    with tprofile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(replays):
            g.replay()
        torch.cuda.synchronize()
    path = tempfile.mktemp(suffix=".json")
    prof.export_chrome_trace(path)
    events = json.load(open(path))["traceEvents"]
    os.unlink(path)
    ks = sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    seq = [
        (
            _short(e["name"]),
            e["ts"],
            e["ts"] + e["dur"],
            tuple(e.get("args", {}).get("grid", ())),
        )
        for e in ks
    ]
    # the low-latency GEMMs by grid: the first layer (a reuse layer) runs wqkv_a, wq_b
    # and wo_b in that order; any other grid is the indexer's wq_b
    first = [k for k in seq if k[0] == "llgemm"][:3]
    names = dict(zip((k[3] for k in first), ("wqkv_a", "wq_b", "wo_b")))
    seq = [
        (names.get(g, "index_wq_b") if n == "llgemm" else n, s, e, g)
        for n, s, e, g in seq
    ]
    starts = [s for n, s, e, g in seq if n == "wqkv_a"]
    nl = len(step.layers)
    per_kind = {}
    for li in range(len(starts) - 1):
        t0, t1 = starts[li], starts[li + 1]
        if li % nl in (0, nl - 1):
            continue
        kind = step.layers[li % nl].kind
        occ, rec = {}, []
        for name, s, e, _ in seq:
            if not t0 <= s < t1:
                continue
            o = occ.get(name, 0)
            occ[name] = o + 1
            rec.append(((name, o), s - t0, e - t0))
        d = per_kind.setdefault(kind, {"period": [], "k": {}})
        d["period"].append(t1 - t0)
        for key, s, e in rec:
            d["k"].setdefault(key, []).append((s, e))
    return per_kind


def print_profile(per_kind):
    for kind, d in sorted(per_kind.items()):
        print(
            f"    kind {kind}: layer period median {statistics.median(d['period']):6.2f} us "
            f"(n={len(d['period'])})"
        )
        rows = []
        for (name, occ), v in d["k"].items():
            s = statistics.median(x[0] for x in v)
            e = statistics.median(x[1] for x in v)
            dur = statistics.median(x[1] - x[0] for x in v)
            rows.append((s, name + (f"#{occ}" if occ else ""), e, dur, len(v)))
        for s, name, e, dur, cnt in sorted(rows):
            print(f"      {name:30s} {s:7.2f} .. {e:7.2f}  dur {dur:6.2f}  n={cnt}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--heads", default="16,8")
    ap.add_argument("--ctx", default="65536")
    ap.add_argument("--rows", default="6")
    ap.add_argument("--pattern", default="R,S,R,F,R,C,R,C,R,F,R,C,R,R")
    ap.add_argument("--arms", default="old,base,stock_safe")
    ap.add_argument("--replays", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--stock", action="store_true", help="SGLang's consumer kernels")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--check", type=int, default=0, help="replays compared across arms")
    a = ap.parse_args()
    pattern = a.pattern.split(",")
    print(
        f"# {torch.cuda.get_device_name()} SMs={torch.cuda.get_device_properties(0).multi_processor_count} "
        f"pattern={a.pattern} stock_consumer={a.stock}",
        flush=True,
    )
    for heads in map(int, a.heads.split(",")):
        for ctx in map(int, a.ctx.split(",")):
            for rows in map(int, a.rows.split(",")):
                step = Step(pattern, heads, rows, ctx, a.stock)
                arms = a.arms.split(",")
                graphs = {arm: step.graph(arm) for arm in arms}
                tag = f"heads={heads} ctx={ctx} rows={rows}"
                if a.check:
                    # every layer's attention output against the first arm's, and each
                    # arm against its own previous replay, by layer kind
                    kinds = [f"{L.kind}{i}" for i, L in enumerate(step.layers)]
                    across, own = {}, {}
                    prev = {}
                    for r in range(a.check):
                        got = {}
                        for arm in arms:
                            graphs[arm][0].replay()
                            torch.cuda.synchronize()
                            got[arm] = [o.clone() for o in graphs[arm][1]]
                            for i, o in enumerate(got[arm]):
                                if arm in prev and not torch.equal(o, prev[arm][i]):
                                    own[(arm, kinds[i])] = own.get((arm, kinds[i]), 0) + 1
                            prev[arm] = got[arm]
                        for arm in arms[1:]:
                            for i, (u, v) in enumerate(zip(got[arms[0]], got[arm])):
                                if not torch.equal(u, v):
                                    key = (arm, kinds[i])
                                    across[key] = across.get(key, 0) + 1
                    print(
                        f"{tag} check over {a.check} replays: differing layer outputs vs "
                        f"{arms[0]} {dict(sorted(across.items()))}; vs the arm's previous "
                        f"replay {dict(sorted(own.items()))}",
                        flush=True,
                    )
                res = {arm: [] for arm in arms}
                for rep in range(a.repeats):
                    order = arms if rep % 2 == 0 else arms[::-1]
                    for arm in order:
                        res[arm].append(
                            statistics.median(_replay_us(graphs[arm][0], a.replays))
                        )
                line = " ".join(
                    f"{arm}={statistics.median(v):.2f}us[{min(v):.2f}-{max(v):.2f}]"
                    for arm, v in res.items()
                )
                print(f"{tag} per-step ({len(pattern)} layers) {line}", flush=True)
                for arm in arms[1:]:
                    diffs = [b - a_ for a_, b in zip(res[arms[0]], res[arm])]
                    print(
                        f"  {arm} - {arms[0]} per repeat: median {statistics.median(diffs):+.2f} "
                        f"range {min(diffs):+.2f} .. {max(diffs):+.2f} us",
                        flush=True,
                    )
                if a.profile:
                    for arm in arms:
                        print(f"  == profile {arm}")
                        print_profile(profile(step, graphs[arm][0], 10))
                del graphs, step
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
