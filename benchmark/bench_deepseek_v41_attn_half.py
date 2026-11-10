# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Microbenchmark on one GPU: one layer's attention half as the chain launches it, every
layer its own weights and KV cache, over a CUDA graph of ``--layers`` layers.

  wqkv_a (low-latency MXFP8) -> q_norm + MXFP8 -> wq_b (low-latency) -> swap-AB partial
  (PDL, RoPE in the kernel, KV before the wait) -> narrow combine (arm fused:
  merged in the attention kernel; arm prewait: slot attn_prewait) -> fused inverse RoPE + wo_a + MXFP8 ->
  wo_b (the low-latency GEMM; hop A's publish + all-reduce + mHC is not on one GPU)

Each row reads a 128-token window plus 512 compressed tokens (index_topk) out of a
compressed cache of ``--ctx`` tokens (ratio 1 layers; a ratio 2 layer holds half).
``--kind index``: the sparse indices arrive from a side stream after wq_b (a full graph
edge, as the indexer's top-k joins an index layer), not before the step.

Reports, per kernel of a middle layer, median over layers and replays (torch profiler,
CUPTI): start / end from the layer's wqkv_a start, own = end - max(start, previous
end), wait = time spent launched before its input ended; and microseconds per layer
from plain graph replays, arms alternating.

  python benchmark/bench_deepseek_v41_attn_half.py --heads 16,8 --ctx 8192,250000
"""

import argparse
import json
import math
import os
import statistics
import tempfile

import torch

from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll
from sgl_nvidia_mega.deepseek_v41.sm103.attention import swapab, wo_a
from sgl_nvidia_mega.deepseek_v41.sm103.attention.q_norm_quant import rmsnorm_mxfp8

HIDDEN, Q_LORA, KV_DIM = 5120, 1280, 512
QKV_N = Q_LORA + KV_DIM  # wqkv_a fused, replicated over the attention ranks
SWA, TOPK = 128, 512
PAGE = 256
EPS = 1e-20


def _weight(n, k):
    w = torch.randn(n, k, device="cuda") / math.sqrt(k)
    q, sf = mxfp8_ll.quantize(w.bfloat16())
    return q.contiguous(), sf.reshape(-1).contiguous()


def _cache(tokens):
    pages = max(1, -(-tokens // PAGE))
    stride = PAGE * swapab.LAYOUT.bytes_per_token + 256
    raw = torch.randint(0, 120, (pages, stride), dtype=torch.uint8, device="cuda")
    return raw.as_strided(
        (pages, PAGE, 1, swapab.LAYOUT.bytes_per_token),
        (stride, swapab.LAYOUT.bytes_per_token, swapab.LAYOUT.bytes_per_token, 1),
    )


class Layer:
    def __init__(self, heads, rows, ctx, gen_pos):
        groups = heads // 8
        self.wqkv = _weight(QKV_N, HIDDEN)
        self.q_norm = (1 + 0.1 * torch.randn(Q_LORA, device="cuda")).bfloat16()
        self.wq_b = _weight(heads * 512, Q_LORA)
        self.wo_a = (
            torch.randn(groups, wo_a.RANK, wo_a.GROUP_K, device="cuda") * 0.02
        ).bfloat16()
        self.wo_b = _weight(HIDDEN, groups * wo_a.RANK)
        self.swa = _cache(4 * PAGE)
        self.extra = _cache(ctx)
        # window: the last 128 slots of a ring page; top-k: distinct compressed slots
        base = torch.randint(
            0, 3 * PAGE, (rows, 1, 1), device="cuda", dtype=torch.int32
        )
        self.ids = base + torch.arange(SWA, device="cuda", dtype=torch.int32)
        self.ei = torch.stack(
            [torch.randperm(ctx, device="cuda")[:TOPK] for _ in range(rows)]
        ).to(torch.int32)[:, None, :]
        self.ei_src = self.ei.clone()
        self.sink = torch.randn(heads, device="cuda")
        self.counter = torch.zeros(
            swapab.WIDE_MAX_BATCH, dtype=torch.int32, device="cuda"
        )


class Chain:
    def __init__(self, heads, rows, ctx, layers, kind):
        self.heads, self.rows, self.kind = heads, rows, kind
        torch.manual_seed(0)
        maxpos = ctx + 64
        angles = torch.rand(maxpos, 32, device="cuda") * 6.3
        self.freqs = (
            torch.view_as_real(torch.polar(torch.ones_like(angles), angles))
            .flatten(-2)
            .contiguous()
        )
        self.pos = torch.randint(ctx - 16, ctx, (rows,), device="cuda")
        self.x = mxfp8_ll.quantize(torch.randn(rows, HIDDEN, device="cuda").bfloat16())
        self.layers = [Layer(heads, rows, ctx, self.pos) for _ in range(layers)]
        self.lengths = torch.full((rows,), SWA, device="cuda", dtype=torch.int32)
        self.el = torch.full((rows,), TOPK, device="cuda", dtype=torch.int32)
        self.side = torch.cuda.Stream()
        self.consumer = mxfp8_ll.WQ_B_TP8 if heads == 8 else None
        mxfp8_ll.precompile(self.layers[0].wq_b[0], [rows], self.consumer)
        mxfp8_ll.precompile(self.layers[0].wqkv[0], [rows])
        mxfp8_ll.precompile(self.layers[0].wo_b[0], [rows])
        wo_a.precompile(rows, groups=heads // 8)

    def run(self, arm="base", stamps=None):
        fused = arm == "fused"
        rows, x = self.rows, self.x
        main = torch.cuda.current_stream()
        for i, L in enumerate(self.layers):
            qkv = mxfp8_ll.linear(x[0], x[1].reshape(-1), *L.wqkv)
            _, q8, s8 = rmsnorm_mxfp8(qkv[:, :Q_LORA], L.q_norm, EPS, pdl=True)
            q = mxfp8_ll.linear(q8, s8, *L.wq_b, consumer=self.consumer)
            if self.kind == "index":
                # the indexer's top-k lands after the query chain, joined by a full edge
                self.side.wait_stream(main)
                with torch.cuda.stream(self.side):
                    torch.add(L.ei_src, 0, out=L.ei)
                main.wait_stream(self.side)
            o = swapab.swapab_attention(
                q.view(rows, 1, self.heads, 512),
                L.swa,
                L.ids,
                self.lengths,
                L.sink,
                L.extra,
                L.ei,
                self.el,
                max_rows=24,
                pdl=True,
                narrow_combine=True,
                q_rope=(self.freqs, self.pos),
                kv_before_wait=True,
                combine_entry_trigger=True,
                combine_counter=L.counter if fused else None,
                stage_before_wait={"prewait": 1, "prewait2": 2}.get(arm, 0),
                stamps=None if stamps is None else stamps[i],
            )
            groups = self.heads // 8
            codes, scales = wo_a.fused_rope_wo_a(
                o.view(rows, groups, -1), L.wo_a, self.freqs, self.pos
            )
            mxfp8_ll.linear(codes, scales.reshape(-1), *L.wo_b)

    def graph(self, **kw):
        for _ in range(2):
            self.run(**kw)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            self.run(**kw)
        g.replay()
        torch.cuda.synchronize()
        return g


def _replay_us(g, layers, replays):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    times = []
    for _ in range(replays):
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / layers)
    return times


def _short(name):
    for key, short in (
        ("LowLatencyBlockscaledGemm", "llgemm"),
        ("_rmsnorm_mxfp8", "q_norm"),
        ("partial_gluon", "attn"),
        ("_combine", "combine"),
        ("wo_a_fused", "wo_a"),
        ("copy", "idx_copy"),
        ("elementwise", "idx_copy"),
    ):
        if key in name:
            return short
    return name[:24]


def profile(chain, g, replays):
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
    info = {}
    for e in ks:
        n = _short(e["name"])
        a = e.get("args", {})
        info.setdefault(
            n,
            (
                a.get("grid"),
                a.get("block"),
                a.get("shared memory"),
                a.get("registers per thread"),
                a.get("blocks per SM"),
            ),
        )
    # split into layers at each wqkv_a: the llgemm following a wo_b llgemm (or first)
    seq = [
        (
            _short(e["name"]),
            e["ts"],
            e["ts"] + e["dur"],
            e.get("args", {}).get("stream"),
        )
        for e in ks
    ]
    layers, cur, nll = [], [], 0
    for k in seq:
        if k[0] == "llgemm":
            nll += 1
            if nll % 3 == 1 and cur:
                layers.append(cur)
                cur = []
        cur.append(k)
    layers.append(cur)
    n = len(chain.layers)
    keep = [L for i, L in enumerate(layers) if 2 <= i % n < n - 1]
    table = {}
    for L in keep:
        t0 = L[0][1]
        prev_end = None
        llc = 0
        idx_end = None
        for name, s, e, stream in L:
            if name == "llgemm":
                llc += 1
                name = ("wqkv_a", "wq_b", "wo_b")[min(llc, 3) - 1]
            if name == "idx_copy":
                idx_end = e
                table.setdefault(name, []).append((s - t0, e - t0, e - s, 0.0))
                continue
            inp = prev_end
            if name == "attn" and idx_end is not None:
                inp = max(inp, idx_end)
            own = e - max(s, inp) if inp is not None else e - s
            wait = max(0.0, inp - s) if inp is not None else 0.0
            table.setdefault(name, []).append((s - t0, e - t0, own, wait))
            prev_end = e
    rows = []
    for name, v in table.items():
        med = [statistics.median(x[j] for x in v) for j in range(4)]
        rows.append((med[0], name, med))
    rows.sort()
    return rows, info, statistics.median(L[-1][2] - L[0][1] for L in keep)


def stamps_report(chain, arm):
    rows = chain.rows
    nt = (SWA + 63) // 64 + (TOPK + 63) // 64
    st = torch.zeros(
        len(chain.layers),
        rows * nt,
        swapab.STAMP_SLOTS,
        dtype=torch.int64,
        device="cuda",
    )
    g = chain.graph(arm=arm, stamps=st)
    for _ in range(5):
        st.zero_()
        g.replay()
    torch.cuda.synchronize()
    s = st.cpu().double()
    n = len(chain.layers)
    out = {}
    for i in range(2, n - 1):
        cta = s[i]
        t0 = cta[:, 0].min()
        for k, label in enumerate(
            (
                "entry",
                "post_top",
                "post_wait",
                "kv_q_smem",
                "qk_mma",
                "pv_mma",
                "stored",
                "row_barrier",
                "combined",
            )
        ):
            col = cta[:, k]
            if (col > 0).all():
                out.setdefault(label, []).append(
                    ((col - t0).max().item() / 1e3, (col - t0).median().item() / 1e3)
                )
    print(
        "  in-kernel stamps (us from the first CTA's entry; median over layers of the max / median CTA):"
    )
    for label, v in out.items():
        print(
            f"    {label:12s} max {statistics.median(x[0] for x in v):6.2f}  median {statistics.median(x[1] for x in v):6.2f}"
        )
    sms = s[2][:, 15]
    print(f"  distinct SMs used by the {rows * nt} CTAs: {len(set(sms.tolist()))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--heads", default="16,8")
    ap.add_argument("--ctx", default="8192,250000")
    ap.add_argument("--rows", default="6")
    ap.add_argument("--kind", default="reuse,index")
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--replays", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--arms", default="base,prewait")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--stamps", action="store_true")
    a = ap.parse_args()
    print(
        f"# {torch.cuda.get_device_name()} SMs={torch.cuda.get_device_properties(0).multi_processor_count} layers={a.layers}",
        flush=True,
    )
    for heads in map(int, a.heads.split(",")):
        for ctx in map(int, a.ctx.split(",")):
            for kind in a.kind.split(","):
                for rows in map(int, a.rows.split(",")):
                    chain = Chain(heads, rows, ctx, a.layers, kind)
                    arms = a.arms.split(",")
                    graphs = {arm: chain.graph(arm=arm) for arm in arms}
                    tag = f"heads={heads} ctx={ctx} kind={kind} rows={rows}"
                    res = {arm: [] for arm in arms}
                    for rep in range(a.repeats):
                        order = arms if rep % 2 == 0 else arms[::-1]
                        for arm in order:
                            res[arm].append(
                                statistics.median(
                                    _replay_us(graphs[arm], a.layers, a.replays)
                                )
                            )
                    line = " ".join(
                        f"{arm}={statistics.median(v):.2f}us[{min(v):.2f}-{max(v):.2f}]"
                        for arm, v in res.items()
                    )
                    print(f"{tag} per-layer {line}", flush=True)
                    if a.profile:
                        for arm in arms:
                            table, info, span = profile(chain, graphs[arm], 10)
                            print(
                                f"  == {arm}: layer span (wqkv_a start -> wo_b end) {span:.2f} us"
                            )
                            print(
                                f"  {'kernel':10s} {'start':>7s} {'end':>7s} {'own':>6s} {'wait':>6s}"
                            )
                            for _, name, (s, e, own, wait) in table:
                                print(
                                    f"  {name:10s} {s:7.2f} {e:7.2f} {own:6.2f} {wait:6.2f}"
                                )
                            for name, (grid, block, smem, regs, bps) in info.items():
                                print(
                                    f"    {name}: grid {grid} block {block} smem {smem} regs {regs} blocks/SM {bps}"
                                )
                    if a.stamps:
                        for arm in arms:
                            print(f"  -- stamps {arm}")
                            stamps_report(chain, arm)
                    del graphs, chain
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
