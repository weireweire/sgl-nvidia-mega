# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot shared_ll's chain against shared_fold / shared_fold_quant on one GB300, at the
DSV4.1-Flash TP4 shapes (``--inter 288``: TP8's shared expert), from CUDA graphs of
``--layers`` MoE halves with their own weights (the routed experts share one layer's
weights, each half its own routing).

Per half, from the fork (the first kernel of the half on either stream):

- ``chain``: the shared expert alone on the alternate stream (row quant, gate_up,
  SwiGLU, quant, down_proj; the fold drops the middle launches): its end and its
  CTA-time (sum of kernel duration x CTAs).
- ``layer``: the same chain beside the main stream's router GEMM, top-k and the
  routed GEMM pair (route, gemm1, gemm2), as ``moe/step.py`` forks them: when gemm1,
  gemm2 and the chain end, and the half's end (the join).

Arms run in palindrome order (``--repeats`` rounds) from a CUDA-graph replay each, timed
from a torch profiler trace; medians over halves and replays per round, then the
median and the spread of the rounds.

With ``--check-replays N`` it instead replays the layer graphs of fold_quant and
fold_quant_early N times each and checks every half's down_proj output bitwise against
one fold_quant replay (the quant buffers are shared by the layers, so a down_proj
reading them early reads another layer's).

python benchmark/bench_deepseek_v41_shared_fold.py --rows 6 --layers 39 --repeats 4
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile

import torch

HIDDEN, INTER, EXPERTS, TOP_K = 5120, 576, 384, 6
LIMIT = 10.0
# fold_quant_early: fold_quant releasing its dependents as FlashInfer's GEMM does
ARMS = ("base", "fold", "fold_quant", "fold_quant_early")


def _routed_layer(seed: int):
    """Random MXFP4 routed experts shuffled as SGLang's TRT-LLM MoE method does."""
    from flashinfer.fp4_quantization import block_scale_interleave
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    from sgl_nvidia_mega.deepseek_v41.sm103.moe import routed_expert_gemm as reg

    g = torch.Generator().manual_seed(seed)
    e, h, p, s = EXPERTS, reg.HIDDEN, reg.PADDED, reg.SLICE
    w13 = torch.randint(0, 256, (e, 2 * p, h // 2), dtype=torch.uint8, generator=g)
    w13_s = torch.randint(118, 126, (e, 2 * p, h // 32), dtype=torch.uint8, generator=g)
    w13.view(e, 2, p, -1)[:, :, s:] = 0
    w2 = torch.randint(0, 256, (e, h, p // 2), dtype=torch.uint8, generator=g)
    w2_s = torch.randint(118, 126, (e, h, p // 32), dtype=torch.uint8, generator=g)
    w2[:, :, s // 2 :] = 0
    w13, w13_s, w2, w2_s = (t.cuda() for t in (w13, w13_s, w2, w2_s))
    cache: dict = {}
    g1, g1s, g2, g2s = [], [], [], []
    for i in range(e):
        perm = _maybe_get_cached_w3_w1_permute_indices(cache, w13[i], 128)
        g1.append(w13[i][perm.cuda()].contiguous())
        perm = _maybe_get_cached_w3_w1_permute_indices(
            cache, w13_s[i], 128, num_elts_per_sf=16
        )
        g1s.append(block_scale_interleave(w13_s[i][perm.cuda()].contiguous()))
        perm = get_w2_permute_indices_with_cache(cache, w2[i], 128)
        g2.append(w2[i][perm.cuda()].contiguous())
        perm = get_w2_permute_indices_with_cache(
            cache, w2_s[i], 128, num_elts_per_sf=16
        )
        g2s.append(block_scale_interleave(w2_s[i][perm.cuda()].contiguous()))
    w13 = torch.stack(g1)
    w13_s = torch.stack(g1s).view(torch.float8_e4m3fn).reshape(e, 2 * p, -1)
    w2 = torch.stack(g2)
    w2_s = torch.stack(g2s).view(torch.float8_e4m3fn).reshape(e, h, -1)
    clamp = torch.full((e,), LIMIT, dtype=torch.float32, device="cuda")
    return reg.RoutedExpertGemm(
        w13, w13_s, w2, w2_s, clamp, top_k=TOP_K, max_rows=reg.SERVED_MAX_ROWS
    )


class Shared:
    """One layer's shared expert: gate_up (plain and interleaved), down_proj."""

    def __init__(self, seed: int, rows: int, inter: int):
        from sgl_nvidia_mega.deepseek_v41.moe.shared_ll import SwigluFold
        from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll

        g = torch.Generator(device="cuda").manual_seed(seed)
        self.inter = inter
        w = torch.randn(2 * inter, HIDDEN, device="cuda", generator=g) / HIDDEN**0.5
        self.w, self.w_s = (t.contiguous() for t in mxfp8_ll.quantize(w.bfloat16()))
        self.w_s = self.w_s.reshape(-1)
        d = torch.randn(HIDDEN, inter, device="cuda", generator=g) / inter**0.5
        self.d, self.d_s = (t.contiguous() for t in mxfp8_ll.quantize(d.bfloat16()))
        self.d_s = self.d_s.reshape(-1)
        self.fold = SwigluFold(self.w, self.w_s, LIMIT, 48, quant=False)
        self.fold_quant = SwigluFold(self.w, self.w_s, LIMIT, 48, quant=True)
        self.fold_quant_early = SwigluFold(
            self.w, self.w_s, LIMIT, 48, quant=True, early_release=True
        )
        mxfp8_ll.precompile(self.w, [rows])
        self.fold.precompile([rows])
        self.fold_quant.precompile([rows])
        self.fold_quant_early.precompile([rows])

    def down(self, h):
        from sglang.srt.layers.quantization.fp8_utils import (
            flashinfer_mxfp8_blockscaled_linear,
        )

        if isinstance(h, tuple):
            return flashinfer_mxfp8_blockscaled_linear(
                h[0], self.d, self.d_s, input_scale=h[1], backend="cute-dsl"
            )
        return flashinfer_mxfp8_blockscaled_linear(
            h, self.d, self.d_s, backend="cute-dsl"
        )

    def __call__(self, x, arm: str):
        from sgl_nvidia_mega.deepseek_v41 import sglang_ops
        from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll
        from sgl_nvidia_mega.deepseek_v41.sm103.attention.q_norm_quant import (
            mxfp8_row_tiles,
        )

        x_q, x_s = mxfp8_row_tiles(x)
        if arm == "fold":
            return self.down(self.fold(x_q, x_s))
        if arm in ("fold_quant", "fold_quant_early"):
            return self.down(tuple(getattr(self, arm)(x_q, x_s)))
        gate_up = mxfp8_ll.linear(x_q, x_s, self.w, self.w_s)
        h = gate_up.new_empty((x.shape[0], self.inter))
        sglang_ops.silu_and_mul_clamp(gate_up, h, LIMIT)
        return self.down(h)


def _graph(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def _kernels(graph, replays: int):
    """(stream, start us, end us, CTAs, name) of every kernel of ``replays`` replays."""
    from torch.profiler import ProfilerActivity, profile

    graph.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trace.json")
        prof.export_chrome_trace(path)
        with open(path) as f:
            trace = json.load(f)
    out = []
    for ev in trace["traceEvents"]:
        if ev.get("cat") != "kernel":
            continue
        args = ev.get("args", {})
        grid = args.get("grid", [1, 1, 1])
        ctas = grid[0] * grid[1] * grid[2]
        out.append(
            (args.get("stream"), ev["ts"], ev["ts"] + ev["dur"], ctas, ev["name"])
        )
    out.sort(key=lambda k: k[1])
    return out


def _split(kernels, per_half: int):
    """The kernels in launch-completion groups of one half each."""
    return [kernels[i : i + per_half] for i in range(0, len(kernels), per_half)]


def _chain_stats(kernels, kernels_per_half):
    ends, cta_time = [], []
    for half in _split(kernels, kernels_per_half):
        if len(half) < kernels_per_half:
            continue
        fork = half[0][1]
        ends.append(max(k[2] for k in half) - fork)
        cta_time.append(sum((k[2] - k[1]) * k[3] for k in half))
    return statistics.median(ends), statistics.median(cta_time)


def _halves(kernels):
    """The kernels of each half: every kernel name occurs once a half, so the j-th
    occurrence of each name (by start) belongs to half j."""
    by_name: dict = {}
    for k in kernels:
        by_name.setdefault(k[4], []).append(k)
    counts = {len(v) for v in by_name.values()}
    assert len(counts) == 1, f"names repeat within a half: {counts}"
    return [[v[j] for v in by_name.values()] for j in range(counts.pop())]


def _layer_stats(kernels, chain_names, verbose=False):
    """Per half, from the fork (its first kernel start): gemm1 end, gemm2 end, the
    shared chain's end, the half's end, and the chain's CTA-time. A graph replay runs
    its branches on streams of its own, so the chain's kernels are told by name (the
    chain-only graph's)."""
    rows, timeline = [], {}
    for half in _halves(kernels):
        alt = [k for k in half if k[4] in chain_names]
        named = {k[4]: k for k in half}
        if "dsv41_routed_expert_gemm2" not in named or not alt:
            continue
        fork = min(k[1] for k in half)
        seen: dict = {}
        for k in half:
            key = (k[4][:56], k[4] in chain_names, seen.setdefault(k[4], 0))
            seen[k[4]] += 1
            timeline.setdefault(key, []).append((k[1] - fork, k[2] - fork))
        rows.append(
            (
                named["dsv41_routed_expert_gemm1"][2] - fork,
                named["dsv41_routed_expert_gemm2"][2] - fork,
                max(k[2] for k in alt) - fork,
                max(k[2] for k in half) - fork,
                sum((k[2] - k[1]) * k[3] for k in alt),
            )
        )
    if verbose:
        spans = {
            key: (
                statistics.median(t[0] for t in v),
                statistics.median(t[1] for t in v),
            )
            for key, v in timeline.items()
        }
        for key, (t0, t1) in sorted(spans.items(), key=lambda kv: kv[1][0]):
            print(f"    {'alt ' if key[1] else 'main'} {t0:7.2f} {t1:7.2f}  {key[0]}")
    return tuple(statistics.median(c) for c in zip(*rows))


def _gate_topk(logits, bias, weights, ids):
    """The fast gate top-k (slot gate) as moe/step.py launches it."""
    import ctypes

    from sgl_nvidia_mega.deepseek_v41.sm103.moe import routed_expert_gemm as reg

    reg._launch(
        reg.load_gate_topk(logits.device.index),
        (logits.shape[0], 1, 1),
        [
            reg._pointer(logits),
            reg._pointer(bias),
            reg._null_or_pointer(None),
            reg._null_or_pointer(None),
            ctypes.c_int32(0),
            ctypes.c_int64(-1),
            reg._null_or_pointer(None),
            reg._pointer(weights),
            reg._pointer(ids),
            ctypes.c_float(1.5),
            ctypes.c_float(0.0),
        ],
        torch.cuda.current_stream().cuda_stream,
    )


def _check_replays(layer, replays: int) -> None:
    graphs, outs = {}, {}
    for arm in ("fold_quant", "fold_quant_early"):
        outs[arm] = []
        graphs[arm] = _graph(layer(arm, outs[arm]))
    graphs["fold_quant"].replay()
    torch.cuda.synchronize()
    want = torch.stack(outs["fold_quant"]).view(torch.int16).clone()
    for arm in ("fold_quant", "fold_quant_early"):
        bad = torch.zeros((), dtype=torch.int64, device="cuda")
        bad_replays = torch.zeros((), dtype=torch.int64, device="cuda")
        for _ in range(replays):
            graphs[arm].replay()
            got = torch.stack(outs[arm]).view(torch.int16)
            diff = (got != want).sum()
            bad += diff
            bad_replays += (diff > 0).long()
        torch.cuda.synchronize()
        print(
            f"CHECK {arm}: replays={replays} halves={got.shape[0]} "
            f"bad_replays={int(bad_replays)} bad_values={int(bad)}"
        )
        assert int(bad) == 0, arm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--layers", type=int, default=39)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--inter", type=int, default=INTER)
    parser.add_argument("--check-replays", type=int, default=0)
    args = parser.parse_args()
    rows, layers = args.rows, args.layers

    torch.manual_seed(0)
    shared = [Shared(100 + i, rows, args.inter) for i in range(layers)]
    routed = _routed_layer(7)
    routers = [
        (torch.randn(EXPERTS, HIDDEN, device="cuda") / HIDDEN**0.5).bfloat16()
        for _ in range(layers)
    ]
    bias = torch.randn(EXPERTS, device="cuda") * 0.05
    # a request's rows pick overlapping experts, as the verify rows of one request do
    xs = [
        (base + 0.2 * torch.randn(rows, HIDDEN, device="cuda")).bfloat16()
        for base in torch.randn(layers, 1, HIDDEN, device="cuda")
    ]
    weights = torch.empty((rows, TOP_K), dtype=torch.float32, device="cuda")
    ids = torch.empty((rows, TOP_K), dtype=torch.int32, device="cuda")
    alt, quant_stream = torch.cuda.Stream(), torch.cuda.Stream()

    from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    def chain_only(arm):
        def run():
            for i in range(layers):
                shared[i](xs[i], arm)

        return run

    def layer(arm, outs=None):
        """moe/step.py's dual-stream half: the router and routed pair on the main
        stream, the routed input quant on its own stream, the shared expert on the
        alternate stream, all from one fork."""

        def run():
            if outs is not None:
                outs.clear()
            current = torch.cuda.current_stream()
            for i in range(layers):
                x = xs[i]
                alt.wait_stream(current)
                quant_stream.wait_stream(current)
                logits = tiny_gemm_bf16(x, routers[i], out_dtype=torch.float32)
                _gate_topk(logits, bias, weights, ids)
                with torch.cuda.stream(quant_stream):
                    x_q, x_s = flashinfer_mxfp8_quantize(x, False, alignment=HIDDEN)
                    x_s = x_s.view(torch.float8_e4m3fn).reshape(rows, -1)
                current.wait_stream(quant_stream)
                routed(x_q, x_s, ids, weights)
                with torch.cuda.stream(alt):
                    out = shared[i](x, arm)
                current.wait_stream(alt)
                if outs is not None:
                    outs.append(out)

        return run

    if args.check_replays:
        _check_replays(layer, args.check_replays)
        return

    graphs = {arm: (_graph(chain_only(arm)), _graph(layer(arm))) for arm in ARMS}
    torch.cuda.synchronize()

    # the chain's kernel names and count
    probe = {}
    for arm in ARMS:
        ks = _kernels(graphs[arm][0], 1)
        ls = _kernels(graphs[arm][1], 1)
        names = {k[4] for k in ks}
        chain_k = len(ks) // layers
        print(
            f"{arm}: chain kernels/half {chain_k}, layer kernels/half {len(ls) // layers}"
        )
        print(f"  chain: {[k[4][:60] for k in ks[:chain_k]]}")
        assert sum(k[4] in names for k in ls) == len(ks), "a main kernel shares a name"
        probe[arm] = (chain_k, names)

    order = []
    for r in range(args.repeats):
        order += list(ARMS) if r % 2 == 0 else list(reversed(ARMS))
    results = {arm: [] for arm in ARMS}
    printed = set()
    for arm in order:
        chain_k, names = probe[arm]
        ks = _kernels(graphs[arm][0], args.replays)
        chain_end, chain_cta = _chain_stats(ks, chain_k)
        ls = _kernels(graphs[arm][1], args.replays)
        g1, g2, alt_end, half_end, alt_cta = _layer_stats(
            ls, names, verbose=arm not in printed
        )
        printed.add(arm)
        results[arm].append(
            dict(
                chain_end=chain_end,
                chain_cta_us=chain_cta,
                gemm1_end=g1,
                gemm2_end=g2,
                alt_end=alt_end,
                half_end=half_end,
                alt_cta_us=alt_cta,
            )
        )
        print(f"round {arm}: {results[arm][-1]}")

    print(
        f"\nrows={rows} inter={args.inter} layers={layers} repeats={args.repeats}"
        " (us; median [min, max])"
    )
    keys = list(results[ARMS[0]][0])
    print("metric".ljust(14) + "".join(a.rjust(26) for a in ARMS))
    for key in keys:
        line = key.ljust(14)
        for arm in ARMS:
            v = [r[key] for r in results[arm]]
            line += f"{statistics.median(v):9.2f} [{min(v):6.2f},{max(v):6.2f}]".rjust(
                26
            )
        print(line)


if __name__ == "__main__":
    main()
