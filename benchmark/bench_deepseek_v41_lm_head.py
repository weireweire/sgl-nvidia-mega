# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Microbenchmark: the LM-head shard at the verify / draft rows, microseconds per launch.

Arms, each on this rank's shard [N, K] (TP4 N 32320, TP8 16160; K 5120):
  bf16-pinned  the BF16 low-M head as served (slots target_lm_head + lm_head_pinned_tactic)
  mxfp8        the MXFP8 stream (slots target_lm_head_mxfp8 / draft_lm_head_mxfp8)
  cublas       torch.matmul on the BF16 weight (SGLang's own head)
  w8a8-ll      FlashInfer's low-latency MXFP8 GEMM (mxfp8_ll.linear) after quantizing x
Every launch reads a cold weight: each arm rotates over --copies copies, more bytes than
the L2 holds. A CUDA graph of --inner launches is replayed --repeats times per arm, the
arms alternating in palindrome order (A B B A ...); the table gives the median, min and
max per launch over the replays of each arm.

  python benchmark/bench_deepseek_v41_lm_head.py --n 32320 --rows 6,5
"""

import argparse
import statistics

import torch

from sgl_nvidia_mega.deepseek_v41.sampler.pinned import pinned_mm
from sgl_nvidia_mega.deepseek_v41.sm103 import lm_head_mxfp8 as lh
from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll


def _arms(n: int, k: int, rows: int, copies: int, names):
    g = torch.Generator(device="cuda").manual_seed(0)
    weights = [
        (torch.randn(n, k, generator=g, device="cuda") * 0.02).bfloat16()
        for _ in range(copies)
    ]
    quants = [lh.quantize(w) for w in weights]
    x = torch.randn(rows, k, generator=g, device="cuda").bfloat16()
    out = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
    lh.precompile(n, k, x.device)
    arms = {
        "bf16-pinned": [lambda w=w: pinned_mm(x, w.t()) for w in weights],
        "mxfp8": [lambda q=q: lh.matmul(x, q, out=out) for q in quants],
        "cublas": [lambda w=w: torch.matmul(x, w.t()) for w in weights],
    }
    if mxfp8_ll.supported():
        mxfp8_ll.precompile(quants[0].codes, [rows])

        def w8a8(q):
            xq, xs = mxfp8_ll.quantize(x)
            return mxfp8_ll.linear(xq, xs, q.codes, q.scales, out=out)

        arms["w8a8-ll"] = [lambda q=q: w8a8(q) for q in quants]
    return {name: fns for name, fns in arms.items() if name in names}


def _graph(fns, inner: int):
    for fn in fns:
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(inner):
            fns[i % len(fns)]()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _replay_us(graph, inner: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / inner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=32320)
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--rows", default="6,5")
    parser.add_argument("--copies", type=int, default=3)
    parser.add_argument("--inner", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--arms", default="bf16-pinned,mxfp8,cublas,w8a8-ll")
    args = parser.parse_args()
    print(
        f"{torch.cuda.get_device_name()} N {args.n} K {args.k}, {args.copies} cold copies, "
        f"{args.inner} launches a replay, {args.repeats} replays an arm"
    )
    for rows in (int(r) for r in args.rows.split(",")):
        arms = _arms(args.n, args.k, rows, args.copies, args.arms.split(","))
        graphs = {name: _graph(fns, args.inner) for name, fns in arms.items()}
        names = list(graphs)
        times = {name: [] for name in names}
        for r in range(args.repeats):
            for name in names if r % 2 == 0 else names[::-1]:
                times[name].append(_replay_us(graphs[name], args.inner))
        base = statistics.median(times[names[0]])
        print(f"rows {rows}:")
        for name in names:
            t = times[name]
            med = statistics.median(t)
            print(
                f"  {name:12s} median {med:7.2f} us  min {min(t):7.2f}  max {max(t):7.2f}"
                f"  vs {names[0]} {med - base:+7.2f} us"
            )
        del graphs, arms
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
