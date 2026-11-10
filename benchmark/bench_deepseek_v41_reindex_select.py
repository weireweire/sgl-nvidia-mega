# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Microbenchmark of one indexer consumer ("Reindex") layer at decode: SGLang's
``sparse_logits`` + ``topk_transform_sparse`` against the fused kernel
(sm103/attention/reindex_select.py), on one GPU.

Each arm is a CUDA graph of ``--calls`` back-to-back layers; ``cold`` puts an L2 flush
before every layer and subtracts a graph of the flushes alone. Arms alternate
(A B B A ...) for ``--repeats`` rounds; the table prints the median per-layer time of
each arm, the spread of the round medians, and the paired difference.

    python benchmark/bench_deepseek_v41_reindex_select.py --rows 6 --repeats 10
"""

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test" / "kernels"))

import test_deepseek_v41_reindex_select as cases  # noqa: E402

from sgl_nvidia_mega.deepseek_v41.sm103.attention import (  # noqa: E402
    reindex_select as rs,
)


def _graph(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def _time(graph, iters: int) -> float:
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    graph.replay()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--length", type=int, default=60000)
    parser.add_argument("--calls", type=int, default=16)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    layers = []
    for i in range(args.calls):
        case = cases.Case([args.length + r for r in range(args.rows)], seed=i)
        phys, valid_lens, schedule = case.publish()
        out = torch.empty(args.rows, cases.TOPK, dtype=torch.int32, device="cuda")
        layers.append((case, phys, valid_lens, schedule, out))
    _, _, _, logits_op, topk_op = cases._stock_ops()
    flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")

    def stock(case, phys, valid_lens, schedule, out):
        logits = logits_op(
            case.q_fp4,
            case.q_sf,
            case.k_cache,
            case.weights.to(torch.bfloat16),
            schedule,
            cases.TOPK_BLOCKS,
        )
        topk_op(logits, valid_lens, phys, out)

    def fused(case, phys, valid_lens, schedule, out):
        rs.reindex_select(
            case.q_fp4,
            case.q_sf,
            case.weights,
            case.k_cache,
            phys,
            valid_lens,
            out,
            inputs_before_wait=True,
        )

    def arm(op, cold):
        def run():
            for layer in layers:
                if cold:
                    flush.zero_()
                op(*layer)

        return _graph(run)

    flushes = _graph(lambda: [flush.zero_() for _ in layers])
    graphs = {
        (name, cold): arm(op, cold)
        for name, op in (("stock", stock), ("fused", fused))
        for cold in (False, True)
    }
    for cold in (False, True):
        samples = {"stock": [], "fused": []}
        for r in range(args.repeats):
            order = ("stock", "fused") if r % 2 == 0 else ("fused", "stock")
            base = _time(flushes, args.iters) if cold else 0.0
            for name in order:
                t = _time(graphs[(name, cold)], args.iters)
                samples[name].append((t - base) / args.calls)
        diffs = [f - s for f, s in zip(samples["fused"], samples["stock"])]
        label = "cold (L2 flushed)" if cold else "warm"
        print(
            f"rows {args.rows}, {label}, us per index layer over {args.repeats} rounds:"
        )
        for name in ("stock", "fused"):
            v = samples[name]
            print(
                f"  {name:6s} median {statistics.median(v):7.2f}  "
                f"range {min(v):7.2f} .. {max(v):7.2f}"
            )
        print(
            f"  fused - stock median {statistics.median(diffs):+7.2f}  "
            f"range {min(diffs):+7.2f} .. {max(diffs):+7.2f}"
        )


if __name__ == "__main__":
    main()
