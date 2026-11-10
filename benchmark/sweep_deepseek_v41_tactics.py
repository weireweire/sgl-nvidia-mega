# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tactic sweep on one GPU, microseconds per launch from CUDA graph replay, for the fixed
tactic tables: the low-latency MXFP8 GEMM (sm103/mxfp8_ll.py, ``--gemm N,K``) and the
low-M BF16 LM head (sampler/pinned.py, ``--lm-head N``).

The GEMM runs as a layer chain does: ``--layers`` weight copies, one launch each, so the
weights stream from DRAM; past 8 rows the activation scales are in the row-tile layout.
The LM head runs one weight (larger than L2 at the vocab shards) ``--inner`` times.
Every offered tactic of every runner is timed; the fastest few print per row count.

  python benchmark/sweep_deepseek_v41_tactics.py --gemm 5120,1024 --rows 1,2,4,6,8,12,16
  python benchmark/sweep_deepseek_v41_tactics.py --lm-head 16160 --rows 1,2,4,6,8,12,16
"""

import argparse
import statistics

import torch

from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll


def _graph_us(fn, launches: int, replays: int) -> float:
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    times = []
    for _ in range(replays):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / launches)
    return statistics.median(times)


def _row_tile_quant(x):
    tiles = [mxfp8_ll.quantize(x[t : t + 8]) for t in range(0, x.shape[0], 8)]
    return torch.cat([q for q, _ in tiles]), torch.cat(
        [s.reshape(-1) for _, s in tiles]
    )


def sweep_gemm(n: int, k: int, rows_list, layers: int, replays: int, top: int):
    gen = torch.Generator(device="cuda").manual_seed(0)
    weights = []
    for _ in range(layers):
        w = torch.randn(n, k, device="cuda", generator=gen) / k**0.5
        q, sf = mxfp8_ll.quantize(w.bfloat16())
        weights.append((q.contiguous(), sf.reshape(-1).contiguous()))
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    for rows in rows_list:
        x = torch.randn(rows, k, device="cuda", generator=gen).bfloat16()
        a, a_sf = (
            mxfp8_ll.quantize(x) if rows <= mxfp8_ll.MAX_ROWS else _row_tile_quant(x)
        )
        a_sf = a_sf.reshape(-1)
        out = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
        bucket = mxfp8_ll.row_bucket(rows)
        current = mxfp8_ll.select_tactic(sm, n, k, bucket)
        results = []
        for tactic in mxfp8_ll.offered_tactics(n, k, min(bucket, mxfp8_ll.MAX_ROWS)):
            try:
                us = _graph_us(
                    lambda: [
                        mxfp8_ll.linear(a, a_sf, w, s, out=out, tactic=tactic)
                        for w, s in weights
                    ],
                    layers,
                    replays,
                )
            except Exception as error:  # a tactic the wide rows cannot run
                print(f"gemm N={n} K={k} rows={rows} tactic={tactic} failed: {error}")
                continue
            results.append((us, tactic))
        results.sort()
        cur = next((us for us, t in results if t == tuple(current)), float("nan"))
        print(
            f"gemm N={n} K={k} rows={rows} bucket={bucket} tactics={len(results)} "
            f"current={tuple(current)} {cur:.2f}us best="
            + " ".join(f"{t}:{us:.2f}" for us, t in results[:top]),
            flush=True,
        )


def sweep_lm_head(n: int, k: int, rows_list, inner: int, replays: int, top: int):
    from flashinfer.gemm import gemm_base

    torch.manual_seed(0)
    weight = (torch.randn(n, k, device="cuda") * 0.02).bfloat16()
    weight_t = weight.t()
    empty = torch.empty(0, dtype=torch.uint8, device="cuda")
    for rows in rows_list:
        x = torch.randn(rows, k, device="cuda").bfloat16()
        out = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
        inputs = [x, weight_t, None, True, out, empty]
        results = []
        runners = gemm_base._cute_dsl_bf16_runners(inputs)
        for index, runner in enumerate(runners):
            if not runner.supports_inputs(inputs):
                continue
            name = type(runner).__name__
            tactics = [tuple(t) for t in runner.get_valid_tactics(inputs, None)]
            if index == 0:
                tactics.append(-1)  # FlashInfer's default
            for tactic in tactics:
                try:
                    us = _graph_us(
                        lambda: [
                            runner(inputs=inputs, tactic=tactic) for _ in range(inner)
                        ],
                        inner,
                        replays,
                    )
                except Exception as error:
                    print(f"lm_head N={n} rows={rows} {name} {tactic} failed: {error}")
                    continue
                results.append((us, name, tactic))
        results.sort(key=lambda r: r[0])
        default = next((us for us, _, t in results if t == -1), float("nan"))
        print(
            f"lm_head N={n} K={k} rows={rows} tactics={len(results)} default={default:.2f}us "
            "best=" + " ".join(f"{nm}{t}:{us:.2f}" for us, nm, t in results[:top]),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gemm", help="N,K of the low-latency MXFP8 GEMM")
    parser.add_argument("--lm-head", type=int, help="N of the LM head shard (K 5120)")
    parser.add_argument("--rows", default="1,2,4,6,8")
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--inner", type=int, default=20)
    parser.add_argument("--replays", type=int, default=30)
    parser.add_argument("--top", type=int, default=4)
    args = parser.parse_args()
    rows = [int(r) for r in args.rows.split(",")]
    print(f"# {torch.cuda.get_device_name()}", flush=True)
    if args.gemm:
        n, k = (int(v) for v in args.gemm.split(","))
        sweep_gemm(n, k, rows, args.layers, args.replays, args.top)
    if args.lm_head:
        sweep_lm_head(args.lm_head, 5120, rows, args.inner, args.replays, args.top)


if __name__ == "__main__":
    main()
