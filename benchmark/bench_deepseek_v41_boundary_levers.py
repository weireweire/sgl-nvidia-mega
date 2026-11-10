# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Microseconds per mHC all-reduce seam (AllReduceMHC: publish + Lamport consumer) at
batch 1 (6 rows) on 4 GB300, with and without the boundary levers.

Each arm is one CUDA graph of 64 seams captured with the serving triggers: hop A
(attention post: publish, consumer with the FFN norm), hop B (MoE post: finalize
publish, consumer with the next layer's MXFP8 input), the two alternating (AB), or, as
in a step, hop A published by the wo_b GEMM's epilogue alternating with hop B (GB; the
GEMM's time is in the seam). Kind T is the step's tail: the last hop B (streams only),
then the head's FP32 cast, mhc_combine and RMSNorm; the collapse arm writes the collapse
in hop B and runs the RMSNorm only. Arms are timed in a palindrome order per repeat (base, levers..., levers
reversed, base), every rank replaying the same graph; the per-seam time is the
graph's replay time over 64. Run with
``torchrun --nproc_per_node=4 benchmark/bench_deepseek_v41_boundary_levers.py``.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "test", "kernels"))
import test_deepseek_v41_boundary_levers as t  # noqa: E402

from sgl_nvidia_mega.deepseek_v41.sm103.wo_b_publish import (  # noqa: E402
    PublishingMxfp8Linear,
)

SEAMS = 64
K = 2048  # wo_b's input width at TP4


def _wo_b(device, rows):
    """A random MXFP8 wo_b shard [HIDDEN, K] and a 128x4-quantized activation."""
    from flashinfer import mxfp8_quantize

    gen = torch.Generator(device=device).manual_seed(5)
    w = torch.randn(t.HIDDEN, K, device=device, generator=gen) / K**0.5
    w_q, w_sf = mxfp8_quantize(w.bfloat16(), is_sf_swizzled_layout=True)
    linear = SimpleNamespace(
        weight=w_q.contiguous(),
        weight_scale_inv_swizzled=w_sf.reshape(-1).contiguous(),
        bias=None,
    )
    x = torch.randn(rows, K, device=device, generator=gen).bfloat16()
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    q, sf = flashinfer_mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
    return linear, (q, sf.reshape(-1))


ARMS = {
    "base": {},
    "mbar": dict(mbarrier_exchange=True),
    "collapse": dict(final_collapse=True),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--arms", default="base,mbar")
    parser.add_argument("--kinds", default="A,B,AB,GB")
    args = parser.parse_args()
    e = t._init_dist()
    arms = args.arms.split(",")
    inp = t._inputs(args.rows, 1234, e.rank, e.device)
    serving = t.TRIGGERS["serving"]
    graphs = {}
    with torch.inference_mode():
        hops = {a: t._hop(e.device, e.group, **serving, **ARMS[a]) for a in arms}
        wo_b, qa = _wo_b(e.device, args.rows)
        publishers = {a: PublishingMxfp8Linear(hops[a]) for a in arms}
        for p in publishers.values():
            p.precompile(wo_b, args.rows)

        from sglang.kernels.ops.layernorm.mhc import mhc_combine
        from sglang.srt.layers.layernorm import RMSNorm

        head_norm = RMSNorm(t.HIDDEN, eps=t.EPS).to(e.device, torch.bfloat16)
        moe = (inp.gemm2, inp.weights, inp.idx, inp.shared, inp.residual, inp.post)
        moe += (inp.comb,)

        def seam(arm, kind):
            if kind == "t":
                if ARMS[arm].get("final_collapse"):
                    _, c = hops[arm].finalize(*moe, inp.pre, None, t.EPS, collapse=True)
                else:
                    streams = hops[arm].finalize(*moe, None, None, t.EPS)[0]
                    c = mhc_combine(
                        streams.flatten(1).float(), inp.pre, t.HC, streams.dtype
                    )
                return head_norm(c)
            if kind == "g":
                x = publishers[arm](wo_b, publishers[arm].prepare(wo_b, qa))
                return t._run(hops[arm], "a", SimpleNamespace(**{**vars(inp), "x": x}))
            return t._run(hops[arm], kind, inp)

        stream = torch.cuda.Stream()
        for kind in args.kinds.split(","):
            seams = {
                "A": ["a"],
                "B": ["quant"],
                "AB": ["a", "quant"],
                "GB": ["g", "quant"],
                "T": ["t"],
            }[kind]
            for arm in arms:
                for _ in range(3):  # compile and warm outside the capture
                    for s in seams:
                        seam(arm, s)
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
                    for i in range(SEAMS):
                        seam(arm, seams[i % len(seams)])
                torch.cuda.current_stream().wait_stream(stream)
                graphs[kind, arm] = graph
        torch.cuda.synchronize()
        dist.barrier(group=e.cpu_group)

        def time_us(graph) -> float:
            for _ in range(20):
                graph.replay()
            torch.cuda.synchronize()
            dist.barrier(group=e.cpu_group)
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(args.replays):
                graph.replay()
            end.record()
            end.synchronize()
            ms = start.elapsed_time(end)
            # the slowest rank's time: every seam waits for all of them
            v = torch.tensor([ms], device=e.device)
            dist.all_reduce(v, op=dist.ReduceOp.MAX, group=e.group)
            return float(v) * 1e3 / args.replays / SEAMS

        results = {key: [] for key in graphs}
        for repeat in range(args.repeats):
            order = arms + arms[::-1]
            if repeat % 2:
                order = order[::-1]
            for kind in args.kinds.split(","):
                per = {a: [] for a in arms}
                for arm in order:
                    per[arm].append(time_us(graphs[kind, arm]))
                for arm in arms:
                    results[kind, arm].append(statistics.mean(per[arm]))
    if e.rank == 0:
        print(f"rows={args.rows} seams/graph={SEAMS} replays={args.replays} "
              f"repeats={args.repeats} (us per seam; diff vs base paired per repeat)")  # fmt: skip
        for kind in args.kinds.split(","):
            base = results[kind, arms[0]]
            for arm in arms:
                vals = results[kind, arm]
                diffs = [v - b for v, b in zip(vals, base)]
                print(
                    f"{kind:3s} {arm:12s} mean {statistics.mean(vals):7.3f} "
                    f"sd {statistics.pstdev(vals):6.3f} | diff {statistics.mean(diffs):+7.3f} "
                    f"sd {statistics.pstdev(diffs):6.3f} min {min(diffs):+7.3f} "
                    f"max {max(diffs):+7.3f}",
                    flush=True,
                )
    dist.barrier(group=e.cpu_group)


if __name__ == "__main__":
    main()
