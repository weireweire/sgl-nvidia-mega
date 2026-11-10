# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Microbenchmark on one GPU: the attention kernels at attention TP4 vs TP8 shapes,
microseconds per launch from CUDA graph replay of a 40-layer chain.

- swap-AB sparse attention (PDL, RoPE in the kernel, narrow combine, entry trigger) on
  16 heads (TP4) and 8 heads (TP8, slot swapab_h8): a 128-token window plus 512
  compressed tokens per row, every layer its own cache;
- the fused inverse RoPE + wo_a + MXFP8 on 2 groups (TP4) and 1 group (TP8, slot
  wo_a_g1), every layer its own weight; and SGLang's TP8 arm (in-place inverse RoPE +
  BF16 einsum + FlashInfer's MXFP8 quantize for wo_b).

  python benchmark/bench_deepseek_v41_tp8_attn.py --rows 1,6,12,24
"""

import argparse
import statistics

import torch

from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll
from sgl_nvidia_mega.deepseek_v41.sm103.attention import swapab, wo_a

PAGE, PAGES = 256, 64
SWA, EXTRA = 128, 512


def _graph_us(fn, launches: int, replays: int) -> float:
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
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


def _cache():
    stride = PAGE * swapab.LAYOUT.bytes_per_token + 256
    raw = torch.randint(0, 120, (PAGES, stride), dtype=torch.uint8, device="cuda")
    return raw.as_strided(
        (PAGES, PAGE, 1, swapab.LAYOUT.bytes_per_token),
        (stride, swapab.LAYOUT.bytes_per_token, swapab.LAYOUT.bytes_per_token, 1),
    )


def bench_swapab(rows: int, layers: int, replays: int) -> dict:
    caches = [(_cache(), _cache()) for _ in range(layers)]
    angles = torch.rand(70000, 32, device="cuda") * 6.3
    freqs = torch.view_as_real(torch.polar(torch.ones_like(angles), angles)).flatten(-2)
    positions = torch.randint(70000, (rows,), device="cuda")
    ids = torch.randint(
        0, PAGE * PAGES, (rows, 1, SWA), device="cuda", dtype=torch.int32
    )
    ei = torch.randint(
        0, PAGE * PAGES, (rows, 1, EXTRA), device="cuda", dtype=torch.int32
    )
    lengths = torch.full((rows,), SWA, device="cuda", dtype=torch.int32)
    el = torch.full((rows,), EXTRA, device="cuda", dtype=torch.int32)
    sink = torch.randn(16, device="cuda")
    result = {}
    for heads in (16, 8):
        q = torch.randn(rows, heads, 512, device="cuda").bfloat16()

        def chain():
            for kv, extra in caches:
                swapab.swapab_attention(
                    q,
                    kv,
                    ids,
                    lengths,
                    sink[:heads],
                    extra,
                    ei,
                    el,
                    max_rows=24,
                    pdl=True,
                    narrow_combine=True,
                    q_rope=(freqs, positions),
                    combine_entry_trigger=True,
                )

        result[f"swapab_h{heads}"] = _graph_us(chain, layers, replays)
    return result


def bench_wo_a(rows: int, layers: int, replays: int) -> dict:
    from sglang.kernels.ops.attention.dsv4 import fused_rope_inplace

    freqs = torch.randn(4096, wo_a.ROPE_DIM, device="cuda")
    freqs_cis = torch.view_as_complex(freqs.view(4096, -1, 2))
    positions = torch.randint(0, 4096, (rows,), device="cuda", dtype=torch.int32)
    result = {}
    for groups in (2, 1):
        weights = [
            (
                torch.randn(groups, wo_a.RANK, wo_a.GROUP_K, device="cuda") * 0.02
            ).bfloat16()
            for _ in range(layers)
        ]
        x = torch.randn(rows, groups * 8, 512, device="cuda").bfloat16()
        wo_a.precompile(rows, groups=groups)

        def fused():
            for w in weights:
                wo_a.fused_rope_wo_a(x.view(rows, groups, -1), w, freqs, positions)

        result[f"wo_a_g{groups}"] = _graph_us(fused, layers, replays)
        if groups == 1:

            def sglang_tp8():
                for w in weights:
                    o = x.clone()
                    fused_rope_inplace(
                        o[..., -wo_a.ROPE_DIM :],
                        None,
                        freqs_cis,
                        positions=positions,
                        inverse=True,
                    )
                    y = torch.einsum("tgd,grd->tgr", o.view(rows, 1, -1), w)
                    mxfp8_ll.quantize(y.reshape(rows, -1))

            result["wo_a_g1_sglang"] = _graph_us(sglang_tp8, layers, replays)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", default="1,6,12,24")
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--replays", type=int, default=50)
    args = parser.parse_args()
    print(f"# {torch.cuda.get_device_name()} layers={args.layers}", flush=True)
    for rows in [int(r) for r in args.rows.split(",")]:
        result = {
            **bench_swapab(rows, args.layers, args.replays),
            **bench_wo_a(rows, args.layers, args.replays),
        }
        print(
            f"rows={rows} " + " ".join(f"{k}={v:.2f}us" for k, v in result.items()),
            flush=True,
        )


if __name__ == "__main__":
    main()
