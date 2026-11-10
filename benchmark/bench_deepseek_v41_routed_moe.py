# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""One DeepSeek-V4.1 routed MoE half at batch 1 (6 verify rows, top-6 of 384 experts):
route + GEMM1 + GEMM2 on cold weights, optionally beside a shared-expert-sized chain on
a second stream, replayed from a CUDA graph of 39 halves.

  python benchmark/bench_deepseek_v41_routed_moe.py --distinct 16 24 32 \
      --variants strided clc fused --side 0 1 --repeats 5 [--check] [--stamps out.json]

Reports us per half (graph time / halves, minus a graph of the spacers alone), variants
interleaved in alternating order per repeat. --check compares every variant's GEMM2 rows
bit for bit against the first. With --stamps, a second replay of GEMM copies that write
%globaltimer / %smid per CTA (start, grid-dependency wait, first weight TMA, each tile's
epilogue, end) for the timeline; <out>_raw.json keeps the per-CTA words of 8 layers.

Variants: strided (the SGL_NVIDIA_MEGA_MOE_STRIDED_GEMM pair), clc (the cluster-launch-
control pair), custom:<gemm1>:<grid>:<gemm2>:<grid> over the kernels below, fused[:<grid>]
(one persistent launch, GEMM2 tiles gated per expert), preroute+<variant> (route kernel
off the measured chain). The bench-only kernels are text transforms of the shipped
sources: L2 bulk prefetch of weight tiles (gemm1_pf*, gemm2_pf*), an atomic GEMM2 tile
counter (gemm2_dyn), GEMM1 releasing dependents at its wait (gemm1_early), route words
prefetched into L1 (gemm2_pre), lane-parallel barrier init (*_pinit). All are bitwise
with the shipped pair; none was faster (B5 report).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import re
import statistics

import torch

from sgl_nvidia_mega.common import jit
from sgl_nvidia_mega.deepseek_v41.sm103.moe import routed_expert_gemm as reg

ROWS, TOP_K, EXPERTS, LAYERS = 6, 6, 384, 39
STAMP_WORDS = 32
SIDE_SMS = reg.STRIDED_SIDE_SMS


# ------------------------------------------------------------------ stamped kernel copies

_HELPERS = """
__device__ __forceinline__ unsigned long long b5_now() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned long long b5_smid() {
    unsigned int s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s; }
"""


def _sub_once(pattern, repl, text, what):
    new, count = re.subn(pattern, repl, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"stamp insertion failed: {what}")
    return new


def stamped(source: str) -> str:
    s = source.replace('extern "C" {', _HELPERS + '\nextern "C" {', 1)
    fused = "dsv41_routed_expert_gemm12(" in s
    s = _sub_once(
        r"(dsv41_routed_expert_gemm\d+\([^)]*)\)\n\{",
        r"\1, unsigned long long* __restrict__ b5_stamps)\n{",
        s,
        "signature",
    )
    s = _sub_once(
        r"(const int lane = tid % 32;\n)",
        r"\1    unsigned long long* _st = b5_stamps + (unsigned long long)blockIdx.x * "
        f"{STAMP_WORDS};\n"
        r"    if (tid == 0) { _st[0] = b5_now(); _st[1] = b5_smid(); }\n",
        s,
        "start",
    )
    s = _sub_once(
        r"(if \(warp <= 3\) \{.*?griddepcontrol\.wait;\" ::: \"memory\"\);)",
        r"\1 if (tid == 0) _st[2] = b5_now();",
        s,
        "epilogue wait",
    )
    s = _sub_once(
        r"(int expert(?:_1)? = route\[4 \+ n_tile_4\];)",
        r"\1 if (lane == 0 && _unit_4 == 0) _st[3] = b5_now();",
        s,
        "first weight TMA",
    )
    if "dsv41_routed_expert_gemm2(" in s:
        s = _sub_once(
            r"(fence\.mbarrier_init\.release\.cluster;\" ::: \"memory\"\);)",
            r"\1 _st[24] = b5_now();",
            s,
            "after init",
        )
        s = _sub_once(
            r"(\n    __syncthreads\(\);\n    asm volatile\(\"tcgen05\.fence::after_thread_sync;\" ::: \"memory\"\);)",
            r"\1\n    if (tid == 0) _st[25] = b5_now();",
            s,
            "after alloc sync",
        )
        s = _sub_once(
            r"(if \(warp == 11\) \{\n\s*\{\n)",
            r"\1            if (lane == 0) _st[26] = b5_now();\n",
            s,
            "gemm2 producer entry",
        )
        s = _sub_once(
            r"(if \(warp == 14\) \{.*?tcgen05\.fence::after_thread_sync;\"\);)",
            r"\1 if (lane == 0 && _unit_7 == 0) _st[28] = b5_now();",
            s,
            "gemm2 first mma",
        )
    if fused:
        s = _sub_once(
            r"(\n(\s*)acc_stage \+= 1;)",
            r"\n\2if (tid == 0 && _unit < 4) { _st[4 + _unit] = b5_now(); _st[8] = _unit + 1; }\1",
            s,
            "phase-1 tile end",
        )
        s = _sub_once(
            r"(uint8_t\* b5_H8 = reinterpret_cast<uint8_t\*>\(H\);)",
            r"\1 if (threadIdx.x == 0) b5_stamps[(unsigned long long)blockIdx.x * "
            + str(STAMP_WORDS)
            + "+ 9] = b5_now();",
            s,
            "phase-2 start",
        )
        idx = s.index("uint8_t* b5_H8")
        tail = s[idx:]
        tail = _sub_once(
            r"(\n(\s*)acc_stage \+= 1;)",
            r"\n\2if (tid == 0 && _unit < 20) { b5_stamps[(unsigned long long)blockIdx.x * "
            + str(STAMP_WORDS)
            + " + 10 + _unit] = b5_now(); b5_stamps[(unsigned long long)blockIdx.x * "
            + str(STAMP_WORDS)
            + r" + 30] = _unit + 1; }\1",
            tail,
            "phase-2 tile end",
        )
        s = s[:idx] + tail
    else:
        s = _sub_once(
            r"(\n(\s*)acc_stage \+= 1;)",
            r"\n\2if (tid == 0 && _unit < 26) { _st[4 + _unit] = b5_now(); _st[30] = _unit + 1; }\1",
            s,
            "tile end",
        )
    s = _sub_once(
        r"(\n    if \(warp == 0\) \{\n        asm volatile\(\"tcgen05\.dealloc)",
        r"\n    if (threadIdx.x == 0) b5_stamps[(unsigned long long)blockIdx.x * "
        + str(STAMP_WORDS)
        + r" + 31] = b5_now();\1",
        s,
        "end",
    )
    return s


# ------------------------------------------------- L2 prefetch variants (bench-only)

_PF = """
__device__ __forceinline__ void b5_prefetch(const unsigned char* p, unsigned int bytes, unsigned int piece) {
    for (unsigned int o = 0; o < bytes; o += piece)
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p + o), "r"(min(piece, bytes - o)) : "memory");
}
"""
G1_TILE = 128 * reg.HIDDEN // 2  # one 128-row w13 tile, contiguous
G2_TILE = 128 * reg.PADDED // 2  # one 128-row w2 tile, contiguous


def _with_weight_param(s):
    s = s.replace('extern "C" {', _PF + '\nextern "C" {', 1)
    return _sub_once(
        r"(dsv41_routed_expert_gemm\d\([^)]*)\)\n\{",
        r"\1, const unsigned char* __restrict__ b5_w)\n{",
        s,
        "weight param",
    )


def gemm1_prefetch(source: str, mode: str, piece: int = 32768) -> str:
    """mode cur: the TMA warp prefetches each tile's contiguous 320 KB into L2 as it starts
    it; next: also the CTA's next tile; first: only the first tile."""
    s = _with_weight_param(source)
    tile = f"((unsigned long long)expert_1 * {2 * reg.PADDED} + m_tile_4 * 128u) * {reg.HIDDEN // 2}ull"
    body = f"if (elect_sync()) {{ b5_prefetch(b5_w + {tile}, {G1_TILE}u, {piece}u);"
    if mode == "first":
        body = f"if (_unit_4 == 0 && elect_sync()) {{ b5_prefetch(b5_w + {tile}, {G1_TILE}u, {piece}u);"
    if mode == "next":
        body += (
            " unsigned int _nt = n_tile_4 * 9u + m_tile_4 + wl_balanced_stride(route, 9u);"
            " if (_nt / 9u < (unsigned int)route[1]) b5_prefetch(b5_w + ((unsigned long long)route[4 + _nt / 9u]"
            f" * {2 * reg.PADDED} + (_nt % 9u) * 128u) * {reg.HIDDEN // 2}ull, {G1_TILE}u, {piece}u);"
        )
    body += " }"
    return _sub_once(
        r"(int expert_1 = route\[4 \+ n_tile_4\];)", r"\1 " + body, s, "gemm1 prefetch"
    )


def gemm2_prefetch(source: str, skip: int = 1, depth: int = 64) -> str:
    """Before the loop (pre-wait), the TMA warp prefetches the w2 tiles of this CTA's
    units skip..skip+depth into L2."""
    s = _with_weight_param(source)
    pre = (
        "if (lane == 0) { for (unsigned int _u = "
        + str(skip)
        + "u; _u < "
        + str(skip + depth)
        + "u; ++_u) {"
        " unsigned int _t = blockIdx.x + _u * gridDim.x; unsigned int _n = _t / 40u;"
        " if (_n >= (unsigned int)route[1]) break;"
        f" b5_prefetch(b5_w + ((unsigned long long)route[4 + _n] * {reg.HIDDEN} + (_t % 40u) * 128u) * {reg.PADDED // 2}ull, {G2_TILE}u, {G2_TILE}u); }} }}"
    )
    return _sub_once(
        r"(if \(warp == 11\) \{\n\s*\{\n)",
        r"\1            " + pre.replace("\\", "\\\\") + "\n",
        s,
        "gemm2 prefetch",
    )


def dynamic_schedule(source: str, units: int) -> str:
    """Tiles past the first round are claimed from a per-launch global counter instead
    of the static stride; the last CTA to finish resets the counter for the next launch.
    """
    s = _sub_once(
        r"(dsv41_routed_expert_gemm\d\([^)]*)\)\n\{",
        r"\1, unsigned int* __restrict__ b5_sched)\n{",
        source,
        "sched param",
    )
    s = _sub_once(
        r"unsigned int wl_t = n_tile_8 \* %du \+ m_tile_8 \+ (gridDim\.x|wl_stride_8);"
        % units,
        r"unsigned int wl_t = gridDim.x + atomicAdd(b5_sched, 1u);",
        s,
        "dynamic claim",
    )
    return _sub_once(
        r"(\n    __syncthreads\(\);\n)(\n    if \(warp == 0\) \{\n        asm volatile\(\"tcgen05\.dealloc)",
        r"\1    if (threadIdx.x == 0) { __threadfence(); if (atomicAdd(b5_sched + 1, 1u) == gridDim.x - 1u) { b5_sched[0] = 0u; b5_sched[1] = 0u; __threadfence(); } }\n\2",
        s,
        "dynamic reset",
    )


_FUSE_HELPERS = """
__device__ __forceinline__ void b5_wait(const unsigned int* p, unsigned int target) {
    unsigned int v;
    do { asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); } while (v < target);
}
__device__ __forceinline__ void b5_signal(unsigned int* p) {
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(p) : "memory");
}
"""

_DEALLOC = re.compile(
    r"\n    if \(warp == 0\) \{\n        asm volatile\(\"tcgen05\.dealloc[^\n]*\n    \}"
)


def _kernel_parts(text):
    head, kernel = text.split('extern "C" {', 1)
    start = kernel.index(")\n{") + 3
    end = kernel.rindex("\n}\n\n}")
    return head + kernel[: kernel.index("__global__")], kernel[start:end]


def fuse_gemm12(g1: str, g2: str) -> str:
    """One persistent launch: each CTA runs its GEMM1 tiles, then (same smem, same TMEM)
    its GEMM2 tiles; a GEMM2 tile waits on its expert's nine GEMM1 tiles through a
    per-expert counter instead of on the whole GEMM1 grid."""
    head, body1 = _kernel_parts(g1)
    _, body2 = _kernel_parts(g2)
    barriers = re.findall(r"mbarrier_init\(smem \+ (\d+), \d+\)", body1)
    body1 = body1.replace(
        'asm volatile("griddepcontrol.launch_dependents;" ::: "memory");', "", 1
    )
    signal = (
        '__threadfence(); asm volatile("barrier.sync 6, 128;" ::: "memory");'
        " if (tid == 0) b5_signal(b5_sync + 1 + n_tile);"
    )
    body1, n = re.subn(
        r"(\n(\s*)acc_stage \+= 1;)",
        lambda m: "\n" + m.group(2) + signal + m.group(1),
        body1,
        count=1,
    )
    assert n == 1
    inval = " ".join(
        f'asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(smem + {b}) : "memory");'
        for b in barriers
    )
    body1, n = _DEALLOC.subn(
        "\n    b5_taddr = taddr;\n    if (warp == 0 && lane == 0) { " + inval + " }",
        body1,
    )
    assert n == 1
    undef = "".join(
        f"#undef {name}\n" for name in re.findall(r"#define (\w+_addr) ", body2)
    )
    body2 = re.sub(r"\n\s*asm volatile\(\"setmaxnreg[^\n]*", "", body2)
    body2 = _sub_once(
        r"\n    if \(warp == 0\) \{\n        int _tmem_hold.*?relinquish_alloc_permit[^\n]*\n    \}",
        "",
        body2,
        "gemm2 alloc",
    )
    body2 = body2.replace("tmem_addr_storage[0]", "b5_taddr")
    body2 = body2.replace("(&A)", "(&A2)").replace("(&SFA)", "(&SFA2)")
    body2 = body2.replace('"l"(H + (', '"l"(b5_H8 + (')
    body2 = _sub_once(
        r"(int valid_rows_1 = [^;]*;)",
        r"\1 b5_wait(b5_sync + 1 + n_tile_2, 9u);",
        body2,
        "wait B",
    )
    body2 = _sub_once(
        r"(int valid_rows_2 = [^;]*;)",
        r"\1 b5_wait(b5_sync + 1 + n_tile_3, 9u);",
        body2,
        "wait SFB",
    )
    reset = (
        "\n    if (threadIdx.x == 0) { __threadfence();"
        " if (atomicAdd(b5_sync, 1u) == gridDim.x - 1u) {"
        " const unsigned int n = (unsigned int)route[1];"
        " for (unsigned int i = 0; i < n; ++i) b5_sync[1 + i] = 0u;"
        " b5_sync[0] = 0u; __threadfence(); } }\n"
    )
    kernel = (
        "__global__ __launch_bounds__(512, 1) void\n"
        "dsv41_routed_expert_gemm12(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap SFA,"
        " uint8_t* __restrict__ X, uint8_t* __restrict__ XS, int* route, float* clamp,"
        " uint16_t* __restrict__ H, uint8_t* __restrict__ HS, int cta_words,"
        " const __grid_constant__ CUtensorMap A2, const __grid_constant__ CUtensorMap SFA2,"
        " unsigned int* __restrict__ Y, unsigned int* __restrict__ b5_sync)\n{\n"
        "    int b5_taddr = 0;\n"
        "    {\n"
        + body1
        + "\n    }\n    __syncthreads();\n"
        + undef
        + "    {\n    uint8_t* b5_H8 = reinterpret_cast<uint8_t*>(H);\n"
        + body2
        + reset
        + "    }\n}\n\n}\n"
    )
    return head + _FUSE_HELPERS + 'extern "C" {\n\n' + kernel


def gemm2_route_prefetch(source: str) -> str:
    """At entry, before the barrier set-up, thread 0 pulls the route words the first
    tile reads into L1 (the record is complete before GEMM2 can launch)."""
    words = (
        "route + 1",
        "route + 4 + blockIdx.x / 40u",
        "route + 4 + cta_words + blockIdx.x / 40u",
    )
    pre = (
        "    if (tid == 0) { "
        + " ".join(
            f'asm volatile("prefetch.global.L1 [%0];" :: "l"({w}));' for w in words
        )
        + " }\n"
    )
    new, n = re.subn(
        r"(const int lane = tid % 32;\n)", lambda m: m.group(1) + pre, source, count=1
    )
    assert n == 1
    return new


def gemm1_early_trigger(source: str) -> str:
    """GEMM1 releases its dependents right after its own grid-dependency wait instead of
    after its last weight load, so GEMM2 CTAs take SMs no GEMM1 CTA holds."""
    trig = 'asm volatile("griddepcontrol.launch_dependents;" ::: "memory");'
    assert source.count(trig) == 1
    s = source.replace(trig, "")
    new, n = re.subn(
        r"(if \(warp == 11\) \{\n\s*\{\n\s*asm volatile\(\"griddepcontrol\.wait;\" ::: \"memory\"\);)",
        lambda m: m.group(1) + " " + trig,
        s,
        count=1,
    )
    assert n == 1
    return new


def parallel_barrier_init(source: str) -> str:
    """The prologue's mbarrier inits spread over warp 0's lanes instead of one thread."""
    m = re.search(
        r"    if \(warp == 0\) \{\n        uint32_t leader = elect_sync\(\);\n        if \(leader\) \{\n"
        r"((?:\s*mbarrier_init\(smem \+ \d+, \d+\);\n)+)"
        r"\s*asm volatile\(\"fence\.mbarrier_init\.release\.cluster;\" ::: \"memory\"\);\n        \}\n    \}\n",
        source,
    )
    assert m, "barrier init block"
    inits = re.findall(r"mbarrier_init\(smem \+ (\d+), (\d+)\)", m.group(1))
    offs = ", ".join(o for o, _ in inits)
    cnts = ", ".join(c for _, c in inits)
    block = (
        "    if (warp == 0) {\n"
        f"        const unsigned short b5_off[{len(inits)}] = {{{offs}}};\n"
        f"        const unsigned short b5_cnt[{len(inits)}] = {{{cnts}}};\n"
        f"        for (int i = lane; i < {len(inits)}; i += 32) mbarrier_init(smem + b5_off[i], b5_cnt[i]);\n"
        '        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n'
        "    }\n"
    )
    return source[: m.start()] + block + source[m.end() :]


def prefetch_sources() -> dict:
    g1 = (reg.SOURCE_DIR / "gemm1_strided_balanced.cu").read_text()
    g2 = (reg.SOURCE_DIR / "gemm2_strided.cu").read_text()
    return {
        "gemm1_pfcur": gemm1_prefetch(g1, "cur"),
        "gemm1_pfnext": gemm1_prefetch(g1, "next"),
        "gemm1_pffirst": gemm1_prefetch(g1, "first"),
        "gemm2_pf1": gemm2_prefetch(g2, 1),
        "gemm2_pf2": gemm2_prefetch(g2, 2),
        "gemm2_dyn": dynamic_schedule(g2, 40),
        "gemm1_fused": fuse_gemm12(g1, g2),
        "gemm2_pre": gemm2_route_prefetch(g2),
        "gemm1_early": gemm1_early_trigger(g1),
        "gemm2_pinit": parallel_barrier_init(g2),
        "gemm1_pinit": parallel_barrier_init(g1),
    }


SOURCES = {
    "gemm1": "gemm1.cu",
    "gemm2": "gemm2.cu",
    "gemm1_strided": "gemm1_strided.cu",
    "gemm1_balanced": "gemm1_strided_balanced.cu",
    "gemm2_strided": "gemm2_strided.cu",
}


def load_kernels(device_index: int, extra: dict | None = None) -> dict:
    """Every GEMM source, plain and stamped, plus the route kernel."""
    major, minor = torch.cuda.get_device_capability(device_index)
    arch = f"sm_{major}{minor}a"
    facts = json.loads((reg.SOURCE_DIR / "launch.json").read_text())
    kernels = {"route": reg.load_kernels(device_index)["route"]}
    sources = {k: (reg.SOURCE_DIR / f).read_text() for k, f in SOURCES.items()}
    sources.update(extra or {})
    for name, text in sources.items():
        entry = "gemm1" if name.startswith("gemm1") else "gemm2"
        fact = facts[entry]
        smem = int(fact["shared_memory_bytes"])
        m = re.search(r"#define SMEM_TOTAL (\d+)", text)
        if m:
            smem = int(m.group(1))
        variants = [("", text)]
        try:
            variants.append(("+st", stamped(text)))
        except RuntimeError:
            pass  # the clang-formatted cluster-launch-control sources: timing only
        for tag, src in variants:
            kernels[name + tag] = jit.load(
                reg.compile_source(src, name + tag.replace("+", "_"), arch),
                "dsv41_routed_expert_gemm12" if "gemm12(" in text else fact["symbol"],
                fact["block"],
                smem,
            )
    return kernels


# ------------------------------------------------------------------------------- layers


def random_layer(generator, device):
    e, h, p = EXPERTS, reg.HIDDEN, reg.PADDED

    def rand(shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, dtype=torch.uint8, generator=generator).to(
            device
        )

    w13 = rand((e, 2 * p, h // 2))
    w13_s = rand((e, 2 * p, h // 32), 118, 126)
    w2 = rand((e, h, p // 2))
    w2_s = rand((e, h, p // 32), 118, 126)
    clamp = torch.full((e,), 10.0, dtype=torch.float32, device=device)
    tensors = (w13, w13_s, w2, w2_s, clamp)
    maps1 = (reg._weight_map(w13, 2 * p, h, 2), reg._scale_map(w13_s, 4))
    maps2 = (reg._weight_map(w2, h, p, 2), reg._scale_map(w2_s, 1))
    return tensors, maps1, maps2


def routing(distinct: int, generator) -> torch.Tensor:
    """6 x 6 ids over exactly ``distinct`` experts (6 <= distinct <= 36)."""
    pool = torch.randperm(EXPERTS, generator=generator)[:distinct]
    while True:
        rows = [
            pool[torch.randperm(distinct, generator=generator)[:TOP_K]]
            for _ in range(ROWS)
        ]
        ids = torch.stack(rows)
        if ids.unique().numel() == distinct:
            return ids.to(torch.int32)
        # force coverage: put the missing experts into random free slots
        missing = [x for x in pool.tolist() if x not in set(ids.flatten().tolist())]
        flat = ids.flatten().tolist()
        counts = {x: flat.count(x) for x in flat}
        for x in missing:
            for i in torch.randperm(len(flat), generator=generator).tolist():
                r = i // TOP_K
                if counts[flat[i]] > 1 and x not in flat[r * TOP_K : (r + 1) * TOP_K]:
                    counts[flat[i]] -= 1
                    flat[i] = x
                    counts[x] = 1
                    break
        ids = torch.tensor(flat).view(ROWS, TOP_K)
        if ids.unique().numel() == distinct:
            return ids.to(torch.int32)


# ------------------------------------------------------------------------------ variants


def grids(variant: str, ctas: int, sms: int):
    """(gemm1 name, grid1), (gemm2 name, grid2) of a variant."""
    strided = sms - SIDE_SMS
    t1, t2 = reg.GEMM1_UNITS * ctas, reg.GEMM2_UNITS * ctas
    if variant == "clc":
        return ("gemm1", (reg.GEMM1_UNITS, ctas, 1)), (
            "gemm2",
            (reg.GEMM2_UNITS, ctas, 1),
        )
    if variant == "strided":
        g1 = min(strided, t1)
        return (
            ("gemm1_balanced" if t1 > g1 else "gemm1_strided", (g1, 1, 1)),
            ("gemm2_strided", (min(strided, t2), 1, 1)),
        )
    if variant.startswith("fused"):
        # fused[:<grid>]: one launch of GEMM1 then GEMM2 tiles per CTA
        grid = int(variant.split(":")[1]) if ":" in variant else strided
        return ("gemm1_fused", (min(grid, t1), 1, 1)), (None, (1, 1, 1))
    if variant.startswith("custom:"):
        # custom:<gemm1 name>:<grid1>:<gemm2 name>:<grid2>
        _, n1, g1, n2, g2 = variant.split(":")
        return (n1, (min(int(g1), t1), 1, 1)), (n2, (min(int(g2), t2), 1, 1))
    raise ValueError(variant)


class Bench:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        g = torch.Generator().manual_seed(7)
        self.layers = [random_layer(g, self.device) for _ in range(2)]
        self.kernels = load_kernels(self.device.index, prefetch_sources())
        from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

        x = torch.randn(ROWS, reg.HIDDEN, generator=g).to(torch.bfloat16).cuda()
        self.x_bf16 = x
        x_q, x_s = flashinfer_mxfp8_quantize(x, False, alignment=reg.HIDDEN)
        self.x_q, self.x_s = x_q, x_s.view(torch.uint8).reshape(ROWS, -1)
        self.ctas = reg.max_ctas(ROWS, TOP_K, EXPERTS)
        self.words = reg.cta_words(self.ctas)
        self.barrier = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.flag = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.side = SideChain(self.device, g) if 1 in args.side else None
        self.side_stream = torch.cuda.Stream(self.device)

    def buffers(self):
        padded = reg.TILE * self.ctas
        return dict(
            sched=torch.zeros(2, dtype=torch.int32, device=self.device),
            sync=torch.zeros(1 + self.ctas, dtype=torch.int32, device=self.device),
            permuted=torch.empty((ROWS, TOP_K), dtype=torch.int32, device=self.device),
            record=torch.empty(
                reg.route_words(self.ctas), dtype=torch.int32, device=self.device
            ),
            h=torch.empty((padded, reg.PADDED), dtype=torch.uint8, device=self.device),
            hs=torch.empty(
                (padded + 127) // 128 * 128 * reg.PADDED // 32,
                dtype=torch.uint8,
                device=self.device,
            ),
            y=torch.empty(
                (ROWS * TOP_K * reg.TILE, reg.HIDDEN),
                dtype=torch.bfloat16,
                device=self.device,
            ),
        )

    def route(self, ids, buf):
        reg._launch(
            self.kernels["route"],
            (1, 1, 1),
            [
                reg._pointer(ids),
                reg._pointer(buf["permuted"]),
                reg._pointer(buf["record"]),
                reg._pointer(self.barrier),
                ctypes.c_int32(ROWS),
                ctypes.c_int32(TOP_K),
                ctypes.c_int32(EXPERTS),
            ],
            torch.cuda.current_stream().cuda_stream,
        )

    def half(self, layer, ids, buf, variant, stamps=None, x_q=None, x_s=None):
        stream = torch.cuda.current_stream().cuda_stream
        _, maps1, maps2 = layer
        if variant.startswith("preroute+"):
            variant = variant[len("preroute+") :]
        else:
            self.route(ids, buf)
        (n1, g1), (n2, g2) = grids(variant, self.ctas, self.sms)
        tag = "+st" if stamps is not None else ""
        words = ctypes.c_int32(self.words)
        extra1 = [reg._pointer(layer[0][0])] if "_pf" in n1 else []
        if n2 is None:
            extra1 = [*maps2, reg._pointer(buf["y"]), reg._pointer(buf["sync"])]
        extra2 = [reg._pointer(layer[0][2])] if n2 and "_pf" in n2 else []
        if n2 and "_dyn" in n2:
            extra2.append(reg._pointer(buf["sched"]))
        if stamps is not None:
            extra1.append(reg._pointer(stamps[0]))
            extra2.append(reg._pointer(stamps[1]))
        reg._launch(
            self.kernels[n1 + tag],
            g1,
            [
                *maps1,
                reg._pointer(self.x_q if x_q is None else x_q),
                reg._pointer(self.x_s if x_s is None else x_s),
                reg._pointer(buf["record"]),
                reg._pointer(layer[0][4]),
                reg._pointer(buf["h"]),
                reg._pointer(buf["hs"]),
                words,
                *extra1,
            ],
            stream,
        )
        if n2 is None:
            return
        reg._launch(
            self.kernels[n2 + tag],
            g2,
            [
                *maps2,
                reg._pointer(buf["h"]),
                reg._pointer(buf["hs"]),
                reg._pointer(buf["record"]),
                reg._pointer(buf["y"]),
                words,
                *extra2,
            ],
            stream,
        )

    def graph(self, distinct, variant, side, stamped=False, spacers_only=False):
        g = torch.Generator().manual_seed(1000 + distinct)
        ids = [routing(distinct, g).to(self.device) for _ in range(LAYERS)]
        bufs = [self.buffers() for _ in range(LAYERS)]
        stamps = None
        if stamped:
            (_, g1), (_, g2) = grids(variant.split("+")[-1], self.ctas, self.sms)
            n1 = g1[0] * g1[1] * g1[2]
            n2 = g2[0] * g2[1] * g2[2]
            stamps = [
                (
                    torch.zeros(
                        n1 * STAMP_WORDS, dtype=torch.int64, device=self.device
                    ),
                    torch.zeros(
                        n2 * STAMP_WORDS, dtype=torch.int64, device=self.device
                    ),
                )
                for _ in range(LAYERS)
            ]

        def body():
            main = torch.cuda.current_stream()
            for i in range(LAYERS):
                if variant.startswith("preroute+") and not spacers_only:
                    self.route(ids[i], bufs[i])  # off the measured chain
                self.flag.add_(1)  # spacer: no PDL, the next half starts after this
                if spacers_only:
                    continue
                if side:
                    self.side_stream.wait_stream(main)
                    with torch.cuda.stream(self.side_stream):
                        self.side()
                self.half(
                    self.layers[i % 2],
                    ids[i],
                    bufs[i],
                    variant,
                    stamps[i] if stamps else None,
                )
                if side:
                    main.wait_stream(self.side_stream)

        body()  # warm-up and compile outside capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        torch.cuda.synchronize()
        return graph, stamps, bufs, ids

    def time(self, graph, replays):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        graph.replay()
        torch.cuda.synchronize()
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1e3 / replays / LAYERS


class SideChain:
    """A shared-expert-sized chain (row-tile MXFP8 quant, 1152 x 5120 MXFP8 gate_up on the
    low-latency kernel, clamped SwiGLU, MXFP8 quant, 5120 x 640 down) on its own weights.
    """

    def __init__(self, device, g):
        from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll

        self.mxfp8_ll = mxfp8_ll

        def weight(n, k):
            w = (
                (torch.randn(n, k, generator=g) * 0.05)
                .to(device)
                .to(torch.float8_e4m3fn)
            )
            s = torch.randint(
                120, 128, (n * k // 32,), dtype=torch.uint8, generator=g
            ).to(device)
            return w, s

        self.gate_up = weight(2 * reg.SLICE, reg.HIDDEN)
        self.down = weight(reg.HIDDEN, reg.PADDED)
        self.x = (
            torch.randn(ROWS, reg.HIDDEN, generator=g).to(torch.bfloat16).to(device)
        )
        mxfp8_ll.precompile(self.gate_up[0], [ROWS])
        mxfp8_ll.precompile(self.down[0], [ROWS])

    def __call__(self):
        from sgl_nvidia_mega.deepseek_v41.sm103.attention.q_norm_quant import (
            mxfp8_row_tiles,
        )

        x_q, x_s = mxfp8_row_tiles(self.x)
        gu = self.mxfp8_ll.linear(x_q, x_s, *self.gate_up)
        from sgl_nvidia_mega.deepseek_v41 import sglang_ops

        mid = torch.empty(ROWS, reg.SLICE, dtype=torch.bfloat16, device=gu.device)
        sglang_ops.silu_and_mul_clamp(gu, mid, 10.0)
        h = torch.nn.functional.pad(mid, (0, reg.PADDED - reg.SLICE))
        h_q, h_s = self.mxfp8_ll.quantize(h)
        return self.mxfp8_ll.linear(h_q, h_s, *self.down)


# ------------------------------------------------------------------------------ analysis


def _fused_facts(a, tiles):
    t0 = min(r[0] for r in a)
    us = lambda t: (t - t0) / 1e3  # noqa: E731
    p1 = [r for r in a if r[8] > 0]
    p1_last = [us(r[4 + r[8] - 1]) for r in p1]
    p2_start = sorted(us(r[9]) for r in a)
    p2 = [r for r in a if r[30] > 0]
    first2 = [(r[10] - r[9]) / 1e3 for r in p2]
    mid2 = [
        (r[10 + i + 1] - r[10 + i]) / 1e3 for r in p2 for i in range(min(r[30], 20) - 1)
    ]
    return dict(
        tiles=tiles,
        g1_wait=us(min(r[2] for r in a if r[2])),
        g1_busy=len(p1),
        g1_lasttile_p0=min(p1_last),
        g1_end=max(p1_last),
        p2_start_p0=p2_start[0],
        p2_start_p50=p2_start[len(p2_start) // 2],
        p2_start_max=p2_start[-1],
        p2_first_p50=statistics.median(first2),
        p2_mid_p50=statistics.median(mid2) if mid2 else 0.0,
        p2_rounds=max(r[30] for r in a),
        g2_end=us(max(r[31] for r in a)),
        g2_tail=us(max(r[31] for r in a)) - max(p1_last),
    )


def analyse(stamps, bufs, variant, ctas, sms):
    """Per-layer timeline facts (us) from the stamp buffers, medians over layers."""
    facts = []
    for (s1, s2), buf in zip(stamps, bufs):
        tiles = int(buf["record"][1].item())
        a = s1.view(-1, STAMP_WORDS).cpu().tolist()
        b = s2.view(-1, STAMP_WORDS).cpu().tolist()
        t0 = min(r[0] for r in a)
        us = lambda t: (t - t0) / 1e3  # noqa: E731
        if variant.startswith("fused"):
            facts.append(_fused_facts(a, tiles))
            continue
        busy1 = [r for r in a if r[30] > 0]
        busy2 = [r for r in b if r[30] > 0]
        g1_end = max(r[31] for r in a)
        g1_wait = min(r[2] for r in a if r[2])
        g2_first = min(r[0] for r in b)
        g2_end = max(r[31] for r in b)
        g2_wait = min(r[2] for r in b if r[2])
        resident = sum(1 for r in b if r[0] < g1_end)
        # gemm1 per-tile durations (consecutive tile ends of one CTA, first from wait)
        d1 = []
        for r in busy1:
            ends = [r[4 + i] for i in range(min(r[30], 26))]
            prev = r[2]
            for e in ends:
                d1.append((e - prev) / 1e3)
                prev = e
        d2 = []
        for r in busy2:
            ends = [r[4 + i] for i in range(min(r[30], 26))]
            prev = max(r[2], g1_end)
            for e in ends:
                d2.append((e - prev) / 1e3)
                prev = e
        g2_starts = sorted(us(r[0]) for r in b)
        late = sum(1 for r in b if r[0] >= g1_end)
        facts.append(
            dict(
                tiles=tiles,
                g1_ctas=len(a),
                g1_busy=len(busy1),
                g1_rounds=max(r[30] for r in a),
                g1_wait=us(g1_wait),
                g1_end=us(g1_end),
                g1_last_start=us(max(r[0] for r in a)),
                g2_first=us(g2_first),
                g2_wait=us(g2_wait),
                g2_end=us(g2_end),
                g2_ctas=len(b),
                g2_busy=len(busy2),
                g2_rounds=max(r[30] for r in b),
                g2_resident_before_g1_end=resident,
                g2_start_after_g1_end=late,
                g2_last_start=g2_starts[-1],
                g2_p50_start=g2_starts[len(g2_starts) // 2],
                g1_tile_p50=statistics.median(d1) if d1 else 0.0,
                g2_tile_p50=statistics.median(d2) if d2 else 0.0,
                g2_tail=us(g2_end) - us(g1_end),
            )
        )
    keys = facts[0].keys()
    return {k: statistics.median(f[k] for f in facts) for k in keys}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--distinct", type=int, nargs="+", default=[16, 20, 24, 28, 32])
    p.add_argument("--variants", nargs="+", default=["strided"])
    p.add_argument("--side", type=int, nargs="+", default=[0, 1])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--replays", type=int, default=20)
    p.add_argument("--stamps", default="")
    p.add_argument("--bandwidth", action="store_true")
    p.add_argument(
        "--check", action="store_true", help="bitwise rows vs the first variant"
    )
    args = p.parse_args()
    bench = Bench(args)
    print(f"sms {bench.sms} ctas {bench.ctas}", flush=True)
    if args.bandwidth:
        big = torch.empty(4 << 30, dtype=torch.uint8, device=bench.device).view(
            torch.float32
        )
        big.normal_()
        for _ in range(3):
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            big.sum()
            e.record()
            torch.cuda.synchronize()
            print(
                f"read bandwidth {big.numel() * 4 / s.elapsed_time(e) / 1e9:.2f} TB/s"
            )
        del big
    spacer, *_ = bench.graph(16, "strided", False, spacers_only=True)
    spacer_us = statistics.median(bench.time(spacer, args.replays) for _ in range(3))
    print(f"spacer {spacer_us:.2f} us/half", flush=True)
    if args.check:
        for distinct in args.distinct:
            outs = {}
            for v in args.variants:
                graph, _, bufs, _ = bench.graph(distinct, v, False)
                graph.replay()
                torch.cuda.synchronize()
                outs[v] = [
                    (b["y"][b["permuted"].flatten().long()].clone(), b["h"].clone())
                    for b in bufs
                ]
                del graph
            ref = outs[args.variants[0]]
            for v in args.variants[1:]:
                same = all(
                    torch.equal(a[0].view(torch.int16), b[0].view(torch.int16))
                    for a, b in zip(ref, outs[v])
                )
                print(f"check D={distinct} {v}: {'bitwise' if same else 'MISMATCH'}")
    results, raw = [], []
    for distinct in args.distinct:
        for side in args.side:
            graphs = {v: bench.graph(distinct, v, side)[0] for v in args.variants}
            times = {v: [] for v in args.variants}
            for r in range(args.repeats):
                order = args.variants if r % 2 == 0 else args.variants[::-1]
                for v in order:
                    times[v].append(bench.time(graphs[v], args.replays) - spacer_us)
            for v in args.variants:
                t = times[v]
                row = dict(
                    distinct=distinct,
                    side=side,
                    variant=v,
                    us=statistics.median(t),
                    lo=min(t),
                    hi=max(t),
                )
                results.append(row)
                print(
                    f"D={distinct:2d} side={side} {v:40s} {row['us']:7.2f} us "
                    f"[{row['lo']:.2f} .. {row['hi']:.2f}]",
                    flush=True,
                )
            del graphs
            if args.stamps:
                for v in args.variants:
                    if v == "clc":
                        continue
                    graph, stamps, bufs, _ = bench.graph(
                        distinct, v, side, stamped=True
                    )
                    for s1, s2 in stamps:
                        s1.zero_()
                        s2.zero_()
                    graph.replay()
                    torch.cuda.synchronize()
                    facts = analyse(stamps, bufs, v, bench.ctas, bench.sms)
                    facts.update(distinct=distinct, side=side, variant=v)
                    results.append(dict(stamps=facts))
                    raw.append(
                        dict(
                            distinct=distinct,
                            side=side,
                            variant=v,
                            layers=[
                                dict(
                                    tiles=int(b["record"][1].item()),
                                    g1=s1.view(-1, STAMP_WORDS).cpu().tolist(),
                                    g2=s2.view(-1, STAMP_WORDS).cpu().tolist(),
                                )
                                for (s1, s2), b in list(zip(stamps, bufs))[:8]
                            ],
                        )
                    )
                    print(
                        "  stamps "
                        + " ".join(
                            f"{k}={val:.2f}" if isinstance(val, float) else f"{k}={val}"
                            for k, val in facts.items()
                        ),
                        flush=True,
                    )
                    del graph, stamps
    if args.stamps:
        with open(args.stamps, "w") as f:
            json.dump(results, f, indent=1)
        with open(args.stamps.replace(".json", "_raw.json"), "w") as f:
            json.dump(raw, f)


if __name__ == "__main__":
    main()
