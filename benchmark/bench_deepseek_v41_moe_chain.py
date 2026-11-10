# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DeepSeek-V4.1 MoE half at batch 1 (6 verify rows) as the plugin chains it, from
hop A's output to the routed GEMM pair's consumer, with fixed routing, replayed from a
CUDA graph of 39 halves:

  hop A stand-in (PDL, early trigger, writes the FFN input x)
    -> router GEMM (SGLang's tiny_n_gemm, or a bitwise replica that can stamp)
       | routed-input MXFP8 quant on its own stream, joined before the top-k (as SGLang)
    -> top-k (gate_topk.cu) -> route (route.cu)      [variant base]
       or top-k + route in one cluster launch       [variant fused: slot gate_route]
    -> GEMM1 -> GEMM2 -> join of the shared-expert chain (alternate stream) -> consumer
       stand-in (PDL, as hop B's finalize)

  python benchmark/bench_deepseek_v41_moe_chain.py --local-experts 192 --offset 0 \
      --variants base fused --gemm clc --side 0 1 --repeats 6 [--check] [--stamps out.json]

Routing is fixed: each layer's router weight is built so its logits pick a chosen set of
experts per row (``--pool`` global experts, half of them in the rank's window under EP),
so every variant and repeat routes the same. Reports us per half (graph time / 39), the
variants in alternating order per repeat. With --stamps, copies of hop A, the router
replica, the top-k, route and fused kernels and the strided GEMM pair write
%globaltimer per CTA (entry, after the grid-dependency wait, milestones, end); the
timeline is reported relative to the end of hop A's last CTA, medians over layers.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_deepseek_v41_routed_moe as b5  # noqa: E402

from sgl_nvidia_mega.common import jit  # noqa: E402
from sgl_nvidia_mega.deepseek_v41.sm103.moe import routed_expert_gemm as reg  # noqa: E402

ROWS, TOP_K, ROUTER_EXPERTS, LAYERS = 6, 6, 384, 39
HIDDEN = reg.HIDDEN
STAMPS = 8  # words per CTA; word 7 = %smid
HOP_A_CTAS = 30
CONSUMER_CTAS = 60

_STAMP_HEAD = r"""
__device__ __forceinline__ unsigned long long n3_now() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned int n3_smid() {
  unsigned int s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s; }
__shared__ unsigned long long* n3_st;
#define DSV41_STAMP(i) do { if (threadIdx.x == 0 && n3_st != nullptr) n3_st[blockIdx.x * 8 + (i)] = n3_now(); } while (0)
#define N3_BEGIN(stamps) do { if (threadIdx.x == 0) { n3_st = (stamps); if (n3_st != nullptr) n3_st[blockIdx.x * 8 + 7] = n3_smid(); } } while (0)
"""

_GATE_SOURCE = (
    _STAMP_HEAD
    + r"""
#include "gate_route.cuh"

extern "C" __global__ __launch_bounds__(384) void n3_gate_topk(
    const float* logits, const float* bias, const float* bias_alt, const void* input_ids, int ids64,
    long long alt_token_id, const int* num_valid, float* __restrict__ weights, int* __restrict__ ids,
    float scale, float eps, unsigned long long* stamps) {
  N3_BEGIN(stamps);
  GateTopK<384, 6>(logits, bias, bias_alt, input_ids, ids64, alt_token_id, num_valid, weights, ids, scale, eps);
}

extern "C" __global__ __launch_bounds__(384) void n3_gate_route(
    const float* logits, const float* bias, const float* bias_alt, const void* input_ids, int ids64,
    long long alt_token_id, const int* num_valid, float* __restrict__ weights, int* __restrict__ ids,
    float scale, float eps, int* __restrict__ permuted, int* __restrict__ record, int experts,
    int expert_offset, unsigned long long* stamps) {
  N3_BEGIN(stamps);
  GateRoute<384, 6>(logits, bias, bias_alt, input_ids, ids64, alt_token_id, num_valid, weights, ids, scale, eps,
                    permuted, record, experts, expert_offset);
}

// route.cu's body with stamps
extern "C" __global__ __launch_bounds__(64) void n3_route(
    const int *ids, int *__restrict__ permuted, int *__restrict__ record,
    unsigned int *barrier, int rows, int top_k, int experts, int expert_offset,
    unsigned long long* stamps) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  N3_BEGIN(stamps);
  DSV41_STAMP(0);
  __shared__ route::Scratch scratch;
  const int thread = static_cast<int>(threadIdx.x);
  route::Clear(scratch, thread, 64);
  __syncthreads();
  asm volatile("griddepcontrol.wait;" ::: "memory");
  DSV41_STAMP(1);
  route::Zero(record, rows, top_k, experts, thread, 64);
  route::Build(scratch, ids, permuted, record, barrier, rows, top_k, experts, expert_offset, thread);
  DSV41_STAMP(2);
}
"""
)

# The plain program (no ftz): hop A and consumer stand-ins, and a replica of SGLang's
# tiny_n_gemm (N 384, K 5120, N_SPLIT 3, one 32-byte K chunk per thread over 320 threads,
# the same fma order, xor butterfly and cross-warp order; the weight load before the
# grid-dependency wait), which can trigger its dependents at entry.
_MISC_SOURCE = (
    _STAMP_HEAD
    + r"""
extern "C" __global__ void n3_hop_a(const unsigned short* __restrict__ src, unsigned short* __restrict__ x,
                                    int words, unsigned long long spin_ns, int trigger_at_entry,
                                    unsigned long long* stamps) {
  if (trigger_at_entry) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  N3_BEGIN(stamps);
  DSV41_STAMP(0);
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const unsigned long long t = n3_now();
  DSV41_STAMP(1);
  while (n3_now() < t + spin_ns) {}
  const int per = (words + gridDim.x - 1) / gridDim.x;
  const int lo = blockIdx.x * per;
  const int hi = min(words, lo + per);
  for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) x[i] = src[i];
  __syncthreads();
  DSV41_STAMP(2);
  if (!trigger_at_entry) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

extern "C" __global__ void n3_consumer(const unsigned short* __restrict__ y, unsigned long long spin_ns,
                                       unsigned long long* stamps) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  N3_BEGIN(stamps);
  DSV41_STAMP(0);
  asm volatile("griddepcontrol.wait;" ::: "memory");
  DSV41_STAMP(1);
  const unsigned long long t = n3_now();
  while (n3_now() < t + spin_ns) {}
  DSV41_STAMP(2);
}

namespace {
constexpr int kExperts = 384, kHidden = 5120, kSplit = 3, kThreads = 320, kWarps = 10, kChunk = 16;
__device__ __forceinline__ float FmaBf16(unsigned short a, unsigned short b, float c) {
  float d; asm("fma.rn.f32.bf16 %0, %1, %2, %3;" : "=f"(d) : "h"(a), "h"(b), "f"(c)); return d; }
__device__ __forceinline__ float AddRn(float a, float b) {
  float d; asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
struct Chunk { unsigned int word[8]; };
__device__ __forceinline__ Chunk LoadChunk(const unsigned short* p) {
  const uint4 lo = reinterpret_cast<const uint4*>(p)[0];
  const uint4 hi = reinterpret_cast<const uint4*>(p)[1];
  return {{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w}};
}
__device__ __forceinline__ float Dot(const Chunk& x, const Chunk& w) {
  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc = FmaBf16((unsigned short)(x.word[i] & 0xFFFFu), (unsigned short)(w.word[i] & 0xFFFFu), acc);
    acc = FmaBf16((unsigned short)(x.word[i] >> 16), (unsigned short)(w.word[i] >> 16), acc);
  }
  return acc;
}
__device__ __forceinline__ float WarpSum(float v) {
#pragma unroll
  for (int mask = 16; mask >= 1; mask >>= 1) v = AddRn(v, __shfl_xor_sync(0xFFFFFFFFu, v, mask, 32));
  return v;
}
}  // namespace

extern "C" __global__ __launch_bounds__(320) void n3_router(
    const unsigned short* __restrict__ x, const unsigned short* __restrict__ w, float* __restrict__ logits,
    int rows, int trigger_at_entry, unsigned long long* stamps) {
  if (trigger_at_entry) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  N3_BEGIN(stamps);
  DSV41_STAMP(0);
  const int tx = threadIdx.x, bx = blockIdx.x, warp = tx / 32;
  Chunk wv[kSplit];
#pragma unroll
  for (int n = 0; n < kSplit; ++n) wv[n] = LoadChunk(w + (long long)(bx * kSplit + n) * kHidden + tx * kChunk);
  asm volatile("griddepcontrol.wait;" ::: "memory");
  DSV41_STAMP(1);
  __shared__ float s_acc[kWarps][8 * kSplit];
  for (int m = 0; m < rows; ++m) {
    const Chunk xv = LoadChunk(x + m * kHidden + tx * kChunk);
#pragma unroll
    for (int n = 0; n < kSplit; ++n) {
      const float sum = WarpSum(Dot(xv, wv[n]));
      if ((tx & 31) == 0) s_acc[warp][m * kSplit + n] = sum;
    }
  }
  DSV41_STAMP(2);
  if (!trigger_at_entry) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  __syncthreads();
  if (tx < rows * kSplit) {
    float acc = s_acc[0][tx];
#pragma unroll
    for (int i = 1; i < kWarps; ++i) acc = AddRn(acc, s_acc[i][tx]);
    logits[(tx / kSplit) * kExperts + bx * kSplit + tx % kSplit] = acc;
  }
  DSV41_STAMP(3);
}

// Router replica 2: every row's x chunk loaded before the dot products (as SGLang's
// compile-time-M kernel does), kSplitN router columns per CTA (numerics per column are
// independent of the split).
template <int kSplitN>
__device__ __forceinline__ void Router2(const unsigned short* __restrict__ x, const unsigned short* __restrict__ w,
                                        float* __restrict__ logits, int rows, int trigger_at_entry) {
  if (trigger_at_entry) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  DSV41_STAMP(0);
  const int tx = threadIdx.x, bx = blockIdx.x, warp = tx / 32;
  Chunk wv[kSplitN];
#pragma unroll
  for (int n = 0; n < kSplitN; ++n) wv[n] = LoadChunk(w + (long long)(bx * kSplitN + n) * kHidden + tx * kChunk);
  asm volatile("griddepcontrol.wait;" ::: "memory");
  DSV41_STAMP(1);
  __shared__ float s_acc[kWarps][8 * kSplitN];
  Chunk xv[8];
#pragma unroll
  for (int m = 0; m < 8; ++m) if (m < rows) xv[m] = LoadChunk(x + m * kHidden + tx * kChunk);
#pragma unroll
  for (int m = 0; m < 8; ++m) {
    if (m < rows) {
#pragma unroll
      for (int n = 0; n < kSplitN; ++n) {
        const float sum = WarpSum(Dot(xv[m], wv[n]));
        if ((tx & 31) == 0) s_acc[warp][m * kSplitN + n] = sum;
      }
    }
  }
  DSV41_STAMP(2);
  if (!trigger_at_entry) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  __syncthreads();
  if (tx < rows * kSplitN) {
    float acc = s_acc[0][tx];
#pragma unroll
    for (int i = 1; i < kWarps; ++i) acc = AddRn(acc, s_acc[i][tx]);
    logits[(tx / kSplitN) * kExperts + bx * kSplitN + tx % kSplitN] = acc;
  }
  DSV41_STAMP(3);
}

// Router replica 3: Router2's arithmetic, its weight words staged before the grid-dependency
// wait. kMode 1: each thread cp.async's its own 32-byte words into shared memory (an async
// copy keeps its place above the wait; ptxas sinks plain register loads below it) and
// reads them back after the wait; kMode 2: an L2 prefetch of the CTA's rows before the
// wait, the register loads after it.
template <int kSplitN, int kMode>
__device__ __forceinline__ void Router3(const unsigned short* __restrict__ x, const unsigned short* __restrict__ w,
                                        float* __restrict__ logits, int rows, int trigger_at_entry) {
  if (trigger_at_entry) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  DSV41_STAMP(0);
  const int tx = threadIdx.x, bx = blockIdx.x, warp = tx / 32;
  __shared__ __align__(16) uint4 stage[kSplitN][2][kThreads];
  if constexpr (kMode == 1) {
#pragma unroll
    for (int n = 0; n < kSplitN; ++n) {
      const unsigned short* src = w + (long long)(bx * kSplitN + n) * kHidden + tx * kChunk;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const unsigned int dst = static_cast<unsigned int>(__cvta_generic_to_shared(&stage[n][h][tx]));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src + 8 * h) : "memory");
      }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  } else {
    const char* rows_base = reinterpret_cast<const char*>(w + (long long)(bx * kSplitN) * kHidden);
    for (int line = tx; line < kSplitN * kHidden * 2 / 128; line += kThreads)
      asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(rows_base + line * 128));
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  DSV41_STAMP(1);
  Chunk xv[8];
#pragma unroll
  for (int m = 0; m < 8; ++m) if (m < rows) xv[m] = LoadChunk(x + m * kHidden + tx * kChunk);
  Chunk wv[kSplitN];
  if constexpr (kMode == 1) {
    asm volatile("cp.async.wait_all;" ::: "memory");
#pragma unroll
    for (int n = 0; n < kSplitN; ++n) {
      const uint4 lo = stage[n][0][tx], hi = stage[n][1][tx];
      wv[n] = {{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w}};
    }
  } else {
#pragma unroll
    for (int n = 0; n < kSplitN; ++n) wv[n] = LoadChunk(w + (long long)(bx * kSplitN + n) * kHidden + tx * kChunk);
  }
  __shared__ float s_acc[kWarps][8 * kSplitN];
#pragma unroll
  for (int m = 0; m < 8; ++m) {
    if (m < rows) {
#pragma unroll
      for (int n = 0; n < kSplitN; ++n) {
        const float sum = WarpSum(Dot(xv[m], wv[n]));
        if ((tx & 31) == 0) s_acc[warp][m * kSplitN + n] = sum;
      }
    }
  }
  DSV41_STAMP(2);
  if (!trigger_at_entry) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  __syncthreads();
  if (tx < rows * kSplitN) {
    float acc = s_acc[0][tx];
#pragma unroll
    for (int i = 1; i < kWarps; ++i) acc = AddRn(acc, s_acc[i][tx]);
    logits[(tx / kSplitN) * kExperts + bx * kSplitN + tx % kSplitN] = acc;
  }
  DSV41_STAMP(3);
}

#define N3_ROUTER3(name, split, mode)                                                                       \
  extern "C" __global__ __launch_bounds__(320) void name(                                                    \
      const unsigned short* __restrict__ x, const unsigned short* __restrict__ w, float* __restrict__ logits, \
      int rows, int trigger_at_entry, unsigned long long* stamps) {                                          \
    N3_BEGIN(stamps);                                                                                        \
    Router3<split, mode>(x, w, logits, rows, trigger_at_entry);                                              \
  }
N3_ROUTER3(n3_router3_s3_cp, 3, 1)
N3_ROUTER3(n3_router3_s1_cp, 1, 1)
N3_ROUTER3(n3_router3_s3_pf, 3, 2)
N3_ROUTER3(n3_router3_s1_pf, 1, 2)

extern "C" __global__ __launch_bounds__(320) void n3_router2_s3(
    const unsigned short* __restrict__ x, const unsigned short* __restrict__ w, float* __restrict__ logits,
    int rows, int trigger_at_entry, unsigned long long* stamps) {
  N3_BEGIN(stamps);
  Router2<3>(x, w, logits, rows, trigger_at_entry);
}

extern "C" __global__ __launch_bounds__(320) void n3_router2_s1(
    const unsigned short* __restrict__ x, const unsigned short* __restrict__ w, float* __restrict__ logits,
    int rows, int trigger_at_entry, unsigned long long* stamps) {
  N3_BEGIN(stamps);
  Router2<1>(x, w, logits, rows, trigger_at_entry);
}
"""
)


def _load(device_index):
    torch.zeros(1, device=torch.device("cuda", device_index))  # a current context
    major, minor = torch.cuda.get_device_capability(device_index)
    arch = f"sm_{major}{minor}a"
    headers = {h: (reg.SOURCE_DIR / h).read_text() for h in reg.GATE_ROUTE_HEADERS}
    gate = reg.compile_source(
        _GATE_SOURCE, "n3_gate", arch, reg.GATE_TOPK_NVRTC_OPTIONS, headers=headers
    )
    gate_late = reg.compile_source(
        "#define GATE_ROUTE_BIAS_AFTER_WAIT 1\n" + _GATE_SOURCE,
        "n3_gate_late",
        arch,
        reg.GATE_TOPK_NVRTC_OPTIONS,
        headers=headers,
    )
    misc = reg.compile_source(
        _MISC_SOURCE, "n3_misc", arch, ("-std=c++17", "-default-device")
    )
    return {
        "gate_topk+st": jit.load(gate, "n3_gate_topk", (384, 1, 1), 0),
        "gate_route+st": jit.load(gate, "n3_gate_route", (384, 1, 1), 0),
        "route+st": jit.load(gate, "n3_route", (64, 1, 1), 0),
        "hop_a": jit.load(misc, "n3_hop_a", (256, 1, 1), 0),
        "consumer": jit.load(misc, "n3_consumer", (128, 1, 1), 0),
        "router": jit.load(misc, "n3_router", (320, 1, 1), 0),
        "router2_s3": jit.load(misc, "n3_router2_s3", (320, 1, 1), 0),
        "router2_s1": jit.load(misc, "n3_router2_s1", (320, 1, 1), 0),
        **{
            f"router3_s{sp}_{mode}": jit.load(misc, f"n3_router3_s{sp}_{mode}", (320, 1, 1), 0)
            for sp in (3, 1)
            for mode in ("cp", "pf")
        },
        "gate_route_late+st": jit.load(gate_late, "n3_gate_route", (384, 1, 1), 0),
        "gate_topk": reg.load_gate_topk(device_index),
        "gate_route": reg.load_gate_route(device_index),
        "route": reg.load_kernels(device_index)["route"],
    }


def _p(t):
    return reg._pointer(t)


def _np(t):
    return reg._null_or_pointer(t)


class Layer:
    """One MoE layer's router (weight built for a fixed routing), bias and expert half."""

    def __init__(self, x, routing, gen, device, experts_layer):
        self.routing = routing  # [ROWS, TOP_K] global ids
        w = torch.randn(ROUTER_EXPERTS, HIDDEN, generator=gen) * 0.004
        xf = x.float().cpu()
        for r in range(ROWS):
            for j in routing[r].tolist():
                w[j] += 6.0 * xf[r] / xf[r].dot(xf[r])
        self.w = w.to(torch.bfloat16).to(device)
        self.bias = (torch.randn(ROUTER_EXPERTS, generator=gen) * 0.01).to(device)
        self.bias_alt = (torch.randn(ROUTER_EXPERTS, generator=gen) * 0.01).to(device)
        self.experts = experts_layer  # (tensors, maps1, maps2) from b5.random_layer


class Chain:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.k = _load(self.device.index)
        gen = torch.Generator().manual_seed(11)
        self.local = args.local_experts
        self.offset = args.offset
        b5.EXPERTS = self.local  # b5.random_layer's expert count
        self.expert_layers = [b5.random_layer(gen, self.device) for _ in range(2)]
        self.gemm_kernels = b5.load_kernels(self.device.index, {})
        self.src = torch.randn(ROWS, HIDDEN, generator=gen).to(torch.bfloat16).to(self.device)
        self.x = torch.empty_like(self.src)
        self.ctas = reg.max_ctas(ROWS, TOP_K, self.local)
        self.words = reg.cta_words(self.ctas)
        self.barrier = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.flag = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.num_valid = torch.tensor([ROWS], dtype=torch.int32, device=self.device)
        # DSV4.1 routes with the vision bias on every step: token ids (none an image token)
        self.input_ids = torch.arange(100, 100 + ROWS, dtype=torch.int64, device=self.device)
        self.quant_stream = torch.cuda.Stream(self.device)
        self.alt_stream = torch.cuda.Stream(self.device)
        self.side = None
        if 1 in args.side:
            b5.ROWS = ROWS
            self.side = _SideChain(self.device, gen, args.shared_n, args.shared_k)
        self.layers = []
        self.local_distinct = []
        for i in range(LAYERS):
            routing = self._routing(gen)
            ids = routing.flatten().tolist()
            self.local_distinct.append(
                len({j for j in ids if self.offset <= j < self.offset + self.local})
            )
            self.layers.append(
                Layer(self.src, routing, gen, self.device, self.expert_layers[i % 2])
            )

    def _routing(self, gen):
        """ROWS x TOP_K global ids from a pool of args.pool experts, half in the window
        (all of them without EP)."""
        pool_n = self.args.pool
        window = torch.arange(self.offset, self.offset + self.local)
        if self.local == ROUTER_EXPERTS:
            pool = window[torch.randperm(self.local, generator=gen)[:pool_n]]
        else:
            outside = torch.tensor(
                [j for j in range(ROUTER_EXPERTS) if not self.offset <= j < self.offset + self.local]
            )
            half = pool_n // 2
            pool = torch.cat(
                [
                    window[torch.randperm(self.local, generator=gen)[:half]],
                    outside[torch.randperm(outside.numel(), generator=gen)[: pool_n - half]],
                ]
            )
        rows = [pool[torch.randperm(pool_n, generator=gen)[:TOP_K]] for _ in range(ROWS)]
        return torch.stack(rows)

    def buffers(self):
        d, padded = self.device, reg.TILE * self.ctas
        return dict(
            logits=torch.empty(ROWS, ROUTER_EXPERTS, dtype=torch.float32, device=d),
            weights=torch.empty(ROWS, TOP_K, dtype=torch.float32, device=d),
            ids=torch.empty(ROWS, TOP_K, dtype=torch.int32, device=d),
            permuted=torch.empty(ROWS, TOP_K, dtype=torch.int32, device=d),
            record=torch.empty(reg.route_words(self.ctas), dtype=torch.int32, device=d),
            h=torch.empty((padded, reg.PADDED), dtype=torch.uint8, device=d),
            hs=torch.empty((padded + 127) // 128 * 128 * reg.PADDED // 32, dtype=torch.uint8, device=d),
            y=torch.empty((ROWS * TOP_K * reg.TILE, HIDDEN), dtype=torch.bfloat16, device=d),
        )

    def stamp_buffers(self, gemm):
        d = self.device
        g1, g2 = b5.grids(gemm, self.ctas, self.sms)
        n1 = g1[1][0] * g1[1][1] * g1[1][2]
        n2 = g2[1][0] * g2[1][1] * g2[1][2]
        z = lambda n: torch.zeros(n * STAMPS, dtype=torch.int64, device=d)  # noqa: E731
        return dict(
            hop_a=z(HOP_A_CTAS),
            router=z(ROUTER_EXPERTS),
            gate=z(ROWS),
            route=z(1),
            consumer=z(CONSUMER_CTAS),
            g1=torch.zeros(n1 * b5.STAMP_WORDS, dtype=torch.int64, device=d),
            g2=torch.zeros(n2 * b5.STAMP_WORDS, dtype=torch.int64, device=d),
        )

    # ------------------------------------------------------------------ one half
    def half(self, layer, buf, variant, st):
        a = self.args
        stream = torch.cuda.current_stream()
        s = stream.cuda_stream
        main = stream
        reg._launch(
            self.k["hop_a"],
            (HOP_A_CTAS, 1, 1),
            [
                _p(self.src),
                _p(self.x),
                ctypes.c_int32(ROWS * HIDDEN),
                ctypes.c_uint64(int(a.hop_a_us * 1000)),
                ctypes.c_int32(1),
                _np(st["hop_a"] if st else None),
            ],
            s,
        )
        # SGLang's fork: the shared expert and the routed quant from hop A's output
        self.alt_stream.wait_stream(main)
        self.quant_stream.wait_stream(main)
        router = a.router if not (st and a.router == "sglang") else "replica"
        if router == "sglang":
            from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16

            tiny_gemm_bf16(self.x, layer.w, out=buf["logits"], max_m=a.router_max_m)
        else:
            # replica | early | r2s3 | r2s1 | r2s3e | r2s1e | r3s{3,1}{cp,pf}[e]
            name, ctas = "router", 128
            if router.startswith("r2"):
                split = int(router[3])
                name, ctas = f"router2_s{split}", ROUTER_EXPERTS // split
            if router.startswith("r3"):
                split = int(router[3])
                name, ctas = f"router3_s{split}_{router[4:6]}", ROUTER_EXPERTS // split
            reg._launch(
                self.k[name],
                (ctas, 1, 1),
                [
                    _p(self.x),
                    _p(layer.w),
                    _p(buf["logits"]),
                    ctypes.c_int32(ROWS),
                    ctypes.c_int32(1 if router == "early" or (router[:2] in ("r2", "r3") and router.endswith("e")) else 0),
                    _np(st["router"] if st else None),
                ],
                s,
            )
        from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

        with torch.cuda.stream(self.quant_stream):
            x_q, x_s = flashinfer_mxfp8_quantize(self.x, False, alignment=HIDDEN)
            ready = self.quant_stream.record_event()
        x_s = x_s.view(torch.uint8).reshape(ROWS, -1)
        # fused_nojoin (timing only, GEMM1 races the quant): the upper bound of a top-k
        # kernel that quantizes the routed input itself and so launches early behind the router
        nojoin = variant.endswith("_nojoin")
        variant = variant.removesuffix("_nojoin")
        if not nojoin:
            main.wait_event(ready)
        vision = a.vision
        gate_args = [
            _p(buf["logits"]),
            _p(layer.bias),
            _np(layer.bias_alt if vision else None),
            _np(self.input_ids if vision else None),
            ctypes.c_int32(1),
            ctypes.c_int64(129264 if vision else -1),
            _p(self.num_valid),
            _p(buf["weights"]),
            _p(buf["ids"]),
            ctypes.c_float(1.5),
            ctypes.c_float(1e-20 if vision else 0.0),
        ]
        tag = "+st" if st else ""
        extra = [_p(st["gate"])] if st else []
        sizes = [
            ctypes.c_int32(ROWS),
            ctypes.c_int32(TOP_K),
            ctypes.c_int32(self.local),
            ctypes.c_int32(self.offset),
        ]
        if variant.startswith("fused"):
            name = "gate_route" + tag
            if variant == "fused_late":
                name = "gate_route_late+st"
                extra = extra or [_np(None)]
            reg._launch(
                self.k[name],
                (ROWS, 1, 1),
                gate_args
                + [
                    _p(buf["permuted"]),
                    _p(buf["record"]),
                    ctypes.c_int32(self.local),
                    ctypes.c_int32(self.offset),
                ]
                + extra,
                s,
                cluster=(ROWS, 1, 1),
            )
        else:
            reg._launch(self.k["gate_topk" + tag], (ROWS, 1, 1), gate_args + extra, s)
            reg._launch(
                self.k["route" + tag],
                (1, 1, 1),
                [_p(buf["ids"]), _p(buf["permuted"]), _p(buf["record"]), _p(self.barrier)]
                + sizes
                + ([_p(st["route"])] if st else []),
                s,
            )
        self._gemms(layer, buf, x_q, x_s, a.gemm if not st else "strided", st)
        if nojoin:
            main.wait_event(ready)
        if self.side is not None:
            with torch.cuda.stream(self.alt_stream):
                self.side()
            main.wait_stream(self.alt_stream)
        reg._launch(
            self.k["consumer"],
            (CONSUMER_CTAS, 1, 1),
            [_p(buf["y"]), ctypes.c_uint64(int(a.consumer_us * 1000)), _np(st["consumer"] if st else None)],
            s,
        )

    def _gemms(self, layer, buf, x_q, x_s, gemm, st):
        _, maps1, maps2 = layer.experts
        (n1, g1), (n2, g2) = b5.grids(gemm, self.ctas, self.sms)
        tag = "+st" if st else ""
        s = torch.cuda.current_stream().cuda_stream
        words = ctypes.c_int32(self.words)
        reg._launch(
            self.gemm_kernels[n1 + tag],
            g1,
            [*maps1, _p(x_q), _p(x_s), _p(buf["record"]), _p(layer.experts[0][4]), _p(buf["h"]), _p(buf["hs"]), words]
            + ([_p(st["g1"])] if st else []),
            s,
        )
        reg._launch(
            self.gemm_kernels[n2 + tag],
            g2,
            [*maps2, _p(buf["h"]), _p(buf["hs"]), _p(buf["record"]), _p(buf["y"]), words]
            + ([_p(st["g2"])] if st else []),
            s,
        )

    def graph(self, variant, stamped=False, spacers_only=False):
        bufs = [self.buffers() for _ in range(LAYERS)]
        sts = [self.stamp_buffers("strided") for _ in range(LAYERS)] if stamped else None

        def body():
            for i in range(LAYERS):
                self.flag.add_(1)  # spacer: no PDL edge into the next half
                if spacers_only:
                    continue
                self.half(self.layers[i], bufs[i], variant, sts[i] if sts else None)

        body()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
        torch.cuda.synchronize()
        return g, bufs, sts

    def time(self, g, replays):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        g.replay()
        torch.cuda.synchronize()
        start.record()
        for _ in range(replays):
            g.replay()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1e3 / replays / LAYERS


class _SideChain(b5.SideChain):
    """b5's shared-expert stand-in at a given gate_up N and down K (TP8: 576 / 288 padded to 384)."""

    def __init__(self, device, g, n, k):
        from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll

        self.mxfp8_ll = mxfp8_ll
        self.n, self.k = n, k

        def weight(rows, cols):
            w = (torch.randn(rows, cols, generator=g) * 0.05).to(device).to(torch.float8_e4m3fn)
            s = torch.randint(120, 128, (rows * cols // 32,), dtype=torch.uint8, generator=g).to(device)
            return w, s

        self.gate_up = weight(n, HIDDEN)
        self.down = weight(HIDDEN, k)
        self.x = torch.randn(ROWS, HIDDEN, generator=g).to(torch.bfloat16).to(device)
        mxfp8_ll.precompile(self.gate_up[0], [ROWS])
        mxfp8_ll.precompile(self.down[0], [ROWS])

    def __call__(self):
        from sgl_nvidia_mega.deepseek_v41 import sglang_ops
        from sgl_nvidia_mega.deepseek_v41.sm103.attention.q_norm_quant import mxfp8_row_tiles

        x_q, x_s = mxfp8_row_tiles(self.x)
        gu = self.mxfp8_ll.linear(x_q, x_s, *self.gate_up)
        mid = torch.empty(ROWS, self.n // 2, dtype=torch.bfloat16, device=gu.device)
        sglang_ops.silu_and_mul_clamp(gu, mid, 10.0)
        h = torch.nn.functional.pad(mid, (0, self.k - self.n // 2))
        h_q, h_s = self.mxfp8_ll.quantize(h)
        return self.mxfp8_ll.linear(h_q, h_s, *self.down)


# ---------------------------------------------------------------------------- analysis


def _rows(t, words):
    return t.view(-1, words).cpu().tolist()


def timeline(st, variant):
    """us relative to hop A's end (its last CTA), one layer."""
    hop = _rows(st["hop_a"], STAMPS)
    t0 = max(r[2] for r in hop)
    us = lambda t: (t - t0) / 1e3  # noqa: E731
    f = dict(hop_a_wait_last=us(max(r[1] for r in hop)))
    rt = [r for r in _rows(st["router"], STAMPS) if r[0]]
    f.update(
        router_entry_first=us(min(r[0] for r in rt)),
        router_entry_last=us(max(r[0] for r in rt)),
        router_wait_first=us(min(r[1] for r in rt)),
        router_wait_last=us(max(r[1] for r in rt)),
        router_dot_last=us(max(r[2] for r in rt)),
        router_end=us(max(r[3] for r in rt)),
    )
    ga = _rows(st["gate"], STAMPS)
    f.update(
        gate_entry_first=us(min(r[0] for r in ga)),
        gate_entry_last=us(max(r[0] for r in ga)),
        gate_wait_last=us(max(r[1] for r in ga)),
        gate_select_last=us(max(r[2] for r in ga)),
    )
    if variant.startswith("fused"):
        f.update(
            gate_cta0_gathered=us(ga[0][3]),
            gate_end=us(max(r[4] for r in ga)),
        )
    else:
        f.update(gate_end=us(max(r[3] for r in ga)))
        ro = _rows(st["route"], STAMPS)[0]
        f.update(route_entry=us(ro[0]), route_wait=us(ro[1]), route_end=us(ro[2]))
    g1 = _rows(st["g1"], b5.STAMP_WORDS)
    g2 = _rows(st["g2"], b5.STAMP_WORDS)
    f.update(
        g1_entry_first=us(min(r[0] for r in g1)),
        g1_entry_p50=statistics.median(us(r[0]) for r in g1),
        g1_wait_first=us(min(r[2] for r in g1 if r[2])),
        g1_first_tma=us(min(r[3] for r in g1 if r[3])),
        g1_end=us(max(r[31] for r in g1)),
        g2_wait_first=us(min(r[2] for r in g2 if r[2])),
        g2_end=us(max(r[31] for r in g2)),
    )
    co = _rows(st["consumer"], STAMPS)
    f.update(consumer_entry_first=us(min(r[0] for r in co)), consumer_wait=us(max(r[1] for r in co)))
    return f


def check_router(chain):
    """Each router replica's logits against SGLang's tiny_n_gemm, bit for bit, at 1-8 rows."""
    from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16

    gen = torch.Generator(device=chain.device).manual_seed(5)
    w = (torch.randn(ROUTER_EXPERTS, HIDDEN, device=chain.device, generator=gen) * 0.02).bfloat16()
    for rows in range(1, 9):
        x = torch.randn(rows, HIDDEN, device=chain.device, generator=gen).bfloat16()
        want = tiny_gemm_bf16(x, w, out_dtype=torch.float32, max_m=16)
        for name, ctas in (
            ("router", 128),
            ("router2_s3", 128),
            ("router2_s1", 384),
            ("router3_s3_cp", 128),
            ("router3_s1_cp", 384),
            ("router3_s3_pf", 128),
            ("router3_s1_pf", 384),
        ):
            for early in (0, 1):
                got = torch.empty(rows, ROUTER_EXPERTS, dtype=torch.float32, device=chain.device)
                reg._launch(
                    chain.k[name],
                    (ctas, 1, 1),
                    [_p(x), _p(w), _p(got), ctypes.c_int32(rows), ctypes.c_int32(early), _np(None)],
                    torch.cuda.current_stream().cuda_stream,
                )
                torch.cuda.synchronize()
                same = torch.equal(got.view(torch.int32), want.view(torch.int32))
                if not same:
                    print(f"check router {name} early={early} rows={rows}: MISMATCH", flush=True)
                    return
    print("check router replicas vs tiny_n_gemm at 1-8 rows: bitwise", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local-experts", type=int, default=192)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--pool", type=int, default=24, help="global experts the rows choose from")
    p.add_argument("--variants", nargs="+", default=["base", "fused"])
    p.add_argument("--gemm", default="clc", choices=["clc", "strided"])
    p.add_argument(
        "--router",
        default="sglang",
        choices=["sglang", "replica", "early", "r2s3", "r2s1", "r2s3e", "r2s1e"]
        + [f"r3s{sp}{m}{e}" for sp in (3, 1) for m in ("cp", "pf") for e in ("", "e")],
    )
    p.add_argument("--router-max-m", type=int, default=16)
    p.add_argument("--side", type=int, nargs="+", default=[0, 1])
    p.add_argument("--shared-n", type=int, default=576)
    p.add_argument("--shared-k", type=int, default=384)
    p.add_argument("--hop-a-us", type=float, default=3.0)
    p.add_argument("--consumer-us", type=float, default=1.5)
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--replays", type=int, default=20)
    p.add_argument("--stamps", default="")
    p.add_argument("--check", action="store_true")
    p.add_argument("--vision", type=int, default=1, help="DSV4.1's vision routing inputs")
    args = p.parse_args()
    results = []
    for side in args.side:
        args_side = argparse.Namespace(**vars(args))
        args_side.side = [side]
        chain = Chain(args_side)
        print(
            f"sms {chain.sms} local experts {chain.local} offset {chain.offset} side {side} "
            f"local distinct p50 {statistics.median(chain.local_distinct)} "
            f"[{min(chain.local_distinct)}..{max(chain.local_distinct)}] router {args.router} gemm {args.gemm}",
            flush=True,
        )
        if args.check:
            check_router(chain)
            outs = {}
            for v in args.variants:
                g, bufs, _ = chain.graph(v)
                g.replay()
                torch.cuda.synchronize()
                outs[v] = bufs
                del g
            ref = outs[args.variants[0]]
            want = torch.stack([lay.routing for lay in chain.layers]).to(chain.device).sort(-1).values
            got_ids = torch.stack([b["ids"] for b in ref]).sort(-1).values
            print(f"check routing as built: {'yes' if torch.equal(want.int(), got_ids) else 'NO'}")
            for v in args.variants[1:]:
                same = all(
                    torch.equal(a[k].view(torch.int32) if a[k].dtype == torch.float32 else a[k], b[k].view(torch.int32) if b[k].dtype == torch.float32 else b[k])
                    for a, b in zip(ref, outs[v])
                    for k in ("weights", "ids", "permuted", "record")
                ) and all(
                    torch.equal(
                        a["y"][a["permuted"].clamp(min=0).flatten().long()].view(torch.int16),
                        b["y"][b["permuted"].clamp(min=0).flatten().long()].view(torch.int16),
                    )
                    for a, b in zip(ref, outs[v])
                )
                print(f"check {v} vs {args.variants[0]}: {'bitwise' if same else 'MISMATCH'}", flush=True)
            del outs
        spacer, _, _ = chain.graph("base", spacers_only=True)
        spacer_us = statistics.median(chain.time(spacer, args.replays) for _ in range(3))
        graphs = {v: chain.graph(v)[0] for v in args.variants}
        times = {v: [] for v in args.variants}
        for r in range(args.repeats):
            for v in args.variants if r % 2 == 0 else args.variants[::-1]:
                times[v].append(chain.time(graphs[v], args.replays) - spacer_us)
        for v in args.variants:
            t = times[v]
            row = dict(side=side, variant=v, us=statistics.median(t), lo=min(t), hi=max(t), all=t,
                       local=chain.local, offset=chain.offset, router=args.router, gemm=args.gemm)
            results.append(row)
            print(f"side={side} {v:8s} {row['us']:7.2f} us/half [{row['lo']:.2f} .. {row['hi']:.2f}]", flush=True)
        del graphs
        if args.stamps:
            for v in args.variants:
                g, bufs, sts = chain.graph(v, stamped=True)
                facts = []
                for _ in range(3):
                    for st in sts:
                        for t in st.values():
                            t.zero_()
                    g.replay()
                    torch.cuda.synchronize()
                    facts += [timeline(st, v.removesuffix("_nojoin")) for st in sts]
                med = {k: statistics.median(f[k] for f in facts) for k in facts[0]}
                results.append(dict(stamps=med, side=side, variant=v, local=chain.local))
                print(f"  stamps side={side} {v}: " + " ".join(f"{k}={x:.2f}" for k, x in med.items()), flush=True)
                del g
        del chain
        torch.cuda.empty_cache()
    if args.stamps:
        with open(args.stamps, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
