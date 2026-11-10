// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// The LM-head shard at 1-8 rows as a weight stream: out[rows, N] = x[rows, K] @ W[N, K]^T,
// W read once from HBM as MXFP8 (E4M3 codes [N, K] + UE8M0 scales of 32 columns in
// FlashInfer's 128x4 swizzle), x BF16 (lm_head_mxfp8.py).
//
// A warp owns a 16-row tile of W (mma.m16n8k16, W as A, the rows of x as the 8 columns
// of B) and LH_SPLIT warps of a CTA split its K; the CTA holds x in shared memory, its
// 4-column groups stored in the order (0, 2, 1, 3) so that a lane's B fragment is one
// 8-byte read. Per 128 columns of rows g and g + 8 a lane streams codes [16t, 16t + 16)
// and [64 + 16t, 80 + 16t) (each load instruction reads whole 64-byte row segments) and
// the four scales as one word, LH_DEPTH chunks ahead; the first chunks load before the
// PDL wait.
// A code becomes BF16 bits exactly (sign, then exponent and mantissa shifted by 4: the
// value times 2^-120, subnormals included) and one BF16 multiply by 2^(scale - 7) gives
// the dequantized weight, which BF16 holds exactly. LH_FP8 0 reads a BF16 W instead
// through the same fragments and MMAs: the reference that MXFP8 output equals bit for
// bit on the dequantized W. Per tile: FP32 MMA accumulation in K order, the split parts
// added in part order, one BF16 rounding.
// Grid: CTAs <= SMs; tile t runs on CTA t % grid, slot t / grid. Block LH_WARPS * 32, PDL.

#include <cuda_bf16.h>

#ifndef LH_K
#define LH_K 5120
#endif
#ifndef LH_FP8
#define LH_FP8 1
#endif
#ifndef LH_SPLIT
#define LH_SPLIT 1
#endif
#ifndef LH_DEPTH
#define LH_DEPTH 4
#endif
#ifndef LH_WARPS
#define LH_WARPS 16
#endif

namespace {

constexpr int kK = LH_K;
constexpr bool kFp8 = LH_FP8 != 0;
constexpr int kSplit = LH_SPLIT;
constexpr int kDepth = LH_DEPTH;
constexpr int kWarps = LH_WARPS;
constexpr int kSlots = kWarps / kSplit;
constexpr int kChunk = 128;  // columns per lane step: 4 lanes x 32
constexpr int kChunks = kK / kChunk / kSplit;  // per warp per tile
constexpr int kMaxRows = 8;
constexpr int kXStride = kK + 8;  // BF16 elements a shared row; +16 B spreads the banks
constexpr int kScaleCols = kK / 32;
// a stage: 32 codes (FP8) or 32 BF16 values of rows g and g + 8, as u32 words, in two
// segments of 16 columns 64 apart
constexpr int kWords = kFp8 ? 8 : 16;
constexpr int kSegmentBytes = kFp8 ? 16 : 32;

static_assert(kK % (kChunk * kSplit) == 0, "K splits into whole 128-column chunks");
static_assert(kChunks % kDepth == 0, "the prefetch depth divides the chunks");
static_assert(kWarps % kSplit == 0, "whole tiles a CTA");
static_assert(kSplit == 1 || kSlots < 16, "a named barrier per slot");

struct Stage {
  unsigned w[2][kWords];
  unsigned scales[2];  // the chunk's four UE8M0 bytes of each row
};

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

__device__ __forceinline__ uint4 load_stream(const void* p) {
  uint4 v;
  // the 256-byte L2 prefetch brings the lane's next segments with this one
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ unsigned load_word(const unsigned char* p) {
  unsigned v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ unsigned prmt(unsigned a, unsigned b, unsigned sel) {
  unsigned d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

// FlashInfer's 128x4 swizzle of UE8M0 scales over rows padded to 128 (mxfp8_ll.quantize)
__device__ __forceinline__ long long scale_offset(int row, int block) {
  return ((long long)(row >> 7) * (kScaleCols >> 2) + (block >> 2)) * 512 + (row & 31) * 16 +
         ((row >> 5) & 3) * 4 + (block & 3);
}

// chunk columns [chunk, chunk + 128): this lane's segments start at chunk + 16t, + 64
__device__ __forceinline__ void load_stage(Stage& s, const unsigned char* w, const unsigned char* sf,
                                           int row0, int row1, int chunk, int t) {
  const int rows[2] = {row0, row1};
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const unsigned char* p = w + ((long long)rows[h] * kK + chunk) * (kFp8 ? 1 : 2) + kSegmentBytes * t;
#pragma unroll
    for (int v = 0; v < kWords / 4; ++v) {
      // segment v / (kWords / 8): 64 columns on
      const int seg = v / (kWords / 8), part = v % (kWords / 8);
      uint4 q = load_stream(p + seg * 64 * (kFp8 ? 1 : 2) + 16 * part);
      s.w[h][4 * v] = q.x;
      s.w[h][4 * v + 1] = q.y;
      s.w[h][4 * v + 2] = q.z;
      s.w[h][4 * v + 3] = q.w;
    }
    if (kFp8) s.scales[h] = load_word(sf + scale_offset(rows[h], chunk >> 5));
  }
}

// 2^(scale - 7) as BF16 in both halves for scale byte i of the word: undoes the code's
// 2^-120 and applies the scale
__device__ __forceinline__ unsigned scale_pair(unsigned scales, int i) {
  unsigned bits = (((scales >> (8 * i)) & 0xFFu) + 120u) << 7;
  return bits | (bits << 16);
}

__device__ __forceinline__ unsigned bf16_mul(unsigned a, unsigned b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<unsigned*>(&r);
}

// codes (c0, c1, c2, c3) of a word -> BF16 pairs (c0, c2) and (c1, c3), dequantized
__device__ __forceinline__ void dequant(unsigned w, unsigned scale, unsigned& even, unsigned& odd) {
  const unsigned sign = 0x80008000u;
  unsigned e = ((w << 8) & sign) | ((w & 0x007F007Fu) << 4);
  unsigned o = (w & sign) | ((w & 0x7F007F00u) >> 4);
  even = bf16_mul(e, scale);
  odd = bf16_mul(o, scale);
}

__device__ __forceinline__ void mma(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                    unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
      "{%8, %9}, {%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// xs: this lane's x (permuted) at chunk + 16t, for its column g of B
__device__ __forceinline__ void consume(float (&acc)[4], const Stage& s, const unsigned* xs, int t) {
  // the 32 x values of the two segments: words 0-7 at xs, 8-15 at xs + 32 (64 columns on)
  uint4 xv[4];
#pragma unroll
  for (int v = 0; v < 4; ++v) xv[v] = reinterpret_cast<const uint4*>(xs + (v >> 1) * 32)[v & 1];
  const unsigned* xw = reinterpret_cast<const unsigned*>(xv);
  unsigned sc[2][2] = {{0, 0}, {0, 0}};  // [row half][segment]
  if (kFp8) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      sc[h][0] = scale_pair(s.scales[h], t >> 1);
      sc[h][1] = scale_pair(s.scales[h], 2 + (t >> 1));
    }
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    unsigned a0, a1, a2, a3;  // (row g: c0 c2), (row g+8: c0 c2), (row g: c1 c3), (row g+8: c1 c3)
    if (kFp8) {
      dequant(s.w[0][j], sc[0][j >> 2], a0, a2);
      dequant(s.w[1][j], sc[1][j >> 2], a1, a3);
    } else {
      a0 = prmt(s.w[0][2 * j], s.w[0][2 * j + 1], 0x5410);
      a2 = prmt(s.w[0][2 * j], s.w[0][2 * j + 1], 0x7632);
      a1 = prmt(s.w[1][2 * j], s.w[1][2 * j + 1], 0x5410);
      a3 = prmt(s.w[1][2 * j], s.w[1][2 * j + 1], 0x7632);
    }
    mma(acc, a0, a1, a2, a3, xw[2 * j], xw[2 * j + 1]);
  }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kWarps * 32, 1)
    dsv41_lm_head_stream(const unsigned char* __restrict__ weight, const unsigned char* __restrict__ scales,
                         const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, int n, int rows,
                         int x_stride, int out_stride) {
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned* xs = reinterpret_cast<unsigned*>(smem);  // [kMaxRows][kXStride / 2] words
  float* partial = reinterpret_cast<float*>(smem + kMaxRows * kXStride * 2);

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int slot = warp / kSplit, part = warp % kSplit;
  const int tiles = (n + 15) >> 4;
  const int col_base = part * kChunks * kChunk;

  int tile = blockIdx.x + gridDim.x * slot;
  Stage stage[kDepth];
  auto rows_of = [&](int tl, int& r0, int& r1) {
    // rows past N read row N - 1 and are not stored
    r0 = min(tl * 16 + g, n - 1);
    r1 = min(tl * 16 + g + 8, n - 1);
  };
  if (tile < tiles) {
    int r0, r1;
    rows_of(tile, r0, r1);
#pragma unroll
    for (int d = 0; d < kDepth; ++d) load_stage(stage[d], weight, scales, r0, r1, col_base + d * kChunk, t);
  }

  pdl_wait();
  // x into shared memory, each 4-column group as (0, 2, 1, 3); rows past `rows` zero
  for (int i = threadIdx.x; i < kMaxRows * (kK / 4); i += kWarps * 32) {
    const int r = i / (kK / 4), q = i % (kK / 4);
    unsigned lo = 0, hi = 0;
    if (r < rows) {
      const uint2 v = *reinterpret_cast<const uint2*>(x + (long long)r * x_stride + 4 * q);
      lo = v.x;
      hi = v.y;
    }
    xs[r * (kXStride / 2) + 2 * q] = prmt(lo, hi, 0x5410);
    xs[r * (kXStride / 2) + 2 * q + 1] = prmt(lo, hi, 0x7632);
  }
  __syncthreads();

  const unsigned* xrow = xs + g * (kXStride / 2) + (col_base + 16 * t) / 2;
  for (; tile < tiles; tile += gridDim.x * kSlots) {
    int r0, r1;
    rows_of(tile, r0, r1);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int c0 = 0; c0 < kChunks; c0 += kDepth) {
#pragma unroll
      for (int d = 0; d < kDepth; ++d) {
        const int c = c0 + d;
        consume(acc, stage[d], xrow + c * (kChunk / 2), t);
        if (c + kDepth < kChunks) {
          load_stage(stage[d], weight, scales, r0, r1, col_base + (c + kDepth) * kChunk, t);
        }
      }
    }
    const int next = tile + gridDim.x * kSlots;
    if (next < tiles) {
      int n0, n1;
      rows_of(next, n0, n1);
#pragma unroll
      for (int d = 0; d < kDepth; ++d) load_stage(stage[d], weight, scales, n0, n1, col_base + d * kChunk, t);
    }
    if (kSplit > 1) {
      float* mine = partial + (slot * (kSplit - 1) * 32 + lane) * 4;
      if (part > 0) {
        *reinterpret_cast<float4*>(mine + (part - 1) * 128) = make_float4(acc[0], acc[1], acc[2], acc[3]);
      }
      asm volatile("bar.sync %0, %1;" ::"r"(slot + 1), "r"(kSplit * 32) : "memory");
      if (part == 0) {
#pragma unroll
        for (int p = 1; p < kSplit; ++p) {
          const float4 o = *reinterpret_cast<const float4*>(mine + (p - 1) * 128);
          acc[0] += o.x;
          acc[1] += o.y;
          acc[2] += o.z;
          acc[3] += o.w;
        }
      }
      asm volatile("bar.sync %0, %1;" ::"r"(slot + 1), "r"(kSplit * 32) : "memory");
    }
    if (part == 0) {
      const int row = tile * 16 + g;
      const int col = 2 * t;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = row + (i >> 1) * 8, c = col + (i & 1);
        if (r < n && c < rows) out[(long long)c * out_stride + r] = __float2bfloat16_rn(acc[i]);
      }
    }
  }
  pdl_trigger();
}
