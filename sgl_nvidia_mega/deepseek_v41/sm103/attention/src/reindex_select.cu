// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// The DeepSeek-V4.1 indexer's candidate consumer ("Reindex" layer) at decode, in one
// cluster launch: the bf16 logits of a row's published blocks and their top-k, written
// as pool slots into SGLang's selection buffer. It replaces
//   deep_gemm.fp8_fp4_paged_sparse_mqa_logits   (sparse_logits)
//   + topk_bf16_small_kernel                    (topk_transform_sparse)
// and reproduces both:
//
// - Logits: per 128-key tile, the two K-64 MXF4 block32 MMAs DeepGEMM issues (the keys
//   as UMMA M, the row's 32 heads as N; an accumulator element does not depend on the
//   tiling), then DeepGEMM's head sum: cvt.rn.relu.bf16x2, two bf16 fma chains over
//   heads 4i,4i+1 and 4i+2,4i+3, hadd2, hadd.
// - Top-k: topk_bf16_small's selection (the k-th value in its "distorted" bf16 order,
//   float compares in the census). Its choice among the elements equal to that value
//   and its output order depend on warp arrival; this kernel writes the outcome of
//   in-order arrival: ascending columns, and in the 512-thread slice holding the last
//   admitted equal element, that slice's highest equal columns (its eq_mask keeps the
//   high bits).
//
// Grid (cluster, rows), cluster (cluster, 1, 1), 256 threads. CTA r of row b scores
// columns [r * W, r * W + W) of the row (W = RS_TILES * 128), RS_TILES tiles of 16
// published blocks of 8 keys: warp w gathers tile w (TMA for the keys, cp.async for
// their scales), warp 4 issues the MMAs, warps 0-3 read the accumulators. The select
// runs two radix rounds over the cluster, each CTA sending its histogram into every
// CTA with st.async; each keeps every source CTA's counts, which give it the elements
// above the pivot and equal to it in every CTA before it. The last admitted equal
// element is located in the CTA holding it, which sends the ranks to take to every CTA.
//
// PDL: the keys and their scales are read after the grid-dependency wait. With
// `inputs_before_wait`, the published blocks, lengths, query and weights are read
// before it (the caller's contract: no kernel this one may overlap writes them).
//
// Variant macros: RS_CLUSTER (CTAs per row), RS_TILES (tiles per CTA) and
// RS_SMEM_BYTES (the launcher's dynamic shared memory).

#if !defined(RS_CLUSTER) || !defined(RS_TILES) || !defined(RS_SMEM_BYTES)
#error "reindex_select.cu: the launcher passes RS_CLUSTER, RS_TILES and RS_SMEM_BYTES"
#endif

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef signed int int32_t;
typedef unsigned long long uint64_t;
typedef signed long long int64_t;

#if defined(__CUDACC_RTC__)
typedef struct __align__(64) {
  uint64_t opaque[16];
} CUtensorMap;
#else
#include <cuda.h>
#endif

#include <cuda_bf16.h>

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kWarps = kThreads / 32;
constexpr uint32_t kCluster = RS_CLUSTER;
constexpr uint32_t kTiles = RS_TILES;
constexpr uint32_t kSlots = 4;  // TMEM accumulator ring
constexpr uint32_t kHeads = 32;
constexpr uint32_t kKeyBytes = 64;  // 128 e2m1 values
constexpr uint32_t kTileKeys = 128;
constexpr uint32_t kBlockKeys = 8;
constexpr uint32_t kTileBlocks = kTileKeys / kBlockKeys;
constexpr uint32_t kCols = kTiles * kTileKeys;  // columns per CTA
constexpr uint32_t kBlocks = kCols / kBlockKeys;
constexpr uint32_t kPerThread = kCols / kThreads;  // contiguous columns per thread
constexpr uint32_t kBins = 256;
// Round 0 also carries the +0 / -0 counts (words 256-257); round 1 the CTA's first and
// last 32 logits (words 260-275, 276-291), where a top-k slice may cross into it
constexpr uint32_t kEdgeWord = kBins + 4;
constexpr uint32_t kHistWords = kEdgeWord + 32;
constexpr uint32_t kRoundGroups[2] = {kEdgeWord / 4, kHistWords / 4};
// topk_bf16_small: 512 threads of up to 4 vectors of 8 scores
constexpr uint32_t kStockThreads = 512;
constexpr uint32_t kTmemCols = 256;
constexpr uint32_t kSfkCol = kSlots * kHeads;
constexpr uint32_t kSfqCol = kSfkCol + kTiles * 4;
static_assert(kTiles == kWarps, "one gathering warp per tile");
static_assert(kPerThread == 4, "8-byte logits loads");
static_assert(kSfqCol + 4 <= kTmemCols, "TMEM columns");
static_assert(kCluster >= 2 && kThreads == kCluster * 16, "16 lanes per source CTA");

// Instruction descriptor of kind::mxf4 block32: E2M1 x E2M1 -> F32, UE8M0 scales,
// M 128, N 32, both K-major; the scale ids go in bits 4-5 (B) and 29-30 (A)
constexpr uint32_t kInstrDesc =
    (1u << 7) | (1u << 10) | ((kHeads >> 3) << 17) | (1u << 23) | ((kTileKeys >> 4) << 24);

struct Smem {
  alignas(1024) uint8_t k[kTiles][kTileKeys * kKeyBytes];  // SW64 K-major
  alignas(512) uint8_t q[kHeads * kKeyBytes];               // SW64 K-major
  alignas(16) uint32_t sf_k[kTiles][kTileKeys];             // UTCCP 32x128b order
  alignas(16) uint32_t sf_q[128];
  alignas(16) uint16_t logits[kCols];                       // raw bf16
  int32_t blocks[kBlocks];
  alignas(16) uint32_t hist_local[2][kHistWords];
  alignas(16) uint32_t hist[2][kCluster][kHistWords];  // [round][source CTA], st.async
  uint32_t above[kCluster];  // per source CTA: elements above the pivot
  uint32_t equal[kCluster];  // and equal to it
  uint32_t scan[kWarps];
  uint32_t pivot[2];
  uint32_t cstar;
  alignas(16) uint32_t ranks[4];  // (A, T, U) from the CTA holding the last equal
  alignas(8) uint64_t round_full[3];
  uint64_t tile_full[kTiles];
  uint64_t mma_done[kTiles];
  uint64_t slot_free[kSlots];
  uint32_t tmem_base;
};
static_assert(sizeof(Smem) <= RS_SMEM_BYTES, "the launcher's dynamic shared memory");

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

__device__ __forceinline__ void cluster_arrive() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }

__device__ __forceinline__ uint32_t remote_addr(const void* local, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(smem_addr(local)), "r"(rank));
  return r;
}

__device__ __forceinline__ void st_async4(uint32_t addr, uint4 v, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.u32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(addr),
               "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(bar)
               : "memory");
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_addr(dst)), "l"(src) : "memory");
}

__device__ __forceinline__ void cp_async4(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem_addr(dst)), "l"(src) : "memory");
}

__device__ __forceinline__ void cp_async_arrive(uint64_t* bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_addr(bar)) : "memory");
}

__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_addr(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_addr(bar)) : "memory");
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_addr(bar)), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void mbar_wait_cluster(uint64_t* bar, uint32_t phase) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=:\n\t"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT_%=;\n\t}" ::"r"(smem_addr(bar)),
      "r"(phase)
      : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT_%=;\n\t}" ::"r"(smem_addr(bar)),
      "r"(phase)
      : "memory");
}

// One published block: 8 rows of 64 bytes, landing 64B-swizzled
__device__ __forceinline__ void tma_block(void* dst, const CUtensorMap* map, int32_t token, int32_t page,
                                          uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], "
      "[%5];" ::"r"(smem_addr(dst)),
      "l"(map), "r"(0), "r"(token), "r"(page), "r"(smem_addr(bar))
      : "memory");
}

__device__ __forceinline__ void tc_after_sync() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_before_sync() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }

// UMMA shared-memory descriptor (SM100 version 1)
__device__ __forceinline__ uint64_t smem_desc(const void* p, uint32_t sbo, uint32_t layout) {
  return static_cast<uint64_t>((smem_addr(p) >> 4) & 0x3FFF) | (static_cast<uint64_t>(sbo >> 4) << 32) |
         (1ull << 46) | (static_cast<uint64_t>(layout) << 61);
}
constexpr uint32_t kSwizzleNone = 0;
constexpr uint32_t kSwizzle64B = 4;

__device__ __forceinline__ void utccp_32x128b(uint32_t tmem, uint64_t desc) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(tmem), "l"(desc) : "memory");
}

__device__ __forceinline__ void mma_mxf4(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc,
                                         uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 [%0], %1, %2, %3, [%5], [%6], p;\n\t}" ::"r"(d),
      "l"(a), "l"(b), "r"(idesc), "r"(acc), "r"(sfa), "r"(sfb)
      : "memory");
}

__device__ __forceinline__ void mma_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_addr(bar))
               : "memory");
}

// two adjacent accumulators: columns [taddr, taddr + 64)
__device__ __forceinline__ void tmem_ld64(uint32_t taddr, float (&v)[64]) {
  uint32_t r[64];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
      "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, "
      "%37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, "
      "%59, %60, %61, %62, %63}, [%64];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
        "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
        "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
        "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]),
        "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]),
        "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]),
        "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]),
        "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63])
      : "r"(taddr));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
  for (int i = 0; i < 64; ++i) v[i] = __uint_as_float(r[i]);
}

// DeepGEMM's ptx::cvt_relu_bf16x2_f32: x in the low half
__device__ __forceinline__ __nv_bfloat162 cvt_relu_bf16x2(float x, float y) {
  uint32_t packed;
  asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;" : "=r"(packed) : "f"(y), "f"(x));
  return *reinterpret_cast<__nv_bfloat162*>(&packed);
}

// topk_bf16_small's value order: negatives below positives, -0 below +0
__device__ __forceinline__ uint32_t distort(uint32_t raw) { return raw ^ ((raw & 0x8000u) ? 0xFFFFu : 0x8000u); }
__device__ __forceinline__ uint32_t undistort(uint32_t key) {
  return (key & 0x8000u) ? (key ^ 0x8000u) : (~key & 0xFFFFu);
}
__device__ __forceinline__ float bf16_bits_to_float(uint32_t raw) { return __uint_as_float(raw << 16); }

// Exclusive prefix over the block in thread order; *total gets the sum
__device__ __forceinline__ uint32_t block_exclusive_scan(uint32_t v, uint32_t* scratch, uint32_t* total) {
  const uint32_t lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  uint32_t inc = v;
#pragma unroll
  for (uint32_t o = 1; o < 32; o <<= 1) {
    const uint32_t t = __shfl_up_sync(0xFFFFFFFFu, inc, o);
    if (lane >= o) inc += t;
  }
  if (lane == 31) scratch[warp] = inc;
  __syncthreads();
  uint32_t before = 0, sum = 0;
#pragma unroll
  for (uint32_t w = 0; w < kWarps; ++w) {
    const uint32_t s = scratch[w];
    before += w < warp ? s : 0;
    sum += s;
  }
  __syncthreads();
  *total = sum;
  return before + inc - v;
}

// One radix round over the cluster: every CTA sends its whole local histogram into its
// slot of every CTA's `round` histograms (st.async, counted by round_full[round]);
// thread b then sums bin b over the sources, and the bin holding the k-th largest key
// is stored as pivot[round] = bin | (elements of it to take) << 8.
__device__ __forceinline__ void radix_round(Smem& s, uint32_t round, uint32_t me, uint32_t k) {
  const uint32_t groups = kRoundGroups[round];
  const uint32_t tid = threadIdx.x;
  __syncthreads();  // hist_local[round]
  for (uint32_t i = tid; i < kCluster * groups; i += kThreads) {
    const uint32_t r = i / groups, g = i % groups;
    st_async4(remote_addr(&s.hist[round][me][4 * g], r), reinterpret_cast<const uint4*>(s.hist_local[round])[g],
              remote_addr(&s.round_full[round], r));
  }
  mbar_wait_cluster(&s.round_full[round], 0);
  uint32_t count = 0;
#pragma unroll
  for (uint32_t r = 0; r < kCluster; ++r) count += s.hist[round][r][tid];
  uint32_t total;
  const uint32_t below = block_exclusive_scan(count, s.scan, &total);
  // elements in bins >= b, and in bins > b
  const uint32_t at_or_above = total - below;
  const uint32_t above = at_or_above - count;
  if (at_or_above >= k && above < k) s.pivot[round] = tid | ((k - above) << 8);
  __syncthreads();
}

// Bits of a lane's 32 columns [lo, lo + 32) that fall in [a, b)
__device__ __forceinline__ uint32_t range_bits(uint32_t lo, uint32_t a, uint32_t b) {
  const uint32_t start = a > lo ? min(a - lo, 32u) : 0u;
  const uint32_t end = b > lo ? min(b - lo, 32u) : 0u;
  const uint32_t upto_end = end >= 32 ? 0xFFFFFFFFu : (1u << end) - 1;
  const uint32_t upto_start = start >= 32 ? 0xFFFFFFFFu : (1u << start) - 1;
  return upto_end & ~upto_start;
}

__device__ __forceinline__ uint32_t warp_sum(uint32_t v) {
#pragma unroll
  for (uint32_t o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
  return v;
}

}  // namespace

extern "C" __global__ __launch_bounds__(kThreads, 1) void dsv41_reindex_select(
    const __grid_constant__ CUtensorMap k_map, const uint8_t* __restrict__ kv, int64_t page_bytes,
    int32_t page_tokens, int32_t page_shift, const uint8_t* __restrict__ q, const uint32_t* __restrict__ q_sf,
    const void* __restrict__ weights, int32_t weights_bf16, const int32_t* __restrict__ blocks,
    int64_t blocks_stride, const int32_t* __restrict__ valid_lens, int32_t* __restrict__ out, int64_t out_stride,
    int32_t topk, uint16_t* __restrict__ logits_out, int64_t logits_stride, int32_t inputs_before_wait) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>(smem_raw);
  const uint32_t tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const uint32_t me = cluster_rank();
  const uint32_t row = blockIdx.y;
  const uint32_t col0 = me * kCols;
  const uint32_t k = static_cast<uint32_t>(topk);

  if (warp == 5) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_addr(&s.tmem_base)),
                 "r"(kTmemCols)
                 : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(&k_map) : "memory");
#pragma unroll
    for (uint32_t t = 0; t < kTiles; ++t) {
      mbar_init(&s.tile_full[t], 32 + 1);  // the warp's cp.async arrivals + the TMA bytes
      mbar_init(&s.mma_done[t], 1);
    }
#pragma unroll
    for (uint32_t t = 0; t < kSlots; ++t) mbar_init(&s.slot_free[t], 128);
    // each round: every CTA's message; then the ranks from one CTA
#pragma unroll
    for (uint32_t r = 0; r < 3; ++r) mbar_init(&s.round_full[r], 1);
    mbar_arrive_expect_tx(&s.round_full[0], kCluster * kRoundGroups[0] * 16);
    mbar_arrive_expect_tx(&s.round_full[1], kCluster * kRoundGroups[1] * 16);
    mbar_arrive_expect_tx(&s.round_full[2], 16);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (uint32_t i = tid; i < 2 * kHistWords; i += kThreads) (&s.hist_local[0][0])[i] = 0;
  // the query's scale words land at 4h; UTCCP reads the rest as zero
  if (tid < 128 && tid % 4 != 0) s.sf_q[tid] = 0;
  // no CTA sends into another before that one's round barriers are initialized
  cluster_arrive();

  const int32_t* row_blocks = blocks + row * blocks_stride;
  int32_t* row_out = out + row * out_stride;
  const uint8_t* q_row = q + static_cast<uint64_t>(row) * kHeads * kKeyBytes;
  __nv_bfloat162 w2[kHeads / 2];
  // the published blocks, the query (visible to the tensor cores after the next
  // barrier) and, in the epilogue warps, the head weights
  const auto load_inputs = [&]() {
    for (uint32_t j = tid; j < kBlocks; j += kThreads) s.blocks[j] = row_blocks[me * kBlocks + j];
    if (tid < kHeads * 4) {
      const uint32_t h = tid / 4, part = tid % 4;
      cp_async16(s.q + h * kKeyBytes + ((part ^ ((h >> 1) & 3)) * 16), q_row + h * kKeyBytes + part * 16);
    }
    if (tid < kHeads) cp_async4(&s.sf_q[tid * 4], q_sf + row * kHeads + tid);
    if (warp < 4) {
#pragma unroll
      for (uint32_t i = 0; i < kHeads / 2; ++i) {
        // the stock path hands DeepGEMM bf16 weights (a round-to-nearest-even cast)
        if (weights_bf16) {
          w2[i] = reinterpret_cast<const __nv_bfloat162*>(weights)[row * (kHeads / 2) + i];
        } else {
          const float2 f = reinterpret_cast<const float2*>(weights)[row * (kHeads / 2) + i];
          w2[i] = __halves2bfloat162(__float2bfloat16_rn(f.x), __float2bfloat16_rn(f.y));
        }
      }
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    fence_proxy_async();
  };

  uint32_t len = 0;
  if (inputs_before_wait) {
    len = static_cast<uint32_t>(valid_lens[row]);
    if (len > k) load_inputs();
  }
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  const uint32_t tmem = s.tmem_base;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (!inputs_before_wait) {
    len = static_cast<uint32_t>(valid_lens[row]);
    if (len > k) load_inputs();
    __syncthreads();
  }

  if (len <= k) {
    // topk_bf16_small's short row: every column, -1 past the row
    for (uint32_t t = me * kThreads + tid; t < k; t += kCluster * kThreads)
      row_out[t] = t < len ? (row_blocks[t >> 3] << 3) | static_cast<int32_t>(t & 7) : -1;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    cluster_wait();
    __syncthreads();
    if (warp == 5) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(kTmemCols));
    return;
  }
  // the row's blocks [0, valid_blocks): later ones are past its last key
  const uint32_t valid_blocks = (len + kBlockKeys - 1) / kBlockKeys - me * kBlocks;
  const auto block_at = [&](uint32_t j) -> int32_t {
    return static_cast<int32_t>(j) < static_cast<int32_t>(valid_blocks) ? s.blocks[j] : -1;
  };

  // ---- gather: warp w loads tile w ----
  {
    const uint32_t tile = warp;
    uint64_t* full = &s.tile_full[tile];
    const uint32_t in_page_mask = (1u << page_shift) - 1;
    const int32_t mine = block_at(tile * kTileBlocks + lane % kTileBlocks);
    if (lane < kTileBlocks && mine >= 0)
      tma_block(&s.k[tile][lane * kBlockKeys * kKeyBytes], &k_map,
                static_cast<int32_t>((mine & in_page_mask) * kBlockKeys), mine >> page_shift, full);
    const uint32_t loaded = __popc(__ballot_sync(0xFFFFFFFFu, lane < kTileBlocks && mine >= 0));
    if (lane == 0) mbar_arrive_expect_tx(full, loaded * kBlockKeys * kKeyBytes);
    // keys lane + 32 i fill this lane's 16-byte UTCCP row
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      const uint32_t key = lane + 32 * i;
      const int32_t phys = block_at(tile * kTileBlocks + key / kBlockKeys);
      if (phys >= 0) {
        const uint8_t* page = kv + static_cast<uint64_t>(phys >> page_shift) * page_bytes;
        const uint32_t token = (phys & in_page_mask) * kBlockKeys + key % kBlockKeys;
        cp_async4(&s.sf_k[tile][lane * 4 + i], page + static_cast<uint64_t>(page_tokens) * kKeyBytes + token * 4);
      }
    }
    cp_async_arrive(full);
  }

  if (warp == 4 && lane == 0) {
    tc_after_sync();
    utccp_32x128b(tmem + kSfqCol, smem_desc(s.sf_q, 128, kSwizzleNone));
    const uint64_t q_desc = smem_desc(s.q, 8 * kKeyBytes, kSwizzle64B);
#pragma unroll 1
    for (uint32_t tile = 0; tile < kTiles; ++tile) {
      const uint32_t slot = tile % kSlots;
      mbar_wait(&s.tile_full[tile], 0);
      fence_proxy_async();
      if (tile >= kSlots) mbar_wait(&s.slot_free[slot], 0);
      tc_after_sync();
      utccp_32x128b(tmem + kSfkCol + tile * 4, smem_desc(s.sf_k[tile], 128, kSwizzleNone));
      const uint64_t k_desc = smem_desc(s.k[tile], 8 * kKeyBytes, kSwizzle64B);
#pragma unroll
      for (uint32_t kk = 0; kk < 2; ++kk) {
        // scales 0-1, then 2-3 of each 128-value key; the operands advance 32 bytes
        const uint32_t sf_id = kk * 2;
        mma_mxf4(tmem + slot * kHeads, k_desc + kk * 2, q_desc + kk * 2, kInstrDesc | (sf_id << 4) | (sf_id << 29),
                 kk, tmem + kSfkCol + tile * 4, tmem + kSfqCol);
      }
      mma_commit(&s.mma_done[tile]);
    }
  }

  // ---- epilogue: warps 0-3 own TMEM lanes 32w..32w+31, two tiles per load ----
  if (warp < 4) {
    uint32_t zeros_pos = 0, zeros_neg = 0;
#pragma unroll 1
    for (uint32_t pair = 0; pair < kTiles; pair += 2) {
      const uint32_t slot = pair % kSlots;
      // a commit tracks every MMA issued before it
      mbar_wait(&s.mma_done[pair + 1], 0);
      tc_after_sync();
      float acc[64];
      tmem_ld64(tmem + ((warp * 32) << 16) + slot * kHeads, acc);
      tc_before_sync();
      if (pair + kSlots < kTiles) {
        mbar_arrive(&s.slot_free[slot]);
        mbar_arrive(&s.slot_free[slot + 1]);
      }
#pragma unroll
      for (uint32_t half = 0; half < 2; ++half) {
        const float* a = acc + half * kHeads;
        __nv_bfloat162 sum0 = __floats2bfloat162_rn(0.0f, 0.0f);
        __nv_bfloat162 sum1 = __floats2bfloat162_rn(0.0f, 0.0f);
#pragma unroll
        for (uint32_t h = 0; h < kHeads; h += 4) {
          sum0 = __hfma2(cvt_relu_bf16x2(a[h], a[h + 1]), w2[h / 2], sum0);
          sum1 = __hfma2(cvt_relu_bf16x2(a[h + 2], a[h + 3]), w2[h / 2 + 1], sum1);
        }
        const __nv_bfloat162 sum = __hadd2_rn(sum0, sum1);
        const __nv_bfloat16 reduced = __hadd_rn(sum.x, sum.y);
        const uint32_t col = (pair + half) * kTileKeys + warp * 32 + lane;
        const uint16_t raw = *reinterpret_cast<const uint16_t*>(&reduced);
        s.logits[col] = raw;
        const bool valid = col0 + col < len;
        if (valid) {
          atomicAdd(&s.hist_local[0][distort(raw) >> 8], 1u);
          if (logits_out != nullptr) logits_out[row * logits_stride + col0 + col] = raw;
        }
        zeros_pos += __popc(__ballot_sync(0xFFFFFFFFu, valid && raw == 0x0000u));
        zeros_neg += __popc(__ballot_sync(0xFFFFFFFFu, valid && raw == 0x8000u));
      }
    }
    if (lane == 0 && (zeros_pos | zeros_neg)) {
      atomicAdd(&s.hist_local[0][kBins], zeros_pos);
      atomicAdd(&s.hist_local[0][kBins + 1], zeros_neg);
    }
  }
  tc_before_sync();

  // ---- top-k: two radix rounds over the cluster ----
  cluster_wait();  // every CTA's round barriers are initialized
  radix_round(s, 0, me, k);
  if (warp == 5) {
    tc_after_sync();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(kTmemCols));
  }
  const uint32_t hi = s.pivot[0] & 0xFF, k2 = s.pivot[0] >> 8;
  const uint32_t first = tid * kPerThread;
  uint32_t raw[kPerThread];
  {
    const uint2 two = *reinterpret_cast<const uint2*>(&s.logits[first]);
    raw[0] = two.x & 0xFFFF, raw[1] = two.x >> 16, raw[2] = two.y & 0xFFFF, raw[3] = two.y >> 16;
  }
  bool valid[kPerThread];
#pragma unroll
  for (uint32_t i = 0; i < kPerThread; ++i) {
    valid[i] = col0 + first + i < len;
    const uint32_t key = distort(raw[i]);
    if (valid[i] && (key >> 8) == hi) atomicAdd(&s.hist_local[1][key & 0xFF], 1u);
  }
  if (tid < 32) {
    const uint32_t* words = reinterpret_cast<const uint32_t*>(s.logits);
    s.hist_local[1][kEdgeWord + tid] = words[tid < 16 ? tid : kCols / 2 - 32 + tid];
  }
  radix_round(s, 1, me, k2);
  const uint32_t lo = s.pivot[1] & 0xFF;
  const uint32_t pivot_raw = undistort(hi << 8 | lo);
  const float pivot = bf16_bits_to_float(pivot_raw);

  // Per source CTA: elements above the pivot in the distorted order, and equal to it;
  // thread (r, j) sums bins j, j + 16, ... of source r. The census compares floats, so
  // a zero pivot also matches the other zero (round 0's +0 / -0 counts).
  {
    const uint32_t r = tid / 16, j = tid % 16;
    uint32_t above = 0;
#pragma unroll
    for (uint32_t b = j; b < kBins; b += 16) {
      above += b > hi ? s.hist[0][r][b] : 0;
      above += b > lo ? s.hist[1][r][b] : 0;
    }
#pragma unroll
    for (uint32_t o = 8; o > 0; o >>= 1) above += __shfl_xor_sync(0xFFFFFFFFu, above, o);
    if (j == 0) {
      uint32_t equal = s.hist[1][r][lo];
      if (pivot_raw == 0x0000u) equal += s.hist[0][r][kBins + 1];
      if (pivot_raw == 0x8000u) {
        above -= s.hist[0][r][kBins];
        equal += s.hist[0][r][kBins];
      }
      s.above[r] = above;
      s.equal[r] = equal;
    }
  }
  // the census of this thread's columns, and its warp's inclusive prefix
  uint32_t gt_bits = 0, eq_bits = 0;
#pragma unroll
  for (uint32_t i = 0; i < kPerThread; ++i) {
    const float x = bf16_bits_to_float(raw[i]);
    gt_bits |= (valid[i] && x > pivot) ? 1u << i : 0u;
    eq_bits |= (valid[i] && x == pivot) ? 1u << i : 0u;
  }
  const uint32_t packed = __popc(gt_bits) << 16 | __popc(eq_bits);
  uint32_t packed_inc = packed;
#pragma unroll
  for (uint32_t o = 1; o < 32; o <<= 1) {
    const uint32_t t = __shfl_up_sync(0xFFFFFFFFu, packed_inc, o);
    if (lane >= o) packed_inc += t;
  }
  if (lane == 31) s.scan[warp] = packed_inc;
  __syncthreads();
  uint32_t above_all = 0, above_before = 0, eq_prefix[kCluster + 1];
  eq_prefix[0] = 0;
#pragma unroll
  for (uint32_t r = 0; r < kCluster; ++r) {
    above_all += s.above[r];
    above_before += r < me ? s.above[r] : 0;
    eq_prefix[r + 1] = eq_prefix[r] + s.equal[r];
  }
  // the +0 bin holds the pivot -0 only: then every zero is in the equal quota
  const uint32_t eq_take = pivot_raw == 0x8000u ? k - above_all : (s.pivot[1] >> 8);

  // The CTA holding the last admitted equal element (rank eq_take - 1), its warp 0: the
  // topk_bf16_small slice holding it (at most 32 columns; neighbors' edges from round 1)
  // and the equal ranks the slice spans, sent to every CTA (lane l reads columns
  // [32 l, 32 l + 32)).
  {
    const uint32_t target = eq_take - 1;
    uint32_t holder = 0;
#pragma unroll
    for (uint32_t r = 0; r < kCluster; ++r) holder += eq_prefix[r + 1] <= target ? 1 : 0;
    if (holder == me && warp == 0) {
      const uint32_t rho = target - eq_prefix[me];
      const uint32_t base = col0 + 32 * lane;
      uint32_t mask = 0;
#pragma unroll
      for (uint32_t c = 0; c < 4; ++c) {
        const uint4 v = reinterpret_cast<const uint4*>(&s.logits[32 * lane])[c];
        const uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (uint32_t e = 0; e < 8; ++e) {
          const uint32_t bit = 8 * c + e;
          const uint32_t r16 = (words[e / 2] >> (16 * (e % 2))) & 0xFFFF;
          mask |= (base + bit < len && bf16_bits_to_float(r16) == pivot) ? 1u << bit : 0u;
        }
      }
      const uint32_t n = __popc(mask);
      uint32_t inc = n;
#pragma unroll
      for (uint32_t o = 1; o < 32; o <<= 1) {
        const uint32_t t = __shfl_up_sync(0xFFFFFFFFu, inc, o);
        if (lane >= o) inc += t;
      }
      const uint32_t excl = inc - n;
      const bool owns = rho >= excl && rho < inc;
      uint32_t cstar = 0;
      if (owns) {
        uint32_t bits = mask;
        for (uint32_t m = rho - excl; m > 0; --m) bits &= bits - 1;
        cstar = base + (__ffs(bits) - 1);
      }
      cstar = __shfl_sync(0xFFFFFFFFu, cstar, __ffs(__ballot_sync(0xFFFFFFFFu, owns)) - 1);
      const uint32_t vecs = (len + 7) / 8, per = vecs / kStockThreads, extra = vecs % kStockThreads;
      const uint32_t v = cstar / 8;
      uint32_t start, count;
      if (v < extra * (per + 1)) {
        start = v / (per + 1) * (per + 1);
        count = per + 1;
      } else {
        start = extra * (per + 1) + (v - extra * (per + 1)) / per * per;
        count = per;
      }
      const uint32_t s0 = start * 8, s1 = min((start + count) * 8, len);
      uint32_t in_slice = warp_sum(__popc(mask & range_bits(base, s0, s1)));
      uint32_t below = warp_sum(__popc(mask & range_bits(base, s0, cstar)));
      // a slice crossing into a neighbor CTA (rows not aligned to the slices)
      const auto edge_eq = [&](uint32_t src, uint32_t index) -> bool {
        const uint32_t word = s.hist[1][src][kEdgeWord + index / 2];
        return bf16_bits_to_float((word >> (16 * (index % 2))) & 0xFFFF) == pivot;
      };
      if (s0 < col0) {
        const uint32_t c = s0 + lane;  // in the last 32 columns of CTA me - 1
        const bool eq = c < col0 && edge_eq(me - 1, 32 + c - (col0 - 32));
        const uint32_t n2 = __popc(__ballot_sync(0xFFFFFFFFu, eq));
        in_slice += n2;
        below += n2;
      }
      if (s1 > col0 + kCols) {
        const uint32_t c = col0 + kCols + lane;  // in the first 32 columns of CTA me + 1
        in_slice += __popc(__ballot_sync(0xFFFFFFFFu, c < s1 && edge_eq(me + 1, lane)));
      }
      const uint32_t a = target - below, u = a + in_slice;
      if (lane < kCluster)
        st_async4(remote_addr(s.ranks, lane), make_uint4(a, u - (eq_take - a), u, 0),
                  remote_addr(&s.round_full[2], lane));
    }
  }
  mbar_wait_cluster(&s.round_full[2], 0);
  const uint32_t ra = s.ranks[0], rt = s.ranks[1], ru = s.ranks[2];
  // equal ranks taken: [0, ra) and [rt, ru)
  const auto taken_below = [&](uint32_t rank) -> uint32_t {
    return min(rank, ra) + (min(rank, ru) > rt ? min(rank, ru) - rt : 0);
  };
  uint32_t packed_before = packed_inc - packed;
#pragma unroll
  for (uint32_t w = 0; w < kWarps; ++w) packed_before += w < warp ? s.scan[w] : 0;
  uint32_t eq_rank = eq_prefix[me] + (packed_before & 0xFFFF);
  // the CTAs and threads before this one: their elements above the pivot and their
  // taken equal ranks
  uint32_t pos = above_before + (packed_before >> 16) + taken_below(eq_rank);
  uint32_t take_bits = gt_bits;
#pragma unroll
  for (uint32_t i = 0; i < kPerThread; ++i) {
    if (eq_bits >> i & 1u) {
      if (eq_rank < ra || (eq_rank >= rt && eq_rank < ru)) take_bits |= 1u << i;
      ++eq_rank;
    }
  }
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#pragma unroll
  for (uint32_t i = 0; i < kPerThread; ++i) {
    if (take_bits >> i & 1u) {
      const uint32_t local = first + i;
      row_out[pos++] = (s.blocks[local / kBlockKeys] << 3) | static_cast<int32_t>(local & 7);
    }
  }
}
