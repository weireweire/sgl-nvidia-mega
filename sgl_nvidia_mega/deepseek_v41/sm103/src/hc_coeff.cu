// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// The K-split body is adapted from DeepGEMM (https://github.com/deepseek-ai/DeepGEMM),
// deep_gemm/include/deep_gemm/impls/sm100_tf32_hc_prenorm_gemm.cuh
// (sm100_tf32_hc_prenorm_gemm_impl) as SGLang's 2026-10-30 nightly ships it; modified by
// NVIDIA (plain stores of the partials, the finish ticket and the Sinkhorn tail). Its
// license:
//
//   MIT License
//
//   Copyright (c) 2025 DeepSeek
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy
//   of this software and associated documentation files (the "Software"), to deal
//   in the Software without restriction, including without limitation the rights
//   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//   copies of the Software, and to permit persons to whom the Software is
//   furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included in all
//   copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//   SOFTWARE.
//
// The mHC coefficients (pre, post, comb) of up to 64 rows in one launch, bit for bit
// what DeepGEMM's TF32 prenorm (kNumSplits K slices) followed by SGLang's Triton
// reduce + Sinkhorn kernel (_hc_mix_reduce_sinkhorn_kernel) compute:
//  - every CTA runs one K slice exactly as DeepGEMM's kernel does (the same TMA tiles,
//    the same BF16 -> FP32 cast into tensor memory, the same TF32 UMMA sequence and the
//    same FFMA2 square sums), and stores its partials with plain stores;
//  - the CTA that takes the last ticket sums the slices in slice order and runs the
//    Triton kernel's arithmetic: the same PTX operations in the same order (fma.rn, add /
//    mul / sub left contractible, div.full / sqrt.approx.ftz / ex2.approx, shuffle-butterfly sums
//    over the same lanes, one warp per row as its one-warp CTA per row).
#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

#if !defined(HC_EPS) || !defined(HC_HC) || !defined(HC_K) || !defined(HC_SPLITS) || !defined(HC_STAGES) || !defined(HC_ITERS)
#error "define HC_EPS (a float literal), HC_HC, HC_K, HC_SPLITS, HC_STAGES and HC_ITERS"
#endif

namespace sgl_nvidia_mega {

using namespace deep_gemm;

template <uint32_t kSwizzleMode, uint32_t kSwizzleBase = 16>
CUTLASS_DEVICE uint32_t swizzled_smem_offset(const uint32_t& offset, const uint32_t& lane_idx) {
    const auto bank_group_idx = offset + lane_idx * (kSwizzleMode / kSwizzleBase);
    constexpr uint32_t kNumBankGroups = 128 / kSwizzleBase;
    constexpr bool kHasShortcut = (kSwizzleMode / kSwizzleBase) == kNumBankGroups;
    auto row = kHasShortcut ? (offset / kNumBankGroups + lane_idx) : (bank_group_idx / kNumBankGroups);
    auto col = kHasShortcut ? (offset) : (bank_group_idx % kNumBankGroups);
    col ^= row % (kSwizzleMode / kSwizzleBase);
    return row * 128 + col * kSwizzleBase;
}

// The Triton kernel's operations, as its PTX spells them: add / sub / mul without a
// rounding mode, so ptxas contracts them as it does Triton's (it folds the multiply of
// a div.full into the add of eps that follows)
CUTLASS_DEVICE float div_full(float a, float b) {
    float r;
    asm volatile("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
CUTLASS_DEVICE float sqrt_approx_ftz(float a) {
    float r;
    asm volatile("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}
CUTLASS_DEVICE float ex2_approx(float a) {
    float r;
    asm volatile("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}
CUTLASS_DEVICE float fma_rn(float a, float b, float c) {
    float r;
    asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}
CUTLASS_DEVICE float add_f32(float a, float b) {
    float r;
    asm volatile("add.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
CUTLASS_DEVICE float sub_f32(float a, float b) {
    float r;
    asm volatile("sub.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
CUTLASS_DEVICE float mul_f32(float a, float b) {
    float r;
    asm volatile("mul.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
CUTLASS_DEVICE float max_f32(float a, float b) {
    float r;
    asm volatile("max.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
CUTLASS_DEVICE float bfly(float v, int mask) {
    return __shfl_xor_sync(0xffffffffu, v, mask, 32);
}

// One row's coefficients by one warp, the Triton kernel's lane layout: lane bits 0-1
// index the column (and pre / post), bits 2-3 the comb row; lanes 16-31 repeat 0-15.
template <uint32_t MIX, uint32_t HC, uint32_t kNumSplits, uint32_t kIters>
CUTLASS_DEVICE void sinkhorn_row(const uint32_t row, const uint32_t shape_m, const uint32_t lane,
                                 const float* part_mix, const float* part_sq,
                                 const float* scale, const float* base,
                                 float* pre_out, float* post_out, float* comb_out,
                                 const float inv_k, const float rms_eps) {
    const uint32_t k = lane & 3, j = (lane >> 2) & 3;
    float a_pre = 0.0f, a_post = 0.0f, a_comb = 0.0f, sq = 0.0f;
    #pragma unroll
    for (uint32_t s = 0; s < kNumSplits; ++ s) {
        const float* p = part_mix + (s * shape_m + row) * MIX;
        const float v_pre = __ldcg(p + k), v_post = __ldcg(p + HC + k);
        const float v_comb = __ldcg(p + 2 * HC + j * HC + k);
        const float v_sq = __ldcg(part_sq + s * shape_m + row);
        a_pre = add_f32(a_pre, v_pre);
        a_post = add_f32(a_post, v_post);
        a_comb = add_f32(a_comb, v_comb);
        sq = add_f32(sq, v_sq);
    }
    const float rsqrt = div_full(1.0f, sqrt_approx_ftz(fma_rn(inv_k, sq, rms_eps)));
    const float s0 = scale[0], s1 = scale[1], s2 = scale[2];
    constexpr float kNegLog2e = -1.44269502162933349609375f;  // 0xBFB8AA3B
    constexpr float kLog2e = 1.44269502162933349609375f;      // 0x3FB8AA3B

    float x = fma_rn(mul_f32(a_pre, rsqrt), s0, base[k]);
    const float pre = add_f32(div_full(1.0f, add_f32(ex2_approx(mul_f32(x, kNegLog2e)), 1.0f)), HC_EPS);
    x = fma_rn(mul_f32(a_post, rsqrt), s1, base[HC + k]);
    float post = div_full(1.0f, add_f32(ex2_approx(mul_f32(x, kNegLog2e)), 1.0f));
    post = add_f32(post, post);

    float c = fma_rn(mul_f32(a_comb, rsqrt), s2, base[2 * HC + j * HC + k]);
    float m = max_f32(c, bfly(c, 2));
    m = max_f32(m, bfly(m, 1));
    c = ex2_approx(mul_f32(sub_f32(c, m), kLog2e));
    float t = add_f32(c, bfly(c, 2));
    t = add_f32(t, bfly(t, 1));
    c = add_f32(div_full(c, t), HC_EPS);
    t = add_f32(c, bfly(c, 8));
    t = add_f32(t, bfly(t, 4));
    c = div_full(c, add_f32(t, HC_EPS));
    #pragma unroll 1
    for (uint32_t it = 1; it < kIters; ++ it) {
        t = add_f32(c, bfly(c, 2));
        t = add_f32(t, bfly(t, 1));
        c = div_full(c, add_f32(t, HC_EPS));
        t = add_f32(c, bfly(c, 8));
        t = add_f32(t, bfly(t, 4));
        c = div_full(c, add_f32(t, HC_EPS));
    }
    if (lane < 4) {
        pre_out[row * HC + k] = pre;
        post_out[row * HC + k] = post;
    }
    if (lane < 16)
        comb_out[row * HC * HC + j * HC + k] = c;
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumSplits, uint32_t kNumStages,
          uint32_t HC, uint32_t kIters>
CUTLASS_DEVICE void
hc_coeff(const uint32_t shape_m,
         const cute::TmaDescriptor& tensor_map_a,
         const cute::TmaDescriptor& tensor_map_b,
         float* part_mix, float* part_sq,
         const float* scale, const float* base,
         float* pre_out, float* post_out, float* comb_out,
         uint32_t* ticket, const float inv_k, const float rms_eps) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000))
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    constexpr uint32_t kNumMMAThreads = 128, kNumCastAndReduceThreads = 128;
    constexpr uint32_t kNumCastStages = 2;
    constexpr uint32_t kSwizzleAMode = cute::min(BLOCK_K * sizeof(nv_bfloat16), 128);
    constexpr uint32_t kSwizzleBMode = cute::min(BLOCK_K * sizeof(float), 128);
    constexpr auto kMajorA = cute::UMMA::Major::K;
    constexpr auto kMajorB = cute::UMMA::Major::K;
    static_assert(BLOCK_M == 64, "one 64-row tile");
    static_assert(SHAPE_N == (2 + HC) * HC, "the mix width");

    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(float);
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<BLOCK_K * kNumCastStages + BLOCK_N>();

    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
    }
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<nv_bfloat16*>(smem_buffer + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers       = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto full_cast_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto empty_barriers      = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto empty_cast_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + i); });
    auto tmem_full_barrier   = barrier_start_ptr + kNumStages * 4;
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 4 + 1);
    auto last_flag = tmem_ptr_in_smem + 1;

    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            full_cast_barriers[i]->init(kNumCastAndReduceThreads);
            empty_barriers[i]->init(1);
            empty_cast_barriers[i]->init(1);
        }
        tmem_full_barrier->init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();
    const uint32_t tmem_base = *tmem_ptr_in_smem;

    constexpr uint32_t kNumKBlocks = math::constexpr_ceil_div(SHAPE_K, BLOCK_K);
    constexpr uint32_t kNumKBlocksPerSplit = kNumKBlocks / kNumSplits;
    constexpr uint32_t kRemainKBlocks = kNumKBlocks % kNumSplits;
    const uint32_t k_split_idx = __shfl_sync(0xffffffff, blockIdx.x, 0);
    const uint32_t k_offset = (k_split_idx * kNumKBlocksPerSplit + cute::min(k_split_idx, kRemainKBlocks)) * BLOCK_K;
    const uint32_t m_offset = shape_m * k_split_idx;
    const uint32_t num_total_stages = kNumKBlocksPerSplit + (k_split_idx < kRemainKBlocks);

    cudaGridDependencySynchronize();

    if (warp_idx < kNumMMAThreads / 32) {
        if (warp_idx == 0 and cute::elect_one_sync()) {
            for (uint32_t s = 0; s < num_total_stages; ++ s) {
                const auto& stage_idx = s % kNumStages;
                empty_barriers[stage_idx]->wait(((s / kNumStages) & 1) ^ 1);
                const uint32_t k_idx = k_offset + s * BLOCK_K;
                tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode>(&tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, 0);
                tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode>(&tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, 0);
                full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
            }
        }
        if (warp_idx == 1) {
            constexpr uint32_t UMMA_M = BLOCK_M;
            constexpr uint32_t UMMA_N = BLOCK_N;
            constexpr uint32_t UMMA_K = 32 / sizeof(float);
            constexpr uint32_t BLOCK_SWIZZLED_BK = kSwizzleBMode / sizeof(float);
            using umma_t = cute::SM100_MMA_TF32_TS<cutlass::tfloat32_t, cutlass::tfloat32_t, float,
                                                   BLOCK_M, BLOCK_N, kMajorA, kMajorB>;
            auto instr_desc = cute::UMMA::make_instr_desc<cutlass::tfloat32_t, cutlass::tfloat32_t, float,
                                                          UMMA_M, UMMA_N, kMajorA, kMajorB>();
            const auto& runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
            auto b_desc = mma::sm100::make_umma_desc<kMajorB, BLOCK_N, BLOCK_SWIZZLED_BK, kSwizzleBMode>(smem_b[0], 0, 0);
            const uint32_t& b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;
            for (uint32_t s = 0; s < num_total_stages; ++ s) {
                const auto& stage_idx = s % kNumStages;
                const auto& cast_stage_idx = s % kNumCastStages;
                full_cast_barriers[cast_stage_idx]->wait((s / kNumCastStages) & 1);
                ptx::tcgen05_after_thread_sync();
                const auto& b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(stage_idx));
                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                    const uint32_t& atom_idx = (k * UMMA_K) / BLOCK_SWIZZLED_BK;
                    const uint32_t& in_atom_idx = (k * UMMA_K) % BLOCK_SWIZZLED_BK;
                    const uint32_t& offset = atom_idx * BLOCK_N * BLOCK_SWIZZLED_BK;
                    b_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorB, BLOCK_N, kSwizzleBMode, float>(b_desc_base_lo, offset, in_atom_idx);
                    umma_t::fma(tmem_base + BLOCK_K * cast_stage_idx + k * UMMA_K, b_desc, tmem_base + BLOCK_K * kNumCastStages, s > 0 or k > 0, runtime_instr_desc);
                }
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_cast_barriers[cast_stage_idx]));
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
            }
            cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barrier));
        }

        // The slice's partials: row (16 x warp + lane) at the warp's lanes 0-15 (layout F)
        tmem_full_barrier->wait(0);
        ptx::tcgen05_after_thread_sync();
        const uint32_t m_idx = warp_idx * (BLOCK_M / 4) + lane_idx;
        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_N / 4; ++ i) {
            uint32_t values[4];
            cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_base + BLOCK_K * kNumCastStages + i * 4, values[0], values[1], values[2], values[3]);
            cutlass::arch::fence_view_async_tmem_load();
            #pragma unroll
            for (uint32_t v = 0; v < 4; ++ v) {
                if (lane_idx < 16 and m_idx < shape_m and i * 4 + v < SHAPE_N)
                    part_mix[(m_offset + m_idx) * SHAPE_N + i * 4 + v] = __uint_as_float(values[v]);
            }
            __syncwarp();
        }
        cutlass::arch::NamedBarrier::sync(kNumMMAThreads, 0);
        if (warp_idx == 1)
            cute::TMEM::Allocator1Sm().free(tmem_base, kNumTmemCols);
    } else {
        constexpr uint32_t BLOCK_M_PER_WARP = BLOCK_M / 4;
        const uint32_t sub_warp_idx = warp_idx - kNumMMAThreads / 32;
        float2 sum[2] = {float2{0, 0}, float2{0, 0}};
        #pragma unroll kNumStages
        for (uint32_t s = 0; s < num_total_stages; ++ s) {
            const auto& stage_idx = s % kNumStages;
            full_barriers[stage_idx]->wait((s / kNumStages) & 1);
            constexpr uint32_t kNumBankGroupBytes = 16;
            constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(nv_bfloat16);
            constexpr uint32_t kNumLoads = BLOCK_K / kNumElemsPerBankGroup;
            const auto& cast_stage_idx = s % kNumCastStages;
            const auto& smem_base_ptr = reinterpret_cast<uint8_t*>(smem_a[stage_idx]) +
                                        sub_warp_idx * BLOCK_M_PER_WARP * kSwizzleAMode;
            uint32_t uint32_values[2][kNumLoads];
            #pragma unroll
            for (uint32_t i = 0; i < kNumLoads; i += 2) {
                auto smem_ptr = smem_base_ptr + swizzled_smem_offset<kSwizzleAMode>(i + lane_idx / 16, lane_idx % 16);
                ptx::SM90_U32x4_LDSM_N::copy(uint32_values[0][i + 0], uint32_values[1][i + 0],
                                             uint32_values[0][i + 1], uint32_values[1][i + 1],
                                             smem_ptr);
            }
            empty_cast_barriers[cast_stage_idx]->wait(((s / kNumCastStages) & 1) ^ 1);
            float2 fp32x2_values[2][kNumLoads];
            const auto& upper_view = reinterpret_cast<uint32_t*>(&fp32x2_values[0]);
            const auto& lower_view = reinterpret_cast<uint32_t*>(&fp32x2_values[1]);
            #pragma unroll
            for (uint32_t i = 0; i < kNumLoads; ++ i) {
                #pragma unroll
                for (uint32_t u = 0; u < 2; ++ u) {
                    fp32x2_values[u][i] = make_float2(__uint_as_float(uint32_values[u][i] << 16),
                                                      __uint_as_float(uint32_values[u][i] & 0xffff0000u));
                    sum[u] = __ffma2_rn(fp32x2_values[u][i], fp32x2_values[u][i], sum[u]);
                }
                const auto idx_0 = i * 2, idx_1 = i * 2 + 1;
                cute::SM100_TMEM_STORE_16dp256b1x::copy(
                    upper_view[idx_0], upper_view[idx_1],
                    lower_view[idx_0], lower_view[idx_1],
                    tmem_base + cast_stage_idx * BLOCK_K + i * 8);
            }
            cutlass::arch::fence_view_async_tmem_store();
            cutlass::arch::fence_view_async_shared();
            ptx::tcgen05_before_thread_sync();
            full_cast_barriers[cast_stage_idx]->arrive();
        }
        #pragma unroll
        for (uint32_t u = 0; u < 2; ++ u) {
            const auto reduced_sum = math::warp_reduce_sum<4>(sum[u].x + sum[u].y);
            const auto m_idx = sub_warp_idx * BLOCK_M_PER_WARP + lane_idx / 4 + u * 8;
            if (lane_idx % 4 == 0 and m_idx < shape_m)
                part_sq[m_offset + m_idx] = reduced_sum;
        }
    }

    // The last slice to finish runs the reduce + Sinkhorn over all of them
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
        *last_flag = atomicAdd(ticket, 1u) == kNumSplits - 1;
    __syncthreads();
    if (*last_flag == 0)
        return;
    __threadfence();
    for (uint32_t row = warp_idx; row < shape_m; row += 8)
        sinkhorn_row<SHAPE_N, HC, kNumSplits, kIters>(row, shape_m, lane_idx, part_mix, part_sq, scale, base,
                                                      pre_out, post_out, comb_out, inv_k, rms_eps);
    if (threadIdx.x == 0)
        *ticket = 0;
#endif
}

}  // namespace sgl_nvidia_mega

// HC_K: the K of the streams (hc x hidden); HC_SPLITS and HC_STAGES as DeepGEMM's
// heuristic picks for it; HC_ITERS: the Sinkhorn iterations
extern "C" __global__ void __launch_bounds__(256, 1)
dsv41_hc_coeff(const uint32_t shape_m,
               const __grid_constant__ cute::TmaDescriptor tensor_map_a,
               const __grid_constant__ cute::TmaDescriptor tensor_map_b,
               float* part_mix, float* part_sq,
               const float* scale, const float* base,
               float* pre_out, float* post_out, float* comb_out,
               uint32_t* ticket, const float inv_k, const float rms_eps) {
    sgl_nvidia_mega::hc_coeff<(2 + HC_HC) * HC_HC, HC_K, 64, 32, 64, HC_SPLITS, HC_STAGES, HC_HC, HC_ITERS>(
        shape_m, tensor_map_a, tensor_map_b, part_mix, part_sq, scale, base,
        pre_out, post_out, comb_out, ticket, inv_k, rms_eps);
}
