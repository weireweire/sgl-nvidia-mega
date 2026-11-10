// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Router top-k of the DeepSeek-V4.1 MoE for the low-batch routed-expert GEMMs
// (routed_expert_gemm.py, gate_topk_and_gemms): the Triton moe_fused_gate's routing (sqrtsoftplus via
// log1p, fp32 bias, top-k, renormalize, routed scale) with one CTA per row and one expert
// per thread, so a row's scoring and selection take a few warp reductions. GateTopK<kExperts,
// kTopK> is instantiated by gate_topk.cu (target, 384 experts, top-6) and gate_topk_draft.cu
// (DSpark draft, 128 experts, top-3), each its own NVRTC program and extern "C" symbol.
//
// Numerics follow moe_fused_gate bit for bit: the same libdevice exp / log1p / sqrt under
// Triton's reflection (flush to zero, approximate sqrt; see GATE_TOPK_NVRTC_OPTIONS), NaNs
// ranked above every finite score, the lowest expert id wins ties, the routed sum in the
// Triton reduction's order, the same approximate division (div.full.f32), then the scale.
//
// PDL: dependents (the route kernel) launch at once and read the ids after their own wait;
// the static bias is read before this kernel's grid-dependency wait, the logits after it.

#pragma once

// Timeline hook of benchmark builds (benchmark/bench_deepseek_v41_moe_chain.py); empty here.
#ifndef DSV41_STAMP
#define DSV41_STAMP(i)
#endif

namespace {
// Order-preserving u32 key of a score (+0 for -0); 0 marks a taken expert.
__device__ __forceinline__ unsigned int Key(float v) {
  const unsigned int bits = __float_as_uint(v + 0.0f);
  return (bits & 0x80000000U) ? ~bits : (bits | 0x80000000U);
}

// The Triton kernel's fp32 division lowers to div.full.f32, not the IEEE div.rn.
__device__ __forceinline__ float DivFull(float a, float b) {
  float q;
  asm("div.full.f32 %0, %1, %2;" : "=f"(q) : "f"(a), "f"(b));
  return q;
}

// The routed sum in the order moe_fused_gate's tl.sum adds its BLOCK_K slots (padding slots
// add +0.0, exact on these non-negative weights). Top-6 (BLOCK_K 8): two threads hold 4 of
// the 8 slots each. Top-3 (BLOCK_K 4): one thread holds a row's 4 slots, (0 + 2) + (1 + 3).
template <int kTopK>
__device__ __forceinline__ float RoutedSum(const float (&w)[kTopK]) {
  static_assert(kTopK == 6 || kTopK == 3, "no Triton reduction order recorded for this top-k");
  if constexpr (kTopK == 6) {
    return ((w[0] + w[2]) + (w[1] + w[3])) + (w[4] + w[5]);
  } else {
    return (w[0] + w[2]) + w[1];
  }
}

// One row's routing, block = kExperts threads (one expert each): the row's score s and its
// bias b (thread e's), the row's validity. Every thread of the block calls it (it
// synchronizes the block once); afterwards warp 0's lanes < kTopK hold slot `lane`'s weight
// and id (id -1, weight 0 for an invalid row) and every lane of warp 0 holds all kTopK ids
// in row_ids; the other warps' results are unspecified. eps > 0 normalizes by sum + eps,
// as renormalize_epsilon; else by the guarded sum.
template <int kExperts, int kTopK>
struct GateScratch {
  unsigned int cand_key[(kExperts / 32) * kTopK];
  int cand_id[(kExperts / 32) * kTopK];
  float cand_act[(kExperts / 32) * kTopK];
};

template <int kExperts, int kTopK>
__device__ __forceinline__ void SelectRow(
    GateScratch<kExperts, kTopK>& scratch, float s, float b, bool valid_row, float scale, float eps,
    float& out_w, int& out_id, int (&row_ids)[kTopK]) {
  constexpr int kThreads = kExperts;
  constexpr int kWarps = kThreads / 32;
  constexpr int kCandidates = kWarps * kTopK;
  constexpr int kSlots = (kCandidates + 31) / 32;  // candidates per merging lane
  static_assert(kExperts % 32 == 0 && kTopK <= 32, "one expert per thread, whole warps");
  const int e = static_cast<int>(threadIdx.x);
  const int lane = e & 31;
  const int warp = e >> 5;
  const float sp = s > 20.0f ? s : log1pf(expf(s));
  const float act = sqrtf(sp);
  const float biased = act + b;
  unsigned int key = Key(biased == biased ? biased : __int_as_float(0x7f800000));

  // This warp's top-k (its ids are contiguous, so the lowest lane wins a tie).
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    const unsigned int top = __reduce_max_sync(0xffffffffU, key);
    const unsigned int win = __reduce_min_sync(0xffffffffU, key == top ? static_cast<unsigned int>(e) : 0xffffffffU);
    if (static_cast<unsigned int>(e) == win) {
      scratch.cand_key[warp * kTopK + k] = key;
      scratch.cand_id[warp * kTopK + k] = e;
      scratch.cand_act[warp * kTopK + k] = act;
      key = 0U;
    }
  }
  __syncthreads();
  if (warp != 0) return;

  // Merge the kWarps * kTopK candidates: every global top-k expert is in its warp's top-k.
  unsigned int ck[kSlots];
  int ci[kSlots];
#pragma unroll
  for (int j = 0; j < kSlots; ++j) {
    const int c = lane + 32 * j;
    ck[j] = c < kCandidates ? scratch.cand_key[c] : 0U;
    ci[j] = c < kCandidates ? scratch.cand_id[c] : kExperts;
  }
  float sel_w[kTopK];
  int sel_i[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    unsigned int best = ck[0];
#pragma unroll
    for (int j = 1; j < kSlots; ++j) best = max(best, ck[j]);
    const unsigned int top = __reduce_max_sync(0xffffffffU, best);
    unsigned int mine = 0xffffffffU;
#pragma unroll
    for (int j = 0; j < kSlots; ++j) {
      if (ck[j] == top) mine = min(mine, static_cast<unsigned int>(ci[j]));
    }
    const int id = static_cast<int>(__reduce_min_sync(0xffffffffU, mine));
    int slot = -1;
#pragma unroll
    for (int j = 0; j < kSlots; ++j) {
      if (ck[j] == top && ci[j] == id) { slot = lane + 32 * j; ck[j] = 0U; }
    }
    const unsigned int owner = __ballot_sync(0xffffffffU, slot >= 0);
    const int src = __shfl_sync(0xffffffffU, slot, __ffs(owner) - 1);
    sel_w[k] = scratch.cand_act[src];
    sel_i[k] = id;
  }
  const float sum = RoutedSum<kTopK>(sel_w);
  const float norm = eps > 0.0f ? sum + eps : (sum > 0.0f ? sum : 1.0f);
#pragma unroll
  for (int k = 0; k < kTopK; ++k) row_ids[k] = valid_row ? sel_i[k] : -1;
  float w = 0.0f;
  int id = 0;
  if (lane < kTopK) {
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      if (lane == k) { w = sel_w[k]; id = sel_i[k]; }
    }
    w = DivFull(w, norm) * scale;
    if (!valid_row) { w = 0.0f; id = -1; }
  }
  out_w = w;
  out_id = id;
}

// grid = rows, block = kExperts. num_valid: rows >= *num_valid get weight 0 and id -1 (padded
// graph rows), as moe_fused_gate's num_token_non_padded; null means every row is valid. With
// input_ids (int32, or int64 when ids64), a row whose token is alt_token_id ranks with
// bias_alt instead of bias (DSV4.1 VL routing). eps > 0 normalizes by sum + eps, as
// renormalize_epsilon; else by the guarded sum.
template <int kExperts, int kTopK>
__device__ __forceinline__ void GateTopK(
    const float* logits, const float* bias, const float* bias_alt, const void* input_ids, int ids64,
    long long alt_token_id, const int* num_valid, float* __restrict__ weights, int* __restrict__ ids,
    float scale, float eps) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  DSV41_STAMP(0);
  __shared__ GateScratch<kExperts, kTopK> scratch;
  const int e = static_cast<int>(threadIdx.x);
  const int lane = e & 31;
  const int warp = e >> 5;
  const int row = static_cast<int>(blockIdx.x);
  const float b_main = bias[e];
  const float b_alt = bias_alt != nullptr ? bias_alt[e] : b_main;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  DSV41_STAMP(1);

  const int valid = num_valid != nullptr ? *num_valid : static_cast<int>(gridDim.x);
  bool alt = false;
  if (input_ids != nullptr && row < valid) {
    const long long token = ids64 ? static_cast<const long long*>(input_ids)[row]
                                  : static_cast<long long>(static_cast<const int*>(input_ids)[row]);
    alt = token == alt_token_id;
  }
  const float b = alt ? b_alt : b_main;
  const float s = logits[row * kExperts + e];
  float w;
  int id;
  int row_ids[kTopK];
  SelectRow<kExperts, kTopK>(scratch, s, b, row < valid, scale, eps, w, id, row_ids);
  DSV41_STAMP(2);
  if (warp == 0 && lane < kTopK) {
    weights[row * kTopK + lane] = w;
    ids[row * kTopK + lane] = id;
  }
  DSV41_STAMP(3);
}
}  // namespace
