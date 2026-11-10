// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Router top-k and route record of the DeepSeek-V4.1 low-batch routed-expert GEMMs in one
// launch (slot gate_route): gate_topk.cuh's SelectRow per row and route.cuh's Build over
// the rows' ids, the outputs of dsv41_gate_topk followed by dsv41_routed_expert_route bit
// for bit (the same code), without the second launch and its grid-dependency hop.
//
// grid = cluster = rows (1-8), block = kExperts. CTA r routes row r and writes its weights
// and ids; CTAs r > 0 send their ids to CTA 0's shared memory with st.async, each store
// completing its bytes on CTA 0's mbarrier (no release fence); CTA 0 waits for the bytes
// and builds the record and the slot -> row map. The cluster barrier that publishes CTA 0's
// mbarrier init is split and relaxed: every thread arrives at entry and waits only before
// its first remote access (CTA 0: before it waits on the mbarrier), so its latency
// overlaps the grid-dependency wait and the top-k.
//
// PDL: dependents (GEMM1) launch at once and read the record after their own wait. The
// static biases are read before this kernel's grid-dependency wait; the logits, the valid
// row count and the token ids after it, as independent loads. No global access touches
// the outputs before the wait.

#pragma once

#include "gate_topk.cuh"
#include "route.cuh"

namespace {

__device__ __forceinline__ unsigned int SharedAddress(const void* p) {
  return static_cast<unsigned int>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ unsigned int ClusterRank() {
  unsigned int r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

__device__ __forceinline__ unsigned int MapToRank(unsigned int address, unsigned int rank) {
  unsigned int r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(address), "r"(rank));
  return r;
}

template <int kExperts, int kTopK>
__device__ __forceinline__ void GateRoute(
    const float* logits, const float* bias, const float* bias_alt, const void* input_ids, int ids64,
    long long alt_token_id, const int* num_valid, float* __restrict__ weights, int* __restrict__ ids,
    float scale, float eps, int* __restrict__ permuted, int* __restrict__ record, int experts,
    int expert_offset) {
  static_assert(kExperts >= route::kTile * 8, "the block must cover rows * top_k slots");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  DSV41_STAMP(0);
  __shared__ GateScratch<kExperts, kTopK> scratch;
  __shared__ route::Scratch route_scratch;
  __shared__ int row_ids_of[route::kTile * kTopK];      // CTA 0: every row's ids
  __shared__ __align__(8) unsigned long long received;  // CTA 0: the other rows' ids landed
  const int e = static_cast<int>(threadIdx.x);
  const int lane = e & 31;
  const int warp = e >> 5;
  const int row = static_cast<int>(blockIdx.x);
  const int rows = static_cast<int>(gridDim.x);
  const unsigned int rank = ClusterRank();
  const unsigned int received_address = SharedAddress(&received);
  if (rank == 0) {
    route::Clear(route_scratch, e, kExperts);
    if (e == 0) {
      // one arrival (this thread's, with the byte count), completed by the other rows'
      // st.async transactions
      const unsigned int bytes = static_cast<unsigned int>((rows - 1) * kTopK * 4);
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(received_address) : "memory");
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(received_address), "r"(bytes)
                   : "memory");
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
  }
  asm volatile("barrier.cluster.arrive.relaxed;" ::: "memory");
#ifndef GATE_ROUTE_BIAS_AFTER_WAIT
  const float b_main = bias[e];
  const float b_alt = bias_alt != nullptr ? bias_alt[e] : b_main;
#endif
  asm volatile("griddepcontrol.wait;" ::: "memory");
  DSV41_STAMP(1);
#ifdef GATE_ROUTE_BIAS_AFTER_WAIT
  const float b_main = bias[e];
  const float b_alt = bias_alt != nullptr ? bias_alt[e] : b_main;
#endif

  // independent loads: the row's token (in bounds for every launched row) and the count
  const int valid = num_valid != nullptr ? *num_valid : rows;
  long long token = 0;
  if (input_ids != nullptr) {
    token = ids64 ? static_cast<const long long*>(input_ids)[row]
                  : static_cast<long long>(static_cast<const int*>(input_ids)[row]);
  }
  const float s = logits[row * kExperts + e];
  const bool valid_row = row < valid;
  const bool alt = input_ids != nullptr && valid_row && token == alt_token_id;
  const float b = alt ? b_alt : b_main;
  float w;
  int id;
  int row_ids[kTopK];
  SelectRow<kExperts, kTopK>(scratch, s, b, valid_row, scale, eps, w, id, row_ids);
  DSV41_STAMP(2);
  if (warp == 0 && lane < kTopK) {
    weights[row * kTopK + lane] = w;
    ids[row * kTopK + lane] = id;
  }

  if (rank != 0) {
    asm volatile("barrier.cluster.wait;" ::: "memory");  // CTA 0's mbarrier is initialized
    if (warp == 0 && lane < kTopK) {
      const unsigned int remote = MapToRank(SharedAddress(&row_ids_of[row * kTopK + lane]), 0);
      const unsigned int remote_received = MapToRank(received_address, 0);
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(remote),
                   "r"(id), "r"(remote_received)
                   : "memory");
    }
    DSV41_STAMP(4);
    return;
  }

  // CTA 0: its own ids, then the other rows'
  if (warp == 0 && lane < kTopK) row_ids_of[lane] = id;
  route::Zero(record, rows, kTopK, experts, e, kExperts);
  asm volatile("barrier.cluster.wait;" ::: "memory");
  if (e == 0) {
    unsigned int done = 0;
    while (!done) {
      asm volatile(
          "{\n"
          ".reg .pred p;\n"
          "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;\n"
          "selp.u32 %0, 1, 0, p;\n"
          "}\n"
          : "=r"(done)
          : "r"(received_address)
          : "memory");
    }
  }
  __syncthreads();
  DSV41_STAMP(3);
  route::Build(route_scratch, row_ids_of, permuted, record, nullptr, rows, kTopK, experts, expert_offset, e);
  DSV41_STAMP(4);
}

}  // namespace
