# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SwigluLowLatencyBlockscaledGemmKernel.epilog_warp is adapted from FlashInfer's
# LowLatencyBlockscaledGemmKernel.epilog_warp, which is distributed under this notice:
#
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The epilogue is FlashInfer's LowLatencyBlockscaledGemmKernel.epilog_warp
# (flashinfer/gemm/kernels/cute_dsl/low_latency_blockscaled_gemm.py, flashinfer-python
# 0.7.3) up to its split-K reduction; what follows the reduction is new.

"""The shared expert's gate_up on the low-latency MXFP8 GEMM, writing SwiGLU.

Problem. The shared expert (slot ``shared_ll``) runs its gate_up GEMM, then SGLang's
clamped SwiGLU (``silu_and_mul_clamp``) and the MXFP8 quant of ``down_proj``'s input
as two more launches on the alternate stream, beside the routed GEMMs.

Layout. :func:`interleave_gate_up` reorders the weight rows so each 128-row tile
holds 64 gate rows and their 64 up rows: tile ``t`` is gate rows ``64t .. 64t+63``
then up rows ``I + 64t .. I + 64t + 63`` (``I`` = N / 2, any multiple of 32). When
64 does not divide I (TP8's shard: I = 288) the last tile holds the 32 remaining
gate rows at 0-31 and their up rows at 64-95, the rest zero rows: the weight is
padded to whole tiles. Each output element is the same K reduction as before, in
the same order (the tactic is the one the unpadded N selects, and the split-K
partials are added in the base kernel's order), so the BF16 gate_up values are the
unfused GEMM's.

Epilogue. After the split-K reduction every CTA of the cluster sends its reduced
FP32 shard to the cluster's first CTA, into the A-stage shared memory (dead once
the last MMA completed, which the peers know: they reduce only after receiving
that CTA's partials, sent after its MMA). That CTA rounds to BF16 and computes,
per element, SGLang's clamped SwiGLU with the instructions its CUDA kernel compiles
to (``-use_fast_math``: bf16 min / max clamp, ex2.approx, div.approx, ftz), then

- ``quant=False``: stores the BF16 rows ``h`` [M, I] into C;
- ``quant=True``: quantizes ``h`` as FlashInfer's cute-dsl ``mxfp8_quantize`` does
  (its own PTX helpers; 128x4 swizzled UE8M0 scales with zeroed padding) into
  buffers at compile-time addresses, the input ``down_proj`` takes as
  ``Mxfp8SwizzledInput``.

PDL. The A-load warp does not release the dependents; the epilogue does, after the
CTA's last store (FlashInfer's kernel releases them once its loads are issued, which
``early_release`` keeps, for measurement).
"""

from __future__ import annotations

from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import PipelineUserType, make_pipeline_state
from flashinfer.gemm.kernels.cute_dsl.low_latency_blockscaled_gemm import (
    LowLatencyBlockscaledGemmKernel,
    WorkTileInfo,
)
from flashinfer.quantization.quantization_cute_dsl_utils import (
    INV_FLOAT8_E4M3_MAX,
    bfloat2_to_fp8x2_scaled,
    float_to_ue8m0_fast,
    ue8m0_to_inv_scale_fast,
)

from sgl_nvidia_mega.common.sync import _asm

TILE_M = 128  # weight rows per CTA: 64 gate + 64 up
HALF = TILE_M // 2
SF_VEC = 32  # MXFP8 block
# the 128x4 scale atom: 128 rows x 4 scale columns
SF_ATOM_ROWS, SF_ATOM_COLS = 128, 4


# --------------------------------------------------------------------- weight layout


def _unswizzle_128x4(scale: torch.Tensor, n: int, k_sf: int) -> torch.Tensor:
    """FlashInfer's 128x4-swizzled UE8M0 bytes as [n, k_sf] rows (n % 128 == 0)."""
    s = scale.reshape(-1).view(torch.uint8)
    if s.numel() < n * k_sf:
        raise ValueError(f"{s.numel()} scale bytes, the 128x4 layout needs {n * k_sf}")
    s = s[: n * k_sf]
    s = s.view(n // 128, k_sf // 4, 32, 4, 4)  # (row block, col block, r32, r4, c4)
    return s.permute(0, 3, 2, 1, 4).reshape(n, k_sf)


def _swizzle_128x4(linear: torch.Tensor) -> torch.Tensor:
    n, k_sf = linear.shape
    s = linear.view(n // 128, 4, 32, k_sf // 4, 4)
    return s.permute(0, 3, 2, 1, 4).reshape(-1).contiguous()


def _tiles(n: int) -> int:
    return -(-(n // 2) // HALF)


def gate_up_order(n: int, device=None) -> torch.Tensor:
    """The source row of each interleaved row (whole 128-row tiles): per tile, 64
    gate rows then the matching 64 up rows; -1 for a padding row."""
    inter = n // 2
    p = torch.arange(_tiles(n) * TILE_M, device=device)
    tile, local = p // TILE_M, p % TILE_M
    j = tile * HALF + local % HALF  # the SwiGLU column
    return torch.where(j < inter, torch.where(local < HALF, j, inter + j), -1)


def interleave_gate_up(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The gate_up weight [2I, K] (E4M3) and its 128x4-swizzled UE8M0 scales with the
    rows in :func:`gate_up_order`, padding rows zero; new tensors, the inputs are
    untouched."""
    n, k = weight.shape
    if n % (2 * SF_VEC) or k % 128:
        raise ValueError(f"gate_up {n} x {k}: needs N % 64 == 0 and K % 128 == 0")
    order = gate_up_order(n, weight.device)
    valid = order >= 0
    w = torch.zeros((order.numel(), k), dtype=torch.uint8, device=weight.device)
    w[valid] = weight.view(torch.uint8).index_select(0, order[valid])
    k_sf = k // SF_VEC
    linear = _unswizzle_128x4(weight_scale, -(-n // TILE_M) * TILE_M, k_sf)
    interleaved = torch.zeros(
        (order.numel(), k_sf), dtype=torch.uint8, device=weight.device
    )
    interleaved[valid] = linear.index_select(0, order[valid])
    s = _swizzle_128x4(interleaved)
    return (
        w.view(weight.dtype),
        s.view(weight_scale.dtype) if weight_scale.dtype != torch.uint8 else s,
    )


def limit_bits(limit: float) -> int:
    """``limit`` as SGLang's kernel holds it: rounded to BF16."""
    return (
        int(
            torch.tensor([limit], dtype=torch.float32)
            .to(torch.bfloat16)
            .view(torch.int16)
            .item()
        )
        & 0xFFFF
    )


def scale_layout(rows: int, inter: int) -> tuple[int, int]:
    """(padded rows, padded scale columns) of FlashInfer's 128x4 scales for [rows, I]."""
    return (
        -(-rows // SF_ATOM_ROWS) * SF_ATOM_ROWS,
        -(-(inter // SF_VEC) // SF_ATOM_COLS) * SF_ATOM_COLS,
    )


# ------------------------------------------------------------------------- PTX ops


@dsl_user_op
def swiglu_clamp_bf16(
    gate: Float32, up: Float32, limit: int, *, loc=None, ip=None
) -> Uint32:
    """SGLang's ``silu_and_mul<true>`` on the BF16 roundings of two FP32
    accumulators, as its CUDA kernel compiles under ``-use_fast_math``; the BF16
    result bits in the low half."""
    text = (
        "{\n\t.reg .b16 g, u, l, nl;\n\t.reg .f32 gf, uf, e, s, y;\n\t"
        "cvt.rn.bf16.f32 g, $1;\n\tcvt.rn.bf16.f32 u, $2;\n\t"
        f"mov.b16 l, {limit:#06x};\n\tmov.b16 nl, {limit ^ 0x8000:#06x};\n\t"
        "min.bf16 g, g, l;\n\tmax.bf16 u, u, nl;\n\tmin.bf16 u, u, l;\n\t"
        "cvt.f32.bf16 gf, g;\n\tcvt.f32.bf16 uf, u;\n\t"
        "mul.ftz.f32 e, gf, 0fBFB8AA3B;\n\tex2.approx.ftz.f32 e, e;\n\t"
        "add.ftz.f32 e, e, 0f3F800000;\n\tdiv.approx.ftz.f32 s, gf, e;\n\t"
        "mul.ftz.f32 y, uf, s;\n\tcvt.rn.bf16.f32 g, y;\n\t"
        "cvt.u32.u16 $0, g;\n\t}"
    )
    return Uint32(
        _asm(
            T.i32(),
            [gate.ir_value(loc=loc, ip=ip), up.ir_value(loc=loc, ip=ip)],
            text,
            "=r,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16_bits_abs_f32(bits: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [bits.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 w;\n\tshl.b32 w, $1, 16;\n\tand.b32 w, w, 0x7FFFFFFF;\n\t"
            "mov.b32 $0, w;\n\t}",
            "=f,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _max_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_global_v2_u32(
    address: Int64, lo: Uint32, hi: Uint32, *, loc=None, ip=None
) -> None:
    _asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            lo.ir_value(loc=loc, ip=ip),
            hi.ir_value(loc=loc, ip=ip),
        ],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_u32(address: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_u8(address: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 v;\n\tcvt.u16.u32 v, $1;\n\tst.global.b8 [$0], v;\n\t}",
        "l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _sf_offset(row: Int32, col: Int32, padded_cols: cutlass.Constexpr[int]) -> Int32:
    """FlashInfer's 128x4 swizzle (``compute_sf_index_swizzled_128x4_gpu``)."""
    return (
        (row // SF_ATOM_ROWS) * (padded_cols * SF_ATOM_ROWS)
        + (col // SF_ATOM_COLS) * (SF_ATOM_ROWS * SF_ATOM_COLS)
        + (row % 32) * 16
        + ((row % SF_ATOM_ROWS) // 32) * 4
        + col % SF_ATOM_COLS
    )


# -------------------------------------------------------------------------- kernel


class SwigluLowLatencyBlockscaledGemmKernel(LowLatencyBlockscaledGemmKernel):
    """FlashInfer's low-latency block-scaled GEMM on an interleaved gate_up weight
    whose epilogue writes the clamped SwiGLU (and with ``quant`` its MXFP8)."""

    def __init__(
        self,
        *,
        limit: int,
        quant: bool = False,
        inter: int = 0,
        q_address: int = 0,
        sf_address: int = 0,
        early_release: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if self.cta_m != TILE_M or self.cta_n != 8:
            raise ValueError("the fold takes 128 x 8 output tiles")
        if self.use_bias or self.use_scale:
            raise ValueError("no bias or output scale with the fold")
        if inter <= 0 or inter % SF_VEC:
            raise ValueError("the fold takes I % 32 == 0")
        if quant and (q_address % 16 or sf_address % 16):
            raise ValueError("quant: 16-byte aligned buffers")
        self.limit = limit
        self.early_release = early_release
        self.quant = quant
        self.inter = inter
        self.q_address = q_address
        self.sf_address = sf_address
        self.last_m = -(-inter // HALF) - 1
        self.sf_cols = inter // SF_VEC
        self.padded_sf_cols = -(-self.sf_cols // SF_ATOM_COLS) * SF_ATOM_COLS

    def dma_a_warp(self, *args):
        """The base A-load warp, without its release unless ``early_release`` (then
        the epilogue does not release)."""
        use_pdl = self.use_pdl
        self.use_pdl = use_pdl and self.early_release
        super().dma_a_warp(*args)
        self.use_pdl = use_pdl

    @cute.jit
    def epilog_warp(
        self,
        tma_epilog_full_bar,
        mma_epilog_full_bar,
        tmem_alloc_result_bar,
        tiled_mma: cute.TiledMma,
        mC_mnl: cute.Tensor,
        mScale: Optional[cute.Tensor],  # noqa: UP045 (as the base class)
        mBias_mnl: Optional[cute.Tensor],  # noqa: UP045
        tmem_base_smem_ptr,
        mailbox_mbar,
        dsmem_mailbox_ptr,
        epi_tid: cutlass.Int32,
        mma_tile_coord_v: cutlass.Int32,
        work_tile_info: WorkTileInfo,
        split_rank: cutlass.Int32,
        k_tile_count: cutlass.Int32,
    ):
        """FlashInfer's epilogue up to the split-K reduction, then the gather and the
        SwiGLU (+ quant) on the cluster's first CTA."""
        gC_mnl = cute.local_tile(mC_mnl, (self.cta_m, self.cta_n), (None, None, None))
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgC = thr_mma.partition_C(gC_mnl)

        acc_shape = tiled_mma.partition_shape_C((self.cta_m, self.cta_n))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)
        cute.arch.mbarrier_wait(tmem_alloc_result_bar, 0)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_smem_ptr)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            (self.cta_m, self.cta_n, self.cta_k),
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            (self.cta_m, self.cta_n),
            self.use_2cta_instrs,
        )
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tCtAcc[((None, None), 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tid)
        tDtAcc = thr_copy_t2r.partition_S(tCtAcc[((None, None), 0, 0)])
        tDgC = thr_copy_t2r.partition_D(tCgC[((None, None), 0, 0, None, None, None)])
        epi_frag_shape = tDgC[(None, None, None, 0, 0, 0)].shape
        tDrAcc = cute.make_rmem_tensor(epi_frag_shape, self.acc_dtype)

        coordC = cute.make_identity_tensor(mC_mnl.shape)
        gCcoord = cute.local_tile(coordC, (self.cta_m, self.cta_n), (None, None, None))
        tCcC = thr_mma.partition_C(gCcoord)
        tDcC = thr_copy_t2r.partition_D(tCcC[((None, None), 0, 0, None, None, None)])
        tDcC = tDcC[
            (
                None,
                None,
                None,
                work_tile_info.M_idx,
                work_tile_info.N_idx,
                work_tile_info.L_idx,
            )
        ]

        mma_epilog_state = make_pipeline_state(PipelineUserType.Consumer, 1)
        cute.arch.mbarrier_wait(
            mma_epilog_full_bar + mma_epilog_state.index, mma_epilog_state.phase
        )

        # FOLD: the gather area past the barrier/mailbox struct, inside the A stages
        # (sA starts at most 127 bytes past the struct): an mbarrier, then the
        # [8 tokens][128 rows] FP32 tile.
        struct_end = dsmem_mailbox_ptr + self._mailbox_total_elems
        gather_bar = cute.recast_ptr(struct_end + 32, dtype=cutlass.Int64)
        tile = cute.make_tensor(
            struct_end + 64, cute.make_layout((self.cta_n * self.cta_m,))
        )
        if cutlass.const_expr(self.split_k > 1):
            if split_rank == 0 and epi_tid == 0:
                cute.arch.mbarrier_init(gather_bar, 1)
                cute.arch.mbarrier_init_fence()
                cute.arch.mbarrier_arrive_and_expect_tx(
                    gather_bar, self._mailbox_tx_total
                )

        cute.copy(tiled_copy_t2r, tDtAcc, tDrAcc)
        cute.arch.fence_view_async_tmem_load()

        if cutlass.const_expr(self.split_k > 1):  # noqa: SIM102
            if k_tile_count == 0:
                tDrAcc.fill(0.0)

        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)

        m_base = work_tile_info.M_idx * self.cta_m
        n_base = work_tile_info.N_idx * self.cta_n
        elems_per_thread = self._mailbox_elems_per_thread
        if cutlass.const_expr(self.split_k == 1):
            for i in cutlass.range_constexpr(elems_per_thread):
                coord = tDcC[i]
                tile[(coord[1] - n_base) * self.cta_m + coord[0] - m_base] = tDrAcc[i]
        else:
            shard_ept = self._shard_elems_per_thread

            if epi_tid == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mailbox_mbar, self._mailbox_tx_total
                )

            cute.arch.barrier(barrier_id=15, number_of_threads=128)

            for peer in range(self.split_k):
                if peer != split_rank:
                    shard_start = peer * shard_ept
                    sender_idx = split_rank - (
                        1 if split_rank > cutlass.Int32(peer) else 0
                    )
                    mailbox_base = sender_idx * 128 * shard_ept
                    for i in range(shard_ept):
                        cute.arch.store_async_dsmem(
                            dsmem_mailbox_ptr + mailbox_base + i * 128 + epi_tid,
                            tDrAcc[shard_start + i].bitcast(cutlass.Int32),
                            mailbox_mbar,
                            cutlass.Int32(peer),
                        )

            cute.arch.mbarrier_wait(mailbox_mbar, 0)

            my_shard_start = split_rank * shard_ept
            for s in range(self.split_k - 1):
                mailbox_base = s * 128 * shard_ept
                for i in range(shard_ept):
                    tDrAcc[my_shard_start + i] = (
                        tDrAcc[my_shard_start + i]
                        + (dsmem_mailbox_ptr + mailbox_base + i * 128 + epi_tid).load()
                    )

            # FOLD: this CTA's reduced shard into the first CTA's tile
            for i in range(shard_ept):
                coord = tDcC[my_shard_start + i]
                slot = (coord[1] - n_base) * self.cta_m + coord[0] - m_base
                if split_rank == 0:
                    tile[slot] = tDrAcc[my_shard_start + i]
                else:
                    cute.arch.store_async_dsmem(
                        tile.iterator + slot,
                        tDrAcc[my_shard_start + i].bitcast(cutlass.Int32),
                        gather_bar,
                        cutlass.Int32(0),
                    )

        if split_rank == 0:
            if cutlass.const_expr(self.split_k > 1):
                cute.arch.mbarrier_wait(gather_bar, 0)
            cute.arch.barrier(barrier_id=14, number_of_threads=128)
            self.swiglu_tile(
                tile, mC_mnl, work_tile_info.M_idx, work_tile_info.N_idx, epi_tid
            )
        if cutlass.const_expr(not self.early_release):
            cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def swiglu_tile(
        self,
        tile,
        mC_mnl: cute.Tensor,
        m_idx: Int32,
        n_idx: Int32,
        epi_tid: Int32,
    ):
        """The 64 SwiGLU columns x 8 tokens of a gathered tile: 16 threads a token,
        4 columns a thread (8 threads: one 32-column MXFP8 block)."""
        rows = mC_mnl.shape[1]
        token_local = epi_tid // 16
        col = (epi_tid % 16) * 4
        token = n_idx * self.cta_n + token_local
        bits = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
        for q in cutlass.range_constexpr(4):
            gate = token_local * self.cta_m + col + q
            bits[q] = swiglu_clamp_bf16(tile[gate], tile[gate + HALF], self.limit)
        h_col = m_idx * HALF + col
        if cutlass.const_expr(not self.quant):
            if token < rows:  # noqa: SIM102
                if h_col < self.inter:
                    address = mC_mnl.iterator.toint() + Int64(
                        (token * self.inter + h_col) * 2
                    )
                    _st_global_v2_u32(
                        address,
                        bits[0] | (bits[1] << Uint32(16)),
                        bits[2] | (bits[3] << Uint32(16)),
                    )
        else:
            amax = _max_f32(
                _max_f32(_bf16_bits_abs_f32(bits[0]), _bf16_bits_abs_f32(bits[1])),
                _max_f32(_bf16_bits_abs_f32(bits[2]), _bf16_bits_abs_f32(bits[3])),
            )
            for offset in cutlass.range_constexpr(3):
                amax = _max_f32(
                    amax, cute.arch.shuffle_sync_bfly(amax, offset=1 << offset)
                )
            normalized_max = amax * Float32(INV_FLOAT8_E4M3_MAX)
            scale_u32 = float_to_ue8m0_fast(normalized_max)
            inv_scale = ue8m0_to_inv_scale_fast(scale_u32)
            fp8_lo = bfloat2_to_fp8x2_scaled(
                bits[0] | (bits[1] << Uint32(16)), inv_scale
            )
            fp8_hi = bfloat2_to_fp8x2_scaled(
                bits[2] | (bits[3] << Uint32(16)), inv_scale
            )
            sf_col = h_col // SF_VEC
            # the last tile's columns past I (a half tile) are zero rows: no store
            if h_col < self.inter:
                first_of_block = (epi_tid % 8) == 0
                if token < rows:
                    _st_global_u32(
                        Int64(self.q_address) + Int64(token * self.inter + h_col),
                        (fp8_lo & Uint32(0xFFFF))
                        | ((fp8_hi & Uint32(0xFFFF)) << Uint32(16)),
                    )
                    if first_of_block:
                        _st_global_u8(
                            Int64(self.sf_address)
                            + Int64(_sf_offset(token, sf_col, self.padded_sf_cols)),
                            scale_u32,
                        )
                elif first_of_block:
                    _st_global_u8(
                        Int64(self.sf_address)
                        + Int64(_sf_offset(token, sf_col, self.padded_sf_cols)),
                        Uint32(0),
                    )
            self.zero_scale_padding(rows, m_idx, n_idx, epi_tid)

    @cute.jit
    def zero_scale_padding(
        self, rows: Int32, m_idx: Int32, n_idx: Int32, epi_tid: Int32
    ):
        """The scale bytes FlashInfer zeroes that no token tile covers: the padding
        rows past the last tile (by the last token tile's CTAs, their real columns)
        and the padding columns (by the last column tile's CTAs)."""
        padded_rows = (rows + SF_ATOM_ROWS - 1) // SF_ATOM_ROWS * SF_ATOM_ROWS
        last_n = (rows + self.cta_n - 1) // self.cta_n - 1
        tile_end = (n_idx + 1) * self.cta_n
        col0 = m_idx * (HALF // SF_VEC)
        cols = Int32(HALF // SF_VEC)
        if m_idx == self.last_m:
            cols = self.sf_cols - col0
        if n_idx == last_n:
            index = epi_tid
            span = (padded_rows - tile_end) * cols
            while index < span:
                _st_global_u8(
                    Int64(self.sf_address)
                    + Int64(
                        _sf_offset(
                            tile_end + index // cols,
                            col0 + index % cols,
                            self.padded_sf_cols,
                        )
                    ),
                    Uint32(0),
                )
                index = index + 128
        if cutlass.const_expr(self.padded_sf_cols > self.sf_cols):
            pad_cols = self.padded_sf_cols - self.sf_cols
            if m_idx == self.last_m:
                row_end = tile_end
                if n_idx == last_n:
                    row_end = padded_rows
                index = epi_tid
                span = (row_end - n_idx * self.cta_n) * pad_cols
                while index < span:
                    _st_global_u8(
                        Int64(self.sf_address)
                        + Int64(
                            _sf_offset(
                                n_idx * self.cta_n + index // pad_cols,
                                self.sf_cols + index % pad_cols,
                                self.padded_sf_cols,
                            )
                        ),
                        Uint32(0),
                    )
                    index = index + 128


__all__ = [
    "SwigluLowLatencyBlockscaledGemmKernel",
    "gate_up_order",
    "interleave_gate_up",
    "limit_bits",
    "scale_layout",
    "swiglu_clamp_bf16",
]
