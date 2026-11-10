# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# QNormLowLatencyGemmKernel.__call__ and .kernel are adapted from FlashInfer's
# LowLatencyBlockscaledGemmKernel (flashinfer/gemm/kernels/cute_dsl/
# low_latency_blockscaled_gemm.py, flashinfer-python 0.7.2), which is distributed
# under this notice:
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

"""The query-LoRA RMSNorm + MXFP8 quant inside the low-latency MXFP8 GEMM that reads
it (slot attn_q_norm_gemm).

Problem. At 1-8 rows the chain runs wqkv_a, then ``q_norm_quant.rmsnorm_mxfp8`` (one
Triton launch, ~0.5 us of work), then wq_b on FlashInfer's low-latency GEMM; each
hop is a PDL wait on the whole previous grid.

Kernel. FlashInfer's GEMM with its B operand (the activation rows) made in the
kernel instead of loaded: warps 3-6, idle on the direct-SFB path, wait on the grid
dependency, read the BF16 rows, and write the e4m3 codes into the swizzled B stages
and the UE8M0 scales into the SFB stages; the A (weight) loads, the MMA, the split-K
reduction and the epilogue are FlashInfer's, with the tactic the unfused GEMM takes.

Bitwise. Every CTA recomputes each row's sum of squares with the instruction tree
the Triton norm compiles to (8 warps of 32 threads, 8 contiguous elements a thread,
``ld.global.v4`` then ``mul.rn.f32x2`` / ``add.rn.f32x2``; butterfly 16, 8, 4, 2, 1;
the warp sums through shared memory, butterfly 4, 2, 1; ``div.full.f32`` by K,
``+ eps``, ``rsqrt.approx.ftz.f32``; ``(inv * v) * w`` and ``cvt.rn.bf16.f32``), the
same MXFP8 epilogue (``q_norm_quant.mxfp8_epilogue``), and feeds the MMA the bytes
TMA would have loaded. With ``write_quant`` the CTAs of the first M tile also store
the codes and 128x4 scales (padding rows zero), bitwise ``rmsnorm_mxfp8``'s ``(q, s)``,
for a GEMM that follows on the same rows (the index query's, or wq_b behind it).

Limits: 1-8 rows (one N tile), K in (1024, 2048] with K % 128 == 0 (the Triton
kernel's 2048-wide block), MXFP8 A/B with UE8M0 scales, BF16 out.
"""

from __future__ import annotations

import os
import struct
from typing import NamedTuple, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    PipelineAsync,
    PipelineUmmaAsync,
    pipeline_init_arrive,
    pipeline_init_wait,
)
from flashinfer.gemm.kernels.cute_dsl.low_latency_blockscaled_gemm import (
    LowLatencyBlockscaledGemmKernel,
    WorkTileInfo,
)

from sgl_nvidia_mega.common.sync import _asm, global_timer_ns

MAX_ROWS = 8
# The Triton norm's block: 2048 elements, 8 warps x 32 threads x 8 elements
TRITON_BLOCK = 2048
CHUNK = 8  # elements per Triton thread
WARP_CHUNKS = 32  # Triton threads per warp
PRODUCER_WARP0 = 3  # warps 3-6: idle on FlashInfer's direct-SFB path
PRODUCER_THREADS = 256  # warps 3-6 then the epilogue's 8-11 (idle until the MMA ends)
PRODUCER_BARRIER = 13  # named barrier id; FlashInfer's epilogue uses 14 / 15
# Test only: print the B / SFB stage layouts the producer writes when compiling
LAYOUT_ENV = "SGL_NVIDIA_MEGA_PRINT_Q_NORM_GEMM_LAYOUTS"
INV_E4M3_MAX_BITS = struct.unpack("<I", struct.pack("<f", 1.0 / 448.0))[0]


def f32_bits(value: float) -> int:
    """``value`` as Triton bakes a Python float constant: rounded to FP32."""
    return struct.unpack("<I", struct.pack("<f", value))[0]


# ------------------------------------------------------------------------- PTX ops


def _struct(n):
    return llvm.StructType.get_literal([T.i32()] * n)


def _words(result, n, loc, ip):
    return [
        Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
        for i in range(n)
    ]


@dsl_user_op
def _ld_global_v4(address: Int64, valid: Int32, *, loc=None, ip=None):
    """16 bytes, zeros where ``valid`` is 0; ordered after the grid dependency wait
    (a side effect, so the compiler keeps it there)."""
    result = _asm(
        _struct(4),
        [address.ir_value(loc=loc, ip=ip), valid.ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $5, 0;\n\t"
        "mov.b32 $0, 0;\n\tmov.b32 $1, 0;\n\tmov.b32 $2, 0;\n\tmov.b32 $3, 0;\n\t"
        "@p ld.global.v4.b32 {$0, $1, $2, $3}, [$4];\n\t}",
        "=r,=r,=r,=r,l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return _words(result, 4, loc, ip)


@dsl_user_op
def _chunk_square_sum(
    w0: Uint32, w1: Uint32, w2: Uint32, w3: Uint32, *, loc=None, ip=None
) -> Float32:
    """One Triton thread's partial sum of 8 BF16 squares (words: elements 2i, 2i+1),
    in its order: ``((s0+s2)+(s4+s6)) + ((s1+s3)+(s5+s7))``."""
    text = (
        "{\n\t.reg .f32 e<8>;\n\t.reg .b32 t;\n\t"
        "shl.b32 t, $1, 16;\n\tmov.b32 e0, t;\n\tand.b32 t, $1, 0xFFFF0000;\n\tmov.b32 e1, t;\n\t"
        "shl.b32 t, $2, 16;\n\tmov.b32 e2, t;\n\tand.b32 t, $2, 0xFFFF0000;\n\tmov.b32 e3, t;\n\t"
        "shl.b32 t, $3, 16;\n\tmov.b32 e4, t;\n\tand.b32 t, $3, 0xFFFF0000;\n\tmov.b32 e5, t;\n\t"
        "shl.b32 t, $4, 16;\n\tmov.b32 e6, t;\n\tand.b32 t, $4, 0xFFFF0000;\n\tmov.b32 e7, t;\n\t"
        "mul.rn.f32 e0, e0, e0;\n\tmul.rn.f32 e1, e1, e1;\n\tmul.rn.f32 e2, e2, e2;\n\t"
        "mul.rn.f32 e3, e3, e3;\n\tmul.rn.f32 e4, e4, e4;\n\tmul.rn.f32 e5, e5, e5;\n\t"
        "mul.rn.f32 e6, e6, e6;\n\tmul.rn.f32 e7, e7, e7;\n\t"
        "add.rn.f32 e0, e0, e2;\n\tadd.rn.f32 e1, e1, e3;\n\t"
        "add.rn.f32 e4, e4, e6;\n\tadd.rn.f32 e5, e5, e7;\n\t"
        "add.rn.f32 e0, e0, e4;\n\tadd.rn.f32 e1, e1, e5;\n\t"
        "add.rn.f32 $0, e0, e1;\n\t}"
    )
    return Float32(
        _asm(
            T.f32(),
            [w.ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
            text,
            "=f,r,r,r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _add_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _inv_rms(total: Float32, k: int, eps_bits: int, *, loc=None, ip=None) -> Float32:
    """``rsqrt(total / K + eps)`` as Triton compiles it."""
    return Float32(
        _asm(
            T.f32(),
            [total.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .f32 m;\n\t"
            f"div.full.f32 m, $1, 0f{f32_bits(float(k)):08X};\n\t"
            f"add.rn.f32 m, m, 0f{eps_bits:08X};\n\t"
            "rsqrt.approx.ftz.f32 $0, m;\n\t}",
            "=f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _norm_bf16x2(inv: Float32, x: Uint32, w: Uint32, *, loc=None, ip=None) -> Uint32:
    """Two normed BF16 elements ``bf16((inv * v) * w)`` from packed BF16 pairs."""
    text = (
        "{\n\t.reg .f32 v0, v1, w0, w1;\n\t.reg .b32 t;\n\t"
        "shl.b32 t, $2, 16;\n\tmov.b32 v0, t;\n\tand.b32 t, $2, 0xFFFF0000;\n\tmov.b32 v1, t;\n\t"
        "shl.b32 t, $3, 16;\n\tmov.b32 w0, t;\n\tand.b32 t, $3, 0xFFFF0000;\n\tmov.b32 w1, t;\n\t"
        "mul.rn.f32 v0, $1, v0;\n\tmul.rn.f32 v1, $1, v1;\n\t"
        "mul.rn.f32 v0, v0, w0;\n\tmul.rn.f32 v1, v1, w1;\n\t"
        "cvt.rn.bf16x2.f32 $0, v1, v0;\n\t}"
    )
    return Uint32(
        _asm(
            T.i32(),
            [
                inv.ir_value(loc=loc, ip=ip),
                x.ir_value(loc=loc, ip=ip),
                w.ir_value(loc=loc, ip=ip),
            ],
            text,
            "=r,f,r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _amax_bf16x8(
    y0: Uint32, y1: Uint32, y2: Uint32, y3: Uint32, *, loc=None, ip=None
) -> Float32:
    text = (
        "{\n\t.reg .f32 a<8>;\n\t.reg .b32 t;\n\t"
        "shl.b32 t, $1, 16;\n\tand.b32 t, t, 0x7FFFFFFF;\n\tmov.b32 a0, t;\n\t"
        "and.b32 t, $1, 0x7FFF0000;\n\tmov.b32 a1, t;\n\t"
        "shl.b32 t, $2, 16;\n\tand.b32 t, t, 0x7FFFFFFF;\n\tmov.b32 a2, t;\n\t"
        "and.b32 t, $2, 0x7FFF0000;\n\tmov.b32 a3, t;\n\t"
        "shl.b32 t, $3, 16;\n\tand.b32 t, t, 0x7FFFFFFF;\n\tmov.b32 a4, t;\n\t"
        "and.b32 t, $3, 0x7FFF0000;\n\tmov.b32 a5, t;\n\t"
        "shl.b32 t, $4, 16;\n\tand.b32 t, t, 0x7FFFFFFF;\n\tmov.b32 a6, t;\n\t"
        "and.b32 t, $4, 0x7FFF0000;\n\tmov.b32 a7, t;\n\t"
        "max.f32 a0, a0, a1;\n\tmax.f32 a2, a2, a3;\n\tmax.f32 a4, a4, a5;\n\t"
        "max.f32 a6, a6, a7;\n\tmax.f32 a0, a0, a2;\n\tmax.f32 a4, a4, a6;\n\t"
        "max.f32 $0, a0, a4;\n\t}"
    )
    return Float32(
        _asm(
            T.f32(),
            [w.ir_value(loc=loc, ip=ip) for w in (y0, y1, y2, y3)],
            text,
            "=f,r,r,r,r",
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
def _ue8m0(amax: Float32, *, loc=None, ip=None):
    """``q_norm_quant.ue8m0_scale``: (scale byte, multiplier into the e4m3 range)."""
    text = (
        "{\n\t.reg .f32 n;\n\t.reg .b32 bits, e, m, sf, t;\n\t"
        ".reg .pred pm, pz, pl, pb, pn, p0;\n\t"
        f"mul.rn.f32 n, $2, 0f{INV_E4M3_MAX_BITS:08X};\n\t"
        "mov.b32 bits, n;\n\tshr.u32 e, bits, 23;\n\tand.b32 e, e, 255;\n\t"
        "and.b32 m, bits, 0x7FFFFF;\n\t"
        "setp.ne.u32 pm, m, 0;\n\tsetp.eq.u32 pz, e, 0;\n\tsetp.le.u32 pl, m, 0x400000;\n\t"
        "and.pred pz, pz, pl;\n\tnot.pred pz, pz;\n\tand.pred pb, pm, pz;\n\t"
        "selp.u32 t, 1, 0, pb;\n\tadd.u32 sf, e, t;\n\tmin.u32 sf, sf, 254;\n\t"
        "setp.le.f32 pn, n, 0f00000000;\n\tselp.u32 sf, 0, sf, pn;\n\t"
        "setp.eq.u32 p0, sf, 0;\n\tsub.u32 t, 254, sf;\n\tshl.b32 t, t, 23;\n\t"
        "selp.b32 t, 0, t, p0;\n\tmov.b32 $0, sf;\n\tmov.b32 $1, t;\n\t}"
    )
    result = _asm(
        _struct(2),
        [amax.ir_value(loc=loc, ip=ip)],
        text,
        "=r,=r,f",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    sf, scale_bits = _words(result, 2, loc, ip)
    return sf, scale_bits


@dsl_user_op
def _e4m3x4(lo: Uint32, hi: Uint32, scale_bits: Uint32, *, loc=None, ip=None) -> Uint32:
    """Four BF16 elements (two packed pairs) as e4m3 bytes, element 0 lowest:
    ``clamp(v * scale, -448, 448)`` then ``cvt.rn.satfinite``."""
    text = (
        "{\n\t.reg .f32 s, v<4>;\n\t.reg .b32 t;\n\t.reg .b16 h0, h1;\n\t"
        "mov.b32 s, $3;\n\t"
        "shl.b32 t, $1, 16;\n\tmov.b32 v0, t;\n\tand.b32 t, $1, 0xFFFF0000;\n\tmov.b32 v1, t;\n\t"
        "shl.b32 t, $2, 16;\n\tmov.b32 v2, t;\n\tand.b32 t, $2, 0xFFFF0000;\n\tmov.b32 v3, t;\n\t"
        "mul.rn.f32 v0, v0, s;\n\tmul.rn.f32 v1, v1, s;\n\t"
        "mul.rn.f32 v2, v2, s;\n\tmul.rn.f32 v3, v3, s;\n\t"
        "max.f32 v0, v0, 0fC3E00000;\n\tmax.f32 v1, v1, 0fC3E00000;\n\t"
        "max.f32 v2, v2, 0fC3E00000;\n\tmax.f32 v3, v3, 0fC3E00000;\n\t"
        "min.f32 v0, v0, 0f43E00000;\n\tmin.f32 v1, v1, 0f43E00000;\n\t"
        "min.f32 v2, v2, 0f43E00000;\n\tmin.f32 v3, v3, 0f43E00000;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h0, v1, v0;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h1, v3, v2;\n\t"
        "mov.b32 $0, {h0, h1};\n\t}"
    )
    return Uint32(
        _asm(
            T.i32(),
            [w.ir_value(loc=loc, ip=ip) for w in (lo, hi, scale_bits)],
            text,
            "=r,r,r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_shared_v2(address: Int32, lo: Uint32, hi: Uint32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            lo.ir_value(loc=loc, ip=ip),
            hi.ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v2.b32 [$0], {$1, $2};",
        "r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_shared_u32(address: Int32, value: Uint32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_shared_u8(address: Int32, value: Uint32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 v;\n\tcvt.u16.u32 v, $1;\n\tst.shared.b8 [$0], v;\n\t}",
        "r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_v2(address: Int64, lo: Uint32, hi: Uint32, *, loc=None, ip=None) -> None:
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
def _st_global_u64(address: Int64, value: Int64, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "st.global.b64 [$0], $1;",
        "l,l",
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


@dsl_user_op
def _fence_async_shared(*, loc=None, ip=None) -> None:
    _asm(
        None,
        [],
        "fence.proxy.async.shared::cta;",
        "",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _swizzle128(address: Int32) -> Int32:
    """The 128-byte swizzle TMA and the UMMA descriptor apply: the 16-byte chunk index
    XOR address bits 7-9."""
    return address ^ (((address >> 7) & 7) << 4)


@cute.jit
def _sf_offset(row: Int32, col: Int32) -> Int32:
    """Byte of (row, scale column) in a 128x4 scale atom sequence (rows < 128)."""
    return (col // 4) * 512 + (row % 32) * 16 + (row // 32) * 4 + col % 4


# -------------------------------------------------------------------------- kernel


class QNormLowLatencyGemmKernel(LowLatencyBlockscaledGemmKernel):
    """FlashInfer's low-latency block-scaled GEMM whose B operand is the MXFP8 of
    ``rmsnorm(x) * w`` made in the kernel (module docstring)."""

    def __init__(
        self,
        *,
        eps: float,
        k: int,
        write_quant: bool = False,
        stamps: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # bench only: an int64 [CTAs, 8] buffer at this address takes %globaltimer stamps
        self.stamps = stamps
        if self.cta_n != MAX_ROWS or self.cta_m != 128 or self.sf_vec_size != 32:
            raise ValueError("the fused norm takes 128 x 8 tiles of MXFP8")
        if self.use_bias or self.use_scale or self.sfb_tmem_store:
            raise ValueError("no bias, output scale or SFB TMEM staging")
        if k % 256 or not TRITON_BLOCK // 2 < k <= TRITON_BLOCK:
            raise ValueError(f"K {k}: the Triton norm's 2048-wide block, K % 256 == 0")
        if self.cta_k not in (128, 256) or k % self.cta_k:
            raise ValueError(f"cta_k {self.cta_k} with K {k}")
        self.k = k
        self.eps_bits = f32_bits(eps)
        self.write_quant = write_quant
        # chunks of 8 elements a producer thread holds: c = 32 j + t, j < K / 256
        self.thread_chunks = k // (CHUNK * WARP_CHUNKS)
        self.k_tiles = k // self.cta_k
        # K tiles per split-K rank, as the kernel splits them
        self.split_tiles = -(-self.k_tiles // self.split_k)

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        x_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        q_ptr: cute.Pointer,
        sf_ptr: cute.Pointer,
        problem_mnkl: Tuple,
        x_stride: Int32,
        stream: cuda.CUstream,
    ):
        """``c = a @ mxfp8(rmsnorm(x) * w).T``: ``a`` / ``sfa`` the MXFP8 weight and its
        swizzled scales, ``x`` BF16 rows [N, K] at row stride ``x_stride`` elements,
        ``w`` the BF16 norm weight; ``q`` / ``sf`` the quant outputs (``write_quant``)."""
        m, n, k, l = problem_mnkl
        a = cute.make_tensor(
            a_ptr, cute.make_ordered_layout((m, k, l), order=(1, 0, 2))
        )
        # B is never loaded: its tensor only fixes the operand types and major mode
        b = cute.make_tensor(
            q_ptr, cute.make_ordered_layout((n, k, l), order=(1, 0, 2))
        )
        c = cute.make_tensor(
            c_ptr, cute.make_ordered_layout((m, n, l), order=(0, 1, 2))
        )
        self.a_dtype = a_ptr.value_type
        self.b_dtype = q_ptr.value_type
        self.c_dtype = c_ptr.value_type
        self.bias_dtype = self.c_dtype
        self.sf_dtype = sfa_ptr.value_type
        self.mxf8f6f4 = self.needs_unpack_tma(self.a_dtype, self.b_dtype)
        self.smem_alloc_a_dtype = self.a_dtype
        self.smem_alloc_b_dtype = self.b_dtype
        self.a_major_mode = cutlass.utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = cutlass.utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = cutlass.utils.LayoutEnum.from_tensor(c)
        sfa = cute.make_tensor(
            sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size)
        )
        self._setup_attributes()
        if cutlass.const_expr(os.environ.get(LAYOUT_ENV) == "1"):
            print("q_norm_gemm B stages", self.b_smem_layout_staged)
            print("q_norm_gemm SFB stages", self._sfb_smem_layout_padded)
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            (self.cta_m, self.cta_n),
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            (self.cta_m, self.cta_n, self.cta_k),
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            sfa,
            sfa_smem_layout,
            (self.cta_m, self.cta_n, self.cta_k),
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        self.tma_bytes_a = (a_copy_size + sfa_copy_size) * atom_thr_size
        grid = (
            cute.ceil_div(c.layout.shape[0], self.cta_m),
            cute.ceil_div(c.layout.shape[1], self.cta_n),
            c.layout.shape[2] * self.split_k,
        )
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_sfa,
            tma_tensor_sfa,
            c,
            x_ptr,
            w_ptr,
            q_ptr,
            sf_ptr,
            x_stride,
            n,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self._sfb_smem_layout_flat,
            self._sfb_smem_layout_padded,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, self.split_k),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        sfa: cute.Tensor,
        mC_mnl: cute.Tensor,
        x_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        q_ptr: cute.Pointer,
        sf_ptr: cute.Pointer,
        x_stride: Int32,
        rows: Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_flat_staged: cute.Layout,
        sfb_smem_layout_padded_staged: cute.Layout,
    ):
        """FlashInfer's kernel: 0 = DMA_A, 2 = MMA, 8-11 = epilogue; warps 3-6 make B
        (``norm_producer``); warp 1 idles."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_sfa)

        _, _, split_rank = cute.arch.block_in_cluster_idx()
        _, _, l_idx = cute.arch.cluster_idx()

        total_k_tiles = cute.ceil_div(cute.size(mA_mkl, mode=[1]), self.cta_k)
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        k_tiles_per_split = cute.ceil_div(total_k_tiles, self.split_k)
        k_start = split_rank * k_tiles_per_split
        if k_start > total_k_tiles:
            k_start = total_k_tiles
        k_end = k_start + k_tiles_per_split
        if k_end > total_k_tiles:
            k_end = total_k_tiles
        work_tile_info = WorkTileInfo(
            M_idx=bidx // cute.size(tiled_mma.thr_id.shape),
            N_idx=bidy,
            L_idx=l_idx,
            K_idx_start=k_start,
            K_idx_end=k_end,
        )
        k_tile_count = work_tile_info.K_idx_end - work_tile_info.K_idx_start

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        DMA_Stage = self.num_ab_stage
        SFB_Stage = self.num_sfb_tmem_stage
        MailboxElems = self._mailbox_total_elems

        @cute.struct
        class SharedStorage:
            ab_pipeline_bars: cute.struct.MemRange[cutlass.Int64, DMA_Stage * 2]
            sfb_pipeline_bars: cute.struct.MemRange[cutlass.Int64, SFB_Stage * 2]
            mma_epilog_bars: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_allocation_result_barrier: cutlass.Int64
            tma_epilog_full_barrier: cutlass.Int64
            tmem_base_ptr: cutlass.Int32
            dsmem_mailbox_barrier: cutlass.Int64
            dsmem_mailbox: cute.struct.MemRange[cutlass.Float32, MailboxElems]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # producers: the A-load warp (TMA bytes) and the norm producers (one arrive)
        ab_pipeline = PipelineAsync.create(
            barrier_storage=storage.ab_pipeline_bars.data_ptr(),
            num_stages=DMA_Stage,
            producer_group=CooperativeGroup(Agent.Thread, 2),
            consumer_group=CooperativeGroup(Agent.Thread),
            defer_sync=True,
        )
        ab_full_bar = ab_pipeline.sync_object_full.barrier_storage
        ab_empty_bar = ab_pipeline.sync_object_empty.barrier_storage

        mma_epilog_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.mma_epilog_bars.data_ptr(),
            num_stages=1,
            producer_group=CooperativeGroup(Agent.Thread),
            consumer_group=CooperativeGroup(Agent.Thread, 128),
            defer_sync=True,
        )
        mma_epilog_full_bar = mma_epilog_pipeline.sync_object_full.barrier_storage

        tmem_alloc_result_bar = storage.tmem_allocation_result_barrier.ptr
        tma_epilog_full_bar = storage.tma_epilog_full_barrier.ptr
        tmem_base_smem_ptr = storage.tmem_base_ptr.ptr
        mailbox_mbar = storage.dsmem_mailbox_barrier.ptr
        dsmem_mailbox_ptr = storage.dsmem_mailbox.data_ptr()
        sfb_full_bar = storage.sfb_pipeline_bars.data_ptr()
        sfb_empty_bar = sfb_full_bar + SFB_Stage

        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(tmem_alloc_result_bar, 32 + 128)
                if self.split_k > 1:
                    cute.arch.mbarrier_init(mailbox_mbar, 1)

        cluster_layout = (
            cute.make_layout((*self.cluster_shape_mn, self.split_k))
            if self.split_k > 1
            else None
        )
        pipeline_init_arrive(cluster_shape_mn=cluster_layout, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=cluster_layout)

        sA = smem.allocate_tensor(
            element_type=self.smem_alloc_a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.smem_alloc_b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sSFA = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfa_smem_layout_staged,
            byte_alignment=128,
        )
        sSFB_flat = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfb_smem_layout_flat_staged,
            byte_alignment=128,
        )
        sSFB_padded = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfb_smem_layout_padded_staged,
            byte_alignment=128,
        )

        if warp_idx == 0:
            self.dma_a_warp(
                ab_full_bar,
                ab_empty_bar,
                tma_atom_a,
                mA_mkl,
                sA,
                tma_atom_sfa,
                sfa,
                sSFA,
                tiled_mma,
                cluster_layout_vmnk,
                block_in_cluster_coord_vmnk,
                mma_tile_coord_v,
                work_tile_info,
                k_tile_count,
            )
        elif warp_idx == 2:
            self.mma_warp(
                ab_full_bar,
                ab_empty_bar,
                mma_epilog_full_bar,
                tmem_alloc_result_bar,
                sfb_full_bar,
                sfb_empty_bar,
                tiled_mma,
                sA,
                sB,
                sSFA,
                sSFB_padded,
                sfa_smem_layout_staged,
                sfb_smem_layout_padded_staged,
                tmem_base_smem_ptr,
                k_tile_count,
            )
        elif warp_idx >= 8:
            # the epilogue warps make the B operand's rows 4-7 before their epilogue
            self.norm_producer(
                ab_full_bar,
                ab_empty_bar,
                sB,
                sSFB_padded,
                x_ptr,
                w_ptr,
                q_ptr,
                sf_ptr,
                x_stride,
                rows,
                work_tile_info,
                tidx - 256 + 128,
            )
            epi_tid = tidx - 256
            self.epilog_warp(
                tma_epilog_full_bar,
                mma_epilog_full_bar,
                tmem_alloc_result_bar,
                tiled_mma,
                mC_mnl,
                None,
                None,
                tmem_base_smem_ptr,
                mailbox_mbar,
                dsmem_mailbox_ptr,
                epi_tid,
                mma_tile_coord_v,
                work_tile_info,
                split_rank,
                k_tile_count,
            )
        elif (warp_idx >= PRODUCER_WARP0) & (warp_idx < PRODUCER_WARP0 + 4):
            self.norm_producer(
                ab_full_bar,
                ab_empty_bar,
                sB,
                sSFB_padded,
                x_ptr,
                w_ptr,
                q_ptr,
                sf_ptr,
                x_stride,
                rows,
                work_tile_info,
                tidx - PRODUCER_WARP0 * 32,
            )

        cute.arch.barrier()
        if cutlass.const_expr(self.stamps != 0):
            if tidx == 0:
                self.stamp(work_tile_info.M_idx * self.split_k + split_rank, 4)
        return

    @cute.jit
    def norm_producer(
        self,
        ab_full_bar,
        ab_empty_bar,
        sB: cute.Tensor,
        sSFB: cute.Tensor,
        x_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        q_ptr: cute.Pointer,
        sf_ptr: cute.Pointer,
        x_stride: Int32,
        rows: Int32,
        work_tile_info: WorkTileInfo,
        ptid: Int32,
    ):
        """One warp a row (``ptid // 32``), lane ``t`` holding chunks ``32 j + t``: the
        Triton norm's own thread mapping, so its butterflies are this warp's. Every
        chunk's codes are computed first (independent chains), then this CTA's K tiles
        are stored into the B / SFB stages, all at once when each has a stage."""
        row = ptid // WARP_CHUNKS
        t = ptid % WARP_CHUNKS
        nj = self.thread_chunks
        stages = self.num_ab_stage
        tile_bytes = MAX_ROWS * self.cta_k
        sf_cols = self.cta_k // 32
        sf_stage_bytes = 128 * sf_cols
        b_base = cute.recast_ptr(sB.iterator, dtype=cutlass.Uint8).toint()
        sf_base = cute.recast_ptr(sSFB.iterator, dtype=cutlass.Uint8).toint()
        x_row = x_ptr.toint() + Int64(row) * Int64(x_stride) * 2
        w_addr = w_ptr.toint()
        valid = Int32(row < rows)
        k_start, k_end = work_tile_info.K_idx_start, work_tile_info.K_idx_end
        cta = work_tile_info.M_idx * self.split_k + k_start // self.split_tiles
        if ptid == 0:
            self.stamp(cta, 0)

        # the scale rows past the tile's 8 (the padded 128-row atom): zero, once
        if (ptid >= MAX_ROWS) & (ptid < 128):
            for st in cutlass.range_constexpr(stages):
                for blk in cutlass.range_constexpr(sf_cols // 4):
                    _st_shared_u32(
                        sf_base + st * sf_stage_bytes + _sf_offset(ptid, blk * 4),
                        Uint32(0),
                    )
        # the norm weight is static: read before the grid dependency wait
        wv = []
        for j in cutlass.range_constexpr(nj):
            wv.append(
                _ld_global_v4(w_addr + Int64((WARP_CHUNKS * j + t) * 16), Int32(1))
            )

        cute.arch.griddepcontrol_wait()
        if ptid == 0:
            self.stamp(cta, 1)

        xv = []
        for j in cutlass.range_constexpr(nj):
            xv.append(_ld_global_v4(x_row + Int64((WARP_CHUNKS * j + t) * 16), valid))
        warp_sum = []
        for j in cutlass.range_constexpr(nj):
            v = _chunk_square_sum(*xv[j])
            for m in cutlass.range_constexpr(5):
                v = _add_rn(v, cute.arch.shuffle_sync_bfly(v, offset=16 >> m))
            warp_sum.append(v)
        inv = _inv_rms(self.cross_warp_sum(warp_sum), self.k, self.eps_bits)
        if ptid == 0:
            if inv != Float32(-1.0):
                self.stamp(cta, 2)

        keep = Uint32(0) - Uint32(valid)
        codes = []
        for j in cutlass.range_constexpr(nj):
            codes.append(self.quantize_chunk(inv, xv[j], wv[j], keep))

        m0 = work_tile_info.M_idx == 0
        if cutlass.const_expr(self.split_tiles <= stages):
            # every K tile of this CTA has a stage of its own (its first use: no empty
            # wait): store them all, then one fence and one barrier release the MMA
            for j in cutlass.range_constexpr(nj):
                e = (WARP_CHUNKS * j + t) * CHUNK
                gt = e // self.cta_k
                if (gt >= k_start) & (gt < k_end):
                    self.store_chunk(
                        codes[j],
                        row,
                        t,
                        e,
                        gt - k_start,
                        b_base,
                        sf_base,
                        m0,
                        valid,
                        q_ptr,
                        sf_ptr,
                    )
            self.zero_padding_rows(ptid, rows, k_start, k_end, m0, sf_ptr)
            _fence_async_shared()
            cute.arch.barrier(
                barrier_id=PRODUCER_BARRIER, number_of_threads=PRODUCER_THREADS
            )
            if ptid == 0:
                for local in cutlass.range(k_end - k_start, unroll=1):
                    cute.arch.mbarrier_arrive(ab_full_bar + local)
                self.stamp(cta, 3)
        else:
            self.zero_padding_rows(ptid, rows, k_start, k_end, m0, sf_ptr)
            for tile in cutlass.range_constexpr(self.k_tiles):
                if (tile >= k_start) & (tile < k_end):
                    local = tile - k_start
                    stage = local % stages
                    cute.arch.mbarrier_wait(
                        ab_empty_bar + stage, ((local // stages) & 1) ^ 1
                    )
                    for j in cutlass.range_constexpr(nj):
                        e = (WARP_CHUNKS * j + t) * CHUNK
                        if e // self.cta_k == tile:
                            self.store_chunk(
                                codes[j],
                                row,
                                t,
                                e,
                                stage,
                                b_base,
                                sf_base,
                                m0,
                                valid,
                                q_ptr,
                                sf_ptr,
                            )
                    _fence_async_shared()
                    cute.arch.barrier(
                        barrier_id=PRODUCER_BARRIER, number_of_threads=PRODUCER_THREADS
                    )
                    if ptid == 0:
                        cute.arch.mbarrier_arrive(ab_full_bar + stage)

    @cute.jit
    def store_chunk(
        self, code, row, t, e, stage, b_base, sf_base, m0, valid, q_ptr, sf_ptr
    ):
        """A chunk's 8 codes into its B stage (128-byte swizzle, 1024-byte K atoms) and,
        from the first lane of its 32-element group, its scale byte into the SFB stage;
        with ``write_quant`` the first M tile also stores both to global."""
        lo, hi, sf = code
        kk = e % self.cta_k
        address = b_base + stage * (MAX_ROWS * self.cta_k) + (kk // 128) * 1024
        _st_shared_v2(_swizzle128(address + row * 128 + kk % 128), lo, hi)
        first = t % 4 == 0
        if first:
            _st_shared_u8(
                sf_base
                + stage * (128 * (self.cta_k // 32))
                + _sf_offset(row, kk // 32),
                sf,
            )
        if cutlass.const_expr(self.write_quant):
            if m0 & (valid != 0):
                _st_global_v2(q_ptr.toint() + Int64(row * self.k + e), lo, hi)
                if first:
                    _st_global_u8(sf_ptr.toint() + Int64(_sf_offset(row, e // 32)), sf)

    @cute.jit
    def zero_padding_rows(self, ptid, rows, k_start, k_end, m0, sf_ptr):
        """``write_quant``: the 128x4 scale rows past ``rows`` FlashInfer's quantize
        zeroes, this CTA's columns (a thread a row)."""
        if cutlass.const_expr(self.write_quant):
            sf_cols = self.cta_k // 32
            if m0 & (ptid >= rows) & (ptid < 128):
                for tile in cutlass.range_constexpr(self.k_tiles):
                    if (tile >= k_start) & (tile < k_end):
                        for col in cutlass.range_constexpr(sf_cols):
                            _st_global_u8(
                                sf_ptr.toint()
                                + Int64(_sf_offset(ptid, tile * sf_cols + col)),
                                Uint32(0),
                            )

    @cute.jit
    def stamp(self, cta: Int32, slot: int):
        if cutlass.const_expr(self.stamps != 0):
            _st_global_u64(
                Int64(self.stamps) + Int64(cta * 64 + slot * 8), global_timer_ns()
            )

    @staticmethod
    def cross_warp_sum(warp_sum):
        """The Triton second stage over 8 warp sums (missing warps are +0, an exact
        no-op on a sum of squares): ``((w0+w4) + (w2+w6)) + ((w1+w5) + (w3+w7))``."""
        padded = list(warp_sum) + [None] * (8 - len(warp_sum))

        def add(a, b):
            if a is None:
                return b
            if b is None:
                return a
            return _add_rn(a, b)

        return add(
            add(add(padded[0], padded[4]), add(padded[2], padded[6])),
            add(add(padded[1], padded[5]), add(padded[3], padded[7])),
        )

    @cute.jit
    def quantize_chunk(self, inv, xw, ww, keep):
        """A chunk's 8 BF16 norms, its 32-element group's amax (4 lanes), and its
        codes and scale byte (zero for a row past ``rows``): ``(lo, hi, sf)``."""
        y0 = _norm_bf16x2(inv, xw[0], ww[0])
        y1 = _norm_bf16x2(inv, xw[1], ww[1])
        y2 = _norm_bf16x2(inv, xw[2], ww[2])
        y3 = _norm_bf16x2(inv, xw[3], ww[3])
        amax = _amax_bf16x8(y0, y1, y2, y3)
        amax = _max_f32(amax, cute.arch.shuffle_sync_bfly(amax, offset=1))
        amax = _max_f32(amax, cute.arch.shuffle_sync_bfly(amax, offset=2))
        sf, scale_bits = _ue8m0(amax)
        lo = _e4m3x4(y0, y1, scale_bits)
        hi = _e4m3x4(y2, y3, scale_bits)
        return lo & keep, hi & keep, sf & keep


# ---------------------------------------------------------------------- host side


class QuantRows(NamedTuple):
    """``rmsnorm_mxfp8``'s ``(q, s)`` as the fused kernel writes them (1-8 rows):
    e4m3 [rows, K] and the 128x4 UE8M0 scales of one 128-row atom."""

    quant: torch.Tensor
    scale: torch.Tensor


_COMPILED: dict = {}


def _ptr(dtype, address: int, align: int):
    from cutlass.cute.runtime import make_ptr

    return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)


def _compiled(
    tactic, device: torch.device, k: int, eps: float, write_quant: bool, stamps: int = 0
):
    key = (device.index, tuple(tactic), k, f32_bits(eps), write_quant, stamps)
    if key not in _COMPILED:
        from cutlass.cute.runtime import make_fake_stream

        cta_k, ab_stages, sfb_tmem_stages, split_k = tactic
        gemm = QNormLowLatencyGemmKernel(
            eps=eps,
            k=k,
            write_quant=write_quant,
            stamps=stamps,
            acc_dtype=cutlass.Float32,
            mma_tiler_mnk=(128, 8, cta_k),
            num_ab_stage=ab_stages,
            num_sfb_tmem_stage=sfb_tmem_stages,
            sf_vec_size=32,
            use_pdl=True,
            split_k=split_k,
        )
        e4m3, e8m0, bf16 = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, cutlass.BFloat16
        with torch.accelerator.device_index(device.index):
            _COMPILED[key] = cute.compile(
                gemm,
                _ptr(e4m3, 16, 16),
                _ptr(e8m0, 32, 32),
                _ptr(bf16, 16, 16),
                _ptr(bf16, 16, 16),
                _ptr(bf16, 16, 16),
                _ptr(e4m3, 16, 16),
                _ptr(cutlass.Uint8, 16, 16),
                tuple(Int32(v) for v in (128, 8, k, 1)),
                Int32(k),
                make_fake_stream(),
            )
    return _COMPILED[key]


def tactic_for(weight: torch.Tensor, rows: int, consumer: Optional[str] = None):
    """The unfused GEMM's tactic for this weight and row count (``mxfp8_ll``): the
    fused kernel keeps its K split and reduction order."""
    from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll

    n, k = weight.shape
    major, minor = torch.cuda.get_device_capability(weight.device)
    return tuple(
        mxfp8_ll.select_tactic(
            major * 10 + minor, n, k, mxfp8_ll.row_bucket(rows), consumer
        )
    )


def reason(weight: torch.Tensor, rows: int, consumer: Optional[str] = None) -> str:
    """Why the fused kernel cannot run this GEMM at ``rows``, or ""."""
    n, k = weight.shape
    if not 0 < rows <= MAX_ROWS:
        return f"{rows} rows (1-{MAX_ROWS})"
    if k % 256 or not TRITON_BLOCK // 2 < k <= TRITON_BLOCK:
        return f"K {k}"
    cta_k = tactic_for(weight, rows, consumer)[0]
    if cta_k not in (128, 256):
        return f"cta_k {cta_k}"
    return ""


def precompile(
    weight: torch.Tensor,
    rows: list[int],
    eps: float,
    *,
    consumer: Optional[str] = None,
    write_quant: bool = False,
) -> None:
    """Compile the kernels these row counts select; before CUDA graph capture."""
    k = weight.shape[1]
    for r in rows:
        if not reason(weight, r, consumer):
            _compiled(
                tactic_for(weight, r, consumer), weight.device, k, eps, write_quant
            )


def q_norm_linear(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    consumer: Optional[str] = None,
    write_quant: bool = False,
    stamps: Optional[torch.Tensor] = None,
):
    """``mxfp8_ll.linear`` on ``rmsnorm_mxfp8(x, norm_weight, eps)``'s quant, one
    launch (PDL): BF16 [rows, N]; with ``write_quant`` also that quant (``QuantRows``).
    ``x``: BF16 [rows, K] with unit column stride (a view of wqkv_a's output)."""
    from cutlass import BFloat16, Float8E4M3FN, Float8E8M0FNU, Uint8

    from sgl_nvidia_mega.common.sync import current_cu_stream

    rows, k = x.shape
    n = weight.shape[0]
    why = reason(weight, rows, consumer)
    assert not why, why
    assert weight.shape[1] == k and weight.is_contiguous()
    assert x.dtype == norm_weight.dtype == torch.bfloat16 and x.stride(1) == 1
    assert x.data_ptr() % 16 == 0 and (x.stride(0) * 2) % 16 == 0
    assert norm_weight.is_contiguous() and norm_weight.numel() == k
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=x.device)
    quant = torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((k // 32) * 128, dtype=torch.uint8, device=x.device)
    tactic = tactic_for(weight, rows, consumer)
    _compiled(
        tactic,
        x.device,
        k,
        eps,
        write_quant,
        0 if stamps is None else stamps.data_ptr(),
    )(
        _ptr(Float8E4M3FN, weight.data_ptr(), 16),
        _ptr(Float8E8M0FNU, weight_scale.data_ptr(), 32),
        _ptr(BFloat16, x.data_ptr(), 16),
        _ptr(BFloat16, norm_weight.data_ptr(), 16),
        _ptr(BFloat16, out.data_ptr(), 16),
        _ptr(Float8E4M3FN, quant.data_ptr(), 16),
        _ptr(Uint8, scale.data_ptr(), 16),
        (Int32(n), Int32(rows), Int32(k), Int32(1)),
        Int32(x.stride(0)),
        current_cu_stream(),
    )
    if write_quant:
        return out, QuantRows(quant, scale)
    return out
