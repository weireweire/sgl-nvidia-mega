# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The mHC coefficients of up to 48 rows in one launch (slot ``hc_coeff_fused``).

``src/hc_coeff.cu``: DeepGEMM's TF32 prenorm over 16 K slices, one CTA per slice,
and the CTA that finishes last runs SGLang's reduce + Sinkhorn over the partials, so
the pair ``tf32_hc_prenorm_gemm`` + ``_hc_mix_reduce_sinkhorn_kernel`` becomes one
kernel with the same bits. The finish tickets are a ring allocated once; the kernel
resets the one it takes.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import torch

from sgl_nvidia_mega.common.jit import cache_dir, check, driver, launch, load

SOURCE = Path(__file__).resolve().parent / "src" / "hc_coeff.cu"
BLOCK_M, BLOCK_N, BLOCK_K = 64, 32, 64
THREADS = 256
# DeepGEMM's pick for this shape (prenorm.SLICES slices, 12 stages)
STAGES = 12
# Calls in flight before a ticket comes round again (one stream orders them anyway)
TICKETS = 64
# DeepGEMM's own nvcc build (its headers include the C++ standard library, which
# NVRTC does not ship)
NVCC_OPTIONS = (
    "-std=c++20",
    "-O3",
    "--expt-relaxed-constexpr",
    "--diag-suppress=39,161,174,177,186,940",
)


def _deep_gemm_include() -> str:
    import deep_gemm

    root = Path(deep_gemm.__file__).resolve().parent / "include"
    if not (root / "deep_gemm" / "impls" / "sm100_tf32_hc_prenorm_gemm.cuh").is_file():
        raise RuntimeError(f"DeepGEMM headers not found under {root}")
    return str(root)


def _float_literal(value: float) -> str:
    """``value`` rounded to FP32 as a C++ hex literal (as Triton rounds a constexpr)."""
    (f32,) = struct.unpack("<f", struct.pack("<f", value))
    return f"{f32.hex()}f"


def compile_nvcc(source: Path, arch: str, options: tuple[str, ...]) -> bytes:
    """A cubin of ``source`` from nvcc, cached on disk by source, headers' root,
    options and nvcc version; refuses CUDA-graph capture as ``jit.compile_source``."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("compiling hc_coeff inside CUDA-graph capture")
    nvcc = _nvcc()
    version = subprocess.run(
        [nvcc, "--version"], capture_output=True, text=True, check=True
    ).stdout
    key = hashlib.sha256(
        "\0".join([source.read_text(), arch, *options, version]).encode()
    ).hexdigest()
    cached = cache_dir() / f"hc_coeff-{key[:24]}.cubin"
    if cached.is_file():
        return cached.read_bytes()
    cached.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cached.parent) as tmp:
        out = Path(tmp) / "hc_coeff.cubin"
        done = subprocess.run(
            [nvcc, "-cubin", f"-arch={arch}", *options, "-o", str(out), str(source)],
            capture_output=True,
            text=True,
        )
        if done.returncode:
            raise RuntimeError(f"nvcc failed on {source.name}: {done.stderr[-4000:]}")
        out.replace(cached)
    return cached.read_bytes()


def _nvcc() -> str:
    for root in (
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        "/usr/local/cuda",
    ):
        if root and (Path(root) / "bin" / "nvcc").is_file():
            return str(Path(root) / "bin" / "nvcc")
    found = shutil.which("nvcc")
    if not found:
        raise RuntimeError("hc_coeff needs nvcc (DeepGEMM's JIT needs it too)")
    return found


def smem_bytes(stages: int = STAGES) -> int:
    stage = BLOCK_M * BLOCK_K * 2 + BLOCK_N * BLOCK_K * 4
    return stages * stage + (4 * stages + 1) * 8 + 8


class HcCoeff:
    """``__call__(x_flat, fn, scale, base) -> (pre, post, comb)`` for ``x_flat``
    [rows <= max_rows, hc * hidden] BF16 and ``fn`` [(2 + hc) * hc, hc * hidden] FP32,
    on the current stream; the outputs are fresh tensors as SGLang's."""

    def __init__(
        self,
        *,
        hc: int,
        k: int,
        slices: int,
        iters: int,
        eps: float,
        rms_eps: float,
        max_rows: int,
        device,
    ) -> None:
        if max_rows > BLOCK_M:
            raise ValueError(f"at most {BLOCK_M} rows")
        self.hc, self.k, self.slices, self.max_rows = hc, k, slices, max_rows
        self.mix = (2 + hc) * hc
        if self.mix > BLOCK_N or k % BLOCK_K:
            raise ValueError(f"unsupported mix {self.mix} or K {k}")
        self.inv_k = ctypes.c_float(1.0 / k)
        self.rms_eps = ctypes.c_float(rms_eps)
        device = torch.device(device)
        with torch.cuda.device(device):
            major, minor = torch.cuda.get_device_capability(device)
            options = NVCC_OPTIONS + (
                f"-I{_deep_gemm_include()}",
                f"-DHC_EPS={_float_literal(eps)}",
                f"-DHC_HC={hc}",
                f"-DHC_K={k}",
                f"-DHC_SPLITS={slices}",
                f"-DHC_STAGES={STAGES}",
                f"-DHC_ITERS={iters}",
            )
            cubin = compile_nvcc(SOURCE, f"sm_{major}{minor}a", options)
            self._kernel = load(cubin, "dsv41_hc_coeff", (THREADS, 1, 1), smem_bytes())
            self._tickets = torch.zeros(TICKETS, dtype=torch.int32, device=device)
        self._next = 0
        self._fn_maps: dict[int, object] = {}

    def __call__(self, x_flat: torch.Tensor, fn, scale, base):
        rows, k = x_flat.shape
        if not 0 < rows <= self.max_rows or k != self.k:
            raise ValueError(f"x_flat {tuple(x_flat.shape)} outside the build")
        hc, device = self.hc, x_flat.device
        pre = torch.empty((rows, hc), dtype=torch.float32, device=device)
        post = torch.empty_like(pre)
        comb = torch.empty((rows, hc, hc), dtype=torch.float32, device=device)
        part_mix = torch.empty(
            (self.slices, rows, self.mix), dtype=torch.float32, device=device
        )
        part_sq = torch.empty((self.slices, rows), dtype=torch.float32, device=device)
        ticket = self._tickets[self._next]
        self._next = (self._next + 1) % TICKETS
        launch(
            self._kernel,
            (self.slices, 1, 1),
            [
                ctypes.c_uint32(rows),
                _map(x_flat.data_ptr(), k, rows, k * 2, BLOCK_K, BLOCK_M, bf16=True),
                self._fn_map(fn),
                ctypes.c_void_p(part_mix.data_ptr()),
                ctypes.c_void_p(part_sq.data_ptr()),
                ctypes.c_void_p(scale.data_ptr()),
                ctypes.c_void_p(base.data_ptr()),
                ctypes.c_void_p(pre.data_ptr()),
                ctypes.c_void_p(post.data_ptr()),
                ctypes.c_void_p(comb.data_ptr()),
                ctypes.c_void_p(ticket.data_ptr()),
                self.inv_k,
                self.rms_eps,
            ],
            torch.cuda.current_stream().cuda_stream,
        )
        return pre, post, comb

    def _fn_map(self, fn: torch.Tensor):
        """A layer's weight never moves: one map per address."""
        if fn.shape != (self.mix, self.k) or fn.dtype != torch.float32:
            raise ValueError(f"fn {tuple(fn.shape)} {fn.dtype} outside the build")
        tensor_map = self._fn_maps.get(fn.data_ptr())
        if tensor_map is None:
            tensor_map = _map(
                fn.data_ptr(), self.k, self.mix, self.k * 4, 32, BLOCK_N, bf16=False
            )
            self._fn_maps[fn.data_ptr()] = tensor_map
        return tensor_map


def _map(base: int, cols: int, rows: int, stride: int, box_cols, box_rows, *, bf16):
    """A 128B-swizzled TMA map of a row-major [rows, cols] matrix; rows past ``rows``
    read as zero. As DeepGEMM's: the FP32 weight loads as TF32 (TMA rounds it to
    nearest; a FLOAT32 map would leave the MMA to truncate it)."""
    cu = driver()
    u64, u32 = cu.cuuint64_t, cu.cuuint32_t
    dtype = cu.CUtensorMapDataType
    (tensor_map,) = check(
        cu.cuTensorMapEncodeTiled(
            dtype.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
            if bf16
            else dtype.CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
            2,
            base,
            [u64(cols), u64(rows)],
            [u64(stride)],
            [u32(box_cols), u32(box_rows)],
            [u32(1)] * 2,
            cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
            cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        ),
        "cuTensorMapEncodeTiled(hc_coeff)",
    )
    return tensor_map
