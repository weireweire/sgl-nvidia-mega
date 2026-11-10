# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang's python/sglang/kernels/ops/attention/dsv4/wo_a.py and the host side of
# python/sglang/kernels/jit/csrc/deepseek_v4/wo_a_fused.cuh (Apache-2.0).
"""Fused inverse RoPE + grouped wo_a BF16 GEMM + MXFP8 quantization at 1-48 rows.

src/wo_a_fused.cu is SGLang's ``wo_a_fused.cuh`` device code built with NVRTC, one
variant per (row tile, position dtype, early trigger, row-tile scales); this module is
its host side: the two TMA maps, the output allocation and a driver launch of 16 per
local group clusters of 8 CTAs (the cluster shape is compiled in) with PDL. G local
groups: 2 at attention TP4, 1 at TP8. Outputs are bit for bit SGLang's
``fused_rope_wo_a_bf16(..., out_mxfp8=True)`` (whose kernel takes 2 or 4 groups; each
group's columns are computed alone, so 1 group equals the first group of 2): e4m3 codes
``[T, G * 1024]`` and UE8M0 scales ``[G * 4096]`` in FlashInfer's 128x4 swizzle (rows
padded to 128, written zero), or ``[ceil(T / 8) * G * 4096]`` in ``q_norm_quant``'s
row-tile layout with ``row_tile_scales``. Variants compile at build
(:func:`precompile`), never under CUDA-graph capture.
"""

from __future__ import annotations

import ctypes
from collections import OrderedDict
from functools import cache
from pathlib import Path

import torch

from sgl_nvidia_mega.common.jit import Kernel as _Kernel
from sgl_nvidia_mega.common.jit import check as _check
from sgl_nvidia_mega.common.jit import compile_source as _compile_source
from sgl_nvidia_mega.common.jit import driver as _driver
from sgl_nvidia_mega.common.jit import launch as _launch
from sgl_nvidia_mega.common.jit import load as _load
from sgl_nvidia_mega.common.jit import pointer as _pointer

SOURCE = Path(__file__).resolve().parent / "src" / "wo_a_fused.cu"
SYMBOL = "dsv41_wo_a_fused"

GROUPS = 2  # TP4's local groups, the default
GROUP_COUNTS = (1, 2)  # TP8, TP4
RANK = 1024
N_OUT = GROUPS * RANK
GROUP_K = 4096
ROPE_DIM = 64
SCALE_BYTES = 8192  # one UE8M0 byte per 32-column block, for 128 padded rows (2 groups)
MAX_ROWS = 48
TILE_ROWS = (16, 32, 48)
N_TILE = 64
BLOCK_K = 64
SPLIT_K = 8  # CTAs per cluster, __cluster_dims__ in the kernel
GRID = N_OUT // N_TILE * SPLIT_K


def scale_bytes(groups: int) -> int:
    """UE8M0 bytes of 128 padded rows (one 8-row tile with row-tile scales)."""
    return groups * RANK // 32 * 128


def grid(groups: int) -> int:
    return groups * RANK // N_TILE * SPLIT_K


THREADS = 256
# sizeof(WoATrait<tile>::Smem); the kernel static_asserts the value it is given.
SHARED_BYTES = {16: 87040, 32: 107520, 48: 115712}
# cudaSharedmemCarveoutMaxShared: two CTAs per SM need the full carveout.
SHARED_CARVEOUT = 100
# SGLang builds the kernel with -std=c++20 -O3 --expt-relaxed-constexpr and no fast
# math. NVRTC has no relaxed-constexpr flag; -default-device makes the constexpr
# helpers device functions instead, and FP flags stay at the same defaults.
NVRTC_OPTIONS = ("-std=c++20", "-default-device")
# Arbitrary bound on cached activation maps; eager callers pass a new buffer each step.
X_MAP_CACHE_SIZE = 256

_weight_maps: dict[tuple[int, int], object] = {}
_x_maps: OrderedDict[tuple[int, int, int, int], object] = OrderedDict()


def tile_rows(rows: int) -> int:
    """The kernel row tile (MMA N) serving ``rows`` rows."""
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"fused_rope_wo_a serves 1-{MAX_ROWS} rows, got {rows}")
    return next(tile for tile in TILE_ROWS if rows <= tile)


def _variant_name(
    tile: int,
    int64_positions: bool,
    early_trigger: bool,
    row_tile_scales: bool,
    groups: int = GROUPS,
) -> str:
    return (
        f"wo_a_fused_m{tile}_{'i64' if int64_positions else 'i32'}"
        f"{'_et' if early_trigger else ''}{'_rt' if row_tile_scales else ''}"
        f"{'' if groups == GROUPS else f'_g{groups}'}"
    )


@cache
def load_kernel(
    device_index: int,
    tile: int,
    int64_positions: bool,
    early_trigger: bool,
    row_tile_scales: bool = False,
    groups: int = GROUPS,
) -> _Kernel:
    """Compile (or read from the cache) and load one variant on ``device_index``."""
    if tile not in SHARED_BYTES:
        raise ValueError(f"row tile {tile} is not one of {TILE_ROWS}")
    if groups not in GROUP_COUNTS:
        raise ValueError(f"{groups} local groups is not one of {GROUP_COUNTS}")
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        torch.zeros(1, device=device)  # make the primary context current
        major, minor = torch.cuda.get_device_capability(device)
        options = NVRTC_OPTIONS + (
            f"-DWO_A_MAX_TOKENS={tile}",
            f"-DWO_A_POS_INT64={int(int64_positions)}",
            f"-DWO_A_EARLY_TRIGGER={int(early_trigger)}",
            f"-DWO_A_ROW_TILE_SCALES={int(row_tile_scales)}",
            f"-DWO_A_SMEM_BYTES={SHARED_BYTES[tile]}",
            f"-DWO_A_GROUPS={groups}",
        )
        cubin = _compile_source(
            SOURCE.read_text(),
            _variant_name(
                tile, int64_positions, early_trigger, row_tile_scales, groups
            ),
            f"sm_{major}{minor}a",
            options,
        )
        kernel = _load(cubin, SYMBOL, (THREADS, 1, 1), SHARED_BYTES[tile])
        cu = _driver()
        _check(
            cu.cuFuncSetAttribute(
                kernel.function,
                cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                SHARED_CARVEOUT,
            ),
            "cuFuncSetAttribute(carveout)",
        )
        return kernel


def precompile(
    rows: int,
    *,
    device_index: int | None = None,
    early_trigger: bool = True,
    row_tile_scales: bool = False,
    groups: int = GROUPS,
) -> None:
    """Build every variant that serves 1..``rows`` rows, for int32 and int64 positions."""
    if device_index is None:
        device_index = torch.cuda.current_device()
    for tile in TILE_ROWS:
        if tile > tile_rows(rows):
            break
        for int64_positions in (False, True):
            load_kernel(
                device_index,
                tile,
                int64_positions,
                early_trigger,
                row_tile_scales,
                groups,
            )


def _bf16_map(base: int, cols: int, rows: int, row_stride_bytes: int, box_rows: int):
    """TMA map of a bf16 ``[rows, cols]`` matrix, 64-column boxes, 128B swizzle; rows
    past ``rows`` read as zero, which masks the tokens past T."""
    cu = _driver()
    u64, u32 = cu.cuuint64_t, cu.cuuint32_t
    (tensor_map,) = _check(
        cu.cuTensorMapEncodeTiled(
            cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            base,
            [u64(cols), u64(rows)],
            [u64(row_stride_bytes)],
            [u32(BLOCK_K), u32(box_rows)],
            [u32(1)] * 2,
            cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
            cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        ),
        "cuTensorMapEncodeTiled(wo_a)",
    )
    return tensor_map


def _weight_map(weight: torch.Tensor):
    """The weight map, cached by address: a layer's weight never moves."""
    base = weight.data_ptr()
    key = (base, weight.shape[0])
    tensor_map = _weight_maps.get(key)
    if tensor_map is None:
        tensor_map = _bf16_map(
            base, GROUP_K, weight.shape[0] * RANK, GROUP_K * 2, N_TILE
        )
        _weight_maps[key] = tensor_map
    return tensor_map


def _x_map(x: torch.Tensor, rows: int):
    groups = x.shape[1]
    key = (x.data_ptr(), rows, x.stride(0), groups)
    tensor_map = _x_maps.get(key)
    if tensor_map is None:
        tensor_map = _bf16_map(
            key[0], groups * GROUP_K, rows, x.stride(0) * 2, tile_rows(rows)
        )
        _x_maps[key] = tensor_map
        if len(_x_maps) > X_MAP_CACHE_SIZE:
            _x_maps.popitem(last=False)
    else:
        _x_maps.move_to_end(key)
    return tensor_map


def _check_inputs(x, weight, freqs, positions) -> None:
    rows = x.size(0) if x.dim() == 3 else -1
    groups = x.size(1) if x.dim() == 3 else -1
    if not (
        x.dtype == torch.bfloat16
        and x.dim() == 3
        and groups in GROUP_COUNTS
        and x.size(2) == GROUP_K
        and (groups == 1 or x.stride(1) == GROUP_K)
        and x.stride(2) == 1
        and x.stride(0) >= groups * GROUP_K
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
    ):
        raise ValueError(
            f"x must be bf16 [T, G in {GROUP_COUNTS}, {GROUP_K}] with unit K stride, "
            f"16-byte aligned rows; got {tuple(x.shape)} {x.dtype} strides {x.stride()}"
        )
    tile_rows(rows)
    if (
        weight.dtype != torch.bfloat16
        or tuple(weight.shape) != (groups, RANK, GROUP_K)
        or not weight.is_contiguous()
        or weight.data_ptr() % 16
    ):
        raise ValueError(
            f"weight must be a contiguous bf16 [{groups}, {RANK}, {GROUP_K}], got "
            f"{tuple(weight.shape)} {weight.dtype}"
        )
    if (
        freqs.dtype != torch.float32
        or freqs.dim() != 2
        or freqs.size(1) != ROPE_DIM
        or not freqs.is_contiguous()
    ):
        raise ValueError(f"freqs must be a contiguous fp32 [S, {ROPE_DIM}]")
    if (
        positions.dtype not in (torch.int32, torch.int64)
        or tuple(positions.shape) != (rows,)
        or not positions.is_contiguous()
    ):
        raise ValueError(f"positions must be a contiguous int32/int64 [{rows}]")
    if not (x.device == weight.device == freqs.device == positions.device) or (
        x.device.type != "cuda"
    ):
        raise ValueError("x, weight, freqs and positions must share one CUDA device")


def fused_rope_wo_a(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    positions: torch.Tensor,
    *,
    early_trigger: bool = True,
    row_tile_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse RoPE of each head's last 64 lanes, then ``x @ weight^T`` per group,
    quantized to MXFP8.

    ``x`` [T, G, 4096] bf16 (a strided view of the [T, heads, 512] attention output),
    G = 2 (TP4) or 1 (TP8), ``weight`` [G, 1024, 4096] bf16, ``freqs`` [S, 64] fp32
    (cos0, sin0, cos1, ...), ``positions`` [T] int32 or int64, 1 <= T <= 48.
    ``early_trigger`` lets the PDL dependent launch at this kernel's start. Returns
    (codes [T, G * 1024] e4m3, scales).
    """
    _check_inputs(x, weight, freqs, positions)
    rows, groups = x.size(0), x.size(1)
    device = x.device
    kernel = load_kernel(
        device.index,
        tile_rows(rows),
        positions.dtype == torch.int64,
        early_trigger,
        row_tile_scales,
        groups,
    )
    codes = torch.empty((rows, groups * RANK), dtype=torch.float8_e4m3fn, device=device)
    tile_bytes = scale_bytes(groups)
    nbytes = (rows + 7) // 8 * tile_bytes if row_tile_scales else tile_bytes
    scales = torch.empty(nbytes, dtype=torch.uint8, device=device)
    _launch(
        kernel,
        (grid(groups), 1, 1),
        [
            _weight_map(weight),
            _x_map(x, rows),
            _pointer(freqs),
            _pointer(positions),
            _pointer(codes),
            _pointer(scales),
            ctypes.c_void_p(0),  # the bf16 output, unused
            ctypes.c_uint32(rows),
        ],
        torch.cuda.current_stream(device).cuda_stream,
    )
    return codes, scales
