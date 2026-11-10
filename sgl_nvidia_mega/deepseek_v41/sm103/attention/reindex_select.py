# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The indexer's candidate consumer in one cluster launch: the logits of a row's
published blocks and their top-k (src/reindex_select.cu), in place of SGLang's
``sparse_logits`` + ``topk_transform_sparse``.

The logits are DeepGEMM's bit for bit. The selection is the stock top-k's on them: the
same elements above the k-th value and as many equal to it. The stock top-k's pick among
the equal elements and its output order follow warp arrival; this kernel writes the
outcome of in-order arrival (ascending columns), one of the stock kernel's own outputs.
Variants compile at build (:func:`precompile`), never under CUDA-graph capture.
"""

from __future__ import annotations

import ctypes
from collections import OrderedDict
from functools import cache
from pathlib import Path
from typing import Optional

import torch

from sgl_nvidia_mega.common.jit import Kernel as _Kernel
from sgl_nvidia_mega.common.jit import check as _check
from sgl_nvidia_mega.common.jit import compile_source as _compile_source
from sgl_nvidia_mega.common.jit import driver as _driver
from sgl_nvidia_mega.common.jit import launch as _launch
from sgl_nvidia_mega.common.jit import load as _load
from sgl_nvidia_mega.common.jit import null_or_pointer as _null_or_pointer
from sgl_nvidia_mega.common.jit import pointer as _pointer

SOURCE = Path(__file__).resolve().parent / "src" / "reindex_select.cu"
SYMBOL = "dsv41_reindex_select"

HEADS = 32
KEY_BYTES = 64  # 128 e2m1 values; the page's 4-byte scales follow its keys
BLOCK = 8  # keys per published block (SGLang's CANDIDATE_BLOCK_SIZE)
TILE = 128
TILES = 8  # per CTA: one gathering warp each
THREADS = 256
CLUSTER = 16  # CTAs per row (a non-portable cluster size)
WIDTH = CLUSTER * TILES * TILE  # 16384: DeepSeek-V4.1-Flash's 2048 candidate blocks
MAX_ROWS = 8  # rows * CLUSTER CTAs within one wave of 148 SMs
MAX_TOPK = 2048
NVRTC_OPTIONS = ("-std=c++17", "-default-device")
# Arbitrary bound on cached index-K maps (one per layer pool in serving)
MAP_CACHE_SIZE = 64

_k_maps: OrderedDict[tuple, object] = OrderedDict()


def shared_bytes() -> int:
    """At least sizeof(Smem) (the kernel static_asserts it), with room for its 1024-byte
    alignment."""
    cols = TILES * TILE
    size = cols * KEY_BYTES + HEADS * KEY_BYTES  # k, q
    size += cols * 4 + 128 * 4  # sf_k, sf_q
    size += cols * 2 + cols // BLOCK * 4  # logits, blocks
    size += (2 + 2 * CLUSTER) * 292 * 4  # hist_local, hist
    size += 2 * CLUSTER * 4 + 64 + (2 * TILES + 7) * 8 + 32
    return (size + 1023) // 1024 * 1024 + 1024


def reason(rows: int, width: int, topk: int, page_size: int, heads: int) -> str:
    """Why the fused kernel cannot take this consumer call, or ""."""
    if not 1 <= rows <= MAX_ROWS:
        return f"{rows} rows (the fused consumer takes 1-{MAX_ROWS})"
    if heads != HEADS:
        return f"{heads} index heads"
    if width != WIDTH:
        return f"{width} candidate columns (the fused consumer takes {WIDTH})"
    if not 0 < topk <= MAX_TOPK:
        return f"top-{topk}"
    if page_size % BLOCK or (page_size // BLOCK) & (page_size // BLOCK - 1):
        return f"index page size {page_size}"
    return ""


@cache
def load_kernel(device_index: int) -> _Kernel:
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        torch.zeros(1, device=device)  # make the primary context current
        major, minor = torch.cuda.get_device_capability(device)
        options = NVRTC_OPTIONS + (
            f"-DRS_CLUSTER={CLUSTER}",
            f"-DRS_TILES={TILES}",
            f"-DRS_SMEM_BYTES={shared_bytes()}",
        )
        cubin = _compile_source(
            SOURCE.read_text(),
            f"reindex_select_c{CLUSTER}_t{TILES}",
            f"sm_{major}{minor}a",
            options,
        )
        kernel = _load(cubin, SYMBOL, (THREADS, 1, 1), shared_bytes())
        cu = _driver()
        _check(
            cu.cuFuncSetAttribute(
                kernel.function,
                cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                1,
            ),
            "cuFuncSetAttribute(non-portable cluster)",
        )
        return kernel


def precompile(device_index: int) -> None:
    load_kernel(device_index)


def _k_map(k_cache: torch.Tensor):
    """TMA map of the index-K pool's keys: ``[pages, page_size, 64]`` bytes, one
    published block (8 keys) per box, 64B swizzle (the MMA's K-major operand)."""
    pages, page_size = k_cache.shape[0], k_cache.shape[1]
    key = (k_cache.data_ptr(), pages, page_size, k_cache.stride(0))
    tensor_map = _k_maps.get(key)
    if tensor_map is not None:
        _k_maps.move_to_end(key)
        return tensor_map
    cu = _driver()
    u64, u32 = cu.cuuint64_t, cu.cuuint32_t
    (tensor_map,) = _check(
        cu.cuTensorMapEncodeTiled(
            cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
            3,
            k_cache.data_ptr(),
            [u64(KEY_BYTES), u64(page_size), u64(pages)],
            [u64(KEY_BYTES), u64(k_cache.stride(0))],
            [u32(KEY_BYTES), u32(BLOCK), u32(1)],
            [u32(1)] * 3,
            cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
            cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        ),
        "cuTensorMapEncodeTiled(index K)",
    )
    _k_maps[key] = tensor_map
    if len(_k_maps) > MAP_CACHE_SIZE:
        _k_maps.popitem(last=False)
    return tensor_map


def reindex_select(
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    weights: torch.Tensor,
    k_cache: torch.Tensor,
    blocks: torch.Tensor,
    valid_lens: torch.Tensor,
    out: torch.Tensor,
    logits: Optional[torch.Tensor] = None,
    inputs_before_wait: bool = False,
    trigger_after_write: bool = True,
) -> None:
    """``topk_transform_sparse(sparse_logits(q_fp4, q_sf, k_cache, weights.bf16,
    schedule, n), valid_lens, blocks, out)`` with ``n = blocks.shape[1]``, on the
    current stream: ``q_fp4`` ``[rows, 1, 32, 64]`` int8, ``q_sf`` ``[rows, 1, 32]``
    int32, ``weights`` ``[rows, 32]`` fp32 or bf16, ``k_cache`` ``[pages, page_size,
    1, 68]`` uint8, ``blocks`` the published pool slots / 8 ``[rows, n]`` int32,
    ``valid_lens`` ``[rows]`` int32, ``out`` ``[rows, k]`` int32. ``logits``, when
    given (``[rows, >= 8 n]`` bf16), receives each row's valid logits.

    ``inputs_before_wait``: the kernel reads everything but the index K before its PDL
    wait; only when no kernel it may overlap writes those inputs. In a captured graph
    a stream join into this launch is a programmatic edge too, so an event wait does
    not order an input that its writer publishes after its own trigger.
    ``trigger_after_write``: the dependents launch once ``out`` is written and fenced, so
    one may read it before its own wait (else just before it is written)."""
    rows, width = blocks.shape[0], blocks.shape[1] * BLOCK
    page_size = k_cache.shape[1]
    why = reason(rows, width, out.shape[1], page_size, q_fp4.shape[2])
    if why:
        raise ValueError(f"reindex_select: {why}")
    assert q_fp4.is_contiguous() and q_sf.is_contiguous() and weights.is_contiguous()
    assert weights.dtype in (torch.float32, torch.bfloat16)
    assert blocks.dtype == torch.int32 and blocks.stride(1) == 1
    assert valid_lens.dtype == torch.int32 and valid_lens.is_contiguous()
    assert out.dtype == torch.int32 and out.stride(1) == 1
    assert k_cache.dtype == torch.uint8 and k_cache.stride(0) % 128 == 0
    kernel = load_kernel(q_fp4.device.index)
    args = [
        _k_map(k_cache),
        _pointer(k_cache),
        ctypes.c_int64(k_cache.stride(0)),
        ctypes.c_int32(page_size),
        ctypes.c_int32((page_size // BLOCK).bit_length() - 1),
        _pointer(q_fp4),
        _pointer(q_sf),
        _pointer(weights),
        ctypes.c_int32(int(weights.dtype == torch.bfloat16)),
        _pointer(blocks),
        ctypes.c_int64(blocks.stride(0)),
        _pointer(valid_lens),
        _pointer(out),
        ctypes.c_int64(out.stride(0)),
        ctypes.c_int32(out.shape[1]),
        _null_or_pointer(logits),
        ctypes.c_int64(0 if logits is None else logits.stride(0)),
        ctypes.c_int32(int(inputs_before_wait)),
        ctypes.c_int32(int(trigger_after_write)),
    ]
    _launch(
        kernel,
        (CLUSTER, rows, 1),
        args,
        torch.cuda.current_stream().cuda_stream,
        cluster=(CLUSTER, 1, 1),
    )
