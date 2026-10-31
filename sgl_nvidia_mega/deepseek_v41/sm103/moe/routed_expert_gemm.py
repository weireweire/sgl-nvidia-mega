# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Low-batch routed-expert GEMMs for DeepSeek-V4.1 decode.

A MoE layer at MoE TP4 (intermediate 576 padded to 640; all experts local, or under
expert parallelism the rank's consecutive window of them) replaces FlashInfer's
TRT-LLM-gen routing + two batched GEMMs by a route kernel and a GEMM pair of 8-row
tcgen05 block-scaled tiles (persistent schedule, PDL), compiled with NVRTC on first use. An expert chosen by more than 8 rows gets several
consecutive tiles (route_wide.cu). The kernels read the same shuffled MXFP4 weights and
MXFP8 activations as FlashInfer and return its deferred-finalize triple, so the fused
finalize + all-reduce + mHC consumer is unchanged. On the target and DSpark draft
routers, gate_topk.cu / gate_topk_draft.cu run the router's top-k ahead of the route
kernel.
"""

from __future__ import annotations

import ctypes
import json
import os
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

SOURCE_DIR = Path(__file__).resolve().parent / "src"

HIDDEN = 5120
SLICE = 576
PADDED = 640
MAX_EXPERTS = 384
MAX_TOP_K = 8
TILE = 8
MAX_ROWS = 8
WIDE_MAX_ROWS = 64  # route_wide.cu: one 64-bit row mask per expert
GEMM1_UNITS = 9  # 128-row tiles of w13 holding the 576-wide slice (10th is padding)
GEMM2_UNITS = HIDDEN // 128
ROUTE_THREADS = 64
ROUTE_WIDE_THREADS = 512
NVRTC_OPTIONS = ("-std=c++17", "-default-device", "--use_fast_math")
HEADERS = ("route.cuh",)
GEMM1_PARAMETERS = ("A", "SFA", "X", "XS", "route", "clamp", "H", "HS", "cta_words")
GEMM2_PARAMETERS = ("A", "SFA", "H", "HS", "route", "Y", "cta_words")
PARAMETERS = {"gemm1": GEMM1_PARAMETERS, "gemm2": GEMM2_PARAMETERS}
# SGL_NVIDIA_MEGA_MOE_STRIDED_GEMM=1: the GEMM pair on a strided 1-D grid of
# min(SMs - STRIDED_SIDE_SMS, tiles) CTAs instead of cluster launch control over every
# SM, so the shared expert and hC kernels on the side streams keep SMs (the reference
# tools/routed_expert_gemm_strided.py builds of these same kernels). The 384-expert
# top-6 GEMM1 whose largest routing can need a partial second round strides over a
# balanced CTA count instead. Every tile stays whole on one CTA: bitwise either way.
STRIDED_GEMM = os.environ.get("SGL_NVIDIA_MEGA_MOE_STRIDED_GEMM", "0") == "1"
STRIDED_SIDE_SMS = 8
# (kernel name, launch.json entry, source)
STRIDED_KERNELS = (
    ("gemm1_strided", "gemm1", "gemm1_strided.cu"),
    ("gemm1_balanced", "gemm1", "gemm1_strided_balanced.cu"),
    ("gemm2_strided", "gemm2", "gemm2_strided.cu"),
)


def compile_source(source, name, arch, options=NVRTC_OPTIONS, headers=None) -> bytes:
    return _compile_source(source, name, arch, options, headers)


# The rows the forward routes through the pair; larger steps keep FlashInfer
SERVED_MAX_ROWS = 48
# gate_topk.cuh specialized per router (experts, top_k), one NVRTC program and symbol
# each: the DSV4.1 target's and the DSpark draft's.
GATE_TOPK_EXPERTS = 384
GATE_TOPK_TOP_K = 6
DRAFT_GATE_TOPK_EXPERTS = 128
DRAFT_GATE_TOPK_TOP_K = 3
GATE_TOPK_VARIANTS = {
    (GATE_TOPK_EXPERTS, GATE_TOPK_TOP_K): ("gate_topk", "dsv41_gate_topk"),
    (DRAFT_GATE_TOPK_EXPERTS, DRAFT_GATE_TOPK_TOP_K): (
        "gate_topk_draft",
        "dsv41_gate_topk_e128_k3",
    ),
}


def max_ctas(rows: int, top_k: int, experts: int) -> int:
    """The most 8-row token tiles any routing of ``rows * top_k`` slots needs."""
    slots = rows * top_k
    filled = min(experts, slots)
    return filled + (slots - filled) // TILE


def cta_words(ctas: int) -> int:
    return (ctas + 3) // 4 * 4


def route_words(ctas: int) -> int:
    """int32 length of the route record for ``ctas`` token tiles."""
    return 4 + 2 * cta_words(ctas) + TILE * ctas


@cache
def load_kernels(device_index: int) -> dict[str, _Kernel]:
    """Compile (or read from the cache) and load the route kernel and GEMMs."""
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        torch.zeros(1, device=device)  # make the primary context current
        major, minor = torch.cuda.get_device_capability(device)
        arch = f"sm_{major}{minor}a"
        facts = json.loads((SOURCE_DIR / "launch.json").read_text())
        headers = {name: (SOURCE_DIR / name).read_text() for name in HEADERS}
        kernels = {
            "route": _load(
                compile_source(
                    (SOURCE_DIR / "route.cu").read_text(),
                    "route",
                    arch,
                    headers=headers,
                ),
                "dsv41_routed_expert_route",
                (ROUTE_THREADS, 1, 1),
                0,
            )
        }
        for name, fact in facts.items():
            if tuple(fact["parameters"]) != PARAMETERS.get(name):
                raise RuntimeError(
                    f"{name}: unexpected launch.json parameters {fact['parameters']}"
                )
            source = (SOURCE_DIR / f"{name}.cu").read_text()
            kernels[name] = _load(
                compile_source(source, name, arch),
                fact["symbol"],
                fact["block"],
                int(fact["shared_memory_bytes"]),
            )
        if STRIDED_GEMM:
            for name, entry, file in STRIDED_KERNELS:
                fact = facts[entry]
                kernels[name] = _load(
                    compile_source((SOURCE_DIR / file).read_text(), name, arch),
                    fact["symbol"],
                    fact["block"],
                    int(fact["shared_memory_bytes"]),
                )
        return kernels


@cache
def load_route_wide(device_index: int) -> _Kernel:
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        torch.zeros(1, device=device)
        major, minor = torch.cuda.get_device_capability(device)
        source = (SOURCE_DIR / "route_wide.cu").read_text()
        cubin = compile_source(source, "route_wide", f"sm_{major}{minor}a")
        return _load(
            cubin, "dsv41_routed_expert_route_wide", (ROUTE_WIDE_THREADS, 1, 1), 0
        )


# Triton's libdevice reflection (flush to zero, approximate sqrt and division), so the router
# math is bit for bit moe_fused_gate's.
GATE_TOPK_NVRTC_OPTIONS = (
    "-std=c++17",
    "-default-device",
    "--ftz=true",
    "--prec-sqrt=false",
    "--prec-div=false",
)
GATE_TOPK_HEADERS = ("gate_topk.cuh",)


@cache
def load_gate_topk(
    device_index: int, experts: int = GATE_TOPK_EXPERTS, top_k: int = GATE_TOPK_TOP_K
) -> _Kernel:
    """The gate top-k kernel of the (experts, top_k) router, one thread per expert."""
    name, symbol = GATE_TOPK_VARIANTS[(experts, top_k)]
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        torch.zeros(1, device=device)
        major, minor = torch.cuda.get_device_capability(device)
        source = (SOURCE_DIR / f"{name}.cu").read_text()
        headers = {h: (SOURCE_DIR / h).read_text() for h in GATE_TOPK_HEADERS}
        cubin = compile_source(
            source,
            name,
            f"sm_{major}{minor}a",
            GATE_TOPK_NVRTC_OPTIONS,
            headers=headers,
        )
        return _load(cubin, symbol, (experts, 1, 1), 0)


def _weight_map(weights: torch.Tensor, m: int, k: int, chunks: int):
    """TMA map of MXFP4 weights ``[E, m, k / 2]`` (16U4, 128B swizzle)."""
    cu = _driver()
    if weights.data_ptr() % 32:
        raise ValueError("packed expert weights must be 32-byte aligned")
    u64, u32 = cu.cuuint64_t, cu.cuuint32_t
    (tensor_map,) = _check(
        cu.cuTensorMapEncodeTiled(
            cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
            4,
            weights.data_ptr(),
            [u64(128), u64(m), u64(k // 128), u64(weights.size(0))],
            [u64(k // 2), u64(64), u64(m * k // 2)],
            [u32(128), u32(128), u32(chunks), u32(1)],
            [u32(1)] * 4,
            cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
            cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        ),
        "cuTensorMapEncodeTiled(weights)",
    )
    return tensor_map


def _scale_map(scales: torch.Tensor, groups: int):
    """TMA map of 128x4-layout scale bytes viewed as ``[bytes/256, 256]``."""
    cu = _driver()
    total = scales.numel() * scales.element_size()
    if total % 256 or scales.data_ptr() % 16:
        raise ValueError("weight scales must be whole 256-byte rows, 16-byte aligned")
    u64, u32 = cu.cuuint64_t, cu.cuuint32_t
    (tensor_map,) = _check(
        cu.cuTensorMapEncodeTiled(
            cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
            2,
            scales.data_ptr(),
            [u64(256), u64(total // 256)],
            [u64(256)],
            [u32(256), u32(2 * groups)],
            [u32(1)] * 2,
            cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
            cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        ),
        "cuTensorMapEncodeTiled(scales)",
    )
    return tensor_map


def check_weights(w13, w13_scale, w2, w2_scale, clamp, top_k: int) -> None:
    """Raise ValueError unless the weights have the layout the kernels read."""
    experts = w13.size(0)
    expected = (
        (w13, (experts, 2 * PADDED, HIDDEN // 2)),
        (w13_scale, (experts, 2 * PADDED, HIDDEN // 32)),
        (w2, (experts, HIDDEN, PADDED // 2)),
        (w2_scale, (experts, HIDDEN, PADDED // 32)),
    )
    for tensor, shape in expected:
        if (
            tuple(tensor.shape) != shape
            or tensor.element_size() != 1
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"expected a contiguous {shape} 1-byte tensor, got {tuple(tensor.shape)} {tensor.dtype}"
            )
    if experts % 32 or not 0 < experts <= MAX_EXPERTS:
        raise ValueError(
            f"expected a multiple of 32 local experts <= {MAX_EXPERTS}, got {experts}"
        )
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"expected a top-k of 1-{MAX_TOP_K}, got {top_k}")
    if (
        clamp is None
        or clamp.dtype != torch.float32
        or tuple(clamp.shape) != (experts,)
    ):
        raise ValueError("expected an FP32 [E] SwiGLU clamp limit")
    # GEMM1 never reads w13's padding tile, so the padding must be zero, as must the
    # matching K columns of w2.
    if (
        torch.count_nonzero(w13.view(torch.uint8)[:, 2 * SLICE :]).item()
        or torch.count_nonzero(w2.view(torch.uint8)[:, :, SLICE // 2 :]).item()
    ):
        raise ValueError("the intermediate padding of w13/w2 is not zero")


class RoutedExpertGemm:
    """The route kernel and GEMM pair bound to one MoE layer's weights."""

    def __init__(
        self,
        w13,
        w13_scale,
        w2,
        w2_scale,
        clamp,
        top_k: int,
        max_rows: int,
        gate_topk: bool = False,
        router_experts: Optional[int] = None,
        expert_offset: int = 0,
    ):
        """``router_experts``: the router's expert count (default: the local ones);
        ``expert_offset``: the router id of local expert 0 (expert parallelism)."""
        w13, w13_scale = w13.view(torch.uint8), w13_scale.view(torch.uint8)
        w2, w2_scale = w2.view(torch.uint8), w2_scale.view(torch.uint8)
        check_weights(w13, w13_scale, w2, w2_scale, clamp, top_k)
        if not MAX_ROWS <= max_rows <= WIDE_MAX_ROWS:
            raise ValueError(
                f"max_rows {max_rows} is outside {MAX_ROWS}-{WIDE_MAX_ROWS}"
            )
        self.experts = int(w13.size(0))
        self.router_experts = (
            self.experts if router_experts is None else int(router_experts)
        )
        self.expert_offset = int(expert_offset)
        if not (
            0 <= self.expert_offset
            and self.expert_offset + self.experts <= self.router_experts
        ):
            raise ValueError(
                f"local experts [{self.expert_offset}, "
                f"{self.expert_offset + self.experts}) are outside the router's "
                f"{self.router_experts}"
            )
        self.top_k = top_k
        self.max_rows = max_rows
        self.kernels = load_kernels(w13.device.index)
        self.route_wide = (
            load_route_wide(w13.device.index) if max_rows > MAX_ROWS else None
        )
        # the router's (experts, top_k) picks the variant: target or DSpark draft
        fits_gate_topk = (self.router_experts, top_k) in GATE_TOPK_VARIANTS
        self.gate_topk_kernel = (
            load_gate_topk(w13.device.index, self.router_experts, top_k)
            if gate_topk and fits_gate_topk
            else None
        )
        # route.cu flips bit 31 of this word on every launch.
        self.barrier = torch.zeros(1, dtype=torch.int32, device=w13.device)
        self._tensors = (
            w13,
            w13_scale,
            w2,
            w2_scale,
            clamp,
        )  # the maps hold their addresses
        self.clamp = clamp
        self.gemm1_maps = (
            _weight_map(w13, 2 * PADDED, HIDDEN, 2),
            _scale_map(w13_scale, 4),
        )
        self.gemm2_maps = (_weight_map(w2, HIDDEN, PADDED, 2), _scale_map(w2_scale, 1))
        self._strided_sms = (
            torch.cuda.get_device_properties(w13.device).multi_processor_count
            - STRIDED_SIDE_SMS
            if STRIDED_GEMM
            else 0
        )

    def _gemm_launches(self, ctas: int, top_k: int):
        """(GEMM1 kernel, grid), (GEMM2 kernel, grid) for ``ctas`` token tiles; fixed
        by the call's shape, never by its routes, so one graph keeps one kernel."""
        if not self._strided_sms:
            return (
                (self.kernels["gemm1"], (GEMM1_UNITS, ctas, 1)),
                (self.kernels["gemm2"], (GEMM2_UNITS, ctas, 1)),
            )
        tiles1 = GEMM1_UNITS * ctas
        grid1 = min(self._strided_sms, tiles1)
        balanced = self.experts == 384 and top_k == 6 and tiles1 > grid1
        return (
            (
                self.kernels["gemm1_balanced" if balanced else "gemm1_strided"],
                (grid1, 1, 1),
            ),
            (
                self.kernels["gemm2_strided"],
                (min(self._strided_sms, GEMM2_UNITS * ctas), 1, 1),
            ),
        )

    def supports(self, x_quant, x_scale, topk_ids, topk_weights) -> bool:
        rows = x_quant.size(0)
        return (
            1 <= rows <= self.max_rows
            and topk_ids.dim() == 2
            and topk_ids.size(1) == self.top_k
            and tuple(x_quant.shape) == (rows, HIDDEN)
            and x_quant.element_size() == 1
            and x_quant.is_contiguous()
            and x_scale is not None
            and x_scale.element_size() == 1
            and x_scale.is_contiguous()
            and x_scale.numel() >= rows * HIDDEN // 32
            and topk_weights.dtype == torch.float32
        )

    def __call__(self, x_quant, x_scale, topk_ids, topk_weights):
        """Route + GEMM1 + GEMM2 on the current stream; returns FlashInfer's
        deferred-finalize triple (gemm2 rows, weights, slot -> permuted row)."""
        rows, top_k = topk_ids.shape
        device = x_quant.device
        stream = torch.cuda.current_stream(device).cuda_stream
        ids = topk_ids.to(torch.int32).contiguous()
        ctas = max_ctas(rows, top_k, self.experts)
        permuted = torch.empty((rows, top_k), dtype=torch.int32, device=device)
        record = torch.empty(route_words(ctas), dtype=torch.int32, device=device)
        if rows > MAX_ROWS:
            _launch(
                self.route_wide,
                (1, 1, 1),
                [
                    _pointer(ids),
                    _pointer(permuted),
                    _pointer(record),
                    ctypes.c_int32(rows),
                    ctypes.c_int32(top_k),
                    ctypes.c_int32(self.experts),
                    ctypes.c_int32(self.expert_offset),
                ],
                stream,
            )
        else:
            _launch(
                self.kernels["route"],
                (1, 1, 1),
                [
                    _pointer(ids),
                    _pointer(permuted),
                    _pointer(record),
                    _pointer(self.barrier),
                    ctypes.c_int32(rows),
                    ctypes.c_int32(top_k),
                    ctypes.c_int32(self.experts),
                    ctypes.c_int32(self.expert_offset),
                ],
                stream,
            )
        padded_rows = TILE * ctas
        h_values = torch.empty((padded_rows, PADDED), dtype=torch.uint8, device=device)
        h_scales = torch.empty(
            (padded_rows + 127) // 128 * 128 * PADDED // 32,
            dtype=torch.uint8,
            device=device,
        )
        gemm2_rows = torch.empty(
            (rows * top_k * TILE, HIDDEN), dtype=torch.bfloat16, device=device
        )
        words = ctypes.c_int32(cta_words(ctas))
        (gemm1, grid1), (gemm2, grid2) = self._gemm_launches(ctas, top_k)
        _launch(
            gemm1,
            grid1,
            [
                *self.gemm1_maps,
                _pointer(x_quant),
                _pointer(x_scale),
                _pointer(record),
                _pointer(self.clamp),
                _pointer(h_values),
                _pointer(h_scales),
                words,
            ],
            stream,
        )
        _launch(
            gemm2,
            grid2,
            [
                *self.gemm2_maps,
                _pointer(h_values),
                _pointer(h_scales),
                _pointer(record),
                _pointer(gemm2_rows),
                words,
            ],
            stream,
        )
        return gemm2_rows, topk_weights.contiguous().view(rows, top_k), permuted

    def gate_topk_and_gemms(
        self,
        x_quant: torch.Tensor,
        x_scale: torch.Tensor,
        logits: torch.Tensor,
        bias: torch.Tensor,
        routed_scaling_factor: float,
        num_valid: Optional[torch.Tensor],
        alt: Optional[tuple[torch.Tensor, torch.Tensor, int]] = None,
        renormalize_epsilon: float = 0.0,
    ):
        """``__call__`` from the router's fp32 logits: the router's gate_topk.cuh variant
        does moe_fused_gate's top-k (sqrtsoftplus via log1p, renormalized, scaled), then
        the route kernel. ``alt`` = (bias_alt, input_ids, token id): rows whose token is
        that id rank with bias_alt (DSV4.1 VL routing). ``num_valid`` None means every row is valid. The
        caller joins the quantized input before this call: a stream join between the route
        kernel and GEMM1 would cost GEMM1 its PDL early launch.
        The ids are the router's; a slot whose expert another rank holds maps to row -1.
        Returns (gemm2 rows, top-k weights, slot -> row, top-k ids)."""
        rows = logits.size(0)
        device = logits.device
        stream = torch.cuda.current_stream(device).cuda_stream
        weights = torch.empty((rows, self.top_k), dtype=torch.float32, device=device)
        ids = torch.empty((rows, self.top_k), dtype=torch.int32, device=device)
        bias_alt, input_ids, alt_token_id = alt if alt is not None else (None, None, -1)
        _launch(
            self.gate_topk_kernel,
            (rows, 1, 1),
            [
                _pointer(logits),
                _pointer(bias),
                _null_or_pointer(bias_alt),
                _null_or_pointer(input_ids),
                ctypes.c_int32(
                    input_ids is not None and input_ids.dtype == torch.int64
                ),
                ctypes.c_int64(alt_token_id),
                _null_or_pointer(num_valid),
                _pointer(weights),
                _pointer(ids),
                ctypes.c_float(routed_scaling_factor),
                ctypes.c_float(renormalize_epsilon),
            ],
            stream,
        )
        ctas = max_ctas(rows, self.top_k, self.experts)
        permuted, record = self._route(ids, rows, ctas)
        return self._gemms(x_quant, x_scale, record, ctas), weights, permuted, ids

    def _route(self, ids: torch.Tensor, rows: int, ctas: int):
        """The route kernel __call__ launches (route_wide.cu above 8 rows)."""
        stream = torch.cuda.current_stream(ids.device).cuda_stream
        permuted = torch.empty((rows, self.top_k), dtype=torch.int32, device=ids.device)
        record = torch.empty(route_words(ctas), dtype=torch.int32, device=ids.device)
        common = [_pointer(ids), _pointer(permuted), _pointer(record)]
        sizes = [
            ctypes.c_int32(rows),
            ctypes.c_int32(self.top_k),
            ctypes.c_int32(self.experts),
            ctypes.c_int32(self.expert_offset),
        ]
        if rows > MAX_ROWS:
            _launch(self.route_wide, (1, 1, 1), common + sizes, stream)
        else:
            _launch(
                self.kernels["route"],
                (1, 1, 1),
                common + [_pointer(self.barrier)] + sizes,
                stream,
            )
        return permuted, record

    def supports_inputs(self, x_quant, x_scale) -> bool:
        rows = x_quant.size(0)
        return (
            1 <= rows <= self.max_rows
            and tuple(x_quant.shape) == (rows, HIDDEN)
            and x_quant.element_size() == 1
            and x_quant.is_contiguous()
            and x_scale is not None
            and x_scale.element_size() == 1
            and x_scale.is_contiguous()
            and x_scale.numel() >= rows * HIDDEN // 32
        )

    def supports_gate_topk(self, logits: torch.Tensor, bias: torch.Tensor) -> bool:
        return (
            self.gate_topk_kernel is not None
            and logits.dtype == torch.float32
            and logits.dim() == 2
            and logits.is_contiguous()
            and tuple(logits.shape[1:]) == (self.router_experts,)
            and 1 <= logits.size(0) <= self.max_rows
            and bias.dtype == torch.float32
            and bias.is_contiguous()
        )

    def _gemms(self, x_quant, x_scale, record, ctas):
        """GEMM1 + GEMM2 over a built route record."""
        device = x_quant.device
        stream = torch.cuda.current_stream(device).cuda_stream
        rows = x_quant.size(0)
        padded_rows = TILE * ctas
        h_values = torch.empty((padded_rows, PADDED), dtype=torch.uint8, device=device)
        h_scales = torch.empty(
            (padded_rows + 127) // 128 * 128 * PADDED // 32,
            dtype=torch.uint8,
            device=device,
        )
        gemm2_rows = torch.empty(
            (rows * self.top_k * TILE, HIDDEN), dtype=torch.bfloat16, device=device
        )
        words = ctypes.c_int32(cta_words(ctas))
        (gemm1, grid1), (gemm2, grid2) = self._gemm_launches(ctas, self.top_k)
        _launch(
            gemm1,
            grid1,
            [
                *self.gemm1_maps,
                _pointer(x_quant),
                _pointer(x_scale),
                _pointer(record),
                _pointer(self.clamp),
                _pointer(h_values),
                _pointer(h_scales),
                words,
            ],
            stream,
        )
        _launch(
            gemm2,
            grid2,
            [
                *self.gemm2_maps,
                _pointer(h_values),
                _pointer(h_scales),
                _pointer(record),
                _pointer(gemm2_rows),
                words,
            ],
            stream,
        )
        return gemm2_rows
