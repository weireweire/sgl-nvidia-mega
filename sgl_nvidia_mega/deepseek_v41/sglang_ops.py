# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SGLang operators the plugin calls, read wherever the running SGLang keeps them: SGLang's
regroup of its kernels by computation moved these modules and renamed ``hc_combine``, and
its layer-boundary push-plane fusion moved ``MoeDeferredFinalize``.
Each resolves once, newest location first; test_deepseek_v41_sglang_ops pins that every one
resolves on the SGLang under test."""

from __future__ import annotations

import functools
import importlib

import torch

# (module, attribute) candidates per operator, newest SGLang first
CANDIDATES = {
    "rmsnorm_mxfp8": (
        ("sglang.kernels.ops.layernorm.rmsnorm_mxfp8", "rmsnorm_mxfp8"),
        ("sglang.kernels.ops.layernorm.mxfp8_epilogue", "rmsnorm_mxfp8"),
    ),
    "per_token_group_quant_fp8_wo_a": (
        (
            "sglang.kernels.ops.quantization.fp8_group_major_quant_ue8m0",
            "sglang_per_token_group_quant_fp8_dsv4_wo_a",
        ),
        (
            "sglang.kernels.ops.attention.dsv4",
            "sglang_per_token_group_quant_fp8_dsv4_wo_a",
        ),
    ),
    "hc_combine": (
        ("sglang.kernels.ops.layernorm.mhc", "mhc_combine"),
        ("sglang.kernels.ops.layernorm.mhc", "hc_combine"),
    ),
    "stream_mean": (("sglang.kernels.ops.layernorm.mhc", "mhc_stream_mean"),),
    "silu_and_mul_clamp": (
        ("sglang.kernels.ops.activation.silu_and_mul_clamp", "silu_and_mul_clamp"),
        ("sglang.kernels.ops.moe.dsv4", "silu_and_mul_clamp"),
    ),
    "MoeDeferredFinalize": (
        (
            "sglang.srt.layers.layer_boundary.fusions.moe_finalize",
            "MoeDeferredFinalize",
        ),
        ("sglang.srt.layers.layer_boundary.fusions.cutedsl", "MoeDeferredFinalize"),
    ),
    "is_unified_kv_triton": (
        ("sglang.srt.layers.attention.dsv4.unified_kv_gate", "is_unified_kv_triton"),
        (
            "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate",
            "is_unified_kv_triton",
        ),
    ),
}


@functools.cache
def resolve(name: str):
    """The first of ``name``'s candidates the running SGLang provides."""
    for module, attr in CANDIDATES[name]:
        try:
            return getattr(importlib.import_module(module), attr)
        except (ImportError, AttributeError):
            continue
    raise ImportError(f"SGLang provides none of {CANDIDATES[name]}")


def rmsnorm_mxfp8(*args, **kwargs):
    return resolve("rmsnorm_mxfp8")(*args, **kwargs)


def per_token_group_quant_fp8_wo_a(*args, **kwargs):
    return resolve("per_token_group_quant_fp8_wo_a")(*args, **kwargs)


def hc_combine(*args, **kwargs):
    return resolve("hc_combine")(*args, **kwargs)


def hc_combine_widens() -> bool:
    """SGLang's newer combine widens BF16 streams to FP32 as it loads them."""
    return resolve("hc_combine").__name__ == "mhc_combine"


def stream_mean(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """``x.mean(dim=1)`` written into ``out`` (bitwise either way)."""
    try:
        return resolve("stream_mean")(x, out=out)
    except ImportError:
        return out.copy_(x.mean(dim=1))


def silu_and_mul_clamp(*args, **kwargs):
    return resolve("silu_and_mul_clamp")(*args, **kwargs)


def unified_kv_triton() -> bool:
    return resolve("is_unified_kv_triton")()


def moe_deferred_finalize_from_flashinfer(*args, **kwargs):
    return resolve("MoeDeferredFinalize").from_flashinfer(*args, **kwargs)
