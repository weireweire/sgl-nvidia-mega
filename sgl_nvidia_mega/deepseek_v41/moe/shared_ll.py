# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/deepseek_v2.py (DeepseekV2MLP.forward, its CUDA clamped path).
"""The shared expert of a MoE layer with its gate_up GEMM on the low-latency MXFP8
kernel (``sm103/mxfp8_ll``; slot ``shared_ll``), 3.5-5.7 us
faster than SGLang's dense MXFP8 GEMM at 6-48 rows. On the alternate stream, as SGLang's
shared expert: the rows' MXFP8 quant in the kernel's row-tile layout (bitwise the shared
linear's own quant, scales moved; it triggers the GEMM at once), the GEMM, then SGLang's
clamped SwiGLU and down_proj, as ``DeepseekV2MLP.forward`` runs them. The chain starts
at the layer's fork: one that waits on the routed quant launches after the routed GEMMs
hold every SM and runs in their tail (measured +84 / +242 us a step at batch 1 / 8)."""

from __future__ import annotations

import torch

from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll
from sgl_nvidia_mega.deepseek_v41.sm103.attention.q_norm_quant import mxfp8_row_tiles

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/deepseek_v2.py": (
        "shared_experts",
        "num_fused_shared_experts",
        "swiglu_limit",
        "allow_fused_down",
        "use_fused_clamp_act_mul",
        "_enable_nvfp4_gemm_swiglu_fusion",
        "gate_up_proj",
        "down_proj",
    ),
    "srt/layers/quantization/fp8.py": ("weight_scale_inv_swizzled",),
}


def why_not(mlp: torch.nn.Module) -> str:
    """Why the layer's shared expert cannot take the slot ("" when it can): it must run
    SGLang's CUDA clamped-SwiGLU path with an MXFP8 gate_up."""
    shared = getattr(mlp, "shared_experts", None)
    if shared is None or mlp.num_fused_shared_experts:
        return "no separate shared expert"
    if not mxfp8_ll.supported():
        return "no low-latency MXFP8 GEMM on this GPU"
    if (
        shared.swiglu_limit is None
        or shared.use_fused_clamp_act_mul
        or getattr(shared, "_enable_nvfp4_gemm_swiglu_fusion", False)
    ):
        return "not SGLang's CUDA clamped SwiGLU"
    down = shared.down_proj
    if (
        shared.allow_fused_down
        and getattr(down, "weight", None) is not None
        and down.weight.dtype == torch.uint8
        and hasattr(down, "weight_scale_inv")
    ):
        return "DeepGEMM's fused down_proj"
    gate_up = shared.gate_up_proj
    if (
        getattr(gate_up, "weight_scale_inv_swizzled", None) is None
        or gate_up.weight.dtype != torch.float8_e4m3fn
        or not gate_up.weight.is_contiguous()
    ):
        return "gate_up has no MXFP8 view"
    n, k = gate_up.weight.shape
    # N: the TP8 shard's 576 (2 x 288) ends in half a 128-row tile, which the kernel's
    # TMA zero-fills and its predicated stores drop; its swizzled scales are padded
    if n % 64 or k % 128:
        return f"gate_up {n} x {k}: the kernel takes N % 64 == 0 and K % 128 == 0"
    return ""


class SharedLL:
    """One layer's shared expert at 1-``max_rows`` rows."""

    def __init__(self, mlp: torch.nn.Module, max_rows: int):
        shared = mlp.shared_experts
        gate_up = shared.gate_up_proj
        self.weight, self.weight_scale = (
            gate_up.weight,
            gate_up.weight_scale_inv_swizzled,
        )
        self.down_proj = shared.down_proj
        self.limit = float(shared.swiglu_limit)
        self.max_rows = max_rows

    def precompile(self, widths) -> None:
        """The GEMM's and the quant's kernels for these widths (the quant's row count is
        a compile-time constant); before CUDA graph capture."""
        rows = [w for w in widths if w <= self.max_rows]
        mxfp8_ll.precompile(self.weight, rows)
        x = torch.zeros(
            max(rows, default=1),
            self.weight.shape[1],
            dtype=torch.bfloat16,
            device=self.weight.device,
        )
        for r in rows:
            mxfp8_row_tiles(x[:r])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """The shared expert's output for BF16 rows ``x``, on the current stream."""
        from sgl_nvidia_mega.deepseek_v41 import sglang_ops

        rows = x.shape[0]
        assert 0 < rows <= self.max_rows
        x_q, x_s = mxfp8_row_tiles(x)
        gate_up = mxfp8_ll.linear(x_q, x_s, self.weight, self.weight_scale)
        h = gate_up.new_empty((rows, gate_up.shape[1] // 2))
        sglang_ops.silu_and_mul_clamp(gate_up, h, self.limit)
        return self.down_proj(h)[0]
