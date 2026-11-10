# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The mHC coefficients at decode rows from one DeepGEMM TF32 prenorm (slot
``hc_prenorm``).

Below 128 rows SGLang predicts each sublayer's (pre, post, comb) from a Triton
tf32x3 split-K over 80 K slices, then a reduce + Sinkhorn row kernel over the same
80 partials. Here one DeepGEMM ``tf32_hc_prenorm_gemm`` over 16 K slices reads the
fp32 mixing weight as TF32 (no compensation term), and SGLang's own reduce +
Sinkhorn kernel sums the 16 partials. Both kernels stay on the stats stream SGLang
forks for them; the coefficients differ from SGLang's by the TF32 rounding of the
weight, within ``spec.NUMERICS`` (``hc_prenorm``).

With slot ``hc_coeff_fused`` one kernel (``sm103/hc_coeff.py``) does both, bit for bit
the pair: its build checks that on the model's own weights.
"""

from __future__ import annotations

import torch

# The SGLang names this module reads, by file under the sglang package;
# test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/layers/deep_gemm_wrapper/entrypoint.py": ("tf32_hc_prenorm_gemm",),
    "srt/layers/deep_gemm_wrapper/configurer.py": ("ENABLE_JIT_DEEPGEMM",),
    "kernels/ops/layernorm/mhc.py": ("_hc_mix_reduce_sinkhorn_kernel",),
    "srt/models/deepseek_v4_mhc.py": (
        "mix_stats",
        "fn",
        "scale",
        "base",
        "cfg",
        "mult",
        "sinkhorn_iters",
        "rms_eps",
        "eps",
    ),
    "srt/batch_invariant_ops/batch_invariant_ops.py": (
        "is_batch_invariant_mode_enabled",
    ),
    "srt/runtime_context.py": ("get_platform", '"is_blackwell"'),
    # the module alias SGLang's DSV4 model imports too
    "srt/models/deepseek_v4.py": ("mhc",),
}

# The K slices of the prenorm GEMM, and so of the reduce
SLICES = 16
# The widest step it takes; past it SGLang's path
MAX_ROWS = 48
# Slot hc_coeff_fused: (K, hc, Sinkhorn iterations, eps, rms_eps) -> its one-launch
# kernel, which serves up to FUSED_ROWS rows (past them its Sinkhorn tail, a row per
# warp of one CTA, is slower than the pair's CTA per row)
_FUSED: dict = {}
FUSED_ROWS = 8


def reason() -> str:
    """Why the prenorm cannot run on this setup ("" when it can)."""
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang.srt.layers.deep_gemm_wrapper.configurer import ENABLE_JIT_DEEPGEMM
    from sglang.srt.runtime_context import get_platform

    if not get_platform().is_blackwell:
        return "needs SM100-family GPUs"
    if not ENABLE_JIT_DEEPGEMM:
        return "needs DeepGEMM's JIT"
    if is_batch_invariant_mode_enabled():
        return "batch-invariant mode keeps SGLang's split-K"
    return ""


def takes(x: torch.Tensor) -> bool:
    """Whether the prenorm computes the coefficients of these streams (shape and
    dtype only, so fixed per graph key and the same on every rank)."""
    x_flat = x.flatten(1)
    return (
        0 < x_flat.shape[0] <= MAX_ROWS
        and x.dtype == torch.bfloat16
        and x_flat.is_contiguous()
    )


def mix_stats(hc, x: torch.Tensor, stats_stream=None):
    """SGLang's ``mix_stats(hc, x, stats_stream)``, with the prenorm GEMM where it
    ``takes`` the streams: the same stream handling, so callers join it as they join
    SGLang's."""
    from sglang.srt.models import deepseek_v4_mhc as mhc

    if not takes(x):
        return mhc.mix_stats(hc, x, stats_stream)
    x_flat = x.flatten(1)
    if stats_stream is None:
        return _coefficients(hc, x_flat)
    main_stream = torch.cuda.current_stream()
    x.record_stream(stats_stream)
    with torch.cuda.stream(stats_stream):
        coefficients = _coefficients(hc, x_flat)
    for coefficient in coefficients:
        coefficient.record_stream(main_stream)
    return coefficients


def build_fused(subs, device, group) -> str:
    """Slot hc_coeff_fused: compile the one-launch kernel for every mHC config among
    ``subs`` (HcSubLayers) and check it against the pair, bit for bit, at every row
    count on each sublayer's weights. "" when served, else why not (alike on every
    rank: each rank's verdict is gathered)."""
    from sgl_nvidia_mega.deepseek_v41.sm103.hc_coeff import HcCoeff

    kernels, error = {}, ""
    try:
        gen = torch.Generator(device=device).manual_seed(0)
        for hc in subs:
            cfg, k = hc.cfg, hc.fn.shape[1]
            key = _key(cfg, k)
            if key not in kernels:
                kernels[key] = HcCoeff(
                    hc=cfg.mult,
                    k=k,
                    slices=SLICES,
                    iters=cfg.sinkhorn_iters,
                    eps=cfg.eps,
                    rms_eps=cfg.rms_eps,
                    max_rows=FUSED_ROWS,
                    device=device,
                )
            for rows in range(1, FUSED_ROWS + 1):
                x = torch.randn((rows, k), device=device, generator=gen).bfloat16()
                want = _pair(hc, x)
                got = kernels[key](x, hc.fn, *_scale_base(hc))
                if any(
                    not torch.equal(g.view(torch.int32), w.view(torch.int32))
                    for g, w in zip(got, want)
                ):
                    raise RuntimeError(f"differs from the pair at {rows} rows")
        torch.cuda.synchronize(device)
    except Exception as e:  # gathered, then the same verdict on every rank
        error = f"{type(e).__name__}: {e}"
    errors = [None] * group.world_size
    torch.distributed.all_gather_object(errors, error, group=group.cpu_group)
    failed = [f"rank {r}: {e}" for r, e in enumerate(errors) if e]
    if failed:
        return "; ".join(failed)
    _FUSED.update(kernels)
    return ""


def fused() -> dict:
    """The one-launch kernels built (slot hc_coeff_fused), by mHC config."""
    return dict(_FUSED)


def _key(cfg, k: int):
    return (k, cfg.mult, cfg.sinkhorn_iters, cfg.eps, cfg.rms_eps)


def _scale_base(hc):
    return hc.scale.float().contiguous(), hc.base.float().contiguous()


def _coefficients(hc, x_flat: torch.Tensor):
    fused = _FUSED.get(_key(hc.cfg, x_flat.shape[1]))
    if fused is not None and x_flat.shape[0] <= FUSED_ROWS:
        return fused(x_flat, hc.fn, *_scale_base(hc))
    return _pair(hc, x_flat)


def _pair(hc, x_flat: torch.Tensor):
    from sglang.kernels.ops.layernorm.mhc import _hc_mix_reduce_sinkhorn_kernel
    from sglang.srt.layers.deep_gemm_wrapper.entrypoint import tf32_hc_prenorm_gemm

    cfg = hc.cfg
    rows, k = x_flat.shape
    mult = cfg.mult
    mix = (2 + mult) * mult
    device = x_flat.device
    part_mix = torch.empty((SLICES, rows, mix), dtype=torch.float32, device=device)
    part_sq = torch.empty((SLICES, rows), dtype=torch.float32, device=device)
    tf32_hc_prenorm_gemm(x_flat, hc.fn, part_mix, part_sq, SLICES)
    pre = torch.empty((rows, mult), dtype=torch.float32, device=device)
    post = torch.empty_like(pre)
    comb = torch.empty((rows, mult, mult), dtype=torch.float32, device=device)
    _hc_mix_reduce_sinkhorn_kernel[(rows,)](
        part_mix,
        part_sq,
        *_scale_base(hc),
        pre,
        post,
        comb,
        rows,
        1.0 / k,
        cfg.rms_eps,
        MIX=mix,
        HC=mult,
        NUM_SLICES=SLICES,
        ITERS=cfg.sinkhorn_iters,
        EPS=cfg.eps,
        num_warps=1,
    )
    return pre, post, comb
