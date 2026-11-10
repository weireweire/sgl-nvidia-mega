# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot lm_head_pinned_tactic: the low-M LM head (the draft's and, with target_lm_head,
the target's) on one fixed FlashInfer cute-dsl kernel and tactic per row count, launched
directly instead of through FlashInfer's autotuner.

A per-process autotune picks among kernels whose split-K sums K in different orders,
so each rank's logits shard, and with it a near-tie argmax, could change from one launch
to the next. A fixed tactic gives the same bits on every rank and every launch, and the
heads need no tuning pass."""

from __future__ import annotations

import logging
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)

# (SM version, N, K, rows) -> (FlashInfer runner class, tactic); rows absent here, or a
# tactic the runner no longer runs, run FlashInfer's default (its first runner, tactic
# -1), which is fixed per shape too. GB300 TP4 shard (N 32320, K 5120), CUDA graph
# replay: the fastest offered tactic per row count, 0.4-1.4 us under the default.
_DIRECT, _WARP = "CuteDSLDirectBf16Runner", "CuteDSLWarpSplitKBf16Runner"
PINNED_TACTICS: dict[tuple[int, int, int, int], tuple[str, tuple[int, ...]]] = {
    (103, 32320, 5120, 1): (_DIRECT, (64, 2, 1)),
    (103, 32320, 5120, 2): (_WARP, (16, 8, 256, 3, 1)),
    (103, 32320, 5120, 4): (_WARP, (16, 8, 256, 3, 1)),
    (103, 32320, 5120, 6): (_WARP, (16, 8, 256, 4, 2)),
    (103, 32320, 5120, 8): (_WARP, (16, 8, 128, 12, 1)),
    (103, 32320, 5120, 12): (_WARP, (16, 16, 256, 4, 2)),
    (103, 32320, 5120, 16): (_WARP, (16, 16, 256, 4, 2)),
    # GB300 TP8 shard (N 16160): 1.8-2.4 us under the default (23.9-26.0 us vs
    # 25.6-28.0; benchmark/sweep_deepseek_v41_tactics.py, which also re-picks the TP4 pin
    # at 6 rows)
    (103, 16160, 5120, 1): (_DIRECT, (128, 2, 1)),
    (103, 16160, 5120, 2): (_WARP, (32, 8, 256, 2, 1)),
    (103, 16160, 5120, 4): (_WARP, (32, 8, 256, 2, 1)),
    (103, 16160, 5120, 5): (_WARP, (32, 8, 256, 2, 1)),
    (103, 16160, 5120, 6): (_WARP, (32, 8, 256, 2, 1)),
    (103, 16160, 5120, 8): (_WARP, (32, 8, 256, 2, 1)),
    (103, 16160, 5120, 12): (_WARP, (32, 16, 128, 3, 2)),
    (103, 16160, 5120, 16): (_WARP, (32, 16, 128, 3, 2)),
}
_LOGGED_ROWS: set[int] = set()


def _runners(x: torch.Tensor, weight_t: torch.Tensor, out: torch.Tensor) -> list:
    from flashinfer.gemm import gemm_base

    return gemm_base._cute_dsl_bf16_runners(_inputs(x, weight_t, out))


def _inputs(x: torch.Tensor, weight_t: torch.Tensor, out: torch.Tensor) -> list:
    # mm_bf16's runner inputs: a, b, bias, pdl, out, workspace (the low-M kernels take none)
    return [x, weight_t, None, True, out, _empty(x.device)]


@lru_cache(maxsize=None)
def _empty(device: torch.device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.uint8, device=device)


@lru_cache(maxsize=None)
def _choice(sm_version: int, n: int, k: int, rows: int, device: torch.device):
    x = torch.empty(rows, k, dtype=torch.bfloat16, device=device)
    weight_t = torch.empty(n, k, dtype=torch.bfloat16, device=device).t()
    out = torch.empty(rows, n, dtype=torch.bfloat16, device=device)
    runners = _runners(x, weight_t, out)
    pin = PINNED_TACTICS.get((sm_version, n, k, rows))
    if pin is not None:
        name, tactic = pin
        for runner in runners:
            if type(runner).__name__ != name or not runner.supports_inputs(
                _inputs(x, weight_t, out)
            ):
                continue
            if _runs(runner, _inputs(x, weight_t, out), tactic):
                return runner, tactic
        logger.warning(
            "NVIDIA megakernel: pinned LM-head tactic %s %s not offered at %d rows; "
            "FlashInfer's default instead",
            name,
            tactic,
            rows,
        )
    return runners[0], -1


def _runs(runner, inputs: list, tactic: tuple[int, ...]) -> bool:
    """Whether the runner offers the tactic, or (FlashInfer 0.7.3 prunes its warp split-K
    autotune list) its own tactic check accepts it."""
    if tactic in [tuple(t) for t in runner.get_valid_tactics(inputs, None)]:
        return True
    compatible = getattr(runner, "is_tactic_compatible", None)
    return compatible is not None and compatible(inputs, tactic)


def pinned_mm(x: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
    """BF16 ``x @ weight_t`` at 1-32 rows on this row count's fixed kernel and tactic."""
    rows, (k, n) = x.shape[0], weight_t.shape
    major, minor = torch.cuda.get_device_capability(x.device)
    runner, tactic = _choice(major * 10 + minor, n, k, rows, x.device)
    if rows not in _LOGGED_ROWS:
        _LOGGED_ROWS.add(rows)
        logger.info(
            "NVIDIA megakernel: LM head pinned at %d rows: %s %s",
            rows,
            type(runner).__name__,
            tactic,
        )
    out = torch.empty(rows, n, dtype=torch.bfloat16, device=x.device)
    runner(inputs=_inputs(x, weight_t, out), tactic=tactic)
    return out
