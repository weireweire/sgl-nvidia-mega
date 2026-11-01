# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/deepseek_v4.py (the tails of DeepseekV4Model.forward and DeepseekV4ForCausalLM.forward).
"""The head after the last layer: the mHC combine + norm (``head``), then the LM
head and logits (``logits``, SGLang's logits processor). Stock: the tail of
``DeepseekV4Model.forward`` and of ``DeepseekV4ForCausalLM.forward``.

Slot target_lm_head (``build``): the logits processor gets a stand-in for the LM head
whose quant method runs the shard on FlashInfer's low-M BF16 kernel at
1-TARGET_MAX_ROWS rows (batch 1) and SGLang's own matmul otherwise; the TP gather, the logit
scale and the output fields stay SGLang's."""

from __future__ import annotations

import logging

import torch

from sgl_nvidia_mega.deepseek_v41.sampler import (
    LM_HEAD_BACKEND,
    low_m_lm_head_supported,
)

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/deepseek_v4.py": (
        "norm",
        "hc_mult",
        "logits_processor",
        "lm_head",
    ),
    "srt/layers/logits_processor.py": (
        "forward(hidden_states_before_norm)",
        "use_fp32_lm_head",
        "rl_on_policy_target",
        # the stand-in's route: SGLang applies a quant method outside its unquantized
        # names (test_deepseek_v41_target_lm_head drives the real one)
        "_compute_lm_head(lm_head)",
        "should_apply_lm_head_quant_method",
        "should_apply_lm_head_quant_method(quant_method)",
        "_UNQUANTIZED_LM_HEAD_METHODS",
    ),
    "srt/distributed/parallel_state.py": ("world_size", "cpu_group"),
}
# GB300 TP4 shard, SGLang's matmul vs the low-M kernel: 55.4 vs 47.5 us at 6 rows, but
# 51.4 vs 50.8 us at 12 (cuBLAS is fast there), so batch 2 up keeps SGLang's bitwise
TARGET_MAX_ROWS = 8
# FlashInfer's token buckets: tuned at these, the low-M kernel is at least as fast at
# the verify rows as tuned at them
TUNING_ROWS = (1, 2, 4, 8)
# build checks the logits against SGLang's head at these rows (spec.NUMERICS)
CHECK_ROWS = (1, 2, 4, 6, 8)


class LowMLmHeadMethod:
    """The stand-in's quant method. SGLang's ``_compute_lm_head`` calls ``apply`` for
    any method outside its unquantized names, ahead of its FP32 and on-policy branches
    (so ``build`` refuses those)."""

    def __init__(self, weight: torch.Tensor, *, pinned: bool = False):
        self.weight = weight
        # the CuTe DSL kernels take operands over DLPack, which refuses grad tensors
        self.weight_t = weight.detach().t()
        self.captured_rows: set[int] = set()
        # slot lm_head_pinned_tactic: one fixed tactic per row count, no tuning
        self.pinned = pinned

    def low_m(self, x: torch.Tensor, bias) -> bool:
        return (
            bias is None
            and x.dim() == 2
            and 1 <= x.shape[0] <= TARGET_MAX_ROWS
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
        )

    def apply(self, layer, x: torch.Tensor, bias=None) -> torch.Tensor:
        if not self.low_m(x, bias):
            # SGLang's BF16 branch (it adds no bias either)
            return torch.matmul(x.to(self.weight.dtype), self.weight.T)
        rows = x.shape[0]
        if _capturing() and rows not in self.captured_rows:
            # a replay runs no Python: the captured row counts are where it serves
            self.captured_rows.add(rows)
            logger.info("NVIDIA megakernel: target_lm_head captured at %d rows", rows)
        if self.pinned:
            from sgl_nvidia_mega.deepseek_v41.sampler.pinned import pinned_mm

            return pinned_mm(x, self.weight_t)
        from flashinfer import mm_bf16

        return mm_bf16(x, self.weight_t, pdl=True, backend=LM_HEAD_BACKEND)

    def tune(self) -> None:
        """The same calls on every rank, before any graph is captured; the draft's head
        shares the weight, so this and the draft's tuning share FlashInfer's cache."""
        if self.pinned:
            return
        from flashinfer.autotuner import autotune

        w = self.weight
        with autotune(True):
            for rows in TUNING_ROWS:
                x = torch.zeros(rows, w.shape[1], dtype=w.dtype, device=w.device)
                self.apply(None, x)


class TargetLmHead:
    """What the logits processor gets for the LM head: SGLang's module, with the
    low-M quant method."""

    def __init__(self, lm_head, *, pinned: bool = False):
        self.lm_head, self.weight = lm_head, lm_head.weight
        self.quant_method = LowMLmHeadMethod(lm_head.weight, pinned=pinned)

    def __getattr__(self, name):
        return getattr(self.lm_head, name)


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def target_lm_head_reason(lm) -> str:
    """Why slot target_lm_head cannot serve this model's head ("" when it can)."""
    from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method

    head, proc = lm.lm_head, lm.logits_processor
    weight = head.weight
    if hasattr(head, "set_lora") and hasattr(head, "apply_lora"):
        return "the LM head is LoRA-wrapped"
    if should_apply_lm_head_quant_method(head, getattr(head, "quant_method", None)):
        return "SGLang runs the LM head through its quant method"
    if weight.dtype != torch.bfloat16 or not weight.is_contiguous():
        return f"the LM head weight is {weight.dtype}, not contiguous BF16"
    if proc.use_fp32_lm_head:
        return "SGLang computes the LM head in FP32"
    if proc.rl_on_policy_target is not None:
        return "SGLang's on-policy target branch"
    if not low_m_lm_head_supported():
        return f"FlashInfer's {LM_HEAD_BACKEND} BF16 GEMM is unavailable"
    return ""


def build(model, ctx, group):
    """Slot target_lm_head's stand-in (tuned), or None; ``st.bound["logits"]``."""
    from sgl_nvidia_mega.deepseek_v41.runtime import SLOTS

    if "target_lm_head" not in SLOTS:
        return None
    reason = target_lm_head_reason(model)
    if reason:
        logger.warning("NVIDIA megakernel: target LM head not built: %s", reason)
        return None
    head = TargetLmHead(model.lm_head, pinned="lm_head_pinned_tactic" in SLOTS)
    head.quant_method.tune()
    error = check_against_sglang(model, head)
    if not _all_ranks_true(group, not error):
        logger.warning(
            "NVIDIA megakernel: target LM head not built: %s",
            error or "another rank's logits differ from SGLang's head",
        )
        return None
    return head


def check_against_sglang(model, head) -> str:
    """The stand-in's logits against SGLang's head through SGLang's own
    ``_compute_lm_head`` on seeded random rows at every served row count ("" when
    within spec.NUMERICS); the logit scale, TP gather and buffers follow either."""
    from sgl_nvidia_mega.deepseek_v41.spec import NUMERICS

    tol = NUMERICS.tolerances["target_lm_head.logits.rel"]
    proc, w = model.logits_processor, head.weight
    gen = torch.Generator(device=w.device).manual_seed(0)
    for rows in CHECK_ROWS:
        x = torch.randn(rows, w.shape[1], generator=gen, device=w.device)
        x = x.to(torch.bfloat16)
        ours = proc._compute_lm_head(x, head)
        theirs = proc._compute_lm_head(x, model.lm_head)
        if ours.dtype != theirs.dtype or ours.shape != theirs.shape:
            return (
                f"it gives {ours.dtype} {tuple(ours.shape)}, SGLang's head "
                f"{theirs.dtype} {tuple(theirs.shape)} at {rows} rows"
            )
        rel = (
            (ours.float() - theirs.float()).abs().max() / theirs.float().abs().max()
        ).item()
        if not rel <= tol:
            return (
                f"its logits differ by {rel:.2e} of SGLang's peak at {rows} rows "
                f"(> {tol:.2e})"
            )
    return ""


def _all_ranks_true(group, value: bool) -> bool:
    """``value`` on every rank of SGLang's TP ``group`` (None or one rank: this one)."""
    if group is None or group.world_size == 1:
        return value
    flag = torch.tensor([int(value)], dtype=torch.int32)
    torch.distributed.all_reduce(
        flag, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
    )
    return bool(flag.item())


def head(st, residual, pre):
    """(hidden [rows, dim], pre_hc_head [rows, hc * dim]) from the final streams."""
    from sgl_nvidia_mega.deepseek_v41 import sglang_ops

    m = st.model
    pre_hc_head = residual.flatten(1)
    streams = pre_hc_head if sglang_ops.hc_combine_widens() else pre_hc_head.float()
    hidden = m.norm(sglang_ops.hc_combine(streams, pre, m.hc_mult, residual.dtype))
    return hidden, pre_hc_head


def logits(st, hidden, pre_hc_head, aux):
    """SGLang's ``LogitsProcessorOutput`` for the step."""
    lm = st.lm
    return lm.logits_processor(
        st.input_ids,
        hidden,
        st.bound.get("logits") or lm.lm_head,
        st.fb,
        aux,
        hidden_states_before_norm=None if aux is not None else pre_hc_head,
    )
