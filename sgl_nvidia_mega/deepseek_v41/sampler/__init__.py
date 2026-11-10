# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4.1-Flash's DSpark draft proposal on SM103 (GB300): the draft LM head and
the Markov chain of every draft step in one vocab-sharded kernel per rank.

SGLang's folded proposal (``DsparkDraftSampler``, the draft CUDA graph's tail) asks the
megakernel's draft spec for its ``draft_sampler`` once, on every rank, with its own
buffers; ``build_steps`` returns the step for each batch size the chain serves, which
replaces ``compute_base_logits`` + ``draft_markov_block`` there. The step projects the
draft hidden states through this rank's LM-head shard (FlashInfer's low-M BF16 kernel
at up to ``LM_HEAD_MAX_ROWS`` rows, SGLang's projection above), then runs the chain
(``sm103/markov.py``): ranks agree on each step's token inside the kernel, so the
proposal needs no per-step TP sync. The draft noise is SGLang's (same Philox key,
counters and offset), drawn on a side stream while the LM head runs.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads, by file under the sglang package (``f(p)``: a
# parameter); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/runtime_context.py": ("get_parallel",),
    "srt/distributed/parallel_state.py": ("rank_in_group", "world_size", "cpu_group"),
    "srt/models/deepseek_v4_dspark.py": (
        "markov_head",
        "markov_w1",
        "markov_w2",
        "vocab_size",
        "step_logits_dtype",
        "lm_head",
        "stages",
        "norm",
        "_use_fp32_lm_head",
        "compute_base_logits",
        "collapse_hc_head",
    ),
    "srt/layers/vocab_parallel_embedding.py": (
        "shard_indices",
        "org_vocab_start_index",
        "num_embeddings_per_partition",
        "weight",
    ),
    "srt/models/dspark.py": ("project_through_lm_head",),
    "srt/speculative/dspark_components/dspark_draft.py": (
        "key",
        "offset",
        "DraftNoise",
        "draft_markov_block",
    ),
}

# Test only: inside each captured proposal, also run SGLang's own base logits + Markov
# block on the same inputs into scratch buffers and count the tokens and kept logits
# that differ; every rank logs the counts every 30 s
PARITY = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_MARKOV_PARITY", "0") == "1"

# GB300 TP4 shard (N 32320, K 5120), CUDA graph replay: the low-M kernel beats cuBLAS at
# 5-15 rows (47.4 vs 55.3 us at 5) and loses at 20-24 (55.0 vs 51.6 us)
LM_HEAD_MAX_ROWS = 16
# the low-M logits against SGLang's projection: only the K summation order differs
LM_HEAD_LOGITS_REL = 2**-7
LM_HEAD_BACKEND = "cute-dsl"


def disable_reason(lm, *, gamma: int, world: int, rank: int) -> str:
    """Why the chain cannot serve this draft, or ""; the same on every rank."""
    from sgl_nvidia_mega.deepseek_v41.sm103 import markov

    head, lm_head = lm.markov_head, lm.lm_head
    if lm_head is None:
        return "the draft has no LM head attached"
    if lm._use_fp32_lm_head:
        return "an FP32 draft LM head (SGLANG_DSPARK_FP32_LM_HEAD)"
    vocab = int(head.vocab_size)
    # the chain reads this rank's base logits, the LM head's even vocab split
    shard = lm_head.shard_indices
    if int(lm_head.num_embeddings_per_partition) != vocab // world or int(
        shard.org_vocab_start_index
    ) != rank * (vocab // world):
        return "an LM-head vocab partition that is not the even split"
    if head.markov_w2.weight.dtype != torch.bfloat16:
        # SGLang's FP32 bias path rounds differently from the kernel's BF16 table
        return "an FP32 markov_w2 (set SGLANG_DSPARK_OPT_MARKOV_W2_BF16=1)"
    return (
        markov.unsupported_reason(
            world=world,
            vocab_size=vocab,
            num_steps=gamma,
            embed_weight=head.markov_w1.weight,
            head_weight=head.markov_w2.weight,
            cache_dtype=head.step_logits_dtype(torch.bfloat16),
            capability=torch.cuda.get_device_capability(),
        )
        or ""
    )


def buffer_reason(*, max_bs: int, gamma: int, draft_tokens, step_temperatures) -> str:
    """Why SGLang's buffers are not the ones the chain writes into, or ""."""
    rows, tokens, temperatures = max_bs, draft_tokens, step_temperatures
    if tokens.dtype != torch.int64 or tuple(tokens.shape) != (rows * gamma,):
        return f"draft tokens {tokens.dtype} {tuple(tokens.shape)}, not int64 [{rows * gamma}]"
    if temperatures.dtype != torch.float32 or tuple(temperatures.shape) != (rows,):
        return (
            f"step temperatures {temperatures.dtype} {tuple(temperatures.shape)}, "
            f"not float32 [{rows}]"
        )
    return ""


def low_m_lm_head_supported() -> bool:
    major, minor = torch.cuda.get_device_capability()
    try:
        from flashinfer import mm_bf16

        return bool(mm_bf16.is_backend_supported(LM_HEAD_BACKEND, major * 10 + minor))
    except (ImportError, AttributeError):
        return False


class DraftLmHead:
    """This rank's LM-head shard for the draft's rows: FlashInfer's low-M BF16 kernel
    at 1-LM_HEAD_MAX_ROWS rows, SGLang's projection otherwise. With slot
    draft_lm_head_mxfp8 (measurement only, plan.MEASUREMENT_ONLY_SLOTS) the 1-8 rows
    stream the shard's MXFP8 copy, the one the target's head reads."""

    def __init__(self, lm_head, *, low_m: bool, pinned: bool = False, mxfp8=None):
        self.lm_head = lm_head
        # the CuTe DSL kernels take operands over DLPack, which refuses grad tensors
        self.weight_t = lm_head.weight.detach().t()
        self.low_m = low_m and lm_head.weight.dtype == torch.bfloat16
        # slot lm_head_pinned_tactic: one fixed tactic per row count, no tuning
        self.pinned = pinned
        # slot draft_lm_head_mxfp8: lm_head_mxfp8.Mxfp8Weight of this shard
        self.mxfp8 = mxfp8 if self.low_m else None

    def takes(self, rows: int) -> bool:
        return self.low_m and rows <= LM_HEAD_MAX_ROWS

    def streams(self, rows: int) -> bool:
        from sgl_nvidia_mega.deepseek_v41.sm103 import lm_head_mxfp8

        return self.mxfp8 is not None and rows <= lm_head_mxfp8.MAX_ROWS

    def kind(self, rows: int) -> str:
        if self.streams(rows):
            return "MXFP8 stream"
        return "low-M" if self.takes(rows) else "SGLang's"

    def tune(self, rows: tuple[int, ...]) -> None:
        """Tune the low-M kernel at the row counts the draft runs (compile the MXFP8
        stream's); the same calls on every rank, before any graph is captured."""
        if self.mxfp8 is not None:
            from sgl_nvidia_mega.deepseek_v41.sm103 import lm_head_mxfp8

            lm_head_mxfp8.precompile(*self.mxfp8.shape, self.lm_head.weight.device)
        if self.pinned:
            return
        from flashinfer.autotuner import autotune

        weight = self.lm_head.weight
        with autotune(True):
            for r in rows:
                if self.takes(r) and not self.streams(r):
                    self(
                        torch.zeros(
                            r, weight.shape[1], dtype=weight.dtype, device=weight.device
                        )
                    )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.streams(x.shape[0]):
            from sgl_nvidia_mega.deepseek_v41.sm103 import lm_head_mxfp8

            return lm_head_mxfp8.matmul(x.contiguous(), self.mxfp8)
        if not self.takes(x.shape[0]):
            from sglang.srt.models.dspark import project_through_lm_head

            return project_through_lm_head(x, self.lm_head)
        if self.pinned:
            from sgl_nvidia_mega.deepseek_v41.sampler.pinned import pinned_mm

            return pinned_mm(x.contiguous(), self.weight_t)
        from flashinfer import mm_bf16

        return mm_bf16(x.contiguous(), self.weight_t, pdl=True, backend=LM_HEAD_BACKEND)


def build_steps(
    lm,
    *,
    low_m: bool,
    deadline_s: float,
    gamma: int,
    max_bs: int,
    greedy_draft: bool,
    noise,
    draft_tokens: torch.Tensor,
    step_logits: Optional[torch.Tensor],
    step_temperatures: torch.Tensor,
) -> dict:
    """The per-batch-size steps over SGLang's buffers; collective over the TP group, so
    every rank runs it. Raises on every rank when any rank cannot run the chain, or
    when the ranks would serve different sizes (their graphs would wait on each other)."""
    from sglang.srt.runtime_context import get_parallel

    from sgl_nvidia_mega.deepseek_v41.sm103 import markov

    group = get_parallel().tp_group
    world, rank = group.world_size, group.rank_in_group
    # one rank: plain device memory and no collectives
    cpu_group = None if world == 1 else group.cpu_group
    head = lm.markov_head
    reason = disable_reason(lm, gamma=gamma, world=world, rank=rank)
    reasons = _gather(
        cpu_group,
        reason
        or buffer_reason(
            max_bs=max_bs,
            gamma=gamma,
            draft_tokens=draft_tokens,
            step_temperatures=step_temperatures,
        ),
    )
    refusal = refusal_message(reasons)
    if refusal:
        raise ValueError(refusal)
    cache_dtype = head.step_logits_dtype(torch.bfloat16)
    # symmetric memory is invisible to memory_allocated
    free_before, _ = torch.cuda.mem_get_info()
    chain = markov.ShardedMarkov(
        embed_weight=head.markov_w1.weight,
        head_weight=head.markov_w2.weight,
        vocab_size=int(head.vocab_size),
        num_steps=gamma,
        cache_dtype=cache_dtype,
        group=cpu_group,
        rank=rank,
        world=world,
        max_bs=max_bs,
        deadline_s=deadline_s,
        keeps_logits=not greedy_draft,
    )
    from sgl_nvidia_mega.deepseek_v41.runtime import SLOTS

    lm_head = DraftLmHead(
        lm.lm_head,
        low_m=low_m and low_m_lm_head_supported(),
        pinned="lm_head_pinned_tactic" in SLOTS,
        mxfp8=draft_mxfp8_copy(lm.lm_head.weight, SLOTS),
    )
    mismatch = mismatch_message(
        _gather(
            cpu_group,
            (chain.max_rows, lm_head.low_m, lm_head.mxfp8 is not None),
        )
    )
    if mismatch:
        raise RuntimeError(mismatch)
    lm_head.tune(tuple(bs * gamma for bs in range(1, chain.max_rows + 1)))
    steps = {
        bs: _Step(
            lm=lm,
            chain=chain,
            lm_head=lm_head,
            bs=bs,
            gamma=gamma,
            greedy_draft=greedy_draft,
            noise=noise,
            draft_tokens=draft_tokens,
            step_logits=step_logits,
            step_temperatures=step_temperatures,
        )
        for bs in range(1, chain.max_rows + 1)
    }
    logger.info(
        "NVIDIA megakernel: draft proposal on rank %d: Markov chain in one kernel "
        "(%d ranks, batch 1-%d, %d draft tokens), LM head %s at 1-%d rows%s; wait "
        "deadline %.0f s; %.1f MiB of device memory",
        rank,
        world,
        chain.max_rows,
        gamma,
        "low-M" if lm_head.low_m else "SGLang's",
        LM_HEAD_MAX_ROWS,
        " (MXFP8 stream at 1-8)" if lm_head.mxfp8 is not None else "",
        chain.timeout_ns / 1e9,
        (free_before - torch.cuda.mem_get_info()[0]) / 2**20,
    )
    return steps


def draft_mxfp8_copy(weight: torch.Tensor, slots):
    """Slot draft_lm_head_mxfp8's MXFP8 copy of the shard (the target head's, when that
    head streams too), or None; build_steps refuses ranks that disagree."""
    if "draft_lm_head_mxfp8" not in slots:
        return None
    from sgl_nvidia_mega.deepseek_v41.logits import mxfp8_copy
    from sgl_nvidia_mega.deepseek_v41.sm103 import lm_head_mxfp8

    reason = lm_head_mxfp8.unsupported_reason(weight)
    copy = None if reason else mxfp8_copy(weight)
    reason = reason or lm_head_mxfp8.scale_reason(copy)
    if reason:
        logger.warning(
            "NVIDIA megakernel: slot draft_lm_head_mxfp8 not built: %s", reason
        )
        return None
    logger.warning(
        "NVIDIA megakernel: slot draft_lm_head_mxfp8 is on: the draft's LM head reads "
        "an MXFP8 copy below the checkpoint's BF16; for measurement only"
    )
    return copy


class _Step:
    """One batch size's proposal: LM head, then the chain."""

    def __init__(
        self,
        *,
        lm,
        chain,
        lm_head,
        bs,
        gamma,
        greedy_draft,
        noise,
        draft_tokens,
        step_logits,
        step_temperatures,
    ):
        self.lm, self.chain, self.lm_head = lm, chain, lm_head
        self.bs, self.gamma, self.greedy_draft, self.noise = (
            bs,
            gamma,
            greedy_draft,
            noise,
        )
        self.tokens = draft_tokens[: bs * gamma].view(bs, gamma)
        self.step_logits = step_logits
        self.temperatures = step_temperatures
        self.is_logged = False

    def __call__(self, hidden_states: torch.Tensor, anchor_tokens: torch.Tensor):
        if not self.is_logged and torch.cuda.is_current_stream_capturing():
            # engagement, through the model's dispatch, once per captured graph
            logger.info(
                "NVIDIA megakernel: draft proposal engaged: batch %d, %d LM-head "
                "rows (%s)",
                self.bs,
                hidden_states.shape[0],
                self.lm_head.kind(hidden_states.shape[0]),
            )
            self.is_logged = True
        if not self.greedy_draft:
            self.chain.fork_noise(
                rows=self.bs, key=self.noise.key, offset=self.noise.offset
            )
        # as compute_base_logits: identity (with a 2-D assert) when the draft collapses
        # its hc streams in forward(), as V4.1's does
        hidden_states = self.lm.collapse_hc_head(hidden_states)
        x = self.lm.stages[-1].norm(hidden_states)
        base_logits = self.lm_head(x).view(self.bs, self.gamma, -1)
        if not self.greedy_draft:
            self.chain.join_noise()
        self.chain.sample(
            base_logits=base_logits,
            anchor_tokens=anchor_tokens,
            temperatures=self.temperatures,
            logits_cache=self.step_logits,
            draft_tokens=self.tokens,
        )
        if not self.greedy_draft:
            # one block advances the counter once, as draft_markov_block does
            self.noise.offset.add_(1)
        if PARITY:
            _parity(self.lm).check(self, hidden_states, anchor_tokens)
        return self.tokens, hidden_states


class _Parity:
    """Test only (``PARITY``): SGLang's proposal beside the plugin's, counted on the GPU."""

    def __init__(self, lm, rank: int):
        self.lm, self.rank = lm, rank
        device = lm.lm_head.weight.device
        # [blocks, token mismatches, kept-logit mismatches at 1-8 rows, at 9+ rows]
        self.counts = torch.zeros(4, dtype=torch.int64, device=device)
        # [max |kept-logit difference|, max |kept logit|]
        self.extent = torch.zeros(2, dtype=torch.float32, device=device)
        self.offset = torch.zeros(1, dtype=torch.int64, device=device)
        self.logits = None
        # the reporter reads host memory only: no device sync while graphs capture
        self.host_counts = torch.zeros(4, dtype=torch.int64).pin_memory()
        self.host_extent = torch.zeros(2, dtype=torch.float32).pin_memory()
        threading.Thread(target=self._report, daemon=True).start()

    def check(self, step: _Step, hidden_states, anchor_tokens) -> None:
        from sglang.srt.speculative.dspark_components.dspark_draft import (
            DraftNoise,
            draft_markov_block,
        )

        bs, gamma = step.bs, step.gamma
        # the plugin advanced the counter after drawing at offset - 1
        self.offset.copy_(step.noise.offset - (0 if step.greedy_draft else 1))
        base, _ = self.lm.compute_base_logits(hidden_states)
        logits_out = None
        if not step.greedy_draft:
            if self.logits is None or self.logits.shape[0] < bs:
                assert (
                    not torch.cuda.is_current_stream_capturing()
                    or self.logits is not None
                )
                self.logits = torch.empty_like(step.step_logits)
            logits_out = self.logits[:bs]
        tokens, _ = draft_markov_block(
            markov_head=self.lm.markov_head,
            base_logits=base.view(bs, gamma, -1),
            anchor_tokens=anchor_tokens,
            draft_hidden=hidden_states.view(bs, gamma, -1),
            step_temperatures=step.temperatures[:bs],
            noise=DraftNoise(key=step.noise.key, offset=self.offset),
            tp_sync=_NoSync(),
            logits_out=logits_out,
            greedy_draft=step.greedy_draft,
        )
        self.counts[0].add_(1)
        self.counts[1].add_((tokens != step.tokens).sum())
        if logits_out is not None:
            # temperature-0 rows keep no logits on the chain
            kept = (step.temperatures[:bs] > 0).view(bs, 1, 1)
            differ = (logits_out != step.step_logits[:bs]) & kept
            self.counts[2 if bs <= 8 else 3].add_(differ.sum())
            delta = (logits_out - step.step_logits[:bs]).abs() * kept
            reference = logits_out.abs().nan_to_num(posinf=0.0) * kept
            torch.maximum(
                self.extent[:1], delta.nan_to_num(0.0).amax(), out=self.extent[:1]
            )
            torch.maximum(self.extent[1:], reference.amax(), out=self.extent[1:])
        self.host_counts.copy_(self.counts, non_blocking=True)
        self.host_extent.copy_(self.extent, non_blocking=True)

    def _report(self) -> None:
        import time

        last = None
        while True:
            time.sleep(30)
            counts = self.host_counts.tolist() + self.host_extent.tolist()
            if counts != last:
                logger.info(
                    "NVIDIA megakernel: Markov parity rank %d: blocks %d, token mismatches "
                    "%d, kept-logit mismatches %d at 1-8 rows and %d at 9+ rows, max "
                    "|difference| %.3g, max |logit| %.3g",
                    self.rank,
                    *counts,
                )
                last = counts


class _NoSync:
    def sync(self, site, values):
        return values


_PARITIES: dict = {}


def _parity(lm) -> _Parity:
    if id(lm) not in _PARITIES:
        from sglang.srt.runtime_context import get_parallel

        _PARITIES[id(lm)] = _Parity(lm, get_parallel().tp_group.rank_in_group)
    return _PARITIES[id(lm)]


def refusal_message(reasons: list[str]) -> str:
    """Every rank's reason the ``markov`` slot cannot run, or "" when none has one."""
    found = [f"rank {rank}: {reason}" for rank, reason in enumerate(reasons) if reason]
    if not found:
        return ""
    return "NVIDIA megakernel: the markov slot cannot run: " + "; ".join(found)


def mismatch_message(built: list[tuple]) -> str:
    """ "" when every rank built the same (batch sizes, low-M LM head, MXFP8 stream),
    else the split."""
    if len(set(built)) <= 1:
        return ""
    return "NVIDIA megakernel: the ranks built different draft proposals: " + "; ".join(
        f"rank {rank}: batch 1-{rows}, low-M LM head {low_m}"
        + (f", MXFP8 stream {mxfp8[0]}" if mxfp8 else "")
        for rank, (rows, low_m, *mxfp8) in enumerate(built)
    )


def _gather(cpu_group, value) -> list:
    if cpu_group is None:
        return [value]
    import torch.distributed as dist

    values = [None] * dist.get_world_size(cpu_group)
    dist.all_gather_object(values, value, group=cpu_group)
    return values
