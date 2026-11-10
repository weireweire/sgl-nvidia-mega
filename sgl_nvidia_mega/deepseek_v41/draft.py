# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4.1-Flash's DSpark draft on SM103 (GB300): the megakernel draft spec.

With slot "draft" the draft's forward runs on the megakernel's steps
(``draft_step.DraftDecode``); otherwise it stays SGLang's. Either way the spec offers
SGLang's folded proposal a ``draft_sampler`` (``sampler.build_steps``).
"""

from __future__ import annotations

from typing import Callable, Optional

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import (
        Feature,
        MegakernelSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.spec import (
        Feature,
        MegakernelSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )

from sgl_nvidia_mega.deepseek_v41 import spec as target

# The SGLang names this module reads, by file under the sglang package;
# test/unit/test_contract.py pins them.
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "Feature",
        "MegakernelSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
    ),
}

NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        # the draft tokens SGLang's Markov block picks from the same base logits
        "draft_markov.tokens.max_abs": 0.0,
        # the kept step logits: one BF16 ulp of the step's largest bias (MMA vs GEMM order)
        "draft_markov.step_logits.rel": 2**-7,
        # the draft's base logits on the low-M kernel: the K summation order differs
        "draft_lm_head.logits.rel": 2**-7,
        # slot draft_lm_head_mxfp8 (measurement only): the MXFP8 stream against the BF16
        # head; the weight's quantization (test_deepseek_v41_lm_head_mxfp8.py)
        "draft_lm_head_mxfp8.logits.rel": 4e-2,
    },
)


class DraftProposal:
    """The draft spec's built object: SGLang's forward on every step, and the proposal."""

    widths: tuple[int, ...] = ()
    draft_loop = None
    draft_extend = None
    begin_draft_round = None
    graph_variants = None
    speculative_runtime_hooks = None

    def __init__(self, lm, *, deadline_s: float, slots: frozenset[str]):
        self.lm, self.deadline_s = lm, deadline_s
        self.low_m = "draft_lm_head" in slots
        self.mxfp8 = self.low_m and "draft_lm_head_mxfp8" in slots
        # SGLang's proposal runs when the markov slot is off
        self.draft_sampler: Optional[Callable] = (
            self._draft_sampler if "markov" in slots else None
        )

    def _draft_sampler(self, **buffers) -> Callable[[int], Optional[Callable]]:
        from sgl_nvidia_mega.deepseek_v41 import sampler

        steps = sampler.build_steps(
            self.lm,
            low_m=self.low_m,
            deadline_s=self.deadline_s,
            **buffers,
        )
        return steps.get

    def summary(self) -> str:
        if self.draft_sampler is None:
            return "DSpark draft: SGLang's proposal (markov slot off)"
        lm_head = "low-M" if self.low_m else "SGLang's"
        if self.mxfp8:
            lm_head += " (MXFP8 stream at 1-8 rows, measurement only)"
        return f"DSpark draft proposal: Markov chain in one kernel, LM head {lm_head}"

    def plan(self, input_ids, forward_batch, *args, **kwargs):
        return None

    def forward(self, step, original, *args, **kwargs):
        raise AssertionError("the draft covers no step width")


def _build(model, hf_config, ctx) -> DraftProposal:
    from sgl_nvidia_mega.deepseek_v41.runtime import SLOTS

    if "draft" in SLOTS:
        from sgl_nvidia_mega.deepseek_v41.draft_step import DraftDecode

        built = DraftDecode(
            model,
            deadline_s=ctx.deadline_s,
            slots=SLOTS,
            # SGLang's runtime state for the draft forward (megakernels/binding.py)
            binding=getattr(ctx, "binding", None),
        )
    else:
        built = DraftProposal(model, deadline_s=ctx.deadline_s, slots=SLOTS)
    return built


def _max_rows() -> int:
    """The draft forward's widest step with slot "draft", else 0 (SGLang's forward)."""
    from sgl_nvidia_mega.deepseek_v41.runtime import SLOTS

    return 48 if "draft" in SLOTS else 0


SPEC = MegakernelSpec(
    name="deepseek_v41_dspark_draft_nvidia",
    # the target's device, TP size, dtype and decode graphs: the draft shares its GPUs
    requires=Requirements(
        isas=target.SPEC.requires.isas,
        tp_sizes=target.SPEC.requires.tp_sizes,
        dtypes=target.SPEC.requires.dtypes,
        decode_graph_backends=target.SPEC.requires.decode_graph_backends,
    ),
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    numerics=NUMERICS,
    # SGLang's draft forward unless slot "draft" covers it
    max_rows=_max_rows(),
    build=_build,
    # the contract version this package is written against, as the target's
    api_version=target.API_VERSION,
)
