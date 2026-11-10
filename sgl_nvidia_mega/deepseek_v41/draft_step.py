# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DSpark draft's forward on the megakernel's step packages (slot "draft").

``DraftDecode.forward`` stands in for ``DeepseekV4ForCausalLMDSpark.forward`` on a
covered draft block (TARGET_VERIFY, whole requests of the block's rows): the
embedding, every stage through ``boundary`` / the draft attention / ``moe``, and the
head's combine. Stages keep SGLang's draft semantics: no cross-stage fusion and,
until a slot says otherwise, SGLang's unfused posts.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import torch

from sgl_nvidia_mega.deepseek_v41 import boundary, draft_attention, draft_slots, moe
from sgl_nvidia_mega.deepseek_v41 import plan as slot_plan
from sgl_nvidia_mega.deepseek_v41.draft import DraftProposal
from sgl_nvidia_mega.deepseek_v41.runtime.state import StepState

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/deepseek_v4_dspark.py": (
        "stages",
        "forward_embed",
        "hc_mult",
        "sample_from_anchor",
        "_run_boundary_attention",
        "uses_own_vocab_modules",
        "attach_shared_modules",
    ),
    "srt/models/deepseek_v4.py": ("maybe_use_decode_attn_tp", "self_attn", "layer_id"),
    "srt/layers/logits_processor.py": ("LogitsProcessorOutput", "hidden_states"),
    "srt/eplb/expert_distribution.py": (
        "get_global_expert_distribution_recorder",
        "disable_this_region",
    ),
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": ("Step",),
    "srt/runtime_context.py": ("get_spec", "get_parallel", '"tp_group"'),
    "srt/arg_groups/fields/*.py": ("speculative_num_draft_tokens",),
    "srt/speculative/dspark_components/dspark_config.py": (
        "dspark_gamma_from_num_draft_tokens",
    ),
}

# Composition parity (test only): SGLang's draft forward, then this one, on each eager
# covered step, logging the largest difference of the hidden states
PARITY_MODE = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_PARITY", "0")
PARITY = PARITY_MODE in ("1", "control")

# The draft blocks' widest covered step: 8 requests of up to 6 rows
DRAFT_MAX_ROWS = 48  # = draft._max_rows() with slot "draft"


def tokens_per_request(lm) -> int:
    """A draft block's rows a request: gamma, or gamma + 1 without the anchor."""
    from sglang.srt.runtime_context import get_spec
    from sglang.srt.speculative.dspark_components.dspark_config import (
        dspark_gamma_from_num_draft_tokens,
    )

    gamma = dspark_gamma_from_num_draft_tokens(
        int(get_spec().speculative_num_draft_tokens)
    )
    return gamma if lm.sample_from_anchor else gamma + 1


class _DraftLm:
    """The draft as ``StepState`` reads a model: its stages as the layers."""

    def __init__(self, lm):
        self.lm = lm
        stages = list(lm.stages)
        self.model = SimpleNamespace(
            layers=stages, start_layer=0, end_layer=len(stages)
        )


class DraftDecode(DraftProposal):
    """The draft spec's built object with slot "draft": the draft forward at the
    built widths, and the proposal (``draft_sampler``) as before."""

    def __init__(self, lm, *, deadline_s: float, slots: frozenset[str], binding=None):
        super().__init__(lm, deadline_s=deadline_s, slots=slots)
        self._binding = binding
        self._tokens = tokens = tokens_per_request(lm)
        self.widths = tuple(range(tokens, DRAFT_MAX_ROWS + 1, tokens))
        self._view = _DraftLm(lm)
        stages = self._view.model.layers
        moe_bound = moe.routed.Bound(reason="stock (slot draft_moe off)")
        if "draft_moe" in slots:
            from sglang.srt.runtime_context import get_parallel

            moe_bound = moe.routed.build_layers(
                stages, get_parallel().tp_group, gate_topk="draft_gate" in slots
            )
        # slot draft_hop: the posts on the plugin's Lamport hop where SGLang has no
        # fused post (TP8 with EP2)
        hop, self._hop_reason = None, "not requested"
        if "draft_post" in slots and "draft_hop" in slots:
            from sgl_nvidia_mega.deepseek_v41.boundary import draft_hop

            hop, self._hop_reason = draft_hop.build(
                stages, deadline_s=deadline_s, max_rows=DRAFT_MAX_ROWS
            )
        self._bound = {
            # slot draft_post: SGLang's fused all-reduce + mHC posts on the stages;
            # draft_seam: each stage's MoE post also writes the next stage's input;
            # draft_prenorm: the mHC coefficients from the target's DeepGEMM prenorm
            "boundary": boundary.Boundaries(
                stages,
                cross_layer="draft_post" in slots and "draft_seam" in slots,
                mhc_fusions=True if "draft_post" in slots else None,
                prenorm=_prenorm(slots),
                draft_hop=hop,
            ),
            "moe": moe_bound,
        }
        # slot draft_embed: the draft's tokens from the target's gathered table
        self._embed_table = _embed_table(lm, slots)
        self._embed_fell_back = False
        routed = bool(moe_bound.slots)
        self._slots = {
            w: slot_plan.Slots(
                hop_a=False,
                fast_gate=routed
                and w in slot_plan.fast_gate_rows("gate_wide" in slots),
                routed=routed and w <= slot_plan.ROUTED_MAX_ROWS,
            )
            for w in self.widths
        }
        # hc_combine widens each load to FP32: the BF16 streams give the same result
        self._tail_bf16 = "draft_tail" in slots
        # the ported attention reads the draft forward's KV pool and metadata through
        # the binding; without one (an SGLang that offers none) SGLang's call runs
        self._attn = draft_attention.build(stages, slots) if binding is not None else {}
        if self._attn and torch.cuda.is_available():
            draft_attention.precompile(stages, self._attn, self.widths)
        self.slot_states = draft_slots.states(self, slots)
        draft_slots.warn_declined(self.slot_states)

    def _embed(self, input_ids):
        """``forward_embed``: the lookup broadcast to the hc streams, from the whole
        table without the TP all-reduce when slot draft_embed is built."""
        from sgl_nvidia_mega.deepseek_v41.runtime import embedding

        lm = self.lm
        if self._embed_table is _SHARED_TABLE:
            self._embed_table = _shared_table(lm.embed_tokens)
            logger.info(
                "NVIDIA megakernel: draft_embed bound the target's whole table %s",
                tuple(self._embed_table.shape),
            )
        if self._embed_table is not None:
            x = embedding._local(lm.embed_tokens, self._embed_table, input_ids)
            if x is not None:
                return x.unsqueeze(1).repeat(1, lm.hc_mult, 1)
            if not self._embed_fell_back:
                # SGLang's sharded lookup for input its fused lookup does not take
                self._embed_fell_back = True
                logger.info(
                    "NVIDIA megakernel: draft_embed: a step took SGLang's lookup"
                )
        return lm.forward_embed(input_ids)

    def summary(self) -> str:
        moe_bound = self._bound["moe"]
        return (
            f"DSpark draft forward: routed pair in {len(moe_bound.slots)} stages "
            f"({moe_bound.reason or 'built'}), fast gate in "
            f"{sum(s.gate for s in moe_bound.slots.values())}, attention ported in "
            f"{len(self._attn)} stages ({_kernel_counts(self._attn.values())}), "
            f"embedding {_embed_summary(self._embed_table)}"
            f", mHC prenorm {'on' if self._bound['boundary'].prenorm else 'off'}, BF16 tail "
            f"{'on' if self._tail_bf16 else 'off'}; {super().summary()}"
        )

    def plan(self, input_ids, forward_batch, *args, **kwargs):
        try:
            from sglang.kernels.plugin_kernels.nvidia.spec import Step
        except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
            from sglang.srt.megakernels.spec import Step

        from sgl_nvidia_mega.deepseek_v41.spec import serves_this_step

        if not forward_batch.forward_mode.is_target_verify() or not serves_this_step():
            return None
        rows = int(input_ids.shape[0])
        return Step(mode="target_verify", rows=rows) if rows in self.widths else None

    def forward(
        self,
        step,
        original,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        *args,
        **kwargs,
    ):
        if PARITY and not torch.cuda.is_current_stream_capturing():
            want = original(input_ids, positions, forward_batch, input_embeds)
            want = want.hidden_states.clone()
            if PARITY_MODE == "control":
                # SGLang's draft against itself: the floor of its own run-to-run noise
                got = original(input_ids, positions, forward_batch, input_embeds)
            else:
                got = self._forward(
                    step, input_ids, positions, forward_batch, input_embeds
                )
            d = (want.float() - got.hidden_states.float()).abs()
            logger.info(
                "NVIDIA megakernel parity%s draft rows=%d hidden_max_abs=%.3g "
                "(%.2f%% differ)",
                " control" if PARITY_MODE == "control" else "",
                step.rows,
                d.max().item(),
                100.0 * (d > 0).float().mean().item(),
            )
            return got
        return self._forward(step, input_ids, positions, forward_batch, input_embeds)

    def _forward(self, step, input_ids, positions, forward_batch, input_embeds):
        """``DeepseekV4ForCausalLMDSpark.forward`` for a covered draft block."""
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        from sgl_nvidia_mega.deepseek_v41 import sglang_ops

        lm = self.lm
        # the draft forward's KV pool and attention metadata, for the ported attention
        view = self._binding.forward_view() if self._attn else None
        st = StepState(
            lm=self._view,
            forward_batch=forward_batch,
            positions=positions,
            input_ids=input_ids,
            input_ids_global=input_ids,
            slots=self._slots[step.rows],
            bound=self._bound,
        )
        try:
            x = self._embed(input_ids) if input_embeds is None else input_embeds
            boundary.start(st, x)
            # draft stages have no recorder layer indices
            with get_global_expert_distribution_recorder().disable_this_region():
                for stage in lm.stages:
                    st.layer_id = stage.layer_id
                    with st.exit:
                        a_in = boundary.pre(stage, "attn", st)
                        part = attention_step(stage, a_in, st, self._attn, view)
                        m_in = boundary.post(stage, "attn", part, st)
                        out = moe.step(stage, m_in, st)
                        boundary.post(stage, "ffn", out, st)
            residual, pre = boundary.final(st)
        finally:
            boundary.clear(st)
        streams = residual.flatten(1)
        if not self._tail_bf16:
            streams = streams.float()
        x = sglang_ops.hc_combine(streams, pre, lm.hc_mult, residual.dtype)
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=x)


def _prenorm(slots) -> bool:
    """Slot draft_prenorm: a load-time fact alike on every rank; requested and not
    servable raises, so a run never measures it as on."""
    if "draft_prenorm" not in slots:
        return False
    from sgl_nvidia_mega.deepseek_v41.boundary import prenorm

    if why := prenorm.reason():
        raise RuntimeError(f"NVIDIA megakernel: slot draft_prenorm: {why}")
    return True


# slot draft_embed before its first forward: SGLang attaches the target's embedding to
# the draft only after the draft's build, so the table is looked up then
_SHARED_TABLE = object()


def _embed_table(lm, slots):
    """Slot draft_embed: ``_SHARED_TABLE`` (the target's gathered table, looked up at the
    first forward), or None without the slot."""
    if "draft_embed" not in slots:
        return None
    if lm.uses_own_vocab_modules:
        # attach_shared_modules then keeps the draft's own table: a second whole one
        raise RuntimeError(
            "NVIDIA megakernel: slot draft_embed: the draft has its own vocabulary "
            "modules, not the target's"
        )
    return _SHARED_TABLE


def _embed_summary(table) -> str:
    if table is None:
        return "sharded"
    # requested here; the bind line at the first forward shows the table bound
    return (
        "whole table (binds at the first forward)"
        if table is _SHARED_TABLE
        else "whole table"
    )


def _shared_table(embed_tokens):
    """The table the target's embed slot gathered for this (shared) module; a host
    lookup, the same on every rank. None of it raises, so the slot never runs off."""
    from sgl_nvidia_mega.deepseek_v41.runtime import embedding

    table = embedding.gathered_table(embed_tokens)
    if table is None:
        raise RuntimeError(
            "NVIDIA megakernel: slot draft_embed: the target's embed slot gathered no "
            f"table for the draft's embedding ({type(embed_tokens).__name__})"
        )
    return table


def attention_step(stage, x, st, ported, view=None):
    """The draft stage's attention (``DSparkAttention.forward`` with the sum deferred to
    the post): the plugin's port with the stage's kernels over the draft forward's
    ``view`` (the binding's ``ForwardView``), or SGLang's own call."""
    attn = stage.self_attn
    with attn.maybe_use_decode_attn_tp(st.fb):
        kernels = ported.get(stage.layer_id)
        if kernels is None:
            return stage._run_boundary_attention(x, st.positions, st.fb, st.x_quant)
        return draft_attention.forward(
            attn, x, st.positions, st.fb, kernels=kernels, view=view
        )


def _kernel_counts(kernels) -> str:
    kernels = list(kernels)
    return ", ".join(
        f"{name} {sum(bool(getattr(k, name)) for k in kernels)}"
        for name in draft_attention.Kernels._fields
    )
