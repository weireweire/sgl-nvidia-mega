# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/deepseek_v4.py (DeepseekV4ForCausalLM.forward, DeepseekV4Model.forward and its layer loop).
"""The executor: one covered decode step of the DeepSeek-V4.1 target.

``MegakernelDecode.plan`` decides from host state alone, identical on every rank,
whether a step is covered (a DSpark verify step of a built width).
``MegakernelDecode.forward`` takes and returns what ``DeepseekV4ForCausalLM.forward``
does.
It runs the step from the plugin's step functions over SGLang's loaded modules and
weights, never through a model-level or decoder-layer ``forward``:

1. SGLang's input preparation, the embedding and the Engram hashes, then the mHC
   broadcast (``boundary.start``);
2. per layer: Engram (``engram.apply``) and the draft's capture on the materialized
   streams, then
   ``a_in = boundary.pre(layer, "attn", st)``, ``part = attention.step(layer, a_in, st)``,
   ``m_in = boundary.post(layer, "attn", part, st)``, ``out = moe.step(layer, m_in, st)``,
   ``boundary.post(layer, "ffn", out, st)``;
3. ``boundary.final``, the head combine + norm and the logits (``logits``).

``SGLANG_NVIDIA_MEGAKERNEL_PARITY`` (test only) runs SGLang's own forward on the same
eager step first and logs the largest difference of the logits and auxiliary hidden states
("control": SGLang's forward twice, the run-to-run floor).
"""

from __future__ import annotations

import logging
import os

import torch

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import Step
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.spec import Step

from sgl_nvidia_mega.deepseek_v41 import (
    attention,
    boundary,
    engram,
    logits,
    moe,
    sglang_ops,
)
from sgl_nvidia_mega.deepseek_v41.runtime import SLOTS, embedding
from sgl_nvidia_mega.deepseek_v41.runtime.state import AuxPack, StepState

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    # Step, and the BuiltMegakernel members SGLang reads
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "Step",
        "mode",
        "rows",
        "widths",
        "draft_sampler",
        "summary",
        "plan",
        "forward",
    ),
    "srt/layers/layer_boundary/__init__.py": ("get_attn_tp_context",),
    "srt/layers/layer_boundary/adapters/attention.py": (
        "get_attn_tp_context",
        "maybe_input_scattered",
    ),
    "kernels/ops/layernorm/mhc.py": ("hc_broadcast",),
    # DeepseekV4ForCausalLM.forward, which the executor stands in for
    "srt/models/deepseek_v4.py": (
        "prepare_language_model_inputs",
        "capture_aux_hidden_states",
        "model",
        "forward(input_ids)",
        "forward(positions)",
        "forward(forward_batch)",
        "forward(input_embeds)",
        "forward(input_ids_global)",
        "hc_mult",
        "start_layer",
        "end_layer",
        "engram",
        "dspark_layers_to_capture",
        "layers",
        '"freqs_cis_c4"',
        '"freqs_cis_c128"',
    ),
    "srt/layers/logits_processor.py": ("next_token_logits", "hidden_states"),
    "srt/eplb/expert_distribution.py": (
        "get_global_expert_distribution_recorder",
        "with_current_layer",
    ),
}

# Composition parity (test only), see the module docstring
PARITY_MODE = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_PARITY", "0")
PARITY = PARITY_MODE in ("1", "control")
# slot aux_mean_stats: the DSpark capture means on the mHC stats stream (bitwise)
AUX_MEAN_STATS = "aux_mean_stats" in SLOTS

PACKAGES = {
    "boundary": boundary,
    "attention": attention,
    "moe": moe,
    "engram": engram,
    "logits": logits,
}


class MegakernelDecode:
    """One rank's covered steps: the model, each width's slots and each package's
    build result."""

    # the target offers no speculative entries or graph variants (BuiltMegakernel)
    draft_sampler = None
    draft_loop = None
    draft_extend = None
    begin_draft_round = None
    graph_variants = None
    speculative_runtime_hooks = None

    def __init__(
        self,
        lm,
        slots: dict,
        bound: dict,
        widths: tuple[int, ...],
        summary: str = "",
        binding=None,
    ):
        self.lm, self.slots, self.bound = lm, slots, bound
        # SGLang's MegakernelBinding (BuildContext.binding)
        self.binding = binding
        self.widths = tuple(sorted(widths))
        self._summary = summary

    def summary(self) -> str:
        """What was built, for SGLang's startup line."""
        return self._summary

    def plan(
        self, input_ids, forward_batch, input_embeds=None, pp_proxy_tensors=None
    ) -> Step | None:
        """The step if this forward is covered, else None (SGLang's forward runs)."""
        from sgl_nvidia_mega.deepseek_v41.spec import serves_this_step

        mode = forward_batch.forward_mode
        rows = int(input_ids.shape[0])
        if pp_proxy_tensors is not None or not mode.is_target_verify():
            return None
        if not serves_this_step():
            return None
        return Step(mode="target_verify", rows=rows) if rows in self.widths else None

    def forward(
        self,
        step: Step,
        original,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
    ):
        """The covered step (``original``: the model's own forward, bound, for the
        parity check)."""
        if PARITY and not torch.cuda.is_current_stream_capturing():
            return self._with_parity(
                step, original, input_ids, positions, forward_batch, input_embeds
            )
        return self._forward(step, input_ids, positions, forward_batch, input_embeds)

    def _forward(self, step, input_ids, positions, forward_batch, input_embeds):
        """``DeepseekV4ForCausalLM.forward`` for a covered step."""
        from sglang.srt.layers.layer_boundary import get_attn_tp_context

        lm = self.lm
        input_ids, input_embeds = lm.prepare_language_model_inputs(
            input_ids, forward_batch, input_embeds
        )
        st = StepState(
            lm=lm,
            forward_batch=forward_batch,
            positions=positions,
            input_ids=input_ids,
            input_ids_global=getattr(forward_batch, "input_ids_global", input_ids),
            slots=self.slots[step.rows],
            bound=self.bound,
            view=step.view,
            binding=self.binding,
        )
        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden, pre_hc_head = self._model_forward(st, input_embeds)
        packed = lm.capture_aux_hidden_states and st.aux is not None
        aux = st.aux.packed() if packed else None
        return logits.logits(st, hidden, pre_hc_head, aux)

    def _model_forward(self, st, input_embeds):
        """``DeepseekV4Model.forward``: the embedding, the layers, the head."""
        from sglang.kernels.ops.layernorm.mhc import hc_broadcast

        m = st.model
        try:
            attention.head(st)
            x = embedding.embed(st) if input_embeds is None else input_embeds
            x = hc_broadcast(x, m.hc_mult)
            capture = m.dspark_layers_to_capture
            st.aux = AuxPack(x, len(capture)) if capture is not None else None
            # the compressor caches each step's rotary tables on the batch
            for attr in ("freqs_cis_c4", "freqs_cis_c128"):
                if hasattr(st.fb, attr):
                    delattr(st.fb, attr)
            residual, pre = self._layers(st, x)
        finally:
            boundary.clear(st)
        return logits.head(st, residual, pre)

    def _layers(self, st, x):
        """``DeepseekV4Model._forward_layers_hc_pre_from_prev`` for a covered step."""
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )

        m = st.model
        capture = m.dspark_layers_to_capture if st.aux is not None else ()
        engram.hashes(st)
        boundary.start(st, x)
        recorder = get_global_expert_distribution_recorder()
        for i in range(m.start_layer, m.end_layer):
            layer = m.layers[i]
            st.layer_id = i
            engram.before_layer(i, st)
            if layer.engram is not None or i in capture:
                boundary.materialize(layer, st)
            if layer.engram is not None:
                engram.apply(layer, st)
            # the draft head reads the attention input of its target layers: with slot
            # aux_mean_stats its mean runs on the mHC stats stream (joined in final)
            aux = st.aux if i in capture and AUX_MEAN_STATS else None
            if i in capture and aux is None:
                sglang_ops.stream_mean(st.state.residual, out=st.aux.take())
            with recorder.with_current_layer(i), st.exit:
                a_in = boundary.pre(layer, "attn", st, aux)
                part = attention.step(layer, a_in, st)
                m_in = boundary.post(layer, "attn", part, st)
                engram.before_moe(i, st)
                out = moe.step(layer, m_in, st)
                boundary.post(layer, "ffn", out, st)
        return boundary.final(st)

    def _with_parity(
        self, step, original, input_ids, positions, forward_batch, input_embeds
    ):
        """SGLang's forward, then the executor's, on the same step; logs the largest
        difference of the logits and the auxiliary hidden states."""
        want = original(input_ids, positions, forward_batch, input_embeds)
        want = (_clone(want.next_token_logits), _clone(want.hidden_states))
        if PARITY_MODE == "control":
            got = original(input_ids, positions, forward_batch, input_embeds)
        else:
            got = self._forward(step, input_ids, positions, forward_batch, input_embeds)
        diffs = [
            _max_abs(a, b)
            for a, b in zip(want, (got.next_token_logits, got.hidden_states))
        ]
        logger.info(
            "NVIDIA megakernel parity%s %s rows=%d logits_max_abs=%s hidden_max_abs=%s",
            " control" if PARITY_MODE == "control" else "",
            step.mode,
            step.rows,
            *diffs,
        )
        return got


def _clone(t):
    return None if t is None else t.clone()


def _max_abs(a, b):
    """max |a - b|, that over max |a|, and the fraction of elements that differ."""
    if a is None or b is None:
        return "n/a" if a is None and b is None else "missing"
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    d = (a.float() - b.float()).abs()
    peak = a.float().abs().max().clamp_min(1e-30)
    return (
        f"{d.max().item():.3g} (rel {(d.max() / peak).item():.2e}, "
        f"{(d > 0).float().mean().item():.2%} differ)"
    )
