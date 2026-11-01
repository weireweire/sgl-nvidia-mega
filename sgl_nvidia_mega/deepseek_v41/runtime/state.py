# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``StepState`` (``st``): what one covered decode step carries between its steps.

The runtime creates one per forward and passes it to every step function. Fields
fixed for the step are set at creation; the boundary owns the residual-stream
fields; the per-layer fields are rewritten for each layer. A package keeps values
of its own in ``scratch`` under ``"<package>.<name>"``, and its build result in
``bound["<package>"]``.
"""

from __future__ import annotations

import contextlib
from typing import Any, Optional

import torch


class StepState:
    def __init__(
        self,
        *,
        lm,
        forward_batch,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        input_ids_global: torch.Tensor,
        slots,
        bound: dict,
        view=None,
        binding=None,
    ):
        # fixed for the step
        self.lm = lm  # DeepseekV4ForCausalLM
        self.model = lm.model  # DeepseekV4Model: layers, embedding, head norm
        self.fb = forward_batch
        self.positions = positions
        self.input_ids = input_ids
        self.input_ids_global = input_ids_global
        self.slots = slots  # plan.Slots: which kernels fill this width's steps
        self.bound: dict[str, Any] = bound  # package -> its build() result
        # SGLang's ForwardView of a covered forward (Step.view), and its build-time
        # MegakernelBinding for the forwards it hands no view
        self.view, self.binding = view, binding
        # the materialized residual streams (residual + pre-mix) where a reader needs
        # them between layers (Engram, the draft's capture); the boundary keeps them
        self.state: Any = None
        # per layer
        self.layer_id: int = -1
        # the sublayer input's MXFP8 pair, when the boundary wrote one
        self.x_quant: Any = None
        self.transport: Any = None  # the MoE operand transport of this layer
        # scopes a boundary opens across a sublayer; the runtime closes it per layer
        self.exit: contextlib.ExitStack = contextlib.ExitStack()
        # step outputs and per-step inputs
        self.hash_ids: Optional[torch.Tensor] = None  # Engram token hashes
        self.aux: Optional[AuxPack] = None  # the draft's captured hidden states
        self.scratch: dict[str, Any] = {}
        # FlashMLA schedules forked at the step head, joined by the first attention
        self.flashmla_sched: list = []

    def forward_view(self):
        """SGLang's ForwardView of this forward: the step's own, else the one the
        binding builds from the forward context running now."""
        return self.view if self.view is not None else self.binding.forward_view()


class AuxPack:
    """The draft's captured hidden states in one [rows, count * dim] buffer: SGLang's
    packed form, which the logits processor and the commit projection read as is."""

    def __init__(self, streams: torch.Tensor, count: int):
        rows, _, dim = streams.shape
        self.buffer = streams.new_empty((rows, count * dim))
        self._dim, self._count, self._taken = dim, count, 0

    def take(self) -> torch.Tensor:
        """The next capture's [rows, dim] slot."""
        start = self._taken * self._dim
        self._taken += 1
        return self.buffer[:, start : start + self._dim]

    def packed(self) -> torch.Tensor:
        assert self._taken == self._count, (self._taken, self._count)
        return self.buffer
