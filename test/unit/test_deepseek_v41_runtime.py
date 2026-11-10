# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The executor and SGLang's binding on CPU: only captured steps are served, the
binding reaches every reader, the slots' reads follow SGLang's real init order
(scheduler_init.py), and the Engram prefetch takes its host table from the binding."""

import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

pytest.importorskip("sglang.srt.speculative.dspark_components.dspark_worker_v2")

from scheduler_init import (  # noqa: E402
    Scenario,
    SchedulerInit,
)

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import Step
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.spec import Step

from sgl_nvidia_mega.deepseek_v41 import runtime, spec  # noqa: E402
from sgl_nvidia_mega.deepseek_v41.engram import prefetch  # noqa: E402
from sgl_nvidia_mega.deepseek_v41.runtime import executor  # noqa: E402

# ------------------------------------------------------------------------------------
# The covered step runs only while SGLang captures the decode graphs, or under a
# test-only switch.


def test_the_engram_prefetch_holds_the_reduced_rows_until_gather(monkeypatch):
    """reduce()'s all-reduced rows are read on the side stream after reduce() returns.
    An out-of-place all-reduce's record_stream does not keep them (the allocator ignores
    it on DLPack tensors), so the prefetch holds them until gather() joins wkv_done."""
    import gc
    import weakref

    waits, refs = [], []

    class Stream:
        def wait_event(self, event):
            waits.append((event, refs[-1]() is not None if refs else None))

        def wait_stream(self, other):
            pass

    class Event:
        def record(self, stream=None):
            pass

    class Rows:  # an out-of-place all-reduce result
        def record_stream(self, stream):
            pass  # what the caching allocator does for a tensor it did not allocate

        def __getitem__(self, j):
            return self

    def reduce_owned_rows(looked):
        out = Rows()
        refs.append(weakref.ref(out))
        return out

    main = Stream()
    monkeypatch.setattr(prefetch.torch.cuda, "current_stream", lambda: main)
    monkeypatch.setattr(
        prefetch.torch.cuda, "stream", lambda s: contextlib.nullcontext()
    )
    engram = SimpleNamespace(
        embed=SimpleNamespace(_reduce_owned_rows=reduce_owned_rows)
    )
    layer = SimpleNamespace(
        engram=engram, n=4, cols=slice(0, 4), quarter=lambda rows: torch.ones(2, 4)
    )
    p = object.__new__(prefetch.EngramPrefetch)
    p.layers, p.stream, p.rows, p.hash_cols, p.dim = {3: layer}, Stream(), 2, 1, 4
    p.looked, p.parts = torch.zeros(8), torch.zeros(8)
    p.looked_up, p.reduced, p.wkv_done = Event(), Event(), Event()
    p.group, p.kv, p._emb = SimpleNamespace(all_reduce=lambda x: x), None, None

    p.reduce()
    gc.collect()
    assert refs[0]() is not None  # held past reduce(), while the side stream reads it
    p.gather()
    gc.collect()
    # still held when main waits for the side stream's WKV; released after
    assert (p.wkv_done, True) in waits and refs[0]() is None


def _batch(mode="TARGET_VERIFY"):
    is_verify = mode == "TARGET_VERIFY"
    return SimpleNamespace(
        forward_mode=SimpleNamespace(name=mode, is_target_verify=lambda: is_verify)
    )


def _capturing(monkeypatch, on=True):
    import sglang.srt.model_executor.runner as runner

    monkeypatch.setattr(runner, "get_is_capture_mode", lambda: on)


@pytest.mark.parametrize("which", ["target", "draft"])
@pytest.mark.parametrize(
    "capturing, switch, served",
    [
        (False, None, False),
        (True, None, True),
        (False, ("SGLANG_NVIDIA_MEGAKERNEL_PARITY", "1"), True),
        (False, ("SGLANG_NVIDIA_MEGAKERNEL_FAULT_INJECT", "1:hop_a:5000"), True),
    ],
)
def test_only_captured_steps_are_served(monkeypatch, which, capturing, switch, served):
    """Under FULL decode CUDA graphs SGLang may still step eagerly (an uncaptured
    batch); such a step runs SGLang's forward, and only the graph capture (its
    warm-up runs included) or a test-only switch runs the covered step."""
    from sgl_nvidia_mega.deepseek_v41 import draft_step

    _capturing(monkeypatch, capturing)
    for name in spec.EAGER_TEST_SWITCHES:
        monkeypatch.delenv(name, raising=False)
    if switch:
        monkeypatch.setenv(*switch)
    if which == "target":
        built = executor.MegakernelDecode(None, {}, {}, (6, 12))
    else:
        built = draft_step.DraftDecode.__new__(draft_step.DraftDecode)
        built.widths = (6, 12)
    step = built.plan(torch.zeros(12, dtype=torch.int64), _batch())
    assert step == (Step(mode="target_verify", rows=12) if served else None)


# ------------------------------------------------------------------------------------
# SGLang's binding (BuildContext.binding) reaches every reader on the real build and
# forward paths: each package's build context, the executor and its per-step state, and
# the draft's built object. A package handed none builds at no width (10-18 boot crash).


class _Stop(Exception):
    pass


def test_runtime_build_hands_every_package_the_binding(monkeypatch):
    """runtime.build gives each package of PACKAGES the same PackageContext, and it
    carries the BuildContext's binding."""
    seen = {}

    def recorder(name, last):
        def build(model, ctx, group):
            seen[name] = ctx
            if last:
                raise _Stop  # the rest of the build needs real packages
            return None

        return SimpleNamespace(build=build)

    names = list(executor.PACKAGES)
    monkeypatch.setattr(
        executor,
        "PACKAGES",
        {n: recorder(n, i == len(names) - 1) for i, n in enumerate(names)},
    )
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_parallel",
        lambda: SimpleNamespace(tp_group=None),
    )
    try:
        from sglang.srt.kernel_plugins.nvidia.binding import MegakernelBinding
    except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
        from sglang.srt.megakernels.binding import MegakernelBinding

    model = SimpleNamespace()
    binding = MegakernelBinding(model=model)  # SGLang's own, as install builds it
    ctx = SimpleNamespace(
        widths=(6,), tokens_per_request=6, deadline_s=1.0, binding=binding
    )
    with pytest.raises(_Stop):
        runtime.build(model, ctx)
    assert sorted(seen) == sorted(names)
    assert all(c.binding is binding for c in seen.values()), seen


def test_the_draft_build_keeps_the_binding():
    """The draft spec's build gets its own BuildContext.binding; the draft forward's
    ported attention reads its view from it."""
    from sgl_nvidia_mega.deepseek_v41 import draft, draft_step

    binding = object()
    with (
        mock.patch.object(draft_step, "DraftDecode") as decode,
        mock.patch.object(runtime, "SLOTS", frozenset({"draft"})),
    ):
        draft._build("model", None, SimpleNamespace(deadline_s=1.0, binding=binding))
    assert decode.call_args.kwargs["binding"] is binding


# ------------------------------------------------------------------------------------
# Every slot that reads model or runner state after load, in SGLang's real init order
# (scheduler_init.py): no read before the stage that sets it, and the slot engages.

# FU2's draft forward slots (draft_embed apart: v5 drops it; its scenario adds it back)
DRAFT_SLOTS = frozenset(
    {
        "markov",
        "draft_lm_head",
        "draft",
        "draft_attn",
        "draft_moe",
        "draft_gate",
        "draft_tail",
        "draft_post",
        "draft_hop",
        "draft_seam",
        "draft_qkv_a",
        "draft_qfront",
        "draft_wo_a",
        "draft_swapab",
        "draft_prenorm",
        "draft_qkv_a_ll",
        "draft_wo_b_ll",
    }
)


def _binding(init: SchedulerInit) -> SimpleNamespace:
    """SGLang's MegakernelBinding, built at install: it reads nothing until asked; its
    forward view reads the running forward's attention backend (here the draft runner's)."""

    def forward_view():
        backend = init.draft_runner.attn_backend
        return SimpleNamespace(kv_pool=SimpleNamespace(pool=backend.token_to_kv_pool))

    return SimpleNamespace(forward_view=forward_view)


def _draft_decode_at_install(init: SchedulerInit) -> None:
    """install: with slot draft the draft spec builds DraftDecode when the draft model
    loads, before attach_shared_modules. Its stage builders only see the stages (they
    compile kernels on GPU), so they are recorded here instead of run; so is
    draft_prenorm's hardware check."""
    from sgl_nvidia_mega.deepseek_v41 import draft, draft_step, runtime

    built = []

    def record(name):
        return lambda stages, *a, **k: (
            built.append(name)
            or SimpleNamespace(slots={}, reason="", summary=lambda: "")
        )

    with (
        mock.patch.object(runtime, "SLOTS", init.slots),
        mock.patch.object(draft_step.moe.routed, "build_layers", record("moe")),
        mock.patch.object(draft_step.boundary, "Boundaries", record("boundary")),
        mock.patch.object(draft_step.draft_attention, "build", record("attention")),
        mock.patch("torch.cuda.is_available", return_value=False),
        mock.patch(
            "sgl_nvidia_mega.deepseek_v41.boundary.prenorm.reason", return_value=""
        ),
    ):
        init.state.draft_built = draft._build(
            init.draft_model,
            None,
            SimpleNamespace(deadline_s=1.0, binding=_binding(init)),
        )
    assert isinstance(init.state.draft_built, draft_step.DraftDecode)
    assert sorted(built) == ["attention", "boundary", "moe"]


def _draft_decode_engages(init: SchedulerInit) -> None:
    """draft_graphs: the built widths are gamma rows a request, and draft_embed's table
    (deferred at install) binds to the attached embedding only now."""
    from sgl_nvidia_mega.deepseek_v41 import draft_step

    built = init.state.draft_built
    assert built.widths[0] == init.gamma
    # the draft forward's view resolves only once the draft has its attention backend
    assert built._binding.forward_view().kv_pool.pool is init.draft_pool
    if "draft_embed" in init.slots:
        assert built._embed_table is draft_step._SHARED_TABLE
        assert init.draft_model.embed_tokens is not None


DRAFT_DECODE = Scenario(
    "draft_decode",
    DRAFT_SLOTS,
    (
        ("install", _draft_decode_at_install),
        ("draft_graphs", _draft_decode_engages),
        # the first decode after the scheduler's init: the same reads again
        ("overlap", _draft_decode_engages),
    ),
)
DRAFT_DECODE_EMBED = DRAFT_DECODE._replace(
    name="draft_decode_embed", slots=DRAFT_SLOTS | {"draft_embed", "embed"}
)


SCENARIOS = [DRAFT_DECODE, DRAFT_DECODE_EMBED]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_slot_reads_nothing_before_sglang_sets_it(scenario):
    SchedulerInit().run(scenario)


# ------------------------------------------------------------------------------------
# The Engram prefetch takes each layer's host table from SGLang's binding, never from
# the layer itself.


def _model():
    engram = SimpleNamespace(layer_hash_index=0)
    layers = [SimpleNamespace(engram=None), SimpleNamespace(engram=engram)]
    return SimpleNamespace(
        model=SimpleNamespace(layers=layers, start_layer=0, end_layer=2)
    )


def _ctx(binding=None):
    """The context the runtime hands each package (not SGLang's BuildContext)."""
    from sgl_nvidia_mega.deepseek_v41.runtime import PackageContext

    return PackageContext(
        widths=(6,), tokens_per_request=6, deadline_s=1.0, copies={}, binding=binding
    )


def test_the_host_table_comes_from_the_binding(monkeypatch):
    """SGLang's own MegakernelBinding hands the layer's host table (a shared one here,
    which the prefetch refuses)."""
    try:
        from sglang.srt.kernel_plugins.nvidia.binding import MegakernelBinding
    except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
        from sglang.srt.megakernels.binding import MegakernelBinding

    model = _model()
    engram = model.model.layers[1].engram
    engram.embed = SimpleNamespace(host_table=SimpleNamespace(layout="shared"))
    binding = MegakernelBinding(model=model)
    asked = []
    real = binding.engram_host_table
    monkeypatch.setattr(
        binding, "engram_host_table", lambda e: asked.append(e) or real(e)
    )
    ctx = _ctx(binding)
    # one rank: the gathered error is this rank's own
    monkeypatch.setattr(prefetch, "_any_error", lambda group, error: error)
    built, why = prefetch.build(model, ctx, SimpleNamespace(world_size=4))
    assert asked == [model.model.layers[1].engram]
    assert built is None and "per-rank host Engram table" in why


def test_the_engram_image_select_runs_only_on_a_multimodal_batch(monkeypatch):
    """A text-only batch (every decode / verify capture) keeps the gate's output as
    is, with no select in the step; a multimodal batch keeps the residual on its
    image-placeholder rows."""
    from sgl_nvidia_mega.deepseek_v41 import boundary, engram

    x = torch.arange(3 * 2 * 4, dtype=torch.float32).view(3, 2, 4)
    gated = -x
    st = SimpleNamespace(
        state=SimpleNamespace(residual=x),
        layer_id=1,
        input_ids=torch.tensor([5, 7, 9]),
        slots=SimpleNamespace(engram_prefetch=True),
        bound={
            "engram": SimpleNamespace(prefetch=SimpleNamespace(gate=lambda i, h: gated))
        },
        model=SimpleNamespace(
            config=SimpleNamespace(
                model_type="deepseek_v41", vision_n_layers=1, image_token_id=7
            )
        ),
    )
    written = []
    monkeypatch.setattr(boundary, "set_residual", lambda s, out: written.append(out))
    layer = SimpleNamespace(engram=SimpleNamespace())
    for mm in (False, True):
        st.fb = SimpleNamespace(contains_mm_inputs=lambda mm=mm: mm)
        engram.apply(layer, st)
    text, multimodal = written
    assert text is gated
    assert torch.equal(multimodal[1], x[1])
    assert torch.equal(multimodal[[0, 2]], gated[[0, 2]])
