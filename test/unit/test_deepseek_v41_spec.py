# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DeepSeek-V4.1 spec through SGLang's startup checks on CPU, as the plugin
registers it: the documented setup passes, and every other setup is refused at
argument resolution with each reason named."""

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import pytest
from sglang.srt.environ import envs

try:
    from sglang.kernels.plugin_kernels.nvidia import spec as registry
    from sglang.kernels.plugin_kernels.nvidia.spec import Feature
    from sglang.srt.kernel_plugins.nvidia import binding as guards
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels import guards
    from sglang.srt.megakernels import spec as registry
    from sglang.srt.megakernels.spec import Feature

from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    default_cuda_graph_config,
    with_phase,
)
from sglang.srt.server_args import ServerArgs

from sgl_nvidia_mega import plugin
from sgl_nvidia_mega.deepseek_v41 import HF_CONFIG, spec

DOCUMENTED = dict(
    model_path="dummy",
    enable_nvidia_megakernel=True,
    tp_size=4,
    kv_cache_dtype="fp8_e4m3",
    speculative_algorithm="DSPARK",
    moe_a2a_backend="none",
    dsv4_attn_backend="flashmla",
    cuda_graph_config=default_cuda_graph_config(),
)
# the documented launch's environment (benchmark/README.md)
DOCUMENTED_ENV = {
    "SGLANG_SPEC_TP_SYNC": "rng,-dspark-accept-sample",
    "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": True,
    "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": "per_rank",
}


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.delenv(plugin.PARITY_SORT_ENV, raising=False)
    plugin.register()


def _refusal(hf=None, dtype="bfloat16", isa="sm_103", env=None, **args) -> str:
    """SGLang's resolution step on these arguments: "" when it accepts them, else
    its error message."""
    hf_config = SimpleNamespace(
        architectures=[spec.ARCHITECTURE],
        **{**HF_CONFIG, "model_type": "deepseek_v41", **(hf or {})},
    )
    with (
        contextlib.ExitStack() as stack,
        patch.object(guards, "plugin_reasons", return_value=[]),
        patch.object(guards, "device_isa", return_value=isa),
        patch.object(guards, "model_config_of") as model_config,
        patch.object(spec, "flashinfer_ll_gemm_digest", return_value=None),
        patch.object(spec.importlib.util, "find_spec", return_value=object()),
    ):
        model_config.return_value = SimpleNamespace(hf_config=hf_config, dtype=dtype)
        for name, value in {**DOCUMENTED_ENV, **(env or {})}.items():
            stack.enter_context(getattr(envs, name).override(value))
        try:
            guards.handle_nvidia_megakernel(ServerArgs(**{**DOCUMENTED, **args}))
        except ValueError as e:
            return str(e)
    return ""


def test_the_documented_setup_is_accepted():
    assert _refusal() == ""


BREAKABLE = with_phase(
    default_cuda_graph_config(), Phase.DECODE, backend=Backend.BREAKABLE
)


@pytest.mark.parametrize(
    "kw, want",
    [
        (dict(tp_size=8), "tp_size 8 with ep_size 1"),
        (dict(tp_size=2), "tp_size 2"),
        (dict(kv_cache_dtype="auto"), "kv_cache_dtype 'auto'"),
        (dict(speculative_algorithm="EAGLE"), "speculative_algorithm 'EAGLE'"),
        (dict(cuda_graph_config=BREAKABLE), "decode CUDA graphs"),
        (dict(isa="sm_100"), "not sm_100"),
    ],
)
def test_each_refusal_names_its_reason(kw, want):
    assert want in _refusal(**kw)


# attention TP8 + MoE TP4 x EP2 over two GB300 trays
TP8_EP2 = dict(
    tp_size=8, ep_size=2, nnodes=2, node_rank=0, dist_init_addr="127.0.0.1:5000"
)


def test_tp8_with_ep2_over_two_trays_is_accepted():
    assert _refusal(**TP8_EP2) == ""
    assert {Feature.MULTI_NODE_TP, Feature.EXPERT_PARALLEL} <= spec.SPEC.features


@pytest.mark.parametrize(
    "kw, want",
    [
        (dict(TP8_EP2, ep_size=1), "tp_size 8 with ep_size 1"),
        (dict(TP8_EP2, ep_size=4), "tp_size 8 with ep_size 4"),
        (dict(TP8_EP2, ep_size=8), "tp_size 8 with ep_size 8"),
        (dict(ep_size=2), "tp_size 4 with ep_size 2"),
        (dict(ep_size=4), "tp_size 4 with ep_size 4"),
        (
            dict(nnodes=2, node_rank=0, dist_init_addr="127.0.0.1:5000"),
            "TP4 serves one tray",
        ),
        (dict(TP8_EP2, enable_eplb=True), "EPLB"),
    ],
)
def test_every_other_topology_is_refused_with_its_reason(kw, want):
    assert want in _refusal(**kw)


def test_a_replicated_shared_expert_is_refused():
    """The plugin follows SGLang's default shared expert, split over the TP group; the
    replicated one (SGLANG_SHARED_EXPERT_TP1) would change what the posts sum."""
    message = _refusal(env={"SGLANG_SHARED_EXPERT_TP1": True}, **TP8_EP2)
    assert "SGLANG_SHARED_EXPERT_TP1" in message, message


def test_the_topology_table_is_tp4_and_tp8_ep2():
    assert spec.TOPOLOGIES == {(4, 1), (8, 2)}
    assert spec.REQUIREMENTS.tp_sizes == {4, 8}
    for tp, ep in [(4, 1), (8, 2)]:
        cfg = SimpleNamespace(tp_size=tp, ep_size=ep, nnodes=tp // 4, moe_dp_size=1)
        assert spec.topology_reasons(cfg) == []
    cfg = SimpleNamespace(tp_size=8, ep_size=2, nnodes=2, moe_dp_size=2)
    assert spec.topology_reasons(cfg) == ["moe_dp_size 2 is not 1"]


DISABLED = with_phase(
    default_cuda_graph_config(), Phase.DECODE, backend=Backend.DISABLED
)


def test_decode_without_cuda_graphs_is_refused_with_its_reason():
    """The eager forward loses long-context accuracy, so a launch without decode CUDA
    graphs is refused, saying why."""
    message = _refusal(cuda_graph_config=DISABLED)
    assert "decode CUDA graphs 'disabled' is not one of ['full']" in message, message
    assert (
        "the eager forward is not supported yet; enable decode CUDA graphs" in message
    )


@pytest.mark.parametrize(
    "switch, value",
    [
        ("SGLANG_NVIDIA_MEGAKERNEL_PARITY", "1"),
        ("SGLANG_NVIDIA_MEGAKERNEL_FAULT_INJECT", "1:hop_a:5000"),
    ],
)
def test_a_test_only_switch_runs_eager_steps(monkeypatch, switch, value):
    """The stock-parity and hop-fault serving tests step eagerly; only their switches
    let a launch without decode CUDA graphs through."""
    assert switch in spec.EAGER_TEST_SWITCHES
    monkeypatch.setenv(switch, value)
    assert spec.decode_graph_backends() == {"full", "disabled"}
    # SPEC's requirements are read at import, under the server's environment
    requires = msgspec.structs.replace(
        spec.SPEC.requires, decode_graph_backends=spec.decode_graph_backends()
    )
    monkeypatch.setattr(
        spec, "SPEC", msgspec.structs.replace(spec.SPEC, requires=requires)
    )
    assert _refusal(cuda_graph_config=DISABLED) == ""


def test_a_launch_without_the_measured_environment_is_refused():
    """Each variable of the measured launch is named; SGLang's defaults are refused."""
    defaults = {
        "SGLANG_SPEC_TP_SYNC": "all",
        "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": False,
        "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": "shared",
    }
    message = _refusal(env=defaults)
    assert message.count("\n  - ") == len(defaults), message
    for name in defaults:
        assert name in message


def test_the_draft_spec_follows_the_target_spec():
    """The draft shares the target's GPUs and contract version; a drifted copy would
    accept a setup the target refuses, or pass an API check the target fails."""
    from sgl_nvidia_mega.deepseek_v41 import draft, runtime, spec

    assert draft.SPEC.api_version == spec.API_VERSION
    for field in ("isas", "tp_sizes", "dtypes", "decode_graph_backends"):
        assert getattr(draft.SPEC.requires, field) == getattr(spec.SPEC.requires, field)
    # the draft forward's widest step with slot "draft", else SGLang's forward
    assert draft.SPEC.max_rows == (48 if "draft" in runtime.SLOTS else 0)


def test_sm148_lowers_the_targets_sm_floor_by_name(monkeypatch):
    """The target needs GB300's 152 SMs, or a B300's 148 with "sm148" listed, a name
    the slot check knows; the draft needs none."""
    from sgl_nvidia_mega.deepseek_v41 import draft, runtime

    monkeypatch.setattr(spec, "SLOTS", frozenset())
    assert spec.resident_blocks() == 152
    monkeypatch.setattr(spec, "SLOTS", frozenset({"sm148"}))
    assert spec.resident_blocks() == 148
    assert draft.SPEC.requires.resident_blocks == 0
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"sm148"}))
    assert _refusal() == ""
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"sm184"}))
    assert "unknown slot 'sm184'" in _refusal()
