# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DeepSeek-V4.1 megakernel's spec: the setups it serves and its build.

SGLang checks the requirements at argument resolution and the model at load
(``sglang.kernels.plugin_kernels.nvidia``); an unsupported setup stops the server on every rank,
listing every reason. This module stays light: SGLang imports it in the launcher.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

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

from sgl_nvidia_mega.deepseek_v41 import HF_CONFIG
from sgl_nvidia_mega.deepseek_v41.plan import max_rows
from sgl_nvidia_mega.deepseek_v41.runtime import SLOTS

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "Feature",
        "SPECULATIVE_DECODING",
        "MegakernelSpec",
        "Numerics",
        "NumericsKind",
        "TOLERANCE",
        "Requirements",
        "decode_graph_backends",
        "config_disable_reasons",
    ),
    # the decode CUDA graph backend _config_disable_reasons reads
    "srt/model_executor/cuda_graph_config.py": ("decode", "backend", '"disabled"'),
    "srt/model_executor/runner/__init__.py": ("get_is_capture_mode",),
    "srt/arg_groups/fields/*.py": (
        "cuda_graph_config",
        "kv_cache_dtype",
        "speculative_algorithm",
        "moe_a2a_backend",
        "dsv4_attn_backend",
        "enable_encoder_swa_bounded_replay",
        "startup_weight_load_mode",
    ),
}

ARCHITECTURE = "DeepseekV4ForCausalLM"
# the SGLang megakernel contract this spec is written against
API_VERSION = 2
# The FlashInfer low-latency GEMM file the wo_b publish epilogue subclasses, by the
# sha256 of its source. 0.7.3 only memoizes autotune_tactics (same tactics, same
# order); the kernel class is unchanged.
FLASHINFER_LL_GEMM = "gemm/kernels/cute_dsl/low_latency_blockscaled_gemm.py"
FLASHINFER_LL_GEMM_SHA256 = frozenset(
    {
        "118744bf9553da5fcbf6504739d499ec208dcf2034d473799ec11661e1313280",  # 0.7.0-0.7.2
        "c9d16930cc99c98d2de57e10882f52611a784268ba497b2b10d613a935a5596c",  # 0.7.3
    }
)

# Test-only switches that run the covered step eagerly: the composition check
# (runtime/parity.py) and the injected hop fault (sm103/runner.py)
EAGER_TEST_SWITCHES = (
    "SGLANG_NVIDIA_MEGAKERNEL_PARITY",
    "SGLANG_NVIDIA_MEGAKERNEL_FAULT_INJECT",
)


def eager_steps_allowed() -> bool:
    """Decode without CUDA graphs is refused (the eager forward loses long-context
    accuracy) except under a test-only switch."""
    return any(os.environ.get(k, "") not in ("", "0") for k in EAGER_TEST_SWITCHES)


def serves_this_step() -> bool:
    """A covered step runs the megakernel only while SGLang captures its decode CUDA
    graphs (their warm-up runs included), or eagerly under a test-only switch; any
    other step, e.g. an eager fallback under FULL graphs, runs SGLang's forward."""
    from sglang.srt.model_executor.runner import get_is_capture_mode

    return get_is_capture_mode() or eager_steps_allowed()


def decode_graph_backends() -> frozenset:
    """FULL decode CUDA graphs; eager steps too under a test-only switch."""
    return frozenset({"full", "disabled"} if eager_steps_allowed() else {"full"})


def resident_blocks() -> int:
    """GB300's 152 SMs, within which the GEMM pair's persistent schedule and the
    Lamport consumer stay; with "sm148" listed, a B300's 148: no DeepSeek-V4.1 kernel
    pins a 152-CTA grid."""
    return 148 if "sm148" in SLOTS else 152


REQUIREMENTS = Requirements(
    isas=frozenset({"sm_103"}),
    tp_sizes=frozenset({4}),
    resident_blocks=resident_blocks(),
    dtypes=frozenset({"bfloat16"}),
    decode_graph_backends=decode_graph_backends(),
    server_args={
        "kv_cache_dtype": frozenset({"fp8_e4m3"}),
        # DSpark's target steps are uniform verify steps
        "speculative_algorithm": frozenset({"DSPARK"}),
        "moe_a2a_backend": frozenset({"none"}),
        # the wo_b publish follows FlashMLA's sparse decode; on GB300 it is also
        # the faster backend at batch 1-8
        "dsv4_attn_backend": frozenset({"flashmla"}),
        # the paged SWA cache, not a request window
        "enable_encoder_swa_bounded_replay": frozenset({False}),
        # the kernels read weights in place, after SGLang's last conversion
        "startup_weight_load_mode": frozenset({"serial"}),
    },
    hf_config={**HF_CONFIG, "model_type": "deepseek_v41"},
    env={
        # the TP sync setting every measured launch ran with
        "SGLANG_SPEC_TP_SYNC": "rng,-dspark-accept-sample",
        # slot engram reads the per-rank host table SGLang builds only with these (no
        # per-table NUMA placement; SGLang's process membind, SGLANG_NUMA_BIND_V2, only)
        **(
            {
                "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": True,
                "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": "per_rank",
            }
            if "engram" in SLOTS
            else {}
        ),
    },
)

# Per kernel output against SGLang's op on the same GPU (test/kernels); the
# tolerances are the measured ones
NUMERICS = Numerics(
    kind=NumericsKind.TOLERANCE,
    tolerances={
        # only SGLang's own writers touch the KV caches; with slot attn_qkv_a (or the
        # draft's draft_qkv_a_ll: the same GEMM shape and kernels), wqkv_a on the
        # low-latency GEMM at 1-8 rows against SGLang's (split-K order): its BF16
        # output over its norm, and the window KV it stores, per region of SGLang's
        # serving layout V4 (448 FP8 + 64 BF16 RoPE + 7 scales a token; V4.1 alike for
        # FP8), measured on 4 checkpoint layers (test_deepseek_v41_attn_gemm.py): the
        # fraction of values moved, the largest move in native units (near-zero values
        # span many fine codes / ulps; an earlier "<= 1 e4m3 step" claim was wrong, V4.1
        # reaches 4) and the bound on it, against the block's coarsest FP8 quantum (32 x
        # the block scale) or the row's peak RoPE magnitude; scale bytes never move
        **(
            {
                "attn.wqkv_a.rel": 2.7e-4,
                "attn.kv.fp8.moved_fraction": 2.1e-5,
                # the V4.1 value; the V4 serving layout measured 1
                "attn.kv.fp8.max_steps": 4,
                "attn.kv.fp8.block_top_steps": 1.0,
                "attn.kv.rope.moved_fraction": 2.1e-4,
                "attn.kv.rope.max_bf16_ulp": 48,
                "attn.kv.rope.row_peak_rel": 1.3e-2,
                "attn.kv.scale.max_abs": 0.0,
            }
            if {"attn_qkv_a", "draft_qkv_a_ll"} & SLOTS
            else {"attn.kv.max_abs": 0.0}
        ),
        # wo_b's published partial, as FlashInfer's low-latency GEMM writes it
        "wo_b.partial.max_abs": 0.0,
        # the router's expert ids and weights, as moe_fused_gate picks them
        "gate.topk.max_abs": 0.0,
        # the prefetched Engram WKV rows, as SGLang's inline lookup + WKV writes them
        "engram.kv.max_abs": 0.0,
        # the Lamport hop's streams and normed row against all-reduce + mHC in torch
        "hop_a.residual.rel": 2e-2,
        "hop_a.normed.rel": 3e-2,
        # the routed experts against FlashInfer's routed MoE: one BF16 rounding
        "moe.routed.rel": 2**-7,
        # wq_b on the low-latency MXFP8 GEMM against SGLang's MXFP8 GEMM, same input
        "attn.wq_b.max_abs": 0.0,
        # the PDL norm's BF16 q-LoRA row against FlashInfer's RMSNorm (reduction order)
        "attn.q_lora.bf16_ulp": 1,
        # the fused inverse RoPE + wo_a at 33-48 rows against a float64 reference
        "attn.wo_a.fp8_steps": 1,
        # the attention step's wo_b partial against SGLang's, per layer, over the
        # layer's peak: swap-AB vs FlashMLA at 12-24 rows 1.8e-2, 25-48 rows 6.1e-3
        "attn.out.rel": 2e-2,
        # at the 1-8 rows slot attn_qkv_a serves: up to 2.13e-2 (6 rows, ratio 1)
        **({"attn.out.rel_1_8_rows": 2.5e-2} if "attn_qkv_a" in SLOTS else {}),
        # slot draft_wo_b_ll: the draft's wo_b on the low-latency GEMM at 1-8 rows, fed
        # the fused wo_a's MXFP8 rows, against SGLang's MXFP8 linear on the same rows
        # (test_deepseek_v41_attn_gemm.py: 7.0e-7 worst over rows 1-8)
        **({"draft.wo_b.rel": 1e-6} if "draft_wo_b_ll" in SLOTS else {}),
        # the two-shot hops against SGLang's fused all-reduce + mHC kernels (streams,
        # normed row, MXFP8 next input at up to 128 rows): the same rank-order sum
        "two_shot.residual.max_abs": 0.0,
        "two_shot.normed.ulp": 0,
        "two_shot.mxfp8.max_abs": 0.0,
        # hop B against SGLang's fused MoE post: the Lamport consumer's mHC-post
        # rounding order, as hop A's; an MXFP8 value follows its normed element
        "hop_b.residual.ulp": 1,
        "hop_b.normed.ulp": 2,
        "hop_b.mxfp8.ulp": 1,
        "hop_b.mxfp8_scales.max_abs": 0.0,
        # slot hc_prenorm: the mHC coefficients (pre / post / comb) with the weight read
        # as TF32, against an exact projection through the same reduce + Sinkhorn; 3.81e-4
        # measured at 6-48 rows on synthetic inputs (SGLang's split-K: 4.8e-7)
        "hc_prenorm.coeff.max_abs": 3.9e-4,
        # slot target_lm_head: the target's logits shard on the low-M kernel against
        # SGLang's matmul, checked at build on every rank (the K summation order differs)
        "target_lm_head.logits.rel": 2**-7,
    },
)


def toolchain_reasons() -> list[str]:
    """The kernel toolchains, and the FlashInfer GEMM the wo_b publish extends."""
    reasons = []
    for module, what in (
        ("cutlass", "CuTe DSL (nvidia-cutlass-dsl)"),
        ("flashinfer", "FlashInfer"),
        ("cuda.bindings", "the CUDA Python bindings"),
    ):
        if importlib.util.find_spec(module) is None:
            reasons.append(f"needs {what}")
    digest = flashinfer_ll_gemm_digest()
    if digest is not None and digest not in FLASHINFER_LL_GEMM_SHA256:
        reasons.append(
            f"FlashInfer's {FLASHINFER_LL_GEMM} (sha256 {digest[:12]}) is not the "
            "low-latency GEMM the wo_b publish epilogue extends"
        )
    return reasons


def flashinfer_ll_gemm_digest():
    """The sha256 of the installed FlashInfer low-latency GEMM source, or None
    without FlashInfer (refused above)."""
    found = importlib.util.find_spec("flashinfer")
    if found is None or found.origin is None:
        return None
    path = Path(found.origin).parent / FLASHINFER_LL_GEMM
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    from sgl_nvidia_mega.deepseek_v41 import runtime

    reasons = toolchain_reasons() + runtime.slot_reasons()
    if cfg.cuda_graph_config.decode.backend == "disabled" and not eager_steps_allowed():
        reasons.append(
            "decode CUDA graphs are disabled: the eager forward is not supported yet; "
            "enable decode CUDA graphs"
        )
    return reasons


def _model_disable_reasons(model, hf_config) -> list[str]:
    from sgl_nvidia_mega.deepseek_v41 import runtime

    return runtime.model_disable_reasons(model)


def _build(model, hf_config, ctx):
    from sgl_nvidia_mega.deepseek_v41 import runtime

    return runtime.build(model, ctx)


SPEC = MegakernelSpec(
    name="DeepSeek-V4.1",
    requires=REQUIREMENTS,
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    numerics=NUMERICS,
    # the two-shot slot alone covers the widths past the routed-expert pair
    max_rows=max_rows("two_shot" in SLOTS),
    build=_build,
    api_version=API_VERSION,
    model_disable_reasons=_model_disable_reasons,
    config_disable_reasons=_config_disable_reasons,
)
