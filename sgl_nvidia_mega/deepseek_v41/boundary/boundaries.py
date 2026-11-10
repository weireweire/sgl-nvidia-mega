# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/deepseek_v4_mhc.py and deepseek_v4_boundary.py (the decode stage boundaries).
"""The mHC boundaries of a covered step: SGLang's stock decode boundary, with
hop A's Lamport collect as the attention post where wo_b published, the
two-shot hops as both posts from 49 rows, and hop B (the Lamport MoE finalize)
as the MoE post at hop-A widths.

Between two sublayers, SGLang's stage boundary for DeepSeek-V4.1 with the mHC
pre-mix carried from the previous sublayer:

1. ``pre``: forks the mHC stats stream, then collapses the residual streams into
   the sublayer's input (combine + RMSNorm, MXFP8 too where the attention takes
   it), or takes the input the previous post already built;
2. the sublayer runs; its TP sum is left to the post;
3. ``post``: predicts the next mix triplet on the stats stream, joins it, and runs
   the fused all-reduce + mHC post, which also builds the next sublayer's input
   where the seam allows (the FFN norm after attention; the next layer's input
   norm, MXFP8 at up to 128 rows, after the MoE).

Each step calls SGLang's own mHC functions with the arguments its stage boundary
passes, so with every hop off the streams, inputs and coefficients are SGLang's
bit for bit. A hop replaces only a post's collective kernel: the triplet, the next
boundary and the next input's form stay SGLang's.
"""

from __future__ import annotations

import logging
from typing import Optional

import msgspec
import torch

from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll

logger = logging.getLogger(__name__)

# The SGLang names this module reads, by file under the sglang package;
# test/unit/test_contract.py pins them.
# the widest step whose capture mean runs on the mHC stats stream
AUX_MEAN_MAX_ROWS = 16

SGLANG_READS = {
    "srt/models/deepseek_v4_mhc.py": (
        "HcState",
        "AttnOutput",
        "make_boundary",
        "use_stats_stream",
        "mix_stats",
        "HcNormed",
        "HcQuantized",
        "is_deferred_finalize",
        "norm",
        "norm_fusable",
        "can_fuse_post",
        "accepts_mxfp8",
        "fork_stats_stream",
        "combine",
        "run_attn_post",
        "run_moe_post",
        "release",
        "residual",
        "stats",
        "materialized",
        "with_residual",
    ),
    "srt/models/deepseek_v4.py": (
        "boundary_mhc_fusions",
        "boundary_pre_from_prev",
        "hc_stats_stream",
        "hc_cfg",
        "attn_hc",
        "ffn_hc",
        "input_layernorm",
        "post_attention_layernorm",
        "accepts_mxfp8_swizzled_input",
        "attn_tp_size",
        "engram",
        "_can_fuse_attn_mhc",
        "_can_fuse_ffn_mhc",
        "self_attn",
        # the module alias SGLang's DSV4 model imports too
        "mhc",
    ),
    "srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py": ("can_fuse_all_reduce",),
    "srt/models/deepseek_v2.py": (
        "tp_size",
        "_shared_expert_tp1",
        "MoEOutput",
        "routed",
        "shared",
        "shared_is_replicated",
    ),
    "srt/layers/quantization/mxfp8_input.py": ("Mxfp8SwizzledInput",),
    "srt/layers/moe/moe_runner/flashinfer_trtllm.py": (
        "gemm2_out",
        "expanded_idx_to_permuted_idx",
        "expert_weights",
        "top_k",
    ),
    "srt/runtime_context.py": (
        "get_parallel",
        '"attn_tp_group"',
        '"tp_size"',
        '"moe_ep_size"',
    ),
    "srt/layers/moe/__init__.py": ("get_moe_a2a_backend",),
    "srt/layers/moe/utils.py": ("should_skip_post_experts_all_reduce",),
}


class Open(msgspec.Struct):
    """A sublayer between its pre and its post: the residual streams its post
    mixes into, and the mix triplet's stream."""

    streams: torch.Tensor
    stats: Optional[tuple] = None
    stats_stream: Optional[torch.cuda.Stream] = None


class BoundaryState(msgspec.Struct):
    """The mHC state of one step: the streams between two hyper-connections
    (SGLang's ``HcState``), and the open sublayer, if any."""

    hc: object  # deepseek_v4_mhc.HcState
    open: Optional[Open] = None
    # slot aux_mean_stats: the capture means issued on a stats stream, joined in final
    aux_done: list = []


class Boundaries:
    """The boundaries of the target model's layers, built once from the loaded
    model: each layer's next-sublayer boundaries are load-time facts."""

    def __init__(
        self,
        layers,
        *,
        cross_layer: bool = True,
        hop_a=None,
        hop_a_reason: str = "",
        two_shot=None,
        two_shot_reason: str = "",
        hop_b: bool = False,
        mhc_fusions: Optional[bool] = None,
        prenorm: bool = False,
    ):
        from sglang.srt.models import deepseek_v4_mhc as mhc

        from sgl_nvidia_mega.deepseek_v41.boundary import prenorm as _prenorm

        self._mhc = mhc
        # the coefficients: SGLang's, or the DeepGEMM prenorm's at decode rows
        self.prenorm = prenorm
        self._prenorm = _prenorm
        self._mix_stats = _prenorm.mix_stats if prenorm else mhc.mix_stats
        self.cross_layer = cross_layer
        # None: each layer's own boundary_mhc_fusions (the draft stages build without)
        self._mhc_fusions = mhc_fusions
        self.hop_a = hop_a  # hop_a.HopA, or None
        self.hop_a_reason = hop_a_reason
        self.two_shot = two_shot  # sm103.two_shot_mhc.TwoShotMHC, or None
        self.two_shot_reason = two_shot_reason
        self.hop_b = hop_b and hop_a is not None  # the MoE post on hop A's mailbox
        self._served: set = set()
        self._attn_mxfp8 = {}
        self._ffn_next = {}  # the attention post's next: the FFN norm
        self._attn_next = {}  # the next layer's attention boundary (None: closed)
        self._next_layer = {}  # layer id -> the next layer's id
        # layers whose attention reads row-tile MXFP8 rows past 8 rows (wqkv_a on the
        # wide low-latency GEMM): set by the runtime from the built attention steps
        self.row_tile_layers: frozenset = frozenset()
        ids = [layer.layer_id for layer in layers]
        for layer in layers:
            self._attn_mxfp8[layer.layer_id] = (
                layer.self_attn.accepts_mxfp8_swizzled_input()
            )
            self._ffn_next[layer.layer_id] = (
                mhc.make_boundary(
                    layer.post_attention_layernorm,
                    accepts_mxfp8=False,
                    hc=layer.ffn_hc,
                )
                if self._fuses(layer)
                else None
            )
        for prev, layer in zip(ids, layers[1:]):
            self._attn_next[prev] = self._attention_boundary(layer)
            self._next_layer[prev] = layer.layer_id
        self._attn_next[ids[-1]] = None

    def _fuses(self, layer) -> bool:
        if self._mhc_fusions is None:
            return layer.boundary_mhc_fusions
        return self._mhc_fusions

    def _attention_boundary(self, layer):
        # SGLang reads an Engram layer's attention input after the injection,
        # so the MoE post before it leaves the streams only
        if not self._fuses(layer) or layer.engram is not None:
            return None
        return self._mhc.make_boundary(
            layer.input_layernorm,
            accepts_mxfp8=self._attn_mxfp8[layer.layer_id],
            hc=layer.attn_hc,
        )

    def start(self, streams: torch.Tensor) -> BoundaryState:
        """The state before layer 0: the embedding broadcast to the hc streams."""
        return BoundaryState(self._mhc.HcState(streams))

    def pre(self, layer, kind: str, st: BoundaryState, forward_batch, aux=None):
        """The sublayer's input: (rows, MXFP8 rows or None). With ``aux`` (a DSpark
        capture layer), also appends the streams' mean, on the stats stream."""
        mhc = self._mhc
        attention = _attention(kind)
        if st.open is not None:
            raise RuntimeError("a sublayer is still open: its post did not run")
        state = st.hc
        streams = state.residual
        stream = (
            layer.hc_stats_stream
            if mhc.use_stats_stream(layer.hc_cfg, forward_batch, streams)
            else None
        )
        mhc.fork_stats_stream(stream)
        if aux is not None:
            aux.append(self._aux_mean(streams, stream, st))
        quantized = [] if attention and self._attn_mxfp8[layer.layer_id] else None
        hc = layer.attn_hc if attention else layer.ffn_hc
        x = mhc.combine(hc, state, quantized)
        stats = state.stats
        state.release()
        st.open = Open(streams, stats, stream)
        return x, (quantized[0] if quantized else None)

    def post(self, layer, kind: str, partial, st: BoundaryState) -> None:
        """Reduce the open sublayer's TP-partial output into the streams.

        ``partial`` is the attention's wo_b rows before their TP sum (bare or in
        SGLang's ``AttnOutput``), or the
        MoE's output as SGLang's MoE returns it with the sum left to the post
        (a ``MoEOutput`` from ``leave_output_sum=True``, its routed finalize
        deferred where ``defers_sum``); past the push plane SGLang's own post
        merges, sums and posts it unfused."""
        mhc = self._mhc
        opened, st.open = st.open, None
        if opened is None:
            raise RuntimeError("post without a pre")
        if _attention(kind):
            if isinstance(partial, mhc.AttnOutput):  # SGLang's attention defers on TP
                partial = partial.partial
            if self.hop_a is not None and self.hop_a.published(partial):
                st.hc = self._collect_hop_a(layer, partial, opened)
                return
            if self._two_shot_fits(partial.shape[0], layer, kind):
                st.hc = self._two_shot_attn(layer, partial, opened)
                return
            if self.defers_sum(layer, kind, partial.shape[0]):
                out = mhc.AttnOutput(partial)
            else:
                # past the push plane SGLang's transport sums before a plain post
                from sglang.srt.runtime_context import get_parallel

                out = get_parallel().attn_tp_group.all_reduce(partial)
            st.hc = mhc.run_attn_post(
                layer.attn_hc,
                out,
                opened.streams,
                stats_stream=opened.stats_stream,
                next=self._ffn_next[layer.layer_id],
                world_size=layer.self_attn.attn_tp_size,
                precomputed=self._precomputed(layer.attn_hc, opened),
            )
            return
        from sglang.srt.models.deepseek_v2 import MoEOutput

        nxt = self._attn_next[layer.layer_id] if self.cross_layer else None
        if isinstance(partial, MoEOutput) and mhc.is_deferred_finalize(partial.routed):
            rows = partial.routed.expert_weights.shape[0]
            # the two-shot adds a TP-partial shared expert before the sum
            if (
                self._two_shot_fits(rows, layer, kind)
                and not partial.shared_is_replicated
            ):
                st.hc = self._two_shot_moe(layer, partial, opened, nxt)
                return
            if self._hop_b_takes(rows, partial):
                st.hc = self._hop_b_moe(layer, partial, opened, nxt)
                return
        st.hc = mhc.run_moe_post(
            layer.ffn_hc,
            partial,
            opened.streams,
            stats_stream=opened.stats_stream,
            next=nxt,
            world_size=layer.mlp.tp_size,
            precomputed=self._precomputed(layer.ffn_hc, opened),
        )

    def _precomputed(self, hc, opened):
        """The triplet SGLang's post takes as precomputed: the one carried in, else
        with the prenorm the plugin's (joined), else None (SGLang computes it)."""
        if opened.stats is not None or not self.prenorm:
            return opened.stats
        return self._triplet(hc, opened)

    def publish_wo_b(self, wo_b, o):
        """For the attention step at a hop-A width: wo_b's partial rows, published
        into the mailbox; None when hop A is off or its GEMM cannot take them."""
        return None if self.hop_a is None else self.hop_a.publish_wo_b(wo_b, o)

    def _triplet(self, hc, opened):
        """(pre, post, comb) as SGLang's post computes them: on the stats stream
        beside the sublayer, joined before the post reads them."""
        if self.prenorm and self._prenorm.takes(opened.streams):
            self._log_served("mHC prenorm", opened.streams.shape[0])
        coefficients = self._mix_stats(hc, opened.streams, opened.stats_stream)
        if opened.stats_stream is not None:
            torch.cuda.current_stream().wait_stream(opened.stats_stream)
        return coefficients

    def _collect_hop_a(self, layer, partial, opened):
        """``run_attn_post`` with the Lamport collect in place of SGLang's fused
        collective: the same triplet on the stats stream and the same FFN norm."""
        mhc = self._mhc
        self._log_served("hop A attention post", partial.shape[0])
        pre, post_mix, comb = self._triplet(layer.attn_hc, opened)
        norm = self._ffn_next[layer.layer_id].norm
        updated, normed = self.hop_a.collect(
            partial,
            opened.streams,
            post_mix,
            comb,
            pre,
            norm.weight,
            norm.variance_epsilon,
        )
        return mhc.HcState(updated, pre, mhc.HcNormed(normed))

    def defers_sum(self, layer, kind: str, rows: int) -> bool:
        """Whether the sublayer leaves its TP sum to the post at ``rows`` (the MoE
        then returns its finalize deferred): SGLang's transport check, or a
        two-shot width."""
        from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
            can_fuse_all_reduce,
        )

        return _fusable(layer, kind, self._fuses(layer)) and (
            self._two_shot_fits(rows, layer, kind)
            or can_fuse_all_reduce(rows, layer.hc_cfg.hidden)
        )

    def _two_shot_fits(self, rows: int, layer, kind: str) -> bool:
        return (
            self.two_shot is not None
            and self.two_shot.fits(rows, layer.hc_cfg.hidden)
            and _fusable(layer, kind, self._fuses(layer))
        )

    def summary(self) -> str:
        """The hops this boundary built, for the startup line."""
        hop_a = f"at {list(self.hop_a.rows)} rows" if self.hop_a else "off"
        two_shot = (
            f"at {MIN_TWO_SHOT_ROWS}-{self.two_shot.capacity} rows"
            if self.two_shot
            else "off"
        )
        return (
            f"hop A {hop_a} ({self.hop_a_reason or 'built'}); "
            f"hop B {f'at {list(self.hop_a.hop_b_rows)} rows' if self.hop_b else 'off'}"
            f"{' (row-tile scales past 8 rows)' if self.row_tile_layers else ''}; "
            f"two-shot {two_shot} ({self.two_shot_reason or 'built'}); "
            f"mHC prenorm {self._prenorm_summary()}"
        )

    def _prenorm_summary(self) -> str:
        from sgl_nvidia_mega.deepseek_v41.boundary import prenorm

        if not self.prenorm:
            return "SGLang's"
        return f"DeepGEMM TF32 at 1-{prenorm.MAX_ROWS} rows ({prenorm.SLICES} slices)"

    def _aux_mean(self, streams, stream, st: BoundaryState) -> torch.Tensor:
        """The draft head's capture of a layer's attention input: on the stats stream
        beside the attention when the layer has one and the step is narrow (the main
        stream otherwise: at wide steps the side stream competes with the attention)."""
        if stream is None or streams.shape[0] > AUX_MEAN_MAX_ROWS:
            return streams.mean(dim=1)
        self._log_served(
            "DSpark capture mean on the mHC stats stream", streams.shape[0]
        )
        main = torch.cuda.current_stream()
        streams.record_stream(stream)
        with torch.cuda.stream(stream):
            mean = streams.mean(dim=1)
            done = torch.cuda.Event()
            done.record()
        mean.record_stream(main)
        st.aux_done.append(done)
        return mean

    @staticmethod
    def join_aux(st: BoundaryState) -> None:
        """The main stream waits for the capture means before anything reads them."""
        if not st.aux_done:
            return
        main = torch.cuda.current_stream()
        for done in st.aux_done:
            main.wait_event(done)
        st.aux_done = []

    def _log_served(self, hop: str, rows: int) -> None:
        # engagement, from the step itself: once per (hop, width) on every rank
        if (hop, rows) not in self._served:
            self._served.add((hop, rows))
            logger.info("NVIDIA megakernel: %s serves %d rows", hop, rows)

    def _two_shot_attn(self, layer, partial, opened):
        self._log_served("two-shot attention post", partial.shape[0])
        mhc = self._mhc
        pre, post_mix, comb = self._triplet(layer.attn_hc, opened)
        norm = self._ffn_next[layer.layer_id].norm
        updated, normed = self.two_shot.attn_post(
            partial,
            opened.streams,
            post_mix,
            comb,
            pre,
            norm.weight,
            norm.variance_epsilon,
        )
        from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
            can_fuse_all_reduce,
        )

        # past SGLang's fused collective its FFN combine norms the streams itself;
        # the streams match, the in-kernel norm can differ from it by an ULP. At TP8
        # with EP2 SGLang has no fused collective at all (its combine always norms),
        # so the kernel's norm stands in for that launch, as hop A's does
        if not (
            can_fuse_all_reduce(partial.shape[0], layer.hc_cfg.hidden)
            or _tp8_ep2_fusable(layer, "attn")
        ):
            return mhc.HcState(updated, pre)
        return mhc.HcState(updated, pre, mhc.HcNormed(normed))

    def _two_shot_moe(self, layer, out, opened, nxt):
        """The two-shot MoE post; the next input as SGLang's MoE post builds it:
        MXFP8 where its fused quant would (an MXFP8 receiver whose norm folds, at
        up to 128 rows), else the streams only."""
        self._log_served("two-shot MoE post", out.routed.expert_weights.shape[0])
        mhc = self._mhc
        pre, post_mix, comb = self._triplet(layer.ffn_hc, opened)
        r = out.routed
        args = (r.gemm2_out, r.expanded_idx_to_permuted_idx, r.expert_weights, r.top_k)
        args += (out.shared, opened.streams, post_mix, comb)
        rows = r.expert_weights.shape[0]
        if not (nxt is not None and _quantizes(nxt, rows)):
            return mhc.HcState(self.two_shot.moe_post_only(*args), pre)
        norm = (pre, nxt.norm.weight, nxt.norm.variance_epsilon)
        q, sf = _mxfp8_buffers(
            r.expert_weights, opened.streams, self.two_shot.scale_bytes
        )
        updated, normed = self.two_shot.moe_post(*args, *norm, mxfp8_out=(q, sf))
        return mhc.HcState(updated, pre, _quantized(normed, q, sf))

    def _hop_b_takes(self, rows: int, out) -> bool:
        # the Lamport finalize adds a TP-partial shared expert before the sum
        return (
            self.hop_b
            and rows in self.hop_a.hop_b_rows
            and out.shared is not None
            and not out.shared_is_replicated
        )

    def _hop_b_moe(self, layer, out, opened, nxt):
        """Hop B: the Lamport MoE finalize; the next input as SGLang's post builds
        it at these widths: MXFP8 for an MXFP8 receiver whose norm folds, else the
        streams only."""
        from sgl_nvidia_mega.deepseek_v41.sm103.all_reduce_mhc import mxfp8_scale_bytes

        self._log_served("hop B MoE post", out.routed.expert_weights.shape[0])
        mhc = self._mhc
        pre, post_mix, comb = self._triplet(layer.ffn_hc, opened)
        r = out.routed
        rows = r.expert_weights.shape[0]
        args = (r.gemm2_out, r.expert_weights, r.expanded_idx_to_permuted_idx)
        args += (out.shared, opened.streams, post_mix, comb)
        if not (nxt is not None and _quantizes(nxt, rows)):
            updated, _ = self.hop_a.finalize(*args, None, None, 0.0, rows=rows)
            return mhc.HcState(updated, pre)
        hidden = opened.streams.shape[-1]
        row_tiles = self._row_tiles(layer.layer_id, rows)
        scale_bytes = mxfp8_ll.scale_bytes if row_tiles else mxfp8_scale_bytes
        q, sf = _mxfp8_buffers(
            r.expert_weights, opened.streams, lambda m: scale_bytes(m, hidden)
        )
        out = mxfp8_ll.RowTileMxfp8(q, sf) if row_tiles else (q, sf)
        updated, normed = self.hop_a.finalize(
            *args,
            pre,
            nxt.norm.weight,
            nxt.norm.variance_epsilon,
            rows=rows,
            mxfp8_out=out,
        )
        return mhc.HcState(updated, pre, _quantized(normed, q, sf, row_tiles))

    def _row_tiles(self, layer_id: int, rows: int) -> bool:
        """Whether hop B writes the next input's scales per 8-row tile: past 8 rows, when
        the next layer's chain reads them (wqkv_a on the wide low-latency GEMM)."""
        return (
            rows > mxfp8_ll.MAX_ROWS
            and self._next_layer.get(layer_id) in self.row_tile_layers
        )

    @staticmethod
    def residual(st: BoundaryState) -> torch.Tensor:
        """The materialized streams [rows, hc, hidden] (Engram, capture, head)."""
        return st.hc.residual

    def with_residual(self, st: BoundaryState, streams: torch.Tensor) -> None:
        """Replace the streams (an Engram injection); the pre-mix survives."""
        st.hc = st.hc.with_residual(streams)


# SGLang's fused MoE post quantizes the next input up to one 128-row scale tile
SGLANG_QUANT_MAX_ROWS = 128
MIN_TWO_SHOT_ROWS = 49


def _quantizes(nxt, rows: int) -> bool:
    return nxt.accepts_mxfp8 and nxt.norm_fusable and rows <= SGLANG_QUANT_MAX_ROWS


def _fusable(layer, kind: str, fuses: bool) -> bool:
    """The static half of SGLang's fusion gate (load-time facts of the layer); at TP8
    (MoE TP4 x EP2), where SGLang has no fused all-reduce + mHC kernels, the same
    gate without its world-size check, so the plugin's own hops serve the posts."""
    fusable = layer._can_fuse_attn_mhc if _attention(kind) else layer._can_fuse_ffn_mhc
    return (
        fuses
        and layer.boundary_pre_from_prev
        and (fusable or _tp8_ep2_fusable(layer, kind))
    )


def _tp8_ep2_fusable(layer, kind: str) -> bool:
    """SGLang's ``_can_fuse_attn_mhc`` / ``_can_fuse_ffn_mhc`` at TP8 with EP2, less
    ``fused_mhc_tp_ok`` (TP2 / TP4 only): the EP and MoE-TP sums merge into one TP
    all-reduce (moe_dp_size 1), which the hops run over the 8 ranks."""
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.layers.moe.utils import should_skip_post_experts_all_reduce
    from sglang.srt.models import deepseek_v4_mhc as mhc
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if (parallel.tp_size, parallel.moe_ep_size) != (8, 2):
        return False
    if _attention(kind):
        boundary = mhc.make_boundary(
            layer.post_attention_layernorm, accepts_mxfp8=False, hc=layer.ffn_hc
        )
        return (
            parallel.tp_size == layer.self_attn.attn_tp_size
            and boundary.norm_fusable
            and mhc.can_fuse_post(layer.hc_cfg)
        )
    return (
        get_moe_a2a_backend().is_none()
        and not layer.mlp._shared_expert_tp1
        and not should_skip_post_experts_all_reduce(is_tp_path=True)
        and mhc.can_fuse_post(layer.hc_cfg)
    )


def _mxfp8_buffers(weights, streams, scale_bytes):
    """Empty [rows, hidden] E4M3 values and swizzled 128x4 UE8M0 scales."""
    rows, hidden = weights.shape[0], streams.shape[-1]
    q = torch.empty((rows, hidden), dtype=torch.float8_e4m3fn, device=streams.device)
    sf = torch.empty(scale_bytes(rows), dtype=torch.uint8, device=streams.device)
    return q, sf


def _quantized(normed, q, sf, row_tiles: bool = False):
    from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput
    from sglang.srt.models import deepseek_v4_mhc as mhc

    rows = mxfp8_ll.RowTileMxfp8(q, sf) if row_tiles else Mxfp8SwizzledInput(q, sf)
    return mhc.HcQuantized(normed, rows)


def _attention(kind: str) -> bool:
    if kind not in ("attn", "ffn"):
        raise ValueError(f"unknown sublayer kind {kind!r}")
    return kind == "attn"
