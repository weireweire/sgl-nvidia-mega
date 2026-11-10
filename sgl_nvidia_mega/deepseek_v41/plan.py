# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Which kernel fills each slot of a routed step, per width (rows), fixed at build.

The row bounds are the measured ones: the wo_b publish + Lamport hop wins at 1-8
rows (FlashInfer's low-latency GEMM takes at most 8), the routed-expert pair at
1-48 rows, and the fast gate top-k only at 1-16 and 48 rows (serving measured it
slower at 24). Everything else is SGLang's own op.
"""

from __future__ import annotations

import msgspec

MAX_ROWS = 384  # the widest covered step: the two-shot hops' mailbox
ROUTED_MAX_ROWS = 48  # the largest width of the routed-expert pair
TWO_SHOT_MIN_ROWS = 49  # measured slower than SGLang's fused kernel at 37 rows
# hop A: the publishing wo_b GEMM takes 1-8 rows on 128x4 scales and, with slot
# hop_a_batch2, 9-16 on the fused wo_a's row-tile scales; the mailbox holds this many rows
HOP_A_MAX_ROWS = 16
HOP_A_BATCH1_ROWS = 8
# hop B (the MoE post) serves hop A's widths; with slot hop_b_wide up to the routed pair's
# widest step, and past 8 rows it writes the next input's scales per 8-row tile for
# wqkv_a's wide low-latency GEMM
HOP_B_MAX_ROWS = 48
FAST_GATE_MAX_SMALL_ROWS = 16
FAST_GATE_ROWS = frozenset(range(1, FAST_GATE_MAX_SMALL_ROWS + 1)) | {ROUTED_MAX_ROWS}
# slot gate_wide: every routed width, to re-measure the 17-47-row cap (one launch per arm)
FAST_GATE_WIDE_ROWS = frozenset(range(1, ROUTED_MAX_ROWS + 1))
# the Engram prefetch serves batch 1 only (6 rows): at batch 2 (12 rows) the A/B measured
# +78 us, the quarters' HBM reads overlapping a MoE that is no longer latency-bound
ENGRAM_PREFETCH_MAX_ROWS = 8
# Slots that lower numerics below what the checkpoint ships, for measurement only:
# draft_lm_head_mxfp8 runs the DSpark draft's LM head on an MXFP8 copy, while the released
# DeepSeek-V4.1-Flash ships that head in BF16 (one `head.weight`, shared by the target and
# the draft); InferenceX's review rules forbid a draft head below the checkpoint's
# precision. target_lm_head_mxfp8 (the verifier's head) is not here: it is gated on evals.
MEASUREMENT_ONLY_SLOTS = frozenset({"draft_lm_head_mxfp8"})


class Slots(msgspec.Struct, frozen=True):
    hop_a: bool  # wo_b publishes into the Lamport all-reduce + mHC
    fast_gate: bool  # the router's top-k runs in gate_topk.cu
    routed: bool  # the routed experts run on the GEMM pair
    embed: bool = False  # the embedding looks up this rank's whole table
    engram_prefetch: bool = False  # Engram lookups + WKVs run on the prefetch stream
    two_shot: bool = False  # both posts run the two-shot all-reduce + mHC
    # the Engram hashes forked at the step start, the embedding written into the streams
    prologue_fork: bool = False
    # the Engram prefetch's two all-reduces on its own stream, joined at the first gate
    engram_side_ar: bool = False


class Facts(msgspec.Struct, frozen=True):
    """What build found, which every rank must agree on. All off: every step
    function runs its stock code."""

    hop_a: bool = False  # the wo_b publish and the Lamport mailbox built
    hop_a_reason: str = ""
    routed: bool = False  # the routed-expert GEMM pair (and its top-k) built
    embed: bool = False  # the whole embedding table replicated on every rank
    engram_prefetch: bool = False  # every Engram layer prefetches, checked at build
    two_shot: bool = False  # the two-shot mailbox and kernels built
    fast_gate_wide: bool = False  # slot gate_wide: the fast top-k at every routed width
    prologue_fork: bool = False  # slot prologue_fork
    engram_side_ar: bool = False  # slot engram_side_ar, with the Engram prefetch built


def max_rows(two_shot: bool) -> int:
    """The widest covered step: past the routed pair only the two-shot hops serve."""
    return MAX_ROWS if two_shot else ROUTED_MAX_ROWS


def hop_a_max_rows(batch2: bool) -> int:
    return HOP_A_MAX_ROWS if batch2 else HOP_A_BATCH1_ROWS


def hop_b_max_rows(wide: bool, hop_a_rows: int) -> int:
    """Hop B's widest step: hop A's, or with slot hop_b_wide the routed pair's."""
    return HOP_B_MAX_ROWS if wide else hop_a_rows


class HopLayout(msgspec.Struct, frozen=True):
    """Hop A / hop B widths and the mailboxes that hold them: hop A's widths (and hop B
    at them) on a mailbox of hop A's capacity; hop B's wider widths (slot hop_b_wide)
    on a second one, so the batch-1 / batch-2 hops never address the 48-row mailbox."""

    hop_a_rows: tuple
    hop_b_rows: tuple
    capacity: int  # the hop A mailbox
    wide_capacity: int  # the hop_b_wide mailbox; 0 when no hop B width is past hop A's
    row_tiles: bool  # hop B on the hop A mailbox writes row-tile scales past 8 rows
    wide_row_tiles: bool


def hop_layout(
    widths, *, hop_b: bool, batch2: bool, hop_b_wide: bool, cap16: bool = False
) -> HopLayout:
    capacity = hop_a_max_rows(batch2)
    hop_a_rows = tuple(w for w in widths if w <= capacity)
    # slot hop_b_cap16: hop B stays on hop A's widths, so no wide mailbox is built
    hop_b_max = hop_b_max_rows(hop_b_wide and not cap16, capacity)
    hop_b_rows = tuple(w for w in widths if w <= hop_b_max) if hop_b else ()
    wide = [w for w in hop_b_rows if w > capacity]
    return HopLayout(
        hop_a_rows=hop_a_rows,
        hop_b_rows=hop_b_rows,
        capacity=capacity,
        wide_capacity=max(wide) if wide else 0,
        # only slot hop_b_wide's chains read row tiles (runtime.row_tile_layers)
        row_tiles=hop_b_wide
        and any(HOP_A_BATCH1_ROWS < w <= capacity for w in hop_b_rows),
        wide_row_tiles=bool(wide),
    )


def fast_gate_rows(wide: bool) -> frozenset:
    return FAST_GATE_WIDE_ROWS if wide else FAST_GATE_ROWS


def slots(rows: int, facts: Facts) -> Slots:
    if not 1 <= rows <= max_rows(facts.two_shot):
        raise ValueError(
            f"{rows} rows are outside the covered widths 1-{max_rows(facts.two_shot)}"
        )
    return Slots(
        hop_a=facts.hop_a and rows <= HOP_A_MAX_ROWS,
        fast_gate=facts.routed and rows in fast_gate_rows(facts.fast_gate_wide),
        routed=facts.routed and rows <= ROUTED_MAX_ROWS,
        embed=facts.embed,
        engram_prefetch=facts.engram_prefetch and rows <= ENGRAM_PREFETCH_MAX_ROWS,
        two_shot=facts.two_shot and rows >= TWO_SHOT_MIN_ROWS,
        prologue_fork=facts.prologue_fork,
        engram_side_ar=facts.engram_side_ar
        and facts.engram_prefetch
        and rows <= ENGRAM_PREFETCH_MAX_ROWS,
    )


def plan(widths: tuple[int, ...], facts: Facts) -> dict[int, Slots]:
    return {w: slots(w, facts) for w in widths}
