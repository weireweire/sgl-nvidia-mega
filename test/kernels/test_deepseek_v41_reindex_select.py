# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The fused indexer consumer (sm103/attention/reindex_select.py) against SGLang's
decode consumer on the same published table: ``sparse_logits`` (DeepGEMM) then
``topk_transform_sparse`` (topk_bf16_small), at 1-8 rows on SGLang's own table ops.

- The logits: bit for bit DeepGEMM's in every valid column.
- The selection: exactly the in-order outcome of the stock top-k on the stock logits
  (``canonical``: its elements above the pivot, its equal quota taken as 512 threads
  arriving in order take it, ascending columns); and the stock top-k's own output holds
  the same elements, sorted, wherever the equal quota takes every equal element.
- Short rows (at most k columns), rows past the candidate span (2048 or more blocks),
  scattered pages and selected blocks, ties, and every valid-length edge.
"""

import random

import numpy as np
import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.sm103.attention import reindex_select as rs

TOPK = 512
TOPK_BLOCKS = 2048  # DeepSeek-V4.1-Flash candidate_topk_blocks
HEADS = 32


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import deep_gemm  # noqa: F401
        from sglang.kernels.ops.attention.dsv4 import candidate_table  # noqa: F401
    except ImportError:
        return False
    return True


needs_gpu = pytest.mark.skipif(not _ready(), reason="needs an SM100-family GPU, SGLang")


def _stock_ops():
    from sglang.kernels.ops.attention.dsv4.candidate_blocks import candidate_row_lens
    from sglang.kernels.ops.attention.dsv4.candidate_table import (
        build_sparse_indexer_schedule,
        sort_candidate_blocks,
    )
    from sglang.kernels.ops.attention.dsv4.index_logits import sparse_logits
    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_sparse

    return (
        candidate_row_lens,
        sort_candidate_blocks,
        build_sparse_indexer_schedule,
        sparse_logits,
        topk_transform_sparse,
    )


class Case:
    """One consumer call: a request's rows (``rows_per_request`` consecutive positions)
    over a paged index-K pool, and the table its source published."""

    def __init__(
        self,
        lens,
        rows_per_request=None,
        page_size=256,
        seed=0,
        ties=None,
        weights="normal",
    ):
        g = torch.Generator().manual_seed(seed)
        rng = random.Random(seed)
        dev = "cuda"
        rows = len(lens)
        rows_per_request = rows_per_request or rows
        requests = rows // rows_per_request
        self.rows, self.page_size = rows, page_size
        max_len = max(lens)
        pages_per_req = -(-max_len // page_size) + 1
        pages = requests * pages_per_req + 3
        # the pool: random e2m1 bytes and UE8M0 scales near 2^0
        pool = torch.randint(
            0, 256, (pages, page_size * 68), generator=g, dtype=torch.uint8
        )
        scales = pool[:, page_size * 64 :]
        scales.copy_(
            torch.randint(118, 131, scales.shape, generator=g, dtype=torch.uint8)
        )
        if ties == "few_keys":
            # 24 distinct keys repeated over the pool: equal logits everywhere
            keys = pool[0, : page_size * 64].view(page_size, 64)[:24].clone()
            sf = scales[0].view(page_size, 4)[:24].clone()
            for p in range(pages):
                idx = torch.randint(0, 24, (page_size,), generator=g)
                pool[p, : page_size * 64].view(page_size, 64).copy_(keys[idx])
                scales[p].view(page_size, 4).copy_(sf[idx])
        elif ties == "zero":
            pool[:, : page_size * 64] = 0
        self.k_cache = pool.to(dev).view(pages, page_size, 1, 68)
        # scattered pages per request; rows of a request share the table row
        table = torch.zeros(rows, pages_per_req, dtype=torch.int32)
        perm = list(range(pages))
        rng.shuffle(perm)
        for r in range(requests):
            own = torch.tensor(perm[r * pages_per_req : (r + 1) * pages_per_req])
            table[r * rows_per_request : (r + 1) * rows_per_request] = own
        self.page_table = table.to(dev)
        self.seq_lens = torch.tensor(lens, dtype=torch.int32, device=dev)
        self.req = torch.arange(rows, dtype=torch.int32, device=dev) // rows_per_request
        # the source's block selection: the newest block and random others
        blocks = torch.full((rows, TOPK_BLOCKS), -1, dtype=torch.int32)
        for b, n in enumerate(lens):
            nb = -(-n // 8)
            if nb <= TOPK_BLOCKS:
                ids = list(range(nb))
            else:
                ids = rng.sample(range(nb - 1), TOPK_BLOCKS - 1) + [nb - 1]
            rng.shuffle(ids)
            blocks[b, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
        self.blocks = blocks.to(dev)
        q = torch.randint(
            -128, 128, (rows, 1, HEADS, 64), generator=g, dtype=torch.int8
        )
        self.q_fp4 = q.to(dev)
        sf = torch.randint(
            118, 131, (rows, 1, HEADS, 4), generator=g, dtype=torch.uint8
        )
        self.q_sf = sf.to(dev).view(torch.int32).view(rows, 1, HEADS)
        w = torch.randn(rows, HEADS, generator=g)
        if weights == "positive":
            w = w.abs()
        self.weights = (w * 0.05).to(dev)

    def publish(self):
        lens_op, sort_op, sched_op, _, _ = _stock_ops()
        _, valid_lens = lens_op(self.seq_lens, TOPK_BLOCKS)
        blocks = self.blocks.clone()
        phys = sort_op(blocks, self.seq_lens, self.page_table, self.page_size)
        schedule = sched_op(
            blocks,
            self.seq_lens,
            self.page_table,
            self.page_size,
            self.q_fp4.dtype,
            self.req,
        )
        return phys, valid_lens, schedule

    def stock(self, phys, valid_lens, schedule):
        _, _, _, logits_op, topk_op = _stock_ops()
        logits = logits_op(
            self.q_fp4,
            self.q_sf,
            self.k_cache,
            self.weights.to(torch.bfloat16),
            schedule,
            TOPK_BLOCKS,
        )
        out = torch.full((self.rows, TOPK), -7, dtype=torch.int32, device="cuda")
        topk_op(logits, valid_lens, phys, out)
        return logits, out

    def fused(self, phys, valid_lens, before_wait=False):
        out = torch.full((self.rows, TOPK), -7, dtype=torch.int32, device="cuda")
        logits = torch.zeros(
            self.rows, TOPK_BLOCKS * 8, dtype=torch.bfloat16, device="cuda"
        )
        rs.reindex_select(
            self.q_fp4,
            self.q_sf,
            self.weights,
            self.k_cache,
            phys,
            valid_lens,
            out,
            logits,
            inputs_before_wait=before_wait,
        )
        return logits, out


def _distort(bits):
    return np.where(bits & 0x8000, bits ^ 0xFFFF, bits ^ 0x8000)


def _undistort(key):
    return int(key ^ 0x8000) if key & 0x8000 else int(~key & 0xFFFF)


def _as_float(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _stock_slice(col, length):
    """The columns [s0, s1) the stock top-k's thread owning ``col`` holds."""
    vecs = -(-length // 8)
    per, extra = divmod(vecs, 512)
    v = col // 8
    if v < extra * (per + 1):
        start, count = v // (per + 1) * (per + 1), per + 1
    else:
        start = extra * (per + 1) + (v - extra * (per + 1)) // per * per
        count = per
    return start * 8, min((start + count) * 8, length)


def canonical(bits_row, length, k, phys_row):
    """The stock top-k's output when its 512 threads take the equal quota in order."""
    phys_row = phys_row.astype(np.int64)
    if length <= k:
        cols = np.arange(k)
        return np.where(
            cols < length, (phys_row[cols // 8 % len(phys_row)] << 3) | (cols & 7), -1
        )
    bits = bits_row[:length].astype(np.uint32)
    keys = _distort(bits)
    pivot_key = np.sort(keys)[::-1][k - 1]
    remain = k - int((keys > pivot_key).sum())
    pivot_raw = _undistort(int(pivot_key))
    pivot = _as_float(np.array([pivot_raw], dtype=np.uint32))[0]
    vals = _as_float(bits)
    gt, eq = vals > pivot, vals == pivot
    take = k - int(gt.sum()) if pivot_raw == 0x8000 else remain
    eq_cols = np.nonzero(eq)[0]
    s0, s1 = _stock_slice(int(eq_cols[take - 1]), length)
    a, u = int((eq_cols < s0).sum()), int((eq_cols < s1).sum())
    t = u - (take - a)
    ranks = np.arange(len(eq_cols))
    kept = eq_cols[(ranks < a) | ((ranks >= t) & (ranks < u))]
    cols = np.sort(np.concatenate([np.nonzero(gt)[0], kept]))
    assert len(cols) == k
    return (phys_row[cols // 8] << 3) | (cols & 7), int(eq.sum()) == take, gt, eq


def _check(case, before_wait=False):
    phys, valid_lens, schedule = case.publish()
    stock_logits, stock_out = case.stock(phys, valid_lens, schedule)
    fused_logits, fused_out = case.fused(phys, valid_lens, before_wait)
    torch.cuda.synchronize()
    lens = valid_lens.cpu().numpy()
    sl = stock_logits.view(torch.int16).cpu().numpy().view(np.uint16)
    fl = fused_logits.view(torch.int16).cpu().numpy().view(np.uint16)
    phys_np, so, fo = (
        phys.cpu().numpy(),
        stock_out.cpu().numpy(),
        fused_out.cpu().numpy(),
    )
    exact_quota = 0
    for b in range(case.rows):
        n = int(lens[b])
        if n > TOPK:
            np.testing.assert_array_equal(
                fl[b, :n], sl[b, :n], err_msg=f"logits row {b}"
            )
        ref = canonical(sl[b], n, TOPK, phys_np[b])
        if n <= TOPK:
            np.testing.assert_array_equal(fo[b], ref, err_msg=f"short row {b}")
            np.testing.assert_array_equal(so[b], ref)
            continue
        want, whole_quota, gt, eq = ref
        np.testing.assert_array_equal(
            fo[b], want, err_msg=f"selection row {b} (len {n})"
        )
        if whole_quota:
            exact_quota += 1
            np.testing.assert_array_equal(np.sort(so[b]), np.sort(want))
        else:
            # the stock pick among equal elements is arbitrary: same elements above the
            # pivot, the same count of equal ones
            slots = lambda cols: set(
                ((phys_np[b][cols // 8].astype(np.int64) << 3) | (cols & 7)).tolist()
            )
            above, equal = slots(np.nonzero(gt)[0]), slots(np.nonzero(eq)[0])
            got = set(so[b].tolist())
            assert above <= got and len(got - above) == TOPK - len(above)
            assert got - above <= equal
    return exact_quota


ROWS = (1, 2, 4, 6, 8)


@needs_gpu
@pytest.mark.parametrize("before_wait", (False, True))
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("seed", range(3))
def test_past_the_candidate_span(rows, seed, before_wait):
    """2048 or more blocks: the source's scattered selection, newest block partial;
    the inputs read before or after the PDL wait."""
    base = 20000 + 7919 * seed
    _check(Case([base + i for i in range(rows)], seed=seed), before_wait)


@needs_gpu
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize(
    "length", (1, 300, 511, 512, 513, 519, 520, 4097, 16383, 16384, 16385, 16391)
)
def test_valid_length_edges(rows, length):
    """Short rows take every column; the identity table to 2048 blocks; just past it."""
    _check(Case([length + i for i in range(rows)], seed=length))


@needs_gpu
@pytest.mark.parametrize("ties", ("few_keys", "zero"))
@pytest.mark.parametrize("rows", (1, 6, 8))
@pytest.mark.parametrize("weights", ("normal", "positive"))
def test_ties(ties, rows, weights):
    _check(
        Case(
            [30000 + 3 * i for i in range(rows)], seed=rows, ties=ties, weights=weights
        )
    )


@needs_gpu
@pytest.mark.parametrize("page_size", (128, 256))
def test_two_requests_and_page_sizes(page_size):
    """Two requests of four rows each, on their own pages."""
    lens = [9000, 9001, 9002, 9003, 40000, 40001, 40002, 40003]
    _check(Case(lens, rows_per_request=4, page_size=page_size, seed=page_size))


@needs_gpu
def test_bf16_weights_and_graph_replay():
    """bf16 head weights as given; a captured launch replayed on new inputs."""
    case = Case([25000 + i for i in range(6)], seed=11)
    phys, valid_lens, schedule = case.publish()
    out = torch.full((6, TOPK), -7, dtype=torch.int32, device="cuda")
    w = case.weights.clone()
    rs.reindex_select(case.q_fp4, case.q_sf, w, case.k_cache, phys, valid_lens, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rs.reindex_select(case.q_fp4, case.q_sf, w, case.k_cache, phys, valid_lens, out)
    for seed in range(3):
        other = Case([25000 + i for i in range(6)], seed=100 + seed)
        case.q_fp4.copy_(other.q_fp4)
        w.copy_(other.weights)
        graph.replay()
        want = torch.full_like(out, -7)
        rs.reindex_select(
            case.q_fp4,
            case.q_sf,
            w.to(torch.bfloat16),
            case.k_cache,
            phys,
            valid_lens,
            want,
        )
        torch.cuda.synchronize()
        assert torch.equal(out, want)
