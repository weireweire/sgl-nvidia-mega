# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot attn_indexer on CPU: which layers and steps it takes, that it runs SGLang's
decode indexer paths (candidate source, consumer, full top-k) with SGLang's kernel ops,
and that a step it takes never also runs SGLang's index query (replaced, not added)."""

import sys
import types
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.attention import chain, indexer, prepare


def _paged(chunked=False):
    return SimpleNamespace(
        deep_gemm_metadata=[1, 2] if chunked else "dg",
        compressed_seq_lens=torch.tensor([[40]]),
        page_table="pt",
        max_compressed_seq_len=40,
        compressed_page_size=64,
        topk_metadata="tm",
        rows_per_request=1,
    )


def _view(c1=None, c2=None):
    out = {
        1: torch.zeros(6, 8, dtype=torch.int32),
        2: torch.zeros(6, 8, dtype=torch.int32),
    }
    meta = SimpleNamespace(
        c1_indexer_metadata=c1,
        c2_indexer_metadata=c2,
        core_metadata=SimpleNamespace(sparse_page_indices=lambda r: out[r]),
    )
    return SimpleNamespace(metadata=meta, kv_pool=SimpleNamespace(pool="pool"))


def _fb(verify=True):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_target_verify=lambda: verify)
    )


def _plugin():
    p = indexer.StepIndexer.__new__(indexer.StepIndexer)
    p.topk_blocks, p.stream, p.check = 4, None, False
    p._served_rows, p._declined, p.compared, p.differ = set(), {}, 0, 0
    p._meta, p._serves, p._table = None, False, None
    p.fused, p._fused_logged, p._fused_now = False, set(), False
    p._fused_layer = None
    return p


@pytest.mark.parametrize(
    "c1, c2, verify, rows, want",
    [
        (_paged(), _paged(), True, 6, True),
        (_paged(), None, True, 48, True),
        (_paged(), _paged(chunked=True), True, 6, False),  # chunked: SGLang's call
        (_paged(), _paged(), False, 6, False),  # not a verify step
        (_paged(), _paged(), True, 72, False),  # past the low-latency GEMM's rows
        (None, None, True, 6, False),
    ],
)
def test_a_step_is_taken_whole_or_not_at_all(c1, c2, verify, rows, want):
    p = _plugin()
    view = _view(c1, c2)
    assert p.serves(view, rows, _fb(verify)) is want
    # decided once per forward: the same metadata keeps the decision
    view.metadata.c1_indexer_metadata = _paged(chunked=True)
    assert p.serves(view, rows, _fb(verify)) is want


def test_a_new_forward_drops_the_last_table():
    p = _plugin()
    view = _view(_paged())
    p.serves(view, 6, _fb())
    p._table = "last step's"
    p.serves(_view(_paged()), 6, _fb())
    assert p._table is None


def _ops_modules(log):
    """SGLang's kernel-op and helper modules the indexer imports, as recorders."""

    def rec(name, ret=None):
        def f(*a, **k):
            log.append(name)
            return ret(*a, **k) if callable(ret) else ret

        return f

    logits = torch.zeros(6, 40)
    mods = {
        "sglang.kernels.ops.attention.dsv4.index_logits": dict(
            deep_gemm_fp4_paged_mqa_logits=rec("paged_logits", logits),
            sparse_logits=rec("sparse_logits", logits),
        ),
        "sglang.kernels.ops.attention.dsv4.topk": dict(
            topk_transform_paged_v2=rec("topk_paged"),
            topk_transform_sparse=rec("topk_sparse"),
        ),
        "sglang.kernels.ops.attention.dsv4.candidate_blocks": dict(
            candidate_row_lens=rec("row_lens", (torch.zeros(1), torch.zeros(1))),
            amax_topk_blocks=rec("blocks", torch.zeros(1, 4)),
        ),
        "sglang.kernels.ops.attention.dsv4.candidate_table": dict(
            sort_candidate_blocks=rec("sort", torch.zeros(1, 4)),
            build_sparse_indexer_schedule=rec("schedule", torch.zeros(1)),
        ),
        "sglang.srt.layers.attention.dsv4.indexer": dict(
            topk_transform_paged_bf16=rec("topk_paged_bf16"),
            topk_transform_paged_from_metadata=rec("topk_full"),
        ),
        "sglang.srt.layers.attention.dsv4.v41_indexer.scoring": dict(
            get_index_k_cache=rec("k_cache", "k"),
        ),
    }
    patched = {}
    for name, attrs in mods.items():
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        patched[name] = m
    return mock.patch.dict(sys.modules, patched)


def test_paged_logits_groups_consecutive_rows_by_request():
    data = indexer.Query(
        torch.arange(6 * 4 * 64).view(6, 1, 4, 64),
        torch.arange(6 * 4).view(6, 1, 4),
        torch.zeros(6, 4),
    )
    paged = _paged()
    paged.rows_per_request = 3
    paged.compressed_seq_lens = torch.arange(40, 46).view(6, 1)
    paged.page_table = torch.arange(6 * 8).view(6, 8)
    with (
        _ops_modules([]),
        mock.patch.object(
            sys.modules["sglang.kernels.ops.attention.dsv4.index_logits"],
            "deep_gemm_fp4_paged_mqa_logits",
        ) as logits,
    ):
        indexer.StepIndexer._paged_logits(data, "k", paged)
    logits.assert_called_once()
    (q_fp4, q_sf), _, _, lens, table, _, _ = logits.call_args.args
    torch.testing.assert_close(q_fp4, data.q_fp4.view(2, 3, 4, 64))
    torch.testing.assert_close(q_sf, data.q_sf.view(2, 3, 4))
    torch.testing.assert_close(lens, paged.compressed_seq_lens.view(2, 3))
    torch.testing.assert_close(table, paged.page_table[::3])


@pytest.mark.parametrize(
    "source, consumer, skip_filter, skip_all, want",
    [
        (True, False, False, False, ["k_cache", "paged_logits", "topk_paged", "row_lens", "blocks", "sort", "schedule"]),
        (False, False, False, False, ["k_cache", "paged_logits", "topk_full"]),
        (True, False, True, False, ["k_cache", "paged_logits", "topk_full"]),
        (False, False, False, True, []),
    ],
)  # fmt: skip
def test_the_layer_runs_sglangs_path_for_its_role(
    source, consumer, skip_filter, skip_all, want
):
    log = []
    p = _plugin()
    attn = SimpleNamespace(
        compress_ratio=1,
        layer_id=7,
        indexer=SimpleNamespace(is_candidate_source=source, uses_candidates=consumer),
    )
    data = indexer.Query(
        torch.zeros(6, 1, 4, 64), torch.zeros(6, 1, 4), torch.zeros(6, 4)
    )
    with (
        _ops_modules(log),
        mock.patch(
            "sglang.srt.model_executor.runner_utils.capture_mode.skip_low_ratio_indexer",
            return_value=skip_all,
        ),
        mock.patch(
            "sglang.srt.model_executor.runner_utils.capture_mode.skip_candidate_filtering",
            return_value=skip_filter,
        ),
        mock.patch.object(torch.cuda, "current_stream"),
        mock.patch.object(torch.cuda, "stream", lambda s: nullcontext()),
        mock.patch.object(torch.cuda, "Event"),
        mock.patch.object(torch.Tensor, "record_stream", lambda self, s: None),
    ):
        p.stream = mock.Mock()
        p.topk(attn, data, torch.zeros(6, dtype=torch.int64), _view(_paged()))
    assert log == want
    assert (p._table is not None) is (source and not skip_filter and not skip_all)


def _attrs(obj):
    """Every attribute of SGLang's objects the indexer reads, by identity."""
    return {k: id(v) for k, v in vars(obj).items()}


@pytest.mark.parametrize("role", ["source", "consumer", "full"])
def test_a_served_layer_leaves_sglangs_state_as_it_found_it(role):
    """A layer the plugin serves writes its selection into SGLang's buffer and nothing
    else: no attribute of the view, the metadata (candidate_metadata included), the
    paged metadata or the pool is set, replaced or added, so a later step on SGLang's
    own path (a declined width, another captured graph) sees what it would without
    the slot."""
    p = _plugin()
    view = _view(_paged(), _paged())
    meta = view.metadata
    objs = (view, meta, meta.core_metadata, meta.c1_indexer_metadata, view.kv_pool)
    before = [_attrs(o) for o in objs]
    attn = SimpleNamespace(
        compress_ratio=1,
        layer_id=7,
        indexer=SimpleNamespace(
            is_candidate_source=role == "source", uses_candidates=role == "consumer"
        ),
    )
    data = indexer.Query(
        torch.zeros(6, 1, 4, 64), torch.zeros(6, 1, 4), torch.zeros(6, 4)
    )
    with (
        _ops_modules([]),
        mock.patch(
            "sglang.srt.model_executor.runner_utils.capture_mode.skip_low_ratio_indexer",
            return_value=False,
        ),
        mock.patch(
            "sglang.srt.model_executor.runner_utils.capture_mode.skip_candidate_filtering",
            return_value=False,
        ),
        mock.patch.object(torch.cuda, "current_stream"),
        mock.patch.object(torch.cuda, "stream", lambda s: nullcontext()),
        mock.patch.object(torch.cuda, "Event"),
        mock.patch.object(torch.Tensor, "record_stream", lambda self, s: None),
    ):
        p.stream = mock.Mock()
        p.serves(view, 6, _fb())
        if role == "consumer":  # the step's source published it
            p._table = indexer.Table(torch.zeros(1, 4), "s", "pb", "vl", "ready")
        p.topk(attn, data, torch.zeros(6, dtype=torch.int64), view)
    assert [_attrs(o) for o in objs] == before
    assert not hasattr(meta, "candidate_metadata")


def test_a_consumer_reads_the_sources_table():
    log = []
    p = _plugin()
    attn = SimpleNamespace(
        compress_ratio=1,
        layer_id=9,
        indexer=SimpleNamespace(is_candidate_source=False, uses_candidates=True),
    )
    data = indexer.Query(
        torch.zeros(6, 1, 4, 64), torch.zeros(6, 1, 4), torch.zeros(6, 4)
    )
    with (
        _ops_modules(log),
        mock.patch(
            "sglang.srt.model_executor.runner_utils.capture_mode.skip_low_ratio_indexer",
            return_value=False,
        ),
        mock.patch(
            "sglang.srt.model_executor.runner_utils.capture_mode.skip_candidate_filtering",
            return_value=False,
        ),
        mock.patch.object(torch.cuda, "current_stream"),
    ):
        with pytest.raises(AssertionError, match="consumer before its source"):
            p.topk(attn, data, None, _view(_paged()))
        p._table = indexer.Table(torch.zeros(1, 4), "s", "pb", "vl", "ready")
        p.topk(attn, data, None, _view(_paged()))
    assert log == ["k_cache", "k_cache", "sparse_logits", "topk_sparse"]


class _Stream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_stream(self, other):
        self.log.append(("wait", self.name, other.name))

    def wait_event(self, event):
        pass


def _low_ratio(plugin, log, *, quant=True):
    current = _Stream("main", log)
    streams = [_Stream(n, log) for n in ("kv", "query", "sources")]
    attn = SimpleNamespace(
        alt_streams=streams,
        compressor=None,
        indexer=SimpleNamespace(),
        fuse_wqa_wkv=False,
        compress_ratio=1,
    )
    q_for_wqb = chain.Mxfp8Rows(torch.zeros(6, 4), torch.zeros(4)) if quant else "bf16"
    q_ops = SimpleNamespace(
        indexer=plugin,
        wq_b_ll=quant,
        compute_q_a=lambda attn, x, qkv_a: log.append("q_a") or ("q_lora", q_for_wqb),
        compute_q_b=lambda *a: log.append("q_b") or "q",
        index_q_input=lambda attn, q_lora, q: "index_input",
        kv_to_cache=lambda *a: log.append("kv"),
    )
    view = SimpleNamespace(
        metadata=SimpleNamespace(),
        low_ratio_sources=lambda **kw: log.append(
            ("sglang_sources", kw.get("run_compressor", True))
        ),
    )
    stream_of = {"cur": current}

    def stream_ctx(s):
        prev = stream_of["cur"]
        stream_of["cur"] = s

        class Ctx:
            def __enter__(self):
                return s

            def __exit__(self, *a):
                stream_of["cur"] = prev

        return Ctx()

    with (
        mock.patch.object(torch.cuda, "current_stream", lambda: stream_of["cur"]),
        mock.patch.object(torch.cuda, "stream", stream_ctx),
        mock.patch.object(torch.Tensor, "record_stream", lambda self, s: None),
        mock.patch.object(
            indexer, "request_rows", lambda view, fb, rows: (torch.zeros(6), None)
        ),
    ):
        return prepare.low_ratio(
            attn, torch.zeros(6, 4), torch.zeros(6), _fb(), view, None, None, q_ops
        )


def test_a_taken_step_replaces_sglangs_index_query():
    log = []
    plugin = mock.Mock(check=False)
    plugin.serves.return_value = True
    plugin.query.side_effect = lambda *a: (
        log.append("plugin_query")
        or indexer.Query(torch.zeros(1), torch.zeros(1), torch.zeros(1))
    )
    plugin.topk.side_effect = lambda *a: log.append("plugin_topk")
    plugin.head_weights.side_effect = lambda attn, x: log.append(
        ("head_weights", torch.cuda.current_stream().name)
    )
    _low_ratio(plugin, log)
    assert "plugin_query" in log and "plugin_topk" in log
    # SGLang's sources call (and its _index_q_and_weights) does not run for the indexer
    assert not [e for e in log if isinstance(e, tuple) and e[0] == "sglang_sources"]
    # the head weights fork with the layer, before q_lora; the query waits for q_lora
    # and runs before the query chain; the scorer waits for it on the sources stream and
    # the main stream joins the query stream at the end
    fork = [i for i, e in enumerate(log) if e == ("wait", "query", "main")]
    assert len(fork) == 2
    assert fork[0] < log.index(("head_weights", "query")) < log.index("q_a") < fork[1]
    assert fork[1] < log.index("plugin_query") < log.index("q_b")
    assert ("wait", "sources", "query") in log and ("wait", "main", "query") in log


@pytest.mark.parametrize("serves, quant", [(False, True), (True, False)])
def test_a_declined_step_keeps_sglangs_call(serves, quant):
    log = []
    plugin = mock.Mock(check=False)
    plugin.serves.return_value = serves
    _low_ratio(plugin, log, quant=quant)
    plugin.head_weights.assert_not_called()
    plugin.query.assert_not_called()
    plugin.topk.assert_not_called()
    assert ("sglang_sources", False) in log


def test_the_check_runs_sglangs_call_on_the_same_inputs():
    log = []
    plugin = mock.Mock(check=True)
    plugin.serves.return_value = True
    plugin.query.return_value = indexer.Query(
        torch.zeros(1), torch.zeros(1), torch.zeros(1)
    )
    plugin.compare.side_effect = lambda attn, view, run: run()
    _low_ratio(plugin, log)
    assert ("sglang_sources", False) in log
    plugin.compare.assert_called_once()


def test_the_check_keeps_the_plugins_selection(monkeypatch):
    """After SGLang's rerun writes its own selection, the plugin's goes back, so the
    forward (and a CHECK run's accuracy) uses the plugin path."""
    p = _plugin()
    view = _view(_paged())
    out = view.metadata.core_metadata.sparse_page_indices(1)
    out.fill_(3)  # the plugin's selection
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attn = SimpleNamespace(compress_ratio=1)
    p.compare(attn, view, lambda: out.fill_(5))  # SGLang's differs
    assert p.compared == 1 and p.differ == 1
    assert bool((out == 3).all())


def test_a_chunked_step_is_counted():
    p = _plugin()
    for _ in range(3):
        p.serves(_view(_paged(), _paged(chunked=True)), 6, _fb())
    assert p._declined == {(6, "chunked metadata"): 3}


def test_the_check_ignores_the_emission_order(monkeypatch):
    p = _plugin()
    view = _view(_paged())
    out = view.metadata.core_metadata.sparse_page_indices(1)
    out.copy_(torch.arange(48, dtype=torch.int32).view(6, 8))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    p.compare(SimpleNamespace(compress_ratio=1), view, lambda: out.copy_(out.flip(-1)))
    assert p.differ == 0


# ---------------------------------------------- slot attn_indexer_fused: the consumers


@pytest.mark.parametrize(
    "rows, width, topk, page_size, heads, declined",
    [
        (6, 16384, 512, 256, 32, False),
        (1, 16384, 512, 128, 32, False),
        (8, 16384, 512, 64, 32, False),
        (0, 16384, 512, 256, 32, True),
        (9, 16384, 512, 256, 32, True),  # past one wave of 16-CTA clusters
        (12, 16384, 512, 256, 32, True),
        (6, 8192, 512, 256, 32, True),  # another candidate span
        (6, 16384, 4096, 256, 32, True),  # past the stock top-k's k
        (6, 16384, 512, 96, 32, True),  # pages of a non-power-of-two block count
        (6, 16384, 512, 12, 32, True),
        (6, 16384, 512, 256, 64, True),  # sharded or other index heads
    ],
)
def test_the_fused_consumer_declines_what_it_does_not_reproduce(
    rows, width, topk, page_size, heads, declined
):
    from sgl_nvidia_mega.deepseek_v41.sm103.attention import reindex_select

    assert bool(reindex_select.reason(rows, width, topk, page_size, heads)) is declined


def _consumer_call(p, rows, log):
    attn = SimpleNamespace(
        compress_ratio=1,
        layer_id=9,
        indexer=SimpleNamespace(is_candidate_source=False, uses_candidates=True),
    )
    data = indexer.Query(
        torch.zeros(rows, 1, 32, 64), torch.zeros(rows, 1, 32), torch.zeros(rows, 32)
    )
    out = torch.zeros(rows, 512, dtype=torch.int32)
    view = _view(_paged())
    view.metadata.core_metadata = SimpleNamespace(sparse_page_indices=lambda r: out)
    p._table = indexer.Table(torch.zeros(rows, 2048), "s", "pb", "vl", "ready")
    with (
        _ops_modules(log),
        mock.patch(
            "sglang.srt.model_executor.runner_utils.capture_mode.skip_low_ratio_indexer",
            return_value=False,
        ),
        mock.patch(
            "sglang.srt.model_executor.runner_utils.capture_mode.skip_candidate_filtering",
            return_value=False,
        ),
        mock.patch.object(torch.cuda, "current_stream"),
        mock.patch(
            "sgl_nvidia_mega.deepseek_v41.sm103.attention.reindex_select.reindex_select",
            side_effect=lambda *a, **kw: log.append(
                ("fused", kw["inputs_before_wait"])
            ),
        ),
        mock.patch.object(
            sys.modules["sglang.srt.layers.attention.dsv4.v41_indexer.scoring"],
            "get_index_k_cache",
            lambda **kw: log.append("k_cache") or torch.zeros(4, 256, 1, 68),
        ),
    ):
        p.topk(attn, data, None, view)


@pytest.mark.parametrize(
    "fused, rows, want",
    [
        (True, 6, ["k_cache", ("fused", True)]),
        (True, 1, ["k_cache", ("fused", True)]),
        (True, 8, ["k_cache", ("fused", True)]),
        (True, 12, ["k_cache", "sparse_logits", "topk_sparse"]),  # past 8 rows
        (False, 6, ["k_cache", "sparse_logits", "topk_sparse"]),  # slot off
    ],
)
def test_a_consumer_takes_the_fused_kernel_where_it_applies(fused, rows, want):
    """The fused launch replaces the sparse logits and the sparse top-k (one launch,
    reading its inputs before the PDL wait); past it the stock pair runs."""
    log = []
    p = _plugin()
    p.topk_blocks, p.fused = 2048, fused
    _consumer_call(p, rows, log)
    assert log == want
    assert p._fused_now is (want[-1] == ("fused", True))


def test_sources_and_full_layers_keep_the_stock_kernels_with_the_slot():
    log = []
    p = _plugin()
    p.topk_blocks, p.fused = 2048, True
    attn = SimpleNamespace(
        compress_ratio=1,
        layer_id=7,
        indexer=SimpleNamespace(is_candidate_source=False, uses_candidates=False),
    )
    data = indexer.Query(
        torch.zeros(6, 1, 32, 64), torch.zeros(6, 1, 32), torch.zeros(6, 32)
    )
    with (
        _ops_modules(log),
        mock.patch(
            "sglang.srt.model_executor.runner_utils.capture_mode.skip_low_ratio_indexer",
            return_value=False,
        ),
        mock.patch(
            "sglang.srt.model_executor.runner_utils.capture_mode.skip_candidate_filtering",
            return_value=False,
        ),
        mock.patch(
            "sgl_nvidia_mega.deepseek_v41.sm103.attention.reindex_select.reindex_select",
            side_effect=AssertionError("a full layer took the consumer kernel"),
        ),
    ):
        p.topk(attn, data, None, _view(_paged()))
    assert log == ["k_cache", "paged_logits", "topk_full"]


def test_the_fused_slot_needs_attn_indexer(monkeypatch):
    from sgl_nvidia_mega.deepseek_v41 import runtime

    for slots, needs in (
        ({"attn", "attn_indexer_fused"}, "attn_indexer"),
        ({"attn", "attn_indexer_narrow"}, "attn_indexer"),
    ):
        monkeypatch.setattr(runtime, "SLOTS", frozenset(slots))
        reasons = runtime.slot_reasons()
        assert any(f"needs slot {needs!r}" in r for r in reasons), reasons
    monkeypatch.setattr(
        runtime, "SLOTS", frozenset({"attn", "attn_indexer", "attn_indexer_fused"})
    )
    assert runtime.slot_reasons() == []
    assert "attn_indexer_fused" not in runtime.SHIPPED_SLOTS


@pytest.mark.parametrize(
    "fused, rows, want", [(True, 6, True), (True, 12, False), (False, 6, False)]
)
def test_the_fused_consumer_marks_the_selection_it_wrote(fused, rows, want):
    """Slot attn_index_after_wait reads a selection before the attention's wait only
    where the fused consumer wrote it (it fences the selection before its trigger)."""
    p = _plugin()
    p.topk_blocks, p.fused = 2048, fused
    _consumer_call(p, rows, [])
    assert p.wrote_fused(9) is want
    assert not p.wrote_fused(7)
    # a new forward forgets it
    p.serves(_view(_paged()), rows, _fb())
    assert not p.wrote_fused(9)


def test_the_after_wait_slot_is_opt_in_and_needs_kv_before_wait(monkeypatch):
    from sgl_nvidia_mega.deepseek_v41 import runtime

    assert "attn_index_after_wait" in runtime.OPT_IN_SLOTS
    assert "attn_index_after_wait" not in runtime.SHIPPED_SLOTS
    monkeypatch.setattr(runtime, "SLOTS", frozenset({"attn", "attn_index_after_wait"}))
    assert any("needs slot 'kv_before_wait'" in r for r in runtime.slot_reasons())
    monkeypatch.setattr(
        runtime,
        "SLOTS",
        frozenset({"attn", "kv_before_wait", "attn_index_after_wait"}),
    )
    assert runtime.slot_reasons() == []


@pytest.mark.parametrize(
    "slot, has_indexer, fused_wrote, want",
    [
        (True, True, False, True),  # a stock selection: after the wait
        (True, True, True, False),  # the fused consumer's: before it, as shipped
        (True, False, False, False),  # a reuse layer
        (False, True, False, False),  # slot off
    ],
)
def test_swapab_reads_a_stock_selection_after_its_wait(
    slot, has_indexer, fused_wrote, want
):
    import importlib

    step_mod = importlib.import_module("sgl_nvidia_mega.deepseek_v41.attention.step")

    attn = SimpleNamespace(
        layer_id=3,
        compress_ratio=1,
        indexer=object() if has_indexer else None,
        alt_streams=None,
    )
    st = step_mod.AttentionStep.__new__(step_mod.AttentionStep)
    st.attn = attn
    st.kv_before_wait = True
    st.index_after_wait = slot and has_indexer
    st.fused_wo_a = False
    st.freqs = None
    plugin = SimpleNamespace(wrote_fused=lambda lid: fused_wrote and lid == 3)
    st.chain = (None, SimpleNamespace(indexer=plugin))
    seen = {}

    class Stop(Exception):
        pass

    def sparse_decode(*a, **kw):
        seen.update(kw)
        raise Stop

    attn.maybe_use_decode_attn_tp = lambda fb: nullcontext()
    view = SimpleNamespace(kv_pool=SimpleNamespace(uniform_fp8=False))
    with (
        mock.patch.object(prepare, "low_ratio", return_value="q"),
        mock.patch.object(st, "takes_swapab", return_value=True),
        mock.patch.object(
            st, "_prepare_path", side_effect=lambda *a: prepare.low_ratio
        ),
        mock.patch.object(chain, "sparse_decode", side_effect=sparse_decode),
        pytest.raises(Stop),
    ):
        st.run(torch.zeros(6, 4), None, None, view=view)
    assert seen["kv_before_wait"] is True
    assert seen["extra_after_wait"] is want
