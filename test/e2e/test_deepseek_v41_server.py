# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4.1-Flash with --enable-nvidia-megakernel on GB300, TP4, DSpark, across
the switch point of the post hops: up to 8 requests (48 rows) the routed-expert pair
and SGLang's fused posts, from 9 requests (54 rows) the two-shot hops; 12 and 16
requests step across it and back: GSM8K runs across the switch point (gross
failures).

SGLANG_NVIDIA_MEGAKERNEL_E2E_URL runs the serving tests against a server that is already up, so
the same checks also score the switch-off arm."""

import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import requests
from deepseek_v41_server import ARGS, ENV, MODEL, URL
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

from sgl_nvidia_mega.deepseek_v41.attention.indexer import CHECK_ENV
from sgl_nvidia_mega.deepseek_v41.runtime import SHIPPED_SLOTS, SLOTS_ENV
from sgl_nvidia_mega.plugin import FLAG_CLI

LAUNCH_TIMEOUT = 3600
# Few-shot GSM8K-200 (8 shots, 512 new tokens), for gross failures only; compare with
# the switch-off arm's own band for anything finer
GSM8K_BOUND = float(os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K_BOUND", "0.85"))
URL_ = URL or DEFAULT_URL_FOR_TEST


def _launch(args=ARGS, extra_env=None, extra_args=(), log=None, backend=True):
    # the product configuration: every slot the package ships, not a caller's override
    env = {
        **{k: v for k, v in os.environ.items() if k != SLOTS_ENV},
        **ENV,
        **(extra_env or {}),
    }
    return popen_launch_server(
        MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=LAUNCH_TIMEOUT,
        other_args=args + ([FLAG_CLI] if backend else []) + list(extra_args),
        env=env,
        return_stdout_stderr=log,
    )


def _gsm8k(parallel):
    host, port = re.match(r"https?://([^:/]+):(\d+)", URL_).groups()
    args = SimpleNamespace(
        num_shots=8,
        data_path=os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_GSM8K"),
        num_questions=200,
        parallel=parallel,
        max_new_tokens=512,
        host=host,
        port=int(port),
    )
    metrics = run_eval_few_shot_gsm8k(args)
    print(f"gsm8k parallel={parallel} {metrics=}")
    return metrics


class TestMegakernelVerify(CustomTestCase):
    """DSpark verify steps of 1-64 requests (6-384 rows) run the plugin."""

    @classmethod
    def setUpClass(cls):
        cls.process, cls.logs = None, ()
        if URL:
            return
        cls.logs = tuple(
            tempfile.NamedTemporaryFile("w+", suffix=s) for s in (".out", ".err")
        )
        cls.process = _launch(extra_args=["--watchdog-timeout", "600"], log=cls.logs)

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)
        for log in cls.logs:
            log.close()

    def test_gsm8k_across_the_two_shot_switch(self):
        # 12 and 16 requests: batches move between <= 8 (routed pair, SGLang's fused
        # posts) and >= 9 (two-shot hops)
        scores = {p: _gsm8k(p)["accuracy"] for p in (12, 16)}
        for parallel, accuracy in scores.items():
            self.assertGreater(accuracy, GSM8K_BOUND, f"parallel {parallel}")


# Past the indexer's candidate span: 2048 blocks of 8 compressed positions per row, at
# compress ratio 2 about 33K tokens; the pass key sits about 47K tokens before the
# question
PASSKEY = "71432"
FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
PASSKEY_PROMPT = (
    FILLER * 300
    + f"The pass key is {PASSKEY}. Remember it. {PASSKEY} is the pass key. "
    + FILLER * 2400
    + "What is the pass key? The pass key is"
)
# SGLANG_NVIDIA_MEGAKERNEL_E2E_INDEXER_CHECK=1: eager verify steps, each fused consumer
# checked against SGLang's kernels (CHECK_ENV)
INDEXER_CHECK = os.environ.get("SGLANG_NVIDIA_MEGAKERNEL_E2E_INDEXER_CHECK") == "1"


@unittest.skipIf(URL, "launches its own server with slot attn_indexer_fused")
class TestIndexerFusedConsumer(CustomTestCase):
    """Slot attn_indexer_fused at batch 1 (6 verify rows): the Reindex layers' logits and
    top-k in one launch."""

    SLOTS = (*SHIPPED_SLOTS, "attn_indexer_fused")

    @classmethod
    def setUpClass(cls):
        cls.logs = tuple(
            tempfile.NamedTemporaryFile("w+", suffix=s) for s in (".out", ".err")
        )
        env = {SLOTS_ENV: ",".join(cls.SLOTS)}
        extra = ["--watchdog-timeout", "600"]
        if INDEXER_CHECK:
            env[CHECK_ENV] = "1"
            extra.append("--disable-decode-cuda-graph")
        cls.process = _launch(extra_env=env, extra_args=extra, log=cls.logs)

    @classmethod
    def tearDownClass(cls):
        terminate_and_kill_process_tree(cls.process)
        for log in cls.logs:
            log.close()

    def _log(self) -> str:
        return "".join(Path(f.name).read_text() for f in self.logs)

    def test_passkey_past_the_indexers_candidate_span(self):
        r = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": PASSKEY_PROMPT,
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=1800,
        )
        r.raise_for_status()
        out = r.json()
        prompt_tokens = out["meta_info"]["prompt_tokens"]
        print(f"passkey: {prompt_tokens} prompt tokens -> {out['text']!r}")
        self.assertGreater(prompt_tokens, 40000)
        self.assertIn(PASSKEY, out["text"])
        self.assertIn("fused consumer serves 6 rows", self._log())

    def test_gsm8k_at_batch_1(self):
        accuracy = _gsm8k(1)["accuracy"]
        self.assertGreater(accuracy, GSM8K_BOUND)
        log = self._log()
        self.assertIn("fused consumer serves 6 rows", log)
        if INDEXER_CHECK:
            checks = re.findall(r"indexer check: (\d+) of (\d+) layer selections", log)
            self.assertTrue(checks, "no indexer check ran")
            self.assertEqual(checks[-1][0], "0", f"selections differ: {checks[-1]}")


@unittest.skipIf(URL, "launches its own server with slot attn_index_after_wait")
class TestIndexSelectionAfterWait(TestIndexerFusedConsumer):
    """Slot attn_index_after_wait with the fused consumer: the attention of the indexer
    layers reads a stock top-k's selection after its PDL wait."""

    SLOTS = (*SHIPPED_SLOTS, "attn_indexer_fused", "attn_index_after_wait")

    def test_the_slot_is_built(self):
        self.assertIn("a stock selection after the wait", self._log())


if __name__ == "__main__":
    unittest.main()
