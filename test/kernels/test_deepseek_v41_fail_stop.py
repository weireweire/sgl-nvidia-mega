# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fail-stop waits on 4 SM100-family GPUs (GB300): when one rank never joins, the ranks
that wait on it fault within their deadline instead of hanging: the Lamport consumer of
the wo_b publish (1-8 rows), the two-shot hop (49-384 rows), whose deadline comes
from its build alone, and the draft's MoE post on its own mailbox (slot draft_hop).

Each case runs its own torchrun, since a faulted rank leaves its CUDA context unusable.
Run with ``python test/kernels/test_deepseek_v41_fail_stop.py``.
"""

import os
import subprocess
import sys
import textwrap
import time

import pytest
import torch

# --------------------------------------------------------------------------------- hops

FAULT_DEADLINE_S = 2.0
HOP_FAULT_WORKER = textwrap.dedent(
    """
    import os, sys, time
    import torch
    from sglang.srt.distributed import parallel_state as ps
    from sglang.test.test_utils import publish_build_topology

    case, deadline = sys.argv[1], float(sys.argv[2])
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    ps.set_custom_all_reduce(True)
    ps.init_distributed_environment(world_size=world, rank=rank,
        local_rank=int(os.environ["LOCAL_RANK"]), distributed_init_method="env://")
    publish_build_topology(tp_size=world, world_rank=rank)
    ps.initialize_model_parallel()
    tp = ps.get_tp_group()
    dev = torch.device("cuda", torch.cuda.current_device())
    H, HC = 5120, 4
    if case == "lamport":
        from sgl_nvidia_mega.deepseek_v41.sm103 import all_reduce_mhc as m
        m.set_wait_deadline(deadline)
        hop = m.AllReduceMHC(hidden_size=H, hc_mult=HC, max_num_tokens=16, top_k=6, device=dev,
                             group=tp.device_group)
        rows = 6
        def run():
            x = torch.randn(rows, H, device=dev).bfloat16()
            return hop(x, torch.randn(rows, HC, H, device=dev).bfloat16(),
                       torch.rand(rows, HC, device=dev), torch.rand(rows, HC, HC, device=dev),
                       torch.rand(rows, HC, device=dev), torch.randn(H, device=dev).bfloat16(), 1e-6)
    elif case == "draft_hop":
        # the draft's mailbox as draft_hop.build makes it; its deadline is the build's
        from types import SimpleNamespace
        from sgl_nvidia_mega.deepseek_v41.boundary import draft_hop
        stage = SimpleNamespace(hc_cfg=SimpleNamespace(hidden=H, mult=HC),
            self_attn=SimpleNamespace(wo_b=SimpleNamespace(weight=torch.empty(1, device=dev))))
        hop = draft_hop.DraftHop(draft_hop._mailbox(stage, 3, tp, deadline, 48))
        rows = 5
        def run():
            # the draft MoE post: finalize, streams only
            idx = torch.arange(rows * 3, device=dev, dtype=torch.int32).view(rows, 3)
            return hop.finalize(torch.randn(rows * 3, H, device=dev).bfloat16(),
                                torch.rand(rows, 3, device=dev), idx,
                                torch.randn(rows, H, device=dev).bfloat16(),
                                torch.randn(rows, HC, H, device=dev).bfloat16(),
                                torch.rand(rows, HC, device=dev),
                                torch.rand(rows, HC, HC, device=dev), None, None, 0.0,
                                rows=rows)
    elif case == "two_shot":
        # no set_wait_deadline: the hop's deadline is its build argument only
        from sgl_nvidia_mega.deepseek_v41.sm103.two_shot_mhc import TwoShotMHC
        hop = TwoShotMHC(hidden_size=H, hc_mult=HC, max_num_tokens=384, top_k=6, device=dev,
                         group=tp.device_group, deadline_s=deadline)
        rows = 64
        def run():
            x = torch.randn(rows, H, device=dev).bfloat16()
            return hop.attn_post(x, torch.randn(rows, HC, H, device=dev).bfloat16(),
                                 torch.rand(rows, HC, device=dev), torch.rand(rows, HC, HC, device=dev),
                                 torch.rand(rows, HC, device=dev), torch.randn(H, device=dev).bfloat16(),
                                 1e-6)
    run(); torch.cuda.synchronize()  # every rank joins once: compiled and healthy
    torch.distributed.barrier(group=tp.cpu_group)
    print(f"RANK{rank} HEALTHY", flush=True)
    if rank == world - 1:
        time.sleep(4 * deadline + 20)  # never joins the second hop
        os._exit(0)
    t0 = time.time()
    try:
        run(); torch.cuda.synchronize()
        print(f"RANK{rank} NO-FAULT", flush=True)
    except Exception as e:
        print(f"RANK{rank} FAULTED after {time.time() - t0:.1f} s: {type(e).__name__}", flush=True)
    os._exit(0)
    """
)


def _gpu4() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 10
        and torch.cuda.device_count() >= 4
    )


@pytest.mark.skipif(not _gpu4(), reason="needs 4 SM100-family GPUs")
@pytest.mark.parametrize("case", ["lamport", "two_shot", "draft_hop"])
def test_a_missing_rank_faults_its_hop_peers_in_time(case, tmp_path):
    script = tmp_path / "worker.py"
    script.write_text(HOP_FAULT_WORKER)
    port = 29500 + (os.getpid() + 37 * len(case)) % 2000
    t0 = time.time()
    out = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=4",
         f"--master_port={port}", str(script), case, str(FAULT_DEADLINE_S)],
        capture_output=True, text=True, timeout=600,
    )  # fmt: skip
    log = out.stdout + out.stderr
    assert log.count("HEALTHY") == 4, log[-4000:]
    # the waiting ranks fault (the kernel's store to 0x0 surfaces as a CUDA error);
    # none returns as if the hop had completed
    assert "NO-FAULT" not in log, log[-4000:]
    faulted = [l for l in log.splitlines() if "FAULTED after" in l]
    assert len(faulted) == 3, log[-4000:]
    for line in faulted:
        assert (
            float(line.split("after ")[1].split(" s")[0]) < 4 * FAULT_DEADLINE_S + 10
        ), line
    assert time.time() - t0 < 300


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
