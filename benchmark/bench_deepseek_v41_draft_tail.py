# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Microbenchmark: the draft proposal's tail at one request, the low-M LM head on this
rank's shard followed by the sharded Markov chain, with slot markov_pdl off and on.

One process per GPU (vocab 129,280, hidden 5,120, 5 draft positions, FP32 step logits).
Each arm is a CUDA graph of --inner tails (LM head -> chain) after an L2-flushing write
(a graph of the writes alone is subtracted). Arms alternate replay by replay; rank 0
prints the median per tail of each arm and of their paired difference.

  python benchmark/bench_deepseek_v41_draft_tail.py --world 4 --temps sampled
"""

import argparse
import os
import socket
import statistics
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sgl_nvidia_mega.deepseek_v41 import sampler
from sgl_nvidia_mega.deepseek_v41.sm103.markov import ShardedMarkov

VOCAB = 129_280
HIDDEN = 5120
RANK_DIM = 256
STEPS = 5
KEY = 0x1234_5678_9ABC_DEF


def _worker(rank: int, args) -> None:
    world = args.world
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args.port)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    group = None
    if world > 1:
        dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world)
        group = dist.new_group(list(range(world)), backend="gloo")
    g = torch.Generator(device=device).manual_seed(1234)
    embed = (torch.randn(VOCAB, RANK_DIM, generator=g, device=device) * 0.05).bfloat16()
    head = (torch.randn(VOCAB, RANK_DIM, generator=g, device=device) * 0.05).bfloat16()
    local = VOCAB // world
    weight = (torch.randn(local, HIDDEN, generator=g, device=device) * 0.02).bfloat16()
    lm_head = sampler.DraftLmHead(SimpleNamespace(weight=weight), low_m=True)
    lm_head.tune((STEPS,))
    sampled = args.temps == "sampled"
    chains = {
        pdl: ShardedMarkov(
            embed_weight=embed,
            head_weight=head,
            vocab_size=VOCAB,
            num_steps=STEPS,
            cache_dtype=torch.float32,
            group=group,
            rank=rank,
            world=world,
            max_bs=4,
            deadline_s=20.0,
            keeps_logits=sampled,
            pdl=pdl,
        )
        for pdl in (False, True)
    }
    x = torch.randn(STEPS, HIDDEN, generator=g, device=device).bfloat16()
    temperatures = torch.full((4,), 1.0 if sampled else 0.0, device=device)
    offset = torch.tensor([7], dtype=torch.int64, device=device)
    cache = torch.zeros(4, STEPS, VOCAB, device=device) if sampled else None
    anchor = torch.zeros(1, dtype=torch.int64, device=device)
    tokens = torch.zeros(1, STEPS, dtype=torch.int64, device=device)
    flush = torch.empty(
        2 * torch.cuda.get_device_properties(device).L2_cache_size,
        dtype=torch.uint8,
        device=device,
    )

    def tail(chain):
        if sampled:
            chain.fork_noise(rows=1, key=KEY, offset=offset)
        if chain.pdl:
            chain.stage_anchor(anchor)
        base = lm_head(x).view(1, STEPS, -1)
        if sampled:
            chain.join_noise()
        chain.sample(
            base_logits=base,
            anchor_tokens=anchor,
            temperatures=temperatures,
            logits_cache=cache,
            draft_tokens=tokens,
            anchor_staged=chain.pdl,
        )

    def body(arm):
        for _ in range(args.inner):
            flush.zero_()
            if arm is not None:
                tail(chains[arm])

    graphs = {}
    for arm in (False, True, None):
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            body(arm)
        torch.cuda.current_stream(device).wait_stream(side)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body(arm)
        graphs[arm] = graph
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    times = {arm: [] for arm in graphs}
    for replay in range(args.warmup + args.replays):
        order = (False, True, None) if replay % 2 == 0 else (True, False, None)
        for arm in order:
            if group is not None:
                torch.cuda.synchronize(device)
                dist.barrier(group=group)
            start.record()
            graphs[arm].replay()
            end.record()
            torch.cuda.synchronize(device)
            if replay >= args.warmup:
                times[arm].append(start.elapsed_time(end) * 1000.0 / args.inner)
    if rank == 0:
        off = [t - f for t, f in zip(times[False], times[None])]
        on = [t - f for t, f in zip(times[True], times[None])]
        diff = sorted(b - a for a, b in zip(off, on))
        print(
            f"world={world} temps={args.temps} inner={args.inner} replays={len(off)}: "
            f"off {statistics.median(off):.2f} us, on {statistics.median(on):.2f} us, "
            f"paired on-off median {statistics.median(diff):.2f} "
            f"(p10 {diff[len(diff) // 10]:.2f}, p90 {diff[(9 * len(diff)) // 10]:.2f})",
            flush=True,
        )
    if group is not None:
        dist.barrier(group=group)
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--temps", choices=("greedy", "sampled"), default="sampled")
    parser.add_argument("--inner", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--replays", type=int, default=100)
    args = parser.parse_args()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        args.port = s.getsockname()[1]
    torch.multiprocessing.spawn(_worker, args=(args,), nprocs=args.world, join=True)


if __name__ == "__main__":
    main()
