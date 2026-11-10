# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Microbenchmark: the sharded DSpark Markov chain (sm103/markov.py), microseconds per launch.

One process per GPU, set up as serving builds it (vocab 129,280, rank dim 256, 5 draft
positions, FP32 step-logits cache, BF16 local base-logits shard, max_bs 64). For every row
count it captures a CUDA graph of --inner launches and replays it on new random base logits
and anchors, timing each replay with CUDA events on rank 0. With --l2 cold (default) every
launch follows a write of 2x the L2 size (serving runs the whole model between two draft
chains); a graph of the writes alone is replayed too and subtracted per replay.

  python benchmark/bench_deepseek_v41_markov.py --world 4 --rows 1,2,4,8,16,32,64

Under torchrun (RANK / WORLD_SIZE set, e.g. world 8 over two GB300 trays) every process
is one rank and --world is ignored.
"""

import argparse
import os
import socket
import statistics

import torch
import torch.distributed as dist

from sgl_nvidia_mega.deepseek_v41.sm103.markov import ShardedMarkov

VOCAB = 129_280
RANK_DIM = 256
STEPS = 5
MAX_BS = 64
KEY = 0x1234_5678_9ABC_DEF


def _worker(rank: int, args) -> None:
    world = args.world
    local_rank = rank
    if args.torchrun:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(args.port)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    group = None
    if world > 1:
        dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world)
        group = dist.new_group(list(range(world)), backend="gloo")
    g = torch.Generator(device=device).manual_seed(1234)  # same weights on every rank
    embed = (torch.randn(VOCAB, RANK_DIM, generator=g, device=device) * 0.05).bfloat16()
    head = (torch.randn(VOCAB, RANK_DIM, generator=g, device=device) * 0.05).bfloat16()
    chain = ShardedMarkov(
        embed_weight=embed,
        head_weight=head,
        vocab_size=VOCAB,
        num_steps=STEPS,
        cache_dtype=torch.float32,
        group=group,
        rank=rank,
        world=world,
        max_bs=MAX_BS,
        deadline_s=20.0,
        keeps_logits=True,
    )
    vocab_local = chain.vocab_local
    temperature = 0.0 if args.temps == "greedy" else 1.0
    temperatures = torch.full((MAX_BS,), temperature, device=device)
    offset = torch.tensor([7], dtype=torch.int64, device=device)
    cache = torch.zeros(MAX_BS, STEPS, VOCAB, device=device)
    base = torch.zeros(MAX_BS, STEPS, vocab_local, dtype=torch.bfloat16, device=device)
    anchor = torch.zeros(MAX_BS, dtype=torch.int64, device=device)
    tokens = torch.zeros(MAX_BS, STEPS, dtype=torch.int64, device=device)
    l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
    flush = torch.empty(
        2 * l2_bytes if args.l2 == "cold" else 0, dtype=torch.uint8, device=device
    )
    if rank == 0:
        print(
            f"# {torch.cuda.get_device_name(device)} world={world} ctas={chain.ctas} "
            f"narrow_rows={chain.narrow_rows} wide_rows={chain.wide_rows} "
            f"temps={args.temps} inner={args.inner} l2={args.l2}",
            flush=True,
        )
    for rows in [int(r) for r in args.rows.split(",")]:

        def body(flush_only: bool):
            for _ in range(args.inner):
                if flush.numel():
                    flush.zero_()
                if flush_only:
                    continue
                chain.sample(
                    base_logits=base[:rows],
                    anchor_tokens=anchor[:rows],
                    temperatures=temperatures,
                    logits_cache=cache,
                    draft_tokens=tokens[:rows],
                )

        if args.temps != "greedy":
            chain.launch_noise(rows=rows, key=KEY, offset=offset)
        graphs = {}
        for flush_only in (False, True) if flush.numel() else (False,):
            body(flush_only)  # warm-up outside capture
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                body(flush_only)
            graphs[flush_only] = graph
        times = {key: [] for key in graphs}
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for replay in range(args.warmup + args.replays):
            gen = torch.Generator(device=device).manual_seed(replay)
            anchor.copy_(
                torch.randint(0, VOCAB, (MAX_BS,), generator=gen, device=device)
            )
            gen_local = torch.Generator(device=device).manual_seed(replay * 64 + rank)
            base.copy_(torch.randn(base.shape, generator=gen_local, device=device) * 3)
            for key in graphs:
                if group is not None:
                    torch.cuda.synchronize(device)
                    dist.barrier(group=group)
                start.record()
                graphs[key].replay()
                end.record()
                torch.cuda.synchronize(device)
                if replay >= args.warmup:
                    times[key].append(start.elapsed_time(end) * 1000.0 / args.inner)
        per_launch = times[False]
        if True in times:
            per_launch = [t - f for t, f in zip(times[False], times[True])]
        if rank == 0:
            ordered = sorted(per_launch)
            p10 = ordered[int(0.1 * len(ordered))]
            p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
            print(
                f"rows={rows} median_us={statistics.median(ordered):.2f} "
                f"p10_us={p10:.2f} p90_us={p90:.2f} replays={len(ordered)}",
                flush=True,
            )
    if group is not None:
        dist.barrier(group=group)
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--rows", default="1,2,4,8,16,32,64")
    parser.add_argument("--temps", choices=("greedy", "sampled"), default="greedy")
    parser.add_argument("--l2", choices=("cold", "hot"), default="cold")
    parser.add_argument("--inner", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--replays", type=int, default=200)
    args = parser.parse_args()
    args.torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if args.torchrun:
        args.world = int(os.environ["WORLD_SIZE"])
        _worker(int(os.environ["RANK"]), args)
        return
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        args.port = s.getsockname()[1]
    torch.multiprocessing.spawn(_worker, args=(args,), nprocs=args.world, join=True)


if __name__ == "__main__":
    main()
