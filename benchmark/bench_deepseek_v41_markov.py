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
is one rank and --world is ignored. --stamps DIR builds the narrow kernel with phase
stamps (markov.cu MARKOV_STAMPS), saves each rank's stamps of the timed launches to DIR
and prints rank 0's per-step phase table (us, median over launches and CTAs).
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
        stamps=bool(args.stamps),
        lowlat=args.lowlat,
        extra_defines=tuple(d for d in args.defines.split(",") if d),
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
            f"lowlat={int(args.lowlat)} defines={args.defines or '-'} "
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
        stamped = []
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
                if (
                    args.stamps
                    and not key
                    and replay >= args.warmup
                    and len(stamped) < args.stamp_replays
                    and rows <= chain.narrow_rows
                ):
                    stamped.append(
                        _last_stamps(chain, args.inner, rows <= chain.direct_max_rows)
                    )
        per_launch = times[False]
        if True in times:
            per_launch = [t - f for t, f in zip(times[False], times[True])]
        if stamped:
            path = os.path.join(
                args.stamps, f"stamps_w{world}_{args.temps}_r{rows}_rank{rank}.pt"
            )
            torch.save(torch.cat(stamped), path)
            if group is not None:
                dist.barrier(group=group)
            if rank == 0:
                _report_stamps(args.stamps, world, args.temps, rows, STEPS)
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


def _last_stamps(chain, launches: int, direct: bool) -> torch.Tensor:
    """The stamps of the last ``launches`` narrow launches, oldest first (CPU int64)."""
    from sgl_nvidia_mega.deepseek_v41.sm103.markov import STAMP_RING

    # CTA 0's launch counter: the direct exchange's at word 512, the fold's at 256
    count = int(chain.control[512 if direct else 256].item())
    order = [(count - launches + i) % STAMP_RING for i in range(launches)]
    return chain.stamps[order].cpu()


# Phase names of markov.cu's per-step stamp slots 8 + 8 s + p.
_PHASES = ("start", "tiles", "reduce", "posted", "seen", "decided", "embedded", "rank_post")


def _report_stamps(directory: str, world: int, temps: str, rows: int, steps: int) -> None:
    """Per-step phase table of rank 0 (clock64 intervals, us) and, from every rank's
    globaltimer, how late the last record arrives after the last post of any rank."""
    ranks = [
        torch.load(os.path.join(directory, f"stamps_w{world}_{temps}_r{rows}_rank{r}.pt"))
        for r in range(world)
    ]
    s = ranks[0].double()  # [launch, cta, slot, (gt, clk)]
    gt, clk = s[..., 0], s[..., 1]
    ghz = (clk[:, :, 3] - clk[:, :, 0]) / (gt[:, :, 3] - gt[:, :, 0])  # cycles per ns

    def interval(a: int, b: int) -> torch.Tensor:
        return (clk[:, :, b] - clk[:, :, a]) / ghz / 1000.0  # us per (launch, cta)

    def med(x: torch.Tensor) -> float:
        return float(x.flatten().median())

    def top(x: torch.Tensor) -> float:  # median over launches of the max over CTAs
        return float(x.amax(dim=1).median())

    fold = bool((s[:, :, 8 + 7, 0] != 0).any())
    # launch boundary on rank 0's globaltimer (us): first CTA entry -> last CTA entry,
    # entry -> slot 0, and last CTA exit -> next launch's first CTA entry (in a graph of
    # back-to-back launches, i.e. --l2 hot, this is the launch gap)
    entry, exit_ = gt[:, :, 4], gt[:, :, 3]
    spread = (entry.amax(dim=1) - entry.amin(dim=1)) / 1000.0
    gap = (entry[1:].amin(dim=1) - exit_[:-1].amax(dim=1)) / 1000.0
    span = (exit_.amax(dim=1) - entry.amin(dim=1)) / 1000.0
    print(
        f"## boundary: entry spread {float(spread.median()):.2f} us, entry->slot0 "
        f"{med(interval(4, 0)):.2f} us, first entry->last exit {float(span.median()):.2f} us, "
        f"exit->next entry {float(gap.median()):.2f} us"
    )
    print(
        f"## stamps world={world} temps={temps} rows={rows} launches={s.shape[0]} "
        f"ctas={s.shape[1]} exchange={'fold' if fold else 'direct'} "
        f"clock={float(ghz.median()):.3f} GHz kernel={med(interval(0, 3)):.2f} us "
        f"prologue={med(interval(0, 1)):.2f} epilogue={med(interval(2, 3)):.2f}"
    )
    print(
        "| step | tiles | reduce | post | wait | decide | embed | step | wait max | "
        "local skew | after last post |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for step in range(steps):
        b = 8 + 8 * step
        prev = 1 if step == 0 else 8 + 8 * (step - 1) + 6
        end = b + 6 if step + 1 < steps else 2
        # within the rank: how much later than this CTA's post the rank's last post was
        post_slot = b + 7 if fold else b + 3
        posts = gt[:, :, post_slot]
        if fold:
            rank_post = posts.amax(dim=1, keepdim=True)
            local_skew = (rank_post - gt[:, :, b + 3]) / 1000.0
        else:
            local_skew = (posts.amax(dim=1, keepdim=True) - posts) / 1000.0
        # across ranks (globaltimers of one node agree only roughly; read with care)
        last_post = torch.stack(
            [r[:, :, post_slot, 0].double().amax(dim=1) for r in ranks]
        ).amax(dim=0)
        # the CTA's last record seen, on its globaltimer through its clock calibration
        seen = gt[:, :, 0] + (clk[:, :, b + 4] - clk[:, :, 0]) / ghz
        after = (seen - last_post[:, None]) / 1000.0
        print(
            f"| {step} | {med(interval(b, b + 1)):.2f} | {med(interval(b + 1, b + 2)):.2f} | "
            f"{med(interval(b + 2, b + 3)):.2f} | {med(interval(b + 3, b + 4)):.2f} | "
            f"{med(interval(b + 4, b + 5)):.2f} | "
            f"{med(interval(b + 5, b + 6)) if step + 1 < steps else 0.0:.2f} | "
            f"{med(interval(prev, end)):.2f} | {top(interval(b + 3, b + 4)):.2f} | "
            f"{med(local_skew):.2f} | {med(after):.2f} |",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--rows", default="1,2,4,8,16,32,64")
    parser.add_argument("--temps", choices=("greedy", "sampled"), default="greedy")
    parser.add_argument("--l2", choices=("cold", "hot"), default="cold")
    parser.add_argument("--inner", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--stamps", default="", help="directory for phase stamps")
    parser.add_argument("--lowlat", action="store_true", help="slot markov_lowlat's kernel")
    parser.add_argument("--defines", default="", help="extra NVRTC defines (experiments)")
    parser.add_argument("--stamp-replays", type=int, default=20)
    args = parser.parse_args()
    if args.stamps:
        os.makedirs(args.stamps, exist_ok=True)
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
