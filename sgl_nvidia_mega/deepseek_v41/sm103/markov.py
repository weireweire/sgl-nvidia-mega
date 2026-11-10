# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DSpark draft's Markov chain in one cooperative, vocab-sharded kernel per rank.

Problem. The DSV4.1 DSpark draft samples ``num_steps`` tokens per request row, one after
the other: step i adds the bias ``markov_head . markov_embed[previous token]`` to the
base logits, picks the argmax (temperature 0) or the Gumbel-max sample, and keeps the
step logits for the accept. Run step by step, every step is a GEMV, a sample and a
cross-rank exchange of the winner.

Responsibility. :class:`ShardedMarkov` runs all steps of all rows in one launch on each
rank (markov.cu). Rank k keeps rows ``[k V / world, (k + 1) V / world)`` of the Markov head
in registers, adds the bias to its shard of the base logits, picks its best (score,
index) key and exchanges one 64-bit key per rank, request row and step through a mailbox
in symmetric memory; the kept step logits reach every rank through peer staging. Up to
``narrow_rows`` rows run the narrow kernel (the direct exchange at one row, the
rank-local fold at 2-8), more rows (up to 64) the wide kernel. Every cross-rank wait is
bounded by ``deadline_s``; a rank whose peer stops faults instead of hanging.

Numerics. Tokens and kept logits equal SGLang's ``draft_markov_block`` step by step when
the bias products sum exactly: the bias is rounded to BF16, added in FP32 and rounded
once to the cache dtype; the noise is SGLang's Philox draft noise (``markov_sample.py``:
same key, counters and offset), drawn for this rank's shard before the chain. The
caller advances the noise offset per block, as ``draft_markov_block`` does.
"""

from __future__ import annotations

import ctypes
import functools
from pathlib import Path
from typing import NamedTuple, Optional

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sgl_nvidia_mega.common.jit import check as _check
from sgl_nvidia_mega.common.jit import compile_source as _compile_source
from sgl_nvidia_mega.common.jit import driver as _driver
from sgl_nvidia_mega.common.jit import pointer as _pointer

SOURCE = Path(__file__).resolve().parent / "markov.cu"
NARROW_SYMBOL = "dsv41_markov_narrow"
WIDE_SYMBOL = "dsv41_markov_wide"
NVRTC_OPTIONS = ("-std=c++17", "-default-device")
_C_TYPES = {torch.bfloat16: "__nv_bfloat16", torch.float32: "float"}

# Must match the constants of markov.cu.
RANK_DIM = 256
THREADS = 512
WARPS = THREADS // 32
MAX_STEPS = 8
MAX_RANKS = 8
# MMA columns: request rows of the narrow kernel.
MAX_ROWS = 8
# Request rows of the wide kernel.
WIDE_ROWS = 64
# The candidate key's 17-bit index field.
MAX_VOCAB = (1 << 17) - 1
# Element types of the base logits and of the step-logits cache.
LOGIT_DTYPES = (torch.bfloat16, torch.float32)
# Peer buffer bytes before the staging logits: the fold's mailbox records.
STAGING_OFFSET = 16384
# Per-CTA launch counters at words [256, 512) (fold) and [512, 768) (direct).
CONTROL_WORDS = 768
MAX_CTAS = 256
# The wide kernel's control words: combine, arrivals, per-CTA counters at 1280.
WIDE_CONTROL_WORDS = 1280 + MAX_CTAS
WIDE_RECORD_BYTES = 2 * MAX_STEPS * WIDE_ROWS * MAX_RANKS * 8
NOISE_BLOCK = 1024

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))
SUPPORTED_WORLDS = (1, 2, 4, MAX_RANKS)
# Measured on GB300 TP4: the direct exchange wins at 1 row, the rank-local fold at 2-8.
DIRECT_MAX_ROWS = 1

# markov_sample.py's Philox draft noise.
_UNIFORM_SCALE = tl.constexpr(2.3283064365386963e-10)
_PHILOX_ROUNDS = tl.constexpr(10)


@triton.jit
def _shard_noise_kernel(
    NOISE,
    OFFSET,
    KEY,
    VOCAB_START,
    VOCAB_LOCAL,
    STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gumbel noise of (row, step) for this rank's vocab shard, as SGLang's draft step draws it."""
    sample = tl.program_id(0)
    row = sample // STEPS
    step = sample % STEPS
    part = tl.program_id(1)
    quad = (part * BLOCK + VOCAB_START) // 4 + tl.arange(0, BLOCK // 4)
    lane = tl.zeros([BLOCK // 4], tl.int64)
    w0, w1, w2, w3 = tl.philox(
        KEY,
        quad.to(tl.uint32),
        (lane + row).to(tl.uint32),
        (lane + step).to(tl.uint32),
        ((lane + tl.load(OFFSET)) & 0xFFFFFFFF).to(tl.uint32),
        _PHILOX_ROUNDS,
    )
    bits = tl.interleave(tl.interleave(w0, w2), tl.interleave(w1, w3))
    uniform = (bits.to(tl.float32) + 0.5) * _UNIFORM_SCALE
    exp_noise = -libdevice.log1p(-uniform)
    gumbel = -tl.log(exp_noise)
    cols = part * BLOCK + tl.arange(0, BLOCK)
    tl.store(NOISE + sample * VOCAB_LOCAL + cols, gumbel, mask=cols < VOCAB_LOCAL)


def unsupported_reason(
    *,
    world: int,
    vocab_size: int,
    num_steps: int,
    embed_weight: torch.Tensor,
    head_weight: torch.Tensor,
    cache_dtype: torch.dtype,
    capability: tuple[int, int],
) -> Optional[str]:
    """Why the chain kernel cannot serve this configuration, or None; the same on every rank."""
    if world not in SUPPORTED_WORLDS:
        return f"a tensor-parallel size of {world}"
    if tuple(capability) not in SUPPORTED_CAPABILITIES:
        return f"compute capability {capability} (needs SM100 or SM103)"
    if not 1 <= num_steps <= MAX_STEPS:
        return f"{num_steps} draft positions"
    if vocab_size > MAX_VOCAB or vocab_size % (16 * world) != 0:
        return f"a vocabulary of {vocab_size} for {world} ranks"
    if cache_dtype not in LOGIT_DTYPES:
        return f"a {cache_dtype} step-logits buffer"
    for name, weight in (("embedding", embed_weight), ("head", head_weight)):
        if (
            weight.dim() != 2
            or weight.shape[1] != RANK_DIM
            or weight.dtype != torch.bfloat16
            or not weight.is_contiguous()
        ):
            return f"a Markov {name} that is not contiguous BF16 [*, {RANK_DIM}]"
    if head_weight.shape[0] < vocab_size or embed_weight.shape[0] < vocab_size:
        return "a Markov table smaller than the vocabulary"
    return None


def narrow_row_limit(*, world: int, max_bs: int) -> int:
    """Request rows of the narrow kernel: its exchange gives each (row, rank) pair one lane."""
    return max(1, min(MAX_ROWS, 32 // world, max_bs))


def direct_mailbox_bytes(*, steps: int, world: int, ctas: int, max_rows: int) -> int:
    """Direct-exchange records u64 [parity 2][step][source rank][cta][row]."""
    return 2 * steps * world * ctas * max_rows * 8


def nvrtc_defines(
    *, steps: int, base_dtype: torch.dtype, cache_dtype: torch.dtype, direct: bool
) -> tuple[str, ...]:
    return (
        f"-DBASE_T={_C_TYPES[base_dtype]}",
        f"-DCACHE_T={_C_TYPES[cache_dtype]}",
        f"-DMARKOV_DIRECT={int(direct)}",
        f"-DSTEPS={steps}",
    )


def compile_cubin(
    *,
    steps: int,
    base_dtype: torch.dtype,
    cache_dtype: torch.dtype,
    direct: bool,
    arch: str,
) -> bytes:
    """The module of one exchange: the narrow kernel, plus the wide kernel with the fold."""
    options = NVRTC_OPTIONS + nvrtc_defines(
        steps=steps, base_dtype=base_dtype, cache_dtype=cache_dtype, direct=direct
    )
    return _compile_source(SOURCE.read_text(), "markov", arch, options)


class _Kernel(NamedTuple):
    function: object  # cuda.bindings.driver.CUfunction
    max_ctas: int  # co-resident CTAs on the device (the cooperative grid's bound)


@functools.cache
def load_kernels(
    device_index: int,
    steps: int,
    base_dtype: torch.dtype,
    cache_dtype: torch.dtype,
    direct: bool,
) -> dict[str, _Kernel]:
    """Compile (or read from the cache) and load one exchange's kernels."""
    cu = _driver()
    device = torch.device("cuda", device_index)
    major, minor = torch.cuda.get_device_capability(device)
    cubin = compile_cubin(
        steps=steps,
        base_dtype=base_dtype,
        cache_dtype=cache_dtype,
        direct=direct,
        arch=f"sm_{major}{minor}a",
    )
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    symbols = {"narrow": NARROW_SYMBOL}
    if not direct:
        symbols["wide"] = WIDE_SYMBOL
    kernels = {}
    with torch.cuda.device(device):
        torch.zeros(1, device=device)  # make the primary context current
        (module,) = _check(cu.cuModuleLoadData(cubin), "cuModuleLoadData")
        for name, symbol in symbols.items():
            (function,) = _check(
                cu.cuModuleGetFunction(module, symbol.encode()), symbol
            )
            (per_sm,) = _check(
                cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(function, THREADS, 0),
                "cuOccupancyMaxActiveBlocksPerMultiprocessor",
            )
            kernels[name] = _Kernel(function, int(per_sm) * sms)
    return kernels


def _launch_cooperative(
    kernel: _Kernel, ctas: int, arguments: list, stream: int, pdl: bool = False
) -> None:
    """Cooperative launch (co-resident grid), with ``pdl`` also a programmatic dependent
    of the stream's previous kernel; graph-capturable."""
    cu = _driver()
    addresses = [ctypes.addressof(argument) for argument in arguments]
    pointer_array = (ctypes.c_void_p * len(addresses))(*addresses)
    config = cu.CUlaunchConfig()
    config.gridDimX, config.gridDimY, config.gridDimZ = ctas, 1, 1
    config.blockDimX, config.blockDimY, config.blockDimZ = THREADS, 1, 1
    config.sharedMemBytes = 0
    config.hStream = cu.CUstream(stream)
    attribute = cu.CUlaunchAttribute()
    attribute.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_COOPERATIVE
    attribute.value.cooperative = 1
    config.attrs = [attribute]
    if pdl:
        early = cu.CUlaunchAttribute()
        early.id = (
            cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        )
        early.value.programmaticStreamSerializationAllowed = 1
        config.attrs = [attribute, early]
    config.numAttrs = len(config.attrs)
    _check(
        cu.cuLaunchKernelEx(
            config, kernel.function, ctypes.addressof(pointer_array), 0
        ),
        "cuLaunchKernelEx",
    )


def _all_ranks_true(group, value: bool) -> bool:
    """``value`` on every rank of the CPU ``group`` (None: this rank only)."""
    if group is None:
        return value
    flag = torch.tensor([int(value)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


class ShardedMarkov:
    """Persistent buffers, peer mappings and launchers of the sharded chain.

    Every rank of ``group`` (the shard's CPU process group, None for one rank) must
    construct it: the constructor runs CPU collectives. It compiles every kernel a
    launch may pick and checks their co-residency before the collective set-up, and
    raises on every rank if any rank failed.
    """

    def __init__(
        self,
        *,
        embed_weight: torch.Tensor,
        head_weight: torch.Tensor,
        vocab_size: int,
        num_steps: int,
        cache_dtype: torch.dtype,
        group,
        rank: int,
        world: int,
        max_bs: int,
        deadline_s: float,
        keeps_logits: bool,
        pdl: bool = False,
    ):
        device = embed_weight.device
        reason = unsupported_reason(
            world=world,
            vocab_size=vocab_size,
            num_steps=num_steps,
            embed_weight=embed_weight,
            head_weight=head_weight,
            cache_dtype=cache_dtype,
            capability=torch.cuda.get_device_capability(device),
        )
        if reason is not None:
            raise ValueError(f"sharded Markov sampling does not support {reason}")
        if deadline_s <= 0:
            raise ValueError(
                f"sharded Markov sampling needs a deadline, got {deadline_s}"
            )
        self.group, self.rank, self.world = group, rank, world
        self.device = device
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.cache_dtype = cache_dtype
        # without it the peer buffer has no staging for kept logits
        self.keeps_logits = keeps_logits
        self.timeout_ns = int(deadline_s * 1e9)
        # slot markov_pdl: launched as a programmatic dependent of the LM head, the
        # chain loads its head tiles while the LM head drains (stage_anchor first)
        self.pdl = pdl
        self.vocab_local = vocab_size // world
        self.vocab_offset = rank * self.vocab_local
        self.embed_weight = embed_weight
        self.head_shard = head_weight[
            self.vocab_offset : self.vocab_offset + self.vocab_local
        ]
        tiles = self.vocab_local // 16
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.ctas = min(sms, -(-tiles // WARPS))
        if self.ctas > MAX_CTAS:
            raise ValueError(
                f"the Markov chain counts launches for up to {MAX_CTAS} CTAs, not {self.ctas}"
            )
        self.narrow_rows = narrow_row_limit(world=world, max_bs=max_bs)
        # The wide kernel takes more rows when every warp owns one head tile.
        wide = max_bs > self.narrow_rows and self.ctas * WARPS >= tiles
        self.wide_rows = min(WIDE_ROWS, max_bs) if wide else 0
        self.max_rows = max(self.narrow_rows, self.wide_rows)
        self.noise = torch.empty(
            self.max_rows * num_steps,
            self.vocab_local,
            dtype=torch.float32,
            device=device,
        )
        self.control = torch.zeros(CONTROL_WORDS, dtype=torch.int32, device=device)
        # Request row r reads temperature[r] and keeps its step logits at cache[r].
        rows = torch.arange(self.max_rows, dtype=torch.int32, device=device)
        self.sampled_rows = rows.repeat_interleave(num_steps)
        self.inert_rows = torch.full_like(self.sampled_rows, -1)
        self.anchor = torch.zeros(self.max_rows, dtype=torch.int32, device=device)
        size = self._layout_peer_buffer(keeps_logits=keeps_logits)
        self.wide_control = (
            torch.zeros(WIDE_CONTROL_WORDS, dtype=torch.int32, device=device)
            if self.wide_rows
            else None
        )
        # Compile before the collective set-up so a build failure cannot leave peers waiting.
        self._kernels: dict[tuple[torch.dtype, str], _Kernel] = {}
        failure = None
        try:
            self._load_kernels()
        except (RuntimeError, ImportError) as error:
            failure = error
        if not _all_ranks_true(group, failure is None):
            raise RuntimeError(
                f"kernel set-up failed on a rank: {failure or 'another rank'}"
            )
        self._peer_buffer, peer_pointers = self._allocate_peer_buffer(size)
        self.peers = torch.tensor(peer_pointers, dtype=torch.int64, device=device)
        self._noise_stream = torch.cuda.Stream(device)

    def _layout_peer_buffer(self, *, keeps_logits: bool) -> int:
        """Byte size of the peer buffer; sets the direct mailbox and wide-kernel offsets.

        Fold records, then staging logits [exchange 2][parity 2][narrow rows] (none for a
        greedy draft, which never writes them), then the direct mailbox, then the wide
        kernel's records and staging logits [parity 2][wide rows].
        """
        row_bytes = self.num_steps * self.vocab_size * self.cache_dtype.itemsize
        if not keeps_logits:
            row_bytes = 0
        size = STAGING_OFFSET + 4 * self.narrow_rows * row_bytes
        self.mailbox_offset = size
        size += direct_mailbox_bytes(
            steps=self.num_steps,
            world=self.world,
            ctas=self.ctas,
            max_rows=self.narrow_rows,
        )
        self.wide_offset = -(-size // 16) * 16
        if self.wide_rows:
            size = self.wide_offset + WIDE_RECORD_BYTES
            size += 2 * self.wide_rows * row_bytes
        return size

    def _kernel_names(self) -> list[str]:
        """``narrow-direct``, ``narrow-fold`` and ``wide``: those a launch may pick."""
        names = []
        for rows in range(1, self.narrow_rows + 1):
            name = "narrow-direct" if rows <= DIRECT_MAX_ROWS else "narrow-fold"
            if name not in names:
                names.append(name)
        if self.wide_rows:
            names.append("wide")
        return names

    def _load_kernels(self) -> None:
        index = self.device.index
        if index is None:
            index = torch.cuda.current_device()
        for base_dtype in LOGIT_DTYPES:
            for name in self._kernel_names():
                module = load_kernels(
                    index,
                    self.num_steps,
                    base_dtype,
                    self.cache_dtype,
                    name == "narrow-direct",
                )
                kernel = module["wide" if name == "wide" else "narrow"]
                if self.ctas > kernel.max_ctas:
                    raise RuntimeError(
                        f"sharded Markov sampling needs {self.ctas} co-resident CTAs; "
                        f"the device holds {kernel.max_ctas}"
                    )
                self._kernels[base_dtype, name] = kernel

    def _allocate_peer_buffer(self, size: int) -> tuple[torch.Tensor, list[int]]:
        if self.group is None:
            buffer = torch.zeros(size, dtype=torch.uint8, device=self.device)
            return buffer, [buffer.data_ptr()]
        import torch.distributed._symmetric_memory as symm_mem

        buffer = symm_mem.empty(size, dtype=torch.uint8, device=self.device)
        handle = symm_mem.rendezvous(buffer, self.group.group_name)
        pointers = [
            handle.get_buffer(peer, (size,), torch.uint8, storage_offset=0).data_ptr()
            for peer in range(self.world)
        ]
        buffer.zero_()
        self._handle = handle
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        return buffer, pointers

    def supports(self, base_logits: torch.Tensor) -> bool:
        """``[rows, num_steps, vocab or vocab // world]`` base logits the chain takes; depends
        only on shapes, which tensor parallelism keeps identical across ranks."""
        return (
            base_logits.dim() == 3
            and 1 <= base_logits.shape[0] <= self.max_rows
            and base_logits.shape[1] == self.num_steps
            and base_logits.shape[2] in (self.vocab_size, self.vocab_local)
            and base_logits.dtype in LOGIT_DTYPES
            and base_logits.stride(2) == 1
        )

    def launch_noise(self, *, rows: int, key: int, offset: torch.Tensor) -> None:
        """The Gumbel noise of ``rows`` rows on the current stream; ``offset`` is SGLang's
        int64 draft-noise counter (read on the device)."""
        grid = (rows * self.num_steps, triton.cdiv(self.vocab_local, NOISE_BLOCK))
        _shard_noise_kernel[grid](
            self.noise,
            offset,
            key,
            self.vocab_offset,
            self.vocab_local,
            STEPS=self.num_steps,
            BLOCK=NOISE_BLOCK,
        )

    def fork_noise(self, *, rows: int, key: int, offset: torch.Tensor) -> None:
        """Draw the noise on a side stream; join_noise must precede sample()."""
        self._noise_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._noise_stream):
            self.launch_noise(rows=rows, key=key, offset=offset)

    def join_noise(self) -> None:
        torch.cuda.current_stream(self.device).wait_stream(self._noise_stream)

    def stage_anchor(self, anchor_tokens: torch.Tensor) -> None:
        """The anchors' copy ``sample(anchor_staged=True)`` reads; issued before the
        LM head so the chain's launch follows the LM head directly."""
        self.anchor[: anchor_tokens.shape[0]].copy_(anchor_tokens)

    def sample(
        self,
        *,
        base_logits: torch.Tensor,
        anchor_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        logits_cache: Optional[torch.Tensor],
        draft_tokens: torch.Tensor,
        anchor_staged: bool = False,
    ) -> None:
        """Draft tokens of every row into ``draft_tokens`` [rows, num_steps] and, when
        ``logits_cache`` [>= rows, num_steps, vocab] is given, the full-vocab step logits
        of every row at a nonzero temperature (temperature-0 rows are left untouched);
        without it every row is argmaxed. The noise must be drawn already."""
        rows = base_logits.shape[0]
        assert self.supports(base_logits)
        assert temperatures.dtype == torch.float32
        assert draft_tokens.dtype == torch.int64 and draft_tokens.stride(1) == 1
        wide = rows > self.narrow_rows
        if not anchor_staged:
            self.anchor[:rows].copy_(anchor_tokens)
        if logits_cache is None:
            idx_mapping, cache, stride0, stride1 = self.inert_rows, self.noise, 0, 0
        else:
            assert self.keeps_logits, "built without logit staging (keeps_logits=False)"
            assert logits_cache.dtype == self.cache_dtype
            assert logits_cache.size(2) >= self.vocab_size
            assert logits_cache.stride(2) == 1 and logits_cache.data_ptr() % 16 == 0
            idx_mapping, cache = self.sampled_rows, logits_cache
            stride0, stride1 = logits_cache.stride(0), logits_cache.stride(1)
        base_column = (
            self.vocab_offset if base_logits.shape[2] == self.vocab_size else 0
        )
        # Host-static per row count, so a CUDA graph of one batch size captures one kernel.
        if wide:
            name = "wide"
        elif rows <= DIRECT_MAX_ROWS:
            name = "narrow-direct"
        else:
            name = "narrow-fold"
        arguments = [
            _pointer(base_logits),
            ctypes.c_int64(base_logits.stride(0)),
            ctypes.c_int64(base_logits.stride(1)),
            ctypes.c_int64(base_column),
            _pointer(self.embed_weight),
            _pointer(self.head_shard),
            _pointer(self.anchor),
            ctypes.c_int32(1),
            _pointer(idx_mapping),
            _pointer(temperatures),
            _pointer(self.noise),
            _pointer(cache),
            ctypes.c_int64(stride0),
            ctypes.c_int64(stride1),
            _pointer(draft_tokens),
            ctypes.c_int64(draft_tokens.stride(0)),
            _pointer(self.peers),
            _pointer(self.wide_control if wide else self.control),
            ctypes.c_int32(self.rank),
            ctypes.c_int32(self.world),
            ctypes.c_int32(self.vocab_local),
            ctypes.c_int32(rows),
            ctypes.c_int32(self.wide_rows if wide else self.narrow_rows),
            ctypes.c_int64(self.timeout_ns),
            ctypes.c_int64(self.wide_offset if wide else self.mailbox_offset),
        ]
        stream = torch.cuda.current_stream(self.device).cuda_stream
        _launch_cooperative(
            self._kernels[base_logits.dtype, name],
            self.ctas,
            arguments,
            stream,
            pdl=self.pdl,
        )
