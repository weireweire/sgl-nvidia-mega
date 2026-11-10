# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang's python/sglang/kernels/ops/attention/dsv4/decode_attention_sm100_gluon.py (Apache-2.0).
"""Native 16-head Blackwell (tcgen05) MMA layouts for paged V4-layout attention. ``HQ`` < 16
real heads ride in the 16-column MMA: the missing query heads load as zero and their
columns are never stored, so each real head's arithmetic is the 16-head kernel's."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    fence_async_shared,
    mbarrier,
    tcgen05_commit,
    tcgen05_mma,
)

try:
    # Triton 3.7.1 API
    from triton.experimental.gluon.language.nvidia.blackwell import get_tmem_reg_layout
except ImportError:
    # Triton 3.8 moved this onto the descriptor type (triton-lang/triton#9594)
    from triton.experimental.gluon.language.nvidia.blackwell import (
        tensor_memory_descriptor_type,
    )
    from triton.runtime.jit import constexpr_function

    @constexpr_function
    def get_tmem_reg_layout(element_ty, shape, layout, num_warps):
        ty = tensor_memory_descriptor_type(element_ty, shape, layout, shape)
        return ty.get_reg_layout(num_warps=num_warps)


@gluon.jit
def _load_v4(
    CACHE,
    ids,
    valid,
    PAGE: gl.constexpr,
    STRIDE: gl.constexpr,
    KV_LAYOUT: gl.constexpr,
    DATA_BYTES: gl.constexpr,
    SCALE_BYTES: gl.constexpr,
    TILE: gl.constexpr,
):
    # V4 row: 448 fp8 nope + 64 bf16 rope = DATA_BYTES, plus one ue8m0 scale per
    # TILE values in the page's scale rows.
    d = gl.arange(0, 512, gl.SliceLayout(0, KV_LAYOUT))
    base = (ids // PAGE).to(gl.int64)[:, None] * STRIDE
    slot = (ids % PAGE)[:, None]
    mask = valid[:, None] & (d[None, :] < 448)
    bits = gl.load(CACHE + base + slot * DATA_BYTES + d[None, :], mask, 0)
    fp8 = bits.to(gl.float8e4nv, bitcast=True).to(gl.float32)
    exponent = gl.load(
        CACHE + base + PAGE * DATA_BYTES + slot * SCALE_BYTES + d[None, :] // TILE,
        mask,
        0,
    ).to(gl.int32)
    scale = gl.where(exponent == 0, 0x00400000, exponent << 23).to(
        gl.float32, bitcast=True
    )
    rope_ptr = (CACHE + base + slot * DATA_BYTES + 448 + (d[None, :] - 448) * 2).to(
        gl.pointer_type(gl.bfloat16)
    )
    rope = gl.load(rope_ptr, valid[:, None] & (d[None, :] >= 448), 0)
    return gl.where(d[None, :] < 448, fp8 * scale, rope.to(gl.float32)).to(gl.bfloat16)


# Fused combine (FUSED): the row's tile CTAs meet at a per-row counter. Each thread
# fences its partial stores, thread 0 arrives (release), every thread polls (acquire)
# until all NT tiles arrived, or traps past the deadline (ns). Thread 0 then arrives a
# second time; the row's last second arrival resets the counter for the next launch.
_ARRIVE_WAIT: gl.constexpr = gl.constexpr(
    "{\n"
    ".reg .pred p;\n"
    ".reg .u32 tid;\n"
    ".reg .u64 t0, t1;\n"
    "mov.u32 tid, %tid.x;\n"
    "setp.eq.u32 p, tid, 0;\n"
    "@p red.release.gpu.global.add.u32 [$1], 1;\n"
    "mov.u64 t0, %globaltimer;\n"
    "POLL:\n"
    "ld.acquire.gpu.global.u32 $0, [$1];\n"
    "setp.ge.u32 p, $0, $2;\n"
    "@p bra DONE;\n"
    "mov.u64 t1, %globaltimer;\n"
    "sub.u64 t1, t1, t0;\n"
    "setp.gt.u64 p, t1, $3;\n"
    "@p trap;\n"
    "bra POLL;\n"
    "DONE:\n"
    "}"
)
_DEPART: gl.constexpr = gl.constexpr(
    "{\n"
    ".reg .pred p;\n"
    ".reg .u32 tid, v;\n"
    "mov.u32 $0, 0;\n"
    "mov.u32 tid, %tid.x;\n"
    "setp.eq.u32 p, tid, 0;\n"
    "@p atom.relaxed.gpu.global.add.u32 v, [$1], 1;\n"
    "@p setp.eq.u32 p, v, $2;\n"
    "@p st.relaxed.gpu.global.u32 [$1], 0;\n"
    "}"
)


@gluon.jit
def _globaltimer():
    return gl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l", [], dtype=gl.int64, is_pure=False, pack=1
    )


@gluon.jit
def _stamp(STAMP, base, k: gl.constexpr):
    gl.store(STAMP + base + k, _globaltimer())


@gluon.jit
def _row_barrier(CNT, NT: gl.constexpr, DEADLINE):
    gl.inline_asm_elementwise(
        "fence.acq_rel.gpu; // $0", "=r", [], dtype=gl.int32, is_pure=False, pack=1
    )
    gl.barrier()
    address = CNT.to(gl.int64)
    gl.inline_asm_elementwise(
        _ARRIVE_WAIT,
        "=r,l,r,l",
        [address, gl.to_tensor(NT), DEADLINE.to(gl.int64)],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
    gl.barrier()
    gl.inline_asm_elementwise(
        _DEPART,
        "=r,l,r",
        [address, gl.to_tensor(2 * NT - 1)],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _combine_units(
    PART,
    MAX,
    SUM,
    SINK,
    OUT,
    b,
    t,
    NT: gl.constexpr,
    ST: gl.constexpr,
    HQ: gl.constexpr,
    ROUNDS: gl.constexpr,
):
    # swapab._combine's narrow program (one warp per head and 128 columns, the same
    # layouts along the tiles), the row's HQ * 4 programs spread over its tile CTAs
    units: gl.constexpr = HQ * 4
    stat: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    vals_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 4], [1, 1, 32], [4, 1, 1], [2, 1, 0]
    )
    wide: gl.constexpr = gl.SliceLayout(2, vals_layout)
    out_layout: gl.constexpr = gl.SliceLayout(1, vals_layout)
    wv = gl.arange(0, 4, gl.SliceLayout(1, stat))
    tv = gl.arange(0, ST, gl.SliceLayout(0, stat))
    w3 = gl.arange(0, 4, gl.SliceLayout(1, wide))
    t3 = gl.arange(0, ST, gl.SliceLayout(0, wide))
    d3 = gl.arange(0, 128, gl.SliceLayout(0, out_layout))
    wo = gl.arange(0, 4, gl.SliceLayout(1, out_layout))
    for r in gl.static_range(ROUNDS):
        u = (r * 4 + wv) * NT + t
        ok = u < units
        h = u // 4
        soff = (b * NT + tv[None, :]) * HQ + h[:, None]
        smask = (tv[None, :] < NT) & ok[:, None]
        den = gl.load(SUM + soff, smask, 0.0)
        mx = gl.load(MAX + soff, smask, 0.0)
        mx = gl.where(den > 0, mx, -float("inf"))
        sink = gl.load(SINK + h, ok, 0.0)
        m = gl.maximum(gl.max(mx, 1), sink)
        m = gl.where(gl.abs(m) == float("inf"), 0.0, m)
        factor = gl.exp(mx - m[:, None])
        denominator = gl.sum(den * factor, 1) + gl.exp(sink - m)
        keep = (denominator > 0) & (sink != float("inf"))
        u3 = (r * 4 + w3) * NT + t
        ok3 = u3 < units
        row = (b * NT + t3[None, :]) * HQ + (u3 // 4)[:, None]
        off = (
            row[:, :, None] * 512 + ((u3 % 4) * 128)[:, None, None] + d3[None, None, :]
        )
        vmask = ((t3[None, :] < NT) & ok3[:, None])[:, :, None]
        vals = gl.load(PART + off, vmask, 0.0)
        f3 = gl.convert_layout(factor, wide)
        acc = gl.sum(vals * f3[:, :, None], 1)
        dn = gl.convert_layout(denominator, gl.SliceLayout(1, out_layout))
        kp = gl.convert_layout(keep, gl.SliceLayout(1, out_layout))
        o = acc / dn[:, None]
        o = gl.where(kp[:, None], o, 0.0)
        uo = (r * 4 + wo) * NT + t
        ooff = (b * HQ + uo // 4) * 512 + (uo % 4) * 128
        gl.store(
            OUT + ooff[:, None] + d3[None, :],
            o.to(gl.bfloat16),
            (uo < units)[:, None],
        )


@gluon.jit
def partial_gluon(
    Q,
    K,
    E,
    IDX,
    EI,
    L,
    EL,
    PART,
    MAX,
    SUM,
    QS: gl.constexpr,
    QH: gl.constexpr,
    IS: gl.constexpr,
    EIS: gl.constexpr,
    KP: gl.constexpr,
    KS: gl.constexpr,
    EP: gl.constexpr,
    ES: gl.constexpr,
    NK: gl.constexpr,
    NE: gl.constexpr,
    NT: gl.constexpr,
    KT: gl.constexpr,
    BT: gl.constexpr,
    H: gl.constexpr,
    SCALE: gl.constexpr,
    KTOKENS: gl.constexpr,
    ETOKENS: gl.constexpr,
    COMPENSATE: gl.constexpr,
    SWAP_AB: gl.constexpr,
    DATA_BYTES: gl.constexpr,
    SCALE_BYTES: gl.constexpr,
    TILE: gl.constexpr,
    USE_GDC: gl.constexpr = False,
    FREQS=None,
    POS=None,
    ROPE_Q: gl.constexpr = False,
    KV_BEFORE_WAIT: gl.constexpr = False,
    HQ: gl.constexpr = 16,
    FUSED: gl.constexpr = False,
    CNT=None,
    SINK=None,
    OUT=None,
    DEADLINE=None,
    ST: gl.constexpr = 16,
    ROUNDS: gl.constexpr = 1,
    STAMPS: gl.constexpr = False,
    STAMP=None,
    STAGE_BEFORE_WAIT: gl.constexpr = 0,
):
    gl.static_assert(SWAP_AB and H == 16 and (BT == 64 or BT == 128))
    gl.static_assert(HQ == H or HQ == 8)
    gl.static_assert(not STAGE_BEFORE_WAIT or (KV_BEFORE_WAIT and USE_GDC))
    if FUSED:
        # the row's tiles are adjacent in launch order
        t, b = gl.program_id(0), gl.program_id(1)
    else:
        b, t = gl.program_id(0), gl.program_id(1)
    if STAMPS:
        sbase = (b * NT + t) * 16
        _stamp(STAMP, sbase, 0)
        smid = gl.inline_asm_elementwise(
            "mov.u32 $0, %smid;", "=r", [], dtype=gl.int32, is_pure=True, pack=1
        )
        gl.store(STAMP + sbase + 15, smid.to(gl.int64))
    if USE_GDC:
        if not KV_BEFORE_WAIT:
            # tl.extra.cuda.gdc_wait + gdc_launch_dependents; every input is read after the wait.
            gl.inline_asm_elementwise(
                "griddepcontrol.wait; // $0",
                "=r",
                [],
                dtype=gl.int32,
                is_pure=False,
                pack=1,
            )
        gl.inline_asm_elementwise(
            "griddepcontrol.launch_dependents; // $0",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
    if STAMPS:
        _stamp(STAMP, sbase, 1)
    kv_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    n = gl.arange(0, BT, gl.SliceLayout(1, kv_layout))
    if t < KT:
        at = t * BT + n
        length = gl.load(L + b)
        ids = gl.load(IDX + b * IS + at, at < NK, -1)
        valid = (at < NK) & (at < length) & (ids >= 0) & (ids < KTOKENS)
        kv = _load_v4(
            K,
            gl.maximum(ids, 0),
            valid,
            KP,
            KS,
            kv_layout,
            DATA_BYTES,
            SCALE_BYTES,
            TILE,
        )
    else:
        at = (t - KT) * BT + n
        length = gl.load(EL + b)
        ids = gl.load(EI + b * EIS + at, at < NE, -1)
        valid = (at < NE) & (at < length) & (ids >= 0) & (ids < ETOKENS)
        kv = _load_v4(
            E,
            gl.maximum(ids, 0),
            valid,
            EP,
            ES,
            kv_layout,
            DATA_BYTES,
            SCALE_BYTES,
            TILE,
        )
    qh = gl.arange(0, H, gl.SliceLayout(1, kv_layout))
    qd = gl.arange(0, 512, gl.SliceLayout(0, kv_layout))
    if STAGE_BEFORE_WAIT:
        # The query's RoPE row loaded (the positions are step inputs, the table a
        # constant) and, at 2, the KV tile staged in shared memory, while Q is in flight
        if STAGE_BEFORE_WAIT == 2:
            kv_smem = gl.allocate_shared_memory(
                gl.bfloat16,
                [BT, 512],
                gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16),
                value=kv,
            )
        if ROPE_Q:
            is_rope = qd >= 448
            pos = gl.load(POS + b)
            f_idx = pos * 64 + ((qd - 448) // 2) * 2
            cos = gl.load(FREQS + f_idx, is_rope, 0.0)[None, :]
            sin = gl.load(FREQS + f_idx + 1, is_rope, 0.0)[None, :]
    if USE_GDC and KV_BEFORE_WAIT:
        # Only Q comes from the PDL predecessor; the KV rows and indices were ordered
        # before this launch by the caller's stream joins.
        gl.inline_asm_elementwise(
            "griddepcontrol.wait; // $0",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
        if STAMPS:
            _stamp(STAMP, sbase, 2)
    q_ptr = Q + b * QS + qh[:, None] * QH + qd[None, :]
    if HQ < H:
        q = gl.load(q_ptr, (qh < HQ)[:, None], 0.0)
    else:
        q = gl.load(q_ptr)
    if ROPE_Q:
        # The model's interleaved (real, imag) RoPE on the last 64 dims, from the
        # BF16 projection; explicit fmas match the q RoPE kernel's SASS bit for bit:
        # real = fma(xr, cos, -(xi * sin)), imag = fma(xr, sin, xi * cos).
        is_rope = qd >= 448
        pair_ptr = Q + b * QS + qh[:, None] * QH + (qd ^ 1)[None, :]
        if HQ < H:
            pair = gl.load(pair_ptr, (qh < HQ)[:, None], 0.0).to(gl.float32)
        else:
            pair = gl.load(pair_ptr).to(gl.float32)
        x = q.to(gl.float32)
        if not STAGE_BEFORE_WAIT:
            pos = gl.load(POS + b)
            f_idx = pos * 64 + ((qd - 448) // 2) * 2
            cos = gl.load(FREQS + f_idx, is_rope, 0.0)[None, :]
            sin = gl.load(FREQS + f_idx + 1, is_rope, 0.0)[None, :]
        even = (qd % 2 == 0)[None, :]
        rotated = gl.where(
            even, gl.fma(x, cos, -(pair * sin)), gl.fma(pair, sin, x * cos)
        )
        q = gl.where(is_rope[None, :], rotated.to(gl.bfloat16), q)
    q_smem = gl.allocate_shared_memory(
        gl.bfloat16,
        [H, 512],
        gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16),
        value=q,
    )
    if STAGE_BEFORE_WAIT != 2:
        kv_smem = gl.allocate_shared_memory(
            gl.bfloat16,
            [BT, 512],
            gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16),
            value=kv,
        )
    if STAMPS:
        _stamp(STAMP, sbase, 3)
    score_tmem = allocate_tensor_memory(
        gl.float32, [BT, H], TensorMemoryLayout(block=(BT, H), col_stride=1)
    )
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    fence_async_shared()
    tcgen05_mma(kv_smem, q_smem.permute((1, 0)), score_tmem, use_acc=False)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    score_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32, (BT, H), TensorMemoryLayout(block=(BT, H), col_stride=1), 4
    )
    if STAMPS:
        _stamp(STAMP, sbase, 4)
    scores = score_tmem.load(score_layout) * SCALE
    valid = gl.convert_layout(valid, gl.SliceLayout(1, score_layout))
    scores = gl.where(valid[:, None], scores, -float("inf"))
    mx = gl.max(scores, 0)
    mx = gl.where(mx == -float("inf"), 0.0, mx)
    prob = gl.exp(scores - mx[None, :])
    denom = gl.sum(prob, 0)
    p_hi = prob.to(gl.bfloat16)
    p_smem = gl.allocate_shared_memory(
        gl.bfloat16,
        [BT, H],
        gl.NVMMASharedLayout(swizzle_byte_width=32, element_bitwidth=16),
        value=p_hi,
    )
    out_tmem = allocate_tensor_memory(
        gl.float32, [512, H], TensorMemoryLayout(block=(128, H), col_stride=1)
    )
    fence_async_shared()
    tcgen05_mma(kv_smem.permute((1, 0)), p_smem, out_tmem, use_acc=False)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=1)
    if COMPENSATE:
        p_lo = (prob - p_hi.to(gl.float32)).to(gl.bfloat16)
        p_smem.store(p_lo)
        fence_async_shared()
        tcgen05_mma(kv_smem.permute((1, 0)), p_smem, out_tmem, use_acc=True)
        tcgen05_commit(bar)
        mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    if STAMPS:
        _stamp(STAMP, sbase, 5)
    out_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32, (512, H), TensorMemoryLayout(block=(128, H), col_stride=1), 4
    )
    value = out_tmem.load(out_layout)
    hd_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    value_hd = gl.convert_layout(value.permute((1, 0)), hd_layout)
    h = gl.arange(0, H, gl.SliceLayout(1, hd_layout))
    d = gl.arange(0, 512, gl.SliceLayout(0, hd_layout))
    stat_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    hs = gl.arange(0, H, stat_layout)
    part_ptr = PART + ((b * NT + t) * HQ + h[:, None]) * 512 + d[None, :]
    if HQ < H:
        gl.store(part_ptr, value_hd, (h < HQ)[:, None])
        gl.store(
            MAX + (b * NT + t) * HQ + hs, gl.convert_layout(mx, stat_layout), hs < HQ
        )
        gl.store(
            SUM + (b * NT + t) * HQ + hs, gl.convert_layout(denom, stat_layout), hs < HQ
        )
    else:
        gl.store(part_ptr, value_hd)
        gl.store(MAX + (b * NT + t) * H + hs, gl.convert_layout(mx, stat_layout))
        gl.store(SUM + (b * NT + t) * H + hs, gl.convert_layout(denom, stat_layout))
    if STAMPS:
        _stamp(STAMP, sbase, 6)
    if FUSED:
        _row_barrier(CNT + b, NT, DEADLINE)
        if STAMPS:
            _stamp(STAMP, sbase, 7)
        _combine_units(PART, MAX, SUM, SINK, OUT, b, t, NT, ST, HQ, ROUNDS)
        if STAMPS:
            _stamp(STAMP, sbase, 8)
