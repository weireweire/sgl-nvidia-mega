# SGLang-NVIDIA-Mega

## Introduction

**SGLang-NVIDIA-Mega** holds the NVIDIA megakernels of the [SGLang](https://github.com/sgl-project/sglang) framework: for each model it covers, the code of its small-batch decode step and the kernels that only make sense inside such a step, which hand their outputs to each other through mailboxes and programmatic dependent launch instead of through per-op boundaries.

The package loads as an SGLang [general plugin](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/plugin.mdx) (the `sglang.srt.plugins` entry point) and is turned on with SGLang's `--enable-nvidia-megakernel` server flag. The package registers each model's spec with SGLang's megakernels framework (`sglang.kernels.plugin_kernels.nvidia`, `sglang.srt.megakernels` before sgl-project/sglang#45389). With the flag, SGLang checks the setup at argument resolution, before the weights load. After the model runner has prepared a covered model (before the KV cache is sized), SGLang builds the megakernel and replaces that model instance's `forward`, which runs each covered step itself and leaves every other step to the model's own forward. Every rank then logs `NVIDIA megakernel: active (…)`. Without the flag, SGLang runs untouched.

> **Experimental:** the kernels, the models and setups they cover, and the switch may change.

## Features

### Supported Models

| Model | GPU | Tensor parallel | Coverage |
| --- | --- | --- | --- |
| DeepSeek-V4.1-Flash | GB300 (SM103); 148-SM B300 (SM103) with `sm148` | 4 | DSpark target verify steps of 1 to 384 rows (batch 1 to 64 at 6 rows a request); on B300, speed was measured at one request only, serving passed a GSM8K canary at client concurrency 12 and 16 with decode CUDA graphs up to batch 16 (DSpark block 5), and larger batches and other block sizes rely on the kernel tests |
| Qwen3.5-397B-A17B (NVFP4) | GB300 (SM103) | 8 (two trays, one NVLink domain) | MTP target verify steps of 7 rows (batch 1) |
| Kimi K3 (packed MXFP4) | GB300 (SM103, 152 SMs) | 8 | Decode rows 1–8, 16, 32 and 64; fixed q4–q8 verification within 64 token rows. Admission and tested configurations differ; see [`spec.py`](sgl_nvidia_mega/kimi_k3/spec.py) |

For DeepSeek-V4.1-Flash, larger verify steps, prefill and extend run
SGLang's standard path. Qwen3.5 also offers the explicit GDN prefill slot below. Their draft specs provide the DSpark proposal and the
Qwen3.5 EAGLE entries, respectively. For Kimi K3, the plugin covers the 93-layer
language model (24 MLA and 69 KDA layers) at the listed widths; every other call
runs the standard path.

The Kimi K3 adapter is experimental. Initial ordinary-decode validation used
context length 8192 on GB300/TP8: every listed CUDA-graph width and padding
boundary passed dummy-weight
capture/replay, B1 and B16 passed stock-versus-plugin real-weight canaries, and a
specialized-route/captured-replay canary passed, and the GSM8K quality protocol
matched its frozen stock-path reference score. This is bounded functional and
quality evidence; it does not claim performance, unlisted configurations or production
readiness.

Earlier matched real-weight functional checks covered ordinary decode and
five-draft/q6 DSpark at context length 262144, with DCP off and stock draft
execution. Inputs span 32768, 131072, 240000 and 255872 tokens, plus a
262000-input/128-output deployment-edge request. Long B1 and mixed-B2 requests,
short B4 controls, chunked prefill, page boundaries, retirement, request-slot
reuse and full decode graph replay passed the predefined numerical checks.
Long-input output tokens matched stock; short controls can differ within the
numerical bounds. This is not global bitwise equivalence, an AgentX benchmark,
long-context task-quality validation or a performance claim. Other fixed draft
lengths have component coverage, not equivalent long-context serving evidence.

These results belong to the revisions tested, not automatically to every later
composition. Combined default-OFF ordinary-decode and fixed-q6 checks passed
within the recorded scopes, including near-cap requests, mixed lengths, reuse
and graph replay. A separate registered-default width-64 control matched the
published and private implementations exactly for all 64 sequences and 4,096
selected-token log probabilities. That control does not establish native-stock
accuracy or resolve the five numerical failures in the broader configured-width
experiment. Those failures, including at the already-admitted width 64, are not
waived by component tests or the default-OFF additional-width API. These are
bounded results, not a blanket numerical or performance PASS.

Optional prefetch, embedding and selected-width preparation have bounded
correctness evidence and remain dormant. The TokenSpeed MLA alternative has
matched ordinary-decode and fixed-q6 evidence at TP8, DCP off and context length
8192; TRTLLM remains the default. DCP resources and execution interfaces are
included for explicit integration, with matched ordinary-decode and fixed-q6
evidence at TP8 and context length 8192, including native acceptance, state
commit, hidden-state handoff and graph replay. The DCP adapters require additional
framework interfaces; stock-framework imports and explicit unsupported-activation
rejection are separate CPU contracts, not GPU qualification. No DCP feature
admission is added to the registered spec, and no DCP long-context, optional
combined-ON, default activation or performance claim follows from these checks.

Enabling the plugin does not itself enable DSpark: ordinary decode remains
ordinary decode; five-draft/q6 is the qualified speculative configuration when
DSpark is explicitly configured. An admitted shape is not a claim that every
batch, context and optional-provider combination has passed full-model testing.

### What a covered DeepSeek-V4.1-Flash step runs

The step is the plugin's own code over the weights SGLang loaded (`deepseek_v41/runtime`): the embedding and the Engram hashes, then per layer the mHC boundary (`boundary/`), the attention (`attention/`), the MoE (`moe/`) and Engram (`engram/`), then the head and the logits (`logits/`). Each package has a stock implementation that reproduces SGLang's decode step bit for bit, and puts this package's kernels in where the slot plan (`plan.py`) gives them a width:

| Slot | Rows | Kernel |
| --- | --- | --- |
| `wo_b` + attention all-reduce + mHC post + next norm (`hop_a`, `hop_a_batch2`) | 1–16 | `wo_b` GEMM whose epilogue stores its rows into a Lamport mailbox on every rank, and a Lamport consumer that reduces them and runs the mHC post, combine and RMSNorm |
| MoE all-reduce + mHC post + next norm (`hop_b`, `hop_b_wide`) | 1–48 | Hop B: the top-k finalize and the shared-expert add on a Lamport mailbox, whose consumer runs the mHC post, combine and RMSNorm; past 8 rows it writes the next input's MXFP8 scales per 8-row tile, so the next layer's `wqkv_a` runs on the low-latency GEMM. Hop A's widths use hop A's mailbox, the wider ones a second one |
| Routed experts (`moe`) | 1–48 | A route kernel and a pair of block-scaled tcgen05 GEMMs with 8-row tiles; an expert chosen by more than 8 rows takes several tiles |
| Router top-k (`gate`, `gate_wide`) | 1–48 | One CTA per row, bit for bit SGLang's fused gate (including the vision-token routing) |
| Shared expert gate_up (`shared_ll`) | 1–48 | FlashInfer's low-latency MXFP8 GEMM with 8-row tiles on the alternate stream from the layer's fork, after a one-pass MXFP8 quant into its row-tile layout (bit for bit the shared linear's own quant); SGLang's clamped SwiGLU and `down_proj` follow |
| Replicated embedding (`embed`) | 1–48 | SGLang's fused lookup on the whole table, all-gathered once at build; no all-reduce |
| Indexer top-k (`attn_indexer`) | 30–48 (batch 5–8) | The decode indexer's top-k in the plugin's forward, reading SGLang's paged indexer metadata and writing its selection into SGLang's buffer; the index query and head weights run on the layer's second stream, under the compressor. At 6–24 rows, chunked metadata or without paged metadata SGLang's indexer runs |
| Engram prefetch (`engram`) | 1–8 (batch 1) | Every Engram layer's lookup and WKV quarter on a side stream; their two all-reduces on the main stream (SGLang's TP group) |
| Attention and MoE all-reduce + mHC post + next norm (`two_shot`) | 49–384 | Two-shot exchange: each rank sends row t only to its owner rank (t mod 4), which sums the four partials in rank order and sends the row back; the MoE post also folds the top-k finalize and the shared-expert add, and quantizes the next input to MXFP8 where SGLang's fused post does |

With the megakernel on, the default 39 slots run: the ones above, the attention chain, hop B's triggers, and the DSpark draft step's 16 slots. `SGLANG_NVIDIA_MEGAKERNEL_SLOTS` (comma-separated) overrides that list for tests and bisecting; empty runs every step function's stock code. The opt-in slots run only when listed there: `target_lm_head` (the target's LM head on the low-M kernel at 1-8 rows), `lm_head_pinned_tactic` (both low-M LM heads on one fixed tactic per row count, untuned), `markov_pdl` (the draft's Markov chain launched as a programmatic dependent of the draft LM head, loading its head tiles before the wait; bitwise), `hop_b_quad_trigger` (hop B's finalize-publish launches its consumer right after its own wait, batch 1 only), `hop_b_cap16` and `draft_embed`; listing `sm148` there admits the 148 SMs of a B300 instead of the 152 required otherwise.

Qwen3.5 (`qwen3_5/runtime`) runs the same way: per layer the stage boundary (`boundary/`), the GatedDeltaNet or the attention (`gdn/`, `attention/`) and the MoE (`moe/`), each with a stock implementation that reproduces SGLang's verify step bit for bit. It reads the running forward's metadata and pools through SGLang's EAGLE binding (`sglang.srt.kernel_plugins.nvidia.eagle_binding`). With the megakernel enabled, its shape-bounded bitwise paths run by default. `SGLANG_NVIDIA_MEGAKERNEL_SLOTS` overrides that policy for testing and bisecting; an empty value runs the stock paths.

| Slot | Rows | Kernel |
| --- | --- | --- |
| `route_topk`, `mkm_m32` … `mkm_m512` | 1–896 | Materialized stock routing and the shape-selected fused wide routed-expert engines |
| `hop`, `hop_fp8` | 1–15 | The layer boundary's fused all-reduce + residual add + RMSNorm, also writing the next static-FP8 input |
| `multi_request` | 2–128 requests | The target graphs keep every offered request count instead of padding to a coarser width |
| `draft_loop`, `draft_extend`, `draft_multi`, `draft_embed` | 1–128 requests | EAGLE draft entries, exact sharded top-1 and embedding exchange; the pre-FC norms write directly into the FC input |

The remaining kernel families are opt-in through that one slot policy. These change the arithmetic within the tolerances `spec.py` declares:

| Slot | Rows | Kernel |
| --- | --- | --- |
| `route`, `route_gemm`, `mkm`, `mkm_wide` | 1–32 | Fused router fronts and the small routed-expert engines |
| `gdn_verify`, `gdn_pdl`, `gdn_conv` | 7–896 | Explicit numeric verify chain and dependent variants; FP8 norm uses the v2 relative-L2 bound of 2e-2 |
| `gdn_core` | 7–224 | ReplaySSM verify, gated norm and FP8 output in one launch |
| `hop_mid` (with `hop`) | 16–28 | Extends the one-shot boundary mailbox through four requests instead of returning those widths to the stock boundary |
| `gdn_push` (with `gdn_verify`, `hop`) | 7 | The GatedDeltaNet `out_proj` as the hop's GEMM, whose epilogue pushes the partial into the hop's mailbox |
| `gdn_front`, `gdn_trt_front` | 7–28 | Fused GatedDeltaNet projections and conv before the dependent verify |
| `route_shared` (with `route`) | 1–8 | The shared expert's static-FP8 gate/up in the router front's launch, so the shared expert on the alternate stream waits for that launch only |
| `route_lin`, `skv2`, `mka`, `mka_push` | shape-bounded | Router/attention overlap, split-KV attention and fused attention boundaries |
| `lowm_in`, `lowm_out`, `lowm_se`, `lowm_push` | shape-bounded | Validated low-row projection and shared-expert launches |
| `draft_moe`, `draft_q1`, `draft_xchg`, `draft_hop`, `draft_fin`, `draft_front`, `draft_skv2` | shape-bounded | Fused MTP draft kernels and boundaries |

The tuned Qwen3.5 TP8/TEP8 serving policy explicitly includes `hop_mid` whenever it
includes `hop`.  It remains opt-in rather than part of the bitwise default policy;
an explicit `SGLANG_NVIDIA_MEGAKERNEL_SLOTS` value replaces, rather than extends,
that default list.

The TP8 plan (`topology.plan`, aggregated and PD decode) also runs `route_lin`
whenever the policy holds the slots it runs through (`route`, `mkm`, `hop`,
`gdn_push`, `mka`, `mka_push`, and `lowm_se` with `route_shared`): its router logits
stay within 1.05x stock's RMS error against FP64 on the same quantized operands.
The same TP8 plan adds `mkm_wide_bal` for `mkm_m256` / `mkm_m512`: balanced
CTA assignment at 129–512 verify rows, with bitwise-identical outputs.
TEP8, DEP8, TP2 and TP4 run exactly the requested slots their topology supports.

TEP8's routed body takes its NVFP4 MoE tactic per token bucket from a fixed table
(`moe/tactics.py`) on FlashInfer 0.7.1, SM103 and the routed experts' MoERunner
signature; any other configuration or bucket asks the autotuner, and the log names
each width's source. `SGLANG_NVIDIA_MEGAKERNEL_MOE_TACTICS` switches the table; unset,
it follows `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION`.

The router-only `route`, `route_topk` and `route_gemm` slots support TP2, TP4, TP8 and TEP8 with the TRTLLM runner and no all-to-all. PD decode TP2 uses the same local row counts: batch 6/10 gives 42/70 verify rows. These fronts leave the expert GEMMs with the stock runner, so TP2 needs no fused expert engine. `route_shared` keeps its TP8/TEP8 projection bounds; DEP8 keeps stock MoE dispatch and routing.

`mka` uses the fused projection kernels through 14 verify rows on TP8 and 8 on
TP4. At wider rows, and at every TP2 width, it keeps SGLang's projection GEMMs
and fuses Q/K normalization, RoPE and the FP8 KV write, then the output gate
and static FP8 quantization. TP2 has 16 query heads and one KV head per rank;
TP4 has 8/1 and TP8 has 4/1. The same front path serves PD decode TP2 and DEP8 (32/2 per rank).
`multi_request` coverage is required for more than one request. `skv2` keeps
its one-request, one-KV-head scope; `skv2_cp` additionally requires replicated
KV heads across ranks.

The sampling and acceptance after the verify stay SGLang's own code. `gdn_prefill_split_conv` is an explicit bitwise slot for ordinary text prefill on TP1/PP4 with the data-only prefill binding: it splits the native conv by Q/K/V channels above 8192 channels, retaining native recurrence, norm and state writes. Unsupported layouts use SGLang; an empty slot policy calls the original prefill forward directly. In an SGLang that has the switches, `SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH=1` runs the sampling and acceptance in CUDA graphs, and `SGLANG_ENABLE_GDN_PREFILL_GRAPH_EXTEND=1` captures the GatedDeltaNet extend inside the breakable prefill CUDA graphs.

### Design

- **Built at load.** Buffers, kernels and tactics for every covered width are allocated and compiled when the model is loaded, before the KV cache is sized. Nothing is compiled or allocated inside a step.
- **Graph-safe coverage.** A step is covered by its forward mode and row count alone, so every rank decides the same and a captured graph replays the decision made at capture.
- **Fail-stop synchronization.** DeepSeek-V4.1-Flash waits on another GPU have
  a deadline below SGLang's watchdog. A rank that misses it faults at address
  0x0, so no token from a failed step is returned. Kimi K3 validates SGLang's
  finite build deadline and compiles it into all six reachable TP8 remote-wait
  paths; expiry likewise stores through address 0x0 and traps.
- **No silent fallback.** With `--enable-nvidia-megakernel`, an unsupported setup stops the server at startup, before the weights load, listing every reason; a model the loaded weights do not fit stops every rank after the load.

## Quick Start

### Requirements

- An SGLang with the megakernels framework (`sglang.kernels.plugin_kernels.nvidia`, or `sglang.srt.megakernels` before sgl-project/sglang#45389; `--enable-nvidia-megakernel`); Qwen3.5 also needs `sglang.srt.kernel_plugins.nvidia.eagle_binding`
- SGLang's CUDA 13 image: FlashInfer (its low-latency block-scaled GEMM as in 0.7.0 to 0.7.2), CuTe DSL (`nvidia-cutlass-dsl`) and the CUDA Python bindings
- DeepSeek-V4.1-Flash: 4 x GB300, or 4 x 148-SM B300 (SM103) with `sm148` in `SGLANG_NVIDIA_MEGAKERNEL_SLOTS`
- Qwen3.5: 8 x GB300, two trays of one NVLink domain
- Kimi K3: 8 x GB300 (SM103, 152 SMs), BF16, context length up to 262144 within the
  native model limit, TP8, DP1 and DCP off,
  FP8 E4M3 KV cache with page size 64, `trtllm_mla` for prefill and decode,
  `flashinfer_mxfp4` with default precision, and full decode CUDA graphs
- Kimi K3 DSpark requires the target-verification interfaces in
  [SGLang #45406](https://github.com/sgl-project/sglang/pull/45406). The tested
  configuration uses `RadixArk/Kimi-K3-DSpark` with five draft proposals (q6);
  the draft model and acceptance remain on SGLang's stock path.
- Kimi K3 checks for CuTe DSL 4.8.0, a coherent FlashInfer Python/cubin/JIT-cache
  wheel trio and the private ABI used by its prepared kernels. The general
  dependency list above is not a promise of compatibility with arbitrary newer
  versions; see `sgl_nvidia_mega/kimi_k3/compat.py`.

### Installation

```bash
git clone https://github.com/sgl-project/sgl-nvidia-mega.git
cd sgl-nvidia-mega
pip install -e .
```

### Usage

DeepSeek-V4.1-Flash:

```bash
SGLANG_SPEC_TP_SYNC=rng,-dspark-accept-sample \
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank \
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4.1-Flash \
  --tp 4 \
  --ep-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --dsv4-attn-backend flashmla \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --enable-nvidia-megakernel
```

The three environment variables are required by the DeepSeek-V4.1 spec; without them the server stops at startup, naming each.

Qwen3.5, on each of the two trays (`--node-rank 0` and `1`):

```bash
python -m sglang.launch_server \
  --model-path nvidia/Qwen3.5-397B-A17B-NVFP4-V2 \
  --tp 8 --nnodes 2 --node-rank 0 --dist-init-addr <tray-0 host>:<port> \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-algorithm NEXTN --speculative-num-steps 6 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 7 \
  --attention-backend trtllm_mha \
  --linear-attn-decode-backend triton \
  --enable-linear-replayssm-spec --mamba-ssm-dtype bfloat16 \
  --moe-runner-backend flashinfer_trtllm \
  --flashinfer-allreduce-fusion-backend cutedsl \
  --enable-nvidia-megakernel
```

### Kimi K3 optional startup providers

The hybrid runtime and DSpark interfaces are upstream in SGLang through
[#45406](https://github.com/sgl-project/sglang/pull/45406), merge
`aa3e34aaebf880d72482cea74dbbfef54da728e2`. API v2 is required. The optional
interfaces below do not expand the registered hardware, model or width guards.

Native SGLang embedding and no cross-layer weight prefetch remain the defaults.
For experiments, set either of these plugin-owned switches before launching
the server with `--enable-nvidia-megakernel`:

| Environment variable | Unset or `0` | `1` |
| --- | --- | --- |
| `SGLANG_KIMI_K3_MEGAKERNEL_PREFETCH` | No optional prefetch preparation/resources | Prepare cross-layer weight prefetch using the existing selective TP8 token-row policy |
| `SGLANG_KIMI_K3_MEGAKERNEL_EMBEDDING` | Native embedding; no optional gather preparation/resources | Prepare the specialized gather for supported native embedding layouts; retain native fallback otherwise |

Use the same settings on every tensor-parallel rank. Only `0` and `1` are
accepted; other values fail during build, before optional resource preparation.
The plugin reads these switches at build time, not in the decode/verification
loop. Restart the server to change them; they are not live graph toggles.
Supplied `inputs_embeds`, vocabulary sharding and native communication semantics
remain unchanged. Neither option enables DCP, alternate MLA or wider batch
admission.

Each option has bounded ordinary-decode and q6 serving correctness evidence.
The prefetch serving screen and embedding component measurements did not
establish a useful benefit in their tested scopes; neither justifies default
activation. Both remain OFF. Their separate proofs do not qualify both options
together or establish arbitrary workload, long-context ON or performance coverage.

### Kimi K3 optional capture-width build API

`sgl_nvidia_mega.kimi_k3.spec.build_with_capture_widths` is an explicit
adapter-facing alternative to `SPEC.build`. The registered build remains
unchanged; no CLI or environment setting automatically enables this policy.

Call the alternative on every TP rank, passing the usual model, HF config and
`BuildContext`, plus keyword `capture_token_rows`. `None` preserves the existing
width policy. A tuple selects baseline token rows `1–8,16,32,64` plus the supplied
capture rows. Sizes must be positive integers; duplicates are deduplicated,
and sizes above capacity are excluded. An empty tuple selects the baseline.
Use actual configured capture rows, not a dense capacity range inferred from
`BuildContext.widths`. For q6, request batches 1/2/4 correspond to token rows
6/12/24. Verification rounds up to whole requests, excludes rounded rows above
64 and excludes ordinary-only baseline rows incompatible with its query width.
This does not add ordinary decode to a verification-only provider.

The API checks cross-rank agreement before preparation and snapshots the selected
widths into a new build context. All ranks must call the same build API; this is
not a live toggle. It does not implicitly enable another provider. The prefetch
and embedding environment settings above are still read independently; alternate
MLA and DCP remain unselected.
Normal SGLang startup continues to use `SPEC.build`; an adapter must explicitly
choose the alternative when activating it later.

CPU selection/forwarding tests do not establish GPU correctness or performance.
Wider full-model numerical checks retain unresolved failures; this API is
experimental and default-OFF, not qualification or a fix for those failures.

### Kimi K3 dormant attention and DCP interfaces

These lower-level interfaces retain tested implementations for explicit future
integration. They are not server switches and do not change `SPEC.build`.

| Capability | Plugin interface | Registered default / integration boundary |
| --- | --- | --- |
| Alternate MLA | `providers.build_resources(..., mla_core="tokenspeed")` | `mla_core="trtllm"`; an adapter must select and prepare the alternative before graph capture |
| DCP8 resources | `providers.build_resources(..., mla_core="tokenspeed", dcp=dcp_resources.DcpBuildOptions())` | `dcp=None`; requires the matching native TP/DCP groups before preparation |
| DCP ordinary decode | `dcp_execution.PreparedDcpHybrid` | Requires the separate SGLang DCP binding extension and its owned pools/buckets |
| DCP fixed-q6 verification | `dcp_verify_execution.PreparedDcpVerify` | Requires the separate SGLang DCP verification binding extension, including native acceptance and state commit |

Stock SGLang at the merge above does not contain those DCP extensions. Importing
the plugin does not require them; attempting the unsupported DCP handoff rejects
explicitly rather than substituting the ordinary pool layout. Callers own native
metadata preparation, graph-safe updates and resource lifetimes. The plugin's
readiness/ownership checks must not be bypassed. These are startup integration
choices, not live backend or graph toggles.

TokenSpeed selection requires `tokenspeed-mla==2.0.2`; DCP requires this selector
explicitly, not the default TRTLLM core. Supplying `dcp` alone is rejected.

The bounded attention/DCP results described above do not qualify arbitrary
combinations with prefetch, embedding, additional widths or long-context DCP.
Their dormant inclusion does not enable them or imply a performance benefit.

## Repository Structure

```text
sgl-nvidia-mega/
├── sgl_nvidia_mega/
│   ├── plugin.py          # registers the specs with SGLang's megakernels framework
│   ├── common/            # NVRTC builds and PDL launches; CuTe DSL sync primitives
│   ├── deepseek_v41/
│   │   ├── spec.py          # the spec: the setups it serves and its build
│   │   ├── plan.py          # which kernel fills each slot, per width
│   │   ├── runtime/         # the build, the executor and its per-step state, the embedding
│   │   ├── boundary/        # the mHC between sublayers
│   │   ├── attention/       # the attention step
│   │   ├── moe/             # the MoE step
│   │   ├── engram/          # the Engram layers
│   │   ├── logits/          # the head and the logits
│   │   └── sm103/           # kernels and the build-time runner; no SGLang imports
│   ├── qwen3_5/              # spec, plan, runtime, boundary, GDN, attention, MoE and SM103 kernels
│   └── kimi_k3/              # the Kimi K3 spec, adapter, executor and CuTe DSL kernels
├── benchmark/             # the static decode table behind the speed numbers
└── test/
    ├── unit/              # plugin contracts and setup checks; selected CUDA metadata/graph cases
    ├── kernels/           # GPU: each kernel against SGLang's or FlashInfer's op, fail-stop
    └── e2e/               # GPU: serving accuracy, KL, passkey and fail-stop, switch on and off
```

## Testing

```bash
pytest test/unit               # host tests plus CUDA cases when a compatible GPU is available
pytest test/kernels            # DeepSeek-V4.1-Flash: 4 x GB300
pytest test/e2e                # DeepSeek-V4.1-Flash: 4 x GB300, with SGLANG_NVIDIA_MEGAKERNEL_E2E_MODEL_PATH set
```

Qwen3.5 runs TP8, which spans two GB300 trays: its e2e tests read a server launched on them through `SGLANG_NVIDIA_MEGAKERNEL_E2E_URL` and the files its ranks log to (`SGLANG_NVIDIA_MEGAKERNEL_E2E_LOGS`, comma-separated).

Kimi K3 full-serving qualification is separate from these unit tests; the
`test/kernels` and `test/e2e` commands above remain DeepSeek-V4.1-Flash suites.

## Contributing

Open a pull request against `main`. The PR title follows `[Type] scope: title` (see `.github/pull_request_template.md`), and `pre-commit run --all-files` must pass.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
