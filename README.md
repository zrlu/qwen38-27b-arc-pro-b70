# Qwen3.8-27B x Intel Arc Pro B70 - one-click Docker, pi-agent setup, benchmarks

Abliterated **Qwen3.8-27B** -> **GPTQ-INT4 (sym G128, MTP-BF16)**, tuned and
published for a **single Intel Arc Pro B70** (Xe2, 32 GB class). Reference
stack: vLLM XPU `0.28.0`, kernels `0.1.12.3`, MTP3 speculative decoding
(BF16 draft, `DRAFT_INT4=0`), prefix caching, `qwen3_xml` tool-call parser.

Images:

| Tag | What |
|---|---|
| `zrlu/qwen38-27b-arc-pro-b70:latest` | same image as `0.28.0-apcfix` (retagged so the default pull gets the fixed build) |
| `zrlu/qwen38-27b-arc-pro-b70:0.28.0-apcfix` | **current**: the 0.28.0 snapshot + the hybrid MTP/prefix-cache correctness fixes |
| `zrlu/qwen38-27b-arc-pro-b70:snapshot-0.28.0-apc-broken` | the original image, kept unchanged as the rollback point (local) |

| Artifact | Link |
|---|---|
| Model (HF, huihui) | [zrlu/Huihui-Qwen3.8-27B-abliterated-GPTQ-Int4-sym-G128-MTP-BF16-B70](https://huggingface.co/zrlu/Huihui-Qwen3.8-27B-abliterated-GPTQ-Int4-sym-G128-MTP-BF16-B70) |
| Upstream reference | [SergiioB/intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook) |

## Session report — 2026-09-12

**Reported symptoms.** (1) After a while the model emitted runs of `!`, worst at
long context; restarting cleared it, but continuing the same conversation
reproduced it immediately, and even loading a skill could trigger it.
(2) When that did not happen, throughput decayed over a session — "the cache
fills up", a few tok/s, a fresh session was slow too, only a container restart
recovered it. (3) The long-context decode rate was low.

**Root cause (1)+(2).** Not the KV cache, not VRAM, not the `tile_mask` story
this README used to tell. `Qwen3_5ForConditionalGeneration` is hybrid, and
`--enable-prefix-caching` forces `mamba_cache_mode="align"` on it; combined with
MTP speculative decode on the V1 model runner, two open upstream bugs corrupt
the recurrent state and the accepted-token counter, and the bad state is written
**back into the SSM/prefix cache**, so it is persistent and never self-corrects.
Details and evidence: [Correctness](#correctness-hybrid-mtp--prefix-caching-read-this-first)
and `docker/opt-qwen38/README-corrections.md`.

**Fix.** Two vendored upstream ports applied (fail-closed) at container boot:
`patch_fix_backward_copy.py` (vllm#53505) and `patch_fix_accepted_sync.py`
(vllm#53919). A third, `patch_fix_eagle_drop.py` (vllm#48375), is vendored but
disabled: on 0.28.0 it introduces NaN (see
[Correctness](#correctness-hybrid-mtp--prefix-caching-read-this-first)).

**Measured.**

| | before | after |
|---|---:|---:|
| decayed engine, 25 k ctx | 13 tok/s, output `!` | — |
| fresh engine, 8 k / 32 k / 100 k | — | 35 / 49 / 32 tok/s |
| 100-120 k warm-turn TTFT (91-96 % cache hits) | — | 10-13 s |
| pi working point, 117 393 prompt + 32 768 out | — | accepted, 1372 tokens generated |

The decayed→fresh delta (**2.7x**) *is* the fix: the model no longer degrades, so
no restart is needed. Raw decode speed was not the problem.

**Verification (all on the shipped image).**

- `python benchmarks/soak_hybrid_mtp.py 120000 40 3000` →
  `SOAK CLEAN: reached 121358 tokens, 40 turns, code=ZQX-3395, fails=0`
- the 94 396-token conversation that reproducibly went NaN before the fix is
  clean, cold and warm, over repeated replays
- a 25-turn / 100 906-token soak also clean; `qwen3_xml` tool calls verified
- boot log shows all three patch lines with the expected verdicts

**Context decision.** The client window was lowered to 150 000 while the server
ceiling stayed at 200 000, because the KV pool is a shared budget between the
live session and the prefix cache. Rationale and tables:
[Context sizing](#context-sizing-server-200k-client-150k).

**Upgrade attempt.** vLLM 0.29.0 was built and tested and **does not run this
model on WSL2** (oneDNN W4A16 needs an OpenCL compiler the WSL driver does not
provide). Full evidence: [vLLM 0.29.0 is blocked on WSL2](#vllm-0290-is-blocked-on-wsl2-gptq-int4).

**Headroom.** Prefill is at the platform ceiling; decode still has ~1.7-2.3x,
mostly in DRAM traffic rather than in "waiting for a fix". Analysis:
[Headroom](#headroom-is-the-hardware-maxed-out).

**Artifacts.** `zrlu/qwen38-27b-arc-pro-b70:0.28.0-apcfix` (= `:latest`) is the
shipped image; `:snapshot-0.28.0-apc-broken` is the untouched original and the
rollback point. New repo files: `docker/opt-qwen38/patch_fix_*.py`,
`docker/opt-qwen38/README-corrections.md`, `benchmarks/bench_context.py`,
`benchmarks/soak_hybrid_mtp.py`.

## How to run (Windows + Docker Desktop/WSL2)

```powershell
./start-qwen38-27b-ablit-xpu-int4.ps1
```

First start auto-downloads the HF model (~18 GB) into `/model`, then serves in
~3.5-4 min.

Native Linux: replace `--device /dev/dxg` with `--device /dev/dri` +
`--group-add $(stat -c '%g' /dev/dri/render*)`, drop the wsl-lib mounts.

Tuned defaults baked in (all overridable with `B70_*` env vars, see the top of
the script): MTP3 (the MTP sweep in `benchmarks/bench-results/
mtp-sweep-comparison.md` shows 3 draft tokens is the throughput sweet spot on
the B70; 4 only wins at 32k and collapses at 48k), server `MAX_MODEL_LEN=200000`,
KV pool 7.5 GiB, `MAX_NUM_SEQS=1`, prefix cache ON, `qwen3_xml` parser.
Sampling (`temperature 1.0, top_k 20, top_p 0.95`) is taken from the model's
own `generation_config.json`, which vLLM applies automatically.
The **client** window (`pi-agent/models.json`) is **150000** — see
[Context sizing](#context-sizing-server-200k-client-150k).

## Context sizing: server 200k, client 150k

These are two different numbers on purpose.

| Knob | Value | Why |
|---|---:|---|
| server `MAX_MODEL_LEN` | 200000 | ceiling only. Costs **no** VRAM — the KV pool is sized by `KV_CACHE_MEMORY_BYTES`, not by this |
| `KV_CACHE_MEMORY_BYTES` | 7.5 GiB = **205,714 tokens** | the pool. Deliberately larger than the working window |
| pi `contextWindow` | **150000** | the working window; pi compacts at `contextWindow - reserveTokens` (150000 - 32768 = ~117k prompt) |

The pool is a single shared budget: however long the current session is, that
much has to stay resident, and **the leftover is what the prefix cache can keep
for the next turn**.

| session length | pool slack | warm-turn behaviour |
|---:|---:|---|
| ≤ 120k | ~85k | 91-96 % cache hits, TTFT 10-13 s (measured, 40-turn soak) |
| 150k | ~55k | recommended working point |
| ~200k | ~0 | the hit boundary slides every turn, re-prefill, TTFT tens of seconds |

Decode throughput is **flat vs context** (48 of 64 layers are linear attention,
O(1) state), so this is a cache/latency knob, not a throughput knob:

| context | 8k | 16k | 32k | 64k | 100k |
|---|---:|---:|---:|---:|---:|
| tok/s | 35 | 36 | 49 | 37 | 32 |

**Do not shrink `KV_CACHE_MEMORY_BYTES` to match a smaller window** — the slack
*is* the feature. If you ever need genuinely long sessions, raise the pool
(`B70_KV_MEM_BYTES`) rather than lowering the client window; at ~9.5 GiB for
200k sessions, keep the desktop display off.

## Correctness: hybrid MTP + prefix caching (read this first)

**Symptom.** After a while (typically once a session passes ~90 k tokens) the
model starts emitting runs of `!`. Token 0 of the Qwen tokenizer is `!`, so
this is flat / NaN logits, not a sampling accident. Restarting the container
clears it, but continuing the same long conversation brings it straight back,
and the KV/SSM cache fills with entries that cannot be reused, so every new
session gets slower until the container is restarted again.

**Cause.** `Qwen3_5ForConditionalGeneration` is hybrid, and
`--enable-prefix-caching` forces `mamba_cache_mode="align"` on it. That keeps
recurrent (conv + SSM) state checkpoints at attention-page boundaries and
reuses them through the prefix cache. MTP speculative decode interacts with
that bookkeeping through two open upstream bugs (vllm#53505, vllm#53919), and
the running model runner is V1 (the hybrid architecture is not on the V2
allowlist), where both apply. The corrupted state is written back into the
cache, so it is persistent and does not self-correct.

**Fix.** `docker/opt-qwen38/patch_fix_backward_copy.py` +
`patch_fix_accepted_sync.py`, applied at container boot. Full write-up,
evidence table and A/B knobs: `docker/opt-qwen38/README-corrections.md`.

**Regression test.**

```bash
python benchmarks/soak_hybrid_mtp.py 120000 40 3000
# SOAK CLEAN: reached 121358 tokens, 40 turns, code=ZQX-3395, fails=0
```

It grows a cached agentic conversation turn by turn and checks every turn for
`!` degeneration *and* for silent state corruption (a secret code buried in the
system prompt — a wrong recurrent state can return another position's content
without ever producing `!`). Measured on the B70 with the shipped config:
96 % prefix-cache hit rate at ~120 k tokens, every needle probe passing.

**Do not** re-enable `B70_FIX_EAGLE_DROP=1` on 0.28.0 — the script is vendored,
but on this vLLM version it makes a hit land on a state boundary the scheduler
never materialized and the logits go NaN. The soak table in
`README-corrections.md` shows both directions.

## Throughput (measured, fresh engine, fp8 KV, MTP3, prefix cache on)

`python benchmarks/bench_context.py ctx` (natural Markdown corpus, exact prompt
sizes via `/tokenize`, client post-first timing):

| context | decode tok/s | MTP accept | prefill tok/s |
|---:|---:|---:|---:|
| 8 k | 35 | 48 % | 1 515 |
| 16 k | 36 | 52 % | 1 421 |
| 32 k | 49 | 90 % | 1 501 |
| 64 k | 37 | 63 % | 1 362 |
| 100 k | 32 | 58 % | 1 558 |

Agentic soak (prefix-cache warm, 91-96 % hit rate): 28-40 tok/s decode,
10-13 s TTFT at 100-120 k tokens.

For comparison on the same engine: MTP off = 22-26 tok/s, MTP4 = 30-42 tok/s
(MTP3 wins here), and prefix caching off (the safe fallback) = a 122 k-token
turn costs ~160 s to re-prefill. All of these are `benchmarks/bench_context.py`.

> A degraded long-running engine has been measured at ~13 tok/s at 25 k context
> (vs ~36 fresh). If throughput drifts down over a session instead of staying
> flat, that is the corruption accumulating — check the soak before blaming the
> hardware.

## Why breakable CUDA graph is enabled (VLLM_USE_BREAKABLE_CUDAGRAPH=1)

The image bakes `VLLM_XPU_ENABLE_XPU_GRAPH=1` **and**
`VLLM_USE_BREAKABLE_CUDAGRAPH=1`. On the XPU backend, CUDA-graph capture
compiles the forward into a static replay, but the GDN (linear-attention)
custom op reads per-step state (conv/ssm) from buffers staged from live block
tables. Under a normal (non-breakable) PIECEWISE graph, a long prefill can bind
those state indices to capture-time buffers and poison later requests, so this
was originally kept on for correctness.

`VLLM_USE_BREAKABLE_CUDAGRAPH=1` (the upstream experimental switch, on by
default here) marks the GDN custom op as an *eager break point*: capture ends the
current graph segment at the op, the op runs eagerly re-reading the live
per-step metadata, and capture resumes. All other layers remain in the captured
graph.

**Status: kept enabled, but note what it is and is not.** Re-testing in
September 2026 isolated the actual `!` cause to the hybrid MTP + align-mode
prefix-cache corruption (see above) — a fresh engine with this configuration
passed a 121 k-token needle soak, and the 94 k-token reproducer that used to go
NaN is clean. The breakable graph is therefore retained as a defensive default
rather than as the fix. A/B it with `-e VLLM_USE_BREAKABLE_CUDAGRAPH=0`, or
`--enforce-eager`, only when debugging — `--enforce-eager` was measured at a few
tok/s on this stack.

## Other runtime patches (why they exist)

`start.sh` applies a set of vLLM 0.28.0 patches at container boot. The two
correctness patches above are the load-bearing ones for `!`; these are the rest.

### `patch_draft_mtp_int4_v2.py` — MTP draft INT4 quantization (disabled, but kept)

**1. MTP INT4 quantization is blocked on this model variant.**

The v2 patch attempts to quantize 4 MTP linear layers (`qkv_proj`, `o_proj`,
`gate_up_proj`, `down_proj`) to INT4 using the `int4_gemm_w4a16` kernel,
while explicitly skipping the problematic `fc` layer.

However, this approach fails on the Huihui-Qwen3.8-27B-abliterated model due
to a fundamental shape mismatch:

- the MTP module expects `hidden_size=640` at runtime
- the checkpoint stores MTP weights with `hidden_size=5120` in their shape
- when quantizing `qkv_proj` (weight.shape = `[14336, 5120]` →
  `qweight.shape = [640, 14336]`), reshaping `640 * batch_size` to
  `[batch_size, 14336]` is mathematically impossible and crashes
  `torch.compile`

**2. This patch is historically significant.** The original
`patch_draft_mtp_int4.py` was created to reduce MTP DRAM reads by
quantizing the MTP layers. The v2 patch is the improved version that:

- fixes the `int4_gemm_w4a16` call signature (adds `.t()` and an explicit
  `group_idx`). This matters beyond the crash: v1's `apply()` passed the
  already-transposed qweight to the kernel and reshaped the output with
  `qweight.shape[1]` (the input dim), so with INT4 draft **enabled**, v1 is
  itself a source of garbage/NaN draft output
- skips the `fc` layer, which caused earlier dimension-mismatch errors
- preserves the quantization logic for debugging and future compatibility

> **Attribution caveat.** An earlier revision of this README blamed the
> infinite `!` on this path and on `patch_tile_mask.py`. That was wrong for
> this deployment: the `!` is the hybrid MTP + align-mode prefix-cache
> corruption documented in `docker/opt-qwen38/README-corrections.md`. The MTP
> INT4 path is disabled (`DRAFT_INT4=0`), so it cannot be the cause here; it is
> kept only as a reference and a future option.

**3. Why we keep it despite being disabled.**

- a future vLLM or XPU-kernels update may resolve the shape mismatch
- it documents the exact quantization approach that was attempted
- it serves as a reference for anyone porting MTP INT4 quantization to other
  models
- it can be re-enabled by setting `B70_DRAFT_MTP_INT4=1` once the
  underlying issue is fixed

**4. Current workaround.** Since INT4 quantization is incompatible, we set
`DRAFT_INT4=0` to keep the MTP module in BF16 precision.
`patch_draft_lmhead_int4.py` can still quantize the draft LM head on its own
(`B70_DRAFT_LMHEAD_INT4=1`, currently off) — it is the "phase S" half of the
cookbook's draft-INT4 overlay. Verify the target's emitted sequence before
enabling it: it only helps if the draft really owns a separate LM head on this
vLLM version.

> **Pairing rule:** the `DRAFT_INT4=0` setting must be used together with the
> v2 patch (`patch_draft_mtp_int4_v2.py`), never with the original
> `patch_draft_mtp_int4.py`. The v2 patch is the one that understands this
> model variant (correct `int4_gemm_w4a16` call signature — `.t()` on
> qweight, explicit `group_idx`, and the `fc` layer skipped). `DRAFT_INT4=0`
> turns the v2 patch's quantization path off while keeping its compatible
> scaffolding in place; pairing `DRAFT_INT4=0` with the v1 patch instead
> reintroduces the shape crash on the enabled path.

### `patch_tile_mask.py` — inert on this deployment (kept for reference)

The patch hardens the `USE_TD` branch of
`v1/attention/ops/triton_unified_attention.py`
([vllm#44850](https://github.com/vllm-project/vllm/pull/44850)). This image
serves with the **FLASH_ATTN** attention backend (`XpuPlatform` selects it and
the boot log says `Using Flash Attention backend` / `Setting kv cache block
size to 64 for FLASH_ATTN`), so `triton_unified_attention.py` is not on the hot
path and the patch does not do any `!`-fighting work here. It is left in place
because it is harmless and cheap, but **do not** treat it as the `!` fix — the
`!` was the hybrid MTP + align-mode prefix-cache corruption described above.

All patches are idempotent (marker-guarded) and re-apply on every container
start, since the base vLLM image does not contain them.

## vLLM 0.29.0 is blocked on WSL2 (GPTQ-INT4)

A `0.29.0-apcfix` image was built and tested on 2026-09-12. It boots
(`Using V2 Model Runner`, kernels 0.1.14.1, all nine patch scripts apply cleanly)
but **every prefill of ≥ ~64 tokens crashes EngineCore**:

```
onednn_verbose,v1,primitive,error,ocl,errcode -3,CL_COMPILER_NOT_AVAILABLE,src/gpu/intel/ocl/engine.cpp:325
... vllm/model_executor/kernels/linear/mixed_precision/xpu.py:113, in apply_weights
      out = torch.ops._xpu_C.int4_gemm_w4a16(...)
RuntimeError: could not create a primitive
```

What the investigation established (each step measured, not inferred):

| Probe | Result |
|---|---|
| Prompt 3 / 33 tokens | OK |
| Prompt 129 / 513 / 2049 tokens | fatal `could not create a primitive` |
| `VLLM_USE_V2_MODEL_RUNNER=0` (force V1) | same crash → not the runner |
| kernels `0.1.13.1` instead of `0.1.14.1` | same crash |
| kernels `0.1.12.3` with vLLM 0.29.0 | unsupported combo, engine hangs |
| 0.28.0 image, same prompt sizes | all OK |
| `mixed_precision/xpu.py`, `auto_gptq.py`, `MPLinearKernel.py`, `_xpu_C` w4a16 symbols | **byte-identical between 0.28 and 0.29** |

`int4_gemm_w4a16` *is* `dnnl_matmul_w4a16_int4`: the XPU kernels route GPTQ
W4A16 through oneDNN, which JIT-compiles its GPU kernels. On WSL2 the container
has no `/dev/dri`, so OpenCL comes only from the WSL driver shim — which has the
device but **no OpenCL compiler**. The vLLM 0.28 image's toolchain happens to
satisfy oneDNN without a JIT; the official v0.29.0 image does not.

The 0.29.0 image was therefore removed. To retry after an Intel driver / vLLM
XPU fix:

```powershell
docker pull vllm/vllm-openai-xpu:v0.29.0
docker build -t zrlu/qwen38-27b-arc-pro-b70:0.29.0-apcfix docker `
  --build-arg BASE_IMAGE=vllm/vllm-openai-xpu:v0.29.0     # ~10 s
$env:B70_IMAGE='zrlu/qwen38-27b-arc-pro-b70:0.29.0-apcfix'
./start-qwen38-27b-ablit-xpu-int4.ps1
# then: python benchmarks/bench_context.py ctx   (any >64-token prefill is the canary)
```

The base image is kept locally so this is a 10-second rebuild. If disk is tight,
`docker rmi vllm/vllm-openai-xpu:v0.29.0` frees ~12 GB; it is a 5-minute re-pull.

Two further notes from that test, useful when retrying:

- 0.29.0 runs **Model Runner V2** for this model, so `patch_fix_accepted_sync.py`
  becomes inert (V2 keeps the accepted-token counters GPU-resident; the script
  detects this and says so). `patch_fix_backward_copy.py` stays active.
- 0.29.0 does **not** contain upstream #53945/#54713 (the mamba align-cache
  state-position fixes), so `B70_FIX_EAGLE_DROP=1` would still need re-testing.

## Headroom: is the hardware maxed out?

Derived from the measured numbers above, not from theory. Per-step DRAM traffic
for this model (INT4 body 4.125 bit/param → 14.0 GB, shared BF16 lm_head
2.54 GB read once per forward, MTP layer 0.85 GB):

| mode | traffic/step | ideal step @ ~520 GB/s | measured step |
|---|---:|---:|---:|
| no-spec | ~16.5 GB | 32 ms | 30.4 ms (cookbook, 32.9 t/s) |
| MTP1 | ~19.9 GB | 38 ms | 38.3 ms (cookbook, 52 t/s) |
| MTP3 | ~26.9 GB | 50 ms | **59-76 ms (this repo)** |
| MTP4 | ~30.3 GB | 56 ms | 57.7 ms (cookbook, 83.7 t/s) |

**Prefill is maxed.** 1.4-1.9 k tok/s here vs 1.77-1.85 k in the cookbook; a
bigger `--max-num-batched-tokens` is flat on this dense model (the cookbook
measured a gain only on MoE).

**Decode step efficiency is 20-35 % above the roofline** while the cookbook's
build sits at 96-100 % — so there is real, bounded headroom in *step latency*,
plus a larger amount in *traffic*.

| lever | expected | available today? |
|---|---:|---|
| draft-INT4 overlay (phase S+M1) | **+33-39 %** | yes, patches in the image; needs the quality gate re-run, and this model may hit the documented MTP shape mismatch (only the LM-head half may apply) |
| step efficiency: `VLLM_USE_BREAKABLE_CUDAGRAPH` A/B | +5-20 % | yes (cheap: one restart + the ctx bench) |
| acceptance x MTP depth | up to +50 % | content-dependent, not a knob (this repo's Markdown corpus reads lower than real agentic traffic) |
| fused GDN MTP (vllm#52539) | +5-10 % | no — needs 0.29, blocked |
| FULL cudagraph for spec decode (vllm#53407, #50488) | +5-10 % | no — needs 0.29, blocked |

Why the 0.29 items matter: the engine captures **one** FULL decode graph
(`Capturing CUDA graphs (decode, FULL): 1/1`) and runs the spec-decode step
through a PIECEWISE graph, so the 48 GDN layers launch per layer; and this
model's 16:48 key/value head ratio falls back to the Triton GDN decode path
instead of the fused kernel.

**Ceiling.** The cookbook's own Windows standalone kits for the same class of
model are documented as "~70 tok/s class" on WSL2, and the validated single-card
maximum for a Qwen3.8-27B is **112.65 tok/s** (MTP4 + draft-INT4, cache off,
230 W). So 35-49 tok/s today → ~70-90 tok/s is the realistic target without
upstream fixes, and that is a *traffic* problem, not a compute problem: the
shared BF16 lm_head is ~38 % of every step's DRAM traffic.

Related vLLM 0.29.0 PRs, for when the WSL blocker clears:
[#53407](https://github.com/vllm-project/vllm/pull/53407),
[#50488](https://github.com/vllm-project/vllm/pull/50488),
[#52539](https://github.com/vllm-project/vllm/pull/52539),
[#52389](https://github.com/vllm-project/vllm/pull/52389) (would retire
`patch_xpu_single_gpu_warmup.py`),
[#48109](https://github.com/vllm-project/vllm/pull/48109),
[#53945](https://github.com/vllm-project/vllm/pull/53945) +
[#54713](https://github.com/vllm-project/vllm/pull/54713) (would let
`B70_FIX_EAGLE_DROP=1` come back).

`docker/Dockerfile` takes `ARG BASE_IMAGE` / `ARG KERNELS_VERSION`, so a
retry is a one-line build plus re-checking the fail-closed patch anchors.

## License / credits

Apache-2.0 (inherited; quantization only). Models derived from
[huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated),
(Qwen/Qwen3.8-27B lineage); tuning methodology from the SergiioB B70 cookbook.
Benchmarks are Windows WSL2, self-reported - not comparable cell-for-cell with
native-Linux cookbook numbers.
