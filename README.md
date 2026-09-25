# Qwen3.8-27B x Intel Arc Pro B70 - one-click Docker, pi-agent setup, benchmarks

Abliterated **Qwen3.8-27B** -> **GPTQ-INT4 (sym G128, MTP-BF16)**, tuned and
published for a **single Intel Arc Pro B70** (Xe2, 32 GB class). Reference
stack: vLLM XPU `0.28.0`, kernels `0.1.12.3`, MTP3 speculative decoding
(BF16 draft, `DRAFT_INT4=0`), prefix caching, `qwen3_xml` tool-call parser.

Images:

| Tag | What |
|---|---|
| `zrlu/qwen38-27b-arc-pro-b70:0.30.0` | **current / default**: vLLM 0.30.0 + kernels 0.1.15.4 + UMD 26.35/IGC 2.41.5. V1 runner + `B70_PATCH_SET=minimal`. Validated: 121k-token 40-turn soak clean. |
| `zrlu/qwen38-27b-arc-pro-b70:0.29.1-nightly` | previous: vLLM 0.29.1 nightly + kernels 0.1.14.1. Same wedge, same throughput. |
| `zrlu/qwen38-27b-arc-pro-b70:0.28.0-apcfix` | **stable fallback**: vLLM 0.28.0 + the vendored mamba correctness patches + the draft-INT4 overlay. ~10-20 % slower, no driver requirement, never wedged in testing. |

| Artifact | Link |
|---|---|
| Model (HF, huihui) | [zrlu/Huihui-Qwen3.8-27B-abliterated-GPTQ-Int4-sym-G128-MTP-BF16-B70](https://huggingface.co/zrlu/Huihui-Qwen3.8-27B-abliterated-GPTQ-Int4-sym-G128-MTP-BF16-B70) |
| Upstream reference | [SergiioB/intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook) |

## Session report — 2026-09-12, updated 2026-09-21

**Update 2026-09-21 (2): draft-INT4 overlay, +20-55 % and quality-neutral.**
The cookbook's "phase S" overlay (a private INT4 copy of the draft's LM head;
the target's fp16 lm_head is never mutated) is now on by default
(`B70_DRAFT_LMHEAD_INT4=1`). It cuts the draft's per-step DRAM reads from
4 x 2.54 GB to 4 x 0.66 GB and is verified output-neutral: see
[Draft-INT4 overlay](#draft-int4-overlay).

**Update 2026-09-21 (1): migrated to vLLM 0.29.1 nightly, then back.** An Intel
Windows driver update (32.0.101.9030, 2026-09-16) removed the blocker that made
every 0.29.x build unusable here, and the nightly is genuinely faster
(52.7/47.5/47.4/38.1/39.2 vs 35/36/49/37/32 tok/s at 8k-100k). But it **wedges
intermittently** — 3 of 4 runs of a 7-distinct-prompt sequence hung on the 6th
request (EngineCore 100 % CPU, `Running: 1`, no progress, only a restart
recovers), and there were two intermittent boot segfaults. The overlay alone
recovers most of the speed on the stable 0.28 stack. Both are kept:
**0.29.1-nightly is the default** (faster, and it fixes the 0 %-acceptance
collapse), with **0.28.0-apcfix + overlay as the one-command stable fallback**.

**Original report (2026-09-12).**

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
shipped image; `:0.28.0-apcfix` is the previous generation and the rollback point.
(The pre-fix `:snapshot-0.28.0-apc-broken` reference image is no longer retained
locally — it was only ever useful for reproducing the `!` bug, which is fixed.)
New repo files: `docker/opt-qwen38/patch_fix_*.py`,
`docker/opt-qwen38/README-corrections.md`, `benchmarks/bench_context.py`,
`benchmarks/soak_hybrid_mtp.py`.

## How to run (Windows + Docker Desktop/WSL2)

```powershell
./start-qwen38-27b-ablit-xpu-int4.ps1
```

First start auto-downloads the HF model (~18 GB) into `/model`, then serves in
~3.5-4 min.

> **Requirement for the default (0.29.1-nightly) image:** Intel Arc Windows
> driver **32.0.101.9030 (2026-09-16) or newer**. On older drivers its MTP path
> hangs on any prefill above ~130 tokens. The `stable` fallback image does not
> need it. See [Upgrading](#upgrading-to-vllm-0291-nightly-experimental).

Native Linux: replace `--device /dev/dxg` with `--device /dev/dri` +
`--group-add $(stat -c '%g' /dev/dri/render*)`, drop the wsl-lib mounts.

Tuned defaults baked in (all overridable with `B70_*` env vars, see the top of
the script): MTP3 (the MTP sweep in `benchmarks/bench-results/
mtp-sweep-comparison.md` shows 3 draft tokens is the throughput sweet spot on
the B70; 4 only wins at 32k and collapses at 48k), server `MAX_MODEL_LEN=200000`,
KV pool 7.5 GiB, `MAX_NUM_SEQS=1`, prefix cache ON, `qwen3_xml` parser,
`B70_PATCH_SET=none` (0.29.1 image).
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

**Fix.** On the 0.28 generation:
`docker/opt-qwen38/patch_fix_backward_copy.py` +
`patch_fix_accepted_sync.py`, applied at container boot. On the current
**0.29.1-nightly** image upstream already carries the equivalent fixes
(#53945 / #54713 / #55450, plus Model Runner V2 which removes the
accepted-token race by design), so it runs with `B70_PATCH_SET=none`. Full
write-up, evidence table and A/B knobs: `docker/opt-qwen38/README-corrections.md`.

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

`python benchmarks/bench_context.py ctx`, decode tok/s (client post-first):

| context | 0.28.0 | **0.28.0 + draft-INT4 (shipped)** | 0.29.1-nightly | nightly + draft-INT4 |
|---:|---:|---:|---:|---:|
| 8 k | 35 | **54.4** | 52.7 | 65.3 |
| 16 k | 36 | **49.1** | 47.5 | 55.8 |
| 32 k | 49 | **50.2** | 47.4 | 57.6 |
| 64 k | 37 | **38.4** | 38.1 | 46.8 |
| 100 k | 32 | **45.7** | 39.2 | 45.7 |
| prefill tok/s | 1400-1560 | 1441-2246 | 1643-2500 | 1394-2530 |

The shipped config (**0.28.0-apcfix + overlay**) is ~1.5x the original 0.28 at
8 k and ~1.4x at 100 k. The nightly (now the default) adds another ~10-20 %
on top.

Agentic soak on the nightly (prefix-cache warm, 79-94 % hit rate): 37-52 tok/s
decode, 7-15 s TTFT at 100-120 k tokens. On 0.28 (no overlay) the same soak was
28-40 tok/s with 10-13 s TTFT.

Other points on the same engine: MTP off = 22-26 tok/s, MTP4 = 30-42 tok/s
(MTP3 wins here), and prefix caching off (the safe fallback) = a 122 k-token
turn costs ~160 s to re-prefill.

> A degraded long-running engine has been measured at ~13 tok/s at 25 k context
> (vs ~36 fresh). If throughput drifts down over a session instead of staying
> flat, that is the corruption accumulating — check the soak before blaming the
> hardware.

## Draft-INT4 overlay

`B70_DRAFT_LMHEAD_INT4=1` (on by default in the launcher) installs
`patch_draft_lmhead_int4.py`, the "phase S" half of the cookbook's draft-INT4
overlay. At the first forward it quantizes a **private** INT4 g128 copy of the
draft's LM head into `model._b70_lmhead_int4`:

```
[B70] draft LM head INT4: 2.54 GB fp16 -> 0.66 GB INT4 (ahorro 1887.2 MB/lectura)
```

The target's fp16 lm_head is **never mutated**, so verification is unchanged —
which is why the output is bit-identical, not merely "similar".

Why it is worth it: the shared BF16 lm_head is ~38 % of every decode step's DRAM
traffic (the draft reads it once per draft position). Cutting it to INT4 removes
~5.7 GB/step at MTP3. Measured: **+20-55 %** decode, MTP acceptance unchanged.

**Quality gate** — `python benchmarks/quality_ab.py <label>` runs 7 varied
greedy prompts and prints a SHA per prompt:

| | 0.28 + overlay | 0.28 plain |
|---|---|---|
| 7 prompts | 7/7 **bit-identical** | — |
| repeat run | identical to the first (deterministic) | — |

On the 0.29.1 nightly the same 7 prompts were also output-neutral except one
borderline Chinese prompt that differed in 1 of 4 runs (run-to-run noise could
not be fully separated there; on 0.28 it is clean).

Knobs: `B70_DRAFT_LMHEAD_INT4=0` disables it; `B70_DRAFT_INT4=1` additionally
enables the MTP-linear phase ("M1"), which this model's README notes may hit a
shape mismatch — not used here.

## Why breakable CUDA graph is now DISABLED (VLLM_USE_BREAKABLE_CUDAGRAPH=0)

The image bakes `VLLM_XPU_ENABLE_XPU_GRAPH=1` while
`VLLM_USE_BREAKABLE_CUDAGRAPH=0`. The breakable switch marks the GDN
(linear-attention) custom op as an *eager break point*: capture ends the current
graph segment there, the op runs eagerly against live per-step metadata, and
capture resumes. That splits every decode step into several replay calls, each
one a host<->device round trip.

**A/B, 2026-09-25, same container, same prompts, identical prefix-cache hit
rates (so the comparison is fair):**

| context | breakable=1 ttft / step | breakable=0 ttft / step |
|---|---:|---:|
| 3–10k | 15.7 s / 56.9 ms | **2.6 s** / **50.8 ms** |
| 10–20k | 15.3 s / 59.3 ms | **3.0 s** / **52.6 ms** |
| 20–40k | 15.6 s / 64.0 ms | **3.7 s** / **57.0 ms** |

Request-start drops ~76–83%, decode steps ~11%. A `py-spy record` during a slow
step put 80% of the CPU samples in `breakable_cudagraph.replay` ->
`torch/xpu/graphs.py:107 replay`, reached from the MTP draft path
(`propose_draft_token_ids`); the target forward was only 16% and the GPU sat at
26–28 W of a 230 W budget — host-side replay overhead, not GPU work.

**Correctness gate (the reason it used to be on):** the original justification
was that a non-breakable PIECEWISE graph could bind the GDN per-step state to
capture-time buffers and poison later requests. Testing in September 2026
isolated the real `!` cause to hybrid MTP + align-mode prefix-cache corruption
(the `#53505` / `#53919` bugs, see above), and with
`VLLM_USE_BREAKABLE_CUDAGRAPH=0` a fresh engine passed a **121,282-token /
40-turn needle soak with 0 failures**. The old claim is retired: breakable is off
by default and only worth turning on as a debugging lever. `--enforce-eager`
remains far worse (a few tok/s on this stack).

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

## Upgrading to vLLM 0.29.1 nightly

### 2026-09-24: v0.30.0 tried — the wedge is unchanged

vLLM v0.30.0 (2026-09-22) + `vllm-xpu-kernels` 0.1.15.4 was built and tested
(`zrlu/...:0.30.0`, `docker/Dockerfile.nightly` with
`--build-arg BASE_IMAGE=vllm/vllm-openai-xpu:v0.30.0 --build-arg
KERNELS_VERSION=0.1.15.4`). Findings:

- **The GPTQ checkpoint still loads** (18.32 GiB). v0.30.0 removed GPTQ
  activation ordering (`g_idx`, #54809), which was a real risk for this model.
- **The V2-runner wedge is NOT fixed.** It reproduces at the same place, with a
  byte-identical `py-spy` stack — `urEventWait` (`libze_intel_gpu.so.1.17.39758`)
  -> `build()` at `gdn_attn.py:307`, the same source line. v0.30.0's
  [#51565](https://github.com/vllm-project/vllm/pull/51565) (*"Fix stateless
  first-chunk classification"*, a real bug: reusable Mamba state pages are not
  zeroed, so a misclassified first token can consume a previous request's state)
  is a **different** bug and does not cover this one.
- **V1 still avoids it**, and throughput is unchanged: 61.5 / 54.3 / 49.7 tok/s
  at 1.2k / 3k / 8k context, and a clean 121,281-token / 40-turn soak.
- The patch trio still applies to the v0.30.0 tree, so `B70_PATCH_SET=minimal`
  (vllm#53919 + vllm#53505) is used as on the nightly.

So 0.30.0 becomes the default (a release, not a nightly, plus #51565, the WSL
V2 fix #56908, GDN capture metadata without a device sync #55404, kernels warmed
before capture #55341, and `FULL_DECODE_ONLY` fallback #55095) — with **V1** and
the wedge still open upstream.

### The wedge, and its root cause

Symptom: the EngineCore stops making progress while the HTTP API keeps working
(`/health` and `/metrics` answer, the container shows `Up`), CPU pegged at
~100 %, `num_requests_running` stuck, only a restart recovers. It happened both
mid-request (3 of 4 runs of a 7-distinct-prompt sequence, always on the 6th
request) and after a long idle period.

Captured with `py-spy` (the launcher passes `--cap-add SYS_PTRACE` and the image
ships py-spy), full dump in `docs/wedge-v2-gdn-ureventwait.txt`:

```
Thread 422 (active): "MainThread"
    sched_yield (libc.so.6)
    0x... (libze_intel_gpu.so.1.17.39758)
    0x... (libze_intel_gpu.so.1.17.39758)
    ur::level_zero::urEventWait (libur_adapter_level_zero.so.0)
    build (vllm/v1/attention/backends/gdn_attn.py:307)
    build_attn_metadata (vllm/v1/worker/gpu/attn_utils.py:496)
    prepare_attn (vllm/v1/worker/gpu/model_states/mamba_hybrid.py:332)
    execute_model (vllm/v1/worker/gpu/model_runner.py:1816)
```

So it is a **Level Zero event wait that never returns**, spun on by the UMD,
while the GPU sits idle (verified: a fresh process in the same container ran a
XPU op in 0.9 s during a wedge, and Windows GPU compute counters stayed below
0.5 %). It is not a vLLM Python loop (`gdn_attn.py` has no `while`/`for`).

### The fix: run the V1 model runner

The whole path is **Model Runner V2** (`mamba_hybrid.py prepare_attn` -> V2's
metadata build). The 0.28 generation runs V1 and never showed this.

| runner | 7-distinct-prompt sequence |
|---|---|
| V2 (vLLM default) | wedged 3 of 4 runs, always on request 6 |
| **V1** (`VLLM_USE_V2_MODEL_RUNNER=0`) | **21/21 requests passed, 3 rounds** |

Decode throughput is unaffected (`bench_context.py ctx`, tok/s):

| context | 8 k | 16 k | 32 k | 64 k | 100 k |
|---|---:|---:|---:|---:|---:|
| V2 | 65.3 | 55.8 | 57.6 | 46.8 | 45.7 |
| **V1** | 61.5 | 54.4 | 56.7 | 43.1 | **50.9** |

V1 has the accepted-token race that V2 fixes by design, so the image also sets
`B70_PATCH_SET=minimal` (vllm#53919 accepted-token sync + vllm#53505 backward
state copy, both still unmerged upstream). Boot log should say
`[accept-sync] ... V1 VIVO: el fix esta ACTIVO`.

### Other things tried (and kept)

- **Intel UMD upgraded** to compute-runtime `26.35.39758.10` + IGC `2.41.5`
  (`libze_intel_gpu.so.1.15.39122` -> `1.17.39758`). This did **not** fix the
  wedge — it reproduces identically on the new UMD — but it is the current
  release and is kept.
- **Triton JIT cache persisted** across container recreations
  (`$repoRoot\.triton-cache` -> `/root/.triton/cache`), so the spec-decode
  kernels are not recompiled on every start.
- **`--cap-add SYS_PTRACE` + py-spy** in the image: without them a wedge is
  undiagnosable (the first attempt failed with `Permission denied`).

The image is `zrlu/qwen38-27b-arc-pro-b70:0.29.1-nightly`:

```
vLLM 0.29.1rc1.dev422+gd05da62e9.xpu   kernels 0.1.14.1   Model Runner V2
```

Built with `docker/Dockerfile.nightly`. It needs **no runtime patches** — upstream
now contains the mamba align-cache fixes this repo used to vendor (#53945 /
#54713 / #55450, all merged 2026-09-08..11, i.e. after the v0.29.0 release
branch was cut).

### Two WSL2 workarounds the image bakes in (required, not tuning)

**1. Library search order.** `LD_LIBRARY_PATH` must put the container's own libs
before `/usr/lib/wsl/lib`. The WSL driver ships `libigdfcl.so.2`, which otherwise
shadows the container's IGC 2.38.2 in `/usr/local/lib`; that ABI mix makes
oneDNN's GPTQ W4A16 GEMM report `CL_COMPILER_NOT_AVAILABLE` → `RuntimeError:
could not create a primitive` on any prefill of roughly ≥ 64 tokens.

```
LD_LIBRARY_PATH=/usr/local/lib:/opt/venv/lib:/usr/lib/wsl/lib:/opt/ucx/lib:/tmp/ucx_install/lib
```

This was the actual cause of the long-standing "0.29 is blocked" — one line of
library ordering, not a missing OpenCL compiler.

**2. `ONEAPI_DEVICE_SELECTOR=level_zero:*`** hides the SYCL OpenCL backend so
oneDNN cannot select an OCL engine that cannot compile on WSL2.

### The other blocker was the Windows driver

With the two workarounds above, 0.29.1 nightly still hung on any prefill above
~130 tokens — EngineCore at 100 % CPU with no progress, and once wedged the
engine never served another request (`Running: 0 reqs`, `/health` still 200).
It reproduced with **zero runtime patches**, so it was upstream, not ours.

| MTP depth | 2026-09-16 driver and older | 32.0.101.9030 (2026-09-16) |
|---|---|---|
| 0 (no spec) | OK | OK |
| 1 | OK | OK |
| 2 | **hang** | OK |
| 3 | **hang** | OK |

Note the WSL-visible user-space libs under `C:\Windows\System32\lxss\lib` were
*not* replaced by that driver update (still dated 2026-07-14); the fix came from
the kernel-mode driver. So: **this stack requires the 2026-09-16 or newer Intel
Arc driver.**

### Patch policy

`start.sh` takes `B70_PATCH_SET`:

| Value | Applies | Used by |
|---|---|---|
| `none` | nothing | the 0.29.1-nightly image (its ENV sets it) |
| `minimal` | only the still-missing vllm#53505 backward-state-copy guard | offered for a future vLLM that lacks it |
| `full` (default) | the whole 0.28-era stack | the 0.28.0 image |

Applying the 0.28-era patches (`patch_gdn_mixed_split_v5.py` et al.) **on top of**
the nightly makes the engine wedge again — upstream's own handling conflicts with
the 0.28-era rewrite. Do not set `B70_PATCH_SET=full` on the nightly.

### Switching

```powershell
./start-qwen38-27b-ablit-xpu-int4.ps1       # DEFAULT: 0.29.1-nightly
./start-qwen38-27b-ablit-xpu-nightly.ps1    # same thing, explicit
./start-qwen38-27b-ablit-xpu-stable.ps1     # fallback: 0.28.0-apcfix + overlay
```

The wrappers just set `B70_IMAGE` and the matching `B70_LD_LIBRARY_PATH` and
call the main launcher (`B70_PATCH_SET=none` is baked into the nightly image's
ENV; the 0.28 image falls back to `full`).

### Rollback

```powershell
$env:B70_IMAGE='zrlu/qwen38-27b-arc-pro-b70:0.28.0-apcfix'
$env:B70_LD_LIBRARY_PATH='/usr/lib/wsl/lib:/opt/ucx/lib:/tmp/ucx_install/lib:/opt/venv/lib:/usr/local/lib'
./start-qwen38-27b-ablit-xpu-int4.ps1
```

(0.28 was validated with the WSL driver's libs first; the default order in the
launcher is the one 0.29.1 needs. `B70_PATCH_SET` is left unset so the 0.28 image
falls back to `full`.)

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
