# Hybrid MTP + prefix-caching corrections

Vendored from [SergiioB/intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)
(`patches/patch_fix_*.py`, 2026-09). Verbatim; do not edit by hand — re-pull from
upstream if they change. The scripts are fail-closed: a moved source anchor makes
the container exit instead of serving silently corrupted output.

## The defect this fixes

`Qwen3_5ForConditionalGeneration` is a hybrid linear/full-attention model. With
`--enable-prefix-caching` vLLM forces `mamba_cache_mode="align"`, which keeps
recurrent (conv + SSM) state checkpoints at attention-page boundaries so they can
be reused. Combined with **MTP speculative decode** and the **V1 model runner**
(what this image runs — the hybrid architecture is not on the V2 allowlist), two
upstream bugs silently corrupt that state:

| Patch | Upstream | What it fixes |
|---|---|---|
| `patch_fix_backward_copy.py` | [vllm#53505](https://github.com/vllm-project/vllm/pull/53505) | An align-boundary state copy with `dst < src` overwrites an *earlier* checkpoint column — the one the in-flight request reads and the prefix cache has already published — with state from a *later* position. Three copy sites: `postprocess_mamba_fused_kernel`, `precopy_mamba_align_fused_kernel`, `collect_mamba_copy_meta`. |
| `patch_fix_accepted_sync.py` | [vllm#53919](https://github.com/vllm-project/vllm/pull/53919) | The step-N accepted-token D2H copy lands in step-N row order while step N+1's `_update_states`/`condense()` permute the same pinned host buffer before the event is awaited, and `_prepare_inputs` gathers it a second time. A request decodes with another request's accepted count, which becomes its conv offset and GDN snapshot selector. V1-only; the script detects the live runner and says so. |

Neither fix is in vLLM 0.29.0 (both PRs were still open on 2026-09-11).

## `patch_fix_eagle_drop.py` is vendored but **disabled** on this image

`B70_FIX_EAGLE_DROP=0`. Reason: it is correct in intent
([vllm#48375](https://github.com/vllm-project/vllm/pull/48375) — the mamba finder
accepts `drop_eagle_block` and ignores it), but on vLLM **0.28.0** it makes a
prefix-cache hit land one block lower than the position the scheduler actually
materialized a state checkpoint at. The restored state is uninitialized memory
and the logits go NaN (which surfaces as repeats of token 0, `!`).

Measured on the B70 (vLLM 0.28.0, kernels 0.1.12.3, fp8 KV, MTP3, prefix cache
on), replaying one fixed 94 396-token agentic conversation:

| `B70_FIX_EAGLE_DROP` | result |
|---|---|
| `1` | degenerate `The user wants!!!!!…`, logprobs endpoint returns `nan` |
| `0` | clean, both cold prefill and prefix-cache-hit replay |

Upstream fixed the materialization side in
[#53945](https://github.com/vllm-project/vllm/pull/53945) ("Cache the Mamba state
at the block-grid position of EAGLE resume", merged 2026-09-08) plus follow-ups
#54713 / #55450. Those are **not** in the vLLM 0.29.0 image (checked 2026-09-12),
and 0.29.0 additionally cannot run this GPTQ-INT4 model on WSL2 at all — see the
"vLLM 0.29.0 is blocked on WSL2" section of the repository README. Re-test
`B70_FIX_EAGLE_DROP=1` after an upstream fix lands.

## Evidence that the current configuration is correct

`benchmarks/soak_hybrid_mtp.py` grows a cached agentic conversation turn by turn
and checks every turn for (a) degenerate output and (b) silent state corruption
via a secret code buried in the system prompt (a wrong recurrent state can return
another position's content without ever producing `!`).

```
python benchmarks/soak_hybrid_mtp.py 120000 40 3000
SOAK CLEAN: reached 121358 tokens, 40 turns, code=ZQX-3395, fails=0
```

With this config the same soak reaches ~120 k tokens at 96 % prefix-cache hit
rate, 28-40 tok/s decode, ~10-13 s TTFT, and every needle probe passes. Before
the fixes, the conversation degenerated into `!` around turn 30 / 94 k tokens.

## A/B knobs

```
-e B70_FIX_ACCEPT_SYNC=0      # repro the accepted-token race
-e B70_FIX_BACKWARD_COPY=0    # repro the backward state copy
-e B70_FIX_EAGLE_DROP=1       # enable the drop (NaN on 0.28.0 — see above)
-e B70_EAGLE_DROP_FINE_UNITS=page
```

Nuclear alternatives, both costing real performance: `PREFIX_CACHE=0` (drops
align mode entirely — correct, but a 122 k-token turn then costs ~160 s to
re-prefill) or `MTP_TOKENS=0` (correct, but ~1.4x slower decode on this stack).

## Boot-log acceptance check

```
[backward-copy] OK: 2 arbol(es) x 3 sitios de copia cubiertos …
[eagle-drop] desactivado por B70_FIX_EAGLE_DROP=0 (vLLM intacto)
[accept-sync] … runner V1 VIVO: el fix queda ACTIVO en este build
[accept-sync] OK: 4 hunks aplicados ahora en 2 arbol(es) (V1 VIVO: el fix esta ACTIVO)
```

A missing anchor for the two required patches makes the container exit non-zero.
