# Qwen3.8-27B x Intel Arc Pro B70 - one-click Docker, pi-agent setup, benchmarks

Abliterated **Qwen3.8-27B** -> **GPTQ-INT4 (sym G128, MTP-BF16)**, tuned and
published for a **single Intel Arc Pro B70** (Xe2, 32 GB class). Reference
stack: vLLM XPU `0.28.0`, kernels `0.1.12.3`, MTP3 speculative decoding
(BF16 draft, `DRAFT_INT4=0`), prefix caching, `qwen3_xml` tool-call parser.

| Artifact | Link |
|---|---|
| Model (HF, huihui) | [zrlu/Huihui-Qwen3.8-27B-abliterated-GPTQ-Int4-sym-G128-MTP-BF16-B70](https://huggingface.co/zrlu/Huihui-Qwen3.8-27B-abliterated-GPTQ-Int4-sym-G128-MTP-BF16-B70) |
| Image (Docker Hub) | `docker pull zrlu/qwen38-27b-arc-pro-b70:latest` |
| Upstream reference | [SergiioB/intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook) |

## How to run (Windows + Docker Desktop/WSL2)

```powershell
./start-qwen38-27b-ablit-xpu-int4.ps1
```

First start auto-downloads the HF model (~18 GB) into `/model`, then serves in
~4-5 min

Native Linux: replace `--device /dev/dxg` with `--device /dev/dri` +
`--group-add $(stat -c '%g' /dev/dri/render*)`, drop the wsl-lib mounts.

Tuned defaults baked in: MTP3 (the MTP sweep in `benchmarks/bench-results/
mtp-sweep-comparison.md` shows 3 draft tokens is the throughput sweet spot on
the B70; 4 only wins at 32k and collapses at 48k), Context=100000,
KV=4617089843, `MAX_NUM_SEQS=1`, prefix cache ON, `qwen3_xml` parser.
Sampling (`temperature 1.0, top_k 20, top_p 0.95`) is taken from the model's
own `generation_config.json`, which vLLM applies automatically.

## Why breakable CUDA graph is enabled (VLLM_USE_BREAKABLE_CUDAGRAPH=1)

The image bakes `VLLM_XPU_ENABLE_XPU_GRAPH=1` **and**
`VLLM_USE_BREAKABLE_CUDAGRAPH=1`. On the XPU backend, CUDA-graph capture
compiles the whole forward into a static replay, but the GDN (linear-
attention) custom op reads per-step state (conv/ssm) from buffers that are
staged from live block tables. Under a normal (non-breakable) PIECEWISE
graph, a long prefill can bind those state indices to capture-time buffers,
corrupt the shared GDN state, and poison later requests (flat logits
repeating a single token).

`VLLM_USE_BREAKABLE_CUDAGRAPH=1` (the upstream experimental switch, on by
default here) marks the GDN custom op as an *eager break point*: capture
ends the current graph segment at the op, the op runs eagerly re-reading the
live per-step metadata, and capture resumes. Every inference replay therefore
uses correct, current GDN state. All other layers remain in the captured
graph, so the speed benefit of CUDA graphs is kept (decode ~2x vs.
`--enforce-eager` on this part, measured under `benchmarks/`).

Setting it to 0 (e.g. `-e VLLM_USE_BREAKABLE_CUDAGRAPH=0`) or forcing
`--enforce-eager` are the two ways back to the unsafe/slow paths and are
only for A/B debugging — do NOT ship the image without breakable enabled.

## Runtime patches (why they exist)

`start.sh` applies a set of vLLM 0.28.0 patches at container boot. Three of
them are load-bearing and easy to "clean up" by accident:

### `patch_draft_mtp_int4_v2.py` — MTP draft INT4 quantization (disabled, but required)

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
  itself a source of garbage/NaN draft output — the same infinite-`!`
  symptom. The v2 patch fixes `!` on that path; `patch_tile_mask.py`
  (below) fixes the other, unrelated `!` source (vLLM 0.28.0's unmasked KV
  loads). With `DRAFT_INT4=0` the v2 quantization path is inert, so in the
  current deployment only the tile_mask fix is doing `!`-fighting work
- skips the `fc` layer, which caused earlier dimension-mismatch errors
- preserves the quantization logic for debugging and future compatibility

**3. Why we keep it despite being disabled.**

- a future vLLM or XPU-kernels update may resolve the shape mismatch
- it documents the exact quantization approach that was attempted
- it serves as a reference for anyone porting MTP INT4 quantization to other
  models
- it can be re-enabled by setting `B70_DRAFT_MTP_INT4=1` once the
  underlying issue is fixed

**4. Current workaround.** Since INT4 quantization is incompatible, we set
`DRAFT_INT4=0` to keep the MTP module in BF16 precision. The remaining
active `!`-risk in that configuration is the vLLM 0.28.0 KV-cache NaN bug,
which the `patch_tile_mask.py` fix (below) handles; and
`patch_draft_lmhead_int4.py` safely quantizes the LM head to INT4 without
the same shape constraints.

> **Pairing rule:** the `DRAFT_INT4=0` setting must be used together with the
> v2 patch (`patch_draft_mtp_int4_v2.py`), never with the original
> `patch_draft_mtp_int4.py`. The v2 patch is the one that understands this
> model variant (correct `int4_gemm_w4a16` call signature — `.t()` on
> qweight, explicit `group_idx`, and the `fc` layer skipped). `DRAFT_INT4=0`
> turns the v2 patch's quantization path off while keeping its compatible
> scaffolding in place; pairing `DRAFT_INT4=0` with the v1 patch instead
> reintroduces the shape crash on the enabled path.

### `patch_tile_mask.py` — fixes vLLM 0.28.0 XPU "infinite `!`" NaN bug

vLLM 0.28.0's XPU attention backend has a bug in the `USE_TD` branch: `_load_kv_tile_td`
loads K and V tensors from the paged KV cache **without** applying
tile_mask, allowing uninitialized memory (containing NaN values) to
propagate through attention → softmax → logits. The final output then
degenerates into repeated `!` characters (token ID 0). The patch adds
`tl.where(tile_mask[None, :], K_load, 0.0)` and
`tl.where(tile_mask[:, None], V_load, 0.0)` to filter out invalid cache
slots, preventing NaN propagation and fixing the infinite-`!` output issue.
Upstream: [vllm-project/vllm#44850](https://github.com/vllm-project/vllm/pull/44850).

All patches are idempotent (marker-guarded) and re-apply on every container
start, since the base vLLM image does not contain them.

## License / credits

Apache-2.0 (inherited; quantization only). Models derived from
[huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated),
(Qwen/Qwen3.8-27B lineage); tuning methodology from the SergiioB B70 cookbook.
Benchmarks are Windows WSL2, self-reported - not comparable cell-for-cell with
native-Linux cookbook numbers.
