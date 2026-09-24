### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code> (CPU vulnerability list trimmed)</summary>

```text
==============================
        System Info
==============================
OS                           : Ubuntu 24.04.5 LTS (x86_64)
GCC version                  : (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
CMake version                : version 4.4.3
Libc version                  : glibc-2.39

==============================
       PyTorch Info
==============================
PyTorch version              : 2.13.0+xpu
Is debug build               : False
XPU used to build PyTorch    : 20260000

==============================
      Python Environment
==============================
Python version               : 3.12.3 (64-bit runtime)
Python platform              : Linux-6.18.40.1-microsoft-standard-WSL2-x86_64-with-glibc2.39

==============================
      Intel XPU / GPU Info
==============================
Is XPU available             : True
XPU runtime version          : 20260000
Intel GPU models             : GPU 0: Intel(R) Graphics [0xe223]   # Arc Pro B70 (BMG G31)

--Runtime--
Intel Graphics Compiler (IGC): 2.41.5
Intel GMM (libigdgmm)        : 22.10.1-1~24.04~ppa1
Level Zero loader version    : 1.32.0
Level Zero driver version    : 26.35.39758.10-0
vLLM XPU kernels version     : 0.1.14.1

==============================
         vLLM Info
==============================
vLLM Version                 : 0.29.1rc1.dev422+gd05da62e9 (git sha: d05da62e9)
vLLM Build Flags:
  CUDA Archs: Not Set; ROCm: Disabled; XPU: Enabled

==============================
     Environment Variables
==============================
VLLM_USE_V2_MODEL_RUNNER=0        # set to 0 only as the workaround; default (V2) reproduces the hang
VLLM_XPU_ENABLE_XPU_GRAPH=1
VLLM_USE_BREAKABLE_CUDAGRAPH=1
VLLM_TRITON_USE_TD=1
SYCL_UR_USE_LEVEL_ZERO_V2=0
SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0
ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
ZE_AFFINITY_MASK=0
CCL_ATL_TRANSPORT=ofi
CCL_ENABLE_SYCL_KERNELS=0
TORCH_LLM_ALLREDUCE=1
PYTORCH_ALLOC_CONF=expandable_segments:True
LD_LIBRARY_PATH=/usr/local/lib:/opt/venv/lib:/usr/lib/wsl/lib:/opt/ucx/lib:/tmp/ucx_install/lib
```

</details>

### 🐛 Describe the bug

**Summary.** On Intel XPU, the V1 engine busy-loop hangs forever inside
`GDNMetadataBuilder.build()` while building GDN (linear-attention) attention
metadata. `py-spy` shows the thread spinning in `urEventWait` inside the Level
Zero user-mode driver, with the GPU completely idle. The HTTP API keeps
answering (`/health` 200, `/metrics` 200) and the container still reports `Up`,
so it presents as "the server is alive but every request times out".
`VLLM_USE_V2_MODEL_RUNNER=0` (Model Runner V1) does not reproduce it.

Note: the model is a **hybrid** architecture (`Qwen3_5ForConditionalGeneration`,
64 layers = 48 linear/GDN + 16 full attention) served with **MTP speculative
decoding** and `--enable-prefix-caching` (which forces
`mamba_cache_mode="align"`).

**Environment / how it is run**

- Intel Arc Pro B70 (BMG G31, `0xe223`), Windows 11 host + WSL2 + Docker Desktop
  (the container gets `/dev/dxg`; there is no `/dev/dri`).
- Windows graphics driver `32.0.101.9030` (2026-09-16).
- Image: locally built `FROM vllm/vllm-openai-xpu:nightly`, i.e.
  vLLM `0.29.1rc1.dev422+gd05da62e9`, `vllm-xpu-kernels 0.1.14.1`,
  torch `2.13.0+xpu`.
- Container UMD upgraded to compute-runtime `26.35.39758.10` + IGC `2.41.5`
  (`libze_intel_gpu.so.1.17.39758`). **The hang also reproduces on the previous
  UMD `26.27.39122.11` (`libze_intel_gpu.so.1.15.39122`), so this is not fixed
  by the newest compute-runtime.**
- Model: a Qwen3.8-27B-family checkpoint, GPTQ-INT4 (sym G128) with the MTP
  heads preserved in BF16. `--dtype float16 --quantization gptq`.
- Serve flags: `--max-model-len 200000 --gpu-memory-utilization 0.88
  --kv-cache-dtype fp8 --kv-cache-memory-bytes 8053063680 --max-num-seqs 1
  --max-num-batched-tokens 8192 --enable-prefix-caching --language-model-only
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
  plus `--enable-auto-tool-choice --tool-call-parser qwen3_xml`.

**Reproduction.** Send 7 *distinct* short prompts in sequence to
`/v1/completions` (`temperature 0`, `max_tokens 120`, `ignore_eos`). It is
independent of the prompt contents: I also sent the same prompt 6 times in a row
without a hang, and the hang always lands on the 6th distinct request.

```python
import json, urllib.request
BASE = "http://127.0.0.1:8000"
PROMPTS = [  # reason, code, json, zh, list, long, instruct
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the "
    "ball. How much does the ball cost? Show the reasoning, then give the final "
    "answer on its own line.",
    "Write a Python function `sum_of_squares(n)` ...",
    "Return a JSON object with keys name, age, city for a fictional person. "
    "Output only the JSON.",
    "用三句话解释什么是投机解码（speculative decoding），并说明它为什么能加速。",
    "List the first 12 prime numbers, separated by commas.",
    "The quick brown fox jumps over the lazy dog. " * 40 +
        "\n\nHow many times does the word 'fox' appear above? Answer with a number.",
    "Summarize the following in exactly two sentences: ...",
]
for p in PROMPTS:
    body = {"model": "<served-model-name>", "prompt": p, "max_tokens": 120,
            "temperature": 0.0, "ignore_eos": True, "top_k": 0, "top_p": 1.0}
    req = urllib.request.Request(BASE + "/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=60)   # hangs on the 6th prompt (V2 runner)
```

**Observed.** The 6th request never returns. The engine stops emitting its
periodic `loggers.py` line entirely, and:

```
$ curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health   # 200
$ curl -s http://127.0.0.1:8000/metrics | grep num_requests_running
vllm:num_requests_running{...} 1.0
$ docker stats --no-stream --format '{{.CPUPerc}}' <container>            # ~100%
```

`docker exec ... top` shows the EngineCore process at ~90-100 % CPU, i.e. it is
spinning, not blocked.

**Root cause evidence — `py-spy dump --native` on the wedged EngineCore**

```
Thread 422 (active): "MainThread"
    sched_yield (libc.so.6)
    0x733e5b36f2f7 (libze_intel_gpu.so.1.17.39758)
    0x733e5b36dd40 (libze_intel_gpu.so.1.17.39758)
    ur::level_zero::urEventWait (libur_adapter_level_zero.so.0)
    build (vllm/v1/attention/backends/gdn_attn.py:307)
    build_attn_metadata (vllm/v1/worker/gpu/attn_utils.py:496)
    prepare_attn (vllm/v1/worker/gpu/model_states/mamba_hybrid.py:332)
    execute_model (vllm/v1/worker/gpu/model_runner.py:1816)
    decorate_context (torch/utils/_contextlib.py:124)
    execute_model (vllm/v1/worker/gpu_worker.py:1216)
    execute_model (vllm/v1/worker/worker_base.py:373)
    run_method (vllm/v1/serial_utils.py:508)
    collective_rpc (vllm/v1/executor/uniproc_executor.py:109)
    execute_model (vllm/v1/executor/uniproc_executor.py:122)
    step_with_batch_queue (vllm/v1/engine/core.py:703)
    _process_engine_step (vllm/v1/engine/core.py:1529)
    run_busy_loop (vllm/v1/engine/core.py:1478)
```

Three samples taken seconds apart show the identical frame, and
`v1/attention/backends/gdn_attn.py` contains no `while`/`for` loop, so this is
not a Python-level loop — the thread is inside the Level Zero event wait. The
Python line at the bottom of the C stack is:

```python
# vllm/v1/attention/backends/gdn_attn.py:305-308 (pure-spec branch)
# Filter by spec_sequence_masks to exclude padded sequences
spec_state_indices_tensor = block_table_tensor[
    spec_sequence_masks_cpu, : self.num_spec + 1
]
```

**It is not the GPU being stuck, and not the driver state**

- While wedged, a *new* process in the same container completes an XPU op fine:
  `torch.ones(64, device="xpu").sum().cpu()` → `64.0` in 0.90 s.
- Windows GPU engine counters (`\GPU Engine(*engtype_Compute)\Utilization
  Percentage`) stay **below 0.5 %** during the wedge, so no kernel is executing.
- The EngineCore process still responds to signals (a `SIGABRT` terminates it),
  so it is a userspace spin, not a kernel-level deadlock.

**V1 vs V2 control experiment**

| model runner | 7-distinct-prompt sequence |
|---|---|
| V2 (vLLM default) | **hung in 3 of 4 runs**, always on request 6 |
| V1 (`VLLM_USE_V2_MODEL_RUNNER=0`) | **21/21 requests passed (3 consecutive rounds)** |

Decode throughput is unaffected (`tok/s`, same harness):

| context | 8 k | 16 k | 32 k | 64 k | 100 k |
|---|---:|---:|---:|---:|---:|
| V2 | 65.3 | 55.8 | 57.6 | 46.8 | 45.7 |
| V1 | 61.5 | 54.4 | 56.7 | 43.1 | 50.9 |

So the trigger is in the **V2 model runner's GDN metadata path**
(`mamba_hybrid.py:332 prepare_attn` -> `attn_utils.py:496 build_attn_metadata`
-> `gdn_attn.py build()`), and on XPU it ends in a Level Zero event wait that
never completes.

**Related issues**

- intel/llm-scaler#668 — "Windows Arc B580/G21: native CUTE attention forward
  hangs after PR #659". Same symptom class: on Windows/Arc, a GPU op "hangs in a
  CPU busy wait" while the equivalent PyTorch SDPA path completes. Different GPU
  (B580/G21) and different op, so this report is not a duplicate.
- vllm#44185 — "vLLM hangs during speculative decoding with MoE draft model near
  max_model_len" (different trigger: MoE draft / max_model_len boundary).
- vllm#40756 — "MTP speculative decoding crash with illegal memory access on long
  sequences" (a crash, not a hang).

I searched for `urEventWait` in vllm-project/vllm (0 results),
vllm-project/vllm-xpu-kernels and intel/compute-runtime; and for `"busy wait"` in
intel/llm-scaler (only #668). I could not find an existing report of this
signature.

**Workaround.** `VLLM_USE_V2_MODEL_RUNNER=0` avoids it entirely (V1 has the
accepted-token race that V2 removes by design, so I also apply the open PR
#53919 patch on V1).
