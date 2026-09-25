# Upstream issue draft — XPU graph replay accumulates cost per step

Status: **draft, not filed.** Sanitised: no usernames, no host paths, no
checkpoint/HF ids, no registry org. Review before posting.

Candidate homes, most specific first:

1. `pytorch/pytorch` — label `module: xpu`, the accumulation is inside the XPU
   graph replay (`torch/xpu/graphs.py` -> `c10::xpu::XPUGraph::replay` ->
   UR/L0). This is the code path that both leaks and is slow.
2. `intel/llm-scaler` / vLLM-XPU — if the above is judged "not torch".
3. `vllm-project/vllm` — only as a cross-reference; vLLM's part is choosing the
   graph, and `VLLM_XPU_ENABLE_XPU_GRAPH=0` is already a clean workaround.

---

**Title:** XPU graph replay accumulates ~0.7 us per replay — decode step time
grows linearly with the number of steps executed and only a process restart
clears it

**Summary.** Decoding on XPU with `VLLM_XPU_ENABLE_XPU_GRAPH=1`, the per-step
host time grows linearly with the number of decode steps the process has
already executed. Identical work (same prompt, same sampler, same batch size)
gets monotonically slower; a fresh process is fast again. Nothing recovers it
in-process: no idle time, no cache reset, no session change.

**Environment.**

- 1x Intel Arc Pro B70 (Battlemage), WSL2 + Docker, Windows 11 host,
  UMD `libze_intel_gpu.so.1.17.39758` (26.35), IGC 2.41.5, host driver 32.0.101.9030
- vLLM 0.30.0 (XPU), PyTorch XPU, kernels 0.1.15.4
- Model: 27B hybrid (48 gated-delta-net linear layers + 16 full attention),
  GPTQ-INT4 weights, FP8 KV cache, MTP speculative decoding (3 draft tokens)
- `--enable-prefix-caching`, `--max-num-seqs 1`, `--max-model-len 200000`,
  205k-token KV pool, 7.5 GiB KV budget
- `VLLM_USE_BREAKABLE_CUDAGRAPH=0`, V1 model runner

**Reproducer (minutes, not hours).** Send one long forced generation
(`ignore_eos=true`, `temperature=0`, `stream=true`), record the interval between
streamed chunks (= one decode step each), then repeat the *same* request 3–6
times without restarting the server. Script: `benchmarks/step_drift.py`
(included in the repo this draft comes from); the essential part is

```python
# per request: POST /v1/completions {prompt, max_tokens, ignore_eos: true,
#                                    temperature: 0.0, stream: true}
# record t[i] for each streamed chunk; deltas = step times
```

**Observed.**

| | first step of a request | within one request | change per request (1650 steps) |
|---|---:|---|---:|
| `VLLM_XPU_ENABLE_XPU_GRAPH=1` | 51.3 ms | +7–8% (quintiles 50.8 → 54.6) | +3.3 … +4.7 ms |
| `VLLM_XPU_ENABLE_XPU_GRAPH=0` | 67.9 ms | −2 … +2% | flat (~0.14 µs/step) |

Six identical back-to-back requests, graph on:

```
first-step intervals: 51.3 54.6 59.1 62.6 65.6 68.6 ms
```

i.e. the same work is 33% slower after ~10 k steps, and the rate is steady:
**~2.8 µs per decode step**. Over a 9-hour session of ordinary use the same
effect moved the step time from 48 ms to 93 ms.

With `VLLM_USE_BREAKABLE_CUDAGRAPH=1` (which splits each forward into more
replay segments) the accumulation rate is ~1.8x higher for the same token
throughput, which is why the rate is better expressed **per graph replay:
~0.7 µs per replay**.

**Where the time goes.** `py-spy record --native` during a degraded step
(percent of samples): `libze_intel_gpu.so` internals 25%, `resetCommandLists`
(`libur_adapter_level_zero.so`) 9.2%, `ioctl` 7.4%, `sched_yield` 5.0%; the
hottest Python frame is `replay (torch/xpu/graphs.py:107)`, reached from the
speculative-decoding draft path. The GPU itself is idle-ish (26–28 W of a 230 W
budget, core clock at its 2800 MHz maximum), and `xpu-smi` memory bandwidth is
near zero during those windows.

**Not explained by:**

- memory: host RSS, fd count, container RSS and device memory stay flat across
  the degradation (no growth);
- KV / prefix cache state: resetting the session (new prompt, new blocks) does
  not restore the speed, and prefix-cache hit rate is unaffected;
- power/thermal: clock pinned at max, power far below the cap;
- workload: the work is byte-identical between the slow and the fast
  measurement.

**Impact.** Long-running serving decays by ~2x per 30 k decode steps; at ~100 k
steps of ordinary chat the observed generation rate falls to single-digit
tok/s. The only recovery is restarting the engine process. Disabling the XPU
graph removes the decay entirely at the cost of a ~34% higher baseline step
time in low-acceptance workloads (equal end-to-end throughput in a
prefix-cache-heavy agent workload), so we ship with the graph off — but a graph
path that gets permanently slower the more you use it looks like a real bug
rather than a tuning knob.
