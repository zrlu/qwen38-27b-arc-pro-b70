# MTP sweep: 2 vs 3 vs 4 draft tokens — Arc Pro B70

- Model: `zrlu/Huihui-Qwen3.8-27B-abliterated-GPTQ-Int4-sym-G128-MTP-BF16-B70`
- Runtime: vLLM XPU 0.28.0, kernels 0.1.12.3, single B70, WSL2, MAX_NUM_SEQS=1, prefix cache ON
- Only `MTP_TOKENS` varied (2 / 3 / 4); `DRAFT_INT4=0` in all runs
- Harness: `bench_suite.py` (identical copy for all runs; fresh server restart + warmup before
  the MTP3/MTP4 runs). The MTP2 run targeted a server that had already been serving for ~3 h.
- Date: 2026-08-30 (Windows local)

## Throughput (tok/s)

| metric | MTP2 | MTP3 | MTP4 |
|---|---:|---:|---:|
| decode 128 (prose) | 15.6 | 27.1 | **27.2** |
| decode 256 (prose) | 19.5 | **33.5** | 32.8 |
| wave4 median | 11.2 | **20.4** | 19.6 |
| wave4 aggregate | 45.2 | **79.8** | 77.1 |
| long-ctx decode @16k | 13.12 | **17.51** | 16.79 |
| long-ctx decode @32k | 10.89 | 14.06 | **15.37** |
| long-ctx decode @48k | 11.13 | **14.33** | 11.50 |
| prefill 512 | 307.1 | 321.9 | 319.0 |
| prefill 2048 | 799.3 | 851.6 | 853.8 |
| prefill 8192 | 1365.3 | 1388.4 | 1391.0 |

## Speculative-decoding counters (vLLM /metrics)

| run | draft tokens | accepted | acceptance | window |
|---|---:|---:|---:|---|
| MTP2 | 69,234 | 50,699 | 73.2% | 3h mixed traffic |
| MTP3 | 4,032 | 2,529 | 62.7% | bench run only |
| MTP4 | 4,880 | 2,773 | 56.8% | bench run only |

Acceptance falls monotonically with chain length (expected); tok/s peaks at MTP3.

## Verdict

- **MTP3 is the sweet spot on this B70.** It leads on 256-decode, 4-way concurrency, 16k and
  48k context (clearly at 48k: 14.3 vs 11.5 — the 4th draft position's KV-attention cost
  outweighs its rarely-accepted token once context is long). MTP4 edges ahead only at 32k
  (15.4 vs 14.1); at 128-decode the two are statistically tied.
- Single-stream prose decode: 27–33 tok/s (MTP3) vs 16–20 (MTP2) ≈ **1.7×**; 4-way concurrency
  scales to ~80 tok/s aggregate on MTP3.
- The 3rd draft position is the last one that pays off; the 4th accepts too rarely (~57% chain
  survival) to cover its extra draft pass.
- Prefill is unaffected by MTP depth (≤6% spread = run noise), as expected.
- Acceptance percentages are not same-window comparable across runs (see "window" column);
  trust the tok/s deltas over the acceptance deltas.

Per-run data (markdown): `b70-vllm0.28.0-20260830.md` (MTP2),
`b70-vllm0.28.0-mtp3-20260830.md`, `b70-vllm0.28.0-mtp4-20260830.md`
(machine-readable twins: same names with `.json`).
