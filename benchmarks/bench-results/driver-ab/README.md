# Driver A/B — B70 INT4 stack (before/after a Windows graphics-driver change)

Baseline recorded on the **Pro** driver. Purpose: measure whether moving to the
**consumer** Arc driver changes the vLLM XPU INT4 throughput, independently of
the Vulkan/FWHT reason for the upgrade.

## Recorded "before" state

| item | value |
|---|---|
| Intel driver | `32.0.101.8805` (Pro line, 2026-07-06) |
| Vulkan `driverVersion` | `101.8805` — **inside** the PrismML fork's FWHT-disabled range `[101.8509, 101.8860)` |
| Image | `zrlu/qwen38-27b-arc-pro-b70:0.28.0-apcfix` @ `sha256:2483c8f1…fd3e5` |
| Model | `huihui-qwen38-27b-abliterated-int4` |
| MTP | 3, `DRAFT_INT4=0`, fp8 KV, `MAX_NUM_SEQS=1`, prefix cache ON |
| Power limit | **230 W** (`xpu-smi --query-gpu=power.limit`) |
| Idle temps | core 52-58 °C, mem 66-76 °C, fan 22-28 % |

Files: `before-env.json`, `before-8805-ctx-20260920.json`.

## Before (2026-09-20, MTP3, accept 62-71 %)

| ctx | tok/s | ttft | prefill t/s |
|---:|---:|---:|---:|
| 8037 | 23.5 | 5.6 s | 1445 |
| 16005 | 22.8 | 7.3 s | 2185 |
| 32255 | 24.5 | 21.9 s | 1473 |
| 64258 | 21.3 | 39.8 s | 1613 |
| 96181 | 20.9 | 53.0 s | 1814 |

This sits in the same band as the repo's own recorded MTP3 run
(`../b70-vllm0.28.0-mtp3-20260830.json`: 16k 17.5 / 32k 14.1 / 48k 14.3 on the
`bench_suite` curve harness), so the host has not regressed. The README's
35/36/49/37/32 curve is optimistic relative to both.

## After the driver update — keep identical

1. Power limit back to **230 W**: `xpu-smi config -d 0 --powerlimit 230`
   (a driver install can reset it — verify, do not assume).
2. Same container, do **not** rebuild or restart into a different image.
3. Same harness, same CWD:

```powershell
Set-Location 'C:\LocalLLM\qwen38-27b-ablit-xpu\benchmarks\bench-results\driver-ab'
python ..\bench_context.py ctx
Copy-Item bench_ctx.json after-9030-ctx-<date>.json
```

4. Confirm the new Vulkan number before trusting anything:

```powershell
vulkaninfo.exe --summary | Select-String driverVersion,deviceName
# want Intel's driverVersion >= 101.8860 (9030), i.e. outside the gated range
```

5. Diff:

```powershell
python compare.py before-8805-ctx-20260920.json after-9030-ctx-<date>.json
```

## Reading the result

- `tok/s` moves only if the kernels or the IGC codegen moved. Decode is
  DRAM-bound; expect **±0-5 %**, and treat anything inside that as noise unless
  it reproduces across depths *and* acceptance is unchanged.
- If `accept` shifts by more than a few points, the run is not comparable —
  the corpus/prompt mix differs, not the driver.
- Prefill is compute-bound and is the metric most likely to move from a
  compiler change. Watch it separately.

## Rollback

Keep the Pro `32.0.101.8805` installer. If a regression appears, also try
decoupling the driver's IGC from the container:

```powershell
$env:B70_LD_LIBRARY_PATH = '/opt/ucx/lib:/tmp/ucx_install/lib:/opt/venv/lib:/usr/local/lib:/usr/lib/wsl/lib'
```

(the default puts `/usr/lib/wsl/lib` first, which shadows the container's own
IGC 2.38.2 with the driver's `libigdfcl.so.2` — see the comment in
`start-qwen38-27b-ablit-xpu-int4.ps1`).
