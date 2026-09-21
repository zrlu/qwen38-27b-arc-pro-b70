<#
.SYNOPSIS
  Launch the B70 (Arc Pro) vLLM container with the legacy GPTQ INT4 model.

  Fixed preset (no parameters needed):
    - image        : zrlu/qwen38-27b-arc-pro-b70:0.28.0-apcfix
                     (vLLM 0.28.0 XPU + kernels 0.1.12.3 + hybrid MTP/
                     prefix-cache correctness fixes; see
                     docker/opt-qwen38/README-corrections.md)
                     Experimental, faster but intermittently wedges:
                     :0.29.1-nightly (see README "Upgrading to vLLM 0.29.1")
    - model        : C:\LocalLLM\qwen38-27b-ablit-xpu\model (GPTQ INT4, fp16)
    - maxModelLen  : 200000 (server ceiling; costs no VRAM - the client window
                     is pi's contextWindow = 150000, see the block below)
    - MTP          : 3 (native MTP spec decode, BF16 draft)
    - draft INT4   : ON (B70_DRAFT_LMHEAD_INT4=1) - INT4 copy of the draft's
                     LM head only; +20-55% decode, output bit-identical
    - KV cache     : manual 7.5 GiB pool = 205,714 tokens (150k session +
                     ~55k prefix-cache slack). Do not shrink it.
    - graph        : ENFORCE_EAGER=0 (default; GPU graph + breakable cudagraph
                     is on for throughput, see README "Breakable CUDA graph")
    - served name  : huihui-qwen38-27b-abliterated-int4

  Measured (B70, MTP3 + breakable graph, single-threaded):
    decode 128/256         ~32-38 token/s
    long-ctx decode 8K-16K ~13 token/s
    prefill 512/2048       ~470 / ~1170 token/s

  RECOMMENDED (headless): run this WITHOUT a display attached.
  The Arc B70 shares VRAM with the desktop; a desktop monitor makes the
  display compositor hold framebuffer memory and can push the 262K KV
  budget (18.2 wt + 8.7 KV + act ~= 28.9 GiB of 31.16) over the edge.
  Headless (or an idle/blank desktop) leaves the ~1.5 GiB headroom that
  keeps 240K stable with ~2.2 GiB headroom. Keep ENABLE_TOOLS/MTP as preset below.

.EXAMPLE
  powershell -File start-qwen38-27b-ablit-xpu-int4.ps1
#>

$ErrorActionPreference = "Continue"
$containerName = "qwen38-27b-ablit-xpu"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

# ---- fixed int4 preset ------------------------------------------------
$image = "zrlu/qwen38-27b-arc-pro-b70:0.28.0-apcfix"
$modelPath = Join-Path $repoRoot "model"
$modelName = "huihui-qwen38-27b-abliterated-int4"

# A/B overrides for experiments. Unset -> the baked-in production default.
function EnvInt($name, $default) {
  $v = [Environment]::GetEnvironmentVariable($name)
  if ($v) { [int]$v } else { $default }
}
function EnvStr($name, $default) {
  $v = [Environment]::GetEnvironmentVariable($name)
  if ($v) { "$v" } else { $default }
}

# ---- context sizing ---------------------------------------------------
# Two different numbers, on purpose:
#
#   MAX_MODEL_LEN = 200000   server-side ceiling. Costs no VRAM (the KV pool is
#                            sized by KV_CACHE_MEMORY_BYTES, not by this), so it
#                            stays generous.
#   pi contextWindow = 150000 the working window (pi-agent/models.json). pi
#                            compacts at contextWindow - reserveTokens
#                            (150000 - 32768 = ~117k prompt), so a session
#                            never eats the whole cache.
#
# Why that matters: the KV pool holds 205,714 tokens. Whatever a session is
# long must stay resident, and the leftover is what the prefix cache can keep
# for the NEXT turn. A 200k session leaves ~0 -> the hit boundary slides,
# every turn re-prefills and TTFT goes to tens of seconds. At a 150k window
# the pool keeps ~55k of slack, and warm turns stay at 91-96% cache hits with
# ~10-13 s TTFT. Decode speed itself is flat vs context (32-49 tok/s from 8k
# to 100k), so this is a cache/latency knob, not a throughput knob.
#
# Do NOT shrink KV_MEM_BYTES to match a smaller context: the slack IS the
# feature. 7.5 GiB = 205,714 tokens = session (150k) + slack (~55k).
$maxModelLen = EnvInt "B70_MAX_MODEL_LEN" 200000
$mtpTokens = EnvInt "B70_MTP_TOKENS" 3
$draftInt4 = EnvInt "B70_DRAFT_INT4" 0   # BF16 MTP draft (no INT4 draft quant)
# Phase S of the cookbook's draft-INT4 overlay: a private INT4 copy of the
# draft's LM head (target's fp16 lm_head is untouched). ON by default:
# +20-55% decode, 7/7 bit-identical greedy outputs in benchmarks/quality_ab.py.
$draftLmheadInt4 = EnvInt "B70_DRAFT_LMHEAD_INT4" 1
$maxImages = EnvInt "B70_MM_IMAGES" 16
$prefixCache = EnvInt "B70_PREFIX_CACHE" 1
$enforceEager = EnvInt "B70_ENFORCE_EAGER" 0
$kvCacheDtype = EnvStr "B70_KV_CACHE_DTYPE" "fp8"
$kvMemBytes = EnvInt "B70_KV_MEM_BYTES" 8053063680   # 7.5 GiB pool = 205,714 tokens: 150k session + ~55k prefix-cache slack
$maxNumBatched = EnvInt "B70_MAX_NUM_BATCHED" 8192
$maxNumSeqs = EnvInt "B70_MAX_NUM_SEQS" 1
$gpuMemUtil = EnvStr "B70_GPU_MEM_UTIL" "0.88"
# 1 / 0 / unset: force vLLM's V2 / V1 model runner, or leave vLLM's default.
# On WSL2 the 0.29.0 V2 runner makes the oneDNN W4A16 GEMM JIT and the WSL
# OpenCL driver has no compiler -> "could not create a primitive". Harmless
# (unset) on 0.28.0, which is the shipped default.
$v2Runner = EnvStr "B70_V2_RUNNER" ""
$image = EnvStr "B70_IMAGE" $image
# Correctness fixes for hybrid MTP + align-mode prefix caching.
# accept-sync + backward-copy are required. eagle-drop is upstream-correct but
# on 0.28.0 it makes a hit land on a state boundary the scheduler never
# materialized -> NaN logits (token 0, "!"). Upstream 0.29.0 (#53945) moves the
# materialization down to match; until we upgrade, keep it off. Verified: a
# 40-turn / 121k-token agentic soak with a context needle is clean.
$fixAcceptSync = EnvInt "B70_FIX_ACCEPT_SYNC" 1
$fixBackwardCopy = EnvInt "B70_FIX_BACKWARD_COPY" 1
$fixEagleDrop = EnvInt "B70_FIX_EAGLE_DROP" 0
# 0.28-era runtime patch stack: full | minimal | none. Empty = leave it to the
# image's own ENV (0.29.1-nightly sets none; the 0.28 image has no default, so
# start.sh falls back to full).
$patchSet = EnvStr "B70_PATCH_SET" ""
if ($patchSet -ne "") { $extraEnv += @("-e", "B70_PATCH_SET=$patchSet") }

Write-Host "[start] INT4 preset: $modelPath"
Write-Host "[start] maxModelLen=$maxModelLen MTP=$mtpTokens KV=$kvMemBytes eager=$enforceEager"

# Create placeholder file (WSL interop shims)
$placeholderFile = Join-Path $env:TEMP "placeholder-empty"
New-Item -Path $placeholderFile -ItemType File -Force | Out-Null

# Optional extra -e args
$extraEnv = @()
if ($v2Runner -ne "") { $extraEnv += @("-e", "VLLM_USE_V2_MODEL_RUNNER=$v2Runner") }
# Restrict the SYCL runtime's exposed backends. Setting e.g. "level_zero:*" hides
# the OpenCL backend, which is what makes oneDNN try to JIT an OCL primitive and
# fail on WSL2 (CL_COMPILER_NOT_AVAILABLE -> "could not create a primitive").
$oneapiSelector = EnvStr "B70_ONEAPI_SELECTOR" ""
if ($oneapiSelector -ne "") { $extraEnv += @("-e", "ONEAPI_DEVICE_SELECTOR=$oneapiSelector") }
# Library search order. The default puts the WSL driver first (needed for
# libcuda/libdxcore/libwsl_compute_helper), but that also shadows the
# container's own IGC (/usr/local/lib, 2.38.2) with the Windows driver's
# libigdfcl.so.2. Override to test IGC version coupling.
$ldPath = EnvStr "B70_LD_LIBRARY_PATH" "/usr/lib/wsl/lib:/opt/ucx/lib:/tmp/ucx_install/lib:/opt/venv/lib:/usr/local/lib"
# NOTE: to run the experimental 0.29.1-nightly image instead, set
#   B70_IMAGE=zrlu/qwen38-27b-arc-pro-b70:0.29.1-nightly
#   B70_LD_LIBRARY_PATH=/usr/local/lib:/opt/venv/lib:/usr/lib/wsl/lib:/opt/ucx/lib:/tmp/ucx_install/lib
# (the nightly needs the container's own IGC first, and B70_PATCH_SET=none is
# baked into that image's ENV).

# Host start.sh
$startSh = Join-Path $repoRoot "docker\opt-qwen38\start.sh"

# Always clean up old container first
Write-Host "[start] Cleaning up old container..."
docker rm -f $containerName 2>$null | Out-Null

Write-Host "[start] Launching $containerName"
docker run -d --name $containerName `
  --device /dev/dxg `
  --shm-size 16g `
  -p 127.0.0.1:8000:8000 `
  -v /usr/lib/wsl/lib:/usr/lib/wsl/lib:ro `
  -v ${placeholderFile}:/usr/lib/wsl/lib/libnvidia-ml.so.1:ro `
  -v ${placeholderFile}:/usr/lib/wsl/lib/libcuda.so.1:ro `
  -v /usr/lib/wsl/drivers:/usr/lib/wsl/drivers:ro `
  -v "${startSh}:/opt/qwen38/start.sh:ro" `
  --mount type=bind,source=${modelPath},target=/model `
  -e MODEL_NAME=$modelName `
  -e LD_LIBRARY_PATH=$ldPath `
  -e VLLM_TARGET_DEVICE=xpu `
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE `
  -e ZE_AFFINITY_MASK=0 `
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 `
  -e CCL_ATL_TRANSPORT=ofi `
  -e CCL_ENABLE_SYCL_KERNELS=0 `
  -e CCL_TOPO_P2P_ACCESS=0 `
  -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 `
  -e CCL_ZE_CACHE_OPEN_IPC_HANDLES=0 `
  -e SYCL_UR_USE_LEVEL_ZERO_V2=0 `
  -e SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 `
  -e TORCH_LLM_ALLREDUCE=1 `
  -e MTP_TOKENS=$mtpTokens `
  -e DRAFT_INT4=$draftInt4 `
  -e B70_DRAFT_LMHEAD_INT4=$draftLmheadInt4 `
  -e MAX_MODEL_LEN=$maxModelLen `
  -e KV_CACHE_DTYPE=$kvCacheDtype `
  -e PREFIX_CACHE=$prefixCache `
  -e ENFORCE_EAGER=$enforceEager `
  -e MAX_NUM_SEQS=$maxNumSeqs `
  -e GPU_MEMORY_UTILIZATION=$gpuMemUtil `
  -e KV_CACHE_MEMORY_BYTES=$kvMemBytes `
  -e MAX_NUM_BATCHED_TOKENS=$maxNumBatched `
  -e MM_IMAGES=$maxImages `
  $extraEnv `
  -e B70_FIX_ACCEPT_SYNC=$fixAcceptSync `
  -e B70_FIX_BACKWARD_COPY=$fixBackwardCopy `
  -e B70_FIX_EAGLE_DROP=$fixEagleDrop `
  $image

if ($LASTEXITCODE -ne 0) {
  Write-Host "[start] docker run FAILED (rc=$LASTEXITCODE)"
  exit 1
}

# --- Wait for readiness ---
# curl.exe (Win10+) bypasses any system proxy env var that makes
# Invoke-WebRequest silently fail against 127.0.0.1.
Write-Host "[start] Waiting for vLLM on :8000..."
$maxAttempts = 240
for ($i = 1; $i -le $maxAttempts; $i++) {
  & curl.exe --silent --max-time 3 --noproxy '*' -o NUL "http://127.0.0.1:8000/health" 2>$null
  if ($LASTEXITCODE -eq 0) {
    Write-Host "[start] Up after $($i*5)s."
    exit 0
  }
  # bail out early if the container died (fail-closed patch / OOM)
  $running = (docker inspect -f '{{.State.Running}}' $containerName 2>$null)
  if ($running -ne "true") {
    Write-Host "[start] Container exited during boot - inspect: docker logs $containerName"
    exit 1
  }
  if ($i -eq $maxAttempts) {
    Write-Host "[start] Still not ready after ~20 min - check: docker logs $containerName"
    exit 1
  }
  Start-Sleep -Seconds 5
}