<#
.SYNOPSIS
  Launch the B70 (Arc Pro) vLLM container with the legacy GPTQ INT4 model.

  Fixed preset (no parameters needed):
    - model        : C:\LocalLLM\qwen38-27b-ablit-xpu\model (GPTQ INT4, fp16)
    - maxModelLen  : 240000 (keeps ~0.75 GiB extra VRAM headroom vs 262K)
    - MTP          : 3 (native MTP spec decode, INT4 draft)
    - KV cache     : manual 9.0 GiB pool for 240K (MTP3, fp8)
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
$modelPath = Join-Path $repoRoot "model"
$modelName = "huihui-qwen38-27b-abliterated-int4"
$maxModelLen = 240000
$mtpTokens = 3
$draftInt4 = 0            # match image default: MTP draft stays BF16 (NO INT4 draft quant)
$maxImages = 16
$prefixCache = 1
$enforceEager = 0
$kvMemBytes = 9663676416     # 9.0 GiB KV pool for 240K ctx (MTP3)

Write-Host "[start] INT4 preset: $modelPath"
Write-Host "[start] maxModelLen=$maxModelLen MTP=$mtpTokens KV=$kvMemBytes eager=$enforceEager"

# Create placeholder file (WSL interop shims)
$placeholderFile = Join-Path $env:TEMP "placeholder-empty"
New-Item -Path $placeholderFile -ItemType File -Force | Out-Null

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
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib:/tmp/ucx_install/lib:/opt/venv/lib:/usr/local/lib `
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
  -e MAX_MODEL_LEN=$maxModelLen `
  -e KV_CACHE_DTYPE=fp8 `
  -e PREFIX_CACHE=$prefixCache `
  -e ENFORCE_EAGER=$enforceEager `
  -e MAX_NUM_SEQS=1 `
  -e GPU_MEMORY_UTILIZATION=0.88 `
  -e KV_CACHE_MEMORY_BYTES=$kvMemBytes `
  -e MAX_NUM_BATCHED_TOKENS=8192 `
  -e MM_IMAGES=$maxImages `
  zrlu/qwen38-27b-arc-pro-b70:latest

if ($LASTEXITCODE -ne 0) {
  Write-Host "[start] docker run FAILED (rc=$LASTEXITCODE)"
  exit 1
}

# --- Wait for readiness ---
Write-Host "[start] Waiting for vLLM on :8000..."
$maxAttempts = 150
for ($i = 1; $i -le $maxAttempts; $i++) {
  try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
      Write-Host "[start] Up after $($i*5)s."
      exit 0
    }
  } catch {
    # Not ready yet
  }

  if ($i -eq $maxAttempts) {
    Write-Host "[start] Still not ready after ~12.5 min - check: docker logs $containerName"
    exit 1
  }

  Start-Sleep -Seconds 5
}