#!/usr/bin/env pwsh
# qwen38-ablit - vLLM (Intel Arc/XPU) launch script

$ErrorActionPreference = "Continue"
$containerName = "qwen38-27b-ablit-xpu"

# Create placeholder file
$placeholderFile = Join-Path $env:TEMP "placeholder-empty"
New-Item -Path $placeholderFile -ItemType File -Force | Out-Null

# Always clean up old container first
Write-Host "[start] Cleaning up old container..."
docker rm -f qwen38-27b-ablit-xpu 2>$null | Out-Null

$modelPath = (Get-Location).Path + "\model"

# Launch container
Write-Host "[start] Launching $containerName"
docker run -d --name $containerName `
  --device /dev/dxg `
  --shm-size 16g `
  -p 127.0.0.1:8000:8000 `
  -v /usr/lib/wsl/lib:/usr/lib/wsl/lib:ro `
  -v ${placeholderFile}:/usr/lib/wsl/lib/libnvidia-ml.so.1:ro `
  -v ${placeholderFile}:/usr/lib/wsl/lib/libcuda.so.1:ro `
  -v /usr/lib/wsl/drivers:/usr/lib/wsl/drivers:ro `
  --mount type=bind,source=${modelPath},target=/model `
  -e MODEL_NAME=huihui-qwen38-27b-abliterated-int4 `
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
  -e MTP_TOKENS=4 `
  -e MAX_MODEL_LEN=100000 `
  -e KV_CACHE_DTYPE=fp8 `
  -e DRAFT_INT4=1 `
  -e PREFIX_CACHE=1 `
  -e MAX_NUM_SEQS=1 `
  -e GPU_MEMORY_UTILIZATION=0.6 `
  -e KV_CACHE_MEMORY_BYTES=4617089843 `
  -e MAX_NUM_BATCHED_TOKENS=8192 `
  -e MM_IMAGES=16 `
  zrlu/qwen38-27b-arc-pro-b70:latest

# --- Wait for readiness ---
Write-Host "[start] Waiting for vLLM on :8000..."
$maxAttempts = 120
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
    Write-Host "[start] Still not ready after 10 min - check: docker logs $containerName"
    exit 1
  }
  
  Start-Sleep -Seconds 5
}
