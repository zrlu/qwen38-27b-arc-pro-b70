<#
.SYNOPSIS
  Run the STABLE vLLM 0.28.0 + draft-INT4 stack instead of the default nightly.

  Use this when you want the generation that has never wedged in testing. It is
  ~10-20 % slower than the nightly and it does not need the 2026-09-16 driver,
  but it can still hit the "draft acceptance collapses to 0 %" bug after a large
  fresh prefill (decode drops to ~15 tok/s; restart to recover).

  Switch back with the plain launcher:

      ./start-qwen38-27b-ablit-xpu-int4.ps1

.EXAMPLE
  pwsh -File ./start-qwen38-27b-ablit-xpu-stable.ps1
#>

$ErrorActionPreference = "Continue"

# 0.28 was validated with the WSL driver's libs first (the opposite of what the
# nightly needs -- see docker/Dockerfile.nightly).
$env:B70_IMAGE = "zrlu/qwen38-27b-arc-pro-b70:0.28.0-apcfix"
$env:B70_LD_LIBRARY_PATH = "/usr/lib/wsl/lib:/opt/ucx/lib:/tmp/ucx_install/lib:/opt/venv/lib:/usr/local/lib"
# That image has no B70_PATCH_SET in its ENV, so start.sh falls back to `full`
# (the 0.28-era patch stack). The draft-INT4 overlay stays on.

& (Join-Path $PSScriptRoot "start-qwen38-27b-ablit-xpu-int4.ps1") @args
