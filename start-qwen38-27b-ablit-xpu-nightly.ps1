<#
.SYNOPSIS
  Run the vLLM 0.29.1-nightly stack explicitly.

  This is now the DEFAULT of the plain launcher
  (./start-qwen38-27b-ablit-xpu-int4.ps1), so this wrapper is just an explicit,
  self-documenting way to ask for it. The stable fallback is
  ./start-qwen38-27b-ablit-xpu-stable.ps1.

.REQUIREMENTS
  Intel Arc Windows driver 32.0.101.9030 (2026-09-16) or newer. On older drivers
  the MTP path hangs on any prefill above ~130 tokens.

.KNOWN ISSUE (why this is not the default)
  The nightly wedges intermittently: 3 of 4 runs of a 7-distinct-prompt sequence
  hung on the 6th request (EngineCore at 100 % CPU, "Running: 1" with zero
  throughput, /health still 200, only a restart recovers), and there were
  intermittent boot segfaults (Exited 139). See the README section
  "Upgrading to vLLM 0.29.1 nightly (experimental)".

  If the engine stops making progress (or you see "Running: 1" with
  "Avg generation throughput: 0.0"), just re-run this script: it recreates the
  container.

.EXAMPLE
  pwsh -File ./start-qwen38-27b-ablit-xpu-nightly.ps1
#>

$ErrorActionPreference = "Continue"

# The nightly needs the container's own IGC ahead of the WSL driver's copy
# (otherwise oneDNN's W4A16 GEMM reports CL_COMPILER_NOT_AVAILABLE).
$env:B70_IMAGE = "zrlu/qwen38-27b-arc-pro-b70:0.29.1-nightly"
$env:B70_LD_LIBRARY_PATH = "/usr/local/lib:/opt/venv/lib:/usr/lib/wsl/lib:/opt/ucx/lib:/tmp/ucx_install/lib"
# B70_PATCH_SET=none is baked into that image's ENV; B70_DRAFT_LMHEAD_INT4
# defaults to 1 in the main launcher, so the draft-INT4 overlay stays on.

& (Join-Path $PSScriptRoot "start-qwen38-27b-ablit-xpu-int4.ps1") @args
