<#
.SYNOPSIS
  Degradation monitor: samples vLLM step time and GPU state over time.

.DESCRIPTION
  The symptom this exists for: the engine starts at ~40 tok/s and decays to
  ~9 tok/s over tens of hours, with acceptance unchanged -- i.e. the step time
  grows. The vLLM log already reports the two numbers needed to see it:

      Avg generation throughput: Y tokens/s ... XPU KV cache usage: Z%, Prefix cache hit rate: W%
      Mean acceptance length: X

  steps/s = Y / X, step_ms = 1000 / steps/s. A rising step_ms at constant X is
  the degradation. Samples go to a CSV so the curve survives.

  Every $GpuEveryNth samples it also records xpu-smi power / GPU clock / memory
  bandwidth, to rule power or clock throttling in or out.

.EXAMPLE
  pwsh -File ./watch_degradation.ps1 -Container qwen38-jp-rp-xpu
#>
param(
    [string]$Container = "qwen38-jp-rp-xpu",
    [int]$IntervalSec = 60,
    [int]$GpuEveryNth = 5,
    [string]$Csv
)

$ErrorActionPreference = "Continue"
if (-not $Csv) { $Csv = Join-Path $PSScriptRoot ("degradation-" + $Container + ".csv") }

if (-not (Test-Path $Csv)) {
    "timestamp,thr_tok_s,accept_len,tokens_per_step,step_ms,kv_pct,hit_pct,power_w,gpu_mhz,mem_read_gbs" |
        Out-File -FilePath $Csv -Encoding utf8
}

Write-Host "watching $Container -> $Csv (every ${IntervalSec}s)"

$n = 0
$baseStepMs = $null

while ($true) {
    $n++
    $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")

    # --- engine numbers from the log -------------------------------------
    $log = docker logs --tail 60 $Container 2>&1
    $acc = $null; $thr = $null; $kv = $null; $hit = $null
    foreach ($line in $log) {
        if ($line -match "Mean acceptance length:\s*([0-9.]+)") { $acc = [double]$Matches[1] }
        if ($line -match "Avg generation throughput:\s*([0-9.]+) tokens/s.*XPU KV cache usage:\s*([0-9.]+)%.*Prefix cache hit rate:\s*([0-9.]+)%") {
            $thr = [double]$Matches[1]; $kv = [double]$Matches[2]; $hit = [double]$Matches[3]
        }
    }

    $tps = $null; $stepMs = $null
    if ($null -ne $thr -and $null -ne $acc -and $acc -gt 0 -and $thr -gt 0) {
        $tps = $acc                                 # tokens produced per step
        $stepMs = 1000.0 / ($thr / $acc)
        if ($null -eq $baseStepMs -and $stepMs -gt 0) { $baseStepMs = $stepMs }
    }

    # --- GPU numbers (occasionally) --------------------------------------
    $power = ""; $mhz = ""; $memGb = ""
    if ($n % $GpuEveryNth -eq 1) {
        $g = & xpu-smi stats -d 0 2>$null
        foreach ($line in $g) {
            if ($line -match "GPU Power \(W\)\s+Tile 0: avg:\s*([0-9]+)") { $power = $Matches[1] }
            if ($line -match "GPU Frequency \(MHz\)\s+Tile 0: avg:\s*([0-9]+)") { $mhz = $Matches[1] }
            if ($line -match "GPU Memory Read \(kB/s\)\s+Tile 0: avg:\s*([0-9]+)") { $memGb = [math]::Round([double]$Matches[1] / 1e6, 1) }
        }
    }

    $row = "$stamp,$thr,$acc,$tps,$stepMs,$kv,$hit,$power,$mhz,$memGb"
    $row | Out-File -FilePath $Csv -Encoding utf8 -Append

    $ratio = if ($null -ne $stepMs -and $null -ne $baseStepMs -and $baseStepMs -gt 0) {
        " (x{0:N2} vs first)" -f ($stepMs / $baseStepMs)
    } else { "" }
    Write-Host ("$stamp thr=$thr accept=$acc step_ms=$stepMs kv=$kv hit=$hit$ratio")

    Start-Sleep -Seconds $IntervalSec
}
