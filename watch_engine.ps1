<#
.SYNOPSIS
  Watchdog for the B70 vLLM container: detects the EngineCore wedge and restarts.

.DESCRIPTION
  The 0.29.1-nightly stack can wedge: the EngineCore stops making progress while
  the HTTP API keeps answering (so /health stays 200 and the container looks
  "Up"). It has been observed both mid-request and while idle, and it needs a
  container restart to clear.

  This script polls every $IntervalSec and decides:

    * running requests > 0 and the token counters have not moved for
      $WedgeAfterSec  -> wedged
    * no running requests -> send a 1-token probe; if it does not answer within
      $ProbeTimeoutSec (and no request started in the meantime) -> wedged

  On a wedge it logs the metrics, tries `py-spy dump` on the EngineCore (the
  container needs --cap-add SYS_PTRACE, which the launcher passes), and -- unless
  -NoAutoRestart is given -- re-runs the launcher. Two consecutive detections are
  required before restarting, so a slow prefill cannot trigger a false positive.

.PARAMETER Once
  Run a single check, print the verdict and exit. Useful for testing.

.EXAMPLE
  pwsh -File ./watch_engine.ps1                 # watch, auto-restart on wedge
  pwsh -File ./watch_engine.ps1 -NoAutoRestart  # watch and only log
  pwsh -File ./watch_engine.ps1 -Once           # one-shot health verdict
#>
param(
    [int]$IntervalSec = 30,
    [int]$WedgeAfterSec = 90,
    [int]$ProbeTimeoutSec = 20,
    [switch]$NoAutoRestart,
    [switch]$Once,
    [string]$LogFile
)

$ErrorActionPreference = "Continue"
$containerName = "qwen38-27b-ablit-xpu"
$modelName = "huihui-qwen38-27b-abliterated-int4"
$baseUrl = "http://127.0.0.1:8000"
$launcher = Join-Path $PSScriptRoot "start-qwen38-27b-ablit-xpu-int4.ps1"
if (-not $LogFile) { $LogFile = Join-Path $PSScriptRoot "engine_watchdog.log" }

function Write-Log([string]$msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

function Get-Metrics {
    $raw = (& curl.exe -s --noproxy '*' --max-time 8 "$baseUrl/metrics" 2>$null) -join "`n"
    if (-not $raw) { return $null }
    $out = @{ running = $null; progress = $null }
    $m = [regex]::Match($raw, 'vllm:num_requests_running\{[^}]*\}\s+([0-9.eE+]+)')
    if (-not $m.Success) { return $null }
    $out.running = [double]$m.Groups[1].Value
    $p = 0.0
    foreach ($pat in @(
        'vllm:prompt_tokens_total\{[^}]*\}\s+([0-9.eE+]+)',
        'vllm:generation_tokens_total\{[^}]*\}\s+([0-9.eE+]+)',
        'vllm:spec_decode_num_draft_tokens_total\{[^}]*\}\s+([0-9.eE+]+)')) {
        $mm = [regex]::Match($raw, $pat)
        if ($mm.Success) { $p += [double]$mm.Groups[1].Value }
    }
    $out.progress = $p
    return $out
}

function Test-Probe {
    # 1-token completion; returns $true if the engine answered.
    $body = '{"model":"' + $modelName + '","prompt":"ping","max_tokens":1,"temperature":0}'
    $code = (& curl.exe -s --noproxy '*' --max-time $ProbeTimeoutSec -o NUL -w "%{http_code}" `
        -X POST "$baseUrl/v1/completions" -H "Content-Type: application/json" -d $body 2>$null) -join ""
    return ($code -eq "200")
}

function Save-Stack {
    $pidText = (& docker exec $containerName bash -c "pgrep -f EngineCore | head -1" 2>$null) -join ""
    $pidText = $pidText.Trim()
    if (-not $pidText) { Write-Log "  (no EngineCore pid found)"; return }
    Write-Log "  EngineCore pid=$pidText -- dumping python stack"
    $dump = (& docker exec $containerName bash -c "/opt/venv/bin/py-spy dump --pid $pidText 2>&1" 2>$null) -join "`n"
    if ($dump) { Add-Content -Path $LogFile -Value $dump; Write-Host $dump }
    else { Write-Log "  (py-spy produced nothing)" }
}

function Invoke-Restart {
    Write-Log "  restarting via $([System.IO.Path]::GetFileName($launcher))"
    & $launcher 2>&1 | ForEach-Object { Add-Content -Path $LogFile -Value "    $_" }
}

$containerUp = (& docker inspect -f '{{.State.Running}}' $containerName 2>$null) -join ""
if ($containerUp.Trim() -ne "true") {
    Write-Log "container '$containerName' is not running (running='$($containerUp.Trim())')"
    if (-not $NoAutoRestart -and -not $Once) { Invoke-Restart }
    elseif ($Once) { Write-Host "VERDICT: container-down"; exit 2 }
}

$strikes = 0
$lastProgress = $null
$lastProgressAt = Get-Date

while ($true) {
    $m = Get-Metrics
    if ($null -eq $m) {
        $strikes++
        Write-Log "metrics endpoint unavailable (strike $strikes)"
    }
    elseif ($m.running -gt 0) {
        if ($null -ne $lastProgress -and $m.progress -eq $lastProgress) {
            $idleFor = [int]((Get-Date) - $lastProgressAt).TotalSeconds
            if ($idleFor -ge $WedgeAfterSec) {
                $strikes++
                Write-Log "no progress for ${idleFor}s with $($m.running) running request(s) (strike $strikes)"
            }
        } else {
            $strikes = 0
            $lastProgress = $m.progress
            $lastProgressAt = Get-Date
        }
    } else {
        # idle: the engine should answer a tiny probe immediately
        if (Test-Probe) {
            if ($strikes -ne 0) { Write-Log "probe OK -- clearing $strikes strike(s)" }
            $strikes = 0
            $lastProgress = $m.progress
            $lastProgressAt = Get-Date
        } else {
            $m2 = Get-Metrics
            if ($null -ne $m2 -and $m2.running -gt 0) {
                Write-Log "probe timed out but a request is now running -- treating as busy"
                $strikes = 0
            } else {
                $strikes++
                Write-Log "idle probe timed out (strike $strikes)"
            }
        }
    }

    if ($Once) {
        if ($strikes -ge 1) { Write-Host "VERDICT: wedged"; exit 1 }
        Write-Host "VERDICT: healthy"; exit 0
    }

    if ($strikes -ge 2) {
        Write-Log "WEDGE CONFIRMED after $strikes strikes -- collecting evidence"
        Save-Stack
        if ($NoAutoRestart) { Write-Log "  -NoAutoRestart: not restarting"; $strikes = 0 }
        else { Invoke-Restart; $strikes = 0; $lastProgress = $null }
    }

    Start-Sleep -Seconds $IntervalSec
}
