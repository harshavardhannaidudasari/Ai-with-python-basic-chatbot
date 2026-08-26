# Starts the chatbot server (if not already running) and opens a free
# Cloudflare Tunnel to it, printing the public URL you can share.
#
# Usage: right-click > Run with PowerShell, or from a terminal:
#   powershell -ExecutionPolicy Bypass -File scripts\go_live.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$flaskUrl = "http://127.0.0.1:5000"
$cloudflaredExe = Join-Path $root "scripts\tools\cloudflared.exe"
$tunnelLog = Join-Path $env:TEMP "cloudflared-chatbot.log"
$urlFile = Join-Path $root "scripts\current_live_url.txt"
$runLog = Join-Path $root "scripts\go_live_last_run.log"

Start-Transcript -Path $runLog -Force | Out-Null

function Test-Url($uri, $timeoutSec) {
    try {
        $resp = Invoke-WebRequest -Uri $uri -Method Head -TimeoutSec $timeoutSec -UseBasicParsing
        return $true
    } catch [System.Net.WebException] {
        # A server response (even an error status) still proves the URL is reachable.
        if ($_.Exception.Response) { return $true }
        return $false
    } catch {
        return $false
    }
}

# 1. Start the Flask app if it isn't already answering.
$flaskUp = Test-Url $flaskUrl 3

if (-not $flaskUp) {
    Write-Host "Starting chatbot server (first load can take up to a minute)..."
    Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "web_app.py" -WorkingDirectory $root -WindowStyle Minimized
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 1
        if (Test-Url $flaskUrl 2) { $flaskUp = $true; break }
    }
    if (-not $flaskUp) {
        Write-Host "WARNING: chatbot server did not respond after 60s. Check for errors." -ForegroundColor Yellow
    } else {
        Write-Host "Chatbot server is up."
    }
} else {
    Write-Host "Chatbot server already running."
}

# 2. Start the Cloudflare tunnel, retrying with a fresh subdomain if the
#    assigned one doesn't actually resolve (quick tunnels occasionally hand
#    out a dead subdomain).
$existing = Get-Process cloudflared -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Stopping existing tunnel to get a fresh URL..."
    $existing | Stop-Process -Force
    Start-Sleep -Seconds 2
}

$url = $null
for ($attempt = 1; $attempt -le 5; $attempt++) {
    Write-Host "Starting Cloudflare Tunnel (attempt $attempt)..."
    Remove-Item $tunnelLog -ErrorAction SilentlyContinue
    Start-Process -FilePath $cloudflaredExe -ArgumentList "tunnel --url $flaskUrl" `
        -WindowStyle Hidden -RedirectStandardError $tunnelLog

    $candidate = $null
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Seconds 1
        if (Test-Path $tunnelLog) {
            $match = Select-String -Path $tunnelLog -Pattern "https://[a-zA-Z0-9\-]+\.trycloudflare\.com" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($match) { $candidate = $match.Matches[0].Value; break }
        }
    }

    if (-not $candidate) {
        Write-Host "  No URL announced yet, retrying..." -ForegroundColor Yellow
        Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force
        continue
    }

    # Quick tunnels are anonymous and occasionally hand out a subdomain that
    # takes a while (or fails) to resolve publicly - this is a known
    # limitation of Cloudflare's free no-signup tunnels, not our setup. Give
    # each candidate up to ~45s before giving up on it.
    Write-Host "  Got $candidate, checking it resolves (can take up to 45s)..."
    $reachable = $false
    for ($i = 0; $i -lt 15; $i++) {
        if (Test-Url $candidate 6) { $reachable = $true; break }
        Start-Sleep -Seconds 3
    }

    if ($reachable) {
        $url = $candidate
        break
    } else {
        Write-Host "  $candidate did not respond, discarding and trying a new one..." -ForegroundColor Yellow
        Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force
        Start-Sleep -Seconds 2
    }
}

if ($url) {
    Write-Host ""
    Write-Host "=======================================================" -ForegroundColor Green
    Write-Host " ORBITPROAI is live at:" -ForegroundColor Green
    Write-Host " $url" -ForegroundColor Green
    Write-Host "=======================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Login with the username/password set in your .env file."
    Write-Host "This URL changes each time the tunnel restarts - rerun this script to get a new one."
    try { Set-Clipboard -Value $url } catch {}
    "$url`n(started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))" | Set-Content -Path $urlFile
} else {
    Write-Host "Could not get a working tunnel URL after 5 attempts. Check $tunnelLog for details." -ForegroundColor Red
    "FAILED to get a live URL as of $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - see go_live_last_run.log" | Set-Content -Path $urlFile
}

Stop-Transcript | Out-Null
