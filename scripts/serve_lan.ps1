param(
    [int]$Port = 8010
)

$ErrorActionPreference = "Stop"

Write-Host "=== JKLU CareerOS LAN Server ===" -ForegroundColor Green

$ips = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object {
        $_.IPAddress -ne "127.0.0.1" -and
        $_.IPAddress -notlike "169.254.*" -and
        $_.PrefixOrigin -ne "WellKnown"
    } |
    Select-Object -ExpandProperty IPAddress -Unique

if (-not $ips) {
    $ips = @("127.0.0.1")
}

Write-Host ""
Write-Host "Detected LAN IPv4 address(es):" -ForegroundColor Cyan
foreach ($ip in $ips) {
    Write-Host "  - Mobile / Desktop URL : http://${ip}:${Port}"
    Write-Host "  - Health Check         : http://${ip}:${Port}/healthz"
    Write-Host "  - Recruiter JNF Base   : http://${ip}:${Port}/jnf/<token>"
}

Write-Host ""
Write-Host "If devices on the same Wi-Fi/Hotspot cannot connect, run (as Administrator):" -ForegroundColor Yellow
Write-Host "  New-NetFirewallRule -DisplayName `"JKLU CareerOS ${Port}`" -Direction Inbound -LocalPort ${Port} -Protocol TCP -Action Allow"

Write-Host ""
Write-Host "For USB Android demo without Wi-Fi:" -ForegroundColor Yellow
Write-Host "  adb reverse tcp:${Port} tcp:${Port}"
Write-Host "  (Then use http://127.0.0.1:${Port} in the Android app)"
Write-Host ""

uv run uvicorn app.api.__main__:app --host 0.0.0.0 --port $Port
