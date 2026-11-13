# BT-Lite starten.
#
# Beendet einen noch laufenden Server auf demselben Port und startet neu.
# Ohne das Beenden laeuft der alte Prozess mit dem alten Code weiter - genau
# die Falle, in die man nach einem Update tappt.
#
#   .\start.ps1              Live-Daten, Port 8050
#   .\start.ps1 -Offline     eingefrorene Daten, startet ohne Netz
#   .\start.ps1 -Port 8060   anderer Port
#
# Anhalten mit Strg+C.

param(
    [int]$Port = 8050,
    [switch]$Offline
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Kein .venv gefunden unter $python" -ForegroundColor Red
    Write-Host "Zuerst einrichten:" -ForegroundColor Yellow
    Write-Host '  python -m uv pip install --python ".venv\Scripts\python.exe" -e ".[dev]"'
    exit 1
}

# Laeuft dort noch ein Server? Dann beenden, sonst sieht man weiter den
# alten Stand und sucht den Fehler an der falschen Stelle.
$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
foreach ($conn in $busy) {
    $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "Beende laufenden Server auf Port $Port (PID $($proc.Id))" -ForegroundColor Yellow
        Stop-Process -Id $proc.Id -Force
        Start-Sleep -Milliseconds 700
    }
}

if ($Offline) {
    $env:BTLITE_DATA_OFFLINE = "1"
    Write-Host "Offline-Modus: eingefrorene Daten, kein Netzzugriff." -ForegroundColor DarkGray
    Write-Host "Screening ueber echte Indizes braucht Live-Daten." -ForegroundColor DarkGray
} else {
    $env:BTLITE_DATA_OFFLINE = "0"
    Write-Host "Live-Daten aktiv. Erster Screening-Lauf eines grossen Index dauert einige Minuten." -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "BT-Lite startet auf http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host "Anhalten mit Strg+C." -ForegroundColor DarkGray
Write-Host ""

& $python -m btlite --port $Port
