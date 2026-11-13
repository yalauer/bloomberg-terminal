@echo off
rem BT-Lite starten - per Doppelklick.
rem
rem Ruft start.ps1 mit -ExecutionPolicy Bypass auf. Ohne das scheitert der
rem Start: die Ausfuehrungsrichtlinie steht auf diesem Rechner auf
rem "Restricted", PowerShell fuehrt dann ueberhaupt keine Skripte aus.
rem
rem Live-Daten sind der Default. Fuer den Offline-Modus: START-OFFLINE.bat

cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*

rem Fenster offen halten, falls der Start scheitert - sonst waere die
rem Fehlermeldung nach einem Doppelklick sofort wieder weg.
if errorlevel 1 (
    echo.
    echo Der Start ist fehlgeschlagen. Meldung siehe oben.
    pause
)
