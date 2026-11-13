@echo off
rem BT-Lite im Offline-Modus starten - per Doppelklick.
rem
rem Eingefrorene Daten, kein Netzzugriff. Startet sofort, zeigt aber nur
rem zehn Titel mit Kursen. Screening ueber echte Indizes braucht Live-Daten,
rem dafuer START.bat nehmen.

cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" -Offline %*

if errorlevel 1 (
    echo.
    echo Der Start ist fehlgeschlagen. Meldung siehe oben.
    pause
)
