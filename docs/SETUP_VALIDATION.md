# Einrichtung und Validierung

Stand: 30. Oktober 2026, Windows, lokales Repository `E:\GitHub\bloomberg-terminal`.

## Import und Repository

- Quelle: `C:\Users\Yannick\Downloads\Bloomberg Terminal.zip`.
- 291 Projektdateien importiert; Umgebungen, Caches, `.hypothesis`, `.env`,
  Nutzerbestand unter `/data` und Coverage-Ausgaben ausgeschlossen.
- 289 unveraenderte Dateien per SHA-256 gegen das ZIP geprueft. Bewusst angepasst
  wurden nur die Ignore-Regeln und die datumsabhaengigen Watchlist-Tests.
- Remote: <https://github.com/yalauer/bloomberg-terminal>, privat.
- Branch `main`; vorhandener GitHub-README-Startcommit bleibt erhalten.
- Git-Identitaet ausschliesslich lokal: `Yannick <Yannick@PCYANNICK>`.
- Vor dem Staging Dateinamen und typische Token-/Private-Key-Muster geprueft;
  keine entsprechenden Zugangsdaten gefunden. GitHub-Credentials wurden nur
  im Arbeitsspeicher verwendet.

## Lokale Umgebung

Python 3.13.18 in `.venv`, verwalteter Interpreter unter `.python`.
Installation mit lokalem uv 0.13.6 unter `.bootstrap`, ohne globale Python-Pakete
zu veraendern. Installiert: `.[dev,backtest]` sowie der vorhandene Task-Runner nox.
Wichtige Versionen: Dash 4.5.0, pandas 3.1.0, NumPy 2.5.4, vectorbt 1.2.0,
yfinance 1.7.0, PyPortfolioOpt 1.6.0, Ruff 0.17.3, Mypy 2.4.1, pytest 9.2.0.
`uv pip check` bestaetigt kompatible Abhaengigkeiten (141 Pakete).

Erneute Installation im eingerichteten Projekt:

```powershell
.\.bootstrap\bin\uv.exe pip install --python .venv/Scripts/python.exe -e ".[dev,backtest]" nox
```

Start mit [START.bat](../START.bat) fuer Live-Daten oder
[START-OFFLINE.bat](../START-OFFLINE.bat) fuer eingefrorene Daten.

## Pruefungen

- Ruff: `check src tests noxfile.py` bestanden.
- Ruff: `format --check src tests noxfile.py` bestanden, 156 Dateien.
- Mypy strict: keine Fehler in 87 Quelldateien.
- `python -m btlite --version`: `btlite 0.1.0`.
- Offline-Start ueber `start.ps1 -Offline -Port 18050`: Startseite,
  `/_dash-layout` und `/_dash-dependencies` jeweils HTTP 200; Testserver danach beendet.
- Gezielte Tests nach Korrektur: 119 bestanden (Watchlist und Earnings).

Die erste vollstaendige Offline-Suite ergab 2.922 bestandene und zwei fehlgeschlagene
Tests bei 88,54 Prozent Coverage. Ursache waren abgelaufene, eingefrorene
Earnings-Termine (20., 28. und 29. Oktober 2026): Die Anwendung unterdrueckt sie
korrekt als vergangen, die UI-Tests erwarteten jedoch kommende Termine. Die
betroffenen Tests verwenden jetzt den festen Bezugstag 1. Oktober 2026;
der Sortiertest verlangt ausserdem drei tatsaechlich vorhandene Termine.
Die Produktlogik und die eingefrorenen Finanzdaten bleiben unveraendert.

Vollstaendige Abschlusspruefung am 30. Oktober 2026 abgeschlossen:

- **2.924 Tests bestanden**, keine fehlgeschlagen, Laufzeit 559,22 Sekunden.
- Gesamt-Coverage inklusive Branches: **88,65 Prozent**.
- Analytics: 93,70 Prozent; Portfolio: 92,74 Prozent; Screening: 95,22 Prozent;
  Backtest-Metriken: 90,49 Prozent. Alle geforderten Schichten ueber 80 Prozent.
- 693 Warnungen, unter anderem aus Drittanbieter-Paketen; keine Testfehler.
- Vollstaendige Ausgabe lokal in `validation-results.log`, maschinenlesbare
  Coverage in `.cache/setup-coverage.json`.

## Start und Live-Pruefung

- VS-Code-Task `BT-Lite: Live-Terminal starten` im uebergeordneten Workspace
  eingerichtet und gestartet. Er ruft `start.ps1` im Live-Modus auf.
- Das Terminal laeuft lokal unter <http://127.0.0.1:8050>.
- Startseite, `/_dash-layout` und `/_dash-dependencies`: HTTP 200.
- Browser: AAPL-Profil mit Kurs, Unternehmensdaten, Jahresabschluss-Kennzahlen
  und Herkunftszeile erfolgreich geladen.
- Separater Abruf ueber `YFinanceProvider.get_quotes(['AAPL'])`: erfolgreich,
  Quelle `yfinance`, `from_cache=False`, Kurszeitstempel vorhanden.
  Live-Modus bedeutet Abruf beim Provider; der angezeigte Kurs kann weiterhin
  verzoegert sein oder vom letzten Handelstag stammen.
- Erneuter Paketcheck: alle 141 Pakete kompatibel; Ruff, Formatpruefung und
  Mypy strict erneut bestanden.

Die Tests sperren externe Socket-Verbindungen. Drittanbieter melden
Deprecation- und Resource-Warnungen.
