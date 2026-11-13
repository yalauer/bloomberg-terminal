# BT-Lite

Ein lokal laufendes, terminal-artiges Finanz-Analyse-Frontend mit Bloomberg-Anmutung:
dunkel, tastaturgetrieben, mnemonische Kommandozeile. Es holt Kurse und Jahresabschlüsse,
rechnet daraus Kennzahlen, Risikomaße, Screenings, Backtests und Portfolio-Optimierungen
und stellt sie auf sechs Seiten dar. Alles läuft auf `127.0.0.1`, ohne Konto, ohne Cloud.

> ## Haftungshinweis
>
> **BT-Lite ist ein Werkzeug für die eigene Recherche und gibt keine Finanzberatung.**
> Es spricht keine Anlageempfehlungen aus und bewertet keine Titel — es zeigt Kennzahlen.
>
> **Kursdaten sind verzögert, nicht echtzeitig.** Yahoo Finance liefert typischerweise
> 15 Minuten Verzug und korrigiert Historien rückwirkend ohne Ankündigung. Im
> Offline-Modus sind die Daten Tage bis Monate alt; jede Seite weist Quelle, Zeitstempel
> und das tatsächliche Alter sichtbar aus.
>
> Keine Broker-Anbindung, keine Orderausführung, kein Live-Trading. Die Rebalancing-Seite
> schlägt Orders vor, führt aber keine aus. Für Vollständigkeit und Richtigkeit der Daten
> wird nicht eingestanden. Screening-Ergebnisse und Backtests unterliegen einem
> Survivorship Bias.

## Was es nicht ist

Kein Broker-Terminal, kein Live-Trading-System, kein Robo-Advisor. Kein Multi-User-Dienst,
keine Authentifizierung, kein Cloud-Deployment. Kein Echtzeitfeed und kein Ersatz für eine
kostenpflichtige Datenquelle. Nichts davon ist vergessen worden — es ist bewusst
ausgeschlossen, siehe [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Setup

### Lokale Einrichtung auf diesem Rechner

Die Umgebung ist bereits mit Python 3.13.18 und den Extras `dev,backtest`
eingerichtet. Zum Starten [START.bat](START.bat) doppelklicken und
<http://127.0.0.1:8050> im Browser oeffnen. Fuer eingefrorene Beispieldaten
[START-OFFLINE.bat](START-OFFLINE.bat) verwenden.

Im VS-Code-Workspace `E:\GitHub` gibt es ausserdem den Task
`BT-Lite: Live-Terminal starten` unter **Terminal > Task ausfuehren**.

Das lokale uv liegt unter `.bootstrap/bin/uv.exe`. Falls Abhaengigkeiten
erneut installiert werden sollen, im Projektordner ausfuehren:

```powershell
.\.bootstrap\bin\uv.exe pip install --python .venv/Scripts/python.exe -e ".[dev,backtest]" nox
```

Pruefergebnisse und Versionsstand: [Einrichtungsprotokoll](docs/SETUP_VALIDATION.md).

### Allgemeine Installation

Voraussetzung: Python 3.12 oder 3.13. Geprüft ist 3.13.11.

Bei einer neuen Installation mit vorhandenem Python und uv das Ziel-Environment explizit angeben:

```bash
python -m uv venv .venv --python 3.13
python -m uv pip install --python ".venv\Scripts\python.exe" -e ".[dev]"
```

Danach wird das Projekt immer über den Interpreter des `.venv` angesprochen, nie über ein
aktiviertes Environment:

```bash
.venv\Scripts\python.exe -m btlite --version
```

Die Backtest-Engine ist ein optionales Extra, weil `vectorbt` enge Versionsgrenzen für
`pandas`, `numpy` und `numba` erzwingt. Ohne das Extra bleibt das Kernprodukt
installierbar, die Seite `BT` meldet die Engine dann als nicht verfügbar:

```bash
python -m uv pip install --python ".venv\Scripts\python.exe" -e ".[dev,backtest]"
```

`backtrader` ist ein zweiter, optionaler Adapter (`.[backtrader]`). Er ist ein
Strukturnachweis, kein erprobter Pfad.

## Start

**Der einfachste Weg: Doppelklick auf `START.bat`** im Projektordner. Live-Daten,
Port 8050. Für eingefrorene Daten `START-OFFLINE.bat`.

Die `.bat`-Dateien gibt es nicht aus Bequemlichkeit: Windows setzt die
PowerShell-Ausführungsrichtlinie standardmäßig auf `Restricted`, dann scheitert
`.\start.ps1` mit „die Ausführung von Skripts auf diesem System ist deaktiviert".
Die Batch-Datei ruft dasselbe Skript mit `-ExecutionPolicy Bypass` auf — das gilt
nur für diesen einen Aufruf und ändert nichts an den Systemeinstellungen.

In PowerShell direkt, falls Skripte dort erlaubt sind:

```powershell
.\start.ps1            # Live-Daten, Port 8050
.\start.ps1 -Offline   # eingefrorene Daten, ohne Netz
.\start.ps1 -Port 8060
```

Das Terminal läuft danach auf **http://127.0.0.1:8050**. Die Bind-Adresse ist bewusst
lokal.

Das Skript **beendet einen noch laufenden Server auf demselben Port**, bevor es neu
startet. Ohne das läuft der alte Prozess mit dem alten Code weiter — die häufigste
Ursache dafür, dass Änderungen nicht ankommen.

Direkt geht es auch, dann aber selbst um den alten Prozess kümmern:

```bash
.venv\Scripts\python.exe -m btlite
```

### „Die Ausführung von Skripts ist deaktiviert"

Die Standardeinstellung von Windows. `START.bat` umgeht sie für diesen einen
Aufruf. Wer lieber dauerhaft freigibt — nur für das eigene Benutzerkonto, ohne
Administratorrechte:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Änderungen kommen nicht an?

Der laufende Prozess hält den Code, mit dem er gestartet wurde. Nach einem Update also
neu starten — `.\start.ps1` erledigt das. Wer von Hand startet, beendet den alten
Prozess vorher mit Strg+C im betreffenden Fenster oder findet ihn so:

```powershell
Get-NetTCPConnection -LocalPort 8050 -State Listen | Select-Object OwningProcess
Stop-Process -Id <PID> -Force
```

Zeigt der Browser trotzdem den alten Stand, liegt es an seinem Zwischenspeicher:
Strg+F5 lädt die Seite samt Stylesheet neu.

Optionen:

```bash
.venv\Scripts\python.exe -m btlite --port 8060
.venv\Scripts\python.exe -m btlite --debug
.venv\Scripts\python.exe -m btlite --host 127.0.0.1 --port 8060 --log-level DEBUG
```

| Flag | Wirkung | Default |
|------|---------|---------|
| `--host` | Bind-Adresse | `127.0.0.1` |
| `--port` | TCP-Port | `8050` |
| `--debug` | Dash-Debugmodus mit Hot Reload | aus |
| `--log-level` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |
| `--version` | Version ausgeben und beenden | — |

## Offline ist der Default

**Die App startet gegen eingefrorene Fixtures, nicht gegen das Netz.** Das überrascht
sonst: die angezeigten Kurse sind nicht die von heute.

Der Grund ist Absicht — die App startet ohne Netzverbindung, ohne Rate-Limit-Risiko und
zeigt sofort etwas. Kurse, Bilanzen und Backtests gibt es offline für zehn Titel:

`AAPL`, `JNJ`, `JPM`, `KO`, `MSFT`, `NVDA`, `PG`, `SPY`, `WMT`, `XOM`

**Die Namenssuche kennt deutlich mehr.** `tests/fixtures/directory.json` trägt Kürzel,
Firmennamen und Branche der Mitglieder von S&P 500, Nasdaq 100, DAX und Dow Jones —
558 Titel. "tesla" oder "siemens" findet die Suche also auch offline. Zu diesen Titeln
liegen dann aber keine Kurse vor; die Seiten sagen das ausdrücklich und nennen den Weg
zu Live-Daten.

**Screening über echte Indizes braucht Live-Daten.** Die Bestände werden abgerufen, nicht
mitgeliefert. Offline weicht die Seite sichtbar auf die zehn eingefrorenen Titel aus —
sie sagt das im Kopf, statt stillschweigend ein anderes Universum zu screenen.

Live-Daten holt man über eine einzige Umgebungsvariable, oder einfach mit `.\start.ps1`:

```bash
BTLITE_DATA_OFFLINE=0 .venv\Scripts\python.exe -m btlite
```

Es gibt für diese Entscheidung genau einen Schalter. Im Offline-Modus weist jede Seite in
der Herkunftszeile `FIXTURE` sowie das **echte** Alter der Daten aus, nicht die 15 Minuten
einer Live-Quelle.

Die Fixtures liegen in `tests/fixtures/` und werden von `scripts/refresh_fixtures.py`
erzeugt. Das ist der einzige Weg, sie zu erneuern — es braucht Netz.

## Die sieben Seiten

Eingabeformat der Kommandozeile: `<TICKER[,TICKER...]> <MNEMONIC> [ARGUMENT] [GO]`.
Das abschließende `GO` ist optional, Enter löst ohnehin aus. Die vollständige Referenz
steht in [`docs/COMMANDS.md`](docs/COMMANDS.md).

| Mnemonic | Seite | Inhalt | Beispiel |
|----------|-------|--------|----------|
| `DES` | Description | Stammdaten, aktueller Quote, Kennzahlen-Überblick | `AAPL DES GO` |
| `GP` | Graph Price | Kerzen-, OHLC- und Linienchart, Overlays (SMA, EMA, Bollinger, RSI, MACD), Mehr-Ticker-Vergleich auf Basis 100 | `AAPL,MSFT GP 5y GO` |
| `FA` | Financial Analysis | GuV, Bilanz, Kapitalflussrechnung, 30 Kennzahlen, Peer-Vergleich | `AAPL,MSFT FA ttm GO` |
| `EQS` | Equity Screening | Index als Universum (S&P 500, Dow Jones, Nasdaq 100, Russell 2000, DAX), Presets auf Bilanz- und Kurskennzahlen, CSV-Export | `EQS dax value GO` |
| `BT` | Backtest | Vier Strategien, Equity-Kurve, Drawdown, Trade-Liste | `AAPL BT sma_cross GO` |
| `PORT` | Portfolio & Risk | Gewichtsoptimierung, Effizienzgrenze, Korrelation, Risikometriken | `AAPL,MSFT,KO PORT min_volatility GO` |
| `WL` | Watchlist & Depot | Gespeicherter Bestand mit Einstand und Marktwert, Watchlist | `WL GO` |

Ein Tippfehler im Mnemonic wird nicht stumm verschluckt, sondern beantwortet:
`AAPL FAA GO` meldet „Unbekanntes Kommando 'FAA'. Meinten Sie FA?".

**Ohne Mnemonic geht es auch.** Die Suchleiste in der Kopfzeile findet Titel über ihren
Namen — "servicenow" führt zu NYSE:NOW, "siemens" zu SIE.DE — und bietet danach die
Auswertungen in Klartext an. Von dort lässt sich ein Titel auch direkt auf die Watchlist
legen.

## Konfiguration

Alle Einstellungen sind über Umgebungsvariablen mit Präfix `BTLITE_` oder eine
`.env`-Datei im Repo-Root überschreibbar. Verschachtelte Gruppen tragen ihren eigenen
Präfix.

| Variable | Wirkung | Default |
|----------|---------|---------|
| `BTLITE_DATA_OFFLINE` | Fixtures statt Netz | `true` |
| `BTLITE_DATA_AUTO_ADJUST_PRICES` | split- und dividendenbereinigte Kurse | `true` |
| `BTLITE_DATA_QUOTE_POLL_INTERVAL_S` | Poll-Intervall des Quote-Streams | `15.0` |
| `BTLITE_UI_PORT` | TCP-Port | `8050` |
| `BTLITE_UI_HOST` | Bind-Adresse | `127.0.0.1` |
| `BTLITE_UI_DISPLAY_TIMEZONE` | Zeitzone **nur** der Anzeige, intern ist alles UTC | `Europe/Berlin` |
| `BTLITE_ANALYTICS_RISK_FREE_RATE` | risikofreier Zins, dezimal | `0.02` |
| `BTLITE_ANALYTICS_BENCHMARK_TICKER` | Benchmark für Beta und Alpha | `SPY` |
| `BTLITE_BACKTEST_FEES` | Kommission je Trade, dezimal | `0.0005` |
| `BTLITE_BACKTEST_SLIPPAGE` | Slippage je Trade, dezimal | `0.0005` |
| `BTLITE_RATELIMIT_YFINANCE_RPS` | Requests pro Sekunde gegen yfinance | `2.0` |
| `BTLITE_CACHE_ENABLED` | globaler Cache-Schalter | `true` |
| `BTLITE_LOG_LEVEL` | Log-Level | `INFO` |

Die vollständige Referenz mit allen Feldern und ihren Beschreibungen steht in
[`src/btlite/config.py`](src/btlite/config.py).

## Screenshots

Die Bilder liegen unter [`docs/screenshots/`](docs/screenshots/). Was dort hingehört und
in welcher Form, steht in der README dieses Verzeichnisses.

## Architektur in fünf Regeln

Details in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

1. `models/` ist der Vertrag. Kein Modul sieht rohe Provider-Antworten.
2. `data/` ist die einzige Schicht, die Netzwerk macht. Alles andere bekommt Daten
   injiziert.
3. `viz/` gibt `Figure` zurück und zeigt nie selbst etwas an.
4. `ui/` enthält keine Geschäftslogik, nur Verdrahtung.
5. Alle Zeitreihen sind tz-aware in UTC. Konvertiert wird erst an der UI-Grenze.

Dazu zwei durchgängige Konventionen: ein nicht berechenbarer Wert ist `None` und erscheint
im UI als „n/a", nie als stille Null. Renditen, Volatilitäten und Risikomaße sind
annualisiert und dezimal; Drawdown, VaR und CVaR sind negativ.

## Datenquellen

Kurse, Quotes und Statements kommen von `yfinance`, Screening-Filter von `finvizfinance`.
Beide Quellen sind verzögert und werden konservativ gedrosselt — Rate Limits werden nie
umgangen, ein `RateLimitError` geht sichtbar nach oben. Die Semantik der Felder,
Vorzeichenkonventionen, Fiskaljahr-Versatz und die bekannten Fallstricke stehen in
[`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

## Qualität

```bash
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format .
.venv\Scripts\mypy.exe
.venv\Scripts\python.exe -m pytest
```

`mypy` läuft im Strict-Modus. Die Testsuite läuft vollständig offline gegen die Fixtures;
`tests/conftest.py` blockiert Sockets prozessweit, ein versehentlicher Netzwerkzugriff
schlägt laut fehl. Coverage-Ziel sind mindestens 80 Prozent auf `analytics/`,
`portfolio/`, `screening/` und `backtest/metrics.py`.

## Lizenz

Das Projekt selbst steht unter der **MIT-Lizenz**.

**Achtung beim optionalen Extra `[backtest]`:** `vectorbt` steht unter Apache 2.0
**mit Commons Clause**. Die Commons Clause untersagt den Verkauf der Software und
kommerzieller Dienste, deren Wert im Wesentlichen aus ihrer Funktionalität stammt. Wer
BT-Lite mit installiertem `vectorbt` kommerziell nutzen oder weitergeben will, muss das
prüfen. Ohne dieses Extra ist das Kernprodukt davon nicht betroffen — deshalb ist es
optional und wird in `backtest/vbt_engine.py` lazy importiert.
