# BT-Lite - Konventionen

Lokales, terminal-artiges Finanz-Analyse-Frontend. Dieses Dokument ist die gekuerzte
Arbeitsfassung. Der Entwurf steht in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), die
Datensemantik in [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

## Kommunikation

- Antworte auf Deutsch, technisch praezise, ohne Fuellsaetze.
- Bei Design-Entscheidungen mit Trade-off: kurz beide Optionen, Empfehlung, dann fragen.
- Keine stillen Annahmen ueber Datenquellen-Semantik. Was nicht in `docs/DATA_SOURCES.md`
  steht, ist nicht entschieden.
- Wenn ein Paket anders funktioniert als angenommen: erst die tatsaechliche API pruefen,
  dann implementieren. Nicht raten.

## Harte Architekturregeln

1. `models/` ist der Vertrag. Kein Modul greift auf rohe Provider-Antworten zu.
2. `data/` ist die einzige Schicht, die Netzwerk macht. Alles andere bekommt Daten
   injiziert, ueber die Protocols aus `data/base.py`.
3. `viz/` gibt `Figure` zurueck und ruft nie `.show()` auf.
4. `ui/` enthaelt keine Geschaeftslogik, nur Callbacks, die Analytics und Viz verdrahten.
5. Jede oeffentliche Funktion: Type Hints, Docstring mit Einheiten (etwa "annualisiert,
   dezimal"), Fehler als eigene Typen aus `core/errors.py`.
6. Alle Zeitreihen: `DatetimeIndex`, tz-aware UTC. Konvertierung nur an der UI-Grenze.

## Konventionen im Detail

- **Spaltennamen** nie als String-Literal, immer als Konstante aus `models/common.py`.
- **Fehlende Werte** sind `None`, nie `0.0`, nie stilles `NaN`. Im UI erscheint "n/a".
- **Einheiten**: Renditen, Volatilitaeten und Risikomasse annualisiert und dezimal
  (0.12 = 12 Prozent). Drawdowns, VaR und CVaR negativ.
- **Kosten**: `fees` und `slippage` dezimale Anteile des Ordervolumens, bei Ein- und
  Ausstieg je angewandt.
- **Vorzeichen**: GuV-Aufwaende positiv, Cashflow-Abfluesse negativ (CapEx also negativ).
- **Look-ahead-Bias**: Signal in Zeile `t` nutzt nur Daten bis `t`. Ausfuehrung regelt
  `SignalSet.execution_lag`, Default 1 Bar.
- **Rate Limits** werden nie umgangen. `RateLimitError` geht nach oben.

## Nicht-Ziele

Keine Broker-Anbindung, keine Orderausfuehrung, kein Live-Trading. Keine
Anlageempfehlungen im Produkt. Kein Multi-User, kein Auth, kein Cloud-Deployment. Keine
Umgehung von Rate Limits, Paywalls oder Bot-Erkennung.

## Qualitaet

```bash
.venv/Scripts/ruff.exe check .
.venv/Scripts/ruff.exe format .
.venv/Scripts/mypy.exe
.venv/Scripts/python.exe -m pytest
```

Alle drei muessen gruen sein. `mypy` laeuft im Strict-Modus. Tests laufen offline;
`tests/conftest.py` blockiert Sockets prozessweit.

Coverage-Ziel: mindestens 80 Prozent auf `analytics/`, `portfolio/`, `screening/` und
`backtest/metrics.py`.

## Umgebung

Python 3.13 im lokalen `.venv`. Wichtig: `uv` ist ueber `python -m uv` installiert und
zielt per Default auf das conda-Base. Deshalb immer explizit:

```bash
python -m uv pip install --python ".venv\Scripts\python.exe" -e ".[dev]"
```

## Versionsstand der Abhaengigkeiten

Gegen die real installierten Versionen geprueft, Stand 4. September 2026:

- **Dash 4.x**: `run_server()` ist entfernt, der Einstieg ist `app.run()`.
  `long_callback` heisst jetzt `background`.
- **yfinance 1.7**: Statements sind transponiert (Zeilen = Positionen).
  `auto_adjust=True` ist Default, es gibt keine `Adj Close`-Spalte.
- **PyPortfolioOpt 1.6**: `risk_free_rate` hat Default `0.0`, nicht `0.02`. Immer explizit
  setzen. Ein `EfficientFrontier`-Objekt ist nach einem Optimize-Aufruf verbraucht.
- **vectorbt 1.1**: optionales Extra. Braucht `freq=`, sonst sind Sharpe und
  Annualisierung falsch. Lizenz mit Commons Clause.
- **finvizfinance 1.5**: Filter-Keys exakt inklusive Punkt, `Market Cap.` nicht
  `Market Cap`.

## Phasenplan

| Phase | Inhalt | Status |
|-------|--------|--------|
| 0 | Skeleton, Config, Fehler, alle Vertraege und Protocols | abgeschlossen |
| 1 | `data-engineer` und `viz-dev`, parallel | abgeschlossen |
| 2 | `fundamentals-analyst`, `screener-dev`, `backtest-dev`, `portfolio-quant`, parallel | abgeschlossen |
| 3 | `ui-dev`, sequenziell | abgeschlossen |
| 4 | `qa-engineer`, dann `doc-writer` | abgeschlossen |
| 5 | Lesbarkeit, Namenssuche, Depot und Watchlist, technisches Screening | abgeschlossen |

Nach jeder Phase: Statusbericht, Commit mit sprechender Message, Freigabe abwarten.
Zwei Agents bearbeiten nie gleichzeitig dieselbe Datei.

### Erfahrung aus Phase 5

Drei Parallel-Agents haben sich gegenseitig beschaedigt. Ein `git checkout --`
auf eine Datei ausserhalb des eigenen Scopes loeschte 373 Zeilen fremder,
uncommitteter Arbeit; ausgeloest hatte es ein `ruff check --fix .` ueber das
ganze Repo. Fuer Agents gilt deshalb: `ruff` und `git` nur auf die eigenen
Dateien, nie repoweit. Der Scope gehoert in den Auftrag, samt der Anweisung,
Fehler ausserhalb davon zu melden statt zu beheben.

Zwei weitere Faelle zeigen, warum Berichte nachgeprueft werden: ein Agent
nannte die Ursache des gemeldeten Darstellungsfehlers korrekt anders als
vorgegeben (Dash-4-Designtokens statt fehlender Klassenregeln, am Paket
verifiziert). Ein anderer lieferte rechnerisch richtige Trendwerte unter
Provider-Rohnamen, die der Rest des Projekts nicht findet - seine Tests
prueften gegen sich selbst statt gegen die Normalisierung.

## Seiten

Sieben Mnemonics: `DES`, `GP`, `FA`, `EQS`, `BT`, `PORT`, `WL`. Die
Abgrenzung von `PORT` und `WL` ist inhaltlich: `PORT` rechnet mit
Zielgewichten aus der Optimierung, `WL` zeigt den tatsaechlichen Bestand
mit Stueckzahlen und Einstandskursen.

Neben der Kommandozeile (`AAPL FA GO`) gibt es die Namenssuche: sie findet
`NYSE:NOW` zu "servicenow" und bietet danach die Auswertungen in Klartext
an. Kein Mnemonic noetig.
