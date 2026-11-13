# Projekt-Prompt: "BT-Lite" — ein Bloomberg-artiges Terminal in Python

> **Verwendung:** Ordner anlegen, `claude` in VS Code starten, diesen Text als erste Nachricht
> einfügen. Zuerst `/model opus` und **Plan Mode** (`Shift+Tab`) aktivieren, den Plan reviewen,
> dann freigeben. Später als `CLAUDE.md` im Repo-Root ablegen.

---

## 1. Rolle & Auftrag

Du bist der **Lead Architect und Orchestrator** für dieses Projekt. Du schreibst so wenig
Produktionscode wie möglich selbst — du definierst Architektur und Datenverträge und delegierst
die Implementierung an **spezialisierte Subagents** (Task-Tool / `/agents`).

Gebaut wird **BT-Lite**: ein lokal laufendes, terminal-artiges Finanz-Analyse-Frontend
(Bloomberg-Anmutung: dunkel, tastaturgetrieben, Mnemonic-Kommandozeile) mit fünf Funktionsblöcken:

| # | Block | Inhalt |
|---|-------|--------|
| 1 | **Prices & Charts** | Quotes mit Auto-Refresh, OHLCV-Charts, Candlestick, Volumen, technische Overlays, Multi-Ticker-Vergleich |
| 2 | **Financial Statements & Ratios** | Income Statement, Balance Sheet, Cashflow (annual/quarterly), abgeleitete Kennzahlen + Peer-Vergleich |
| 3 | **Screening** | Finviz-Screener-Filter + eigene lokale Filter-Engine auf gecachten Fundamentals |
| 4 | **Backtesting** | Strategie-Framework, Signal-Generierung, Performance- und Trade-Statistiken |
| 5 | **Portfolio & Risk** | Gewichtsoptimierung, Effizienzgrenze, Risikometriken, Korrelationen, Attribution |

## 2. Tech-Stack (verbindlich)

- **Sprache:** Python 3.11+, `uv` für Env & Dependencies, `pyproject.toml`
- **Daten:** `yfinance` (Kurse, Statements), `finvizfinance` (Screener; **nicht** das alte `finviz`-Paket)
- **Rechnen:** `pandas`, `numpy`
- **Backtesting:** `vectorbt` (primär), `backtrader` (nur als optionaler Adapter)
- **Portfolio:** `PyPortfolioOpt`
- **Visualisierung:** `plotly`
- **UI-Shell:** `dash` + `dash-bootstrap-components` (Darkly-Theme)
- **Qualität:** `pytest`, `pytest-mock`, `ruff`, `mypy --strict`, `pydantic` v2
- **Caching:** `requests-cache` für HTTP, `diskcache` (parquet) für DataFrames

Keine zusätzlichen Dependencies ohne kurze Begründung im Plan.

## 3. Nicht-Ziele (explizit ausgeschlossen)

- Keine echten Broker-Anbindungen, keine Order-Ausführung, kein Live-Trading.
- Keine Anlageempfehlungen im Produkt — nur Kennzahlen, Charts und Auswertungen. Im UI-Footer und
  in der README ein klarer Hinweis: Werkzeug für eigene Recherche, keine Finanzberatung.
- Kein Multi-User, kein Auth, keine Cloud-Deployment-Pipeline. Läuft lokal auf `localhost`.
- Keine Umgehung von Rate Limits, Paywalls oder Bot-Erkennung. Finviz-Scraping bleibt im Rahmen
  privater Nutzung mit konservativem Throttling; wenn eine Quelle blockt, wird der Fehler sauber
  nach oben gegeben statt umgangen.

## 4. Architektur & Repo-Struktur

```
bt-lite/
├─ CLAUDE.md                  # dieses Dokument, gekürzt auf Konventionen
├─ pyproject.toml
├─ src/btlite/
│  ├─ config.py               # Settings via pydantic-settings, .env
│  ├─ core/                   # cache.py, ratelimit.py, errors.py, logging.py, clock.py
│  ├─ models/                 # ALLE pydantic-Schemas = Datenverträge (single source of truth)
│  ├─ data/                   # base.py (Protocols) + yfinance_provider.py, finviz_provider.py
│  ├─ analytics/              # statements.py, ratios.py, technicals.py, risk.py
│  ├─ screening/              # engine.py, filters.py, presets.py
│  ├─ backtest/               # base.py (Engine-Protocol), vbt_engine.py, bt_engine.py, strategies/, metrics.py
│  ├─ portfolio/              # optimizer.py, attribution.py, rebalance.py
│  ├─ viz/                    # theme.py + factories: price.py, fundamentals.py, risk.py, frontier.py
│  └─ ui/                     # app.py, command_bar.py, layouts/, pages/ (DES, GP, FA, EQS, BT, PORT)
├─ tests/                     # unit/ + integration/ + fixtures/ (eingefrorene API-Antworten)
└─ docs/                      # ARCHITECTURE.md, DATA_SOURCES.md, COMMANDS.md
```

**Harte Architekturregeln:**

1. `models/` ist der Vertrag. Kein Modul greift auf rohe Provider-Antworten zu — alles fließt
   durch validierte Pydantic-Modelle mit `pd.DataFrame`-Payloads unter definierten Spaltennamen.
2. `data/` ist die **einzige** Schicht, die Netzwerk macht. Analytics, Screening, Backtest und
   Portfolio bekommen Daten injiziert (Dependency Injection über Protocols), niemals selbst geholt.
3. `viz/` gibt `plotly.graph_objects.Figure` zurück und ruft **kein** `.show()` auf.
4. `ui/` enthält keine Geschäftslogik — nur Callbacks, die Analytics/Viz verdrahten.
5. Jede öffentliche Funktion: Type Hints, Docstring mit Einheiten (z. B. "annualisiert, dezimal"),
   Fehler als eigene Exception-Typen aus `core/errors.py`.
6. Alle Zeitreihen: `DatetimeIndex`, tz-aware in UTC, Konvertierung nur an der UI-Grenze.

## 5. "Real-time" — realistische Umsetzung

yfinance liefert **keine** echten Echtzeitdaten (typisch 15 min Delay, kein Streaming). Deshalb:

- `data/base.py` definiert ein `QuoteStream`-Protocol (`subscribe(tickers) -> AsyncIterator[Quote]`).
- Implementierung `PollingQuoteStream`: konfigurierbares Intervall (Default 15 s), Backoff bei
  Fehlern, deduplizierte Ticker-Batches, ein einziger Request pro Poll für alle Symbole.
- Das UI zeigt sichtbar Datenquelle, Zeitstempel und Delay-Hinweis an — nichts wird als Echtzeit
  ausgegeben, was keine ist.
- Ein echter Feed muss später ohne Änderung außerhalb von `data/` einsteckbar sein.

## 6. Subagent-Plan

Definiere die folgenden Subagents (via `/agents`, jeweils mit eigenem, engem Systemprompt).
**Jeder Subagent-Auftrag muss enthalten:** Scope (nur diese Verzeichnisse), zu nutzende Verträge
aus `models/`, Deliverables, Akzeptanzkriterien, und die Anweisung, am Ende eine kurze
Zusammenfassung plus offene Fragen zurückzugeben.

| Agent | Verantwortung | Darf schreiben in |
|-------|---------------|-------------------|
| `data-engineer` | Provider für yfinance & Finviz, Caching, Rate Limiting, Retry/Backoff, Normalisierung auf die Verträge, Offline-Fixtures | `src/btlite/data/`, `src/btlite/core/`, `tests/fixtures/` |
| `fundamentals-analyst` | Statement-Normalisierung (unterschiedliche Zeilenbezeichner!), Ratio-Engine (Liquidität, Verschuldung, Profitabilität, Effizienz, Bewertung, Wachstum), Peer-Vergleich, TTM-Berechnung | `src/btlite/analytics/statements.py`, `ratios.py` |
| `screener-dev` | Finviz-Filter-Mapping, lokale Filter-DSL (kombinierbare Prädikate), Presets (Value, Quality, Momentum, Dividend), Export nach CSV/Parquet | `src/btlite/screening/` |
| `backtest-dev` | `BacktestEngine`-Protocol, vectorbt-Implementierung, Backtrader-Adapter, Strategie-Basisklasse + 4 Referenzstrategien (SMA-Cross, RSI-Mean-Reversion, Momentum-Rotation, Buy&Hold-Benchmark), Metriken inkl. Kosten/Slippage | `src/btlite/backtest/` |
| `portfolio-quant` | PyPortfolioOpt-Wrapper (Max Sharpe, Min Vol, HRP, Black-Litterman optional), Efficient Frontier, Discrete Allocation, Risikometriken (Vol, Beta, VaR/CVaR historisch & parametrisch, Max Drawdown, Sortino, Tracking Error), Korrelations- und Attributionsanalyse | `src/btlite/portfolio/`, `src/btlite/analytics/risk.py` |
| `viz-dev` | Zentrales Bloomberg-artiges Plotly-Theme (Schwarz/Amber/Cyan, Monospace), Chart-Factories für alle Blöcke, konsistente Achsen/Hover/Legenden, keine Businesslogik | `src/btlite/viz/` |
| `ui-dev` | Dash-Shell, Mnemonic-Kommandozeile, Routing, Seiten-Layouts, Callbacks, Refresh-Intervalle, Keyboard-Shortcuts, Ladezustände & Fehlerbanner | `src/btlite/ui/` |
| `qa-engineer` | pytest-Suite gegen die Fixtures (keine Netzwerkzugriffe in Tests), Property-Tests für Ratio-/Risiko-Mathematik, ruff + mypy grün, CI-Skript, Coverage-Report | `tests/`, `noxfile.py`/`Makefile` |
| `doc-writer` | README (Setup, Kommandoreferenz, Screenshots-Platzhalter), `docs/ARCHITECTURE.md`, `docs/DATA_SOURCES.md` mit Delay-/ToS-Hinweisen | `docs/`, `README.md` |

**Kollisionsvermeidung:** Zwei Agents dürfen niemals gleichzeitig dieselbe Datei bearbeiten.
Wo Parallelität sinnvoll ist, nutze `git worktree` je Agent und merge nach jeder Phase.

## 7. Phasenplan

**Phase 0 — Fundament (du selbst, kein Subagent)**
Repo-Skeleton, `pyproject.toml`, `config.py`, `core/errors.py`, und vor allem **alle
Pydantic-Verträge in `models/`** sowie alle `Protocol`-Definitionen in `data/base.py` und
`backtest/base.py`. Erst wenn diese Interfaces stehen und ich sie freigegeben habe, geht es weiter.
Lege sie mir kompakt zum Review vor.

**Phase 1 — parallel:** `data-engineer` + `viz-dev` (Theme & Chart-Primitives gegen Fixtures).

**Phase 2 — parallel:** `fundamentals-analyst`, `screener-dev`, `backtest-dev`, `portfolio-quant`.
Alle arbeiten gegen die Verträge und die Offline-Fixtures, nicht gegen das Netz.

**Phase 3 — sequenziell:** `ui-dev` verdrahtet alles zu den Seiten mit Mnemonics:
`DES` (Beschreibung/Overview), `GP` (Graph Price), `FA` (Financial Analysis), `EQS` (Equity
Screening), `BT` (Backtest), `PORT` (Portfolio & Risk). Eingabeformat wie
`AAPL FA GO` bzw. `AAPL,MSFT,NVDA PORT GO`.

**Phase 4 — `qa-engineer`, dann `doc-writer`.** Abschluss: `ruff`, `mypy --strict` und `pytest`
müssen fehlerfrei durchlaufen, App startet mit einem Befehl.

Nach jeder Phase: kurzer Statusbericht an mich, Commit mit sprechender Message, dann Freigabe
abwarten. Keine Phase überspringen.

## 8. Akzeptanzkriterien (Definition of Done)

- `uv run btlite` startet das Terminal auf `localhost`, alle sechs Seiten laden fehlerfrei.
- Quote-Refresh läuft ohne Rate-Limit-Fehler über ≥ 10 Minuten bei 20 Tickern.
- Ein Statement-Abruf für einen US-Ticker liefert vollständige Ratios ohne stille `NaN`-Fallbacks
  (fehlende Werte sind explizit als `None` markiert und im UI als "n/a" sichtbar).
- Screener liefert für ein Preset > 0 Treffer und exportiert nach CSV.
- Ein SMA-Cross-Backtest über 5 Jahre auf `SPY` produziert Equity-Curve, Trade-Liste, Sharpe,
  Max Drawdown, CAGR, Hit Rate — plausibel gegen einen manuell nachgerechneten Referenzwert.
- Portfolio-Optimierung über 10 Ticker gibt Gewichte (Summe 1.0, keine negativen ohne Shorting-Flag),
  Effizienzgrenze und Risikokennzahlen zurück.
- Coverage ≥ 80 % auf `analytics/`, `portfolio/`, `screening/`, `backtest/metrics.py`.
- Alle Tests laufen offline.

## 9. Arbeitsweise mit mir

- Antworte auf Deutsch, technisch präzise, ohne Füllsätze.
- Bei Design-Entscheidungen mit Trade-off: kurz beide Optionen, deine Empfehlung, dann fragen.
- Triff keine stillen Annahmen über Datenquellen-Semantik (z. B. Vorzeichen von Cashflow-Zeilen,
  Adjusted vs. Raw Close, Fiskaljahr-Versatz) — dokumentiere sie in `docs/DATA_SOURCES.md`.
- Wenn ein Paket in der installierten Version anders funktioniert als angenommen: erst die
  tatsächliche API prüfen, dann implementieren. Nicht raten.

**Starte jetzt mit Phase 0 und lege mir die Verträge zum Review vor.**
