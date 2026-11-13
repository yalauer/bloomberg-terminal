# Architektur

Dieses Dokument beschreibt den **Ist-Stand** nach Phase 3. Wo der ursprüngliche Entwurf
aus Phase 0 abweicht, ist die Abweichung benannt und begründet.

## Schichtenmodell

```
ui/          Dash-Shell, Kommandozeile, Callbacks       <- keine Geschaeftslogik
viz/         Plotly-Theme und Chart-Factories           <- gibt Figures zurueck
analytics/   Statements, Ratios, Technicals, Risk   \
screening/   Filter-Engine und Presets                > Rechenschichten, netzfrei
backtest/    Engines, Strategien, Metriken            /
portfolio/   Optimierung, Attribution, Rebalancing  /
data/        Provider, Caching, Rate Limiting           <- EINZIGE Netzwerkschicht
models/      Pydantic-Vertraege                         <- Single Source of Truth
core/        Fehler, Zeit, Logging, Cache, Ratelimit
```

Der Datenfluss laeuft ausschliesslich nach oben. Keine Rechenschicht ruft `data/` direkt
auf; sie bekommt Provider ueber Dependency Injection hereingereicht.

## Die fuenf harten Regeln

### 1. `models/` ist der Vertrag

Kein Modul sieht rohe Provider-Antworten. Alles fliesst durch validierte Pydantic-Modelle
mit `pd.DataFrame`-Payloads unter den in `models/common.py` festgelegten Spaltennamen.

Spaltennamen stehen **nie** als String-Literal im Code, sondern immer als Konstante:
`OHLCV_COLUMNS`, `DATE_INDEX_NAME`, `STATEMENT_INDEX_NAME`, `RETURN_COLUMN`.

Die Modelle sind streng konfiguriert: `extra="forbid"` und `validate_assignment=True`. Ein
Vertragsbruch faellt sofort auf, nicht erst drei Schichten weiter oben.

### 2. `data/` ist die einzige Netzwerkschicht

Alles andere bekommt Daten injiziert, ueber die Protocols in `data/base.py`:
`InstrumentProvider`, `PriceProvider`, `FundamentalsProvider`, `ScreenerProvider`,
`IndexProvider`, `ExchangeRateProvider`, `QuoteStream`.

Die Protocols sind `runtime_checkable` und strukturell. Eine Implementierung muss nicht
erben, nur passen. Das macht Test-Doubles trivial: ein Objekt mit den richtigen Methoden
genuegt.

Konsequenz fuer Tests: die gesamte Suite laeuft offline gegen Fixtures. `tests/conftest.py`
blockiert Sockets prozessweit, ein versehentlicher Netzwerkzugriff schlaegt laut fehl.
Loopback ist dabei erlaubt, weil die asyncio-Eventloop unter Windows intern ein lokales
Socket-Paar oeffnet; externe Ziele bleiben blockiert.

**Abweichung vom Entwurf:** Der Offline-Modus gilt nicht mehr nur fuer Tests. Seit Phase 3
steht `DataSettings.offline` auf `True`, die laufende App startet also ebenfalls gegen die
Fixtures. Sie braucht damit kein Netz, um etwas zu zeigen, und laeuft niemandem
versehentlich ins Rate Limit. Live-Daten holt `BTLITE_DATA_OFFLINE=0`.

Das ist bewusst **ein** Schalter. Ein zweiter, parallel eingefuehrter `BTLITE_UI_LIVE_DATA`
wurde wieder entfernt: zwei Schalter fuer dieselbe Entscheidung sind eine Fehlerquelle.

### 3. `viz/` gibt Figures zurueck

Chart-Factories liefern `plotly.graph_objects.Figure` und rufen **nie** `.show()` auf. Sie
enthalten keine Geschaeftslogik und keine Datenbeschaffung. Damit sind sie ohne laufendes
UI testbar.

### 4. `ui/` verdrahtet nur

Callbacks rufen Analytics und Viz auf und setzen Ergebnisse ins Layout. Sie rechnen nicht
selbst. Wo im UI eine Berechnung entsteht, gehoert sie nach `analytics/`.

### 5. Zeit ist UTC

Alle Zeitreihen tragen einen tz-aware `DatetimeIndex` in UTC. Kein Modul ruft
`datetime.now()` direkt auf, das laeuft ueber `core/clock.py`. Konvertiert wird
ausschliesslich an der UI-Grenze.

## Fehlerbehandlung

Alle Fehler erben von `BTLiteError` und tragen zwei getrennte Felder:

- `user_message`: nicht-technisch, erscheint im UI-Fehlerbanner.
- `detail`: technischer Kontext, geht nur ins Log.

Rohe Provider-Exceptions verlassen `data/` nie. Sie werden auf `ProviderUnavailableError`,
`RateLimitError`, `SymbolNotFoundError` oder `DataValidationError` abgebildet.

`RateLimitError` wird bewusst **nach oben gereicht statt umgangen**. Wenn eine Quelle
drosselt, sieht der Nutzer das.

## Fehlende Werte

Ein nicht berechenbarer Wert ist `None`, nie `0.0` und nie stilles `NaN`. `Ratio` traegt
zusaetzlich `inputs_missing` mit den fehlenden Eingangspositionen. Das UI rendert das als
"n/a". Diese Regel ist ein Akzeptanzkriterium des Projekts, kein Stilwunsch: eine still zu
Null gewordene Kennzahl verfaelscht jeden Peer-Vergleich, der sie mittelt.

## Backtest: zweistufige Trennung

Die Trennung von Signal und Ausfuehrung ist bewusst:

1. Eine `Strategy` erzeugt aus Kursdaten einen `SignalSet`. Sie kennt keine Engine, keine
   Kosten, kein Kapital.
2. Eine `BacktestEngine` fuehrt die Signale aus und liefert ein `BacktestResult`.

Damit sind Strategien ohne vectorbt testbar, und die Engine ist austauschbar. `vectorbt`
ist primaer, `backtrader` ein optionaler Adapter. Beide fuellen dasselbe Ergebnismodell.

**Look-ahead-Bias**: Ein Signal in Zeile `t` darf ausschliesslich Daten bis einschliesslich
`t` verwenden. Die Ausfuehrungsverzoegerung regelt `SignalSet.execution_lag`, Default 1 Bar.

## Optionale Abhaengigkeiten

`vectorbt` erzwingt enge Versionsgrenzen: `pandas>=3.0.3`, `numpy>=2.4.6`, `numba>=0.66`
und damit Python 3.12 oder neuer. Es steht deshalb als Extra `[backtest]` und wird in
`vbt_engine.py` lazy importiert. `BacktestEngine.is_available()` prueft das zur Laufzeit,
damit das UI nur anbietet, was wirklich da ist.

Das Kernprodukt bleibt dadurch installierbar, auch wenn dieses Versions-Kartell an einer
Stelle bricht.

`plotly` traegt zusaetzlich eine **harte Obergrenze** `<7.0`: vectorbt 1.1 deklariert nur
`plotly>=4.12`, benutzt intern aber `scattermapbox`, das Plotly 7 entfernt hat. Bereits
`import vectorbt` schlaegt damit fehl. Die Deklaration des Pakets ist fehlerhaft, der Pin
im `pyproject.toml` faengt das ab.

Stand im aktuellen `.venv`: `vectorbt` ist installiert, `VectorbtEngine.is_available()`
meldet `True`. `backtrader` ist **nicht** installiert, `BacktraderEngine.is_available()`
meldet `False` und `run()` wirft. Der Adapter ist ein Strukturnachweis und wurde nie
ausgefuehrt.

## Module und ihre Verantwortung

Der Ist-Stand. Kursiv gesetzt sind die Module, die im Entwurf aus Phase 0 noch nicht
vorgesehen waren.

### `core/` — Querschnitt

| Modul | Verantwortung |
|-------|---------------|
| `errors.py` | Fehlerhierarchie ab `BTLiteError`, je mit `user_message` und `detail` |
| `clock.py` | `utc_now()` und `ensure_utc()`. Kein Modul ruft `datetime.now()` selbst |
| `logging.py` | Klartext- oder JSON-Logging, ein Logger je Modul |
| `cache.py` | zweistufig: `requests-cache` für HTTP, `diskcache`/parquet für DataFrames |
| `ratelimit.py` | Token-Bucket je Quelle, Backoff mit Jitter, prozessweite Registry |

### `models/` — der Vertrag

`common.py` (Enums, Spaltenkonstanten, `Provenance`, `normalize_ticker`), `market.py`
(`Instrument`, `PriceSeries`, `Quote`, `PriceComparison`), `fundamentals.py`
(`FinancialStatement`, `StatementBundle`, `Ratio`, `RatioSet`, `PeerComparison`),
`screening.py`, `backtest.py`, `portfolio.py`.

### `data/` — die einzige Netzwerkschicht

| Modul | Verantwortung |
|-------|---------------|
| `base.py` | die Protocols, siehe unten |
| `yfinance_provider.py` | Kurse, Quotes, Statements, Stammdaten |
| `finviz_provider.py` | Screener-Filter |
| *`fixtures.py`* | `FixtureProvider`: erfüllt dieselben Protocols, geht nie ins Netz |
| *`quote_stream.py`* | `PollingQuoteStream`, holt alle Symbole in **einem** Request pro Poll |
| `indices.py` | `IndexConstituentsProvider`: Bestaende von S&P 500, DAX, STOXX 600 und weiteren |
| `rates.py` | `YFinanceRateProvider`: Wechselkurse ueber `EURUSD=X`, nur fuer die Depotbewertung |

`fixtures.py` ist **kein Nebenprodukt der Tests**, sondern ein zentrales Deliverable. Er
bedient den Offline-Default der laufenden App, nicht nur die Suite.

Die Protocols sind `runtime_checkable` und strukturell: `InstrumentProvider`,
`PriceProvider`, `FundamentalsProvider`, `ScreenerProvider`, `IndexProvider`,
`ExchangeRateProvider`, `QuoteStream`. Dazu kam *`MarketDataProvider`*, die Komposition
der ersten drei — die meisten Verbraucher brauchen alle drei zusammen, und drei einzeln
injizierte Provider wären in der Praxis immer dasselbe Objekt gewesen.

`ExchangeRateProvider` ist die einzige Quelle einer Waehrungsumrechnung im Projekt, und
sie bedient **nur** `portfolio/`. Ein Depot hat eine Berichtswaehrung, seine Positionen
duerfen in einer anderen gekauft sein; ohne Umrechnung waere die Summe eine Addition
ungleicher Einheiten. Analytik, Screening und Backtest rechnen bewusst nicht um — die
Semantik samt der geprueften Eigenheiten der Quelle steht in `docs/DATA_SOURCES.md`.

### `analytics/` — Rechenschicht, netzfrei

| Modul | Verantwortung |
|-------|---------------|
| `statements.py` | Normalisierung der Provider-Rohnamen auf `LineItem`, TTM-Aggregation, Fiskaljahr-Ableitung |
| `ratios.py` | 30 Kennzahlen in sechs Kategorien, je mit Einheit und Klartext-Formel |
| `technicals.py` | SMA, EMA, RSI nach Wilder, Bollinger, MACD, ATR — selbst gebaut |
| `risk.py` | füllt `RiskMetrics`: Volatilität, Sharpe, Sortino, Beta, Alpha, VaR, CVaR, Drawdown |
| *`series.py`* | Umformung mehrerer Kursreihen zu gemeinsamen Matrizen, siehe unten |
| `setups.py` | Setup-Checkliste auf Tageskursen: vier messbare Chart-Bedingungen, Zonen aus bestätigten Swings |

**Setup-Checkliste (`analytics/setups.py`, Vertrag `models/setups.py`).** Je Setup ein
Status `met`, `not_met` oder `insufficient_data` plus Messwerte mit Einheit — bewusst
**kein Gesamturteil, kein Score, keine Ampel**, keine Anlageempfehlung. Die vier Setups:
Trend-Pullback (Kurs und SMA50 über SMA200, |Abstand zur SMA50| ≤ 3 %), RSI-Erholung
(RSI 14 in den 10 Bars vor dem Stichtag unter 30, am Stichtag ≥ 30), Ausbruch (Schluss
über dem höchsten Schluss der 252 Vorbars, Volumen über dem 63-Tage-Schnitt) und
Unterstützung + MACD (Kurs höchstens 3 % über der nächsten Unterstützungszone, jüngste
MACD-Kreuzung der letzten 5 Bars nach oben). Swings sind Extrema gegen je 5 Bars links und
rechts und zählen erst nach Bestätigung (kein Look-ahead); Zonen fassen Niveaus innerhalb
1.5 % über dem niedrigsten Niveau der Zone zusammen. Zu kurze Historie ergibt
`insufficient_data`, im Screening `None`. Die Schwellen stehen als `Final`-Konstanten im
Modul, die Begründungen im Modul-Docstring. `screening/technicals.py` bildet den Status auf
`setup_*`-Spalten (1.0/0.0/None) ab, GP zeigt die Checkliste unter dem Chart.

### `screening/`

`filters.py` (Filter-DSL mit elf Operatoren und rekursiven Gruppen), `engine.py`
(Ausführung gegen ein injiziertes Universum, Export nach CSV und Parquet), `presets.py`
(vier Presets), *`universe.py`* (das Bindeglied, siehe unten).

### `backtest/`

`base.py` (Protocols `BacktestEngine`, `StrategyRegistry`, Basisklasse `Strategy`),
`metrics.py` (engine-unabhängige Kennzahlen), `vbt_engine.py`, `bt_engine.py`,
*`registry.py`* (konkrete Registry hinter dem Protocol), `strategies/` mit
`buy_and_hold.py`, `sma_cross.py`, `rsi_mean_reversion.py`, `momentum_rotation.py` und
*`_panel.py`* (geteilter Aufbau der Mehr-Ticker-Kursmatrix).

`metrics.py` existiert, damit `vectorbt` und `backtrader` vergleichbare Zahlen liefern und
nicht jede Engine ihre eigene Annualisierung mitbringt. Der Faktor kommt aus
`Interval.periods_per_year`, nie hartcodiert.

### `portfolio/`

`optimizer.py` (acht Zielfunktionen, Effizienzgrenze, Discrete Allocation),
`attribution.py`, `rebalance.py`. Letzteres **schlägt Orders vor**; das Projekt führt
bewusst keine aus.

### `viz/`

`theme.py` (Plotly-Template), `price.py`, `fundamentals.py`, `risk.py`, `frontier.py` und
*`backtest.py`* (Equity-Kurve, Trade-Marker, Monatsrendite-Heatmap).

Steigend ist **grün** (`#00E676`), fallend **rot** (`#E03131`) — die vertraute
Börsenkodierung, aber gemessen statt gegriffen. Ein *typisches* Grün/Rot fällt bei
Rot-Grün-Sehschwäche zusammen (Delta E 6.5, unter der Schwelle von 8); dieses Paar
kommt auf 20.3, weil die **Helligkeit** die Unterscheidung trägt: helles Grün gegen
dunkles Rot. Sattere Grüntöne derselben Familie bestehen die Prüfung nicht — `#16A34A`
erreicht gegen dasselbe Rot nur 4.5.

Dieselbe Bindung gilt für die Polaritäts-Rampe (`DIVERGING`, Korrelation und
Monatsrenditen): ihre Pole sind die Richtungsfarben, damit ein positiver Monat in der
Heatmap dieselbe Farbe trägt wie eine steigende Kerze. Auch dort steigt die Helligkeit
monoton von Rot nach Grün — zwei gleich dunkle Enden fielen sonst zusammen (gemessen
6.2 gegenüber 55.2 mit Helligkeitsverlauf).

Die Kategorienpalette bleibt davon unberührt: dort bedeuten Blau und Orange „Serie 1"
und „Serie 2", keine Richtung. Und unabhängig von der Farbe trägt jede Kerze ihre
Richtung in der Geometrie, jede Kennzahl ihren Pfeil und ihr Vorzeichen — Farbe ist nie
der einzige Träger der Aussage.

### `ui/`

`app.py` (Dash-Shell, Routing, alle Callbacks), `command_bar.py` (reiner Parser),
*`state.py`* (`AppState`: Provider-Instanzen und LRU-Puffer), *`formatting.py`*
(Zahlenformatierung, „n/a" für `None`), `layouts/components.py`, `pages/` mit je einem
Modul pro Mnemonic.

`state.py` und `formatting.py` waren im Entwurf nicht vorgesehen. Sie halten Regel 4 ein:
`AppState` beschafft und puffert, rechnet aber nicht, und `formatting.py` ist reine
Darstellung.

**Die Seite WL schreibt.** Sie ist die einzige, die Nutzereingaben persistiert: Depots
und Watchlists anlegen und loeschen, Kaeufe erfassen, aendern und entfernen, Notizen
pflegen. Regel 4 gilt unveraendert - der Callback liest die Felder, ruft
`portfolio/store.py` und rendert neu. Er rechnet nicht.

Zwei Eigenheiten, die man beim Lesen der Callbacks kennen muss:

*Eine Zeile je Kauf, nicht je Titel.* `Portfolio.holdings` ist eine Liste; zwei Kaeufe
desselben Symbols sind zwei Eintraege, und der Store adressiert sie ueber ihren
nullbasierten Index. Die Bewertungstabelle darueber fasst dieselben Kaeufe je Ticker
zusammen. Beide Darstellungen stehen bewusst nebeneinander: die eine ist die
Kaufhistorie, die andere der Bestand.

*Der Index ist eine wandernde Adresse.* Nach `remove_holding(0)` ist die vormalige
Position 1 die neue Position 0. Deshalb speichert ein Callback immer nur **eine** Zeile,
findet sie ueber `ctx.triggered_id` und ordnet die Werte ueber die mitgelieferte Liste
der Feld-IDs zu - nicht ueber die Listenreihenfolge. Anschliessend wird neu gerendert,
damit die Zeilen wieder ihre aktuellen Indizes tragen.

Fehlerverhalten: ein abgewiesener Wert laesst den Bestand stehen (`no_update`) und
wechselt nur die Rueckmeldung. Ein Nutzer, dem nach einem Tippfehler das Depot
verschwindet, haelt es fuer geloescht.

## Die beiden Nachzügler aus Phase 3

Der `ui-dev` meldete zwei fehlende Rechenfunktionen, statt sie in `ui/` zu bauen. Sie
liegen jetzt dort, wo sie hingehören.

### `analytics/series.py`

- `build_comparison()` erzeugt die auf Basis 100 normierte Vergleichsmatrix, die
  `viz.price.comparison` erwartet. Ohne sie zeigte `GP` bei mehreren Tickern nur
  Einzelcharts mit je eigener Skala — optisch ein Vergleich, rechnerisch keiner.
- `portfolio_returns()` gewichtet Einzelreihen zu einer Portfolio-Renditereihe. Sie
  unterscheidet zwei Lesarten: laufend zurückgewichtet oder einmal gewichtet und gehalten.
  Über fünf Jahre liegen die beiden 57 Prozentpunkte auseinander, das ist keine Feinheit
  und darf nicht implizit bleiben.

Beide bringen die Reihen auf ihre **Schnittmenge**, statt sie vorwärtszutragen. Ein später
startender Titel ginge sonst mit erfundenen konstanten Kursen ein und drückte die
Kovarianz nach unten.

Damit füllt der Optimizer auch `PortfolioResult.risk`. Die Seite `PORT` zeigt erwartete
und realisierte Kennzahlen nebeneinander, statt „n/a".

### `screening/universe.py`

Das Bindeglied zwischen Kennzahlen und lokalen Filtern. Die lokale Filter-DSL erwartet die
Schlüssel aus `ratios.py` (`pe`, `roe`, `current_ratio`) in deren Einheiten — Margen und
Renditen dezimal. Ein Finviz-Ergebnis liefert dagegen Spalten wie `P/E` und `Market Cap`.
Ohne dieses Modul liefen die lokalen Prädikate ins Leere und die Presets waren nicht
lauffähig.

`build_ratio_universe()` reichert eine Tickerliste über den injizierten Provider zu einem
DataFrame an, indiziert nach Ticker, mit Kennzahlen- und Stammdatenspalten. Es holt nichts
selbst (Regel 2).

## Datenfluss

Am Beispiel `AAPL,MSFT FA ttm GO`:

1. **`ui/command_bar.parse_command`** zerlegt die Zeile zu `Command(mnemonic="FA",
   tickers=("AAPL","MSFT"), args=("ttm",))`. Reiner Parser, kennt weder Dash noch Provider.
2. **`ui/app.build_href`** übersetzt das in `/fa?t=AAPL%2CMSFT&a=ttm` und setzt die
   Adresse. Die Navigation läuft über die URL, nicht über einen versteckten Zustand —
   damit sind Seiten teilbar und der Zurückknopf funktioniert.
3. **`ui/app._on_route`** liest die Adresse zurück und ruft `render_page`, das jeden
   `BTLiteError` abfängt und als Banner rendert, nie als leere Seite.
4. **`ui/pages/fa.layout`** fragt `AppState` nach dem Statement-Bundle.
5. **`ui/state.AppState`** hält den Provider — im Offline-Default `FixtureProvider`, sonst
   `YFinanceProvider` — und einen LRU-Puffer. Hier und nur hier entscheidet sich, ob Netz
   im Spiel ist.
6. **`data/`** liefert ein validiertes `StatementBundle` mit `Provenance`. Rohe
   Provider-Antworten enden hier; eine Provider-Exception wird auf einen eigenen Fehlertyp
   abgebildet.
7. **`analytics/statements.normalize_bundle`** mappt die Rohnamen auf `LineItem` und
   aggregiert TTM. **`analytics/ratios.compute_ratios`** rechnet die Kennzahlen. Beide
   bekommen das Bundle **injiziert** und holen nichts nach.
8. **`viz/fundamentals`** baut aus den Modellen `Figure`-Objekte und Tabellenzeilen, ohne
   `.show()`.
9. **`ui/formatting`** setzt Zahlen in deutsche Schreibweise und `None` in „n/a".
   **`ui/layouts/components`** rahmt das Ergebnis, inklusive der pflichtigen
   Herkunftszeile mit Quelle, Zeitstempel und Verzug.

Der Fluss läuft ausschließlich nach oben. Keine Rechenschicht ruft `data/` auf; der
einzige Ort, an dem eine Schicht einen Provider kennt, ist `ui/state.py` — und die reicht
ihn nach unten durch, statt ihn dort zu verstecken.

## Verzeichnisstruktur

Alles, was zur Laufzeit entsteht, bleibt **innerhalb des Projektordners**:

```
<Projektwurzel>/
├─ data/                holdings.json - Depots und Watchlists (in .gitignore)
├─ .cache/              HTTP-Cache, DataFrame-Cache, yfinance/ (in .gitignore)
└─ src/btlite/          Quellcode
```

```
src/btlite/
├─ __main__.py          CLI-Einstieg, argparse, app.run()
├─ config.py            Settings via pydantic-settings, .env, Praefix BTLITE_
├─ core/                errors.py, clock.py, logging.py, cache.py, ratelimit.py
├─ models/              common.py, market.py, fundamentals.py,
│                       screening.py, backtest.py, portfolio.py
├─ data/                base.py (Protocols), yfinance_provider.py,
│                       finviz_provider.py, fixtures.py, quote_stream.py
├─ analytics/           statements.py, ratios.py, technicals.py, risk.py, series.py
├─ screening/           engine.py, filters.py, presets.py, universe.py
├─ backtest/            base.py (Protocols), registry.py, metrics.py,
│                       vbt_engine.py, bt_engine.py, strategies/
├─ portfolio/           optimizer.py, attribution.py, rebalance.py
├─ viz/                 theme.py, price.py, fundamentals.py, risk.py,
│                       frontier.py, backtest.py
└─ ui/                  app.py, command_bar.py, state.py, formatting.py,
                        layouts/, pages/
```

## Kommandozeile

Eingabeformat: `<TICKER[,TICKER...]> <MNEMONIC> [ARGUMENT...] [GO]`

Das abschliessende `GO` ist optional. Gegenueber dem Entwurf kam ein Argumentteil hinter
dem Mnemonic dazu, ueber den die Seiten ihren Zeitraum, ihre Periodenbasis, ihr Preset,
ihre Strategie oder ihre Zielfunktion vorbelegen.

| Mnemonic | Seite | Argument |
|----------|-------|----------|
| `DES` | Stammdaten und Ueberblick | keines |
| `GP` | Charts, Candlestick, Overlays, Vergleich | Zeitraum |
| `FA` | Statements, Kennzahlen, Peer-Vergleich | Periodenbasis |
| `EQS` | Screening | Preset-Key |
| `BT` | Backtest | Strategie-Key |
| `PORT` | Portfolio und Risiko | Zielfunktions-Key |

Der Parser in `ui/command_bar.py` ist rein: er kennt weder Dash noch Provider und ist ohne
laufendes UI testbar. Ein unbekanntes Mnemonic beantwortet er ueber `difflib` mit dem
naechstliegenden Vorschlag. Die vollstaendige Referenz steht in
[`COMMANDS.md`](COMMANDS.md).

Ein Kommando wird in eine Adresse uebersetzt (`/fa?t=AAPL,MSFT&a=ttm`), nicht in einen
verborgenen Zustand. Damit traegt die URL die gesamte Anfrage und der Zurueckknopf
funktioniert.

`CommandHistory` in `command_bar.py` ist implementiert und getestet, aber **nicht ans UI
verdrahtet**: es gibt derzeit keine Pfeiltasten-Historie in der laufenden App.

## Ablage: nichts ausserhalb des Projektordners

Das Werkzeug schreibt ausschliesslich unterhalb seiner eigenen Wurzel. Wer den
Ordner kopiert, hat Bestand und Cache dabei; wer ihn loescht, laesst nichts
zurueck. Zwei Orte:

| Pfad | Inhalt | Einstellung |
|------|--------|-------------|
| `data/holdings.json` | Depots und Watchlists | `BTLITE_PORTFOLIO_HOLDINGS_FILE` |
| `.cache/` | HTTP-Antworten, DataFrames, yfinance-Eigencache | `BTLITE_CACHE_DIR` |

Beide stehen in `.gitignore`. Depotbestand ist Nutzerdatum, kein Projektinhalt,
und aus einer Git-Historie bekaeme man ihn nur mit Aufwand wieder heraus.

**Der Sonderfall yfinance.** Das Paket legt unabhaengig von unserer Cache-Schicht
eigene SQLite-Dateien an - Zeitzonen je Ticker, Cookies, ISIN-Zuordnungen - und
zwar per Default in einem Nutzerverzeichnis ausserhalb des Projekts (unter
Windows `%LOCALAPPDATA%` / `py-yfinance`). `core.cache.redirect_yfinance_cache`
verlegt sie nach `.cache/yfinance/`. Aufgerufen wird sie an jeder Stelle, die
`yfinance` importiert; der Aufruf ist idempotent und laeuft einmal je Prozess.

Am Paket geprueft, nicht angenommen: `yfinance.set_tz_cache_location` ruft
intern `set_cache_location` und deckt damit **alle drei** Caches ab, nicht nur
den Zeitzonen-Cache, den sein Name nennt.

## Nicht-Ziele

Bewusst ausgeschlossen und nicht nachtraeglich hinzuzufuegen, ohne den Entwurf zu
ueberdenken:

- Keine Broker-Anbindung, keine Orderausfuehrung, kein Live-Trading.
- Keine Anlageempfehlungen. Das Produkt zeigt Kennzahlen, es bewertet sie nicht.
- Kein Multi-User, kein Auth, kein Cloud-Deployment. Bind-Adresse ist `127.0.0.1`.
- Keine Umgehung von Rate Limits, Paywalls oder Bot-Erkennung.
