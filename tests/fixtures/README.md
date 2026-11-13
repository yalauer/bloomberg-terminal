# Fixtures - eingefrorene API-Antworten

Diese Dateien sind **echte, einmal live gezogene Antworten** von Yahoo Finance und
Finviz. Sie sind die einzige Datengrundlage der Testsuite: `tests/conftest.py`
blockiert Sockets prozessweit, ein versehentlicher Netzwerkzugriff schlaegt laut
fehl.

Wer offline gegen Daten entwickeln will, nimmt
`btlite.data.fixtures.FixtureProvider`. Der erfuellt dieselben Protocols wie
`YFinanceProvider` und `FinvizProvider` und liest ausschliesslich aus diesem
Verzeichnis.

## Stand

| Abschnitt | Gezogen am (UTC) |
|-----------|------------------|
| Kurshistorien | 2026-09-04 06:36 |
| Statements | 2026-09-04 06:37 |
| Stammdaten | 2026-09-04 06:37 |
| Screener | 2026-09-04 06:37 |
| Symbolsuche | 2026-09-20 12:57 |
| Wechselkurse | 2026-09-24 15:41 |
| Quartalszahlen-Termine | 2026-09-24 16:44 |

Gezogen gegen: yfinance 1.7.0, finvizfinance 1.5.0, pandas 3.0.5, pyarrow 25.0.1.
Der maschinenlesbare Stand steht in `manifest.json`.

## Inhalt

### `prices/<TICKER>_1d.parquet`

OHLCV-Tageshistorien ueber 5 Jahre (je 1254 Handelstage, 2021-09-07 bis
2026-09-03) fuer zehn Ticker:

    AAPL  MSFT  NVDA  SPY  JNJ  XOM  KO  PG  JPM  WMT

Zehn Stueck, damit Phase 2 eine Portfolio-Optimierung ueber zehn Titel testen
kann. `SPY` dient zusaetzlich als Benchmark fuer Beta, Alpha und Tracking Error.

Schema: Index `date`, tz-aware UTC, aufsteigend, ohne Duplikate. Spalten
`open`, `high`, `low`, `close`, `volume`, `dividends`, `stock_splits`.

Gezogen mit `auto_adjust=True`. Es gibt daher **keine** `Adj Close`-Spalte,
`close` ist bereits split- und dividendenbereinigt.

### `statements/<TICKER>_<typ>_<periode>.parquet`

Alle drei Statement-Typen (`income`, `balance`, `cashflow`) jeweils `annual`
und `quarterly`, fuer alle Kursticker ausser SPY - ein ETF fuehrt keine
Statements. Also AAPL, MSFT, NVDA, JNJ, XOM, KO, PG, JPM und WMT: 54 Dateien.

Die Breite ist Absicht. Ein Screening-Preset braucht ein Universum mit
Streuung: mit nur drei teuren Qualitaetstiteln findet ein Value-Screen per
Definition nichts, und das Akzeptanzkriterium "Preset liefert mehr als null
Treffer" waere nicht pruefbar. XOM, KO, JPM und WMT bringen niedrigere
Bewertungen und hoehere Dividendenrenditen ins Universum.

Schema: **Zeilen = Positionen** (Index `line_item`), **Spalten = Perioden**.
Das ist die yfinance-Orientierung, die der Vertrag `FinancialStatement`
bewusst beibehaelt.

Wichtig: Die Zeilenbezeichner sind die **rohen** yfinance-Namen
(`TotalRevenue`, `NetIncome`, ...), nicht die kanonischen aus `LineItem`. Die
Normalisierung passiert genau einmal, in `analytics/statements.py` - nicht in
der Datenschicht und nicht hier.

Die Periodenspalten liegen als ISO-Strings vor, weil parquet String-Spaltennamen
verlangt. `FixtureProvider` wandelt sie beim Laden in einen tz-aware
UTC-`DatetimeIndex` zurueck.

Zu beachten: AAPL schliesst sein Geschaeftsjahr Ende **September** ab, nicht im
Dezember. Genau dafuer traegt `FinancialStatement.fiscal_year_end_month` den
Monat mit - ein Peer-Vergleich AAPL gegen JNJ ist nicht periodengenau.

### `info/<TICKER>.json`

Stammdaten aus `Ticker.info` fuer alle zehn Kurs-Ticker. Rund 180 Felder je
Aktie, 103 fuer den ETF `SPY`. Nicht JSON-serialisierbare Werte wurden beim
Einfrieren verworfen.

Genutzt fuer `Instrument`, die Waehrungsangabe und
`fiscal_year_end_month` (aus `lastFiscalYearEnd`).

### `screener/overview.parquet`

Ein Finviz-Screener-Ergebnis der `Overview`-View, 40 Treffer:
`{"Index": "S&P 500", "Sector": "Technology"}`. Die Anfrage steht daneben in
`overview_request.json`.

Schema: Index `ticker`, Spalten `Company`, `Sector`, `Industry`, `Country`,
`Market Cap`, `P/E`, `Price`, `Change %`, `Volume`.

### `search/<slug>.json`

Eingefrorene Antworten der Yahoo-Symbolsuche (`/v1/finance/search`), je
Suchbegriff eine Datei. Der Dateiname ist der Slug aus
`btlite.data.search.query_slug`: kleingeschrieben, ohne Sonderzeichen,
Leerzeichen als `_`. `AAPL` und `aapl` landen damit in derselben Datei.

Format: `{"query": "<Roheingabe>", "quotes": [...]}`. Gespeichert wird **nur**
die `quotes`-Liste der Antwort; News, Listen und Zeitmessungen sind fuer die
Suche bedeutungslos.

Vorhanden sind die neun Anfragen `servicenow`, `apple`, `microsoft`, `nvidia`,
`johnson`, `bank`, `now`, `aapl` und `microsft`.

Relevante Felder je Treffer: `symbol`, `longname`/`shortname`, `exchDisp`
(lesbarer Boersenname wie `NYSE`) neben `exchange` (internes Kuerzel wie
`NYQ`), `quoteType` und `score`.

**`microsft.json` ist bewusst leer.** Yahoos Suche kennt ohne
`enableFuzzyQuery` keine Tippfehler und liefert zu diesem Vertipper null
Treffer. Das ist eine Eigenschaft der Quelle, kein defektes Fixture: der Fall
wird erst durch das lokale `difflib`-Ranking in `btlite/data/search.py` ueber
den Stammdatenbestand aufgeloest. Genau deshalb sucht der `FixtureProvider`
zusaetzlich in den `info/`-Stammdaten und nicht nur in diesen Antworten.

### `earnings/<TICKER>.json`

Der naechste Quartalszahlen-Termin je Titel, Grundlage der Termin-Spalte in
`PORT` und `WL`. Zwoelf Dateien: die zehn Kursticker plus `RKLB` und `SAP.DE`.

    {"ticker": "AAPL", "has_earnings": true, "dates": ["2026-10-29"]}

Abgelegt ist der **Rohbefund** von `yf.Ticker(sym).calendar`, nicht ein
vorberechnetes "naechstes Datum":

- `has_earnings` haelt fest, ob die Quelle den Schluessel `Earnings Date`
  ueberhaupt fuehrt. `false` heisst: das Instrument fuehrt keine
  Quartalszahlen. Intern laeuft dort ein HTTP 404, den `yfinance` abfaengt.
- `dates` sind die gelieferten Kalendertage, meist genau einer.

**`SPY` ist bewusst dabei, obwohl die Datei nichts enthaelt** (`has_earnings:
false`). Ein ETF hat keine Quartalszahlen, und genau dieser Fall muss offline
pruefbar sein - sonst fiele erst live auf, ob der 404 als Fehler hochschlaegt
und die Depotseite kippt.

**Diese Fixtures veralten kontrolliert.** Ist der eingefrorene Termin
inzwischen verstrichen, liefert der Provider `next_date=None` statt eines
Datums in der Vergangenheit; die Auswahl des naechsten Termins laeuft bei jedem
Lesen neu gegen den heutigen Tag. Die Datei muss dafuer nicht angefasst werden,
und die Tests gehen mit der Zeit nicht kaputt: sie erzeugen ihre eigenen
Termine relativ zu `utc_today()` und pruefen die ausgelieferten Fixtures nur
auf `has_earnings`, nie auf ein festes Datum.

Ein fehlendes Fixture ist hier **kein Fehler** - anders als bei den
Wechselkursen. "Kein Termin bekannt" ist eine legitime Aussage, ein geratener
Wechselkurs waere dagegen eine Falschaussage. Ein Titel ohne Datei bekommt
`has_earnings=true` und `next_date=None`, also "n/a" statt des Strichs fuer
"gibt es nicht".

### `rates/<SLUG>.parquet`

FX-Tageshistorien ueber 5 Jahre (je 1300 Zeilen, 2021-09-23 bis 2026-09-23) fuer
20 Waehrungspaare. Sie sind die Grundlage der **Depotbewertung**: die
Berichtswaehrung eines Depots ist standardmaessig EUR, die Kurse von yfinance
kommen in USD. Ohne diese Fixtures stuende offline - dem Default beim Start der
App und in der gesamten Testsuite - bei jeder Dollar-Position "n/a".

**Auswahl der Paare.** Nicht alle Kreuze, sondern fuer jede Waehrung aus
`models.common.Currency` (ausser `OTHER`) die **beiden USD-Schenkel**:

    AUDUSD  CADUSD  CHFUSD  DKKUSD  EURUSD  GBPUSD  HKDUSD  JPYUSD  NOKUSD  SEKUSD
    USDAUD  USDCAD  USDCHF  USDDKK  USDEUR  USDGBP  USDHKD  USDJPY  USDNOK  USDSEK

Das genuegt, weil `data/rates.py` bei einem fehlenden Direktkurs ueber
`PIVOT_CURRENCY` (USD) trianguliert: jedes Paar aus dem Vertrag ist damit
offline bedienbar. Alle 110 Kreuze zu ziehen waere nicht nur unnoetig, sondern
teils unmoeglich - Yahoo fuehrt fuer exotische Paare zwar einen Live-Kurs, aber
**keine Historie** (`HKDNOK=X` lieferte ueber ein Jahr genau eine Zeile, die
beiden USD-Schenkel je 263).

Beide Richtungen je Waehrung, weil der Provider das Symbol aus der Anfrage baut
und nicht den Kehrwert bildet: `USD -> EUR` fragt `USDEUR=X` ab, nicht
`1 / EURUSD=X`.

**Dateiname.** Der Slug aus `btlite.data.rates.fx_fixture_slug`: das bei jedem
FX-Symbol identische `=X` faellt weg, `EURUSD=X` wird zu `EURUSD.parquet`. Die
Abbildung ist umkehrbar (`fx_symbol_from_slug`). Windows erlaubt `=` im
Dateinamen durchaus - in Shell-Aufrufen und Globs ist es nur laestig.

**Schema.** Index `date`, tz-aware UTC, aufsteigend, ohne Duplikate. Genau eine
Spalte `close` als float. Gespeichert wird das Ergebnis von
`normalize_fx_history`, also dieselbe Form, die ein Live-Abruf nach der
Normalisierung hat - der Offline-Zweig und der Netz-Zweig liefern damit
identisch geformte Frames.

**Zeitzone - der wichtigste Punkt.** Yahoo liefert den Index der FX-Historien in
`Europe/London`, **nicht** in UTC. Nach der Normalisierung liegt ein Londoner
Handelstag im Winter auf 00:00 UTC, im Sommer (BST) auf **23:00 UTC des
Vortages**. Verglichen wird deshalb nie ueber den rohen Zeitstempel, sondern
ueber das Londoner Kalenderdatum (`rates.py::_london_dates`). Gemessen an
`EURUSD.parquet`: bei **763 von 1300** Zeilen weicht das naive UTC-Datum vom
Londoner Kalendertag ab. Wer hier naiv rechnet, liefert bei knapp 60 Prozent
aller Kauftage den Kurs des falschen Tages.

Die Ablage bewahrt diese Eigenschaft: der UTC-Stempel ist verlustfrei, das
Londoner Datum bleibt rekonstruierbar. Konkret steht in `EURUSD.parquet` der
15.01.2026 auf `2026-01-15 00:00:00+00:00` und der 15.07.2026 auf
`2026-07-14 23:00:00+00:00`.

**Richtung.** `EURUSD=X` notiert USD je EUR, also `betrag_in_usd =
betrag_in_eur * kurs`. 1.1370 am 23.09.2026.

**Wochenenden und Feiertage** fuehren keine Zeile; es gilt der letzte
Handelstag davor (`asof`-Semantik). Ein Datum **vor** der ersten Zeile hat
keinen Vorgaenger und ist ein Fehler - der aelteste bekannte Kurs wird nicht
vorgetaeuscht.

**Fehlt ein Paar, wird geworfen.** Kein Rueckfall auf 1.0, kein Schaetzwert,
auch nicht offline. Eine Luecke erscheint im UI sichtbar als "n/a"; ein
geratener Kurs saehe richtig aus und verfaelschte still jede Summe ueber
fremdes Vermoegen.

## Erneuern

```bash
.venv\Scripts\python.exe scripts\refresh_fixtures.py
```

Das Skript ist das **einzige** Stueck Code im Repository, das absichtlich ins
Netz geht. Es laeuft **nie** automatisch in der Testsuite.

Abschnittsweise erneuern, um die Quellen zu schonen:

```bash
.venv\Scripts\python.exe scripts\refresh_fixtures.py --only prices
.venv\Scripts\python.exe scripts\refresh_fixtures.py --only rates
.venv\Scripts\python.exe scripts\refresh_fixtures.py --only statements --only info
.venv\Scripts\python.exe scripts\refresh_fixtures.py --only screener
.venv\Scripts\python.exe scripts\refresh_fixtures.py --only search
.venv\Scripts\python.exe scripts\refresh_fixtures.py --tickers AAPL,MSFT
```

Nach jedem Lauf: die Tabelle unter "Stand" oben aktualisieren und die Suite
laufen lassen. Ein voller Lauf dauert einige Minuten - das Skript wartet
bewusst zwischen den Requests.

## Was beim Erneuern kaputtgehen kann

- **Kurse aendern sich rueckwirkend.** Yahoo korrigiert Historien ohne
  Ankuendigung. Tests, die auf konkrete Kurswerte pruefen, brechen dann. Tests
  sollten daher Struktur pruefen (Spalten, Index, Sortierung), nicht Betraege.
- **Der Screener liefert andere Treffer.** Finviz aktualisiert taeglich. Tests
  duerfen sich nicht auf eine bestimmte Trefferzahl oder Ticker-Menge
  verlassen.
- **Statement-Zeilen kommen und gehen.** Yahoo fuehrt nicht fuer jedes
  Unternehmen jede Position. Das ist der Grund, warum fehlende Werte im Projekt
  `None` sind und nie `0.0`.
- **Die Wechselkurse enden am Abrufdatum.** Ein Bewertungsstichtag nach dem
  letzten Handelstag der Fixture bekommt den letzten bekannten Kurs
  (`asof`-Semantik), kein Fehler. Das ist gewollt, altert aber: nach einem
  halben Jahr rechnet ein Offline-Depot noch immer mit dem Kurs vom
  Abrufdatum. Der Herkunftsnachweis weist das Alter aus.
- **Kommt eine Waehrung zu `models.common.Currency` hinzu**, fehlen ihre beiden
  USD-Schenkel. `fx_pairs()` in `refresh_fixtures.py` leitet die Liste aus dem
  Enum ab, zieht sie aber nicht von selbst nach - also `--only rates` laufen
  lassen. `tests/unit/test_data_rates.py::TestShippedFixtures` prueft genau
  das ab und wird sonst rot.
- **Die Suche liefert bei Null Treffern keine Datei.** `refresh_fixtures.py`
  ueberspringt leere Antworten, damit ein Ausfall der Quelle kein leeres
  Fixture ueber ein gutes schreibt. Fuer `microsft` ist die leere Antwort aber
  der *erwartete* Zustand; die Datei wurde daher von Hand mit
  `{"query": "microsft", "quotes": []}` angelegt. Nach einem `--only search`
  pruefen, ob sie noch existiert - sonst faellt der Tippfehler-Test aus.
