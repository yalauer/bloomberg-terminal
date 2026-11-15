# Datenquellen

Dieses Dokument haelt die **Semantik** der externen Daten fest. Regel des Projekts:
keine stillen Annahmen ueber Vorzeichen, Bereinigung oder Fiskaljahre. Was hier nicht
steht, ist nicht entschieden.

Stand der API-Pruefung: 4. September 2026, gegen die real installierten Versionen.

## Uebersicht

| Quelle | Paket | Verwendung | Verzug |
|--------|-------|------------|--------|
| Yahoo Finance | `yfinance` 1.7 | Kurse, Quotes, Statements, Stammdaten | ca. 15 min, kein Streaming |
| Finviz | `finvizfinance` 1.5 | Screener-Filter | Tagesbasis, HTML-Scraping |
| Fixtures | — | eingefrorene Antworten der beiden obigen | **Tage bis Monate**, siehe unten |

**Die Fixtures sind der Default**, nicht die Ausnahme. Seit Phase 3 startet auch die
laufende App offline. Live-Daten holt `BTLITE_DATA_OFFLINE=0`.

## yfinance

### Kein Echtzeitfeed

yfinance liefert **keine** Echtzeitdaten und keinen Push-Stream. Typisch sind 15 Minuten
Verzug. Daraus folgt der Entwurf in `data/base.py`:

- `QuoteStream` ist ein Protocol, kein konkreter Feed.
- `PollingQuoteStream` pollt in einem konfigurierbaren Intervall (Default 15 s) und holt
  **alle** abonnierten Symbole in genau einem Request pro Poll.
- Jedes `Quote` traegt eine `Provenance` mit `delay_minutes`. Das UI zeigt Quelle,
  Zeitstempel und Verzug sichtbar an.
- Ein echter Feed ist spaeter einsteckbar, ohne dass sich etwas ausserhalb von `data/`
  aendert.

### Adjusted vs. Raw Close

`auto_adjust=True` ist seit yfinance 0.2 der **Default** von `history()` und `download()`.
Damit gilt:

- Es gibt **keine** `Adj Close`-Spalte mehr. `Close` ist bereits split- und
  dividendenbereinigt.
- Aelterer Beispielcode aus dem Netz, der `Adj Close` liest, bricht hier.

`PriceSeries.adjusted` speichert diese Entscheidung explizit mit. Fuer Backtests ist die
Unterscheidung entscheidend: bereinigte Kurse unterstellen die Wiederanlage von Dividenden.
Der Default des Projekts ist `BTLITE_DATA_AUTO_ADJUST_PRICES=true`.

### Statements sind transponiert

yfinance liefert Statements mit **Zeilen = Positionen** und **Spalten = Perioden**. Das ist
die Transponierte der ueblichen Zeitreihen-Erwartung.

Das Projekt behaelt diese Orientierung im Vertrag `FinancialStatement` bei, weil eine
Position damit ueber die Zeit eine Zeile bleibt. Die Spaltenachse traegt tz-aware
UTC-Zeitstempel und ist aufsteigend sortiert.

Relevante Properties:

| Property | Basis |
|----------|-------|
| `income_stmt`, `balance_sheet`, `cash_flow` | jaehrlich |
| `quarterly_income_stmt`, `quarterly_balance_sheet`, `quarterly_cashflow` | quartalsweise |
| `ttm_income_stmt`, `ttm_cashflow` | Trailing Twelve Months |

### Uneinheitliche Zeilenbezeichner

Dieselbe Position heisst je nach Emittent unterschiedlich: `Total Revenue`, `TotalRevenue`,
`Revenues`. Die Normalisierung auf `LineItem` passiert **genau einmal**, in
`analytics/statements.py`. Nicht gemappte Rohzeilen bleiben mit Originalnamen im Payload
und werden in `unmapped_line_items` gelistet, damit nichts still verloren geht.

### Vorzeichenkonvention

Verbindlich fuer das gesamte Projekt:

- **Aufwaende in der GuV**: positiv gespeichert.
- **Cashflow-Abfluesse**: negativ gespeichert, so wie der Provider sie liefert. CapEx ist
  also eine negative Zahl. Wer Free Cash Flow rechnet, addiert CapEx, subtrahiert es nicht.

### Fiskaljahr-Versatz

Nicht jedes Unternehmen schliesst zum 31. Dezember ab. Apple etwa endet Ende September.
`FinancialStatement.fiscal_year_end_month` haelt den Monat fest. Peer-Vergleiche ueber
Unternehmen mit verschiedenen Geschaeftsjahren sind dadurch nur eingeschraenkt
periodengenau. Das UI weist darauf hin, statt es zu verschweigen.

### Fehlende Werte

Fehlende Kennzahlen sind **immer** explizit `None`, nie ein stiller Nullwert und nie ein
NaN-Fallback. `Ratio.inputs_missing` nennt die fehlenden Eingangspositionen. Das UI rendert
das als "n/a".

### Rate Limiting

yfinance nutzt intern `curl_cffi` mit Browser-Impersonation. Ab Version 0.2.52 gibt es
`YFRateLimitError`. Praktische Grenze sind rund 950 Symbole am Stueck, danach folgt HTTP 429.

Das Projekt drosselt konservativ: 2 Requests pro Sekunde, Token-Bucket-Kapazitaet 5,
exponentieller Backoff mit Jitter. Konfigurierbar unter `BTLITE_RATELIMIT_*`.

Bekannter Fallstrick: ein zu altes `curl_cffi` wirft `ImpersonateError: chrome136 not
supported`. Dann `curl_cffi` mit aktualisieren.

### Quartalszahlen-Termine

Quelle ist `yf.Ticker(sym).calendar`, ein Dictionary. Relevant ist der Schluessel
`Earnings Date`. Gegen yfinance 1.7 am **24. September 2026** an **78 Symbolen** gemessen
(US-Aktien, europaeische und asiatische Listings, ETFs, Krypto, Index, Fantasiesymbol):

| Fall | Beispiel | `calendar` | `Earnings Date` |
|------|----------|------------|-----------------|
| Aktie mit Termin | AAPL, MSFT, SAP.DE, RKLB, ASML.AS | gefuellt | `[datetime.date(2026, 10, 29)]` |
| Aktie ohne Termin | NESN.SW, OR.PA | gefuellt | `[]` |
| ETF | SPY, QQQ, GLD, VWRL.AS | `{}` | Schluessel fehlt |
| Krypto | BTC-USD, ETH-USD | `{}` | Schluessel fehlt |
| Index | ^GSPC | `{}` | Schluessel fehlt |
| Unbekanntes Symbol | ZZZZNOTREAL | `{}` | Schluessel fehlt |

**`calendar` ist nie `None`.** Bei einem ETF laeuft intern ein HTTP 404
(`No fundamentals data found for symbol: SPY`); `yfinance` faengt ihn ab und liefert ein
**leeres Dictionary**. Im Quelltext von `scrapers/quote.py` ist das explizit:
`if result is None: self._calendar = {}`. Der Code behandelt `None` trotzdem gleichwertig,
falls eine kuenftige Version das aendert.

#### Drei Lagen, nicht zwei

Ein 404 bei einem ETF ist **kein Fehler, sondern die richtige Antwort**: ein ETF hat keine
Quartalszahlen. Er schlaegt deshalb nicht als `ProviderUnavailableError` hoch. `None` als
Datum ist aber mehrdeutig, und der Unterschied ist im UI sichtbar. Der Vertrag
`EarningsDate` trennt ihn ueber `has_earnings`:

| Lage | `has_earnings` | `next_date` | UI |
|------|----------------|-------------|-----|
| Termin bekannt | `True` | Datum | das Datum |
| Quelle belegt: fuehrt keine Earnings (ETF, Krypto, Index) | `False` | `None` | Strich |
| Termin unbekannt (Aktie ohne Ankuendigung, kein Fixture, Abruf misslungen) | `True` | `None` | "n/a" |

`has_earnings=False` setzt einen **Befund** voraus: die Quelle hat geantwortet und fuehrt
zu diesem Instrument keine Fundamentaldaten. Blosses Nichtwissen - offline kein Fixture,
ein defektes JSON, ein fehlgeschlagener Abruf - ergibt `True`. Das ist die schwaechere und
damit ehrlichere Aussage; andernfalls bekaeme ein vertippter Ticker den Strich fuer
"gibt es nicht" statt "n/a".

#### Mehrere Termine

Yahoo fuehrt laut Doku bis zu zwei Eintraege, wenn der Termin unbestaetigt ist - sie
spannen dann ein Zeitfenster auf. **In allen 78 gemessenen Symbolen kam nie mehr als ein
Datum vor.** Der Fall wird trotzdem behandelt, weil das Schema ihn zulaesst: es gilt der
**frueheste kuenftige** Termin. Begruendung: gefragt ist der *naechste* Termin; der
spaetere Rand eines Fensters waere zu optimistisch und wuerde dem Nutzer suggerieren, er
habe mehr Zeit als tatsaechlich.

#### Bestaetigt oder geschaetzt

**Yahoo kennzeichnet nicht, ob ein Termin bestaetigt oder geschaetzt ist.** Es gibt kein
Feld dafuer. `EarningsDate.estimated` steht deshalb immer auf `True`: das Feld haelt die
Unsicherheit im Vertrag fest, statt sie zu verschweigen, und das UI weist darauf hin. Ein
Termin ist eine Orientierung, kein Datum, auf das man sich verlassen kann - Unternehmen
verschieben ihn.

#### Kein Termin in der Vergangenheit

Die Quelle liefert durchaus abgelaufene Termine: **MC.PA lieferte am 24.09.2026 den
27.07.2026**, also ein Datum zwei Monate in der Vergangenheit. Ein solches Datum als
"naechsten Termin" anzuzeigen waere eine Falschaussage. Liegt der juengste gelieferte
Termin vor dem heutigen Tag, ist `next_date` deshalb `None`. Ein Termin **am** heutigen
Tag zaehlt noch als naechster.

#### Einheiten und Zeitzone

`next_date` ist ein **Kalendertag** (`datetime.date`), ohne Uhrzeit und ohne Zeitzone. Die
Quelle nennt zwar eine Uhrzeit, sie ist aber der Boerse des Emittenten zugeordnet und fuer
die Anzeige eines Termins ohne Belang; eine Umrechnung nach UTC koennte den Kalendertag
sogar um einen Tag verschieben. Das ist die bewusste Ausnahme von der UTC-Regel, die fuer
**Zeitreihen** gilt - ein Termin ist keine Zeitreihe.

#### Abruf und Drosselung

Es gibt **keinen Sammelendpunkt** fuer `calendar`; `get_earnings_dates` fragt je Symbol
einzeln. Bei 20 Depotpositionen sind das 20 Requests - genau dafuer ist der Token-Bucket
da, jeder Abruf entnimmt einen Token. Ein `RateLimitError` beendet den Sammelabruf und
geht nach oben; er wird nie umgangen und nie zu "kein Termin" verharmlost. Der Ausfall
eines einzelnen Symbols laesst die uebrigen dagegen intakt.

Die TTL des Caches betraegt **24 Stunden** - ein Quartalstermin bewegt sich selten.
Zwischengespeichert wird der **Rohbefund**, nicht der fertige "naechste Termin": der
Vergangenheitsabgleich laeuft bei jedem Zugriff neu. Sonst gaebe ein 23 Stunden alter
Cache-Eintrag nach Mitternacht einen bereits abgelaufenen Termin als naechsten aus.

#### Fixtures veralten kontrolliert

`tests/fixtures/earnings/<TICKER>.json` haelt denselben Rohbefund:

```json
{"ticker": "AAPL", "has_earnings": true, "dates": ["2026-10-29"]}
```

Bewusst **nicht** ein vorberechnetes "naechstes Datum". Ist der eingefrorene Termin
verstrichen, liefert der Provider `next_date=None` statt eines Termins in der
Vergangenheit - ohne dass jemand die Datei anfassen muss. Die Tests gehen davon nicht
kaputt: sie erzeugen ihre Termine relativ zu `utc_today()` und pruefen die ausgelieferten
Fixtures nur auf `has_earnings`, nicht auf ein festes Datum.

`SPY` ist als Fixture bewusst dabei, obwohl es nichts enthaelt: ohne ihn waere der
ETF-Fall offline gar nicht pruefbar, und ob ein 404 als Fehler hochschlaegt, fiele erst
live auf.

### Intraday-Kerzen

Die Seite Price waehlt je Zeitraum automatisch die **kleinste** Aufloesung, die die Quelle
liefert. Live gemessen am 26.09.2026 (AAPL, Kerzen je Abruf):

| Zeitraum | 1m | 5m | 15m | 30m | 1h |
|---|---|---|---|---|---|
| 1d | 390 | 78 | 26 | 13 | 7 |
| 5d | 1950 | 390 | 130 | 65 | 35 |
| 1mo | - | 1716 | 572 | 286 | 154 |
| 3mo bis 2y | - | - | - | - | 448 bis 3487 |
| 3y | - | - | - | - | - |

Daraus: 1D und 1W in 1-Minuten-Kerzen, 1M in 5 Minuten, 3M bis 1Y stuendlich, ab 3Y
taeglich. Ausserhalb dieser Grenzen liefert Yahoo eine **leere Reihe statt eines Fehlers**;
die Zuordnung steht deshalb fest in `analytics.series.CHART_INTERVALS` und wird nie
ausprobiert.

- Nur regulaere Handelszeit (`prepost=False`, Default von yfinance), Verzoegerung wie bei
  allen Kursen der Quelle (etwa 15 Minuten).
- Zwischengespeichert **60 Sekunden** (`cache.intraday_ttl_s`), nicht eine Stunde wie
  Tageskerzen, und nicht im Speicherpuffer der Oberflaeche, der keine Ablaufzeit kennt.
- Intern UTC wie alles. Erst die Chartachse zeigt die Anzeigezeitzone
  (`ui.display_timezone`), Naechte und Wochenenden sind dort ausgeblendet.
- Offline liegen nur Tageskerzen vor; die Seite sagt das und zeichnet taeglich.

### Stammdaten (`Ticker.info`)

Name, Sektor, Branche, Land, Dividende, Marktkapitalisierung und das
Fiskaljahresende kommen aus `Ticker.info`. Die Antwort wird **4 Stunden**
zwischengespeichert (`cache.info_ttl_s`), als Rohbefund, nur bei Erfolg. Fehler und
`RateLimitError` werden nicht gepuffert.

Entscheidung vom 26.09.2026. Ohne Puffer lud jedes Screening die Stammdaten jedes Titels
zweimal neu; ein Folgelauf ueber den S&P 500 dauerte gemessen 8.6 Minuten, so lange wie
der erste. Mit Puffer sind es rund 20 Sekunden.

Folge fuer die Semantik: die **Marktkapitalisierung** und alle daraus abgeleiteten
Kennzahlen koennen bis zu 4 Stunden alt sein, also hoechstens einen halben Handelstag.
Der **Kurs** selbst ist davon nicht betroffen: er kommt weiter aus dem Quote-Abruf
(15 Minuten Verzoegerung der Quelle).

## Finviz

### Nutzungsrahmen

`finvizfinance` ist ein HTML-Scraper. Das Projekt bleibt im Rahmen privater Nutzung mit
konservativem Throttling: 0.5 Requests pro Sekunde, Token-Bucket-Kapazitaet 2. Es gibt
**keine** Umgehung von Rate Limits, Paywalls oder Bot-Erkennung. Wenn die Quelle blockt,
wird der Fehler als `RateLimitError` sauber nach oben gegeben.

### Filter-Notation

Filter-Keys werden gegen ein internes Dictionary validiert, exakt inklusive Punkt und
Gross-/Kleinschreibung. `Market Cap.` mit Punkt ist gueltig, `Market Cap` ohne Punkt wirft
einen Fehler.

Beispiele fuer Keys: `Exchange`, `Index`, `Sector`, `Industry`, `Country`, `Market Cap.`,
`P/E`, `Forward P/E`, `PEG`, `P/S`, `P/B`, `Dividend Yield`, `Average Volume`, `Price`,
`Beta`, `RSI (14)`, `Return on Equity`.

Zwei Stolperstellen, gegen das installierte Paket verifiziert:

- Der RSI-Key heisst **`RSI (14)`**, nicht `RSI`. Ohne die Klammer wirft das Paket einen
  Fehler.
- **Werte sind die vollen Anzeigetexte**, keine Kurzformen. `Market Cap.` nimmt
  `Mega ($200bln and more)`, `Large ($10bln to $200bln)`, `Mid ($2bln to $10bln)`,
  `Small ($300mln to $2bln)`. Ein blosses `Mega` wird abgelehnt.

Weitere Werte: `Index` nimmt `S&P 500`, `NASDAQ 100`, `DJIA`, `RUSSELL 2000`. `P/E` nimmt
`Low (<15)`, `Profitable (>0)`, `High (>50)`, `Under 5`.

`ScreenerProvider.available_filters()` liest Keys und Werte zur Laufzeit aus dem Paket aus.
Nutze das, statt Werte im Code fest zu verdrahten.

### Views

Verfuegbar sind `Overview`, `Valuation`, `Financial`, `Ownership`, `Performance`,
`Technical` und `Ticker`. Alle teilen dieselbe Methodensignatur und unterscheiden sich nur
in den gelieferten Spalten.

### Fallstricke

- Der Default `sleep_sec=1` bedeutet: ein grosses `limit` dauert lang, etwa eine Sekunde je
  Seite zu rund 20 Zeilen. `ScreenRequest.limit` ist deshalb auf 200 vorbelegt.
- Als HTML-Scraper bricht die Bibliothek bei Layoutaenderungen von Finviz. Solche Fehler
  erscheinen als `DataValidationError`, nicht als stilles leeres Ergebnis.

## Fixtures

`data/fixtures.py` liefert eingefrorene Antworten beider Quellen und geht **nie** ins
Netz. Er erfuellt dieselben Protocols wie die echten Provider, ohne von ihnen zu erben —
was gegen ihn laeuft, laeuft auch gegen die echte Quelle.

### Der ausgewiesene Verzug ist das echte Alter

Dies ist die wichtigste Zusage dieses Abschnitts. Der `FixtureProvider` weist **nicht** die
15 Minuten einer Live-Quelle aus, sondern den tatsaechlichen Abstand zwischen dem
Einfrierzeitpunkt und jetzt:

- `Provenance.source` ist `DataSource.FIXTURE`, `from_cache` ist `True`.
- `delay_minutes` ist der echte Abstand in Minuten, berechnet aus `manifest.json`.
- `note` traegt zusaetzlich das Stand-Datum im Klartext, etwa `Stand 2026-09-04`.

Fixture-Preise sind Tage bis Monate alt. Sie als viertelstuendlich verzoegert auszuweisen
waere eine Falschangabe, und zwar eine, die jemand fuer bare Muenze nehmen koennte. Der
Zeitstempel faellt auf die Mtime des Verzeichnisses zurueck, wenn kein Manifest lesbar
ist, und zuletzt auf `utc_now()`.

### Umfang

Erzeugt von `scripts/refresh_fixtures.py`, Stand `2026-09-04`, gezogen gegen yfinance
1.7.0 und finvizfinance 1.5.0:

| Ablage | Inhalt |
|--------|--------|
| `prices/<TICKER>_<interval>.parquet` | 10 Ticker, Tagesbars ueber 5 Jahre, UTC-Index |
| `statements/<TICKER>_<typ>_<periode>.parquet` | 9 Ticker mal 3 Typen mal 3 Perioden |
| `info/<TICKER>.json` | 10 Stammdatensaetze aus `Ticker.info` |
| `screener/overview.parquet` | ein Finviz-Screener-Ergebnis |
| `earnings/<TICKER>.json` | 12 Termin-Rohbefunde, inkl. des ETF-Falls SPY |

Das Universum: `AAPL`, `JNJ`, `JPM`, `KO`, `MSFT`, `NVDA`, `PG`, `SPY`, `WMT`, `XOM`.

`SPY` ist ein ETF und fuehrt keine Statements — daher 9 statt 10 Statement-Saetze. Das ist
die Natur des Instruments und kein fehlender Datensatz; `DES` stellt den Kennzahlenblock
dort zurueckgenommen dar statt als Fehler.

Das Universum wurde in Phase 2b bewusst von 3 auf 9 Statement-Ticker erweitert: mit nur
drei teuren Qualitaetstiteln findet ein Value-Screen per Definition nichts, und ein leeres
Ergebnis haette wie ein Fehler ausgesehen.

### Erneuern

```bash
.venv\Scripts\python.exe scripts\refresh_fixtures.py
.venv\Scripts\python.exe scripts\refresh_fixtures.py --only prices
.venv\Scripts\python.exe scripts\refresh_fixtures.py --tickers AAPL,MSFT
```

Das ist der einzige Weg, und er braucht Netz — es ist das **einzige Stueck Code im
Repository, das absichtlich ins Netz geht**, und es laeuft nie automatisch in der
Testsuite. Das Skript schreibt `manifest.json` mit `last_run` und den verwendeten
Paketversionen neu; davon haengt der ausgewiesene Verzug ab. Nach dem Lauf gehoert das
Abrufdatum zusaetzlich in `tests/fixtures/README.md`.

## Zeitzonen

Alle Zeitreihen im Projekt tragen einen tz-aware `DatetimeIndex` in **UTC**. Konvertiert
wird ausschliesslich an der UI-Grenze, gesteuert ueber `BTLITE_UI_DISPLAY_TIMEZONE`
(Default `Europe/Berlin`). Provider liefern teils naive Zeitstempel; `core/clock.py`
interpretiert diese als UTC.

## Waehrungen

`PriceSeries`, `Quote` und `FinancialStatement` tragen jeweils ihre eigene `currency`. In
Analytik, Screening und Backtest gibt es **keine** automatische Umrechnung. Ein
Peer-Vergleich ueber Waehrungsgrenzen hinweg vergleicht deshalb waehrungsbehaftete
Kennzahlen wie Marktkapitalisierung nicht direkt. Verhaeltniszahlen wie ROE oder Current
Ratio sind davon nicht betroffen.

### Ausnahme: Depotbewertung

Die einzige Stelle mit Umrechnung ist das Depot. Ein Depot hat genau eine
Berichtswaehrung; Positionen duerfen in einer anderen Waehrung gekauft worden sein. Ohne
Umrechnung waere die Summe ueber solche Positionen eine Addition ungleicher Einheiten -
also eine Falschaussage ueber das Vermoegen.

**Die Standardwaehrung ist Euro.** Sie gilt fuer neue Depots, fuer den Kaufkurs einer
neuen Position und fuer das Backtest-Konto - ueberall dort, wo der *Nutzer* die Waehrung
waehlt. Modelle, die beschreiben, was eine **Quelle** geliefert hat (`Quote`,
`PriceSeries`, `Instrument`, die Statements), behalten USD als Vorgabe: dort waere ein
Euro-Default eine Falschangabe ueber fremde Daten, denn yfinance liefert die Waehrung
mit und sie ist bei US-Titeln nun einmal Dollar.

Der haeufigste Fall braucht damit gar keine Umrechnung mehr: ein Euro-Depot mit Titeln,
die in Euro notieren (`SAP.DE` etwa), kommt ohne einen einzigen Kursabruf aus - auch
offline.

**Kurse kommen von yfinance** ueber das Symbolschema `<VON><NACH>=X`, etwa `EURUSD=X`.
Gegen die Quelle geprueft am 23. September 2026:

| Eigenschaft | Befund |
|-------------|--------|
| Historie | ueber `history(period=...)`, rund 260 Handelstage je Jahr |
| Index-Zeitzone | `Europe/London`, **nicht** UTC - `core/clock.py` normalisiert |
| Wochenenden | keine Zeile; es gilt der letzte Handelstag davor |
| Richtung | `EURUSD=X` ist USD je EUR (1.1405 am 23.09.2026), also `USD = EUR * Kurs` |

**Der Kurs des Kauftages zaehlt, nicht der heutige.** Der Einstand wird einmal zum Kurs
des Kauftages umgerechnet und bleibt danach fest; der Marktwert nutzt den aktuellen Kurs.
Die ausgewiesene Rendite enthaelt damit den Waehrungseffekt - das ist die Rendite, die ein
Anleger in der Berichtswaehrung tatsaechlich erzielt hat. Wuerde man beide Seiten zum
heutigen Kurs rechnen, aendere sich der historische Einstand mit jedem Tag.

**Fehlt ein Kurs, bleibt der Betrag `None`.** Offline, bei nicht erreichbarer Quelle oder
fuer ein Waehrungspaar ohne Notierung wird nicht geschaetzt und nicht mit 1.0 gerechnet.
Die Position erscheint als nicht bewertbar, und die Summe wird - wie bei einem fehlenden
Aktienkurs auch - gar nicht erst ausgewiesen.

Umgerechnet wird ausschliesslich in `portfolio/`. Ein Kurs ist ein
**Marktkurs, kein Abrechnungskurs**: die Bank rechnet mit eigenem Aufschlag ab, und
Steuerrecht verlangt je nach Land andere Stichtage. Die Anzeige ist eine Orientierung,
keine steuerliche Bemessungsgrundlage.

## Was diese Quellen nicht leisten

- Keine Echtzeitkurse, kein Level-2-Orderbuch, keine Tick-Daten.
- Keine garantierte Vollstaendigkeit oder Richtigkeit. Yahoo korrigiert Historien
  rueckwirkend, ohne das anzukuendigen.
- Keine Delisting-bereinigten Universen. Screening-Ergebnisse unterliegen einem
  Survivorship Bias, Backtests auf selbst zusammengestellten Universen ebenso.

## Portfolio look-through

Die PORT-Zusammensetzung bevorzugt die iShares-Produktliste und deren
`holdings.all`-Datenschnittstelle. Ein ETF wird über eine eindeutige ISIN
oder über Fondsnamen plus verifizierte Börsen- und Tickerzuordnung identifiziert.
Die Antwort enthält ISIN, Anlageklasse, Land, Sektor, Gewicht und das
Bestandsdatum. Die Produktliste wird sieben Tage, die Bestände einen Tag
lokal gecacht. Der originale Abrufzeitpunkt bleibt erhalten.

Optional kann `BTLITE_DATA_EODHD_API_TOKEN` gesetzt werden. Der EODHD-Adapter
ruft `ETF_Data.Holdings` ab, wenn kein iShares-Bestand verfügbar ist. Die
API-Gewichte in Prozent werden in dezimale Fondsanteile umgerechnet. Ein
fehlender Rest bleibt unbekannt. Ein Bestandsdatum wird von dieser Antwort
nicht garantiert. Ohne Token wird EODHD nicht kontaktiert.

Als letzter automatischer Rückfall dient `yfinance.Ticker.funds_data.top_holdings`.
`Holding Percent` wird als dezimaler Anteil am gesamten Fonds übernommen.
Die Quelle liefert keine garantierte vollständige Bestandsliste und kein
verlässliches Bestandsdatum. Der Cache behält den ursprünglichen Abrufzeitpunkt
und nutzt die Stammdaten-TTL. Die UI zeigt diesen als Abrufzeit, nicht Bestandsdatum.

Für direkte Aktienpositionen stammen Land und Sektor aus Instrument-Stammdaten.
Bei iShares-Beständen ist Land das gemeldete Risikoland (`countryOfRisk`), bei
Direktaktien der Unternehmenssitz. Regionen sind geografische Gruppen einer expliziten
Länderliste; nicht zugeordnete Länder bleiben in der Regionenansicht unbekannt.
ETF-Domizile werden nicht als Anlageexposure verwendet. Verschachtelte Fonds,
nicht unterstützte Anlageklassen und fehlende Bestände bleiben unaufgeschlüsselt.
Negative Cash- oder Derivategewichte der iShares-Bestände werden mit Vorzeichen
dem unbekannten Nettoanteil zugerechnet. Dadurch können bekannte Aktiengewichte
über 100 Prozent liegen, während die Nettosumme weiterhin 100 Prozent beträgt.
Top-Positionen werden niemals auf 100 Prozent normalisiert. Sektoranteile werden
aus den aufgeschlüsselten Aktien abgeleitet, nicht aus Yahoo-Sektoraggregaten,
deren Bezugsbasis für gemischte Fonds abweichen kann.

Ein optionaler Nutzer-CSV-Import ersetzt ETF-Bestände für die aktuelle Seitenansicht.
Er unterstützt Aktienbestände mit Prozentgewichten am Gesamtfonds. Doppelte Symbole,
negative/nicht endliche Gewichte und Summen über 100 Prozent werden abgewiesen.
Quelle ist `user`, Zeitstempel ist Importzeit; Bestandsdatum und Klassifikationen
liegen in der Verantwortung des Nutzers. Kein Netzwerkzugriff durch den CSV-Parser.

Portfolioanteile = umgerechneter aktueller Positionswert / gesamter Depotwert ×
Bestandsanteil im Fonds. Fehlende Bewertung verhindert die Gesamtanalyse. Beiträge
gleicher ISINs werden addiert. Falls die ISIN fehlt, wird nur bei eindeutigem
normalisiertem Firmennamen und gleichem gemeldeten Land zugeordnet, sonst nach Symbol.
`yfinance.Ticker.isin` ist experimentell und kann `-` liefern; ungültige ISINs
werden verworfen. Unterschiedliche Aktiengattungen bleiben ohne gemeinsame ISIN
getrennt. Die Zuordnung behauptet keine Deckungsgleichheit verschiedener
Derivate oder Indexbeteiligungen.
