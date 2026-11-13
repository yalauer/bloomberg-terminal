# Kommandoreferenz

Die Kommandozeile steht oben in der Kopfleiste und hat beim Start den Fokus. Enter löst
aus. Alle hier gezeigten Eingaben sind gegen den Parser in
[`src/btlite/ui/command_bar.py`](../src/btlite/ui/command_bar.py) geprüft.

## Eingabeformat

```
<TICKER[,TICKER...]> <MNEMONIC> [ARGUMENT...] [GO]
```

Die Regeln des Parsers, der Reihe nach:

1. **Das Mnemonic ist das erste Token, das eines ist.** Alles davor sind Ticker, alles
   danach sind Argumente. Daraus folgt: `AAPL GP` funktioniert, `GP AAPL` nicht — dort
   stünde kein Ticker vor dem Mnemonic und die Eingabe wird abgelehnt.
2. **Groß- und Kleinschreibung ist egal.** `aapl fa go` ist identisch zu `AAPL FA GO`.
3. **Beliebig viele Leerzeichen** zwischen den Token sind erlaubt.
4. **Trenner der Tickerliste** sind Komma, Semikolon und Leerzeichen. `AAPL,MSFT`,
   `AAPL;MSFT` und `AAPL MSFT` sind gleichwertig.
5. **Duplikate werden stabil entfernt.** `AAPL,AAPL,MSFT` ergibt `AAPL, MSFT`.
6. **Das abschließende `GO` ist optional** und wird verworfen. Es bleibt zulässig, weil es
   der Bedienung eines Bloomberg-Terminals entspricht. `GO` allein ist kein Kommando.

Ein gültiges Kommando wird in eine Adresse übersetzt: `AAPL,MSFT FA ttm GO` wird zu
`/fa?t=AAPL%2CMSFT&a=ttm`. Die Adresse ist damit teilbar und über den Browser-Zurückknopf
navigierbar.

## Überblick

| Mnemonic | Seite | Ticker | Argument |
|----------|-------|--------|----------|
| `DES` | Description | genau einer wird genutzt | keines |
| `GP` | Graph Price | einer oder mehrere | Zeitraum |
| `FA` | Financial Analysis | erster ist Subjekt, weitere sind Peers | Periodenbasis |
| `EQS` | Equity Screening | optional, bilden das Universum | Preset-Key |
| `BT` | Backtest | einer oder mehrere | Strategie-Key |
| `PORT` | Portfolio & Risk | mindestens zwei | Zielfunktions-Key |

**`EQS` ist das einzige Mnemonic, das ohne Ticker auskommt.** Bei allen anderen meldet
eine tickerlose Eingabe einen Fehler: `PORT` allein ergibt „PORT braucht mindestens einen
Ticker, etwa AAPL PORT GO."

Ein nicht erkanntes Argument wird **stillschweigend ignoriert** und der jeweilige Default
verwendet. Der Parser reicht alle Argumente durch; erst die Seite entscheidet, welche sie
versteht.

---

## `DES` — Description

Stammdaten, aktueller Quote und ein Kennzahlen-Überblick. Genutzt wird der **erste**
genannte Ticker; weitere werden ohne Meldung verworfen.

```
AAPL DES GO
msft des
```

**Argumente:** keine. Die Seite ignoriert alles hinter dem Mnemonic.

Über die Kommandozeile verlangt `DES` einen Ticker. Wird die Adresse `/des` dagegen direkt
aufgerufen, greift die Vorbelegung aus `BTLITE_UI_DEFAULT_TICKERS` (Default
`AAPL, MSFT, NVDA`) und es erscheint deren erster Eintrag.

Gezeigt werden: Kurs, Veränderung, Tagesspanne und Volumen; Name, Börse, Assetklasse,
Währung, Sektor und Branche; dazu KGV, KBV, KUV, Dividendenrendite, Eigenkapitalrendite,
Nettomarge, Current Ratio und Verschuldungsgrad.

Fonds und Indizes (etwa `SPY`) führen keine Jahresabschlüsse. Der Kennzahlenblock bleibt
dort bewusst leer und erscheint als zurückgenommener Hinweis, nicht als Fehlerbanner —
das ist die Natur des Instruments, kein Datenfehler.

---

## `GP` — Graph Price

Kursverlauf mit Volumen, technischen Overlays und Mehr-Ticker-Vergleich.

```
AAPL GP GO
AAPL,MSFT,NVDA GP 5y GO
```

**Argument: der Zeitraum**, in yfinance-Notation. Erkannt wird das erste passende
Argument:

| Wert | Bedeutung |
|------|-----------|
| `1mo` | ein Monat |
| `3mo` | drei Monate |
| `6mo` | sechs Monate |
| `1y` | ein Jahr (**Default**) |
| `3y` | drei Jahre |
| `5y` | fünf Jahre |
| `max` | gesamte verfügbare Historie |

`AAPL GP 10y GO` ist kein Fehler — `10y` steht nicht in der Liste, wird ignoriert und die
Seite zeigt `1y`. Der Zeitraum lässt sich danach jederzeit im Auswahlfeld ändern.

**Nur über die Bedienelemente, nicht über die Kommandozeile** einstellbar sind:

- Chart-Art: Kerzen (Default), OHLC, Linie.
- Overlays: SMA 20/50/200 (Default), EMA 12/26, Bollinger, RSI, MACD.

Bei **mehreren Tickern** erscheint zusätzlich ein Vergleichschart, auf Basis 100 normiert.
Die Reihen werden dafür auf ihre gemeinsame Schnittmenge gebracht, nicht vorwärtsgetragen:
ein später startender Titel ginge sonst mit erfundenen konstanten Kursen ein.

---

## `FA` — Financial Analysis

Gewinn- und Verlustrechnung, Bilanz, Kapitalflussrechnung, abgeleitete Kennzahlen und
Peer-Vergleich. Der **erste** Ticker ist das Subjekt, alle **weiteren** sind die
Peer-Gruppe.

```
AAPL FA GO
AAPL,MSFT,GOOGL FA ttm GO
```

**Argument: die Periodenbasis.**

| Wert | Bedeutung |
|------|-----------|
| `annual` | Geschäftsjahre (**Default**) |
| `quarterly` | Quartale |
| `ttm` | Trailing Twelve Months |

Zwei Hinweise zur Auslegung der Zahlen:

- **Vorzeichen:** Aufwände in der GuV sind positiv gespeichert, Cashflow-Abflüsse negativ.
  CapEx ist damit eine negative Zahl und wird zum Free Cash Flow addiert, nicht subtrahiert.
- **Fiskaljahr-Versatz:** Nicht jedes Unternehmen schließt zum 31. Dezember ab, Apple etwa
  Ende September. Der Peer-Vergleich weist abweichende Geschäftsjahre und abweichende
  Währungen sichtbar aus, statt eine unvergleichbare Matrix stumm auszugeben.

Ein nicht berechenbarer Wert erscheint als „n/a", nie als Null. Dahinter steht, welche
Eingangsposition gefehlt hat.

---

## `EQS` — Equity Screening

Screening gegen ein Kennzahlen-Universum, mit vier Presets und CSV-Export. Das einzige
Mnemonic ohne Tickerpflicht.

```
EQS GO
EQS quality GO
AAPL,MSFT,JNJ,KO,PG EQS value GO
```

**Ticker sind optional und bilden das Universum.** Ohne Ticker wird alles genommen, was
die Quelle führt — im Offline-Default sind das die zehn Fixture-Titel.

**Argument: der Preset-Key.**

| Key | Preset | Auswahl |
|-----|--------|---------|
| `value` | Value | Günstig bewertete, profitable Unternehmen mit niedrigem KGV und KBV (**Default**) |
| `quality` | Quality | Hohe Kapitalrendite bei geringer Verschuldung und solider Liquidität |
| `momentum` | Momentum | Titel im Aufwärtstrend über der 200-Tage-Linie, noch nicht überkauft |
| `dividend` | Dividend | Tragfähige Ausschüttungen: Rendite über 3 Prozent, Quote unter 80 Prozent |

Die angewandten Filter werden über der Trefferliste ausgeschrieben, die Ergebnisse lassen
sich als CSV herunterladen.

**Zur NaN-Semantik der Filter:** Ein fehlender Wert erfüllt **kein** Vergleichsprädikat,
auch `NEQ` nicht. `NEQ` heißt „bekannt und ungleich" und ist damit nicht die Negation von
`EQ`. Nur `IS_NULL` trifft Fehlwerte. Ein Titel ohne KGV fällt aus einem KGV-Filter heraus,
statt ihn zufällig zu bestehen.

Screening-Ergebnisse unterliegen einem Survivorship Bias: das Universum enthält keine
delisteten Titel.

---

## `BT` — Backtest

Strategie-Backtest mit Equity-Kurve, Drawdown-Verlauf, Kennzahlen und Trade-Liste. Die
Kursbasis ist fest auf fünf Jahre gesetzt.

```
AAPL BT GO
AAPL,MSFT,NVDA BT momentum_rotation GO
```

**Argument: der Strategie-Key.**

| Key | Strategie | Default-Parameter |
|-----|-----------|-------------------|
| `buy_and_hold` | Buy and Hold (**Default**) | keine |
| `momentum_rotation` | Momentum Rotation | `lookback=126`, `top_n=3`, `rebalance_every=21` |
| `rsi_mean_reversion` | RSI Mean Reversion | `period=14`, `oversold=30.0`, `overbought=70.0` |
| `sma_cross` | SMA Crossover | `fast=50`, `slow=200` |

Parameter lassen sich direkt in der Kommandozeile setzen, als `name=wert`-Paare hinter
dem Strategie-Key:

```
SPY BT sma_cross fast=10 slow=30 GO
AAPL BT rsi_mean_reversion period=21 oversold=25 GO
```

Nicht genannte Parameter behalten ihren Default. Dasselbe Format nimmt auch das Feld
„Parameter" auf der Seite entgegen, getrennt durch Leerzeichen oder Komma:

```
fast=20 slow=100
period=21, oversold=25
```

Werte werden nach `int`, dann `float`, sonst als Text übernommen. Ein unbekannter Name
bleibt erhalten und wird von der Strategie selbst als `StrategyConfigError` abgelehnt.

**Zum Look-ahead-Bias:** Ein Signal in Zeile `t` nutzt ausschließlich Daten bis
einschließlich `t`. Die Ausführung erfolgt verzögert, geregelt über
`SignalSet.execution_lag`, Default ein Bar. Signal und Ausführung sind bewusst getrennt:
die Strategie kennt weder Engine noch Kosten noch Kapital.

Kosten sind mit je 5 Basispunkten für Kommission und Slippage vorbelegt
(`BTLITE_BACKTEST_FEES`, `BTLITE_BACKTEST_SLIPPAGE`), jeweils bei Ein- und Ausstieg
angewandt.

Die Engine `vectorbt` gehört zum optionalen Extra `[backtest]`. Fehlt sie, meldet die
Seite das, statt zu scheitern. Der `backtrader`-Adapter ist ein Strukturnachweis und nicht
erprobt.

---

## `PORT` — Portfolio & Risk

Gewichtsoptimierung, Effizienzgrenze, Korrelationsmatrix und Risikometriken. Schätzbasis
sind drei Jahre.

```
AAPL,MSFT,KO PORT GO
AAPL,MSFT,JNJ,KO,PG,XOM PORT min_volatility GO
```

**Mindestens zwei Ticker.** Bei nur einem fällt die Seite stillschweigend auf die
Vorbelegung zurück — ein Portfolio aus einem Titel ist keines.

**Argument: die Zielfunktion.**

| Key | Zielfunktion |
|-----|--------------|
| `max_sharpe` | maximales Sharpe-Verhältnis (**Default**) |
| `min_volatility` | minimale Volatilität |
| `hrp` | Hierarchical Risk Parity |
| `equal_weight` | Gleichgewichtung |
| `max_quadratic_utility` | maximaler quadratischer Nutzen |

Die Optimierungsschicht beherrscht darüber hinaus `efficient_return`, `efficient_risk` und
`black_litterman`. **Über die Kommandozeile sind diese drei nicht erreichbar**, weil sie
zusätzliche Vorgaben brauchen (Zielrendite, Zielrisiko beziehungsweise Views), für die die
Zeile kein Format hat.

Der risikofreie Zins wird explizit gesetzt (`BTLITE_ANALYTICS_RISK_FREE_RATE`, Default
0.02) und im Seitenkopf ausgewiesen — PyPortfolioOpt hätte hier sonst einen Default von
0.0, was das Sharpe-Verhältnis stillschweigend überhöht.

Zu den Vorzeichen: Volatilitäten und Renditen sind annualisiert und dezimal. Maximaler
Drawdown, VaR und CVaR sind **negativ**, und CVaR ist stets kleiner oder gleich VaR.

---

## Tippfehler

Ein unbekanntes Mnemonic wird nicht stumm verschluckt. Der Parser vergleicht das Token
über `difflib` mit den sechs gültigen Mnemonics und schlägt bei hinreichender Ähnlichkeit
(Schwelle 0.5) das nächstliegende vor:

```
AAPL FAA GO   ->  Unbekanntes Kommando 'FAA'. Meinten Sie FA?
AAPL PRT GO   ->  Unbekanntes Kommando 'PRT'. Meinten Sie PORT?
AAPL GPP GO   ->  Unbekanntes Kommando 'GPP'. Meinten Sie GP?
```

Liegt nichts nahe genug, werden alle gültigen Mnemonics aufgezählt:

```
AAPL XYZ GO   ->  Unbekanntes Kommando 'XYZ'. Bekannt sind: DES, GP, FA, EQS, BT, PORT.
```

**Geprüft wird immer das letzte Token**, weil dort das Mnemonic steht. Bei `PRT AAPL GO`
ist der Kandidat deshalb `AAPL` und es gibt keinen brauchbaren Vorschlag, obwohl `PRT`
offensichtlich `PORT` meint. Das Mnemonic gehört hinter die Ticker.

Die Meldung erscheint als Banner unter der Kommandozeile. Die Eingabe bleibt dabei im Feld
stehen und lässt sich korrigieren; die aktuell angezeigte Seite wechselt nicht.

## Direkte Adressen

Jede Seite ist auch ohne Kommandozeile erreichbar. Die Navigationsleiste unter der
Eingabezeile verlinkt alle sechs. Ticker und Argumente lassen sich als Query-Parameter
anhängen:

```
/fa?t=AAPL,MSFT&a=ttm
/bt?t=NVDA&a=sma_cross
/eqs?a=quality
```

`t` trägt die Tickerliste, `a` die durch Leerzeichen getrennten Argumente. Ein unbekannter
Pfad fällt auf `/des` zurück.
