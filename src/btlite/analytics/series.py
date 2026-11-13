"""Umformungen mehrerer Kursreihen zu gemeinsamen Matrizen.

Zwei Aufgaben, die mehrfach gebraucht werden und in keine der bestehenden
Rechenschichten allein gehoeren:

- :func:`build_comparison` erzeugt die auf Basis 100 normierte
  Vergleichsmatrix, die :func:`btlite.viz.price.comparison` erwartet.
- :func:`portfolio_returns` gewichtet Einzelreihen zu einer
  Portfolio-Renditereihe, ohne die :mod:`btlite.analytics.risk` kein
  aggregiertes Portfolio-Risiko rechnen kann.

Beide arbeiten auf injizierten Reihen und holen keine Daten
(Architekturregel 2).

Gemeinsame Regel fuer den Index: Reihen werden auf ihre **Schnittmenge**
gebracht, nicht per Vorwaertstragen auf die Vereinigung gestreckt. Ein
Ticker, der erst spaeter notiert, wuerde sonst mit einem erfundenen
konstanten Kurs in die Rechnung eingehen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import pandas as pd

from btlite.core.errors import InsufficientDataError
from btlite.core.logging import get_logger
from btlite.models.common import (
    DATE_INDEX_NAME,
    INTRADAY_INTERVALS,
    OHLCV_COLUMNS,
    DataSource,
    Interval,
    Provenance,
)
from btlite.models.market import PriceComparison

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from btlite.models.market import PriceSeries

_LOG = get_logger(__name__)

#: Basiswert der Normierung. 100 ist die uebliche Indexbasis.
COMPARISON_BASE: Final[float] = 100.0

#: Mindestzahl gemeinsamer Beobachtungen fuer einen sinnvollen Vergleich.
MIN_COMMON_OBSERVATIONS: Final[int] = 2


def align_closes(prices: Mapping[str, PriceSeries]) -> pd.DataFrame:
    """Fuehrt die Schlusskurse mehrerer Reihen auf einen Index zusammen.

    Es gilt die **Schnittmenge** der Indizes: nur Zeitpunkte, zu denen jede
    Reihe einen Wert hat. Damit vergleicht die Matrix ausschliesslich
    tatsaechlich beobachtete Kurse.

    Args:
        prices: Kursreihen je Ticker.

    Returns:
        Wide-DataFrame, eine Spalte je Ticker, tz-aware UTC-Index.

    Raises:
        InsufficientDataError: Wenn keine oder zu wenige gemeinsame
            Beobachtungen bleiben.
    """
    if not prices:
        msg = "No ticker supplied"
        raise InsufficientDataError(msg, required=1, available=0)

    frame = pd.DataFrame({ticker: series.close for ticker, series in prices.items()})
    aligned = frame.dropna(how="any")
    aligned.index.name = DATE_INDEX_NAME

    available = len(aligned)
    if available < MIN_COMMON_OBSERVATIONS:
        tickers = ", ".join(sorted(prices))
        msg = (
            f"Too few common observations for {tickers}. The series barely overlap, or not at all."
        )
        raise InsufficientDataError(msg, required=MIN_COMMON_OBSERVATIONS, available=available)

    dropped = len(frame) - available
    if dropped:
        _LOG.debug("%d Zeitpunkte ohne Wert in allen Reihen verworfen", dropped)
    return aligned


def normalize_to_base(frame: pd.DataFrame, base: float = COMPARISON_BASE) -> pd.DataFrame:
    """Normiert jede Spalte auf einen gemeinsamen Startwert.

    Args:
        frame: Wide-DataFrame mit Kursen, eine Spalte je Ticker.
        base: Startwert, ueblich 100.

    Returns:
        Normierter Frame. Spalten, deren erster Wert 0 ist, bleiben ``NaN``:
        eine Division durch Null waere hier kein Nullwert, sondern
        undefiniert.
    """
    first = frame.iloc[0]
    safe = first.replace(0.0, pd.NA)
    return frame.div(safe, axis=1) * base


def build_comparison(
    prices: Mapping[str, PriceSeries],
    *,
    normalized: bool = True,
    interval: Interval = Interval.D1,
) -> PriceComparison:
    """Baut die Mehr-Ticker-Vergleichsmatrix fuer die Seite ``GP``.

    Normiert wird auf Basis 100 ab dem ersten **gemeinsamen** Zeitpunkt.
    Nur so beginnen alle Linien im Chart am selben Punkt und die Abstaende
    sind als relative Wertentwicklung lesbar.

    Args:
        prices: Kursreihen je Ticker, mindestens eine.
        normalized: Auf Basis 100 normieren. ``False`` liefert Rohkurse,
            die bei stark verschiedenen Kursniveaus kaum vergleichbar sind.
        interval: Kerzen-Intervall der Reihen, fuer den Vertrag.

    Returns:
        Vergleichsobjekt, das :func:`btlite.viz.price.comparison` zeichnet.

    Raises:
        InsufficientDataError: Wenn die Reihen sich nicht ausreichend
            ueberlappen.
    """
    aligned = align_closes(prices)
    data = normalize_to_base(aligned) if normalized else aligned

    return PriceComparison(
        tickers=list(prices),
        interval=interval,
        normalized=normalized,
        data=data,
        provenance=_merged_provenance(prices, normalized=normalized),
    )


def portfolio_returns(
    weights: Mapping[str, float],
    prices: Mapping[str, PriceSeries],
    *,
    rebalanced: bool = True,
) -> pd.Series:
    """Gewichtet Einzelreihen zu einer Portfolio-Renditereihe.

    Das ist die Eingabe fuer :func:`btlite.analytics.risk.compute_risk_metrics`,
    wenn das Risiko eines **Portfolios** statt eines Einzelwerts gefragt ist.

    Zwei Lesarten, die sich messbar unterscheiden:

    - ``rebalanced=True``: die Gewichte gelten in jeder Periode neu. Das
      unterstellt staendiges Zurueckgewichten und ist die uebliche Annahme
      hinter einer Optimierung, deren Gewichte konstant bleiben sollen.
    - ``rebalanced=False``: die Gewichte gelten einmal zu Beginn, danach
      laeuft das Portfolio ungestoert. Gewinner bekommen ueber die Zeit mehr
      Gewicht. Das entspricht einem Kauf-und-Halten-Depot.

    Args:
        weights: Zielgewichte je Ticker, dezimal. Muessen sich nicht exakt
            auf 1.0 summieren; sie werden auf ihre Summe normiert.
        prices: Kursreihen je Ticker. Jeder gewichtete Ticker muss
            enthalten sein.
        rebalanced: Siehe oben.

    Returns:
        Periodenrenditen des Portfolios, dezimal (0.01 = 1 Prozent), mit
        tz-aware UTC-Index. Die erste Periode fehlt, weil eine Rendite zwei
        Kurse braucht.

    Raises:
        KeyError: Wenn ein gewichteter Ticker keine Kursreihe hat.
        InsufficientDataError: Wenn die Reihen sich nicht ausreichend
            ueberlappen oder die Gewichtssumme 0 ist.
    """
    missing = sorted(set(weights) - set(prices))
    if missing:
        msg = f"No price series for {', '.join(missing)}"
        raise KeyError(msg)

    used = {t: w for t, w in weights.items() if w != 0.0}
    total = sum(used.values())
    if not used or total == 0:
        msg = "Weights sum to 0 - no portfolio can be determined"
        raise InsufficientDataError(msg, required=1, available=0)

    aligned = align_closes({t: prices[t] for t in used})
    normalized_weights = pd.Series({t: w / total for t, w in used.items()})

    if rebalanced:
        # Gewichte gelten in jeder Periode neu.
        asset_returns = aligned.pct_change().dropna(how="any")
        result = asset_returns.mul(normalized_weights, axis=1).sum(axis=1)
    else:
        # Gewichte gelten einmal; danach entwickelt sich das Depot frei.
        value = normalize_to_base(aligned, base=1.0).mul(normalized_weights, axis=1).sum(axis=1)
        result = value.pct_change().dropna()

    result.name = "portfolio"
    result.index.name = DATE_INDEX_NAME
    return result


def _merged_provenance(prices: Mapping[str, PriceSeries], *, normalized: bool) -> Provenance:
    """Baut den Herkunftsnachweis einer abgeleiteten Matrix.

    Args:
        prices: Zugrunde liegende Kursreihen.
        normalized: Ob normiert wurde. Steht im Hinweistext.

    Returns:
        Provenance mit ``source=COMPUTED``. Der Verzug ist der **groesste**
        der Einzelreihen: die Matrix ist nur so aktuell wie ihr aeltester
        Bestandteil.
    """
    sources = [series.provenance for series in prices.values()]
    delays = [p.delay_minutes for p in sources if p.delay_minutes is not None]
    origins = sorted({p.source.value for p in sources})

    note = "Normalised to base 100" if normalized else "Raw prices"
    return Provenance(
        source=DataSource.COMPUTED,
        fetched_at=max(p.fetched_at for p in sources),
        from_cache=any(p.from_cache for p in sources),
        delay_minutes=max(delays) if delays else None,
        note=f"{note}, derived from {', '.join(origins)}",
    )


#: pandas-Regel fuer Wochenkerzen: Montag bis Sonntag, beschriftet mit dem
#: Montag - wie die Wochenkerzen von Yahoo Finance.
_WEEK_RULE: Final[str] = "W-MON"

#: Versatz, der den UTC-Stempel einer Tageskerze auf ihren Handelstag
#: bringt. yfinance stempelt Tageskerzen auf Mitternacht der Boersenzeit;
#: in UTC liegt das zwischen 11:00 am Vortag (UTC+13, Neuseeland) und 05:00
#: am selben Tag (UTC-5, New York im Winter). Live gesehen: SAP.DE 22:00 am
#: Vortag, 7203.T 15:00 am Vortag, AAPL 04:00. Plus 15 Stunden landet jeder
#: dieser Stempel auf dem richtigen Kalendertag. Ohne Versatz fiel die
#: Montags-Sitzung einer europaeischen Aktie in die Vorwoche.
_TRADING_DAY_SHIFT: Final[pd.Timedelta] = pd.Timedelta(hours=15)

#: Unterstuetzte Zielauflosungen von :func:`resample_ohlcv`.
RESAMPLE_INTERVALS: Final[frozenset[Interval]] = frozenset({Interval.D1, Interval.W1})


def _split_factor(values: pd.Series) -> float:
    """Fasst Split-Faktoren einer Woche zusammen.

    Args:
        values: Tageswerte der Spalte ``stock_splits``; 0 heisst kein Split.

    Returns:
        Produkt der Faktoren ungleich 0, oder 0.0 ohne Split in der Woche.
    """
    factors = values[values.fillna(0.0) != 0.0]
    if factors.empty:
        return 0.0
    product = 1.0
    for factor in factors.to_numpy(dtype="float64"):
        product *= float(factor)
    return product


def resample_ohlcv(series: PriceSeries, interval: Interval) -> PriceSeries:
    """Verdichtet Tageskerzen zu einer groeberen Aufloesung.

    Wochenkerzen entstehen lokal aus den Tageskerzen statt aus einem eigenen
    Abruf: das kostet keine weitere Anfrage an die gedrosselte Quelle, nutzt
    den vorhandenen Cache und funktioniert offline, wo nur Tagesdaten liegen.

    Eingeteilt wird nach dem **Handelstag** jeder Kerze, nicht nach ihrem
    UTC-Stempel - siehe :data:`_TRADING_DAY_SHIFT`.

    Live gegen die Wochenkerzen von Yahoo geprueft (26.09.2026, je 25
    Wochen): AAPL identisch bis 1e-7. Bei SAP.DE und 7203.T stimmen Hoch,
    Schluss und Volumen exakt; Eroeffnung und Tief weichen in einzelnen
    Wochen ab, weil Yahoos Wochenkerze dort den eigenen Tageskerzen
    widerspricht (SAP, Woche 04.05.: Tagestief Montag 143.29, Wochentief laut
    Yahoo 145.24). Die hier gebildete Kerze passt immer zum Tageschart.

    Regeln je Woche (Montag bis Sonntag, Zeitstempel = Montag, UTC):
    Eroeffnung = erster Kurs, Hoch = Maximum, Tief = Minimum, Schluss =
    letzter Kurs, Volumen und Dividenden = Summe, Split-Faktoren = Produkt.
    Die laufende Woche ist eine unvollstaendige Kerze bis zum letzten
    Handelstag - wie in jedem Chartwerkzeug. Wochen ohne Handelstag
    entfallen, statt als leere Kerze zu erscheinen.

    Args:
        series: Kursreihe in Tagesaufloesung.
        interval: Zielaufloesung, :attr:`Interval.D1` oder :attr:`Interval.W1`.

    Returns:
        Neue Kursreihe. Bei ``D1`` die unveraenderte Eingabe.

    Raises:
        InsufficientDataError: Wenn die Eingabe nicht in Tagesaufloesung
            vorliegt oder die Zielaufloesung nicht unterstuetzt wird.
    """
    if interval not in RESAMPLE_INTERVALS:
        msg = f"Resolution {interval.value} is not supported for charts."
        raise InsufficientDataError(
            msg, detail=f"erlaubt={sorted(i.value for i in RESAMPLE_INTERVALS)}"
        )
    if series.interval is not Interval.D1:
        msg = "Only daily price series can be condensed."
        raise InsufficientDataError(msg, detail=f"eingang={series.interval.value}")
    if interval is Interval.D1:
        return series

    open_, high, low, close, volume = OHLCV_COLUMNS
    frame = series.data
    rules: dict[Hashable, Any] = {
        open_: "first",
        high: "max",
        low: "min",
        close: "last",
        volume: "sum",
    }
    for column in frame.columns:
        if column in rules:
            continue
        if column == "dividends":
            rules[column] = "sum"
        elif column == "stock_splits":
            rules[column] = _split_factor
        else:
            rules[column] = "last"
    by_trading_day = frame.set_axis(frame.index + _TRADING_DAY_SHIFT, axis=0)
    weekly = by_trading_day.resample(_WEEK_RULE, label="left", closed="left").agg(rules)
    weekly = weekly.dropna(subset=[close])
    weekly.index.name = frame.index.name
    return series.model_copy(update={"interval": interval, "data": weekly})


#: Waehlbare Kerzen-Aufloesungen je Chart-Zeitraum, feinste zuerst.
#:
#: Live gegen Yahoo Finance gemessen am 26.09.2026 (AAPL, Kerzen je
#: Abruf): 1m nur bis 5d (1950 Kerzen), 5m/15m/30m bis 1mo, 1h bis 2y; bei
#: 3y liefert auch 1h nichts mehr. Ausserhalb dieser Grenzen antwortet die
#: Quelle mit einer LEEREN Reihe statt mit einem Fehler - deshalb steht die
#: Zuordnung fest hier und wird nicht ausprobiert.
#:
#: Grobere Aufloesungen sind nur aufgefuehrt, wo sie mindestens eine Handvoll
#: Kerzen ergeben: eine einzelne Tageskerze fuer "1d" zeigt nichts.
CHART_INTERVALS: Final[dict[str, tuple[Interval, ...]]] = {
    "1d": (Interval.M1, Interval.M5, Interval.M15, Interval.H1),
    "5d": (Interval.M1, Interval.M5, Interval.M15, Interval.H1, Interval.D1),
    "1mo": (Interval.M5, Interval.M15, Interval.H1, Interval.D1),
    "3mo": (Interval.H1, Interval.D1, Interval.W1),
    "6mo": (Interval.H1, Interval.D1, Interval.W1),
    "1y": (Interval.H1, Interval.D1, Interval.W1),
    "3y": (Interval.D1, Interval.W1),
    "5y": (Interval.D1, Interval.W1),
    "max": (Interval.D1, Interval.W1),
}

#: Ohne Live-Quelle liegen nur Tageskerzen vor (Fixtures).
_OFFLINE_INTERVALS: Final[tuple[Interval, ...]] = (Interval.D1, Interval.W1)


def allowed_intervals(period: str, *, intraday_available: bool = True) -> tuple[Interval, ...]:
    """Listet die Aufloesungen, die fuer einen Zeitraum anzeigbar sind.

    Args:
        period: Zeitraum in yfinance-Notation, etwa ``"5d"``.
        intraday_available: False, wenn nur Tagesdaten vorliegen (offline).

    Returns:
        Intervalle, feinstes zuerst. Ein unbekannter Zeitraum ergibt Tages-
        und Wochenkerzen.
    """
    chosen = CHART_INTERVALS.get(period, _OFFLINE_INTERVALS)
    if not intraday_available:
        chosen = tuple(i for i in chosen if i not in INTRADAY_INTERVALS) or _OFFLINE_INTERVALS
    return chosen


def finest_interval(period: str, *, intraday_available: bool = True) -> Interval:
    """Die kleinste Aufloesung, die fuer einen Zeitraum anzeigbar ist.

    Args:
        period: Zeitraum in yfinance-Notation.
        intraday_available: False, wenn nur Tagesdaten vorliegen (offline).

    Returns:
        Das feinste erlaubte Intervall.
    """
    return allowed_intervals(period, intraday_available=intraday_available)[0]
