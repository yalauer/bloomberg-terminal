"""Technische Kennzahlen als Screening-Spalten.

Ergaenzt das Kennzahlen-Universum aus :mod:`btlite.screening.universe` um
kursbasierte Groessen. Erst damit sind Praedikate wie "Kurs unter der
200-Tage-Linie" oder "RSI unter 30" moeglich - die Filter-DSL arbeitet auf
Spalten, und diese Spalten fehlten bislang.

Jede Spalte ist ein **Stichtagswert** zum letzten Kurs der Reihe, kein
Zeitverlauf. Das Screening beantwortet die Frage "wie steht der Titel
heute", nicht "wie stand er damals".

Architekturregel 2: dieses Modul holt keine Daten. Kursreihen werden
injiziert.

Einheiten:
    - Abstaende zu gleitenden Durchschnitten sind **dezimal** und
      **vorzeichenbehaftet**: ``-0.08`` bedeutet acht Prozent unter der
      Linie. Damit trifft ein ``LT 0``-Praedikat genau die Titel unter der
      Linie, ohne dass der Nutzer eine Richtung im Kopf umdrehen muss.
    - Volatilitaet und Wertentwicklungen annualisiert beziehungsweise
      dezimal, wie im ganzen Projekt.
    - Drawdown negativ.
    - Setup-Spalten (``setup_*``) aus :mod:`btlite.analytics.setups`:
      ``1.0`` = Bedingung erfuellt, ``0.0`` = nicht erfuellt, ``None`` =
      Historie reicht nicht (kein stilles "nicht erfuellt"). Keine
      Anlageempfehlung, nur messbare Chart-Bedingungen.
    - ``dist_support`` und ``dist_resistance``: ``(Kurs - Zone) / Zone``,
      dezimal. Zur Unterstuetzung null oder positiv, zum Widerstand
      negativ; ``None``, wenn es auf der Seite keine bestaetigte Zone gibt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pandas as pd

from btlite.analytics import technicals as tech
from btlite.analytics.setups import setup_report
from btlite.core.errors import BTLiteError
from btlite.core.logging import get_logger
from btlite.models.common import Interval
from btlite.models.setups import SetupKey, SetupStatus

if TYPE_CHECKING:
    from collections.abc import Mapping

    from btlite.models.market import PriceSeries

_LOG = get_logger(__name__)

#: Fensterlaengen der gleitenden Durchschnitte, die als Spalten entstehen.
SMA_WINDOWS: Final[tuple[int, ...]] = (20, 50, 200)

#: Handelstage je Zeitraum fuer die Wertentwicklungs-Spalten.
PERFORMANCE_WINDOWS: Final[dict[str, int]] = {
    "perf_1m": 21,
    "perf_3m": 63,
    "perf_6m": 126,
    "perf_12m": 252,
}

#: Fenster der Volatilitaets-Spalte in Handelstagen.
VOLATILITY_WINDOW: Final[int] = 252

#: Setup-Spalten je Setup aus :mod:`btlite.analytics.setups`.
SETUP_COLUMNS: Final[dict[SetupKey, str]] = {
    SetupKey.PULLBACK: "setup_pullback",
    SetupKey.RSI_RECOVERY: "setup_rsi_recovery",
    SetupKey.BREAKOUT: "setup_breakout",
    SetupKey.SUPPORT_MACD: "setup_support_macd",
}

#: Numerische Abbildung des Setup-Status fuer die Filter-DSL.
_STATUS_VALUES: Final[dict[SetupStatus, float | None]] = {
    SetupStatus.MET: 1.0,
    SetupStatus.NOT_MET: 0.0,
    SetupStatus.INSUFFICIENT_DATA: None,
}

#: Alle Spalten, die dieses Modul erzeugt. Fuer Dokumentation und UI.
TECHNICAL_COLUMNS: Final[tuple[str, ...]] = (
    "price",
    "sma_20",
    "sma_50",
    "sma_200",
    "dist_sma_20",
    "dist_sma_50",
    "dist_sma_200",
    "above_sma_200",
    "golden_cross",
    "rsi_14",
    "volatility_1y",
    "max_drawdown_1y",
    "perf_1m",
    "perf_3m",
    "perf_6m",
    "perf_12m",
    "from_high_52w",
    "from_low_52w",
    "avg_volume_3m",
    "setup_pullback",
    "setup_rsi_recovery",
    "setup_breakout",
    "setup_support_macd",
    "dist_support",
    "dist_resistance",
)


def _last(series: pd.Series) -> float | None:
    """Liest den letzten belastbaren Wert einer Reihe.

    Args:
        series: Beliebige Zahlenreihe.

    Returns:
        Letzter Wert, oder None wenn die Reihe leer ist oder nur Fehlwerte
        traegt. Nie ein stiller Nullwert.
    """
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    return float(cleaned.iloc[-1])


def _relative(value: float | None, reference: float | None) -> float | None:
    """Berechnet den vorzeichenbehafteten relativen Abstand.

    Args:
        value: Betrachteter Wert, etwa der aktuelle Kurs.
        reference: Bezugswert, etwa ein gleitender Durchschnitt.

    Returns:
        ``(value - reference) / reference`` als Dezimalzahl, oder None wenn
        ein Wert fehlt oder der Bezugswert 0 ist. Negativ bedeutet
        unterhalb des Bezugswerts.
    """
    if value is None or reference is None or reference == 0:
        return None
    return (value - reference) / reference


def _performance(close: pd.Series, bars: int) -> float | None:
    """Wertentwicklung ueber eine Anzahl Handelstage.

    Args:
        close: Schlusskursreihe.
        bars: Rueckblick in Handelstagen.

    Returns:
        Dezimale Veraenderung, oder None wenn die Historie nicht reicht
        oder der Startkurs 0 ist.
    """
    if len(close) <= bars:
        return None
    start = float(close.iloc[-1 - bars])
    if start == 0:
        return None
    return float(close.iloc[-1]) / start - 1.0


def technical_row(series: PriceSeries) -> dict[str, float | None]:
    """Berechnet die technischen Kennzahlen eines einzelnen Titels.

    Indikatoren, deren Fenster laenger als die Historie ist, ergeben
    ``None`` statt eines Fehlers. Ein junger Titel soll im Screening
    auftauchen, nur eben ohne 200-Tage-Linie. Die Filter-DSL behandelt den
    Fehlwert konsistent: er erfuellt kein Vergleichspraedikat.

    Args:
        series: Kursreihe mit mindestens den Spalten ``close`` und
            ``volume``.

    Returns:
        Abbildung Spaltenname auf Wert. Schluessel sind genau
        :data:`TECHNICAL_COLUMNS`.
    """
    close = series.close
    row: dict[str, float | None] = dict.fromkeys(TECHNICAL_COLUMNS)

    price = _last(close)
    row["price"] = price
    if price is None:
        _LOG.debug("Kursreihe %s ohne belastbaren Schlusskurs", series.ticker)
        return row

    # Gleitende Durchschnitte und der vorzeichenbehaftete Abstand dazu.
    for window in SMA_WINDOWS:
        if len(close) < window:
            continue
        line = _last(tech.sma(close, window))
        row[f"sma_{window}"] = line
        row[f"dist_sma_{window}"] = _relative(price, line)

    sma_200 = row["sma_200"]
    if sma_200 is not None:
        # 1.0 statt True: die Filter-DSL vergleicht numerisch.
        row["above_sma_200"] = 1.0 if price > sma_200 else 0.0

    sma_50 = row["sma_50"]
    if sma_50 is not None and sma_200 is not None:
        row["golden_cross"] = 1.0 if sma_50 > sma_200 else 0.0

    if len(close) > tech.DEFAULT_RSI_WINDOW:
        row["rsi_14"] = _last(tech.rsi(close, tech.DEFAULT_RSI_WINDOW))

    returns = close.pct_change().dropna()
    if len(returns) >= VOLATILITY_WINDOW:
        from btlite.analytics.risk import volatility

        row["volatility_1y"] = volatility(returns.iloc[-VOLATILITY_WINDOW:], interval=Interval.D1)

    if len(returns) >= VOLATILITY_WINDOW:
        from btlite.analytics.risk import drawdown_series

        last_year = returns.iloc[-VOLATILITY_WINDOW:]
        row["max_drawdown_1y"] = _last(drawdown_series(last_year).cummin())

    for name, bars in PERFORMANCE_WINDOWS.items():
        row[name] = _performance(close, bars)

    # Abstand zum Jahreshoch und -tief. Ersteres ist negativ oder null.
    year = close.iloc[-VOLATILITY_WINDOW:] if len(close) >= VOLATILITY_WINDOW else close
    row["from_high_52w"] = _relative(price, float(year.max()))
    row["from_low_52w"] = _relative(price, float(year.min()))

    volume = series.data.get("volume")
    if volume is not None and len(volume) >= PERFORMANCE_WINDOWS["perf_3m"]:
        row["avg_volume_3m"] = float(volume.iloc[-PERFORMANCE_WINDOWS["perf_3m"] :].mean())

    # Setups: Rechnung ausschliesslich in analytics.setups, hier nur Abbildung.
    report = setup_report(series)
    for key, column in SETUP_COLUMNS.items():
        check = report.check(key)
        row[column] = _STATUS_VALUES[check.status] if check is not None else None
    row["dist_support"] = report.dist_support
    row["dist_resistance"] = report.dist_resistance

    return row


def build_technical_universe(
    prices: Mapping[str, PriceSeries],
    *,
    skip_failures: bool = True,
) -> pd.DataFrame:
    """Baut die technischen Spalten fuer mehrere Titel.

    Args:
        prices: Kursreihen je Ticker.
        skip_failures: Titel, deren Berechnung scheitert, uebergehen statt
            den gesamten Aufbau abzubrechen.

    Returns:
        DataFrame mit Ticker als Index und :data:`TECHNICAL_COLUMNS` als
        Spalten. Leer, wenn kein Titel auswertbar war.

    Raises:
        BTLiteError: Wenn ``skip_failures`` False ist und ein Titel
            scheitert.
    """
    rows: dict[str, dict[str, float | None]] = {}
    for ticker, series in prices.items():
        try:
            rows[ticker] = technical_row(series)
        except BTLiteError as exc:
            if not skip_failures:
                raise
            _LOG.info("Technische Kennzahlen fuer %s uebergangen: %s", ticker, exc.user_message)

    if not rows:
        frame = pd.DataFrame(columns=list(TECHNICAL_COLUMNS))
        frame.index.name = "ticker"
        return frame

    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "ticker"
    return frame.reindex(columns=list(TECHNICAL_COLUMNS))


def enrich_with_technicals(
    universe: pd.DataFrame,
    prices: Mapping[str, PriceSeries],
) -> pd.DataFrame:
    """Haengt die technischen Spalten an ein bestehendes Universum.

    Bei Namenskollision gewinnt die technische Spalte, weil sie aus der
    Kursreihe stammt und damit aktueller ist als ein aus Statements
    abgeleiteter Wert.

    Args:
        universe: Kennzahlen-Universum, Index sind Ticker.
        prices: Kursreihen je Ticker.

    Returns:
        Zusammengefuehrtes Universum. Titel ohne Kursreihe behalten ihre
        Zeile mit ``NaN`` in den technischen Spalten.
    """
    technical = build_technical_universe(prices)
    if technical.empty:
        return universe

    keep = [column for column in universe.columns if column not in technical.columns]
    merged = universe[keep].join(technical, how="left")
    merged.index.name = universe.index.name or "ticker"
    return merged


def describe_columns() -> dict[str, str]:
    """Beschreibt die technischen Spalten fuer die Oberflaeche.

    Returns:
        Abbildung Spaltenname auf eine Erklaerung in einem Satz, samt
        Einheit. Das UI zeigt sie als Hilfe zur Filterauswahl.
    """
    return {
        "price": "Last close in price currency.",
        "sma_20": "20-day moving average in price currency.",
        "sma_50": "50-day moving average in price currency.",
        "sma_200": "200-day moving average in price currency.",
        "dist_sma_20": "Distance to the 20-day moving average, decimal. Negative means below.",
        "dist_sma_50": "Distance to the 50-day moving average, decimal. Negative means below.",
        "dist_sma_200": "Distance to the 200-day moving average, decimal. Negative means below.",
        "above_sma_200": "1 if the price is above the 200-day moving average, else 0.",
        "golden_cross": "1 if the 50-day is above the 200-day moving average, else 0.",
        "rsi_14": "Relative Strength Index over 14 days, scale 0 to 100.",
        "volatility_1y": "Annualised volatility over the last 252 days, decimal.",
        "max_drawdown_1y": "Largest drawdown over the last 252 days, decimal and negative.",
        "perf_1m": "Performance over 21 trading days, decimal.",
        "perf_3m": "Performance over 63 trading days, decimal.",
        "perf_6m": "Performance over 126 trading days, decimal.",
        "perf_12m": "Performance over 252 trading days, decimal.",
        "from_high_52w": "Distance to the 52-week high, decimal. Zero or negative.",
        "from_low_52w": "Distance to the 52-week low, decimal. Zero or positive.",
        "avg_volume_3m": "Average daily volume over the last 63 days, in shares.",
        "setup_pullback": (
            "1 if price and 50-day MA are above the 200-day MA and the price is within "
            "3% of the 50-day MA, else 0. Empty below 200 days of history."
        ),
        "setup_rsi_recovery": (
            "1 if RSI 14 was below 30 in the 10 days before the last close and is "
            "30 or higher now, else 0."
        ),
        "setup_breakout": (
            "1 if the last close is above the highest close of the prior 252 days on "
            "volume above the 63-day average, else 0."
        ),
        "setup_support_macd": (
            "1 if the price is at most 3% above the nearest support zone and the MACD "
            "crossed above its signal line in the last 5 days, else 0."
        ),
        "dist_support": (
            "Distance to the nearest support zone below the price, decimal. Zero or positive."
        ),
        "dist_resistance": (
            "Distance to the nearest resistance zone above the price, decimal. Negative."
        ),
    }
