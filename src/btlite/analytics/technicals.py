"""Technische Indikatoren.

Erzeugt die Reihen, die :class:`~btlite.viz.price.TechnicalOverlays` erwartet.
Dieses Modul rechnet, es stellt nichts dar und holt keine Daten
(Architekturregeln 2 und 3).

Konventionen des Moduls:

- Eingabe ist immer eine :class:`~btlite.models.market.PriceSeries` oder eine
  ``pd.Series`` mit tz-aware UTC-Index.
- Die Anlaufphase eines Fensters bleibt ``NaN`` und wird **nicht** aufgefuellt.
  Ein aufgefuellter Wert waere eine Behauptung ueber Daten, die es nicht gibt,
  und im Chart eine Linie, die zu weit links beginnt.
- Die Spaltennamen von Bollinger und MACD stammen aus
  :mod:`btlite.models.common` und sind Vertrag, kein Detail dieses Moduls.
- Kein Look-ahead: jeder Wert an Position t benutzt ausschliesslich Daten bis
  einschliesslich t. Alle verwendeten pandas-Operationen sind
  rueckwaertsgerichtet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from btlite.core.errors import InsufficientDataError
from btlite.models.common import BOLLINGER_COLUMNS, MACD_COLUMNS

if TYPE_CHECKING:
    from btlite.models.market import PriceSeries
    from btlite.viz.price import TechnicalOverlays

#: Neutrale Mitte der RSI-Skala.
RSI_NEUTRAL: Final[float] = 50.0

#: Obere und untere Grenze der RSI-Skala.
RSI_MIN: Final[float] = 0.0
RSI_MAX: Final[float] = 100.0

#: Uebliche Parameter, damit UI und Strategien nicht jeweils eigene erfinden.
DEFAULT_SMA_WINDOWS: Final[tuple[int, ...]] = (20, 50, 200)
DEFAULT_EMA_WINDOWS: Final[tuple[int, ...]] = (12, 26)
DEFAULT_RSI_WINDOW: Final[int] = 14
DEFAULT_BOLLINGER_WINDOW: Final[int] = 20
DEFAULT_BOLLINGER_STD: Final[float] = 2.0
DEFAULT_MACD_FAST: Final[int] = 12
DEFAULT_MACD_SLOW: Final[int] = 26
DEFAULT_MACD_SIGNAL: Final[int] = 9


def _as_series(source: PriceSeries | pd.Series) -> pd.Series:
    """Holt die Schlusskursreihe aus der Eingabe.

    Args:
        source: Kursreihe oder bereits extrahierte Series.

    Returns:
        Schlusskurs-Series mit UTC-Index.
    """
    if isinstance(source, pd.Series):
        return source
    return source.close


def _require_length(series: pd.Series, window: int, *, what: str) -> None:
    """Prueft, dass genug Beobachtungen fuer ein Fenster vorliegen.

    Args:
        series: Eingangsreihe.
        window: Benoetigte Fensterlaenge in Perioden.
        what: Bezeichnung fuer die Fehlermeldung.

    Raises:
        InsufficientDataError: Wenn die Reihe kuerzer als das Fenster ist.
    """
    available = len(series)
    if available < window:
        msg = f"{what} requires {window} observations, {available} available"
        raise InsufficientDataError(msg, required=window, available=available)


def sma(source: PriceSeries | pd.Series, window: int) -> pd.Series:
    """Einfacher gleitender Durchschnitt.

    Args:
        source: Kursreihe oder Series.
        window: Fensterlaenge in Perioden.

    Returns:
        Series in Kurswaehrung. Die ersten ``window - 1`` Werte sind ``NaN``.

    Raises:
        ValueError: Wenn ``window`` nicht positiv ist.
        InsufficientDataError: Wenn die Reihe kuerzer als das Fenster ist.
    """
    if window <= 0:
        msg = "window muss positiv sein"
        raise ValueError(msg)
    close = _as_series(source)
    _require_length(close, window, what=f"SMA {window}")
    out = close.rolling(window=window, min_periods=window).mean()
    out.name = f"SMA {window}"
    return out


def ema(source: PriceSeries | pd.Series, window: int) -> pd.Series:
    """Exponentiell gewichteter gleitender Durchschnitt.

    Nutzt ``adjust=False``, also die rekursive Form, wie sie in der
    Chartanalyse ueblich ist. Die ersten ``window - 1`` Werte bleiben ``NaN``,
    damit die Linie nicht auf duenner Datenbasis beginnt.

    Args:
        source: Kursreihe oder Series.
        window: Glaettungsperiode (Span).

    Returns:
        Series in Kurswaehrung.

    Raises:
        ValueError: Wenn ``window`` nicht positiv ist.
        InsufficientDataError: Wenn die Reihe kuerzer als das Fenster ist.
    """
    if window <= 0:
        msg = "window muss positiv sein"
        raise ValueError(msg)
    close = _as_series(source)
    _require_length(close, window, what=f"EMA {window}")
    out = close.ewm(span=window, adjust=False, min_periods=window).mean()
    out.name = f"EMA {window}"
    return out


def rsi(source: PriceSeries | pd.Series, window: int = DEFAULT_RSI_WINDOW) -> pd.Series:
    """Relative Strength Index nach Wilder.

    Verwendet die Wilder-Glaettung (``alpha = 1 / window``), nicht den
    einfachen Mittelwert. Das ist die Standardform; ein einfacher Mittelwert
    ergibt sichtbar andere Werte.

    Randfaelle sind bewusst definiert:

    - Nur Gewinne im Fenster (kein Verlust) ergibt 100.
    - Nur Verluste ergibt 0.
    - Keine Bewegung ergibt 50, den neutralen Wert. Ein ``NaN`` waere hier
      irrefuehrend, weil der Indikator sehr wohl definiert ist.

    Args:
        source: Kursreihe oder Series.
        window: Fensterlaenge in Perioden, ueblich 14.

    Returns:
        Series auf der Skala 0 bis 100. Die Anlaufphase ist ``NaN``.

    Raises:
        ValueError: Wenn ``window`` nicht positiv ist.
        InsufficientDataError: Wenn die Reihe zu kurz ist.
    """
    if window <= 0:
        msg = "window muss positiv sein"
        raise ValueError(msg)
    close = _as_series(source)
    # Die erste Differenz kostet eine Beobachtung.
    _require_length(close, window + 1, what=f"RSI {window}")

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    alpha = 1.0 / window
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=window).mean()

    out = pd.Series(np.nan, index=close.index, dtype="float64")
    valid = avg_gain.notna() & avg_loss.notna()

    # Regulaerer Fall: beide Mittelwerte liegen vor, Verluste sind nicht null.
    normal = valid & (avg_loss > 0)
    rs = avg_gain[normal] / avg_loss[normal]
    out[normal] = RSI_MAX - (RSI_MAX / (1.0 + rs))

    # Keine Verluste: der Indikator ist am oberen Anschlag, sofern es
    # ueberhaupt Gewinne gab, sonst neutral.
    no_loss = valid & (avg_loss == 0)
    out[no_loss & (avg_gain > 0)] = RSI_MAX
    out[no_loss & (avg_gain == 0)] = RSI_NEUTRAL

    out.name = f"RSI {window}"
    return out


def bollinger_bands(
    source: PriceSeries | pd.Series,
    window: int = DEFAULT_BOLLINGER_WINDOW,
    num_std: float = DEFAULT_BOLLINGER_STD,
) -> pd.DataFrame:
    """Bollinger-Baender um einen gleitenden Durchschnitt.

    Args:
        source: Kursreihe oder Series.
        window: Fensterlaenge in Perioden.
        num_std: Anzahl Standardabweichungen fuer die Bandbreite.

    Returns:
        DataFrame mit den Spalten ``upper``, ``middle`` und ``lower`` in
        Kurswaehrung, in dieser Reihenfolge.

    Raises:
        ValueError: Wenn ``window`` nicht positiv oder ``num_std`` negativ ist.
        InsufficientDataError: Wenn die Reihe kuerzer als das Fenster ist.
    """
    if window <= 0:
        msg = "window muss positiv sein"
        raise ValueError(msg)
    if num_std < 0:
        msg = "num_std darf nicht negativ sein"
        raise ValueError(msg)

    close = _as_series(source)
    _require_length(close, window, what=f"Bollinger {window}")

    rolling = close.rolling(window=window, min_periods=window)
    middle = rolling.mean()
    # ddof=0: Standardabweichung der Fensterpopulation, nicht der Stichprobe.
    # Das ist die in der Chartanalyse uebliche Variante.
    spread = rolling.std(ddof=0) * num_std

    upper_col, middle_col, lower_col = BOLLINGER_COLUMNS
    return pd.DataFrame(
        {upper_col: middle + spread, middle_col: middle, lower_col: middle - spread}
    )


def macd(
    source: PriceSeries | pd.Series,
    fast: int = DEFAULT_MACD_FAST,
    slow: int = DEFAULT_MACD_SLOW,
    signal: int = DEFAULT_MACD_SIGNAL,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence.

    Args:
        source: Kursreihe oder Series.
        fast: Periode des schnellen EMA.
        slow: Periode des langsamen EMA.
        signal: Periode der Signallinie.

    Returns:
        DataFrame mit den Spalten ``macd``, ``signal`` und ``histogram`` in
        Kurswaehrung.

    Raises:
        ValueError: Wenn Perioden nicht positiv sind oder ``fast`` nicht
            kleiner als ``slow`` ist.
        InsufficientDataError: Wenn die Reihe zu kurz ist.
    """
    if min(fast, slow, signal) <= 0:
        msg = "Alle Perioden muessen positiv sein"
        raise ValueError(msg)
    if fast >= slow:
        msg = f"fast ({fast}) muss kleiner als slow ({slow}) sein"
        raise ValueError(msg)

    close = _as_series(source)
    _require_length(close, slow, what=f"MACD {fast}/{slow}")

    fast_ema = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = fast_ema - slow_ema
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()

    macd_col, signal_col, hist_col = MACD_COLUMNS
    return pd.DataFrame(
        {
            macd_col: line,
            signal_col: signal_line,
            hist_col: line - signal_line,
        }
    )


def atr(series: PriceSeries, window: int = DEFAULT_RSI_WINDOW) -> pd.Series:
    """Average True Range, ein Volatilitaetsmass in Kurswaehrung.

    Die True Range beruecksichtigt Eroeffnungsluecken, ist also nicht nur
    Hoch minus Tief.

    Args:
        series: Kursreihe mit den Spalten ``high``, ``low`` und ``close``.
        window: Fensterlaenge in Perioden.

    Returns:
        Series in Kurswaehrung.

    Raises:
        ValueError: Wenn ``window`` nicht positiv ist.
        InsufficientDataError: Wenn die Reihe zu kurz ist.
    """
    if window <= 0:
        msg = "window muss positiv sein"
        raise ValueError(msg)

    data = series.data
    _require_length(data["close"], window + 1, what=f"ATR {window}")

    prev_close = data["close"].shift(1)
    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - prev_close).abs(),
            (data["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out = true_range.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    out.name = f"ATR {window}"
    return out


def build_overlays(
    series: PriceSeries,
    *,
    sma_windows: tuple[int, ...] = DEFAULT_SMA_WINDOWS,
    ema_windows: tuple[int, ...] = (),
    with_bollinger: bool = False,
    with_rsi: bool = False,
    with_macd: bool = False,
    bollinger_window: int = DEFAULT_BOLLINGER_WINDOW,
    bollinger_std: float = DEFAULT_BOLLINGER_STD,
    rsi_window: int = DEFAULT_RSI_WINDOW,
) -> TechnicalOverlays:
    """Baut das Overlay-Paket fuer die Chart-Schicht.

    Indikatoren, deren Fenster laenger als die Historie ist, werden
    **uebergangen** statt einen Fehler auszuloesen. Bei einer kurzen Reihe soll
    das UI die Indikatoren zeigen, die berechenbar sind, statt einen leeren
    Chart. Welche fehlen, ist am Ergebnis ablesbar.

    Args:
        series: Kursreihe.
        sma_windows: Fensterlaengen der einfachen Durchschnitte.
        ema_windows: Fensterlaengen der exponentiellen Durchschnitte.
        with_bollinger: Bollinger-Baender berechnen.
        with_rsi: RSI berechnen (eigener Subplot).
        with_macd: MACD berechnen (eigener Subplot).
        bollinger_window: Fensterlaenge der Baender.
        bollinger_std: Bandbreite in Standardabweichungen.
        rsi_window: Fensterlaenge des RSI.

    Returns:
        Gefuelltes Overlay-Paket. Nicht berechenbare Indikatoren fehlen.
    """
    from btlite.viz.price import TechnicalOverlays

    close = series.close
    overlays = TechnicalOverlays()

    for window in sma_windows:
        if len(close) >= window:
            overlays.sma[f"SMA {window}"] = sma(close, window)

    for window in ema_windows:
        if len(close) >= window:
            overlays.ema[f"EMA {window}"] = ema(close, window)

    if with_bollinger and len(close) >= bollinger_window:
        overlays.bollinger = bollinger_bands(close, bollinger_window, bollinger_std)

    if with_rsi and len(close) > rsi_window:
        overlays.rsi = rsi(close, rsi_window)

    if with_macd and len(close) >= DEFAULT_MACD_SLOW:
        overlays.macd = macd(close)

    return overlays


def crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """Findet Aufwaertskreuzungen zweier Reihen.

    Ein Signal an Position t bedeutet: bei t liegt ``fast`` ueber ``slow``, bei
    t-1 lag es darunter oder gleichauf. Es werden ausschliesslich Daten bis
    einschliesslich t verwendet, also kein Look-ahead.

    Args:
        fast: Schnellere Reihe.
        slow: Langsamere Reihe.

    Returns:
        Boolesche Series. Positionen mit ``NaN`` in einer Eingangsreihe sind
        False, nicht True.
    """
    above = fast > slow
    was_above = above.shift(1, fill_value=False)
    valid = fast.notna() & slow.notna() & fast.shift(1).notna() & slow.shift(1).notna()
    return above & ~was_above & valid


def crossunder(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """Findet Abwaertskreuzungen zweier Reihen.

    Args:
        fast: Schnellere Reihe.
        slow: Langsamere Reihe.

    Returns:
        Boolesche Series ohne Look-ahead, siehe :func:`crossover`.
    """
    return crossover(slow, fast)
