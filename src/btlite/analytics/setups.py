"""Chart-Setup-Checkliste auf Tageskursen.

Prueft vier gaengige, **messbare** Chart-Bedingungen zum Stichtag, der
letzten Bar der Kursreihe. Je Setup gibt es genau einen Status - ``met``,
``not_met`` oder ``insufficient_data`` - plus die Messwerte, auf denen er
beruht.

**Keine Anlageempfehlung.** Das Modul bildet kein Gesamturteil, keinen
Score und keine Ampel. Ein erfuelltes Setup sagt, dass eine Bedingung im
Kursverlauf vorliegt, nicht, dass ein Kauf oder Verkauf angezeigt waere.
Ob eine Bedingung ein Anlass zum Handeln ist, entscheidet dieses Werkzeug
nicht. Die Texte fuer das UI sind entsprechend neutral formuliert
("Condition met", "not met").

Architekturregel 2: kein Netzwerk. Kursreihen werden injiziert.

Kein Look-ahead: jede Pruefung benutzt ausschliesslich Bars bis
einschliesslich Stichtag. Swing-Hochs und -Tiefs sind erst ``SWING_ORDER``
Bars nach dem Extremum bestaetigt und zaehlen vorher nicht.

Semantik der vier Setups
------------------------

1. **Trend-Pullback** (``pullback``): Aufwaertstrend heisst Kurs ueber der
   200-Tage-Linie **und** 50-Tage- ueber 200-Tage-Linie. Dazu liegt der Kurs
   nahe der 50-Tage-Linie: ``|dist_sma_50| <= PULLBACK_BAND`` (0.03).
   Braucht 200 Bars.
2. **RSI-Erholung** (``rsi_recovery``): der RSI(14) war in den
   ``RSI_LOOKBACK`` (10) Bars **vor** dem Stichtag mindestens einmal unter
   ``RSI_OVERSOLD`` (30) und steht am Stichtag bei 30 oder darueber. Der
   Stichtag selbst zaehlt nicht zum Fenster, sonst koennte dieselbe Bar
   Tief und Erholung zugleich sein. Braucht 25 Bars.
3. **Ausbruch** (``breakout``): Schlusskurs am Stichtag ueber dem hoechsten
   Schlusskurs der ``BREAKOUT_LOOKBACK`` (252) vorangegangenen Bars
   (Stichtag ausgeschlossen) **und** Volumen am Stichtag ueber dem
   Durchschnitt der ``VOLUME_LOOKBACK`` (63) vorangegangenen Bars. Braucht
   253 Bars und Volumendaten.
4. **Unterstuetzung + MACD** (``support_macd``): Swing-Hochs (auf ``high``)
   und Swing-Tiefs (auf ``low``) mit Extremum in den letzten
   ``SWING_LOOKBACK`` (126) Bars. Ein Swing ist ein striktes Extremum
   gegenueber den ``SWING_ORDER`` (5) Bars links und mindestens gleich
   gegenueber den 5 Bars rechts; er gilt erst als bestaetigt, wenn diese 5
   Bars rechts vorliegen. Alle Swing-Niveaus werden aufsteigend sortiert
   und zu Zonen zusammengefasst: ein Niveau gehoert zur laufenden Zone,
   solange es hoechstens ``ZONE_TOLERANCE`` (1.5 Prozent) ueber dem
   niedrigsten Niveau dieser Zone liegt. Verankert am niedrigsten Niveau,
   damit eine Zone nicht kettenartig beliebig breit wird. Zonen mit
   Mittelwert bis zum Kurs sind Unterstuetzung, darueber Widerstand -
   ein frueheres Hoch kann so zur Unterstuetzung werden. MACD(12, 26, 9):
   juengste Kreuzung der MACD-Linie ueber oder unter die Signallinie in
   den letzten ``MACD_CROSS_WINDOW`` (5) Bars inklusive Stichtag. Das
   Setup ist erfuellt, wenn der Kurs hoechstens ``SUPPORT_BAND`` (3
   Prozent) ueber der naechsten Unterstuetzung liegt **und** die juengste
   Kreuzung nach oben zeigt. Braucht 131 Bars.

Zur Wahl der Zonentoleranz: fest 1.5 Prozent statt 1 ATR(14). Eine feste
relative Toleranz ist ueber Titel hinweg vergleichbar, haengt nicht von der
Qualitaet der Hoch/Tief-Daten ab und passt zur halben Breite des
3-Prozent-Bands, mit dem das Setup die Naehe zur Zone misst. Nachteil:
bei sehr volatilen Titeln entstehen mehr, schmalere Zonen als mit einer
ATR-Toleranz.

Einheiten: Abstaende dezimal und vorzeichenbehaftet, ``(Kurs - Bezug) /
Bezug``; negativ heisst unterhalb des Bezugs. RSI in Punkten 0 bis 100,
Volumenverhaeltnis als Vielfaches, Zeitabstaende in Bars (Handelstagen).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from btlite.analytics import technicals as tech
from btlite.core.errors import InsufficientDataError
from btlite.models.common import MACD_COLUMNS, OHLCV_COLUMNS
from btlite.models.setups import (
    CrossDirection,
    Measurement,
    MeasurementUnit,
    PriceZone,
    SetupCheck,
    SetupKey,
    SetupReport,
    SetupStatus,
    ZoneKind,
)

if TYPE_CHECKING:
    from btlite.models.market import PriceSeries

_, _HIGH, _LOW, _CLOSE, _VOLUME = OHLCV_COLUMNS
_MACD_LINE, _MACD_SIGNAL, _ = MACD_COLUMNS

# --------------------------------------------------------------------------
# Schwellen und Fenster
# --------------------------------------------------------------------------

#: Schnelle und langsame Durchschnittslinie des Trend-Pullbacks, in Bars.
SMA_FAST: Final[int] = 50
SMA_SLOW: Final[int] = 200

#: Maximaler Abstand zur 50-Tage-Linie, dezimal, beidseitig.
PULLBACK_BAND: Final[float] = 0.03

#: RSI-Periode in Bars.
RSI_WINDOW: Final[int] = tech.DEFAULT_RSI_WINDOW

#: Ueberverkauft-Schwelle auf der RSI-Skala 0 bis 100.
RSI_OVERSOLD: Final[float] = 30.0

#: Anzahl Bars vor dem Stichtag, in denen das RSI-Tief liegen muss.
RSI_LOOKBACK: Final[int] = 10

#: Rueckblick fuer das bisherige Schlusskurs-Hoch, in Bars, Stichtag exklusive.
BREAKOUT_LOOKBACK: Final[int] = 252

#: Rueckblick fuer das Durchschnittsvolumen, in Bars, Stichtag exklusive.
VOLUME_LOOKBACK: Final[int] = 63

#: Das Volumen am Stichtag muss dieses Vielfache des Durchschnitts uebertreffen.
VOLUME_RATIO_MIN: Final[float] = 1.0

#: Rueckblick fuer Swing-Hochs und -Tiefs, in Bars.
SWING_LOOKBACK: Final[int] = 126

#: Bars links und rechts, gegen die ein Swing ein Extremum sein muss.
SWING_ORDER: Final[int] = 5

#: Relative Breite einer Zone, dezimal, gemessen ab ihrem niedrigsten Niveau.
ZONE_TOLERANCE: Final[float] = 0.015

#: Maximaler Abstand des Kurses ueber der Unterstuetzung, dezimal.
SUPPORT_BAND: Final[float] = 0.03

#: Fenster fuer die MACD-Kreuzung in Bars, inklusive Stichtag.
MACD_CROSS_WINDOW: Final[int] = 5

#: Mindestlaengen je Setup in Bars.
PULLBACK_MIN_BARS: Final[int] = SMA_SLOW
RSI_MIN_BARS: Final[int] = RSI_WINDOW + RSI_LOOKBACK + 1
BREAKOUT_MIN_BARS: Final[int] = BREAKOUT_LOOKBACK + 1
#: Erste Signallinie bei Index slow + signal - 2; die Kreuzung braucht dazu
#: die Vorbar, und das ganze Fenster soll bewertbar sein.
MACD_MIN_BARS: Final[int] = (
    tech.DEFAULT_MACD_SLOW + tech.DEFAULT_MACD_SIGNAL - 1 + MACD_CROSS_WINDOW
)
SUPPORT_MIN_BARS: Final[int] = max(SWING_LOOKBACK + SWING_ORDER, MACD_MIN_BARS)

#: Englische Bezeichnungen der Setups fuer das UI.
SETUP_LABELS: Final[dict[SetupKey, str]] = {
    SetupKey.PULLBACK: "Pullback to the 50-day MA within an uptrend",
    SetupKey.RSI_RECOVERY: "RSI recovery from below 30",
    SetupKey.BREAKOUT: "Breakout above the 252-day closing high on higher volume",
    SetupKey.SUPPORT_MACD: "Near support with a MACD cross up",
}


# --------------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------------


def _frame(series: PriceSeries) -> pd.DataFrame:
    """Bereitet den Payload auf: Zeilen ohne Schlusskurs fallen weg.

    Fehlende Hochs und Tiefs werden durch den Schlusskurs ersetzt, damit
    eine einzelne Luecke keinen Swing erfindet oder verschluckt.

    Args:
        series: Kursreihe.

    Returns:
        Bereinigter OHLCV-Frame mit UTC-Index.
    """
    data = series.data.dropna(subset=[_CLOSE])
    if data[[_HIGH, _LOW]].isna().to_numpy().any():
        data = data.assign(
            **{_HIGH: data[_HIGH].fillna(data[_CLOSE]), _LOW: data[_LOW].fillna(data[_CLOSE])}
        )
    return data


def _relative(value: float, reference: float) -> float | None:
    """Vorzeichenbehafteter relativer Abstand.

    Args:
        value: Betrachteter Wert.
        reference: Bezugswert.

    Returns:
        ``(value - reference) / reference`` dezimal, None bei Bezug 0 oder
        fehlenden Werten.
    """
    if reference == 0 or math.isnan(value) or math.isnan(reference):
        return None
    return (value - reference) / reference


def _describe_distance(distance: float | None, what: str) -> str:
    """Formuliert einen Abstand neutral, etwa ``1.2% above the 50-day MA``.

    Args:
        distance: Dezimaler, vorzeichenbehafteter Abstand.
        what: Bezugsgroesse im Text.

    Returns:
        Englischer Halbsatz.
    """
    if distance is None:
        return f"distance to {what} n/a"
    side = "above" if distance >= 0 else "below"
    return f"{abs(distance):.1%} {side} {what}"


def _bars_ago(bars: int) -> str:
    """Formuliert einen Zeitabstand in Bars, etwa ``1 bar ago`` oder ``today``.

    Args:
        bars: Abstand in Bars, 0 = Stichtag.

    Returns:
        Englischer Halbsatz.
    """
    if bars == 0:
        return "today"
    return f"{bars} bar ago" if bars == 1 else f"{bars} bars ago"


def _insufficient(key: SetupKey, required: int, available: int) -> SetupCheck:
    """Baut eine Pruefung mit Status ``insufficient_data``.

    Args:
        key: Setup.
        required: Benoetigte Bars.
        available: Vorhandene Bars.

    Returns:
        Pruefung ohne Messwerte, mit Begruendung.
    """
    label = SETUP_LABELS[key]
    return SetupCheck(
        key=key,
        label=label,
        status=SetupStatus.INSUFFICIENT_DATA,
        explanation=(
            f"{label}: insufficient data ({required} daily bars required, {available} available)"
        ),
    )


def _status(flag: bool) -> SetupStatus:
    """Uebersetzt eine erfuellte Bedingung in den Status.

    Args:
        flag: Ob die Bedingung erfuellt ist.

    Returns:
        ``met`` oder ``not_met``.
    """
    return SetupStatus.MET if flag else SetupStatus.NOT_MET


# --------------------------------------------------------------------------
# Setup 1: Trend-Pullback
# --------------------------------------------------------------------------


def check_pullback(series: PriceSeries) -> SetupCheck:
    """Prueft den Pullback zur 50-Tage-Linie im Aufwaertstrend.

    Args:
        series: Tageskursreihe.

    Returns:
        Pruefung mit den Messwerten ``dist_sma_50`` und ``dist_sma_200``
        (dezimal, vorzeichenbehaftet).
    """
    key = SetupKey.PULLBACK
    close = _frame(series)[_CLOSE]
    if len(close) < PULLBACK_MIN_BARS:
        return _insufficient(key, PULLBACK_MIN_BARS, len(close))

    price = float(close.iloc[-1])
    fast = float(tech.sma(close, SMA_FAST).iloc[-1])
    slow = float(tech.sma(close, SMA_SLOW).iloc[-1])
    dist_fast = _relative(price, fast)
    dist_slow = _relative(price, slow)

    uptrend = price > slow and fast > slow
    near = dist_fast is not None and abs(dist_fast) <= PULLBACK_BAND
    status = _status(uptrend and near)

    label = SETUP_LABELS[key]
    where = _describe_distance(dist_fast, "the 50-day MA")
    if status is SetupStatus.MET:
        reason = where
    elif not uptrend:
        reason = f"no uptrend by definition; {_describe_distance(dist_slow, 'the 200-day MA')}"
    else:
        reason = f"{where}, band is \u00b1{PULLBACK_BAND:.0%}"
    return SetupCheck(
        key=key,
        label=label,
        status=status,
        measurements=[
            Measurement(
                key="dist_sma_50",
                label="Distance to 50-day MA",
                value=dist_fast,
                unit=MeasurementUnit.DECIMAL,
            ),
            Measurement(
                key="dist_sma_200",
                label="Distance to 200-day MA",
                value=dist_slow,
                unit=MeasurementUnit.DECIMAL,
            ),
        ],
        explanation=f"{label}: {_status_text(status)} ({reason})",
    )


def _status_text(status: SetupStatus) -> str:
    """Kurzform des Status fuer Erklaerungssaetze.

    Args:
        status: Status.

    Returns:
        ``met``, ``not met`` oder ``insufficient data``.
    """
    return status.value.replace("_", " ")


# --------------------------------------------------------------------------
# Setup 2: RSI-Erholung
# --------------------------------------------------------------------------


def check_rsi_recovery(series: PriceSeries) -> SetupCheck:
    """Prueft die Erholung des RSI(14) aus dem ueberverkauften Bereich.

    Args:
        series: Tageskursreihe.

    Returns:
        Pruefung mit den Messwerten ``rsi`` (Punkte), ``rsi_min`` (Punkte,
        Minimum der 10 Bars vor dem Stichtag) und ``bars_since_low``
        (Bars). Das Ereignis ist das RSI-Tief, sofern es unter 30 lag.
    """
    key = SetupKey.RSI_RECOVERY
    close = _frame(series)[_CLOSE]
    if len(close) < RSI_MIN_BARS:
        return _insufficient(key, RSI_MIN_BARS, len(close))

    values = tech.rsi(close, RSI_WINDOW)
    today = float(values.iloc[-1])
    window = values.iloc[-1 - RSI_LOOKBACK : -1]
    low_pos = int(np.argmin(window.to_numpy()))
    low = float(window.iloc[low_pos])
    low_at = pd.Timestamp(window.index[low_pos])
    bars_since = RSI_LOOKBACK - low_pos

    was_oversold = low < RSI_OVERSOLD
    status = _status(was_oversold and today >= RSI_OVERSOLD)

    label = SETUP_LABELS[key]
    reason = (
        f"RSI {today:.1f} today, low {low:.1f} {_bars_ago(bars_since)} "
        f"within the prior {RSI_LOOKBACK} bars"
    )
    return SetupCheck(
        key=key,
        label=label,
        status=status,
        measurements=[
            Measurement(key="rsi", label="RSI 14", value=today, unit=MeasurementUnit.INDEX),
            Measurement(
                key="rsi_min",
                label=f"RSI low, prior {RSI_LOOKBACK} bars",
                value=low,
                unit=MeasurementUnit.INDEX,
            ),
            Measurement(
                key="bars_since_low",
                label="Bars since RSI low",
                value=float(bars_since),
                unit=MeasurementUnit.BARS,
            ),
        ],
        explanation=f"{label}: {_status_text(status)} ({reason})",
        event_at=low_at.to_pydatetime() if was_oversold else None,
        event_price=float(close.loc[low_at]) if was_oversold else None,
        event_label=f"RSI low {low:.1f}" if was_oversold else None,
    )


# --------------------------------------------------------------------------
# Setup 3: Ausbruch
# --------------------------------------------------------------------------


def check_breakout(series: PriceSeries) -> SetupCheck:
    """Prueft den Ausbruch ueber das 252-Tage-Schlusskurshoch bei hoeherem Volumen.

    Args:
        series: Tageskursreihe mit Volumen.

    Returns:
        Pruefung mit den Messwerten ``dist_prior_high`` (dezimal,
        vorzeichenbehaftet zum bisherigen Hoch) und ``volume_ratio``
        (Vielfaches des 63-Tage-Durchschnitts). Das Ereignis ist der
        Stichtag, sofern der Schlusskurs ueber dem bisherigen Hoch liegt.
    """
    key = SetupKey.BREAKOUT
    data = _frame(series)
    if len(data) < BREAKOUT_MIN_BARS:
        return _insufficient(key, BREAKOUT_MIN_BARS, len(data))

    close = data[_CLOSE]
    volume = data[_VOLUME]
    price = float(close.iloc[-1])
    prior_high = float(close.iloc[-1 - BREAKOUT_LOOKBACK : -1].max())
    dist_high = _relative(price, prior_high)

    prior_volume = volume.iloc[-1 - VOLUME_LOOKBACK : -1]
    avg_volume = float(prior_volume.mean())
    today_volume = float(volume.iloc[-1])
    label = SETUP_LABELS[key]
    if (
        prior_volume.isna().any()
        or math.isnan(today_volume)
        or math.isnan(avg_volume)
        or avg_volume <= 0
    ):
        return SetupCheck(
            key=key,
            label=label,
            status=SetupStatus.INSUFFICIENT_DATA,
            measurements=[
                Measurement(
                    key="dist_prior_high",
                    label="Distance to prior 252-day high",
                    value=dist_high,
                    unit=MeasurementUnit.DECIMAL,
                ),
                Measurement(
                    key="volume_ratio",
                    label="Volume vs 63-day average",
                    value=None,
                    unit=MeasurementUnit.MULTIPLE,
                ),
            ],
            explanation=f"{label}: insufficient data (volume missing in the last 64 bars)",
        )

    ratio = today_volume / avg_volume
    new_high = price > prior_high
    status = _status(new_high and ratio > VOLUME_RATIO_MIN)
    reason = (
        f"close {_describe_distance(dist_high, 'the prior 252-day closing high')}, "
        f"volume {ratio:.2f}x the 63-day average"
    )
    as_of = pd.Timestamp(close.index[-1])
    return SetupCheck(
        key=key,
        label=label,
        status=status,
        measurements=[
            Measurement(
                key="dist_prior_high",
                label="Distance to prior 252-day high",
                value=dist_high,
                unit=MeasurementUnit.DECIMAL,
            ),
            Measurement(
                key="volume_ratio",
                label="Volume vs 63-day average",
                value=ratio,
                unit=MeasurementUnit.MULTIPLE,
            ),
        ],
        explanation=f"{label}: {_status_text(status)} ({reason})",
        event_at=as_of.to_pydatetime() if new_high else None,
        event_price=price if new_high else None,
        event_label="New 252-day closing high" if new_high else None,
    )


# --------------------------------------------------------------------------
# Setup 4: Unterstuetzung/Widerstand und MACD
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Swing:
    """Ein bestaetigtes Swing-Hoch oder -Tief.

    Attributes:
        position: Nullbasierte Zeilenposition des Extremums in der Reihe.
        at: Zeitstempel des Extremums, tz-aware UTC.
        level: Hoch beziehungsweise Tief in Kurswaehrung.
        is_high: True fuer ein Swing-Hoch, False fuer ein Swing-Tief.
    """

    position: int
    at: pd.Timestamp
    level: float
    is_high: bool


def confirmed_swings(
    series: PriceSeries,
    *,
    lookback: int = SWING_LOOKBACK,
    order: int = SWING_ORDER,
) -> list[Swing]:
    """Findet bestaetigte Swing-Hochs und -Tiefs.

    Ein Swing-Hoch an Position i: ``high[i]`` ist strikt groesser als die
    ``order`` Hochs davor und mindestens so gross wie die ``order`` Hochs
    danach. Swing-Tiefs spiegelbildlich auf ``low``. Nur Extrema, deren
    rechte Nachbarn vollstaendig vorliegen, zaehlen - ein Swing ist damit
    fruehestens ``order`` Bars nach dem Extremum bekannt (kein Look-ahead).

    Args:
        series: Tageskursreihe mit ``high`` und ``low``.
        lookback: Nur Extrema in den letzten ``lookback`` Bars, in Bars.
        order: Nachbarn je Seite, in Bars.

    Returns:
        Swings in zeitlicher Reihenfolge.

    Raises:
        ValueError: Wenn ``lookback`` oder ``order`` nicht positiv ist.
    """
    if lookback <= 0 or order <= 0:
        msg = "lookback und order muessen positiv sein"
        raise ValueError(msg)
    data = _frame(series)
    high = data[_HIGH].to_numpy(dtype="float64")
    low = data[_LOW].to_numpy(dtype="float64")
    index = data.index
    n = len(data)

    swings: list[Swing] = []
    # Letzte bestaetigbare Position: n - 1 - order, deshalb stop exklusiv n - order.
    for i in range(max(order, n - lookback), n - order):
        left_high = high[i - order : i]
        right_high = high[i + 1 : i + order + 1]
        if high[i] > left_high.max() and high[i] >= right_high.max():
            swings.append(Swing(i, pd.Timestamp(index[i]), float(high[i]), is_high=True))
        left_low = low[i - order : i]
        right_low = low[i + 1 : i + order + 1]
        if low[i] < left_low.min() and low[i] <= right_low.min():
            swings.append(Swing(i, pd.Timestamp(index[i]), float(low[i]), is_high=False))
    return swings


def price_zones(
    series: PriceSeries,
    *,
    tolerance: float = ZONE_TOLERANCE,
) -> list[PriceZone]:
    """Fasst bestaetigte Swings zu Unterstuetzungs- und Widerstandszonen zusammen.

    Die Swing-Niveaus werden aufsteigend sortiert. Ein Niveau gehoert zur
    laufenden Zone, solange es hoechstens ``tolerance`` ueber deren
    niedrigstem Niveau liegt. Zonen mit Mittelwert kleiner oder gleich dem
    Kurs am Stichtag sind Unterstuetzung, die uebrigen Widerstand.

    Args:
        series: Tageskursreihe.
        tolerance: Relative Zonenbreite, dezimal.

    Returns:
        Zonen aufsteigend nach Niveau. Leer, wenn es keine bestaetigten
        Swings gibt.

    Raises:
        ValueError: Wenn ``tolerance`` negativ ist.
    """
    if tolerance < 0:
        msg = "tolerance darf nicht negativ sein"
        raise ValueError(msg)
    swings = confirmed_swings(series)
    if not swings:
        return []
    price = float(_frame(series)[_CLOSE].iloc[-1])

    clusters: list[list[float]] = []
    for level in sorted(s.level for s in swings if s.level > 0):
        if clusters and level <= clusters[-1][0] * (1.0 + tolerance):
            clusters[-1].append(level)
        else:
            clusters.append([level])

    zones: list[PriceZone] = []
    for members in clusters:
        mean = float(np.mean(members))
        zones.append(
            PriceZone(
                level=mean,
                lower=min(members),
                upper=max(members),
                kind=ZoneKind.SUPPORT if mean <= price else ZoneKind.RESISTANCE,
                touches=len(members),
            )
        )
    return zones


@dataclass(frozen=True, slots=True)
class MacdCross:
    """Juengste MACD-Kreuzung im Beobachtungsfenster.

    Attributes:
        direction: ``up``, ``down`` oder ``none``.
        bars_since: Bars seit der Kreuzung, 0 = am Stichtag. None ohne
            Kreuzung.
        at: Zeitstempel der Kreuzung, tz-aware UTC, oder None.
        price: Schlusskurs an der Kreuzung in Kurswaehrung, oder None.
    """

    direction: CrossDirection
    bars_since: int | None
    at: pd.Timestamp | None
    price: float | None


def macd_cross(series: PriceSeries, *, window: int = MACD_CROSS_WINDOW) -> MacdCross:
    """Findet die juengste Kreuzung von MACD- und Signallinie.

    Grundlage ist MACD(12, 26, 9) aus :mod:`btlite.analytics.technicals`,
    Kreuzungen ueber :func:`~btlite.analytics.technicals.crossover` und
    :func:`~btlite.analytics.technicals.crossunder`.

    Args:
        series: Tageskursreihe.
        window: Fenster in Bars inklusive Stichtag.

    Returns:
        Richtung und Alter der juengsten Kreuzung im Fenster.

    Raises:
        InsufficientDataError: Wenn die Reihe fuer das ganze Fenster zu kurz
            ist.
    """
    close = _frame(series)[_CLOSE]
    required = MACD_MIN_BARS - MACD_CROSS_WINDOW + window
    if len(close) < required:
        msg = f"MACD cross requires {required} observations, {len(close)} available"
        raise InsufficientDataError(msg, required=required, available=len(close))

    frame = tech.macd(close)
    line, signal = frame[_MACD_LINE], frame[_MACD_SIGNAL]
    up = tech.crossover(line, signal).iloc[-window:].to_numpy()
    down = tech.crossunder(line, signal).iloc[-window:].to_numpy()
    for offset in range(window - 1, -1, -1):
        if up[offset] or down[offset]:
            position = len(close) - window + offset
            return MacdCross(
                direction=CrossDirection.UP if up[offset] else CrossDirection.DOWN,
                bars_since=window - 1 - offset,
                at=pd.Timestamp(close.index[position]),
                price=float(close.iloc[position]),
            )
    return MacdCross(direction=CrossDirection.NONE, bars_since=None, at=None, price=None)


@dataclass(frozen=True, slots=True)
class _Levels:
    """Naechste Zonen unter und ueber dem Kurs samt Abstand (dezimal)."""

    support: float | None
    resistance: float | None
    dist_support: float | None
    dist_resistance: float | None


def _nearest_levels(zones: list[PriceZone], price: float) -> _Levels:
    """Waehlt die naechste Unterstuetzung und den naechsten Widerstand.

    Args:
        zones: Zonen aus :func:`price_zones`.
        price: Kurs am Stichtag.

    Returns:
        Niveaus und Abstaende ``(Kurs - Niveau) / Niveau``, None wenn es
        auf einer Seite keine Zone gibt.
    """
    supports = [z.level for z in zones if z.kind is ZoneKind.SUPPORT]
    resistances = [z.level for z in zones if z.kind is ZoneKind.RESISTANCE]
    support = max(supports) if supports else None
    resistance = min(resistances) if resistances else None
    return _Levels(
        support=support,
        resistance=resistance,
        dist_support=_relative(price, support) if support is not None else None,
        dist_resistance=_relative(price, resistance) if resistance is not None else None,
    )


def check_support_macd(
    series: PriceSeries,
    zones: list[PriceZone] | None = None,
    cross: MacdCross | None = None,
) -> SetupCheck:
    """Prueft Naehe zur Unterstuetzung zusammen mit einer MACD-Kreuzung nach oben.

    Args:
        series: Tageskursreihe.
        zones: Bereits berechnete Zonen, sonst werden sie hier berechnet.
        cross: Bereits berechnete MACD-Kreuzung, sonst wird sie hier
            berechnet.

    Returns:
        Pruefung mit den Messwerten ``support`` (Kurs), ``dist_support``
        (dezimal, >= 0), ``resistance`` (Kurs), ``dist_resistance``
        (dezimal, < 0) und ``bars_since_cross`` (Bars). Das Ereignis ist
        die MACD-Kreuzung, sofern eine im Fenster liegt.
    """
    key = SetupKey.SUPPORT_MACD
    close = _frame(series)[_CLOSE]
    if len(close) < SUPPORT_MIN_BARS:
        return _insufficient(key, SUPPORT_MIN_BARS, len(close))

    price = float(close.iloc[-1])
    levels = _nearest_levels(zones if zones is not None else price_zones(series), price)
    if cross is None:
        cross = macd_cross(series)

    near_support = levels.dist_support is not None and 0.0 <= levels.dist_support <= SUPPORT_BAND
    status = _status(near_support and cross.direction is CrossDirection.UP)

    label = SETUP_LABELS[key]
    if levels.support is None:
        where = "no confirmed support zone below the price"
    else:
        where = _describe_distance(levels.dist_support, f"support {levels.support:.2f}")
    if cross.direction is CrossDirection.NONE:
        macd_text = f"no MACD cross in the last {MACD_CROSS_WINDOW} bars"
    else:
        macd_text = f"MACD crossed {cross.direction.value} {_bars_ago(cross.bars_since or 0)}"
    cross_label = (
        "Bars since MACD cross"
        if cross.direction is CrossDirection.NONE
        else f"Bars since MACD cross {cross.direction.value}"
    )
    return SetupCheck(
        key=key,
        label=label,
        status=status,
        measurements=[
            Measurement(
                key="support",
                label="Nearest support",
                value=levels.support,
                unit=MeasurementUnit.PRICE,
            ),
            Measurement(
                key="dist_support",
                label="Distance to support",
                value=levels.dist_support,
                unit=MeasurementUnit.DECIMAL,
            ),
            Measurement(
                key="resistance",
                label="Nearest resistance",
                value=levels.resistance,
                unit=MeasurementUnit.PRICE,
            ),
            Measurement(
                key="dist_resistance",
                label="Distance to resistance",
                value=levels.dist_resistance,
                unit=MeasurementUnit.DECIMAL,
            ),
            Measurement(
                key="bars_since_cross",
                label=cross_label,
                value=float(cross.bars_since) if cross.bars_since is not None else None,
                unit=MeasurementUnit.BARS,
            ),
        ],
        explanation=f"{label}: {_status_text(status)} ({where}; {macd_text})",
        event_at=cross.at.to_pydatetime() if cross.at is not None else None,
        event_price=cross.price,
        event_label=(
            f"MACD cross {cross.direction.value}"
            if cross.direction is not CrossDirection.NONE
            else None
        ),
    )


# --------------------------------------------------------------------------
# Gesamtbericht
# --------------------------------------------------------------------------


def setup_report(series: PriceSeries) -> SetupReport:
    """Prueft alle vier Setups zum Stichtag, der letzten Bar der Reihe.

    Kein Gesamturteil: der Bericht enthaelt nur die einzelnen Pruefungen,
    die Zonen und die MACD-Kreuzung. Zu kurze Historie ergibt je Setup den
    Status ``insufficient_data``, keinen Fehler.

    Args:
        series: Tageskursreihe.

    Returns:
        Bericht mit Stichtag (tz-aware UTC), Kurs in Kurswaehrung, vier
        Pruefungen und den Zonen. Abstaende dezimal, siehe Modul-Docstring.

    Raises:
        InsufficientDataError: Wenn die Reihe keinen einzigen Schlusskurs
            enthaelt.
    """
    close = _frame(series)[_CLOSE]
    if close.empty:
        msg = f"No closing prices for {series.ticker}"
        raise InsufficientDataError(msg, required=1, available=0)

    price = float(close.iloc[-1])
    as_of = pd.Timestamp(close.index[-1]).to_pydatetime()
    enough = len(close) >= SUPPORT_MIN_BARS
    zones = price_zones(series) if enough else []
    levels = _nearest_levels(zones, price)
    cross = (
        macd_cross(series)
        if enough
        else MacdCross(direction=CrossDirection.NONE, bars_since=None, at=None, price=None)
    )

    return SetupReport(
        ticker=series.ticker,
        as_of=as_of,
        price=price,
        checks=[
            check_pullback(series),
            check_rsi_recovery(series),
            check_breakout(series),
            check_support_macd(series, zones, cross),
        ],
        zones=zones,
        nearest_support=levels.support,
        nearest_resistance=levels.resistance,
        dist_support=levels.dist_support,
        dist_resistance=levels.dist_resistance,
        macd_cross=cross.direction,
        bars_since_cross=cross.bars_since,
    )
