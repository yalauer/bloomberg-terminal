"""Gemeinsame Hilfen der Referenzstrategien.

Alle Strategien arbeiten auf demselben Zwischenformat: einem
Schlusskurs-Panel mit einer Spalte je Ticker auf gemeinsamem UTC-Index.
Diese Umformung liegt hier zentral, damit die Strategie-Module nur noch
ihre eigentliche Signal-Logik enthalten.

**Look-ahead-Freiheit** ist hier verankert: :func:`close_panel` sortiert und
vereinigt nur, es wird niemals rueckwaerts gefuellt. Ein ``ffill`` fuellt
ausschliesslich aus der Vergangenheit; ein ``bfill`` waere Look-ahead und
kommt in diesem Modul bewusst nicht vor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pandas as pd

from btlite.core.errors import InsufficientDataError, StrategyConfigError

if TYPE_CHECKING:
    from btlite.models.market import PriceSeries

#: Kleinste sinnvolle Historie: unter zwei Bars gibt es keine Rendite.
MIN_BARS: Final[int] = 2


def close_panel(prices: dict[str, PriceSeries]) -> pd.DataFrame:
    """Baut ein Schlusskurs-Panel aus mehreren Kursreihen.

    Der gemeinsame Index ist die **Vereinigung** aller Einzelindizes; Luecken
    einzelner Ticker werden nur aus der Vergangenheit gefuellt (``ffill``).
    Fuehrende Luecken bleiben NaN und erzeugen damit kein Signal, statt
    einen spaeteren Kurs nach vorne zu ziehen.

    Args:
        prices: Kursreihen je Ticker.

    Returns:
        DataFrame mit UTC-``DatetimeIndex`` und einer float-Spalte je
        Ticker, Spalten stabil nach Ticker sortiert.

    Raises:
        StrategyConfigError: Wenn keine Kursreihe uebergeben wurde.
        InsufficientDataError: Wenn weniger als zwei Bars vorliegen.
    """
    if not prices:
        msg = "No price series supplied"
        raise StrategyConfigError(msg, detail="prices ist leer")

    columns = {
        ticker: pd.to_numeric(series.data["close"], errors="coerce").astype(float)
        for ticker, series in sorted(prices.items())
    }
    panel = pd.DataFrame(columns).sort_index()
    panel = panel.ffill()

    if len(panel) < MIN_BARS:
        msg = "Too few bars for a signal computation"
        raise InsufficientDataError(
            msg,
            detail=f"bars={len(panel)}",
            required=MIN_BARS,
            available=len(panel),
        )
    return panel


def require_history(panel: pd.DataFrame, *, required: int, what: str) -> None:
    """Prueft, dass das Panel ein Fenster der geforderten Laenge zulaesst.

    Args:
        panel: Schlusskurs-Panel.
        required: Mindestanzahl Bars, die die Parameter verlangen.
        what: Bezeichnung fuer die Fehlermeldung, etwa ``"SMA(200)"``.

    Raises:
        InsufficientDataError: Wenn die Historie kuerzer als ``required``
            ist.
    """
    if len(panel) < required:
        msg = f"History too short for {what}"
        raise InsufficientDataError(
            msg,
            detail=f"benoetigt={required}, vorhanden={len(panel)}",
            required=required,
            available=len(panel),
        )


def as_bool_frame(frame: pd.DataFrame, *, like: pd.DataFrame) -> pd.DataFrame:
    """Normalisiert eine Signalmatrix auf lueckenlose bool-Werte.

    Args:
        frame: Rohe Signalmatrix, darf NaN enthalten.
        like: Panel, dessen Index und Spalten das Ergebnis uebernimmt.

    Returns:
        Boolescher DataFrame ohne fehlende Werte, deckungsgleich mit
        ``like``.
    """
    return frame.reindex(index=like.index, columns=like.columns).fillna(False).astype(bool)


def require_positive_number(params: dict[str, object], name: str) -> float:
    """Liest einen Parameter als positive Zahl.

    Ergaenzt ``Strategy.require_positive_int`` fuer Parameter, die auch
    gebrochen sein duerfen (Schwellen, Anteile).

    Args:
        params: Parameter-Dictionary.
        name: Name des Parameters.

    Returns:
        Der validierte Wert als float.

    Raises:
        StrategyConfigError: Wenn der Wert fehlt oder nicht positiv ist.
    """
    value = params.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        msg = f"Parameter {name!r} must be a positive number"
        raise StrategyConfigError(msg, detail=f"erhalten={value!r}")
    return float(value)
