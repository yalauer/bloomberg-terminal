"""Darstellung von Intraday-Charts: Ortszeit und ausgeblendete Pausen.

Zwei Dinge unterscheiden einen Minutenchart von einem Tageschart:

1. **Uhrzeit statt Datum.** Intern ist alles UTC (Architekturregel 6). Eine
   New Yorker Eroeffnung um 13:30 UTC liest aber niemand als 15:30 Berlin.
   Die Umrechnung geschieht hier, an der Anzeigegrenze, und nur fuer die
   Achse - die Kursreihe selbst bleibt UTC.
2. **Pausen.** Eine Woche Minutenkerzen besteht aus fuenf Sitzungen zu je
   6.5 Stunden. Auf einer durchgehenden Zeitachse waeren ueber 70 Prozent
   der Flaeche leer. Die Luecken werden als ``rangebreaks`` ausgeblendet,
   berechnet aus den Daten selbst - ohne Handelszeiten je Boerse zu kennen.

Architekturregel 3: Funktionen geben die Figure zurueck und zeigen nichts selbst an.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise
from typing import TYPE_CHECKING, Final

import pandas as pd

if TYPE_CHECKING:
    from datetime import timedelta

    import plotly.graph_objects as go

#: Eine Luecke gilt als Pause, wenn sie laenger als so viele Kerzen ist.
#: Groesser 1, damit eine einzelne fehlende Minute nicht ausgeblendet wird
#: und die Achse dort ehrlich einen Sprung zeigt.
_GAP_FACTOR: Final[float] = 3.0

#: Beschriftung der Zeitachse, die :mod:`btlite.viz.price` setzt.
_UTC_AXIS_TITLE: Final[str] = "Date (UTC)"


def localize_time_axis(fig: go.Figure, timezone: str) -> go.Figure:
    """Rechnet die Zeitachse aller Spuren in die Anzeigezeitzone um.

    Args:
        fig: Figure mit zeitzonenbewussten x-Werten (UTC).
        timezone: IANA-Name, etwa ``"Europe/Berlin"``.

    Returns:
        Dieselbe Figure. Spuren ohne Zeitachse bleiben unveraendert.
    """
    for trace in fig.data:
        values = getattr(trace, "x", None)
        if values is None or len(values) == 0:
            continue
        try:
            stamps = pd.DatetimeIndex(pd.to_datetime(values))
        except (TypeError, ValueError):
            continue
        if stamps.tz is None:
            continue
        trace.x = stamps.tz_convert(timezone).tz_localize(None)
    label = f"Time ({timezone})"
    fig.for_each_xaxis(
        lambda axis: axis.update(title_text=label) if axis.title.text == _UTC_AXIS_TITLE else None
    )
    return fig


def session_breaks(stamps: pd.DatetimeIndex, step: timedelta) -> list[dict[str, object]]:
    """Berechnet die ``rangebreaks`` fuer die Pausen einer Kerzenfolge.

    Eine Pause beginnt eine Kerzenlaenge nach der letzten Kerze vor der
    Luecke und endet mit der ersten Kerze danach. Gleich lange Pausen
    (etwa jede Nacht) teilen sich einen Eintrag - Plotly wird mit hunderten
    einzelner Eintraege spuerbar langsam.

    Args:
        stamps: Zeitpunkte der Kerzen, aufsteigend, in Anzeigekoordinaten.
        step: Laenge einer Kerze.

    Returns:
        Liste fuer ``xaxis.rangebreaks``; leer, wenn es keine Pause gibt.
    """
    if len(stamps) < 2:
        return []
    step_td = pd.Timedelta(step)
    grouped: dict[int, list[str]] = defaultdict(list)
    ordered = stamps.sort_values()
    for before, after in pairwise(ordered):
        gap = after - before
        if gap <= step_td * _GAP_FACTOR:
            continue
        hidden = gap - step_td
        start = before + step_td
        grouped[int(hidden / pd.Timedelta(milliseconds=1))].append(start.isoformat())
    return [{"values": values, "dvalue": dvalue} for dvalue, values in sorted(grouped.items())]


def hide_session_gaps(fig: go.Figure, step: timedelta) -> go.Figure:
    """Blendet Naechte, Wochenenden und Feiertage auf der Zeitachse aus.

    Gerechnet wird auf der ersten Spur mit Zeitachse, also dem Kurs. Nach
    :func:`localize_time_axis` aufrufen: die Pausen muessen in denselben
    Koordinaten stehen wie die Kerzen.

    Args:
        fig: Figure mit dem Kurs als erster Spur.
        step: Laenge einer Kerze.

    Returns:
        Dieselbe Figure, mit ``rangebreaks`` auf allen x-Achsen.
    """
    for trace in fig.data:
        values = getattr(trace, "x", None)
        if values is None or len(values) < 2:
            continue
        try:
            stamps = pd.DatetimeIndex(pd.to_datetime(values))
        except (TypeError, ValueError):
            continue
        breaks = session_breaks(stamps, step)
        if breaks:
            fig.update_xaxes(rangebreaks=breaks)
        break
    return fig
