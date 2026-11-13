"""Chart-Ebene der Setup-Checkliste: Zonen und Ereignis-Marker.

Legt auf einen bestehenden Kurschart aus :mod:`btlite.viz.price` die
Unterstuetzungs- und Widerstandszonen als horizontale Linien mit Band sowie
Marker fuer RSI-Tief, Ausbruchstag und MACD-Kreuzung.

Architekturregel 3: dieses Modul rechnet nicht. Zonen, Ereigniszeitpunkte
und Markerkurse stehen fertig im :class:`~btlite.models.setups.SetupReport`.

Farbe traegt nie allein die Bedeutung: jede Linie hat einen Legendennamen
wie "Support 182.40" und eine Beschriftung am rechten Rand, jeder Marker
eine Textbeschriftung und eine eigene Form.

Keine Anlageempfehlung: die Beschriftungen benennen Beobachtungen
("MACD cross up"), keine Handlungen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import plotly.graph_objects as go

from btlite.models.setups import SetupKey, ZoneKind
from btlite.viz import theme

if TYPE_CHECKING:
    from datetime import datetime

    from btlite.models.setups import SetupCheck, SetupReport

#: Deckkraft der Zonenbaender, dezimal.
ZONE_BAND_OPACITY: Final[float] = 0.12

#: Markerform je Setup. Die Form unterscheidet die Ereignisse auch ohne Farbe.
_MARKER_SYMBOLS: Final[dict[SetupKey, str]] = {
    SetupKey.RSI_RECOVERY: "circle-open",
    SetupKey.BREAKOUT: "star",
    SetupKey.SUPPORT_MACD: "diamond",
}

#: Legendengruppe aller Setup-Elemente, damit sie gemeinsam ausblendbar sind.
_LEGEND_GROUP: Final[str] = "setups"

#: Legendengruppe der Ereignis-Marker.
_EVENT_GROUP: Final[str] = f"{_LEGEND_GROUP}-events"

#: Name des Sammeleintrags der Ereignis-Marker.
_EVENT_LEGEND: Final[str] = "Setup events"


def _zone_group(kind: ZoneKind) -> str:
    """Legendengruppe aller Zonen einer Art.

    Args:
        kind: Unterstuetzung oder Widerstand.

    Returns:
        Kennung, etwa ``"setups-support"``.
    """
    return f"{_LEGEND_GROUP}-{kind.value}"


def zone_legend_name(kind: ZoneKind, count: int) -> str:
    """Name des Sammeleintrags einer Zonenart, etwa ``Support zones (4)``.

    Args:
        kind: Unterstuetzung oder Widerstand.
        count: Anzahl der Zonen dieser Art.

    Returns:
        Legendenname.
    """
    return f"{kind.value.capitalize()} zones ({count})"


def _zone_name(kind: ZoneKind, level: float) -> str:
    """Legendenname einer Zone, etwa ``Support 182.40``.

    Args:
        kind: Unterstuetzung oder Widerstand.
        level: Niveau in Kurswaehrung.

    Returns:
        Beschriftung mit Art und Niveau.
    """
    return f"{kind.value.capitalize()} {level:.2f}"


def _in_range(when: datetime, start: datetime | None, end: datetime | None) -> bool:
    """Prueft, ob ein Zeitpunkt im dargestellten Zeitraum liegt.

    Args:
        when: Zeitpunkt des Ereignisses.
        start: Beginn des Charts oder None.
        end: Ende des Charts oder None.

    Returns:
        True, wenn der Zeitpunkt innerhalb liegt oder keine Grenze gesetzt ist.
    """
    if start is not None and when < start:
        return False
    return not (end is not None and when > end)


def _add_marker(fig: go.Figure, check: SetupCheck, *, first: bool) -> None:
    """Setzt den Ereignis-Marker einer Pruefung.

    Args:
        fig: Zielfigur.
        check: Pruefung mit gesetztem ``event_at`` und ``event_price``.
        first: Ob dieser Marker den Sammeleintrag "Setup events" in der
            Legende traegt. Die uebrigen erscheinen dort nicht.
    """
    if check.event_at is None or check.event_price is None:
        return
    label = check.event_label or check.label
    color = theme.CYAN if check.key is SetupKey.SUPPORT_MACD else theme.AMBER
    symbol = _MARKER_SYMBOLS.get(check.key, "circle")
    if check.key is SetupKey.SUPPORT_MACD and label.endswith("down"):
        color = theme.INK_MUTED
    fig.add_trace(
        go.Scatter(
            x=[check.event_at],
            y=[check.event_price],
            mode="markers+text",
            name=_EVENT_LEGEND if first else f"{label} {check.event_at:%Y-%m-%d}",
            text=[label],
            textposition="top center",
            textfont={"color": theme.INK, "size": theme.FONT_SIZE},
            marker={
                "symbol": symbol,
                "size": 12,
                "color": color,
                "line": {"color": theme.INK, "width": 1},
            },
            hovertemplate=f"{label}<br>%{{x|%Y-%m-%d}}: %{{y:,.2f}}<extra></extra>",
            # Ein Sammeleintrag fuer alle Marker; die Beschriftung steht am
            # Marker selbst.
            legendgroup=_EVENT_GROUP,
            showlegend=first,
        )
    )


def add_setup_annotations(
    fig: go.Figure,
    report: SetupReport,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    markers: bool = True,
) -> go.Figure:
    """Zeichnet Zonen und Ereignis-Marker in den Kursplot.

    Zonen erscheinen als horizontale Linie auf ihrem Niveau (Kurswaehrung)
    ueber die ganze Breite des Kursplots, mit einem schwachen Band von
    ``lower`` bis ``upper``. Marker erscheinen nur, wenn ihr Zeitpunkt im
    dargestellten Zeitraum liegt - sonst wuerde die Zeitachse ueber den
    gewaehlten Zeitraum hinaus gedehnt.

    Die Elemente liegen auf den Achsen ``x``/``y``, also im ersten
    Subplot. Das ist bei allen Charts aus :mod:`btlite.viz.price` der
    Kursplot.

    Args:
        fig: Kurschart, wird ergaenzt und zurueckgegeben.
        report: Fertiger Setup-Bericht.
        start: Beginn des dargestellten Zeitraums, tz-aware UTC, oder None.
        end: Ende des dargestellten Zeitraums, tz-aware UTC, oder None.
        markers: Ereignis-Marker zeichnen. Im Intraday-Chart aus: die
            Ereignisse sind Tageswerte, ihr Zeitstempel faellt dort in eine
            ausgeblendete Nacht.

    Returns:
        Dieselbe Figure mit Zonen und Markern.
    """
    # Die Legende liegt ueber dem Chart. Ein Eintrag je Zone liess sie bei
    # zehn Zonen in mehrere Zeilen umbrechen, die von der festen Chart-Hoehe
    # abgingen - der Kursbereich wurde gestaucht. Deshalb ein Sammeleintrag
    # je Art: ein Klick blendet alle Zonen dieser Art ein oder aus. Das
    # Niveau jeder Zone steht als Beschriftung am rechten Rand, die ganze
    # Liste aufklappbar unter dem Chart.
    counts = {kind: sum(1 for z in report.zones if z.kind is kind) for kind in ZoneKind}
    legend_shown: set[ZoneKind] = set()
    for zone in report.zones:
        color = theme.UP if zone.kind is ZoneKind.SUPPORT else theme.DOWN
        name = _zone_name(zone.kind, zone.level)
        group = _zone_group(zone.kind)
        first = zone.kind not in legend_shown
        legend_shown.add(zone.kind)
        if zone.upper > zone.lower:
            fig.add_shape(
                type="rect",
                xref="x domain",
                yref="y",
                x0=0.0,
                x1=1.0,
                y0=zone.lower,
                y1=zone.upper,
                fillcolor=color,
                opacity=ZONE_BAND_OPACITY,
                line={"width": 0},
                layer="below",
                legendgroup=group,
            )
        fig.add_shape(
            type="line",
            xref="x domain",
            yref="y",
            x0=0.0,
            x1=1.0,
            y0=zone.level,
            y1=zone.level,
            line={"color": color, "width": 1, "dash": "dash"},
            name=zone_legend_name(zone.kind, counts[zone.kind]),
            showlegend=first,
            legendgroup=group,
            label={
                "text": name,
                "textposition": "end",
                "font": {"color": theme.INK, "size": theme.FONT_SIZE},
            },
        )

    first_marker = True
    for check in report.checks if markers else []:
        if check.event_at is not None and _in_range(check.event_at, start, end):
            _add_marker(fig, check, first=first_marker)
            first_marker = False
    return fig
