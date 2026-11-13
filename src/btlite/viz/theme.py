"""Zentrales Plotly-Theme und Darstellungs-Helper von BT-Lite.

Architekturregel 3: Dieses Modul rechnet nicht. Es definiert Farben, das
registrierte Plotly-Template und Helper, die alle Chart-Factories teilen.

Farbwahl - Gruen/Rot, aber gemessen
-----------------------------------
Steigend ist gruen, fallend rot - die vertraute Boersenkodierung:

    UP   = #00E676 (Gruen)    DOWN = #E03131 (Rot)

Das ist die Kodierung, die am haeufigsten schiefgeht. Protanopie und
Deuteranopie betreffen rund acht Prozent der maennlichen Bevoelkerung, und
ein *typisches* Gruen/Rot-Paar faellt fuer sie zusammen: nachgemessen
(OKLab-Delta-E, CVD-Simulation nach Machado-Oliveira-Fernandes 2009 bei
Severity 1.0) liegt etwa #199E70 gegen #E66767 bei Delta E 6.5 - unter der
Schwelle von 8, also nicht sicher trennbar.

Dieses Paar ist deshalb nicht gegriffen, sondern gesucht worden. Gemessen:

    normal 43.5    Protanopie 38.2    Deuteranopie 20.3

Der Trick ist der Helligkeitsunterschied. Ein *helles* Gruen gegen ein
*dunkles* Rot bleibt auch dann unterscheidbar, wenn die Farbinformation
wegfaellt - was zusammenfaellt, ist der Farbton, nicht die Helligkeit.
Dunklere Gruentoene derselben Familie bestehen die Pruefung nicht: #16A34A
kommt gegen dasselbe Rot nur auf 4.5, #1DB954 auf 6.5.

Beide Farben liegen ueber 4.5:1 Kontrast gegen den schwarzen Grund. Und
unabhaengig von der Farbe traegt jede Kerze ihre Richtung in der Geometrie,
jede Kennzahl ihren Pfeil und ihr Vorzeichen: Farbe ist nie der einzige
Traeger der Aussage.

Die Kategorienpalette ist die dark-Variante der validierten Referenzpalette.
Fuer Streudiagramme und Kleinserien-Vergleiche gilt ein Serien-Limit von
drei Slots: nur die ersten drei bestehen die Pruefung ueber **alle** Paare
(schlechtestes Paar Delta E 9.4). Darueber hinaus greift
:data:`CATEGORICAL` mit der Adjazenz-Garantie, die fuer Linien und Balken
ausreicht, weil dort nur Nachbarn aneinandergrenzen.
"""

from __future__ import annotations

from typing import Any, Final

import plotly.graph_objects as go
import plotly.io as pio

# --------------------------------------------------------------------------
# Grundflaechen und Schrift
# --------------------------------------------------------------------------

#: Zeichenflaeche der Charts. Schwarz wie ein Terminal.
SURFACE: Final[str] = "#000000"

#: Flaeche hinter der Zeichenflaeche (Papier).
PAPER: Final[str] = "#000000"

#: Leicht abgesetzte Flaeche fuer Tabellen-Kopfzeilen und Hover-Boxen.
SURFACE_RAISED: Final[str] = "#12120F"

#: Primaerfarbe des Terminals. Bernstein, 10.3:1 Kontrast gegen Schwarz.
AMBER: Final[str] = "#FFA028"

#: Gedimmtes Bernstein fuer sekundaere Beschriftungen.
AMBER_DIM: Final[str] = "#B36F16"

#: Sekundaerfarbe. Cyan, 11.6:1 Kontrast gegen Schwarz.
CYAN: Final[str] = "#22D3EE"

#: Haupttext in Diagrammen.
INK: Final[str] = "#E8E6DC"

#: Nachgeordneter Text, etwa Achsentitel.
INK_MUTED: Final[str] = "#898781"

#: Gitterlinien. Bewusst zurueckhaltend (1.5:1), damit Daten dominieren.
GRID: Final[str] = "#2C2C2A"

#: Achsen- und Nulllinie, etwas kraeftiger als das Gitter.
AXIS_LINE: Final[str] = "#4A4A46"

#: Durchgehende Monospace-Schrift, wie in einem Terminal.
FONT_FAMILY: Final[str] = (
    '"JetBrains Mono","Cascadia Mono","Consolas","DejaVu Sans Mono",'
    '"Courier New",ui-monospace,monospace'
)

#: Basis-Schriftgroesse in Punkt.
FONT_SIZE: Final[int] = 11

# --------------------------------------------------------------------------
# Richtungsfarben - siehe Modul-Docstring zur Begruendung
# --------------------------------------------------------------------------

#: Steigend. Helles Gruen - die Helligkeit traegt die Unterscheidung,
#: wenn die Farbwahrnehmung ausfaellt. Siehe Modul-Docstring.
UP: Final[str] = "#00E676"

#: Fallend. Dunkles Rot, Gegenstueck zu :data:`UP`.
DOWN: Final[str] = "#E03131"

#: Gedaempftes Steigend fuer Flaechen und Volumenbalken.
UP_FILL: Final[str] = "rgba(0,230,118,0.55)"

#: Gedaempftes Fallend fuer Flaechen und Volumenbalken.
DOWN_FILL: Final[str] = "rgba(224,49,49,0.55)"

# --------------------------------------------------------------------------
# Kategorien-, Sequential- und Diverging-Paletten
# --------------------------------------------------------------------------

#: Kategorienpalette fuer Mehr-Serien-Charts. Feste Reihenfolge, nie zyklisch
#: erweitert. Die Reihenfolge ist der Mechanismus der Farbenblind-Sicherheit.
CATEGORICAL: Final[tuple[str, ...]] = (
    "#3987E5",  # 1 blau
    "#D95926",  # 2 orange
    "#199E70",  # 3 aqua
    "#C98500",  # 4 gelb
    "#D55181",  # 5 magenta
    "#008300",  # 6 gruen
    "#9085E9",  # 7 violett
    "#E66767",  # 8 rot
)

#: Obergrenze fuer Darstellungsformen, in denen beliebige Marken benachbart
#: sein koennen (Streudiagramme). Nur diese Slots bestehen die Pruefung ueber
#: alle Paare. Darueber hinaus zusammenfassen statt neue Farben erfinden.
CATEGORICAL_SCATTER_CAP: Final[int] = 3

#: Sequentielle Rampe fuer Betragsgroessen (Heatmaps), hell nach dunkel
#: umgekehrt fuer den dunklen Grund: niedrig = dunkel, hoch = hell.
SEQUENTIAL: Final[tuple[str, ...]] = (
    "#0D366B",
    "#184F95",
    "#256ABF",
    "#3987E5",
    "#5598E7",
    "#86B6EF",
    "#CDE2FB",
)

#: Divergierende Rampe fuer Polaritaet (Korrelation, Monatsrenditen).
#: Rot-Pol, neutraler Grau-Mittelpunkt, Gruen-Pol. Symmetrisch um 0 und
#: an :data:`UP` und :data:`DOWN` angebunden, damit ein positiver Monat in
#: der Heatmap dieselbe Farbe traegt wie eine steigende Kerze.
#:
#: Die Helligkeit steigt **monoton** von der roten zur gruenen Seite. Das
#: ist hier der eigentliche Mechanismus: zwei gleich dunkle Toene an den
#: Enden - ein dunkles Rot gegen ein dunkles Gruen - faellen bei
#: Farbenblindheit zusammen (gemessen 6.2, unter der Schwelle von 8). Mit
#: dem Helligkeitsverlauf liegen die Pole bei 55.2.
DIVERGING: Final[tuple[str, ...]] = (
    "#6E1616",
    "#9C2020",
    "#C92B2B",
    "#E86B6B",
    "#383835",
    "#2BC96F",
    "#00E676",
    "#5EF0A0",
    "#A8F8CE",
)

#: Status-Farben. Reservierte Bedeutung, nie als Serienfarbe verwendet.
STATUS_GOOD: Final[str] = "#0CA30C"
STATUS_WARNING: Final[str] = "#FAB219"
STATUS_CRITICAL: Final[str] = "#D03B3B"

#: Name des registrierten Plotly-Templates.
TEMPLATE_NAME: Final[str] = "btlite"

#: Text fuer nicht vorhandene Werte. Nie "0", nie leer.
NA_TEXT: Final[str] = "n/a"


def _build_template() -> go.layout.Template:
    """Baut das Plotly-Template des Terminals.

    Returns:
        Template mit schwarzem Grund, Monospace-Schrift und den
        Farbkonstanten dieses Moduls.
    """
    axis = {
        "gridcolor": GRID,
        "gridwidth": 1,
        "linecolor": AXIS_LINE,
        "linewidth": 1,
        "zerolinecolor": AXIS_LINE,
        "zerolinewidth": 1,
        "tickcolor": AXIS_LINE,
        "tickfont": {"color": INK_MUTED, "family": FONT_FAMILY, "size": FONT_SIZE - 1},
        "title": {"font": {"color": INK_MUTED, "family": FONT_FAMILY, "size": FONT_SIZE}},
        "showline": True,
        "automargin": True,
    }
    layout = go.Layout(
        paper_bgcolor=PAPER,
        plot_bgcolor=SURFACE,
        font={"family": FONT_FAMILY, "size": FONT_SIZE, "color": INK},
        title={
            "font": {"family": FONT_FAMILY, "size": FONT_SIZE + 3, "color": AMBER},
            "x": 0.0,
            "xanchor": "left",
        },
        colorway=list(CATEGORICAL),
        colorscale={
            "sequential": [[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
            "diverging": [[i / (len(DIVERGING) - 1), c] for i, c in enumerate(DIVERGING)],
        },
        xaxis=axis,
        yaxis=axis,
        legend={
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": GRID,
            "borderwidth": 1,
            "font": {"family": FONT_FAMILY, "size": FONT_SIZE - 1, "color": INK},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0.0,
        },
        hoverlabel={
            "bgcolor": SURFACE_RAISED,
            "bordercolor": AMBER_DIM,
            "font": {"family": FONT_FAMILY, "size": FONT_SIZE, "color": INK},
        },
        hovermode="x unified",
        margin={"l": 60, "r": 24, "t": 48, "b": 40},
        separators=".,",
    )
    return go.layout.Template(layout=layout)


def register_template() -> str:
    """Registriert das BT-Lite-Template in ``plotly.io.templates``.

    Idempotent: mehrfacher Aufruf ueberschreibt lediglich denselben Eintrag.
    Setzt bewusst **nicht** ``pio.templates.default``, damit der Import
    dieses Moduls keine globale Nebenwirkung fuer fremde Figures hat.

    Returns:
        Name des registrierten Templates.
    """
    pio.templates[TEMPLATE_NAME] = _build_template()
    return TEMPLATE_NAME


# Registrierung beim Import, damit ``template=TEMPLATE_NAME`` ueberall wirkt.
register_template()


def apply_theme(fig: go.Figure, *, title: str | None = None) -> go.Figure:
    """Setzt Template und optional den Titel auf eine Figure.

    Args:
        fig: Zu gestaltende Figure. Wird in-place veraendert.
        title: Optionale Ueberschrift.

    Returns:
        Dieselbe Figure, zur Verkettung.
    """
    fig.update_layout(template=TEMPLATE_NAME)
    if title is not None:
        fig.update_layout(title={"text": title})
    return fig


def series_color(index: int) -> str:
    """Liefert die Kategorienfarbe eines Serien-Index.

    Die Zuordnung folgt der festen Reihenfolge aus :data:`CATEGORICAL`. Ab
    dem neunten Index wiederholt sich die Palette; das ist bewusst der
    Notfall und kein Entwurfsziel - Aufrufer sollten vorher zusammenfassen.

    Args:
        index: Nullbasierter Serien-Index.

    Returns:
        Hex-Farbwert.
    """
    return CATEGORICAL[index % len(CATEGORICAL)]


def sequential_scale() -> list[list[Any]]:
    """Sequentielle Farbskala fuer Betragsgroessen.

    Returns:
        Plotly-Colorscale als Liste aus ``[position, farbe]``, Position
        dezimal zwischen 0 und 1.
    """
    return [[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)]


def diverging_scale() -> list[list[Any]]:
    """Divergierende Farbskala fuer Polaritaet, neutral in der Mitte.

    Returns:
        Plotly-Colorscale als Liste aus ``[position, farbe]``, Position
        dezimal zwischen 0 und 1.
    """
    return [[i / (len(DIVERGING) - 1), c] for i, c in enumerate(DIVERGING)]


def style_axes(
    fig: go.Figure,
    *,
    x_title: str | None = None,
    y_title: str | None = None,
    log_y: bool = False,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Vereinheitlicht Achsenbeschriftung und Skalierung.

    Args:
        fig: Zu gestaltende Figure.
        x_title: Titel der X-Achse, inklusive Einheit.
        y_title: Titel der Y-Achse, inklusive Einheit.
        log_y: Logarithmische Y-Achse. Rein darstellungsbezogen.
        row: Subplot-Zeile, einsbasiert. None = alle.
        col: Subplot-Spalte, einsbasiert. None = alle.

    Returns:
        Dieselbe Figure, zur Verkettung.
    """
    target: dict[str, int] = {}
    if row is not None:
        target["row"] = row
    if col is not None:
        target["col"] = col
    if x_title is not None:
        fig.update_xaxes(title_text=x_title, **target)
    if y_title is not None:
        fig.update_yaxes(title_text=y_title, **target)
    if log_y:
        fig.update_yaxes(type="log", **target)
    return fig


def add_crosshair(fig: go.Figure) -> go.Figure:
    """Aktiviert das Fadenkreuz-Hover ueber alle Subplots.

    Args:
        fig: Zu gestaltende Figure.

    Returns:
        Dieselbe Figure, zur Verkettung.
    """
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor=AMBER_DIM,
        spikethickness=1,
        spikedash="dot",
    )
    fig.update_yaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor=AMBER_DIM,
        spikethickness=1,
        spikedash="dot",
    )
    return fig


#: Zeitraum-Schnellwahl der Kursseiten. 1M/3M/6M/1Y/5Y/MAX.
RANGE_SELECTOR_BUTTONS: Final[tuple[dict[str, Any], ...]] = (
    {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
    {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
    {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
    {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
    {"count": 5, "label": "5Y", "step": "year", "stepmode": "backward"},
    {"step": "all", "label": "MAX"},
)


def add_range_selector(
    fig: go.Figure, *, row: int | None = None, col: int | None = None
) -> go.Figure:
    """Haengt die Zeitraum-Schnellwahl an eine Zeitachse.

    Args:
        fig: Zu gestaltende Figure.
        row: Subplot-Zeile, einsbasiert. None = erste X-Achse.
        col: Subplot-Spalte, einsbasiert. None = erste X-Achse.

    Returns:
        Dieselbe Figure, zur Verkettung.
    """
    target: dict[str, int] = {}
    if row is not None:
        target["row"] = row
    if col is not None:
        target["col"] = col
    fig.update_xaxes(
        rangeselector={
            "buttons": [dict(b) for b in RANGE_SELECTOR_BUTTONS],
            "bgcolor": SURFACE_RAISED,
            "activecolor": AMBER_DIM,
            "bordercolor": GRID,
            "borderwidth": 1,
            "font": {"family": FONT_FAMILY, "size": FONT_SIZE - 1, "color": INK},
            "x": 0.0,
            "y": 1.0,
        },
        **target,
    )
    return fig


def hover_template(
    *,
    label: str,
    value_format: str = ".2f",
    unit: str = "",
    include_x: bool = False,
) -> str:
    """Baut ein einheitliches Hover-Template.

    Args:
        label: Bezeichnung der Groesse, erscheint vor dem Wert.
        value_format: d3-Formatstring des Y-Werts, etwa ``".2%"``.
        unit: Optionale Einheit, wird angehaengt.
        include_x: Wenn True, wird der X-Wert mit ausgegeben. Bei
            ``hovermode="x unified"`` unnoetig, da die Achse ihn schon zeigt.

    Returns:
        Plotly-Hovertemplate-String.
    """
    suffix = f" {unit}" if unit else ""
    x_part = "%{x|%Y-%m-%d}<br>" if include_x else ""
    return f"{x_part}{label}: %{{y:{value_format}}}{suffix}<extra></extra>"


def format_value(value: float | None, *, value_format: str = ",.2f", unit: str = "") -> str:
    """Formatiert einen Wert fuer die Anzeige, fehlende Werte als "n/a".

    Setzt die Projektregel um, dass ein nicht berechenbarer Wert nie als
    ``0`` erscheint.

    Args:
        value: Anzuzeigender Wert, oder None wenn nicht berechenbar.
        value_format: Python-Formatspezifikation fuer den Zahlenteil.
        unit: Optionale Einheit, wird angehaengt.

    Returns:
        Formatierter Text, oder :data:`NA_TEXT` bei fehlendem Wert.
    """
    import math

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return NA_TEXT
    suffix = f" {unit}" if unit else ""
    return f"{value:{value_format}}{suffix}"


def empty_figure(message: str = "No data", *, title: str | None = None) -> go.Figure:
    """Erzeugt eine Hinweis-Figure statt eines leeren Rahmens.

    Jede Factory ruft das auf, wenn ihr Eingangsmodell keine darstellbaren
    Werte enthaelt. So sieht der Nutzer den Grund, statt ein leeres Gitter
    zu deuten.

    Args:
        message: Zentrierter Hinweistext.
        title: Optionale Ueberschrift, damit der Chart-Rahmen benannt bleibt.

    Returns:
        Figure ohne Achsen mit zentriertem Hinweis.
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"family": FONT_FAMILY, "size": FONT_SIZE + 2, "color": AMBER_DIM},
    )
    fig.update_layout(
        template=TEMPLATE_NAME,
        xaxis={"visible": False},
        yaxis={"visible": False},
        hovermode=False,
        showlegend=False,
    )
    if title is not None:
        fig.update_layout(title={"text": title})
    return fig
