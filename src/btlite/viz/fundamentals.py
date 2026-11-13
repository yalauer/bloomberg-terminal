"""Chart-Factories fuer Block 2: Statements, Kennzahlen und Peer-Vergleich.

Architekturregel 3: Dieses Modul rechnet nicht. Kennzahlen kommen fertig aus
``analytics/ratios.py``, Statements normalisiert aus
``analytics/statements.py``.

Entscheidung zur Statement-Tabelle
----------------------------------
Die Statement-Tabelle wird als ``go.Table``-Figure gebaut, nicht als
Dash-``DataTable``-Struktur. Begruendung:

1. Architekturregel 3 verlangt, dass ``viz/`` Figures zurueckgibt. Eine
   Dash-Komponente waere ein Bruch dieser Signatur und wuerde ``viz/`` an
   das UI-Framework binden.
2. Damit bleibt die Tabelle ohne laufendes Dash testbar und exportierbar,
   genau wie jeder andere Chart.

Der Preis ist, dass Sortieren und Filtern im Browser entfallen. Wo das
gebraucht wird, baut ``ui/`` eine eigene ``DataTable`` direkt aus dem
Modell - dafuer stellt :func:`statement_table_records` die reine
Datenstruktur bereit, ohne Figure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from btlite.models.fundamentals import LineItem, RatioCategory
from btlite.viz import theme

if TYPE_CHECKING:
    from btlite.models.fundamentals import (
        FinancialStatement,
        PeerComparison,
        RatioSet,
    )

#: Anzeigenamen der Kennzahlen-Kategorien.
CATEGORY_LABELS: dict[RatioCategory, str] = {
    RatioCategory.LIQUIDITY: "Liquidity",
    RatioCategory.LEVERAGE: "Leverage",
    RatioCategory.PROFITABILITY: "Profitability",
    RatioCategory.EFFICIENCY: "Efficiency",
    RatioCategory.VALUATION: "Valuation",
    RatioCategory.GROWTH: "Growth",
}

#: Skalierungsschwellen fuer die Kurzdarstellung grosser Betraege.
_MAGNITUDES: tuple[tuple[float, str], ...] = (
    (1e12, "T"),
    (1e9, "B"),
    (1e6, "M"),
    (1e3, "K"),
)


def _format_amount(value: object) -> str:
    """Formatiert einen Statement-Betrag kompakt, fehlende Werte als "n/a".

    Args:
        value: Roher Zellwert. ``None`` und ``NaN`` gelten als fehlend.

    Returns:
        Gekuerzter Betrag mit Groessenordnung, oder "n/a".
    """
    if value is None or not isinstance(value, int | float) or bool(pd.isna(value)):
        return theme.NA_TEXT
    number = float(value)
    sign = "-" if number < 0 else ""
    magnitude = abs(number)
    for threshold, suffix in _MAGNITUDES:
        if magnitude >= threshold:
            return f"{sign}{magnitude / threshold:,.2f} {suffix}"
    return f"{sign}{magnitude:,.2f}"


def _pretty_line_item(raw: object) -> str:
    """Macht einen Positionsbezeichner lesbar.

    Args:
        raw: Kanonischer Bezeichner oder Rohname aus dem Index.

    Returns:
        Lesbarer Text, etwa "Net Income" aus ``net_income``.
    """
    return str(raw).replace("_", " ").title()


def statement_table_records(statement: FinancialStatement) -> list[dict[str, str]]:
    """Baut die Statement-Tabelle als reine Datenstruktur.

    Fuer ``ui/``, wenn dort eine interaktive Dash-``DataTable`` noetig ist.
    Diese Funktion erzeugt bewusst keine Figure.

    Args:
        statement: Normalisiertes Statement. Betraege in
            ``statement.currency``, absolut.

    Returns:
        Eine Zeile je Position. Schluessel ``"position"`` plus je Periode
        ein Schluessel im Format ``"YYYY-MM-DD"``. Fehlende Werte stehen
        als "n/a" im Text.
    """
    if statement.data.empty:
        return []
    periods = [str(p) for p in statement.periods]
    records: list[dict[str, str]] = []
    for item, row in statement.data.iterrows():
        record = {"position": _pretty_line_item(item)}
        for period, value in zip(periods, row.tolist(), strict=False):
            record[period] = _format_amount(value)
        records.append(record)
    return records


def statement_table(
    statement: FinancialStatement,
    *,
    title: str | None = None,
    max_rows: int | None = None,
) -> go.Figure:
    """Statement-Tabelle als Figure.

    Args:
        statement: Normalisiertes Statement. Betraege in
            ``statement.currency``, absolut (nicht in Millionen).
        title: Ueberschrift. Default nennt Ticker und Statement-Art.
        max_rows: Obergrenze der dargestellten Positionen. None = alle.

    Returns:
        Figure mit einer Tabelle. Fehlende Werte erscheinen als "n/a",
        nie als 0. Bei leerem Payload eine Hinweis-Figure.
    """
    head = (
        title
        if title is not None
        else f"{statement.ticker} - {statement.statement_type.value} ({statement.period.value})"
    )
    if statement.data.empty:
        return theme.empty_figure("No statement available", title=head)

    frame = statement.data if max_rows is None else statement.data.head(max_rows)
    periods = [str(p) for p in statement.periods]
    columns = ["Line item", *periods]
    cells: list[list[str]] = [[_pretty_line_item(i) for i in frame.index]]
    for pos in range(len(periods)):
        cells.append([_format_amount(v) for v in frame.iloc[:, pos].tolist()])

    fig = go.Figure(
        go.Table(
            header={
                "values": [f"<b>{c}</b>" for c in columns],
                "fill_color": theme.SURFACE_RAISED,
                "font": {
                    "color": theme.AMBER,
                    "family": theme.FONT_FAMILY,
                    "size": theme.FONT_SIZE,
                },
                "align": ["left", *["right"] * len(periods)],
                "line_color": theme.GRID,
            },
            cells={
                "values": cells,
                "fill_color": theme.SURFACE,
                "font": {"color": theme.INK, "family": theme.FONT_FAMILY, "size": theme.FONT_SIZE},
                "align": ["left", *["right"] * len(periods)],
                "line_color": theme.GRID,
                "height": 22,
            },
            columnwidth=[2.2, *[1.0] * len(periods)],
        )
    )
    theme.apply_theme(fig, title=f"{head} - amounts in {statement.currency.value}")
    fig.update_layout(height=min(760, 120 + 22 * len(frame)))
    return fig


def statement_trend(
    statement: FinancialStatement,
    items: tuple[LineItem | str, ...] = (
        LineItem.REVENUE,
        LineItem.NET_INCOME,
        LineItem.FREE_CASHFLOW,
    ),
    *,
    title: str | None = None,
) -> go.Figure:
    """Balkendiagramm ausgewaehlter Positionen ueber die Perioden.

    Der uebliche Anwendungsfall ist Revenue, Net Income und Free Cashflow
    nebeneinander.

    Args:
        statement: Normalisiertes Statement. Betraege in
            ``statement.currency``, absolut.
        items: Darzustellende Positionen. Nicht vorhandene Positionen
            werden uebersprungen, nicht als Null gezeichnet.
        title: Ueberschrift.

    Returns:
        Gruppiertes Balkendiagramm, eine Gruppe je Periode. Bei leerem
        Payload oder wenn keine der Positionen vorliegt, eine Hinweis-Figure.
    """
    head = title if title is not None else f"{statement.ticker} - trend"
    if statement.data.empty:
        return theme.empty_figure("No statement available", title=head)

    periods = [str(p) for p in statement.periods]
    fig = go.Figure()
    drawn = 0
    for slot, item in enumerate(items):
        key = item.value if isinstance(item, LineItem) else str(item)
        if key not in statement.data.index:
            continue
        row = statement.data.loc[key]
        if isinstance(row, pd.DataFrame):
            continue
        values = [None if bool(pd.isna(v)) else float(v) for v in row.tolist()]
        if all(v is None for v in values):
            continue
        fig.add_trace(
            go.Bar(
                x=periods,
                y=values,
                name=_pretty_line_item(key),
                marker={"color": theme.series_color(slot)},
                customdata=[_format_amount(v) for v in values],
                hovertemplate=(
                    f"{_pretty_line_item(key)}: %{{customdata}} "
                    f"{statement.currency.value}<extra></extra>"
                ),
            )
        )
        drawn += 1

    if drawn == 0:
        return theme.empty_figure("None of the line items present", title=head)

    theme.apply_theme(fig, title=head)
    theme.style_axes(fig, x_title="Period end", y_title=f"Amount ({statement.currency.value})")
    fig.update_layout(barmode="group", bargap=0.25, height=420, hovermode="closest")
    return fig


def ratio_panel(
    ratio_set: RatioSet,
    categories: tuple[RatioCategory, ...] | None = None,
    *,
    title: str | None = None,
) -> go.Figure:
    """Kennzahlen als horizontale Balken, gruppiert nach Kategorie.

    Je Kategorie entsteht ein eigener Subplot. Nicht berechenbare
    Kennzahlen erscheinen mit dem Hinweis "n/a" an der Nulllinie statt als
    Balken der Laenge 0 - das haelt die Projektregel ein, dass ein fehlender
    Wert nie wie eine gemessene Null aussieht.

    Args:
        ratio_set: Fertig berechnete Kennzahlen aus ``analytics/ratios``.
        categories: Darzustellende Kategorien. None = alle mit Inhalt.
        title: Ueberschrift.

    Returns:
        Figure mit einem Subplot je Kategorie. Bei leerem Kennzahlensatz
        eine Hinweis-Figure.
    """
    head = title if title is not None else f"{ratio_set.ticker} - ratios"
    wanted = categories if categories is not None else tuple(RatioCategory)
    present = [c for c in wanted if ratio_set.by_category(c)]
    if not present:
        return theme.empty_figure("No ratios available", title=head)

    fig = make_subplots(
        rows=len(present),
        cols=1,
        subplot_titles=[CATEGORY_LABELS[c] for c in present],
        vertical_spacing=0.10 / max(1, len(present) - 1) if len(present) > 1 else 0.1,
    )

    total_rows = 0
    for row, category in enumerate(present, start=1):
        ratios = ratio_set.by_category(category)
        labels = [r.label for r in ratios]
        values = [r.value for r in ratios]
        texts = [
            theme.format_value(r.value, value_format=",.2f")
            if r.unit != "percent"
            else theme.format_value(
                None if r.value is None else r.value * 100.0, value_format=",.2f", unit="%"
            )
            for r in ratios
        ]
        colors = [theme.series_color(row - 1) if v is not None else theme.GRID for v in values]
        fig.add_trace(
            go.Bar(
                x=[0.0 if v is None else v for v in values],
                y=labels,
                orientation="h",
                marker={"color": colors},
                text=texts,
                textposition="outside",
                textfont={"color": theme.INK, "family": theme.FONT_FAMILY},
                customdata=texts,
                hovertemplate="%{y}: %{customdata}<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        total_rows += len(ratios)

    theme.apply_theme(fig, title=head)
    fig.update_layout(
        height=max(320, 90 + 34 * total_rows + 40 * len(present)), hovermode="closest"
    )
    fig.update_yaxes(autorange="reversed")
    for annotation in fig.layout.annotations:
        annotation.font.color = theme.AMBER
        annotation.font.family = theme.FONT_FAMILY
    return fig


def peer_comparison(
    peers: PeerComparison,
    ratio_keys: tuple[str, ...] | None = None,
    *,
    as_heatmap: bool = False,
    title: str | None = None,
) -> go.Figure:
    """Peer-Vergleich als gruppierte Balken oder Heatmap.

    Das Subjekt wird in beiden Varianten hervorgehoben: als Balken traegt
    es die Primaerfarbe des Terminals und eine Umrandung, waehrend die Peers
    zurueckgenommen sind; in der Heatmap markiert ein Rahmen seine Zeile und
    sein Label wird mit einem Pfeil ausgezeichnet.

    Args:
        peers: Vergleichsmatrix, Zeilen sind Ticker, Spalten Kennzahl-Keys.
        ratio_keys: Darzustellende Kennzahlen. None = alle Spalten.
        as_heatmap: Wenn True, Heatmap statt Balken. Sinnvoll ab etwa acht
            Kennzahlen, wo Balkengruppen unuebersichtlich werden. Die
            Heatmap zeigt Perzentil-Raenge innerhalb der Gruppe, damit
            unterschiedliche Einheiten vergleichbar bleiben.
        title: Ueberschrift.

    Returns:
        Figure des Vergleichs. Bei leerer Matrix eine Hinweis-Figure.
    """
    head = title if title is not None else f"Peer comparison: {peers.subject}"
    if peers.data.empty:
        return theme.empty_figure("No peer data", title=head)

    keys = list(ratio_keys) if ratio_keys is not None else [str(c) for c in peers.data.columns]
    keys = [k for k in keys if k in peers.data.columns]
    if not keys:
        return theme.empty_figure("None of the ratios present", title=head)

    frame = peers.data[keys]
    tickers = [str(t) for t in frame.index]

    if as_heatmap:
        # Rein darstellungsbezogene Normierung: Rang je Spalte, damit
        # Kennzahlen unterschiedlicher Einheit eine gemeinsame Skala haben.
        ranked = frame.rank(axis=0, pct=True, na_option="keep")
        text = [
            [theme.format_value(v, value_format=",.2f") for v in row] for row in frame.to_numpy()
        ]
        fig = go.Figure(
            go.Heatmap(
                z=ranked.to_numpy(),
                x=keys,
                y=tickers,
                colorscale=theme.sequential_scale(),
                zmin=0.0,
                zmax=1.0,
                text=text,
                texttemplate="%{text}",
                textfont={"family": theme.FONT_FAMILY, "size": theme.FONT_SIZE - 1},
                hovertemplate=(
                    "%{y} - %{x}<br>Value: %{text}<br>Percentile: %{z:.0%}<extra></extra>"
                ),
                colorbar={
                    "title": {"text": "Percentile", "font": {"color": theme.INK_MUTED}},
                    "tickformat": ".0%",
                    "outlinecolor": theme.GRID,
                    "tickfont": {"color": theme.INK_MUTED},
                },
                xgap=2,
                ygap=2,
            )
        )
        if peers.subject in tickers:
            pos = tickers.index(peers.subject)
            fig.add_shape(
                type="rect",
                x0=-0.5,
                x1=len(keys) - 0.5,
                y0=pos - 0.5,
                y1=pos + 0.5,
                line={"color": theme.AMBER, "width": 2},
                fillcolor="rgba(0,0,0,0)",
            )
        ticktext = [f"<b>{t} &lt;&lt;</b>" if t == peers.subject else t for t in tickers]
        fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(len(tickers))),
            ticktext=ticktext,
        )
        theme.apply_theme(fig, title=f"{head} - colour = percentile within the group")
        theme.style_axes(fig, x_title="Ratio", y_title="Ticker")
        fig.update_layout(height=max(320, 120 + 30 * len(tickers)), hovermode="closest")
        return fig

    fig = go.Figure()
    for slot, key in enumerate(keys):
        column = frame[key]
        values = [None if bool(pd.isna(v)) else float(v) for v in column.tolist()]
        colors = [theme.AMBER if t == peers.subject else theme.series_color(slot) for t in tickers]
        widths = [2 if t == peers.subject else 0 for t in tickers]
        fig.add_trace(
            go.Bar(
                x=tickers,
                y=values,
                name=key,
                marker={
                    "color": colors,
                    "line": {"color": theme.INK, "width": widths},
                },
                customdata=[theme.format_value(v, value_format=",.2f") for v in values],
                hovertemplate=f"%{{x}} - {key}: %{{customdata}}<extra></extra>",
            )
        )

    theme.apply_theme(fig, title=f"{head} - subject highlighted in amber")
    theme.style_axes(fig, x_title="Ticker", y_title="Value")
    fig.update_layout(barmode="group", bargap=0.25, height=440, hovermode="closest")
    return fig
