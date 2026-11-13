"""Wiederverwendbare Bausteine der Seiten.

Reine Darstellung: Kacheln, Tabellen, Banner und der Datenherkunfts-Hinweis.
Diese Funktionen rechnen nicht, sie formatieren fertige Modellwerte
(Architekturregel 4). Die CSS-Klassen stammen aus
``ui/assets/btlite.css``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Final

from dash import dcc, html

from btlite.ui import formatting as fmt

if TYPE_CHECKING:
    import pandas as pd
    from plotly.graph_objects import Figure

    from btlite.core.errors import BTLiteError
    from btlite.models.common import Provenance

#: Hoehe der grossen Charts in Pixeln.
CHART_HEIGHT = 520

#: Hoehe der Nebencharts in Pixeln.
CHART_HEIGHT_SMALL = 340

#: Typ-Schluessel der Sortier-Ueberschriften. Die Tabelle weiss nicht, auf
#: welcher Seite sie steht; ``table`` trennt mehrere Tabellen einer Seite.
SORT_HEADER_TYPE: Final[str] = "bt-sort-header"

#: Zeichen der Sortierrichtung. Neben der Farbe traegt damit auch die Form
#: die Aussage - Farbe allein genuegt nie.
SORT_ARROWS: Final[dict[str, str]] = {"asc": "▴", "desc": "▾"}


def section(title: str, *children: Any, className: str = "") -> html.Div:  # noqa: N803 - Dash-Prop
    """Rahmt einen Seitenabschnitt mit Ueberschrift.

    Args:
        title: Ueberschrift des Abschnitts.
        *children: Inhalt des Abschnitts.
        className: Zusaetzliche CSS-Klassen des Rahmens.

    Returns:
        Abschnitts-Container.
    """
    classes = f"bt-section {className}".strip()
    return html.Div(
        [html.Div(title, className="card-header"), html.Div(list(children), className="bt-body")],
        className=classes,
    )


def tile(label: str, value: str, *, value_class: str = "") -> html.Div:
    """Baut eine Kennzahlen-Kachel.

    Args:
        label: Beschriftung der Kachel.
        value: Bereits formatierter Wert. Fehlende Werte als "n/a".
        value_class: Zusatzklasse des Wertes, etwa ``"bt-up"``.

    Returns:
        Kachel-Element.
    """
    classes = f"bt-tile-value {value_class}".strip()
    return html.Div(
        [html.Span(label, className="bt-tile-label"), html.Span(value, className=classes)],
        className="bt-tile",
    )


def tiles(items: Sequence[tuple[str, ...]]) -> html.Div:
    """Baut ein Kachel-Raster.

    Args:
        items: Paare ``(label, wert)`` oder Tripel
            ``(label, wert, wertklasse)``. Beide Formen sind in derselben
            Liste mischbar - die meisten Kacheln brauchen keine
            Richtungsklasse, einige schon.

    Returns:
        Raster-Container.

    Raises:
        ValueError: Wenn ein Eintrag weder zwei noch drei Felder hat.
    """
    built: list[html.Div] = []
    for item in items:
        if len(item) not in (2, 3):
            msg = f"Kachel braucht 2 oder 3 Felder, erhalten {len(item)}"
            raise ValueError(msg)
        label, value = item[0], item[1]
        css = item[2] if len(item) == 3 else ""
        built.append(tile(label, value, value_class=css))
    return html.Div(built, className="bt-tiles")


def error_banner(error: BTLiteError | str) -> html.Div:
    """Baut das Fehlerbanner aus einer Nutzermeldung.

    Der technische ``detail`` erscheint bewusst **nicht** - er gehoert ins
    Log, nicht auf den Bildschirm.

    Args:
        error: Gefangener Fehler oder fertige Meldung.

    Returns:
        Banner-Element.
    """
    message = error if isinstance(error, str) else error.user_message
    return html.Div(
        [html.Strong("Error: "), html.Span(message)],
        className="bt-error",
        role="alert",
    )


def warning_banner(messages: Iterable[str]) -> html.Div | None:
    """Baut ein Hinweisbanner fuer nicht-fatale Warnungen.

    Args:
        messages: Warntexte der Rechenschicht.

    Returns:
        Banner-Element, oder None wenn keine Warnung vorliegt.
    """
    items = [m for m in messages if m]
    if not items:
        return None
    return html.Div(
        [html.Strong("Note: "), html.Span(" ".join(items))],
        className="bt-warning",
        role="status",
    )


def provenance_note(provenance: Provenance | None, timezone: str) -> html.Div:
    """Baut den sichtbaren Datenherkunfts-Hinweis einer Seite.

    Pflicht auf jeder Seite: Quelle, Zeitstempel und Verzug. Nichts wird
    als Echtzeit ausgegeben, was keine ist.

    Args:
        provenance: Herkunftsnachweis des angezeigten Datensatzes.
        timezone: IANA-Name der Anzeigezeitzone.

    Returns:
        Hinweiszeile.
    """
    return html.Div(
        fmt.provenance_text(provenance, timezone),
        className="bt-provenance bt-notice",
        role="status",
    )


def page_header(title: str, subtitle: str = "") -> html.Div:
    """Baut die Kopfzeile einer Seite.

    Args:
        title: Titel, etwa ``"AAPL - Financial Analysis"``.
        subtitle: Ergaenzender Text, etwa der Unternehmensname.

    Returns:
        Kopfzeilen-Element.
    """
    children: list[Any] = [html.H2(title, className="bt-page-title")]
    if subtitle:
        children.append(html.Div(subtitle, className="bt-page-subtitle bt-notice"))
    return html.Div(children, className="bt-page-header")


def graph(
    figure: Figure,
    *,
    element_id: str | None = None,
    height: int = CHART_HEIGHT,
) -> dcc.Graph:
    """Bettet eine fertige Figure ein.

    ``viz/`` liefert die Figure, dieses Modul setzt nur die Rahmung.

    Args:
        figure: Fertige Plotly-Figure aus ``viz/``.
        element_id: Optionale Element-ID fuer Callbacks.
        height: Hoehe in Pixeln.

    Returns:
        Graph-Komponente.
    """
    kwargs: dict[str, Any] = {
        "figure": figure,
        "config": {"displaylogo": False, "responsive": True},
        "style": {"height": f"{height}px"},
    }
    if element_id is not None:
        kwargs["id"] = element_id
    return dcc.Graph(**kwargs)


def loading(
    *children: Any, element_id: str | None = None, delay_show_ms: int | None = None
) -> dcc.Loading:
    """Legt eine Ladeanzeige ueber teure Inhalte.

    Args:
        *children: Zu umhuellende Komponenten.
        element_id: Optionale Element-ID.
        delay_show_ms: Wartezeit in Millisekunden, bevor der Spinner
            erscheint. ``dcc.Loading`` blendet waehrend des Ladens die
            Kinder aus (``visibility: hidden``, am Dash-4.4.1-Bundle
            geprueft). Ein Bereich, den ein schneller Callback jede Sekunde
            neu schreibt, flackerte ohne Verzoegerung im Sekundentakt.

    Returns:
        Loading-Komponente im Terminal-Stil.
    """
    kwargs: dict[str, Any] = {"children": list(children), "type": "dot", "color": "#ffa028"}
    if element_id is not None:
        kwargs["id"] = element_id
    if delay_show_ms is not None:
        kwargs["delay_show"] = delay_show_ms
    return dcc.Loading(**kwargs)


def data_table(
    frame: pd.DataFrame,
    *,
    index_label: str = "",
    formatters: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    max_rows: int | None = 200,
    highlight_rows: Sequence[str] = (),
    sort_table: str | None = None,
    sort_by: str | None = None,
    sort_desc: bool = False,
    column_hints: dict[str, str] | None = None,
    index_cell: Callable[[str], Any] | None = None,
) -> html.Table:
    """Rendert einen DataFrame als dichte Terminal-Tabelle.

    Der Frame wird nicht umgeformt, nur gelesen: jede Zelle laeuft durch
    ihren Formatter, fehlende Werte erscheinen als "n/a".

    Args:
        frame: Anzuzeigende Daten. Index wird als erste Spalte gesetzt.
        index_label: Ueberschrift der Indexspalte.
        formatters: Abbildung Spaltenname auf Formatierungsfunktion. Nicht
            genannte Spalten laufen durch
            :func:`~btlite.ui.formatting.format_number` beziehungsweise
            werden als Text uebernommen.
        headers: Abbildung Spaltenname auf eine ausgeschriebene
            Ueberschrift. Nicht genannte Spalten behalten ihren Rohnamen -
            eine neue Spalte soll sichtbar sein und nicht verschwinden.
            Der Rohname bleibt als Tooltip erreichbar, damit die Zuordnung
            zur CSV-Ausgabe nachvollziehbar bleibt.
        max_rows: Obergrenze der Zeilen. None zeigt alle.
        highlight_rows: Indexwerte, die hervorgehoben werden.
        sort_table: Kennung dieser Tabelle. Ist sie gesetzt, werden die
            Ueberschriften anklickbar und tragen Wildcard-IDs. ``None``
            laesst die Tabelle unveraendert - bestehende Aufrufe bleiben
            wie sie sind.
        sort_by: Aktuell sortierte Spalte, oder der Wert von
            ``index_label`` fuer die Indexspalte.
        sort_desc: Richtung der aktuellen Sortierung.
        column_hints: Abbildung Spaltenname auf einen Hinweistext, der als
            Tooltip an der Ueberschrift haengt. Ohne Eintrag bleibt der
            Rohname der Tooltip.
        index_cell: Baut den Inhalt der Indexzelle aus dem Indexwert als
            Text, etwa Ticker plus Links auf die Auswertungen. ``None``
            laesst die Zelle reiner Text - bestehende Aufrufe bleiben wie
            sie sind. Sortierung und Hervorhebung laufen weiter ueber den
            Rohwert, nicht ueber das Gerenderte.

    Returns:
        Tabellen-Element.
    """
    if frame.empty:
        return html.Table(
            [html.Tbody([html.Tr([html.Td("No data", className="bt-na")])])],
            className="bt-table",
        )

    shown = frame if max_rows is None else frame.head(max_rows)
    columns = [str(c) for c in shown.columns]
    specs = formatters or {}
    highlight = {str(h) for h in highlight_rows}

    titles = headers or {}
    hints = column_hints or {}

    def _header(column: str, label: str) -> html.Th:
        """Baut eine Ueberschrift, je nach Modus klickbar.

        Args:
            column: Spaltenname, wie er in ``sort_by`` erscheint.
            label: Sichtbare Beschriftung.

        Returns:
            Kopfzelle mit oder ohne Sortier-Verdrahtung.
        """
        tooltip = hints.get(column, column)
        if sort_table is None:
            return html.Th(label, title=tooltip)

        active = sort_by == column
        arrow = SORT_ARROWS["desc" if sort_desc else "asc"] if active else ""
        return html.Th(
            html.Button(
                [label, html.Span(arrow, className="bt-sort-arrow")],
                id={"type": SORT_HEADER_TYPE, "table": sort_table, "column": column},
                n_clicks=0,
                className="bt-sort-header" + (" bt-sort-active" if active else ""),
                title=f"{tooltip} - click to sort",
            ),
            className="bt-th-sortable",
        )

    head = html.Thead(
        html.Tr(
            [
                _header(index_label, index_label),
                *[_header(c, titles.get(c, c)) for c in columns],
            ]
        ),
    )

    rows: list[html.Tr] = []
    for index_value, record in zip(shown.index, shown.to_dict(orient="records"), strict=False):
        label = str(index_value)
        cells: list[html.Td] = [html.Td(label if index_cell is None else index_cell(label))]
        for column in columns:
            raw = record.get(column)
            formatter = specs.get(column)
            text = formatter(raw) if formatter is not None else _default_cell(raw)
            cells.append(html.Td(text, className="bt-na" if text == fmt.NA_TEXT else ""))
        rows.append(html.Tr(cells, className="bt-subject" if str(index_value) in highlight else ""))

    return html.Table([head, html.Tbody(rows)], className="bt-table")


def sort_frame(
    frame: pd.DataFrame,
    *,
    by: str | None,
    descending: bool = False,
    index_label: str = "",
) -> pd.DataFrame:
    """Sortiert einen Anzeige-Frame nach einer Spalte.

    Sortiert wird ueber die **Rohwerte** des Frames. Das ist der Grund,
    warum die Seiten ihre Frames unformatiert uebergeben und die
    Formatierung den Formattern der Tabelle ueberlassen: nach der
    Zeichenkette "1,234.50 EUR" zu sortieren ergaebe eine alphabetische
    Reihenfolge, in der 9 nach 10 kaeme.

    Fehlende Werte landen **immer am Ende**, in beiden Richtungen. Ein
    ``None`` ist keine Aussage ueber die Groesse; es nach oben zu sortieren
    liesse es wie den kleinsten Wert aussehen.

    Args:
        frame: Anzuzeigende Daten, mit Rohwerten.
        by: Spaltenname, oder ``index_label`` fuer die Indexspalte.
            ``None`` oder ein unbekannter Name laesst die Reihenfolge
            unveraendert - eine verschwundene Spalte darf die Seite nicht
            kippen.
        descending: Absteigend statt aufsteigend.
        index_label: Name, unter dem die Indexspalte adressiert wird.

    Returns:
        Neuer Frame in der gewuenschten Reihenfolge. Der Eingabe-Frame
        bleibt unveraendert.
    """
    if by is None or frame.empty:
        return frame
    if index_label and by == index_label:
        return frame.sort_index(ascending=not descending)
    if by not in frame.columns:
        return frame
    return frame.sort_values(
        by=by,
        ascending=not descending,
        na_position="last",
        kind="stable",
    )


def next_sort_state(
    current_column: str | None,
    current_desc: bool,
    clicked: str,
) -> tuple[str, bool]:
    """Bestimmt die neue Sortierung nach einem Klick auf eine Ueberschrift.

    Ein Klick auf eine **andere** Spalte sortiert sie aufsteigend. Ein
    erneuter Klick auf dieselbe Spalte dreht die Richtung um. Das ist die
    Konvention, die Tabellen ueberall sonst auch nutzen.

    Args:
        current_column: Bisher sortierte Spalte, oder ``None``.
        current_desc: Bisherige Richtung.
        clicked: Angeklickte Spalte.

    Returns:
        Paar aus neuer Spalte und neuer Richtung.
    """
    if clicked == current_column:
        return clicked, not current_desc
    return clicked, False


def _default_cell(raw: object) -> str:
    """Formatiert einen Zellwert ohne eigenen Formatter.

    Args:
        raw: Rohwert aus dem DataFrame.

    Returns:
        Anzeigetext. Fehlende Werte als "n/a".
    """
    if isinstance(raw, bool):
        return "yes" if raw else "no"
    if isinstance(raw, int | float):
        return fmt.format_number(float(raw))
    if raw is None:
        return fmt.NA_TEXT
    text = str(raw).strip()
    return text if text and text.lower() != "nan" else fmt.NA_TEXT


def controls(*children: Any) -> html.Div:
    """Reiht Bedienelemente in einer Zeile.

    Args:
        *children: Steuerelemente.

    Returns:
        Zeilen-Container.
    """
    return html.Div(list(children), className="bt-controls")


def labeled(label: str, control: Any) -> html.Div:
    """Beschriftet ein einzelnes Bedienelement.

    Args:
        label: Beschriftung.
        control: Das Steuerelement.

    Returns:
        Beschriftetes Element.
    """
    return html.Div(
        [html.Label(label, className="bt-control-label"), control],
        className="bt-control",
    )


def placeholder(message: str) -> html.Div:
    """Zeigt einen neutralen Hinweis statt eines leeren Bereichs.

    Args:
        message: Hinweistext.

    Returns:
        Hinweis-Element.
    """
    return html.Div(message, className="bt-placeholder bt-na")
