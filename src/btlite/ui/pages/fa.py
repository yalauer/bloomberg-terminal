"""Seite FA - Statements, Kennzahlen-Panel und Peer-Vergleich.

Die Seite verdrahtet ``analytics.statements`` und ``analytics.ratios`` mit
``viz.fundamentals``. Normalisierung, TTM-Aggregation und Kennzahlen kommen
fertig aus der Rechenschicht.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, cast

from dash import dcc, html

from btlite.core.errors import BTLiteError, SymbolNotFoundError
from btlite.models.common import Period
from btlite.models.fundamentals import RatioCategory, StatementType
from btlite.ui import formatting as fmt
from btlite.ui.layouts import components as c

if TYPE_CHECKING:
    from btlite.analytics.statements import StatementTrend
    from btlite.models.fundamentals import FinancialStatement, PeerComparison, RatioSet
    from btlite.ui.state import AppState

#: Route dieser Seite.
PATH = "/fa"

#: Element-IDs der Bedienelemente.
TICKER_STORE = "fa-tickers"
PERIOD_INPUT = "fa-period"
STATEMENT_OUTPUT = "fa-statements"

#: Waehlbare Periodenbasen.
PERIOD_CHOICES: Final[tuple[tuple[str, str], ...]] = (
    ("Annual", Period.ANNUAL.value),
    ("Quarterly", Period.QUARTERLY.value),
    ("TTM", Period.TTM.value),
)

#: Anzeigenamen der Statement-Arten.
_STATEMENT_LABELS: Final[dict[StatementType, str]] = {
    StatementType.INCOME: "Income statement",
    StatementType.BALANCE: "Balance sheet",
    StatementType.CASHFLOW: "Cash flow statement",
}

#: Anzeigenamen der Kennzahlen-Kategorien. Der Rohwert des Enums
#: ("liquidity") gehoert nicht in die Oberflaeche.
_CATEGORY_LABELS: Final[dict[RatioCategory, str]] = {
    RatioCategory.LIQUIDITY: "Liquidity",
    RatioCategory.LEVERAGE: "Leverage",
    RatioCategory.PROFITABILITY: "Profitability",
    RatioCategory.EFFICIENCY: "Efficiency",
    RatioCategory.VALUATION: "Valuation",
    RatioCategory.GROWTH: "Growth",
}

#: Kennzahlen des Peer-Vergleichs.
_PEER_KEYS: Final[tuple[str, ...]] = (
    "pe",
    "pb",
    "ps",
    "ev_to_ebitda",
    "roe",
    "roa",
    "net_margin",
    "debt_to_equity",
)


def layout(state: AppState, tickers: list[str], args: tuple[str, ...] = ()) -> html.Div:
    """Baut die Seite FA.

    Args:
        state: UI-Zustand mit Provider und Puffern.
        tickers: Symbole aus der Kommandozeile. Das erste ist das Subjekt,
            die uebrigen sind die Peer-Gruppe.
        args: Weitere Argumente. ``annual``, ``quarterly`` oder ``ttm``
            setzen die Periodenbasis.

    Returns:
        Seiteninhalt mit Umschalter, Statements, Kennzahlen und Peers.
    """
    symbols = tickers or state.default_tickers
    subject = symbols[0]
    peers = symbols[1:]
    period = _period_from_args(args)

    return html.Div(
        [
            c.page_header(
                f"{subject} - Financial analysis",
                f"Peer group: {', '.join(peers)}" if peers else "No peer group",
            ),
            dcc.Store(id=TICKER_STORE, data=symbols),
            c.controls(
                c.labeled(
                    "Period basis",
                    dcc.RadioItems(
                        id=PERIOD_INPUT,
                        options=[{"label": lab, "value": val} for lab, val in PERIOD_CHOICES],
                        value=period.value,
                        inline=True,
                        className="bt-radio",
                    ),
                ),
            ),
            c.loading(
                html.Div(
                    id=STATEMENT_OUTPUT,
                    children=render_body(state, subject, peers, period),
                )
            ),
        ],
        className="bt-page bt-page-fa",
    )


def _period_from_args(args: tuple[str, ...]) -> Period:
    """Liest die Periodenbasis aus den Kommandozeilen-Argumenten.

    Args:
        args: Argumente hinter dem Mnemonic.

    Returns:
        Gewaehlte Basis, Default ``Period.ANNUAL``.
    """
    for arg in args:
        candidate = arg.strip().lower()
        for period in Period:
            if candidate == period.value:
                return period
    return Period.ANNUAL


def render_body(
    state: AppState,
    subject: str,
    peers: list[str],
    period: Period,
) -> html.Div:
    """Baut Statements, Kennzahlen-Panel und Peer-Vergleich.

    Args:
        state: UI-Zustand.
        subject: Betrachtetes Symbol.
        peers: Vergleichsgruppe.
        period: Periodenbasis.

    Returns:
        Container mit allen Abschnitten. Fehler erscheinen als Banner.
    """
    from btlite.analytics.statements import compute_trend_bundle
    from btlite.viz import fundamentals as viz_fa

    try:
        bundle = state.statements(subject, period=period)
        ratios = state.ratios(subject, period=period)
    except SymbolNotFoundError as exc:
        # Ein bekanntes Instrument ohne Statements ist kein Fehler: Fonds
        # und Indizes fuehren keine eigene Bilanz. Ein voellig unbekanntes
        # Symbol dagegen schon - sonst verschwindet ein Tippfehler lautlos.
        if not _instrument_exists(state, subject):
            return html.Div([c.error_banner(exc)])
        return html.Div(
            [
                c.placeholder(
                    f"No financial statements are available for {subject}. "
                    "Funds and indices do not file their own balance sheet."
                )
            ]
        )
    except BTLiteError as exc:
        return html.Div([c.error_banner(exc)])

    blocks: list[Any] = [c.provenance_note(bundle.provenance, state.timezone)]

    # Die Trendrechnung liegt vollstaendig in ``analytics.statements``.
    # Hier wird sie nur abgerufen und angezeigt (Architekturregel 4).
    trends = compute_trend_bundle(bundle)

    for kind, statement in (
        (StatementType.INCOME, bundle.income),
        (StatementType.BALANCE, bundle.balance),
        (StatementType.CASHFLOW, bundle.cashflow),
    ):
        label = _STATEMENT_LABELS[kind]
        if statement is None:
            blocks.append(c.section(label, c.placeholder("Not available.")))
            continue
        blocks.append(c.section(label, _statement_table(statement, trends.get(kind), viz_fa)))

    blocks.append(
        c.section(
            f"Ratios ({period.value})",
            c.graph(
                viz_fa.ratio_panel(ratios, title=f"{subject} - Ratios"),
                height=c.CHART_HEIGHT,
            ),
            _ratio_summary(ratios),
        )
    )

    blocks.append(_peer_section(state, ratios, peers, period, viz_fa))
    return html.Div(blocks, className="bt-fa-body")


def _instrument_exists(state: AppState, ticker: str) -> bool:
    """Prueft, ob die Quelle das Symbol ueberhaupt kennt.

    Args:
        state: UI-Zustand.
        ticker: Kanonisches Symbol.

    Returns:
        True wenn Stammdaten vorliegen.
    """
    try:
        state.instrument(ticker)
    except BTLiteError:
        return False
    return True


def _statement_table(
    statement: FinancialStatement,
    trend: StatementTrend | None,
    viz_fa: Any,
) -> html.Div:
    """Rendert ein Statement als Tabelle, mit Veraenderung zur Vorperiode.

    Die Datensaetze kommen fertig formatiert aus
    ``viz.fundamentals.statement_table_records`` - dort stehen fehlende
    Werte bereits als "n/a". Die Veraenderung je Zelle kommt aus
    ``analytics.statements.compute_trend_bundle``; sie ist dezimal und wird
    hier nur in Text uebersetzt.

    Args:
        statement: Normalisiertes Statement.
        trend: Trendtabelle desselben Statement, oder None wenn keine
            vorliegt. Zeilen- und Spaltenfolge sind deckungsgleich mit
            ``statement.data``.
        viz_fa: Das Modul ``viz.fundamentals``.

    Returns:
        Tabellen-Container, oder ein Hinweis bei leerem Payload.
    """
    records = viz_fa.statement_table_records(statement)
    if not records:
        return c.placeholder("No statement payload.")

    columns = [k for k in records[0] if k != "position"]
    line_items = [str(name) for name in statement.data.index]
    head = html.Thead(html.Tr([html.Th("Line item"), *[html.Th(col) for col in columns]]))
    body = html.Tbody(
        [
            html.Tr(
                [
                    html.Td(record["position"]),
                    *[
                        _amount_cell(
                            record.get(col, fmt.NA_TEXT),
                            _trend_cell(trend, line_items, row_index, col_index),
                            first_period=col_index == 0,
                        )
                        for col_index, col in enumerate(columns)
                    ],
                ]
            )
            for row_index, record in enumerate(records)
        ]
    )
    return html.Div(
        [
            html.Div(
                f"Absolute amounts in {statement.currency.value}. "
                "Below each amount, the change against the previous period in percent - "
                "▲ and ▼ state the direction of the calculation only, not a judgement. "
                "For expense line items, ▲ means higher expenses.",
                className="bt-notice",
            ),
            html.Table([head, body], className="bt-table"),
        ],
        className="bt-statement",
    )


def _trend_cell(
    trend: StatementTrend | None,
    line_items: list[str],
    row_index: int,
    col_index: int,
) -> tuple[float | None, str | None]:
    """Liest einen Trendwert samt Begruendung aus der Trendtabelle.

    Args:
        trend: Trendtabelle, oder None.
        line_items: Positionsbezeichner des Statement, in Zeilenfolge.
        row_index: Zeilennummer in ``line_items``.
        col_index: Spaltennummer, also Periodenindex.

    Returns:
        Tupel aus relativer Veraenderung (dezimal) und Begruendung. Genau
        eines der beiden ist gefuellt; bei fehlender Trendtabelle beide
        ``None``.
    """
    if trend is None or row_index >= len(line_items):
        return None, None
    point = trend.point(line_items[row_index], period_index=col_index)
    if point is None:
        return None, None
    return point.change_pct, point.reason


def _amount_cell(
    amount: str,
    change: tuple[float | None, str | None],
    *,
    first_period: bool = False,
) -> html.Td:
    """Baut eine Betragszelle mit darunterliegender Veraenderung.

    Farbe traegt die Bedeutung nie allein: der Prozentwert fuehrt immer ein
    Vorzeichen und ein Richtungszeichen mit sich, und beide beschreiben nur
    die arithmetische Richtung. Das Produkt bewertet nicht, ob ein Plus bei
    einer Aufwandsposition guenstig ist.

    Args:
        amount: Fertig formatierter Betrag, "n/a" wenn nicht vorhanden.
        change: Tupel aus dezimaler Veraenderung und Begruendung.
        first_period: Ob die Zelle in der aeltesten Spalte steht. Dort gibt
            es per Konstruktion keine Vorperiode; die Begruendung waere in
            jeder Zeile dieselbe und bliebe als Rauschen stehen.

    Returns:
        Tabellenzelle mit Betrag und Veraenderungszeile.
    """
    value, reason = change
    if value is None:
        # Kein Prozentsatz interpretierbar. Der Grund steht dabei, damit
        # "n/a" nicht raetselhaft bleibt - nie eine irrefuehrende 0.
        show = reason is not None and not first_period
        change_node = html.Div(
            f"{fmt.NA_TEXT} ({reason})" if show else "",
            className="bt-na" if show else "",
        )
    else:
        change_node = html.Div(
            fmt.format_change(value),
            className=fmt.direction_class(value),
        )
    return html.Td(
        [
            html.Div(amount, className="bt-na" if amount == fmt.NA_TEXT else ""),
            change_node,
        ]
    )


def _ratio_summary(ratios: RatioSet) -> html.Div:
    """Listet die Kennzahlen je Kategorie als Tabelle.

    Args:
        ratios: Kennzahlensatz des Subjekts.

    Returns:
        Container mit je einer Tabelle pro Kategorie.
    """
    blocks: list[Any] = []
    for category in RatioCategory:
        entries = ratios.by_category(category)
        if not entries:
            continue
        rows = [
            html.Tr(
                [
                    html.Td(r.label),
                    html.Td(
                        fmt.format_ratio_value(r.value, r.unit),
                        className=fmt.value_class(r.value),
                    ),
                    html.Td(r.formula or "", className="bt-notice"),
                ]
            )
            for r in entries
        ]
        blocks.append(
            html.Details(
                [
                    html.Summary(
                        _CATEGORY_LABELS.get(category, category.value), className="bt-subhead"
                    ),
                    html.Table(
                        [
                            html.Thead(
                                html.Tr([html.Th("Ratio"), html.Th("Value"), html.Th("Formula")])
                            ),
                            html.Tbody(rows),
                        ],
                        className="bt-table",
                    ),
                ],
                className="bt-ratio-group",
            )
        )
    if not blocks:
        return c.placeholder("No ratios can be computed.")
    return html.Div(blocks, className="bt-ratio-groups")


def _peer_section(
    state: AppState,
    subject_ratios: RatioSet,
    peers: list[str],
    period: Period,
    viz_fa: Any,
) -> html.Div:
    """Baut den Peer-Vergleich.

    Args:
        state: UI-Zustand.
        subject_ratios: Kennzahlen des Subjekts.
        peers: Symbole der Vergleichsgruppe.
        period: Periodenbasis.
        viz_fa: Das Modul ``viz.fundamentals``.

    Returns:
        Abschnitt mit Chart und Tabelle, oder ein Hinweis.
    """
    from btlite.analytics.ratios import compare_peers

    if not peers:
        return c.section(
            "Peer comparison",
            c.placeholder(
                "No peer group given. Enter several tickers, for example AAPL,MSFT,NVDA FA GO."
            ),
        )

    peer_sets: list[RatioSet] = []
    skipped: list[str] = []
    for symbol in peers:
        try:
            peer_sets.append(state.ratios(symbol, period=period))
        except BTLiteError:
            skipped.append(symbol)

    if not peer_sets:
        return c.section(
            "Peer comparison",
            c.error_banner(f"No ratios are available for any peer: {', '.join(peers)}."),
        )

    # ``compare_peers`` ist als ``tuple[object, list[str]]`` annotiert und
    # verliert damit den Typ. Laut Docstring ist das erste Element eine
    # ``PeerComparison``; hier wird der Typ nur wieder benannt, nicht
    # umgedeutet.
    raw_comparison, warnings = compare_peers(subject_ratios, peer_sets, ratio_keys=list(_PEER_KEYS))
    comparison = cast("PeerComparison", raw_comparison)
    children: list[Any] = []
    if skipped:
        children.append(c.warning_banner([f"Skipped for lack of data: {', '.join(skipped)}."]))
    banner = c.warning_banner(warnings)
    if banner is not None:
        children.append(banner)

    children.append(
        c.graph(
            viz_fa.peer_comparison(comparison, tuple(_PEER_KEYS)),
            height=c.CHART_HEIGHT,
        )
    )
    children.append(
        c.data_table(
            comparison.data,
            index_label="Ticker",
            highlight_rows=[comparison.subject],
        )
    )
    return c.section("Peer comparison", *children)
