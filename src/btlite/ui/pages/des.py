"""Seite DES - Stammdaten, aktueller Quote und Kennzahlen-Ueberblick.

Die Seite verdrahtet ``AppState`` (Stammdaten, Quote, Kennzahlen) mit den
Bausteinen aus ``ui/layouts``. Sie rechnet nichts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dash import html

from btlite.core.errors import BTLiteError, SymbolNotFoundError
from btlite.models.common import Period
from btlite.models.fundamentals import RatioCategory
from btlite.ui import formatting as fmt
from btlite.ui.layouts import components as c

if TYPE_CHECKING:
    from btlite.models.fundamentals import RatioSet
    from btlite.models.market import Instrument, Quote
    from btlite.ui.state import AppState

#: Route dieser Seite.
PATH = "/des"

#: Rueckmeldung des Merken-Knopfes. Eigener Container, damit die Meldung
#: unter der Schaltflaeche steht statt die Trefferliste der Suche zu
#: ueberschreiben - die ist zu diesem Zeitpunkt laengst geleert.
WATCH_FEEDBACK: str = "des-watch-feedback"

#: Anzeigenamen der Kennzahlen-Kategorien. Der Rohwert des Enums
#: ("liquidity") gehoert nicht in die Oberflaeche.
_CATEGORY_LABELS: dict[RatioCategory, str] = {
    RatioCategory.LIQUIDITY: "Liquidity",
    RatioCategory.LEVERAGE: "Leverage",
    RatioCategory.PROFITABILITY: "Profitability",
    RatioCategory.EFFICIENCY: "Efficiency",
    RatioCategory.VALUATION: "Valuation",
    RatioCategory.GROWTH: "Growth",
}

#: Kennzahlen des Ueberblicks, in dieser Reihenfolge.
_OVERVIEW_KEYS: tuple[str, ...] = (
    "pe",
    "pb",
    "ps",
    "dividend_yield",
    "roe",
    "net_margin",
    "current_ratio",
    "debt_to_equity",
)


def layout(state: AppState, tickers: list[str], args: tuple[str, ...] = ()) -> html.Div:
    """Baut die Seite DES fuer den ersten genannten Ticker.

    Args:
        state: UI-Zustand mit Provider und Puffern.
        tickers: Symbole aus der Kommandozeile. Genutzt wird das erste.
        args: Weitere Argumente. Ohne Wirkung auf dieser Seite.

    Returns:
        Seiteninhalt. Fehler erscheinen als Banner, nie als leere Seite.
    """
    del args
    symbol = tickers[0] if tickers else state.default_tickers[0]

    try:
        instrument = state.instrument(symbol)
    except BTLiteError as exc:
        return html.Div([c.page_header(f"{symbol} - Profile"), c.error_banner(exc)])

    quote = _safe_quote(state, symbol)
    ratios, ratio_error = _safe_ratios(state, symbol)

    blocks: list[Any] = [
        c.page_header(f"{instrument.ticker} - Profile", instrument.name or ""),
        c.provenance_note(instrument.provenance, state.timezone),
        _watch_action(instrument.ticker),
    ]
    if quote is not None:
        blocks.append(c.section("Current price", _quote_tiles(quote)))
    else:
        blocks.append(c.section("Current price", c.placeholder("No quote available.")))

    blocks.append(c.section("Company data", _master_tiles(instrument)))

    if isinstance(ratio_error, SymbolNotFoundError):
        # Fonds und Indizes fuehren keine Statements. Das ist kein Fehler,
        # sondern die Natur des Instruments - entsprechend zurueckgenommen
        # dargestellt statt als Fehlerbanner.
        blocks.append(
            c.section(
                "Ratios",
                c.placeholder(
                    "No financial statements are available for this instrument. "
                    "Funds and indices do not file their own balance sheet."
                ),
            )
        )
    elif ratio_error is not None:
        blocks.append(c.section("Ratios", c.error_banner(ratio_error)))
    elif ratios is not None:
        blocks.append(c.section("Key ratios", _ratio_tiles(ratios)))
        blocks.append(c.section("Ratios by category", _category_table(ratios)))

    if instrument.description:
        blocks.append(
            c.section("Business description", html.P(instrument.description, className="bt-text"))
        )

    return html.Div(blocks, className="bt-page bt-page-des")


def _safe_quote(state: AppState, symbol: str) -> Quote | None:
    """Holt den Quote und schluckt Datenfehler.

    Args:
        state: UI-Zustand.
        symbol: Kanonisches Symbol.

    Returns:
        Quote, oder None wenn die Quelle keinen liefert.
    """
    try:
        return state.quote(symbol)
    except BTLiteError:
        return None


def _safe_ratios(state: AppState, symbol: str) -> tuple[RatioSet | None, BTLiteError | None]:
    """Holt die Kennzahlen und faengt Fehler ab.

    Args:
        state: UI-Zustand.
        symbol: Kanonisches Symbol.

    Returns:
        Paar aus Kennzahlensatz und Fehler. Genau eines ist gesetzt.
    """
    try:
        return state.ratios(symbol, period=Period.ANNUAL), None
    except BTLiteError as exc:
        return None, exc


def _quote_tiles(quote: Quote) -> html.Div:
    """Baut die Kacheln des aktuellen Kurses.

    Args:
        quote: Momentaufnahme des Preises.

    Returns:
        Kachel-Raster.
    """
    change_abs = quote.change_abs
    change_pct = quote.change_pct
    return c.tiles(
        [
            ("Last price", fmt.format_currency(quote.price, quote.currency)),
            (
                "Change",
                fmt.format_currency(change_abs, quote.currency),
                fmt.direction_class(change_abs),
            ),
            (
                "Change %",
                fmt.format_percent(change_pct, signed=True),
                fmt.direction_class(change_pct),
            ),
            ("Previous close", fmt.format_currency(quote.previous_close, quote.currency)),
            ("Open", fmt.format_currency(quote.open, quote.currency)),
            ("Day high", fmt.format_currency(quote.day_high, quote.currency)),
            ("Day low", fmt.format_currency(quote.day_low, quote.currency)),
            ("Volume", fmt.format_compact(quote.volume, decimals=1)),
        ]
    )


def _master_tiles(instrument: Instrument) -> html.Div:
    """Baut die Kacheln der Stammdaten.

    Args:
        instrument: Stammdaten des Titels.

    Returns:
        Kachel-Raster.
    """
    from btlite.ui import search_bar

    return c.tiles(
        [
            (
                "Asset class",
                search_bar.ASSET_CLASS_LABELS.get(
                    instrument.asset_class.value, instrument.asset_class.value
                ),
            ),
            ("Exchange", instrument.exchange or fmt.NA_TEXT),
            ("Currency", instrument.currency.value),
            ("Sector", instrument.sector or fmt.NA_TEXT),
            ("Industry", instrument.industry or fmt.NA_TEXT),
            ("Country", instrument.country or fmt.NA_TEXT),
            ("Employees", fmt.format_integer(instrument.employees)),
            (
                "Market cap",
                fmt.format_currency(instrument.market_cap, instrument.currency, compact=True),
            ),
            ("Shares outstanding", fmt.format_compact(instrument.shares_outstanding, decimals=2)),
            (
                "Dividend per share",
                fmt.format_currency(instrument.dividend_rate, instrument.currency),
            ),
            ("Dividend yield", fmt.format_percent(instrument.dividend_yield, decimals=2)),
        ]
    )


def _ratio_tiles(ratios: RatioSet) -> html.Div:
    """Baut die Kacheln des Kennzahlen-Ueberblicks.

    Args:
        ratios: Kennzahlensatz des Titels.

    Returns:
        Kachel-Raster. Nicht berechenbare Kennzahlen zeigen "n/a".
    """
    items: list[tuple[str, str, str]] = []
    for key in _OVERVIEW_KEYS:
        ratio = ratios.ratios.get(key)
        if ratio is None:
            continue
        items.append(
            (
                ratio.label,
                fmt.format_ratio_value(ratio.value, ratio.unit),
                fmt.value_class(ratio.value),
            )
        )
    if not items:
        return c.placeholder("No ratios can be computed.")
    return c.tiles(items)


def _category_table(ratios: RatioSet) -> html.Div:
    """Listet alle Kennzahlen nach Kategorie.

    Args:
        ratios: Kennzahlensatz des Titels.

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
                    html.Td(
                        ", ".join(r.inputs_missing) if r.inputs_missing else "",
                        className="bt-notice",
                    ),
                ]
            )
            for r in entries
        ]
        blocks.append(
            html.Div(
                [
                    html.H4(_CATEGORY_LABELS.get(category, category.value), className="bt-subhead"),
                    html.Table(
                        [
                            html.Thead(
                                html.Tr(
                                    [html.Th("Ratio"), html.Th("Value"), html.Th("Missing inputs")]
                                )
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
        return c.placeholder("No ratios available.")
    return html.Div(blocks, className="bt-ratio-groups")


def _watch_action(ticker: str) -> html.Div:
    """Baut die Schaltflaeche, die den Titel auf die Watchlist legt.

    Sie steht auf dem Steckbrief und nicht in der Trefferliste der Suche:
    hier hat der Nutzer Kurs und Stammdaten vor sich, wenn er entscheidet,
    ob er den Titel beobachten will.

    Args:
        ticker: Kanonisches Symbol des angezeigten Titels.

    Returns:
        Container mit Schaltflaeche und leerem Meldungsbereich.
    """
    from btlite.ui import search_bar

    return html.Div(
        [
            html.Button(
                search_bar.WATCH_LABEL,
                id={"type": search_bar.WATCH_TYPE, "ticker": ticker},
                n_clicks=0,
                title=f"Add {ticker} to the {search_bar.DEFAULT_WATCHLIST!r} watchlist",
                className="btn btn-primary",
            ),
            html.Div(id=WATCH_FEEDBACK),
        ],
        className="bt-page-actions",
    )
