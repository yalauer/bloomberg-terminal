"""Seite WL - gespeichertes Depot und Watchlist.

Abgrenzung zu PORT: dort geht es um **Zielgewichte** aus der Optimierung,
hier um den **tatsaechlichen Bestand** mit Stueckzahlen und Einstandskursen.
Beides in eine Seite zu legen haette zwei verschiedene Bedeutungen von
"Portfolio" nebeneinandergestellt.

Architekturregel 4: die Seite rechnet nicht. Der Bestand kommt aus
:mod:`btlite.portfolio.store`, die Bewertung aus
:mod:`btlite.portfolio.valuation`, die Kurse aus dem injizierten Provider.

Zur Darstellung fehlender Kurse: kann eine Position nicht bewertet werden,
weist die Seite **keinen** Depotwert aus, sondern nennt die betroffenen
Titel. Eine Teilsumme als Gesamtwert waere eine Falschaussage ueber das
Vermoegen des Nutzers.

Die Seite bewertet und bucht, sie empfiehlt nicht. Aus keiner Zahl wird
eine Handlung abgeleitet.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any, Final

import pandas as pd
from dash import dcc, html

from btlite.core.errors import BTLiteError
from btlite.core.logging import get_logger
from btlite.models.common import Currency
from btlite.portfolio.valuation import value_portfolio
from btlite.ui import formatting as fmt
from btlite.ui.layouts import components as c

if TYPE_CHECKING:
    from btlite.models.common import Provenance
    from btlite.models.holdings import Holding, Portfolio, Watchlist
    from btlite.models.market import EarningsDate
    from btlite.portfolio.valuation import PortfolioValuation
    from btlite.ui.state import AppState

_LOG = get_logger(__name__)

#: Route dieser Seite.
PATH = "/wl"

#: Element-IDs.
PORTFOLIO_SELECT = "wl-portfolio-select"
PORTFOLIO_OUTPUT = "wl-portfolio-output"
WATCHLIST_SELECT = "wl-watchlist-select"
WATCHLIST_OUTPUT = "wl-watchlist-output"

#: Element-IDs der Depotverwaltung (anlegen und loeschen).
PORTFOLIO_NEW_NAME: Final[str] = "wl-portfolio-new-name"
PORTFOLIO_CREATE: Final[str] = "wl-portfolio-create"
PORTFOLIO_DELETE: Final[str] = "wl-portfolio-delete"
PORTFOLIO_FEEDBACK: Final[str] = "wl-portfolio-feedback"

#: Element-IDs der Watchlist-Verwaltung (anlegen und loeschen).
WATCHLIST_NEW_NAME: Final[str] = "wl-watchlist-new-name"
WATCHLIST_CREATE: Final[str] = "wl-watchlist-create"
WATCHLIST_DELETE: Final[str] = "wl-watchlist-delete"
WATCHLIST_FEEDBACK: Final[str] = "wl-watchlist-feedback"

#: Element-IDs des Formulars fuer eine neue Position.
HOLDING_NEW_TICKER: Final[str] = "wl-holding-new-ticker"
HOLDING_NEW_SHARES: Final[str] = "wl-holding-new-shares"
HOLDING_NEW_PRICE: Final[str] = "wl-holding-new-price"
HOLDING_NEW_DATE: Final[str] = "wl-holding-new-date"
HOLDING_NEW_CURRENCY: Final[str] = "wl-holding-new-currency"
HOLDING_ADD: Final[str] = "wl-holding-add"
HOLDING_FEEDBACK: Final[str] = "wl-holding-feedback"

#: Wildcard-Typen der bestehenden Positionszeilen. Der Schluessel ist der
#: nullbasierte INDEX in ``Portfolio.holdings``, denn genau so adressiert
#: der Store eine Haltung (``update_holding(name, index, ...)``). Ein
#: Ticker taugt hier nicht: derselbe Titel darf mehrfach vorkommen.
HOLDING_SHARES_TYPE: Final[str] = "wl-holding-shares"
HOLDING_PRICE_TYPE: Final[str] = "wl-holding-price"
HOLDING_DATE_TYPE: Final[str] = "wl-holding-date"
HOLDING_SAVE_TYPE: Final[str] = "wl-holding-save"
HOLDING_REMOVE_TYPE: Final[str] = "wl-holding-remove"

#: Element-IDs des Formulars fuer einen neuen Watchlist-Eintrag.
WATCHLIST_NEW_TICKER: Final[str] = "wl-watchlist-new-ticker"
WATCHLIST_NEW_NOTE: Final[str] = "wl-watchlist-new-note"
WATCHLIST_ADD: Final[str] = "wl-watchlist-add"

#: Wildcard-Typen der bestehenden Watchlist-Zeilen. Hier ist der TICKER der
#: Schluessel, nicht der Index: ``remove_from_watchlist`` nimmt den Ticker,
#: und eine Watchlist haelt jedes Symbol hoechstens einmal.
WATCH_NOTE_TYPE: Final[str] = "wl-watch-note"
WATCH_SAVE_TYPE: Final[str] = "wl-watch-save"
WATCH_REMOVE_TYPE: Final[str] = "wl-watch-remove"

#: Ueberschriften der editierbaren Positionszeilen. "Ticker" und "Currency"
#: sind reiner Text: ``update_holding`` nimmt weder das eine noch das
#: andere entgegen - ein anderes Symbol waere ein anderer Kauf.
HOLDING_EDIT_COLUMNS: Final[tuple[str, ...]] = (
    "Ticker",
    "Shares",
    "Buy price",
    "Buy date",
    "Currency",
    "",
)

#: Platzhalter des Datumsfeldes. Gewaehlt wurde ein Textfeld statt
#: ``dcc.DatePickerSingle``: dessen Kalender-Overlay bringt eigene helle
#: Fremdfarben mit, fuer die das Stylesheet keine Regeln hat - genau der
#: Fehlerfall, den ``bt-dropdown`` schon einmal ausgeloest hat.
DATE_PLACEHOLDER: Final[str] = "YYYY-MM-DD"

#: Ueberschriften der Bestandstabelle.
POSITION_COLUMNS: Final[tuple[str, ...]] = (
    "Shares",
    "Buy price",
    "Price",
    "Cost basis",
    "Market value",
    "P&L",
    "P&L %",
    "Next earnings",
)

#: Ueberschriften der Watchlist-Tabelle.
WATCHLIST_COLUMNS: Final[tuple[str, ...]] = ("Price", "Change", "Next earnings", "Note")

#: Hinweis an der Termin-Spalte. Die Quelle kennzeichnet nicht, ob ein
#: Termin bestaetigt ist; wer danach plant, soll das wissen.
EARNINGS_HINT: Final[str] = (
    "Next quarterly results. Dates come from the data source, are not marked "
    "as confirmed, and may shift."
)

#: Kennungen der beiden sortierbaren Tabellen. Sie unterscheiden die
#: Ueberschriften-Klicks, wenn beide Tabellen gleichzeitig sichtbar sind.
POSITIONS_TABLE: Final[str] = "wl-positions"
WATCHLIST_TABLE: Final[str] = "wl-watchlist"

#: Speicher der Sortierung je Tabelle. Ohne ihn faellt die Reihenfolge bei
#: jeder anderen Aenderung auf der Seite zurueck.
POSITIONS_SORT: Final[str] = "wl-positions-sort"
WATCHLIST_SORT: Final[str] = "wl-watchlist-sort"


def layout(state: AppState, tickers: list[str], args: tuple[str, ...] = ()) -> html.Div:
    """Baut die Seite WL.

    Args:
        state: UI-Zustand mit Provider und Depot-Speicher.
        tickers: Symbole aus der Kommandozeile. Werden hier nicht
            ausgewertet - der Bestand kommt aus der gespeicherten Datei,
            nicht aus der Eingabe.
        args: Optionaler Name eines Depots.

    Returns:
        Seiteninhalt mit Bestand und Watchlist.
    """
    del tickers

    try:
        portfolios = state.holdings.list_portfolios()
        watchlists = state.holdings.list_watchlists()
    except BTLiteError as exc:
        _LOG.warning("Depot-Datei nicht lesbar: %s", exc.detail or exc.user_message)
        return html.Div(
            [c.page_header("Portfolio and watchlist"), c.error_banner(exc)],
            className="bt-page bt-page-wl",
        )

    selected = _selected_name(args, portfolios)
    selected_watchlist = watchlists[0] if watchlists else None

    return html.Div(
        [
            # Die Sortierung je Tabelle. Ohne diesen Speicher faellt die
            # Reihenfolge zurueck, sobald irgendetwas anderes auf der Seite
            # neu gerendert wird - etwa nach dem Anlegen einer Position.
            dcc.Store(id=POSITIONS_SORT, data={"column": None, "desc": False}),
            dcc.Store(id=WATCHLIST_SORT, data={"column": None, "desc": False}),
            c.page_header(
                "Portfolio and watchlist",
                f"Stored in {state.settings.portfolio.holdings_file}",
            ),
            # Die Bewertung haengt an Kursen, und die sind verzoegert. Ohne
            # diesen Hinweis liesse sich der Depotwert fuer Echtzeit halten.
            c.provenance_note(_quote_provenance(state, portfolios), state.timezone),
            c.section(
                "Portfolio",
                c.controls(
                    c.labeled(
                        "Portfolio",
                        dcc.Dropdown(
                            id=PORTFOLIO_SELECT,
                            options=portfolio_options(state),
                            value=selected,
                            clearable=False,
                            placeholder="No portfolio saved yet",
                            className="bt-dropdown",
                        ),
                    ),
                    _manage_controls(
                        name_id=PORTFOLIO_NEW_NAME,
                        create_id=PORTFOLIO_CREATE,
                        delete_id=PORTFOLIO_DELETE,
                        placeholder="New portfolio name",
                        delete_title="Delete the selected portfolio and all its positions",
                    ),
                ),
                html.Div(id=PORTFOLIO_FEEDBACK, className="bt-feedback", role="status"),
                c.loading(
                    html.Div(
                        id=PORTFOLIO_OUTPUT,
                        children=render_portfolio(state, selected),
                    )
                ),
            ),
            c.section(
                "Watchlist",
                c.controls(
                    c.labeled(
                        "Watchlist",
                        dcc.Dropdown(
                            id=WATCHLIST_SELECT,
                            options=watchlist_options(state),
                            value=selected_watchlist,
                            clearable=False,
                            placeholder="No watchlist saved yet",
                            className="bt-dropdown",
                        ),
                    ),
                    _manage_controls(
                        name_id=WATCHLIST_NEW_NAME,
                        create_id=WATCHLIST_CREATE,
                        delete_id=WATCHLIST_DELETE,
                        placeholder="New watchlist name",
                        delete_title="Delete the selected watchlist",
                    ),
                ),
                html.Div(id=WATCHLIST_FEEDBACK, className="bt-feedback", role="status"),
                c.loading(
                    html.Div(
                        id=WATCHLIST_OUTPUT,
                        children=render_watchlist(state, selected_watchlist),
                    )
                ),
            ),
        ],
        className="bt-page bt-page-wl",
    )


def _quote_provenance(state: AppState, portfolios: list[str]) -> Provenance | None:
    """Ermittelt die Herkunft der Kurse, auf denen die Bewertung beruht.

    Genommen wird der erste bewertete Titel des ersten Depots. Alle Kurse
    einer Ansicht stammen aus demselben Abruf, die Angabe gilt also fuer
    die ganze Seite.

    Args:
        state: UI-Zustand.
        portfolios: Namen der gespeicherten Depots.

    Returns:
        Herkunftsnachweis, oder None wenn kein Kurs geladen werden konnte.
        Dann erscheint der neutrale Hinweistext.
    """
    for name in portfolios:
        portfolio = state.holdings.get_portfolio(name)
        if portfolio is None or not portfolio.tickers:
            continue
        try:
            quotes = state.quotes(sorted(portfolio.tickers))
        except BTLiteError:
            return None
        for quote in quotes.values():
            return quote.provenance
    return None


def _selected_name(args: tuple[str, ...], available: list[str]) -> str | None:
    """Waehlt das anzuzeigende Depot.

    Args:
        args: Argumente hinter dem Mnemonic.
        available: Namen der gespeicherten Depots.

    Returns:
        Name aus den Argumenten, wenn er existiert, sonst das erste Depot,
        sonst None.
    """
    for arg in args:
        candidate = arg.strip()
        for name in available:
            if candidate.casefold() == name.casefold():
                return name
    return available[0] if available else None


def _manage_controls(
    *,
    name_id: str,
    create_id: str,
    delete_id: str,
    placeholder: str,
    delete_title: str,
) -> html.Div:
    """Baut Namensfeld, Anlegen- und Loeschen-Knopf einer Sammlung.

    Depot und Watchlist werden gleich verwaltet, deshalb eine Funktion fuer
    beide. Der Loeschen-Knopf bezieht sich immer auf die aktuelle Auswahl
    des jeweiligen Dropdowns.

    Args:
        name_id: Element-ID des Namensfeldes fuer die Neuanlage.
        create_id: Element-ID des Anlegen-Knopfs.
        delete_id: Element-ID des Loeschen-Knopfs.
        placeholder: Platzhalter des Namensfeldes.
        delete_title: Tooltip des Loeschen-Knopfs. Er sagt, was verloren
            geht - ein blosses "Delete" waere folgenlos aussehend.

    Returns:
        Zeile aus Eingabefeld und beiden Knoepfen.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Name", className="bt-field-label"),
                    dcc.Input(
                        id=name_id,
                        type="text",
                        value="",
                        placeholder=placeholder,
                        debounce=True,
                        className="bt-field-input bt-field-wide",
                    ),
                ],
                className="bt-field",
            ),
            html.Button(
                "Create",
                id=create_id,
                n_clicks=0,
                className="bt-manage-action",
            ),
            html.Button(
                "Delete",
                id=delete_id,
                n_clicks=0,
                title=delete_title,
                className="bt-manage-action bt-manage-danger",
            ),
        ],
        className="bt-manage",
    )


def portfolio_options(state: AppState) -> list[dict[str, str]]:
    """Liefert die Auswahlliste der gespeicherten Depots.

    Der Callback-Agent setzt damit die Optionen des Dropdowns neu, nachdem
    ein Depot angelegt oder geloescht wurde. Das Dropdown liegt ausserhalb
    der Ausgabe-Divs und wird deshalb getrennt aktualisiert.

    Args:
        state: UI-Zustand mit Depot-Speicher.

    Returns:
        Liste aus ``{"label": name, "value": name}``, alphabetisch. Bei
        nicht lesbarer Datei eine leere Liste - die Fehlermeldung traegt
        das Banner der Seite, nicht das Dropdown.
    """
    try:
        names = state.holdings.list_portfolios()
    except BTLiteError as exc:
        _LOG.warning("Depotliste nicht lesbar: %s", exc.detail or exc.user_message)
        return []
    return [{"label": name, "value": name} for name in names]


def watchlist_options(state: AppState) -> list[dict[str, str]]:
    """Liefert die Auswahlliste der gespeicherten Watchlists.

    Args:
        state: UI-Zustand mit Depot-Speicher.

    Returns:
        Liste aus ``{"label": name, "value": name}``, alphabetisch. Bei
        nicht lesbarer Datei eine leere Liste.
    """
    try:
        names = state.holdings.list_watchlists()
    except BTLiteError as exc:
        _LOG.warning("Watchlisten nicht lesbar: %s", exc.detail or exc.user_message)
        return []
    return [{"label": name, "value": name} for name in names]


def render_portfolio(
    state: AppState,
    name: str | None,
    *,
    sort_by: str | None = None,
    sort_desc: bool = False,
) -> html.Div:
    """Rendert Bestand und Bewertung eines Depots.

    Args:
        state: UI-Zustand.
        name: Name des Depots, oder None wenn keines existiert.
        sort_by: Spalte der Bestandstabelle, nach der sortiert wird.
        sort_desc: Absteigend statt aufsteigend.

    Returns:
        Bewerteter Bestand, Hinweis, oder Fehlerbanner.
    """
    if name is None:
        return c.placeholder(
            "No portfolio saved yet. Create one above, or use the search to pick a ticker."
        )

    try:
        portfolio = state.holdings.get_portfolio(name)
    except BTLiteError as exc:
        _LOG.warning("Depot %s nicht lesbar: %s", name, exc.detail or exc.user_message)
        return html.Div([c.error_banner(exc)])

    if portfolio is None:
        return c.placeholder(f"Portfolio {name} no longer exists.")

    # Ein leeres Depot zeigt trotzdem das Anlege-Formular. Ohne das kaeme
    # der Nutzer nie in ein frisch erzeugtes Depot hinein.
    if not portfolio.holdings:
        return html.Div(
            [
                c.placeholder(f"Portfolio {name} is empty."),
                _holdings_editor(portfolio),
            ]
        )

    try:
        quotes = state.quotes(sorted(portfolio.tickers))
        valuation = value_portfolio(portfolio, quotes, state.rates)
    except BTLiteError as exc:
        _LOG.warning("Depot %s nicht bewertbar: %s", name, exc.detail or exc.user_message)
        # Der Bestand ist bekannt, nur seine Bewertung nicht. Die Bearbeitung
        # bleibt erreichbar, sonst sperrte ein Kursausfall die Korrektur.
        return html.Div([c.error_banner(exc), _holdings_editor(portfolio)])

    children: list[Any] = [_summary_tiles(valuation)]

    if valuation.unpriced:
        # Zwei verschiedene Ursachen fuehren hierher, und der Nutzer kann
        # nur die eine beheben: ein fehlender Aktienkurs ist ein Mangel der
        # Quelle, ein fehlender Wechselkurs trifft offline jeden Kauf in
        # fremder Waehrung. Beides als "kein Kurs" zu melden, liesse ihn
        # an der falschen Stelle suchen.
        messages = [
            "No price is available for "
            + ", ".join(valuation.unpriced)
            + ". The total value is therefore not reported, only the "
            "individual positions that can be valued."
        ]
        if valuation.cost_basis is None:
            messages.append(
                "At least one position was bought in a different currency and the "
                "exchange rate for its purchase date is unavailable - offline, or the "
                "source could not be reached."
            )
        banner = c.warning_banner(messages)
        if banner is not None:
            children.append(banner)

    # Die Termine sind eine Zugabe: scheitert ihr Abruf, bleibt die Spalte
    # leer und die Bewertung steht trotzdem. state.earnings_dates faengt
    # dafuer selbst und liefert dann ein leeres Ergebnis.
    earnings = state.earnings_dates(sorted(portfolio.tickers))
    children.append(_positions_table(valuation, earnings, sort_by=sort_by, sort_desc=sort_desc))
    children.append(_holdings_editor(portfolio))
    return html.Div(children)


def _summary_tiles(valuation: PortfolioValuation) -> html.Div:
    """Baut die Kopfzahlen eines Depots.

    Args:
        valuation: Bewertetes Depot.

    Returns:
        Kachelreihe mit Einstand, Marktwert und Ergebnis.
    """
    currency = valuation.currency
    result_class = fmt.direction_class(valuation.gain_pct)
    result = (
        f"{fmt.direction_arrow(valuation.gain_pct)} "
        f"{fmt.format_currency(valuation.gain_abs, currency=currency)}"
        if valuation.gain_abs is not None
        else fmt.NA_TEXT
    )
    return c.tiles(
        [
            ("Positions", str(len(valuation.positions))),
            ("Cost basis", fmt.format_currency(valuation.cost_basis, currency=currency)),
            ("Market value", fmt.format_currency(valuation.market_value, currency=currency)),
            ("P&L", result, result_class),
            (
                "P&L %",
                fmt.format_percent(valuation.gain_pct, decimals=2, signed=True),
                result_class,
            ),
        ]
    )


def _directional_cell(raw: object, text: str) -> html.Span | str:
    """Faerbt eine vorzeichenbehaftete Tabellenzelle nach ihrer Richtung.

    Positive Werte tragen ``bt-up``, negative ``bt-down``, exakt 0 bleibt
    neutral. Der Richtungspfeil steht bereits in ``text`` - die Farbe ist
    nur zusaetzlicher Traeger, nie der einzige.

    Ein fehlender Wert bleibt der **reine String** :data:`fmt.NA_TEXT`:
    ``data_table`` erkennt fehlende Zellen am Textvergleich und gibt ihnen
    die Klasse ``bt-na``. In einem Span verpackt ginge das verloren.

    Args:
        raw: Rohwert der Zelle, dezimal oder als Waehrungsbetrag.
        text: Fertig formatierter Zelltext samt Pfeil.

    Returns:
        Span mit Richtungsklasse, oder ``text`` unveraendert bei fehlendem
        oder neutralem Wert.
    """
    if not isinstance(raw, float) or text == fmt.NA_TEXT:
        return text
    css = fmt.direction_class(raw)
    if css not in ("bt-up", "bt-down"):
        return text
    return html.Span(text, className=css)


def _positions_table(
    valuation: PortfolioValuation,
    earnings: dict[str, EarningsDate],
    *,
    sort_by: str | None = None,
    sort_desc: bool = False,
) -> html.Table:
    """Baut die Bestandstabelle.

    Der Frame traegt **Rohwerte**, die Formatierung uebernehmen die
    Formatter der Tabelle. Nur so laesst sich nach Zahlen sortieren:
    nach der Zeichenkette "1,234.50 EUR" sortiert, kaeme 9 nach 10.

    Args:
        valuation: Bewertetes Depot.
        earnings: Quartalstermine je Ticker. Fehlende Eintraege sind
            erlaubt - ein unbekannter Termin ist kein Fehler.
        sort_by: Spalte, nach der sortiert wird.
        sort_desc: Absteigend statt aufsteigend.

    Returns:
        Tabelle mit einer Zeile je Titel.
    """
    currency = valuation.currency
    rows: dict[str, list[Any]] = {
        position.ticker: [
            position.shares,
            position.average_price,
            position.price,
            position.cost_basis,
            position.market_value,
            position.gain_abs,
            position.gain_pct,
            _earnings_value(earnings.get(position.ticker)),
        ]
        for position in valuation.positions
    }

    frame = pd.DataFrame.from_dict(rows, orient="index", columns=list(POSITION_COLUMNS))
    frame.index.name = "Ticker"
    frame = c.sort_frame(frame, by=sort_by, descending=sort_desc, index_label="Ticker")

    def _money(raw: object) -> str:
        return fmt.format_currency(raw if isinstance(raw, float) else None, currency=currency)

    # NaN statt None: fehlt der Wert nur in EINIGEN Zeilen, macht pandas die
    # Spalte zu float64 und aus None ein NaN - ohne ``is_missing`` stuende
    # dann " n/a" mit Leerzeichen in der Zelle und ``bt-na`` griffe nicht.
    #
    # P&L und P&L % stehen nebeneinander und werden gleich gefaerbt: ein
    # rotes Prozent neben einem ungefaerbten Betrag saehe nach Widerspruch aus.
    def _signed_money(raw: object) -> html.Span | str:
        if not isinstance(raw, float) or fmt.is_missing(raw):
            return fmt.NA_TEXT
        text = f"{fmt.direction_arrow(raw)} {fmt.format_currency(raw, currency=currency)}"
        return _directional_cell(raw, text)

    def _signed_pct(raw: object) -> html.Span | str:
        if not isinstance(raw, float) or fmt.is_missing(raw):
            return fmt.NA_TEXT
        text = f"{fmt.direction_arrow(raw)} {fmt.format_percent(raw, decimals=2, signed=True)}"
        return _directional_cell(raw, text)

    # Die Terminspalte wird hier fertig formatiert, nicht ueber einen
    # Formatter: sie braucht den Ticker der Zeile, um ein "kein Termin"
    # von einem "fuehrt keine Earnings" zu unterscheiden.
    without = _earnings_markers(earnings)
    shown = frame.copy()
    shown["Next earnings"] = [
        _earnings_cell(str(ticker), value, without)
        for ticker, value in zip(frame.index, frame["Next earnings"], strict=True)
    ]

    return c.data_table(
        shown,
        index_label="Ticker",
        formatters={
            "Shares": lambda v: fmt.format_number(v, decimals=4),
            "Buy price": _money,
            "Price": _money,
            "Cost basis": _money,
            "Market value": _money,
            "P&L": _signed_money,
            "P&L %": _signed_pct,
            "Next earnings": str,
        },
        column_hints={"Next earnings": EARNINGS_HINT},
        sort_table=POSITIONS_TABLE,
        sort_by=sort_by,
        sort_desc=sort_desc,
    )


def _earnings_value(entry: EarningsDate | None) -> date | None:
    """Macht aus einem Termin einen sortierbaren Zellwert.

    Die Tabelle sortiert ueber die Rohwerte, und die muessen untereinander
    vergleichbar sein. Deshalb steht hier **ausschliesslich** ein ``date``
    oder ``None`` - ein Markerstring fuer "fuehrt keine Earnings" haette
    die Sortierung zum Absturz gebracht:

        TypeError: '<' not supported between instances of 'str' and 'date'

    Genau dieser Fall trat auf, sobald ein ETF im Depot lag und der Nutzer
    nach dem Termin sortierte. Ob ein Instrument ueberhaupt Quartalszahlen
    fuehrt, steht deshalb getrennt in ``_earnings_markers``.

    Fehlende Termine landen als ``None`` am Ende der Sortierung - dort
    gehoeren sie hin, denn "kein Termin bekannt" ist keine Aussage ueber
    die Naehe eines Termins.

    Args:
        entry: Termin-Eintrag, oder ``None`` wenn keiner vorliegt.

    Returns:
        Das Datum, oder ``None``.
    """
    if entry is None or not entry.has_earnings:
        return None
    return entry.next_date


def _earnings_markers(earnings: dict[str, EarningsDate]) -> set[str]:
    """Nennt die Ticker, die grundsaetzlich keine Quartalszahlen fuehren.

    Getrennt vom Sortierwert gehalten, siehe :func:`_earnings_value`.

    Args:
        earnings: Termine je Ticker.

    Returns:
        Ticker von ETFs, Krypto und Indizes.
    """
    return {ticker for ticker, entry in earnings.items() if not entry.has_earnings}


def _earnings_cell(ticker: str, raw: object, without: set[str]) -> str:
    """Formatiert die Zelle eines Quartalstermins.

    Args:
        ticker: Symbol der Zeile.
        raw: Rohwert aus dem Frame.
        without: Ticker ohne Quartalszahlen.

    Returns:
        Datum mit Abstand in Tagen, der Gedankenstrich fuer Instrumente
        ohne Quartalszahlen, sonst ``"n/a"``.
    """
    if ticker in without:
        return fmt.NO_EARNINGS_TEXT
    if isinstance(raw, date):
        return fmt.format_earnings_date(raw)
    return fmt.NA_TEXT


def _holdings_editor(portfolio: Portfolio) -> html.Div:
    """Baut den bearbeitbaren Abschnitt "Holdings".

    Anders als die Bewertungstabelle darueber fasst dieser Abschnitt
    **nichts** zusammen: jede Zeile ist ein Kauf. Zwei Kaeufe desselben
    Titels stehen als zwei Zeilen da, weil der Store sie ueber ihren Index
    einzeln adressiert und sonst nur einer davon korrigierbar waere.

    Args:
        portfolio: Depot, dessen Haltungen bearbeitbar werden.

    Returns:
        Abschnitt aus Anlege-Formular, Rueckmeldezeile und einer Zeile je
        Kauf.
    """
    rows: list[Any] = [
        _holding_row(index, holding) for index, holding in enumerate(portfolio.holdings)
    ]
    return c.section(
        "Holdings",
        html.Div(
            "One row per purchase. A second buy of the same ticker keeps its own "
            "cost basis and is edited separately.",
            className="bt-notice",
        ),
        _new_holding_form(portfolio.currency),
        html.Div(id=HOLDING_FEEDBACK, className="bt-feedback", role="status"),
        html.Div([_holdings_header(), *rows], className="bt-holdings"),
        className="bt-holdings-section",
    )


def _holdings_header() -> html.Div:
    """Baut die Kopfzeile ueber den bearbeitbaren Positionszeilen.

    Returns:
        Zeile mit den Spaltenbeschriftungen aus
        :data:`HOLDING_EDIT_COLUMNS`.
    """
    return html.Div(
        [html.Div(title, className="bt-holdings-head-cell") for title in HOLDING_EDIT_COLUMNS],
        className="bt-holdings-row bt-holdings-head",
    )


def _new_holding_form(default_currency: Currency) -> html.Div:
    """Baut das Formular fuer einen zusaetzlichen Kauf.

    Das Datum ist ein Textfeld mit Platzhalter ``YYYY-MM-DD``, kein
    ``dcc.DatePickerSingle``: dessen Kalender-Overlay bringt eigene helle
    Fremdfarben mit, fuer die dieses Projekt keine Regeln hat.

    Args:
        default_currency: Berichtswaehrung des Depots. Sie ist die
            Vorbelegung, nicht die Vorschrift - gekauft wird oft in einer
            anderen Waehrung.

    Returns:
        Formularzeile mit allen fuenf Feldern und dem Knopf "Add".
    """
    return html.Div(
        [
            _field(
                "Ticker",
                dcc.Input(
                    id=HOLDING_NEW_TICKER,
                    type="text",
                    value="",
                    placeholder="AAPL",
                    debounce=True,
                    className="bt-field-input",
                ),
            ),
            _field(
                "Shares",
                dcc.Input(
                    id=HOLDING_NEW_SHARES,
                    type="number",
                    min=0,
                    step="any",
                    placeholder="25",
                    className="bt-field-input",
                ),
            ),
            _field(
                "Buy price",
                dcc.Input(
                    id=HOLDING_NEW_PRICE,
                    type="number",
                    min=0,
                    step="any",
                    placeholder="142.50",
                    className="bt-field-input",
                ),
            ),
            _field(
                "Buy date",
                dcc.Input(
                    id=HOLDING_NEW_DATE,
                    type="text",
                    value="",
                    placeholder=DATE_PLACEHOLDER,
                    debounce=True,
                    className="bt-field-input",
                ),
            ),
            _field(
                "Currency",
                dcc.Dropdown(
                    id=HOLDING_NEW_CURRENCY,
                    options=[{"label": c.value, "value": c.value} for c in Currency],
                    value=default_currency.value,
                    clearable=False,
                    className="bt-dropdown bt-field-select",
                ),
            ),
            html.Button(
                "Add",
                id=HOLDING_ADD,
                n_clicks=0,
                title="Add this purchase to the portfolio",
                className="bt-manage-action",
            ),
        ],
        className="bt-holdings-form",
    )


def _holding_row(index: int, holding: Holding) -> html.Div:
    """Baut eine bearbeitbare Zeile fuer einen einzelnen Kauf.

    Ticker und Waehrung stehen als Text da, nicht als Eingabefeld:
    ``update_holding`` nimmt beides nicht entgegen. Ein anderes Symbol
    waere ein anderer Kauf, keine Korrektur.

    Der Index erscheint bewusst nirgends auf dem Bildschirm. Er ist die
    technische Adresse der Haltung im Store, keine Information ueber das
    Depot.

    Args:
        index: Nullbasierter Index der Haltung in ``Portfolio.holdings``.
            Der Callback adressiert damit ``update_holding`` und
            ``remove_holding``.
        holding: Die anzuzeigende Haltung. Ihre Werte sind die Vorbelegung
            der Felder.

    Returns:
        Zeile aus Text, drei Eingabefeldern und zwei Knoepfen.
    """
    return html.Div(
        [
            html.Div(holding.ticker, className="bt-holdings-cell bt-holdings-ticker"),
            _field(
                "Shares",
                dcc.Input(
                    id={"type": HOLDING_SHARES_TYPE, "index": index},
                    type="number",
                    min=0,
                    step="any",
                    value=holding.shares,
                    debounce=True,
                    className="bt-field-input",
                ),
            ),
            _field(
                "Buy price",
                dcc.Input(
                    id={"type": HOLDING_PRICE_TYPE, "index": index},
                    type="number",
                    min=0,
                    step="any",
                    value=holding.buy_price,
                    debounce=True,
                    className="bt-field-input",
                ),
            ),
            _field(
                "Buy date",
                dcc.Input(
                    id={"type": HOLDING_DATE_TYPE, "index": index},
                    type="text",
                    value=holding.buy_date.isoformat(),
                    placeholder=DATE_PLACEHOLDER,
                    debounce=True,
                    className="bt-field-input",
                ),
            ),
            html.Div(holding.currency.value, className="bt-holdings-cell"),
            html.Div(
                [
                    html.Button(
                        "Save",
                        id={"type": HOLDING_SAVE_TYPE, "index": index},
                        n_clicks=0,
                        title=f"Save the changed values of this {holding.ticker} purchase",
                        className="bt-manage-action",
                    ),
                    html.Button(
                        "Remove",
                        id={"type": HOLDING_REMOVE_TYPE, "index": index},
                        n_clicks=0,
                        title=f"Remove this {holding.ticker} purchase from the portfolio",
                        className="bt-manage-action bt-manage-danger",
                    ),
                ],
                className="bt-holdings-actions",
            ),
        ],
        className="bt-holdings-row",
    )


def _field(label: str, control: Any) -> html.Div:
    """Beschriftet ein Eingabefeld einer Formularzeile.

    Args:
        label: Sichtbare Beschriftung, englisch.
        control: Das Eingabeelement.

    Returns:
        Beschriftetes Feld.
    """
    return html.Div(
        [html.Label(label, className="bt-field-label"), control],
        className="bt-field",
    )


def render_watchlist(
    state: AppState,
    name: str | None,
    *,
    sort_by: str | None = None,
    sort_desc: bool = False,
) -> html.Div:
    """Rendert eine Watchlist mit aktuellen Kursen.

    Args:
        state: UI-Zustand.
        name: Name der Watchlist, oder None wenn keine existiert.
        sort_by: Spalte, nach der sortiert wird.
        sort_desc: Absteigend statt aufsteigend.

    Returns:
        Kurstabelle, Hinweis, oder Fehlerbanner.
    """
    if name is None:
        return c.placeholder(
            "No watchlist saved yet. Create one above, or use the search to pick a ticker."
        )

    try:
        watchlist = state.holdings.get_watchlist(name)
    except BTLiteError as exc:
        _LOG.warning("Watchlist %s nicht ladbar: %s", name, exc.detail or exc.user_message)
        return html.Div([c.error_banner(exc)])

    if watchlist is None:
        return c.placeholder(f"Watchlist {name} no longer exists.")

    # Auch eine leere Liste zeigt das Aufnahme-Formular, sonst bliebe eine
    # frisch angelegte Watchlist fuer immer leer.
    if not watchlist.entries:
        return html.Div(
            [c.placeholder(f"Watchlist {name} is empty."), _watchlist_editor(watchlist)]
        )

    try:
        quotes = state.quotes([entry.ticker for entry in watchlist.entries])
    except BTLiteError as exc:
        _LOG.warning("Watchlist %s ohne Kurse: %s", name, exc.detail or exc.user_message)
        # Ohne Kurse bleibt die Liste bearbeitbar: ein Ausfall der Quelle
        # darf das Entfernen eines Eintrags nicht verhindern.
        return html.Div([c.error_banner(exc), _watchlist_editor(watchlist)])

    earnings = state.earnings_dates([entry.ticker for entry in watchlist.entries])
    return html.Div(
        [
            _watchlist_table(watchlist, quotes, earnings, sort_by=sort_by, sort_desc=sort_desc),
            _watchlist_editor(watchlist),
        ]
    )


def _watchlist_editor(watchlist: Watchlist) -> html.Div:
    """Baut den bearbeitbaren Abschnitt der Watchlist.

    Args:
        watchlist: Die anzuzeigende Liste.

    Returns:
        Abschnitt aus Aufnahme-Formular und einer Zeile je Eintrag.
    """
    rows: list[Any] = [_watch_row(entry.ticker, entry.note) for entry in watchlist.entries]
    return c.section(
        "Watched symbols",
        _new_watch_form(),
        html.Div([_watch_header(), *rows], className="bt-holdings"),
        className="bt-holdings-section",
    )


def _watch_header() -> html.Div:
    """Baut die Kopfzeile ueber den bearbeitbaren Watchlist-Zeilen.

    Returns:
        Zeile mit den Spaltenbeschriftungen.
    """
    return html.Div(
        [html.Div(title, className="bt-holdings-head-cell") for title in ("Ticker", "Note", "")],
        className="bt-holdings-row bt-holdings-head",
    )


def _new_watch_form() -> html.Div:
    """Baut das Formular fuer ein zusaetzlich beobachtetes Symbol.

    Returns:
        Formularzeile aus Symbol, Notiz und dem Knopf "Watch".
    """
    return html.Div(
        [
            _field(
                "Ticker",
                dcc.Input(
                    id=WATCHLIST_NEW_TICKER,
                    type="text",
                    value="",
                    placeholder="KO",
                    debounce=True,
                    className="bt-field-input",
                ),
            ),
            _field(
                "Note",
                dcc.Input(
                    id=WATCHLIST_NEW_NOTE,
                    type="text",
                    value="",
                    placeholder="Why you are watching it",
                    debounce=True,
                    className="bt-field-input bt-field-wide",
                ),
            ),
            html.Button(
                "Watch",
                id=WATCHLIST_ADD,
                n_clicks=0,
                title="Add this symbol to the watchlist",
                className="bt-manage-action",
            ),
        ],
        className="bt-holdings-form",
    )


def _watch_row(ticker: str, note: str | None) -> html.Div:
    """Baut eine bearbeitbare Zeile eines beobachteten Titels.

    Schluessel ist hier der Ticker, nicht ein Index: eine Watchlist haelt
    jedes Symbol hoechstens einmal, und ``remove_from_watchlist`` nimmt
    genau den Ticker.

    Args:
        ticker: Kanonisches Symbol des Eintrags.
        note: Gespeicherte Notiz, oder None. None wird zum leeren Feld -
            ein Platzhaltertext im Wertfeld waere eine erfundene Notiz.

    Returns:
        Zeile aus Symbol, Notizfeld und zwei Knoepfen.
    """
    return html.Div(
        [
            html.Div(ticker, className="bt-holdings-cell bt-holdings-ticker"),
            _field(
                "Note",
                dcc.Input(
                    id={"type": WATCH_NOTE_TYPE, "ticker": ticker},
                    type="text",
                    value=note or "",
                    placeholder="No note",
                    debounce=True,
                    className="bt-field-input bt-field-wide",
                ),
            ),
            html.Div(
                [
                    html.Button(
                        "Save",
                        id={"type": WATCH_SAVE_TYPE, "ticker": ticker},
                        n_clicks=0,
                        title=f"Save the note for {ticker}",
                        className="bt-manage-action",
                    ),
                    html.Button(
                        "Remove",
                        id={"type": WATCH_REMOVE_TYPE, "ticker": ticker},
                        n_clicks=0,
                        title=f"Remove {ticker} from the watchlist",
                        className="bt-manage-action bt-manage-danger",
                    ),
                ],
                className="bt-holdings-actions",
            ),
        ],
        className="bt-holdings-row",
    )


def _watchlist_table(
    watchlist: Watchlist,
    quotes: dict[str, Any],
    earnings: dict[str, EarningsDate],
    *,
    sort_by: str | None = None,
    sort_desc: bool = False,
) -> html.Table:
    """Baut die Watchlist-Tabelle.

    Wie die Bestandstabelle traegt der Frame Rohwerte; die Formatierung
    macht die Tabelle. Sortiert wird damit nach Zahlen, nicht nach Text.

    Args:
        watchlist: Gespeicherte Watchlist.
        quotes: Kurse je Ticker. Fehlende Eintraege sind erlaubt.
        earnings: Quartalstermine je Ticker, ebenfalls optional.
        sort_by: Spalte, nach der sortiert wird.
        sort_desc: Absteigend statt aufsteigend.

    Returns:
        Tabelle mit einer Zeile je beobachtetem Titel.
    """
    rows: dict[str, list[Any]] = {}
    currencies: dict[str, Any] = {}
    for entry in watchlist.entries:
        quote = quotes.get(entry.ticker)
        price = getattr(quote, "price", None)
        previous = getattr(quote, "previous_close", None)
        currencies[entry.ticker] = getattr(quote, "currency", None)

        # Tagesveraenderung nur, wenn beide Werte vorliegen. Ein fehlender
        # Vortagesschluss ist kein Nullwert.
        change = (price - previous) / previous if price is not None and previous else None
        rows[entry.ticker] = [
            price,
            change,
            _earnings_value(earnings.get(entry.ticker)),
            entry.note or "",
        ]

    frame = pd.DataFrame.from_dict(rows, orient="index", columns=list(WATCHLIST_COLUMNS))
    frame.index.name = "Ticker"
    frame = c.sort_frame(frame, by=sort_by, descending=sort_desc, index_label="Ticker")

    # NaN abfangen wie in der Bestandstabelle: ein fehlender Kurs neben
    # bekannten macht die Spalte zu float64.
    def _change(raw: object) -> html.Span | str:
        if not isinstance(raw, float) or fmt.is_missing(raw):
            return fmt.NA_TEXT
        text = f"{fmt.direction_arrow(raw)} {fmt.format_percent(raw, decimals=2, signed=True)}"
        return _directional_cell(raw, text)

    # Die Waehrung haengt am TITEL, nicht an der Spalte: eine Watchlist darf
    # Titel aus verschiedenen Waehrungsraeumen mischen. Deshalb wird die
    # Preisspalte hier fertig formatiert - nach dem Sortieren, das auf den
    # Rohwerten lief.
    without = _earnings_markers(earnings)
    shown = frame.copy()
    shown["Next earnings"] = [
        _earnings_cell(str(ticker), value, without)
        for ticker, value in zip(frame.index, frame["Next earnings"], strict=True)
    ]
    shown["Price"] = [
        fmt.format_currency(
            value if isinstance(value, float) else None, currency=currencies.get(ticker)
        )
        for ticker, value in zip(frame.index, frame["Price"], strict=True)
    ]

    return c.data_table(
        shown,
        index_label="Ticker",
        formatters={
            "Price": str,
            "Change": _change,
            "Next earnings": str,
            "Note": lambda v: str(v or ""),
        },
        column_hints={"Next earnings": EARNINGS_HINT},
        sort_table=WATCHLIST_TABLE,
        sort_by=sort_by,
        sort_desc=sort_desc,
    )
