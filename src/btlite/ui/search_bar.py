"""Namenssuche mit direktem Sprung auf den Steckbrief.

Die Kommandozeile aus ``ui/command_bar.py`` setzt voraus, dass der Nutzer
das Kuerzel und das Mnemonic bereits kennt - ``AAPL FA GO``. Wer nur
"servicenow" kennt, kommt dort nicht weiter. Dieses Modul schliesst die
Luecke.

Die Eingabe geht an :class:`SymbolSearch` des injizierten Providers. Die
Trefferliste fuehrt den **vollen Firmennamen** als Zeile an, darunter das
Kuerzel mit Boerse (``NASDAQ:RKLB``) und die ausgeschriebene
Instrumentenklasse. Wer "rocket lab" sucht, kennt den Namen und nicht das
Kuerzel - also steht der Name oben.

Ein Klick auf den Treffer oeffnet **unmittelbar den Steckbrief** des
Titels. Eine fruehere Fassung schob eine Auswahl der Auswertungen
dazwischen; der Nutzer erwartete nach dem Klick auf den Namen die Seite
der Firma und bekam eine weitere Liste. Die anderen Auswertungen sind vom
Steckbrief aus ueber die Navigation erreichbar, das Merken ueber die
Schaltflaeche auf der Steckbriefseite selbst.

Architekturregel 4: hier steht keine Geschaeftslogik. Das Ranking macht
``data/search.py``, den Abruf der Provider; dieses Modul baut ausschliesslich
Komponenten und liest fertige Modellobjekte.

Warum der Steckbrief das Ziel ist
---------------------------------
Der Steckbrief ist die einzige Seite, die zu **jedem** gefundenen Titel
etwas zeigt. Kursverlauf und Backtest brauchen eine Kurshistorie, die
Bilanzseite braucht Abschluesse - ein frisch gelisteter Wert oder ein ETF
laeuft dort ins Leere. Der Steckbrief zeigt zur Not Stammdaten allein und
benennt fehlende Teile als solche. Er ist damit das einzige Ziel, das
nach einem Klick auf einen Namen nie in einer leeren Seite endet.

Die Watchlist erreicht der Nutzer ueber die Schaltflaeche auf dem
Steckbrief, nicht mehr aus der Trefferliste heraus: der Titel steht dort
mit Kurs und Stammdaten vor ihm, und das ist der Moment, in dem die
Entscheidung "merken" faellt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from btlite.core.logging import get_logger
from btlite.models.common import Currency
from btlite.ui.command_bar import Command

if TYPE_CHECKING:
    from dash import html

    from btlite.models.market import SearchHit
    from btlite.ui.state import AppState

_LOG = get_logger(__name__)

#: Element-IDs der Suchleiste.
SEARCH_INPUT: Final[str] = "symbol-search"
RESULT_OUTPUT: Final[str] = "symbol-search-results"
HIT_TYPE: Final[str] = "symbol-search-hit"

#: Wartezeit in Sekunden, bevor eine Eingabe zur Suche wird. Lang genug,
#: dass "rocket lab" eine Anfrage ausloest statt neun, kurz genug, dass es
#: sich wie Tippen anfuehlt.
SEARCH_DEBOUNCE_S: Final[float] = 0.4

#: Maximale Trefferzahl der Liste. Mehr als das ueberfuellt die Kopfzeile.
MAX_HITS: Final[int] = 8

#: Mnemonic der Seite, die ein Treffer oeffnet. Siehe Modul-Docstring:
#: der Steckbrief ist das einzige Ziel, das zu jedem Titel etwas zeigt.
TARGET_MNEMONIC: Final[str] = "DES"

#: Instrumentenklassen ausgeschrieben. Der Nutzer liest "Stock", nicht
#: den Rohwert "equity" der Quelle - Kleinschreibung und technische
#: Kuerzel gehoeren nicht in die Oberflaeche.
ASSET_CLASS_LABELS: Final[dict[str, str]] = {
    "equity": "Stock",
    "etf": "ETF",
    "index": "Index",
    "crypto": "Cryptocurrency",
    "fund": "Fund",
    "other": "Other",
}


#: Name der Watchlist, auf der die Suche ablegt. Bewusst eine feste Liste
#: statt einer Auswahl: die Aktion soll ein Klick sein, kein Dialog. Wer
#: mehrere Listen fuehrt, ordnet auf der Seite WL um.
DEFAULT_WATCHLIST: Final[str] = "Watchlist"

#: ID der Schaltflaeche, die einen Titel auf die Watchlist legt. Sie steht
#: sowohl an der Trefferzeile als auch auf der Steckbriefseite: der Nutzer
#: wollte direkt aus der Suche heraus merken koennen, ohne die Seite zu
#: oeffnen - wer erst hinsehen will, findet den Knopf dort wieder.
WATCH_TYPE: Final[str] = "symbol-search-watch"

#: Name des Depots, in das die Suche ablegt. Wie bei der Watchlist eine
#: feste Liste statt einer Auswahl - wer mehrere Depots fuehrt, ordnet auf
#: der Seite WL um.
DEFAULT_PORTFOLIO: Final[str] = "Portfolio"

#: IDs rund um die Depot-Aufnahme. Der Toggle klappt das Formular auf,
#: Submit legt die Position an; die beiden Felder tragen Stueckzahl und
#: Einstandskurs, die ``Holding`` beide groesser null verlangt.
PORTFOLIO_TOGGLE_TYPE: Final[str] = "symbol-search-portfolio"
PORTFOLIO_FORM_TYPE: Final[str] = "symbol-search-portfolio-form"
PORTFOLIO_SUBMIT_TYPE: Final[str] = "symbol-search-portfolio-add"
SHARES_INPUT_TYPE: Final[str] = "symbol-search-shares"
PRICE_INPUT_TYPE: Final[str] = "symbol-search-price"

#: Waehrung, in der gekauft wurde. Wer an der Frankfurter Boerse kauft,
#: zahlt Euro - das Depot rechnet in seiner Berichtswaehrung. Der Betrag
#: wird zum Kurs des Kauftages umgerechnet, siehe docs/DATA_SOURCES.md.
CURRENCY_INPUT_TYPE: Final[str] = "symbol-search-currency"

#: Container fuer die Rueckmeldung der beiden Aktionen. Eigener Bereich
#: unter der Trefferliste, damit die Liste beim Ablegen stehen bleibt -
#: wer mehrere Titel merken will, soll nicht neu suchen muessen.
FEEDBACK_OUTPUT: Final[str] = "symbol-search-feedback"

#: Beschriftung des Depot-Knopfes.
PORTFOLIO_LABEL: Final[str] = "Portfolio"

#: Route der Seite, die die Watchlist zeigt. Konstante statt Literal, damit
#: eine Umbenennung der Route hier auffaellt.
WATCHLIST_PATH: Final[str] = "/wl"

#: Beschriftung dieser Schaltflaeche.
WATCH_LABEL: Final[str] = "Watchlist"


def search_field() -> html.Div:
    """Baut das Eingabefeld der Namenssuche.

    Returns:
        Feldcontainer fuer die Kopfzeile. Das Ergebnis der Suche landet in
        einem eigenen Container, siehe :func:`result_container`.
    """
    from dash import dcc, html

    return html.Div(
        [
            dcc.Input(
                id=SEARCH_INPUT,
                # NICHT type="search": der Browser setzt dort ein eigenes
                # Loesch-Kreuz, das die Eingabe leert, ohne dass Dash den
                # Callback ausloest - das Feld sah dann kaputt aus.
                type="text",
                placeholder="Search: servicenow",
                # debounce=True haette den Wert erst bei Enter oder beim
                # Verlassen des Feldes geschickt; der Nutzer tippte und
                # nichts geschah. Eine Zahl heisst: senden, sobald er
                # SEARCH_DEBOUNCE_S lang nicht getippt hat. Kein Abruf je
                # Tastendruck, aber auch kein Enter noetig.
                debounce=SEARCH_DEBOUNCE_S,
                value="",
                autoComplete="off",
                className="bt-search-input",
            )
        ],
        className="bt-search",
    )


def result_container() -> html.Div:
    """Baut den zunaechst leeren Container der Trefferliste.

    Returns:
        Container, den der Such-Callback fuellt.
    """
    from dash import html

    return html.Div(id=RESULT_OUTPUT, className="bt-search-results")


def asset_class_label(hit: SearchHit) -> str:
    """Schreibt die Instrumentenklasse eines Treffers aus.

    Args:
        hit: Treffer der Symbolsuche.

    Returns:
        Ausgeschriebene Bezeichnung, etwa "Stock". Unbekannte Werte der
        Quelle erscheinen unveraendert statt zu verschwinden - eine neue
        Klasse soll sichtbar sein, nicht still zu "Other" werden.
    """
    raw = str(hit.asset_class)
    return ASSET_CLASS_LABELS.get(raw, raw)


def hit_subtitle(hit: SearchHit) -> str:
    """Baut die Unterzeile eines Treffers.

    Args:
        hit: Treffer der Symbolsuche.

    Returns:
        Kuerzel mit Boerse, Instrumentenklasse und - falls die Quelle ihn
        nennt - der Sektor, mit Mittelpunkt getrennt.
    """
    parts = [hit.display, asset_class_label(hit)]
    if hit.sector:
        parts.append(hit.sector)
    return " · ".join(parts)


def hit_button(hit: SearchHit) -> html.Button:
    """Baut die Schaltflaeche eines Suchtreffers.

    Der **Firmenname** steht oben und traegt die Zeile: wer nach
    "rocket lab" sucht, kennt den Namen und sucht ihn in der Liste. Das
    Kuerzel stand frueher an dieser Stelle und der Name klein darunter -
    genau andersherum als der Nutzer liest. Fehlt der Name, tritt das
    Kuerzel an seine Stelle, damit die Zeile nie ohne Kopf bleibt.

    Args:
        hit: Treffer der Symbolsuche.

    Returns:
        Schaltflaeche mit Name und Unterzeile. Die ID traegt das Kuerzel,
        damit der Callback es ohne weiteren Abruf kennt.
    """
    from dash import html

    headline = hit.name or hit.display
    children: list[Any] = [
        html.Span(headline, className="bt-hit-name"),
        html.Span(hit_subtitle(hit), className="bt-hit-meta"),
    ]
    return html.Button(
        children,
        id={"type": HIT_TYPE, "ticker": hit.ticker},
        n_clicks=0,
        title=f"{hit.label} - open profile",
        className="bt-hit",
    )


def hit_row(hit: SearchHit) -> html.Div:
    """Baut eine Trefferzeile samt ihrer beiden Aktionen.

    Der Nutzer wollte direkt aus der Trefferliste heraus merken und
    kaufen koennen, ohne den Umweg ueber eine weitere Seite. Der
    Namensknopf oeffnet den Steckbrief, die beiden schmalen Knoepfe
    rechts legen den Titel ab.

    Beim Depot bleibt es nicht bei einem Klick: ``Holding`` verlangt
    Stueckzahl und Einstandskurs groesser null (beide ``gt=0``). Eine
    Position mit Nullwerten waere kein schneller Weg, sondern ein
    falscher Depotwert. Der Knopf klappt deshalb zwei Felder auf.

    Args:
        hit: Treffer der Symbolsuche.

    Returns:
        Zeile aus Namensknopf, Aktionsknoepfen und dem zunaechst
        verborgenen Eingabebereich des Depots.
    """
    from dash import html

    return html.Div(
        [
            html.Div(
                [
                    hit_button(hit),
                    html.Div(
                        [
                            html.Button(
                                WATCH_LABEL,
                                id={"type": WATCH_TYPE, "ticker": hit.ticker},
                                n_clicks=0,
                                title=f"Add {hit.ticker} to the {DEFAULT_WATCHLIST!r} watchlist",
                                className="bt-hit-action",
                            ),
                            html.Button(
                                PORTFOLIO_LABEL,
                                id={"type": PORTFOLIO_TOGGLE_TYPE, "ticker": hit.ticker},
                                n_clicks=0,
                                title=f"Add {hit.ticker} to the {DEFAULT_PORTFOLIO!r} portfolio",
                                className="bt-hit-action",
                            ),
                        ],
                        className="bt-hit-actions",
                    ),
                ],
                className="bt-hit-row",
            ),
            portfolio_form(hit.ticker),
        ],
        className="bt-hit-entry",
    )


def portfolio_form(ticker: str) -> html.Div:
    """Baut die Eingabe fuer eine neue Depotposition.

    Zunaechst verborgen; der Portfolio-Knopf der Zeile klappt sie auf.
    Stueckzahl und Einstandskurs sind Pflicht, weil das Datenmodell
    beides groesser null verlangt - ein Depot mit Nullpositionen wuerde
    einen Wert ausweisen, den es nicht gibt. Das Kaufdatum ist der
    heutige Tag; wer rueckwirkend erfasst, aendert es auf der Seite WL.

    Die Waehrung ist die des KAUFS, nicht die des Depots. Vorbelegt ist
    Euro; wer an einer US-Boerse kauft, stellt auf USD um. Der Einstand
    wird dann zum Kurs des Kauftages umgerechnet, der Marktwert zum
    heutigen - siehe ``docs/DATA_SOURCES.md``.

    Args:
        ticker: Kanonisches Symbol des Treffers.

    Returns:
        Verborgener Container mit beiden Feldern und dem Bestaetigen-Knopf.
    """
    from dash import dcc, html

    return html.Div(
        [
            html.Div(
                [
                    html.Label("Shares", className="bt-field-label"),
                    dcc.Input(
                        id={"type": SHARES_INPUT_TYPE, "ticker": ticker},
                        type="number",
                        min=0,
                        step="any",
                        placeholder="25",
                        className="bt-field-input",
                    ),
                ],
                className="bt-field",
            ),
            html.Div(
                [
                    html.Label("Buy price", className="bt-field-label"),
                    dcc.Input(
                        id={"type": PRICE_INPUT_TYPE, "ticker": ticker},
                        type="number",
                        min=0,
                        step="any",
                        placeholder="142.50",
                        className="bt-field-input",
                    ),
                ],
                className="bt-field",
            ),
            html.Div(
                [
                    html.Label("Currency", className="bt-field-label"),
                    dcc.Dropdown(
                        id={"type": CURRENCY_INPUT_TYPE, "ticker": ticker},
                        options=[{"label": c.value, "value": c.value} for c in Currency],
                        value=Currency.EUR.value,
                        clearable=False,
                        className="bt-dropdown bt-field-select",
                    ),
                ],
                className="bt-field",
            ),
            html.Button(
                "Add",
                id={"type": PORTFOLIO_SUBMIT_TYPE, "ticker": ticker},
                n_clicks=0,
                title=f"Add {ticker} to the portfolio",
                className="bt-hit-action",
            ),
        ],
        id={"type": PORTFOLIO_FORM_TYPE, "ticker": ticker},
        className="bt-hit-form",
        hidden=True,
    )


def hit_list(hits: list[SearchHit]) -> html.Div:
    """Baut die Trefferliste einer Suche.

    Args:
        hits: Treffer, bereits nach Relevanz sortiert.

    Returns:
        Abschnitt mit einer Zeile je Treffer, oder ein Hinweis wenn die
        Suche nichts gefunden hat.
    """
    from dash import html

    from btlite.ui.layouts import components as c

    if not hits:
        return c.section(
            "Search results",
            c.placeholder("No match. Try another name or the ticker symbol."),
        )
    count = f"{len(hits)} matches" if len(hits) != 1 else "1 match"
    return c.section(
        f"Search results - {count}",
        html.Div(
            "Click a company to open its profile.",
            className="bt-notice",
        ),
        html.Div([hit_row(hit) for hit in hits], className="bt-hits"),
        html.Div(id=FEEDBACK_OUTPUT),
    )


def hit_href(ticker: str) -> str:
    """Baut die Adresse des Steckbriefs zu einem Treffer.

    Nutzt denselben Adressbau wie die Kommandozeile, damit Suche und
    Kommando dieselben Adressen erzeugen.

    Args:
        ticker: Kanonisches Symbol.

    Returns:
        Pfad mit Query-String, etwa ``"/des?t=RKLB"``.
    """
    return analysis_href(TARGET_MNEMONIC, ticker)


def analysis_href(mnemonic: str, ticker: str) -> str:
    """Baut die Route einer gewaehlten Analyse.

    Nutzt denselben Adressbau wie die Kommandozeile, damit Suche und
    Kommando dieselben Adressen erzeugen.

    Args:
        mnemonic: Kanonisches Mnemonic, etwa ``"FA"``.
        ticker: Kanonisches Symbol.

    Returns:
        Pfad mit Query-String, etwa ``"/fa?t=NOW"``.
    """
    from btlite.ui.app import build_href

    return build_href(Command(mnemonic=mnemonic, tickers=(ticker,)))


def run_search(state: AppState, query: str | None) -> html.Div | None:
    """Fuehrt eine Suche aus und baut das Ergebnis.

    Der Abruf laeuft ueber den injizierten Provider (Architekturregel 2);
    dieses Modul macht selbst kein Netz. Eine leere Eingabe ist kein
    Fehler, sondern der Normalfall waehrend des Tippens.

    Args:
        state: UI-Zustand mit dem aktiven Provider.
        query: Roheingabe des Nutzers.

    Returns:
        Trefferliste, ein Fehlerbanner bei einem ``BTLiteError``, oder
        ``None`` wenn die Eingabe leer ist.
    """
    from btlite.core.errors import BTLiteError
    from btlite.ui.layouts import components as c

    if query is None or not query.strip():
        return None

    searcher = getattr(state.provider, "search", None)
    if searcher is None:
        return c.section(
            "Search results",
            c.error_banner("The active data source cannot search by name."),
        )

    try:
        hits: list[SearchHit] = searcher(query, limit=MAX_HITS)
    except BTLiteError as exc:
        _LOG.warning("Symbolsuche %r: %s (%s)", query, exc.user_message, exc.detail)
        return c.section("Search results", c.error_banner(exc))
    return hit_list(hits)


def _looks_like_ticker(value: str) -> bool:
    """Prueft, ob eine Zeichenkette als Boersensymbol durchgehen kann.

    Zugelassen sind Buchstaben, Ziffern und die in echten Symbolen
    ueblichen Trenner - ``BRK.B`` und ``RDS-A`` muessen passieren. Ein
    Symbol ohne einen einzigen Buchstaben oder eine Ziffer nicht.

    Args:
        value: Rohes Symbol.

    Returns:
        True, wenn das Symbol plausibel ist.
    """
    candidate = value.strip()
    if not candidate or len(candidate) > 20:
        return False
    if not any(ch.isalnum() for ch in candidate):
        return False
    return all(ch.isalnum() or ch in {".", "-", "^", "="} for ch in candidate)


def add_to_watchlist(state: AppState, ticker: str) -> html.Div:
    """Legt einen gefundenen Titel auf die Standard-Watchlist.

    Existiert die Liste noch nicht, wird sie angelegt. Steht der Titel
    bereits darauf, ist das kein Fehler, sondern die gewuenschte Lage - der
    Nutzer erfaehrt es als Hinweis und nicht als Fehlerbanner.

    Args:
        state: UI-Zustand mit dem Depot-Speicher.
        ticker: Kanonisches Symbol des Treffers.

    Returns:
        Bestaetigung, Hinweis, oder Fehlerbanner. Nie None: der Klick muss
        immer eine sichtbare Antwort haben.
    """
    from btlite.core.errors import BTLiteError
    from btlite.ui.layouts import components as c

    # normalize_ticker prueft nur auf Leere, nicht auf Plausibilitaet -
    # bewusst, weil echte Symbole Punkte und Bindestriche tragen (BRK.B).
    # Die Schaltflaeche traegt ihr Symbol aber aus der Trefferliste, also
    # muss hier nichts ankommen, was kein Symbol ist. Kommt es doch, ist
    # das ein Aufrufsfehler und kein Nutzerfehler.
    if not _looks_like_ticker(ticker):
        _LOG.warning("Watchlist-Eintrag mit unzulaessigem Symbol %r abgelehnt", ticker)
        return c.section(
            "Watchlist",
            c.error_banner(f"{ticker!r} is not a valid symbol."),
        )

    store = state.holdings
    try:
        if store.get_watchlist(DEFAULT_WATCHLIST) is None:
            store.create_watchlist(DEFAULT_WATCHLIST)
        watchlist = store.get_watchlist(DEFAULT_WATCHLIST)
        if watchlist is not None and ticker in watchlist.tickers:
            return c.section(
                "Watchlist",
                c.placeholder(f"{ticker} is already on the {DEFAULT_WATCHLIST!r} watchlist."),
            )
        store.add_to_watchlist(DEFAULT_WATCHLIST, ticker)
    except BTLiteError as exc:
        _LOG.warning("Watchlist-Eintrag %s: %s (%s)", ticker, exc.user_message, exc.detail)
        return c.section("Watchlist", c.error_banner(exc))

    return c.section(
        "Watchlist",
        c.placeholder(f"{ticker} added to the {DEFAULT_WATCHLIST!r} watchlist."),
    )


def add_to_portfolio(
    state: AppState,
    ticker: str,
    shares: float | None,
    buy_price: float | None,
    currency: str | None = None,
) -> html.Div:
    """Legt einen gefundenen Titel als Depotposition an.

    Anders als bei der Watchlist genuegt hier kein Klick: ``Holding``
    verlangt Stueckzahl und Einstandskurs beide groesser null. Fehlende
    oder nicht positive Werte werden deshalb abgewiesen statt auf null
    gesetzt - eine Position ohne Wert wuerde den Depotwert verfaelschen,
    und genau das soll die Anzeige nie tun.

    Das Kaufdatum ist der heutige Tag. Wer rueckwirkend erfasst, aendert
    es auf der Seite WL; ein drittes Feld an dieser Stelle waere ein
    Formular und kein Schnellzugriff mehr.

    Args:
        state: UI-Zustand mit dem Depot-Speicher.
        ticker: Kanonisches Symbol des Treffers.
        shares: Eingegebene Stueckzahl. ``None`` heisst leeres Feld.
        buy_price: Eingegebener Einstandskurs je Stueck, in ``currency``.
            ``None`` heisst leeres Feld.
        currency: ISO-4217-Code der Kaufwaehrung, etwa ``"EUR"``. ``None``
            nimmt die Berichtswaehrung des Depots. Ein unbekannter Code
            wird abgewiesen statt still durch den Default ersetzt - sonst
            stuende ein Euro-Kauf als Dollar-Kauf im Depot.

    Returns:
        Bestaetigung oder Fehlerbanner. Nie None: der Klick muss immer
        eine sichtbare Antwort haben.
    """
    from btlite.core.clock import utc_now
    from btlite.core.errors import BTLiteError
    from btlite.ui.layouts import components as c

    if not _looks_like_ticker(ticker):
        _LOG.warning("Depot-Eintrag mit unzulaessigem Symbol %r abgelehnt", ticker)
        return c.section("Portfolio", c.error_banner(f"{ticker!r} is not a valid symbol."))

    if shares is None or buy_price is None:
        return c.section(
            "Portfolio",
            c.placeholder("Enter both shares and buy price before adding to the portfolio."),
        )
    if shares <= 0 or buy_price <= 0:
        return c.section(
            "Portfolio",
            c.error_banner("Shares and buy price must both be greater than zero."),
        )

    bought_in: Currency | None = None
    if currency:
        try:
            bought_in = Currency(currency)
        except ValueError:
            _LOG.warning("Depot-Eintrag mit unbekannter Waehrung %r abgelehnt", currency)
            return c.section("Portfolio", c.error_banner(f"{currency!r} is not a known currency."))

    store = state.holdings
    try:
        if store.get_portfolio(DEFAULT_PORTFOLIO) is None:
            store.create_portfolio(DEFAULT_PORTFOLIO)
        store.add_holding(
            DEFAULT_PORTFOLIO,
            ticker,
            shares=shares,
            buy_price=buy_price,
            buy_date=utc_now().date(),
            currency=bought_in,
        )
    except BTLiteError as exc:
        _LOG.warning("Depot-Eintrag %s: %s (%s)", ticker, exc.user_message, exc.detail)
        return c.section("Portfolio", c.error_banner(exc))

    unit = bought_in.value if bought_in is not None else ""
    priced = f"{buy_price:g} {unit}".strip()
    return c.section(
        "Portfolio",
        c.placeholder(
            f"{ticker} added to the {DEFAULT_PORTFOLIO!r} portfolio: {shares:g} shares at {priced}."
        ),
    )
