"""Dash-Shell von BT-Lite.

Architekturregel 4: Dieses Paket enthaelt **keine** Geschaeftslogik.
Callbacks verdrahten ausschliesslich Analytics und Viz.

Routing laeuft ueber ``dcc.Location`` und eine eigene Tabelle statt ueber
``dash.register_page``: die Seiten brauchen Ticker und Argumente aus der
Kommandozeile, und die stehen im Query-String, nicht im Pfad. Eine
Registrierung ueber das Pages-Feature wuerde diese Parameter nicht
mittragen, ohne fuer jede Kombination eine eigene Route anzulegen.

Jeder Callback faengt ``BTLiteError`` und rendert ``user_message`` in ein
Fehlerbanner. Der technische ``detail`` geht nur ins Log.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import parse_qsl, urlencode

from btlite.core.logging import get_logger
from btlite.ui import search_bar
from btlite.ui.command_bar import (
    MNEMONICS,
    NAV_LABELS,
    TICKERLESS_MNEMONICS,
    Command,
    ParseError,
    parse_command,
)

if TYPE_CHECKING:
    from dash import Dash, html

    from btlite.config import Settings
    from btlite.ui.state import AppState

_LOG = get_logger(__name__)

#: Im Footer und in der README verpflichtender Hinweis.
DISCLAIMER = (
    "BT-Lite is a tool for your own research and does not provide "
    "financial advice. Price data is delayed, not real time. "
    "No broker connection, no order execution."
)

#: Startseite, wenn die Adresse keine Route nennt.
DEFAULT_PATH: Final[str] = "/des"

#: Query-Parameter, ueber die Ticker und Argumente transportiert werden.
TICKER_PARAM: Final[str] = "t"
ARGS_PARAM: Final[str] = "a"

#: Typ der Navigationslinks. Sie tragen eine Wildcard-ID, weil ihre
#: Adressen dem gewaehlten Titel folgen muessen: wer "rocket lab" sucht
#: und dann auf "Bilanz" klickt, will die Bilanz von RKLB sehen und nicht
#: die der Vorbelegung. Vorher waren die Links feste Pfade ohne Ticker.
NAV_LINK_TYPE: Final[str] = "nav-link"


def build_href(command: Command) -> str:
    """Setzt ein Kommando in eine Adresse um.

    Args:
        command: Geparstes Kommando.

    Returns:
        Pfad mit Query-String, etwa ``"/fa?t=AAPL,MSFT"``.
    """
    query: list[tuple[str, str]] = []
    if command.tickers:
        query.append((TICKER_PARAM, ",".join(command.tickers)))
    if command.args:
        query.append((ARGS_PARAM, " ".join(command.args)))
    suffix = f"?{urlencode(query)}" if query else ""
    return f"{command.path}{suffix}"


def parse_href(pathname: str | None, search: str | None) -> tuple[str, list[str], tuple[str, ...]]:
    """Liest Mnemonic, Ticker und Argumente aus der Adresse.

    Args:
        pathname: Pfad der aktuellen Adresse.
        search: Query-String inklusive fuehrendem ``?``.

    Returns:
        Tripel aus Mnemonic, Symbolliste und Argumenten. Ein unbekannter
        Pfad faellt auf :data:`DEFAULT_PATH` zurueck.
    """
    from btlite.ui.command_bar import split_tickers
    from btlite.ui.pages import ROUTES

    path = (pathname or DEFAULT_PATH).rstrip("/").lower() or DEFAULT_PATH
    mnemonic = ROUTES.get(path, ROUTES[DEFAULT_PATH])

    params = dict(parse_qsl((search or "").lstrip("?")))
    tickers = split_tickers(params.get(TICKER_PARAM, ""))
    raw_args = params.get(ARGS_PARAM, "")
    args = tuple(a for a in raw_args.split() if a)
    return mnemonic, tickers, args


def render_page(
    state: AppState,
    mnemonic: str,
    tickers: list[str],
    args: tuple[str, ...],
) -> html.Div:
    """Ruft die Seitenfunktion auf und faengt jeden Fehler ab.

    Args:
        state: UI-Zustand.
        mnemonic: Kanonisches Mnemonic.
        tickers: Symbole der Anfrage.
        args: Argumente hinter dem Mnemonic.

    Returns:
        Seiteninhalt, oder ein Fehlerbanner statt einer leeren Seite.
    """
    from dash import html

    from btlite.core.errors import BTLiteError
    from btlite.ui.layouts import components as c
    from btlite.ui.pages import PAGES

    page = PAGES.get(mnemonic)
    if page is None:
        return html.Div([c.error_banner(f"Unknown page {mnemonic!r}.")])

    try:
        return page(state, tickers, args)
    except BTLiteError as exc:
        _LOG.warning("Seite %s: %s (%s)", mnemonic, exc.user_message, exc.detail)
        return html.Div([c.error_banner(exc)])
    except Exception as exc:
        # Bewusst breit: eine Seite darf unter keinen Umstaenden leer
        # bleiben. Der technische Kontext geht ins Log, nicht ins Banner.
        _LOG.exception("Seite %s ist unerwartet gescheitert", mnemonic)
        return html.Div(
            [
                c.error_banner("The page could not be built. Details are in the log."),
                html.Div(type(exc).__name__, className="bt-notice"),
            ]
        )


def parse_buy_date(raw: str | None) -> date | str:
    """Liest ein Kaufdatum aus dem Eingabefeld.

    Ein unparsbares Datum ist ein **Nutzerfehler**, kein Absturz und auch
    kein stilles Heute-Datum: wer sich vertippt, bekaeme sonst einen Kauf
    mit falschem Datum ins Depot geschrieben und merkte es nie.

    ``date.fromisoformat`` nimmt ausser ``YYYY-MM-DD`` noch weitere
    ISO-8601-Formen an, etwa ``20240115`` und ``2024-W03-1``. Das ist
    hier gewollt: alle sind eindeutig und ergeben dasselbe Datum. Abgewiesen
    werden gerade die mehrdeutigen Schreibweisen wie ``15.01.2024``, bei
    denen Tag und Monat vertauscht sein koennten.

    Args:
        raw: Roher Feldinhalt, erwartet wird ``"YYYY-MM-DD"``.

    Returns:
        Das Datum, oder eine englische Fehlermeldung fuer das Feedback-Div.
    """
    text = (raw or "").strip()
    if not text:
        return "Enter a buy date as YYYY-MM-DD."
    try:
        return date.fromisoformat(text)
    except ValueError:
        return f"{text!r} is not a valid date. Use YYYY-MM-DD."


def parse_amount(value: float | None, label: str) -> float | str:
    """Prueft eine Zahleneingabe auf den Modellvertrag ``gt=0``.

    ``dcc.Input`` mit ``type="number"`` liefert ``None``, wenn das Feld
    leer ist. Weder ``None`` noch ``0`` duerfen still zu einem Wert werden -
    eine Position ohne Stueckzahl verfaelschte den Depotwert.

    Args:
        value: Roher Feldwert.
        label: Englische Bezeichnung des Feldes fuer die Meldung.

    Returns:
        Der Wert, oder eine englische Fehlermeldung fuer das Feedback-Div.
    """
    if value is None:
        return f"Enter {label}."
    if value <= 0:
        return f"{label.capitalize()} must be greater than zero."
    return float(value)


def header_notice(state: AppState) -> str:
    """Baut den Quellenhinweis der Kopfzeile.

    Args:
        state: UI-Zustand.

    Returns:
        Hinweistext. Weist eingefrorene Daten als solche aus.
    """
    if state.is_offline:
        return f"Source: fixtures - {state.offline_reason}"
    return (
        f"Source: yfinance - approx. "
        f"{state.settings.data.quote_delay_notice_min} min delay, not real time"
    )


def create_app(settings: Settings, state: AppState | None = None) -> Dash:
    """Baut die Dash-Anwendung mit allen Callbacks.

    Args:
        settings: Prozessweite Konfiguration.
        state: Vorbereiteter UI-Zustand. ``None`` baut ihn aus den
            Settings - so laesst sich in Tests ein Fixture-Zustand
            einsetzen.

    Returns:
        Konfigurierte, noch nicht gestartete Dash-App.
    """
    import dash_bootstrap_components as dbc
    from dash import ALL, MATCH, Dash, Input, Output, State, ctx, dcc, html, no_update
    from dash.exceptions import MissingCallbackContextException

    from btlite.core.errors import BTLiteError
    from btlite.ui.layouts import components as c
    from btlite.ui.pages import des, eqs
    from btlite.ui.pages import wl as wl_page
    from btlite.ui.state import get_state, set_state

    if state is not None:
        set_state(state)
    app_state = get_state(settings)

    app = Dash(
        __name__,
        title="BT-Lite",
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
    )

    app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            dcc.Interval(
                id="quote-refresh",
                interval=settings.ui.refresh_interval_ms,
                n_intervals=0,
            ),
            html.Header(
                [
                    html.Span("BT-LITE", className="bt-brand"),
                    dcc.Input(
                        id="command-bar",
                        type="text",
                        placeholder="AAPL FA GO",
                        debounce=True,
                        autoFocus=True,
                        n_submit=0,
                        className="bt-command",
                    ),
                    search_bar.search_field(),
                    html.Span(
                        header_notice(app_state),
                        id="data-source-notice",
                        className="bt-notice",
                    ),
                ],
                className="bt-header",
            ),
            html.Div(id="command-error", className="bt-command-error"),
            search_bar.result_container(),
            html.Nav(
                [
                    dcc.Link(
                        [
                            html.Span(NAV_LABELS[key], className="bt-navlink-name"),
                            html.Span(key, className="bt-navlink-code"),
                        ],
                        # Der Ticker haengt der Adresse an, sobald einer
                        # gewaehlt ist - siehe _on_nav_tickers. Ohne das
                        # sprang jeder Klick zurueck auf die Vorbelegung.
                        id={"type": NAV_LINK_TYPE, "mnemonic": key},
                        href=f"/{key.lower()}",
                        title=label,
                        className="bt-navlink",
                    )
                    for key, label in MNEMONICS.items()
                ],
                className="bt-nav",
            ),
            html.Main(id="page-content", className="bt-main"),
            html.Footer(DISCLAIMER, className="bt-footer"),
        ],
        className="bt-root",
    )

    @app.callback(
        Output("url", "pathname"),
        Output("url", "search"),
        Output("command-error", "children"),
        Output("command-bar", "value"),
        Input("command-bar", "n_submit"),
        Input("command-bar", "value"),
        prevent_initial_call=True,
    )
    def _on_command(n_submit: int, raw: str | None) -> tuple[Any, Any, Any, Any]:
        """Wertet die Kommandozeile aus und navigiert.

        Args:
            n_submit: Anzahl der Enter-Ausloesungen.
            raw: Rohe Eingabezeile.

        Returns:
            Pfad, Query-String, Fehlerbanner und der neue Feldinhalt.
        """
        del n_submit
        if not raw or not raw.strip():
            return no_update, no_update, None, no_update

        parsed = parse_command(raw)
        if isinstance(parsed, ParseError):
            return no_update, no_update, c.error_banner(parsed.user_message), no_update

        href = build_href(parsed)
        path, _, query = href.partition("?")
        return path, f"?{query}" if query else "", None, ""

    @app.callback(
        Output(search_bar.RESULT_OUTPUT, "children"),
        Input(search_bar.SEARCH_INPUT, "value"),
        prevent_initial_call=True,
    )
    def _on_search(query: str | None) -> Any:
        """Sucht Symbole nach Namen und zeigt die Treffer.

        Args:
            query: Roheingabe des Suchfelds.

        Returns:
            Trefferliste, Fehlerbanner, oder None bei leerer Eingabe.
        """
        return search_bar.run_search(app_state, query)

    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Output("url", "search", allow_duplicate=True),
        Output(search_bar.RESULT_OUTPUT, "children", allow_duplicate=True),
        Output(search_bar.SEARCH_INPUT, "value"),
        Input({"type": search_bar.HIT_TYPE, "ticker": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _on_search_hit(n_clicks: list[int | None]) -> tuple[Any, Any, Any, Any]:
        """Oeffnet den Steckbrief des angeklickten Treffers.

        Der Klick auf einen Firmennamen navigiert unmittelbar. Eine
        fruehere Fassung zeigte erst eine Auswahl der Auswertungen; der
        Nutzer erwartete die Seite der Firma und bekam eine zweite Liste.

        Trefferliste und Suchfeld werden dabei geleert: die Antwort auf
        die Suche ist die geoeffnete Seite, eine stehenbleibende Liste
        verdeckte sie nur.

        Dash wertet einen Callback mit ``ALL``-Eingabe als
        Wildcard-Callback. Die Rueckgabe bleibt hier ein Tupel, weil
        mehrere Outputs bedient werden - die Listenform gilt nur fuer den
        Fall eines einzelnen Outputs.

        Args:
            n_clicks: Klickzaehler aller Treffer-Schaltflaechen.

        Returns:
            Pfad, Query-String, geleerte Trefferliste und geleertes
            Suchfeld.
        """
        if not any(n_clicks or ()):
            return no_update, no_update, no_update, no_update
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return no_update, no_update, no_update, no_update
        ticker = str(triggered.get("ticker", ""))
        if not ticker:
            return no_update, no_update, no_update, no_update

        href = search_bar.hit_href(ticker)
        path, _, query = href.partition("?")
        return path, f"?{query}" if query else "", None, ""

    @app.callback(
        Output(search_bar.FEEDBACK_OUTPUT, "children"),
        Output(des.WATCH_FEEDBACK, "children"),
        Input({"type": search_bar.WATCH_TYPE, "ticker": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _on_search_watch(n_clicks: list[int | None]) -> tuple[Any, Any]:
        """Legt den gewaehlten Titel auf die Watchlist.

        Der Knopf steht an zwei Orten: an der Trefferzeile der Suche und
        auf dem Steckbrief. Beide bekommen ihre eigene Rueckmeldung, damit
        sie dort erscheint, wo geklickt wurde - welcher der beiden
        Container gerade existiert, entscheidet die angezeigte Seite.

        Bewusst **kein** Sprung mehr auf die Watchlist-Seite: wer aus der
        Trefferliste heraus merkt, will meist mehrere Titel ablegen. Ein
        Sprung haette die Liste nach dem ersten Klick verworfen.

        Args:
            n_clicks: Klickzaehler aller Watchlist-Schaltflaechen.

        Returns:
            Die Rueckmeldung, einmal fuer den Suchbereich und einmal fuer
            den Steckbrief.
        """
        if not any(n_clicks or ()):
            return no_update, no_update
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return no_update, no_update
        ticker = str(triggered.get("ticker", ""))
        if not ticker:
            return no_update, no_update

        feedback = search_bar.add_to_watchlist(app_state, ticker)
        return feedback, feedback

    @app.callback(
        Output({"type": search_bar.PORTFOLIO_FORM_TYPE, "ticker": MATCH}, "hidden"),
        Input({"type": search_bar.PORTFOLIO_TOGGLE_TYPE, "ticker": MATCH}, "n_clicks"),
        State({"type": search_bar.PORTFOLIO_FORM_TYPE, "ticker": MATCH}, "hidden"),
        prevent_initial_call=True,
    )
    def _on_portfolio_toggle(n_clicks: int | None, hidden: bool) -> Any:
        """Klappt die Depot-Eingabe eines Treffers auf und wieder zu.

        ``MATCH`` statt ``ALL``: jede Zeile schaltet nur ihr eigenes
        Formular, sonst oeffneten sich alle Treffer gleichzeitig.

        Args:
            n_clicks: Klickzaehler des Knopfes dieser Zeile.
            hidden: Aktueller Zustand des Formulars dieser Zeile.

        Returns:
            Der umgekehrte Zustand.
        """
        if not n_clicks:
            return no_update
        return not hidden

    @app.callback(
        Output(search_bar.FEEDBACK_OUTPUT, "children", allow_duplicate=True),
        Input({"type": search_bar.PORTFOLIO_SUBMIT_TYPE, "ticker": ALL}, "n_clicks"),
        State({"type": search_bar.SHARES_INPUT_TYPE, "ticker": ALL}, "value"),
        State({"type": search_bar.PRICE_INPUT_TYPE, "ticker": ALL}, "value"),
        State({"type": search_bar.CURRENCY_INPUT_TYPE, "ticker": ALL}, "value"),
        State({"type": search_bar.SHARES_INPUT_TYPE, "ticker": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _on_portfolio_add(
        n_clicks: list[int | None],
        shares: list[float | None],
        prices: list[float | None],
        currencies: list[str | None],
        ids: list[dict[str, str]],
    ) -> Any:
        """Legt die eingegebene Position im Depot an.

        Die Felder kommen als gleichlange Listen ueber ``ALL``. Welcher
        Eintrag zum geklickten Treffer gehoert, sagt die Liste der
        Feld-IDs - sich auf die Reihenfolge zu verlassen waere ein Fehler,
        sobald die Trefferzahl wechselt.

        Args:
            n_clicks: Klickzaehler aller Bestaetigen-Schaltflaechen.
            shares: Werte aller Stueckzahl-Felder.
            prices: Werte aller Kurs-Felder.
            currencies: Werte aller Waehrungsfelder.
            ids: IDs der Stueckzahl-Felder, zur Zuordnung ueber den Ticker.

        Returns:
            Die Rueckmeldung, oder ``no_update`` ohne Klick.
        """
        if not any(n_clicks or ()):
            return no_update
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return no_update
        ticker = str(triggered.get("ticker", ""))
        if not ticker:
            return no_update

        index = next(
            (i for i, element in enumerate(ids) if element.get("ticker") == ticker),
            None,
        )
        if index is None:
            _LOG.warning("Depot-Eingabe zu %s nicht gefunden", ticker)
            return no_update
        return search_bar.add_to_portfolio(
            app_state, ticker, shares[index], prices[index], currencies[index]
        )

    @app.callback(
        Output("page-content", "children"),
        Input("url", "pathname"),
        Input("url", "search"),
    )
    def _on_route(pathname: str | None, search: str | None) -> Any:
        """Baut die Seite zur aktuellen Adresse.

        Args:
            pathname: Pfad der Adresse.
            search: Query-String der Adresse.

        Returns:
            Seiteninhalt.
        """
        mnemonic, tickers, args = parse_href(pathname, search)
        return render_page(app_state, mnemonic, tickers, args)

    @app.callback(
        Output({"type": NAV_LINK_TYPE, "mnemonic": ALL}, "href"),
        Input("url", "pathname"),
        Input("url", "search"),
    )
    def _on_nav_tickers(pathname: str | None, search: str | None) -> list[str]:
        """Fuehrt die Navigationslinks am gewaehlten Titel nach.

        Der Nutzer sucht einen Titel, landet auf dem Steckbrief und klickt
        dann auf "Bilanz" - und sah die Bilanz der Vorbelegung, weil die
        Links feste Pfade ohne Ticker waren und jede Seite ohne ``?t=``
        auf ``default_tickers`` zurueckfaellt. Die Auswahl bleibt jetzt
        ueber die Navigation hinweg bestehen.

        Screening und Watchlist bekommen bewusst keinen Ticker: EQS
        screent ein Universum, WL liest den gespeicherten Bestand. Ein
        angehaengter Titel wuerde dort das Universum auf einen Wert
        verengen beziehungsweise wirkungslos verpuffen - siehe
        :data:`TICKERLESS_MNEMONICS`.

        Args:
            pathname: Pfad der aktuellen Adresse.
            search: Query-String der aktuellen Adresse.

        Returns:
            Adressen aller Navigationslinks, in der Reihenfolge der
            Wildcard-Eingaben.
        """
        _, tickers, _ = parse_href(pathname, search)
        return [
            build_href(Command(mnemonic=key, tickers=() if tickerless else tuple(tickers)))
            for key in MNEMONICS
            for tickerless in (key in TICKERLESS_MNEMONICS,)
        ]

    @app.callback(
        Output("gp-chart", "children"),
        Output("gp-resolution", "options"),
        Output("gp-resolution", "value"),
        Input("gp-period", "value"),
        Input("gp-chart-kind", "value"),
        Input("gp-overlays", "value"),
        State("gp-tickers", "data"),
        # Zuletzt, damit die bisherige Argumentreihenfolge gilt: Dash
        # uebergibt Inputs und States in der Reihenfolge der Deklaration.
        Input("gp-resolution", "value"),
        prevent_initial_call=True,
    )
    def _on_gp_controls(
        period: str | None,
        kind: str | None,
        overlays: list[str] | None,
        tickers: list[str] | None,
        resolution: str | None = None,
    ) -> tuple[Any, Any, Any]:
        """Zeichnet die Charts der Seite GP neu.

        Wechselt der Zeitraum, springt die Aufloesung auf die kleinste, die
        der Zeitraum hergibt - so gewuenscht. Eine von Hand gewaehlte
        groebere Aufloesung bleibt nur, solange der Zeitraum bleibt.
        Aufloesung ist zugleich Eingang und Ausgang dieses Callbacks; Dash
        erlaubt das innerhalb eines Callbacks, ohne ihn erneut auszuloesen.

        Args:
            period: Gewaehlter Zeitraum in yfinance-Notation.
            kind: Gewaehlte Chart-Art.
            overlays: Gewaehlte Overlay-Schluessel.
            tickers: Symbole der Seite.
            resolution: Gewaehlte Kerzen-Aufloesung.

        Returns:
            Chartbereich, Optionen und Wert der Aufloesungswahl.
        """
        from btlite.ui.pages import gp

        symbols = tickers or app_state.default_tickers
        chosen_period = period or gp.DEFAULT_PERIOD
        intraday = not app_state.is_offline
        try:
            triggered = ctx.triggered_id
        except MissingCallbackContextException:
            # Direkter Aufruf ohne Dash-Anfrage, etwa im Test.
            triggered = None
        wanted = gp.DEFAULT_RESOLUTION if triggered == "gp-period" else resolution
        interval = gp.effective_interval(chosen_period, wanted, intraday_available=intraday)
        options = gp.resolution_options(chosen_period, intraday_available=intraday)
        try:
            chart = gp.render_charts(
                app_state,
                symbols,
                chosen_period,
                kind=kind or "candlestick",
                overlays=tuple(overlays or ()),
                resolution=interval.value,
            )
        except BTLiteError as exc:
            chart = c.error_banner(exc)
        return chart, options, interval.value

    @app.callback(
        Output("fa-statements", "children"),
        Input("fa-period", "value"),
        State("fa-tickers", "data"),
        prevent_initial_call=True,
    )
    def _on_fa_period(period: str | None, tickers: list[str] | None) -> Any:
        """Schaltet die Periodenbasis der Seite FA um.

        Args:
            period: Gewaehlte Basis als Enum-Wert.
            tickers: Symbole der Seite.

        Returns:
            Neuer Inhalt der Seite FA.
        """
        from btlite.models.common import Period
        from btlite.ui.pages import fa

        symbols = tickers or app_state.default_tickers
        try:
            basis = Period(period) if period else Period.ANNUAL
            return fa.render_body(app_state, symbols[0], symbols[1:], basis)
        except (BTLiteError, ValueError) as exc:
            return c.error_banner(exc if isinstance(exc, BTLiteError) else str(exc))

    @app.callback(
        Output("eqs-universe", "data"),
        Output("eqs-result", "children", allow_duplicate=True),
        Input("eqs-index", "value"),
        State("eqs-preset", "value"),
        *(State(key, "value") for key in eqs.REFINEMENT_INPUTS),
        prevent_initial_call=True,
    )
    def _on_eqs_index(
        index_key: str | None, preset_key: str | None, *refine: object
    ) -> tuple[Any, Any]:
        """Laedt den Bestand des gewaehlten Index, ohne ihn zu screenen.

        Das Laden der Mitgliederliste dauert Sekunden, das Screening
        Minuten. Frueher loeste jeder Wechsel im Universums-Feld sofort
        den vollen Lauf aus - die Seite stand dann minutenlang auf
        "updating", ohne dass jemand darum gebeten hatte. Gerechnet wird
        nur noch auf Knopfdruck, siehe :func:`_on_eqs_run`.

        Args:
            index_key: Schluessel des Index, etwa ``"sp500"``.
            preset_key: Aktuell gewaehltes Preset. Bestimmt allein die
                Dauerabschaetzung im Hinweis.
            *refine: Werte der Zusatzfilter, siehe
                :data:`eqs.REFINEMENT_INPUTS`.

        Returns:
            Neues Universum und ein Hinweis, was der Lauf kosten wuerde.
            Bei einem Fehler bleibt das Universum unveraendert und nur das
            Banner erscheint.
        """
        if not index_key:
            return no_update, no_update
        try:
            constituents = app_state.indices.get_constituents(index_key)
            symbols = [member.ticker for member in constituents.members]
        except BTLiteError as exc:
            return no_update, c.error_banner(exc)
        if not symbols:
            return [], c.placeholder("The index returns no titles.")
        try:
            refinement = eqs.refinement_from_inputs(*refine)
        except BTLiteError as exc:
            return symbols, c.error_banner(exc)
        return symbols, eqs.initial_result(app_state, preset_key or "value", symbols, refinement)

    @app.callback(
        Output("eqs-result", "children", allow_duplicate=True),
        Output(eqs.JOB_STORE, "data"),
        Output(eqs.PROGRESS_TICK, "disabled"),
        Output(eqs.CANCEL_BUTTON, "hidden"),
        Input("eqs-run", "n_clicks"),
        State("eqs-preset", "value"),
        State("eqs-universe", "data"),
        *(State(key, "value") for key in eqs.REFINEMENT_INPUTS),
        prevent_initial_call=True,
    )
    def _on_eqs_run(
        n_clicks: int | None,
        preset_key: str | None,
        universe: list[str] | None,
        *refine: object,
    ) -> tuple[Any, Any, Any, Any]:
        """Startet den Screen in einem eigenen Thread.

        Bei einem grossen Universum rechnet die Seite nicht von allein -
        ein Klick im Navigationsmenue wuerde sonst minutenlang blockieren.

        Gerechnet wird im Hintergrund, nicht im Web-Thread: ein Screening
        ueber den S&P 500 dauert gemessen fast eine halbe Stunde. Der
        Nutzer sah in dieser Zeit nur einen Ladebalken und hielt die Seite
        fuer haengengeblieben.

        Args:
            n_clicks: Klickzaehler der Schaltflaeche.
            preset_key: Gewaehltes Preset.
            universe: Aktuelles Universum.
            *refine: Werte der Zusatzfilter, siehe
                :data:`eqs.REFINEMENT_INPUTS`.

        Returns:
            Anzeige, Job-Kennung, Zustand des Abfrage-Intervalls und des
            Abbrechen-Knopfs.
        """
        from btlite.ui.jobs import REGISTRY

        if not n_clicks:
            return no_update, no_update, no_update, no_update
        if not universe:
            return (
                c.placeholder("No universe: without titles there is nothing to filter."),
                None,
                True,
                True,
            )

        preset = preset_key or "value"
        symbols = list(universe)
        try:
            # Vor dem Start pruefen: ein ungueltiger RSI-Wert soll sofort
            # auffallen, nicht nach zwanzig Minuten Rechnen.
            refinement = eqs.refinement_from_inputs(*refine)
        except BTLiteError as exc:
            return c.error_banner(exc), None, True, True

        def work(job: Any) -> Any:
            """Rechnet das Screening und meldet den Stand.

            Fortschritt und Abbruch gehen ueber GETRENNTE Kanaele. Der
            erste Entwurf warf den Abbruch als Ausnahme aus dem
            Fortschritts-Rueckruf - und der schluckt Ausnahmen
            absichtlich, damit eine kaputte Anzeige den Lauf nicht
            abbricht. Gemessene Folge: ein Klick auf Cancel nach drei
            Sekunden liess den Lauf sechzig Sekunden weiterrechnen und
            legte nebenbei die Anzeige still.
            """
            job.progress(0, len(symbols))
            return eqs.render_result(
                app_state,
                preset,
                symbols,
                job.progress,
                lambda: bool(job.cancelled),
                refinement,
            )

        # Ein noch laufender frueherer Lauf wuerde die Drosselung der Quelle
        # mit diesem teilen und ihn auf halbes Tempo bremsen.
        REGISTRY.cancel_running()
        job = REGISTRY.start(work)
        return eqs.progress_view(0, len(symbols), preset), job.job_id, False, False

    @app.callback(
        Output("eqs-result", "children", allow_duplicate=True),
        Output(eqs.PROGRESS_TICK, "disabled", allow_duplicate=True),
        Output(eqs.CANCEL_BUTTON, "hidden", allow_duplicate=True),
        Output(eqs.JOB_STORE, "data", allow_duplicate=True),
        Input(eqs.PROGRESS_TICK, "n_intervals"),
        State(eqs.JOB_STORE, "data"),
        State("eqs-preset", "value"),
        State(eqs.UNIVERSE_STORE, "data"),
        prevent_initial_call=True,
    )
    def _on_eqs_progress(
        _ticks: int | None,
        job_id: str | None,
        preset_key: str | None,
        universe: list[str] | None = None,
    ) -> tuple[Any, Any, Any, Any]:
        """Fragt den Stand des laufenden Screenings ab.

        Args:
            _ticks: Zaehler des Intervalls, nur Ausloeser.
            job_id: Kennung der laufenden Arbeit.
            preset_key: Gewaehltes Preset, fuer die Restzeit-Schaetzung.
            universe: Aktuelles Universum. Seine Titelzahl trennt den
                ersten Schritt eines Laufs vom zweiten, dem Kursabruf fuer
                die Treffer eines fundamentalen Presets.

        Returns:
            Anzeige, Zustand des Intervalls und des Abbrechen-Knopfs sowie
            die Job-Kennung. Solange der Lauf dauert, wechselt nur die
            Anzeige.
        """
        from btlite.ui.jobs import REGISTRY

        job = REGISTRY.get(job_id)
        if job is None:
            # Kennung unbekannt - etwa nach einem Neustart des Servers.
            # Das Intervall abstellen, sonst fragt es endlos ins Leere.
            return no_update, True, True, None

        done, total, finished, _cancelled = job.snapshot()
        if not finished:
            titles = len(universe) if universe else None
            view = eqs.progress_view(
                done, total, preset_key or "value", titles, job.seconds_per_unit()
            )
            return view, False, False, no_update

        REGISTRY.discard(job_id)
        if job.error is not None:
            return c.error_banner(job.error), True, True, None
        # Ein Abbruch ist kein Fehler: gezeigt wird, was bis dahin
        # zusammenkam. Der Nutzer hat schliesslich selbst gestoppt.
        return job.result, True, True, None

    @app.callback(
        Output(eqs.CANCEL_BUTTON, "hidden", allow_duplicate=True),
        Input(eqs.CANCEL_BUTTON, "n_clicks"),
        State(eqs.JOB_STORE, "data"),
        prevent_initial_call=True,
    )
    def _on_eqs_cancel(n_clicks: int | None, job_id: str | None) -> Any:
        """Bittet das laufende Screening aufzuhoeren.

        Der Knopf verschwindet sofort, das Ergebnis holt die
        Fortschrittsabfrage ab - der Lauf endet erst am naechsten
        Haltepunkt, nicht mitten im Schreiben.

        Args:
            n_clicks: Klickzaehler.
            job_id: Kennung der laufenden Arbeit.

        Returns:
            Neuer Zustand des Knopfs.
        """
        from btlite.ui.jobs import REGISTRY

        if not n_clicks:
            return no_update
        job = REGISTRY.get(job_id)
        if job is not None:
            job.cancel()
        return True

    @app.callback(
        Output("eqs-result", "children"),
        Input("eqs-preset", "value"),
        State("eqs-universe", "data"),
        *(State(key, "value") for key in eqs.REFINEMENT_INPUTS),
        prevent_initial_call=True,
    )
    def _on_eqs_preset(preset_key: str | None, universe: list[str] | None, *refine: object) -> Any:
        """Uebernimmt das gewaehlte Preset, ohne sofort zu rechnen.

        Bei einem kleinen Universum kostet der Lauf nichts und startet
        direkt. Bei einem ganzen Index waeren es Minuten - dann erscheint
        stattdessen der Hinweis mit der Dauer, und der Nutzer entscheidet.

        Args:
            preset_key: Key des Presets.
            universe: Startuniversum.
            *refine: Werte der Zusatzfilter, siehe
                :data:`eqs.REFINEMENT_INPUTS`.

        Returns:
            Trefferliste bei kleinem Universum, sonst der Hinweis.
        """
        symbols = universe or app_state.default_tickers
        try:
            refinement = eqs.refinement_from_inputs(*refine)
        except BTLiteError as exc:
            return c.error_banner(exc)
        return eqs.initial_result(app_state, preset_key or "value", symbols, refinement)

    @app.callback(
        Output("eqs-download", "data"),
        Input("eqs-export", "n_clicks"),
        State("eqs-preset", "value"),
        State("eqs-universe", "data"),
        *(State(key, "value") for key in eqs.REFINEMENT_INPUTS),
        prevent_initial_call=True,
    )
    def _on_eqs_export(
        n_clicks: int | None,
        preset_key: str | None,
        universe: list[str] | None,
        *refine: object,
    ) -> Any:
        """Exportiert die Trefferliste als CSV.

        Args:
            n_clicks: Anzahl der Klicks auf den Export-Knopf.
            preset_key: Key des Presets.
            universe: Startuniversum.
            *refine: Werte der Zusatzfilter. Der Export muss dieselbe
                Liste liefern, die auf dem Schirm steht.

        Returns:
            Download-Nutzlast, oder ``no_update`` bei einem Fehler.
        """
        if not n_clicks:
            return no_update
        symbols = universe or app_state.default_tickers
        key = preset_key or "value"
        try:
            refinement = eqs.refinement_from_inputs(*refine)
            result = eqs.run_screen(app_state, key, symbols, refinement=refinement)
        except BTLiteError as exc:
            _LOG.warning("CSV-Export gescheitert: %s (%s)", exc.user_message, exc.detail)
            return no_update
        # ``dcc.send_data_frame`` ist untypisiert. Die Nutzlast ist ein
        # schlichtes Dictionary; sie hier direkt zu bauen erspart den
        # untypisierten Aufruf und macht das Format sichtbar.
        return {
            "content": result.data.to_csv(index=True),
            "filename": f"btlite_screen_{key}.csv",
            "type": "text/csv",
            "base64": False,
        }

    @app.callback(
        Output("bt-result", "children"),
        Input("bt-strategy", "value"),
        Input("bt-params", "value"),
        State("bt-tickers", "data"),
        prevent_initial_call=True,
    )
    def _on_bt_controls(
        strategy_key: str | None,
        params_text: str | None,
        tickers: list[str] | None,
    ) -> Any:
        """Rechnet den Backtest mit der gewaehlten Strategie neu.

        Args:
            strategy_key: Key der Strategie.
            params_text: Parameter als Eingabetext.
            tickers: Zu handelnde Symbole.

        Returns:
            Neues Ergebnis der Seite BT.
        """
        from btlite.ui.pages import bt

        symbols = tickers or app_state.default_tickers
        key = strategy_key or "sma_cross"
        try:
            defaults = app_state.strategy_registry().get(key).default_params
            params = bt.parse_params(params_text or "", dict(defaults))
            return bt.render_result(app_state, symbols, key, params)
        except BTLiteError as exc:
            return c.error_banner(exc)

    @app.callback(
        Output("port-result", "children"),
        Input("port-objective", "value"),
        State("port-tickers", "data"),
        prevent_initial_call=True,
    )
    def _on_port_objective(objective: str | None, tickers: list[str] | None) -> Any:
        """Optimiert das Portfolio mit der gewaehlten Zielfunktion.

        Args:
            objective: Zielfunktion als Enum-Wert.
            tickers: Anlageuniversum.

        Returns:
            Neues Ergebnis der Seite PORT.
        """
        from btlite.models.portfolio import OptimizationObjective
        from btlite.ui.pages import port

        symbols = tickers or app_state.default_tickers
        try:
            goal = (
                OptimizationObjective(objective) if objective else OptimizationObjective.MAX_SHARPE
            )
            return port.render_result(app_state, symbols, goal)
        except (BTLiteError, ValueError) as exc:
            return c.error_banner(exc if isinstance(exc, BTLiteError) else str(exc))

    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Output("url", "search", allow_duplicate=True),
        Output("command-bar", "value", allow_duplicate=True),
        Input("port-portfolio-picker", "value"),
        prevent_initial_call=True,
    )
    def _on_port_portfolio(name: str | None) -> tuple[Any, Any, Any]:
        """Uebernimmt die Titel eines gespeicherten Depots als Universum.

        Die Auswahl fuehrt ueber die **Adresse**, nicht ueber einen
        stillen Zustand: danach steht das Universum in der Kommandozeile
        und in der URL. Der Nutzer sieht, womit gerechnet wird, kann
        einzelne Titel streichen und die Ansicht als Lesezeichen sichern.

        Gewichtet wird nach der Optimierung, nicht nach dem Bestand -
        Stueckzahlen und Einstandskurse bleiben hier aussen vor. Was das
        Depot heute wert ist, steht auf der Seite WL.

        Args:
            name: Gewaehltes Depot, oder None beim Leeren des Feldes.

        Returns:
            Pfad, Query und neuer Inhalt der Kommandozeile.
        """
        from btlite.ui.pages import port

        if not name:
            return no_update, no_update, no_update

        symbols = port.tickers_of(app_state, name)
        if len(symbols) < port.MIN_TICKERS:
            # Sollte das Dropdown gar nicht erst anbieten; ein Depot kann
            # aber zwischen Seitenaufbau und Klick geleert worden sein.
            return no_update, no_update, no_update

        command = Command(mnemonic="PORT", tickers=tuple(symbols))
        href = build_href(command)
        path, _, query = href.partition("?")
        return path, f"?{query}" if query else "", f"{','.join(symbols)} PORT"

    @app.callback(
        Output("composition-result", "children"),
        Input("composition-run", "n_clicks"),
        State("composition-portfolio", "value"),
        State("composition-compare", "value"),
        State("composition-dimension", "value"),
        State("composition-upload", "contents"),
        prevent_initial_call=True,
    )
    def _on_composition(
        clicks: int,
        name: str | None,
        compare: str | None,
        dimension: str,
        contents: str | None,
    ) -> Any:
        """Analyze actual holdings on explicit request, including ETF constituents."""
        from btlite.ui.pages.composition import render

        return render(app_state, name, compare, dimension, contents)

    # ------------------------------------------------------- Seite WL ----
    #
    # Zum Zuschnitt dieser Callbacks: jeder von ihnen wendet **genau eine**
    # Aenderung an und rendert danach neu. Das ist keine Stilfrage, sondern
    # die Antwort auf die wandernde Adresse: der Index einer Haltung ist
    # ihre Position in ``Portfolio.holdings``, und nach einem
    # ``remove_holding(3)`` ist die vormalige 4 die neue 3. Ein Callback,
    # der mehrere Zeilen in einem Durchgang schriebe, traefe ab der ersten
    # Loeschung reihenweise die falschen. Deshalb adressiert jeder
    # Schreibvorgang nur die geklickte Zeile, und die frisch gerenderte
    # Ausgabe vergibt die Indizes danach neu.
    #
    # ``ALL`` statt ``MATCH`` bei Save und Remove: der Output ist der eine
    # gemeinsame Container PORTFOLIO_OUTPUT, nicht ein Element derselben
    # Zeile. ``MATCH`` verlangt einen mitwandernden Output - hier wird aber
    # die ganze Liste neu gebaut. Welche Zeile gemeint war, sagt
    # ``ctx.triggered_id``.

    def _sort_of(store: dict[str, Any] | None) -> tuple[str | None, bool]:
        """Liest Spalte und Richtung aus einem Sortier-Speicher.

        Args:
            store: Inhalt des ``dcc.Store``, oder None beim ersten Aufbau.

        Returns:
            Paar aus Spaltenname und Richtung. Ohne Speicher gilt die
            Reihenfolge, die die Seite selbst liefert.
        """
        if not isinstance(store, dict):
            return None, False
        column = store.get("column")
        return (column if isinstance(column, str) else None), bool(store.get("desc"))

    def _wl_portfolio_view(name: str | None, sort: dict[str, Any] | None = None) -> Any:
        """Rendert den Depotbereich neu.

        Args:
            name: Name des anzuzeigenden Depots, oder None.
            sort: Sortierung aus dem Speicher. Wird durchgereicht, damit
                die Reihenfolge eine Aenderung an der Seite ueberlebt -
                sonst sprang die Tabelle nach jedem Kauf zurueck.

        Returns:
            Inhalt fuer PORTFOLIO_OUTPUT.
        """
        column, desc = _sort_of(sort)
        return wl_page.render_portfolio(app_state, name, sort_by=column, sort_desc=desc)

    def _wl_watchlist_view(name: str | None, sort: dict[str, Any] | None = None) -> Any:
        """Rendert den Watchlist-Bereich neu.

        Args:
            name: Name der anzuzeigenden Watchlist, oder None.
            sort: Sortierung aus dem Speicher.

        Returns:
            Inhalt fuer WATCHLIST_OUTPUT.
        """
        column, desc = _sort_of(sort)
        return wl_page.render_watchlist(app_state, name, sort_by=column, sort_desc=desc)

    @app.callback(
        Output(wl_page.PORTFOLIO_OUTPUT, "children"),
        Input(wl_page.PORTFOLIO_SELECT, "value"),
        State(wl_page.POSITIONS_SORT, "data"),
        prevent_initial_call=True,
    )
    def _on_wl_portfolio_select(name: str | None, sort: dict[str, Any] | None = None) -> Any:
        """Zeigt das im Dropdown gewaehlte Depot.

        Args:
            name: Gewaehlter Depotname.
            sort: Aktuelle Sortierung der Tabelle.

        Returns:
            Neuer Inhalt des Depotbereichs.
        """
        return _wl_portfolio_view(name, sort)

    @app.callback(
        Output(wl_page.WATCHLIST_OUTPUT, "children"),
        Input(wl_page.WATCHLIST_SELECT, "value"),
        State(wl_page.WATCHLIST_SORT, "data"),
        prevent_initial_call=True,
    )
    def _on_wl_watchlist_select(name: str | None, sort: dict[str, Any] | None = None) -> Any:
        """Zeigt die im Dropdown gewaehlte Watchlist.

        Args:
            name: Gewaehlter Listenname.
            sort: Aktuelle Sortierung der Tabelle.

        Returns:
            Neuer Inhalt des Watchlist-Bereichs.
        """
        return _wl_watchlist_view(name, sort)

    @app.callback(
        Output(wl_page.PORTFOLIO_SELECT, "options"),
        Output(wl_page.PORTFOLIO_SELECT, "value"),
        Output(wl_page.PORTFOLIO_OUTPUT, "children", allow_duplicate=True),
        Output(wl_page.PORTFOLIO_FEEDBACK, "children"),
        Input(wl_page.PORTFOLIO_CREATE, "n_clicks"),
        Input(wl_page.PORTFOLIO_DELETE, "n_clicks"),
        State(wl_page.PORTFOLIO_NEW_NAME, "value"),
        State(wl_page.PORTFOLIO_SELECT, "value"),
        State(wl_page.POSITIONS_SORT, "data"),
        prevent_initial_call=True,
    )
    def _on_wl_portfolio_manage(
        create_clicks: int | None,
        delete_clicks: int | None,
        new_name: str | None,
        selected: str | None,
        sort: dict[str, Any] | None = None,
    ) -> tuple[Any, Any, Any, Any]:
        """Legt ein Depot an oder loescht das gewaehlte.

        Anlegen und Loeschen teilen sich einen Callback, weil beide
        dieselben vier Outputs bedienen; welcher Knopf gedrueckt wurde,
        sagt ``ctx.triggered_id``.

        Scheitert der Aufruf, bleibt der bisherige Inhalt stehen und nur
        das Feedback wechselt. Ein Nutzer, dem nach einem Tippfehler das
        Depot verschwindet, haelt es fuer geloescht.

        Args:
            create_clicks: Klickzaehler des Anlegen-Knopfs.
            delete_clicks: Klickzaehler des Loeschen-Knopfs.
            new_name: Inhalt des Namensfeldes.
            selected: Aktuell gewaehltes Depot.
            sort: Aktuelle Sortierung der Tabelle.

        Returns:
            Optionen, Auswahl, Inhalt und Rueckmeldung.
        """
        del create_clicks, delete_clicks
        triggered = ctx.triggered_id
        try:
            if triggered == wl_page.PORTFOLIO_CREATE:
                name = (new_name or "").strip()
                if not name:
                    return no_update, no_update, no_update, c.error_banner("Enter a name first.")
                app_state.holdings.create_portfolio(name)
                message = f"Portfolio {name!r} created."
            elif triggered == wl_page.PORTFOLIO_DELETE:
                if not selected:
                    return (
                        no_update,
                        no_update,
                        no_update,
                        c.error_banner("Select a portfolio to delete."),
                    )
                app_state.holdings.delete_portfolio(selected)
                remaining = wl_page.portfolio_options(app_state)
                name = str(remaining[0]["value"]) if remaining else ""
                message = f"Portfolio {selected!r} deleted."
            else:
                return no_update, no_update, no_update, no_update
        except BTLiteError as exc:
            _LOG.warning("WL Depotverwaltung: %s (%s)", exc.user_message, exc.detail)
            return no_update, no_update, no_update, c.error_banner(exc)

        value = name or None
        return (
            wl_page.portfolio_options(app_state),
            value,
            _wl_portfolio_view(value, sort),
            c.placeholder(message),
        )

    @app.callback(
        Output(wl_page.WATCHLIST_SELECT, "options"),
        Output(wl_page.WATCHLIST_SELECT, "value"),
        Output(wl_page.WATCHLIST_OUTPUT, "children", allow_duplicate=True),
        Output(wl_page.WATCHLIST_FEEDBACK, "children"),
        Input(wl_page.WATCHLIST_CREATE, "n_clicks"),
        Input(wl_page.WATCHLIST_DELETE, "n_clicks"),
        State(wl_page.WATCHLIST_NEW_NAME, "value"),
        State(wl_page.WATCHLIST_SELECT, "value"),
        State(wl_page.WATCHLIST_SORT, "data"),
        prevent_initial_call=True,
    )
    def _on_wl_watchlist_manage(
        create_clicks: int | None,
        delete_clicks: int | None,
        new_name: str | None,
        selected: str | None,
        sort: dict[str, Any] | None = None,
    ) -> tuple[Any, Any, Any, Any]:
        """Legt eine Watchlist an oder loescht die gewaehlte.

        Args:
            create_clicks: Klickzaehler des Anlegen-Knopfs.
            delete_clicks: Klickzaehler des Loeschen-Knopfs.
            new_name: Inhalt des Namensfeldes.
            selected: Aktuell gewaehlte Liste.
            sort: Aktuelle Sortierung der Tabelle.

        Returns:
            Optionen, Auswahl, Inhalt und Rueckmeldung.
        """
        del create_clicks, delete_clicks
        triggered = ctx.triggered_id
        try:
            if triggered == wl_page.WATCHLIST_CREATE:
                name = (new_name or "").strip()
                if not name:
                    return no_update, no_update, no_update, c.error_banner("Enter a name first.")
                app_state.holdings.create_watchlist(name)
                message = f"Watchlist {name!r} created."
            elif triggered == wl_page.WATCHLIST_DELETE:
                if not selected:
                    return (
                        no_update,
                        no_update,
                        no_update,
                        c.error_banner("Select a watchlist to delete."),
                    )
                app_state.holdings.delete_watchlist(selected)
                remaining = wl_page.watchlist_options(app_state)
                name = str(remaining[0]["value"]) if remaining else ""
                message = f"Watchlist {selected!r} deleted."
            else:
                return no_update, no_update, no_update, no_update
        except BTLiteError as exc:
            _LOG.warning("WL Watchlistverwaltung: %s (%s)", exc.user_message, exc.detail)
            return no_update, no_update, no_update, c.error_banner(exc)

        value = name or None
        return (
            wl_page.watchlist_options(app_state),
            value,
            _wl_watchlist_view(value, sort),
            c.placeholder(message),
        )

    @app.callback(
        Output(wl_page.PORTFOLIO_OUTPUT, "children", allow_duplicate=True),
        Output(wl_page.HOLDING_FEEDBACK, "children"),
        Input(wl_page.HOLDING_ADD, "n_clicks"),
        State(wl_page.PORTFOLIO_SELECT, "value"),
        State(wl_page.HOLDING_NEW_TICKER, "value"),
        State(wl_page.HOLDING_NEW_SHARES, "value"),
        State(wl_page.HOLDING_NEW_PRICE, "value"),
        State(wl_page.HOLDING_NEW_DATE, "value"),
        State(wl_page.HOLDING_NEW_CURRENCY, "value"),
        State(wl_page.POSITIONS_SORT, "data"),
        prevent_initial_call=True,
    )
    def _on_wl_holding_add(
        n_clicks: int | None,
        portfolio: str | None,
        ticker: str | None,
        shares: float | None,
        price: float | None,
        buy_date: str | None,
        currency: str | None,
        sort: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        """Nimmt einen zusaetzlichen Kauf in das gewaehlte Depot auf.

        Args:
            n_clicks: Klickzaehler des Knopfes "Add".
            portfolio: Gewaehltes Depot.
            ticker: Eingegebenes Symbol.
            shares: Eingegebene Stueckzahl.
            price: Eingegebener Kaufkurs.
            buy_date: Eingegebenes Kaufdatum als ``"YYYY-MM-DD"``.
            currency: Gewaehlte Kaufwaehrung.
            sort: Aktuelle Sortierung der Tabelle.

        Returns:
            Neuer Depotinhalt und Rueckmeldung. Bei einem Fehler bleibt der
            Inhalt stehen.
        """
        from btlite.models.common import Currency

        if not n_clicks:
            return no_update, no_update
        if not portfolio:
            return no_update, c.error_banner("Select a portfolio first.")
        symbol = (ticker or "").strip()
        if not symbol:
            return no_update, c.error_banner("Enter a ticker.")

        count = parse_amount(shares, "shares")
        if isinstance(count, str):
            return no_update, c.error_banner(count)
        amount = parse_amount(price, "buy price")
        if isinstance(amount, str):
            return no_update, c.error_banner(amount)
        bought_on = parse_buy_date(buy_date)
        if isinstance(bought_on, str):
            return no_update, c.error_banner(bought_on)

        try:
            bought_in = Currency(currency) if currency else None
        except ValueError:
            return no_update, c.error_banner(f"{currency!r} is not a known currency.")

        try:
            app_state.holdings.add_holding(
                portfolio,
                symbol,
                shares=count,
                buy_price=amount,
                buy_date=bought_on,
                currency=bought_in,
            )
        except BTLiteError as exc:
            _LOG.warning("WL Position anlegen: %s (%s)", exc.user_message, exc.detail)
            return no_update, c.error_banner(exc)

        return _wl_portfolio_view(portfolio, sort), c.placeholder(
            f"Added {symbol} to {portfolio!r}."
        )

    @app.callback(
        Output(wl_page.PORTFOLIO_OUTPUT, "children", allow_duplicate=True),
        Output(wl_page.HOLDING_FEEDBACK, "children", allow_duplicate=True),
        Input({"type": wl_page.HOLDING_SAVE_TYPE, "index": ALL}, "n_clicks"),
        State(wl_page.PORTFOLIO_SELECT, "value"),
        State({"type": wl_page.HOLDING_SHARES_TYPE, "index": ALL}, "value"),
        State({"type": wl_page.HOLDING_PRICE_TYPE, "index": ALL}, "value"),
        State({"type": wl_page.HOLDING_DATE_TYPE, "index": ALL}, "value"),
        State({"type": wl_page.HOLDING_SHARES_TYPE, "index": ALL}, "id"),
        State(wl_page.POSITIONS_SORT, "data"),
        prevent_initial_call=True,
    )
    def _on_wl_holding_save(
        n_clicks: list[int | None],
        portfolio: str | None,
        shares: list[float | None],
        prices: list[float | None],
        dates: list[str | None],
        ids: list[dict[str, Any]],
        sort: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        """Speichert die geaenderten Werte **einer** Position.

        Geschrieben wird allein die geklickte Zeile. Alle Zeilen in einem
        Durchgang zu speichern waere der Fehler, den die wandernde Adresse
        bestraft, und schriebe ausserdem fremde Felder mit, die der Nutzer
        nur versehentlich beruehrt hat.

        Welcher Eintrag der Wertelisten zur geklickten Zeile gehoert, sagt
        die Liste der Feld-IDs; auf die Reihenfolge zu vertrauen waere
        falsch, sobald eine Zeile fehlt.

        Args:
            n_clicks: Klickzaehler aller Speichern-Knoepfe.
            portfolio: Gewaehltes Depot.
            shares: Werte aller Stueckzahl-Felder.
            prices: Werte aller Kursfelder.
            dates: Werte aller Datumsfelder.
            ids: IDs der Stueckzahl-Felder, zur Zuordnung ueber den Index.
            sort: Aktuelle Sortierung der Tabelle.

        Returns:
            Neuer Depotinhalt und Rueckmeldung.
        """
        if not any(n_clicks or ()):
            return no_update, no_update
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or not portfolio:
            return no_update, no_update
        index = triggered.get("index")
        if not isinstance(index, int):
            return no_update, no_update

        row = next((i for i, element in enumerate(ids) if element.get("index") == index), None)
        if row is None:
            _LOG.warning("WL Position %s nicht in den Feldern gefunden", index)
            return no_update, no_update

        count = parse_amount(shares[row], "shares")
        if isinstance(count, str):
            return no_update, c.error_banner(count)
        amount = parse_amount(prices[row], "buy price")
        if isinstance(amount, str):
            return no_update, c.error_banner(amount)
        bought_on = parse_buy_date(dates[row])
        if isinstance(bought_on, str):
            return no_update, c.error_banner(bought_on)

        try:
            app_state.holdings.update_holding(
                portfolio,
                index,
                shares=count,
                buy_price=amount,
                buy_date=bought_on,
            )
        except BTLiteError as exc:
            _LOG.warning("WL Position aendern: %s (%s)", exc.user_message, exc.detail)
            return no_update, c.error_banner(exc)

        return _wl_portfolio_view(portfolio, sort), c.placeholder("Position saved.")

    @app.callback(
        Output(wl_page.PORTFOLIO_OUTPUT, "children", allow_duplicate=True),
        Output(wl_page.HOLDING_FEEDBACK, "children", allow_duplicate=True),
        Input({"type": wl_page.HOLDING_REMOVE_TYPE, "index": ALL}, "n_clicks"),
        State(wl_page.PORTFOLIO_SELECT, "value"),
        State(wl_page.POSITIONS_SORT, "data"),
        prevent_initial_call=True,
    )
    def _on_wl_holding_remove(
        n_clicks: list[int | None],
        portfolio: str | None,
        sort: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        """Entfernt **eine** Position aus dem gewaehlten Depot.

        Nach dem Entfernen ruecken alle folgenden Haltungen um eine
        Position vor. Genau deshalb rendert der Callback anschliessend neu:
        die neuen Zeilen tragen die neuen Indizes, und die naechste
        Aenderung trifft wieder die gemeinte Zeile.

        Args:
            n_clicks: Klickzaehler aller Entfernen-Knoepfe.
            portfolio: Gewaehltes Depot.
            sort: Aktuelle Sortierung der Tabelle.

        Returns:
            Neuer Depotinhalt und Rueckmeldung.
        """
        if not any(n_clicks or ()):
            return no_update, no_update
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or not portfolio:
            return no_update, no_update
        index = triggered.get("index")
        if not isinstance(index, int):
            return no_update, no_update

        try:
            app_state.holdings.remove_holding(portfolio, index)
        except BTLiteError as exc:
            _LOG.warning("WL Position entfernen: %s (%s)", exc.user_message, exc.detail)
            return no_update, c.error_banner(exc)

        return _wl_portfolio_view(portfolio, sort), c.placeholder("Position removed.")

    @app.callback(
        Output(wl_page.WATCHLIST_OUTPUT, "children", allow_duplicate=True),
        Output(wl_page.WATCHLIST_FEEDBACK, "children", allow_duplicate=True),
        Input(wl_page.WATCHLIST_ADD, "n_clicks"),
        State(wl_page.WATCHLIST_SELECT, "value"),
        State(wl_page.WATCHLIST_NEW_TICKER, "value"),
        State(wl_page.WATCHLIST_NEW_NOTE, "value"),
        State(wl_page.WATCHLIST_SORT, "data"),
        prevent_initial_call=True,
    )
    def _on_wl_watch_add(
        n_clicks: int | None,
        watchlist: str | None,
        ticker: str | None,
        note: str | None,
        sort: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        """Nimmt ein Symbol in die gewaehlte Watchlist auf.

        Args:
            n_clicks: Klickzaehler des Knopfes "Watch".
            watchlist: Gewaehlte Liste.
            ticker: Eingegebenes Symbol.
            note: Eingegebene Notiz. Leer heisst "keine Notiz".
            sort: Aktuelle Sortierung der Tabelle.

        Returns:
            Neuer Listeninhalt und Rueckmeldung.
        """
        if not n_clicks:
            return no_update, no_update
        if not watchlist:
            return no_update, c.error_banner("Select a watchlist first.")
        symbol = (ticker or "").strip()
        if not symbol:
            return no_update, c.error_banner("Enter a ticker.")

        try:
            app_state.holdings.add_to_watchlist(
                watchlist, symbol, note=(note or "").strip() or None
            )
        except BTLiteError as exc:
            _LOG.warning("WL Watchlist-Eintrag: %s (%s)", exc.user_message, exc.detail)
            return no_update, c.error_banner(exc)

        return _wl_watchlist_view(watchlist, sort), c.placeholder(
            f"Added {symbol} to {watchlist!r}."
        )

    @app.callback(
        Output(wl_page.WATCHLIST_OUTPUT, "children", allow_duplicate=True),
        Output(wl_page.WATCHLIST_FEEDBACK, "children", allow_duplicate=True),
        Input({"type": wl_page.WATCH_SAVE_TYPE, "ticker": ALL}, "n_clicks"),
        State(wl_page.WATCHLIST_SELECT, "value"),
        State({"type": wl_page.WATCH_NOTE_TYPE, "ticker": ALL}, "value"),
        State({"type": wl_page.WATCH_NOTE_TYPE, "ticker": ALL}, "id"),
        State(wl_page.WATCHLIST_SORT, "data"),
        prevent_initial_call=True,
    )
    def _on_wl_watch_save(
        n_clicks: list[int | None],
        watchlist: str | None,
        notes: list[str | None],
        ids: list[dict[str, Any]],
        sort: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        """Speichert die Notiz **eines** beobachteten Titels.

        Schluessel ist hier der Ticker, nicht ein Index: eine Watchlist
        haelt jedes Symbol hoechstens einmal, diese Adresse wandert also
        nicht. Trotzdem schreibt der Callback nur die geklickte Zeile -
        alle Notizen in einem Durchgang zu speichern schriebe fremde
        Felder mit.

        Args:
            n_clicks: Klickzaehler aller Speichern-Knoepfe.
            watchlist: Gewaehlte Liste.
            notes: Werte aller Notizfelder.
            ids: IDs der Notizfelder, zur Zuordnung ueber den Ticker.
            sort: Aktuelle Sortierung der Tabelle.

        Returns:
            Neuer Listeninhalt und Rueckmeldung.
        """
        if not any(n_clicks or ()):
            return no_update, no_update
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or not watchlist:
            return no_update, no_update
        ticker = str(triggered.get("ticker", ""))
        if not ticker:
            return no_update, no_update

        row = next((i for i, element in enumerate(ids) if element.get("ticker") == ticker), None)
        if row is None:
            _LOG.warning("WL Notizfeld zu %s nicht gefunden", ticker)
            return no_update, no_update

        try:
            app_state.holdings.set_watchlist_note(watchlist, ticker, notes[row])
        except BTLiteError as exc:
            _LOG.warning("WL Notiz speichern: %s (%s)", exc.user_message, exc.detail)
            return no_update, c.error_banner(exc)

        return _wl_watchlist_view(watchlist, sort), c.placeholder(f"Note for {ticker} saved.")

    @app.callback(
        Output(wl_page.WATCHLIST_OUTPUT, "children", allow_duplicate=True),
        Output(wl_page.WATCHLIST_FEEDBACK, "children", allow_duplicate=True),
        Input({"type": wl_page.WATCH_REMOVE_TYPE, "ticker": ALL}, "n_clicks"),
        State(wl_page.WATCHLIST_SELECT, "value"),
        State(wl_page.WATCHLIST_SORT, "data"),
        prevent_initial_call=True,
    )
    def _on_wl_watch_remove(
        n_clicks: list[int | None],
        watchlist: str | None,
        sort: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        """Entfernt ein Symbol aus der gewaehlten Watchlist.

        Args:
            n_clicks: Klickzaehler aller Entfernen-Knoepfe.
            watchlist: Gewaehlte Liste.
            sort: Aktuelle Sortierung der Tabelle.

        Returns:
            Neuer Listeninhalt und Rueckmeldung.
        """
        if not any(n_clicks or ()):
            return no_update, no_update
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or not watchlist:
            return no_update, no_update
        ticker = str(triggered.get("ticker", ""))
        if not ticker:
            return no_update, no_update

        try:
            app_state.holdings.remove_from_watchlist(watchlist, ticker)
        except BTLiteError as exc:
            _LOG.warning("WL Watchlist-Eintrag entfernen: %s (%s)", exc.user_message, exc.detail)
            return no_update, c.error_banner(exc)

        return _wl_watchlist_view(watchlist, sort), c.placeholder(f"{ticker} removed.")

    # Die Sortier-Callbacks halten sich an Architekturregel 4: sie lesen
    # den Klick, fragen ``next_sort_state`` nach der neuen Ordnung und
    # rendern neu. Sortiert wird in ``sort_frame``, nicht hier.
    #
    # Die neue Sortierung geht zugleich in den Speicher **und** in den
    # View-Helfer. Der Speicher allein genuegt nicht: ein ``State`` liest
    # den Stand vor diesem Durchgang, die Tabelle bekaeme also die alte
    # Ordnung und spraenge erst beim naechsten Klick nach.

    @app.callback(
        Output(wl_page.POSITIONS_SORT, "data"),
        Output(wl_page.PORTFOLIO_OUTPUT, "children", allow_duplicate=True),
        Input(
            {"type": c.SORT_HEADER_TYPE, "table": wl_page.POSITIONS_TABLE, "column": ALL},
            "n_clicks",
        ),
        State(wl_page.POSITIONS_SORT, "data"),
        State(wl_page.PORTFOLIO_SELECT, "value"),
        prevent_initial_call=True,
    )
    def _on_wl_positions_sort(
        n_clicks: list[int | None],
        sort: dict[str, Any] | None,
        portfolio: str | None,
    ) -> tuple[Any, Any]:
        """Sortiert die Depottabelle nach der angeklickten Spalte.

        Dash wertet einen Callback mit ``ALL``-Eingabe als
        Wildcard-Callback. Die Rueckgabe bleibt hier ein Tupel, weil zwei
        Outputs bedient werden - die Listenform gilt nur fuer den Fall
        eines einzelnen Outputs.

        Args:
            n_clicks: Klickzaehler aller Ueberschriften der Tabelle.
            sort: Aktuelle Sortierung der Tabelle.
            portfolio: Gewaehltes Depot.

        Returns:
            Neue Sortierung und neu gerenderter Depotinhalt.
        """
        if not any(n_clicks or ()):
            return no_update, no_update
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return no_update, no_update
        clicked = str(triggered.get("column", ""))
        if not clicked:
            return no_update, no_update

        current_column, current_desc = _sort_of(sort)
        column, desc = c.next_sort_state(current_column, current_desc, clicked)
        updated: dict[str, Any] = {"column": column, "desc": desc}
        return updated, _wl_portfolio_view(portfolio, updated)

    @app.callback(
        Output(wl_page.WATCHLIST_SORT, "data"),
        Output(wl_page.WATCHLIST_OUTPUT, "children", allow_duplicate=True),
        Input(
            {"type": c.SORT_HEADER_TYPE, "table": wl_page.WATCHLIST_TABLE, "column": ALL},
            "n_clicks",
        ),
        State(wl_page.WATCHLIST_SORT, "data"),
        State(wl_page.WATCHLIST_SELECT, "value"),
        prevent_initial_call=True,
    )
    def _on_wl_watchlist_sort(
        n_clicks: list[int | None],
        sort: dict[str, Any] | None,
        watchlist: str | None,
    ) -> tuple[Any, Any]:
        """Sortiert die Watchlist-Tabelle nach der angeklickten Spalte.

        Args:
            n_clicks: Klickzaehler aller Ueberschriften der Tabelle.
            sort: Aktuelle Sortierung der Tabelle.
            watchlist: Gewaehlte Liste.

        Returns:
            Neue Sortierung und neu gerenderter Listeninhalt.
        """
        if not any(n_clicks or ()):
            return no_update, no_update
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return no_update, no_update
        clicked = str(triggered.get("column", ""))
        if not clicked:
            return no_update, no_update

        current_column, current_desc = _sort_of(sort)
        column, desc = c.next_sort_state(current_column, current_desc, clicked)
        updated: dict[str, Any] = {"column": column, "desc": desc}
        return updated, _wl_watchlist_view(watchlist, updated)

    return app
