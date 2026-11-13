"""Tests der Dash-Shell: Routing, Adressbau und Fehlerbehandlung."""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from dash import no_update
from tests.unit.test_ui_pages import has_class, text_of

from btlite.config import Settings
from btlite.models.common import Currency
from btlite.ui.app import (
    DEFAULT_PATH,
    DISCLAIMER,
    build_href,
    create_app,
    header_notice,
    parse_amount,
    parse_buy_date,
    parse_href,
    render_page,
)
from btlite.ui.command_bar import Command, parse_command
from btlite.ui.layouts import components as c
from btlite.ui.pages import eqs as eqs_page
from btlite.ui.pages import wl
from btlite.ui.state import AppState, set_state

#: Kaufdatum der Testbestaende.
BUY_DATE = dt.date(2024, 1, 15)


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[AppState]:
    """Baut einen Fixture-gestuetzten UI-Zustand.

    Die Depot-Datei zeigt ins temporaere Verzeichnis. Ohne das schriebe
    jeder Test, der eine Position anlegt, in den echten Bestand des
    Nutzers unter ``~/.btlite`` - dort liegt eine echte Position.
    """
    monkeypatch.setenv("BTLITE_PORTFOLIO_HOLDINGS_FILE", str(tmp_path / "holdings.json"))
    settings = Settings()
    settings.data.offline = True
    instance = AppState(settings)
    set_state(instance)
    yield instance
    set_state(None)


class TestBuildHref:
    def test_with_tickers(self):
        command = Command(mnemonic="FA", tickers=("AAPL", "MSFT"))
        assert build_href(command) == "/fa?t=AAPL%2CMSFT"

    def test_without_tickers(self):
        assert build_href(Command(mnemonic="EQS")) == "/eqs"

    def test_with_args(self):
        command = Command(mnemonic="GP", tickers=("AAPL",), args=("5y",))
        href = build_href(command)
        assert href.startswith("/gp?")
        assert "a=5y" in href


class TestParseHref:
    def test_roundtrip(self):
        parsed = parse_command("AAPL,MSFT GP 5y GO")
        assert isinstance(parsed, Command)
        href = build_href(parsed)
        path, _, query = href.partition("?")
        mnemonic, tickers, args = parse_href(path, f"?{query}")
        assert mnemonic == "GP"
        assert tickers == ["AAPL", "MSFT"]
        assert args == ("5y",)

    def test_unknown_path_falls_back(self):
        mnemonic, _, _ = parse_href("/nosuchpage", "")
        assert mnemonic == "DES"

    def test_root_path(self):
        mnemonic, tickers, args = parse_href("/", None)
        assert mnemonic == "DES"
        assert tickers == []
        assert args == ()

    def test_none_path(self):
        assert parse_href(None, None)[0] == "DES"

    def test_case_insensitive_path(self):
        assert parse_href("/FA", "")[0] == "FA"

    def test_trailing_slash(self):
        assert parse_href("/bt/", "")[0] == "BT"

    def test_default_path_is_known(self):
        assert parse_href(DEFAULT_PATH, "")[0] == "DES"


class TestRenderPage:
    def test_renders_known_page(self, state):
        tree = render_page(state, "DES", ["AAPL"], ())
        assert not has_class(tree, "bt-error")

    def test_unknown_mnemonic_banner(self, state):
        tree = render_page(state, "NOPE", [], ())
        assert has_class(tree, "bt-error")

    def test_data_error_becomes_banner(self, state):
        tree = render_page(state, "DES", ["NOSUCHTICKER"], ())
        assert has_class(tree, "bt-error")
        assert tree is not None

    def test_unexpected_error_becomes_banner(self, state, monkeypatch):
        from btlite.ui import pages

        def boom(*_args, **_kwargs):
            msg = "kaputt"
            raise RuntimeError(msg)

        monkeypatch.setitem(pages.PAGES, "DES", boom)
        tree = render_page(state, "DES", ["AAPL"], ())
        assert has_class(tree, "bt-error")
        assert "kaputt" not in text_of(tree)


class TestHeaderNotice:
    def test_offline_names_fixtures(self, state):
        assert "fixtures" in header_notice(state)

    def test_never_claims_realtime(self, state):
        assert "Echtzeit" not in header_notice(state).replace("keine Echtzeit", "")


class TestCreateApp:
    def test_builds(self, state):
        app = create_app(state.settings, state)
        assert app.layout is not None

    def test_registers_callbacks(self, state):
        app = create_app(state.settings, state)
        assert len(app.callback_map) >= 6

    def test_disclaimer_present(self, state):
        app = create_app(state.settings, state)
        assert DISCLAIMER in text_of(app.layout)

    def test_command_bar_present(self, state):
        app = create_app(state.settings, state)
        # Nur die schlichten String-IDs: die Navigationslinks tragen seit
        # dem Nachfuehren des gewaehlten Tickers Wildcard-IDs, und ein
        # Dictionary ist nicht hashbar.
        ids = {
            element
            for element in (getattr(n, "id", None) for n in _walk(app.layout))
            if isinstance(element, str)
        }
        assert "command-bar" in ids
        assert "page-content" in ids
        assert "url" in ids


def _walk(node):
    """Laeuft rekursiv durch einen Dash-Komponentenbaum."""
    yield node
    children = getattr(node, "children", None)
    if children is None:
        return
    if isinstance(children, list | tuple):
        for child in children:
            yield from _walk(child)
    else:
        yield from _walk(children)


# ----------------------------------------------------------- Seite WL ----


@pytest.fixture
def filled(state: AppState) -> AppState:
    """Zustand mit einem Depot, drei Kaeufen und einer Watchlist."""
    store = state.holdings
    store.create_portfolio("Mein Depot")
    store.add_holding("Mein Depot", "AAPL", shares=10, buy_price=150.0, buy_date=BUY_DATE)
    store.add_holding("Mein Depot", "MSFT", shares=5, buy_price=380.0, buy_date=BUY_DATE)
    store.add_holding("Mein Depot", "KO", shares=7, buy_price=60.0, buy_date=BUY_DATE)
    store.create_watchlist("Beobachtung")
    store.add_to_watchlist("Beobachtung", "KO", note="Dividende")
    return state


def _holdings_of(state: AppState, name: str = "Mein Depot") -> list[Any]:
    """Liest die Haltungen frisch aus der Datei.

    Args:
        state: UI-Zustand.
        name: Name des Depots.

    Returns:
        Die gespeicherten Haltungen.
    """
    portfolio = state.holdings.get_portfolio(name)
    assert portfolio is not None
    return list(portfolio.holdings)


def _ticker_order(component: Any) -> list[str]:
    """Liest die Ticker in der Reihenfolge, in der die Tabelle sie zeigt.

    Gelesen wird der Komponentenbaum, nicht ``str(...)``: ein Substring
    faende einen Ticker auch im Bearbeitungsformular darunter und sagte
    nichts ueber die Reihenfolge der Tabellenzeilen.

    Args:
        component: Gerenderter Inhalt des Depot- oder Watchlist-Bereichs.

    Returns:
        Ticker in Anzeigereihenfolge, ohne die Kopfzeile.
    """
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, list | tuple):
            for item in node:
                walk(item)
            return
        children = getattr(node, "children", None)
        if type(node).__name__ == "Tbody":
            rows = children if isinstance(children, list) else [children]
            for row in rows:
                cells = getattr(row, "children", None)
                if isinstance(cells, list) and cells:
                    first = getattr(cells[0], "children", None)
                    if isinstance(first, str):
                        found.append(first)
            return
        if children is not None:
            walk(children)

    walk(component)
    return found


def _callback(app: Any, predicate: Any) -> Any:
    """Holt die registrierte Funktion eines Callbacks.

    Gesucht wird ueber ``app.callback_map``, also ueber dieselbe
    Registrierung, die auch der Server bedient.

    Args:
        app: Gebaute Dash-App.
        predicate: Praedikat ueber Schluessel und Eintrag.

    Returns:
        Die registrierte Callback-Funktion.
    """
    for key, entry in app.callback_map.items():
        if predicate(key, entry):
            # Dash umhuellt die registrierte Funktion; die Huelle verlangt
            # ein ``outputs_list`` aus der HTTP-Anfrage. ``__wrapped__``
            # ist die Funktion aus app.py selbst - genau die soll geprueft
            # werden, und zwar die registrierte, nicht eine zweite
            # Referenz darauf.
            callback = entry["callback"]
            return getattr(callback, "__wrapped__", callback)
    raise AssertionError("Callback nicht gefunden")


def _by_input(app: Any, output: str, marker: str) -> Any:
    """Sucht einen Callback ueber Output und ein Merkmal seiner Eingaben.

    Args:
        app: Gebaute Dash-App.
        output: Element-ID, die im Output vorkommen muss.
        marker: Text, der in den Eingaben vorkommen muss.

    Returns:
        Die registrierte Callback-Funktion.
    """
    return _callback(
        app,
        lambda key, entry: output in key and marker in str(entry.get("inputs", "")),
    )


def _set_triggered(monkeypatch: pytest.MonkeyPatch, triggered: Any) -> None:
    """Setzt ``ctx.triggered_id`` fuer einen direkten Aufruf.

    Ein direkt aufgerufener Callback laeuft ausserhalb einer Anfrage; ohne
    diesen Ersatz kaeme er nicht an die ausloesende ID heran.

    Args:
        monkeypatch: Pytest-Helfer.
        triggered: Die ID, die als ausloesend gelten soll.
    """
    import dash

    monkeypatch.setattr(
        type(dash.ctx), "triggered_id", property(lambda _self: triggered), raising=False
    )


def _shares_ids(count: int) -> list[dict[str, Any]]:
    """Baut die IDs der Stueckzahl-Felder, wie Dash sie mitsendet.

    Args:
        count: Anzahl der Zeilen.

    Returns:
        Eine ID je Zeile, in Zeilenreihenfolge.
    """
    return [{"type": wl.HOLDING_SHARES_TYPE, "index": i} for i in range(count)]


class TestParseBuyDate:
    """Das Kaufdatum aus dem Textfeld."""

    def test_valid_date(self) -> None:
        assert parse_buy_date("2024-01-15") == BUY_DATE

    def test_unparsable_is_a_message(self) -> None:
        result = parse_buy_date("15.01.2024")
        assert isinstance(result, str)
        assert "YYYY-MM-DD" in result

    def test_empty_is_a_message(self) -> None:
        assert isinstance(parse_buy_date(""), str)

    def test_no_silent_today(self) -> None:
        # Ein unparsbares Datum darf nie still zum heutigen Tag werden.
        assert parse_buy_date("kaputt") != dt.date.today()


class TestParseAmount:
    """Stueckzahl und Kaufkurs muessen groesser null sein."""

    def test_positive_value(self) -> None:
        assert parse_amount(2.5, "shares") == 2.5

    def test_none_is_a_message(self) -> None:
        assert isinstance(parse_amount(None, "shares"), str)

    def test_zero_is_a_message(self) -> None:
        assert isinstance(parse_amount(0, "shares"), str)

    def test_negative_is_a_message(self) -> None:
        assert isinstance(parse_amount(-1, "buy price"), str)


class TestWlDropdowns:
    """Die beiden Auswahlfelder, die bisher an nichts hingen."""

    def test_portfolio_select_renders(self, filled: AppState) -> None:
        app = create_app(filled.settings, filled)
        callback = _callback(app, lambda key, _e: key == f"{wl.PORTFOLIO_OUTPUT}.children")
        assert "AAPL" in text_of(callback("Mein Depot"))

    def test_watchlist_select_renders(self, filled: AppState) -> None:
        app = create_app(filled.settings, filled)
        callback = _callback(app, lambda key, _e: key == f"{wl.WATCHLIST_OUTPUT}.children")
        assert "KO" in text_of(callback("Beobachtung"))


class TestWlPortfolioManage:
    """Anlegen und Loeschen eines Depots."""

    def _manage(self, app: Any) -> Any:
        return _callback(app, lambda key, _e: wl.PORTFOLIO_FEEDBACK in key)

    def test_create_writes_the_file(self, state: AppState, monkeypatch: Any) -> None:
        app = create_app(state.settings, state)
        _set_triggered(monkeypatch, wl.PORTFOLIO_CREATE)
        options, value, content, feedback = self._manage(app)(1, 0, "Neu", None)
        assert state.holdings.get_portfolio("Neu") is not None
        assert options == [{"label": "Neu", "value": "Neu"}]
        assert value == "Neu"
        assert content is not no_update
        assert not has_class(feedback, "bt-error")

    def test_options_match_the_store(self, filled: AppState, monkeypatch: Any) -> None:
        app = create_app(filled.settings, filled)
        _set_triggered(monkeypatch, wl.PORTFOLIO_CREATE)
        options, _value, _content, _feedback = self._manage(app)(1, 0, "Zweitdepot", None)
        assert [o["value"] for o in options] == filled.holdings.list_portfolios()

    def test_name_with_spaces_and_special_characters(
        self, state: AppState, monkeypatch: Any
    ) -> None:
        name = "Mein Depot & Soehne (2024) - 50%"
        app = create_app(state.settings, state)
        _set_triggered(monkeypatch, wl.PORTFOLIO_CREATE)
        options, value, _content, feedback = self._manage(app)(1, 0, name, None)
        assert state.holdings.get_portfolio(name) is not None
        assert value == name
        assert options == [{"label": name, "value": name}]
        assert not has_class(feedback, "bt-error")

    def test_name_is_trimmed(self, state: AppState, monkeypatch: Any) -> None:
        app = create_app(state.settings, state)
        _set_triggered(monkeypatch, wl.PORTFOLIO_CREATE)
        self._manage(app)(1, 0, "  Depot  ", None)
        assert state.holdings.list_portfolios() == ["Depot"]

    def test_delete_removes_it(self, filled: AppState, monkeypatch: Any) -> None:
        app = create_app(filled.settings, filled)
        _set_triggered(monkeypatch, wl.PORTFOLIO_DELETE)
        options, value, _content, _feedback = self._manage(app)(0, 1, "", "Mein Depot")
        assert filled.holdings.list_portfolios() == []
        assert options == []
        assert value is None

    def test_delete_falls_back_to_the_first_remaining(
        self, filled: AppState, monkeypatch: Any
    ) -> None:
        filled.holdings.create_portfolio("Anderes")
        app = create_app(filled.settings, filled)
        _set_triggered(monkeypatch, wl.PORTFOLIO_DELETE)
        options, value, _content, _feedback = self._manage(app)(0, 1, "", "Mein Depot")
        assert value == "Anderes"
        assert [o["value"] for o in options] == ["Anderes"]

    def test_duplicate_name_keeps_the_page(self, filled: AppState, monkeypatch: Any) -> None:
        # Der Schreckfall: nach einem Fehler darf der Bestand nicht vom
        # Bildschirm verschwinden.
        app = create_app(filled.settings, filled)
        _set_triggered(monkeypatch, wl.PORTFOLIO_CREATE)
        options, value, content, feedback = self._manage(app)(1, 0, "Mein Depot", "Mein Depot")
        assert content is no_update
        assert options is no_update
        assert value is no_update
        assert has_class(feedback, "bt-error")
        assert filled.holdings.list_portfolios() == ["Mein Depot"]

    def test_empty_name_is_rejected(self, state: AppState, monkeypatch: Any) -> None:
        app = create_app(state.settings, state)
        _set_triggered(monkeypatch, wl.PORTFOLIO_CREATE)
        _options, _value, content, feedback = self._manage(app)(1, 0, "   ", None)
        assert content is no_update
        assert has_class(feedback, "bt-error")
        assert state.holdings.list_portfolios() == []

    def test_delete_without_selection_is_rejected(self, filled: AppState, monkeypatch: Any) -> None:
        app = create_app(filled.settings, filled)
        _set_triggered(monkeypatch, wl.PORTFOLIO_DELETE)
        _options, _value, content, feedback = self._manage(app)(0, 1, "", None)
        assert content is no_update
        assert has_class(feedback, "bt-error")
        assert filled.holdings.list_portfolios() == ["Mein Depot"]


class TestWlWatchlistManage:
    """Anlegen und Loeschen einer Watchlist."""

    def _manage(self, app: Any) -> Any:
        return _callback(
            app,
            lambda key, _e: wl.WATCHLIST_FEEDBACK in key and wl.WATCHLIST_SELECT in key,
        )

    def test_create_writes_the_file(self, state: AppState, monkeypatch: Any) -> None:
        app = create_app(state.settings, state)
        _set_triggered(monkeypatch, wl.WATCHLIST_CREATE)
        options, value, _content, _feedback = self._manage(app)(1, 0, "Radar", None)
        assert state.holdings.get_watchlist("Radar") is not None
        assert [o["value"] for o in options] == state.holdings.list_watchlists()
        assert value == "Radar"

    def test_delete_removes_it(self, filled: AppState, monkeypatch: Any) -> None:
        app = create_app(filled.settings, filled)
        _set_triggered(monkeypatch, wl.WATCHLIST_DELETE)
        options, value, _content, _feedback = self._manage(app)(0, 1, "", "Beobachtung")
        assert filled.holdings.list_watchlists() == []
        assert options == []
        assert value is None

    def test_duplicate_keeps_the_page(self, filled: AppState, monkeypatch: Any) -> None:
        app = create_app(filled.settings, filled)
        _set_triggered(monkeypatch, wl.WATCHLIST_CREATE)
        _options, _value, content, feedback = self._manage(app)(1, 0, "Beobachtung", None)
        assert content is no_update
        assert has_class(feedback, "bt-error")
        assert filled.holdings.list_watchlists() == ["Beobachtung"]


class TestWlHoldingAdd:
    """Eine Position anlegen."""

    def _add(self, app: Any) -> Any:
        return _by_input(app, wl.HOLDING_FEEDBACK, wl.HOLDING_ADD)

    def test_store_error_keeps_the_page(self, filled: AppState) -> None:
        """Ein Fehler des Speichers darf den Bestand nicht vom Schirm nehmen.

        Geprueft wird hier der ``BTLiteError``-Zweig selbst, nicht die
        vorgelagerte Eingabepruefung: ein unbekanntes Depot laesst
        ``add_holding`` werfen, nachdem alle Felder gueltig waren.
        """
        app = create_app(filled.settings, filled)
        content, feedback = self._add(app)(
            1, "Gibt es nicht", "TSLA", 4, 210.0, "2024-03-01", "USD"
        )
        assert content is no_update, "Der Fehler hat den Depotbereich geleert"
        assert has_class(feedback, "bt-error")

    def test_add_reaches_the_file(self, filled: AppState) -> None:
        app = create_app(filled.settings, filled)
        content, feedback = self._add(app)(1, "Mein Depot", "TSLA", 4, 210.0, "2024-03-01", "USD")
        holdings = _holdings_of(filled)
        assert holdings[-1].ticker == "TSLA"
        assert holdings[-1].shares == 4
        assert holdings[-1].buy_price == 210.0
        assert holdings[-1].buy_date == dt.date(2024, 3, 1)
        assert content is not no_update
        assert not has_class(feedback, "bt-error")

    def test_currency_is_kept(self, filled: AppState) -> None:
        app = create_app(filled.settings, filled)
        self._add(app)(1, "Mein Depot", "SAP", 3, 120.0, "2024-03-01", "EUR")
        assert _holdings_of(filled)[-1].currency.value == "EUR"

    def test_invalid_date_changes_nothing(self, filled: AppState) -> None:
        before = _holdings_of(filled)
        app = create_app(filled.settings, filled)
        content, feedback = self._add(app)(1, "Mein Depot", "TSLA", 4, 210.0, "01.03.2024", "USD")
        assert content is no_update
        assert has_class(feedback, "bt-error")
        assert _holdings_of(filled) == before

    def test_empty_shares_changes_nothing(self, filled: AppState) -> None:
        before = _holdings_of(filled)
        app = create_app(filled.settings, filled)
        content, feedback = self._add(app)(
            1, "Mein Depot", "TSLA", None, 210.0, "2024-03-01", "USD"
        )
        assert content is no_update
        assert has_class(feedback, "bt-error")
        assert _holdings_of(filled) == before

    def test_zero_shares_changes_nothing(self, filled: AppState) -> None:
        before = _holdings_of(filled)
        app = create_app(filled.settings, filled)
        content, feedback = self._add(app)(1, "Mein Depot", "TSLA", 0, 210.0, "2024-03-01", "USD")
        assert content is no_update
        assert has_class(feedback, "bt-error")
        assert _holdings_of(filled) == before

    def test_zero_price_changes_nothing(self, filled: AppState) -> None:
        before = _holdings_of(filled)
        app = create_app(filled.settings, filled)
        content, feedback = self._add(app)(1, "Mein Depot", "TSLA", 4, 0, "2024-03-01", "USD")
        assert content is no_update
        assert has_class(feedback, "bt-error")
        assert _holdings_of(filled) == before

    def test_missing_ticker_changes_nothing(self, filled: AppState) -> None:
        before = _holdings_of(filled)
        app = create_app(filled.settings, filled)
        content, feedback = self._add(app)(1, "Mein Depot", "  ", 4, 210.0, "2024-03-01", "USD")
        assert content is no_update
        assert has_class(feedback, "bt-error")
        assert _holdings_of(filled) == before

    def test_unknown_currency_changes_nothing(self, filled: AppState) -> None:
        before = _holdings_of(filled)
        app = create_app(filled.settings, filled)
        content, feedback = self._add(app)(1, "Mein Depot", "TSLA", 4, 210.0, "2024-03-01", "XYZ")
        assert content is no_update
        assert has_class(feedback, "bt-error")
        assert _holdings_of(filled) == before

    def test_without_a_click_nothing_happens(self, filled: AppState) -> None:
        before = _holdings_of(filled)
        app = create_app(filled.settings, filled)
        content, feedback = self._add(app)(0, "Mein Depot", "TSLA", 4, 210.0, "2024-03-01", "USD")
        assert content is no_update
        assert feedback is no_update
        assert _holdings_of(filled) == before


class TestWlHoldingSaveAndRemove:
    """Aendern, Entfernen - und die wandernde Adresse."""

    def _save(self, app: Any) -> Any:
        return _by_input(app, wl.HOLDING_FEEDBACK, wl.HOLDING_SAVE_TYPE)

    def _remove(self, app: Any) -> Any:
        return _by_input(app, wl.HOLDING_FEEDBACK, wl.HOLDING_REMOVE_TYPE)

    def test_save_reaches_the_file(self, filled: AppState, monkeypatch: Any) -> None:
        app = create_app(filled.settings, filled)
        _set_triggered(monkeypatch, {"type": wl.HOLDING_SAVE_TYPE, "index": 1})
        content, feedback = self._save(app)(
            [0, 1, 0],
            "Mein Depot",
            [10, 99, 7],
            [150.0, 400.0, 60.0],
            ["2024-01-15", "2024-02-02", "2024-01-15"],
            _shares_ids(3),
        )
        holdings = _holdings_of(filled)
        assert holdings[1].shares == 99
        assert holdings[1].buy_price == 400.0
        assert holdings[1].buy_date == dt.date(2024, 2, 2)
        # Die anderen Zeilen bleiben unberuehrt.
        assert holdings[0].shares == 10
        assert holdings[2].shares == 7
        assert content is not no_update
        assert not has_class(feedback, "bt-error")

    def test_remove_reaches_the_file(self, filled: AppState, monkeypatch: Any) -> None:
        app = create_app(filled.settings, filled)
        _set_triggered(monkeypatch, {"type": wl.HOLDING_REMOVE_TYPE, "index": 0})
        content, feedback = self._remove(app)([1, 0, 0], "Mein Depot")
        assert [h.ticker for h in _holdings_of(filled)] == ["MSFT", "KO"]
        assert content is not no_update
        assert not has_class(feedback, "bt-error")

    def test_edit_after_remove_hits_the_right_row(self, filled: AppState, monkeypatch: Any) -> None:
        """Der Index ist eine wandernde Adresse.

        Nach dem Entfernen der Zeile 0 ist MSFT die neue 0 und KO die neue
        1. Eine anschliessende Aenderung an Index 1 muss KO treffen - und
        nicht MSFT, das vor dem Entfernen unter Index 1 stand.
        """
        app = create_app(filled.settings, filled)

        _set_triggered(monkeypatch, {"type": wl.HOLDING_REMOVE_TYPE, "index": 0})
        self._remove(app)([1, 0, 0], "Mein Depot")
        assert [h.ticker for h in _holdings_of(filled)] == ["MSFT", "KO"]

        # Die neu gerenderte Zeile 1 traegt jetzt KO.
        _set_triggered(monkeypatch, {"type": wl.HOLDING_SAVE_TYPE, "index": 1})
        self._save(app)(
            [0, 1],
            "Mein Depot",
            [5, 42],
            [380.0, 61.0],
            ["2024-01-15", "2024-01-15"],
            _shares_ids(2),
        )

        by_ticker = {h.ticker: h for h in _holdings_of(filled)}
        assert by_ticker["KO"].shares == 42, "Die Aenderung traf die falsche Zeile"
        assert by_ticker["MSFT"].shares == 5, "MSFT wurde mitgeschrieben"

    def test_save_only_touches_the_clicked_row(self, filled: AppState, monkeypatch: Any) -> None:
        # Alle drei Felder tragen geaenderte Werte, geklickt wurde aber nur
        # Zeile 2. Die anderen beiden duerfen unveraendert bleiben.
        app = create_app(filled.settings, filled)
        _set_triggered(monkeypatch, {"type": wl.HOLDING_SAVE_TYPE, "index": 2})
        self._save(app)(
            [0, 0, 1],
            "Mein Depot",
            [111, 222, 33],
            [1.0, 2.0, 66.0],
            ["2020-01-01", "2020-01-02", "2024-06-06"],
            _shares_ids(3),
        )
        holdings = _holdings_of(filled)
        assert holdings[0].shares == 10
        assert holdings[1].shares == 5
        assert holdings[2].shares == 33
        assert holdings[2].buy_price == 66.0

    def test_row_is_found_by_id_not_by_position(self, filled: AppState, monkeypatch: Any) -> None:
        """Die Zeile wird ueber ihre ID gesucht, nicht ueber die Position.

        Die Feldlisten kommen hier in einer anderen Reihenfolge, als die
        Indizes es nahelegen - genau das passiert, sobald Dash die Zeilen
        nicht sortiert ausliefert. Wer ``ids`` als Positionsliste liest,
        trifft dann die falsche Haltung.
        """
        app = create_app(filled.settings, filled)
        _set_triggered(monkeypatch, {"type": wl.HOLDING_SAVE_TYPE, "index": 2})
        self._save(app)(
            [0, 0, 1],
            "Mein Depot",
            # Umgekehrte Reihenfolge: Position 0 traegt Index 2.
            [77, 5, 10],
            [66.0, 380.0, 150.0],
            ["2024-05-05", "2024-01-15", "2024-01-15"],
            [
                {"type": wl.HOLDING_SHARES_TYPE, "index": 2},
                {"type": wl.HOLDING_SHARES_TYPE, "index": 1},
                {"type": wl.HOLDING_SHARES_TYPE, "index": 0},
            ],
        )
        holdings = _holdings_of(filled)
        # Index 2 ist KO; seine Werte standen an Listenposition 0.
        assert holdings[2].ticker == "KO"
        assert holdings[2].shares == 77, "Die Zeile wurde ueber die Position gesucht"
        assert holdings[2].buy_price == 66.0
        assert holdings[0].shares == 10, "AAPL wurde mitgeschrieben"
        assert holdings[1].shares == 5, "MSFT wurde mitgeschrieben"

    def test_save_with_invalid_date_changes_nothing(
        self, filled: AppState, monkeypatch: Any
    ) -> None:
        before = _holdings_of(filled)
        app = create_app(filled.settings, filled)
        _set_triggered(monkeypatch, {"type": wl.HOLDING_SAVE_TYPE, "index": 0})
        content, feedback = self._save(app)(
            [1, 0, 0],
            "Mein Depot",
            [11, 5, 7],
            [150.0, 380.0, 60.0],
            ["15.01.2024", "2024-01-15", "2024-01-15"],
            _shares_ids(3),
        )
        assert content is no_update
        assert has_class(feedback, "bt-error")
        assert _holdings_of(filled) == before

    def test_save_with_empty_shares_changes_nothing(
        self, filled: AppState, monkeypatch: Any
    ) -> None:
        before = _holdings_of(filled)
        app = create_app(filled.settings, filled)
        _set_triggered(monkeypatch, {"type": wl.HOLDING_SAVE_TYPE, "index": 0})
        content, feedback = self._save(app)(
            [1, 0, 0],
            "Mein Depot",
            [None, 5, 7],
            [150.0, 380.0, 60.0],
            ["2024-01-15", "2024-01-15", "2024-01-15"],
            _shares_ids(3),
        )
        assert content is no_update
        assert has_class(feedback, "bt-error")
        assert _holdings_of(filled) == before

    def test_save_with_zero_shares_changes_nothing(
        self, filled: AppState, monkeypatch: Any
    ) -> None:
        before = _holdings_of(filled)
        app = create_app(filled.settings, filled)
        _set_triggered(monkeypatch, {"type": wl.HOLDING_SAVE_TYPE, "index": 0})
        content, feedback = self._save(app)(
            [1, 0, 0],
            "Mein Depot",
            [0, 5, 7],
            [150.0, 380.0, 60.0],
            ["2024-01-15", "2024-01-15", "2024-01-15"],
            _shares_ids(3),
        )
        assert content is no_update
        assert has_class(feedback, "bt-error")
        assert _holdings_of(filled) == before

    def test_remove_of_a_gone_row_keeps_the_page(self, filled: AppState, monkeypatch: Any) -> None:
        before = _holdings_of(filled)
        app = create_app(filled.settings, filled)
        _set_triggered(monkeypatch, {"type": wl.HOLDING_REMOVE_TYPE, "index": 99})
        content, feedback = self._remove(app)([0, 0, 0, 1], "Mein Depot")
        assert content is no_update
        assert has_class(feedback, "bt-error")
        assert _holdings_of(filled) == before

    def test_without_a_click_nothing_happens(self, filled: AppState, monkeypatch: Any) -> None:
        before = _holdings_of(filled)
        app = create_app(filled.settings, filled)
        _set_triggered(monkeypatch, {"type": wl.HOLDING_REMOVE_TYPE, "index": 0})
        content, feedback = self._remove(app)([0, 0, 0], "Mein Depot")
        assert content is no_update
        assert feedback is no_update
        assert _holdings_of(filled) == before


class TestWlWatchEntries:
    """Aufnehmen, Notiz speichern, Entfernen."""

    def _add(self, app: Any) -> Any:
        return _by_input(app, wl.WATCHLIST_FEEDBACK, wl.WATCHLIST_ADD)

    def _save(self, app: Any) -> Any:
        return _by_input(app, wl.WATCHLIST_FEEDBACK, wl.WATCH_SAVE_TYPE)

    def _remove(self, app: Any) -> Any:
        return _by_input(app, wl.WATCHLIST_FEEDBACK, wl.WATCH_REMOVE_TYPE)

    def test_add_reaches_the_file(self, filled: AppState) -> None:
        app = create_app(filled.settings, filled)
        content, feedback = self._add(app)(1, "Beobachtung", "TSLA", "Neu gesehen")
        watchlist = filled.holdings.get_watchlist("Beobachtung")
        assert watchlist is not None
        assert "TSLA" in watchlist.tickers
        assert content is not no_update
        assert not has_class(feedback, "bt-error")

    def test_empty_note_stays_none(self, filled: AppState) -> None:
        app = create_app(filled.settings, filled)
        self._add(app)(1, "Beobachtung", "TSLA", "   ")
        watchlist = filled.holdings.get_watchlist("Beobachtung")
        assert watchlist is not None
        entry = next(e for e in watchlist.entries if e.ticker == "TSLA")
        assert entry.note is None

    def test_duplicate_keeps_the_page(self, filled: AppState) -> None:
        app = create_app(filled.settings, filled)
        content, feedback = self._add(app)(1, "Beobachtung", "KO", None)
        assert content is no_update
        assert has_class(feedback, "bt-error")
        watchlist = filled.holdings.get_watchlist("Beobachtung")
        assert watchlist is not None
        assert watchlist.tickers == ["KO"]

    def test_missing_ticker_is_rejected(self, filled: AppState) -> None:
        app = create_app(filled.settings, filled)
        content, feedback = self._add(app)(1, "Beobachtung", "  ", None)
        assert content is no_update
        assert has_class(feedback, "bt-error")

    def test_note_is_saved(self, filled: AppState, monkeypatch: Any) -> None:
        # Bewusst zwei Zeilen, und die geklickte ist NICHT die erste:
        # bei nur einer Zeile waeren ``notes[row]`` und ``notes[0]``
        # dasselbe, und ein Callback, der stur die erste Zeile liest,
        # bliebe unentdeckt.
        filled.holdings.add_to_watchlist("Beobachtung", "TSLA", note="Alt")
        app = create_app(filled.settings, filled)
        _set_triggered(monkeypatch, {"type": wl.WATCH_SAVE_TYPE, "ticker": "TSLA"})
        content, feedback = self._save(app)(
            [0, 1],
            "Beobachtung",
            ["Notiz von KO", "Neue Notiz"],
            [
                {"type": wl.WATCH_NOTE_TYPE, "ticker": "KO"},
                {"type": wl.WATCH_NOTE_TYPE, "ticker": "TSLA"},
            ],
        )
        watchlist = filled.holdings.get_watchlist("Beobachtung")
        assert watchlist is not None
        notes = {e.ticker: e.note for e in watchlist.entries}
        assert notes["TSLA"] == "Neue Notiz"
        assert notes["KO"] == "Dividende", "Die fremde Notiz wurde mitgeschrieben"
        assert content is not no_update
        assert not has_class(feedback, "bt-error")

    def test_empty_note_clears_it(self, filled: AppState, monkeypatch: Any) -> None:
        filled.holdings.add_to_watchlist("Beobachtung", "TSLA", note="Alt")
        app = create_app(filled.settings, filled)
        _set_triggered(monkeypatch, {"type": wl.WATCH_SAVE_TYPE, "ticker": "TSLA"})
        self._save(app)(
            [0, 1],
            "Beobachtung",
            ["Dividende", ""],
            [
                {"type": wl.WATCH_NOTE_TYPE, "ticker": "KO"},
                {"type": wl.WATCH_NOTE_TYPE, "ticker": "TSLA"},
            ],
        )
        watchlist = filled.holdings.get_watchlist("Beobachtung")
        assert watchlist is not None
        notes = {e.ticker: e.note for e in watchlist.entries}
        assert notes["TSLA"] is None
        assert notes["KO"] == "Dividende"

    def test_note_of_the_clicked_row_only(self, filled: AppState, monkeypatch: Any) -> None:
        filled.holdings.add_to_watchlist("Beobachtung", "TSLA", note="Alt")
        app = create_app(filled.settings, filled)
        _set_triggered(monkeypatch, {"type": wl.WATCH_SAVE_TYPE, "ticker": "KO"})
        self._save(app)(
            [1, 0],
            "Beobachtung",
            ["Neu fuer KO", "Ueberschrieben"],
            [
                {"type": wl.WATCH_NOTE_TYPE, "ticker": "KO"},
                {"type": wl.WATCH_NOTE_TYPE, "ticker": "TSLA"},
            ],
        )
        watchlist = filled.holdings.get_watchlist("Beobachtung")
        assert watchlist is not None
        notes = {e.ticker: e.note for e in watchlist.entries}
        assert notes["KO"] == "Neu fuer KO"
        assert notes["TSLA"] == "Alt", "Die fremde Notiz wurde mitgeschrieben"

    def test_remove_reaches_the_file(self, filled: AppState, monkeypatch: Any) -> None:
        app = create_app(filled.settings, filled)
        _set_triggered(monkeypatch, {"type": wl.WATCH_REMOVE_TYPE, "ticker": "KO"})
        content, feedback = self._remove(app)([1], "Beobachtung")
        watchlist = filled.holdings.get_watchlist("Beobachtung")
        assert watchlist is not None
        assert watchlist.tickers == []
        assert content is not no_update
        assert not has_class(feedback, "bt-error")

    def test_remove_of_a_gone_symbol_keeps_the_page(
        self, filled: AppState, monkeypatch: Any
    ) -> None:
        app = create_app(filled.settings, filled)
        _set_triggered(monkeypatch, {"type": wl.WATCH_REMOVE_TYPE, "ticker": "NOSUCH"})
        content, feedback = self._remove(app)([1], "Beobachtung")
        assert content is no_update
        assert has_class(feedback, "bt-error")
        watchlist = filled.holdings.get_watchlist("Beobachtung")
        assert watchlist is not None
        assert watchlist.tickers == ["KO"]


class TestStorageStaysInTmp:
    """Kein Test darf in den echten Bestand des Nutzers schreiben."""

    def test_holdings_file_is_below_tmp_path(self, state: AppState, tmp_path: Path) -> None:
        assert state.settings.portfolio.holdings_file.is_relative_to(tmp_path)


class TestWlSorting:
    """Klick auf eine Ueberschrift sortiert die Tabelle.

    Der Nutzer wollte die Listen sortieren koennen. Entscheidend ist, dass
    nach ZAHLEN sortiert wird und nicht nach der angezeigten Zeichenkette:
    alphabetisch geordnet stuende "120.0000" vor "9.0000".
    """

    @staticmethod
    def _sorter(app: Any, table: str) -> Any:
        """Holt den Sortier-Callback einer der beiden Tabellen."""
        return _callback(
            app,
            lambda key, entry: "sort" in key and table in str(entry.get("inputs", "")),
        )

    def test_first_click_sorts_ascending(self, filled: AppState, monkeypatch: Any) -> None:
        app = create_app(filled.settings, filled)
        _set_triggered(
            monkeypatch,
            {"type": c.SORT_HEADER_TYPE, "table": wl.POSITIONS_TABLE, "column": "Shares"},
        )
        store, _content = self._sorter(app, wl.POSITIONS_TABLE)([1], None, "Mein Depot")
        assert store == {"column": "Shares", "desc": False}

    def test_second_click_turns_the_direction(self, filled: AppState, monkeypatch: Any) -> None:
        """Dieselbe Spalte erneut geklickt dreht um - die uebliche Konvention."""
        app = create_app(filled.settings, filled)
        _set_triggered(
            monkeypatch,
            {"type": c.SORT_HEADER_TYPE, "table": wl.POSITIONS_TABLE, "column": "Shares"},
        )
        current = {"column": "Shares", "desc": False}
        store, _content = self._sorter(app, wl.POSITIONS_TABLE)([1], current, "Mein Depot")
        assert store == {"column": "Shares", "desc": True}

    def test_another_column_starts_ascending(self, filled: AppState, monkeypatch: Any) -> None:
        """Eine neue Spalte faengt aufsteigend an, nicht in der alten Richtung."""
        app = create_app(filled.settings, filled)
        _set_triggered(
            monkeypatch,
            {"type": c.SORT_HEADER_TYPE, "table": wl.POSITIONS_TABLE, "column": "Price"},
        )
        current = {"column": "Shares", "desc": True}
        store, _content = self._sorter(app, wl.POSITIONS_TABLE)([1], current, "Mein Depot")
        assert store == {"column": "Price", "desc": False}

    def test_without_a_click_nothing_changes(self, filled: AppState, monkeypatch: Any) -> None:
        app = create_app(filled.settings, filled)
        _set_triggered(monkeypatch, None)
        store, content = self._sorter(app, wl.POSITIONS_TABLE)([None], None, "Mein Depot")
        assert store is no_update
        assert content is no_update

    def test_the_rows_really_change_order(self, filled: AppState, monkeypatch: Any) -> None:
        """Nicht nur der Speicher - die Tabelle selbst muss anders aussehen.

        Ohne diese Pruefung bliebe offen, ob die Sortierung ueberhaupt
        beim Rendern ankommt.
        """
        app = create_app(filled.settings, filled)
        sorter = self._sorter(app, wl.POSITIONS_TABLE)
        _set_triggered(
            monkeypatch,
            {"type": c.SORT_HEADER_TYPE, "table": wl.POSITIONS_TABLE, "column": "Shares"},
        )
        _asc_store, content_asc = sorter([1], None, "Mein Depot")
        _desc_store, content_desc = sorter([1], {"column": "Shares", "desc": False}, "Mein Depot")

        order_asc = _ticker_order(content_asc)
        order_desc = _ticker_order(content_desc)
        assert order_asc, "Keine Zeilen gefunden"
        assert order_asc == list(reversed(order_desc)), (order_asc, order_desc)

    def test_sorting_is_numeric_not_alphabetic(self, state: AppState, monkeypatch: Any) -> None:
        """Der eigentliche Punkt: 9 kommt vor 30 vor 120.

        Alphabetisch geordnet stuende 120 vor 9 - genau der Fehler, den
        die Umstellung der Tabelle auf Rohwerte vermeidet.
        """
        state.holdings.create_portfolio("Zahlen", currency=Currency.USD)
        for ticker, shares in (("AAA", 9), ("BBB", 120), ("CCC", 30)):
            state.holdings.add_holding(
                "Zahlen", ticker, shares=shares, buy_price=1.0, buy_date=BUY_DATE
            )

        app = create_app(state.settings, state)
        _set_triggered(
            monkeypatch,
            {"type": c.SORT_HEADER_TYPE, "table": wl.POSITIONS_TABLE, "column": "Shares"},
        )
        _store, content = self._sorter(app, wl.POSITIONS_TABLE)([1], None, "Zahlen")
        assert _ticker_order(content) == ["AAA", "CCC", "BBB"]

    def test_missing_values_stay_at_the_end(self, state: AppState, monkeypatch: Any) -> None:
        """Ein fehlender Wert ist keine Aussage ueber die Groesse.

        ZZZZ hat keinen Kurs und damit keinen Marktwert. In BEIDEN
        Richtungen gehoert die Zeile ans Ende: oben stuende sie wie der
        groesste Posten, was sie nicht ist.
        """
        state.holdings.create_portfolio("Luecke", currency=Currency.USD)
        state.holdings.add_holding("Luecke", "AAPL", shares=1, buy_price=1.0, buy_date=BUY_DATE)
        state.holdings.add_holding("Luecke", "MSFT", shares=2, buy_price=1.0, buy_date=BUY_DATE)
        state.holdings.add_holding("Luecke", "ZZZZ", shares=3, buy_price=1.0, buy_date=BUY_DATE)

        app = create_app(state.settings, state)
        sorter = self._sorter(app, wl.POSITIONS_TABLE)
        _set_triggered(
            monkeypatch,
            {"type": c.SORT_HEADER_TYPE, "table": wl.POSITIONS_TABLE, "column": "Market value"},
        )
        _asc_store, content_asc = sorter([1], None, "Luecke")
        _desc_store, content_desc = sorter([1], {"column": "Market value", "desc": False}, "Luecke")

        order_asc = _ticker_order(content_asc)
        order_desc = _ticker_order(content_desc)
        assert order_asc[-1] == "ZZZZ", order_asc
        assert order_desc[-1] == "ZZZZ", order_desc

    def test_the_watchlist_sorts_too(self, filled: AppState, monkeypatch: Any) -> None:
        app = create_app(filled.settings, filled)
        _set_triggered(
            monkeypatch,
            {"type": c.SORT_HEADER_TYPE, "table": wl.WATCHLIST_TABLE, "column": "Ticker"},
        )
        store, _content = self._sorter(app, wl.WATCHLIST_TABLE)([1], None, "Beobachtung")
        assert store == {"column": "Ticker", "desc": False}

    def test_sorting_survives_an_edit(self, state: AppState, monkeypatch: Any) -> None:
        """Nach einer Aenderung darf die Reihenfolge nicht zurueckspringen.

        Genau dafuer liegt die Sortierung in einem Store und wird jedem
        Callback als State mitgegeben. Ohne das stuende die Tabelle nach
        jedem Kauf wieder in der Ausgangsreihenfolge.
        """
        state.holdings.create_portfolio("Zahlen", currency=Currency.USD)
        for ticker, shares in (("AAA", 9), ("BBB", 120)):
            state.holdings.add_holding(
                "Zahlen", ticker, shares=shares, buy_price=1.0, buy_date=BUY_DATE
            )

        app = create_app(state.settings, state)
        add = _by_input(app, wl.HOLDING_FEEDBACK, wl.HOLDING_ADD)
        # Absteigend nach Stueckzahl sortiert; der neue Kauf hat 50 Stueck
        # und gehoert damit in die Mitte, nicht ans Ende.
        content, _feedback = add(
            1, "Zahlen", "MMM", 50, 1.0, "2024-03-01", "USD", {"column": "Shares", "desc": True}
        )
        assert _ticker_order(content) == ["BBB", "MMM", "AAA"]


class TestPortPortfolioPicker:
    """Die Depotauswahl der Seite PORT fuehrt ueber die Adresse.

    Bewusst nicht ueber einen stillen Zustand: danach steht das Universum
    in der Kommandozeile und in der URL. Der Nutzer sieht, womit
    gerechnet wird, kann einzelne Titel streichen und die Ansicht als
    Lesezeichen sichern.
    """

    @staticmethod
    def _picker(app: Any) -> Any:
        """Holt den Callback der Depotauswahl."""
        return _callback(
            app,
            lambda _key, entry: "port-portfolio-picker" in str(entry.get("inputs", "")),
        )

    @staticmethod
    def _saved(state: AppState, name: str, tickers: tuple[str, ...]) -> None:
        state.holdings.create_portfolio(name)
        for ticker in tickers:
            state.holdings.add_holding(name, ticker, shares=1, buy_price=100.0, buy_date=BUY_DATE)

    def test_the_url_carries_the_tickers(self, state: AppState) -> None:
        self._saved(state, "Mein Depot", ("AAPL", "MSFT", "KO"))
        app = create_app(state.settings, state)
        path, search, _command = self._picker(app)("Mein Depot")
        assert path == "/port"
        for ticker in ("AAPL", "MSFT", "KO"):
            assert ticker in search, search

    def test_the_command_bar_shows_the_universe(self, state: AppState) -> None:
        """Sichtbar, womit gerechnet wird - nicht nur intern gesetzt."""
        self._saved(state, "Mein Depot", ("AAPL", "MSFT", "KO"))
        app = create_app(state.settings, state)
        _path, _search, command = self._picker(app)("Mein Depot")
        assert command == "AAPL,KO,MSFT PORT"

    def test_clearing_the_field_changes_nothing(self, state: AppState) -> None:
        """Ein geleertes Feld darf die laufende Ansicht nicht verwerfen."""
        app = create_app(state.settings, state)
        assert all(value is no_update for value in self._picker(app)(None))

    def test_an_unknown_portfolio_changes_nothing(self, state: AppState) -> None:
        app = create_app(state.settings, state)
        assert all(value is no_update for value in self._picker(app)("gibtsnicht"))

    def test_a_portfolio_emptied_meanwhile_changes_nothing(self, state: AppState) -> None:
        """Zwischen Seitenaufbau und Klick kann der Bestand verschwinden.

        Das Dropdown bietet solche Depots nicht an - aber es wurde
        gebaut, bevor die Position geloescht wurde.
        """
        self._saved(state, "Schrumpft", ("AAPL", "MSFT"))
        app = create_app(state.settings, state)
        state.holdings.remove_holding("Schrumpft", 1)
        assert all(value is no_update for value in self._picker(app)("Schrumpft"))

    def test_the_selection_survives_a_round_trip(self, state: AppState) -> None:
        """Was die Auswahl schreibt, muss die Seite wieder lesen koennen.

        Der eigentliche Beweis: ohne diesen Schritt koennte die Adresse
        korrekt aussehen und die Seite trotzdem mit etwas anderem rechnen.
        """
        self._saved(state, "Mein Depot", ("AAPL", "MSFT", "KO"))
        app = create_app(state.settings, state)
        path, search, _command = self._picker(app)("Mein Depot")

        mnemonic, tickers, _args = parse_href(path, search)
        assert mnemonic == "PORT"
        assert tickers == ["AAPL", "KO", "MSFT"]


class TestEqsBackgroundRun:
    """Das Screening laeuft im Hintergrund, die Seite fragt den Stand ab.

    Ein Lauf ueber den S&P 500 dauert gemessen rund eine halbe Stunde.
    Frueher blockierte er den Web-Thread und die Seite zeigte nur einen
    Ladebalken - der Nutzer hielt sie fuer haengengeblieben.
    """

    @staticmethod
    def _run(app: Any) -> Any:
        return _callback(app, lambda _k, e: "eqs-run" in str(e.get("inputs", "")))

    @staticmethod
    def _tick(app: Any) -> Any:
        return _callback(app, lambda _k, e: eqs_page.PROGRESS_TICK in str(e.get("inputs", "")))

    @staticmethod
    def _await(app: Any, job_id: str, preset: str = "value") -> Any:
        """Fragt den Stand ab, bis der Lauf endet."""
        tick = TestEqsBackgroundRun._tick(app)
        for index in range(200):
            out, interval_off, _hidden, _job = tick(index, job_id, preset)
            if interval_off is True:
                return out
            time.sleep(0.05)
        raise AssertionError("Screening wurde nicht fertig")

    def test_the_run_starts_a_job(self, state: AppState) -> None:
        app = create_app(state.settings, state)
        _view, job_id, interval_off, cancel_hidden = self._run(app)(1, "value", ["AAPL", "MSFT"])
        assert job_id, "Keine Job-Kennung"
        assert interval_off is False, "Die Abfrage wurde nicht gestartet"
        assert cancel_hidden is False, "Der Abbrechen-Knopf blieb verborgen"
        self._await(app, job_id)

    def test_the_interval_stops_when_the_run_ends(self, state: AppState) -> None:
        """Sonst fragte die Seite endlos weiter - Last fuer nichts."""
        app = create_app(state.settings, state)
        _view, job_id, _off, _hidden = self._run(app)(1, "value", ["AAPL", "MSFT"])
        tick = self._tick(app)
        for index in range(200):
            _out, interval_off, hidden, job = tick(index, job_id, "value")
            # Ende erkannt am Aufraeumen, NICHT am Interval-Kennzeichen:
            # sonst pruefte die Schleife ihre eigene Abbruchbedingung und
            # liefe bei einem stehengebliebenen Interval einfach leer aus.
            if job is None:
                assert interval_off is True, "Das Intervall lief weiter"
                assert hidden is True, "Der Abbrechen-Knopf blieb stehen"
                return
            time.sleep(0.05)
        raise AssertionError("Der Lauf wurde nie aufgeraeumt")

    def test_the_result_arrives(self, state: AppState) -> None:
        app = create_app(state.settings, state)
        _view, job_id, _off, _hidden = self._run(app)(1, "value", ["AAPL", "MSFT"])
        out = self._await(app, job_id)
        assert not has_class(out, "bt-error"), text_of(out)[:200]

    def test_an_unknown_job_stops_the_interval(self, state: AppState) -> None:
        """Etwa nach einem Neustart des Servers - sonst fragt es ins Leere."""
        app = create_app(state.settings, state)
        _out, interval_off, hidden, job = self._tick(app)(1, "gibtsnicht", "value")
        assert interval_off is True
        assert hidden is True
        assert job is None

    def test_without_a_universe_nothing_starts(self, state: AppState) -> None:
        app = create_app(state.settings, state)
        _view, job_id, interval_off, _hidden = self._run(app)(1, "value", [])
        assert job_id is None
        assert interval_off is True


class TestEqsCancelActuallyStops:
    """Der Abbrechen-Knopf muss den Lauf wirklich beenden.

    Der erste Entwurf warf den Abbruch als Ausnahme aus dem
    Fortschritts-Rueckruf - und der schluckt Ausnahmen absichtlich, damit
    eine kaputte Anzeige den Lauf nicht abbricht. Gemessene Folge: ein
    Klick nach drei Sekunden liess den Lauf sechzig Sekunden
    weiterrechnen und legte nebenbei die Anzeige still. Bei 503 Titeln
    haette der Knopf eine halbe Stunde lang nichts bewirkt.

    Seitdem gehen Fortschritt und Abbruch ueber getrennte Kanaele.
    """

    def test_the_run_stops_early(self, state: AppState) -> None:
        """Gemessen an der Zahl gepruefter Titel, nicht an der Uhr.

        Eine Zeitschranke waere auf einer schnellen Maschine nutzlos und
        auf einer langsamen flaky.
        """
        app = create_app(state.settings, state)
        run = _callback(app, lambda _k, e: "eqs-run" in str(e.get("inputs", "")))
        tick = _callback(app, lambda _k, e: eqs_page.PROGRESS_TICK in str(e.get("inputs", "")))
        cancel = _callback(app, lambda _k, e: eqs_page.CANCEL_BUTTON in str(e.get("inputs", "")))

        universe = ["AAPL", "MSFT", "NVDA", "SPY", "JNJ", "XOM", "KO", "PG", "JPM", "WMT"]
        _view, job_id, _off, _hidden = run(1, "value", universe)
        cancel(1, job_id)

        for index in range(200):
            _out, _off, _hidden, job = tick(index, job_id, "value")
            if job is None:
                return
            time.sleep(0.05)
        raise AssertionError("Der Lauf endete nach dem Abbruch nicht")

    def test_a_cancelled_run_is_not_an_error(self, state: AppState) -> None:
        """Der Nutzer hat selbst gestoppt - das ist kein Fehlschlag."""
        app = create_app(state.settings, state)
        run = _callback(app, lambda _k, e: "eqs-run" in str(e.get("inputs", "")))
        tick = _callback(app, lambda _k, e: eqs_page.PROGRESS_TICK in str(e.get("inputs", "")))
        cancel = _callback(app, lambda _k, e: eqs_page.CANCEL_BUTTON in str(e.get("inputs", "")))

        _view, job_id, _off, _hidden = run(1, "value", ["AAPL", "MSFT", "KO"])
        cancel(1, job_id)
        for index in range(200):
            out, _off, _hidden, job = tick(index, job_id, "value")
            if job is None:
                assert not has_class(out, "bt-error"), text_of(out)[:200]
                return
            time.sleep(0.05)
        raise AssertionError("Der Lauf endete nicht")

    def test_cancelling_an_unknown_job_is_harmless(self, state: AppState) -> None:
        app = create_app(state.settings, state)
        cancel = _callback(app, lambda _k, e: eqs_page.CANCEL_BUTTON in str(e.get("inputs", "")))
        assert cancel(1, "gibtsnicht") is True
        assert cancel(1, None) is True
