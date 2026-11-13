"""Tests der Seite WL und ihrer Anbindung an den Depot-Speicher.

Gerendert wird ueber ``render_page``, also denselben Weg, den die App zur
Laufzeit nimmt. Ein Test, der ``layout`` direkt aufruft, wuerde die
Fehlerbehandlung der Shell ueberspringen.

Die Depots hier laufen bewusst auf **USD**, obwohl der Default Euro ist:
die Fixture-Kurse notieren in Dollar, und diese Tests pruefen die
Darstellung, nicht die Waehrungsumrechnung. Ohne die explizite Angabe
braeuchte jede Zeile zusaetzlich einen Wechselkurs, und ein fehlender
liesse die Tests an der falschen Stelle scheitern.

Der Depot-Speicher zeigt in jedem Test auf ein temporaeres Verzeichnis.
Niemals darf eine Testausfuehrung in den echten Bestand des Nutzers unter
``~/.btlite`` schreiben.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import TYPE_CHECKING, Any

import pytest
from dash import dcc
from dash.development.base_component import Component

from btlite.config import Settings
from btlite.core.errors import ProviderUnavailableError
from btlite.models.common import Currency
from btlite.ui.app import render_page
from btlite.ui.command_bar import MNEMONICS, TICKERLESS_MNEMONICS
from btlite.ui.pages import PAGES, ROUTES, wl
from btlite.ui.state import AppState

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

BUY_DATE = dt.date(2024, 1, 15)


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppState:
    """Zustand mit Depot-Datei im temporaeren Verzeichnis."""
    # Die Earnings-Fixtures liegen im Oktober 2026. Die Anzeige und
    # Sortierung brauchen einen festen Bezugstag vor diesen Terminen.
    monkeypatch.setattr(
        "btlite.core.clock.utc_now", lambda: dt.datetime(2026, 10, 1, tzinfo=dt.UTC)
    )
    monkeypatch.setenv("BTLITE_PORTFOLIO_HOLDINGS_FILE", str(tmp_path / "holdings.json"))
    return AppState(Settings())


@pytest.fixture
def filled(state: AppState) -> AppState:
    """Zustand mit einem Depot und einer Watchlist."""
    state.holdings.create_portfolio("Mein Depot", currency=Currency.USD)
    state.holdings.add_holding("Mein Depot", "AAPL", shares=10, buy_price=150.0, buy_date=BUY_DATE)
    state.holdings.add_holding("Mein Depot", "AAPL", shares=3, buy_price=210.0, buy_date=BUY_DATE)
    state.holdings.add_holding("Mein Depot", "MSFT", shares=5, buy_price=380.0, buy_date=BUY_DATE)
    state.holdings.create_watchlist("Beobachtung")
    state.holdings.add_to_watchlist("Beobachtung", "KO", note="Dividende")
    return state


def _render(state: AppState, args: tuple[str, ...] = ()) -> str:
    return str(render_page(state, "WL", [], args))


class TestRegistration:
    def test_page_is_registered(self):
        assert "WL" in PAGES
        assert "/wl" in ROUTES

    def test_mnemonic_is_documented(self):
        """Ohne Eintrag in MNEMONICS kennt die Kommandozeile das Kuerzel nicht."""
        assert "WL" in MNEMONICS
        assert MNEMONICS["WL"]

    def test_wl_needs_no_ticker(self):
        """Der Bestand kommt aus der Datei, nicht aus der Eingabe."""
        assert "WL" in TICKERLESS_MNEMONICS

    def test_command_parses_without_ticker(self):
        from btlite.ui.command_bar import parse_command

        command = parse_command("WL GO")
        assert command.mnemonic == "WL"
        assert list(command.tickers) == []


class TestEmptyState:
    def test_renders_without_error_banner(self, state: AppState):
        assert "bt-error" not in _render(state)

    def test_explains_how_to_start(self, state: AppState):
        """Eine leere Seite waere eine Sackgasse."""
        rendered = _render(state)
        assert "No portfolio saved yet" in rendered
        assert "No watchlist saved yet" in rendered


class TestFilledPortfolio:
    def test_renders_without_error_banner(self, filled: AppState):
        assert "bt-error" not in _render(filled)

    def test_shows_both_tickers(self, filled: AppState):
        rendered = _render(filled)
        assert "AAPL" in rendered
        assert "MSFT" in rendered

    def test_repeated_buys_appear_once(self, filled: AppState):
        """Zwei AAPL-Kaeufe, eine Zeile mit 13 Stueck."""
        assert "13.0000" in _render(filled)

    def test_cost_basis_is_the_sum_of_all_buys(self, filled: AppState):
        # 10*150 + 3*210 + 5*380 = 4030
        assert "4,030.00" in _render(filled)

    def test_direction_is_not_conveyed_by_colour_alone(self, filled: AppState):
        """Das CSS verlangt zusaetzlich ein Zeichen im Text."""
        rendered = _render(filled)
        assert "▲" in rendered or "▼" in rendered

    def test_watchlist_entry_and_note_appear(self, filled: AppState):
        rendered = _render(filled)
        assert "KO" in rendered
        assert "Dividende" in rendered


class TestMissingQuote:
    def test_no_total_when_a_position_cannot_be_priced(self, state: AppState):
        """Eine Teilsumme als Depotwert waere eine Falschaussage."""
        state.holdings.create_portfolio("Gemischt", currency=Currency.USD)
        state.holdings.add_holding(
            "Gemischt", "AAPL", shares=10, buy_price=150.0, buy_date=BUY_DATE
        )
        state.holdings.add_holding("Gemischt", "ZZZZ", shares=5, buy_price=50.0, buy_date=BUY_DATE)
        rendered = _render(state)
        assert "bt-error" not in rendered
        assert "ZZZZ" in rendered
        # Der bekannte Einstand bleibt sichtbar.
        assert "1,750.00" in rendered

    def test_names_the_unpriced_ticker(self, state: AppState):
        state.holdings.create_portfolio("Gemischt", currency=Currency.USD)
        state.holdings.add_holding("Gemischt", "AAPL", shares=1, buy_price=150.0, buy_date=BUY_DATE)
        state.holdings.add_holding("Gemischt", "ZZZZ", shares=1, buy_price=50.0, buy_date=BUY_DATE)
        assert "No price is available for ZZZZ" in _render(state)


class TestSelection:
    def test_named_portfolio_is_selected(self, state: AppState):
        state.holdings.create_portfolio("Erstes", currency=Currency.USD)
        state.holdings.create_portfolio("Zweites", currency=Currency.USD)
        state.holdings.add_holding("Zweites", "AAPL", shares=1, buy_price=1.0, buy_date=BUY_DATE)
        assert "AAPL" in _render(state, ("Zweites",))

    def test_selection_ignores_case(self, state: AppState):
        state.holdings.create_portfolio("Erstes", currency=Currency.USD)
        state.holdings.create_portfolio("Zweites", currency=Currency.USD)
        state.holdings.add_holding("Zweites", "AAPL", shares=1, buy_price=1.0, buy_date=BUY_DATE)
        assert "AAPL" in _render(state, ("zweites",))

    def test_unknown_name_falls_back_to_the_first(self, state: AppState):
        state.holdings.create_portfolio("Erstes", currency=Currency.USD)
        state.holdings.add_holding("Erstes", "MSFT", shares=1, buy_price=1.0, buy_date=BUY_DATE)
        assert "MSFT" in _render(state, ("gibt-es-nicht",))


class TestPersistence:
    def test_holdings_survive_a_new_state(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Der Bestand liegt in einer Datei, nicht im Prozess."""
        monkeypatch.setenv("BTLITE_PORTFOLIO_HOLDINGS_FILE", str(tmp_path / "h.json"))
        first = AppState(Settings())
        first.holdings.create_portfolio("Bleibt", currency=Currency.USD)
        first.holdings.add_holding("Bleibt", "AAPL", shares=7, buy_price=100.0, buy_date=BUY_DATE)

        second = AppState(Settings())
        assert "Bleibt" in second.holdings.list_portfolios()
        assert "AAPL" in _render(second)

    def test_state_does_not_cache_the_store(self, state: AppState):
        """Ein Puffer wuerde gerade gespeicherte Positionen verschwinden lassen."""
        assert "AAPL" not in _render(state)
        state.holdings.create_portfolio("Neu", currency=Currency.USD)
        state.holdings.add_holding("Neu", "AAPL", shares=1, buy_price=100.0, buy_date=BUY_DATE)
        assert "AAPL" in _render(state)


# ---------------------------------------------------------------------------
# Bearbeitbare Oberflaeche
# ---------------------------------------------------------------------------
#
# Diese Tests laufen ueber den Komponentenbaum, nicht ueber den gerenderten
# Text. Ein ``in``-Vergleich auf dem repr faende ``value=10.0`` auch dann,
# wenn die Zahl in einer ganz anderen Zeile stuende - die Zuordnung von
# Index zu Vorbelegung ist aber genau das, was hier zu pruefen ist.


def _walk(node: object) -> Iterator[Any]:
    """Laeuft rekursiv durch einen Dash-Komponentenbaum.

    Args:
        node: Komponente, Liste oder einfacher Wert.

    Yields:
        Jede Komponente des Baums, einschliesslich ``node`` selbst.
    """
    if isinstance(node, list | tuple):
        for item in node:
            yield from _walk(item)
        return
    if not isinstance(node, Component):
        return
    yield node
    yield from _walk(getattr(node, "children", None))


def _tree(state: AppState, args: tuple[str, ...] = ()) -> Any:
    """Rendert die Seite WL als Komponentenbaum.

    Args:
        state: UI-Zustand.
        args: Argumente hinter dem Mnemonic.

    Returns:
        Wurzel des gerenderten Baums.
    """
    return render_page(state, "WL", [], args)


def _by_id(state: AppState, wanted: object, args: tuple[str, ...] = ()) -> list[Any]:
    """Sammelt alle Komponenten mit einer bestimmten ID.

    Args:
        state: UI-Zustand.
        wanted: Gesuchte ID. String fuer feste IDs, Dict fuer Wildcard-IDs.
        args: Argumente hinter dem Mnemonic.

    Returns:
        Passende Komponenten in Dokumentreihenfolge.
    """
    return [n for n in _walk(_tree(state, args)) if getattr(n, "id", None) == wanted]


def _one(state: AppState, wanted: object, args: tuple[str, ...] = ()) -> Any:
    """Holt die einzige Komponente mit dieser ID.

    Args:
        state: UI-Zustand.
        wanted: Gesuchte ID.
        args: Argumente hinter dem Mnemonic.

    Returns:
        Die gefundene Komponente.
    """
    found = _by_id(state, wanted, args)
    assert len(found) == 1, f"ID {wanted!r}: {len(found)} Treffer, erwartet genau 1"
    return found[0]


def _wildcard(state: AppState, kind: str, key: str = "index") -> list[Any]:
    """Sammelt alle Komponenten eines Wildcard-Typs.

    Args:
        state: UI-Zustand.
        kind: Wert des Feldes ``type`` in der ID.
        key: Name des uebrigen Schluesselfeldes.

    Returns:
        Passende Komponenten in Dokumentreihenfolge.
    """
    return [
        n
        for n in _walk(_tree(state))
        if isinstance(getattr(n, "id", None), dict) and n.id.get("type") == kind and key in n.id
    ]


class TestManagementControls:
    """Ohne Anlege-Formular kaeme der Nutzer nie in ein Depot hinein."""

    @pytest.mark.parametrize(
        "element_id",
        [
            wl.PORTFOLIO_NEW_NAME,
            wl.PORTFOLIO_CREATE,
            wl.PORTFOLIO_DELETE,
            wl.PORTFOLIO_FEEDBACK,
            wl.WATCHLIST_NEW_NAME,
            wl.WATCHLIST_CREATE,
            wl.WATCHLIST_DELETE,
            wl.WATCHLIST_FEEDBACK,
        ],
    )
    def test_control_exists_on_an_empty_page(self, state: AppState, element_id: str):
        """Auch ohne jeden Bestand muessen Anlegen und Loeschen erreichbar sein."""
        assert _by_id(state, element_id), f"{element_id} fehlt auf der leeren Seite"

    def test_create_and_delete_are_labelled_in_english(self, state: AppState):
        assert _one(state, wl.PORTFOLIO_CREATE).children == "Create"
        assert _one(state, wl.PORTFOLIO_DELETE).children == "Delete"

    def test_delete_button_says_what_is_lost(self, state: AppState):
        """Ein blosses "Delete" liesse die Folge offen."""
        assert "positions" in _one(state, wl.PORTFOLIO_DELETE).title

    def test_option_helpers_list_the_saved_names(self, filled: AppState):
        assert wl.portfolio_options(filled) == [{"label": "Mein Depot", "value": "Mein Depot"}]
        assert wl.watchlist_options(filled) == [{"label": "Beobachtung", "value": "Beobachtung"}]

    def test_option_helpers_are_empty_without_data(self, state: AppState):
        assert wl.portfolio_options(state) == []
        assert wl.watchlist_options(state) == []


class TestNewHoldingForm:
    """Das Anlege-Formular ist der einzige Weg in ein frisches Depot."""

    @pytest.mark.parametrize(
        "element_id",
        [
            wl.HOLDING_NEW_TICKER,
            wl.HOLDING_NEW_SHARES,
            wl.HOLDING_NEW_PRICE,
            wl.HOLDING_NEW_DATE,
            wl.HOLDING_NEW_CURRENCY,
            wl.HOLDING_ADD,
            wl.HOLDING_FEEDBACK,
        ],
    )
    def test_field_exists_for_a_filled_portfolio(self, filled: AppState, element_id: str):
        assert _by_id(filled, element_id), f"{element_id} fehlt"

    def test_form_is_present_in_an_empty_portfolio(self, state: AppState):
        """Ein leeres Depot ohne Formular waere eine Sackgasse."""
        state.holdings.create_portfolio("Leer", currency=Currency.USD)
        assert "is empty" in str(_tree(state))
        for element_id in (wl.HOLDING_NEW_TICKER, wl.HOLDING_ADD):
            assert _by_id(state, element_id), f"{element_id} fehlt im leeren Depot"

    def test_date_field_states_its_format(self, filled: AppState):
        """Ohne Formatangabe raet der Nutzer zwischen TT.MM. und MM/DD."""
        assert _one(filled, wl.HOLDING_NEW_DATE).placeholder == "YYYY-MM-DD"

    def test_currency_dropdown_carries_the_dark_theme_class(self, filled: AppState):
        """Ohne bt-dropdown steht das Feld auf hellen Dash-Standardfarben."""
        assert "bt-dropdown" in _one(filled, wl.HOLDING_NEW_CURRENCY).className

    def test_currency_dropdown_offers_every_currency(self, filled: AppState):
        values = {o["value"] for o in _one(filled, wl.HOLDING_NEW_CURRENCY).options}
        assert values == {c.value for c in Currency}


class TestEditableHoldingRows:
    """Eine Zeile je KAUF - die Bewertungstabelle fasst dieselben zusammen."""

    def test_repeated_buys_get_one_row_each(self, filled: AppState):
        """Zwei AAPL-Kaeufe und ein MSFT-Kauf ergeben drei Zeilen."""
        assert len(_wildcard(filled, wl.HOLDING_SHARES_TYPE)) == 3

    def test_summary_table_still_aggregates_the_same_buys(self, filled: AppState):
        """Beide Aussagen gelten: drei Zeilen zum Bearbeiten, eine zum Lesen."""
        assert "13.0000" in str(_tree(filled))

    def test_rows_are_addressed_by_their_store_index(self, filled: AppState):
        """Der Index adressiert update_holding und remove_holding."""
        for index in range(3):
            for kind in (
                wl.HOLDING_SHARES_TYPE,
                wl.HOLDING_PRICE_TYPE,
                wl.HOLDING_DATE_TYPE,
                wl.HOLDING_SAVE_TYPE,
                wl.HOLDING_REMOVE_TYPE,
            ):
                assert _by_id(filled, {"type": kind, "index": index}), (
                    f"{kind} fehlt fuer Index {index}"
                )

    def test_indices_start_at_zero_and_have_no_gaps(self, filled: AppState):
        indices = sorted(n.id["index"] for n in _wildcard(filled, wl.HOLDING_SHARES_TYPE))
        assert indices == [0, 1, 2]

    @pytest.mark.parametrize(
        ("index", "shares", "price"),
        [(0, 10.0, 150.0), (1, 3.0, 210.0), (2, 5.0, 380.0)],
    )
    def test_fields_are_prefilled_from_the_stored_holding(
        self, filled: AppState, index: int, shares: float, price: float
    ):
        """Ein leeres Feld waere ein Loeschantrag, kein Bearbeitungsstand."""
        assert _one(filled, {"type": wl.HOLDING_SHARES_TYPE, "index": index}).value == shares
        assert _one(filled, {"type": wl.HOLDING_PRICE_TYPE, "index": index}).value == price
        assert _one(filled, {"type": wl.HOLDING_DATE_TYPE, "index": index}).value == "2024-01-15"

    def test_the_second_buy_keeps_its_own_price(self, filled: AppState):
        """Der eigene Einstand je Kauf ist der Grund fuer die Index-Adressierung."""
        first = _one(filled, {"type": wl.HOLDING_PRICE_TYPE, "index": 0}).value
        second = _one(filled, {"type": wl.HOLDING_PRICE_TYPE, "index": 1}).value
        assert (first, second) == (150.0, 210.0)

    def test_ticker_is_text_and_not_an_input(self, filled: AppState):
        """Der Ticker steht als Text da, nicht als Eingabefeld.

        update_holding kann ihn nicht aendern; ein Feld dafuer waere ein
        Versprechen, das der Store nicht einloest.
        """
        held = {"AAPL", "MSFT"}
        offenders = [
            n
            for n in _walk(_tree(filled))
            if isinstance(n, dcc.Input) and getattr(n, "value", None) in held
        ]
        assert offenders == [], f"Ticker als Eingabefeld: {[n.id for n in offenders]}"

    def test_currency_per_row_is_text_and_not_a_dropdown(self, filled: AppState):
        """update_holding nimmt keine currency entgegen."""
        per_row = [
            n
            for n in _walk(_tree(filled))
            if isinstance(n, dcc.Dropdown) and isinstance(getattr(n, "id", None), dict)
        ]
        assert per_row == []

    def test_the_technical_index_is_not_shown_to_the_user(self, filled: AppState):
        """Der Index ist eine Adresse im Store, keine Depotinformation."""
        for index in range(3):
            button = _one(filled, {"type": wl.HOLDING_SAVE_TYPE, "index": index})
            assert button.children == "Save"

    def test_rows_offer_save_and_remove(self, filled: AppState):
        assert _one(filled, {"type": wl.HOLDING_SAVE_TYPE, "index": 0}).children == "Save"
        assert _one(filled, {"type": wl.HOLDING_REMOVE_TYPE, "index": 0}).children == "Remove"

    def test_every_row_has_exactly_one_remove_button(self, filled: AppState):
        """Ein fehlender Knopf liesse genau eine Position unloeschbar."""
        assert len(_wildcard(filled, wl.HOLDING_REMOVE_TYPE)) == 3


class TestEditableWatchlist:
    """Die Watchlist wird ueber den Ticker adressiert, nicht ueber den Index."""

    @pytest.mark.parametrize(
        "element_id",
        [wl.WATCHLIST_NEW_TICKER, wl.WATCHLIST_NEW_NOTE, wl.WATCHLIST_ADD],
    )
    def test_add_form_exists(self, filled: AppState, element_id: str):
        assert _by_id(filled, element_id), f"{element_id} fehlt"

    def test_add_form_is_present_in_an_empty_watchlist(self, state: AppState):
        state.holdings.create_watchlist("Leer")
        assert _by_id(state, wl.WATCHLIST_ADD), "Leere Watchlist ohne Aufnahme-Formular"

    def test_row_is_keyed_by_ticker(self, filled: AppState):
        for kind in (wl.WATCH_NOTE_TYPE, wl.WATCH_SAVE_TYPE, wl.WATCH_REMOVE_TYPE):
            assert _by_id(filled, {"type": kind, "ticker": "KO"}), f"{kind} fehlt fuer KO"

    def test_rows_are_not_keyed_by_index(self, filled: AppState):
        """Ein Index waere hier falsch: remove_from_watchlist nimmt den Ticker."""
        assert _wildcard(filled, wl.WATCH_NOTE_TYPE, key="index") == []

    def test_note_field_is_prefilled(self, filled: AppState):
        assert _one(filled, {"type": wl.WATCH_NOTE_TYPE, "ticker": "KO"}).value == "Dividende"

    def test_missing_note_becomes_an_empty_field(self, state: AppState):
        """Ein Platzhalter im Wertfeld waere eine erfundene Notiz."""
        state.holdings.create_watchlist("Ohne")
        state.holdings.add_to_watchlist("Ohne", "MSFT")
        assert _one(state, {"type": wl.WATCH_NOTE_TYPE, "ticker": "MSFT"}).value == ""


class TestEarningsColumn:
    """Die Spalte mit dem naechsten Quartalstermin.

    Der Nutzer wollte sie in Depot und Watchlist sehen. Drei Zustaende
    muessen unterscheidbar bleiben: ein bekannter Termin, ein unbekannter
    (Aktie, Quelle nennt keinen) und "gibt es nicht" (ETF, Krypto).
    """

    def test_the_column_exists_in_both_tables(self, filled: AppState) -> None:
        rendered = _render(filled)
        assert rendered.count("Next earnings") >= 2, "Spalte fehlt in einer der Tabellen"

    def test_the_hint_names_the_uncertainty(self, filled: AppState) -> None:
        """Die Quelle kennzeichnet nicht, ob ein Termin bestaetigt ist.

        Wer nach einem Termin plant, soll wissen, dass er sich verschieben
        kann - sonst liest sich das Datum wie eine Zusage.

        Gezaehlt statt gesucht: mit einem blossen ``in`` bliebe der Test
        gruen, wenn der Hinweis nur aus einer der beiden Tabellen
        verschwindet - die andere traegt ihn ja noch.
        """
        assert _render(filled).count("may shift") >= 2

    def test_a_failing_lookup_leaves_the_page_intact(
        self, state: AppState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scheitert der Terminabruf, bleibt die Bewertung stehen.

        Die Termine sind eine Zugabe. Ihretwegen darf die Depotanzeige
        nicht ausfallen - sonst kostet eine Nebensaechlichkeit den Blick
        auf das eigene Vermoegen.
        """

        def boom(_tickers: object) -> dict[str, object]:
            raise ProviderUnavailableError("Earnings unavailable.", detail="Testfall")

        state.holdings.create_portfolio("Kaputt", currency=Currency.USD)
        state.holdings.add_holding("Kaputt", "AAPL", shares=2, buy_price=100.0, buy_date=BUY_DATE)
        monkeypatch.setattr(state.provider, "get_earnings_dates", boom, raising=False)

        rendered = _render(state, ("Kaputt",))
        assert "bt-error" not in rendered, "Der Termin-Ausfall hat die Seite gekippt"
        assert "AAPL" in rendered
        assert "Next earnings" in rendered

    def test_a_known_date_shows_with_days(self, state: AppState) -> None:
        """Das Datum allein sagt wenig; der Abstand in Tagen schon."""
        state.holdings.create_portfolio("Termine", currency=Currency.USD)
        state.holdings.add_holding("Termine", "AAPL", shares=1, buy_price=100.0, buy_date=BUY_DATE)
        rendered = _render(state, ("Termine",))
        assert re.search(r"20\d\d-\d\d-\d\d \(in \d+d\)", rendered), rendered[:400]

    def test_an_etf_shows_a_dash_not_na(self, state: AppState) -> None:
        """Ein ETF fuehrt keine Quartalszahlen - das ist eine Aussage.

        "n/a" hiesse "wir wissen es nicht" und liesse den Nutzer nach
        einem Termin suchen, den es nicht gibt.
        """
        state.holdings.create_portfolio("Fonds", currency=Currency.USD)
        state.holdings.add_holding("Fonds", "SPY", shares=1, buy_price=100.0, buy_date=BUY_DATE)
        assert "\u2014" in _render(state, ("Fonds",))

    def test_missing_earnings_never_break_the_page(self, state: AppState) -> None:
        """Die Termine sind eine Zugabe, kein Grund fuer ein Fehlerbanner."""
        state.holdings.create_portfolio("Unbekannt", currency=Currency.USD)
        state.holdings.add_holding(
            "Unbekannt", "ZZZZ", shares=1, buy_price=100.0, buy_date=BUY_DATE
        )
        rendered = _render(state, ("Unbekannt",))
        assert "bt-error" not in rendered
        assert "ZZZZ" in rendered


class TestSortingByEarnings:
    """Sortieren nach Termin - mit gemischten Zeilen.

    Dieser Fall ist einmal abgestuerzt: die Zelle trug entweder ein Datum
    oder einen Markerstring fuer "fuehrt keine Earnings", und pandas kann
    beide nicht vergleichen:

        TypeError: '<' not supported between instances of 'str' and 'date'

    Aufgefallen ist es erst beim Rendern - weder mypy noch die uebrigen
    Tests haetten es gezeigt. Seitdem traegt das Datenfeld nur noch ein
    ``date`` oder ``None``.
    """

    @staticmethod
    def _earnings_column(content: object) -> list[tuple[str, str]]:
        """Liest Ticker und Termin-Zelle jeder Tabellenzeile."""
        found: list[tuple[str, str]] = []

        def walk(node: object) -> None:
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
                        ticker = getattr(cells[0], "children", None)
                        last = getattr(cells[-1], "children", None)
                        if isinstance(ticker, str):
                            found.append((ticker, str(last)))
                return
            if children is not None:
                walk(children)

        walk(content)
        return found

    @pytest.fixture
    def mixed(self, state: AppState) -> AppState:
        """Depot mit drei Aktien und einem ETF."""
        state.holdings.create_portfolio("Gemischt", currency=Currency.USD)
        for ticker in ("AAPL", "SPY", "MSFT", "KO"):
            state.holdings.add_holding(
                "Gemischt", ticker, shares=1, buy_price=100.0, buy_date=BUY_DATE
            )
        return state

    def test_sorting_by_date_does_not_crash(self, mixed: AppState) -> None:
        """Der Absturzfall: Aktien und ein ETF in einer Tabelle."""
        content = wl.render_portfolio(mixed, "Gemischt", sort_by="Next earnings")
        assert self._earnings_column(content), "Keine Zeilen gerendert"

    def test_the_nearest_date_comes_first(self, mixed: AppState) -> None:
        content = wl.render_portfolio(mixed, "Gemischt", sort_by="Next earnings")
        dates = [cell for _t, cell in self._earnings_column(content) if cell[:2] == "20"]
        assert len(dates) == 3, dates
        assert dates == sorted(dates), dates

    def test_instruments_without_earnings_stay_last(self, mixed: AppState) -> None:
        """In BEIDEN Richtungen - ein ETF ist nie der naechste Termin."""
        for descending in (False, True):
            content = wl.render_portfolio(
                mixed, "Gemischt", sort_by="Next earnings", sort_desc=descending
            )
            rows = self._earnings_column(content)
            assert rows[-1][0] == "SPY", (descending, rows)

    def test_descending_reverses_only_the_dates(self, mixed: AppState) -> None:
        asc = self._earnings_column(wl.render_portfolio(mixed, "Gemischt", sort_by="Next earnings"))
        desc = self._earnings_column(
            wl.render_portfolio(mixed, "Gemischt", sort_by="Next earnings", sort_desc=True)
        )
        dated_asc = [t for t, cell in asc if cell[:2] == "20"]
        dated_desc = [t for t, cell in desc if cell[:2] == "20"]
        assert dated_asc == list(reversed(dated_desc)), (dated_asc, dated_desc)
