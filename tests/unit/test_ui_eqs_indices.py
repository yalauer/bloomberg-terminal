"""Tests der Indexauswahl auf der Screening-Seite.

Der Nutzer verlangte: "beim screening sollen alle fimen eines index
(s&p 500, dax, dow jones, nasdaq 100, etc.) durchsucht werden und nicht
nur vordefinierte firmen." Vorher screente die Seite die zehn Ticker mit
Offline-Fixtures.

Alle Tests laufen offline. Der Indexabruf wird durch eine Attrappe
ersetzt; echte Netzzugriffe blockiert ``tests/conftest.py`` ohnehin
prozessweit.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest

from btlite.config import Settings
from btlite.core.errors import ProviderUnavailableError
from btlite.data.indices import DEFAULT_INDEX_KEY, INDEX_REGISTRY
from btlite.ui.app import render_page
from btlite.ui.pages import eqs
from btlite.ui.state import AppState

if TYPE_CHECKING:
    from pathlib import Path

#: Ein handliches Ersatzuniversum mit Tickern, zu denen Fixtures vorliegen.
FAKE_MEMBERS = ["AAPL", "MSFT", "KO", "WMT", "JNJ"]


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppState:
    """Zustand, der sich fuer die Seite wie ein Live-Lauf verhaelt.

    Die Indexauswahl greift nur online: offline nimmt die Seite bewusst
    die eingefrorenen Titel, weil zu einem 503er-Bestand nur zehn
    Kursreihen vorliegen. Die Tests hier pruefen genau den Online-Pfad,
    stellen aber selbst keine Verbindung her - der Indexabruf wird in
    jedem Test durch eine Attrappe ersetzt.
    """
    monkeypatch.setenv("BTLITE_PORTFOLIO_HOLDINGS_FILE", str(tmp_path / "holdings.json"))
    instance = AppState(Settings())
    monkeypatch.setattr(type(instance), "is_offline", property(lambda _self: False))
    return instance


@pytest.fixture
def state_with_index(state: AppState, monkeypatch: pytest.MonkeyPatch) -> AppState:
    """Zustand, dessen Indexabruf eine feste Liste liefert."""
    monkeypatch.setattr(
        type(state.indices),
        "get_constituents",
        lambda _self, _key: _constituents(list(FAKE_MEMBERS)),
    )
    return state


def _constituents(tickers: list[str], *, source_count: int | None = None):
    """Baut eine Bestandsantwort mit den gegebenen Tickern.

    Args:
        tickers: Symbole des Universums.
        source_count: Wie viele Titel die Quelle fuehrte. Groesser als die
            Tickerzahl bedeutet, dass Titel uebergangen wurden.

    Returns:
        Bestand, wie ihn der Indexanbieter liefert.
    """
    from btlite.core.clock import utc_now
    from btlite.data.indices import INDEX_REGISTRY
    from btlite.models.common import DataSource, Provenance
    from btlite.models.screening import IndexConstituents, IndexMember, SkippedMember

    # source_count ist abgeleitet (Mitglieder plus Uebersprungene), kein
    # Eingabefeld. Eine Luecke entsteht also ueber skipped.
    total = source_count if source_count is not None else len(tickers)
    skipped = [
        SkippedMember(
            source_ticker=f"SKIP{i}",
            name=f"Uebergangen {i}",
            country="Testland",
            reason="kein eindeutiges Symbol",
        )
        for i in range(max(0, total - len(tickers)))
    ]
    return IndexConstituents(
        definition=INDEX_REGISTRY["sp500"],
        members=[IndexMember(ticker=t, name=t) for t in tickers],
        provenance=Provenance(source=DataSource.FINVIZ, fetched_at=utc_now()),
        skipped=skipped,
    )


def _render(state: AppState, args: tuple[str, ...] = ()) -> str:
    return str(render_page(state, "EQS", [], args))


class TestIndexRegistryIsOffered:
    def test_user_named_indices_are_available(self, state: AppState):
        """S&P 500, DAX, Dow Jones und Nasdaq 100 waren ausdruecklich genannt."""
        keys = {definition.key for definition in state.indices.available_indices()}
        assert {"sp500", "dax", "djia", "nasdaq100"} <= keys

    def test_default_is_the_sp500(self):
        assert DEFAULT_INDEX_KEY == "sp500"

    def test_dropdown_lists_every_registered_index(self, state: AppState):
        options = eqs._index_options(state)
        assert {option["value"] for option in options} == set(INDEX_REGISTRY)

    def test_dropdown_labels_are_readable(self, state: AppState):
        """Der Nutzer waehlt "S&P 500", nicht "sp500"."""
        labels = {option["label"] for option in eqs._index_options(state)}
        assert "S&P 500" in labels
        assert "DAX" in labels


class TestIndexFromArguments:
    def test_known_key_is_accepted(self):
        assert eqs._index_from_args(("dax",)) == "dax"

    def test_case_is_ignored(self):
        assert eqs._index_from_args(("DAX",)) == "dax"

    def test_unknown_key_falls_back_to_the_default(self):
        assert eqs._index_from_args(("gibt-es-nicht",)) == DEFAULT_INDEX_KEY

    def test_no_argument_yields_the_default(self):
        assert eqs._index_from_args(()) == DEFAULT_INDEX_KEY

    def test_a_preset_key_does_not_select_an_index(self):
        """``EQS value GO`` waehlt ein Preset, keinen Index."""
        assert eqs._index_from_args(("value",)) == DEFAULT_INDEX_KEY


class TestUniverseResolution:
    def test_index_members_become_the_universe(self, state_with_index: AppState):
        universe, note, error = eqs._resolve_universe(state_with_index, [], "sp500")
        assert universe == FAKE_MEMBERS
        assert error is None
        assert "S&P 500" in note

    def test_note_names_the_number_of_titles(self, state_with_index: AppState):
        _, note, _ = eqs._resolve_universe(state_with_index, [], "sp500")
        assert str(len(FAKE_MEMBERS)) in note

    def test_explicit_tickers_win_over_the_index(self, state_with_index: AppState):
        """``AAPL,MSFT EQS GO`` meint genau diese beiden."""
        universe, note, error = eqs._resolve_universe(state_with_index, ["AAPL", "MSFT"], "sp500")
        assert universe == ["AAPL", "MSFT"]
        assert error is None
        assert "S&P 500" not in note

    def test_fallback_is_never_silent(self, state: AppState, monkeypatch):
        """Ein STILLES Ersatzuniversum waere der gemeldete Mangel.

        Offline bleiben die eingefrorenen Titel als Notbehelf - aber der
        Nutzer waehlte "S&P 500" und muss erkennen, dass er etwas anderes
        sieht. Der Ersatz ist erlaubt, das Verschweigen nicht.
        """

        def _boom(_self: Any, _key: str) -> list[str]:
            raise ProviderUnavailableError("Quelle weg", detail="Test")

        monkeypatch.setattr(type(state.indices), "get_constituents", _boom)
        universe, note, error = eqs._resolve_universe(state, [], "sp500")
        assert error is not None, "Der Fehlschlag darf nicht verloren gehen"
        assert "unavailable" in note
        assert "S&P 500" in note, "Der gewuenschte Index muss genannt sein"
        if universe:
            assert "frozen titles instead" in note

    def test_no_fallback_yields_an_empty_universe(self, state: AppState, monkeypatch):
        """Ohne Ersatztitel bleibt die Seite leer statt irgendetwas zu zeigen."""

        def _boom(_self: Any, _key: str) -> list[str]:
            raise ProviderUnavailableError("Quelle weg", detail="Test")

        monkeypatch.setattr(type(state.indices), "get_constituents", _boom)
        monkeypatch.setattr(type(state), "available_tickers", lambda _self: [])
        universe, note, error = eqs._resolve_universe(state, [], "sp500")
        assert universe == []
        assert error is not None
        assert "unavailable" in note

    def test_slow_universe_announces_the_wait(self, state: AppState, monkeypatch):
        """Ohne Hinweis wirkt die Seite bei 500 Titeln haengend."""
        many = [f"T{i:04d}" for i in range(400)]
        monkeypatch.setattr(
            type(state.indices), "get_constituents", lambda _s, _k: _constituents(list(many))
        )
        _, note, _ = eqs._resolve_universe(state, [], "sp500")
        assert "minutes" in note

    def test_very_large_index_states_its_real_cost(self, state: AppState, monkeypatch):
        """Der Russell 2000 hat knapp 2000 Titel.

        Der Nutzer hat entschieden, dass die Auswahl ohne Rueckfrage
        laeuft. Dann muss die Kopfzeile die Dauer aber ehrlich nennen -
        eine untertriebene Schaetzung waere schlimmer als keine.
        """
        many = [f"T{i:05d}" for i in range(1944)]
        monkeypatch.setattr(
            type(state.indices), "get_constituents", lambda _s, _k: _constituents(list(many))
        )
        _, note, _ = eqs._resolve_universe(state, [], "russell2000")
        assert "1944 titles" in note
        minutes = int(re.search(r"about (\d+) minutes", note).group(1))
        assert minutes >= 10, f"Dauer untertrieben: {note}"

    def test_small_universe_stays_quiet(self, state_with_index: AppState):
        _, note, _ = eqs._resolve_universe(state_with_index, [], "djia")
        assert "minutes" not in note


class TestRenderedPage:
    def test_page_renders_with_index_members(self, state_with_index: AppState):
        rendered = _render(state_with_index)
        assert "bt-error" not in rendered
        assert "eqs-index" in rendered

    def test_unreachable_index_is_visible(self, state: AppState, monkeypatch):
        """Ein nicht abrufbarer Bestand muss sichtbar sein, nicht still.

        Der Abruf wird hier ausdruecklich zum Scheitern gebracht. Ohne das
        haengt der Test daran, ob zufaellig ein Bestand im Cache liegt -
        der wird naemlich auch offline bedient.
        """

        def _boom(_self: Any, _key: str) -> object:
            raise ProviderUnavailableError(
                "Offline-Modus: der Bestand laesst sich nicht abrufen. "
                "Fuer Live-Daten BTLITE_DATA_OFFLINE=0 setzen.",
                detail="Test",
            )

        monkeypatch.setattr(type(state.indices), "get_constituents", _boom)
        rendered = _render(state)
        assert "bt-warning" in rendered or "bt-error" in rendered
        assert "unavailable" in rendered

    def test_substitute_universe_is_named(self, state: AppState, monkeypatch):
        """Der Nutzer muss erkennen, dass er nicht den S&P 500 sieht."""

        def _boom(_self: Any, _key: str) -> object:
            raise ProviderUnavailableError("weg", detail="Test")

        monkeypatch.setattr(type(state.indices), "get_constituents", _boom)
        rendered = _render(state)
        assert "S&P 500" in rendered
        assert "frozen titles instead" in rendered

    def test_failure_message_names_the_way_out(self, state: AppState, monkeypatch):
        def _boom(_self: Any, _key: str) -> object:
            raise ProviderUnavailableError(
                "Offline-Modus. Fuer Live-Daten BTLITE_DATA_OFFLINE=0 setzen.",
                detail="Test",
            )

        monkeypatch.setattr(type(state.indices), "get_constituents", _boom)
        assert "BTLITE_DATA_OFFLINE" in _render(state)

    def test_explicit_tickers_render_without_the_index(self, state: AppState):
        """Mit ausdruecklicher Liste braucht die Seite keinen Indexabruf."""
        rendered = str(render_page(state, "EQS", ["AAPL", "MSFT"], ()))
        assert "bt-error" not in rendered


class TestSymbolSearchIsWired:
    """Der Indexanbieter braucht die Symbolsuche, sonst fehlen Titel.

    Die zweite Aufloesungsstufe (Ticker ueber den Firmennamen finden)
    laeuft nur mit einem Sucher. Ohne ihn scheitert beim STOXX Europe 600
    die erste Stufe an rund 110 von 462 Titeln - der Index waere
    stillschweigend ein Viertel kleiner. Genau dieser Fehler war in
    ``ui/state.py`` und ist hier abgesichert.
    """

    def test_state_passes_a_symbol_search(self, state: AppState):
        assert state.indices._search is not None

    def test_the_search_satisfies_the_protocol(self, state: AppState):
        from btlite.data.base import SymbolSearch

        assert isinstance(state.indices._search, SymbolSearch)

    def test_the_search_is_the_active_provider(self, state: AppState):
        """Kein zweiter Provider: eine Quelle, eine Drosselung."""
        assert state.indices._search is state.provider


class TestIncompleteUniverseIsVisible:
    """Nicht auflösbare Titel duerfen nicht stillschweigend fehlen.

    Beim STOXX Europe 600 fuehrt die Quelle 462 Titel, davon lassen sich
    40 keinem eindeutigen Boersensymbol zuordnen. Sie bleiben draussen -
    lieber fehlt ein Wert, als dass die falsche Firma im Screening steht.
    Aber der Nutzer muss erfahren, dass er 422 statt 462 screent.
    """

    def test_header_names_the_gap(self, state: AppState, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            type(state.indices),
            "get_constituents",
            lambda _s, _k: _constituents(["AAPL", "MSFT"], source_count=5),
        )
        _, note, _ = eqs._resolve_universe(state, [], "sp500")
        assert "2 titles of 5" in note
        assert "3 skipped without a unique symbol" in note

    def test_complete_universe_stays_quiet(self, state_with_index: AppState):
        """Ohne Luecke kein Hinweis - sonst stumpft er ab."""
        _, note, _ = eqs._resolve_universe(state_with_index, [], "sp500")
        assert "uebergangen" not in note

    def test_gap_appears_on_the_rendered_page(
        self, state: AppState, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            type(state.indices),
            "get_constituents",
            lambda _s, _k: _constituents(["AAPL", "MSFT"], source_count=5),
        )
        assert "skipped without a unique symbol" in _render(state)


class TestLargeUniverseWaitsForTheUser:
    """Ein Klick im Navigationsmenue darf nicht minutenlang blockieren.

    Der Nutzer meldete: "wenn eqs ausgewählt wird passiert nichts nach
    5min außer das die seite updating sagt". Die Seite lud beim Oeffnen
    ungefragt den ganzen S&P 500. Jetzt zeigt sie erst, was es kosten
    wuerde, und rechnet auf Knopfdruck.
    """

    def _big(self, state: AppState, monkeypatch: pytest.MonkeyPatch, count: int = 503) -> None:
        many = [f"T{i:05d}" for i in range(count)]
        monkeypatch.setattr(
            type(state.indices), "get_constituents", lambda _s, _k: _constituents(many)
        )

    def test_page_offers_a_start_button(self, state: AppState, monkeypatch: pytest.MonkeyPatch):
        self._big(state, monkeypatch)
        assert eqs.RUN_BUTTON in _render(state)

    def test_large_universe_does_not_compute_on_open(
        self, state: AppState, monkeypatch: pytest.MonkeyPatch
    ):
        """Der teure Aufruf darf beim Seitenaufbau gar nicht erst fallen."""
        self._big(state, monkeypatch)
        calls: list[int] = []
        monkeypatch.setattr(
            eqs, "render_result", lambda *a, **k: calls.append(1) or html_div_stub()
        )
        _render(state)
        assert calls == [], "Die Seite hat ungefragt gerechnet"

    def test_prompt_states_the_cost(self, state: AppState, monkeypatch: pytest.MonkeyPatch):
        self._big(state, monkeypatch)
        rendered = _render(state)
        assert "503 titles ready" in rendered
        assert "minutes" in rendered

    def test_small_universe_still_computes_immediately(self, state_with_index: AppState):
        """Bei fuenf Titeln waere ein Extraklick nur laestig."""
        calls: list[int] = []
        original = eqs.render_result

        def _counting(*args: object, **kwargs: object) -> object:
            calls.append(1)
            return original(*args, **kwargs)  # type: ignore[arg-type]

        eqs.render_result = _counting  # type: ignore[assignment]
        try:
            _render(state_with_index)
        finally:
            eqs.render_result = original  # type: ignore[assignment]
        assert calls, "Kleines Universum sollte sofort rechnen"


class TestCallbacksAlsoWaitForTheUser:
    """Der Seitenaufbau fragte, die Callbacks rechneten trotzdem durch.

    Der Nutzer meldete erneut "eqs laed auch nicht". Beim Oeffnen fragte
    die Seite korrekt - aber jeder Wechsel von Universum oder Preset rief
    unmittelbar ``render_result`` auf und blockierte damit minutenlang,
    genau wie vorher. Die Entscheidung liegt jetzt an einer Stelle:
    :func:`eqs.initial_result`.
    """

    def _big(self, count: int = 503) -> list[str]:
        return [f"T{i:05d}" for i in range(count)]

    def test_large_universe_asks_instead_of_computing(
        self, state: AppState, monkeypatch: pytest.MonkeyPatch
    ):
        calls: list[int] = []
        monkeypatch.setattr(
            eqs, "render_result", lambda *a, **k: calls.append(1) or html_div_stub()
        )
        rendered = str(eqs.initial_result(state, "value", self._big()))
        assert calls == [], "Ein grosses Universum wurde ungefragt gerechnet"
        assert "503 titles ready" in rendered

    def test_small_universe_computes(self, state_with_index: AppState):
        calls: list[int] = []
        original = eqs.render_result

        def _counting(*args: object, **kwargs: object) -> object:
            calls.append(1)
            return original(*args, **kwargs)  # type: ignore[arg-type]

        eqs.render_result = _counting  # type: ignore[assignment]
        try:
            eqs.initial_result(state_with_index, "value", list(FAKE_MEMBERS))
        finally:
            eqs.render_result = original  # type: ignore[assignment]
        assert calls, "Ein kleines Universum sollte sofort rechnen"

    def test_empty_universe_is_a_placeholder_not_a_crash(self, state: AppState):
        assert "No universe" in str(eqs.initial_result(state, "value", []))

    def test_index_switch_does_not_screen(self, state: AppState, monkeypatch: pytest.MonkeyPatch):
        """Der Wechsel des Universums laedt nur die Mitgliederliste."""
        members = self._big()
        monkeypatch.setattr(
            type(state.indices), "get_constituents", lambda _s, _k: _constituents(members)
        )
        calls: list[int] = []
        monkeypatch.setattr(
            eqs, "render_result", lambda *a, **k: calls.append(1) or html_div_stub()
        )
        callback = _callback_source("_on_eqs_index")
        assert "render_result" not in callback, (
            "Der Index-Callback ruft render_result direkt auf und rechnet damit ungefragt"
        )
        assert "initial_result" in callback

    def test_preset_switch_does_not_screen(self):
        """Der Wechsel des Presets ebenso - er aendert nur die Filter."""
        callback = _callback_source("_on_eqs_preset")
        assert "render_result" not in callback, "Der Preset-Callback ruft render_result direkt auf"
        assert "initial_result" in callback

    def test_the_run_button_is_the_one_that_computes(self):
        """Gerechnet wird genau auf Knopfdruck - sonst gaebe es kein Ziel."""
        callback = _callback_source("_on_eqs_run")
        assert "render_result" in callback


def _callback_source(name: str) -> str:
    """Liest den Quelltext eines Callbacks aus ``ui/app.py``.

    Die Callbacks sind in ``create_app`` eingeschlossen und von aussen
    nicht erreichbar; geprueft wird deshalb am Quelltext, welche der
    beiden Einstiegsfunktionen sie verwenden.

    Args:
        name: Name der Callback-Funktion.

    Returns:
        Quelltext der Funktion bis zum naechsten Callback.
    """
    import inspect

    from btlite.ui import app as app_module

    source = inspect.getsource(app_module)
    start = source.index(f"def {name}(")
    rest = source[start:]
    end = rest.find(chr(10) + "    @app.callback")
    return rest if end < 0 else rest[:end]


def html_div_stub() -> object:
    """Platzhalter fuer einen abgefangenen render_result-Aufruf."""
    from dash import html

    return html.Div()


class TestRuntimeEstimateMatchesMeasurement:
    """Die Dauerangabe hat den Nutzer zweimal in die Irre gefuehrt.

    Erster Anlauf: die Seite nannte vier Minuten, gemessen waren es
    siebeneinhalb. Die damalige Erklaerung - Bilanzdaten kosten doppelt
    so viel wie Kurse (0.89s gegen 0.45s) - stimmte nicht.

    Zweiter Anlauf, nachgemessen am 25.09.2026 mit je 20 Titeln und
    eigenem leerem Cache-Verzeichnis:

        value          kalt 2.97 s/Titel    warm 1.02 s/Titel
        near_52w_high  kalt 3.55 s/Titel    warm 1.02 s/Titel

    Zwei Befunde. Erstens lagen die alten Werte um den Faktor vier zu
    niedrig: real vergehen fuer den S&P 500 rund 28 Minuten, nicht vier
    bis sieben. Zweitens traegt die Unterscheidung technisch/fundamental
    nicht - das technische Preset war sogar das teurere. Beide Arten
    laden je Titel eine Kurshistorie, und die dominiert.

    Massgeblich ist stattdessen, ob der Cache warm ist. Dieser
    Unterschied ist echt und gross (Faktor drei).
    """

    def test_the_cold_case_is_estimated_higher(self):
        """Der erste Lauf ist teurer als der zweite - das ist der Kern."""
        assert eqs._estimate_minutes(503) > eqs._estimate_minutes(503, warm=True)

    def test_sp500_matches_the_measurement(self):
        """Gemessen rund 28 Minuten kalt - die Angabe darf nicht darunter liegen.

        Nicht auf den Punkt geprueft, sondern auf eine Untergrenze: die
        Quelle antwortet mal schneller, mal langsamer. Zu niedrig darf
        die Angabe nie sein, denn dann bricht der Nutzer ab, waehrend die
        Seite arbeitet - genau das ist zweimal passiert.
        """
        assert eqs._estimate_minutes(503) >= 25

    def test_a_warm_run_takes_about_a_minute(self):
        """Seit die Stammdaten gepuffert sind, gemessen 0.03-0.09 s/Titel.

        Frueher 8.6 min fuer einen warmen S&P-500-Lauf, weil ``Ticker.info``
        bei jedem Lauf neu geladen wurde.
        """
        assert 1 <= eqs._estimate_minutes(503, warm=True) <= 2

    def test_the_preset_no_longer_changes_the_estimate(self):
        """Die Messung widerlegt die alte Unterscheidung.

        Der Parameter bleibt in der Signatur, weil mehrere Aufrufstellen
        ihn uebergeben - er darf aber nichts mehr bewirken. Ohne diesen
        Test koennte jemand die alte Logik wieder einbauen.
        """
        for count in (50, 503, 1944):
            assert eqs._estimate_minutes(count, "value") == eqs._estimate_minutes(
                count, "below_200ma"
            )
            assert eqs._estimate_minutes(count, "") == eqs._estimate_minutes(count, "value")

    def test_never_below_one_minute(self):
        assert eqs._estimate_minutes(1, "value") == 1

    def test_nothing_to_do_takes_no_time(self):
        """Null Titel sind null Minuten, nicht eine.

        Die Restzeit-Anzeige rechnet damit: bei den letzten Titeln soll
        dort nicht "about 1 minute left" stehenbleiben.
        """
        assert eqs._estimate_minutes(0) == 0


class TestOfflineUsesFixtureTickers:
    """Offline nuetzt der beste Indexbestand nichts.

    Der Bestands-Cache haelt sieben Tage und wird auch offline bedient.
    Ein Screening ueber 503 Titel faende dort aber nur zu zehn davon
    Kurse - der Rest waere still leer. Deshalb nimmt die Seite offline
    die eingefrorenen Titel und sagt das auch.
    """

    @pytest.fixture
    def offline_state(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppState:
        monkeypatch.setenv("BTLITE_PORTFOLIO_HOLDINGS_FILE", str(tmp_path / "h.json"))
        return AppState(Settings())

    def test_offline_takes_the_fixture_universe(self, offline_state: AppState):
        assert offline_state.is_offline
        universe, _, _ = eqs._resolve_universe(offline_state, [], "sp500", "value")
        assert universe == offline_state.available_tickers()

    def test_offline_says_why(self, offline_state: AppState):
        _, note, _ = eqs._resolve_universe(offline_state, [], "sp500", "value")
        assert "Offline mode" in note
        assert "S&P 500" in note

    def test_offline_is_not_an_error(self, offline_state: AppState):
        """Es funktioniert ja - nur mit weniger Titeln."""
        _, _, error = eqs._resolve_universe(offline_state, [], "sp500", "value")
        assert error is None

    def test_offline_computes_immediately(self, offline_state: AppState):
        """Zehn Titel brauchen keinen Startknopf."""
        rendered = _render(offline_state)
        assert "bt-error" not in rendered
        assert "titles ready" not in rendered

    def test_explicit_tickers_still_win_offline(self, offline_state: AppState):
        universe, _, _ = eqs._resolve_universe(offline_state, ["AAPL"], "sp500", "value")
        assert universe == ["AAPL"]


class TestProgressView:
    """Die Anzeige waehrend eines laufenden Screenings.

    Der Nutzer meldete, das Screening "gibt keine ergebnisse". Es gab
    welche - nur sah er waehrend der halben Stunde nichts als einen
    Ladebalken und brach ab. Deshalb zeigt die Seite jetzt Zahlen: ein
    Balken allein beweist bei langsamem Fortschritt nicht, dass etwas
    vorangeht.
    """

    def test_the_numbers_are_visible(self):
        """Ohne Zahlen waere die Anzeige wieder nur ein Balken."""
        text = str(eqs.progress_view(137, 503, "value"))
        assert "137" in text
        assert "503" in text

    def test_the_remaining_time_is_named(self):
        text = str(eqs.progress_view(137, 503, "value"))
        assert "left" in text

    def test_the_bar_grows_with_progress(self):
        """Die Breite kommt als Inline-Style; sie muss dem Anteil folgen."""
        early = str(eqs.progress_view(50, 500, "value"))
        late = str(eqs.progress_view(400, 500, "value"))
        assert "10%" in early, early
        assert "80%" in late, late

    def test_an_unknown_total_does_not_divide_by_zero(self):
        """Vor der ersten Meldung ist die Gesamtzahl noch 0."""
        text = str(eqs.progress_view(0, 0, "value"))
        assert text.strip(), "Leere Anzeige"

    def test_the_last_stretch_says_almost_done(self):
        """Nicht "about 0 minutes left" - das liest sich wie ein Fehler."""
        assert "almost done" in str(eqs.progress_view(503, 503, "value"))


class TestEmptyUniverseIsNotAnError:
    """Kein Titel lieferte Daten - das ist kein Fehler des Nutzers.

    Gemeldet als "das screening endet in einem error". Die Seite zeigte

        Error: Field 'pe' does not exist in the universe.

    Das ist die Filter-Pruefung, die auf einem Frame ohne Spalten
    nachsieht, ob die gefilterte Spalte existiert. Technisch korrekt und
    fuer den Nutzer unbrauchbar: er hat nichts falsch gemacht und
    erfaehrt nicht, was los ist.

    Live tritt der Fall ein, wenn die Quelle ausfaellt oder ihr
    Rate-Limit alle Abrufe blockiert. ``RateLimitError`` ist ein
    ``DataError`` und wird in der Aufbauschleife bewusst uebersprungen,
    damit ein einzelner Ausfall nicht den Lauf kippt - uebersprungen sie
    alle, blieb ein leerer Frame zurueck.
    """

    @pytest.mark.parametrize("preset", ["value", "above_200ma"])
    def test_no_error_banner_when_nothing_arrives(self, state: AppState, preset: str):
        """Weder der fundamentale noch der technische Pfad darf werfen."""
        tree = eqs.render_result(state, preset, ["FAKE1", "FAKE2", "FAKE3"])
        assert "bt-error" not in str(tree), str(tree)[:300]

    @pytest.mark.parametrize("preset", ["value", "above_200ma"])
    def test_the_message_names_the_likely_cause(self, state: AppState, preset: str):
        """Die alte Meldung - Field 'pe' does not exist - half niemandem."""
        text = str(eqs.render_result(state, preset, ["FAKE1", "FAKE2"]))
        assert "No data arrived" in text
        assert "rate limit" in text or "unreachable" in text

    @pytest.mark.parametrize("preset", ["value", "above_200ma"])
    def test_an_empty_list_says_so(self, state: AppState, preset: str):
        """Ohne Titel gibt es nichts zu filtern - auch kein Fehler."""
        tree = eqs.render_result(state, preset, [])
        assert "bt-error" not in str(tree)
        assert "No universe" in str(tree)

    @pytest.mark.parametrize("preset", ["value", "above_200ma"])
    def test_one_good_title_still_yields_a_result(self, state: AppState, preset: str):
        """Die Ausnahme gilt nur, wenn WIRKLICH nichts ankam.

        Ohne diesen Test koennte jemand die Pruefung zu frueh ansetzen
        und ein brauchbares Teilergebnis verwerfen.
        """
        tree = eqs.render_result(state, preset, ["AAPL", "FAKE1", "FAKE2"])
        assert "bt-error" not in str(tree)
        assert "No data arrived" not in str(tree)

    def test_the_normal_case_is_untouched(self, state: AppState):
        tree = eqs.render_result(state, "value", ["AAPL", "MSFT", "KO"])
        assert "bt-error" not in str(tree)
        assert "Matches" in str(tree)
