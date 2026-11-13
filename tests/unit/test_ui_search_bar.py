"""Tests der Namenssuche und der anschliessenden Analyse-Auswahl.

Alle Tests laufen offline gegen den ``FixtureProvider`` - nicht gegen
Attrappen der eigenen Logik. Der Kern der Anforderung ("servicenow soll
NYSE:NOW finden") wird deshalb an der echten Suche geprueft.

Die Callbacks werden ueber ``app.callback_map`` aufgerufen, also durch
dieselbe Registrierung, die auch der Server bedient. Das hat einen echten
Fehler gefunden: ein Callback mit ``ALL``-Eingabe ist fuer Dash ein
Wildcard-Callback und muss auch bei einem einzelnen Output eine Liste
zurueckgeben.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from tests.unit.test_ui_pages import has_class, text_of, walk

from btlite.config import Settings
from btlite.core.errors import ProviderUnavailableError
from btlite.ui import search_bar
from btlite.ui.app import ARGS_PARAM, TICKER_PARAM, create_app, parse_href
from btlite.ui.command_bar import MNEMONICS
from btlite.ui.state import AppState, set_state

if TYPE_CHECKING:
    from btlite.models.market import SearchHit


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[AppState]:
    """Baut einen Fixture-gestuetzten UI-Zustand.

    Die Depot-Datei zeigt ins temporaere Verzeichnis. Ohne das wuerde die
    Watchlist-Aktion in den echten Bestand des Nutzers unter ``~/.btlite``
    schreiben.
    """
    monkeypatch.setenv("BTLITE_PORTFOLIO_HOLDINGS_FILE", str(tmp_path / "holdings.json"))
    settings = Settings()
    settings.data.offline = True
    instance = AppState(settings)
    set_state(instance)
    yield instance
    set_state(None)


def ids_of(tree: Any) -> list[Any]:
    """Sammelt alle gesetzten Element-IDs eines Komponentenbaums.

    Args:
        tree: Wurzel des Baums.

    Returns:
        IDs in Dokumentreihenfolge.
    """
    return [node.id for node in walk(tree) if getattr(node, "id", None) is not None]


class TestRunSearch:
    """Die Suche selbst, gegen die eingefrorenen Daten."""

    def test_servicenow_finds_now(self, state: AppState) -> None:
        # Die woertliche Anforderung des Nutzers.
        tree = search_bar.run_search(state, "servicenow")
        assert tree is not None
        assert "NYSE:NOW" in text_of(tree)
        assert {"type": search_bar.HIT_TYPE, "ticker": "NOW"} in ids_of(tree)

    def test_hit_shows_issuer_name(self, state: AppState) -> None:
        tree = search_bar.run_search(state, "servicenow")
        assert tree is not None
        assert "ServiceNow" in text_of(tree)

    def test_typo_still_finds(self, state: AppState) -> None:
        tree = search_bar.run_search(state, "microsft")
        assert tree is not None
        assert "MSFT" in text_of(tree)

    def test_ticker_query_finds_itself(self, state: AppState) -> None:
        tree = search_bar.run_search(state, "AAPL")
        assert tree is not None
        assert {"type": search_bar.HIT_TYPE, "ticker": "AAPL"} in ids_of(tree)

    def test_empty_query_is_no_error(self, state: AppState) -> None:
        assert search_bar.run_search(state, "") is None
        assert search_bar.run_search(state, "   ") is None
        assert search_bar.run_search(state, None) is None

    def test_no_hits_shows_hint_not_crash(self, state: AppState) -> None:
        tree = search_bar.run_search(state, "zzzqqqxyz")
        assert tree is not None
        assert not has_class(tree, "bt-error")
        assert "No match" in text_of(tree)

    def test_provider_error_becomes_banner(self, state: AppState, monkeypatch) -> None:
        def boom(_query: str, *, limit: int = 10) -> list[Any]:
            raise ProviderUnavailableError("Quelle nicht erreichbar.", detail="technisch")

        monkeypatch.setattr(state.provider, "search", boom, raising=False)
        tree = search_bar.run_search(state, "apple")
        assert tree is not None
        assert has_class(tree, "bt-error")
        assert "Quelle nicht erreichbar." in text_of(tree)
        # Der technische Detailtext gehoert ins Log, nicht auf den Schirm.
        assert "technisch" not in text_of(tree)

    def test_provider_without_search_is_handled(self, state: AppState, monkeypatch) -> None:
        monkeypatch.delattr(type(state.provider), "search", raising=False)
        tree = search_bar.run_search(state, "apple")
        assert tree is not None
        assert has_class(tree, "bt-error")

    def test_respects_max_hits(self, state: AppState) -> None:
        tree = search_bar.run_search(state, "a")
        assert tree is not None
        hits = [i for i in ids_of(tree) if isinstance(i, dict) and i["type"] == search_bar.HIT_TYPE]
        assert len(hits) <= search_bar.MAX_HITS


class TestHitPresentation:
    """Was eine Trefferzeile zeigt.

    Der Nutzer suchte "rocket lab" und sah eine Zeile, die mit dem
    Kuerzel begann. Wer den Namen sucht, sucht ihn auch in der Liste -
    also fuehrt der volle Firmenname die Zeile an.
    """

    def _hit(self, **kwargs: Any) -> SearchHit:
        from btlite.models.market import SearchHit

        base: dict[str, Any] = {
            "ticker": "RKLB",
            "name": "Rocket Lab Corporation",
            "exchange": "NASDAQ",
        }
        base.update(kwargs)
        return SearchHit(**base)

    def test_full_company_name_is_shown(self) -> None:
        text = text_of(search_bar.hit_button(self._hit()))
        assert "Rocket Lab Corporation" in text

    def test_company_name_leads_the_line(self) -> None:
        """Der Name traegt die Zeile, das Kuerzel steht darunter."""
        button = search_bar.hit_button(self._hit())
        headline = next(
            child for child in button.children if getattr(child, "className", "") == "bt-hit-name"
        )
        assert headline.children == "Rocket Lab Corporation"

    def test_ticker_and_exchange_stay_visible(self) -> None:
        text = text_of(search_bar.hit_button(self._hit()))
        assert "NASDAQ:RKLB" in text

    def test_asset_class_is_spelled_out(self) -> None:
        """Nicht der Rohwert "equity" der Quelle - der Nutzer liest "Stock"."""
        text = text_of(search_bar.hit_button(self._hit()))
        assert "Stock" in text
        assert "equity" not in text

    def test_every_asset_class_has_a_german_label(self) -> None:
        from btlite.models.common import AssetClass

        for member in AssetClass:
            assert str(member) in search_bar.ASSET_CLASS_LABELS

    def test_unknown_asset_class_is_not_swallowed(self) -> None:
        """Eine neue Klasse der Quelle soll sichtbar sein, nicht stumm."""
        hit = self._hit()
        assert search_bar.asset_class_label(hit) == "Stock"
        object.__setattr__(hit, "asset_class", "spac")
        assert search_bar.asset_class_label(hit) == "spac"

    def test_sector_is_shown_when_known(self) -> None:
        text = text_of(search_bar.hit_button(self._hit(sector="Industrials")))
        assert "Industrials" in text

    def test_missing_name_falls_back_to_the_ticker(self) -> None:
        """Ohne Namen darf die Zeile nicht ohne Kopf dastehen."""
        text = text_of(search_bar.hit_button(self._hit(name=None)))
        assert "NASDAQ:RKLB" in text

    def test_list_says_what_a_click_does(self, state: AppState) -> None:
        text = text_of(search_bar.run_search(state, "servicenow"))
        assert "profile" in text


class TestHitOpensTheProfile:
    """Ein Klick auf den Treffer fuehrt direkt auf die Uebersichtsseite."""

    def test_target_is_the_profile_page(self) -> None:
        assert search_bar.hit_href("NOW") == f"/des?{TICKER_PARAM}=NOW"

    def test_target_mnemonic_is_a_real_page(self) -> None:
        assert search_bar.TARGET_MNEMONIC in MNEMONICS

    def test_roundtrips_through_parse_href(self) -> None:
        href = search_bar.hit_href("NOW")
        path, _, query = href.partition("?")
        mnemonic, tickers, args = parse_href(path, f"?{query}")
        assert mnemonic == search_bar.TARGET_MNEMONIC
        assert tickers == ["NOW"]
        assert args == ()

    def test_uses_the_shared_query_parameters(self) -> None:
        href = search_bar.hit_href("AAPL")
        assert f"{TICKER_PARAM}=" in href
        assert f"{ARGS_PARAM}=" not in href

    def test_no_intermediate_choice_remains(self) -> None:
        """Die Zwischenauswahl ist weg - sonst gaebe es zwei Wege."""
        assert not hasattr(search_bar, "analysis_choice")

    def test_the_profile_renders_for_a_found_title(self, state: AppState) -> None:
        """Das Ziel darf nie in einem Fehler enden."""
        from btlite.ui.app import render_page

        tree = render_page(state, search_bar.TARGET_MNEMONIC, ["AAPL"], ())
        assert not has_class(tree, "bt-error")

    def test_profile_works_without_statements(self, state: AppState) -> None:
        """Der Grund fuer die Wahl des Steckbriefs: er zeigt immer etwas.

        Ein Titel ohne Abschluesse laeuft auf der Bilanzseite ins Leere,
        auf dem Steckbrief bleiben die Stammdaten.
        """
        from btlite.ui.app import render_page

        tree = render_page(state, search_bar.TARGET_MNEMONIC, ["SPY"], ())
        assert not has_class(tree, "bt-error")


class TestComponents:
    """Rahmen und Stil der Bausteine."""

    def test_search_field_has_expected_id(self) -> None:
        assert search_bar.SEARCH_INPUT in ids_of(search_bar.search_field())

    def test_result_container_starts_empty(self) -> None:
        container = search_bar.result_container()
        assert container.id == search_bar.RESULT_OUTPUT
        assert not getattr(container, "children", None)

    def test_only_existing_css_classes_are_used(self) -> None:
        # ``btlite.css`` gehoert einem anderen Arbeitsschritt; dieses Modul
        # darf deshalb ausschliesslich bereits definierte Klassen vergeben.
        # ``test_ui_styling.test_every_assigned_class_has_a_rule`` prueft das
        # global, hier steht es nochmals fuer dieses Modul allein.
        import re
        from pathlib import Path

        module = Path(search_bar.__file__).read_text(encoding="utf8")
        used = set()
        for literal in re.findall(r'className="([^"]*)"', module):
            used.update(literal.split())

        css = Path(__file__).resolve().parents[2]
        css = css / "src" / "btlite" / "ui" / "assets" / "btlite.css"
        body = re.sub(r"/\*.*?\*/", "", css.read_text(encoding="utf8"), flags=re.DOTALL)
        selectors = " ".join(re.findall(r"([^{}]+)\{", body))
        declared = set(re.findall(r"\.([A-Za-z0-9_-]+)", selectors))
        assert used <= declared, f"Ohne CSS-Regel: {sorted(used - declared)}"


def _pattern_id(**parts: str) -> str:
    """Baut die ID eines Pattern-Matching-Elements, wie Dash sie sendet.

    Dash serialisiert Wildcard-IDs mit alphabetisch sortierten Schluesseln
    und ohne Leerzeichen.

    Args:
        **parts: Felder der ID.

    Returns:
        JSON-Text der ID.
    """
    return json.dumps(dict(sorted(parts.items())), separators=(",", ":"))


def _outputs_of(key: str) -> Any:
    """Zerlegt einen Schluessel der ``callback_map`` in Output-Spezifikationen.

    Mehrfach-Outputs stehen als ``..a.b...c.d..``; ``allow_duplicate`` haengt
    hinter den Property-Namen ein ``@<hash>``, das der Server nicht im Spec
    erwartet.

    Wichtig: ein Callback mit genau **einem** Output bekommt die Spezifikation
    als blanke Abbildung, nicht als einelementige Liste. Der Browser sendet es
    so, und Dash leitet aus einer Liste sonst einen Wildcard-Multi-Output ab
    und verwirft die Rueckgabe.

    Args:
        key: Schluessel aus ``app.callback_map``.

    Returns:
        Eine Abbildung bei einem Output, sonst eine Liste davon.
    """
    if not key.startswith(".."):
        element_id, _, prop = key.partition(".")
        return {"id": element_id, "property": prop.split("@")[0]}
    specs: list[dict[str, str]] = []
    for part in key[2:-2].split("..."):
        element_id, _, prop = part.partition(".")
        specs.append({"id": element_id, "property": prop.split("@")[0]})
    return specs


@pytest.fixture
def app(state: AppState) -> Any:
    """Baut die App mit dem Fixture-Zustand.

    Auf Modulebene, weil mehrere Testklassen ueber HTTP pruefen - eine
    zweite Kopie in der naechsten Klasse waere nur eine Fehlerquelle.
    """
    return create_app(state.settings, state)


@pytest.fixture
def client(app: Any) -> Any:
    """Liefert einen Testclient auf den Flask-Server der App."""
    return app.server.test_client()


class TestCallbacksInTheApp:
    """Die Verdrahtung in der Shell, ueber den echten Dash-Dispatch.

    Die Callbacks werden nicht direkt aufgerufen, sondern ueber
    ``/_dash-update-component`` - also genau den Weg, den auch der Browser
    nimmt. Nur so faellt auf, wenn Dash eine Rueckgabe ablehnt: ein
    Wildcard-Callback (``ALL``-Eingabe), der eine blanke Komponente statt
    einer Liste liefert, quittiert der Server mit 500, waehrend ein
    direkter Aufruf der Funktion problemlos durchliefe.
    """

    def _key(self, app: Any, predicate: Any) -> str:
        """Sucht den Schluessel eines registrierten Callbacks.

        Args:
            app: Gebaute Dash-App.
            predicate: Praedikat ueber den Schluessel.

        Returns:
            Der passende Schluessel der ``callback_map``.
        """
        return next(k for k in app.callback_map if predicate(k))

    def _dispatch(self, client: Any, key: str, inputs: list[Any], changed: list[str]) -> Any:
        """Ruft einen Callback ueber den HTTP-Endpunkt auf.

        Args:
            client: Flask-Testclient.
            key: Schluessel des Callbacks.
            inputs: Eingaben in der Form, die Dash sendet.
            changed: ``changedPropIds`` der Anfrage.

        Returns:
            Dekodierte Antwort des Servers.

        Raises:
            AssertionError: Wenn der Server nicht mit 200 antwortet.
        """
        response = client.post(
            "/_dash-update-component",
            json={
                "output": key,
                "outputs": _outputs_of(key),
                "inputs": inputs,
                "changedPropIds": changed,
            },
        )
        body = response.get_data(as_text=True)
        assert response.status_code == 200, f"Dash lehnte die Rueckgabe ab: {body[:400]}"
        return json.loads(body)

    def test_index_page_is_served(self, client: Any) -> None:
        # Dash liefert nur die Huelle; das Layout baut der Renderer im
        # Browser. Geprueft wird deshalb, dass die Huelle ausgeliefert wird
        # und das Layout ueber den Layout-Endpunkt die Suchleiste enthaelt.
        assert client.get("/").status_code == 200
        layout = client.get("/_dash-layout")
        assert layout.status_code == 200
        assert search_bar.SEARCH_INPUT in layout.get_data(as_text=True)

    def test_search_field_is_in_the_layout(self, app: Any) -> None:
        assert search_bar.SEARCH_INPUT in ids_of(app.layout)
        assert search_bar.RESULT_OUTPUT in ids_of(app.layout)

    def test_search_over_http_finds_now(self, app: Any, client: Any) -> None:
        key = self._key(app, lambda k: k == f"{search_bar.RESULT_OUTPUT}.children")
        payload = self._dispatch(
            client,
            key,
            [{"id": search_bar.SEARCH_INPUT, "property": "value", "value": "servicenow"}],
            [f"{search_bar.SEARCH_INPUT}.value"],
        )
        blob = json.dumps(payload, ensure_ascii=False)
        assert "NYSE:NOW" in blob
        assert "ServiceNow" in blob

    def test_search_over_http_with_empty_input(self, app: Any, client: Any) -> None:
        key = self._key(app, lambda k: k == f"{search_bar.RESULT_OUTPUT}.children")
        payload = self._dispatch(
            client,
            key,
            [{"id": search_bar.SEARCH_INPUT, "property": "value", "value": ""}],
            [f"{search_bar.SEARCH_INPUT}.value"],
        )
        assert payload["response"][search_bar.RESULT_OUTPUT]["children"] is None

    def test_clicking_a_hit_opens_the_profile(self, app: Any, client: Any) -> None:
        """Der gemeldete Fall: ein Klick auf den Namen muss die Seite oeffnen.

        Geprueft ueber HTTP gegen den echten Callback, nicht gegen die
        Funktion allein - der Fehler lag zuletzt genau in der Verdrahtung.
        """
        key = self._key(app, lambda k: "url.pathname@" in k and "symbol-search.value" in k)
        element_id = _pattern_id(type=search_bar.HIT_TYPE, ticker="NOW")
        payload = self._dispatch(
            client,
            key,
            [
                [
                    {
                        "id": {"type": search_bar.HIT_TYPE, "ticker": "NOW"},
                        "property": "n_clicks",
                        "value": 1,
                    }
                ]
            ],
            [f"{element_id}.n_clicks"],
        )
        response = payload["response"]
        assert response["url"]["pathname"] == "/des"
        assert response["url"]["search"] == f"?{TICKER_PARAM}=NOW"

    def test_clicking_a_hit_clears_the_search(self, app: Any, client: Any) -> None:
        """Die geoeffnete Seite ist die Antwort - die Liste darf nicht stehenbleiben."""
        key = self._key(app, lambda k: "url.pathname@" in k and "symbol-search.value" in k)
        element_id = _pattern_id(type=search_bar.HIT_TYPE, ticker="NOW")
        payload = self._dispatch(
            client,
            key,
            [
                [
                    {
                        "id": {"type": search_bar.HIT_TYPE, "ticker": "NOW"},
                        "property": "n_clicks",
                        "value": 1,
                    }
                ]
            ],
            [f"{element_id}.n_clicks"],
        )
        response = payload["response"]
        assert response[search_bar.SEARCH_INPUT]["value"] == ""
        assert response[search_bar.RESULT_OUTPUT]["children"] is None

    def test_hit_list_over_http_shows_the_full_name(self, app: Any, client: Any) -> None:
        """Der Vollname muss im ausgelieferten Baum stehen, nicht nur lokal."""
        key = self._key(app, lambda k: k == f"{search_bar.RESULT_OUTPUT}.children")
        payload = self._dispatch(
            client,
            key,
            [{"id": search_bar.SEARCH_INPUT, "property": "value", "value": "servicenow"}],
            [f"{search_bar.SEARCH_INPUT}.value"],
        )
        blob = json.dumps(payload, ensure_ascii=False)
        assert "ServiceNow" in blob
        assert "bt-hit-name" in blob

    def test_route_from_search_renders_without_error(self, app: Any, client: Any) -> None:
        # Der Vollzug: die Adresse, die die Auswahl erzeugt, baut auch
        # wirklich eine Seite - hier mit einem Ticker, den die Fixtures
        # vollstaendig bedienen.
        href = search_bar.hit_href("AAPL")
        path, _, query = href.partition("?")
        payload = self._dispatch(
            client,
            "page-content.children",
            [
                {"id": "url", "property": "pathname", "value": path},
                {"id": "url", "property": "search", "value": f"?{query}"},
            ],
            ["url.pathname"],
        )
        blob = json.dumps(payload, ensure_ascii=False)
        assert "Profile" in blob
        assert '"bt-error"' not in blob


class TestWatchlistAction:
    """Die Watchlist als Aktion auf der Steckbriefseite.

    Der Nutzer wollte Titel speichern koennen. Seit der Klick auf einen
    Treffer direkt den Steckbrief oeffnet, sitzt die Schaltflaeche dort:
    der Titel steht mit Kurs und Stammdaten vor ihm, wenn er entscheidet.
    """

    def test_button_is_offered_on_the_profile(self, state: AppState) -> None:
        from btlite.ui.app import render_page

        ids = ids_of(render_page(state, "DES", ["AAPL"], ()))
        types = {i["type"] for i in ids if isinstance(i, dict) and "type" in i}
        assert search_bar.WATCH_TYPE in types

    def test_button_is_labelled_in_plain_german(self, state: AppState) -> None:
        from btlite.ui.app import render_page

        text = text_of(render_page(state, "DES", ["AAPL"], ()))
        assert "Watchlist" in text

    def test_feedback_container_is_on_the_profile(self, state: AppState) -> None:
        """Die Rueckmeldung braucht ein Ziel auf derselben Seite."""
        from btlite.ui.app import render_page
        from btlite.ui.pages import des

        assert des.WATCH_FEEDBACK in ids_of(render_page(state, "DES", ["AAPL"], ()))

    def test_adding_creates_the_list_if_absent(self, state: AppState) -> None:
        """Ein Klick soll genuegen, kein vorheriges Anlegen noetig."""
        assert state.holdings.get_watchlist(search_bar.DEFAULT_WATCHLIST) is None
        search_bar.add_to_watchlist(state, "NOW")
        watchlist = state.holdings.get_watchlist(search_bar.DEFAULT_WATCHLIST)
        assert watchlist is not None
        assert "NOW" in watchlist.tickers

    def test_confirmation_is_shown(self, state: AppState) -> None:
        rendered = str(search_bar.add_to_watchlist(state, "NOW"))
        assert "NOW" in rendered
        assert "bt-error" not in rendered

    def test_adding_twice_is_a_notice_not_an_error(self, state: AppState) -> None:
        """Der Titel steht danach drauf - genau das wollte der Nutzer."""
        search_bar.add_to_watchlist(state, "NOW")
        rendered = str(search_bar.add_to_watchlist(state, "NOW"))
        assert "already" in rendered
        assert "bt-error" not in rendered

    def test_entry_survives_and_appears_on_the_page(self, state: AppState) -> None:
        """Querprobe ueber den echten Seiten-Renderer, nicht ueber den Store."""
        from btlite.ui.app import render_page

        search_bar.add_to_watchlist(state, "NOW")
        page = str(render_page(state, "WL", [], ()))
        assert "NOW" in page
        assert "bt-error" not in page

    def test_invalid_ticker_yields_an_error_banner(self, state: AppState) -> None:
        rendered = str(search_bar.add_to_watchlist(state, "!!!"))
        assert "bt-error" in rendered

    def test_watchlist_path_matches_the_registered_route(self) -> None:
        """Eine umbenannte Route darf die Aktion nicht ins Leere laufen lassen."""
        from btlite.ui.pages import ROUTES

        assert search_bar.WATCHLIST_PATH in ROUTES
        assert ROUTES[search_bar.WATCHLIST_PATH] == "WL"


class TestSelectionSurvivesNavigation:
    """Der gewaehlte Titel bleibt beim Wechsel der Seite erhalten.

    Der Nutzer verlangte: "wenn man nach einem stock such und diesen
    anklickt, soll dieser nun der ausgewaehlte sein, wenn man sich durch
    describtion oder financial analysis klickt." Vorher waren die
    Navigationslinks feste Pfade ohne Ticker, und jede Seite faellt ohne
    ``?t=`` auf ``default_tickers`` zurueck - ein Klick auf "Bilanz"
    zeigte also AAPL statt des gesuchten Titels.
    """

    def _hrefs(self, app: Any, client: Any, search: str) -> dict[str, str]:
        """Fragt die Adressen aller Navigationslinks ueber HTTP ab."""
        from btlite.ui.app import NAV_LINK_TYPE

        key = next(k for k in app.callback_map if NAV_LINK_TYPE in k and "href" in k)
        specs = [
            {"id": {"type": NAV_LINK_TYPE, "mnemonic": m}, "property": "href"} for m in MNEMONICS
        ]
        response = client.post(
            "/_dash-update-component",
            json={
                "output": key,
                "outputs": specs,
                "inputs": [
                    {"id": "url", "property": "pathname", "value": "/des"},
                    {"id": "url", "property": "search", "value": search},
                ],
                "changedPropIds": ["url.pathname"],
            },
        )
        assert response.status_code == 200, response.get_data(as_text=True)[:400]
        payload = response.get_json()["response"]
        return {
            m: payload[
                json.dumps(
                    {"mnemonic": m, "type": NAV_LINK_TYPE}, separators=(",", ":"), sort_keys=True
                )
            ]["href"]
            for m in MNEMONICS
        }

    def test_single_ticker_pages_keep_the_selection(self, app: Any, client: Any) -> None:
        """Genau der gemeldete Fall: Steckbrief und Bilanz behalten RKLB."""
        hrefs = self._hrefs(app, client, f"?{TICKER_PARAM}=RKLB")
        for mnemonic in ("DES", "FA", "GP", "BT"):
            assert hrefs[mnemonic] == f"/{mnemonic.lower()}?{TICKER_PARAM}=RKLB", mnemonic

    def test_universe_pages_stay_without_a_ticker(self, app: Any, client: Any) -> None:
        """EQS screent ein Universum, WL liest den Bestand - beide ohne Titel."""
        from btlite.ui.command_bar import TICKERLESS_MNEMONICS

        hrefs = self._hrefs(app, client, f"?{TICKER_PARAM}=RKLB")
        for mnemonic in TICKERLESS_MNEMONICS:
            assert hrefs[mnemonic] == f"/{mnemonic.lower()}", mnemonic

    def test_without_a_selection_links_stay_plain(self, app: Any, client: Any) -> None:
        hrefs = self._hrefs(app, client, "")
        assert all("?" not in href for href in hrefs.values()), hrefs

    def test_every_navigation_entry_is_covered(self, app: Any, client: Any) -> None:
        """Kein Menuepunkt darf beim Nachfuehren vergessen werden."""
        assert set(self._hrefs(app, client, f"?{TICKER_PARAM}=RKLB")) == set(MNEMONICS)

    def test_the_page_really_renders_the_selected_title(self, state: AppState) -> None:
        """Der Vollzug: die nachgefuehrte Adresse zeigt auch diesen Titel."""
        from btlite.ui.app import render_page

        page = str(render_page(state, "FA", ["MSFT"], ()))
        assert "MSFT" in page
        assert "bt-error" not in page


class TestPortfolioAction:
    """Die Depot-Aufnahme direkt an der Trefferzeile.

    Der Nutzer wollte einen Knopf, der den Treffer ins Depot legt. Anders
    als bei der Watchlist geht das nicht mit einem Klick: ``Holding``
    verlangt Stueckzahl und Einstandskurs beide groesser null. Eine
    Position mit Nullwerten waere kein Schnellweg, sondern ein falscher
    Depotwert.
    """

    def test_both_actions_are_on_the_hit_row(self, state: AppState) -> None:
        ids = ids_of(search_bar.run_search(state, "servicenow"))
        types = {i["type"] for i in ids if isinstance(i, dict) and "type" in i}
        assert search_bar.WATCH_TYPE in types
        assert search_bar.PORTFOLIO_TOGGLE_TYPE in types

    def test_the_form_starts_hidden(self, state: AppState) -> None:
        """Zwei Felder je Treffer wuerden die Liste sonst unlesbar machen."""
        form = search_bar.portfolio_form("NOW")
        assert form.hidden is True

    def test_the_form_asks_for_both_required_values(self, state: AppState) -> None:
        text = text_of(search_bar.portfolio_form("NOW"))
        assert "Shares" in text
        assert "Buy price" in text

    def test_adding_creates_the_portfolio_if_absent(self, state: AppState) -> None:
        assert state.holdings.get_portfolio(search_bar.DEFAULT_PORTFOLIO) is None
        search_bar.add_to_portfolio(state, "NOW", 25, 142.5)
        portfolio = state.holdings.get_portfolio(search_bar.DEFAULT_PORTFOLIO)
        assert portfolio is not None
        assert [h.ticker for h in portfolio.holdings] == ["NOW"]

    def test_the_stored_values_are_the_entered_ones(self, state: AppState) -> None:
        search_bar.add_to_portfolio(state, "NOW", 25, 142.5)
        holding = state.holdings.get_portfolio(search_bar.DEFAULT_PORTFOLIO).holdings[0]
        assert holding.shares == 25
        assert holding.buy_price == 142.5
        assert holding.cost_basis == pytest.approx(3562.5)

    @pytest.mark.parametrize(
        ("shares", "price"),
        [(None, None), (25, None), (None, 142.5)],
    )
    def test_missing_values_are_a_notice_not_a_crash(
        self, state: AppState, shares: float | None, price: float | None
    ) -> None:
        rendered = str(search_bar.add_to_portfolio(state, "NOW", shares, price))
        assert "Enter both" in rendered
        assert state.holdings.get_portfolio(search_bar.DEFAULT_PORTFOLIO) is None

    @pytest.mark.parametrize(("shares", "price"), [(0, 142.5), (25, 0), (-1, 142.5), (25, -5)])
    def test_non_positive_values_are_refused(
        self, state: AppState, shares: float, price: float
    ) -> None:
        """Das Datenmodell verlangt beide groesser null - nicht stillschweigend runden."""
        rendered = str(search_bar.add_to_portfolio(state, "NOW", shares, price))
        assert "bt-error" in rendered
        assert state.holdings.get_portfolio(search_bar.DEFAULT_PORTFOLIO) is None

    def test_invalid_ticker_yields_an_error_banner(self, state: AppState) -> None:
        assert "bt-error" in str(search_bar.add_to_portfolio(state, "!!!", 25, 142.5))

    def test_entry_appears_on_the_portfolio_page(self, state: AppState) -> None:
        """Querprobe ueber den echten Seiten-Renderer, nicht ueber den Store."""
        from btlite.ui.app import render_page

        search_bar.add_to_portfolio(state, "NOW", 25, 142.5)
        page = str(render_page(state, "WL", [], ()))
        assert "NOW" in page
        assert "bt-error" not in page


class TestSearchFieldIsUsable:
    """Das Suchfeld muss in der Kopfzeile auch Platz bekommen.

    Der Nutzer meldete "die suche funktioniert nicht". Serverseitig war
    alles korrekt - das Feld sass aber in ``.bt-control``, der Klasse fuer
    Formularspalten (``flex-direction: column``, ``min-width: 140px``),
    und die Kopfzeile hatte eine starre ``height`` von 44px. Neben der
    Kommandozeile mit ``flex: 1 1 auto`` blieb nichts uebrig.
    """

    def _css(self) -> str:
        from pathlib import Path

        import btlite.ui as ui_pkg

        return (Path(ui_pkg.__file__).parent / "assets" / "btlite.css").read_text(encoding="utf-8")

    def test_field_does_not_use_the_form_column_class(self) -> None:
        """bt-control ist eine senkrechte Formularspalte, keine Kopfzeile."""
        from dash import html

        field = search_bar.search_field()
        assert isinstance(field, html.Div)
        assert "bt-control" not in (field.className or "")

    def test_field_and_results_have_their_own_classes(self) -> None:
        assert "bt-search" in (search_bar.search_field().className or "")
        assert "bt-search-results" in (search_bar.result_container().className or "")

    def test_every_search_class_has_a_rule(self) -> None:
        """Ohne Regel erbt das Feld Fremdfarben und Fremdmasse."""
        css = self._css()
        for cls in ("bt-search", "bt-search-input", "bt-search-results"):
            assert f".{cls}" in css, f"{cls} hat keine CSS-Regel"

    def test_header_can_grow_and_wrap(self) -> None:
        """Eine starre Hoehe quetscht das Suchfeld zusammen."""
        css = self._css()
        header = css[css.index(".bt-header {") : css.index(".bt-header {") + 400]
        assert "min-height:" in header
        assert "flex-wrap: wrap" in header
        # Eine feste height wuerde min-height wirkungslos machen.
        assert "\n  height:" not in header

    def test_search_input_has_a_usable_width(self) -> None:
        css = self._css()
        block = css[css.index(".bt-search {") : css.index(".bt-search {") + 200]
        assert "min-width" in block or "flex:" in block


class TestResultsAreActuallyVisible:
    """Die Trefferliste muss man auch sehen.

    Der Nutzer meldete zu "rocket lab": "es öffnet sich nichts". Der
    Server lieferte drei Treffer, sein eigener Server ebenso - die
    Antwort kam also an. Sie fiel nur optisch durch: kein eigener
    Hintergrund zwischen zwei Flaechen, die einen haben, und beim
    Scrollen wanderte sie aus dem Bild, waehrend das Suchfeld in der
    haftenden Kopfzeile stehen blieb.
    """

    def _css(self) -> str:
        from pathlib import Path

        import btlite.ui as ui_pkg

        return (Path(ui_pkg.__file__).parent / "assets" / "btlite.css").read_text(encoding="utf-8")

    def _block(self) -> str:
        css = self._css()
        start = css.index(".bt-search-results {")
        return css[start : css.index("}", start)]

    def test_results_have_their_own_surface(self) -> None:
        """Ohne Hintergrund verschwindet die Liste zwischen den Nachbarn."""
        assert "background-color" in self._block()

    def test_results_are_set_off_from_the_page(self) -> None:
        block = self._block()
        assert "border-bottom" in block or "border" in block

    def test_results_have_vertical_padding(self) -> None:
        """padding: 0 12px liess die Kacheln am Rand kleben."""
        block = self._block()
        padding = [line for line in block.splitlines() if "padding:" in line]
        assert padding, "kein padding gesetzt"
        assert "0 12px" not in padding[0], "kein Abstand nach oben und unten"

    def test_results_follow_the_sticky_header(self) -> None:
        """Das Suchfeld bleibt beim Scrollen stehen - die Antwort auch."""
        assert "position: sticky" in self._block()

    def test_results_stay_below_the_header(self) -> None:
        """Sonst legt sich die Liste ueber die Kommandozeile."""
        css = self._css()
        header = css[css.index(".bt-header {") : css.index(".bt-header {") + 400]
        header_z = int(re.search(r"z-index:\s*(\d+)", header).group(1))
        results_z = int(re.search(r"z-index:\s*(\d+)", self._block()).group(1))
        assert results_z < header_z

    def test_long_result_lists_scroll_inside(self) -> None:
        """Acht Treffer duerfen die halbe Seite nicht verdecken."""
        block = self._block()
        assert "max-height" in block
        assert "overflow-y" in block
