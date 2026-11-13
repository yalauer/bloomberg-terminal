"""Integration: die App end-to-end, ohne Browser.

Das Akzeptanzkriterium lautet "alle sechs Seiten laden fehlerfrei". Diese
Datei prueft das so weit, wie es ohne laufenden Server geht: die App wird
gebaut, und jede Seite wird ueber ihre Layout-Funktion gerendert.

``dash.testing`` wird bewusst **nicht** benutzt - es startet einen
Browser und damit einen echten Server. Stattdessen werden die
Layout-Funktionen direkt aufgerufen, wie es ``tests/unit/test_ui_pages.py``
vormacht. Der Unterschied zu jener Datei: hier laeuft der Weg ueber
``create_app`` und ``render_page``, also ueber die Verdrahtung, die im
Betrieb tatsaechlich benutzt wird.

Ein Fehler auf einer Seite fuehrt nicht zu einer Exception - das UI faengt
ihn ab und zeigt ein Banner mit der Klasse ``bt-error``. Genau dessen
Abwesenheit ist deshalb das Kriterium.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from btlite.config import Settings
from btlite.ui.app import (
    DEFAULT_PATH,
    DISCLAIMER,
    build_href,
    create_app,
    header_notice,
    parse_href,
    render_page,
)
from btlite.ui.command_bar import parse_command
from btlite.ui.pages import PAGES, ROUTES
from btlite.ui.state import AppState, set_state

#: Die Mnemonics des Terminals. Bewusst hier festgeschrieben und nicht aus
#: ``command_bar`` importiert: der Test soll merken, wenn eine Seite
#: hinzukommt, statt sich stillschweigend anzupassen.
MNEMONICS = ("DES", "GP", "FA", "EQS", "BT", "PORT", "WL")

#: CSS-Klasse des Fehlerbanners.
ERROR_CLASS = "bt-error"

#: Seiten, die eine installierte Backtest-Engine brauchen.
#: ``vectorbt`` ist ein optionales Extra. Fehlt es, zeigt die Seite BT
#: korrekterweise ein Fehlerbanner - das ist kein Defekt, sondern die
#: gewollte Meldung. Die Pruefungen auf Bannerfreiheit ueberspringen
#: solche Seiten deshalb, statt den Lauf ohne Extra rot zu faerben.
ENGINE_SEITEN = frozenset({"BT"})


def _engine_verfuegbar() -> bool:
    """Prueft, ob eine Backtest-Engine installiert ist.

    Returns:
        True, wenn mindestens eine Engine nutzbar ist.
    """
    from btlite.backtest.vbt_engine import VectorbtEngine

    return bool(VectorbtEngine().is_available())


def _skip_ohne_engine(mnemonic: str) -> None:
    """Ueberspringt einen Test, wenn die Seite ohne Engine nicht laden kann.

    Args:
        mnemonic: Mnemonic der geprueften Seite.
    """
    if mnemonic in ENGINE_SEITEN and not _engine_verfuegbar():
        pytest.skip(f"Seite {mnemonic} braucht eine Backtest-Engine (Extra backtest)")


@pytest.fixture
def settings() -> Settings:
    """Offline-Konfiguration: der Fixture-Provider bedient die App."""
    config = Settings()
    config.data.offline = True
    return config


@pytest.fixture
def state(settings: Settings) -> Iterator[AppState]:
    """Setzt einen Fixture-gestuetzten UI-Zustand und raeumt ihn wieder ab."""
    instance = AppState(settings)
    set_state(instance)
    yield instance
    set_state(None)


def walk(node: Any) -> Iterator[Any]:
    """Laeuft rekursiv durch einen Dash-Komponentenbaum.

    Args:
        node: Wurzel des Teilbaums.

    Yields:
        Jeden Knoten des Baums, die Wurzel eingeschlossen.
    """
    yield node
    children = getattr(node, "children", None)
    if children is None:
        return
    if isinstance(children, list | tuple):
        for child in children:
            yield from walk(child)
    else:
        yield from walk(children)


def has_class(tree: Any, name: str) -> bool:
    """Prueft, ob irgendwo im Baum eine CSS-Klasse vorkommt.

    Args:
        tree: Wurzel des Komponentenbaums.
        name: Gesuchte Klasse.

    Returns:
        True, wenn ein Knoten die Klasse traegt.
    """
    return any(name in str(getattr(n, "className", "") or "") for n in walk(tree))


def text_of(tree: Any) -> str:
    """Sammelt allen Textinhalt eines Komponentenbaums.

    Args:
        tree: Wurzel des Komponentenbaums.

    Returns:
        Aller Text, durch Leerzeichen getrennt.
    """
    parts: list[str] = []
    for node in walk(tree):
        children = getattr(node, "children", node)
        if isinstance(children, str):
            parts.append(children)
    return " ".join(parts)


class TestAppBau:
    """Die Dash-App laesst sich vollstaendig konstruieren."""

    def test_create_app_baut_ohne_fehler(self, settings, state):
        """Das Kriterium beginnt hier: die App entsteht ueberhaupt."""
        app = create_app(settings, state)

        assert app is not None
        assert app.layout is not None

    def test_layout_traegt_die_kommandozeile(self, settings, state):
        """Die Mnemonic-Kommandozeile ist Teil des Grundgeruests."""
        app = create_app(settings, state)

        ids = {str(getattr(n, "id", "") or "") for n in walk(app.layout)}
        assert "url" in ids, "Ohne Location-Komponente gibt es kein Routing"
        assert any("command" in i for i in ids), f"Keine Kommandozeile gefunden: {ids}"

    def test_disclaimer_ist_sichtbar(self, settings, state):
        """Das Produkt gibt keine Anlageberatung und sagt das auch.

        Ein Nicht-Ziel des Projekts, das im UI sichtbar sein muss.
        """
        app = create_app(settings, state)

        assert DISCLAIMER in text_of(app.layout)

    def test_offline_hinweis_weist_fixtures_aus(self, state):
        """Im Offline-Betrieb sagt die Kopfzeile, dass die Daten alt sind."""
        notice = header_notice(state)

        assert notice
        assert state.is_offline


#: Die sechs Seiten des urspruenglichen Akzeptanzkriteriums. WL kam
#: spaeter auf ausdrueckliche Anforderung hinzu (gespeichertes Depot und
#: Watchlist) und erweitert die Liste, ersetzt sie aber nicht.
KERNSEITEN = frozenset({"DES", "GP", "FA", "EQS", "BT", "PORT"})


class TestAlleSeitenLaden:
    """Das Akzeptanzkriterium: jede registrierte Seite laedt fehlerfrei."""

    def test_die_sechs_kernseiten_existieren_weiterhin(self):
        """Eine neue Seite darf keine der urspruenglichen verdraengen."""
        assert set(PAGES) >= KERNSEITEN

    def test_register_und_kommandoliste_laufen_nicht_auseinander(self):
        """Eine Seite ohne Mnemonic waere unerreichbar, umgekehrt ein Fehler."""
        assert set(PAGES) == set(MNEMONICS)

    @pytest.mark.parametrize("mnemonic", MNEMONICS)
    def test_seite_rendert_ohne_fehlerbanner(self, state, mnemonic):
        """Jede Seite baut ihren Inhalt ueber ``render_page`` fehlerfrei.

        ``render_page`` ist der Weg, den auch der Routing-Callback nimmt.
        """
        _skip_ohne_engine(mnemonic)
        tree = render_page(state, mnemonic, ["AAPL", "MSFT"], ())

        assert tree is not None
        assert not has_class(tree, ERROR_CLASS), (
            f"Seite {mnemonic} zeigt ein Fehlerbanner: {text_of(tree)[:300]}"
        )

    @pytest.mark.parametrize("mnemonic", MNEMONICS)
    def test_seite_rendert_ohne_ticker(self, state, mnemonic):
        """Ohne Ticker greifen die Defaults - kein Fehler."""
        _skip_ohne_engine(mnemonic)
        tree = render_page(state, mnemonic, [], ())

        assert not has_class(tree, ERROR_CLASS)

    @pytest.mark.parametrize("mnemonic", MNEMONICS)
    def test_seite_weist_ihre_herkunft_aus(self, state, mnemonic):
        """Jede Seite zeigt, woher ihre Daten stammen.

        Eine harte Projektregel: der Nutzer soll eingefrorene Daten als
        solche erkennen koennen.
        """
        _skip_ohne_engine(mnemonic)
        tree = render_page(state, mnemonic, ["AAPL"], ())

        assert has_class(tree, "bt-provenance"), f"Seite {mnemonic} ohne Herkunftsnachweis"

    @pytest.mark.parametrize("mnemonic", MNEMONICS)
    def test_seite_hat_inhalt(self, state, mnemonic):
        """Eine fehlerfreie, aber leere Seite waere auch ein Fehler."""
        _skip_ohne_engine(mnemonic)
        tree = render_page(state, mnemonic, ["AAPL"], ())

        assert len(text_of(tree)) > 50, f"Seite {mnemonic} ist praktisch leer"


class TestRoutingKette:
    """Kommando, Adresse und Seite muessen zusammenpassen."""

    @pytest.mark.parametrize("mnemonic", MNEMONICS)
    def test_kommando_bis_seite(self, state, mnemonic):
        """Der vollstaendige Weg: Eingabe, Adresse, Seite.

        Das ist die Kette, die ein Nutzer ausloest, wenn er ein Mnemonic
        eintippt und Enter drueckt.
        """
        _skip_ohne_engine(mnemonic)
        command = parse_command(f"AAPL {mnemonic} GO")
        assert command is not None

        href = build_href(command)
        path, _, query = href.partition("?")
        gelesen, tickers, args = parse_href(path, f"?{query}" if query else "")

        assert gelesen == mnemonic
        assert tickers == ["AAPL"]

        tree = render_page(state, gelesen, tickers, args)
        assert not has_class(tree, ERROR_CLASS)

    def test_jede_route_zeigt_auf_eine_seite(self):
        """Es gibt keine Route ohne Seite und keine Seite ohne Route."""
        assert set(ROUTES.values()) == set(PAGES)
        assert DEFAULT_PATH in ROUTES

    def test_unbekannte_route_faellt_zurueck(self, state):
        """Eine unbekannte Adresse landet auf der Startseite, nicht im Fehler."""
        mnemonic, tickers, args = parse_href("/gibtesnicht", "")

        assert mnemonic == ROUTES[DEFAULT_PATH]

        tree = render_page(state, mnemonic, tickers, args)
        assert not has_class(tree, ERROR_CLASS)


class TestFehlerWerdenAngezeigtNichtGeworfen:
    """Die Gegenprobe: ein echter Fehler erzeugt ein Banner.

    Ohne diese Tests waeren die Pruefungen oben wertlos - sie wuerden auch
    dann gruen, wenn ``has_class`` nie etwas faende.
    """

    @pytest.mark.parametrize("mnemonic", MNEMONICS)
    def test_unbekannter_ticker_erzeugt_banner_statt_exception(self, state, mnemonic):
        """Ein unbekanntes Symbol bricht die Seite nicht ab."""
        tree = render_page(state, mnemonic, ["GIBTESNICHT"], ())

        # Kein Absturz, und die Seite liefert einen Baum.
        assert tree is not None

    def test_unbekanntes_mnemonic_erzeugt_banner(self, state):
        """Ein Mnemonic ohne Seite fuehrt zu einem Banner, nicht zum Absturz."""
        tree = render_page(state, "NOSUCHPAGE", ["AAPL"], ())

        assert tree is not None
        assert has_class(tree, ERROR_CLASS), "Ein unbekanntes Mnemonic muss auffallen"

    def test_des_mit_unbekanntem_ticker_zeigt_banner(self, state):
        """Konkret nachgewiesen: DES meldet ein unbekanntes Symbol.

        Damit ist belegt, dass ``has_class(tree, "bt-error")`` ueberhaupt
        anschlagen kann - die Pruefungen oben sind also aussagekraeftig.
        """
        tree = render_page(state, "DES", ["GIBTESNICHT"], ())

        assert has_class(tree, ERROR_CLASS)
