"""Tests der Symbolsuche: Ranking, Fixture-Provider und Fehler-Mapping.

Alle Tests laufen offline. ``tests/conftest.py`` blockiert Sockets prozessweit,
ein versehentlicher Netzwerkzugriff schlaegt laut fehl - genau das prueft
:class:`TestOfflineGarantie` ausdruecklich.

Der Schwerpunkt liegt auf ``data/search.py``: die Ranking-Logik ist rein und
damit ohne Provider, ohne Netz und ohne Fixtures pruefbar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from btlite.core.errors import ProviderUnavailableError, RateLimitError
from btlite.data.base import SymbolSearch
from btlite.data.fixtures import FixtureProvider
from btlite.data.search import (
    deduplicate_hits,
    display_exchange,
    hit_from_quote,
    hits_from_quotes,
    normalize_name,
    normalize_query,
    prepare_results,
    query_slug,
    rank_hits,
)
from btlite.models.common import AssetClass
from btlite.models.market import SearchHit


def make_hit(
    ticker: str,
    name: str | None = None,
    *,
    exchange: str | None = "NASDAQ",
    asset_class: AssetClass = AssetClass.EQUITY,
    score: float | None = None,
) -> SearchHit:
    """Baut einen Treffer fuer die Ranking-Tests.

    Args:
        ticker: Kanonisches Symbol.
        name: Name des Emittenten.
        exchange: Anzeigename der Boerse.
        asset_class: Instrumentenklasse.
        score: Relevanz der Quelle, dimensionslos.

    Returns:
        Der gebaute Treffer.
    """
    return SearchHit(
        ticker=ticker,
        name=name,
        exchange=exchange,
        asset_class=asset_class,
        score=score,
    )


@pytest.fixture(scope="module")
def fixtures_root() -> Path:
    """Wurzel der eingefrorenen Fixtures.

    Returns:
        Pfad auf ``tests/fixtures``.
    """
    return Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def provider(fixtures_root: Path) -> FixtureProvider:
    """Ein FixtureProvider auf dem echten Fixture-Bestand.

    Args:
        fixtures_root: Wurzel der Fixtures.

    Returns:
        Der Provider.
    """
    return FixtureProvider(fixtures_dir=fixtures_root)


# --------------------------------------------------------------- Vertrag ---
class TestSearchHit:
    """Das Modell und seine Anzeige-Properties."""

    def test_display_verbindet_boerse_und_kuerzel(self) -> None:
        """``NYSE:NOW`` ist der Anzeigetext, den der Nutzer sehen will."""
        hit = make_hit("NOW", "ServiceNow, Inc.", exchange="NYSE")
        assert hit.display == "NYSE:NOW"

    def test_display_ohne_boerse_ist_nur_das_kuerzel(self) -> None:
        """Ohne Boerse entfaellt der Doppelpunkt, statt leer zu bleiben."""
        hit = make_hit("NOW", "ServiceNow, Inc.", exchange=None)
        assert hit.display == "NOW"

    def test_label_haengt_den_namen_an(self) -> None:
        """Die Trefferzeile nennt zusaetzlich den Emittenten."""
        hit = make_hit("NOW", "ServiceNow, Inc.", exchange="NYSE")
        assert hit.label == "NYSE:NOW - ServiceNow, Inc."

    def test_label_ohne_namen_faellt_auf_display_zurueck(self) -> None:
        """Ohne Namen bleibt die Zeile beim Anzeigetext."""
        hit = make_hit("NOW", None, exchange="NYSE")
        assert hit.label == "NYSE:NOW"

    def test_ticker_wird_normalisiert(self) -> None:
        """Der Vertrag erzwingt die kanonische Grossschreibung."""
        assert make_hit("  now  ").ticker == "NOW"

    def test_negativer_score_ist_ein_vertragsbruch(self) -> None:
        """``score`` ist mit ``ge=0`` deklariert."""
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            SearchHit(ticker="NOW", score=-1.0)


# ------------------------------------------------------- Normalisierung ---
class TestNormalisierung:
    """Die reinen Hilfsfunktionen der Vergleichsformen."""

    @pytest.mark.parametrize(
        ("roh", "erwartet"),
        [
            ("  ServiceNow  ", "servicenow"),
            ("Johnson & Johnson", "johnson johnson"),
            ("BRK-B", "brk b"),
            ("***", ""),
            ("   ", ""),
            ("", ""),
        ],
    )
    def test_normalize_query(self, roh: str, erwartet: str) -> None:
        """Kleinschreibung, Sonderzeichen weg, Leerzeichen zusammengezogen."""
        assert normalize_query(roh) == erwartet

    @pytest.mark.parametrize(
        ("roh", "erwartet"),
        [
            ("Apple Inc.", "apple"),
            ("Microsoft Corporation", "microsoft"),
            ("Johnson Controls International plc", "johnson controls international"),
            ("ServiceNow, Inc.", "servicenow"),
            (None, ""),
            ("", ""),
        ],
    )
    def test_normalize_name_entfernt_rechtsformen(self, roh: str | None, erwartet: str) -> None:
        """Rechtsformzusaetze duerfen den exakten Namenstreffer nicht verhindern."""
        assert normalize_name(roh) == erwartet

    def test_normalize_name_behaelt_einen_reinen_rechtsformnamen(self) -> None:
        """Ein Name, der nur aus einem Zusatz besteht, verschwindet nicht."""
        assert normalize_name("Inc") == "inc"

    @pytest.mark.parametrize(
        ("roh", "erwartet"),
        [("AAPL", "aapl"), ("aapl", "aapl"), ("service now", "service_now"), ("***", "")],
    )
    def test_query_slug(self, roh: str, erwartet: str) -> None:
        """Schreiber und Leser der Fixtures teilen sich diese Regel."""
        assert query_slug(roh) == erwartet

    @pytest.mark.parametrize(
        ("roh", "erwartet"),
        [("NMS", "NASDAQ"), ("NYQ", "NYSE"), ("NASDAQ", "NASDAQ"), ("XYZ", "XYZ"), (None, None)],
    )
    def test_display_exchange(self, roh: str | None, erwartet: str | None) -> None:
        """Interne Kuerzel werden lesbar, Unbekanntes bleibt stehen."""
        assert display_exchange(roh) == erwartet


# -------------------------------------------------------------- Ranking ---
class TestRanking:
    """Die Rangfolge, das Kernstueck der Suche."""

    def test_exaktes_kuerzel_steht_immer_oben(self) -> None:
        """Auch gegen einen weit hoeheren Score der Quelle."""
        hits = [
            make_hit("AAPU", "Direxion Daily AAPL Bull 2X Shares", score=9999.0),
            make_hit("AAPL", "Apple Inc.", score=1.0),
        ]
        assert [h.ticker for h in rank_hits(hits, "AAPL")] == ["AAPL", "AAPU"]

    def test_exaktes_kuerzel_schlaegt_exakten_namen(self) -> None:
        """Wer ein Kuerzel tippt, meint das Kuerzel."""
        hits = [
            make_hit("XYZ", "Now", score=500.0),
            make_hit("NOW", "ServiceNow, Inc.", score=1.0),
        ]
        assert rank_hits(hits, "now")[0].ticker == "NOW"

    def test_namensanfang_vor_teiltreffer(self) -> None:
        """``micro`` bringt den Anfang vor die Mitte."""
        hits = [
            make_hit("AMD", "Advanced Micro Devices, Inc.", score=900.0),
            make_hit("MSFT", "Microsoft Corporation", score=1.0),
        ]
        assert [h.ticker for h in rank_hits(hits, "micro")] == ["MSFT", "AMD"]

    def test_exakter_name_vor_namensanfang(self) -> None:
        """``apple`` trifft "Apple Inc." exakt, "Apple Hospitality" nur am Anfang."""
        hits = [
            make_hit("APLE", "Apple Hospitality REIT, Inc.", score=900.0),
            make_hit("AAPL", "Apple Inc.", score=1.0),
        ]
        assert [h.ticker for h in rank_hits(hits, "apple")] == ["AAPL", "APLE"]

    def test_score_entscheidet_innerhalb_derselben_stufe(self) -> None:
        """Bei gleicher Trefferguete gewinnt die Relevanz der Quelle."""
        hits = [
            make_hit("BNY", "The Bank of New York Mellon Corporation", score=10.0),
            make_hit("BAC", "Bank of America Corporation", score=900.0),
        ]
        assert rank_hits(hits, "bank of")[0].ticker == "BAC"

    def test_unpassende_treffer_fallen_heraus(self) -> None:
        """Eine Quelle liefert zu jeder Anfrage auch Rauschen."""
        hits = [make_hit("KO", "Coca-Cola Company"), make_hit("AAPL", "Apple Inc.")]
        assert [h.ticker for h in rank_hits(hits, "apple")] == ["AAPL"]

    def test_ranking_ist_unabhaengig_von_der_eingangsreihenfolge(self) -> None:
        """Dieselbe Menge ergibt dieselbe Rangfolge, egal wie sortiert."""
        hits = [
            make_hit("AAPL", "Apple Inc.", score=5.0),
            make_hit("APLE", "Apple Hospitality REIT, Inc.", score=3.0),
            make_hit("AAPU", "Direxion Daily AAPL Bull 2X Shares", score=1.0),
        ]
        vorwaerts = [h.ticker for h in rank_hits(hits, "apple")]
        rueckwaerts = [h.ticker for h in rank_hits(list(reversed(hits)), "apple")]
        assert vorwaerts == rueckwaerts

    @pytest.mark.parametrize("leer", ["", "   ", "***", "$$$ %%", "\t\n"])
    def test_leere_und_sinnlose_eingabe_ergibt_leere_liste(self, leer: str) -> None:
        """Kein Absturz, kein Fehler - waehrend des Tippens ist das normal."""
        hits = [make_hit("AAPL", "Apple Inc.")]
        assert rank_hits(hits, leer) == []

    def test_ohne_treffer_bleibt_die_liste_leer(self) -> None:
        """Eine Anfrage ohne Bezug liefert nichts, statt irgendetwas."""
        hits = [make_hit("AAPL", "Apple Inc.")]
        assert rank_hits(hits, "zzzqqq") == []

    def test_treffer_ohne_namen_stuerzt_nicht_ab(self) -> None:
        """Die Quelle liefert nicht zu jedem Symbol einen Namen."""
        hits = [make_hit("NOW", None)]
        assert [h.ticker for h in rank_hits(hits, "NOW")] == ["NOW"]


class TestFuzzy:
    """Tippfehler-Toleranz ueber ``difflib``."""

    def test_tippfehler_findet_microsoft(self) -> None:
        """``microsft`` ist der klassische Vertipper."""
        hits = [make_hit("MSFT", "Microsoft Corporation")]
        assert [h.ticker for h in rank_hits(hits, "microsft")] == ["MSFT"]

    @pytest.mark.parametrize("vertippt", ["microsft", "micrsoft", "mircosoft"])
    def test_mehrere_vertipper_finden_microsoft(self, vertippt: str) -> None:
        """Die Schwelle ist grosszuegig genug fuer gaengige Dreher."""
        hits = [make_hit("MSFT", "Microsoft Corporation")]
        assert [h.ticker for h in rank_hits(hits, vertippt)] == ["MSFT"]

    def test_fuzzy_steht_hinter_einem_echten_treffer(self) -> None:
        """Ein direkter Treffer schlaegt jede Aehnlichkeit."""
        hits = [
            make_hit("MSFT", "Microsoft Corporation"),
            make_hit("APPL", "Apple Inc."),
        ]
        geordnet = [h.ticker for h in rank_hits(hits, "apple")]
        assert geordnet[0] == "APPL"

    def test_unaehnliches_wird_nicht_fuzzy_gefunden(self) -> None:
        """Sonst waere jede Anfrage ein Treffer auf alles."""
        hits = [make_hit("KO", "Coca-Cola Company")]
        assert rank_hits(hits, "microsft") == []


class TestDeduplizierung:
    """Derselbe Emittent an mehreren Boersen."""

    def test_boersen_dubletten_werden_zusammengefasst(self) -> None:
        """ServiceNow liegt bei Yahoo an sechs Boersen - der Nutzer will eine Zeile."""
        hits = [
            make_hit("NOW", "ServiceNow, Inc.", exchange="NYSE", score=55769.0),
            make_hit("4S0.DE", "ServiceNow, Inc.", exchange="XETRA", score=20012.0),
            make_hit("SNOW.VI", "ServiceNow, Inc.", exchange="Vienna", score=20006.0),
        ]
        assert [h.ticker for h in deduplicate_hits(hits)] == ["NOW"]

    def test_stammaktie_schlaegt_zertifikat(self) -> None:
        """Ein CDR-Zertifikat ist nicht der gesuchte Titel, auch bei hoherem Score."""
        hits = [
            make_hit(
                "NOWS.TO",
                "ServiceNow, Inc.",
                exchange="Toronto",
                asset_class=AssetClass.ETF,
                score=99999.0,
            ),
            make_hit("NOW", "ServiceNow, Inc.", exchange="NYSE", score=1.0),
        ]
        assert deduplicate_hits(hits)[0].ticker == "NOW"

    def test_verschiedene_emittenten_bleiben_getrennt(self) -> None:
        """Gruppiert wird ueber den Namen, nicht ueber das Kuerzel."""
        hits = [make_hit("JNJ", "Johnson & Johnson"), make_hit("JCI", "Johnson Controls")]
        assert len(deduplicate_hits(hits)) == 2

    def test_treffer_ohne_namen_bilden_eigene_gruppen(self) -> None:
        """Ueber einen namenlosen Treffer laesst sich nichts aussagen."""
        hits = [make_hit("AAA", None), make_hit("BBB", None)]
        assert len(deduplicate_hits(hits)) == 2

    def test_reihenfolge_der_gruppen_bleibt_erhalten(self) -> None:
        """Eine vorher sortierte Liste darf hier nicht durcheinandergeraten."""
        hits = [
            make_hit("AAPL", "Apple Inc."),
            make_hit("MSFT", "Microsoft Corporation"),
            make_hit("AAPL.TO", "Apple Inc.", exchange="Toronto"),
        ]
        assert [h.ticker for h in deduplicate_hits(hits)] == ["AAPL", "MSFT"]


class TestPrepareResults:
    """Der gemeinsame Abschluss: ranken, zusammenfassen, kuerzen."""

    def test_erst_deduplizieren_dann_kuerzen(self) -> None:
        """Andersherum bliebe von sechs Listings genau ein Treffer uebrig."""
        hits = [
            make_hit("NOW", "ServiceNow, Inc.", exchange="NYSE", score=55769.0),
            make_hit("4S0.DE", "ServiceNow, Inc.", exchange="XETRA", score=20012.0),
            make_hit("NOWL", "GraniteShares 2x Long NOW Daily", score=10.0),
        ]
        ergebnis = prepare_results(hits, "now", limit=2)
        assert [h.ticker for h in ergebnis] == ["NOW", "NOWL"]

    def test_limit_kuerzt_die_liste(self) -> None:
        """Mehr als angefragt kommt nie zurueck."""
        hits = [make_hit(f"BANK{i}", f"Bank Nummer {i}", score=float(i)) for i in range(9)]
        assert len(prepare_results(hits, "bank", limit=3)) == 3

    @pytest.mark.parametrize("limit", [0, -1, -99])
    def test_limit_kleiner_eins_ergibt_leere_liste(self, limit: int) -> None:
        """Kein Absturz bei einem unsinnigen Limit."""
        hits = [make_hit("AAPL", "Apple Inc.")]
        assert prepare_results(hits, "apple", limit=limit) == []

    def test_deduplizierung_abschaltbar(self) -> None:
        """Wer alle Listings sehen will, bekommt sie."""
        hits = [
            make_hit("NOW", "ServiceNow, Inc.", exchange="NYSE", score=55769.0),
            make_hit("4S0.DE", "ServiceNow, Inc.", exchange="XETRA", score=20012.0),
        ]
        assert len(prepare_results(hits, "servicenow", limit=10, deduplicate=False)) == 2


# ------------------------------------------------------- Roh-Quote-Parser ---
class TestQuoteParser:
    """Die Abbildung der Yahoo-Rohantwort auf den Vertrag."""

    def test_exchdisp_gewinnt_vor_exchange(self) -> None:
        """Der Nutzer will ``NYSE:NOW`` sehen, nicht ``NYQ:NOW``."""
        hit = hit_from_quote({"symbol": "NOW", "exchange": "NYQ", "exchDisp": "NYSE"})
        assert hit is not None
        assert hit.exchange == "NYSE"

    def test_kuerzel_wird_aufgeloest_wenn_exchdisp_fehlt(self) -> None:
        """Ohne Anzeigenamen greift die Aufloesungstabelle."""
        hit = hit_from_quote({"symbol": "MSFT", "exchange": "NMS"})
        assert hit is not None
        assert hit.exchange == "NASDAQ"

    def test_longname_gewinnt_vor_shortname(self) -> None:
        """Der lange Name ist der vollstaendigere."""
        hit = hit_from_quote(
            {"symbol": "NOW", "shortname": "ServiceNow", "longname": "ServiceNow, Inc."}
        )
        assert hit is not None
        assert hit.name == "ServiceNow, Inc."

    def test_quotetype_wird_auf_assetclass_abgebildet(self) -> None:
        """Die Tabelle ist deckungsgleich mit der des Stammdaten-Abrufs."""
        hit = hit_from_quote({"symbol": "SPY", "quoteType": "ETF"})
        assert hit is not None
        assert hit.asset_class is AssetClass.ETF

    def test_unbekannter_quotetype_wird_other(self) -> None:
        """Yahoo liefert zu ``bank`` auch Futures."""
        hit = hit_from_quote({"symbol": "XBAC0=F", "quoteType": "FUTURE"})
        assert hit is not None
        assert hit.asset_class is AssetClass.OTHER

    @pytest.mark.parametrize("roh", [{}, {"symbol": ""}, {"symbol": "   "}, {"symbol": 42}])
    def test_eintrag_ohne_symbol_faellt_still_heraus(self, roh: dict[str, Any]) -> None:
        """Ein unbrauchbarer Eintrag darf den Abruf nicht scheitern lassen."""
        assert hit_from_quote(roh) is None

    def test_negativer_score_wird_verworfen(self) -> None:
        """Der Vertrag laesst nur ``score >= 0`` zu."""
        hit = hit_from_quote({"symbol": "NOW", "score": -5.0})
        assert hit is not None
        assert hit.score is None

    def test_nicht_dictionaries_werden_uebersprungen(self) -> None:
        """Robust gegen eine unerwartet geformte Antwort."""
        assert [h.ticker for h in hits_from_quotes(["kaputt", 42, {"symbol": "NOW"}])] == ["NOW"]


# ------------------------------------------------------- FixtureProvider ---
class TestFixtureProviderSuche:
    """Die Suche gegen die eingefrorenen Antworten."""

    def test_erfuellt_das_protocol(self, provider: FixtureProvider) -> None:
        """Strukturelle Kompatibilitaet, ohne zu erben - Architekturregel 2."""
        assert isinstance(provider, SymbolSearch)

    def test_servicenow_findet_now(self, provider: FixtureProvider) -> None:
        """Der Kernfall des Auftrags: Name rein, Kuerzel raus."""
        treffer = provider.search("servicenow")
        assert treffer[0].ticker == "NOW"

    def test_servicenow_zeigt_boerse_und_kuerzel(self, provider: FixtureProvider) -> None:
        """Der Nutzer soll ``NYSE:NOW`` lesen."""
        assert provider.search("servicenow")[0].display == "NYSE:NOW"

    def test_tippfehler_findet_microsoft(self, provider: FixtureProvider) -> None:
        """Yahoo liefert zu ``microsft`` nichts - das lokale Fuzzy-Ranking rettet den Fall."""
        treffer = provider.search("microsft")
        assert treffer[0].ticker == "MSFT"

    @pytest.mark.parametrize(
        ("anfrage", "erwartet"),
        [
            ("apple", "AAPL"),
            ("microsoft", "MSFT"),
            ("nvidia", "NVDA"),
            ("johnson", "JNJ"),
            ("bank", "BAC"),
            ("AAPL", "AAPL"),
            ("NOW", "NOW"),
        ],
    )
    def test_eingefrorene_anfragen_treffen(
        self, provider: FixtureProvider, anfrage: str, erwartet: str
    ) -> None:
        """Jede gezogene Suchanfrage liefert ihren erwarteten Spitzentreffer."""
        assert provider.search(anfrage)[0].ticker == erwartet

    def test_boersen_werden_einheitlich_angezeigt(self, provider: FixtureProvider) -> None:
        """Derselbe Titel darf nicht je nach Quelle anders aussehen."""
        aus_suche = provider.search("microsoft")[0]
        aus_stammdaten = provider.search("microsft")[0]
        assert aus_suche.exchange == aus_stammdaten.exchange == "NASDAQ"

    @pytest.mark.parametrize("leer", ["", "   ", "***", "$$$ %%"])
    def test_leere_eingabe_ergibt_leere_liste(self, provider: FixtureProvider, leer: str) -> None:
        """Kein Absturz, kein Netz, kein Fehler."""
        assert provider.search(leer) == []

    @pytest.mark.parametrize("limit", [0, -1])
    def test_limit_kleiner_eins_ergibt_leere_liste(
        self, provider: FixtureProvider, limit: int
    ) -> None:
        """Ein unsinniges Limit ist kein Fehler."""
        assert provider.search("apple", limit=limit) == []

    def test_limit_wird_eingehalten(self, provider: FixtureProvider) -> None:
        """Mehr als angefragt kommt nie zurueck."""
        assert len(provider.search("bank", limit=2)) <= 2

    def test_unbekannte_anfrage_ohne_fixture_liefert_leer(self, provider: FixtureProvider) -> None:
        """Ohne eingefrorene Antwort und ohne Namensaehnlichkeit gibt es nichts."""
        assert provider.search("zzzqqqxyz") == []

    def test_gross_und_kleinschreibung_egal(self, provider: FixtureProvider) -> None:
        """``ServiceNow`` und ``servicenow`` sind dieselbe Anfrage."""
        assert provider.search("ServiceNow")[0].ticker == provider.search("servicenow")[0].ticker

    def test_fehlendes_such_fixture_ist_kein_fehler(self, tmp_path: Path) -> None:
        """Ohne Fixture-Verzeichnis bleibt die Suche still leer."""
        leer = FixtureProvider(fixtures_dir=tmp_path)
        assert leer.search("servicenow") == []

    def test_defektes_such_fixture_ist_kein_fehler(self, tmp_path: Path) -> None:
        """Ein kaputtes JSON darf die Suche nicht sprengen."""
        (tmp_path / "search").mkdir()
        (tmp_path / "search" / "apple.json").write_text("{kaputt", encoding="utf-8")
        assert FixtureProvider(fixtures_dir=tmp_path).search("apple") == []

    def test_such_fixture_ohne_quotes_liste(self, tmp_path: Path) -> None:
        """Eine Antwort in unerwarteter Form wird uebersprungen."""
        (tmp_path / "search").mkdir()
        (tmp_path / "search" / "apple.json").write_text('{"quotes": "kaputt"}', encoding="utf-8")
        assert FixtureProvider(fixtures_dir=tmp_path).search("apple") == []


class TestSuchFixtures:
    """Der eingefrorene Bestand selbst."""

    @pytest.mark.parametrize(
        "slug",
        [
            "servicenow",
            "apple",
            "microsoft",
            "nvidia",
            "johnson",
            "bank",
            "now",
            "aapl",
            "microsft",
        ],
    )
    def test_jede_geforderte_anfrage_liegt_vor(self, fixtures_root: Path, slug: str) -> None:
        """Der Auftrag nennt diese neun Anfragen ausdruecklich."""
        assert (fixtures_root / "search" / f"{slug}.json").is_file()

    def test_fixtures_tragen_die_erwartete_form(self, fixtures_root: Path) -> None:
        """``query`` und ``quotes`` sind der Vertrag der Ablage."""
        payload = json.loads(
            (fixtures_root / "search" / "servicenow.json").read_text(encoding="utf-8")
        )
        assert payload["query"] == "servicenow"
        assert isinstance(payload["quotes"], list)
        assert payload["quotes"][0]["symbol"] == "NOW"

    def test_microsft_fixture_ist_bewusst_leer(self, fixtures_root: Path) -> None:
        """Yahoos Suche kennt keine Tippfehler - das ist eine Eigenschaft, kein Defekt."""
        payload = json.loads(
            (fixtures_root / "search" / "microsft.json").read_text(encoding="utf-8")
        )
        assert payload["quotes"] == []


# ------------------------------------------------------ Offline und Fehler ---
class TestOfflineGarantie:
    """Der Offline-Modus geht nicht ins Netz."""

    def test_yfinance_provider_ist_offline_gesperrt(self) -> None:
        """Der Konstruktor verweigert im Offline-Modus den Dienst."""
        from btlite.config import Settings

        settings = Settings()
        settings.data.offline = True

        from btlite.data.yfinance_provider import YFinanceProvider

        with pytest.raises(ProviderUnavailableError, match="Offline mode: yfinance is blocked"):
            YFinanceProvider(settings)

    def test_fixture_suche_oeffnet_keinen_socket(self, provider: FixtureProvider) -> None:
        """``conftest`` wuerde einen externen Zugriff laut abbrechen lassen."""
        assert provider.search("servicenow")[0].ticker == "NOW"


class TestFehlerMapping:
    """Rohe Provider-Fehler verlassen ``data/`` nie."""

    def _provider(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        """Baut einen YFinanceProvider mit abgeschaltetem Offline-Modus.

        Args:
            monkeypatch: Pytest-Patcher.

        Returns:
            Der Provider. Er geht in diesen Tests nie wirklich ins Netz,
            weil der Abruf selbst ersetzt wird.
        """
        from btlite.config import Settings
        from btlite.data.yfinance_provider import YFinanceProvider

        settings = Settings()
        settings.data.offline = False
        settings.ratelimit.max_retries = 0
        return YFinanceProvider(settings)

    def test_netzfehler_wird_provider_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ein beliebiger Transportfehler wird zum Projektfehler."""
        provider = self._provider(monkeypatch)

        def _boom(*args: object, **kwargs: object) -> None:
            msg = "connection reset by peer"
            raise OSError(msg)

        monkeypatch.setattr("yfinance.data.YfData.get_raw_json", _boom)

        with pytest.raises(ProviderUnavailableError):
            provider.search("servicenow")

    def test_rate_limit_wird_durchgereicht(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ein Rate Limit wird nie umgangen, sondern gemeldet."""
        from yfinance.exceptions import YFRateLimitError

        provider = self._provider(monkeypatch)

        def _throttled(*args: object, **kwargs: object) -> None:
            raise YFRateLimitError

        monkeypatch.setattr("yfinance.data.YfData.get_raw_json", _throttled)

        with pytest.raises(RateLimitError):
            provider.search("servicenow")

    def test_unerwartete_antwortform_wird_validierungsfehler(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Eine Antwort, die kein Objekt ist, passt nicht auf den Vertrag."""
        from btlite.core.errors import DataValidationError

        provider = self._provider(monkeypatch)
        monkeypatch.setattr("yfinance.data.YfData.get_raw_json", lambda *a, **k: ["unerwartet"])

        with pytest.raises(DataValidationError):
            provider.search("servicenow")

    def test_antwort_ohne_quotes_liefert_leer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Findet die Quelle nichts, ist das kein Fehler."""
        provider = self._provider(monkeypatch)
        monkeypatch.setattr("yfinance.data.YfData.get_raw_json", lambda *a, **k: {})

        assert provider.search("zzzqqq") == []

    def test_leere_anfrage_geht_gar_nicht_erst_raus(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ohne verwertbare Eingabe wird kein Request gestellt."""
        provider = self._provider(monkeypatch)

        def _darf_nicht_aufgerufen_werden(*args: object, **kwargs: object) -> None:
            pytest.fail("Die Suche hat trotz leerer Eingabe einen Request gestellt")

        monkeypatch.setattr("yfinance.data.YfData.get_raw_json", _darf_nicht_aufgerufen_werden)

        assert provider.search("   ") == []
        assert provider.search("apple", limit=0) == []
