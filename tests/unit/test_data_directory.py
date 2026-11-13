"""Tests des Titelverzeichnisses als Namensquelle der Suche.

Der Nutzer meldete "die suche ergibt nix". Die Suche kannte nur die neun
eingefrorenen Anfragen und die zehn Ticker mit vollen Stammdaten - "tesla"
oder "sap" fanden deshalb nichts, obwohl beide in den unterstuetzten
Indizes stehen.

``tests/fixtures/directory.json`` traegt Kuerzel, Firmenname und Branche
der Mitglieder von S&P 500, Dow Jones, Nasdaq 100 und DAX. Diese Tests
sichern, dass die Suche daraus speist und dass die Datei brauchbar bleibt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from btlite.config import Settings
from btlite.data.fixtures import FixtureProvider

if TYPE_CHECKING:
    from btlite.models.market import SearchHit

#: Indizes, die das Verzeichnis abdecken muss.
EXPECTED_INDICES = frozenset({"sp500", "djia", "nasdaq100", "dax"})


@pytest.fixture(scope="module")
def directory() -> dict[str, object]:
    """Liest die Verzeichnisdatei roh."""
    path = Settings().data.fixtures_dir / "directory.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def provider() -> FixtureProvider:
    return FixtureProvider()


class TestDirectoryFile:
    def test_file_exists(self):
        assert (Settings().data.fixtures_dir / "directory.json").is_file()

    def test_covers_every_supported_index(self, directory):
        entries = directory["entries"]
        found = {index for entry in entries for index in entry["indices"]}
        assert found >= EXPECTED_INDICES

    def test_holds_a_realistic_number_of_titles(self, directory):
        """Ein Bestand von wenigen Dutzend waere ein Abbruch beim Holen."""
        assert len(directory["entries"]) > 400

    def test_every_entry_has_a_ticker(self, directory):
        for entry in directory["entries"]:
            assert entry["ticker"].strip()

    def test_names_are_present(self, directory):
        """Ohne Namen findet die Namenssuche nichts - das war der Fehler."""
        named = [e for e in directory["entries"] if e.get("name", "").strip()]
        assert len(named) == len(directory["entries"])

    def test_german_tickers_keep_their_exchange_suffix(self, directory):
        """Yfinance braucht .DE - ohne Endung waere SAP.DE nicht abrufbar."""
        dax = [e for e in directory["entries"] if "dax" in e["indices"]]
        assert dax
        assert any(e["ticker"].endswith((".DE", ".PA")) for e in dax)


class TestSearchFindsIndexMembers:
    """Die Faelle, die der Nutzer gemeldet hat beziehungsweise die vorher leer waren."""

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("servicenow", "NOW"),
            ("tesla", "TSLA"),
            ("adobe", "ADBE"),
            ("netflix", "NFLX"),
            ("sap", "SAP.DE"),
            ("siemens", "SIE.DE"),
            ("allianz", "ALV.DE"),
        ],
    )
    def test_company_name_finds_its_ticker(self, provider, query, expected):
        hits = provider.search(query)
        assert expected in [hit.ticker for hit in hits], f"{query!r} findet {expected} nicht"

    def test_the_best_hit_is_the_named_company(self, provider):
        """Nicht irgendein Treffer - der gesuchte muss oben stehen."""
        assert provider.search("tesla")[0].ticker == "TSLA"
        assert provider.search("netflix")[0].ticker == "NFLX"

    def test_hits_carry_the_company_name(self, provider):
        hit = provider.search("tesla")[0]
        assert hit.name
        assert "Tesla" in hit.name

    def test_typo_still_works(self, provider):
        """Das Fuzzy-Ranking darf durch die neue Quelle nicht verwaessern."""
        assert "MSFT" in [hit.ticker for hit in provider.search("microsft")]

    def test_nonsense_finds_nothing(self, provider):
        """558 Eintraege duerfen nicht dazu fuehren, dass alles matcht."""
        assert provider.search("XYZQFOO") == []

    def test_empty_query_is_not_an_error(self, provider):
        assert provider.search("") == []


class TestDirectoryIsNotAPriceSource:
    """Ein Treffer heisst "dieses Papier gibt es", nicht "es gibt Kurse"."""

    def test_directory_ticker_is_searchable(self, provider):
        """Der Titel ist auffindbar - ob Kurse vorliegen, sagt das nicht."""
        assert "TSLA" in [hit.ticker for hit in provider.search("tesla")]

    def test_missing_prices_fail_with_a_clean_message(self, provider):
        """Ein Verzeichnistreffer ohne Kursfixture darf nicht roh abstuerzen.

        Offline liegen Kurse nur zu den zehn Fixture-Tickern vor. Die
        Suche findet aber 558 Titel - der Unterschied muss als
        verstaendliche Meldung ankommen, nicht als Traceback.
        """
        from btlite.core.errors import SymbolNotFoundError

        with pytest.raises(SymbolNotFoundError, match="TSLA"):
            provider.get_history("TSLA")


class TestRobustness:
    def test_missing_file_is_not_fatal(self, tmp_path: Path, monkeypatch):
        """Ohne Verzeichnis bedienen die anderen Quellen die Suche allein."""
        settings = Settings()
        settings.data.fixtures_dir = tmp_path
        provider = FixtureProvider(settings=settings)
        assert provider._load_directory() == []

    def test_broken_file_is_not_fatal(self, tmp_path: Path):
        (tmp_path / "directory.json").write_text("{kein json", encoding="utf-8")
        settings = Settings()
        settings.data.fixtures_dir = tmp_path
        provider = FixtureProvider(settings=settings)
        assert provider._load_directory() == []

    def test_entry_without_ticker_is_skipped(self, tmp_path: Path):
        payload = {"entries": [{"name": "Ohne Kuerzel"}, {"ticker": "OK", "name": "Gut"}]}
        (tmp_path / "directory.json").write_text(json.dumps(payload), encoding="utf-8")
        settings = Settings()
        settings.data.fixtures_dir = tmp_path
        provider = FixtureProvider(settings=settings)
        hits: list[SearchHit] = provider._load_directory()
        assert [hit.ticker for hit in hits] == ["OK"]

    def test_result_is_cached(self, provider):
        """Jede Tastatureingabe darf nicht 558 Eintraege neu parsen."""
        first = provider._directory_hits()
        second = provider._directory_hits()
        assert len(first) == len(second)
        assert provider._directory_hit_cache is not None
