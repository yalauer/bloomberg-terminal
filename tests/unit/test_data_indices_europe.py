"""Tests der Symbolaufloesung fuer europaeische Indizes.

Kein Test hier geht ins Netz - die Socket-Sperre der Suite verbietet es.
Die Quellen sind Attrappen; ihre Antwortformen sind gegen die echten
Quellen verifiziert und hier eingefroren:

* Wikipedia STOXX Europe 600: Spalten ``Ticker``, ``Company``,
  ``ICB Sector``, ``Country``, ``Headquarters``. Die Ticker tragen
  **keine** Boersen-Endung - genau eine von 467 Zeilen hat einen Punkt.
* Die Symbolsuche liefert :class:`~btlite.models.market.SearchHit` mit
  Symbol, Name und Instrumentenklasse, wie
  ``YFinanceProvider.search`` es tut.

Die geprueften Faelle stammen aus einem echten Abruf, nicht aus der
Vorstellung: ``DG`` als Dollar General statt Vinci, ``CTS.DE`` als AVIC
International statt CTS Eventim, ``Ferrari`` mit zwei namensgleichen
Treffern.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from btlite.config import Settings
from btlite.core.errors import RateLimitError
from btlite.data import indices
from btlite.data.indices import (
    INDEX_REGISTRY,
    IndexConstituentsProvider,
    candidate_symbol,
    confirms_candidate,
    resolve_from_hits,
    suffix_for_country,
)
from btlite.models.common import AssetClass
from btlite.models.market import SearchHit
from btlite.models.screening import IndexSource

# --------------------------------------------------------------- Helfer --


def _online_settings(tmp_path: Any) -> Settings:
    """Settings mit abgeschaltetem Offline-Modus und temporaerem Cache."""
    settings = Settings()
    settings.data.offline = False
    settings.cache.dir = tmp_path / "cache"
    return settings


def _hit(ticker: str, name: str, asset_class: AssetClass = AssetClass.EQUITY) -> SearchHit:
    """Baut einen Suchtreffer, wie die Quelle ihn liefert."""
    return SearchHit(ticker=ticker, name=name, asset_class=asset_class)


def _stoxx_frame(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """Antwortform der STOXX-600-Tabelle auf Wikipedia.

    Args:
        rows: Tupel aus Ticker, Company, ICB Sector und Country.
    """
    return pd.DataFrame(
        {
            "Ticker": [r[0] for r in rows],
            "Company": [r[1] for r in rows],
            "ICB Sector": [r[2] for r in rows],
            "Country": [r[3] for r in rows],
            "Headquarters": ["-"] * len(rows),
        }
    )


class _FakeSearch:
    """Attrappe eines :class:`~btlite.data.base.SymbolSearch`.

    Antwortet aus einer Tabelle Anfrage -> Treffer und zaehlt die Aufrufe.
    Eine unbekannte Anfrage liefert eine leere Liste, so wie die echte
    Quelle es tut, wenn sie nichts kennt.
    """

    def __init__(
        self, responses: dict[str, list[SearchHit]], *, raises: Exception | None = None
    ) -> None:
        self._responses = responses
        self._raises = raises
        self.queries: list[str] = []

    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        self.queries.append(query)
        if self._raises is not None:
            raise self._raises
        return list(self._responses.get(query, []))


class _CountingReadHtml:
    """Attrappe von ``pandas.read_html``, zaehlt die Aufrufe."""

    def __init__(self, tables: list[pd.DataFrame]) -> None:
        self._tables = tables
        self.calls = 0

    def __call__(self, url: str, *, storage_options: dict[str, str] | None = None) -> list[Any]:
        self.calls += 1
        return list(self._tables)


# -------------------------------------------------------------- Registry --


class TestRegistryEntry:
    def test_stoxx600_is_registered(self):
        assert "stoxx600" in INDEX_REGISTRY

    def test_label_is_the_index_name(self):
        assert INDEX_REGISTRY["stoxx600"].label == "STOXX Europe 600"

    def test_source_is_wikipedia_because_finviz_has_no_european_indices(self):
        assert INDEX_REGISTRY["stoxx600"].source is IndexSource.WIKIPEDIA

    def test_definition_names_the_country_column(self):
        # Gesetzte Country-Spalte ist das Signal: die Ticker dieser Quelle
        # brauchen eine Aufloesung, bevor yfinance sie kennt.
        assert INDEX_REGISTRY["stoxx600"].wikipedia_country_column == "Country"

    def test_us_indices_keep_no_country_column(self):
        # Finviz und die DAX-Tabelle liefern fertige Symbole. Wer dort eine
        # Aufloesung anwirft, zahlt 500 Suchanfragen ohne Gegenwert.
        for key in ("sp500", "djia", "nasdaq100", "russell2000", "dax"):
            assert INDEX_REGISTRY[key].wikipedia_country_column is None


# ---------------------------------------------------------------- Stufe 1 --


class TestCountrySuffixes:
    @pytest.mark.parametrize(
        ("country", "suffix"),
        [
            ("United Kingdom", ".L"),
            ("Germany", ".DE"),
            ("France", ".PA"),
            ("Switzerland", ".SW"),
            ("Sweden", ".ST"),
            ("Netherlands", ".AS"),
            ("Spain", ".MC"),
            ("Italy", ".MI"),
            ("Belgium", ".BR"),
            ("Finland", ".HE"),
            ("Norway", ".OL"),
            ("Denmark", ".CO"),
            ("Austria", ".VI"),
            ("Ireland", ".IR"),
            ("Portugal", ".LS"),
            ("Luxembourg", ".LU"),
            ("Poland", ".WA"),
            ("Greece", ".AT"),
        ],
    )
    def test_known_countries_map_to_their_home_exchange(self, country, suffix):
        assert suffix_for_country(country) == suffix

    def test_countries_without_an_index_exchange_have_no_suffix(self):
        # Bermuda (Hiscox) und Israel (Teva) fuehren je einen Titel, der
        # ausserhalb seines Sitzlands notiert. Stufe 2 klaert sie.
        assert suffix_for_country("Bermuda") is None
        assert suffix_for_country("Israel") is None

    def test_missing_country_has_no_suffix(self):
        assert suffix_for_country(None) is None

    def test_suffix_is_appended_to_the_bare_ticker(self):
        assert candidate_symbol("SAF", "France") == "SAF.PA"
        assert candidate_symbol("ZURN", "Switzerland") == "ZURN.SW"

    def test_the_dangerous_case_gets_its_exchange(self):
        # Kern der Aufgabe: DG allein ist Dollar General, ein
        # US-Discounter. Gemeint ist Vinci.
        assert candidate_symbol("DG", "France") == "DG.PA"

    def test_share_classes_use_yahoos_hyphen(self):
        # Die Quelle schreibt "AMBU B", Yahoo "AMBU-B.CO".
        assert candidate_symbol("AMBU B", "Denmark") == "AMBU-B.CO"
        assert candidate_symbol("NOVO B", "Denmark") == "NOVO-B.CO"

    def test_an_existing_suffix_is_left_alone(self):
        # BT.A ist die einzige Zeile der Quelle mit einem Punkt. Ein
        # zweites Suffix wuerde sie kaputtmachen.
        assert candidate_symbol("BT.A", "United Kingdom") == "BT.A"

    def test_unknown_country_leaves_the_ticker_bare(self):
        assert candidate_symbol("TEV", "Israel") == "TEV"


# --------------------------------------------------- Stufe 1, Bestaetigung --


class TestCandidateConfirmation:
    def test_a_symbol_the_source_knows_is_confirmed(self):
        hits = [_hit("DG.PA", "Vinci SA")]
        assert confirms_candidate(hits, candidate="DG.PA") is True

    def test_an_unknown_symbol_is_not_confirmed(self):
        # Die Suche nach CTS.DE geht leer aus - der Titel wandert damit in
        # Stufe 2 und wird dort richtig auf EVD.DE aufgeloest. Ein
        # Kursabruf haette hier die AVIC International Holdings geliefert.
        assert confirms_candidate([], candidate="CTS.DE") is False

    def test_a_hit_for_another_symbol_does_not_confirm(self):
        hits = [_hit("DGRO", "iShares Core Dividend Growth ETF")]
        assert confirms_candidate(hits, candidate="DG.PA") is False

    @pytest.mark.parametrize(
        ("candidate", "source_name", "yahoo_name"),
        [
            ("BMW.DE", "BMW", "Bayerische Motoren Werke Aktiengesellschaft"),
            ("ITX.MC", "Inditex", "Industria de Diseno Textil, S.A."),
            ("MUV2.DE", "Munich Re", "Muenchener Rueckversicherungs-Gesellschaft AG"),
            ("DHL.DE", "DHL Group", "Deutsche Post AG"),
            ("SBRY.L", "Sainsbury's", "J Sainsbury plc"),
            ("AKZA.AS", "AkzoNobel", "Akzo Nobel N.V."),
        ],
    )
    def test_a_differing_company_name_does_not_reject_the_symbol(
        self, candidate, source_name, yahoo_name
    ):
        # Beide Quellen nennen dieselbe Firma verschieden. Wer hier den
        # Namen vergleicht, verwirft ein halbes Dutzend Schwergewichte des
        # Index, ohne einen einzigen Falschtreffer zu verhindern: das
        # Symbol traegt bereits die Boerse des Sitzlandes.
        hits = [_hit(candidate, yahoo_name)]
        assert confirms_candidate(hits, candidate=candidate) is True


# ---------------------------------------------------------------- Stufe 2 --


class TestResolveFromHits:
    def test_the_home_exchange_wins_over_the_first_hit(self):
        # Die Suche nach Thales liefert Muenchen zuerst. Richtig ist Paris:
        # die Country-Spalte steuert die Auswahl, nicht die Reihenfolge.
        hits = [_hit("CSF.MU", "Thales S.A."), _hit("HO.PA", "Thales S.A.")]
        ticker, _ = resolve_from_hits(hits, source_name="Thales", country="France")
        assert ticker == "HO.PA"

    def test_ferrari_is_not_confused_with_the_amsterdam_listing(self):
        # Ferrari steht als Italien im Index, die Suche liefert Amsterdam
        # zuerst - und zwar eine voellig andere Firma.
        hits = [_hit("FERGR.AS", "Ferrari Group PLC"), _hit("RACE", "Ferrari N.V.")]
        ticker, reason = resolve_from_hits(hits, source_name="Ferrari", country="Italy")
        assert ticker is None
        assert "eindeutig" in reason

    def test_ambiguity_is_reported_with_the_candidates(self):
        hits = [_hit("FERGR.AS", "Ferrari Group PLC"), _hit("RACE", "Ferrari N.V.")]
        _, reason = resolve_from_hits(hits, source_name="Ferrari", country="Italy")
        assert "FERGR.AS" in reason
        assert "RACE" in reason

    def test_listing_beats_domicile_when_they_disagree(self):
        # InPost steht als Luxemburg im Index, notiert aber in Amsterdam.
        # Die tatsaechliche Notierung ist richtig.
        hits = [_hit("669.HA", "Inpost SA"), _hit("INPST.AS", "InPost S.A.")]
        ticker, reason = resolve_from_hits(hits, source_name="InPost", country="Luxembourg")
        assert ticker == "INPST.AS"
        assert "ausserhalb des Sitzlandes" in reason

    def test_a_us_listing_is_accepted_for_a_country_without_an_exchange(self):
        # Teva steht als Israel im Index und notiert an der NYSE.
        hits = [_hit("TEVA", "Teva Pharmaceutical Industries Limited")]
        ticker, _ = resolve_from_hits(
            hits, source_name="Teva Pharmaceutical Industries", country="Israel"
        )
        assert ticker == "TEVA"

    def test_a_looser_name_passes_at_the_home_exchange(self):
        # Die Quelle schreibt "Air Liquide", Yahoo "L'Air Liquide S.A.".
        # Das passende Laender-Suffix traegt die lockere Namenspruefung.
        hits = [_hit("AIL.MU", "Air Liquide SA"), _hit("AI.PA", "L'Air Liquide S.A.")]
        ticker, _ = resolve_from_hits(hits, source_name="Air Liquide", country="France")
        assert ticker == "AI.PA"

    def test_regional_second_listings_are_rejected(self):
        # AIL.MU ist Muenchen: real, aber marginal gehandelt. Ohne einen
        # Treffer an der Heimatboerse faellt der Titel lieber heraus.
        hits = [_hit("AIL.MU", "Air Liquide SA")]
        ticker, _ = resolve_from_hits(hits, source_name="Air Liquide", country="France")
        assert ticker is None

    def test_an_index_is_not_a_company(self):
        # Die Suche nach "Euronext" liefert den Euronext-100-Index.
        hits = [_hit("^N100", "Euronext 100 Index", AssetClass.INDEX)]
        ticker, _ = resolve_from_hits(hits, source_name="Euronext", country="Netherlands")
        assert ticker is None

    def test_a_fund_is_not_a_company(self):
        hits = [_hit("0P00000CW4.F", "Baloise ESG International DWS LD", AssetClass.FUND)]
        ticker, _ = resolve_from_hits(hits, source_name="Baloise", country="Switzerland")
        assert ticker is None

    def test_a_similar_name_at_a_foreign_exchange_is_refused(self):
        # BT Group ist britisch; BTS Group AB ist schwedisch. Der
        # Namensvergleich an einer fremden Boerse muss streng sein.
        hits = [_hit("BTS-B.ST", "BTS Group AB (publ)")]
        ticker, _ = resolve_from_hits(hits, source_name="BT Group", country="United Kingdom")
        assert ticker is None

    def test_a_prefix_collision_is_refused(self):
        # LifeCo ist irisch, Lifecore Biomedical amerikanisch.
        hits = [_hit("LFCR", "Lifecore Biomedical, Inc.")]
        ticker, _ = resolve_from_hits(hits, source_name="LifeCo", country="Ireland")
        assert ticker is None

    def test_no_hits_at_all_is_reported_as_such(self):
        ticker, reason = resolve_from_hits([], source_name="Royal Mail", country="United Kingdom")
        assert ticker is None
        assert "kennt keinen passenden Titel" in reason


# ------------------------------------------------------- Abruf, verdrahtet --


class TestStoxxFetch:
    def test_bare_tickers_get_their_exchange_suffix(self, monkeypatch, tmp_path):
        frame = _stoxx_frame(
            [
                ("ZURN", "Zurich Insurance Group", "Insurance", "Switzerland"),
                ("SAF", "Safran", "Aerospace", "France"),
                ("GSK", "GSK", "Pharmaceuticals", "United Kingdom"),
            ]
        )
        monkeypatch.setattr(indices.pd, "read_html", _CountingReadHtml([frame]))
        search = _FakeSearch(
            {
                "ZURN.SW": [_hit("ZURN.SW", "Zurich Insurance Group AG")],
                "SAF.PA": [_hit("SAF.PA", "Safran SA")],
                "GSK.L": [_hit("GSK.L", "GSK plc")],
            }
        )

        provider = IndexConstituentsProvider(_online_settings(tmp_path), symbol_search=search)
        result = provider.get_constituents("stoxx600")

        assert result.tickers == ["ZURN.SW", "SAF.PA", "GSK.L"]
        assert result.skipped == []

    def test_vinci_does_not_become_dollar_general(self, monkeypatch, tmp_path):
        # Der gefaehrlichste Fall des ganzen Auftrags. Rohes DG ist ein
        # US-Discounter; das Universum muss DG.PA fuehren oder gar nichts.
        frame = _stoxx_frame([("DG", "Vinci", "Construction", "France")])
        monkeypatch.setattr(indices.pd, "read_html", _CountingReadHtml([frame]))
        search = _FakeSearch(
            {
                "DG.PA": [_hit("DG.PA", "Vinci SA")],
                "DG": [_hit("DG", "Dollar General Corporation")],
            }
        )

        provider = IndexConstituentsProvider(_online_settings(tmp_path), symbol_search=search)
        result = provider.get_constituents("stoxx600")

        assert result.tickers == ["DG.PA"]
        assert "DG" not in result.tickers
        assert result.members[0].name == "Vinci"

    def test_a_derived_suffix_the_source_does_not_know_falls_through_to_search(
        self, monkeypatch, tmp_path
    ):
        # CTS.DE ist die naheliegende Ableitung aus "CTS" plus Deutschland,
        # und sie ist falsch: die Symbolsuche kennt das Symbol nicht (ein
        # Kursabruf haette dagegen die AVIC International Holdings
        # geliefert). Die Namenssuche korrigiert auf EVD.DE.
        frame = _stoxx_frame([("CTS", "CTS Eventim", "Media", "Germany")])
        monkeypatch.setattr(indices.pd, "read_html", _CountingReadHtml([frame]))
        search = _FakeSearch({"CTS Eventim": [_hit("EVD.DE", "CTS Eventim AG & Co. KGaA")]})

        provider = IndexConstituentsProvider(_online_settings(tmp_path), symbol_search=search)
        result = provider.get_constituents("stoxx600")

        assert result.tickers == ["EVD.DE"]

    def test_a_heavyweight_with_a_different_official_name_stays_in(self, monkeypatch, tmp_path):
        # Regression: eine Namenspruefung in Stufe 1 warf BMW, Inditex und
        # Munich Re aus dem Universum, weil Yahoo sie anders nennt als
        # Wikipedia. Der Index besteht zum guten Teil aus solchen Titeln.
        frame = _stoxx_frame(
            [
                ("BMW", "BMW", "Automobiles", "Germany"),
                ("ITX", "Inditex", "Retail", "Spain"),
                ("MUV2", "Munich Re", "Insurance", "Germany"),
            ]
        )
        monkeypatch.setattr(indices.pd, "read_html", _CountingReadHtml([frame]))
        search = _FakeSearch(
            {
                "BMW.DE": [_hit("BMW.DE", "Bayerische Motoren Werke Aktiengesellschaft")],
                "ITX.MC": [_hit("ITX.MC", "Industria de Diseno Textil, S.A.")],
                "MUV2.DE": [_hit("MUV2.DE", "Muenchener Rueckversicherungs-Gesellschaft AG")],
            }
        )

        provider = IndexConstituentsProvider(_online_settings(tmp_path), symbol_search=search)
        result = provider.get_constituents("stoxx600")

        assert result.tickers == ["BMW.DE", "ITX.MC", "MUV2.DE"]
        assert result.skipped == []

    def test_an_unresolvable_title_is_skipped_not_guessed(self, monkeypatch, tmp_path):
        frame = _stoxx_frame([("RMG", "Royal Mail", "Logistics", "United Kingdom")])
        monkeypatch.setattr(indices.pd, "read_html", _CountingReadHtml([frame]))
        search = _FakeSearch({})

        provider = IndexConstituentsProvider(_online_settings(tmp_path), symbol_search=search)
        with pytest.raises(indices.DataValidationError):
            # Nur ein Titel, und der faellt heraus: ein leerer Bestand ist
            # ein Fehler, keine stille leere Liste.
            provider.get_constituents("stoxx600")

    def test_skipped_titles_are_counted_and_visible(self, monkeypatch, tmp_path):
        frame = _stoxx_frame(
            [
                ("SAF", "Safran", "Aerospace", "France"),
                ("RMG", "Royal Mail", "Logistics", "United Kingdom"),
                ("RACE", "Ferrari", "Automobiles", "Italy"),
            ]
        )
        monkeypatch.setattr(indices.pd, "read_html", _CountingReadHtml([frame]))
        search = _FakeSearch(
            {
                "SAF.PA": [_hit("SAF.PA", "Safran SA")],
                "Ferrari": [
                    _hit("FERGR.AS", "Ferrari Group PLC"),
                    _hit("RACE", "Ferrari N.V."),
                ],
            }
        )

        provider = IndexConstituentsProvider(_online_settings(tmp_path), symbol_search=search)
        result = provider.get_constituents("stoxx600")

        assert result.tickers == ["SAF.PA"]
        assert len(result.skipped) == 2
        assert {s.source_ticker for s in result.skipped} == {"RMG", "RACE"}
        assert result.source_count == 3
        assert result.is_complete is False

    def test_every_skip_carries_a_readable_reason(self, monkeypatch, tmp_path):
        frame = _stoxx_frame(
            [
                ("SAF", "Safran", "Aerospace", "France"),
                ("RMG", "Royal Mail", "Logistics", "United Kingdom"),
            ]
        )
        monkeypatch.setattr(indices.pd, "read_html", _CountingReadHtml([frame]))
        search = _FakeSearch({"SAF.PA": [_hit("SAF.PA", "Safran SA")]})

        provider = IndexConstituentsProvider(_online_settings(tmp_path), symbol_search=search)
        result = provider.get_constituents("stoxx600")

        assert result.skipped[0].reason
        assert result.skipped[0].name == "Royal Mail"
        assert result.skipped[0].country == "United Kingdom"

    def test_a_complete_fetch_reports_itself_as_complete(self, monkeypatch, tmp_path):
        frame = _stoxx_frame([("SAF", "Safran", "Aerospace", "France")])
        monkeypatch.setattr(indices.pd, "read_html", _CountingReadHtml([frame]))
        search = _FakeSearch({"SAF.PA": [_hit("SAF.PA", "Safran SA")]})

        provider = IndexConstituentsProvider(_online_settings(tmp_path), symbol_search=search)
        result = provider.get_constituents("stoxx600")

        assert result.is_complete is True
        assert result.source_count == len(result.members)

    def test_sector_and_name_survive_the_resolution(self, monkeypatch, tmp_path):
        frame = _stoxx_frame([("SAF", "Safran", "Aerospace and Defence", "France")])
        monkeypatch.setattr(indices.pd, "read_html", _CountingReadHtml([frame]))
        search = _FakeSearch({"SAF.PA": [_hit("SAF.PA", "Safran SA")]})

        provider = IndexConstituentsProvider(_online_settings(tmp_path), symbol_search=search)
        member = provider.get_constituents("stoxx600").members[0]

        assert member.name == "Safran"
        assert member.sector == "Aerospace and Defence"

    def test_the_table_is_chosen_by_its_columns(self, monkeypatch, tmp_path):
        # Die Seite fuehrt 15 Tabellen; ein fester Index waere beim
        # naechsten Umbau falsch. read_html(match=...) braucht html5lib.
        decoys = [
            pd.DataFrame({"Year": [2024], "No. of points": [500]}),
            pd.DataFrame({"Date": ["2024"], "Added": ["X"], "Removed": ["Y"]}),
        ]
        frame = _stoxx_frame([("SAF", "Safran", "Aerospace", "France")])
        monkeypatch.setattr(indices.pd, "read_html", _CountingReadHtml([*decoys, frame]))
        search = _FakeSearch({"SAF.PA": [_hit("SAF.PA", "Safran SA")]})

        provider = IndexConstituentsProvider(_online_settings(tmp_path), symbol_search=search)
        assert provider.get_tickers("stoxx600") == ["SAF.PA"]

    def test_duplicates_after_resolution_collapse(self, monkeypatch, tmp_path):
        # Helvetia und Baloise sind fusioniert; beide Namen loesen auf
        # dasselbe Symbol auf. Doppelt darf es nicht im Universum stehen.
        frame = _stoxx_frame(
            [
                ("HELN", "Helvetia", "Insurance", "Switzerland"),
                ("BALN", "Baloise", "Insurance", "Switzerland"),
            ]
        )
        monkeypatch.setattr(indices.pd, "read_html", _CountingReadHtml([frame]))
        search = _FakeSearch(
            {
                "Helvetia": [_hit("HBAN.SW", "Helvetia Baloise Holding AG")],
                "Baloise": [_hit("HBAN.SW", "Helvetia Baloise Holding AG")],
            }
        )

        provider = IndexConstituentsProvider(_online_settings(tmp_path), symbol_search=search)
        assert provider.get_tickers("stoxx600") == ["HBAN.SW"]


# ------------------------------------------------------- ohne Symbolsucher --


class TestWithoutSymbolSearch:
    def test_stage_one_alone_still_produces_a_universe(self, monkeypatch, tmp_path):
        # Anforderung: ohne injizierten Sucher muss Stufe 1 allein
        # funktionieren.
        frame = _stoxx_frame(
            [
                ("SAF", "Safran", "Aerospace", "France"),
                ("ZURN", "Zurich Insurance Group", "Insurance", "Switzerland"),
            ]
        )
        monkeypatch.setattr(indices.pd, "read_html", _CountingReadHtml([frame]))

        provider = IndexConstituentsProvider(_online_settings(tmp_path))
        assert provider.get_tickers("stoxx600") == ["SAF.PA", "ZURN.SW"]

    def test_a_country_without_a_suffix_is_skipped_rather_than_guessed(self, monkeypatch, tmp_path):
        # Ohne Sucher und ohne Laender-Endung bleibt nur das rohe Symbol -
        # und das traefe womoeglich eine andere Firma. Also heraus damit.
        frame = _stoxx_frame(
            [
                ("SAF", "Safran", "Aerospace", "France"),
                ("TEV", "Teva Pharmaceutical Industries", "Pharma", "Israel"),
            ]
        )
        monkeypatch.setattr(indices.pd, "read_html", _CountingReadHtml([frame]))

        provider = IndexConstituentsProvider(_online_settings(tmp_path))
        result = provider.get_constituents("stoxx600")

        assert result.tickers == ["SAF.PA"]
        assert [s.source_ticker for s in result.skipped] == ["TEV"]

    def test_no_search_requests_are_made(self, monkeypatch, tmp_path):
        frame = _stoxx_frame([("SAF", "Safran", "Aerospace", "France")])
        monkeypatch.setattr(indices.pd, "read_html", _CountingReadHtml([frame]))
        search = _FakeSearch({})

        provider = IndexConstituentsProvider(_online_settings(tmp_path))
        provider.get_constituents("stoxx600")
        assert search.queries == []


# ----------------------------------------------------------------- Cache --


class TestResolutionCaching:
    def test_the_expensive_resolution_runs_only_once(self, monkeypatch, tmp_path):
        # Kern der Cache-Anforderung: die Namensaufloesung kostet einen
        # Request je offenem Titel und darf nicht bei jedem Aufruf laufen.
        frame = _stoxx_frame([("CTS", "CTS Eventim", "Media", "Germany")])
        fake = _CountingReadHtml([frame])
        monkeypatch.setattr(indices.pd, "read_html", fake)
        search = _FakeSearch({"CTS Eventim": [_hit("EVD.DE", "CTS Eventim AG & Co. KGaA")]})
        settings = _online_settings(tmp_path)

        first = IndexConstituentsProvider(settings, symbol_search=search).get_constituents(
            "stoxx600"
        )
        queries_after_first = len(search.queries)
        second = IndexConstituentsProvider(settings, symbol_search=search).get_constituents(
            "stoxx600"
        )

        assert fake.calls == 1
        assert len(search.queries) == queries_after_first, "Die Aufloesung lief erneut"
        assert first.tickers == second.tickers
        assert second.provenance.from_cache is True

    def test_skipped_titles_survive_the_cache(self, monkeypatch, tmp_path):
        # Sonst zeigte der erste Abruf "1 von 2" und der zweite "1" - die
        # Luecke waere nach einem Neustart unsichtbar.
        frame = _stoxx_frame(
            [
                ("SAF", "Safran", "Aerospace", "France"),
                ("RMG", "Royal Mail", "Logistics", "United Kingdom"),
            ]
        )
        monkeypatch.setattr(indices.pd, "read_html", _CountingReadHtml([frame]))
        search = _FakeSearch({"SAF.PA": [_hit("SAF.PA", "Safran SA")]})
        settings = _online_settings(tmp_path)

        IndexConstituentsProvider(settings, symbol_search=search).get_constituents("stoxx600")
        cached = IndexConstituentsProvider(settings, symbol_search=search).get_constituents(
            "stoxx600"
        )

        assert cached.tickers == ["SAF.PA"]
        assert len(cached.skipped) == 1
        assert cached.skipped[0].source_ticker == "RMG"
        assert cached.skipped[0].name == "Royal Mail"
        assert cached.skipped[0].country == "United Kingdom"
        assert cached.skipped[0].reason
        assert cached.is_complete is False

    def test_a_complete_index_caches_without_skips(self, monkeypatch, tmp_path):
        frame = _stoxx_frame([("SAF", "Safran", "Aerospace", "France")])
        monkeypatch.setattr(indices.pd, "read_html", _CountingReadHtml([frame]))
        search = _FakeSearch({"SAF.PA": [_hit("SAF.PA", "Safran SA")]})
        settings = _online_settings(tmp_path)

        IndexConstituentsProvider(settings, symbol_search=search).get_constituents("stoxx600")
        cached = IndexConstituentsProvider(settings, symbol_search=search).get_constituents(
            "stoxx600"
        )

        assert cached.skipped == []
        assert cached.is_complete is True

    def test_the_dax_cache_still_round_trips(self, monkeypatch, tmp_path):
        # Regression: die Cache-Tabelle hat eine Spalte mehr bekommen. Die
        # Indizes ohne Aufloesung duerfen davon nichts merken.
        frame = pd.DataFrame(
            {
                "Ticker": ["ADS.DE", "ALV.DE"],
                "Company": ["Adidas", "Allianz"],
                "Prime Standard Sector": ["Retail", "Insurance"],
            }
        )
        monkeypatch.setattr(indices.pd, "read_html", _CountingReadHtml([frame]))
        settings = _online_settings(tmp_path)

        IndexConstituentsProvider(settings).get_constituents("dax")
        cached = IndexConstituentsProvider(settings).get_constituents("dax")

        assert cached.tickers == ["ADS.DE", "ALV.DE"]
        assert cached.members[0].name == "Adidas"
        assert cached.members[0].sector == "Retail"
        assert cached.skipped == []


# ------------------------------------------------------------ Rate Limits --


class TestRateLimits:
    def test_every_search_takes_a_token(self, monkeypatch, tmp_path):
        frame = _stoxx_frame(
            [
                ("SAF", "Safran", "Aerospace", "France"),
                ("ZURN", "Zurich Insurance Group", "Insurance", "Switzerland"),
            ]
        )
        monkeypatch.setattr(indices.pd, "read_html", _CountingReadHtml([frame]))
        search = _FakeSearch(
            {
                "SAF.PA": [_hit("SAF.PA", "Safran SA")],
                "ZURN.SW": [_hit("ZURN.SW", "Zurich Insurance Group AG")],
            }
        )

        class _CountingBucket:
            def __init__(self) -> None:
                self.acquired = 0

            def acquire(self, tokens: float = 1.0, *, timeout_s: float | None = None) -> float:
                self.acquired += 1
                return 0.0

            def try_acquire(self, tokens: float = 1.0) -> bool:
                self.acquired += 1
                return True

        bucket = _CountingBucket()
        provider = IndexConstituentsProvider(
            _online_settings(tmp_path), symbol_search=search, search_bucket=bucket
        )
        provider.get_constituents("stoxx600")

        assert bucket.acquired == len(search.queries)
        assert bucket.acquired >= 2

    def test_a_throttled_source_is_not_worked_around(self, monkeypatch, tmp_path):
        # Harte Projektregel: ein Limit wird nie umgangen, der Fehler geht
        # nach oben - er wird nicht zu einem uebersprungenen Titel verdaut.
        frame = _stoxx_frame([("SAF", "Safran", "Aerospace", "France")])
        monkeypatch.setattr(indices.pd, "read_html", _CountingReadHtml([frame]))
        search = _FakeSearch({}, raises=RateLimitError("Yahoo drosselt"))

        provider = IndexConstituentsProvider(_online_settings(tmp_path), symbol_search=search)
        with pytest.raises(RateLimitError):
            provider.get_constituents("stoxx600")

    def test_a_single_failed_search_does_not_kill_the_fetch(self, monkeypatch, tmp_path):
        # Ein Aussetzer bei einem Titel darf nicht 466 andere kosten. Der
        # betroffene Titel landet sichtbar bei den uebersprungenen.
        frame = _stoxx_frame(
            [
                ("SAF", "Safran", "Aerospace", "France"),
                ("XYZ", "Irgendwas", "Sonstiges", "France"),
            ]
        )
        monkeypatch.setattr(indices.pd, "read_html", _CountingReadHtml([frame]))

        class _FlakySearch(_FakeSearch):
            def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
                self.queries.append(query)
                if query.startswith("XYZ") or query == "Irgendwas":
                    msg = "Verbindung abgebrochen"
                    raise OSError(msg)
                return list(self._responses.get(query, []))

        search = _FlakySearch({"SAF.PA": [_hit("SAF.PA", "Safran SA")]})
        provider = IndexConstituentsProvider(_online_settings(tmp_path), symbol_search=search)
        result = provider.get_constituents("stoxx600")

        assert result.tickers == ["SAF.PA"]
        assert [s.source_ticker for s in result.skipped] == ["XYZ"]


# --------------------------------------------------------------- Offline --


class TestOffline:
    def test_the_index_is_listed_offline(self, tmp_path):
        settings = Settings()
        settings.cache.dir = tmp_path / "cache"
        provider = IndexConstituentsProvider(settings)

        assert any(d.key == "stoxx600" for d in provider.available_indices())

    def test_no_live_fetch_happens_offline(self, tmp_path):
        settings = Settings()
        settings.cache.dir = tmp_path / "cache"
        provider = IndexConstituentsProvider(settings)

        with pytest.raises(indices.ProviderUnavailableError) as excinfo:
            provider.get_constituents("stoxx600")
        assert "STOXX Europe 600" in excinfo.value.user_message
