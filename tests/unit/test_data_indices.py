"""Tests des Indexbestands-Abrufs.

Kein Test hier geht ins Netz - die Socket-Sperre der Suite verbietet es.
Die Quellen werden durch Attrappen ersetzt, die genau die Tabellenform
liefern, die der Live-Abruf tatsaechlich zurueckgibt:

* Finviz-Overview: Spalten ``Ticker``, ``Company``, ``Sector``, ...
* Wikipedia DAX: Spalten ``Ticker``, ``Company``, ``Prime Standard Sector``

Beide Formen wurden gegen die echten Quellen verifiziert, bevor sie hier
eingefroren wurden.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from btlite.config import Settings
from btlite.core.errors import (
    DataValidationError,
    ProviderUnavailableError,
    RateLimitError,
)
from btlite.data import indices
from btlite.data.base import IndexProvider
from btlite.data.indices import (
    CONSTITUENTS_TTL_S,
    DEFAULT_INDEX_KEY,
    INDEX_REGISTRY,
    IndexConstituentsProvider,
    _map_finviz_exception,
    _map_wikipedia_exception,
    _parse_finviz_embedded_page,
    available_indices,
    get_index_definition,
)
from btlite.models.common import DataSource
from btlite.models.screening import IndexDefinition, IndexSource

# --------------------------------------------------------------- Helfer --


def _online_settings(tmp_path: Any) -> Settings:
    """Settings mit abgeschaltetem Offline-Modus und temporaerem Cache.

    ``DataSettings.offline`` ist per Default True. Diese Tests pruefen das
    Verhalten diesseits dieser Sperre; einen Request machen sie trotzdem
    nicht, die Quelle ist jeweils eine Attrappe.
    """
    settings = Settings()
    settings.data.offline = False
    settings.cache.dir = tmp_path / "cache"
    return settings


def _finviz_frame(tickers: list[str]) -> pd.DataFrame:
    """Antwortform des Finviz-Overview-Screeners."""
    return pd.DataFrame(
        {
            "Ticker": tickers,
            "Company": [f"{t} Corporation" for t in tickers],
            "Sector": ["Technology"] * len(tickers),
            "Industry": ["Software"] * len(tickers),
            "Country": ["USA"] * len(tickers),
            "Market Cap": [1.0e12] * len(tickers),
        }
    )


def _dax_frame() -> pd.DataFrame:
    """Antwortform der DAX-Tabelle auf Wikipedia, inklusive Endungen."""
    return pd.DataFrame(
        {
            "Ticker": ["ADS.DE", "AIR.PA", "ALV.DE", "BAS.DE"],
            "Logo": ["", "", "", ""],
            "Company": ["Adidas", "Airbus", "Allianz", "BASF"],
            "Prime Standard Sector": ["Retail", "Aerospace", "Insurance", "Chemicals"],
        }
    )


class _FakeScreener:
    """Attrappe von ``finvizfinance.screener.overview.Overview``."""

    def __init__(self, frame: pd.DataFrame | None, *, raises: Exception | None = None) -> None:
        self._frame = frame
        self._raises = raises
        self.filters: dict[str, str] | None = None
        self.limit: int | None = None
        self.calls = 0

    def set_filter(self, *, filters_dict: dict[str, str]) -> None:
        self.filters = dict(filters_dict)

    def screener_view(self, *, limit: int, verbose: int) -> pd.DataFrame | None:
        self.calls += 1
        self.limit = limit
        if self._raises is not None:
            raise self._raises
        return self._frame


def _patch_finviz(monkeypatch: pytest.MonkeyPatch, screener: _FakeScreener) -> None:
    """Haengt die Attrappe in den Importpfad des Providers."""
    import finvizfinance.screener.overview as overview_module

    monkeypatch.setattr(overview_module, "Overview", lambda: screener)


class _CountingReadHtml:
    """Attrappe von ``pandas.read_html``, zaehlt die Aufrufe."""

    def __init__(self, tables: list[pd.DataFrame], *, raises: Exception | None = None) -> None:
        self._tables = tables
        self._raises = raises
        self.calls = 0
        self.storage_options: dict[str, str] | None = None

    def __call__(self, url: str, *, storage_options: dict[str, str] | None = None) -> list[Any]:
        self.calls += 1
        self.storage_options = storage_options
        if self._raises is not None:
            raise self._raises
        return list(self._tables)


# -------------------------------------------------------------- Registry --


class TestRegistry:
    def test_covers_the_indices_the_user_asked_for(self):
        for key in ("sp500", "djia", "nasdaq100", "russell2000", "dax", "stoxx600"):
            assert key in INDEX_REGISTRY, f"{key} fehlt in der Registry"

    def test_default_is_the_sp500(self):
        assert DEFAULT_INDEX_KEY == "sp500"
        assert available_indices()[0].key == "sp500"

    def test_available_indices_lists_every_entry_once(self):
        keys = [d.key for d in available_indices()]
        assert sorted(keys) == sorted(INDEX_REGISTRY)
        assert len(keys) == len(set(keys))

    def test_lookup_is_case_insensitive(self):
        assert get_index_definition("SP500").key == "sp500"
        assert get_index_definition("  dax  ").key == "dax"

    def test_unknown_index_raises(self):
        with pytest.raises(ValueError, match=r"Unknown index 'ftse100'\. Allowed:"):
            get_index_definition("ftse100")

    def test_finviz_values_match_the_packages_filter_dictionary(self):
        from finvizfinance.constants import filter_dict

        allowed = filter_dict["Index"]["option"]
        for definition in INDEX_REGISTRY.values():
            if definition.source is IndexSource.FINVIZ:
                assert definition.finviz_index_value in allowed, (
                    f"{definition.key}: {definition.finviz_index_value!r} ist kein Finviz-Wert"
                )

    def test_a_new_index_needs_only_a_registry_entry(self, monkeypatch, tmp_path):
        # Die Anforderung nennt ausdruecklich "etc." - weitere Indizes muessen
        # ohne Codeaenderung an den Abrufpfaden dazukommen koennen.
        extra = IndexDefinition(
            key="cac40",
            label="CAC 40",
            source=IndexSource.WIKIPEDIA,
            wikipedia_url="https://example.invalid/CAC_40",
            wikipedia_ticker_column="Ticker",
            wikipedia_name_column="Company",
        )
        monkeypatch.setitem(INDEX_REGISTRY, "cac40", extra)
        fake = _CountingReadHtml(
            [pd.DataFrame({"Ticker": ["AI.PA", "OR.PA"], "Company": ["Air Liquide", "L'Oreal"]})]
        )
        monkeypatch.setattr(indices.pd, "read_html", fake)

        provider = IndexConstituentsProvider(_online_settings(tmp_path))
        assert provider.get_tickers("cac40") == ["AI.PA", "OR.PA"]


class TestProtocolConformance:
    def test_provider_satisfies_the_protocol(self, tmp_path):
        provider = IndexConstituentsProvider(_online_settings(tmp_path))
        assert isinstance(provider, IndexProvider)


# --------------------------------------------------------------- Finviz --


class TestFinvizFetch:
    def test_returns_tickers_names_and_sectors(self, monkeypatch, tmp_path):
        screener = _FakeScreener(_finviz_frame(["AAPL", "MSFT", "NVDA"]))
        _patch_finviz(monkeypatch, screener)

        provider = IndexConstituentsProvider(_online_settings(tmp_path))
        result = provider.get_constituents("sp500")

        assert result.tickers == ["AAPL", "MSFT", "NVDA"]
        assert result.members[0].name == "AAPL Corporation"
        assert result.members[0].sector == "Technology"
        assert len(result) == 3

    def test_sends_the_exact_finviz_index_value(self, monkeypatch, tmp_path):
        screener = _FakeScreener(_finviz_frame(["AAPL"]))
        _patch_finviz(monkeypatch, screener)

        IndexConstituentsProvider(_online_settings(tmp_path)).get_constituents("nasdaq100")
        assert screener.filters == {"Index": "NASDAQ 100"}

    def test_requests_far_more_than_the_ten_frozen_tickers(self, monkeypatch, tmp_path):
        # Kern der Anforderung: der Abruf darf nicht bei einer Handvoll
        # Titel gedeckelt sein.
        screener = _FakeScreener(_finviz_frame(["AAPL"]))
        _patch_finviz(monkeypatch, screener)

        IndexConstituentsProvider(_online_settings(tmp_path)).get_constituents("sp500")
        assert screener.limit is not None
        assert screener.limit >= 2000

    def test_five_hundred_members_survive_the_roundtrip(self, monkeypatch, tmp_path):
        tickers = [f"T{i:03d}" for i in range(503)]
        screener = _FakeScreener(_finviz_frame(tickers))
        _patch_finviz(monkeypatch, screener)

        result = IndexConstituentsProvider(_online_settings(tmp_path)).get_constituents("sp500")
        assert len(result) == 503

    def test_duplicates_are_dropped_keeping_the_first(self, monkeypatch, tmp_path):
        screener = _FakeScreener(_finviz_frame(["AAPL", "MSFT", "AAPL"]))
        _patch_finviz(monkeypatch, screener)

        result = IndexConstituentsProvider(_online_settings(tmp_path)).get_constituents("sp500")
        assert result.tickers == ["AAPL", "MSFT"]

    def test_empty_response_is_an_error_not_a_silent_empty_list(self, monkeypatch, tmp_path):
        screener = _FakeScreener(pd.DataFrame())
        _patch_finviz(monkeypatch, screener)
        monkeypatch.setattr(IndexConstituentsProvider, "_fetch_finviz_modern", lambda *_: [])

        provider = IndexConstituentsProvider(_online_settings(tmp_path))
        with pytest.raises(DataValidationError, match="The source returned no constituents for"):
            provider.get_constituents("sp500")

    def test_new_finviz_layout_uses_json_fallback(self, monkeypatch, tmp_path):
        screener = _FakeScreener(None)
        _patch_finviz(monkeypatch, screener)
        expected = [indices.IndexMember(ticker="AAPL", name="Apple Inc", sector="Technology")]
        monkeypatch.setattr(
            IndexConstituentsProvider,
            "_fetch_finviz_modern",
            lambda *_: expected,
        )

        result = IndexConstituentsProvider(_online_settings(tmp_path)).get_constituents("sp500")

        assert result.members == expected

    def test_parses_new_finviz_embedded_payload(self):
        payload = {
            "data": {
                "rows": [
                    {
                        "text": {
                            "ticker": "brk-b",
                            "company": "Berkshire Hathaway Inc",
                            "sector": "Financial",
                        },
                        "numbers": {"price": 500.0},
                    }
                ],
                "totalCount": 503,
                "firstRow": 0,
                "pageSize": 20,
            }
        }
        html = (
            '<html><script type="application/json">'
            + indices.json.dumps(payload)
            + "</script></html>"
        )

        members, total, page_size = _parse_finviz_embedded_page(html)

        assert members[0].ticker == "BRK-B"
        assert members[0].name == "Berkshire Hathaway Inc"
        assert members[0].sector == "Financial"
        assert (total, page_size) == (503, 20)

    def test_missing_ticker_column_is_a_layout_error(self, monkeypatch, tmp_path):
        screener = _FakeScreener(pd.DataFrame({"Symbol": ["AAPL"], "Company": ["Apple"]}))
        _patch_finviz(monkeypatch, screener)

        provider = IndexConstituentsProvider(_online_settings(tmp_path))
        with pytest.raises(DataValidationError, match="The source returns no ticker column"):
            provider.get_constituents("sp500")

    def test_provenance_names_finviz_and_is_not_from_cache(self, monkeypatch, tmp_path):
        screener = _FakeScreener(_finviz_frame(["AAPL"]))
        _patch_finviz(monkeypatch, screener)

        result = IndexConstituentsProvider(_online_settings(tmp_path)).get_constituents("djia")
        assert result.provenance.source is DataSource.FINVIZ
        assert result.provenance.from_cache is False


# ------------------------------------------------------------ Wikipedia --


class TestWikipediaFetch:
    def test_dax_keeps_its_exchange_suffixes(self, monkeypatch, tmp_path):
        # Kritisch: yfinance loest ADS ohne .DE nicht auf den DAX-Titel auf.
        fake = _CountingReadHtml([pd.DataFrame({"Year": [2024]}), _dax_frame()])
        monkeypatch.setattr(indices.pd, "read_html", fake)

        result = IndexConstituentsProvider(_online_settings(tmp_path)).get_constituents("dax")
        assert result.tickers == ["ADS.DE", "AIR.PA", "ALV.DE", "BAS.DE"]

    def test_table_is_selected_by_column_name_not_by_position(self, monkeypatch, tmp_path):
        # pd.read_html(match=...) braucht html5lib, das nicht installiert ist.
        # Die Auswahl laeuft deshalb ueber die Spaltennamen.
        decoys = [pd.DataFrame({"Year": [2020]}), pd.DataFrame({"Milestone": ["x"]})]
        fake = _CountingReadHtml([*decoys, _dax_frame()])
        monkeypatch.setattr(indices.pd, "read_html", fake)

        result = IndexConstituentsProvider(_online_settings(tmp_path)).get_constituents("dax")
        assert len(result) == 4

    def test_names_and_sectors_come_through(self, monkeypatch, tmp_path):
        fake = _CountingReadHtml([_dax_frame()])
        monkeypatch.setattr(indices.pd, "read_html", fake)

        result = IndexConstituentsProvider(_online_settings(tmp_path)).get_constituents("dax")
        assert result.members[0].name == "Adidas"
        assert result.members[0].sector == "Retail"

    def test_sends_a_meaningful_user_agent(self, monkeypatch, tmp_path):
        fake = _CountingReadHtml([_dax_frame()])
        monkeypatch.setattr(indices.pd, "read_html", fake)

        IndexConstituentsProvider(_online_settings(tmp_path)).get_constituents("dax")
        assert fake.storage_options is not None
        assert "BT-Lite" in fake.storage_options["User-Agent"]

    def test_rebuilt_page_without_the_table_is_an_error(self, monkeypatch, tmp_path):
        fake = _CountingReadHtml([pd.DataFrame({"Year": [2020]})])
        monkeypatch.setattr(indices.pd, "read_html", fake)

        provider = IndexConstituentsProvider(_online_settings(tmp_path))
        with pytest.raises(DataValidationError, match="no longer lists a constituents table"):
            provider.get_constituents("dax")

    def test_missing_optional_values_become_none(self, monkeypatch, tmp_path):
        frame = pd.DataFrame(
            {
                "Ticker": ["ADS.DE", "ALV.DE"],
                "Company": ["Adidas", None],
                "Prime Standard Sector": ["Retail", "-"],
            }
        )
        monkeypatch.setattr(indices.pd, "read_html", _CountingReadHtml([frame]))

        result = IndexConstituentsProvider(_online_settings(tmp_path)).get_constituents("dax")
        assert result.members[1].name is None
        assert result.members[1].sector is None


# ----------------------------------------------------------------- Cache --


class TestCaching:
    def test_second_call_does_not_touch_the_source(self, monkeypatch, tmp_path):
        screener = _FakeScreener(_finviz_frame(["AAPL", "MSFT"]))
        _patch_finviz(monkeypatch, screener)
        settings = _online_settings(tmp_path)

        first = IndexConstituentsProvider(settings).get_constituents("sp500")
        second = IndexConstituentsProvider(settings).get_constituents("sp500")

        assert screener.calls == 1, "Der zweite Abruf ging erneut zur Quelle"
        assert first.tickers == second.tickers
        assert second.provenance.from_cache is True

    def test_cached_payload_keeps_names_and_sectors(self, monkeypatch, tmp_path):
        screener = _FakeScreener(_finviz_frame(["AAPL"]))
        _patch_finviz(monkeypatch, screener)
        settings = _online_settings(tmp_path)

        IndexConstituentsProvider(settings).get_constituents("sp500")
        cached = IndexConstituentsProvider(settings).get_constituents("sp500")

        assert cached.members[0].name == "AAPL Corporation"
        assert cached.members[0].sector == "Technology"

    def test_wikipedia_is_cached_too(self, monkeypatch, tmp_path):
        fake = _CountingReadHtml([_dax_frame()])
        monkeypatch.setattr(indices.pd, "read_html", fake)
        settings = _online_settings(tmp_path)

        IndexConstituentsProvider(settings).get_constituents("dax")
        IndexConstituentsProvider(settings).get_constituents("dax")
        assert fake.calls == 1

    def test_indices_do_not_share_a_cache_entry(self, monkeypatch, tmp_path):
        frames = {"S&P 500": _finviz_frame(["AAPL"]), "DJIA": _finviz_frame(["BA", "CAT"])}

        class _Routing(_FakeScreener):
            def screener_view(self, *, limit: int, verbose: int) -> pd.DataFrame:
                self.calls += 1
                assert self.filters is not None
                return frames[self.filters["Index"]]

        screener = _Routing(None)
        _patch_finviz(monkeypatch, screener)
        settings = _online_settings(tmp_path)

        provider = IndexConstituentsProvider(settings)
        assert provider.get_tickers("sp500") == ["AAPL"]
        assert provider.get_tickers("djia") == ["BA", "CAT"]

    def test_ttl_is_a_week(self):
        # Bestaende aendern sich quartalsweise; eine Woche ist der Kompromiss
        # aus Aktualitaet und Schonung der Quelle.
        assert CONSTITUENTS_TTL_S == 7 * 24 * 3600

    def test_cache_stores_the_payload_under_the_ttl(self, monkeypatch, tmp_path):
        screener = _FakeScreener(_finviz_frame(["AAPL"]))
        _patch_finviz(monkeypatch, screener)
        settings = _online_settings(tmp_path)

        recorded: list[float | None] = []
        from btlite.core.cache import DataFrameCache

        real_set = DataFrameCache.set_df

        def _spy(self, key, df, ttl_s=None):
            recorded.append(ttl_s)
            return real_set(self, key, df, ttl_s)

        monkeypatch.setattr(DataFrameCache, "set_df", _spy)
        IndexConstituentsProvider(settings).get_constituents("sp500")
        assert recorded == [CONSTITUENTS_TTL_S]


# --------------------------------------------------------------- Offline --


class TestOfflineBehaviour:
    def test_live_fetch_is_refused_offline(self, tmp_path):
        settings = Settings()
        settings.cache.dir = tmp_path / "cache"
        assert settings.data.offline is True

        provider = IndexConstituentsProvider(settings)
        with pytest.raises(ProviderUnavailableError) as excinfo:
            provider.get_constituents("sp500")

        # Kein Absturz, keine stille leere Liste: ein typisierter Fehler mit
        # einer Meldung, die den Ausweg nennt.
        assert "BTLITE_DATA_OFFLINE" in excinfo.value.user_message
        assert "S&P 500" in excinfo.value.user_message

    def test_offline_does_not_silently_fall_back_to_a_stub_universe(self, tmp_path):
        settings = Settings()
        settings.cache.dir = tmp_path / "cache"
        provider = IndexConstituentsProvider(settings)
        with pytest.raises(ProviderUnavailableError):
            provider.get_tickers("dax")

    def test_index_list_is_available_offline(self, tmp_path):
        settings = Settings()
        settings.cache.dir = tmp_path / "cache"
        provider = IndexConstituentsProvider(settings)

        listed = provider.available_indices()
        assert next(d.key for d in listed) == "sp500"
        assert len(listed) == len(INDEX_REGISTRY)

    def test_cached_constituents_are_served_offline(self, monkeypatch, tmp_path):
        # Ein einmal geholter Bestand bleibt ohne Netz nutzbar - der Cache ist
        # die Bruecke ueber den Offline-Modus.
        screener = _FakeScreener(_finviz_frame(["AAPL", "MSFT"]))
        _patch_finviz(monkeypatch, screener)

        online = _online_settings(tmp_path)
        IndexConstituentsProvider(online).get_constituents("sp500")

        offline = Settings()
        offline.cache.dir = tmp_path / "cache"
        assert offline.data.offline is True
        result = IndexConstituentsProvider(offline).get_constituents("sp500")

        assert result.tickers == ["AAPL", "MSFT"]
        assert result.provenance.from_cache is True


# ---------------------------------------------------------- Fehlerpfade --


class TestErrorMapping:
    def test_finviz_block_becomes_a_rate_limit_error(self):
        from finvizfinance.exceptions import FinvizBlockedError

        mapped = _map_finviz_exception(FinvizBlockedError("blocked"))
        assert isinstance(mapped, RateLimitError)

    def test_finviz_parse_error_becomes_a_validation_error(self):
        from finvizfinance.exceptions import FinvizParseError

        mapped = _map_finviz_exception(FinvizParseError("layout"))
        assert isinstance(mapped, DataValidationError)

    def test_unknown_finviz_error_becomes_provider_unavailable(self):
        mapped = _map_finviz_exception(OSError("connection reset"))
        assert isinstance(mapped, ProviderUnavailableError)

    def test_already_mapped_errors_pass_through_unchanged(self):
        original = RateLimitError("schon abgebildet")
        assert _map_finviz_exception(original) is original

    def test_rate_limit_is_never_retried(self, monkeypatch, tmp_path):
        # Harte Projektregel: ein Block der Quelle wird nicht ausgesessen.
        from finvizfinance.exceptions import FinvizBlockedError

        screener = _FakeScreener(None, raises=FinvizBlockedError("blocked"))
        _patch_finviz(monkeypatch, screener)

        provider = IndexConstituentsProvider(_online_settings(tmp_path))
        with pytest.raises(RateLimitError):
            provider.get_constituents("sp500")
        assert screener.calls == 1, "RateLimitError darf keinen Retry ausloesen"

    def test_wikipedia_throttling_becomes_a_rate_limit_error(self):
        definition = INDEX_REGISTRY["dax"]
        mapped = _map_wikipedia_exception(OSError("HTTP Error 429: Too Many Requests"), definition)
        assert isinstance(mapped, RateLimitError)

    def test_wikipedia_without_any_table_is_a_validation_error(self):
        definition = INDEX_REGISTRY["dax"]
        mapped = _map_wikipedia_exception(ValueError("No tables found"), definition)
        assert isinstance(mapped, DataValidationError)

    def test_wikipedia_network_failure_is_provider_unavailable(self, monkeypatch, tmp_path):
        fake = _CountingReadHtml([], raises=OSError("dns failure"))
        monkeypatch.setattr(indices.pd, "read_html", fake)

        provider = IndexConstituentsProvider(_online_settings(tmp_path))
        with pytest.raises(ProviderUnavailableError):
            provider.get_constituents("dax")
