"""Tests der Finviz-Filtervalidierung, Normalisierung und des Fehler-Mappings.

Kein Test hier geht ins Netz. Geprueft wird gegen das Filter-Dictionary des
installierten Pakets - genau darum ist im Provider nichts hardcodiert.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from btlite.config import Settings
from btlite.core.errors import (
    DataValidationError,
    ProviderUnavailableError,
    RateLimitError,
    UnknownFilterError,
)
from btlite.data.finviz_provider import (
    FinvizProvider,
    _map_exception,
    normalize_screen_table,
)


def _online_settings() -> Settings:
    """Settings mit abgeschaltetem Offline-Modus.

    ``DataSettings.offline`` ist per Default True, damit die App ohne Netz
    startet. Der FinvizProvider verweigert dann bewusst den Dienst. Diese
    Tests pruefen sein Verhalten diesseits dieser Sperre - sie machen
    trotzdem keinen Request, das verbietet die Socket-Sperre der Suite.
    """
    settings = Settings()
    settings.data.offline = False
    return settings


@pytest.fixture
def provider(tmp_path):
    """Provider mit temporaerem Cache. Macht ohne Aufruf keinen Request."""
    settings = _online_settings()
    settings.cache.dir = tmp_path / "cache"
    return FinvizProvider(settings)


class TestAvailableFilters:
    def test_reads_keys_from_the_installed_package(self, provider):
        filters = provider.available_filters()
        assert filters, "Filterliste darf nicht leer sein"
        assert "Sector" in filters
        assert "Index" in filters

    def test_market_cap_key_carries_its_dot(self, provider):
        filters = provider.available_filters()
        assert "Market Cap." in filters, "Der Punkt gehoert zum Key"
        assert "Market Cap" not in filters

    def test_values_are_the_full_display_texts(self, provider):
        # Wichtig fuer Phase 2: die Werte sind NICHT "Mega"/"Large", sondern
        # die vollen Anzeigetexte der Quelle.
        values = provider.available_filters()["Market Cap."]
        assert "Mega ($200bln and more)" in values
        assert "Mega" not in values

    def test_index_values_match_the_documented_ones(self, provider):
        values = provider.available_filters()["Index"]
        for expected in ("S&P 500", "NASDAQ 100", "DJIA", "RUSSELL 2000"):
            assert expected in values

    def test_rsi_key_is_rsi_14_not_rsi(self, provider):
        # docs/DATA_SOURCES.md nennt "RSI"; das Paket kennt nur "RSI (14)".
        filters = provider.available_filters()
        assert "RSI (14)" in filters
        assert "RSI" not in filters


class TestValidateFilters:
    def test_accepts_valid_filters(self, provider):
        provider.validate_filters({"Index": "S&P 500", "Sector": "Technology"})

    def test_accepts_empty_filters(self, provider):
        provider.validate_filters({})

    def test_rejects_key_without_its_dot(self, provider):
        with pytest.raises(UnknownFilterError) as excinfo:
            provider.validate_filters({"Market Cap": "Mega ($200bln and more)"})
        assert "Market Cap." in str(excinfo.value.detail), "Hinweis muss den Punkt nennen"

    def test_rejects_typo_in_key(self, provider):
        with pytest.raises(UnknownFilterError, match=r"Filter 'Sektor' is unknown to Finviz\."):
            provider.validate_filters({"Sektor": "Technology"})

    def test_rejects_wrong_case(self, provider):
        with pytest.raises(UnknownFilterError):
            provider.validate_filters({"sector": "Technology"})

    def test_rejects_invalid_value_for_valid_key(self, provider):
        with pytest.raises(
            UnknownFilterError, match=r"Value 'DAX 40' is not allowed for filter 'Index'\."
        ):
            provider.validate_filters({"Index": "DAX 40"})

    def test_rejects_shorthand_market_cap_value(self, provider):
        with pytest.raises(UnknownFilterError):
            provider.validate_filters({"Market Cap.": "Mega"})

    def test_error_names_the_offending_filter(self, provider):
        with pytest.raises(UnknownFilterError) as excinfo:
            provider.validate_filters({"Quatsch": "egal"})
        assert "Quatsch" in excinfo.value.user_message


class TestNormalizeScreenTable:
    @pytest.fixture
    def raw(self):
        return pd.DataFrame(
            {
                "Ticker": ["aapl", "MSFT", "NVDA"],
                "Company": ["Apple Inc", "Microsoft", "NVIDIA"],
                "P/E": [30.0, 35.0, 60.0],
            }
        )

    def test_ticker_becomes_the_index(self, raw):
        out = normalize_screen_table(raw)
        assert out.index.name == "ticker"
        assert "Ticker" not in out.columns

    def test_tickers_are_normalized_to_uppercase(self, raw):
        out = normalize_screen_table(raw)
        assert "AAPL" in out.index

    def test_other_columns_are_preserved(self, raw):
        out = normalize_screen_table(raw)
        assert list(out.columns) == ["Company", "P/E"]

    def test_duplicate_tickers_are_dropped(self, raw):
        doubled = pd.concat([raw, raw], ignore_index=True)
        out = normalize_screen_table(doubled)
        assert not out.index.has_duplicates
        assert len(out) == 3

    def test_none_yields_empty_frame(self):
        out = normalize_screen_table(None)
        assert out.empty
        assert out.index.name == "ticker"

    def test_empty_frame_stays_empty(self):
        assert normalize_screen_table(pd.DataFrame()).empty

    def test_missing_ticker_column_is_a_layout_break(self):
        # Layoutaenderung erscheint als Fehler, nicht als still leeres Ergebnis.
        raw = pd.DataFrame({"Company": ["Apple"], "P/E": [30.0]})
        with pytest.raises(DataValidationError, match=r"Finviz result without a ticker column\."):
            normalize_screen_table(raw)


class TestErrorMapping:
    def test_cloudflare_wall_maps_to_rate_limit(self):
        from finvizfinance.exceptions import FinvizBlockedError

        mapped = _map_exception(FinvizBlockedError(url="https://finviz.com"))
        assert isinstance(mapped, RateLimitError), "Eine Wall wird nicht umgangen"

    def test_parse_error_maps_to_validation_error(self):
        from finvizfinance.exceptions import FinvizParseError

        mapped = _map_exception(FinvizParseError(url="https://finviz.com", selector="table"))
        assert isinstance(mapped, DataValidationError)

    def test_value_error_maps_to_unknown_filter(self):
        mapped = _map_exception(ValueError("Invalid filter 'Market Cap'"))
        assert isinstance(mapped, UnknownFilterError)

    def test_network_error_maps_to_provider_unavailable(self):
        mapped = _map_exception(ConnectionError("weg"))
        assert isinstance(mapped, ProviderUnavailableError)

    def test_already_mapped_errors_pass_through(self):
        original = RateLimitError("gedrosselt")
        assert _map_exception(original) is original

    def test_raw_finviz_exceptions_never_leak(self):
        from finvizfinance.exceptions import FinvizError

        mapped = _map_exception(FinvizError("irgendwas"))
        assert not isinstance(mapped, FinvizError)


class TestProviderConstruction:
    def test_defaults_to_overview_view(self, provider):
        assert provider.view == "Overview"

    @pytest.mark.parametrize(
        "view", ["Overview", "Valuation", "Financial", "Ownership", "Performance", "Technical"]
    )
    def test_accepts_documented_views(self, tmp_path, view):
        settings = _online_settings()
        settings.cache.dir = tmp_path / "cache"
        assert FinvizProvider(settings, view=view).view == view

    def test_rejects_unknown_view(self, tmp_path):
        settings = _online_settings()
        settings.cache.dir = tmp_path / "cache"
        with pytest.raises(ValueError, match=r"Unknown view 'Quatsch'\. Allowed:"):
            FinvizProvider(settings, view="Quatsch")

    def test_offline_mode_refuses_the_network_provider(self, tmp_path):
        settings = _online_settings()
        settings.cache.dir = tmp_path / "cache"
        settings.data.offline = True
        with pytest.raises(ProviderUnavailableError, match="Offline mode: Finviz is blocked"):
            FinvizProvider(settings)

    def test_uses_the_conservative_finviz_rate(self, provider):
        assert provider._bucket.rps <= 1.0, "Scraping wird bewusst niedrig gedrosselt"

    def test_new_layout_falls_back_to_embedded_json(self, provider, monkeypatch):
        from finvizfinance import util

        from btlite.models.screening import ScreenRequest

        payload = {
            "data": {
                "rows": [
                    {
                        "text": {"ticker": "aapl", "company": "Apple", "sector": "Technology"},
                        "numbers": {"PE": 25.0},
                    },
                    {
                        "text": {"ticker": "msft", "company": "Microsoft", "sector": "Technology"},
                        "numbers": {"PE": 30.0},
                    },
                ],
                "totalCount": 2,
                "pageSize": 20,
            }
        }
        response = SimpleNamespace(
            text='<script type="application/json">' + json.dumps(payload) + "</script>"
        )
        monkeypatch.setattr(util, "fetch", lambda *_args, **_kwargs: response)
        screener = SimpleNamespace(url="https://finviz.test/screener", request_params={"v": 111})

        frame = provider._fetch_modern(screener, ScreenRequest(limit=2))

        assert list(frame.index) == ["AAPL", "MSFT"]
        assert list(frame["Company"]) == ["Apple", "Microsoft"]
