"""Tests der yfinance-Normalisierung und des Fehler-Mappings.

Kein Test hier geht ins Netz: geprueft werden die reinen
Normalisierungsfunktionen und die Abbildung roher yfinance-Exceptions auf die
Fehlerhierarchie des Projekts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btlite.config import Settings
from btlite.core.cache import DataFrameCache
from btlite.core.errors import (
    DataValidationError,
    ProviderUnavailableError,
    RateLimitError,
    SymbolNotFoundError,
)
from btlite.data.yfinance_provider import (
    YFinanceProvider,
    _map_exception,
    normalize_ohlcv,
    normalize_statement,
    to_currency,
)
from btlite.models.common import OHLCV_COLUMNS, Currency


@pytest.fixture
def raw_history():
    """Rohframe, wie ``Ticker.history`` ihn liefert: Gross-Spalten, naiver Index."""
    idx = pd.DatetimeIndex(["2024-01-03", "2024-01-02", "2024-01-04"], name="Date")
    return pd.DataFrame(
        {
            "Open": [101.0, 100.0, 102.0],
            "High": [103.0, 101.0, 104.0],
            "Low": [99.0, 98.0, 100.0],
            "Close": [102.0, 100.5, 103.5],
            "Volume": [1_000_000, 900_000, 1_100_000],
            "Dividends": [0.0, 0.0, 0.0],
            "Stock Splits": [0.0, 0.0, 0.0],
        },
        index=idx,
    )


def test_screening_history_skips_the_currency_request(
    tmp_path, monkeypatch: pytest.MonkeyPatch, raw_history: pd.DataFrame
) -> None:
    class _Ticker:
        def history(self, **_kwargs: object) -> pd.DataFrame:
            return raw_history

    settings = Settings()
    settings.data.offline = False
    settings.cache.dir = tmp_path / "cache"
    provider = YFinanceProvider(settings, cache=DataFrameCache(settings.cache))
    monkeypatch.setattr(provider, "_ticker", lambda _symbol: _Ticker())
    monkeypatch.setattr(
        provider,
        "_currency_of",
        lambda _symbol: (_ for _ in ()).throw(AssertionError("currency request")),
    )

    result = provider.get_history("AAPL", include_currency=False)

    assert result.currency is Currency.OTHER
    assert not result.data.empty


class TestNormalizeOhlcvColumns:
    def test_lowercases_to_contract_columns(self, raw_history):
        out = normalize_ohlcv(raw_history, ticker="AAPL")
        for column in OHLCV_COLUMNS:
            assert column in out.columns

    def test_maps_stock_splits_with_underscore(self, raw_history):
        out = normalize_ohlcv(raw_history, ticker="AAPL")
        assert "stock_splits" in out.columns, "'Stock Splits' -> 'stock_splits'"
        assert "Stock Splits" not in out.columns

    def test_keeps_optional_columns(self, raw_history):
        out = normalize_ohlcv(raw_history, ticker="AAPL")
        assert "dividends" in out.columns

    def test_drops_unknown_columns(self, raw_history):
        raw_history["Irgendwas"] = 1.0
        out = normalize_ohlcv(raw_history, ticker="AAPL")
        assert "irgendwas" not in out.columns

    def test_no_adj_close_expected(self, raw_history):
        # auto_adjust=True ist Default seit yfinance 0.2 - es gibt keine
        # 'Adj Close'-Spalte mehr, 'close' ist bereits bereinigt.
        out = normalize_ohlcv(raw_history, ticker="AAPL")
        assert "adj_close" not in out.columns

    def test_flattens_multiindex_columns(self):
        idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
        cols = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], ["AAPL"]],
        )
        raw = pd.DataFrame([[1.0] * 5, [2.0] * 5], index=idx, columns=cols)
        out = normalize_ohlcv(raw, ticker="AAPL")
        assert list(out.columns) == list(OHLCV_COLUMNS)

    def test_missing_required_column_is_a_contract_breach(self, raw_history):
        with pytest.raises(DataValidationError, match="Price data for AAPL is incomplete"):
            normalize_ohlcv(raw_history.drop(columns=["Close"]), ticker="AAPL")


class TestNormalizeOhlcvIndex:
    def test_index_becomes_tz_aware_utc(self, raw_history):
        out = normalize_ohlcv(raw_history, ticker="AAPL")
        assert str(out.index.tz) == "UTC"

    def test_index_is_named_date(self, raw_history):
        out = normalize_ohlcv(raw_history, ticker="AAPL")
        assert out.index.name == "date"

    def test_index_is_sorted_ascending(self, raw_history):
        out = normalize_ohlcv(raw_history, ticker="AAPL")
        assert out.index.is_monotonic_increasing, "Rohframe war absichtlich unsortiert"

    def test_converts_non_utc_timezone(self, raw_history):
        raw_history.index = raw_history.index.tz_localize("America/New_York")
        out = normalize_ohlcv(raw_history, ticker="AAPL")
        assert str(out.index.tz) == "UTC"

    def test_duplicate_index_entries_are_removed(self, raw_history):
        doubled = pd.concat([raw_history, raw_history])
        out = normalize_ohlcv(doubled, ticker="AAPL")
        assert not out.index.has_duplicates
        assert len(out) == 3

    def test_last_duplicate_wins(self):
        # Yahoo korrigiert Historien rueckwirkend - der spaetere Wert gilt.
        idx = pd.DatetimeIndex(["2024-01-02", "2024-01-02"])
        raw = pd.DataFrame(
            {
                "Open": [1.0, 2.0],
                "High": [1.0, 2.0],
                "Low": [1.0, 2.0],
                "Close": [10.0, 20.0],
                "Volume": [1.0, 2.0],
            },
            index=idx,
        )
        out = normalize_ohlcv(raw, ticker="AAPL")
        assert out["close"].iloc[0] == 20.0

    def test_rows_without_close_are_dropped(self, raw_history):
        raw_history.loc[raw_history.index[0], "Close"] = np.nan
        out = normalize_ohlcv(raw_history, ticker="AAPL")
        assert len(out) == 2, "Eine Kerze ohne Schlusskurs traegt keine Information"

    def test_non_datetime_index_is_rejected(self, raw_history):
        raw_history.index = ["a", "b", "c"]
        with pytest.raises(DataValidationError):
            normalize_ohlcv(raw_history, ticker="AAPL")


class TestNormalizeOhlcvEmpty:
    def test_empty_frame_yields_empty_contract_frame(self):
        out = normalize_ohlcv(pd.DataFrame(), ticker="AAPL")
        assert out.empty
        assert list(out.columns) == list(OHLCV_COLUMNS)
        assert str(out.index.tz) == "UTC"

    def test_input_is_not_mutated(self, raw_history):
        before = raw_history.copy()
        normalize_ohlcv(raw_history, ticker="AAPL")
        pd.testing.assert_frame_equal(raw_history, before)


class TestNormalizeStatement:
    @pytest.fixture
    def raw_statement(self):
        """Rohframe wie yfinance ihn liefert: Zeilen = Positionen."""
        cols = pd.DatetimeIndex(["2024-09-30", "2022-09-30", "2023-09-30"])
        return pd.DataFrame(
            {
                cols[0]: [391_035_000_000.0, 93_736_000_000.0],
                cols[1]: [394_328_000_000.0, 99_803_000_000.0],
                cols[2]: [383_285_000_000.0, 96_995_000_000.0],
            },
            index=["TotalRevenue", "NetIncome"],
        )

    def test_orientation_is_preserved_not_transposed(self, raw_statement):
        # Der Vertrag will genau die yfinance-Orientierung: eine Position
        # bleibt ueber die Zeit eine Zeile.
        out = normalize_statement(raw_statement, ticker="AAPL")
        assert "TotalRevenue" in out.index
        assert out.shape == (2, 3)

    def test_index_is_named_line_item(self, raw_statement):
        out = normalize_statement(raw_statement, ticker="AAPL")
        assert out.index.name == "line_item"

    def test_columns_become_tz_aware_utc(self, raw_statement):
        out = normalize_statement(raw_statement, ticker="AAPL")
        assert str(out.columns.tz) == "UTC"

    def test_columns_are_sorted_ascending(self, raw_statement):
        out = normalize_statement(raw_statement, ticker="AAPL")
        assert list(out.columns) == sorted(out.columns), "Rohframe war unsortiert"

    def test_raw_line_item_names_are_kept(self, raw_statement):
        # Die semantische Normalisierung passiert genau einmal, in
        # analytics/statements.py - nicht in der Datenschicht.
        out = normalize_statement(raw_statement, ticker="AAPL")
        assert "TotalRevenue" in out.index
        assert "revenue" not in out.index

    def test_empty_statement_is_empty_not_an_error(self):
        out = normalize_statement(pd.DataFrame(), ticker="AAPL")
        assert out.empty

    def test_non_period_columns_are_rejected(self):
        raw = pd.DataFrame({"a": [1.0], "b": [2.0]}, index=["TotalRevenue"])
        with pytest.raises(
            DataValidationError, match="The statement columns for AAPL are not periods"
        ):
            normalize_statement(raw, ticker="AAPL")


class TestErrorMapping:
    def test_rate_limit_maps_to_rate_limit_error(self):
        from yfinance.exceptions import YFRateLimitError

        mapped = _map_exception(YFRateLimitError(), "AAPL")
        assert isinstance(mapped, RateLimitError), "429 muss als RateLimitError erscheinen"

    def test_missing_prices_maps_to_symbol_not_found(self):
        from yfinance.exceptions import YFPricesMissingError

        mapped = _map_exception(YFPricesMissingError("ZZZZ", ""), "ZZZZ")
        assert isinstance(mapped, SymbolNotFoundError)

    def test_missing_timezone_maps_to_symbol_not_found(self):
        from yfinance.exceptions import YFTzMissingError

        mapped = _map_exception(YFTzMissingError("ZZZZ"), "ZZZZ")
        assert isinstance(mapped, SymbolNotFoundError)

    def test_invalid_period_maps_to_validation_error(self):
        from yfinance.exceptions import YFInvalidPeriodError

        mapped = _map_exception(YFInvalidPeriodError("AAPL", "77y", "1d,5d"), "AAPL")
        assert isinstance(mapped, DataValidationError)

    def test_network_error_maps_to_provider_unavailable(self):
        mapped = _map_exception(ConnectionError("DNS weg"), "AAPL")
        assert isinstance(mapped, ProviderUnavailableError)

    def test_unknown_exception_maps_to_provider_unavailable(self):
        mapped = _map_exception(RuntimeError("was auch immer"), "AAPL")
        assert isinstance(mapped, ProviderUnavailableError)

    def test_already_mapped_errors_pass_through_unchanged(self):
        original = SymbolNotFoundError("kenne ich nicht")
        assert _map_exception(original, "AAPL") is original

    def test_mapped_errors_carry_user_message_and_detail(self):
        mapped = _map_exception(ConnectionError("DNS weg"), "AAPL")
        assert mapped.user_message
        assert mapped.detail is not None
        assert "AAPL" in mapped.detail

    def test_raw_yfinance_exceptions_never_leak(self):
        from yfinance.exceptions import YFException

        for exc in (YFException("x"), ValueError("y"), OSError("z")):
            mapped = _map_exception(exc, "AAPL")
            assert not isinstance(mapped, YFException | ValueError | OSError)


class TestCurrencyMapping:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("USD", Currency.USD), ("eur", Currency.EUR), (" GBP ", Currency.GBP)],
    )
    def test_known_codes_are_mapped(self, raw, expected):
        assert to_currency(raw) is expected

    @pytest.mark.parametrize("raw", ["XYZ", "", None, 42])
    def test_unknown_input_falls_back_to_other(self, raw):
        assert to_currency(raw) is Currency.OTHER
