"""Tests des FixtureProvider - des Providers, gegen den Phase 2 arbeitet.

Er ist damit ein zentrales Deliverable, kein Nebenprodukt: erfuellt er die
Protocols nicht exakt, brechen alle darauf aufbauenden Schichten.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from btlite.core.clock import utc_now
from btlite.core.errors import SymbolNotFoundError
from btlite.data.base import (
    FundamentalsProvider,
    InstrumentProvider,
    MarketDataProvider,
    PriceProvider,
    ScreenerProvider,
)
from btlite.data.fixtures import FixtureProvider
from btlite.models.common import OHLCV_COLUMNS, DataSource, Interval, Period
from btlite.models.screening import ScreenRequest

#: Die zehn Ticker, fuer die Kursfixtures vorliegen.
EXPECTED_TICKERS = ("AAPL", "MSFT", "NVDA", "SPY", "JNJ", "XOM", "KO", "PG", "JPM", "WMT")

#: Ticker mit vollstaendigen Statement-Fixtures.
STATEMENT_TICKERS = ("AAPL", "MSFT", "JNJ")


@pytest.fixture
def provider():
    """FixtureProvider auf dem echten Fixture-Verzeichnis."""
    return FixtureProvider()


class TestProtocolCompliance:
    # Der isinstance-Check gegen die runtime_checkable Protocols ist die
    # eigentliche Zusage an alle nachgelagerten Schichten.
    @pytest.mark.parametrize(
        "protocol",
        [
            InstrumentProvider,
            PriceProvider,
            FundamentalsProvider,
            ScreenerProvider,
            MarketDataProvider,
        ],
    )
    def test_satisfies_protocol(self, provider, protocol):
        assert isinstance(provider, protocol)

    def test_is_a_full_market_data_provider(self, provider):
        # Die Sammelrolle, die das UI haelt und weiterreicht.
        assert isinstance(provider, MarketDataProvider)


class TestAvailability:
    def test_all_ten_price_tickers_are_present(self, provider):
        # Zehn Ticker, damit Phase 2 eine Portfolio-Optimierung testen kann.
        assert set(provider.available_tickers()) == set(EXPECTED_TICKERS)

    def test_benchmark_is_available(self, provider):
        assert "SPY" in provider.available_tickers(), "SPY ist der Default-Benchmark"

    def test_unknown_directory_yields_no_tickers(self, tmp_path):
        assert FixtureProvider(tmp_path / "leer").available_tickers() == []


class TestGetHistory:
    def test_returns_contract_columns(self, provider):
        series = provider.get_history("AAPL")
        for column in OHLCV_COLUMNS:
            assert column in series.data.columns

    def test_index_is_tz_aware_utc_ascending_unique(self, provider):
        data = provider.get_history("AAPL").data
        assert str(data.index.tz) == "UTC"
        assert data.index.name == "date"
        assert data.index.is_monotonic_increasing
        assert not data.index.has_duplicates

    def test_has_several_years_of_daily_bars(self, provider):
        series = provider.get_history("AAPL")
        assert len(series.data) > 1000, "Mehrere Jahre Tagesdaten erwartet"

    def test_no_adj_close_column(self, provider):
        # auto_adjust=True: close ist bereits bereinigt.
        assert "adj_close" not in provider.get_history("AAPL").data.columns

    def test_adjusted_flag_is_recorded(self, provider):
        assert provider.get_history("AAPL", auto_adjust=True).adjusted is True

    def test_provenance_marks_fixture_origin(self, provider):
        prov = provider.get_history("AAPL").provenance
        assert prov.source is DataSource.FIXTURE
        assert prov.from_cache is True, "Eingefrorene Daten sind nie frisch"

    def test_currency_comes_from_the_info_fixture(self, provider):
        assert provider.get_history("AAPL").currency.value == "USD"

    def test_start_slices_the_series(self, provider):
        full = provider.get_history("AAPL")
        cut = provider.get_history("AAPL", start=datetime(2024, 1, 1, tzinfo=UTC))
        assert len(cut.data) < len(full.data)
        assert cut.data.index[0] >= pd.Timestamp("2024-01-01", tz="UTC")

    def test_end_slices_the_series(self, provider):
        cut = provider.get_history("AAPL", end=datetime(2023, 1, 1, tzinfo=UTC))
        assert cut.data.index[-1] <= pd.Timestamp("2023-01-01", tz="UTC")

    def test_unknown_symbol_raises(self, provider):
        with pytest.raises(SymbolNotFoundError):
            provider.get_history("ZZZZ")

    def test_empty_slice_raises_instead_of_returning_empty(self, provider):
        with pytest.raises(SymbolNotFoundError):
            provider.get_history("AAPL", start=datetime(2099, 1, 1, tzinfo=UTC))

    def test_returns_are_computable(self, provider):
        returns = provider.get_history("AAPL").returns()
        assert len(returns) > 1000
        assert not returns.isna().any(), "Renditen duerfen keine Luecken haben"

    def test_caller_cannot_mutate_the_internal_buffer(self, provider):
        first = provider.get_history("AAPL")
        first.data.loc[first.data.index[0], "close"] = -999.0
        second = provider.get_history("AAPL")
        assert second.data["close"].iloc[0] != -999.0


class TestGetHistories:
    def test_returns_all_ten(self, provider):
        out = provider.get_histories(EXPECTED_TICKERS)
        assert set(out) == set(EXPECTED_TICKERS)

    def test_unknown_symbols_are_skipped_not_fatal(self, provider):
        out = provider.get_histories(["AAPL", "ZZZZ", "MSFT"])
        assert set(out) == {"AAPL", "MSFT"}, "Ein unbekanntes Symbol kippt nicht den Aufruf"

    def test_duplicates_are_collapsed(self, provider):
        out = provider.get_histories(["AAPL", "aapl", "AAPL"])
        assert set(out) == {"AAPL"}


class TestGetQuotes:
    def test_derives_quotes_from_the_last_bar(self, provider):
        quotes = provider.get_quotes(["AAPL", "MSFT"])
        assert set(quotes) == {"AAPL", "MSFT"}
        history = provider.get_history("AAPL")
        assert quotes["AAPL"].price == pytest.approx(history.data["close"].iloc[-1])

    def test_timestamp_is_the_bar_not_now(self, provider):
        quote = provider.get_quotes(["AAPL"])["AAPL"]
        last_bar = provider.get_history("AAPL").data.index[-1]
        assert quote.timestamp == last_bar.to_pydatetime(), "Ehrlich datiert"

    def test_previous_close_enables_change_calculation(self, provider):
        quote = provider.get_quotes(["AAPL"])["AAPL"]
        assert quote.previous_close is not None
        assert quote.change_pct is not None

    def test_unknown_symbols_are_skipped(self, provider):
        assert provider.get_quotes(["AAPL", "ZZZZ"]).keys() == {"AAPL"}

    def test_delay_reflects_real_fixture_age_not_a_live_quote(self, provider):
        """Der ausgewiesene Verzug ist das echte Alter der Fixtures.

        Fixture-Preise als viertelstuendlich verzoegert auszuweisen waere eine
        Falschangabe: sie sind Tage alt. Das Projekt gibt nichts als aktueller
        aus, als es ist.
        """
        prov = provider.get_quotes(["AAPL"])["AAPL"].provenance
        assert prov.source is DataSource.FIXTURE
        assert prov.from_cache is True
        assert prov.delay_minutes is not None
        # Gegen den Einfrierzeitpunkt geprueft, nicht gegen eine feste Zahl.
        # Ein blosses "!= 15" schlaegt fehl, sobald das echte Alter zufaellig
        # 15 Minuten betraegt - direkt nach einem Lauf von
        # refresh_fixtures.py fuer exakt eine Minute. Eine feste Untergrenze
        # waere genauso falsch: frisch gezogene Fixtures sind legitim jung.
        # Und ``fetched_at`` taugt nicht als Bezug, das ist der LESEzeitpunkt.
        age = (utc_now() - provider._frozen_at).total_seconds() / 60
        assert prov.delay_minutes == pytest.approx(age, abs=2), (
            f"delay_minutes={prov.delay_minutes} passt nicht zum Fixture-Alter "
            f"{age:.0f} min - der Wert muss das echte Alter sein, nicht der "
            "Delay eines Live-Quotes"
        )
        assert prov.note is not None
        assert "As of" in prov.note, "Einfrierdatum muss im UI sichtbar sein"

    def test_empty_input_yields_empty_output(self, provider):
        assert provider.get_quotes([]) == {}


class TestGetStatements:
    @pytest.mark.parametrize("ticker", STATEMENT_TICKERS)
    def test_annual_bundle_is_complete(self, provider, ticker):
        bundle = provider.get_statements(ticker, period=Period.ANNUAL)
        assert bundle.income is not None
        assert bundle.balance is not None
        assert bundle.cashflow is not None

    @pytest.mark.parametrize("ticker", STATEMENT_TICKERS)
    def test_quarterly_bundle_is_complete(self, provider, ticker):
        bundle = provider.get_statements(ticker, period=Period.QUARTERLY)
        assert bundle.income is not None
        assert bundle.balance is not None
        assert bundle.cashflow is not None

    def test_orientation_is_rows_are_line_items(self, provider):
        # Die yfinance-Orientierung, die der Vertrag bewusst beibehaelt.
        income = provider.get_statements("AAPL").income
        assert income.data.index.name == "line_item"
        assert income.data.shape[0] > income.data.shape[1]

    def test_columns_are_tz_aware_utc_periods(self, provider):
        income = provider.get_statements("AAPL").income
        assert str(income.data.columns.tz) == "UTC"

    def test_periods_are_ascending(self, provider):
        periods = provider.get_statements("AAPL").income.periods
        assert periods == sorted(periods)

    def test_line_item_names_are_still_raw(self, provider):
        # Die Normalisierung passiert erst in analytics/statements.py.
        index = provider.get_statements("AAPL").income.data.index
        assert any("Revenue" in str(i) for i in index)

    def test_apple_fiscal_year_ends_in_september(self, provider):
        # Fiskaljahr-Versatz: Apple schliesst Ende September ab, nicht im
        # Dezember. Ein Peer-Vergleich ist dadurch nicht periodengenau.
        assert provider.get_statements("AAPL").income.fiscal_year_end_month == 9

    def test_get_reads_the_latest_period(self, provider):
        income = provider.get_statements("AAPL").income
        revenue_rows = [i for i in income.data.index if str(i) == "TotalRevenue"]
        if revenue_rows:
            assert income.get("TotalRevenue") is not None

    def test_missing_line_item_is_none_not_zero(self, provider):
        # Harte Projektregel: fehlende Werte sind None, nie ein stiller Nullwert.
        assert provider.get_statements("AAPL").income.get("GibtEsNicht") is None

    def test_unknown_symbol_raises(self, provider):
        with pytest.raises(SymbolNotFoundError):
            provider.get_statements("ZZZZ")

    def test_currency_is_recorded(self, provider):
        assert provider.get_statements("AAPL").currency.value == "USD"


class TestGetInstrument:
    def test_returns_master_data(self, provider):
        instrument = provider.get_instrument("AAPL")
        assert instrument.ticker == "AAPL"
        assert instrument.name
        assert instrument.sector == "Technology"

    def test_asset_class_is_mapped(self, provider):
        assert provider.get_instrument("AAPL").asset_class.value == "equity"

    def test_etf_is_recognized(self, provider):
        assert provider.get_instrument("SPY").asset_class.value == "etf"

    def test_market_cap_is_populated(self, provider):
        assert provider.get_instrument("AAPL").market_cap > 0

    def test_ticker_is_normalized(self, provider):
        assert provider.get_instrument("aapl").ticker == "AAPL"

    def test_unknown_symbol_raises(self, provider):
        with pytest.raises(SymbolNotFoundError):
            provider.get_instrument("ZZZZ")


class TestScreen:
    def test_returns_the_frozen_result(self, provider):
        result = provider.screen(ScreenRequest(finviz_filters={"Index": "S&P 500"}))
        assert len(result.data) > 0
        assert result.data.index.name == "ticker"

    def test_limit_is_applied(self, provider):
        result = provider.screen(ScreenRequest(finviz_filters={}, limit=5))
        assert len(result.data) == 5

    def test_truncation_is_reported(self, provider):
        result = provider.screen(ScreenRequest(finviz_filters={}, limit=5))
        assert result.is_truncated is True
        assert result.total_matches > 5

    def test_tickers_are_exposed(self, provider):
        result = provider.screen(ScreenRequest(finviz_filters={}, limit=3))
        assert len(result.tickers) == 3
        assert all(t.isupper() for t in result.tickers)

    def test_provenance_marks_fixture_origin(self, provider):
        result = provider.screen(ScreenRequest(finviz_filters={}))
        assert result.provenance.source is DataSource.FIXTURE

    def test_available_filters_are_exposed(self, provider):
        filters = provider.available_filters()
        assert "Market Cap." in filters, "Der Punkt gehoert zum Key"


class TestOfflineGuarantee:
    def test_full_workflow_runs_without_network(self, provider):
        # Die gesamte Suite laeuft mit blockierten Sockets. Dass dieser Test
        # durchlaeuft, ist der Beweis: der Provider geht nie ins Netz.
        instrument = provider.get_instrument("AAPL")
        history = provider.get_history("AAPL", interval=Interval.D1)
        quotes = provider.get_quotes(["AAPL", "MSFT"])
        statements = provider.get_statements("AAPL")
        screen = provider.screen(ScreenRequest(finviz_filters={}, limit=10))

        assert instrument.ticker == "AAPL"
        assert not history.data.empty
        assert quotes
        assert statements.income is not None
        assert not screen.data.empty
