"""Tests der Statement-Normalisierung und TTM-Bildung."""

from __future__ import annotations

import pandas as pd
import pytest

from btlite.analytics.statements import (
    TTM_QUARTERS,
    compute_ttm,
    compute_ttm_bundle,
    get_value,
    infer_fiscal_year_end_month,
    normalize_bundle,
    normalize_statement,
)
from btlite.core.errors import AnalyticsError, InsufficientDataError
from btlite.data.fixtures import FixtureProvider
from btlite.models.common import DataSource, Period
from btlite.models.fundamentals import FinancialStatement, LineItem, StatementType


@pytest.fixture
def provider() -> FixtureProvider:
    return FixtureProvider()


@pytest.fixture
def annual(provider):
    return normalize_bundle(provider.get_statements("AAPL", period=Period.ANNUAL))


@pytest.fixture
def quarterly(provider):
    return normalize_bundle(provider.get_statements("AAPL", period=Period.QUARTERLY))


class TestNormalization:
    def test_maps_real_provider_names(self, annual):
        """Die echten yfinance-Rohnamen sind CamelCase, nicht "Total Revenue"."""
        assert get_value(annual, LineItem.REVENUE) is not None
        assert get_value(annual, LineItem.NET_INCOME) is not None
        assert get_value(annual, LineItem.TOTAL_ASSETS) is not None
        assert get_value(annual, LineItem.OPERATING_CASHFLOW) is not None

    def test_canonical_names_replace_raw_names(self, annual):
        assert "revenue" in annual.income.data.index
        assert "TotalRevenue" not in annual.income.data.index

    def test_unmapped_rows_survive(self, annual):
        """Nicht gemappte Rohzeilen bleiben erhalten - nichts geht verloren."""
        assert annual.income.unmapped_line_items
        for name in annual.income.unmapped_line_items:
            assert name in annual.income.data.index

    def test_no_row_is_lost(self, provider):
        raw = provider.get_statements("AAPL", period=Period.ANNUAL)
        normalized = normalize_statement(raw.income)
        assert len(normalized.data.index) == len(raw.income.data.index)

    def test_periods_are_preserved(self, provider):
        raw = provider.get_statements("AAPL", period=Period.ANNUAL)
        normalized = normalize_statement(raw.income)
        assert list(normalized.data.columns) == list(raw.income.data.columns)

    def test_original_is_not_mutated(self, provider):
        raw = provider.get_statements("AAPL", period=Period.ANNUAL)
        before = list(raw.income.data.index)
        normalize_statement(raw.income)
        assert list(raw.income.data.index) == before

    def test_index_is_named(self, annual):
        assert annual.income.data.index.name == "line_item"

    def test_empty_alias_does_not_win(self, provenance):
        """Ein Alias mit ausschliesslich Fehlwerten darf einen echten nicht verdraengen."""
        cols = pd.DatetimeIndex(["2024-12-31"], tz="UTC")
        frame = pd.DataFrame(
            {cols[0]: [float("nan"), 1.0e9]},
            index=["TotalRevenue", "OperatingRevenue"],
        )
        stmt = FinancialStatement(
            ticker="X",
            statement_type=StatementType.INCOME,
            period=Period.ANNUAL,
            data=frame,
            provenance=provenance,
        )
        assert normalize_statement(stmt).get(LineItem.REVENUE) == pytest.approx(1.0e9)


class TestFiscalYear:
    def test_apple_ends_in_september(self, annual):
        """Apples Geschaeftsjahr endet Ende September, nicht im Dezember."""
        assert annual.income.fiscal_year_end_month == 9

    def test_quarterly_basis_yields_no_fiscal_month(self, provider):
        raw = provider.get_statements("AAPL", period=Period.QUARTERLY)
        assert infer_fiscal_year_end_month(raw.income) is None


class TestTtm:
    def test_flow_items_are_summed(self, quarterly):
        ttm = compute_ttm(quarterly.income)
        four_q = quarterly.income.data.loc["revenue"].iloc[-TTM_QUARTERS:]
        assert ttm.get(LineItem.REVENUE) == pytest.approx(float(four_q.sum()))

    def test_balance_items_are_point_in_time(self, quarterly):
        """Eine Bilanzsumme zu addieren waere fachlich falsch."""
        ttm = compute_ttm(quarterly.balance)
        latest = quarterly.balance.data.loc["total_assets"].iloc[-1]
        assert ttm.get(LineItem.TOTAL_ASSETS) == pytest.approx(float(latest))

    def test_eps_is_a_flow_and_gets_summed(self, quarterly):
        """EPS ist eine Flussgroesse: ein Quartalswert als Jahreswert waere falsch."""
        ttm = compute_ttm(quarterly.income)
        four_q = quarterly.income.data.loc["eps_diluted"].iloc[-TTM_QUARTERS:]
        assert ttm.get(LineItem.EPS_DILUTED) == pytest.approx(float(four_q.sum()))

    def test_share_count_is_not_summed(self, quarterly):
        """Aktienzahlen zu addieren wuerde die vierfache Menge ergeben."""
        ttm = compute_ttm(quarterly.income)
        latest = quarterly.income.data.loc["shares_diluted"].iloc[-1]
        assert ttm.get(LineItem.SHARES_DILUTED) == pytest.approx(float(latest))

    def test_ttm_eps_is_consistent_with_income_over_shares(self, quarterly):
        """Querprobe: TTM-EPS muss ungefaehr Gewinn je Aktie entsprechen."""
        ttm = compute_ttm_bundle(quarterly)
        eps = get_value(ttm, LineItem.EPS_DILUTED)
        net_income = get_value(ttm, LineItem.NET_INCOME)
        shares = get_value(ttm, LineItem.SHARES_DILUTED)
        assert eps is not None
        assert net_income is not None
        assert shares is not None
        assert eps == pytest.approx(net_income / shares, rel=0.05)

    def test_single_column_result(self, quarterly):
        assert len(compute_ttm(quarterly.income).data.columns) == 1

    def test_period_is_marked_ttm(self, quarterly):
        assert compute_ttm(quarterly.income).period is Period.TTM

    def test_provenance_marks_value_as_computed(self, quarterly):
        """Ein gerechneter Wert darf nicht als berichteter erscheinen."""
        assert compute_ttm(quarterly.income).provenance.source is DataSource.COMPUTED

    def test_annual_basis_is_rejected(self, annual):
        with pytest.raises(AnalyticsError, match="TTM requires a quarterly basis"):
            compute_ttm(annual.income)

    def test_too_few_quarters_raise(self, quarterly, provenance):
        short = FinancialStatement(
            ticker="AAPL",
            statement_type=StatementType.INCOME,
            period=Period.QUARTERLY,
            data=quarterly.income.data.iloc[:, -2:],
            provenance=provenance,
        )
        with pytest.raises(InsufficientDataError) as exc:
            compute_ttm(short)
        assert exc.value.required == TTM_QUARTERS
        assert exc.value.available == 2

    def test_gap_in_flow_series_yields_none_not_partial_sum(self, provenance):
        """Eine Summe mit Luecke sieht wie ein Jahr aus, deckt aber weniger ab."""
        cols = pd.DatetimeIndex(["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"], tz="UTC")
        frame = pd.DataFrame(
            {cols[0]: [1.0], cols[1]: [float("nan")], cols[2]: [3.0], cols[3]: [4.0]},
            index=["revenue"],
        )
        stmt = FinancialStatement(
            ticker="X",
            statement_type=StatementType.INCOME,
            period=Period.QUARTERLY,
            data=frame,
            provenance=provenance,
        )
        assert compute_ttm(stmt).get(LineItem.REVENUE) is None


class TestGetValue:
    def test_searches_all_three_statements(self, annual):
        assert get_value(annual, LineItem.REVENUE) is not None
        assert get_value(annual, LineItem.TOTAL_ASSETS) is not None
        assert get_value(annual, LineItem.OPERATING_CASHFLOW) is not None

    def test_missing_item_is_none(self, annual):
        assert get_value(annual, LineItem.MINORITY_INTEREST) is None

    def test_historical_period_is_reachable(self, annual):
        now = get_value(annual, LineItem.REVENUE)
        then = get_value(annual, LineItem.REVENUE, period_index=-2)
        assert now is not None
        assert then is not None
        assert now != then


class TestSignConvention:
    def test_capex_is_negative(self, annual):
        """Cashflow-Abfluesse bleiben negativ, wie vom Provider geliefert."""
        capex = get_value(annual, LineItem.CAPEX)
        assert capex is not None
        assert capex < 0

    def test_dividends_paid_is_negative(self, annual):
        dividends = get_value(annual, LineItem.DIVIDENDS_PAID)
        assert dividends is not None
        assert dividends < 0

    def test_operating_cashflow_is_positive_for_apple(self, annual):
        ocf = get_value(annual, LineItem.OPERATING_CASHFLOW)
        assert ocf is not None
        assert ocf > 0

    def test_cost_of_revenue_is_positive(self, annual):
        """GuV-Aufwaende sind positiv gespeichert."""
        cogs = get_value(annual, LineItem.COST_OF_REVENUE)
        assert cogs is not None
        assert cogs > 0


class TestMultipleTickers:
    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "JNJ"])
    def test_normalization_works_across_issuers(self, provider, ticker):
        bundle = normalize_bundle(provider.get_statements(ticker, period=Period.ANNUAL))
        assert get_value(bundle, LineItem.REVENUE) is not None
        assert get_value(bundle, LineItem.NET_INCOME) is not None
        assert get_value(bundle, LineItem.TOTAL_ASSETS) is not None
