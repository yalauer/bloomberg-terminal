"""Tests der Kennzahlen-Engine.

Der Schwerpunkt liegt auf zwei Zusagen des Projekts:

1. Werte stimmen. Geprueft gegen handgerechnete Referenzen.
2. Fehlende Werte sind ``None`` mit Begruendung, nie ``0.0``, nie ``NaN``,
   nie ``inf``.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from btlite.analytics.ratios import DAYS_PER_YEAR, compare_peers, compute_ratios
from btlite.analytics.statements import get_value, normalize_bundle
from btlite.data.fixtures import FixtureProvider
from btlite.models.common import Currency, DataSource, Period
from btlite.models.fundamentals import (
    FinancialStatement,
    LineItem,
    RatioCategory,
    StatementBundle,
    StatementType,
)


@pytest.fixture
def provider() -> FixtureProvider:
    return FixtureProvider()


@pytest.fixture
def bundle(provider):
    return normalize_bundle(provider.get_statements("AAPL", period=Period.ANNUAL))


@pytest.fixture
def ratios(bundle):
    return compute_ratios(bundle)


def _bundle_from(
    provenance,
    *,
    income: dict[str, float] | None = None,
    balance: dict[str, float] | None = None,
    cashflow: dict[str, float] | None = None,
) -> StatementBundle:
    """Baut ein Buendel aus kanonischen Positionen fuer exakte Rechenproben."""
    cols = pd.DatetimeIndex(["2024-12-31"], tz="UTC")

    def make(rows: dict[str, float] | None, kind: StatementType):
        if rows is None:
            return None
        frame = pd.DataFrame({cols[0]: pd.Series(rows, dtype="float64")})
        return FinancialStatement(
            ticker="TEST",
            statement_type=kind,
            period=Period.ANNUAL,
            data=frame,
            provenance=provenance,
        )

    return StatementBundle(
        ticker="TEST",
        period=Period.ANNUAL,
        income=make(income, StatementType.INCOME),
        balance=make(balance, StatementType.BALANCE),
        cashflow=make(cashflow, StatementType.CASHFLOW),
        provenance=provenance,
    )


class TestHandCalculatedReferences:
    """Exakte Rechenproben gegen von Hand gesetzte Zahlen."""

    def test_current_ratio(self, provenance):
        b = _bundle_from(
            provenance, balance={"current_assets": 300.0, "current_liabilities": 150.0}
        )
        assert compute_ratios(b).value_of("current_ratio") == pytest.approx(2.0)

    def test_quick_ratio_excludes_inventory(self, provenance):
        b = _bundle_from(
            provenance,
            balance={"current_assets": 300.0, "inventory": 100.0, "current_liabilities": 100.0},
        )
        assert compute_ratios(b).value_of("quick_ratio") == pytest.approx(2.0)

    def test_gross_margin_is_decimal_not_percent(self, provenance):
        b = _bundle_from(provenance, income={"revenue": 1000.0, "gross_profit": 400.0})
        value = compute_ratios(b).value_of("gross_margin")
        assert value == pytest.approx(0.40), "0.40 bedeutet 40 Prozent"
        assert value != pytest.approx(40.0)

    def test_roe(self, provenance):
        b = _bundle_from(provenance, income={"net_income": 250.0}, balance={"total_equity": 1000.0})
        assert compute_ratios(b).value_of("roe") == pytest.approx(0.25)

    def test_debt_to_equity(self, provenance):
        b = _bundle_from(provenance, balance={"total_debt": 750.0, "total_equity": 500.0})
        assert compute_ratios(b).value_of("debt_to_equity") == pytest.approx(1.5)

    def test_interest_coverage_uses_absolute_interest(self, provenance):
        b = _bundle_from(provenance, income={"ebit": 500.0, "interest_expense": 50.0})
        assert compute_ratios(b).value_of("interest_coverage") == pytest.approx(10.0)

    def test_net_debt_to_ebitda_subtracts_cash(self, provenance):
        b = _bundle_from(
            provenance,
            income={"ebitda": 200.0},
            balance={"total_debt": 500.0, "cash_and_equivalents": 100.0},
        )
        assert compute_ratios(b).value_of("net_debt_to_ebitda") == pytest.approx(2.0)

    def test_roic_applies_tax_rate(self, provenance):
        # Steuerquote 25 Prozent, NOPAT = 400 * 0.75 = 300, Kapital = 1000 + 500.
        b = _bundle_from(
            provenance,
            income={"ebit": 400.0, "tax_provision": 100.0, "pretax_income": 400.0},
            balance={"total_equity": 1000.0, "total_debt": 500.0},
        )
        assert compute_ratios(b).value_of("roic") == pytest.approx(300.0 / 1500.0)

    def test_dso(self, provenance):
        b = _bundle_from(provenance, income={"revenue": 3650.0}, balance={"receivables": 365.0})
        assert compute_ratios(b).value_of("dso") == pytest.approx(36.5)

    def test_ccc_combines_the_three_day_metrics(self, provenance):
        # DSO = 365/3650*365 = 36.5, DIO = 36.5, DPO = 730/3650*365 = 73.0
        # -> CCC = 36.5 + 36.5 - 73.0 = 0.0
        b = _bundle_from(
            provenance,
            income={"revenue": 3650.0, "cost_of_revenue": 3650.0},
            balance={"receivables": 365.0, "inventory": 365.0, "payables": 730.0},
        )
        rs = compute_ratios(b)
        dso, dio, dpo = rs.value_of("dso"), rs.value_of("dio"), rs.value_of("dpo")
        assert dso == pytest.approx(36.5)
        assert dio == pytest.approx(36.5)
        assert dpo == pytest.approx(73.0)
        assert rs.value_of("ccc") == pytest.approx(dso + dio - dpo)
        assert rs.value_of("ccc") == pytest.approx(0.0)

    def test_ccc_is_negative_when_suppliers_are_paid_late(self, provenance):
        """Ein negativer Zyklus ist fachlich moeglich und darf nicht geklemmt werden."""
        b = _bundle_from(
            provenance,
            income={"revenue": 3650.0, "cost_of_revenue": 3650.0},
            balance={"receivables": 365.0, "inventory": 365.0, "payables": 1460.0},
        )
        assert compute_ratios(b).value_of("ccc") == pytest.approx(36.5 + 36.5 - 146.0)

    def test_fcf_yield_adds_negative_capex(self, provenance):
        # CapEx ist negativ gespeichert -> FCF = 1000 + (-300) = 700.
        b = _bundle_from(
            provenance, cashflow={"operating_cashflow": 1000.0, "capital_expenditure": -300.0}
        )
        rs = compute_ratios(b, market_cap=7000.0)
        assert rs.value_of("fcf_yield") == pytest.approx(0.10)

    def test_pe(self, provenance):
        b = _bundle_from(provenance, income={"eps_diluted": 5.0})
        assert compute_ratios(b, price=100.0).value_of("pe") == pytest.approx(20.0)

    def test_ev_to_ebitda(self, provenance):
        # EV = 1000 + 200 - 100 = 1100.
        b = _bundle_from(
            provenance,
            income={"ebitda": 110.0},
            balance={"total_debt": 200.0, "cash_and_equivalents": 100.0},
        )
        rs = compute_ratios(b, market_cap=1000.0)
        assert rs.value_of("ev_to_ebitda") == pytest.approx(10.0)

    def test_revenue_growth(self, provenance):
        cols = pd.DatetimeIndex(["2023-12-31", "2024-12-31"], tz="UTC")
        frame = pd.DataFrame({cols[0]: [1000.0], cols[1]: [1200.0]}, index=["revenue"])
        income = FinancialStatement(
            ticker="TEST",
            statement_type=StatementType.INCOME,
            period=Period.ANNUAL,
            data=frame,
            provenance=provenance,
        )
        b = StatementBundle(
            ticker="TEST", period=Period.ANNUAL, income=income, provenance=provenance
        )
        assert compute_ratios(b).value_of("revenue_growth_yoy") == pytest.approx(0.20)

    def test_revenue_cagr_over_three_years(self, provenance):
        cols = pd.DatetimeIndex(["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"], tz="UTC")
        # Verdopplung in drei Jahren -> CAGR = 2^(1/3) - 1.
        frame = pd.DataFrame(
            {cols[0]: [1000.0], cols[1]: [1100.0], cols[2]: [1500.0], cols[3]: [2000.0]},
            index=["revenue"],
        )
        income = FinancialStatement(
            ticker="TEST",
            statement_type=StatementType.INCOME,
            period=Period.ANNUAL,
            data=frame,
            provenance=provenance,
        )
        b = StatementBundle(
            ticker="TEST", period=Period.ANNUAL, income=income, provenance=provenance
        )
        expected = 2.0 ** (1.0 / 3.0) - 1.0
        assert compute_ratios(b).value_of("revenue_cagr_3y") == pytest.approx(expected)


class TestMissingValueDiscipline:
    """Die Kernzusage: nie stille Nullwerte, nie NaN, nie inf."""

    def test_missing_input_yields_none_not_zero(self, provenance):
        b = _bundle_from(provenance, balance={"current_assets": 300.0})
        ratio = compute_ratios(b).ratios["current_ratio"]
        assert ratio.value is None
        assert ratio.value != 0.0
        assert "current_liabilities" in ratio.inputs_missing

    def test_zero_denominator_yields_none_not_inf(self, provenance):
        b = _bundle_from(provenance, balance={"current_assets": 300.0, "current_liabilities": 0.0})
        ratio = compute_ratios(b).ratios["current_ratio"]
        assert ratio.value is None
        assert any("current_liabilities is 0" in m for m in ratio.inputs_missing)

    def test_no_ratio_is_ever_nan_or_inf(self, ratios):
        for key, ratio in ratios.ratios.items():
            if ratio.value is not None:
                assert math.isfinite(ratio.value), f"{key} ist nicht endlich"

    def test_unavailable_ratio_names_its_missing_inputs(self, provenance):
        b = _bundle_from(provenance, income={"revenue": 100.0})
        for ratio in compute_ratios(b).ratios.values():
            if ratio.value is None:
                assert ratio.inputs_missing, f"{ratio.key} nennt keinen Grund"

    def test_valuation_needs_injected_price(self, bundle):
        """Ohne Preis bleiben KGV und Dividendenrendite None, nicht 0."""
        without = compute_ratios(bundle)
        assert without.value_of("pe") is None
        assert without.value_of("dividend_yield") is None

    def test_valuation_appears_once_price_is_given(self, bundle):
        with_price = compute_ratios(bundle, price=250.0, market_cap=3.7e12)
        assert with_price.value_of("pe") is not None
        assert with_price.value_of("ps") is not None

    def test_growth_needs_two_periods(self, provenance):
        b = _bundle_from(provenance, income={"revenue": 1000.0})
        ratio = compute_ratios(b).ratios["revenue_growth_yoy"]
        assert ratio.value is None
        assert any("periods" in m for m in ratio.inputs_missing)

    def test_negative_base_makes_growth_uninterpretable(self, provenance):
        cols = pd.DatetimeIndex(["2023-12-31", "2024-12-31"], tz="UTC")
        frame = pd.DataFrame({cols[0]: [-100.0], cols[1]: [50.0]}, index=["net_income"])
        income = FinancialStatement(
            ticker="TEST",
            statement_type=StatementType.INCOME,
            period=Period.ANNUAL,
            data=frame,
            provenance=provenance,
        )
        b = StatementBundle(
            ticker="TEST", period=Period.ANNUAL, income=income, provenance=provenance
        )
        ratio = compute_ratios(b).ratios["earnings_growth_yoy"]
        assert ratio.value is None
        assert any("negativ" in m for m in ratio.inputs_missing)


class TestMetadata:
    def test_every_ratio_has_a_formula(self, ratios):
        for ratio in ratios.ratios.values():
            assert ratio.formula, f"{ratio.key} hat keine Formel"

    def test_every_category_is_populated(self, ratios):
        for category in RatioCategory:
            assert ratios.by_category(category), f"{category} ist leer"

    def test_units_are_declared(self, ratios):
        allowed = {"ratio", "percent", "days", "x", "currency"}
        for ratio in ratios.ratios.values():
            assert ratio.unit in allowed

    def test_margins_are_declared_as_percent(self, ratios):
        for key in ("gross_margin", "operating_margin", "net_margin", "roe", "roa"):
            assert ratios.ratios[key].unit == "percent"

    def test_day_metrics_are_declared_as_days(self, ratios):
        for key in ("dso", "dio", "dpo", "ccc"):
            assert ratios.ratios[key].unit == "days"

    def test_provenance_marks_values_as_computed(self, ratios):
        assert ratios.provenance.source is DataSource.COMPUTED

    def test_period_end_is_set(self, ratios):
        assert ratios.period_end is not None
        assert ratios.period_end.month == 9, "AAPL schliesst im September ab"


class TestRealData:
    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "JNJ"])
    def test_most_ratios_are_computable_from_real_statements(self, provider, ticker):
        bundle = normalize_bundle(provider.get_statements(ticker, period=Period.ANNUAL))
        instrument = provider.get_instrument(ticker)
        quote = provider.get_quotes([ticker])[ticker]
        rs = compute_ratios(bundle, price=quote.price, market_cap=instrument.market_cap)
        available = sum(1 for r in rs.ratios.values() if r.is_available)
        assert available >= len(rs.ratios) * 0.8, f"{ticker}: nur {available} Kennzahlen"

    def test_margins_are_plausible(self, provider):
        bundle = normalize_bundle(provider.get_statements("AAPL", period=Period.ANNUAL))
        rs = compute_ratios(bundle)
        assert 0.2 < rs.value_of("gross_margin") < 0.8
        assert 0.0 < rs.value_of("net_margin") < 0.5

    def test_ratios_match_manual_computation(self, provider):
        """Querprobe der Engine gegen direkt aus dem Buendel gelesene Werte."""
        bundle = normalize_bundle(provider.get_statements("AAPL", period=Period.ANNUAL))
        rs = compute_ratios(bundle)
        net_income = get_value(bundle, LineItem.NET_INCOME)
        equity = get_value(bundle, LineItem.TOTAL_EQUITY)
        revenue = get_value(bundle, LineItem.REVENUE)
        gross = get_value(bundle, LineItem.GROSS_PROFIT)
        assert rs.value_of("roe") == pytest.approx(net_income / equity)
        assert rs.value_of("gross_margin") == pytest.approx(gross / revenue)


class TestPeerComparison:
    def test_matrix_shape(self, provider):
        sets = [
            compute_ratios(normalize_bundle(provider.get_statements(t, period=Period.ANNUAL)))
            for t in ("AAPL", "MSFT", "JNJ")
        ]
        comparison, _ = compare_peers(sets[0], sets[1:], ratio_keys=["roe", "net_margin"])
        assert list(comparison.data.index) == ["AAPL", "MSFT", "JNJ"]
        assert list(comparison.data.columns) == ["roe", "net_margin"]

    def test_percentile_of_subject(self, provider):
        sets = [
            compute_ratios(normalize_bundle(provider.get_statements(t, period=Period.ANNUAL)))
            for t in ("AAPL", "MSFT", "JNJ")
        ]
        comparison, _ = compare_peers(sets[0], sets[1:], ratio_keys=["roe"])
        rank = comparison.percentile_of_subject("roe")
        assert rank is not None
        assert 0.0 <= rank <= 1.0

    def test_differing_fiscal_years_are_reported_not_hidden(self, provider):
        """AAPL schliesst im September, MSFT im Juni - das muss auffallen."""
        sets = [
            compute_ratios(normalize_bundle(provider.get_statements(t, period=Period.ANNUAL)))
            for t in ("AAPL", "MSFT")
        ]
        _, warnings = compare_peers(sets[0], sets[1:], ratio_keys=["roe"])
        assert any("Differing fiscal year ends" in w for w in warnings)

    def test_differing_currencies_are_reported(self, provider, provenance):
        base = compute_ratios(
            normalize_bundle(provider.get_statements("AAPL", period=Period.ANNUAL))
        )
        foreign_bundle = _bundle_from(provenance, income={"revenue": 100.0, "net_income": 10.0})
        foreign = compute_ratios(foreign_bundle)
        foreign.currency = Currency.EUR
        _, warnings = compare_peers(base, [foreign], ratio_keys=["net_margin"])
        assert any("Different reporting currencies" in w for w in warnings)

    def test_missing_values_stay_nan_in_the_matrix(self, provider, provenance):
        base = compute_ratios(
            normalize_bundle(provider.get_statements("AAPL", period=Period.ANNUAL))
        )
        sparse = compute_ratios(_bundle_from(provenance, income={"revenue": 100.0}))
        comparison, _ = compare_peers(base, [sparse], ratio_keys=["roe"])
        assert pd.isna(comparison.data.loc["TEST", "roe"])


class TestConstants:
    def test_days_per_year_is_calendar_based(self):
        assert DAYS_PER_YEAR == 365
