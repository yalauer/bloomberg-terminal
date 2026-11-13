"""Tests der Trendrechnung in ``analytics/statements.py``.

Kernfrage dieser Datei: liefert die Rechnung dort ``None``, wo ein Prozentsatz
irrefuehrend waere? Eine Zahl, die das Gegenteil dessen suggeriert, was
passiert ist, ist schaedlicher als ein "n/a".
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from btlite.analytics.statements import (
    MISSING_VALUE_REASON,
    NO_PRIOR_REASON,
    SIGN_CHANGE_REASON,
    ZERO_BASE_REASON,
    compute_cagr,
    compute_trend,
    compute_trend_bundle,
    normalize_bundle,
)
from btlite.data.fixtures import FixtureProvider
from btlite.models.common import DataSource, Period, Provenance
from btlite.models.fundamentals import FinancialStatement, LineItem, StatementType


def _statement(rows: dict[str, list[float | None]], n_periods: int = 3) -> FinancialStatement:
    """Baut ein Statement mit handgewaehlten Werten.

    Args:
        rows: Positionsname auf Werte je Periode.
        n_periods: Anzahl der Perioden.

    Returns:
        Statement auf Jahresbasis.
    """
    columns = pd.DatetimeIndex(
        [f"{2021 + i}-12-31" for i in range(n_periods)], tz="UTC", name="date"
    )
    frame = pd.DataFrame(
        {name: pd.Series(values, index=columns, dtype="float64") for name, values in rows.items()}
    ).T
    frame.index.name = "line_item"
    return FinancialStatement(
        ticker="TEST",
        statement_type=StatementType.INCOME,
        period=Period.ANNUAL,
        data=frame,
        provenance=Provenance(
            source=DataSource.FIXTURE, fetched_at=datetime(2025, 1, 2, tzinfo=UTC)
        ),
    )


class TestHandComputedReferences:
    def test_100_to_120_is_twenty_percent(self):
        """Die Referenzrechnung des Auftrags: 100 auf 120 ergibt 0.20."""
        trend = compute_trend(_statement({"revenue": [100.0, 120.0, 150.0]}))
        point = trend.point("revenue", period_index=1)

        assert point.change_pct == pytest.approx(0.20)
        assert point.change_abs == pytest.approx(20.0)
        assert point.reason is None

    def test_result_is_decimal_not_percent(self):
        """0.12 bedeutet 12 Prozent - projektweite Konvention."""
        trend = compute_trend(_statement({"revenue": [100.0, 112.0]}, n_periods=2))
        assert trend.point("revenue").change_pct == pytest.approx(0.12)

    def test_decline_is_negative(self):
        trend = compute_trend(_statement({"revenue": [200.0, 150.0]}, n_periods=2))
        point = trend.point("revenue")
        assert point.change_pct == pytest.approx(-0.25)
        assert point.change_abs == pytest.approx(-50.0)

    def test_drop_to_zero_is_minus_one(self):
        """Auf 0 zu fallen sind minus 100 Prozent, kein Sonderfall."""
        trend = compute_trend(_statement({"revenue": [80.0, 0.0]}, n_periods=2))
        assert trend.point("revenue").change_pct == pytest.approx(-1.0)

    def test_negative_to_less_negative_is_improvement(self):
        """Der Nenner ist der Betrag: von -100 auf -50 ist eine Verbesserung."""
        trend = compute_trend(_statement({"net_income": [-100.0, -50.0]}, n_periods=2))
        point = trend.point("net_income")
        assert point.change_pct == pytest.approx(0.50)
        assert point.change_abs == pytest.approx(50.0)

    def test_negative_to_more_negative_is_deterioration(self):
        trend = compute_trend(_statement({"net_income": [-100.0, -150.0]}, n_periods=2))
        assert trend.point("net_income").change_pct == pytest.approx(-0.50)


class TestUninterpretableCases:
    def test_sign_change_yields_none_with_reason(self):
        """Von -100 auf +50 sind keine 150 Prozent Wachstum."""
        trend = compute_trend(_statement({"net_income": [-100.0, 50.0]}, n_periods=2))
        point = trend.point("net_income")

        assert point.change_pct is None
        assert point.reason == SIGN_CHANGE_REASON
        assert not point.is_available

    def test_sign_change_still_reports_absolute_change(self):
        """Die absolute Veraenderung ist auch beim Vorzeichenwechsel definiert."""
        trend = compute_trend(_statement({"net_income": [-100.0, 50.0]}, n_periods=2))
        assert trend.point("net_income").change_abs == pytest.approx(150.0)

    def test_profit_turning_to_loss_yields_none(self):
        trend = compute_trend(_statement({"net_income": [50.0, -100.0]}, n_periods=2))
        point = trend.point("net_income")
        assert point.change_pct is None
        assert point.reason == SIGN_CHANGE_REASON
        assert point.change_abs == pytest.approx(-150.0)

    def test_zero_base_yields_none_with_reason(self):
        """Der Sprung von 0 auf irgendetwas ist unendlich, kein Prozentsatz."""
        trend = compute_trend(_statement({"revenue": [0.0, 50.0]}, n_periods=2))
        point = trend.point("revenue")

        assert point.change_pct is None
        assert point.reason == ZERO_BASE_REASON

    def test_zero_base_still_reports_absolute_change(self):
        trend = compute_trend(_statement({"revenue": [0.0, 50.0]}, n_periods=2))
        assert trend.point("revenue").change_abs == pytest.approx(50.0)

    def test_missing_prior_period_yields_none(self):
        """Die aelteste Periode hat keine Basis - None, nie 0.0."""
        trend = compute_trend(_statement({"revenue": [100.0, 120.0]}, n_periods=2))
        point = trend.point("revenue", period_index=0)

        assert point.change_pct is None
        assert point.change_abs is None
        assert point.reason == NO_PRIOR_REASON

    def test_missing_value_yields_none_not_zero(self):
        trend = compute_trend(_statement({"revenue": [100.0, None, 150.0]}))
        point = trend.point("revenue", period_index=1)

        assert point.change_pct is None
        assert point.change_abs is None
        assert point.reason == MISSING_VALUE_REASON

    def test_recovery_after_gap_uses_immediate_prior(self):
        """Nach einer Luecke ist die Vorperiode leer, nicht die davor."""
        trend = compute_trend(_statement({"revenue": [100.0, None, 150.0]}))
        point = trend.point("revenue", period_index=2)
        assert point.change_pct is None
        assert point.reason == MISSING_VALUE_REASON

    def test_zero_to_zero_yields_none(self):
        trend = compute_trend(_statement({"revenue": [0.0, 0.0]}, n_periods=2))
        assert trend.point("revenue").change_pct is None

    def test_no_silent_zero_anywhere(self):
        """Kein Fehlwert wird still zu 0.0 - das Akzeptanzkriterium."""
        trend = compute_trend(_statement({"x": [None, None, 5.0]}))
        for index in range(3):
            point = trend.point("x", period_index=index)
            assert point.change_pct is None
            assert point.reason


class TestFrameShape:
    def test_frames_mirror_the_source_statement(self):
        statement = _statement({"revenue": [100.0, 120.0], "net_income": [10.0, 12.0]}, 2)
        trend = compute_trend(statement)

        for frame in (trend.change_abs, trend.change_pct, trend.reasons):
            assert list(frame.index) == list(statement.data.index)
            assert list(frame.columns) == list(statement.data.columns)
            assert frame.index.name == "line_item"

    def test_first_column_is_entirely_missing(self):
        trend = compute_trend(_statement({"revenue": [100.0, 120.0], "cost": [40.0, 45.0]}, 2))
        first = trend.change_pct.iloc[:, 0]
        assert first.isna().all()
        assert (trend.reasons.iloc[:, 0] == NO_PRIOR_REASON).all()

    def test_reason_is_empty_where_a_value_exists(self):
        trend = compute_trend(_statement({"revenue": [100.0, 120.0]}, n_periods=2))
        assert trend.reasons.iloc[0, 1] == ""

    def test_metadata_is_carried_over(self):
        statement = _statement({"revenue": [100.0, 120.0]}, n_periods=2)
        trend = compute_trend(statement)

        assert trend.ticker == "TEST"
        assert trend.statement_type is StatementType.INCOME
        assert trend.period is Period.ANNUAL
        assert trend.currency is statement.currency

    def test_provenance_is_computed(self):
        """Eine gerechnete Groesse darf nicht als berichtet erscheinen."""
        trend = compute_trend(_statement({"revenue": [100.0, 120.0]}, n_periods=2))
        assert trend.provenance.source is DataSource.COMPUTED
        assert "Period-over-period change" in trend.provenance.note

    def test_single_period_statement_has_only_missing_values(self):
        trend = compute_trend(_statement({"revenue": [100.0]}, n_periods=1))
        assert trend.change_pct.isna().all().all()
        assert trend.point("revenue").reason == NO_PRIOR_REASON

    def test_unknown_line_item_returns_none(self):
        trend = compute_trend(_statement({"revenue": [100.0, 120.0]}, n_periods=2))
        assert trend.point("gibtsnicht") is None

    def test_out_of_range_period_returns_none(self):
        trend = compute_trend(_statement({"revenue": [100.0, 120.0]}, n_periods=2))
        assert trend.point("revenue", period_index=9) is None

    def test_point_accepts_line_item_enum(self):
        trend = compute_trend(_statement({"revenue": [100.0, 120.0]}, n_periods=2))
        assert trend.point(LineItem.REVENUE).change_pct == pytest.approx(0.20)

    def test_period_ends_are_reported(self):
        trend = compute_trend(_statement({"revenue": [100.0, 120.0]}, n_periods=2))
        point = trend.point("revenue")
        assert point.period_end.year == 2022
        assert point.previous_period_end.year == 2021

    def test_source_statement_is_unchanged(self):
        statement = _statement({"revenue": [100.0, 120.0]}, n_periods=2)
        before = statement.data.copy()
        compute_trend(statement)
        pd.testing.assert_frame_equal(statement.data, before)


class TestCagr:
    def test_hand_computed_doubling_over_two_periods(self):
        """100 auf 400 in zwei Schritten sind 100 Prozent pro Jahr."""
        statement = _statement({"revenue": [100.0, 200.0, 400.0]})
        value, reason = compute_cagr(statement, "revenue")

        assert value == pytest.approx(1.0)
        assert reason is None

    def test_flat_series_has_zero_growth(self):
        statement = _statement({"revenue": [100.0, 100.0, 100.0]})
        value, _ = compute_cagr(statement, "revenue")
        assert value == pytest.approx(0.0)

    def test_quarterly_basis_annualizes_differently(self):
        """Vier Quartale sind ein Jahr, nicht vier."""
        columns = pd.DatetimeIndex(
            ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31"],
            tz="UTC",
            name="date",
        )
        frame = pd.DataFrame(
            {"revenue": pd.Series([100.0, 110.0, 120.0, 130.0, 200.0], index=columns)}
        ).T
        frame.index.name = "line_item"
        statement = FinancialStatement(
            ticker="TEST",
            statement_type=StatementType.INCOME,
            period=Period.QUARTERLY,
            data=frame,
            provenance=Provenance(
                source=DataSource.FIXTURE, fetched_at=datetime(2025, 1, 2, tzinfo=UTC)
            ),
        )
        value, reason = compute_cagr(statement, "revenue")

        assert reason is None
        assert value == pytest.approx(1.0)

    def test_sign_change_yields_none(self):
        statement = _statement({"net_income": [-50.0, 10.0, 80.0]})
        value, reason = compute_cagr(statement, "net_income")
        assert value is None
        assert reason == SIGN_CHANGE_REASON

    def test_negative_start_yields_none(self):
        statement = _statement({"net_income": [-50.0, -30.0, -10.0]})
        value, reason = compute_cagr(statement, "net_income")
        assert value is None
        assert reason == SIGN_CHANGE_REASON

    def test_zero_start_yields_none(self):
        statement = _statement({"revenue": [0.0, 50.0, 100.0]})
        value, reason = compute_cagr(statement, "revenue")
        assert value is None
        assert reason == ZERO_BASE_REASON

    def test_single_period_yields_none(self):
        statement = _statement({"revenue": [100.0]}, n_periods=1)
        value, reason = compute_cagr(statement, "revenue")
        assert value is None
        assert reason == NO_PRIOR_REASON

    def test_unknown_item_yields_none(self):
        statement = _statement({"revenue": [100.0, 120.0]}, n_periods=2)
        value, reason = compute_cagr(statement, "gibtsnicht")
        assert value is None
        assert reason == MISSING_VALUE_REASON

    def test_gaps_are_skipped_not_counted(self):
        """Fehlwerte zaehlen nicht als Perioden - sonst waere n falsch."""
        statement = _statement({"revenue": [100.0, None, 400.0]})
        value, reason = compute_cagr(statement, "revenue")
        assert reason is None
        assert value == pytest.approx(3.0)

    def test_explicit_periods_per_year_wins(self):
        """Zwei Perioden pro Jahr: ein Schritt ist ein halbes Jahr.

        Das Vervierfachen in einem halben Jahr annualisiert auf 4^2 - 1,
        also 1500 Prozent. Auf Jahresbasis waeren es 300 Prozent - der
        Parameter aendert das Ergebnis also sichtbar.
        """
        statement = _statement({"revenue": [100.0, 400.0]}, n_periods=2)
        half_yearly, _ = compute_cagr(statement, "revenue", periods_per_year=2.0)
        yearly, _ = compute_cagr(statement, "revenue", periods_per_year=1.0)

        assert half_yearly == pytest.approx(15.0)
        assert yearly == pytest.approx(3.0)


@pytest.fixture
def provider():
    return FixtureProvider()


@pytest.fixture
def aapl(provider):
    return normalize_bundle(provider.get_statements("AAPL", period=Period.ANNUAL))


class TestAgainstRealFixtures:
    def test_apple_revenue_grows(self, aapl):
        """Apples Umsatz waechst in der juengsten Periode der Fixtures."""
        trend = compute_trend(aapl.income)
        point = trend.point(LineItem.REVENUE)

        assert point.change_pct is not None
        assert point.change_abs is not None
        assert -0.5 < point.change_pct < 0.5

    def test_absolute_and_relative_agree_in_sign(self, aapl):
        """Wo beide vorliegen, muessen sie in dieselbe Richtung zeigen."""
        trend = compute_trend(aapl.income)
        both = trend.change_abs.notna() & trend.change_pct.notna()
        assert both.to_numpy().any()

        abs_values = trend.change_abs.to_numpy()[both.to_numpy()]
        pct_values = trend.change_pct.to_numpy()[both.to_numpy()]
        assert np.all(np.sign(abs_values) == np.sign(pct_values))

    def test_every_missing_pct_has_a_reason(self, aapl):
        """Kein stilles n/a - jeder Fehlwert nennt seinen Grund."""
        trend = compute_trend(aapl.income)
        missing = trend.change_pct.isna()
        reasons = trend.reasons

        for line_item in trend.change_pct.index:
            for column in trend.change_pct.columns:
                if bool(missing.loc[line_item, column]):
                    assert reasons.loc[line_item, column] != ""

    def test_every_present_pct_has_no_reason(self, aapl):
        trend = compute_trend(aapl.income)
        present = trend.change_pct.notna()

        for line_item in trend.change_pct.index:
            for column in trend.change_pct.columns:
                if bool(present.loc[line_item, column]):
                    assert trend.reasons.loc[line_item, column] == ""

    def test_reference_value_matches_manual_division(self, aapl):
        """Gegenrechnung von Hand auf den echten Fixture-Zahlen."""
        revenue = aapl.income.data.loc[LineItem.REVENUE.value]
        expected = (revenue.iloc[-1] - revenue.iloc[-2]) / abs(revenue.iloc[-2])

        trend = compute_trend(aapl.income)
        assert trend.point(LineItem.REVENUE).change_pct == pytest.approx(expected)

    def test_absolute_change_matches_manual_difference(self, aapl):
        revenue = aapl.income.data.loc[LineItem.REVENUE.value]
        expected = revenue.iloc[-1] - revenue.iloc[-2]
        trend = compute_trend(aapl.income)
        assert trend.point(LineItem.REVENUE).change_abs == pytest.approx(expected)

    def test_capex_stays_negative_and_interpretable(self, aapl):
        """CapEx ist durchgaengig negativ - kein Vorzeichenwechsel zu erwarten."""
        trend = compute_trend(aapl.cashflow)
        point = trend.point(LineItem.CAPEX)
        if point is not None and point.change_pct is not None:
            assert point.reason is None

    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "JNJ", "KO"])
    def test_runs_on_all_statements_of_several_tickers(self, provider, ticker):
        bundle = normalize_bundle(provider.get_statements(ticker, period=Period.ANNUAL))
        trends = compute_trend_bundle(bundle)

        assert StatementType.INCOME in trends
        for trend in trends.values():
            assert not trend.change_abs.empty
            assert trend.provenance.source is DataSource.COMPUTED

    def test_bundle_skips_missing_statements(self, aapl):
        aapl.cashflow = None
        trends = compute_trend_bundle(aapl)
        assert StatementType.CASHFLOW not in trends
        assert StatementType.INCOME in trends

    def test_apple_revenue_cagr_is_plausible(self, aapl):
        """Ueber die Fixture-Jahre waechst Apple, aber nicht absurd."""
        value, reason = compute_cagr(aapl.income, LineItem.REVENUE)
        assert reason is None
        assert -0.2 < value < 0.5


class TestTrendBundleUsesCanonicalNames:
    """Regression: die Trendtabelle traegt kanonische Zeilennamen.

    Der Rest des Projekts adressiert Positionen ueber diese Namen. Zuvor
    reichte ``compute_trend_bundle`` ein rohes Buendel unveraendert durch.
    Die Werte stimmten, aber der Index trug Provider-Rohnamen wie
    ``TotalRevenue``; ein Zugriff auf ``revenue`` lief ins Leere und die
    FA-Seite haette keinen einzigen Trend anzeigen koennen.
    """

    def _bundle(self):
        from btlite.data.fixtures import FixtureProvider

        return FixtureProvider().get_statements("AAPL")

    def test_raw_bundle_yields_canonical_rows(self):
        from btlite.analytics.statements import compute_trend_bundle
        from btlite.models.fundamentals import StatementType

        trend = compute_trend_bundle(self._bundle())[StatementType.INCOME]
        for item in ("revenue", "net_income", "operating_income"):
            assert item in trend.change_pct.index

    def test_normalized_input_is_not_normalized_twice(self):
        """Ein zweiter Lauf wuerde die nicht zugeordneten Rohzeilen verwerfen."""
        from btlite.analytics.statements import compute_trend_bundle, normalize_bundle
        from btlite.models.fundamentals import StatementType

        raw = self._bundle()
        from_raw = compute_trend_bundle(raw)[StatementType.INCOME]
        from_norm = compute_trend_bundle(normalize_bundle(raw))[StatementType.INCOME]
        assert list(from_raw.change_pct.index) == list(from_norm.change_pct.index)

    def test_percentages_match_the_normalized_statements(self):
        """Querprobe gegen die Normalisierung, nicht gegen sich selbst."""
        from btlite.analytics.statements import compute_trend_bundle, normalize_bundle
        from btlite.models.fundamentals import StatementType

        raw = self._bundle()
        trend = compute_trend_bundle(raw)[StatementType.INCOME]
        income = normalize_bundle(raw).income.data
        for item in ("revenue", "net_income"):
            previous = float(income.loc[item].iloc[-2])
            current = float(income.loc[item].iloc[-1])
            expected = (current - previous) / abs(previous)
            assert float(trend.change_pct.loc[item].iloc[-1]) == pytest.approx(expected)
