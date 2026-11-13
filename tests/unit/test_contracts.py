"""Tests der Phase-0-Datenvertraege.

Diese Suite friert die Vertragsgarantien ein, auf die alle Subagents in den
folgenden Phasen bauen. Sie prueft vor allem die Negativfaelle: was der
Vertrag ablehnen **muss**.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from btlite.core.errors import BTLiteError, DataValidationError, RateLimitError
from btlite.models.backtest import BacktestRequest, SignalSet, StrategyConfig
from btlite.models.common import Interval, Period
from btlite.models.fundamentals import (
    FinancialStatement,
    LineItem,
    Ratio,
    RatioCategory,
    RatioSet,
    StatementType,
)
from btlite.models.market import PriceHistoryRequest, PriceSeries, Quote
from btlite.models.portfolio import PortfolioRequest, PortfolioResult, Position
from btlite.models.screening import FilterGroup, FilterPredicate, Operator


class TestTickerNormalization:
    def test_uppercases_and_strips(self, ohlcv, provenance):
        series = PriceSeries(ticker="  aapl ", adjusted=True, data=ohlcv, provenance=provenance)
        assert series.ticker == "AAPL"

    @pytest.mark.parametrize("raw", ["BRK-B", "BMW.DE", "brk.b"])
    def test_preserves_meaningful_punctuation(self, raw, ohlcv, provenance):
        series = PriceSeries(ticker=raw, adjusted=True, data=ohlcv, provenance=provenance)
        assert series.ticker == raw.strip().upper()

    def test_rejects_empty(self, ohlcv, provenance):
        with pytest.raises(ValidationError):
            PriceSeries(ticker="   ", adjusted=True, data=ohlcv, provenance=provenance)


class TestPriceSeriesContract:
    def test_accepts_valid_payload(self, ohlcv, provenance):
        series = PriceSeries(ticker="AAPL", adjusted=True, data=ohlcv, provenance=provenance)
        assert series.interval is Interval.D1
        assert series.start is not None
        assert series.end is not None
        assert series.start < series.end

    def test_rejects_naive_index(self, ohlcv, provenance):
        with pytest.raises(DataValidationError, match="tz-aware"):
            PriceSeries(
                ticker="AAPL", adjusted=True, data=ohlcv.tz_localize(None), provenance=provenance
            )

    def test_rejects_non_utc_index(self, ohlcv, provenance):
        with pytest.raises(DataValidationError, match="UTC"):
            PriceSeries(
                ticker="AAPL",
                adjusted=True,
                data=ohlcv.tz_convert("Europe/Berlin"),
                provenance=provenance,
            )

    def test_rejects_missing_column(self, ohlcv, provenance):
        with pytest.raises(DataValidationError, match="Pflichtspalten"):
            PriceSeries(
                ticker="AAPL",
                adjusted=True,
                data=ohlcv.drop(columns=["volume"]),
                provenance=provenance,
            )

    def test_rejects_unsorted_index(self, ohlcv, provenance):
        with pytest.raises(DataValidationError, match="aufsteigend"):
            PriceSeries(ticker="AAPL", adjusted=True, data=ohlcv.iloc[::-1], provenance=provenance)

    def test_rejects_duplicate_index(self, ohlcv, provenance):
        dupes = pd.concat([ohlcv, ohlcv.iloc[[0]]]).sort_index()
        with pytest.raises(DataValidationError, match="Duplikate"):
            PriceSeries(ticker="AAPL", adjusted=True, data=dupes, provenance=provenance)

    def test_rejects_non_dataframe(self, provenance):
        with pytest.raises(ValidationError):
            PriceSeries(ticker="AAPL", adjusted=True, data={"close": [1]}, provenance=provenance)

    def test_returns_drop_first_observation(self, ohlcv, provenance):
        series = PriceSeries(ticker="AAPL", adjusted=True, data=ohlcv, provenance=provenance)
        simple = series.returns()
        log = series.returns(log=True)
        assert len(simple) == len(ohlcv) - 1
        assert len(log) == len(ohlcv) - 1
        assert simple.name == "return"
        # Log- und einfache Renditen liegen bei kleinen Werten dicht beieinander.
        assert abs(float(simple.iloc[0]) - float(log.iloc[0])) < 0.01

    def test_adjusted_flag_is_required(self, ohlcv, provenance):
        with pytest.raises(ValidationError):
            PriceSeries(ticker="AAPL", data=ohlcv, provenance=provenance)


class TestQuoteContract:
    def test_change_computation(self, provenance):
        quote = Quote(
            ticker="AAPL",
            price=110.0,
            previous_close=100.0,
            timestamp=datetime.now(UTC),
            provenance=provenance,
        )
        assert quote.change_abs == pytest.approx(10.0)
        assert quote.change_pct == pytest.approx(0.10)

    def test_change_is_none_without_previous_close(self, provenance):
        quote = Quote(
            ticker="AAPL", price=110.0, timestamp=datetime.now(UTC), provenance=provenance
        )
        assert quote.change_abs is None
        assert quote.change_pct is None

    def test_rejects_non_positive_price(self, provenance):
        with pytest.raises(ValidationError):
            Quote(ticker="AAPL", price=0.0, timestamp=datetime.now(UTC), provenance=provenance)

    def test_provenance_carries_delay(self, provenance):
        quote = Quote(ticker="AAPL", price=1.0, timestamp=datetime.now(UTC), provenance=provenance)
        assert quote.provenance.delay_minutes == 15


class TestPriceHistoryRequest:
    def test_period_and_range_are_exclusive(self):
        with pytest.raises(ValidationError, match="schliessen sich aus"):
            PriceHistoryRequest(
                tickers=["AAPL"], period="1y", start=datetime(2024, 1, 1, tzinfo=UTC)
            )

    def test_requires_one_of_them(self):
        with pytest.raises(ValidationError, match="Entweder period oder"):
            PriceHistoryRequest(tickers=["AAPL"], period=None)

    def test_start_must_precede_end(self):
        with pytest.raises(ValidationError, match="start muss vor end"):
            PriceHistoryRequest(
                tickers=["AAPL"],
                period=None,
                start=datetime(2024, 6, 1, tzinfo=UTC),
                end=datetime(2024, 1, 1, tzinfo=UTC),
            )


class TestStatementContract:
    def test_line_item_lookup(self, statement_frame, provenance):
        stmt = FinancialStatement(
            ticker="AAPL",
            statement_type=StatementType.INCOME,
            period=Period.ANNUAL,
            data=statement_frame,
            provenance=provenance,
        )
        assert stmt.get(LineItem.REVENUE) == pytest.approx(3.9e11)
        assert len(stmt.periods) == 3

    def test_missing_item_is_none_not_zero(self, statement_frame, provenance):
        stmt = FinancialStatement(
            ticker="AAPL",
            statement_type=StatementType.INCOME,
            period=Period.ANNUAL,
            data=statement_frame,
            provenance=provenance,
        )
        assert stmt.get(LineItem.EBITDA) is None

    def test_nan_in_latest_period_is_none(self, statement_frame, provenance):
        """Ein NaN im juengsten Jahr darf nicht still zu 0.0 werden."""
        stmt = FinancialStatement(
            ticker="AAPL",
            statement_type=StatementType.CASHFLOW,
            period=Period.ANNUAL,
            data=statement_frame,
            provenance=provenance,
        )
        assert stmt.get(LineItem.OPERATING_CASHFLOW) is None

    def test_index_is_renamed(self, statement_frame, provenance):
        stmt = FinancialStatement(
            ticker="AAPL",
            statement_type=StatementType.BALANCE,
            period=Period.ANNUAL,
            data=statement_frame,
            provenance=provenance,
        )
        assert stmt.data.index.name == "line_item"


class TestRatioContract:
    def test_unavailable_ratio_reports_missing_inputs(self):
        ratio = Ratio(
            key="current_ratio",
            label="Current Ratio",
            category=RatioCategory.LIQUIDITY,
            value=None,
            inputs_missing=["current_liabilities"],
        )
        assert ratio.is_available is False
        assert ratio.inputs_missing == ["current_liabilities"]

    def test_set_lookup_and_grouping(self, provenance):
        ratios = {
            "current_ratio": Ratio(
                key="current_ratio",
                label="Current Ratio",
                category=RatioCategory.LIQUIDITY,
                value=1.4,
            ),
            "roe": Ratio(key="roe", label="ROE", category=RatioCategory.PROFITABILITY, value=0.31),
        }
        rs = RatioSet(ticker="AAPL", period=Period.ANNUAL, ratios=ratios, provenance=provenance)
        assert rs.value_of("roe") == pytest.approx(0.31)
        assert rs.value_of("unknown_key") is None
        assert [r.key for r in rs.by_category(RatioCategory.LIQUIDITY)] == ["current_ratio"]


class TestFilterDsl:
    def test_between_requires_both_bounds(self):
        with pytest.raises(ValidationError, match="BETWEEN"):
            FilterPredicate(field="pe", op=Operator.BETWEEN, value=5)

    def test_in_requires_values(self):
        with pytest.raises(ValidationError, match="values-Liste"):
            FilterPredicate(field="sector", op=Operator.IN)

    def test_null_operators_take_no_value(self):
        with pytest.raises(ValidationError, match="nimmt keinen"):
            FilterPredicate(field="pe", op=Operator.IS_NULL, value=1)

    def test_comparison_requires_value(self):
        with pytest.raises(ValidationError, match="benoetigt einen Vergleichswert"):
            FilterPredicate(field="pe", op=Operator.LT)

    def test_empty_group_rejected(self):
        with pytest.raises(ValidationError, match="mindestens ein Praedikat"):
            FilterGroup()

    def test_nested_group_allowed(self):
        inner = FilterGroup(predicates=[FilterPredicate(field="pe", op=Operator.LT, value=15)])
        outer = FilterGroup(groups=[inner])
        assert outer.groups[0].predicates[0].field == "pe"


class TestPortfolioContract:
    def _result(self, weights: dict[str, float], provenance, **kwargs: object) -> PortfolioResult:
        req = PortfolioRequest(tickers=list(weights), **kwargs)
        return PortfolioResult(
            request=req,
            positions=[Position(ticker=t, weight=w) for t, w in weights.items()],
            computed_at=datetime.now(UTC),
            provenance=provenance,
        )

    def test_weights_must_sum_to_one(self, provenance):
        with pytest.raises(ValidationError, match="summieren sich"):
            self._result({"AAPL": 0.7, "MSFT": 0.7}, provenance)

    def test_valid_weights_accepted(self, provenance):
        result = self._result({"AAPL": 0.6, "MSFT": 0.4}, provenance)
        assert result.weights == {"AAPL": 0.6, "MSFT": 0.4}

    def test_shorting_rejected_without_flag(self, provenance):
        with pytest.raises(ValidationError, match="Negative Gewichte"):
            self._result({"AAPL": 1.2, "MSFT": -0.2}, provenance)

    def test_shorting_allowed_with_flag(self, provenance):
        result = self._result({"AAPL": 1.2, "MSFT": -0.2}, provenance, allow_shorting=True)
        assert result.weights["MSFT"] == pytest.approx(-0.2)

    def test_efficient_return_needs_target(self):
        from btlite.models.portfolio import OptimizationObjective

        with pytest.raises(ValidationError, match="target_return"):
            PortfolioRequest(
                tickers=["AAPL", "MSFT"], objective=OptimizationObjective.EFFICIENT_RETURN
            )

    def test_universe_needs_two_tickers(self):
        with pytest.raises(ValidationError):
            PortfolioRequest(tickers=["AAPL"])


class TestBacktestContract:
    def test_signal_alignment_enforced(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="B", tz="UTC")
        entries = pd.DataFrame({"AAPL": [True] * 5}, index=idx)
        exits = pd.DataFrame({"MSFT": [False] * 5}, index=idx)
        with pytest.raises(ValidationError, match="dieselben Spalten"):
            SignalSet(tickers=["AAPL"], entries=entries, exits=exits)

    def test_execution_lag_defaults_to_one(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="B", tz="UTC")
        frame = pd.DataFrame({"AAPL": [False] * 5}, index=idx)
        signals = SignalSet(tickers=["AAPL"], entries=frame, exits=frame.copy())
        assert signals.execution_lag == 1, "Default muss Look-ahead-Bias vermeiden"

    def test_costs_are_bounded(self):
        with pytest.raises(ValidationError):
            BacktestRequest(tickers=["SPY"], strategy=StrategyConfig(key="sma_cross"), fees=1.5)


class TestErrorHierarchy:
    def test_all_errors_share_base(self):
        assert issubclass(DataValidationError, BTLiteError)
        assert issubclass(RateLimitError, BTLiteError)

    def test_user_message_is_separate_from_detail(self):
        err = DataValidationError("Kurse unvollstaendig", detail="fehlend=['volume']")
        assert err.user_message == "Kurse unvollstaendig"
        assert "volume" in str(err)

    def test_rate_limit_carries_retry_after(self):
        err = RateLimitError("Quelle drosselt", retry_after_s=30.0)
        assert err.retry_after_s == 30.0


class TestIntervalMath:
    @pytest.mark.parametrize(
        ("interval", "expected"),
        [(Interval.D1, 252), (Interval.W1, 52), (Interval.MO1, 12)],
    )
    def test_annualization_factors(self, interval, expected):
        assert interval.periods_per_year == expected

    def test_intraday_classification(self):
        assert Interval.M5.is_intraday is True
        assert Interval.D1.is_intraday is False
