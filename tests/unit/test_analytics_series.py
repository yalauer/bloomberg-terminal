"""Tests der Mehr-Reihen-Umformungen."""

from __future__ import annotations

import pandas as pd
import pytest

from btlite.analytics.series import (
    COMPARISON_BASE,
    MIN_COMMON_OBSERVATIONS,
    align_closes,
    build_comparison,
    normalize_to_base,
    portfolio_returns,
)
from btlite.core.errors import InsufficientDataError
from btlite.data.fixtures import FixtureProvider
from btlite.models.common import DataSource
from btlite.models.market import PriceSeries

TICKERS = ["AAPL", "MSFT", "NVDA"]


@pytest.fixture
def provider():
    return FixtureProvider()


@pytest.fixture
def prices(provider):
    return provider.get_histories(TICKERS)


def _series(values: list[float], *, ticker: str, provenance, start: str = "2024-01-01"):
    """Baut eine minimale Kursreihe aus Schlusskursen."""
    idx = pd.date_range(start, periods=len(values), freq="B", tz="UTC", name="date")
    frame = pd.DataFrame(
        {
            "open": values,
            "high": [v + 1 for v in values],
            "low": [v - 1 for v in values],
            "close": values,
            "volume": [1e6] * len(values),
        },
        index=idx,
    )
    return PriceSeries(ticker=ticker, adjusted=True, data=frame, provenance=provenance)


class TestAlignCloses:
    def test_intersection_not_union(self, provenance):
        """Ein spaet startender Ticker darf nicht rueckwirkend erfunden werden."""
        a = _series([10.0] * 10, ticker="A", provenance=provenance, start="2024-01-01")
        b = _series([20.0] * 5, ticker="B", provenance=provenance, start="2024-01-08")
        aligned = align_closes({"A": a, "B": b})
        assert len(aligned) == 5, "Nur der gemeinsame Zeitraum zaehlt"
        assert not aligned.isna().to_numpy().any()

    def test_columns_match_tickers(self, prices):
        assert list(align_closes(prices).columns) == TICKERS

    def test_index_is_utc_and_named(self, prices):
        aligned = align_closes(prices)
        assert str(aligned.index.tz) == "UTC"
        assert aligned.index.name == "date"

    def test_empty_input_raises(self):
        with pytest.raises(InsufficientDataError):
            align_closes({})

    def test_disjoint_series_raise(self, provenance):
        a = _series([10.0] * 5, ticker="A", provenance=provenance, start="2024-01-01")
        b = _series([20.0] * 5, ticker="B", provenance=provenance, start="2025-01-01")
        with pytest.raises(InsufficientDataError) as exc:
            align_closes({"A": a, "B": b})
        assert exc.value.required == MIN_COMMON_OBSERVATIONS


class TestNormalizeToBase:
    def test_every_column_starts_at_base(self, prices):
        result = normalize_to_base(align_closes(prices))
        for ticker in TICKERS:
            assert result[ticker].iloc[0] == pytest.approx(COMPARISON_BASE)

    def test_relative_moves_are_preserved(self, provenance):
        # Verdopplung muss zu 200 werden, Halbierung zu 50.
        a = _series([10.0, 20.0], ticker="A", provenance=provenance)
        b = _series([80.0, 40.0], ticker="B", provenance=provenance)
        result = normalize_to_base(align_closes({"A": a, "B": b}))
        assert result["A"].iloc[-1] == pytest.approx(200.0)
        assert result["B"].iloc[-1] == pytest.approx(50.0)

    def test_zero_start_stays_undefined_not_zero(self, provenance):
        """Division durch Null ist undefiniert, nicht null."""
        a = _series([0.0, 5.0, 7.0], ticker="A", provenance=provenance)
        result = normalize_to_base(align_closes({"A": a}))
        assert result["A"].isna().all()

    def test_custom_base(self, prices):
        result = normalize_to_base(align_closes(prices), base=1.0)
        assert result[TICKERS[0]].iloc[0] == pytest.approx(1.0)


class TestBuildComparison:
    def test_fulfils_the_contract(self, prices):
        comparison = build_comparison(prices)
        assert comparison.tickers == TICKERS
        assert list(comparison.data.columns) == TICKERS
        assert comparison.normalized is True

    def test_all_lines_start_together(self, prices):
        data = build_comparison(prices).data
        assert data.iloc[0].to_numpy() == pytest.approx([COMPARISON_BASE] * len(TICKERS))

    def test_raw_mode_keeps_price_levels(self, prices):
        comparison = build_comparison(prices, normalized=False)
        assert comparison.normalized is False
        first = comparison.data["AAPL"].iloc[0]
        assert first != pytest.approx(COMPARISON_BASE)

    def test_viz_layer_can_draw_it(self, prices):
        """Querprobe gegen die Chart-Factory, die das Objekt konsumiert."""
        from btlite.viz.price import comparison as draw

        figure = draw(build_comparison(prices))
        assert len(figure.data) == len(TICKERS)

    def test_provenance_marks_result_as_computed(self, prices):
        assert build_comparison(prices).provenance.source is DataSource.COMPUTED

    def test_delay_is_the_worst_of_its_inputs(self, prices):
        """Die Matrix ist nur so aktuell wie ihr aeltester Bestandteil."""
        merged = build_comparison(prices).provenance
        worst = max(series.provenance.delay_minutes or 0 for series in prices.values())
        assert merged.delay_minutes == worst

    def test_single_ticker_is_allowed(self, provider):
        comparison = build_comparison(provider.get_histories(["AAPL"]))
        assert comparison.tickers == ["AAPL"]


class TestPortfolioReturns:
    def test_matches_manual_weighted_sum(self, prices):
        weights = {"AAPL": 0.4, "MSFT": 0.4, "NVDA": 0.2}
        result = portfolio_returns(weights, prices, rebalanced=True)
        aligned = align_closes(prices)
        expected = (aligned.pct_change().dropna() * pd.Series(weights)).sum(axis=1)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_single_asset_equals_its_own_returns(self, prices):
        result = portfolio_returns({"AAPL": 1.0}, prices)
        expected = prices["AAPL"].close.pct_change().dropna()
        assert result.iloc[-1] == pytest.approx(float(expected.iloc[-1]))

    def test_weights_are_normalized_to_their_sum(self, prices):
        half = portfolio_returns({"AAPL": 0.5, "MSFT": 0.5}, prices)
        doubled = portfolio_returns({"AAPL": 1.0, "MSFT": 1.0}, prices)
        pd.testing.assert_series_equal(half, doubled)

    def test_rebalancing_changes_the_outcome(self, prices):
        """Ohne Zurueckgewichten laufen Gewinner weiter - das muss sich zeigen."""
        weights = {"AAPL": 0.4, "MSFT": 0.4, "NVDA": 0.2}
        rebalanced = (1 + portfolio_returns(weights, prices, rebalanced=True)).prod()
        held = (1 + portfolio_returns(weights, prices, rebalanced=False)).prod()
        assert rebalanced != pytest.approx(held)

    def test_zero_weights_are_ignored(self, prices):
        with_zero = portfolio_returns({"AAPL": 0.5, "MSFT": 0.5, "NVDA": 0.0}, prices)
        without = portfolio_returns({"AAPL": 0.5, "MSFT": 0.5}, prices)
        pd.testing.assert_series_equal(with_zero, without)

    def test_unknown_ticker_raises(self, prices):
        with pytest.raises(KeyError, match="ZZZZ"):
            portfolio_returns({"ZZZZ": 1.0}, prices)

    def test_zero_total_weight_raises(self, prices):
        with pytest.raises(InsufficientDataError):
            portfolio_returns({"AAPL": 0.0, "MSFT": 0.0}, prices)

    def test_index_is_utc(self, prices):
        result = portfolio_returns({"AAPL": 1.0}, prices)
        assert str(result.index.tz) == "UTC"

    def test_feeds_the_risk_layer(self, prices, provider):
        """Der eigentliche Zweck: aggregiertes Portfolio-Risiko statt n/a."""
        from btlite.analytics.risk import compute_risk_metrics

        returns = portfolio_returns({"AAPL": 0.5, "MSFT": 0.5}, prices)
        benchmark = provider.get_history("SPY", period="5y").close.pct_change().dropna()
        metrics = compute_risk_metrics(returns, benchmark_returns=benchmark, risk_free_rate=0.03)
        assert metrics.volatility is not None
        assert metrics.max_drawdown is not None
        assert metrics.max_drawdown < 0
