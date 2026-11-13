"""Tests der Risiko-Chart-Factories (Block 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from btlite.models.portfolio import RiskMetrics
from btlite.viz import risk, theme


@pytest.fixture
def correlation() -> pd.DataFrame:
    tickers = ["AAPL", "MSFT", "NVDA"]
    values = np.array([[1.0, 0.72, 0.55], [0.72, 1.0, 0.61], [0.55, 0.61, 1.0]])
    return pd.DataFrame(values, index=tickers, columns=tickers)


@pytest.fixture
def drawdown(ohlcv) -> pd.Series:
    close = ohlcv["close"]
    return close / close.cummax() - 1.0


@pytest.fixture
def returns(ohlcv) -> pd.Series:
    return ohlcv["close"].pct_change().dropna()


@pytest.fixture
def metrics() -> RiskMetrics:
    return RiskMetrics(
        var_historical=-0.031,
        cvar_historical=-0.047,
        confidence=0.95,
        max_drawdown=-0.22,
    )


class TestCorrelationHeatmap:
    def test_returns_figure(self, correlation):
        assert isinstance(risk.correlation_heatmap(correlation), go.Figure)

    def test_is_heatmap(self, correlation):
        fig = risk.correlation_heatmap(correlation)
        assert fig.data[0].type == "heatmap"

    def test_scale_is_symmetric_around_zero(self, correlation):
        trace = risk.correlation_heatmap(correlation).data[0]
        assert trace.zmin == -1.0
        assert trace.zmax == 1.0
        assert trace.zmid == 0.0

    def test_uses_diverging_scale(self, correlation):
        trace = risk.correlation_heatmap(correlation).data[0]
        assert len(trace.colorscale) == len(theme.DIVERGING)

    def test_values_can_be_hidden(self, correlation):
        trace = risk.correlation_heatmap(correlation, show_values=False).data[0]
        assert trace.texttemplate is None

    def test_empty_matrix_yields_hint(self):
        fig = risk.correlation_heatmap(pd.DataFrame())
        assert len(fig.data) == 0
        assert fig.layout.annotations


class TestDrawdownChart:
    def test_returns_figure(self, drawdown):
        assert isinstance(risk.drawdown_chart(drawdown), go.Figure)

    def test_fills_to_zero(self, drawdown):
        trace = risk.drawdown_chart(drawdown).data[0]
        assert trace.fill == "tozeroy"

    def test_uses_down_colour(self, drawdown):
        trace = risk.drawdown_chart(drawdown).data[0]
        assert trace.line.color == theme.DOWN

    def test_max_drawdown_marked(self, drawdown):
        fig = risk.drawdown_chart(drawdown, max_drawdown=-0.22)
        assert len(fig.layout.shapes) == 1

    def test_axis_states_units(self, drawdown):
        fig = risk.drawdown_chart(drawdown)
        assert "decimal" in fig.layout.yaxis.title.text

    def test_empty_series_yields_hint(self):
        fig = risk.drawdown_chart(pd.Series(dtype=float))
        assert len(fig.data) == 0
        assert fig.layout.annotations

    def test_all_nan_series_yields_hint(self, drawdown):
        fig = risk.drawdown_chart(pd.Series(np.nan, index=drawdown.index))
        assert fig.layout.annotations


class TestReturnDistribution:
    def test_returns_figure(self, returns):
        assert isinstance(risk.return_distribution(returns), go.Figure)

    def test_is_histogram(self, returns):
        fig = risk.return_distribution(returns)
        assert fig.data[0].type == "histogram"

    def test_var_and_cvar_are_marked(self, returns, metrics):
        fig = risk.return_distribution(returns, metrics)
        assert len(fig.layout.shapes) == 2

    def test_missing_var_is_not_drawn_at_zero(self, returns):
        partial = RiskMetrics(var_historical=None, cvar_historical=None)
        fig = risk.return_distribution(returns, partial)
        assert len(fig.layout.shapes) == 0

    def test_only_available_marks_are_drawn(self, returns):
        partial = RiskMetrics(var_historical=-0.03, cvar_historical=None)
        fig = risk.return_distribution(returns, partial)
        assert len(fig.layout.shapes) == 1

    def test_works_without_metrics(self, returns):
        fig = risk.return_distribution(returns)
        assert len(fig.layout.shapes) == 0

    def test_empty_series_yields_hint(self):
        fig = risk.return_distribution(pd.Series(dtype=float))
        assert len(fig.data) == 0
        assert fig.layout.annotations


class TestRollingSeries:
    def test_volatility_returns_figure(self, returns):
        vol = returns.rolling(30).std() * np.sqrt(252)
        assert isinstance(risk.rolling_volatility(vol), go.Figure)

    def test_volatility_axis_states_annualised(self, returns):
        vol = returns.rolling(30).std() * np.sqrt(252)
        fig = risk.rolling_volatility(vol)
        assert "annualised" in fig.layout.yaxis.title.text

    def test_volatility_window_label_in_name(self, returns):
        vol = returns.rolling(30).std() * np.sqrt(252)
        fig = risk.rolling_volatility(vol, window_label="30 Tage")
        assert "30 Tage" in fig.data[0].name

    def test_beta_returns_figure(self, returns):
        beta = pd.Series(np.linspace(0.8, 1.3, len(returns)), index=returns.index)
        assert isinstance(risk.rolling_beta(beta), go.Figure)

    def test_beta_has_reference_line_at_one(self, returns):
        beta = pd.Series(np.linspace(0.8, 1.3, len(returns)), index=returns.index)
        fig = risk.rolling_beta(beta)
        assert len(fig.layout.shapes) == 1

    def test_beta_names_the_benchmark(self, returns):
        beta = pd.Series(np.linspace(0.8, 1.3, len(returns)), index=returns.index)
        fig = risk.rolling_beta(beta, benchmark="SPY")
        assert "SPY" in fig.data[0].name

    @pytest.mark.parametrize("factory", [risk.rolling_volatility, risk.rolling_beta])
    def test_empty_series_yields_hint(self, factory):
        fig = factory(pd.Series(dtype=float))
        assert len(fig.data) == 0
        assert fig.layout.annotations
