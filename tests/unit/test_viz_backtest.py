"""Tests der Backtest-Chart-Factories (Block 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from btlite.models.backtest import (
    BacktestRequest,
    BacktestResult,
    PerformanceMetrics,
    StrategyConfig,
    Trade,
    TradeSide,
)
from btlite.viz import backtest, theme


@pytest.fixture
def equity_frame(ohlcv) -> pd.DataFrame:
    close = ohlcv["close"]
    equity = 100_000 * (close / close.iloc[0])
    return pd.DataFrame(
        {
            "equity": equity,
            "benchmark_equity": 250.0 * (close / close.iloc[0]),
            "drawdown": equity / equity.cummax() - 1.0,
        },
        index=ohlcv.index,
    )


@pytest.fixture
def request_model() -> BacktestRequest:
    return BacktestRequest(
        tickers=["AAPL"],
        strategy=StrategyConfig(key="sma_cross"),
        benchmark="SPY",
    )


@pytest.fixture
def trades(ohlcv) -> list[Trade]:
    index = ohlcv.index
    return [
        Trade(
            ticker="AAPL",
            side=TradeSide.LONG,
            entry_time=index[20].to_pydatetime(),
            entry_price=float(ohlcv["close"].iloc[20]),
            exit_time=index[60].to_pydatetime(),
            exit_price=float(ohlcv["close"].iloc[60]),
            size=100.0,
            pnl=1250.0,
        ),
        Trade(
            ticker="AAPL",
            side=TradeSide.LONG,
            entry_time=index[120].to_pydatetime(),
            entry_price=float(ohlcv["close"].iloc[120]),
            size=100.0,
        ),
    ]


@pytest.fixture
def result(request_model, equity_frame, trades, provenance) -> BacktestResult:
    return BacktestResult(
        request=request_model,
        equity_curve=equity_frame,
        trades=trades,
        metrics=PerformanceMetrics(total_return=0.31, max_drawdown=-0.18),
        engine_used="vectorbt",
        provenance=provenance,
    )


@pytest.fixture
def empty_result(request_model, equity_frame, provenance) -> BacktestResult:
    return BacktestResult(
        request=request_model,
        equity_curve=equity_frame.iloc[0:0],
        metrics=PerformanceMetrics(),
        engine_used="vectorbt",
        provenance=provenance,
    )


class TestEquityCurve:
    def test_returns_figure(self, result):
        assert isinstance(backtest.equity_curve(result), go.Figure)

    def test_plots_strategy_and_benchmark(self, result):
        fig = backtest.equity_curve(result)
        names = [t.name for t in fig.data]
        assert "Strategy" in names
        assert "SPY" in names

    def test_benchmark_is_rebased_to_strategy_start(self, result):
        fig = backtest.equity_curve(result)
        strategy = next(t for t in fig.data if t.name == "Strategy")
        bench = next(t for t in fig.data if t.name == "SPY")
        assert float(bench.y[0]) == pytest.approx(float(strategy.y[0]))

    def test_drawdown_subplot_present(self, result):
        fig = backtest.equity_curve(result)
        assert "Drawdown" in [t.name for t in fig.data]

    def test_drawdown_can_be_disabled(self, result):
        fig = backtest.equity_curve(result, show_drawdown=False)
        assert "Drawdown" not in [t.name for t in fig.data]

    def test_trade_markers_optional(self, result):
        without = backtest.equity_curve(result)
        with_marks = backtest.equity_curve(result, show_trades=True)
        assert len(with_marks.data) > len(without.data)

    def test_axis_states_currency(self, result):
        """Die Achse nennt die Kontowaehrung - welche, sagt das Ergebnis.

        Vorher stand hier "USD" als Literal. Das prueft die Vorgabe des
        Modells mit, nicht die Beschriftung: waehlt jemand ein anderes
        Konto, faellt der Test, obwohl die Achse korrekt bleibt.
        """
        fig = backtest.equity_curve(result)
        assert result.currency.value in fig.layout.yaxis.title.text

    def test_log_axis(self, result):
        fig = backtest.equity_curve(result, log_y=True)
        assert fig.layout.yaxis.type == "log"

    def test_missing_benchmark_column_is_fine(self, request_model, equity_frame, provenance):
        model = BacktestResult(
            request=request_model,
            equity_curve=equity_frame.drop(columns=["benchmark_equity"]),
            metrics=PerformanceMetrics(),
            engine_used="vectorbt",
            provenance=provenance,
        )
        fig = backtest.equity_curve(model)
        assert "Strategy" in [t.name for t in fig.data]

    def test_empty_curve_yields_hint(self, empty_result):
        fig = backtest.equity_curve(empty_result)
        assert len(fig.data) == 0
        assert fig.layout.annotations


class TestTradeMarkers:
    def test_returns_figure(self, result, ohlcv):
        fig = backtest.trade_markers(result, ohlcv["close"])
        assert isinstance(fig, go.Figure)

    def test_marks_entries_and_exits(self, result, ohlcv):
        fig = backtest.trade_markers(result, ohlcv["close"])
        names = [t.name for t in fig.data]
        assert "Entry" in names
        assert "Exit" in names

    def test_open_trade_has_no_exit_marker(self, result, ohlcv):
        fig = backtest.trade_markers(result, ohlcv["close"])
        entries = next(t for t in fig.data if t.name == "Entry")
        exits = next(t for t in fig.data if t.name == "Exit")
        assert len(entries.x) == 2
        assert len(exits.x) == 1

    def test_direction_colours_are_colourblind_safe(self, result, ohlcv):
        fig = backtest.trade_markers(result, ohlcv["close"])
        entries = next(t for t in fig.data if t.name == "Entry")
        exits = next(t for t in fig.data if t.name == "Exit")
        assert entries.marker.color == theme.UP
        assert exits.marker.color == theme.DOWN

    def test_without_trades_only_price_is_drawn(
        self, request_model, equity_frame, provenance, ohlcv
    ):
        model = BacktestResult(
            request=request_model,
            equity_curve=equity_frame,
            trades=[],
            metrics=PerformanceMetrics(),
            engine_used="vectorbt",
            provenance=provenance,
        )
        fig = backtest.trade_markers(model, ohlcv["close"])
        assert len(fig.data) == 1

    def test_empty_prices_yield_hint(self, result):
        fig = backtest.trade_markers(result, pd.Series(dtype=float))
        assert len(fig.data) == 0
        assert fig.layout.annotations


class TestMonthlyReturnsHeatmap:
    @pytest.fixture
    def monthly(self) -> pd.Series:
        index = pd.date_range("2023-01-31", periods=26, freq="ME", tz="UTC")
        rng = np.random.default_rng(3)
        return pd.Series(rng.normal(0.01, 0.04, len(index)), index=index)

    def test_returns_figure(self, monthly):
        assert isinstance(backtest.monthly_returns_heatmap(monthly), go.Figure)

    def test_is_heatmap_with_month_columns(self, monthly):
        fig = backtest.monthly_returns_heatmap(monthly)
        assert fig.data[0].type == "heatmap"
        assert len(fig.data[0].x) == 12

    def test_rows_are_years(self, monthly):
        fig = backtest.monthly_returns_heatmap(monthly)
        assert set(fig.data[0].y) == {"2023", "2024", "2025"}

    def test_scale_is_symmetric_around_zero(self, monthly):
        trace = backtest.monthly_returns_heatmap(monthly).data[0]
        assert trace.zmid == 0.0
        assert trace.zmin == pytest.approx(-trace.zmax)

    def test_months_without_data_render_as_na(self, monthly):
        fig = backtest.monthly_returns_heatmap(monthly)
        flat = [cell for row in fig.data[0].text for cell in row]
        # 26 Monate fuellen drei Jahreszeilen nicht vollstaendig.
        assert theme.NA_TEXT in flat

    def test_empty_series_yields_hint(self):
        fig = backtest.monthly_returns_heatmap(pd.Series(dtype=float))
        assert len(fig.data) == 0
        assert fig.layout.annotations
