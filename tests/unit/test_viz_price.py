"""Tests der Kurs-Chart-Factories (Block 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from btlite.models.market import PriceComparison, PriceSeries
from btlite.viz import price, theme
from btlite.viz.price import TechnicalOverlays


@pytest.fixture
def series(ohlcv, provenance) -> PriceSeries:
    return PriceSeries(ticker="AAPL", adjusted=True, data=ohlcv, provenance=provenance)


@pytest.fixture
def empty_series(ohlcv, provenance) -> PriceSeries:
    return PriceSeries(ticker="AAPL", adjusted=True, data=ohlcv.iloc[0:0], provenance=provenance)


@pytest.fixture
def overlays(ohlcv) -> TechnicalOverlays:
    close = ohlcv["close"]
    index = ohlcv.index
    rng = np.random.default_rng(7)
    return TechnicalOverlays(
        sma={"SMA 50": close.rolling(50).mean()},
        ema={"EMA 20": close.ewm(span=20).mean()},
        bollinger=pd.DataFrame(
            {"upper": close * 1.02, "middle": close, "lower": close * 0.98}, index=index
        ),
        rsi=pd.Series(np.clip(50 + rng.normal(0, 15, len(index)), 0, 100), index=index),
        macd=pd.DataFrame(
            {
                "macd": rng.normal(0, 1, len(index)),
                "signal": rng.normal(0, 1, len(index)),
                "histogram": rng.normal(0, 1, len(index)),
            },
            index=index,
        ),
    )


@pytest.fixture
def comparison_model(ohlcv, provenance) -> PriceComparison:
    close = ohlcv["close"]
    frame = pd.DataFrame({"AAPL": close, "MSFT": close * 1.4}, index=ohlcv.index)
    return PriceComparison(
        tickers=["AAPL", "MSFT"], normalized=False, data=frame, provenance=provenance
    )


class TestCandlestick:
    def test_returns_figure(self, series):
        assert isinstance(price.candlestick(series), go.Figure)

    def test_has_candlestick_and_volume(self, series):
        fig = price.candlestick(series)
        kinds = [t.type for t in fig.data]
        assert "candlestick" in kinds
        assert "bar" in kinds

    def test_volume_can_be_disabled(self, series):
        fig = price.candlestick(series, show_volume=False)
        assert [t.type for t in fig.data] == ["candlestick"]

    def test_uses_colourblind_safe_direction_colors(self, series):
        fig = price.candlestick(series)
        candle = next(t for t in fig.data if t.type == "candlestick")
        assert candle.increasing.fillcolor == theme.UP
        assert candle.decreasing.fillcolor == theme.DOWN

    def test_applies_theme(self, series):
        fig = price.candlestick(series)
        assert fig.layout.template.layout.plot_bgcolor == theme.SURFACE

    def test_rangeslider_disabled_but_selector_present(self, series):
        fig = price.candlestick(series)
        assert fig.layout.xaxis.rangeslider.visible is False
        labels = [b.label for b in fig.layout.xaxis.rangeselector.buttons]
        assert labels == ["1M", "3M", "6M", "1Y", "5Y", "MAX"]

    def test_range_selector_can_be_disabled(self, series):
        fig = price.candlestick(series, range_selector=False)
        assert not fig.layout.xaxis.rangeselector.buttons

    def test_crosshair_enabled(self, series):
        fig = price.candlestick(series)
        assert fig.layout.hovermode == "x unified"

    def test_log_axis_is_display_only(self, series):
        fig = price.candlestick(series, log_y=True)
        assert fig.layout.yaxis.type == "log"

    def test_axis_titles_carry_units(self, series):
        fig = price.candlestick(series)
        assert "USD" in fig.layout.yaxis.title.text
        assert "UTC" in fig.layout.xaxis2.title.text

    def test_empty_series_yields_hint_figure(self, empty_series):
        fig = price.candlestick(empty_series)
        assert len(fig.data) == 0
        assert "No price data" in fig.layout.annotations[0].text

    def test_empty_series_does_not_raise(self, empty_series):
        assert isinstance(price.candlestick(empty_series), go.Figure)


class TestOverlays:
    def test_overlays_add_traces(self, series, overlays):
        plain = price.candlestick(series)
        rich = price.candlestick(series, overlays)
        assert len(rich.data) > len(plain.data)

    def test_named_overlays_appear_in_legend(self, series, overlays):
        fig = price.candlestick(series, overlays)
        names = [t.name for t in fig.data]
        assert "SMA 50" in names
        assert "EMA 20" in names

    def test_rsi_gets_own_subplot_with_bounds(self, series, overlays):
        fig = price.candlestick(series, overlays)
        rsi_axis = next(
            axis
            for key, axis in fig.layout.to_plotly_json().items()
            if key.startswith("yaxis")
            and isinstance(axis, dict)
            and axis.get("title", {}).get("text") == "RSI (0-100)"
        )
        assert rsi_axis["range"] == [0, 100]

    def test_macd_gets_own_subplot(self, series, overlays):
        fig = price.candlestick(series, overlays)
        titles = [
            axis.get("title", {}).get("text")
            for key, axis in fig.layout.to_plotly_json().items()
            if key.startswith("yaxis") and isinstance(axis, dict)
        ]
        assert "MACD (price currency)" in titles

    def test_partial_overlays_are_allowed(self, series, ohlcv):
        only_sma = TechnicalOverlays(sma={"SMA 20": ohlcv["close"].rolling(20).mean()})
        fig = price.candlestick(series, only_sma)
        assert "SMA 20" in [t.name for t in fig.data]

    def test_empty_overlays_behave_like_none(self, series):
        assert len(price.candlestick(series, TechnicalOverlays()).data) == len(
            price.candlestick(series).data
        )

    def test_overlay_flags_reflect_content(self, overlays):
        assert overlays.has_rsi is True
        assert overlays.has_macd is True
        assert TechnicalOverlays().has_rsi is False
        assert TechnicalOverlays().has_macd is False


class TestOhlcAndLine:
    def test_ohlc_returns_figure(self, series):
        fig = price.ohlc(series)
        assert isinstance(fig, go.Figure)
        assert "ohlc" in [t.type for t in fig.data]

    def test_line_returns_figure(self, series):
        fig = price.line(series)
        assert isinstance(fig, go.Figure)
        assert [t.type for t in fig.data] == ["scatter"]

    def test_line_plots_close(self, series, ohlcv):
        fig = price.line(series)
        assert list(fig.data[0].y) == pytest.approx(list(ohlcv["close"]))

    @pytest.mark.parametrize("factory", [price.ohlc, price.line])
    def test_empty_yields_hint(self, factory, empty_series):
        fig = factory(empty_series)
        assert len(fig.data) == 0
        assert fig.layout.annotations


class TestComparison:
    def test_returns_figure(self, comparison_model):
        assert isinstance(price.comparison(comparison_model), go.Figure)

    def test_one_trace_per_ticker(self, comparison_model):
        fig = price.comparison(comparison_model)
        assert [t.name for t in fig.data] == ["AAPL", "MSFT"]

    def test_normalizes_to_base_100(self, comparison_model):
        fig = price.comparison(comparison_model)
        for trace in fig.data:
            assert float(trace.y[0]) == pytest.approx(100.0)

    def test_already_normalized_payload_is_untouched(self, ohlcv, provenance):
        frame = pd.DataFrame(
            {"AAPL": np.full(len(ohlcv), 100.0), "MSFT": np.full(len(ohlcv), 250.0)},
            index=ohlcv.index,
        )
        model = PriceComparison(
            tickers=["AAPL", "MSFT"], normalized=True, data=frame, provenance=provenance
        )
        fig = price.comparison(model)
        assert float(fig.data[1].y[0]) == pytest.approx(250.0)

    def test_series_get_distinct_palette_colors(self, comparison_model):
        fig = price.comparison(comparison_model)
        colors = [t.line.color for t in fig.data]
        assert colors == [theme.CATEGORICAL[0], theme.CATEGORICAL[1]]
        assert len(set(colors)) == len(colors)

    def test_axis_title_states_the_base(self, comparison_model):
        fig = price.comparison(comparison_model)
        assert "100" in fig.layout.yaxis.title.text

    def test_empty_payload_yields_hint(self, ohlcv, provenance):
        model = PriceComparison(
            tickers=["AAPL"],
            normalized=True,
            data=pd.DataFrame({"AAPL": []}, index=ohlcv.index[:0]),
            provenance=provenance,
        )
        fig = price.comparison(model)
        assert len(fig.data) == 0
        assert fig.layout.annotations

    def test_all_nan_column_yields_hint_not_empty_frame(self, ohlcv, provenance):
        frame = pd.DataFrame({"AAPL": np.full(len(ohlcv), np.nan)}, index=ohlcv.index)
        model = PriceComparison(
            tickers=["AAPL"], normalized=True, data=frame, provenance=provenance
        )
        fig = price.comparison(model)
        assert fig.layout.annotations
