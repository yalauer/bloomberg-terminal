"""Tests der Effizienzgrenzen- und Gewichts-Factories (Block 5)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from btlite.models.portfolio import (
    EfficientFrontier,
    PortfolioRequest,
    PortfolioResult,
    Position,
)
from btlite.viz import frontier, theme


@pytest.fixture
def curve() -> EfficientFrontier:
    volatility = np.linspace(0.10, 0.30, 25)
    expected = 0.02 + 0.45 * volatility - 0.35 * volatility**2
    return EfficientFrontier(
        data=pd.DataFrame(
            {
                "volatility": volatility,
                "expected_return": expected,
                "sharpe": (expected - 0.02) / volatility,
            }
        ),
        tickers=["AAPL", "MSFT", "NVDA"],
    )


@pytest.fixture
def request_model() -> PortfolioRequest:
    return PortfolioRequest(tickers=["AAPL", "MSFT", "NVDA"])


@pytest.fixture
def result(request_model, provenance) -> PortfolioResult:
    return PortfolioResult(
        request=request_model,
        positions=[
            Position(ticker="AAPL", weight=0.5),
            Position(ticker="MSFT", weight=0.3),
            Position(ticker="NVDA", weight=0.2),
        ],
        computed_at=datetime(2025, 1, 2, tzinfo=UTC),
        provenance=provenance,
    )


@pytest.fixture
def empty_result(request_model, provenance) -> PortfolioResult:
    return PortfolioResult(
        request=request_model,
        positions=[],
        computed_at=datetime(2025, 1, 2, tzinfo=UTC),
        provenance=provenance,
    )


class TestEfficientFrontier:
    def test_returns_figure(self, curve):
        assert isinstance(frontier.efficient_frontier(curve), go.Figure)

    def test_colours_points_by_sharpe(self, curve):
        fig = frontier.efficient_frontier(curve)
        markers = next(t for t in fig.data if t.name == "Sample points")
        assert list(markers.marker.color) == pytest.approx(list(curve.data["sharpe"]))

    def test_marks_optimum_automatically(self, curve):
        fig = frontier.efficient_frontier(curve)
        optimum = next(t for t in fig.data if t.name == "Optimum")
        best = curve.data.loc[curve.data["sharpe"].idxmax()]
        assert float(optimum.x[0]) == pytest.approx(float(best["volatility"]))

    def test_explicit_optimum_wins(self, curve):
        fig = frontier.efficient_frontier(curve, optimal=(0.18, 0.09))
        optimum = next(t for t in fig.data if t.name == "Optimum")
        assert float(optimum.x[0]) == pytest.approx(0.18)
        assert float(optimum.y[0]) == pytest.approx(0.09)

    def test_optimum_uses_amber_star(self, curve):
        fig = frontier.efficient_frontier(curve)
        optimum = next(t for t in fig.data if t.name == "Optimum")
        assert optimum.marker.color == theme.AMBER
        assert optimum.marker.symbol == "star"

    def test_individual_assets_are_scattered(self, curve):
        assets = pd.DataFrame(
            {"volatility": [0.22, 0.28], "expected_return": [0.08, 0.11]},
            index=["AAPL", "NVDA"],
        )
        fig = frontier.efficient_frontier(curve, assets=assets)
        single = next(t for t in fig.data if t.name == "Individual assets")
        assert list(single.text) == ["AAPL", "NVDA"]

    def test_axes_state_units(self, curve):
        fig = frontier.efficient_frontier(curve)
        assert "annualised" in fig.layout.xaxis.title.text
        assert "annualised" in fig.layout.yaxis.title.text

    def test_empty_frontier_yields_hint(self):
        empty = EfficientFrontier(
            data=pd.DataFrame({"volatility": [], "expected_return": [], "sharpe": []}),
            tickers=["AAPL"],
        )
        fig = frontier.efficient_frontier(empty)
        assert len(fig.data) == 0
        assert fig.layout.annotations


class TestWeightsBar:
    def test_returns_figure(self, result):
        assert isinstance(frontier.weights_bar(result), go.Figure)

    def test_shows_every_position(self, result):
        fig = frontier.weights_bar(result)
        assert len(fig.data[0].y) == 3

    def test_is_horizontal(self, result):
        assert frontier.weights_bar(result).data[0].orientation == "h"

    def test_min_weight_filters(self, result):
        fig = frontier.weights_bar(result, min_weight=0.25)
        assert len(fig.data[0].y) == 2

    def test_negative_weight_uses_down_colour(self, request_model, provenance):
        shorted = PortfolioRequest(tickers=["AAPL", "MSFT"], allow_shorting=True)
        model = PortfolioResult(
            request=shorted,
            positions=[
                Position(ticker="AAPL", weight=1.2),
                Position(ticker="MSFT", weight=-0.2),
            ],
            computed_at=datetime(2025, 1, 2, tzinfo=UTC),
            provenance=provenance,
        )
        fig = frontier.weights_bar(model)
        colors = list(fig.data[0].marker.color)
        assert theme.DOWN in colors

    def test_axis_states_unit(self, result):
        fig = frontier.weights_bar(result)
        assert "decimal" in fig.layout.xaxis.title.text

    def test_empty_positions_yield_hint(self, empty_result):
        fig = frontier.weights_bar(empty_result)
        assert len(fig.data) == 0
        assert fig.layout.annotations


class TestWeightsDonut:
    def test_returns_figure(self, result):
        assert isinstance(frontier.weights_donut(result), go.Figure)

    def test_is_a_donut(self, result):
        trace = frontier.weights_donut(result).data[0]
        assert trace.type == "pie"
        assert trace.hole > 0

    def test_labels_are_tickers(self, result):
        trace = frontier.weights_donut(result).data[0]
        assert set(trace.labels) == {"AAPL", "MSFT", "NVDA"}

    def test_empty_positions_yield_hint(self, empty_result):
        fig = frontier.weights_donut(empty_result)
        assert len(fig.data) == 0
        assert fig.layout.annotations
