"""Tests der Fundamental-Chart-Factories (Block 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from btlite.models.common import Period
from btlite.models.fundamentals import (
    FinancialStatement,
    LineItem,
    PeerComparison,
    Ratio,
    RatioCategory,
    RatioSet,
    StatementType,
)
from btlite.viz import fundamentals, theme


@pytest.fixture
def income(statement_frame, provenance) -> FinancialStatement:
    return FinancialStatement(
        ticker="AAPL",
        statement_type=StatementType.INCOME,
        period=Period.ANNUAL,
        data=statement_frame,
        provenance=provenance,
    )


@pytest.fixture
def empty_income(statement_frame, provenance) -> FinancialStatement:
    return FinancialStatement(
        ticker="AAPL",
        statement_type=StatementType.INCOME,
        period=Period.ANNUAL,
        data=statement_frame.iloc[0:0],
        provenance=provenance,
    )


@pytest.fixture
def ratios(provenance) -> RatioSet:
    return RatioSet(
        ticker="AAPL",
        period=Period.ANNUAL,
        provenance=provenance,
        ratios={
            "current_ratio": Ratio(
                key="current_ratio",
                label="Current Ratio",
                category=RatioCategory.LIQUIDITY,
                value=1.35,
            ),
            "quick_ratio": Ratio(
                key="quick_ratio",
                label="Quick Ratio",
                category=RatioCategory.LIQUIDITY,
                value=None,
                inputs_missing=["inventory"],
            ),
            "net_margin": Ratio(
                key="net_margin",
                label="Net Margin",
                category=RatioCategory.PROFITABILITY,
                value=0.245,
                unit="percent",
            ),
        },
    )


@pytest.fixture
def peers(provenance) -> PeerComparison:
    frame = pd.DataFrame(
        {
            "pe": [28.4, 31.2, np.nan, 19.8],
            "roe": [0.42, 0.35, 0.28, 0.31],
        },
        index=["AAPL", "MSFT", "GOOG", "META"],
    )
    return PeerComparison(
        subject="AAPL",
        peers=["MSFT", "GOOG", "META"],
        period=Period.ANNUAL,
        data=frame,
        provenance=provenance,
    )


class TestStatementTable:
    def test_returns_figure(self, income):
        assert isinstance(fundamentals.statement_table(income), go.Figure)

    def test_is_a_table_trace(self, income):
        fig = fundamentals.statement_table(income)
        assert fig.data[0].type == "table"

    def test_missing_values_render_as_na_never_zero(self, income):
        fig = fundamentals.statement_table(income)
        # Die Fixture hat ein NaN in der letzten Periode.
        last_column = fig.data[0].cells.values[-1]  # noqa: PD011 - Plotly-Attribut
        assert theme.NA_TEXT in last_column

    def test_rows_cover_all_line_items(self, income, statement_frame):
        fig = fundamentals.statement_table(income)
        assert len(fig.data[0].cells.values[0]) == len(statement_frame)  # noqa: PD011

    def test_max_rows_truncates(self, income):
        fig = fundamentals.statement_table(income, max_rows=2)
        assert len(fig.data[0].cells.values[0]) == 2  # noqa: PD011

    def test_title_names_the_currency(self, income):
        fig = fundamentals.statement_table(income)
        assert "USD" in fig.layout.title.text

    def test_empty_statement_yields_hint(self, empty_income):
        fig = fundamentals.statement_table(empty_income)
        assert len(fig.data) == 0
        assert fig.layout.annotations


class TestStatementTableRecords:
    def test_returns_plain_records_not_a_figure(self, income):
        records = fundamentals.statement_table_records(income)
        assert isinstance(records, list)
        assert all(isinstance(r, dict) for r in records)

    def test_every_record_has_position(self, income):
        records = fundamentals.statement_table_records(income)
        assert all("position" in r for r in records)

    def test_missing_value_is_na(self, income):
        records = fundamentals.statement_table_records(income)
        flat = [v for r in records for v in r.values()]
        assert theme.NA_TEXT in flat

    def test_empty_statement_yields_empty_list(self, empty_income):
        assert fundamentals.statement_table_records(empty_income) == []


class TestStatementTrend:
    def test_returns_figure(self, income):
        assert isinstance(fundamentals.statement_trend(income), go.Figure)

    def test_draws_only_present_line_items(self, income):
        # Die Fixture kennt revenue und net_income, aber kein free_cashflow.
        fig = fundamentals.statement_trend(income)
        names = [t.name for t in fig.data]
        assert "Revenue" in names
        assert "Net Income" in names
        assert "Free Cashflow" not in names

    def test_missing_line_item_is_not_drawn_as_zero(self, income):
        fig = fundamentals.statement_trend(income, items=(LineItem.FREE_CASHFLOW,))
        assert len(fig.data) == 0
        assert fig.layout.annotations

    def test_missing_period_value_is_none_not_zero(self, income):
        fig = fundamentals.statement_trend(income, items=(LineItem.OPERATING_CASHFLOW,))
        trace = fig.data[0]
        assert trace.y[-1] is None

    def test_grouped_bars(self, income):
        fig = fundamentals.statement_trend(income)
        assert fig.layout.barmode == "group"

    def test_axis_title_names_currency(self, income):
        fig = fundamentals.statement_trend(income)
        assert "USD" in fig.layout.yaxis.title.text

    def test_empty_statement_yields_hint(self, empty_income):
        fig = fundamentals.statement_trend(empty_income)
        assert fig.layout.annotations


class TestRatioPanel:
    def test_returns_figure(self, ratios):
        assert isinstance(fundamentals.ratio_panel(ratios), go.Figure)

    def test_one_subplot_per_populated_category(self, ratios):
        fig = fundamentals.ratio_panel(ratios)
        # Liquiditaet und Rentabilitaet sind belegt, die uebrigen nicht.
        assert len(fig.data) == 2

    def test_unavailable_ratio_is_labelled_na(self, ratios):
        fig = fundamentals.ratio_panel(ratios)
        texts = [t for trace in fig.data for t in trace.text]
        assert theme.NA_TEXT in texts

    def test_unavailable_ratio_is_greyed_not_coloured(self, ratios):
        fig = fundamentals.ratio_panel(ratios)
        liquidity = fig.data[0]
        assert theme.GRID in list(liquidity.marker.color)

    def test_percent_unit_is_scaled_for_display(self, ratios):
        fig = fundamentals.ratio_panel(ratios)
        texts = [t for trace in fig.data for t in trace.text]
        assert any("24.50" in t for t in texts)

    def test_category_filter(self, ratios):
        fig = fundamentals.ratio_panel(ratios, categories=(RatioCategory.LIQUIDITY,))
        assert len(fig.data) == 1

    def test_empty_ratio_set_yields_hint(self, provenance):
        empty = RatioSet(ticker="AAPL", period=Period.ANNUAL, provenance=provenance)
        fig = fundamentals.ratio_panel(empty)
        assert len(fig.data) == 0
        assert fig.layout.annotations


class TestPeerComparison:
    def test_returns_figure(self, peers):
        assert isinstance(fundamentals.peer_comparison(peers), go.Figure)

    def test_bars_mark_the_subject_in_amber(self, peers):
        fig = fundamentals.peer_comparison(peers)
        colors = list(fig.data[0].marker.color)
        assert colors[0] == theme.AMBER  # AAPL steht an erster Stelle
        assert colors[1] != theme.AMBER

    def test_missing_peer_value_is_none_not_zero(self, peers):
        fig = fundamentals.peer_comparison(peers, ratio_keys=("pe",))
        assert fig.data[0].y[2] is None

    def test_heatmap_variant_returns_figure(self, peers):
        fig = fundamentals.peer_comparison(peers, as_heatmap=True)
        assert fig.data[0].type == "heatmap"

    def test_heatmap_marks_subject_row(self, peers):
        fig = fundamentals.peer_comparison(peers, as_heatmap=True)
        assert len(fig.layout.shapes) == 1
        assert fig.layout.shapes[0].line.color == theme.AMBER

    def test_heatmap_shows_na_for_missing(self, peers):
        fig = fundamentals.peer_comparison(peers, as_heatmap=True)
        flat = [c for row in fig.data[0].text for c in row]
        assert theme.NA_TEXT in flat

    def test_ratio_key_filter(self, peers):
        fig = fundamentals.peer_comparison(peers, ratio_keys=("roe",))
        assert len(fig.data) == 1
        assert fig.data[0].name == "roe"

    def test_unknown_ratio_key_yields_hint(self, peers):
        fig = fundamentals.peer_comparison(peers, ratio_keys=("does_not_exist",))
        assert fig.layout.annotations

    def test_empty_matrix_yields_hint(self, provenance):
        empty = PeerComparison(
            subject="AAPL",
            peers=[],
            period=Period.ANNUAL,
            data=pd.DataFrame(),
            provenance=provenance,
        )
        fig = fundamentals.peer_comparison(empty)
        assert len(fig.data) == 0
        assert fig.layout.annotations
