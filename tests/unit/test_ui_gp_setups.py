"""Tests der Setup-Checkliste auf der Seite GP.

Geprueft wird: das Panel erscheint unter dem Chart, traegt den Status als
Text, zeigt den Hinweissatz, und der gerenderte Text enthaelt keine
Empfehlungswoerter. Dazu die Chart-Ebene: Zonen mit Legendennamen,
Marker nur im dargestellten Zeitraum.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import pytest

from btlite.analytics.setups import setup_report
from btlite.config import Settings
from btlite.data.fixtures import FixtureProvider
from btlite.models.setups import (
    PriceZone,
    SetupCheck,
    SetupKey,
    SetupReport,
    SetupStatus,
    ZoneKind,
)
from btlite.ui.pages import gp
from btlite.ui.state import AppState, set_state
from btlite.viz.setups import add_setup_annotations

#: Woerter, die im Produkt nicht vorkommen duerfen (Nicht-Ziel).
FORBIDDEN = ("buy", "recommend", "entry", "sell", "bullish", "bearish")

OFFLINE_TICKERS = ("AAPL", "MSFT", "NVDA", "SPY", "JNJ", "XOM", "KO", "PG", "JPM", "WMT")


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[AppState]:
    monkeypatch.setenv("BTLITE_PORTFOLIO_HOLDINGS_FILE", str(tmp_path / "holdings.json"))
    settings = Settings()
    settings.data.offline = True
    instance = AppState(settings)
    set_state(instance)
    yield instance
    set_state(None)


def walk(node: Any) -> Iterator[Any]:
    yield node
    children = getattr(node, "children", None)
    if children is None:
        return
    if isinstance(children, list | tuple):
        for child in children:
            yield from walk(child)
    else:
        yield from walk(children)


def text_of(tree: Any) -> str:
    parts: list[str] = []
    for node in walk(tree):
        if isinstance(node, str):
            parts.append(node)
        title = getattr(node, "title", None)
        if isinstance(title, str):
            parts.append(title)
    return " ".join(parts)


def graphs(tree: Any) -> list[Any]:
    return [n for n in walk(tree) if type(n).__name__ == "Graph"]


class TestPanel:
    def test_setups_is_a_default_overlay(self):
        assert "setups" in {value for _, value in gp.OVERLAY_CHOICES}
        assert "setups" in gp.DEFAULT_OVERLAYS

    def test_panel_rendered_under_the_chart(self, state):
        tree = gp.render_charts(state, ["AAPL"], "1y")
        text = text_of(tree)
        assert "AAPL - Chart setups" in text
        for header in ("Setup", "Status", "Measurements"):
            assert header in text
        assert gp.SETUP_DISCLAIMER in text
        assert "bt-error" not in str(tree)

    def test_rsi_chart_container_matches_the_taller_figure(self, state):
        """Der RSI-Subplot darf das nachfolgende Setup-Panel nicht ueberdecken."""
        tree = gp.render_charts(state, ["AAPL"], "1y", overlays=("rsi", "setups"))
        graph = graphs(tree)[0]

        assert graph.figure.layout.height > gp.c.CHART_HEIGHT
        assert graph.style["height"] == f"{graph.figure.layout.height}px"

    def test_status_is_text_with_symbol(self, state):
        text = text_of(gp.render_charts(state, ["AAPL"], "1y"))
        statuses = ("\u2713 met", "\u2013 not met", "? insufficient data")
        assert any(status in text for status in statuses)

    def test_every_setup_listed(self, state):
        text = text_of(gp.render_charts(state, ["WMT"], "1y"))
        report = setup_report(FixtureProvider().get_history("WMT", period=None))
        for check in report.checks:
            assert check.label in text

    def test_off_without_the_overlay(self, state):
        tree = gp.render_charts(state, ["AAPL"], "1y", overlays=("sma",))
        assert "Chart setups" not in text_of(tree)

    def test_panel_per_ticker(self, state):
        text = text_of(gp.render_charts(state, ["AAPL", "MSFT"], "1y"))
        assert "AAPL - Chart setups" in text
        assert "MSFT - Chart setups" in text

    def test_unknown_ticker_does_not_crash(self, state):
        tree = gp.render_charts(state, ["NOSUCHTICKER"], "1y")
        assert "bt-error" in str(tree)

    @pytest.mark.parametrize("ticker", OFFLINE_TICKERS)
    def test_no_recommendation_words(self, state, ticker):
        text = text_of(gp.render_charts(state, [ticker], "1y")).lower()
        # Der Pflicht-Hinweissatz verneint die Empfehlung; er allein darf das Wort tragen.
        assert gp.SETUP_DISCLAIMER.lower() in text
        text = text.replace(gp.SETUP_DISCLAIMER.lower(), "")
        for word in FORBIDDEN:
            assert not re.search(rf"\b{word}", text), f"'{word}' in panel for {ticker}"

    def test_insufficient_data_shown_as_such(self):
        report = SetupReport(
            ticker="TEST",
            as_of=datetime(2026, 9, 3, tzinfo=UTC),
            price=10.0,
            checks=[
                SetupCheck(
                    key=SetupKey.BREAKOUT,
                    label="Breakout",
                    status=SetupStatus.INSUFFICIENT_DATA,
                    explanation="Breakout: insufficient data",
                )
            ],
        )
        panel = gp.setup_panel(report)
        cells = [n for n in walk(panel) if type(n).__name__ == "Td"]
        assert cells[1].children == "? insufficient data"
        assert cells[1].className == "bt-na"

    def test_missing_measurement_is_na_not_zero(self, state):
        report = setup_report(FixtureProvider().get_history("SPY", period=None))
        check = report.check(SetupKey.SUPPORT_MACD)
        assert check is not None
        text = text_of(gp.setup_panel(report))
        for item in check.measurements:
            if item.value is None:
                assert f"{item.label}: n/a" in text


def _report_with_zone_and_marker() -> SetupReport:
    return SetupReport(
        ticker="TEST",
        as_of=datetime(2026, 9, 3, tzinfo=UTC),
        price=100.0,
        checks=[
            SetupCheck(
                key=SetupKey.SUPPORT_MACD,
                label="Near support",
                status=SetupStatus.MET,
                explanation="Near support: met",
                event_at=datetime(2026, 9, 1, tzinfo=UTC),
                event_price=99.0,
                event_label="MACD cross up",
            ),
            SetupCheck(
                key=SetupKey.RSI_RECOVERY,
                label="RSI",
                status=SetupStatus.NOT_MET,
                explanation="RSI: not met",
                event_at=datetime(2025, 1, 1, tzinfo=UTC),
                event_price=80.0,
                event_label="RSI low 25.0",
            ),
        ],
        zones=[
            PriceZone(level=98.0, lower=97.5, upper=98.5, kind=ZoneKind.SUPPORT, touches=3),
            PriceZone(level=110.0, lower=110.0, upper=110.0, kind=ZoneKind.RESISTANCE, touches=1),
        ],
    )


class TestChartLayer:
    def test_zones_carry_their_level_at_the_edge(self):
        """Das Niveau jeder Zone steht als Beschriftung am rechten Rand."""
        fig = add_setup_annotations(go.Figure(), _report_with_zone_and_marker())
        labels = [s.label.text for s in fig.layout.shapes if s.label and s.label.text]
        assert "Support 98.00" in labels
        assert "Resistance 110.00" in labels

    def test_one_legend_entry_per_zone_kind(self):
        """Eingeklappt: ein Sammeleintrag je Art statt einer je Zone.

        Ein Eintrag je Zone liess die Legende ueber dem Chart in mehrere
        Zeilen umbrechen; die gingen von der festen Hoehe ab und stauchten
        den Kursbereich.
        """
        report = _report_with_zone_and_marker().model_copy(
            update={
                "zones": [
                    PriceZone(level=lv, lower=lv, upper=lv, kind=ZoneKind.SUPPORT, touches=1)
                    for lv in (90.0, 94.0, 98.0)
                ]
                + [
                    PriceZone(level=lv, lower=lv, upper=lv, kind=ZoneKind.RESISTANCE, touches=2)
                    for lv in (110.0, 115.0)
                ]
            }
        )
        fig = add_setup_annotations(go.Figure(), report)
        shown = [s.name for s in fig.layout.shapes if s.showlegend]
        assert shown == ["Support zones (3)", "Resistance zones (2)"]

    def test_a_legend_click_hides_every_zone_of_that_kind(self):
        """Linie und Band einer Art teilen sich die Legendengruppe."""
        fig = add_setup_annotations(go.Figure(), _report_with_zone_and_marker())
        support = [s for s in fig.layout.shapes if s.y0 is not None and s.y0 < 100]
        assert {s.type for s in support} == {"line", "rect"}
        assert len({s.legendgroup for s in support}) == 1
        resistance = [s for s in fig.layout.shapes if s.y0 is not None and s.y0 >= 100]
        assert {s.legendgroup for s in resistance}.isdisjoint({s.legendgroup for s in support})

    def test_events_share_one_legend_entry(self):
        fig = add_setup_annotations(go.Figure(), _report_with_zone_and_marker())
        shown = [t.name for t in fig.data if t.showlegend is not False]
        assert shown == ["Setup events"]
        assert len(fig.data) == 2  # beide Marker gezeichnet, einer in der Legende

    def test_band_only_for_zones_with_width(self):
        fig = add_setup_annotations(go.Figure(), _report_with_zone_and_marker())
        rects = [s for s in fig.layout.shapes if s.type == "rect"]
        assert len(rects) == 1
        assert (rects[0].y0, rects[0].y1) == (97.5, 98.5)

    def test_markers_only_inside_the_shown_period(self):
        fig = add_setup_annotations(
            go.Figure(),
            _report_with_zone_and_marker(),
            start=datetime(2026, 1, 1, tzinfo=UTC),
            end=datetime(2026, 9, 3, tzinfo=UTC),
        )
        texts = [t.text[0] for t in fig.data]
        assert texts == ["MACD cross up"]

    def test_returns_the_figure(self):
        fig = go.Figure()
        assert add_setup_annotations(fig, _report_with_zone_and_marker()) is fig

    def test_gp_chart_contains_zone_shapes(self, state):
        tree = gp.render_charts(state, ["WMT"], "1y")
        figure = go.Figure(graphs(tree)[0].figure)
        names = [s.name for s in figure.layout.shapes if s.name]
        assert any(name.startswith(("Support", "Resistance")) for name in names)


class TestZoneDetails:
    """Die Einzelheiten der Zonen stehen zugeklappt unter dem Chart."""

    def test_the_list_is_collapsed_by_default(self):
        panel = gp.setup_panel(_report_with_zone_and_marker())
        details = [n for n in _walk(panel) if getattr(n, "_type", "") == "Details"]
        assert len(details) == 1
        assert not getattr(details[0], "open", False)

    def test_every_zone_is_listed_with_touches(self):
        text = text_of(gp.setup_panel(_report_with_zone_and_marker()))
        assert "Price zones (2)" in text
        assert "98.00" in text
        assert "110.00" in text
        assert "Touches" in text

    def test_real_chart_legend_stays_short(self, state):
        """WMT hat offline ueber zehn Zonen - die Legende bleibt kurz."""
        tree = gp.render_charts(state, ["WMT"], "1y")
        figure = go.Figure(graphs(tree)[0].figure)
        setup_entries = [s.name for s in figure.layout.shapes if s.showlegend]
        assert len(setup_entries) <= 2
        assert len([s for s in figure.layout.shapes if s.type == "line"]) > 2


def _walk(node):
    yield node
    kids = getattr(node, "children", None)
    for kid in kids if isinstance(kids, list) else [kids]:
        if kid is not None and hasattr(kid, "_type"):
            yield from _walk(kid)
