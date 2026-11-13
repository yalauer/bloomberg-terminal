"""Tests: Zeitraum 1D/1W und automatisch kleinstmoegliche Aufloesung.

Die Grenzen der Quelle sind live gemessen (26.09.2026, AAPL): 1m bis 5d,
5m bis 1mo, 1h bis 2y, darueber nur Tageskerzen. Ausserhalb liefert Yahoo
eine leere Reihe statt eines Fehlers - deshalb eine feste Tabelle.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

import pandas as pd
import plotly.graph_objects as go
import pytest

from btlite.analytics.series import CHART_INTERVALS, allowed_intervals, finest_interval
from btlite.config import Settings
from btlite.models.common import Interval
from btlite.ui.app import create_app, render_page
from btlite.ui.pages import gp
from btlite.ui.state import AppState
from btlite.viz.intraday import hide_session_gaps, localize_time_axis, session_breaks

if TYPE_CHECKING:
    from pathlib import Path


# --------------------------------------------------------------------------
# Tabelle der Aufloesungen


class TestFinestInterval:
    @pytest.mark.parametrize(
        ("period", "finest"),
        [
            ("1d", Interval.M1),
            ("5d", Interval.M1),
            ("1mo", Interval.M5),
            ("3mo", Interval.H1),
            ("6mo", Interval.H1),
            ("1y", Interval.H1),
            ("3y", Interval.D1),
            ("5y", Interval.D1),
            ("max", Interval.D1),
        ],
    )
    def test_matches_the_measured_source_limits(self, period: str, finest: Interval) -> None:
        assert finest_interval(period) is finest

    def test_every_gp_period_has_an_entry(self) -> None:
        for _label, period in gp.GP_PERIOD_CHOICES:
            assert period in CHART_INTERVALS, period

    def test_offline_only_daily_and_weekly(self) -> None:
        for period in CHART_INTERVALS:
            allowed = allowed_intervals(period, intraday_available=False)
            assert allowed[0] is Interval.D1
            assert set(allowed) <= {Interval.D1, Interval.W1}

    def test_an_unknown_period_falls_back_to_daily(self) -> None:
        assert finest_interval("17y") is Interval.D1


# --------------------------------------------------------------------------
# Seite: Auswahl und effektive Aufloesung


class TestGpChoices:
    def test_one_day_and_one_week_are_offered(self) -> None:
        values = [value for _label, value in gp.GP_PERIOD_CHOICES]
        assert values[:2] == ["1d", "5d"]
        assert [label for label, _v in gp.GP_PERIOD_CHOICES][:2] == ["1D", "1W"]

    def test_a_fitting_choice_is_kept(self) -> None:
        assert gp.effective_interval("1y", "1wk") is Interval.W1

    def test_an_impossible_choice_becomes_the_finest(self) -> None:
        """1 Minute fuer ein Jahr liefert die Quelle nicht."""
        assert gp.effective_interval("1y", "1m") is Interval.H1

    def test_no_choice_means_the_finest(self) -> None:
        assert gp.effective_interval("1d", "") is Interval.M1
        assert gp.effective_interval("1mo", None) is Interval.M5

    def test_unfitting_options_are_disabled_not_hidden(self) -> None:
        options = {o["value"]: o["disabled"] for o in gp.resolution_options("1y")}
        assert set(options) == {"1m", "5m", "15m", "1h", "1d", "1wk"}
        assert options["1m"] is True
        assert options["1h"] is False
        assert options["1wk"] is False

    def test_command_line_period_and_resolution(self) -> None:
        assert gp._period_from_args(("1d",)) == "1d"
        assert gp._period_from_args(("5d",)) == "5d"
        assert gp._resolution_from_args(("1h",)) == "1h"
        # "1d" ist ein Zeitraum, keine Aufloesung.
        assert gp._resolution_from_args(("1d",)) == gp.DEFAULT_RESOLUTION


# --------------------------------------------------------------------------
# Intraday-Darstellung


def _minute_series(template: Any) -> Any:
    """Zwei Handelstage in 1-Minuten-Kerzen, 13:30 bis 19:59 UTC."""
    days = [pd.Timestamp("2026-09-24", tz="UTC"), pd.Timestamp("2026-09-25", tz="UTC")]
    stamps = pd.DatetimeIndex(
        [
            ts
            for day in days
            for ts in pd.date_range(day + timedelta(hours=13, minutes=30), periods=390, freq="1min")
        ],
        name="date",
    )
    n = len(stamps)
    frame = pd.DataFrame(
        {
            "open": [100.0 + i * 0.01 for i in range(n)],
            "high": [100.5 + i * 0.01 for i in range(n)],
            "low": [99.5 + i * 0.01 for i in range(n)],
            "close": [100.2 + i * 0.01 for i in range(n)],
            "volume": [1000.0] * n,
        },
        index=stamps,
    )
    return template.model_copy(update={"interval": Interval.M1, "data": frame})


class TestSessionBreaks:
    def test_the_night_is_one_break(self) -> None:
        stamps = pd.DatetimeIndex(
            ["2026-09-24 15:58", "2026-09-24 15:59", "2026-09-25 09:30", "2026-09-25 09:31"]
        )
        (entry,) = session_breaks(stamps, timedelta(minutes=1))
        assert entry["values"] == ["2026-09-24T16:00:00"]
        # 16:00 bis 09:30 = 17.5 h
        assert entry["dvalue"] == int(17.5 * 3600 * 1000)

    def test_equal_nights_share_one_entry(self) -> None:
        """Plotly wird mit hunderten Eintraegen langsam."""
        sessions = [
            pd.date_range(f"2026-09-{day} 09:30", f"2026-09-{day} 15:59", freq="1min")
            for day in (22, 23, 24)
        ]
        stamps = pd.DatetimeIndex(sessions[0].append([sessions[1], sessions[2]]))
        breaks = session_breaks(stamps, timedelta(minutes=1))
        assert len(breaks) == 1
        assert len(breaks[0]["values"]) == 2  # type: ignore[arg-type]

    def test_a_missing_minute_is_not_hidden(self) -> None:
        """Eine kurze Luecke bleibt als ehrlicher Sprung sichtbar."""
        stamps = pd.DatetimeIndex(["2026-09-24 10:00", "2026-09-24 10:02"])
        assert session_breaks(stamps, timedelta(minutes=1)) == []


class TestLocalTime:
    def test_axis_shows_display_time(self) -> None:
        fig = go.Figure(go.Scatter(x=pd.DatetimeIndex(["2026-09-25 13:30"], tz="UTC"), y=[1]))
        fig.update_xaxes(title_text="Date (UTC)")
        localize_time_axis(fig, "Europe/Berlin")
        assert pd.Timestamp(fig.data[0].x[0]) == pd.Timestamp("2026-09-25 15:30")
        assert fig.layout.xaxis.title.text == "Time (Europe/Berlin)"

    def test_gaps_are_computed_in_display_time(self) -> None:
        fig = go.Figure(
            go.Scatter(
                x=pd.DatetimeIndex(["2026-09-24 19:59", "2026-09-25 13:30"], tz="UTC"), y=[1, 2]
            )
        )
        localize_time_axis(fig, "Europe/Berlin")
        hide_session_gaps(fig, timedelta(minutes=1))
        (entry,) = fig.layout.xaxis.rangebreaks
        assert entry.values == ("2026-09-24T22:00:00",)  # noqa: PD011 - Plotly-Attribut, kein pandas


class TestIntradayPage:
    @pytest.fixture
    def online(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppState:
        """Offline-Zustand, der sich als online ausgibt, mit Minutenreihe."""
        monkeypatch.setenv("BTLITE_PORTFOLIO_HOLDINGS_FILE", str(tmp_path / "h.json"))
        state = AppState(Settings())
        monkeypatch.setattr(type(state), "is_offline", property(lambda _s: False))
        original = state.history
        template = original("AAPL", period="1y", interval=Interval.D1)
        asked: list[Interval] = []

        def history(ticker: str, *, period: str = "1y", interval: Interval = Interval.D1) -> Any:
            asked.append(interval)
            if interval is Interval.M1:
                return _minute_series(template)
            return original(ticker, period=period, interval=interval)

        monkeypatch.setattr(state, "history", history)
        state.asked = asked  # type: ignore[attr-defined]
        return state

    @staticmethod
    def _graph(tree: Any) -> Any:
        found: list[Any] = []

        def walk(node: Any) -> None:
            if getattr(node, "_type", "") == "Graph":
                found.append(node)
            kids = getattr(node, "children", None)
            for kid in kids if isinstance(kids, list) else [kids]:
                if kid is not None and hasattr(kid, "_type"):
                    walk(kid)

        walk(tree)
        return found[0].figure

    def test_one_day_draws_minute_candles(self, online: AppState) -> None:
        tree = gp.render_charts(online, ["AAPL"], "1d", overlays=())
        assert online.asked == [Interval.M1]  # type: ignore[attr-defined]
        assert "(1m," in str(tree)

    def test_times_are_in_display_time_and_nights_hidden(self, online: AppState) -> None:
        fig = self._graph(gp.render_charts(online, ["AAPL"], "1d", overlays=()))
        assert pd.Timestamp(fig.data[0].x[0]) == pd.Timestamp("2026-09-24 15:30")
        assert fig.layout.xaxis.rangebreaks, "Die Nacht muss ausgeblendet sein"

    def test_no_range_selector_in_a_minute_chart(self, online: AppState) -> None:
        fig = self._graph(gp.render_charts(online, ["AAPL"], "1d", overlays=()))
        assert not fig.layout.xaxis.rangeselector.buttons

    def test_the_notice_names_time_zone_and_delay(self, online: AppState) -> None:
        text = str(gp.render_charts(online, ["AAPL"], "1d", overlays=()))
        assert "times in Europe/Berlin" in text
        assert "regular trading hours only" in text

    def test_setup_markers_are_off_zones_stay(
        self, online: AppState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from btlite.viz import setups as viz_setups

        calls: list[dict[str, Any]] = []
        original = viz_setups.add_setup_annotations

        def spy(fig: Any, report: Any, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return original(fig, report, **kwargs)

        monkeypatch.setattr(viz_setups, "add_setup_annotations", spy)
        gp.render_charts(online, ["AAPL"], "1d", overlays=("setups",))
        assert calls
        assert calls[0]["markers"] is False

    def test_the_page_preselects_the_finest(self, online: AppState) -> None:
        page = str(render_page(online, "GP", ["AAPL"], ("1d",)))
        assert "'value': '1m'" in page or "value='1m'" in page


class TestOffline:
    @pytest.fixture
    def state(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppState:
        monkeypatch.setenv("BTLITE_PORTFOLIO_HOLDINGS_FILE", str(tmp_path / "h.json"))
        return AppState(Settings())

    def test_one_day_offline_says_why_it_is_daily(self, state: AppState) -> None:
        text = str(gp.render_charts(state, ["AAPL"], "1d"))
        assert "(1d," in text
        assert "Offline: only daily candles" in text


# --------------------------------------------------------------------------
# Callback: Zeitraumwechsel setzt die Aufloesung zurueck


class _Ctx:
    def __init__(self, triggered: str) -> None:
        self.triggered_id = triggered


class TestCallback:
    def _callback(self, state: AppState, monkeypatch: pytest.MonkeyPatch, triggered: str) -> Any:
        import dash

        monkeypatch.setattr(dash, "ctx", _Ctx(triggered))
        app = create_app(state.settings, state)
        return next(
            getattr(e["callback"], "__wrapped__", e["callback"])
            for e in app.callback_map.values()
            if "gp-resolution" in str(e.get("inputs", ""))
        )

    @pytest.fixture
    def state(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppState:
        monkeypatch.setenv("BTLITE_PORTFOLIO_HOLDINGS_FILE", str(tmp_path / "h.json"))
        return AppState(Settings())

    def test_a_new_period_resets_to_the_finest(
        self, state: AppState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        callback = self._callback(state, monkeypatch, "gp-period")
        _chart, options, value = callback("1y", "candlestick", [], ["AAPL"], "1wk")
        assert value == "1d"  # offline die kleinste
        assert {o["value"] for o in options if not o["disabled"]} == {"1d", "1wk"}

    def test_a_chosen_resolution_stays_while_the_period_stays(
        self, state: AppState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        callback = self._callback(state, monkeypatch, "gp-resolution")
        _chart, _options, value = callback("1y", "candlestick", [], ["AAPL"], "1wk")
        assert value == "1wk"


# --------------------------------------------------------------------------
# Daten: kurzer Cache fuer Intraday


class TestIntradayCache:
    def test_provider_uses_the_short_ttl(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from btlite.core.cache import DataFrameCache
        from btlite.data.yfinance_provider import YFinanceProvider

        settings = Settings()
        settings.data.offline = False
        settings.cache.dir = tmp_path / "cache"
        provider = YFinanceProvider(settings, cache=DataFrameCache(settings.cache))
        seen: list[float | None] = []

        class _StopError(Exception):
            pass

        def spy(_key: str, ttl: float | None, _loader: Any) -> Any:
            seen.append(ttl)
            raise _StopError

        monkeypatch.setattr(provider._cache, "cached_df", spy)
        for interval in (Interval.M1, Interval.H1, Interval.D1):
            with pytest.raises(_StopError):
                provider.get_history("AAPL", interval=interval, period="5d")
        assert settings.cache.intraday_ttl_s == 60
        assert seen == [60, 60, settings.cache.prices_ttl_s]

    def test_state_does_not_keep_intraday_in_memory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BTLITE_PORTFOLIO_HOLDINGS_FILE", str(tmp_path / "h.json"))
        state = AppState(Settings())
        daily = state.provider.get_history("AAPL")
        calls: list[Interval] = []

        def fake(_ticker: str, *, interval: Interval, **_kw: Any) -> Any:
            calls.append(interval)
            return daily

        monkeypatch.setattr(state._provider, "get_history", fake)
        for _ in range(2):
            state.history("AAPL", period="1d", interval=Interval.M1)
            state.history("AAPL", period="1y", interval=Interval.D1)
        assert calls.count(Interval.M1) == 2, "Minutenkerzen muessen jedes Mal frisch sein"
        assert calls.count(Interval.D1) == 1
