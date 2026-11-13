"""Tests der Chart-Aufloesung 1 Tag / 1 Woche.

Wochenkerzen entstehen lokal aus den Tageskerzen
(:func:`btlite.analytics.series.resample_ohlcv`): keine weitere Anfrage an
die gedrosselte Quelle, und offline verfuegbar, wo nur Tagesdaten liegen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from btlite.analytics.series import resample_ohlcv
from btlite.config import Settings
from btlite.core.errors import InsufficientDataError
from btlite.models.common import Interval
from btlite.models.market import PriceSeries
from btlite.ui.app import create_app, render_page
from btlite.ui.pages import gp
from btlite.ui.state import AppState

if TYPE_CHECKING:
    from pathlib import Path


def _daily(days: list[str], **extra: list[float]) -> PriceSeries:
    """Tagesreihe mit bekannten Werten: Kurs = laufende Nummer."""
    n = len(days)
    data = {
        "open": [10.0 + i for i in range(n)],
        "high": [20.0 + i for i in range(n)],
        "low": [5.0 + i for i in range(n)],
        "close": [15.0 + i for i in range(n)],
        "volume": [100.0 * (i + 1) for i in range(n)],
        **extra,
    }
    frame = pd.DataFrame(data, index=pd.DatetimeIndex(days, tz="UTC", name="date"))
    from btlite.data.fixtures import FixtureProvider

    template = FixtureProvider(settings=Settings()).get_history("AAPL")
    return template.model_copy(update={"data": frame})


# Mo 31.08. bis Fr 04.09. und Mo 07.09. bis Fr 11.09.2026
_TWO_WEEKS = (
    [f"2026-08-{d}" for d in (31,)]
    + [f"2026-09-0{d}" for d in range(1, 5)]
    + [f"2026-09-{d:02d}" for d in range(7, 12)]
)


class TestWeeklyCandles:
    def test_one_candle_per_week_labelled_monday(self) -> None:
        weekly = resample_ohlcv(_daily(_TWO_WEEKS), Interval.W1)
        assert weekly.interval is Interval.W1
        assert list(weekly.data.index) == [
            pd.Timestamp("2026-08-31", tz="UTC"),
            pd.Timestamp("2026-09-07", tz="UTC"),
        ]

    def test_ohlcv_rules(self) -> None:
        weekly = resample_ohlcv(_daily(_TWO_WEEKS), Interval.W1).data
        first = weekly.iloc[0]
        assert first["open"] == 10.0  # erster Tag
        assert first["high"] == 24.0  # Maximum der fuenf Tage
        assert first["low"] == 5.0  # Minimum
        assert first["close"] == 19.0  # letzter Tag
        assert first["volume"] == 100.0 + 200 + 300 + 400 + 500

    def test_a_week_without_trading_days_is_dropped(self) -> None:
        days = ["2026-08-31", "2026-09-01", "2026-09-14", "2026-09-15"]
        weekly = resample_ohlcv(_daily(days), Interval.W1).data
        assert len(weekly) == 2, "Die leere Woche dazwischen darf keine Kerze sein"
        assert weekly["close"].notna().all()

    def test_dividends_add_up_and_splits_multiply(self) -> None:
        days = _TWO_WEEKS[:5]
        weekly = resample_ohlcv(
            _daily(
                days,
                dividends=[0.0, 0.25, 0.0, 0.25, 0.0],
                stock_splits=[0.0, 2.0, 0.0, 0.0, 4.0],
            ),
            Interval.W1,
        ).data
        assert weekly["dividends"].iloc[0] == pytest.approx(0.5)
        # 2 und 4: Produkt 8. Summe (mit oder ohne Startwert 1) gaebe 6 oder 7.
        assert weekly["stock_splits"].iloc[0] == pytest.approx(8.0)

    def test_a_week_without_split_stays_zero(self) -> None:
        weekly = resample_ohlcv(_daily(_TWO_WEEKS[:5], stock_splits=[0.0] * 5), Interval.W1).data
        assert weekly["stock_splits"].iloc[0] == 0.0

    def test_daily_is_returned_unchanged(self) -> None:
        daily = _daily(_TWO_WEEKS)
        assert resample_ohlcv(daily, Interval.D1) is daily

    def test_the_input_is_not_mutated(self) -> None:
        daily = _daily(_TWO_WEEKS)
        before = daily.data.copy()
        resample_ohlcv(daily, Interval.W1)
        pd.testing.assert_frame_equal(daily.data, before)

    def test_only_daily_input_is_accepted(self) -> None:
        weekly = resample_ohlcv(_daily(_TWO_WEEKS), Interval.W1)
        with pytest.raises(InsufficientDataError):
            resample_ohlcv(weekly, Interval.W1)

    def test_unsupported_targets_are_rejected(self) -> None:
        with pytest.raises(InsufficientDataError):
            resample_ohlcv(_daily(_TWO_WEEKS), Interval.H1)


class TestExchangeTimestamps:
    """yfinance stempelt Tageskerzen auf Mitternacht der Boersenzeit.

    Live gesehen: SAP.DE 22:00 UTC am Vortag, 7203.T 15:00 UTC am Vortag,
    AAPL 04:00 UTC. Die Montags-Sitzung von SAP steht also auf Sonntag
    22:00 UTC - sie darf nicht in die Vorwoche fallen.
    """

    @pytest.mark.parametrize(
        ("stamp_hour", "previous_day"),
        [(22, True), (15, True), (11, True), (4, False), (5, False), (0, False)],
        ids=["frankfurt", "tokyo", "auckland", "new-york-summer", "new-york-winter", "utc"],
    )
    def test_monday_session_lands_in_its_own_week(
        self, stamp_hour: int, previous_day: bool
    ) -> None:
        sessions = pd.date_range("2026-09-07", "2026-09-18", freq="B")  # zwei Wochen Mo-Fr
        offset = pd.Timedelta(days=1 if previous_day else 0) - pd.Timedelta(hours=stamp_hour)
        stamps = [(d - offset).strftime("%Y-%m-%d %H:%M") for d in sessions]
        weekly = resample_ohlcv(_daily(stamps), Interval.W1).data
        assert list(weekly.index) == [
            pd.Timestamp("2026-09-07", tz="UTC"),
            pd.Timestamp("2026-09-14", tz="UTC"),
        ]
        # Montag (erste Sitzung) eroeffnet die erste Woche, Freitag schliesst sie.
        assert weekly["open"].iloc[0] == 10.0
        assert weekly["close"].iloc[0] == 19.0


class TestAgainstRealData:
    """Gegenprobe auf den Fixture-Kursen ueber einen unabhaengigen Weg."""

    def test_weekly_close_is_the_last_daily_close_of_the_iso_week(self) -> None:
        from btlite.data.fixtures import FixtureProvider

        daily = FixtureProvider(settings=Settings()).get_history("AAPL")
        weekly = resample_ohlcv(daily, Interval.W1).data

        iso = daily.data.index.isocalendar()
        by_hand = daily.data.groupby([iso.year, iso.week])
        assert len(weekly) == by_hand.ngroups
        assert list(weekly["close"]) == pytest.approx(list(by_hand["close"].last()))
        assert list(weekly["high"]) == pytest.approx(list(by_hand["high"].max()))
        assert list(weekly["volume"]) == pytest.approx(list(by_hand["volume"].sum()))


class TestGpPage:
    @pytest.fixture
    def state(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppState:
        monkeypatch.setenv("BTLITE_PORTFOLIO_HOLDINGS_FILE", str(tmp_path / "h.json"))
        return AppState(Settings())

    def test_the_page_offers_both_resolutions(self, state: AppState) -> None:
        page = str(render_page(state, "GP", ["AAPL"], ()))
        assert gp.RESOLUTION_INPUT in page
        assert "1 day" in page
        assert "1 week" in page

    def test_daily_stays_the_default(self, state: AppState) -> None:
        text = str(gp.render_charts(state, ["AAPL"], "1y"))
        assert "(1d," in text
        assert "Weekly candles" not in text

    def test_weekly_charts_weekly_candles(self, state: AppState) -> None:
        text = str(gp.render_charts(state, ["AAPL"], "1y", resolution="1wk"))
        assert "(1wk," in text
        assert "Weekly candles" in text
        assert "SMA 50 spans 50 weeks" in text

    def test_weekly_has_about_a_fifth_of_the_candles(self, state: AppState) -> None:
        def candles(tree: Any) -> int:
            graphs: list[Any] = []

            def walk(node: Any) -> None:
                if getattr(node, "_type", "") == "Graph":
                    graphs.append(node)
                kids = getattr(node, "children", None)
                for kid in kids if isinstance(kids, list) else [kids]:
                    if kid is not None and hasattr(kid, "_type"):
                        walk(kid)

            walk(tree)
            return len(graphs[0].figure.data[0].x)

        daily = candles(gp.render_charts(state, ["AAPL"], "1y", overlays=()))
        weekly = candles(gp.render_charts(state, ["AAPL"], "1y", overlays=(), resolution="1wk"))
        assert daily / 6 < weekly < daily / 4

    def test_the_comparison_follows_the_resolution(self, state: AppState) -> None:
        """Der Vergleichschart hat so viele Punkte wie die Wochenkerzen."""

        def comparison_points(resolution: str) -> int:
            tree = gp.render_charts(
                state, ["AAPL", "MSFT"], "1y", overlays=(), resolution=resolution
            )
            graphs: list[Any] = []

            def walk(node: Any) -> None:
                if getattr(node, "_type", "") == "Graph":
                    graphs.append(node)
                kids = getattr(node, "children", None)
                for kid in kids if isinstance(kids, list) else [kids]:
                    if kid is not None and hasattr(kid, "_type"):
                        walk(kid)

            walk(tree)
            # Der Vergleich steht vor den Einzelcharts.
            return len(graphs[0].figure.data[0].x)

        daily, weekly = comparison_points("1d"), comparison_points("1wk")
        assert daily / 6 < weekly < daily / 4

    def test_an_unknown_resolution_falls_back_to_daily(self, state: AppState) -> None:
        text = str(gp.render_charts(state, ["AAPL"], "1y", resolution="7h"))
        assert "(1d," in text

    def test_command_line_sets_the_resolution(self, state: AppState) -> None:
        page = str(render_page(state, "GP", ["AAPL"], ("5y", "1wk")))
        assert "(1wk," in page

    def test_the_callback_takes_the_resolution(self, state: AppState) -> None:
        app = create_app(state.settings, state)
        callback = next(
            getattr(e["callback"], "__wrapped__", e["callback"])
            for e in app.callback_map.values()
            if "gp-resolution" in str(e.get("inputs", ""))
        )
        assert "(1wk," in str(callback("1y", "candlestick", [], ["AAPL"], "1wk"))
        # Alte Aufrufform ohne Aufloesung bleibt gueltig.
        assert "(1d," in str(callback("1y", "candlestick", [], ["AAPL"]))
