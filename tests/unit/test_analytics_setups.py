"""Tests der Chart-Setup-Checkliste.

Jedes Setup bekommt konstruierte Kursreihen fuer "met", "not met" und
"insufficient data", dazu Grenzfaelle an den Schwellen und Fenstern. Die
Erwartungen stammen aus der Konstruktion der Reihe oder aus einer
unabhaengigen Rechnung mit ``analytics.technicals`` - nicht aus dem
Modul selbst.

Look-ahead: ein Swing zaehlt erst, wenn die Bars rechts davon vorliegen,
und der Bericht am Stichtag haengt nicht von spaeteren Bars ab.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from btlite.analytics import setups as su
from btlite.analytics import technicals as tech
from btlite.core.errors import InsufficientDataError
from btlite.data.fixtures import FixtureProvider
from btlite.models.common import DataSource, Provenance
from btlite.models.market import PriceSeries
from btlite.models.setups import (
    CrossDirection,
    PriceZone,
    SetupKey,
    SetupReport,
    SetupStatus,
    ZoneKind,
)

_PROVENANCE = Provenance(source=DataSource.COMPUTED, fetched_at=datetime(2026, 1, 1, tzinfo=UTC))


def make_series(
    close: list[float] | np.ndarray,
    *,
    volume: list[float] | np.ndarray | None = None,
    high: list[float] | np.ndarray | None = None,
    low: list[float] | np.ndarray | None = None,
) -> PriceSeries:
    """Baut eine Tageskursreihe aus Schlusskursen; Hoch/Tief default = Schluss."""
    values = np.asarray(close, dtype="float64")
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B", tz="UTC", name="date")
    frame = pd.DataFrame(
        {
            "open": values,
            "high": values if high is None else np.asarray(high, dtype="float64"),
            "low": values if low is None else np.asarray(low, dtype="float64"),
            "close": values,
            "volume": (
                np.full(len(values), 1_000_000.0)
                if volume is None
                else np.asarray(volume, dtype="float64")
            ),
        },
        index=idx,
    )
    return PriceSeries(ticker="TEST", adjusted=True, data=frame, provenance=_PROVENANCE)


def linear(n: int, start: float, slope: float) -> np.ndarray:
    """Linearer Kursverlauf."""
    return start + slope * np.arange(n, dtype="float64")


# --------------------------------------------------------------------------
# Setup 1: Trend-Pullback
# --------------------------------------------------------------------------


class TestPullback:
    def test_met_in_gentle_uptrend(self):
        # SMA 50 einer Geraden liegt 24.5 Steigungen unter dem Kurs.
        series = make_series(linear(300, 100.0, 0.1))
        check = su.check_pullback(series)
        price, sma50 = 129.9, 129.9 - 2.45
        assert check.status is SetupStatus.MET
        assert check.measurement("dist_sma_50") == pytest.approx((price - sma50) / sma50)
        assert check.measurement("dist_sma_200") > 0

    def test_measurements_match_indicator_layer(self):
        series = make_series(linear(300, 100.0, 0.1))
        check = su.check_pullback(series)
        sma200 = float(tech.sma(series.close, 200).iloc[-1])
        assert check.measurement("dist_sma_200") == pytest.approx(129.9 / sma200 - 1.0)

    def test_not_met_when_far_above_50_day(self):
        series = make_series(linear(300, 100.0, 1.0))  # Abstand rund 6.5 Prozent
        check = su.check_pullback(series)
        assert check.status is SetupStatus.NOT_MET
        assert check.measurement("dist_sma_50") > su.PULLBACK_BAND

    def test_band_edges(self):
        """Knapp innerhalb (2.5 %) met, knapp ausserhalb (3.5 %) nicht."""

        def slope_for(target: float, price: float = 200.0) -> float:
            # dist = 24.5 s / (price - 24.5 s)  =>  s = target * price / (24.5 (1 + target))
            return target * price / (24.5 * (1.0 + target))

        inside = linear(300, 200.0 - 299 * slope_for(0.025), slope_for(0.025))
        outside = linear(300, 200.0 - 299 * slope_for(0.035), slope_for(0.035))
        assert su.check_pullback(make_series(inside)).measurement("dist_sma_50") == (
            pytest.approx(0.025)
        )
        assert su.check_pullback(make_series(inside)).status is SetupStatus.MET
        assert su.check_pullback(make_series(outside)).status is SetupStatus.NOT_MET

    def test_below_50_day_within_band_is_met(self):
        """Der Abstand zaehlt beidseitig: leicht unter der 50-Tage-Linie."""
        values = np.concatenate([linear(290, 100.0, 0.2), linear(10, 157.8, -0.5)])
        check = su.check_pullback(make_series(values))
        dist = check.measurement("dist_sma_50")
        assert dist is not None
        assert -su.PULLBACK_BAND <= dist < 0
        assert check.status is SetupStatus.MET

    def test_downtrend_is_not_met(self):
        check = su.check_pullback(make_series(linear(300, 400.0, -0.1)))
        assert check.status is SetupStatus.NOT_MET
        assert "no uptrend" in check.explanation

    def test_requires_50_above_200(self):
        """Kurs ueber der 200er und nahe der 50er, aber 50er unter 200er: nicht met."""
        values = np.concatenate([np.full(250, 100.0), np.full(49, 99.0), [101.0]])
        series = make_series(values)
        sma50 = float(tech.sma(series.close, 50).iloc[-1])
        sma200 = float(tech.sma(series.close, 200).iloc[-1])
        assert values[-1] > sma200
        assert sma50 < sma200
        check = su.check_pullback(series)
        assert abs(check.measurement("dist_sma_50")) <= su.PULLBACK_BAND
        assert check.status is SetupStatus.NOT_MET

    def test_insufficient_below_200_bars(self):
        check = su.check_pullback(make_series(linear(199, 100.0, 0.1)))
        assert check.status is SetupStatus.INSUFFICIENT_DATA
        assert check.measurements == []
        assert "200" in check.explanation


# --------------------------------------------------------------------------
# Setup 2: RSI-Erholung
# --------------------------------------------------------------------------


def rsi_recovery_values(bars_after_bounce: int) -> np.ndarray:
    """Abwaertslauf, ein kraeftiger Aufwaertstag, danach flach.

    Flache Kurse lassen den Wilder-RSI unveraendert, das Tief liegt damit
    genau ``bars_after_bounce`` Bars vor dem Stichtag.
    """
    base = linear(40, 100.0, 0.1)  # Anlauf mit leichtem Anstieg
    decline = linear(20, 103.0, -1.5)[1:]  # 19 Verlusttage
    bounce = np.array([decline[-1] + 12.0])
    flat = np.full(bars_after_bounce - 1, bounce[0])
    return np.concatenate([base, [103.0], decline, bounce, flat])


class TestRsiRecovery:
    def test_constructed_series_has_the_intended_shape(self):
        values = rsi_recovery_values(5)
        rsi = tech.rsi(pd.Series(values), 14)
        low_pos = len(values) - 1 - 5
        assert rsi.iloc[low_pos] < 30
        assert rsi.iloc[-1] >= 30
        assert rsi.iloc[-1] == pytest.approx(rsi.iloc[low_pos + 1])

    def test_met_after_recovery(self):
        values = rsi_recovery_values(5)
        check = su.check_rsi_recovery(make_series(values))
        rsi = tech.rsi(pd.Series(values), 14)
        assert check.status is SetupStatus.MET
        assert check.measurement("rsi") == pytest.approx(float(rsi.iloc[-1]))
        assert check.measurement("rsi_min") == pytest.approx(float(rsi.iloc[-11:-1].min()))
        assert check.measurement("bars_since_low") == 5
        assert check.event_at is not None
        assert check.event_price == pytest.approx(values[-6])

    def test_low_exactly_ten_bars_back_counts(self):
        check = su.check_rsi_recovery(make_series(rsi_recovery_values(10)))
        assert check.status is SetupStatus.MET
        assert check.measurement("bars_since_low") == 10

    def test_low_eleven_bars_back_is_outside_the_window(self):
        check = su.check_rsi_recovery(make_series(rsi_recovery_values(11)))
        assert check.status is SetupStatus.NOT_MET
        assert check.measurement("rsi_min") >= su.RSI_OVERSOLD

    def test_still_oversold_today_is_not_met(self):
        values = np.concatenate([linear(40, 100.0, 0.1), linear(25, 103.0, -1.5)])
        check = su.check_rsi_recovery(make_series(values))
        assert check.measurement("rsi") < su.RSI_OVERSOLD
        assert check.status is SetupStatus.NOT_MET

    def test_never_oversold_is_not_met(self):
        rng = np.random.default_rng(1)
        values = 100.0 + np.cumsum(np.abs(rng.normal(0.2, 0.1, 80)))
        check = su.check_rsi_recovery(make_series(values))
        assert check.status is SetupStatus.NOT_MET
        assert check.event_at is None

    def test_insufficient_below_25_bars(self):
        check = su.check_rsi_recovery(make_series(linear(24, 100.0, 0.1)))
        assert check.status is SetupStatus.INSUFFICIENT_DATA
        assert su.check_rsi_recovery(make_series(linear(25, 100.0, 0.1))).status is not (
            SetupStatus.INSUFFICIENT_DATA
        )


# --------------------------------------------------------------------------
# Setup 3: Ausbruch
# --------------------------------------------------------------------------


def breakout_values(last: float, *, n: int = 300) -> np.ndarray:
    """Seitwaertsbewegung zwischen 90 und 100, am Ende der Kurs ``last``."""
    body = 95.0 + 5.0 * np.sin(np.arange(n - 1) / 7.0)
    return np.concatenate([body, [last]])


class TestBreakout:
    def test_met_on_new_high_with_volume(self):
        values = breakout_values(105.0)
        volume = np.full(300, 1_000_000.0)
        volume[-1] = 2_000_000.0
        check = su.check_breakout(make_series(values, volume=volume))
        prior_high = float(values[-253:-1].max())
        assert check.status is SetupStatus.MET
        assert check.measurement("dist_prior_high") == pytest.approx(105.0 / prior_high - 1.0)
        assert check.measurement("volume_ratio") == pytest.approx(2.0)
        assert check.event_price == pytest.approx(105.0)

    def test_not_met_without_volume(self):
        values = breakout_values(105.0)
        volume = np.full(300, 1_000_000.0)
        volume[-1] = 600_000.0
        check = su.check_breakout(make_series(values, volume=volume))
        assert check.status is SetupStatus.NOT_MET
        assert check.measurement("volume_ratio") == pytest.approx(0.6)

    def test_volume_equal_to_average_is_not_enough(self):
        check = su.check_breakout(make_series(breakout_values(105.0)))
        assert check.measurement("volume_ratio") == pytest.approx(1.0)
        assert check.status is SetupStatus.NOT_MET

    def test_not_met_below_prior_high(self):
        volume = np.full(300, 1_000_000.0)
        volume[-1] = 3_000_000.0
        check = su.check_breakout(make_series(breakout_values(99.0), volume=volume))
        assert check.status is SetupStatus.NOT_MET
        assert check.measurement("dist_prior_high") < 0
        assert check.event_at is None

    @pytest.mark.parametrize(("bars_back", "expected"), [(253, "met"), (252, "not_met")])
    def test_lookback_is_252_bars_excluding_today(self, bars_back, expected):
        """Ein altes Hoch von 120 zaehlt nur innerhalb der 252 Vorbars."""
        values = breakout_values(105.0)
        values[-1 - bars_back] = 120.0
        volume = np.full(300, 1_000_000.0)
        volume[-1] = 2_000_000.0
        check = su.check_breakout(make_series(values, volume=volume))
        assert check.status.value == expected

    def test_volume_average_uses_63_prior_bars(self):
        """Hohes Volumen genau 64 Bars zurueck zaehlt nicht mehr zum Schnitt."""
        values = breakout_values(105.0)
        volume = np.full(300, 1_000_000.0)
        volume[-1] = 1_500_000.0
        volume[-65] = 100_000_000.0  # 64 Bars vor dem Stichtag, ausserhalb
        assert su.check_breakout(make_series(values, volume=volume)).status is SetupStatus.MET
        volume[-64] = 100_000_000.0  # 63 Bars vor dem Stichtag, innerhalb
        assert su.check_breakout(make_series(values, volume=volume)).status is (SetupStatus.NOT_MET)

    def test_insufficient_below_253_bars(self):
        check = su.check_breakout(make_series(breakout_values(105.0, n=252)))
        assert check.status is SetupStatus.INSUFFICIENT_DATA

    def test_missing_volume_is_insufficient_not_zero(self):
        volume = np.full(300, 1_000_000.0)
        volume[-10] = np.nan
        check = su.check_breakout(make_series(breakout_values(105.0), volume=volume))
        assert check.status is SetupStatus.INSUFFICIENT_DATA
        assert check.measurement("volume_ratio") is None


# --------------------------------------------------------------------------
# Setup 4: Swings, Zonen, MACD
# --------------------------------------------------------------------------


def v_shape(low_at_from_end: int, *, n: int = 200) -> np.ndarray:
    """Fallend bis zu einem Tief ``low_at_from_end`` Bars vor dem Ende, dann steigend."""
    pos = n - 1 - low_at_from_end
    idx = np.arange(n, dtype="float64")
    return 100.0 + np.abs(idx - pos)


class TestSwings:
    def test_swing_low_confirmed_after_five_bars(self):
        swings = su.confirmed_swings(make_series(v_shape(5)))
        lows = [s for s in swings if not s.is_high]
        assert [s.position for s in lows] == [194]
        assert lows[0].level == pytest.approx(100.0)

    @pytest.mark.parametrize("bars_after", [0, 1, 3, 4])
    def test_unconfirmed_swing_does_not_count(self, bars_after):
        """Look-ahead: ohne fuenf Bars rechts ist das Tief kein Swing."""
        swings = su.confirmed_swings(make_series(v_shape(bars_after)))
        assert [s for s in swings if not s.is_high] == []

    def test_swing_appears_once_confirmation_arrives(self):
        full = v_shape(5)
        assert not [s for s in su.confirmed_swings(make_series(full[:-1])) if not s.is_high]
        assert [s for s in su.confirmed_swings(make_series(full)) if not s.is_high]

    def test_swing_high_uses_high_column(self):
        close = np.full(200, 100.0)
        high = close + 1.0
        high[150] = 110.0
        swings = su.confirmed_swings(make_series(close, high=high, low=close - 1.0))
        highs = [s for s in swings if s.is_high]
        assert [(s.position, s.level) for s in highs] == [(150, 110.0)]

    def test_only_last_126_bars(self):
        close = np.full(300, 100.0)
        low = close - 1.0
        low[300 - 127] = 80.0  # Position 173: ausserhalb des Fensters
        low[300 - 120] = 85.0  # Position 180: innerhalb
        swings = su.confirmed_swings(make_series(close, high=close + 1.0, low=low))
        assert [s.level for s in swings if not s.is_high] == [85.0]

    def test_rejects_bad_parameters(self):
        with pytest.raises(ValueError, match="positiv"):
            su.confirmed_swings(make_series(v_shape(10)), order=0)


def with_lows(lows: list[float], *, highs: tuple[float, ...] = ()) -> PriceSeries:
    """Deutliche Tiefs (und Hochs) im Fenster, je 15 Bars auseinander, Kurs am Ende 120."""
    close = np.full(200, 110.0)
    close[-1] = 120.0
    low = close - 0.5
    high = close + 0.5
    for i, level in enumerate(lows):
        low[90 + 15 * i] = level
    for i, level in enumerate(highs):
        high[97 + 15 * i] = level
    return make_series(close, high=high, low=low)


def two_lows(first: float, second: float) -> PriceSeries:
    """Zwei deutliche Tiefs im Fenster."""
    return with_lows([first, second])


class TestZones:
    def test_levels_within_tolerance_merge(self):
        zones = su.price_zones(two_lows(100.0, 101.0))
        lows = [z for z in zones if z.upper <= 101.0]
        assert len(lows) == 1
        assert lows[0].touches == 2
        assert lows[0].level == pytest.approx(100.5)
        assert (lows[0].lower, lows[0].upper) == (100.0, 101.0)

    def test_levels_beyond_tolerance_stay_apart(self):
        zones = su.price_zones(two_lows(100.0, 102.0))
        assert [z.level for z in zones if z.level < 105.0] == [100.0, 102.0]

    def test_tolerance_edges(self):
        zones = su.price_zones(two_lows(100.0, 101.4))
        assert [z.touches for z in zones if z.level < 105.0] == [2]
        zones = su.price_zones(two_lows(100.0, 101.6))
        assert len([z for z in zones if z.level < 105.0]) == 2

    def test_tolerance_is_anchored_at_the_lowest_level(self):
        """Keine Verkettung: 100, 101.2, 102.2 ergibt zwei Zonen, nicht eine."""
        zones = su.price_zones(with_lows([100.0, 101.2, 102.2]))
        assert [z.touches for z in zones if z.level < 105.0] == [2, 1]

    def test_kind_relative_to_price(self):
        zones = su.price_zones(with_lows([100.0], highs=(130.0,)))
        kinds = {round(z.level): z.kind for z in zones}
        assert kinds[100] is ZoneKind.SUPPORT
        assert kinds[130] is ZoneKind.RESISTANCE

    def test_no_swings_no_zones(self):
        assert su.price_zones(make_series(linear(200, 100.0, 0.1))) == []

    def test_negative_tolerance_rejected(self):
        with pytest.raises(ValueError, match="negativ"):
            su.price_zones(two_lows(100.0, 101.0), tolerance=-0.01)


def macd_turn(bars_since_turn: int) -> np.ndarray:
    """Langer Abwaertslauf, dann Anstieg - erzwingt eine MACD-Kreuzung nach oben."""
    down = linear(150, 200.0, -0.5)
    up = linear(bars_since_turn + 1, down[-1], 1.5)[1:]
    return np.concatenate([down, up])


def expected_cross(values: np.ndarray) -> tuple[str, int | None]:
    """Unabhaengige Rechnung: juengste Kreuzung in den letzten 5 Bars."""
    frame = tech.macd(pd.Series(values))
    line, signal = frame["macd"], frame["signal"]
    diff = np.sign((line - signal).to_numpy())
    for back in range(5):
        i = len(values) - 1 - back
        if diff[i] > 0 >= diff[i - 1]:
            return "up", back
        if diff[i] < 0 <= diff[i - 1]:
            return "down", back
    return "none", None


class TestMacdCross:
    @pytest.mark.parametrize("turn", [3, 6, 10, 25])
    def test_matches_independent_computation(self, turn):
        values = macd_turn(turn)
        cross = su.macd_cross(make_series(values))
        direction, bars = expected_cross(values)
        assert cross.direction.value == direction
        assert cross.bars_since == bars

    def test_finds_an_up_cross_somewhere(self):
        found = [su.macd_cross(make_series(macd_turn(t))) for t in range(1, 15)]
        ups = [c for c in found if c.direction is CrossDirection.UP]
        assert ups
        assert all(0 <= (c.bars_since or 0) < su.MACD_CROSS_WINDOW for c in ups)
        assert all(c.price is not None and c.at is not None for c in ups)

    def test_down_cross(self):
        values = np.concatenate([linear(120, 100.0, 0.5), linear(4, 159.5, -2.0)[1:]])
        direction, bars = expected_cross(values)
        cross = su.macd_cross(make_series(values))
        assert (cross.direction.value, cross.bars_since) == (direction, bars)
        assert (direction, bars) == ("down", 2)

    def test_too_short_raises(self):
        with pytest.raises(InsufficientDataError):
            su.macd_cross(make_series(linear(su.MACD_MIN_BARS - 1, 100.0, 0.1)))


def up_cross(bars: int = 1) -> su.MacdCross:
    return su.MacdCross(direction=CrossDirection.UP, bars_since=bars, at=None, price=None)


def support_zone(level: float) -> PriceZone:
    return PriceZone(level=level, lower=level, upper=level, kind=ZoneKind.SUPPORT, touches=2)


class TestSupportMacd:
    @pytest.fixture
    def flat(self) -> PriceSeries:
        return make_series(np.full(200, 100.0))

    def test_met_near_support_with_cross_up(self, flat):
        check = su.check_support_macd(flat, [support_zone(98.0)], up_cross())
        assert check.status is SetupStatus.MET
        assert check.measurement("dist_support") == pytest.approx(100.0 / 98.0 - 1.0)
        assert check.measurement("bars_since_cross") == 1

    def test_not_met_far_above_support(self, flat):
        check = su.check_support_macd(flat, [support_zone(96.0)], up_cross())
        assert check.measurement("dist_support") > su.SUPPORT_BAND
        assert check.status is SetupStatus.NOT_MET

    def test_band_edge(self, flat):
        inside = su.check_support_macd(flat, [support_zone(100.0 / 1.029)], up_cross())
        outside = su.check_support_macd(flat, [support_zone(100.0 / 1.031)], up_cross())
        assert inside.status is SetupStatus.MET
        assert outside.status is SetupStatus.NOT_MET

    @pytest.mark.parametrize("direction", [CrossDirection.DOWN, CrossDirection.NONE])
    def test_not_met_without_up_cross(self, flat, direction):
        cross = su.MacdCross(
            direction=direction,
            bars_since=None if direction is CrossDirection.NONE else 2,
            at=None,
            price=None,
        )
        check = su.check_support_macd(flat, [support_zone(99.0)], cross)
        assert check.status is SetupStatus.NOT_MET

    def test_no_support_is_not_met_with_none_measurement(self, flat):
        resistance = PriceZone(
            level=105.0, lower=105.0, upper=105.0, kind=ZoneKind.RESISTANCE, touches=1
        )
        check = su.check_support_macd(flat, [resistance], up_cross())
        assert check.status is SetupStatus.NOT_MET
        assert check.measurement("dist_support") is None
        assert check.measurement("dist_resistance") == pytest.approx(100.0 / 105.0 - 1.0)

    def test_nearest_support_is_the_highest_below(self, flat):
        zones = [support_zone(90.0), support_zone(99.0)]
        check = su.check_support_macd(flat, zones, up_cross())
        assert check.measurement("support") == pytest.approx(99.0)

    def test_computes_zones_and_cross_itself(self):
        check = su.check_support_macd(make_series(macd_turn(3)))
        assert check.status in {SetupStatus.MET, SetupStatus.NOT_MET}

    def test_insufficient_below_131_bars(self):
        check = su.check_support_macd(make_series(np.full(130, 100.0)))
        assert check.status is SetupStatus.INSUFFICIENT_DATA


# --------------------------------------------------------------------------
# Gesamtbericht und Look-ahead
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_prices() -> dict[str, PriceSeries]:
    provider = FixtureProvider()
    return {t: provider.get_history(t, period=None) for t in ("AAPL", "WMT", "SPY")}


def truncate(series: PriceSeries, length: int) -> PriceSeries:
    return series.model_copy(update={"data": series.data.iloc[:length]})


class TestSetupReport:
    def test_all_four_checks_in_fixed_order(self, fixture_prices):
        report = su.setup_report(fixture_prices["AAPL"])
        assert [c.key for c in report.checks] == list(SetupKey)
        assert report.as_of.tzinfo is not None
        assert report.price == pytest.approx(float(fixture_prices["AAPL"].close.iloc[-1]))

    def test_no_overall_verdict_in_the_contract(self):
        fields = set(SetupReport.model_fields)
        assert not fields & {"score", "verdict", "rating", "signal", "recommendation"}

    def test_zones_sorted_and_nearest_consistent(self, fixture_prices):
        report = su.setup_report(fixture_prices["WMT"])
        levels = [z.level for z in report.zones]
        assert levels == sorted(levels)
        supports = [z.level for z in report.zones if z.kind is ZoneKind.SUPPORT]
        assert report.nearest_support == (max(supports) if supports else None)
        if report.dist_support is not None:
            assert report.dist_support >= 0
        if report.dist_resistance is not None:
            assert report.dist_resistance < 0

    def test_short_history_is_insufficient_not_not_met(self, fixture_prices):
        report = su.setup_report(truncate(fixture_prices["AAPL"], 30))
        statuses = {c.key: c.status for c in report.checks}
        assert statuses[SetupKey.RSI_RECOVERY] is not SetupStatus.INSUFFICIENT_DATA
        for key in (SetupKey.PULLBACK, SetupKey.BREAKOUT, SetupKey.SUPPORT_MACD):
            assert statuses[key] is SetupStatus.INSUFFICIENT_DATA
        assert report.zones == []
        assert report.dist_support is None

    def test_empty_series_raises(self, fixture_prices):
        with pytest.raises(InsufficientDataError):
            su.setup_report(truncate(fixture_prices["AAPL"], 0))

    def test_missing_high_low_fall_back_to_close(self, fixture_prices):
        series = fixture_prices["SPY"]
        data = series.data.copy()
        data.iloc[-50, data.columns.get_loc("high")] = np.nan
        report = su.setup_report(series.model_copy(update={"data": data}))
        assert len(report.checks) == 4

    @pytest.mark.parametrize("cut", [400, 700, 1000])
    def test_future_bars_do_not_change_the_result_at_t(self, fixture_prices, cut):
        """Look-ahead: Bars nach t veraendern, anhaengen, abschneiden - Ergebnis gleich."""
        series = fixture_prices["AAPL"]
        at_t = su.setup_report(truncate(series, cut))
        future = series.data.copy()
        future.iloc[cut:, :5] = future.iloc[cut:, :5] * 3.0  # Zukunft grob verfaelscht
        tampered = series.model_copy(update={"data": future})
        again = su.setup_report(truncate(tampered, cut))
        assert again.model_dump() == at_t.model_dump()

    def test_zone_from_swing_needs_confirmation_in_report(self):
        """Ein Tief 3 Bars vor dem Stichtag erzeugt noch keine Zone."""
        base = np.full(200, 110.0)
        low = base - 0.5
        low[-4] = 90.0
        unconfirmed = su.setup_report(make_series(base, high=base + 0.5, low=low))
        assert all(z.lower > 90.0 for z in unconfirmed.zones)
        extended = np.full(202, 110.0)
        low2 = extended - 0.5
        low2[-6] = 90.0
        confirmed = su.setup_report(make_series(extended, high=extended + 0.5, low=low2))
        assert any(z.lower == pytest.approx(90.0) for z in confirmed.zones)

    def test_explanations_are_neutral(self, fixture_prices):
        for series in fixture_prices.values():
            for check in su.setup_report(series).checks:
                text = check.explanation.lower()
                for word in ("buy", "sell", "recommend", "entry", "bullish", "bearish"):
                    assert word not in text
