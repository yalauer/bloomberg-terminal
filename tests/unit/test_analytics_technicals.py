"""Tests der technischen Indikatoren.

Schwerpunkte:

1. Werte stimmen, geprueft gegen unabhaengige Handrechnung.
2. Kein Look-ahead-Bias: ein Wert an Position t benutzt nur Daten bis t.
3. Die Anlaufphase bleibt ``NaN`` und wird nicht aufgefuellt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btlite.analytics import technicals as tech
from btlite.core.errors import InsufficientDataError
from btlite.data.fixtures import FixtureProvider
from btlite.models.common import BOLLINGER_COLUMNS, MACD_COLUMNS


@pytest.fixture
def prices():
    return FixtureProvider().get_history("AAPL", period="5y")


@pytest.fixture
def close(prices):
    return prices.close


def _series(values: list[float]) -> pd.Series:
    """Baut eine Series mit UTC-Index aus einer Werteliste."""
    idx = pd.date_range("2024-01-01", periods=len(values), freq="B", tz="UTC")
    return pd.Series(values, index=idx, dtype="float64")


class TestSma:
    def test_matches_manual_mean(self, close):
        result = tech.sma(close, 20)
        assert result.iloc[-1] == pytest.approx(float(close.iloc[-20:].mean()))

    def test_warmup_stays_nan(self, close):
        assert tech.sma(close, 20).isna().sum() == 19

    def test_exact_on_known_input(self):
        series = _series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = tech.sma(series, 3)
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[-1] == pytest.approx(4.0)

    def test_accepts_price_series_directly(self, prices):
        assert not tech.sma(prices, 20).dropna().empty

    def test_rejects_non_positive_window(self, close):
        with pytest.raises(ValueError, match="positiv"):
            tech.sma(close, 0)

    def test_too_short_series_raises(self):
        with pytest.raises(InsufficientDataError) as exc:
            tech.sma(_series([1.0, 2.0]), 20)
        assert exc.value.required == 20
        assert exc.value.available == 2


class TestEma:
    def test_matches_recursive_definition(self, close):
        window = 12
        alpha = 2.0 / (window + 1)
        expected = float(close.iloc[:window].mean())
        for value in close.iloc[window:]:
            expected = alpha * float(value) + (1.0 - alpha) * expected
        assert tech.ema(close, window).iloc[-1] == pytest.approx(expected, rel=1e-9)

    def test_warmup_stays_nan(self, close):
        assert tech.ema(close, 12).isna().sum() == 11

    def test_reacts_faster_than_sma(self):
        # Sprung nach oben: der EMA muss naeher am neuen Niveau liegen.
        series = _series([100.0] * 20 + [200.0] * 5)
        assert tech.ema(series, 10).iloc[-1] > tech.sma(series, 10).iloc[-1]

    def test_rejects_non_positive_window(self, close):
        with pytest.raises(ValueError, match="positiv"):
            tech.ema(close, -1)


class TestRsi:
    def test_stays_within_scale(self, close):
        result = tech.rsi(close, 14).dropna()
        assert result.min() >= tech.RSI_MIN
        assert result.max() <= tech.RSI_MAX

    def test_only_gains_reaches_upper_bound(self):
        rising = _series([100.0 + i for i in range(40)])
        assert tech.rsi(rising, 14).iloc[-1] == pytest.approx(tech.RSI_MAX)

    def test_only_losses_reaches_lower_bound(self):
        falling = _series([140.0 - i for i in range(40)])
        assert tech.rsi(falling, 14).iloc[-1] == pytest.approx(tech.RSI_MIN)

    def test_no_movement_is_neutral_not_nan(self):
        """Bei Stillstand ist der Indikator definiert: 50, nicht NaN."""
        flat = _series([100.0] * 40)
        assert tech.rsi(flat, 14).iloc[-1] == pytest.approx(tech.RSI_NEUTRAL)

    def test_warmup_stays_nan(self, close):
        assert tech.rsi(close, 14).iloc[:14].isna().all()

    def test_needs_one_more_than_window(self):
        """Die erste Differenz kostet eine Beobachtung."""
        with pytest.raises(InsufficientDataError):
            tech.rsi(_series([100.0] * 14), 14)

    def test_rejects_non_positive_window(self, close):
        with pytest.raises(ValueError, match="positiv"):
            tech.rsi(close, 0)


class TestBollinger:
    def test_matches_manual_computation(self, close):
        bands = tech.bollinger_bands(close, 20, 2.0)
        window = close.iloc[-20:]
        mean = float(window.mean())
        std = float(window.std(ddof=0))
        assert bands["middle"].iloc[-1] == pytest.approx(mean)
        assert bands["upper"].iloc[-1] == pytest.approx(mean + 2.0 * std)
        assert bands["lower"].iloc[-1] == pytest.approx(mean - 2.0 * std)

    def test_column_names_follow_the_contract(self, close):
        bands = tech.bollinger_bands(close, 20)
        assert tuple(bands.columns) == BOLLINGER_COLUMNS

    def test_bands_enclose_the_middle(self, close):
        bands = tech.bollinger_bands(close, 20).dropna()
        assert (bands["upper"] >= bands["middle"]).all()
        assert (bands["lower"] <= bands["middle"]).all()

    def test_zero_std_collapses_the_bands(self, close):
        bands = tech.bollinger_bands(close, 20, 0.0).dropna()
        assert bands["upper"].equals(bands["middle"])

    def test_flat_series_has_zero_width(self):
        bands = tech.bollinger_bands(_series([50.0] * 30), 20).dropna()
        assert bands["upper"].iloc[-1] == pytest.approx(bands["lower"].iloc[-1])

    def test_rejects_negative_std(self, close):
        with pytest.raises(ValueError, match="num_std"):
            tech.bollinger_bands(close, 20, -1.0)


class TestMacd:
    def test_column_names_follow_the_contract(self, close):
        assert tuple(tech.macd(close).columns) == MACD_COLUMNS

    def test_histogram_is_line_minus_signal(self, close):
        frame = tech.macd(close).dropna()
        difference = frame["macd"] - frame["signal"] - frame["histogram"]
        assert difference.abs().max() == pytest.approx(0.0, abs=1e-12)

    def test_line_is_fast_ema_minus_slow_ema(self, close):
        frame = tech.macd(close, 12, 26, 9)
        fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
        slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
        expected = (fast - slow).dropna()
        assert frame["macd"].dropna().iloc[-1] == pytest.approx(float(expected.iloc[-1]))

    def test_fast_must_be_shorter_than_slow(self, close):
        with pytest.raises(ValueError, match="kleiner als"):
            tech.macd(close, 26, 12)

    def test_rejects_non_positive_periods(self, close):
        with pytest.raises(ValueError, match="positiv"):
            tech.macd(close, 0, 26)


class TestAtr:
    def test_is_positive_and_in_price_units(self, prices):
        result = tech.atr(prices, 14).dropna()
        assert (result > 0).all()
        assert result.iloc[-1] < float(prices.close.iloc[-1])

    def test_accounts_for_gaps(self, provenance):
        """Die True Range ist mehr als Hoch minus Tief: Luecken zaehlen."""
        from btlite.models.market import PriceSeries

        idx = pd.date_range("2024-01-01", periods=20, freq="B", tz="UTC", name="date")
        # Bar 10 eroeffnet mit einer Luecke nach oben.
        close = [100.0] * 10 + [150.0] * 10
        frame = pd.DataFrame(
            {
                "open": close,
                "high": [c + 1.0 for c in close],
                "low": [c - 1.0 for c in close],
                "close": close,
                "volume": [1_000_000.0] * 20,
            },
            index=idx,
        )
        series = PriceSeries(ticker="GAP", adjusted=True, data=frame, provenance=provenance)
        result = tech.atr(series, 5).dropna()
        # Ohne Luecken-Beruecksichtigung waere die Range konstant 2.0.
        assert result.max() > 2.0

    def test_rejects_non_positive_window(self, prices):
        with pytest.raises(ValueError, match="positiv"):
            tech.atr(prices, 0)


class TestCrossings:
    def test_detects_upward_cross(self):
        fast = _series([1.0, 2.0, 3.0, 4.0])
        slow = _series([2.5, 2.5, 2.5, 2.5])
        result = tech.crossover(fast, slow)
        assert not result.iloc[1]
        assert result.iloc[2], "Kreuzung genau bei Index 2"
        assert not result.iloc[3], "Danach kein neues Signal"

    def test_downward_cross_is_the_mirror_case(self):
        fast = _series([4.0, 3.0, 2.0, 1.0])
        slow = _series([2.5, 2.5, 2.5, 2.5])
        assert tech.crossunder(fast, slow).iloc[2]

    def test_nan_does_not_produce_a_signal(self):
        fast = _series([float("nan"), float("nan"), 3.0, 4.0])
        slow = _series([2.5, 2.5, 2.5, 2.5])
        assert not tech.crossover(fast, slow).iloc[:3].any()

    def test_no_signal_when_merely_touching(self):
        fast = _series([1.0, 2.5, 2.0])
        slow = _series([2.5, 2.5, 2.5])
        assert not tech.crossover(fast, slow).any()


class TestNoLookAhead:
    """Ein Indikatorwert bei t darf sich nicht aendern, wenn spaetere Bars fehlen."""

    @pytest.mark.parametrize(
        "compute",
        [
            lambda s: tech.sma(s, 20),
            lambda s: tech.ema(s, 12),
            lambda s: tech.rsi(s, 14),
            lambda s: tech.bollinger_bands(s, 20)["middle"],
            lambda s: tech.macd(s)["macd"],
        ],
    )
    def test_truncating_the_future_leaves_the_past_unchanged(self, close, compute):
        cut = 800
        full = compute(close)
        partial = compute(close.iloc[:cut])
        pd.testing.assert_series_equal(
            full.iloc[:cut].dropna(),
            partial.dropna(),
            check_freq=False,
            check_names=False,
        )

    def test_crossover_signals_are_stable_under_truncation(self, close):
        cut = 900
        fast_full, slow_full = tech.sma(close, 20), tech.sma(close, 50)
        signals_full = tech.crossover(fast_full, slow_full).iloc[:cut]

        short = close.iloc[:cut]
        signals_partial = tech.crossover(tech.sma(short, 20), tech.sma(short, 50))

        assert list(signals_full) == list(signals_partial)


class TestBuildOverlays:
    def test_default_set(self, prices):
        overlays = tech.build_overlays(prices)
        assert set(overlays.sma) == {"SMA 20", "SMA 50", "SMA 200"}
        assert overlays.bollinger is None
        assert overlays.has_rsi is False
        assert overlays.has_macd is False

    def test_full_set(self, prices):
        overlays = tech.build_overlays(
            prices, with_bollinger=True, with_rsi=True, with_macd=True, ema_windows=(12, 26)
        )
        assert overlays.bollinger is not None
        assert overlays.has_rsi
        assert overlays.has_macd
        assert set(overlays.ema) == {"EMA 12", "EMA 26"}

    def test_indicators_longer_than_history_are_skipped_not_fatal(self, provenance):
        """Bei kurzer Historie zeigt das UI, was berechenbar ist."""
        from btlite.models.market import PriceSeries

        idx = pd.date_range("2024-01-01", periods=30, freq="B", tz="UTC", name="date")
        rng = np.random.default_rng(1)
        close = 100.0 + np.cumsum(rng.normal(0, 1, 30))
        frame = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": [1e6] * 30,
            },
            index=idx,
        )
        series = PriceSeries(ticker="SHORT", adjusted=True, data=frame, provenance=provenance)
        overlays = tech.build_overlays(series, with_rsi=True, with_macd=True)
        assert "SMA 20" in overlays.sma
        assert "SMA 200" not in overlays.sma, "Zu lang fuer 30 Bars"
        assert overlays.has_rsi, "14er RSI passt in 30 Bars"
        assert overlays.has_macd, "26er MACD passt gerade"

    def test_series_share_the_price_index(self, prices):
        overlays = tech.build_overlays(prices, with_rsi=True)
        assert overlays.sma["SMA 20"].index.equals(prices.data.index)
        assert overlays.rsi is not None
        assert overlays.rsi.index.equals(prices.data.index)


class TestAcrossTickers:
    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "SPY", "KO"])
    def test_indicators_are_computable_for_every_fixture(self, ticker):
        series = FixtureProvider().get_history(ticker, period="5y")
        overlays = tech.build_overlays(series, with_bollinger=True, with_rsi=True, with_macd=True)
        assert overlays.sma
        assert overlays.has_rsi
        assert overlays.has_macd
        assert overlays.rsi is not None
        assert overlays.rsi.dropna().between(tech.RSI_MIN, tech.RSI_MAX).all()
