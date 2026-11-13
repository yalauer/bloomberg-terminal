"""Tests der Risikometriken (Block 5).

Die Kernwerte sind gegen **handgerechnete** Referenzwerte geprueft, nicht
gegen die Ausgabe der eigenen Implementierung. Die Herleitung steht jeweils
im Docstring des Tests, damit sie ohne Nachrechnen nachvollziehbar ist.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from btlite.analytics import risk
from btlite.config import AnalyticsSettings
from btlite.core.errors import InsufficientDataError
from btlite.models.common import Interval

#: Handgerechnete Referenzreihe. Zehn Renditen mit Mittelwert exakt 0.
#:
#: Summe = 0.01-0.02+0.03+0.00-0.01+0.02-0.03+0.04+0.01-0.05 = 0.00
_REFERENCE_RETURNS: tuple[float, ...] = (
    0.01,
    -0.02,
    0.03,
    0.00,
    -0.01,
    0.02,
    -0.03,
    0.04,
    0.01,
    -0.05,
)

#: Stichprobenvarianz der Referenzreihe, ddof=1.
#: Da der Mittelwert 0 ist, ist sie sum(r^2)/(n-1) = 0.0070/9.
_REFERENCE_VARIANCE = 0.0070 / 9.0
_REFERENCE_SD = math.sqrt(_REFERENCE_VARIANCE)

#: Verkettetes Wachstum der Referenzreihe, Produkt aller (1+r).
_REFERENCE_GROWTH = 0.9964834569252763


@pytest.fixture
def reference() -> pd.Series:
    """Die handgerechnete Referenzreihe mit UTC-Index."""
    idx = pd.date_range("2024-01-02", periods=10, freq="B", tz="UTC", name="date")
    return pd.Series(_REFERENCE_RETURNS, index=idx, name="return")


@pytest.fixture
def benchmark() -> pd.Series:
    """Benchmark-Reihe. Das Portfolio ist exakt ihr Doppeltes."""
    idx = pd.date_range("2024-01-02", periods=8, freq="B", tz="UTC", name="date")
    return pd.Series([0.01, -0.02, 0.03, 0.00, -0.01, 0.02, -0.03, 0.04], index=idx)


@pytest.fixture
def doubled(benchmark: pd.Series) -> pd.Series:
    """Portfolio-Reihe, exakt das Doppelte der Benchmark."""
    return benchmark * 2.0


@pytest.fixture
def long_returns() -> pd.Series:
    """Laengere, deterministische Reihe fuer Tests mit min_observations."""
    idx = pd.date_range("2023-01-02", periods=300, freq="B", tz="UTC", name="date")
    rng = np.random.default_rng(11)
    return pd.Series(rng.normal(0.0004, 0.012, len(idx)), index=idx)


class TestVolatility:
    def test_matches_hand_computed(self, reference):
        """Standardabweichung sqrt(0.0070/9), annualisiert mal sqrt(252)."""
        expected = _REFERENCE_SD * math.sqrt(252)
        assert risk.volatility(reference) == pytest.approx(expected, abs=1e-12)

    def test_is_annualized_by_interval(self, reference):
        """Wochendaten annualisieren mit sqrt(52), nicht sqrt(252)."""
        weekly = risk.volatility(reference, interval=Interval.W1)
        assert weekly == pytest.approx(_REFERENCE_SD * math.sqrt(52), abs=1e-12)

    def test_not_hardcoded_252(self, reference):
        """Der Faktor kommt aus Interval, nicht aus einer Konstante."""
        daily = risk.volatility(reference, interval=Interval.D1)
        monthly = risk.volatility(reference, interval=Interval.MO1)
        assert daily / monthly == pytest.approx(math.sqrt(252 / 12), abs=1e-12)

    def test_single_observation_is_none(self):
        series = pd.Series([0.01], index=pd.date_range("2024-01-02", periods=1, tz="UTC"))
        assert risk.volatility(series) is None

    def test_never_negative(self, long_returns):
        assert risk.volatility(long_returns) >= 0.0


class TestDownsideDeviation:
    def test_matches_hand_computed(self, reference):
        """Nur Unterschreitungen: 0.02, 0.01, 0.03, 0.05.

        Summe der Quadrate = 0.0004+0.0001+0.0009+0.0025 = 0.0039.
        Nenner ist n = 10, nicht die Zahl der Unterschreitungen.
        """
        periodic = math.sqrt(0.0039 / 10.0)
        expected = periodic * math.sqrt(252)
        assert risk.downside_deviation(reference) == pytest.approx(expected, abs=1e-12)

    def test_all_positive_returns_gives_zero(self):
        idx = pd.date_range("2024-01-02", periods=5, freq="B", tz="UTC")
        series = pd.Series([0.01, 0.02, 0.03, 0.01, 0.02], index=idx)
        assert risk.downside_deviation(series) == pytest.approx(0.0)

    def test_target_shifts_the_measure(self, reference):
        """Eine hoehere Zielrendite macht mehr Beobachtungen zu Unterschreitungen."""
        base = risk.downside_deviation(reference, target=0.0)
        higher = risk.downside_deviation(reference, target=0.02)
        assert higher > base


class TestAnnualizedReturn:
    def test_matches_hand_computed(self, reference):
        """Geometrisch: growth^(252/10) - 1 mit growth = prod(1+r)."""
        expected = _REFERENCE_GROWTH ** (252 / 10) - 1.0
        assert risk.annualized_return(reference) == pytest.approx(expected, rel=1e-12)

    def test_total_loss_is_none_not_minus_one(self):
        """Ein -100-Prozent-Tag macht die Verkettung 0 und damit unbestimmt."""
        idx = pd.date_range("2024-01-02", periods=3, freq="B", tz="UTC")
        series = pd.Series([0.01, -1.0, 0.02], index=idx)
        assert risk.annualized_return(series) is None


class TestSharpeAndSortino:
    def test_sharpe_is_zero_when_mean_equals_rf(self, reference):
        """Der Mittelwert der Referenzreihe ist exakt 0, also auch die Ueberrendite."""
        assert risk.sharpe_ratio(reference, risk_free_rate=0.0) == pytest.approx(0.0, abs=1e-12)

    def test_risk_free_rate_changes_sharpe_measurably(self, long_returns):
        """Der Fallstrick des Pakets: rf muss wirklich durchgereicht werden."""
        zero = risk.sharpe_ratio(long_returns, risk_free_rate=0.0)
        two = risk.sharpe_ratio(long_returns, risk_free_rate=0.02)
        assert zero != pytest.approx(two, abs=1e-6)
        assert zero > two

    def test_sharpe_hand_computed(self, long_returns):
        """(mean - rf/252) / sd * sqrt(252), direkt nachgerechnet."""
        rate = 0.03
        excess = long_returns - rate / 252
        expected = float(excess.mean()) / float(excess.std(ddof=1)) * math.sqrt(252)
        got = risk.sharpe_ratio(long_returns, risk_free_rate=rate)
        assert got == pytest.approx(expected, rel=1e-12)

    def test_sortino_uses_only_downside(self, long_returns):
        """Sortino bestraft nur Verluste, ist hier also groesser als Sharpe."""
        sharpe = risk.sharpe_ratio(long_returns, risk_free_rate=0.02)
        sortino = risk.sortino_ratio(long_returns, risk_free_rate=0.02)
        assert sortino > sharpe

    def test_sortino_none_without_downside(self):
        """Ohne jede Unterschreitung ist der Nenner 0 und das Mass unbestimmt."""
        idx = pd.date_range("2024-01-02", periods=5, freq="B", tz="UTC")
        series = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=idx)
        assert risk.sortino_ratio(series, risk_free_rate=0.0) is None

    def test_zero_variance_is_none(self):
        idx = pd.date_range("2024-01-02", periods=10, freq="B", tz="UTC")
        series = pd.Series([0.01] * 10, index=idx)
        assert risk.sharpe_ratio(series, risk_free_rate=0.0) is None


class TestValueAtRisk:
    def test_historical_matches_hand_computed(self, reference):
        """np.quantile interpoliert linear: das 5-Prozent-Quantil ist -0.041.

        Sortiert: -0.05, -0.03, -0.02, -0.01, 0.00, 0.01, 0.01, 0.02, 0.03, 0.04.
        Position 0.05*(10-1) = 0.45 zwischen -0.05 und -0.03:
        -0.05 + 0.45*0.02 = -0.041.
        """
        assert risk.value_at_risk_historical(reference) == pytest.approx(-0.041, abs=1e-12)

    def test_cvar_matches_hand_computed(self, reference):
        """Nur -0.05 liegt bei oder unter dem VaR von -0.041."""
        assert risk.conditional_value_at_risk(reference) == pytest.approx(-0.05, abs=1e-12)

    def test_parametric_matches_hand_computed(self, reference):
        """Formel mean + sd * z(0.05); Mittelwert 0, z(0.05) = -1.6448536..."""
        expected = 0.0 + _REFERENCE_SD * -1.6448536269514722
        assert risk.value_at_risk_parametric(reference) == pytest.approx(expected, abs=1e-8)

    @pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
    def test_all_var_measures_are_negative(self, long_returns, confidence):
        assert risk.value_at_risk_historical(long_returns, confidence=confidence) <= 0.0
        assert risk.value_at_risk_parametric(long_returns, confidence=confidence) <= 0.0
        assert risk.conditional_value_at_risk(long_returns, confidence=confidence) <= 0.0

    @pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
    def test_cvar_never_above_var(self, long_returns, confidence):
        """Der Erwartungswert jenseits der Schwelle kann sie nicht uebertreffen."""
        var = risk.value_at_risk_historical(long_returns, confidence=confidence)
        cvar = risk.conditional_value_at_risk(long_returns, confidence=confidence)
        assert cvar <= var

    def test_higher_confidence_is_more_severe(self, long_returns):
        assert risk.value_at_risk_historical(
            long_returns, confidence=0.99
        ) <= risk.value_at_risk_historical(long_returns, confidence=0.90)

    def test_only_gains_is_clipped_to_zero(self):
        """Eine Reihe ohne Verluste ergibt kein positives Risikomass."""
        idx = pd.date_range("2024-01-02", periods=10, freq="B", tz="UTC")
        series = pd.Series([0.01] * 10, index=idx)
        assert risk.value_at_risk_historical(series) == 0.0

    def test_empty_is_none(self):
        empty = pd.Series(dtype="float64")
        assert risk.value_at_risk_historical(empty) is None
        assert risk.conditional_value_at_risk(empty) is None
        assert risk.value_at_risk_parametric(empty) is None


class TestNormalQuantile:
    @pytest.mark.parametrize(
        ("probability", "expected"),
        [
            (0.01, -2.3263478740408408),
            (0.025, -1.9599639845400545),
            (0.05, -1.6448536269514722),
            (0.5, 0.0),
            (0.95, 1.6448536269514722),
            (0.99, 2.3263478740408408),
        ],
    )
    def test_matches_known_quantiles(self, probability, expected):
        """Bekannte Tabellenwerte der Standardnormalverteilung."""
        assert risk._normal_ppf(probability) == pytest.approx(expected, abs=1e-8)

    @pytest.mark.parametrize("probability", [0.0, 1.0, -0.1, 1.5])
    def test_outside_open_interval_is_none(self, probability):
        assert risk._normal_ppf(probability) is None

    def test_is_symmetric(self):
        assert risk._normal_ppf(0.3) == pytest.approx(-risk._normal_ppf(0.7), abs=1e-9)


class TestMaxDrawdown:
    def test_matches_hand_computed(self, reference):
        """Der Hoechststand liegt bei Index 8, der Tiefpunkt am Schluss.

        Equity: ... 1.04892995 (Index 8), dann -5 Prozent auf 0.99648346.
        Der Drawdown ist damit exakt -0.05.
        """
        value, start, end = risk.max_drawdown(reference)
        assert value == pytest.approx(-0.05, abs=1e-12)
        assert start == reference.index[8]
        assert end == reference.index[9]

    def test_is_negative_and_ordered(self, long_returns):
        value, start, end = risk.max_drawdown(long_returns)
        assert value < 0.0
        assert start < end

    def test_monotonic_series_has_no_drawdown(self):
        """Eine nur steigende Reihe liefert None, nicht 0.0."""
        idx = pd.date_range("2024-01-02", periods=10, freq="B", tz="UTC")
        series = pd.Series([0.01] * 10, index=idx)
        value, start, end = risk.max_drawdown(series)
        assert value is None
        assert start is None
        assert end is None

    def test_drawdown_series_is_never_positive(self, long_returns):
        assert (risk.drawdown_series(long_returns) <= 1e-12).all()


class TestRelativeMetrics:
    def test_beta_of_doubled_series_is_two(self, doubled, benchmark):
        """Portfolio = 2*Benchmark, also cov/var = 2*var(b)/var(b) = 2."""
        assert risk.beta(doubled, benchmark) == pytest.approx(2.0, abs=1e-12)

    def test_r_squared_of_linear_relation_is_one(self, doubled, benchmark):
        assert risk.r_squared(doubled, benchmark) == pytest.approx(1.0, abs=1e-12)

    def test_alpha_of_pure_beta_exposure_is_zero(self, doubled, benchmark):
        """Bei rf=0 und p=2b ist mean_p - 2*mean_b exakt 0."""
        assert risk.alpha(doubled, benchmark, risk_free_rate=0.0) == pytest.approx(0.0, abs=1e-12)

    def test_beta_of_identical_series_is_one(self, benchmark):
        assert risk.beta(benchmark, benchmark) == pytest.approx(1.0, abs=1e-12)

    def test_tracking_error_hand_computed(self, doubled, benchmark):
        """Die Aktivrendite p-b ist hier genau b, also TE = sd(b)*sqrt(252)."""
        expected = float(benchmark.std(ddof=1)) * math.sqrt(252)
        assert risk.tracking_error(doubled, benchmark) == pytest.approx(expected, rel=1e-12)

    def test_tracking_error_against_self_is_zero(self, benchmark):
        assert risk.tracking_error(benchmark, benchmark) == pytest.approx(0.0, abs=1e-12)

    def test_information_ratio_is_none_without_tracking_error(self, benchmark):
        assert risk.information_ratio(benchmark, benchmark) is None

    def test_constant_benchmark_gives_no_beta(self):
        """Ohne Varianz in der Benchmark ist Beta unbestimmt, nicht 0."""
        idx = pd.date_range("2024-01-02", periods=10, freq="B", tz="UTC")
        flat = pd.Series([0.01] * 10, index=idx)
        moving = pd.Series(np.linspace(-0.02, 0.02, 10), index=idx)
        assert risk.beta(moving, flat) is None
        assert risk.r_squared(moving, flat) is None

    def test_disjoint_indices_give_none(self):
        a = pd.Series([0.01] * 5, index=pd.date_range("2024-01-02", periods=5, tz="UTC"))
        b = pd.Series([0.01] * 5, index=pd.date_range("2025-01-02", periods=5, tz="UTC"))
        assert risk.beta(a, b) is None

    def test_r_squared_stays_within_bounds(self, long_returns):
        rng = np.random.default_rng(3)
        other = pd.Series(rng.normal(0, 0.01, len(long_returns)), index=long_returns.index)
        value = risk.r_squared(long_returns, other)
        assert 0.0 <= value <= 1.0


class TestShape:
    def test_symmetric_series_has_zero_skew(self):
        idx = pd.date_range("2024-01-02", periods=8, freq="B", tz="UTC")
        series = pd.Series([-0.03, -0.02, -0.01, 0.0, 0.0, 0.01, 0.02, 0.03], index=idx)
        assert risk.skewness(series) == pytest.approx(0.0, abs=1e-12)

    def test_kurtosis_is_excess(self):
        """pandas.kurt liefert den Exzess: eine Normalverteilung liegt nahe 0."""
        rng = np.random.default_rng(5)
        idx = pd.date_range("2020-01-02", periods=5000, freq="B", tz="UTC")
        series = pd.Series(rng.normal(0, 0.01, 5000), index=idx)
        assert risk.kurtosis(series) == pytest.approx(0.0, abs=0.2)

    def test_fat_tails_raise_kurtosis(self, long_returns):
        spiked = long_returns.copy()
        spiked.iloc[0] = 0.5
        assert risk.kurtosis(spiked) > risk.kurtosis(long_returns)

    def test_too_short_is_none(self):
        idx = pd.date_range("2024-01-02", periods=3, freq="B", tz="UTC")
        series = pd.Series([0.01, 0.02, 0.03], index=idx)
        assert risk.kurtosis(series) is None


class TestRolling:
    def test_rolling_volatility_is_annualized(self, long_returns):
        rolled = risk.rolling_volatility(long_returns, window=63)
        window = long_returns.iloc[:63]
        expected = float(window.std(ddof=1)) * math.sqrt(252)
        assert rolled.iloc[62] == pytest.approx(expected, rel=1e-12)

    def test_rolling_volatility_warmup_is_nan(self, long_returns):
        rolled = risk.rolling_volatility(long_returns, window=63)
        assert rolled.iloc[:62].isna().all()
        assert not math.isnan(rolled.iloc[62])

    def test_rolling_volatility_keeps_index(self, long_returns):
        rolled = risk.rolling_volatility(long_returns, window=20)
        assert rolled.index.equals(long_returns.index)

    def test_rolling_beta_of_doubled_series_is_two(self, long_returns):
        doubled = long_returns * 2.0
        rolled = risk.rolling_beta(doubled, long_returns, window=30)
        assert rolled.dropna().to_numpy() == pytest.approx(2.0, abs=1e-9)

    def test_rolling_beta_has_no_infinities(self, long_returns):
        """Ein Fenster ohne Benchmark-Varianz wird NaN, nie inf."""
        flat = pd.Series(0.01, index=long_returns.index)
        rolled = risk.rolling_beta(long_returns, flat, window=30)
        assert not np.isinf(rolled.to_numpy()).any()

    @pytest.mark.parametrize("window", [0, 1, -5])
    def test_invalid_window_raises(self, long_returns, window):
        with pytest.raises(ValueError, match="window"):
            risk.rolling_volatility(long_returns, window=window)
        with pytest.raises(ValueError, match="window"):
            risk.rolling_beta(long_returns, long_returns, window=window)


class TestComputeRiskMetrics:
    def test_fills_all_absolute_metrics(self, long_returns):
        metrics = risk.compute_risk_metrics(long_returns, risk_free_rate=0.02)
        for field in (
            "volatility",
            "downside_deviation",
            "var_historical",
            "var_parametric",
            "cvar_historical",
            "max_drawdown",
            "sharpe",
            "sortino",
            "skewness",
            "kurtosis",
        ):
            assert getattr(metrics, field) is not None, field

    def test_relative_metrics_are_none_without_benchmark(self, long_returns):
        """Ohne Referenz sind sie nicht definiert - und dann None, nicht 0.0."""
        metrics = risk.compute_risk_metrics(long_returns, risk_free_rate=0.02)
        assert metrics.beta is None
        assert metrics.alpha is None
        assert metrics.r_squared is None
        assert metrics.tracking_error is None
        assert metrics.information_ratio is None

    def test_relative_metrics_present_with_benchmark(self, long_returns):
        rng = np.random.default_rng(21)
        bench = pd.Series(rng.normal(0.0003, 0.01, len(long_returns)), index=long_returns.index)
        metrics = risk.compute_risk_metrics(long_returns, bench, risk_free_rate=0.02)
        assert metrics.beta is not None
        assert metrics.r_squared is not None
        assert metrics.tracking_error is not None

    def test_signs_hold(self, long_returns):
        metrics = risk.compute_risk_metrics(long_returns, risk_free_rate=0.02)
        assert metrics.var_historical <= 0.0
        assert metrics.var_parametric <= 0.0
        assert metrics.cvar_historical <= 0.0
        assert metrics.cvar_historical <= metrics.var_historical
        assert metrics.max_drawdown <= 0.0
        assert metrics.volatility >= 0.0

    def test_drawdown_dates_are_ordered(self, long_returns):
        metrics = risk.compute_risk_metrics(long_returns, risk_free_rate=0.02)
        assert metrics.max_drawdown_start <= metrics.max_drawdown_end

    def test_risk_free_rate_is_passed_through(self, long_returns):
        low = risk.compute_risk_metrics(long_returns, risk_free_rate=0.0)
        high = risk.compute_risk_metrics(long_returns, risk_free_rate=0.05)
        assert low.sharpe > high.sharpe
        assert low.sortino > high.sortino

    def test_too_few_observations_raise(self):
        idx = pd.date_range("2024-01-02", periods=5, freq="B", tz="UTC")
        series = pd.Series([0.01, -0.01, 0.02, 0.0, -0.02], index=idx)
        with pytest.raises(InsufficientDataError) as info:
            risk.compute_risk_metrics(series, risk_free_rate=0.02)
        assert info.value.required == 30
        assert info.value.available == 5

    def test_threshold_comes_from_settings(self):
        """min_observations ist konfigurierbar, nicht hartcodiert."""
        idx = pd.date_range("2024-01-02", periods=5, freq="B", tz="UTC")
        series = pd.Series([0.01, -0.01, 0.02, 0.0, -0.02], index=idx)
        relaxed = AnalyticsSettings(min_observations=4)
        metrics = risk.compute_risk_metrics(series, settings=relaxed, risk_free_rate=0.02)
        assert metrics.volatility is not None

    def test_confidence_is_recorded(self, long_returns):
        metrics = risk.compute_risk_metrics(long_returns, confidence=0.99, risk_free_rate=0.02)
        assert metrics.confidence == 0.99

    def test_nan_inputs_are_dropped(self, long_returns):
        with_gaps = long_returns.copy()
        with_gaps.iloc[5] = np.nan
        with_gaps.iloc[50] = np.nan
        metrics = risk.compute_risk_metrics(with_gaps, risk_free_rate=0.02)
        assert metrics.volatility is not None

    def test_no_metric_is_nan_or_inf(self, long_returns):
        """Projektregel: fehlende Werte sind None, nie NaN und nie inf."""
        rng = np.random.default_rng(31)
        bench = pd.Series(rng.normal(0.0003, 0.01, len(long_returns)), index=long_returns.index)
        metrics = risk.compute_risk_metrics(long_returns, bench, risk_free_rate=0.02)
        for name, value in metrics.model_dump().items():
            if isinstance(value, float):
                assert math.isfinite(value), name

    def test_benchmark_without_overlap_leaves_relatives_none(self, long_returns):
        other = pd.Series(
            [0.01] * 10, index=pd.date_range("2030-01-02", periods=10, freq="B", tz="UTC")
        )
        metrics = risk.compute_risk_metrics(long_returns, other, risk_free_rate=0.02)
        assert metrics.beta is None


class TestProperties:
    @given(
        values=st.lists(
            st.floats(min_value=-0.25, max_value=0.25, allow_nan=False, allow_infinity=False),
            min_size=40,
            max_size=200,
        )
    )
    @settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_cvar_never_exceeds_var(self, values):
        """Invariante ueber beliebigen Renditereihen."""
        idx = pd.date_range("2024-01-02", periods=len(values), freq="B", tz="UTC")
        series = pd.Series(values, index=idx)
        var = risk.value_at_risk_historical(series)
        cvar = risk.conditional_value_at_risk(series)
        if var is not None and cvar is not None:
            assert cvar <= var + 1e-12

    @given(
        values=st.lists(
            st.floats(min_value=-0.25, max_value=0.25, allow_nan=False, allow_infinity=False),
            min_size=40,
            max_size=200,
        )
    )
    @settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_var_measures_are_never_positive(self, values):
        idx = pd.date_range("2024-01-02", periods=len(values), freq="B", tz="UTC")
        series = pd.Series(values, index=idx)
        for value in (
            risk.value_at_risk_historical(series),
            risk.value_at_risk_parametric(series),
            risk.conditional_value_at_risk(series),
        ):
            if value is not None:
                assert value <= 0.0

    @given(
        values=st.lists(
            st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
            min_size=5,
            max_size=120,
        )
    )
    @settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_drawdown_is_never_positive(self, values):
        idx = pd.date_range("2024-01-02", periods=len(values), freq="B", tz="UTC")
        series = pd.Series(values, index=idx)
        value, start, end = risk.max_drawdown(series)
        if value is not None:
            assert value < 0.0
            assert start <= end

    @given(
        values=st.lists(
            st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False),
            min_size=5,
            max_size=120,
        ),
        factor=st.floats(min_value=0.1, max_value=10.0),
    )
    @settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_volatility_scales_linearly(self, values, factor):
        """Volatilitaet ist homogen vom Grad 1: vol(c*r) = c*vol(r)."""
        idx = pd.date_range("2024-01-02", periods=len(values), freq="B", tz="UTC")
        series = pd.Series(values, index=idx)
        base = risk.volatility(series)
        scaled = risk.volatility(series * factor)
        if base is not None and scaled is not None:
            assert scaled == pytest.approx(base * factor, rel=1e-9, abs=1e-12)

    @given(
        values=st.lists(
            st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=100,
        )
    )
    @settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_beta_against_self_is_one(self, values):
        idx = pd.date_range("2024-01-02", periods=len(values), freq="B", tz="UTC")
        series = pd.Series(values, index=idx)
        value = risk.beta(series, series)
        if value is not None:
            assert value == pytest.approx(1.0, rel=1e-9)
