"""Tests der Portfolio-Optimierung (Block 5).

Akzeptanzkriterium des Projekts: die Optimierung laeuft ueber alle zehn
Ticker des Fixture, die Gewichte summieren sich auf 1.0, ohne Flag gibt es
keine negativen Gewichte, und die Effizienzgrenze ist plausibel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from btlite.core.errors import InsufficientDataError, OptimizationFailedError
from btlite.data.fixtures import FixtureProvider
from btlite.models.portfolio import (
    WEIGHT_SUM_TOLERANCE,
    CovarianceMethod,
    EfficientFrontier,
    OptimizationObjective,
    PortfolioRequest,
    PortfolioResult,
    ReturnsMethod,
)
from btlite.portfolio import optimizer

#: Zielvorgaben, die einzelne Zielfunktionen zwingend brauchen.
_TARGETS: dict[OptimizationObjective, dict[str, float]] = {
    OptimizationObjective.EFFICIENT_RETURN: {"target_return": 0.20},
    OptimizationObjective.EFFICIENT_RISK: {"target_volatility": 0.18},
}


def _request(tickers: list[str], objective: OptimizationObjective, **kwargs) -> PortfolioRequest:
    """Baut eine Anfrage mit den noetigen Zielvorgaben."""
    return PortfolioRequest(
        tickers=tickers, objective=objective, **_TARGETS.get(objective, {}), **kwargs
    )


@pytest.fixture(scope="module")
def provider() -> FixtureProvider:
    return FixtureProvider()


@pytest.fixture(scope="module")
def universe(provider: FixtureProvider) -> list[str]:
    """Alle zehn Ticker des Fixture."""
    return provider.available_tickers()


@pytest.fixture(scope="module")
def prices(provider: FixtureProvider, universe: list[str]):
    return provider.get_histories(universe)


@pytest.fixture(scope="module")
def matrix(prices) -> pd.DataFrame:
    return optimizer.build_price_matrix(prices)


@pytest.fixture(scope="module")
def estimates(matrix: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    mu = optimizer.estimate_returns(
        matrix, ReturnsMethod.MEAN_HISTORICAL, frequency=252, risk_free_rate=0.02
    )
    cov = optimizer.estimate_covariance(matrix, CovarianceMethod.LEDOIT_WOLF, frequency=252)
    return mu, cov


class TestPriceMatrix:
    def test_has_all_ten_tickers(self, matrix, universe):
        assert len(universe) == 10
        assert sorted(matrix.columns) == sorted(universe)

    def test_has_no_missing_values(self, matrix):
        assert not matrix.isna().to_numpy().any()

    def test_index_is_sorted_utc(self, matrix):
        assert matrix.index.is_monotonic_increasing
        assert str(matrix.index.tz) == "UTC"

    def test_empty_input_raises(self):
        with pytest.raises(InsufficientDataError):
            optimizer.build_price_matrix({})

    def test_too_short_history_raises(self, provider, universe):
        short = provider.get_histories(universe[:3])
        trimmed = {}
        for ticker, series in short.items():
            clipped = series.model_copy()
            clipped.data = series.data.head(5)
            trimmed[ticker] = clipped
        with pytest.raises(InsufficientDataError) as info:
            optimizer.build_price_matrix(trimmed)
        assert info.value.required == optimizer.MIN_PRICE_OBSERVATIONS


class TestEstimators:
    @pytest.mark.parametrize("method", list(ReturnsMethod))
    def test_every_returns_method_runs(self, matrix, method):
        mu = optimizer.estimate_returns(matrix, method, frequency=252, risk_free_rate=0.02)
        assert len(mu) == len(matrix.columns)
        assert np.isfinite(mu.to_numpy()).all()

    @pytest.mark.parametrize("method", list(CovarianceMethod))
    def test_every_covariance_method_is_psd(self, matrix, method):
        """Eine nicht positiv semidefinite Matrix laesst den Solver scheitern."""
        cov = optimizer.estimate_covariance(matrix, method, frequency=252)
        eigenvalues = np.linalg.eigvalsh(cov.to_numpy())
        assert eigenvalues.min() >= -1e-10

    def test_covariance_is_square_and_symmetric(self, estimates):
        _, cov = estimates
        assert cov.shape[0] == cov.shape[1]
        assert np.allclose(cov.to_numpy(), cov.to_numpy().T)

    def test_capm_uses_risk_free_rate(self, matrix):
        low = optimizer.estimate_returns(
            matrix, ReturnsMethod.CAPM, frequency=252, risk_free_rate=0.0
        )
        high = optimizer.estimate_returns(
            matrix, ReturnsMethod.CAPM, frequency=252, risk_free_rate=0.05
        )
        assert not np.allclose(low.to_numpy(), high.to_numpy())


class TestCorrelation:
    def test_is_square_with_unit_diagonal(self, matrix):
        corr = optimizer.correlation_matrix(matrix)
        assert corr.shape == (10, 10)
        assert np.allclose(np.diag(corr.to_numpy()), 1.0)

    def test_values_stay_in_range(self, matrix):
        corr = optimizer.correlation_matrix(matrix).to_numpy()
        assert corr.min() >= -1.0 - 1e-12
        assert corr.max() <= 1.0 + 1e-12

    def test_is_symmetric(self, matrix):
        corr = optimizer.correlation_matrix(matrix).to_numpy()
        assert np.allclose(corr, corr.T)


class TestAllObjectives:
    """Akzeptanzkriterium: jedes OptimizationObjective laeuft durch."""

    @pytest.mark.parametrize("objective", list(OptimizationObjective))
    def test_objective_runs_over_ten_tickers(self, universe, prices, objective):
        result = optimizer.optimize(_request(universe, objective, risk_free_rate=0.02), prices)
        assert len(result.positions) == 10

    @pytest.mark.parametrize("objective", list(OptimizationObjective))
    def test_weights_sum_to_one(self, universe, prices, objective):
        result = optimizer.optimize(
            _request(universe, objective, risk_free_rate=0.02), prices, with_frontier=False
        )
        total = sum(p.weight for p in result.positions)
        assert abs(total - 1.0) <= WEIGHT_SUM_TOLERANCE

    @pytest.mark.parametrize("objective", list(OptimizationObjective))
    def test_no_negative_weights_without_flag(self, universe, prices, objective):
        result = optimizer.optimize(
            _request(universe, objective, risk_free_rate=0.02), prices, with_frontier=False
        )
        assert all(p.weight >= -WEIGHT_SUM_TOLERANCE for p in result.positions)

    @pytest.mark.parametrize("objective", list(OptimizationObjective))
    def test_metrics_are_plausible(self, universe, prices, objective):
        result = optimizer.optimize(
            _request(universe, objective, risk_free_rate=0.02), prices, with_frontier=False
        )
        assert result.expected_volatility > 0.0
        assert -1.0 < result.expected_return < 3.0
        assert -5.0 < result.sharpe_ratio < 10.0


class TestSpecificObjectives:
    def test_min_volatility_beats_equal_weight(self, universe, prices):
        """Die Minimum-Varianz-Loesung muss risikoaermer sein als 1/N."""
        minimal = optimizer.optimize(
            _request(universe, OptimizationObjective.MIN_VOLATILITY, risk_free_rate=0.02),
            prices,
            with_frontier=False,
        )
        equal = optimizer.optimize(
            _request(universe, OptimizationObjective.EQUAL_WEIGHT, risk_free_rate=0.02),
            prices,
            with_frontier=False,
        )
        assert minimal.expected_volatility < equal.expected_volatility

    def test_max_sharpe_beats_equal_weight(self, universe, prices):
        best = optimizer.optimize(
            _request(universe, OptimizationObjective.MAX_SHARPE, risk_free_rate=0.02),
            prices,
            with_frontier=False,
        )
        equal = optimizer.optimize(
            _request(universe, OptimizationObjective.EQUAL_WEIGHT, risk_free_rate=0.02),
            prices,
            with_frontier=False,
        )
        assert best.sharpe_ratio > equal.sharpe_ratio

    def test_equal_weight_is_exactly_one_over_n(self, universe, prices):
        result = optimizer.optimize(
            _request(universe, OptimizationObjective.EQUAL_WEIGHT, risk_free_rate=0.02),
            prices,
            with_frontier=False,
        )
        for position in result.positions:
            assert position.weight == pytest.approx(0.1, abs=1e-9)

    def test_efficient_return_hits_its_target(self, universe, prices):
        result = optimizer.optimize(
            _request(universe, OptimizationObjective.EFFICIENT_RETURN, risk_free_rate=0.02),
            prices,
            with_frontier=False,
        )
        assert result.expected_return == pytest.approx(0.20, abs=1e-3)

    def test_efficient_risk_respects_its_budget(self, universe, prices):
        result = optimizer.optimize(
            _request(universe, OptimizationObjective.EFFICIENT_RISK, risk_free_rate=0.02),
            prices,
            with_frontier=False,
        )
        assert result.expected_volatility <= 0.18 + 1e-3

    def test_hrp_is_diversified(self, universe, prices):
        """HRP verteilt breit: keine Position dominiert das Portfolio."""
        result = optimizer.optimize(
            _request(universe, OptimizationObjective.HRP, risk_free_rate=0.02),
            prices,
            with_frontier=False,
        )
        assert all(p.weight > 0.0 for p in result.positions)
        assert max(p.weight for p in result.positions) < 0.5

    def test_hrp_needs_two_assets(self, matrix):
        single = matrix.iloc[:, :1].pct_change().dropna()
        with pytest.raises(OptimizationFailedError):
            optimizer._hrp_weights(single)


class TestRiskFreeRate:
    """Der dokumentierte Fallstrick von PyPortfolioOpt 1.6."""

    def test_sharpe_changes_measurably_with_rate(self, universe, prices):
        results = {
            rate: optimizer.optimize(
                _request(universe, OptimizationObjective.MAX_SHARPE, risk_free_rate=rate),
                prices,
                with_frontier=False,
                with_allocation=False,
            )
            for rate in (0.0, 0.02, 0.05)
        }
        sharpes = [results[r].sharpe_ratio for r in (0.0, 0.02, 0.05)]
        assert sharpes[0] > sharpes[1] > sharpes[2]
        assert abs(sharpes[0] - sharpes[1]) > 1e-3

    def test_rate_reaches_the_solver_not_only_the_report(self, universe, prices):
        """Ein anderer Zins muss auch andere Gewichte liefern."""
        low = optimizer.optimize(
            _request(universe, OptimizationObjective.MAX_SHARPE, risk_free_rate=0.0),
            prices,
            with_frontier=False,
            with_allocation=False,
        )
        high = optimizer.optimize(
            _request(universe, OptimizationObjective.MAX_SHARPE, risk_free_rate=0.08),
            prices,
            with_frontier=False,
            with_allocation=False,
        )
        assert low.weights != high.weights

    def test_default_is_not_silently_zero(self):
        """Der Vertrag setzt 0.02, nicht den Paket-Default 0.0."""
        assert PortfolioRequest(tickers=["AAPL", "MSFT"]).risk_free_rate == 0.02


class TestShorting:
    def test_flag_enables_negative_weights(self, universe, prices):
        result = optimizer.optimize(
            _request(
                universe,
                OptimizationObjective.MAX_SHARPE,
                risk_free_rate=0.02,
                allow_shorting=True,
                min_weight=-1.0,
            ),
            prices,
            with_frontier=False,
        )
        assert min(p.weight for p in result.positions) < 0.0

    def test_weights_still_sum_to_one_when_shorting(self, universe, prices):
        result = optimizer.optimize(
            _request(
                universe,
                OptimizationObjective.MAX_SHARPE,
                risk_free_rate=0.02,
                allow_shorting=True,
                min_weight=-1.0,
            ),
            prices,
            with_frontier=False,
        )
        assert sum(p.weight for p in result.positions) == pytest.approx(1.0, abs=1e-9)

    def test_without_flag_stays_long_only(self, universe, prices):
        result = optimizer.optimize(
            _request(universe, OptimizationObjective.MAX_SHARPE, risk_free_rate=0.02),
            prices,
            with_frontier=False,
        )
        assert min(p.weight for p in result.positions) >= 0.0


class TestWeightBounds:
    def test_max_weight_is_respected(self, universe, prices):
        result = optimizer.optimize(
            _request(
                universe, OptimizationObjective.MAX_SHARPE, risk_free_rate=0.02, max_weight=0.2
            ),
            prices,
            with_frontier=False,
        )
        assert max(p.weight for p in result.positions) <= 0.2 + 1e-6

    def test_tight_cap_forces_diversification(self, universe, prices):
        result = optimizer.optimize(
            _request(
                universe, OptimizationObjective.MAX_SHARPE, risk_free_rate=0.02, max_weight=0.15
            ),
            prices,
            with_frontier=False,
        )
        held = [p for p in result.positions if p.weight > 1e-6]
        assert len(held) >= 7


class TestFrontier:
    @pytest.fixture(scope="class")
    @classmethod
    def curve(cls, estimates) -> EfficientFrontier:
        mu, cov = estimates
        return optimizer.compute_frontier(
            mu,
            cov,
            points=25,
            bounds=(0.0, 1.0),
            risk_free_rate=0.02,
            tickers=list(mu.index),
        )

    def test_has_required_columns(self, curve):
        """Vertrag mit viz/frontier.py."""
        for column in ("volatility", "expected_return", "sharpe"):
            assert column in curve.data.columns

    def test_is_sorted_by_volatility(self, curve):
        assert curve.data["volatility"].is_monotonic_increasing

    def test_return_rises_with_risk(self, curve):
        """Die Effizienzgrenze ist monoton: mehr Risiko, mehr erwartete Rendite."""
        assert curve.data["expected_return"].is_monotonic_increasing

    def test_values_are_plausible(self, curve):
        assert (curve.data["volatility"] > 0.0).all()
        assert (curve.data["expected_return"] > -1.0).all()
        assert np.isfinite(curve.data["volatility"].to_numpy()).all()

    def test_carries_weight_columns(self, curve):
        weight_columns = [c for c in curve.data.columns if c.startswith(optimizer.WEIGHT_PREFIX)]
        assert len(weight_columns) == 10

    def test_weight_rows_sum_to_one(self, curve):
        weight_columns = [c for c in curve.data.columns if c.startswith(optimizer.WEIGHT_PREFIX)]
        sums = curve.data[weight_columns].sum(axis=1)
        assert np.allclose(sums.to_numpy(), 1.0, atol=1e-4)

    def test_point_count_is_configurable(self, estimates):
        mu, cov = estimates
        curve = optimizer.compute_frontier(
            mu, cov, points=12, bounds=(0.0, 1.0), risk_free_rate=0.02, tickers=list(mu.index)
        )
        assert 1 <= len(curve.data) <= 12

    def test_minimum_volatility_is_the_left_edge(self, estimates, universe, prices):
        mu, cov = estimates
        curve = optimizer.compute_frontier(
            mu, cov, points=25, bounds=(0.0, 1.0), risk_free_rate=0.02, tickers=list(mu.index)
        )
        minimal = optimizer.optimize(
            _request(universe, OptimizationObjective.MIN_VOLATILITY, risk_free_rate=0.02),
            prices,
            with_frontier=False,
        )
        assert curve.data["volatility"].min() >= minimal.expected_volatility - 1e-6

    def test_sharpe_column_uses_the_rate(self, estimates):
        mu, cov = estimates
        low = optimizer.compute_frontier(
            mu, cov, points=10, bounds=(0.0, 1.0), risk_free_rate=0.0, tickers=list(mu.index)
        )
        high = optimizer.compute_frontier(
            mu, cov, points=10, bounds=(0.0, 1.0), risk_free_rate=0.06, tickers=list(mu.index)
        )
        assert not np.allclose(low.data["sharpe"].to_numpy(), high.data["sharpe"].to_numpy())


class TestAssetScatter:
    def test_shape_matches_viz_contract(self, estimates):
        mu, cov = estimates
        scatter = optimizer.asset_scatter(mu, cov)
        assert list(scatter.columns) == ["volatility", "expected_return"]
        assert len(scatter) == 10

    def test_volatilities_are_positive(self, estimates):
        mu, cov = estimates
        assert (optimizer.asset_scatter(mu, cov)["volatility"] > 0.0).all()


class TestDiscreteAllocation:
    def test_produces_whole_shares(self, universe, prices):
        result = optimizer.optimize(
            _request(universe, OptimizationObjective.MAX_SHARPE, risk_free_rate=0.02),
            prices,
            with_frontier=False,
        )
        held = [p for p in result.positions if p.shares is not None]
        assert held
        for position in held:
            assert position.shares == pytest.approx(round(position.shares))

    def test_leftover_cash_is_never_negative(self, universe, prices):
        result = optimizer.optimize(
            _request(universe, OptimizationObjective.MAX_SHARPE, risk_free_rate=0.02),
            prices,
            with_frontier=False,
        )
        assert result.leftover_cash >= 0.0

    def test_allocation_stays_within_budget(self, universe, prices):
        result = optimizer.optimize(
            _request(
                universe,
                OptimizationObjective.MAX_SHARPE,
                risk_free_rate=0.02,
                total_value=50_000.0,
            ),
            prices,
            with_frontier=False,
        )
        invested = sum(p.value for p in result.positions if p.value is not None)
        assert invested <= 50_000.0 + 1e-6

    def test_can_be_switched_off(self, universe, prices):
        result = optimizer.optimize(
            _request(universe, OptimizationObjective.MAX_SHARPE, risk_free_rate=0.02),
            prices,
            with_frontier=False,
            with_allocation=False,
        )
        assert result.leftover_cash is None
        assert all(p.shares is None for p in result.positions)


class TestResultShape:
    @pytest.fixture(scope="class")
    @classmethod
    def result(cls, universe, prices) -> PortfolioResult:
        return optimizer.optimize(
            _request(universe, OptimizationObjective.MAX_SHARPE, risk_free_rate=0.02), prices
        )

    def test_carries_correlation_matrix(self, result):
        assert result.correlation.shape == (10, 10)

    def test_carries_frontier(self, result):
        assert result.frontier is not None
        assert len(result.frontier.tickers) == 10

    def test_computed_at_is_utc(self, result):
        assert result.computed_at.tzinfo is not None

    def test_provenance_is_from_fixtures(self, result):
        assert result.provenance.source.value == "fixture"

    def test_weights_property_matches_positions(self, result):
        assert result.weights == {p.ticker: p.weight for p in result.positions}

    def test_positions_are_sorted(self, result):
        tickers = [p.ticker for p in result.positions]
        assert tickers == sorted(tickers)


class TestFailureModes:
    def test_missing_tickers_are_reported_as_warning(self, universe, prices):
        request = PortfolioRequest(tickers=[*universe, "DOESNOTEXIST"], risk_free_rate=0.02)
        result = optimizer.optimize(request, prices, with_frontier=False)
        assert any("DOESNOTEXIST" in w for w in result.warnings)
        assert len(result.positions) == 10

    def test_fewer_than_two_priced_tickers_raises(self, provider, prices):
        request = PortfolioRequest(tickers=["AAPL", "NOPE"], risk_free_rate=0.02)
        with pytest.raises(InsufficientDataError):
            optimizer.optimize(request, {"AAPL": prices["AAPL"]})

    def test_unreachable_target_return_fails_loudly(self, universe, prices):
        """Ein unmoegliches Ziel darf kein stilles Ergebnis liefern."""
        request = PortfolioRequest(
            tickers=universe,
            objective=OptimizationObjective.EFFICIENT_RETURN,
            target_return=50.0,
            risk_free_rate=0.02,
        )
        with pytest.raises(OptimizationFailedError):
            optimizer.optimize(request, prices, with_frontier=False)

    def test_zero_weight_sum_is_rejected(self):
        with pytest.raises(OptimizationFailedError):
            optimizer._normalize_weights({"A": 0.0, "B": 0.0})


class TestNormalizeWeights:
    def test_rounding_residue_is_absorbed(self):
        """clean_weights rundet auf fuenf Stellen und verletzt so die Summe."""
        raw = {"A": 0.33333, "B": 0.33333, "C": 0.33333}
        out = optimizer._normalize_weights(raw)
        assert sum(out.values()) == pytest.approx(1.0, abs=1e-15)

    def test_already_normalized_stays_put(self):
        raw = {"A": 0.5, "B": 0.5}
        out = optimizer._normalize_weights(raw)
        assert out == pytest.approx(raw)

    def test_residue_goes_to_largest_position(self):
        out = optimizer._normalize_weights({"A": 0.9, "B": 0.0999})
        assert sum(out.values()) == pytest.approx(1.0, abs=1e-15)
        assert out["A"] > out["B"]

    def test_negative_weights_survive_normalization(self):
        out = optimizer._normalize_weights({"A": 1.4, "B": -0.4})
        assert sum(out.values()) == pytest.approx(1.0, abs=1e-15)
        assert out["B"] < 0.0


class TestProperties:
    @given(
        objective=st.sampled_from(
            [
                OptimizationObjective.MAX_SHARPE,
                OptimizationObjective.MIN_VOLATILITY,
                OptimizationObjective.HRP,
                OptimizationObjective.EQUAL_WEIGHT,
            ]
        ),
        count=st.integers(min_value=2, max_value=10),
    )
    @settings(
        max_examples=15,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_weights_always_sum_to_one(self, universe, prices, objective, count):
        """Vertragsregel ueber beliebigen Teiluniversen."""
        subset = universe[:count]
        result = optimizer.optimize(
            _request(subset, objective, risk_free_rate=0.02),
            {t: prices[t] for t in subset},
            with_frontier=False,
            with_allocation=False,
        )
        assert abs(sum(p.weight for p in result.positions) - 1.0) <= WEIGHT_SUM_TOLERANCE

    @given(rate=st.floats(min_value=0.0, max_value=0.10))
    @settings(
        max_examples=15,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_sharpe_falls_as_rate_rises(self, universe, prices, rate):
        """Bei fester Rendite und Volatilitaet sinkt Sharpe mit steigendem Zins."""
        result = optimizer.optimize(
            _request(universe, OptimizationObjective.MIN_VOLATILITY, risk_free_rate=rate),
            prices,
            with_frontier=False,
            with_allocation=False,
        )
        expected = (result.expected_return - rate) / result.expected_volatility
        assert result.sharpe_ratio == pytest.approx(expected, rel=1e-9)
