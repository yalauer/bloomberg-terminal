"""Tests der Performance-Attribution (Block 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from btlite.core.errors import InsufficientDataError
from btlite.data.fixtures import FixtureProvider
from btlite.models.common import Interval, Provenance
from btlite.models.market import PriceSeries
from btlite.portfolio import attribution


def _series(ticker: str, closes: list[float], provenance: Provenance) -> PriceSeries:
    """Baut eine Kursreihe mit vorgegebenen Schlusskursen."""
    idx = pd.date_range("2024-01-02", periods=len(closes), freq="B", tz="UTC", name="date")
    data = pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1_000_000.0] * len(closes),
        },
        index=idx,
    )
    return PriceSeries(
        ticker=ticker,
        interval=Interval.D1,
        adjusted=True,
        data=data,
        provenance=provenance,
    )


@pytest.fixture
def simple_prices(provenance) -> dict[str, PriceSeries]:
    """Drei Titel mit exakt bekannten Renditen: +10, -20 und +50 Prozent."""
    return {
        "AAA": _series("AAA", [100.0, 105.0, 110.0], provenance),
        "BBB": _series("BBB", [50.0, 45.0, 40.0], provenance),
        "CCC": _series("CCC", [20.0, 25.0, 30.0], provenance),
    }


@pytest.fixture
def fixture_prices():
    provider = FixtureProvider()
    return provider.get_histories(["AAPL", "MSFT", "NVDA", "KO"])


class TestPeriodReturn:
    def test_matches_hand_computed(self, simple_prices):
        """110/100-1 = +0.10, 40/50-1 = -0.20, 30/20-1 = +0.50."""
        assert attribution.period_return(simple_prices["AAA"]) == pytest.approx(0.10)
        assert attribution.period_return(simple_prices["BBB"]) == pytest.approx(-0.20)
        assert attribution.period_return(simple_prices["CCC"]) == pytest.approx(0.50)

    def test_single_observation_is_none(self, provenance):
        assert attribution.period_return(_series("X", [100.0], provenance)) is None


class TestAttribute:
    def test_has_required_columns(self, simple_prices, provenance):
        """Vertrag: weight, return, contribution."""
        result = attribution.attribute(
            {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, simple_prices, provenance=provenance
        )
        for column in ("weight", "return", "contribution"):
            assert column in result.data.columns

    def test_contribution_is_weight_times_return(self, simple_prices, provenance):
        result = attribution.attribute(
            {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, simple_prices, provenance=provenance
        )
        expected = result.data["weight"] * result.data["return"]
        assert np.allclose(result.data["contribution"].to_numpy(), expected.to_numpy())

    def test_total_return_matches_hand_computed(self, simple_prices, provenance):
        """0.5*0.10 + 0.3*(-0.20) + 0.2*0.50 = 0.05 - 0.06 + 0.10 = 0.09."""
        result = attribution.attribute(
            {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, simple_prices, provenance=provenance
        )
        assert result.total_return == pytest.approx(0.09, abs=1e-12)

    def test_contributions_sum_to_total(self, simple_prices, provenance):
        """Die Kernidentitaet der Zerlegung."""
        result = attribution.attribute(
            {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, simple_prices, provenance=provenance
        )
        assert result.data["contribution"].sum() == pytest.approx(result.total_return)

    def test_index_is_ticker(self, simple_prices, provenance):
        result = attribution.attribute(
            {"AAA": 0.5, "BBB": 0.5}, simple_prices, provenance=provenance
        )
        assert result.data.index.name == "ticker"
        assert sorted(result.data.index) == ["AAA", "BBB"]

    def test_period_bounds_are_recorded(self, simple_prices, provenance):
        result = attribution.attribute({"AAA": 1.0}, simple_prices, provenance=provenance)
        assert result.period_start is not None
        assert result.period_start <= result.period_end

    def test_tickers_without_prices_are_dropped(self, simple_prices, provenance):
        result = attribution.attribute(
            {"AAA": 0.5, "BBB": 0.3, "MISSING": 0.2}, simple_prices, provenance=provenance
        )
        assert "MISSING" not in result.data.index

    def test_empty_weights_raise(self, simple_prices, provenance):
        with pytest.raises(InsufficientDataError):
            attribution.attribute({}, simple_prices, provenance=provenance)

    def test_no_overlap_raises(self, simple_prices, provenance):
        with pytest.raises(InsufficientDataError):
            attribution.attribute({"NOPE": 1.0}, simple_prices, provenance=provenance)

    def test_negative_weights_are_allowed(self, simple_prices, provenance):
        """Ein Short traegt bei fallendem Kurs positiv bei."""
        result = attribution.attribute(
            {"AAA": 1.2, "BBB": -0.2}, simple_prices, provenance=provenance
        )
        assert result.data.loc["BBB", "contribution"] == pytest.approx(0.04)


class TestBenchmarkComparison:
    def test_adds_optional_columns(self, simple_prices, provenance):
        result = attribution.attribute(
            {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
            simple_prices,
            benchmark_weights={"AAA": 0.34, "BBB": 0.33, "CCC": 0.33},
            provenance=provenance,
        )
        assert "active_weight" in result.data.columns
        assert "allocation_effect" in result.data.columns

    def test_active_weight_is_the_difference(self, simple_prices, provenance):
        result = attribution.attribute(
            {"AAA": 0.5, "BBB": 0.5},
            simple_prices,
            benchmark_weights={"AAA": 0.3, "BBB": 0.7},
            provenance=provenance,
        )
        assert result.data.loc["AAA", "active_weight"] == pytest.approx(0.2)
        assert result.data.loc["BBB", "active_weight"] == pytest.approx(-0.2)

    def test_active_return_is_the_difference(self, simple_prices, provenance):
        result = attribution.attribute(
            {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
            simple_prices,
            benchmark_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
            provenance=provenance,
        )
        assert result.active_return == pytest.approx(0.0, abs=1e-12)

    def test_identical_weights_give_no_active_weight(self, simple_prices, provenance):
        result = attribution.attribute(
            {"AAA": 0.5, "BBB": 0.5},
            simple_prices,
            benchmark_weights={"AAA": 0.5, "BBB": 0.5},
            provenance=provenance,
        )
        assert np.allclose(result.data["active_weight"].to_numpy(), 0.0)

    def test_benchmark_return_matches_hand_computed(self, simple_prices, provenance):
        """0.34*0.10 + 0.33*(-0.20) + 0.33*0.50 = 0.034 - 0.066 + 0.165 = 0.133."""
        result = attribution.attribute(
            {"AAA": 1.0},
            simple_prices,
            benchmark_weights={"AAA": 0.34, "BBB": 0.33, "CCC": 0.33},
            provenance=provenance,
        )
        assert result.benchmark_return == pytest.approx(0.133, abs=1e-12)

    def test_overweighting_a_winner_helps_allocation(self, simple_prices, provenance):
        """CCC laeuft besser als die Benchmark: Uebergewicht wirkt positiv."""
        result = attribution.attribute(
            {"AAA": 0.2, "BBB": 0.2, "CCC": 0.6},
            simple_prices,
            benchmark_weights={"AAA": 0.34, "BBB": 0.33, "CCC": 0.33},
            provenance=provenance,
        )
        assert result.data.loc["CCC", "allocation_effect"] > 0.0

    def test_benchmark_returns_can_be_injected(self, simple_prices, provenance):
        result = attribution.attribute(
            {"AAA": 1.0},
            simple_prices,
            benchmark_weights={"AAA": 1.0},
            benchmark_returns={"AAA": 0.05},
            provenance=provenance,
        )
        assert result.benchmark_return == pytest.approx(0.05)
        assert result.active_return == pytest.approx(0.05)


class TestAttributeFromReturns:
    def test_matches_hand_computed(self, provenance):
        result = attribution.attribute_from_returns(
            {"AAA": 0.6, "BBB": 0.4}, {"AAA": 0.10, "BBB": -0.05}, provenance=provenance
        )
        assert result.total_return == pytest.approx(0.6 * 0.10 + 0.4 * -0.05)

    def test_has_required_columns(self, provenance):
        result = attribution.attribute_from_returns(
            {"AAA": 1.0}, {"AAA": 0.1}, provenance=provenance
        )
        for column in ("weight", "return", "contribution"):
            assert column in result.data.columns

    def test_nan_returns_are_dropped(self, provenance):
        result = attribution.attribute_from_returns(
            {"AAA": 0.5, "BBB": 0.5},
            {"AAA": 0.1, "BBB": float("nan")},
            provenance=provenance,
        )
        assert "BBB" not in result.data.index

    def test_no_usable_returns_raise(self, provenance):
        with pytest.raises(InsufficientDataError):
            attribution.attribute_from_returns({"AAA": 1.0}, {"BBB": 0.1}, provenance=provenance)


class TestTopContributors:
    def test_orders_by_contribution(self, simple_prices, provenance):
        result = attribution.attribute(
            {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, simple_prices, provenance=provenance
        )
        top = attribution.top_contributors(result, count=1)
        assert top.index[0] == "CCC"

    def test_ascending_finds_the_drag(self, simple_prices, provenance):
        result = attribution.attribute(
            {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, simple_prices, provenance=provenance
        )
        worst = attribution.top_contributors(result, count=1, ascending=True)
        assert worst.index[0] == "BBB"

    def test_count_limits_rows(self, simple_prices, provenance):
        result = attribution.attribute(
            {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, simple_prices, provenance=provenance
        )
        assert len(attribution.top_contributors(result, count=2)) == 2


class TestWithFixtures:
    def test_runs_over_fixture_universe(self, fixture_prices, provenance):
        weights = dict.fromkeys(fixture_prices, 0.25)
        result = attribution.attribute(weights, fixture_prices, provenance=provenance)
        assert len(result.data) == 4
        assert result.total_return is not None

    def test_contributions_sum_to_total(self, fixture_prices, provenance):
        weights = dict.fromkeys(fixture_prices, 0.25)
        result = attribution.attribute(weights, fixture_prices, provenance=provenance)
        assert result.data["contribution"].sum() == pytest.approx(result.total_return)


class TestWeightsSumToOne:
    def test_accepts_exact_sum(self):
        assert attribution.weights_sum_to_one({"A": 0.5, "B": 0.5})

    def test_rejects_clear_deviation(self):
        assert not attribution.weights_sum_to_one({"A": 0.5, "B": 0.4})

    def test_tolerates_float_noise(self):
        assert attribution.weights_sum_to_one({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})


class TestProperties:
    @given(
        weights=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=2, max_size=6
        ),
        returns=st.lists(
            st.floats(min_value=-0.9, max_value=2.0, allow_nan=False), min_size=2, max_size=6
        ),
    )
    @settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_contribution_identity_always_holds(self, weights, returns, provenance):
        """Beitrag ist weight mal return, ueber beliebigen Eingaben."""
        count = min(len(weights), len(returns))
        tickers = [f"T{i}" for i in range(count)]
        result = attribution.attribute_from_returns(
            dict(zip(tickers, weights[:count], strict=True)),
            dict(zip(tickers, returns[:count], strict=True)),
            provenance=provenance,
        )
        product = result.data["weight"] * result.data["return"]
        assert np.allclose(result.data["contribution"].to_numpy(), product.to_numpy())

    @given(
        returns=st.lists(
            st.floats(min_value=-0.5, max_value=0.5, allow_nan=False), min_size=2, max_size=6
        )
    )
    @settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_equal_weights_give_the_mean(self, returns, provenance):
        """Bei Gleichgewichtung ist die Portfoliorendite der Mittelwert."""
        tickers = [f"T{i}" for i in range(len(returns))]
        share = 1.0 / len(returns)
        result = attribution.attribute_from_returns(
            dict.fromkeys(tickers, share),
            dict(zip(tickers, returns, strict=True)),
            provenance=provenance,
        )
        assert result.total_return == pytest.approx(sum(returns) / len(returns), abs=1e-12)
