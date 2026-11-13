"""Tests der Drift-Berechnung und Order-Vorschlaege (Block 5).

Das Projekt hat bewusst keine Broker-Anbindung. Dieses Modul erzeugt nur
Vorschlaege - die Tests halten das ausdruecklich fest.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from btlite.core.errors import PortfolioError
from btlite.models.portfolio import PortfolioRequest, PortfolioResult, Position
from btlite.portfolio import rebalance
from btlite.portfolio.rebalance import OrderSide


@pytest.fixture
def prices() -> dict[str, float]:
    """Kurse mit glatten Werten, damit Stueckzahlen von Hand pruefbar sind."""
    return {"AAA": 100.0, "BBB": 50.0, "CCC": 25.0}


@pytest.fixture
def holdings() -> dict[str, float]:
    """Bestand mit Gesamtwert 100*50 + 50*100 + 25*200 = 15000."""
    return {"AAA": 50.0, "BBB": 100.0, "CCC": 200.0}


@pytest.fixture
def result(provenance) -> PortfolioResult:
    return PortfolioResult(
        request=PortfolioRequest(tickers=["AAA", "BBB", "CCC"]),
        positions=[
            Position(ticker="AAA", weight=0.5),
            Position(ticker="BBB", weight=0.3),
            Position(ticker="CCC", weight=0.2),
        ],
        computed_at=datetime(2025, 1, 2, tzinfo=UTC),
        provenance=provenance,
    )


class TestPortfolioValue:
    def test_matches_hand_computed(self, holdings, prices):
        """50*100 + 100*50 + 200*25 = 5000 + 5000 + 5000 = 15000."""
        assert rebalance.portfolio_value(holdings, prices) == pytest.approx(15_000.0)

    def test_cash_is_added(self, holdings, prices):
        assert rebalance.portfolio_value(holdings, prices, cash=1_000.0) == pytest.approx(16_000.0)

    def test_positions_without_price_are_skipped(self, prices):
        assert rebalance.portfolio_value({"AAA": 10.0, "ZZZ": 5.0}, prices) == pytest.approx(
            1_000.0
        )

    def test_empty_holdings_give_cash_only(self, prices):
        assert rebalance.portfolio_value({}, prices, cash=500.0) == pytest.approx(500.0)


class TestCurrentWeights:
    def test_matches_hand_computed(self, holdings, prices):
        """Jede Position ist 5000 von 15000, also exakt ein Drittel."""
        weights = rebalance.current_weights(holdings, prices)
        for ticker in ("AAA", "BBB", "CCC"):
            assert weights[ticker] == pytest.approx(1 / 3)

    def test_weights_sum_to_one(self, holdings, prices):
        assert sum(rebalance.current_weights(holdings, prices).values()) == pytest.approx(1.0)

    def test_missing_price_raises(self, prices):
        with pytest.raises(PortfolioError, match="Kurs"):
            rebalance.current_weights({"AAA": 10.0, "ZZZ": 1.0}, prices)

    def test_zero_value_gives_empty(self, prices):
        assert rebalance.current_weights({"AAA": 0.0}, prices) == {}

    def test_short_position_is_negative(self, prices):
        weights = rebalance.current_weights({"AAA": 100.0, "BBB": -50.0}, prices)
        assert weights["BBB"] < 0.0


class TestComputeDrift:
    def test_drift_is_current_minus_target(self):
        drift = rebalance.compute_drift({"AAA": 0.6, "BBB": 0.4}, {"AAA": 0.5, "BBB": 0.5})
        assert drift.loc["AAA", "drift"] == pytest.approx(0.1)
        assert drift.loc["BBB", "drift"] == pytest.approx(-0.1)

    def test_has_expected_columns(self):
        drift = rebalance.compute_drift({"AAA": 1.0}, {"AAA": 1.0})
        for column in ("current_weight", "target_weight", "drift"):
            assert column in drift.columns

    def test_value_columns_appear_with_total(self):
        drift = rebalance.compute_drift(
            {"AAA": 0.6, "BBB": 0.4}, {"AAA": 0.5, "BBB": 0.5}, total_value=10_000.0
        )
        for column in ("current_value", "target_value", "trade_value"):
            assert column in drift.columns

    def test_trade_value_is_signed_hand_computed(self):
        """AAA ist 10 Punkte uebergewichtet: bei 10000 sind das -1000 zu verkaufen."""
        drift = rebalance.compute_drift(
            {"AAA": 0.6, "BBB": 0.4}, {"AAA": 0.5, "BBB": 0.5}, total_value=10_000.0
        )
        assert drift.loc["AAA", "trade_value"] == pytest.approx(-1_000.0)
        assert drift.loc["BBB", "trade_value"] == pytest.approx(1_000.0)

    def test_new_target_position_appears(self):
        drift = rebalance.compute_drift({"AAA": 1.0}, {"AAA": 0.5, "NEW": 0.5})
        assert "NEW" in drift.index
        assert drift.loc["NEW", "current_weight"] == 0.0

    def test_dropped_position_appears(self):
        drift = rebalance.compute_drift({"OLD": 0.5, "AAA": 0.5}, {"AAA": 1.0})
        assert drift.loc["OLD", "target_weight"] == 0.0
        assert drift.loc["OLD", "drift"] == pytest.approx(0.5)

    def test_index_is_ticker(self):
        drift = rebalance.compute_drift({"AAA": 1.0}, {"AAA": 1.0})
        assert drift.index.name == "ticker"


class TestNeedsRebalancing:
    def test_true_above_threshold(self):
        drift = rebalance.compute_drift({"AAA": 0.7, "BBB": 0.3}, {"AAA": 0.5, "BBB": 0.5})
        assert rebalance.needs_rebalancing(drift, threshold=0.05)

    def test_false_below_threshold(self):
        drift = rebalance.compute_drift({"AAA": 0.51, "BBB": 0.49}, {"AAA": 0.5, "BBB": 0.5})
        assert not rebalance.needs_rebalancing(drift, threshold=0.05)

    def test_false_when_on_target(self):
        drift = rebalance.compute_drift({"AAA": 0.5, "BBB": 0.5}, {"AAA": 0.5, "BBB": 0.5})
        assert not rebalance.needs_rebalancing(drift)

    def test_empty_frame_is_false(self):
        drift = rebalance.compute_drift({}, {})
        assert not rebalance.needs_rebalancing(drift)


class TestProposeOrders:
    def test_buy_and_sell_directions(self, prices):
        """AAA untergewichtet -> Kauf, BBB uebergewichtet -> Verkauf."""
        proposals = rebalance.propose_orders(
            {"AAA": 0.3, "BBB": 0.7},
            {"AAA": 0.5, "BBB": 0.5},
            prices,
            total_value=10_000.0,
        )
        sides = {p.ticker: p.side for p in proposals}
        assert sides["AAA"] is OrderSide.BUY
        assert sides["BBB"] is OrderSide.SELL

    def test_shares_are_hand_computable(self, prices):
        """2000 Euro zu kaufen bei Kurs 100 sind genau 20 Stueck."""
        proposals = rebalance.propose_orders(
            {"AAA": 0.3, "BBB": 0.7},
            {"AAA": 0.5, "BBB": 0.5},
            prices,
            total_value=10_000.0,
        )
        buy = next(p for p in proposals if p.ticker == "AAA")
        assert buy.shares == pytest.approx(20.0)

    def test_amounts_are_positive(self, prices):
        proposals = rebalance.propose_orders(
            {"AAA": 0.3, "BBB": 0.7},
            {"AAA": 0.5, "BBB": 0.5},
            prices,
            total_value=10_000.0,
        )
        assert all(p.amount > 0.0 for p in proposals)

    def test_no_proposal_when_on_target(self, prices):
        proposals = rebalance.propose_orders(
            {"AAA": 0.5, "BBB": 0.5},
            {"AAA": 0.5, "BBB": 0.5},
            prices,
            total_value=10_000.0,
        )
        assert proposals == []

    def test_threshold_suppresses_small_drift(self, prices):
        proposals = rebalance.propose_orders(
            {"AAA": 0.52, "BBB": 0.48},
            {"AAA": 0.5, "BBB": 0.5},
            prices,
            total_value=10_000.0,
            threshold=0.05,
        )
        assert proposals == []

    def test_whole_shares_are_integers(self, prices):
        proposals = rebalance.propose_orders(
            {"AAA": 0.31, "BBB": 0.69},
            {"AAA": 0.5, "BBB": 0.5},
            prices,
            total_value=9_999.0,
        )
        for proposal in proposals:
            assert proposal.shares == pytest.approx(round(proposal.shares))

    def test_fractional_shares_when_allowed(self, prices):
        proposals = rebalance.propose_orders(
            {"AAA": 0.31, "BBB": 0.69},
            {"AAA": 0.5, "BBB": 0.5},
            prices,
            total_value=9_999.0,
            whole_shares=False,
        )
        assert any(p.shares != round(p.shares) for p in proposals)

    def test_min_order_value_filters(self, prices):
        proposals = rebalance.propose_orders(
            {"AAA": 0.49, "BBB": 0.51},
            {"AAA": 0.5, "BBB": 0.5},
            prices,
            total_value=10_000.0,
            min_order_value=500.0,
        )
        assert proposals == []

    def test_sorted_by_volume(self, prices):
        proposals = rebalance.propose_orders(
            {"AAA": 0.1, "BBB": 0.6, "CCC": 0.3},
            {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
            prices,
            total_value=10_000.0,
        )
        amounts = [p.amount for p in proposals]
        assert amounts == sorted(amounts, reverse=True)

    def test_missing_price_gives_no_shares(self):
        proposals = rebalance.propose_orders(
            {"AAA": 0.5, "XXX": 0.5},
            {"AAA": 1.0, "XXX": 0.0},
            {"AAA": 100.0},
            total_value=10_000.0,
        )
        without = next(p for p in proposals if p.ticker == "XXX")
        assert without.shares is None
        assert without.price is None

    def test_non_positive_value_raises(self, prices):
        with pytest.raises(PortfolioError):
            rebalance.propose_orders({"AAA": 1.0}, {"AAA": 1.0}, prices, total_value=0.0)

    def test_proposals_carry_context(self, prices):
        proposals = rebalance.propose_orders(
            {"AAA": 0.3, "BBB": 0.7},
            {"AAA": 0.5, "BBB": 0.5},
            prices,
            total_value=10_000.0,
        )
        buy = next(p for p in proposals if p.ticker == "AAA")
        assert buy.current_weight == pytest.approx(0.3)
        assert buy.target_weight == pytest.approx(0.5)
        assert buy.drift == pytest.approx(-0.2)


class TestOrderProposal:
    def test_repr_is_readable(self, prices):
        proposals = rebalance.propose_orders(
            {"AAA": 0.3, "BBB": 0.7},
            {"AAA": 0.5, "BBB": 0.5},
            prices,
            total_value=10_000.0,
        )
        assert "AAA" in repr(proposals[0]) or "BBB" in repr(proposals[0])

    def test_equality_is_fieldwise(self):
        args = {
            "ticker": "AAA",
            "side": OrderSide.BUY,
            "amount": 100.0,
            "shares": 1.0,
            "price": 100.0,
            "current_weight": 0.4,
            "target_weight": 0.5,
            "drift": -0.1,
        }
        assert rebalance.OrderProposal(**args) == rebalance.OrderProposal(**args)

    def test_differs_on_side(self):
        args = {
            "ticker": "AAA",
            "amount": 100.0,
            "shares": 1.0,
            "price": 100.0,
            "current_weight": 0.4,
            "target_weight": 0.5,
            "drift": -0.1,
        }
        buy = rebalance.OrderProposal(side=OrderSide.BUY, **args)
        sell = rebalance.OrderProposal(side=OrderSide.SELL, **args)
        assert buy != sell

    def test_not_equal_to_other_types(self):
        proposal = rebalance.OrderProposal(
            ticker="AAA",
            side=OrderSide.BUY,
            amount=1.0,
            shares=1.0,
            price=1.0,
            current_weight=0.0,
            target_weight=1.0,
            drift=-1.0,
        )
        assert proposal != "AAA"


class TestProposalsToFrame:
    def test_columns_and_index(self, prices):
        proposals = rebalance.propose_orders(
            {"AAA": 0.3, "BBB": 0.7},
            {"AAA": 0.5, "BBB": 0.5},
            prices,
            total_value=10_000.0,
        )
        frame = rebalance.proposals_to_frame(proposals)
        assert frame.index.name == "ticker"
        for column in ("side", "amount", "shares", "price", "drift"):
            assert column in frame.columns

    def test_empty_input_gives_empty_frame(self):
        frame = rebalance.proposals_to_frame([])
        assert frame.empty
        assert "side" in frame.columns

    def test_row_count_matches(self, prices):
        proposals = rebalance.propose_orders(
            {"AAA": 0.1, "BBB": 0.6, "CCC": 0.3},
            {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
            prices,
            total_value=10_000.0,
        )
        assert len(rebalance.proposals_to_frame(proposals)) == len(proposals)


class TestRebalanceToResult:
    def test_produces_proposals(self, holdings, prices, result):
        proposals = rebalance.rebalance_to_result(holdings, result, prices)
        assert proposals

    def test_moves_towards_target(self, holdings, prices, result):
        """Ist ist je ein Drittel, Ziel 0.5/0.3/0.2: AAA kaufen, CCC verkaufen."""
        proposals = {p.ticker: p for p in rebalance.rebalance_to_result(holdings, result, prices)}
        assert proposals["AAA"].side is OrderSide.BUY
        assert proposals["CCC"].side is OrderSide.SELL

    def test_cash_raises_portfolio_value(self, holdings, prices, result):
        without = rebalance.rebalance_to_result(holdings, result, prices)
        with_cash = rebalance.rebalance_to_result(holdings, result, prices, cash=15_000.0)
        assert sum(p.amount for p in with_cash) > sum(p.amount for p in without)

    def test_unpriced_holdings_are_ignored(self, prices, result):
        proposals = rebalance.rebalance_to_result({"AAA": 50.0, "UNKNOWN": 100.0}, result, prices)
        assert all(p.ticker != "UNKNOWN" for p in proposals)

    def test_empty_portfolio_raises(self, prices, result):
        with pytest.raises(PortfolioError):
            rebalance.rebalance_to_result({}, result, prices)

    def test_threshold_is_honoured(self, holdings, prices, result):
        assert rebalance.rebalance_to_result(holdings, result, prices, threshold=0.9) == []


class TestNoExecution:
    """Nicht-Ziel des Projekts: keine Broker-Anbindung, keine Ausfuehrung."""

    def test_module_exposes_no_execution_function(self):
        forbidden = {"execute", "submit", "send_order", "place_order", "trade", "broker"}
        exported = {name.lower() for name in dir(rebalance) if not name.startswith("_")}
        assert not (forbidden & exported)

    def test_proposal_has_no_execute_method(self):
        assert not hasattr(rebalance.OrderProposal, "execute")


class TestProperties:
    @given(
        current=st.floats(min_value=0.0, max_value=1.0),
        target=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=80, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_drift_identity_holds(self, current, target):
        drift = rebalance.compute_drift({"AAA": current}, {"AAA": target})
        assert drift.loc["AAA", "drift"] == pytest.approx(current - target, abs=1e-12)

    @given(
        weights=st.lists(
            st.floats(min_value=0.01, max_value=1.0, allow_nan=False), min_size=2, max_size=5
        ),
        total=st.floats(min_value=1_000.0, max_value=1_000_000.0),
    )
    @settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_trade_values_net_to_zero(self, weights, total):
        """Umschichtung ohne Zufluss: die Handelsbetraege heben sich auf."""
        share = sum(weights)
        current = {f"T{i}": w / share for i, w in enumerate(weights)}
        target = dict.fromkeys(current, 1.0 / len(current))
        drift = rebalance.compute_drift(current, target, total_value=total)
        assert drift["trade_value"].sum() == pytest.approx(0.0, abs=total * 1e-9)

    @given(
        holding=st.floats(min_value=1.0, max_value=1_000.0),
        price=st.floats(min_value=1.0, max_value=1_000.0),
    )
    @settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_single_position_is_always_full_weight(self, holding, price):
        weights = rebalance.current_weights({"AAA": holding}, {"AAA": price})
        assert weights["AAA"] == pytest.approx(1.0)
