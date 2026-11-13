"""Tests der beiden Backtest-Engines.

Der vectorbt-Lauf geht ueber die echte SPY-Fixture (fuenf Jahre Tagesdaten)
und prueft die Plausibilitaet des Gesamtergebnisses. Der backtrader-Adapter
ist in dieser Umgebung nicht installiert; getestet wird deshalb nur der
Nicht-verfuegbar-Pfad.
"""

from __future__ import annotations

import math

import pytest

from btlite.backtest.bt_engine import BacktraderEngine
from btlite.backtest.strategies import BuyAndHoldStrategy, SmaCrossStrategy
from btlite.backtest.vbt_engine import VectorbtEngine
from btlite.core.errors import BacktestError, EngineUnavailableError
from btlite.data.fixtures import FixtureProvider
from btlite.models.backtest import (
    BacktestRequest,
    BacktestResult,
    Direction,
    StrategyConfig,
    TradeSide,
)
from btlite.models.common import Interval

#: Fenster des SMA-Crossover in den Engine-Tests. Kuerzer als der Default,
#: damit die Fixture-Historie mehrere Trades erzeugt.
FAST, SLOW = 20, 60


@pytest.fixture(scope="module")
def provider() -> FixtureProvider:
    """Offline-Provider auf den eingefrorenen Fixtures."""
    return FixtureProvider()


@pytest.fixture(scope="module")
def spy(provider):
    """Fuenf Jahre SPY-Tagesdaten aus der Fixture."""
    return provider.get_history("SPY", interval=Interval.D1)


@pytest.fixture(scope="module")
def engine() -> VectorbtEngine:
    """Die vectorbt-Engine. Ueberspringt alles, wenn das Extra fehlt.

    ``vectorbt`` ist ein optionales Extra. Ohne es soll die Suite sich
    ueberspringen statt zu scheitern - das Produkt verhaelt sich korrekt und
    meldet die Engine als nicht verfuegbar.
    """
    instance = VectorbtEngine()
    if not instance.is_available():
        pytest.skip("vectorbt ist nicht installiert - optionales Extra")
    return instance


def _request(**overrides) -> BacktestRequest:
    """Baut eine Standard-Anfrage fuer SPY."""
    params = {
        "tickers": ["SPY"],
        "strategy": StrategyConfig(key="sma_cross", params={"fast": FAST, "slow": SLOW}),
        "interval": Interval.D1,
        "init_cash": 100_000.0,
        "fees": 0.0005,
        "slippage": 0.0005,
    }
    params.update(overrides)
    return BacktestRequest(**params)


def _strategy() -> SmaCrossStrategy:
    """SMA-Crossover mit den Testfenstern."""
    return SmaCrossStrategy(StrategyConfig(key="sma_cross", params={"fast": FAST, "slow": SLOW}))


def _signals(spy):
    """SMA-Crossover-Signale auf SPY."""
    return _strategy().generate_signals({"SPY": spy})


@pytest.fixture(scope="module")
def spy_result(engine, spy):
    """Ergebnis eines SMA-Crossover-Laufs ueber fuenf Jahre SPY."""
    return engine.run(_request(), {"SPY": spy}, _signals(spy), benchmark=spy)


# --------------------------------------------------------------------------
# Verfuegbarkeit
# --------------------------------------------------------------------------


def test_vectorbt_engine_is_available(engine):
    assert engine.is_available() is True
    assert engine.name == "vectorbt"


def test_is_available_returns_false_instead_of_raising(engine, monkeypatch):
    """Ein kaputter Import darf nicht nach oben durchschlagen."""

    def boom():
        msg = "kaputte Importkette"
        raise ImportError(msg)

    monkeypatch.setattr(engine, "_import_vbt", boom)
    assert engine.is_available() is False


def test_run_raises_when_engine_unavailable(engine, spy, monkeypatch):
    monkeypatch.setattr(engine, "is_available", lambda: False)
    with pytest.raises(EngineUnavailableError):
        engine.run(_request(), {"SPY": spy}, _signals(spy))


# --------------------------------------------------------------------------
# Der SPY-Lauf: Plausibilitaet des Gesamtergebnisses
# --------------------------------------------------------------------------


def test_result_fulfils_the_contract(spy_result):
    assert isinstance(spy_result, BacktestResult)
    assert spy_result.engine_used == "vectorbt"
    assert "equity" in spy_result.equity_curve.columns
    assert spy_result.equity_curve.index.tz is not None
    assert spy_result.signals is not None


def test_equity_curve_has_the_optional_columns_viz_uses(spy_result):
    for column in ("equity", "cash", "drawdown", "benchmark_equity"):
        assert column in spy_result.equity_curve.columns


def test_equity_curve_starts_at_init_cash_and_stays_positive(spy_result):
    equity = spy_result.equity_curve["equity"]
    assert equity.iloc[0] == pytest.approx(100_000.0, rel=1e-6)
    assert (equity > 0.0).all()
    assert len(equity) > 1_000  # fuenf Jahre Tagesdaten


def test_drawdown_column_is_non_positive(spy_result):
    assert (spy_result.equity_curve["drawdown"] <= 1e-12).all()


def test_spy_metrics_are_plausible(spy_result):
    metrics = spy_result.metrics
    # Eine Trendfolge auf SPY ueber fuenf Jahre: die Groessenordnungen
    # muessen stimmen, ohne den Wert selbst festzunageln.
    assert metrics.cagr is not None
    assert -0.5 < metrics.cagr < 0.5
    assert metrics.volatility is not None
    assert 0.0 < metrics.volatility < 0.6
    assert metrics.sharpe is not None
    assert -5.0 < metrics.sharpe < 5.0
    assert metrics.max_drawdown is not None
    assert -1.0 <= metrics.max_drawdown < 0.0
    assert metrics.max_drawdown_duration_days is not None
    assert 0 <= metrics.max_drawdown_duration_days <= 5 * 365


def test_spy_trade_statistics_are_plausible(spy_result):
    metrics = spy_result.metrics
    assert metrics.n_trades is not None
    assert metrics.n_trades >= 1
    assert metrics.hit_rate is not None
    assert 0.0 <= metrics.hit_rate <= 1.0
    assert metrics.exposure is not None
    assert 0.0 < metrics.exposure <= 1.0
    assert metrics.total_fees is not None
    assert metrics.total_fees > 0.0


def test_trade_list_is_consistent(spy_result):
    assert spy_result.trades
    for trade in spy_result.trades:
        assert trade.ticker == "SPY"
        assert trade.side is TradeSide.LONG
        assert trade.entry_price > 0.0
        assert trade.size > 0.0
        assert trade.fees_paid >= 0.0
        if not trade.is_open:
            assert trade.exit_time is not None
            assert trade.exit_time >= trade.entry_time
            assert trade.duration_bars is not None
            assert trade.duration_bars >= 0
            assert trade.pnl is not None


def test_closed_trade_count_matches_metrics(spy_result):
    closed = [t for t in spy_result.trades if not t.is_open]
    assert spy_result.metrics.n_trades == len(closed)


def test_benchmark_metrics_are_filled(spy_result):
    bench = spy_result.benchmark_metrics
    assert bench is not None
    assert bench.total_return is not None
    assert bench.cagr is not None
    # Die Benchmark handelt nicht.
    assert bench.n_trades is None
    assert bench.exposure == pytest.approx(1.0)


def test_beta_against_own_benchmark_is_between_zero_and_one(spy_result):
    # Die Strategie ist nur zeitweise investiert, ihr Beta gegen SPY muss
    # deshalb unter 1 liegen, aber deutlich positiv sein.
    assert spy_result.metrics.beta is not None
    assert 0.0 < spy_result.metrics.beta < 1.0


def test_no_metric_is_nan_or_infinite(spy_result):
    for name, value in spy_result.metrics.model_dump().items():
        if isinstance(value, float):
            assert math.isfinite(value), f"{name} ist {value}"


# --------------------------------------------------------------------------
# Kosten wirken
# --------------------------------------------------------------------------


def test_higher_fees_reduce_the_final_equity(engine, spy):
    signals = _signals(spy)
    cheap = engine.run(_request(fees=0.0, slippage=0.0), {"SPY": spy}, signals)
    pricey = engine.run(_request(fees=0.01, slippage=0.01), {"SPY": spy}, signals)

    assert pricey.equity_curve["equity"].iloc[-1] < cheap.equity_curve["equity"].iloc[-1]
    assert pricey.metrics.total_fees > cheap.metrics.total_fees
    assert cheap.metrics.total_fees == pytest.approx(0.0, abs=1e-9)


def test_higher_slippage_alone_reduces_the_final_equity(engine, spy):
    signals = _signals(spy)
    low = engine.run(_request(fees=0.0, slippage=0.0), {"SPY": spy}, signals)
    high = engine.run(_request(fees=0.0, slippage=0.02), {"SPY": spy}, signals)
    assert high.equity_curve["equity"].iloc[-1] < low.equity_curve["equity"].iloc[-1]


def test_costs_apply_on_entry_and_exit(engine, spy):
    """Bei Ein- und Ausstieg fallen je Kosten an.

    Ein geschlossener Trade traegt deshalb rund doppelt so viel Gebuehr wie
    ein einzelner Ordervorgang.
    """
    fees = 0.01
    result = engine.run(_request(fees=fees, slippage=0.0), {"SPY": spy}, _signals(spy))
    closed = [t for t in result.trades if not t.is_open]
    assert closed

    trade = closed[0]
    entry_volume = trade.entry_price * trade.size
    assert trade.exit_price is not None
    exit_volume = trade.exit_price * trade.size
    expected = (entry_volume + exit_volume) * fees
    assert trade.fees_paid == pytest.approx(expected, rel=0.02)


# --------------------------------------------------------------------------
# Execution-Lag und Signalverarbeitung
# --------------------------------------------------------------------------


def test_execution_lag_delays_the_first_trade(engine, spy):
    signals = _signals(spy)
    lagged = engine.run(_request(), {"SPY": spy}, signals)

    immediate = signals.model_copy(update={"execution_lag": 0})
    instant = engine.run(_request(), {"SPY": spy}, immediate)

    assert lagged.trades
    assert instant.trades
    # Mit Lag wird spaeter eingestiegen als ohne.
    assert lagged.trades[0].entry_time > instant.trades[0].entry_time


def test_buy_and_hold_tracks_the_benchmark_closely(engine, spy):
    signals = BuyAndHoldStrategy().generate_signals({"SPY": spy})
    result = engine.run(
        _request(
            strategy=StrategyConfig(key="buy_and_hold", params={}),
            fees=0.0,
            slippage=0.0,
        ),
        {"SPY": spy},
        signals,
        benchmark=spy,
    )
    assert result.benchmark_metrics is not None
    assert result.metrics.total_return is not None
    # Kostenfreies Buy and Hold auf SPY gegen die SPY-Benchmark: fast
    # deckungsgleich, bis auf den einen Bar Ausfuehrungsverzoegerung.
    assert result.metrics.total_return == pytest.approx(
        result.benchmark_metrics.total_return, rel=0.05
    )
    assert result.metrics.exposure is not None
    assert result.metrics.exposure > 0.95


def test_multi_ticker_run_shares_one_cash_account(engine, provider):
    tickers = ["AAPL", "MSFT", "JNJ"]
    prices = {t: provider.get_history(t, interval=Interval.D1) for t in tickers}
    signals = _strategy().generate_signals(prices)

    result = engine.run(_request(tickers=tickers), prices, signals)
    equity = result.equity_curve["equity"]
    # Gemeinsame Kasse: der Start ist das Startkapital, nicht dessen
    # Dreifaches.
    assert equity.iloc[0] == pytest.approx(100_000.0, rel=1e-6)
    assert {t.ticker for t in result.trades} <= set(tickers)


def test_missing_prices_for_a_signal_column_produce_a_warning(engine, provider, spy):
    prices = {"SPY": spy, "AAPL": provider.get_history("AAPL", interval=Interval.D1)}
    signals = _strategy().generate_signals(prices)

    # AAPL-Kurse absichtlich weglassen.
    result = engine.run(_request(tickers=["SPY", "AAPL"]), {"SPY": spy}, signals)
    assert any("AAPL" in w for w in result.warnings)


def test_run_without_any_matching_prices_raises(engine, provider, spy):
    aapl = provider.get_history("AAPL", interval=Interval.D1)
    signals = _strategy().generate_signals({"AAPL": aapl})
    with pytest.raises(BacktestError):
        engine.run(_request(tickers=["AAPL"]), {"SPY": spy}, signals)


def test_short_direction_produces_short_trades(engine, spy):
    signals = _signals(spy)
    result = engine.run(
        _request(
            strategy=StrategyConfig(
                key="sma_cross",
                params={"fast": FAST, "slow": SLOW},
                direction=Direction.LONG_SHORT,
            )
        ),
        {"SPY": spy},
        signals,
    )
    sides = {t.side for t in result.trades}
    assert TradeSide.SHORT in sides


def test_benchmark_outside_the_period_is_reported_as_warning(engine, spy):
    truncated = spy.model_copy(update={"data": spy.data.iloc[:1].copy()})
    result = engine.run(_request(), {"SPY": spy}, _signals(spy), benchmark=truncated)
    assert result.benchmark_metrics is None
    assert any("Benchmark" in w for w in result.warnings)
    assert result.metrics.beta is None


def test_run_without_benchmark_leaves_relative_metrics_open(engine, spy):
    result = engine.run(_request(), {"SPY": spy}, _signals(spy))
    assert result.benchmark_metrics is None
    assert result.metrics.beta is None
    assert result.metrics.alpha is None
    assert "benchmark_equity" not in result.equity_curve.columns


# --------------------------------------------------------------------------
# BacktraderEngine: nur der Nicht-verfuegbar-Pfad
# --------------------------------------------------------------------------


def test_backtrader_engine_name():
    assert BacktraderEngine().name == "backtrader"


def test_backtrader_engine_is_not_available():
    # backtrader ist in dieser Umgebung bewusst nicht installiert.
    assert BacktraderEngine().is_available() is False


def test_backtrader_run_raises_engine_unavailable(spy):
    engine = BacktraderEngine()
    with pytest.raises(EngineUnavailableError) as excinfo:
        engine.run(_request(engine="backtrader"), {"SPY": spy}, _signals(spy))
    assert "backtrader" in excinfo.value.user_message
    assert excinfo.value.detail is not None


def test_backtrader_is_available_swallows_import_errors(monkeypatch):
    engine = BacktraderEngine()

    def boom():
        msg = "kaputte Importkette"
        raise ImportError(msg)

    monkeypatch.setattr(engine, "_import_bt", boom)
    assert engine.is_available() is False


def test_both_engines_satisfy_the_protocol():
    from btlite.backtest.base import BacktestEngine

    assert isinstance(VectorbtEngine(), BacktestEngine)
    assert isinstance(BacktraderEngine(), BacktestEngine)


class TestResultCurrency:
    """Das Ergebnis nennt die Waehrung der Eingangsdaten, nicht den Default.

    Keine Engine setzte die Waehrung, bis Euro zur Vorgabe wurde: ein Lauf
    auf SPY - Kurse in Dollar - haette sich dann selbst als Euro-Ergebnis
    beschriftet, samt Achsenbeschriftung der Equity-Kurve. Der Backtest
    rechnet nicht um; er sagt jetzt richtig, worin er rechnet.
    """

    def test_currency_comes_from_the_price_series(self, spy_result, spy):
        assert spy_result.currency is spy.currency

    def test_the_model_default_is_not_inherited(self, spy_result):
        """SPY notiert in Dollar - Euro waere hier der Default, nicht die Wahrheit."""
        from btlite.models.common import Currency

        assert spy_result.currency is Currency.USD
        assert spy_result.currency is not Currency.EUR
