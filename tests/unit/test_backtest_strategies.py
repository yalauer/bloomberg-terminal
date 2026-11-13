"""Tests der vier Referenzstrategien.

Kernkriterium dieser Datei ist der **Look-ahead-Test**: die Kursreihe wird
verkuerzt und geprueft, dass sich zuvor erzeugte Signale nicht aendern. Wenn
eine Strategie in Zeile t auf Daten nach t zugreift, muss sich mindestens
ein frueheres Signal verschieben, sobald die Zukunft wegfaellt.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from btlite.backtest.registry import BUILTIN_STRATEGIES, default_registry
from btlite.backtest.strategies import (
    BuyAndHoldStrategy,
    MomentumRotationStrategy,
    RsiMeanReversionStrategy,
    SmaCrossStrategy,
)
from btlite.backtest.strategies._panel import close_panel
from btlite.backtest.strategies.rsi_mean_reversion import RSI_MAX, rsi
from btlite.core.errors import InsufficientDataError, StrategyConfigError
from btlite.models.backtest import SignalSet, StrategyConfig
from btlite.models.common import DataSource, Interval, Provenance
from btlite.models.market import PriceSeries

#: Bars der synthetischen Testreihen. Reicht fuer alle Default-Fenster.
BARS = 600


def _provenance() -> Provenance:
    """Herkunftsnachweis fuer synthetische Reihen."""
    return Provenance(
        source=DataSource.FIXTURE,
        fetched_at=datetime(2026, 1, 2, tzinfo=UTC),
        from_cache=True,
    )


def _price_series(ticker: str, *, seed: int, bars: int = BARS) -> PriceSeries:
    """Baut eine deterministische Kursreihe mit Trendwechseln.

    Der Drift wechselt bewusst das Vorzeichen, damit Crossover- und
    RSI-Strategien tatsaechlich Signale erzeugen.
    """
    idx = pd.date_range("2022-01-03", periods=bars, freq="B", tz="UTC", name="date")
    rng = np.random.default_rng(seed)
    drift = np.where(np.arange(bars) % 200 < 100, 0.0016, -0.0014)
    close = 100.0 * np.exp(np.cumsum(drift + rng.normal(0.0, 0.011, bars)))
    return PriceSeries(
        ticker=ticker,
        interval=Interval.D1,
        adjusted=True,
        data=pd.DataFrame(
            {
                "open": close,
                "high": close * 1.004,
                "low": close * 0.996,
                "close": close,
                "volume": np.full(bars, 1_000_000.0),
            },
            index=idx,
        ),
        provenance=_provenance(),
    )


@pytest.fixture
def single_price() -> dict[str, PriceSeries]:
    """Eine Kursreihe."""
    return {"AAA": _price_series("AAA", seed=1)}


@pytest.fixture
def multi_price() -> dict[str, PriceSeries]:
    """Fuenf Kursreihen fuer die Rotation."""
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    return {t: _price_series(t, seed=i + 1) for i, t in enumerate(tickers)}


def _truncate(prices: dict[str, PriceSeries], *, keep: int) -> dict[str, PriceSeries]:
    """Verkuerzt alle Kursreihen auf die ersten ``keep`` Bars."""
    return {
        ticker: PriceSeries(
            ticker=series.ticker,
            interval=series.interval,
            adjusted=series.adjusted,
            data=series.data.iloc[:keep].copy(),
            provenance=series.provenance,
        )
        for ticker, series in prices.items()
    }


# --------------------------------------------------------------------------
# LOOK-AHEAD: das Kernkriterium
# --------------------------------------------------------------------------

#: Die vier Strategien mit einer Parametrisierung, die auf 600 Bars greift.
_LOOKAHEAD_CASES = [
    pytest.param(SmaCrossStrategy, {"fast": 20, "slow": 60}, id="sma_cross"),
    pytest.param(
        RsiMeanReversionStrategy,
        {"period": 14, "oversold": 35.0, "overbought": 65.0},
        id="rsi_mean_reversion",
    ),
    pytest.param(
        MomentumRotationStrategy,
        {"lookback": 60, "top_n": 2, "rebalance_every": 21},
        id="momentum_rotation",
    ),
    pytest.param(BuyAndHoldStrategy, {}, id="buy_and_hold"),
]


@pytest.mark.parametrize(("strategy_cls", "params"), _LOOKAHEAD_CASES)
@pytest.mark.parametrize("keep", [300, 420, 500])
def test_no_look_ahead_bias_signals_are_stable_under_truncation(
    strategy_cls, params, keep, multi_price
):
    """Signale duerfen sich nicht aendern, wenn die Zukunft wegfaellt.

    Das ist der direkte Nachweis der Look-ahead-Freiheit: ein Signal in
    Zeile t darf nur Daten bis t nutzen. Wer die Zukunft mitliest, erzeugt
    auf der verkuerzten Reihe zwangslaeufig andere fruehere Signale.
    """
    strategy = strategy_cls(StrategyConfig(key=strategy_cls.key, params=params))

    full = strategy.generate_signals(multi_price)
    short = strategy.generate_signals(_truncate(multi_price, keep=keep))

    common = short.entries.index
    assert len(common) == keep

    pd.testing.assert_frame_equal(
        full.entries.loc[common],
        short.entries,
        check_names=False,
        obj="entries nach Verkuerzung",
    )
    pd.testing.assert_frame_equal(
        full.exits.loc[common],
        short.exits,
        check_names=False,
        obj="exits nach Verkuerzung",
    )


@pytest.mark.parametrize(("strategy_cls", "params"), _LOOKAHEAD_CASES)
def test_no_look_ahead_bias_bar_by_bar(strategy_cls, params, single_price):
    """Prueft Stabilitaet ueber viele Schnitte hintereinander.

    Ein Look-ahead von nur einem Bar faellt bei groben Schnitten
    moeglicherweise nicht auf. Deshalb hier eine feine Staffel.
    """
    strategy = strategy_cls(StrategyConfig(key=strategy_cls.key, params=params))
    full = strategy.generate_signals(single_price)

    for keep in range(400, 600, 17):
        short = strategy.generate_signals(_truncate(single_price, keep=keep))
        index = short.entries.index
        assert full.entries.loc[index].equals(short.entries), f"entries weichen ab bei keep={keep}"
        assert full.exits.loc[index].equals(short.exits), f"exits weichen ab bei keep={keep}"


def test_look_ahead_test_would_catch_a_violation(single_price):
    """Gegenprobe: eine absichtlich fehlerhafte Strategie fliegt auf.

    Ohne diesen Test koennte der Look-ahead-Test gruen sein, weil er
    nichts prueft. Hier schaut die Strategie bewusst einen Bar in die
    Zukunft - der Vergleich muss fehlschlagen.
    """

    class LookAheadStrategy(SmaCrossStrategy):
        key = "look_ahead_broken"

        def generate_signals(self, prices):
            panel = close_panel(prices)
            future = panel.shift(-1)  # BLICK IN DIE ZUKUNFT
            entries = (future > panel).fillna(False).astype(bool)
            exits = (future < panel).fillna(False).astype(bool)
            return SignalSet(
                tickers=list(panel.columns), entries=entries, exits=exits, execution_lag=1
            )

    strategy = LookAheadStrategy(
        StrategyConfig(key="look_ahead_broken", params={"fast": 20, "slow": 60})
    )
    full = strategy.generate_signals(single_price)
    short = strategy.generate_signals(_truncate(single_price, keep=500))

    # Der letzte Bar der verkuerzten Reihe hat keine Zukunft mehr, also
    # weicht dort mindestens eine der beiden Matrizen ab.
    entries_stable = full.entries.loc[short.entries.index].equals(short.entries)
    exits_stable = full.exits.loc[short.exits.index].equals(short.exits)
    assert not (entries_stable and exits_stable)


# --------------------------------------------------------------------------
# SMA Crossover
# --------------------------------------------------------------------------


def test_sma_cross_defaults():
    strategy = SmaCrossStrategy()
    assert strategy.key == "sma_cross"
    assert strategy.params == {"fast": 50, "slow": 200}


def test_sma_cross_rejects_unknown_parameter():
    with pytest.raises(StrategyConfigError, match="Unknown parameters for strategy"):
        SmaCrossStrategy(StrategyConfig(key="sma_cross", params={"windows": 10}))


def test_sma_cross_rejects_fast_not_shorter_than_slow():
    with pytest.raises(
        StrategyConfigError, match="The fast SMA window must be shorter than the slow one"
    ):
        SmaCrossStrategy(StrategyConfig(key="sma_cross", params={"fast": 100, "slow": 100}))


@pytest.mark.parametrize("bad", [0, -5, 2.5, True, None])
def test_sma_cross_rejects_non_positive_int_windows(bad):
    with pytest.raises(StrategyConfigError):
        SmaCrossStrategy(StrategyConfig(key="sma_cross", params={"fast": bad, "slow": 200}))


def test_sma_cross_signals_on_a_constructed_crossover():
    # Konstruierte Reihe: fallend, steigend, wieder fallend. Der schnelle
    # SMA schneidet den langsamen genau einmal von unten (Einstieg) und
    # danach einmal von oben (Ausstieg).
    values = (
        list(np.linspace(100.0, 80.0, 40))
        + list(np.linspace(80.0, 140.0, 40))
        + list(np.linspace(140.0, 90.0, 40))
    )
    idx = pd.date_range("2024-01-01", periods=len(values), freq="B", tz="UTC", name="date")
    series = PriceSeries(
        ticker="AAA",
        interval=Interval.D1,
        adjusted=True,
        data=pd.DataFrame(
            {
                "open": values,
                "high": values,
                "low": values,
                "close": values,
                "volume": [1.0] * len(values),
            },
            index=idx,
        ),
        provenance=_provenance(),
    )
    strategy = SmaCrossStrategy(StrategyConfig(key="sma_cross", params={"fast": 5, "slow": 20}))
    signals = strategy.generate_signals({"AAA": series})

    assert int(signals.entries["AAA"].sum()) == 1
    assert int(signals.exits["AAA"].sum()) == 1
    # Der Einstieg liegt in der Aufwaertsphase, nicht in der Abwaertsphase.
    entry_position = int(np.flatnonzero(signals.entries["AAA"].to_numpy())[0])
    assert entry_position > 40


def test_sma_cross_raises_on_too_short_history():
    strategy = SmaCrossStrategy()  # slow = 200
    short = _truncate({"AAA": _price_series("AAA", seed=3)}, keep=100)
    with pytest.raises(InsufficientDataError):
        strategy.generate_signals(short)


def test_sma_cross_emits_no_signal_before_window_is_full(single_price):
    strategy = SmaCrossStrategy(StrategyConfig(key="sma_cross", params={"fast": 20, "slow": 60}))
    signals = strategy.generate_signals(single_price)
    # Vor dem 60. Bar ist der langsame SMA nicht definiert.
    assert not signals.entries.iloc[:60].to_numpy().any()
    assert not signals.exits.iloc[:60].to_numpy().any()


# --------------------------------------------------------------------------
# RSI Mean Reversion
# --------------------------------------------------------------------------


def test_rsi_defaults():
    strategy = RsiMeanReversionStrategy()
    assert strategy.params == {"period": 14, "oversold": 30.0, "overbought": 70.0}


def test_rsi_rejects_inverted_thresholds():
    with pytest.raises(StrategyConfigError, match="oversold"):
        RsiMeanReversionStrategy(
            StrategyConfig(key="rsi_mean_reversion", params={"oversold": 80.0, "overbought": 20.0})
        )


def test_rsi_rejects_threshold_above_scale():
    with pytest.raises(StrategyConfigError):
        RsiMeanReversionStrategy(
            StrategyConfig(key="rsi_mean_reversion", params={"oversold": 30.0, "overbought": 120.0})
        )


def test_rsi_rejects_unknown_parameter():
    with pytest.raises(StrategyConfigError, match="Unknown parameters for strategy"):
        RsiMeanReversionStrategy(StrategyConfig(key="rsi_mean_reversion", params={"treshold": 30}))


def test_rsi_is_hundred_for_a_monotone_rise():
    values = np.linspace(100.0, 200.0, 60)
    panel = pd.DataFrame({"AAA": values}, index=pd.date_range("2024-01-01", periods=60, tz="UTC"))
    out = rsi(panel, period=14)
    # Ohne jede Abwaertsbewegung ist der RSI definitionsgemaess 100.
    assert out["AAA"].iloc[-1] == pytest.approx(RSI_MAX)


def test_rsi_stays_within_scale(single_price):
    panel = close_panel(single_price)
    out = rsi(panel, period=14).dropna()
    assert float(out.to_numpy().min()) >= 0.0
    assert float(out.to_numpy().max()) <= RSI_MAX


def test_rsi_produces_signals_on_a_volatile_series(single_price):
    strategy = RsiMeanReversionStrategy(
        StrategyConfig(key="rsi_mean_reversion", params={"oversold": 35.0, "overbought": 65.0})
    )
    signals = strategy.generate_signals(single_price)
    assert int(signals.entries.to_numpy().sum()) > 0
    assert int(signals.exits.to_numpy().sum()) > 0


def test_rsi_raises_on_too_short_history():
    strategy = RsiMeanReversionStrategy(
        StrategyConfig(key="rsi_mean_reversion", params={"period": 50})
    )
    with pytest.raises(InsufficientDataError):
        strategy.generate_signals(_truncate({"AAA": _price_series("AAA", seed=4)}, keep=20))


# --------------------------------------------------------------------------
# Momentum Rotation
# --------------------------------------------------------------------------


def test_momentum_defaults():
    strategy = MomentumRotationStrategy()
    assert strategy.params == {"lookback": 126, "top_n": 3, "rebalance_every": 21}


def test_momentum_rejects_unknown_parameter():
    with pytest.raises(StrategyConfigError, match="Unknown parameters for strategy"):
        MomentumRotationStrategy(StrategyConfig(key="momentum_rotation", params={"topn": 3}))


@pytest.mark.parametrize("field", ["lookback", "top_n", "rebalance_every"])
def test_momentum_rejects_non_positive(field):
    with pytest.raises(StrategyConfigError):
        MomentumRotationStrategy(StrategyConfig(key="momentum_rotation", params={field: 0}))


def test_momentum_holds_at_most_top_n_at_any_time(multi_price):
    strategy = MomentumRotationStrategy(
        StrategyConfig(
            key="momentum_rotation", params={"lookback": 60, "top_n": 2, "rebalance_every": 21}
        )
    )
    signals = strategy.generate_signals(multi_price)

    # Positionszustand rekonstruieren und pruefen, dass nie mehr als
    # top_n Ticker gleichzeitig gehalten werden.
    held = pd.DataFrame(
        False, index=signals.entries.index, columns=signals.entries.columns, dtype=bool
    )
    state = dict.fromkeys(signals.entries.columns, False)
    for timestamp in signals.entries.index:
        for ticker in signals.entries.columns:
            if signals.entries.loc[timestamp, ticker]:
                state[ticker] = True
            elif signals.exits.loc[timestamp, ticker]:
                state[ticker] = False
        held.loc[timestamp] = pd.Series(state)

    assert int(held.sum(axis=1).max()) <= 2


def test_momentum_selects_the_stronger_ticker():
    # AAA steigt, BBB faellt. Nach dem Lookback muss AAA gewaehlt werden.
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC", name="date")
    up = 100.0 * np.exp(np.linspace(0.0, 0.5, n))
    down = 100.0 * np.exp(np.linspace(0.0, -0.5, n))

    def build(ticker: str, values: np.ndarray) -> PriceSeries:
        return PriceSeries(
            ticker=ticker,
            interval=Interval.D1,
            adjusted=True,
            data=pd.DataFrame(
                {
                    "open": values,
                    "high": values,
                    "low": values,
                    "close": values,
                    "volume": np.full(n, 1.0),
                },
                index=idx,
            ),
            provenance=_provenance(),
        )

    strategy = MomentumRotationStrategy(
        StrategyConfig(
            key="momentum_rotation", params={"lookback": 60, "top_n": 1, "rebalance_every": 21}
        )
    )
    signals = strategy.generate_signals({"AAA": build("AAA", up), "BBB": build("BBB", down)})

    assert int(signals.entries["AAA"].sum()) == 1
    assert int(signals.entries["BBB"].sum()) == 0


def test_momentum_rebalances_only_on_schedule(multi_price):
    strategy = MomentumRotationStrategy(
        StrategyConfig(
            key="momentum_rotation", params={"lookback": 60, "top_n": 2, "rebalance_every": 21}
        )
    )
    signals = strategy.generate_signals(multi_price)
    changes = signals.entries.to_numpy().any(axis=1) | signals.exits.to_numpy().any(axis=1)
    positions = np.flatnonzero(changes)
    # Alle Umschichtungen liegen auf dem Raster ab dem ersten gueltigen Bar.
    assert all((p - 60) % 21 == 0 for p in positions), positions.tolist()


def test_momentum_raises_on_too_short_history(multi_price):
    strategy = MomentumRotationStrategy()  # lookback = 126
    with pytest.raises(InsufficientDataError):
        strategy.generate_signals(_truncate(multi_price, keep=50))


# --------------------------------------------------------------------------
# Buy and Hold
# --------------------------------------------------------------------------


def test_buy_and_hold_has_no_parameters():
    assert BuyAndHoldStrategy().default_params == {}


def test_buy_and_hold_rejects_any_parameter():
    with pytest.raises(StrategyConfigError, match="Unknown parameters for strategy"):
        BuyAndHoldStrategy(StrategyConfig(key="buy_and_hold", params={"anything": 1}))


def test_buy_and_hold_enters_once_and_never_exits(multi_price):
    signals = BuyAndHoldStrategy().generate_signals(multi_price)
    assert (signals.entries.sum() == 1).all()
    assert not signals.exits.to_numpy().any()
    # Der Einstieg liegt auf dem ersten Bar.
    assert signals.entries.iloc[0].all()


# --------------------------------------------------------------------------
# Gemeinsamer Vertrag aller Strategien
# --------------------------------------------------------------------------


@pytest.mark.parametrize("strategy_cls", BUILTIN_STRATEGIES, ids=lambda c: c.key)
def test_every_strategy_returns_a_valid_signal_set(strategy_cls, multi_price):
    strategy = strategy_cls()
    if strategy_cls is MomentumRotationStrategy:
        strategy = strategy_cls(
            StrategyConfig(key=strategy_cls.key, params={"lookback": 60, "top_n": 2})
        )
    if strategy_cls is SmaCrossStrategy:
        strategy = strategy_cls(
            StrategyConfig(key=strategy_cls.key, params={"fast": 20, "slow": 60})
        )

    signals = strategy.generate_signals(multi_price)

    assert isinstance(signals, SignalSet)
    assert signals.entries.dtypes.eq(bool).all()
    assert signals.exits.dtypes.eq(bool).all()
    assert not signals.entries.isna().to_numpy().any()
    assert not signals.exits.isna().to_numpy().any()
    assert signals.execution_lag == 1
    assert list(signals.entries.columns) == sorted(multi_price)
    assert signals.entries.index.tz is not None


@pytest.mark.parametrize("strategy_cls", BUILTIN_STRATEGIES, ids=lambda c: c.key)
def test_every_strategy_calls_super_validate_params(strategy_cls):
    """Jede Unterklasse muss unbekannte Parameter ablehnen.

    Das gelingt nur, wenn ``validate_params`` die Basisimplementierung
    aufruft. Ein vergessenes ``super()`` faellt hier auf.
    """
    with pytest.raises(StrategyConfigError, match="Unknown parameters for strategy"):
        strategy_cls(StrategyConfig(key=strategy_cls.key, params={"nicht_existent": 1}))


@pytest.mark.parametrize("strategy_cls", BUILTIN_STRATEGIES, ids=lambda c: c.key)
def test_every_strategy_has_key_and_label(strategy_cls):
    assert strategy_cls.key
    assert strategy_cls.label
    assert isinstance(strategy_cls.default_params, dict)


def test_empty_price_dict_is_rejected():
    with pytest.raises(StrategyConfigError):
        BuyAndHoldStrategy().generate_signals({})


def test_all_strategy_keys_are_unique():
    keys = [cls.key for cls in BUILTIN_STRATEGIES]
    assert len(keys) == len(set(keys))
    assert set(keys) == set(default_registry().keys())
