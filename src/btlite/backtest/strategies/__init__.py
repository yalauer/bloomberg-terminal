"""Referenzstrategien.

Vier Strategien, die zusammen das Spektrum abdecken, gegen das sich jede
weitere messen laesst: Trendfolge (:class:`SmaCrossStrategy`), Gegenbewegung
(:class:`RsiMeanReversionStrategy`), Querschnitts-Momentum
(:class:`MomentumRotationStrategy`) und die passive Benchmark
(:class:`BuyAndHoldStrategy`).

Alle erfuellen denselben Vertrag: sie erben von
:class:`~btlite.backtest.base.Strategy`, erzeugen einen
:class:`~btlite.models.backtest.SignalSet` und sind frei von
Look-ahead-Bias.
"""

from __future__ import annotations

from btlite.backtest.strategies.buy_and_hold import BuyAndHoldStrategy
from btlite.backtest.strategies.momentum_rotation import MomentumRotationStrategy
from btlite.backtest.strategies.rsi_mean_reversion import RsiMeanReversionStrategy
from btlite.backtest.strategies.sma_cross import SmaCrossStrategy

__all__ = [
    "BuyAndHoldStrategy",
    "MomentumRotationStrategy",
    "RsiMeanReversionStrategy",
    "SmaCrossStrategy",
]
