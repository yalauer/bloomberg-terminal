"""Buy and Hold: die Benchmark-Strategie.

Einstieg am ersten Bar mit gueltigem Kurs, danach kein weiteres Signal. Die
Position bleibt bis zum Ende des Zeitraums offen; die Engine bewertet sie
zum letzten Kurs.

Diese Strategie ist der Bezugspunkt fuer jede andere: wer sie nach Kosten
nicht schlaegt, hat nichts gewonnen. Look-ahead-Bias ist konstruktiv
ausgeschlossen, da nur der erste gueltige Kurs eines Tickers ausgewertet
wird.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from btlite.backtest.base import Strategy
from btlite.backtest.strategies._panel import close_panel
from btlite.models.backtest import SignalSet

if TYPE_CHECKING:
    from btlite.models.market import PriceSeries


class BuyAndHoldStrategy(Strategy):
    """Kauft einmal und haelt bis zum Ende des Zeitraums.

    Attributes:
        key: ``"buy_and_hold"``.
        label: Anzeigename fuer das UI.
        default_params: Leer - diese Strategie hat keine Parameter.
    """

    key = "buy_and_hold"
    label = "Buy and Hold"
    default_params: dict[str, Any] = {}  # noqa: RUF012 - Form wie in der Basisklasse

    def generate_signals(self, prices: dict[str, PriceSeries]) -> SignalSet:
        """Setzt je Ticker genau ein Einstiegssignal, keinen Ausstieg.

        Args:
            prices: Kursreihen je Ticker mit gemeinsamem UTC-Index.

        Returns:
            Signalmatrix mit genau einem True je Spalte in ``entries`` und
            einer durchgehend False belegten ``exits``-Matrix.
            ``execution_lag=1``, damit auch die Benchmark denselben
            Ausfuehrungszeitpunkt wie jede andere Strategie hat.

        Raises:
            InsufficientDataError: Wenn weniger als zwei Bars vorliegen.
        """
        panel = close_panel(prices)

        entries = pd.DataFrame(False, index=panel.index, columns=panel.columns, dtype=bool)
        for column in panel.columns:
            first = panel[column].first_valid_index()
            if first is not None:
                entries.loc[first, column] = True

        exits = pd.DataFrame(False, index=panel.index, columns=panel.columns, dtype=bool)

        return SignalSet(
            tickers=list(panel.columns),
            entries=entries,
            exits=exits,
            execution_lag=1,
        )
