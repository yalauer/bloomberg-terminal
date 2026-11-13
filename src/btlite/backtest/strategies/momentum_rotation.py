"""Momentum-Rotation: periodisches Umschichten in die Top-N Ticker.

An jedem Rebalancing-Termin wird das Momentum aller Ticker ueber ein
Lookback-Fenster gemessen. Die besten ``top_n`` werden gehalten, alle
uebrigen verkauft. Zwischen den Terminen bleibt die Allokation unveraendert.

Look-ahead-Freiheit: das Momentum an Position t ist ``close[t] / close[t-L]
- 1`` und nutzt damit ausschliesslich Bars bis t. Das Ranking erfolgt
zeilenweise (``rank(axis=1)``), also ohne Blick auf spaetere Zeilen. Die
Rebalancing-Termine werden allein aus dem Positionsindex abgeleitet
(``t % every == 0`` relativ zum ersten gueltigen Bar) und nicht aus
kuenftigen Kursen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from btlite.backtest.base import Strategy
from btlite.backtest.strategies._panel import as_bool_frame, close_panel, require_history
from btlite.models.backtest import SignalSet

if TYPE_CHECKING:
    from btlite.models.market import PriceSeries


class MomentumRotationStrategy(Strategy):
    """Haelt die ``top_n`` Ticker mit dem hoechsten Momentum.

    Attributes:
        key: ``"momentum_rotation"``.
        label: Anzeigename fuer das UI.
        default_params: ``lookback`` (Momentum-Fenster in Bars),
            ``top_n`` (Anzahl gehaltener Ticker), ``rebalance_every``
            (Abstand der Rebalancing-Termine in Bars).
    """

    key = "momentum_rotation"
    label = "Momentum Rotation"
    default_params: dict[str, Any] = {  # noqa: RUF012 - Form wie in der Basisklasse
        "lookback": 126,
        "top_n": 3,
        "rebalance_every": 21,
    }

    def validate_params(self, params: dict[str, Any]) -> None:
        """Prueft Lookback, Top-N und Rebalancing-Abstand.

        Args:
            params: Zusammengefuehrte Parameter.

        Raises:
            StrategyConfigError: Bei unbekannten oder nicht-positiven
                ganzzahligen Parametern.
        """
        super().validate_params(params)
        self.require_positive_int(params, "lookback")
        self.require_positive_int(params, "top_n")
        self.require_positive_int(params, "rebalance_every")

    def generate_signals(self, prices: dict[str, PriceSeries]) -> SignalSet:
        """Erzeugt Rotations-Signale.

        Ein Einstieg entsteht, wenn ein Ticker an einem Rebalancing-Termin
        neu in die Top-N kommt, ein Ausstieg, wenn er sie verlaesst.

        Args:
            prices: Kursreihen je Ticker mit gemeinsamem UTC-Index.

        Returns:
            Signalmatrix mit einer Spalte je Ticker, ``execution_lag=1``.

        Raises:
            InsufficientDataError: Wenn die Historie kuerzer als das
                Lookback-Fenster ist.
        """
        lookback = int(self.params["lookback"])
        top_n = int(self.params["top_n"])
        every = int(self.params["rebalance_every"])

        panel = close_panel(prices)
        require_history(panel, required=lookback + 1, what=f"Momentum({lookback})")

        momentum = panel.pct_change(periods=lookback)

        # Ranking je Zeile: 1 = staerkstes Momentum. Nur Ticker mit
        # vollstaendigem Lookback nehmen teil.
        ranks = momentum.rank(axis=1, ascending=False, method="first")
        selected = (ranks <= top_n) & momentum.notna()

        target = self._hold_between_rebalances(selected, every=every, first_valid=lookback)

        held_before = target.shift(1).astype("boolean").fillna(False)
        entries = target & ~held_before
        exits = ~target & held_before

        return SignalSet(
            tickers=list(panel.columns),
            entries=as_bool_frame(entries, like=panel),
            exits=as_bool_frame(exits, like=panel),
            execution_lag=1,
        )

    def _hold_between_rebalances(
        self, selected: pd.DataFrame, *, every: int, first_valid: int
    ) -> pd.DataFrame:
        """Haelt die Auswahl eines Termins bis zum naechsten Termin konstant.

        Args:
            selected: Boolesche Auswahl je Bar, wie sie das Ranking liefert.
            every: Abstand der Rebalancing-Termine in Bars.
            first_valid: Position des ersten Bar mit vollstaendigem
                Lookback-Fenster.

        Returns:
            Ziel-Allokation je Bar, boolesch. Vor dem ersten Termin
            durchgehend False.
        """
        positions = np.arange(len(selected))
        is_rebalance = (positions >= first_valid) & ((positions - first_valid) % every == 0)

        # Auswahl nur an Terminen uebernehmen, dazwischen nach vorne tragen.
        # Der Umweg ueber float ist noetig, damit ``ffill`` eine Luecke
        # ueberhaupt erkennt: ein bool-DataFrame kennt kein NaN.
        numeric = selected.astype(float).where(
            pd.Series(is_rebalance, index=selected.index), other=float("nan")
        )
        return numeric.ffill().fillna(0.0).astype(bool)
