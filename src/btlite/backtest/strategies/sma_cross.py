"""SMA-Crossover: der klassische Trendfolge-Referenzfall.

Einstieg, wenn der schnelle gleitende Durchschnitt den langsamen von unten
schneidet, Ausstieg beim Gegenschnitt.

Look-ahead-Freiheit: ``rolling(n).mean()`` an Position t mittelt die Bars
``t-n+1 .. t``, also ausschliesslich Vergangenheit und Gegenwart. Der
Schnitt vergleicht diesen Wert mit dem des **vorherigen** Bars ueber
``shift(1)``. Beides blickt nie nach vorn. Die Ausfuehrung verzoegert
``SignalSet.execution_lag`` zusaetzlich um einen Bar.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from btlite.backtest.base import Strategy
from btlite.backtest.strategies._panel import as_bool_frame, close_panel, require_history
from btlite.core.errors import StrategyConfigError
from btlite.models.backtest import SignalSet

if TYPE_CHECKING:
    from btlite.models.market import PriceSeries


class SmaCrossStrategy(Strategy):
    """Trendfolge ueber zwei gleitende Durchschnitte.

    Attributes:
        key: ``"sma_cross"``.
        label: Anzeigename fuer das UI.
        default_params: ``fast`` (Fenster des schnellen SMA in Bars),
            ``slow`` (Fenster des langsamen SMA in Bars).
    """

    key = "sma_cross"
    label = "SMA Crossover"
    default_params: dict[str, Any] = {  # noqa: RUF012 - Form wie in der Basisklasse
        "fast": 50,
        "slow": 200,
    }

    def validate_params(self, params: dict[str, Any]) -> None:
        """Prueft die Fensterlaengen.

        Args:
            params: Zusammengefuehrte Parameter.

        Raises:
            StrategyConfigError: Bei unbekannten Parametern, nicht-positiven
                Fenstern oder wenn das schnelle Fenster nicht kuerzer als
                das langsame ist.
        """
        super().validate_params(params)
        fast = self.require_positive_int(params, "fast")
        slow = self.require_positive_int(params, "slow")
        if fast >= slow:
            msg = "The fast SMA window must be shorter than the slow one"
            raise StrategyConfigError(msg, detail=f"fast={fast}, slow={slow}")

    def generate_signals(self, prices: dict[str, PriceSeries]) -> SignalSet:
        """Erzeugt Crossover-Signale je Ticker.

        Args:
            prices: Kursreihen je Ticker mit gemeinsamem UTC-Index.

        Returns:
            Signalmatrix mit einer Spalte je Ticker, ``execution_lag=1``.

        Raises:
            InsufficientDataError: Wenn die Historie kuerzer als das
                langsame Fenster ist.
        """
        fast_window = int(self.params["fast"])
        slow_window = int(self.params["slow"])

        panel = close_panel(prices)
        require_history(panel, required=slow_window + 1, what=f"SMA({slow_window})")

        fast = panel.rolling(fast_window, min_periods=fast_window).mean()
        slow = panel.rolling(slow_window, min_periods=slow_window).mean()

        above = fast > slow
        previous = above.shift(1)

        entries = above & ~previous.astype("boolean").fillna(False)
        exits = ~above & previous.astype("boolean").fillna(False)

        # Bars ohne vollstaendiges Fenster liefern kein Signal.
        valid = fast.notna() & slow.notna() & fast.shift(1).notna() & slow.shift(1).notna()

        return SignalSet(
            tickers=list(panel.columns),
            entries=as_bool_frame(entries & valid, like=panel),
            exits=as_bool_frame(exits & valid, like=panel),
            execution_lag=1,
        )
