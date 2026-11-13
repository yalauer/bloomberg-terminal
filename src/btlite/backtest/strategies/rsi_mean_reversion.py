"""RSI-Mean-Reversion: Einstieg im ueberverkauften Bereich.

Einstieg, wenn der RSI die untere Schwelle von oben nach unten durchbricht,
Ausstieg, wenn er die obere Schwelle von unten nach oben durchbricht.

Look-ahead-Freiheit: der RSI nutzt Wilders Glaettung als exponentiell
gewichteten Mittelwert der Kursdifferenzen. Sowohl ``diff()`` als auch
``ewm(...).mean()`` sind streng kausal - der Wert an Position t haengt nur
von den Bars bis t ab. Der Schwellendurchbruch vergleicht mit ``shift(1)``,
also mit dem Vorgaenger-Bar.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from btlite.backtest.base import Strategy
from btlite.backtest.strategies._panel import (
    as_bool_frame,
    close_panel,
    require_history,
    require_positive_number,
)
from btlite.core.errors import StrategyConfigError
from btlite.models.backtest import SignalSet

if TYPE_CHECKING:
    from btlite.models.market import PriceSeries

#: Obergrenze der RSI-Skala.
RSI_MAX: float = 100.0


def rsi(close: pd.DataFrame, *, period: int) -> pd.DataFrame:
    """Relative Strength Index nach Wilder.

    Geglaettet wird mit einem exponentiell gewichteten Mittel
    (``alpha = 1/period``), wie es Wilders Original entspricht. Die
    Berechnung ist streng kausal.

    Args:
        close: Schlusskurs-Panel, eine Spalte je Ticker.
        period: Glaettungsperiode in Bars.

    Returns:
        RSI je Bar und Ticker, Skala 0 bis 100. Die ersten ``period`` Bars
        sind NaN, solange die Glaettung nicht eingeschwungen ist.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # Ohne Abwaertsbewegung ist RS unendlich; der RSI ist dann definitionsgemaess 100.
    rs = avg_gain / avg_loss.where(avg_loss > 0.0)
    out = RSI_MAX - RSI_MAX / (1.0 + rs)
    return out.where(avg_loss > 0.0, other=RSI_MAX).where(avg_gain.notna() & avg_loss.notna())


class RsiMeanReversionStrategy(Strategy):
    """Gegenbewegung nach Ueberverkauft-Signal.

    Attributes:
        key: ``"rsi_mean_reversion"``.
        label: Anzeigename fuer das UI.
        default_params: ``period`` (RSI-Glaettung in Bars), ``oversold``
            (Einstiegsschwelle, RSI-Punkte), ``overbought``
            (Ausstiegsschwelle, RSI-Punkte).
    """

    key = "rsi_mean_reversion"
    label = "RSI Mean Reversion"
    default_params: dict[str, Any] = {  # noqa: RUF012 - Form wie in der Basisklasse
        "period": 14,
        "oversold": 30.0,
        "overbought": 70.0,
    }

    def validate_params(self, params: dict[str, Any]) -> None:
        """Prueft Periode und Schwellen.

        Args:
            params: Zusammengefuehrte Parameter.

        Raises:
            StrategyConfigError: Bei unbekannten Parametern, nicht-positiver
                Periode, Schwellen ausserhalb von 0 bis 100 oder wenn die
                Einstiegsschwelle nicht unter der Ausstiegsschwelle liegt.
        """
        super().validate_params(params)
        self.require_positive_int(params, "period")
        oversold = require_positive_number(params, "oversold")
        overbought = require_positive_number(params, "overbought")

        if not oversold < overbought < RSI_MAX:
            msg = "The following must hold: 0 < oversold < overbought < 100"
            raise StrategyConfigError(msg, detail=f"oversold={oversold}, overbought={overbought}")

    def generate_signals(self, prices: dict[str, PriceSeries]) -> SignalSet:
        """Erzeugt Schwellendurchbruch-Signale je Ticker.

        Args:
            prices: Kursreihen je Ticker mit gemeinsamem UTC-Index.

        Returns:
            Signalmatrix mit einer Spalte je Ticker, ``execution_lag=1``.

        Raises:
            InsufficientDataError: Wenn die Historie kuerzer als die
                RSI-Periode ist.
        """
        period = int(self.params["period"])
        oversold = float(self.params["oversold"])
        overbought = float(self.params["overbought"])

        panel = close_panel(prices)
        require_history(panel, required=period + 2, what=f"RSI({period})")

        values = rsi(panel, period=period)
        previous = values.shift(1)

        # Durchbruch, nicht blosses Unterschreiten: verhindert Dauersignale.
        entries = (values < oversold) & (previous >= oversold)
        exits = (values > overbought) & (previous <= overbought)

        valid = values.notna() & previous.notna()

        return SignalSet(
            tickers=list(panel.columns),
            entries=as_bool_frame(entries & valid, like=panel),
            exits=as_bool_frame(exits & valid, like=panel),
            execution_lag=1,
        )
