"""Vertraege fuer Backtesting.

Block 4 des Produkts. Der Vertrag ist engine-agnostisch: vectorbt und
backtrader fuellen dasselbe :class:`BacktestResult`. Kein Modul ausserhalb
von ``backtest/`` sieht Engine-spezifische Objekte.

Kostenkonvention: ``fees`` und ``slippage`` sind dezimale Anteile des
Ordervolumens je Trade (0.0005 = 5 Basispunkte), und werden bei Ein- und
Ausstieg jeweils angewandt.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import Field, model_validator

from btlite.models.common import (
    BTModel,
    Currency,
    DataFrameField,
    Interval,
    Provenance,
    Ticker,
)


class Direction(StrEnum):
    """Erlaubte Positionsrichtungen einer Strategie."""

    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"
    LONG_SHORT = "long_short"


class TradeSide(StrEnum):
    """Richtung eines einzelnen Trade."""

    LONG = "long"
    SHORT = "short"


class SignalSet(BTModel):
    """Ein- und Ausstiegssignale einer Strategie.

    ``entries`` und ``exits`` sind boolesche DataFrames mit identischem
    UTC-``DatetimeIndex`` und einer Spalte je Ticker. Ein True in Zeile t
    bedeutet: Signal am Bar t, Ausfuehrung gemaess ``execution_lag`` Bars
    spaeter, um Look-ahead-Bias zu vermeiden.
    """

    tickers: list[Ticker] = Field(min_length=1, description="Bespielte Symbole.")
    entries: DataFrameField = Field(description="Boolesche Einstiegssignale.")
    exits: DataFrameField = Field(description="Boolesche Ausstiegssignale.")
    execution_lag: int = Field(
        default=1,
        ge=0,
        description="Bars zwischen Signal und Ausfuehrung. 1 = naechster Bar, kein Look-ahead.",
    )
    size: DataFrameField | None = Field(
        default=None, description="Optionale Ordergroessen je Signal, sonst gleichgewichtet."
    )

    @model_validator(mode="after")
    def _validate_alignment(self) -> Self:
        """Prueft, dass Ein- und Ausstiegsmatrix deckungsgleich sind.

        Returns:
            Das validierte Modell.

        Raises:
            ValueError: Bei abweichendem Index oder Spaltensatz.
        """
        if not self.entries.index.equals(self.exits.index):
            msg = "entries und exits muessen denselben Index haben"
            raise ValueError(msg)
        if list(self.entries.columns) != list(self.exits.columns):
            msg = "entries und exits muessen dieselben Spalten haben"
            raise ValueError(msg)
        return self


class StrategyConfig(BTModel):
    """Konfiguration einer Strategie.

    ``params`` ist bewusst offen, damit neue Strategien keine Vertrags-
    aenderung erzwingen. Jede Strategie validiert ihre eigenen Parameter in
    ``backtest/strategies/`` und wirft ``StrategyConfigError``.
    """

    key: str = Field(description="Bezeichner der Strategie, etwa sma_cross.")
    label: str | None = Field(default=None, description="Anzeigename.")
    params: dict[str, Any] = Field(default_factory=dict, description="Strategie-Parameter.")
    direction: Direction = Field(default=Direction.LONG_ONLY, description="Positionsrichtung.")


class BacktestRequest(BTModel):
    """Anfrage an eine Backtest-Engine."""

    tickers: list[Ticker] = Field(min_length=1, description="Zu handelnde Symbole.")
    strategy: StrategyConfig = Field(description="Strategie und Parameter.")
    interval: Interval = Field(default=Interval.D1, description="Bar-Intervall.")
    start: datetime | None = Field(default=None, description="Startzeitpunkt, tz-aware UTC.")
    end: datetime | None = Field(default=None, description="Endzeitpunkt, tz-aware UTC.")
    init_cash: float = Field(default=100_000.0, gt=0, description="Startkapital.")
    fees: float = Field(default=0.0005, ge=0, le=1, description="Kommission je Trade, dezimal.")
    slippage: float = Field(default=0.0005, ge=0, le=1, description="Slippage je Trade, dezimal.")
    benchmark: Ticker | None = Field(
        default="SPY", description="Vergleichsindex fuer Alpha und Tracking Error."
    )
    engine: str = Field(default="vectorbt", description="Gewuenschte Engine.")


class Trade(BTModel):
    """Ein abgeschlossener oder offener Trade."""

    ticker: Ticker = Field(description="Gehandeltes Symbol.")
    side: TradeSide = Field(description="Richtung der Position.")
    entry_time: datetime = Field(description="Einstiegszeitpunkt, tz-aware UTC.")
    entry_price: float = Field(gt=0, description="Einstiegskurs nach Slippage.")
    exit_time: datetime | None = Field(
        default=None, description="Ausstiegszeitpunkt. None = Position noch offen."
    )
    exit_price: float | None = Field(default=None, gt=0, description="Ausstiegskurs nach Slippage.")
    size: float = Field(gt=0, description="Positionsgroesse in Stueck.")
    pnl: float | None = Field(default=None, description="Realisierter Gewinn nach Kosten.")
    return_pct: float | None = Field(default=None, description="Rendite des Trade, dezimal.")
    fees_paid: float = Field(default=0.0, ge=0, description="Kommission des Trade, absolut.")
    duration_bars: int | None = Field(default=None, ge=0, description="Haltedauer in Bars.")

    @property
    def is_open(self) -> bool:
        """Ob die Position noch offen ist.

        Returns:
            True wenn kein Ausstieg vorliegt.
        """
        return self.exit_time is None

    @property
    def is_winner(self) -> bool | None:
        """Ob der Trade im Gewinn geschlossen wurde.

        Returns:
            True bei positivem PnL, None bei offener Position.
        """
        return None if self.pnl is None else self.pnl > 0


class PerformanceMetrics(BTModel):
    """Kennzahlen eines Backtest oder einer beliebigen Equity-Kurve.

    Alle Renditen und Volatilitaeten sind **annualisiert und dezimal**
    (0.12 = 12 Prozent pro Jahr). Drawdowns sind negativ.
    """

    total_return: float | None = Field(default=None, description="Gesamtrendite, dezimal.")
    cagr: float | None = Field(default=None, description="Jaehrliche Wachstumsrate, dezimal.")
    volatility: float | None = Field(
        default=None, ge=0, description="Annualisierte Volatilitaet, dezimal."
    )
    sharpe: float | None = Field(default=None, description="Sharpe Ratio, annualisiert.")
    sortino: float | None = Field(default=None, description="Sortino Ratio, annualisiert.")
    calmar: float | None = Field(default=None, description="CAGR geteilt durch Max Drawdown.")
    max_drawdown: float | None = Field(
        default=None, le=0, description="Groesster Rueckgang, dezimal und negativ."
    )
    max_drawdown_duration_days: int | None = Field(
        default=None, ge=0, description="Laengste Underwater-Phase in Tagen."
    )
    var_95: float | None = Field(
        default=None, description="Historischer Value at Risk, dezimal und negativ."
    )
    cvar_95: float | None = Field(default=None, description="Conditional VaR, dezimal und negativ.")
    beta: float | None = Field(default=None, description="Beta gegen die Benchmark.")
    alpha: float | None = Field(default=None, description="Jensens Alpha, annualisiert dezimal.")
    tracking_error: float | None = Field(
        default=None, ge=0, description="Annualisierte Standardabweichung der Aktivrendite."
    )
    information_ratio: float | None = Field(
        default=None, description="Aktivrendite je Tracking Error."
    )

    # --- Trade-Statistiken ---
    n_trades: int | None = Field(default=None, ge=0, description="Anzahl geschlossener Trades.")
    hit_rate: float | None = Field(
        default=None, ge=0, le=1, description="Anteil profitabler Trades, dezimal."
    )
    profit_factor: float | None = Field(
        default=None, ge=0, description="Summe Gewinne geteilt durch Summe Verluste."
    )
    avg_win: float | None = Field(default=None, description="Mittlerer Gewinn je Gewinner-Trade.")
    avg_loss: float | None = Field(
        default=None, description="Mittlerer Verlust je Verlierer-Trade."
    )
    avg_duration_bars: float | None = Field(default=None, ge=0, description="Mittlere Haltedauer.")
    exposure: float | None = Field(
        default=None, ge=0, le=1, description="Anteil der Zeit im Markt, dezimal."
    )
    total_fees: float | None = Field(default=None, ge=0, description="Summe aller Kosten, absolut.")


class BacktestResult(BTModel):
    """Vollstaendiges Ergebnis eines Backtest.

    ``equity_curve`` ist ein DataFrame mit UTC-Index und mindestens der
    Spalte ``equity`` (Portfoliowert in Kontowaehrung). Optional enthaelt er
    ``cash``, ``drawdown`` und ``benchmark_equity``.
    """

    request: BacktestRequest = Field(description="Die ausgefuehrte Anfrage.")
    equity_curve: DataFrameField = Field(description="Portfoliowert ueber die Zeit.")
    trades: list[Trade] = Field(default_factory=list, description="Trade-Liste.")
    metrics: PerformanceMetrics = Field(description="Kennzahlen der Strategie.")
    benchmark_metrics: PerformanceMetrics | None = Field(
        default=None, description="Dieselben Kennzahlen fuer die Benchmark."
    )
    signals: SignalSet | None = Field(default=None, description="Die erzeugten Signale.")
    currency: Currency = Field(
        default=Currency.EUR, description="Kontowaehrung des Backtests. Default Euro."
    )
    engine_used: str = Field(description="Tatsaechlich verwendete Engine.")
    provenance: Provenance = Field(description="Herkunft der Kursbasis.")
    warnings: list[str] = Field(
        default_factory=list, description="Nicht-fatale Hinweise, im UI sichtbar."
    )

    @model_validator(mode="after")
    def _validate_equity(self) -> Self:
        """Prueft die Equity-Kurve auf ihren Vertrag.

        Returns:
            Das validierte Modell.
        """
        from btlite.models.common import require_columns, require_utc_index

        what = "BacktestResult.equity_curve"
        require_columns(self.equity_curve, ("equity",), what=what)
        require_utc_index(self.equity_curve, what=what)
        return self
