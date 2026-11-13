"""Vertraege fuer Portfolio-Optimierung, Risiko und Attribution.

Block 5 des Produkts. Konventionen:
    - Gewichte sind dezimal und summieren sich auf 1.0 (Toleranz 1e-6).
    - Negative Gewichte sind nur zulaessig, wenn ``allow_shorting`` gesetzt war.
    - Alle Renditen und Risikomasse sind annualisiert und dezimal.
    - VaR und CVaR sind als **negative** Zahlen gespeichert (Verlust).
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from btlite.models.common import (
    BTModel,
    Currency,
    DataFrameField,
    Provenance,
    Ticker,
)

#: Zulaessige Abweichung der Gewichtssumme von 1.0.
WEIGHT_SUM_TOLERANCE = 1e-6


class OptimizationObjective(StrEnum):
    """Zielfunktion der Optimierung."""

    MAX_SHARPE = "max_sharpe"
    MIN_VOLATILITY = "min_volatility"
    EFFICIENT_RETURN = "efficient_return"
    EFFICIENT_RISK = "efficient_risk"
    MAX_QUADRATIC_UTILITY = "max_quadratic_utility"
    HRP = "hrp"
    EQUAL_WEIGHT = "equal_weight"
    BLACK_LITTERMAN = "black_litterman"


class CovarianceMethod(StrEnum):
    """Schaetzer der Kovarianzmatrix."""

    SAMPLE = "sample"
    LEDOIT_WOLF = "ledoit_wolf"
    EXP_WEIGHTED = "exp_weighted"
    SEMICOVARIANCE = "semicovariance"


class ReturnsMethod(StrEnum):
    """Schaetzer der erwarteten Renditen."""

    MEAN_HISTORICAL = "mean_historical"
    EMA_HISTORICAL = "ema_historical"
    CAPM = "capm"


class Position(BTModel):
    """Eine Position im Portfolio."""

    ticker: Ticker = Field(description="Kanonisches Symbol.")
    weight: float = Field(description="Zielgewicht, dezimal. Negativ nur bei Shorting.")
    shares: float | None = Field(
        default=None, description="Stueckzahl aus der Discrete Allocation."
    )
    price: float | None = Field(default=None, gt=0, description="Preis der Allokation.")
    value: float | None = Field(default=None, description="Positionswert in ``currency``.")


class PortfolioRequest(BTModel):
    """Anfrage an die Optimierung."""

    tickers: list[Ticker] = Field(min_length=2, description="Anlageuniversum, mindestens zwei.")
    objective: OptimizationObjective = Field(
        default=OptimizationObjective.MAX_SHARPE, description="Zielfunktion."
    )
    returns_method: ReturnsMethod = Field(
        default=ReturnsMethod.MEAN_HISTORICAL, description="Schaetzer der erwarteten Renditen."
    )
    covariance_method: CovarianceMethod = Field(
        default=CovarianceMethod.LEDOIT_WOLF, description="Schaetzer der Kovarianz."
    )
    lookback_period: str = Field(
        default="3y", description="Historie fuer die Schaetzung, yfinance-Notation."
    )
    risk_free_rate: float = Field(
        default=0.02, description="Jaehrlicher risikofreier Zins, dezimal."
    )
    allow_shorting: bool = Field(default=False, description="Erlaubt negative Gewichte.")
    min_weight: float = Field(default=0.0, description="Untergrenze je Position, dezimal.")
    max_weight: float = Field(
        default=1.0, gt=0, le=1, description="Obergrenze je Position, dezimal."
    )
    target_return: float | None = Field(
        default=None, description="Zielrendite, nur bei ``EFFICIENT_RETURN``."
    )
    target_volatility: float | None = Field(
        default=None, gt=0, description="Zielvolatilitaet, nur bei ``EFFICIENT_RISK``."
    )
    total_value: float = Field(
        default=100_000.0, gt=0, description="Anlagebetrag fuer die Discrete Allocation."
    )
    benchmark: Ticker | None = Field(default="SPY", description="Benchmark fuer Beta und Alpha.")

    @model_validator(mode="after")
    def _validate_targets(self) -> Self:
        """Prueft, dass Zielvorgaben zur Zielfunktion passen.

        Returns:
            Das validierte Modell.

        Raises:
            ValueError: Wenn eine noetige Zielvorgabe fehlt.
        """
        if self.objective is OptimizationObjective.EFFICIENT_RETURN and self.target_return is None:
            msg = "EFFICIENT_RETURN benoetigt target_return"
            raise ValueError(msg)
        if (
            self.objective is OptimizationObjective.EFFICIENT_RISK
            and self.target_volatility is None
        ):
            msg = "EFFICIENT_RISK benoetigt target_volatility"
            raise ValueError(msg)
        if not self.allow_shorting and self.min_weight < 0:
            msg = "min_weight < 0 erfordert allow_shorting=True"
            raise ValueError(msg)
        return self


class RiskMetrics(BTModel):
    """Risikokennzahlen eines Portfolios oder Einzelwerts.

    Alle Werte annualisiert und dezimal, sofern nicht anders vermerkt.
    VaR und CVaR sind negativ.
    """

    volatility: float | None = Field(default=None, ge=0, description="Annualisierte Volatilitaet.")
    downside_deviation: float | None = Field(
        default=None, ge=0, description="Annualisierte Abweichung unterhalb der Zielrendite."
    )
    beta: float | None = Field(default=None, description="Beta gegen die Benchmark.")
    alpha: float | None = Field(default=None, description="Jensens Alpha, annualisiert.")
    r_squared: float | None = Field(
        default=None, ge=0, le=1, description="Bestimmtheitsmass gegen die Benchmark."
    )
    var_historical: float | None = Field(
        default=None, le=0, description="Historischer VaR, dezimal und negativ."
    )
    var_parametric: float | None = Field(
        default=None, le=0, description="Parametrischer VaR (Normalverteilung), negativ."
    )
    cvar_historical: float | None = Field(
        default=None, le=0, description="Historischer Conditional VaR, negativ."
    )
    confidence: float = Field(
        default=0.95, gt=0, lt=1, description="Konfidenzniveau der VaR-Masse, dezimal."
    )
    max_drawdown: float | None = Field(
        default=None, le=0, description="Groesster Rueckgang, negativ."
    )
    max_drawdown_start: date | None = Field(default=None, description="Beginn des Max Drawdown.")
    max_drawdown_end: date | None = Field(default=None, description="Tiefpunkt des Max Drawdown.")
    sharpe: float | None = Field(default=None, description="Sharpe Ratio, annualisiert.")
    sortino: float | None = Field(default=None, description="Sortino Ratio, annualisiert.")
    tracking_error: float | None = Field(
        default=None, ge=0, description="Annualisierte Standardabweichung der Aktivrendite."
    )
    information_ratio: float | None = Field(
        default=None, description="Aktivrendite je Tracking Error."
    )
    skewness: float | None = Field(default=None, description="Schiefe der Renditeverteilung.")
    kurtosis: float | None = Field(
        default=None, description="Exzess-Kurtosis der Renditeverteilung."
    )


class EfficientFrontier(BTModel):
    """Stuetzstellen der Effizienzgrenze.

    Der Payload hat die Spalten ``volatility``, ``expected_return`` und
    ``sharpe``, alle annualisiert und dezimal. Optional enthaelt er je
    Ticker eine Gewichtsspalte mit Praefix ``w_``.
    """

    data: DataFrameField = Field(description="Stuetzstellen der Grenze.")
    tickers: list[Ticker] = Field(description="Zugrunde liegendes Universum.")
    optimal_point: Position | None = Field(
        default=None, description="Reserviert fuer die Markierung des Optimums."
    )

    @model_validator(mode="after")
    def _validate_columns(self) -> Self:
        """Prueft die Pflichtspalten.

        Returns:
            Das validierte Modell.
        """
        from btlite.models.common import require_columns

        require_columns(
            self.data,
            ("volatility", "expected_return", "sharpe"),
            what="EfficientFrontier",
        )
        return self


class PortfolioResult(BTModel):
    """Ergebnis einer Portfolio-Optimierung."""

    request: PortfolioRequest = Field(description="Die ausgefuehrte Anfrage.")
    positions: list[Position] = Field(description="Optimierte Positionen.")
    expected_return: float | None = Field(
        default=None, description="Erwartete Rendite, annualisiert dezimal."
    )
    expected_volatility: float | None = Field(
        default=None, ge=0, description="Erwartete Volatilitaet, annualisiert dezimal."
    )
    sharpe_ratio: float | None = Field(default=None, description="Erwartete Sharpe Ratio.")
    risk: RiskMetrics | None = Field(
        default=None, description="Realisierte Risikokennzahlen der Backtest-Historie."
    )
    frontier: EfficientFrontier | None = Field(default=None, description="Effizienzgrenze.")
    correlation: DataFrameField | None = Field(
        default=None, description="Korrelationsmatrix der Renditen, quadratisch."
    )
    leftover_cash: float | None = Field(
        default=None, ge=0, description="Nicht allokierter Rest der Discrete Allocation."
    )
    currency: Currency = Field(
        default=Currency.EUR, description="Waehrung der Betraege. Default Euro."
    )
    computed_at: datetime = Field(description="Zeitpunkt der Berechnung, tz-aware UTC.")
    provenance: Provenance = Field(description="Herkunft der Kursbasis.")
    warnings: list[str] = Field(default_factory=list, description="Nicht-fatale Hinweise.")

    @model_validator(mode="after")
    def _validate_weights(self) -> Self:
        """Prueft Gewichtssumme und Shorting-Regel.

        Returns:
            Das validierte Modell.

        Raises:
            ValueError: Wenn die Summe nicht 1.0 ist oder unerlaubt geshortet wird.
        """
        if not self.positions:
            return self
        total = sum(p.weight for p in self.positions)
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            msg = f"Gewichte summieren sich auf {total!r}, erwartet 1.0"
            raise ValueError(msg)
        if not self.request.allow_shorting:
            negative = [p.ticker for p in self.positions if p.weight < -WEIGHT_SUM_TOLERANCE]
            if negative:
                msg = f"Negative Gewichte ohne allow_shorting: {negative}"
                raise ValueError(msg)
        return self

    @property
    def weights(self) -> dict[str, float]:
        """Gewichte als Abbildung Ticker auf Gewicht.

        Returns:
            Dictionary der Zielgewichte.
        """
        return {p.ticker: p.weight for p in self.positions}


class AttributionResult(BTModel):
    """Performance-Attribution eines Portfolios.

    Der Payload hat den Index Ticker und die Spalten ``weight``,
    ``return``, ``contribution`` (Gewicht mal Rendite) sowie optional
    ``active_weight`` und ``allocation_effect`` beim Vergleich gegen eine
    Benchmark. Alle Werte dezimal.
    """

    data: DataFrameField = Field(description="Attributionstabelle, Index = Ticker.")
    total_return: float | None = Field(default=None, description="Portfoliorendite, dezimal.")
    benchmark_return: float | None = Field(default=None, description="Benchmarkrendite, dezimal.")
    active_return: float | None = Field(default=None, description="Differenz beider, dezimal.")
    period_start: date | None = Field(default=None, description="Beginn des Auswertungszeitraums.")
    period_end: date | None = Field(default=None, description="Ende des Auswertungszeitraums.")
    provenance: Provenance = Field(description="Herkunft der Kursbasis.")

    @model_validator(mode="after")
    def _validate_columns(self) -> Self:
        """Prueft die Pflichtspalten.

        Returns:
            Das validierte Modell.
        """
        from btlite.models.common import require_columns

        require_columns(self.data, ("weight", "return", "contribution"), what="AttributionResult")
        return self
