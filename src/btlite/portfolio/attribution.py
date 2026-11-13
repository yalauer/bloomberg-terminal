"""Performance-Attribution eines Portfolios.

Zerlegt die Portfoliorendite in die Beitraege der Einzelpositionen und,
gegen eine Benchmark, in Allokations- und Selektionseffekte.

Architekturregel 2: Kurse werden **injiziert**. Dieses Modul holt nichts.

Einheiten: alle Gewichte, Renditen und Effekte dezimal (0.05 bedeutet
5 Prozent). Der Beitrag einer Position ist ``weight * return``.

Der Vertrag verlangt die Spalten ``weight``, ``return`` und
``contribution``, optional ``active_weight`` und ``allocation_effect``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from btlite.core.errors import InsufficientDataError
from btlite.core.logging import get_logger
from btlite.models.portfolio import WEIGHT_SUM_TOLERANCE, AttributionResult

if TYPE_CHECKING:
    from btlite.models.common import Provenance
    from btlite.models.market import PriceSeries

_LOG = get_logger(__name__)

#: Pflichtspalten der Attributionstabelle. Vertrag mit ``models/portfolio.py``.
WEIGHT_COLUMN: Final[str] = "weight"
RETURN_COLUMN: Final[str] = "return"
CONTRIBUTION_COLUMN: Final[str] = "contribution"

#: Optionale Spalten beim Vergleich gegen eine Benchmark.
ACTIVE_WEIGHT_COLUMN: Final[str] = "active_weight"
ALLOCATION_EFFECT_COLUMN: Final[str] = "allocation_effect"
SELECTION_EFFECT_COLUMN: Final[str] = "selection_effect"

#: Name des Tabellenindex.
TICKER_INDEX_NAME: Final[str] = "ticker"


def period_return(series: PriceSeries) -> float | None:
    """Gesamtrendite einer Kursreihe ueber ihren vollen Zeitraum.

    Args:
        series: Kursreihe mit mindestens zwei Beobachtungen.

    Returns:
        Rendite von erster bis letzter Beobachtung, dezimal. None wenn die
        Reihe zu kurz ist oder der Startkurs nicht positiv ist.
    """
    close = series.close.dropna()
    if len(close) < 2:
        return None
    start = float(close.iloc[0])
    if start <= 0.0:
        return None
    value = float(close.iloc[-1]) / start - 1.0
    return value if np.isfinite(value) else None


def _returns_from_prices(prices: dict[str, PriceSeries]) -> dict[str, float]:
    """Berechnet die Periodenrendite je Titel.

    Args:
        prices: Abbildung Ticker auf Kursreihe.

    Returns:
        Abbildung Ticker auf Rendite, dezimal. Titel ohne berechenbare
        Rendite fehlen im Ergebnis.
    """
    out: dict[str, float] = {}
    for ticker, series in prices.items():
        value = period_return(series)
        if value is None:
            _LOG.debug("Keine Periodenrendite fuer %s", ticker)
            continue
        out[ticker] = value
    return out


def _period_bounds(
    prices: dict[str, PriceSeries],
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Ermittelt den gemeinsamen Auswertungszeitraum.

    Args:
        prices: Abbildung Ticker auf Kursreihe.

    Returns:
        Tupel aus spaetestem Start und fruehestem Ende ueber alle Reihen -
        also dem Zeitraum, den wirklich alle abdecken. Beide None, wenn
        keine Reihe Daten hat.
    """
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for series in prices.values():
        index = series.data.index
        if len(index) == 0:
            continue
        starts.append(pd.Timestamp(index[0]))
        ends.append(pd.Timestamp(index[-1]))
    if not starts or not ends:
        return None, None
    return max(starts), min(ends)


def attribute(
    weights: dict[str, float],
    prices: dict[str, PriceSeries],
    *,
    benchmark_weights: dict[str, float] | None = None,
    benchmark_returns: dict[str, float] | None = None,
    provenance: Provenance,
) -> AttributionResult:
    """Zerlegt die Portfoliorendite in Positionsbeitraege.

    Der Beitrag einer Position ist ``weight * return``. Die Summe der
    Beitraege ist die Portfoliorendite - das ist die Kernidentitaet dieser
    Zerlegung und wird von den Tests geprueft.

    Mit ``benchmark_weights`` entstehen zusaetzlich das Aktivgewicht
    (``portfolio - benchmark``) und der Allokationseffekt nach
    Brinson-Hood-Beebower:
    ``allocation_effect = (w_p - w_b) * (r_b_i - r_b_total)``.
    Er misst den Beitrag der Ueber- oder Untergewichtung allein, unabhaengig
    davon, wie gut der Titel selbst gelaufen ist.

    Args:
        weights: Portfoliogewichte je Ticker, dezimal.
        prices: Kursreihen je Ticker. Bestimmen die Renditen des Zeitraums.
        benchmark_weights: Benchmarkgewichte je Ticker, dezimal. Ohne sie
            bleiben die Aktiv-Spalten weg.
        benchmark_returns: Renditen der Benchmark-Bestandteile, dezimal.
            ``None`` nimmt die Renditen aus ``prices`` - passend, wenn
            Portfolio und Benchmark dasselbe Universum unterschiedlich
            gewichten.
        provenance: Herkunft der Kursbasis.

    Returns:
        Attributionstabelle mit Index Ticker und den Pflichtspalten.

    Raises:
        InsufficientDataError: Wenn kein Titel eine berechenbare Rendite hat.
    """
    if not weights:
        msg = "Attribution is not possible without weights."
        raise InsufficientDataError(msg, detail="weights ist leer", required=1, available=0)

    returns = _returns_from_prices(prices)
    covered = {t: w for t, w in weights.items() if t in returns}
    if not covered:
        msg = "No return is available for any weighted title."
        raise InsufficientDataError(
            msg,
            detail=f"gewichtet={sorted(weights)}, mit Rendite={sorted(returns)}",
            required=1,
            available=0,
        )

    missing = sorted(set(weights) - set(covered))
    if missing:
        _LOG.warning("Ohne Kursrendite aus der Attribution genommen: %s", missing)

    index = pd.Index(sorted(covered), name=TICKER_INDEX_NAME)
    weight_series = pd.Series({t: float(covered[t]) for t in index}, index=index, dtype="float64")
    return_series = pd.Series({t: float(returns[t]) for t in index}, index=index, dtype="float64")

    table = pd.DataFrame(
        {
            WEIGHT_COLUMN: weight_series,
            RETURN_COLUMN: return_series,
            CONTRIBUTION_COLUMN: weight_series * return_series,
        }
    )

    total = float(table[CONTRIBUTION_COLUMN].sum())
    benchmark_total: float | None = None
    active: float | None = None

    if benchmark_weights:
        bench_returns = benchmark_returns if benchmark_returns is not None else returns
        bench_weight = pd.Series(
            {t: float(benchmark_weights.get(t, 0.0)) for t in index},
            index=index,
            dtype="float64",
        )
        bench_return = pd.Series(
            {t: float(bench_returns.get(t, float("nan"))) for t in index},
            index=index,
            dtype="float64",
        )

        # Die Benchmarkrendite wird ueber ALLE Benchmark-Bestandteile
        # gerechnet, auch die nicht im Portfolio gehaltenen. Sonst waere der
        # Vergleichsmassstab stillschweigend das Portfoliouniversum.
        benchmark_total = float(
            sum(
                w * bench_returns[t]
                for t, w in benchmark_weights.items()
                if t in bench_returns and np.isfinite(bench_returns[t])
            )
        )
        active = total - benchmark_total

        table[ACTIVE_WEIGHT_COLUMN] = weight_series - bench_weight
        table[ALLOCATION_EFFECT_COLUMN] = table[ACTIVE_WEIGHT_COLUMN] * (
            bench_return - benchmark_total
        )
        table[SELECTION_EFFECT_COLUMN] = bench_weight * (return_series - bench_return)

    start, end = _period_bounds({t: prices[t] for t in index if t in prices})

    return AttributionResult(
        data=table,
        total_return=total if np.isfinite(total) else None,
        benchmark_return=(
            benchmark_total
            if benchmark_total is not None and np.isfinite(benchmark_total)
            else None
        ),
        active_return=active if active is not None and np.isfinite(active) else None,
        period_start=start.date() if start is not None else None,
        period_end=end.date() if end is not None else None,
        provenance=provenance,
    )


def attribute_from_returns(
    weights: dict[str, float],
    returns: dict[str, float],
    *,
    provenance: Provenance,
    period_start: pd.Timestamp | None = None,
    period_end: pd.Timestamp | None = None,
) -> AttributionResult:
    """Attribution direkt aus fertigen Renditen, ohne Kursreihen.

    Fuer Faelle, in denen die Renditen schon anderswo berechnet wurden, etwa
    aus einem Backtest-Ergebnis.

    Args:
        weights: Portfoliogewichte je Ticker, dezimal.
        returns: Renditen je Ticker, dezimal.
        provenance: Herkunft der Datenbasis.
        period_start: Beginn des Auswertungszeitraums.
        period_end: Ende des Auswertungszeitraums.

    Returns:
        Attributionstabelle mit Index Ticker und den Pflichtspalten.

    Raises:
        InsufficientDataError: Wenn kein gewichteter Titel eine Rendite hat.
    """
    covered = {t: w for t, w in weights.items() if t in returns and np.isfinite(float(returns[t]))}
    if not covered:
        msg = "No return is available for any weighted title."
        raise InsufficientDataError(
            msg,
            detail=f"gewichtet={sorted(weights)}, mit Rendite={sorted(returns)}",
            required=1,
            available=0,
        )

    index = pd.Index(sorted(covered), name=TICKER_INDEX_NAME)
    weight_series = pd.Series({t: float(covered[t]) for t in index}, index=index, dtype="float64")
    return_series = pd.Series({t: float(returns[t]) for t in index}, index=index, dtype="float64")
    table = pd.DataFrame(
        {
            WEIGHT_COLUMN: weight_series,
            RETURN_COLUMN: return_series,
            CONTRIBUTION_COLUMN: weight_series * return_series,
        }
    )
    total = float(table[CONTRIBUTION_COLUMN].sum())

    return AttributionResult(
        data=table,
        total_return=total if np.isfinite(total) else None,
        period_start=period_start.date() if period_start is not None else None,
        period_end=period_end.date() if period_end is not None else None,
        provenance=provenance,
    )


def top_contributors(
    result: AttributionResult,
    *,
    count: int = 5,
    ascending: bool = False,
) -> pd.DataFrame:
    """Sortiert die Positionen nach Renditebeitrag.

    Args:
        result: Fertige Attribution.
        count: Anzahl der zurueckgegebenen Zeilen.
        ascending: ``True`` liefert die groessten Belastungen, ``False`` die
            groessten Beitraege.

    Returns:
        Ausschnitt der Attributionstabelle, nach ``contribution`` sortiert.
    """
    ordered = result.data.sort_values(CONTRIBUTION_COLUMN, ascending=ascending)
    return ordered.head(max(count, 0))


def weights_sum_to_one(weights: dict[str, float]) -> bool:
    """Prueft die Gewichtssumme gegen die Vertragstoleranz.

    Args:
        weights: Gewichte je Ticker, dezimal.

    Returns:
        True wenn die Summe innerhalb
        :data:`~btlite.models.portfolio.WEIGHT_SUM_TOLERANCE` bei 1.0 liegt.
    """
    return abs(sum(weights.values()) - 1.0) <= WEIGHT_SUM_TOLERANCE
