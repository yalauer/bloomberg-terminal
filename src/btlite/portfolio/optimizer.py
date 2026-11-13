"""Portfolio-Optimierung, ein Wrapper um PyPortfolioOpt.

Architekturregel 2: Kurse werden **injiziert** (``dict[str, PriceSeries]``).
Dieses Modul holt nichts selbst.

Vertragsregeln, die der Code einhalten muss, weil das Modell sie prueft:
    - Gewichte summieren sich auf 1.0, Toleranz
      :data:`~btlite.models.portfolio.WEIGHT_SUM_TOLERANCE`.
    - Negative Gewichte nur bei ``allow_shorting``.

Einheiten: erwartete Renditen und Volatilitaeten annualisiert und dezimal
(0.12 bedeutet 12 Prozent). Gewichte dezimal.

Fallstricke von PyPortfolioOpt 1.6, hier bewusst behandelt:
    - ``risk_free_rate`` hat Default ``0.0``. Wird er nicht gesetzt, sind
      Sharpe-Werte still zu hoch. Er wird deshalb ueberall explizit
      durchgereicht.
    - Ein ``EfficientFrontier``-Objekt ist nach **einem** Optimize-Aufruf
      verbraucht. Fuer jede Zielfunktion und jede Stuetzstelle der Grenze
      entsteht deshalb eine neue Instanz.
    - ``portfolio_performance`` ignoriert nach ``max_sharpe`` den ihm
      uebergebenen Zins und nutzt den von ``max_sharpe``. Beide bekommen
      hier denselben Wert, damit die Kennzahl konsistent bleibt.
    - ``HRPOpt.optimize`` prueft den Linkage-Namen gegen ein privates
      scipy-Attribut, das aktuelle scipy-Versionen nicht mehr fuehren, und
      scheitert dort mit ``AttributeError``. HRP ist deshalb in reinem numpy
      nachgebaut, siehe :func:`_hrp_weights` - das spart zugleich die
      undeklarierte scipy-Abhaengigkeit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

import numpy as np
import pandas as pd
from pypfopt import (
    BlackLittermanModel,
    CovarianceShrinkage,
    DiscreteAllocation,
    get_latest_prices,
    risk_models,
)
from pypfopt import (
    EfficientFrontier as _PyPfOptFrontier,
)
from pypfopt import (
    expected_returns as _expected_returns,
)
from pypfopt.exceptions import InstantiationError, OptimizationError

from btlite.config import get_settings
from btlite.core.clock import utc_now
from btlite.core.errors import BTLiteError, InsufficientDataError, OptimizationFailedError
from btlite.core.logging import get_logger
from btlite.models.common import Provenance
from btlite.models.portfolio import (
    WEIGHT_SUM_TOLERANCE,
    CovarianceMethod,
    EfficientFrontier,
    OptimizationObjective,
    PortfolioRequest,
    PortfolioResult,
    Position,
    ReturnsMethod,
    RiskMetrics,
)

if TYPE_CHECKING:
    from btlite.config import PortfolioSettings
    from btlite.models.market import PriceSeries

_LOG = get_logger(__name__)

#: Spaltennamen der Frontier-Stuetzstellen. Vertrag mit ``viz/frontier.py``.
VOLATILITY_COLUMN: Final[str] = "volatility"
EXPECTED_RETURN_COLUMN: Final[str] = "expected_return"
SHARPE_COLUMN: Final[str] = "sharpe"

#: Praefix der optionalen Gewichtsspalten je Ticker in der Frontier.
WEIGHT_PREFIX: Final[str] = "w_"

#: Mindestanzahl gemeinsamer Kursbeobachtungen fuer eine Optimierung.
#: Unterhalb davon ist die Kovarianzmatrix reines Rauschen.
MIN_PRICE_OBSERVATIONS: Final[int] = 30

#: Risikoaversion fuer ``MAX_QUADRATIC_UTILITY``. PyPortfolioOpt-Default.
DEFAULT_RISK_AVERSION: Final[float] = 1.0


def build_price_matrix(prices: dict[str, PriceSeries]) -> pd.DataFrame:
    """Baut die Kursmatrix aus injizierten Kursreihen.

    Nur Zeitpunkte, an denen **alle** Titel einen Kurs haben, gehen ein. Eine
    per Forward-Fill aufgefuellte Reihe wuerde die Kovarianz nach unten
    verzerren, weil kuenstlich konstante Tage die Varianz senken.

    Args:
        prices: Abbildung Ticker auf Kursreihe.

    Returns:
        DataFrame mit UTC-Index und einer Spalte je Ticker, ohne
        Fehlwerte, aufsteigend sortiert.

    Raises:
        InsufficientDataError: Wenn keine oder zu wenige gemeinsame
            Beobachtungen vorliegen.
    """
    if not prices:
        msg = "Optimization is not possible without price series."
        raise InsufficientDataError(msg, detail="prices ist leer", required=1, available=0)

    frame = pd.DataFrame({ticker: series.close for ticker, series in prices.items()})
    frame = frame.sort_index().dropna(how="any")

    if len(frame) < MIN_PRICE_OBSERVATIONS:
        msg = "Too few common trading days for an optimization."
        raise InsufficientDataError(
            msg,
            detail=f"benoetigt={MIN_PRICE_OBSERVATIONS}, vorhanden={len(frame)}",
            required=MIN_PRICE_OBSERVATIONS,
            available=len(frame),
        )
    return frame


def estimate_returns(
    prices: pd.DataFrame,
    method: ReturnsMethod,
    *,
    frequency: int,
    risk_free_rate: float,
) -> pd.Series:
    """Schaetzt die erwarteten Renditen.

    Args:
        prices: Kursmatrix, eine Spalte je Ticker.
        method: Schaetzverfahren.
        frequency: Perioden pro Jahr fuer die Annualisierung.
        risk_free_rate: Jaehrlicher risikofreier Zins, dezimal. Nur ``CAPM``
            nutzt ihn - dort aber wesentlich, siehe Modulhinweis.

    Returns:
        Erwartete Renditen je Ticker, annualisiert und dezimal.
    """
    if method is ReturnsMethod.EMA_HISTORICAL:
        estimated = _expected_returns.ema_historical_return(prices, frequency=frequency)
    elif method is ReturnsMethod.CAPM:
        estimated = _expected_returns.capm_return(
            prices, frequency=frequency, risk_free_rate=risk_free_rate
        )
    else:
        estimated = _expected_returns.mean_historical_return(prices, frequency=frequency)
    return cast("pd.Series", estimated)


def estimate_covariance(
    prices: pd.DataFrame,
    method: CovarianceMethod,
    *,
    frequency: int,
) -> pd.DataFrame:
    """Schaetzt die Kovarianzmatrix.

    Der Default des Projekts ist Ledoit-Wolf-Shrinkage. Die
    Stichprobenkovarianz ist bei vielen Titeln und wenigen Beobachtungen
    schlecht konditioniert und nicht zwingend positiv semidefinit, was den
    Solver scheitern laesst. Alle Ergebnisse laufen hier zusaetzlich durch
    ``fix_nonpositive_semidefinite``.

    Args:
        prices: Kursmatrix, eine Spalte je Ticker.
        method: Schaetzverfahren.
        frequency: Perioden pro Jahr fuer die Annualisierung.

    Returns:
        Annualisierte Kovarianzmatrix, quadratisch und positiv semidefinit.
    """
    if method is CovarianceMethod.SAMPLE:
        matrix = risk_models.sample_cov(prices, frequency=frequency)
    elif method is CovarianceMethod.EXP_WEIGHTED:
        matrix = risk_models.exp_cov(prices, frequency=frequency)
    elif method is CovarianceMethod.SEMICOVARIANCE:
        matrix = risk_models.semicovariance(prices, frequency=frequency)
    else:
        matrix = CovarianceShrinkage(prices, frequency=frequency).ledoit_wolf()
    return cast("pd.DataFrame", risk_models.fix_nonpositive_semidefinite(matrix))


def correlation_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Korrelationsmatrix der Perioden-Renditen.

    Args:
        prices: Kursmatrix, eine Spalte je Ticker.

    Returns:
        Quadratische Korrelationsmatrix, Werte dezimal in [-1, 1]. Fuer
        ``viz.risk.correlation_heatmap``.
    """
    return prices.pct_change().dropna(how="any").corr()


def _weight_bounds(
    *,
    allow_shorting: bool,
    min_weight: float,
    max_weight: float,
) -> tuple[float, float]:
    """Leitet die Gewichtsgrenzen aus der Anfrage ab.

    Args:
        allow_shorting: Ob negative Gewichte erlaubt sind.
        min_weight: Untergrenze je Position, dezimal.
        max_weight: Obergrenze je Position, dezimal.

    Returns:
        Tupel ``(untere, obere)`` Grenze fuer PyPortfolioOpt.
    """
    lower = min_weight if allow_shorting else max(min_weight, 0.0)
    if allow_shorting and lower >= 0.0:
        # Shorting angefordert, aber keine negative Untergrenze gesetzt:
        # -1 ist die uebliche Konvention fuer ein 130/30-artiges Mandat.
        lower = -1.0
    return lower, max_weight


def _new_frontier(
    mu: pd.Series,
    covariance: pd.DataFrame,
    bounds: tuple[float, float],
) -> _PyPfOptFrontier:
    """Erzeugt eine frische Optimizer-Instanz.

    Notwendig, weil eine Instanz nach einem Optimize-Aufruf verbraucht ist.

    Args:
        mu: Erwartete Renditen, annualisiert.
        covariance: Kovarianzmatrix, annualisiert.
        bounds: Gewichtsgrenzen.

    Returns:
        Neue, unbenutzte Instanz.
    """
    return _PyPfOptFrontier(mu, covariance, weight_bounds=bounds)


def _single_linkage_order(distance: np.ndarray) -> list[int]:
    """Ordnet Titel per Single-Linkage-Clusterung quasi-diagonal an.

    Bewusst in reinem numpy statt ueber ``scipy.cluster.hierarchy``: scipy
    ist keine deklarierte Abhaengigkeit des Projekts, es kaeme nur transitiv
    ueber PyPortfolioOpt herein. Zudem prueft ``HRPOpt.optimize`` den
    Linkage-Namen gegen das private ``sch._LINKAGE_METHODS``, das aktuelle
    scipy-Versionen nicht mehr fuehren - der Aufruf scheitert dort mit
    ``AttributeError``, bevor er ueberhaupt rechnet.

    Das Verfahren baut denselben Dendrogramm-Baum wie SciPys
    ``linkage(..., "single")``: es verschmilzt wiederholt die beiden
    naechstgelegenen Cluster, wobei der Abstand zweier Cluster der kleinste
    Abstand ihrer Mitglieder ist. Neue Cluster bekommen aufsteigende
    Bezeichner ab ``n``, genau wie in SciPys Verkettungsmatrix. Die
    Reihenfolge entsteht danach durch rekursives Ablaufen des Baums, sodass
    aehnliche Titel nebeneinander stehen - genau das braucht die Bisektion
    von HRP, um entlang echter Korrelationsbloecke zu schneiden.

    Args:
        distance: Quadratische Abstandsmatrix, symmetrisch.

    Returns:
        Positionen der Titel in quasi-diagonaler Reihenfolge.
    """
    size = distance.shape[0]
    if size < 2:
        return list(range(size))

    members: dict[int, list[int]] = {i: [i] for i in range(size)}
    children: dict[int, tuple[int, int]] = {}
    active: list[int] = list(range(size))
    next_id = size

    while len(active) > 1:
        best: tuple[float, int, int] | None = None
        for pos_a, key_a in enumerate(active):
            for key_b in active[pos_a + 1 :]:
                gap = float(np.min(distance[np.ix_(members[key_a], members[key_b])]))
                if best is None or gap < best[0]:
                    best = (gap, key_a, key_b)
        if best is None:  # pragma: no cover - bei size >= 2 unerreichbar
            break

        _, key_a, key_b = best
        children[next_id] = (key_a, key_b)
        members[next_id] = members[key_a] + members[key_b]
        active = [k for k in active if k not in (key_a, key_b)]
        active.append(next_id)
        next_id += 1

    # Baum iterativ ablaufen: ein innerer Knoten wird durch seine beiden
    # Kinder ersetzt, ein Blatt bleibt stehen.
    order = [active[0]]
    while any(node >= size for node in order):
        expanded: list[int] = []
        for node in order:
            if node < size:
                expanded.append(node)
            else:
                left, right = children[node]
                expanded.extend((left, right))
        order = expanded
    return order


def _inverse_variance_weights(covariance: np.ndarray, members: list[int]) -> np.ndarray:
    """Inverse-Varianz-Gewichte innerhalb eines Clusters.

    Args:
        covariance: Quadratische Kovarianzmatrix.
        members: Positionen der Clustermitglieder.

    Returns:
        Gewichte in der Reihenfolge von ``members``, Summe 1.0.
    """
    variances = np.diag(covariance)[members]
    # Eine Varianz von 0 wuerde durch Null teilen. Sie kann bei konstanten
    # Reihen auftreten und bekommt dann ein neutrales Gewicht.
    safe = np.where(variances > 0.0, variances, np.inf)
    inverse = 1.0 / safe
    total = float(inverse.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.full(len(members), 1.0 / len(members))
    return inverse / total


def _cluster_variance(covariance: np.ndarray, members: list[int]) -> float:
    """Varianz eines inverse-varianz-gewichteten Clusters.

    Args:
        covariance: Quadratische Kovarianzmatrix.
        members: Positionen der Clustermitglieder.

    Returns:
        Portfoliovarianz des Clusters.
    """
    weights = _inverse_variance_weights(covariance, members)
    block = covariance[np.ix_(members, members)]
    return float(weights @ block @ weights)


def _hrp_weights(returns: pd.DataFrame) -> dict[str, float]:
    """Hierarchical Risk Parity nach Lopez de Prado.

    Drei Schritte: Korrelationsabstaende clustern, die Titel quasi-diagonal
    ordnen, dann rekursiv bisektieren und das Kapital zwischen den beiden
    Haelften invers zu ihrer Clustervarianz aufteilen.

    HRP braucht keine Matrixinversion und keinen Solver. Es ist damit robust
    gegen die schlecht konditionierten Kovarianzmatrizen, an denen die
    Mittelwert-Varianz-Optimierung scheitert.

    Args:
        returns: Perioden-Renditen, eine Spalte je Ticker. HRP arbeitet auf
            **Renditen**, nicht auf Preisen.

    Returns:
        Gewichte je Ticker, dezimal und in Summe 1.0.

    Raises:
        OptimizationFailedError: Wenn die Clusterung scheitert.
    """
    try:
        correlation = returns.corr()
        covariance = returns.cov()
        tickers = [str(t) for t in correlation.index]
        if len(tickers) < 2:
            msg = "HRP needs at least two titles."
            raise OptimizationFailedError(msg, detail=f"titel={tickers}")

        # Abstandsmass nach Lopez de Prado, geklippt gegen
        # Fliesskomma-Ausreisser knapp ausserhalb von [0, 1].
        distance = np.sqrt(np.clip((1.0 - correlation.to_numpy()) / 2.0, a_min=0.0, a_max=1.0))
        cov_matrix = covariance.to_numpy()
        order = _single_linkage_order(distance)

        weights = np.ones(len(tickers), dtype="float64")
        clusters: list[list[int]] = [order]
        while clusters:
            # Jeden Cluster mit mehr als einem Mitglied halbieren.
            clusters = [
                half
                for cluster in clusters
                for half in (cluster[: len(cluster) // 2], cluster[len(cluster) // 2 :])
                if len(cluster) > 1
            ]
            for i in range(0, len(clusters), 2):
                left, right = clusters[i], clusters[i + 1]
                var_left = _cluster_variance(cov_matrix, left)
                var_right = _cluster_variance(cov_matrix, right)
                total = var_left + var_right
                # Der risikoaermere Cluster bekommt den groesseren Anteil.
                alpha = 0.5 if total <= 0.0 else 1.0 - var_left / total
                weights[left] *= alpha
                weights[right] *= 1.0 - alpha
    except (ValueError, TypeError, IndexError, np.linalg.LinAlgError) as exc:
        msg = "The HRP clustering failed."
        raise OptimizationFailedError(msg, detail=f"{type(exc).__name__}: {exc}") from exc

    return {ticker: float(weights[i]) for i, ticker in enumerate(tickers)}


def _black_litterman_inputs(
    mu: pd.Series,
    covariance: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """Baut Posterior-Renditen und -Kovarianz nach Black-Litterman.

    Ohne Marktkapitalisierungen und ohne Nutzer-Views ist der Prior die
    geschaetzte Rendite selbst und die Views sind mit ihm identisch. Das
    Ergebnis unterscheidet sich dann nur ueber die Unsicherheitsgewichtung
    von der Ausgangsschaetzung - fachlich korrekt und ohne erfundene
    Annahmen. Views injiziert spaeter das UI.

    Args:
        mu: Prior-Renditen, annualisiert und dezimal.
        covariance: Kovarianzmatrix, annualisiert.

    Returns:
        Tupel aus Posterior-Renditen und Posterior-Kovarianz.

    Raises:
        OptimizationFailedError: Wenn das Modell nicht loesbar ist.
    """
    try:
        model = BlackLittermanModel(
            covariance,
            pi=mu,
            absolute_views={str(t): float(mu[t]) for t in mu.index},
        )
        return model.bl_returns(), model.bl_cov()
    except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
        msg = "The Black-Litterman model is not solvable."
        raise OptimizationFailedError(msg, detail=f"{type(exc).__name__}: {exc}") from exc


def _optimize_weights(
    objective: OptimizationObjective,
    mu: pd.Series,
    covariance: pd.DataFrame,
    *,
    bounds: tuple[float, float],
    risk_free_rate: float,
    target_return: float | None,
    target_volatility: float | None,
    returns: pd.DataFrame,
) -> dict[str, float]:
    """Loest die gewaehlte Zielfunktion.

    Args:
        objective: Zielfunktion.
        mu: Erwartete Renditen, annualisiert.
        covariance: Kovarianzmatrix, annualisiert.
        bounds: Gewichtsgrenzen.
        risk_free_rate: Jaehrlicher risikofreier Zins, dezimal. Wird explizit
            an ``max_sharpe`` gereicht, sonst waere er still 0.0.
        target_return: Zielrendite, nur bei ``EFFICIENT_RETURN``.
        target_volatility: Zielvolatilitaet, nur bei ``EFFICIENT_RISK``.
        returns: Perioden-Renditen, nur fuer HRP.

    Returns:
        Bereinigte Gewichte je Ticker, dezimal.

    Raises:
        OptimizationFailedError: Wenn der Solver nicht konvergiert oder eine
            noetige Zielvorgabe fehlt.
    """
    if objective is OptimizationObjective.EQUAL_WEIGHT:
        share = 1.0 / len(mu.index)
        return {str(t): share for t in mu.index}

    if objective is OptimizationObjective.HRP:
        return _hrp_weights(returns)

    working_mu, working_cov = mu, covariance
    if objective is OptimizationObjective.BLACK_LITTERMAN:
        working_mu, working_cov = _black_litterman_inputs(mu, covariance)

    optimizer = _new_frontier(working_mu, working_cov, bounds)
    try:
        if objective is OptimizationObjective.MIN_VOLATILITY:
            optimizer.min_volatility()
        elif objective is OptimizationObjective.EFFICIENT_RETURN:
            if target_return is None:  # pragma: no cover - Modell validiert das
                msg = "EFFICIENT_RETURN requires target_return."
                raise OptimizationFailedError(msg, detail="target_return=None")
            optimizer.efficient_return(target_return)
        elif objective is OptimizationObjective.EFFICIENT_RISK:
            if target_volatility is None:  # pragma: no cover - Modell validiert das
                msg = "EFFICIENT_RISK requires target_volatility."
                raise OptimizationFailedError(msg, detail="target_volatility=None")
            optimizer.efficient_risk(target_volatility)
        elif objective is OptimizationObjective.MAX_QUADRATIC_UTILITY:
            optimizer.max_quadratic_utility(risk_aversion=DEFAULT_RISK_AVERSION)
        else:
            optimizer.max_sharpe(risk_free_rate=risk_free_rate)
    except (OptimizationError, InstantiationError, ValueError, TypeError) as exc:
        msg = "The optimization did not converge."
        raise OptimizationFailedError(
            msg, detail=f"objective={objective.value}, {type(exc).__name__}: {exc}"
        ) from exc

    return {str(k): float(v) for k, v in optimizer.clean_weights().items()}


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Erzwingt die Gewichtssumme 1.0 innerhalb der Vertragstoleranz.

    ``clean_weights`` rundet auf fuenf Nachkommastellen und kann die Summe
    dabei um einige 1e-6 verschieben - genug, um den Modellvalidator
    auszuloesen. Die Restdifferenz kommt auf die betragsgroesste Position,
    weil sie dort relativ am wenigsten veraendert.

    Args:
        weights: Rohe Gewichte je Ticker.

    Returns:
        Gewichte mit Summe 1.0.

    Raises:
        OptimizationFailedError: Wenn die Summe 0 ist und sich nicht
            normieren laesst.
    """
    total = sum(weights.values())
    if abs(total) < WEIGHT_SUM_TOLERANCE:
        msg = "The optimization returned a weight sum of 0."
        raise OptimizationFailedError(msg, detail=f"summe={total!r}")

    out = dict(weights)
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE / 10.0:
        out = {k: v / total for k, v in out.items()}

    residual = 1.0 - sum(out.values())
    if residual:
        anchor = max(out, key=lambda k: abs(out[k]))
        out[anchor] += residual
    return out


def _performance(
    weights: dict[str, float],
    mu: pd.Series,
    covariance: pd.DataFrame,
    *,
    risk_free_rate: float,
) -> tuple[float | None, float | None, float | None]:
    """Berechnet erwartete Rendite, Volatilitaet und Sharpe Ratio.

    Bewusst selbst gerechnet statt ueber ``portfolio_performance``: dieses
    ignoriert nach ``max_sharpe`` den uebergebenen Zins und meldet dann eine
    Sharpe Ratio zum falschen Zins.

    Args:
        weights: Gewichte je Ticker, dezimal.
        mu: Erwartete Renditen, annualisiert.
        covariance: Kovarianzmatrix, annualisiert.
        risk_free_rate: Jaehrlicher risikofreier Zins, dezimal.

    Returns:
        Tupel aus erwarteter Rendite, Volatilitaet und Sharpe Ratio, alle
        annualisiert und dezimal. Einzelne Werte sind None, wenn sie nicht
        berechenbar sind.
    """
    order = list(mu.index)
    vector = np.array([weights.get(str(t), 0.0) for t in order], dtype="float64")

    expected = float(vector @ mu.to_numpy())
    variance = float(vector @ covariance.to_numpy() @ vector)
    volatility = float(np.sqrt(variance)) if variance > 0.0 else None

    sharpe: float | None = None
    if volatility is not None and volatility > 0.0:
        sharpe = (expected - risk_free_rate) / volatility

    return (
        expected if np.isfinite(expected) else None,
        volatility if volatility is not None and np.isfinite(volatility) else None,
        sharpe if sharpe is not None and np.isfinite(sharpe) else None,
    )


def compute_frontier(
    mu: pd.Series,
    covariance: pd.DataFrame,
    *,
    points: int,
    bounds: tuple[float, float],
    risk_free_rate: float,
    tickers: list[str],
    include_weights: bool = True,
) -> EfficientFrontier:
    """Berechnet die Effizienzgrenze als Stuetzstellen.

    Die Grenze entsteht durch Abtasten der Zielrendite zwischen der Rendite
    des Minimum-Varianz-Portfolios und der hoechsten Einzelrendite. Fuer
    jede Stuetzstelle wird eine **neue** Optimizer-Instanz gebaut, weil eine
    Instanz nach einem Aufruf verbraucht ist. Unloesbare Stuetzstellen werden
    uebersprungen, nicht als Fehler gemeldet - die Grenze ist am Rand des
    zulaessigen Bereichs naturgemaess luecklig.

    Args:
        mu: Erwartete Renditen, annualisiert und dezimal.
        covariance: Kovarianzmatrix, annualisiert.
        points: Anzahl Stuetzstellen.
        bounds: Gewichtsgrenzen.
        risk_free_rate: Jaehrlicher risikofreier Zins, dezimal. Geht in die
            Sharpe-Spalte ein.
        tickers: Universum, fuer das Ergebnismodell.
        include_weights: Ob je Ticker eine Spalte ``w_<TICKER>`` mitgegeben
            wird.

    Returns:
        Stuetzstellen mit den Spalten ``volatility``, ``expected_return``
        und ``sharpe``, aufsteigend nach Volatilitaet.

    Raises:
        OptimizationFailedError: Wenn keine einzige Stuetzstelle loesbar ist.
    """
    lower_optimizer = _new_frontier(mu, covariance, bounds)
    try:
        lower_optimizer.min_volatility()
        floor = float(lower_optimizer.portfolio_performance(risk_free_rate=risk_free_rate)[0])
    except (OptimizationError, InstantiationError, ValueError) as exc:
        msg = "The efficient frontier cannot be computed."
        raise OptimizationFailedError(msg, detail=f"{type(exc).__name__}: {exc}") from exc

    ceiling = float(mu.max())
    if ceiling <= floor:
        # Entartetes Universum: alle Titel haben dieselbe erwartete Rendite.
        ceiling = floor + abs(floor) * 0.01 + 1e-4

    rows: list[dict[str, float]] = []
    for target in np.linspace(floor, ceiling, points):
        optimizer = _new_frontier(mu, covariance, bounds)
        try:
            optimizer.efficient_return(float(target))
            weights = {str(k): float(v) for k, v in optimizer.clean_weights().items()}
        except (OptimizationError, InstantiationError, ValueError) as exc:
            _LOG.debug("Stuetzstelle bei Zielrendite %.6f uebersprungen: %s", target, exc)
            continue

        expected, volatility, sharpe = _performance(
            weights, mu, covariance, risk_free_rate=risk_free_rate
        )
        if expected is None or volatility is None:
            continue

        row: dict[str, float] = {
            VOLATILITY_COLUMN: volatility,
            EXPECTED_RETURN_COLUMN: expected,
            SHARPE_COLUMN: sharpe if sharpe is not None else float("nan"),
        }
        if include_weights:
            row.update({f"{WEIGHT_PREFIX}{t}": weights.get(str(t), 0.0) for t in mu.index})
        rows.append(row)

    if not rows:
        msg = "The efficient frontier has no solvable support point."
        raise OptimizationFailedError(msg, detail=f"points={points}, bounds={bounds}")

    data = pd.DataFrame(rows).sort_values(VOLATILITY_COLUMN).reset_index(drop=True)
    return EfficientFrontier(data=data, tickers=tickers)


def asset_scatter(
    mu: pd.Series,
    covariance: pd.DataFrame,
) -> pd.DataFrame:
    """Einzelwerte als Streupunkte fuer die Frontier-Grafik.

    Args:
        mu: Erwartete Renditen, annualisiert und dezimal.
        covariance: Kovarianzmatrix, annualisiert.

    Returns:
        DataFrame mit Index Ticker und den Spalten ``volatility`` und
        ``expected_return``. Format von ``viz.frontier.efficient_frontier``
        im Parameter ``assets``.
    """
    variances = np.diag(covariance.to_numpy())
    return pd.DataFrame(
        {
            VOLATILITY_COLUMN: np.sqrt(np.clip(variances, a_min=0.0, a_max=None)),
            EXPECTED_RETURN_COLUMN: mu.to_numpy(),
        },
        index=pd.Index([str(t) for t in mu.index], name="ticker"),
    )


def allocate(
    weights: dict[str, float],
    prices: pd.DataFrame,
    *,
    total_value: float,
) -> tuple[dict[str, float], dict[str, float], float]:
    """Rechnet Zielgewichte in ganze Stueckzahlen um.

    ``greedy_portfolio`` ist bewusst gewaehlt: es ist robuster als das
    LP-Verfahren, das bei kleinen Portfolios oder engen Budgets keine
    zulaessige Loesung findet.

    Args:
        weights: Zielgewichte je Ticker, dezimal. Sollten bereits durch
            ``clean_weights`` gelaufen sein, sonst erzeugen Mini-Gewichte
            sinnlose Ein-Stueck-Positionen.
        prices: Kursmatrix, eine Spalte je Ticker.
        total_value: Anlagebetrag in Kontowaehrung.

    Returns:
        Tupel aus Stueckzahlen je Ticker, den zugehoerigen letzten Kursen
        und dem nicht allokierten Restbetrag.
    """
    latest = get_latest_prices(prices)
    # Nur Titel mit echtem Gewicht: eine 0-Position wuerde die Allokation
    # unnoetig aufblaehen.
    relevant = {t: w for t, w in weights.items() if abs(w) > WEIGHT_SUM_TOLERANCE}
    if not relevant:
        return {}, {}, float(total_value)

    try:
        allocation, leftover = DiscreteAllocation(
            relevant, latest, total_portfolio_value=total_value
        ).greedy_portfolio()
    except (ValueError, TypeError, KeyError) as exc:
        _LOG.warning("Discrete Allocation uebersprungen: %s", exc)
        return {}, {}, float(total_value)

    shares = {str(t): float(n) for t, n in allocation.items()}
    used = {str(t): float(latest[t]) for t in shares if t in latest.index}
    return shares, used, float(max(leftover, 0.0))


def optimize(
    request: PortfolioRequest,
    prices: dict[str, PriceSeries],
    *,
    settings: PortfolioSettings | None = None,
    frequency: int | None = None,
    with_frontier: bool = True,
    with_allocation: bool = True,
    provenance: Provenance | None = None,
) -> PortfolioResult:
    """Fuehrt eine vollstaendige Portfolio-Optimierung aus.

    Architekturregel 2: ``prices`` wird injiziert, dieses Modul holt nichts.

    Args:
        request: Die Anfrage mit Zielfunktion, Schaetzern und Grenzen.
        prices: Abbildung Ticker auf Kursreihe. Ticker der Anfrage ohne
            Kursreihe werden uebersprungen und als Warnung vermerkt.
        settings: Portfolio-Settings. ``None`` nimmt die globalen. Liefert
            ``frontier_points``.
        frequency: Perioden pro Jahr. ``None`` leitet sie aus dem Intervall
            der ersten Kursreihe ab, nie hartcodiert.
        with_frontier: Ob die Effizienzgrenze berechnet wird.
        with_allocation: Ob die Discrete Allocation gerechnet wird.
        provenance: Herkunft der Kursbasis. ``None`` uebernimmt sie von der
            ersten Kursreihe.

    Returns:
        Das vollstaendige Ergebnis. Gewichte summieren sich auf 1.0.

    Raises:
        InsufficientDataError: Wenn zu wenige gemeinsame Kurstage vorliegen
            oder weniger als zwei Ticker Kurse haben.
        OptimizationFailedError: Wenn der Solver nicht konvergiert.
    """
    config = settings if settings is not None else get_settings().portfolio
    warnings: list[str] = []

    available = {t: prices[t] for t in request.tickers if t in prices}
    missing = [t for t in request.tickers if t not in prices]
    if missing:
        warnings.append(f"Skipped, no price series: {', '.join(missing)}")

    if len(available) < 2:
        msg = "An optimization needs at least two titles with prices."
        raise InsufficientDataError(
            msg,
            detail=f"vorhanden={sorted(available)}, fehlend={missing}",
            required=2,
            available=len(available),
        )

    matrix = build_price_matrix(available)
    first = next(iter(available.values()))
    periods = frequency if frequency is not None else int(first.interval.periods_per_year)

    mu = estimate_returns(
        matrix,
        request.returns_method,
        frequency=periods,
        risk_free_rate=request.risk_free_rate,
    )
    covariance = estimate_covariance(matrix, request.covariance_method, frequency=periods)
    period_returns = matrix.pct_change().dropna(how="any")

    bounds = _weight_bounds(
        allow_shorting=request.allow_shorting,
        min_weight=request.min_weight,
        max_weight=request.max_weight,
    )

    raw = _optimize_weights(
        request.objective,
        mu,
        covariance,
        bounds=bounds,
        risk_free_rate=request.risk_free_rate,
        target_return=request.target_return,
        target_volatility=request.target_volatility,
        returns=period_returns,
    )
    weights = _normalize_weights(raw)

    if not request.allow_shorting:
        negative = {t: w for t, w in weights.items() if w < -WEIGHT_SUM_TOLERANCE}
        if negative:  # pragma: no cover - Grenzen schliessen das aus
            msg = "The optimization returned negative weights without allow_shorting."
            raise OptimizationFailedError(msg, detail=f"negativ={negative}")

    expected, volatility, sharpe = _performance(
        weights, mu, covariance, risk_free_rate=request.risk_free_rate
    )

    shares: dict[str, float] = {}
    used_prices: dict[str, float] = {}
    leftover: float | None = None
    if with_allocation:
        shares, used_prices, leftover = allocate(weights, matrix, total_value=request.total_value)

    positions = [
        Position(
            ticker=ticker,
            weight=weights[ticker],
            shares=shares.get(ticker),
            price=used_prices.get(ticker),
            value=(
                shares[ticker] * used_prices[ticker]
                if ticker in shares and ticker in used_prices
                else None
            ),
        )
        for ticker in sorted(weights)
    ]

    frontier: EfficientFrontier | None = None
    if with_frontier:
        try:
            frontier = compute_frontier(
                mu,
                covariance,
                points=config.frontier_points,
                bounds=bounds,
                risk_free_rate=request.risk_free_rate,
                tickers=list(matrix.columns),
            )
        except OptimizationFailedError as exc:
            # Die Grenze ist Beiwerk. Scheitert sie, bleibt das Ergebnis
            # gueltig und der Nutzer erfaehrt den Grund.
            warnings.append(f"Efficient frontier cannot be computed: {exc.user_message}")
            _LOG.debug("Frontier uebersprungen: %s", exc)

    return PortfolioResult(
        request=request,
        positions=positions,
        expected_return=expected,
        expected_volatility=volatility,
        sharpe_ratio=sharpe,
        risk=_realized_risk(weights, period_returns, request, warnings),
        frontier=frontier,
        correlation=correlation_matrix(matrix),
        leftover_cash=leftover,
        currency=first.currency,
        computed_at=utc_now(),
        provenance=provenance if provenance is not None else first.provenance,
        warnings=warnings,
    )


def _realized_risk(
    weights: dict[str, float],
    period_returns: pd.DataFrame,
    request: PortfolioRequest,
    warnings: list[str],
) -> RiskMetrics | None:
    """Berechnet die realisierten Risikokennzahlen des Gesamtportfolios.

    Gewichtet die Einzelrenditen zu einer Portfolio-Renditereihe und misst
    deren Historie. Das unterscheidet sich von ``expected_volatility``: die
    erwarteten Werte stammen aus den Schaetzern der Optimierung, die
    realisierten aus der tatsaechlich beobachteten Kurshistorie. Beide
    nebeneinander zu sehen ist der Sinn der Sache.

    Die Gewichte gelten in jeder Periode neu, passend zur Annahme einer
    Optimierung mit konstanten Zielgewichten.

    Args:
        weights: Optimierte Gewichte je Ticker, dezimal.
        period_returns: Periodenrenditen je Ticker, bereits auf einen
            gemeinsamen Index gebracht.
        request: Die Anfrage, fuer den risikofreien Zins.
        warnings: Warnliste, die im Fehlerfall ergaenzt wird.

    Returns:
        Risikokennzahlen, oder ``None`` wenn die Historie dafuer nicht
        reicht. Dann steht im UI sichtbar "n/a" statt einer erfundenen Zahl.
    """
    from btlite.analytics.risk import compute_risk_metrics

    try:
        aligned = pd.Series(weights).reindex(period_returns.columns).fillna(0.0)
        portfolio = period_returns.mul(aligned, axis=1).sum(axis=1)
        portfolio.name = "portfolio"
        return compute_risk_metrics(portfolio, risk_free_rate=request.risk_free_rate)
    except (BTLiteError, KeyError, ValueError) as exc:
        _LOG.info("Realisiertes Portfolio-Risiko nicht berechenbar: %s", exc)
        warnings.append("Realized risk metrics cannot be computed.")
        return None
