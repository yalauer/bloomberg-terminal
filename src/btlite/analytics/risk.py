"""Risikometriken fuer Portfolios und Einzelwerte.

Fuellt :class:`~btlite.models.portfolio.RiskMetrics` aus einer Rendite-Reihe,
optional gegen eine Benchmark-Reihe.

Architekturregel 2: Dieses Modul holt keine Daten. Renditen werden
**injiziert**.

Einheiten und Vorzeichen, durchgaengig eingehalten:
    - Volatilitaeten, Renditen und Risikomasse sind annualisiert und dezimal
      (0.12 bedeutet 12 Prozent).
    - Drawdown, VaR und CVaR sind **negativ**. Ein VaR von -0.03 heisst: mit
      der gewaehlten Konfidenz ist der Tagesverlust nicht schlechter als
      3 Prozent.
    - CVaR ist stets kleiner oder gleich VaR. Der Erwartungswert jenseits der
      Schwelle kann nicht besser sein als die Schwelle selbst.

Der Annualisierungsfaktor kommt aus :attr:`Interval.periods_per_year`, nie
hartcodiert - sonst waeren Wochen- und Monatsreihen still falsch skaliert.

Harte Regel des Projekts: ein nicht berechenbarer Wert ist ``None``, nie
``0.0``, nie ``NaN`` und nie ``inf``. Zu wenige Beobachtungen sind dagegen ein
echter Fehler und werfen :class:`InsufficientDataError` - eine aus drei
Punkten geschaetzte Volatilitaet ist keine Kennzahl, sondern Rauschen.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import pandas as pd

from btlite.config import get_settings
from btlite.core.errors import InsufficientDataError
from btlite.core.logging import get_logger
from btlite.models.common import Interval
from btlite.models.portfolio import RiskMetrics

if TYPE_CHECKING:
    from btlite.config import AnalyticsSettings

_LOG = get_logger(__name__)

#: Mindestanzahl Beobachtungen fuer eine Regression (Beta, Alpha, R-Quadrat).
#: Zwei Punkte legen immer eine Gerade exakt fest, das R-Quadrat waere 1.0.
_MIN_REGRESSION_OBS: Final[int] = 3

#: Default-Fenster der rollierenden Kennzahlen in Perioden (ein Quartal).
DEFAULT_ROLLING_WINDOW: Final[int] = 63

#: Relative Schwelle, unterhalb derer eine Streuung als "keine" gilt.
#: Die Standardabweichung von zehn identischen Werten ist nicht exakt 0,
#: sondern liegt bei etwa 1e-18. Ein Test auf ``<= 0.0`` greift dort nicht
#: und die Kennzahl explodiert auf 1e16 statt ``None`` zu liefern. Die
#: Schwelle ist relativ zum Betragsniveau der Reihe, weil eine absolute
#: Grenze echte Reihen mit sehr kleinen Renditen faelschlich verwerfen wuerde.
_DISPERSION_EPSILON: Final[float] = 1e-12


def _has_dispersion(series: pd.Series, spread: float) -> bool:
    """Prueft, ob eine Reihe echte Streuung hat und nicht nur Rundungsrest.

    Args:
        series: Die betrachtete Reihe.
        spread: Ihre Standardabweichung oder Wurzel der Varianz.

    Returns:
        True wenn die Streuung relativ zum Niveau der Reihe bedeutsam ist.
    """
    if not np.isfinite(spread) or spread <= 0.0:
        return False
    scale = float(np.abs(series.to_numpy()).mean())
    return spread > _DISPERSION_EPSILON * max(scale, 1.0)


#: Koeffizienten der Acklam-Approximation der inversen Normalverteilung.
#: Bewusst ohne ``scipy``: das Paket ist keine deklarierte Abhaengigkeit des
#: Projekts, es kaeme nur transitiv ueber PyPortfolioOpt herein. Die
#: Approximation ist auf etwa 1e-9 genau und damit fuer einen VaR um
#: Groessenordnungen praeziser als noetig.
_ACKLAM_A: Final[tuple[float, ...]] = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_ACKLAM_B: Final[tuple[float, ...]] = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_ACKLAM_C: Final[tuple[float, ...]] = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_ACKLAM_D: Final[tuple[float, ...]] = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)

#: Grenze zwischen zentralem und Randbereich der Acklam-Approximation.
_ACKLAM_BREAK: Final[float] = 0.02425


def _normal_ppf(p: float) -> float | None:
    """Quantilfunktion der Standardnormalverteilung.

    Args:
        p: Wahrscheinlichkeit, echt zwischen 0 und 1.

    Returns:
        Das Quantil, oder None wenn ``p`` ausserhalb des offenen
        Intervalls (0, 1) liegt.
    """
    if not 0.0 < p < 1.0:
        return None

    if _ACKLAM_BREAK <= p <= 1.0 - _ACKLAM_BREAK:
        q = p - 0.5
        r = q * q
        numerator = _horner(_ACKLAM_A, r) * q
        denominator = _horner(_ACKLAM_B, r) * r + 1.0
        return numerator / denominator

    # Randbereiche: die Approximation arbeitet dort auf einer anderen
    # Variablen. Der obere Rand ist die Spiegelung des unteren.
    lower = p < _ACKLAM_BREAK
    q = math.sqrt(-2.0 * math.log(p if lower else 1.0 - p))
    numerator = _horner(_ACKLAM_C, q)
    denominator = _horner(_ACKLAM_D, q) * q + 1.0
    value = numerator / denominator
    return value if lower else -value


def _horner(coefficients: tuple[float, ...], x: float) -> float:
    """Wertet ein Polynom nach dem Horner-Schema aus.

    Args:
        coefficients: Koeffizienten, hoechste Potenz zuerst.
        x: Auswertungsstelle.

    Returns:
        Der Polynomwert.
    """
    total = 0.0
    for coefficient in coefficients:
        total = total * x + coefficient
    return total


def _finite(value: object) -> float | None:
    """Gibt einen Wert nur zurueck, wenn er endlich ist.

    Der Waechter fuer die Projektregel "kein stilles NaN": jede Kennzahl
    laeuft hier durch, bevor sie ins Modell geht.

    Args:
        value: Roher Rechenwert. ``object``, weil pandas-Reduktionen wie
            ``skew`` und ``kurt`` eine Union ueber mehrere Skalartypen
            zurueckgeben.

    Returns:
        Der Wert als float, oder None wenn er fehlt, NaN oder unendlich ist,
        oder sich nicht in einen float wandeln laesst.
    """
    if value is None:
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _clean(returns: pd.Series) -> pd.Series:
    """Entfernt fehlende Werte und erzwingt float-Dtype.

    Args:
        returns: Rohe Rendite-Reihe, dezimal.

    Returns:
        Reihe ohne NaN, als float64.
    """
    return pd.to_numeric(returns, errors="coerce").astype("float64").dropna()


def _require_observations(
    returns: pd.Series,
    *,
    minimum: int,
    what: str,
) -> pd.Series:
    """Prueft die Mindestanzahl Beobachtungen.

    Args:
        returns: Rendite-Reihe, dezimal.
        minimum: Erforderliche Anzahl.
        what: Bezeichnung fuer die Fehlermeldung.

    Returns:
        Die bereinigte Reihe.

    Raises:
        InsufficientDataError: Wenn zu wenige Beobachtungen vorliegen.
    """
    clean = _clean(returns)
    if len(clean) < minimum:
        msg = f"Too few observations for {what}."
        raise InsufficientDataError(
            msg,
            detail=f"benoetigt={minimum}, vorhanden={len(clean)}",
            required=minimum,
            available=len(clean),
        )
    return clean


def _align(portfolio: pd.Series, benchmark: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Bringt zwei Rendite-Reihen auf gemeinsame Zeitpunkte.

    Nur die Schnittmenge der Zeitstempel ist auswertbar. Ein Beta gegen
    teilweise disjunkte Reihen waere bedeutungslos.

    Args:
        portfolio: Portfolio-Renditen, dezimal.
        benchmark: Benchmark-Renditen, dezimal.

    Returns:
        Tupel beider Reihen, identisch indiziert und ohne NaN.
    """
    joined = pd.concat(
        {"portfolio": _clean(portfolio), "benchmark": _clean(benchmark)},
        axis=1,
        join="inner",
    ).dropna()
    return joined["portfolio"], joined["benchmark"]


# ------------------------------------------------------------ Einzelmasse --
def volatility(returns: pd.Series, *, interval: Interval = Interval.D1) -> float | None:
    """Annualisierte Volatilitaet der Renditen.

    Verwendet die Stichproben-Standardabweichung (``ddof=1``).

    Args:
        returns: Perioden-Renditen, dezimal.
        interval: Intervall der Reihe, bestimmt die Annualisierung.

    Returns:
        Annualisierte Volatilitaet, dezimal und nicht negativ. None wenn
        nicht berechenbar.
    """
    clean = _clean(returns)
    if len(clean) < 2:
        return None
    return _finite(clean.std(ddof=1) * np.sqrt(interval.periods_per_year))


def downside_deviation(
    returns: pd.Series,
    *,
    target: float = 0.0,
    interval: Interval = Interval.D1,
) -> float | None:
    """Annualisierte Abweichung unterhalb einer Zielrendite.

    Nur Unterschreitungen gehen ein, Ueberschreitungen zaehlen als 0. Der
    Nenner ist die Gesamtzahl der Beobachtungen, nicht nur die der
    Unterschreitungen - so bleibt das Mass mit der Volatilitaet vergleichbar
    und sinkt, wenn Verluste seltener werden.

    Args:
        returns: Perioden-Renditen, dezimal.
        target: Zielrendite je Periode, dezimal.
        interval: Intervall der Reihe, bestimmt die Annualisierung.

    Returns:
        Annualisierte Downside Deviation, dezimal und nicht negativ. None
        wenn nicht berechenbar.
    """
    clean = _clean(returns)
    if clean.empty:
        return None
    shortfall = np.minimum(clean.to_numpy() - target, 0.0)
    root = float(np.sqrt(np.mean(np.square(shortfall))))
    return _finite(root * np.sqrt(interval.periods_per_year))


def annualized_return(returns: pd.Series, *, interval: Interval = Interval.D1) -> float | None:
    """Geometrisch annualisierte Rendite.

    Geometrisch statt arithmetisch: nur die Verkettung entspricht dem
    tatsaechlich erzielten Endvermoegen.

    Args:
        returns: Perioden-Renditen, dezimal.
        interval: Intervall der Reihe, bestimmt die Annualisierung.

    Returns:
        Annualisierte Rendite, dezimal. None wenn nicht berechenbar oder
        wenn ein Totalverlust die Verkettung auf 0 zieht.
    """
    clean = _clean(returns)
    if clean.empty:
        return None
    growth = _finite((1.0 + clean).prod())
    if growth is None:
        return None
    if growth <= 0.0:
        return None
    exponent = interval.periods_per_year / len(clean)
    return _finite(growth**exponent - 1.0)


def sharpe_ratio(
    returns: pd.Series,
    *,
    risk_free_rate: float,
    interval: Interval = Interval.D1,
) -> float | None:
    """Sharpe Ratio, annualisiert.

    Der risikofreie Zins wird auf die Periodenfrequenz heruntergerechnet und
    von jeder Rendite abgezogen; erst das Ergebnis wird annualisiert.

    Args:
        returns: Perioden-Renditen, dezimal.
        risk_free_rate: Jaehrlicher risikofreier Zins, dezimal. Bewusst ohne
            Default - ein stiller Nullzins hebt die Kennzahl an.
        interval: Intervall der Reihe, bestimmt die Annualisierung.

    Returns:
        Annualisierte Sharpe Ratio, dimensionslos. None wenn die
        Volatilitaet 0 oder nicht berechenbar ist.
    """
    clean = _clean(returns)
    if len(clean) < 2:
        return None
    periods = interval.periods_per_year
    excess = clean - risk_free_rate / periods
    sigma = float(excess.std(ddof=1))
    if not _has_dispersion(excess, sigma):
        return None
    return _finite(float(excess.mean()) / sigma * np.sqrt(periods))


def sortino_ratio(
    returns: pd.Series,
    *,
    risk_free_rate: float,
    interval: Interval = Interval.D1,
) -> float | None:
    """Sortino Ratio, annualisiert.

    Wie Sharpe, misst das Risiko aber nur an den Unterschreitungen der
    Zielrendite. Die Zielrendite ist hier der risikofreie Zins.

    Args:
        returns: Perioden-Renditen, dezimal.
        risk_free_rate: Jaehrlicher risikofreier Zins, dezimal.
        interval: Intervall der Reihe, bestimmt die Annualisierung.

    Returns:
        Annualisierte Sortino Ratio, dimensionslos. None wenn es keine
        Unterschreitung gibt und das Mass damit unbestimmt ist.
    """
    clean = _clean(returns)
    if clean.empty:
        return None
    periods = interval.periods_per_year
    per_period = risk_free_rate / periods
    excess = clean - per_period
    shortfall = np.minimum(excess.to_numpy(), 0.0)
    denominator = float(np.sqrt(np.mean(np.square(shortfall))))
    if not _has_dispersion(excess, denominator):
        return None
    return _finite(float(excess.mean()) / denominator * np.sqrt(periods))


def value_at_risk_historical(returns: pd.Series, *, confidence: float = 0.95) -> float | None:
    """Historischer Value at Risk als empirisches Quantil.

    Args:
        returns: Perioden-Renditen, dezimal.
        confidence: Konfidenzniveau, dezimal (0.95 = 95 Prozent).

    Returns:
        VaR der Periode, dezimal und **negativ**. None wenn nicht
        berechenbar. Ein positives Quantil (nur Gewinne in der Reihe) wird
        auf 0.0 begrenzt, damit das Vorzeichen des Vertrags haelt.
    """
    clean = _clean(returns)
    if clean.empty:
        return None
    quantile = _finite(np.quantile(clean.to_numpy(), 1.0 - confidence))
    if quantile is None:
        return None
    return min(quantile, 0.0)


def value_at_risk_parametric(returns: pd.Series, *, confidence: float = 0.95) -> float | None:
    """Parametrischer Value at Risk unter Normalverteilungsannahme.

    Die Annahme ist bewusst dokumentiert: Renditen sind empirisch
    leptokurtisch, dieser VaR unterschaetzt Extremverluste deshalb
    systematisch. Er steht neben dem historischen, nicht an seiner Stelle.

    Args:
        returns: Perioden-Renditen, dezimal.
        confidence: Konfidenzniveau, dezimal.

    Returns:
        VaR der Periode, dezimal und **negativ**. None wenn nicht
        berechenbar.
    """
    clean = _clean(returns)
    if len(clean) < 2:
        return None
    sigma = float(clean.std(ddof=1))
    if not _has_dispersion(clean, sigma):
        return None
    z = _normal_ppf(1.0 - confidence)
    if z is None:
        return None
    value = _finite(float(clean.mean()) + sigma * z)
    if value is None:
        return None
    return min(value, 0.0)


def conditional_value_at_risk(returns: pd.Series, *, confidence: float = 0.95) -> float | None:
    """Historischer Conditional VaR (Expected Shortfall).

    Mittelwert aller Renditen, die den historischen VaR erreichen oder
    unterschreiten. Damit gilt konstruktionsbedingt CVaR <= VaR.

    Args:
        returns: Perioden-Renditen, dezimal.
        confidence: Konfidenzniveau, dezimal.

    Returns:
        CVaR der Periode, dezimal und **negativ**. None wenn nicht
        berechenbar.
    """
    clean = _clean(returns)
    if clean.empty:
        return None
    threshold = value_at_risk_historical(clean, confidence=confidence)
    if threshold is None:
        return None
    tail = clean[clean <= threshold]
    if tail.empty:
        # Kann bei sehr kurzen Reihen auftreten, wenn das Quantil
        # interpoliert und unter jede Beobachtung faellt.
        return threshold
    value = _finite(tail.mean())
    if value is None:
        return None
    return min(value, 0.0)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Underwater-Reihe der Renditen.

    Args:
        returns: Perioden-Renditen, dezimal.

    Returns:
        Reihe der relativen Rueckgaenge vom bisherigen Hoch, dezimal und
        nicht positiv, mit dem Index der Eingangsreihe.
    """
    clean = _clean(returns)
    if clean.empty:
        return pd.Series(dtype="float64", index=clean.index)
    equity = (1.0 + clean).cumprod()
    return equity / equity.cummax() - 1.0


def max_drawdown(
    returns: pd.Series,
) -> tuple[float | None, pd.Timestamp | None, pd.Timestamp | None]:
    """Groesster Rueckgang samt Anfangs- und Tiefpunkt.

    Args:
        returns: Perioden-Renditen, dezimal.

    Returns:
        Tupel aus Drawdown (dezimal, negativ), Zeitpunkt des vorangehenden
        Hochs und Zeitpunkt des Tiefpunkts. Alle drei sind None, wenn kein
        Drawdown bestimmbar ist.
    """
    under = drawdown_series(returns)
    if under.empty or under.dropna().empty:
        return None, None, None

    trough_pos = int(np.argmin(under.to_numpy()))
    value = _finite(under.iloc[trough_pos])
    if value is None or value >= 0.0:
        # Eine monoton steigende Reihe hat keinen Drawdown. Das ist ein
        # fehlender Wert, keine Null.
        return None, None, None

    equity = (1.0 + _clean(returns)).cumprod()
    peak_pos = int(np.argmax(equity.to_numpy()[: trough_pos + 1]))
    index = under.index
    return value, index[peak_pos], index[trough_pos]


def tracking_error(
    portfolio: pd.Series,
    benchmark: pd.Series,
    *,
    interval: Interval = Interval.D1,
) -> float | None:
    """Annualisierte Standardabweichung der Aktivrendite.

    Args:
        portfolio: Portfolio-Renditen, dezimal.
        benchmark: Benchmark-Renditen, dezimal.
        interval: Intervall beider Reihen.

    Returns:
        Tracking Error, annualisiert, dezimal und nicht negativ. None wenn
        nicht berechenbar.
    """
    port, bench = _align(portfolio, benchmark)
    if len(port) < 2:
        return None
    active = port - bench
    return _finite(float(active.std(ddof=1)) * np.sqrt(interval.periods_per_year))


def information_ratio(
    portfolio: pd.Series,
    benchmark: pd.Series,
    *,
    interval: Interval = Interval.D1,
) -> float | None:
    """Aktivrendite je Einheit Tracking Error, annualisiert.

    Args:
        portfolio: Portfolio-Renditen, dezimal.
        benchmark: Benchmark-Renditen, dezimal.
        interval: Intervall beider Reihen.

    Returns:
        Information Ratio, dimensionslos. None wenn der Tracking Error 0
        oder nicht berechenbar ist.
    """
    port, bench = _align(portfolio, benchmark)
    if len(port) < 2:
        return None
    active = port - bench
    sigma = float(active.std(ddof=1))
    if not _has_dispersion(active, sigma):
        return None
    periods = interval.periods_per_year
    return _finite(float(active.mean()) / sigma * np.sqrt(periods))


def beta(portfolio: pd.Series, benchmark: pd.Series) -> float | None:
    """Beta gegen die Benchmark.

    Args:
        portfolio: Portfolio-Renditen, dezimal.
        benchmark: Benchmark-Renditen, dezimal.

    Returns:
        Beta, dimensionslos. None wenn die Benchmark keine Varianz hat oder
        zu wenige gemeinsame Punkte vorliegen.
    """
    port, bench = _align(portfolio, benchmark)
    if len(port) < _MIN_REGRESSION_OBS:
        return None
    variance = float(bench.var(ddof=1))
    if not _has_dispersion(bench, math.sqrt(variance) if variance > 0.0 else 0.0):
        return None
    covariance = float(port.cov(bench))
    return _finite(covariance / variance)


def alpha(
    portfolio: pd.Series,
    benchmark: pd.Series,
    *,
    risk_free_rate: float,
    interval: Interval = Interval.D1,
) -> float | None:
    """Jensens Alpha, annualisiert.

    Ueberrendite gegenueber der vom Beta implizierten Risikopraemie:
    ``alpha = r_p - r_f - beta * (r_b - r_f)``, gerechnet auf
    Periodenbasis und dann annualisiert.

    Args:
        portfolio: Portfolio-Renditen, dezimal.
        benchmark: Benchmark-Renditen, dezimal.
        risk_free_rate: Jaehrlicher risikofreier Zins, dezimal.
        interval: Intervall beider Reihen.

    Returns:
        Annualisiertes Alpha, dezimal. None wenn das Beta nicht bestimmbar
        ist.
    """
    port, bench = _align(portfolio, benchmark)
    factor = beta(port, bench)
    if factor is None:
        return None
    periods = interval.periods_per_year
    per_period = risk_free_rate / periods
    excess_port = float(port.mean()) - per_period
    excess_bench = float(bench.mean()) - per_period
    return _finite((excess_port - factor * excess_bench) * periods)


def r_squared(portfolio: pd.Series, benchmark: pd.Series) -> float | None:
    """Bestimmtheitsmass der Regression gegen die Benchmark.

    Args:
        portfolio: Portfolio-Renditen, dezimal.
        benchmark: Benchmark-Renditen, dezimal.

    Returns:
        R-Quadrat in [0, 1], dimensionslos. None wenn eine der Reihen keine
        Varianz hat oder zu wenige gemeinsame Punkte vorliegen.
    """
    port, bench = _align(portfolio, benchmark)
    if len(port) < _MIN_REGRESSION_OBS:
        return None
    port_var = float(port.var(ddof=1))
    bench_var = float(bench.var(ddof=1))
    if not _has_dispersion(port, math.sqrt(max(port_var, 0.0))) or not _has_dispersion(
        bench, math.sqrt(max(bench_var, 0.0))
    ):
        return None
    correlation = _finite(port.corr(bench))
    if correlation is None:
        return None
    # Numerisches Clipping: der Vertrag verlangt [0, 1], Rundung kann
    # minimal darueber hinausschiessen.
    return float(min(max(correlation**2, 0.0), 1.0))


def skewness(returns: pd.Series) -> float | None:
    """Schiefe der Renditeverteilung.

    Args:
        returns: Perioden-Renditen, dezimal.

    Returns:
        Schiefe, dimensionslos. None wenn nicht berechenbar (weniger als
        drei Punkte oder keine Streuung).
    """
    clean = _clean(returns)
    if len(clean) < _MIN_REGRESSION_OBS:
        return None
    return _finite(clean.skew())


def kurtosis(returns: pd.Series) -> float | None:
    """Exzess-Kurtosis der Renditeverteilung.

    ``pandas.Series.kurt`` liefert bereits den Exzess (Normalverteilung = 0),
    nicht die rohe Kurtosis.

    Args:
        returns: Perioden-Renditen, dezimal.

    Returns:
        Exzess-Kurtosis, dimensionslos. None wenn nicht berechenbar.
    """
    clean = _clean(returns)
    if len(clean) < 4:
        return None
    return _finite(clean.kurt())


# ---------------------------------------------------------- Rollierendes --
def rolling_volatility(
    returns: pd.Series,
    *,
    window: int = DEFAULT_ROLLING_WINDOW,
    interval: Interval = Interval.D1,
) -> pd.Series:
    """Rollierende annualisierte Volatilitaet.

    Args:
        returns: Perioden-Renditen, dezimal.
        window: Fensterbreite in Perioden.
        interval: Intervall der Reihe, bestimmt die Annualisierung.

    Returns:
        Reihe der annualisierten Volatilitaet, dezimal, mit dem Index der
        Eingangsreihe. Die ersten ``window - 1`` Werte sind NaN, weil das
        Fenster dort unvollstaendig ist. Fuer ``viz.risk.rolling_volatility``.

    Raises:
        ValueError: Wenn ``window`` kleiner als 2 ist.
    """
    if window < 2:
        msg = f"window muss mindestens 2 sein, erhalten {window}"
        raise ValueError(msg)
    clean = _clean(returns)
    out = cast("pd.Series", clean.rolling(window).std(ddof=1) * np.sqrt(interval.periods_per_year))
    out.name = "rolling_volatility"
    return out


def rolling_beta(
    portfolio: pd.Series,
    benchmark: pd.Series,
    *,
    window: int = DEFAULT_ROLLING_WINDOW,
) -> pd.Series:
    """Rollierendes Beta gegen die Benchmark.

    Args:
        portfolio: Portfolio-Renditen, dezimal.
        benchmark: Benchmark-Renditen, dezimal.
        window: Fensterbreite in Perioden.

    Returns:
        Reihe des Beta, dimensionslos, auf den gemeinsamen Zeitpunkten
        beider Reihen. Fenster ohne Benchmark-Varianz sind NaN, nicht 0.
        Fuer ``viz.risk.rolling_beta``.

    Raises:
        ValueError: Wenn ``window`` kleiner als 2 ist.
    """
    if window < 2:
        msg = f"window muss mindestens 2 sein, erhalten {window}"
        raise ValueError(msg)
    port, bench = _align(portfolio, benchmark)
    covariance = port.rolling(window).cov(bench)
    variance = bench.rolling(window).var(ddof=1)
    out = covariance / variance.where(variance > 0.0)
    out = out.replace([np.inf, -np.inf], np.nan)
    out.name = "rolling_beta"
    return out


# --------------------------------------------------------------- Fassade --
def compute_risk_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    *,
    interval: Interval = Interval.D1,
    risk_free_rate: float | None = None,
    confidence: float | None = None,
    settings: AnalyticsSettings | None = None,
) -> RiskMetrics:
    """Berechnet den vollstaendigen Satz Risikokennzahlen.

    Ohne ``benchmark_returns`` bleiben die relativen Masse (Beta, Alpha,
    R-Quadrat, Tracking Error, Information Ratio) ``None`` - sie sind ohne
    Referenz nicht definiert, nicht null.

    Args:
        returns: Perioden-Renditen des Portfolios, dezimal.
        benchmark_returns: Perioden-Renditen der Benchmark, dezimal. Wird
            auf die gemeinsamen Zeitpunkte geschnitten.
        interval: Intervall der Reihen, bestimmt die Annualisierung.
        risk_free_rate: Jaehrlicher risikofreier Zins, dezimal. ``None``
            nimmt :attr:`AnalyticsSettings.risk_free_rate`.
        confidence: Konfidenzniveau fuer VaR und CVaR, dezimal. ``None``
            nimmt :attr:`AnalyticsSettings.var_confidence`.
        settings: Analytics-Settings. ``None`` nimmt die globalen.

    Returns:
        Gefuellte Kennzahlen. Nicht berechenbare Felder sind ``None``.

    Raises:
        InsufficientDataError: Wenn weniger als
            :attr:`AnalyticsSettings.min_observations` Renditen vorliegen.
    """
    config = settings if settings is not None else get_settings().analytics
    rate = config.risk_free_rate if risk_free_rate is None else risk_free_rate
    level = config.var_confidence if confidence is None else confidence

    clean = _require_observations(
        returns,
        minimum=config.min_observations,
        what="risk metrics",
    )

    drawdown, start, end = max_drawdown(clean)

    relative: dict[str, float | None] = {
        "beta": None,
        "alpha": None,
        "r_squared": None,
        "tracking_error": None,
        "information_ratio": None,
    }
    if benchmark_returns is not None:
        port, bench = _align(clean, benchmark_returns)
        if port.empty:
            _LOG.debug("Benchmark und Portfolio haben keine gemeinsamen Zeitpunkte")
        else:
            relative = {
                "beta": beta(port, bench),
                "alpha": alpha(port, bench, risk_free_rate=rate, interval=interval),
                "r_squared": r_squared(port, bench),
                "tracking_error": tracking_error(port, bench, interval=interval),
                "information_ratio": information_ratio(port, bench, interval=interval),
            }

    return RiskMetrics(
        volatility=volatility(clean, interval=interval),
        downside_deviation=downside_deviation(clean, interval=interval),
        var_historical=value_at_risk_historical(clean, confidence=level),
        var_parametric=value_at_risk_parametric(clean, confidence=level),
        cvar_historical=conditional_value_at_risk(clean, confidence=level),
        confidence=level,
        max_drawdown=drawdown,
        max_drawdown_start=start.date() if start is not None else None,
        max_drawdown_end=end.date() if end is not None else None,
        sharpe=sharpe_ratio(clean, risk_free_rate=rate, interval=interval),
        sortino=sortino_ratio(clean, risk_free_rate=rate, interval=interval),
        skewness=skewness(clean),
        kurtosis=kurtosis(clean),
        **relative,
    )
