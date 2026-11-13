"""Engine-unabhaengige Kennzahlen einer Equity-Kurve.

Dieses Modul ist die einzige Stelle, an der Performance-Kennzahlen entstehen.
Beide Engines - ``VectorbtEngine`` und ``BacktraderEngine`` - rufen es auf,
damit ihre Zahlen direkt vergleichbar sind und nicht von der jeweiligen
Bibliotheks-Konvention abhaengen.

Einheiten-Konvention des Projekts, durchgehend eingehalten:

* Renditen, Volatilitaeten und Risikomasse sind **annualisiert und dezimal**
  (0.12 = 12 Prozent pro Jahr). Ausnahme sind ``total_return`` (Rendite ueber
  den gesamten Zeitraum, dezimal) und die absoluten Trade-Groessen
  ``avg_win``, ``avg_loss`` und ``total_fees`` (Kontowaehrung).
* Drawdown, VaR und CVaR sind **negativ**. ``cvar_95 <= var_95``.
* Ein nicht berechenbarer Wert ist ``None`` - nie ``0.0``, nie ``NaN``,
  nie ``inf``.

Der Annualisierungsfaktor kommt aus :attr:`~btlite.models.common.Interval.periods_per_year`
und ist bewusst nirgends als 252 hartcodiert: fuer Wochen- oder
Intraday-Intervalle waere das falsch.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from btlite.models.backtest import PerformanceMetrics
from btlite.models.common import Currency, Interval

if TYPE_CHECKING:
    from collections.abc import Sequence

    from btlite.models.backtest import Trade
    from btlite.models.market import PriceSeries

#: Konfidenzniveau von VaR und CVaR, dezimal.
VAR_CONFIDENCE: Final[float] = 0.95

#: Mindestanzahl Renditebeobachtungen fuer Streuungsmasse. Unter zwei Punkten
#: ist die Stichprobenvarianz undefiniert (ddof=1).
MIN_OBS_DISPERSION: Final[int] = 2

#: Mindestanzahl Renditebeobachtungen fuer eine Regression gegen die
#: Benchmark. Mit zwei Punkten ist Beta zwar rechenbar, aber wertlos.
MIN_OBS_REGRESSION: Final[int] = 3

#: Absolute Untergrenze, unterhalb der ein Nenner als Null gilt.
_EPS: Final[float] = 1e-12


def _clean(value: float | None) -> float | None:
    """Filtert nicht darstellbare Gleitkommawerte auf ``None``.

    Zentrale Durchsetzung der Projektregel "kein stilles NaN": jeder
    Rueckgabewert dieses Moduls laeuft durch diese Funktion.

    Args:
        value: Rohwert einer Berechnung.

    Returns:
        Der Wert, oder None wenn er fehlt, NaN oder unendlich ist.
    """
    if value is None:
        return None
    out = float(value)
    if not math.isfinite(out):
        return None
    return out


def _as_float_series(values: pd.Series) -> pd.Series:
    """Bringt eine Reihe auf float und entfernt fehlende Werte.

    Args:
        values: Beliebige numerische Reihe.

    Returns:
        Reihe mit dtype float64 ohne NaN, Index erhalten.
    """
    return pd.to_numeric(values, errors="coerce").astype(float).dropna()


def to_returns(equity: pd.Series) -> pd.Series:
    """Rechnet eine Equity-Kurve in einfache periodische Renditen um.

    Nicht-positive Equity-Werte machen eine Renditeberechnung sinnlos (ein
    Totalverlust ist nicht fortsetzbar), deshalb werden sie verworfen statt
    stillschweigend zu ``inf`` zu fuehren.

    Args:
        equity: Portfoliowert je Bar, Kontowaehrung, UTC-Index.

    Returns:
        Einfache Renditen je Bar, dezimal. Die erste Beobachtung entfaellt.
    """
    clean = _as_float_series(equity)
    clean = clean[clean > 0.0]
    if len(clean) < MIN_OBS_DISPERSION:
        return pd.Series(dtype=float, index=clean.index[:0])
    return clean.pct_change().dropna()


def annualization_factor(interval: Interval) -> float:
    """Liefert die Anzahl Perioden pro Jahr fuer ein Bar-Intervall.

    Args:
        interval: Bar-Intervall der Equity-Kurve.

    Returns:
        Perioden pro Jahr, dezimal (252.0 fuer Tagesdaten).
    """
    return float(interval.periods_per_year)


def total_return(equity: pd.Series) -> float | None:
    """Gesamtrendite ueber den gesamten Zeitraum.

    Args:
        equity: Portfoliowert je Bar, Kontowaehrung.

    Returns:
        Rendite ueber den Zeitraum, dezimal. None bei zu kurzer oder
        nicht-positiver Reihe.
    """
    clean = _as_float_series(equity)
    if len(clean) < MIN_OBS_DISPERSION:
        return None
    first = float(clean.iloc[0])
    if abs(first) < _EPS:
        return None
    return _clean(float(clean.iloc[-1]) / first - 1.0)


def cagr(equity: pd.Series, *, interval: Interval = Interval.D1) -> float | None:
    """Geometrische Jahresrendite (Compound Annual Growth Rate).

    Die Laufzeit wird in Bars gemessen und ueber
    :func:`annualization_factor` in Jahre umgerechnet. Das ist konsistent
    mit allen anderen annualisierten Kennzahlen dieses Moduls und
    unabhaengig davon, ob die Reihe Kalender- oder Handelstage enthaelt.

    Args:
        equity: Portfoliowert je Bar, Kontowaehrung.
        interval: Bar-Intervall fuer die Annualisierung.

    Returns:
        Jaehrliche Wachstumsrate, annualisiert und dezimal. None bei zu
        kurzer Reihe oder wenn das Kapital auf oder unter Null fiel - eine
        geometrische Rate ist dann nicht definiert.
    """
    clean = _as_float_series(equity)
    if len(clean) < MIN_OBS_DISPERSION:
        return None
    first = float(clean.iloc[0])
    last = float(clean.iloc[-1])
    if first <= 0.0 or last <= 0.0:
        return None
    years = (len(clean) - 1) / annualization_factor(interval)
    if years <= 0.0:
        return None
    return _clean((last / first) ** (1.0 / years) - 1.0)


def volatility(returns: pd.Series, *, interval: Interval = Interval.D1) -> float | None:
    """Annualisierte Standardabweichung der periodischen Renditen.

    Args:
        returns: Periodische Renditen, dezimal.
        interval: Bar-Intervall fuer die Annualisierung.

    Returns:
        Annualisierte Volatilitaet, dezimal und nicht-negativ. None bei
        weniger als zwei Beobachtungen.
    """
    clean = _as_float_series(returns)
    if len(clean) < MIN_OBS_DISPERSION:
        return None
    std = float(clean.std(ddof=1))
    return _clean(std * math.sqrt(annualization_factor(interval)))


def sharpe(
    returns: pd.Series,
    *,
    interval: Interval = Interval.D1,
    risk_free_rate: float = 0.0,
) -> float | None:
    """Sharpe Ratio, annualisiert.

    Der risikofreie Zins wird geometrisch auf die Periodenlaenge
    heruntergebrochen, damit er bei Wochen- oder Intraday-Bars nicht
    verzerrt.

    Args:
        returns: Periodische Renditen, dezimal.
        interval: Bar-Intervall fuer die Annualisierung.
        risk_free_rate: Risikofreier Zins, annualisiert und dezimal.

    Returns:
        Annualisierte Sharpe Ratio. None bei zu wenigen Beobachtungen oder
        verschwindender Streuung.
    """
    clean = _as_float_series(returns)
    if len(clean) < MIN_OBS_DISPERSION:
        return None
    periods = annualization_factor(interval)
    excess = clean - _periodic_rate(risk_free_rate, periods)
    std = float(excess.std(ddof=1))
    if std < _EPS:
        return None
    return _clean(float(excess.mean()) / std * math.sqrt(periods))


def sortino(
    returns: pd.Series,
    *,
    interval: Interval = Interval.D1,
    risk_free_rate: float = 0.0,
) -> float | None:
    """Sortino Ratio, annualisiert.

    Im Nenner steht die Downside-Deviation: die quadratischen Abweichungen
    werden ueber **alle** Perioden gemittelt, nicht nur ueber die negativen.
    Das ist die uebliche Definition und macht den Wert zwischen Strategien
    mit unterschiedlich vielen Verlusttagen vergleichbar.

    Args:
        returns: Periodische Renditen, dezimal.
        interval: Bar-Intervall fuer die Annualisierung.
        risk_free_rate: Risikofreier Zins, annualisiert und dezimal.

    Returns:
        Annualisierte Sortino Ratio. None wenn keine Abwaertsabweichung
        vorliegt - ohne Verluste ist das Verhaeltnis unbeschraenkt und
        damit keine Kennzahl.
    """
    clean = _as_float_series(returns)
    if len(clean) < MIN_OBS_DISPERSION:
        return None
    periods = annualization_factor(interval)
    excess = clean - _periodic_rate(risk_free_rate, periods)
    downside = excess.clip(upper=0.0)
    dd = math.sqrt(float((downside**2).mean()))
    if dd < _EPS:
        return None
    return _clean(float(excess.mean()) / dd * math.sqrt(periods))


def _periodic_rate(annual_rate: float, periods_per_year: float) -> float:
    """Bricht einen Jahreszins geometrisch auf eine Periode herunter.

    Args:
        annual_rate: Zins, annualisiert und dezimal.
        periods_per_year: Perioden pro Jahr.

    Returns:
        Periodenzins, dezimal. 0.0 wenn der Jahreszins 0.0 ist.
    """
    if abs(annual_rate) < _EPS or periods_per_year <= 0.0:
        return 0.0
    return float((1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0)


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Underwater-Kurve: relativer Abstand zum bisherigen Hoechststand.

    Args:
        equity: Portfoliowert je Bar, Kontowaehrung.

    Returns:
        Drawdown je Bar, dezimal und nicht-positiv (0.0 am Hoechststand).
        Leere Reihe bei leerer Eingabe.
    """
    clean = _as_float_series(equity)
    if clean.empty:
        return pd.Series(dtype=float, index=clean.index[:0])
    peak = clean.cummax()
    out = clean / peak.where(peak.abs() > _EPS) - 1.0
    return out.fillna(0.0).clip(upper=0.0)


def max_drawdown(equity: pd.Series) -> float | None:
    """Groesster relativer Rueckgang vom Hoechststand.

    Args:
        equity: Portfoliowert je Bar, Kontowaehrung.

    Returns:
        Maximaler Drawdown, dezimal und **negativ** (-0.2 = 20 Prozent
        Rueckgang). None bei leerer Reihe.
    """
    dd = drawdown_series(equity)
    if dd.empty:
        return None
    return _clean(float(dd.min()))


def max_drawdown_duration_days(equity: pd.Series) -> int | None:
    """Laengste Underwater-Phase in Kalendertagen.

    Gemessen wird vom Bar nach einem Hoechststand bis zu dem Bar, an dem
    dieser Hoechststand wieder erreicht wird. Eine am Ende der Reihe noch
    offene Phase zaehlt bis zum letzten Bar mit - sie ist ein realer,
    lediglich noch nicht abgeschlossener Rueckgang.

    Args:
        equity: Portfoliowert je Bar, UTC-``DatetimeIndex``.

    Returns:
        Dauer in Kalendertagen, nicht-negativ. None bei leerer Reihe oder
        wenn der Index kein ``DatetimeIndex`` ist.
    """
    clean = _as_float_series(equity)
    if clean.empty or not isinstance(clean.index, pd.DatetimeIndex):
        return None
    underwater = drawdown_series(clean) < 0.0
    if not bool(underwater.any()):
        return 0
    index = clean.index
    longest = pd.Timedelta(0)
    start: pd.Timestamp | None = None
    previous: pd.Timestamp = index[0]
    for timestamp, is_under in zip(index, underwater.to_numpy(), strict=True):
        if is_under and start is None:
            start = previous
        elif not is_under and start is not None:
            longest = max(longest, timestamp - start)
            start = None
        previous = timestamp
    if start is not None:  # Phase am Reihenende noch offen
        longest = max(longest, index[-1] - start)
    return int(longest.days)


def calmar(
    equity: pd.Series,
    *,
    interval: Interval = Interval.D1,
) -> float | None:
    """CAGR geteilt durch den Betrag des maximalen Drawdown.

    Args:
        equity: Portfoliowert je Bar, Kontowaehrung.
        interval: Bar-Intervall fuer die Annualisierung.

    Returns:
        Calmar Ratio. None wenn CAGR nicht berechenbar ist oder gar kein
        Drawdown auftrat - die Division waere unbeschraenkt.
    """
    growth = cagr(equity, interval=interval)
    mdd = max_drawdown(equity)
    if growth is None or mdd is None or abs(mdd) < _EPS:
        return None
    return _clean(growth / abs(mdd))


def value_at_risk(returns: pd.Series, *, confidence: float = VAR_CONFIDENCE) -> float | None:
    """Historischer Value at Risk der periodischen Renditen.

    Nicht annualisiert: der Wert bezieht sich auf **eine** Periode der
    Eingangsreihe. Ermittelt wird das empirische Quantil, es wird keine
    Normalverteilung angenommen.

    Args:
        returns: Periodische Renditen, dezimal.
        confidence: Konfidenzniveau, dezimal (0.95 = 95 Prozent).

    Returns:
        VaR je Periode, dezimal und **negativ**. Ein positives empirisches
        Quantil (nur Gewinne in der Stichprobe) wird auf 0.0 begrenzt, da
        ein negativer Verlust keine Risikoaussage ist. None bei zu wenigen
        Beobachtungen.
    """
    clean = _as_float_series(returns)
    if len(clean) < MIN_OBS_DISPERSION:
        return None
    quantile = float(np.quantile(clean.to_numpy(), 1.0 - confidence))
    return _clean(min(quantile, 0.0))


def conditional_value_at_risk(
    returns: pd.Series, *, confidence: float = VAR_CONFIDENCE
) -> float | None:
    """Erwarteter Verlust in den schlechtesten Perioden (Expected Shortfall).

    Gemittelt wird ueber alle Beobachtungen, die den VaR unterschreiten
    oder treffen. Damit gilt konstruktiv ``cvar <= var``.

    Args:
        returns: Periodische Renditen, dezimal.
        confidence: Konfidenzniveau, dezimal.

    Returns:
        CVaR je Periode, dezimal und **negativ**, kleiner oder gleich dem
        VaR. None bei zu wenigen Beobachtungen.
    """
    clean = _as_float_series(returns)
    if len(clean) < MIN_OBS_DISPERSION:
        return None
    var = value_at_risk(clean, confidence=confidence)
    if var is None:
        return None
    quantile = float(np.quantile(clean.to_numpy(), 1.0 - confidence))
    tail = clean[clean <= quantile]
    if tail.empty:
        return var
    return _clean(min(float(tail.mean()), var))


def _align(returns: pd.Series, benchmark_returns: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Schneidet zwei Renditereihen auf ihre gemeinsamen Zeitstempel.

    Args:
        returns: Renditen der Strategie, dezimal.
        benchmark_returns: Renditen der Benchmark, dezimal.

    Returns:
        Beide Reihen auf identischem Index, ohne fehlende Werte.
    """
    left = _as_float_series(returns)
    right = _as_float_series(benchmark_returns)
    joined = pd.concat([left.rename("a"), right.rename("b")], axis=1, join="inner").dropna()
    return joined["a"], joined["b"]


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float | None:
    """Sensitivitaet der Strategie gegenueber der Benchmark.

    Args:
        returns: Renditen der Strategie, dezimal.
        benchmark_returns: Renditen der Benchmark, dezimal, gleiche Frequenz.

    Returns:
        Beta, dimensionslos. None bei zu wenigen gemeinsamen Beobachtungen
        oder verschwindender Benchmark-Varianz.
    """
    left, right = _align(returns, benchmark_returns)
    if len(left) < MIN_OBS_REGRESSION:
        return None
    variance = float(right.var(ddof=1))
    if variance < _EPS:
        return None
    covariance = float(left.cov(right))
    return _clean(covariance / variance)


def alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    interval: Interval = Interval.D1,
    risk_free_rate: float = 0.0,
) -> float | None:
    """Jensens Alpha, annualisiert.

    Berechnet als Ueberrendite gegenueber der vom CAPM erwarteten Rendite,
    hochskaliert auf ein Jahr.

    Args:
        returns: Renditen der Strategie, dezimal.
        benchmark_returns: Renditen der Benchmark, dezimal.
        interval: Bar-Intervall fuer die Annualisierung.
        risk_free_rate: Risikofreier Zins, annualisiert und dezimal.

    Returns:
        Alpha, annualisiert und dezimal. None wenn Beta nicht berechenbar
        ist.
    """
    left, right = _align(returns, benchmark_returns)
    factor = beta(left, right)
    if factor is None:
        return None
    periods = annualization_factor(interval)
    rf = _periodic_rate(risk_free_rate, periods)
    expected = rf + factor * (float(right.mean()) - rf)
    return _clean((float(left.mean()) - expected) * periods)


def tracking_error(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    interval: Interval = Interval.D1,
) -> float | None:
    """Annualisierte Standardabweichung der Aktivrendite.

    Args:
        returns: Renditen der Strategie, dezimal.
        benchmark_returns: Renditen der Benchmark, dezimal.
        interval: Bar-Intervall fuer die Annualisierung.

    Returns:
        Tracking Error, annualisiert, dezimal und nicht-negativ. None bei
        zu wenigen gemeinsamen Beobachtungen.
    """
    left, right = _align(returns, benchmark_returns)
    if len(left) < MIN_OBS_DISPERSION:
        return None
    active = left - right
    std = float(active.std(ddof=1))
    return _clean(std * math.sqrt(annualization_factor(interval)))


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    interval: Interval = Interval.D1,
) -> float | None:
    """Annualisierte Aktivrendite je Einheit Tracking Error.

    Args:
        returns: Renditen der Strategie, dezimal.
        benchmark_returns: Renditen der Benchmark, dezimal.
        interval: Bar-Intervall fuer die Annualisierung.

    Returns:
        Information Ratio. None bei verschwindendem Tracking Error - dann
        gibt es keine Aktivrendite, die man normieren koennte.
    """
    left, right = _align(returns, benchmark_returns)
    if len(left) < MIN_OBS_DISPERSION:
        return None
    active = left - right
    std = float(active.std(ddof=1))
    if std < _EPS:
        return None
    periods = annualization_factor(interval)
    return _clean(float(active.mean()) / std * math.sqrt(periods))


def trade_statistics(trades: Sequence[Trade]) -> dict[str, float | int | None]:
    """Verdichtet eine Trade-Liste zu Kennzahlen.

    Ausgewertet werden nur **geschlossene** Trades mit bekanntem PnL: eine
    offene Position hat kein Ergebnis, das man in eine Trefferquote
    einrechnen koennte. ``total_fees`` summiert dagegen ueber alle Trades,
    weil Einstiegskosten auch bei offenen Positionen angefallen sind.

    Args:
        trades: Trades des Backtest, offene und geschlossene.

    Returns:
        Dictionary mit ``n_trades``, ``hit_rate``, ``profit_factor``,
        ``avg_win``, ``avg_loss``, ``avg_duration_bars`` und
        ``total_fees``. ``avg_win``, ``avg_loss`` und ``total_fees`` in
        Kontowaehrung, ``hit_rate`` dezimal. Nicht berechenbare Werte sind
        None.
    """
    fees = _clean(float(sum(t.fees_paid for t in trades))) if trades else None
    closed = [t for t in trades if not t.is_open and t.pnl is not None]
    if not closed:
        return {
            "n_trades": 0,
            "hit_rate": None,
            "profit_factor": None,
            "avg_win": None,
            "avg_loss": None,
            "avg_duration_bars": None,
            "total_fees": fees,
        }

    pnls = [float(t.pnl) for t in closed if t.pnl is not None]
    wins = [p for p in pnls if p > 0.0]
    losses = [p for p in pnls if p < 0.0]
    gross_loss = abs(sum(losses))
    durations = [float(t.duration_bars) for t in closed if t.duration_bars is not None]

    return {
        "n_trades": len(closed),
        "hit_rate": _clean(len(wins) / len(closed)),
        # Ohne Verlust-Trade waere der Profit Factor unbeschraenkt, nicht
        # unendlich gross im Sinne einer Kennzahl -> None.
        "profit_factor": _clean(sum(wins) / gross_loss) if gross_loss > _EPS else None,
        "avg_win": _clean(sum(wins) / len(wins)) if wins else None,
        "avg_loss": _clean(sum(losses) / len(losses)) if losses else None,
        "avg_duration_bars": _clean(sum(durations) / len(durations)) if durations else None,
        "total_fees": fees,
    }


def exposure_from_mask(in_market: pd.Series) -> float | None:
    """Anteil der Bars, an denen eine Position offen war.

    Args:
        in_market: Boolesche Reihe je Bar, True = im Markt.

    Returns:
        Exposure, dezimal zwischen 0.0 und 1.0. None bei leerer Reihe.
    """
    if in_market.empty:
        return None
    mask = in_market.astype(bool)
    return _clean(float(mask.sum()) / float(len(mask)))


def compute_metrics(
    equity: pd.Series,
    *,
    trades: Sequence[Trade] | None = None,
    benchmark_equity: pd.Series | None = None,
    interval: Interval = Interval.D1,
    risk_free_rate: float = 0.0,
    exposure: float | None = None,
) -> PerformanceMetrics:
    """Berechnet den vollstaendigen Kennzahlensatz einer Equity-Kurve.

    Einstiegspunkt fuer beide Engines. Alle Rueckgabefelder folgen der
    Modul-Konvention: annualisiert und dezimal, Drawdown und VaR negativ,
    nicht berechenbar gleich ``None``.

    Args:
        equity: Portfoliowert je Bar in Kontowaehrung, UTC-``DatetimeIndex``.
        trades: Trade-Liste fuer die Trade-Statistiken. None laesst diese
            Felder offen.
        benchmark_equity: Vergleichsreihe fuer Beta, Alpha, Tracking Error
            und Information Ratio. None laesst diese Felder offen.
        interval: Bar-Intervall. Bestimmt den Annualisierungsfaktor ueber
            ``Interval.periods_per_year``.
        risk_free_rate: Risikofreier Zins, annualisiert und dezimal.
        exposure: Anteil der Zeit im Markt, dezimal. Liefert die Engine, da
            nur sie die Positionsreihe kennt.

    Returns:
        Gefuellte Kennzahlen. Bei einer zu kurzen Equity-Kurve bleiben die
        renditebasierten Felder None, statt Platzhalterwerte zu erfinden.
    """
    returns = to_returns(equity)
    stats = trade_statistics(trades or ())

    bench_returns: pd.Series | None = None
    if benchmark_equity is not None:
        candidate = to_returns(benchmark_equity)
        if not candidate.empty:
            bench_returns = candidate

    return PerformanceMetrics(
        total_return=total_return(equity),
        cagr=cagr(equity, interval=interval),
        volatility=volatility(returns, interval=interval),
        sharpe=sharpe(returns, interval=interval, risk_free_rate=risk_free_rate),
        sortino=sortino(returns, interval=interval, risk_free_rate=risk_free_rate),
        calmar=calmar(equity, interval=interval),
        max_drawdown=max_drawdown(equity),
        max_drawdown_duration_days=max_drawdown_duration_days(equity),
        var_95=value_at_risk(returns),
        cvar_95=conditional_value_at_risk(returns),
        beta=None if bench_returns is None else beta(returns, bench_returns),
        alpha=(
            None
            if bench_returns is None
            else alpha(returns, bench_returns, interval=interval, risk_free_rate=risk_free_rate)
        ),
        tracking_error=(
            None
            if bench_returns is None
            else tracking_error(returns, bench_returns, interval=interval)
        ),
        information_ratio=(
            None
            if bench_returns is None
            else information_ratio(returns, bench_returns, interval=interval)
        ),
        n_trades=None if trades is None else int(stats["n_trades"] or 0),
        hit_rate=_as_opt_float(stats["hit_rate"]),
        profit_factor=_as_opt_float(stats["profit_factor"]),
        avg_win=_as_opt_float(stats["avg_win"]),
        avg_loss=_as_opt_float(stats["avg_loss"]),
        avg_duration_bars=_as_opt_float(stats["avg_duration_bars"]),
        exposure=_clean(exposure),
        total_fees=_as_opt_float(stats["total_fees"]),
    )


def _as_opt_float(value: float | int | None) -> float | None:
    """Normalisiert einen Statistikwert auf ``float | None``.

    Args:
        value: Wert aus :func:`trade_statistics`.

    Returns:
        Der Wert als float, oder None.
    """
    return None if value is None else _clean(float(value))


def currency_of(prices: dict[str, PriceSeries]) -> Currency:
    """Liest die Waehrung, in der ein Backtest rechnet, aus den Kursreihen.

    Der Backtest rechnet **nicht** um: Kurse, Startkapital und Ergebnis
    stehen alle in derselben Einheit wie die Eingangsdaten. Diese Funktion
    sagt nur, welche das ist - damit die Beschriftung stimmt.

    Bei mehreren Titeln in verschiedenen Waehrungen gibt es keine richtige
    Antwort; dann faellt sie auf ``OTHER`` zurueck, statt eine der
    beteiligten Waehrungen zu behaupten.

    Args:
        prices: Kursreihen des Laufs, je Ticker eine.

    Returns:
        Die gemeinsame Waehrung, oder ``Currency.OTHER`` wenn die Reihen
        sich unterscheiden oder keine vorliegt.
    """
    found = {series.currency for series in prices.values()}
    if len(found) == 1:
        return found.pop()
    return Currency.OTHER
