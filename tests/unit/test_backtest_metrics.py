"""Tests der engine-unabhaengigen Kennzahlen.

Kern dieser Datei sind **handgerechnete Referenzwerte**: jede Erwartung
unten ist entweder analytisch geschlossen (konstante Rendite, perfekte
Korrelation) oder von Hand aus einer kurzen, exakt darstellbaren Reihe
nachgerechnet. Es wird bewusst nichts gegen die Implementierung selbst
verglichen - sonst wuerde der Test nur bestaetigen, was der Code ohnehin
tut.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from btlite.backtest import metrics as m
from btlite.models.backtest import Trade, TradeSide
from btlite.models.common import Currency, DataSource, Interval, Provenance
from btlite.models.market import PriceSeries

# --------------------------------------------------------------------------
# Referenzfaelle - Konstruktion
# --------------------------------------------------------------------------

#: Konstante Bar-Rendite des Referenzfalls A, dezimal.
CONST_RETURN = 0.0004

#: Bars des Referenzfalls A: 253 Punkte = 252 Perioden = exakt ein Jahr.
CONST_BARS = 253


def _index(n: int, start: str = "2024-01-02") -> pd.DatetimeIndex:
    """Baut einen UTC-Index aus n Geschaeftstagen."""
    return pd.date_range(start, periods=n, freq="B", tz="UTC", name="date")


@pytest.fixture
def constant_equity() -> pd.Series:
    """Referenzfall A: konstante Rendite von 4 bp ueber genau ein Jahr.

    Analytisch geschlossen: total_return == cagr, Volatilitaet null,
    Drawdown null.
    """
    values = [100.0 * (1.0 + CONST_RETURN) ** i for i in range(CONST_BARS)]
    return pd.Series(values, index=_index(CONST_BARS))


@pytest.fixture
def stepped_equity() -> pd.Series:
    """Referenzfall B: fuenf Punkte mit exakt darstellbaren Verhaeltnissen.

    Renditen: +10 %, -10 %, +10 %, +1/11. Alle Streuungsmasse sind von Hand
    nachrechenbar, siehe die Konstanten in den Tests.
    """
    return pd.Series([100.0, 110.0, 99.0, 108.9, 118.8], index=_index(5))


# --------------------------------------------------------------------------
# Referenzfall A: konstante Rendite
# --------------------------------------------------------------------------


def test_total_return_matches_hand_calculation(constant_equity):
    # (1.0004 ** 252) - 1, von Hand gerechnet.
    expected = (1.0 + CONST_RETURN) ** (CONST_BARS - 1) - 1.0
    assert m.total_return(constant_equity) == pytest.approx(expected, rel=1e-12)
    assert m.total_return(constant_equity) == pytest.approx(0.10603311665109372, rel=1e-12)


def test_cagr_equals_total_return_over_exactly_one_year(constant_equity):
    # 252 Perioden bei 252 Perioden pro Jahr = genau ein Jahr Laufzeit,
    # deshalb muessen Gesamt- und Jahresrendite identisch sein.
    assert m.cagr(constant_equity, interval=Interval.D1) == pytest.approx(
        m.total_return(constant_equity), rel=1e-12
    )
    assert m.cagr(constant_equity, interval=Interval.D1) == pytest.approx(
        0.10603311665109372, rel=1e-12
    )


def test_constant_return_has_no_volatility_and_no_sharpe(constant_equity):
    returns = m.to_returns(constant_equity)
    # Streuung exakt null -> Volatilitaet 0.0, Sharpe nicht definierbar.
    assert m.volatility(returns, interval=Interval.D1) == pytest.approx(0.0, abs=1e-12)
    assert m.sharpe(returns, interval=Interval.D1) is None
    assert m.sortino(returns, interval=Interval.D1) is None


def test_monotone_curve_has_no_drawdown(constant_equity):
    assert m.max_drawdown(constant_equity) == pytest.approx(0.0, abs=1e-15)
    assert m.max_drawdown_duration_days(constant_equity) == 0
    # Ohne Drawdown ist Calmar unbeschraenkt -> None, nicht inf.
    assert m.calmar(constant_equity, interval=Interval.D1) is None


# --------------------------------------------------------------------------
# Referenzfall B: gestufte Kurve, alle Streuungsmasse von Hand
# --------------------------------------------------------------------------


def test_stepped_returns_are_exact(stepped_equity):
    returns = m.to_returns(stepped_equity)
    assert list(returns.round(10)) == [0.1, -0.1, 0.1, round(1 / 11, 10)]


def test_volatility_matches_hand_calculation(stepped_equity):
    # mean = 0.04772727272727273, std(ddof=1) = 0.09857804448250378
    # vol = std * sqrt(252)
    returns = m.to_returns(stepped_equity)
    assert m.volatility(returns, interval=Interval.D1) == pytest.approx(
        1.5648779425906074, rel=1e-12
    )


def test_sharpe_matches_hand_calculation(stepped_equity):
    # sharpe = mean / std * sqrt(252) = 0.04772727... / 0.09857804... * 15.8745
    returns = m.to_returns(stepped_equity)
    assert m.sharpe(returns, interval=Interval.D1) == pytest.approx(7.685757719456348, rel=1e-10)


def test_sortino_matches_hand_calculation(stepped_equity):
    # Downside-Deviation: sqrt(mean(min(r,0)^2)) = sqrt(0.01/4) = 0.05
    # sortino = 0.04772727... / 0.05 * sqrt(252)
    returns = m.to_returns(stepped_equity)
    assert m.sortino(returns, interval=Interval.D1) == pytest.approx(15.152939327006292, rel=1e-10)


def test_max_drawdown_matches_hand_calculation(stepped_equity):
    # Hoechststand 110.0, Tief 99.0 -> 99/110 - 1 = -0.1
    assert m.max_drawdown(stepped_equity) == pytest.approx(-0.1, rel=1e-12)


def test_total_return_of_stepped_curve(stepped_equity):
    # 118.8 / 100.0 - 1
    assert m.total_return(stepped_equity) == pytest.approx(0.188, rel=1e-12)


def test_calmar_matches_hand_calculation(stepped_equity):
    growth = m.cagr(stepped_equity, interval=Interval.D1)
    assert growth is not None
    assert m.calmar(stepped_equity, interval=Interval.D1) == pytest.approx(growth / 0.1, rel=1e-10)


# --------------------------------------------------------------------------
# Annualisierung: nicht hartcodiert 252
# --------------------------------------------------------------------------


def test_annualization_uses_interval_not_hardcoded_252():
    assert m.annualization_factor(Interval.D1) == 252.0
    assert m.annualization_factor(Interval.W1) == 52.0
    assert m.annualization_factor(Interval.MO1) == 12.0
    assert m.annualization_factor(Interval.H1) == pytest.approx(252 * 6.5)


def test_volatility_scales_with_interval(stepped_equity):
    returns = m.to_returns(stepped_equity)
    daily = m.volatility(returns, interval=Interval.D1)
    weekly = m.volatility(returns, interval=Interval.W1)
    assert daily is not None
    assert weekly is not None
    # Verhaeltnis ist exakt sqrt(252/52).
    assert daily / weekly == pytest.approx(math.sqrt(252 / 52), rel=1e-12)


def test_cagr_uses_interval_for_year_length():
    # 53 Wochenbars = 52 Perioden = genau ein Jahr bei W1.
    values = [100.0 * 1.001**i for i in range(53)]
    equity = pd.Series(values, index=pd.date_range("2024-01-05", periods=53, freq="W", tz="UTC"))
    assert m.cagr(equity, interval=Interval.W1) == pytest.approx(m.total_return(equity), rel=1e-12)


# --------------------------------------------------------------------------
# Drawdown-Dauer
# --------------------------------------------------------------------------


def test_drawdown_duration_counts_calendar_days():
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-11", "2024-01-21", "2024-01-31"], tz="UTC")
    # Hoch am 01.01., dann unter Wasser bis zur Rueckkehr am 31.01.
    equity = pd.Series([100.0, 90.0, 95.0, 100.0], index=idx)
    assert m.max_drawdown_duration_days(equity) == 30


def test_drawdown_duration_includes_open_phase_at_series_end():
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-11", "2024-01-21"], tz="UTC")
    # Nie erholt: die Phase laeuft bis zum letzten Bar.
    equity = pd.Series([100.0, 90.0, 85.0], index=idx)
    assert m.max_drawdown_duration_days(equity) == 20


def test_drawdown_series_is_non_positive_and_zero_at_peaks(stepped_equity):
    dd = m.drawdown_series(stepped_equity)
    assert (dd <= 0.0).all()
    assert dd.iloc[0] == pytest.approx(0.0)
    assert dd.iloc[-1] == pytest.approx(0.0)  # 118.8 ist neuer Hoechststand


# --------------------------------------------------------------------------
# VaR und CVaR
# --------------------------------------------------------------------------


def test_var_and_cvar_match_hand_calculation():
    # 20 Renditen von -0.05 bis +0.14 in 1-Prozent-Schritten.
    # numpy-Quantil bei 5 %: -0.0405. Einzige Beobachtung darunter: -0.05.
    values = [round(-0.05 + 0.01 * i, 4) for i in range(20)]
    returns = pd.Series(values, index=_index(20))
    assert m.value_at_risk(returns) == pytest.approx(-0.0405, rel=1e-10)
    assert m.conditional_value_at_risk(returns) == pytest.approx(-0.05, rel=1e-10)


def test_cvar_never_exceeds_var():
    rng = np.random.default_rng(11)
    returns = pd.Series(rng.normal(0.0005, 0.02, 500), index=_index(500))
    var = m.value_at_risk(returns)
    cvar = m.conditional_value_at_risk(returns)
    assert var is not None
    assert cvar is not None
    assert cvar <= var


def test_var_is_clipped_to_zero_when_only_gains():
    returns = pd.Series([0.01, 0.02, 0.03, 0.04], index=_index(4))
    # Ein "positiver Verlust" ist keine Risikoaussage -> auf 0.0 begrenzt.
    assert m.value_at_risk(returns) == 0.0


# --------------------------------------------------------------------------
# Benchmark-Kennzahlen
# --------------------------------------------------------------------------


def test_beta_is_exactly_two_for_doubled_benchmark():
    bench = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02], index=_index(5))
    strat = bench * 2.0
    assert m.beta(strat, bench) == pytest.approx(2.0, rel=1e-12)


def test_alpha_is_zero_when_strategy_is_pure_leverage():
    bench = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02], index=_index(5))
    strat = bench * 2.0
    # CAPM-Erwartung bei rf=0: beta * mean_bench. Genau das liefert 2*bench.
    assert m.alpha(strat, bench, interval=Interval.D1) == pytest.approx(0.0, abs=1e-12)


def test_beta_is_one_against_itself():
    rng = np.random.default_rng(5)
    bench = pd.Series(rng.normal(0.0004, 0.01, 200), index=_index(200))
    assert m.beta(bench, bench) == pytest.approx(1.0, rel=1e-10)


def test_tracking_error_is_zero_against_itself():
    rng = np.random.default_rng(6)
    bench = pd.Series(rng.normal(0.0004, 0.01, 200), index=_index(200))
    assert m.tracking_error(bench, bench, interval=Interval.D1) == pytest.approx(0.0, abs=1e-12)
    # Ohne Aktivrendite ist die Information Ratio nicht definiert.
    assert m.information_ratio(bench, bench, interval=Interval.D1) is None


def test_tracking_error_matches_hand_calculation():
    bench = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02], index=_index(5))
    strat = bench * 2.0
    # Aktivrendite ist exakt bench; std(ddof=1) * sqrt(252).
    assert m.tracking_error(strat, bench, interval=Interval.D1) == pytest.approx(
        0.3291808013842849, rel=1e-10
    )


def test_benchmark_metrics_align_on_common_dates_only():
    strat = pd.Series([0.01, 0.02, 0.03], index=_index(3))
    bench = pd.Series([0.01, 0.02, 0.03, 0.04], index=_index(4))
    # Beta darf nicht an der laengeren Reihe scheitern.
    assert m.beta(strat, bench) is not None


# --------------------------------------------------------------------------
# Trade-Statistiken
# --------------------------------------------------------------------------


def _trade(
    pnl: float | None,
    *,
    ticker: str = "AAA",
    fees: float = 1.0,
    duration: int | None = 5,
    is_open: bool = False,
) -> Trade:
    """Baut einen Trade fuer die Statistik-Tests."""
    return Trade(
        ticker=ticker,
        side=TradeSide.LONG,
        entry_time=datetime(2024, 1, 2, tzinfo=UTC),
        entry_price=100.0,
        exit_time=None if is_open else datetime(2024, 1, 9, tzinfo=UTC),
        exit_price=None if is_open else 110.0,
        size=10.0,
        pnl=None if is_open else pnl,
        fees_paid=fees,
        duration_bars=None if is_open else duration,
    )


def test_trade_statistics_match_hand_calculation():
    trades = [_trade(100.0), _trade(50.0), _trade(-30.0), _trade(-20.0)]
    stats = m.trade_statistics(trades)
    assert stats["n_trades"] == 4
    assert stats["hit_rate"] == pytest.approx(0.5)
    # Gewinne 150, Verluste 50 -> Profit Factor 3.0
    assert stats["profit_factor"] == pytest.approx(3.0)
    assert stats["avg_win"] == pytest.approx(75.0)
    assert stats["avg_loss"] == pytest.approx(-25.0)
    assert stats["avg_duration_bars"] == pytest.approx(5.0)
    assert stats["total_fees"] == pytest.approx(4.0)


def test_profit_factor_is_none_without_losing_trades():
    stats = m.trade_statistics([_trade(100.0), _trade(50.0)])
    # Unbeschraenkt ist keine Kennzahl -> None statt inf.
    assert stats["profit_factor"] is None
    assert stats["hit_rate"] == pytest.approx(1.0)
    assert stats["avg_loss"] is None


def test_open_trades_are_excluded_from_results_but_not_from_fees():
    trades = [_trade(100.0, fees=2.0), _trade(None, fees=3.0, is_open=True)]
    stats = m.trade_statistics(trades)
    # Eine offene Position hat kein Ergebnis, aber Einstiegskosten.
    assert stats["n_trades"] == 1
    assert stats["hit_rate"] == pytest.approx(1.0)
    assert stats["total_fees"] == pytest.approx(5.0)


def test_trade_statistics_on_empty_list():
    stats = m.trade_statistics([])
    assert stats["n_trades"] == 0
    assert stats["hit_rate"] is None
    assert stats["profit_factor"] is None
    assert stats["total_fees"] is None


def test_exposure_from_mask():
    mask = pd.Series([True, True, False, False, False], index=_index(5))
    assert m.exposure_from_mask(mask) == pytest.approx(0.4)
    assert m.exposure_from_mask(pd.Series([], dtype=bool)) is None


# --------------------------------------------------------------------------
# compute_metrics: Gesamtvertrag
# --------------------------------------------------------------------------


def test_compute_metrics_fills_all_fields(stepped_equity):
    bench = stepped_equity * 0.9 + 10.0
    result = m.compute_metrics(
        stepped_equity,
        trades=[_trade(100.0), _trade(-40.0)],
        benchmark_equity=bench,
        interval=Interval.D1,
        exposure=0.75,
    )
    assert result.total_return == pytest.approx(0.188, rel=1e-12)
    assert result.max_drawdown == pytest.approx(-0.1, rel=1e-12)
    assert result.n_trades == 2
    assert result.hit_rate == pytest.approx(0.5)
    assert result.exposure == pytest.approx(0.75)
    assert result.beta is not None
    assert result.tracking_error is not None


def test_compute_metrics_leaves_benchmark_fields_open_without_benchmark(stepped_equity):
    result = m.compute_metrics(stepped_equity, interval=Interval.D1)
    assert result.beta is None
    assert result.alpha is None
    assert result.tracking_error is None
    assert result.information_ratio is None


def test_compute_metrics_on_too_short_series():
    equity = pd.Series([100.0], index=_index(1))
    result = m.compute_metrics(equity, interval=Interval.D1)
    # Nicht berechenbar heisst None - nie 0.0 als Platzhalter.
    assert result.total_return is None
    assert result.cagr is None
    assert result.volatility is None
    assert result.sharpe is None


def test_compute_metrics_never_returns_nan_or_inf():
    # Kurve mit Totalverlust: viele Kennzahlen sind hier nicht definiert.
    equity = pd.Series([100.0, 50.0, 10.0, 0.0, 0.0], index=_index(5))
    result = m.compute_metrics(equity, interval=Interval.D1)
    for name, value in result.model_dump().items():
        if isinstance(value, float):
            assert math.isfinite(value), f"{name} ist {value}"


def test_metrics_reject_non_positive_equity_silently():
    # Nicht-positive Werte werden verworfen, nicht in inf-Renditen gedreht.
    equity = pd.Series([100.0, 0.0, 110.0, 120.0], index=_index(4))
    returns = m.to_returns(equity)
    assert np.isfinite(returns.to_numpy()).all()


# --------------------------------------------------------------------------
# Property-Tests der Metrik-Mathematik
# --------------------------------------------------------------------------

_finite_returns = st.lists(
    st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False),
    min_size=3,
    max_size=60,
)


@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(values=_finite_returns)
def test_property_volatility_is_non_negative(values):
    returns = pd.Series(values, index=_index(len(values)))
    vol = m.volatility(returns, interval=Interval.D1)
    assert vol is None or vol >= 0.0


@settings(max_examples=200)
@given(values=_finite_returns)
def test_property_cvar_never_exceeds_var(values):
    returns = pd.Series(values, index=_index(len(values)))
    var = m.value_at_risk(returns)
    cvar = m.conditional_value_at_risk(returns)
    if var is not None and cvar is not None:
        assert cvar <= var + 1e-12


@settings(max_examples=200)
@given(values=_finite_returns)
def test_property_var_and_cvar_are_non_positive(values):
    returns = pd.Series(values, index=_index(len(values)))
    for value in (m.value_at_risk(returns), m.conditional_value_at_risk(returns)):
        assert value is None or value <= 0.0


@settings(max_examples=200)
@given(
    values=st.lists(
        st.floats(min_value=1.0, max_value=1e6, allow_nan=False), min_size=3, max_size=60
    )
)
def test_property_max_drawdown_is_non_positive_and_bounded(values):
    equity = pd.Series(values, index=_index(len(values)))
    mdd = m.max_drawdown(equity)
    assert mdd is not None
    # Ein Drawdown kann nie schlimmer als -100 Prozent sein.
    assert -1.0 <= mdd <= 0.0


@settings(max_examples=200)
@given(
    values=st.lists(
        st.floats(min_value=1.0, max_value=1e6, allow_nan=False), min_size=3, max_size=60
    )
)
def test_property_drawdown_series_matches_its_minimum(values):
    equity = pd.Series(values, index=_index(len(values)))
    assert m.max_drawdown(equity) == pytest.approx(float(m.drawdown_series(equity).min()))


@settings(max_examples=200)
@given(
    values=st.lists(
        st.floats(min_value=1.0, max_value=1e6, allow_nan=False), min_size=3, max_size=60
    ),
    factor=st.floats(min_value=0.1, max_value=100.0, allow_nan=False),
)
def test_property_returns_are_scale_invariant(values, factor):
    equity = pd.Series(values, index=_index(len(values)))
    scaled = equity * factor
    # Eine andere Kontogroesse aendert die Renditen nicht.
    assert m.total_return(scaled) == pytest.approx(m.total_return(equity), rel=1e-9)
    assert m.max_drawdown(scaled) == pytest.approx(m.max_drawdown(equity), rel=1e-9)


@settings(max_examples=200)
@given(values=_finite_returns, scale=st.floats(min_value=0.1, max_value=10.0))
def test_property_sharpe_scales_inversely_with_volatility(values, scale):
    returns = pd.Series(values, index=_index(len(values)))
    base = m.sharpe(returns, interval=Interval.D1)
    assume(base is not None)
    scaled = m.sharpe(returns * scale, interval=Interval.D1)
    assume(scaled is not None)
    # Ein konstanter Hebel laesst die Sharpe Ratio unveraendert.
    assert scaled == pytest.approx(base, rel=1e-6)


@settings(max_examples=200)
@given(values=_finite_returns)
def test_property_all_metrics_are_finite_or_none(values):
    returns = pd.Series(values, index=_index(len(values)))
    equity = (1.0 + returns).cumprod() * 100.0
    result = m.compute_metrics(equity, interval=Interval.D1)
    for name, value in result.model_dump().items():
        if isinstance(value, float):
            assert math.isfinite(value), f"{name} ist {value}"


class TestCurrencyOf:
    """Der Backtest sagt, worin er rechnet - er rechnet nicht um.

    Vorher setzte keine Engine die Waehrung, das Ergebnis erbte den
    Modell-Default. Solange der USD war, stimmte die Beschriftung bei
    US-Titeln zufaellig. Seit Euro die Vorgabe ist, waere ein Lauf auf
    AAPL mit "EUR" beschriftet, obwohl durchgehend in Dollar gerechnet
    wird - eine Falschaussage ueber das Ergebnis.
    """

    @staticmethod
    def _series(currency: Currency) -> PriceSeries:
        index = pd.date_range("2024-01-01", periods=3, freq="D", tz=UTC)
        frame = pd.DataFrame(
            {
                "open": [1.0, 1.0, 1.0],
                "high": [1.0, 1.0, 1.0],
                "low": [1.0, 1.0, 1.0],
                "close": [1.0, 1.0, 1.0],
                "volume": [1, 1, 1],
            },
            index=index,
        )
        return PriceSeries(
            ticker="X",
            interval=Interval.D1,
            data=frame,
            currency=currency,
            adjusted=True,
            provenance=Provenance(
                source=DataSource.FIXTURE,
                fetched_at=datetime(2026, 9, 20, tzinfo=UTC),
                delay_minutes=15,
            ),
        )

    def test_a_single_currency_is_taken(self):
        prices = {"AAPL": self._series(Currency.USD)}
        assert m.currency_of(prices) is Currency.USD

    def test_the_default_is_not_assumed(self):
        """Eine Dollar-Reihe ergibt Dollar, auch wenn der Default Euro ist."""
        assert m.currency_of({"AAPL": self._series(Currency.USD)}) is not Currency.EUR

    def test_several_series_in_one_currency(self):
        prices = {"A": self._series(Currency.EUR), "B": self._series(Currency.EUR)}
        assert m.currency_of(prices) is Currency.EUR

    def test_mixed_currencies_are_not_guessed(self):
        """Es gibt keine richtige Antwort - also wird keine behauptet.

        Eine der beteiligten Waehrungen zu nennen, waere schlimmer als
        OTHER: die Summe stuende dann unter einem Etikett, das fuer die
        Haelfte der Titel falsch ist.
        """
        prices = {"A": self._series(Currency.USD), "B": self._series(Currency.EUR)}
        assert m.currency_of(prices) is Currency.OTHER

    def test_no_series_yields_other(self):
        assert m.currency_of({}) is Currency.OTHER
