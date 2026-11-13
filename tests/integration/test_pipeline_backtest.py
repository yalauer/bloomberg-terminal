"""Integration: Provider -> Strategie -> Engine -> Metriken -> Viz.

Der komplette Backtest-Pfad. Die Unit-Tests pruefen Strategien ohne Engine
und Metriken ohne Strategie. Hier laeuft alles zusammen, und zwar fuer
jede der vier Referenzstrategien.

Die Schnittstellen, die nur hier auffallen koennen:

* Der ``SignalSet`` einer Strategie muss zur Erwartung der Engine passen
  (Spaltennamen, Index, Dtype).
* Das ``BacktestResult`` der Engine muss zur Erwartung von ``viz`` passen.
* Beide Engines muessen ueber ``backtest/metrics.py`` rechnen, damit ihre
  Zahlen vergleichbar sind.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from btlite.backtest import metrics as metrics_mod
from btlite.backtest.registry import default_registry
from btlite.backtest.vbt_engine import EQUITY_COLUMN, VectorbtEngine
from btlite.data.fixtures import FixtureProvider
from btlite.models.backtest import BacktestRequest, StrategyConfig
from btlite.models.common import Interval
from btlite.viz import backtest as viz_backtest

#: Strategien, die mit einem einzelnen Ticker sinnvoll laufen.
EINZEL_STRATEGIEN = ("sma_cross", "rsi_mean_reversion", "buy_and_hold")

#: Strategien, die ein Mehr-Ticker-Universum brauchen.
PANEL_STRATEGIEN = ("momentum_rotation",)


@pytest.fixture(scope="module")
def provider() -> FixtureProvider:
    """Offline-Provider fuer die gesamte Datei."""
    return FixtureProvider()


@pytest.fixture(scope="module")
def engine() -> VectorbtEngine:
    """Die primaere Engine. Ueberspringt alles, wenn das Extra fehlt."""
    instance = VectorbtEngine()
    if not instance.is_available():
        pytest.skip("vectorbt ist nicht installiert - optionales Extra")
    return instance


def _lauf(
    engine: VectorbtEngine,
    provider: FixtureProvider,
    key: str,
    tickers: list[str],
    *,
    benchmark: str | None = None,
):
    """Faehrt einen kompletten Backtest ueber alle Schichten.

    Args:
        engine: Die Backtest-Engine.
        provider: Kursquelle.
        key: Bezeichner der Strategie in der Registry.
        tickers: Zu handelnde Symbole.
        benchmark: Optionales Vergleichssymbol.

    Returns:
        Das vollstaendige Backtest-Ergebnis.
    """
    registry = default_registry()
    strategy_cls = registry.get(key)
    strategy = strategy_cls(StrategyConfig(key=key, params={}))

    prices = provider.get_histories(tickers, interval=Interval.D1)
    signals = strategy.generate_signals(prices)

    request = BacktestRequest(
        tickers=tickers,
        strategy=StrategyConfig(key=key, params={}),
        interval=Interval.D1,
        init_cash=100_000.0,
        fees=0.0005,
        slippage=0.0005,
        benchmark=benchmark,
    )
    bench_series = provider.get_history(benchmark, interval=Interval.D1) if benchmark else None
    return engine.run(request, prices, signals, benchmark=bench_series)


class TestRegistryDecktAlleStrategienAb:
    """Die Registry ist das Bindeglied zwischen UI und Strategien."""

    def test_registry_kennt_die_vier_referenzstrategien(self):
        """Alle vier Strategien sind registriert und auffindbar."""
        registry = default_registry()
        keys = set(registry.keys())

        erwartet = set(EINZEL_STRATEGIEN) | set(PANEL_STRATEGIEN)
        fehlend = erwartet - keys
        assert not fehlend, f"Nicht registrierte Strategien: {fehlend}"


class TestVollstaendigerBacktestPfad:
    """Von der Kursreihe bis zur fertigen Kennzahl."""

    @pytest.mark.parametrize("key", EINZEL_STRATEGIEN)
    def test_einzelticker_strategien_laufen_durch(self, engine, provider, key):
        """Jede Einzelticker-Strategie liefert ein vollstaendiges Ergebnis."""
        result = _lauf(engine, provider, key, ["AAPL"])

        assert not result.equity_curve.empty
        assert EQUITY_COLUMN in result.equity_curve.columns
        assert (result.equity_curve[EQUITY_COLUMN] > 0).all()
        assert result.engine_used == engine.name
        assert result.metrics.total_return is not None
        assert result.metrics.max_drawdown is not None
        assert result.metrics.max_drawdown <= 0.0

    @pytest.mark.parametrize("key", PANEL_STRATEGIEN)
    def test_panel_strategien_laufen_durch(self, engine, provider, key):
        """Strategien ueber mehrere Ticker liefern ein Ergebnis."""
        result = _lauf(engine, provider, key, ["AAPL", "MSFT", "KO", "XOM", "JNJ"])

        assert not result.equity_curve.empty
        assert result.metrics.total_return is not None

    def test_signale_passen_zu_den_kursen(self, provider):
        """Der SignalSet traegt genau die Spalten der Kursreihen.

        Passte das nicht, wuerde die Engine still auf einer Teilmenge
        handeln - ein Fehler, der im Ergebnis kaum auffaellt.
        """
        registry = default_registry()
        strategy = registry.get("sma_cross")(StrategyConfig(key="sma_cross", params={}))

        tickers = ["AAPL", "MSFT"]
        prices = provider.get_histories(tickers, interval=Interval.D1)
        signals = strategy.generate_signals(prices)

        assert set(signals.entries.columns) == set(tickers)
        assert set(signals.exits.columns) == set(tickers)
        assert signals.entries.index.tz is not None
        assert signals.entries.dtypes.eq(bool).all(), "Signale sind nicht boolesch"

    def test_benchmark_fuellt_die_relativen_kennzahlen(self, engine, provider):
        """Mit Benchmark entstehen Beta, Alpha und Tracking Error.

        Ohne Benchmark muessen sie None bleiben - das ist die Gegenprobe.
        """
        mit = _lauf(engine, provider, "sma_cross", ["AAPL"], benchmark="SPY")
        ohne = _lauf(engine, provider, "sma_cross", ["AAPL"], benchmark=None)

        assert mit.metrics.beta is not None
        assert mit.metrics.alpha is not None
        assert mit.benchmark_metrics is not None

        assert ohne.metrics.beta is None, "Beta ohne Benchmark ist nicht definiert"
        assert ohne.metrics.alpha is None

    def test_engine_rechnet_ueber_das_metrik_modul(self, engine, provider):
        """Die Engine erfindet keine eigenen Kennzahlen.

        Beide Engines sollen dieselbe Quelle nutzen, sonst sind ihre
        Ergebnisse nicht vergleichbar.
        """
        result = _lauf(engine, provider, "sma_cross", ["AAPL"])
        equity = result.equity_curve[EQUITY_COLUMN]

        nachgerechnet = metrics_mod.compute_metrics(
            equity,
            trades=result.trades,
            interval=Interval.D1,
            risk_free_rate=0.0,
        )

        assert result.metrics.total_return == pytest.approx(nachgerechnet.total_return, rel=1e-9)
        assert result.metrics.max_drawdown == pytest.approx(nachgerechnet.max_drawdown, rel=1e-9)
        assert result.metrics.cagr == pytest.approx(nachgerechnet.cagr, rel=1e-9)

    def test_buy_and_hold_folgt_dem_kurs(self, engine, provider):
        """Buy-and-Hold muss ungefaehr die Kursentwicklung abbilden.

        Der schaerfste Plausibilitaetstest des gesamten Pfads: bei dieser
        Strategie ist das erwartete Ergebnis von aussen bekannt. Weicht die
        Equity stark vom Kurs ab, stimmt etwas in der Ausfuehrung nicht.
        """
        result = _lauf(engine, provider, "buy_and_hold", ["AAPL"])

        kurs = provider.get_history("AAPL", interval=Interval.D1).data["close"]
        kurs_rendite = float(kurs.iloc[-1] / kurs.iloc[0] - 1.0)
        strategie_rendite = result.metrics.total_return

        assert strategie_rendite is not None
        # Kosten, Slippage und der Ausfuehrungsversatz erklaeren die Luecke.
        assert strategie_rendite == pytest.approx(kurs_rendite, rel=0.10)


class TestErgebnisBisFigure:
    """Das Backtest-Ergebnis geht direkt in die Visualisierung."""

    @pytest.fixture
    def result(self, engine, provider):
        """Ein Ergebnis mit Benchmark, fuer alle Chart-Tests."""
        return _lauf(engine, provider, "sma_cross", ["AAPL"], benchmark="SPY")

    def test_equity_chart(self, result):
        """Die Equity-Kurve laesst sich mit Drawdown-Subplot zeichnen."""
        figure = viz_backtest.equity_curve(result, show_drawdown=True)

        assert isinstance(figure, go.Figure)
        assert figure.data

    def test_equity_chart_mit_trade_markern(self, result):
        """Die Trade-Liste laesst sich auf die Kurve legen."""
        figure = viz_backtest.equity_curve(result, show_trades=True)

        assert isinstance(figure, go.Figure)
        assert figure.data

    def test_monatsrenditen_heatmap(self, result):
        """Aus der Equity-Kurve entstehen Monatsrenditen und daraus die Heatmap.

        Die Verdichtung auf Monatsrenditen liegt zwischen Engine und Chart
        und wird sonst von keinem Test beruehrt.
        """
        equity = result.equity_curve[EQUITY_COLUMN]
        monatsrenditen = equity.resample("ME").last().pct_change().dropna()

        assert not monatsrenditen.empty, "Fuenf Jahre muessen Monatsrenditen hergeben"

        figure = viz_backtest.monthly_returns_heatmap(monatsrenditen)

        assert isinstance(figure, go.Figure)
        assert figure.data
