"""Integration: Provider -> Analytics -> Viz.

Geprueft wird die Kette, nicht die einzelne Funktion: aus echten Fixtures
entstehen Indikatoren, daraus eine Plotly-Figure. Die Unit-Tests pruefen,
ob jede Stufe fuer sich korrekt rechnet. Hier geht es darum, ob die Stufen
zusammenpassen - ob also das, was ``analytics`` ausgibt, genau das ist,
was ``viz`` entgegennimmt.

Architekturregel 3: ``viz`` gibt eine ``Figure`` zurueck und ruft nie
``.show()`` auf.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from btlite.analytics import risk, series, technicals
from btlite.data.fixtures import FixtureProvider
from btlite.models.common import Interval
from btlite.viz import price as viz_price
from btlite.viz import risk as viz_risk


@pytest.fixture(scope="module")
def provider() -> FixtureProvider:
    """Offline-Provider fuer die gesamte Datei."""
    return FixtureProvider()


class TestPreisketteBisFigure:
    """Kursreihe, Indikatoren und Chart in einem Durchlauf."""

    def test_kerzenchart_mit_allen_overlays(self, provider):
        """Die volle Kette: Historie, Overlays, Kerzenchart.

        ``build_overlays`` ist die Schnittstelle zwischen den beiden
        Schichten. Passt ihr Ergebnis nicht zu dem, was ``candlestick``
        erwartet, faellt es genau hier auf.
        """
        history = provider.get_history("AAPL", interval=Interval.D1)
        overlays = technicals.build_overlays(
            history,
            sma_windows=(20, 50),
            ema_windows=(12,),
            with_rsi=True,
            with_macd=True,
            with_bollinger=True,
        )
        figure = viz_price.candlestick(history, overlays)

        assert isinstance(figure, go.Figure)
        assert figure.data, "Die Figure hat keine Traces"

        namen = {str(trace.name) for trace in figure.data if trace.name}
        assert any("SMA" in n for n in namen), f"Kein SMA-Trace: {namen}"
        assert any("RSI" in n.upper() for n in namen), f"Kein RSI-Trace: {namen}"

    def test_indikatoren_teilen_den_index_der_kursreihe(self, provider):
        """Overlays liegen auf demselben Zeitraster wie die Kurse.

        Waere das nicht so, zeichnete Plotly die Linien gegeneinander
        verschoben - ein Fehler, den man im Bild kaum sieht.
        """
        history = provider.get_history("AAPL", interval=Interval.D1)
        sma = technicals.sma(history, 50)

        assert sma.index.equals(history.data.index)
        assert sma.index.tz is not None, "Der Indikator ist nicht tz-aware"

    def test_vergleichschart_aus_mehreren_reihen(self, provider):
        """Mehrere Ticker werden ausgerichtet, normiert und gezeichnet."""
        histories = provider.get_histories(["AAPL", "MSFT", "KO"], interval=Interval.D1)
        comparison = series.build_comparison(histories, normalized=True)
        figure = viz_price.comparison(comparison)

        assert isinstance(figure, go.Figure)
        assert len(figure.data) == 3, "Je Ticker eine Linie erwartet"

        # Normiert heisst: alle Linien starten bei 100.
        for trace in figure.data:
            assert float(trace.y[0]) == pytest.approx(series.COMPARISON_BASE, abs=1e-6)

    @pytest.mark.parametrize("kind", ["candlestick", "ohlc", "line"])
    def test_jede_chart_art(self, provider, kind):
        """Alle drei Kursdarstellungen vertragen echte Fixture-Daten."""
        history = provider.get_history("MSFT", interval=Interval.D1)
        factory = {
            "candlestick": viz_price.candlestick,
            "ohlc": viz_price.ohlc,
            "line": viz_price.line,
        }[kind]

        figure = factory(history)

        assert isinstance(figure, go.Figure)
        assert figure.data


class TestRisikoketteBisFigure:
    """Renditen, Risikokennzahlen und Risiko-Charts in einem Durchlauf."""

    def test_risikokennzahlen_aus_echten_kursen(self, provider):
        """Aus einer Fixture-Kursreihe entsteht ein vollstaendiger Risikosatz."""
        history = provider.get_history("AAPL", interval=Interval.D1)
        benchmark = provider.get_history("SPY", interval=Interval.D1)

        returns = history.data["close"].pct_change().dropna()
        bench_returns = benchmark.data["close"].pct_change().dropna()

        metrics = risk.compute_risk_metrics(returns, bench_returns, interval=Interval.D1)

        assert metrics.volatility is not None
        assert metrics.volatility > 0
        assert metrics.max_drawdown is not None
        assert metrics.max_drawdown <= 0.0
        assert metrics.var_historical is not None
        assert metrics.var_historical <= 0.0
        # Mit Benchmark sind die relativen Masse definiert.
        assert metrics.beta is not None
        assert metrics.r_squared is not None
        assert 0.0 <= metrics.r_squared <= 1.0

    def test_drawdown_chart_aus_echten_renditen(self, provider):
        """Die Drawdown-Reihe der Analytics geht direkt in den Chart."""
        history = provider.get_history("AAPL", interval=Interval.D1)
        returns = history.data["close"].pct_change().dropna()

        drawdown = risk.drawdown_series(returns)
        figure = viz_risk.drawdown_chart(drawdown)

        assert isinstance(figure, go.Figure)
        assert figure.data
        assert (drawdown <= 1e-9).all(), "Ein Drawdown darf nie positiv sein"

    def test_korrelations_heatmap_aus_echten_kursen(self, provider):
        """Aus mehreren Kursreihen entsteht eine zeichenbare Korrelationsmatrix."""
        histories = provider.get_histories(["AAPL", "MSFT", "KO", "XOM"], interval=Interval.D1)
        closes = series.align_closes(histories)
        matrix = closes.pct_change().dropna(how="any").corr()

        figure = viz_risk.correlation_heatmap(matrix)

        assert isinstance(figure, go.Figure)
        assert figure.data
        assert matrix.shape == (4, 4)

    def test_rollierende_volatilitaet_bis_zur_figure(self, provider):
        """Rollierende Kennzahlen behalten ihren Zeitindex bis in den Chart."""
        history = provider.get_history("AAPL", interval=Interval.D1)
        returns = history.data["close"].pct_change().dropna()

        rolling = risk.rolling_volatility(returns, window=63, interval=Interval.D1)
        figure = viz_risk.rolling_volatility(rolling)

        assert isinstance(figure, go.Figure)
        assert figure.data
        verfuegbar = rolling.dropna()
        assert not verfuegbar.empty
        assert (verfuegbar >= 0).all(), "Volatilitaet kann nicht negativ sein"


class TestVizGibtNurFiguresZurueck:
    """Architekturregel 3: viz zeichnet, zeigt aber nie an."""

    def test_keine_show_aufrufe_im_viz_paket(self):
        """Kein Modul unter ``viz/`` ruft ``.show()`` auf.

        Ein ``.show()`` wuerde in einer Testsuite oder im Server einen
        Browser oeffnen - deshalb ist es projektweit verboten.
        """
        from pathlib import Path

        import btlite.viz

        paket = Path(btlite.viz.__file__).parent
        treffer = [
            path.name
            for path in paket.glob("*.py")
            if ".show()" in path.read_text(encoding="utf-8")
        ]
        assert not treffer, f"Verbotener .show()-Aufruf in: {treffer}"
