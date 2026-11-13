"""Integration: Provider -> Optimizer -> Risk -> Attribution -> Viz.

Die komplette Portfolio-Analyse in einem Durchlauf: Kurse holen,
optimieren, das Ergebnis mit den Risikokennzahlen der Analytics-Schicht
gegenrechnen, die Rendite attribuieren, Rebalancing-Vorschlaege ableiten
und alles zeichnen.

Der interessante Teil ist die Konsistenz zwischen den Schichten: die
Gewichte des Optimizers, die Renditen der Analytics und die Beitraege der
Attribution muessen dieselbe Geschichte erzaehlen.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from btlite.analytics import risk, series
from btlite.data.fixtures import FixtureProvider
from btlite.models.common import Interval
from btlite.models.portfolio import (
    WEIGHT_SUM_TOLERANCE,
    OptimizationObjective,
    PortfolioRequest,
)
from btlite.portfolio import attribution, rebalance
from btlite.portfolio.optimizer import (
    asset_scatter,
    build_price_matrix,
    estimate_covariance,
    estimate_returns,
    optimize,
)
from btlite.viz import frontier as viz_frontier
from btlite.viz import risk as viz_risk

#: Das Universum der Portfolio-Tests.
UNIVERSUM = ["AAPL", "MSFT", "KO", "XOM", "JNJ", "PG", "JPM", "WMT"]


@pytest.fixture(scope="module")
def provider() -> FixtureProvider:
    """Offline-Provider fuer die gesamte Datei."""
    return FixtureProvider()


@pytest.fixture(scope="module")
def prices(provider: FixtureProvider):
    """Kursreihen des Universums."""
    return provider.get_histories(UNIVERSUM, interval=Interval.D1)


@pytest.fixture(scope="module")
def result(prices):
    """Eine Max-Sharpe-Optimierung ueber das Universum."""
    request = PortfolioRequest(
        tickers=UNIVERSUM,
        objective=OptimizationObjective.MAX_SHARPE,
        risk_free_rate=0.02,
    )
    return optimize(request, prices)


class TestOptimiererLiefertKonsistentesErgebnis:
    """Das Ergebnis muss in sich stimmig sein, nicht nur vorhanden."""

    def test_alle_titel_haben_ein_gewicht(self, result):
        """Jeder Ticker mit Kursen erscheint als Position."""
        assert {p.ticker for p in result.positions} == set(UNIVERSUM)

    def test_gewichte_summieren_auf_eins(self, result):
        """Die definierende Eigenschaft eines Portfolios."""
        summe = sum(p.weight for p in result.positions)
        assert summe == pytest.approx(1.0, abs=WEIGHT_SUM_TOLERANCE)

    def test_erwartete_kennzahlen_sind_konsistent(self, result):
        """Sharpe ist Rendite minus Zins, geteilt durch Volatilitaet.

        Die drei Zahlen kommen aus derselben Optimierung. Passten sie nicht
        zueinander, waere eine davon aus einem anderen Lauf.
        """
        rendite = result.expected_return
        vola = result.expected_volatility
        sharpe = result.sharpe_ratio
        zins = result.request.risk_free_rate

        assert rendite is not None
        assert vola is not None
        assert vola > 0
        assert sharpe is not None

        assert sharpe == pytest.approx((rendite - zins) / vola, rel=1e-6)

    def test_frontier_umschliesst_das_optimum(self, result):
        """Das optimierte Portfolio liegt nicht oberhalb der Effizienzgrenze.

        Die Grenze ist per Definition der bestmoegliche Rand. Laege das
        Ergebnis darueber, waere eine der beiden Rechnungen falsch.
        """
        assert result.frontier is not None
        data = result.frontier.data

        beste_rendite_auf_grenze = float(data["expected_return"].max())
        assert result.expected_return is not None
        assert result.expected_return <= beste_rendite_auf_grenze + 1e-6


class TestOptimiererGegenAnalyticsRisiko:
    """Die Gewichte des Optimizers gehen zurueck in die Risiko-Schicht."""

    def test_realisiertes_risiko_stimmt_mit_der_analytics_schicht(self, result, prices):
        """Das mitgelieferte Risiko ist das der gewichteten Renditereihe.

        Der Optimizer liefert ``result.risk`` selbst. Hier wird derselbe
        Wert unabhaengig ueber ``analytics/series`` und ``analytics/risk``
        nachgerechnet - die beiden Wege muessen sich treffen.
        """
        assert result.risk is not None

        gewichte = {p.ticker: p.weight for p in result.positions}
        portfolio_renditen = series.portfolio_returns(gewichte, prices, rebalanced=True)
        nachgerechnet = risk.compute_risk_metrics(portfolio_renditen, interval=Interval.D1)

        assert result.risk.volatility is not None
        assert nachgerechnet.volatility is not None
        assert result.risk.volatility == pytest.approx(nachgerechnet.volatility, rel=1e-6)
        assert result.risk.max_drawdown == pytest.approx(nachgerechnet.max_drawdown, rel=1e-6)

    def test_portfolio_ist_weniger_volatil_als_der_schlechteste_einzelwert(self, result, prices):
        """Diversifikation wirkt - das ist der Sinn der Uebung.

        Ein Portfolio aus acht Titeln muss schwankungsaermer sein als der
        volatilste Einzelwert darin. Sonst rechnet die Gewichtung nicht.
        """
        gewichte = {p.ticker: p.weight for p in result.positions}
        portfolio_renditen = series.portfolio_returns(gewichte, prices)
        portfolio_vola = risk.volatility(portfolio_renditen, interval=Interval.D1)

        einzel_volas = [
            risk.volatility(p.data["close"].pct_change().dropna(), interval=Interval.D1)
            for p in prices.values()
        ]
        hoechste = max(v for v in einzel_volas if v is not None)

        assert portfolio_vola is not None
        assert portfolio_vola < hoechste

    def test_benchmark_relative_kennzahlen(self, result, prices, provider):
        """Gegen SPY entstehen Beta und Tracking Error."""
        gewichte = {p.ticker: p.weight for p in result.positions}
        portfolio_renditen = series.portfolio_returns(gewichte, prices)

        spy = provider.get_history("SPY", interval=Interval.D1)
        bench_renditen = spy.data["close"].pct_change().dropna()

        metrics = risk.compute_risk_metrics(
            portfolio_renditen, bench_renditen, interval=Interval.D1
        )

        assert metrics.beta is not None
        assert metrics.tracking_error is not None
        assert metrics.tracking_error >= 0.0
        assert metrics.r_squared is not None
        assert 0.0 <= metrics.r_squared <= 1.0


class TestAttributionUndRebalancing:
    """Die Gewichte gehen weiter in Attribution und Order-Vorschlaege."""

    def test_beitraege_summieren_zur_portfoliorendite(self, result, prices):
        """Die Kernidentitaet der Attribution.

        Die Summe der Positionsbeitraege ist die Portfoliorendite. Diese
        Identitaet verbindet Optimizer und Attribution.
        """
        gewichte = {p.ticker: p.weight for p in result.positions}
        table = attribution.attribute(gewichte, prices, provenance=result.provenance)

        summe = float(table.data["contribution"].sum())
        assert table.total_return == pytest.approx(summe, rel=1e-9)

    def test_top_beitraege_sind_sortiert(self, result, prices):
        """Die Rangliste fuer das UI ist absteigend nach Beitrag."""
        gewichte = {p.ticker: p.weight for p in result.positions}
        table = attribution.attribute(gewichte, prices, provenance=result.provenance)

        top = attribution.top_contributors(table, count=3)

        assert len(top) <= 3
        werte = top["contribution"].tolist()
        assert werte == sorted(werte, reverse=True)

    def test_drift_und_order_vorschlaege(self, result, provider):
        """Aus Ist- und Zielgewichten entstehen Order-Vorschlaege.

        Das Projekt fuehrt keine Orders aus - es schlaegt sie nur vor.
        """
        ziel = {p.ticker: p.weight for p in result.positions}
        quotes = provider.get_quotes(UNIVERSUM)
        kurse = {t: q.price for t, q in quotes.items()}

        # Ein bewusst schiefes Ist-Depot: ueberall dieselbe Stueckzahl.
        bestand = dict.fromkeys(UNIVERSUM, 10.0)
        ist = rebalance.current_weights(bestand, kurse)
        wert = rebalance.portfolio_value(bestand, kurse)

        vorschlaege = rebalance.propose_orders(ist, ziel, kurse, total_value=wert)

        assert vorschlaege, "Ein schiefes Depot muss Vorschlaege ergeben"
        for order in vorschlaege:
            assert order.ticker in UNIVERSUM
            assert order.amount > 0, "Das Volumen ist immer positiv"
            assert order.side in {rebalance.OrderSide.BUY, rebalance.OrderSide.SELL}
            # Die Richtung folgt dem Vorzeichen der Abweichung.
            if order.side is rebalance.OrderSide.BUY:
                assert order.drift < 0, "Gekauft wird, was untergewichtet ist"
            else:
                assert order.drift > 0, "Verkauft wird, was uebergewichtet ist"

    def test_ohne_drift_keine_vorschlaege(self, provider):
        """Ein bereits passendes Depot erzeugt keine Orders.

        Die Gegenprobe: schluege die Funktion immer etwas vor, waere der
        Test oben wertlos.
        """
        quotes = provider.get_quotes(UNIVERSUM)
        kurse = {t: q.price for t, q in quotes.items()}
        bestand = dict.fromkeys(UNIVERSUM, 10.0)

        ist = rebalance.current_weights(bestand, kurse)
        wert = rebalance.portfolio_value(bestand, kurse)

        # Ziel gleich Ist: es gibt nichts zu tun.
        vorschlaege = rebalance.propose_orders(ist, ist, kurse, total_value=wert)

        assert not vorschlaege

    def test_aktuelle_gewichte_summieren_auf_eins(self, provider):
        """Aus Bestand und Kursen ergeben sich Ist-Gewichte."""
        quotes = provider.get_quotes(UNIVERSUM)
        kurse = {t: q.price for t, q in quotes.items()}
        bestand = dict.fromkeys(UNIVERSUM, 10.0)

        ist = rebalance.current_weights(bestand, kurse)

        assert sum(ist.values()) == pytest.approx(1.0, abs=1e-9)


class TestPortfolioBisFigure:
    """Das Ergebnis geht in die Visualisierung."""

    def test_effizienzgrenze_mit_einzelwerten(self, result, prices):
        """Grenze und Streupunkte kommen aus demselben Optimierungslauf."""
        matrix = build_price_matrix(prices)
        mu = estimate_returns(
            matrix,
            result.request.returns_method,
            frequency=252,
            risk_free_rate=result.request.risk_free_rate,
        )
        covariance = estimate_covariance(matrix, result.request.covariance_method, frequency=252)
        assets = asset_scatter(mu, covariance)

        assert result.frontier is not None
        figure = viz_frontier.efficient_frontier(result.frontier, assets=assets)

        assert isinstance(figure, go.Figure)
        assert figure.data
        assert set(assets.index) == set(UNIVERSUM)

    def test_gewichts_charts(self, result):
        """Balken und Donut nehmen das Optimierungsergebnis direkt entgegen."""
        balken = viz_frontier.weights_bar(result)
        donut = viz_frontier.weights_donut(result)

        assert isinstance(balken, go.Figure)
        assert isinstance(donut, go.Figure)
        assert balken.data
        assert donut.data

    def test_korrelations_heatmap_aus_dem_ergebnis(self, result):
        """Die Korrelationsmatrix des Ergebnisses ist direkt zeichenbar."""
        assert result.correlation is not None
        figure = viz_risk.correlation_heatmap(result.correlation)

        assert isinstance(figure, go.Figure)
        assert figure.data

    def test_renditeverteilung_des_portfolios(self, result, prices):
        """Die gewichtete Renditereihe geht in das Histogramm."""
        gewichte = {p.ticker: p.weight for p in result.positions}
        renditen = series.portfolio_returns(gewichte, prices)

        figure = viz_risk.return_distribution(renditen)

        assert isinstance(figure, go.Figure)
        assert figure.data
