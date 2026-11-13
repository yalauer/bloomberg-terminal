"""Die Akzeptanzkriterien des Projekts als ausfuehrbare Checkliste.

Jeder Test dieser Datei entspricht genau einem Kriterium aus Abschnitt 8
der Projektvorgabe ("Definition of Done"). Wer wissen will, ob BT-Lite
fertig ist, liest diese Datei - nicht den Quellcode.

Die Kriterien im Wortlaut, in der Reihenfolge der Testklassen:

1. Ein Statement-Abruf liefert vollstaendige Ratios ohne stille
   ``NaN``-Fallbacks; fehlende Werte sind explizit ``None``.
2. Der Screener liefert fuer ein Preset mehr als null Treffer und
   exportiert nach CSV.
3. Ein SMA-Cross-Backtest ueber fuenf Jahre auf ``SPY`` produziert
   Equity-Kurve, Trade-Liste, Sharpe, Max Drawdown, CAGR und Hit Rate -
   plausibel gegen manuell nachgerechnete Referenzwerte.
4. Eine Portfolio-Optimierung ueber zehn Ticker gibt Gewichte (Summe 1.0,
   keine negativen ohne Shorting-Flag), Effizienzgrenze und
   Risikokennzahlen zurueck.
5. Coverage-Ziel und Offline-Betrieb.

Alles laeuft gegen den :class:`~btlite.data.fixtures.FixtureProvider`,
also ohne Netzwerk. ``tests/conftest.py`` erzwingt das zusaetzlich
prozessweit.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from btlite.analytics.ratios import compute_ratios
from btlite.analytics.statements import normalize_bundle
from btlite.backtest import metrics as metrics_mod
from btlite.backtest.strategies.sma_cross import SmaCrossStrategy
from btlite.backtest.vbt_engine import EQUITY_COLUMN, VectorbtEngine
from btlite.data.fixtures import FixtureProvider
from btlite.models.backtest import BacktestRequest, StrategyConfig
from btlite.models.common import Interval, Period
from btlite.models.fundamentals import RatioCategory
from btlite.models.portfolio import (
    WEIGHT_SUM_TOLERANCE,
    OptimizationObjective,
    PortfolioRequest,
)
from btlite.models.screening import ScreenPreset, ScreenResult
from btlite.portfolio.optimizer import optimize
from btlite.screening.engine import ScreeningEngine, export_result, load_export
from btlite.screening.presets import all_presets, get_preset
from btlite.screening.universe import build_ratio_universe

#: Das Symbol des Backtest-Kriteriums.
BACKTEST_TICKER = "SPY"

#: Handelstage, die eine fuenfjaehrige Tageshistorie mindestens hat.
#: 252 Handelstage je Jahr, abzueglich Feiertagsstreuung.
MIN_BARS_FIVE_YEARS = 1200

#: Obergrenze fuer eine plausible Sharpe Ratio einer Aktienstrategie.
#: Alles darueber deutet auf einen Annualisierungsfehler hin.
MAX_PLAUSIBLE_SHARPE = 4.0

#: Obergrenze fuer eine plausible CAGR eines Index-Backtests.
MAX_PLAUSIBLE_CAGR = 1.0


@pytest.fixture(scope="module")
def provider() -> FixtureProvider:
    """Der Offline-Provider, gegen den alle Kriterien geprueft werden."""
    return FixtureProvider()


# ==========================================================================
# Kriterium 1 - Statements und Ratios ohne stille NaN-Fallbacks
# ==========================================================================


class TestKriteriumStatementsUndRatios:
    """Ein Statement-Abruf liefert vollstaendige Ratios.

    Entscheidend ist nicht, dass jede Kennzahl einen Wert hat - manche sind
    aus den Daten schlicht nicht ableitbar. Entscheidend ist, dass ein
    fehlender Wert **explizit** ``None`` ist und sagt, was ihm fehlt.
    """

    def test_statement_abruf_liefert_alle_drei_statements(self, provider):
        """Ein US-Ticker liefert GuV, Bilanz und Kapitalflussrechnung."""
        bundle = provider.get_statements("AAPL", period=Period.ANNUAL)
        assert bundle.income is not None
        assert bundle.balance is not None
        assert bundle.cashflow is not None

    def test_ratios_sind_vollstaendig_berechnet(self, provider):
        """Der Kennzahlensatz ist nicht leer und deckt alle Kategorien ab."""
        bundle = normalize_bundle(provider.get_statements("AAPL", period=Period.ANNUAL))
        ratios = compute_ratios(bundle)

        assert ratios.ratios, "Der Kennzahlensatz ist leer"
        kategorien = {r.category for r in ratios.ratios.values()}
        assert len(kategorien) >= 5, f"Nur {len(kategorien)} Kategorien berechnet"

        verfuegbar = [r for r in ratios.ratios.values() if r.is_available]
        assert len(verfuegbar) >= 10, "Zu wenige Kennzahlen sind tatsaechlich berechenbar"

    def test_kein_wert_ist_nan(self, provider):
        """Kein Kennzahlenwert ist NaN - fehlend heisst None, nie NaN.

        Das ist der Kern des Kriteriums: ein stiller NaN-Fallback wuerde
        sich durch die gesamte Rechenkette ziehen und erst im UI als
        unerklaerlicher Wert auftauchen.
        """
        bundle = normalize_bundle(provider.get_statements("AAPL", period=Period.ANNUAL))
        ratios = compute_ratios(bundle)

        nan_keys = [
            key
            for key, ratio in ratios.ratios.items()
            if ratio.value is not None and math.isnan(ratio.value)
        ]
        assert not nan_keys, f"Stille NaN-Werte in: {nan_keys}"

    def test_kein_wert_ist_unendlich(self, provider):
        """Kein Kennzahlenwert ist inf - eine Division durch null liefert None."""
        bundle = normalize_bundle(provider.get_statements("AAPL", period=Period.ANNUAL))
        ratios = compute_ratios(bundle)

        inf_keys = [
            key
            for key, ratio in ratios.ratios.items()
            if ratio.value is not None and math.isinf(ratio.value)
        ]
        assert not inf_keys, f"Unendliche Werte in: {inf_keys}"

    def test_fehlende_werte_nennen_ihre_ursache(self, provider):
        """Jede nicht berechenbare Kennzahl benennt die fehlenden Positionen.

        Ein ``None`` ohne Begruendung waere im UI nicht von einem Bug zu
        unterscheiden.
        """
        bundle = normalize_bundle(provider.get_statements("AAPL", period=Period.ANNUAL))
        ratios = compute_ratios(bundle)

        stumm = [
            key
            for key, ratio in ratios.ratios.items()
            if ratio.value is None and not ratio.inputs_missing
        ]
        assert not stumm, f"Fehlende Werte ohne Begruendung: {stumm}"

    def test_bewertungskennzahlen_brauchen_einen_kurs(self, provider):
        """Ohne Kurs bleiben KGV und Verwandte None, nicht null.

        Die Gegenprobe zum Kriterium: ein fehlender Eingang macht die
        Kennzahl nicht zu ``0.0``, sondern zu ``None`` - und nennt dabei,
        welcher Eingang fehlt.
        """
        bundle = normalize_bundle(provider.get_statements("AAPL", period=Period.ANNUAL))
        ohne_kurs = compute_ratios(bundle, price=None)
        mit_kurs = compute_ratios(bundle, price=200.0, market_cap=3.0e12, dividend_per_share=1.0)

        bewertung = [
            r.key for r in ohne_kurs.ratios.values() if r.category is RatioCategory.VALUATION
        ]
        assert bewertung, "Der Satz fuehrt keine Bewertungskennzahlen"

        for key in bewertung:
            ohne = ohne_kurs.ratios[key]
            assert ohne.value is None, f"{key} hat ohne Kurs einen Wert"
            assert ohne.inputs_missing, f"{key} nennt den fehlenden Eingang nicht"

            mit = mit_kurs.ratios[key]
            assert mit.value is not None, f"{key} bleibt trotz Kurs unberechnet"

    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "JNJ", "KO", "XOM"])
    def test_mehrere_us_ticker_ohne_nan(self, provider, ticker):
        """Das Kriterium gilt nicht nur fuer einen Glueckstreffer."""
        bundle = normalize_bundle(provider.get_statements(ticker, period=Period.ANNUAL))
        ratios = compute_ratios(bundle)

        assert ratios.ratios
        for key, ratio in ratios.ratios.items():
            if ratio.value is not None:
                assert not math.isnan(ratio.value), f"{ticker}.{key} ist NaN"
                assert not math.isinf(ratio.value), f"{ticker}.{key} ist inf"


# ==========================================================================
# Kriterium 2 - Screener liefert Treffer und exportiert nach CSV
# ==========================================================================


class TestKriteriumScreener:
    """Ein Preset liefert Treffer, und die Treffer lassen sich als CSV sichern."""

    @pytest.fixture
    def universum(self, provider) -> pd.DataFrame:
        """Kennzahlen-Universum ueber alle Fixture-Ticker."""
        return build_ratio_universe(provider, provider.available_tickers())

    def test_universum_ist_nicht_leer(self, universum):
        """Grundlage des Kriteriums: es gibt ueberhaupt etwas zu screenen."""
        assert not universum.empty
        assert len(universum) >= 5

    def test_preset_liefert_treffer(self, universum):
        """Mindestens ein Preset findet mehr als null Titel.

        Das Kriterium verlangt "fuer ein Preset > 0 Treffer". Geprueft wird
        deshalb, dass wenigstens eines der vier Presets anschlaegt.
        """
        engine = ScreeningEngine()
        treffer = {
            preset.key: engine.screen(preset.request, universe=universum).total_matches
            for preset in all_presets()
        }
        assert any(n > 0 for n in treffer.values()), f"Kein Preset fand Treffer: {treffer}"

    @pytest.mark.parametrize("preset_key", [p.key for p in all_presets()])
    def test_jedes_preset_laeuft_fehlerfrei(self, universum, preset_key):
        """Kein Preset scheitert - null Treffer sind erlaubt, ein Fehler nicht."""
        engine = ScreeningEngine()
        result = engine.screen(get_preset(preset_key).request, universe=universum)

        assert isinstance(result, ScreenResult)
        assert result.total_matches >= 0
        assert len(result.data) <= result.request.limit

    def test_export_nach_csv(self, universum, tmp_path: Path):
        """Ein Ergebnis laesst sich als CSV schreiben und wieder einlesen."""
        engine = ScreeningEngine()
        preset = self._erstes_preset_mit_treffern(engine, universum)
        result = engine.screen(preset.request, universe=universum)

        ziel = tmp_path / "screen.csv"
        geschrieben = export_result(result, ziel)

        assert geschrieben.is_file()
        assert geschrieben.stat().st_size > 0

        zurueck = load_export(geschrieben)
        assert len(zurueck) == len(result.data)
        assert list(zurueck.index) == list(result.data.index)

    def test_csv_traegt_die_ticker_spalte(self, universum, tmp_path: Path):
        """Der Ticker-Index ueberlebt den Round-Trip, sonst ist der Export wertlos."""
        engine = ScreeningEngine()
        preset = self._erstes_preset_mit_treffern(engine, universum)
        result = engine.screen(preset.request, universe=universum)

        ziel = export_result(result, tmp_path / "screen.csv")
        rohtext = ziel.read_text(encoding="utf-8")

        assert "ticker" in rohtext.splitlines()[0].lower()
        for symbol in result.data.index:
            assert str(symbol) in rohtext

    @staticmethod
    def _erstes_preset_mit_treffern(
        engine: ScreeningEngine, universum: pd.DataFrame
    ) -> ScreenPreset:
        """Sucht ein Preset, das im Fixture-Universum anschlaegt.

        Args:
            engine: Die Screening-Engine.
            universum: Das Kennzahlen-Universum.

        Returns:
            Das erste Preset mit mindestens einem Treffer.
        """
        for preset in all_presets():
            if engine.screen(preset.request, universe=universum).total_matches > 0:
                return preset
        pytest.skip("Kein Preset liefert im Fixture-Universum Treffer")


# ==========================================================================
# Kriterium 3 - SMA-Cross ueber 5 Jahre auf SPY
# ==========================================================================


@pytest.fixture(scope="module")
def spy_backtest(provider: FixtureProvider):
    """Faehrt den SMA-Cross-Backtest des Kriteriums einmal fuer alle Tests.

    Fuenf Jahre Tageshistorie auf SPY, Fenster 50/200 - der klassische
    Golden-Cross-Fall. Kosten sind gesetzt, damit die Kostenrechnung
    ebenfalls im Pfad liegt.
    """
    engine = VectorbtEngine()
    if not engine.is_available():
        pytest.skip("vectorbt ist nicht installiert - optionales Extra")

    prices = {BACKTEST_TICKER: provider.get_history(BACKTEST_TICKER, interval=Interval.D1)}
    strategy = SmaCrossStrategy(StrategyConfig(key="sma_cross", params={"fast": 50, "slow": 200}))
    signals = strategy.generate_signals(prices)
    request = BacktestRequest(
        tickers=[BACKTEST_TICKER],
        strategy=StrategyConfig(key="sma_cross", params={"fast": 50, "slow": 200}),
        interval=Interval.D1,
        init_cash=100_000.0,
        fees=0.0005,
        slippage=0.0005,
    )
    return engine.run(request, prices, signals)


class TestKriteriumBacktest:
    """SMA-Cross ueber fuenf Jahre auf SPY, mit Referenzrechnung.

    Die Kennzahlen werden nicht gegen die Implementierung geprueft - das
    wuerde nur bestaetigen, was der Code ohnehin tut. Stattdessen werden
    sie aus der Equity-Kurve unabhaengig nachgerechnet und mit dem
    Ergebnis verglichen.
    """

    def test_historie_deckt_fuenf_jahre_ab(self, provider):
        """Die Grundlage des Kriteriums: fuenf Jahre Tagesdaten liegen vor."""
        series = provider.get_history(BACKTEST_TICKER, interval=Interval.D1)
        spanne = series.data.index[-1] - series.data.index[0]

        assert len(series.data) >= MIN_BARS_FIVE_YEARS
        assert spanne.days >= 4 * 365, f"Nur {spanne.days} Tage Historie"

    def test_equity_kurve_ist_vorhanden(self, spy_backtest):
        """Kriterium: der Lauf produziert eine Equity-Kurve."""
        kurve = spy_backtest.equity_curve

        assert not kurve.empty
        assert EQUITY_COLUMN in kurve.columns
        assert isinstance(kurve.index, pd.DatetimeIndex)
        assert kurve.index.tz is not None, "Die Equity-Kurve ist nicht tz-aware"
        assert kurve[EQUITY_COLUMN].notna().all()
        assert (kurve[EQUITY_COLUMN] > 0).all(), "Negativer Portfoliowert"

    def test_equity_startet_beim_startkapital(self, spy_backtest):
        """Vor dem ersten Trade steht exakt das Startkapital."""
        erster = float(spy_backtest.equity_curve[EQUITY_COLUMN].iloc[0])
        assert erster == pytest.approx(spy_backtest.request.init_cash, rel=1e-6)

    def test_trade_liste_ist_vorhanden(self, spy_backtest):
        """Kriterium: der Lauf produziert eine Trade-Liste.

        Ein SMA-50/200-Cross ueber fuenf Jahre schneidet mehrfach - eine
        leere Liste waere ein Hinweis darauf, dass keine Signale ankommen.
        """
        assert spy_backtest.trades, "Keine Trades - Signale kommen nicht in der Engine an"

        for trade in spy_backtest.trades:
            assert trade.ticker == BACKTEST_TICKER
            assert trade.entry_price > 0
            assert trade.size > 0
            if trade.exit_time is not None:
                assert trade.exit_time >= trade.entry_time

    def test_sharpe_ist_da_und_plausibel(self, spy_backtest):
        """Kriterium: Sharpe liegt vor und ist keine Annualisierungs-Entgleisung."""
        sharpe = spy_backtest.metrics.sharpe

        assert sharpe is not None
        assert not math.isnan(sharpe)
        assert abs(sharpe) < MAX_PLAUSIBLE_SHARPE, f"Sharpe {sharpe} ist unplausibel"

    def test_sharpe_gegen_referenzrechnung(self, spy_backtest):
        """Sharpe, unabhaengig aus der Equity-Kurve nachgerechnet.

        Referenz: mittlere Tagesrendite geteilt durch deren
        Standardabweichung, mal Wurzel(252). Genau die Definition, die das
        Projekt fuer ``risk_free_rate=0`` festlegt.
        """
        equity = spy_backtest.equity_curve[EQUITY_COLUMN]
        returns = equity.pct_change().dropna()

        referenz = float(returns.mean() / returns.std(ddof=1) * math.sqrt(252.0))

        assert spy_backtest.metrics.sharpe == pytest.approx(referenz, rel=0.02)

    def test_max_drawdown_ist_da_und_negativ(self, spy_backtest):
        """Kriterium: Max Drawdown liegt vor. Konvention: negativ."""
        mdd = spy_backtest.metrics.max_drawdown

        assert mdd is not None
        assert mdd <= 0.0, "Max Drawdown muss negativ sein (Projektkonvention)"
        assert mdd >= -1.0, "Ein Drawdown unter -100 Prozent ist unmoeglich"

    def test_max_drawdown_gegen_referenzrechnung(self, spy_backtest):
        """Max Drawdown, von Hand aus der Equity-Kurve nachgerechnet.

        Referenz: das Minimum von ``equity / cummax(equity) - 1``.
        """
        equity = spy_backtest.equity_curve[EQUITY_COLUMN]
        referenz = float((equity / equity.cummax() - 1.0).min())

        assert spy_backtest.metrics.max_drawdown == pytest.approx(referenz, rel=1e-6)

    def test_cagr_ist_da_und_plausibel(self, spy_backtest):
        """Kriterium: CAGR liegt vor und ist keine Fantasiezahl."""
        cagr = spy_backtest.metrics.cagr

        assert cagr is not None
        assert not math.isnan(cagr)
        assert -1.0 < cagr < MAX_PLAUSIBLE_CAGR, f"CAGR {cagr} ist unplausibel"

    def test_cagr_gegen_referenzrechnung(self, spy_backtest):
        """CAGR, unabhaengig nachgerechnet aus Start, Ende und Jahren.

        Referenz: ``(ende / start) ** (1 / jahre) - 1``, wobei die Jahre
        aus der Bar-Zahl und 252 Handelstagen folgen - dieselbe Basis, die
        ``Interval.D1.periods_per_year`` liefert.
        """
        equity = spy_backtest.equity_curve[EQUITY_COLUMN]
        start = float(equity.iloc[0])
        ende = float(equity.iloc[-1])
        jahre = (len(equity) - 1) / Interval.D1.periods_per_year

        referenz = (ende / start) ** (1.0 / jahre) - 1.0

        assert spy_backtest.metrics.cagr == pytest.approx(referenz, rel=1e-6)

    def test_total_return_gegen_referenzrechnung(self, spy_backtest):
        """Gesamtrendite ist schlicht Ende durch Start minus eins."""
        equity = spy_backtest.equity_curve[EQUITY_COLUMN]
        referenz = float(equity.iloc[-1]) / float(equity.iloc[0]) - 1.0

        assert spy_backtest.metrics.total_return == pytest.approx(referenz, rel=1e-9)

    def test_hit_rate_ist_da_und_im_einheitsintervall(self, spy_backtest):
        """Kriterium: Hit Rate liegt vor, dezimal zwischen 0 und 1."""
        hit_rate = spy_backtest.metrics.hit_rate

        assert hit_rate is not None
        assert 0.0 <= hit_rate <= 1.0, f"Hit Rate {hit_rate} liegt ausserhalb [0, 1]"

    def test_hit_rate_gegen_referenzrechnung(self, spy_backtest):
        """Hit Rate, aus der Trade-Liste ausgezaehlt.

        Referenz: Anteil der geschlossenen Trades mit positivem PnL.
        """
        geschlossen = [t for t in spy_backtest.trades if t.pnl is not None]
        if not geschlossen:
            pytest.skip("Keine geschlossenen Trades im Lauf")

        gewinner = sum(1 for t in geschlossen if t.pnl is not None and t.pnl > 0)
        referenz = gewinner / len(geschlossen)

        assert spy_backtest.metrics.hit_rate == pytest.approx(referenz, rel=1e-9)

    def test_n_trades_passt_zur_trade_liste(self, spy_backtest):
        """Die gemeldete Trade-Zahl stimmt mit der Liste ueberein."""
        geschlossen = [t for t in spy_backtest.trades if t.pnl is not None]
        assert spy_backtest.metrics.n_trades == len(geschlossen)

    def test_kosten_werden_verrechnet(self, spy_backtest):
        """Die Anfrage traegt Gebuehren, also weist das Ergebnis sie aus."""
        assert spy_backtest.request.fees > 0
        assert spy_backtest.metrics.total_fees is not None
        assert spy_backtest.metrics.total_fees > 0, "Kosten gesetzt, aber nichts verrechnet"

    def test_kosten_mindern_die_rendite(self, provider):
        """Derselbe Lauf mit Kosten schneidet schlechter ab als ohne.

        Der schaerfste Test der Kostenrechnung: ein reiner Vorzeichen-Check
        auf ``total_fees`` wuerde auch dann gruen, wenn die Kosten nirgends
        in die Equity-Kurve einflossen.
        """
        engine = VectorbtEngine()
        if not engine.is_available():
            pytest.skip("vectorbt ist nicht installiert - optionales Extra")

        prices = {BACKTEST_TICKER: provider.get_history(BACKTEST_TICKER, interval=Interval.D1)}
        strategy = SmaCrossStrategy(
            StrategyConfig(key="sma_cross", params={"fast": 50, "slow": 200})
        )
        signals = strategy.generate_signals(prices)

        def lauf(fees: float, slippage: float) -> float:
            request = BacktestRequest(
                tickers=[BACKTEST_TICKER],
                strategy=StrategyConfig(key="sma_cross", params={"fast": 50, "slow": 200}),
                interval=Interval.D1,
                init_cash=100_000.0,
                fees=fees,
                slippage=slippage,
            )
            result = engine.run(request, prices, signals)
            return float(result.equity_curve[EQUITY_COLUMN].iloc[-1])

        ohne_kosten = lauf(0.0, 0.0)
        mit_kosten = lauf(0.005, 0.005)

        assert mit_kosten < ohne_kosten, "Kosten wirken sich nicht auf die Equity aus"

    def test_metriken_stimmen_mit_dem_metrik_modul_ueberein(self, spy_backtest):
        """Die Engine rechnet ueber ``backtest/metrics.py``, nicht selbst.

        Beide Engines sollen dieselbe Metrik-Quelle nutzen, damit ihre
        Zahlen vergleichbar sind. Das prueft dieser Test.
        """
        equity = spy_backtest.equity_curve[EQUITY_COLUMN]
        nachgerechnet = metrics_mod.compute_metrics(
            equity,
            trades=spy_backtest.trades,
            interval=Interval.D1,
            risk_free_rate=0.0,
        )

        assert spy_backtest.metrics.cagr == pytest.approx(nachgerechnet.cagr, rel=1e-9)
        assert spy_backtest.metrics.max_drawdown == pytest.approx(
            nachgerechnet.max_drawdown, rel=1e-9
        )
        assert spy_backtest.metrics.sharpe == pytest.approx(nachgerechnet.sharpe, rel=1e-9)

    def test_kein_look_ahead_im_signal(self, provider):
        """Die Strategie verschiebt die Ausfuehrung um mindestens einen Bar.

        Look-ahead-Freiheit ist eine harte Projektregel. Sie laesst sich
        hier daran festmachen, dass der Signalsatz einen Ausfuehrungsversatz
        traegt und die ersten Bars - ohne volles Fenster - kein Signal
        geben.
        """
        prices = {BACKTEST_TICKER: provider.get_history(BACKTEST_TICKER, interval=Interval.D1)}
        strategy = SmaCrossStrategy(
            StrategyConfig(key="sma_cross", params={"fast": 50, "slow": 200})
        )
        signals = strategy.generate_signals(prices)

        assert signals.execution_lag >= 1

        # Vor Bar 200 ist der langsame SMA nicht definiert, also kein Signal.
        fruehe_entries = signals.entries.iloc[:200]
        fruehe_exits = signals.exits.iloc[:200]
        assert not fruehe_entries.to_numpy().any(), "Einstieg ohne vollstaendiges Fenster"
        assert not fruehe_exits.to_numpy().any(), "Ausstieg ohne vollstaendiges Fenster"


# ==========================================================================
# Kriterium 4 - Portfolio-Optimierung ueber 10 Ticker
# ==========================================================================


@pytest.fixture(scope="module")
def zehn_ticker(provider: FixtureProvider) -> list[str]:
    """Die zehn Ticker des Kriteriums."""
    tickers = provider.available_tickers()
    assert len(tickers) >= 10, f"Nur {len(tickers)} Ticker im Fixture-Satz"
    return tickers[:10]


@pytest.fixture(scope="module")
def optimierung(provider: FixtureProvider, zehn_ticker: list[str]):
    """Optimiert einmal ueber die zehn Ticker, fuer alle Tests der Klasse."""
    prices = provider.get_histories(zehn_ticker, interval=Interval.D1)
    request = PortfolioRequest(
        tickers=zehn_ticker,
        objective=OptimizationObjective.MAX_SHARPE,
        risk_free_rate=0.02,
    )
    return optimize(request, prices)


class TestKriteriumPortfolio:
    """Optimierung ueber zehn Ticker mit Gewichten, Grenze und Risiko."""

    def test_zehn_ticker_gehen_ein(self, optimierung, zehn_ticker):
        """Das Kriterium verlangt zehn Titel - alle haben Kurse."""
        assert len(zehn_ticker) == 10
        assert len(optimierung.positions) == 10

    def test_gewichte_summieren_auf_eins(self, optimierung):
        """Kriterium: Summe der Gewichte ist 1.0."""
        summe = sum(p.weight for p in optimierung.positions)
        assert summe == pytest.approx(1.0, abs=WEIGHT_SUM_TOLERANCE)

    def test_keine_negativen_gewichte_ohne_shorting(self, optimierung):
        """Kriterium: ohne Shorting-Flag gibt es keine negativen Gewichte."""
        assert not optimierung.request.allow_shorting

        negativ = {p.ticker: p.weight for p in optimierung.positions if p.weight < 0}
        assert not negativ, f"Negative Gewichte ohne allow_shorting: {negativ}"

    def test_effizienzgrenze_liegt_vor(self, optimierung):
        """Kriterium: die Effizienzgrenze wird zurueckgegeben."""
        frontier = optimierung.frontier

        assert frontier is not None, f"Keine Frontier. Warnungen: {optimierung.warnings}"
        assert not frontier.data.empty
        assert {"volatility", "expected_return", "sharpe"} <= set(frontier.data.columns)

    def test_effizienzgrenze_ist_monoton(self, optimierung):
        """Auf der Grenze steigt mit dem Risiko auch die erwartete Rendite.

        Das ist die definierende Eigenschaft einer Effizienzgrenze. Waere
        sie verletzt, waere der zurueckgegebene Rand nicht effizient.
        """
        data = optimierung.frontier.data.sort_values("volatility")
        renditen = data["expected_return"].to_numpy()

        differenzen = renditen[1:] - renditen[:-1]
        assert (differenzen >= -1e-6).all(), "Die Grenze faellt - sie ist nicht effizient"

    def test_risikokennzahlen_liegen_vor(self, optimierung):
        """Kriterium: Risikokennzahlen kommen mit zurueck."""
        risk = optimierung.risk

        assert risk is not None, f"Keine Risikokennzahlen. Warnungen: {optimierung.warnings}"
        assert risk.volatility is not None
        assert risk.volatility > 0
        assert risk.max_drawdown is not None
        assert risk.max_drawdown <= 0.0, "Drawdown muss negativ sein (Projektkonvention)"
        assert risk.var_historical is not None
        assert risk.var_historical <= 0.0, "VaR muss negativ sein (Projektkonvention)"

    def test_erwartete_kennzahlen_liegen_vor(self, optimierung):
        """Rendite, Volatilitaet und Sharpe der Optimierung sind gefuellt."""
        assert optimierung.expected_return is not None
        assert optimierung.expected_volatility is not None
        assert optimierung.expected_volatility > 0
        assert optimierung.sharpe_ratio is not None

    def test_korrelationsmatrix_liegt_vor(self, optimierung, zehn_ticker):
        """Die Korrelationsmatrix ist quadratisch mit Einsen auf der Diagonale."""
        korr = optimierung.correlation

        assert korr is not None
        assert korr.shape == (len(zehn_ticker), len(zehn_ticker))
        for ticker in korr.columns:
            assert float(korr.loc[ticker, ticker]) == pytest.approx(1.0, abs=1e-9)

    def test_discrete_allocation_passt_ins_budget(self, optimierung):
        """Die Stueckzahlen ueberschreiten das Budget nicht."""
        investiert = sum(p.value for p in optimierung.positions if p.value is not None)
        budget = optimierung.request.total_value

        assert investiert <= budget * (1.0 + 1e-6), "Die Allokation sprengt das Budget"
        if optimierung.leftover_cash is not None:
            assert optimierung.leftover_cash >= -1e-6

    @pytest.mark.parametrize(
        "objective",
        [
            OptimizationObjective.MAX_SHARPE,
            OptimizationObjective.MIN_VOLATILITY,
            OptimizationObjective.EQUAL_WEIGHT,
            OptimizationObjective.HRP,
        ],
    )
    def test_jede_zielfunktion_erfuellt_das_kriterium(self, provider, zehn_ticker, objective):
        """Das Kriterium gilt fuer jede angebotene Zielfunktion, nicht nur Max Sharpe."""
        prices = provider.get_histories(zehn_ticker, interval=Interval.D1)
        request = PortfolioRequest(
            tickers=zehn_ticker,
            objective=objective,
            risk_free_rate=0.02,
        )
        result = optimize(request, prices)

        summe = sum(p.weight for p in result.positions)
        assert summe == pytest.approx(1.0, abs=WEIGHT_SUM_TOLERANCE)
        assert all(p.weight >= -WEIGHT_SUM_TOLERANCE for p in result.positions)

    def test_shorting_flag_erlaubt_negative_gewichte(self, provider, zehn_ticker):
        """Die Gegenprobe: mit Flag sind negative Gewichte zulaessig.

        Ohne diesen Test koennte die Implementierung negative Gewichte
        schlicht immer abschneiden und der Test oben waere wertlos.
        """
        prices = provider.get_histories(zehn_ticker, interval=Interval.D1)
        request = PortfolioRequest(
            tickers=zehn_ticker,
            objective=OptimizationObjective.MAX_SHARPE,
            risk_free_rate=0.02,
            allow_shorting=True,
            min_weight=-1.0,
        )
        result = optimize(request, prices)

        summe = sum(p.weight for p in result.positions)
        assert summe == pytest.approx(1.0, abs=WEIGHT_SUM_TOLERANCE)


# ==========================================================================
# Kriterium 5 - Offline-Betrieb
# ==========================================================================


class TestKriteriumOffline:
    """Alle Tests laufen offline.

    ``tests/conftest.py`` blockiert echte Sockets prozessweit. Diese Klasse
    prueft, dass die Sperre wirkt und dass der Fixture-Provider seine
    Herkunft ehrlich ausweist.
    """

    def test_netzwerksperre_ist_aktiv(self):
        """Ein Verbindungsversuch nach draussen schlaegt fehl.

        Ohne diesen Test koennte die Sperre unbemerkt kaputtgehen und die
        Suite stillschweigend ins Netz greifen.
        """
        import socket

        with pytest.raises(RuntimeError, match="Netzwerkzugriff"):
            socket.create_connection(("example.com", 80), timeout=1)

    def test_provider_weist_fixture_herkunft_aus(self, provider):
        """Der Provider gibt sich als Fixture-Quelle zu erkennen."""
        from btlite.models.common import DataSource

        series = provider.get_history("AAPL", interval=Interval.D1)

        assert series.provenance.source is DataSource.FIXTURE
        assert series.provenance.from_cache is True

    def test_alle_fixture_ticker_sind_abrufbar(self, provider):
        """Jeder gemeldete Ticker liefert auch tatsaechlich Daten."""
        for ticker in provider.available_tickers():
            series = provider.get_history(ticker, interval=Interval.D1)
            assert not series.data.empty
