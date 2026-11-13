"""Tests der sechs Seiten.

Alle laufen offline gegen den FixtureProvider. Geprueft wird, dass jede
Seite ihr Layout ohne Exception baut, den Herkunftsnachweis zeigt und im
Fehlerfall ein Banner statt einer leeren Seite liefert.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from btlite.config import Settings
from btlite.core.errors import SymbolNotFoundError
from btlite.models.common import Period
from btlite.models.portfolio import OptimizationObjective
from btlite.ui.pages import PAGES, ROUTES, bt, des, eqs, fa, gp, port
from btlite.ui.state import AppState, set_state


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[AppState]:
    """Baut einen Fixture-gestuetzten UI-Zustand.

    Die Depot-Datei zeigt ins temporaere Verzeichnis. Bisher war das
    folgenlos, weil keine der hier geprueften Seiten den Bestand las -
    seit PORT eine Auswahl der gespeicherten Depots anbietet, liefe der
    Test sonst gegen den echten Bestand des Nutzers.
    """
    monkeypatch.setenv("BTLITE_PORTFOLIO_HOLDINGS_FILE", str(tmp_path / "holdings.json"))
    settings = Settings()
    settings.data.offline = True
    instance = AppState(settings)
    set_state(instance)
    yield instance
    set_state(None)


def walk(node: Any) -> Iterator[Any]:
    """Laeuft rekursiv durch einen Dash-Komponentenbaum."""
    yield node
    children = getattr(node, "children", None)
    if children is None:
        return
    if isinstance(children, list | tuple):
        for child in children:
            yield from walk(child)
    else:
        yield from walk(children)


def classes(tree: Any) -> list[str]:
    """Sammelt alle CSS-Klassen eines Komponentenbaums."""
    return [str(getattr(n, "className", "") or "") for n in walk(tree)]


def has_class(tree: Any, name: str) -> bool:
    """Prueft, ob irgendwo im Baum eine CSS-Klasse vorkommt."""
    return any(name in c for c in classes(tree))


def text_of(tree: Any) -> str:
    """Sammelt allen Textinhalt eines Komponentenbaums."""
    parts: list[str] = []
    for node in walk(tree):
        children = getattr(node, "children", node)
        if isinstance(children, str):
            parts.append(children)
    return " ".join(parts)


class TestStateWiring:
    def test_offline_uses_fixtures(self, state):
        assert state.is_offline
        assert state.offline_reason

    def test_available_tickers(self, state):
        assert "AAPL" in state.available_tickers()

    def test_default_tickers_are_servable(self, state):
        assert set(state.default_tickers) <= set(state.available_tickers())

    def test_history_is_cached(self, state):
        first = state.history("AAPL", period="1y")
        second = state.history("AAPL", period="1y")
        assert first is second

    def test_ratios_cached(self, state):
        assert state.ratios("AAPL") is state.ratios("AAPL")

    def test_clear_cache(self, state):
        first = state.history("AAPL", period="1y")
        state.clear_cache()
        assert state.history("AAPL", period="1y") is not first

    def test_unknown_symbol_raises(self, state):
        with pytest.raises(SymbolNotFoundError):
            state.instrument("NOSUCHTICKER")

    def test_histories_skips_unknown(self, state):
        result = state.histories(["AAPL", "NOSUCHTICKER"])
        assert set(result) == {"AAPL"}

    def test_ttm_bundle(self, state):
        bundle = state.statements("AAPL", period=Period.TTM)
        assert bundle.period is Period.TTM


def _skip_if_engine_missing(mnemonic: str) -> None:
    """Ueberspringt BT-Faelle, wenn das optionale vectorbt-Extra fehlt.

    Ohne die Engine zeigt die Seite BT korrekt ein Fehlerbanner. Das ist
    richtiges Produktverhalten, kein Testfehler - die Suite darf daran
    nicht scheitern.
    """
    if mnemonic != "BT":
        return
    from btlite.backtest.vbt_engine import VectorbtEngine

    if not VectorbtEngine().is_available():
        pytest.skip("vectorbt ist nicht installiert - optionales Extra")


class TestAllPagesBuild:
    @pytest.mark.parametrize("mnemonic", sorted(PAGES))
    def test_layout_without_exception(self, state, mnemonic):
        tree = PAGES[mnemonic](state, ["AAPL", "MSFT"], ())
        assert tree is not None

    @pytest.mark.parametrize("mnemonic", sorted(PAGES))
    def test_no_error_banner_with_fixtures(self, state, mnemonic):
        _skip_if_engine_missing(mnemonic)
        tree = PAGES[mnemonic](state, ["AAPL", "MSFT"], ())
        assert not has_class(tree, "bt-error")

    @pytest.mark.parametrize("mnemonic", sorted(PAGES))
    def test_shows_provenance(self, state, mnemonic):
        _skip_if_engine_missing(mnemonic)
        tree = PAGES[mnemonic](state, ["AAPL", "MSFT"], ())
        assert has_class(tree, "bt-provenance")

    @pytest.mark.parametrize("mnemonic", sorted(PAGES))
    def test_without_tickers_uses_defaults(self, state, mnemonic):
        _skip_if_engine_missing(mnemonic)
        tree = PAGES[mnemonic](state, [], ())
        assert not has_class(tree, "bt-error")

    def test_routes_cover_all_pages(self):
        assert set(ROUTES.values()) == set(PAGES)


class TestDes:
    def test_shows_master_data(self, state):
        tree = des.layout(state, ["AAPL"], ())
        assert "Apple" in text_of(tree)

    def test_shows_quote_tiles(self, state):
        assert has_class(des.layout(state, ["AAPL"], ()), "bt-tile")

    def test_unknown_ticker_shows_banner(self, state):
        tree = des.layout(state, ["NOSUCHTICKER"], ())
        assert has_class(tree, "bt-error")

    def test_etf_without_statements_is_not_an_error(self, state):
        # SPY ist ein ETF und fuehrt keine eigene Bilanz. Das ist die Natur
        # des Instruments, kein Fehler - also Hinweis statt Fehlerbanner.
        tree = des.layout(state, ["SPY"], ())
        assert not has_class(tree, "bt-error")
        assert has_class(tree, "bt-placeholder")


class TestGp:
    def test_default_chart(self, state):
        tree = gp.layout(state, ["AAPL"], ())
        assert any(type(n).__name__ == "Graph" for n in walk(tree))

    def test_period_from_args(self, state):
        assert gp._period_from_args(("5y",)) == "5y"
        assert gp._period_from_args(()) == gp.DEFAULT_PERIOD
        assert gp._period_from_args(("bogus",)) == gp.DEFAULT_PERIOD

    def test_multi_ticker_gets_a_comparison_plus_one_chart_each(self, state):
        """Vor den Einzelcharts steht der normierte Vergleich."""
        tree = gp.render_charts(state, ["AAPL", "MSFT"], "1y")
        graphs = [n for n in walk(tree) if type(n).__name__ == "Graph"]
        assert len(graphs) == 3, "Vergleich + zwei Einzelcharts"
        assert "base 100" in text_of(tree)
        assert not has_class(tree, "bt-error")

    def test_single_ticker_has_no_comparison(self, state):
        tree = gp.render_charts(state, ["AAPL"], "1y")
        graphs = [n for n in walk(tree) if type(n).__name__ == "Graph"]
        assert len(graphs) == 1
        assert "base 100" not in text_of(tree)

    @pytest.mark.parametrize("kind", ["candlestick", "ohlc", "line"])
    def test_chart_kinds(self, state, kind):
        tree = gp.render_charts(state, ["AAPL"], "1y", kind=kind)
        assert not has_class(tree, "bt-error")

    @pytest.mark.parametrize("overlay", ["sma", "ema", "bollinger", "rsi", "macd"])
    def test_each_overlay(self, state, overlay):
        tree = gp.render_charts(state, ["AAPL"], "1y", overlays=(overlay,))
        assert not has_class(tree, "bt-error")

    def test_unknown_ticker_banner(self, state):
        tree = gp.render_charts(state, ["NOSUCHTICKER"], "1y")
        assert has_class(tree, "bt-error")

    def test_summary_tiles(self, state):
        assert has_class(gp.summary_tiles(state, ["AAPL"]), "bt-tile")


class TestFa:
    @pytest.mark.parametrize("period", list(Period))
    def test_every_period_basis(self, state, period):
        tree = fa.render_body(state, "AAPL", [], period)
        assert not has_class(tree, "bt-error")

    def test_period_from_args(self):
        assert fa._period_from_args(("quarterly",)) is Period.QUARTERLY
        assert fa._period_from_args(()) is Period.ANNUAL

    def test_peer_comparison_present(self, state):
        tree = fa.render_body(state, "AAPL", ["MSFT", "NVDA"], Period.ANNUAL)
        assert "Peer" in text_of(tree)

    def test_without_peers_shows_hint(self, state):
        tree = fa.render_body(state, "AAPL", [], Period.ANNUAL)
        assert has_class(tree, "bt-placeholder")

    def test_unknown_subject_banner(self, state):
        tree = fa.render_body(state, "NOSUCHTICKER", [], Period.ANNUAL)
        assert has_class(tree, "bt-error")

    def test_etf_without_statements_is_not_an_error(self, state):
        tree = fa.render_body(state, "SPY", [], Period.ANNUAL)
        assert not has_class(tree, "bt-error")
        assert has_class(tree, "bt-placeholder")


class TestEqs:
    def test_runs_all_presets(self, state):
        from btlite.screening.presets import all_presets

        universe = state.available_tickers()
        for preset in all_presets():
            result = eqs.run_screen(state, preset.key, universe)
            assert result.total_matches >= 0

    def test_result_table(self, state):
        tree = eqs.render_result(state, "momentum", state.available_tickers())
        assert has_class(tree, "bt-table")

    def test_filter_description_visible(self, state):
        tree = eqs.render_result(state, "value", state.available_tickers())
        assert "Local filters" in text_of(tree)

    def test_unknown_preset_banner(self, state):
        tree = eqs.render_result(state, "nosuchpreset", state.available_tickers())
        assert has_class(tree, "bt-error")

    def test_preset_from_args(self):
        from btlite.screening.presets import all_presets

        presets = all_presets()
        assert eqs._preset_from_args(("quality",), presets) == "quality"
        assert eqs._preset_from_args((), presets) == presets[0].key


class TestBt:
    def test_parse_params(self):
        assert bt.parse_params("fast=20 slow=100", {"fast": 50, "slow": 200}) == {
            "fast": 20,
            "slow": 100,
        }

    def test_parse_params_keeps_defaults(self):
        assert bt.parse_params("", {"fast": 50}) == {"fast": 50}

    def test_parse_params_float_and_comma(self):
        assert bt.parse_params("oversold=25.5, overbought=70", {}) == {
            "oversold": 25.5,
            "overbought": 70,
        }

    def test_parse_params_ignores_malformed(self):
        assert bt.parse_params("garbage", {"fast": 5}) == {"fast": 5}

    def test_backtest_runs(self, state):
        _skip_if_engine_missing("BT")
        result = bt.run_backtest(state, ["AAPL"], "sma_cross", {"fast": 20, "slow": 50})
        assert not result.equity_curve.empty
        assert result.engine_used

    def test_render_result_has_metrics(self, state):
        _skip_if_engine_missing("BT")
        tree = bt.render_result(state, ["AAPL"], "sma_cross", {"fast": 20, "slow": 50})
        assert not has_class(tree, "bt-error")
        assert "Sharpe" in text_of(tree)

    def test_invalid_params_show_banner(self, state):
        tree = bt.render_result(state, ["AAPL"], "sma_cross", {"fast": 200, "slow": 20})
        assert has_class(tree, "bt-error")

    def test_unknown_strategy_banner(self, state):
        tree = bt.render_result(state, ["AAPL"], "nosuchstrategy", {})
        assert has_class(tree, "bt-error")

    def test_unknown_ticker_banner(self, state):
        tree = bt.render_result(state, ["NOSUCHTICKER"], "buy_and_hold", {})
        assert has_class(tree, "bt-error")

    def test_strategy_from_args(self, state):
        registry = state.strategy_registry()
        # registry.keys() ist eine Methode der Registry, keine dict-View.
        known = list(registry.keys())
        assert bt._strategy_from_args(("buy_and_hold",), registry) == "buy_and_hold"
        assert bt._strategy_from_args((), registry) in known

    def test_empty_trade_list_placeholder(self, state):
        _skip_if_engine_missing("BT")
        result = bt.run_backtest(state, ["AAPL"], "buy_and_hold", {})
        tree = bt.trade_table([], result)
        assert has_class(tree, "bt-placeholder")


class TestPort:
    def test_optimization_runs(self, state):
        result = port.run_optimization(
            state, ["AAPL", "MSFT", "KO"], OptimizationObjective.MAX_SHARPE
        )
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6

    @pytest.mark.parametrize(
        "objective",
        [
            OptimizationObjective.MAX_SHARPE,
            OptimizationObjective.MIN_VOLATILITY,
            OptimizationObjective.EQUAL_WEIGHT,
            OptimizationObjective.HRP,
        ],
    )
    def test_each_objective(self, state, objective):
        tree = port.render_result(state, ["AAPL", "MSFT", "KO"], objective)
        assert not has_class(tree, "bt-error")

    def test_single_ticker_shows_banner(self, state):
        tree = port.render_result(state, ["NOSUCHTICKER"], OptimizationObjective.MAX_SHARPE)
        assert has_class(tree, "bt-error")

    def test_correlation_and_frontier_present(self, state):
        tree = port.render_result(state, ["AAPL", "MSFT", "KO"], OptimizationObjective.MAX_SHARPE)
        text = text_of(tree)
        assert "Efficient frontier" in text
        assert "Correlation matrix" in text

    def test_realized_risk_is_shown_not_na(self, state):
        """Das realisierte Portfolio-Risiko wird gerechnet, nicht als n/a gezeigt."""
        tree = port.render_result(state, ["AAPL", "MSFT", "KO"], OptimizationObjective.MAX_SHARPE)
        rendered = text_of(tree)
        assert "Realized volatility" in rendered
        assert not has_class(tree, "bt-error")
        # Die Kachel traegt einen Prozentwert, keinen Platzhalter.
        index = rendered.index("Realized volatility")
        assert "n/a" not in rendered[index : index + 60]

    def test_objective_from_args(self):
        assert (
            port._objective_from_args(("min_volatility",)) is OptimizationObjective.MIN_VOLATILITY
        )
        assert port._objective_from_args(()) is OptimizationObjective.MAX_SHARPE


class TestPortPortfolioPicker:
    """Die Auswahl eines gespeicherten Depots auf der Seite PORT.

    Der Nutzer musste die Ticker seines Depots bisher von Hand in die
    Kommandozeile tippen - die Seite kannte den gespeicherten Bestand
    nicht. Die Auswahl setzt sie ein.

    Was die Seite dabei NICHT uebernimmt: Stueckzahlen und
    Einstandskurse. PORT fragt, wie man diese Titel gewichten sollte,
    nicht wie das Depot heute dasteht. Letzteres steht auf WL.
    """

    @staticmethod
    def _saved(state: AppState, name: str, tickers: tuple[str, ...]) -> None:
        """Legt ein Depot mit den genannten Titeln an."""
        state.holdings.create_portfolio(name)
        for ticker in tickers:
            state.holdings.add_holding(
                name, ticker, shares=1, buy_price=100.0, buy_date=date(2024, 1, 15)
            )

    def test_saved_portfolios_appear_as_options(self, state):
        self._saved(state, "Mein Depot", ("AAPL", "MSFT", "KO"))
        options = port.portfolio_options(state)
        assert [o["value"] for o in options] == ["Mein Depot"]

    def test_the_label_names_the_count(self, state):
        """Die Titelzahl steht im Label - sie entscheidet ueber die Dauer."""
        self._saved(state, "Mein Depot", ("AAPL", "MSFT", "KO"))
        assert port.portfolio_options(state)[0]["label"] == "Mein Depot (3)"

    def test_a_portfolio_with_one_title_is_not_offered(self, state):
        """Mit einem Titel gibt es nichts zu gewichten.

        Ein Eintrag, der beim Anklicken nur eine Fehlermeldung bringt,
        ist schlechter als keiner.
        """
        self._saved(state, "Einzeln", ("AAPL",))
        assert port.portfolio_options(state) == []

    def test_an_empty_portfolio_is_not_offered(self, state):
        state.holdings.create_portfolio("Leer")
        assert port.portfolio_options(state) == []

    def test_without_any_portfolio_the_list_is_empty(self, state):
        assert port.portfolio_options(state) == []

    def test_tickers_are_unique_and_sorted(self, state):
        """Zwei Kaeufe desselben Titels sind ein Titel im Universum."""
        self._saved(state, "Doppelt", ("MSFT", "AAPL", "AAPL"))
        assert port.tickers_of(state, "Doppelt") == ["AAPL", "MSFT"]

    def test_an_unknown_name_yields_nothing(self, state):
        assert port.tickers_of(state, "gibtsnicht") == []

    def test_none_yields_nothing(self, state):
        assert port.tickers_of(state, None) == []

    def test_the_picker_is_in_the_page(self, state):
        self._saved(state, "Mein Depot", ("AAPL", "MSFT", "KO"))
        text = text_of(port.layout(state, ["AAPL", "MSFT"]))
        assert "Saved portfolio" in text

    def test_the_page_works_without_any_portfolio(self, state):
        """Ohne gespeichertes Depot bleibt die Seite benutzbar."""
        tree = port.layout(state, ["AAPL", "MSFT"])
        assert not has_class(tree, "bt-error")
