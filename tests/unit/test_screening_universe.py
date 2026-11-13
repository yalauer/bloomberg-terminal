"""Tests der Universum-Anreicherung.

Das Modul ist das Bindeglied zwischen Datenschicht und Screening. Ohne es
laufen lokale Praedikate ins Leere, weil ein Finviz-Ergebnis Spalten wie
``P/E`` traegt, die Filter-DSL aber die Kennzahlen-Schluessel aus
``analytics/ratios.py`` erwartet.
"""

from __future__ import annotations

import pandas as pd
import pytest

from btlite.core.errors import DataError, EmptyUniverseError, SymbolNotFoundError
from btlite.data.fixtures import FixtureProvider
from btlite.models.common import Period
from btlite.screening.engine import ScreeningEngine
from btlite.screening.presets import all_presets
from btlite.screening.universe import (
    PROGRESS_EVERY,
    TICKER_INDEX_NAME,
    build_ratio_universe,
)


@pytest.fixture
def provider() -> FixtureProvider:
    return FixtureProvider()


@pytest.fixture
def universe(provider):
    return build_ratio_universe(provider, provider.available_tickers())


class TestBuildRatioUniverse:
    def test_index_is_the_ticker(self, universe):
        assert universe.index.name == TICKER_INDEX_NAME
        assert "AAPL" in universe.index

    def test_ratio_columns_use_the_engine_keys(self, universe):
        """Die Spalten muessen den Schluesseln aus ratios.py entsprechen."""
        for key in ("pe", "pb", "roe", "net_margin", "current_ratio", "debt_to_equity"):
            assert key in universe.columns, f"{key} fehlt im Universum"

    def test_metadata_columns_are_present(self, universe):
        for column in ("sector", "industry", "market_cap", "price"):
            assert column in universe.columns

    def test_values_are_decimal_not_percent(self, universe):
        """Margen und Renditen dezimal: 0.27 bedeutet 27 Prozent."""
        margins = universe["net_margin"].dropna()
        assert not margins.empty
        assert (margins.abs() < 5.0).all(), "Werte sehen nach Prozentpunkten aus"

    def test_valuation_needs_quotes(self, provider):
        """Ohne Kurse bleiben KGV und Dividendenrendite leer, nicht null."""
        without = build_ratio_universe(provider, ["AAPL"], with_quotes=False)
        assert pd.isna(without.loc["AAPL", "pe"])

    def test_valuation_appears_with_quotes(self, universe):
        assert pd.notna(universe.loc["AAPL", "pe"])

    def test_dividend_yield_is_populated(self, universe):
        """Die Dividendenrendite kommt aus den Stammdaten, nicht aus der Bilanz."""
        yields = universe["dividend_yield"].dropna()
        assert not yields.empty, "Kein einziger Titel mit Ausschuettung"
        assert (yields >= 0).all()
        assert (yields < 1.0).all(), "Dezimal erwartet, nicht Prozentpunkte"

    def test_empty_input_yields_empty_frame(self, provider):
        result = build_ratio_universe(provider, [])
        assert result.empty
        assert result.index.name == TICKER_INDEX_NAME

    def test_symbols_are_normalized(self, provider):
        result = build_ratio_universe(provider, ["  aapl  "], with_quotes=False)
        assert list(result.index) == ["AAPL"]

    def test_unknown_symbol_is_skipped_by_default(self, provider):
        result = build_ratio_universe(provider, ["AAPL", "ZZZZ"], with_quotes=False)
        assert list(result.index) == ["AAPL"]

    def test_unknown_symbol_raises_when_asked(self, provider):
        with pytest.raises(DataError):
            build_ratio_universe(provider, ["ZZZZ"], with_quotes=False, skip_failures=False)

    def test_all_unknown_raises_instead_of_an_empty_frame(self, provider):
        """Lieferte frueher einen leeren Frame - mit schlechten Folgen.

        Der lief weiter in die Filter, und die meldeten dort "Field 'pe'
        does not exist in the universe": eine Aussage ueber das falsche
        Problem. Der Nutzer las einen technischen Fehler, obwohl nur die
        Quelle nichts geliefert hatte.
        """
        with pytest.raises(EmptyUniverseError):
            build_ratio_universe(provider, ["ZZZ1", "ZZZ2"], with_quotes=False)

    def test_the_error_names_how_many_were_asked_for(self, provider):
        """Fuer das Log - die Oberflaeche zeigt nur die user_message."""
        with pytest.raises(EmptyUniverseError) as excinfo:
            build_ratio_universe(provider, ["ZZZ1", "ZZZ2"], with_quotes=False)
        assert "2" in (excinfo.value.detail or "")

    def test_one_known_title_still_yields_a_frame(self, provider):
        """Die Ausnahme gilt nur, wenn WIRKLICH nichts ankam."""
        result = build_ratio_universe(provider, ["ZZZ1", "AAPL"], with_quotes=False)
        assert not result.empty
        assert list(result.index) == ["AAPL"]

    def test_quarterly_basis_is_supported(self, provider):
        result = build_ratio_universe(
            provider, ["AAPL"], period=Period.QUARTERLY, with_quotes=False
        )
        assert "roe" in result.columns

    def test_missing_ratio_is_nan_not_zero(self, universe):
        """Ein fehlender Wert darf nicht als 0 im Screening landen."""
        assert universe.isna().to_numpy().any(), "Erwartet mindestens eine Luecke"
        assert (universe.select_dtypes("number").fillna(-1) != 0).to_numpy().any()


class TestPresetsAgainstRealUniverse:
    """Akzeptanzkriterium: ein Preset liefert mehr als null Treffer."""

    @pytest.mark.parametrize("preset", all_presets(), ids=lambda p: p.key)
    def test_preset_runs_and_finds_something(self, provider, universe, preset):
        result = ScreeningEngine(provider=provider).screen(preset.request, universe=universe)
        assert result.total_matches > 0, f"{preset.key} findet nichts"
        assert set(result.tickers) <= set(universe.index)

    def test_value_screen_prefers_cheap_titles(self, provider, universe):
        from btlite.screening.presets import get_preset

        result = ScreeningEngine(provider=provider).screen(
            get_preset("value").request, universe=universe
        )
        for ticker in result.tickers:
            assert universe.loc[ticker, "pe"] <= 30.0

    def test_dividend_screen_only_returns_payers(self, provider, universe):
        from btlite.screening.presets import get_preset

        result = ScreeningEngine(provider=provider).screen(
            get_preset("dividend").request, universe=universe
        )
        for ticker in result.tickers:
            assert universe.loc[ticker, "dividend_yield"] >= 0.015


class TestInstrumentDividendFields:
    def test_dividend_rate_and_yield_are_read(self, provider):
        instrument = provider.get_instrument("KO")
        assert instrument.dividend_rate is not None
        assert instrument.dividend_yield is not None
        assert instrument.dividend_rate > 0

    def test_yield_is_decimal(self, provider):
        instrument = provider.get_instrument("PG")
        assert instrument.dividend_yield is not None
        assert 0.0 < instrument.dividend_yield < 0.2

    def test_unknown_symbol_still_raises(self, provider):
        with pytest.raises(SymbolNotFoundError):
            provider.get_instrument("ZZZZ")


class TestProgressReporting:
    """Der Fortschritt haengt an der Schleife, nicht am Ergebnis.

    Das Screening des S&P 500 laeuft bei rund einer Sekunde je Titel acht bis
    neun Minuten. Ohne Rueckmeldung sieht der Nutzer nur einen Balken und
    bricht ab. Diese Tests sichern die Meldungen, nicht ihre Darstellung.
    """

    @staticmethod
    def _many(provider: FixtureProvider) -> list[str]:
        """Genug Symbole, dass das Melde-Intervall mehrfach greift.

        Args:
            provider: Quelle der bekannten Symbole.

        Returns:
            Symbolliste mit einem Vielfachen von ``PROGRESS_EVERY`` Eintraegen.
        """
        return provider.available_tickers() * 6

    def test_without_callback_result_is_unchanged(self, provider):
        """``None`` heisst: Verhalten exakt wie bisher."""
        tickers = provider.available_tickers()
        expected = build_ratio_universe(provider, tickers, with_quotes=False)
        actual = build_ratio_universe(provider, tickers, with_quotes=False, on_progress=None)
        pd.testing.assert_frame_equal(expected, actual)

    def test_callback_receives_monotonic_progress(self, provider):
        calls: list[tuple[int, int]] = []
        tickers = self._many(provider)
        build_ratio_universe(
            provider, tickers, with_quotes=False, on_progress=lambda d, t: calls.append((d, t))
        )
        assert calls, "Kein einziger Fortschritt gemeldet"
        done = [d for d, _ in calls]
        assert done == sorted(done), "Zaehler laeuft nicht monoton"
        assert all(t == len(tickers) for _, t in calls), "Gesamtzahl wechselt"
        assert all(0 < d <= len(tickers) for d in done)

    def test_last_call_is_total_total(self, provider):
        """Sonst bliebe die Anzeige kurz vor Schluss stehen."""
        calls: list[tuple[int, int]] = []
        tickers = self._many(provider)
        build_ratio_universe(
            provider, tickers, with_quotes=False, on_progress=lambda d, t: calls.append((d, t))
        )
        assert calls[-1] == (len(tickers), len(tickers))

    def test_last_call_is_total_even_on_odd_count(self, provider):
        """Auch wenn die Zahl der Titel kein Vielfaches des Intervalls ist."""
        calls: list[tuple[int, int]] = []
        tickers = (provider.available_tickers() * 2)[:13]
        build_ratio_universe(
            provider, tickers, with_quotes=False, on_progress=lambda d, t: calls.append((d, t))
        )
        assert calls[-1] == (13, 13)

    def test_every_title_is_reported(self, provider):
        """Eine Meldung je Titel.

        Frueher jede zehnte: bei rund drei Sekunden je Titel im kalten Lauf
        stand die Anzeige dann bis zu dreissig Sekunden still und wirkte
        haengengeblieben. Eine Meldung kostet nur zwei Zuweisungen.
        """
        calls: list[tuple[int, int]] = []
        tickers = self._many(provider)
        build_ratio_universe(
            provider, tickers, with_quotes=False, on_progress=lambda d, t: calls.append((d, t))
        )
        assert PROGRESS_EVERY == 1
        assert [d for d, _t in calls] == list(range(1, len(tickers) + 1))

    def test_throwing_callback_does_not_break_the_run(self, provider):
        """Der Fortschritt ist Beiwerk - das Screening laeuft trotzdem zu Ende."""
        tickers = self._many(provider)
        expected = build_ratio_universe(provider, tickers, with_quotes=False)

        def boom(done: int, total: int) -> None:
            raise RuntimeError("Anzeige kaputt")

        actual = build_ratio_universe(provider, tickers, with_quotes=False, on_progress=boom)
        pd.testing.assert_frame_equal(expected, actual)

    def test_throwing_callback_is_logged_only_once(self, provider, caplog):
        """Ein defekter Rueckruf darf das Log nicht fluten."""
        calls = 0

        def boom(done: int, total: int) -> None:
            nonlocal calls
            calls += 1
            raise RuntimeError("Anzeige kaputt")

        with caplog.at_level("WARNING", logger="btlite.screening.universe"):
            build_ratio_universe(
                provider, self._many(provider), with_quotes=False, on_progress=boom
            )

        assert calls == 1, "Nach dem ersten Fehler darf nicht weiter gemeldet werden"
        flooded = [r for r in caplog.records if "Fortschritt" in r.message]
        assert len(flooded) == 1

    def test_empty_universe_reports_nothing(self, provider):
        calls: list[tuple[int, int]] = []
        result = build_ratio_universe(
            provider, [], with_quotes=False, on_progress=lambda d, t: calls.append((d, t))
        )
        assert result.empty
        assert calls in ([], [(0, 0)])

    def test_skipped_symbols_still_count_as_done(self, provider):
        """Der Balken misst den Fortschritt, nicht die Trefferzahl.

        Ein Universum aus lauter unbekannten Titeln wirft inzwischen -
        die Meldungen davor muessen trotzdem angekommen sein, sonst
        bliebe der Balken bei einem solchen Lauf einfach stehen.
        """
        calls: list[tuple[int, int]] = []
        tickers = [f"ZZZ{i}" for i in range(PROGRESS_EVERY * 2)]
        with pytest.raises(EmptyUniverseError):
            build_ratio_universe(
                provider, tickers, with_quotes=False, on_progress=lambda d, t: calls.append((d, t))
            )
        assert calls[-1] == (len(tickers), len(tickers))


class TestHistoriesProgress:
    """Dieselbe Zusage fuer die Schleife hinter den technischen Presets."""

    def test_without_callback_result_is_unchanged(self, provider):
        from btlite.ui.state import AppState

        state = AppState()
        tickers = provider.available_tickers()
        expected = state.histories(tickers, period="1y")
        assert state.histories(tickers, period="1y", on_progress=None).keys() == expected.keys()

    def test_last_call_is_total_total(self):
        from btlite.ui.state import AppState

        state = AppState()
        # Bewusst KEIN Vielfaches des Intervalls: sonst faellt der letzte
        # Aufruf mit einer Intervallmeldung zusammen und der Test wuerde
        # gruen bleiben, auch wenn die Schlussmeldung fehlt.
        tickers = (state.provider.available_tickers() * 3)[:23]
        calls: list[tuple[int, int]] = []
        state.histories(tickers, period="1mo", on_progress=lambda d, t: calls.append((d, t)))
        assert calls[-1] == (len(tickers), len(tickers))
        assert len(calls) == len(tickers)

    def test_throwing_callback_does_not_break_the_run(self):
        from btlite.ui.state import AppState

        state = AppState()
        tickers = state.provider.available_tickers()

        def boom(done: int, total: int) -> None:
            raise RuntimeError("Anzeige kaputt")

        expected = state.histories(tickers, period="1mo")
        actual = state.histories(tickers, period="1mo", on_progress=boom)
        assert actual.keys() == expected.keys()


class TestStopRequest:
    """Ein Abbruch muss die Schleifen wirklich anhalten.

    Der erste Entwurf schickte ihn als Ausnahme durch den
    Fortschritts-Rueckruf - und der schluckt Ausnahmen absichtlich, damit
    eine kaputte Anzeige den Lauf nicht abbricht. Beides zusammen hiess:
    der Abbruch wurde verschluckt und nebenbei die Anzeige stillgelegt.
    Gemessen lief ein abgebrochener Lauf sechzig statt drei Sekunden.

    Seitdem ist ``should_stop`` ein eigener Kanal.
    """

    def test_the_universe_build_stops(self, provider):
        tickers = provider.available_tickers()
        calls = {"n": 0}

        def stop() -> bool:
            calls["n"] += 1
            return calls["n"] > 3

        frame = build_ratio_universe(provider, tickers, should_stop=stop)
        assert len(frame) < len(tickers), "Der Aufbau lief durch"

    def test_what_was_gathered_is_kept(self, provider):
        """Ein Abbruch verwirft nicht, was schon da ist."""
        tickers = provider.available_tickers()
        calls = {"n": 0}

        def stop() -> bool:
            calls["n"] += 1
            return calls["n"] > 3

        frame = build_ratio_universe(provider, tickers, should_stop=stop)
        assert not frame.empty, "Alles verworfen"

    def test_without_the_predicate_everything_runs(self, provider):
        """Ohne Abbruch bleibt das Verhalten unveraendert."""
        tickers = provider.available_tickers()
        full = build_ratio_universe(provider, tickers)
        never = build_ratio_universe(provider, tickers, should_stop=lambda: False)
        assert len(never) == len(full)

    def test_the_history_loop_stops_too(self, provider):
        """Die technischen Presets laufen ueber eine andere Schleife.

        Sie nahm ``should_stop`` entgegen, benutzte es aber nicht - ein
        Mutationstest hat das aufgedeckt. Ohne diesen Test waere der
        Abbrechen-Knopf dort weiterhin wirkungslos.
        """
        from btlite.ui.state import AppState

        state = AppState()
        tickers = provider.available_tickers()
        calls = {"n": 0}

        def stop() -> bool:
            calls["n"] += 1
            return calls["n"] > 3

        out = state.histories(tickers, period="1y", should_stop=stop)
        assert len(out) < len(tickers), "Der Kursabruf lief durch"
