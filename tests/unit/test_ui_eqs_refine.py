"""Tests der Zusatzfilter auf der Seite EQS.

Gewuenscht: RSI im Screening und frei kombinierbare Filter wie "unter der
200-Tage-Linie und RSI unter 70". Entschieden: eine Filterzeile, die zu
jedem Preset per UND hinzukommt; bei fundamentalen Presets werden Kurse
nur bei aktivem Kursfilter geladen.

Die Offline-Daten sind bekannt: ``dividend`` trifft PG, XOM und JNJ mit
RSI 54, 57 und 66; unter ihrer 200-Tage-Linie liegt von den zehn Titeln
nur WMT.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import pytest

from btlite.config import Settings
from btlite.screening.refine import build_refinement
from btlite.ui.app import create_app, render_page
from btlite.ui.pages import eqs
from btlite.ui.state import AppState

if TYPE_CHECKING:
    from pathlib import Path

UNIVERSE = ["AAPL", "MSFT", "NVDA", "SPY", "JNJ", "XOM", "KO", "PG", "JPM", "WMT"]
_RSI_BELOW_70 = build_refinement(rsi_below=70)


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppState:
    monkeypatch.setenv("BTLITE_PORTFOLIO_HOLDINGS_FILE", str(tmp_path / "holdings.json"))
    return AppState(Settings())


def _callback(app: Any, marker: str, output: str = "") -> Any:
    """Holt die registrierte Funktion ueber ihren Ausloeser."""
    for key, entry in app.callback_map.items():
        if marker in str(entry.get("inputs", "")) and output in key:
            callback = entry["callback"]
            return getattr(callback, "__wrapped__", callback)
    raise AssertionError(f"Callback {marker} nicht gefunden")


def _matches(result: Any) -> set[str]:
    return {str(t) for t in result.data.index}


class TestFundamentalPresetWithPriceFilters:
    def test_rsi_below_narrows_the_preset(self, state: AppState) -> None:
        plain = eqs.run_screen(state, "dividend", UNIVERSE)
        refined = eqs.run_screen(
            state, "dividend", UNIVERSE, refinement=build_refinement(rsi_below=60)
        )
        assert _matches(plain) == {"PG", "XOM", "JNJ"}
        # JNJ hat RSI 66 und faellt heraus.
        assert _matches(refined) == {"PG", "XOM"}

    def test_the_requested_combination(self, state: AppState) -> None:
        """Unter der 200-Tage-Linie und RSI unter 70.

        Kein Dividendentitel liegt unter der Linie - die Liste ist leer,
        und das ist richtig.
        """
        refined = eqs.run_screen(
            state, "dividend", UNIVERSE, refinement=build_refinement("below", rsi_below=70)
        )
        assert _matches(refined) == set()

    def test_rsi_is_a_column_of_the_result(self, state: AppState) -> None:
        result = eqs.run_screen(
            state, "dividend", UNIVERSE, refinement=build_refinement(rsi_below=70)
        )
        assert "rsi_14" in result.data.columns
        assert result.data["rsi_14"].notna().all()
        assert (result.data["rsi_14"] < 70).all()

    def test_prices_load_only_for_preset_matches(
        self, state: AppState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Die Einsparung: Kurse nur fuer die Titel, die das Preset bestehen.

        Beim S&P 500 sind das wenige Dutzend statt 503 Abrufe. Das
        Ergebnis bleibt gleich, weil die Kursfilter per UND gelten.
        """
        asked: list[list[str]] = []
        original = state.histories

        def spy(tickers: list[str], *args: Any, **kwargs: Any) -> Any:
            asked.append(list(tickers))
            return original(tickers, *args, **kwargs)

        monkeypatch.setattr(state, "histories", spy)
        eqs.run_screen(state, "dividend", UNIVERSE, refinement=build_refinement(rsi_below=60))
        assert len(asked) == 1
        assert set(asked[0]) == {"PG", "XOM", "JNJ"}

    def test_without_price_filters_no_prices_are_loaded(
        self, state: AppState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ohne Kursfilter bleibt der Lauf so schnell wie vorher."""

        def forbidden(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("Kursreihen geladen, obwohl kein Kursfilter aktiv ist")

        monkeypatch.setattr(state, "histories", forbidden)
        eqs.run_screen(state, "dividend", UNIVERSE)
        eqs.run_screen(state, "dividend", UNIVERSE, refinement=build_refinement())

    def test_the_prefilter_gives_the_same_result_as_prices_for_all(self, state: AppState) -> None:
        """Gegenprobe ueber einen unabhaengigen Weg.

        Alle Titel mit Kursen, von Hand gefiltert: Preset-Treffer, deren
        RSI unter 60 liegt. Muss mit dem vorausgewaehlten Lauf
        uebereinstimmen.
        """
        from btlite.screening.technicals import build_technical_universe

        technical = build_technical_universe(state.histories(UNIVERSE, period="2y"))
        preset_hits = _matches(eqs.run_screen(state, "dividend", UNIVERSE))
        by_hand = {t for t in preset_hits if technical.loc[t, "rsi_14"] < 60}
        refined = eqs.run_screen(
            state, "dividend", UNIVERSE, refinement=build_refinement(rsi_below=60)
        )
        assert _matches(refined) == by_hand

    def test_no_preset_match_means_no_price_fetch_and_no_error(
        self, state: AppState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ohne Kandidaten gibt es nichts zu laden - kein 'Feld fehlt'."""
        monkeypatch.setattr(
            state,
            "histories",
            lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("nicht erwartet")),
        )
        tree = str(
            eqs.render_result(state, "dividend", ["WMT"], refinement=build_refinement(rsi_below=70))
        )
        assert "bt-error" not in tree
        assert "No title matches" in tree


class TestTechnicalPresetWithPriceFilters:
    def test_technical_screen_never_loads_fundamental_ratios(
        self, state: AppState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def forbidden(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("Technischer Screen darf keine Statements laden")

        monkeypatch.setattr(state, "ratio_universe", forbidden)
        result = eqs.run_screen(state, "above_200ma", UNIVERSE)
        assert not result.data.empty

    def test_rsi_filter_applies_on_top(self, state: AppState) -> None:
        plain = eqs.run_screen(state, "above_200ma", UNIVERSE)
        refined = eqs.run_screen(
            state, "above_200ma", UNIVERSE, refinement=build_refinement(rsi_below=60)
        )
        assert _matches(refined) < _matches(plain)
        assert (refined.data["rsi_14"] < 60).all()

    def test_below_50_day_line(self, state: AppState) -> None:
        """WMT ist der einzige Titel unter seiner 50-Tage-Linie."""
        refined = eqs.run_screen(
            state, "below_200ma", UNIVERSE, refinement=build_refinement(sma_50="below")
        )
        assert _matches(refined) == {"WMT"}


class TestRendering:
    def test_the_page_offers_the_four_filters(self, state: AppState) -> None:
        page = str(render_page(state, "EQS", ["AAPL", "MSFT"], ()))
        for component_id in eqs.REFINEMENT_INPUTS:
            assert component_id in page

    def test_active_filters_are_named_above_the_list(self, state: AppState) -> None:
        tree = str(
            eqs.render_result(
                state, "dividend", UNIVERSE, refinement=build_refinement("below", rsi_below=70)
            )
        )
        assert "Extra price filters" in tree
        assert "dist_sma_200 < 0.0" in tree
        assert "rsi_14 < 70.0" in tree

    def test_rsi_column_shows_for_a_refined_fundamental_preset(self, state: AppState) -> None:
        refined = str(eqs.render_result(state, "dividend", UNIVERSE, refinement=_RSI_BELOW_70))
        plain = str(eqs.render_result(state, "dividend", UNIVERSE))
        assert "RSI (14 days)" in refined
        assert "RSI (14 days)" not in plain

    def test_unrelated_price_columns_stay_out_of_the_table(self, state: AppState) -> None:
        """Nur Kurs, Abstaende und RSI - nicht alle 19 Kursspalten."""
        tree = str(eqs.render_result(state, "dividend", UNIVERSE, refinement=_RSI_BELOW_70))
        assert "Distance to 200-day MA" in tree
        assert "Avg volume (3 months)" not in tree.split("Column reference")[0]

    def test_run_prompt_mentions_the_second_step(self) -> None:
        refined = str(eqs._run_prompt(503, "dividend", build_refinement(rsi_below=70)))
        plain = str(eqs._run_prompt(503, "dividend"))
        technical = str(eqs._run_prompt(503, "above_200ma", build_refinement(rsi_below=70)))
        assert "price history for the titles that pass" in refined
        assert "price history" not in plain
        # Ein technisches Preset laedt die Kurse ohnehin - kein zweiter Schritt.
        assert "price history for the titles that pass" not in technical


class TestProgressOfTheSecondStep:
    def test_first_step_counts_titles(self) -> None:
        assert "Screening 100 of 503 titles" in str(eqs.progress_view(100, 503, "dividend", 503))

    def test_second_step_counts_its_own_matches(self) -> None:
        """Nicht "520 of 540 titles", wenn der Index 503 hat."""
        text = str(eqs.progress_view(520, 540, "dividend", 503))
        assert "17 of 37 matches" in text
        assert "of 540" not in text

    def test_without_title_count_it_stays_as_before(self) -> None:
        assert "Screening 137 of 503 titles" in str(eqs.progress_view(137, 503, "value"))


class TestCallbacks:
    def test_an_invalid_rsi_stops_before_the_run(self, state: AppState) -> None:
        """Sofort auffallen, nicht nach zwanzig Minuten Rechnen."""
        run = _callback(create_app(state.settings, state), "eqs-run")
        view, job_id, interval_off, cancel_hidden = run(
            1, "dividend", UNIVERSE, "any", "any", 70, 30
        )
        assert job_id is None
        assert interval_off is True
        assert cancel_hidden is True
        assert "leaves nothing" in str(view)

    def test_the_run_applies_the_filters(self, state: AppState) -> None:
        app = create_app(state.settings, state)
        run = _callback(app, "eqs-run")
        tick = _callback(app, eqs.PROGRESS_TICK, "eqs-result")
        _view, job_id, _off, _hidden = run(1, "dividend", UNIVERSE, "any", "any", None, 60)
        for index in range(400):
            out, _off, _hidden, job = tick(index, job_id, "dividend", UNIVERSE)
            if job is None:
                text = str(out)
                assert "PG" in text
                assert "XOM" in text
                assert "JNJ" not in text
                return
            time.sleep(0.05)
        raise AssertionError("Lauf endete nicht")

    def test_the_export_matches_the_screen(self, state: AppState) -> None:
        export = _callback(create_app(state.settings, state), "eqs-export")
        payload = export(1, "dividend", UNIVERSE, "any", "any", None, 60)
        lines = payload["content"].strip().splitlines()
        tickers = {line.split(",")[0] for line in lines[1:]}
        assert tickers == {"PG", "XOM"}

    def test_preset_change_carries_the_filters(self, state: AppState) -> None:
        preset = _callback(create_app(state.settings, state), "eqs-preset", "eqs-result")
        text = str(preset("dividend", UNIVERSE, "any", "any", None, 60))
        assert "rsi_14 < 60.0" in text

    def test_old_call_shape_still_works(self, state: AppState) -> None:
        """Ohne Filterwerte verhaelt sich alles wie vorher."""
        preset = _callback(create_app(state.settings, state), "eqs-preset", "eqs-result")
        text = str(preset("dividend", UNIVERSE))
        assert "Extra price filters" not in text
        assert "PG" in text


class TestCancelBeforeData:
    def test_cancel_before_anything_arrived_is_not_blamed_on_the_source(
        self, state: AppState
    ) -> None:
        tree = str(eqs.render_result(state, "value", UNIVERSE, should_stop=lambda: True))
        assert "cancelled" in tree
        assert "unreachable" not in tree


class TestSetupFilterOnThePage:
    """Chart-Setups als Filter auf der Seite.

    Offline: PG erfuellt Pullback und Support+MACD, JNJ nur Support+MACD,
    WMT die RSI-Erholung. ``dividend`` trifft PG, XOM und JNJ.
    """

    def test_pullback_narrows_the_dividend_preset(self, state: AppState) -> None:
        refined = eqs.run_screen(
            state, "dividend", UNIVERSE, refinement=build_refinement(setups=["pullback"])
        )
        assert _matches(refined) == {"PG"}

    def test_support_macd(self, state: AppState) -> None:
        refined = eqs.run_screen(
            state, "dividend", UNIVERSE, refinement=build_refinement(setups=["support_macd"])
        )
        assert _matches(refined) == {"PG", "JNJ"}

    def test_two_setups_must_both_hold(self, state: AppState) -> None:
        refined = eqs.run_screen(
            state,
            "dividend",
            UNIVERSE,
            refinement=build_refinement(setups=["pullback", "support_macd"]),
        )
        assert _matches(refined) == {"PG"}

    def test_on_a_technical_preset(self, state: AppState) -> None:
        refined = eqs.run_screen(
            state, "below_200ma", UNIVERSE, refinement=build_refinement(setups=["rsi_recovery"])
        )
        assert _matches(refined) == {"WMT"}

    def test_the_chosen_setup_is_a_readable_column(self, state: AppState) -> None:
        tree = str(
            eqs.render_result(
                state, "dividend", UNIVERSE, refinement=build_refinement(setups=["pullback"])
            )
        )
        assert "Setup: pullback" in tree
        assert "setup_pullback == 1.0" in tree

    def test_the_page_offers_the_setup_choice_with_gp_labels(self, state: AppState) -> None:
        from btlite.analytics.setups import SETUP_LABELS

        page = str(render_page(state, "EQS", ["AAPL", "MSFT"], ()))
        assert eqs.SETUP_INPUT in page
        for label in SETUP_LABELS.values():
            assert label in page
        assert "not a recommendation" in page

    def test_the_run_callback_takes_the_setups(self, state: AppState) -> None:
        export = _callback(create_app(state.settings, state), "eqs-export")
        payload = export(1, "dividend", UNIVERSE, "any", "any", None, None, ["pullback"])
        tickers = {line.split(",")[0] for line in payload["content"].strip().splitlines()[1:]}
        assert tickers == {"PG"}
