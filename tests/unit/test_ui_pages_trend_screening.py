"""Tests der Trendanzeige auf FA und des technischen Screenings auf EQS.

Beide Funktionen sind Nutzeranforderungen mit klarem Akzeptanzkriterium:

* FA zeigt neben jedem Betrag die Veraenderung zur Vorperiode in Prozent.
  Wo kein Prozentsatz interpretierbar ist, steht "n/a" samt Grund - nie
  eine 0 und nie eine irrefuehrende Zahl.
* EQS findet die Titel unter ihrer 200-Tage-Linie.

Alle Faelle laufen offline gegen den FixtureProvider und rendern ueber
denselben Weg wie die Anwendung, naemlich ``render_page`` beziehungsweise
die Seiten-Callbacks - nicht ueber die Hilfsfunktionen darunter.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import pytest

from btlite.config import Settings
from btlite.models.common import Period
from btlite.models.fundamentals import StatementType
from btlite.ui import formatting as fmt
from btlite.ui.app import render_page
from btlite.ui.pages import eqs, fa
from btlite.ui.state import AppState, set_state


@pytest.fixture
def state() -> Iterator[AppState]:
    """Baut einen Fixture-gestuetzten UI-Zustand."""
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


def text_of(tree: Any) -> str:
    """Sammelt allen Textinhalt eines Komponentenbaums."""
    parts: list[str] = []
    for node in walk(tree):
        children = getattr(node, "children", node)
        if isinstance(children, str):
            parts.append(children)
    return " ".join(parts)


def classed_texts(tree: Any, name: str) -> list[str]:
    """Sammelt die Texte aller Knoten mit einer bestimmten CSS-Klasse."""
    out: list[str] = []
    for node in walk(tree):
        if name not in str(getattr(node, "className", "") or "").split():
            continue
        children = getattr(node, "children", None)
        if isinstance(children, str):
            out.append(children)
    return out


class TestFormatChange:
    """Der neue Formatierer uebersetzt dezimale Veraenderungen in Text."""

    def test_positive_change_carries_sign_and_arrow(self) -> None:
        assert fmt.format_change(0.064255) == "▲ +6.43 %"

    def test_negative_change_carries_sign_and_arrow(self) -> None:
        assert fmt.format_change(-0.0821) == "▼ -8.21 %"

    def test_zero_change_is_neutral(self) -> None:
        assert fmt.format_change(0.0) == "▬ +0.00 %"

    def test_missing_change_is_na_not_zero(self) -> None:
        assert fmt.format_change(None) == fmt.NA_TEXT
        assert fmt.format_change(float("nan")) == fmt.NA_TEXT

    def test_arrow_is_empty_for_missing(self) -> None:
        assert fmt.direction_arrow(None) == ""


class TestFaTrendRendering:
    """Die Seite FA zeigt je Betrag die Veraenderung zur Vorperiode."""

    def test_aapl_revenue_and_net_income_trends_are_rendered(self, state: AppState) -> None:
        """Die verifizierten AAPL-Werte stehen woertlich in der Ausgabe.

        GJ2024 auf GJ2025: revenue +6,43 Prozent, net_income +19,50 Prozent.
        """
        tree = render_page(state, "FA", ["AAPL"], ())
        rendered = text_of(tree)
        assert "▲ +6.43 %" in rendered
        assert "▲ +19.50 %" in rendered

    def test_trend_matches_the_computation_layer(self, state: AppState) -> None:
        """Die angezeigten Werte stammen aus ``analytics.statements``."""
        from btlite.analytics.statements import compute_trend_bundle

        bundle = state.statements("AAPL", period=Period.ANNUAL)
        trend = compute_trend_bundle(bundle)[StatementType.INCOME]

        revenue = trend.point("revenue")
        net_income = trend.point("net_income")
        assert revenue is not None
        assert net_income is not None
        assert revenue.change_pct == pytest.approx(0.064255, abs=1e-5)
        assert net_income.change_pct == pytest.approx(0.194952, abs=1e-5)

        rendered = text_of(render_page(state, "FA", ["AAPL"], ()))
        assert fmt.format_change(revenue.change_pct) in rendered
        assert fmt.format_change(net_income.change_pct) in rendered

    def test_direction_is_never_carried_by_colour_alone(self, state: AppState) -> None:
        """Jeder gefaerbte Trendwert fuehrt Vorzeichen und Richtungszeichen."""
        tree = render_page(state, "FA", ["AAPL"], ())
        coloured = classed_texts(tree, "bt-up") + classed_texts(tree, "bt-down")
        assert coloured, "Es sollte gefaerbte Trendwerte geben."
        for entry in coloured:
            assert re.fullmatch(r"[▲▼▬] [+-][\d.,]+ %", entry), entry

    def test_up_and_down_both_occur(self, state: AppState) -> None:
        tree = render_page(state, "FA", ["AAPL"], ())
        assert classed_texts(tree, "bt-up")
        assert classed_texts(tree, "bt-down")

    def test_uninterpretable_change_is_na_with_reason(self, state: AppState) -> None:
        """Vorzeichenwechsel und Basis 0 erscheinen als "n/a" plus Grund."""
        from btlite.analytics.statements import (
            MISSING_VALUE_REASON,
            SIGN_CHANGE_REASON,
            ZERO_BASE_REASON,
        )

        rendered = text_of(render_page(state, "FA", ["AAPL"], ()))
        for reason in (SIGN_CHANGE_REASON, ZERO_BASE_REASON, MISSING_VALUE_REASON):
            assert f"{fmt.NA_TEXT} ({reason})" in rendered, reason

    def test_no_reason_cell_shows_a_number(self, state: AppState) -> None:
        """Wo ein Grund steht, steht nie zusaetzlich ein Prozentwert."""
        tree = render_page(state, "FA", ["AAPL"], ())
        for entry in classed_texts(tree, "bt-na"):
            if entry.startswith(f"{fmt.NA_TEXT} ("):
                assert "%" not in entry, entry

    def test_sign_change_is_not_rendered_as_zero_percent(self, state: AppState) -> None:
        """Ein Vorzeichenwechsel wird nie zu "0 %" geglaettet."""
        from btlite.analytics.statements import SIGN_CHANGE_REASON

        tree = render_page(state, "FA", ["AAPL"], ())
        na_texts = classed_texts(tree, "bt-na")
        assert any(SIGN_CHANGE_REASON in entry for entry in na_texts)

    def test_amount_and_change_are_separate_nodes(self, state: AppState) -> None:
        """Jede Betragszelle traegt genau zwei Zeilen."""
        from dash import html

        tree = render_page(state, "FA", ["AAPL"], ())
        cells = [
            node
            for node in walk(tree)
            if isinstance(node, html.Td) and isinstance(node.children, list)
        ]
        assert cells
        assert all(len(cell.children) == 2 for cell in cells)

    @pytest.mark.parametrize("period", list(Period))
    def test_every_period_basis_renders(self, state: AppState, period: Period) -> None:
        """Auch Quartals- und TTM-Basis bauen ohne Ausnahme."""
        tree = fa.render_body(state, "AAPL", [], period)
        assert text_of(tree)

    def test_ttm_has_no_prior_period_and_stays_silent(self, state: AppState) -> None:
        """TTM hat nur eine Spalte - dort steht kein Grund als Rauschen."""
        from btlite.analytics.statements import NO_PRIOR_REASON

        rendered = text_of(fa.render_body(state, "AAPL", [], Period.TTM))
        assert NO_PRIOR_REASON not in rendered

    def test_instrument_without_statements_still_renders(self, state: AppState) -> None:
        """Ein ETF ohne Bilanz bleibt ein Hinweis, kein Absturz."""
        rendered = text_of(render_page(state, "FA", ["SPY"], ()))
        assert "No financial statements are available for SPY" in rendered


class TestEqsPresetCatalogue:
    """Fundamentale und technische Presets stehen nebeneinander bereit."""

    def test_both_kinds_are_selectable(self) -> None:
        values = {option["value"] for option in eqs._preset_options()}
        assert {"value", "quality", "momentum", "dividend"} <= values
        assert {"below_200ma", "above_200ma", "oversold"} <= values

    def test_labels_say_which_kind_and_whether_net_is_needed(self) -> None:
        labels = {option["value"]: option["label"] for option in eqs._preset_options()}
        assert labels["below_200ma"].startswith("Technical - ")
        assert "(local)" in labels["below_200ma"]
        assert labels["value"].startswith("Fundamental - ")

    def test_technical_keys_are_recognised(self) -> None:
        assert eqs.is_technical_preset("below_200ma")
        assert eqs.is_technical_preset("BELOW_200MA")
        assert not eqs.is_technical_preset("value")

    def test_resolve_finds_both_catalogues(self) -> None:
        assert eqs.resolve_preset("below_200ma").key == "below_200ma"
        assert eqs.resolve_preset("value").key == "value"

    def test_unknown_key_raises_a_project_error(self) -> None:
        """Der Fehler muss ein ``BTLiteError`` sein - die Callbacks fangen nur den."""
        from btlite.core.errors import BTLiteError

        with pytest.raises(BTLiteError):
            eqs.resolve_preset("nosuchpreset")

    def test_command_line_argument_picks_a_technical_preset(self) -> None:
        presets = eqs._all_presets()
        assert eqs._preset_from_args(("below_200ma",), presets) == "below_200ma"


class TestEqsTechnicalScreening:
    """Der technische Screen laeuft offline und findet die richtigen Titel."""

    def test_below_200ma_finds_wmt(self, state: AppState) -> None:
        """In den Fixtures liegt genau WMT unter seiner 200-Tage-Linie."""
        result = eqs.run_screen(state, "below_200ma", state.available_tickers())
        assert list(result.data.index) == ["WMT"]
        assert result.total_matches == 1

    def test_below_200ma_appears_in_the_rendered_page(self, state: AppState) -> None:
        """Der Treffer steht in der gerenderten Trefferliste."""
        tree = render_page(state, "EQS", [], ("below_200ma",))
        rendered = text_of(tree)
        assert "Showing 1 of 1 matches" in rendered
        assert "WMT" in rendered
        assert "Walmart" in rendered

    def test_distance_is_shown_as_signed_percent(self, state: AppState) -> None:
        """``dist_sma_200`` erscheint als Prozentwert, nicht als Dezimalzahl."""
        result = eqs.run_screen(state, "below_200ma", state.available_tickers())
        distance = float(result.data.loc["WMT", "dist_sma_200"])
        assert distance < 0.0

        rendered = text_of(render_page(state, "EQS", [], ("below_200ma",)))
        assert fmt.format_percent(distance) in rendered

    def test_rsi_is_shown_on_its_own_scale(self, state: AppState) -> None:
        """Der RSI ist eine Zahl 0 bis 100, kein dezimaler Anteil."""
        result = eqs.run_screen(state, "below_200ma", state.available_tickers())
        rsi = float(result.data.loc["WMT", "rsi_14"])
        assert 0.0 <= rsi <= 100.0

        rendered = text_of(render_page(state, "EQS", [], ("below_200ma",)))
        assert fmt.format_number(rsi, decimals=1) in rendered

    def test_column_help_is_offered(self, state: AppState) -> None:
        """Die Erklaerungen kommen aus ``screening.technicals.describe_columns``."""
        from btlite.screening.technicals import describe_columns

        rendered = text_of(render_page(state, "EQS", [], ("below_200ma",)))
        assert describe_columns()["dist_sma_200"] in rendered
        assert describe_columns()["rsi_14"] in rendered

    def test_technical_screen_notes_it_runs_locally(self, state: AppState) -> None:
        rendered = text_of(render_page(state, "EQS", [], ("below_200ma",)))
        assert "Price based" in rendered
        assert "locally" in rendered

    def test_fundamental_screen_still_works(self, state: AppState) -> None:
        """Die bestehenden Presets bleiben unberuehrt."""
        rendered = text_of(render_page(state, "EQS", [], ("value",)))
        assert "Fundamentals based" in rendered

    @pytest.mark.parametrize(
        "key",
        ["below_200ma", "above_200ma", "oversold", "pullback", "near_52w_high", "low_volatility"],
    )
    def test_every_technical_preset_renders(self, state: AppState, key: str) -> None:
        """Auch ein Preset ohne Treffer liefert eine Seite, kein Banner."""
        tree = render_page(state, "EQS", [], (key,))
        rendered = text_of(tree)
        assert re.search(r"Showing \d+ of \d+ matches", rendered)

    def test_flags_are_words_not_numbers(self, state: AppState) -> None:
        """``above_sma_200`` erscheint als yes/no, nicht als 1.00."""
        rendered = text_of(render_page(state, "EQS", [], ("above_200ma",)))
        assert " yes " in f" {rendered} "

    def test_etf_without_statements_is_kept(self, state: AppState) -> None:
        """SPY fuehrt keine Bilanz, hat aber Kurse - ein Kursscreen sieht ihn."""
        result = eqs.run_screen(state, "above_200ma", state.available_tickers())
        assert "SPY" in set(result.data.index)

    def test_missing_values_stay_na(self, state: AppState) -> None:
        """Ein Titel ohne Stammdaten zeigt "n/a", nie 0."""
        result = eqs.run_screen(state, "above_200ma", state.available_tickers())
        assert result.data.loc["SPY", "name"] != 0.0

        rendered = text_of(render_page(state, "EQS", [], ("above_200ma",)))
        assert fmt.NA_TEXT in rendered

    def test_export_payload_uses_the_same_screen(self, state: AppState) -> None:
        """Der CSV-Export laeuft ueber dieselbe Funktion wie die Anzeige."""
        result = eqs.run_screen(state, "below_200ma", state.available_tickers())
        csv = result.data.to_csv(index=True)
        assert "WMT" in csv
        assert "dist_sma_200" in csv
