"""Tests der zusaetzlichen Kursfilter (``screening.refine``).

Gewuenscht war "unter der 200-Tage-Linie und RSI unter 70" als frei
kombinierbarer Filter zu jedem Preset. Geprueft wird die Uebersetzung in
Praedikate, die UND-Verknuepfung mit den Preset-Filtern, die
Fehlwert-Semantik und das Einlesen der Eingabefelder.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
from pydantic import ValidationError

from btlite.core.errors import ScreeningError
from btlite.models.screening import (
    FilterGroup,
    FilterPredicate,
    MaPosition,
    Operator,
    ScreenRequest,
    TechnicalRefinement,
)
from btlite.models.setups import SetupKey
from btlite.screening.filters import apply_filters
from btlite.screening.refine import (
    build_refinement,
    refine_request,
    refinement_columns,
    refinement_group,
    refinement_predicates,
    tickers_passing,
)


def _frame() -> pd.DataFrame:
    """Kleines Universum mit bekannten Lagen und RSI-Werten."""
    return pd.DataFrame(
        {
            "pe": [10.0, 30.0, 12.0, 14.0, 11.0, 13.0],
            "dist_sma_200": [-0.10, -0.05, 0.08, -0.02, math.nan, 0.0],
            "dist_sma_50": [-0.03, 0.02, 0.01, -0.01, 0.02, 0.0],
            "rsi_14": [45.0, 75.0, 60.0, 71.0, 40.0, 50.0],
        },
        index=pd.Index(["LOW", "HOT", "UP", "EDGE", "YOUNG", "FLAT"], name="ticker"),
    )


class TestPredicates:
    def test_nothing_set_means_nothing_to_filter(self) -> None:
        refinement = TechnicalRefinement()
        assert not refinement.is_active
        assert refinement_predicates(refinement) == []
        assert refinement_group(refinement) is None

    def test_below_means_negative_distance(self) -> None:
        """Der Abstand ist vorzeichenbehaftet: negativ heisst darunter."""
        (predicate,) = refinement_predicates(TechnicalRefinement(sma_200=MaPosition.BELOW))
        assert predicate == FilterPredicate(field="dist_sma_200", op=Operator.LT, value=0.0)

    def test_above_means_positive_distance(self) -> None:
        (predicate,) = refinement_predicates(TechnicalRefinement(sma_50=MaPosition.ABOVE))
        assert predicate == FilterPredicate(field="dist_sma_50", op=Operator.GT, value=0.0)

    def test_rsi_bounds_are_strict(self) -> None:
        """RSI unter 70 heisst strikt darunter."""
        predicates = refinement_predicates(TechnicalRefinement(rsi_above=30, rsi_below=70))
        assert predicates == [
            FilterPredicate(field="rsi_14", op=Operator.GT, value=30.0),
            FilterPredicate(field="rsi_14", op=Operator.LT, value=70.0),
        ]


class TestTheRequestedCombination:
    """Die Abfrage aus der Anforderung: unter 200-Tage-Linie, RSI unter 70."""

    def test_finds_exactly_the_matching_titles(self) -> None:
        group = refinement_group(TechnicalRefinement(sma_200=MaPosition.BELOW, rsi_below=70))
        matched = set(apply_filters(_frame(), group).index)
        # LOW: darunter, RSI 45. HOT: darunter, aber RSI 75. EDGE: RSI 71.
        # UP liegt darueber, FLAT genau auf der Linie, YOUNG ohne Linie.
        assert matched == {"LOW"}

    def test_a_title_without_a_200_day_line_is_neither_above_nor_below(self) -> None:
        """Zu kurze Historie ist keine Aussage - sie erfuellt keine Richtung."""
        below = apply_filters(_frame(), refinement_group(TechnicalRefinement(sma_200="below")))
        above = apply_filters(_frame(), refinement_group(TechnicalRefinement(sma_200="above")))
        assert "YOUNG" not in below.index
        assert "YOUNG" not in above.index

    def test_exactly_on_the_line_counts_as_neither(self) -> None:
        below = apply_filters(_frame(), refinement_group(TechnicalRefinement(sma_200="below")))
        above = apply_filters(_frame(), refinement_group(TechnicalRefinement(sma_200="above")))
        assert "FLAT" not in below.index
        assert "FLAT" not in above.index


class TestRefineRequest:
    def test_without_refinement_the_request_is_untouched(self) -> None:
        request = ScreenRequest(name="x")
        assert refine_request(request, None) is request
        assert refine_request(request, TechnicalRefinement()) is request

    def test_preset_filters_and_extra_filters_both_apply(self) -> None:
        """UND, nicht ODER: das Preset bleibt in Kraft."""
        preset_filters = FilterGroup(
            predicates=[FilterPredicate(field="pe", op=Operator.LT, value=13.0)]
        )
        request = ScreenRequest(name="value", local_filters=preset_filters)
        refined = refine_request(request, TechnicalRefinement(sma_200="below", rsi_below=70))

        matched = set(apply_filters(_frame(), refined.local_filters).index)
        # Das Preset allein liesse LOW, UP, YOUNG (pe < 13) durch; die
        # Kursfilter lassen davon nur LOW uebrig.
        assert matched == {"LOW"}

    def test_the_preset_request_is_not_mutated(self) -> None:
        preset_filters = FilterGroup(
            predicates=[FilterPredicate(field="pe", op=Operator.LT, value=13.0)]
        )
        request = ScreenRequest(name="value", local_filters=preset_filters)
        before = request.model_dump()
        refine_request(request, TechnicalRefinement(rsi_below=70))
        assert request.model_dump() == before

    def test_a_request_without_own_filters_gets_just_the_extra_ones(self) -> None:
        refined = refine_request(ScreenRequest(name="x"), TechnicalRefinement(rsi_below=50))
        assert set(apply_filters(_frame(), refined.local_filters).index) == {"LOW", "YOUNG"}


class TestBuildRefinement:
    """Einlesen der Eingabefelder - hier kommt Nutzertext an."""

    def test_empty_fields_mean_inactive(self) -> None:
        assert not build_refinement(None, None, None, None).is_active
        assert not build_refinement("", "", "", "  ").is_active
        assert not build_refinement("any", "any").is_active

    def test_values_are_taken_over(self) -> None:
        refinement = build_refinement("below", "above", 30, "70")
        assert refinement.sma_200 is MaPosition.BELOW
        assert refinement.sma_50 is MaPosition.ABOVE
        assert refinement.rsi_above == 30.0
        assert refinement.rsi_below == 70.0

    @pytest.mark.parametrize("raw", ["abc", [70]])
    def test_non_numbers_are_a_readable_error(self, raw: object) -> None:
        with pytest.raises(ScreeningError, match="between 0 and 100"):
            build_refinement(rsi_below=raw)

    @pytest.mark.parametrize("raw", [-1, 100.5, 150])
    def test_values_off_the_scale_are_rejected(self, raw: float) -> None:
        with pytest.raises(ScreeningError, match="between 0 and 100"):
            build_refinement(rsi_above=raw)

    @pytest.mark.parametrize(("above", "below"), [(70, 30), (50, 50)])
    def test_an_empty_range_is_rejected_before_it_empties_the_list(
        self, above: float, below: float
    ) -> None:
        """Sonst saehe der Nutzer eine leere Liste ohne Grund."""
        with pytest.raises(ScreeningError, match="leaves nothing"):
            build_refinement(rsi_above=above, rsi_below=below)

    def test_an_unknown_ma_setting_is_rejected(self) -> None:
        with pytest.raises(ScreeningError, match="moving-average"):
            build_refinement(sma_200="sideways")

    def test_the_model_itself_guards_the_range_too(self) -> None:
        """Wer das Modell direkt baut, umgeht build_refinement nicht."""
        with pytest.raises(ValidationError):
            TechnicalRefinement(rsi_above=70, rsi_below=30)
        with pytest.raises(ValidationError):
            TechnicalRefinement(rsi_below=101)


class TestTickersPassing:
    def test_lists_titles_that_pass_in_frame_order(self) -> None:
        group = FilterGroup(predicates=[FilterPredicate(field="pe", op=Operator.LT, value=13.0)])
        assert tickers_passing(_frame(), group) == ["LOW", "UP", "YOUNG"]

    def test_no_filters_pass_everyone(self) -> None:
        assert tickers_passing(_frame(), None) == list(_frame().index)


class TestSetupFilter:
    """Chart-Setups als Zusatzfilter: alle gewaehlten muessen erfuellt sein."""

    @staticmethod
    def _setups() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "setup_pullback": [1.0, 1.0, 0.0, math.nan],
                "setup_breakout": [1.0, 0.0, 1.0, 1.0],
            },
            index=pd.Index(["BOTH", "PULL", "BREAK", "YOUNG"], name="ticker"),
        )

    def test_a_chosen_setup_must_be_met(self) -> None:
        group = refinement_group(build_refinement(setups=["pullback"]))
        assert set(apply_filters(self._setups(), group).index) == {"BOTH", "PULL"}

    def test_several_setups_combine_with_and(self) -> None:
        group = refinement_group(build_refinement(setups=["pullback", "breakout"]))
        assert set(apply_filters(self._setups(), group).index) == {"BOTH"}

    def test_insufficient_history_is_not_met(self) -> None:
        """YOUNG hat keinen Pullback-Wert - unbekannt ist nicht erfuellt."""
        group = refinement_group(build_refinement(setups=["pullback"]))
        assert "YOUNG" not in apply_filters(self._setups(), group).index

    def test_setups_alone_make_the_refinement_active(self) -> None:
        assert build_refinement(setups=["breakout"]).is_active
        assert not build_refinement(setups=[]).is_active

    def test_a_single_key_and_duplicates_are_accepted(self) -> None:
        assert build_refinement(setups="pullback").setups == [SetupKey.PULLBACK]
        assert build_refinement(setups=["pullback", "pullback"]).setups == [SetupKey.PULLBACK]

    def test_an_unknown_setup_is_a_readable_error(self) -> None:
        with pytest.raises(ScreeningError, match="Unknown chart setup"):
            build_refinement(setups=["moonshot"])

    def test_chosen_setup_columns_become_visible(self) -> None:
        columns = refinement_columns(build_refinement(setups=["breakout"]))
        assert "setup_breakout" in columns
        assert "setup_pullback" not in columns
