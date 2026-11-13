"""Tests der lokalen Filter-DSL.

Getestet wird direkt auf selbst gebauten DataFrames. ``FixtureProvider.screen``
ignoriert die Finviz-Filter und liefert immer dasselbe eingefrorene Ergebnis -
die lokale Filterlogik laesst sich daran nicht pruefen.

Der Schwerpunkt liegt auf der NaN-Semantik: ein fehlender Wert erfuellt kein
Vergleichspraedikat, auch kein negiertes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btlite.core.errors import ScreeningError, UnknownFilterError
from btlite.models.screening import FilterGroup, FilterPredicate, LogicalOp, Operator
from btlite.screening.filters import (
    apply_filters,
    build_mask,
    describe_group,
    describe_predicate,
    evaluate_group,
    evaluate_predicate,
)


@pytest.fixture
def universe() -> pd.DataFrame:
    """Universum mit Luecken in jeder relevanten Spalte."""
    return pd.DataFrame(
        {
            "pe": [10.0, 15.0, 25.0, np.nan, 5.0],
            "roe": [0.20, 0.05, np.nan, 0.30, -0.10],
            "sector": ["Technology", "Energy", "Technology", None, "Utilities"],
        },
        index=pd.Index(["AAA", "BBB", "CCC", "DDD", "EEE"], name="ticker"),
    )


def _hits(frame: pd.DataFrame, predicate: FilterPredicate) -> list[str]:
    """Werte ein Praedikat aus und gib die getroffenen Ticker zurueck."""
    mask = evaluate_predicate(frame, predicate)
    return list(frame.index[mask])


# ------------------------------------------------------------ Operatoren --


def test_gt(universe):
    hits = _hits(universe, FilterPredicate(field="pe", op=Operator.GT, value=10.0))
    assert hits == ["BBB", "CCC"]


def test_gte(universe):
    hits = _hits(universe, FilterPredicate(field="pe", op=Operator.GTE, value=10.0))
    assert hits == ["AAA", "BBB", "CCC"]


def test_lt(universe):
    hits = _hits(universe, FilterPredicate(field="pe", op=Operator.LT, value=15.0))
    assert hits == ["AAA", "EEE"]


def test_lte(universe):
    hits = _hits(universe, FilterPredicate(field="pe", op=Operator.LTE, value=15.0))
    assert hits == ["AAA", "BBB", "EEE"]


def test_eq_numeric(universe):
    hits = _hits(universe, FilterPredicate(field="pe", op=Operator.EQ, value=15.0))
    assert hits == ["BBB"]


def test_eq_string(universe):
    hits = _hits(universe, FilterPredicate(field="sector", op=Operator.EQ, value="Technology"))
    assert hits == ["AAA", "CCC"]


def test_neq_string(universe):
    """NEQ heisst 'bekannt und ungleich' - DDD hat keinen Sektor und faellt raus."""
    hits = _hits(universe, FilterPredicate(field="sector", op=Operator.NEQ, value="Technology"))
    assert hits == ["BBB", "EEE"]


def test_between_inclusive(universe):
    hits = _hits(
        universe, FilterPredicate(field="pe", op=Operator.BETWEEN, value=10.0, value2=15.0)
    )
    assert hits == ["AAA", "BBB"]


def test_between_rejects_inverted_bounds(universe):
    predicate = FilterPredicate(field="pe", op=Operator.BETWEEN, value=20.0, value2=5.0)
    with pytest.raises(ScreeningError, match="BETWEEN has its bounds swapped"):
        evaluate_predicate(universe, predicate)


def test_in(universe):
    hits = _hits(
        universe,
        FilterPredicate(field="sector", op=Operator.IN, values=["Technology", "Utilities"]),
    )
    assert hits == ["AAA", "CCC", "EEE"]


def test_not_in(universe):
    """NOT_IN schliesst Fehlwerte aus - DDD taucht nicht auf."""
    hits = _hits(
        universe,
        FilterPredicate(field="sector", op=Operator.NOT_IN, values=["Technology"]),
    )
    assert hits == ["BBB", "EEE"]


def test_is_null(universe):
    hits = _hits(universe, FilterPredicate(field="pe", op=Operator.IS_NULL))
    assert hits == ["DDD"]


def test_not_null(universe):
    hits = _hits(universe, FilterPredicate(field="roe", op=Operator.NOT_NULL))
    assert hits == ["AAA", "BBB", "DDD", "EEE"]


# --------------------------------------------------------- NaN-Semantik --


@pytest.mark.parametrize(
    "op",
    [
        Operator.GT,
        Operator.GTE,
        Operator.LT,
        Operator.LTE,
        Operator.EQ,
        Operator.NEQ,
    ],
)
def test_nan_never_satisfies_a_comparison(universe, op):
    """Kein Vergleichsoperator trifft die Zeile ohne pe-Wert."""
    mask = evaluate_predicate(universe, FilterPredicate(field="pe", op=op, value=15.0))
    assert not bool(mask.loc["DDD"])


@pytest.mark.parametrize(
    ("op", "kwargs"),
    [
        (Operator.BETWEEN, {"value": 0.0, "value2": 100.0}),
        (Operator.IN, {"values": [10.0, 15.0]}),
        (Operator.NOT_IN, {"values": [10.0, 15.0]}),
    ],
)
def test_nan_never_satisfies_set_operators(universe, op, kwargs):
    mask = evaluate_predicate(universe, FilterPredicate(field="pe", op=op, **kwargs))
    assert not bool(mask.loc["DDD"])


def test_is_null_and_not_null_are_complements(universe):
    is_null = evaluate_predicate(universe, FilterPredicate(field="roe", op=Operator.IS_NULL))
    not_null = evaluate_predicate(universe, FilterPredicate(field="roe", op=Operator.NOT_NULL))
    assert not (is_null & not_null).any()
    assert (is_null | not_null).all()


def test_eq_and_neq_are_not_complements_on_missing_values(universe):
    """Beide sind falsch, wo der Wert fehlt - das ist die dokumentierte Regel."""
    eq = evaluate_predicate(
        universe, FilterPredicate(field="sector", op=Operator.EQ, value="Technology")
    )
    neq = evaluate_predicate(
        universe, FilterPredicate(field="sector", op=Operator.NEQ, value="Technology")
    )
    assert not bool(eq.loc["DDD"])
    assert not bool(neq.loc["DDD"])
    assert not (eq | neq).all()


def test_negate_group_reintroduces_missing_values(universe):
    """Negate invertiert die fertige Maske und holt Fehlwerte damit herein."""
    group = FilterGroup(
        predicates=[FilterPredicate(field="sector", op=Operator.EQ, value="Technology")],
        negate=True,
    )
    hits = list(universe.index[evaluate_group(universe, group)])
    assert hits == ["BBB", "DDD", "EEE"]


def test_masks_carry_no_missing_values(universe):
    for op in (Operator.GT, Operator.NEQ, Operator.IS_NULL, Operator.NOT_NULL):
        kwargs = {} if op in (Operator.IS_NULL, Operator.NOT_NULL) else {"value": 10.0}
        mask = evaluate_predicate(universe, FilterPredicate(field="pe", op=op, **kwargs))
        assert mask.dtype == bool
        assert not mask.isna().any()


def test_incomparable_types_do_not_raise(universe):
    """Ein Vergleich von String gegen Zahl gilt als nicht erfuellt."""
    mask = evaluate_predicate(universe, FilterPredicate(field="sector", op=Operator.GT, value=5.0))
    assert not mask.any()


# ------------------------------------------------------------- Gruppen ---


def test_group_and(universe):
    group = FilterGroup(
        op=LogicalOp.AND,
        predicates=[
            FilterPredicate(field="pe", op=Operator.LT, value=20.0),
            FilterPredicate(field="roe", op=Operator.GT, value=0.0),
        ],
    )
    assert list(universe.index[evaluate_group(universe, group)]) == ["AAA", "BBB"]


def test_group_or(universe):
    group = FilterGroup(
        op=LogicalOp.OR,
        predicates=[
            FilterPredicate(field="pe", op=Operator.GT, value=20.0),
            FilterPredicate(field="roe", op=Operator.GTE, value=0.30),
        ],
    )
    assert list(universe.index[evaluate_group(universe, group)]) == ["CCC", "DDD"]


def test_nested_groups(universe):
    """(pe < 20 UND roe > 0.1) ODER sector == Utilities."""
    group = FilterGroup(
        op=LogicalOp.OR,
        groups=[
            FilterGroup(
                op=LogicalOp.AND,
                predicates=[
                    FilterPredicate(field="pe", op=Operator.LT, value=20.0),
                    FilterPredicate(field="roe", op=Operator.GT, value=0.10),
                ],
            ),
            FilterGroup(
                predicates=[
                    FilterPredicate(field="sector", op=Operator.EQ, value="Utilities"),
                ],
            ),
        ],
    )
    assert list(universe.index[evaluate_group(universe, group)]) == ["AAA", "EEE"]


def test_nested_negate(universe):
    """NICHT (sector in [Technology]) UND pe vorhanden."""
    group = FilterGroup(
        op=LogicalOp.AND,
        predicates=[FilterPredicate(field="pe", op=Operator.NOT_NULL)],
        groups=[
            FilterGroup(
                predicates=[
                    FilterPredicate(field="sector", op=Operator.IN, values=["Technology"]),
                ],
                negate=True,
            ),
        ],
    )
    assert list(universe.index[evaluate_group(universe, group)]) == ["BBB", "EEE"]


def test_mixed_predicates_and_groups_use_same_op(universe):
    group = FilterGroup(
        op=LogicalOp.OR,
        predicates=[FilterPredicate(field="pe", op=Operator.EQ, value=5.0)],
        groups=[
            FilterGroup(predicates=[FilterPredicate(field="pe", op=Operator.EQ, value=25.0)]),
        ],
    )
    assert list(universe.index[evaluate_group(universe, group)]) == ["CCC", "EEE"]


def test_deeply_nested_group(universe):
    group = FilterGroup(
        op=LogicalOp.AND,
        groups=[
            FilterGroup(
                op=LogicalOp.OR,
                groups=[
                    FilterGroup(
                        predicates=[FilterPredicate(field="pe", op=Operator.LT, value=12.0)]
                    ),
                    FilterGroup(
                        predicates=[FilterPredicate(field="pe", op=Operator.GT, value=20.0)]
                    ),
                ],
            ),
        ],
    )
    assert list(universe.index[evaluate_group(universe, group)]) == ["AAA", "CCC", "EEE"]


# --------------------------------------------------- unbekannte Spalten --


def test_unknown_column_raises_with_suggestion(universe):
    predicate = FilterPredicate(field="p_e", op=Operator.LT, value=15.0)
    with pytest.raises(UnknownFilterError) as excinfo:
        evaluate_predicate(universe, predicate)
    assert "p_e" in excinfo.value.user_message
    assert excinfo.value.detail is not None
    assert "pe" in excinfo.value.detail


def test_unknown_column_without_close_match_lists_columns(universe):
    predicate = FilterPredicate(field="zzzzzzzz", op=Operator.LT, value=1.0)
    with pytest.raises(UnknownFilterError) as excinfo:
        evaluate_predicate(universe, predicate)
    assert excinfo.value.detail is not None
    assert "vorhandene Spalten" in excinfo.value.detail


def test_unknown_column_is_not_a_silent_empty_result(universe):
    group = FilterGroup(predicates=[FilterPredicate(field="nope", op=Operator.GT, value=1.0)])
    with pytest.raises(UnknownFilterError):
        apply_filters(universe, group)


def test_duplicate_column_raises():
    frame = pd.DataFrame([[1.0, 2.0]], columns=["pe", "pe"], index=pd.Index(["AAA"], name="ticker"))
    with pytest.raises(ScreeningError, match=r"Field 'pe' occurs more than once in the universe"):
        evaluate_predicate(frame, FilterPredicate(field="pe", op=Operator.GT, value=0.0))


# -------------------------------------------------------- apply/build ----


def test_build_mask_without_filters_matches_all(universe):
    mask = build_mask(universe, None)
    assert mask.all()
    assert len(mask) == len(universe)


def test_apply_filters_without_filters_returns_all(universe):
    out = apply_filters(universe, None)
    assert list(out.index) == list(universe.index)


def test_apply_filters_does_not_mutate_input(universe):
    before = universe.copy()
    group = FilterGroup(predicates=[FilterPredicate(field="pe", op=Operator.LT, value=12.0)])
    apply_filters(universe, group)
    pd.testing.assert_frame_equal(universe, before)


def test_apply_filters_preserves_columns_and_order(universe):
    group = FilterGroup(predicates=[FilterPredicate(field="pe", op=Operator.NOT_NULL)])
    out = apply_filters(universe, group)
    assert list(out.columns) == list(universe.columns)
    assert list(out.index) == ["AAA", "BBB", "CCC", "EEE"]


def test_empty_frame_yields_empty_mask():
    frame = pd.DataFrame({"pe": pd.Series(dtype=float)}, index=pd.Index([], name="ticker"))
    group = FilterGroup(predicates=[FilterPredicate(field="pe", op=Operator.LT, value=15.0)])
    out = apply_filters(frame, group)
    assert out.empty


def test_none_value_counts_as_missing():
    frame = pd.DataFrame({"pe": [10.0, None]}, index=pd.Index(["AAA", "BBB"], name="ticker"))
    hits = _hits(frame, FilterPredicate(field="pe", op=Operator.IS_NULL))
    assert hits == ["BBB"]


def test_empty_string_is_a_value_not_a_gap():
    frame = pd.DataFrame({"sector": ["", "Energy"]}, index=pd.Index(["AAA", "BBB"], name="ticker"))
    assert _hits(frame, FilterPredicate(field="sector", op=Operator.IS_NULL)) == []
    assert _hits(frame, FilterPredicate(field="sector", op=Operator.EQ, value="")) == ["AAA"]


def test_boolean_column_eq():
    frame = pd.DataFrame({"flag": [True, False]}, index=pd.Index(["AAA", "BBB"], name="ticker"))
    assert _hits(frame, FilterPredicate(field="flag", op=Operator.EQ, value=True)) == ["AAA"]


# ------------------------------------------------------------ Klartext ---


@pytest.mark.parametrize(
    ("predicate", "expected"),
    [
        (FilterPredicate(field="pe", op=Operator.LT, value=15), "pe < 15.0"),
        (FilterPredicate(field="pe", op=Operator.GT, value=1), "pe > 1.0"),
        (FilterPredicate(field="pe", op=Operator.GTE, value=1), "pe >= 1.0"),
        (FilterPredicate(field="pe", op=Operator.LTE, value=1), "pe <= 1.0"),
        (FilterPredicate(field="pe", op=Operator.EQ, value=1), "pe == 1.0"),
        (FilterPredicate(field="pe", op=Operator.NEQ, value=1), "pe != 1.0"),
        (
            FilterPredicate(field="roe", op=Operator.BETWEEN, value=0.1, value2=0.4),
            "roe between 0.1 and 0.4",
        ),
        (
            FilterPredicate(field="s", op=Operator.IN, values=["A"]),
            "s in ['A']",
        ),
        (
            FilterPredicate(field="s", op=Operator.NOT_IN, values=["A"]),
            "s not in ['A']",
        ),
        (FilterPredicate(field="pe", op=Operator.IS_NULL), "pe missing"),
        (FilterPredicate(field="pe", op=Operator.NOT_NULL), "pe present"),
    ],
)
def test_describe_predicate(predicate, expected):
    assert describe_predicate(predicate) == expected


def test_describe_group_and():
    group = FilterGroup(
        op=LogicalOp.AND,
        predicates=[
            FilterPredicate(field="pe", op=Operator.LT, value=15),
            FilterPredicate(field="roe", op=Operator.GT, value=0.15),
        ],
    )
    assert describe_group(group) == "(pe < 15.0 AND roe > 0.15)"


def test_describe_group_or_negated_and_nested():
    group = FilterGroup(
        op=LogicalOp.OR,
        predicates=[FilterPredicate(field="pe", op=Operator.LT, value=15)],
        groups=[FilterGroup(predicates=[FilterPredicate(field="roe", op=Operator.GT, value=0.1)])],
        negate=True,
    )
    assert describe_group(group) == "NOT (pe < 15.0 OR (roe > 0.1))"
