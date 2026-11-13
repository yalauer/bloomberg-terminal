"""Tests der Screening-Orchestrierung.

Der Provider wird injiziert. Ein Test-Double genuegt, weil
:class:`~btlite.data.base.ScreenerProvider` ein strukturelles Protocol ist -
Vererbung ist nicht noetig.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from btlite.core.errors import ScreeningError, UnknownFilterError
from btlite.data.base import ScreenerProvider
from btlite.models.common import DataSource, Provenance
from btlite.models.screening import (
    FilterGroup,
    FilterPredicate,
    Operator,
    ScreenRequest,
    ScreenResult,
)
from btlite.screening.engine import (
    ScreeningEngine,
    empty_universe,
    export_result,
    load_export,
    sort_frame,
    universe_frame,
)


class StubProvider:
    """Screener-Provider mit festem Ergebnis, zaehlt seine Aufrufe."""

    def __init__(self, frame: pd.DataFrame) -> None:
        """Merkt sich den Frame, den jeder screen-Aufruf zurueckgibt."""
        self._frame = frame
        self.calls: list[ScreenRequest] = []

    def screen(self, request: ScreenRequest) -> ScreenResult:
        self.calls.append(request)
        return ScreenResult(
            request=request,
            data=self._frame,
            total_matches=len(self._frame),
            provenance=Provenance(
                source=DataSource.FINVIZ,
                fetched_at=datetime(2026, 9, 4, 12, 0, tzinfo=UTC),
                delay_minutes=None,
            ),
        )

    def available_filters(self) -> dict[str, list[str]]:
        return {"Market Cap.": ["Mega ($200bln and more)"]}


@pytest.fixture
def remote_frame() -> pd.DataFrame:
    """Remote-Ergebnis mit Kennzahlen und Luecken."""
    return pd.DataFrame(
        {
            "pe": [10.0, 12.0, 30.0, np.nan, 8.0],
            "roe": [0.25, 0.05, 0.40, 0.30, np.nan],
            "sector": ["Technology", "Energy", "Technology", "Health", "Utilities"],
        },
        index=pd.Index(["AAA", "BBB", "CCC", "DDD", "EEE"], name="ticker"),
    )


@pytest.fixture
def engine(remote_frame: pd.DataFrame) -> ScreeningEngine:
    return ScreeningEngine(StubProvider(remote_frame))


# ------------------------------------------------------------- Protocol --


def test_stub_satisfies_screener_protocol(remote_frame):
    assert isinstance(StubProvider(remote_frame), ScreenerProvider)


# ------------------------------------------------------------ Universum --


def test_empty_universe_has_ticker_index():
    frame = empty_universe()
    assert frame.empty
    assert frame.index.name == "ticker"


def test_universe_frame_dedupes_and_keeps_order():
    frame = universe_frame(["AAA", "BBB", "AAA", "CCC"])
    assert list(frame.index) == ["AAA", "BBB", "CCC"]
    assert frame.index.name == "ticker"


def test_universe_from_request_without_provider():
    engine = ScreeningEngine(None)
    request = ScreenRequest(universe=["AAA", "BBB"])
    result = engine.screen(request)
    assert result.tickers == ["AAA", "BBB"]
    assert result.total_matches == 2
    assert result.provenance.source is DataSource.COMPUTED


def test_finviz_filters_without_provider_raise():
    engine = ScreeningEngine(None)
    request = ScreenRequest(finviz_filters={"Market Cap.": "Mega ($200bln and more)"})
    with pytest.raises(
        ScreeningError, match="Finviz filters require the engine to have a screener provider"
    ):
        engine.screen(request)


def test_request_without_universe_and_filters_raises():
    engine = ScreeningEngine(None)
    with pytest.raises(ScreeningError, match="names neither a universe nor any Finviz filters"):
        engine.screen(ScreenRequest())


def test_passed_universe_takes_precedence_over_provider(remote_frame):
    provider = StubProvider(remote_frame)
    engine = ScreeningEngine(provider)
    own = pd.DataFrame({"pe": [1.0]}, index=pd.Index(["ZZZ"], name="ticker"))
    result = engine.screen(
        ScreenRequest(finviz_filters={"Market Cap.": "Mega ($200bln and more)"}),
        universe=own,
    )
    assert result.tickers == ["ZZZ"]
    assert provider.calls == []
    assert result.provenance.source is DataSource.COMPUTED


def test_engine_never_fetches_without_being_asked(remote_frame):
    provider = StubProvider(remote_frame)
    engine = ScreeningEngine(provider)
    engine.screen(ScreenRequest(universe=["AAA"]))
    assert provider.calls == []


def test_provider_property(remote_frame):
    provider = StubProvider(remote_frame)
    assert ScreeningEngine(provider).provider is provider
    assert ScreeningEngine(None).provider is None


# ------------------------------------------------- remote + lokal kombi --


def test_remote_then_local_filters(engine):
    request = ScreenRequest(
        finviz_filters={"Market Cap.": "Mega ($200bln and more)"},
        local_filters=FilterGroup(
            predicates=[
                FilterPredicate(field="pe", op=Operator.LT, value=15.0),
                FilterPredicate(field="roe", op=Operator.GT, value=0.10),
            ]
        ),
    )
    result = engine.screen(request)
    assert result.tickers == ["AAA"]
    assert result.provenance.source is DataSource.FINVIZ


def test_remote_result_passes_through_without_local_filters(engine):
    request = ScreenRequest(finviz_filters={"Market Cap.": "Mega ($200bln and more)"})
    result = engine.screen(request)
    assert result.tickers == ["AAA", "BBB", "CCC", "DDD", "EEE"]


def test_request_universe_acts_as_whitelist_over_remote(engine):
    request = ScreenRequest(
        finviz_filters={"Market Cap.": "Mega ($200bln and more)"},
        universe=["BBB", "CCC", "XXX"],
    )
    result = engine.screen(request)
    assert result.tickers == ["BBB", "CCC"]


def test_local_filter_on_unknown_column_raises(engine):
    request = ScreenRequest(
        finviz_filters={"Market Cap.": "Mega ($200bln and more)"},
        local_filters=FilterGroup(
            predicates=[FilterPredicate(field="nonexistent", op=Operator.GT, value=1.0)]
        ),
    )
    with pytest.raises(UnknownFilterError):
        engine.screen(request)


def test_local_filter_on_bare_universe_raises_instead_of_empty_result():
    """Ohne angereicherte Daten kann keine Kennzahl geprueft werden."""
    engine = ScreeningEngine(None)
    request = ScreenRequest(
        universe=["AAA", "BBB"],
        local_filters=FilterGroup(
            predicates=[FilterPredicate(field="pe", op=Operator.LT, value=15.0)]
        ),
    )
    with pytest.raises(UnknownFilterError):
        engine.screen(request)


# --------------------------------------------- limit und total_matches ---


def test_total_matches_is_set_before_limit(engine):
    request = ScreenRequest(finviz_filters={"Market Cap.": "Mega ($200bln and more)"}, limit=2)
    result = engine.screen(request)
    assert result.total_matches == 5
    assert len(result.data) == 2
    assert result.is_truncated


def test_total_matches_counts_after_local_filters(engine):
    request = ScreenRequest(
        finviz_filters={"Market Cap.": "Mega ($200bln and more)"},
        local_filters=FilterGroup(predicates=[FilterPredicate(field="pe", op=Operator.NOT_NULL)]),
        limit=2,
    )
    result = engine.screen(request)
    assert result.total_matches == 4
    assert len(result.data) == 2


def test_not_truncated_when_limit_exceeds_matches(engine):
    request = ScreenRequest(finviz_filters={"Market Cap.": "Mega ($200bln and more)"}, limit=100)
    result = engine.screen(request)
    assert result.total_matches == 5
    assert not result.is_truncated


def test_limit_keeps_the_top_rows_after_sorting(engine):
    request = ScreenRequest(
        finviz_filters={"Market Cap.": "Mega ($200bln and more)"},
        sort_by="pe",
        ascending=True,
        limit=2,
    )
    result = engine.screen(request)
    assert result.tickers == ["EEE", "AAA"]


# ---------------------------------------------------------- Sortierung ---


def test_sort_ascending(engine):
    request = ScreenRequest(
        finviz_filters={"Market Cap.": "Mega ($200bln and more)"},
        sort_by="pe",
        ascending=True,
    )
    assert engine.screen(request).tickers == ["EEE", "AAA", "BBB", "CCC", "DDD"]


def test_sort_descending(engine):
    request = ScreenRequest(
        finviz_filters={"Market Cap.": "Mega ($200bln and more)"},
        sort_by="pe",
        ascending=False,
    )
    assert engine.screen(request).tickers == ["CCC", "BBB", "AAA", "EEE", "DDD"]


def test_missing_values_sort_last_in_both_directions(remote_frame):
    ascending = sort_frame(remote_frame, "pe", ascending=True)
    descending = sort_frame(remote_frame, "pe", ascending=False)
    assert list(ascending.index)[-1] == "DDD"
    assert list(descending.index)[-1] == "DDD"


def test_sort_by_none_keeps_order(remote_frame):
    out = sort_frame(remote_frame, None, ascending=True)
    assert list(out.index) == list(remote_frame.index)


def test_sort_by_ticker_index(remote_frame):
    out = sort_frame(remote_frame, "ticker", ascending=False)
    assert list(out.index) == ["EEE", "DDD", "CCC", "BBB", "AAA"]


def test_unknown_sort_column_is_ignored(remote_frame, caplog):
    out = sort_frame(remote_frame, "does_not_exist", ascending=True)
    assert list(out.index) == list(remote_frame.index)


def test_sort_is_stable_on_ties():
    frame = pd.DataFrame(
        {"pe": [10.0, 10.0, 5.0]}, index=pd.Index(["AAA", "BBB", "CCC"], name="ticker")
    )
    out = sort_frame(frame, "pe", ascending=True)
    assert list(out.index) == ["CCC", "AAA", "BBB"]


# ------------------------------------------------------------- Export ----


def test_csv_round_trip(engine, tmp_path):
    result = engine.screen(ScreenRequest(finviz_filters={"Market Cap.": "Mega ($200bln and more)"}))
    path = export_result(result, tmp_path / "screen.csv")
    assert path.is_file()
    back = load_export(path)
    assert list(back.index) == result.tickers
    assert back.index.name == "ticker"
    assert list(back.columns) == list(result.data.columns)
    pd.testing.assert_series_equal(
        back["pe"], result.data["pe"], check_names=False, check_index=False
    )


def test_parquet_round_trip(engine, tmp_path):
    result = engine.screen(ScreenRequest(finviz_filters={"Market Cap.": "Mega ($200bln and more)"}))
    path = export_result(result, tmp_path / "screen.parquet")
    back = load_export(path)
    assert list(back.index) == result.tickers
    assert back.index.name == "ticker"
    pd.testing.assert_frame_equal(back, result.data, check_dtype=False)


def test_export_pq_suffix(engine, tmp_path):
    result = engine.screen(ScreenRequest(universe=["AAA", "BBB"]))
    path = export_result(result, tmp_path / "screen.pq")
    assert load_export(path).index.tolist() == ["AAA", "BBB"]


def test_export_creates_parent_directories(engine, tmp_path):
    result = engine.screen(ScreenRequest(universe=["AAA"]))
    path = export_result(result, tmp_path / "deep" / "nested" / "screen.csv")
    assert path.is_file()


def test_export_rejects_unknown_suffix(engine, tmp_path):
    result = engine.screen(ScreenRequest(universe=["AAA"]))
    with pytest.raises(ScreeningError, match=r"Unknown export format \'\.xlsx\'"):
        export_result(result, tmp_path / "screen.xlsx")


def test_load_rejects_unknown_suffix(tmp_path):
    path = tmp_path / "screen.xlsx"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(ScreeningError, match=r"Unknown import format \'\.xlsx\'"):
        load_export(path)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(ScreeningError):
        load_export(tmp_path / "absent.csv")


def test_export_respects_limit(engine, tmp_path):
    request = ScreenRequest(finviz_filters={"Market Cap.": "Mega ($200bln and more)"}, limit=2)
    result = engine.screen(request)
    back = load_export(export_result(result, tmp_path / "screen.csv"))
    assert len(back) == 2
    assert result.total_matches == 5


def test_export_of_empty_result(engine, tmp_path):
    request = ScreenRequest(
        finviz_filters={"Market Cap.": "Mega ($200bln and more)"},
        local_filters=FilterGroup(
            predicates=[FilterPredicate(field="pe", op=Operator.GT, value=1000.0)]
        ),
    )
    result = engine.screen(request)
    assert result.total_matches == 0
    back = load_export(export_result(result, tmp_path / "empty.csv"))
    assert back.empty
