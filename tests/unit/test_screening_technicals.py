"""Tests der technischen Screening-Spalten und ihrer Voreinstellungen."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btlite.analytics.technicals import rsi, sma
from btlite.models.market import PriceSeries
from btlite.screening.engine import ScreeningEngine
from btlite.screening.technical_presets import (
    all_technical_presets,
    get_technical_preset,
    technical_preset_keys,
)
from btlite.screening.technicals import (
    TECHNICAL_COLUMNS,
    build_technical_universe,
    describe_columns,
    enrich_with_technicals,
    technical_row,
)

TICKERS = ["AAPL", "MSFT", "SPY", "WMT", "KO"]


@pytest.fixture
def provider():
    from btlite.data.fixtures import FixtureProvider

    return FixtureProvider()


@pytest.fixture
def prices(provider):
    return provider.get_histories(TICKERS)


@pytest.fixture
def universe(prices):
    return build_technical_universe(prices)


def _synthetic(values: list[float], *, ticker: str, provenance) -> PriceSeries:
    """Baut eine Kursreihe aus vorgegebenen Schlusskursen."""
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B", tz="UTC", name="date")
    frame = pd.DataFrame(
        {
            "open": values,
            "high": [v * 1.01 for v in values],
            "low": [v * 0.99 for v in values],
            "close": values,
            "volume": [1_000_000.0] * len(values),
        },
        index=idx,
    )
    return PriceSeries(ticker=ticker, adjusted=True, data=frame, provenance=provenance)


class TestTechnicalRow:
    def test_all_declared_columns_are_present(self, prices):
        row = technical_row(prices["AAPL"])
        assert set(row) == set(TECHNICAL_COLUMNS)

    def test_sma_matches_the_analytics_layer(self, prices):
        """Querprobe gegen die Indikator-Schicht, nicht gegen sich selbst."""
        row = technical_row(prices["AAPL"])
        expected = float(sma(prices["AAPL"].close, 200).dropna().iloc[-1])
        assert row["sma_200"] == pytest.approx(expected)

    def test_rsi_matches_the_analytics_layer(self, prices):
        row = technical_row(prices["AAPL"])
        expected = float(rsi(prices["AAPL"].close, 14).dropna().iloc[-1])
        assert row["rsi_14"] == pytest.approx(expected)

    def test_distance_is_signed_and_relative(self, prices):
        row = technical_row(prices["AAPL"])
        manual = (row["price"] - row["sma_200"]) / row["sma_200"]
        assert row["dist_sma_200"] == pytest.approx(manual)

    def test_negative_distance_means_below_the_line(self, provenance):
        """Die Leserichtung ist der Kern der Spalte: negativ heisst darunter."""
        falling = _synthetic([100.0] * 250 + [70.0] * 10, ticker="FALL", provenance=provenance)
        row = technical_row(falling)
        assert row["dist_sma_200"] < 0
        assert row["price"] < row["sma_200"]
        assert row["above_sma_200"] == 0.0

    def test_positive_distance_means_above_the_line(self, provenance):
        rising = _synthetic(
            [100.0 + i * 0.5 for i in range(260)], ticker="RISE", provenance=provenance
        )
        row = technical_row(rising)
        assert row["dist_sma_200"] > 0
        assert row["above_sma_200"] == 1.0

    def test_golden_cross_flag(self, provenance):
        rising = _synthetic(
            [100.0 + i * 0.5 for i in range(260)], ticker="RISE", provenance=provenance
        )
        row = technical_row(rising)
        assert row["golden_cross"] == 1.0
        assert row["sma_50"] > row["sma_200"]

    def test_drawdown_is_negative_or_zero(self, universe):
        for value in universe["max_drawdown_1y"].dropna():
            assert value <= 0

    def test_distance_from_high_is_never_positive(self, universe):
        for value in universe["from_high_52w"].dropna():
            assert value <= 1e-12

    def test_distance_from_low_is_never_negative(self, universe):
        for value in universe["from_low_52w"].dropna():
            assert value >= -1e-12

    def test_short_history_yields_none_not_an_error(self, provenance):
        """Ein junger Titel soll erscheinen, nur ohne lange Linien."""
        young = _synthetic([100.0 + i for i in range(30)], ticker="YOUNG", provenance=provenance)
        row = technical_row(young)
        assert row["price"] is not None
        assert row["sma_20"] is not None
        assert row["sma_200"] is None, "30 Bars tragen keine 200er Linie"
        assert row["dist_sma_200"] is None
        assert row["perf_12m"] is None

    def test_no_value_is_nan_or_infinite(self, universe):
        for column in universe.columns:
            for value in universe[column].dropna():
                assert np.isfinite(value), f"{column} ist nicht endlich"


class TestBuildUniverse:
    def test_one_row_per_ticker(self, universe):
        assert list(universe.index) == TICKERS

    def test_index_is_named(self, universe):
        assert universe.index.name == "ticker"

    def test_columns_follow_the_declared_order(self, universe):
        assert list(universe.columns) == list(TECHNICAL_COLUMNS)

    def test_empty_input_yields_empty_frame(self):
        frame = build_technical_universe({})
        assert frame.empty
        assert list(frame.columns) == list(TECHNICAL_COLUMNS)


class TestEnrich:
    def test_merges_onto_an_existing_universe(self, prices):
        base = pd.DataFrame({"roe": [0.3, 0.4]}, index=["AAPL", "MSFT"])
        base.index.name = "ticker"
        merged = enrich_with_technicals(base, prices)
        assert "roe" in merged.columns
        assert "dist_sma_200" in merged.columns
        assert merged.loc["AAPL", "roe"] == pytest.approx(0.3)

    def test_ticker_without_prices_keeps_its_row(self, prices):
        base = pd.DataFrame({"roe": [0.3, 0.1]}, index=["AAPL", "ZZZZ"])
        base.index.name = "ticker"
        merged = enrich_with_technicals(base, prices)
        assert "ZZZZ" in merged.index
        assert pd.isna(merged.loc["ZZZZ", "dist_sma_200"])

    def test_technical_column_wins_on_collision(self, prices):
        """Der Kurs aus der Reihe ist aktueller als ein uebergebener."""
        base = pd.DataFrame({"price": [1.0], "roe": [0.3]}, index=["AAPL"])
        base.index.name = "ticker"
        merged = enrich_with_technicals(base, prices)
        assert merged.loc["AAPL", "price"] != pytest.approx(1.0)


class TestPresets:
    def test_keys_are_unique(self):
        keys = [preset.key for preset in all_technical_presets()]
        assert len(keys) == len(set(keys))

    def test_lookup_by_key(self):
        for key in technical_preset_keys():
            assert get_technical_preset(key).key == key

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError, match="Unbekanntes"):
            get_technical_preset("gibt-es-nicht")

    def test_every_preset_filters_on_existing_columns(self, universe):
        """Ein Tippfehler im Spaltennamen faellt hier auf, nicht zur Laufzeit."""
        known = set(universe.columns)
        for preset in all_technical_presets():
            group = preset.request.local_filters
            assert group is not None
            for predicate in group.predicates:
                assert predicate.field in known, f"{preset.key}: {predicate.field} fehlt"

    def test_every_preset_runs_without_error(self, universe):
        engine = ScreeningEngine()
        for preset in all_technical_presets():
            result = engine.screen(preset.request, universe=universe)
            assert result.total_matches >= 0

    def test_below_200ma_finds_exactly_the_titles_under_the_line(self, universe):
        """Das vom Nutzer gewuenschte Preset gegen die Rohdaten gegengeprueft."""
        engine = ScreeningEngine()
        result = engine.screen(get_technical_preset("below_200ma").request, universe=universe)
        expected = set(universe.index[universe["dist_sma_200"] < 0])
        # Das Preset schliesst RSI unter 20 aus, deshalb Teilmenge.
        assert set(result.tickers) <= expected
        for ticker in result.tickers:
            assert universe.loc[ticker, "dist_sma_200"] < 0

    def test_above_and_below_are_disjoint(self, universe):
        engine = ScreeningEngine()
        below = set(
            engine.screen(get_technical_preset("below_200ma").request, universe=universe).tickers
        )
        above = set(
            engine.screen(get_technical_preset("above_200ma").request, universe=universe).tickers
        )
        assert not (below & above), "Ein Titel kann nicht ueber und unter der Linie liegen"

    def test_sort_columns_exist(self, universe):
        for preset in all_technical_presets():
            if preset.request.sort_by:
                assert preset.request.sort_by in universe.columns

    def test_presets_need_no_remote_source(self):
        """Technische Presets laufen rein lokal, ohne Screener-Quelle."""
        for preset in all_technical_presets():
            assert not preset.request.finviz_filters


class TestDocumentation:
    def test_every_column_is_described(self):
        assert set(describe_columns()) == set(TECHNICAL_COLUMNS)

    def test_descriptions_name_their_unit(self):
        """Ohne Einheit ist eine Kennzahl nicht interpretierbar."""
        marker = ("decimal", "price currency", "scale", "shares", "1 if")
        for column, text in describe_columns().items():
            assert any(word in text for word in marker), f"{column} nennt keine Einheit"


class TestSetupColumns:
    """Setup-Spalten: Abbildung aus analytics.setups, keine eigene Logik."""

    SETUP_COLUMNS = (
        "setup_pullback",
        "setup_rsi_recovery",
        "setup_breakout",
        "setup_support_macd",
    )

    def test_columns_declared_and_described(self):
        descriptions = describe_columns()
        for column in (*self.SETUP_COLUMNS, "dist_support", "dist_resistance"):
            assert column in TECHNICAL_COLUMNS
            assert column in descriptions

    @pytest.mark.parametrize("ticker", TICKERS)
    def test_values_follow_the_setup_report(self, prices, ticker):
        """Querprobe gegen die Analytik-Schicht: met 1, not met 0, sonst None."""
        from btlite.analytics.setups import setup_report
        from btlite.models.setups import SetupKey, SetupStatus

        row = technical_row(prices[ticker])
        report = setup_report(prices[ticker])
        expected = {SetupStatus.MET: 1.0, SetupStatus.NOT_MET: 0.0}
        keys = (
            SetupKey.PULLBACK,
            SetupKey.RSI_RECOVERY,
            SetupKey.BREAKOUT,
            SetupKey.SUPPORT_MACD,
        )
        for column, key in zip(self.SETUP_COLUMNS, keys, strict=True):
            check = report.check(key)
            assert check is not None
            assert row[column] == expected.get(check.status)
        assert row["dist_support"] == report.dist_support
        assert row["dist_resistance"] == report.dist_resistance

    def test_signs_of_the_zone_distances(self, universe):
        support = universe["dist_support"].dropna()
        resistance = universe["dist_resistance"].dropna()
        assert (support >= 0).all()
        assert (resistance < 0).all()

    def test_insufficient_history_is_none_not_zero(self, provenance):
        young = _synthetic([100.0 + i for i in range(30)], ticker="YOUNG", provenance=provenance)
        row = technical_row(young)
        assert row["setup_pullback"] is None
        assert row["setup_breakout"] is None
        assert row["setup_support_macd"] is None
        assert row["setup_rsi_recovery"] in (0.0, 1.0)
        assert row["dist_support"] is None

    def test_filter_on_a_setup_column(self, universe):
        from btlite.models.screening import (
            FilterGroup,
            FilterPredicate,
            LogicalOp,
            Operator,
            ScreenRequest,
        )

        request = ScreenRequest(
            name="Pullback met",
            local_filters=FilterGroup(
                op=LogicalOp.AND,
                predicates=[FilterPredicate(field="setup_pullback", op=Operator.EQ, value=1.0)],
            ),
        )
        result = ScreeningEngine().screen(request, universe=universe)
        expected = set(universe.index[universe["setup_pullback"] == 1.0])
        assert set(result.data.index) == expected
