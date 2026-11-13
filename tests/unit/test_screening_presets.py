"""Tests der vier Screening-Presets.

Kern dieser Datei: jedes Preset wird gegen das **echte**
``finvizfinance.constants.filter_dict`` validiert. Ein Tippfehler in einem
Filter-Key oder -Wert faellt damit hier auf und nicht erst zur Laufzeit gegen
die Quelle. Der Zugriff auf das Dictionary ist ein Import, kein Netzwerkruf -
die Suite bleibt offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btlite.core.errors import UnknownFilterError
from btlite.models.screening import ScreenPreset
from btlite.screening.engine import ScreeningEngine
from btlite.screening.filters import apply_filters
from btlite.screening.presets import (
    all_presets,
    dividend_preset,
    get_preset,
    momentum_preset,
    preset_keys,
    quality_preset,
    value_preset,
)

_EXPECTED_KEYS = ["value", "quality", "momentum", "dividend"]


@pytest.fixture(scope="module")
def filter_dict() -> dict[str, dict[str, object]]:
    """Das echte Filter-Dictionary des installierten Pakets."""
    from finvizfinance.constants import filter_dict as real

    return dict(real)


def _all_filter_pairs() -> list[tuple[str, str, str]]:
    """Alle (Preset-Key, Filter-Key, Filter-Wert)-Tripel der vier Presets."""
    return [
        (preset.key, key, value)
        for preset in all_presets()
        for key, value in preset.request.finviz_filters.items()
    ]


# ------------------------------------------ Validierung gegen die Quelle --


@pytest.mark.parametrize(("preset_key", "filter_key", "filter_value"), _all_filter_pairs())
def test_preset_filter_key_exists(filter_dict, preset_key, filter_key, filter_value):
    """Jeder Filter-Key eines Presets steht im echten Dictionary."""
    assert filter_key in filter_dict, (
        f"Preset {preset_key!r}: Filter-Key {filter_key!r} ist Finviz unbekannt"
    )


@pytest.mark.parametrize(("preset_key", "filter_key", "filter_value"), _all_filter_pairs())
def test_preset_filter_value_exists(filter_dict, preset_key, filter_key, filter_value):
    """Jeder Filter-Wert ist ein voller Anzeigetext aus dem Dictionary."""
    options = filter_dict[filter_key]["option"]
    assert filter_value in options, (
        f"Preset {preset_key!r}: Wert {filter_value!r} ist fuer {filter_key!r} unzulaessig"
    )


def test_market_cap_uses_the_dotted_key(filter_dict):
    """Der Key traegt den Punkt - 'Market Cap' ohne Punkt gibt es nicht."""
    assert "Market Cap." in filter_dict
    assert "Market Cap" not in filter_dict
    for preset in all_presets():
        assert "Market Cap" not in preset.request.finviz_filters


def test_market_cap_values_are_full_display_texts(filter_dict):
    """Ein blosses 'Mega' waere unzulaessig, der Anzeigetext ist noetig."""
    options = filter_dict["Market Cap."]["option"]
    assert "Mega" not in options
    assert "Mega ($200bln and more)" in options


def test_rsi_key_carries_the_period(filter_dict):
    """Der RSI-Key heisst 'RSI (14)', nicht 'RSI'."""
    assert "RSI (14)" in filter_dict
    assert "RSI" not in filter_dict
    assert "RSI (14)" in momentum_preset().request.finviz_filters


def test_no_preset_uses_a_shortened_value(filter_dict):
    """Kein Wert eines Presets ist eine Kurzform eines gueltigen Werts."""
    for preset in all_presets():
        for key, value in preset.request.finviz_filters.items():
            options = list(filter_dict[key]["option"])
            assert value in options
            longer = [o for o in options if o != value and o.startswith(f"{value} ")]
            assert not longer, f"{preset.key}: {value!r} sieht wie eine Kurzform von {longer} aus"


def test_available_filters_of_provider_accepts_every_preset():
    """Gegenprobe ueber die Protocol-Methode statt ueber den Direktimport."""
    from btlite.data.fixtures import FixtureProvider

    known = FixtureProvider().available_filters()
    assert known, "FixtureProvider liefert keine Filterliste"
    for preset in all_presets():
        for key, value in preset.request.finviz_filters.items():
            assert key in known
            assert value in known[key]


def test_fixture_provider_accepts_every_preset_request():
    """Die Presets laufen ohne UnknownFilterError durch die Validierung."""
    from btlite.data.fixtures import FixtureProvider

    provider = FixtureProvider()
    for preset in all_presets():
        provider.screen(preset.request)


# --------------------------------------------------------- Registry ------


def test_four_presets_in_display_order():
    assert preset_keys() == _EXPECTED_KEYS


def test_all_presets_are_screen_presets():
    for preset in all_presets():
        assert isinstance(preset, ScreenPreset)


def test_preset_keys_are_unique():
    keys = preset_keys()
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("key", _EXPECTED_KEYS)
def test_get_preset(key):
    assert get_preset(key).key == key


def test_get_preset_is_case_insensitive():
    assert get_preset("VALUE").key == "value"
    assert get_preset("  Quality  ").key == "quality"


def test_get_preset_unknown_raises_with_valid_keys():
    with pytest.raises(UnknownFilterError) as excinfo:
        get_preset("groath")
    assert excinfo.value.detail is not None
    assert "value" in excinfo.value.detail


def test_presets_are_fresh_instances():
    """Kein geteilter Zustand zwischen Aufrufen."""
    assert all_presets()[0] is not all_presets()[0]


# ------------------------------------------------------------ Inhalt -----


@pytest.mark.parametrize(
    "factory", [value_preset, quality_preset, momentum_preset, dividend_preset]
)
def test_preset_shape(factory):
    preset = factory()
    assert preset.key
    assert preset.label
    assert preset.description.endswith(".")
    assert preset.request.finviz_filters
    assert preset.request.local_filters is not None
    assert 1 <= preset.request.limit <= 5000


#: Kennzahlen-Keys, die ``analytics/ratios.py`` erzeugt. Lokale Praedikate
#: der Presets duerfen nur diese nennen, sonst treffen sie im angereicherten
#: Universum keine Spalte.
_RATIO_KEYS = frozenset(
    {
        "asset_turnover",
        "cash_ratio",
        "ccc",
        "current_ratio",
        "debt_to_assets",
        "debt_to_equity",
        "dio",
        "dividend_yield",
        "dpo",
        "dso",
        "ev_to_ebitda",
        "fcf_yield",
        "interest_coverage",
        "inventory_turnover",
        "net_debt_to_ebitda",
        "pb",
        "pe",
        "ps",
        "quick_ratio",
        "receivables_turnover",
        "revenue_cagr_3y",
        "roa",
        "roe",
        "roic",
    }
)


@pytest.mark.parametrize(
    "factory", [value_preset, quality_preset, momentum_preset, dividend_preset]
)
def test_preset_local_filters_reference_ratio_keys(factory):
    """Lokale Praedikate nennen nur Keys, die ratios.py auch erzeugt."""
    group = factory().request.local_filters
    assert group is not None
    fields = {p.field for p in group.predicates}
    for sub in group.groups:
        fields |= {p.field for p in sub.predicates}
    assert fields, "Preset ohne lokale Praedikate"
    assert fields <= _RATIO_KEYS, f"unbekannte Kennzahlen: {sorted(fields - _RATIO_KEYS)}"


def test_value_preset_sorts_cheapest_first():
    request = value_preset().request
    assert request.sort_by == "pe"
    assert request.ascending is True


def test_quality_preset_sorts_by_roic_descending():
    request = quality_preset().request
    assert request.sort_by == "roic"
    assert request.ascending is False


def test_dividend_preset_caps_the_payout_ratio():
    """Die Quotengrenze ist der Kern des Presets."""
    filters = dividend_preset().request.finviz_filters
    assert filters["Payout Ratio"] == "Under 80%"
    assert filters["Dividend Yield"] == "Over 3%"


def test_momentum_preset_avoids_overbought():
    filters = momentum_preset().request.finviz_filters
    assert filters["RSI (14)"] == "Not Overbought (<60)"
    assert filters["200-Day Simple Moving Average"] == "Price above SMA200"


def test_all_presets_share_the_liquidity_floor():
    for preset in all_presets():
        filters = preset.request.finviz_filters
        assert filters["Market Cap."] == "+Mid (over $2bln)"
        assert filters["Average Volume"] == "Over 500K"


# --------------------------------------------- Presets in der Engine -----


@pytest.fixture
def universe() -> pd.DataFrame:
    """Universum mit allen Kennzahlen, die die Presets lokal pruefen."""
    return pd.DataFrame(
        {
            "pe": [9.0, 40.0, 12.0, np.nan],
            "pb": [1.5, 8.0, 2.0, 1.0],
            "ps": [1.0, 12.0, 2.5, 1.0],
            "roe": [0.22, 0.35, 0.18, 0.10],
            "roic": [0.18, 0.30, 0.12, 0.05],
            "roa": [0.10, 0.20, 0.08, 0.02],
            "debt_to_equity": [0.4, 0.3, 0.9, 2.0],
            "current_ratio": [2.0, 3.0, 1.8, 1.0],
            "dividend_yield": [0.045, 0.0, 0.035, np.nan],
        },
        index=pd.Index(["GOOD", "PRICEY", "OKAY", "WEAK"], name="ticker"),
    )


@pytest.mark.parametrize(
    "factory", [value_preset, quality_preset, momentum_preset, dividend_preset]
)
def test_preset_local_filters_run_against_a_universe(factory, universe):
    """Die lokalen Praedikate laufen ohne UnknownFilterError durch."""
    out = apply_filters(universe, factory().request.local_filters)
    assert isinstance(out, pd.DataFrame)
    assert set(out.index) <= set(universe.index)


def test_value_preset_selects_the_cheap_profitable_names(universe):
    out = apply_filters(universe, value_preset().request.local_filters)
    assert list(out.index) == ["GOOD", "OKAY"]


def test_quality_preset_selects_high_return_low_debt(universe):
    """WEAK faellt an ROE, ROIC und der Verschuldung durch."""
    out = apply_filters(universe, quality_preset().request.local_filters)
    assert list(out.index) == ["GOOD", "PRICEY", "OKAY"]
    assert "WEAK" not in out.index


def test_dividend_preset_skips_missing_and_zero_yields(universe):
    out = apply_filters(universe, dividend_preset().request.local_filters)
    assert list(out.index) == ["GOOD", "OKAY"]
    assert "WEAK" not in out.index


def test_preset_runs_end_to_end_in_the_engine(universe):
    engine = ScreeningEngine(None)
    result = engine.screen(value_preset().request, universe=universe)
    assert result.tickers == ["GOOD", "OKAY"]
    assert result.total_matches == 2
    assert not result.is_truncated


def test_preset_sorting_applies_in_the_engine(universe):
    engine = ScreeningEngine(None)
    result = engine.screen(value_preset().request, universe=universe)
    assert result.tickers == ["GOOD", "OKAY"]
    assert result.data["pe"].tolist() == [9.0, 12.0]


def test_preset_limit_and_total_matches_in_the_engine(universe):
    engine = ScreeningEngine(None)
    request = value_preset().request.model_copy(update={"limit": 1})
    result = engine.screen(request, universe=universe)
    assert result.total_matches == 2
    assert len(result.data) == 1
    assert result.is_truncated
