"""Tests der Strategie-Registry."""

from __future__ import annotations

import pytest

from btlite.backtest.base import Strategy, StrategyRegistry
from btlite.backtest.registry import (
    BUILTIN_STRATEGIES,
    InMemoryStrategyRegistry,
    default_registry,
)
from btlite.backtest.strategies import (
    BuyAndHoldStrategy,
    MomentumRotationStrategy,
    RsiMeanReversionStrategy,
    SmaCrossStrategy,
)
from btlite.core.errors import StrategyConfigError
from btlite.models.backtest import SignalSet


class _Dummy(Strategy):
    """Minimale Strategie fuer die Registry-Tests."""

    key = "dummy"
    label = "Dummy"

    def generate_signals(self, prices):  # pragma: no cover - nie aufgerufen
        raise NotImplementedError


def test_registry_satisfies_the_protocol():
    assert isinstance(default_registry(), StrategyRegistry)


def test_default_registry_contains_all_four_strategies():
    registry = default_registry()
    assert len(registry) == 4
    assert registry.keys() == [
        "buy_and_hold",
        "momentum_rotation",
        "rsi_mean_reversion",
        "sma_cross",
    ]


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("sma_cross", SmaCrossStrategy),
        ("rsi_mean_reversion", RsiMeanReversionStrategy),
        ("momentum_rotation", MomentumRotationStrategy),
        ("buy_and_hold", BuyAndHoldStrategy),
    ],
)
def test_get_returns_the_registered_class(key, expected):
    assert default_registry().get(key) is expected


def test_duplicate_key_raises_strategy_config_error():
    registry = default_registry()
    with pytest.raises(StrategyConfigError, match="Strategy key 'sma_cross' is already taken"):
        registry.register(SmaCrossStrategy)


def test_duplicate_key_names_both_classes_in_detail():
    registry = InMemoryStrategyRegistry()
    registry.register(SmaCrossStrategy)

    class Clashing(Strategy):
        key = "sma_cross"
        label = "Kollision"

        def generate_signals(self, prices):  # pragma: no cover
            raise NotImplementedError

    with pytest.raises(StrategyConfigError) as excinfo:
        registry.register(Clashing)
    assert excinfo.value.detail is not None
    assert "SmaCrossStrategy" in excinfo.value.detail
    assert "Clashing" in excinfo.value.detail


def test_unknown_key_raises_and_lists_available_keys():
    with pytest.raises(StrategyConfigError) as excinfo:
        default_registry().get("sma_crossover")
    assert excinfo.value.detail is not None
    assert "sma_cross" in excinfo.value.detail


def test_register_rejects_non_strategy():
    registry = InMemoryStrategyRegistry()
    with pytest.raises(StrategyConfigError, match="Only subclasses of Strategy can be registered"):
        registry.register(dict)  # type: ignore[arg-type]


def test_register_rejects_empty_key():
    class NoKey(Strategy):
        key = ""

        def generate_signals(self, prices):  # pragma: no cover
            raise NotImplementedError

    with pytest.raises(StrategyConfigError, match="key"):
        InMemoryStrategyRegistry().register(NoKey)


def test_list_all_is_sorted_by_key():
    classes = default_registry().list_all()
    assert [c.key for c in classes] == sorted(c.key for c in classes)
    assert set(classes) == set(BUILTIN_STRATEGIES)


def test_registry_supports_contains_and_len():
    registry = default_registry()
    assert "sma_cross" in registry
    assert "does_not_exist" not in registry
    assert len(registry) == 4


def test_default_registry_returns_a_fresh_instance():
    first, second = default_registry(), default_registry()
    assert first is not second
    # Eine Aenderung darf die andere Registry nicht beeinflussen.
    first.register(_Dummy)
    assert "dummy" in first
    assert "dummy" not in second


def test_registered_strategies_can_be_instantiated_from_the_registry():
    registry = default_registry()
    for strategy_cls in registry.list_all():
        key = strategy_cls.key
        instance = registry.get(key)()
        assert isinstance(instance, Strategy)
        assert instance.key == key


def test_signal_set_is_the_common_return_type():
    # Vertragspruefung ohne Kursdaten: alle Strategien deklarieren
    # SignalSet als Rueckgabe.
    for cls in BUILTIN_STRATEGIES:
        annotation = cls.generate_signals.__annotations__.get("return")
        assert annotation in (SignalSet, "SignalSet")
