"""Nachschlagewerk der verfuegbaren Strategien.

Erfuellt das Protocol :class:`~btlite.backtest.base.StrategyRegistry`. Das
UI fuellt sein Auswahlmenue ueber :func:`default_registry` und muss die
einzelnen Strategie-Module nicht kennen.

Die Registry ist bewusst eine Instanz und kein Modul-Singleton mit
globalem Zustand: :func:`default_registry` baut bei jedem Aufruf eine
frische Registry. Tests koennen damit registrieren, ohne sich gegenseitig
zu beeinflussen.
"""

from __future__ import annotations

from btlite.backtest.base import Strategy
from btlite.backtest.strategies import (
    BuyAndHoldStrategy,
    MomentumRotationStrategy,
    RsiMeanReversionStrategy,
    SmaCrossStrategy,
)
from btlite.core.errors import StrategyConfigError

#: Die mitgelieferten Referenzstrategien, in fachlicher Reihenfolge.
BUILTIN_STRATEGIES: tuple[type[Strategy], ...] = (
    SmaCrossStrategy,
    RsiMeanReversionStrategy,
    MomentumRotationStrategy,
    BuyAndHoldStrategy,
)


class InMemoryStrategyRegistry:
    """Registry im Prozessspeicher.

    Ein doppelt vergebener Key ist ein Fehler und kein stilles Ueber-
    schreiben: sonst haengt es von der Importreihenfolge ab, welche
    Strategie ein Nutzer tatsaechlich bekommt.
    """

    def __init__(self) -> None:
        """Erzeugt eine leere Registry."""
        self._strategies: dict[str, type[Strategy]] = {}

    def register(self, strategy_cls: type[Strategy]) -> None:
        """Nimmt eine Strategie auf.

        Args:
            strategy_cls: Zu registrierende Klasse. Muss von
                :class:`~btlite.backtest.base.Strategy` erben und einen
                nicht-leeren ``key`` tragen.

        Raises:
            StrategyConfigError: Wenn der Key leer oder bereits vergeben
                ist, oder die Klasse nicht von ``Strategy`` erbt.
        """
        # Laufzeit-Schutz: das UI kann die Registry auch dynamisch fuellen,
        # der statische Typ ist dort nicht garantiert.
        is_strategy = isinstance(strategy_cls, type) and issubclass(strategy_cls, Strategy)
        if not is_strategy:
            msg = "Only subclasses of Strategy can be registered"
            raise StrategyConfigError(msg, detail=f"erhalten={strategy_cls!r}")

        key = strategy_cls.key
        if not key:
            msg = "A strategy needs a non-empty key"
            raise StrategyConfigError(msg, detail=f"klasse={strategy_cls.__name__}")

        existing = self._strategies.get(key)
        if existing is not None:
            msg = f"Strategy key {key!r} is already taken"
            raise StrategyConfigError(
                msg,
                detail=f"belegt_von={existing.__name__}, neu={strategy_cls.__name__}",
            )
        self._strategies[key] = strategy_cls

    def get(self, key: str) -> type[Strategy]:
        """Sucht eine Strategie nach ihrem Key.

        Args:
            key: Bezeichner der Strategie.

        Returns:
            Die registrierte Klasse.

        Raises:
            StrategyConfigError: Wenn der Key unbekannt ist. Die Meldung
                nennt die verfuegbaren Keys, damit ein Tippfehler sofort
                sichtbar wird.
        """
        try:
            return self._strategies[key]
        except KeyError:
            msg = f"Unknown strategy {key!r}"
            raise StrategyConfigError(
                msg, detail=f"verfuegbar={sorted(self._strategies)}"
            ) from None

    def list_all(self) -> list[type[Strategy]]:
        """Listet alle registrierten Strategien.

        Returns:
            Registrierte Klassen, stabil nach Key sortiert.
        """
        return [self._strategies[key] for key in sorted(self._strategies)]

    def keys(self) -> list[str]:
        """Listet die registrierten Keys.

        Returns:
            Keys, aufsteigend sortiert.
        """
        return sorted(self._strategies)

    def __len__(self) -> int:
        """Anzahl registrierter Strategien.

        Returns:
            Anzahl der Eintraege.
        """
        return len(self._strategies)

    def __contains__(self, key: object) -> bool:
        """Prueft, ob ein Key registriert ist.

        Args:
            key: Zu pruefender Bezeichner.

        Returns:
            True wenn der Key vergeben ist.
        """
        return key in self._strategies


def default_registry() -> InMemoryStrategyRegistry:
    """Baut eine Registry mit allen mitgelieferten Strategien.

    Returns:
        Frische Registry mit den vier Referenzstrategien aus
        :data:`BUILTIN_STRATEGIES`.
    """
    registry = InMemoryStrategyRegistry()
    for strategy_cls in BUILTIN_STRATEGIES:
        registry.register(strategy_cls)
    return registry
