"""Protocols und Basisklassen der Backtest-Schicht.

Die Trennung ist bewusst zweistufig:
    1. Eine :class:`Strategy` erzeugt aus Kursdaten einen :class:`SignalSet`.
       Sie kennt keine Engine, keine Kosten und kein Kapital.
    2. Eine :class:`BacktestEngine` fuehrt diese Signale aus und liefert ein
       :class:`BacktestResult`.

Dadurch sind Strategien engine-unabhaengig testbar: die Signal-Erzeugung
laesst sich ohne vectorbt oder backtrader pruefen.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from btlite.core.errors import StrategyConfigError
from btlite.models.backtest import (
    BacktestRequest,
    BacktestResult,
    SignalSet,
    StrategyConfig,
)
from btlite.models.market import PriceSeries


@runtime_checkable
class BacktestEngine(Protocol):
    """Fuehrt Signale gegen Kursdaten aus.

    Implementierungen: ``VectorbtEngine`` (primaer) und ``BacktraderEngine``
    (optionaler Adapter). Beide fuellen dasselbe Ergebnis-Modell.
    """

    @property
    def name(self) -> str:
        """Bezeichner der Engine, erscheint in ``BacktestResult.engine_used``.

        Returns:
            Name der Engine, etwa ``"vectorbt"``.
        """
        ...

    def is_available(self) -> bool:
        """Prueft, ob die zugrunde liegende Bibliothek importierbar ist.

        ``vectorbt`` und ``backtrader`` sind optionale Extras. Das UI fragt
        dies ab, bevor es eine Engine anbietet.

        Returns:
            True wenn die Engine nutzbar ist.
        """
        ...

    def run(
        self,
        request: BacktestRequest,
        prices: dict[str, PriceSeries],
        signals: SignalSet,
        *,
        benchmark: PriceSeries | None = None,
    ) -> BacktestResult:
        """Fuehrt den Backtest aus.

        Die Engine holt **keine** Daten selbst - Kurse und Benchmark werden
        injiziert.

        Args:
            request: Parameter des Laufs inklusive Kosten und Startkapital.
            prices: Kursreihen je Ticker, bereits auf den Zeitraum beschnitten.
            signals: Ein- und Ausstiegssignale.
            benchmark: Optionale Vergleichsreihe fuer Beta und Alpha.

        Returns:
            Vollstaendiges Ergebnis mit Equity-Kurve, Trades und Metriken.

        Raises:
            EngineUnavailableError: Bibliothek nicht installiert.
            BacktestError: Ausfuehrung fehlgeschlagen.
            InsufficientDataError: Zu wenige Bars fuer den Zeitraum.
        """
        ...


class Strategy(ABC):
    """Basisklasse aller Strategien.

    Eine Unterklasse implementiert :meth:`generate_signals` und optional
    :meth:`validate_params`. Sie darf **nicht** auf Netzwerk oder globalen
    Zustand zugreifen - alle Eingaben kommen ueber die Parameter.

    Attributes:
        key: Stabiler Bezeichner, muss ueber alle Strategien eindeutig sein.
        label: Anzeigename fuer das UI.
        default_params: Vorbelegung der Parameter.
    """

    key: str = ""
    label: str = ""
    default_params: dict[str, Any] = {}  # noqa: RUF012 - bewusst Klassenattribut

    def __init__(self, config: StrategyConfig | None = None) -> None:
        """Initialisiert die Strategie mit gemergten Parametern.

        Args:
            config: Konfiguration. None nutzt ausschliesslich die Defaults.

        Raises:
            StrategyConfigError: Wenn die Parameter unzulaessig sind.
        """
        self.config = config or StrategyConfig(key=self.key, params={})
        self.params: dict[str, Any] = {**self.default_params, **self.config.params}
        self.validate_params(self.params)

    def validate_params(self, params: dict[str, Any]) -> None:
        """Prueft die Parameter der Strategie.

        Die Basisimplementierung lehnt Parameter ab, die in
        :attr:`default_params` nicht vorgesehen sind. Damit fallen Tippfehler
        in Parameternamen sofort auf, statt still ignoriert zu werden.
        Unterklassen erweitern sie um fachliche Pruefungen und rufen
        ``super().validate_params(params)`` auf.

        Args:
            params: Zusammengefuehrte Parameter.

        Raises:
            StrategyConfigError: Bei unbekannten oder unzulaessigen Werten.
        """
        unknown = sorted(set(params) - set(self.default_params))
        if unknown:
            msg = f"Unknown parameters for strategy {self.key!r}: {unknown}"
            raise StrategyConfigError(msg, detail=f"erlaubt={sorted(self.default_params)}")

    @abstractmethod
    def generate_signals(self, prices: dict[str, PriceSeries]) -> SignalSet:
        """Erzeugt Ein- und Ausstiegssignale aus Kursdaten.

        Die Implementierung darf **keinen** Look-ahead-Bias erzeugen: ein
        Signal in Zeile t darf ausschliesslich Daten bis einschliesslich t
        verwenden. Die Ausfuehrungsverzoegerung regelt
        ``SignalSet.execution_lag``.

        Args:
            prices: Kursreihen je Ticker mit gemeinsamem UTC-Index.

        Returns:
            Signalmatrix mit einer Spalte je Ticker.

        Raises:
            InsufficientDataError: Zu wenige Bars fuer die Parameter,
                etwa ein SMA-Fenster laenger als die Historie.
        """
        raise NotImplementedError

    def require_positive_int(self, params: dict[str, Any], name: str) -> int:
        """Liest einen Parameter als positive ganze Zahl.

        Hilfsmethode fuer :meth:`validate_params`.

        Args:
            params: Parameter-Dictionary.
            name: Name des Parameters.

        Returns:
            Der validierte Wert.

        Raises:
            StrategyConfigError: Wenn der Wert fehlt oder nicht positiv ist.
        """
        value = params.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            msg = f"Parameter {name!r} must be a positive integer"
            raise StrategyConfigError(msg, detail=f"erhalten={value!r}")
        return value


@runtime_checkable
class StrategyRegistry(Protocol):
    """Nachschlagewerk der verfuegbaren Strategien.

    Das UI fuellt damit sein Auswahlmenue, ohne die Strategie-Module zu
    kennen.
    """

    def register(self, strategy_cls: type[Strategy]) -> None:
        """Nimmt eine Strategie in die Registry auf.

        Args:
            strategy_cls: Zu registrierende Klasse.

        Raises:
            StrategyConfigError: Wenn der Key bereits vergeben ist.
        """
        ...

    def get(self, key: str) -> type[Strategy]:
        """Sucht eine Strategie nach ihrem Key.

        Args:
            key: Bezeichner der Strategie.

        Returns:
            Die registrierte Klasse.

        Raises:
            StrategyConfigError: Wenn der Key unbekannt ist.
        """
        ...

    def list_all(self) -> list[type[Strategy]]:
        """Listet alle registrierten Strategien.

        Returns:
            Registrierte Klassen, stabil nach Key sortiert.
        """
        ...
