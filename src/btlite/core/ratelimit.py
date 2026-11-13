"""Token-Bucket-Rate-Limiter und Retry-Backoff fuer die Datenschicht.

Architekturregel des Projekts: Rate Limits werden **nie umgangen**. Dieses
Modul drosselt ausgehende Requests konservativ und reicht ein echtes
``RateLimitError`` unveraendert nach oben - es wird nicht stillschweigend
weggeschluckt und auch nicht durch Retries "ausgesessen".

Zwei Bausteine:

* :class:`TokenBucket` - thread-safe, fuellt mit ``rps`` Token pro Sekunde
  nach und haelt maximal ``burst`` Token vor.
* :func:`retry_with_backoff` - exponentieller Backoff mit Jitter fuer
  transiente Fehler.

Beide beziehen ihre Parameter aus :class:`btlite.config.RateLimitSettings`.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from btlite.config import RateLimitSettings, get_settings
from btlite.core.errors import ProviderUnavailableError, RateLimitError
from btlite.core.logging import get_logger

_LOG = get_logger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")


class TokenBucket:
    """Thread-safe Token-Bucket zur Begrenzung der Requestrate.

    Der Bucket startet voll (``burst`` Token) und fuellt kontinuierlich mit
    ``rps`` Token pro Sekunde nach. :meth:`acquire` blockiert genau so lange,
    bis wieder genug Token vorhanden sind.

    Ein voller Bucket erlaubt bewusst einen kurzen Burst: das erste Dutzend
    Requests einer Sitzung darf sofort raus, der Dauerdurchsatz bleibt
    trotzdem auf ``rps`` begrenzt.
    """

    def __init__(self, rps: float, burst: int, *, name: str = "bucket") -> None:
        """Initialisiert den Bucket.

        Args:
            rps: Nachfuellrate in Token pro Sekunde. Muss groesser als 0 sein.
            burst: Kapazitaet in Token. Muss mindestens 1 sein.
            name: Bezeichner fuer Logmeldungen.

        Raises:
            ValueError: Wenn ``rps`` nicht positiv oder ``burst`` kleiner 1 ist.
        """
        if rps <= 0:
            msg = f"rps muss > 0 sein, erhalten {rps}"
            raise ValueError(msg)
        if burst < 1:
            msg = f"burst muss >= 1 sein, erhalten {burst}"
            raise ValueError(msg)

        self._rps = float(rps)
        self._burst = float(burst)
        self._name = name
        self._tokens = float(burst)
        self._lock = threading.Lock()
        self._last_refill = time.monotonic()

    @property
    def rps(self) -> float:
        """Nachfuellrate in Token pro Sekunde.

        Returns:
            Konfigurierte Rate, Token pro Sekunde.
        """
        return self._rps

    @property
    def burst(self) -> int:
        """Kapazitaet des Buckets in Token.

        Returns:
            Maximal vorhaltbare Tokenzahl.
        """
        return int(self._burst)

    def _refill_locked(self) -> None:
        """Schreibt die seit dem letzten Aufruf entstandenen Token gut.

        Muss unter gehaltenem Lock aufgerufen werden.
        """
        now = time.monotonic()
        elapsed_s = now - self._last_refill
        self._last_refill = now
        self._tokens = min(self._burst, self._tokens + elapsed_s * self._rps)

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Versucht, Token zu entnehmen, ohne zu blockieren.

        Args:
            tokens: Anzahl der zu entnehmenden Token.

        Returns:
            True wenn die Entnahme gelungen ist, sonst False.
        """
        with self._lock:
            self._refill_locked()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def acquire(self, tokens: float = 1.0, *, timeout_s: float | None = None) -> float:
        """Entnimmt Token und blockiert, bis sie verfuegbar sind.

        Args:
            tokens: Anzahl der zu entnehmenden Token.
            timeout_s: Maximale Wartezeit in Sekunden. ``None`` wartet
                unbegrenzt.

        Returns:
            Tatsaechlich gewartete Zeit in Sekunden.

        Raises:
            ValueError: Wenn mehr Token angefordert werden, als der Bucket
                je halten kann - das koennte nie erfuellt werden.
            ProviderUnavailableError: Wenn ``timeout_s`` ueberschritten wurde.
        """
        if tokens > self._burst:
            msg = f"{tokens} Token angefordert, Bucket fasst nur {self._burst}"
            raise ValueError(msg)

        waited_s = 0.0
        while True:
            with self._lock:
                self._refill_locked()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited_s
                deficit = tokens - self._tokens
                sleep_s = deficit / self._rps

            if timeout_s is not None and waited_s + sleep_s > timeout_s:
                msg = "Rate limiter wait time exceeded"
                raise ProviderUnavailableError(
                    msg,
                    detail=f"bucket={self._name}, timeout_s={timeout_s}",
                )

            time.sleep(sleep_s)
            waited_s += sleep_s

    def __enter__(self) -> TokenBucket:
        """Entnimmt beim Betreten des Kontexts genau einen Token.

        Returns:
            Der Bucket selbst.
        """
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        """Verlaesst den Kontext. Token werden nicht zurueckgegeben."""
        return None


def backoff_delay_s(
    attempt: int,
    settings: RateLimitSettings | None = None,
    *,
    rng: random.Random | None = None,
) -> float:
    """Berechnet die Wartezeit vor einem Retry-Versuch.

    Exponentieller Backoff mit symmetrischem Jitter::

        base * 2**attempt, gedeckelt auf backoff_max_s,
        multipliziert mit (1 +/- backoff_jitter)

    Args:
        attempt: Nullbasierter Versuchszaehler. 0 ist der erste Retry.
        settings: Rate-Limit-Settings. ``None`` nimmt die globalen.
        rng: Zufallsquelle fuer den Jitter. ``None`` nimmt das Modul-RNG.
            In Tests injizierbar, damit die Wartezeit deterministisch wird.

    Returns:
        Wartezeit in Sekunden, immer groesser oder gleich 0.
    """
    cfg = settings if settings is not None else get_settings().ratelimit
    source = rng if rng is not None else random
    raw_s = min(cfg.backoff_base_s * (2.0**attempt), cfg.backoff_max_s)
    jitter = source.uniform(-cfg.backoff_jitter, cfg.backoff_jitter)
    return max(0.0, raw_s * (1.0 + jitter))


def retry_with_backoff(
    *,
    settings: RateLimitSettings | None = None,
    retry_on: tuple[type[Exception], ...] = (ProviderUnavailableError,),
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Dekorator: wiederholt transiente Fehler mit exponentiellem Backoff.

    Wichtig: :class:`~btlite.core.errors.RateLimitError` wird **nie**
    wiederholt. Wenn eine Quelle drosselt, ist das kein transienter Fehler,
    sondern eine Aussage. Der Fehler geht unveraendert nach oben, das UI zeigt
    ihn an. Das ist eine harte Projektregel, kein Stilwunsch.

    Args:
        settings: Rate-Limit-Settings mit ``max_retries``, ``backoff_base_s``,
            ``backoff_max_s`` und ``backoff_jitter``. ``None`` nimmt die
            globalen Settings.
        retry_on: Exception-Typen, die als transient gelten.
        sleep: Wartefunktion, Sekunden als Argument. In Tests injizierbar.
        rng: Zufallsquelle fuer den Jitter. In Tests injizierbar.

    Returns:
        Dekorator, der die gewrappte Funktion mit Retry-Logik versieht.
    """

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            cfg = settings if settings is not None else get_settings().ratelimit
            last_exc: Exception | None = None
            for attempt in range(cfg.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RateLimitError:
                    # Bewusst nicht wiederholen - siehe Docstring.
                    raise
                except retry_on as exc:
                    last_exc = exc
                    if attempt >= cfg.max_retries:
                        break
                    delay_s = backoff_delay_s(attempt, cfg, rng=rng)
                    _LOG.warning(
                        "Transienter Fehler in %s (Versuch %d/%d), warte %.2f s: %s",
                        func.__qualname__,
                        attempt + 1,
                        cfg.max_retries + 1,
                        delay_s,
                        exc,
                    )
                    sleep(delay_s)
            if last_exc is None:  # pragma: no cover - Schleife bricht nur mit Fehler ab
                msg = "Retry loop ended without a result and without an error"
                raise ProviderUnavailableError(msg, detail=func.__qualname__)
            raise last_exc

        return wrapper

    return decorator


class RateLimiterRegistry:
    """Haelt genau einen Bucket je Datenquelle.

    Die Provider teilen sich diese Buckets prozessweit. Zwei Instanzen von
    ``YFinanceProvider`` drosseln damit gemeinsam und nicht jede fuer sich.
    """

    def __init__(self, settings: RateLimitSettings | None = None) -> None:
        """Initialisiert die Registry aus den Settings.

        Args:
            settings: Rate-Limit-Settings. ``None`` nimmt die globalen.
        """
        cfg = settings if settings is not None else get_settings().ratelimit
        self._settings = cfg
        self._buckets: dict[str, TokenBucket] = {
            "yfinance": TokenBucket(cfg.yfinance_rps, cfg.yfinance_burst, name="yfinance"),
            "finviz": TokenBucket(cfg.finviz_rps, cfg.finviz_burst, name="finviz"),
        }

    @property
    def settings(self) -> RateLimitSettings:
        """Die zugrunde liegenden Rate-Limit-Settings.

        Returns:
            Die Settings-Instanz.
        """
        return self._settings

    def bucket(self, source: str) -> TokenBucket:
        """Liefert den Bucket einer Quelle.

        Args:
            source: Name der Quelle, etwa ``"yfinance"`` oder ``"finviz"``.

        Returns:
            Der zugehoerige Token-Bucket.

        Raises:
            KeyError: Wenn die Quelle unbekannt ist.
        """
        return self._buckets[source]


_REGISTRY: RateLimiterRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def get_registry() -> RateLimiterRegistry:
    """Liefert die prozessweite Rate-Limiter-Registry.

    Returns:
        Die gemeinsam genutzte :class:`RateLimiterRegistry`.
    """
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            _REGISTRY = RateLimiterRegistry()
        return _REGISTRY
