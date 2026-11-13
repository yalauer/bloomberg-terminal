"""Polling-basierter Quote-Stream.

yfinance liefert **keinen** Push-Feed und keine Echtzeitdaten. Statt das zu
verschweigen, macht dieses Modul es explizit: :class:`PollingQuoteStream`
pollt in einem konfigurierbaren Intervall (Default 15 s) und holt dabei
**alle** abonnierten Symbole in genau einem Request pro Poll. Jedes
:class:`~btlite.models.market.Quote` traegt eine
:class:`~btlite.models.common.Provenance` mit ``delay_minutes``, damit das UI
den Verzug sichtbar anzeigen kann.

Die Ein-Request-Garantie ist die Grundlage des Akzeptanzkriteriums: 20 Ticker
ueber 10 Minuten im 15-Sekunden-Takt sind 40 Requests, nicht 800. Genau darum
verlangt :meth:`~btlite.data.base.PriceProvider.get_quotes` einen einzigen
Request und nicht eine Schleife.

Robustheit: ein fehlgeschlagener Poll beendet den Iterator **nicht**. Der
Stream geht in exponentiellen Backoff und liefert weiter, sobald die Quelle
wieder antwortet.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Iterable
from typing import TYPE_CHECKING, Protocol

from btlite.config import Settings, get_settings
from btlite.core.errors import RateLimitError
from btlite.core.logging import get_logger
from btlite.core.ratelimit import backoff_delay_s

if TYPE_CHECKING:
    from collections.abc import Sequence

    from btlite.models.market import Quote

_LOG = get_logger(__name__)


class _QuoteSource(Protocol):
    """Minimalvertrag, den der Stream von seiner Quelle braucht.

    Bewusst schmaler als :class:`~btlite.data.base.PriceProvider`: der Stream
    holt nur Quotes. Damit genuegt in Tests ein Objekt mit dieser einen
    Methode, und der :class:`~btlite.data.fixtures.FixtureProvider` passt
    ebenso hinein wie der echte Provider.
    """

    def get_quotes(self, tickers: Sequence[str]) -> dict[str, Quote]:
        """Holt Momentaufnahmen fuer mehrere Symbole in einem Request.

        Args:
            tickers: Kanonische Symbole.

        Returns:
            Abbildung Symbol auf Quote.
        """
        ...


class PollingQuoteStream:
    """Fragt Quotes in festem Intervall ab und liefert sie als Stream.

    Erfuellt :class:`~btlite.data.base.QuoteStream` strukturell, ohne von dem
    Protocol zu erben.

    Der Stream ist kein Echtzeitfeed. Er macht aus einer Quelle, die nur
    Momentaufnahmen kennt, einen fortlaufenden Strom - mit allen Grenzen der
    Quelle, die die ``Provenance`` jedes Quote mitfuehrt.
    """

    def __init__(
        self,
        provider: _QuoteSource,
        settings: Settings | None = None,
        *,
        interval_s: float | None = None,
        rng: random.Random | None = None,
    ) -> None:
        """Initialisiert den Stream.

        Args:
            provider: Quelle der Quotes. Muss ``get_quotes`` in genau einem
                Request je Aufruf erfuellen.
            settings: Projekt-Settings. ``None`` nimmt die globalen.
            interval_s: Poll-Intervall in Sekunden. ``None`` nimmt
                ``DataSettings.quote_poll_interval_s`` (Default 15 s).
            rng: Zufallsquelle fuer den Backoff-Jitter. In Tests injizierbar.
        """
        self._provider = provider
        self._settings = settings if settings is not None else get_settings()
        self._interval_s = (
            self._settings.data.quote_poll_interval_s if interval_s is None else interval_s
        )
        self._rng = rng
        self._closed = asyncio.Event()
        self._poll_count = 0

    @property
    def interval_s(self) -> float:
        """Poll-Intervall in Sekunden.

        Returns:
            Zeit zwischen zwei Polls.
        """
        return self._interval_s

    @property
    def poll_count(self) -> int:
        """Anzahl bisher ausgefuehrter Polls.

        Zaehlt Versuche, nicht Erfolge. Ein fehlgeschlagener Poll zaehlt mit.

        Returns:
            Bisherige Poll-Versuche.
        """
        return self._poll_count

    @property
    def is_closed(self) -> bool:
        """Ob der Stream bereits geschlossen wurde.

        Returns:
            True nach :meth:`aclose`.
        """
        return self._closed.is_set()

    async def subscribe(self, tickers: Iterable[str]) -> AsyncIterator[Quote]:
        """Abonniert Kursaktualisierungen fuer eine Symbolmenge.

        Der Iterator laeuft, bis :meth:`aclose` gerufen oder er selbst
        geschlossen wird. Ein fehlgeschlagener Poll beendet ihn nicht: der
        Stream wartet mit exponentiellem Backoff und liefert weiter, sobald
        die Quelle wieder antwortet.

        Ein :class:`~btlite.core.errors.RateLimitError` wird nicht umgangen.
        Der Stream wartet die volle Backoff-Zeit ab, statt sofort erneut
        anzufragen - und loggt ihn sichtbar.

        Args:
            tickers: Zu abonnierende Symbole. Duplikate werden entfernt, die
                Reihenfolge bleibt stabil.

        Yields:
            Einzelne Quotes. Je Poll ein Quote pro Symbol, fuer das die
            Quelle einen verwertbaren Preis geliefert hat.
        """
        from btlite.models.common import normalize_ticker

        seen: set[str] = set()
        symbols: list[str] = []
        for raw in tickers:
            symbol = normalize_ticker(raw)
            if symbol not in seen:
                seen.add(symbol)
                symbols.append(symbol)

        if not symbols:
            _LOG.warning("PollingQuoteStream ohne Symbole abonniert - liefert nichts.")
            return

        failures = 0
        while not self._closed.is_set():
            self._poll_count += 1
            try:
                # EIN Request fuer alle Symbole. Siehe Modul-Docstring.
                quotes = await asyncio.to_thread(self._provider.get_quotes, symbols)
            except RateLimitError as exc:
                failures += 1
                delay_s = self._backoff_s(failures, exc.retry_after_s)
                _LOG.warning(
                    "Quelle drosselt den Quote-Poll, warte %.1f s: %s", delay_s, exc.user_message
                )
                if await self._sleep(delay_s):
                    return
                continue
            except Exception as exc:  # Stream ueberlebt jeden Poll-Fehler
                failures += 1
                delay_s = self._backoff_s(failures, None)
                _LOG.warning("Quote-Poll fehlgeschlagen, warte %.1f s: %s", delay_s, exc)
                if await self._sleep(delay_s):
                    return
                continue

            failures = 0
            for symbol in symbols:
                quote = quotes.get(symbol)
                if quote is not None:
                    yield quote

            if await self._sleep(self._interval_s):
                return

    def _backoff_s(self, failures: int, retry_after_s: float | None) -> float:
        """Berechnet die Wartezeit nach einem fehlgeschlagenen Poll.

        Args:
            failures: Anzahl aufeinanderfolgender Fehlschlaege, mindestens 1.
            retry_after_s: Von der Quelle genannte Wartezeit in Sekunden,
                falls vorhanden. Sie hat Vorrang - eine Quelle, die sagt, wie
                lange sie Ruhe braucht, wird nicht ueberstimmt.

        Returns:
            Wartezeit in Sekunden, mindestens das Poll-Intervall.
        """
        if retry_after_s is not None:
            return max(retry_after_s, self._interval_s)
        delay_s = backoff_delay_s(failures - 1, self._settings.ratelimit, rng=self._rng)
        return max(delay_s, self._interval_s)

    async def _sleep(self, seconds: float) -> bool:
        """Wartet, bricht aber sofort ab, wenn der Stream geschlossen wird.

        Args:
            seconds: Wartezeit in Sekunden.

        Returns:
            True wenn waehrend des Wartens geschlossen wurde, sonst False.
        """
        try:
            await asyncio.wait_for(self._closed.wait(), timeout=seconds)
        except TimeoutError:
            return False
        return True

    async def aclose(self) -> None:
        """Beendet den Stream und gibt alle Ressourcen frei.

        Laufende Iteratoren beenden sich beim naechsten Warteschritt. Ein
        zweiter Aufruf ist wirkungslos.
        """
        self._closed.set()

    async def __aenter__(self) -> PollingQuoteStream:
        """Betritt den asynchronen Kontext.

        Returns:
            Der Stream selbst.
        """
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Schliesst den Stream beim Verlassen des Kontexts."""
        await self.aclose()
