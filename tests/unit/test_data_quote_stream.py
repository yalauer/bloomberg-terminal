"""Tests des PollingQuoteStream.

Kernzusagen, die hier abgesichert werden:

* **Ein** Request je Poll fuer alle Symbole - Grundlage dafuer, dass 20 Ticker
  ueber 10 Minuten nicht ins Rate Limit laufen.
* Ein fehlgeschlagener Poll beendet den Iterator **nicht**.
* ``aclose`` beendet den Stream sauber und zeitnah.

Die Tests treiben die Coroutinen ueber :func:`run_async` selbst an, statt
``pytest-asyncio`` vorauszusetzen - das Projekt fuehrt es nicht als
Abhaengigkeit, und ein Test soll keine neue erzwingen.
"""

from __future__ import annotations

import asyncio
import random
from datetime import UTC, datetime

import pytest

from btlite.config import Settings
from btlite.core.errors import ProviderUnavailableError, RateLimitError
from btlite.data.base import QuoteStream
from btlite.data.quote_stream import PollingQuoteStream
from btlite.models.common import DataSource, Provenance
from btlite.models.market import Quote


def run_async(coro):
    """Fuehrt eine Coroutine in einem frischen Event Loop aus.

    Ersetzt ``pytest-asyncio``, das im Projekt nicht installiert ist. Jeder
    Test bekommt einen eigenen Loop, damit kein Zustand ueberspringt.

    Args:
        coro: Auszufuehrende Coroutine.

    Returns:
        Das Ergebnis der Coroutine.
    """
    return asyncio.run(coro)


def make_quote(ticker: str, price: float = 100.0) -> Quote:
    """Baut ein Quote fuer Testzwecke."""
    return Quote(
        ticker=ticker,
        price=price,
        timestamp=datetime(2025, 1, 2, 15, 30, tzinfo=UTC),
        provenance=Provenance(
            source=DataSource.FIXTURE,
            fetched_at=datetime(2025, 1, 2, 15, 30, tzinfo=UTC),
            delay_minutes=15,
        ),
    )


class RecordingSource:
    """Quote-Quelle, die jeden Aufruf mitschreibt."""

    def __init__(self, *, fail_times: int = 0, error: Exception | None = None) -> None:
        """Initialisiert die Quelle mit einer Zahl vorgetaeuschter Fehlschlaege."""
        self.calls: list[list[str]] = []
        self._fail_times = fail_times
        self._error = error or ProviderUnavailableError("Quelle kurz weg")

    def get_quotes(self, tickers):
        self.calls.append(list(tickers))
        if len(self.calls) <= self._fail_times:
            raise self._error
        return {t: make_quote(t, 100.0 + i) for i, t in enumerate(tickers)}

    @property
    def call_count(self) -> int:
        return len(self.calls)


@pytest.fixture
def fast_settings():
    """Settings mit sehr kurzem Poll-Intervall, damit Tests schnell laufen."""
    settings = Settings()
    settings.data.quote_poll_interval_s = 1.0
    settings.ratelimit.backoff_base_s = 0.01
    settings.ratelimit.backoff_jitter = 0.0
    return settings


async def take(stream_iter, n: int, timeout: float = 5.0) -> list[Quote]:
    """Nimmt die naechsten ``n`` Quotes aus einem Iterator."""

    async def _collect() -> list[Quote]:
        out: list[Quote] = []
        async for quote in stream_iter:
            out.append(quote)
            if len(out) >= n:
                break
        return out

    return await asyncio.wait_for(_collect(), timeout=timeout)


class TestProtocolCompliance:
    def test_satisfies_the_quote_stream_protocol(self, fast_settings):
        stream = PollingQuoteStream(RecordingSource(), fast_settings)
        assert isinstance(stream, QuoteStream)

    def test_subscribe_returns_an_async_iterator(self, fast_settings):
        stream = PollingQuoteStream(RecordingSource(), fast_settings)
        it = stream.subscribe(["AAPL"])
        assert hasattr(it, "__anext__")


class TestSingleRequestPerPoll:
    def test_one_call_covers_all_symbols(self, fast_settings):
        async def _body():
            source = RecordingSource()
            stream = PollingQuoteStream(source, fast_settings, interval_s=0.02)
            symbols = ["AAPL", "MSFT", "NVDA", "SPY", "JNJ"]

            quotes = await take(stream.subscribe(symbols), len(symbols))
            await stream.aclose()

            assert source.call_count == 1, "Alle Symbole in EINEM Request"
            assert source.calls[0] == symbols
            assert [q.ticker for q in quotes] == symbols

        run_async(_body())

    def test_twenty_tickers_cost_one_request_per_poll(self, fast_settings):
        async def _body():
            # Akzeptanzkriterium: 20 Ticker duerfen je Poll nur einen Request kosten.
            source = RecordingSource()
            stream = PollingQuoteStream(source, fast_settings, interval_s=0.02)
            symbols = [f"T{i:02d}" for i in range(20)]

            await take(stream.subscribe(symbols), 40)  # zwei volle Polls
            await stream.aclose()

            assert source.call_count == 2, "Zwei Polls, zwei Requests - nicht 40"

        run_async(_body())

    def test_duplicates_are_removed_before_the_request(self, fast_settings):
        async def _body():
            source = RecordingSource()
            stream = PollingQuoteStream(source, fast_settings, interval_s=0.02)

            await take(stream.subscribe(["AAPL", "aapl", "AAPL", "MSFT"]), 2)
            await stream.aclose()

            assert source.calls[0] == ["AAPL", "MSFT"], "Duplikate kosten keinen Platz"

        run_async(_body())

    def test_tickers_are_normalized(self, fast_settings):
        async def _body():
            source = RecordingSource()
            stream = PollingQuoteStream(source, fast_settings, interval_s=0.02)

            await take(stream.subscribe([" aapl ", "msft"]), 2)
            await stream.aclose()

            assert source.calls[0] == ["AAPL", "MSFT"]

        run_async(_body())

    def test_empty_subscription_yields_nothing(self, fast_settings):
        async def _body():
            source = RecordingSource()
            stream = PollingQuoteStream(source, fast_settings, interval_s=0.02)

            out = [q async for q in stream.subscribe([])]

            assert out == []
            assert source.call_count == 0

        run_async(_body())


class TestPollingCadence:
    def test_polls_repeatedly(self, fast_settings):
        async def _body():
            source = RecordingSource()
            stream = PollingQuoteStream(source, fast_settings, interval_s=0.02)

            quotes = await take(stream.subscribe(["AAPL"]), 3)
            await stream.aclose()

            assert len(quotes) == 3
            assert source.call_count == 3, "Je Poll ein Request"

        run_async(_body())

    def test_interval_comes_from_settings(self, fast_settings):
        async def _body():
            fast_settings.data.quote_poll_interval_s = 7.5
            stream = PollingQuoteStream(RecordingSource(), fast_settings)
            assert stream.interval_s == 7.5

        run_async(_body())

    def test_explicit_interval_overrides_settings(self, fast_settings):
        async def _body():
            stream = PollingQuoteStream(RecordingSource(), fast_settings, interval_s=0.5)
            assert stream.interval_s == 0.5

        run_async(_body())

    def test_waits_between_polls(self, fast_settings):
        async def _body():
            source = RecordingSource()
            stream = PollingQuoteStream(source, fast_settings, interval_s=0.15)

            start = asyncio.get_running_loop().time()
            await take(stream.subscribe(["AAPL"]), 2)
            elapsed = asyncio.get_running_loop().time() - start
            await stream.aclose()

            assert elapsed >= 0.15, "Zwischen zwei Polls muss das Intervall liegen"

        run_async(_body())


class TestFailureResilience:
    def test_iterator_survives_a_failed_poll(self, fast_settings):
        async def _body():
            # Der zentrale Punkt: ein Fehlschlag beendet den Stream NICHT.
            source = RecordingSource(fail_times=2)
            stream = PollingQuoteStream(
                source, fast_settings, interval_s=0.02, rng=random.Random(0)
            )

            quotes = await take(stream.subscribe(["AAPL"]), 1)
            await stream.aclose()

            assert len(quotes) == 1, "Nach zwei Fehlschlaegen kommt wieder ein Quote"
            assert source.call_count == 3

        run_async(_body())

    def test_rate_limit_does_not_kill_the_stream(self, fast_settings):
        async def _body():
            source = RecordingSource(fail_times=1, error=RateLimitError("gedrosselt"))
            stream = PollingQuoteStream(
                source, fast_settings, interval_s=0.02, rng=random.Random(0)
            )

            quotes = await take(stream.subscribe(["AAPL"]), 1)
            await stream.aclose()

            assert len(quotes) == 1
            assert source.call_count == 2

        run_async(_body())

    def test_retry_after_is_respected_over_the_interval(self, fast_settings):
        async def _body():
            stream = PollingQuoteStream(RecordingSource(), fast_settings, interval_s=0.02)
            # Die Quelle sagt, wie lange sie Ruhe braucht - das wird nicht
            # unterboten.
            assert stream._backoff_s(1, retry_after_s=5.0) == 5.0

        run_async(_body())

    def test_backoff_never_undercuts_the_interval(self, fast_settings):
        async def _body():
            stream = PollingQuoteStream(
                RecordingSource(), fast_settings, interval_s=3.0, rng=random.Random(0)
            )
            assert stream._backoff_s(1, None) >= 3.0

        run_async(_body())

    def test_backoff_grows_with_consecutive_failures(self, fast_settings):
        async def _body():
            fast_settings.ratelimit.backoff_base_s = 1.0
            fast_settings.ratelimit.backoff_max_s = 100.0
            stream = PollingQuoteStream(RecordingSource(), fast_settings, interval_s=0.01)
            delays = [stream._backoff_s(i, None) for i in (1, 2, 3)]
            assert delays[0] < delays[1] < delays[2]

        run_async(_body())

    def test_failure_counter_resets_after_success(self, fast_settings):
        async def _body():
            source = RecordingSource(fail_times=1)
            stream = PollingQuoteStream(
                source, fast_settings, interval_s=0.02, rng=random.Random(0)
            )

            quotes = await take(stream.subscribe(["AAPL"]), 2)
            await stream.aclose()

            assert len(quotes) == 2, "Nach dem ersten Erfolg laeuft es normal weiter"

        run_async(_body())

    def test_symbols_without_a_quote_are_skipped_silently(self, fast_settings):
        async def _body():
            class PartialSource:
                def __init__(self) -> None:
                    self.calls = 0

                def get_quotes(self, tickers):
                    self.calls += 1
                    return {"AAPL": make_quote("AAPL")}

            source = PartialSource()
            stream = PollingQuoteStream(source, fast_settings, interval_s=0.02)

            quotes = await take(stream.subscribe(["AAPL", "ZZZZ"]), 2)
            await stream.aclose()

            assert {q.ticker for q in quotes} == {"AAPL"}

        run_async(_body())


class TestClose:
    def test_aclose_marks_the_stream_closed(self, fast_settings):
        async def _body():
            stream = PollingQuoteStream(RecordingSource(), fast_settings, interval_s=0.02)
            assert stream.is_closed is False
            await stream.aclose()
            assert stream.is_closed is True

        run_async(_body())

    def test_aclose_is_idempotent(self, fast_settings):
        async def _body():
            stream = PollingQuoteStream(RecordingSource(), fast_settings, interval_s=0.02)
            await stream.aclose()
            await stream.aclose()
            assert stream.is_closed is True

        run_async(_body())

    def test_aclose_stops_a_running_iterator_promptly(self, fast_settings):
        async def _body():
            source = RecordingSource()
            # Langes Intervall: ohne sauberes Abbrechen wuerde der Test in den
            # Timeout laufen statt zuegig zurueckzukommen.
            stream = PollingQuoteStream(source, fast_settings, interval_s=30.0)

            collected: list[Quote] = []

            async def consume() -> None:
                async for quote in stream.subscribe(["AAPL"]):
                    collected.append(quote)

            task = asyncio.create_task(consume())
            await asyncio.sleep(0.05)
            await stream.aclose()
            await asyncio.wait_for(task, timeout=2.0)

            assert len(collected) == 1, "Der erste Poll kam noch durch"
            assert source.call_count == 1

        run_async(_body())

    def test_closed_stream_yields_nothing(self, fast_settings):
        async def _body():
            source = RecordingSource()
            stream = PollingQuoteStream(source, fast_settings, interval_s=0.02)
            await stream.aclose()

            out = [q async for q in stream.subscribe(["AAPL"])]

            assert out == []
            assert source.call_count == 0

        run_async(_body())

    def test_context_manager_closes_on_exit(self, fast_settings):
        async def _body():
            async with PollingQuoteStream(
                RecordingSource(), fast_settings, interval_s=0.02
            ) as stream:
                assert stream.is_closed is False
            assert stream.is_closed is True

        run_async(_body())

    def test_poll_count_is_exposed(self, fast_settings):
        async def _body():
            source = RecordingSource()
            stream = PollingQuoteStream(source, fast_settings, interval_s=0.02)
            await take(stream.subscribe(["AAPL"]), 2)
            await stream.aclose()
            assert stream.poll_count == 2

        run_async(_body())


class TestProvenance:
    def test_quotes_carry_a_delay_notice(self, fast_settings):
        async def _body():
            stream = PollingQuoteStream(RecordingSource(), fast_settings, interval_s=0.02)
            quotes = await take(stream.subscribe(["AAPL"]), 1)
            await stream.aclose()
            # yfinance ist nicht Echtzeit - das muss am Quote sichtbar bleiben.
            assert quotes[0].provenance.delay_minutes == 15

        run_async(_body())
