"""Tests des Token-Bucket-Rate-Limiters und des Retry-Backoff."""

from __future__ import annotations

import random
import threading
import time

import pytest

from btlite.config import RateLimitSettings
from btlite.core.errors import ProviderUnavailableError, RateLimitError, SymbolNotFoundError
from btlite.core.ratelimit import (
    RateLimiterRegistry,
    TokenBucket,
    backoff_delay_s,
    get_registry,
    retry_with_backoff,
)


class TestTokenBucket:
    def test_starts_full_so_burst_passes_immediately(self):
        bucket = TokenBucket(rps=1.0, burst=5)
        start = time.monotonic()
        for _ in range(5):
            bucket.acquire()
        assert time.monotonic() - start < 0.1, "Ein voller Bucket darf sofort bursten"

    def test_enforces_rate_once_burst_is_spent(self):
        # 20 Token/s, Burst 2: der dritte Token kostet rund 50 ms.
        bucket = TokenBucket(rps=20.0, burst=2)
        bucket.acquire()
        bucket.acquire()
        start = time.monotonic()
        bucket.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.04, f"Rate nicht eingehalten, nur {elapsed:.3f} s gewartet"

    def test_refills_over_time(self):
        bucket = TokenBucket(rps=100.0, burst=2)
        bucket.acquire()
        bucket.acquire()
        assert not bucket.try_acquire(), "Bucket muss jetzt leer sein"
        time.sleep(0.05)
        assert bucket.try_acquire(), "Nach 50 ms muessen wieder Token da sein"

    def test_refill_is_capped_at_burst(self):
        bucket = TokenBucket(rps=1000.0, burst=3)
        time.sleep(0.02)  # wuerde rechnerisch 20 Token erzeugen
        assert bucket.try_acquire(3.0)
        assert not bucket.try_acquire(), "Kapazitaet darf burst nicht ueberschreiten"

    def test_try_acquire_never_blocks(self):
        bucket = TokenBucket(rps=0.001, burst=1)
        assert bucket.try_acquire() is True
        start = time.monotonic()
        assert bucket.try_acquire() is False
        assert time.monotonic() - start < 0.05

    def test_acquire_returns_waited_seconds(self):
        bucket = TokenBucket(rps=50.0, burst=1)
        bucket.acquire()
        waited = bucket.acquire()
        assert waited > 0.0

    def test_timeout_raises_instead_of_waiting_forever(self):
        bucket = TokenBucket(rps=0.1, burst=1)
        bucket.acquire()
        with pytest.raises(ProviderUnavailableError):
            bucket.acquire(timeout_s=0.05)

    def test_request_larger_than_capacity_is_rejected(self):
        bucket = TokenBucket(rps=1.0, burst=2)
        with pytest.raises(ValueError, match="Bucket fasst nur"):
            bucket.acquire(5.0)

    @pytest.mark.parametrize(("rps", "burst"), [(0.0, 1), (-1.0, 1), (1.0, 0)])
    def test_invalid_parameters_are_rejected(self, rps, burst):
        with pytest.raises(ValueError, match="muss"):
            TokenBucket(rps=rps, burst=burst)

    def test_is_thread_safe(self):
        # 8 Threads holen je 5 Token aus einem Bucket mit 40 Token Kapazitaet.
        # Kein Thread darf blockieren, und zusammen duerfen es nie mehr als
        # 40 sein - ohne Lock wuerde der Zaehler hier auseinanderlaufen.
        bucket = TokenBucket(rps=1.0, burst=40)
        acquired: list[bool] = []
        lock = threading.Lock()

        def worker() -> None:
            for _ in range(5):
                ok = bucket.try_acquire()
                with lock:
                    acquired.append(ok)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sum(acquired) == 40, "Genau die Kapazitaet darf entnommen werden"

    def test_context_manager_takes_one_token(self):
        bucket = TokenBucket(rps=1.0, burst=1)
        with bucket:
            pass
        assert not bucket.try_acquire()


class TestBackoffDelay:
    def test_grows_exponentially(self):
        cfg = RateLimitSettings(backoff_base_s=1.0, backoff_max_s=100.0, backoff_jitter=0.0)
        delays = [backoff_delay_s(i, cfg) for i in range(4)]
        assert delays == [1.0, 2.0, 4.0, 8.0]

    def test_is_capped(self):
        cfg = RateLimitSettings(backoff_base_s=1.0, backoff_max_s=5.0, backoff_jitter=0.0)
        assert backoff_delay_s(10, cfg) == 5.0

    def test_jitter_stays_within_band(self):
        cfg = RateLimitSettings(backoff_base_s=10.0, backoff_max_s=100.0, backoff_jitter=0.25)
        rng = random.Random(1234)
        values = [backoff_delay_s(0, cfg, rng=rng) for _ in range(200)]
        assert all(7.5 <= v <= 12.5 for v in values), "Jitter verlaesst das Band"
        assert len(set(values)) > 1, "Jitter muss tatsaechlich streuen"

    def test_is_never_negative(self):
        cfg = RateLimitSettings(backoff_base_s=1.0, backoff_max_s=10.0, backoff_jitter=1.0)
        rng = random.Random(7)
        assert all(backoff_delay_s(0, cfg, rng=rng) >= 0.0 for _ in range(100))


class TestRetryWithBackoff:
    def test_returns_first_success_without_sleeping(self):
        cfg = RateLimitSettings(max_retries=3)
        slept: list[float] = []

        @retry_with_backoff(settings=cfg, sleep=slept.append)
        def ok() -> str:
            return "fertig"

        assert ok() == "fertig"
        assert slept == []

    def test_retries_transient_errors_then_succeeds(self):
        cfg = RateLimitSettings(max_retries=3, backoff_base_s=0.01, backoff_jitter=0.0)
        slept: list[float] = []
        calls = {"n": 0}

        @retry_with_backoff(settings=cfg, sleep=slept.append)
        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise ProviderUnavailableError("kurz weg")
            return "fertig"

        assert flaky() == "fertig"
        assert calls["n"] == 3
        assert len(slept) == 2, "Zwei Fehlschlaege, zwei Wartezeiten"

    def test_gives_up_after_max_retries_and_reraises(self):
        cfg = RateLimitSettings(max_retries=2, backoff_base_s=0.01, backoff_jitter=0.0)
        slept: list[float] = []
        calls = {"n": 0}

        @retry_with_backoff(settings=cfg, sleep=slept.append)
        def broken() -> None:
            calls["n"] += 1
            raise ProviderUnavailableError("dauerhaft weg")

        with pytest.raises(ProviderUnavailableError, match="dauerhaft weg"):
            broken()
        assert calls["n"] == 3, "Erstversuch plus zwei Retries"
        assert len(slept) == 2

    def test_rate_limit_error_is_never_retried(self):
        # Harte Projektregel: Rate Limits werden nicht umgangen, auch nicht
        # durch Aussitzen. Der Fehler geht sofort nach oben.
        cfg = RateLimitSettings(max_retries=5, backoff_base_s=0.01)
        slept: list[float] = []
        calls = {"n": 0}

        @retry_with_backoff(settings=cfg, sleep=slept.append)
        def throttled() -> None:
            calls["n"] += 1
            raise RateLimitError("gedrosselt", retry_after_s=30.0)

        with pytest.raises(RateLimitError) as excinfo:
            throttled()
        assert calls["n"] == 1, "Kein einziger Retry bei 429"
        assert slept == []
        assert excinfo.value.retry_after_s == 30.0, "retry_after_s muss erhalten bleiben"

    def test_unlisted_exceptions_pass_through_untouched(self):
        cfg = RateLimitSettings(max_retries=3, backoff_base_s=0.01)
        calls = {"n": 0}

        @retry_with_backoff(settings=cfg, sleep=lambda _: None)
        def wrong_symbol() -> None:
            calls["n"] += 1
            raise SymbolNotFoundError("kein solches Symbol")

        with pytest.raises(SymbolNotFoundError):
            wrong_symbol()
        assert calls["n"] == 1, "Ein fachlicher Fehler ist nicht transient"

    def test_zero_retries_means_single_attempt(self):
        cfg = RateLimitSettings(max_retries=0)
        calls = {"n": 0}

        @retry_with_backoff(settings=cfg, sleep=lambda _: None)
        def once() -> None:
            calls["n"] += 1
            raise ProviderUnavailableError("weg")

        with pytest.raises(ProviderUnavailableError):
            once()
        assert calls["n"] == 1

    def test_preserves_function_metadata(self):
        @retry_with_backoff(settings=RateLimitSettings())
        def documented() -> None:
            """Ein Docstring."""

        assert documented.__name__ == "documented"
        assert documented.__doc__ == "Ein Docstring."

    def test_custom_retry_on_types(self):
        cfg = RateLimitSettings(max_retries=2, backoff_base_s=0.01, backoff_jitter=0.0)
        calls = {"n": 0}

        @retry_with_backoff(settings=cfg, retry_on=(TimeoutError,), sleep=lambda _: None)
        def slow() -> str:
            calls["n"] += 1
            if calls["n"] < 2:
                raise TimeoutError
            return "fertig"

        assert slow() == "fertig"
        assert calls["n"] == 2


class TestRegistry:
    def test_provides_one_bucket_per_source(self):
        registry = RateLimiterRegistry(RateLimitSettings())
        assert registry.bucket("yfinance") is registry.bucket("yfinance")
        assert registry.bucket("yfinance") is not registry.bucket("finviz")

    def test_buckets_use_configured_rates(self):
        cfg = RateLimitSettings(yfinance_rps=2.0, yfinance_burst=5, finviz_rps=0.5, finviz_burst=2)
        registry = RateLimiterRegistry(cfg)
        assert registry.bucket("yfinance").rps == 2.0
        assert registry.bucket("yfinance").burst == 5
        assert registry.bucket("finviz").rps == 0.5, "Finviz bewusst niedriger, Scraping"
        assert registry.bucket("finviz").burst == 2

    def test_unknown_source_raises(self):
        registry = RateLimiterRegistry(RateLimitSettings())
        with pytest.raises(KeyError):
            registry.bucket("bloomberg")

    def test_process_registry_is_shared(self):
        assert get_registry() is get_registry()
