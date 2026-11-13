"""Tests des zweistufigen Cache: DataFrame-Ebene und HTTP-Session."""

from __future__ import annotations

import time

import pandas as pd
import pytest

from btlite.config import CacheSettings
from btlite.core.cache import DataFrameCache, cache_key, make_cached_session


@pytest.fixture
def settings(tmp_path):
    """Cache-Settings mit temporaerem Verzeichnis."""
    return CacheSettings(dir=tmp_path / "cache", enabled=True)


@pytest.fixture
def cache(settings):
    """Aktiver DataFrame-Cache auf temporaerem Verzeichnis."""
    df_cache = DataFrameCache(settings)
    yield df_cache
    df_cache.close()


@pytest.fixture
def frame():
    """Kleiner Frame mit tz-aware UTC-Index."""
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC", name="date")
    return pd.DataFrame({"close": [1.0, 2.0, 3.0], "volume": [10.0, 20.0, 30.0]}, index=idx)


class TestCacheKey:
    def test_joins_parts_readably(self):
        assert cache_key("yf", "history", "AAPL", "1d") == "yf|history|AAPL|1d"

    def test_is_stable_for_same_input(self):
        assert cache_key("a", 1, None) == cache_key("a", 1, None)

    def test_differs_for_different_input(self):
        assert cache_key("yf", "AAPL") != cache_key("yf", "MSFT")

    def test_long_keys_are_hashed_but_stay_bounded(self):
        key = cache_key("yf", "x" * 500)
        assert len(key) <= 200
        assert "sha256:" in key

    def test_long_keys_stay_distinct(self):
        assert cache_key("a" * 300) != cache_key("b" * 300)


class TestDataFrameCacheRoundtrip:
    def test_miss_returns_none(self, cache):
        assert cache.get_df("gibt-es-nicht") is None

    def test_set_then_get_roundtrips(self, cache, frame):
        assert cache.set_df("k", frame, 60) is True
        out = cache.get_df("k")
        assert out is not None
        # check_freq=False: parquet speichert das freq-Attribut des Index
        # nicht mit. Das ist Metadatum, kein Datum - Werte, Dtypes und
        # Zeitzone kommen unveraendert zurueck.
        pd.testing.assert_frame_equal(out, frame, check_freq=False)

    def test_preserves_tz_aware_index(self, cache, frame):
        cache.set_df("k", frame, 60)
        out = cache.get_df("k")
        assert str(out.index.tz) == "UTC", "parquet muss die Zeitzone halten"
        assert out.index.name == "date"

    def test_preserves_dtypes(self, cache, frame):
        cache.set_df("k", frame, 60)
        out = cache.get_df("k")
        assert out["close"].dtype == frame["close"].dtype

    def test_delete_removes_entry(self, cache, frame):
        cache.set_df("k", frame, 60)
        assert cache.delete("k") is True
        assert cache.get_df("k") is None

    def test_clear_empties_cache(self, cache, frame):
        cache.set_df("a", frame, 60)
        cache.set_df("b", frame, 60)
        cache.clear()
        assert cache.get_df("a") is None
        assert cache.get_df("b") is None

    def test_entries_survive_reopen(self, settings, frame):
        first = DataFrameCache(settings)
        first.set_df("k", frame, 60)
        first.close()

        second = DataFrameCache(settings)
        assert second.get_df("k") is not None, "Cache muss persistent sein"
        second.close()

    def test_context_manager_closes(self, settings, frame):
        with DataFrameCache(settings) as c:
            c.set_df("k", frame, 60)
        assert DataFrameCache(settings).get_df("k") is not None


class TestTtl:
    def test_entry_expires_after_ttl(self, cache, frame):
        cache.set_df("k", frame, 0.15)
        assert cache.get_df("k") is not None, "Direkt nach dem Schreiben ein Hit"
        time.sleep(0.3)
        assert cache.get_df("k") is None, "Nach Ablauf der TTL ein Miss"

    def test_ttl_none_does_not_expire(self, cache, frame):
        cache.set_df("k", frame, None)
        time.sleep(0.1)
        assert cache.get_df("k") is not None


class TestCachedDf:
    def test_miss_calls_loader_and_stores(self, cache, frame):
        calls = {"n": 0}

        def loader() -> pd.DataFrame:
            calls["n"] += 1
            return frame

        out, from_cache = cache.cached_df("k", 60, loader)
        assert calls["n"] == 1
        assert from_cache is False, "Erster Aufruf ist ein Miss"
        pd.testing.assert_frame_equal(out, frame)

    def test_hit_skips_loader(self, cache, frame):
        calls = {"n": 0}

        def loader() -> pd.DataFrame:
            calls["n"] += 1
            return frame

        cache.cached_df("k", 60, loader)
        out, from_cache = cache.cached_df("k", 60, loader)
        assert calls["n"] == 1, "Der Loader darf beim Hit nicht laufen"
        assert from_cache is True
        pd.testing.assert_frame_equal(out, frame, check_freq=False)

    def test_from_cache_flag_feeds_provenance(self, cache, frame):
        _, first = cache.cached_df("k", 60, lambda: frame)
        _, second = cache.cached_df("k", 60, lambda: frame)
        assert (first, second) == (False, True)

    def test_different_keys_are_independent(self, cache, frame):
        calls = {"n": 0}

        def loader() -> pd.DataFrame:
            calls["n"] += 1
            return frame

        cache.cached_df("a", 60, loader)
        cache.cached_df("b", 60, loader)
        assert calls["n"] == 2


class TestDisabledCache:
    @pytest.fixture
    def disabled(self, tmp_path):
        return DataFrameCache(CacheSettings(dir=tmp_path / "cache", enabled=False))

    def test_get_is_always_a_miss(self, disabled, frame):
        disabled.set_df("k", frame, 60)
        assert disabled.get_df("k") is None

    def test_set_reports_no_write(self, disabled, frame):
        assert disabled.set_df("k", frame, 60) is False

    def test_cached_df_always_calls_loader(self, disabled, frame):
        calls = {"n": 0}

        def loader() -> pd.DataFrame:
            calls["n"] += 1
            return frame

        for _ in range(3):
            out, from_cache = disabled.cached_df("k", 60, loader)
            assert from_cache is False
            pd.testing.assert_frame_equal(out, frame)
        assert calls["n"] == 3, "Ohne Cache laedt jeder Aufruf neu"

    def test_creates_no_directory(self, tmp_path):
        cfg = CacheSettings(dir=tmp_path / "cache", enabled=False)
        cache = DataFrameCache(cfg)
        cache.get_df("k")
        cache.set_df("k", pd.DataFrame({"a": [1]}), 60)
        assert not (tmp_path / "cache").exists(), "Ein toter Cache legt nichts an"

    def test_enabled_flag_is_exposed(self, disabled, cache):
        assert disabled.enabled is False
        assert cache.enabled is True


class TestHttpSession:
    def test_returns_plain_session_when_disabled(self, tmp_path):
        cfg = CacheSettings(dir=tmp_path / "c", enabled=False)
        session = make_cached_session(cfg)
        assert not hasattr(session, "settings"), "Ohne Cache eine gewoehnliche Session"

    def test_offline_without_cache_is_rejected(self, tmp_path):
        # Offline ohne Cache koennte nur scheitern - das faellt sofort auf,
        # statt spaeter als leeres Ergebnis zu erscheinen.
        cfg = CacheSettings(dir=tmp_path / "c", enabled=False)
        with pytest.raises(RuntimeError, match="Offline mode requires an active HTTP cache"):
            make_cached_session(cfg, offline=True)

    def test_cached_session_is_created(self, settings):
        session = make_cached_session(settings)
        assert hasattr(session, "settings"), "requests-cache-Session erwartet"
        session.close()

    def test_offline_session_only_serves_from_cache(self, settings):
        session = make_cached_session(settings, offline=True)
        assert session.settings.only_if_cached is True, "Offline darf nie ins Netz"
        session.close()

    def test_expire_after_is_configurable(self, settings):
        session = make_cached_session(settings, expire_after_s=42)
        assert session.settings.expire_after == 42
        session.close()
