"""Tests der Kern-Infrastruktur: Zeit und Konfiguration."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from btlite.config import Settings, get_settings
from btlite.core.clock import ensure_utc, index_to_utc, to_display_tz, utc_now


class TestClock:
    def test_utc_now_is_aware(self):
        assert utc_now().tzinfo is not None

    def test_naive_is_treated_as_utc(self):
        naive = datetime(2025, 3, 1, 12, 0)
        assert ensure_utc(naive).tzinfo is UTC
        assert ensure_utc(naive).hour == 12

    def test_aware_is_converted_not_relabelled(self):
        berlin = datetime(2025, 3, 1, 13, 0, tzinfo=UTC).astimezone()
        assert ensure_utc(berlin) == berlin

    def test_display_conversion_only_changes_representation(self):
        moment = datetime(2025, 7, 1, 12, 0, tzinfo=UTC)
        shown = to_display_tz(moment, "Europe/Berlin")
        assert shown.hour == 14, "Sommerzeit: UTC+2"
        assert shown == moment

    def test_index_to_utc_localizes_naive(self):
        df = pd.DataFrame({"x": [1, 2]}, index=pd.date_range("2024-01-01", periods=2))
        out = index_to_utc(df)
        assert str(out.index.tz) == "UTC"
        assert out.index.name == "date"

    def test_index_to_utc_converts_other_tz(self):
        idx = pd.date_range("2024-01-01", periods=2, tz="Europe/Berlin")
        out = index_to_utc(pd.DataFrame({"x": [1, 2]}, index=idx))
        assert str(out.index.tz) == "UTC"

    def test_index_to_utc_does_not_mutate_input(self):
        df = pd.DataFrame({"x": [1, 2]}, index=pd.date_range("2024-01-01", periods=2))
        index_to_utc(df)
        assert df.index.tz is None, "Original muss unveraendert bleiben"

    def test_index_to_utc_rejects_non_datetime(self):
        df = pd.DataFrame({"x": [1, 2]}, index=["a", "b"])
        with pytest.raises(TypeError):
            index_to_utc(df)


class TestSettings:
    def test_defaults_are_local_only(self):
        settings = Settings()
        assert settings.ui.host == "127.0.0.1", "Bewusst kein Bind auf 0.0.0.0"

    def test_get_settings_is_cached(self):
        assert get_settings() is get_settings()

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("BTLITE_UI_PORT", "8099")
        assert Settings().ui.port == 8099

    def test_log_level_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("BTLITE_LOG_LEVEL", "debug")
        assert Settings().log_level == "DEBUG"

    def test_invalid_port_rejected(self, monkeypatch):
        from pydantic import ValidationError

        monkeypatch.setenv("BTLITE_UI_PORT", "80")
        with pytest.raises(ValidationError):
            Settings()

    def test_delay_notice_is_configured(self):
        assert Settings().data.quote_delay_notice_min == 15

    def test_finviz_throttling_is_conservative(self):
        settings = Settings()
        assert settings.ratelimit.finviz_rps <= 1.0, "Scraping muss langsam bleiben"
