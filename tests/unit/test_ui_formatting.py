"""Tests der Zahlen- und Waehrungsdarstellung.

Akzeptanzkriterium des Projekts: ein fehlender Wert erscheint als "n/a",
nie als 0 und nie leer.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from btlite.models.common import Currency, DataSource, Provenance
from btlite.ui import formatting as fmt


class TestMissingValues:
    @pytest.mark.parametrize(
        "func",
        [
            fmt.format_number,
            fmt.format_percent,
            fmt.format_compact,
            fmt.format_integer,
        ],
    )
    def test_none_is_na(self, func):
        assert func(None) == fmt.NA_TEXT

    @pytest.mark.parametrize(
        "func",
        [fmt.format_number, fmt.format_percent, fmt.format_compact, fmt.format_integer],
    )
    def test_nan_is_na(self, func):
        assert func(float("nan")) == fmt.NA_TEXT

    def test_currency_none_is_na(self):
        assert fmt.format_currency(None, Currency.USD) == fmt.NA_TEXT

    def test_zero_is_not_na(self):
        assert fmt.format_number(0.0) != fmt.NA_TEXT
        assert fmt.format_percent(0.0) != fmt.NA_TEXT

    def test_is_missing(self):
        assert fmt.is_missing(None)
        assert fmt.is_missing(float("nan"))
        assert not fmt.is_missing(0.0)
        assert not fmt.is_missing(-1.5)


class TestFormatNumber:
    def test_german_separators(self):
        assert fmt.format_number(1234567.891) == "1,234,567.89"

    def test_decimals(self):
        assert fmt.format_number(1.23456, decimals=3) == "1.235"

    def test_without_thousands(self):
        assert fmt.format_number(1234.5, thousands=False) == "1234.50"

    def test_signed(self):
        assert fmt.format_number(1.5, signed=True) == "+1.50"
        assert fmt.format_number(-1.5, signed=True) == "-1.50"

    def test_negative(self):
        assert fmt.format_number(-1234.5) == "-1,234.50"


class TestFormatPercent:
    def test_decimal_to_percent(self):
        assert fmt.format_percent(0.31) == "31.0 %"

    def test_small_value(self):
        assert fmt.format_percent(0.0023, decimals=2) == "0.23 %"

    def test_negative_drawdown(self):
        assert fmt.format_percent(-0.2) == "-20.0 %"

    def test_signed(self):
        assert fmt.format_percent(0.015, signed=True) == "+1.5 %"

    def test_above_one(self):
        assert fmt.format_percent(1.52) == "152.0 %"


class TestFormatCompact:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1.2e12, "1.20 T"),
            (1.2e9, "1.20 B"),
            (1.2e6, "1.20 M"),
            (1.2e3, "1.20 K"),
            (999.0, "999.00"),
        ],
    )
    def test_magnitudes(self, value, expected):
        assert fmt.format_compact(value) == expected

    def test_negative(self):
        assert fmt.format_compact(-1.2e9) == "-1.20 B"

    def test_decimals(self):
        assert fmt.format_compact(1.234e9, decimals=1) == "1.2 B"


class TestFormatCurrency:
    def test_usd_symbol(self):
        assert fmt.format_currency(1234.5, Currency.USD) == "$ 1,234.50"

    def test_eur_symbol(self):
        """Euro hat ein Zeichen wie der Dollar - es ist die Standardwaehrung."""
        assert fmt.format_currency(1234.5, Currency.EUR) == "€ 1,234.50"

    def test_currency_without_symbol_falls_back_to_the_code(self):
        """Nicht jede Waehrung hat ein gelaeufiges Zeichen."""
        assert fmt.format_currency(1234.5, Currency.CHF) == "CHF 1,234.50"

    def test_compact(self):
        assert fmt.format_currency(4.79e12, Currency.USD, compact=True) == "$ 4.79 T"

    def test_without_currency(self):
        assert fmt.format_currency(10.0) == "10.00"

    def test_currency_symbol_other(self):
        assert fmt.currency_symbol(Currency.OTHER) == "OTHER"

    def test_currency_symbol_none(self):
        assert fmt.currency_symbol(None) == ""


class TestFormatRatioValue:
    def test_percent_unit(self):
        assert fmt.format_ratio_value(0.31, "percent") == "31.0 %"

    def test_days_unit(self):
        assert fmt.format_ratio_value(63.98, "days") == "64 days"

    def test_multiple_unit(self):
        assert fmt.format_ratio_value(40.21, "x") == "40.21x"

    def test_currency_unit(self):
        assert fmt.format_ratio_value(1.2e9, "currency") == "1.20 B"

    def test_ratio_unit(self):
        assert fmt.format_ratio_value(0.89, "ratio") == "0.89"

    def test_none_any_unit(self):
        for unit in ("percent", "days", "x", "currency", "ratio"):
            assert fmt.format_ratio_value(None, unit) == fmt.NA_TEXT


class TestFormatInteger:
    def test_thousands(self):
        assert fmt.format_integer(164_000) == "164,000"

    def test_float_truncates(self):
        assert fmt.format_integer(1234.9) == "1,234"

    def test_zero(self):
        assert fmt.format_integer(0) == "0"


class TestFormatTimestamp:
    def test_converts_to_display_timezone(self):
        ts = datetime(2026, 9, 4, 11, 46, tzinfo=UTC)
        text = fmt.format_timestamp(ts, "Europe/Berlin")
        assert text.startswith("2026-09-04 13:46")

    def test_none(self):
        assert fmt.format_timestamp(None, "Europe/Berlin") == fmt.NA_TEXT


class TestFormatDelay:
    def test_minutes(self):
        assert fmt.format_delay(15) == "15 min"

    def test_hours(self):
        assert fmt.format_delay(180) == "3.0 h"

    def test_days(self):
        assert fmt.format_delay(60 * 24 * 16) == "16.0 days"

    def test_unknown(self):
        assert fmt.format_delay(None) == "unknown"

    def test_zero(self):
        assert fmt.format_delay(0) == "0 min"


class TestProvenanceText:
    @pytest.fixture
    def fixture_provenance(self):
        return Provenance(
            source=DataSource.FIXTURE,
            fetched_at=datetime(2026, 9, 20, 10, 0, tzinfo=UTC),
            from_cache=True,
            delay_minutes=22815,
            note="As of 2026-09-04",
        )

    def test_names_source_time_and_delay(self, fixture_provenance):
        text = fmt.provenance_text(fixture_provenance, "Europe/Berlin")
        assert "Fixture" in text
        assert "2026-09-20" in text
        assert "Delay" in text
        assert "As of 2026-09-04" in text

    def test_marks_cache(self, fixture_provenance):
        assert "cache" in fmt.provenance_text(fixture_provenance, "UTC")

    def test_none(self):
        assert fmt.provenance_text(None, "UTC") == "Source: unknown"

    def test_source_label(self, fixture_provenance):
        assert fmt.source_label(fixture_provenance) == "Fixture (frozen)"

    def test_does_not_claim_realtime(self, fixture_provenance):
        text = fmt.provenance_text(fixture_provenance, "Europe/Berlin").lower()
        assert "echtzeit" not in text
        assert "realtime" not in text


class TestDirectionAndValueClass:
    def test_up(self):
        assert fmt.direction_class(0.5) == "bt-up"

    def test_down(self):
        assert fmt.direction_class(-0.5) == "bt-down"

    def test_zero(self):
        assert fmt.direction_class(0.0) == ""

    def test_missing(self):
        assert fmt.direction_class(None) == "bt-na"
        assert fmt.direction_class(math.nan) == "bt-na"

    def test_value_class(self):
        assert fmt.value_class(None) == "bt-na"
        assert fmt.value_class(1.0) == ""
