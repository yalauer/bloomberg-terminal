"""Tests des Wechselkurs-Providers.

Kein Test hier geht ins Netz - die Socket-Sperre der Suite verbietet es.
``yfinance.Ticker`` wird durch eine Attrappe ersetzt, die genau die Form
liefert, die der Live-Abruf tatsaechlich zurueckgibt: Spalten in
Grossschreibung und ein Index in ``Europe/London``, nicht in UTC. Beides
wurde am 23.09.2026 gegen die echte Quelle verifiziert.

Der wichtigste Test dieser Datei ist die Richtungsprobe: eine vertauschte
Division faellt sonst niemandem auf, weil 1.14 und 0.877 beide plausibel
aussehen.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from btlite.config import Settings
from btlite.core.cache import DataFrameCache
from btlite.core.errors import ProviderUnavailableError, RateLimitError, SymbolNotFoundError
from btlite.core.ratelimit import TokenBucket
from btlite.data.base import ExchangeRateProvider
from btlite.data.rates import (
    FIXTURE_SUBDIR,
    FX_SOURCE_TZ,
    PIVOT_CURRENCY,
    YFinanceRateProvider,
    fx_fixture_slug,
    fx_symbol,
    fx_symbol_from_slug,
    normalize_fx_history,
    rate_from_history,
)
from btlite.models.common import Currency

# --------------------------------------------------------------- Helfer --

#: Der am 23.09.2026 gegen die Quelle geprueftre Kurs: USD je EUR.
EURUSD_23_09 = 1.1405


def _online_settings(tmp_path: Any) -> Settings:
    """Settings mit abgeschaltetem Offline-Modus und temporaerem Cache.

    ``DataSettings.offline`` ist per Default True. Diese Tests pruefen das
    Verhalten diesseits dieser Sperre; einen Request machen sie trotzdem
    nicht, die Quelle ist eine Attrappe.
    """
    settings = Settings()
    settings.data.offline = False
    settings.cache.dir = tmp_path / "cache"
    return settings


def _raw_fx(rows: dict[str, float]) -> pd.DataFrame:
    """Rohframe, wie ``Ticker.history`` ihn fuer ein ``=X``-Symbol liefert.

    Spalten in Grossschreibung, Index in ``Europe/London`` - nicht UTC. Genau
    diese Form hat die echte Quelle.
    """
    idx = pd.DatetimeIndex(list(rows), name="Date").tz_localize(FX_SOURCE_TZ)
    values = list(rows.values())
    return pd.DataFrame(
        {
            "Open": values,
            "High": values,
            "Low": values,
            "Close": values,
            "Volume": [0] * len(values),
        },
        index=idx,
    )


class _FakeTicker:
    """Attrappe von ``yfinance.Ticker``. Zaehlt die Abrufe je Symbol."""

    def __init__(self, frames: dict[str, pd.DataFrame], calls: list[str]) -> None:
        self._frames = frames
        self._calls = calls
        self._symbol = ""

    def __call__(self, symbol: str) -> _FakeTicker:
        self._symbol = symbol
        return self

    def history(self, **_kwargs: Any) -> pd.DataFrame:
        self._calls.append(self._symbol)
        frame = self._frames.get(self._symbol)
        if frame is None:
            from yfinance.exceptions import YFPricesMissingError

            raise YFPricesMissingError(self._symbol, "no price data found")
        return frame


def _patch_yf(
    monkeypatch: pytest.MonkeyPatch,
    frames: dict[str, pd.DataFrame],
) -> list[str]:
    """Haengt die Ticker-Attrappe in yfinance und liefert das Aufrufprotokoll."""
    import yfinance as yf

    calls: list[str] = []
    monkeypatch.setattr(yf, "Ticker", _FakeTicker(frames, calls))
    return calls


#: Zeilen der Offline-Fixture in den Tests. Fr, Mo, Di, Mi - das Wochenende
#: dazwischen fehlt, genau wie bei der echten Quelle.
_FIXTURE_ROWS: dict[str, float] = {
    "2026-09-18": 1.1350,
    "2026-09-21": 1.1380,
    "2026-09-22": 1.1390,
    "2026-09-23": EURUSD_23_09,
}


def _write_rate_fixtures(root: Any, fixtures: dict[str, dict[str, float]]) -> Any:
    """Schreibt FX-Fixtures im Ablageformat von ``refresh_fixtures.py``.

    Durchlaeuft bewusst dieselbe Normalisierung wie das Skript, damit der Test
    gegen die echte Ablageform prueft und nicht gegen eine eigene.

    Args:
        root: Verzeichnis, unter dem ``rates/`` angelegt wird.
        fixtures: Slug (etwa ``"EURUSD"``) auf Datum-zu-Kurs-Abbildung. Die
            Kurse sind Einheiten der Ziel- je einer Einheit der
            Ausgangswaehrung, dezimal.

    Returns:
        Das Fixture-Wurzelverzeichnis.
    """
    target = root / "rates"
    target.mkdir(parents=True, exist_ok=True)
    for slug, rows in fixtures.items():
        normalize_fx_history(_raw_fx(rows)).to_parquet(target / f"{slug}.parquet", engine="pyarrow")
    return root


def _offline_provider(
    tmp_path: Any, *, fixtures: dict[str, dict[str, float]] | None
) -> YFinanceRateProvider:
    """Provider im Offline-Modus gegen ein eigenes Fixture-Verzeichnis.

    Args:
        tmp_path: Temporaeres Verzeichnis des Tests.
        fixtures: Anzulegende Historien, oder ``None`` fuer ein leeres
            Verzeichnis.

    Returns:
        Provider, dessen ``fixtures_dir`` auf dieses Verzeichnis zeigt.
    """
    root = tmp_path / "fixtures"
    root.mkdir(parents=True, exist_ok=True)
    if fixtures:
        _write_rate_fixtures(root, fixtures)
    settings = Settings()
    settings.data.offline = True
    settings.data.fixtures_dir = root
    settings.cache.dir = tmp_path / "cache"
    return YFinanceRateProvider(settings, cache=DataFrameCache(settings.cache))


def _shipped_offline_settings() -> Settings:
    """Offline-Settings gegen die ausgelieferten Fixtures des Repositories."""
    settings = Settings()
    settings.data.offline = True
    return settings


@pytest.fixture
def provider(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Provider mit EURUSD-Attrappe. Liefert (provider, calls)."""
    frames = {
        "EURUSD=X": _raw_fx(
            {
                # Fr, Mo, Di, Mi - das Wochenende dazwischen fehlt, genau wie
                # bei der echten Quelle.
                "2026-09-18": 1.1350,
                "2026-09-21": 1.1380,
                "2026-09-22": 1.1390,
                "2026-09-23": EURUSD_23_09,
            }
        )
    }
    calls = _patch_yf(monkeypatch, frames)
    settings = _online_settings(tmp_path)
    return YFinanceRateProvider(settings, cache=DataFrameCache(settings.cache)), calls


# ------------------------------------------------------------- Vertrag --


class TestProtocol:
    def test_satisfies_exchange_rate_provider(self, provider):
        prov, _ = provider
        assert isinstance(prov, ExchangeRateProvider)

    def test_symbol_follows_source_scheme(self):
        assert fx_symbol(Currency.EUR, Currency.USD) == "EURUSD=X"


class TestSameCurrency:
    def test_returns_exactly_one(self, provider):
        prov, _ = provider
        assert prov.get_rate(Currency.EUR, Currency.EUR) == 1.0

    def test_does_not_fetch(self, provider):
        prov, calls = provider
        prov.get_rate(Currency.USD, Currency.USD)
        assert calls == [], "gleiche Waehrung darf keinen Abruf ausloesen"

    def test_works_offline(self, tmp_path):
        # Offline ist Default. Die Identitaet braucht keine Quelle und darf
        # deshalb auch offline nicht scheitern.
        settings = Settings()
        settings.data.offline = True
        settings.cache.dir = tmp_path / "cache"
        prov = YFinanceRateProvider(settings, cache=DataFrameCache(settings.cache))
        assert prov.get_rate(Currency.CHF, Currency.CHF) == 1.0


# ----------------------------------------------------------- Richtung ---


class TestDirection:
    def test_eur_to_usd_multiplies(self, provider):
        # Die wichtigste Probe der Datei: EURUSD=X notiert USD je EUR.
        prov, _ = provider
        rate = prov.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 23))
        assert rate == pytest.approx(EURUSD_23_09)

    def test_hundred_eur_become_114_usd(self, provider):
        prov, _ = provider
        rate = prov.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 23))
        assert 100.0 * rate == pytest.approx(114.05, abs=0.01)

    def test_is_not_the_inverse(self, provider):
        # 87.7 waere das Ergebnis einer vertauschten Division. Beide Zahlen
        # sehen plausibel aus - nur eine ist richtig.
        prov, _ = provider
        rate = prov.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 23))
        assert 100.0 * rate != pytest.approx(87.7, abs=0.5)
        assert rate > 1.0

    def test_reverse_pair_uses_reverse_symbol(self, tmp_path, monkeypatch):
        # USD nach EUR fragt USDEUR=X ab, nicht den Kehrwert von EURUSD=X.
        frames = {"USDEUR=X": _raw_fx({"2026-09-23": 0.8768})}
        calls = _patch_yf(monkeypatch, frames)
        settings = _online_settings(tmp_path)
        prov = YFinanceRateProvider(settings, cache=DataFrameCache(settings.cache))
        rate = prov.get_rate(Currency.USD, Currency.EUR, date(2026, 9, 23))
        assert calls == ["USDEUR=X"]
        assert rate == pytest.approx(0.8768)


# ------------------------------------------------------------ Kalender --


class TestTradingDays:
    def test_plain_weekday(self, provider):
        prov, _ = provider
        assert prov.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 22)) == pytest.approx(1.1390)

    def test_saturday_uses_preceding_friday(self, provider):
        # Sa, 19.09.2026: keine Zeile. Es gilt Fr, der 18.
        prov, _ = provider
        assert prov.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 19)) == pytest.approx(1.1350)

    def test_sunday_uses_preceding_friday(self, provider):
        prov, _ = provider
        assert prov.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 20)) == pytest.approx(1.1350)

    def test_none_takes_latest(self, provider):
        prov, _ = provider
        assert prov.get_rate(Currency.EUR, Currency.USD) == pytest.approx(EURUSD_23_09)

    def test_date_before_history_raises(self, provider):
        # Kein Handelstag davor - es wird nicht der aelteste Kurs vorgetaeuscht.
        prov, _ = provider
        with pytest.raises(SymbolNotFoundError):
            prov.get_rate(Currency.EUR, Currency.USD, date(2020, 1, 2))


class TestSummerTimeOffset:
    def test_july_day_is_not_shifted_by_one(self, tmp_path, monkeypatch):
        # Im Sommer liegt ein Londoner Handelstag nach der UTC-Normalisierung
        # auf 23:00 UTC des Vortages. Ein Vergleich auf dem UTC-Datum lieferte
        # hier den Kurs des Vortages.
        frames = {
            "EURUSD=X": _raw_fx(
                {
                    "2026-07-14": 1.1000,
                    "2026-07-15": 1.2000,
                }
            )
        }
        _patch_yf(monkeypatch, frames)
        settings = _online_settings(tmp_path)
        prov = YFinanceRateProvider(settings, cache=DataFrameCache(settings.cache))
        assert prov.get_rate(Currency.EUR, Currency.USD, date(2026, 7, 15)) == pytest.approx(1.2)


class TestNormalizeFxHistory:
    def test_index_becomes_tz_aware_utc(self):
        out = normalize_fx_history(_raw_fx({"2026-09-23": 1.14}))
        assert str(out.index.tz) == "UTC"

    def test_source_is_london_not_utc(self):
        # Belegt die Ausgangslage: die Quelle liefert eben nicht UTC.
        raw = _raw_fx({"2026-01-15": 1.14})
        assert str(raw.index.tz) == FX_SOURCE_TZ

    def test_keeps_only_close(self):
        out = normalize_fx_history(_raw_fx({"2026-09-23": 1.14}))
        assert list(out.columns) == ["close"]

    def test_empty_input_stays_empty(self):
        assert normalize_fx_history(pd.DataFrame()).empty

    def test_sorts_ascending(self):
        raw = _raw_fx({"2026-09-23": 1.14, "2026-09-21": 1.13})
        out = normalize_fx_history(raw)
        assert out.index.is_monotonic_increasing


# -------------------------------------------------------- fehlende Kurse --


class TestMissingRate:
    def test_unknown_pair_raises_and_never_returns_one(self, tmp_path, monkeypatch):
        # Weder Direktkurs noch USD-Schenkel vorhanden.
        _patch_yf(monkeypatch, {})
        settings = _online_settings(tmp_path)
        prov = YFinanceRateProvider(settings, cache=DataFrameCache(settings.cache))
        with pytest.raises(SymbolNotFoundError) as excinfo:
            prov.get_rate(Currency.SEK, Currency.NOK, date(2026, 9, 23))
        assert "SEK" in str(excinfo.value)

    def test_empty_history_raises(self):
        with pytest.raises(SymbolNotFoundError):
            rate_from_history(normalize_fx_history(pd.DataFrame()), on=None, symbol="EURUSD=X")

    def test_nan_close_is_not_a_rate(self):
        raw = _raw_fx({"2026-09-23": float("nan")})
        with pytest.raises(SymbolNotFoundError):
            rate_from_history(normalize_fx_history(raw), on=None, symbol="EURUSD=X")

    def test_zero_close_is_not_a_rate(self):
        raw = _raw_fx({"2026-09-23": 0.0})
        with pytest.raises(SymbolNotFoundError):
            rate_from_history(normalize_fx_history(raw), on=None, symbol="EURUSD=X")

    def test_unknown_currency_raises_without_fetch(self, tmp_path, monkeypatch):
        calls = _patch_yf(monkeypatch, {})
        settings = _online_settings(tmp_path)
        prov = YFinanceRateProvider(settings, cache=DataFrameCache(settings.cache))
        with pytest.raises(SymbolNotFoundError):
            prov.get_rate(Currency.OTHER, Currency.USD)
        assert calls == [], "aus Currency.OTHER laesst sich kein Symbol bauen"

    def test_never_falls_back_to_one(self, tmp_path, monkeypatch):
        _patch_yf(monkeypatch, {})
        settings = _online_settings(tmp_path)
        prov = YFinanceRateProvider(settings, cache=DataFrameCache(settings.cache))
        for on in (None, date(2026, 9, 23)):
            with pytest.raises(SymbolNotFoundError):
                prov.get_rate(Currency.EUR, Currency.USD, on)


# ------------------------------------------------------------- Offline --


class TestOfflineMode:
    """Offline bedienen die eingefrorenen Historien unter ``rates/``.

    Der Default des Produkts ist ``BTLITE_DATA_OFFLINE=1``. Vor diesem
    Abschnitt warf der Provider dort fuer jedes Paar, womit im Depot bei
    jeder Fremdwaehrungsposition "n/a" stand.
    """

    def test_pair_without_fixture_still_raises(self, tmp_path, monkeypatch):
        # Kein Fixture, also kein Kurs. Es wird nicht geraten.
        calls = _patch_yf(monkeypatch, {"EURUSD=X": _raw_fx({"2026-09-23": EURUSD_23_09})})
        prov = _offline_provider(tmp_path, fixtures=None)
        with pytest.raises(ProviderUnavailableError) as excinfo:
            prov.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 23))
        assert "Offline mode" in excinfo.value.user_message
        assert calls == [], "offline wird die Quelle nie befragt"

    def test_does_not_touch_the_source(self, tmp_path, monkeypatch):
        # Auch mit vorhandenem Fixture geht offline kein Request raus. Die
        # Attrappe liefert bewusst einen absurden Kurs - er darf nicht
        # auftauchen.
        calls = _patch_yf(monkeypatch, {"EURUSD=X": _raw_fx({"2026-09-23": 99.0})})
        prov = _offline_provider(tmp_path, fixtures={"EURUSD": _FIXTURE_ROWS})
        rate = prov.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 23))
        assert calls == []
        assert rate == pytest.approx(EURUSD_23_09)

    def test_user_message_is_english(self, tmp_path, monkeypatch):
        _patch_yf(monkeypatch, {})
        prov = _offline_provider(tmp_path, fixtures=None)
        with pytest.raises(ProviderUnavailableError) as excinfo:
            prov.get_rate(Currency.EUR, Currency.USD)
        assert "BTLITE_DATA_OFFLINE=0" in excinfo.value.user_message

    def test_detail_is_german_and_names_the_path(self, tmp_path, monkeypatch):
        _patch_yf(monkeypatch, {})
        prov = _offline_provider(tmp_path, fixtures=None)
        with pytest.raises(ProviderUnavailableError) as excinfo:
            prov.get_rate(Currency.EUR, Currency.USD)
        assert "kein Fixture" in (excinfo.value.detail or "")

    def test_reads_the_rate_from_the_fixture(self, tmp_path, monkeypatch):
        _patch_yf(monkeypatch, {})
        prov = _offline_provider(tmp_path, fixtures={"EURUSD": _FIXTURE_ROWS})
        rate = prov.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 23))
        assert rate == pytest.approx(EURUSD_23_09)

    def test_offline_rate_is_not_one(self, tmp_path, monkeypatch):
        # Die Kernregel: nie ein Rueckfall auf 1.0.
        _patch_yf(monkeypatch, {})
        prov = _offline_provider(tmp_path, fixtures={"EURUSD": _FIXTURE_ROWS})
        assert prov.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 23)) != 1.0

    def test_same_currency_needs_no_fixture(self, tmp_path, monkeypatch):
        calls = _patch_yf(monkeypatch, {})
        prov = _offline_provider(tmp_path, fixtures=None)
        assert prov.get_rate(Currency.CHF, Currency.CHF) == 1.0
        assert calls == []

    def test_uses_the_purchase_day_not_the_latest(self, tmp_path, monkeypatch):
        # Der Einstand wird zum Kurs des Kauftages umgerechnet, nicht zum
        # heutigen. Die Fixture endet am 23.09. mit 1.1405.
        _patch_yf(monkeypatch, {})
        prov = _offline_provider(tmp_path, fixtures={"EURUSD": _FIXTURE_ROWS})
        rate = prov.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 21))
        assert rate == pytest.approx(1.1380)
        assert rate != pytest.approx(EURUSD_23_09)

    def test_weekend_falls_back_to_the_preceding_friday(self, tmp_path, monkeypatch):
        _patch_yf(monkeypatch, {})
        prov = _offline_provider(tmp_path, fixtures={"EURUSD": _FIXTURE_ROWS})
        assert prov.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 20)) == pytest.approx(1.1350)

    def test_date_before_the_fixture_raises(self, tmp_path, monkeypatch):
        # Kein Vorgaenger - der aelteste bekannte Kurs wird nicht vorgetaeuscht.
        _patch_yf(monkeypatch, {})
        prov = _offline_provider(tmp_path, fixtures={"EURUSD": _FIXTURE_ROWS})
        with pytest.raises(SymbolNotFoundError):
            prov.get_rate(Currency.EUR, Currency.USD, date(2019, 1, 2))

    def test_summer_day_is_not_shifted_by_one(self, tmp_path, monkeypatch):
        # Der DST-Fall, gegen **fest notierte** UTC-Stempel statt gegen die
        # eigene Normalisierung. Ginge die Fixture durch `_raw_fx` und
        # `normalize_fx_history`, hoebe sich ein Fehler in `_london_dates`
        # beim Schreiben und beim Lesen gegenseitig auf - der Test pruefte
        # dann gegen sich selbst und liesse eine naive UTC-Umrechnung durch.
        #
        # Die beiden Stempel sind die Londoner Handelstage 14. und 15. Juli
        # 2026 (BST, UTC+1), am 23.09.2026 gegen die Quelle verifiziert.
        target = tmp_path / "fixtures" / "rates"
        target.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(
            {"close": [1.1000, 1.2000]},
            index=pd.DatetimeIndex(
                ["2026-07-13T23:00:00+00:00", "2026-07-14T23:00:00+00:00"],
                name="date",
            ),
        )
        frame.to_parquet(target / "EURUSD.parquet", engine="pyarrow")

        _patch_yf(monkeypatch, {})
        settings = Settings()
        settings.data.offline = True
        settings.data.fixtures_dir = tmp_path / "fixtures"
        settings.cache.dir = tmp_path / "cache"
        prov = YFinanceRateProvider(settings, cache=DataFrameCache(settings.cache))

        # 15.07. ist Londoner Handelstag -> 1.2000. Eine naive UTC-Lesart
        # ordnete den Stempel dem 14.07. zu und lieferte 1.1000.
        assert prov.get_rate(Currency.EUR, Currency.USD, date(2026, 7, 15)) == pytest.approx(1.2)
        assert prov.get_rate(Currency.EUR, Currency.USD, date(2026, 7, 14)) == pytest.approx(1.1)

    def test_fixture_index_keeps_the_dst_offset(self, tmp_path):
        # Belegt die Ablageform: der Sommertag steht wirklich auf 23:00 UTC
        # des Vortages, der Wintertag auf 00:00 UTC. Ginge die Fixture naiv
        # als UTC auf die Platte, waere der Sommertag nicht rekonstruierbar.
        root = _write_rate_fixtures(tmp_path, {"EURUSD": {"2026-01-15": 1.16, "2026-07-15": 1.14}})
        stored = pd.read_parquet(root / "rates" / "EURUSD.parquet")
        assert str(stored.index.tz) == "UTC"
        assert list(stored.columns) == ["close"]
        stamps = {str(ts) for ts in stored.index}
        assert "2026-01-15 00:00:00+00:00" in stamps
        assert "2026-07-14 23:00:00+00:00" in stamps

    def test_triangulation_works_offline(self, tmp_path, monkeypatch):
        # HKD nach NOK gibt es direkt nicht, die USD-Schenkel schon.
        calls = _patch_yf(monkeypatch, {})
        prov = _offline_provider(
            tmp_path,
            fixtures={
                "HKDUSD": {"2026-09-23": 0.1275},
                "USDNOK": {"2026-09-23": 9.475},
            },
        )
        rate = prov.get_rate(Currency.HKD, Currency.NOK, date(2026, 9, 23))
        assert rate == pytest.approx(0.1275 * 9.475)
        assert calls == []

    def test_triangulation_with_one_missing_leg_raises(self, tmp_path, monkeypatch):
        _patch_yf(monkeypatch, {})
        prov = _offline_provider(tmp_path, fixtures={"HKDUSD": {"2026-09-23": 0.1275}})
        with pytest.raises((SymbolNotFoundError, ProviderUnavailableError)):
            prov.get_rate(Currency.HKD, Currency.NOK, date(2026, 9, 23))


class TestShippedFixtures:
    """Gegen die ausgelieferten Fixtures, nicht gegen selbst gebaute."""

    def test_eur_usd_is_available_offline(self):
        # Die App startet offline und muss ein EUR-Depot bewerten koennen.
        prov = YFinanceRateProvider(_shipped_offline_settings())
        rate = prov.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 23))
        assert 0.5 < rate < 2.0
        assert rate != 1.0

    def test_every_known_currency_converts_to_eur(self):
        # Jede Waehrung aus dem Vertrag muss offline in EUR umrechenbar sein,
        # direkt oder ueber USD. Sonst steht im Depot wieder "n/a".
        prov = YFinanceRateProvider(_shipped_offline_settings())
        for currency in Currency:
            if currency is Currency.OTHER:
                continue
            assert prov.get_rate(currency, Currency.EUR, date(2026, 9, 23)) > 0.0, currency

    def test_historical_purchase_day_is_covered(self):
        # Ein Kauftag vor Jahren muss bedient werden, sonst bringt die
        # Fixture fuer den Einstand nichts.
        prov = YFinanceRateProvider(_shipped_offline_settings())
        old = prov.get_rate(Currency.EUR, Currency.USD, date(2022, 6, 15))
        recent = prov.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 23))
        assert 0.5 < old < 2.0
        assert old != pytest.approx(recent), "ein Kauftag von 2022 ist nicht der heutige Kurs"

    def test_fixture_index_is_utc_with_close_column(self):
        path = Settings().data.fixtures_dir / FIXTURE_SUBDIR / "EURUSD.parquet"
        stored = pd.read_parquet(path)
        assert str(stored.index.tz) == "UTC"
        assert list(stored.columns) == ["close"]
        assert stored.index.is_monotonic_increasing

    def test_slug_roundtrips(self):
        for symbol in ("EURUSD=X", "USDJPY=X", "HKDUSD=X"):
            assert fx_symbol_from_slug(fx_fixture_slug(symbol)) == symbol


# --------------------------------------------------------------- Cache --


class TestCaching:
    def test_second_call_does_not_refetch(self, provider):
        prov, calls = provider
        prov.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 23))
        prov.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 23))
        assert calls == ["EURUSD=X"], "der zweite Aufruf muss aus dem Cache kommen"

    def test_ten_positions_same_day_are_one_fetch(self, provider):
        # Der Regelfall: ein Depot mit zehn Positionen, alle am selben Tag
        # gekauft. Das darf genau einen Abruf ausloesen.
        prov, calls = provider
        for _ in range(10):
            prov.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 22))
        assert len(calls) == 1

    def test_different_dates_share_one_fetch(self, provider):
        # Alle Stichtage werden aus derselben Historie bedient.
        prov, calls = provider
        prov.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 18))
        prov.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 21))
        prov.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 22))
        assert len(calls) == 1

    def test_cache_survives_a_new_provider_instance(self, tmp_path, monkeypatch):
        frames = {"EURUSD=X": _raw_fx({"2026-09-21": 1.1380, "2026-09-22": 1.1390})}
        calls = _patch_yf(monkeypatch, frames)
        settings = _online_settings(tmp_path)
        first = YFinanceRateProvider(settings, cache=DataFrameCache(settings.cache))
        first.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 22))
        second = YFinanceRateProvider(settings, cache=DataFrameCache(settings.cache))
        second.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 22))
        assert len(calls) == 1, "der Plattencache haelt ueber Instanzen hinweg"

    def test_historical_ttl_is_longer_than_latest(self, tmp_path):
        # Ein abgeschlossener Handelstag aendert sich nicht mehr, der
        # aktuellste Kurs schon. Geprueft an den Settings, nicht an
        # Modulkonstanten: massgeblich ist, was der Provider wirklich liest.
        cache = _online_settings(tmp_path).cache
        assert cache.fx_historical_ttl_s > cache.fx_latest_ttl_s

    def test_disabled_cache_still_deduplicates(self, tmp_path, monkeypatch):
        frames = {"EURUSD=X": _raw_fx({"2026-09-22": 1.1390})}
        calls = _patch_yf(monkeypatch, frames)
        settings = _online_settings(tmp_path)
        settings.cache.enabled = False
        prov = YFinanceRateProvider(settings, cache=DataFrameCache(settings.cache))
        prov.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 22))
        prov.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 22))
        assert len(calls) == 1, "das Prozess-Memo greift auch ohne Plattencache"


# --------------------------------------------------------- Kreuzkurse ---


class TestCrossRates:
    def test_falls_back_to_usd_triangulation(self, tmp_path, monkeypatch):
        # HKDNOK=X hat bei der echten Quelle keine Historie, die USD-Schenkel
        # schon. Genau dieser Fall.
        frames = {
            "HKDUSD=X": _raw_fx({"2026-09-23": 0.1275}),
            "USDNOK=X": _raw_fx({"2026-09-23": 9.475}),
        }
        calls = _patch_yf(monkeypatch, frames)
        settings = _online_settings(tmp_path)
        prov = YFinanceRateProvider(settings, cache=DataFrameCache(settings.cache))
        rate = prov.get_rate(Currency.HKD, Currency.NOK, date(2026, 9, 23))
        assert rate == pytest.approx(0.1275 * 9.475)
        assert calls[0] == "HKDNOK=X", "der Direktkurs wird zuerst versucht"
        assert "HKDUSD=X" in calls
        assert "USDNOK=X" in calls

    def test_triangulated_matches_direct_quote(self, tmp_path, monkeypatch):
        # Gegenprobe gegen die am 23.09.2026 gemessene Direktnotierung 1.2032.
        frames = {
            "HKDUSD=X": _raw_fx({"2026-09-23": 0.1275}),
            "USDNOK=X": _raw_fx({"2026-09-23": 9.475}),
        }
        _patch_yf(monkeypatch, frames)
        settings = _online_settings(tmp_path)
        prov = YFinanceRateProvider(settings, cache=DataFrameCache(settings.cache))
        rate = prov.get_rate(Currency.HKD, Currency.NOK, date(2026, 9, 23))
        assert rate == pytest.approx(1.2032, rel=0.01)

    def test_direct_quote_wins_when_available(self, tmp_path, monkeypatch):
        frames = {
            "HKDNOK=X": _raw_fx({"2026-09-23": 1.2032}),
            "HKDUSD=X": _raw_fx({"2026-09-23": 0.1275}),
            "USDNOK=X": _raw_fx({"2026-09-23": 9.475}),
        }
        calls = _patch_yf(monkeypatch, frames)
        settings = _online_settings(tmp_path)
        prov = YFinanceRateProvider(settings, cache=DataFrameCache(settings.cache))
        rate = prov.get_rate(Currency.HKD, Currency.NOK, date(2026, 9, 23))
        assert rate == pytest.approx(1.2032)
        assert calls == ["HKDNOK=X"], "kein Umweg, wenn der Direktkurs da ist"

    def test_no_triangulation_when_pivot_is_involved(self, tmp_path, monkeypatch):
        # EUR nach USD kann nicht ueber USD trianguliert werden - ein
        # Fehlschlag ist hier endgueltig und darf keinen zweiten Abruf kosten.
        calls = _patch_yf(monkeypatch, {})
        settings = _online_settings(tmp_path)
        prov = YFinanceRateProvider(settings, cache=DataFrameCache(settings.cache))
        with pytest.raises(SymbolNotFoundError):
            prov.get_rate(Currency.EUR, PIVOT_CURRENCY, date(2026, 9, 23))
        assert calls == ["EURUSD=X"]

    def test_both_legs_missing_raises(self, tmp_path, monkeypatch):
        frames = {"HKDUSD=X": _raw_fx({"2026-09-23": 0.1275})}
        _patch_yf(monkeypatch, frames)
        settings = _online_settings(tmp_path)
        prov = YFinanceRateProvider(settings, cache=DataFrameCache(settings.cache))
        with pytest.raises(SymbolNotFoundError) as excinfo:
            prov.get_rate(Currency.HKD, Currency.NOK, date(2026, 9, 23))
        assert PIVOT_CURRENCY.value in excinfo.value.user_message


# --------------------------------------------------------- Rate Limits --


class TestRateLimit:
    def test_rate_limit_propagates_and_is_not_triangulated(self, tmp_path, monkeypatch):
        # Eine Drosselung darf keinen zweiten und dritten Request nachziehen.
        import yfinance as yf
        from yfinance.exceptions import YFRateLimitError

        calls: list[str] = []

        class _Throttled:
            def __call__(self, symbol: str) -> _Throttled:
                calls.append(symbol)
                return self

            def history(self, **_kwargs: Any) -> pd.DataFrame:
                raise YFRateLimitError

        monkeypatch.setattr(yf, "Ticker", _Throttled())
        settings = _online_settings(tmp_path)
        settings.ratelimit.max_retries = 0
        prov = YFinanceRateProvider(settings, cache=DataFrameCache(settings.cache))
        with pytest.raises(RateLimitError):
            prov.get_rate(Currency.HKD, Currency.NOK, date(2026, 9, 23))
        assert calls == ["HKDNOK=X"]

    def test_bucket_is_consulted(self, tmp_path, monkeypatch):
        frames = {"EURUSD=X": _raw_fx({"2026-09-23": EURUSD_23_09})}
        _patch_yf(monkeypatch, frames)
        settings = _online_settings(tmp_path)
        bucket = TokenBucket(rps=100.0, burst=10, name="test")
        prov = YFinanceRateProvider(settings, cache=DataFrameCache(settings.cache), bucket=bucket)
        prov.get_rate(Currency.EUR, Currency.USD, date(2026, 9, 23))
        assert bucket.burst == 10
