"""Tests des Quartalszahlen-Termins.

Kein Test hier geht ins Netz - die Socket-Sperre der Suite verbietet es.
``yfinance.Ticker`` wird durch eine Attrappe ersetzt, die genau die Form
liefert, die der Live-Abruf tatsaechlich zurueckgibt. Am 24.09.2026 an 78
Symbolen gegen yfinance 1.7 gemessen:

* Aktie mit Termin: ``{"Earnings Date": [date(2026, 10, 29)], ...}`` - in
  78 Symbolen kam **nie** mehr als ein Datum vor.
* ETF, Krypto, Index, unbekanntes Symbol: ``{}``. yfinance faengt den
  HTTP 404 der Quelle intern ab - ``calendar`` ist ein **leeres Dictionary**,
  nicht ``None``, und der Schluessel fehlt ganz.
* Aktie ohne angekuendigten Termin: ``{"Earnings Date": []}``.

Die beiden wichtigsten Tests dieser Datei: der ETF darf keinen Fehler
ausloesen (ein ETF hat schlicht keine Quartalszahlen), und ein vergangener
Termin darf nie als "naechster" erscheinen. Letzteres ist kein theoretischer
Fall - MC.PA lieferte am 24.09.2026 live den 27.07.2026, also ein Datum in
der Vergangenheit.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from btlite.config import Settings
from btlite.core.cache import DataFrameCache
from btlite.core.clock import utc_today
from btlite.core.errors import RateLimitError
from btlite.core.ratelimit import TokenBucket
from btlite.data.base import EarningsCalendarProvider
from btlite.data.fixtures import FixtureProvider
from btlite.data.yfinance_provider import (
    EARNINGS_CALENDAR_KEY,
    YFinanceProvider,
    next_earnings_from_calendar,
)
from btlite.models.market import EarningsDate

# --------------------------------------------------------------- Helfer --

#: Ein Termin sicher in der Zukunft, unabhaengig vom Tag des Testlaufs.
FUTURE = utc_today() + timedelta(days=35)

#: Ein Termin sicher in der Vergangenheit.
PAST = utc_today() - timedelta(days=35)


def _stock(day: date) -> dict[str, Any]:
    """Kalender einer Aktie mit Termin, in der Form der echten Quelle."""
    return {
        "Dividend Date": day,
        "Ex-Dividend Date": day,
        EARNINGS_CALENDAR_KEY: [day],
        "Earnings Average": 1.23,
        "Earnings High": 1.30,
        "Earnings Low": 1.10,
        "Revenue Average": 1_000.0,
        "Revenue High": 1_100.0,
        "Revenue Low": 900.0,
    }


#: Kalender eines ETF. yfinance faengt den HTTP 404 ab und liefert ``{}``.
ETF: dict[str, Any] = {}

#: Kalender einer Aktie ohne angekuendigten Termin - der Schluessel ist da,
#: die Liste ist leer. Gemessen an NESN.SW und OR.PA.
NO_DATE: dict[str, Any] = {EARNINGS_CALENDAR_KEY: []}


class _FakeTicker:
    """Attrappe von ``yfinance.Ticker``. Zaehlt die Abrufe je Symbol."""

    def __init__(self, calendars: dict[str, Any], calls: list[str]) -> None:
        self._calendars = calendars
        self._calls = calls
        self._symbol = ""

    def __call__(self, symbol: str) -> _FakeTicker:
        self._symbol = symbol
        return self

    @property
    def calendar(self) -> Any:
        self._calls.append(self._symbol)
        if self._symbol not in self._calendars:
            # Unbekanntes Symbol: die Quelle antwortet mit 404, yfinance
            # macht daraus ein leeres Dictionary.
            return {}
        value = self._calendars[self._symbol]
        if isinstance(value, Exception):
            raise value
        return value


def _patch_yf(monkeypatch: pytest.MonkeyPatch, calendars: dict[str, Any]) -> list[str]:
    """Haengt die Ticker-Attrappe in yfinance und liefert das Aufrufprotokoll."""
    import yfinance as yf

    calls: list[str] = []
    monkeypatch.setattr(yf, "Ticker", _FakeTicker(calendars, calls))
    return calls


def _online_settings(tmp_path: Path) -> Settings:
    """Settings mit abgeschaltetem Offline-Modus und temporaerem Cache."""
    settings = Settings()
    settings.data.offline = False
    settings.cache.dir = tmp_path / "cache"
    return settings


def _provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calendars: dict[str, Any],
    *,
    bucket: TokenBucket | None = None,
) -> tuple[YFinanceProvider, list[str]]:
    """Live-Provider gegen die Attrappe. Liefert (provider, calls)."""
    calls = _patch_yf(monkeypatch, calendars)
    settings = _online_settings(tmp_path)
    prov = YFinanceProvider(settings, cache=DataFrameCache(settings.cache), bucket=bucket)
    return prov, calls


def _write_earnings_fixture(root: Path, symbol: str, *, has: bool, days: list[str]) -> None:
    """Schreibt ein Termin-Fixture im Ablageformat von ``refresh_fixtures.py``."""
    target = root / "earnings"
    target.mkdir(parents=True, exist_ok=True)
    payload = {"ticker": symbol, "has_earnings": has, "dates": days}
    (target / f"{symbol}.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")


def _offline_provider(tmp_path: Path) -> FixtureProvider:
    """Fixture-Provider gegen ein leeres, eigenes Fixture-Verzeichnis."""
    root = tmp_path / "fixtures"
    root.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    settings.data.offline = True
    settings.data.fixtures_dir = root
    return FixtureProvider(root, settings)


# ------------------------------------------------------------- Vertrag --


class TestProtocol:
    def test_yfinance_satisfies_protocol(self, tmp_path, monkeypatch):
        prov, _ = _provider(tmp_path, monkeypatch, {})
        assert isinstance(prov, EarningsCalendarProvider)

    def test_fixtures_satisfy_protocol(self, tmp_path):
        assert isinstance(_offline_provider(tmp_path), EarningsCalendarProvider)

    def test_ttl_is_24_hours(self):
        # Ein Quartalstermin bewegt sich selten - oefter zu fragen belastet
        # die Quelle ohne Nutzen. Geprueft an den Settings, nicht an einer
        # Modulkonstante: massgeblich ist, was der Provider wirklich liest.
        assert Settings().cache.earnings_ttl_s == 24 * 3600

    def test_the_ttl_can_be_overridden(self, monkeypatch):
        # Wer die Termine haeufiger will, soll das koennen, ohne den Code
        # anzufassen.
        monkeypatch.setenv("BTLITE_CACHE_EARNINGS_TTL_S", "60")
        assert Settings().cache.earnings_ttl_s == 60


class TestModelContract:
    def test_no_earnings_forbids_a_date(self):
        # has_earnings=False heisst "fuehrt grundsaetzlich keine Termine".
        # Ein Datum daneben waere ein Widerspruch.
        with pytest.raises(ValueError, match="has_earnings"):
            EarningsDate(ticker="SPY", next_date=FUTURE, has_earnings=False)

    def test_estimated_defaults_to_true(self):
        # Yahoo kennzeichnet nicht, ob ein Termin bestaetigt ist. Der Vertrag
        # gibt keine Sicherheit vor, die die Quelle nicht hergibt.
        assert EarningsDate(ticker="AAPL", next_date=FUTURE).estimated is True


# ------------------------------------------------------------ Live-Fall --


class TestStockWithDate:
    def test_returns_the_date(self, tmp_path, monkeypatch):
        prov, _ = _provider(tmp_path, monkeypatch, {"AAPL": _stock(FUTURE)})
        assert prov.get_earnings_dates(["AAPL"])["AAPL"].next_date == FUTURE

    def test_marks_it_as_having_earnings(self, tmp_path, monkeypatch):
        prov, _ = _provider(tmp_path, monkeypatch, {"AAPL": _stock(FUTURE)})
        assert prov.get_earnings_dates(["AAPL"])["AAPL"].has_earnings is True

    def test_provenance_names_yfinance(self, tmp_path, monkeypatch):
        prov, _ = _provider(tmp_path, monkeypatch, {"AAPL": _stock(FUTURE)})
        prov_field = prov.get_earnings_dates(["AAPL"])["AAPL"].provenance
        assert prov_field is not None
        assert prov_field.source == "yfinance"


class TestEtfIsNotAnError:
    """Der wichtigste Test der Datei.

    Ein ETF fuehrt keine Quartalszahlen. Der HTTP 404 der Quelle ist dort die
    **richtige Antwort**, kein Fehler - er darf nicht als
    ProviderUnavailableError hochschlagen und die ganze Seite kippen.
    """

    def test_etf_returns_none_without_raising(self, tmp_path, monkeypatch):
        prov, _ = _provider(tmp_path, monkeypatch, {"SPY": ETF})
        assert prov.get_earnings_dates(["SPY"])["SPY"].next_date is None

    def test_etf_is_marked_as_having_no_earnings(self, tmp_path, monkeypatch):
        # Der Unterschied, den die UI braucht: "gibt es nicht" statt
        # "wissen wir nicht".
        prov, _ = _provider(tmp_path, monkeypatch, {"SPY": ETF})
        assert prov.get_earnings_dates(["SPY"])["SPY"].has_earnings is False

    def test_etf_stays_in_the_result(self, tmp_path, monkeypatch):
        prov, _ = _provider(tmp_path, monkeypatch, {"SPY": ETF})
        assert "SPY" in prov.get_earnings_dates(["SPY"])

    def test_calendar_none_is_also_tolerated(self, tmp_path, monkeypatch):
        # Gemessen liefert yfinance 1.7 ein leeres Dictionary. Ein None darf
        # trotzdem nicht durchschlagen.
        prov, _ = _provider(tmp_path, monkeypatch, {"BTC-USD": None})
        assert prov.get_earnings_dates(["BTC-USD"])["BTC-USD"].has_earnings is False


class TestStockWithoutDate:
    def test_empty_list_is_no_error(self, tmp_path, monkeypatch):
        prov, _ = _provider(tmp_path, monkeypatch, {"NESN.SW": NO_DATE})
        assert prov.get_earnings_dates(["NESN.SW"])["NESN.SW"].next_date is None

    def test_empty_list_still_means_it_has_earnings(self, tmp_path, monkeypatch):
        # Eine Aktie ohne angekuendigten Termin ist nicht dasselbe wie ein
        # ETF: hier kann spaeter einer dazukommen.
        prov, _ = _provider(tmp_path, monkeypatch, {"NESN.SW": NO_DATE})
        assert prov.get_earnings_dates(["NESN.SW"])["NESN.SW"].has_earnings is True


class TestUnknownTicker:
    def test_does_not_crash(self, tmp_path, monkeypatch):
        prov, _ = _provider(tmp_path, monkeypatch, {})
        assert prov.get_earnings_dates(["ZZZZNOTREAL"])["ZZZZNOTREAL"].next_date is None

    def test_is_reported_as_without_earnings(self, tmp_path, monkeypatch):
        # Live ist das ein Befund: die Quelle hat geantwortet, und zwar mit
        # 404 auf die Fundamentaldaten. Offline gilt das Gegenteil, siehe
        # TestOfflineWithoutFixture - dort belegt ein fehlendes Fixture nichts.
        prov, _ = _provider(tmp_path, monkeypatch, {})
        result = prov.get_earnings_dates(["ZZZZNOTREAL"])
        assert result["ZZZZNOTREAL"].has_earnings is False


# ------------------------------------------------------- Sammelabruf ----


class TestBatch:
    def test_missing_date_does_not_topple_the_others(self, tmp_path, monkeypatch):
        # Der Kernfall des Depots: ETF, Aktie ohne Termin und Aktie mit
        # Termin nebeneinander. Keiner darf die anderen mitreissen.
        prov, _ = _provider(
            tmp_path,
            monkeypatch,
            {"AAPL": _stock(FUTURE), "SPY": ETF, "NESN.SW": NO_DATE},
        )
        out = prov.get_earnings_dates(["AAPL", "SPY", "NESN.SW"])
        assert out["AAPL"].next_date == FUTURE
        assert out["SPY"].next_date is None
        assert out["NESN.SW"].next_date is None

    def test_exception_on_one_symbol_keeps_the_rest(self, tmp_path, monkeypatch):
        prov, _ = _provider(
            tmp_path,
            monkeypatch,
            {"AAPL": _stock(FUTURE), "BOOM": RuntimeError("kaputt")},
        )
        out = prov.get_earnings_dates(["BOOM", "AAPL"])
        assert out["AAPL"].next_date == FUTURE
        assert "BOOM" not in out

    def test_duplicates_are_fetched_once(self, tmp_path, monkeypatch):
        # Geprueft wird die Entduplizierung selbst, nicht der Cache: ein
        # Token-Bucket-Zaehler faellt auf die Wiederholung herein nicht
        # herein, weil er vor dem Cache liegt. Ueber ``calls`` allein waere
        # der Test gruen geblieben, auch ohne jede Entduplizierung - der
        # zweite Abruf haette schlicht den Cache getroffen.
        acquired: list[int] = []

        class _CountingBucket(TokenBucket):
            def acquire(self, tokens: int = 1) -> float:
                acquired.append(tokens)
                return 0.0

        prov, calls = _provider(
            tmp_path,
            monkeypatch,
            {"AAPL": _stock(FUTURE)},
            bucket=_CountingBucket(rps=100.0, burst=10, name="test"),
        )
        prov.get_earnings_dates(["AAPL", "AAPL", "aapl"])
        assert calls == ["AAPL"]
        assert len(acquired) == 1, "dasselbe Symbol darf nur einen Token kosten"

    def test_duplicates_collapse_to_one_entry(self, tmp_path, monkeypatch):
        # Gross- und Kleinschreibung fallen zusammen: das Ergebnis traegt das
        # kanonische Symbol, nicht beide Schreibweisen.
        prov, _ = _provider(tmp_path, monkeypatch, {"AAPL": _stock(FUTURE)})
        out = prov.get_earnings_dates(["AAPL", "aapl"])
        assert list(out) == ["AAPL"]

    def test_empty_input_returns_empty(self, tmp_path, monkeypatch):
        prov, calls = _provider(tmp_path, monkeypatch, {})
        assert prov.get_earnings_dates([]) == {}
        assert calls == []

    def test_rate_limit_is_not_swallowed(self, tmp_path, monkeypatch):
        # Projektregel: Drosselung wird nie umgangen und nie zu "kein Termin"
        # verharmlost. Der Fehler geht nach oben.
        prov, _ = _provider(
            tmp_path,
            monkeypatch,
            {"AAPL": RateLimitError("Yahoo Finance is currently throttling requests.")},
        )
        with pytest.raises(RateLimitError):
            prov.get_earnings_dates(["AAPL"])


# ----------------------------------------------------------- Ratelimit --


class TestRateLimiting:
    def test_bucket_is_used_per_request(self, tmp_path, monkeypatch):
        # Bei 20 Positionen sind das 20 Anfragen - genau dafuer ist der
        # Token-Bucket da.
        acquired: list[int] = []

        class _CountingBucket(TokenBucket):
            def acquire(self, tokens: int = 1) -> float:
                acquired.append(tokens)
                return 0.0

        bucket = _CountingBucket(rps=100.0, burst=10, name="test")
        prov, _ = _provider(
            tmp_path,
            monkeypatch,
            {"AAPL": _stock(FUTURE), "MSFT": _stock(FUTURE)},
            bucket=bucket,
        )
        prov.get_earnings_dates(["AAPL", "MSFT"])
        assert len(acquired) == 2, "je Abruf muss ein Token entnommen werden"


# ------------------------------------------------------- Vergangenheit --


class TestPastDateIsNotNext:
    """Ein abgelaufener Termin ist kein *naechster*.

    Kein theoretischer Fall: MC.PA lieferte am 24.09.2026 live den
    27.07.2026. Wer das ungeprueft anzeigt, behauptet einen Termin, der
    laengst vorbei ist.
    """

    def test_past_date_becomes_none(self, tmp_path, monkeypatch):
        prov, _ = _provider(tmp_path, monkeypatch, {"MC.PA": _stock(PAST)})
        assert prov.get_earnings_dates(["MC.PA"])["MC.PA"].next_date is None

    def test_past_date_keeps_has_earnings_true(self, tmp_path, monkeypatch):
        # Der Titel fuehrt sehr wohl Quartalszahlen - nur ist der naechste
        # Termin noch nicht bekannt.
        prov, _ = _provider(tmp_path, monkeypatch, {"MC.PA": _stock(PAST)})
        assert prov.get_earnings_dates(["MC.PA"])["MC.PA"].has_earnings is True

    def test_today_still_counts_as_next(self):
        # Am Tag der Zahlen ist der Termin noch der naechste.
        today = utc_today()
        out = next_earnings_from_calendar(_stock(today), ticker="AAPL")
        assert out.next_date == today

    def test_past_is_skipped_in_favour_of_future(self):
        calendar = {EARNINGS_CALENDAR_KEY: [PAST, FUTURE]}
        assert next_earnings_from_calendar(calendar, ticker="X").next_date == FUTURE

    def test_earliest_future_date_wins(self):
        # Yahoo fuehrt laut Doku bis zu zwei Eintraege als unbestaetigtes
        # Zeitfenster. Gilt der Fall doch einmal, ist der fruehere der
        # naechste - ein spaeterer waere zu optimistisch.
        later = FUTURE + timedelta(days=7)
        calendar = {EARNINGS_CALENDAR_KEY: [later, FUTURE]}
        assert next_earnings_from_calendar(calendar, ticker="X").next_date == FUTURE

    def test_today_parameter_is_honoured(self):
        # Erlaubt den Test gegen einen festen Stichtag, ohne die Uhr zu
        # stellen.
        calendar = {EARNINGS_CALENDAR_KEY: [date(2026, 10, 29)]}
        out = next_earnings_from_calendar(calendar, ticker="AAPL", today=date(2026, 11, 1))
        assert out.next_date is None


# ---------------------------------------------------------- Offline ----


class TestOfflineFromFixtures:
    def test_reads_the_shipped_fixture(self):
        # Offline ist der Default beim Start der App. Ohne Fixtures zeigte
        # die Spalte immer "n/a" - dann waere das Feature offline wertlos.
        settings = Settings()
        settings.data.offline = True
        out = FixtureProvider(settings=settings).get_earnings_dates(["AAPL"])
        assert out["AAPL"].has_earnings is True

    def test_shipped_etf_fixture_has_no_earnings(self):
        # SPY ist bewusst als Fixture dabei: sonst waere der ETF-Fall
        # offline gar nicht pruefbar.
        settings = Settings()
        settings.data.offline = True
        out = FixtureProvider(settings=settings).get_earnings_dates(["SPY"])
        assert out["SPY"].has_earnings is False
        assert out["SPY"].next_date is None

    def test_provenance_is_fixture(self):
        settings = Settings()
        settings.data.offline = True
        out = FixtureProvider(settings=settings).get_earnings_dates(["AAPL"])
        prov = out["AAPL"].provenance
        assert prov is not None
        assert prov.source == "fixture"

    def test_future_fixture_date_is_returned(self, tmp_path):
        prov = _offline_provider(tmp_path)
        _write_earnings_fixture(prov.fixtures_dir, "AAPL", has=True, days=[FUTURE.isoformat()])
        assert prov.get_earnings_dates(["AAPL"])["AAPL"].next_date == FUTURE

    def test_stale_fixture_date_is_not_returned(self, tmp_path):
        # Der Kern der Veraltungs-Loesung: das Fixture haelt den Rohbefund,
        # die Auswahl des naechsten Termins laeuft bei jedem Lesen neu gegen
        # den heutigen Tag. Ein abgelaufenes Datum wird zu None, ohne dass
        # jemand die Datei anfassen muss.
        prov = _offline_provider(tmp_path)
        _write_earnings_fixture(prov.fixtures_dir, "AAPL", has=True, days=[PAST.isoformat()])
        result = prov.get_earnings_dates(["AAPL"])["AAPL"]
        assert result.next_date is None
        assert result.has_earnings is True


class TestOfflineWithoutFixture:
    """Anders als bei Wechselkursen ist ein fehlendes Fixture hier kein Fehler.

    Ein fehlender Termin ist eine legitime Aussage ("kein Termin bekannt"),
    ein fehlender Wechselkurs nicht - dort waere jede Zahl geraten.
    """

    def test_returns_none_instead_of_raising(self, tmp_path):
        prov = _offline_provider(tmp_path)
        assert prov.get_earnings_dates(["AAPL"])["AAPL"].next_date is None

    def test_symbol_stays_in_the_result(self, tmp_path):
        prov = _offline_provider(tmp_path)
        assert "AAPL" in prov.get_earnings_dates(["AAPL"])

    def test_missing_fixture_means_unknown_not_proven_absent(self, tmp_path):
        # Der Unterschied, den eine reine next_date-Pruefung nicht faengt:
        # "nicht eingefroren" ist fehlendes Wissen, kein Beleg. Sonst bekaeme
        # ein vertippter Ticker den Strich fuer "fuehrt keine Earnings" statt
        # "n/a".
        prov = _offline_provider(tmp_path)
        assert prov.get_earnings_dates(["TIPPFEHLER"])["TIPPFEHLER"].has_earnings is True

    def test_frozen_etf_stays_proven_absent(self, tmp_path):
        # Gegenprobe: ein eingefrorenes has_earnings=false ist sehr wohl ein
        # Befund und bleibt False.
        prov = _offline_provider(tmp_path)
        _write_earnings_fixture(prov.fixtures_dir, "SPY", has=False, days=[])
        assert prov.get_earnings_dates(["SPY"])["SPY"].has_earnings is False

    def test_broken_json_does_not_topple_the_page(self, tmp_path):
        prov = _offline_provider(tmp_path)
        target = prov.fixtures_dir / "earnings"
        target.mkdir(parents=True, exist_ok=True)
        (target / "AAPL.json").write_text("{kaputt", encoding="utf-8")
        assert prov.get_earnings_dates(["AAPL"])["AAPL"].next_date is None

    def test_broken_json_proves_nothing(self, tmp_path):
        # Eine kaputte Datei ist kein Beleg dafuer, dass es keine Earnings gibt.
        prov = _offline_provider(tmp_path)
        target = prov.fixtures_dir / "earnings"
        target.mkdir(parents=True, exist_ok=True)
        (target / "AAPL.json").write_text("{kaputt", encoding="utf-8")
        assert prov.get_earnings_dates(["AAPL"])["AAPL"].has_earnings is True


# ------------------------------------------------------------- Helfer ---


class TestNextEarningsFromCalendar:
    def test_missing_key_means_no_earnings(self):
        assert next_earnings_from_calendar({}, ticker="SPY").has_earnings is False

    def test_present_but_empty_means_unknown_date(self):
        out = next_earnings_from_calendar(NO_DATE, ticker="NESN.SW")
        assert out.has_earnings is True
        assert out.next_date is None

    def test_non_dict_is_treated_as_no_earnings(self):
        assert next_earnings_from_calendar(None, ticker="X").has_earnings is False

    def test_garbage_entries_are_ignored(self):
        calendar = {EARNINGS_CALENDAR_KEY: ["kein Datum", None, FUTURE]}
        assert next_earnings_from_calendar(calendar, ticker="X").next_date == FUTURE

    def test_ticker_is_normalized(self):
        assert next_earnings_from_calendar({}, ticker="sap.de").ticker == "SAP.DE"
