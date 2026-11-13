"""Offline-Provider auf Basis eingefrorener API-Antworten.

Dieser Provider ist **kein Nebenprodukt der Tests**, sondern ein zentrales
Deliverable: alle Rechenschichten des Projekts entwickeln und testen gegen ihn.
Er erfuellt dieselben Protocols wie
:class:`~btlite.data.yfinance_provider.YFinanceProvider` und
:class:`~btlite.data.finviz_provider.FinvizProvider`, geht aber **nie** ins
Netz - er liest ausschliesslich aus ``tests/fixtures``.

Damit gilt fuer jede Schicht darueber: was gegen den FixtureProvider laeuft,
laeuft auch gegen die echte Quelle, weil beide denselben Vertrag erfuellen.

Ablage der Fixtures (erzeugt von ``scripts/refresh_fixtures.py``):

* ``prices/<TICKER>_<interval>.parquet`` - OHLCV mit UTC-Index.
* ``statements/<TICKER>_<typ>_<periode>.parquet`` - Zeilen = Positionen,
  Spalten = Perioden als ISO-Strings.
* ``info/<TICKER>.json`` - Stammdaten aus ``Ticker.info``.
* ``screener/overview.parquet`` - ein Finviz-Screener-Ergebnis.
* ``search/<slug>.json`` - eingefrorene Antworten der Symbolsuche.
* ``earnings/<TICKER>.json`` - naechster Quartalszahlen-Termin.

Die ``Provenance`` traegt hier :attr:`~btlite.models.common.DataSource.FIXTURE`
und ``from_cache=True``: die Daten sind eingefroren, nicht frisch. Wer sie fuer
aktuell haelt, soll das am Herkunftsnachweis erkennen koennen.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd
from pydantic import ValidationError

from btlite.config import Settings, get_settings
from btlite.core.clock import ensure_utc, utc_now
from btlite.core.errors import DataValidationError, SymbolNotFoundError, UnknownFilterError
from btlite.core.logging import get_logger
from btlite.data.search import (
    display_exchange,
    hits_from_quotes,
    normalize_query,
    prepare_results,
    query_slug,
)
from btlite.data.yfinance_provider import (
    EARNINGS_CALENDAR_KEY,
    next_earnings_from_calendar,
)
from btlite.models.common import (
    AssetClass,
    Currency,
    DataSource,
    Interval,
    Period,
    Provenance,
    normalize_ticker,
)
from btlite.models.fundamentals import FinancialStatement, StatementBundle, StatementType
from btlite.models.market import EarningsDate, Instrument, PriceSeries, Quote, SearchHit
from btlite.models.screening import ScreenRequest, ScreenResult

_LOG = get_logger(__name__)

#: Zusatz aller Meldungen ueber fehlende Offline-Daten. Die Suche kennt
#: inzwischen 558 Titel, Kurse liegen offline aber nur zu wenigen vor -
#: ohne diesen Hinweis wuesste der Nutzer nicht, woran es liegt.
_OFFLINE_HINT: Final[str] = (
    "Offline mode only knows the frozen titles. Set BTLITE_DATA_OFFLINE=0 for live data."
)

#: Sentinel fuer "zu diesem Symbol liegt offline gar kein Termin-Fixture vor".
#:
#: Bewusst von ``None`` verschieden: ``None`` ist der **Befund** "fuehrt keine
#: Quartalszahlen" (ETF), dieser Wert dagegen blosses Nichtwissen. Beides auf
#: ``None`` abzubilden hiesse, einem vertippten Ticker zu bescheinigen, er
#: fuehre keine Quartalszahlen.
_UNKNOWN_OFFLINE: Final[object] = object()

#: Abbildung ``Period`` auf das Dateinamens-Suffix der Statement-Fixtures.
_PERIOD_SUFFIX: Final[dict[Period, str]] = {
    Period.ANNUAL: "annual",
    Period.QUARTERLY: "quarterly",
    Period.TTM: "ttm",
}

#: Abbildung yfinance-``quoteType`` auf die Instrumentenklasse des Vertrags.
_ASSET_CLASS_MAP: Final[dict[str, AssetClass]] = {
    "EQUITY": AssetClass.EQUITY,
    "ETF": AssetClass.ETF,
    "INDEX": AssetClass.INDEX,
    "CRYPTOCURRENCY": AssetClass.CRYPTO,
    "MUTUALFUND": AssetClass.FUND,
}


class FixtureProvider:
    """Marktdaten aus eingefrorenen Antworten, ohne jeden Netzwerkzugriff.

    Erfuellt :class:`~btlite.data.base.MarketDataProvider` und
    :class:`~btlite.data.base.ScreenerProvider` strukturell, ohne von den
    Protocols zu erben.

    Der Provider haelt geladene Dateien im Speicher vor. Ein DataFrame wird
    beim Ausliefern kopiert, damit ein Aufrufer den Cache nicht versehentlich
    veraendert.
    """

    def __init__(
        self,
        fixtures_dir: Path | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialisiert den Provider.

        Args:
            fixtures_dir: Wurzel der Fixtures. ``None`` nimmt
                ``DataSettings.fixtures_dir``.
            settings: Projekt-Settings. ``None`` nimmt die globalen.
        """
        self._settings = settings if settings is not None else get_settings()
        self._dir = (
            Path(fixtures_dir) if fixtures_dir is not None else self._settings.data.fixtures_dir
        )
        self._frames: dict[Path, pd.DataFrame] = {}
        self._infos: dict[str, dict[str, Any]] = {}
        self._instrument_hit_cache: list[SearchHit] | None = None
        self._directory_hit_cache: list[SearchHit] | None = None
        self._frozen_at = self._read_frozen_at()

    def _read_frozen_at(self) -> datetime:
        """Liest den Einfrierzeitpunkt der Fixtures aus dem Manifest.

        Returns:
            Zeitpunkt des letzten ``refresh_fixtures.py``-Laufs, tz-aware UTC.
            Faellt auf die Mtime des Verzeichnisses zurueck, wenn kein
            Manifest vorliegt, und zuletzt auf ``utc_now()``.
        """
        manifest = self._dir / "manifest.json"
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            return ensure_utc(datetime.fromisoformat(str(raw["last_run"])))
        except (OSError, KeyError, ValueError):
            _LOG.debug("Kein lesbares Fixture-Manifest unter %s", manifest)

        try:
            return ensure_utc(datetime.fromtimestamp(self._dir.stat().st_mtime, tz=UTC))
        except OSError:
            return utc_now()

    @property
    def fixtures_dir(self) -> Path:
        """Wurzelverzeichnis der Fixtures.

        Returns:
            Pfad zum Fixture-Verzeichnis.
        """
        return self._dir

    def available_tickers(self) -> list[str]:
        """Listet die Symbole, fuer die eine Kurshistorie vorliegt.

        Returns:
            Sortierte Symbolliste. Leer, wenn keine Fixtures vorhanden sind.
        """
        prices = self._dir / "prices"
        if not prices.is_dir():
            return []
        return sorted({p.stem.split("_")[0] for p in prices.glob("*.parquet")})

    # ------------------------------------------------------------ intern --
    def _provenance(self, note: str | None = None) -> Provenance:
        """Baut den Herkunftsnachweis fuer eine Fixture-Antwort.

        Der ausgewiesene Verzug ist der **tatsaechliche** Abstand zwischen dem
        Einfrierzeitpunkt der Fixtures und jetzt, nicht der 15-Minuten-Wert
        einer Live-Quelle. Fixture-Preise sind Tage alt; sie als
        viertelstuendlich verzoegert auszuweisen waere eine Falschangabe.

        Args:
            note: Optionaler Freitext-Hinweis fuer das UI.

        Returns:
            Provenance mit ``source=FIXTURE``, ``from_cache=True`` und dem
            echten Alter der eingefrorenen Daten in Minuten.
        """
        now = utc_now()
        age_minutes = max(0, int((now - self._frozen_at).total_seconds() // 60))
        stamp = f"As of {self._frozen_at:%Y-%m-%d}"
        return Provenance(
            source=DataSource.FIXTURE,
            fetched_at=now,
            from_cache=True,
            delay_minutes=age_minutes,
            note=f"{note} {stamp}" if note else stamp,
        )

    def _read_parquet(self, path: Path) -> pd.DataFrame:
        """Liest eine parquet-Datei, gepuffert im Speicher.

        Args:
            path: Pfad zur Datei.

        Returns:
            Kopie des Frames, damit Aufrufer den Puffer nicht veraendern.

        Raises:
            DataValidationError: Wenn die Datei nicht lesbar ist.
        """
        if path not in self._frames:
            try:
                self._frames[path] = pd.read_parquet(path)
            except Exception as exc:
                msg = f"Fixture {path.name} cannot be read."
                raise DataValidationError(msg, detail=f"{type(exc).__name__}: {exc}") from exc
        return self._frames[path].copy()

    def _read_info(self, symbol: str) -> dict[str, Any]:
        """Liest die Stammdaten eines Symbols aus dem JSON-Fixture.

        Args:
            symbol: Kanonisches Symbol.

        Returns:
            Das Roh-Dictionary.

        Raises:
            SymbolNotFoundError: Wenn kein Fixture existiert.
            DataValidationError: Wenn das JSON defekt ist.
        """
        if symbol in self._infos:
            return self._infos[symbol]
        path = self._dir / "info" / f"{symbol}.json"
        if not path.is_file():
            msg = f"No master data available offline for {symbol}. {_OFFLINE_HINT}"
            raise SymbolNotFoundError(msg, detail=f"erwartet: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            msg = f"The master data fixture for {symbol} is corrupt."
            raise DataValidationError(msg, detail=str(exc)) from exc
        if not isinstance(payload, dict):
            msg = f"The master data fixture for {symbol} is not an object."
            raise DataValidationError(msg, detail=f"typ={type(payload).__name__}")
        self._infos[symbol] = payload
        return payload

    def _currency_of(self, symbol: str) -> Currency:
        """Ermittelt die Waehrung eines Symbols aus dem Stammdaten-Fixture.

        Args:
            symbol: Kanonisches Symbol.

        Returns:
            Waehrung laut Fixture, sonst :attr:`Currency.USD`.
        """
        try:
            info = self._read_info(symbol)
        except (SymbolNotFoundError, DataValidationError):
            return Currency.USD
        raw = info.get("currency")
        if not isinstance(raw, str):
            return Currency.USD
        try:
            return Currency(raw.strip().upper())
        except ValueError:
            return Currency.OTHER

    # ------------------------------------------------- InstrumentProvider --
    def get_instrument(self, ticker: str) -> Instrument:
        """Holt die Stammdaten eines Symbols aus dem Fixture.

        Args:
            ticker: Kanonisches Symbol.

        Returns:
            Validierte Stammdaten. Nicht hinterlegte Felder sind ``None``.

        Raises:
            SymbolNotFoundError: Wenn kein Fixture existiert.
            DataValidationError: Wenn das Fixture defekt ist.
        """
        symbol = normalize_ticker(ticker)
        info = self._read_info(symbol)
        quote_type = str(info.get("quoteType", "")).upper()
        employees = info.get("fullTimeEmployees")
        return Instrument(
            ticker=symbol,
            name=info.get("longName") or info.get("shortName"),
            asset_class=_ASSET_CLASS_MAP.get(quote_type, AssetClass.OTHER),
            currency=self._currency_of(symbol),
            exchange=info.get("exchange"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            country=info.get("country"),
            website=info.get("website"),
            description=info.get("longBusinessSummary"),
            employees=int(employees) if isinstance(employees, int | float) else None,
            market_cap=_opt_float(info.get("marketCap")),
            shares_outstanding=_opt_float(info.get("sharesOutstanding")),
            dividend_rate=_opt_float(info.get("trailingAnnualDividendRate")),
            dividend_yield=_opt_float(info.get("trailingAnnualDividendYield")),
            provenance=self._provenance(),
        )

    # ------------------------------------------------------- SymbolSearch --
    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        """Sucht Instrumente in den eingefrorenen Daten, ohne jedes Netz.

        Durchsucht zwei Quellen und fuehrt sie zusammen:

        1. Die eingefrorene Suchantwort zu genau dieser Anfrage, falls unter
           ``search/<slug>.json`` eine liegt.
        2. Die Stammdaten der Fixture-Ticker, aufbereitet als Treffer.

        Der zweite Teil ist kein Beiwerk: Yahoos Suche kennt keine
        Tippfehler, die eingefrorene Antwort auf ``microsft`` ist deshalb
        leer. Erst das lokale Fuzzy-Ranking aus ``data/search.py`` ueber den
        Stammdatenbestand findet dort MSFT.

        Args:
            query: Suchbegriff, Name oder Kuerzel. Gross- und Kleinschreibung
                spielt keine Rolle.
            limit: Maximale Trefferzahl. Werte kleiner 1 liefern eine leere
                Liste.

        Returns:
            Treffer, absteigend nach Relevanz, ueber Boersen hinweg
            zusammengefasst. Hoechstens ``limit`` Eintraege. Leer bei leerer
            Eingabe - das ist kein Fehler.
        """
        if limit < 1 or not normalize_query(query):
            return []

        candidates: list[SearchHit] = [
            *self._frozen_search_hits(query),
            *self._instrument_hits(),
            *self._directory_hits(),
        ]
        return prepare_results(candidates, query, limit=limit)

    def _frozen_search_hits(self, query: str) -> list[SearchHit]:
        """Liest die eingefrorene Suchantwort zu einer Anfrage.

        Args:
            query: Roheingabe des Nutzers.

        Returns:
            Treffer aus dem Fixture, oder eine leere Liste wenn zu dieser
            Anfrage keines vorliegt. Ein fehlendes oder defektes Fixture ist
            hier **kein** Fehler: die Stammdaten-Treffer bedienen die Suche
            dann allein.
        """
        slug = query_slug(query)
        if not slug:
            return []
        path = self._dir / "search" / f"{slug}.json"
        if not path.is_file():
            _LOG.debug("Keine eingefrorene Suchantwort unter %s", path)
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.warning("Such-Fixture %s ist defekt: %s", path.name, exc)
            return []
        if not isinstance(payload, dict):
            return []
        quotes = payload.get("quotes")
        if not isinstance(quotes, list):
            return []
        return hits_from_quotes(quotes)

    def _directory_hits(self) -> list[SearchHit]:
        """Baut Treffer aus dem Verzeichnis der Indexmitglieder.

        Ohne dieses Verzeichnis kennt die Suche nur die knapp zehn Ticker
        mit vollen Stammdaten-Fixtures - "tesla" oder "sap" fanden deshalb
        nichts. Die Datei traegt Kuerzel, Firmenname und Branche der
        Mitglieder der unterstuetzten Indizes und ist damit die
        Namensquelle der Offline-Suche.

        Kursdaten stehen nicht darin: ein Treffer bedeutet "dieses Papier
        gibt es", nicht "dazu liegen Kurse vor". Die Seiten melden
        fehlende Kurse selbst.

        Returns:
            Ein Treffer je Eintrag, ohne Score. Leer, wenn die Datei fehlt
            oder defekt ist - das ist kein Fehler, die anderen beiden
            Quellen bedienen die Suche dann allein.
        """
        if self._directory_hit_cache is None:
            self._directory_hit_cache = self._load_directory()
        return list(self._directory_hit_cache)

    def _load_directory(self) -> list[SearchHit]:
        """Liest das Verzeichnis der Indexmitglieder von der Platte.

        Returns:
            Treffer je Eintrag, oder eine leere Liste bei fehlender oder
            defekter Datei.
        """
        path = self._dir / "directory.json"
        if not path.is_file():
            _LOG.debug("Kein Titelverzeichnis unter %s", path)
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.warning("Titelverzeichnis %s ist defekt: %s", path.name, exc)
            return []
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            return []

        hits: list[SearchHit] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            symbol = str(entry.get("ticker", "")).strip()
            if not symbol:
                continue
            try:
                hits.append(
                    SearchHit(
                        ticker=symbol,
                        name=str(entry.get("name") or "") or None,
                        exchange=None,
                        sector=str(entry.get("sector") or "") or None,
                        score=None,
                    )
                )
            except ValidationError as exc:
                _LOG.debug("Verzeichniseintrag %s uebergangen: %s", symbol, exc)
        return hits

    def _instrument_hits(self) -> list[SearchHit]:
        """Baut Treffer aus den Stammdaten aller Fixture-Ticker.

        Das Ergebnis wird gepuffert: die Stammdaten aendern sich waehrend
        eines Prozesslaufs nicht, und jede Tastatureingabe im UI loest sonst
        zehn JSON-Leseoperationen aus.

        Returns:
            Ein Treffer je Ticker mit lesbaren Stammdaten, ohne Score - die
            Rangfolge bestimmt damit allein die Namensaehnlichkeit.
        """
        if self._instrument_hit_cache is None:
            hits: list[SearchHit] = []
            for symbol in self.available_tickers():
                try:
                    instrument = self.get_instrument(symbol)
                except (SymbolNotFoundError, DataValidationError) as exc:
                    _LOG.debug("Stammdaten fuer %s nicht durchsuchbar: %s", symbol, exc)
                    continue
                hits.append(
                    SearchHit(
                        ticker=instrument.ticker,
                        name=instrument.name,
                        exchange=display_exchange(instrument.exchange),
                        asset_class=instrument.asset_class,
                        sector=instrument.sector,
                        score=None,
                    )
                )
            self._instrument_hit_cache = hits
        return list(self._instrument_hit_cache)

    # ------------------------------------------------------ PriceProvider --
    def get_history(
        self,
        ticker: str,
        *,
        interval: Interval = Interval.D1,
        period: str | None = "1y",
        start: datetime | None = None,
        end: datetime | None = None,
        auto_adjust: bool = True,
        include_currency: bool = True,
    ) -> PriceSeries:
        """Holt die OHLCV-Historie eines Symbols aus dem Fixture.

        ``period`` wird bewusst **ignoriert**: das Fixture hat den Zeitraum,
        den es beim Einfrieren hatte. ``start`` und ``end`` schneiden die
        vorhandene Reihe dagegen zu, damit Tests einen Ausschnitt anfordern
        koennen.

        Args:
            ticker: Kanonisches Symbol.
            interval: Kerzen-Intervall. Es liegen nur Tagesdaten vor.
            period: Relativer Zeitraum. Ohne Wirkung, siehe oben.
            start: Startzeitpunkt, tz-aware UTC. Schneidet die Reihe.
            end: Endzeitpunkt, tz-aware UTC. Schneidet die Reihe.
            auto_adjust: Wird in ``PriceSeries.adjusted`` uebernommen. Die
                Fixtures sind mit ``auto_adjust=True`` gezogen.
            include_currency: Ohne Wirkung; Fixtures tragen die Waehrung
                bereits lokal.

        Returns:
            Kursreihe mit UTC-Index, aufsteigend, ohne Duplikate.

        Raises:
            SymbolNotFoundError: Wenn kein Fixture existiert oder der
                gewaehlte Ausschnitt leer ist.
            DataValidationError: Wenn das Fixture defekt ist.
        """
        del include_currency  # Fixtures tragen die Waehrung bereits lokal.
        symbol = normalize_ticker(ticker)
        path = self._dir / "prices" / f"{symbol}_{interval.value}.parquet"
        if not path.is_file():
            msg = f"No price history available offline for {symbol}. {_OFFLINE_HINT}"
            raise SymbolNotFoundError(
                msg, detail=f"erwartet: {path}, vorhanden: {self.available_tickers()}"
            )

        df = self._read_parquet(path)
        if start is not None:
            df = df[df.index >= ensure_utc(start)]
        if end is not None:
            df = df[df.index <= ensure_utc(end)]
        if df.empty:
            msg = f"The requested period contains no price data for {symbol}."
            raise SymbolNotFoundError(msg, detail=f"start={start}, end={end}")

        return PriceSeries(
            ticker=symbol,
            interval=interval,
            currency=self._currency_of(symbol),
            adjusted=auto_adjust,
            data=df,
            provenance=self._provenance("Frozen price series from tests/fixtures."),
        )

    def get_histories(
        self,
        tickers: Sequence[str],
        *,
        interval: Interval = Interval.D1,
        period: str | None = "1y",
        start: datetime | None = None,
        end: datetime | None = None,
        auto_adjust: bool = True,
    ) -> dict[str, PriceSeries]:
        """Holt Historien fuer mehrere Symbole aus den Fixtures.

        Symbole ohne Fixture fehlen im Ergebnis, statt den gesamten Aufruf
        scheitern zu lassen - dasselbe Verhalten wie beim echten Provider.

        Args:
            tickers: Kanonische Symbole. Duplikate werden entfernt.
            interval: Kerzen-Intervall.
            period: Relativer Zeitraum. Ohne Wirkung, siehe :meth:`get_history`.
            start: Startzeitpunkt, tz-aware UTC.
            end: Endzeitpunkt, tz-aware UTC.
            auto_adjust: Wird in ``PriceSeries.adjusted`` uebernommen.

        Returns:
            Abbildung Symbol auf Kursreihe, nur fuer vorhandene Fixtures.
        """
        out: dict[str, PriceSeries] = {}
        for symbol in _unique(tickers):
            try:
                out[symbol] = self.get_history(
                    symbol,
                    interval=interval,
                    period=period,
                    start=start,
                    end=end,
                    auto_adjust=auto_adjust,
                )
            except (SymbolNotFoundError, DataValidationError) as exc:
                _LOG.debug("Fixture-Historie fuer %s uebersprungen: %s", symbol, exc)
        return out

    def get_quotes(self, tickers: Sequence[str]) -> dict[str, Quote]:
        """Erzeugt Momentaufnahmen aus der letzten Kerze der Fixtures.

        Es gibt keine eingefrorenen Live-Quotes - ein Quote ist per Definition
        fluechtig. Der Provider leitet ihn deshalb aus der jeweils letzten
        Kerze der Kurshistorie ab: ``price`` ist deren ``close``,
        ``previous_close`` der ``close`` der Kerze davor. Der Zeitstempel ist
        der der Kerze, nicht "jetzt" - so bleibt der Wert ehrlich datiert.

        Args:
            tickers: Kanonische Symbole. Duplikate werden entfernt.

        Returns:
            Abbildung Symbol auf Quote, nur fuer vorhandene Fixtures.
        """
        out: dict[str, Quote] = {}
        for symbol in _unique(tickers):
            try:
                series = self.get_history(symbol, interval=Interval.D1, period=None)
            except (SymbolNotFoundError, DataValidationError) as exc:
                _LOG.debug("Fixture-Quote fuer %s uebersprungen: %s", symbol, exc)
                continue

            df = series.data
            last = df.iloc[-1]
            price = _opt_float(last["close"])
            if price is None or price <= 0:
                continue
            previous = _opt_float(df["close"].iloc[-2]) if len(df) >= 2 else None
            timestamp: pd.Timestamp = df.index[-1]

            out[symbol] = Quote(
                ticker=symbol,
                price=price,
                currency=series.currency,
                previous_close=previous if previous and previous > 0 else None,
                open=_positive(last.get("open")),
                day_high=_positive(last.get("high")),
                day_low=_positive(last.get("low")),
                volume=_opt_float(last.get("volume")),
                bid=None,
                ask=None,
                timestamp=timestamp.to_pydatetime(),
                provenance=self._provenance("Derived from the last candle of the fixture."),
            )
        return out

    # ---------------------------------------- EarningsCalendarProvider --
    def get_earnings_dates(self, tickers: Sequence[str]) -> dict[str, EarningsDate]:
        """Liest die naechsten Quartalszahlen-Termine aus den Fixtures.

        Anders als bei einem fehlenden Wechselkurs ist ein fehlendes Fixture
        hier **kein Fehler**: "kein Termin bekannt" ist eine legitime
        Aussage, ein geratener Kurs waere dagegen eine Falschaussage. Ein
        Symbol ohne Datei bekommt deshalb ``next_date=None`` statt einer
        Exception - die Spalte zeigt dann "n/a", die Seite bleibt stehen.

        Ein eingefrorener Termin **veraltet**. Liegt er heute in der
        Vergangenheit, liefert diese Methode ``next_date=None``, nicht das
        abgelaufene Datum: ein vergangener Termin ist kein *naechster*. Die
        Auswertung laeuft bei jedem Aufruf neu gegen den heutigen Tag, das
        Fixture selbst bleibt unveraendert.

        Args:
            tickers: Kanonische Symbole. Duplikate werden entfernt.

        Returns:
            Abbildung Symbol auf Termin, fuer **jedes** angefragte Symbol.
            Reihenfolge nicht garantiert.
        """
        out: dict[str, EarningsDate] = {}
        for symbol in _unique(tickers):
            calendar = self._read_earnings(symbol)
            if calendar is _UNKNOWN_OFFLINE:
                # Kein Fixture heisst nur "nicht eingefroren", nicht "fuehrt
                # keine Quartalszahlen". Ein vertippter Ticker bekaeme sonst
                # den Strich fuer "gibt es nicht" statt "n/a".
                out[symbol] = EarningsDate(
                    ticker=symbol,
                    next_date=None,
                    has_earnings=True,
                    provenance=self._provenance(),
                )
                continue
            out[symbol] = next_earnings_from_calendar(
                calendar,
                ticker=symbol,
                provenance=self._provenance(),
            )
        return out

    def _read_earnings(self, symbol: str) -> dict[str, list[date]] | object | None:
        """Liest das Termin-Fixture eines Symbols.

        Unterscheidet drei Lagen, weil sie fuer die Anzeige Verschiedenes
        bedeuten:

        * :data:`_UNKNOWN_OFFLINE` - kein oder kein lesbares Fixture. Das ist
          **fehlendes Wissen**, kein Beleg: offline heisst "nicht eingefroren",
          nicht "fuehrt keine Quartalszahlen".
        * ``None`` - das Fixture haelt fest, dass die Quelle keine
          Quartalszahlen fuehrt (``has_earnings: false``, etwa SPY). Das ist
          ein **Befund** und wird als solcher weitergegeben.
        * Ein Dictionary - die eingefrorenen Termine.

        Args:
            symbol: Kanonisches Symbol.

        Returns:
            Einen der drei obigen Werte. Ein defektes JSON gilt als fehlendes
            Wissen - eine kaputte Datei darf die Depotseite nicht kippen und
            darf erst recht nichts belegen.
        """
        path = self._dir / "earnings" / f"{symbol}.json"
        if not path.is_file():
            return _UNKNOWN_OFFLINE
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.debug("Termin-Fixture fuer %s nicht lesbar: %s", symbol, exc)
            return _UNKNOWN_OFFLINE
        if not isinstance(payload, dict):
            return _UNKNOWN_OFFLINE
        if not payload.get("has_earnings", False):
            return None
        days: list[date] = []
        for raw in payload.get("dates", []):
            try:
                days.append(date.fromisoformat(str(raw)))
            except ValueError:
                _LOG.debug("Termin-Fixture fuer %s: unlesbares Datum %r", symbol, raw)
        return {EARNINGS_CALENDAR_KEY: days}

    # ----------------------------------------------- FundamentalsProvider --
    def get_statements(self, ticker: str, *, period: Period = Period.ANNUAL) -> StatementBundle:
        """Holt die drei Statements eines Symbols aus den Fixtures.

        Die Payloads behalten die Orientierung des Vertrags: Zeilen sind
        Positionen, Spalten sind Perioden. Die Zeilenbezeichner sind die
        **rohen** yfinance-Namen - die semantische Normalisierung passiert
        genau einmal in ``analytics/statements.py``, nicht hier.

        Args:
            ticker: Kanonisches Symbol.
            period: Jahres- oder Quartalsbasis.

        Returns:
            Buendel der drei Statements. Einzelne sind None, wenn kein
            Fixture vorliegt.

        Raises:
            SymbolNotFoundError: Wenn zu diesem Symbol gar kein Statement
                hinterlegt ist.
            DataValidationError: Wenn ein Fixture defekt ist.
        """
        symbol = normalize_ticker(ticker)
        suffix = _PERIOD_SUFFIX[period]
        currency = self._currency_of(symbol)
        fiscal_month = self._fiscal_year_end_month(symbol)

        statements: dict[StatementType, FinancialStatement | None] = {}
        for kind in (StatementType.INCOME, StatementType.BALANCE, StatementType.CASHFLOW):
            path = self._dir / "statements" / f"{symbol}_{kind.value}_{suffix}.parquet"
            if not path.is_file():
                statements[kind] = None
                continue
            df = _restore_period_columns(self._read_parquet(path), ticker=symbol)
            statements[kind] = FinancialStatement(
                ticker=symbol,
                statement_type=kind,
                period=period,
                currency=currency,
                fiscal_year_end_month=fiscal_month,
                data=df,
                unmapped_line_items=[],
                provenance=self._provenance("Frozen statement from tests/fixtures."),
            )

        if all(s is None for s in statements.values()):
            msg = f"No financial statement data available offline for {symbol}. {_OFFLINE_HINT}"
            raise SymbolNotFoundError(msg, detail=f"period={period.value}")

        return StatementBundle(
            ticker=symbol,
            period=period,
            income=statements[StatementType.INCOME],
            balance=statements[StatementType.BALANCE],
            cashflow=statements[StatementType.CASHFLOW],
            currency=currency,
            provenance=self._provenance(),
        )

    def _fiscal_year_end_month(self, symbol: str) -> int | None:
        """Liest den Monat des Geschaeftsjahresendes aus dem Stammdaten-Fixture.

        Args:
            symbol: Kanonisches Symbol.

        Returns:
            Monat von 1 bis 12, oder None wenn das Fixture ihn nicht fuehrt.
        """
        try:
            info = self._read_info(symbol)
        except (SymbolNotFoundError, DataValidationError):
            return None
        epoch = _opt_float(info.get("lastFiscalYearEnd"))
        if epoch is None:
            return None
        return int(pd.Timestamp(epoch, unit="s", tz="UTC").month)

    # --------------------------------------------------- ScreenerProvider --
    def screen(self, request: ScreenRequest) -> ScreenResult:
        """Liefert das eingefrorene Screener-Ergebnis.

        Die Finviz-Filter der Anfrage werden auf Gueltigkeit geprueft, aber
        **nicht** ausgewertet - das Fixture ist das Ergebnis genau eines
        eingefrorenen Screens. Nur ``limit`` wird angewandt, damit die
        Kuerzungslogik von ``ScreenResult.is_truncated`` testbar bleibt.

        Args:
            request: Screening-Anfrage.

        Returns:
            Das eingefrorene Ergebnis, auf ``request.limit`` gekuerzt.

        Raises:
            UnknownFilterError: Wenn ein Filter-Key unbekannt ist.
            DataValidationError: Wenn das Fixture defekt ist.
        """
        self._validate_filters(request.finviz_filters)

        path = self._dir / "screener" / "overview.parquet"
        if not path.is_file():
            _LOG.warning("Kein Screener-Fixture unter %s", path)
            df = pd.DataFrame(index=pd.Index([], name="ticker"))
        else:
            df = self._read_parquet(path)

        total = len(df)
        return ScreenResult(
            request=request,
            data=df.head(request.limit),
            total_matches=total,
            provenance=self._provenance("Frozen screener result from tests/fixtures."),
        )

    def available_filters(self) -> dict[str, list[str]]:
        """Listet die unterstuetzten Filter.

        Liest sie aus dem installierten ``finvizfinance`` aus, falls
        vorhanden - so ist die Validierung offline dieselbe wie online. Fehlt
        das Paket, ist das Ergebnis leer und :meth:`screen` validiert nicht.

        Returns:
            Abbildung Filter-Key auf die Liste zulaessiger Werte.
        """
        try:
            from finvizfinance.constants import filter_dict
        except ImportError:  # pragma: no cover - Pflichtabhaengigkeit
            return {}
        return {key: list(spec["option"]) for key, spec in filter_dict.items()}

    def _validate_filters(self, filters: dict[str, str]) -> None:
        """Prueft Filter-Keys gegen die verfuegbare Filterliste.

        Args:
            filters: Filter als Key-Value-Paare.

        Raises:
            UnknownFilterError: Wenn ein Key unbekannt ist.
        """
        known = self.available_filters()
        if not known:
            return
        for key in filters:
            if key not in known:
                msg = f"Filter {key!r} is unknown to Finviz."
                raise UnknownFilterError(msg, detail="Keys sind exakt zu schreiben, mit Punkt.")


def _restore_period_columns(df: pd.DataFrame, *, ticker: str) -> pd.DataFrame:
    """Wandelt die ISO-Spaltennamen eines Statement-Fixture zurueck in Perioden.

    parquet verlangt String-Spaltennamen. ``scripts/refresh_fixtures.py``
    schreibt die Perioden deshalb im ISO-Format; hier entsteht daraus wieder
    ein tz-aware UTC-``DatetimeIndex``, wie der Vertrag ihn verlangt.

    Args:
        df: Rohframe aus dem Fixture.
        ticker: Symbol, fuer Fehlermeldungen.

    Returns:
        Frame mit tz-aware UTC-Spaltenachse, aufsteigend sortiert.

    Raises:
        DataValidationError: Wenn die Spalten keine Perioden tragen.
    """
    if df.empty:
        return df
    try:
        cols = pd.DatetimeIndex([pd.Timestamp(c) for c in df.columns])
    except (TypeError, ValueError) as exc:
        msg = f"The statement fixture for {ticker} has no period columns."
        raise DataValidationError(msg, detail=f"columns={list(df.columns)}: {exc}") from exc

    out = df.copy()
    out.columns = cols.tz_localize("UTC") if cols.tz is None else cols.tz_convert("UTC")
    out.index.name = "line_item"
    return out.sort_index(axis=1)


def _opt_float(raw: object) -> float | None:
    """Konvertiert einen Rohwert vorsichtig nach float.

    Args:
        raw: Rohwert aus dem Fixture.

    Returns:
        Der Wert als float, oder None wenn er fehlt, NaN oder nicht
        konvertierbar ist. Nie ``0.0`` als Ersatz fuer einen fehlenden Wert.
    """
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN ist nicht gleich sich selbst
        return None
    return value


def _positive(raw: object) -> float | None:
    """Wie :func:`_opt_float`, verwirft aber Werte kleiner oder gleich 0.

    Args:
        raw: Rohwert aus dem Fixture.

    Returns:
        Positiver float, oder None.
    """
    value = _opt_float(raw)
    if value is None or value <= 0.0:
        return None
    return value


def _unique(tickers: Sequence[str]) -> list[str]:
    """Normalisiert eine Symbolliste und entfernt Duplikate stabil.

    Args:
        tickers: Rohe Symbolliste.

    Returns:
        Kanonische Symbole in urspruenglicher Reihenfolge, ohne Duplikate.
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw in tickers:
        symbol = normalize_ticker(raw)
        if symbol not in seen:
            seen.add(symbol)
            out.append(symbol)
    return out
