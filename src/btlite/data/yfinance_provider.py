"""Yahoo-Finance-Provider auf Basis von ``yfinance`` 1.7.

Erfuellt :class:`~btlite.data.base.MarketDataProvider`, also
:class:`~btlite.data.base.InstrumentProvider`,
:class:`~btlite.data.base.PriceProvider` und
:class:`~btlite.data.base.FundamentalsProvider`.

Gepruefte Eigenschaften der installierten Version (siehe docs/DATA_SOURCES.md):

* ``auto_adjust=True`` ist Default. Es gibt **keine** ``Adj Close``-Spalte;
  ``Close`` ist bereits split- und dividendenbereinigt.
* Statements kommen **transponiert** (Zeilen = Positionen, Spalten =
  Perioden). Genau diese Orientierung verlangt auch der Vertrag
  :class:`~btlite.models.fundamentals.FinancialStatement`, es wird also
  nicht gedreht.
* ``yf.download`` startet je Symbol einen eigenen Request (Thread-Pool). Fuer
  :meth:`YFinanceProvider.get_quotes` ist das ungeeignet. Der Batch-Quote
  laeuft deshalb ueber ``/v7/finance/quote`` mit kommaseparierter
  ``symbols``-Liste - **ein** HTTP-Request fuer alle Symbole.
* ``yfinance.Search`` kapselt ``/v1/finance/search``, holt die Daten aber
  bereits im Konstruktor. Ein Token-Bucket liesse sich davor nicht mehr
  setzen. :meth:`YFinanceProvider.search` spricht den Endpunkt deshalb direkt
  ueber ``YfData`` an - dieselbe Session, dieselbe Impersonation.

Rohe yfinance-Exceptions verlassen dieses Modul nie. Sie werden auf
:class:`~btlite.core.errors.SymbolNotFoundError`,
:class:`~btlite.core.errors.ProviderUnavailableError`,
:class:`~btlite.core.errors.RateLimitError` und
:class:`~btlite.core.errors.DataValidationError` abgebildet.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any, Final

import pandas as pd

from btlite.config import Settings, get_settings
from btlite.core.cache import DataFrameCache, cache_key, redirect_yfinance_cache
from btlite.core.clock import ensure_utc, utc_now, utc_today
from btlite.core.errors import (
    DataValidationError,
    ProviderUnavailableError,
    RateLimitError,
    SymbolNotFoundError,
)
from btlite.core.logging import get_logger
from btlite.core.ratelimit import TokenBucket, get_registry, retry_with_backoff
from btlite.data.search import hits_from_quotes, normalize_query, prepare_results
from btlite.models.common import (
    INTRADAY_INTERVALS,
    OHLCV_COLUMNS,
    OHLCV_OPTIONAL_COLUMNS,
    AssetClass,
    Currency,
    DataSource,
    Interval,
    Period,
    Provenance,
    normalize_ticker,
)
from btlite.models.composition import FundComposition, valid_isin
from btlite.models.fundamentals import FinancialStatement, StatementBundle, StatementType
from btlite.models.market import EarningsDate, Instrument, PriceSeries, Quote, SearchHit

_LOG = get_logger(__name__)

#: Batch-Quote-Endpunkt. Nimmt eine kommaseparierte ``symbols``-Liste
#: entgegen und liefert alle Symbole in genau einer Antwort.
_QUOTE_URL: Final[str] = "https://query1.finance.yahoo.com/v7/finance/quote"

#: Symbolsuche nach Name oder Kuerzel. Derselbe Endpunkt, den ``yfinance.Search``
#: intern anspricht - hier direkt ueber ``YfData``, siehe :meth:`YFinanceProvider.search`.
_SEARCH_URL: Final[str] = "https://query2.finance.yahoo.com/v1/finance/search"

#: Obergrenze der von der Quelle angeforderten Rohtreffer. Bewusst groesser
#: als das ``limit`` des Aufrufers: erst nach dem Zusammenfassen der
#: Boersen-Dubletten steht fest, wie viele Emittenten uebrig bleiben.
_SEARCH_FETCH_MAX: Final[int] = 25

#: Schluessel des Termin-Eintrags im ``Ticker.calendar``-Dictionary.
#:
#: Fehlt dieser Schluessel ganz, fuehrt das Instrument keine Quartalszahlen -
#: siehe :func:`next_earnings_from_calendar`.
EARNINGS_CALENDAR_KEY: Final[str] = "Earnings Date"

#: TTL des Termin-Caches in Sekunden, 24 Stunden. Ein Quartalstermin bewegt
#: sich selten; ihn oefter abzufragen belastet die Quelle ohne Nutzen.
#:
#: Die TTL steht in ``settings.cache.earnings_ttl_s``. Ein Quartalstermin
#: bewegt sich selten; zwischengespeichert wird der Rohbefund, damit ein
#: ueber Nacht alter Eintrag keinen abgelaufenen Termin als "naechsten"
#: ausgibt - der Vergangenheitsabgleich laeuft bei jedem Lesen neu.

#: Abbildung yfinance-Spaltenname auf die Konstanten aus ``models/common.py``.
_COLUMN_MAP: Final[dict[str, str]] = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "dividends": "dividends",
    "stock splits": "stock_splits",
    "stock_splits": "stock_splits",
    "capital gains": "capital_gains",
}

#: Abbildung ``Period`` auf den ``freq``-String der yfinance-Getter.
_FREQ_MAP: Final[dict[Period, str]] = {
    Period.ANNUAL: "yearly",
    Period.QUARTERLY: "quarterly",
    Period.TTM: "trailing",
}

#: Abbildung yfinance-``quoteType`` auf die Instrumentenklasse des Vertrags.
_ASSET_CLASS_MAP: Final[dict[str, AssetClass]] = {
    "EQUITY": AssetClass.EQUITY,
    "ETF": AssetClass.ETF,
    "INDEX": AssetClass.INDEX,
    "CRYPTOCURRENCY": AssetClass.CRYPTO,
    "MUTUALFUND": AssetClass.FUND,
}


def to_currency(raw: object) -> Currency:
    """Bildet einen Waehrungscode der Quelle auf :class:`Currency` ab.

    Args:
        raw: Rohwert der Quelle, etwa ``"USD"``. Andere Typen und unbekannte
            Codes fallen auf :attr:`Currency.OTHER` zurueck.

    Returns:
        Die passende Waehrung, sonst :attr:`Currency.OTHER`.
    """
    if not isinstance(raw, str):
        return Currency.OTHER
    try:
        return Currency(raw.strip().upper())
    except ValueError:
        return Currency.OTHER


def _as_float(raw: object) -> float | None:
    """Konvertiert einen Rohwert vorsichtig nach float.

    Fehlende und unbrauchbare Werte werden zu ``None``, nie zu ``0.0``. Das
    ist eine harte Projektregel: ein still zu Null gewordener Wert verfaelscht
    jeden Vergleich, der ihn mittelt.

    Args:
        raw: Rohwert der Quelle.

    Returns:
        Der Wert als float, oder None wenn er fehlt, NaN oder nicht
        konvertierbar ist.
    """
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    # NaN ist nicht gleich sich selbst - der Vergleich haelt NaN heraus,
    # ohne dass mypy den Zweig als unerreichbar einstuft.
    if value != value:
        return None
    return value


def _positive_float(raw: object) -> float | None:
    """Wie :func:`_as_float`, verwirft aber Werte kleiner oder gleich 0.

    Der Vertrag verlangt fuer Preisfelder ``gt=0``. Ein von der Quelle
    geliefertes 0.0 ist kein Preis, sondern ein fehlender Wert.

    Args:
        raw: Rohwert der Quelle.

    Returns:
        Positiver float, oder None.
    """
    value = _as_float(raw)
    if value is None or value <= 0.0:
        return None
    return value


def next_earnings_from_calendar(
    calendar: object,
    *,
    ticker: str,
    today: date | None = None,
    provenance: Provenance | None = None,
) -> EarningsDate:
    """Leitet den naechsten Termin aus einer ``Ticker.calendar``-Antwort ab.

    Rein rechnend, ohne Netz - deshalb teilen sich Live-Provider, Fixtures und
    das Refresh-Skript genau diese Auswertung. Gegen yfinance 1.7 am
    24.09.2026 an 78 Symbolen gemessen (siehe docs/DATA_SOURCES.md):

    * Aktie mit Termin: ``{"Earnings Date": [date(2026, 10, 29)], ...}``.
    * ETF, Krypto, Index, unbekanntes Symbol: ``{}``. yfinance faengt den
      HTTP 404 der Quelle intern ab; der Schluessel fehlt dann **ganz**.
      Das ist die richtige Antwort, kein Fehler - ein ETF hat keine
      Quartalszahlen.
    * Aktie ohne angekuendigten Termin: ``{"Earnings Date": []}``.

    Genau dieser Unterschied traegt :attr:`EarningsDate.has_earnings`: ein
    fehlender Schluessel heisst "fuehrt keine Quartalszahlen", eine leere
    Liste heisst "Aktie, Termin unbekannt". Der Negativbefund gilt nur, weil
    die Quelle **geantwortet** hat. Blosses Nichtwissen - kein Fixture, kein
    geglueckter Abruf - darf hier nicht hineingereicht werden; dafuer gilt
    ``has_earnings=True``.

    Liegt der juengste gelieferte Termin in der Vergangenheit, ist das
    Ergebnis ``next_date=None``. Ein abgelaufener Termin ist kein *naechster*;
    ihn auszugeben waere eine Falschaussage.

    Args:
        calendar: Rohantwort der Quelle. Alles, was kein Dictionary ist -
            auch ``None`` - gilt als "fuehrt keine Quartalszahlen".
        ticker: Kanonisches Symbol fuer das Ergebnis.
        today: Stichtag, ab dem ein Termin als zukuenftig gilt. ``None``
            nimmt den heutigen UTC-Tag. Ein Termin **am** Stichtag zaehlt
            noch als naechster.
        provenance: Herkunftsnachweis fuer das Ergebnis.

    Returns:
        Der Termin als Vertrag. ``next_date`` ist ein Kalendertag oder ``None``.
    """
    if not isinstance(calendar, dict) or EARNINGS_CALENDAR_KEY not in calendar:
        # Belegter Negativbefund: yfinance hat geantwortet, aber die Quelle
        # fuehrt zu diesem Instrument keine Fundamentaldaten (HTTP 404).
        return EarningsDate(
            ticker=ticker, next_date=None, has_earnings=False, provenance=provenance
        )

    raw = calendar[EARNINGS_CALENDAR_KEY]
    candidates = [d for d in _as_dates(raw) if d >= (today if today is not None else utc_today())]
    # Yahoo fuehrt laut Doku bis zu zwei Eintraege als unbestaetigtes
    # Zeitfenster. In 78 gemessenen Symbolen kam nie mehr als einer vor;
    # trifft der Fall doch ein, gilt der **frueheste** kuenftige Termin -
    # das ist der naechste, und ein spaeterer waere zu optimistisch.
    return EarningsDate(
        ticker=ticker,
        next_date=min(candidates) if candidates else None,
        has_earnings=True,
        provenance=provenance,
    )


def _as_dates(raw: object) -> list[date]:
    """Filtert aus einer Rohliste die verwertbaren Kalendertage.

    Args:
        raw: Rohwert unter ``"Earnings Date"``. Erwartet wird eine Liste von
            ``datetime.date``; alles andere wird verworfen statt geraten.

    Returns:
        Die enthaltenen Kalendertage. ``datetime`` wird auf seinen Datumsteil
        reduziert, sonstige Typen werden uebergangen.
    """
    if not isinstance(raw, list | tuple):
        return []
    out: list[date] = []
    for item in raw:
        if isinstance(item, datetime):
            out.append(item.date())
        elif isinstance(item, date):
            out.append(item)
    return out


def _map_exception(exc: Exception, ticker: str) -> Exception:
    """Bildet eine rohe yfinance-Exception auf die Projekt-Hierarchie ab.

    Args:
        exc: Die von yfinance geworfene Exception.
        ticker: Betroffenes Symbol, fuer die Fehlermeldung.

    Returns:
        Der passende Fehler aus :mod:`btlite.core.errors`. Bereits
        abgebildete Fehler werden unveraendert durchgereicht.
    """
    from btlite.core.errors import BTLiteError

    if isinstance(exc, BTLiteError):
        return exc

    from yfinance.exceptions import (
        YFInvalidPeriodError,
        YFPricesMissingError,
        YFRateLimitError,
        YFTickerMissingError,
        YFTzMissingError,
    )

    if isinstance(exc, YFRateLimitError):
        return RateLimitError(
            "Yahoo Finance is currently throttling requests.",
            detail=f"ticker={ticker}: {exc}",
            retry_after_s=None,
        )
    if isinstance(exc, YFTzMissingError | YFPricesMissingError | YFTickerMissingError):
        return SymbolNotFoundError(
            f"Symbol {ticker} is not available on Yahoo Finance.",
            detail=f"{type(exc).__name__}: {exc}",
        )
    if isinstance(exc, YFInvalidPeriodError):
        return DataValidationError(
            f"The requested period is not allowed for {ticker}.",
            detail=f"{type(exc).__name__}: {exc}",
        )
    return ProviderUnavailableError(
        "Yahoo Finance is currently unreachable.",
        detail=f"ticker={ticker}, {type(exc).__name__}: {exc}",
    )


def normalize_ohlcv(df: pd.DataFrame, *, ticker: str) -> pd.DataFrame:
    """Normalisiert einen yfinance-Kursframe auf den Projektvertrag.

    Die Normalisierung umfasst:

    * Spaltennamen auf :data:`~btlite.models.common.OHLCV_COLUMNS`
      (kleingeschrieben) abbilden, MultiIndex-Spalten aufloesen.
    * Index auf tz-aware UTC setzen, Name ``date``.
    * aufsteigend sortieren und Index-Duplikate entfernen (letzter Wert
      gewinnt, weil Yahoo Historien rueckwirkend korrigiert).
    * Zeilen ohne ``close`` verwerfen - eine Kerze ohne Schlusskurs traegt
      keine Information.

    Args:
        df: Rohframe aus ``Ticker.history`` oder ``yf.download``.
        ticker: Symbol, fuer Fehlermeldungen.

    Returns:
        Normalisierter Frame mit den Pflichtspalten und UTC-Index.

    Raises:
        DataValidationError: Wenn Pflichtspalten fehlen oder der Index nicht
            in einen ``DatetimeIndex`` konvertierbar ist.
    """
    if df is None or df.empty:
        empty = pd.DataFrame(
            {c: pd.Series(dtype="float64") for c in OHLCV_COLUMNS},
            index=pd.DatetimeIndex([], tz="UTC", name="date"),
        )
        return empty

    out = df.copy()

    # yf.download liefert bei mehreren Tickern MultiIndex-Spalten. Ist der
    # Frame bereits auf ein Symbol reduziert, bleibt eine ueberfluessige
    # Ebene stehen - die faellt hier weg.
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    renamed: dict[Any, str] = {}
    for col in out.columns:
        key = str(col).strip().lower()
        renamed[col] = _COLUMN_MAP.get(key, key.replace(" ", "_"))
    out = out.rename(columns=renamed)

    # Doppelte Spalten (etwa "Close" und "close" nebeneinander) waeren im
    # Payload nicht eindeutig adressierbar.
    out = out.loc[:, ~out.columns.duplicated(keep="first")]

    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        msg = f"Price data for {ticker} is incomplete"
        raise DataValidationError(
            msg,
            detail=f"fehlende Spalten={missing}, vorhanden={list(out.columns)}",
        )

    keep = [*OHLCV_COLUMNS, *(c for c in OHLCV_OPTIONAL_COLUMNS if c in out.columns)]
    out = out[keep]

    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.DatetimeIndex(out.index)
        except (TypeError, ValueError) as exc:
            msg = f"The price index for {ticker} is not a date index"
            raise DataValidationError(msg, detail=str(exc)) from exc

    idx: pd.DatetimeIndex = out.index
    out.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out.index.name = "date"

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.dropna(subset=["close"])

    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def normalize_statement(df: pd.DataFrame, *, ticker: str) -> pd.DataFrame:
    """Normalisiert einen yfinance-Statementframe auf den Projektvertrag.

    yfinance liefert bereits die vom Vertrag gewuenschte Orientierung
    (Zeilen = Positionen, Spalten = Perioden). Es wird also **nicht**
    transponiert, sondern nur die Spaltenachse auf tz-aware UTC gebracht,
    aufsteigend sortiert und von Duplikaten befreit.

    Die semantische Normalisierung der Zeilenbezeichner (``Total Revenue``
    nach ``revenue``) passiert bewusst **nicht** hier, sondern genau einmal
    in ``analytics/statements.py``.

    Args:
        df: Rohframe aus ``Ticker.get_income_stmt`` und Geschwistern.
        ticker: Symbol, fuer Fehlermeldungen.

    Returns:
        Normalisierter Frame. Werte sind absolut in Berichtswaehrung.

    Raises:
        DataValidationError: Wenn die Spaltenachse keine Perioden traegt.
    """
    if df is None or df.empty:
        return pd.DataFrame(index=pd.Index([], name="line_item"))

    out = df.copy()
    out.index = pd.Index([str(i) for i in out.index], name="line_item")

    try:
        cols = pd.DatetimeIndex(out.columns)
    except (TypeError, ValueError) as exc:
        msg = f"The statement columns for {ticker} are not periods"
        raise DataValidationError(msg, detail=f"columns={list(out.columns)}: {exc}") from exc

    out.columns = cols.tz_localize("UTC") if cols.tz is None else cols.tz_convert("UTC")
    out = out.loc[:, ~out.columns.duplicated(keep="last")]
    out = out.sort_index(axis=1)
    out = out[~out.index.duplicated(keep="first")]
    return out.astype("float64", errors="ignore")


class YFinanceProvider:
    """Marktdaten aus Yahoo Finance.

    Erfuellt :class:`~btlite.data.base.MarketDataProvider` strukturell, ohne
    von den Protocols zu erben.

    Alle ausgehenden Requests laufen durch einen Token-Bucket
    (Default 2 Requests pro Sekunde, Burst 5). Ein von der Quelle
    signalisiertes Rate Limit wird **nicht** umgangen, sondern als
    :class:`~btlite.core.errors.RateLimitError` nach oben gereicht.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        cache: DataFrameCache | None = None,
        bucket: TokenBucket | None = None,
    ) -> None:
        """Initialisiert den Provider.

        Args:
            settings: Projekt-Settings. ``None`` nimmt die globalen.
            cache: DataFrame-Cache. ``None`` baut einen aus den Settings.
            bucket: Token-Bucket. ``None`` nimmt den prozessweiten
                yfinance-Bucket aus der Registry.

        Raises:
            ProviderUnavailableError: Wenn der Offline-Modus aktiv ist. Dann
                ist :class:`~btlite.data.fixtures.FixtureProvider` zu nutzen.
        """
        self._settings = settings if settings is not None else get_settings()
        if self._settings.data.offline:
            msg = "Offline mode: yfinance is blocked, use the FixtureProvider."
            raise ProviderUnavailableError(msg, detail="BTLITE_DATA_OFFLINE=true")

        self._cache = cache if cache is not None else DataFrameCache(self._settings.cache)
        self._bucket = bucket if bucket is not None else get_registry().bucket("yfinance")
        self._retry = retry_with_backoff(settings=self._settings.ratelimit)

    # ------------------------------------------------------------ intern --
    def _provenance(self, *, from_cache: bool, note: str | None = None) -> Provenance:
        """Baut den Herkunftsnachweis fuer eine Antwort dieses Providers.

        Args:
            from_cache: Ob die Antwort aus dem lokalen Cache kam.
            note: Optionaler Freitext-Hinweis fuer das UI.

        Returns:
            Gefuellte :class:`~btlite.models.common.Provenance`. ``delay_minutes``
            stammt aus ``DataSettings.quote_delay_notice_min``.
        """
        return Provenance(
            source=DataSource.YFINANCE,
            fetched_at=utc_now(),
            from_cache=from_cache,
            delay_minutes=self._settings.data.quote_delay_notice_min,
            note=note,
        )

    def _ticker(self, symbol: str) -> Any:
        """Baut ein ``yfinance.Ticker``-Objekt und entnimmt einen Token.

        Args:
            symbol: Kanonisches Symbol.

        Returns:
            Das ``yfinance.Ticker``-Objekt.
        """
        import yfinance as yf

        redirect_yfinance_cache(self._settings.cache)
        self._bucket.acquire()
        return yf.Ticker(symbol)

    def _raw_info(self, symbol: str) -> dict[str, Any]:
        """Holt ``Ticker.info`` und bildet Fehler auf die Projekthierarchie ab.

        Gepuffert fuer ``settings.cache.info_ttl_s`` (Default 4 h). Ohne
        Puffer lud jedes Screening die Stammdaten jedes Titels zweimal neu
        (Stammdaten und Fiskaljahresende): gemessen rund eine Sekunde je
        Titel, bei jedem Lauf - ein Folgelauf ueber den S&P 500 war damit
        so langsam wie der erste. Gepuffert wird nur ein Erfolg; Fehler
        und Rate Limits gehen wie bisher nach oben.

        Args:
            symbol: Kanonisches Symbol.

        Returns:
            Das Roh-Dictionary der Quelle.

        Raises:
            SymbolNotFoundError: Symbol unbekannt oder ohne Stammdaten.
            ProviderUnavailableError: Quelle nicht erreichbar.
            RateLimitError: Quelle hat gedrosselt.
        """

        def _fetch() -> dict[str, Any]:
            try:
                info = self._ticker(symbol).info
            except Exception as exc:
                raise _map_exception(exc, symbol) from exc
            if not isinstance(info, dict) or not info:
                msg = f"Symbol {symbol} returns no master data."
                raise SymbolNotFoundError(msg, detail="Ticker.info ist leer")
            return info

        key = cache_key("yf", "info", symbol)
        cached = self._cache.get_df(key)
        if cached is not None and not cached.empty:
            try:
                restored = json.loads(str(cached.iloc[0, 0]))
            except (TypeError, ValueError):
                restored = None
            if isinstance(restored, dict) and restored:
                return restored

        info = self._retry(_fetch)()
        # Als JSON in einer Zelle: der Cache speichert DataFrames, und
        # ``Ticker.info`` ist ein verschachteltes Dictionary. ``default=str``
        # faengt die seltenen Werte ab, die JSON nicht kennt.
        self._cache.set_df(
            key,
            pd.DataFrame({"info": [json.dumps(info, default=str)]}),
            self._settings.cache.info_ttl_s,
        )
        return info

    # ------------------------------------------------- InstrumentProvider --
    def get_instrument(self, ticker: str) -> Instrument:
        """Holt die Stammdaten eines Symbols.

        Args:
            ticker: Kanonisches Symbol.

        Returns:
            Validierte Stammdaten. Nicht gelieferte Felder sind ``None``.

        Raises:
            SymbolNotFoundError: Symbol existiert bei der Quelle nicht.
            ProviderUnavailableError: Quelle nicht erreichbar.
            RateLimitError: Quelle hat gedrosselt.
        """
        symbol = normalize_ticker(ticker)
        info = self._raw_info(symbol)
        return self._instrument_from_info(symbol, info, from_cache=False)

    def _instrument_from_info(
        self, symbol: str, info: dict[str, Any], *, from_cache: bool
    ) -> Instrument:
        """Baut ein :class:`Instrument` aus einem yfinance-``info``-Dictionary.

        Args:
            symbol: Kanonisches Symbol.
            info: Roh-Stammdaten der Quelle.
            from_cache: Ob die Rohdaten aus dem Cache stammen.

        Returns:
            Validierte Stammdaten.
        """
        quote_type = str(info.get("quoteType", "")).upper()
        employees = _as_float(info.get("fullTimeEmployees"))
        return Instrument(
            ticker=symbol,
            name=info.get("longName") or info.get("shortName"),
            asset_class=_ASSET_CLASS_MAP.get(quote_type, AssetClass.OTHER),
            currency=to_currency(info.get("currency")),
            exchange=info.get("exchange"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            country=info.get("country"),
            website=info.get("website"),
            description=info.get("longBusinessSummary"),
            employees=None if employees is None else int(employees),
            market_cap=_as_float(info.get("marketCap")),
            shares_outstanding=_as_float(info.get("sharesOutstanding")),
            dividend_rate=_as_float(info.get("trailingAnnualDividendRate")),
            dividend_yield=_as_float(info.get("trailingAnnualDividendYield")),
            provenance=self._provenance(from_cache=from_cache),
        )

    # ------------------------------------------------------- SymbolSearch --
    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        """Sucht Instrumente nach Name oder Kuerzel.

        Geht ueber ``/v1/finance/search``, denselben Endpunkt, den auch
        ``yfinance.Search`` anspricht - aber direkt ueber dessen ``YfData``,
        damit Impersonation und Session-Handling von yfinance greifen. Die
        Klasse ``yfinance.Search`` selbst scheidet aus: sie holt die Daten
        bereits im Konstruktor, womit sich der Token-Bucket nicht mehr
        davorschalten laesst, und sie fordert nebenbei News, Listen und
        Empfehlungen an, die hier niemand braucht.

        Das Ranking uebernimmt anschliessend ``data/search.py``, nicht die
        Quelle: Yahoo bewertet nach eigenem Score, ein exakter Kuerzel-Treffer
        muss aber immer oben stehen.

        Args:
            query: Suchbegriff, Name oder Kuerzel. Gross- und Kleinschreibung
                spielt keine Rolle.
            limit: Maximale Trefferzahl. Werte kleiner 1 liefern eine leere
                Liste.

        Returns:
            Treffer, absteigend nach Relevanz, ueber Boersen hinweg
            zusammengefasst. Hoechstens ``limit`` Eintraege.

        Raises:
            ProviderUnavailableError: Quelle nicht erreichbar.
            RateLimitError: Quelle hat gedrosselt.
        """
        if limit < 1 or not normalize_query(query):
            return []

        raw_quotes = self._fetch_search(query.strip())
        return prepare_results(hits_from_quotes(raw_quotes), query, limit=limit)

    def _fetch_search(self, query: str) -> list[Any]:
        """Holt die Rohtreffer der Symbolsuche in genau einem Request.

        ``enableFuzzyQuery`` bleibt aus: Yahoos Fuzzy-Modus bringt viel
        Rauschen in die Antwort. Tippfehler faengt stattdessen das lokale
        ``difflib``-Ranking in ``data/search.py`` ab - das ist ohne Netz
        testbar und kostet keinen zusaetzlichen Request.

        Args:
            query: Bereits getrimmter Suchbegriff.

        Returns:
            Rohliste aus dem Feld ``quotes`` der Antwort. Leer, wenn die
            Quelle nichts findet.

        Raises:
            ProviderUnavailableError: Quelle nicht erreichbar.
            RateLimitError: Quelle hat gedrosselt.
            DataValidationError: Antwort hat nicht die erwartete Form.
        """

        def _fetch() -> list[Any]:
            from yfinance.data import YfData

            self._bucket.acquire()
            params = {
                "q": query,
                "quotesCount": _SEARCH_FETCH_MAX,
                "newsCount": 0,
                "listsCount": 0,
                "recommendedCount": 0,
                "enableFuzzyQuery": False,
                "quotesQueryId": "tss_match_phrase_query",
                "enableCb": False,
                "enableNavLinks": False,
                "enableResearchReports": False,
                "enableCulturalAssets": False,
            }
            try:
                payload = YfData().get_raw_json(
                    _SEARCH_URL,
                    params=params,
                    timeout=self._settings.data.request_timeout_s,
                )
            except Exception as exc:
                raise _map_exception(exc, query) from exc

            if not isinstance(payload, dict):
                msg = "Unexpected response from the symbol search."
                raise DataValidationError(msg, detail=f"typ={type(payload).__name__}")
            quotes = payload.get("quotes")
            if not isinstance(quotes, list):
                return []
            return quotes

        return self._retry(_fetch)()

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
        """Holt die OHLCV-Historie eines Symbols.

        Entweder ``period`` oder ``start``/``end`` setzen, nicht beides.

        Args:
            ticker: Kanonisches Symbol.
            interval: Kerzen-Intervall.
            period: Relativer Zeitraum, etwa ``"5y"``.
            start: Startzeitpunkt, tz-aware UTC.
            end: Endzeitpunkt, tz-aware UTC.
            auto_adjust: Split- und dividendenbereinigte Kurse anfordern.
            include_currency: Handelswaehrung separat aufloesen. Bei False
                wird ``Currency.OTHER`` gesetzt; Kursrechnungen bleiben
                unveraendert und sparen einen Netzwerkabruf.
                Bei True gibt es keine ``Adj Close``-Spalte, ``close`` ist
                bereits bereinigt.

        Returns:
            Kursreihe mit UTC-Index, aufsteigend, ohne Index-Duplikate.

        Raises:
            SymbolNotFoundError: Symbol existiert bei der Quelle nicht.
            ProviderUnavailableError: Quelle nicht erreichbar.
            RateLimitError: Quelle hat gedrosselt.
            DataValidationError: Antwort passt nicht auf den Vertrag.
        """
        symbol = normalize_ticker(ticker)
        use_period = period if (start is None and end is None) else None
        key = cache_key(
            "yf",
            "history",
            symbol,
            interval.value,
            use_period,
            None if start is None else ensure_utc(start).isoformat(),
            None if end is None else ensure_utc(end).isoformat(),
            auto_adjust,
        )

        def _load() -> pd.DataFrame:
            def _fetch() -> pd.DataFrame:
                try:
                    raw = self._ticker(symbol).history(
                        period=use_period,
                        interval=interval.value,
                        start=None if start is None else ensure_utc(start),
                        end=None if end is None else ensure_utc(end),
                        auto_adjust=auto_adjust,
                        actions=True,
                        raise_errors=True,
                        timeout=self._settings.data.request_timeout_s,
                    )
                except Exception as exc:
                    raise _map_exception(exc, symbol) from exc
                return normalize_ohlcv(raw, ticker=symbol)

            return self._retry(_fetch)()

        # Intraday-Kerzen veralten im Minutentakt; eine Stunde Cache hiesse,
        # dass ein 1-Minuten-Chart eine Stunde lang stillsteht.
        ttl = (
            self._settings.cache.intraday_ttl_s
            if interval in INTRADAY_INTERVALS
            else self._settings.cache.prices_ttl_s
        )
        df, from_cache = self._cache.cached_df(key, ttl, _load)
        if df.empty:
            msg = f"Yahoo Finance returns no price data for {symbol}."
            raise SymbolNotFoundError(msg, detail=f"interval={interval.value}, period={use_period}")

        return PriceSeries(
            ticker=symbol,
            interval=interval,
            currency=self._currency_of(symbol) if include_currency else Currency.OTHER,
            adjusted=auto_adjust,
            data=df,
            provenance=self._provenance(from_cache=from_cache),
        )

    def _currency_of(self, symbol: str) -> Currency:
        """Ermittelt die Handelswaehrung eines Symbols.

        Ein fehlgeschlagener Abruf ist hier kein Fehler: die Kursreihe selbst
        liegt bereits vor, nur die Waehrungsangabe fehlt. Dann gilt der
        Default USD.

        Args:
            symbol: Kanonisches Symbol.

        Returns:
            Die Waehrung laut Quelle, sonst :attr:`Currency.USD`.
        """
        key = cache_key("yf", "currency", symbol)
        cached = self._cache.get_df(key)
        if cached is not None and not cached.empty:
            return to_currency(cached.iloc[0, 0])
        try:
            import yfinance as yf

            redirect_yfinance_cache(self._settings.cache)
            self._bucket.acquire()
            raw = yf.Ticker(symbol).fast_info["currency"]
        except Exception as exc:  # Waehrung ist optional, kein Abbruchgrund
            _LOG.debug("Waehrung fuer %s nicht ermittelbar: %s", symbol, exc)
            return Currency.USD
        currency = to_currency(raw)
        self._cache.set_df(
            key,
            pd.DataFrame({"currency": [currency.value]}),
            self._settings.cache.statements_ttl_s,
        )
        return currency

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
        """Holt Historien fuer mehrere Symbole.

        Symbole, die die Quelle nicht kennt, fehlen im Ergebnis, statt den
        gesamten Aufruf scheitern zu lassen. Ein Rate Limit dagegen bricht
        sofort ab - weiterzumachen wuerde die Drosselung nur verschaerfen.

        Args:
            tickers: Kanonische Symbole. Duplikate werden entfernt.
            interval: Kerzen-Intervall.
            period: Relativer Zeitraum.
            start: Startzeitpunkt, tz-aware UTC.
            end: Endzeitpunkt, tz-aware UTC.
            auto_adjust: Bereinigte Kurse anfordern.

        Returns:
            Abbildung Symbol auf Kursreihe, nur fuer erfolgreiche Abrufe.

        Raises:
            ProviderUnavailableError: Quelle nicht erreichbar.
            RateLimitError: Quelle hat gedrosselt.
        """
        out: dict[str, PriceSeries] = {}
        for symbol in _unique_tickers(tickers):
            try:
                out[symbol] = self.get_history(
                    symbol,
                    interval=interval,
                    period=period,
                    start=start,
                    end=end,
                    auto_adjust=auto_adjust,
                )
            except RateLimitError:
                raise
            except (SymbolNotFoundError, DataValidationError) as exc:
                _LOG.warning("Historie fuer %s uebersprungen: %s", symbol, exc)
        return out

    def get_quotes(self, tickers: Sequence[str]) -> dict[str, Quote]:
        """Holt Momentaufnahmen fuer mehrere Symbole in **einem** Request.

        Nutzt ``/v7/finance/quote`` mit kommaseparierter ``symbols``-Liste.
        ``yf.download`` und ``fast_info`` scheiden aus: beide erzeugen einen
        Request je Symbol und laufen bei 20 Tickern im Poll-Takt ins Rate
        Limit. Ueberschreitet die Symbolzahl ``DataSettings.quote_batch_size``,
        wird in entsprechend viele Batches zerlegt.

        Args:
            tickers: Kanonische Symbole. Duplikate werden entfernt.

        Returns:
            Abbildung Symbol auf Quote, nur fuer erfolgreiche Abrufe. Symbole
            ohne verwertbaren Preis fehlen.

        Raises:
            ProviderUnavailableError: Quelle nicht erreichbar.
            RateLimitError: Quelle hat gedrosselt.
        """
        symbols = _unique_tickers(tickers)
        if not symbols:
            return {}

        batch_size = self._settings.data.quote_batch_size
        out: dict[str, Quote] = {}
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            for raw in self._fetch_quote_batch(batch):
                quote = self._quote_from_raw(raw)
                if quote is not None:
                    out[quote.ticker] = quote
        return out

    def _fetch_quote_batch(self, symbols: Sequence[str]) -> list[dict[str, Any]]:
        """Holt einen Batch Quotes in genau einem HTTP-Request.

        Args:
            symbols: Symbole dieses Batches.

        Returns:
            Liste der Roh-Quote-Dictionaries der Quelle.

        Raises:
            ProviderUnavailableError: Quelle nicht erreichbar.
            RateLimitError: Quelle hat gedrosselt.
        """

        def _fetch() -> list[dict[str, Any]]:
            from yfinance.data import YfData

            self._bucket.acquire()
            params = {
                "symbols": ",".join(symbols),
                "formatted": "false",
                "corsDomain": "finance.yahoo.com",
            }
            try:
                payload = YfData().get_raw_json(
                    _QUOTE_URL,
                    params=params,
                    timeout=self._settings.data.request_timeout_s,
                )
            except Exception as exc:
                raise _map_exception(exc, ",".join(symbols)) from exc

            if not isinstance(payload, dict):
                msg = "Unexpected response from the quote endpoint."
                raise DataValidationError(msg, detail=f"typ={type(payload).__name__}")
            result = payload.get("quoteResponse", {}).get("result")
            if result is None:
                return []
            return [r for r in result if isinstance(r, dict)]

        return self._retry(_fetch)()

    def _quote_from_raw(self, raw: dict[str, Any]) -> Quote | None:
        """Baut ein :class:`Quote` aus einem Roh-Quote der Quelle.

        Args:
            raw: Roh-Quote-Dictionary aus ``/v7/finance/quote``.

        Returns:
            Validiertes Quote, oder None wenn Symbol oder Preis fehlen. Ein
            fehlender Preis ist kein Fehler des Gesamtabrufs - das Symbol
            faellt still aus dem Ergebnis.
        """
        symbol_raw = raw.get("symbol")
        if not isinstance(symbol_raw, str) or not symbol_raw.strip():
            return None
        price = _positive_float(raw.get("regularMarketPrice"))
        if price is None:
            _LOG.debug("Quote fuer %s ohne verwertbaren Preis", symbol_raw)
            return None

        epoch = _as_float(raw.get("regularMarketTime"))
        timestamp = (
            utc_now() if epoch is None else pd.Timestamp(epoch, unit="s", tz="UTC").to_pydatetime()
        )

        return Quote(
            ticker=normalize_ticker(symbol_raw),
            price=price,
            currency=to_currency(raw.get("currency")),
            previous_close=_positive_float(raw.get("regularMarketPreviousClose")),
            open=_positive_float(raw.get("regularMarketOpen")),
            day_high=_positive_float(raw.get("regularMarketDayHigh")),
            day_low=_positive_float(raw.get("regularMarketDayLow")),
            volume=_as_float(raw.get("regularMarketVolume")),
            bid=_positive_float(raw.get("bid")),
            ask=_positive_float(raw.get("ask")),
            timestamp=timestamp,
            provenance=self._provenance(from_cache=False),
        )

    # ----------------------------------------------- FundamentalsProvider --
    # ---------------------------------------- EarningsCalendarProvider --
    def get_earnings_dates(self, tickers: Sequence[str]) -> dict[str, EarningsDate]:
        """Holt die naechsten Quartalszahlen-Termine fuer mehrere Symbole.

        yfinance bietet keinen Sammelendpunkt fuer ``calendar``; es wird je
        Symbol einzeln gefragt. Jeder Abruf entnimmt einen Token aus dem
        Bucket - genau dafuer ist er da. Antworten liegen 24 Stunden im
        Cache (``settings.cache.earnings_ttl_s``), ein Quartalstermin bewegt
        sich selten.

        Ein fehlender Termin ist **kein Fehler**: ein ETF fuehrt keine
        Quartalszahlen, und ein HTTP 404 der Quelle ist dort die richtige
        Antwort. Er darf den Sammelabruf nicht abbrechen und nicht als
        :class:`~btlite.core.errors.ProviderUnavailableError` hochschlagen.
        Auch der Ausfall eines einzelnen Symbols laesst die uebrigen intakt.

        Args:
            tickers: Kanonische Symbole. Duplikate werden entfernt.

        Returns:
            Abbildung Symbol auf Termin, Reihenfolge nicht garantiert. Nur
            Symbole, zu denen sich gar keine Aussage treffen laesst, fehlen.

        Raises:
            RateLimitError: Quelle hat gedrosselt. Wird nie umgangen - der
                Fehler beendet den Sammelabruf und geht nach oben.
        """
        out: dict[str, EarningsDate] = {}
        for symbol in _unique_tickers(tickers):
            try:
                out[symbol] = self._earnings_date(symbol)
            except RateLimitError:
                # Projektregel: Drosselung wird nicht umgangen und nicht
                # stillschweigend zu "kein Termin" verharmlost.
                raise
            except Exception as exc:  # ein Symbol kippt die anderen nicht
                _LOG.debug("Termin fuer %s nicht ermittelbar: %s", symbol, exc)
        return out

    def _earnings_date(self, symbol: str) -> EarningsDate:
        """Holt den Termin eines einzelnen Symbols, mit Cache.

        Zwischengespeichert wird der **Rohbefund** der Quelle, nicht das
        fertige Modell: der Vergangenheitsabgleich muss bei jedem Zugriff neu
        laufen, sonst gaebe ein 23 Stunden alter Cache-Eintrag nach
        Mitternacht einen bereits abgelaufenen Termin als "naechsten" aus.

        Args:
            symbol: Kanonisches Symbol.

        Returns:
            Der Termin als Vertrag.

        Raises:
            RateLimitError: Quelle hat gedrosselt.
            ProviderUnavailableError: Quelle nicht erreichbar.
        """
        key = cache_key("yf", "earnings", symbol)

        def _load() -> pd.DataFrame:
            def _fetch() -> pd.DataFrame:
                try:
                    calendar = self._ticker(symbol).calendar
                except Exception as exc:
                    raise _map_exception(exc, symbol) from exc
                return _earnings_to_frame(calendar)

            return self._retry(_fetch)()

        ttl = self._settings.cache.earnings_ttl_s
        df, from_cache = self._cache.cached_df(key, ttl, _load)
        return next_earnings_from_calendar(
            _frame_to_calendar(df),
            ticker=symbol,
            provenance=self._provenance(from_cache=from_cache),
        )

    def get_statements(self, ticker: str, *, period: Period = Period.ANNUAL) -> StatementBundle:
        """Holt Income Statement, Bilanz und Cashflow eines Symbols.

        Die Statements kommen von yfinance bereits transponiert (Zeilen =
        Positionen, Spalten = Perioden) - genau so, wie der Vertrag sie
        verlangt. Die semantische Normalisierung der Zeilenbezeichner bleibt
        ``analytics/statements.py`` vorbehalten.

        Args:
            ticker: Kanonisches Symbol.
            period: Jahres-, Quartals- oder TTM-Basis.

        Returns:
            Buendel der drei Statements. Einzelne sind None, wenn die Quelle
            sie nicht fuehrt - TTM etwa kennt keine Bilanz.

        Raises:
            SymbolNotFoundError: Symbol existiert bei der Quelle nicht.
            ProviderUnavailableError: Quelle nicht erreichbar.
            RateLimitError: Quelle hat gedrosselt.
            DataValidationError: Antwort passt nicht auf den Vertrag.
        """
        symbol = normalize_ticker(ticker)
        freq = _FREQ_MAP[period]
        currency = self._currency_of(symbol)
        fiscal_month = self._fiscal_year_end_month(symbol)

        statements: dict[StatementType, FinancialStatement | None] = {}
        for kind, getter in (
            (StatementType.INCOME, "get_income_stmt"),
            (StatementType.BALANCE, "get_balance_sheet"),
            (StatementType.CASHFLOW, "get_cash_flow"),
        ):
            df, from_cache = self._load_statement(symbol, kind, getter, freq)
            if df.empty:
                statements[kind] = None
                continue
            statements[kind] = FinancialStatement(
                ticker=symbol,
                statement_type=kind,
                period=period,
                currency=currency,
                fiscal_year_end_month=fiscal_month,
                data=df,
                unmapped_line_items=[],
                provenance=self._provenance(from_cache=from_cache),
            )

        if all(s is None for s in statements.values()):
            msg = f"Yahoo Finance returns no statements for {symbol}."
            raise SymbolNotFoundError(msg, detail=f"period={period.value}")

        return StatementBundle(
            ticker=symbol,
            period=period,
            income=statements[StatementType.INCOME],
            balance=statements[StatementType.BALANCE],
            cashflow=statements[StatementType.CASHFLOW],
            currency=currency,
            provenance=self._provenance(from_cache=False),
        )

    def _load_statement(
        self, symbol: str, kind: StatementType, getter: str, freq: str
    ) -> tuple[pd.DataFrame, bool]:
        """Laedt ein einzelnes Statement, gecacht.

        Args:
            symbol: Kanonisches Symbol.
            kind: Art des Statement.
            getter: Name der yfinance-Methode, etwa ``"get_income_stmt"``.
            freq: yfinance-Frequenzstring: ``yearly``, ``quarterly`` oder
                ``trailing``.

        Returns:
            Tupel aus normalisiertem Frame und Cache-Flag. Ein leerer Frame
            bedeutet: die Quelle fuehrt dieses Statement nicht.

        Raises:
            ProviderUnavailableError: Quelle nicht erreichbar.
            RateLimitError: Quelle hat gedrosselt.
        """
        key = cache_key("yf", "statement", symbol, kind.value, freq)

        def _load() -> pd.DataFrame:
            def _fetch() -> pd.DataFrame:
                try:
                    raw = getattr(self._ticker(symbol), getter)(freq=freq)
                except Exception as exc:
                    mapped = _map_exception(exc, symbol)
                    if isinstance(mapped, SymbolNotFoundError):
                        # Ein fehlendes Einzel-Statement (etwa Bilanz bei TTM)
                        # ist kein Fehler des Gesamtabrufs.
                        return pd.DataFrame()
                    raise mapped from exc
                if not isinstance(raw, pd.DataFrame):
                    return pd.DataFrame()
                return normalize_statement(raw, ticker=symbol)

            return self._retry(_fetch)()

        return self._cache.cached_df(key, self._settings.cache.statements_ttl_s, _load)

    def get_composition(self, ticker: str) -> FundComposition:
        """Fetch Yahoo top holdings, cached with the original retrieval timestamp.

        Weights are fractions of fund assets, not of the reported subset.
        Yahoo does not supply a guaranteed full holdings list or holdings date.
        """
        from btlite.models.composition import Constituent

        symbol = normalize_ticker(ticker)
        key = cache_key("yf", "composition", symbol)
        cached = self._cache.get_df(key)
        if cached is not None and not cached.empty:
            result = FundComposition.model_validate_json(str(cached.iloc[0, 0]))
            return result.model_copy(
                update={"provenance": result.provenance.model_copy(update={"from_cache": True})}
            )

        def fetch() -> FundComposition:
            try:
                frame = self._ticker(symbol).funds_data.top_holdings
                holdings = [
                    Constituent(
                        ticker=str(index),
                        name=str(row["Name"]),
                        weight=float(row["Holding Percent"]),
                    )
                    for index, row in frame.iterrows()
                ]
                return FundComposition(
                    ticker=symbol,
                    holdings=holdings,
                    provenance=self._provenance(
                        from_cache=False, note="Yahoo Top Holdings; Bestandsdatum unbekannt."
                    ),
                )
            except Exception as exc:
                raise _map_exception(exc, symbol) from exc

        result = self._retry(fetch)()
        self._cache.set_df(
            key,
            pd.DataFrame({"composition": [result.model_dump_json()]}),
            self._settings.cache.info_ttl_s,
        )
        return result

    def get_isin(self, ticker: str) -> str | None:
        """Resolve an equity ISIN via yfinance's optional identifier lookup.

        The experimental upstream lookup can return '-'; this is treated as
        missing. Only validated identifiers are cached and used for matching.
        """
        symbol = normalize_ticker(ticker)
        key = cache_key("yf", "isin", symbol)
        cached = self._cache.get_df(key)
        if cached is not None and not cached.empty:
            return valid_isin(str(cached.iloc[0, 0]))

        def fetch() -> str | None:
            try:
                return valid_isin(self._ticker(symbol).isin)
            except Exception as exc:
                raise _map_exception(exc, symbol) from exc

        result = self._retry(fetch)()
        self._cache.set_df(
            key, pd.DataFrame({"isin": [result or "-"]}), self._settings.cache.info_ttl_s
        )
        return result

    def _fiscal_year_end_month(self, symbol: str) -> int | None:
        """Ermittelt den Monat des Geschaeftsjahresendes.

        Relevant bei Fiskaljahr-Versatz: Apple etwa schliesst Ende September
        ab. Peer-Vergleiche sind dadurch nur eingeschraenkt periodengenau.

        Args:
            symbol: Kanonisches Symbol.

        Returns:
            Monat von 1 bis 12, oder None wenn die Quelle ihn nicht fuehrt.
        """
        try:
            info = self._raw_info(symbol)
        except Exception as exc:  # Zusatzinfo, kein Pflichtfeld
            _LOG.debug("Fiskaljahresende fuer %s nicht ermittelbar: %s", symbol, exc)
            return None
        epoch = _as_float(info.get("lastFiscalYearEnd"))
        if epoch is None:
            return None
        return int(pd.Timestamp(epoch, unit="s", tz="UTC").month)


def _earnings_to_frame(calendar: object) -> pd.DataFrame:
    """Serialisiert den Termin-Rohbefund fuer den DataFrame-Cache.

    Der Cache des Projekts speichert DataFrames. Festgehalten wird genau die
    Unterscheidung, die :func:`next_earnings_from_calendar` braucht: ob der
    Schluessel vorhanden war, und welche Tage darunter standen.

    Args:
        calendar: Rohantwort von ``Ticker.calendar``.

    Returns:
        Frame mit den Spalten ``has_key`` und ``day`` (ISO-Datum als String).
        Ein fehlender Schluessel ergibt eine einzelne Zeile mit
        ``has_key=False``.
    """
    if not isinstance(calendar, dict) or EARNINGS_CALENDAR_KEY not in calendar:
        return pd.DataFrame({"has_key": [False], "day": [None]})
    days = [d.isoformat() for d in _as_dates(calendar[EARNINGS_CALENDAR_KEY])]
    if not days:
        return pd.DataFrame({"has_key": [True], "day": [None]})
    return pd.DataFrame({"has_key": [True] * len(days), "day": days})


def _frame_to_calendar(df: pd.DataFrame) -> dict[str, list[date]] | None:
    """Kehrt :func:`_earnings_to_frame` um.

    Args:
        df: Gecachter Frame.

    Returns:
        Ein ``calendar``-aehnliches Dictionary, oder ``None`` wenn der
        Schluessel seinerzeit fehlte - beides versteht
        :func:`next_earnings_from_calendar`.
    """
    if df.empty or not bool(df["has_key"].iloc[0]):
        return None
    days = [date.fromisoformat(str(d)) for d in df["day"] if d is not None and str(d) != "None"]
    return {EARNINGS_CALENDAR_KEY: days}


def _unique_tickers(tickers: Sequence[str]) -> list[str]:
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
