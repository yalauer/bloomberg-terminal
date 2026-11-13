"""Wechselkurse fuer die Depotbewertung aus Yahoo Finance.

Erfuellt :class:`~btlite.data.base.ExchangeRateProvider` strukturell, ohne
vom Protocol zu erben.

Die Umrechnung ist die **einzige** Stelle im Projekt, an der Waehrungen
ineinander umgerechnet werden: ein Depot hat genau eine Berichtswaehrung,
seine Positionen aber nicht. Analytik, Screening und Backtest rechnen
bewusst nicht um, siehe ``docs/DATA_SOURCES.md``.

Gepruefte Eigenschaften der Quelle (23. September 2026, gegen
``query1.finance.yahoo.com/v8/finance/chart`` verifiziert):

* Symbolschema ``<VON><NACH>=X``, etwa ``EURUSD=X``. Ein Jahr Historie sind
  rund 263 Zeilen.
* **Der Index kommt in ``Europe/London``, nicht in UTC.** Die Normalisierung
  laeuft ueber :func:`~btlite.core.clock.index_to_utc`; danach liegt der
  Zeitstempel eines Londoner Handelstages auf 00:00 UTC im Winter und auf
  23:00 UTC des Vortages im Sommer. Verglichen wird deshalb nie ueber den
  rohen Zeitstempel, sondern ueber das **Londoner Kalenderdatum** - siehe
  :func:`_london_dates`.
* An Wochenenden und Feiertagen fuehrt die Quelle keine Zeile. Fuer ein
  solches Datum gilt der letzte Handelstag davor (``asof``-Semantik).
* ``EURUSD=X`` notiert USD je EUR (1.1405 am 23.09.2026), also
  ``betrag_in_usd = betrag_in_eur * kurs``.

Kreuzkurse: Yahoo notiert zwar auch exotische Paare live, fuehrt fuer sie
aber teils **keine Historie**. Gemessen am 23.09.2026 lieferte ``HKDNOK=X``
ueber ``range=1y`` genau eine Zeile, waehrend ``HKDUSD=X`` und ``USDNOK=X``
je 263 Zeilen lieferten. Ein reiner Direktabruf waere fuer historische
Kauftage also genau bei den Paaren blind, die das Depot braucht. Deshalb
faellt :meth:`ExchangeRateProvider.get_rate` bei einem Direktfehlschlag auf
die Triangulation ueber USD zurueck. Die Gegenprobe am selben Tag:
``HKDUSD * USDNOK = 1.20806`` gegen direkt ``1.2032``, eine Abweichung von
0.4 Prozent aus der vierstelligen Rundung der Einzelkurse.

Rohe yfinance-Exceptions verlassen dieses Modul nie. Sie werden ueber
``_map_exception`` des Kursproviders auf
:class:`~btlite.core.errors.SymbolNotFoundError`,
:class:`~btlite.core.errors.ProviderUnavailableError` und
:class:`~btlite.core.errors.RateLimitError` abgebildet. Ein
:class:`~btlite.core.errors.RateLimitError` geht unveraendert nach oben und
wird **nie** umgangen.

Im **Offline-Modus** (``BTLITE_DATA_OFFLINE=1``, der Default) liest der
Provider statt aus dem Netz aus den eingefrorenen Historien unter
``tests/fixtures/rates/``. Sie werden von ``scripts/refresh_fixtures.py``
gezogen und liegen im selben normalisierten Format wie ein Live-Abruf - die
Umrechnung nach UTC ist verlustfrei, :func:`_london_dates` greift also
unveraendert. Der Dateiname ist der Slug aus :func:`fx_fixture_slug`
(``EURUSD=X`` -> ``EURUSD.parquet``); ``=X`` ist bei jedem Symbol derselbe
Zusatz und darum umkehrbar weggelassen.

Fehlt ein Kurs, wirft dieses Modul. Es weicht **niemals** auf 1.0 aus: ein
geratener Kurs saehe plausibel aus und verfaelschte still jede Summe, waehrend
eine fehlende Zahl im UI sichtbar als "n/a" erscheint. Das gilt offline
genauso: ein Paar ohne Fixture ist ein Fehler, kein Schaetzwert.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any, Final

import pandas as pd

from btlite.config import Settings, get_settings
from btlite.core.cache import DataFrameCache, cache_key, redirect_yfinance_cache
from btlite.core.clock import index_to_utc, utc_today
from btlite.core.errors import ProviderUnavailableError, SymbolNotFoundError
from btlite.core.logging import get_logger
from btlite.core.ratelimit import TokenBucket, get_registry, retry_with_backoff
from btlite.data.yfinance_provider import _map_exception
from btlite.models.common import Currency

if TYPE_CHECKING:
    from collections.abc import Callable

_LOG = get_logger(__name__)

#: Zeitzone, in der Yahoo den Index der FX-Historien liefert. Verifiziert am
#: 23.09.2026: ``exchangeTimezoneName`` ist fuer jedes ``=X``-Symbol
#: ``Europe/London``, **nicht** UTC.
FX_SOURCE_TZ: Final[str] = "Europe/London"

#: Vorlage des Yahoo-FX-Symbols. Erste Stelle Ausgangs-, zweite Zielwaehrung.
FX_SYMBOL_TEMPLATE: Final[str] = "{base}{quote}=X"

#: Zwischenwaehrung der Triangulation. USD ist bei Yahoo die Waehrung mit der
#: dichtesten Historie ueber alle Paare hinweg.
PIVOT_CURRENCY: Final[Currency] = Currency.USD

#: TTL eines **historischen** Kurses in Sekunden (30 Tage). Ein abgeschlossener
#: Handelstag aendert sich nicht mehr; die Grenze existiert nur, damit der
#: Cache nicht unbegrenzt waechst.
#: Die TTLs stehen in ``settings.cache.fx_historical_ttl_s`` und
#: ``fx_latest_ttl_s``. Gestaffelt, weil ein abgeschlossener Handelstag
#: sich nicht mehr aendert, der aktuellste Kurs aber schon.

#: TTL des **aktuellsten** Kurses in Sekunden (15 Minuten). Entspricht dem
#: Delay-Hinweis, den das UI ohnehin ausweist: ein taufrischer Kurs ist bei
#: dieser Quelle gar nicht zu haben.
#: Angeforderter Historienzeitraum. Grosszuegig, weil ein Kauftag Jahre
#: zurueckliegen kann und ein Abruf ohnehin gecacht wird.
_HISTORY_PERIOD: Final[str] = "10y"

#: Spalte, aus der der Kurs gelesen wird. Bei FX-Symbolen liefert Yahoo keine
#: Dividenden oder Splits; ``close`` ist der Schlusskurs des Handelstages.
_CLOSE: Final[str] = "close"

#: Unterverzeichnis der eingefrorenen FX-Historien unterhalb von
#: ``DataSettings.fixtures_dir``.
FIXTURE_SUBDIR: Final[str] = "rates"

#: Anhaengsel, das jedes Yahoo-FX-Symbol traegt. Im Dateinamen weggelassen,
#: weil es bei jedem Symbol identisch ist - der Slug bleibt damit umkehrbar.
_SYMBOL_SUFFIX: Final[str] = "=X"


def fx_symbol(base: Currency, quote: Currency) -> str:
    """Baut das Yahoo-Symbol eines Waehrungspaares.

    Args:
        base: Ausgangswaehrung, die umgerechnet wird.
        quote: Zielwaehrung, in der das Ergebnis steht.

    Returns:
        Symbol nach dem Schema ``<VON><NACH>=X``, etwa ``EURUSD=X``. Das
        Symbol notiert Einheiten von ``quote`` je einer Einheit von ``base``.
    """
    return FX_SYMBOL_TEMPLATE.format(base=base.value, quote=quote.value)


def fx_fixture_slug(symbol: str) -> str:
    """Bildet ein Yahoo-FX-Symbol auf den Namen seiner Fixture-Datei ab.

    Der Slug laesst das bei jedem FX-Symbol identische ``=X`` weg und ist
    damit umkehrbar: ``fx_symbol_from_slug(fx_fixture_slug(s)) == s``. Das
    Zeichen ``=`` waere unter Windows zwar zulaessig, in Shell-Aufrufen und
    Globs aber laestig.

    Args:
        symbol: Yahoo-FX-Symbol, etwa ``EURUSD=X``.

    Returns:
        Dateiname ohne Endung, etwa ``EURUSD``.
    """
    return symbol.removesuffix(_SYMBOL_SUFFIX)


def fx_symbol_from_slug(slug: str) -> str:
    """Kehrt :func:`fx_fixture_slug` um.

    Args:
        slug: Dateiname ohne Endung, etwa ``EURUSD``.

    Returns:
        Yahoo-FX-Symbol, etwa ``EURUSD=X``.
    """
    return f"{slug}{_SYMBOL_SUFFIX}"


def _require_known(currency: Currency, *, role: str) -> None:
    """Verweigert :attr:`~btlite.models.common.Currency.OTHER`.

    ``OTHER`` ist der Sammelwert fuer Codes, die der Vertrag nicht kennt. Aus
    ihm laesst sich kein Yahoo-Symbol bauen - ``OTHERUSD=X`` existiert nicht.
    Der Fehler faellt hier, ohne Abruf, statt als unverstaendlicher
    404 der Quelle.

    Args:
        currency: Zu pruefende Waehrung.
        role: ``"base"`` oder ``"quote"``, fuer die Meldung.

    Raises:
        SymbolNotFoundError: Wenn die Waehrung ``OTHER`` ist.
    """
    if currency is not Currency.OTHER:
        return
    msg = f"The {role} currency is unknown, so no exchange rate can be determined."
    raise SymbolNotFoundError(msg, detail=f"Currency.OTHER als {role} uebergeben")


def _london_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Bildet einen UTC-Index auf die Londoner Kalendertage ab.

    Notwendig, weil der Vergleich mit dem angefragten Handelstag sonst an der
    Sommerzeit scheitert: ein Londoner Handelstag im Juli liegt nach der
    Normalisierung auf 23:00 UTC des **Vortages**. Ein naiver Vergleich auf
    dem UTC-Datum verschoebe den Kurs um einen Tag.

    Args:
        index: tz-aware UTC-Index einer FX-Historie.

    Returns:
        Index der zugehoerigen Londoner Kalendertage, tz-naiv und auf
        Mitternacht normalisiert.
    """
    return pd.DatetimeIndex(index.tz_convert(FX_SOURCE_TZ).normalize().tz_localize(None))


def rate_from_history(
    df: pd.DataFrame,
    *,
    on: date | None,
    symbol: str,
) -> float:
    """Liest den gueltigen Kurs eines Handelstages aus einer FX-Historie.

    Faellt ``on`` auf ein Wochenende oder einen Feiertag, gilt der letzte
    Handelstag **davor** - die Quelle fuehrt dort keine Zeile. Liegt ``on``
    vor der ersten Zeile der Historie, gibt es keinen solchen Vorgaenger und
    der Aufruf scheitert, statt den aeltesten bekannten Kurs vorzutaeuschen.

    Args:
        df: Normalisierte Historie mit tz-aware UTC-Index und Spalte
            ``close``.
        on: Gesuchter Handelstag. ``None`` nimmt die letzte Zeile, also den
            aktuellsten verfuegbaren Kurs.
        symbol: Yahoo-Symbol, fuer die Fehlermeldung.

    Returns:
        Einheiten der Zielwaehrung je einer Einheit der Ausgangswaehrung,
        dezimal und strikt positiv.

    Raises:
        SymbolNotFoundError: Wenn die Historie leer ist, kein Handelstag am
            oder vor ``on`` liegt, oder der Schlusskurs fehlt.
    """
    if df.empty or _CLOSE not in df.columns:
        msg = f"No exchange rate available for {symbol}."
        raise SymbolNotFoundError(msg, detail=f"Historie leer oder ohne Spalte {_CLOSE!r}")

    closes = df[_CLOSE].dropna()
    if closes.empty:
        msg = f"No exchange rate available for {symbol}."
        raise SymbolNotFoundError(msg, detail="Historie enthaelt nur fehlende Schlusskurse")

    if on is None:
        raw = closes.iloc[-1]
    else:
        # asof auf dem Londoner Kalendertag: der letzte Handelstag am oder
        # vor dem Stichtag. Genau die vom Vertrag verlangte Semantik.
        days = _london_dates(pd.DatetimeIndex(closes.index))
        eligible = closes[days <= pd.Timestamp(on)]
        if eligible.empty:
            msg = f"No exchange rate available for {symbol} on {on.isoformat()}."
            raise SymbolNotFoundError(
                msg,
                detail=f"kein Handelstag am oder vor {on.isoformat()} in der Historie",
            )
        raw = eligible.iloc[-1]

    value = float(raw)
    # NaN ist nicht gleich sich selbst. Ein fehlender Kurs ist ein Fehler,
    # nie eine 1.0 und nie eine 0.0.
    if value != value or value <= 0.0:
        msg = f"No exchange rate available for {symbol}."
        raise SymbolNotFoundError(msg, detail=f"unbrauchbarer Schlusskurs: {raw!r}")
    return value


class YFinanceRateProvider:
    """Wechselkurse aus Yahoo Finance, mit Plattencache.

    Erfuellt :class:`~btlite.data.base.ExchangeRateProvider` strukturell.

    Alle ausgehenden Requests laufen durch denselben Token-Bucket wie der
    Kursprovider - es ist dieselbe Quelle, also dasselbe Limit. Ein von der
    Quelle signalisiertes Rate Limit wird nicht umgangen, sondern als
    :class:`~btlite.core.errors.RateLimitError` nach oben gereicht.

    Anders als :class:`~btlite.data.yfinance_provider.YFinanceProvider` ist
    der Offline-Modus hier **kein** Konstruktorfehler, sondern wird erst beim
    Abruf wirksam: gleiche Basis- und Zielwaehrung liefert 1.0 ohne jeden
    Netzzugriff und muss deshalb auch offline funktionieren.

    Offline bedient :meth:`_fixture_history` den Abruf aus
    ``<fixtures_dir>/rates/``. Die Triangulation ueber USD funktioniert damit
    auch offline, sobald beide Schenkel als Fixture vorliegen. Fehlt ein Paar
    auch dort, wirft der Provider - es gibt keinen Rueckfall auf 1.0.
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
                yfinance-Bucket aus der Registry - dieselbe Quelle wie die
                Kursabrufe, also dasselbe Limit.
        """
        self._settings = settings if settings is not None else get_settings()
        self._cache = cache if cache is not None else DataFrameCache(self._settings.cache)
        self._bucket = bucket if bucket is not None else get_registry().bucket("yfinance")
        self._retry = retry_with_backoff(settings=self._settings.ratelimit)
        #: Prozesslokaler Zweitcache vor dem Plattencache. Ein Depot mit zehn
        #: Positionen desselben Kauftages loest damit genau einen Abruf aus,
        #: auch wenn der Plattencache abgeschaltet ist.
        self._memo: dict[str, pd.DataFrame] = {}

    def get_rate(self, base: Currency, quote: Currency, on: date | None = None) -> float:
        """Holt den Kurs eines Waehrungspaares.

        Richtung: das Ergebnis sind Einheiten von ``quote`` je **einer**
        Einheit von ``base``. Ein Betrag rechnet sich also mit
        ``betrag_in_quote = betrag_in_base * rate`` um - bei EUR nach USD und
        einem Kurs von 1.14 werden aus 100 EUR 114 USD, nicht 87.7.

        Laesst sich das Paar nicht direkt abrufen, triangluliert die Methode
        ueber USD (:data:`PIVOT_CURRENCY`): ``base -> USD -> quote``. Das ist
        kein Komfort, sondern noetig, weil Yahoo fuer exotische Kreuze zwar
        einen Live-Kurs, aber keine Historie fuehrt (``HKDNOK=X`` lieferte am
        23.09.2026 ueber ein Jahr genau eine Zeile, die beiden USD-Schenkel je
        263). Der Umweg ist mathematisch exakt und kostet nur die Rundung der
        beiden Einzelkurse - die Gegenprobe wich um 0.4 Prozent ab.

        Args:
            base: Ausgangswaehrung, die umgerechnet wird.
            quote: Zielwaehrung, in der das Ergebnis steht.
            on: Handelstag des gesuchten Kurses, als Kalenderdatum. ``None``
                nimmt den aktuellsten verfuegbaren. Faellt das Datum auf ein
                Wochenende oder einen Feiertag, gilt der letzte Handelstag
                davor - die Quelle fuehrt dort keine Zeile.

        Returns:
            Einheiten von ``quote`` je einer Einheit von ``base``, dezimal und
            strikt positiv. Bei gleicher Waehrung exakt ``1.0``, ohne Abruf.

        Raises:
            ProviderUnavailableError: Quelle nicht erreichbar, oder der
                Offline-Modus verbietet den Abruf. Es wird **nicht** auf 1.0
                ausgewichen: ein geratener Kurs waere schlimmer als eine
                fehlende Zahl.
            RateLimitError: Quelle hat gedrosselt.
            SymbolNotFoundError: Das Waehrungspaar wird nicht notiert, eine
                der Waehrungen ist ``OTHER``, oder es gibt keinen Handelstag
                am oder vor ``on``.
        """
        # Identitaet zuerst: kein Symbol, kein Abruf, kein Offline-Fehler.
        if base is quote:
            return 1.0

        _require_known(base, role="base")
        _require_known(quote, role="quote")

        try:
            return self._direct_rate(base, quote, on)
        except (SymbolNotFoundError, ProviderUnavailableError) as exc:
            # RateLimitError faengt dieser Zweig bewusst nicht: eine
            # Drosselung geht unveraendert nach oben, statt einen zweiten und
            # dritten Request nachzuschieben.
            if PIVOT_CURRENCY in (base, quote):
                raise
            _LOG.debug(
                "Direktkurs %s nicht verfuegbar (%s), triangulierung ueber %s",
                fx_symbol(base, quote),
                exc,
                PIVOT_CURRENCY.value,
            )
            direct_error = exc

        try:
            via_pivot = self._direct_rate(base, PIVOT_CURRENCY, on)
            pivot_to_quote = self._direct_rate(PIVOT_CURRENCY, quote, on)
        except SymbolNotFoundError as exc:
            msg = (
                f"No exchange rate available for {base.value}/{quote.value}"
                f" - neither directly nor via {PIVOT_CURRENCY.value}."
            )
            raise SymbolNotFoundError(
                msg,
                detail=f"direkt: {direct_error}; trianguliert: {exc}",
            ) from exc
        return via_pivot * pivot_to_quote

    # ------------------------------------------------------------ intern --
    def _direct_rate(self, base: Currency, quote: Currency, on: date | None) -> float:
        """Holt einen direkt notierten Kurs, ohne Triangulation.

        Args:
            base: Ausgangswaehrung.
            quote: Zielwaehrung.
            on: Handelstag, oder ``None`` fuer den aktuellsten Kurs.

        Returns:
            Einheiten von ``quote`` je einer Einheit von ``base``.

        Raises:
            ProviderUnavailableError: Quelle nicht erreichbar oder offline.
            RateLimitError: Quelle hat gedrosselt.
            SymbolNotFoundError: Paar nicht notiert oder kein Kurs am Stichtag.
        """
        symbol = fx_symbol(base, quote)
        return rate_from_history(self._history(symbol, on=on), on=on, symbol=symbol)

    def _history(self, symbol: str, *, on: date | None) -> pd.DataFrame:
        """Liefert die Historie eines FX-Symbols aus Cache oder Netz.

        Dreistufig: Prozess-Memo, dann Plattencache, dann Netz. Der
        Cache-Schluessel enthaelt **nicht** den Stichtag, sondern nur, ob es
        um den aktuellsten Kurs geht. Ein Depot mit zehn Positionen und zehn
        verschiedenen Kauftagen loest damit trotzdem nur einen Abruf aus, denn
        alle Stichtage werden aus derselben Historie bedient.

        Args:
            symbol: Yahoo-FX-Symbol, etwa ``EURUSD=X``.
            on: Stichtag der Anfrage. Steuert nur die TTL: eine Anfrage nach
                dem aktuellsten Kurs (``None``) oder nach einem Datum, das
                heute oder in der Zukunft liegt, darf nur kurz gecacht werden,
                weil sich die letzte Zeile noch aendert.

        Returns:
            Normalisierte Historie mit tz-aware UTC-Index und Spalte
            ``close``.

        Raises:
            ProviderUnavailableError: Quelle nicht erreichbar oder offline.
            RateLimitError: Quelle hat gedrosselt.
            SymbolNotFoundError: Symbol existiert bei der Quelle nicht.
        """
        settled = on is not None and on < utc_today()
        key = cache_key("yf", "fx", symbol, _HISTORY_PERIOD, "settled" if settled else "latest")

        memo = self._memo.get(key)
        if memo is not None:
            return memo

        cache = self._settings.cache
        ttl_s = cache.fx_historical_ttl_s if settled else cache.fx_latest_ttl_s
        df, _from_cache = self._cache.cached_df(key, ttl_s, self._loader(symbol))
        self._memo[key] = df
        return df

    def _loader(self, symbol: str) -> Callable[[], pd.DataFrame]:
        """Baut die Nachladefunktion fuer ein FX-Symbol.

        Der Offline-Test steht bewusst **hier** und nicht im Konstruktor: ein
        Cache-Hit ist auch offline zulaessig, nur der Netzzugriff ist es
        nicht. Offline tritt statt des Abrufs die eingefrorene Historie aus
        :meth:`_fixture_history` an seine Stelle.

        Args:
            symbol: Yahoo-FX-Symbol.

        Returns:
            Funktion, die die normalisierte Historie beschafft.
        """

        def _load() -> pd.DataFrame:
            if self._settings.data.offline:
                return self._fixture_history(symbol)

            def _fetch() -> pd.DataFrame:
                import yfinance as yf

                redirect_yfinance_cache(self._settings.cache)
                self._bucket.acquire()
                try:
                    raw = yf.Ticker(symbol).history(
                        period=_HISTORY_PERIOD,
                        interval="1d",
                        auto_adjust=False,
                        actions=False,
                        raise_errors=True,
                        timeout=self._settings.data.request_timeout_s,
                    )
                except Exception as exc:
                    raise _map_exception(exc, symbol) from exc
                return normalize_fx_history(raw)

            return self._retry(_fetch)()

        return _load

    def _fixture_history(self, symbol: str) -> pd.DataFrame:
        """Liest die eingefrorene Historie eines FX-Symbols von der Platte.

        Der Offline-Ersatz fuer den Netzabruf. Die Dateien liegen unter
        ``<fixtures_dir>/rates/<slug>.parquet`` und wurden von
        ``scripts/refresh_fixtures.py`` bereits durch
        :func:`normalize_fx_history` geschickt - Index tz-aware UTC, eine
        Spalte ``close``. Erneut normalisiert wird trotzdem, damit ein
        aelteres oder von Hand gebautes Fixture dieselbe Form annimmt und
        :func:`_london_dates` greift.

        Fehlt die Datei, wirft die Methode. Es wird **nicht** auf 1.0 oder auf
        ein benachbartes Paar ausgewichen: eine Luecke ist im UI als "n/a"
        sichtbar, ein geratener Kurs waere still falsch.

        Args:
            symbol: Yahoo-FX-Symbol, etwa ``EURUSD=X``.

        Returns:
            Normalisierte Historie mit tz-aware UTC-Index und Spalte
            ``close``. Die Werte sind Einheiten der Zielwaehrung je einer
            Einheit der Ausgangswaehrung, dezimal.

        Raises:
            ProviderUnavailableError: Wenn es zu diesem Paar kein Fixture
                gibt oder die Datei unlesbar ist.
        """
        root = self._settings.data.fixtures_dir / FIXTURE_SUBDIR
        path = root / f"{fx_fixture_slug(symbol)}.parquet"
        if not path.is_file():
            msg = (
                f"Offline mode: the exchange rate {symbol} cannot be retrieved. "
                "Set BTLITE_DATA_OFFLINE=0 for live data."
            )
            raise ProviderUnavailableError(
                msg,
                detail=f"BTLITE_DATA_OFFLINE=true, kein Fixture unter {path}",
            )
        try:
            raw = pd.read_parquet(path)
        except Exception as exc:
            msg = (
                f"Offline mode: the frozen exchange rate {symbol} cannot be read. "
                "Set BTLITE_DATA_OFFLINE=0 for live data."
            )
            raise ProviderUnavailableError(
                msg, detail=f"{path} unlesbar: {type(exc).__name__}: {exc}"
            ) from exc
        _LOG.debug("Offline-Kurs %s aus Fixture %s", symbol, path)
        return normalize_fx_history(raw)


def normalize_fx_history(df: Any) -> pd.DataFrame:
    """Normalisiert eine rohe FX-Historie auf tz-aware UTC und Spalte ``close``.

    Die Quelle liefert den Index in ``Europe/London``
    (:data:`FX_SOURCE_TZ`), das Projekt verlangt UTC. Konvertiert wird ueber
    :func:`~btlite.core.clock.index_to_utc`; das Londoner Kalenderdatum bleibt
    ueber :func:`_london_dates` rekonstruierbar.

    Args:
        df: Rohframe aus ``Ticker.history``. Spalten in Grossschreibung,
            Index datumsartig.

    Returns:
        Frame mit aufsteigendem, duplikatfreiem UTC-Index und der Spalte
        ``close`` als float. Ist die Eingabe leer, ist es auch das Ergebnis -
        der Fehler faellt erst in :func:`rate_from_history`, mit dem Symbol im
        Text.
    """
    empty = pd.DataFrame(
        {_CLOSE: pd.Series(dtype="float64")},
        index=pd.DatetimeIndex([], tz="UTC", name="date"),
    )
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return empty

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    closes = [c for c in out.columns if str(c).strip().lower() == _CLOSE]
    if not closes:
        return empty

    out = out[[closes[0]]]
    out.columns = pd.Index([_CLOSE])
    out = index_to_utc(out)
    out = out.sort_index()
    # Yahoo korrigiert Historien rueckwirkend - der spaetere Wert gilt.
    out = out[~out.index.duplicated(keep="last")]
    out[_CLOSE] = pd.to_numeric(out[_CLOSE], errors="coerce")
    return out
