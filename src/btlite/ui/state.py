"""Verdrahtung des UI mit der Datenschicht.

Haelt genau eine Provider-Instanz je Prozess und puffert die teuren Abrufe,
damit ein Seitenwechsel nicht erneut laedt. Dieses Modul rechnet nicht - es
ruft Datenschicht, Analytics und Screening auf und reicht die Ergebnisse
unveraendert weiter (Architekturregel 4).

Providerwahl: allein ``DataSettings.offline`` entscheidet, welche Quelle
bedient. Der Default ist ``True`` und damit der
:class:`~btlite.data.fixtures.FixtureProvider`: die App startet ohne Netz und
zeigt sofort etwas, statt beim ersten Klick in ein Timeout oder ein Rate
Limit zu laufen.

Live-Daten holt man mit ``BTLITE_DATA_OFFLINE=0``. Laesst sich die Live-Quelle
dann nicht bauen, faellt die Session sichtbar auf die Fixtures zurueck. Der
Herkunftsnachweis jeder Seite weist aus, welche Quelle geantwortet hat.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final

from btlite.config import Settings, get_settings
from btlite.core.cache import DataFrameCache, cache_key
from btlite.core.errors import BTLiteError, DataError
from btlite.core.logging import get_logger

# Laufzeit-Import: das Protocol ist runtime_checkable und wird unten
# fuer isinstance gebraucht, nicht nur als Typannotation.
from btlite.data.base import SymbolSearch
from btlite.data.indices import IndexConstituentsProvider
from btlite.data.rates import YFinanceRateProvider
from btlite.models.common import INTRADAY_INTERVALS, Interval, Period
from btlite.models.fundamentals import RatioSet, StatementBundle
from btlite.models.market import EarningsDate, Instrument, PriceSeries, Quote
from btlite.portfolio.store import HoldingsStore

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from btlite.data.base import MarketDataProvider, ScreenerProvider

_LOG = get_logger(__name__)

#: Obergrenze des Seiten-Caches je Kategorie. Klein gehalten: eine
#: Terminal-Session springt zwischen wenigen Titeln hin und her.
_CACHE_SIZE: Final[int] = 64

#: Zeitraum-Schaltflaechen der Seite GP. Wert ist die yfinance-Notation.
PERIOD_CHOICES: Final[tuple[tuple[str, str], ...]] = (
    ("1M", "1mo"),
    ("3M", "3mo"),
    ("6M", "6mo"),
    ("1Y", "1y"),
    ("3Y", "3y"),
    ("5Y", "5y"),
    ("MAX", "max"),
)


@dataclass
class _Lru:
    """Sehr einfacher LRU-Puffer fuer Modellobjekte.

    ``functools.lru_cache`` scheidet aus, weil der Puffer zur Laufzeit
    geleert werden koennen muss, ohne die gebundene Methode zu kennen.

    Attributes:
        maxsize: Obergrenze der Eintraege.
        store: Abbildung Schluessel auf Wert, aelteste zuerst.
    """

    maxsize: int = _CACHE_SIZE
    store: dict[tuple[object, ...], Any] = field(default_factory=dict)

    def get(self, key: tuple[object, ...]) -> Any | None:
        """Liest einen Eintrag und markiert ihn als zuletzt benutzt.

        Args:
            key: Schluessel des Eintrags.

        Returns:
            Der gepufferte Wert, oder None.
        """
        if key not in self.store:
            return None
        value = self.store.pop(key)
        self.store[key] = value
        return value

    def put(self, key: tuple[object, ...], value: Any) -> None:
        """Legt einen Eintrag ab und verdraengt den aeltesten.

        Args:
            key: Schluessel des Eintrags.
            value: Abzulegender Wert.
        """
        self.store.pop(key, None)
        self.store[key] = value
        while len(self.store) > self.maxsize:
            self.store.pop(next(iter(self.store)))

    def clear(self) -> None:
        """Leert den Puffer."""
        self.store.clear()


class AppState:
    """Zustand einer UI-Session.

    Haelt Provider, Screening-Engine, Strategie-Registry und die Puffer der
    Seiten-Daten. Eine Instanz je Prozess, erreichbar ueber
    :func:`get_state`.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialisiert den Zustand und waehlt den Provider.

        Args:
            settings: Projekt-Settings. ``None`` nimmt die globalen.
        """
        self.settings = settings if settings is not None else get_settings()
        self._provider, self._offline_reason = _build_provider(self.settings)
        self._screener = _build_screener(self.settings, self._provider)

        # Der Depot-Speicher ist bewusst NICHT gepuffert: er ist die
        # Wahrheit ueber den Bestand des Nutzers und wird zwischen zwei
        # Anfragen geschrieben. Ein Puffer wuerde gerade gespeicherte
        # Positionen verschwinden lassen.
        self._holdings = HoldingsStore(self.settings.portfolio.holdings_file)
        # symbol_search ist kein Beiwerk: ohne ihn laeuft nur die erste
        # Aufloesungsstufe (Boersen-Endung aus dem Sitzland), und die
        # scheitert bei europaeischen Titeln an 110 von 462 Werten. Der
        # aktive Provider erfuellt das SymbolSearch-Protocol, sofern er
        # live ist - der FixtureProvider tut es ebenso.
        self._indices = IndexConstituentsProvider(
            settings=self.settings,
            symbol_search=self._provider if isinstance(self._provider, SymbolSearch) else None,
        )

        # Waehrungskurse fuer die Depotbewertung. Eigener Anbieter, weil er
        # ein anderes Symbolschema nutzt (EURUSD=X) und andere TTLs braucht
        # als Aktienkurse: ein historischer Kurs aendert sich nie mehr.
        self._rates = YFinanceRateProvider(settings=self.settings)

        self._instruments = _Lru()
        self._quotes = _Lru()
        self._histories = _Lru()
        self._bundles = _Lru()
        self._ratios = _Lru()
        self._earnings = _Lru()
        self._screen_frames = DataFrameCache(self.settings.cache)

    # ----------------------------------------------------------- Provider --
    @property
    def provider(self) -> MarketDataProvider:
        """Die aktive Marktdatenquelle.

        Returns:
            Provider fuer Stammdaten, Kurse und Statements.
        """
        return self._provider

    @property
    def holdings(self) -> HoldingsStore:
        """Der Speicher fuer Depots und Watchlists.

        Liest und schreibt die JSON-Datei aus
        ``settings.portfolio.holdings_file``. Bewusst ungepuffert.

        Returns:
            Der Depot-Speicher dieser Session.
        """
        return self._holdings

    @property
    def rates(self) -> YFinanceRateProvider:
        """Die Quelle der Wechselkurse.

        Wird allein fuer die Depotbewertung gebraucht: ein in Euro
        gekaufter Titel muss in die Berichtswaehrung des Depots umgerechnet
        werden, sonst waere die Summe eine Addition ungleicher Einheiten.
        Analytik, Screening und Backtest rechnen bewusst nicht um, siehe
        ``docs/DATA_SOURCES.md``.

        Returns:
            Der Kursanbieter dieser Session.
        """
        return self._rates

    @property
    def indices(self) -> IndexConstituentsProvider:
        """Die Quelle der Indexbestaende.

        Liefert die Mitglieder von S&P 500, Dow Jones, Nasdaq 100,
        Russell 2000 und DAX. Der Abruf laeuft live und wird eine Woche
        gecacht; offline bedient nur ein bereits gefuellter Cache.

        Returns:
            Der Indexanbieter dieser Session.
        """
        return self._indices

    @property
    def screener(self) -> ScreenerProvider | None:
        """Die aktive Screener-Quelle.

        Returns:
            Screener-Provider, oder None wenn keiner verfuegbar ist.
        """
        return self._screener

    @property
    def is_offline(self) -> bool:
        """Ob die Session gegen eingefrorene Daten laeuft.

        Returns:
            True wenn der FixtureProvider bedient.
        """
        return self._offline_reason is not None

    @property
    def offline_reason(self) -> str | None:
        """Begruendung des Offline-Betriebs fuer die Kopfzeile.

        Returns:
            Hinweistext, oder None im Live-Betrieb.
        """
        return self._offline_reason

    @property
    def timezone(self) -> str:
        """Anzeigezeitzone der Oberflaeche.

        Returns:
            IANA-Name aus den UI-Settings.
        """
        return self.settings.ui.display_timezone

    @property
    def default_tickers(self) -> list[str]:
        """Vorbelegung, wenn ein Kommando ohne Ticker kommt.

        Bevorzugt Symbole, fuer die die aktive Quelle Daten hat.

        Returns:
            Symbolliste, mindestens ein Eintrag.
        """
        configured = list(self.settings.ui.default_tickers)
        available = self.available_tickers()
        if not available:
            return configured
        usable = [t for t in configured if t in available]
        return usable or available[:3]

    def available_tickers(self) -> list[str]:
        """Listet die Symbole, die die Quelle sicher bedienen kann.

        Nur der FixtureProvider kann das beantworten. Eine Live-Quelle
        kennt praktisch jedes Symbol und liefert deshalb eine leere Liste.

        Returns:
            Sortierte Symbolliste, oder leer.
        """
        lister = getattr(self._provider, "available_tickers", None)
        if lister is None:
            return []
        result: list[str] = lister()
        return result

    def clear_cache(self) -> None:
        """Verwirft alle gepufferten Seiten-Daten."""
        for cache in (
            self._instruments,
            self._quotes,
            self._histories,
            self._bundles,
            self._ratios,
            self._earnings,
        ):
            cache.clear()
        self._screen_frames.clear()

    # --------------------------------------------------------- Stammdaten --
    def instrument(self, ticker: str) -> Instrument:
        """Holt Stammdaten, gepuffert.

        Args:
            ticker: Kanonisches Symbol.

        Returns:
            Stammdaten des Instruments.

        Raises:
            BTLiteError: Wenn die Quelle nicht liefert.
        """
        key = ("instrument", ticker)
        cached: Instrument | None = self._instruments.get(key)
        if cached is None:
            cached = self._provider.get_instrument(ticker)
            self._instruments.put(key, cached)
        return cached

    def quotes(self, tickers: list[str]) -> dict[str, Quote]:
        """Holt Momentaufnahmen, gepuffert ueber die gesamte Anfrage.

        Args:
            tickers: Kanonische Symbole.

        Returns:
            Abbildung Symbol auf Quote, nur fuer erfolgreiche Abrufe.

        Raises:
            BTLiteError: Wenn die Quelle nicht liefert.
        """
        key = ("quotes", tuple(tickers))
        cached: dict[str, Quote] | None = self._quotes.get(key)
        if cached is None:
            cached = self._provider.get_quotes(tickers)
            self._quotes.put(key, cached)
        return cached

    def earnings_dates(self, tickers: list[str]) -> dict[str, EarningsDate]:
        """Holt die naechsten Quartalszahlen-Termine, gepuffert.

        Ein fehlender Termin ist **kein Fehler**: ein ETF fuehrt keine
        Quartalszahlen, und zu einer Aktie kennt die Quelle den naechsten
        Termin nicht immer. Scheitert der Abruf ganz - offline ohne
        Fixture, Quelle nicht erreichbar -, liefert diese Methode ein
        leeres Ergebnis statt zu werfen: die Termine sind eine Zugabe zur
        Tabelle, und ihretwegen darf die Depotanzeige nicht ausfallen.

        Args:
            tickers: Kanonische Symbole.

        Returns:
            Abbildung Symbol auf Termin, nur fuer beantwortete Abrufe.
            Leer, wenn die Quelle gar nicht antwortet.
        """
        if not tickers:
            return {}
        key = ("earnings", tuple(sorted(tickers)))
        cached: dict[str, EarningsDate] | None = self._earnings.get(key)
        if cached is None:
            getter = getattr(self._provider, "get_earnings_dates", None)
            if getter is None:
                return {}
            try:
                cached = getter(sorted(tickers))
            except BTLiteError as exc:
                _LOG.warning("Quartalstermine nicht abrufbar: %s", exc.detail or exc.user_message)
                return {}
            self._earnings.put(key, cached)
        return cached

    def quote(self, ticker: str) -> Quote | None:
        """Holt die Momentaufnahme eines einzelnen Symbols.

        Args:
            ticker: Kanonisches Symbol.

        Returns:
            Quote, oder None wenn die Quelle keinen liefert.
        """
        return self.quotes([ticker]).get(ticker)

    # -------------------------------------------------------------- Kurse --
    def history(
        self,
        ticker: str,
        *,
        period: str = "1y",
        interval: Interval = Interval.D1,
        include_currency: bool = True,
    ) -> PriceSeries:
        """Holt eine Kurshistorie, gepuffert.

        Args:
            ticker: Kanonisches Symbol.
            period: Relativer Zeitraum in yfinance-Notation.
            interval: Kerzen-Intervall.
            include_currency: Handelswaehrung aufloesen. Technische Screens
                setzen False und sparen einen separaten Request.

        Returns:
            Kursreihe mit UTC-Index.

        Raises:
            BTLiteError: Wenn die Quelle nicht liefert.
        """
        key = ("history", ticker, period, interval.value, include_currency)
        if interval in INTRADAY_INTERVALS:
            # Der Speicherpuffer kennt keine Ablaufzeit. Ein 1-Minuten-Chart
            # stuende damit bis zum Neustart still; der Plattencache des
            # Providers laeuft dagegen nach ``cache.intraday_ttl_s`` ab.
            return self._provider.get_history(
                ticker,
                interval=interval,
                period=period,
                auto_adjust=self.settings.data.auto_adjust_prices,
                include_currency=include_currency,
            )
        cached: PriceSeries | None = self._histories.get(key)
        if cached is None:
            cached = self._provider.get_history(
                ticker,
                interval=interval,
                period=period,
                auto_adjust=self.settings.data.auto_adjust_prices,
                include_currency=include_currency,
            )
            self._histories.put(key, cached)
        return cached

    def histories(
        self,
        tickers: list[str],
        *,
        period: str = "1y",
        interval: Interval = Interval.D1,
        on_progress: Callable[[int, int], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        include_currency: bool = True,
    ) -> dict[str, PriceSeries]:
        """Holt mehrere Kurshistorien, je Symbol gepuffert.

        Symbole, die die Quelle nicht kennt, fehlen im Ergebnis - das
        entspricht dem Vertrag von ``PriceProvider.get_histories``.

        Args:
            tickers: Kanonische Symbole.
            period: Relativer Zeitraum in yfinance-Notation.
            interval: Kerzen-Intervall.
            on_progress: Optionaler Rueckruf ``(erledigt, gesamt)``. Gemeldet
                wird jedes ``PROGRESS_EVERY``-te Symbol und immer das letzte,
                sodass die Anzeige mit ``(gesamt, gesamt)`` schliesst statt kurz
                davor stehen zu bleiben. Uebersprungene Symbole zaehlen als
                erledigt - der Balken misst den Fortschritt, nicht die Zahl der
                Treffer. Wirft der Rueckruf, wird das einmal geloggt und danach
                nichts mehr gemeldet; der Abruf laeuft unveraendert zu Ende.
                ``None`` meldet nichts.
            should_stop: Optionales Praedikat. Liefert es True, endet der
                Abruf vor dem naechsten Symbol mit dem bis dahin Geholten.
                Bewusst getrennt von ``on_progress``: ein Abbruch ist kein
                gescheiterter Fortschritt, und jener Rueckruf schluckt
                Ausnahmen absichtlich.
            include_currency: Handelswaehrung je Reihe aufloesen. Fuer reine
                technische Kennzahlen nicht erforderlich.

        Returns:
            Abbildung Symbol auf Kursreihe. Bei Abbruch nur die bis dahin
            geholten.
        """
        from btlite.screening.universe import PROGRESS_EVERY, progress_reporter

        symbols = list(tickers)
        total = len(symbols)
        report = progress_reporter(on_progress)
        out: dict[str, PriceSeries] = {}

        def load(ticker: str) -> PriceSeries:
            return self.history(
                ticker,
                period=period,
                interval=interval,
                include_currency=include_currency,
            )

        def accept(ticker: str, future: Future[PriceSeries]) -> None:
            try:
                out[ticker] = future.result()
            except DataError as exc:
                _LOG.info("Kurshistorie fuer %s uebersprungen: %s", ticker, exc.user_message)

        workers = 1 if self.is_offline else min(self.settings.data.screening_workers, total or 1)
        if workers == 1:
            for done, ticker in enumerate(symbols, start=1):
                if should_stop is not None and should_stop():
                    _LOG.info("Kursabruf nach %d von %d Symbolen abgebrochen", done - 1, total)
                    break
                future: Future[PriceSeries] = Future()
                try:
                    future.set_result(load(ticker))
                except Exception as exc:
                    future.set_exception(exc)
                accept(ticker, future)
                report(done, total)
        else:
            iterator = iter(symbols)
            done_count = 0
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="btlite-prices"
            ) as pool:
                pending: dict[Future[PriceSeries], str] = {}

                def submit_next() -> bool:
                    if should_stop is not None and should_stop():
                        return False
                    try:
                        ticker = next(iterator)
                    except StopIteration:
                        return False
                    pending[pool.submit(load, ticker)] = ticker
                    return True

                for _ in range(workers):
                    if not submit_next():
                        break
                while pending:
                    completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        ticker = pending.pop(future)
                        accept(ticker, future)
                        done_count += 1
                        if done_count % PROGRESS_EVERY == 0 or done_count == total:
                            report(done_count, total)
                        submit_next()
        return out

    # --------------------------------------------------------- Statements --
    def statements(self, ticker: str, *, period: Period = Period.ANNUAL) -> StatementBundle:
        """Holt normalisierte Statements, gepuffert.

        ``Period.TTM`` entsteht aus den Quartalszahlen - die Quelle fuehrt
        keine TTM-Statements. Die Aggregation macht
        ``analytics/statements.py``, nicht dieses Modul.

        Args:
            ticker: Kanonisches Symbol.
            period: Jahres-, Quartals- oder TTM-Basis.

        Returns:
            Normalisiertes Statement-Buendel.

        Raises:
            BTLiteError: Wenn die Quelle nicht liefert.
        """
        from btlite.analytics.statements import compute_ttm_bundle, normalize_bundle

        key = ("statements", ticker, period.value)
        cached: StatementBundle | None = self._bundles.get(key)
        if cached is not None:
            return cached

        if period is Period.TTM:
            quarterly = self.statements(ticker, period=Period.QUARTERLY)
            bundle = compute_ttm_bundle(quarterly)
        else:
            bundle = normalize_bundle(self._provider.get_statements(ticker, period=period))

        self._bundles.put(key, bundle)
        return bundle

    def ratios(self, ticker: str, *, period: Period = Period.ANNUAL) -> RatioSet:
        """Berechnet die Kennzahlen eines Titels, gepuffert.

        Kurs, Marktkapitalisierung und Dividende kommen aus Quote und
        Stammdaten; fehlen sie, bleiben die Bewertungskennzahlen ``None``
        und erscheinen als "n/a".

        Args:
            ticker: Kanonisches Symbol.
            period: Basis der Kennzahlen.

        Returns:
            Kennzahlensatz.

        Raises:
            BTLiteError: Wenn die Statements nicht beschaffbar sind.
        """
        from btlite.analytics.ratios import compute_ratios

        key = ("ratios", ticker, period.value)
        cached: RatioSet | None = self._ratios.get(key)
        if cached is not None:
            return cached

        bundle = self.statements(ticker, period=period)

        price: float | None = None
        market_cap: float | None = None
        dividend: float | None = None
        try:
            quote = self.quote(ticker)
            price = None if quote is None else quote.price
        except DataError as exc:
            _LOG.info("Kurs fuer %s nicht verfuegbar: %s", ticker, exc.user_message)
        try:
            instrument = self.instrument(ticker)
            market_cap = instrument.market_cap
            dividend = instrument.dividend_rate
        except DataError as exc:
            _LOG.info("Stammdaten fuer %s nicht verfuegbar: %s", ticker, exc.user_message)

        ratio_set = compute_ratios(
            bundle,
            price=price,
            market_cap=market_cap,
            dividend_per_share=dividend,
        )
        self._ratios.put(key, ratio_set)
        return ratio_set

    # ---------------------------------------------------------- Screening --
    def screening_engine(self) -> Any:
        """Baut eine Screening-Engine mit der aktiven Quelle.

        Returns:
            Frische :class:`~btlite.screening.engine.ScreeningEngine`.
        """
        from btlite.screening.engine import ScreeningEngine

        return ScreeningEngine(provider=self._screener)

    def ratio_universe(
        self,
        tickers: list[str],
        *,
        period: Period = Period.ANNUAL,
        on_progress: Callable[[int, int], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> pd.DataFrame:
        """Baut das Kennzahlen-Universum fuer lokale Filter.

        Args:
            tickers: Zu betrachtende Symbole.
            period: Basis der Kennzahlen.
            on_progress: Optionaler Rueckruf ``(erledigt, gesamt)``. Bei
                einem grossen Universum dauert der Aufbau Minuten; die
                Seite zeigt damit den Stand statt nur eines Ladebalkens.
            should_stop: Optionales Praedikat. Liefert es True, endet der
                Aufbau vor dem naechsten Symbol.

        Returns:
            Matrix Ticker mal Kennzahl, wie ``ScreeningEngine.screen`` sie
            als ``universe`` erwartet.
        """
        from btlite.screening.universe import build_ratio_universe

        symbols = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
        key = cache_key("screening", "ratio-universe-v1", period.value, sorted(symbols))
        cached = self._screen_frames.get_df(key)
        if cached is not None:
            if on_progress is not None:
                on_progress(len(symbols), len(symbols))
            return cached

        frame = build_ratio_universe(
            self._provider,
            symbols,
            period=period,
            on_progress=on_progress,
            should_stop=should_stop,
            max_workers=1 if self.is_offline else self.settings.data.screening_workers,
        )
        # Ein abgebrochener Lauf darf nie als vollstaendiges Universum in
        # spaeteren Screens wiedererscheinen.
        if should_stop is None or not should_stop():
            self._screen_frames.set_df(key, frame, self.settings.cache.statements_ttl_s)
        return frame

    # ----------------------------------------------------------- Backtest --
    def strategy_registry(self) -> Any:
        """Liefert die Registry der verfuegbaren Strategien.

        Returns:
            Frische Registry mit den mitgelieferten Strategien.
        """
        from btlite.backtest.registry import default_registry

        return default_registry()

    def backtest_engine(self) -> Any:
        """Liefert die Default-Backtest-Engine.

        Returns:
            :class:`~btlite.backtest.vbt_engine.VectorbtEngine`, gespeist
            mit dem risikofreien Zins aus den Analytics-Settings.
        """
        from btlite.backtest.vbt_engine import VectorbtEngine

        return VectorbtEngine(risk_free_rate=self.settings.analytics.risk_free_rate)


def _build_provider(settings: Settings) -> tuple[MarketDataProvider, str | None]:
    """Waehlt die Marktdatenquelle anhand von Settings und Umgebung.

    Args:
        settings: Projekt-Settings.

    Returns:
        Paar aus Provider und Offline-Begruendung. Die Begruendung ist
        ``None``, wenn eine Live-Quelle bedient.
    """
    from btlite.data.fixtures import FixtureProvider

    if settings.data.offline:
        _LOG.info("Offline-Modus aktiv: FixtureProvider bedient alle Anfragen")
        return (
            FixtureProvider(settings=settings),
            "Frozen fixtures. Set BTLITE_DATA_OFFLINE=0 for live data",
        )

    try:
        from btlite.data.yfinance_provider import YFinanceProvider

        return YFinanceProvider(settings=settings), None
    except (BTLiteError, ImportError) as exc:
        _LOG.warning("Live-Quelle nicht verfuegbar, weiche auf Fixtures aus: %s", exc)
        return FixtureProvider(settings=settings), "Live source unavailable, fixtures active"


def _build_screener(settings: Settings, fallback: MarketDataProvider) -> ScreenerProvider | None:
    """Waehlt die Screener-Quelle.

    Args:
        settings: Projekt-Settings.
        fallback: Aktive Marktdatenquelle. Erfuellt der FixtureProvider
            zugleich ``ScreenerProvider``, uebernimmt er den Screen.

    Returns:
        Screener-Provider, oder None wenn keiner verfuegbar ist.
    """
    from btlite.data.base import ScreenerProvider as ScreenerProtocol

    if settings.data.offline and isinstance(fallback, ScreenerProtocol):
        return fallback

    try:
        from btlite.data.finviz_provider import FinvizProvider

        return FinvizProvider(settings=settings)
    except (BTLiteError, ImportError) as exc:
        _LOG.warning("Finviz nicht verfuegbar: %s", exc)

    return fallback if isinstance(fallback, ScreenerProtocol) else None


#: Prozessweite Instanz, gesetzt beim Bau der App.
_STATE: AppState | None = None


def get_state(settings: Settings | None = None) -> AppState:
    """Liefert den prozessweiten UI-Zustand.

    Args:
        settings: Projekt-Settings fuer den ersten Aufruf. Spaetere Aufrufe
            ignorieren das Argument und liefern die bestehende Instanz.

    Returns:
        Die :class:`AppState`-Instanz dieses Prozesses.
    """
    global _STATE
    if _STATE is None:
        _STATE = AppState(settings)
    return _STATE


def set_state(state: AppState | None) -> None:
    """Setzt den prozessweiten Zustand, fuer Tests und den App-Bau.

    Args:
        state: Neue Instanz, oder None zum Zuruecksetzen.
    """
    global _STATE
    _STATE = state


def latest_timestamp(*candidates: datetime | None) -> datetime | None:
    """Waehlt den juengsten von mehreren Zeitstempeln.

    Args:
        *candidates: Zeitpunkte, ``None`` wird uebergangen.

    Returns:
        Juengster Zeitpunkt, oder None wenn keiner vorliegt.
    """
    present = [c for c in candidates if c is not None]
    return max(present) if present else None
