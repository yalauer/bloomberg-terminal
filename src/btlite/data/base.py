"""Protocols der Datenschicht.

Architekturregel 2: ``data/`` ist die **einzige** Schicht, die Netzwerk
macht. Alle anderen Module bekommen einen dieser Protocols injiziert und
holen niemals selbst Daten.

Die Protocols sind ``runtime_checkable``, damit Tests mit ``isinstance``
gegen sie pruefen koennen. Implementierungen muessen nicht von ihnen erben -
strukturelle Kompatibilitaet genuegt.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Sequence
from datetime import date, datetime
from typing import Protocol, runtime_checkable

from btlite.models.common import Currency, Interval, Period
from btlite.models.fundamentals import StatementBundle
from btlite.models.market import EarningsDate, Instrument, PriceSeries, Quote, SearchHit
from btlite.models.screening import (
    IndexConstituents,
    IndexDefinition,
    ScreenRequest,
    ScreenResult,
)


@runtime_checkable
class InstrumentProvider(Protocol):
    """Liefert Stammdaten zu einem Instrument."""

    def get_instrument(self, ticker: str) -> Instrument:
        """Holt die Stammdaten eines Symbols.

        Args:
            ticker: Kanonisches Symbol.

        Returns:
            Validierte Stammdaten.

        Raises:
            SymbolNotFoundError: Symbol existiert bei der Quelle nicht.
            ProviderUnavailableError: Quelle nicht erreichbar.
            RateLimitError: Quelle hat gedrosselt.
        """
        ...


@runtime_checkable
class SymbolSearch(Protocol):
    """Findet Symbole ueber den Namen des Emittenten.

    Loest die Einstiegshuerde: ohne diese Rolle muesste ein Nutzer das
    Kuerzel bereits kennen. Die Eingabe ``servicenow`` fuehrt hierueber auf
    ``NOW``.
    """

    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        """Sucht Instrumente nach Name oder Kuerzel.

        Eine leere oder rein aus Trennzeichen bestehende Eingabe liefert eine
        leere Liste und ist **kein** Fehler - waehrend des Tippens ist das der
        Normalfall.

        Args:
            query: Suchbegriff, Name oder Kuerzel. Gross- und Kleinschreibung
                spielt keine Rolle.
            limit: Maximale Trefferzahl. Werte kleiner 1 liefern eine leere
                Liste.

        Returns:
            Treffer, absteigend nach Relevanz sortiert. Der beste Treffer
            steht an Position 0. Hoechstens ``limit`` Eintraege.

        Raises:
            ProviderUnavailableError: Quelle nicht erreichbar.
            RateLimitError: Quelle hat gedrosselt.
        """
        ...


@runtime_checkable
class PriceProvider(Protocol):
    """Liefert Kurshistorien und Momentaufnahmen."""

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
            include_currency: Handelswaehrung aufloesen. Technische Screens
                koennen darauf verzichten und sparen damit einen Request.

        Returns:
            Kursreihe mit UTC-Index. ``adjusted`` spiegelt ``auto_adjust``.

        Raises:
            SymbolNotFoundError: Symbol existiert bei der Quelle nicht.
            ProviderUnavailableError: Quelle nicht erreichbar.
            RateLimitError: Quelle hat gedrosselt.
        """
        ...

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
        """Holt Historien fuer mehrere Symbole in moeglichst wenigen Requests.

        Symbole, die die Quelle nicht kennt, fehlen im Ergebnis-Dictionary,
        statt den gesamten Aufruf scheitern zu lassen.

        Args:
            tickers: Kanonische Symbole.
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
        ...

    def get_quotes(self, tickers: Sequence[str]) -> dict[str, Quote]:
        """Holt Momentaufnahmen fuer mehrere Symbole in **einem** Request.

        Diese Ein-Request-Garantie ist die Grundlage dafuer, dass der
        Polling-Stream bei 20 Tickern nicht ins Rate Limit laeuft.

        Args:
            tickers: Kanonische Symbole.

        Returns:
            Abbildung Symbol auf Quote, nur fuer erfolgreiche Abrufe.

        Raises:
            ProviderUnavailableError: Quelle nicht erreichbar.
            RateLimitError: Quelle hat gedrosselt.
        """
        ...


@runtime_checkable
class FundamentalsProvider(Protocol):
    """Liefert Financial Statements."""

    def get_statements(self, ticker: str, *, period: Period = Period.ANNUAL) -> StatementBundle:
        """Holt Income Statement, Bilanz und Cashflow eines Symbols.

        Der Provider liefert die Statements **roh und transponiert-korrigiert**
        (Index = Positionen), aber ohne semantische Normalisierung der
        Zeilenbezeichner. Die uebernimmt ``analytics/statements.py``.

        Args:
            ticker: Kanonisches Symbol.
            period: Jahres- oder Quartalsbasis.

        Returns:
            Buendel der drei Statements. Einzelne koennen None sein, wenn
            die Quelle sie nicht fuehrt.

        Raises:
            SymbolNotFoundError: Symbol existiert bei der Quelle nicht.
            ProviderUnavailableError: Quelle nicht erreichbar.
            RateLimitError: Quelle hat gedrosselt.
        """
        ...


@runtime_checkable
class ScreenerProvider(Protocol):
    """Fuehrt einen Screen bei der Quelle aus."""

    def screen(self, request: ScreenRequest) -> ScreenResult:
        """Fuehrt die Filter der Anfrage bei der Quelle aus.

        Nur ``request.finviz_filters`` wird remote ausgewertet. Lokale
        Praedikate wendet ``screening/engine.py`` danach an.

        Args:
            request: Screening-Anfrage.

        Returns:
            Treffer der Quelle.

        Raises:
            ProviderUnavailableError: Quelle nicht erreichbar.
            RateLimitError: Quelle hat gedrosselt.
            UnknownFilterError: Ein Filter-Key ist der Quelle unbekannt.
        """
        ...

    def available_filters(self) -> dict[str, list[str]]:
        """Listet die von der Quelle unterstuetzten Filter.

        Returns:
            Abbildung Filter-Key auf die Liste zulaessiger Werte.
        """
        ...


@runtime_checkable
class IndexProvider(Protocol):
    """Liefert die Bestaende gaengiger Aktienindizes.

    Loest den Mangel, dass ein Screening sonst gegen eine Handvoll fest
    verdrahteter Ticker liefe. Die Rolle liefert das **vollstaendige**
    Universum eines Index, etwa alle 503 Titel des S&P 500.
    """

    def available_indices(self) -> list[IndexDefinition]:
        """Listet die unterstuetzten Indizes fuer die Auswahl im UI.

        Muss auch ohne Netz funktionieren: die Auswahlliste ist statisch,
        nur der Bestand dahinter ist es nicht.

        Returns:
            Definitionen in stabiler Reihenfolge, die Vorauswahl zuerst.
        """
        ...

    def get_constituents(self, index_key: str) -> IndexConstituents:
        """Holt den Bestand eines Index.

        Args:
            index_key: Bezeichner des Index, etwa ``sp500``.

        Returns:
            Bestand mit Symbolen und, soweit die Quelle sie fuehrt,
            Firmennamen und Sektoren.

        Raises:
            ValueError: Wenn der Index unbekannt ist.
            ProviderUnavailableError: Quelle nicht erreichbar, oder der
                Offline-Modus verbietet den Abruf.
            RateLimitError: Quelle hat gedrosselt.
            DataValidationError: Die Quelle hat ihr Layout geaendert.
        """
        ...

    def get_tickers(self, index_key: str) -> list[str]:
        """Holt allein die Symbole eines Index.

        Args:
            index_key: Bezeichner des Index, etwa ``sp500``.

        Returns:
            Kanonische Symbole in Quellenreihenfolge.

        Raises:
            ValueError: Wenn der Index unbekannt ist.
            ProviderUnavailableError: Quelle nicht erreichbar, oder der
                Offline-Modus verbietet den Abruf.
            RateLimitError: Quelle hat gedrosselt.
        """
        ...


@runtime_checkable
class ExchangeRateProvider(Protocol):
    """Liefert Wechselkurse fuer die Depotbewertung.

    Die einzige Stelle im Projekt mit Waehrungsumrechnung ist das Depot:
    wer SAP in Euro und Apple in Dollar haelt, bekaeme ohne Umrechnung
    eine Summe aus ungleichen Einheiten - eine Falschaussage ueber sein
    Vermoegen. Analytik, Screening und Backtest rechnen bewusst **nicht**
    um, siehe ``docs/DATA_SOURCES.md``.

    Ein Kurs ist ein Marktkurs, kein Abrechnungskurs. Die Bank rechnet mit
    eigenem Aufschlag ab; die Anzeige ist Orientierung, keine steuerliche
    Bemessungsgrundlage.
    """

    def get_rate(self, base: Currency, quote: Currency, on: date | None = None) -> float:
        """Holt den Kurs eines Waehrungspaares.

        Args:
            base: Ausgangswaehrung, die umgerechnet wird.
            quote: Zielwaehrung, in der das Ergebnis steht.
            on: Handelstag des gesuchten Kurses. ``None`` nimmt den
                aktuellsten verfuegbaren. Faellt das Datum auf ein
                Wochenende oder einen Feiertag, gilt der letzte
                Handelstag davor - die Quelle fuehrt dort keine Zeile.

        Returns:
            Einheiten von ``quote`` je einer Einheit von ``base``. Bei
            gleicher Waehrung exakt ``1.0``, ohne Abruf.

        Raises:
            ProviderUnavailableError: Quelle nicht erreichbar, oder der
                Offline-Modus verbietet den Abruf. **Nicht** auf 1.0
                ausweichen: ein geratener Kurs waere schlimmer als eine
                fehlende Zahl.
            RateLimitError: Quelle hat gedrosselt.
            SymbolNotFoundError: Das Waehrungspaar wird nicht notiert.
        """
        ...


@runtime_checkable
class EarningsCalendarProvider(Protocol):
    """Liefert den naechsten Quartalszahlen-Termin je Instrument.

    Fuellt je eine Spalte in ``PORT`` und ``WL``. Sammelabruf, weil beide
    Seiten eine ganze Positionsliste auf einmal darstellen - auch wenn eine
    Implementierung intern je Ticker einzeln fragen muss.
    """

    def get_earnings_dates(self, tickers: Sequence[str]) -> dict[str, EarningsDate]:
        """Holt die naechsten Termine fuer mehrere Symbole.

        Ein fehlender Termin ist **kein Fehler**, sondern eine gueltige
        Aussage. Die Methode wirft deshalb nicht, wenn ein Symbol keinen
        Termin hat, sondern liefert ein :class:`EarningsDate` mit
        ``next_date=None``:

        * ``has_earnings=False``, ``next_date=None`` - das Instrument fuehrt
          grundsaetzlich keine Quartalszahlen (ETF, Krypto, Index). Endgueltig.
        * ``has_earnings=True``, ``next_date=None`` - eine Aktie, deren naechsten
          Termin die Quelle nicht nennt. Kann sich spaeter fuellen.

        Ein Termin liegt nie in der Vergangenheit: einen abgelaufenen Termin
        als "naechsten" auszugeben waere eine Falschaussage. Er wird zu
        ``next_date=None``.

        Der Ausfall eines einzelnen Symbols bricht den Sammelabruf nicht ab -
        die uebrigen Symbole werden weiter geliefert. Nur ein Symbol, zu dem
        sich ueberhaupt keine Aussage treffen laesst, fehlt im Ergebnis.

        Args:
            tickers: Kanonische Symbole. Duplikate werden entfernt.

        Returns:
            Abbildung Symbol auf Termin. Die **Reihenfolge ist nicht
            garantiert** und die Menge der Schluessel kann kleiner sein als
            ``tickers``; Aufrufer greifen ueber den Schluessel zu, nicht
            ueber die Position.

        Raises:
            ProviderUnavailableError: Quelle fuer **alle** Symbole nicht
                erreichbar.
            RateLimitError: Quelle hat gedrosselt. Wird nie umgangen.
        """
        ...


@runtime_checkable
class QuoteStream(Protocol):
    """Fortlaufender Strom von Kursaktualisierungen.

    Bewusst als Protocol definiert, damit ein echter Push-Feed spaeter ohne
    Aenderung ausserhalb von ``data/`` eingesteckt werden kann. Die
    mitgelieferte Implementierung ``PollingQuoteStream`` pollt lediglich -
    das UI weist den Verzug sichtbar aus.
    """

    def subscribe(self, tickers: Iterable[str]) -> AsyncIterator[Quote]:
        """Abonniert Kursaktualisierungen fuer eine Symbolmenge.

        Der Iterator laeuft, bis er geschlossen wird. Fehler einzelner Polls
        beenden ihn nicht - die Implementierung faellt auf Backoff zurueck
        und liefert weiter, sobald die Quelle wieder antwortet.

        Args:
            tickers: Zu abonnierende Symbole. Duplikate werden entfernt.

        Returns:
            Asynchroner Iterator ueber einzelne Quotes.
        """
        ...

    async def aclose(self) -> None:
        """Beendet den Stream und gibt alle Ressourcen frei."""
        ...


@runtime_checkable
class MarketDataProvider(InstrumentProvider, PriceProvider, FundamentalsProvider, Protocol):
    """Sammelrolle fuer eine Quelle, die Stammdaten, Kurse und Statements liefert.

    ``YFinanceProvider`` erfuellt diese Rolle. Das UI haelt genau eine
    Instanz und reicht sie an die Analytics-Schicht weiter.
    """
