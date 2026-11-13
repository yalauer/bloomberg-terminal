"""Finviz-Screener-Provider auf Basis von ``finvizfinance`` 1.5.

Erfuellt :class:`~btlite.data.base.ScreenerProvider`.

Nutzungsrahmen: ``finvizfinance`` ist ein HTML-Scraper. Das Projekt bleibt im
Rahmen privater Nutzung mit konservativem Throttling (Default 0.5 Requests pro
Sekunde, Burst 2). Es gibt **keine** Umgehung von Rate Limits, Paywalls oder
Bot-Erkennung. Blockt die Quelle, wird der Fehler als
:class:`~btlite.core.errors.RateLimitError` sauber nach oben gegeben.

Filter-Notation: die Keys sind exakt inklusive Punkt und Gross-/Kleinschreibung
zu schreiben - ``Market Cap.`` mit Punkt ist gueltig, ``Market Cap`` ohne Punkt
wirft :class:`~btlite.core.errors.UnknownFilterError`. Sowohl Keys als auch
Werte werden gegen das Dictionary des Pakets validiert, nicht gegen eine
hier gepflegte Kopie: :meth:`FinvizProvider.available_filters` liest sie zur
Laufzeit aus ``finvizfinance.constants.filter_dict`` aus.
"""

from __future__ import annotations

import difflib
import json
from typing import Any, Final

import pandas as pd

from btlite.config import Settings, get_settings
from btlite.core.cache import DataFrameCache, cache_key
from btlite.core.clock import utc_now
from btlite.core.errors import (
    DataValidationError,
    ProviderUnavailableError,
    RateLimitError,
    UnknownFilterError,
)
from btlite.core.logging import get_logger
from btlite.core.ratelimit import TokenBucket, get_registry, retry_with_backoff
from btlite.models.common import DataSource, Provenance, normalize_ticker
from btlite.models.screening import ScreenRequest, ScreenResult

_LOG = get_logger(__name__)

#: Zeilen je Screener-Seite bei Finviz. Bestimmt, wie viele Seiten ein
#: ``limit`` kostet - jede Seite ist ein eigener Request.
_ROWS_PER_PAGE: Final[int] = 20

#: Spaltenname der Ticker-Spalte in den Screener-Tabellen.
_TICKER_COLUMN: Final[str] = "Ticker"

#: Verfuegbare Views. Alle teilen dieselbe Methodensignatur und
#: unterscheiden sich nur in den gelieferten Spalten.
_VIEWS: Final[tuple[str, ...]] = (
    "Overview",
    "Valuation",
    "Financial",
    "Ownership",
    "Performance",
    "Technical",
    "Ticker",
)


def _load_filter_dict() -> dict[str, dict[str, Any]]:
    """Liest das Filter-Dictionary aus dem installierten Paket.

    Bewusst kein Hardcoding: die zulaessigen Keys und Werte aendern sich mit
    dem Paket und mit Finviz selbst. Eine hier gepflegte Kopie waere sofort
    veraltet und wuerde gueltige Filter faelschlich ablehnen.

    Returns:
        Abbildung Filter-Key auf das Rohdictionary des Pakets mit den
        Schluesseln ``prefix`` und ``option``.

    Raises:
        ProviderUnavailableError: Wenn ``finvizfinance`` nicht installiert ist.
    """
    try:
        from finvizfinance.constants import filter_dict
    except ImportError as exc:  # pragma: no cover - Paket ist Pflichtabhaengigkeit
        msg = "The finvizfinance package is not installed."
        raise ProviderUnavailableError(msg, detail=str(exc)) from exc
    return dict(filter_dict)


def _map_exception(exc: Exception) -> Exception:
    """Bildet eine rohe finvizfinance-Exception auf die Projekthierarchie ab.

    Args:
        exc: Die vom Paket geworfene Exception.

    Returns:
        Der passende Fehler aus :mod:`btlite.core.errors`. Bereits
        abgebildete Fehler werden unveraendert durchgereicht.
    """
    from btlite.core.errors import BTLiteError

    if isinstance(exc, BTLiteError):
        return exc

    from finvizfinance.exceptions import FinvizBlockedError, FinvizParseError

    if isinstance(exc, FinvizBlockedError):
        # Eine Wall ist eine Aussage der Quelle, kein transienter Fehler.
        # Sie wird nach oben gereicht, nicht umgangen.
        return RateLimitError(
            "Finviz is currently blocking requests from this address.",
            detail=f"{type(exc).__name__}: {exc}",
            retry_after_s=None,
        )
    if isinstance(exc, FinvizParseError):
        # Layoutaenderung bei Finviz. Erscheint als Fehler, nicht als still
        # leeres Ergebnis.
        return DataValidationError(
            "Finviz changed its page layout, the screener can no longer read it.",
            detail=f"{type(exc).__name__}: {exc}",
        )
    if isinstance(exc, ValueError):
        # validate_choice des Pakets meldet unbekannte Keys und Werte so.
        return UnknownFilterError(
            "A filter is unknown to Finviz.",
            detail=f"{type(exc).__name__}: {exc}",
        )
    return ProviderUnavailableError(
        "Finviz is currently unreachable.",
        detail=f"{type(exc).__name__}: {exc}",
    )


def normalize_screen_table(raw: pd.DataFrame | None) -> pd.DataFrame:
    """Normalisiert eine Finviz-Screener-Tabelle auf den Projektvertrag.

    Der Vertrag :class:`~btlite.models.screening.ScreenResult` verlangt den
    Ticker als Index. Prozentwerte hat das Paket bereits dezimal umgerechnet
    (``number_convert``), sie werden hier nicht angefasst.

    Args:
        raw: Rohtabelle des Pakets. ``None`` bedeutet: keine Treffer.

    Returns:
        Frame mit Ticker-Index, Indexname ``ticker``. Bei keinen Treffern ein
        leerer Frame mit demselben Indexnamen.

    Raises:
        DataValidationError: Wenn die Ticker-Spalte fehlt - das bedeutet eine
            Layoutaenderung bei Finviz und wird nicht als still leeres
            Ergebnis verschwiegen.
    """
    if raw is None or raw.empty:
        return pd.DataFrame(index=pd.Index([], name=_TICKER_COLUMN.lower()))

    out = raw.copy()
    if _TICKER_COLUMN not in out.columns:
        msg = "Finviz result without a ticker column."
        raise DataValidationError(msg, detail=f"Spalten={list(out.columns)}")

    out[_TICKER_COLUMN] = [normalize_ticker(str(t)) for t in out[_TICKER_COLUMN]]
    out = out.drop_duplicates(subset=[_TICKER_COLUMN], keep="first")
    out = out.set_index(_TICKER_COLUMN)
    out.index.name = _TICKER_COLUMN.lower()
    return out


class FinvizProvider:
    """Screening ueber Finviz.

    Erfuellt :class:`~btlite.data.base.ScreenerProvider` strukturell, ohne von
    dem Protocol zu erben.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        cache: DataFrameCache | None = None,
        bucket: TokenBucket | None = None,
        view: str = "Overview",
    ) -> None:
        """Initialisiert den Provider.

        Args:
            settings: Projekt-Settings. ``None`` nimmt die globalen.
            cache: DataFrame-Cache. ``None`` baut einen aus den Settings.
            bucket: Token-Bucket. ``None`` nimmt den prozessweiten
                finviz-Bucket aus der Registry.
            view: Screener-View, siehe :data:`_VIEWS`.

        Raises:
            ValueError: Wenn ``view`` unbekannt ist.
            ProviderUnavailableError: Wenn der Offline-Modus aktiv ist. Dann
                ist :class:`~btlite.data.fixtures.FixtureProvider` zu nutzen.
        """
        if view not in _VIEWS:
            msg = f"Unknown view {view!r}. Allowed: {list(_VIEWS)}"
            raise ValueError(msg)

        self._settings = settings if settings is not None else get_settings()
        if self._settings.data.offline:
            msg = "Offline mode: Finviz is blocked, use the FixtureProvider."
            raise ProviderUnavailableError(msg, detail="BTLITE_DATA_OFFLINE=true")

        self._cache = cache if cache is not None else DataFrameCache(self._settings.cache)
        self._bucket = bucket if bucket is not None else get_registry().bucket("finviz")
        self._view = view
        self._retry = retry_with_backoff(settings=self._settings.ratelimit)

    @property
    def view(self) -> str:
        """Die konfigurierte Screener-View.

        Returns:
            Name der View, etwa ``"Overview"``.
        """
        return self._view

    # ------------------------------------------------------------ Filter --
    def available_filters(self) -> dict[str, list[str]]:
        """Listet die von Finviz unterstuetzten Filter.

        Die Keys und Werte stammen zur Laufzeit aus dem installierten Paket,
        sie sind hier nicht hardcodiert. Zu beachten: die Werte sind die
        vollen Anzeigetexte der Quelle, also etwa
        ``"Mega ($200bln and more)"``, nicht die Kurzform ``"Mega"``.

        Returns:
            Abbildung Filter-Key auf die Liste zulaessiger Werte.

        Raises:
            ProviderUnavailableError: Wenn ``finvizfinance`` fehlt.
        """
        return {key: list(spec["option"]) for key, spec in _load_filter_dict().items()}

    def validate_filters(self, filters: dict[str, str]) -> None:
        """Prueft Filter-Keys und -Werte gegen das Dictionary des Pakets.

        Args:
            filters: Filter als Key-Value-Paare in der Notation der Quelle.

        Raises:
            UnknownFilterError: Wenn ein Key oder ein Wert unbekannt ist. Die
                Meldung nennt den naechstliegenden gueltigen Kandidaten -
                der haeufigste Fehler ist ein fehlender Punkt, etwa
                ``Market Cap`` statt ``Market Cap.``.
            ProviderUnavailableError: Wenn ``finvizfinance`` fehlt.
        """
        spec = _load_filter_dict()
        for key, value in filters.items():
            if key not in spec:
                hint = difflib.get_close_matches(key, list(spec), n=3, cutoff=0.6)
                msg = f"Filter {key!r} is unknown to Finviz."
                raise UnknownFilterError(
                    msg,
                    detail=f"Meinten Sie: {hint}" if hint else f"gueltige Keys: {sorted(spec)}",
                )
            options = spec[key]["option"]
            if value not in options:
                hint = difflib.get_close_matches(value, list(options), n=3, cutoff=0.5)
                msg = f"Value {value!r} is not allowed for filter {key!r}."
                raise UnknownFilterError(
                    msg,
                    detail=(f"Meinten Sie: {hint}" if hint else f"gueltige Werte: {list(options)}"),
                )

    # --------------------------------------------------- ScreenerProvider --
    def screen(self, request: ScreenRequest) -> ScreenResult:
        """Fuehrt die Finviz-Filter der Anfrage bei der Quelle aus.

        Nur ``request.finviz_filters`` wird remote ausgewertet. Lokale
        Praedikate wendet ``screening/engine.py`` danach an.

        Zum Laufzeitverhalten: Finviz liefert 20 Zeilen je Seite und das Paket
        wartet ``sleep_sec`` Sekunden zwischen den Seiten. Ein ``limit`` von
        200 kostet damit rund 10 Requests und ebenso viele Sekunden. Der
        Token-Bucket kommt zusaetzlich obendrauf.

        Args:
            request: Screening-Anfrage. ``limit`` wird respektiert.

        Returns:
            Treffer der Quelle. Index ist der Ticker.

        Raises:
            ProviderUnavailableError: Quelle nicht erreichbar.
            RateLimitError: Quelle hat gedrosselt oder blockt.
            UnknownFilterError: Ein Filter-Key oder -Wert ist unbekannt.
            DataValidationError: Layoutaenderung bei Finviz.
        """
        self.validate_filters(request.finviz_filters)

        key = cache_key(
            "finviz",
            "screen",
            self._view,
            sorted(request.finviz_filters.items()),
            request.limit,
            request.sort_by,
            request.ascending,
        )

        def _load() -> pd.DataFrame:
            return self._fetch(request)

        df, from_cache = self._cache.cached_df(key, self._settings.cache.screener_ttl_s, _load)
        return ScreenResult(
            request=request,
            data=df,
            total_matches=len(df),
            provenance=Provenance(
                source=DataSource.FINVIZ,
                fetched_at=utc_now(),
                from_cache=from_cache,
                delay_minutes=None,
                note="Finviz updates on a daily basis.",
            ),
        )

    def _fetch(self, request: ScreenRequest) -> pd.DataFrame:
        """Holt die Trefferliste von Finviz.

        Args:
            request: Screening-Anfrage.

        Returns:
            Normalisierter Frame, Index = Ticker.

        Raises:
            ProviderUnavailableError: Quelle nicht erreichbar.
            RateLimitError: Quelle hat gedrosselt oder blockt.
            UnknownFilterError: Ein Filter ist unbekannt.
            DataValidationError: Layoutaenderung bei Finviz.
        """

        def _fetch_once() -> pd.DataFrame:
            screener = self._make_screener()
            # Der erste Request wird vorab bezahlt. Erkennt der alte Parser
            # das Layout, wartet das Paket selbst zwischen Folgeseiten; beim
            # modernen Fallback nimmt jede Seite unten ein eigenes Token.
            self._bucket.acquire()
            try:
                screener.set_filter(filters_dict=dict(request.finviz_filters))
                raw = screener.screener_view(
                    order=request.sort_by if request.sort_by else "Ticker",
                    limit=request.limit,
                    ascend=request.ascending,
                    verbose=0,
                )
            except Exception as exc:
                raise _map_exception(exc) from exc
            if raw is None:
                return self._fetch_modern(screener, request)
            return self._normalize(raw)

        return self._retry(_fetch_once)()

    def _fetch_modern(self, screener: Any, request: ScreenRequest) -> pd.DataFrame:
        """Liest das seit 2026 eingebettete JSON des neuen Finviz-Screeners.

        ``finvizfinance`` 1.5 liefert beim neuen ``div.fv-table``-Layout
        ``None``. Fuer die Remote-Vorauswahl werden mindestens die Ticker
        benoetigt; Text- und Zahlfelder werden zusaetzlich uebernommen, damit
        der Provider auch ausserhalb der Vorauswahl brauchbar bleibt.

        Args:
            screener: Bereits konfigurierter finvizfinance-Screener.
            request: Urspruengliche Screening-Anfrage.

        Returns:
            Normalisierte Tabelle mit Ticker-Index.

        Raises:
            DataValidationError: Wenn der JSON-Payload fehlt oder ungueltig ist.
        """
        from bs4 import BeautifulSoup
        from finvizfinance.util import fetch

        params = dict(screener.request_params)
        params.pop("r", None)
        rows: list[dict[str, Any]] = []
        start = 1
        total: int | None = None
        page_size = _ROWS_PER_PAGE

        while total is None or len(rows) < min(total, request.limit):
            self._bucket.acquire()
            page_params = {**params, "r": start}
            try:
                response = fetch(screener.url, page_params)
                soup = BeautifulSoup(response.text, "lxml")
                script = soup.find("script", attrs={"type": "application/json"})
                if script is None or script.string is None:
                    raise KeyError("script[type='application/json']")
                payload = json.loads(script.string)["data"]
                page_rows = payload["rows"]
                page_total = int(payload["totalCount"])
                page_size = int(payload["pageSize"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                msg = "Finviz changed its page layout, the screener payload is invalid."
                raise DataValidationError(msg, detail=f"{type(exc).__name__}: {exc}") from exc

            if not isinstance(page_rows, list) or page_size <= 0:
                msg = "Finviz changed its page layout, the screener rows are invalid."
                raise DataValidationError(
                    msg, detail=f"rows={type(page_rows).__name__}, page_size={page_size}"
                )
            if total is None:
                total = page_total
            elif total != page_total:
                msg = "Finviz changed the match count while pages were loaded."
                raise DataValidationError(msg, detail=f"first={total}, current={page_total}")
            if not page_rows and len(rows) < min(total, request.limit):
                msg = "Finviz returned an empty page before all matches were loaded."
                raise DataValidationError(msg, detail=f"loaded={len(rows)}, total={total}")

            for raw_row in page_rows:
                text = raw_row.get("text", {}) if isinstance(raw_row, dict) else {}
                numbers = raw_row.get("numbers", {}) if isinstance(raw_row, dict) else {}
                ticker = text.get("ticker")
                if ticker:
                    rows.append(
                        {
                            "Ticker": ticker,
                            "Company": text.get("company"),
                            "Sector": text.get("sector"),
                            **numbers,
                        }
                    )
            start += page_size

        return normalize_screen_table(pd.DataFrame(rows[: request.limit]))

    def _make_screener(self) -> Any:
        """Instanziiert die Screener-Klasse der konfigurierten View.

        Returns:
            Die Screener-Instanz des Pakets.

        Raises:
            ProviderUnavailableError: Wenn ``finvizfinance`` fehlt.
        """
        try:
            from finvizfinance.screener import (
                financial,
                overview,
                ownership,
                performance,
                technical,
                ticker,
                valuation,
            )
        except ImportError as exc:  # pragma: no cover - Pflichtabhaengigkeit
            msg = "The finvizfinance package is not installed."
            raise ProviderUnavailableError(msg, detail=str(exc)) from exc

        factories: dict[str, Any] = {
            "Overview": overview.Overview,
            "Valuation": valuation.Valuation,
            "Financial": financial.Financial,
            "Ownership": ownership.Ownership,
            "Performance": performance.Performance,
            "Technical": technical.Technical,
            "Ticker": ticker.Ticker,
        }
        return factories[self._view]()

    @staticmethod
    def _normalize(raw: pd.DataFrame | None) -> pd.DataFrame:
        """Normalisiert die Screener-Tabelle auf den Projektvertrag.

        Args:
            raw: Rohtabelle des Pakets. ``None`` bedeutet: keine Treffer.

        Returns:
            Frame mit Ticker-Index. Bei keinen Treffern leer.

        Raises:
            DataValidationError: Wenn die Ticker-Spalte fehlt.
        """
        return normalize_screen_table(raw)
