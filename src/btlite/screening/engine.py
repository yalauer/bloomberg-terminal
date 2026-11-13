"""Orchestrierung eines Screening-Laufs.

Die Engine setzt die beiden Wege aus dem Vertrag zusammen:

1. **Remote**: ``ScreenRequest.finviz_filters`` gehen an den injizierten
   :class:`~btlite.data.base.ScreenerProvider` und schraenken das Universum
   bei der Quelle ein.
2. **Universum**: ist ``ScreenRequest.universe`` gesetzt, ist das die
   Startmenge - dann faellt der Remote-Aufruf weg, sofern keine
   Finviz-Filter angegeben sind.
3. **Lokal**: ``ScreenRequest.local_filters`` laufen danach auf dem
   Ergebnis, ausgewertet von :mod:`btlite.screening.filters`.
4. **Sortierung und Kuerzung**: ``sort_by`` sortiert, ``limit`` kuerzt.
   ``total_matches`` wird **vor** dem Kuerzen bestimmt, damit
   ``ScreenResult.is_truncated`` die Wahrheit sagt.

Architekturregel 2: die Engine holt **nie** selbst Daten. Der Provider kommt
per Dependency Injection in den Konstruktor. Damit laeuft die gesamte
Screening-Schicht offline gegen
:class:`~btlite.data.fixtures.FixtureProvider`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pandas as pd

from btlite.core.clock import utc_now
from btlite.core.errors import ScreeningError
from btlite.core.logging import get_logger
from btlite.models.common import DataSource, Provenance
from btlite.models.screening import ScreenRequest, ScreenResult
from btlite.screening.filters import apply_filters

if TYPE_CHECKING:
    from pathlib import Path

    from btlite.data.base import ScreenerProvider

_LOG = get_logger(__name__)

#: Indexname des Ergebnisframes. Der Vertrag verlangt den Ticker als Index;
#: ``data/finviz_provider.py`` setzt denselben Namen.
_TICKER_INDEX_NAME: Final[str] = "ticker"

#: Unterstuetzte Exportformate, abgeleitet aus der Dateiendung.
_CSV_SUFFIXES: Final[frozenset[str]] = frozenset({".csv"})
_PARQUET_SUFFIXES: Final[frozenset[str]] = frozenset({".parquet", ".pq"})


def empty_universe() -> pd.DataFrame:
    """Baut ein leeres Universum mit korrektem Indexnamen.

    Returns:
        Leerer Frame mit Index ``ticker``.
    """
    return pd.DataFrame(index=pd.Index([], name=_TICKER_INDEX_NAME, dtype=object))


def universe_frame(tickers: list[str]) -> pd.DataFrame:
    """Baut ein Universum aus einer expliziten Tickerliste.

    Der Frame hat keine Kennzahlenspalten. Lokale Filter auf Kennzahlen
    laufen deshalb in eine :class:`~btlite.core.errors.UnknownFilterError` -
    das ist beabsichtigt: ohne angereicherte Daten kann die Engine keine
    Kennzahl pruefen und soll das melden statt still nichts zu treffen.

    Args:
        tickers: Kanonische Symbole. Duplikate werden entfernt, die
            Reihenfolge des ersten Auftretens bleibt erhalten.

    Returns:
        Frame ohne Spalten, Index = Ticker.
    """
    seen: dict[str, None] = dict.fromkeys(str(t) for t in tickers)
    return pd.DataFrame(index=pd.Index(list(seen), name=_TICKER_INDEX_NAME, dtype=object))


def sort_frame(frame: pd.DataFrame, sort_by: str | None, *, ascending: bool) -> pd.DataFrame:
    """Sortiert das Ergebnis, sofern eine Sortierspalte gesetzt ist.

    Fehlwerte landen unabhaengig von der Richtung **immer am Ende**. Eine
    Zeile ohne Wert soll eine Rangliste nicht anfuehren.

    Args:
        frame: Zu sortierender Frame.
        sort_by: Spaltenname. ``None`` laesst die Reihenfolge unveraendert.
        ascending: True sortiert aufsteigend.

    Returns:
        Sortierte Kopie, oder der unveraenderte Frame wenn ``sort_by`` None
        ist. Eine unbekannte Spalte wird geloggt und ignoriert, weil das
        Ergebnis fachlich korrekt bleibt und nur die Reihenfolge betrifft.
    """
    if sort_by is None:
        return frame
    if sort_by == _TICKER_INDEX_NAME:
        return frame.sort_index(ascending=ascending)
    if sort_by not in frame.columns:
        _LOG.warning("Sortierspalte %r nicht im Ergebnis - Reihenfolge unveraendert", sort_by)
        return frame
    return frame.sort_values(by=sort_by, ascending=ascending, na_position="last", kind="stable")


class ScreeningEngine:
    """Fuehrt Screening-Anfragen aus, remote und lokal kombiniert.

    Der Provider wird injiziert (Architekturregel 2). Die Engine ist
    zustandslos ueber Aufrufe hinweg - ``screen`` veraendert nichts an ihr.
    """

    def __init__(self, provider: ScreenerProvider | None = None) -> None:
        """Initialisiert die Engine.

        Args:
            provider: Quelle fuer die Remote-Filter. ``None`` ist zulaessig,
                solange jede Anfrage ein explizites ``universe`` mitbringt
                oder ein bereits angereichertes Universum uebergeben wird.
        """
        self._provider = provider

    @property
    def provider(self) -> ScreenerProvider | None:
        """Der injizierte Screener-Provider.

        Returns:
            Der Provider, oder None wenn die Engine ohne einen laeuft.
        """
        return self._provider

    def screen(
        self,
        request: ScreenRequest,
        *,
        universe: pd.DataFrame | None = None,
    ) -> ScreenResult:
        """Fuehrt eine Screening-Anfrage vollstaendig aus.

        Ablauf: Startuniversum bestimmen, lokale Praedikate anwenden,
        sortieren, ``total_matches`` festhalten, dann auf ``limit`` kuerzen.

        Args:
            request: Die Anfrage.
            universe: Bereits angereichertes Startuniversum, Index = Ticker,
                Spalten = Kennzahlen in den Einheiten aus
                ``analytics/ratios.py`` (Renditen und Margen dezimal,
                0.12 = 12 Prozent). Hat Vorrang vor Provider und
                ``request.universe`` - so laufen lokale Kennzahlenfilter
                auf gecachten Fundamentals, ohne dass die Engine Daten holt.

        Returns:
            Ergebnis mit ``total_matches`` **vor** der Kuerzung.

        Raises:
            ScreeningError: Wenn kein Startuniversum ermittelbar ist.
            UnknownFilterError: Wenn ein lokales Praedikat eine Spalte
                nennt, die das Universum nicht fuehrt.
            ProviderUnavailableError: Quelle nicht erreichbar.
            RateLimitError: Quelle hat gedrosselt.
        """
        frame, provenance = self._resolve_universe(request, universe)
        frame = apply_filters(frame, request.local_filters)
        frame = sort_frame(frame, request.sort_by, ascending=request.ascending)

        total = len(frame)
        return ScreenResult(
            request=request,
            data=frame.head(request.limit),
            total_matches=total,
            provenance=provenance,
        )

    def _resolve_universe(
        self,
        request: ScreenRequest,
        universe: pd.DataFrame | None,
    ) -> tuple[pd.DataFrame, Provenance]:
        """Bestimmt das Startuniversum und dessen Herkunft.

        Vorrang: uebergebener Frame, dann Remote-Filter beim Provider, dann
        ``request.universe``. Sind Finviz-Filter **und** ``request.universe``
        gesetzt, holt die Engine remote und schneidet das Ergebnis auf die
        genannten Ticker - die Angabe wirkt dann als Whitelist.

        Args:
            request: Die Anfrage.
            universe: Optional vorbereitetes Universum.

        Returns:
            Paar aus Startframe und Herkunftsnachweis.

        Raises:
            ScreeningError: Wenn weder Frame noch Provider noch
                ``request.universe`` ein Universum liefern.
            ProviderUnavailableError: Quelle nicht erreichbar.
            RateLimitError: Quelle hat gedrosselt.
        """
        if universe is not None:
            return universe, self._local_provenance("Supplied universe.")

        if request.finviz_filters:
            if self._provider is None:
                msg = "Finviz filters require the engine to have a screener provider."
                raise ScreeningError(msg, detail="provider=None, finviz_filters gesetzt")
            remote = self._provider.screen(request)
            frame = remote.data
            if request.universe:
                allowed = [str(t) for t in request.universe]
                frame = frame.loc[frame.index.isin(allowed)]
            return frame, remote.provenance

        if request.universe:
            return universe_frame([str(t) for t in request.universe]), self._local_provenance(
                "Explizites Startuniversum aus der Anfrage."
            )

        msg = "The request names neither a universe nor any Finviz filters."
        raise ScreeningError(
            msg, detail="universe leer, finviz_filters leer, kein Frame uebergeben"
        )

    @staticmethod
    def _local_provenance(note: str) -> Provenance:
        """Baut den Herkunftsnachweis fuer ein lokal bestimmtes Ergebnis.

        Args:
            note: Freitext-Hinweis fuer das UI.

        Returns:
            Herkunftsnachweis mit Quelle ``computed``.
        """
        return Provenance(
            source=DataSource.COMPUTED,
            fetched_at=utc_now(),
            from_cache=False,
            delay_minutes=None,
            note=note,
        )


def export_result(result: ScreenResult, path: Path) -> Path:
    """Schreibt ein Screening-Ergebnis nach CSV oder Parquet.

    Das Format ergibt sich aus der Dateiendung: ``.csv`` schreibt CSV,
    ``.parquet`` und ``.pq`` schreiben Parquet. Der Ticker-Index wird in
    beiden Faellen mitgeschrieben, damit ein Round-Trip ihn wiederherstellt.

    Args:
        result: Zu exportierendes Ergebnis. Exportiert wird ``result.data``,
            also die bereits auf ``limit`` gekuerzte Treffermenge.
        path: Zieldatei. Fehlende Elternverzeichnisse werden angelegt.

    Returns:
        Der geschriebene Pfad.

    Raises:
        ScreeningError: Bei unbekannter Dateiendung oder wenn das Schreiben
            fehlschlaegt (etwa fehlende Schreibrechte).
    """
    suffix = path.suffix.lower()
    if suffix not in _CSV_SUFFIXES | _PARQUET_SUFFIXES:
        msg = f"Unknown export format {path.suffix!r}."
        raise ScreeningError(
            msg,
            detail=f"zulaessig: {sorted(_CSV_SUFFIXES | _PARQUET_SUFFIXES)}",
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = result.data
    if frame.index.name is None:
        frame = frame.rename_axis(_TICKER_INDEX_NAME)

    try:
        if suffix in _CSV_SUFFIXES:
            frame.to_csv(path, index=True, encoding="utf-8")
        else:
            frame.to_parquet(path, index=True)
    except OSError as exc:
        msg = "The result could not be written."
        raise ScreeningError(msg, detail=f"{path}: {exc}") from exc
    return path


def load_export(path: Path) -> pd.DataFrame:
    """Liest eine exportierte Trefferliste zurueck ein.

    Gegenstueck zu :func:`export_result`, vor allem fuer Round-Trip-Tests
    und fuer den Import eines gespeicherten Screens ins UI.

    Args:
        path: Quelldatei, ``.csv``, ``.parquet`` oder ``.pq``.

    Returns:
        Frame mit Ticker-Index.

    Raises:
        ScreeningError: Bei unbekannter Dateiendung oder unlesbarer Datei.
    """
    suffix = path.suffix.lower()
    if suffix not in _CSV_SUFFIXES | _PARQUET_SUFFIXES:
        msg = f"Unknown import format {path.suffix!r}."
        raise ScreeningError(
            msg,
            detail=f"zulaessig: {sorted(_CSV_SUFFIXES | _PARQUET_SUFFIXES)}",
        )
    try:
        frame = pd.read_csv(path, index_col=0) if suffix in _CSV_SUFFIXES else pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        msg = "The file could not be read."
        raise ScreeningError(msg, detail=f"{path}: {exc}") from exc
    frame.index.name = _TICKER_INDEX_NAME
    return frame
