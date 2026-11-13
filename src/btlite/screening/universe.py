"""Anreicherung eines Ticker-Universums um Kennzahlen.

Das Bindeglied zwischen Datenschicht und Screening. Ohne dieses Modul laufen
lokale Praedikate ins Leere: ein Finviz-Ergebnis traegt Spalten wie ``P/E``
und ``Market Cap``, die lokale Filter-DSL erwartet dagegen die Schluessel aus
:mod:`btlite.analytics.ratios` (``pe``, ``roe``, ``current_ratio``) in deren
Einheiten - Margen und Renditen dezimal.

Architekturregel 2: dieses Modul holt keine Daten selbst. Provider werden
injiziert.

Typische Verwendung::

    universe = build_ratio_universe(provider, ["AAPL", "MSFT"])
    result = engine.screen(request, universe=universe)
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING

import pandas as pd

from btlite.analytics.ratios import compute_ratios
from btlite.analytics.statements import normalize_bundle
from btlite.core.errors import DataError, EmptyUniverseError
from btlite.core.logging import get_logger
from btlite.models.common import Period

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from btlite.data.base import MarketDataProvider

_LOG = get_logger(__name__)

#: Abstand zwischen zwei Fortschrittsmeldungen, in Symbolen. Frueher 10:
#: bei gemessen rund drei Sekunden je Titel im kalten Lauf stand die Anzeige
#: dann bis zu dreissig Sekunden still und wirkte haengengeblieben. Eine
#: Meldung setzt nur zwei Zahlen unter einer Sperre - sie bremst nichts.
PROGRESS_EVERY = 1

#: Indexname des Universums. Muss zu ``engine._TICKER_INDEX_NAME`` passen.
TICKER_INDEX_NAME = "ticker"

#: Stammdatenspalten, die zusaetzlich zu den Kennzahlen bereitstehen.
#: Sie erlauben Praedikate auf Sektor, Branche oder Marktkapitalisierung.
METADATA_COLUMNS = ("name", "sector", "industry", "country", "market_cap", "price")


def build_ratio_universe(
    provider: MarketDataProvider,
    tickers: Sequence[str],
    *,
    period: Period = Period.ANNUAL,
    with_quotes: bool = True,
    skip_failures: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    max_workers: int = 1,
) -> pd.DataFrame:
    """Baut ein Universum aus Kennzahlen und Stammdaten.

    Fuer jedes Symbol werden Statements geholt, normalisiert und in
    Kennzahlen umgerechnet. Das Ergebnis ist genau die Matrix, die
    :meth:`~btlite.screening.engine.ScreeningEngine.screen` als ``universe``
    erwartet.

    Fehlt eine Kennzahl fuer ein Symbol, steht dort ``NaN``. Die Filter-DSL
    behandelt einen Fehlwert konsistent: er erfuellt kein Vergleichspraedikat,
    wird aber von ``IS_NULL`` getroffen.

    Args:
        provider: Quelle fuer Statements, Stammdaten und Quotes.
        tickers: Zu betrachtende Symbole.
        period: Basis der Kennzahlen. ``ANNUAL`` ist der belastbarste Wert.
        with_quotes: Aktuelle Kurse mitziehen, damit Bewertungskennzahlen wie
            KGV und Dividendenrendite berechenbar sind. Kostet einen
            zusaetzlichen Request fuer alle Symbole zusammen.
        skip_failures: Symbole, deren Daten fehlen, uebergehen statt den
            gesamten Aufbau scheitern zu lassen. Bei ``False`` schlaegt der
            erste Fehler durch.
        should_stop: Optionales Praedikat. Liefert es True, bricht der
            Aufbau am naechsten Symbol ab und gibt zurueck, was bis dahin
            zusammengekommen ist. Bewusst getrennt von ``on_progress``:
            ein Abbruch ist kein gescheiterter Fortschritt, und der
            Fortschritts-Rueckruf schluckt Ausnahmen absichtlich.
        on_progress: Optionaler Rueckruf ``(erledigt, gesamt)``. Gemeldet wird
            jedes ``PROGRESS_EVERY``-te Symbol und immer das letzte, sodass die
            Anzeige mit ``(gesamt, gesamt)`` schliesst statt kurz davor stehen
            zu bleiben. Uebersprungene Symbole zaehlen als erledigt - der Balken
            misst den Fortschritt, nicht die Trefferzahl. Der Fortschritt ist
            Beiwerk: wirft der Rueckruf, wird das einmal geloggt und danach
            nichts mehr gemeldet; der Aufbau laeuft unveraendert zu Ende.
            ``None`` meldet nichts.
        max_workers: Parallele Titelabrufe. Der Provider-Token-Bucket bleibt
            die verbindliche Obergrenze der Request-Rate.

    Returns:
        DataFrame mit Ticker als Index, Kennzahlen- und Stammdatenspalten.
        Leer, wenn kein Symbol Daten lieferte.

    Raises:
        DataError: Wenn ``skip_failures`` False ist und ein Abruf scheitert.
    """
    symbols = [t.strip().upper() for t in tickers if t.strip()]
    if not symbols:
        return _empty_frame()

    prices: dict[str, float] = {}
    if with_quotes:
        try:
            prices = {t: q.price for t, q in provider.get_quotes(symbols).items()}
        except DataError:
            if not skip_failures:
                raise
            _LOG.warning("Quotes nicht verfuegbar - Bewertungskennzahlen bleiben leer")

    total = len(symbols)
    report = progress_reporter(on_progress)
    rows: dict[str, dict[str, float | str | None]] = {}

    def load(symbol: str) -> dict[str, float | str | None]:
        return _row_for(provider, symbol, period=period, price=prices.get(symbol))

    def accept(symbol: str, future: Future[dict[str, float | str | None]]) -> None:
        try:
            rows[symbol] = future.result()
        except DataError as exc:
            if not skip_failures:
                raise
            _LOG.info("Symbol %s uebergangen: %s", symbol, exc.user_message)

    workers = max(1, min(max_workers, total))
    if workers == 1:
        for done, symbol in enumerate(symbols, start=1):
            if should_stop is not None and should_stop():
                _LOG.info("Aufbau nach %d von %d Symbolen abgebrochen", done - 1, total)
                break
            future: Future[dict[str, float | str | None]] = Future()
            try:
                future.set_result(load(symbol))
            except Exception as exc:
                future.set_exception(exc)
            accept(symbol, future)
            report(done, total)
    else:
        iterator = iter(symbols)
        done_count = 0
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="btlite-ratios") as pool:
            pending: dict[Future[dict[str, float | str | None]], str] = {}

            def submit_next() -> bool:
                if should_stop is not None and should_stop():
                    return False
                try:
                    symbol = next(iterator)
                except StopIteration:
                    return False
                pending[pool.submit(load, symbol)] = symbol
                return True

            for _ in range(workers):
                if not submit_next():
                    break
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    symbol = pending.pop(future)
                    accept(symbol, future)
                    done_count += 1
                    report(done_count, total)
                    submit_next()

    if not rows:
        # Ein leerer Frame liefe weiter in die Filter und meldete dort
        # eine fehlende Spalte - eine Aussage ueber das falsche Problem.
        # Das gilt auch fuer ein von vornherein leeres Universum: auch
        # dann hat der Nutzer nichts falsch gemacht und soll es lesbar
        # erfahren.
        _LOG.warning("Kein Symbol lieferte Daten - Universum ist leer")
        msg = "No data arrived for any title."
        raise EmptyUniverseError(msg, detail=f"angefragt={len(symbols)}")

    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = TICKER_INDEX_NAME
    return frame


def progress_reporter(
    on_progress: Callable[[int, int], None] | None,
) -> Callable[[int, int], None]:
    """Kapselt einen Fortschritts-Rueckruf so, dass er nie den Lauf abbricht.

    Wirft der Rueckruf, wird das einmal geloggt und danach nichts mehr
    gemeldet. Ein defekter Rueckruf wuerde sonst bei jedem Symbol erneut ins
    Log schreiben und es fluten.

    Args:
        on_progress: Rueckruf ``(erledigt, gesamt)``, oder None.

    Returns:
        Aufrufbares ``(erledigt, gesamt) -> None``, das nie wirft. Bei ``None``
        tut es nichts.
    """
    if on_progress is None:
        return lambda done, total: None

    broken = False

    def report(done: int, total: int) -> None:
        nonlocal broken
        if broken:
            return
        try:
            on_progress(done, total)
        except Exception:
            broken = True
            _LOG.warning("Fortschrittsanzeige scheitert - Lauf geht ohne sie weiter", exc_info=True)

    return report


def _row_for(
    provider: MarketDataProvider,
    symbol: str,
    *,
    period: Period,
    price: float | None,
) -> dict[str, float | str | None]:
    """Baut die Zeile eines einzelnen Symbols.

    Args:
        provider: Quelle fuer Statements und Stammdaten.
        symbol: Kanonisches Symbol.
        period: Basis der Kennzahlen.
        price: Aktueller Kurs, oder None.

    Returns:
        Abbildung Spaltenname auf Wert.

    Raises:
        DataError: Wenn der Abruf scheitert.
    """
    bundle = normalize_bundle(provider.get_statements(symbol, period=period))

    market_cap: float | None = None
    dividend_rate: float | None = None
    row: dict[str, float | str | None] = {}
    try:
        instrument = provider.get_instrument(symbol)
    except DataError:
        _LOG.debug("Keine Stammdaten fuer %s", symbol)
    else:
        market_cap = instrument.market_cap
        dividend_rate = instrument.dividend_rate
        row |= {
            "name": instrument.name,
            "sector": instrument.sector,
            "industry": instrument.industry,
            "country": instrument.country,
            "market_cap": instrument.market_cap,
        }

    row["price"] = price

    ratios = compute_ratios(
        bundle,
        price=price,
        market_cap=market_cap,
        dividend_per_share=dividend_rate,
    )
    for key, ratio in ratios.ratios.items():
        # None wird zu NaN, damit der Frame numerisch bleibt. Die Filter-DSL
        # unterscheidet ohnehin nur zwischen "vorhanden" und "fehlt".
        row[key] = ratio.value
    return row


def _empty_frame() -> pd.DataFrame:
    """Liefert ein leeres Universum mit korrektem Indexnamen.

    Returns:
        Leerer DataFrame, dessen Index ``ticker`` heisst.
    """
    frame = pd.DataFrame()
    frame.index.name = TICKER_INDEX_NAME
    return frame
