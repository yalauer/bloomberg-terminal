r"""Zieht die eingefrorenen API-Antworten neu und legt sie in ``tests/fixtures``.

**Dieses Skript laeuft NIE automatisch in der Testsuite.** Es ist das einzige
Stueck Code im Repository, das absichtlich ins Netz geht. Die Tests lesen
ausschliesslich die hier erzeugten Dateien - ``tests/conftest.py`` blockiert
Sockets prozessweit, ein versehentlicher Netzwerkzugriff schlaegt dort laut
fehl.

Aufruf::

    .venv\\Scripts\\python.exe scripts\\refresh_fixtures.py
    .venv\\Scripts\\python.exe scripts\\refresh_fixtures.py --only prices
    .venv\\Scripts\\python.exe scripts\\refresh_fixtures.py --tickers AAPL,MSFT

Ablage:

* ``prices/<TICKER>_1d.parquet`` - OHLCV-Historie, mehrere Jahre, taeglich.
* ``statements/<TICKER>_<typ>_<periode>.parquet`` - Zeilen = Positionen,
  Spalten = Perioden (die yfinance-Orientierung, die der Vertrag behaelt).
* ``info/<TICKER>.json`` - Stammdaten aus ``Ticker.info``.
* ``screener/overview.parquet`` - ein Finviz-Screener-Ergebnis.
* ``search/<slug>.json`` - eingefrorene Antworten der Symbolsuche.
* ``earnings/<TICKER>.json`` - naechster Quartalszahlen-Termin.
* ``rates/<SLUG>.parquet`` - FX-Tageshistorien, Slug siehe
  :func:`btlite.data.rates.fx_fixture_slug` (``EURUSD=X`` -> ``EURUSD``).
* ``manifest.json`` - Abrufzeitpunkt und Paketversionen.

Nach dem Lauf gehoert das Abrufdatum in ``tests/fixtures/README.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from btlite.data.finviz_provider import normalize_screen_table  # noqa: E402
from btlite.data.rates import (  # noqa: E402
    PIVOT_CURRENCY,
    fx_fixture_slug,
    fx_symbol,
    normalize_fx_history,
)
from btlite.data.search import query_slug  # noqa: E402
from btlite.data.yfinance_provider import (  # noqa: E402
    EARNINGS_CALENDAR_KEY,
    normalize_ohlcv,
    normalize_statement,
)
from btlite.models.common import Currency  # noqa: E402

#: Ticker der Kursfixtures. Zehn Stueck, damit Phase 2 eine
#: Portfolio-Optimierung ueber zehn Titel testen kann.
PRICE_TICKERS: Final[tuple[str, ...]] = (
    "AAPL",
    "MSFT",
    "NVDA",
    "SPY",
    "JNJ",
    "XOM",
    "KO",
    "PG",
    "JPM",
    "WMT",
)

#: Ticker mit vollstaendigen Statements.
#:
#: Bewusst alle Kursticker ausser SPY (ein ETF fuehrt keine Statements). Ein
#: Screening-Preset braucht ein Universum mit Streuung: mit nur drei teuren
#: Qualitaetstiteln findet ein Value-Screen per Definition nichts, und das
#: Akzeptanzkriterium "Preset liefert > 0 Treffer" waere nicht pruefbar.
#: XOM, KO, JPM und WMT bringen niedrigere Bewertungen und Dividenden ins
#: Universum.
STATEMENT_TICKERS: Final[tuple[str, ...]] = (
    "AAPL",
    "MSFT",
    "NVDA",
    "JNJ",
    "XOM",
    "KO",
    "PG",
    "JPM",
    "WMT",
)

#: Zeitraum der Kurshistorien.
PRICE_PERIOD: Final[str] = "5y"

#: Wartezeit zwischen zwei Tickern in Sekunden. Bewusst konservativ - dieses
#: Skript hat es nicht eilig, und die Quelle soll nicht gedrosselt werden.
SLEEP_BETWEEN_TICKERS_S: Final[float] = 1.5

#: Finviz-Filter des Screener-Fixtures. Keys exakt inklusive Punkt.
SCREENER_FILTERS: Final[dict[str, str]] = {
    "Index": "S&P 500",
    "Sector": "Technology",
}

#: Trefferobergrenze des Screener-Fixtures. Klein halten - jede Seite zu
#: 20 Zeilen ist ein eigener Request.
SCREENER_LIMIT: Final[int] = 40

#: Suchanfragen, deren Antworten eingefroren werden.
#:
#: Die Auswahl deckt die Faelle ab, an denen sich das Ranking entscheidet:
#: ein Name, dessen Kuerzel niemand kennt (``servicenow`` -> ``NOW``), Namen
#: mit klarem Kuerzel (``apple``, ``microsoft``, ``nvidia``), ein
#: mehrdeutiger Nachname (``johnson``), ein Gattungsbegriff mit vielen
#: Treffern (``bank``), zwei reine Kuerzel-Eingaben (``NOW``, ``AAPL``) und
#: ein Tippfehler (``microsft``).
SEARCH_QUERIES: Final[tuple[str, ...]] = (
    "servicenow",
    "apple",
    "microsoft",
    "nvidia",
    "johnson",
    "bank",
    "NOW",
    "AAPL",
    "microsft",
)

#: Rohtreffer je Suchanfrage. Grosszuegig, damit das lokale Ranking im Test
#: etwas zu sortieren hat.
SEARCH_COUNT: Final[int] = 25

#: Ticker der Termin-Fixtures.
#:
#: Die zehn Kursticker plus RKLB und SAP.DE, die der Nutzer real haelt.
#: ``SPY`` ist bewusst dabei: ein ETF fuehrt keine Quartalszahlen, und genau
#: dieser Fall muss offline testbar sein - sonst faellt erst live auf, ob der
#: HTTP 404 der Quelle als Fehler hochschlaegt oder als richtige Antwort gilt.
EARNINGS_TICKERS: Final[tuple[str, ...]] = (*PRICE_TICKERS, "RKLB", "SAP.DE")


def fixtures_dir() -> Path:
    """Liefert das Zielverzeichnis der Fixtures.

    Returns:
        Pfad zu ``tests/fixtures`` im Repository.
    """
    return _REPO_ROOT / "tests" / "fixtures"


def refresh_prices(tickers: tuple[str, ...], out_dir: Path) -> list[str]:
    """Zieht die Kurshistorien und legt sie als parquet ab.

    Args:
        tickers: Abzurufende Symbole.
        out_dir: Wurzel der Fixtures.

    Returns:
        Liste der erfolgreich geschriebenen Symbole.
    """
    import yfinance as yf

    target = out_dir / "prices"
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for symbol in tickers:
        print(f"  Kurse {symbol} ({PRICE_PERIOD}, 1d) ...", flush=True)
        try:
            raw = yf.Ticker(symbol).history(
                period=PRICE_PERIOD, interval="1d", auto_adjust=True, actions=True
            )
            df = normalize_ohlcv(raw, ticker=symbol)
        except Exception as exc:  # Ein Ausfall darf den Lauf nicht abbrechen
            print(f"    FEHLER: {exc}", file=sys.stderr)
            continue
        if df.empty:
            print("    leer, uebersprungen", file=sys.stderr)
            continue
        df.to_parquet(target / f"{symbol}_1d.parquet", engine="pyarrow")
        written.append(symbol)
        print(f"    {len(df)} Zeilen, {df.index[0].date()} bis {df.index[-1].date()}")
        time.sleep(SLEEP_BETWEEN_TICKERS_S)

    return written


def refresh_statements(tickers: tuple[str, ...], out_dir: Path) -> list[str]:
    """Zieht Statements (annual und quarterly, alle drei Typen).

    Args:
        tickers: Abzurufende Symbole.
        out_dir: Wurzel der Fixtures.

    Returns:
        Liste der erfolgreich geschriebenen Symbole.
    """
    import yfinance as yf

    target = out_dir / "statements"
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    getters = {
        "income": "get_income_stmt",
        "balance": "get_balance_sheet",
        "cashflow": "get_cash_flow",
    }

    for symbol in tickers:
        ticker = yf.Ticker(symbol)
        any_written = False
        for period, freq in (("annual", "yearly"), ("quarterly", "quarterly")):
            for kind, getter in getters.items():
                print(f"  Statement {symbol} {kind}/{period} ...", flush=True)
                try:
                    raw = getattr(ticker, getter)(freq=freq)
                    df = normalize_statement(raw, ticker=symbol)
                except Exception as exc:
                    print(f"    FEHLER: {exc}", file=sys.stderr)
                    continue
                if df.empty:
                    print("    leer, uebersprungen", file=sys.stderr)
                    continue
                # parquet verlangt String-Spaltennamen; die Perioden werden
                # beim Laden aus dem ISO-Format zurueckgewandelt.
                to_write = df.copy()
                to_write.columns = [c.isoformat() for c in df.columns]
                to_write.to_parquet(target / f"{symbol}_{kind}_{period}.parquet", engine="pyarrow")
                any_written = True
                print(f"    {df.shape[0]} Positionen x {df.shape[1]} Perioden")
                time.sleep(SLEEP_BETWEEN_TICKERS_S)
        if any_written:
            written.append(symbol)

    return written


def refresh_info(tickers: tuple[str, ...], out_dir: Path) -> list[str]:
    """Zieht die Stammdaten aus ``Ticker.info`` und legt sie als JSON ab.

    Args:
        tickers: Abzurufende Symbole.
        out_dir: Wurzel der Fixtures.

    Returns:
        Liste der erfolgreich geschriebenen Symbole.
    """
    import yfinance as yf

    target = out_dir / "info"
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for symbol in tickers:
        print(f"  Stammdaten {symbol} ...", flush=True)
        try:
            info = dict(yf.Ticker(symbol).info)
        except Exception as exc:
            print(f"    FEHLER: {exc}", file=sys.stderr)
            continue
        if not info:
            print("    leer, uebersprungen", file=sys.stderr)
            continue
        payload = {k: v for k, v in info.items() if _json_safe(v)}
        (target / f"{symbol}.json").write_text(
            json.dumps(payload, indent=1, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        written.append(symbol)
        print(f"    {len(payload)} Felder")
        time.sleep(SLEEP_BETWEEN_TICKERS_S)

    return written


def refresh_earnings(tickers: tuple[str, ...], out_dir: Path) -> list[str]:
    """Zieht die naechsten Quartalszahlen-Termine aus ``Ticker.calendar``.

    Abgelegt wird der **Rohbefund**, nicht ein fertiges "naechstes Datum":
    ob die Quelle den Schluessel ueberhaupt fuehrt (``has_earnings``) und
    welche Tage darunter standen. Die Auswahl des naechsten Termins samt
    Vergangenheitsabgleich passiert erst beim Lesen, in
    :func:`btlite.data.yfinance_provider.next_earnings_from_calendar`.

    Genau deshalb veralten diese Fixtures kontrolliert: ist das eingefrorene
    Datum inzwischen verstrichen, liefert der Provider ``next_date=None``
    statt eines Termins in der Vergangenheit. Die Datei muss dafuer nicht
    angefasst werden.

    Args:
        tickers: Abzurufende Symbole.
        out_dir: Wurzel der Fixtures.

    Returns:
        Liste der erfolgreich geschriebenen Symbole.
    """
    import yfinance as yf

    target = out_dir / "earnings"
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for symbol in tickers:
        print(f"  Termin {symbol} ...", flush=True)
        try:
            calendar = yf.Ticker(symbol).calendar
        except Exception as exc:
            print(f"    FEHLER: {exc}", file=sys.stderr)
            continue

        # Fehlt der Schluessel, fuehrt das Instrument keine Quartalszahlen
        # (ETF, Krypto, Index). yfinance faengt den HTTP 404 intern ab. Das
        # ist die richtige Antwort und wird als solche eingefroren.
        has_key = isinstance(calendar, dict) and EARNINGS_CALENDAR_KEY in calendar
        days: list[str] = []
        if has_key:
            for item in calendar[EARNINGS_CALENDAR_KEY]:
                day = item.date() if isinstance(item, datetime) else item
                days.append(day.isoformat())

        payload = {"ticker": symbol, "has_earnings": has_key, "dates": days}
        (target / f"{symbol}.json").write_text(
            json.dumps(payload, indent=1, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        written.append(symbol)
        print(f"    {'keine Quartalszahlen' if not has_key else days or 'Termin unbekannt'}")
        time.sleep(SLEEP_BETWEEN_TICKERS_S)

    return written


def _json_safe(value: object) -> bool:
    """Prueft, ob ein Wert verlustfrei nach JSON serialisierbar ist.

    Args:
        value: Zu pruefender Wert.

    Returns:
        True wenn ``json.dumps`` den Wert ohne Sonderbehandlung schreibt.
    """
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def refresh_screener(out_dir: Path) -> bool:
    """Zieht ein Finviz-Screener-Ergebnis.

    Args:
        out_dir: Wurzel der Fixtures.

    Returns:
        True wenn das Fixture geschrieben wurde.
    """
    from finvizfinance.screener.overview import Overview

    target = out_dir / "screener"
    target.mkdir(parents=True, exist_ok=True)

    print(f"  Screener {SCREENER_FILTERS} (limit={SCREENER_LIMIT}) ...", flush=True)
    try:
        screener = Overview()
        screener.set_filter(filters_dict=dict(SCREENER_FILTERS))
        raw = screener.screener_view(order="Ticker", limit=SCREENER_LIMIT, verbose=0)
        df = normalize_screen_table(raw)
    except Exception as exc:
        print(f"    FEHLER: {exc}", file=sys.stderr)
        return False

    if df.empty:
        print("    leer, uebersprungen", file=sys.stderr)
        return False

    df.to_parquet(target / "overview.parquet", engine="pyarrow")
    (target / "overview_request.json").write_text(
        json.dumps({"finviz_filters": SCREENER_FILTERS, "limit": SCREENER_LIMIT}, indent=1),
        encoding="utf-8",
    )
    print(f"    {len(df)} Treffer")
    return True


def refresh_search(queries: tuple[str, ...], out_dir: Path) -> list[str]:
    """Zieht die Antworten der Symbolsuche und legt sie als JSON ab.

    Gespeichert wird **nur** die ``quotes``-Liste der Antwort. Der Rest
    (News, Listen, Research, Zeitmessungen) ist fuer die Suche bedeutungslos
    und wuerde die Fixtures unnoetig aufblaehen.

    Args:
        queries: Abzurufende Suchbegriffe.
        out_dir: Wurzel der Fixtures.

    Returns:
        Liste der erfolgreich geschriebenen Suchbegriffe.
    """
    import yfinance as yf

    target = out_dir / "search"
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for query in queries:
        slug = query_slug(query)
        if not slug:
            print(f"    {query!r} ergibt keinen Slug, uebersprungen", file=sys.stderr)
            continue
        print(f"  Suche {query!r} ...", flush=True)
        try:
            result = yf.Search(
                query,
                max_results=SEARCH_COUNT,
                news_count=0,
                lists_count=0,
                recommended=0,
                enable_fuzzy_query=False,
            )
            quotes = [q for q in result.quotes if isinstance(q, dict)]
        except Exception as exc:  # Ein Ausfall darf den Lauf nicht abbrechen
            print(f"    FEHLER: {exc}", file=sys.stderr)
            continue
        if not quotes:
            print("    keine Treffer, uebersprungen", file=sys.stderr)
            continue

        payload = {"query": query, "quotes": quotes}
        (target / f"{slug}.json").write_text(
            json.dumps(payload, indent=1, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        written.append(query)
        print(f"    {len(quotes)} Treffer, bester: {quotes[0].get('symbol')}")
        time.sleep(SLEEP_BETWEEN_TICKERS_S)

    return written


def fx_pairs() -> tuple[tuple[str, str], ...]:
    """Listet die Waehrungspaare, deren Historie eingefroren wird.

    Die Auswahl folgt der Triangulation aus :mod:`btlite.data.rates`: jeder
    Kurs laeuft entweder direkt oder ueber :data:`~btlite.data.rates.
    PIVOT_CURRENCY` (USD). Es genuegen deshalb **nicht** alle Kreuze, sondern
    fuer jede bekannte Waehrung die beiden USD-Schenkel. Damit ist jedes Paar
    aus :class:`~btlite.models.common.Currency` offline bedienbar, ohne dass
    die Quelle 110 Kreuze liefern muesste - was sie fuer exotische Paare
    historisch ohnehin nicht tut (``HKDNOK=X`` fuehrt keine Historie).

    ``Currency.OTHER`` entfaellt: daraus laesst sich kein Symbol bauen.

    Returns:
        Paare als ``(base, quote)`` in ISO-4217, jeweils beide Richtungen
        gegen USD. Aufsteigend sortiert und duplikatfrei.
    """
    pivot = PIVOT_CURRENCY.value
    pairs: set[tuple[str, str]] = set()
    for currency in Currency:
        if currency in (Currency.OTHER, PIVOT_CURRENCY):
            continue
        pairs.add((currency.value, pivot))
        pairs.add((pivot, currency.value))
    return tuple(sorted(pairs))


def refresh_rates(pairs: tuple[tuple[str, str], ...], out_dir: Path) -> list[str]:
    """Zieht die FX-Tageshistorien und legt sie als parquet ab.

    Gespeichert wird das **normalisierte** Ergebnis von
    :func:`~btlite.data.rates.normalize_fx_history`: tz-aware UTC-Index,
    genau eine Spalte ``close``. Bewusst nicht der Rohframe - die Umrechnung
    von ``Europe/London`` nach UTC gehoert in die Datenschicht und passiert
    dort genau einmal. Das Londoner Kalenderdatum bleibt aus dem UTC-Index
    ueber ``_london_dates`` rekonstruierbar, weil die Konvertierung
    verlustfrei ist.

    Args:
        pairs: Paare als ``(base, quote)`` in ISO-4217.
        out_dir: Wurzel der Fixtures.

    Returns:
        Liste der erfolgreich geschriebenen Yahoo-Symbole.
    """
    import yfinance as yf

    target = out_dir / "rates"
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for base, quote in pairs:
        symbol = fx_symbol(Currency(base), Currency(quote))
        print(f"  Wechselkurs {symbol} ({PRICE_PERIOD}, 1d) ...", flush=True)
        try:
            raw = yf.Ticker(symbol).history(
                period=PRICE_PERIOD, interval="1d", auto_adjust=False, actions=False
            )
            df = normalize_fx_history(raw)
        except Exception as exc:  # Ein Ausfall darf den Lauf nicht abbrechen
            print(f"    FEHLER: {exc}", file=sys.stderr)
            continue
        if df.empty:
            print("    leer, uebersprungen", file=sys.stderr)
            continue
        df.to_parquet(target / f"{fx_fixture_slug(symbol)}.parquet", engine="pyarrow")
        written.append(symbol)
        print(f"    {len(df)} Zeilen, {df.index[0].date()} bis {df.index[-1].date()}")
        time.sleep(SLEEP_BETWEEN_TICKERS_S)

    return written


def write_manifest(out_dir: Path, sections: dict[str, Any]) -> None:
    """Schreibt das Manifest mit Abrufzeitpunkt und Paketversionen.

    Args:
        out_dir: Wurzel der Fixtures.
        sections: Was in diesem Lauf geschrieben wurde.
    """
    import importlib.metadata as md

    versions: dict[str, str] = {}
    for package in ("yfinance", "finvizfinance", "pandas", "pyarrow"):
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:  # pragma: no cover - Pflichtabhaengigkeit
            versions[package] = "unbekannt"

    manifest_path = out_dir / "manifest.json"
    previous: dict[str, Any] = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))

    previous.setdefault("sections", {})
    for name, payload in sections.items():
        previous["sections"][name] = {
            "fetched_at": datetime.now(UTC).isoformat(),
            "content": payload,
        }
    previous["package_versions"] = versions
    previous["last_run"] = datetime.now(UTC).isoformat()
    previous["price_period"] = PRICE_PERIOD

    manifest_path.write_text(
        json.dumps(previous, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    print(f"\nManifest geschrieben: {manifest_path}")


def main(argv: list[str] | None = None) -> int:
    """Einstiegspunkt des Skripts.

    Args:
        argv: Argumentliste. ``None`` nimmt ``sys.argv``.

    Returns:
        Exit-Code: 0 bei Erfolg.
    """
    parser = argparse.ArgumentParser(
        description="Zieht die eingefrorenen API-Antworten neu. Geht ins Netz.",
    )
    parser.add_argument(
        "--only",
        choices=["prices", "statements", "info", "screener", "search", "rates", "earnings"],
        action="append",
        help="Nur diese Abschnitte erneuern. Mehrfach angebbar.",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Kommaseparierte Symbolliste statt der Defaults.",
    )
    args = parser.parse_args(argv)

    sections = (
        set(args.only)
        if args.only
        else {"prices", "statements", "info", "screener", "search", "rates", "earnings"}
    )
    price_tickers = PRICE_TICKERS
    statement_tickers = STATEMENT_TICKERS
    override: tuple[str, ...] = ()
    if args.tickers:
        override = tuple(t.strip().upper() for t in args.tickers.split(",") if t.strip())
        price_tickers = override
        statement_tickers = override

    out_dir = fixtures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Any] = {}

    print(f"Fixtures nach {out_dir}\n")

    if "prices" in sections:
        print("== Kurshistorien ==")
        written["prices"] = refresh_prices(price_tickers, out_dir)
    if "statements" in sections:
        print("\n== Statements ==")
        written["statements"] = refresh_statements(statement_tickers, out_dir)
    if "info" in sections:
        print("\n== Stammdaten ==")
        written["info"] = refresh_info(price_tickers, out_dir)
    if "screener" in sections:
        print("\n== Screener ==")
        written["screener"] = refresh_screener(out_dir)
    if "search" in sections:
        print("\n== Symbolsuche ==")
        written["search"] = refresh_search(SEARCH_QUERIES, out_dir)
    if "rates" in sections:
        print("\n== Wechselkurse ==")
        written["rates"] = refresh_rates(fx_pairs(), out_dir)
    if "earnings" in sections:
        print("\n== Quartalszahlen-Termine ==")
        written["earnings"] = refresh_earnings(
            override if args.tickers else EARNINGS_TICKERS, out_dir
        )

    write_manifest(out_dir, written)
    print("\nNicht vergessen: Abrufdatum in tests/fixtures/README.md nachtragen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
