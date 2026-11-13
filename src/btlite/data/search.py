"""Normalisierung und Ranking der Symbolsuche.

Dieses Modul ist **providerunabhaengig und rein**: es macht kein Netzwerk,
liest keine Dateien und haelt keinen Zustand. Es bekommt eine Liste bereits
gebauter :class:`~btlite.models.market.SearchHit` herein und bringt sie in die
Reihenfolge, in der ein Nutzer sie erwartet. Damit ist die gesamte Suchlogik
ohne Netz testbar - der Provider bleibt ein duenner Abrufwrapper.

Die Rangfolge, absteigend:

1. **Exakter Kuerzel-Treffer.** ``AAPL`` bringt AAPL nach ganz oben, auch
   wenn ein anderer Titel einen hoeheren Quellen-Score traegt.
2. **Exakter Namens-Treffer.** ``apple inc`` trifft "Apple Inc.".
3. **Namensanfang.** ``micro`` bringt "Microsoft Corporation" vor einen
   Titel, der "Micro" nur in der Mitte fuehrt.
4. **Kuerzel-Anfang.** ``NV`` bringt NVDA vor einen reinen Teiltreffer.
5. **Teiltreffer** irgendwo im Namen.
6. **Fuzzy-Treffer.** Tippfehler wie ``microsft`` finden ueber
   :mod:`difflib` noch "Microsoft" - dieselbe Technik, mit der
   ``UnknownFilterError`` seinen Vorschlag macht.

Innerhalb derselben Stufe entscheidet der Score der Quelle, danach die
Kuerze des Namens (ein exakter kurzer Name ist meist der gesuchte Emittent)
und zuletzt das Kuerzel alphabetisch. Die Sortierung ist damit **stabil und
deterministisch**, unabhaengig von der Eingangsreihenfolge.

Duplikate ueber Boersen hinweg fasst :func:`deduplicate_hits` zusammen:
ServiceNow liegt bei Yahoo an sechs Boersen, der Nutzer will einen Eintrag.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable, Sequence
from typing import Any, Final

from btlite.models.common import AssetClass, normalize_ticker
from btlite.models.market import SearchHit

#: Rangstufen der Trefferguete. Kleiner ist besser, 0 ist der beste Rang.
#: Die Namen stehen fuer sich; die Werte sind nur Sortierschluessel.
TIER_EXACT_TICKER: Final[int] = 0
TIER_EXACT_NAME: Final[int] = 1
TIER_NAME_PREFIX: Final[int] = 2
TIER_TICKER_PREFIX: Final[int] = 3
TIER_SUBSTRING: Final[int] = 4
TIER_FUZZY: Final[int] = 5

#: Aehnlichkeitsschwelle fuer Fuzzy-Treffer, dimensionslos von 0 bis 1.
#: 0.72 laesst ``microsft`` auf "Microsoft" durch (0.94), haelt aber
#: unverwandte Namen zuverlaessig heraus.
FUZZY_CUTOFF: Final[float] = 0.72

#: Instrumentenklassen, die bei der Dubletten-Zusammenfassung bevorzugt als
#: Vertreter gewaehlt werden. Ein Nutzer, der "servicenow" sucht, will die
#: Stammaktie, nicht ein CDR-Zertifikat an einer Zweitboerse.
_CLASS_PREFERENCE: Final[dict[AssetClass, int]] = {
    AssetClass.EQUITY: 0,
    AssetClass.ETF: 1,
    AssetClass.INDEX: 2,
    AssetClass.FUND: 3,
    AssetClass.CRYPTO: 4,
    AssetClass.OTHER: 5,
}

#: Rechtsformzusaetze, die beim Namensvergleich wegfallen. Sonst scheitert
#: ``apple`` an "Apple Inc." als exakter Treffer, obwohl es der ist.
_LEGAL_SUFFIXES: Final[tuple[str, ...]] = (
    "incorporated",
    "corporation",
    "company",
    "limited",
    "holdings",
    "holding",
    "group",
    "plc",
    "inc",
    "corp",
    "ltd",
    "llc",
    "co",
    "sa",
    "se",
    "nv",
    "ag",
)

#: Alles, was nicht Buchstabe, Ziffer oder Leerzeichen ist.
_NON_WORD: Final[re.Pattern[str]] = re.compile(r"[^0-9a-z ]+")

#: Mehrfache Leerzeichen.
_MULTI_SPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


#: Abbildung yfinance-``quoteType`` auf die Instrumentenklasse des Vertrags.
#: Deckungsgleich mit der Tabelle in ``yfinance_provider.py``; die Suche
#: liefert dieselben Typbezeichner wie ``Ticker.info``.
_ASSET_CLASS_MAP: Final[dict[str, AssetClass]] = {
    "EQUITY": AssetClass.EQUITY,
    "ETF": AssetClass.ETF,
    "INDEX": AssetClass.INDEX,
    "CRYPTOCURRENCY": AssetClass.CRYPTO,
    "MUTUALFUND": AssetClass.FUND,
}


#: Aufloesung der internen Yahoo-Boersenkuerzel auf lesbare Namen.
#:
#: Die Suchantwort liefert unter ``exchDisp`` bereits den Anzeigenamen, die
#: Stammdaten unter ``exchange`` dagegen nur das Kuerzel. Ohne diese Tabelle
#: stuenden fuer dasselbe Papier je nach Quelle ``NASDAQ:MSFT`` oder
#: ``NMS:MSFT`` - derselbe Treffer muss aber gleich aussehen, egal woher er
#: kommt. Abgedeckt sind die Boersen der Fixture-Ticker; unbekannte Kuerzel
#: bleiben unveraendert stehen, statt zu verschwinden.
_EXCHANGE_DISPLAY: Final[dict[str, str]] = {
    "NMS": "NASDAQ",
    "NGM": "NASDAQ",
    "NCM": "NASDAQ",
    "NAS": "NASDAQ",
    "NYQ": "NYSE",
    "PCX": "NYSEArca",
    "ASE": "NYSEAmerican",
    "BTS": "BATS Trading",
    "TOR": "Toronto",
    "GER": "XETRA",
    "FRA": "Frankfurt",
    "LSE": "London",
    "VIE": "Vienna",
    "NEO": "NEO",
}


def display_exchange(raw: str | None) -> str | None:
    """Loest ein Boersenkuerzel auf seinen Anzeigenamen auf.

    Args:
        raw: Kuerzel oder bereits lesbarer Name der Quelle, etwa ``NMS`` oder
            ``NASDAQ``.

    Returns:
        Lesbarer Name, das unveraenderte Kuerzel wenn es unbekannt ist, oder
        None bei leerer Eingabe.
    """
    if not raw or not raw.strip():
        return None
    value = raw.strip()
    return _EXCHANGE_DISPLAY.get(value.upper(), value)


def hit_from_quote(raw: dict[str, Any]) -> SearchHit | None:
    """Baut einen :class:`SearchHit` aus einem Roh-Quote der Yahoo-Suche.

    Gemeinsam genutzt von :class:`~btlite.data.yfinance_provider.YFinanceProvider`
    und :class:`~btlite.data.fixtures.FixtureProvider` - die Fixtures sind
    eingefrorene Antworten desselben Endpunkts, also darf es nur einen Parser
    geben.

    Bei der Boerse gewinnt ``exchDisp`` vor ``exchange``: Yahoo liefert unter
    ``exchange`` das interne Kuerzel ``NYQ``, unter ``exchDisp`` das lesbare
    ``NYSE``. Der Nutzer will ``NYSE:NOW`` sehen, nicht ``NYQ:NOW``.

    Args:
        raw: Ein Eintrag aus ``quotes`` der Antwort von
            ``/v1/finance/search``.

    Returns:
        Validierter Treffer, oder None wenn der Eintrag kein brauchbares
        Symbol traegt. Ein solcher Eintrag faellt still heraus, statt den
        gesamten Abruf scheitern zu lassen.
    """
    symbol = raw.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        return None

    name = raw.get("longname") or raw.get("shortname")
    exchange = raw.get("exchDisp") or raw.get("exchange")
    sector = raw.get("sectorDisp") or raw.get("sector")
    quote_type = str(raw.get("quoteType", "")).upper()

    score_raw = raw.get("score")
    score: float | None = None
    if isinstance(score_raw, int | float) and not isinstance(score_raw, bool):
        value = float(score_raw)
        # NaN ist nicht gleich sich selbst; negative Scores sind vom Vertrag
        # ausgeschlossen und hier bedeutungslos.
        if value == value and value >= 0.0:
            score = value

    return SearchHit(
        ticker=normalize_ticker(symbol),
        name=str(name) if isinstance(name, str) and name.strip() else None,
        exchange=display_exchange(exchange) if isinstance(exchange, str) else None,
        asset_class=_ASSET_CLASS_MAP.get(quote_type, AssetClass.OTHER),
        sector=str(sector) if isinstance(sector, str) and sector.strip() else None,
        score=score,
    )


def hits_from_quotes(quotes: Iterable[Any]) -> list[SearchHit]:
    """Baut Treffer aus der ``quotes``-Liste einer Yahoo-Suchantwort.

    Args:
        quotes: Rohliste aus der Antwort. Eintraege, die keine Dictionaries
            sind oder kein Symbol tragen, werden uebersprungen.

    Returns:
        Liste validierter Treffer, in der Reihenfolge der Quelle.
    """
    out: list[SearchHit] = []
    for raw in quotes:
        if not isinstance(raw, dict):
            continue
        hit = hit_from_quote(raw)
        if hit is not None:
            out.append(hit)
    return out


def normalize_query(query: str) -> str:
    """Bringt eine Roheingabe auf die Vergleichsform des Rankings.

    Kleinschreibung, Sonderzeichen entfernt, Leerzeichen zusammengezogen.
    Eine Eingabe, die danach leer ist - nur Leerzeichen, nur ``***`` -
    liefert einen leeren String. Die Aufrufer werten das als "keine Suche",
    nicht als Fehler.

    Args:
        query: Roheingabe des Nutzers.

    Returns:
        Normalisierte Vergleichsform, moeglicherweise leer.
    """
    lowered = query.strip().lower()
    cleaned = _NON_WORD.sub(" ", lowered)
    return _MULTI_SPACE.sub(" ", cleaned).strip()


def query_slug(query: str) -> str:
    """Bildet eine Suchanfrage auf einen Dateinamen ab.

    Gemeinsame Regel von ``scripts/refresh_fixtures.py`` (schreibt) und
    :class:`~btlite.data.fixtures.FixtureProvider` (liest). Dadurch findet der
    Fixture-Provider genau die Datei wieder, die das Skript angelegt hat -
    ``AAPL`` und ``aapl`` landen auf demselben Slug.

    Args:
        query: Roheingabe des Nutzers.

    Returns:
        Dateinamenstauglicher Slug ohne Endung. Leer, wenn die Anfrage nach
        der Normalisierung nichts uebrig laesst.
    """
    return normalize_query(query).replace(" ", "_")


def normalize_name(name: str | None) -> str:
    """Normalisiert einen Emittentennamen fuer den Vergleich.

    Wie :func:`normalize_query`, entfernt zusaetzlich aber gaengige
    Rechtsformzusaetze am Ende. ``Apple Inc.`` wird damit zu ``apple`` und
    die Eingabe ``apple`` zu einem exakten Namenstreffer.

    Args:
        name: Name laut Quelle, oder None.

    Returns:
        Normalisierter Name, leer wenn ``name`` None oder leer ist.
    """
    if not name:
        return ""
    base = normalize_query(name)
    parts = base.split(" ")
    while len(parts) > 1 and parts[-1] in _LEGAL_SUFFIXES:
        parts.pop()
    return " ".join(parts)


def _ticker_form(query: str) -> str:
    """Bildet die Eingabe auf die Vergleichsform eines Kuerzels ab.

    Kuerzel enthalten Punkte und Bindestriche bedeutungstragend
    (``BRK-B``, ``BMW.DE``), Leerzeichen dagegen nie.

    Args:
        query: Roheingabe des Nutzers.

    Returns:
        Grossgeschriebene, von Leerzeichen befreite Form.
    """
    return query.strip().upper().replace(" ", "")


def _match_tier(hit: SearchHit, query_norm: str, query_ticker: str) -> int | None:
    """Bestimmt die Rangstufe eines Treffers zur Anfrage.

    Args:
        hit: Zu bewertender Treffer.
        query_norm: Normalisierte Anfrage aus :func:`normalize_query`.
        query_ticker: Kuerzel-Vergleichsform aus :func:`_ticker_form`.

    Returns:
        Eine der ``TIER_*``-Konstanten, oder None wenn der Treffer nicht auf
        die Anfrage passt. Fuzzy-Treffer bewertet :func:`_fuzzy_tier`.
    """
    ticker = hit.ticker.upper()
    if query_ticker and ticker == query_ticker:
        return TIER_EXACT_TICKER

    name_norm = normalize_name(hit.name)
    if name_norm and name_norm == query_norm:
        return TIER_EXACT_NAME
    if name_norm and name_norm.startswith(query_norm):
        return TIER_NAME_PREFIX
    if query_ticker and ticker.startswith(query_ticker):
        return TIER_TICKER_PREFIX

    # Teiltreffer: irgendwo im Namen, oder ein Wortanfang im Namen. Die
    # Wortpruefung faengt "Johnson & Johnson" bei Eingabe "johnson" ab,
    # wenn der Gesamtname anders beginnt.
    if name_norm and query_norm and query_norm in name_norm:
        return TIER_SUBSTRING
    if query_ticker and query_ticker in ticker:
        return TIER_SUBSTRING
    return None


def _fuzzy_tier(hit: SearchHit, query_norm: str) -> tuple[int, float] | None:
    """Prueft einen Treffer auf Aehnlichkeit trotz Tippfehler.

    Verglichen wird die Anfrage gegen den normalisierten Gesamtnamen **und**
    gegen jedes einzelne Namenswort. Der Wortvergleich ist noetig, weil
    ``microsft`` gegen "microsoft corporation" als Ganzes unter die Schwelle
    faellt, gegen das Wort "microsoft" aber klar darueber liegt.

    Args:
        hit: Zu bewertender Treffer.
        query_norm: Normalisierte Anfrage.

    Returns:
        Tupel aus :data:`TIER_FUZZY` und der Aehnlichkeit von 0 bis 1, oder
        None wenn die Aehnlichkeit unter :data:`FUZZY_CUTOFF` liegt.
    """
    name_norm = normalize_name(hit.name)
    if not name_norm or not query_norm:
        return None

    candidates = [name_norm, *name_norm.split(" ")]
    best = max(
        difflib.SequenceMatcher(None, query_norm, candidate).ratio() for candidate in candidates
    )
    if best < FUZZY_CUTOFF:
        return None
    return TIER_FUZZY, best


def rank_hits(hits: Iterable[SearchHit], query: str) -> list[SearchHit]:
    """Sortiert Treffer nach Relevanz zur Anfrage.

    Reine Funktion, siehe Moduldoku fuer die Rangfolge. Treffer, die weder
    direkt noch fuzzy auf die Anfrage passen, fallen heraus - eine Quelle
    liefert zu ``bank`` auch Fondsnamen, die nichts damit zu tun haben.

    Args:
        hits: Zu sortierende Treffer, Reihenfolge beliebig.
        query: Roheingabe des Nutzers.

    Returns:
        Neue, absteigend nach Relevanz sortierte Liste. Leer, wenn die
        Anfrage leer ist oder nichts passt.
    """
    query_norm = normalize_query(query)
    # ``normalize_query`` ist die Instanz, die ueber "verwertbare Eingabe"
    # entscheidet. Eine Anfrage wie ``***`` behaelt in der Kuerzelform ihre
    # Sonderzeichen und wuerde sonst als Teiltreffer auf alles passen.
    if not query_norm:
        return []
    query_ticker = _ticker_form(query)

    scored: list[tuple[tuple[int, float, float, int, str], SearchHit]] = []
    for hit in hits:
        tier = _match_tier(hit, query_norm, query_ticker)
        similarity = 1.0
        if tier is None:
            fuzzy = _fuzzy_tier(hit, query_norm)
            if fuzzy is None:
                continue
            tier, similarity = fuzzy

        # Sortierschluessel, aufsteigend zu lesen: kleinere Stufe zuerst,
        # dann hoehere Aehnlichkeit, dann hoeherer Quellen-Score, dann
        # kuerzeres Kuerzel, zuletzt alphabetisch.
        key = (
            tier,
            -similarity,
            -(hit.score if hit.score is not None else 0.0),
            len(hit.ticker),
            hit.ticker,
        )
        scored.append((key, hit))

    scored.sort(key=lambda item: item[0])
    return [hit for _, hit in scored]


def deduplicate_hits(hits: Sequence[SearchHit]) -> list[SearchHit]:
    """Fasst denselben Emittenten ueber Boersen hinweg zu einem Treffer zusammen.

    ServiceNow liegt bei Yahoo an NYSE, Toronto, XETRA, Frankfurt, NEO und
    Wien. Sechs Zeilen fuer ein Unternehmen sind keine Hilfe. Gruppiert wird
    ueber den normalisierten Namen; ein Treffer ohne Namen bildet immer seine
    eigene Gruppe, weil sich ueber ihn nichts aussagen laesst.

    Vertreter der Gruppe wird der Eintrag mit der bevorzugten
    Instrumentenklasse (Stammaktie vor Zertifikat), bei Gleichstand der mit
    dem hoechsten Score. Die Reihenfolge der Gruppen folgt dem ersten
    Auftreten - eine vorher sortierte Liste bleibt sortiert.

    Args:
        hits: Treffer, idealerweise bereits durch :func:`rank_hits` sortiert.

    Returns:
        Neue Liste mit hoechstens einem Treffer je Emittent.
    """
    groups: dict[str, list[SearchHit]] = {}
    order: list[str] = []
    for index, hit in enumerate(hits):
        name_norm = normalize_name(hit.name)
        key = name_norm if name_norm else f"\x00{index}:{hit.ticker}"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(hit)

    out: list[SearchHit] = []
    for key in order:
        members = groups[key]
        out.append(
            min(
                members,
                key=lambda h: (
                    _CLASS_PREFERENCE.get(h.asset_class, len(_CLASS_PREFERENCE)),
                    -(h.score if h.score is not None else 0.0),
                    h.ticker,
                ),
            )
        )
    return out


def prepare_results(
    hits: Iterable[SearchHit],
    query: str,
    *,
    limit: int = 10,
    deduplicate: bool = True,
) -> list[SearchHit]:
    """Bringt Rohtreffer in die endgueltige Form fuer den Aufrufer.

    Der gemeinsame Abschluss aller Provider: erst ranken, dann Dubletten
    zusammenfassen, dann kuerzen. Die Reihenfolge ist wesentlich - erst
    kuerzen und dann deduplizieren wuerde bei sechs ServiceNow-Listings nur
    einen einzigen Treffer uebrig lassen.

    Args:
        hits: Rohtreffer der Quelle, Reihenfolge beliebig.
        query: Roheingabe des Nutzers.
        limit: Maximale Trefferzahl. Werte kleiner 1 liefern eine leere Liste.
        deduplicate: Ob derselbe Emittent ueber Boersen hinweg zusammengefasst
            wird.

    Returns:
        Hoechstens ``limit`` Treffer, absteigend nach Relevanz.
    """
    if limit < 1:
        return []
    ranked = rank_hits(hits, query)
    if deduplicate:
        ranked = deduplicate_hits(ranked)
    return ranked[:limit]
