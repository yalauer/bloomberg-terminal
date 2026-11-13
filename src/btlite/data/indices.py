"""Live-Abruf der Bestaende gaengiger Aktienindizes.

Motivation: ein Screening ueber zehn fest verdrahtete Ticker ist kein
Screening. Dieses Modul liefert das vollstaendige Universum eines Index -
503 Titel beim S&P 500, 40 beim DAX - damit die Screening-Schicht darauf
filtern kann.

Architekturregel 2: dieses Modul gehoert in ``data/``, weil es Netzwerk
macht. Es rechnet und filtert nicht; es holt, normalisiert und gibt
Modelltypen aus ``models/screening.py`` zurueck (Architekturregel 1).

Zwei Quellen, je nach Index:

* **Finviz** (``finvizfinance``) fuer die US-Indizes. Der Filter ``Index``
  kennt genau vier Werte, siehe :data:`_FINVIZ_INDEX_VALUES`. Die Antwort
  fuehrt Ticker, Firmenname **und** Sektor - alle drei werden uebernommen.
* **Wikipedia** (``pandas.read_html``) fuer den DAX, den Finviz nicht kennt,
  weil er nur US-Titel fuehrt.

Die Tabellenauswahl bei Wikipedia laeuft ueber die **Spaltennamen**, nicht
ueber ``pd.read_html(match=...)``: ``match`` verlangt ``html5lib``, und das
ist im Projekt nicht installiert.

Caching ist Pflicht, nicht Kuer. Ein Indexbestand aendert sich in der Regel
quartalsweise (Rebalancing), dazwischen nur bei Uebernahmen und Delistings.
500 Titel bei jedem Seitenaufruf neu zu holen waere Missbrauch der Quelle.
Die TTL ist deshalb mit :data:`CONSTITUENTS_TTL_S` auf sieben Tage gesetzt.

Offline-Verhalten: die App startet per Default offline
(:attr:`~btlite.config.DataSettings.offline`). Ein Live-Abruf ist dann
verboten und dieses Modul wirft
:class:`~btlite.core.errors.ProviderUnavailableError` mit einer
``user_message``, die den Ausweg nennt. Bewusst **kein** stiller Rueckfall
auf die zehn Fixture-Ticker: der Nutzer waehlte "S&P 500" und bekaeme
zehn Titel, ohne dass es ihm jemand sagt - genau der Mangel, den dieses
Modul behebt. Ein typisierter Fehler ist kein Absturz, das UI rendert ihn
als Banner.

Rate Limits werden nie umgangen. Der Finviz-Pfad nutzt denselben
prozessweiten Token-Bucket wie
:class:`~btlite.data.finviz_provider.FinvizProvider`; ein Block der Quelle
wird als :class:`~btlite.core.errors.RateLimitError` nach oben gereicht.

Symbole ohne Boersen-Endung
---------------------------

Die DAX-Tabelle fuehrt fertige Symbole (``ADS.DE``). Die des STOXX
Europe 600 tut das **nicht**: dort steht ``ZURN``, ``SAF``, ``DG``. Genau
eine von 467 Zeilen traegt ueberhaupt einen Punkt. Roh an yfinance gegeben
ist das nicht bloss eine Luecke, sondern eine Falschaussage:

* ``ZURN`` liefert nichts, ``ZURN.SW`` die Zurich Insurance Group.
* ``SAF`` liefert nichts, ``SAF.PA`` die Safran SA.
* ``DG`` liefert die **Dollar General Corporation**, einen US-Discounter.
  Gemeint war Vinci, ``DG.PA``.
* ``CTS.DE`` liefert die AVIC International Holdings - gemeint war
  CTS Eventim, ``EVD.DE``.

Ein Screening, das Vinci durch Dollar General ersetzt, ist falsch, ohne
dass es jemandem auffaellt. Die Aufloesung laeuft deshalb zweistufig:

1. **Aus dem Land.** Die Country-Spalte bestimmt die Endung, siehe
   :data:`_COUNTRY_SUFFIXES`. ``France`` + ``SAF`` ergibt ``SAF.PA``.
   Leerzeichen in Aktienklassen werden zum Bindestrich, den Yahoo
   verwendet: ``NOVO B`` wird ``NOVO-B.CO``.
2. **Ueber den Firmennamen.** Wo Stufe 1 kein bei der Quelle bekanntes
   Symbol ergibt, sucht ein injizierter :class:`~btlite.data.base.SymbolSearch`
   nach dem Firmennamen. Die Auswahl unter den Treffern steuert das Land,
   **nicht** die Trefferreihenfolge: die Suche nach "Thales" liefert
   ``CSF.MU`` vor ``HO.PA``, richtig ist der Pariser Titel.

Fallen Sitzland und Notierung auseinander, gewinnt die **tatsaechliche
Notierung**. InPost steht als Luxemburg im Index, notiert aber in
Amsterdam (``INPST.AS``); Teva steht als Israel und notiert an der NYSE
(``TEVA``). Ein Symbol, das es an der Boerse des Sitzlandes nicht gibt,
waere fuer einen Kursabruf wertlos.

Was sich nicht sicher aufloesen laesst, kommt **nicht** ins Universum.
Mehrdeutigkeit zaehlt dabei als unsicher: die Suche nach "Ferrari" liefert
``FERGR.AS`` (Ferrari Group PLC, eine Logistikfirma) und ``RACE``
(Ferrari N.V.) gleichermassen namensgleich - welcher gemeint ist, sagt
die Quelle nicht, also faellt der Titel heraus. Die uebersprungenen Titel
haengen als :class:`~btlite.models.screening.SkippedMember` am Ergebnis und
sind damit auszaehlbar; still verschwinden darf keiner.

Die Aufloesung ueber Namen kostet einen Request je offenem Titel. Sie
laeuft deshalb hinter demselben Cache wie der Bestand selbst und nutzt den
yfinance-Token-Bucket - ein Bestandsabruf belastet die Quelle so, wie er
sie tatsaechlich belastet.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Final

import pandas as pd

from btlite.config import Settings, get_settings
from btlite.core.cache import DataFrameCache, cache_key
from btlite.core.clock import utc_now
from btlite.core.errors import (
    DataValidationError,
    ProviderUnavailableError,
    RateLimitError,
)
from btlite.core.logging import get_logger
from btlite.core.ratelimit import TokenBucket, get_registry, retry_with_backoff
from btlite.data.search import normalize_name
from btlite.models.common import AssetClass, DataSource, Provenance, normalize_ticker
from btlite.models.screening import (
    IndexConstituents,
    IndexDefinition,
    IndexMember,
    IndexSource,
    SkippedMember,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from btlite.data.base import SymbolSearch
    from btlite.models.market import SearchHit

_LOG = get_logger(__name__)

#: TTL der gecachten Bestaende in Sekunden. Sieben Tage.
#:
#: Begruendung: Indexbestaende aendern sich planmaessig quartalsweise beim
#: Rebalancing, ausserplanmaessig bei Uebernahmen und Delistings. Eine Woche
#: ist damit kurz genug, eine Aenderung zeitnah zu sehen, und lang genug,
#: dass die Quelle nicht bei jedem Seitenaufruf 500 Zeilen liefern muss.
CONSTITUENTS_TTL_S: Final[int] = 7 * 24 * 3600

#: User-Agent fuer Wikipedia-Abrufe. Wikipedia verlangt einen aussagekraeftigen
#: Agent statt des Default-Strings der urllib.
_USER_AGENT: Final[str] = "BT-Lite/0.1 (lokales Finanz-Analyse-Frontend)"

#: Spaltennamen der Finviz-Overview-Antwort.
_FINVIZ_TICKER_COLUMN: Final[str] = "Ticker"
_FINVIZ_NAME_COLUMN: Final[str] = "Company"
_FINVIZ_SECTOR_COLUMN: Final[str] = "Sector"

#: Die vom Finviz-Filter ``Index`` zugelassenen Werte. Bewusst hier notiert,
#: damit die Registry gegen sie geprueft werden kann; die Laufzeitvalidierung
#: liest sie aus dem installierten Paket, nicht aus dieser Kopie.
_FINVIZ_INDEX_VALUES: Final[tuple[str, ...]] = ("S&P 500", "NASDAQ 100", "DJIA", "RUSSELL 2000")

#: Obergrenze fuer einen Finviz-Bestandsabruf. Der groesste unterstuetzte
#: Index ist der Russell 2000 mit rund 2000 Titeln.
_FINVIZ_FETCH_LIMIT: Final[int] = 2500

#: Zeilen je Finviz-Screener-Seite. Bestimmt, wie viele Requests ein Abruf
#: kostet, und damit wie viele Token der Bucket abgibt.
_FINVIZ_ROWS_PER_PAGE: Final[int] = 20

#: View der Finviz-Overview-Tabelle. Wird auch vom modernen JSON-Fallback
#: verwendet, damit beide Abrufpfade dieselben Spalten anfordern.
_FINVIZ_OVERVIEW_VIEW: Final[int] = 111

#: Boersen-Endung je Sitzland, in der Notation von Yahoo Finance.
#:
#: Stufe 1 der Symbolaufloesung. Der Schluessel ist der Wert der
#: Country-Spalte der Quelle, exakt wie Wikipedia ihn schreibt.
#:
#: Gegen die Quelle verifiziert: mit diesen Endungen kennt yfinance
#: ``ZURN.SW``, ``GSK.L``, ``INGA.AS``, ``SAF.PA``, ``MBG.DE``, ``BKT.MC``,
#: ``FORTUM.HE``, ``AGS.BR``, ``EDP.LS``, ``EBS.VI`` und ``LPP.WA``.
#:
#: Nicht jedes Land der Tabelle steht hier: ``Bermuda`` und ``Israel``
#: fuehren je einen Titel (Hiscox, Teva), der an einer Boerse ausserhalb
#: seines Sitzlandes notiert. Fuer sie gibt es keine sinnvolle Endung; sie
#: gehen direkt in Stufe 2.
_COUNTRY_SUFFIXES: Final[dict[str, str]] = {
    "United Kingdom": ".L",
    "Germany": ".DE",
    "France": ".PA",
    "Switzerland": ".SW",
    "Sweden": ".ST",
    "Netherlands": ".AS",
    "Spain": ".MC",
    "Italy": ".MI",
    "Belgium": ".BR",
    "Finland": ".HE",
    "Norway": ".OL",
    "Denmark": ".CO",
    "Austria": ".VI",
    "Ireland": ".IR",
    "Portugal": ".LS",
    "Luxembourg": ".LU",
    "Poland": ".WA",
    "Greece": ".AT",
}

#: Endungen, die ein aufgeloestes Symbol tragen darf. Das sind genau die
#: Heimatboersen der Index-Laender.
#:
#: Hintergrund: die Namenssuche liefert fuer europaeische Titel bereitwillig
#: deutsche Regionalboersen - Air Liquide als ``AIL.MU`` (Muenchen),
#: Ferrovial als ``8ZQ.HM`` (Hamburg), Covestro als ``1COV.SG`` (Stuttgart).
#: Das sind reale, aber marginal gehandelte Zweitlistings; ein Screening auf
#: ihnen misst Liquiditaet, die es nicht gibt. Ein Symbol **ohne** Endung ist
#: dagegen zugelassen: das ist ein US-Listing (``TEVA``, ``STLA``, ``NGG``)
#: und oft die liquideste Notierung des Titels ueberhaupt.
_ACCEPTED_SUFFIXES: Final[frozenset[str]] = frozenset(_COUNTRY_SUFFIXES.values())

#: Maximale Trefferzahl je Namenssuche in Stufe 2.
_SEARCH_LIMIT: Final[int] = 10


def _definition(**kwargs: Any) -> IndexDefinition:
    """Baut eine Registry-Definition.

    Args:
        **kwargs: Felder von :class:`~btlite.models.screening.IndexDefinition`.

    Returns:
        Die validierte Definition.
    """
    return IndexDefinition(**kwargs)


#: Herkunft je Bezugsquelle. Der DAX kommt von Wikipedia, nicht von
#: yfinance - der Nachweis steht sichtbar im UI, eine falsche Angabe waere
#: eine Falschaussage ueber die Datenherkunft.
_PROVENANCE_SOURCES: Final[dict[IndexSource, DataSource]] = {
    IndexSource.FINVIZ: DataSource.FINVIZ,
    IndexSource.WIKIPEDIA: DataSource.WIKIPEDIA,
}


#: Registry der unterstuetzten Indizes, Schluessel in Kleinbuchstaben.
#:
#: Ein weiterer Index kommt allein durch einen Eintrag hier dazu - an den
#: Abrufpfaden ist nichts zu aendern. Das deckt das "etc." der Anforderung ab.
INDEX_REGISTRY: Final[dict[str, IndexDefinition]] = {
    "sp500": _definition(
        key="sp500",
        label="S&P 500",
        source=IndexSource.FINVIZ,
        finviz_index_value="S&P 500",
    ),
    "djia": _definition(
        key="djia",
        label="Dow Jones Industrial Average",
        source=IndexSource.FINVIZ,
        finviz_index_value="DJIA",
    ),
    "nasdaq100": _definition(
        key="nasdaq100",
        label="Nasdaq 100",
        source=IndexSource.FINVIZ,
        finviz_index_value="NASDAQ 100",
    ),
    "russell2000": _definition(
        key="russell2000",
        label="Russell 2000",
        source=IndexSource.FINVIZ,
        finviz_index_value="RUSSELL 2000",
    ),
    "dax": _definition(
        key="dax",
        label="DAX",
        source=IndexSource.WIKIPEDIA,
        wikipedia_url="https://en.wikipedia.org/wiki/DAX",
        wikipedia_ticker_column="Ticker",
        wikipedia_name_column="Company",
        wikipedia_sector_column="Prime Standard Sector",
    ),
    "stoxx600": _definition(
        key="stoxx600",
        label="STOXX Europe 600",
        source=IndexSource.WIKIPEDIA,
        wikipedia_url="https://en.wikipedia.org/wiki/STOXX_Europe_600",
        wikipedia_ticker_column="Ticker",
        wikipedia_name_column="Company",
        wikipedia_sector_column="ICB Sector",
        # Gesetzt, und nur hier: die Ticker dieser Tabelle tragen keine
        # Boersen-Endung. Ohne Aufloesung landet Dollar General statt Vinci
        # im Universum. Siehe Modulkopf.
        wikipedia_country_column="Country",
    ),
}

#: Vorauswahl der Oberflaeche. Der S&P 500 ist das breiteste liquide
#: US-Universum und die uebliche Ausgangsmenge eines Screens.
DEFAULT_INDEX_KEY: Final[str] = "sp500"


def available_indices() -> list[IndexDefinition]:
    """Listet die unterstuetzten Indizes fuer die Auswahl in der Oberflaeche.

    Returns:
        Definitionen in stabiler Reihenfolge, der Default zuerst.
    """
    ordered = [INDEX_REGISTRY[DEFAULT_INDEX_KEY]]
    ordered.extend(d for k, d in INDEX_REGISTRY.items() if k != DEFAULT_INDEX_KEY)
    return ordered


def get_index_definition(key: str) -> IndexDefinition:
    """Schlaegt eine Indexdefinition nach.

    Args:
        key: Bezeichner des Index, Gross- und Kleinschreibung egal.

    Returns:
        Die hinterlegte Definition.

    Raises:
        ValueError: Wenn der Index nicht in der Registry steht.
    """
    normalized = key.strip().lower()
    try:
        return INDEX_REGISTRY[normalized]
    except KeyError:
        msg = f"Unknown index {key!r}. Allowed: {sorted(INDEX_REGISTRY)}"
        raise ValueError(msg) from None


def _clean_text(raw: object) -> str | None:
    """Normalisiert einen Textwert der Quelle.

    Leere Zeichenketten, Platzhalter und Fehlwerte werden zu ``None`` -
    Projektkonvention: fehlende Werte sind ``None``, nie ein leerer String.

    Args:
        raw: Rohwert aus der Quellentabelle.

    Returns:
        Bereinigter Text, oder ``None`` wenn kein Wert vorliegt.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "-", "n/a"}:
        return None
    return text


def _clean_ticker(raw: object) -> str | None:
    """Normalisiert ein Symbol der Quelle auf die kanonische Form.

    Boersen-Endungen wie ``.DE`` und ``.PA`` bleiben erhalten - yfinance
    braucht sie, um ein Nicht-US-Listing aufzuloesen.

    Args:
        raw: Rohwert der Ticker-Spalte.

    Returns:
        Kanonisches Symbol, oder ``None`` wenn die Zelle keines enthaelt.
    """
    text = _clean_text(raw)
    if text is None:
        return None
    try:
        return normalize_ticker(text)
    except ValueError:
        return None


def suffix_for_country(country: str | None) -> str | None:
    """Liefert die Boersen-Endung zum Sitzland eines Titels.

    Stufe 1 der Symbolaufloesung, siehe Modulkopf.

    Args:
        country: Wert der Country-Spalte der Quelle, etwa ``France``.

    Returns:
        Endung inklusive Punkt, etwa ``.PA``, oder ``None`` wenn das Land
        unbekannt ist oder keine eigene Boerse im Index fuehrt (Bermuda,
        Israel).
    """
    if country is None:
        return None
    return _COUNTRY_SUFFIXES.get(country.strip())


def candidate_symbol(raw_ticker: str, country: str | None) -> str:
    """Baut aus Rohsymbol und Land das Kandidatensymbol der Stufe 1.

    Zwei Anpassungen an die Schreibweise von Yahoo Finance:

    * Leerzeichen in Aktienklassen werden zum Bindestrich. Die Quelle
      schreibt ``NOVO B``, Yahoo ``NOVO-B.CO``.
    * Die Endung des Sitzlandes kommt hinten dran, sofern das Rohsymbol
      nicht schon eine traegt. ``BT.A`` bleibt deshalb unveraendert.

    Args:
        raw_ticker: Symbol, wie die Quelle es fuehrt.
        country: Sitzland laut Quelle, oder ``None``.

    Returns:
        Kandidatensymbol. Ein Kandidat ist **kein** Beweis: ``CTS.DE``
        entsteht so aus "CTS Eventim" und gehoert bei Yahoo einer
        chinesischen Firma. Erst die Pruefung in Stufe 2 entscheidet.
    """
    base = raw_ticker.strip().upper().replace(" ", "-")
    if "." in base:
        return base
    suffix = suffix_for_country(country)
    return f"{base}{suffix}" if suffix else base


def _name_contains(source_name: str, hit_name: str | None) -> bool:
    """Lockere Namenspruefung: enthaelt der eine Name den anderen?

    Nur zusammen mit einem passenden Laender-Suffix verwendet. Die
    Kombination aus "richtige Boerse" und "Name steckt drin" ist tragfaehig,
    der Name allein waere es nicht: die Quelle schreibt ``Air Liquide``, Yahoo
    ``L'Air Liquide S.A.`` - hier ist die Lockerung noetig.

    Args:
        source_name: Firmenname laut Indexquelle.
        hit_name: Firmenname laut Suchtreffer, oder ``None``.

    Returns:
        True wenn einer der normalisierten Namen im anderen steckt.
    """
    want = normalize_name(source_name)
    got = normalize_name(hit_name)
    if not want or not got:
        return False
    return want in got or got in want


def _name_matches_strictly(source_name: str, hit_name: str | None) -> bool:
    """Strenge Namenspruefung ueber die fuehrenden Woerter.

    Gilt fuer Treffer an einer **anderen** Boerse als der des Sitzlandes.
    Dort fehlt die Bestaetigung durch das Land, also muss der Name allein
    tragen - und die lockere Pruefung traegt dann nicht: ``BT Group`` steckt
    normalisiert in ``BTS Group AB`` nicht, aber Teilstring-Regeln lassen
    aehnliche Paare durch, und das Ergebnis waere die falsche Firma.

    Verglichen werden die fuehrenden Woerter, damit ``Haleon`` auf
    ``Haleon plc`` passt, ``LifeCo`` aber nicht auf ``Lifecore Biomedical``.

    Args:
        source_name: Firmenname laut Indexquelle.
        hit_name: Firmenname laut Suchtreffer, oder ``None``.

    Returns:
        True bei Gleichheit oder gemeinsamem Wortpraefix.
    """
    want = normalize_name(source_name)
    got = normalize_name(hit_name)
    if not want or not got:
        return False
    if want == got:
        return True
    want_words = want.split(" ")
    got_words = got.split(" ")
    shared = min(len(want_words), len(got_words))
    return want_words[:shared] == got_words[:shared]


def _is_tradeable_equity(hit: SearchHit) -> bool:
    """Prueft, ob ein Suchtreffer eine handelbare Aktie an einer Hauptboerse ist.

    Haelt drei Sorten Falschtreffer heraus, die die Suche real liefert:
    Indizes (``^N100`` zu "Euronext"), Fonds und Zertifikate
    (``LINC.BA``, ein CEDEAR auf Linde) sowie Zweitlistings an
    Regionalboersen (``AIL.MU`` zu Air Liquide), siehe
    :data:`_ACCEPTED_SUFFIXES`.

    Args:
        hit: Ein Treffer der Namenssuche.

    Returns:
        True wenn der Treffer als Universum-Mitglied taugt.
    """
    if hit.asset_class is not AssetClass.EQUITY:
        return False
    if hit.ticker.startswith("^"):
        return False
    if "." not in hit.ticker:
        # Kein Suffix heisst US-Listing - zugelassen, oft die liquideste
        # Notierung des Titels.
        return True
    return any(hit.ticker.endswith(s) for s in _ACCEPTED_SUFFIXES)


def confirms_candidate(hits: Sequence[SearchHit], *, candidate: str) -> bool:
    """Prueft, ob die Symbolsuche das Kandidatensymbol der Stufe 1 kennt.

    Die Suche nach einem Symbol liefert genau dann einen Treffer mit diesem
    Symbol, wenn die Quelle es fuehrt. Das ist die verlaessliche Pruefung -
    ein Kursabruf waere es nicht: ``yf.Ticker("CTS.DE").info`` liefert
    bereitwillig die AVIC International Holdings, waehrend die Suche nach
    ``CTS.DE`` leer ausgeht und den Titel damit korrekt in Stufe 2 schickt.

    Der Firmenname wird hier bewusst **nicht** verglichen. Ein Kandidat
    traegt bereits die Boerse des Sitzlandes; existiert er dort, ist er
    praktisch immer der gemeinte Titel. Die Namen der beiden Quellen gehen
    dabei regelmaessig auseinander, ohne dass etwas falsch waere: Wikipedia
    schreibt "BMW", Yahoo "Bayerische Motoren Werke Aktiengesellschaft",
    "Inditex" gegen "Industria de Diseno Textil", "Munich Re" gegen
    "Muenchener Rueckversicherungs-Gesellschaft". Ein Namensvergleich wuerde
    diese Titel verwerfen und nichts dafuer gewinnen - den einzigen realen
    Verwechslungsfall faengt schon die Existenzpruefung.

    Args:
        hits: Treffer der Suche nach ``candidate``.
        candidate: Das zu pruefende Symbol aus Stufe 1.

    Returns:
        True wenn die Quelle das Symbol fuehrt.
    """
    return any(h.ticker == candidate for h in hits)


def resolve_from_hits(
    hits: Sequence[SearchHit],
    *,
    source_name: str,
    country: str | None,
) -> tuple[str | None, str]:
    """Waehlt unter den Suchtreffern das Symbol eines Titels aus.

    Stufe 2 der Symbolaufloesung. Rein und ohne Netz, damit die
    Auswahlregel ohne Quelle pruefbar ist.

    Die Regel, in dieser Reihenfolge:

    1. **Ein Treffer an der Boerse des Sitzlandes**, dessen Name den der
       Quelle enthaelt. Das Land bestaetigt den Namen, deshalb genuegt hier
       die lockere Pruefung.
    2. **Genau ein** namensgleicher Treffer an einer anderen Hauptboerse.
       Notierung schlaegt Sitzland: InPost steht als Luxemburg im Index und
       notiert in Amsterdam, Teva als Israel und notiert an der NYSE.
    3. Sonst **nichts**. Mehrere gleich gute Treffer sind keine Auswahl,
       sondern eine offene Frage - "Ferrari" liefert die Ferrari Group PLC
       und Ferrari N.V., und die Quelle sagt nicht, welche gemeint ist.

    Args:
        hits: Treffer der Namenssuche, in Relevanzreihenfolge der Quelle.
        source_name: Firmenname laut Indexquelle.
        country: Sitzland laut Indexquelle, oder ``None``.

    Returns:
        Paar aus Symbol und Begruendung. Das Symbol ist ``None``, wenn kein
        Treffer sicher genug war; die Begruendung ist dann fuer den Nutzer
        lesbar und nennt den Grund.
    """
    equities = [h for h in hits if _is_tradeable_equity(h)]
    suffix = suffix_for_country(country)

    if suffix is not None:
        home = [
            h for h in equities if h.ticker.endswith(suffix) and _name_contains(source_name, h.name)
        ]
        if home:
            return home[0].ticker, "ueber den Namen an der Boerse des Sitzlandes gefunden"

    elsewhere = [h for h in equities if _name_matches_strictly(source_name, h.name)]
    if len(elsewhere) == 1:
        return elsewhere[0].ticker, "notiert ausserhalb des Sitzlandes, Treffer eindeutig"
    if len(elsewhere) > 1:
        listed = ", ".join(h.ticker for h in elsewhere)
        return None, f"mehrere gleich gute Treffer ({listed}), Zuordnung nicht eindeutig"
    return None, "die Symbolsuche kennt keinen passenden Titel"


def _members_from_frame(
    df: pd.DataFrame,
    *,
    ticker_column: str,
    name_column: str | None,
    sector_column: str | None,
) -> list[IndexMember]:
    """Baut die Mitgliederliste aus einer Quellentabelle.

    Duplikate werden entfernt, der erste Treffer gewinnt. Die Reihenfolge
    der Quelle bleibt erhalten.

    Args:
        df: Tabelle der Quelle.
        ticker_column: Spaltenname der Symbole.
        name_column: Spaltenname der Firmennamen, oder ``None``.
        sector_column: Spaltenname der Sektoren, oder ``None``.

    Returns:
        Bestand als Modelltypen.

    Raises:
        DataValidationError: Wenn die Ticker-Spalte fehlt. Das bedeutet eine
            Layoutaenderung der Quelle und wird nicht als still leeres
            Ergebnis verschwiegen.
    """
    if ticker_column not in df.columns:
        msg = "The source returns no ticker column."
        raise DataValidationError(
            msg, detail=f"erwartet={ticker_column!r}, vorhanden={list(df.columns)}"
        )

    names = df[name_column] if name_column and name_column in df.columns else None
    sectors = df[sector_column] if sector_column and sector_column in df.columns else None

    members: list[IndexMember] = []
    seen: set[str] = set()
    for pos, raw_ticker in enumerate(df[ticker_column]):
        ticker = _clean_ticker(raw_ticker)
        if ticker is None or ticker in seen:
            continue
        seen.add(ticker)
        members.append(
            IndexMember(
                ticker=ticker,
                name=_clean_text(names.iloc[pos]) if names is not None else None,
                sector=_clean_text(sectors.iloc[pos]) if sectors is not None else None,
            )
        )
    return members


def _parse_finviz_embedded_page(html: str) -> tuple[list[IndexMember], int, int]:
    """Liest eine Seite des neuen Finviz-Screeners aus dessen JSON-Payload.

    Finviz ersetzte 2026 die alte ``table.screener_table`` durch eine
    clientseitige ``div.fv-table``. Die Seite bettet ihre strukturierten
    Zeilen weiterhin als ``application/json`` ein; dieser Vertrag ist
    stabiler als die rein visuelle HTML-Struktur.

    Args:
        html: Vollstaendiger HTML-Text einer Screener-Seite.

    Returns:
        Mitglieder der Seite, Gesamtzahl und Seitengroesse.

    Raises:
        DataValidationError: Wenn der eingebettete Payload fehlt oder nicht
            mehr die erwartete Struktur besitzt.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    script = soup.find("script", attrs={"type": "application/json"})
    if script is None or script.string is None:
        msg = "Finviz changed its page layout, the screener payload is missing."
        raise DataValidationError(msg, detail="script[type='application/json'] fehlt")

    try:
        payload = json.loads(script.string)
        data = payload["data"]
        rows = data["rows"]
        total = int(data["totalCount"])
        page_size = int(data["pageSize"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        msg = "Finviz changed its page layout, the screener payload is invalid."
        raise DataValidationError(msg, detail=f"{type(exc).__name__}: {exc}") from exc

    if not isinstance(rows, list) or page_size <= 0 or total < 0:
        msg = "Finviz changed its page layout, the screener payload is invalid."
        raise DataValidationError(
            msg,
            detail=f"rows={type(rows).__name__}, total={total}, page_size={page_size}",
        )

    members: list[IndexMember] = []
    for row in rows:
        try:
            text = row["text"]
            raw_ticker = text["ticker"]
        except (KeyError, TypeError) as exc:
            msg = "Finviz changed its page layout, a screener row is invalid."
            raise DataValidationError(msg, detail=f"row={row!r}") from exc
        ticker = _clean_ticker(raw_ticker)
        if ticker is None:
            continue
        members.append(
            IndexMember(
                ticker=ticker,
                name=_clean_text(text.get("company")),
                sector=_clean_text(text.get("sector")),
            )
        )
    return members, total, page_size


#: Wert der Spalte ``kind`` im Cache-Frame fuer einen aufgenommenen Titel.
_KIND_MEMBER: Final[str] = "member"

#: Wert der Spalte ``kind`` im Cache-Frame fuer einen uebersprungenen Titel.
_KIND_SKIPPED: Final[str] = "skipped"


def _members_to_frame(
    members: Sequence[IndexMember], skipped: Sequence[SkippedMember] = ()
) -> pd.DataFrame:
    """Serialisiert den Bestand fuer den DataFrame-Cache.

    Der Cache legt parquet ab, deshalb der Umweg ueber eine Tabelle. Fehlende
    Werte werden als leerer String geschrieben und beim Lesen wieder zu
    ``None`` - parquet kennt kein ``None`` in einer Objektspalte.

    Die uebersprungenen Titel stehen in derselben Tabelle, unterschieden
    durch die Spalte ``kind``. Sie gehoeren in den Cache: sonst zeigte der
    erste Abruf "428 von 467", der zweite aus dem Cache aber nur "428", und
    die Luecke waere verschwunden.

    Args:
        members: Zu serialisierender Bestand.
        skipped: Uebersprungene Titel, siehe Modulkopf.

    Returns:
        Tabelle mit den Spalten ``kind``, ``ticker``, ``name``, ``sector``
        und ``reason``.
    """
    return pd.DataFrame(
        {
            "kind": [_KIND_MEMBER] * len(members) + [_KIND_SKIPPED] * len(skipped),
            "ticker": [m.ticker for m in members] + [s.source_ticker for s in skipped],
            "name": [m.name or "" for m in members] + [s.name or "" for s in skipped],
            "sector": [m.sector or "" for m in members] + [s.country or "" for s in skipped],
            "reason": [""] * len(members) + [s.reason for s in skipped],
        },
        dtype="string",
    )


def _members_from_cache_frame(df: pd.DataFrame) -> tuple[list[IndexMember], list[SkippedMember]]:
    """Liest Bestand und uebersprungene Titel aus der gecachten Tabelle zurueck.

    Vertraegt auch Tabellen aus der Zeit vor der Spalte ``kind``: fehlt sie,
    gilt jede Zeile als aufgenommener Titel. Ein alter Cache-Eintrag laeuft
    damit ab, statt einen Fehler zu werfen.

    Args:
        df: Tabelle, wie :func:`_members_to_frame` sie erzeugt.

    Returns:
        Paar aus Bestand und uebersprungenen Titeln, beide als Modelltypen.

    Raises:
        DataValidationError: Wenn die Ticker-Spalte fehlt.
    """
    if "kind" not in df.columns:
        members = _members_from_frame(
            df, ticker_column="ticker", name_column="name", sector_column="sector"
        )
        return members, []

    kinds = df["kind"].astype("object")
    member_rows = df[kinds == _KIND_MEMBER]
    skipped_rows = df[kinds == _KIND_SKIPPED]

    members = _members_from_frame(
        member_rows, ticker_column="ticker", name_column="name", sector_column="sector"
    )
    skipped = [
        SkippedMember(
            source_ticker=str(row_ticker),
            name=_clean_text(row_name),
            country=_clean_text(row_country),
            reason=_clean_text(row_reason) or "ohne Angabe",
        )
        for row_ticker, row_name, row_country, row_reason in zip(
            skipped_rows["ticker"],
            skipped_rows["name"],
            skipped_rows["sector"],
            skipped_rows["reason"] if "reason" in skipped_rows.columns else skipped_rows["name"],
            strict=False,
        )
        if _clean_text(row_ticker) is not None
    ]
    return members, skipped


class IndexConstituentsProvider:
    """Holt Indexbestaende live und cached sie auf der Platte.

    Erfuellt :class:`~btlite.data.base.IndexProvider` strukturell, ohne von
    dem Protocol zu erben.

    Die Instanz ist auch im Offline-Modus konstruierbar - erst der Abruf
    scheitert dann. Das ist Absicht: die Oberflaeche soll die Indexliste
    offline anzeigen koennen, ohne den Provider zu umgehen.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        cache: DataFrameCache | None = None,
        bucket: TokenBucket | None = None,
        symbol_search: SymbolSearch | None = None,
        search_bucket: TokenBucket | None = None,
    ) -> None:
        """Initialisiert den Provider.

        Args:
            settings: Projekt-Settings. ``None`` nimmt die globalen.
            cache: DataFrame-Cache. ``None`` baut einen aus den Settings.
            bucket: Token-Bucket fuer den Finviz-Pfad. ``None`` nimmt den
                prozessweiten finviz-Bucket aus der Registry - derselbe, den
                auch der Screener benutzt.
            symbol_search: Symbolsucher fuer Stufe 2 der Aufloesung, siehe
                Modulkopf. Wird **injiziert**, nicht hier konstruiert -
                Architekturregel 2. ``None`` ist zulaessig: dann laeuft
                Stufe 1 allein, und jeder Titel, den sie nicht sicher
                aufloest, wird uebersprungen statt geraten.
            search_bucket: Token-Bucket fuer die Namenssuche. ``None`` nimmt
                den prozessweiten yfinance-Bucket - dieselbe Quelle, also
                derselbe Bucket.
        """
        self._settings = settings if settings is not None else get_settings()
        self._cache = cache if cache is not None else DataFrameCache(self._settings.cache)
        self._bucket = bucket if bucket is not None else get_registry().bucket("finviz")
        self._retry = retry_with_backoff(settings=self._settings.ratelimit)
        self._search = symbol_search
        self._search_bucket = (
            search_bucket if search_bucket is not None else get_registry().bucket("yfinance")
        )

    def available_indices(self) -> list[IndexDefinition]:
        """Listet die unterstuetzten Indizes.

        Funktioniert auch offline: die Registry ist statisch, nur der
        Bestand ist es nicht.

        Returns:
            Definitionen in stabiler Reihenfolge, der Default zuerst.
        """
        return available_indices()

    def get_constituents(self, index_key: str) -> IndexConstituents:
        """Holt den Bestand eines Index.

        Der erste Abruf geht ins Netz, danach bedient der Cache fuer
        :data:`CONSTITUENTS_TTL_S` Sekunden.

        Args:
            index_key: Bezeichner aus :data:`INDEX_REGISTRY`, etwa ``sp500``.

        Returns:
            Der Bestand mit Symbolen und - soweit die Quelle sie fuehrt -
            Firmennamen und Sektoren.

        Raises:
            ValueError: Wenn der Index nicht in der Registry steht.
            ProviderUnavailableError: Im Offline-Modus, oder wenn die Quelle
                nicht erreichbar ist.
            RateLimitError: Wenn die Quelle gedrosselt hat oder blockt.
            DataValidationError: Wenn die Quelle ihr Layout geaendert hat.
        """
        definition = get_index_definition(index_key)
        key = cache_key("indices", "constituents", definition.key, definition.source.value)

        cached = self._cache.get_df(key)
        if cached is not None:
            _LOG.debug("Indexbestand aus dem Cache: %s", definition.key)
            cached_members, cached_skipped = _members_from_cache_frame(cached)
            return IndexConstituents(
                definition=definition,
                members=cached_members,
                skipped=cached_skipped,
                provenance=self._provenance(definition, from_cache=True),
            )

        self._require_online(definition)
        members, skipped = self._fetch(definition)
        if not members:
            msg = f"The source returned no constituents for {definition.label}."
            raise DataValidationError(msg, detail=f"source={definition.source.value}")

        if skipped:
            # Sichtbar, nicht still: der Nutzer soll erkennen koennen, dass
            # er ein unvollstaendiges Universum screent.
            _LOG.warning(
                "%s: %d von %d Titeln uebersprungen, ihr Symbol liess sich nicht "
                "sicher aufloesen. Beispiele: %s",
                definition.label,
                len(skipped),
                len(members) + len(skipped),
                ", ".join(f"{s.source_ticker} ({s.name})" for s in skipped[:5]),
            )

        self._cache.set_df(key, _members_to_frame(members, skipped), CONSTITUENTS_TTL_S)
        return IndexConstituents(
            definition=definition,
            members=members,
            skipped=skipped,
            provenance=self._provenance(definition, from_cache=False),
        )

    def get_tickers(self, index_key: str) -> list[str]:
        """Holt allein die Symbole eines Index.

        Bequemlichkeitsform von :meth:`get_constituents` fuer Aufrufer, die
        nur das Universum brauchen.

        Args:
            index_key: Bezeichner aus :data:`INDEX_REGISTRY`.

        Returns:
            Kanonische Symbole in Quellenreihenfolge.

        Raises:
            ValueError: Wenn der Index nicht in der Registry steht.
            ProviderUnavailableError: Im Offline-Modus, oder wenn die Quelle
                nicht erreichbar ist.
            RateLimitError: Wenn die Quelle gedrosselt hat oder blockt.
            DataValidationError: Wenn die Quelle ihr Layout geaendert hat.
        """
        return self.get_constituents(index_key).tickers

    # ------------------------------------------------------------ intern --
    def _require_online(self, definition: IndexDefinition) -> None:
        """Verweigert einen Live-Abruf im Offline-Modus.

        Bewusst ein Fehler und kein stiller Rueckfall auf ein
        Ersatzuniversum: ein Nutzer, der "S&P 500" waehlt und zehn Titel
        bekommt, wuerde das Ergebnis fuer vollstaendig halten.

        Args:
            definition: Der angefragte Index, fuer die Meldung.

        Raises:
            ProviderUnavailableError: Wenn der Offline-Modus aktiv ist.
        """
        if not self._settings.data.offline:
            return
        msg = (
            f"Offline mode: the constituents of {definition.label} cannot be "
            "retrieved. Set BTLITE_DATA_OFFLINE=0 for live data."
        )
        raise ProviderUnavailableError(msg, detail="BTLITE_DATA_OFFLINE=true")

    def _provenance(self, definition: IndexDefinition, *, from_cache: bool) -> Provenance:
        """Baut den Herkunftsnachweis eines Bestands.

        Args:
            definition: Der abgefragte Index.
            from_cache: Ob die Antwort aus dem lokalen Cache kam.

        Returns:
            Herkunftsnachweis mit Quelle und Abrufzeitpunkt.
        """
        source = _PROVENANCE_SOURCES[definition.source]
        note = (
            f"Index constituents of {definition.label} from {definition.source.value}. "
            f"Constituents are cached for up to {CONSTITUENTS_TTL_S // 86_400} days."
        )
        return Provenance(
            source=source,
            fetched_at=utc_now(),
            from_cache=from_cache,
            delay_minutes=None,
            note=note,
        )

    def _fetch(self, definition: IndexDefinition) -> tuple[list[IndexMember], list[SkippedMember]]:
        """Holt den Bestand bei der zur Definition passenden Quelle.

        Args:
            definition: Der abzufragende Index.

        Returns:
            Paar aus Bestand und uebersprungenen Titeln, beide als
            Modelltypen. Die zweite Liste ist leer, solange die Quelle
            fertige Symbole liefert.

        Raises:
            ProviderUnavailableError: Quelle nicht erreichbar.
            RateLimitError: Quelle hat gedrosselt oder blockt.
            DataValidationError: Layoutaenderung der Quelle.
        """
        if definition.source is IndexSource.FINVIZ:
            return self._retry(lambda: self._fetch_finviz(definition))(), []
        return self._retry(lambda: self._fetch_wikipedia(definition))()

    def _fetch_finviz(self, definition: IndexDefinition) -> list[IndexMember]:
        """Holt einen Bestand ueber den Finviz-Screener.

        Der Filter ``Index`` liefert die Overview-Ansicht mit Ticker,
        Firmenname und Sektor. Jede Ergebnisseite ist ein Request, deshalb
        werden entsprechend viele Token entnommen.

        Args:
            definition: Index mit gesetztem ``finviz_index_value``.

        Returns:
            Bestand als Modelltypen.

        Raises:
            ProviderUnavailableError: Quelle nicht erreichbar, oder
                ``finvizfinance`` fehlt.
            RateLimitError: Finviz hat gedrosselt oder blockt.
            DataValidationError: Layoutaenderung bei Finviz.
        """
        index_value = definition.finviz_index_value
        if index_value is None:  # pragma: no cover - der Modellvalidator schliesst das aus
            msg = f"Index {definition.key!r} names no Finviz filter value."
            raise DataValidationError(msg, detail="finviz_index_value=None")

        try:
            from finvizfinance.screener import overview
        except ImportError as exc:  # pragma: no cover - Pflichtabhaengigkeit
            msg = "The finvizfinance package is not installed."
            raise ProviderUnavailableError(msg, detail=str(exc)) from exc

        # Ein Token vor dem Aufruf. Die restlichen Seiten werden erst danach
        # verrechnet, anhand der tatsaechlich gelieferten Zeilen: ``limit``
        # ist eine Obergrenze, kein Versprechen. Wer hier fuer das volle
        # Limit vorab zahlt, blockiert bei 0.5 rps minutenlang, auch wenn
        # die Quelle nur 30 Zeilen liefert.
        self._bucket.acquire()

        screener = overview.Overview()
        try:
            screener.set_filter(filters_dict={"Index": index_value})
            raw = screener.screener_view(limit=_FINVIZ_FETCH_LIMIT, verbose=0)
        except Exception as exc:
            raise _map_finviz_exception(exc) from exc

        if raw is None or raw.empty:
            # Finviz 2026: das Paket 1.5 erkennt die neue div.fv-table nicht
            # und liefert None, obwohl die Seite gueltige JSON-Zeilen traegt.
            return self._fetch_finviz_modern(definition)

        self._settle_finviz_pages(len(raw))
        return _members_from_frame(
            raw,
            ticker_column=_FINVIZ_TICKER_COLUMN,
            name_column=_FINVIZ_NAME_COLUMN,
            sector_column=_FINVIZ_SECTOR_COLUMN,
        )

    def _fetch_finviz_modern(self, definition: IndexDefinition) -> list[IndexMember]:
        """Holt alle Seiten aus dem eingebetteten JSON des neuen Screeners.

        Args:
            definition: Index mit gesetztem ``finviz_index_value``.

        Returns:
            Vollstaendiger Bestand bis zur Sicherheitsgrenze.

        Raises:
            ProviderUnavailableError: Quelle nicht erreichbar.
            RateLimitError: Finviz hat gedrosselt oder blockt.
            DataValidationError: JSON-Struktur oder Zeilenzahl ist unplausibel.
        """
        index_value = definition.finviz_index_value
        if index_value is None:  # pragma: no cover - Modellvalidator
            msg = f"Index {definition.key!r} names no Finviz filter value."
            raise DataValidationError(msg, detail="finviz_index_value=None")

        try:
            from finvizfinance.constants import filter_dict
            from finvizfinance.util import fetch

            filter_spec = filter_dict["Index"]
            filter_code = filter_spec["option"][index_value]
            filter_value = f"{filter_spec['prefix']}_{filter_code}"
        except (ImportError, KeyError, TypeError) as exc:
            msg = "The installed finvizfinance package has no usable index filter."
            raise ProviderUnavailableError(msg, detail=f"{type(exc).__name__}: {exc}") from exc

        members: list[IndexMember] = []
        start = 1
        total: int | None = None
        while total is None or len(members) < total:
            if start > _FINVIZ_FETCH_LIMIT:
                msg = f"Finviz returned more than {_FINVIZ_FETCH_LIMIT} constituents."
                raise DataValidationError(msg, detail=f"index={definition.key}, total={total}")

            # Der alte Parser hat bereits einen Request verbraucht. Jeder
            # Request des Fallbacks nimmt deshalb vorab ein eigenes Token.
            self._bucket.acquire()
            try:
                response = fetch(
                    "https://finviz.com/screener.ashx",
                    {
                        "v": _FINVIZ_OVERVIEW_VIEW,
                        "f": filter_value,
                        "o": "ticker",
                        "r": start,
                    },
                )
                page, page_total, page_size = _parse_finviz_embedded_page(response.text)
            except Exception as exc:
                raise _map_finviz_exception(exc) from exc

            if total is None:
                total = page_total
            elif page_total != total:
                msg = "Finviz changed the constituent count while pages were loaded."
                raise DataValidationError(msg, detail=f"first={total}, current={page_total}")

            if not page and len(members) < total:
                msg = "Finviz returned an empty page before all constituents were loaded."
                raise DataValidationError(msg, detail=f"loaded={len(members)}, total={total}")
            members.extend(page)
            start += page_size

        return members[:total]

    def _settle_finviz_pages(self, rows: int) -> None:
        """Verrechnet die Ergebnisseiten eines Finviz-Abrufs im Token-Bucket.

        Das Paket holt intern eine Seite je 20 Zeilen und wartet zwischen den
        Seiten selbst. Die erste Seite ist beim Aufruf bereits bezahlt; die
        uebrigen werden hier nachgetragen, damit ein Bestandsabruf den
        gemeinsamen Finviz-Bucket so belastet, wie er die Quelle belastet
        hat - ein anschliessender Screen wartet dann entsprechend laenger.

        Bewusst nicht blockierend: die Requests sind schon raus, nachtraegliches
        Warten wuerde nur diesen Aufrufer aufhalten, ohne die Quelle zu
        schonen.

        Args:
            rows: Anzahl der gelieferten Ergebniszeilen.
        """
        pages = max(1, -(-rows // _FINVIZ_ROWS_PER_PAGE))
        for _ in range(pages - 1):
            if not self._bucket.try_acquire():
                break

    def _fetch_wikipedia(
        self, definition: IndexDefinition
    ) -> tuple[list[IndexMember], list[SkippedMember]]:
        """Holt einen Bestand aus einer Wikipedia-Tabelle.

        Die richtige Tabelle wird ueber ihre Spaltennamen gefunden, nicht
        ueber ``pd.read_html(match=...)`` - ``match`` setzt ``html5lib``
        voraus, das hier nicht installiert ist.

        Fuehrt die Definition eine ``wikipedia_country_column``, tragen die
        Symbole der Quelle keine Boersen-Endung und werden aufgeloest, siehe
        :meth:`_resolve_members` und den Modulkopf.

        Args:
            definition: Index mit gesetzter ``wikipedia_url`` und
                ``wikipedia_ticker_column``.

        Returns:
            Paar aus Bestand und uebersprungenen Titeln.

        Raises:
            ProviderUnavailableError: Seite nicht erreichbar.
            DataValidationError: Wenn keine Tabelle die erwartete
                Ticker-Spalte fuehrt - die Seite wurde umgebaut.
        """
        ticker_column = definition.wikipedia_ticker_column
        url = definition.wikipedia_url
        if ticker_column is None or url is None:  # pragma: no cover - Modellvalidator
            msg = f"Index {definition.key!r} names no complete Wikipedia source."
            raise DataValidationError(
                msg,
                detail=f"url={url!r}, ticker_column={ticker_column!r}",
            )

        try:
            tables = pd.read_html(url, storage_options={"User-Agent": _USER_AGENT})
        except Exception as exc:
            raise _map_wikipedia_exception(exc, definition) from exc

        table = _select_table(tables, ticker_column)
        if table is None:
            msg = f"The Wikipedia page for {definition.label} no longer lists a constituents table."
            raise DataValidationError(
                msg,
                detail=(
                    f"url={url}, erwartete Spalte={ticker_column!r}, "
                    f"gefundene Tabellen={len(tables)}"
                ),
            )

        if definition.wikipedia_country_column is not None:
            return self._resolve_members(table, definition)

        return (
            _members_from_frame(
                table,
                ticker_column=ticker_column,
                name_column=definition.wikipedia_name_column,
                sector_column=definition.wikipedia_sector_column,
            ),
            [],
        )

    def _resolve_members(
        self, table: pd.DataFrame, definition: IndexDefinition
    ) -> tuple[list[IndexMember], list[SkippedMember]]:
        """Loest Symbole ohne Boersen-Endung auf, zweistufig.

        Stufe 1 leitet die Endung aus dem Sitzland ab, Stufe 2 sucht bei
        Bedarf ueber den Firmennamen. Was beide Stufen nicht sicher
        aufloesen, wird uebersprungen und gezaehlt, nicht geraten -
        siehe Modulkopf.

        Ohne injizierten Symbolsucher laeuft Stufe 1 allein. Das ist der
        vorsichtigere Betrieb, nicht der bequemere: ohne die Pruefung ueber
        die Suche gilt auch ein Stufe-1-Symbol als unbestaetigt, und jeder
        Titel, dessen Endung unbekannt ist, faellt heraus.

        Args:
            table: Die Bestandstabelle der Quelle.
            definition: Index mit gesetzter ``wikipedia_country_column``.

        Returns:
            Paar aus aufgeloestem Bestand und uebersprungenen Titeln.

        Raises:
            DataValidationError: Wenn die Ticker-Spalte fehlt.
            RateLimitError: Wenn die Suche gedrosselt hat. Ein Limit wird
                nie umgangen, der Fehler geht nach oben.
        """
        ticker_column = definition.wikipedia_ticker_column
        if ticker_column is None or ticker_column not in table.columns:  # pragma: no cover
            msg = "The source returns no ticker column."
            raise DataValidationError(
                msg, detail=f"erwartet={ticker_column!r}, vorhanden={list(table.columns)}"
            )

        names = _optional_column(table, definition.wikipedia_name_column)
        sectors = _optional_column(table, definition.wikipedia_sector_column)
        countries = _optional_column(table, definition.wikipedia_country_column)

        members: list[IndexMember] = []
        skipped: list[SkippedMember] = []
        seen: set[str] = set()

        for pos, raw_ticker in enumerate(table[ticker_column]):
            source_ticker = _clean_text(raw_ticker)
            if source_ticker is None:
                continue
            name = _clean_text(names.iloc[pos]) if names is not None else None
            sector = _clean_text(sectors.iloc[pos]) if sectors is not None else None
            country = _clean_text(countries.iloc[pos]) if countries is not None else None

            ticker, reason = self._resolve_one(source_ticker, name=name, country=country)
            if ticker is None:
                skipped.append(
                    SkippedMember(
                        source_ticker=source_ticker, name=name, country=country, reason=reason
                    )
                )
                continue
            if ticker in seen:
                continue
            seen.add(ticker)
            members.append(IndexMember(ticker=ticker, name=name, sector=sector))

        return members, skipped

    def _resolve_one(
        self, source_ticker: str, *, name: str | None, country: str | None
    ) -> tuple[str | None, str]:
        """Loest ein einzelnes Symbol auf.

        Args:
            source_ticker: Symbol, wie die Quelle es fuehrt.
            name: Firmenname laut Quelle, oder ``None``.
            country: Sitzland laut Quelle, oder ``None``.

        Returns:
            Paar aus aufgeloestem Symbol und Begruendung. Das Symbol ist
            ``None``, wenn die Aufloesung nicht sicher war; die Begruendung
            landet dann im :class:`~btlite.models.screening.SkippedMember`.

        Raises:
            RateLimitError: Wenn die Suche gedrosselt hat.
        """
        candidate = candidate_symbol(source_ticker, country)
        has_suffix = "." in candidate

        if self._search is None:
            # Stufe 1 allein. Ohne Sucher laesst sich nichts bestaetigen,
            # also gilt nur, was eine Endung aus einem bekannten Land traegt.
            if has_suffix:
                return candidate, "aus dem Sitzland abgeleitet, ungeprueft"
            return None, (
                "ohne Boersen-Endung und ohne Symbolsuche nicht aufloesbar - "
                "ein rohes Symbol traefe womoeglich eine andere Firma"
            )

        if has_suffix and confirms_candidate(self._search_symbols(candidate), candidate=candidate):
            return candidate, "aus dem Sitzland abgeleitet und bestaetigt"

        if name is None:
            return None, "die Quelle nennt keinen Firmennamen, eine Suche ist nicht moeglich"

        return resolve_from_hits(self._search_symbols(name), source_name=name, country=country)

    def _search_symbols(self, query: str) -> list[SearchHit]:
        """Fragt den injizierten Symbolsucher, gedrosselt.

        Ein Token je Anfrage aus dem yfinance-Bucket - die Suche geht an
        dieselbe Quelle wie die Kursabrufe und teilt sich deren Limit.

        Args:
            query: Firmenname oder Symbol.

        Returns:
            Treffer der Suche, leer wenn die Quelle nichts kennt oder gerade
            nicht antwortet.

        Raises:
            RateLimitError: Wenn die Quelle gedrosselt hat. Geht bewusst nach
                oben, statt umgangen zu werden.
        """
        if self._search is None:  # pragma: no cover - Aufrufer prueft
            return []
        self._search_bucket.acquire()
        try:
            return self._search.search(query, limit=_SEARCH_LIMIT)
        except RateLimitError:
            raise
        except Exception as exc:
            # Ein einzelner Fehlschlag darf den Bestandsabruf nicht kippen.
            # Der Titel landet dann bei den uebersprungenen, sichtbar.
            _LOG.debug("Symbolsuche nach %r fehlgeschlagen: %s", query, exc)
            return []


def _optional_column(df: pd.DataFrame, column: str | None) -> pd.Series | None:
    """Liefert eine Spalte, wenn die Definition sie nennt und sie existiert.

    Args:
        df: Tabelle der Quelle.
        column: Gesuchter Spaltenname, oder ``None``.

    Returns:
        Die Spalte, oder ``None`` wenn sie nicht gefragt ist oder fehlt.
    """
    if column is None or column not in df.columns:
        return None
    return df[column]


def _select_table(tables: Iterable[pd.DataFrame], ticker_column: str) -> pd.DataFrame | None:
    """Waehlt die Tabelle, die die Ticker-Spalte fuehrt.

    Args:
        tables: Alle von ``read_html`` gefundenen Tabellen.
        ticker_column: Gesuchter Spaltenname.

    Returns:
        Die erste passende Tabelle, oder ``None`` wenn keine passt.
    """
    for table in tables:
        if ticker_column in table.columns:
            return table
    return None


def _map_finviz_exception(exc: Exception) -> Exception:
    """Bildet eine rohe finvizfinance-Exception auf die Projekthierarchie ab.

    Args:
        exc: Die vom Paket geworfene Exception.

    Returns:
        Der passende Fehler aus :mod:`btlite.core.errors`. Bereits
        abgebildete Fehler gehen unveraendert durch.
    """
    from btlite.core.errors import BTLiteError

    if isinstance(exc, BTLiteError):
        return exc

    from finvizfinance.exceptions import FinvizBlockedError, FinvizParseError

    if isinstance(exc, FinvizBlockedError):
        # Ein Block ist eine Aussage der Quelle, kein transienter Fehler.
        # Er wird nach oben gereicht, nicht umgangen.
        return RateLimitError(
            "Finviz is currently blocking requests from this address.",
            detail=f"{type(exc).__name__}: {exc}",
            retry_after_s=None,
        )
    if isinstance(exc, FinvizParseError):
        return DataValidationError(
            "Finviz changed its page layout, the index constituents cannot be read.",
            detail=f"{type(exc).__name__}: {exc}",
        )
    return ProviderUnavailableError(
        "Finviz is currently unreachable.",
        detail=f"{type(exc).__name__}: {exc}",
    )


def _map_wikipedia_exception(exc: Exception, definition: IndexDefinition) -> Exception:
    """Bildet einen Fehler des Wikipedia-Abrufs auf die Projekthierarchie ab.

    Args:
        exc: Die geworfene Exception, meist aus urllib oder lxml.
        definition: Der abgefragte Index, fuer die Meldung.

    Returns:
        Der passende Fehler aus :mod:`btlite.core.errors`.
    """
    from btlite.core.errors import BTLiteError

    if isinstance(exc, BTLiteError):
        return exc

    text = str(exc)
    if "429" in text or "Too Many Requests" in text:
        # Wikipedia drosselt. Nicht umgehen, nach oben reichen.
        return RateLimitError(
            "Wikipedia throttled the request.",
            detail=f"{type(exc).__name__}: {exc}",
            retry_after_s=None,
        )
    if isinstance(exc, ValueError):
        # read_html wirft ValueError, wenn es gar keine Tabelle findet.
        return DataValidationError(
            f"The Wikipedia page for {definition.label} contains no readable table.",
            detail=f"{type(exc).__name__}: {exc}",
        )
    return ProviderUnavailableError(
        f"The source for the constituents of {definition.label} is unreachable.",
        detail=f"{type(exc).__name__}: {exc}",
    )
