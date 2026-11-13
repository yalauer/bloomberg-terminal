"""Darstellung von Zahlen, Waehrungen und fehlenden Werten.

Dieses Modul rechnet nicht. Es uebersetzt fertige Modellwerte in Text,
nach den Einheitenkonventionen des Projekts:

* Renditen, Margen und Risikomasse sind **dezimal** (0.31 = 31 Prozent).
* Waehrungsbetraege sind absolut, nicht in Millionen.
* Ein nicht berechenbarer Wert ist ``None`` und erscheint als
  :data:`NA_TEXT`, nie als ``0`` und nie leer. Das ist Akzeptanzkriterium
  des Projekts, kein Stilwunsch.

Dezimaltrennzeichen ist das Komma, Tausendertrennzeichen der Punkt -
deutsche Schreibweise, wie die uebrigen UI-Texte.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Final

from btlite.core.clock import to_display_tz, utc_today
from btlite.models.common import Currency, Provenance

#: Einheitliche Darstellung eines fehlenden Wertes.
NA_TEXT: Final[str] = "n/a"

#: Waehrungssymbole. Nicht gelistete Waehrungen erscheinen mit ISO-Code.
_CURRENCY_SYMBOLS: Final[dict[Currency, str]] = {
    Currency.USD: "$",
    Currency.EUR: "€",
    Currency.GBP: "GBP",
    Currency.CHF: "CHF",
    Currency.JPY: "JPY",
    Currency.CAD: "CAD",
    Currency.AUD: "AUD",
    Currency.HKD: "HKD",
    Currency.SEK: "SEK",
    Currency.NOK: "NOK",
    Currency.DKK: "DKK",
}

#: Stufen der Kuerzung grosser Zahlen, absteigend.
_MAGNITUDES: Final[tuple[tuple[float, str], ...]] = (
    (1e12, "T"),
    (1e9, "B"),
    (1e6, "M"),
    (1e3, "K"),
)

#: Anzeigetexte der Datenquellen.
_SOURCE_LABELS: Final[dict[str, str]] = {
    "yfinance": "yfinance",
    "finviz": "Finviz",
    "fixture": "Fixture (frozen)",
    "computed": "computed locally",
    "user": "user input",
}


def is_missing(value: float | None) -> bool:
    """Prueft, ob ein Wert als fehlend zu behandeln ist.

    ``NaN`` gilt als fehlend: die Rechenschichten liefern ``None``, aber
    aus einem DataFrame kommt derselbe Sachverhalt als ``NaN`` an.

    Args:
        value: Zu pruefender Wert.

    Returns:
        True bei ``None`` oder ``NaN``.
    """
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


def format_number(
    value: float | None,
    *,
    decimals: int = 2,
    thousands: bool = True,
    signed: bool = False,
) -> str:
    """Formatiert eine dimensionslose Zahl.

    Args:
        value: Wert, oder None wenn nicht berechenbar.
        decimals: Nachkommastellen.
        thousands: Tausenderpunkte setzen.
        signed: Positives Vorzeichen ausschreiben.

    Returns:
        Formatierter Text, oder :data:`NA_TEXT`.
    """
    if is_missing(value):
        return NA_TEXT
    spec = f"{'+' if signed else ''}{',' if thousands else ''}.{decimals}f"
    return f"{value:{spec}}"


def format_percent(value: float | None, *, decimals: int = 1, signed: bool = False) -> str:
    """Formatiert einen **dezimalen** Anteil als Prozentwert.

    Args:
        value: Anteil, dezimal (0.31 = 31 Prozent), oder None.
        decimals: Nachkommastellen.
        signed: Positives Vorzeichen ausschreiben.

    Returns:
        Text wie ``"31.0 %"``, oder :data:`NA_TEXT`.
    """
    if is_missing(value) or value is None:
        return NA_TEXT
    return f"{format_number(value * 100.0, decimals=decimals, signed=signed)} %"


def format_compact(value: float | None, *, decimals: int = 2) -> str:
    """Kuerzt grosse Zahlen auf K, M, B oder T.

    Args:
        value: Wert, oder None.
        decimals: Nachkommastellen der gekuerzten Zahl.

    Returns:
        Text wie ``"1.20 B"``, oder :data:`NA_TEXT`.
    """
    if is_missing(value) or value is None:
        return NA_TEXT
    magnitude = abs(value)
    for threshold, suffix in _MAGNITUDES:
        if magnitude >= threshold:
            return f"{format_number(value / threshold, decimals=decimals)} {suffix}"
    return format_number(value, decimals=decimals)


def currency_symbol(currency: Currency | None) -> str:
    """Liefert das Anzeigesymbol einer Waehrung.

    Args:
        currency: Waehrung, oder None.

    Returns:
        Symbol oder ISO-Code. Leerer String bei None.
    """
    if currency is None:
        return ""
    return _CURRENCY_SYMBOLS.get(currency, currency.value)


def format_currency(
    value: float | None,
    currency: Currency | None = None,
    *,
    decimals: int = 2,
    compact: bool = False,
) -> str:
    """Formatiert einen Waehrungsbetrag mit Symbol.

    Args:
        value: Betrag, absolut, oder None.
        currency: Waehrung des Betrags.
        decimals: Nachkommastellen.
        compact: Grosse Betraege kuerzen (1.20 B).

    Returns:
        Text wie ``"$ 1.20 B"``, oder :data:`NA_TEXT`.
    """
    if is_missing(value):
        return NA_TEXT
    number = (
        format_compact(value, decimals=decimals)
        if compact
        else format_number(value, decimals=decimals)
    )
    symbol = currency_symbol(currency)
    return f"{symbol} {number}".strip()


def format_ratio_value(value: float | None, unit: str) -> str:
    """Formatiert eine Kennzahl gemaess der Einheit aus ``models/fundamentals``.

    Args:
        value: Wert der Kennzahl, oder None.
        unit: Einheit: ``"percent"``, ``"days"``, ``"x"``, ``"currency"``
            oder ``"ratio"``.

    Returns:
        Formatierter Text, oder :data:`NA_TEXT`.
    """
    if is_missing(value):
        return NA_TEXT
    if unit == "percent":
        return format_percent(value)
    if unit == "days":
        return f"{format_number(value, decimals=0)} days"
    if unit == "x":
        return f"{format_number(value)}x"
    if unit == "currency":
        return format_compact(value)
    return format_number(value)


def format_integer(value: float | int | None) -> str:
    """Formatiert eine ganze Zahl mit Tausenderpunkten.

    Args:
        value: Wert, oder None.

    Returns:
        Formatierter Text, oder :data:`NA_TEXT`.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return NA_TEXT
    return f"{int(value):,d}"


def format_timestamp(ts: datetime | None, timezone: str) -> str:
    """Formatiert einen UTC-Zeitstempel in der Anzeigezeitzone.

    Die Konvertierung passiert ausschliesslich hier, an der UI-Grenze -
    intern bleibt alles UTC (Architekturregel 5).

    Args:
        ts: Zeitpunkt, tz-aware UTC, oder None.
        timezone: IANA-Name der Anzeigezeitzone.

    Returns:
        Text wie ``"2026-09-04 13:46 CEST"``, oder :data:`NA_TEXT`.
    """
    if ts is None:
        return NA_TEXT
    local = to_display_tz(ts, timezone)
    return f"{local:%Y-%m-%d %H:%M %Z}"


def format_delay(minutes: int | None) -> str:
    """Formatiert einen Verzug in lesbarer Groesse.

    Args:
        minutes: Verzug in Minuten, oder None wenn unbekannt.

    Returns:
        Text wie ``"15 min"``, ``"3.2 h"`` oder ``"15.8 days"``.
    """
    if minutes is None:
        return "unknown"
    if minutes < 90:
        return f"{minutes} min"
    hours = minutes / 60.0
    if hours < 48:
        return f"{format_number(hours, decimals=1)} h"
    return f"{format_number(hours / 24.0, decimals=1)} days"


def source_label(provenance: Provenance) -> str:
    """Liefert den Anzeigenamen der Datenquelle.

    Args:
        provenance: Herkunftsnachweis.

    Returns:
        Lesbarer Quellenname.
    """
    return _SOURCE_LABELS.get(provenance.source.value, provenance.source.value)


def provenance_text(provenance: Provenance | None, timezone: str) -> str:
    """Baut den sichtbaren Datenherkunfts-Hinweis einer Seite.

    Nennt Quelle, Abrufzeitpunkt und Verzug. Nichts wird als Echtzeit
    ausgegeben, was keine ist - der FixtureProvider weist das echte Alter
    seiner Daten aus.

    Args:
        provenance: Herkunftsnachweis, oder None.
        timezone: IANA-Name der Anzeigezeitzone.

    Returns:
        Einzeiliger Hinweistext.
    """
    if provenance is None:
        return "Source: unknown"
    parts = [
        f"Source: {source_label(provenance)}",
        f"Fetched: {format_timestamp(provenance.fetched_at, timezone)}",
        f"Delay: {format_delay(provenance.delay_minutes)}",
    ]
    if provenance.from_cache:
        parts.append("from cache")
    if provenance.note:
        parts.append(provenance.note)
    return " - ".join(parts)


def direction_class(value: float | None) -> str:
    """Liefert die CSS-Klasse fuer eine gerichtete Zahl.

    Farbe traegt die Bedeutung nie allein - die Seiten setzen zusaetzlich
    ein Vorzeichen in den Text.

    Args:
        value: Wert, oder None.

    Returns:
        ``"bt-up"``, ``"bt-down"``, oder ``"bt-na"`` bei fehlendem Wert.
    """
    if is_missing(value) or value is None:
        return "bt-na"
    if value > 0:
        return "bt-up"
    if value < 0:
        return "bt-down"
    return ""


def direction_arrow(value: float | None) -> str:
    """Liefert das Richtungszeichen einer gerichteten Zahl.

    Die Richtungsklassen aus ``btlite.css`` tragen die Bedeutung nie allein
    (Farbe ist kein Traeger). Dieses Zeichen steht deshalb zusaetzlich im
    Text und benennt ausschliesslich die **arithmetische** Richtung der
    Veraenderung, keine Bewertung.

    Args:
        value: Wert, oder None.

    Returns:
        ``"▲"`` bei positivem, ``"▼"`` bei negativem Wert,
        ``"▬"`` bei exakt 0, leerer String bei fehlendem Wert.
    """
    if is_missing(value) or value is None:
        return ""
    if value > 0:
        return "▲"
    if value < 0:
        return "▼"
    return "▬"


def format_change(value: float | None, *, decimals: int = 2) -> str:
    """Formatiert eine **dezimale** Veraenderung mit Richtungszeichen.

    Args:
        value: Relative Veraenderung, dezimal (0.0643 = 6,43 Prozent),
            oder None wenn nicht interpretierbar.
        decimals: Nachkommastellen des Prozentwerts.

    Returns:
        Text wie ``"▲ +6,43 %"``, oder :data:`NA_TEXT`.
    """
    if is_missing(value) or value is None:
        return NA_TEXT
    arrow = direction_arrow(value)
    percent = format_percent(value, decimals=decimals, signed=True)
    return f"{arrow} {percent}".strip()


def value_class(value: float | None) -> str:
    """Liefert die CSS-Klasse fuer einen moeglicherweise fehlenden Wert.

    Args:
        value: Wert, oder None.

    Returns:
        ``"bt-na"`` bei fehlendem Wert, sonst leerer String.
    """
    return "bt-na" if is_missing(value) else ""


#: Text fuer ein Instrument, das grundsaetzlich keine Quartalszahlen fuehrt.
#: Bewusst nicht "n/a": ein ETF hat keinen Termin, das ist eine Aussage und
#: keine Luecke. "n/a" hiesse "wir wissen es nicht" und liesse den Nutzer
#: nach einem Termin suchen, den es nicht gibt.
NO_EARNINGS_TEXT: Final[str] = "\u2014"


def format_earnings_date(
    on: date | None,
    *,
    has_earnings: bool = True,
    today: date | None = None,
) -> str:
    """Formatiert einen Quartalszahlen-Termin mit Abstand in Tagen.

    Args:
        on: Der Termin, oder ``None`` wenn keiner bekannt ist.
        has_earnings: ``False`` fuer Instrumente ohne Quartalszahlen
            (ETF, Krypto, Index). Dann erscheint ein Gedankenstrich statt
            "n/a" - es fehlt nichts, es gibt nichts.
        today: Bezugstag fuer die Tagesangabe. ``None`` nimmt den heutigen
            Tag; der Parameter existiert, damit Tests nicht an einem
            festen Datum haengen.

    Returns:
        Etwa ``"2026-10-29 (in 35d)"``. Ein Termin heute erscheint als
        ``"2026-09-24 (today)"``, ein vergangener ohne Zusatz - er ist
        dann kein "naechster" Termin mehr.
    """
    if not has_earnings:
        return NO_EARNINGS_TEXT
    if on is None:
        return NA_TEXT

    reference = today if today is not None else utc_today()
    days = (on - reference).days
    if days < 0:
        return on.isoformat()
    if days == 0:
        return f"{on.isoformat()} (today)"
    return f"{on.isoformat()} (in {days}d)"
