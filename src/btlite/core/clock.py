"""Zeit-Utilities. Einzige Quelle fuer "jetzt" im gesamten Projekt.

Architekturregel 6: Alle Zeitreihen tragen einen ``DatetimeIndex``, sind
tz-aware in UTC, und werden ausschliesslich an der UI-Grenze in die
Anzeigezeitzone konvertiert. Kein Modul ruft ``datetime.now()`` direkt auf -
das geht immer ueber :func:`utc_now`, damit Tests die Zeit einfrieren koennen.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    import pandas as pd


def utc_now() -> datetime:
    """Liefert den aktuellen Zeitpunkt als tz-aware UTC-Datetime.

    Returns:
        Aktueller Zeitpunkt mit ``tzinfo=UTC``.
    """
    return datetime.now(UTC)


def utc_today() -> date:
    """Liefert das heutige Datum in UTC.

    Returns:
        Aktuelles Kalenderdatum in UTC.
    """
    return utc_now().date()


def ensure_utc(ts: datetime) -> datetime:
    """Normalisiert einen Zeitstempel auf tz-aware UTC.

    Naive Zeitstempel werden als bereits in UTC vorliegend interpretiert -
    Provider liefern teils naive Werte, die faktisch UTC sind.

    Args:
        ts: Zu normalisierender Zeitstempel.

    Returns:
        Aequivalenter Zeitstempel mit ``tzinfo=UTC``.
    """
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def to_display_tz(ts: datetime, tz_name: str) -> datetime:
    """Konvertiert einen UTC-Zeitstempel in die Anzeigezeitzone.

    Nur an der UI-Grenze aufrufen.

    Args:
        ts: Zeitstempel, idealerweise tz-aware.
        tz_name: IANA-Zeitzonenname, zum Beispiel ``"Europe/Berlin"``.

    Returns:
        Zeitstempel in der Zielzeitzone.
    """
    return ensure_utc(ts).astimezone(ZoneInfo(tz_name))


def index_to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Erzwingt einen tz-aware UTC-``DatetimeIndex`` auf einem DataFrame.

    Arbeitet auf einer Kopie und laesst das Original unveraendert.

    Args:
        df: DataFrame mit datumsartigem Index.

    Returns:
        Kopie des DataFrame mit nach UTC lokalisiertem beziehungsweise
        konvertiertem Index.

    Raises:
        TypeError: Wenn der Index nicht in einen ``DatetimeIndex``
            konvertierbar ist.
    """
    import pandas as pd  # lokal, damit core/ importleicht bleibt

    out = df.copy()
    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):
        try:
            idx = pd.DatetimeIndex(idx)
        except (TypeError, ValueError) as exc:  # pragma: no cover - Defensive
            msg = f"Index ist nicht in einen DatetimeIndex konvertierbar: {type(df.index)!r}"
            raise TypeError(msg) from exc

    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out.index = idx
    out.index.name = "date"
    return out
