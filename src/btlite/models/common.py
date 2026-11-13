"""Gemeinsame Bausteine aller Datenvertraege.

Enthaelt Enums, Spaltennamen-Konstanten, die Basis-Modellkonfiguration und
den Validator, der DataFrames auf Schema und UTC-Index prueft.

Wichtig: Spaltennamen sind hier zentral definiert. Kein Modul schreibt
Spaltennamen als String-Literal - immer ueber diese Konstanten.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Final

import pandas as pd
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from btlite.core.errors import DataValidationError

# --------------------------------------------------------------------------
# Spaltennamen - Single Source of Truth
# --------------------------------------------------------------------------

#: Spalten einer OHLCV-Zeitreihe. Index ist immer tz-aware UTC, Name "date".
OHLCV_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "volume")

#: Optionale Zusatzspalten einer Kursreihe.
OHLCV_OPTIONAL_COLUMNS: Final[tuple[str, ...]] = ("dividends", "stock_splits")

#: Name des Zeitindex in allen Zeitreihen-Payloads.
DATE_INDEX_NAME: Final[str] = "date"

#: Index einer Statement-Tabelle: normalisierte Positionsbezeichner.
STATEMENT_INDEX_NAME: Final[str] = "line_item"

#: Spaltenname einer Rendite-Zeitreihe.
RETURN_COLUMN: Final[str] = "return"

#: Spalten eines Bollinger-Band-DataFrame aus ``analytics/technicals.py``.
#: Gemeinsamer Vertrag zwischen Analytics (erzeugt) und Viz (stellt dar).
BOLLINGER_COLUMNS: Final[tuple[str, str, str]] = ("upper", "middle", "lower")

#: Spalten eines MACD-DataFrame aus ``analytics/technicals.py``.
MACD_COLUMNS: Final[tuple[str, str, str]] = ("macd", "signal", "histogram")


class Currency(StrEnum):
    """Waehrungen, die das Produkt kennt. ISO-4217."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CHF = "CHF"
    JPY = "JPY"
    CAD = "CAD"
    AUD = "AUD"
    HKD = "HKD"
    SEK = "SEK"
    NOK = "NOK"
    DKK = "DKK"
    OTHER = "OTHER"


class Interval(StrEnum):
    """Kerzen-Intervall einer Kursreihe.

    Die Werte entsprechen bewusst den yfinance-Strings, damit die
    Provider-Schicht ohne Mapping-Tabelle auskommt.
    """

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    D1 = "1d"
    W1 = "1wk"
    MO1 = "1mo"

    @property
    def is_intraday(self) -> bool:
        """Ob das Intervall unterhalb eines Handelstages liegt.

        Returns:
            True fuer Intraday-Intervalle.
        """
        return self in {Interval.M1, Interval.M5, Interval.M15, Interval.M30, Interval.H1}

    @property
    def periods_per_year(self) -> float:
        """Annualisierungsfaktor fuer dieses Intervall.

        Basis sind 252 Handelstage zu 6.5 Handelsstunden.

        Returns:
            Anzahl Perioden pro Jahr, dezimal.
        """
        return {
            Interval.M1: 252 * 390,
            Interval.M5: 252 * 78,
            Interval.M15: 252 * 26,
            Interval.M30: 252 * 13,
            Interval.H1: 252 * 6.5,
            Interval.D1: 252,
            Interval.W1: 52,
            Interval.MO1: 12,
        }[self]


#: Intervalle unterhalb eines Handelstags. Fuer sie gelten eigene
#: Grenzen der Quelle und ein kurzer Cache, siehe docs/DATA_SOURCES.md.
INTRADAY_INTERVALS: Final[frozenset[Interval]] = frozenset(
    {Interval.M1, Interval.M5, Interval.M15, Interval.M30, Interval.H1}
)


class Period(StrEnum):
    """Berichtsperiode eines Financial Statement."""

    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    TTM = "ttm"


class AssetClass(StrEnum):
    """Grobe Instrumentenklasse."""

    EQUITY = "equity"
    ETF = "etf"
    INDEX = "index"
    CRYPTO = "crypto"
    FUND = "fund"
    OTHER = "other"


class DataSource(StrEnum):
    """Herkunft eines Datensatzes. Wird im UI sichtbar angezeigt."""

    YFINANCE = "yfinance"
    FINVIZ = "finviz"
    #: Indexbestaende, die keine Finanzquelle fuehrt - der DAX etwa. Eine
    #: eigene Herkunft, weil der Nachweis im UI sichtbar ist: "yfinance"
    #: an einer Wikipedia-Liste waere eine Falschaussage.
    WIKIPEDIA = "wikipedia"
    FIXTURE = "fixture"
    COMPUTED = "computed"
    USER = "user"


# --------------------------------------------------------------------------
# Basis-Modell
# --------------------------------------------------------------------------


class BTModel(BaseModel):
    """Basisklasse aller BT-Lite-Modelle.

    Konfiguration bewusst streng: unbekannte Felder sind ein Fehler, damit
    Vertragsbrueche frueh auffallen. ``arbitrary_types_allowed`` erlaubt
    ``pd.DataFrame`` als Feldtyp.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=False,
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class Provenance(BTModel):
    """Herkunftsnachweis eines Datensatzes.

    Jedes Modell, das aus einer externen Quelle stammt, traegt diesen Block.
    Das UI zeigt Quelle, Abrufzeitpunkt und Delay-Hinweis sichtbar an -
    nichts wird als Echtzeit ausgegeben, was keine ist.
    """

    source: DataSource = Field(description="Herkunft der Daten.")
    fetched_at: datetime = Field(description="Abrufzeitpunkt, tz-aware UTC.")
    from_cache: bool = Field(default=False, description="Antwort kam aus dem lokalen Cache.")
    delay_minutes: int | None = Field(
        default=None,
        ge=0,
        description="Bekannter Verzug der Quelle in Minuten. None = unbekannt.",
    )
    note: str | None = Field(default=None, description="Freitext-Hinweis fuer das UI.")


# --------------------------------------------------------------------------
# DataFrame-Validierung
# --------------------------------------------------------------------------


def require_columns(df: pd.DataFrame, columns: tuple[str, ...], *, what: str) -> None:
    """Prueft, dass alle Pflichtspalten vorhanden sind.

    Args:
        df: Zu pruefender DataFrame.
        columns: Erwartete Spaltennamen.
        what: Bezeichnung fuer die Fehlermeldung.

    Raises:
        DataValidationError: Wenn Spalten fehlen.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        msg = f"{what}: Pflichtspalten fehlen"
        raise DataValidationError(msg, detail=f"fehlend={missing}, vorhanden={list(df.columns)}")


def require_utc_index(df: pd.DataFrame, *, what: str) -> None:
    """Prueft, dass der Index ein monoton steigender UTC-``DatetimeIndex`` ist.

    Args:
        df: Zu pruefender DataFrame.
        what: Bezeichnung fuer die Fehlermeldung.

    Raises:
        DataValidationError: Wenn der Index nicht dem Vertrag entspricht.
    """
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        msg = f"{what}: Index muss ein DatetimeIndex sein"
        raise DataValidationError(msg, detail=f"typ={type(idx).__name__}")
    if idx.tz is None:
        msg = f"{what}: Index muss tz-aware sein (UTC)"
        raise DataValidationError(msg, detail="tz=None")
    if str(idx.tz) not in {"UTC", "utc"}:
        msg = f"{what}: Index muss in UTC vorliegen"
        raise DataValidationError(msg, detail=f"tz={idx.tz}")
    if not idx.is_monotonic_increasing:
        msg = f"{what}: Index muss aufsteigend sortiert sein"
        raise DataValidationError(msg, detail="is_monotonic_increasing=False")
    if idx.has_duplicates:
        msg = f"{what}: Index enthaelt Duplikate"
        raise DataValidationError(msg, detail=f"n_dupes={int(idx.duplicated().sum())}")


def _coerce_dataframe(v: Any) -> pd.DataFrame:
    """Akzeptiert nur echte DataFrames als Payload.

    Wirft ``ValueError`` statt ``TypeError``, damit Pydantic den Fehler in
    einen ``ValidationError`` einsammelt, wie bei jedem anderen Feld auch.

    Args:
        v: Zu pruefender Wert.

    Returns:
        Der unveraenderte DataFrame.

    Raises:
        ValueError: Wenn der Wert kein DataFrame ist.
    """
    if not isinstance(v, pd.DataFrame):
        msg = f"Erwartet pd.DataFrame, erhalten {type(v).__name__}"
        raise ValueError(msg)
    return v


#: Feldtyp fuer DataFrame-Payloads in Modellen.
DataFrameField = Annotated[pd.DataFrame, BeforeValidator(_coerce_dataframe)]


def normalize_ticker(value: str) -> str:
    """Normalisiert ein Ticker-Symbol auf die kanonische Form.

    Grossbuchstaben, keine umgebenden Leerzeichen. Punkte und Bindestriche
    bleiben erhalten, da sie bei Klassen-Aktien und Nicht-US-Listings
    bedeutungstragend sind (``BRK-B``, ``BMW.DE``).

    Args:
        value: Roh-Symbol.

    Returns:
        Normalisiertes Symbol.

    Raises:
        ValueError: Wenn das Symbol leer ist.
    """
    out = value.strip().upper()
    if not out:
        msg = "Ticker darf nicht leer sein"
        raise ValueError(msg)
    return out


#: Feldtyp fuer Ticker-Symbole. Normalisiert automatisch.
Ticker = Annotated[str, BeforeValidator(normalize_ticker), Field(min_length=1, max_length=24)]
