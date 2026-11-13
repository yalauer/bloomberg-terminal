"""Vertraege fuer Instrumente, Quotes und Kurshistorien.

Block 1 des Produkts (Prices & Charts) baut ausschliesslich auf diesen
Modellen auf.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Self, cast

import pandas as pd
from pydantic import Field, model_validator

from btlite.models.common import (
    OHLCV_COLUMNS,
    RETURN_COLUMN,
    AssetClass,
    BTModel,
    Currency,
    DataFrameField,
    Interval,
    Provenance,
    Ticker,
    require_columns,
    require_utc_index,
)


class Instrument(BTModel):
    """Stammdaten eines handelbaren Instruments.

    Fuellt die Seite ``DES`` (Description) und die Kopfzeile aller anderen
    Seiten.
    """

    ticker: Ticker = Field(description="Kanonisches Symbol, zum Beispiel AAPL.")
    name: str | None = Field(default=None, description="Vollstaendiger Name des Emittenten.")
    asset_class: AssetClass = Field(default=AssetClass.EQUITY, description="Instrumentenklasse.")
    currency: Currency = Field(default=Currency.USD, description="Handelswaehrung.")
    exchange: str | None = Field(default=None, description="Kuerzel der Boerse, etwa NMS.")
    sector: str | None = Field(default=None, description="Sektor laut Quelle.")
    industry: str | None = Field(default=None, description="Branche laut Quelle.")
    country: str | None = Field(default=None, description="Sitzland des Emittenten.")
    website: str | None = Field(default=None, description="Unternehmenswebsite.")
    description: str | None = Field(default=None, description="Geschaeftsbeschreibung, Langtext.")
    employees: int | None = Field(default=None, ge=0, description="Anzahl Mitarbeiter.")
    market_cap: float | None = Field(
        default=None, ge=0, description="Marktkapitalisierung in ``currency``, absolut."
    )
    shares_outstanding: float | None = Field(
        default=None, ge=0, description="Ausstehende Aktien, Stueck."
    )
    dividend_rate: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Jaehrliche Dividende je Aktie in ``currency``, absolut. Basis der "
            "letzten zwoelf Monate. None bedeutet: keine Ausschuettung bekannt."
        ),
    )
    dividend_yield: float | None = Field(
        default=None,
        ge=0,
        description="Dividendenrendite der letzten zwoelf Monate, dezimal (0.023 = 2.3 Prozent).",
    )
    provenance: Provenance | None = Field(default=None, description="Herkunft der Stammdaten.")


class SearchHit(BTModel):
    """Ein Treffer der Symbolsuche nach Namen.

    Fuellt die Trefferliste, mit der ein Nutzer von ``servicenow`` zu ``NOW``
    kommt, ohne das Kuerzel vorher zu kennen.

    ``score`` ist das dimensionslose Relevanzmass der Quelle beziehungsweise
    des lokalen Rankings aus ``data/search.py``. Groesser ist besser; die
    Skala ist zwischen Quellen **nicht** vergleichbar und taugt nur zum
    Sortieren innerhalb einer Ergebnisliste.
    """

    ticker: Ticker = Field(description="Kanonisches Symbol, zum Beispiel NOW.")
    name: str | None = Field(default=None, description="Name des Emittenten laut Quelle.")
    exchange: str | None = Field(
        default=None,
        description="Anzeigename der Boerse, etwa NYSE. Kuerzel wie NYQ werden aufgeloest.",
    )
    asset_class: AssetClass = Field(default=AssetClass.EQUITY, description="Instrumentenklasse.")
    sector: str | None = Field(default=None, description="Sektor laut Quelle.")
    score: float | None = Field(
        default=None,
        ge=0,
        description="Relevanz, dimensionslos. Groesser ist besser. None = unbekannt.",
    )

    @property
    def display(self) -> str:
        """Anzeigetext aus Boerse und Kuerzel, etwa ``NYSE:NOW``.

        Returns:
            ``<BOERSE>:<TICKER>``, oder nur das Kuerzel wenn die Quelle keine
            Boerse nennt.
        """
        if not self.exchange:
            return self.ticker
        return f"{self.exchange}:{self.ticker}"

    @property
    def label(self) -> str:
        """Vollstaendige Trefferzeile fuer das UI.

        Returns:
            ``<BOERSE>:<TICKER> - <Name>``, oder nur :attr:`display` wenn die
            Quelle keinen Namen nennt.
        """
        if not self.name:
            return self.display
        return f"{self.display} - {self.name}"


class EarningsDate(BTModel):
    """Naechster Quartalszahlen-Termin eines Instruments.

    Fuellt je eine Spalte in ``PORT`` und ``WL``. Die Aussage ist bewusst
    schmal: **der naechste** Termin, nicht die Terminhistorie.

    ``next_date`` ist ein Kalendertag ohne Uhrzeit und ohne Zeitzone. Die Quelle
    nennt zwar eine Uhrzeit, sie ist aber der Boersenzeitzone des Emittenten
    zugeordnet und fuer die Anzeige eines Termins ohne Belang; ein in UTC
    umgerechneter Zeitstempel koennte den Kalendertag sogar verschieben.
    Deshalb weicht dieses Modell bewusst von der UTC-Regel fuer *Zeitreihen*
    ab - ein Termin ist keine Zeitreihe.

    ``None`` heisst **kein Termin bekannt**, nie "Fehler" und nie "heute".
    Welcher der beiden Gruende dahintersteckt, sagt :attr:`has_earnings`.
    """

    ticker: Ticker = Field(description="Kanonisches Symbol.")
    next_date: date | None = Field(
        default=None,
        description=(
            "Naechster Termin als Kalendertag, lokal zur Boerse des Emittenten. "
            "None = kein Termin bekannt. Liegt nie in der Vergangenheit."
        ),
    )
    has_earnings: bool = Field(
        default=True,
        description=(
            "False = die Quelle hat **aktiv belegt**, dass das Instrument keine "
            "Quartalszahlen fuehrt (ETF, Krypto, Index). Dann ist ``next_date`` "
            "immer None, und zwar endgueltig; das UI zeigt einen Strich. "
            "True mit ``next_date is None`` heisst dagegen: unbekannt - eine "
            "Aktie ohne angekuendigten Termin, ein nicht eingefrorenes Symbol "
            "oder ein fehlgeschlagener Abruf. Das UI zeigt 'n/a'. Im Zweifel "
            "gilt True: das ist die schwaechere und damit ehrlichere Aussage."
        ),
    )
    estimated: bool = Field(
        default=True,
        description=(
            "True = der Termin ist nicht als bestaetigt ausgewiesen. Yahoo "
            "kennzeichnet Bestaetigung nicht, der Wert ist daher immer True. "
            "Das Feld haelt die Unsicherheit im Vertrag fest, damit das UI sie "
            "anzeigen kann, statt sie zu verschweigen."
        ),
    )
    provenance: Provenance | None = Field(default=None, description="Herkunft des Termins.")

    @model_validator(mode="after")
    def _validate_consistency(self) -> Self:
        """Stellt sicher, dass ein Instrument ohne Earnings auch kein Datum traegt.

        Returns:
            Das validierte Modell.

        Raises:
            ValueError: Wenn ``has_earnings`` False ist, aber ein Datum gesetzt.
        """
        if not self.has_earnings and self.next_date is not None:
            msg = "has_earnings=False schliesst ein Datum aus"
            raise ValueError(msg)
        return self


class Quote(BTModel):
    """Momentaufnahme des Preises eines Instruments.

    Achtung: yfinance liefert **keine** Echtzeitdaten. ``provenance`` traegt
    den bekannten Verzug, das UI zeigt ihn sichtbar an.
    """

    ticker: Ticker = Field(description="Kanonisches Symbol.")
    price: float = Field(gt=0, description="Letzter Preis in ``currency``.")
    currency: Currency = Field(default=Currency.USD, description="Waehrung des Preises.")
    previous_close: float | None = Field(
        default=None, gt=0, description="Schlusskurs des Vortages, Basis der Tagesveraenderung."
    )
    open: float | None = Field(default=None, gt=0, description="Eroeffnungskurs des Tages.")
    day_high: float | None = Field(default=None, gt=0, description="Tageshoch.")
    day_low: float | None = Field(default=None, gt=0, description="Tagestief.")
    volume: float | None = Field(default=None, ge=0, description="Tagesvolumen in Stueck.")
    bid: float | None = Field(default=None, gt=0, description="Bestes Geld.")
    ask: float | None = Field(default=None, gt=0, description="Bester Brief.")
    timestamp: datetime = Field(description="Zeitpunkt des Preises, tz-aware UTC.")
    provenance: Provenance = Field(description="Herkunft und Verzug.")

    @property
    def change_abs(self) -> float | None:
        """Tagesveraenderung absolut in ``currency``.

        Returns:
            Differenz zum Vortagesschluss, oder None wenn dieser fehlt.
        """
        if self.previous_close is None:
            return None
        return self.price - self.previous_close

    @property
    def change_pct(self) -> float | None:
        """Tagesveraenderung relativ, dezimal (0.015 = 1.5 Prozent).

        Returns:
            Relative Veraenderung, oder None wenn der Vortagesschluss fehlt.
        """
        if self.previous_close is None or self.previous_close == 0:
            return None
        return (self.price - self.previous_close) / self.previous_close


class PriceSeries(BTModel):
    """OHLCV-Kurshistorie eines Instruments.

    Der Payload ist ein DataFrame mit tz-aware UTC-``DatetimeIndex`` namens
    ``date`` und den Spalten aus :data:`OHLCV_COLUMNS`. Optional koennen
    ``dividends`` und ``stock_splits`` enthalten sein.

    ``adjusted`` dokumentiert, ob die Kurse split- und dividendenbereinigt
    sind. Diese Unterscheidung ist fuer Backtests entscheidend und wird nie
    implizit angenommen.
    """

    ticker: Ticker = Field(description="Kanonisches Symbol.")
    interval: Interval = Field(default=Interval.D1, description="Kerzen-Intervall.")
    currency: Currency = Field(default=Currency.USD, description="Waehrung der Kurse.")
    adjusted: bool = Field(
        description="True = split- und dividendenbereinigt (yfinance auto_adjust)."
    )
    data: DataFrameField = Field(description="OHLCV-Payload, siehe Klassendoku.")
    provenance: Provenance = Field(description="Herkunft der Kursreihe.")

    @model_validator(mode="after")
    def _validate_payload(self) -> Self:
        """Prueft Schema und Index des Payloads.

        Returns:
            Das validierte Modell.
        """
        what = f"PriceSeries[{self.ticker}]"
        require_columns(self.data, OHLCV_COLUMNS, what=what)
        require_utc_index(self.data, what=what)
        return self

    @property
    def start(self) -> datetime | None:
        """Erster Zeitstempel der Reihe.

        Returns:
            Frueheste Beobachtung, oder None bei leerer Reihe.
        """
        if self.data.empty:
            return None
        first: pd.Timestamp = cast("pd.DatetimeIndex", self.data.index)[0]
        return first.to_pydatetime()

    @property
    def end(self) -> datetime | None:
        """Letzter Zeitstempel der Reihe.

        Returns:
            Spaeteste Beobachtung, oder None bei leerer Reihe.
        """
        if self.data.empty:
            return None
        last: pd.Timestamp = cast("pd.DatetimeIndex", self.data.index)[-1]
        return last.to_pydatetime()

    @property
    def close(self) -> pd.Series:
        """Schlusskurs-Reihe.

        Returns:
            Series der Spalte ``close`` mit UTC-Index.
        """
        return self.data["close"]

    def returns(self, *, log: bool = False) -> pd.Series:
        """Berechnet einfache oder logarithmische Perioden-Renditen.

        Args:
            log: Wenn True, logarithmische statt einfacher Renditen.

        Returns:
            Rendite-Series, dezimal, ohne den ersten (undefinierten) Wert.
        """
        import numpy as np

        close = self.close
        out: pd.Series = (
            cast("pd.Series", np.log(close / close.shift(1))) if log else close.pct_change()
        )
        out.name = RETURN_COLUMN
        return out.dropna()


class PriceHistoryRequest(BTModel):
    """Anfrage nach einer Kurshistorie.

    Entweder ``period`` (relativ, zum Beispiel ``"5y"``) oder ``start``/``end``
    (absolut) setzen - nicht beides.
    """

    tickers: list[Ticker] = Field(min_length=1, description="Ein oder mehrere Symbole.")
    interval: Interval = Field(default=Interval.D1, description="Kerzen-Intervall.")
    period: str | None = Field(
        default="1y",
        description='Relativer Zeitraum im yfinance-Format, etwa "1y", "5y", "max".',
    )
    start: datetime | None = Field(default=None, description="Startzeitpunkt, tz-aware UTC.")
    end: datetime | None = Field(default=None, description="Endzeitpunkt, tz-aware UTC.")
    auto_adjust: bool = Field(default=True, description="Bereinigte Kurse anfordern.")

    @model_validator(mode="after")
    def _validate_range(self) -> Self:
        """Stellt sicher, dass relativer und absoluter Zeitraum sich nicht mischen.

        Returns:
            Das validierte Modell.

        Raises:
            ValueError: Bei widerspruechlicher Zeitraumangabe.
        """
        if self.start is not None or self.end is not None:
            if self.period is not None:
                msg = "period und start/end schliessen sich aus"
                raise ValueError(msg)
            if self.start is not None and self.end is not None and self.start >= self.end:
                msg = "start muss vor end liegen"
                raise ValueError(msg)
        elif self.period is None:
            msg = "Entweder period oder start/end angeben"
            raise ValueError(msg)
        return self


class PriceComparison(BTModel):
    """Normierter Mehr-Ticker-Vergleich fuer die Seite ``GP``.

    Der Payload ist ein Wide-DataFrame: eine Spalte je Ticker, Index ist der
    gemeinsame UTC-``DatetimeIndex``. Werte sind auf den ersten gemeinsamen
    Beobachtungstag normiert (Basis 100), sofern ``normalized`` True ist.
    """

    tickers: list[Ticker] = Field(min_length=1, description="Verglichene Symbole.")
    interval: Interval = Field(default=Interval.D1, description="Kerzen-Intervall.")
    normalized: bool = Field(default=True, description="Auf Basis 100 normiert.")
    data: DataFrameField = Field(description="Wide-DataFrame, eine Spalte je Ticker.")
    provenance: Provenance = Field(description="Herkunft der zugrunde liegenden Reihen.")

    @model_validator(mode="after")
    def _validate_payload(self) -> Self:
        """Prueft Index und Spaltenabdeckung.

        Returns:
            Das validierte Modell.
        """
        what = "PriceComparison"
        require_utc_index(self.data, what=what)
        require_columns(self.data, tuple(self.tickers), what=what)
        return self
