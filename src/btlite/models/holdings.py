"""Vertraege fuer das gespeicherte Depot und die Watchlists.

Diese Modelle bilden **Nutzereingaben** ab, keine Provider-Antworten. Sie
tragen deshalb keine :class:`~btlite.models.common.Provenance`: die Herkunft
ist immer der Nutzer selbst.

Abgrenzung zu :class:`~btlite.models.portfolio.Position`: jene beschreibt ein
**Zielgewicht** aus der Optimierung, dieses Modul eine **tatsaechlich
gehaltene Stueckzahl** zu einem Kaufkurs. Beides in einen Typ zu zwingen
haette das Gewichtsfeld in der Haelfte der Faelle bedeutungslos gemacht.

Bewusste Auslassung: der **aktuelle Wert** einer Position steht hier nicht.
Er braeuchte Kurse, und Modelle holen keine Daten (Architekturregel 2). Was
ohne Kurse berechenbar ist - der Einstandswert - ist als Property vorhanden.

Einheiten:
    - ``shares``: Stueckzahl, gebrochen zulaessig (Teilaktien).
    - ``buy_price``: Kurs je Stueck in ``currency``, absolut.
    - ``cost_basis``: Stueckzahl mal Kaufkurs, in ``currency``.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Self

from pydantic import Field, model_validator

from btlite.models.common import BTModel, Currency, Ticker

#: Schema-Version der persistierten Datei. Wird bei Formatwechseln erhoeht.
HOLDINGS_SCHEMA_VERSION: int = 1


class Holding(BTModel):
    """Eine tatsaechlich gehaltene Position im Depot.

    Anders als :class:`~btlite.models.portfolio.Position` traegt dieses
    Modell kein Gewicht, sondern die reale Stueckzahl und den Einstandskurs.
    """

    ticker: Ticker = Field(description="Kanonisches Symbol.")
    shares: float = Field(
        gt=0,
        description="Gehaltene Stueckzahl. Gebrochene Werte fuer Teilaktien zulaessig.",
    )
    buy_price: float = Field(gt=0, description="Kaufkurs je Stueck in ``currency``.")
    buy_date: date = Field(description="Kaufdatum (Kalenderdatum, keine Uhrzeit).")
    currency: Currency = Field(
        default=Currency.EUR,
        description=(
            "Waehrung des Kaufkurses. Default Euro - die Waehrung des Nutzers. "
            "Wer an einer US-Boerse kauft, stellt hier auf USD um; der Einstand "
            "wird dann zum Kurs des Kauftages umgerechnet."
        ),
    )
    note: str | None = Field(default=None, max_length=500, description="Freie Notiz des Nutzers.")

    @property
    def cost_basis(self) -> float:
        """Einstandswert der Position.

        Returns:
            Stueckzahl mal Kaufkurs, in ``currency``. Ohne Gebuehren - die
            sind in dieser Ablage nicht erfasst.
        """
        return self.shares * self.buy_price


class Portfolio(BTModel):
    """Ein benanntes Depot aus mehreren Haltungen.

    Mehrere Haltungen desselben Tickers sind zulaessig: wer zweimal zu
    verschiedenen Kursen gekauft hat, verliert diese Information sonst. Der
    gewichtete Einstandskurs je Ticker liefert :meth:`cost_basis_by_ticker`.
    """

    name: str = Field(min_length=1, max_length=100, description="Anzeigename des Depots.")
    holdings: list[Holding] = Field(default_factory=list, description="Gehaltene Positionen.")
    currency: Currency = Field(
        default=Currency.EUR,
        description=(
            "Berichtswaehrung des Depots insgesamt. Default Euro. Positionen "
            "duerfen in anderer Waehrung gekauft sein und werden umgerechnet."
        ),
    )
    created_at: datetime = Field(description="Erstellzeitpunkt, tz-aware UTC.")
    updated_at: datetime = Field(description="Zeitpunkt der letzten Aenderung, tz-aware UTC.")
    note: str | None = Field(default=None, max_length=500, description="Freie Notiz des Nutzers.")

    @model_validator(mode="after")
    def _validate_timestamps(self) -> Self:
        """Prueft, dass beide Zeitstempel tz-aware sind.

        Returns:
            Das validierte Modell.

        Raises:
            ValueError: Wenn ein Zeitstempel naiv ist.
        """
        for label, ts in (("created_at", self.created_at), ("updated_at", self.updated_at)):
            if ts.tzinfo is None:
                msg = f"{label} muss tz-aware sein (UTC)"
                raise ValueError(msg)
        return self

    @property
    def cost_basis(self) -> float:
        """Gesamter Einstandswert des Depots.

        Waehrungen werden **nicht** umgerechnet: die Summe ist nur belastbar,
        wenn alle Haltungen dieselbe Waehrung tragen. :meth:`is_single_currency`
        sagt, ob das der Fall ist.

        Returns:
            Summe aller Einstandswerte, in ``currency``.
        """
        return sum(h.cost_basis for h in self.holdings)

    @property
    def tickers(self) -> list[str]:
        """Enthaltene Symbole, ohne Wiederholung und stabil sortiert.

        Returns:
            Sortierte Liste der Ticker.
        """
        return sorted({h.ticker for h in self.holdings})

    @property
    def is_single_currency(self) -> bool:
        """Ob alle Haltungen dieselbe Waehrung tragen.

        Returns:
            True wenn das Depot leer ist oder nur eine Waehrung vorkommt.
        """
        return len({h.currency for h in self.holdings}) <= 1

    def shares_of(self, ticker: str) -> float:
        """Summiert die Stueckzahl eines Symbols ueber alle Haltungen.

        Args:
            ticker: Symbol, wird normalisiert.

        Returns:
            Gesamte Stueckzahl, 0.0 wenn das Symbol nicht gehalten wird.
            Die 0.0 ist hier kein Fehlwert, sondern eine echte Aussage.
        """
        from btlite.models.common import normalize_ticker

        key = normalize_ticker(ticker)
        return sum(h.shares for h in self.holdings if h.ticker == key)

    def cost_basis_by_ticker(self) -> dict[str, float]:
        """Einstandswerte je Symbol, ueber mehrere Kaeufe aggregiert.

        Returns:
            Abbildung Ticker auf Einstandswert in der jeweiligen Waehrung
            der Haltungen.
        """
        out: dict[str, float] = {}
        for holding in self.holdings:
            out[holding.ticker] = out.get(holding.ticker, 0.0) + holding.cost_basis
        return out


class WatchlistEntry(BTModel):
    """Ein beobachteter Titel ohne Besitz."""

    ticker: Ticker = Field(description="Kanonisches Symbol.")
    added_at: datetime = Field(description="Aufnahmezeitpunkt, tz-aware UTC.")
    note: str | None = Field(default=None, max_length=500, description="Freie Notiz des Nutzers.")

    @model_validator(mode="after")
    def _validate_added_at(self) -> Self:
        """Prueft, dass der Aufnahmezeitpunkt tz-aware ist.

        Returns:
            Das validierte Modell.

        Raises:
            ValueError: Wenn der Zeitstempel naiv ist.
        """
        if self.added_at.tzinfo is None:
            msg = "added_at muss tz-aware sein (UTC)"
            raise ValueError(msg)
        return self


class Watchlist(BTModel):
    """Eine benannte Liste beobachteter Titel.

    Jedes Symbol kommt hoechstens einmal vor - anders als im Depot, wo
    mehrere Kaeufe desselben Titels fachlich sinnvoll sind.
    """

    name: str = Field(min_length=1, max_length=100, description="Anzeigename der Liste.")
    entries: list[WatchlistEntry] = Field(default_factory=list, description="Beobachtete Titel.")
    created_at: datetime = Field(description="Erstellzeitpunkt, tz-aware UTC.")
    updated_at: datetime = Field(description="Zeitpunkt der letzten Aenderung, tz-aware UTC.")

    @model_validator(mode="after")
    def _validate_unique_tickers(self) -> Self:
        """Prueft Eindeutigkeit der Symbole und tz-Bewusstsein der Zeitstempel.

        Returns:
            Das validierte Modell.

        Raises:
            ValueError: Bei doppeltem Symbol oder naivem Zeitstempel.
        """
        for label, ts in (("created_at", self.created_at), ("updated_at", self.updated_at)):
            if ts.tzinfo is None:
                msg = f"{label} muss tz-aware sein (UTC)"
                raise ValueError(msg)
        seen: set[str] = set()
        duplicates: set[str] = set()
        for entry in self.entries:
            if entry.ticker in seen:
                duplicates.add(entry.ticker)
            seen.add(entry.ticker)
        if duplicates:
            msg = f"Watchlist enthaelt doppelte Symbole: {sorted(duplicates)}"
            raise ValueError(msg)
        return self

    @property
    def tickers(self) -> list[str]:
        """Beobachtete Symbole in Eingabereihenfolge.

        Returns:
            Liste der Ticker.
        """
        return [e.ticker for e in self.entries]


class HoldingsFile(BTModel):
    """Wurzel der persistierten JSON-Datei.

    Mehrere benannte Depots und Watchlists stehen nebeneinander, jeweils
    indiziert nach ihrem Namen. Die Schema-Version macht spaetere
    Formatwechsel erkennbar, statt sie still fehlschlagen zu lassen.
    """

    schema_version: int = Field(
        default=HOLDINGS_SCHEMA_VERSION,
        ge=1,
        description="Version des Dateiformats, siehe HOLDINGS_SCHEMA_VERSION.",
    )
    portfolios: dict[str, Portfolio] = Field(
        default_factory=dict, description="Depots, indiziert nach ``Portfolio.name``."
    )
    watchlists: dict[str, Watchlist] = Field(
        default_factory=dict, description="Watchlists, indiziert nach ``Watchlist.name``."
    )
    updated_at: datetime | None = Field(
        default=None, description="Zeitpunkt des letzten Schreibvorgangs, tz-aware UTC."
    )

    @model_validator(mode="after")
    def _validate_keys(self) -> Self:
        """Prueft, dass Schluessel und Objektnamen uebereinstimmen.

        Returns:
            Das validierte Modell.

        Raises:
            ValueError: Wenn ein Schluessel vom Namen des Objekts abweicht.
        """
        for key, portfolio in self.portfolios.items():
            if key != portfolio.name:
                msg = f"Portfolio-Schluessel {key!r} passt nicht zum Namen {portfolio.name!r}"
                raise ValueError(msg)
        for key, watchlist in self.watchlists.items():
            if key != watchlist.name:
                msg = f"Watchlist-Schluessel {key!r} passt nicht zum Namen {watchlist.name!r}"
                raise ValueError(msg)
        if self.updated_at is not None and self.updated_at.tzinfo is None:
            msg = "updated_at muss tz-aware sein (UTC)"
            raise ValueError(msg)
        return self

    @property
    def is_empty(self) -> bool:
        """Ob weder Depot noch Watchlist hinterlegt ist.

        Returns:
            True bei vollstaendig leerem Bestand.
        """
        return not self.portfolios and not self.watchlists
