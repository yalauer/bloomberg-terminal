"""Persistenz der Depots und Watchlists als JSON-Datei.

Bewusst **keine Datenbank**: eine lesbare, sicherbare, von Hand korrigierbare
Datei passt zu einem Einzelplatzwerkzeug ohne Nutzerkonten. Wer sein Depot
retten will, braucht dafuer keinen Client, nur einen Texteditor.

Zwei Eigenschaften sind nicht verhandelbar, weil Nutzerbestand nicht
ersetzbar ist:

Atomares Schreiben:
    Geschrieben wird in eine temporaere Datei im **selben** Verzeichnis,
    danach wird umbenannt. ``os.replace`` ist auf einem Dateisystem atomar.
    Ein abgebrochener Schreibvorgang - abgestuerzter Prozess, volle Platte -
    laesst die bestehende Datei unberuehrt.

Kein stilles Ueberschreiben:
    Eine defekte oder fremde Datei fuehrt zu :class:`PortfolioError`, nie zu
    einem leeren Neuanfang. Ein Tippfehler im JSON darf nicht bedeuten, dass
    der naechste Start das Depot loescht.

Dieses Modul **rechnet nicht mit Kursen**. Es laedt und speichert. Was ein
Depot heute wert ist, entscheidet die Analytics-Schicht mit injizierten
Kursen (Architekturregel 2).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from btlite.core.clock import utc_now
from btlite.core.errors import PortfolioError
from btlite.core.logging import get_logger
from btlite.models.holdings import (
    HOLDINGS_SCHEMA_VERSION,
    Holding,
    HoldingsFile,
    Portfolio,
    Watchlist,
    WatchlistEntry,
)

if TYPE_CHECKING:
    from datetime import date

    from btlite.models.common import Currency

_LOG = get_logger(__name__)

#: Einrueckung der geschriebenen JSON-Datei. Lesbarkeit geht hier vor Groesse.
_JSON_INDENT = 2


def _default_path() -> Path:
    """Liefert den konfigurierten Pfad der Speicherdatei.

    Returns:
        Pfad aus ``Settings.portfolio.holdings_file``.
    """
    from btlite.config import get_settings

    return get_settings().portfolio.holdings_file


class HoldingsStore:
    """Laedt und speichert Depots und Watchlists in einer JSON-Datei.

    Der Store haelt keinen Zustand zwischen den Aufrufen: jede Operation
    liest die Datei, aendert und schreibt sie atomar zurueck. Bei einem
    lokalen Einzelplatzwerkzeug ist das schnell genug und schliesst die
    Klasse von Fehlern aus, bei der ein zwischengespeicherter Stand eine
    fremde Aenderung ueberschreibt.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        """Initialisiert den Store.

        Args:
            path: Pfad der JSON-Datei. None nimmt den Wert aus den Settings.
        """
        self.path = Path(path) if path is not None else _default_path()

    # ------------------------------------------------------------ Lesen ----

    def load(self) -> HoldingsFile:
        """Laedt den gespeicherten Bestand.

        Eine fehlende Datei ist **kein Fehler**: beim ersten Start gibt es
        noch nichts zu laden.

        Returns:
            Der gespeicherte Bestand, oder ein leerer :class:`HoldingsFile`
            wenn die Datei nicht existiert.

        Raises:
            PortfolioError: Wenn die Datei existiert, aber nicht lesbar ist,
                kein gueltiges JSON enthaelt, nicht auf den Vertrag passt
                oder eine unbekannte Schema-Version traegt. In keinem dieser
                Faelle wird die Datei angefasst.
        """
        if not self.path.exists():
            return HoldingsFile()

        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            msg = f"Portfolio file {self.path.name} cannot be read"
            raise PortfolioError(msg, detail=f"{type(exc).__name__}: {exc}") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = (
                f"Portfolio file {self.path.name} is corrupt and was not "
                "changed. Please check the file or restore it from a backup."
            )
            raise PortfolioError(msg, detail=f"{self.path}: {exc}") from exc

        if not isinstance(payload, dict):
            msg = f"Portfolio file {self.path.name} has a foreign format"
            raise PortfolioError(msg, detail=f"typ={type(payload).__name__}")

        self._check_schema_version(payload)

        try:
            return HoldingsFile.model_validate(payload)
        except ValidationError as exc:
            msg = (
                f"Portfolio file {self.path.name} does not match the expected "
                "format and was not changed."
            )
            raise PortfolioError(msg, detail=str(exc)) from exc

    def _check_schema_version(self, payload: dict[str, object]) -> None:
        """Prueft die Schema-Version der geladenen Datei.

        Args:
            payload: Rohes JSON-Objekt der Datei.

        Raises:
            PortfolioError: Wenn die Version fehlt, kein Integer ist oder aus
                der Zukunft stammt. Eine neuere Version still zu lesen waere
                Raten; die unbekannten Felder wuerden von ``extra="forbid"``
                ohnehin verworfen.
        """
        version = payload.get("schema_version")
        if not isinstance(version, int) or isinstance(version, bool):
            msg = f"Portfolio file {self.path.name} carries no valid schema version"
            raise PortfolioError(msg, detail=f"schema_version={version!r}")
        if version > HOLDINGS_SCHEMA_VERSION:
            msg = (
                f"Portfolio file {self.path.name} was written by a newer version "
                "of BT-Lite and cannot be read here."
            )
            raise PortfolioError(
                msg, detail=f"datei={version}, unterstuetzt={HOLDINGS_SCHEMA_VERSION}"
            )

    # --------------------------------------------------------- Schreiben ----

    def save(self, data: HoldingsFile) -> None:
        """Schreibt den Bestand atomar in die Datei.

        Ablauf: temporaere Datei im Zielverzeichnis anlegen, vollstaendig
        schreiben, auf die Platte zwingen (``fsync``), dann per ``os.replace``
        umbenennen. Bricht irgendetwas davor ab, bleibt die alte Datei
        vollstaendig erhalten.

        Args:
            data: Zu speichernder Bestand. ``updated_at`` und
                ``schema_version`` werden dabei gesetzt.

        Raises:
            PortfolioError: Wenn das Verzeichnis oder die Datei nicht
                beschreibbar ist.
        """
        to_write = data.model_copy(
            update={"schema_version": HOLDINGS_SCHEMA_VERSION, "updated_at": utc_now()}
        )
        text = to_write.model_dump_json(indent=_JSON_INDENT)

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            msg = f"The directory for {self.path.name} cannot be created"
            raise PortfolioError(msg, detail=f"{self.path.parent}: {exc}") from exc

        tmp_path: Path | None = None
        try:
            # delete=False, weil die Datei den with-Block ueberleben muss:
            # sie wird danach ueber den Zielpfad umbenannt.
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                tmp_path = Path(handle.name)
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            # Path.replace ist os.replace: auf einem Dateisystem atomar und
            # ueberschreibt ein vorhandenes Ziel auch unter Windows.
            tmp_path.replace(self.path)
        except OSError as exc:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            msg = (
                f"Portfolio file {self.path.name} could not be written. "
                "The previous state is unchanged."
            )
            raise PortfolioError(msg, detail=f"{self.path}: {exc}") from exc

        _LOG.debug(
            "holdings gespeichert",
            extra={"path": str(self.path), "portfolios": len(to_write.portfolios)},
        )

    # ---------------------------------------------------------- Portfolio ----

    def list_portfolios(self) -> list[str]:
        """Nennt die gespeicherten Depots.

        Returns:
            Namen aller Depots, alphabetisch sortiert.
        """
        return sorted(self.load().portfolios)

    def get_portfolio(self, name: str) -> Portfolio | None:
        """Liest ein einzelnes Depot.

        Args:
            name: Anzeigename des Depots.

        Returns:
            Das Depot, oder None wenn es nicht existiert.
        """
        return self.load().portfolios.get(name)

    def create_portfolio(
        self,
        name: str,
        *,
        currency: Currency | None = None,
        note: str | None = None,
    ) -> Portfolio:
        """Legt ein leeres Depot an.

        Args:
            name: Anzeigename, muss eindeutig sein.
            currency: Berichtswaehrung. None nimmt den Modell-Default (USD).
            note: Optionale Notiz.

        Returns:
            Das angelegte Depot.

        Raises:
            PortfolioError: Wenn bereits ein Depot dieses Namens existiert.
        """
        data = self.load()
        if name in data.portfolios:
            msg = f"A portfolio named {name!r} already exists"
            raise PortfolioError(msg, detail=f"vorhanden={sorted(data.portfolios)}")

        now = utc_now()
        fields: dict[str, object] = {"name": name, "created_at": now, "updated_at": now}
        if currency is not None:
            fields["currency"] = currency
        if note is not None:
            fields["note"] = note
        portfolio = Portfolio.model_validate(fields)

        data.portfolios[name] = portfolio
        self.save(data)
        return portfolio

    def delete_portfolio(self, name: str) -> None:
        """Entfernt ein Depot samt seiner Haltungen.

        Args:
            name: Anzeigename des Depots.

        Raises:
            PortfolioError: Wenn kein Depot dieses Namens existiert.
        """
        data = self.load()
        if name not in data.portfolios:
            msg = f"No portfolio named {name!r}"
            raise PortfolioError(msg, detail=f"vorhanden={sorted(data.portfolios)}")
        del data.portfolios[name]
        self.save(data)

    # ----------------------------------------------------------- Holding ----

    def add_holding(
        self,
        portfolio_name: str,
        ticker: str,
        *,
        shares: float,
        buy_price: float,
        buy_date: date,
        currency: Currency | None = None,
        note: str | None = None,
    ) -> Portfolio:
        """Fuegt einem Depot eine Position hinzu.

        Ein bereits gehaltener Ticker wird **nicht** zusammengefasst: ein
        zweiter Kauf zu einem anderen Kurs ist eine eigene Haltung, sonst
        geht der Einstandskurs des ersten verloren.

        Args:
            portfolio_name: Zieldepot.
            ticker: Symbol, wird normalisiert.
            shares: Stueckzahl, gebrochen zulaessig, muss positiv sein.
            buy_price: Kaufkurs je Stueck, muss positiv sein.
            buy_date: Kaufdatum.
            currency: Waehrung des Kaufkurses. None nimmt die Waehrung des
                Depots.
            note: Optionale Notiz.

        Returns:
            Das aktualisierte Depot.

        Raises:
            PortfolioError: Wenn das Depot fehlt oder die Werte unzulaessig
                sind (etwa negative Stueckzahl).
        """
        data = self.load()
        portfolio = self._require_portfolio(data, portfolio_name)

        fields: dict[str, object] = {
            "ticker": ticker,
            "shares": shares,
            "buy_price": buy_price,
            "buy_date": buy_date,
            "currency": currency if currency is not None else portfolio.currency,
        }
        if note is not None:
            fields["note"] = note
        holding = self._build_holding(fields)

        portfolio.holdings.append(holding)
        self._touch(portfolio)
        self.save(data)
        return portfolio

    def remove_holding(self, portfolio_name: str, index: int) -> Portfolio:
        """Entfernt eine Position ueber ihre Position in der Liste.

        Der Index adressiert die Haltung eindeutig, auch wenn ein Ticker
        mehrfach vorkommt. Die Reihenfolge ist die Eingabereihenfolge.

        Args:
            portfolio_name: Zieldepot.
            index: Nullbasierter Index der Haltung.

        Returns:
            Das aktualisierte Depot.

        Raises:
            PortfolioError: Wenn das Depot fehlt oder der Index ausserhalb
                des gueltigen Bereichs liegt.
        """
        data = self.load()
        portfolio = self._require_portfolio(data, portfolio_name)
        if not 0 <= index < len(portfolio.holdings):
            msg = f"Position {index} does not exist in portfolio {portfolio_name!r}"
            raise PortfolioError(msg, detail=f"n_holdings={len(portfolio.holdings)}")

        del portfolio.holdings[index]
        self._touch(portfolio)
        self.save(data)
        return portfolio

    def update_holding(
        self,
        portfolio_name: str,
        index: int,
        *,
        shares: float | None = None,
        buy_price: float | None = None,
        buy_date: date | None = None,
        currency: Currency | None = None,
        note: str | None = None,
    ) -> Portfolio:
        """Aendert Felder einer bestehenden Position.

        Nicht uebergebene Felder bleiben unveraendert. Der Ticker ist nicht
        aenderbar - eine Position auf ein anderes Symbol umzuschreiben waere
        ein anderer Kauf, keine Korrektur.

        Die Waehrung dagegen ist aenderbar, obwohl sie den Einstand
        umdeutet: sie ist eine Angabe **ueber** den Kauf, keine zweite
        Identitaet. Wer beim Erfassen USD stehen liess, obwohl er in Euro
        gekauft hat, korrigiert einen Tippfehler - ihn dafuer die Position
        loeschen und neu anlegen zu lassen, waere Schikane.

        Args:
            portfolio_name: Zieldepot.
            index: Nullbasierter Index der Haltung.
            shares: Neue Stueckzahl, muss positiv sein.
            buy_price: Neuer Kaufkurs, muss positiv sein.
            buy_date: Neues Kaufdatum.
            currency: Neue Waehrung des Kaufkurses.
            note: Neue Notiz.

        Returns:
            Das aktualisierte Depot.

        Raises:
            PortfolioError: Wenn Depot oder Index fehlen oder die neuen Werte
                unzulaessig sind.
        """
        data = self.load()
        portfolio = self._require_portfolio(data, portfolio_name)
        if not 0 <= index < len(portfolio.holdings):
            msg = f"Position {index} does not exist in portfolio {portfolio_name!r}"
            raise PortfolioError(msg, detail=f"n_holdings={len(portfolio.holdings)}")

        current = portfolio.holdings[index]
        changes: dict[str, object] = {}
        if shares is not None:
            changes["shares"] = shares
        if buy_price is not None:
            changes["buy_price"] = buy_price
        if buy_date is not None:
            changes["buy_date"] = buy_date
        if currency is not None:
            changes["currency"] = currency
        if note is not None:
            changes["note"] = note

        merged = current.model_dump()
        merged.update(changes)
        portfolio.holdings[index] = self._build_holding(merged)
        self._touch(portfolio)
        self.save(data)
        return portfolio

    # --------------------------------------------------------- Watchlist ----

    def list_watchlists(self) -> list[str]:
        """Nennt die gespeicherten Watchlists.

        Returns:
            Namen aller Watchlists, alphabetisch sortiert.
        """
        return sorted(self.load().watchlists)

    def get_watchlist(self, name: str) -> Watchlist | None:
        """Liest eine einzelne Watchlist.

        Args:
            name: Anzeigename der Liste.

        Returns:
            Die Watchlist, oder None wenn sie nicht existiert.
        """
        return self.load().watchlists.get(name)

    def create_watchlist(self, name: str) -> Watchlist:
        """Legt eine leere Watchlist an.

        Args:
            name: Anzeigename, muss eindeutig sein.

        Returns:
            Die angelegte Watchlist.

        Raises:
            PortfolioError: Wenn bereits eine Liste dieses Namens existiert.
        """
        data = self.load()
        if name in data.watchlists:
            msg = f"A watchlist named {name!r} already exists"
            raise PortfolioError(msg, detail=f"vorhanden={sorted(data.watchlists)}")

        now = utc_now()
        watchlist = Watchlist(name=name, created_at=now, updated_at=now)
        data.watchlists[name] = watchlist
        self.save(data)
        return watchlist

    def delete_watchlist(self, name: str) -> None:
        """Entfernt eine Watchlist.

        Args:
            name: Anzeigename der Liste.

        Raises:
            PortfolioError: Wenn keine Liste dieses Namens existiert.
        """
        data = self.load()
        if name not in data.watchlists:
            msg = f"No watchlist named {name!r}"
            raise PortfolioError(msg, detail=f"vorhanden={sorted(data.watchlists)}")
        del data.watchlists[name]
        self.save(data)

    def add_to_watchlist(
        self, watchlist_name: str, ticker: str, *, note: str | None = None
    ) -> Watchlist:
        """Nimmt ein Symbol in eine Watchlist auf.

        Args:
            watchlist_name: Zielliste.
            ticker: Symbol, wird normalisiert.
            note: Optionale Notiz.

        Returns:
            Die aktualisierte Watchlist.

        Raises:
            PortfolioError: Wenn die Liste fehlt, das Symbol bereits
                enthalten oder unzulaessig ist.
        """
        from btlite.models.common import normalize_ticker

        data = self.load()
        watchlist = self._require_watchlist(data, watchlist_name)

        try:
            key = normalize_ticker(ticker)
        except ValueError as exc:
            msg = "The symbol is invalid"
            raise PortfolioError(msg, detail=str(exc)) from exc

        if key in watchlist.tickers:
            msg = f"{key} is already on watchlist {watchlist_name!r}"
            raise PortfolioError(msg, detail=f"eintraege={watchlist.tickers}")

        fields: dict[str, object] = {"ticker": key, "added_at": utc_now()}
        if note is not None:
            fields["note"] = note
        try:
            entry = WatchlistEntry.model_validate(fields)
        except ValidationError as exc:
            msg = "The watchlist entry is invalid"
            raise PortfolioError(msg, detail=str(exc)) from exc

        watchlist.entries.append(entry)
        self._touch(watchlist)
        self.save(data)
        return watchlist

    def remove_from_watchlist(self, watchlist_name: str, ticker: str) -> Watchlist:
        """Entfernt ein Symbol aus einer Watchlist.

        Args:
            watchlist_name: Zielliste.
            ticker: Symbol, wird normalisiert.

        Returns:
            Die aktualisierte Watchlist.

        Raises:
            PortfolioError: Wenn die Liste fehlt oder das Symbol nicht
                enthalten ist.
        """
        from btlite.models.common import normalize_ticker

        data = self.load()
        watchlist = self._require_watchlist(data, watchlist_name)

        try:
            key = normalize_ticker(ticker)
        except ValueError as exc:
            msg = "The symbol is invalid"
            raise PortfolioError(msg, detail=str(exc)) from exc

        remaining = [e for e in watchlist.entries if e.ticker != key]
        if len(remaining) == len(watchlist.entries):
            msg = f"{key} is not on watchlist {watchlist_name!r}"
            raise PortfolioError(msg, detail=f"eintraege={watchlist.tickers}")

        watchlist.entries = remaining
        self._touch(watchlist)
        self.save(data)
        return watchlist

    def set_watchlist_note(self, watchlist_name: str, ticker: str, note: str | None) -> Watchlist:
        """Setzt oder loescht die Notiz eines beobachteten Titels.

        Getrennt von :meth:`add_to_watchlist`, weil das Hinzufuegen eines
        bereits enthaltenen Symbols bewusst ein Fehler ist. Eine Notiz zu
        aendern ist kein erneutes Hinzufuegen.

        Args:
            watchlist_name: Zielliste.
            ticker: Symbol, wird normalisiert.
            note: Neuer Text. ``None`` oder ein leerer String loescht die
                Notiz - ein leeres Eingabefeld bedeutet "keine Notiz", nicht
                "Notiz aus Leerzeichen".

        Returns:
            Die aktualisierte Watchlist.

        Raises:
            PortfolioError: Wenn die Liste fehlt, das Symbol nicht enthalten
                oder die Notiz unzulaessig ist (etwa zu lang).
        """
        from btlite.models.common import normalize_ticker

        data = self.load()
        watchlist = self._require_watchlist(data, watchlist_name)

        try:
            key = normalize_ticker(ticker)
        except ValueError as exc:
            msg = "The symbol is invalid"
            raise PortfolioError(msg, detail=str(exc)) from exc

        for position, entry in enumerate(watchlist.entries):
            if entry.ticker != key:
                continue
            cleaned = note.strip() if note is not None else None
            fields = entry.model_dump()
            fields["note"] = cleaned or None
            try:
                watchlist.entries[position] = WatchlistEntry.model_validate(fields)
            except ValidationError as exc:
                msg = "The watchlist entry is invalid"
                raise PortfolioError(msg, detail=str(exc)) from exc
            self._touch(watchlist)
            self.save(data)
            return watchlist

        msg = f"{key} is not on watchlist {watchlist_name!r}"
        raise PortfolioError(msg, detail=f"eintraege={watchlist.tickers}")

    # ------------------------------------------------------------ Helfer ----

    def _require_portfolio(self, data: HoldingsFile, name: str) -> Portfolio:
        """Holt ein Depot oder wirft.

        Args:
            data: Geladener Bestand.
            name: Anzeigename des Depots.

        Returns:
            Das gefundene Depot.

        Raises:
            PortfolioError: Wenn kein Depot dieses Namens existiert.
        """
        portfolio = data.portfolios.get(name)
        if portfolio is None:
            msg = f"No portfolio named {name!r}"
            raise PortfolioError(msg, detail=f"vorhanden={sorted(data.portfolios)}")
        return portfolio

    def _require_watchlist(self, data: HoldingsFile, name: str) -> Watchlist:
        """Holt eine Watchlist oder wirft.

        Args:
            data: Geladener Bestand.
            name: Anzeigename der Liste.

        Returns:
            Die gefundene Watchlist.

        Raises:
            PortfolioError: Wenn keine Liste dieses Namens existiert.
        """
        watchlist = data.watchlists.get(name)
        if watchlist is None:
            msg = f"No watchlist named {name!r}"
            raise PortfolioError(msg, detail=f"vorhanden={sorted(data.watchlists)}")
        return watchlist

    @staticmethod
    def _build_holding(fields: dict[str, object]) -> Holding:
        """Validiert Rohfelder zu einer Haltung.

        Uebersetzt den Pydantic-Fehler in einen Projekttyp, damit das UI nur
        :class:`~btlite.core.errors.BTLiteError` faengt.

        Args:
            fields: Feldwerte der Haltung.

        Returns:
            Die validierte Haltung.

        Raises:
            PortfolioError: Wenn die Werte den Vertrag verletzen, etwa bei
                negativer Stueckzahl oder nicht-positivem Kaufkurs.
        """
        try:
            return Holding.model_validate(fields)
        except ValidationError as exc:
            msg = (
                "The position details are invalid. Quantity and purchase price "
                "must be greater than zero."
            )
            raise PortfolioError(msg, detail=str(exc)) from exc

    @staticmethod
    def _touch(obj: Portfolio | Watchlist) -> None:
        """Setzt den Aenderungszeitstempel auf jetzt.

        Args:
            obj: Zu markierendes Depot oder Watchlist.
        """
        obj.updated_at = utc_now()
