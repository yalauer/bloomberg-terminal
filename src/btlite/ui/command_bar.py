"""Parser der Mnemonic-Kommandozeile.

Eingabeformat, wie in ``docs/ARCHITECTURE.md`` festgelegt::

    <TICKER[,TICKER...]> <MNEMONIC> [ARG...] [GO]

Beispiele: ``AAPL FA GO``, ``AAPL,MSFT,NVDA PORT``, ``EQS value GO``.

Das abschliessende ``GO`` ist optional - Enter loest ohnehin aus. Es bleibt
zulaessig, weil es der Bedienung eines Bloomberg-Terminals entspricht und
Nutzer es gewohnt sind zu tippen.

Die Funktionen hier sind rein: sie kennen weder Dash noch Provider und sind
ohne laufendes UI testbar.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Final

from btlite.models.common import normalize_ticker

#: Mnemonics und ihre Langform. Reihenfolge ist die der Hilfe-Anzeige.
MNEMONICS: Final[dict[str, str]] = {
    "DES": "Profile - company data, price and business",
    "GP": "Price history - chart with indicators",
    "FA": "Financial analysis - income statement, balance sheet, cash flow",
    "EQS": "Equity screening - filter titles by criteria",
    "BT": "Backtest - test a strategy on the price history",
    "PORT": "Portfolio and risk - weights and risk measures",
    "WL": "Watchlist and portfolio - saved titles and holdings",
}

#: Beschriftung der Navigationspunkte. Das Kuerzel bleibt daneben stehen,
#: weil es die Kommandozeile verlangt - aber es steht nicht mehr allein:
#: "DES" sagt niemandem, dass dahinter der Steckbrief liegt.
NAV_LABELS: Final[dict[str, str]] = {
    "DES": "Profile",
    "GP": "Price",
    "FA": "Financials",
    "EQS": "Screening",
    "BT": "Backtest",
    "PORT": "Portfolio",
    "WL": "Watchlist",
}

#: Mnemonics, die ohne Ticker auskommen. EQS screent ein Universum,
#: WL liest den gespeicherten Bestand statt einer Eingabe.
TICKERLESS_MNEMONICS: Final[frozenset[str]] = frozenset({"EQS", "WL"})

#: Optionales Abschlusswort der Eingabe.
_GO: Final[str] = "GO"

#: Trennzeichen innerhalb einer Ticker-Liste.
_TICKER_SEPARATORS: Final[str] = ",;"


@dataclass(frozen=True)
class Command:
    """Eine erfolgreich geparste Eingabe.

    Attributes:
        mnemonic: Kanonisches Mnemonic in Grossbuchstaben.
        tickers: Kanonische Symbole in Eingabereihenfolge, ohne Duplikate.
        args: Weitere Argumente hinter dem Mnemonic, ohne ``GO``.
    """

    mnemonic: str
    tickers: tuple[str, ...] = ()
    args: tuple[str, ...] = ()

    @property
    def path(self) -> str:
        """Route, auf die die Kommandozeile umleitet.

        Returns:
            Pfad wie ``"/fa"``.
        """
        return f"/{self.mnemonic.lower()}"


@dataclass(frozen=True)
class ParseError:
    """Eine nicht interpretierbare Eingabe.

    Attributes:
        user_message: Nicht-technische Meldung fuer das Fehlerbanner.
        suggestion: Naechstliegendes gueltiges Mnemonic, falls eines
            hinreichend aehnlich ist.
    """

    user_message: str
    suggestion: str | None = None


@dataclass
class CommandHistory:
    """Eingabehistorie fuer die Pfeiltasten.

    Der Zeiger steht nach jeder Aufnahme hinter dem juengsten Eintrag.
    ``previous`` laeuft von dort rueckwaerts, ``next`` wieder vorwaerts und
    liefert am Ende ``None``, damit die Zeile geleert werden kann.

    Attributes:
        entries: Eingaben, aelteste zuerst.
        cursor: Position des naechsten Rueckwaertsschritts.
        max_entries: Obergrenze der aufbewahrten Eingaben.
    """

    entries: list[str] = field(default_factory=list)
    cursor: int = 0
    max_entries: int = 50

    def add(self, raw: str) -> None:
        """Nimmt eine Eingabe auf und setzt den Zeiger zurueck.

        Aufeinanderfolgende Wiederholungen werden nicht doppelt abgelegt.

        Args:
            raw: Rohe Eingabezeile.
        """
        text = raw.strip()
        if not text:
            return
        if not self.entries or self.entries[-1] != text:
            self.entries.append(text)
            del self.entries[: max(0, len(self.entries) - self.max_entries)]
        self.cursor = len(self.entries)

    def previous(self) -> str | None:
        """Liefert den vorherigen Eintrag.

        Returns:
            Aeltere Eingabe, oder None wenn keine vorliegt.
        """
        if not self.entries or self.cursor == 0:
            return None
        self.cursor -= 1
        return self.entries[self.cursor]

    def next(self) -> str | None:
        """Liefert den naechstjuengeren Eintrag.

        Returns:
            Juengere Eingabe, oder None am Ende der Historie.
        """
        if self.cursor >= len(self.entries) - 1:
            self.cursor = len(self.entries)
            return None
        self.cursor += 1
        return self.entries[self.cursor]


def split_tickers(raw: str) -> list[str]:
    """Zerlegt eine Ticker-Angabe in kanonische Symbole.

    Akzeptiert Komma und Semikolon als Trenner. Duplikate werden stabil
    entfernt.

    Args:
        raw: Rohe Ticker-Angabe, etwa ``"aapl,msft"``.

    Returns:
        Kanonische Symbole in Eingabereihenfolge.
    """
    normalized = raw
    for sep in _TICKER_SEPARATORS:
        normalized = normalized.replace(sep, " ")

    seen: set[str] = set()
    out: list[str] = []
    for part in normalized.split():
        try:
            symbol = normalize_ticker(part)
        except ValueError:
            continue
        if symbol not in seen:
            seen.add(symbol)
            out.append(symbol)
    return out


def _looks_like_mnemonic(token: str) -> bool:
    """Prueft, ob ein Token ein gueltiges Mnemonic ist.

    Args:
        token: Rohes Token.

    Returns:
        True wenn es exakt einem Mnemonic entspricht.
    """
    return token.strip().upper() in MNEMONICS


def suggest_mnemonic(token: str) -> str | None:
    """Sucht das naechstliegende gueltige Mnemonic.

    Args:
        token: Nicht erkanntes Token.

    Returns:
        Aehnlichstes Mnemonic, oder None wenn keines nahe genug liegt.
    """
    matches = difflib.get_close_matches(token.strip().upper(), list(MNEMONICS), n=1, cutoff=0.5)
    return matches[0] if matches else None


def parse_command(raw: str) -> Command | ParseError:
    """Parst eine Eingabezeile der Kommandozeile.

    Toleriert Kleinschreibung, beliebige Leerzeichen und ein optionales
    abschliessendes ``GO``. Ein Mnemonic ohne Ticker ist gueltig, solange es
    in :data:`TICKERLESS_MNEMONICS` steht oder die Seite mit einer
    Vorbelegung arbeiten kann - das entscheidet der Aufrufer.

    Args:
        raw: Rohe Eingabezeile, etwa ``"AAPL FA GO"``.

    Returns:
        :class:`Command` bei gueltiger Eingabe, sonst :class:`ParseError`
        mit Meldung und - wenn moeglich - einem Vorschlag.
    """
    tokens = raw.replace("\t", " ").split()
    if not tokens:
        return ParseError("Enter a command, for example AAPL FA GO.")

    # Ein abschliessendes GO ist Zierde und wird verworfen.
    if len(tokens) > 1 and tokens[-1].strip().upper() == _GO:
        tokens = tokens[:-1]
    if not tokens:
        return ParseError("GO alone is not a command. Expected: TICKER MNEMONIC GO.")

    # Das Mnemonic ist das erste Token, das eines ist. Alles davor sind
    # Ticker, alles danach sind Argumente.
    index = next((i for i, t in enumerate(tokens) if _looks_like_mnemonic(t)), None)
    if index is None:
        return _unknown_mnemonic(tokens)

    mnemonic = tokens[index].strip().upper()
    tickers = split_tickers(" ".join(tokens[:index]))
    args = tuple(t for t in tokens[index + 1 :] if t.strip())

    if not tickers and mnemonic not in TICKERLESS_MNEMONICS:
        return ParseError(
            f"{mnemonic} needs at least one ticker, for example AAPL {mnemonic} GO.",
            suggestion=None,
        )

    return Command(mnemonic=mnemonic, tickers=tuple(tickers), args=args)


def _unknown_mnemonic(tokens: list[str]) -> ParseError:
    """Baut die Fehlermeldung, wenn kein Token ein Mnemonic ist.

    Args:
        tokens: Bereinigte Tokenliste der Eingabe.

    Returns:
        Fehler mit Vorschlag, wenn ein Token einem Mnemonic aehnelt.
    """
    # Das letzte Token ist der wahrscheinlichste Tippfehler-Kandidat: die
    # Ticker stehen vorn, das Mnemonic hinten.
    candidate = tokens[-1]
    suggestion = suggest_mnemonic(candidate)
    known = ", ".join(MNEMONICS)
    if suggestion is not None:
        return ParseError(
            f"Unknown command {candidate.upper()!r}. Did you mean {suggestion}?",
            suggestion=suggestion,
        )
    return ParseError(
        f"Unknown command {candidate.upper()!r}. Known commands: {known}.",
        suggestion=None,
    )


def format_command(command: Command) -> str:
    """Setzt ein Kommando wieder als Eingabezeile.

    Args:
        command: Geparstes Kommando.

    Returns:
        Kanonische Eingabezeile, etwa ``"AAPL,MSFT GP GO"``.
    """
    parts: list[str] = []
    if command.tickers:
        parts.append(",".join(command.tickers))
    parts.append(command.mnemonic)
    parts.extend(command.args)
    parts.append(_GO)
    return " ".join(parts)
