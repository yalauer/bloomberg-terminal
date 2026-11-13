"""Die sechs Seiten des Terminals, indiziert nach Mnemonic.

Jede Seite stellt genau eine Funktion bereit::

    layout(state: AppState, tickers: list[str], args: tuple[str, ...]) -> html.Div

Sie faengt alle ``BTLiteError`` selbst und liefert im Fehlerfall ein
Fehlerbanner statt einer leeren Seite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol

from btlite.ui.pages import bt, des, eqs, fa, gp, port, wl

if TYPE_CHECKING:
    from dash import html

    from btlite.ui.state import AppState


class PageLayout(Protocol):
    """Signatur einer Seitenfunktion."""

    def __call__(
        self,
        state: AppState,
        tickers: list[str],
        args: tuple[str, ...] = (),
    ) -> html.Div:
        """Baut den Seiteninhalt.

        Args:
            state: UI-Zustand mit Provider und Puffern.
            tickers: Symbole aus der Kommandozeile.
            args: Weitere Argumente hinter dem Mnemonic.

        Returns:
            Seiteninhalt.
        """
        ...


#: Abbildung Mnemonic auf Seitenfunktion.
PAGES: Final[dict[str, PageLayout]] = {
    "DES": des.layout,
    "GP": gp.layout,
    "FA": fa.layout,
    "EQS": eqs.layout,
    "BT": bt.layout,
    "PORT": port.layout,
    "WL": wl.layout,
}

#: Abbildung Route auf Mnemonic, etwa ``"/fa"`` auf ``"FA"``.
ROUTES: Final[dict[str, str]] = {f"/{key.lower()}": key for key in PAGES}

__all__ = ["PAGES", "ROUTES", "PageLayout", "bt", "des", "eqs", "fa", "gp", "port", "wl"]
