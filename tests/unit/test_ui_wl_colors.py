"""Tests der Richtungsfarben in den Tabellen der Seite WL.

Positive Werte tragen ``bt-up``, negative ``bt-down``, exakt 0 bleibt
neutral. Der Pfeil bleibt in jedem Fall im Text - Farbe ist nie der einzige
Bedeutungstraeger. Fehlende Werte bleiben der reine String "n/a", damit
``data_table`` ihnen ``bt-na`` gibt.

Die Tabellen werden direkt aus Modellwerten gebaut, nicht ueber Fixture-
Kurse: nur so liegen Gewinn, Verlust, exakt 0 und ein fehlender Wert
gleichzeitig und kontrolliert vor.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from typing import Any

import pytest
from dash import html

from btlite.models.common import Currency
from btlite.models.holdings import Watchlist, WatchlistEntry
from btlite.portfolio.valuation import PortfolioValuation, PositionValue
from btlite.ui import formatting as fmt
from btlite.ui.pages import wl

NOW = dt.datetime(2026, 9, 1, tzinfo=dt.UTC)


def _position(ticker: str, gain_abs: float | None, gain_pct: float | None) -> PositionValue:
    """Baut eine Position mit vorgegebenem Ergebnis."""
    priced = gain_abs is not None
    return PositionValue(
        ticker=ticker,
        shares=10.0,
        cost_basis=1000.0,
        average_price=100.0,
        price=100.0 + (gain_abs or 0.0) / 10.0 if priced else None,
        market_value=1000.0 + gain_abs if gain_abs is not None else None,
        gain_abs=gain_abs,
        gain_pct=gain_pct,
        currency=Currency.USD,
    )


def _valuation(*positions: PositionValue) -> PortfolioValuation:
    return PortfolioValuation(name="Test", positions=list(positions), currency=Currency.USD)


#: Gewinn, Verlust, exakt 0 und ein fehlender Wert in einer Tabelle. Der
#: fehlende Wert neben bekannten macht die Spalte zu float64 (NaN-Pfad).
MIXED = _valuation(
    _position("UP", 250.0, 0.25),
    _position("DOWN", -120.0, -0.12),
    _position("FLAT", 0.0, 0.0),
    _position("MISS", None, None),
)


def _cells(table: html.Table, columns: tuple[str, ...]) -> dict[str, dict[str, html.Td]]:
    """Liest die Zellen einer Tabelle je Ticker und Spalte.

    Args:
        table: Von ``data_table`` gebaute Tabelle.
        columns: Spaltennamen ohne Indexspalte, in Tabellenreihenfolge.

    Returns:
        Abbildung Ticker auf Abbildung Spaltenname auf Zelle.
    """
    body = table.children[1]
    out: dict[str, dict[str, html.Td]] = {}
    for row in body.children:
        index_cell, *rest = row.children
        out[str(index_cell.children)] = dict(zip(columns, rest, strict=True))
    return out


def _order(table: html.Table) -> list[str]:
    """Liest die Ticker in Zeilenreihenfolge."""
    return [str(row.children[0].children) for row in table.children[1].children]


def _text(cell: html.Td) -> str:
    """Sichtbarer Text einer Zelle, ob String oder Span."""
    content: Any = cell.children
    return str(content.children if isinstance(content, html.Span) else content)


def _colour(cell: html.Td) -> str | None:
    """Richtungsklasse des Zellinhalts, None ohne Span."""
    content: Any = cell.children
    return content.className if isinstance(content, html.Span) else None


def _positions(sort_by: str | None = None, sort_desc: bool = False) -> html.Table:
    return wl._positions_table(MIXED, {}, sort_by=sort_by, sort_desc=sort_desc)


def _watch(sort_by: str | None = None, sort_desc: bool = False) -> html.Table:
    """Watchlist mit steigendem, fallendem, unveraendertem und kurslosem Titel."""
    tickers = ("UP", "DOWN", "FLAT", "MISS")
    watchlist = Watchlist(
        name="W",
        entries=[WatchlistEntry(ticker=t, added_at=NOW) for t in tickers],
        created_at=NOW,
        updated_at=NOW,
    )
    quotes = {
        "UP": SimpleNamespace(price=105.0, previous_close=100.0, currency=Currency.USD),
        "DOWN": SimpleNamespace(price=97.0, previous_close=100.0, currency=Currency.USD),
        "FLAT": SimpleNamespace(price=100.0, previous_close=100.0, currency=Currency.USD),
    }
    return wl._watchlist_table(watchlist, quotes, {}, sort_by=sort_by, sort_desc=sort_desc)


class TestPositionsTable:
    @pytest.mark.parametrize("column", ["P&L %", "P&L"])
    def test_gain_is_green(self, column: str) -> None:
        assert _colour(_cells(_positions(), wl.POSITION_COLUMNS)["UP"][column]) == "bt-up"

    @pytest.mark.parametrize("column", ["P&L %", "P&L"])
    def test_loss_is_red(self, column: str) -> None:
        assert _colour(_cells(_positions(), wl.POSITION_COLUMNS)["DOWN"][column]) == "bt-down"

    @pytest.mark.parametrize("column", ["P&L %", "P&L"])
    def test_zero_is_neutral(self, column: str) -> None:
        cell = _cells(_positions(), wl.POSITION_COLUMNS)["FLAT"][column]
        assert _colour(cell) is None
        assert isinstance(cell.children, str)
        assert cell.children.startswith("▬")

    @pytest.mark.parametrize("column", ["P&L %", "P&L"])
    def test_missing_stays_plain_na(self, column: str) -> None:
        """Reiner String, damit data_table die Klasse bt-na setzt."""
        cell = _cells(_positions(), wl.POSITION_COLUMNS)["MISS"][column]
        assert cell.children == fmt.NA_TEXT
        assert cell.className == "bt-na"

    def test_missing_without_any_priced_row(self) -> None:
        """None-Pfad: die ganze Spalte fehlt, pandas behaelt None."""
        table = wl._positions_table(_valuation(_position("MISS", None, None)), {})
        cell = _cells(table, wl.POSITION_COLUMNS)["MISS"]["P&L %"]
        assert cell.children == fmt.NA_TEXT
        assert cell.className == "bt-na"

    def test_arrow_and_text_stay(self) -> None:
        """Farbe traegt die Bedeutung nie allein."""
        cells = _cells(_positions(), wl.POSITION_COLUMNS)
        assert _text(cells["UP"]["P&L %"]) == "▲ " + fmt.format_percent(
            0.25, decimals=2, signed=True
        )
        assert _text(cells["DOWN"]["P&L %"]).startswith("▼ ")
        assert _text(cells["DOWN"]["P&L"]).startswith("▼ ")

    def test_coloured_cells_are_not_marked_missing(self) -> None:
        cells = _cells(_positions(), wl.POSITION_COLUMNS)
        assert cells["UP"]["P&L %"].className == ""
        assert cells["DOWN"]["P&L"].className == ""

    def test_unsigned_columns_stay_uncoloured(self) -> None:
        cells = _cells(_positions(), wl.POSITION_COLUMNS)
        for column in ("Shares", "Buy price", "Price", "Cost basis", "Market value"):
            assert _colour(cells["UP"][column]) is None, column

    @pytest.mark.parametrize("column", ["P&L %", "P&L"])
    def test_sorting_follows_raw_values(self, column: str) -> None:
        assert _order(_positions(column)) == ["DOWN", "FLAT", "UP", "MISS"]
        assert _order(_positions(column, sort_desc=True)) == ["UP", "FLAT", "DOWN", "MISS"]


class TestWatchlistTable:
    def test_rising_is_green(self) -> None:
        assert _colour(_cells(_watch(), wl.WATCHLIST_COLUMNS)["UP"]["Change"]) == "bt-up"

    def test_falling_is_red(self) -> None:
        assert _colour(_cells(_watch(), wl.WATCHLIST_COLUMNS)["DOWN"]["Change"]) == "bt-down"

    def test_unchanged_is_neutral(self) -> None:
        cell = _cells(_watch(), wl.WATCHLIST_COLUMNS)["FLAT"]["Change"]
        assert _colour(cell) is None
        assert str(cell.children).startswith("▬")

    def test_missing_stays_plain_na(self) -> None:
        cell = _cells(_watch(), wl.WATCHLIST_COLUMNS)["MISS"]["Change"]
        assert cell.children == fmt.NA_TEXT
        assert cell.className == "bt-na"

    def test_arrow_stays(self) -> None:
        cells = _cells(_watch(), wl.WATCHLIST_COLUMNS)
        assert _text(cells["UP"]["Change"]).startswith("▲ +")
        assert _text(cells["DOWN"]["Change"]).startswith("▼ ")

    def test_price_stays_uncoloured(self) -> None:
        assert _colour(_cells(_watch(), wl.WATCHLIST_COLUMNS)["UP"]["Price"]) is None

    def test_sorting_follows_raw_values(self) -> None:
        assert _order(_watch("Change")) == ["DOWN", "FLAT", "UP", "MISS"]
        assert _order(_watch("Change", sort_desc=True)) == ["UP", "FLAT", "DOWN", "MISS"]
