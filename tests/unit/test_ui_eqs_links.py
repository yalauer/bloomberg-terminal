"""Tests der Links aus der Trefferliste der Seite EQS.

Gewuenscht: aus einem Screening-Treffer direkt zu Chart (GP), Steckbrief
(DES) und Finanzanalyse (FA) des Titels. Entschieden: die Tickerzelle
traegt den Ticker plus drei ``dcc.Link``, gebaut ueber
``search_bar.analysis_href``. ``components.data_table`` bekommt dafuer den
optionalen Hook ``index_cell``; alle anderen Aufrufer bleiben unveraendert,
der CSV-Export ebenso.

Die Offline-Daten sind bekannt: ``dividend`` trifft PG, XOM und JNJ.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest
from dash import dcc, html

from btlite.config import Settings
from btlite.ui.app import create_app, parse_href
from btlite.ui.layouts import components as c
from btlite.ui.pages import eqs
from btlite.ui.state import AppState

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

UNIVERSE = ["AAPL", "MSFT", "NVDA", "SPY", "JNJ", "XOM", "KO", "PG", "JPM", "WMT"]


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppState:
    monkeypatch.setenv("BTLITE_PORTFOLIO_HOLDINGS_FILE", str(tmp_path / "holdings.json"))
    return AppState(Settings())


def _walk(node: Any) -> Iterator[Any]:
    """Durchlaeuft einen Dash-Komponentenbaum in Tiefe."""
    yield node
    children = getattr(node, "children", None)
    if isinstance(children, list | tuple):
        for child in children:
            yield from _walk(child)
    elif children is not None and not isinstance(children, str | int | float):
        yield from _walk(children)


def _links(node: Any) -> list[dcc.Link]:
    return [n for n in _walk(node) if isinstance(n, dcc.Link)]


def _result_links(node: Any) -> dict[str, set[str]]:
    """Ticker je Mnemonic aus allen Treffer-Links einer Seite."""
    found: dict[str, set[str]] = {}
    for link in _links(node):
        if link.className != "bt-eqs-link":
            continue
        mnemonic, tickers, _ = parse_href(*_split(str(link.href)))
        assert len(tickers) == 1
        found.setdefault(mnemonic, set()).add(tickers[0])
    return found


def _split(href: str) -> tuple[str, str]:
    path, _, query = href.partition("?")
    return path, f"?{query}"


def _first_cells(table: html.Table) -> list[html.Td]:
    body = next(n for n in _walk(table) if isinstance(n, html.Tbody))
    return [row.children[0] for row in body.children]


class TestTickerCell:
    def test_links_to_gp_des_and_fa_with_the_ticker(self) -> None:
        cell = eqs._ticker_cell("AAPL")
        hrefs = [str(link.href) for link in _links(html.Div(cell))]
        assert hrefs == ["/gp?t=AAPL", "/des?t=AAPL", "/fa?t=AAPL"]

    def test_ticker_stays_visible_as_text(self) -> None:
        cell = eqs._ticker_cell("PG")
        assert isinstance(cell[0], html.Span)
        assert cell[0].children == "PG"

    def test_every_link_has_text_and_tooltip(self) -> None:
        links = _links(html.Div(eqs._ticker_cell("XOM")))
        assert [link.children for link in links] == ["Price", "Profile", "Financials"]
        for link in links:
            assert "XOM" in str(link.title)
            assert str(link.title).startswith("Open the ")

    def test_uses_the_shared_address_builder(self) -> None:
        from btlite.ui.search_bar import analysis_href

        hrefs = [str(link.href) for link in _links(html.Div(eqs._ticker_cell("NYSE:NOW")))]
        assert hrefs == [analysis_href(m, "NYSE:NOW") for m in ("GP", "DES", "FA")]


class TestRenderedResultList:
    def test_fundamental_preset_links_every_match(self, state: AppState) -> None:
        page = eqs.render_result(state, "dividend", UNIVERSE)
        assert _result_links(page) == {
            "GP": {"PG", "XOM", "JNJ"},
            "DES": {"PG", "XOM", "JNJ"},
            "FA": {"PG", "XOM", "JNJ"},
        }

    def test_technical_preset_links_every_match(self, state: AppState) -> None:
        result = eqs.run_screen(state, "above_200ma", UNIVERSE)
        matches = {str(t) for t in result.data.index}
        assert matches, "Offline-Daten sollten Treffer liefern"
        page = eqs.render_result(state, "above_200ma", UNIVERSE)
        assert _result_links(page) == {"GP": matches, "DES": matches, "FA": matches}


class TestDataTableHook:
    FRAME = pd.DataFrame({"x": [1.0, 2.0]}, index=["AAPL", "MSFT"])

    def test_default_index_cell_is_plain_text(self) -> None:
        table = c.data_table(self.FRAME, index_label="Ticker")
        assert [td.children for td in _first_cells(table)] == ["AAPL", "MSFT"]
        assert _links(table) == []

    def test_hook_renders_the_index_cell(self) -> None:
        table = c.data_table(
            self.FRAME, index_label="Ticker", index_cell=lambda t: html.B(t.lower())
        )
        assert [td.children.children for td in _first_cells(table)] == ["aapl", "msft"]

    def test_highlight_still_uses_the_raw_value(self) -> None:
        table = c.data_table(
            self.FRAME,
            index_label="Ticker",
            highlight_rows=["MSFT"],
            index_cell=eqs._ticker_cell,
        )
        body = next(n for n in _walk(table) if isinstance(n, html.Tbody))
        assert [row.className for row in body.children] == ["", "bt-subject"]

    def test_sorting_the_index_uses_raw_tickers(self) -> None:
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=["MSFT", "AAPL", "KO"])
        ordered = c.sort_frame(frame, by="Ticker", index_label="Ticker")
        table = c.data_table(ordered, index_label="Ticker", index_cell=eqs._ticker_cell)
        assert [td.children[0].children for td in _first_cells(table)] == ["AAPL", "KO", "MSFT"]


class TestExportUnchanged:
    def test_csv_index_holds_plain_tickers(self, state: AppState) -> None:
        app = create_app(state.settings, state)
        export = None
        for key, entry in app.callback_map.items():
            if eqs.EXPORT_BUTTON in str(entry.get("inputs", "")) and eqs.DOWNLOAD in key:
                callback = entry["callback"]
                export = getattr(callback, "__wrapped__", callback)
        assert export is not None
        payload = export(1, "dividend", UNIVERSE)
        lines = payload["content"].splitlines()
        tickers = {line.split(",", 1)[0] for line in lines[1:]}
        assert tickers == {"PG", "XOM", "JNJ"}
        assert "/gp" not in payload["content"]
        assert "Profile" not in payload["content"]


class TestLinkContrast:
    def test_cyan_link_readable_on_row_hover_surface(self) -> None:
        from tests.unit.test_ui_styling import AA_NORMAL, contrast_ratio

        assert contrast_ratio("#22d3ee", "#1c1c18") >= AA_NORMAL

    def test_link_css_has_resting_hover_and_focus_rules(self) -> None:
        from tests.unit.test_ui_styling import CSS_PATH

        css = CSS_PATH.read_text(encoding="utf-8")
        for selector in (
            ".bt-eqs-link {",
            ".bt-eqs-link:hover {",
            ".bt-eqs-link:focus-visible {",
            ".bt-eqs-ticker {",
        ):
            assert selector in css
