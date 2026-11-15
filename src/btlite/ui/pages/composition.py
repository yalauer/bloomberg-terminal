"""Portfolio composition controls and look-through presentation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dash import dcc, html

from btlite.core.errors import BTLiteError, RateLimitError
from btlite.data.composition_csv import read_upload
from btlite.models.common import AssetClass
from btlite.models.composition import FundComposition
from btlite.models.market import Instrument
from btlite.portfolio.composition import UNKNOWN, exposures
from btlite.portfolio.valuation import value_portfolio
from btlite.ui.layouts import components as c
from btlite.viz.composition import composition_bar

if TYPE_CHECKING:
    from btlite.ui.state import AppState


def controls(state: AppState) -> html.Div:
    """Offer all saved portfolios, including single-ETF portfolios."""
    try:
        names = state.holdings.list_portfolios()
    except BTLiteError:
        names = []
    return c.section(
        "Portfolio-Zusammensetzung · ETF Look-through",
        html.P("Tatsächliche Depotgewichte nach Marktwert. Depots unter WL anlegen."),
        c.controls(
            c.labeled(
                "Depot",
                dcc.Dropdown(
                    id="composition-portfolio",
                    options=names,
                    placeholder="Depot auswählen",
                    className="bt-dropdown",
                ),
            ),
            c.labeled(
                "Vergleichsdepot (optional)",
                dcc.Dropdown(id="composition-compare", options=names, className="bt-dropdown"),
            ),
            c.labeled(
                "Aufschlüsselung",
                dcc.Dropdown(
                    id="composition-dimension",
                    value="company",
                    clearable=False,
                    options=[
                        {"label": label, "value": key}
                        for key, label in [
                            ("company", "Unternehmen"),
                            ("country", "Länder"),
                            ("region", "Regionen"),
                            ("sector", "Branchen / Sektoren"),
                        ]
                    ],
                    className="bt-dropdown",
                ),
            ),
            html.Button("Zusammensetzung analysieren", id="composition-run", n_clicks=0),
        ),
        html.Details(
            [
                html.Summary("Vollständige ETF-Aktienbestände als CSV ergänzen"),
                html.P(
                    "Optional: ersetzt automatische Bestände der angegebenen ETFs. "
                    "Nur Aktienbestände, keine verschachtelten Fonds. Gewichte in Prozent "
                    "des gesamten Fonds, Länder und Sektoren auf Englisch wie bei Yahoo. "
                    "Der Import gilt bis zum Verlassen der Seite."
                ),
                html.Pre(
                    "fund,ticker,isin,name,weight_percent,country,sector\n"
                    "SPY,AAPL,US0378331005,Apple Inc.,7.0,United States,Technology"
                ),
                dcc.Upload(
                    id="composition-upload",
                    children=html.Button("CSV auswählen"),
                    accept=".csv",
                    multiple=False,
                ),
            ]
        ),
        dcc.Loading(html.Div(id="composition-result")),
    )


def _load(
    state: AppState,
    name: str,
    dimension: str,
    overrides: dict[str, FundComposition],
) -> tuple[dict[str, dict[str, float]], list[Any], dict[str, str]]:
    portfolio = state.holdings.get_portfolio(name)
    if portfolio is None or not portfolio.tickers:
        raise ValueError("Das ausgewählte Depot ist leer oder existiert nicht mehr.")
    quotes = state.quotes(list(portfolio.tickers))
    valuation = value_portfolio(portfolio, quotes, state.rates)
    if not valuation.is_complete:
        raise ValueError("Kurse / Wechselkurse fehlen für: " + ", ".join(valuation.unpriced))
    instruments: dict[str, Instrument] = {}
    funds: dict[str, FundComposition] = {}
    notes: list[Any] = []
    for ticker in portfolio.tickers:
        try:
            instrument = state.instrument(ticker)
            if dimension == "company" and instrument.asset_class == AssetClass.EQUITY:
                try:
                    isin = state.equity_isin(ticker)
                    if isin:
                        instrument = instrument.model_copy(update={"isin": isin})
                except RateLimitError:
                    raise
                except BTLiteError:
                    pass
            instruments[ticker] = instrument
            if instrument.asset_class in {AssetClass.ETF, AssetClass.FUND}:
                if ticker in overrides:
                    funds[ticker] = overrides[ticker]
                    notes.append(
                        html.Div(
                            [
                                html.Strong(ticker + " · CSV · "),
                                c.provenance_note(overrides[ticker].provenance, state.timezone),
                            ]
                        )
                    )
                    continue
                fund = state.composition(ticker, instrument)
                if fund is None:
                    notes.append(html.P(f"{ticker}: keine ETF-Bestandsquelle verfügbar."))
                    continue
                if fund.provenance.source.value == "yfinance":
                    enriched = []
                    for holding in fund.holdings:
                        try:
                            issuer = state.instrument(holding.ticker)
                        except RateLimitError:
                            raise
                        except BTLiteError:
                            enriched.append(holding)
                            continue
                        if issuer.asset_class == AssetClass.EQUITY:
                            enriched.append(
                                holding.model_copy(
                                    update={
                                        "country": issuer.country,
                                        "sector": issuer.sector,
                                        "isin": issuer.isin or holding.isin,
                                    }
                                )
                            )
                        else:
                            enriched.append(holding.model_copy(update={"asset_class": "Fund"}))
                    fund = fund.model_copy(update={"holdings": enriched})
                funds[ticker] = fund
                notes.append(
                    html.Div(
                        [
                            html.Strong(ticker + " · "),
                            c.provenance_note(fund.provenance, state.timezone),
                        ]
                    )
                )
        except RateLimitError:
            raise
        except BTLiteError as exc:
            notes.append(html.P(f"{ticker}: {exc.user_message}"))
    for quote in quotes.values():
        notes.append(c.provenance_note(quote.provenance, state.timezone))
    labels: dict[str, str] = {}
    if dimension == "company":
        for fund in funds.values():
            for holding in fund.holdings:
                if holding.isin and holding.asset_class == "Equity":
                    label = f"{holding.name or holding.ticker} · {holding.isin}"
                    labels.setdefault(holding.isin, label)
        for instrument in instruments.values():
            if instrument.isin and instrument.asset_class == AssetClass.EQUITY:
                label = f"{instrument.name or instrument.ticker} · {instrument.isin}"
                labels.setdefault(instrument.isin, label)
    return exposures(valuation, instruments, funds, dimension), notes, labels


def render(
    state: AppState,
    name: str | None,
    compare: str | None,
    dimension: str,
    contents: str | None = None,
) -> html.Div:
    """Render actual percentages and source contributions, optionally against a second depot."""
    if not name:
        return c.placeholder("Bitte ein Depot auswählen.")
    try:
        overrides = read_upload(contents)
        data, notes, labels = _load(state, name, dimension, overrides)
        other, other_notes, other_labels = (
            _load(state, compare, dimension, overrides) if compare else ({}, [], {})
        )
        labels.update({key: value for key, value in other_labels.items() if key not in labels})
    except (BTLiteError, ValueError) as exc:
        return html.Div(c.error_banner(exc if isinstance(exc, BTLiteError) else str(exc)))
    categories = sorted(data.keys() | other.keys(), key=lambda k: -sum(data.get(k, {}).values()))
    rows = []
    for label in categories:
        contributions = data.get(label, {})
        weight = sum(contributions.values())
        second = sum(other.get(label, {}).values())
        cells = [
            html.Td(labels.get(label, label)),
            html.Td(f"{weight:.2%}"),
            html.Td(
                " · ".join(f"{ticker}: {value:.2%}" for ticker, value in contributions.items())
            ),
        ]
        if compare:
            cells += [html.Td(f"{second:.2%}"), html.Td(f"{(weight - second) * 100:+.2f} PP")]
        rows.append(html.Tr(cells))
    headings = ["Kategorie / Unternehmenssymbol", name, "Beiträge zum Gesamtdepot"]
    if compare:
        headings += [compare, "Differenz"]
    coverage = 1 - sum(data.get(UNKNOWN, {}).values())
    return html.Div(
        [
            html.P(
                f"Aufgeschlüsselt: {coverage:.2%} des Depotwerts "
                "(Nettoanteil; Cash und Derivate können negativ sein)."
            ),
            html.P(
                "Je nach Quelle können ETF-Daten unvollständig sein. Der Rest bleibt "
                "unbekannt; bekannte Anteile werden nicht auf 100 % hochgerechnet. "
                "Länder = bei iShares Risikoland, bei Direktaktien Unternehmenssitz; "
                "keine Umsatzverteilung. Unternehmen werden über ISIN oder bei "
                "eindeutigem Namen und gleichem Land zusammengeführt."
            ),
            c.graph(composition_bar(data, other, name, compare, labels)),
            html.Table(
                [html.Thead(html.Tr([html.Th(h) for h in headings])), html.Tbody(rows)],
                className="bt-table",
            ),
            html.P(
                "Mehrere Beiträge in einer Unternehmenszeile zeigen Überschneidungen "
                "zwischen Direktanlage und ETFs. Alle Beiträge sind Prozent des Gesamtdepots."
            ),
            html.Details([html.Summary("Datenquellen und Abrufzeiten"), *notes, *other_notes]),
        ]
    )
