"""Seite PORT - Optimierung, Effizienzgrenze, Korrelation und Risiko.

Die Seite verdrahtet ``portfolio.optimizer`` mit ``viz.frontier`` und
``viz.risk``. Gewichte, Frontier und Korrelationsmatrix kommen fertig aus
der Rechenschicht.

Hinweis zum Risiko-Abschnitt: die Seite zeigt zwei Arten von Kennzahlen
nebeneinander. Die **erwarteten** Werte stammen aus den Schaetzern der
Optimierung, die **realisierten** aus der tatsaechlich beobachteten
Kurshistorie des gewichteten Portfolios. ``optimize`` fuellt beide; die
realisierten gehen durch ``analytics.series.portfolio_returns``. Bleibt
``PortfolioResult.risk`` ausnahmsweise None, weil die Historie nicht
reicht, erscheint sichtbar "n/a" statt einer erfundenen Zahl.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from dash import dcc, html

from btlite.core.errors import BTLiteError
from btlite.core.logging import get_logger
from btlite.models.common import Interval
from btlite.models.portfolio import OptimizationObjective, PortfolioRequest
from btlite.ui import formatting as fmt
from btlite.ui.layouts import components as c

if TYPE_CHECKING:
    from btlite.models.portfolio import PortfolioResult
    from btlite.ui.state import AppState

_LOG = get_logger(__name__)

#: Route dieser Seite.
PATH = "/port"

#: Element-IDs der Bedienelemente.
TICKER_STORE = "port-tickers"
OBJECTIVE_INPUT = "port-objective"
RESULT_OUTPUT = "port-result"
PORTFOLIO_PICKER = "port-portfolio-picker"

#: Mindestzahl an Titeln fuer eine Optimierung. Mit einem einzigen Titel
#: gibt es nichts zu gewichten.
MIN_TICKERS: Final[int] = 2

#: Zeitraum der Schaetzung.
LOOKBACK: Final[str] = "3y"

#: Anzeigenamen der Zielfunktionen.
OBJECTIVE_LABELS: Final[dict[OptimizationObjective, str]] = {
    OptimizationObjective.MAX_SHARPE: "Max Sharpe",
    OptimizationObjective.MIN_VOLATILITY: "Min volatility",
    OptimizationObjective.HRP: "Hierarchical Risk Parity",
    OptimizationObjective.EQUAL_WEIGHT: "Equal weight",
    OptimizationObjective.MAX_QUADRATIC_UTILITY: "Max utility",
}


def portfolio_options(state: AppState) -> list[dict[str, str]]:
    """Nennt die gespeicherten Depots, die sich analysieren lassen.

    Ein Depot mit weniger als :data:`MIN_TICKERS` Titeln erscheint gar
    nicht erst: die Optimierung haette dort nichts zu verteilen, und ein
    Eintrag, der beim Anklicken nur eine Fehlermeldung bringt, ist
    schlechter als keiner.

    Args:
        state: UI-Zustand mit dem Depot-Speicher.

    Returns:
        Optionen fuer das Dropdown, je mit Titelzahl im Label. Leer, wenn
        die Datei nicht lesbar ist - die Seite bleibt dann benutzbar, nur
        ohne Auswahl.
    """
    try:
        names = state.holdings.list_portfolios()
    except BTLiteError as exc:
        _LOG.warning("Depots nicht lesbar: %s", exc.detail or exc.user_message)
        return []

    options: list[dict[str, str]] = []
    for name in names:
        portfolio = state.holdings.get_portfolio(name)
        if portfolio is None:
            continue
        count = len(portfolio.tickers)
        if count < MIN_TICKERS:
            continue
        options.append({"label": f"{name} ({count})", "value": name})
    return options


def tickers_of(state: AppState, name: str | None) -> list[str]:
    """Liest die Symbole eines gespeicherten Depots.

    Args:
        state: UI-Zustand.
        name: Name des Depots, oder None.

    Returns:
        Sortierte Symbole ohne Wiederholung, leer bei unbekanntem Namen.
        Stueckzahlen und Einstandskurse bleiben aussen vor: die Seite
        fragt, wie man diese Titel gewichten sollte, nicht wie das Depot
        heute dasteht - das steht auf WL.
    """
    if not name:
        return []
    try:
        portfolio = state.holdings.get_portfolio(name)
    except BTLiteError as exc:
        _LOG.warning("Depot %s nicht lesbar: %s", name, exc.detail or exc.user_message)
        return []
    return list(portfolio.tickers) if portfolio is not None else []


def layout(state: AppState, tickers: list[str], args: tuple[str, ...] = ()) -> html.Div:
    """Baut die Seite PORT.

    Args:
        state: UI-Zustand mit Provider und Puffern.
        tickers: Anlageuniversum aus der Kommandozeile, mindestens zwei.
        args: Weitere Argumente. Ein Zielfunktions-Key waehlt das Ziel,
            etwa ``AAPL,MSFT,KO PORT min_volatility GO``.

    Returns:
        Seiteninhalt mit Zielwahl, Gewichten, Frontier und Risiko.
    """
    symbols = tickers if len(tickers) >= 2 else state.default_tickers
    objective = _objective_from_args(args)

    return html.Div(
        [
            c.page_header(
                f"Portfolio and risk - {', '.join(symbols)}",
                f"Estimated over {LOOKBACK}, risk-free rate "
                f"{fmt.format_percent(state.settings.analytics.risk_free_rate, decimals=2)}",
            ),
            dcc.Store(id=TICKER_STORE, data=symbols),
            c.controls(
                c.labeled(
                    "Saved portfolio",
                    dcc.Dropdown(
                        id=PORTFOLIO_PICKER,
                        options=portfolio_options(state),
                        value=None,
                        placeholder="Analyse a saved portfolio",
                        clearable=True,
                        className="bt-dropdown",
                    ),
                ),
                c.labeled(
                    "Objective",
                    dcc.Dropdown(
                        id=OBJECTIVE_INPUT,
                        options=[
                            {"label": label, "value": obj.value}
                            for obj, label in OBJECTIVE_LABELS.items()
                        ],
                        value=objective.value,
                        clearable=False,
                        className="bt-dropdown",
                    ),
                ),
            ),
            c.loading(
                html.Div(
                    id=RESULT_OUTPUT,
                    children=render_result(state, symbols, objective),
                )
            ),
        ],
        className="bt-page bt-page-port",
    )


def _objective_from_args(args: tuple[str, ...]) -> OptimizationObjective:
    """Liest die Zielfunktion aus den Kommandozeilen-Argumenten.

    Args:
        args: Argumente hinter dem Mnemonic.

    Returns:
        Gewaehlte Zielfunktion, Default ``MAX_SHARPE``.
    """
    for arg in args:
        candidate = arg.strip().lower()
        for objective in OBJECTIVE_LABELS:
            if candidate == objective.value:
                return objective
    return OptimizationObjective.MAX_SHARPE


def run_optimization(
    state: AppState,
    tickers: list[str],
    objective: OptimizationObjective,
) -> PortfolioResult:
    """Fuehrt eine Portfolio-Optimierung aus.

    Args:
        state: UI-Zustand.
        tickers: Anlageuniversum, mindestens zwei Symbole.
        objective: Zielfunktion.

    Returns:
        Das Optimierungsergebnis.

    Raises:
        BTLiteError: Wenn Kurse fehlen oder der Solver scheitert.
    """
    from btlite.core.errors import InsufficientDataError
    from btlite.portfolio.optimizer import optimize

    prices = state.histories(tickers, period=LOOKBACK, interval=Interval.D1)
    if len(prices) < 2:
        msg = "An optimization needs at least two tickers with prices."
        raise InsufficientDataError(
            msg,
            detail=f"angefragt={tickers}, mit Kursen={sorted(prices)}",
            required=2,
            available=len(prices),
        )

    request = PortfolioRequest(
        tickers=sorted(prices),
        objective=objective,
        lookback_period=LOOKBACK,
        risk_free_rate=state.settings.analytics.risk_free_rate,
        allow_shorting=state.settings.portfolio.allow_shorting,
        max_weight=state.settings.portfolio.max_weight,
        benchmark=state.settings.analytics.benchmark_ticker,
    )
    result: PortfolioResult = optimize(request, prices)
    return result


def render_result(
    state: AppState,
    tickers: list[str],
    objective: OptimizationObjective,
) -> html.Div:
    """Baut Gewichte, Frontier, Korrelation und Risikometriken.

    Args:
        state: UI-Zustand.
        tickers: Anlageuniversum.
        objective: Zielfunktion.

    Returns:
        Container mit allen Abschnitten, oder ein Fehlerbanner.
    """
    from btlite.viz import frontier as viz_frontier
    from btlite.viz import risk as viz_risk

    try:
        result = run_optimization(state, tickers, objective)
    except BTLiteError as exc:
        return html.Div([c.error_banner(exc)])

    blocks: list[Any] = [c.provenance_note(result.provenance, state.timezone)]

    banner = c.warning_banner(result.warnings)
    if banner is not None:
        blocks.append(banner)

    blocks.append(c.section("Expected metrics", _expected_tiles(result)))
    blocks.append(
        c.section(
            "Target weights",
            c.graph(viz_frontier.weights_bar(result), height=c.CHART_HEIGHT_SMALL),
            _weights_table(result),
        )
    )

    if result.frontier is not None:
        optimal = (
            (result.expected_volatility, result.expected_return)
            if result.expected_volatility is not None and result.expected_return is not None
            else None
        )
        blocks.append(
            c.section(
                "Efficient frontier",
                c.graph(
                    viz_frontier.efficient_frontier(result.frontier, optimal=optimal),
                    height=c.CHART_HEIGHT,
                ),
            )
        )
    else:
        blocks.append(c.section("Efficient frontier", c.placeholder("Not computable.")))

    if result.correlation is not None:
        blocks.append(
            c.section(
                "Correlation matrix",
                c.graph(
                    viz_risk.correlation_heatmap(result.correlation),
                    height=c.CHART_HEIGHT_SMALL,
                ),
            )
        )

    blocks.append(c.section("Risk metrics per holding", _risk_table(state, result)))
    return html.Div(blocks, className="bt-port-body")


def _expected_tiles(result: PortfolioResult) -> html.Div:
    """Baut die Kacheln der erwarteten Kennzahlen.

    Args:
        result: Optimierungsergebnis.

    Returns:
        Kachel-Raster.
    """
    risk = result.risk
    return c.tiles(
        [
            (
                "Expected return",
                fmt.format_percent(result.expected_return, signed=True),
                fmt.direction_class(result.expected_return),
            ),
            ("Expected volatility", fmt.format_percent(result.expected_volatility)),
            ("Expected Sharpe", fmt.format_number(result.sharpe_ratio)),
            (
                "Realized volatility",
                fmt.format_percent(None if risk is None else risk.volatility),
                fmt.value_class(None if risk is None else risk.volatility),
            ),
            (
                "Realized max drawdown",
                fmt.format_percent(None if risk is None else risk.max_drawdown),
                fmt.value_class(None if risk is None else risk.max_drawdown),
            ),
            (
                "Unallocated",
                fmt.format_currency(result.leftover_cash, result.currency),
            ),
            ("Invested amount", fmt.format_currency(result.request.total_value, result.currency)),
            (
                "Objective",
                OBJECTIVE_LABELS.get(result.request.objective, result.request.objective.value),
            ),
        ]
    )


def _weights_table(result: PortfolioResult) -> html.Div | html.Table:
    """Rendert die Zielgewichte als Tabelle.

    Args:
        result: Optimierungsergebnis.

    Returns:
        Tabelle der Gewichte, oder ein Hinweis ohne Positionen.
    """
    if not result.positions:
        return c.placeholder("No positions.")

    head = html.Thead(
        html.Tr(
            [
                html.Th("Ticker"),
                html.Th("Weight"),
                html.Th("Shares"),
                html.Th("Price"),
                html.Th("Value"),
            ]
        )
    )
    rows = [
        html.Tr(
            [
                html.Td(p.ticker),
                html.Td(fmt.format_percent(p.weight, decimals=2)),
                html.Td(fmt.format_number(p.shares, decimals=0)),
                html.Td(fmt.format_currency(p.price, result.currency)),
                html.Td(fmt.format_currency(p.value, result.currency)),
            ]
        )
        for p in sorted(result.positions, key=lambda x: -x.weight)
    ]
    return html.Table([head, html.Tbody(rows)], className="bt-table")


def _risk_table(state: AppState, result: PortfolioResult) -> html.Div:
    """Listet die Risikokennzahlen der Einzelwerte.

    Die Kennzahlen kommen aus ``analytics.risk.compute_risk_metrics``,
    gerechnet auf den Renditen der jeweiligen Kursreihe gegen die
    Benchmark. Ein Portfolio-Aggregat fehlt, siehe Modul-Docstring.

    Args:
        state: UI-Zustand.
        result: Optimierungsergebnis, liefert das Universum.

    Returns:
        Tabellen-Container, oder ein Hinweis.
    """
    from btlite.analytics.risk import compute_risk_metrics

    benchmark_symbol = result.request.benchmark
    benchmark_returns = None
    if benchmark_symbol:
        try:
            benchmark_returns = state.history(benchmark_symbol, period=LOOKBACK).returns()
        except BTLiteError:
            benchmark_returns = None

    rows: list[html.Tr] = []
    for position in sorted(result.positions, key=lambda x: -x.weight):
        try:
            series = state.history(position.ticker, period=LOOKBACK)
            metrics = compute_risk_metrics(
                series.returns(),
                benchmark_returns,
                interval=series.interval,
                risk_free_rate=state.settings.analytics.risk_free_rate,
            )
        except BTLiteError:
            rows.append(
                html.Tr([html.Td(position.ticker), *[html.Td(fmt.NA_TEXT, className="bt-na")] * 7])
            )
            continue

        rows.append(
            html.Tr(
                [
                    html.Td(position.ticker),
                    html.Td(fmt.format_percent(metrics.volatility)),
                    html.Td(fmt.format_number(metrics.sharpe)),
                    html.Td(fmt.format_number(metrics.sortino)),
                    html.Td(
                        fmt.format_percent(metrics.max_drawdown),
                        className=fmt.direction_class(metrics.max_drawdown),
                    ),
                    html.Td(fmt.format_percent(metrics.var_historical)),
                    html.Td(fmt.format_percent(metrics.cvar_historical)),
                    html.Td(fmt.format_number(metrics.beta)),
                ]
            )
        )

    head = html.Thead(
        html.Tr(
            [
                html.Th("Ticker"),
                html.Th("Volatility"),
                html.Th("Sharpe"),
                html.Th("Sortino"),
                html.Th("Max DD"),
                html.Th("VaR 95"),
                html.Th("CVaR 95"),
                html.Th("Beta"),
            ]
        )
    )
    note = html.Div(
        "Risk figures per holding. An aggregated portfolio risk is missing as long as "
        "the analytics layer does not supply a weighted return series.",
        className="bt-notice",
    )
    return html.Div(
        [note, html.Table([head, html.Tbody(rows)], className="bt-table")],
        className="bt-risk",
    )
