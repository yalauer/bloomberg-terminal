"""Seite BT - Strategiewahl, Equity-Kurve, Drawdown und Trade-Liste.

Die Seite verdrahtet ``backtest.registry``, die Strategie und die Engine.
Signale erzeugt die Strategie, ausgefuehrt wird in ``backtest/``; das UI
setzt nur die Anfrage zusammen und stellt das Ergebnis dar.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from dash import dcc, html

from btlite.core.errors import BTLiteError
from btlite.models.backtest import BacktestRequest, StrategyConfig
from btlite.models.common import Interval
from btlite.ui import formatting as fmt
from btlite.ui.layouts import components as c

if TYPE_CHECKING:
    from btlite.models.backtest import BacktestResult, PerformanceMetrics, Trade
    from btlite.ui.state import AppState

#: Route dieser Seite.
PATH = "/bt"

#: Element-IDs der Bedienelemente.
TICKER_STORE = "bt-tickers"
STRATEGY_INPUT = "bt-strategy"
PARAMS_INPUT = "bt-params"
RESULT_OUTPUT = "bt-result"

#: Zeitraum der Kursbasis eines Backtest.
BACKTEST_PERIOD: Final[str] = "5y"

#: Obergrenze der angezeigten Trades. Laengere Listen werden gekuerzt.
MAX_TRADES: Final[int] = 100


def layout(state: AppState, tickers: list[str], args: tuple[str, ...] = ()) -> html.Div:
    """Baut die Seite BT.

    Args:
        state: UI-Zustand mit Provider, Registry und Engine.
        tickers: Zu handelnde Symbole aus der Kommandozeile.
        args: Weitere Argumente. Ein Strategie-Key waehlt die Strategie,
            etwa ``AAPL BT sma_cross GO``. ``name=wert``-Paare setzen
            Parameter, etwa ``SPY BT sma_cross fast=10 slow=30 GO``.

    Returns:
        Seiteninhalt mit Strategie- und Parameterwahl sowie Ergebnis.
    """
    symbols = tickers or state.default_tickers
    registry = state.strategy_registry()
    strategy_key = _strategy_from_args(args, registry)
    # Parameter aus der Kommandozeile ueberschreiben die Defaults der
    # Strategie, damit "SPY BT sma_cross fast=10 GO" nicht still die
    # Voreinstellung rechnet.
    defaults = parse_params(" ".join(args), registry.get(strategy_key).default_params)

    return html.Div(
        [
            c.page_header(
                f"{', '.join(symbols)} - Backtest",
                "Signal and execution are separate. "
                "A signal at t is executed later, according to execution_lag.",
            ),
            dcc.Store(id=TICKER_STORE, data=symbols),
            c.controls(
                c.labeled(
                    "Strategy",
                    dcc.Dropdown(
                        id=STRATEGY_INPUT,
                        options=[
                            {"label": cls.label or cls.key, "value": cls.key}
                            for cls in registry.list_all()
                        ],
                        value=strategy_key,
                        clearable=False,
                        className="bt-dropdown",
                    ),
                ),
                c.labeled(
                    "Parameters",
                    dcc.Input(
                        id=PARAMS_INPUT,
                        type="text",
                        value=_params_text(defaults),
                        debounce=True,
                        className="bt-input",
                    ),
                ),
            ),
            html.Div(
                'Parameters as pairs, for example "fast=20 slow=100". '
                "Leave empty for the defaults.",
                className="bt-notice",
            ),
            c.loading(
                html.Div(
                    id=RESULT_OUTPUT,
                    children=render_result(state, symbols, strategy_key, dict(defaults)),
                )
            ),
        ],
        className="bt-page bt-page-bt",
    )


def _strategy_from_args(args: tuple[str, ...], registry: Any) -> str:
    """Liest den Strategie-Key aus den Kommandozeilen-Argumenten.

    Args:
        args: Argumente hinter dem Mnemonic.
        registry: Strategie-Registry.

    Returns:
        Gewaehlter Key, sonst der erste registrierte.
    """
    keys = registry.keys()
    for arg in args:
        candidate = arg.strip().lower()
        if candidate in keys:
            return candidate
    return str(keys[0]) if keys else "sma_cross"


def _params_text(params: dict[str, Any]) -> str:
    """Setzt ein Parameter-Dictionary als Eingabetext.

    Args:
        params: Strategie-Parameter.

    Returns:
        Text wie ``"fast=50 slow=200"``.
    """
    return " ".join(f"{k}={v}" for k, v in params.items())


def parse_params(raw: str, defaults: dict[str, Any]) -> dict[str, Any]:
    """Liest Parameter aus der Eingabezeile.

    Akzeptiert ``name=wert``-Paare, getrennt durch Leerzeichen oder Komma.
    Werte werden nach int, dann float, sonst als Text uebernommen.
    Unbekannte Namen bleiben erhalten - die Strategie validiert selbst und
    wirft ``StrategyConfigError``.

    Args:
        raw: Rohe Eingabe.
        defaults: Default-Parameter der Strategie als Grundlage.

    Returns:
        Zusammengefuehrte Parameter.
    """
    params = dict(defaults)
    for token in raw.replace(",", " ").split():
        name, _, value = token.partition("=")
        if not name or not value:
            continue
        params[name.strip()] = _coerce(value.strip())
    return params


def _coerce(raw: str) -> Any:
    """Wandelt einen Parameterwert in den engsten passenden Typ.

    Args:
        raw: Rohwert als Text.

    Returns:
        int, float oder der unveraenderte Text.
    """
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def run_backtest(
    state: AppState,
    tickers: list[str],
    strategy_key: str,
    params: dict[str, Any],
) -> BacktestResult:
    """Fuehrt einen Backtest aus.

    Args:
        state: UI-Zustand.
        tickers: Zu handelnde Symbole.
        strategy_key: Key der Strategie.
        params: Strategie-Parameter.

    Returns:
        Das Backtest-Ergebnis.

    Raises:
        BTLiteError: Wenn Kurse, Strategie oder Engine scheitern.
    """
    from btlite.core.errors import InsufficientDataError

    registry = state.strategy_registry()
    engine = state.backtest_engine()

    config = StrategyConfig(key=strategy_key, params=params)
    strategy = registry.get(strategy_key)(config)

    prices = state.histories(tickers, period=BACKTEST_PERIOD, interval=Interval.D1)
    if not prices:
        msg = "No price history is available for any of the tickers."
        raise InsufficientDataError(msg, detail=f"angefragt={tickers}", required=1, available=0)

    # Ohne erreichbare Benchmark bleiben Beta, Alpha und Tracking Error
    # None. Das ist der Vertrag von ``metrics``, kein Fehler.
    benchmark_symbol = state.settings.analytics.benchmark_ticker
    try:
        benchmark = state.history(benchmark_symbol, period=BACKTEST_PERIOD)
    except BTLiteError:
        benchmark = None

    signals = strategy.generate_signals(prices)
    request = BacktestRequest(
        tickers=sorted(prices),
        strategy=config,
        interval=Interval.D1,
        init_cash=state.settings.backtest.init_cash,
        fees=state.settings.backtest.fees,
        slippage=state.settings.backtest.slippage,
        benchmark=benchmark_symbol if benchmark is not None else None,
    )
    result: BacktestResult = engine.run(request, prices, signals, benchmark=benchmark)
    return result


def render_result(
    state: AppState,
    tickers: list[str],
    strategy_key: str,
    params: dict[str, Any],
) -> html.Div:
    """Baut Equity-Kurve, Metriken und Trade-Liste.

    Args:
        state: UI-Zustand.
        tickers: Zu handelnde Symbole.
        strategy_key: Key der Strategie.
        params: Strategie-Parameter.

    Returns:
        Container mit allen Abschnitten, oder ein Fehlerbanner.
    """
    from btlite.viz import backtest as viz_bt
    from btlite.viz import risk as viz_risk

    try:
        result = run_backtest(state, tickers, strategy_key, params)
    except BTLiteError as exc:
        return html.Div([c.error_banner(exc)])

    blocks: list[Any] = [c.provenance_note(result.provenance, state.timezone)]

    banner = c.warning_banner(result.warnings)
    if banner is not None:
        blocks.append(banner)

    blocks.append(
        c.section(
            f"Equity curve ({result.engine_used})",
            c.graph(viz_bt.equity_curve(result, show_trades=True), height=c.CHART_HEIGHT),
        )
    )

    if "drawdown" in result.equity_curve.columns:
        blocks.append(
            c.section(
                "Drawdown",
                c.graph(
                    viz_risk.drawdown_chart(
                        result.equity_curve["drawdown"],
                        max_drawdown=result.metrics.max_drawdown,
                    ),
                    height=c.CHART_HEIGHT_SMALL,
                ),
            )
        )

    blocks.append(c.section("Metrics", metrics_tiles(result.metrics, result.currency.value)))
    if result.benchmark_metrics is not None:
        blocks.append(
            c.section(
                f"Benchmark ({result.request.benchmark})",
                metrics_tiles(result.benchmark_metrics, result.currency.value),
            )
        )
    blocks.append(c.section("Trades", trade_table(result.trades, result)))
    return html.Div(blocks, className="bt-bt-body")


def metrics_tiles(metrics: PerformanceMetrics, currency: str) -> html.Div:
    """Baut das Kennzahlen-Panel eines Backtest.

    Args:
        metrics: Kennzahlen des Laufs. Renditen annualisiert und dezimal.
        currency: Kontowaehrung fuer absolute Betraege.

    Returns:
        Kachel-Raster. Fehlende Kennzahlen zeigen "n/a".
    """
    return c.tiles(
        [
            (
                "Total return",
                fmt.format_percent(metrics.total_return, signed=True),
                fmt.direction_class(metrics.total_return),
            ),
            (
                "CAGR",
                fmt.format_percent(metrics.cagr, signed=True),
                fmt.direction_class(metrics.cagr),
            ),
            ("Volatility", fmt.format_percent(metrics.volatility)),
            ("Sharpe", fmt.format_number(metrics.sharpe)),
            ("Sortino", fmt.format_number(metrics.sortino)),
            ("Calmar", fmt.format_number(metrics.calmar)),
            (
                "Max Drawdown",
                fmt.format_percent(metrics.max_drawdown),
                fmt.direction_class(metrics.max_drawdown),
            ),
            ("Underwater", _days(metrics.max_drawdown_duration_days)),
            ("VaR 95", fmt.format_percent(metrics.var_95)),
            ("CVaR 95", fmt.format_percent(metrics.cvar_95)),
            ("Beta", fmt.format_number(metrics.beta)),
            ("Alpha", fmt.format_percent(metrics.alpha, signed=True)),
            ("Tracking Error", fmt.format_percent(metrics.tracking_error)),
            ("Information Ratio", fmt.format_number(metrics.information_ratio)),
            ("Trades", fmt.format_integer(metrics.n_trades)),
            ("Hit rate", fmt.format_percent(metrics.hit_rate)),
            ("Profit Factor", fmt.format_number(metrics.profit_factor)),
            ("Average win", fmt.format_number(metrics.avg_win)),
            ("Average loss", fmt.format_number(metrics.avg_loss)),
            ("Holding period (bars)", fmt.format_number(metrics.avg_duration_bars, decimals=1)),
            ("Time in market", fmt.format_percent(metrics.exposure)),
            ("Total costs", f"{currency} {fmt.format_number(metrics.total_fees)}"),
        ]
    )


def _days(value: int | None) -> str:
    """Formatiert eine Dauer in Tagen.

    Args:
        value: Anzahl Tage, oder None.

    Returns:
        Text wie ``"412 Tage"``, oder "n/a".
    """
    if value is None:
        return fmt.NA_TEXT
    return f"{fmt.format_integer(value)} days"


def trade_table(trades: list[Trade], result: BacktestResult) -> html.Div:
    """Rendert die Trade-Liste.

    Args:
        trades: Abgeschlossene und offene Trades.
        result: Ergebnis, fuer Waehrung und Zeitzone der Anzeige.

    Returns:
        Tabellen-Container, oder ein Hinweis bei leerer Liste.
    """
    from btlite.ui.state import get_state

    if not trades:
        return c.placeholder("The strategy did not trigger a single trade.")

    timezone = get_state().timezone
    currency = result.currency
    shown = trades[:MAX_TRADES]

    head = html.Thead(
        html.Tr(
            [
                html.Th("Ticker"),
                html.Th("Side"),
                html.Th("Entry"),
                html.Th("Entry price"),
                html.Th("Exit"),
                html.Th("Exit price"),
                html.Th("Shares"),
                html.Th("PnL"),
                html.Th("Return"),
                html.Th("Costs"),
                html.Th("Bars"),
            ]
        )
    )
    rows = [
        html.Tr(
            [
                html.Td(t.ticker),
                html.Td(t.side.value.capitalize()),
                html.Td(fmt.format_timestamp(t.entry_time, timezone)),
                html.Td(fmt.format_currency(t.entry_price, currency)),
                html.Td(fmt.format_timestamp(t.exit_time, timezone)),
                html.Td(fmt.format_currency(t.exit_price, currency)),
                html.Td(fmt.format_number(t.size, decimals=2)),
                html.Td(
                    fmt.format_currency(t.pnl, currency, decimals=2),
                    className=fmt.direction_class(t.pnl),
                ),
                html.Td(
                    fmt.format_percent(t.return_pct, signed=True),
                    className=fmt.direction_class(t.return_pct),
                ),
                html.Td(fmt.format_currency(t.fees_paid, currency)),
                html.Td(fmt.format_integer(t.duration_bars)),
            ]
        )
        for t in shown
    ]

    children: list[Any] = [html.Table([head, html.Tbody(rows)], className="bt-table")]
    if len(trades) > MAX_TRADES:
        children.insert(
            0,
            html.Div(
                f"Showing {MAX_TRADES} of {len(trades)} trades.",
                className="bt-notice",
            ),
        )
    return html.Div(children, className="bt-trades")
