"""Chart-Factories fuer Block 4: Backtest-Ergebnisse.

Architekturregel 3: Dieses Modul rechnet nicht. Equity-Kurve, Drawdown und
Kennzahlen kommen fertig aus ``backtest/`` beziehungsweise
``backtest/metrics.py``.

Einzige Ausnahme sind rein darstellungsbezogene Umformungen: die
Normierung der Benchmark auf den Startwert der Strategie (damit beide
Kurven bei demselben Punkt beginnen und optisch vergleichbar sind) und
das Umsortieren der Monatsrenditen in ein Jahr-mal-Monat-Raster.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from btlite.models.backtest import TradeSide
from btlite.viz import theme

if TYPE_CHECKING:
    from datetime import datetime

    from btlite.models.backtest import BacktestResult

#: Pflichtspalte der Equity-Kurve.
EQUITY_COLUMN: str = "equity"

#: Optionale Spalte der Benchmark-Equity.
BENCHMARK_COLUMN: str = "benchmark_equity"

#: Optionale Spalte der bereits berechneten Drawdown-Reihe.
DRAWDOWN_COLUMN: str = "drawdown"

#: Monatskuerzel fuer die Heatmap-Achse.
MONTH_LABELS: tuple[str, ...] = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def equity_curve(
    result: BacktestResult,
    *,
    show_drawdown: bool = True,
    show_trades: bool = False,
    log_y: bool = False,
    title: str | None = None,
) -> go.Figure:
    """Equity-Kurve mit Benchmark-Vergleich und Drawdown-Subplot.

    Die Benchmark wird rein darstellungsbezogen auf den Startwert der
    Strategie skaliert, damit beide Kurven am selben Punkt beginnen.

    Args:
        result: Backtest-Ergebnis. ``equity_curve`` traegt mindestens die
            Spalte ``equity`` in Kontowaehrung.
        show_drawdown: Drawdown-Subplot anzeigen. Nutzt die Spalte
            ``drawdown``, falls vorhanden.
        show_trades: Ein- und Ausstiege als Marker auf der Equity-Kurve.
            Fuer Marker auf dem Kursverlauf siehe :func:`trade_markers`.
        log_y: Logarithmische Equity-Achse. Rein darstellungsbezogen.
        title: Ueberschrift.

    Returns:
        Figure mit Equity, optionaler Benchmark und Drawdown. Bei leerer
        Kurve eine Hinweis-Figure.
    """
    head = title if title is not None else f"Backtest - {result.request.strategy.key}"
    curve = result.equity_curve
    if curve.empty:
        return theme.empty_figure("No equity data", title=head)

    has_dd = show_drawdown and DRAWDOWN_COLUMN in curve.columns
    rows = 2 if has_dd else 1
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28] if has_dd else [1.0],
    )

    fig.add_trace(
        go.Scatter(
            x=curve.index,
            y=curve[EQUITY_COLUMN],
            name="Strategy",
            line={"color": theme.AMBER, "width": 2},
            hovertemplate=theme.hover_template(
                label="Strategy", value_format=",.0f", unit=result.currency.value
            ),
        ),
        row=1,
        col=1,
    )

    if BENCHMARK_COLUMN in curve.columns and not curve[BENCHMARK_COLUMN].dropna().empty:
        bench = curve[BENCHMARK_COLUMN]
        start_equity = curve[EQUITY_COLUMN].dropna()
        first_bench = bench.dropna()
        if not start_equity.empty and not first_bench.empty and float(first_bench.iloc[0]) != 0.0:
            scale = float(start_equity.iloc[0]) / float(first_bench.iloc[0])
            bench = bench * scale
        label = result.request.benchmark or "Benchmark"
        fig.add_trace(
            go.Scatter(
                x=curve.index,
                y=bench,
                name=label,
                line={"color": theme.CYAN, "width": 2, "dash": "dash"},
                hovertemplate=theme.hover_template(
                    label=label, value_format=",.0f", unit=result.currency.value
                ),
            ),
            row=1,
            col=1,
        )

    if show_trades and result.trades:
        _add_trade_markers(fig, result, row=1, on_equity=True)

    if has_dd:
        fig.add_trace(
            go.Scatter(
                x=curve.index,
                y=curve[DRAWDOWN_COLUMN],
                name="Drawdown",
                line={"color": theme.DOWN, "width": 1},
                fill="tozeroy",
                fillcolor=theme.DOWN_FILL,
                showlegend=False,
                hovertemplate=theme.hover_template(label="Drawdown", value_format=".2%"),
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="Drawdown (decimal)", tickformat=".0%", row=2, col=1)

    theme.apply_theme(fig, title=head)
    theme.add_crosshair(fig)
    fig.update_yaxes(title_text=f"Equity ({result.currency.value})", row=1, col=1)
    if log_y:
        fig.update_yaxes(type="log", row=1, col=1)
    fig.update_xaxes(title_text="Date (UTC)", row=rows, col=1)
    fig.update_layout(height=560 if has_dd else 440)
    return fig


def _add_trade_markers(
    fig: go.Figure,
    result: BacktestResult,
    *,
    row: int | None,
    on_equity: bool,
    prices: pd.Series | None = None,
) -> None:
    """Zeichnet Ein- und Ausstiegsmarker.

    Args:
        fig: Zielfigur.
        result: Backtest-Ergebnis mit der Trade-Liste.
        row: Subplot-Zeile, einsbasiert. ``None`` fuer eine Figure ohne
            Subplot-Raster.
        on_equity: Wenn True, werden die Marker auf der Equity-Kurve
            positioniert, sonst auf der uebergebenen Kursreihe.
        prices: Kursreihe fuer die Y-Position, wenn ``on_equity`` False ist.
    """
    # Ohne Subplot-Raster darf add_trace kein row/col bekommen.
    target: dict[str, int] = {} if row is None else {"row": row, "col": 1}
    reference: pd.Series | None = result.equity_curve[EQUITY_COLUMN] if on_equity else prices

    def y_at(when: datetime, fallback: float | None) -> float | None:
        """Sucht den Referenzwert zum Zeitpunkt eines Trade.

        Args:
            when: Zeitpunkt des Ein- oder Ausstiegs, tz-aware UTC.
            fallback: Rueckfallwert, wenn der Zeitpunkt nicht zuzuordnen ist.

        Returns:
            Referenzwert am naechstgelegenen Bar, sonst ``fallback``.
        """
        if reference is None:
            return fallback
        try:
            idx = int(reference.index.get_indexer(pd.Index([when]), method="nearest")[0])
        except (KeyError, TypeError, ValueError):
            return fallback
        if idx < 0:
            return fallback
        return float(reference.iloc[idx])

    entries_x: list[object] = []
    entries_y: list[float] = []
    entries_text: list[str] = []
    exits_x: list[object] = []
    exits_y: list[float] = []
    exits_text: list[str] = []

    for trade in result.trades:
        side = "Long" if trade.side is TradeSide.LONG else "Short"
        y_in = y_at(trade.entry_time, trade.entry_price)
        if y_in is not None:
            entries_x.append(trade.entry_time)
            entries_y.append(y_in)
            entries_text.append(
                f"{trade.ticker} {side} Entry<br>Price: "
                f"{theme.format_value(trade.entry_price, value_format=',.2f')}"
            )
        if trade.exit_time is not None:
            y_out = y_at(trade.exit_time, trade.exit_price)
            if y_out is not None:
                exits_x.append(trade.exit_time)
                exits_y.append(y_out)
                exits_text.append(
                    f"{trade.ticker} {side} Exit<br>PnL: "
                    f"{theme.format_value(trade.pnl, value_format=',.2f')}"
                )

    if entries_x:
        fig.add_trace(
            go.Scatter(
                x=entries_x,
                y=entries_y,
                mode="markers",
                name="Entry",
                marker={
                    "symbol": "triangle-up",
                    "size": 11,
                    "color": theme.UP,
                    "line": {"color": theme.SURFACE, "width": 1},
                },
                text=entries_text,
                hovertemplate="%{text}<extra></extra>",
            ),
            **target,
        )
    if exits_x:
        fig.add_trace(
            go.Scatter(
                x=exits_x,
                y=exits_y,
                mode="markers",
                name="Exit",
                marker={
                    "symbol": "triangle-down",
                    "size": 11,
                    "color": theme.DOWN,
                    "line": {"color": theme.SURFACE, "width": 1},
                },
                text=exits_text,
                hovertemplate="%{text}<extra></extra>",
            ),
            **target,
        )


def trade_markers(
    result: BacktestResult,
    prices: pd.Series,
    *,
    title: str | None = None,
) -> go.Figure:
    """Kursverlauf mit Ein- und Ausstiegsmarkern der Trades.

    Args:
        result: Backtest-Ergebnis mit der Trade-Liste.
        prices: Kursreihe des gehandelten Instruments, in Kurswaehrung,
            mit UTC-Index. Kommt aus der ``PriceSeries``, nicht aus dem
            Backtest.
        title: Ueberschrift.

    Returns:
        Linienchart mit Trade-Markern. Bei leerer Kursreihe eine
        Hinweis-Figure.
    """
    head = title if title is not None else "Trades on the price series"
    if prices.dropna().empty:
        return theme.empty_figure("No price data", title=head)

    fig = go.Figure(
        go.Scatter(
            x=prices.index,
            y=prices,
            name="Price",
            line={"color": theme.AMBER, "width": 2},
            hovertemplate=theme.hover_template(label="Price", value_format=",.2f"),
        )
    )
    if result.trades:
        _add_trade_markers(fig, result, row=None, on_equity=False, prices=prices)

    theme.apply_theme(fig, title=head)
    theme.add_crosshair(fig)
    theme.style_axes(fig, x_title="Date (UTC)", y_title=f"Price ({result.currency.value})")
    fig.update_layout(height=460)
    return fig


def monthly_returns_heatmap(
    monthly_returns: pd.Series,
    *,
    title: str | None = None,
) -> go.Figure:
    """Monatsrenditen als Jahr-mal-Monat-Heatmap.

    Die Farbskala ist divergierend und symmetrisch um 0, damit Gewinn- und
    Verlustmonate gleich stark wirken. Monate ohne Beobachtung bleiben leer
    und erscheinen nicht als Nullrendite.

    Args:
        monthly_returns: Monatsrenditen, dezimal, mit UTC-Index. Fertig
            berechnet aus ``backtest/metrics.py``.
        title: Ueberschrift.

    Returns:
        Heatmap mit Jahren als Zeilen und Monaten als Spalten. Bei leerer
        Reihe eine Hinweis-Figure.
    """
    head = title if title is not None else "Monthly returns"
    clean = monthly_returns.dropna()
    if clean.empty:
        return theme.empty_figure("No monthly returns", title=head)

    index = pd.DatetimeIndex(clean.index)
    frame = pd.DataFrame({"year": index.year, "month": index.month, "value": clean.to_numpy()})
    grid = frame.pivot_table(index="year", columns="month", values="value", aggfunc="last")
    grid = grid.reindex(columns=range(1, 13))
    years = [str(y) for y in grid.index]

    # Symmetrische Skalierung um 0: der groesste Betrag setzt beide Enden.
    limit = float(grid.abs().max().max())
    if not limit or pd.isna(limit):
        limit = 0.01

    text = [[theme.NA_TEXT if pd.isna(v) else f"{v:.1%}" for v in row] for row in grid.to_numpy()]
    fig = go.Figure(
        go.Heatmap(
            z=grid.to_numpy(),
            x=list(MONTH_LABELS),
            y=years,
            colorscale=theme.diverging_scale(),
            zmid=0.0,
            zmin=-limit,
            zmax=limit,
            text=text,
            texttemplate="%{text}",
            textfont={"family": theme.FONT_FAMILY, "size": theme.FONT_SIZE - 1},
            hovertemplate="%{y} %{x}: %{text}<extra></extra>",
            xgap=2,
            ygap=2,
            colorbar={
                "title": {"text": "Return", "font": {"color": theme.INK_MUTED}},
                "tickformat": ".0%",
                "outlinecolor": theme.GRID,
                "tickfont": {"color": theme.INK_MUTED},
            },
        )
    )
    theme.apply_theme(fig, title=head)
    theme.style_axes(fig, x_title="Month", y_title="Year")
    fig.update_layout(
        height=max(280, 120 + 34 * len(years)),
        hovermode="closest",
        yaxis={"autorange": "reversed"},
    )
    return fig
