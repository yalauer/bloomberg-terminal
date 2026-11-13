"""Chart-Factories fuer Block 5: Risiko-Darstellungen.

Architekturregel 3: Dieses Modul rechnet nicht. Korrelationsmatrix,
Drawdown-Reihe, Rolling-Volatilitaet und Rolling-Beta kommen fertig aus
``analytics/risk.py``; VaR und CVaR aus
:class:`~btlite.models.portfolio.RiskMetrics`.

Einheiten: Renditen und Volatilitaeten dezimal (0.12 = 12 Prozent),
Volatilitaeten annualisiert. Drawdowns, VaR und CVaR sind negativ.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go

from btlite.viz import theme

if TYPE_CHECKING:
    from btlite.models.portfolio import RiskMetrics


def correlation_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str | None = None,
    show_values: bool = True,
) -> go.Figure:
    """Korrelations-Heatmap, divergierend und symmetrisch um 0.

    Die Skala ist fest auf -1 bis +1 verankert, damit die Farbe zwischen
    zwei Charts vergleichbar bleibt und der neutrale Mittelpunkt exakt auf
    der Nullkorrelation sitzt.

    Args:
        matrix: Quadratische Korrelationsmatrix, Werte dezimal in [-1, 1].
        title: Ueberschrift.
        show_values: Werte in die Zellen schreiben.

    Returns:
        Heatmap-Figure. Bei leerer Matrix eine Hinweis-Figure.
    """
    head = title if title is not None else "Correlation matrix"
    if matrix.empty:
        return theme.empty_figure("No correlation data", title=head)

    labels = [str(c) for c in matrix.columns]
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=labels,
            y=[str(i) for i in matrix.index],
            colorscale=theme.diverging_scale(),
            zmid=0.0,
            zmin=-1.0,
            zmax=1.0,
            texttemplate="%{z:.2f}" if show_values else None,
            textfont={"family": theme.FONT_FAMILY, "size": theme.FONT_SIZE - 1},
            hovertemplate="%{y} / %{x}<br>Correlation: %{z:.3f}<extra></extra>",
            colorbar={
                "title": {"text": "Corr.", "font": {"color": theme.INK_MUTED}},
                "tickformat": ".1f",
                "outlinecolor": theme.GRID,
                "tickfont": {"color": theme.INK_MUTED},
            },
            xgap=2,
            ygap=2,
        )
    )
    theme.apply_theme(fig, title=head)
    fig.update_layout(
        height=max(360, 120 + 32 * len(labels)),
        hovermode="closest",
        yaxis={"autorange": "reversed"},
    )
    return fig


def drawdown_chart(
    drawdown: pd.Series,
    *,
    title: str | None = None,
    max_drawdown: float | None = None,
) -> go.Figure:
    """Underwater-Chart der Drawdown-Reihe.

    Args:
        drawdown: Drawdown-Reihe, dezimal und negativ (-0.2 = 20 Prozent
            unter dem bisherigen Hoch), mit UTC-Index.
        title: Ueberschrift.
        max_drawdown: Optionaler Max Drawdown, dezimal und negativ. Wird
            als Referenzlinie markiert.

    Returns:
        Flaechenchart unterhalb der Nulllinie. Bei leerer Reihe eine
        Hinweis-Figure.
    """
    head = title if title is not None else "Drawdown (Underwater)"
    if drawdown.empty or drawdown.dropna().empty:
        return theme.empty_figure("No drawdown data", title=head)

    fig = go.Figure(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            name="Drawdown",
            mode="lines",
            line={"color": theme.DOWN, "width": 2},
            fill="tozeroy",
            fillcolor=theme.DOWN_FILL,
            hovertemplate=theme.hover_template(label="Drawdown", value_format=".2%"),
        )
    )
    if max_drawdown is not None:
        label = theme.format_value(max_drawdown * 100, value_format=".2f", unit="%")
        fig.add_hline(
            y=max_drawdown,
            line={"color": theme.AMBER, "width": 1, "dash": "dash"},
            annotation={
                "text": f"Max DD {label}",
                "font": {"color": theme.AMBER, "family": theme.FONT_FAMILY},
            },
        )
    theme.apply_theme(fig, title=head)
    theme.add_crosshair(fig)
    theme.style_axes(fig, x_title="Date (UTC)", y_title="Drawdown (decimal, negative)")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=340)
    return fig


def return_distribution(
    returns: pd.Series,
    risk: RiskMetrics | None = None,
    *,
    bins: int = 60,
    title: str | None = None,
) -> go.Figure:
    """Histogramm der Renditeverteilung mit VaR- und CVaR-Markierung.

    Args:
        returns: Perioden-Renditen, dezimal.
        risk: Fertig berechnete Risikokennzahlen. ``var_historical`` und
            ``cvar_historical`` sind dezimal und negativ und werden als
            vertikale Linien eingezeichnet. Fehlende Werte werden schlicht
            nicht markiert, nie als 0 gezeichnet.
        bins: Anzahl der Histogramm-Klassen.
        title: Ueberschrift.

    Returns:
        Histogramm mit Markierungen. Bei leerer Reihe eine Hinweis-Figure.
    """
    head = title if title is not None else "Return distribution"
    clean = returns.dropna()
    if clean.empty:
        return theme.empty_figure("No return data", title=head)

    fig = go.Figure(
        go.Histogram(
            x=clean,
            nbinsx=bins,
            name="Returns",
            marker={"color": theme.UP_FILL, "line": {"color": theme.UP, "width": 1}},
            hovertemplate="Return %{x:.2%}<br>Count: %{y}<extra></extra>",
        )
    )

    if risk is not None:
        marks = (
            ("VaR", risk.var_historical, theme.AMBER, "dash"),
            ("CVaR", risk.cvar_historical, theme.DOWN, "dot"),
        )
        for label, value, color, dash in marks:
            if value is None:
                continue
            fig.add_vline(
                x=value,
                line={"color": color, "width": 2, "dash": dash},
                annotation={
                    "text": f"{label} {risk.confidence:.0%}: {value:.2%}",
                    "font": {"color": color, "family": theme.FONT_FAMILY},
                },
            )

    theme.apply_theme(fig, title=head)
    theme.style_axes(fig, x_title="Return (decimal)", y_title="Frequency (count)")
    fig.update_xaxes(tickformat=".1%")
    fig.update_layout(height=380, hovermode="closest", bargap=0.02)
    return fig


def rolling_volatility(
    volatility: pd.Series,
    *,
    title: str | None = None,
    window_label: str | None = None,
) -> go.Figure:
    """Rollierende Volatilitaet ueber die Zeit.

    Args:
        volatility: Rollierende Volatilitaet, annualisiert und dezimal,
            mit UTC-Index.
        title: Ueberschrift.
        window_label: Beschreibung des Fensters fuer die Legende, etwa
            ``"63 Tage"``.

    Returns:
        Linienchart. Bei leerer Reihe eine Hinweis-Figure.
    """
    head = title if title is not None else "Rolling volatility"
    if volatility.dropna().empty:
        return theme.empty_figure("No volatility data", title=head)

    name = "Volatility" if window_label is None else f"Volatility ({window_label})"
    fig = go.Figure(
        go.Scatter(
            x=volatility.index,
            y=volatility,
            name=name,
            line={"color": theme.CYAN, "width": 2},
            hovertemplate=theme.hover_template(label=name, value_format=".2%"),
        )
    )
    theme.apply_theme(fig, title=head)
    theme.add_crosshair(fig)
    theme.style_axes(fig, x_title="Date (UTC)", y_title="Volatility (annualised, decimal)")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=340)
    return fig


def rolling_beta(
    beta: pd.Series,
    *,
    benchmark: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Rollierendes Beta gegen die Benchmark.

    Die Referenzlinie bei 1.0 macht sichtbar, wann das Instrument staerker
    oder schwaecher schwankt als seine Benchmark.

    Args:
        beta: Rollierendes Beta, dimensionslos, mit UTC-Index.
        benchmark: Symbol der Benchmark fuer die Beschriftung.
        title: Ueberschrift.

    Returns:
        Linienchart mit Referenzlinie bei 1.0. Bei leerer Reihe eine
        Hinweis-Figure.
    """
    head = title if title is not None else "Rolling beta"
    if beta.dropna().empty:
        return theme.empty_figure("No beta data", title=head)

    name = "Beta" if benchmark is None else f"Beta vs {benchmark}"
    fig = go.Figure(
        go.Scatter(
            x=beta.index,
            y=beta,
            name=name,
            line={"color": theme.AMBER, "width": 2},
            hovertemplate=theme.hover_template(label=name, value_format=".2f"),
        )
    )
    fig.add_hline(
        y=1.0,
        line={"color": theme.AXIS_LINE, "width": 1, "dash": "dot"},
        annotation={
            "text": "Beta = 1",
            "font": {"color": theme.INK_MUTED, "family": theme.FONT_FAMILY},
        },
    )
    theme.apply_theme(fig, title=head)
    theme.add_crosshair(fig)
    theme.style_axes(fig, x_title="Date (UTC)", y_title="Beta (dimensionless)")
    fig.update_layout(height=340)
    return fig
