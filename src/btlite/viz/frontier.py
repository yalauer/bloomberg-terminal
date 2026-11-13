"""Chart-Factories fuer Block 5: Effizienzgrenze und Gewichte.

Architekturregel 3: Dieses Modul rechnet nicht. Die Stuetzstellen der
Effizienzgrenze und die optimierten Gewichte kommen fertig aus
``portfolio/optimizer.py``.

Einheiten: erwartete Renditen und Volatilitaeten annualisiert und dezimal
(0.12 = 12 Prozent). Gewichte dezimal, Summe 1.0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from btlite.viz import theme

if TYPE_CHECKING:
    import pandas as pd

    from btlite.models.portfolio import EfficientFrontier, PortfolioResult

#: Spaltennamen der Frontier-Stuetzstellen, siehe ``EfficientFrontier``.
VOLATILITY_COLUMN: str = "volatility"
EXPECTED_RETURN_COLUMN: str = "expected_return"
SHARPE_COLUMN: str = "sharpe"

#: Praefix der optionalen Gewichtsspalten je Ticker.
WEIGHT_PREFIX: str = "w_"


def efficient_frontier(
    frontier: EfficientFrontier,
    *,
    assets: pd.DataFrame | None = None,
    optimal: tuple[float, float] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Effizienzgrenze, eingefaerbt nach Sharpe Ratio.

    Args:
        frontier: Stuetzstellen mit den Spalten ``volatility``,
            ``expected_return`` und ``sharpe``. Erste beide annualisiert
            und dezimal.
        assets: Optionale Einzelwerte als Streupunkte. Erwartet dieselben
            Spalten ``volatility`` und ``expected_return``, Index ist der
            Ticker. Fertig berechnet aus ``portfolio/optimizer.py``.
        optimal: Optionaler Optimalpunkt als ``(volatilitaet, rendite)``,
            beide annualisiert und dezimal. Ist er None, wird ersatzweise
            die Stuetzstelle mit der hoechsten Sharpe Ratio markiert.
        title: Ueberschrift.

    Returns:
        Streu-/Linienchart der Grenze. Bei leerem Payload eine
        Hinweis-Figure.
    """
    head = title if title is not None else "Efficient frontier"
    if frontier.data.empty:
        return theme.empty_figure("No frontier data", title=head)

    data = frontier.data.sort_values(VOLATILITY_COLUMN)
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data[VOLATILITY_COLUMN],
            y=data[EXPECTED_RETURN_COLUMN],
            mode="lines",
            name="Efficient frontier",
            line={"color": theme.AMBER_DIM, "width": 2},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data[VOLATILITY_COLUMN],
            y=data[EXPECTED_RETURN_COLUMN],
            mode="markers",
            name="Sample points",
            marker={
                "size": 9,
                "color": data[SHARPE_COLUMN],
                "colorscale": theme.sequential_scale(),
                "showscale": True,
                "line": {"color": theme.SURFACE, "width": 1},
                "colorbar": {
                    "title": {"text": "Sharpe", "font": {"color": theme.INK_MUTED}},
                    "outlinecolor": theme.GRID,
                    "tickfont": {"color": theme.INK_MUTED},
                },
            },
            customdata=data[SHARPE_COLUMN],
            hovertemplate=(
                "Volatility: %{x:.2%}<br>Return: %{y:.2%}"
                "<br>Sharpe: %{customdata:.2f}<extra></extra>"
            ),
        )
    )

    if assets is not None and not assets.empty:
        fig.add_trace(
            go.Scatter(
                x=assets[VOLATILITY_COLUMN],
                y=assets[EXPECTED_RETURN_COLUMN],
                mode="markers+text",
                name="Individual assets",
                text=[str(i) for i in assets.index],
                textposition="top center",
                textfont={"color": theme.INK_MUTED, "family": theme.FONT_FAMILY},
                marker={
                    "size": 10,
                    "color": theme.CYAN,
                    "symbol": "diamond",
                    "line": {"color": theme.SURFACE, "width": 1},
                },
                hovertemplate=(
                    "%{text}<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>"
                ),
            )
        )

    point = optimal
    if point is None and SHARPE_COLUMN in data.columns and not data[SHARPE_COLUMN].dropna().empty:
        # Positionsbasiert statt ueber .loc: ein doppelter Indexwert wuerde
        # dort einen DataFrame liefern statt einer Zeile. ``argmax`` der
        # Series ignoriert NaN, anders als das NumPy-Pendant.
        best_pos = int(data[SHARPE_COLUMN].argmax())
        point = (
            float(data[VOLATILITY_COLUMN].iloc[best_pos]),
            float(data[EXPECTED_RETURN_COLUMN].iloc[best_pos]),
        )
    if point is not None:
        fig.add_trace(
            go.Scatter(
                x=[point[0]],
                y=[point[1]],
                mode="markers",
                name="Optimum",
                marker={
                    "size": 17,
                    "color": theme.AMBER,
                    "symbol": "star",
                    "line": {"color": theme.SURFACE, "width": 1},
                },
                hovertemplate=(
                    "Optimum<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>"
                ),
            )
        )

    theme.apply_theme(fig, title=head)
    theme.style_axes(
        fig,
        x_title="Volatility (annualised, decimal)",
        y_title="Expected return (annualised, decimal)",
    )
    fig.update_xaxes(tickformat=".1%")
    fig.update_yaxes(tickformat=".1%")
    fig.update_layout(height=520, hovermode="closest")
    return fig


def weights_bar(
    result: PortfolioResult,
    *,
    title: str | None = None,
    min_weight: float = 0.0,
) -> go.Figure:
    """Zielgewichte als horizontales Balkendiagramm.

    Balken bevorzugt gegenueber dem Donut, sobald es auf den Vergleich der
    Groessen ankommt: Laengen sind praeziser ablesbar als Kreissegmente.

    Args:
        result: Optimierungsergebnis mit den Positionen.
        title: Ueberschrift.
        min_weight: Positionen mit kleinerem Betrag werden ausgeblendet,
            dezimal. Default 0.0 zeigt alle.

    Returns:
        Balkendiagramm der Gewichte. Ohne Positionen eine Hinweis-Figure.
    """
    head = title if title is not None else "Portfolio weights"
    positions = [p for p in result.positions if abs(p.weight) >= min_weight]
    if not positions:
        return theme.empty_figure("No positions", title=head)

    ordered = sorted(positions, key=lambda p: p.weight)
    tickers = [p.ticker for p in ordered]
    weights = [p.weight for p in ordered]
    # Negative Gewichte (Shorts) tragen die Gegenfarbe, damit die Richtung
    # ohne Vorzeichenlesen erkennbar ist.
    colors = [theme.UP if w >= 0 else theme.DOWN for w in weights]

    fig = go.Figure(
        go.Bar(
            x=weights,
            y=tickers,
            orientation="h",
            marker={"color": colors},
            text=[f"{w:.2%}" for w in weights],
            textposition="outside",
            textfont={"color": theme.INK, "family": theme.FONT_FAMILY},
            hovertemplate="%{y}: %{x:.2%}<extra></extra>",
            showlegend=False,
        )
    )
    theme.apply_theme(fig, title=head)
    theme.style_axes(fig, x_title="Weight (decimal)", y_title="Ticker")
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(height=max(300, 120 + 30 * len(ordered)), hovermode="closest")
    return fig


def weights_donut(
    result: PortfolioResult,
    *,
    title: str | None = None,
    min_weight: float = 0.0,
) -> go.Figure:
    """Zielgewichte als Donut.

    Nur fuer reine Long-Portfolios sinnvoll: negative Gewichte lassen sich
    in einem Kreis nicht darstellen und werden hier deshalb ueber ihren
    Betrag gezeigt, was die Richtung verschweigt. Bei Shorts ist
    :func:`weights_bar` die richtige Wahl.

    Args:
        result: Optimierungsergebnis mit den Positionen.
        title: Ueberschrift.
        min_weight: Positionen mit kleinerem Betrag werden ausgeblendet,
            dezimal.

    Returns:
        Donut der Gewichte. Ohne Positionen eine Hinweis-Figure.
    """
    head = title if title is not None else "Portfolio weights"
    positions = [p for p in result.positions if abs(p.weight) >= min_weight]
    if not positions:
        return theme.empty_figure("No positions", title=head)

    ordered = sorted(positions, key=lambda p: abs(p.weight), reverse=True)
    fig = go.Figure(
        go.Pie(
            labels=[p.ticker for p in ordered],
            values=[abs(p.weight) for p in ordered],
            hole=0.55,
            marker={
                "colors": [theme.series_color(i) for i in range(len(ordered))],
                "line": {"color": theme.SURFACE, "width": 2},
            },
            textinfo="label+percent",
            textfont={"family": theme.FONT_FAMILY, "color": theme.INK},
            hovertemplate="%{label}: %{percent}<extra></extra>",
            sort=False,
        )
    )
    theme.apply_theme(fig, title=head)
    fig.update_layout(height=460, hovermode="closest")
    return fig
