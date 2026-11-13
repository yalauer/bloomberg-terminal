"""Chart-Factories fuer Block 1: Kurse, Overlays und Vergleich.

Architekturregel 3: Dieses Modul rechnet nicht. Insbesondere werden
technische Indikatoren **nicht** hier berechnet - ``analytics/technicals.py``
liefert sie in Phase 2 als fertige ``pd.Series`` beziehungsweise
``pd.DataFrame`` und reicht sie ueber :class:`TechnicalOverlays` herein.

Erlaubt sind ausschliesslich darstellungsbezogene Transformationen:
Normierung auf Basis 100 im Vergleichschart und logarithmische Achsen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from btlite.models.common import BOLLINGER_COLUMNS, MACD_COLUMNS
from btlite.viz import theme

if TYPE_CHECKING:
    import pandas as pd

    from btlite.models.market import PriceComparison, PriceSeries

# Die Spaltennamen sind Vertrag zwischen analytics/ und viz/ und stehen
# deshalb in models/common.py, nicht hier (Architekturregel 1).
BOLLINGER_UPPER, BOLLINGER_MIDDLE, BOLLINGER_LOWER = BOLLINGER_COLUMNS
MACD_LINE, MACD_SIGNAL, MACD_HISTOGRAM = MACD_COLUMNS

#: Ueblich markierte Schwellen des RSI.
RSI_OVERBOUGHT: float = 70.0
RSI_OVERSOLD: float = 30.0


@dataclass(slots=True)
class TechnicalOverlays:
    """Fertig berechnete technische Indikatoren zur Darstellung.

    Das ist der Vertrag zwischen ``analytics/technicals.py`` (Phase 2) und
    dieser Schicht. Alle Reihen tragen denselben tz-aware UTC-Index wie die
    zugehoerige :class:`~btlite.models.market.PriceSeries`; fehlende Werte
    am Rand sind ``NaN`` und werden als Luecke gezeichnet, nie als Null.

    Attributes:
        sma: Gleitende Durchschnitte, Schluessel ist das Label (etwa
            ``"SMA 50"``), Wert die Reihe in Kurswaehrung.
        ema: Exponentielle Durchschnitte, gleiche Struktur wie ``sma``.
        bollinger: DataFrame mit den Spalten ``upper``, ``middle`` und
            ``lower``, in Kurswaehrung.
        rsi: Relative Strength Index, Skala 0 bis 100. Eigener Subplot.
        macd: DataFrame mit den Spalten ``macd``, ``signal`` und
            ``histogram``, in Kurswaehrung. Eigener Subplot.
    """

    sma: dict[str, pd.Series] = field(default_factory=dict)
    ema: dict[str, pd.Series] = field(default_factory=dict)
    bollinger: pd.DataFrame | None = None
    rsi: pd.Series | None = None
    macd: pd.DataFrame | None = None

    @property
    def has_rsi(self) -> bool:
        """Ob ein RSI-Subplot noetig ist.

        Returns:
            True wenn eine nicht-leere RSI-Reihe vorliegt.
        """
        return self.rsi is not None and not self.rsi.empty

    @property
    def has_macd(self) -> bool:
        """Ob ein MACD-Subplot noetig ist.

        Returns:
            True wenn ein nicht-leerer MACD-Rahmen vorliegt.
        """
        return self.macd is not None and not self.macd.empty


def _is_empty(series: PriceSeries) -> bool:
    """Prueft, ob eine Kursreihe keine darstellbaren Zeilen hat.

    Args:
        series: Zu pruefende Kursreihe.

    Returns:
        True wenn der Payload leer ist.
    """
    return series.data.empty


def _volume_colors(data: pd.DataFrame) -> list[str]:
    """Faerbt Volumenbalken nach der Richtung der jeweiligen Kerze.

    Args:
        data: OHLCV-Payload.

    Returns:
        Liste von Farbwerten, ein Eintrag je Zeile.
    """
    up = data["close"] >= data["open"]
    return [theme.UP_FILL if flag else theme.DOWN_FILL for flag in up]


def _add_overlays(fig: go.Figure, overlays: TechnicalOverlays, *, row: int) -> None:
    """Zeichnet Kurs-Overlays in den Hauptplot.

    Args:
        fig: Zielfigur.
        overlays: Fertig berechnete Indikatoren.
        row: Subplot-Zeile des Kurscharts, einsbasiert.
    """
    if overlays.bollinger is not None and not overlays.bollinger.empty:
        band = overlays.bollinger
        if BOLLINGER_UPPER in band.columns and BOLLINGER_LOWER in band.columns:
            fig.add_trace(
                go.Scatter(
                    x=band.index,
                    y=band[BOLLINGER_UPPER],
                    name="Bollinger upper",
                    line={"color": theme.AMBER_DIM, "width": 1, "dash": "dot"},
                    hovertemplate=theme.hover_template(label="BB upper"),
                    legendgroup="bollinger",
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=band.index,
                    y=band[BOLLINGER_LOWER],
                    name="Bollinger lower",
                    line={"color": theme.AMBER_DIM, "width": 1, "dash": "dot"},
                    fill="tonexty",
                    fillcolor="rgba(179,111,22,0.10)",
                    hovertemplate=theme.hover_template(label="BB lower"),
                    legendgroup="bollinger",
                ),
                row=row,
                col=1,
            )
        if BOLLINGER_MIDDLE in band.columns:
            fig.add_trace(
                go.Scatter(
                    x=band.index,
                    y=band[BOLLINGER_MIDDLE],
                    name="Bollinger middle",
                    line={"color": theme.AMBER_DIM, "width": 1},
                    hovertemplate=theme.hover_template(label="BB middle"),
                    legendgroup="bollinger",
                ),
                row=row,
                col=1,
            )

    slot = 0
    for label, values in overlays.sma.items():
        fig.add_trace(
            go.Scatter(
                x=values.index,
                y=values,
                name=label,
                line={"color": theme.series_color(slot), "width": 2},
                hovertemplate=theme.hover_template(label=label),
            ),
            row=row,
            col=1,
        )
        slot += 1
    for label, values in overlays.ema.items():
        fig.add_trace(
            go.Scatter(
                x=values.index,
                y=values,
                name=label,
                line={"color": theme.series_color(slot), "width": 2, "dash": "dash"},
                hovertemplate=theme.hover_template(label=label),
            ),
            row=row,
            col=1,
        )
        slot += 1


def _add_rsi(fig: go.Figure, rsi: pd.Series, *, row: int) -> None:
    """Zeichnet den RSI-Subplot inklusive der ueblichen Schwellen.

    Args:
        fig: Zielfigur.
        rsi: RSI-Reihe, Skala 0 bis 100.
        row: Subplot-Zeile, einsbasiert.
    """
    fig.add_trace(
        go.Scatter(
            x=rsi.index,
            y=rsi,
            name="RSI",
            line={"color": theme.CYAN, "width": 2},
            hovertemplate=theme.hover_template(label="RSI", value_format=".1f"),
        ),
        row=row,
        col=1,
    )
    for level, color in ((RSI_OVERBOUGHT, theme.DOWN), (RSI_OVERSOLD, theme.UP)):
        fig.add_hline(
            y=level,
            line={"color": color, "width": 1, "dash": "dot"},
            row=row,
            col=1,
        )
    fig.update_yaxes(range=[0, 100], title_text="RSI (0-100)", row=row, col=1)


def _add_macd(fig: go.Figure, macd: pd.DataFrame, *, row: int) -> None:
    """Zeichnet den MACD-Subplot.

    Args:
        fig: Zielfigur.
        macd: Rahmen mit den Spalten ``macd``, ``signal`` und ``histogram``.
        row: Subplot-Zeile, einsbasiert.
    """
    if MACD_HISTOGRAM in macd.columns:
        hist = macd[MACD_HISTOGRAM]
        fig.add_trace(
            go.Bar(
                x=macd.index,
                y=hist,
                name="MACD histogram",
                marker={
                    "color": [
                        theme.UP_FILL if bool(v >= 0) else theme.DOWN_FILL for v in hist.fillna(0.0)
                    ]
                },
                hovertemplate=theme.hover_template(label="Histogram", value_format=".3f"),
            ),
            row=row,
            col=1,
        )
    if MACD_LINE in macd.columns:
        fig.add_trace(
            go.Scatter(
                x=macd.index,
                y=macd[MACD_LINE],
                name="MACD",
                line={"color": theme.CYAN, "width": 2},
                hovertemplate=theme.hover_template(label="MACD", value_format=".3f"),
            ),
            row=row,
            col=1,
        )
    if MACD_SIGNAL in macd.columns:
        fig.add_trace(
            go.Scatter(
                x=macd.index,
                y=macd[MACD_SIGNAL],
                name="MACD Signal",
                line={"color": theme.AMBER, "width": 2, "dash": "dash"},
                hovertemplate=theme.hover_template(label="Signal", value_format=".3f"),
            ),
            row=row,
            col=1,
        )
    fig.update_yaxes(title_text="MACD (price currency)", row=row, col=1)


def _price_figure(
    series: PriceSeries,
    overlays: TechnicalOverlays | None,
    *,
    kind: str,
    show_volume: bool,
    log_y: bool,
    title: str | None,
    range_selector: bool,
) -> go.Figure:
    """Gemeinsamer Aufbau von Kerzen-, OHLC- und Linienchart.

    Args:
        series: Kursreihe mit OHLCV-Payload.
        overlays: Optionale, fertig berechnete Indikatoren.
        kind: ``"candlestick"``, ``"ohlc"`` oder ``"line"``.
        show_volume: Volumen-Subplot anzeigen.
        log_y: Logarithmische Preisachse (rein darstellungsbezogen).
        title: Optionale Ueberschrift.
        range_selector: Zeitraum-Schnellwahl anzeigen.

    Returns:
        Fertige Figure. Bei leerem Payload eine Hinweis-Figure.
    """
    head = title if title is not None else f"{series.ticker} - {series.interval.value}"
    if _is_empty(series):
        return theme.empty_figure(f"No price data for {series.ticker}", title=head)

    ov = overlays or TechnicalOverlays()
    data = series.data

    rows = [("price", 0.60 if (show_volume or ov.has_rsi or ov.has_macd) else 1.0)]
    if show_volume:
        rows.append(("volume", 0.16))
    if ov.has_rsi:
        rows.append(("rsi", 0.16))
    if ov.has_macd:
        rows.append(("macd", 0.16))

    total = sum(h for _, h in rows)
    heights = [h / total for _, h in rows]
    index_of = {name: i + 1 for i, (name, _) in enumerate(rows)}

    fig = make_subplots(
        rows=len(rows),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=heights,
    )

    price_row = index_of["price"]
    if kind == "candlestick":
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                name=series.ticker,
                increasing={"line": {"color": theme.UP}, "fillcolor": theme.UP},
                decreasing={"line": {"color": theme.DOWN}, "fillcolor": theme.DOWN},
            ),
            row=price_row,
            col=1,
        )
    elif kind == "ohlc":
        fig.add_trace(
            go.Ohlc(
                x=data.index,
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                name=series.ticker,
                increasing={"line": {"color": theme.UP}},
                decreasing={"line": {"color": theme.DOWN}},
            ),
            row=price_row,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["close"],
                name=series.ticker,
                line={"color": theme.AMBER, "width": 2},
                hovertemplate=theme.hover_template(label=series.ticker, unit=series.currency.value),
            ),
            row=price_row,
            col=1,
        )

    _add_overlays(fig, ov, row=price_row)

    if show_volume:
        vol_row = index_of["volume"]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data["volume"],
                name="Volume",
                marker={"color": _volume_colors(data)},
                hovertemplate=theme.hover_template(
                    label="Volume", value_format=",.0f", unit="shares"
                ),
                showlegend=False,
            ),
            row=vol_row,
            col=1,
        )
        fig.update_yaxes(title_text="Volume (shares)", row=vol_row, col=1)

    if ov.has_rsi and ov.rsi is not None:
        _add_rsi(fig, ov.rsi, row=index_of["rsi"])
    if ov.has_macd and ov.macd is not None:
        _add_macd(fig, ov.macd, row=index_of["macd"])

    theme.apply_theme(fig, title=head)
    theme.add_crosshair(fig)
    fig.update_layout(
        barmode="relative",
        xaxis_rangeslider_visible=False,
        height=520 + 110 * (len(rows) - 1),
    )
    fig.update_yaxes(title_text=f"Price ({series.currency.value})", row=price_row, col=1)
    if log_y:
        fig.update_yaxes(type="log", row=price_row, col=1)
    fig.update_xaxes(title_text="Date (UTC)", row=len(rows), col=1)
    if range_selector:
        theme.add_range_selector(fig, row=1, col=1)
    return fig


def candlestick(
    series: PriceSeries,
    overlays: TechnicalOverlays | None = None,
    *,
    show_volume: bool = True,
    log_y: bool = False,
    title: str | None = None,
    range_selector: bool = True,
) -> go.Figure:
    """Kerzenchart mit Volumen-Subplot und optionalen Indikatoren.

    Args:
        series: Kursreihe. Preise in ``series.currency``, Volumen in Stueck.
        overlays: Fertig berechnete Indikatoren aus ``analytics/technicals``.
            RSI und MACD erhalten je einen eigenen Subplot.
        show_volume: Volumen-Subplot anzeigen.
        log_y: Logarithmische Preisachse. Rein darstellungsbezogen.
        title: Ueberschrift. Default ist ``"<TICKER> - <intervall>"``.
        range_selector: Zeitraum-Schnellwahl 1M/3M/6M/1Y/5Y/MAX anzeigen.

    Returns:
        Figure mit Kerzen, Volumen und den angeforderten Subplots. Bei
        leerem Payload eine Hinweis-Figure statt eines leeren Rahmens.
    """
    return _price_figure(
        series,
        overlays,
        kind="candlestick",
        show_volume=show_volume,
        log_y=log_y,
        title=title,
        range_selector=range_selector,
    )


def ohlc(
    series: PriceSeries,
    overlays: TechnicalOverlays | None = None,
    *,
    show_volume: bool = True,
    log_y: bool = False,
    title: str | None = None,
    range_selector: bool = True,
) -> go.Figure:
    """OHLC-Balkenchart, sonst identisch zu :func:`candlestick`.

    Args:
        series: Kursreihe. Preise in ``series.currency``, Volumen in Stueck.
        overlays: Fertig berechnete Indikatoren.
        show_volume: Volumen-Subplot anzeigen.
        log_y: Logarithmische Preisachse. Rein darstellungsbezogen.
        title: Ueberschrift.
        range_selector: Zeitraum-Schnellwahl anzeigen.

    Returns:
        Figure mit OHLC-Balken. Bei leerem Payload eine Hinweis-Figure.
    """
    return _price_figure(
        series,
        overlays,
        kind="ohlc",
        show_volume=show_volume,
        log_y=log_y,
        title=title,
        range_selector=range_selector,
    )


def line(
    series: PriceSeries,
    overlays: TechnicalOverlays | None = None,
    *,
    show_volume: bool = False,
    log_y: bool = False,
    title: str | None = None,
    range_selector: bool = True,
) -> go.Figure:
    """Linienchart des Schlusskurses.

    Args:
        series: Kursreihe. Preise in ``series.currency``.
        overlays: Fertig berechnete Indikatoren.
        show_volume: Volumen-Subplot anzeigen. Default aus, da der
            Linienchart vor allem dem Ueberblick dient.
        log_y: Logarithmische Preisachse. Rein darstellungsbezogen.
        title: Ueberschrift.
        range_selector: Zeitraum-Schnellwahl anzeigen.

    Returns:
        Figure mit Schlusskurslinie. Bei leerem Payload eine Hinweis-Figure.
    """
    return _price_figure(
        series,
        overlays,
        kind="line",
        show_volume=show_volume,
        log_y=log_y,
        title=title,
        range_selector=range_selector,
    )


def comparison(
    comp: PriceComparison,
    *,
    log_y: bool = False,
    title: str | None = None,
    range_selector: bool = True,
) -> go.Figure:
    """Mehr-Ticker-Vergleich, normiert auf Basis 100.

    Ist ``comp.normalized`` bereits True, wird der Payload unveraendert
    gezeichnet. Andernfalls normiert diese Funktion auf den ersten
    gemeinsamen Beobachtungswert - das ist die einzige hier erlaubte
    Transformation und rein darstellungsbezogen.

    Args:
        comp: Vergleichsmodell mit einer Spalte je Ticker.
        log_y: Logarithmische Y-Achse. Sinnvoll bei langen Zeitraeumen.
        title: Ueberschrift.
        range_selector: Zeitraum-Schnellwahl anzeigen.

    Returns:
        Figure mit einer Linie je Ticker, Y-Achse indexiert auf 100. Bei
        leerem Payload eine Hinweis-Figure.
    """
    head = title if title is not None else f"Comparison: {', '.join(comp.tickers)}"
    if comp.data.empty or not comp.tickers:
        return theme.empty_figure("No comparison data", title=head)

    frame = comp.data[list(comp.tickers)]
    if not comp.normalized:
        base = frame.ffill().bfill().iloc[0]
        base = base.where(base != 0)
        frame = frame.div(base) * 100.0

    fig = go.Figure()
    drawn = 0
    for slot, ticker in enumerate(comp.tickers):
        col = frame[ticker]
        if col.dropna().empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=col,
                name=ticker,
                line={"color": theme.series_color(slot), "width": 2},
                hovertemplate=theme.hover_template(label=ticker, value_format=".1f"),
            )
        )
        drawn += 1

    if drawn == 0:
        return theme.empty_figure("No comparison data", title=head)

    fig.add_hline(y=100.0, line={"color": theme.AXIS_LINE, "width": 1, "dash": "dot"})
    theme.apply_theme(fig, title=head)
    theme.add_crosshair(fig)
    theme.style_axes(
        fig,
        x_title="Date (UTC)",
        y_title="Indexed (base 100)",
        log_y=log_y,
    )
    fig.update_layout(height=520)
    if range_selector:
        theme.add_range_selector(fig)
    return fig
