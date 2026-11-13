"""Seite GP - Kursverlauf, Overlays und Mehr-Ticker-Vergleich.

Die Seite verdrahtet ``analytics.technicals.build_overlays`` mit den
Chart-Factories aus ``viz.price``. Sie berechnet keinen Indikator selbst.

Die Setup-Checkliste ("Chart setups") verdrahtet
``analytics.setups.setup_report`` mit ``viz.setups``. Sie zeigt je Setup
"met", "not met" oder "insufficient data" samt Messwerten - bewusst ohne
Gesamturteil, Score oder Ampel. Keine Anlageempfehlung.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final

from dash import dcc, html

from btlite.core.errors import BTLiteError
from btlite.models.common import INTRADAY_INTERVALS, Interval
from btlite.models.setups import MeasurementUnit, SetupStatus
from btlite.ui import formatting as fmt
from btlite.ui.layouts import components as c
from btlite.ui.state import PERIOD_CHOICES

if TYPE_CHECKING:
    from btlite.models.setups import Measurement, SetupReport
    from btlite.ui.state import AppState

#: Route dieser Seite.
PATH = "/gp"

#: Element-IDs der Bedienelemente.
TICKER_STORE = "gp-tickers"
PERIOD_INPUT = "gp-period"
RESOLUTION_INPUT = "gp-resolution"
CHART_KIND_INPUT = "gp-chart-kind"
OVERLAY_INPUT = "gp-overlays"
CHART_OUTPUT = "gp-chart"

#: Waehlbare Chart-Arten.
CHART_KINDS: Final[tuple[tuple[str, str], ...]] = (
    ("Candlestick", "candlestick"),
    ("OHLC", "ohlc"),
    ("Line", "line"),
)

#: Waehlbare Overlays. Wert ist der interne Schluessel.
OVERLAY_CHOICES: Final[tuple[tuple[str, str], ...]] = (
    ("SMA 20/50/200", "sma"),
    ("EMA 12/26", "ema"),
    ("Bollinger", "bollinger"),
    ("RSI", "rsi"),
    ("MACD", "macd"),
    ("Chart setups", "setups"),
)

#: Vorbelegung der Overlay-Auswahl.
DEFAULT_OVERLAYS: Final[tuple[str, ...]] = ("sma", "setups")

#: Zeitraum der Kursreihe, auf der die Setup-Checkliste rechnet. Unabhaengig
#: vom dargestellten Zeitraum, damit die Checkliste beim Umschalten des
#: Zeitraums gleich bleibt; der Ausbruch braucht 253 Handelstage.
SETUP_HISTORY_PERIOD: Final[str] = "2y"

#: Hinweissatz unter der Checkliste. Pflicht, siehe Nicht-Ziele.
SETUP_DISCLAIMER: Final[str] = "Measurable chart conditions, not a recommendation."

#: Anzeige des Status: Text plus Symbol, die Farbe ist nur Beiwerk.
_STATUS_TEXT: Final[dict[SetupStatus, str]] = {
    SetupStatus.MET: "\u2713 met",
    SetupStatus.NOT_MET: "\u2013 not met",
    SetupStatus.INSUFFICIENT_DATA: "? insufficient data",
}
_STATUS_CLASS: Final[dict[SetupStatus, str]] = {
    SetupStatus.MET: "bt-up",
    SetupStatus.NOT_MET: "",
    SetupStatus.INSUFFICIENT_DATA: "bt-na",
}

#: Fensterlaengen der Gleitenden Durchschnitte.
_SMA_WINDOWS: Final[tuple[int, ...]] = (20, 50, 200)
_EMA_WINDOWS: Final[tuple[int, ...]] = (12, 26)

#: Vorbelegter Zeitraum.
DEFAULT_PERIOD: Final[str] = "1y"

#: Zeitraeume der Seite: die allgemeinen plus ein Tag und eine Woche.
#: "5d" statt "1wk": yfinance kennt als Zeitraum nur Handelstage, fuenf
#: davon sind eine Woche.
GP_PERIOD_CHOICES: Final[tuple[tuple[str, str], ...]] = (
    ("1D", "1d"),
    ("1W", "5d"),
    *PERIOD_CHOICES,
)

#: Waehlbare Kerzen-Aufloesungen. Wert ist der yfinance-String des
#: Intervalls. Welche davon zu einem Zeitraum passen, bestimmt
#: :data:`btlite.analytics.series.CHART_INTERVALS`; vorgewaehlt wird immer
#: die kleinste. Wochenkerzen entstehen lokal aus den Tageskerzen.
RESOLUTION_CHOICES: Final[tuple[tuple[str, str], ...]] = (
    ("1 min", Interval.M1.value),
    ("5 min", Interval.M5.value),
    ("15 min", Interval.M15.value),
    ("1 hour", Interval.H1.value),
    ("1 day", Interval.D1.value),
    ("1 week", Interval.W1.value),
)

#: Leer heisst: die kleinstmoegliche Aufloesung des Zeitraums.
DEFAULT_RESOLUTION: Final[str] = ""

#: Kerzenlaenge je Intraday-Intervall, fuer das Ausblenden der Pausen.
_INTRADAY_STEP: Final[dict[Interval, timedelta]] = {
    Interval.M1: timedelta(minutes=1),
    Interval.M5: timedelta(minutes=5),
    Interval.M15: timedelta(minutes=15),
    Interval.M30: timedelta(minutes=30),
    Interval.H1: timedelta(hours=1),
}

#: Bezeichnung der Kerzen im Charttitel.
_CANDLE_LABEL: Final[dict[Interval, str]] = {
    Interval.M1: "1-minute",
    Interval.M5: "5-minute",
    Interval.M15: "15-minute",
    Interval.M30: "30-minute",
    Interval.H1: "hourly",
    Interval.D1: "daily",
    Interval.W1: "weekly",
}


def layout(state: AppState, tickers: list[str], args: tuple[str, ...] = ()) -> html.Div:
    """Baut die Seite GP.

    Args:
        state: UI-Zustand mit Provider und Puffern.
        tickers: Symbole aus der Kommandozeile.
        args: Weitere Argumente. Ein Argument in yfinance-Notation setzt
            den Zeitraum, etwa ``AAPL GP 5y`` oder ``AAPL GP 1d``; ein
            Intervall wie ``1h`` oder ``1wk`` die Aufloesung, etwa
            ``AAPL GP 1y 1wk``. Ohne Angabe gilt die kleinstmoegliche.

    Returns:
        Seiteninhalt mit Bedienelementen und Chartbereich.
    """
    symbols = tickers or state.default_tickers
    period = _period_from_args(args)
    intraday = not state.is_offline
    resolution = effective_interval(
        period, _resolution_from_args(args), intraday_available=intraday
    ).value

    return html.Div(
        [
            c.page_header(
                f"{', '.join(symbols)} - Price history",
                "Candlestick chart with volume, overlays and multi-ticker comparison",
            ),
            dcc.Store(id=TICKER_STORE, data=symbols),
            c.controls(
                c.labeled(
                    "Period",
                    dcc.RadioItems(
                        id=PERIOD_INPUT,
                        options=[{"label": lab, "value": val} for lab, val in GP_PERIOD_CHOICES],
                        value=period,
                        inline=True,
                        className="bt-radio",
                    ),
                ),
                c.labeled(
                    "Resolution",
                    dcc.RadioItems(
                        id=RESOLUTION_INPUT,
                        options=resolution_options(period, intraday_available=intraday),
                        value=resolution,
                        inline=True,
                        className="bt-radio",
                    ),
                ),
                c.labeled(
                    "Chart type",
                    dcc.RadioItems(
                        id=CHART_KIND_INPUT,
                        options=[{"label": lab, "value": val} for lab, val in CHART_KINDS],
                        value="candlestick",
                        inline=True,
                        className="bt-radio",
                    ),
                ),
                c.labeled(
                    "Overlays",
                    dcc.Checklist(
                        id=OVERLAY_INPUT,
                        options=[{"label": lab, "value": val} for lab, val in OVERLAY_CHOICES],
                        value=list(DEFAULT_OVERLAYS),
                        inline=True,
                        className="bt-checklist",
                    ),
                ),
            ),
            c.loading(
                html.Div(
                    id=CHART_OUTPUT,
                    children=render_charts(state, symbols, period, resolution=resolution),
                )
            ),
        ],
        className="bt-page bt-page-gp",
    )


def _period_from_args(args: tuple[str, ...]) -> str:
    """Liest den Zeitraum aus den Kommandozeilen-Argumenten.

    Args:
        args: Argumente hinter dem Mnemonic.

    Returns:
        Zeitraum in yfinance-Notation, sonst :data:`DEFAULT_PERIOD`.
    """
    valid = {value for _, value in GP_PERIOD_CHOICES}
    for arg in args:
        candidate = arg.strip().lower()
        if candidate in valid:
            return candidate
    return DEFAULT_PERIOD


def _resolution_from_args(args: tuple[str, ...]) -> str:
    """Liest die Aufloesung aus den Kommandozeilen-Argumenten.

    Args:
        args: Argumente hinter dem Mnemonic.

    Returns:
        Ein Intervall wie ``"1h"`` oder ``"1wk"``, sonst
        :data:`DEFAULT_RESOLUTION`. ``1d`` und ``1mo`` sind hier keine
        Aufloesung: beide sind bereits Zeitraeume.
    """
    periods = {value for _, value in GP_PERIOD_CHOICES}
    valid = {value for _, value in RESOLUTION_CHOICES} - periods
    for arg in args:
        candidate = arg.strip().lower()
        if candidate in valid:
            return candidate
    return DEFAULT_RESOLUTION


def effective_interval(
    period: str, resolution: str | None, *, intraday_available: bool = True
) -> Interval:
    """Bestimmt die Aufloesung, mit der tatsaechlich gezeichnet wird.

    Args:
        period: Zeitraum in yfinance-Notation.
        resolution: Gewuenschte Aufloesung, oder leer fuer die kleinste.
        intraday_available: False ohne Live-Quelle (nur Tagesdaten).

    Returns:
        Die gewuenschte Aufloesung, wenn sie zum Zeitraum passt, sonst die
        kleinstmoegliche. Eine unpassende Kombination - etwa 1 Minute fuer
        ein Jahr - liefert die Quelle nicht; sie wird deshalb gar nicht
        erst angefragt.
    """
    from btlite.analytics.series import allowed_intervals

    allowed = allowed_intervals(period, intraday_available=intraday_available)
    valid = {value for _, value in RESOLUTION_CHOICES}
    if resolution in valid and Interval(resolution) in allowed:
        return Interval(resolution)
    return allowed[0]


def resolution_options(period: str, *, intraday_available: bool = True) -> list[dict[str, Any]]:
    """Baut die Auswahl der Aufloesungen fuer einen Zeitraum.

    Nicht passende Aufloesungen bleiben sichtbar, aber gesperrt: so ist
    erkennbar, dass es sie gibt und warum sie hier fehlen.

    Args:
        period: Zeitraum in yfinance-Notation.
        intraday_available: False ohne Live-Quelle.

    Returns:
        Optionen fuer ``dcc.RadioItems``.
    """
    from btlite.analytics.series import allowed_intervals

    allowed = {i.value for i in allowed_intervals(period, intraday_available=intraday_available)}
    return [
        {"label": label, "value": value, "disabled": value not in allowed}
        for label, value in RESOLUTION_CHOICES
    ]


def _series_for(state: AppState, symbol: str, period: str, interval: Interval) -> Any:
    """Holt die Kursreihe in der gewaehlten Aufloesung.

    Intraday-Kerzen kommen direkt von der Quelle; Tages- und Wochenkerzen
    aus der Tagesreihe, Wochen lokal verdichtet.

    Args:
        state: UI-Zustand.
        symbol: Kanonisches Symbol.
        period: Zeitraum in yfinance-Notation.
        interval: Gewaehlte Aufloesung.

    Returns:
        Kursreihe in der gewaehlten Aufloesung.

    Raises:
        BTLiteError: Wenn die Quelle nicht liefert.
    """
    from btlite.analytics.series import resample_ohlcv

    if interval in INTRADAY_INTERVALS:
        return state.history(symbol, period=period, interval=interval)
    daily = state.history(symbol, period=period, interval=Interval.D1)
    return resample_ohlcv(daily, interval)


def _intraday_display(figure: Any, interval: Interval, timezone: str) -> Any:
    """Richtet einen Intraday-Chart fuer die Anzeige ein.

    Args:
        figure: Fertige Figure mit UTC-Zeitachse.
        interval: Intraday-Intervall.
        timezone: Anzeigezeitzone.

    Returns:
        Dieselbe Figure, Achse in Ortszeit, Pausen ausgeblendet.
    """
    from btlite.viz.intraday import hide_session_gaps, localize_time_axis

    localize_time_axis(figure, timezone)
    return hide_session_gaps(figure, _INTRADAY_STEP[interval])


def render_charts(
    state: AppState,
    tickers: list[str],
    period: str,
    *,
    kind: str = "candlestick",
    overlays: tuple[str, ...] = DEFAULT_OVERLAYS,
    resolution: str | None = DEFAULT_RESOLUTION,
) -> html.Div:
    """Baut den Chartbereich fuer die gewaehlten Titel.

    Bei mehreren Tickern steht ein auf Basis 100 normierter Vergleichschart
    voran, darunter je Titel ein eigener Chart mit Volumen und Overlays. Die
    Normierung rechnet ``analytics.series``, nicht diese Schicht.

    Args:
        state: UI-Zustand.
        tickers: Kanonische Symbole.
        period: Zeitraum in yfinance-Notation.
        kind: Chart-Art aus :data:`CHART_KINDS`.
        overlays: Gewaehlte Overlay-Schluessel. ``"setups"`` legt Zonen und
            Marker in den Chart und setzt darunter je Titel die
            Setup-Checkliste. Je Titel, nicht nur fuer den ersten: Zonen
            und Messwerte gehoeren zum jeweiligen Chart.

        resolution: Kerzen-Aufloesung aus :data:`RESOLUTION_CHOICES`.
            Leer oder unpassend: die kleinstmoegliche des Zeitraums.
            Indikatoren zaehlen Kerzen: bei Wochenkerzen umfasst "SMA 50"
            fuenfzig Wochen, bei Minutenkerzen fuenfzig Minuten. Die
            Setup-Checkliste rechnet unabhaengig davon immer auf
            Tageskerzen - ihre Schwellen sind so definiert.

    Returns:
        Container mit einem Chart je Titel, oder ein Fehlerbanner.
    """
    from btlite.analytics.technicals import build_overlays
    from btlite.viz import price as viz_price
    from btlite.viz.setups import add_setup_annotations

    factories = {
        "candlestick": viz_price.candlestick,
        "ohlc": viz_price.ohlc,
        "line": viz_price.line,
    }
    factory = factories.get(kind, viz_price.candlestick)
    intraday_available = not state.is_offline
    interval = effective_interval(period, resolution, intraday_available=intraday_available)
    intraday = interval in INTRADAY_INTERVALS

    blocks: list[Any] = []
    notice = _resolution_notice(interval, state.timezone, intraday_available=intraday_available)
    if notice is not None:
        blocks.append(notice)
    for symbol in tickers:
        try:
            series = _series_for(state, symbol, period, interval)
        except BTLiteError as exc:
            blocks.append(c.section(symbol, c.error_banner(exc)))
            continue

        try:
            technical = build_overlays(
                series,
                sma_windows=_SMA_WINDOWS if "sma" in overlays else (),
                ema_windows=_EMA_WINDOWS if "ema" in overlays else (),
                with_bollinger="bollinger" in overlays,
                with_rsi="rsi" in overlays,
                with_macd="macd" in overlays,
            )
            figure = factory(
                series,
                technical,
                title=f"{symbol} - {period}, {_CANDLE_LABEL.get(interval, interval.value)}",
                # Die Schnellwahl 1M/3M/1Y ergibt in einem Minutenchart keinen Sinn.
                range_selector=not intraday,
            )
        except BTLiteError as exc:
            blocks.append(c.section(symbol, c.error_banner(exc)))
            continue

        setup_block: Any = None
        if "setups" in overlays:
            report, setup_block = _setup_report(state, symbol)
            if report is not None:
                add_setup_annotations(
                    figure, report, start=series.start, end=series.end, markers=not intraday
                )
                setup_block = setup_panel(report)
            setup_block = c.section(f"{symbol} - Chart setups", setup_block)

        if intraday:
            _intraday_display(figure, interval, state.timezone)

        blocks.append(
            c.section(
                f"{symbol} ({series.interval.value}, "
                f"{'adjusted' if series.adjusted else 'unadjusted'})",
                # RSI und MACD vergroessern die Plotly-Figure je um 110 px.
                # Der Dash-Container muss dieselbe Hoehe bekommen, sonst ragt
                # der Plot ueber den Abschnitt hinaus und ueberdeckt das
                # nachfolgende Setup-Panel.
                c.graph(figure, height=int(figure.layout.height or c.CHART_HEIGHT)),
                c.provenance_note(series.provenance, state.timezone),
            )
        )
        if setup_block is not None:
            blocks.append(setup_block)

    if not blocks or all(getattr(b, "className", "") == "bt-notice" for b in blocks):
        return html.Div([c.placeholder("No price series available.")])
    if len(tickers) > 1:
        position = 1 if notice is not None else 0
        blocks.insert(position, _comparison_block(state, tickers, period, interval))
    return html.Div(blocks, className="bt-charts")


def _resolution_notice(
    interval: Interval, timezone: str, *, intraday_available: bool
) -> html.Div | None:
    """Erklaert, was die gewaehlte Aufloesung zeigt.

    Args:
        interval: Tatsaechlich gezeichnete Aufloesung.
        timezone: Anzeigezeitzone.
        intraday_available: False ohne Live-Quelle.

    Returns:
        Hinweiszeile, oder None bei Tageskerzen mit Live-Quelle.
    """
    if interval in INTRADAY_INTERVALS:
        label = _CANDLE_LABEL.get(interval, interval.value).capitalize()
        text = (
            f"{label} candles, times in {timezone}, regular trading hours only, "
            "delayed by the source (about 15 minutes). Nights and weekends are hidden "
            "on the time axis. Moving averages, RSI and MACD count candles. "
            "Chart setups are still computed on daily candles."
        )
    elif interval is Interval.W1:
        text = (
            "Weekly candles, built from the daily series (Monday to Friday). "
            "Moving averages, RSI and MACD count candles: SMA 50 spans 50 weeks. "
            "Chart setups are still computed on daily candles."
        )
    elif not intraday_available:
        text = "Offline: only daily candles are available. Intraday needs the live source."
    else:
        return None
    return html.Div(text, className="bt-notice")


def _setup_report(state: AppState, symbol: str) -> tuple[SetupReport | None, Any]:
    """Holt die Setup-Checkliste eines Titels.

    Rechnet auf :data:`SETUP_HISTORY_PERIOD`, nicht auf dem dargestellten
    Zeitraum. Die Rechnung selbst liegt in ``analytics.setups``.

    Args:
        state: UI-Zustand.
        symbol: Kanonisches Symbol.

    Returns:
        Bericht und None, oder None und ein Fehlerbanner.
    """
    from btlite.analytics.setups import setup_report

    try:
        series = state.history(symbol, period=SETUP_HISTORY_PERIOD, interval=Interval.D1)
        return setup_report(series), None
    except BTLiteError as exc:
        return None, c.error_banner(exc)


def _format_measurement(item: Measurement) -> str:
    """Formatiert einen Messwert samt Einheit, fehlend als "n/a".

    Args:
        item: Messwert mit Einheit.

    Returns:
        Text wie ``"Distance to 50-day MA: +1.2 %"``.
    """
    value = item.value
    if item.unit is MeasurementUnit.DECIMAL:
        text = fmt.format_percent(value, signed=True)
    elif item.unit is MeasurementUnit.BARS:
        text = fmt.format_integer(value)
    elif item.unit is MeasurementUnit.MULTIPLE:
        text = fmt.NA_TEXT if fmt.is_missing(value) else f"{fmt.format_number(value)}x"
    elif item.unit is MeasurementUnit.INDEX:
        text = fmt.format_number(value, decimals=1)
    else:
        text = fmt.format_number(value)
    return f"{item.label}: {text}"


def setup_panel(report: SetupReport) -> html.Div:
    """Rendert die Setup-Checkliste als Tabelle Setup | Status | Measurements.

    Der Status steht als Text mit Symbol da (Haken plus "met", Strich plus "not met",
    "? insufficient data"); die Farbklasse ist nur Beiwerk. Kein
    Gesamturteil, kein Score. Der Hinweissatz :data:`SETUP_DISCLAIMER`
    steht immer dabei.

    Args:
        report: Fertiger Setup-Bericht.

    Returns:
        Container mit Stichtag, Tabelle und Hinweissatz.
    """
    rows = [
        html.Tr(
            [
                html.Td(check.label, title=check.explanation),
                html.Td(_STATUS_TEXT[check.status], className=_STATUS_CLASS[check.status]),
                html.Td([html.Div(_format_measurement(m)) for m in check.measurements]),
            ]
        )
        for check in report.checks
    ]
    table = html.Table(
        [
            html.Thead(html.Tr([html.Th("Setup"), html.Th("Status"), html.Th("Measurements")])),
            html.Tbody(rows),
        ],
        className="bt-table",
    )
    return html.Div(
        [
            html.P(
                f"As of the daily bar of {report.as_of:%Y-%m-%d} (UTC), "
                f"close {fmt.format_number(report.price)}.",
                className="bt-text",
            ),
            table,
            _zone_details(report),
            html.P(SETUP_DISCLAIMER, className="bt-provenance"),
        ]
    )


def _zone_details(report: SetupReport) -> html.Details | html.P:
    """Listet die Zonen aufklappbar, standardmaessig zugeklappt.

    Die Chartlegende zeigt nur je einen Sammeleintrag fuer Unterstuetzung
    und Widerstand; die Einzelheiten stehen hier.

    Args:
        report: Fertiger Setup-Bericht.

    Returns:
        Zugeklappte Liste mit Art, Niveau, Band, Beruehrungen und Abstand
        zum Kurs, oder ein Satz, wenn es keine Zone gibt.
    """
    if not report.zones:
        return html.P("No confirmed price zones in the last 126 days.", className="bt-text")

    def distance(level: float) -> str:
        if level == 0:
            return fmt.NA_TEXT
        return fmt.format_percent((report.price - level) / level, decimals=1, signed=True)

    rows = [
        html.Tr(
            [
                html.Td(zone.kind.value.capitalize()),
                html.Td(fmt.format_number(zone.level)),
                html.Td(f"{fmt.format_number(zone.lower)} - {fmt.format_number(zone.upper)}"),
                html.Td(str(zone.touches)),
                html.Td(distance(zone.level)),
            ]
        )
        for zone in sorted(report.zones, key=lambda z: z.level, reverse=True)
    ]
    return html.Details(
        [
            html.Summary(f"Price zones ({len(report.zones)})", className="bt-subhead"),
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Zone"),
                                html.Th("Level"),
                                html.Th("Band"),
                                html.Th("Touches"),
                                html.Th("Price vs. level"),
                            ]
                        )
                    ),
                    html.Tbody(rows),
                ],
                className="bt-table",
            ),
        ],
        className="bt-ratio-group",
    )


def _comparison_block(
    state: AppState, tickers: list[str], period: str, interval: Interval = Interval.D1
) -> Any:
    """Baut den normierten Vergleichschart ueber alle gewaehlten Titel.

    Die Normierung auf Basis 100 rechnet
    :func:`btlite.analytics.series.build_comparison`, nicht diese Schicht
    (Architekturregel 4). Ohne sie waeren Titel mit sehr verschiedenen
    Kursniveaus im selben Chart nicht vergleichbar.

    Args:
        state: UI-Zustand.
        tickers: Kanonische Symbole, mindestens zwei.
        period: Zeitraum in yfinance-Notation.
        interval: Kerzen-Aufloesung, dieselbe wie die Einzelcharts.

    Returns:
        Abschnitt mit dem Vergleichschart, oder ein Hinweis, wenn die
        Reihen sich nicht ausreichend ueberlappen.
    """
    from btlite.analytics.series import build_comparison
    from btlite.viz import price as viz_price

    try:
        series = {t: _series_for(state, t, period, interval) for t in tickers}
        comparison = build_comparison(series)
        figure = viz_price.comparison(
            comparison,
            title=f"Comparison - {period}",
            range_selector=interval not in INTRADAY_INTERVALS,
        )
        if interval in INTRADAY_INTERVALS:
            _intraday_display(figure, interval, state.timezone)
    except BTLiteError as exc:
        return c.section("Comparison", c.error_banner(exc))

    return c.section(
        f"Comparison, base 100 ({', '.join(tickers)})",
        c.graph(figure, height=c.CHART_HEIGHT),
        c.provenance_note(comparison.provenance, state.timezone),
    )


def summary_tiles(state: AppState, tickers: list[str]) -> html.Div:
    """Baut eine Kurszeile mit den aktuellen Quotes der Titel.

    Args:
        state: UI-Zustand.
        tickers: Kanonische Symbole.

    Returns:
        Kachel-Raster, oder ein Hinweis wenn keine Quotes vorliegen.
    """
    try:
        quotes = state.quotes(tickers)
    except BTLiteError as exc:
        return c.error_banner(exc)

    items: list[tuple[str, str, str]] = []
    for symbol in tickers:
        quote = quotes.get(symbol)
        if quote is None:
            items.append((symbol, fmt.NA_TEXT, "bt-na"))
            continue
        items.append(
            (
                symbol,
                f"{fmt.format_currency(quote.price, quote.currency)} "
                f"({fmt.format_percent(quote.change_pct, signed=True)})",
                fmt.direction_class(quote.change_pct),
            )
        )
    return c.tiles(items)
