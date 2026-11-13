"""Seite EQS - Screening mit Presets, Filteranzeige und CSV-Export.

Die Seite verdrahtet ``screening.presets``, ``screening.universe`` und
``screening.engine``. Die Filterlogik liegt vollstaendig dort.

Das Universum ist ein **Index**, nicht eine Handvoll Titel. Fruehere
Fassungen screenten die zehn Ticker mit Offline-Fixtures - ein Screening
ueber zehn Werte beantwortet keine Frage, die der Nutzer stellt. Waehlbar
sind die Indizes aus :mod:`btlite.data.indices`, voreingestellt der
S&P 500.

Der Bestand kommt live und wird eine Woche gecacht. Offline ist nur ein
bereits gefuellter Cache bedienbar; sonst erscheint ein Fehlerbanner mit
dem Weg zu Live-Daten - ein stiller Rueckfall auf zehn Titel waere
derselbe Mangel in neuer Form.

Zur Laufzeit: das erste Screening eines grossen Index laedt die
Kurshistorie je Titel. Beim S&P 500 sind das ueber 500 Abrufe, die die
Drosselung der Quelle einhalten und entsprechend dauern. Deshalb steht
die Titelzahl im Kopf der Seite, bevor gerechnet wird.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Final

from dash import dcc, html

from btlite.core.errors import BTLiteError, EmptyUniverseError
from btlite.core.logging import get_logger
from btlite.data.indices import DEFAULT_INDEX_KEY, INDEX_REGISTRY
from btlite.models.common import Period
from btlite.ui import formatting as fmt
from btlite.ui.layouts import components as c

if TYPE_CHECKING:
    import pandas as pd

    from btlite.models.screening import ScreenPreset, ScreenResult, TechnicalRefinement
    from btlite.ui.state import AppState

_LOG = get_logger(__name__)

#: Rueckruf fuer den Fortschritt: ``(erledigt, gesamt)``.
ProgressCallback = Callable[[int, int], None] | None

#: Praedikat fuer den Abbruch. Bewusst getrennt vom Fortschritt: ein
#: Abbruch ist kein gescheiterter Fortschritt, und jener Rueckruf
#: schluckt Ausnahmen absichtlich - der Abbruch wuerde mit
#: verschluckt und die Anzeige nebenbei stillgelegt.
StopCheck = Callable[[], bool] | None

#: Route dieser Seite.
PATH = "/eqs"

#: Element-IDs der Bedienelemente.
PRESET_INPUT = "eqs-preset"
INDEX_INPUT = "eqs-index"
UNIVERSE_STORE = "eqs-universe"
RESULT_OUTPUT = "eqs-result"
RUN_BUTTON = "eqs-run"
EXPORT_BUTTON = "eqs-export"
DOWNLOAD = "eqs-download"
CANCEL_BUTTON = "eqs-cancel"
JOB_STORE = "eqs-job"
PROGRESS_TICK = "eqs-progress-tick"
SMA200_INPUT = "eqs-sma200"
SMA50_INPUT = "eqs-sma50"
RSI_ABOVE_INPUT = "eqs-rsi-above"
RSI_BELOW_INPUT = "eqs-rsi-below"
SETUP_INPUT = "eqs-setups"

#: Die vier Zusatzfilter in der Reihenfolge von
#: :func:`btlite.screening.refine.build_refinement`. Die Callbacks haengen
#: sie in genau dieser Reihenfolge als States an.
REFINEMENT_INPUTS: Final[tuple[str, ...]] = (
    SMA200_INPUT,
    SMA50_INPUT,
    RSI_ABOVE_INPUT,
    RSI_BELOW_INPUT,
    SETUP_INPUT,
)

#: Auswertungen, die jede Trefferzeile direkt anbietet: Mnemonic und der
#: Gegenstand im Tooltip. Die Reihenfolge ist die der Links in der Zelle.
#: Die Beschriftung stammt aus ``NAV_LABELS``, damit der Link genauso
#: heisst wie der Navigationspunkt, auf dem er landet.
_RESULT_LINKS: Final[tuple[tuple[str, str], ...]] = (
    ("GP", "price chart"),
    ("DES", "profile"),
    ("FA", "financial analysis"),
)

#: Auswahl fuer die Lage zu einer Linie.
_MA_OPTIONS: Final[list[dict[str, str]]] = [
    {"label": "Any", "value": "any"},
    {"label": "Price above", "value": "above"},
    {"label": "Price below", "value": "below"},
]

#: Abstand der Fortschrittsabfrage in Millisekunden. Eine Sekunde ist
#: haeufig genug, damit die Anzeige lebendig wirkt, und selten genug, dass
#: die Abfrage neben einem minutenlangen Lauf nicht ins Gewicht faellt.
PROGRESS_INTERVAL_MS: Final[int] = 1000

#: Bis zu dieser Titelzahl rechnet die Seite beim Oeffnen oder bei einem
#: Preset-Wechsel sofort, darueber erst auf Knopfdruck im Hintergrund.
#: Frueher 50: das waren kalt bis zu zweieinhalb Minuten im Web-Thread, mit
#: Spinner, ohne Fortschritt und ohne Abbruch. Zehn Titel kosten kalt
#: hoechstens eine halbe Minute, warm eine Sekunde.
_SLOW_UNIVERSE: Final[int] = 10

#: Ab dieser Groesse lohnt der zusaetzliche Finviz-Request, bevor fuer jeden
#: Titel drei Statements und Stammdaten von Yahoo geladen werden.
_REMOTE_PREFILTER_MIN: Final[int] = 25

#: Verzoegerung des Lade-Spinners in Millisekunden. Laenger als ein
#: Fortschrittstakt dauert, kuerzer als eine Pause, die man bemerkt.
_SPINNER_DELAY_MS: Final[int] = 500

#: Gemessene Dauer je Titel in Sekunden, inklusive Drosselung. Nachgemessen
#: am 25.09.2026 gegen die Live-Quelle, je 20 Titel des S&P 500 mit eigenem
#: leerem Cache-Verzeichnis.
#:
#: Die frueheren Werte (0.45 und 0.89) unterschieden technische von
#: bilanzbasierten Presets und lagen um den Faktor vier zu niedrig: die
#: Seite kuendigte vier bis sieben Minuten an, wo real 25 bis 30 vergingen.
#: Ein Nutzer brach deshalb ab, waehrend die Seite noch rechnete, und
#: meldete sie als "liefert keine Ergebnisse".
#:
#: Die Unterscheidung technisch/fundamental traegt nicht: beide Arten laden
#: je Titel eine Kurshistorie, und die dominiert (gemessen 2.97 gegen
#: 3.55 s/Titel - kein Faktor zwei). Massgeblich ist stattdessen, ob der
#: Cache warm ist.
#:
#: Warm am 26.09.2026 nachgemessen: 0.03 bis 0.09 s/Titel (40 Titel des
#: S&P 500, 1.1 bis 3.5 s je Lauf). Vorher 1.05 s - die Stammdaten aus
#: ``Ticker.info`` wurden bei jedem Lauf zweimal je Titel neu geladen, ein
#: Folgelauf war dadurch kaum schneller als der erste. Seit sie vier Stunden
#: gepuffert werden, ist er es. 0.1 ist die aufgerundete Obergrenze.
_SECONDS_PER_TICKER_COLD: Final[float] = 3.3
_SECONDS_PER_TICKER_WARM: Final[float] = 0.1

#: Kennzahlenspalten der Ergebnistabelle, in dieser Reihenfolge.
_RESULT_COLUMNS: Final[tuple[str, ...]] = (
    "name",
    "sector",
    "pe",
    "pb",
    "ps",
    "roe",
    "net_margin",
    "debt_to_equity",
    "dividend_yield",
    "market_cap",
)

#: Ausgeschriebene Ueberschriften der Ergebnisspalten. Die Rohnamen sind
#: Kuerzel aus der Datenschicht - "pe" oder "dist_sma_200" sagt nur dem
#: etwas, der sie ohnehin kennt. Der Rohname bleibt als Tooltip an der
#: Spalte haengen, damit die Zuordnung zum CSV-Export klar bleibt.
_COLUMN_HEADERS: Final[dict[str, str]] = {
    "name": "Name",
    "sector": "Sector",
    "pe": "Price/Earnings",
    "pb": "Price/Book",
    "ps": "Price/Sales",
    "roe": "Return on equity",
    "roa": "Return on assets",
    "net_margin": "Net margin",
    "gross_margin": "Gross margin",
    "operating_margin": "Operating margin",
    "debt_to_equity": "Debt/Equity",
    "current_ratio": "Current ratio",
    "dividend_yield": "Dividend yield",
    "market_cap": "Market cap",
    "revenue": "Revenue",
    "net_income": "Net income",
    "price": "Price",
    "sma_20": "20-day MA",
    "sma_50": "50-day MA",
    "sma_200": "200-day MA",
    "dist_sma_20": "Distance to 20-day MA",
    "dist_sma_50": "Distance to 50-day MA",
    "dist_sma_200": "Distance to 200-day MA",
    "above_sma_200": "Above 200-day MA",
    "golden_cross": "50-day above 200-day",
    "rsi_14": "RSI (14 days)",
    "perf_1m": "Return 1 month",
    "perf_3m": "Return 3 months",
    "perf_6m": "Return 6 months",
    "perf_12m": "Return 12 months",
    "from_high_52w": "Distance to 52-week high",
    "from_low_52w": "Distance to 52-week low",
    "volatility_1y": "Volatility (1 year)",
    "max_drawdown_1y": "Max drawdown (1 year)",
    "avg_volume_3m": "Avg volume (3 months)",
    "setup_pullback": "Setup: pullback",
    "setup_rsi_recovery": "Setup: RSI recovery",
    "setup_breakout": "Setup: breakout",
    "setup_support_macd": "Setup: support + MACD",
    "dist_support": "Distance to support",
    "dist_resistance": "Distance to resistance",
}

#: Spalten, die als dezimaler Anteil vorliegen und als Prozent erscheinen.
_PERCENT_COLUMNS: Final[frozenset[str]] = frozenset(
    {"roe", "roa", "net_margin", "gross_margin", "operating_margin", "dividend_yield"}
)

#: Spalten, die gekuerzt dargestellt werden.
_COMPACT_COLUMNS: Final[frozenset[str]] = frozenset({"market_cap", "revenue", "net_income"})

#: Kennzahlenspalten der Ergebnistabelle eines technischen Screens.
_TECHNICAL_RESULT_COLUMNS: Final[tuple[str, ...]] = (
    "name",
    "sector",
    "price",
    "sma_200",
    "dist_sma_50",
    "dist_sma_200",
    "above_sma_200",
    "golden_cross",
    "rsi_14",
    "perf_3m",
    "perf_12m",
    "from_high_52w",
    "volatility_1y",
    "max_drawdown_1y",
)

#: Technische Spalten, die als dezimaler Anteil vorliegen.
_TECHNICAL_PERCENT_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "dist_sma_20",
        "dist_sma_50",
        "dist_sma_200",
        "perf_1m",
        "perf_3m",
        "perf_6m",
        "perf_12m",
        "from_high_52w",
        "from_low_52w",
        "volatility_1y",
        "max_drawdown_1y",
        "dist_support",
        "dist_resistance",
    }
)

#: Technische Spalten, die als Ja/Nein-Schalter gemeint sind (1.0 / 0.0).
_FLAG_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "above_sma_200",
        "golden_cross",
        "setup_pullback",
        "setup_rsi_recovery",
        "setup_breakout",
        "setup_support_macd",
    }
)

#: Technische Spalten auf einer festen Skala, ohne Einheitenumrechnung.
_SCALE_COLUMNS: Final[frozenset[str]] = frozenset({"rsi_14"})

#: Zeitraum der Kursreihen, die die technischen Spalten speisen. Die
#: 200-Tage-Linie und die Zwoelfmonats-Groessen brauchen mindestens 252
#: Handelstage Historie.
_TECHNICAL_HISTORY_PERIOD: Final[str] = "2y"


def layout(state: AppState, tickers: list[str], args: tuple[str, ...] = ()) -> html.Div:
    """Baut die Seite EQS.

    Args:
        state: UI-Zustand mit Provider und Puffern.
        tickers: Symbole aus der Kommandozeile. Sie bilden das
            Startuniversum. Leer nimmt alles, was die Quelle fuehrt.
        args: Weitere Argumente. Ein Preset-Key waehlt das Preset, etwa
            ``EQS value GO``.

    Returns:
        Seiteninhalt mit Preset-Wahl, Filteranzeige und Trefferliste.
    """
    presets = _all_presets()
    preset_key = _preset_from_args(args, presets)
    index_key = _index_from_args(args)
    universe, universe_note, universe_error = _resolve_universe(
        state, tickers, index_key, preset_key
    )

    body: list[Any] = []
    if universe_error is not None:
        # Mit Ersatzuniversum arbeitet die Seite weiter - das ist ein
        # Hinweis. Ohne jedes Universum ist es ein Fehler.
        if universe:
            banner = c.warning_banner([universe_error.user_message])
            if banner is not None:
                body.append(banner)
        else:
            body.append(c.error_banner(universe_error))

    body.append(
        c.loading(
            html.Div(id=RESULT_OUTPUT, children=initial_result(state, preset_key, universe)),
            # Die Fortschrittsabfrage schreibt jede Sekunde in diesen Bereich
            # und braucht dafuer Millisekunden. Ohne Verzoegerung blendete
            # der Spinner die Anzeige bei jedem Takt kurz aus.
            delay_show_ms=_SPINNER_DELAY_MS,
        )
    )

    return html.Div(
        [
            c.page_header("Equity screening", universe_note),
            dcc.Store(id=UNIVERSE_STORE, data=universe),
            dcc.Store(id=JOB_STORE, data=None),
            # Fragt den Stand des laufenden Screenings ab. Ruht, solange
            # nichts laeuft - ein dauernd tickendes Interval erzeugte
            # sonst im Leerlauf Last fuer nichts.
            dcc.Interval(id=PROGRESS_TICK, interval=PROGRESS_INTERVAL_MS, disabled=True),
            dcc.Download(id=DOWNLOAD),
            c.controls(
                c.labeled(
                    "Universe",
                    dcc.Dropdown(
                        id=INDEX_INPUT,
                        options=_index_options(state),
                        value=index_key,
                        clearable=False,
                        className="bt-dropdown",
                    ),
                ),
                c.labeled(
                    "Preset",
                    dcc.Dropdown(
                        id=PRESET_INPUT,
                        options=_preset_options(),
                        value=preset_key,
                        clearable=False,
                        className="bt-dropdown",
                    ),
                ),
                html.Button(
                    "Run screening",
                    id=RUN_BUTTON,
                    n_clicks=0,
                    className="btn btn-primary",
                ),
                html.Button(
                    "Cancel",
                    id=CANCEL_BUTTON,
                    n_clicks=0,
                    hidden=True,
                    title="Stop the running screening",
                    # Nicht btn-secondary: dafuer gibt es keine eigene Regel,
                    # der Knopf erbte Bootstrap-Fremdfarben. bt-manage-action
                    # ist der zurueckgenommene Knopf des Projekts.
                    className="bt-manage-action",
                ),
                html.Button(
                    "Export CSV",
                    id=EXPORT_BUTTON,
                    n_clicks=0,
                    className="btn btn-primary",
                ),
            ),
            _refinement_controls(),
            *body,
        ],
        className="bt-page bt-page-eqs",
    )


def _refinement_controls() -> html.Div:
    """Baut die Zeile der zusaetzlichen Kursfilter.

    Die Filter wirken erst beim naechsten Lauf, nicht bei jeder Eingabe:
    ein Tastendruck im RSI-Feld darf keinen Lauf ueber 500 Titel starten.

    Returns:
        Hinweiszeile und Bedienelemente.
    """

    def ma_dropdown(component_id: str) -> dcc.Dropdown:
        return dcc.Dropdown(
            id=component_id,
            options=_MA_OPTIONS,
            value="any",
            clearable=False,
            className="bt-dropdown",
        )

    def rsi_input(component_id: str, example: str) -> dcc.Input:
        return dcc.Input(
            id=component_id,
            type="number",
            min=0,
            max=100,
            step="any",
            placeholder=example,
            debounce=True,
            className="bt-field-input",
        )

    return html.Div(
        [
            html.Div(
                "Extra price filters - they narrow down any preset and apply on the next "
                "run. With a fundamentals preset, price history is loaded only for the "
                "titles that pass the preset.",
                className="bt-notice",
            ),
            c.controls(
                c.labeled("200-day MA", ma_dropdown(SMA200_INPUT)),
                c.labeled("50-day MA", ma_dropdown(SMA50_INPUT)),
                c.labeled("RSI (14) above", rsi_input(RSI_ABOVE_INPUT, "e.g. 30")),
                c.labeled("RSI (14) below", rsi_input(RSI_BELOW_INPUT, "e.g. 70")),
                c.labeled(
                    "Chart setups (all must be met)",
                    dcc.Dropdown(
                        id=SETUP_INPUT,
                        options=_setup_options(),
                        value=[],
                        multi=True,
                        placeholder="None",
                        className="bt-dropdown",
                    ),
                ),
            ),
            html.Div(
                "Chart setups are measurable conditions, not a recommendation. "
                "Details per title on the Price page.",
                className="bt-notice",
            ),
        ],
        className="bt-eqs-refine",
    )


def _setup_options() -> list[dict[str, str]]:
    """Auswahl der Chart-Setups, beschriftet wie auf der Seite GP.

    Returns:
        Optionen fuer die Mehrfachauswahl.
    """
    from btlite.analytics.setups import SETUP_LABELS

    return [{"label": label, "value": key.value} for key, label in SETUP_LABELS.items()]


def refinement_from_inputs(*values: object) -> TechnicalRefinement:
    """Liest die Zusatzfilter aus den Werten der vier Eingabefelder.

    Args:
        *values: Werte in der Reihenfolge von :data:`REFINEMENT_INPUTS`.
            Fehlende Werte gelten als nicht gesetzt.

    Returns:
        Die Einstellung.

    Raises:
        ScreeningError: Bei ungueltigen Werten, mit lesbarer Meldung.
    """
    from btlite.screening.refine import build_refinement

    return build_refinement(*values[: len(REFINEMENT_INPUTS)])


def _index_options(state: AppState) -> list[dict[str, str]]:
    """Listet die waehlbaren Indizes fuer das Dropdown.

    Args:
        state: UI-Zustand mit dem Indexanbieter.

    Returns:
        Optionen mit Anzeigename und Schluessel, in Registry-Reihenfolge.
    """
    return [
        {"label": definition.label, "value": definition.key}
        for definition in state.indices.available_indices()
    ]


def _index_from_args(args: tuple[str, ...]) -> str:
    """Liest den gewuenschten Index aus den Kommandozeilen-Argumenten.

    Args:
        args: Argumente hinter dem Mnemonic, etwa ``EQS dax value GO``.

    Returns:
        Schluessel eines bekannten Index, sonst :data:`DEFAULT_INDEX_KEY`.
    """
    for arg in args:
        candidate = arg.strip().casefold()
        if candidate in INDEX_REGISTRY:
            return candidate
    return DEFAULT_INDEX_KEY


def _resolve_universe(
    state: AppState,
    tickers: list[str],
    index_key: str,
    preset_key: str = "",
) -> tuple[list[str], str, BTLiteError | None]:
    """Bestimmt das Anlageuniversum der Seite.

    Eine ausdrueckliche Tickerliste aus der Kommandozeile gewinnt: wer
    ``AAPL,MSFT EQS GO`` eingibt, meint genau diese beiden. Sonst kommt
    der Bestand des gewaehlten Index.

    Faellt der Abruf aus - offline oder Quelle nicht erreichbar -, wird
    **nicht** still auf die Fixture-Ticker zurueckgefallen. Der Nutzer
    waehlte "S&P 500"; zehn Titel als stilles Ersatzuniversum waeren eine
    Falschaussage ueber die Breite der Auswertung.

    Args:
        state: UI-Zustand.
        tickers: Symbole aus der Kommandozeile, moeglicherweise leer.
        index_key: Schluessel des gewaehlten Index.
        preset_key: Gewaehltes Preset. Bestimmt die Dauerabschaetzung im
            Hinweistext, weil Bilanzdaten teurer sind als Kurse.

    Returns:
        Tripel aus Universum, Kopfzeilentext und - falls der Abruf
        scheiterte - dem Fehler zur Anzeige.
    """
    if tickers:
        return tickers, f"Universe: {', '.join(tickers)}", None

    definition = INDEX_REGISTRY.get(index_key)
    label = definition.label if definition is not None else index_key

    # Offline nuetzt der beste Indexbestand nichts: zu 503 Titeln liegen
    # nur zu zehn Kurse vor, der Screen faende fuer den Rest nichts. Der
    # Cache haelt Bestaende eine Woche und wuerde sie auch offline
    # ausliefern - hier waere das eine Falle statt eines Vorteils.
    if state.is_offline:
        fallback = state.available_tickers()
        note = (
            f"Offline mode: {label} needs live data, using the "
            f"{len(fallback)} frozen titles instead"
        )
        return fallback, note, None

    try:
        # get_constituents statt get_tickers: nur so erfaehrt die Seite,
        # wie viele Titel die Quelle fuehrte und wie viele davon nicht
        # sicher zuzuordnen waren. Beim STOXX Europe 600 sind das vierzig
        # von 462 - eine Luecke, die der Nutzer kennen muss.
        constituents = state.indices.get_constituents(index_key)
        members = [member.ticker for member in constituents.members]
    except BTLiteError as exc:
        _LOG.warning("Indexbestand %s nicht abrufbar: %s", index_key, exc.detail)
        # Offline bleiben die eingefrorenen Titel als Notbehelf. Sie
        # werden AUSDRUECKLICH als solche benannt: der Nutzer waehlte
        # "S&P 500" und muss erkennen, dass er etwas anderes sieht. Ein
        # stiller Tausch waere die Falschaussage, die es zu vermeiden gilt.
        fallback = state.available_tickers()
        if fallback:
            note = f"{label} unavailable - using the {len(fallback)} frozen titles instead"
            return fallback, note, exc
        return [], f"Universe {label} unavailable", exc

    note = f"Universe: {label}, {len(members)} titles"
    if not constituents.is_complete:
        skipped = constituents.source_count - len(members)
        note += f" of {constituents.source_count} - {skipped} skipped without a unique symbol"
    if len(members) > _SLOW_UNIVERSE:
        estimate = format_minutes(_estimate_minutes(len(members), preset_key))
        note += f" - first run takes about {estimate}, cached afterwards"
    return members, note, None


def initial_result(
    state: AppState,
    preset_key: str,
    universe: list[str],
    refinement: TechnicalRefinement | None = None,
) -> Any:
    """Waehlt zwischen sofort rechnen und erst fragen.

    Ein Klick im Navigationsmenue oder ein Wechsel des Universums darf
    keinen Siebenminutenlauf ausloesen. Bei einem grossen Universum zeigt
    die Seite deshalb erst, was der Lauf kosten wuerde; gerechnet wird auf
    Knopfdruck. Bei wenigen Titeln kostet der Lauf nichts und startet
    direkt - eine Rueckfrage waere dort nur im Weg.

    Diese Entscheidung steht bewusst an einer Stelle. Sie lag frueher
    dreifach verteilt im Seitenaufbau und in zwei Callbacks, und die
    Callbacks rechneten ungefragt durch: die Seite stand dann minutenlang
    auf "updating", obwohl sie beim Oeffnen korrekt gefragt hatte.

    Args:
        state: UI-Zustand.
        preset_key: Gewaehltes Preset.
        universe: Startuniversum.
        refinement: Zusatzfilter, oder None.

    Returns:
        Trefferliste, Kostenhinweis, oder ein Platzhalter bei leerem
        Universum.
    """
    if not universe:
        return c.placeholder("No universe: without titles there is nothing to filter.")
    if len(universe) > _SLOW_UNIVERSE:
        return _run_prompt(len(universe), preset_key, refinement)
    return render_result(state, preset_key, universe, refinement=refinement)


def _run_prompt(
    count: int, preset_key: str, refinement: TechnicalRefinement | None = None
) -> html.Div:
    """Zeigt an, was ein Screening kosten wuerde, statt es zu starten.

    Bei einem grossen Universum dauert der erste Durchlauf Minuten. Ihn
    ungefragt zu starten, nur weil jemand die Seite geoeffnet hat, blockiert
    die Oberflaeche - der Nutzer sieht "updating" und weiss nicht, wie lange.

    Args:
        count: Anzahl der Titel im Universum.
        preset_key: Gewaehltes Preset, bestimmt die Dauerabschaetzung.
        refinement: Zusatzfilter. Bei einem fundamentalen Preset kommt ein
            zweiter Schritt hinzu, dessen Umfang erst nach dem ersten
            feststeht - das sagt der Hinweis ausdruecklich, statt eine
            Zahl zu erfinden.

    Returns:
        Hinweis mit Dauer und der Aufforderung, selbst zu starten.
    """
    minutes = _estimate_minutes(count, preset_key)
    extra = ""
    if _needs_second_step(preset_key, refinement):
        extra = (
            " With the extra price filters, price history for the titles that pass "
            f"the preset follows, about {_SECONDS_PER_TICKER_COLD:g} seconds each on "
            "the first run."
        )
    return c.placeholder(
        f"{count} titles ready. The first run takes about {format_minutes(minutes)}, "
        f"after that the result is cached.{extra} "
        'Press "Run screening" to start.'
    )


def _needs_second_step(preset_key: str, refinement: TechnicalRefinement | None) -> bool:
    """Ob ein Lauf nach den Bilanzen noch Kursreihen laden muss.

    Args:
        preset_key: Gewaehltes Preset.
        refinement: Zusatzfilter, oder None.

    Returns:
        True bei einem fundamentalen Preset mit aktiven Kursfiltern. Ein
        technisches Preset laedt die Kurse ohnehin.
    """
    active = refinement is not None and refinement.is_active
    return active and not is_technical_preset(preset_key)


def format_minutes(minutes: int) -> str:
    """Schreibt eine Minutenangabe mit richtigem Numerus.

    Args:
        minutes: Anzahl Minuten.

    Returns:
        Etwa ``"28 minutes"`` oder ``"1 minute"``.
    """
    return f"{minutes} minute" if minutes == 1 else f"{minutes} minutes"


def _remaining_text(remaining: int, seconds_per_unit: float | None) -> str:
    """Formuliert die Restzeit eines laufenden Screenings.

    Frueher galt ab dem ersten Titel das warme Tempo, mit der Begruendung,
    die ersten Titel haetten den Cache gefuellt. Das stimmte nicht: jeder
    Titel hat seinen eigenen Cache-Eintrag. Ein kalter Lauf zeigte deshalb
    ein Drittel der echten Restzeit an.

    Args:
        remaining: Noch offene Titel.
        seconds_per_unit: Gemessenes Tempo, oder None.

    Returns:
        Etwa ``"about 12 minutes left"``, ``"less than a minute left"``
        oder ``"almost done"``.
    """
    if remaining <= 0:
        return "almost done"
    if seconds_per_unit is None:
        # Noch nichts gemessen: im Zweifel kalt, siehe _estimate_minutes.
        return f"about {format_minutes(_estimate_minutes(remaining))} left"
    seconds = remaining * seconds_per_unit
    if seconds < 60:
        return "less than a minute left"
    return f"about {format_minutes(math.ceil(seconds / 60))} left"


def _estimate_minutes(count: int, preset_key: str = "", *, warm: bool = False) -> int:
    """Schaetzt die Dauer eines Durchlaufs in Minuten.

    Im Zweifel wird der **kalte** Fall angenommen. Eine unterschaetzte
    Wartezeit laesst den Nutzer abbrechen, waehrend die Seite noch
    arbeitet - genau so ging die urspruengliche Meldung ein. Eine zu
    grosszuegige Schaetzung kostet dagegen nichts als eine angenehme
    Ueberraschung.

    Args:
        count: Anzahl der zu pruefenden Titel.
        preset_key: Gewaehltes Preset. Wird nicht mehr ausgewertet - die
            Art des Presets aendert die Dauer kaum, siehe
            :data:`_SECONDS_PER_TICKER_COLD`. Der Parameter bleibt, weil
            mehrere Aufrufstellen ihn uebergeben.
        warm: Fuer die Restzeit eines bereits laufenden Screenings, bei
            dem die ersten Titel den Cache gefuellt haben.

    Returns:
        Geschaetzte Minuten, mindestens 1 bei positiver Titelzahl.
    """
    del preset_key
    if count <= 0:
        return 0
    per_ticker = _SECONDS_PER_TICKER_WARM if warm else _SECONDS_PER_TICKER_COLD
    return max(1, round(count * per_ticker / 60))


def _preset_from_args(args: tuple[str, ...], presets: list[ScreenPreset]) -> str:
    """Liest den Preset-Key aus den Kommandozeilen-Argumenten.

    Args:
        args: Argumente hinter dem Mnemonic.
        presets: Verfuegbare Presets.

    Returns:
        Gewaehlter Key, sonst der erste verfuegbare.
    """
    keys = {p.key for p in presets}
    for arg in args:
        candidate = arg.strip().lower()
        if candidate in keys:
            return candidate
    return presets[0].key if presets else "value"


def _all_presets() -> list[ScreenPreset]:
    """Listet fundamentale und technische Presets in Anzeigereihenfolge.

    Returns:
        Erst die Bilanz-, dann die Kurs-Presets.
    """
    from btlite.screening.presets import all_presets
    from btlite.screening.technical_presets import all_technical_presets

    return [*all_presets(), *all_technical_presets()]


def is_technical_preset(preset_key: str) -> bool:
    """Prueft, ob ein Key zu einem kursbasierten Preset gehoert.

    Args:
        preset_key: Key des Presets.

    Returns:
        True wenn das Preset auf den technischen Spalten arbeitet.
    """
    from btlite.screening.technical_presets import technical_preset_keys

    return preset_key.strip().lower() in set(technical_preset_keys())


def resolve_preset(preset_key: str) -> ScreenPreset:
    """Sucht ein Preset ueber beide Kataloge.

    Args:
        preset_key: Key des Presets, Gross- und Kleinschreibung egal.

    Returns:
        Das Preset.

    Raises:
        UnknownFilterError: Wenn der Key in keinem Katalog vorkommt. Die
            technische Suche wirft einen ``KeyError``, der die Callbacks
            nicht erreichen darf - deshalb wird hier zuerst am Katalog
            geprueft und danach der Fehlertyp des Projekts verwendet.
    """
    from btlite.screening.presets import get_preset
    from btlite.screening.technical_presets import get_technical_preset

    if is_technical_preset(preset_key):
        return get_technical_preset(preset_key.strip().lower())
    return get_preset(preset_key)


def _technical_universe(
    state: AppState,
    universe: list[str],
    on_progress: ProgressCallback = None,
    should_stop: StopCheck = None,
) -> Any:
    """Baut das Startuniversum eines kursbasierten Screens.

    Die Kursreihen kommen ueber den injizierten Provider, die Spalten baut
    ``screening.technicals`` (Architekturregeln 2 und 4). Die beschreibenden
    Bilanzspalten - Name und Sektor - werden nur angehaengt, wenn sie ohne
    Fehler beschaffbar sind: ein ETF ohne Bilanz soll in einem rein
    technischen Screen nicht verschwinden.

    Args:
        state: UI-Zustand.
        universe: Startuniversum als Symbolliste.
        on_progress: Optionaler Rueckruf ``(erledigt, gesamt)``.
        should_stop: Optionales Praedikat. Liefert es True, endet der
            Lauf am naechsten Symbol mit dem bis dahin Gesammelten.

    Returns:
        Angereichertes Universum, Index = Ticker.
    """
    import pandas as pd

    from btlite.screening.technicals import enrich_with_technicals

    prices = state.histories(
        universe,
        period=_TECHNICAL_HISTORY_PERIOD,
        on_progress=on_progress,
        should_stop=should_stop,
        include_currency=False,
    )
    if not prices:
        # Keine einzige Kursreihe. Der Frame bliebe leer und liefe in die
        # Filter-Pruefung, die dann eine fehlende Spalte meldet - eine
        # Aussage ueber das falsche Problem.
        msg = "No price history arrived for any title."
        raise EmptyUniverseError(msg, detail=f"angefragt={len(universe)}")

    # Technische Presets brauchen keine Statements. Frueher wurde hier fuer
    # alle Titel ratio_universe() allein fuer Name/Sektor aufgebaut: drei
    # Statement- plus ein Info-Request je Titel, obwohl kein Filter diese
    # Spalten benoetigt. Die Ergebnislinks tragen den Ticker bereits selbst.
    index = pd.Index(sorted(prices), name="ticker")
    base = pd.DataFrame(index=index)
    return enrich_with_technicals(base, prices)


def _fundamentals_with_prices(
    state: AppState,
    universe: list[str],
    preset: ScreenPreset,
    on_progress: ProgressCallback = None,
    should_stop: StopCheck = None,
) -> pd.DataFrame:
    """Baut ein Bilanz-Universum und haengt fuer die Treffer Kursspalten an.

    Zwei Schritte. Erst die Kennzahlen aller Titel, dann die Kursreihen -
    aber nur fuer die Titel, die die Filter des Presets bestehen. Die
    Zusatzfilter gelten per UND; wer am Preset scheitert, faellt ohnehin
    heraus. Das Ergebnis ist dasselbe wie mit Kursen fuer alle, der
    zweite Schritt aber meist ein Bruchteil des ersten.

    Der Fortschritt zaehlt beide Schritte zusammen: im ersten bis zur
    Titelzahl, im zweiten darueber hinaus. Daran erkennt die Anzeige,
    in welchem Schritt der Lauf steht.

    Args:
        state: UI-Zustand.
        universe: Startuniversum.
        preset: Das fundamentale Preset.
        on_progress: Optionaler Rueckruf ``(erledigt, gesamt)``.
        should_stop: Optionales Abbruch-Praedikat.

    Returns:
        Universum mit Kennzahlen, fuer die Vorauswahl zusaetzlich mit den
        technischen Spalten. Titel ohne Kursreihe tragen dort Fehlwerte
        und bestehen keinen Kursfilter.

    Raises:
        EmptyUniverseError: Wenn Titel die Vorauswahl bestehen, aber fuer
            keinen eine Kursreihe ankam.
    """
    from btlite.screening.refine import tickers_passing
    from btlite.screening.technicals import TECHNICAL_COLUMNS, enrich_with_technicals

    base = state.ratio_universe(
        universe, period=Period.ANNUAL, on_progress=on_progress, should_stop=should_stop
    )
    first = len(universe)
    candidates = tickers_passing(base, preset.request.local_filters)

    def second_step(done: int, total: int) -> None:
        if on_progress is not None:
            on_progress(first + done, first + total)

    prices = {}
    if candidates and not (should_stop is not None and should_stop()):
        prices = state.histories(
            candidates,
            period=_TECHNICAL_HISTORY_PERIOD,
            on_progress=second_step,
            should_stop=should_stop,
            include_currency=False,
        )
    if candidates and not prices and not (should_stop is not None and should_stop()):
        msg = "No price history arrived for any title that passed the preset."
        raise EmptyUniverseError(msg, detail=f"angefragt={len(candidates)}")

    enriched = enrich_with_technicals(base, prices)
    # Ohne eine einzige Kursreihe - keine Kandidaten oder abgebrochen -
    # haengt enrich nichts an. Die Kursfilter faenden ihre Spalten nicht
    # und meldeten ein unbekanntes Feld. Leere Spalten sagen die Wahrheit:
    # fuer diese Titel ist nichts bekannt, und kein Filter trifft sie.
    missing = [column for column in TECHNICAL_COLUMNS if column not in enriched.columns]
    if missing:
        enriched = enriched.reindex(columns=[*enriched.columns, *missing])
    return enriched


def _fundamental_candidates(
    state: AppState,
    universe: list[str],
    preset: ScreenPreset,
) -> list[str]:
    """Verkleinert ein grosses Fundamentaldaten-Universum bei Finviz.

    Finviz dient nur als grobe Vorauswahl. Die lokalen, aus Statements
    berechneten Praedikate bleiben danach verbindlich. Ist die Quelle nicht
    erreichbar oder liefert sie leer, bleibt das vollstaendige Universum der
    sichere Fallback.

    Args:
        state: UI-Zustand mit optionalem Screener-Provider.
        universe: Vollstaendiger Indexbestand.
        preset: Fundamentales Preset mit Remote-Filtern.

    Returns:
        Kandidaten in der Reihenfolge des Ausgangsuniversums.
    """
    if (
        state.is_offline
        or len(universe) < _REMOTE_PREFILTER_MIN
        or state.screener is None
        or not preset.request.finviz_filters
    ):
        return universe

    request = preset.request.model_copy(
        update={
            "universe": universe,
            "local_filters": None,
            "sort_by": None,
            "limit": len(universe),
        }
    )
    try:
        remote = state.screening_engine().screen(request)
    except BTLiteError as exc:
        _LOG.warning("Finviz-Vorauswahl nicht verfuegbar, volles Universum: %s", exc.detail)
        return universe

    selected = {str(ticker) for ticker in remote.data.index}
    candidates = [ticker for ticker in universe if ticker in selected]
    if not candidates:
        _LOG.warning("Finviz-Vorauswahl war leer, volles Universum bleibt als Fallback")
        return universe
    _LOG.info("Finviz-Vorauswahl: %d von %d Titeln", len(candidates), len(universe))
    return candidates


def run_screen(
    state: AppState,
    preset_key: str,
    universe: list[str],
    on_progress: ProgressCallback = None,
    should_stop: StopCheck = None,
    refinement: TechnicalRefinement | None = None,
) -> ScreenResult:
    """Fuehrt einen Screen gegen das passende Universum aus.

    Die Finviz-Filter des Presets bleiben aussen vor: sie gehoeren zur
    Remote-Auswertung und sind nur mit erreichbarer Quelle sinnvoll. Die
    lokalen Praedikate laufen auf gecachten Kennzahlen beziehungsweise auf
    den aus den Kursreihen gerechneten technischen Spalten.

    Args:
        state: UI-Zustand.
        preset_key: Key des Presets, fundamental oder technisch.
        universe: Startuniversum als Symbolliste.
        on_progress: Optionaler Rueckruf ``(erledigt, gesamt)``.
        should_stop: Optionales Praedikat. Liefert es True, endet der
            Lauf am naechsten Symbol mit dem bis dahin Gesammelten. Der Lauf
            dauert bei einem grossen Universum Minuten; ohne Rueckmeldung
            haelt der Nutzer die Seite fuer haengengeblieben.
        refinement: Zusatzfilter, per UND zu den Filtern des Presets. Bei
            einem fundamentalen Preset loesen sie das Laden der
            Kursreihen aus.

    Returns:
        Das Screening-Ergebnis.

    Raises:
        BTLiteError: Wenn Preset, Universum oder Engine scheitern.
    """
    from btlite.screening.refine import refine_request

    if should_stop is not None and should_stop():
        raise EmptyUniverseError(
            "Screening was cancelled before data was loaded.",
            detail=f"angefragt={len(universe)}",
        )

    preset = resolve_preset(preset_key)
    if is_technical_preset(preset_key):
        frame = _technical_universe(state, universe, on_progress, should_stop)
    else:
        candidates = _fundamental_candidates(state, universe, preset)
        if _needs_second_step(preset_key, refinement):
            frame = _fundamentals_with_prices(
                state, candidates, preset, on_progress, should_stop
            )
        else:
            frame = state.ratio_universe(
                candidates,
                period=Period.ANNUAL,
                on_progress=on_progress,
                should_stop=should_stop,
            )
    engine = state.screening_engine()

    request = refine_request(preset.request, refinement).model_copy(
        update={"finviz_filters": {}, "universe": [str(t) for t in frame.index]}
    )
    result: ScreenResult = engine.screen(request, universe=frame)
    return result


def render_result(
    state: AppState,
    preset_key: str,
    universe: list[str],
    on_progress: ProgressCallback = None,
    should_stop: StopCheck = None,
    refinement: TechnicalRefinement | None = None,
) -> html.Div:
    """Baut Filteranzeige und Ergebnistabelle.

    Args:
        state: UI-Zustand.
        preset_key: Key des gewaehlten Presets.
        universe: Startuniversum.
        on_progress: Optionaler Rueckruf ``(erledigt, gesamt)``.
        should_stop: Optionales Praedikat. Liefert es True, endet der
            Lauf am naechsten Symbol mit dem bis dahin Gesammelten.
        refinement: Zusatzfilter, per UND zu den Filtern des Presets.

    Returns:
        Container mit Filterbeschreibung und Treffern, oder ein
        Fehlerbanner.
    """
    if not universe:
        # Vor jedem Abruf abgefangen: ohne Titel gibt es nichts zu holen
        # und nichts zu filtern. Ohne diesen Zweig liefe der leere Frame
        # in die Filter-Pruefung und meldete eine fehlende Spalte.
        return c.placeholder("No universe: without titles there is nothing to filter.")

    try:
        preset = resolve_preset(preset_key)
        result = run_screen(state, preset_key, universe, on_progress, should_stop, refinement)
    except EmptyUniverseError:
        # Kein Titel lieferte Daten. Das ist kein Fehler des Nutzers und
        # auch keiner der Filter - ohne diesen Zweig meldete die
        # Filter-Pruefung "Field 'pe' does not exist in the universe",
        # was niemandem sagt, was los ist.
        if should_stop is not None and should_stop():
            # Abgebrochen, bevor etwas ankam. "Quelle unerreichbar" waere
            # hier falsch - der Nutzer hat selbst gestoppt.
            return c.placeholder("Screening cancelled before any title was checked.")
        return c.placeholder(
            f"No data arrived for any of the {len(universe)} titles. "
            "The source may be unreachable, or its rate limit stopped the run. "
            "Try again in a few minutes, or pick a smaller universe."
        )
    except BTLiteError as exc:
        return html.Div([c.error_banner(exc)])

    technical = is_technical_preset(preset_key)
    refined = refinement is not None and refinement.is_active
    blocks: list[Any] = [
        c.section(
            "Preset",
            _preset_description(preset, technical=technical, refinement=refinement),
        ),
        c.provenance_note(result.provenance, state.timezone),
        c.section(
            "Matches",
            *_result_children(result, technical=technical, refined=refined, refinement=refinement),
        ),
    ]
    if technical or refined:
        blocks.append(c.section("Column reference", _column_help(result)))
    return html.Div(blocks, className="bt-eqs-body")


def progress_view(
    done: int,
    total: int,
    preset_key: str,
    titles: int | None = None,
    seconds_per_unit: float | None = None,
) -> html.Div:
    """Zeigt den Stand eines laufenden Screenings.

    Bewusst mit Zahlen statt nur einem Balken: "137 von 503" sagt, dass
    etwas vorangeht, ein Balken allein bei langsamem Fortschritt nicht.
    Genau daran ist der erste Anlauf gescheitert - der Nutzer hielt die
    Seite fuer haengengeblieben und brach ab, waehrend sie rechnete.

    Args:
        done: Bereits gepruefte Titel.
        total: Titel insgesamt. 0 heisst "noch unbekannt".
        preset_key: Gewaehltes Preset, fuer die Restzeit-Schaetzung.
        titles: Titelzahl des Universums. Liegt ``total`` darueber, laeuft
            der zweite Schritt eines fundamentalen Screens mit
            Kursfiltern: Kurse fuer die Treffer des Presets.
        seconds_per_unit: Gemessenes Tempo des laufenden Schritts in
            Sekunden je Titel, oder None solange zu wenig gemessen ist.
            Gemessen schlaegt geschaetzt: ob ein Titel aus dem Cache kommt
            oder aus dem Netz, weiss nur der laufende Job.

    Returns:
        Fortschrittsanzeige.
    """
    if total <= 0:
        return c.placeholder("Screening starts...")

    if titles is not None and 0 < titles < total:
        # Zweiter Schritt: nur dessen eigener Stand zaehlt. Sonst stuende
        # "520 of 540 titles" da, obwohl der Index 503 hat.
        done, total = max(0, done - titles), total - titles
        label = "Price history for the preset matches"
        noun = "matches"
    else:
        label = "Screening"
        noun = "titles"

    share = min(1.0, max(0.0, done / total))
    remaining = max(0, total - done)
    left = _remaining_text(remaining, seconds_per_unit)

    return html.Div(
        [
            html.Div(f"{label} {done} of {total} {noun} - {left}", className="bt-progress-text"),
            html.Div(
                html.Div(
                    className="bt-progress-fill",
                    style={"width": f"{share * 100:.0f}%"},
                ),
                className="bt-progress-track",
                # role=presentation: der Balken ist Beiwerk, die Zahl
                # darueber traegt die Aussage. Screenreader lesen den Text
                # und ueberspringen die beiden Divs.
                role="presentation",
            ),
        ],
        className="bt-progress",
    )


def _column_help(result: ScreenResult) -> html.Div:
    """Erklaert die technischen Spalten der Trefferliste.

    Die Texte kommen aus ``screening.technicals.describe_columns`` - dort
    stehen sie samt Einheit, hier werden sie nur angezeigt.

    Args:
        result: Screening-Ergebnis, dessen Spalten erklaert werden.

    Returns:
        Aufklappbare Liste der erklaerten Spalten.
    """
    from btlite.screening.technicals import describe_columns

    descriptions = describe_columns()
    shown = [str(col) for col in result.data.columns if str(col) in descriptions]
    if not shown:
        return c.placeholder("No technical columns in the match list.")

    rows = [
        html.Tr(
            [
                html.Td(_COLUMN_HEADERS.get(name, name)),
                html.Td(descriptions[name]),
            ]
        )
        for name in _ordered_columns(shown, _TECHNICAL_RESULT_COLUMNS)
    ]
    return html.Div(
        html.Details(
            [
                html.Summary("What the columns mean", className="bt-subhead"),
                html.Table(
                    [
                        html.Thead(html.Tr([html.Th("Column"), html.Th("Meaning")])),
                        html.Tbody(rows),
                    ],
                    className="bt-table",
                ),
            ],
            className="bt-ratio-group",
        ),
        className="bt-ratio-groups",
    )


def _ordered_columns(names: list[str], preferred: tuple[str, ...]) -> list[str]:
    """Sortiert Spaltennamen nach einer Wunschreihenfolge.

    Args:
        names: Vorhandene Spaltennamen.
        preferred: Gewuenschte Reihenfolge. Nicht genannte Namen folgen
            hinten in ihrer urspruenglichen Ordnung.

    Returns:
        Sortierte Namensliste.
    """
    front = [name for name in preferred if name in names]
    rest = [name for name in names if name not in set(front)]
    return [*front, *rest]


def _preset_options() -> list[dict[str, str]]:
    """Baut die Auswahlliste des Preset-Dropdowns.

    Beide Arten stehen nebeneinander. Das Label nennt die Gruppe und sagt,
    ob das Preset eine Verbindung braucht - der Nutzer soll das nicht
    raten muessen.

    Returns:
        Optionen fuer ``dcc.Dropdown``.
    """
    from btlite.screening.presets import all_presets
    from btlite.screening.technical_presets import all_technical_presets

    options: list[dict[str, str]] = [
        {"label": f"Fundamental - {preset.label}{_network_hint(preset)}", "value": preset.key}
        for preset in all_presets()
    ]
    options.extend(
        {"label": f"Technical - {preset.label}{_network_hint(preset)}", "value": preset.key}
        for preset in all_technical_presets()
    )
    return options


def _network_hint(preset: ScreenPreset) -> str:
    """Kennzeichnet Presets, die ohne Verbindung unvollstaendig bleiben.

    Args:
        preset: Das Preset.

    Returns:
        Leerer String bei rein lokalen Presets, sonst ein Hinweis.
    """
    return " (needs live source)" if preset.request.finviz_filters else " (local)"


def _preset_description(
    preset: ScreenPreset,
    *,
    technical: bool,
    refinement: TechnicalRefinement | None = None,
) -> html.Div:
    """Beschreibt die Filter eines Presets im Klartext.

    Args:
        preset: Das gewaehlte Preset.
        technical: Ob das Preset auf Kursspalten arbeitet.
        refinement: Zusatzfilter. Aktive erscheinen in einer eigenen
            Zeile - wer eine Trefferliste sieht, muss wissen, woraus sie
            entstand.

    Returns:
        Beschreibungs-Container.
    """
    from btlite.screening.filters import describe_group

    kind = "Price based" if technical else "Fundamentals based"
    basis = (
        "Computed from the price series of the source, runs entirely locally."
        if technical
        else "Computed from the fundamental ratios of the source."
    )
    children: list[Any] = [
        html.Div(f"{kind}. {basis}", className="bt-notice"),
        html.P(preset.description, className="bt-text"),
    ]

    local = preset.request.local_filters
    children.append(
        html.Div(
            [
                html.Strong("Local filters: "),
                html.Code(describe_group(local) if local is not None else "none"),
            ],
            className="bt-filter-line",
        )
    )

    from btlite.screening.refine import refinement_group

    extra = refinement_group(refinement) if refinement is not None else None
    if extra is not None:
        children.append(
            html.Div(
                [
                    html.Strong("Extra price filters (AND): "),
                    html.Code(describe_group(extra)),
                ],
                className="bt-filter-line",
            )
        )

    remote = preset.request.finviz_filters
    if remote:
        pairs = ", ".join(f"{k} = {v}" for k, v in remote.items())
        children.append(
            html.Div(
                [
                    html.Strong("Finviz filters (live source only): "),
                    html.Code(pairs),
                ],
                className="bt-filter-line",
            )
        )
    children.append(
        html.Div(
            f"Sort: {preset.request.sort_by or 'none'}"
            f" ({'ascending' if preset.request.ascending else 'descending'}),"
            f" limit {preset.request.limit}.",
            className="bt-notice",
        )
    )
    return html.Div(children, className="bt-preset")


def _result_children(
    result: ScreenResult,
    *,
    technical: bool = False,
    refined: bool = False,
    refinement: TechnicalRefinement | None = None,
) -> list[Any]:
    """Baut Kopfzeile und Tabelle der Trefferliste.

    Args:
        result: Screening-Ergebnis.
        technical: Ob die technischen Spalten nach vorne sollen.
        refined: Ob Zusatzfilter aktiv waren. Bei einem fundamentalen
            Preset erscheinen dann Kurs, Abstaende und RSI direkt hinter
            Name und Sektor; die uebrigen 15 Kursspalten bleiben der
            Tabelle fern und stehen im CSV-Export.
        refinement: Die Zusatzfilter selbst. Gewaehlte Setups erscheinen
            als Spalten hinter den Kursspalten.

    Returns:
        Liste der Kindelemente des Abschnitts.
    """
    header = html.Div(
        f"Showing {len(result.data)} of {result.total_matches} matches"
        + (" (truncated)" if result.is_truncated else ""),
        className="bt-notice",
    )
    if result.data.empty:
        return [header, c.placeholder("No title matches the filters.")]

    from btlite.screening.refine import refinement_columns
    from btlite.screening.technicals import TECHNICAL_COLUMNS

    shown_extra = refinement_columns(refinement)

    hidden: set[str] = set()
    if technical:
        preferred = _TECHNICAL_RESULT_COLUMNS
    elif refined:
        preferred = (*_RESULT_COLUMNS[:2], *shown_extra, *_RESULT_COLUMNS[2:])
        hidden = set(TECHNICAL_COLUMNS) - set(shown_extra)
    else:
        preferred = _RESULT_COLUMNS
    columns = [col for col in preferred if col in result.data.columns]
    extra = [
        str(col)
        for col in result.data.columns
        if str(col) not in set(columns) and str(col) not in hidden
    ]
    frame = result.data[[*columns, *extra]]

    return [
        header,
        c.data_table(
            frame,
            index_label="Ticker",
            formatters=_formatters(frame),
            headers=_COLUMN_HEADERS,
            index_cell=_ticker_cell,
        ),
    ]


def _ticker_cell(ticker: str) -> list[Any]:
    """Baut die Tickerzelle der Trefferliste mit Links auf die Auswertungen.

    Aus dem Treffer fuehrt ein Klick direkt zu Chart, Steckbrief oder
    Finanzanalyse des Titels. Die Links sind ``dcc.Link`` - echte
    ``<a>``-Elemente, per Tab erreichbar, clientseitig navigiert. Die
    Adressen entstehen ueber denselben Adressbau wie Kommandozeile und
    Namenssuche (:func:`~btlite.ui.search_bar.analysis_href`).

    Die Kommandozeile bleibt dabei unberuehrt, wie bei Navigation und
    Namenssuche: sie ist nach jedem ausgefuehrten Kommando leer, und die
    Navigationslinks ziehen den neuen Ticker aus der Adresse nach.

    Args:
        ticker: Kanonisches Symbol aus dem Index des Ergebnisses.

    Returns:
        Kindelemente der Zelle: der Ticker als Text, danach die Links.
    """
    from btlite.ui.command_bar import NAV_LABELS
    from btlite.ui.search_bar import analysis_href

    return [
        html.Span(ticker, className="bt-eqs-ticker"),
        *[
            dcc.Link(
                NAV_LABELS[mnemonic],
                href=analysis_href(mnemonic, ticker),
                title=f"Open the {subject} of {ticker}",
                className="bt-eqs-link",
            )
            for mnemonic, subject in _RESULT_LINKS
        ],
    ]


def _formatters(frame: Any) -> dict[str, Any]:
    """Waehlt je Spalte die passende Formatierung.

    Args:
        frame: Ergebnis-Frame.

    Returns:
        Abbildung Spaltenname auf Formatierungsfunktion.
    """
    specs: dict[str, Any] = {}
    for column in frame.columns:
        name = str(column)
        if name in _PERCENT_COLUMNS or name in _TECHNICAL_PERCENT_COLUMNS:
            specs[name] = _percent_cell
        elif name in _COMPACT_COLUMNS:
            specs[name] = _compact_cell
        elif name in _FLAG_COLUMNS:
            specs[name] = _flag_cell
        elif name in _SCALE_COLUMNS:
            specs[name] = _scale_cell
    return specs


def _flag_cell(raw: object) -> str:
    """Formatiert einen 1.0/0.0-Schalter als Ja oder Nein.

    Args:
        raw: Rohwert aus dem Frame.

    Returns:
        "ja", "nein", oder "n/a" bei fehlendem Wert.
    """
    if not isinstance(raw, int | float) or isinstance(raw, bool) or fmt.is_missing(float(raw)):
        return fmt.NA_TEXT
    return "yes" if float(raw) > 0.0 else "no"


def _scale_cell(raw: object) -> str:
    """Formatiert einen Wert auf fester Skala, etwa den RSI von 0 bis 100.

    Args:
        raw: Rohwert aus dem Frame.

    Returns:
        Zahl mit einer Nachkommastelle, oder "n/a".
    """
    if not isinstance(raw, int | float) or isinstance(raw, bool):
        return fmt.NA_TEXT
    return fmt.format_number(float(raw), decimals=1)


def _percent_cell(raw: object) -> str:
    """Formatiert einen dezimalen Anteil als Prozentwert.

    Args:
        raw: Rohwert aus dem Frame.

    Returns:
        Anzeigetext, "n/a" bei fehlendem Wert.
    """
    if not isinstance(raw, int | float) or isinstance(raw, bool):
        return fmt.NA_TEXT
    return fmt.format_percent(float(raw))


def _compact_cell(raw: object) -> str:
    """Formatiert eine grosse Zahl gekuerzt.

    Args:
        raw: Rohwert aus dem Frame.

    Returns:
        Anzeigetext, "n/a" bei fehlendem Wert.
    """
    if not isinstance(raw, int | float) or isinstance(raw, bool):
        return fmt.NA_TEXT
    return fmt.format_compact(float(raw))
