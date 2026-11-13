"""Primaere Backtest-Engine auf Basis von vectorbt.

``vectorbt`` ist ein optionales Extra (``pip install btlite[backtest]``) und
wird deshalb **lazy** importiert: der Import passiert erst in
:meth:`VectorbtEngine.is_available` beziehungsweise :meth:`VectorbtEngine.run`,
nicht auf Modulebene. Damit bleibt ``btlite`` importierbar, auch wenn das
enge Versions-Kartell aus pandas, numpy und numba an einer Stelle bricht.

Kostenkonvention des Projekts: ``fees`` und ``slippage`` sind dezimale
Anteile des Ordervolumens und wirken bei Ein- **und** Ausstieg. Genau so
interpretiert vectorbt die Parameter ``fees`` und ``slippage``, es ist also
keine Umrechnung noetig.

Zwei Details der Bibliothek, die das Ergebnis sonst still verfaelschen:

* ``freq=`` muss gesetzt sein. Ohne die Frequenz kann vectorbt nicht
  annualisieren; Sharpe und verwandte Kennzahlen werden falsch oder NaN.
  Dieses Modul leitet ``freq`` aus dem ``Interval`` des Request ab.
* Die Auswertung sind **Methoden** (``pf.value()``), nur
  ``pf.trades.records_readable`` ist eine Property.

Die Kennzahlen selbst berechnet dieses Modul **nicht** mit vectorbt,
sondern ueber :mod:`btlite.backtest.metrics`. Nur so sind die Zahlen mit
denen des backtrader-Adapters vergleichbar.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Final

import pandas as pd

from btlite.backtest import metrics as metrics_mod
from btlite.core.errors import (
    BacktestError,
    EngineUnavailableError,
    InsufficientDataError,
)
from btlite.core.logging import get_logger
from btlite.models.backtest import (
    BacktestResult,
    Direction,
    PerformanceMetrics,
    Trade,
    TradeSide,
)
from btlite.models.common import Interval

if TYPE_CHECKING:
    from btlite.models.backtest import BacktestRequest, SignalSet
    from btlite.models.market import PriceSeries

_log = get_logger(__name__)

#: Name der Engine, erscheint in ``BacktestResult.engine_used``.
ENGINE_NAME: Final[str] = "vectorbt"

#: Pflichtspalte der Equity-Kurve laut ``BacktestResult``.
EQUITY_COLUMN: Final[str] = "equity"

#: Optionale Spalten, die dieses Modul zusaetzlich fuellt. ``viz/backtest.py``
#: nutzt sie, wenn sie vorhanden sind.
CASH_COLUMN: Final[str] = "cash"
DRAWDOWN_COLUMN: Final[str] = "drawdown"
BENCHMARK_COLUMN: Final[str] = "benchmark_equity"

#: Mindestanzahl Bars, unter der ein Backtest keine Aussage hat.
MIN_BARS: Final[int] = 2

#: Uebersetzung der Projekt-Richtung auf den vectorbt-Parameter ``direction``.
_DIRECTION_MAP: Final[dict[Direction, str]] = {
    Direction.LONG_ONLY: "longonly",
    Direction.SHORT_ONLY: "shortonly",
    Direction.LONG_SHORT: "both",
}

#: Pandas-Offset-Alias je Bar-Intervall fuer den ``freq``-Parameter.
_FREQ_MAP: Final[dict[Interval, str]] = {
    Interval.M1: "1min",
    Interval.M5: "5min",
    Interval.M15: "15min",
    Interval.M30: "30min",
    Interval.H1: "1h",
    Interval.D1: "1D",
    Interval.W1: "1W",
    Interval.MO1: "30D",
}

#: Spaltennamen in ``pf.trades.records_readable``, gegen die dieses Modul
#: mappt. Bewusst als Konstanten, nicht als Literale im Code.
_TR_COLUMN = "Column"
_TR_SIZE = "Size"
_TR_ENTRY_TIME = "Entry Timestamp"
_TR_ENTRY_PRICE = "Avg Entry Price"
_TR_ENTRY_FEES = "Entry Fees"
_TR_EXIT_TIME = "Exit Timestamp"
_TR_EXIT_PRICE = "Avg Exit Price"
_TR_EXIT_FEES = "Exit Fees"
_TR_PNL = "PnL"
_TR_RETURN = "Return"
_TR_DIRECTION = "Direction"
_TR_STATUS = "Status"

#: Wert der Status-Spalte fuer noch offene Positionen.
_STATUS_OPEN = "Open"

#: Wert der Richtungs-Spalte fuer Short-Positionen.
_DIRECTION_SHORT = "Short"


class VectorbtEngine:
    """Fuehrt Signale mit ``vectorbt`` aus.

    Erfuellt das Protocol :class:`~btlite.backtest.base.BacktestEngine`. Die
    Engine holt **keine** Daten selbst - Kurse und Benchmark werden
    injiziert.
    """

    def __init__(self, *, risk_free_rate: float = 0.0) -> None:
        """Initialisiert die Engine.

        Args:
            risk_free_rate: Risikofreier Zins fuer Sharpe, Sortino und
                Alpha, annualisiert und dezimal. Default 0.0, damit die
                Kennzahl ohne stille Annahme reproduzierbar ist.
        """
        self.risk_free_rate = risk_free_rate

    @property
    def name(self) -> str:
        """Bezeichner der Engine.

        Returns:
            ``"vectorbt"``.
        """
        return ENGINE_NAME

    def is_available(self) -> bool:
        """Prueft lazy, ob ``vectorbt`` importierbar ist.

        Gibt bei einem Fehlschlag False zurueck und laesst die Exception
        nicht nach oben: ein fehlendes Extra ist kein Absturzgrund, das UI
        blendet die Engine dann nur aus. Auch ein ``ImportError`` aus einer
        inkompatiblen Transitivabhaengigkeit (etwa Plotly 7 ohne
        ``scattermapbox``) faellt hierunter.

        Returns:
            True wenn die Engine nutzbar ist.
        """
        try:
            if find_spec("vectorbt") is None:
                return False
            self._import_vbt()
        except Exception as exc:
            _log.debug("vectorbt nicht verfuegbar: %s", exc)
            return False
        return True

    def _import_vbt(self) -> Any:
        """Importiert ``vectorbt`` und gibt das Modul zurueck.

        Returns:
            Das Modul ``vectorbt``.

        Raises:
            ImportError: Wenn der Import scheitert.
        """
        import vectorbt as vbt

        return vbt

    def run(
        self,
        request: BacktestRequest,
        prices: dict[str, PriceSeries],
        signals: SignalSet,
        *,
        benchmark: PriceSeries | None = None,
    ) -> BacktestResult:
        """Fuehrt den Backtest aus.

        Args:
            request: Parameter des Laufs inklusive Kosten und Startkapital.
            prices: Kursreihen je Ticker, bereits auf den Zeitraum
                beschnitten.
            signals: Ein- und Ausstiegssignale der Strategie.
            benchmark: Optionale Vergleichsreihe fuer Beta und Alpha.

        Returns:
            Vollstaendiges Ergebnis mit Equity-Kurve, Trades und Metriken.
            Die Equity-Kurve traegt die Spalten ``equity``, ``cash``,
            ``drawdown`` und - falls eine Benchmark vorliegt -
            ``benchmark_equity``.

        Raises:
            EngineUnavailableError: ``vectorbt`` ist nicht installiert.
            InsufficientDataError: Zu wenige gemeinsame Bars.
            BacktestError: Die Ausfuehrung ist fehlgeschlagen.
        """
        if not self.is_available():
            msg = "The backtest engine vectorbt is not installed"
            raise EngineUnavailableError(msg, detail="pip install 'btlite[backtest]'")
        vbt = self._import_vbt()

        warnings: list[str] = []
        close = self._close_panel(prices, signals, warnings=warnings)
        entries, exits = self._align_signals(signals, close)

        portfolio = self._build_portfolio(
            vbt, request=request, close=close, entries=entries, exits=exits
        )

        equity = self._series(portfolio.value())
        cash = self._series(portfolio.cash())
        trades = self._extract_trades(portfolio, close=close)
        exposure = self._exposure(portfolio, index=equity.index)

        bench_equity = self._benchmark_equity(
            benchmark, index=equity.index, init_cash=request.init_cash, warnings=warnings
        )

        curve = pd.DataFrame(
            {
                EQUITY_COLUMN: equity,
                CASH_COLUMN: cash.reindex(equity.index),
                DRAWDOWN_COLUMN: metrics_mod.drawdown_series(equity),
            }
        )
        if bench_equity is not None:
            curve[BENCHMARK_COLUMN] = bench_equity

        strategy_metrics = metrics_mod.compute_metrics(
            equity,
            trades=trades,
            benchmark_equity=bench_equity,
            interval=request.interval,
            risk_free_rate=self.risk_free_rate,
            exposure=exposure,
        )
        bench_metrics = self._benchmark_metrics(bench_equity, interval=request.interval)

        provenance = next(iter(prices.values())).provenance

        return BacktestResult(
            request=request,
            equity_curve=curve,
            trades=trades,
            metrics=strategy_metrics,
            benchmark_metrics=bench_metrics,
            signals=signals,
            engine_used=self.name,
            # Siehe bt_engine: die Waehrung kommt aus den Kursreihen.
            currency=metrics_mod.currency_of(prices),
            provenance=provenance,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Eingangsdaten
    # ------------------------------------------------------------------

    def _close_panel(
        self,
        prices: dict[str, PriceSeries],
        signals: SignalSet,
        *,
        warnings: list[str],
    ) -> pd.DataFrame:
        """Baut das Schlusskurs-Panel passend zum Signal-Index.

        Args:
            prices: Kursreihen je Ticker.
            signals: Signale, deren Spalten die zu handelnden Ticker
                festlegen.
            warnings: Liste, in die nicht-fatale Hinweise wandern.

        Returns:
            Schlusskurse mit einer Spalte je Signal-Spalte, auf den
            gemeinsamen Index von Kursen und Signalen beschnitten.

        Raises:
            InsufficientDataError: Wenn weniger als zwei gemeinsame Bars
                bleiben oder kein Ticker Kursdaten hat.
            BacktestError: Wenn zu keiner Signal-Spalte Kurse vorliegen.
        """
        wanted = [str(c) for c in signals.entries.columns]
        available = {t: s for t, s in prices.items() if t in set(wanted)}

        missing = sorted(set(wanted) - set(available))
        if missing:
            warnings.append(f"Skipped, no price data: {', '.join(missing)}")
        usable = [t for t in wanted if t in available]
        if not usable:
            msg = "No price data is available for any signal ticker"
            raise BacktestError(msg, detail=f"signale={wanted}, kurse={sorted(prices)}")

        columns = {
            ticker: pd.to_numeric(available[ticker].data["close"], errors="coerce").astype(float)
            for ticker in usable
        }
        panel = pd.DataFrame(columns).sort_index()
        panel = panel.loc[panel.index.intersection(signals.entries.index)].ffill().dropna()

        if len(panel) < MIN_BARS:
            msg = "Too few common bars for a backtest"
            raise InsufficientDataError(
                msg,
                detail=f"bars={len(panel)}",
                required=MIN_BARS,
                available=len(panel),
            )
        return panel

    def _align_signals(
        self, signals: SignalSet, close: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Verschiebt die Signale um den Execution-Lag und richtet sie aus.

        Der Lag ist der Kern der Look-ahead-Vermeidung auf Engine-Seite: ein
        Signal am Bar t wird erst am Bar ``t + execution_lag`` ausgefuehrt.
        Signale, die dadurch hinter das Reihenende rutschen, fallen weg -
        sie waeren nicht handelbar gewesen.

        Args:
            signals: Signale der Strategie.
            close: Schlusskurs-Panel, das Index und Spalten vorgibt.

        Returns:
            Ein- und Ausstiegsmatrix, boolesch, deckungsgleich mit ``close``.
        """
        lag = signals.execution_lag

        def prepare(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.reindex(index=close.index, columns=close.columns).fillna(False)
            out = out.astype(bool)
            if lag > 0:
                out = out.shift(lag, fill_value=False)
            return out.astype(bool)

        return prepare(signals.entries), prepare(signals.exits)

    def _build_portfolio(
        self,
        vbt: Any,
        *,
        request: BacktestRequest,
        close: pd.DataFrame,
        entries: pd.DataFrame,
        exits: pd.DataFrame,
    ) -> Any:
        """Ruft ``Portfolio.from_signals`` mit den Projektkonventionen.

        Args:
            vbt: Das importierte ``vectorbt``-Modul.
            request: Parameter des Laufs.
            close: Schlusskurs-Panel.
            entries: Einstiegsmatrix nach Lag.
            exits: Ausstiegsmatrix nach Lag.

        Returns:
            Das ``vectorbt``-Portfolio-Objekt.

        Raises:
            BacktestError: Wenn vectorbt die Ausfuehrung abbricht.
        """
        direction = _DIRECTION_MAP[request.strategy.direction]
        freq = _FREQ_MAP[request.interval]
        try:
            # close ist POSITIONAL das erste Argument. group_by und
            # cash_sharing machen aus mehreren Spalten EIN Portfolio mit
            # gemeinsamer Kasse - sonst bekaeme jeder Ticker das volle
            # Startkapital und die Equity-Kurve waere ein Vielfaches davon.
            return vbt.Portfolio.from_signals(
                close,
                entries,
                exits,
                direction=direction,
                init_cash=request.init_cash,
                fees=request.fees,
                slippage=request.slippage,
                freq=freq,
                group_by=True,
                cash_sharing=True,
            )
        except Exception as exc:
            msg = "The backtest could not be executed"
            raise BacktestError(msg, detail=f"{type(exc).__name__}: {exc}") from exc

    # ------------------------------------------------------------------
    # Auswertung
    # ------------------------------------------------------------------

    def _series(self, value: Any) -> pd.Series:
        """Bringt eine vectorbt-Auswertung auf eine float-Serie.

        Bei gruppierten Portfolios liefert vectorbt eine Serie; ein
        DataFrame mit genau einer Spalte wird auf diese reduziert.

        Args:
            value: Rueckgabe einer Portfolio-Methode.

        Returns:
            Float-Serie mit UTC-``DatetimeIndex``.

        Raises:
            BacktestError: Wenn die Form unerwartet ist.
        """
        if isinstance(value, pd.DataFrame):
            if value.shape[1] != 1:
                msg = "Unexpected shape of the portfolio evaluation"
                raise BacktestError(msg, detail=f"spalten={list(value.columns)}")
            value = value.iloc[:, 0]
        if not isinstance(value, pd.Series):
            msg = "Unexpected type of the portfolio evaluation"
            raise BacktestError(msg, detail=f"typ={type(value).__name__}")
        out = value.astype(float)
        out.name = None
        return out

    def _exposure(self, portfolio: Any, *, index: pd.Index) -> float | None:
        """Anteil der Bars mit offener Position.

        Args:
            portfolio: Das vectorbt-Portfolio.
            index: Index der Equity-Kurve.

        Returns:
            Exposure, dezimal zwischen 0.0 und 1.0. None wenn vectorbt
            keinen Positionswert liefert.
        """
        try:
            asset_value = self._series(portfolio.asset_value())
        except (BacktestError, AttributeError):
            return None
        mask = asset_value.reindex(index).fillna(0.0).abs() > 1e-9
        return metrics_mod.exposure_from_mask(mask)

    def _extract_trades(self, portfolio: Any, *, close: pd.DataFrame) -> list[Trade]:
        """Uebersetzt die vectorbt-Trade-Records in den Projektvertrag.

        ``records_readable`` ist eine **Property**, keine Methode. Offene
        Positionen tragen ``Status == "Open"``; ihr ``exit_time`` und
        ``exit_price`` bleiben None, damit ``Trade.is_open`` greift und die
        Trade-Statistik sie nicht als Ergebnis zaehlt.

        Args:
            portfolio: Das vectorbt-Portfolio.
            close: Schlusskurs-Panel, liefert den Bar-Index fuer die
                Haltedauer.

        Returns:
            Trades in chronologischer Reihenfolge des Einstiegs.
        """
        records = portfolio.trades.records_readable
        if records is None or records.empty:
            return []

        positions = {timestamp: i for i, timestamp in enumerate(close.index)}
        out: list[Trade] = []

        for row in records.to_dict(orient="records"):
            trade = self._to_trade(row, positions=positions)
            if trade is not None:
                out.append(trade)

        out.sort(key=lambda t: (t.entry_time, t.ticker))
        return out

    def _to_trade(self, row: dict[str, Any], *, positions: dict[Any, int]) -> Trade | None:
        """Baut einen einzelnen :class:`Trade` aus einem Record.

        Args:
            row: Eine Zeile aus ``records_readable`` als Dictionary.
            positions: Zuordnung Zeitstempel auf Bar-Position.

        Returns:
            Der Trade, oder None wenn Pflichtfelder fehlen oder unplausibel
            sind (Einstiegskurs oder Groesse nicht positiv). Solche Records
            werden uebersprungen statt den ganzen Lauf zu verwerfen.
        """
        entry_time = row.get(_TR_ENTRY_TIME)
        entry_price = _opt_float(row.get(_TR_ENTRY_PRICE))
        size = _opt_float(row.get(_TR_SIZE))

        if entry_time is None or pd.isna(entry_time):
            return None
        if entry_price is None or entry_price <= 0.0 or size is None or size <= 0.0:
            return None

        is_open = str(row.get(_TR_STATUS, "")) == _STATUS_OPEN
        exit_time = row.get(_TR_EXIT_TIME)
        exit_price = _opt_float(row.get(_TR_EXIT_PRICE))

        if is_open or exit_time is None or pd.isna(exit_time):
            exit_time = None
            exit_price = None
        if exit_price is not None and exit_price <= 0.0:
            exit_price = None

        entry_ts = pd.Timestamp(entry_time)
        exit_ts = None if exit_time is None else pd.Timestamp(exit_time)

        duration: int | None = None
        if exit_ts is not None:
            start = positions.get(entry_ts)
            end = positions.get(exit_ts)
            if start is not None and end is not None:
                duration = max(end - start, 0)

        fees = (_opt_float(row.get(_TR_ENTRY_FEES)) or 0.0) + (
            _opt_float(row.get(_TR_EXIT_FEES)) or 0.0
        )
        side = (
            TradeSide.SHORT
            if str(row.get(_TR_DIRECTION, "")) == _DIRECTION_SHORT
            else TradeSide.LONG
        )

        return Trade(
            ticker=str(row[_TR_COLUMN]),
            side=side,
            entry_time=entry_ts.to_pydatetime(),
            entry_price=entry_price,
            exit_time=None if exit_ts is None else exit_ts.to_pydatetime(),
            exit_price=exit_price,
            size=size,
            pnl=None if exit_ts is None else _opt_float(row.get(_TR_PNL)),
            return_pct=None if exit_ts is None else _opt_float(row.get(_TR_RETURN)),
            fees_paid=max(fees, 0.0),
            duration_bars=duration,
        )

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def _benchmark_equity(
        self,
        benchmark: PriceSeries | None,
        *,
        index: pd.Index,
        init_cash: float,
        warnings: list[str],
    ) -> pd.Series | None:
        """Rechnet die Benchmark in eine Equity-Kurve gleichen Startkapitals.

        Kostenfrei und voll investiert - die Benchmark ist der passive
        Bezugspunkt, nicht eine zweite Strategie.

        Args:
            benchmark: Vergleichsreihe. None liefert None.
            index: Index der Strategie-Equity, auf den ausgerichtet wird.
            init_cash: Startkapital in Kontowaehrung.
            warnings: Liste, in die nicht-fatale Hinweise wandern.

        Returns:
            Benchmark-Equity auf ``index``, oder None wenn keine oder eine
            unbrauchbare Reihe vorliegt.
        """
        if benchmark is None:
            return None

        close = pd.to_numeric(benchmark.data["close"], errors="coerce").astype(float)
        close = close[close > 0.0].dropna()

        # Deckungspruefung VOR dem Vorwaertstragen: eine Reihe mit einem
        # einzigen Kurs wuerde sonst zu einer flachen Kurve gefuellt und
        # lieferte eine Rendite von exakt 0.0 statt "nicht berechenbar".
        covered = close.index.intersection(index)
        if len(covered) < MIN_BARS:
            warnings.append(
                f"Benchmark {benchmark.ticker} does not cover the period - "
                "beta and alpha remain undetermined"
            )
            return None

        aligned = close.reindex(close.index.union(index)).ffill().reindex(index).dropna()
        aligned = aligned[aligned > 0.0]

        if len(aligned) < MIN_BARS:
            warnings.append(
                f"Benchmark {benchmark.ticker} does not cover the period - "
                "beta and alpha remain undetermined"
            )
            return None

        equity = aligned / float(aligned.iloc[0]) * init_cash
        return equity.reindex(index).ffill()

    def _benchmark_metrics(
        self, bench_equity: pd.Series | None, *, interval: Interval
    ) -> PerformanceMetrics | None:
        """Berechnet die Kennzahlen der Benchmark selbst.

        Die Benchmark handelt nicht, deshalb bleiben ihre Trade-Statistiken
        offen; ``exposure`` ist 1.0, da sie durchgehend investiert ist.

        Args:
            bench_equity: Benchmark-Equity, oder None.
            interval: Bar-Intervall fuer die Annualisierung.

        Returns:
            Kennzahlen der Benchmark, oder None.
        """
        if bench_equity is None:
            return None
        return metrics_mod.compute_metrics(
            bench_equity,
            trades=None,
            benchmark_equity=None,
            interval=interval,
            risk_free_rate=self.risk_free_rate,
            exposure=1.0,
        )


def _opt_float(value: Any) -> float | None:
    """Wandelt einen Record-Wert in ``float | None``.

    Args:
        value: Rohwert aus einem vectorbt-Record.

    Returns:
        Der Wert als float, oder None wenn er fehlt oder nicht endlich ist.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if pd.notna(out) and out not in (float("inf"), float("-inf")) else None
