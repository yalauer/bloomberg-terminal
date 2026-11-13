"""Optionaler Zweit-Adapter auf Basis von backtrader.

Zweck dieses Adapters ist nicht, vectorbt zu ersetzen, sondern die
Austauschbarkeit der Engine zu belegen: beide erfuellen dasselbe Protocol
und fuellen dasselbe :class:`~btlite.models.backtest.BacktestResult`. Die
Kennzahlen berechnet auch dieser Adapter ueber
:mod:`btlite.backtest.metrics` und **nicht** ueber die Analyzer von
backtrader - nur dadurch sind die Zahlen mit denen der vectorbt-Engine
vergleichbar.

``backtrader`` ist ein optionales Extra (``pip install btlite[backtrader]``)
und in der Referenzumgebung des Projekts **nicht installiert**. Der Import
ist deshalb lazy: :meth:`BacktraderEngine.is_available` liefert dann False
und :meth:`BacktraderEngine.run` wirft
:class:`~btlite.core.errors.EngineUnavailableError`, statt beim Import des
Moduls abzustuerzen.

Ausfuehrungsmodell des Adapters, bewusst an vectorbt angeglichen:

* Signale kommen bereits um ``SignalSet.execution_lag`` verschoben in den
  Feed, die Order geht also mit dem Kurs des Ausfuehrungsbars ab.
* Kosten wirken bei Ein- und Ausstieg. ``fees`` wird als
  ``commission`` gesetzt, ``slippage`` ueber ``set_slippage_perc``, das
  backtrader auf Kauf und Verkauf anwendet.
* Mehrere Ticker teilen eine Kasse und werden gleichgewichtet auf die Zahl
  der zu handelnden Symbole aufgeteilt.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Final, cast

import pandas as pd

from btlite.backtest import metrics as metrics_mod
from btlite.core.errors import (
    BacktestError,
    EngineUnavailableError,
    InsufficientDataError,
)
from btlite.core.logging import get_logger
from btlite.models.backtest import BacktestResult, Direction, Trade, TradeSide

if TYPE_CHECKING:
    from btlite.models.backtest import BacktestRequest, PerformanceMetrics, SignalSet
    from btlite.models.common import Interval
    from btlite.models.market import PriceSeries

_log = get_logger(__name__)

#: Name der Engine, erscheint in ``BacktestResult.engine_used``.
ENGINE_NAME: Final[str] = "backtrader"

#: Pflichtspalte der Equity-Kurve laut ``BacktestResult``.
EQUITY_COLUMN: Final[str] = "equity"

#: Optionale Spalten, die dieser Adapter zusaetzlich fuellt.
CASH_COLUMN: Final[str] = "cash"
DRAWDOWN_COLUMN: Final[str] = "drawdown"
BENCHMARK_COLUMN: Final[str] = "benchmark_equity"

#: Mindestanzahl Bars, unter der ein Backtest keine Aussage hat.
MIN_BARS: Final[int] = 2

#: Hinweis auf das Installations-Extra, geht als ``detail`` ins Log.
_INSTALL_HINT: Final[str] = "pip install 'btlite[backtrader]'"


class BacktraderEngine:
    """Fuehrt Signale mit ``backtrader`` aus.

    Erfuellt das Protocol :class:`~btlite.backtest.base.BacktestEngine`.
    """

    def __init__(self, *, risk_free_rate: float = 0.0) -> None:
        """Initialisiert den Adapter.

        Args:
            risk_free_rate: Risikofreier Zins fuer Sharpe, Sortino und
                Alpha, annualisiert und dezimal.
        """
        self.risk_free_rate = risk_free_rate

    @property
    def name(self) -> str:
        """Bezeichner der Engine.

        Returns:
            ``"backtrader"``.
        """
        return ENGINE_NAME

    def is_available(self) -> bool:
        """Prueft lazy, ob ``backtrader`` importierbar ist.

        Faengt jede Exception der Importkette und gibt False zurueck: ein
        fehlendes Extra ist kein Absturzgrund, das UI blendet die Engine
        dann aus.

        Returns:
            True wenn die Engine nutzbar ist. In der Referenzumgebung des
            Projekts False, da ``backtrader`` dort nicht installiert ist.
        """
        try:
            if find_spec("backtrader") is None:
                return False
            self._import_bt()
        except Exception as exc:
            _log.debug("backtrader nicht verfuegbar: %s", exc)
            return False
        return True

    def _import_bt(self) -> Any:
        """Importiert ``backtrader`` und gibt das Modul zurueck.

        Returns:
            Das Modul ``backtrader``.

        Raises:
            ImportError: Wenn der Import scheitert.
        """
        import backtrader as bt

        return bt

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

        Raises:
            EngineUnavailableError: ``backtrader`` ist nicht installiert.
                Das ist in der Referenzumgebung des Projekts der Regelfall.
            InsufficientDataError: Zu wenige gemeinsame Bars.
            BacktestError: Die Ausfuehrung ist fehlgeschlagen.
        """
        if not self.is_available():
            msg = "The backtest engine backtrader is not installed"
            raise EngineUnavailableError(msg, detail=_INSTALL_HINT)
        bt = self._import_bt()

        warnings: list[str] = []
        close = self._close_panel(prices, signals, warnings=warnings)
        entries, exits = self._align_signals(signals, close)

        equity, cash, trades = self._run_cerebro(
            bt, request=request, close=close, entries=entries, exits=exits
        )
        exposure = self._exposure(entries, exits)

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

        return BacktestResult(
            request=request,
            equity_curve=curve,
            trades=trades,
            metrics=metrics_mod.compute_metrics(
                equity,
                trades=trades,
                benchmark_equity=bench_equity,
                interval=request.interval,
                risk_free_rate=self.risk_free_rate,
                exposure=exposure,
            ),
            benchmark_metrics=self._benchmark_metrics(bench_equity, interval=request.interval),
            signals=signals,
            engine_used=self.name,
            # Der Backtest rechnet nicht um; er nennt die Waehrung, in der
            # die Eingangsdaten notieren. Ohne das erbte das Ergebnis den
            # Modell-Default und beschriftete einen Dollar-Lauf mit Euro.
            #
            # Ungeprueft: backtrader ist ein optionales Extra und hier
            # nicht installiert, die Tests ueberspringen diese Engine.
            # Die Zeile ist identisch mit der in vbt_engine.py, die von
            # TestResultCurrency abgedeckt ist.
            currency=metrics_mod.currency_of(prices),
            provenance=next(iter(prices.values())).provenance,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Eingangsdaten - identische Semantik wie in der vectorbt-Engine
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
            Schlusskurse mit einer Spalte je handelbarem Ticker.

        Raises:
            InsufficientDataError: Zu wenige gemeinsame Bars.
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

        panel = pd.DataFrame(
            {
                ticker: pd.to_numeric(available[ticker].data["close"], errors="coerce").astype(
                    float
                )
                for ticker in usable
            }
        ).sort_index()
        panel = panel.loc[panel.index.intersection(signals.entries.index)].ffill().dropna()

        if len(panel) < MIN_BARS:
            msg = "Too few common bars for a backtest"
            raise InsufficientDataError(
                msg, detail=f"bars={len(panel)}", required=MIN_BARS, available=len(panel)
            )
        return panel

    def _align_signals(
        self, signals: SignalSet, close: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Verschiebt die Signale um den Execution-Lag und richtet sie aus.

        Args:
            signals: Signale der Strategie.
            close: Schlusskurs-Panel, das Index und Spalten vorgibt.

        Returns:
            Ein- und Ausstiegsmatrix, boolesch, deckungsgleich mit
            ``close``.
        """
        lag = signals.execution_lag

        def prepare(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.reindex(index=close.index, columns=close.columns).fillna(False)
            out = out.astype(bool)
            if lag > 0:
                out = out.shift(lag, fill_value=False)
            return out.astype(bool)

        return prepare(signals.entries), prepare(signals.exits)

    # ------------------------------------------------------------------
    # Ausfuehrung
    # ------------------------------------------------------------------

    def _run_cerebro(
        self,
        bt: Any,
        *,
        request: BacktestRequest,
        close: pd.DataFrame,
        entries: pd.DataFrame,
        exits: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series, list[Trade]]:
        """Baut die Cerebro-Instanz, laesst sie laufen und liest sie aus.

        Die Signale reisen als zwei Zusatzlinien im Feed mit. Damit bleibt
        die Signal-Logik dort, wo sie hingehoert - in ``strategies/`` - und
        die backtrader-Strategie ist ein reiner Ausfuehrer.

        Args:
            bt: Das importierte ``backtrader``-Modul.
            request: Parameter des Laufs.
            close: Schlusskurs-Panel.
            entries: Einstiegsmatrix nach Lag.
            exits: Ausstiegsmatrix nach Lag.

        Returns:
            Equity-Kurve, Kassenbestand und Trade-Liste.

        Raises:
            BacktestError: Wenn backtrader die Ausfuehrung abbricht.
        """
        try:
            feed_cls = _signal_feed_class(bt)
            strategy_cls = _signal_strategy_class(bt)

            cerebro = bt.Cerebro(stdstats=False)
            cerebro.broker.setcash(request.init_cash)
            # Kommission wirkt bei Ein- und Ausstieg je Order.
            cerebro.broker.setcommission(commission=request.fees)
            # Slippage ebenfalls beidseitig.
            cerebro.broker.set_slippage_perc(
                perc=request.slippage, slip_open=True, slip_limit=True, slip_match=True
            )

            allow_short = request.strategy.direction is not Direction.LONG_ONLY
            # backtrader rechnet mit naiven Zeitstempeln; die UTC-Zone wird
            # beim Auslesen in _collect_trades wieder angeheftet.
            naive_index = cast("pd.DatetimeIndex", close.index).tz_convert(None)
            for ticker in close.columns:
                frame = pd.DataFrame(
                    {
                        "open": close[ticker],
                        "high": close[ticker],
                        "low": close[ticker],
                        "close": close[ticker],
                        "volume": 0.0,
                        "openinterest": 0.0,
                        "entry": entries[ticker].astype(float),
                        "exit": exits[ticker].astype(float),
                    },
                    index=naive_index,
                )
                cerebro.adddata(feed_cls(dataname=frame), name=str(ticker))

            cerebro.addstrategy(
                strategy_cls,
                target_percent=1.0 / len(close.columns),
                allow_short=allow_short,
            )
            runs = cerebro.run()
        except Exception as exc:
            msg = "The backtest could not be executed"
            raise BacktestError(msg, detail=f"{type(exc).__name__}: {exc}") from exc

        if not runs:
            msg = "backtrader returned no result"
            raise BacktestError(msg, detail="cerebro.run() war leer")

        strategy = runs[0]
        equity = pd.Series(
            strategy.equity_values, index=close.index[: len(strategy.equity_values)], dtype=float
        )
        cash = pd.Series(
            strategy.cash_values, index=close.index[: len(strategy.cash_values)], dtype=float
        )
        trades = self._collect_trades(strategy.closed_trades, close=close)
        return equity, cash, trades

    def _collect_trades(self, raw: list[dict[str, Any]], *, close: pd.DataFrame) -> list[Trade]:
        """Uebersetzt die aufgezeichneten Trades in den Projektvertrag.

        Args:
            raw: Rohaufzeichnungen der backtrader-Strategie.
            close: Schlusskurs-Panel, liefert den Bar-Index fuer die
                Haltedauer.

        Returns:
            Trades in chronologischer Reihenfolge des Einstiegs.
        """
        positions = {timestamp: i for i, timestamp in enumerate(close.index)}
        out: list[Trade] = []

        for record in raw:
            entry_price = float(record["entry_price"])
            size = abs(float(record["size"]))
            if entry_price <= 0.0 or size <= 0.0:
                continue

            entry_ts = pd.Timestamp(record["entry_time"]).tz_localize("UTC")
            exit_raw = record.get("exit_time")
            exit_ts = None if exit_raw is None else pd.Timestamp(exit_raw).tz_localize("UTC")
            exit_price = record.get("exit_price")

            duration: int | None = None
            if exit_ts is not None:
                start, end = positions.get(entry_ts), positions.get(exit_ts)
                if start is not None and end is not None:
                    duration = max(end - start, 0)

            pnl = None if exit_ts is None else float(record["pnl"])
            basis = entry_price * size
            out.append(
                Trade(
                    ticker=str(record["ticker"]),
                    side=TradeSide.SHORT if float(record["size"]) < 0 else TradeSide.LONG,
                    entry_time=entry_ts.to_pydatetime(),
                    entry_price=entry_price,
                    exit_time=None if exit_ts is None else exit_ts.to_pydatetime(),
                    exit_price=None if exit_price is None else float(exit_price),
                    size=size,
                    pnl=pnl,
                    return_pct=None if pnl is None or basis <= 0 else pnl / basis,
                    fees_paid=max(float(record.get("commission", 0.0)), 0.0),
                    duration_bars=duration,
                )
            )

        out.sort(key=lambda t: (t.entry_time, t.ticker))
        return out

    def _exposure(self, entries: pd.DataFrame, exits: pd.DataFrame) -> float | None:
        """Schaetzt den Anteil der Bars mit offener Position aus den Signalen.

        Rekonstruiert den Positionszustand je Ticker durch Vorwaertstragen
        des jeweils letzten Signals und mittelt ueber die Ticker.

        Args:
            entries: Einstiegsmatrix nach Lag.
            exits: Ausstiegsmatrix nach Lag.

        Returns:
            Exposure, dezimal zwischen 0.0 und 1.0. None bei leerer Matrix.
        """
        if entries.empty:
            return None
        # Zustand als float: NaN = unveraendert, 1.0 = Einstieg, 0.0 = Ausstieg.
        # Ein bool-DataFrame kennt kein NaN und liesse sich nicht ffillen.
        state = pd.DataFrame(
            float("nan"), index=entries.index, columns=entries.columns, dtype=float
        )
        state = state.mask(entries, other=1.0).mask(exits & ~entries, other=0.0)
        held = state.ffill().fillna(0.0).astype(bool)
        return metrics_mod.exposure_from_mask(pd.Series(held.to_numpy().ravel()))

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

        Args:
            benchmark: Vergleichsreihe. None liefert None.
            index: Index der Strategie-Equity.
            init_cash: Startkapital in Kontowaehrung.
            warnings: Liste, in die nicht-fatale Hinweise wandern.

        Returns:
            Benchmark-Equity auf ``index``, oder None.
        """
        if benchmark is None:
            return None

        close = pd.to_numeric(benchmark.data["close"], errors="coerce").astype(float)
        close = close[close > 0.0].dropna()

        # Deckungspruefung VOR dem Vorwaertstragen, siehe vbt_engine.py.
        if len(close.index.intersection(index)) < MIN_BARS:
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
        return (aligned / float(aligned.iloc[0]) * init_cash).reindex(index).ffill()

    def _benchmark_metrics(
        self, bench_equity: pd.Series | None, *, interval: Interval
    ) -> PerformanceMetrics | None:
        """Berechnet die Kennzahlen der Benchmark selbst.

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


def _signal_feed_class(bt: Any) -> Any:
    """Baut die Feed-Klasse mit den beiden Signal-Zusatzlinien.

    Die Klasse entsteht erst zur Laufzeit, weil sie von ``backtrader.feeds``
    erbt und das Modul optional ist.

    Args:
        bt: Das importierte ``backtrader``-Modul.

    Returns:
        Eine ``PandasData``-Unterklasse mit den Linien ``entry`` und
        ``exit``.
    """

    class SignalFeed(bt.feeds.PandasData):  # type: ignore[misc]
        """OHLCV-Feed, der die Signalspalten mitfuehrt."""

        lines = ("entry", "exit")
        params = (("entry", -1), ("exit", -1))

    return SignalFeed


def _signal_strategy_class(bt: Any) -> Any:
    """Baut die ausfuehrende backtrader-Strategie.

    Sie enthaelt **keine** Signal-Logik, sondern liest die vorberechneten
    Signale aus dem Feed und legt Orders. Zusaetzlich zeichnet sie Equity,
    Kasse und geschlossene Trades auf, damit die Auswertung nicht von
    backtrader-Analyzern abhaengt.

    Args:
        bt: Das importierte ``backtrader``-Modul.

    Returns:
        Eine ``Strategy``-Unterklasse.
    """

    class SignalExecutor(bt.Strategy):  # type: ignore[misc]
        """Fuehrt vorberechnete Signale aus und protokolliert den Verlauf."""

        params = (("target_percent", 1.0), ("allow_short", False))

        def __init__(self) -> None:
            """Legt die Aufzeichnungspuffer an."""
            self.equity_values: list[float] = []
            self.cash_values: list[float] = []
            self.closed_trades: list[dict[str, Any]] = []

        def next(self) -> None:
            """Verarbeitet einen Bar: erst aufzeichnen, dann handeln."""
            self.equity_values.append(float(self.broker.getvalue()))
            self.cash_values.append(float(self.broker.getcash()))

            for data in self.datas:
                entry = bool(data.entry[0])
                exit_signal = bool(data.exit[0])
                position = self.getposition(data).size

                if entry and position == 0:
                    self.order_target_percent(data=data, target=self.p.target_percent)
                elif exit_signal and position != 0:
                    if self.p.allow_short:
                        self.order_target_percent(data=data, target=-self.p.target_percent)
                    else:
                        self.close(data=data)

        def notify_trade(self, trade: Any) -> None:
            """Zeichnet einen geschlossenen Trade auf.

            Args:
                trade: Das Trade-Objekt von backtrader.
            """
            if not trade.isclosed:
                return
            self.closed_trades.append(
                {
                    "ticker": trade.data._name,
                    "size": float(trade.size) if trade.size else float(trade.history[0].event.size),
                    "entry_price": float(trade.price),
                    "entry_time": bt.num2date(trade.dtopen),
                    "exit_time": bt.num2date(trade.dtclose),
                    "exit_price": float(trade.data.close[0]),
                    "pnl": float(trade.pnlcomm),
                    "commission": float(trade.commission),
                }
            )

    return SignalExecutor
