"""Tests der Ueberarbeitung "Screening laeuft nicht fluessig".

Gemessene Befunde und was hier festgehalten wird:

1. Ein Folgelauf war so langsam wie der erste (40 Titel: 40 s, auch im
   selben Prozess). Ursache: ``Ticker.info`` wurde je Titel zweimal pro Lauf
   neu geladen. Jetzt vier Stunden gepuffert (Entscheidung des Nutzers).
2. Die Fortschrittsanzeige wuchs nur alle zehn Titel - bei rund drei
   Sekunden je Titel stand sie bis zu dreissig Sekunden still.
3. Die Restzeit galt ab dem ersten Titel als "warm" und war im kalten Lauf
   ein Drittel zu niedrig. Jetzt gemessen am laufenden Job.
4. Der Lade-Spinner blendete die Anzeige bei jedem Fortschrittstakt aus.
5. Ein neuer Lauf liess den alten weiterlaufen; beide teilten sich die
   Drosselung der Quelle.
6. Bis 50 Titel rechnete ein Preset-Wechsel im Web-Thread, kalt Minuten.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import pytest

from btlite.config import Settings
from btlite.core.cache import DataFrameCache
from btlite.core.errors import ProviderUnavailableError
from btlite.data.yfinance_provider import YFinanceProvider
from btlite.ui import jobs as jobs_module
from btlite.ui.app import create_app, render_page
from btlite.ui.jobs import Job, JobRegistry
from btlite.ui.layouts import components as c
from btlite.ui.pages import eqs
from btlite.ui.state import AppState

if TYPE_CHECKING:
    from pathlib import Path


# --------------------------------------------------------------------------
# 1) Stammdaten-Puffer


class _InfoTicker:
    """Attrappe fuer ``yf.Ticker`` mit zaehlbarem ``info``."""

    def __init__(self, calls: list[str], infos: dict[str, Any]) -> None:
        self._calls = calls
        self._infos = infos

    def __call__(self, symbol: str) -> _InfoTicker:
        clone = _InfoTicker(self._calls, self._infos)
        clone.symbol = symbol  # type: ignore[attr-defined]
        return clone

    @property
    def info(self) -> Any:
        symbol = self.symbol  # type: ignore[attr-defined]
        self._calls.append(symbol)
        value = self._infos[symbol]
        if isinstance(value, Exception):
            raise value
        return value


def _provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, infos: dict[str, Any]
) -> tuple[YFinanceProvider, list[str]]:
    import yfinance as yf

    calls: list[str] = []
    monkeypatch.setattr(yf, "Ticker", _InfoTicker(calls, infos))
    settings = Settings()
    settings.data.offline = False
    settings.cache.dir = tmp_path / "cache"
    return YFinanceProvider(settings, cache=DataFrameCache(settings.cache)), calls


_INFO = {"longName": "Apple Inc.", "sector": "Technology", "marketCap": 3.0e12}


class TestInfoIsCached:
    def test_second_call_comes_from_the_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider, calls = _provider(tmp_path, monkeypatch, {"AAPL": _INFO})
        first = provider._raw_info("AAPL")
        second = provider._raw_info("AAPL")
        assert first == second == _INFO
        assert calls == ["AAPL"]

    def test_the_cache_survives_a_new_provider(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Genau der gemessene Fall: neuer Prozesszustand, Disk warm."""
        provider, calls = _provider(tmp_path, monkeypatch, {"AAPL": _INFO})
        provider._raw_info("AAPL")
        again, _ = _provider(tmp_path, monkeypatch, {"AAPL": _INFO})
        assert again._raw_info("AAPL") == _INFO
        assert calls == ["AAPL"]

    def test_the_ttl_is_the_decided_four_hours(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert Settings().cache.info_ttl_s == 4 * 3600
        provider, _ = _provider(tmp_path, monkeypatch, {"AAPL": _INFO})
        seen: list[float | None] = []
        original = provider._cache.set_df

        def spy(key: str, df: Any, ttl_s: float | None = None) -> bool:
            seen.append(ttl_s)
            return original(key, df, ttl_s)

        monkeypatch.setattr(provider._cache, "set_df", spy)
        provider._raw_info("AAPL")
        assert seen == [4 * 3600]

    def test_failures_are_not_cached(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        provider, calls = _provider(
            tmp_path, monkeypatch, {"AAPL": ProviderUnavailableError("down")}
        )
        for _ in range(2):
            with pytest.raises(ProviderUnavailableError):
                provider._raw_info("AAPL")
        # Kein Puffer fuer den Fehler: jeder Aufruf fragt neu (mit Retries).
        assert len(calls) >= 2

    def test_a_corrupt_cache_entry_is_refetched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pandas as pd

        from btlite.core.cache import cache_key

        provider, calls = _provider(tmp_path, monkeypatch, {"AAPL": _INFO})
        provider._cache.set_df(
            cache_key("yf", "info", "AAPL"), pd.DataFrame({"info": ["{kaputt"]}), 3600
        )
        assert provider._raw_info("AAPL") == _INFO
        assert calls == ["AAPL"]


# --------------------------------------------------------------------------
# 2) Fortschritt je Titel


def test_complete_ratio_universe_is_reused_from_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BTLITE_PORTFOLIO_HOLDINGS_FILE", str(tmp_path / "h.json"))
    settings = Settings()
    settings.cache.dir = tmp_path / "cache"
    state = AppState(settings)
    calls = 0
    original = state.provider.get_statements

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(state.provider, "get_statements", counted)
    first = state.ratio_universe(["AAPL", "MSFT"])
    after_first = calls
    second = state.ratio_universe(["AAPL", "MSFT"])

    assert after_first == 2
    assert calls == after_first
    assert second.equals(first)


def test_progress_is_reported_for_every_title(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BTLITE_PORTFOLIO_HOLDINGS_FILE", str(tmp_path / "h.json"))
    state = AppState(Settings())
    seen: list[tuple[int, int]] = []
    state.histories(
        ["AAPL", "MSFT", "KO", "PG"], period="1y", on_progress=lambda d, t: seen.append((d, t))
    )
    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]


# --------------------------------------------------------------------------
# 3) Restzeit aus dem gemessenen Tempo


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TestMeasuredRate:
    @pytest.fixture
    def clock(self, monkeypatch: pytest.MonkeyPatch) -> _Clock:
        clock = _Clock()
        monkeypatch.setattr(jobs_module.time, "monotonic", clock)
        return clock

    def test_no_rate_before_three_titles(self, clock: _Clock) -> None:
        job = Job(job_id="x")
        job.progress(0, 100)
        clock.now += 6
        job.progress(2, 100)
        assert job.seconds_per_unit() is None

    def test_rate_since_the_step_began(self, clock: _Clock) -> None:
        job = Job(job_id="x")
        job.progress(0, 100)
        clock.now += 30
        job.progress(10, 100)
        assert job.seconds_per_unit() == pytest.approx(3.0)

    def test_a_new_step_restarts_the_measurement(self, clock: _Clock) -> None:
        """Zweiter Schritt (Kurse fuer die Treffer) hat eigenes Tempo."""
        job = Job(job_id="x")
        job.progress(0, 100)
        clock.now += 300
        job.progress(100, 100)
        job.progress(101, 130)  # neuer Schritt: 30 Treffer
        clock.now += 1
        job.progress(104, 130)
        assert job.seconds_per_unit() == pytest.approx(1 / 3)

    def test_the_view_uses_the_measured_rate(self) -> None:
        """137 von 503 bei 3 s je Titel: 366 offen, gut 18 Minuten."""
        text = str(eqs.progress_view(137, 503, "value", seconds_per_unit=3.0))
        assert "about 19 minutes left" in text

    def test_a_fast_warm_run_says_less_than_a_minute(self) -> None:
        text = str(eqs.progress_view(137, 503, "value", seconds_per_unit=0.05))
        assert "less than a minute left" in text

    def test_without_measurement_the_cold_estimate_holds(self) -> None:
        """Nicht mehr das warme Tempo ab dem ersten Titel."""
        text = str(eqs.progress_view(137, 503, "value"))
        cold = eqs._estimate_minutes(503 - 137)
        assert f"about {eqs.format_minutes(cold)} left" in text

    def test_the_progress_callback_passes_the_rate(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setenv("BTLITE_PORTFOLIO_HOLDINGS_FILE", str(tmp_path / "h.json"))
        state = AppState(Settings())
        app = create_app(state.settings, state)
        tick = next(
            getattr(e["callback"], "__wrapped__", e["callback"])
            for k, e in app.callback_map.items()
            if eqs.PROGRESS_TICK in str(e.get("inputs", "")) and "eqs-result" in k
        )
        registry = jobs_module.REGISTRY
        job = registry.start(lambda j: time.sleep(5))
        monkeypatch.setattr(job, "seconds_per_unit", lambda: 0.05)
        job.progress(10, 503)
        out, *_ = tick(1, job.job_id, "value", ["T"] * 503)
        job.cancel()
        assert "less than a minute left" in str(out)


# --------------------------------------------------------------------------
# 4) Kein Flackern


class TestNoFlicker:
    def test_the_result_area_delays_the_spinner(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setenv("BTLITE_PORTFOLIO_HOLDINGS_FILE", str(tmp_path / "h.json"))
        state = AppState(Settings())
        page = render_page(state, "EQS", ["AAPL", "MSFT"], ())

        def walk(node: Any) -> Any:
            if getattr(node, "_type", "") == "Loading" and any(
                getattr(ch, "id", None) == eqs.RESULT_OUTPUT
                for ch in (node.children if isinstance(node.children, list) else [node.children])
            ):
                return node
            kids = getattr(node, "children", None)
            for kid in kids if isinstance(kids, list) else [kids]:
                if kid is not None and hasattr(kid, "_type"):
                    found = walk(kid)
                    if found is not None:
                        return found
            return None

        loading = walk(page)
        assert loading is not None, "Ergebnisbereich nicht in dcc.Loading"
        # Laenger als ein Fortschrittstakt dauert: der Spinner bleibt weg.
        assert loading.delay_show >= 300

    def test_other_loading_areas_are_unchanged(self) -> None:
        assert getattr(c.loading("x"), "delay_show", None) is None


# --------------------------------------------------------------------------
# 5) Alter Lauf wird abgebrochen


class TestStaleRuns:
    def test_cancel_running_hits_only_unfinished_jobs(self) -> None:
        registry = JobRegistry()
        slow = registry.start(lambda j: time.sleep(2))
        done = registry.start(lambda j: None)
        for _ in range(100):
            if done.snapshot()[2]:
                break
            time.sleep(0.01)
        assert registry.cancel_running() == 1
        assert slow.cancelled
        assert not done.cancelled

    def test_a_new_run_cancels_the_previous_one(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setenv("BTLITE_PORTFOLIO_HOLDINGS_FILE", str(tmp_path / "h.json"))
        state = AppState(Settings())
        app = create_app(state.settings, state)
        run = next(
            getattr(e["callback"], "__wrapped__", e["callback"])
            for e in app.callback_map.values()
            if "eqs-run" in str(e.get("inputs", ""))
        )
        universe = ["AAPL", "MSFT", "NVDA", "SPY", "JNJ", "XOM", "KO", "PG", "JPM", "WMT"]
        _v, first_id, _o, _h = run(1, "value", universe)
        _v, second_id, _o, _h = run(2, "value", universe)
        first = jobs_module.REGISTRY.get(first_id)
        second = jobs_module.REGISTRY.get(second_id)
        assert first is not None
        assert second is not None
        assert first.cancelled or first.snapshot()[2]
        assert not second.cancelled
        second.cancel()


# --------------------------------------------------------------------------
# 6) Nur kleine Universen rechnen sofort


class TestComputeOnlySmallUniverses:
    @pytest.fixture
    def state(self, tmp_path: Path, monkeypatch: Any) -> AppState:
        monkeypatch.setenv("BTLITE_PORTFOLIO_HOLDINGS_FILE", str(tmp_path / "h.json"))
        return AppState(Settings())

    def test_ten_titles_compute_directly(self, state: AppState) -> None:
        universe = ["AAPL", "MSFT", "NVDA", "SPY", "JNJ", "XOM", "KO", "PG", "JPM", "WMT"]
        assert "Matches" in str(eqs.initial_result(state, "value", universe))

    def test_eleven_titles_wait_for_the_button(
        self, state: AppState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def forbidden(*_a: Any, **_k: Any) -> Any:
            raise AssertionError("im Web-Thread gerechnet")

        monkeypatch.setattr(eqs, "render_result", forbidden)
        text = str(eqs.initial_result(state, "value", [f"T{i}" for i in range(11)]))
        assert "Run screening" in text
