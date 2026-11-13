"""Tests der Hintergrundarbeiten der Oberflaeche.

Warum es dieses Modul gibt, steht in ``btlite.ui.jobs``: ein Screening
ueber den S&P 500 laeuft gemessen rund eine halbe Stunde. Ohne
Rueckmeldung sieht der Nutzer nur einen Ladebalken und haelt die Seite
fuer haengengeblieben - genau so ging die urspruengliche Fehlermeldung
ein ("laeuft laenger als 7min und gibt keine ergebnisse").

Die Tests hier warten auf echte Threads. Sie tun das ueber eine
Zeitschranke statt ueber ``sleep``-Schaetzungen: ein Test, der auf eine
feste Wartezeit baut, wird auf einer langsameren Maschine flaky.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from btlite.core.errors import BTLiteError
from btlite.ui.jobs import Job, JobRegistry

#: Obergrenze fuer das Warten auf einen Thread. Grosszuegig, weil die
#: Tests nur Millisekunden brauchen - die Schranke faengt einen haengenden
#: Thread ab, statt die Suite blockieren zu lassen.
_TIMEOUT_S = 5.0


def _await(job: Job) -> Job:
    """Wartet, bis eine Arbeit beendet ist.

    Args:
        job: Zu beobachtende Arbeit.

    Returns:
        Dieselbe Arbeit, dann mit ``finished``.

    Raises:
        AssertionError: Wenn die Arbeit nicht rechtzeitig endet.
    """
    deadline = time.monotonic() + _TIMEOUT_S
    while time.monotonic() < deadline:
        if job.snapshot()[2]:
            return job
        time.sleep(0.005)
    msg = f"Arbeit {job.job_id} wurde nicht fertig"
    raise AssertionError(msg)


@pytest.fixture
def registry() -> JobRegistry:
    """Eigene Registry je Test - die prozessweite bleibt unberuehrt."""
    return JobRegistry()


class TestResult:
    """Eine Arbeit haelt ihr Ergebnis fest."""

    def test_the_result_is_kept(self, registry: JobRegistry) -> None:
        job = _await(registry.start(lambda _job: "fertig"))
        assert job.result == "fertig"

    def test_the_job_is_marked_finished(self, registry: JobRegistry) -> None:
        job = _await(registry.start(lambda _job: None))
        assert job.snapshot()[2] is True

    def test_a_falsy_result_is_not_lost(self, registry: JobRegistry) -> None:
        """Eine leere Liste ist ein Ergebnis, kein fehlendes."""
        job = _await(registry.start(lambda _job: []))
        assert job.result == []
        assert job.error is None


class TestProgress:
    """Der Fortschritt ist der eigentliche Zweck des Moduls."""

    def test_progress_is_readable_from_outside(self, registry: JobRegistry) -> None:
        seen = threading.Event()

        def work(job: Job) -> str:
            job.progress(3, 10)
            seen.set()
            # Warten, damit der Test den Zwischenstand sieht, bevor die
            # Arbeit endet - sonst pruefte er nur den Endzustand.
            time.sleep(0.05)
            return "fertig"

        job = registry.start(work)
        assert seen.wait(_TIMEOUT_S), "Arbeit hat nie gemeldet"
        done, total, _finished, _cancelled = job.snapshot()
        assert (done, total) == (3, 10)
        _await(job)

    def test_snapshot_reads_both_values_together(self, registry: JobRegistry) -> None:
        """Getrennt gelesen ergaebe sich eine Anzeige wie "512 von 503"."""
        job = _await(registry.start(lambda j: j.progress(7, 7)))
        done, total, _f, _c = job.snapshot()
        assert done <= total


class TestFailure:
    """Ein Fehler darf nicht stumm verschwinden."""

    def test_a_known_error_is_kept(self, registry: JobRegistry) -> None:
        def boom(_job: Job) -> None:
            msg = "Screening failed."
            raise BTLiteError(msg, detail="Testfall")

        job = _await(registry.start(boom))
        assert job.error is not None
        assert job.error.user_message == "Screening failed."

    def test_an_unexpected_error_also_ends_the_job(self, registry: JobRegistry) -> None:
        """Sonst wartete die Seite endlos auf ein Ergebnis, das nie kommt.

        Ein ``ZeroDivisionError`` ist kein ``BTLiteError`` - ohne den
        breiten except-Zweig bliebe ``finished`` False und die
        Fortschrittsabfrage liefe ewig weiter.
        """

        def crash(_job: Job) -> float:
            return 1 / 0

        job = _await(registry.start(crash))
        assert job.error is not None
        assert job.result is None

    def test_a_failed_job_carries_no_result(self, registry: JobRegistry) -> None:
        def boom(_job: Job) -> None:
            msg = "Screening failed."
            raise BTLiteError(msg, detail="Testfall")

        job = _await(registry.start(boom))
        assert job.result is None


class TestCancel:
    """Abbrechen wirkt am naechsten Haltepunkt, nicht mitten im Schreiben."""

    def test_the_work_sees_the_request(self, registry: JobRegistry) -> None:
        started = threading.Event()

        def work(job: Job) -> str:
            started.set()
            for index in range(200):
                if job.cancelled:
                    return "abgebrochen"
                job.progress(index, 200)
                time.sleep(0.005)
            return "durchgelaufen"

        job = registry.start(work)
        assert started.wait(_TIMEOUT_S)
        job.cancel()
        assert _await(job).result == "abgebrochen"

    def test_cancelling_shows_in_the_snapshot(self, registry: JobRegistry) -> None:
        job = registry.start(lambda _j: time.sleep(0.05))
        job.cancel()
        assert job.snapshot()[3] is True
        _await(job)


class TestRegistry:
    """Nachschlagen und Aufraeumen."""

    def test_a_job_is_found_by_its_id(self, registry: JobRegistry) -> None:
        job = _await(registry.start(lambda _j: "x"))
        assert registry.get(job.job_id) is job

    def test_an_unknown_id_yields_none(self, registry: JobRegistry) -> None:
        assert registry.get("gibtsnicht") is None

    def test_none_yields_none(self, registry: JobRegistry) -> None:
        """Beim ersten Seitenaufbau ist noch keine Kennung gesetzt."""
        assert registry.get(None) is None

    def test_a_discarded_job_is_gone(self, registry: JobRegistry) -> None:
        job = _await(registry.start(lambda _j: "x"))
        registry.discard(job.job_id)
        assert registry.get(job.job_id) is None

    def test_discarding_an_unknown_id_is_harmless(self, registry: JobRegistry) -> None:
        registry.discard("gibtsnicht")
        registry.discard(None)

    def test_finished_jobs_do_not_pile_up(self, registry: JobRegistry) -> None:
        """Ohne Grenze wuechse das Dict mit jedem Lauf."""
        jobs = [_await(registry.start(lambda _j: "x")) for _ in range(20)]
        alive = sum(1 for job in jobs if registry.get(job.job_id) is not None)
        assert alive < len(jobs), "Nichts wurde weggeraeumt"

    def test_a_running_job_is_never_forgotten(self, registry: JobRegistry) -> None:
        """Sein Ergebnis wird noch gebraucht - anders als bei erledigten."""
        release = threading.Event()
        running = registry.start(lambda _j: release.wait(_TIMEOUT_S))
        try:
            for _ in range(20):
                _await(registry.start(lambda _j: "x"))
            assert registry.get(running.job_id) is running
        finally:
            release.set()
            _await(running)


class TestIsolation:
    """Zwei Arbeiten kommen sich nicht ins Gehege."""

    def test_two_jobs_have_different_ids(self, registry: JobRegistry) -> None:
        first = _await(registry.start(lambda _j: "a"))
        second = _await(registry.start(lambda _j: "b"))
        assert first.job_id != second.job_id

    def test_cancelling_one_leaves_the_other_alone(self, registry: JobRegistry) -> None:
        def work(job: Job) -> Any:
            for _ in range(100):
                if job.cancelled:
                    return "abgebrochen"
                time.sleep(0.005)
            return "durchgelaufen"

        first = registry.start(work)
        second = registry.start(work)
        first.cancel()
        assert _await(first).result == "abgebrochen"
        assert _await(second).result == "durchgelaufen"
