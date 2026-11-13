"""Laufende Hintergrundarbeiten der Oberflaeche.

Warum es dieses Modul gibt: ein Screening ueber den S&P 500 laeuft
Minuten. Ohne Rueckmeldung sieht der Nutzer nur einen Ladebalken und
haelt die Seite fuer haengengeblieben - genau das ist passiert. Also
rechnet die Arbeit in einem eigenen Thread, und die Seite fragt den
Stand ab.

Bewusst **nicht** Dashs ``background=True``: dessen ``DiskcacheManager``
verlangt das Paket ``multiprocess``, das hier nicht installiert ist. Eine
neue Abhaengigkeit fuer eine Fortschrittsanzeige waere zu teuer, zumal
BT-Lite ohnehin lokal fuer einen einzigen Nutzer laeuft. Am Paket
geprueft, nicht vermutet: ``DiskcacheManager(cache)`` wirft ohne
``multiprocess`` einen ``ImportError``.

Der Zustand liegt prozessweit in einem Dict. Das genuegt genau deshalb:
ein Prozess, ein Nutzer, kein Auth, kein Cloud-Deployment - die
Nicht-Ziele des Projekts. Bei mehreren Arbeitsprozessen waere es falsch,
und dieser Satz steht hier, damit das auffaellt, bevor es jemand tut.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final

from btlite.core.errors import BTLiteError
from btlite.core.logging import get_logger

_LOG = get_logger(__name__)

#: Obergrenze gleichzeitig aufbewahrter Arbeiten. Ein abgeschlossener Job
#: bleibt liegen, bis die Seite sein Ergebnis abgeholt hat; ohne Grenze
#: wuechse das Dict mit jedem Lauf.
_MAX_JOBS: Final[int] = 8

#: Mindestzahl erledigter Einheiten, ab der ein gemessenes Tempo gilt.
_MIN_RATE_UNITS: Final[int] = 3


@dataclass
class Job:
    """Eine laufende oder abgeschlossene Hintergrundarbeit.

    Attributes:
        job_id: Kennung, ueber die die Seite den Stand abfragt.
        done: Bereits erledigte Einheiten.
        total: Gesamtzahl der Einheiten. 0 heisst "noch unbekannt".
        finished: Ob die Arbeit beendet ist - erfolgreich oder nicht.
        cancelled: Ob der Nutzer abgebrochen hat.
        result: Ergebnis der Arbeit, oder None.
        error: Aufgetretener Fehler, oder None. Ein Job traegt entweder
            ein Ergebnis oder einen Fehler, nie beides.
    """

    job_id: str
    done: int = 0
    total: int = 0
    finished: bool = False
    cancelled: bool = False
    result: Any = None
    error: BTLiteError | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # Bezugspunkt der Tempomessung: Zeitpunkt (monoton, Sekunden) und Stand.
    # Wechselt ``total``, beginnt ein neuer Schritt mit eigenem Tempo, und
    # der Bezugspunkt springt mit.
    _anchor_t: float = field(default_factory=time.monotonic, repr=False)
    _anchor_done: int = field(default=0, repr=False)

    def progress(self, done: int, total: int) -> None:
        """Vermerkt den Fortschritt.

        Wird aus dem Rechen-Thread gerufen, gelesen aus dem Web-Thread.

        Args:
            done: Erledigte Einheiten.
            total: Gesamtzahl der Einheiten.
        """
        with self._lock:
            if total != self.total:
                self._anchor_t = time.monotonic()
                self._anchor_done = done
            self.done = done
            self.total = total

    def snapshot(self) -> tuple[int, int, bool, bool]:
        """Liest den Stand in einem Zug.

        Einzeln gelesen koennten ``done`` und ``total`` aus verschiedenen
        Augenblicken stammen und eine Anzeige wie "512 von 503" ergeben.

        Returns:
            Tupel aus erledigt, gesamt, fertig und abgebrochen.
        """
        with self._lock:
            return self.done, self.total, self.finished, self.cancelled

    def seconds_per_unit(self, min_units: int = _MIN_RATE_UNITS) -> float | None:
        """Misst das Tempo des laufenden Schritts.

        Args:
            min_units: Mindestzahl erledigter Einheiten seit Schrittbeginn,
                bevor ein Tempo genannt wird. Nach einer einzigen Einheit
                schwankt der Wert zu stark.

        Returns:
            Sekunden je Einheit seit Beginn des Schritts, oder None, solange
            zu wenig gemessen ist.
        """
        with self._lock:
            units = self.done - self._anchor_done
            elapsed = time.monotonic() - self._anchor_t
        if units < max(1, min_units) or elapsed <= 0:
            return None
        return elapsed / units

    def cancel(self) -> None:
        """Bittet die Arbeit, beim naechsten Fortschritt aufzuhoeren.

        Der Thread wird nicht hart beendet - ein mitten im Schreiben
        abgebrochener Thread koennte halbe Zustaende hinterlassen. Die
        Arbeit prueft das Kennzeichen an ihren eigenen Haltepunkten.
        """
        with self._lock:
            self.cancelled = True


class JobRegistry:
    """Verwaltet die laufenden Arbeiten eines Prozesses."""

    def __init__(self) -> None:
        """Legt eine leere Registry an."""
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, work: Callable[[Job], Any]) -> Job:
        """Startet eine Arbeit in einem eigenen Thread.

        Args:
            work: Auszufuehrende Arbeit. Bekommt den :class:`Job`, um
                Fortschritt zu melden und den Abbruchwunsch zu pruefen.

        Returns:
            Der angelegte Job. Sein Ergebnis steht erst bereit, wenn
            ``finished`` gesetzt ist.
        """
        job = Job(job_id=uuid.uuid4().hex)
        with self._lock:
            self._forget_oldest_finished()
            self._jobs[job.job_id] = job

        def runner() -> None:
            """Fuehrt die Arbeit aus und haelt das Ergebnis fest."""
            try:
                job.result = work(job)
            except BTLiteError as exc:
                _LOG.warning("Hintergrundarbeit gescheitert: %s", exc.detail or exc.user_message)
                job.error = exc
            except Exception:
                # Ein unerwarteter Fehler darf den Thread nicht stumm
                # beenden - die Seite wartete sonst endlos auf ein
                # Ergebnis, das nie kommt.
                _LOG.exception("Hintergrundarbeit mit unerwartetem Fehler")
                job.error = BTLiteError(
                    "The screening stopped unexpectedly.",
                    detail="siehe Log",
                )
            finally:
                with job._lock:
                    job.finished = True

        threading.Thread(target=runner, daemon=True).start()
        return job

    def get(self, job_id: str | None) -> Job | None:
        """Sucht eine Arbeit.

        Args:
            job_id: Kennung, oder None.

        Returns:
            Der Job, oder None wenn die Kennung unbekannt ist.
        """
        if not job_id:
            return None
        with self._lock:
            return self._jobs.get(job_id)

    def cancel_running(self) -> int:
        """Bittet alle noch laufenden Arbeiten aufzuhoeren.

        Vor einem neuen Screening gerufen. Ein alter Lauf, dessen Ergebnis
        niemand mehr abholt, teilte sich sonst die Drosselung der Quelle mit
        dem neuen und halbierte dessen Tempo. Prozessweit ist das nur
        richtig, weil die App keinen Mehrbenutzerbetrieb kennt (Nicht-Ziel).

        Returns:
            Anzahl der gebetenen Arbeiten.
        """
        with self._lock:
            running = [job for job in self._jobs.values() if not job.finished]
        for job in running:
            job.cancel()
        return len(running)

    def discard(self, job_id: str | None) -> None:
        """Vergisst eine abgeschlossene Arbeit.

        Args:
            job_id: Kennung, oder None.
        """
        if not job_id:
            return
        with self._lock:
            self._jobs.pop(job_id, None)

    def _forget_oldest_finished(self) -> None:
        """Raeumt abgeschlossene Arbeiten weg, wenn es zu viele werden.

        Laufende Arbeiten bleiben immer erhalten: ihr Ergebnis wird noch
        gebraucht.
        """
        if len(self._jobs) < _MAX_JOBS:
            return
        for job_id, job in list(self._jobs.items()):
            if job.finished:
                del self._jobs[job_id]
            if len(self._jobs) < _MAX_JOBS:
                return


#: Prozessweite Registry. Siehe Modul-Docstring zur Einordnung.
REGISTRY: Final[JobRegistry] = JobRegistry()
