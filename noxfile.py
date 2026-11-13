"""Aufgaben-Runner fuer BT-Lite.

Warum nox und nicht make: ``make`` ist auf dem Zielsystem (Windows) nicht
vorhanden, nox dagegen laeuft als reines Python-Paket ueberall, wo auch das
Projekt laeuft. Ein Makefile waere hier ein Werkzeug, das niemand
ausfuehren kann.

Warum ``python=False`` an jeder Session: nox baut sonst je Session ein
eigenes virtuelles Environment und installiert das Projekt neu. Das ist
hier unerwuenscht - die Sessions sollen **genau** den Interpreter aus
``.venv`` benutzen, in dem auch entwickelt wird. Der Pfad dorthin wird
deshalb explizit bestimmt, nicht aus der ``PATH``-Reihenfolge geraten:
laut ``CLAUDE.md`` zielt ein nacktes ``python`` auf dieser Maschine auf das
conda-Base.

Aufrufe::

    .venv/Scripts/python.exe -m nox -l            # Sessions auflisten
    .venv/Scripts/python.exe -m nox -s lint
    .venv/Scripts/python.exe -m nox -s typecheck
    .venv/Scripts/python.exe -m nox -s test
    .venv/Scripts/python.exe -m nox -s coverage
    .venv/Scripts/python.exe -m nox               # Default: alles

Die Session ``all`` laeuft die vier Schritte in der Reihenfolge, in der
sie am schnellsten scheitern: erst Formatierung und Lint, dann Typen,
dann die Suite.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nox

#: Wurzel des Projekts.
ROOT = Path(__file__).parent

#: Mindest-Coverage ueber das Gesamtprojekt, in Prozent.
#: Das Projektziel nennt 80 Prozent fuer die Rechenschichten; gemessen
#: ueber alles liegt der Ist-Stand darueber. Die Schwelle sichert ihn ab,
#: ohne bei einer kleinen Schwankung rot zu werden.
COVERAGE_MIN = 80

#: Pfade, die Lint und Typpruefung betrachten.
SOURCES = ("src", "tests", "noxfile.py")

# Nur die hier genannten Sessions laufen ohne Argument.
nox.options.sessions = ["lint", "typecheck", "test"]
nox.options.reuse_existing_virtualenvs = True


def _venv_scripts() -> Path:
    """Findet das Skriptverzeichnis des Projekt-venv.

    Returns:
        ``.venv/Scripts`` auf Windows, ``.venv/bin`` sonst.

    Raises:
        FileNotFoundError: Wenn kein ``.venv`` vorliegt. Ohne das lokale
            Environment fehlen ruff, mypy und pytest.
    """
    for name in ("Scripts", "bin"):
        candidate = ROOT / ".venv" / name
        if candidate.is_dir():
            return candidate
    msg = (
        "Kein .venv gefunden. Anlegen mit:\n"
        "  python -m uv venv .venv\n"
        '  python -m uv pip install --python ".venv\\Scripts\\python.exe" -e ".[dev]"'
    )
    raise FileNotFoundError(msg)


def _tool(name: str) -> str:
    """Baut den absoluten Pfad zu einem Werkzeug im Projekt-venv.

    Args:
        name: Name des Werkzeugs ohne Endung, etwa ``"ruff"``.

    Returns:
        Absoluter Pfad zur ausfuehrbaren Datei.

    Raises:
        FileNotFoundError: Wenn das Werkzeug nicht installiert ist.
    """
    scripts = _venv_scripts()
    for suffix in (".exe", ""):
        candidate = scripts / f"{name}{suffix}"
        if candidate.is_file():
            return str(candidate)
    msg = (
        f"{name} ist im .venv nicht installiert. Nachholen mit:\n"
        '  python -m uv pip install --python ".venv\\Scripts\\python.exe" -e ".[dev]"'
    )
    raise FileNotFoundError(msg)


def _python() -> str:
    """Liefert den Interpreter des Projekt-venv.

    Returns:
        Absoluter Pfad zu ``python``.

    Raises:
        FileNotFoundError: Wenn das venv fehlt.
    """
    return _tool("python")


@nox.session(python=False)
def lint(session: nox.Session) -> None:
    """Prueft Stil und Formatierung mit ruff.

    Args:
        session: Die nox-Session.
    """
    ruff = _tool("ruff")
    session.run(ruff, "check", *SOURCES, external=True)
    session.run(ruff, "format", "--check", *SOURCES, external=True)


@nox.session(python=False)
def format(session: nox.Session) -> None:  # noqa: A001 - bewusster Session-Name
    """Formatiert den Code und behebt einfache Lint-Funde.

    Args:
        session: Die nox-Session.
    """
    ruff = _tool("ruff")
    session.run(ruff, "format", *SOURCES, external=True)
    session.run(ruff, "check", "--fix", *SOURCES, external=True)


@nox.session(python=False)
def typecheck(session: nox.Session) -> None:
    """Prueft die Typen mit mypy im Strict-Modus.

    Die Konfiguration steht in ``pyproject.toml``; ``packages`` ist dort
    gesetzt, deshalb bekommt mypy hier keine Pfade uebergeben.

    Args:
        session: Die nox-Session.
    """
    session.run(_tool("mypy"), external=True)


@nox.session(python=False)
def test(session: nox.Session) -> None:
    """Laeuft die Testsuite. Zusaetzliche Argumente gehen an pytest durch.

    Beispiel::

        .venv/Scripts/python.exe -m nox -s test -- -k backtest -x

    Args:
        session: Die nox-Session.
    """
    session.run(_python(), "-m", "pytest", *session.posargs, external=True)


@nox.session(python=False)
def coverage(session: nox.Session) -> None:
    """Laeuft die Suite mit Coverage-Report und Mindestschwelle.

    Schreibt zusaetzlich einen HTML-Report nach ``htmlcov/`` und eine
    ``coverage.xml`` fuer die CI.

    Args:
        session: Die nox-Session.
    """
    session.run(
        _python(),
        "-m",
        "pytest",
        "--cov",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--cov-report=xml",
        f"--cov-fail-under={COVERAGE_MIN}",
        *session.posargs,
        external=True,
    )


@nox.session(python=False, name="all")
def all_checks(session: nox.Session) -> None:
    """Laeuft Lint, Typpruefung und Coverage nacheinander.

    Bricht beim ersten Fehler ab - so steht der erste echte Befund oben in
    der Ausgabe und nicht unter hunderten Folgezeilen.

    Args:
        session: Die nox-Session.
    """
    lint(session)
    typecheck(session)
    coverage(session)


@nox.session(python=False)
def app(session: nox.Session) -> None:
    """Startet das Terminal lokal.

    Bequemlichkeit, kein Pruefschritt: ``all`` ruft diese Session nicht auf.

    Args:
        session: Die nox-Session.
    """
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(ROOT / "src"))
    session.run(_python(), "-m", "btlite", *session.posargs, external=True, env=env)


if __name__ == "__main__":  # pragma: no cover - Bequemlichkeit
    sys.exit(nox.main())
