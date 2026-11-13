"""Einstiegspunkt der Anwendung.

Startet die Dash-App auf ``localhost``. Aufruf::

    uv run btlite
    uv run btlite --port 8060 --debug
"""

from __future__ import annotations

import argparse
import sys

from btlite import __version__
from btlite.config import get_settings
from btlite.core.logging import get_logger, setup_logging


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Liest die Kommandozeilenargumente.

    Args:
        argv: Argumentliste. None nutzt ``sys.argv``.

    Returns:
        Geparste Argumente.
    """
    parser = argparse.ArgumentParser(
        prog="btlite",
        description="BT-Lite - lokales Finanz-Analyse-Terminal. Keine Finanzberatung.",
    )
    parser.add_argument("--host", default=None, help="Bind-Adresse, Default 127.0.0.1")
    parser.add_argument("--port", type=int, default=None, help="TCP-Port, Default 8050")
    parser.add_argument("--debug", action="store_true", help="Dash-Debugmodus mit Hot Reload")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log-Level",
    )
    parser.add_argument("--version", action="version", version=f"btlite {__version__}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Startet das Terminal.

    Args:
        argv: Argumentliste. None nutzt ``sys.argv``.

    Returns:
        Exit-Code des Prozesses.
    """
    args = _parse_args(argv)
    settings = get_settings()

    setup_logging(args.log_level or settings.log_level, as_json=settings.log_json)
    log = get_logger(__name__)

    host = args.host or settings.ui.host
    port = args.port or settings.ui.port
    debug = args.debug or settings.ui.debug

    # Lazy, damit --version und --help ohne Dash-Import funktionieren.
    from btlite.ui.app import create_app

    app = create_app(settings)
    log.info("BT-Lite %s startet auf http://%s:%d", __version__, host, port)
    log.info("Kursdaten sind verzoegert. Werkzeug fuer eigene Recherche, keine Finanzberatung.")

    from btlite.ui.state import get_state

    if get_state().is_offline:
        log.info("Datenquelle: eingefrorene Fixtures. Live-Daten mit BTLITE_DATA_OFFLINE=0")

    # Dash 4: run_server() wurde entfernt, der Einstieg heisst app.run().
    app.run(host=host, port=port, debug=debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
