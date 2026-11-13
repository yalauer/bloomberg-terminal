"""Logging-Setup fuer BT-Lite.

Ein einziger Einstiegspunkt :func:`setup_logging`, aufgerufen aus
``btlite.__main__``. Module holen ihren Logger ueber :func:`get_logger` und
konfigurieren nie selbst.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    """Formatiert Log-Records als einzeilige JSON-Objekte."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialisiert einen Record nach JSON.

        Args:
            record: Der zu formatierende Log-Record.

        Returns:
            Einzeiliger JSON-String.
        """
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", *, as_json: bool = False) -> None:
    """Konfiguriert das Root-Logging genau einmal pro Prozess.

    Args:
        level: Log-Level als Name, zum Beispiel ``"INFO"``.
        as_json: Wenn True, strukturierte JSON-Zeilen statt Klartext.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stderr)
    if as_json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Fremdbibliotheken sind gespraechig - auf WARNING drosseln.
    for noisy in ("urllib3", "requests_cache", "yfinance", "peewee", "werkzeug", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Liefert den Logger fuer ein Modul.

    Args:
        name: Modulname, in der Regel ``__name__``.

    Returns:
        Konfigurierter Logger.
    """
    return logging.getLogger(name)
