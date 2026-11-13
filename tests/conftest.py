"""Gemeinsame Fixtures. Kein Test in dieser Suite geht ins Netz."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from btlite.models.common import DataSource, Provenance

#: Adressen, die trotz Netzwerksperre erlaubt bleiben.
#: asyncio baut auf Windows ein lokales Socket-Paar fuer den Selbstaufweck-
#: Mechanismus der Eventloop. Das ist kein Netzwerkzugriff im Sinne des
#: Akzeptanzkriteriums, wuerde von einer pauschalen Sperre aber getroffen und
#: liesse sich in async-Tests nicht umgehen.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", ""})


def _is_loopback(address: object) -> bool:
    """Prueft, ob ein Socket-Ziel auf dem lokalen Rechner liegt.

    Args:
        address: Zieladresse, wie sie ``socket.connect`` entgegennimmt.

    Returns:
        True fuer Loopback-Ziele und Unix-Sockets, sonst False.
    """
    if isinstance(address, str):  # AF_UNIX
        return True
    if isinstance(address, tuple) and address:
        return str(address[0]) in _LOOPBACK_HOSTS
    return False


@pytest.fixture(autouse=True)
def _forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blockiert echte Socket-Verbindungen in der gesamten Suite.

    Akzeptanzkriterium des Projekts: alle Tests laufen offline. Ein
    versehentlicher Zugriff auf eine externe Adresse schlaegt hier laut fehl,
    statt still langsam zu werden.

    Loopback bleibt erlaubt. Sonst waeren async-Tests unmoeglich, weil die
    asyncio-Eventloop auf Windows intern ein lokales Socket-Paar oeffnet.
    """
    import socket

    real_connect = socket.socket.connect
    real_create_connection = socket.create_connection

    def _guarded_connect(self: socket.socket, address: object, /) -> object:
        if _is_loopback(address):
            return real_connect(self, address)
        msg = f"Netzwerkzugriff im Test ({address!r}) - Fixtures verwenden"
        raise RuntimeError(msg)

    def _guarded_create_connection(address: object, *args: object, **kwargs: object) -> object:
        if _is_loopback(address):
            return real_create_connection(address, *args, **kwargs)  # type: ignore[arg-type]
        msg = f"Netzwerkzugriff im Test ({address!r}) - Fixtures verwenden"
        raise RuntimeError(msg)

    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket, "create_connection", _guarded_create_connection)


@pytest.fixture
def provenance() -> Provenance:
    """Herkunftsnachweis fuer Fixture-Daten."""
    return Provenance(
        source=DataSource.FIXTURE,
        fetched_at=datetime(2025, 1, 2, 12, 0, tzinfo=UTC),
        delay_minutes=15,
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """Deterministische OHLCV-Reihe ueber 250 Handelstage."""
    idx = pd.date_range("2023-01-02", periods=250, freq="B", tz="UTC", name="date")
    rng = np.random.default_rng(42)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(idx))))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, len(idx))),
            "high": close * (1 + abs(rng.normal(0, 0.006, len(idx)))),
            "low": close * (1 - abs(rng.normal(0, 0.006, len(idx)))),
            "close": close,
            "volume": rng.integers(1_000_000, 9_000_000, len(idx)).astype(float),
        },
        index=idx,
    )


@pytest.fixture
def statement_frame() -> pd.DataFrame:
    """Statement-Payload: Index = Positionen, Spalten = Perioden."""
    cols = pd.DatetimeIndex(["2022-12-31", "2023-12-31", "2024-12-31"], tz="UTC", name="date")
    return pd.DataFrame(
        {
            cols[0]: [3.0e11, 9.9e10, 1.2e11, 5.4e10],
            cols[1]: [3.8e11, 1.0e11, 1.3e11, 6.2e10],
            cols[2]: [3.9e11, 9.4e10, 1.4e11, np.nan],
        },
        index=["revenue", "net_income", "total_assets", "operating_cashflow"],
    )
