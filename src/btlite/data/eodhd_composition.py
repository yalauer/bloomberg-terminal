"""Optional EODHD full ETF holdings adapter; no key means no requests."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Final

import pandas as pd
import requests

from btlite.config import Settings, get_settings
from btlite.core.cache import DataFrameCache, cache_key
from btlite.core.clock import utc_now
from btlite.core.errors import DataValidationError, ProviderUnavailableError, RateLimitError
from btlite.core.ratelimit import get_registry, retry_with_backoff
from btlite.models.common import DataSource, Provenance, normalize_ticker
from btlite.models.composition import Constituent, FundComposition

_API_ROOT: Final[str] = "https://eodhd.com/api/v1.1/fundamentals/"
_SUFFIXES: Final[dict[str, str]] = {
    ".DE": ".XETRA",
    ".L": ".LSE",
    ".AS": ".AS",
    ".PA": ".PA",
    ".MI": ".MI",
    ".SW": ".SW",
    ".TO": ".TO",
    ".AX": ".AU",
}


def eodhd_symbol(ticker: str) -> str:
    """Convert supported Yahoo market suffixes to EODHD exchange identifiers."""
    symbol = normalize_ticker(ticker)
    for suffix, replacement in _SUFFIXES.items():
        if symbol.endswith(suffix):
            return symbol[: -len(suffix)] + replacement
    return symbol if "." in symbol else symbol + ".US"


def parse_eodhd(payload: dict[str, Any], ticker: str) -> FundComposition | None:
    """Read full holdings, preserving percent units and unknown remainder."""
    data = payload.get("ETF_Data", payload)
    holdings = data.get("Holdings") if isinstance(data, dict) else None
    if not isinstance(holdings, dict) or not holdings:
        return None
    grouped: dict[str, float] = defaultdict(float)
    details: dict[str, dict[str, Any]] = {}
    try:
        for full_symbol, row in holdings.items():
            if not isinstance(row, dict):
                raise ValueError("holding is not an object")
            symbol = normalize_ticker(str(row.get("Code") or full_symbol))
            if row.get("Exchange") not in {"US", None}:
                symbol = normalize_ticker(str(full_symbol))
            percent = row.get("Assets_%")
            if percent is None:
                continue
            grouped[symbol] += float(percent) / 100
            details[symbol] = row
        positions = [
            Constituent(
                ticker=symbol,
                weight=weight,
                name=row.get("Name"),
                country=row.get("Country"),
                sector=row.get("Sector"),
            )
            for symbol, weight in grouped.items()
            for row in [details[symbol]]
        ]
        return FundComposition(
            ticker=ticker,
            holdings=positions,
            provenance=Provenance(
                source=DataSource.EODHD,
                fetched_at=utc_now(),
                note="EODHD-Fondsbestand; Bestandsdatum unbekannt.",
            ),
        )
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            "EODHD-Bestaende konnten nicht gelesen werden.", detail=str(exc)
        ) from exc


class EodhdCompositionProvider:
    """Fetch optional EODHD holdings with bounded requests and local caching."""

    def __init__(
        self, settings: Settings | None = None, cache: DataFrameCache | None = None
    ) -> None:
        """Prepare API access; the token remains in settings and never enters cache keys."""
        self._settings = settings if settings is not None else get_settings()
        self._cache = cache if cache is not None else DataFrameCache(self._settings.cache)
        self._bucket = get_registry().bucket("eodhd")
        self._retry = retry_with_backoff(settings=self._settings.ratelimit)

    def get_composition(self, ticker: str) -> FundComposition | None:
        """Return an available full list or None if the optional token is absent."""
        token = self._settings.data.eodhd_api_token
        if not token:
            return None
        symbol = eodhd_symbol(ticker)
        key = cache_key("eodhd", "composition", symbol)
        cached = self._cache.get_df(key)
        if cached is not None and not cached.empty:
            result = FundComposition.model_validate_json(str(cached.iloc[0, 0]))
            return result.model_copy(
                update={
                    "ticker": normalize_ticker(ticker),
                    "provenance": result.provenance.model_copy(update={"from_cache": True}),
                }
            )
        if self._settings.data.offline:
            raise ProviderUnavailableError("EODHD-Daten sind offline nicht im Cache.")

        def fetch() -> FundComposition | None:
            self._bucket.acquire()
            try:
                response = requests.get(
                    _API_ROOT + symbol,
                    params={"api_token": token, "fmt": "json", "filter": "ETF_Data"},
                    timeout=self._settings.data.request_timeout_s,
                )
                if response.status_code == 429:
                    raise RateLimitError("EODHD begrenzt derzeit die Abrufe.")
                if response.status_code >= 500:
                    raise ProviderUnavailableError("EODHD ist derzeit nicht erreichbar.")
                if response.status_code != 200:
                    raise DataValidationError(
                        "EODHD liefert keine Bestandsdaten.", detail=str(response.status_code)
                    )
                payload = response.json()
            except requests.RequestException as exc:
                raise ProviderUnavailableError(
                    "EODHD ist derzeit nicht erreichbar.", detail=type(exc).__name__
                ) from exc
            except ValueError as exc:
                raise DataValidationError("EODHD hat ungueltiges JSON geliefert.") from exc
            if not isinstance(payload, dict):
                raise DataValidationError("EODHD hat ungueltige Daten geliefert.")
            return parse_eodhd(payload, ticker)

        fetched = self._retry(fetch)()
        if fetched is not None:
            self._cache.set_df(
                key,
                pd.DataFrame({"composition": [fetched.model_dump_json()]}),
                self._settings.cache.composition_ttl_s,
            )
        return fetched
