"""Public iShares fund holdings, resolved via the issuer's product catalogue."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
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
from btlite.models.market import Instrument

_CATALOG_URL: Final[str] = (
    "https://www.ishares.com/varnish-api/blk-product-screener-server/"
    "api/v1/product-screener/product-data"
)
_PRODUCT_URL: Final[str] = (
    "https://www.ishares.com/varnish-api/uk-retail01-product-data/"
    "product-data/api/v2/get-product-data"
)
_CATALOG_PARAMS: Final[dict[str, str]] = {
    "country": "gb",
    "language": "en",
    "siteName": "ishares-uk",
    "userType": "individual",
}
_EXCHANGE_SUFFIXES: Final[dict[str, str]] = {
    ".L": "London Stock Exchange",
    ".DE": "Deutsche Boerse Xetra",
    ".AS": "Euronext Amsterdam",
    ".MI": "Borsa Italiana",
    ".SW": "SIX Swiss Exchange",
}
_CATALOG_TTL_S: Final[int] = 7 * 24 * 3600


def _base_name(name: str) -> str:
    """Drop only Yahoo's trading-currency/distribution suffix from a fund name."""
    value = re.sub(r"\s+(USD|EUR|GBP)\s*\((Acc|Dist)\)$", "", name, flags=re.I)
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _values(columns: dict[str, Any], field: str, length: int) -> list[Any]:
    """Require a complete, aligned holdings column from the issuer response."""
    data = columns.get(field, {}).get("value")
    if not isinstance(data, list) or len(data) != length:
        raise DataValidationError("iShares-Bestandsdaten sind unvollstaendig.", detail=field)
    return data


def parse_holdings(payload: dict[str, Any], ticker: str) -> FundComposition:
    """Map issuer holdings to validated fractions, retaining non-equity netting."""
    try:
        columns = payload["componentsByNameMap"]["holdings"]["containersByNameMap"]["all"][
            "dataPointsByNameMap"
        ]
        weights = columns["holdingPercent"]["value"]
        if not isinstance(weights, list) or not weights:
            raise ValueError("no holdings")
        length = len(weights)
        tickers = _values(columns, "ticker", length)
        isins = _values(columns, "isin", length)
        names = _values(columns, "issueName", length)
        countries = _values(columns, "countryOfRisk", length)
        sectors = _values(columns, "sectorName", length)
        classes = _values(columns, "assetClass", length)
        as_of = datetime.strptime(str(columns["asOfDate"]["value"]), "%Y%m%d").date()
        grouped: dict[tuple[str, str | None, str], float] = defaultdict(float)
        details: dict[tuple[str, str | None, str], tuple[str | None, str | None, str | None]] = {}
        for symbol, isin, name, country, sector, asset, percent in zip(
            tickers, isins, names, countries, sectors, classes, weights, strict=True
        ):
            identity = (
                normalize_ticker(str(symbol or isin or name)),
                str(isin).upper() if isin and isin != "-" else None,
                str(asset),
            )
            grouped[identity] += float(percent) / 100
            details[identity] = (
                str(name) if name else None,
                str(country) if country else None,
                str(sector) if sector else None,
            )
        holdings = [
            Constituent(
                ticker=symbol,
                isin=isin,
                asset_class=asset,
                weight=weight,
                name=details[key][0],
                country=details[key][1],
                sector=details[key][2],
            )
            for key, weight in grouped.items()
            for symbol, isin, asset in [key]
        ]
        return FundComposition(
            ticker=ticker,
            holdings=holdings,
            as_of=as_of,
            provenance=Provenance(
                source=DataSource.ISHARES,
                fetched_at=utc_now(),
                note=(
                    f"iShares-Bestandsdatum {as_of.isoformat()}; "
                    "vollstaendige veroeffentlichte Liste."
                ),
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, DataValidationError):
            raise
        raise DataValidationError(
            "iShares-Bestandsdaten konnten nicht gelesen werden.", detail=str(exc)
        ) from exc


class IsharesCompositionProvider:
    """Fetch issuer catalogue, exchange listings and full holdings with a local TTL cache."""

    def __init__(
        self, settings: Settings | None = None, cache: DataFrameCache | None = None
    ) -> None:
        """Use the shared settings and persistent cache; offline permits cache hits only."""
        self._settings = settings if settings is not None else get_settings()
        self._cache = cache if cache is not None else DataFrameCache(self._settings.cache)
        self._bucket = get_registry().bucket("ishares")
        self._retry = retry_with_backoff(settings=self._settings.ratelimit)

    def _json(self, url: str, params: dict[str, str], key: str, ttl: int) -> dict[str, Any]:
        """Fetch one issuer JSON document, respecting offline mode and rate limits."""
        cached = self._cache.get_df(key)
        if cached is not None and not cached.empty:
            restored = json.loads(str(cached.iloc[0, 0]))
            if isinstance(restored, dict):
                return restored
        if self._settings.data.offline:
            raise ProviderUnavailableError("iShares-Daten sind offline nicht im Cache.")

        def fetch() -> dict[str, Any]:
            self._bucket.acquire()
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers={"Accept": "application/json", "x-application-id": "pp-ui-csr"},
                    timeout=self._settings.data.request_timeout_s,
                )
                if response.status_code == 429:
                    raise RateLimitError("iShares begrenzt derzeit die Abrufe.")
                if response.status_code >= 500:
                    raise ProviderUnavailableError("iShares ist derzeit nicht erreichbar.")
                if response.status_code != 200:
                    raise DataValidationError(
                        "iShares liefert keine Bestandsdaten.", detail=str(response.status_code)
                    )
                value = response.json()
            except requests.RequestException as exc:
                raise ProviderUnavailableError(
                    "iShares ist derzeit nicht erreichbar.", detail=str(exc)
                ) from exc
            except ValueError as exc:
                raise DataValidationError("iShares hat ungueltiges JSON geliefert.") from exc
            if not isinstance(value, dict):
                raise DataValidationError("iShares hat ungueltige Daten geliefert.")
            return value

        result = self._retry(fetch)()
        self._cache.set_df(key, pd.DataFrame({"json": [json.dumps(result)]}), ttl)
        return result

    def _product(self, product_id: str, component: str) -> dict[str, Any]:
        """Load a verified numeric product ID from the issuer API."""
        if not product_id.isdecimal():
            raise DataValidationError("Ungueltige iShares-Produkt-ID.")
        params = {
            "appSubType": "ISHARES",
            "appType": "PRODUCT_PAGE",
            "component": component,
            "locale": "en_GB",
            "portfolioId": product_id,
            "targetSite": "ishares-uk",
            "userType": "individual",
            "excludeContent": "false",
            "asOfDate": "",
            "includeConfig": "true",
        }
        return self._json(
            _PRODUCT_URL,
            params,
            cache_key("ishares", product_id, component),
            self._settings.cache.composition_ttl_s
            if component == "holdings.all"
            else _CATALOG_TTL_S,
        )

    def _resolve(self, ticker: str, instrument: Instrument) -> str | None:
        """Match ISIN first, then an exact fund name plus a verified exchange listing."""
        catalog = self._json(
            _CATALOG_URL, _CATALOG_PARAMS, cache_key("ishares", "uk-catalog"), _CATALOG_TTL_S
        )
        if instrument.isin:
            matches = [
                pid
                for pid, row in catalog.items()
                if isinstance(row, dict) and row.get("isin") == instrument.isin
            ]
            if len(matches) == 1:
                return matches[0]
        if not instrument.name:
            return None
        base = ticker.split(".")[0].upper()
        candidates = [
            pid
            for pid, row in catalog.items()
            if isinstance(row, dict)
            and _base_name(str(row.get("fundName", ""))) == _base_name(instrument.name)
        ]
        if not candidates:
            return None
        exchange = next(
            (
                value
                for suffix, value in _EXCHANGE_SUFFIXES.items()
                if ticker.upper().endswith(suffix)
            ),
            None,
        )
        matched = []
        for pid in candidates:
            payload = self._product(pid, "listings")
            try:
                columns = payload["componentsByNameMap"]["listings"]["containersByNameMap"][
                    "default"
                ]["dataPointsByNameMap"]
                symbols = columns["ticker"]["value"]
                exchanges = columns["exchange"]["value"]
            except (KeyError, TypeError):
                continue
            if any(
                str(sym).upper() == base and (exchange is None or venue == exchange)
                for sym, venue in zip(symbols, exchanges, strict=False)
            ):
                matched.append(pid)
        return matched[0] if len(matched) == 1 else None

    def get_composition(self, ticker: str, instrument: Instrument) -> FundComposition | None:
        """Return complete holdings for a uniquely identified iShares ETF."""
        if not (instrument.name or "").lower().startswith("ishares"):
            return None
        product_id = self._resolve(normalize_ticker(ticker), instrument)
        if product_id is None:
            return None
        key = cache_key("ishares", "composition", product_id)
        cached = self._cache.get_df(key)
        if cached is not None and not cached.empty:
            saved = FundComposition.model_validate_json(str(cached.iloc[0, 0]))
            return saved.model_copy(
                update={
                    "ticker": normalize_ticker(ticker),
                    "provenance": saved.provenance.model_copy(update={"from_cache": True}),
                }
            )
        payload = self._product(product_id, "holdings.all")
        result = parse_holdings(payload, ticker)
        self._cache.set_df(
            key,
            pd.DataFrame({"model": [result.model_dump_json()]}),
            self._settings.cache.composition_ttl_s,
        )
        return result
