"""Pure portfolio look-through calculations, with explicit unknown exposure."""

import re
from collections import defaultdict
from collections.abc import Mapping

from btlite.models.common import AssetClass
from btlite.models.composition import Constituent, FundComposition
from btlite.models.market import Instrument
from btlite.portfolio.valuation import PortfolioValuation

UNKNOWN = "Nicht aufgeschlüsselt"
REGIONS = {
    "Nordamerika": "United States|Canada|Mexico",
    "Europa": (
        "Germany|France|United Kingdom|Switzerland|Netherlands|Belgium|Luxembourg|"
        "Ireland|Spain|Portugal|Italy|Austria|Sweden|Norway|Denmark|Finland|Poland|"
        "Greece|Czech Republic|Hungary|Romania|Iceland"
    ),
    "Asien": (
        "China|Japan|Taiwan|South Korea|India|Singapore|Hong Kong|Indonesia|Malaysia|"
        "Thailand|Vietnam|Philippines|Israel|Saudi Arabia|United Arab Emirates|Turkey"
    ),
    "Ozeanien": "Australia|New Zealand",
    "Südamerika": "Brazil|Argentina|Chile|Colombia|Peru|Uruguay",
    "Afrika": "South Africa|Egypt|Morocco|Nigeria|Kenya",
}


def region_of(country: str | None) -> str:
    """Map a holding's reported country to a region, never the fund domicile."""
    for region, countries in REGIONS.items():
        if country in countries.split("|"):
            return region
    return UNKNOWN


def _issuer_key(name: str | None, country: str | None) -> tuple[str, str] | None:
    """Create a conservative issuer-name key, requiring a known domicile."""
    if not name or not country:
        return None
    words = re.findall(r"[a-z0-9]+", name.casefold())
    while words and words[-1] in {
        "inc",
        "corp",
        "corporation",
        "ltd",
        "limited",
        "plc",
        "se",
        "ag",
    }:
        words.pop()
    return (" ".join(words), country.casefold()) if words else None


def issuer_isins(funds: Mapping[str, FundComposition]) -> dict[tuple[str, str], str]:
    """Map unique issuer name/country pairs to ISIN; ambiguous pairs stay unresolved."""
    candidates: dict[tuple[str, str], set[str]] = defaultdict(set)
    for fund in funds.values():
        for holding in fund.holdings:
            key = _issuer_key(holding.name, holding.country)
            if key and holding.asset_class == "Equity" and holding.isin:
                candidates[key].add(holding.isin)
    return {key: next(iter(isins)) for key, isins in candidates.items() if len(isins) == 1}


def exposures(
    valuation: PortfolioValuation,
    instruments: Mapping[str, Instrument],
    funds: Mapping[str, FundComposition],
    dimension: str,
) -> dict[str, dict[str, float]]:
    """Return category -> source position -> fraction of total portfolio value.

    Incomplete valuations are rejected. ETF residuals and nested funds remain
    unknown; reported top holdings are never inflated to 100 percent.
    """
    if dimension not in {"company", "country", "region", "sector"}:
        raise ValueError("Unknown composition dimension")
    total = valuation.market_value
    if not valuation.is_complete or total is None or total <= 0:
        raise ValueError("Für die Analyse müssen alle Positionen vollständig bewertet sein.")
    result: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    aliases = issuer_isins(funds) if dimension == "company" else {}
    for position in valuation.positions:
        if position.market_value is None or position.market_value < 0:
            raise ValueError("Position ohne gültigen Marktwert")
        weight = position.market_value / total
        instrument = instruments.get(position.ticker)
        parts: list[Constituent] = []
        if instrument is not None and instrument.asset_class == AssetClass.EQUITY:
            parts = [
                Constituent(
                    ticker=instrument.ticker,
                    isin=instrument.isin,
                    name=instrument.name,
                    weight=1,
                    country=instrument.country,
                    sector=instrument.sector,
                )
            ]
        elif instrument is not None and instrument.asset_class in {AssetClass.ETF, AssetClass.FUND}:
            fund = funds.get(position.ticker)
            if fund is not None:
                parts = fund.holdings
        for part in parts:
            issuer_key = _issuer_key(part.name, part.country)
            mapped_isin = aliases.get(issuer_key) if issuer_key is not None else None
            label = (
                UNKNOWN
                if part.asset_class != "Equity"
                else {
                    "company": part.isin or mapped_isin or part.ticker,
                    "country": part.country or UNKNOWN,
                    "region": region_of(part.country),
                    "sector": part.sector or UNKNOWN,
                }[dimension]
            )
            if part.weight != 0:
                result[label][position.ticker] += weight * part.weight
        residual = 1 - sum(p.weight for p in parts)
        if residual > 1e-10:
            result[UNKNOWN][position.ticker] += weight * residual
    return dict(sorted(result.items(), key=lambda item: -sum(item[1].values())))
