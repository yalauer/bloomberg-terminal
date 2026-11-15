"""Look-through contracts; all weights are fractions of total fund assets."""

from datetime import date
from typing import Self

from pydantic import Field, field_validator, model_validator

from btlite.models.common import BTModel, Provenance, Ticker


def valid_isin(value: str | None) -> str | None:
    """Return an uppercase ISIN only when its format and Luhn check digit are valid."""
    if value is None:
        return None
    isin = value.strip().upper()
    if len(isin) != 12 or not isin[:2].isalpha() or not isin[2:].isalnum():
        return None
    expanded = "".join(str(ord(c) - 55) if c.isalpha() else c for c in isin)
    digits = [int(c) for c in expanded[::-1]]
    total = sum(
        (2 * digit - 9 if 2 * digit > 9 else 2 * digit) if i % 2 else digit
        for i, digit in enumerate(digits)
    )
    return isin if total % 10 == 0 else None


class Constituent(BTModel):
    """One reported holding, with optional issuer classifications."""

    ticker: Ticker
    isin: str | None = None
    name: str | None = None
    weight: float = Field(ge=-1, le=1, allow_inf_nan=False)
    asset_class: str = "Equity"
    country: str | None = None
    sector: str | None = None

    @field_validator("isin")
    @classmethod
    def check_isin(cls, value: str | None) -> str | None:
        """Reject malformed identifiers instead of silently merging securities."""
        if value is None:
            return None
        valid = valid_isin(value)
        if valid is None:
            raise ValueError("Invalid ISIN")
        return valid


class FundComposition(BTModel):
    """Possibly partial holdings; missing assets must never be renormalized."""

    ticker: Ticker
    holdings: list[Constituent] = Field(default_factory=list)
    as_of: date | None = None
    provenance: Provenance

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        """Reject duplicate symbols and invalid long-only totals."""
        if len({(h.ticker, h.isin, h.asset_class) for h in self.holdings}) != len(self.holdings):
            raise ValueError("Duplicate constituent identities")
        if any(h.weight < 0 and h.asset_class == "Equity" for h in self.holdings):
            raise ValueError("Negative equity holding")
        if sum(h.weight for h in self.holdings) > 1.000001:
            raise ValueError("Constituent weights exceed 100%")
        return self
