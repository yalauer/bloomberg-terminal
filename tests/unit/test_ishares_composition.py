"""Issuer data mapping and fallback behavior without external HTTP calls."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from btlite.config import Settings
from btlite.core.errors import DataValidationError, ProviderUnavailableError, RateLimitError
from btlite.data.eodhd_composition import EodhdCompositionProvider, eodhd_symbol, parse_eodhd
from btlite.data.ishares_provider import IsharesCompositionProvider, parse_holdings
from btlite.models.common import AssetClass
from btlite.models.composition import Constituent, FundComposition, valid_isin
from btlite.models.market import Instrument
from btlite.portfolio.composition import UNKNOWN, exposures, issuer_isins
from btlite.portfolio.valuation import PortfolioValuation, PositionValue
from btlite.ui.state import AppState


def issuer_payload():
    values = {
        "asOfDate": 20261112,
        "ticker": ["AAPL", "SAP", "USD"],
        "isin": ["US0378331005", "DE0007164600", None],
        "issueName": ["APPLE", "SAP SE", "USD CASH"],
        "holdingPercent": [60, 42, -2],
        "countryOfRisk": ["United States", "Germany", "United States"],
        "sectorName": ["Information Technology", "Information Technology", "Cash"],
        "assetClass": ["Equity", "Equity", "Cash"],
    }
    return {
        "componentsByNameMap": {
            "holdings": {
                "containersByNameMap": {
                    "all": {"dataPointsByNameMap": {k: {"value": v} for k, v in values.items()}}
                }
            }
        }
    }


def instrument(provenance):
    return Instrument(
        ticker="EUNL.DE",
        name="iShares Core MSCI World UCITS ETF USD (Acc)",
        asset_class=AssetClass.ETF,
        provenance=provenance,
    )


def test_issuer_snapshot_keeps_signed_cash_and_date():
    fund = parse_holdings(issuer_payload(), "EUNL.DE")
    assert fund.as_of == date(2026, 11, 12)
    assert sum(h.weight for h in fund.holdings) == pytest.approx(1)
    assert fund.holdings[0].isin == "US0378331005"
    assert fund.holdings[-1].weight == pytest.approx(-0.02)
    assert fund.provenance.source.value == "ishares"


def test_signed_cash_not_misreported_as_company(provenance):
    valuation = PortfolioValuation(
        name="Depot",
        market_value=100,
        positions=[PositionValue(ticker="EUNL.DE", shares=1, market_value=100)],
    )
    data = exposures(
        valuation,
        {"EUNL.DE": instrument(provenance)},
        {"EUNL.DE": parse_holdings(issuer_payload(), "EUNL.DE")},
        "company",
    )
    assert data["US0378331005"]["EUNL.DE"] == pytest.approx(0.6)
    assert data[UNKNOWN]["EUNL.DE"] == pytest.approx(-0.02)
    assert sum(sum(v.values()) for v in data.values()) == pytest.approx(1)


def test_direct_listing_and_etf_merge_by_isin(provenance):
    valuation = PortfolioValuation(
        name="Depot",
        market_value=200,
        positions=[
            PositionValue(ticker="APC.DE", shares=1, market_value=100),
            PositionValue(ticker="EUNL.DE", shares=1, market_value=100),
        ],
    )
    direct = Instrument(
        ticker="APC.DE",
        name="Apple Inc.",
        country="United States",
        asset_class=AssetClass.EQUITY,
        provenance=provenance,
    )
    data = exposures(
        valuation,
        {"APC.DE": direct, "EUNL.DE": instrument(provenance)},
        {"EUNL.DE": parse_holdings(issuer_payload(), "EUNL.DE")},
        "company",
    )
    assert sum(data["US0378331005"].values()) == pytest.approx(0.8)
    assert set(data["US0378331005"]) == {"APC.DE", "EUNL.DE"}


def test_ambiguous_issuer_names_do_not_merge(provenance):
    fund = FundComposition(
        ticker="ETF",
        provenance=provenance,
        holdings=[
            Constituent(
                ticker="AAA",
                name="Example Inc.",
                country="United States",
                isin="US0378331005",
                weight=0.4,
            ),
            Constituent(
                ticker="BBB",
                name="Example Corp",
                country="United States",
                isin="US5949181045",
                weight=0.6,
            ),
        ],
    )
    assert issuer_isins({"ETF": fund}) == {}


def test_isin_checksum():
    assert valid_isin("us0378331005") == "US0378331005"
    assert valid_isin("US0378331000") is None
    assert valid_isin("-") is None


def test_catalogue_resolution_checks_exchange_and_ambiguity(provenance):
    settings = Settings()
    provider = IsharesCompositionProvider(settings, cache=Mock())
    provider._json = Mock(
        return_value={
            "251882": {"isin": "IE00B4L5Y983", "fundName": "iShares Core MSCI World UCITS ETF"},
            "332655": {"isin": "IE000OHHIBC6", "fundName": "iShares Core MSCI World UCITS ETF"},
        }
    )

    def product(pid, component):
        symbol = "EUNL" if pid == "251882" else "IWLE"
        return {
            "componentsByNameMap": {
                "listings": {
                    "containersByNameMap": {
                        "default": {
                            "dataPointsByNameMap": {
                                "ticker": {"value": [symbol]},
                                "exchange": {"value": ["Deutsche Boerse Xetra"]},
                            }
                        }
                    }
                }
            }
        }

    provider._product = product
    assert provider._resolve("EUNL.DE", instrument(provenance)) == "251882"
    assert provider._resolve("UNKNOWN.DE", instrument(provenance)) is None
    by_isin = instrument(provenance).model_copy(update={"isin": "IE00B4L5Y983"})
    assert provider._resolve("UNKNOWN.DE", by_isin) == "251882"


def test_issuer_composition_cache_preserves_original_time(provenance):
    cache = Mock()
    cache.get_df.return_value = None
    provider = IsharesCompositionProvider(Settings(), cache=cache)
    provider._resolve = Mock(return_value="251882")
    provider._product = Mock(return_value=issuer_payload())
    first = provider.get_composition("EUNL.DE", instrument(provenance))
    assert first is not None
    cache.get_df.return_value = cache.set_df.call_args.args[1]
    second = provider.get_composition("SWDA.L", instrument(provenance))
    assert second is not None
    assert second.provenance.from_cache
    assert second.provenance.fetched_at == first.provenance.fetched_at
    assert second.ticker == "SWDA.L"
    provider._product.assert_called_once()


def test_offline_cache_miss_never_requests_network(monkeypatch):
    settings = Settings()
    settings.data.offline = True
    provider = IsharesCompositionProvider(settings, cache=Mock(get_df=Mock(return_value=None)))
    request = Mock(side_effect=AssertionError("network called"))
    monkeypatch.setattr("btlite.data.ishares_provider.requests.get", request)
    with pytest.raises(ProviderUnavailableError, match="offline"):
        provider._json("https://www.ishares.com/", {}, "missing", 100)
    request.assert_not_called()


def test_malformed_and_ambiguous_holdings_rejected():
    with pytest.raises(DataValidationError):
        parse_holdings({"wrong": "shape"}, "EUNL.DE")
    payload = issuer_payload()
    payload["componentsByNameMap"]["holdings"]["containersByNameMap"]["all"]["dataPointsByNameMap"][
        "ticker"
    ]["value"] = ["AAPL"]
    with pytest.raises(DataValidationError):
        parse_holdings(payload, "EUNL.DE")


def test_yahoo_fallback_is_labeled_and_rate_limit_propagates(provenance):
    state = object.__new__(AppState)
    state._ishares = Mock()
    state._ishares.get_composition.side_effect = ProviderUnavailableError("issuer unavailable")
    yahoo = FundComposition(
        ticker="EUNL.DE", provenance=provenance, holdings=[Constituent(ticker="AAPL", weight=0.07)]
    )
    state._provider = SimpleNamespace(get_composition=Mock(return_value=yahoo))
    result = state.composition("EUNL.DE", instrument(provenance))
    assert result is not None
    assert "iShares nicht verfuegbar" in result.provenance.note
    state._ishares.get_composition.side_effect = RateLimitError("throttled")
    with pytest.raises(RateLimitError):
        state.composition("EUNL.DE", instrument(provenance))


def test_optional_eodhd_provider_keeps_missing_remainder():
    payload = {
        "Holdings": {
            "AAPL.US": {
                "Code": "AAPL",
                "Exchange": "US",
                "Name": "Apple Inc.",
                "Country": "United States",
                "Sector": "Technology",
                "Assets_%": 40.0,
            },
            "SAP.XETRA": {
                "Code": "SAP",
                "Exchange": "XETRA",
                "Name": "SAP SE",
                "Country": "Germany",
                "Sector": "Technology",
                "Assets_%": 35.0,
            },
        }
    }
    fund = parse_eodhd(payload, "ETF.DE")
    assert fund is not None
    assert sum(h.weight for h in fund.holdings) == pytest.approx(0.75)
    assert {h.ticker for h in fund.holdings} == {"AAPL", "SAP.XETRA"}
    assert fund.provenance.source.value == "eodhd"
    assert eodhd_symbol("EUNL.DE") == "EUNL.XETRA"


def test_eodhd_without_key_does_not_request_network(monkeypatch):
    settings = Settings()
    settings.data.eodhd_api_token = None
    provider = EodhdCompositionProvider(settings)
    request = Mock(side_effect=AssertionError("network called"))
    monkeypatch.setattr("btlite.data.eodhd_composition.requests.get", request)
    assert provider.get_composition("VTI") is None
    request.assert_not_called()
