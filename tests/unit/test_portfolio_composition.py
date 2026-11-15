import base64
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from pydantic import ValidationError

from btlite.data.composition_csv import read_upload
from btlite.data.yfinance_provider import YFinanceProvider
from btlite.models.common import AssetClass
from btlite.models.composition import Constituent, FundComposition
from btlite.models.market import Instrument
from btlite.portfolio.composition import UNKNOWN, exposures
from btlite.portfolio.valuation import PortfolioValuation, PositionValue
from btlite.ui.pages.composition import _load, render


@pytest.fixture
def sample(provenance):
    valuation = PortfolioValuation(
        name="Depot",
        market_value=1000,
        positions=[
            PositionValue(ticker="AAPL", shares=1, market_value=200),
            PositionValue(ticker="ETF", shares=8, market_value=800),
        ],
    )
    instruments = {
        "AAPL": Instrument(
            ticker="AAPL", country="United States", sector="Technology", provenance=provenance
        ),
        "ETF": Instrument(
            ticker="ETF", asset_class=AssetClass.ETF, country="Ireland", provenance=provenance
        ),
    }
    funds = {
        "ETF": FundComposition(
            ticker="ETF",
            provenance=provenance,
            holdings=[
                Constituent(
                    ticker="AAPL", weight=0.25, country="United States", sector="Technology"
                ),
                Constituent(ticker="SAP", weight=0.125, country="Germany", sector="Technology"),
            ],
        )
    }
    return valuation, instruments, funds


def test_overlap_preserves_total_and_residual(sample):
    data = exposures(*sample, "company")
    assert data["AAPL"] == {"AAPL": 0.2, "ETF": 0.2}
    assert data["SAP"]["ETF"] == pytest.approx(0.1)
    assert data[UNKNOWN]["ETF"] == pytest.approx(0.5)
    assert sum(sum(parts.values()) for parts in data.values()) == pytest.approx(1)


@pytest.mark.parametrize(
    ("dimension", "label"),
    [
        ("country", "United States"),
        ("region", "Nordamerika"),
        ("sector", "Technology"),
    ],
)
def test_classifications_are_weighted_and_ignore_fund_domicile(sample, dimension, label):
    data = exposures(*sample, dimension)
    assert "Ireland" not in data
    assert sum(data[label].values()) == pytest.approx(0.5 if dimension == "sector" else 0.4)


def test_missing_fund_is_unknown_not_renormalized(sample):
    valuation, instruments, _ = sample
    assert exposures(valuation, instruments, {}, "company")[UNKNOWN] == {"ETF": 0.8}


def test_unknown_country_does_not_erase_company(sample):
    valuation, instruments, funds = sample
    funds["ETF"] = funds["ETF"].model_copy(
        update={"holdings": [Constituent(ticker="AAPL", weight=0.25)]}
    )
    assert sum(exposures(valuation, instruments, funds, "country")[UNKNOWN].values()) == 0.8
    assert sum(exposures(valuation, instruments, funds, "company")["AAPL"].values()) == 0.4


@pytest.mark.parametrize(("total", "unpriced"), [(None, ["ETF"]), (0, []), (1000, ["ETF"])])
def test_incomplete_or_empty_valuation_refused(sample, total, unpriced):
    valuation, instruments, funds = sample
    valuation = valuation.model_copy(update={"market_value": total, "unpriced": unpriced})
    with pytest.raises(ValueError, match="vollständig"):
        exposures(valuation, instruments, funds, "company")


@pytest.mark.parametrize("weights", [[0.7, 0.4], [float("nan")], [-0.1]])
def test_invalid_weights_rejected(provenance, weights):
    with pytest.raises(ValidationError):
        FundComposition(
            ticker="ETF",
            provenance=provenance,
            holdings=[Constituent(ticker=f"X{i}", weight=w) for i, w in enumerate(weights)],
        )


def upload(text):
    return "data:text/csv;base64," + base64.b64encode(text.encode()).decode()


def test_csv_percent_units_and_complete_holdings():
    funds = read_upload(
        upload(
            "fund,ticker,weight_percent,country,sector\n"
            "ETF,AAPL,40,United States,Technology\n"
            "ETF,SAP,60,Germany,Technology\n"
        )
    )
    assert sum(h.weight for h in funds["ETF"].holdings) == 1
    assert funds["ETF"].holdings[0].weight == 0.4
    assert funds["ETF"].provenance.source == "user"


@pytest.mark.parametrize(
    "csv",
    [
        "fund,ticker,weight_percent\nETF,AAPL,80\nETF,SAP,80",
        "fund,ticker,weight_percent\nETF,AAPL,40\nETF,AAPL,40",
        "fund,ticker,weight_percent\nETF,AAPL,NaN",
        "fund,ticker,weight_percent\n",
        "wrong,columns\n1,2",
    ],
)
def test_invalid_csv_rejected(csv):
    with pytest.raises(ValueError, match=r".+"):
        read_upload(upload(csv))


def test_provider_cache_preserves_fetch_time(provenance):
    provider = object.__new__(YFinanceProvider)
    cache = Mock()
    cache.get_df.return_value = None
    provider._cache = cache
    provider._settings = SimpleNamespace(cache=SimpleNamespace(info_ttl_s=3600))
    provider._retry = lambda fn: fn
    provider._provenance = Mock(return_value=provenance)
    provider._ticker = Mock(
        return_value=SimpleNamespace(
            funds_data=SimpleNamespace(
                top_holdings=pd.DataFrame(
                    {"Name": ["Apple"], "Holding Percent": [0.07]}, index=["AAPL"]
                )
            )
        )
    )
    result = provider.get_composition("ETF")
    assert result.holdings[0].weight == 0.07
    cache.get_df.return_value = cache.set_df.call_args.args[1]
    cached = provider.get_composition("ETF")
    assert cached.provenance.from_cache
    assert cached.provenance.fetched_at == result.provenance.fetched_at
    provider._ticker.assert_called_once()


def test_comparison_renders_percentage_points(monkeypatch, sample):
    data = exposures(*sample, "company")
    monkeypatch.setattr("btlite.ui.pages.composition._load", lambda *args: (data, [], {}))
    result = render(Mock(), "Depot A", "Depot B", "company")
    assert "+0.00 PP" in str(result)
    assert "50.00%" in str(result)


def test_no_portfolio_is_actionable():
    assert "Depot auswählen" in str(render(Mock(), None, None, "company"))


def test_imported_fund_bypasses_network_and_preserves_classifications(monkeypatch, sample):
    valuation, instruments, funds = sample
    state = Mock()
    state.holdings.get_portfolio.return_value = SimpleNamespace(tickers=["AAPL", "ETF"])
    state.instrument.side_effect = instruments.__getitem__
    state.quotes.return_value = {}
    state.timezone = "Europe/Berlin"
    monkeypatch.setattr("btlite.ui.pages.composition.value_portfolio", lambda *args: valuation)
    data, notes, _ = _load(state, "Depot", "country", funds)
    assert sum(data["Germany"].values()) == pytest.approx(0.1)
    assert "CSV" in str(notes)
    state.provider.get_composition.assert_not_called()


def test_missing_fund_provider_keeps_explicit_unknown(monkeypatch, sample):
    valuation, instruments, _ = sample
    state = Mock()
    state.provider = object()
    state.composition.return_value = None
    state.equity_isin.return_value = None
    state.timezone = "Europe/Berlin"
    state.holdings.get_portfolio.return_value = SimpleNamespace(tickers=["AAPL", "ETF"])
    state.instrument.side_effect = instruments.__getitem__
    state.quotes.return_value = {}
    monkeypatch.setattr("btlite.ui.pages.composition.value_portfolio", lambda *args: valuation)
    data, notes, _ = _load(state, "Depot", "company", {})
    assert data[UNKNOWN] == {"ETF": 0.8}
    assert "keine ETF-Bestandsquelle" in str(notes)
