"""Tests der Depotbewertung.

Der Schwerpunkt liegt auf den Faellen, in denen eine Zahl **nicht**
ausgewiesen werden darf: fehlender Kurs, Einstand 0, leeres Depot. Eine
falsche Zahl ueber das eigene Vermoegen ist schlimmer als eine fehlende.
"""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from btlite.core.errors import ProviderUnavailableError
from btlite.models.common import Currency, DataSource, Provenance
from btlite.models.holdings import Holding, Portfolio
from btlite.models.market import Quote
from btlite.portfolio.valuation import (
    PortfolioValuation,
    aggregate_positions,
    value_portfolio,
)

BUY_DATE = dt.date(2024, 1, 15)


#: Waehrung der Testfaelle. Explizit gesetzt statt aus dem Modell-Default
#: geerbt: die meisten Tests hier sagen nichts ueber Waehrungen aus und
#: sollen von deren Vorgabe auch nicht abhaengen. Wo eine Umrechnung das
#: Pruefziel ist, steht die Waehrung am Testfall selbst.
_BASE: Currency = Currency.USD


def _holding(ticker: str, shares: float, price: float) -> Holding:
    return Holding(ticker=ticker, shares=shares, buy_price=price, buy_date=BUY_DATE, currency=_BASE)


def _portfolio(*holdings: Holding, name: str = "Test") -> Portfolio:
    now = dt.datetime(2026, 9, 20, tzinfo=dt.UTC)
    return Portfolio(
        name=name, holdings=list(holdings), created_at=now, updated_at=now, currency=_BASE
    )


#: Herkunft der Testkurse. Die Bewertung liest sie nicht, das Modell
#: verlangt sie aber - ein Kurs ohne Quelle ist im Projekt unzulaessig.
_PROVENANCE = Provenance(
    source=DataSource.FIXTURE,
    fetched_at=dt.datetime(2026, 9, 20, 12, 0, tzinfo=dt.UTC),
    delay_minutes=15,
)


def _quote(ticker: str, price: float) -> Quote:
    return Quote(
        ticker=ticker,
        price=price,
        currency=_BASE,
        timestamp=dt.datetime(2026, 9, 20, 12, 0, tzinfo=dt.UTC),
        provenance=_PROVENANCE,
    )


class TestAggregation:
    def test_several_buys_of_one_title_become_one_row(self):
        portfolio = _portfolio(_holding("AAPL", 10, 150.0), _holding("AAPL", 3, 210.0))
        aggregated = aggregate_positions(portfolio)
        shares, cost, lots = aggregated["AAPL"]
        assert shares == pytest.approx(13.0)
        assert cost == pytest.approx(10 * 150.0 + 3 * 210.0)
        assert lots == 2

    def test_distinct_titles_stay_separate(self):
        portfolio = _portfolio(_holding("AAPL", 1, 100.0), _holding("MSFT", 1, 200.0))
        assert set(aggregate_positions(portfolio)) == {"AAPL", "MSFT"}


class TestValuation:
    def test_average_price_is_share_weighted(self):
        """Nicht der Mittelwert der Kurse, sondern nach Stueck gewichtet."""
        portfolio = _portfolio(_holding("AAPL", 10, 150.0), _holding("AAPL", 3, 210.0))
        value = value_portfolio(portfolio, {"AAPL": _quote("AAPL", 200.0)})
        position = value.positions[0]
        expected = (10 * 150.0 + 3 * 210.0) / 13
        assert position.average_price == pytest.approx(expected)
        assert position.average_price != pytest.approx((150.0 + 210.0) / 2)

    def test_market_value_and_gain(self):
        portfolio = _portfolio(_holding("AAPL", 10, 100.0))
        value = value_portfolio(portfolio, {"AAPL": _quote("AAPL", 150.0)})
        position = value.positions[0]
        assert position.market_value == pytest.approx(1500.0)
        assert position.gain_abs == pytest.approx(500.0)
        assert position.gain_pct == pytest.approx(0.5)

    def test_loss_is_negative(self):
        portfolio = _portfolio(_holding("AAPL", 10, 200.0))
        value = value_portfolio(portfolio, {"AAPL": _quote("AAPL", 150.0)})
        assert value.positions[0].gain_abs == pytest.approx(-500.0)
        assert value.positions[0].gain_pct == pytest.approx(-0.25)

    def test_totals_match_the_sum_of_positions(self):
        portfolio = _portfolio(_holding("AAPL", 10, 100.0), _holding("MSFT", 5, 200.0))
        quotes = {"AAPL": _quote("AAPL", 150.0), "MSFT": _quote("MSFT", 250.0)}
        value = value_portfolio(portfolio, quotes)
        assert value.cost_basis == pytest.approx(1000.0 + 1000.0)
        assert value.market_value == pytest.approx(1500.0 + 1250.0)
        assert value.gain_abs == pytest.approx(750.0)

    def test_positions_are_sorted_by_market_value(self):
        portfolio = _portfolio(_holding("SMALL", 1, 10.0), _holding("BIG", 100, 10.0))
        quotes = {"SMALL": _quote("SMALL", 10.0), "BIG": _quote("BIG", 10.0)}
        value = value_portfolio(portfolio, quotes)
        assert [p.ticker for p in value.positions] == ["BIG", "SMALL"]


class TestMissingPrices:
    def test_position_without_quote_has_no_market_value(self):
        portfolio = _portfolio(_holding("AAPL", 10, 100.0))
        value = value_portfolio(portfolio, {})
        position = value.positions[0]
        assert position.price is None
        assert position.market_value is None
        assert position.gain_abs is None
        assert position.market_value != 0.0

    def test_cost_basis_survives_a_missing_quote(self):
        """Der Einstand ist bekannt, auch wenn der Kurs es nicht ist."""
        portfolio = _portfolio(_holding("AAPL", 10, 100.0))
        value = value_portfolio(portfolio, {})
        assert value.positions[0].cost_basis == pytest.approx(1000.0)
        assert value.cost_basis == pytest.approx(1000.0)

    def test_partial_pricing_reports_no_total(self):
        """Eine Teilsumme als Depotwert waere eine Falschaussage."""
        portfolio = _portfolio(_holding("AAPL", 10, 100.0), _holding("XXXX", 5, 50.0))
        value = value_portfolio(portfolio, {"AAPL": _quote("AAPL", 150.0)})
        assert value.market_value is None
        assert value.gain_abs is None
        assert value.unpriced == ["XXXX"]
        assert value.is_complete is False

    def test_unpriced_positions_are_listed_last(self):
        portfolio = _portfolio(_holding("AAPL", 1, 100.0), _holding("XXXX", 1, 100.0))
        value = value_portfolio(portfolio, {"AAPL": _quote("AAPL", 100.0)})
        assert value.positions[-1].ticker == "XXXX"

    def test_complete_pricing_reports_a_total(self):
        portfolio = _portfolio(_holding("AAPL", 10, 100.0))
        value = value_portfolio(portfolio, {"AAPL": _quote("AAPL", 150.0)})
        assert value.is_complete is True
        assert value.market_value == pytest.approx(1500.0)


class TestEdgeCases:
    def test_empty_portfolio_is_not_complete(self):
        value = value_portfolio(_portfolio(), {})
        assert value.positions == []
        assert value.market_value is None
        assert value.cost_basis == pytest.approx(0.0)

    def test_zero_buy_price_is_rejected_at_the_contract(self):
        """Ein Kaufkurs von 0 kommt gar nicht erst ins Modell.

        Die Division durch den Einstand ist im Rechenweg dennoch
        abgesichert: ``Holding`` koennte spaeter gelockert werden, und eine
        Rendite auf einen Einstand von 0 ist undefiniert, nicht unendlich.
        """
        with pytest.raises(ValidationError):
            _holding("GIFT", 10, 0.0)

    def test_empty_portfolio_has_no_percentage(self):
        """Ohne Einstand gibt es keine Rendite, auch nicht 0 Prozent."""
        value = value_portfolio(_portfolio(), {})
        assert value.gain_pct is None
        assert value.gain_abs is None

    def test_fractional_shares(self):
        portfolio = _portfolio(_holding("AAPL", 0.5, 100.0))
        value = value_portfolio(portfolio, {"AAPL": _quote("AAPL", 200.0)})
        assert value.positions[0].market_value == pytest.approx(100.0)


class TestAgainstRealFixtures:
    def test_valuation_against_fixture_quotes(self):
        """Querprobe gegen echte Fixture-Kurse statt gegen erfundene."""
        from btlite.data.fixtures import FixtureProvider

        quotes = FixtureProvider().get_quotes(["AAPL", "MSFT"])
        portfolio = _portfolio(
            _holding("AAPL", 10, 150.0),
            _holding("AAPL", 3, 210.0),
            _holding("MSFT", 5, 380.0),
        )
        value = value_portfolio(portfolio, quotes)
        manual = 13 * quotes["AAPL"].price + 5 * quotes["MSFT"].price
        assert value.market_value == pytest.approx(manual)
        assert value.cost_basis == pytest.approx(10 * 150.0 + 3 * 210.0 + 5 * 380.0)
        assert value.is_complete is True

    def test_result_is_a_contract_model(self):
        value = value_portfolio(_portfolio(), {})
        assert isinstance(value, PortfolioValuation)


class _StubRates:
    """Feste Kurse statt Netz. Zaehlt die Abrufe mit.

    1.09 am Kauftag, 1.14 heute - bewusst verschieden, damit ein Test
    auffaellt, der den falschen Kurs nimmt.
    """

    HISTORIC = 1.09
    CURRENT = 1.14

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, object]] = []

    def get_rate(self, base: Currency, quote: Currency, on: dt.date | None = None) -> float:
        self.calls.append((base.value, quote.value, on))
        if base == quote:
            return 1.0
        if base == Currency.EUR and quote == Currency.USD:
            return self.HISTORIC if on is not None else self.CURRENT
        msg = f"Unerwartetes Paar {base.value}/{quote.value}"
        raise AssertionError(msg)


class _BrokenRates:
    """Kursquelle, die nichts liefert - offline oder Quelle nicht erreichbar."""

    def get_rate(self, base: Currency, quote: Currency, on: dt.date | None = None) -> float:
        if base == quote:
            return 1.0
        raise ProviderUnavailableError("Exchange rate unavailable.", detail="Testfall")


def _eur_holding(ticker: str, shares: float, price: float) -> Holding:
    return Holding(
        ticker=ticker,
        shares=shares,
        buy_price=price,
        buy_date=BUY_DATE,
        currency=Currency.EUR,
    )


def _eur_quote(ticker: str, price: float) -> Quote:
    return Quote(
        ticker=ticker,
        price=price,
        currency=Currency.EUR,
        timestamp=_PROVENANCE.fetched_at,
        provenance=_PROVENANCE,
    )


class TestCurrencyConversion:
    """Ein Kauf in Euro muss in der Berichtswaehrung des Depots erscheinen.

    Der Nutzer verlangte: "bei der erstellung des portfolios soll auch euro
    angegeben werden koennen und dann in dollar umgerechnet werden."

    Vorher summierte die Bewertung ueber Waehrungen hinweg, ohne
    umzurechnen - wer SAP in Euro und Apple in Dollar hielt, bekam eine
    Addition ungleicher Einheiten als Depotwert.
    """

    def test_cost_basis_uses_the_rate_of_the_purchase_day(self) -> None:
        """Der Einstand ist ein historischer Tatbestand, kein Tageswert."""
        portfolio = _portfolio(_eur_holding("SAP", 10, 120.0))
        value = value_portfolio(portfolio, {"SAP": _eur_quote("SAP", 135.0)}, _StubRates())
        assert value.positions[0].cost_basis == pytest.approx(1200.0 * _StubRates.HISTORIC)

    def test_market_value_uses_the_current_rate(self) -> None:
        """Der Marktwert ist eine Aussage ueber heute."""
        portfolio = _portfolio(_eur_holding("SAP", 10, 120.0))
        value = value_portfolio(portfolio, {"SAP": _eur_quote("SAP", 135.0)}, _StubRates())
        assert value.positions[0].market_value == pytest.approx(1350.0 * _StubRates.CURRENT)

    def test_the_return_contains_the_currency_effect(self) -> None:
        """Genau deshalb der Kurs des Kauftages: 17.66 statt 12.50 Prozent.

        Wuerde man beide Seiten zum heutigen Kurs rechnen, bliebe der
        reine Kursgewinn von 12.5 Prozent stehen - und der Anleger saehe
        nicht, was er in seiner Berichtswaehrung verdient hat.
        """
        portfolio = _portfolio(_eur_holding("SAP", 10, 120.0))
        value = value_portfolio(portfolio, {"SAP": _eur_quote("SAP", 135.0)}, _StubRates())
        assert value.positions[0].gain_pct == pytest.approx(0.1766, abs=1e-4)

    def test_the_direction_is_not_inverted(self) -> None:
        """Die haeufigste stille Verwechslung: geteilt statt multipliziert.

        Beide Zahlen saehen plausibel aus - 1308 und 1101 sind beides
        Betraege. Nur die Richtung unterscheidet sie.
        """
        portfolio = _portfolio(_eur_holding("SAP", 10, 120.0))
        value = value_portfolio(portfolio, {"SAP": _eur_quote("SAP", 135.0)}, _StubRates())
        cost = value.positions[0].cost_basis
        assert cost is not None
        assert cost > 1200.0, "Euro in Dollar muss mehr werden, nicht weniger"
        assert cost == pytest.approx(1308.0)

    def test_the_row_is_labelled_in_the_report_currency(self) -> None:
        """Nach der Umrechnung waere die Waehrung der Quelle eine Falschangabe."""
        portfolio = _portfolio(_eur_holding("SAP", 10, 120.0))
        value = value_portfolio(portfolio, {"SAP": _eur_quote("SAP", 135.0)}, _StubRates())
        assert value.positions[0].currency == Currency.USD

    def test_same_currency_needs_no_rate_at_all(self) -> None:
        """Ein Dollar-Depot mit Dollar-Kaeufen darf nichts abrufen."""
        rates = _StubRates()
        portfolio = _portfolio(_holding("AAPL", 5, 200.0))
        value_portfolio(portfolio, {"AAPL": _quote("AAPL", 220.0)}, rates)
        assert rates.calls == []

    def test_mixed_currencies_sum_up_correctly(self) -> None:
        """Der eigentliche Fehler vorher: 1200 EUR + 1000 USD = 2200."""
        portfolio = _portfolio(
            _eur_holding("SAP", 10, 120.0),
            _holding("AAPL", 5, 200.0),
        )
        quotes = {"SAP": _eur_quote("SAP", 135.0), "AAPL": _quote("AAPL", 220.0)}
        value = value_portfolio(portfolio, quotes, _StubRates())
        assert value.cost_basis == pytest.approx(1308.0 + 1000.0)
        assert value.cost_basis != pytest.approx(2200.0), "Waehrungen wurden roh addiert"


class TestUnavailableRate:
    """Ohne Kurs wird nicht geschaetzt - eine Luecke ist ehrlicher."""

    def test_missing_rate_leaves_the_position_unvalued(self) -> None:
        portfolio = _portfolio(_eur_holding("SAP", 10, 120.0))
        value = value_portfolio(portfolio, {"SAP": _eur_quote("SAP", 135.0)}, _BrokenRates())
        assert value.positions[0].cost_basis is None
        assert value.positions[0].market_value is None

    def test_missing_rate_never_falls_back_to_one(self) -> None:
        """1.0 waere die gefaehrlichste Ausweichantwort: sie sieht richtig aus."""
        portfolio = _portfolio(_eur_holding("SAP", 10, 120.0))
        value = value_portfolio(portfolio, {"SAP": _eur_quote("SAP", 135.0)}, _BrokenRates())
        assert value.positions[0].cost_basis != pytest.approx(1200.0)

    def test_without_a_rate_provider_nothing_is_guessed(self) -> None:
        portfolio = _portfolio(_eur_holding("SAP", 10, 120.0))
        value = value_portfolio(portfolio, {"SAP": _eur_quote("SAP", 135.0)})
        assert value.positions[0].cost_basis is None

    def test_one_broken_position_does_not_hide_the_others(self) -> None:
        """Der Dollar-Kauf bleibt bewertbar, nur die Summe nicht."""
        portfolio = _portfolio(
            _eur_holding("SAP", 10, 120.0),
            _holding("AAPL", 5, 200.0),
        )
        quotes = {"SAP": _eur_quote("SAP", 135.0), "AAPL": _quote("AAPL", 220.0)}
        value = value_portfolio(portfolio, quotes, _BrokenRates())
        by_ticker = {p.ticker: p for p in value.positions}
        assert by_ticker["AAPL"].market_value == pytest.approx(1100.0)
        assert by_ticker["SAP"].market_value is None

    def test_the_total_is_withheld_not_partially_summed(self) -> None:
        """Eine Teilsumme saehe wie ein vollstaendiger Depotwert aus."""
        portfolio = _portfolio(
            _eur_holding("SAP", 10, 120.0),
            _holding("AAPL", 5, 200.0),
        )
        quotes = {"SAP": _eur_quote("SAP", 135.0), "AAPL": _quote("AAPL", 220.0)}
        value = value_portfolio(portfolio, quotes, _BrokenRates())
        assert value.cost_basis is None
        assert value.market_value is None
        assert value.is_complete is False
        assert "SAP" in value.unpriced


class TestQuoteCurrencyDiffersFromPortfolio:
    """Auch der aktuelle Kurs kann in einer fremden Waehrung notieren."""

    def test_quote_in_another_currency_is_converted(self) -> None:
        """Ein in Frankfurt notierter Kurs in einem Dollar-Depot."""
        portfolio = _portfolio(_holding("SAP", 10, 100.0))
        value = value_portfolio(portfolio, {"SAP": _eur_quote("SAP", 135.0)}, _StubRates())
        assert value.positions[0].market_value == pytest.approx(1350.0 * _StubRates.CURRENT)

    def test_the_current_rate_is_used_for_the_market_value(self) -> None:
        """Kein Datum an den Abruf: der Marktwert gilt heute."""
        rates = _StubRates()
        portfolio = _portfolio(_holding("SAP", 10, 100.0))
        value_portfolio(portfolio, {"SAP": _eur_quote("SAP", 135.0)}, rates)
        assert any(call[2] is None for call in rates.calls), rates.calls
