"""Tests der JSON-Persistenz von Depots und Watchlists.

Alle Tests schreiben ausschliesslich nach ``tmp_path``. Das echte
Nutzerverzeichnis wird nie angefasst - ein Test, der ein reales Depot
ueberschreibt, waere der Fehler, gegen den dieses Modul schuetzen soll.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from btlite.core.errors import PortfolioError
from btlite.models.common import Currency
from btlite.models.holdings import (
    HOLDINGS_SCHEMA_VERSION,
    Holding,
    HoldingsFile,
    Portfolio,
    Watchlist,
)
from btlite.portfolio.store import HoldingsStore


@pytest.fixture
def store(tmp_path):
    """Store auf einer Wegwerf-Datei."""
    return HoldingsStore(tmp_path / "holdings.json")


@pytest.fixture
def filled(store):
    """Store mit einem Depot, zwei Positionen und einer Watchlist."""
    store.create_portfolio("Depot")
    store.add_holding(
        "Depot",
        "AAPL",
        shares=10.0,
        buy_price=180.5,
        buy_date=date(2024, 3, 1),
        # Bewusst abweichend von der Depotwaehrung: nur so prueft der
        # Roundtrip die Waehrung wirklich. Wuerde sie aus dem Default
        # kommen, bestuende der Test auch dann, wenn sie beim Speichern
        # verlorengeht.
        currency=Currency.USD,
        note="Kernwert",
    )
    store.add_holding("Depot", "MSFT", shares=2.5, buy_price=400.0, buy_date=date(2024, 5, 2))
    store.create_watchlist("Beobachtung")
    store.add_to_watchlist("Beobachtung", "NVDA", note="zu teuer")
    return store


class TestLoad:
    def test_missing_file_is_empty_not_an_error(self, store):
        """Beim ersten Start gibt es nichts zu laden - das ist kein Fehler."""
        data = store.load()
        assert data.is_empty
        assert data.portfolios == {}
        assert data.watchlists == {}
        assert not store.path.exists()

    def test_broken_json_raises_and_does_not_overwrite(self, store):
        """Ein Tippfehler in der Datei darf das Depot nicht loeschen."""
        store.path.parent.mkdir(parents=True, exist_ok=True)
        broken = '{"schema_version": 1, "portfolios": {'
        store.path.write_text(broken, encoding="utf-8")

        with pytest.raises(PortfolioError, match="is corrupt and was not changed"):
            store.load()

        assert store.path.read_text(encoding="utf-8") == broken

    def test_foreign_format_raises(self, store):
        """Eine JSON-Liste ist kein Bestand."""
        store.path.parent.mkdir(parents=True, exist_ok=True)
        store.path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(PortfolioError, match="has a foreign format"):
            store.load()

    def test_missing_schema_version_raises(self, store):
        store.path.parent.mkdir(parents=True, exist_ok=True)
        store.path.write_text('{"portfolios": {}, "watchlists": {}}', encoding="utf-8")
        with pytest.raises(PortfolioError, match="carries no valid schema version"):
            store.load()

    def test_future_schema_version_raises(self, store):
        """Eine neuere Version still zu lesen waere Raten."""
        store.path.parent.mkdir(parents=True, exist_ok=True)
        future = HOLDINGS_SCHEMA_VERSION + 1
        store.path.write_text(
            f'{{"schema_version": {future}, "portfolios": {{}}, "watchlists": {{}}}}',
            encoding="utf-8",
        )
        with pytest.raises(PortfolioError, match="was written by a newer version of BT-Lite"):
            store.load()

    def test_contract_violation_raises_and_keeps_file(self, store):
        """Unbekannte Felder sind ein Vertragsbruch, kein Grund zum Reset."""
        store.path.parent.mkdir(parents=True, exist_ok=True)
        payload = '{"schema_version": 1, "portfolios": {}, "watchlists": {}, "fremd": 1}'
        store.path.write_text(payload, encoding="utf-8")

        with pytest.raises(PortfolioError, match="does not match the expected format"):
            store.load()

        assert store.path.read_text(encoding="utf-8") == payload


class TestRoundtrip:
    def test_all_values_survive(self, filled):
        """Speichern und Laden erhaelt jeden Wert."""
        data = filled.load()
        holding = data.portfolios["Depot"].holdings[0]

        assert holding.ticker == "AAPL"
        assert holding.shares == 10.0
        assert holding.buy_price == 180.5
        assert holding.buy_date == date(2024, 3, 1)
        assert holding.note == "Kernwert"
        assert holding.currency is Currency.USD

    def test_schema_version_is_written(self, filled):
        assert filled.load().schema_version == HOLDINGS_SCHEMA_VERSION

    def test_updated_at_is_tz_aware_utc(self, filled):
        updated = filled.load().updated_at
        assert updated is not None
        assert updated.tzinfo is not None
        assert updated.utcoffset() == UTC.utcoffset(None)

    def test_file_is_human_readable(self, filled):
        """Die Datei muss von Hand korrigierbar sein - also eingerueckt."""
        text = filled.path.read_text(encoding="utf-8")
        assert "\n  " in text
        assert "AAPL" in text

    def test_watchlist_entries_survive(self, filled):
        watchlist = filled.get_watchlist("Beobachtung")
        assert watchlist.tickers == ["NVDA"]
        assert watchlist.entries[0].note == "zu teuer"
        assert watchlist.entries[0].added_at.tzinfo is not None


class TestAtomicWrite:
    def test_interrupted_write_leaves_old_file_intact(self, filled, monkeypatch):
        """Ein Abbruch beim Umbenennen darf den alten Stand nicht zerstoeren."""
        import pathlib

        before = filled.path.read_text(encoding="utf-8")

        def _boom(self, target):
            msg = "Simulierter Abbruch beim Umbenennen"
            raise OSError(msg)

        monkeypatch.setattr(pathlib.Path, "replace", _boom)

        data = filled.load()
        data.portfolios["Depot"].holdings.clear()
        with pytest.raises(
            PortfolioError, match=r"could not be written\. The previous state is unchanged"
        ):
            filled.save(data)

        assert filled.path.read_text(encoding="utf-8") == before
        assert len(filled.load().portfolios["Depot"].holdings) == 2

    def test_no_temp_file_is_left_behind(self, filled, monkeypatch):
        """Nach einem gescheiterten Schreibvorgang bleibt kein Muell liegen."""
        import pathlib

        def _boom(self, target):
            msg = "Simulierter Abbruch"
            raise OSError(msg)

        monkeypatch.setattr(pathlib.Path, "replace", _boom)

        with pytest.raises(PortfolioError):
            filled.save(filled.load())

        leftovers = [p for p in filled.path.parent.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == []

    def test_successful_write_leaves_no_temp_file(self, filled):
        leftovers = [p for p in filled.path.parent.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == []

    def test_parent_directory_is_created(self, tmp_path):
        store = HoldingsStore(tmp_path / "tief" / "verschachtelt" / "holdings.json")
        store.create_portfolio("Depot")
        assert store.path.exists()


class TestPortfolioOperations:
    def test_multiple_portfolios_side_by_side(self, store):
        """Mehrere benannte Depots stehen unabhaengig nebeneinander."""
        store.create_portfolio("Altersvorsorge")
        store.create_portfolio("Spielgeld")
        store.add_holding(
            "Spielgeld", "TSLA", shares=1.0, buy_price=250.0, buy_date=date(2024, 6, 1)
        )

        assert store.list_portfolios() == ["Altersvorsorge", "Spielgeld"]
        assert store.get_portfolio("Altersvorsorge").holdings == []
        assert len(store.get_portfolio("Spielgeld").holdings) == 1

    def test_duplicate_portfolio_name_raises(self, store):
        store.create_portfolio("Depot")
        with pytest.raises(PortfolioError, match=r"A portfolio named 'Depot' already exists"):
            store.create_portfolio("Depot")

    def test_delete_portfolio(self, filled):
        filled.delete_portfolio("Depot")
        assert filled.list_portfolios() == []
        assert filled.list_watchlists() == ["Beobachtung"]

    def test_delete_unknown_portfolio_raises(self, store):
        with pytest.raises(PortfolioError, match="No portfolio named"):
            store.delete_portfolio("gibtsnicht")

    def test_add_holding_to_unknown_portfolio_raises(self, store):
        with pytest.raises(PortfolioError, match="No portfolio named"):
            store.add_holding("Depot", "AAPL", shares=1.0, buy_price=1.0, buy_date=date(2024, 1, 1))

    def test_ticker_is_normalized(self, store):
        store.create_portfolio("Depot")
        store.add_holding("Depot", "  aapl  ", shares=1.0, buy_price=1.0, buy_date=date(2024, 1, 1))
        assert store.get_portfolio("Depot").holdings[0].ticker == "AAPL"

    def test_same_ticker_twice_stays_two_holdings(self, store):
        """Zwei Kaeufe zu verschiedenen Kursen sind zwei Haltungen."""
        store.create_portfolio("Depot")
        store.add_holding("Depot", "AAPL", shares=5.0, buy_price=150.0, buy_date=date(2023, 1, 5))
        store.add_holding("Depot", "AAPL", shares=5.0, buy_price=200.0, buy_date=date(2024, 1, 5))

        portfolio = store.get_portfolio("Depot")
        assert len(portfolio.holdings) == 2
        assert portfolio.shares_of("AAPL") == 10.0
        assert portfolio.cost_basis == pytest.approx(1750.0)

    def test_remove_holding_by_index(self, filled):
        portfolio = filled.remove_holding("Depot", 0)
        assert [h.ticker for h in portfolio.holdings] == ["MSFT"]

    def test_remove_holding_out_of_range_raises(self, filled):
        with pytest.raises(PortfolioError, match="does not exist in portfolio"):
            filled.remove_holding("Depot", 5)

    def test_update_holding_changes_only_given_fields(self, filled):
        portfolio = filled.update_holding("Depot", 0, shares=12.0)
        holding = portfolio.holdings[0]

        assert holding.shares == 12.0
        assert holding.buy_price == 180.5
        assert holding.note == "Kernwert"
        assert holding.ticker == "AAPL"

    def test_update_holding_persists(self, filled):
        filled.update_holding("Depot", 1, buy_price=350.0, buy_date=date(2024, 7, 7))
        holding = filled.load().portfolios["Depot"].holdings[1]
        assert holding.buy_price == 350.0
        assert holding.buy_date == date(2024, 7, 7)

    def test_update_holding_out_of_range_raises(self, filled):
        with pytest.raises(PortfolioError, match="does not exist in portfolio"):
            filled.update_holding("Depot", 9, shares=1.0)

    def test_updated_at_moves_forward(self, filled):
        before = filled.get_portfolio("Depot").updated_at
        filled.add_holding("Depot", "KO", shares=1.0, buy_price=60.0, buy_date=date(2024, 8, 1))
        assert filled.get_portfolio("Depot").updated_at >= before

    def test_unknown_portfolio_returns_none(self, store):
        assert store.get_portfolio("gibtsnicht") is None


class TestValidation:
    def test_fractional_shares_are_allowed(self, store):
        """Teilaktien sind der Normalfall bei Sparplaenen."""
        store.create_portfolio("Depot")
        store.add_holding(
            "Depot", "AMZN", shares=0.3742, buy_price=185.25, buy_date=date(2024, 2, 14)
        )
        assert store.get_portfolio("Depot").holdings[0].shares == pytest.approx(0.3742)

    @pytest.mark.parametrize("shares", [-1.0, 0.0])
    def test_non_positive_shares_rejected(self, store, shares):
        store.create_portfolio("Depot")
        with pytest.raises(PortfolioError, match=r"The position details are invalid\."):
            store.add_holding(
                "Depot", "AAPL", shares=shares, buy_price=100.0, buy_date=date(2024, 1, 1)
            )

    @pytest.mark.parametrize("price", [-10.0, 0.0])
    def test_non_positive_price_rejected(self, store, price):
        store.create_portfolio("Depot")
        with pytest.raises(PortfolioError, match=r"The position details are invalid\."):
            store.add_holding(
                "Depot", "AAPL", shares=1.0, buy_price=price, buy_date=date(2024, 1, 1)
            )

    def test_rejected_holding_does_not_touch_file(self, filled):
        before = filled.path.read_text(encoding="utf-8")
        with pytest.raises(PortfolioError):
            filled.add_holding(
                "Depot", "AAPL", shares=-1.0, buy_price=100.0, buy_date=date(2024, 1, 1)
            )
        assert filled.path.read_text(encoding="utf-8") == before

    def test_update_to_negative_shares_rejected(self, filled):
        with pytest.raises(PortfolioError, match=r"The position details are invalid\."):
            filled.update_holding("Depot", 0, shares=-5.0)


class TestWatchlist:
    def test_multiple_watchlists_side_by_side(self, store):
        store.create_watchlist("Tech")
        store.create_watchlist("Dividende")
        store.add_to_watchlist("Tech", "NVDA")
        store.add_to_watchlist("Dividende", "KO")

        assert store.list_watchlists() == ["Dividende", "Tech"]
        assert store.get_watchlist("Tech").tickers == ["NVDA"]
        assert store.get_watchlist("Dividende").tickers == ["KO"]

    def test_duplicate_watchlist_name_raises(self, store):
        store.create_watchlist("Tech")
        with pytest.raises(PortfolioError, match=r"A watchlist named 'Tech' already exists"):
            store.create_watchlist("Tech")

    def test_ticker_is_normalized(self, store):
        store.create_watchlist("Tech")
        store.add_to_watchlist("Tech", " nvda ")
        assert store.get_watchlist("Tech").tickers == ["NVDA"]

    def test_duplicate_ticker_raises(self, filled):
        with pytest.raises(PortfolioError, match="is already on watchlist"):
            filled.add_to_watchlist("Beobachtung", "nvda")

    def test_remove_entry(self, filled):
        watchlist = filled.remove_from_watchlist("Beobachtung", "NVDA")
        assert watchlist.tickers == []
        assert filled.get_watchlist("Beobachtung").tickers == []

    def test_remove_unknown_entry_raises(self, filled):
        with pytest.raises(PortfolioError, match="is not on watchlist"):
            filled.remove_from_watchlist("Beobachtung", "TSLA")

    def test_unknown_watchlist_raises(self, store):
        with pytest.raises(PortfolioError, match="No watchlist named"):
            store.add_to_watchlist("gibtsnicht", "AAPL")

    def test_delete_watchlist(self, filled):
        filled.delete_watchlist("Beobachtung")
        assert filled.list_watchlists() == []
        assert filled.list_portfolios() == ["Depot"]

    def test_unknown_watchlist_returns_none(self, store):
        assert store.get_watchlist("gibtsnicht") is None


class TestDefaultCurrency:
    """Euro ist die Standardwaehrung - dort, wo der Nutzer sie waehlt.

    Die Trennlinie ist wichtig: Modelle, die beschreiben, was eine QUELLE
    geliefert hat (Quote, PriceSeries, die Statements), behalten USD.
    yfinance liefert die Waehrung mit, und bei US-Titeln ist sie nun
    einmal Dollar - ein Euro-Default waere dort eine Falschangabe ueber
    fremde Daten.
    """

    def test_a_new_portfolio_reports_in_euro(self, store):
        """Ohne Angabe fuehrt ein Depot Euro."""
        assert store.create_portfolio("Neu").currency is Currency.EUR

    def test_a_new_holding_is_bought_in_euro(self, store):
        """Ohne Angabe gilt der Kauf als Euro-Kauf."""
        store.create_portfolio("Neu")
        store.add_holding("Neu", "SAP.DE", shares=1, buy_price=100.0, buy_date=date(2024, 1, 15))
        assert store.get_portfolio("Neu").holdings[0].currency is Currency.EUR

    def test_an_explicit_currency_still_wins(self, store):
        """Der Default ist eine Vorgabe, keine Festlegung."""
        store.create_portfolio("Dollar", currency=Currency.USD)
        store.add_holding(
            "Dollar",
            "AAPL",
            shares=1,
            buy_price=100.0,
            buy_date=date(2024, 1, 15),
            currency=Currency.USD,
        )
        portfolio = store.get_portfolio("Dollar")
        assert portfolio.currency is Currency.USD
        assert portfolio.holdings[0].currency is Currency.USD

    def test_a_holding_inherits_the_portfolio_currency(self, store):
        """Ohne eigene Angabe folgt der Kauf dem Depot, nicht dem Modell.

        Wer sein Depot bewusst auf Dollar stellt, will dort nicht bei
        jeder Position erneut USD auswaehlen muessen.
        """
        store.create_portfolio("Dollar", currency=Currency.USD)
        store.add_holding("Dollar", "AAPL", shares=1, buy_price=100.0, buy_date=date(2024, 1, 15))
        assert store.get_portfolio("Dollar").holdings[0].currency is Currency.USD

    def test_the_holding_model_itself_defaults_to_euro(self):
        """Direkt geprueft, nicht ueber den Store.

        ``add_holding`` setzt die Waehrung immer aus dem Depot, der
        Modell-Default kommt dort nie zum Zug. Er gilt aber beim Laden
        einer Datei, in der das Feld fehlt - also genau dann, wenn
        niemand hinschaut.
        """
        holding = Holding(ticker="SAP.DE", shares=1.0, buy_price=100.0, buy_date=date(2024, 1, 15))
        assert holding.currency is Currency.EUR

    def test_the_portfolio_model_itself_defaults_to_euro(self):
        now = datetime(2026, 9, 20, tzinfo=UTC)
        assert Portfolio(name="X", created_at=now, updated_at=now).currency is Currency.EUR

    def test_a_file_without_the_field_loads_as_euro(self, tmp_path):
        """Der praktische Fall: eine Datei aus der Zeit vor dem Feld."""
        import json

        path = tmp_path / "alt.json"
        now = "2026-09-20T00:00:00Z"
        path.write_text(
            json.dumps(
                {
                    "schema_version": HOLDINGS_SCHEMA_VERSION,
                    "portfolios": {
                        "Alt": {
                            "name": "Alt",
                            "holdings": [
                                {
                                    "ticker": "SAP.DE",
                                    "shares": 1.0,
                                    "buy_price": 100.0,
                                    "buy_date": "2024-01-15",
                                }
                            ],
                            "created_at": now,
                            "updated_at": now,
                        }
                    },
                    "watchlists": {},
                }
            ),
            encoding="utf-8",
        )
        loaded = HoldingsStore(path).get_portfolio("Alt")
        assert loaded is not None
        assert loaded.currency is Currency.EUR
        assert loaded.holdings[0].currency is Currency.EUR

    def test_source_models_keep_the_dollar_default(self):
        """Gegenprobe: die Waehrung einer Quelle wird nicht umdefiniert."""
        from btlite.models.market import Instrument, PriceSeries, Quote

        for model in (Instrument, Quote, PriceSeries):
            field = model.model_fields["currency"]
            assert field.default is Currency.USD, (
                f"{model.__name__}.currency traegt {field.default} - Quellen-Modelle behalten USD"
            )


class TestWatchlistNote:
    """Notizen sind aenderbar, ohne den Eintrag neu anzulegen.

    Die Oberflaeche zeigt die Notiz als Eingabefeld. Ohne eigene Methode
    muesste sie den Eintrag loeschen und neu aufnehmen - das verschoebe das
    Aufnahmedatum und macht aus einer Korrektur eine Neuaufnahme.
    """

    def test_note_can_be_changed(self, filled):
        filled.set_watchlist_note("Beobachtung", "NVDA", "beobachten")
        assert filled.get_watchlist("Beobachtung").entries[0].note == "beobachten"

    def test_empty_note_clears_it(self, filled):
        """Ein leergeraeumtes Feld bedeutet "keine Notiz", nicht "Leerstring"."""
        filled.set_watchlist_note("Beobachtung", "NVDA", "   ")
        assert filled.get_watchlist("Beobachtung").entries[0].note is None

    def test_none_clears_it(self, filled):
        filled.set_watchlist_note("Beobachtung", "NVDA", None)
        assert filled.get_watchlist("Beobachtung").entries[0].note is None

    def test_the_entry_date_survives(self, filled):
        """Eine Notiz zu tippen ist keine Neuaufnahme des Titels."""
        before = filled.get_watchlist("Beobachtung").entries[0].added_at
        filled.set_watchlist_note("Beobachtung", "NVDA", "geaendert")
        assert filled.get_watchlist("Beobachtung").entries[0].added_at == before

    def test_the_ticker_survives(self, filled):
        filled.set_watchlist_note("Beobachtung", "NVDA", "geaendert")
        entries = filled.get_watchlist("Beobachtung").entries
        assert len(entries) == 1
        assert entries[0].ticker == "NVDA"

    def test_ticker_is_normalized(self, filled):
        """Kleinschreibung darf nicht "nicht enthalten" bedeuten."""
        filled.set_watchlist_note("Beobachtung", "nvda", "klein geschrieben")
        assert filled.get_watchlist("Beobachtung").entries[0].note == "klein geschrieben"

    def test_unknown_ticker_raises(self, filled):
        with pytest.raises(PortfolioError):
            filled.set_watchlist_note("Beobachtung", "AAPL", "nicht drin")

    def test_unknown_watchlist_raises(self, filled):
        with pytest.raises(PortfolioError):
            filled.set_watchlist_note("Fehlt", "NVDA", "x")

    def test_too_long_a_note_raises(self, filled):
        """500 Zeichen ist die Modellgrenze; 501 muss auffallen."""
        with pytest.raises(PortfolioError):
            filled.set_watchlist_note("Beobachtung", "NVDA", "x" * 501)

    def test_a_rejected_note_does_not_reach_the_file(self, filled):
        with pytest.raises(PortfolioError):
            filled.set_watchlist_note("Beobachtung", "NVDA", "x" * 501)
        assert filled.get_watchlist("Beobachtung").entries[0].note == "zu teuer"


class TestHoldingCurrencyIsCorrectable:
    """Die Waehrung einer Position laesst sich nachtraeglich richtigstellen.

    Der Ticker ist es bewusst nicht - ein anderes Symbol waere ein anderer
    Kauf. Die Waehrung dagegen ist eine Angabe ueber denselben Kauf: wer
    beim Erfassen den Depot-Default USD stehen liess, obwohl er in Euro
    gekauft hat, korrigiert einen Tippfehler.
    """

    def test_currency_can_be_corrected(self, filled):
        filled.update_holding("Depot", 0, currency=Currency.EUR)
        assert filled.get_portfolio("Depot").holdings[0].currency is Currency.EUR

    def test_the_other_fields_are_untouched(self, filled):
        before = filled.get_portfolio("Depot").holdings[0]
        filled.update_holding("Depot", 0, currency=Currency.EUR)
        after = filled.get_portfolio("Depot").holdings[0]
        assert (after.shares, after.buy_price, after.buy_date, after.ticker) == (
            before.shares,
            before.buy_price,
            before.buy_date,
            before.ticker,
        )

    def test_omitting_the_currency_keeps_it(self, filled):
        """Nicht uebergebene Felder bleiben unveraendert - auch dieses."""
        before = filled.get_portfolio("Depot").holdings[0].currency
        filled.update_holding("Depot", 0, shares=99.0)
        assert filled.get_portfolio("Depot").holdings[0].currency is before

    def test_only_the_addressed_holding_changes(self, filled):
        before = filled.get_portfolio("Depot").holdings[1].currency
        filled.update_holding("Depot", 0, currency=Currency.EUR)
        assert filled.get_portfolio("Depot").holdings[1].currency is before

    def test_the_change_survives_a_reload(self, filled, tmp_path):
        filled.update_holding("Depot", 0, currency=Currency.EUR)
        fresh = HoldingsStore(tmp_path / "holdings.json")
        assert fresh.get_portfolio("Depot").holdings[0].currency is Currency.EUR


class TestModels:
    def test_cost_basis_of_holding(self):
        holding = Holding(ticker="AAPL", shares=2.5, buy_price=200.0, buy_date=date(2024, 1, 1))
        assert holding.cost_basis == pytest.approx(500.0)

    def test_portfolio_cost_basis_by_ticker(self):
        now = datetime(2025, 1, 1, tzinfo=UTC)
        portfolio = Portfolio(
            name="Depot",
            created_at=now,
            updated_at=now,
            holdings=[
                Holding(ticker="AAPL", shares=1.0, buy_price=100.0, buy_date=date(2024, 1, 1)),
                Holding(ticker="AAPL", shares=2.0, buy_price=150.0, buy_date=date(2024, 6, 1)),
                Holding(ticker="KO", shares=4.0, buy_price=50.0, buy_date=date(2024, 6, 1)),
            ],
        )
        assert portfolio.cost_basis_by_ticker() == {"AAPL": 400.0, "KO": 200.0}
        assert portfolio.cost_basis == pytest.approx(600.0)
        assert portfolio.tickers == ["AAPL", "KO"]
        assert portfolio.is_single_currency

    def test_portfolio_rejects_naive_timestamp(self):
        with pytest.raises(ValueError, match="tz-aware"):
            Portfolio(
                name="Depot",
                created_at=datetime(2025, 1, 1),
                updated_at=datetime(2025, 1, 1, tzinfo=UTC),
            )

    def test_watchlist_rejects_duplicate_tickers(self):
        from btlite.models.holdings import WatchlistEntry

        now = datetime(2025, 1, 1, tzinfo=UTC)
        with pytest.raises(ValueError, match="doppelte Symbole"):
            Watchlist(
                name="Tech",
                created_at=now,
                updated_at=now,
                entries=[
                    WatchlistEntry(ticker="NVDA", added_at=now),
                    WatchlistEntry(ticker="nvda", added_at=now),
                ],
            )

    def test_holdings_file_rejects_key_name_mismatch(self):
        now = datetime(2025, 1, 1, tzinfo=UTC)
        portfolio = Portfolio(name="Depot", created_at=now, updated_at=now)
        with pytest.raises(ValueError, match="passt nicht zum Namen"):
            HoldingsFile(portfolios={"Anders": portfolio})

    def test_shares_of_unknown_ticker_is_zero(self):
        now = datetime(2025, 1, 1, tzinfo=UTC)
        portfolio = Portfolio(name="Depot", created_at=now, updated_at=now)
        assert portfolio.shares_of("AAPL") == 0.0


class TestDefaultPath:
    def test_default_comes_from_settings_and_is_not_touched(self, monkeypatch, tmp_path):
        """Der Default kommt aus den Settings, nicht aus einem Literal."""
        from btlite.config import get_settings

        target = tmp_path / "aus_settings.json"
        settings = get_settings()
        monkeypatch.setattr(settings.portfolio, "holdings_file", target)

        store = HoldingsStore()
        assert store.path == target
        assert store.load().is_empty
