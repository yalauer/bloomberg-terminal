"""Bewertung gespeicherter Depots zu aktuellen Kursen.

:mod:`btlite.portfolio.store` haelt fest, **was** jemand gekauft hat: Stueck,
Einstandskurs, Kaufdatum. Dieses Modul beantwortet die andere Haelfte der
Frage, naemlich **was es heute wert ist**.

Architekturregel 2: hier wird nicht beschafft, sondern gerechnet. Kurse
kommen als Abbildung Ticker auf Quote herein. Fehlt ein Kurs, bleibt der
Marktwert ``None`` - eine Position mit unbekanntem Kurs als 0 zu bewerten
waere eine stille Falschaussage ueber das Vermoegen des Nutzers.

Mehrere Kaeufe desselben Titels fasst :func:`aggregate_positions` zu einer
Zeile zusammen. Der ausgewiesene Einstandskurs ist dann der
**mengengewichtete Durchschnitt**, nicht der Kurs des ersten Kaufs.

Einheiten:
    - Betraege in der Waehrung der jeweiligen Position.
    - Renditen dezimal (0.12 = 12 Prozent), Verluste negativ.

Dieses Modul bewertet, es empfiehlt nicht. Es leitet aus keiner Zahl eine
Handlung ab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from btlite.core.errors import BTLiteError
from btlite.core.logging import get_logger
from btlite.models.common import BTModel, Currency

if TYPE_CHECKING:
    from collections.abc import Mapping

    from btlite.data.base import ExchangeRateProvider
    from btlite.models.holdings import Holding, Portfolio
    from btlite.models.market import Quote

_LOG = get_logger(__name__)


class PositionValue(BTModel):
    """Bewertete Sammelposition eines Titels.

    Alle Wertfelder sind ``None``, wenn kein Kurs vorlag. Sie sind nie 0,
    weil 0 ein echter Wert waere und hier Unkenntnis gemeint ist.
    """

    ticker: str = Field(description="Kanonisches Symbol.")
    shares: float = Field(description="Summe der Stueckzahlen ueber alle Kaeufe.")
    cost_basis: float | None = Field(
        default=None,
        description=(
            "Gesamter Einstand in der Berichtswaehrung, ohne Gebuehren. None, wenn ein "
            "Kauf in fremder Waehrung nicht umgerechnet werden konnte."
        ),
    )
    average_price: float | None = Field(
        default=None,
        description="Mengengewichteter Einstandskurs je Stueck, None bei 0 Stueck.",
    )
    price: float | None = Field(
        default=None, description="Aktueller Kurs je Stueck, None wenn unbekannt."
    )
    market_value: float | None = Field(
        default=None, description="Stueck mal aktueller Kurs, None wenn kein Kurs vorlag."
    )
    gain_abs: float | None = Field(
        default=None, description="Marktwert minus Einstand, negativ bei Verlust."
    )
    gain_pct: float | None = Field(
        default=None,
        description="Rendite dezimal auf den Einstand, None bei Einstand 0.",
    )
    currency: Currency = Field(
        default=Currency.EUR,
        description="Waehrung der Position. Immer die Berichtswaehrung des Depots.",
    )
    lots: int = Field(default=1, description="Anzahl der zugrunde liegenden Einzelkaeufe.")


class PortfolioValuation(BTModel):
    """Bewertung eines ganzen Depots.

    ``market_value`` summiert nur Positionen mit bekanntem Kurs.
    ``priced_positions`` sagt, wie viele das waren - ohne diese Zahl liesse
    sich eine Teilsumme nicht von einer vollstaendigen unterscheiden.
    """

    name: str = Field(description="Name des Depots.")
    positions: list[PositionValue] = Field(
        default_factory=list, description="Bewertete Positionen, absteigend nach Marktwert."
    )
    cost_basis: float | None = Field(
        default=0.0,
        description=(
            "Gesamter Einstand aller Positionen in der Berichtswaehrung. None, sobald "
            "eine Position nicht umgerechnet werden konnte."
        ),
    )
    market_value: float | None = Field(
        default=None,
        description="Summe der bewertbaren Positionen, None wenn keine bewertbar war.",
    )
    gain_abs: float | None = Field(
        default=None, description="Gesamtergebnis absolut, None wenn unvollstaendig bewertet."
    )
    gain_pct: float | None = Field(
        default=None, description="Gesamtrendite dezimal, None bei Einstand 0."
    )
    currency: Currency = Field(
        default=Currency.EUR,
        description="Waehrung der Summen. Immer die Berichtswaehrung des Depots.",
    )
    unpriced: list[str] = Field(
        default_factory=list, description="Ticker ohne Kurs. Im UI sichtbar zu machen."
    )

    @property
    def is_complete(self) -> bool:
        """Ob jede Position bewertet werden konnte.

        Returns:
            True, wenn kein Ticker ohne Kurs blieb. Nur dann beschreiben die
            Summen das gesamte Depot.
        """
        return not self.unpriced


def convert_cost_basis(
    holding: Holding,
    target: Currency,
    rates: ExchangeRateProvider | None,
) -> float | None:
    """Rechnet den Einstand eines Kaufs in die Berichtswaehrung um.

    Massgeblich ist der Kurs des **Kauftages**, nicht der heutige. Der
    Einstand ist ein historischer Tatbestand: wer am 15. Januar zu 1.09
    umgerechnet gekauft hat, hat genau diesen Betrag aufgewendet. Wuerde
    man ihn taeglich neu umrechnen, aenderte sich der ausgewiesene
    Einstand ohne jeden Vorgang im Depot, und die Rendite verloere den
    Waehrungseffekt, den der Anleger tatsaechlich getragen hat. Die
    Festlegung steht in ``docs/DATA_SOURCES.md``.

    Args:
        holding: Einzelkauf mit Waehrung und Kaufdatum.
        target: Berichtswaehrung des Depots.
        rates: Kursquelle. ``None`` heisst: keine Umrechnung moeglich.

    Returns:
        Einstand in ``target``, oder ``None`` wenn kein Kurs vorlag. Nie
        ein geschaetzter Wert und nie der Betrag ungerechnet - beides
        waere eine stille Falschaussage ueber das Vermoegen.
    """
    if holding.currency == target:
        return holding.cost_basis
    if rates is None:
        return None
    try:
        rate = rates.get_rate(holding.currency, target, holding.buy_date)
    except BTLiteError as exc:
        _LOG.warning(
            "Kurs %s/%s zum %s nicht verfuegbar: %s",
            holding.currency.value,
            target.value,
            holding.buy_date,
            exc.user_message,
        )
        return None
    return holding.cost_basis * rate


def aggregate_positions(
    portfolio: Portfolio,
    rates: ExchangeRateProvider | None = None,
) -> dict[str, tuple[float | None, float | None, int]]:
    """Fasst mehrere Kaeufe desselben Titels zusammen.

    Kaeufe in fremder Waehrung werden zum Kurs ihres Kauftages in die
    Berichtswaehrung des Depots umgerechnet. Ohne das waere die Summe
    eine Addition ungleicher Einheiten - wer SAP in Euro und Apple in
    Dollar haelt, bekaeme eine Zahl ohne Bedeutung.

    Args:
        portfolio: Depot mit Einzelkaeufen.
        rates: Kursquelle fuer die Umrechnung. ``None`` laesst Positionen
            in fremder Waehrung unbewertet, statt sie ungerechnet zu
            addieren.

    Returns:
        Abbildung Ticker auf ``(stueck, einstand, anzahl_kaeufe)``, der
        Einstand in der Berichtswaehrung. Liess sich auch nur ein Kauf
        eines Titels nicht umrechnen, ist sein Einstand ``None`` - eine
        Teilsumme waere weniger wert als eine ehrliche Luecke.
    """
    out: dict[str, tuple[float | None, float | None, int]] = {}
    for holding in portfolio.holdings:
        shares, cost, lots = out.get(holding.ticker, (0.0, 0.0, 0))
        converted = convert_cost_basis(holding, portfolio.currency, rates)
        total = None if cost is None or converted is None else cost + converted
        out[holding.ticker] = (
            (shares or 0.0) + holding.shares,
            total,
            lots + 1,
        )
    return out


def value_portfolio(
    portfolio: Portfolio,
    quotes: Mapping[str, Quote],
    rates: ExchangeRateProvider | None = None,
) -> PortfolioValuation:
    """Bewertet ein Depot zu den uebergebenen Kursen.

    Positionen ohne Kurs behalten ihren Einstand, bekommen aber keinen
    Marktwert. Die Gesamtsummen bleiben in diesem Fall ``None``, weil eine
    Teilsumme als Depotwert irrefuehrend waere; die betroffenen Ticker
    stehen in :attr:`PortfolioValuation.unpriced`.

    Zwei Umrechnungen koennen noetig sein und scheitern unabhaengig
    voneinander: der **Einstand** eines Kaufs in fremder Waehrung (Kurs
    des Kauftages) und der **Marktwert**, wenn die Quelle den Kurs in
    einer anderen Waehrung notiert als das Depot nutzt (aktueller Kurs).
    Scheitert eine davon, bleibt der betroffene Wert ``None`` und der
    Titel gilt als nicht bewertbar.

    Args:
        portfolio: Gespeichertes Depot.
        quotes: Aktuelle Kurse je Ticker. Fehlende Eintraege sind erlaubt.
        rates: Kursquelle fuer die Waehrungsumrechnung. ``None`` bewertet
            nur Positionen, die ohnehin in der Berichtswaehrung stehen.

    Returns:
        Bewertung mit Positionen absteigend nach Marktwert. Positionen ohne
        Kurs stehen am Ende.
    """
    aggregated = aggregate_positions(portfolio, rates)
    positions: list[PositionValue] = []
    unpriced: list[str] = []
    total_cost: float | None = 0.0
    total_value = 0.0

    for ticker, (shares, cost, lots) in aggregated.items():
        quote = quotes.get(ticker)
        # Der Kurs kann in einer anderen Waehrung notieren als das Depot
        # nutzt - eine in Frankfurt gehandelte Aktie in Euro, das Depot in
        # Dollar. Hier zaehlt der AKTUELLE Kurs, anders als beim Einstand:
        # der Marktwert ist eine Aussage ueber heute.
        price = _price_in(quote, portfolio.currency, rates)
        average = cost / shares if cost is not None and shares else None
        market = price * shares if price is not None and shares is not None else None

        gain_abs = market - cost if market is not None and cost is not None else None
        # Ein Einstand von 0 macht die Rendite undefiniert, nicht unendlich.
        gain_pct = gain_abs / cost if gain_abs is not None and cost else None

        positions.append(
            PositionValue(
                ticker=ticker,
                shares=shares or 0.0,
                cost_basis=cost,
                average_price=average,
                price=price,
                market_value=market,
                gain_abs=gain_abs,
                gain_pct=gain_pct,
                # Alle Betraege der Zeile stehen jetzt in der
                # Berichtswaehrung. Die Waehrung der Quelle waere als
                # Beschriftung falsch, sobald umgerechnet wurde.
                currency=portfolio.currency,
                lots=lots,
            )
        )
        if cost is None:
            # Ein nicht umrechenbarer Einstand macht auch die Gesamtsumme
            # unbestimmt. Ihn zu ueberspringen ergaebe eine zu kleine Zahl,
            # die wie ein vollstaendiger Einstand aussieht.
            total_cost = None
        elif total_cost is not None:
            total_cost += cost
        if market is None:
            unpriced.append(ticker)
        else:
            total_value += market

    # Unbewertbare Positionen ans Ende, der Rest absteigend nach Wert.
    positions.sort(key=lambda p: (p.market_value is None, -(p.market_value or 0.0)))

    complete = not unpriced and bool(positions)
    value = total_value if complete else None
    gain = value - total_cost if value is not None and total_cost is not None else None

    return PortfolioValuation(
        name=portfolio.name,
        positions=positions,
        cost_basis=total_cost,
        market_value=value,
        gain_abs=gain,
        gain_pct=gain / total_cost if gain is not None and total_cost else None,
        currency=portfolio.currency,
        unpriced=sorted(unpriced),
    )


def _price_in(
    quote: Quote | None,
    target: Currency,
    rates: ExchangeRateProvider | None,
) -> float | None:
    """Bringt einen Kurs in die Berichtswaehrung.

    Umgerechnet wird zum **aktuellen** Wechselkurs, nicht zu dem eines
    Stichtags: der Marktwert ist eine Aussage darueber, was die Position
    heute wert ist. ``get_rate`` erhaelt deshalb bewusst kein Datum.

    Gegenstueck ist :func:`convert_cost_basis`, das den Kurs des
    Kauftages nimmt. Beide zusammen ergeben eine Rendite, die den
    Waehrungseffekt enthaelt.

    Args:
        quote: Kurs der Quelle, oder ``None`` wenn keiner vorlag.
        target: Berichtswaehrung des Depots.
        rates: Kursquelle. ``None`` heisst: keine Umrechnung moeglich.

    Returns:
        Kurs je Stueck in ``target``, oder ``None`` wenn er fehlt oder
        sich nicht umrechnen laesst.
    """
    if quote is None:
        return None
    if quote.currency == target:
        return quote.price
    if rates is None:
        return None
    try:
        return quote.price * rates.get_rate(quote.currency, target)
    except BTLiteError as exc:
        _LOG.warning(
            "Kurs %s/%s nicht verfuegbar: %s", quote.currency.value, target.value, exc.user_message
        )
        return None
