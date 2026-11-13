"""Drift-Berechnung und Order-Vorschlaege fuer ein bestehendes Portfolio.

**Nicht-Ziel des Projekts:** keine Broker-Anbindung, keine Orderausfuehrung,
kein Live-Trading. Dieses Modul erzeugt ausschliesslich **Vorschlaege** als
Datenstruktur. Es gibt hier bewusst keine Funktion, die eine Order absendet,
und es soll auch keine dazukommen.

Einheiten: Gewichte und Drift dezimal (0.05 bedeutet 5 Prozentpunkte),
Betraege in der Kontowaehrung, Stueckzahlen in Stueck.

Vorzeichen: ein positiver ``amount`` beziehungsweise ``shares`` ist ein Kauf,
ein negativer ein Verkauf.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from btlite.core.errors import PortfolioError
from btlite.core.logging import get_logger

if TYPE_CHECKING:
    from btlite.models.portfolio import PortfolioResult

_LOG = get_logger(__name__)

#: Spalten der Drift-Tabelle.
CURRENT_WEIGHT_COLUMN: Final[str] = "current_weight"
TARGET_WEIGHT_COLUMN: Final[str] = "target_weight"
DRIFT_COLUMN: Final[str] = "drift"
CURRENT_VALUE_COLUMN: Final[str] = "current_value"
TARGET_VALUE_COLUMN: Final[str] = "target_value"
TRADE_VALUE_COLUMN: Final[str] = "trade_value"

#: Name des Tabellenindex.
TICKER_INDEX_NAME: Final[str] = "ticker"

#: Default-Schwelle, ab der eine Abweichung als handelswuerdig gilt.
#: 0.05 entspricht fuenf Prozentpunkten absoluter Abweichung.
DEFAULT_DRIFT_THRESHOLD: Final[float] = 0.05


class OrderSide(StrEnum):
    """Richtung eines Order-Vorschlags."""

    BUY = "buy"
    SELL = "sell"


class OrderProposal:
    """Ein **Vorschlag** fuer eine Order. Wird nie ausgefuehrt.

    Bewusst eine schlichte Klasse und kein Modell in ``models/``: Vorschlaege
    verlassen die Rechenschicht nur als Anzeige, sie sind kein Datenvertrag
    gegenueber einer externen Quelle.

    Attributes:
        ticker: Kanonisches Symbol.
        side: Kauf oder Verkauf.
        amount: Handelsvolumen in Kontowaehrung, immer positiv.
        shares: Stueckzahl, immer positiv. None wenn kein Kurs vorlag.
        price: Zugrunde gelegter Kurs. None wenn keiner vorlag.
        current_weight: Istgewicht vor dem Trade, dezimal.
        target_weight: Zielgewicht, dezimal.
        drift: Abweichung ``current - target``, dezimal.
    """

    __slots__ = (
        "amount",
        "current_weight",
        "drift",
        "price",
        "shares",
        "side",
        "target_weight",
        "ticker",
    )

    def __init__(
        self,
        *,
        ticker: str,
        side: OrderSide,
        amount: float,
        shares: float | None,
        price: float | None,
        current_weight: float,
        target_weight: float,
        drift: float,
    ) -> None:
        """Initialisiert den Vorschlag.

        Args:
            ticker: Kanonisches Symbol.
            side: Kauf oder Verkauf.
            amount: Handelsvolumen in Kontowaehrung, positiv.
            shares: Stueckzahl, positiv, oder None.
            price: Zugrunde gelegter Kurs, oder None.
            current_weight: Istgewicht, dezimal.
            target_weight: Zielgewicht, dezimal.
            drift: Abweichung ``current - target``, dezimal.
        """
        self.ticker = ticker
        self.side = side
        self.amount = amount
        self.shares = shares
        self.price = price
        self.current_weight = current_weight
        self.target_weight = target_weight
        self.drift = drift

    def __repr__(self) -> str:
        """Kurzdarstellung fuer Logs und Tests.

        Returns:
            Lesbare Repraesentation des Vorschlags.
        """
        return (
            f"OrderProposal({self.ticker} {self.side.value} "
            f"amount={self.amount:.2f} shares={self.shares})"
        )

    def __eq__(self, other: object) -> bool:
        """Vergleicht zwei Vorschlaege feldweise.

        Args:
            other: Vergleichsobjekt.

        Returns:
            True bei Gleichheit aller Felder.
        """
        if not isinstance(other, OrderProposal):
            return NotImplemented
        return all(getattr(self, name) == getattr(other, name) for name in self.__slots__)

    def __hash__(self) -> int:
        """Hash ueber die identifizierenden Felder.

        Returns:
            Hashwert.
        """
        return hash((self.ticker, self.side, self.amount, self.shares))


def current_weights(
    holdings: dict[str, float],
    prices: dict[str, float],
) -> dict[str, float]:
    """Rechnet Stueckzahlen in Istgewichte um.

    Args:
        holdings: Stueckzahlen je Ticker. Negative Werte sind Short-Positionen.
        prices: Aktuelle Kurse je Ticker, in Kontowaehrung.

    Returns:
        Istgewichte je Ticker, dezimal. Leer, wenn der Gesamtwert 0 ist -
        eine Division durch 0 waere hier ein stiller Fehler.

    Raises:
        PortfolioError: Wenn zu einem Bestand der Kurs fehlt.
    """
    missing = sorted(t for t in holdings if t not in prices)
    if missing:
        msg = "The price is missing for a holding."
        raise PortfolioError(msg, detail=f"ohne Kurs: {missing}")

    values = {t: float(n) * float(prices[t]) for t, n in holdings.items()}
    total = sum(values.values())
    if abs(total) < 1e-12:
        _LOG.warning("Portfoliowert ist 0, Istgewichte nicht bestimmbar")
        return {}
    return {t: v / total for t, v in values.items()}


def portfolio_value(
    holdings: dict[str, float],
    prices: dict[str, float],
    *,
    cash: float = 0.0,
) -> float:
    """Berechnet den Gesamtwert eines Portfolios.

    Args:
        holdings: Stueckzahlen je Ticker.
        prices: Aktuelle Kurse je Ticker.
        cash: Zusaetzlicher Barbestand in Kontowaehrung.

    Returns:
        Gesamtwert in Kontowaehrung. Bestaende ohne Kurs zaehlen nicht mit.
    """
    total = float(cash)
    for ticker, count in holdings.items():
        price = prices.get(ticker)
        if price is None:
            _LOG.debug("Bestand %s ohne Kurs, nicht im Portfoliowert", ticker)
            continue
        total += float(count) * float(price)
    return total


def compute_drift(
    current: dict[str, float],
    target: dict[str, float],
    *,
    total_value: float | None = None,
) -> pd.DataFrame:
    """Stellt Ist- und Zielgewichte gegenueber.

    Args:
        current: Istgewichte je Ticker, dezimal.
        target: Zielgewichte je Ticker, dezimal.
        total_value: Portfoliowert in Kontowaehrung. Ist er gesetzt, kommen
            die Wertspalten dazu.

    Returns:
        Tabelle mit Index Ticker und den Spalten ``current_weight``,
        ``target_weight`` und ``drift`` (Ist minus Ziel, dezimal). Mit
        ``total_value`` zusaetzlich ``current_value``, ``target_value`` und
        ``trade_value``. Ein positiver ``trade_value`` ist ein Kauf.
    """
    tickers = sorted(set(current) | set(target))
    index = pd.Index(tickers, name=TICKER_INDEX_NAME)

    current_series = pd.Series(
        {t: float(current.get(t, 0.0)) for t in index}, index=index, dtype="float64"
    )
    target_series = pd.Series(
        {t: float(target.get(t, 0.0)) for t in index}, index=index, dtype="float64"
    )

    table = pd.DataFrame(
        {
            CURRENT_WEIGHT_COLUMN: current_series,
            TARGET_WEIGHT_COLUMN: target_series,
            DRIFT_COLUMN: current_series - target_series,
        }
    )

    if total_value is not None:
        value = float(total_value)
        table[CURRENT_VALUE_COLUMN] = current_series * value
        table[TARGET_VALUE_COLUMN] = target_series * value
        # Positiv = zukaufen, negativ = abbauen.
        table[TRADE_VALUE_COLUMN] = (target_series - current_series) * value

    return table


def needs_rebalancing(
    drift: pd.DataFrame,
    *,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> bool:
    """Prueft, ob irgendeine Position die Drift-Schwelle reisst.

    Args:
        drift: Tabelle aus :func:`compute_drift`.
        threshold: Schwelle der absoluten Abweichung, dezimal.

    Returns:
        True wenn mindestens eine Position betragsmaessig staerker
        abweicht als die Schwelle.
    """
    if drift.empty or DRIFT_COLUMN not in drift.columns:
        return False
    return bool((drift[DRIFT_COLUMN].abs() > threshold).any())


def propose_orders(
    current: dict[str, float],
    target: dict[str, float],
    prices: dict[str, float],
    *,
    total_value: float,
    threshold: float = 0.0,
    whole_shares: bool = True,
    min_order_value: float = 0.0,
) -> list[OrderProposal]:
    """Erzeugt Order-**Vorschlaege**, um das Zielgewicht zu erreichen.

    Ausdruecklich nur Vorschlaege: das Projekt hat keine Broker-Anbindung
    und fuehrt nichts aus. Der Rueckgabewert ist eine Liste von Datensaetzen
    zur Anzeige.

    Args:
        current: Istgewichte je Ticker, dezimal.
        target: Zielgewichte je Ticker, dezimal.
        prices: Aktuelle Kurse je Ticker, in Kontowaehrung.
        total_value: Portfoliowert in Kontowaehrung, Basis der Betraege.
        threshold: Mindestabweichung, ab der ein Vorschlag entsteht,
            dezimal. 0.0 schlaegt jede Abweichung vor.
        whole_shares: Stueckzahlen auf ganze Stuecke abrunden. Ein Bruchteil
            eines Stuecks ist bei den meisten Brokern nicht handelbar.
        min_order_value: Vorschlaege unterhalb dieses Volumens entfallen, in
            Kontowaehrung. Verhindert Kleinstorders, deren Gebuehren den
            Nutzen uebersteigen.

    Returns:
        Vorschlaege, absteigend nach Volumen sortiert. Titel ohne Kurs
        werden uebersprungen und geloggt.

    Raises:
        PortfolioError: Wenn ``total_value`` nicht positiv ist.
    """
    if total_value <= 0.0:
        msg = "The portfolio value must be positive."
        raise PortfolioError(msg, detail=f"total_value={total_value!r}")

    drift = compute_drift(current, target, total_value=total_value)
    proposals: list[OrderProposal] = []

    for ticker in drift.index:
        row = drift.loc[ticker]
        gap = float(row[DRIFT_COLUMN])
        if abs(gap) <= threshold:
            continue

        trade_value = float(row[TRADE_VALUE_COLUMN])
        if not np.isfinite(trade_value) or trade_value == 0.0:
            continue

        price = prices.get(str(ticker))
        shares: float | None = None
        if price is not None and float(price) > 0.0:
            raw = abs(trade_value) / float(price)
            shares = float(np.floor(raw)) if whole_shares else raw
            if whole_shares and shares <= 0.0:
                # Nach dem Abrunden bleibt nichts Handelbares uebrig.
                continue
        else:
            _LOG.debug("Kein Kurs fuer %s, Vorschlag ohne Stueckzahl", ticker)

        amount = abs(trade_value)
        if shares is not None and price is not None:
            # Der Betrag folgt der tatsaechlich handelbaren Stueckzahl.
            amount = shares * float(price)
        if amount < min_order_value:
            continue

        proposals.append(
            OrderProposal(
                ticker=str(ticker),
                side=OrderSide.BUY if trade_value > 0.0 else OrderSide.SELL,
                amount=amount,
                shares=shares,
                price=float(price) if price is not None else None,
                current_weight=float(row[CURRENT_WEIGHT_COLUMN]),
                target_weight=float(row[TARGET_WEIGHT_COLUMN]),
                drift=gap,
            )
        )

    proposals.sort(key=lambda p: p.amount, reverse=True)
    return proposals


def proposals_to_frame(proposals: list[OrderProposal]) -> pd.DataFrame:
    """Wandelt Vorschlaege in eine Tabelle fuer die Anzeige.

    Args:
        proposals: Vorschlaege aus :func:`propose_orders`.

    Returns:
        Tabelle mit Index Ticker und den Spalten ``side``, ``amount``,
        ``shares``, ``price``, ``current_weight``, ``target_weight`` und
        ``drift``. Bei leerer Eingabe eine leere Tabelle mit denselben
        Spalten.
    """
    columns = [
        "side",
        "amount",
        "shares",
        "price",
        CURRENT_WEIGHT_COLUMN,
        TARGET_WEIGHT_COLUMN,
        DRIFT_COLUMN,
    ]
    if not proposals:
        return pd.DataFrame(
            columns=columns, index=pd.Index([], name=TICKER_INDEX_NAME, dtype="object")
        )

    return pd.DataFrame(
        {
            "side": [p.side.value for p in proposals],
            "amount": [p.amount for p in proposals],
            "shares": [p.shares for p in proposals],
            "price": [p.price for p in proposals],
            CURRENT_WEIGHT_COLUMN: [p.current_weight for p in proposals],
            TARGET_WEIGHT_COLUMN: [p.target_weight for p in proposals],
            DRIFT_COLUMN: [p.drift for p in proposals],
        },
        index=pd.Index([p.ticker for p in proposals], name=TICKER_INDEX_NAME),
    )


def rebalance_to_result(
    holdings: dict[str, float],
    result: PortfolioResult,
    prices: dict[str, float],
    *,
    cash: float = 0.0,
    threshold: float = 0.0,
    whole_shares: bool = True,
) -> list[OrderProposal]:
    """Vorschlaege, um einen Bestand auf ein Optimierungsergebnis zu bringen.

    Bequemlichkeits-Fassade fuer den haeufigsten Fall: bestehendes Depot,
    frisch optimierte Zielgewichte.

    Args:
        holdings: Stueckzahlen je Ticker im Bestand.
        result: Optimierungsergebnis mit den Zielgewichten.
        prices: Aktuelle Kurse je Ticker.
        cash: Barbestand in Kontowaehrung, zaehlt zum Portfoliowert.
        threshold: Mindestabweichung fuer einen Vorschlag, dezimal.
        whole_shares: Stueckzahlen auf ganze Stuecke abrunden.

    Returns:
        Order-Vorschlaege, absteigend nach Volumen.

    Raises:
        PortfolioError: Wenn der Portfoliowert nicht positiv ist.
    """
    known = {t: n for t, n in holdings.items() if t in prices}
    unknown = sorted(set(holdings) - set(known))
    if unknown:
        _LOG.warning("Bestaende ohne Kurs bleiben unberuecksichtigt: %s", unknown)

    total = portfolio_value(known, prices, cash=cash)
    if total <= 0.0:
        msg = "The portfolio value must be positive."
        raise PortfolioError(msg, detail=f"wert={total!r}, cash={cash!r}")

    return propose_orders(
        current_weights(known, prices) if known else {},
        result.weights,
        prices,
        total_value=total,
        threshold=threshold,
        whole_shares=whole_shares,
    )
