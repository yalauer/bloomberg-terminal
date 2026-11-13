"""Vertraege der Chart-Setup-Checkliste.

Eine Setup-Checkliste beantwortet je Setup genau eine Frage: ist eine
**messbare Chart-Bedingung** zum Stichtag erfuellt, nicht erfuellt, oder
reicht die Historie nicht fuer eine Aussage. Die Semantik der vier Setups
steht in :mod:`btlite.analytics.setups`.

**Keine Anlageempfehlung.** Diese Modelle tragen bewusst kein
Gesamturteil, keinen Score und keine Ampel. Ein erfuelltes Setup ist eine
Beobachtung ueber den Kursverlauf, keine Aufforderung zu handeln. Ob eine
Bedingung ein Anlass zum Kauf, zum Verkauf oder zu gar nichts ist,
entscheidet dieses Werkzeug nicht.

Einheiten der Messwerte (:class:`MeasurementUnit`):
    - ``decimal``: dezimaler, vorzeichenbehafteter Abstand, ``0.012`` sind
      1.2 Prozent. Negativ bedeutet unterhalb des Bezugswerts.
    - ``index``: Punkte auf einer Indikatorskala, beim RSI 0 bis 100.
    - ``bars``: Anzahl Handelstage (Kerzen).
    - ``multiple``: dimensionsloses Verhaeltnis, ``1.8`` ist das 1.8-fache.
    - ``price``: Kurs in der Waehrung der Kursreihe.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from btlite.models.common import BTModel, Ticker


class SetupStatus(StrEnum):
    """Ergebnis einer einzelnen Setup-Pruefung.

    ``insufficient_data`` ist ausdruecklich **nicht** ``not_met``: bei zu
    kurzer Historie ist keine Aussage moeglich, und das Produkt behauptet
    dann auch keine.
    """

    MET = "met"
    NOT_MET = "not_met"
    INSUFFICIENT_DATA = "insufficient_data"


class SetupKey(StrEnum):
    """Kennung der vier Setups. Stabil, dient als Spalten- und Testschluessel."""

    PULLBACK = "pullback"
    RSI_RECOVERY = "rsi_recovery"
    BREAKOUT = "breakout"
    SUPPORT_MACD = "support_macd"


class MeasurementUnit(StrEnum):
    """Einheit eines Messwerts, siehe Modul-Docstring."""

    DECIMAL = "decimal"
    INDEX = "index"
    BARS = "bars"
    MULTIPLE = "multiple"
    PRICE = "price"


class ZoneKind(StrEnum):
    """Lage einer Kurszone relativ zum Kurs am Stichtag."""

    SUPPORT = "support"
    RESISTANCE = "resistance"


class CrossDirection(StrEnum):
    """Richtung der juengsten MACD-Kreuzung im Beobachtungsfenster."""

    UP = "up"
    DOWN = "down"
    NONE = "none"


def _require_utc(value: datetime | None, *, what: str) -> None:
    """Prueft, dass ein Zeitstempel tz-aware in UTC ist.

    Args:
        value: Zu pruefender Zeitstempel oder None.
        what: Feldname fuer die Fehlermeldung.

    Raises:
        ValueError: Wenn der Zeitstempel naiv ist oder nicht in UTC liegt.
    """
    if value is None:
        return
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        msg = f"{what} muss tz-aware in UTC sein"
        raise ValueError(msg)


class Measurement(BTModel):
    """Ein Messwert einer Setup-Pruefung.

    Der Wert ist ``None``, wenn er nicht berechenbar ist - nie ``0.0`` als
    Platzhalter.
    """

    key: str = Field(min_length=1, description="Stabiler Schluessel, etwa ``dist_sma_50``.")
    label: str = Field(min_length=1, description="Englische Beschriftung fuer das UI.")
    value: float | None = Field(description="Messwert in ``unit``, None wenn nicht berechenbar.")
    unit: MeasurementUnit = Field(description="Einheit des Werts, siehe Modul-Docstring.")


class SetupCheck(BTModel):
    """Ergebnis eines Setups zum Stichtag.

    ``event_*`` beschreibt optional den Zeitpunkt, der das Setup ausmacht
    (RSI-Tief, Ausbruchstag, MACD-Kreuzung). Die Chart-Schicht setzt dort
    einen Marker, ohne selbst etwas nachzuschlagen.
    """

    key: SetupKey = Field(description="Kennung des Setups.")
    label: str = Field(min_length=1, description="Englische Bezeichnung des Setups.")
    status: SetupStatus = Field(description="met, not_met oder insufficient_data.")
    measurements: list[Measurement] = Field(
        default_factory=list, description="Messwerte mit Einheit, in Anzeigereihenfolge."
    )
    explanation: str = Field(
        min_length=1, description="Neutraler englischer Satz: Bedingung, Status, Messwert."
    )
    event_at: datetime | None = Field(
        default=None, description="Zeitpunkt des Ereignisses, tz-aware UTC."
    )
    event_price: float | None = Field(
        default=None, description="Kurs am Ereignis in Kurswaehrung, fuer den Marker."
    )
    event_label: str | None = Field(default=None, description="Beschriftung des Markers.")

    @model_validator(mode="after")
    def _validate_event_at(self) -> Self:
        """Prueft den Ereigniszeitpunkt auf UTC.

        Returns:
            Das validierte Modell.
        """
        _require_utc(self.event_at, what="event_at")
        return self

    def measurement(self, key: str) -> float | None:
        """Liest einen Messwert ueber seinen Schluessel.

        Args:
            key: Schluessel des Messwerts.

        Returns:
            Wert in der Einheit des Messwerts, None wenn fehlend oder
            unbekannt.
        """
        for item in self.measurements:
            if item.key == key:
                return item.value
        return None


class PriceZone(BTModel):
    """Eine Unterstuetzungs- oder Widerstandszone aus bestaetigten Swings.

    ``level`` ist der Mittelwert der zusammengefassten Swing-Niveaus,
    ``lower`` und ``upper`` deren Spanne. Alle Kurse in Kurswaehrung.
    """

    level: float = Field(gt=0, description="Mittleres Niveau der Zone in Kurswaehrung.")
    lower: float = Field(gt=0, description="Niedrigstes Swing-Niveau der Zone.")
    upper: float = Field(gt=0, description="Hoechstes Swing-Niveau der Zone.")
    kind: ZoneKind = Field(description="Unter (support) oder ueber (resistance) dem Kurs.")
    touches: int = Field(ge=1, description="Anzahl bestaetigter Swings in der Zone.")

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        """Prueft ``lower <= level <= upper``.

        Returns:
            Das validierte Modell.

        Raises:
            ValueError: Wenn die Spanne inkonsistent ist.
        """
        if not self.lower <= self.level <= self.upper:
            msg = "PriceZone verlangt lower <= level <= upper"
            raise ValueError(msg)
        return self


class SetupReport(BTModel):
    """Setup-Checkliste eines Titels zum Stichtag.

    Bewusst ohne Gesamturteil: es gibt nur die einzelnen Pruefungen.
    """

    ticker: Ticker = Field(description="Kanonisches Symbol.")
    as_of: datetime = Field(description="Stichtag, Zeitstempel der letzten Bar, tz-aware UTC.")
    price: float = Field(description="Schlusskurs am Stichtag in Kurswaehrung.")
    checks: list[SetupCheck] = Field(description="Eine Pruefung je Setup.")
    zones: list[PriceZone] = Field(
        default_factory=list, description="Alle Zonen, aufsteigend nach Niveau."
    )
    nearest_support: float | None = Field(
        default=None, description="Niveau der naechsten Zone unter dem Kurs."
    )
    nearest_resistance: float | None = Field(
        default=None, description="Niveau der naechsten Zone ueber dem Kurs."
    )
    dist_support: float | None = Field(
        default=None,
        description="(Kurs - Unterstuetzung) / Unterstuetzung, dezimal, >= 0.",
    )
    dist_resistance: float | None = Field(
        default=None,
        description="(Kurs - Widerstand) / Widerstand, dezimal, < 0.",
    )
    macd_cross: CrossDirection = Field(
        default=CrossDirection.NONE,
        description="Richtung der juengsten MACD-Kreuzung im Fenster.",
    )
    bars_since_cross: int | None = Field(
        default=None, ge=0, description="Handelstage seit dieser Kreuzung."
    )

    @model_validator(mode="after")
    def _validate_as_of(self) -> Self:
        """Prueft den Stichtag auf UTC.

        Returns:
            Das validierte Modell.
        """
        _require_utc(self.as_of, what="as_of")
        return self

    def check(self, key: SetupKey) -> SetupCheck | None:
        """Liefert die Pruefung eines Setups.

        Args:
            key: Kennung des Setups.

        Returns:
            Die Pruefung, oder None wenn sie im Bericht fehlt.
        """
        for item in self.checks:
            if item.key == key:
                return item
        return None
