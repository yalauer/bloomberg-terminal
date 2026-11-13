"""Voreinstellungen auf technischen Kennzahlen.

Die Presets in :mod:`btlite.screening.presets` arbeiten auf Bilanzkennzahlen.
Diese hier arbeiten auf den Kursspalten aus
:mod:`btlite.screening.technicals` und beantworten Fragen wie "welche Titel
liegen unter ihrer 200-Tage-Linie".

Alle laufen **rein lokal**. Sie setzen ein Universum voraus, das durch
:func:`~btlite.screening.technicals.enrich_with_technicals` gelaufen ist, und
brauchen deshalb keine Verbindung zur Screener-Quelle.

Zur Leserichtung der Abstaende: ``dist_sma_200`` ist vorzeichenbehaftet und
dezimal. ``-0.08`` bedeutet acht Prozent **unter** der Linie. Ein Praedikat
``LT 0`` trifft damit genau die Titel unterhalb, ohne gedankliche Umkehr.

Diese Presets sind **keine Anlageempfehlung**. Sie filtern nach messbaren
Eigenschaften, mehr nicht. Ob ein Kurs unter seiner Durchschnittslinie ein
Kaufgrund oder eine Warnung ist, entscheidet dieses Werkzeug nicht.
"""

from __future__ import annotations

from btlite.models.screening import (
    FilterGroup,
    FilterPredicate,
    LogicalOp,
    Operator,
    ScreenPreset,
    ScreenRequest,
)


def below_200ma_preset() -> ScreenPreset:
    """Titel unter ihrer 200-Tage-Linie.

    Die vom Nutzer ausdruecklich gewuenschte Abfrage. Zusaetzlich zur Lage
    unter der Linie verlangt das Preset einen nicht voellig ueberverkauften
    RSI, damit nicht ausschliesslich Titel im freien Fall erscheinen.

    Returns:
        Preset, sortiert nach dem Abstand zur Linie: der tiefste Stand oben.
    """
    return ScreenPreset(
        key="below_200ma",
        label="Below 200-day MA",
        description="Price is below the 200-day moving average.",
        request=ScreenRequest(
            name="Below 200-day MA",
            local_filters=FilterGroup(
                op=LogicalOp.AND,
                predicates=[
                    # Negativ heisst unterhalb der Linie.
                    FilterPredicate(field="dist_sma_200", op=Operator.LT, value=0.0),
                    # Unter 20 ist der Titel im freien Fall, das ist eine
                    # andere Situation als eine gewoehnliche Schwaechephase.
                    FilterPredicate(field="rsi_14", op=Operator.GTE, value=20.0),
                ],
            ),
            sort_by="dist_sma_200",
            ascending=True,
            limit=100,
        ),
    )


def above_200ma_preset() -> ScreenPreset:
    """Titel im Aufwaertstrend ueber ihrer 200-Tage-Linie.

    Verlangt zusaetzlich, dass der 50-Tage- ueber dem 200-Tage-Durchschnitt
    liegt. Beides zusammen ist die uebliche Definition eines intakten
    Aufwaertstrends.

    Returns:
        Preset, sortiert nach der Zwoelfmonats-Wertentwicklung.
    """
    return ScreenPreset(
        key="above_200ma",
        label="Above 200-day MA",
        description="Price above the 200-day moving average, 50-day above 200-day.",
        request=ScreenRequest(
            name="Above 200-day MA",
            local_filters=FilterGroup(
                op=LogicalOp.AND,
                predicates=[
                    FilterPredicate(field="dist_sma_200", op=Operator.GT, value=0.0),
                    FilterPredicate(field="golden_cross", op=Operator.EQ, value=1.0),
                ],
            ),
            sort_by="perf_12m",
            ascending=False,
            limit=100,
        ),
    )


def oversold_preset() -> ScreenPreset:
    """Kurzfristig ueberverkaufte Titel in intaktem Aufwaertstrend.

    Die Kombination ist Absicht: ein niedriger RSI allein findet auch
    Titel im Zerfall. Zusammen mit der Lage ueber der 200-Tage-Linie bleibt
    der laengerfristige Trend intakt.

    Returns:
        Preset, sortiert nach dem niedrigsten RSI.
    """
    return ScreenPreset(
        key="oversold",
        label="Oversold in an uptrend",
        description="RSI below 35, price still above the 200-day moving average.",
        request=ScreenRequest(
            name="Oversold in an uptrend",
            local_filters=FilterGroup(
                op=LogicalOp.AND,
                predicates=[
                    FilterPredicate(field="rsi_14", op=Operator.LT, value=35.0),
                    FilterPredicate(field="dist_sma_200", op=Operator.GT, value=0.0),
                ],
            ),
            sort_by="rsi_14",
            ascending=True,
            limit=100,
        ),
    )


def near_52w_high_preset() -> ScreenPreset:
    """Titel nahe ihrem Jahreshoch.

    ``from_high_52w`` ist null oder negativ. Ein Wert ab ``-0.05`` bedeutet
    hoechstens fuenf Prozent unter dem Hoch.

    Returns:
        Preset, sortiert nach der Naehe zum Hoch.
    """
    return ScreenPreset(
        key="near_52w_high",
        label="Near 52-week high",
        description="At most five percent below the 52-week high.",
        request=ScreenRequest(
            name="Near 52-week high",
            local_filters=FilterGroup(
                op=LogicalOp.AND,
                predicates=[
                    FilterPredicate(field="from_high_52w", op=Operator.GTE, value=-0.05),
                ],
            ),
            sort_by="from_high_52w",
            ascending=False,
            limit=100,
        ),
    )


def pullback_preset() -> ScreenPreset:
    """Ruecksetzer in einem laengerfristig intakten Trend.

    Der Kurs liegt unter dem kurzen, aber ueber dem langen Durchschnitt.
    Das beschreibt eine Verschnaufpause, keinen Trendbruch.

    Returns:
        Preset, sortiert nach dem Abstand zur kurzen Linie.
    """
    return ScreenPreset(
        key="pullback",
        label="Pullback in an uptrend",
        description="Below the 20-day but above the 200-day moving average.",
        request=ScreenRequest(
            name="Pullback in an uptrend",
            local_filters=FilterGroup(
                op=LogicalOp.AND,
                predicates=[
                    FilterPredicate(field="dist_sma_20", op=Operator.LT, value=0.0),
                    FilterPredicate(field="dist_sma_200", op=Operator.GT, value=0.0),
                ],
            ),
            sort_by="dist_sma_20",
            ascending=True,
            limit=100,
        ),
    )


def low_volatility_preset() -> ScreenPreset:
    """Schwankungsarme Titel mit begrenztem Rueckgang.

    Returns:
        Preset, sortiert nach der niedrigsten Volatilitaet.
    """
    return ScreenPreset(
        key="low_volatility",
        label="Low volatility",
        description="Annualised volatility below 25 percent, drawdown above -25 percent.",
        request=ScreenRequest(
            name="Low volatility",
            local_filters=FilterGroup(
                op=LogicalOp.AND,
                predicates=[
                    # Volatilitaet annualisiert und dezimal.
                    FilterPredicate(field="volatility_1y", op=Operator.LT, value=0.25),
                    # Drawdown ist negativ; groesser als -0.25 heisst flacher.
                    FilterPredicate(field="max_drawdown_1y", op=Operator.GT, value=-0.25),
                ],
            ),
            sort_by="volatility_1y",
            ascending=True,
            limit=100,
        ),
    )


def all_technical_presets() -> list[ScreenPreset]:
    """Listet alle technischen Voreinstellungen.

    Returns:
        Presets in sinnvoller Anzeigereihenfolge.
    """
    return [
        below_200ma_preset(),
        above_200ma_preset(),
        oversold_preset(),
        pullback_preset(),
        near_52w_high_preset(),
        low_volatility_preset(),
    ]


def technical_preset_keys() -> list[str]:
    """Listet die Schluessel der technischen Presets.

    Returns:
        Sortierte Schluesselliste.
    """
    return sorted(preset.key for preset in all_technical_presets())


def get_technical_preset(key: str) -> ScreenPreset:
    """Sucht ein technisches Preset ueber seinen Schluessel.

    Args:
        key: Schluessel des Presets.

    Returns:
        Das Preset.

    Raises:
        KeyError: Wenn der Schluessel unbekannt ist.
    """
    for preset in all_technical_presets():
        if preset.key == key:
            return preset
    msg = f"Unbekanntes technisches Preset {key!r}. Bekannt: {technical_preset_keys()}"
    raise KeyError(msg)
