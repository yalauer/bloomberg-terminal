"""Vier mitgelieferte Screening-Presets.

Value, Quality, Momentum und Dividend als
:class:`~btlite.models.screening.ScreenPreset`. Jedes Preset kombiniert
Finviz-Filter fuer die Vorauswahl bei der Quelle mit lokalen Praedikaten auf
den gecachten Kennzahlen.

Zur Notation der Finviz-Filter, gegen ``finvizfinance`` 1.5 verifiziert:

- Keys sind exakt zu schreiben, inklusive Punkt: ``Market Cap.``, nicht
  ``Market Cap``. Der RSI-Key heisst ``RSI (14)``, nicht ``RSI``.
- Werte sind die **vollen Anzeigetexte** der Quelle, keine Kurzformen.
  ``Market Cap.`` nimmt ``"Mega ($200bln and more)"``; ein blosses
  ``"Mega"`` wird abgelehnt.

Die gueltigen Keys und Werte stehen zur Laufzeit in
``finvizfinance.constants.filter_dict`` beziehungsweise hinter
:meth:`~btlite.data.base.ScreenerProvider.available_filters`. Hier ist
nichts Ungepruefstes hardcodiert: ``tests/unit/test_screening_presets.py``
validiert jedes Preset gegen dieses Dictionary, damit ein Tippfehler beim
Test auffaellt und nicht erst zur Laufzeit gegen die Quelle.

Einheiten der lokalen Praedikate folgen ``analytics/ratios.py``:

- Renditen und Margen **dezimal**: ``0.15`` bedeutet 15 Prozent.
- Vielfache wie KGV oder KBV dimensionslos.
- Verschuldungsquoten dimensionslos.
"""

from __future__ import annotations

from typing import Final

from btlite.core.errors import UnknownFilterError
from btlite.models.screening import (
    FilterGroup,
    FilterPredicate,
    LogicalOp,
    Operator,
    ScreenPreset,
    ScreenRequest,
)

# --------------------------------------------------------------------------
# Finviz-Filterwerte, gegen finvizfinance 1.5 geprueft
# --------------------------------------------------------------------------

#: Liquiditaets- und Groessenuntergrenze, die alle Presets teilen. Sie haelt
#: Micro Caps und illiquide Titel heraus, bei denen die Kennzahlen der
#: Quelle unzuverlaessig und die Spreads gross sind.
_MIN_CAP: Final[str] = "+Mid (over $2bln)"
_MIN_VOLUME: Final[str] = "Over 500K"

_BASE_FILTERS: Final[dict[str, str]] = {
    "Market Cap.": _MIN_CAP,
    "Average Volume": _MIN_VOLUME,
}


def _merge(extra: dict[str, str]) -> dict[str, str]:
    """Setzt Preset-Filter auf die gemeinsame Basis auf.

    Args:
        extra: Zusaetzliche Filter in der Notation der Quelle, also mit den
            echten Keys wie ``"P/E"`` oder ``"RSI (14)"``.

    Returns:
        Kopie der Basisfilter, erweitert um ``extra``. Kollidierende Keys
        gewinnen aus ``extra``.
    """
    return {**_BASE_FILTERS, **extra}


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------


def value_preset() -> ScreenPreset:
    """Baut das Value-Preset.

    Sucht guenstig bewertete, profitable Unternehmen: niedriges KGV und KBV
    bei der Quelle, lokal zusaetzlich eine Obergrenze fuer das
    Kurs-Umsatz-Verhaeltnis und eine positive Eigenkapitalrendite.

    Returns:
        Das Preset. ``sort_by`` sortiert aufsteigend nach KGV, das
        guenstigste Papier steht oben.
    """
    return ScreenPreset(
        key="value",
        label="Value",
        description="Cheaply valued, profitable companies with a low P/E and P/B.",
        request=ScreenRequest(
            name="Value",
            finviz_filters=_merge(
                {
                    "P/E": "Under 15",
                    "P/B": "Under 3",
                    "Return on Equity": "Positive (>0%)",
                }
            ),
            local_filters=FilterGroup(
                op=LogicalOp.AND,
                predicates=[
                    # KGV dimensionslos, Vielfaches des Gewinns. Die enge
                    # Grenze zieht bereits der Finviz-Filter "Under 15"; lokal
                    # bleibt ein weiterer Rahmen, damit der Screen bei einem
                    # hoch bewerteten Markt nicht leer laeuft und die
                    # Sortierung nach KGV noch etwas zu ordnen hat.
                    FilterPredicate(field="pe", op=Operator.BETWEEN, value=0.0, value2=30.0),
                    # KBV dimensionslos.
                    FilterPredicate(field="pb", op=Operator.LTE, value=3.0),
                    # KUV dimensionslos.
                    FilterPredicate(field="ps", op=Operator.LTE, value=3.0),
                    # Eigenkapitalrendite dezimal, 0.0 = 0 Prozent.
                    FilterPredicate(field="roe", op=Operator.GT, value=0.0),
                ],
            ),
            sort_by="pe",
            ascending=True,
            limit=100,
        ),
    )


def quality_preset() -> ScreenPreset:
    """Baut das Quality-Preset.

    Sucht hohe Kapitalrenditen bei geringer Verschuldung und solider
    Liquiditaet - Unternehmen, die ihr Kapital produktiv einsetzen, ohne das
    ueber die Bilanz zu hebeln.

    Returns:
        Das Preset. ``sort_by`` sortiert absteigend nach ROIC.
    """
    return ScreenPreset(
        key="quality",
        label="Quality",
        description="High return on capital with low debt and solid liquidity.",
        request=ScreenRequest(
            name="Quality",
            finviz_filters=_merge(
                {
                    "Return on Equity": "Over +15%",
                    "Return on Investment": "Over +10%",
                    "Debt/Equity": "Under 1",
                    "Current Ratio": "Over 1.5",
                    "Operating Margin": "Over 10%",
                }
            ),
            local_filters=FilterGroup(
                op=LogicalOp.AND,
                predicates=[
                    # Renditen dezimal: 0.15 = 15 Prozent.
                    FilterPredicate(field="roe", op=Operator.GTE, value=0.15),
                    FilterPredicate(field="roic", op=Operator.GTE, value=0.10),
                    FilterPredicate(field="roa", op=Operator.GT, value=0.0),
                    # Verschuldung dimensionslos.
                    FilterPredicate(field="debt_to_equity", op=Operator.LTE, value=1.0),
                    FilterPredicate(field="current_ratio", op=Operator.GTE, value=1.5),
                ],
            ),
            sort_by="roic",
            ascending=False,
            limit=100,
        ),
    )


def momentum_preset() -> ScreenPreset:
    """Baut das Momentum-Preset.

    Sucht Titel im Aufwaertstrend: Kurs ueber der 200-Tage-Linie, positive
    Halbjahresperformance und ein RSI, der noch nicht ueberkauft ist. Der
    RSI-Filter nutzt bewusst ``"Not Overbought (<60)"``, damit das Preset
    nicht genau in die Ueberhitzung hineinkauft.

    Returns:
        Das Preset. Sortiert wird bei der Quelle, lokal bleibt die
        Reihenfolge - Momentum-Spalten liegen nicht in ``ratios.py``.
    """
    return ScreenPreset(
        key="momentum",
        label="Momentum",
        description="Stocks in an uptrend above the 200-day MA, not yet overbought.",
        request=ScreenRequest(
            name="Momentum",
            finviz_filters=_merge(
                {
                    "200-Day Simple Moving Average": "Price above SMA200",
                    "50-Day Simple Moving Average": "Price above SMA50",
                    "Performance": "Half Up",
                    "RSI (14)": "Not Overbought (<60)",
                }
            ),
            # Lokal wird nur die Profitabilitaet geprueft. Kursbasierte
            # Momentum-Kennzahlen stehen nicht im RatioSet, sie kommen aus
            # analytics/technicals.py und damit nicht aus diesem Universum.
            local_filters=FilterGroup(
                op=LogicalOp.AND,
                predicates=[
                    FilterPredicate(field="roe", op=Operator.GT, value=0.0),
                ],
            ),
            sort_by=None,
            ascending=False,
            limit=100,
        ),
    )


def dividend_preset() -> ScreenPreset:
    """Baut das Dividend-Preset.

    Sucht tragfaehige Ausschuettungen: Rendite ueber 3 Prozent bei einer
    Ausschuettungsquote unter 80 Prozent. Die Quotengrenze ist der Kern des
    Presets - eine hohe Rendite bei einer Quote ueber 100 Prozent ist ein
    Warnsignal, keine Qualitaet.

    Returns:
        Das Preset. ``sort_by`` sortiert absteigend nach Dividendenrendite.
    """
    return ScreenPreset(
        key="dividend",
        label="Dividend",
        description="Sustainable payouts: yield above 3 percent, payout ratio below 80 percent.",
        request=ScreenRequest(
            name="Dividend",
            finviz_filters=_merge(
                {
                    "Dividend Yield": "Over 3%",
                    "Payout Ratio": "Under 80%",
                    "Return on Equity": "Positive (>0%)",
                    "Debt/Equity": "Under 1",
                }
            ),
            local_filters=FilterGroup(
                op=LogicalOp.AND,
                predicates=[
                    # Dividendenrendite dezimal: 0.015 = 1.5 Prozent. Die
                    # 3-Prozent-Huerde steht im Finviz-Filter und wirkt dort
                    # auf das volle Universum. Lokal genuegt eine niedrigere
                    # Schwelle als Sicherheitsnetz: sie soll Titel ohne
                    # nennenswerte Ausschuettung aussortieren, nicht die
                    # Auswahl der Quelle ein zweites Mal treffen.
                    FilterPredicate(field="dividend_yield", op=Operator.GTE, value=0.015),
                    FilterPredicate(field="roe", op=Operator.GT, value=0.0),
                    FilterPredicate(field="debt_to_equity", op=Operator.LTE, value=1.0),
                ],
            ),
            sort_by="dividend_yield",
            ascending=False,
            limit=100,
        ),
    )


def all_presets() -> list[ScreenPreset]:
    """Listet alle mitgelieferten Presets in Anzeigereihenfolge.

    Returns:
        Die vier Presets Value, Quality, Momentum, Dividend.
    """
    return [value_preset(), quality_preset(), momentum_preset(), dividend_preset()]


def preset_keys() -> list[str]:
    """Listet die stabilen Bezeichner aller Presets.

    Returns:
        Die Keys in Anzeigereihenfolge.
    """
    return [preset.key for preset in all_presets()]


def get_preset(key: str) -> ScreenPreset:
    """Holt ein Preset ueber seinen stabilen Bezeichner.

    Args:
        key: Bezeichner, etwa ``"value"``. Gross- und Kleinschreibung wird
            ignoriert.

    Returns:
        Das passende Preset.

    Raises:
        UnknownFilterError: Wenn der Bezeichner unbekannt ist. Die Meldung
            nennt die gueltigen Keys.
    """
    wanted = key.strip().lower()
    for preset in all_presets():
        if preset.key == wanted:
            return preset
    msg = f"Preset {key!r} is unknown."
    raise UnknownFilterError(msg, detail=f"gueltige Keys: {preset_keys()}")
