"""Zusaetzliche Kursfilter auf einem beliebigen Preset.

Ein Preset beantwortet eine feste Frage. Die Einstellung aus
:class:`~btlite.models.screening.TechnicalRefinement` schraenkt sie weiter
ein, etwa "Value, aber nur unter der 200-Tage-Linie und mit RSI unter 70".

Die Filter werden als eigene Gruppe per UND um die Preset-Filter gelegt.
Die Engine bleibt unveraendert: sie wertet verschachtelte Gruppen ohnehin
aus.

Die Felder stammen aus :mod:`btlite.screening.technicals`. Ein Universum,
auf dem diese Filter laufen, muss deshalb durch
:func:`~btlite.screening.technicals.enrich_with_technicals` gegangen sein.

Architekturregel 2: dieses Modul holt keine Daten.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from btlite.core.errors import ScreeningError
from btlite.models.screening import (
    RSI_SCALE_MAX,
    RSI_SCALE_MIN,
    FilterGroup,
    FilterPredicate,
    LogicalOp,
    MaPosition,
    Operator,
    ScreenRequest,
    TechnicalRefinement,
)
from btlite.models.setups import SetupKey
from btlite.screening.filters import apply_filters
from btlite.screening.technicals import SETUP_COLUMNS

if TYPE_CHECKING:
    import pandas as pd

#: Vorzeichenbehafteter Abstand zur 200-Tage-Linie, dezimal. Negativ = darunter.
DIST_SMA_200: Final[str] = "dist_sma_200"

#: Vorzeichenbehafteter Abstand zur 50-Tage-Linie, dezimal. Negativ = darunter.
DIST_SMA_50: Final[str] = "dist_sma_50"

#: RSI ueber 14 Tage, Skala 0 bis 100.
RSI_14: Final[str] = "rsi_14"

#: Spalten, die die Zusatzfilter in der Trefferliste sichtbar machen.
REFINEMENT_COLUMNS: Final[tuple[str, ...]] = ("price", DIST_SMA_50, DIST_SMA_200, RSI_14)


def refinement_columns(refinement: TechnicalRefinement | None) -> tuple[str, ...]:
    """Spalten, die zu einer Einstellung in der Trefferliste gehoeren.

    Args:
        refinement: Zusatzfilter, oder None.

    Returns:
        :data:`REFINEMENT_COLUMNS`, dahinter die Spalten der gewaehlten
        Setups - wer nach einem Setup filtert, soll es in der Liste sehen.
    """
    chosen = refinement.setups if refinement is not None else []
    return (*REFINEMENT_COLUMNS, *(SETUP_COLUMNS[key] for key in chosen))


def _ma_predicate(field: str, position: MaPosition) -> FilterPredicate | None:
    """Uebersetzt die Lage zu einer Linie in ein Praedikat.

    Args:
        field: Abstandsspalte, dezimal und vorzeichenbehaftet.
        position: Gewuenschte Lage.

    Returns:
        ``field < 0`` fuer darunter, ``field > 0`` fuer darueber, None fuer
        egal. Genau auf der Linie (Abstand 0) gilt weder noch.
    """
    if position is MaPosition.BELOW:
        return FilterPredicate(field=field, op=Operator.LT, value=0.0)
    if position is MaPosition.ABOVE:
        return FilterPredicate(field=field, op=Operator.GT, value=0.0)
    return None


def refinement_predicates(refinement: TechnicalRefinement) -> list[FilterPredicate]:
    """Uebersetzt die Einstellung in Filter-Praedikate.

    Args:
        refinement: Gewaehlte Zusatzfilter.

    Returns:
        Praedikate in fester Reihenfolge: 200-Tage-Linie, 50-Tage-Linie,
        RSI-Unter- und -Obergrenze. Leer, wenn nichts gesetzt ist.
    """
    out: list[FilterPredicate] = []
    for field, position in ((DIST_SMA_200, refinement.sma_200), (DIST_SMA_50, refinement.sma_50)):
        predicate = _ma_predicate(field, position)
        if predicate is not None:
            out.append(predicate)
    if refinement.rsi_above is not None:
        out.append(FilterPredicate(field=RSI_14, op=Operator.GT, value=refinement.rsi_above))
    if refinement.rsi_below is not None:
        out.append(FilterPredicate(field=RSI_14, op=Operator.LT, value=refinement.rsi_below))
    # 1.0 heisst erfuellt. Fehlend (zu kurze Historie) erfuellt EQ nicht -
    # genau die gewuenschte Semantik: unbekannt ist nicht erfuellt.
    out.extend(
        FilterPredicate(field=SETUP_COLUMNS[key], op=Operator.EQ, value=1.0)
        for key in refinement.setups
    )
    return out


def refinement_group(refinement: TechnicalRefinement) -> FilterGroup | None:
    """Fasst die Zusatzfilter zu einer UND-Gruppe zusammen.

    Args:
        refinement: Gewaehlte Zusatzfilter.

    Returns:
        Die Gruppe, oder None wenn kein Filter gesetzt ist. Eine leere
        Gruppe waere ein Vertragsbruch von :class:`FilterGroup`.
    """
    predicates = refinement_predicates(refinement)
    if not predicates:
        return None
    return FilterGroup(op=LogicalOp.AND, predicates=predicates)


def refine_request(request: ScreenRequest, refinement: TechnicalRefinement | None) -> ScreenRequest:
    """Legt die Zusatzfilter per UND um die Filter einer Anfrage.

    Args:
        request: Anfrage, meist aus einem Preset.
        refinement: Zusatzfilter, oder None.

    Returns:
        Neue Anfrage. Die uebergebene bleibt unveraendert - Presets werden
        bei jedem Aufruf neu gebaut, aber eine Mutation hier waere trotzdem
        eine Falle fuer den naechsten Aufrufer.
    """
    extra = refinement_group(refinement) if refinement is not None else None
    if extra is None:
        return request
    own = request.local_filters
    combined = extra if own is None else FilterGroup(op=LogicalOp.AND, groups=[own, extra])
    return request.model_copy(update={"local_filters": combined})


def _rsi_bound(raw: object, label: str) -> float | None:
    """Liest eine RSI-Grenze aus einem Eingabefeld.

    Args:
        raw: Rohwert des Feldes. Leer, None oder Leerstring heisst "nicht
            gesetzt".
        label: Name der Grenze fuer die Fehlermeldung.

    Returns:
        Grenze auf der Skala 0 bis 100, oder None.

    Raises:
        ScreeningError: Wenn der Wert keine Zahl ist oder ausserhalb der
            Skala liegt.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        msg = f"{label} must be a number between 0 and 100."
        raise ScreeningError(msg, detail=f"eingegeben={raw!r}") from exc
    if not RSI_SCALE_MIN <= value <= RSI_SCALE_MAX:
        msg = f"{label} must be between 0 and 100, got {value:g}."
        raise ScreeningError(msg, detail=f"eingegeben={raw!r}")
    return value


def _ma_position(raw: object) -> MaPosition:
    """Liest die Lage zu einer Linie aus einem Auswahlfeld.

    Args:
        raw: Rohwert, ``"any"``, ``"above"`` oder ``"below"``. Leer heisst
            egal.

    Returns:
        Die Lage.

    Raises:
        ScreeningError: Bei einem unbekannten Wert.
    """
    if raw is None or raw == "":
        return MaPosition.ANY
    try:
        return MaPosition(str(raw))
    except ValueError as exc:
        msg = f"Unknown moving-average setting {raw!r}."
        raise ScreeningError(msg, detail=f"erlaubt={[m.value for m in MaPosition]}") from exc


def _setup_keys(raw: object) -> list[SetupKey]:
    """Liest die gewaehlten Setups aus einer Mehrfachauswahl.

    Args:
        raw: Liste von Schluesseln, ein einzelner Schluessel, oder leer.

    Returns:
        Schluessel ohne Dubletten, in Auswahlreihenfolge.

    Raises:
        ScreeningError: Bei einem unbekannten Schluessel.
    """
    if raw is None or raw == "":
        return []
    items = [raw] if isinstance(raw, str) else list(raw) if isinstance(raw, list | tuple) else [raw]
    keys: list[SetupKey] = []
    for item in items:
        try:
            key = SetupKey(str(item))
        except ValueError as exc:
            msg = f"Unknown chart setup {item!r}."
            raise ScreeningError(msg, detail=f"erlaubt={[k.value for k in SetupKey]}") from exc
        if key not in keys:
            keys.append(key)
    return keys


def build_refinement(
    sma_200: object = None,
    sma_50: object = None,
    rsi_above: object = None,
    rsi_below: object = None,
    setups: object = None,
) -> TechnicalRefinement:
    """Baut die Zusatzfilter aus den Rohwerten der Eingabefelder.

    Args:
        sma_200: Lage zur 200-Tage-Linie.
        sma_50: Lage zur 50-Tage-Linie.
        rsi_above: RSI muss darueber liegen, Skala 0 bis 100.
        rsi_below: RSI muss darunter liegen, Skala 0 bis 100.
        setups: Schluessel der Chart-Setups, die alle erfuellt sein muessen.

    Returns:
        Die validierte Einstellung.

    Raises:
        ScreeningError: Bei ungueltigen Werten oder einem leeren
            RSI-Bereich. Die Meldung ist fuer den Nutzer lesbar - die
            Validierungsausgabe von Pydantic waere es nicht.
    """
    above = _rsi_bound(rsi_above, "RSI above")
    below = _rsi_bound(rsi_below, "RSI below")
    if above is not None and below is not None and above >= below:
        msg = (
            f"RSI above {above:g} and below {below:g} leaves nothing: "
            "the lower bound has to be smaller than the upper one."
        )
        raise ScreeningError(msg)
    return TechnicalRefinement(
        sma_200=_ma_position(sma_200),
        sma_50=_ma_position(sma_50),
        rsi_above=above,
        rsi_below=below,
        setups=_setup_keys(setups),
    )


def tickers_passing(frame: pd.DataFrame, filters: FilterGroup | None) -> list[str]:
    """Listet die Titel, die eine Filtergruppe bestehen.

    Dient der Vorauswahl: die Zusatzfilter gelten per UND. Wer die
    Filter des Presets nicht besteht, faellt ohnehin heraus - fuer ihn
    muss keine Kursreihe geladen werden. Bei einem Value-Screen ueber den
    S&P 500 sind das wenige Dutzend statt ueber 500 Abrufe.

    Args:
        frame: Universum, Index = Ticker.
        filters: Filter des Presets, oder None fuer alle.

    Returns:
        Ticker in der Reihenfolge des Frames.

    Raises:
        UnknownFilterError: Wenn ein Filter eine fehlende Spalte nennt.
    """
    return [str(ticker) for ticker in apply_filters(frame, filters).index]
