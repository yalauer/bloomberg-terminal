"""Lokale Filter-DSL fuer Screening.

Wertet :class:`~btlite.models.screening.FilterGroup` und
:class:`~btlite.models.screening.FilterPredicate` gegen einen DataFrame aus.
Erwartete Form des Frames: **Index = Ticker**, **Spalten = Kennzahlen** oder
Stammdatenfelder. Die Kennzahlen-Keys stammen aus
``analytics/ratios.py`` (etwa ``pe``, ``roe``, ``debt_to_equity``), die
Stammdatenfelder aus dem Screener-Ergebnis (etwa ``Sector``).

Alle Funktionen dieses Moduls sind **rein**: kein Zustand, keine
Datenbeschaffung (Architekturregel 2), keine Mutation des Eingabeframes.

NaN-Semantik
------------
Verbindlich fuer das gesamte Modul, weil ein fehlender Wert keine Aussage
ist und deshalb keinen Vergleich erfuellen darf:

- **Vergleiche** (``GT``, ``GTE``, ``LT``, ``LTE``, ``EQ``, ``BETWEEN``,
  ``IN``): ein fehlender Wert erfuellt das Praedikat **nicht**. ``NaN`` ist
  nicht "kleiner als 15" - der Wert ist unbekannt.
- **Negierte Vergleiche** (``NEQ``, ``NOT_IN``): ein fehlender Wert erfuellt
  das Praedikat ebenfalls **nicht**. Das ist die entscheidende Stelle: ein
  naives ``series != value`` waere fuer ``NaN`` wahr und wuerde still all
  jene Zeilen einschleusen, ueber die man gerade nichts weiss. ``NEQ`` heisst
  hier "ist bekannt und ungleich", nicht "ist nicht gleich".
- **Nur** ``IS_NULL`` trifft fehlende Werte, ``NOT_NULL`` genau dessen
  Komplement.

Folge: ``NEQ`` ist **nicht** die logische Negation von ``EQ``. Beide sind
falsch, wo der Wert fehlt. Wer das Komplement inklusive Fehlwerten braucht,
setzt eine Gruppe mit ``negate=True`` um das ``EQ``-Praedikat - ``negate``
invertiert die fertige Maske und schliesst Fehlwerte damit ein.

Als fehlend gelten ``NaN``, ``None``, ``pd.NA`` und ``NaT``, ermittelt ueber
``pandas.isna``. Ein leerer String ist ein Wert, kein Fehlwert.
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Final

import pandas as pd

from btlite.core.errors import ScreeningError, UnknownFilterError
from btlite.models.screening import FilterGroup, FilterPredicate, LogicalOp, Operator

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Aehnlichkeitsschwelle fuer den Spaltenvorschlag bei unbekanntem Feld.
#: Bewusst niedrig gehalten, damit auch ein Tippfehler mit zwei falschen
#: Zeichen noch einen Treffer liefert.
_SUGGESTION_CUTOFF: Final[float] = 0.5

#: Maximale Anzahl vorgeschlagener Spaltennamen.
_SUGGESTION_COUNT: Final[int] = 3


def _suggest(field: str, columns: Sequence[str]) -> str:
    """Baut den technischen Hinweis fuer eine unbekannte Spalte.

    Args:
        field: Der angefragte, unbekannte Spaltenname.
        columns: Die tatsaechlich vorhandenen Spaltennamen.

    Returns:
        Detailtext mit den naechstliegenden Kandidaten, oder mit der
        vollstaendigen Spaltenliste wenn nichts hinreichend aehnlich ist.
    """
    hint = difflib.get_close_matches(
        field, list(columns), n=_SUGGESTION_COUNT, cutoff=_SUGGESTION_CUTOFF
    )
    if hint:
        return f"Meinten Sie: {hint}"
    return f"vorhandene Spalten: {sorted(columns)}"


def _column(frame: pd.DataFrame, field: str) -> pd.Series:
    """Holt eine Spalte und meldet ein unbekanntes Feld als Fehler.

    Eine unbekannte Spalte ist ein Konfigurationsfehler, kein leeres
    Ergebnis. Sie still auf "nichts getroffen" abzubilden wuerde einen
    Tippfehler in einem Preset unbemerkt lassen.

    Args:
        frame: Universum, Index = Ticker.
        field: Gesuchter Spaltenname.

    Returns:
        Die Spalte als Series.

    Raises:
        UnknownFilterError: Wenn die Spalte nicht existiert. Der Detailtext
            nennt die naechstliegenden vorhandenen Spaltennamen.
    """
    if field not in frame.columns:
        msg = f"Field {field!r} does not exist in the universe."
        raise UnknownFilterError(msg, detail=_suggest(field, [str(c) for c in frame.columns]))
    hits = int((frame.columns == field).sum())
    if hits > 1:
        # Doppelter Spaltenname im Frame - die Auswahl waere mehrdeutig.
        # frame[field] gaebe hier einen DataFrame statt einer Series zurueck.
        msg = f"Field {field!r} occurs more than once in the universe."
        raise ScreeningError(msg, detail=f"{hits} Spalten mit diesem Namen")
    return frame[field]


def _false_mask(frame: pd.DataFrame) -> pd.Series:
    """Baut eine durchgaengig falsche Maske passend zum Frame-Index.

    Args:
        frame: Universum, Index = Ticker.

    Returns:
        Boolesche Series, ueberall False.
    """
    return pd.Series(False, index=frame.index, dtype=bool)


def _present(column: pd.Series) -> pd.Series:
    """Maske der Zeilen mit vorhandenem Wert.

    Args:
        column: Zu pruefende Spalte.

    Returns:
        Boolesche Series, True wo ein Wert vorliegt.
    """
    return column.notna()


def _compare(column: pd.Series, op: Operator, value: object) -> pd.Series:
    """Wertet einen skalaren Vergleich aus, ohne Fehlwerte einzuschliessen.

    Die Vergleichsoperatoren von pandas liefern fuer ``NaN`` bereits False -
    ausser ``!=``, das True liefert. Diese Funktion behandelt deshalb alle
    Faelle einheitlich: das Ergebnis wird mit der Praesenzmaske verundet.

    Nicht vergleichbare Typkombinationen (etwa ``"Technology" > 5``) gelten
    als nicht erfuellt, statt einen TypeError nach oben zu geben. Ein
    gemischt getypter Frame ist bei gescrapten Quellen die Regel.

    Args:
        column: Zu pruefende Spalte.
        op: Vergleichsoperator.
        value: Vergleichswert.

    Returns:
        Boolesche Series ohne Fehlwerte.
    """
    present = _present(column)
    try:
        if op is Operator.GT:
            raw = column > value
        elif op is Operator.GTE:
            raw = column >= value
        elif op is Operator.LT:
            raw = column < value
        elif op is Operator.LTE:
            raw = column <= value
        elif op is Operator.EQ:
            raw = column == value
        else:  # Operator.NEQ
            raw = column != value
    except TypeError:
        return _false_mask_like(column)
    return raw.fillna(False).astype(bool) & present


def _false_mask_like(column: pd.Series) -> pd.Series:
    """Baut eine durchgaengig falsche Maske passend zu einer Spalte.

    Args:
        column: Spalte, deren Index uebernommen wird.

    Returns:
        Boolesche Series, ueberall False.
    """
    return pd.Series(False, index=column.index, dtype=bool)


def _between(column: pd.Series, low: object, high: object) -> pd.Series:
    """Wertet ein inklusives Intervall aus.

    Die Grenzen sind **beide inklusiv**: ``low <= x <= high``. Vertauschte
    Grenzen werden nicht stillschweigend getauscht, sondern gemeldet - eine
    vertauschte Angabe ist ein Fehler im Preset, kein Wunsch nach dem
    leeren Ergebnis.

    Args:
        column: Zu pruefende Spalte.
        low: Untere Grenze, inklusiv.
        high: Obere Grenze, inklusiv.

    Returns:
        Boolesche Series ohne Fehlwerte.

    Raises:
        ScreeningError: Wenn die untere Grenze ueber der oberen liegt.
    """
    try:
        inverted = low > high  # type: ignore[operator]
    except TypeError:
        inverted = False
    if inverted:
        msg = "BETWEEN has its bounds swapped."
        raise ScreeningError(msg, detail=f"value={low!r} liegt ueber value2={high!r}")
    return _compare(column, Operator.GTE, low) & _compare(column, Operator.LTE, high)


def _isin(column: pd.Series, values: Sequence[float | str]) -> pd.Series:
    """Wertet Mengenzugehoerigkeit aus, ohne Fehlwerte einzuschliessen.

    Args:
        column: Zu pruefende Spalte.
        values: Erlaubte Werte.

    Returns:
        Boolesche Series ohne Fehlwerte.
    """
    return column.isin(list(values)) & _present(column)


def evaluate_predicate(frame: pd.DataFrame, predicate: FilterPredicate) -> pd.Series:
    """Wertet ein einzelnes Praedikat gegen das Universum aus.

    Die NaN-Semantik des Moduls gilt: nur ``IS_NULL`` trifft fehlende Werte,
    jeder andere Operator - ``NEQ`` und ``NOT_IN`` eingeschlossen - liefert
    fuer eine Zeile ohne Wert False.

    Args:
        frame: Universum, Index = Ticker, Spalten = Kennzahlen. Einheiten
            wie in ``analytics/ratios.py``: Renditen und Margen dezimal
            (0.12 = 12 Prozent), Vielfache dimensionslos.
        predicate: Auszuwertendes Praedikat.

    Returns:
        Boolesche Maske mit dem Index des Frames, ohne Fehlwerte.

    Raises:
        UnknownFilterError: Wenn ``predicate.field`` keine Spalte ist.
        ScreeningError: Bei mehrfachem Spaltennamen oder vertauschten
            BETWEEN-Grenzen.
    """
    column = _column(frame, predicate.field)
    op = predicate.op

    if op is Operator.IS_NULL:
        return column.isna()
    if op is Operator.NOT_NULL:
        return _present(column)
    if op is Operator.BETWEEN:
        return _between(column, predicate.value, predicate.value2)
    if op is Operator.IN:
        # PD011: predicate.values ist das Pydantic-Feld, keine Series.
        return _isin(column, predicate.values or [])  # noqa: PD011
    if op is Operator.NOT_IN:
        # Bewusst kein reines Negat: fehlende Werte bleiben ausgeschlossen.
        return ~column.isin(list(predicate.values or [])) & _present(column)  # noqa: PD011
    return _compare(column, op, predicate.value)


def evaluate_group(frame: pd.DataFrame, group: FilterGroup) -> pd.Series:
    """Wertet eine - moeglicherweise verschachtelte - Filtergruppe aus.

    Praedikate und Untergruppen derselben Gruppe werden mit demselben
    ``group.op`` verknuepft. ``negate`` invertiert die fertige Maske der
    Gruppe **nach** der Verknuepfung. Weil ein negierter Vergleich innerhalb
    der Gruppe fehlende Werte ausschliesst, holt ``negate`` sie wieder
    herein - das ist der dokumentierte Weg, das echte Komplement inklusive
    Fehlwerten auszudruecken.

    Args:
        frame: Universum, Index = Ticker.
        group: Auszuwertende Gruppe.

    Returns:
        Boolesche Maske mit dem Index des Frames, ohne Fehlwerte.

    Raises:
        UnknownFilterError: Wenn ein Praedikat eine unbekannte Spalte nennt.
        ScreeningError: Bei mehrfachem Spaltennamen oder vertauschten
            BETWEEN-Grenzen.
    """
    masks: list[pd.Series] = [evaluate_predicate(frame, p) for p in group.predicates]
    masks.extend(evaluate_group(frame, g) for g in group.groups)

    # Der Vertrag verbietet leere Gruppen; diese Zeile ist die Absicherung
    # gegen einen kuenftig gelockerten Validator, nicht toter Code.
    if not masks:  # pragma: no cover - model_validator verhindert das
        return _false_mask(frame)

    combined = masks[0]
    if group.op is LogicalOp.AND:
        for mask in masks[1:]:
            combined = combined & mask
    else:
        for mask in masks[1:]:
            combined = combined | mask

    if group.negate:
        combined = ~combined
    return combined.astype(bool)


def build_mask(frame: pd.DataFrame, filters: FilterGroup | None) -> pd.Series:
    """Baut die Treffermaske fuer ein Universum.

    Args:
        frame: Universum, Index = Ticker.
        filters: Filtergruppe. ``None`` trifft alles.

    Returns:
        Boolesche Maske mit dem Index des Frames, ohne Fehlwerte.

    Raises:
        UnknownFilterError: Wenn ein Praedikat eine unbekannte Spalte nennt.
        ScreeningError: Bei mehrfachem Spaltennamen oder vertauschten
            BETWEEN-Grenzen.
    """
    if filters is None:
        return pd.Series(True, index=frame.index, dtype=bool)
    return evaluate_group(frame, filters)


def apply_filters(frame: pd.DataFrame, filters: FilterGroup | None) -> pd.DataFrame:
    """Filtert ein Universum und gibt die Treffer zurueck.

    Der Eingabeframe wird nicht veraendert; das Ergebnis ist eine Kopie mit
    erhaltener Zeilenreihenfolge.

    Args:
        frame: Universum, Index = Ticker, Spalten = Kennzahlen.
        filters: Filtergruppe. ``None`` gibt alle Zeilen zurueck.

    Returns:
        Gefilterter Frame in der Reihenfolge des Eingabeframes.

    Raises:
        UnknownFilterError: Wenn ein Praedikat eine unbekannte Spalte nennt.
        ScreeningError: Bei mehrfachem Spaltennamen oder vertauschten
            BETWEEN-Grenzen.
    """
    mask = build_mask(frame, filters)
    return frame.loc[mask].copy()


def describe_predicate(predicate: FilterPredicate) -> str:
    """Formuliert ein Praedikat als kurzen Klartext fuer das UI.

    Args:
        predicate: Zu beschreibendes Praedikat.

    Returns:
        Einzeiler, etwa ``"pe < 15"`` oder ``"roe between 0.1 and 0.4"``.
    """
    field, op = predicate.field, predicate.op
    symbols = {
        Operator.GT: ">",
        Operator.GTE: ">=",
        Operator.LT: "<",
        Operator.LTE: "<=",
        Operator.EQ: "==",
        Operator.NEQ: "!=",
    }
    if op in symbols:
        return f"{field} {symbols[op]} {predicate.value}"
    if op is Operator.BETWEEN:
        return f"{field} between {predicate.value} and {predicate.value2}"
    if op is Operator.IN:
        return f"{field} in {list(predicate.values or [])}"  # noqa: PD011
    if op is Operator.NOT_IN:
        return f"{field} not in {list(predicate.values or [])}"  # noqa: PD011
    if op is Operator.IS_NULL:
        return f"{field} missing"
    return f"{field} present"


def describe_group(group: FilterGroup) -> str:
    """Formuliert eine Filtergruppe als geklammerten Klartext fuer das UI.

    Args:
        group: Zu beschreibende Gruppe.

    Returns:
        Einzeiler, etwa ``"(pe < 15 AND roe > 0.15)"``. Eine negierte
        Gruppe wird mit ``NOT`` vorangestellt.
    """
    parts = [describe_predicate(p) for p in group.predicates]
    parts.extend(describe_group(g) for g in group.groups)
    joiner = " AND " if group.op is LogicalOp.AND else " OR "
    body = f"({joiner.join(parts)})"
    return f"NOT {body}" if group.negate else body
