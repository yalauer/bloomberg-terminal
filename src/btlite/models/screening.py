"""Vertraege fuer Screening.

Block 3 des Produkts. Zwei Wege fuehren zu einem Ergebnis:
    1. **Remote**: Finviz-Filter, ausgefuehrt bei der Quelle.
    2. **Lokal**: Praedikate ueber gecachte Fundamentals, ausgefuehrt hier.

Beide liefern dasselbe :class:`ScreenResult`, damit das UI nicht
unterscheiden muss.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from btlite.models.common import BTModel, DataFrameField, Provenance, Ticker
from btlite.models.setups import SetupKey


class Operator(StrEnum):
    """Vergleichsoperatoren der lokalen Filter-DSL."""

    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    EQ = "eq"
    NEQ = "neq"
    BETWEEN = "between"
    IN = "in"
    NOT_IN = "not_in"
    IS_NULL = "is_null"
    NOT_NULL = "not_null"


class LogicalOp(StrEnum):
    """Verknuepfung mehrerer Praedikate."""

    AND = "and"
    OR = "or"


class FilterPredicate(BTModel):
    """Ein einzelnes Filter-Praedikat der lokalen Engine.

    Beispiele::

        FilterPredicate(field="pe_ratio", op=Operator.LT, value=15)
        FilterPredicate(field="sector", op=Operator.IN, values=["Technology"])
        FilterPredicate(field="roe", op=Operator.BETWEEN, value=0.1, value2=0.4)
    """

    field: str = Field(description="Spaltenname im Universum-DataFrame.")
    op: Operator = Field(description="Vergleichsoperator.")
    value: float | str | bool | None = Field(default=None, description="Vergleichswert.")
    value2: float | str | None = Field(default=None, description="Obergrenze, nur bei ``BETWEEN``.")
    values: list[float | str] | None = Field(
        default=None, description="Wertemenge, nur bei ``IN`` und ``NOT_IN``."
    )

    @model_validator(mode="after")
    def _validate_operands(self) -> Self:
        """Prueft, dass die Operanden zum Operator passen.

        Returns:
            Das validierte Praedikat.

        Raises:
            ValueError: Bei fehlenden oder ueberzaehligen Operanden.
        """
        if self.op in {Operator.IN, Operator.NOT_IN}:
            if not self.values:
                msg = f"{self.op} benoetigt eine nicht-leere values-Liste"
                raise ValueError(msg)
        elif self.op is Operator.BETWEEN:
            if self.value is None or self.value2 is None:
                msg = "BETWEEN benoetigt value und value2"
                raise ValueError(msg)
        elif self.op in {Operator.IS_NULL, Operator.NOT_NULL}:
            if self.value is not None:
                msg = f"{self.op} nimmt keinen Vergleichswert"
                raise ValueError(msg)
        elif self.value is None:
            msg = f"{self.op} benoetigt einen Vergleichswert"
            raise ValueError(msg)
        return self


class FilterGroup(BTModel):
    """Rekursiv verschachtelbare Gruppe von Praedikaten.

    Erlaubt Ausdruecke wie ``(A und B) oder C``.
    """

    op: LogicalOp = Field(default=LogicalOp.AND, description="Verknuepfung der Kinder.")
    predicates: list[FilterPredicate] = Field(
        default_factory=list, description="Direkte Praedikate dieser Gruppe."
    )
    groups: list[FilterGroup] = Field(
        default_factory=list, description="Verschachtelte Untergruppen."
    )
    negate: bool = Field(default=False, description="Ergebnis der Gruppe invertieren.")

    @model_validator(mode="after")
    def _validate_nonempty(self) -> Self:
        """Verhindert leere Gruppen.

        Returns:
            Die validierte Gruppe.

        Raises:
            ValueError: Wenn die Gruppe weder Praedikate noch Untergruppen hat.
        """
        if not self.predicates and not self.groups:
            msg = "FilterGroup benoetigt mindestens ein Praedikat oder eine Untergruppe"
            raise ValueError(msg)
        return self


class MaPosition(StrEnum):
    """Lage des Kurses zu einem gleitenden Durchschnitt."""

    ANY = "any"
    ABOVE = "above"
    BELOW = "below"


#: Grenzen der RSI-Skala.
RSI_SCALE_MIN: float = 0.0
RSI_SCALE_MAX: float = 100.0


class TechnicalRefinement(BTModel):
    """Zusaetzliche Kursfilter, die zu jedem Preset hinzukommen.

    Die Presets beantworten feste Fragen. Diese Filter schraenken ein
    Preset weiter ein, etwa "unter der 200-Tage-Linie und RSI unter 70".
    Sie gelten per UND zusaetzlich zu den Filtern des Presets.

    Ein fehlender Wert erfuellt keinen dieser Filter: ein Titel ohne
    200-Tage-Linie, weil die Historie zu kurz ist, liegt weder darueber
    noch darunter.
    """

    sma_200: MaPosition = Field(default=MaPosition.ANY, description="Lage zur 200-Tage-Linie.")
    sma_50: MaPosition = Field(default=MaPosition.ANY, description="Lage zur 50-Tage-Linie.")
    rsi_above: float | None = Field(
        default=None,
        ge=RSI_SCALE_MIN,
        le=RSI_SCALE_MAX,
        description="RSI (14 Tage) muss darueber liegen, Skala 0 bis 100, strikt.",
    )
    rsi_below: float | None = Field(
        default=None,
        ge=RSI_SCALE_MIN,
        le=RSI_SCALE_MAX,
        description="RSI (14 Tage) muss darunter liegen, Skala 0 bis 100, strikt.",
    )
    setups: list[SetupKey] = Field(
        default_factory=list,
        description=(
            "Chart-Setups, die ALLE erfuellt sein muessen. Ein Setup mit zu kurzer "
            "Historie ist nicht erfuellt - unbekannt ist keine Aussage."
        ),
    )

    @model_validator(mode="after")
    def _validate_rsi_range(self) -> Self:
        """Verhindert einen leeren RSI-Bereich.

        Returns:
            Die validierte Einstellung.

        Raises:
            ValueError: Wenn die Untergrenze nicht unter der Obergrenze liegt.
                Ein solcher Filter faende nie einen Titel, und der Nutzer
                saehe eine leere Liste ohne Grund.
        """
        if (
            self.rsi_above is not None
            and self.rsi_below is not None
            and self.rsi_above >= self.rsi_below
        ):
            msg = (
                f"RSI range is empty: above {self.rsi_above:g} "
                f"and below {self.rsi_below:g} excludes every title"
            )
            raise ValueError(msg)
        return self

    @property
    def is_active(self) -> bool:
        """Ob mindestens ein Filter gesetzt ist.

        Returns:
            False, wenn alle Felder auf ihrem neutralen Wert stehen. Dann
            braucht ein fundamentales Preset keine Kursreihen.
        """
        return (
            self.sma_200 is not MaPosition.ANY
            or self.sma_50 is not MaPosition.ANY
            or self.rsi_above is not None
            or self.rsi_below is not None
            or bool(self.setups)
        )


class ScreenRequest(BTModel):
    """Anfrage an die Screening-Schicht.

    ``finviz_filters`` wird bei der Quelle ausgefuehrt und schraenkt das
    Universum ein. ``local_filters`` laeuft danach auf den gecachten
    Fundamentals. Beides ist kombinierbar.
    """

    name: str | None = Field(default=None, description="Name des Screens, etwa aus einem Preset.")
    finviz_filters: dict[str, str] = Field(
        default_factory=dict,
        description="Finviz-Filter als Key-Value-Paare in der Notation der Quelle.",
    )
    local_filters: FilterGroup | None = Field(
        default=None, description="Lokale Praedikate auf gecachten Fundamentals."
    )
    universe: list[Ticker] = Field(
        default_factory=list,
        description="Explizites Startuniversum. Leer = von Finviz beziehen.",
    )
    sort_by: str | None = Field(default=None, description="Spalte fuer die Sortierung.")
    ascending: bool = Field(default=False, description="Sortierrichtung.")
    limit: int = Field(default=200, ge=1, le=5000, description="Maximale Trefferzahl.")


class ScreenResult(BTModel):
    """Ergebnis eines Screening-Laufs.

    Der Payload ist ein DataFrame: Index sind Ticker, Spalten sind
    Kennzahlen und Stammdatenfelder.
    """

    request: ScreenRequest = Field(description="Die ausgefuehrte Anfrage.")
    data: DataFrameField = Field(description="Treffer-Matrix, Index = Ticker.")
    total_matches: int = Field(ge=0, description="Trefferzahl vor Anwendung von ``limit``.")
    provenance: Provenance = Field(description="Herkunft des Ergebnisses.")

    @property
    def tickers(self) -> list[str]:
        """Getroffene Symbole in Ergebnisreihenfolge.

        Returns:
            Liste der Ticker.
        """
        return [str(t) for t in self.data.index]

    @property
    def is_truncated(self) -> bool:
        """Ob das Ergebnis durch ``limit`` gekuerzt wurde.

        Returns:
            True wenn mehr Treffer existieren als zurueckgegeben wurden.
        """
        return self.total_matches > len(self.data)


class ScreenPreset(BTModel):
    """Benannter, wiederverwendbarer Screen.

    Die Presets Value, Quality, Momentum und Dividend liegen in
    ``screening/presets.py``.
    """

    key: str = Field(description="Stabiler Bezeichner, etwa value.")
    label: str = Field(description="Anzeigename.")
    description: str = Field(description="Was der Screen sucht, in einem Satz.")
    request: ScreenRequest = Field(description="Die hinterlegte Anfrage.")


class IndexSource(StrEnum):
    """Herkunft eines Indexbestands.

    Bestimmt, welcher Abrufpfad in ``data/indices.py`` greift. Beide Quellen
    sind oeffentlich und werden gedrosselt abgefragt.
    """

    FINVIZ = "finviz"
    WIKIPEDIA = "wikipedia"


class IndexDefinition(BTModel):
    """Beschreibung eines unterstuetzten Aktienindex.

    Ein Eintrag der Registry in ``data/indices.py``. Die Definition sagt
    **wo** der Bestand herkommt, nicht was er enthaelt - der Bestand selbst
    wird live geholt und ist :class:`IndexConstituents`.

    Weitere Indizes kommen allein durch Hinzufuegen einer Definition dazu;
    an den Abrufpfaden ist dafuer nichts zu aendern.
    """

    key: str = Field(
        min_length=1,
        max_length=32,
        description="Stabiler Bezeichner in Kleinbuchstaben, etwa ``sp500``.",
    )
    label: str = Field(min_length=1, description="Anzeigename fuer die Oberflaeche, etwa S&P 500.")
    source: IndexSource = Field(description="Quelle, aus der der Bestand geholt wird.")
    finviz_index_value: str | None = Field(
        default=None,
        description=(
            "Wert des Finviz-Filters ``Index``, exakt in der Notation der Quelle "
            "(etwa ``S&P 500``). Nur bei ``source=FINVIZ`` gesetzt."
        ),
    )
    wikipedia_url: str | None = Field(
        default=None,
        description="Vollstaendige URL der Wikipedia-Seite. Nur bei ``source=WIKIPEDIA``.",
    )
    wikipedia_ticker_column: str | None = Field(
        default=None,
        description=(
            "Spaltenname der Ticker-Spalte in der Wikipedia-Tabelle. Die Tabelle wird "
            "ueber diesen Spaltennamen ausgewaehlt, nicht ueber ``match`` - das "
            "brauchte ``html5lib``, das nicht installiert ist."
        ),
    )
    wikipedia_name_column: str | None = Field(
        default=None,
        description="Spaltenname der Firmennamen-Spalte, falls die Tabelle sie fuehrt.",
    )
    wikipedia_sector_column: str | None = Field(
        default=None, description="Spaltenname der Sektor-Spalte, falls die Tabelle sie fuehrt."
    )
    wikipedia_country_column: str | None = Field(
        default=None,
        description=(
            "Spaltenname der Laender-Spalte. Gesetzt bedeutet: die Ticker der Quelle "
            "tragen **keine** Boersen-Endung und muessen aufgeloest werden, bevor "
            "yfinance sie kennt. Das Land steuert dabei die Endung - ``France`` "
            "fuehrt auf ``.PA``. Siehe ``data/indices.py``."
        ),
    )

    @model_validator(mode="after")
    def _validate_source_fields(self) -> Self:
        """Prueft, dass die zur Quelle gehoerenden Felder gesetzt sind.

        Returns:
            Die validierte Definition.

        Raises:
            ValueError: Wenn die Angaben nicht zur gewaehlten Quelle passen.
        """
        if self.source is IndexSource.FINVIZ:
            if not self.finviz_index_value:
                msg = f"Index {self.key!r}: source=finviz verlangt finviz_index_value"
                raise ValueError(msg)
        elif not self.wikipedia_url or not self.wikipedia_ticker_column:
            msg = (
                f"Index {self.key!r}: source=wikipedia verlangt wikipedia_url "
                "und wikipedia_ticker_column"
            )
            raise ValueError(msg)
        return self


class IndexMember(BTModel):
    """Ein Mitglied eines Indexbestands.

    ``name`` und ``sector`` sind optional, weil nicht jede Quelle sie fuehrt.
    Fehlende Werte sind ``None``, nie ein leerer String.
    """

    ticker: Ticker = Field(
        description=(
            "Kanonisches Symbol. Boersen-Endungen wie ``.DE`` bleiben erhalten, "
            "yfinance braucht sie fuer Nicht-US-Listings."
        )
    )
    name: str | None = Field(default=None, description="Firmenname laut Quelle.")
    sector: str | None = Field(default=None, description="Sektor oder Branche laut Quelle.")


class SkippedMember(BTModel):
    """Ein Titel der Quelle, der **nicht** ins Universum aufgenommen wurde.

    Entsteht dort, wo die Quelle ein Symbol ohne Boersen-Endung fuehrt und
    sich daraus kein Symbol ableiten laesst, dem man trauen kann. Der
    STOXX Europe 600 listet etwa ``DG`` fuer Vinci; roh an yfinance gegeben
    liefert ``DG`` die Dollar General Corporation - einen US-Discounter.
    Ein solcher Titel faellt lieber heraus, als das Universum mit der
    falschen Firma zu fuellen.

    Die uebersprungenen Titel sind **nicht** still: sie haengen an
    :class:`IndexConstituents` und sind damit im UI auszaehlbar. Ein Nutzer
    muss erkennen koennen, dass er 428 statt 467 Titel screent.
    """

    source_ticker: str = Field(
        description="Symbol, wie die Quelle es fuehrt - roh, ohne Boersen-Endung."
    )
    name: str | None = Field(default=None, description="Firmenname laut Quelle.")
    country: str | None = Field(default=None, description="Sitzland laut Quelle.")
    reason: str = Field(
        description=("Warum der Titel herausfiel, in einem Satz und fuer den Nutzer lesbar.")
    )


class IndexConstituents(BTModel):
    """Der abgerufene Bestand eines Index.

    Der Payload ist bewusst eine Liste von Modellen und kein roher DataFrame
    der Quelle - Architekturregel 1.
    """

    definition: IndexDefinition = Field(description="Der abgefragte Index.")
    members: list[IndexMember] = Field(description="Bestand in der Reihenfolge der Quelle.")
    provenance: Provenance = Field(description="Herkunft des Bestands.")
    skipped: list[SkippedMember] = Field(
        default_factory=list,
        description=(
            "Titel der Quelle, deren Symbol sich nicht sicher aufloesen liess. "
            "Leer bei Indizes, deren Quelle fertige Symbole liefert."
        ),
    )

    @property
    def tickers(self) -> list[str]:
        """Die Symbole des Bestands.

        Returns:
            Kanonische Symbole in Quellenreihenfolge.
        """
        return [m.ticker for m in self.members]

    @property
    def is_complete(self) -> bool:
        """Ob der Bestand alle Titel der Quelle enthaelt.

        Returns:
            False, sobald mindestens ein Titel uebersprungen wurde. Das UI
            weist einen unvollstaendigen Bestand sichtbar aus, statt ihn
            als das vollstaendige Universum auszugeben.
        """
        return not self.skipped

    @property
    def source_count(self) -> int:
        """Anzahl der Titel, die die Quelle gefuehrt hat.

        Returns:
            Summe aus aufgenommenen und uebersprungenen Titeln.
        """
        return len(self.members) + len(self.skipped)

    def __len__(self) -> int:
        """Anzahl der Mitglieder.

        Returns:
            Groesse des Bestands.
        """
        return len(self.members)
