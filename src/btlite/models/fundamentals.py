"""Vertraege fuer Financial Statements und abgeleitete Kennzahlen.

Block 2 des Produkts. Zentrale Designentscheidung: die Provider liefern
uneinheitliche Zeilenbezeichner ("Total Revenue", "TotalRevenue",
"Revenues"). Die Normalisierung auf :class:`LineItem` passiert genau einmal,
in ``analytics/statements.py``. Alles danach arbeitet nur mit den
kanonischen Namen.

Vorzeichenkonvention (verbindlich, siehe docs/DATA_SOURCES.md):
    - Aufwaende in der GuV sind **positiv** gespeichert.
    - Cashflow-Abfluesse sind **negativ** gespeichert, so wie der Provider
      sie liefert (CapEx also negativ).
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import TYPE_CHECKING, Self, cast

from pydantic import Field, model_validator

if TYPE_CHECKING:
    import pandas as pd

from btlite.models.common import (
    STATEMENT_INDEX_NAME,
    BTModel,
    Currency,
    DataFrameField,
    Period,
    Provenance,
    Ticker,
    require_utc_index,
)


class StatementType(StrEnum):
    """Art des Financial Statement."""

    INCOME = "income"
    BALANCE = "balance"
    CASHFLOW = "cashflow"


class LineItem(StrEnum):
    """Kanonische Positionsbezeichner nach der Normalisierung.

    Das Mapping von Provider-Rohnamen auf diese Werte liegt in
    ``analytics/statements.py``. Nicht gemappte Rohzeilen bleiben mit ihrem
    Originalnamen erhalten, damit nichts verloren geht.
    """

    # --- Income Statement ---
    REVENUE = "revenue"
    COST_OF_REVENUE = "cost_of_revenue"
    GROSS_PROFIT = "gross_profit"
    RESEARCH_DEVELOPMENT = "research_development"
    SGA = "selling_general_administrative"
    OPERATING_EXPENSE = "operating_expense"
    OPERATING_INCOME = "operating_income"
    EBITDA = "ebitda"
    EBIT = "ebit"
    INTEREST_EXPENSE = "interest_expense"
    PRETAX_INCOME = "pretax_income"
    TAX_PROVISION = "tax_provision"
    NET_INCOME = "net_income"
    EPS_BASIC = "eps_basic"
    EPS_DILUTED = "eps_diluted"
    SHARES_BASIC = "shares_basic"
    SHARES_DILUTED = "shares_diluted"

    # --- Balance Sheet ---
    CASH_AND_EQUIVALENTS = "cash_and_equivalents"
    SHORT_TERM_INVESTMENTS = "short_term_investments"
    RECEIVABLES = "receivables"
    INVENTORY = "inventory"
    CURRENT_ASSETS = "current_assets"
    PPE_NET = "ppe_net"
    GOODWILL = "goodwill"
    INTANGIBLES = "intangibles"
    TOTAL_ASSETS = "total_assets"
    PAYABLES = "payables"
    CURRENT_DEBT = "current_debt"
    CURRENT_LIABILITIES = "current_liabilities"
    LONG_TERM_DEBT = "long_term_debt"
    TOTAL_DEBT = "total_debt"
    TOTAL_LIABILITIES = "total_liabilities"
    RETAINED_EARNINGS = "retained_earnings"
    TOTAL_EQUITY = "total_equity"
    MINORITY_INTEREST = "minority_interest"
    WORKING_CAPITAL = "working_capital"

    # --- Cashflow ---
    OPERATING_CASHFLOW = "operating_cashflow"
    CAPEX = "capital_expenditure"
    FREE_CASHFLOW = "free_cashflow"
    INVESTING_CASHFLOW = "investing_cashflow"
    FINANCING_CASHFLOW = "financing_cashflow"
    DIVIDENDS_PAID = "dividends_paid"
    SHARE_REPURCHASE = "share_repurchase"
    DEPRECIATION_AMORTIZATION = "depreciation_amortization"
    STOCK_COMPENSATION = "stock_compensation"
    CHANGE_IN_CASH = "change_in_cash"


class FinancialStatement(BTModel):
    """Ein normalisiertes Financial Statement.

    Der Payload ist ein DataFrame mit:
        - Index: normalisierte Positionsbezeichner, Name ``line_item``.
        - Spalten: Periodenende-Zeitstempel, tz-aware UTC, aufsteigend.
        - Werte: Betraege in ``currency``, absolut (nicht in Millionen).

    Diese Orientierung ist die Transponierte der yfinance-Rohform. Sie ist so
    gewaehlt, dass eine Position ueber die Zeit eine Zeile bleibt.
    """

    ticker: Ticker = Field(description="Kanonisches Symbol.")
    statement_type: StatementType = Field(description="Art des Statement.")
    period: Period = Field(description="Jahres- oder Quartalsbasis.")
    currency: Currency = Field(default=Currency.USD, description="Berichtswaehrung.")
    fiscal_year_end_month: int | None = Field(
        default=None,
        ge=1,
        le=12,
        description="Monat des Geschaeftsjahresendes. Relevant bei Fiskaljahr-Versatz.",
    )
    data: DataFrameField = Field(description="Statement-Payload, siehe Klassendoku.")
    unmapped_line_items: list[str] = Field(
        default_factory=list,
        description="Rohzeilen ohne kanonisches Mapping. Bleiben im Payload erhalten.",
    )
    provenance: Provenance = Field(description="Herkunft des Statement.")

    @model_validator(mode="after")
    def _validate_payload(self) -> Self:
        """Prueft Index-Benennung und Spaltenachse.

        Returns:
            Das validierte Modell.
        """
        what = f"FinancialStatement[{self.ticker}/{self.statement_type}]"
        self.data.index.name = STATEMENT_INDEX_NAME
        if not self.data.empty:
            # Spalten sind Perioden -> Pruefung ueber die transponierte Sicht.
            require_utc_index(self.data.T, what=f"{what}.columns")
        return self

    def get(self, item: LineItem | str) -> float | None:
        """Liest den Wert einer Position in der juengsten Periode.

        Args:
            item: Kanonischer Bezeichner oder Rohname.

        Returns:
            Betrag in ``currency``, oder None wenn die Position fehlt oder
            der Wert nicht belastbar ist. Fehlende Werte sind immer
            explizit None, niemals ein stiller Nullwert.
        """
        import pandas as pd

        key = item.value if isinstance(item, LineItem) else item
        if key not in self.data.index or self.data.empty:
            return None
        row = self.data.loc[key]
        if isinstance(row, pd.DataFrame):
            # Doppelter Positionsbezeichner im Payload -> nicht eindeutig.
            return None
        value = row.iloc[-1]
        if bool(pd.isna(value)):
            return None
        return float(value)

    @property
    def periods(self) -> list[date]:
        """Berichtsperioden der Spaltenachse, aufsteigend.

        Returns:
            Liste der Periodenende-Daten.
        """
        columns = cast("pd.DatetimeIndex", self.data.columns)
        return [ts.date() for ts in columns]


class StatementBundle(BTModel):
    """Alle drei Statements eines Tickers fuer eine Periodenbasis.

    Der Vertrag der Ratio-Engine: sie bekommt genau dieses Buendel und holt
    nie selbst Daten.
    """

    ticker: Ticker = Field(description="Kanonisches Symbol.")
    period: Period = Field(description="Jahres- oder Quartalsbasis.")
    income: FinancialStatement | None = Field(default=None, description="Gewinn- und Verlust.")
    balance: FinancialStatement | None = Field(default=None, description="Bilanz.")
    cashflow: FinancialStatement | None = Field(default=None, description="Kapitalflussrechnung.")
    currency: Currency = Field(default=Currency.USD, description="Berichtswaehrung.")
    provenance: Provenance = Field(description="Herkunft des Buendels.")


class RatioCategory(StrEnum):
    """Fachliche Gruppierung der Kennzahlen fuer die UI-Darstellung."""

    LIQUIDITY = "liquidity"
    LEVERAGE = "leverage"
    PROFITABILITY = "profitability"
    EFFICIENCY = "efficiency"
    VALUATION = "valuation"
    GROWTH = "growth"


class Ratio(BTModel):
    """Eine einzelne Kennzahl mit Einheit und Herkunft.

    ``value is None`` bedeutet: nicht berechenbar. Das UI rendert das als
    "n/a". Es gibt keine stillen Nullwerte oder NaN-Fallbacks.
    """

    key: str = Field(description="Stabiler Bezeichner, etwa current_ratio.")
    label: str = Field(description="Anzeigename, etwa Current Ratio.")
    category: RatioCategory = Field(description="Fachliche Gruppe.")
    value: float | None = Field(default=None, description="Wert in der Einheit von ``unit``.")
    unit: str = Field(
        default="ratio",
        description='Einheit: "ratio", "percent" (dezimal), "days", "currency", "x".',
    )
    period_end: date | None = Field(default=None, description="Periodenende der Berechnungsbasis.")
    inputs_missing: list[str] = Field(
        default_factory=list,
        description="Fehlende Eingangspositionen, wenn ``value`` None ist.",
    )
    formula: str | None = Field(default=None, description="Berechnungsformel als Klartext.")

    @property
    def is_available(self) -> bool:
        """Ob die Kennzahl berechnet werden konnte.

        Returns:
            True wenn ein Wert vorliegt.
        """
        return self.value is not None


class RatioSet(BTModel):
    """Alle Kennzahlen eines Tickers zu einem Stichtag."""

    ticker: Ticker = Field(description="Kanonisches Symbol.")
    period: Period = Field(description="Basis der Berechnung.")
    period_end: date | None = Field(default=None, description="Juengstes Periodenende.")
    currency: Currency = Field(
        default=Currency.USD, description="Waehrung waehrungsbehafteter Werte."
    )
    ratios: dict[str, Ratio] = Field(
        default_factory=dict, description="Kennzahlen, indiziert nach ``Ratio.key``."
    )
    provenance: Provenance = Field(description="Herkunft der Berechnungsbasis.")

    def by_category(self, category: RatioCategory) -> list[Ratio]:
        """Filtert die Kennzahlen einer Kategorie.

        Args:
            category: Gewuenschte fachliche Gruppe.

        Returns:
            Kennzahlen der Kategorie, stabil nach Key sortiert.
        """
        return sorted(
            (r for r in self.ratios.values() if r.category is category),
            key=lambda r: r.key,
        )

    def value_of(self, key: str) -> float | None:
        """Liest den Wert einer Kennzahl.

        Args:
            key: Bezeichner der Kennzahl.

        Returns:
            Wert, oder None wenn unbekannt oder nicht berechenbar.
        """
        ratio = self.ratios.get(key)
        return None if ratio is None else ratio.value


class PeerComparison(BTModel):
    """Kennzahlen-Vergleich eines Tickers gegen eine Peer-Gruppe.

    Der Payload ist ein DataFrame: Zeilen sind Ticker, Spalten sind
    Kennzahl-Keys, Werte sind die jeweiligen Kennzahlen. Fehlende Werte sind
    ``NaN`` und werden im UI als "n/a" gerendert.
    """

    subject: Ticker = Field(description="Betrachteter Ticker.")
    peers: list[Ticker] = Field(default_factory=list, description="Vergleichsgruppe.")
    period: Period = Field(description="Basis der Berechnung.")
    data: DataFrameField = Field(description="Matrix Ticker x Kennzahl.")
    provenance: Provenance = Field(description="Herkunft der Vergleichsbasis.")

    def percentile_of_subject(self, ratio_key: str) -> float | None:
        """Perzentil-Rang des Subjekts innerhalb der Peer-Gruppe.

        Args:
            ratio_key: Bezeichner der Kennzahl.

        Returns:
            Rang zwischen 0 und 1, dezimal. None wenn die Kennzahl fehlt
            oder das Subjekt keinen Wert hat.
        """
        import pandas as pd

        if ratio_key not in self.data.columns or self.subject not in self.data.index:
            return None
        col = self.data[ratio_key].dropna()
        if self.subject not in col.index or len(col) < 2:
            return None
        subject_value = col.loc[self.subject]
        if pd.isna(subject_value):
            return None
        return float((col < subject_value).sum() / (len(col) - 1))
