"""Normalisierung von Financial Statements auf die kanonischen Bezeichner.

Das ist die **einzige** Stelle im Projekt, die Provider-Rohnamen kennt. Alles
darueber arbeitet ausschliesslich mit :class:`~btlite.models.fundamentals.LineItem`.

Die Rohnamen von yfinance sind CamelCase ohne Trennzeichen
(``TotalRevenue``, ``NetPPE``, ``CashAndCashEquivalents``). Das Mapping unten
ist gegen die echten Fixtures verifiziert, nicht geraten. Nicht gemappte Zeilen
bleiben mit ihrem Originalnamen im Payload erhalten und werden in
:attr:`FinancialStatement.unmapped_line_items` gelistet - es geht nichts
still verloren.

Vorzeichenkonvention (in den Fixtures bestaetigt, siehe docs/DATA_SOURCES.md):
    - GuV-Aufwaende sind **positiv** gespeichert (``CostOfRevenue`` > 0).
    - Cashflow-Abfluesse sind **negativ** gespeichert, wie vom Provider
      geliefert: ``CapitalExpenditure`` und ``CashDividendsPaid`` < 0.
      Wer Free Cash Flow rechnet, **addiert** CapEx.
"""

from __future__ import annotations

from datetime import date
from typing import Final, cast

import pandas as pd
from pydantic import Field

from btlite.core.errors import AnalyticsError
from btlite.core.logging import get_logger
from btlite.models.common import (
    STATEMENT_INDEX_NAME,
    BTModel,
    Currency,
    DataFrameField,
    DataSource,
    Period,
    Provenance,
    Ticker,
)
from btlite.models.fundamentals import (
    FinancialStatement,
    LineItem,
    StatementBundle,
    StatementType,
)

_LOG = get_logger(__name__)

#: Anzahl Quartale, die eine TTM-Groesse zusammenfasst.
TTM_QUARTERS: Final[int] = 4

# --------------------------------------------------------------------------
# Mapping Rohname -> kanonischer Bezeichner
#
# Reihenfolge innerhalb einer Position ist Praeferenz: der erste in den Daten
# gefundene Rohname gewinnt. Deshalb stehen die spezifischeren Namen vorn.
# --------------------------------------------------------------------------

_INCOME_MAP: Final[dict[str, LineItem]] = {
    "TotalRevenue": LineItem.REVENUE,
    "OperatingRevenue": LineItem.REVENUE,
    "Revenues": LineItem.REVENUE,
    "CostOfRevenue": LineItem.COST_OF_REVENUE,
    "ReconciledCostOfRevenue": LineItem.COST_OF_REVENUE,
    "GrossProfit": LineItem.GROSS_PROFIT,
    "ResearchAndDevelopment": LineItem.RESEARCH_DEVELOPMENT,
    "SellingGeneralAndAdministration": LineItem.SGA,
    "OperatingExpense": LineItem.OPERATING_EXPENSE,
    "OperatingIncome": LineItem.OPERATING_INCOME,
    "TotalOperatingIncomeAsReported": LineItem.OPERATING_INCOME,
    "EBITDA": LineItem.EBITDA,
    "NormalizedEBITDA": LineItem.EBITDA,
    "EBIT": LineItem.EBIT,
    "InterestExpense": LineItem.INTEREST_EXPENSE,
    "InterestExpenseNonOperating": LineItem.INTEREST_EXPENSE,
    "PretaxIncome": LineItem.PRETAX_INCOME,
    "TaxProvision": LineItem.TAX_PROVISION,
    "NetIncome": LineItem.NET_INCOME,
    "NetIncomeCommonStockholders": LineItem.NET_INCOME,
    "NetIncomeIncludingNoncontrollingInterests": LineItem.NET_INCOME,
    "BasicEPS": LineItem.EPS_BASIC,
    "DilutedEPS": LineItem.EPS_DILUTED,
    "BasicAverageShares": LineItem.SHARES_BASIC,
    "DilutedAverageShares": LineItem.SHARES_DILUTED,
}

_BALANCE_MAP: Final[dict[str, LineItem]] = {
    "CashAndCashEquivalents": LineItem.CASH_AND_EQUIVALENTS,
    "CashCashEquivalentsAndShortTermInvestments": LineItem.CASH_AND_EQUIVALENTS,
    "OtherShortTermInvestments": LineItem.SHORT_TERM_INVESTMENTS,
    "Receivables": LineItem.RECEIVABLES,
    "AccountsReceivable": LineItem.RECEIVABLES,
    "Inventory": LineItem.INVENTORY,
    "CurrentAssets": LineItem.CURRENT_ASSETS,
    "NetPPE": LineItem.PPE_NET,
    "Goodwill": LineItem.GOODWILL,
    "GoodwillAndOtherIntangibleAssets": LineItem.GOODWILL,
    "OtherIntangibleAssets": LineItem.INTANGIBLES,
    "TotalAssets": LineItem.TOTAL_ASSETS,
    "Payables": LineItem.PAYABLES,
    "AccountsPayable": LineItem.PAYABLES,
    "PayablesAndAccruedExpenses": LineItem.PAYABLES,
    "CurrentDebt": LineItem.CURRENT_DEBT,
    "CurrentDebtAndCapitalLeaseObligation": LineItem.CURRENT_DEBT,
    "CurrentLiabilities": LineItem.CURRENT_LIABILITIES,
    "LongTermDebt": LineItem.LONG_TERM_DEBT,
    "LongTermDebtAndCapitalLeaseObligation": LineItem.LONG_TERM_DEBT,
    "TotalDebt": LineItem.TOTAL_DEBT,
    "TotalLiabilitiesNetMinorityInterest": LineItem.TOTAL_LIABILITIES,
    "RetainedEarnings": LineItem.RETAINED_EARNINGS,
    "StockholdersEquity": LineItem.TOTAL_EQUITY,
    "CommonStockEquity": LineItem.TOTAL_EQUITY,
    "TotalEquityGrossMinorityInterest": LineItem.TOTAL_EQUITY,
    "MinorityInterest": LineItem.MINORITY_INTEREST,
    "WorkingCapital": LineItem.WORKING_CAPITAL,
}

_CASHFLOW_MAP: Final[dict[str, LineItem]] = {
    "OperatingCashFlow": LineItem.OPERATING_CASHFLOW,
    "CashFlowFromContinuingOperatingActivities": LineItem.OPERATING_CASHFLOW,
    "CapitalExpenditure": LineItem.CAPEX,
    "PurchaseOfPPE": LineItem.CAPEX,
    "FreeCashFlow": LineItem.FREE_CASHFLOW,
    "InvestingCashFlow": LineItem.INVESTING_CASHFLOW,
    "CashFlowFromContinuingInvestingActivities": LineItem.INVESTING_CASHFLOW,
    "FinancingCashFlow": LineItem.FINANCING_CASHFLOW,
    "CashFlowFromContinuingFinancingActivities": LineItem.FINANCING_CASHFLOW,
    "CashDividendsPaid": LineItem.DIVIDENDS_PAID,
    "CommonStockDividendPaid": LineItem.DIVIDENDS_PAID,
    "RepurchaseOfCapitalStock": LineItem.SHARE_REPURCHASE,
    "CommonStockPayments": LineItem.SHARE_REPURCHASE,
    "DepreciationAmortizationDepletion": LineItem.DEPRECIATION_AMORTIZATION,
    "DepreciationAndAmortization": LineItem.DEPRECIATION_AMORTIZATION,
    "ReconciledDepreciation": LineItem.DEPRECIATION_AMORTIZATION,
    "StockBasedCompensation": LineItem.STOCK_COMPENSATION,
    "ChangesInCash": LineItem.CHANGE_IN_CASH,
}

#: Mapping je Statement-Typ.
_MAPS: Final[dict[StatementType, dict[str, LineItem]]] = {
    StatementType.INCOME: _INCOME_MAP,
    StatementType.BALANCE: _BALANCE_MAP,
    StatementType.CASHFLOW: _CASHFLOW_MAP,
}

#: Bestandsgroessen (Stichtagswerte). Sie werden fuer TTM **nicht** summiert,
#: sondern es gilt der juengste Wert. Alles andere sind Flussgroessen.
_STOCK_STATEMENTS: Final[frozenset[StatementType]] = frozenset({StatementType.BALANCE})

#: Positionen, die auch in Fluss-Statements Bestandscharakter haben und
#: deshalb bei der TTM-Bildung nicht summiert werden duerfen.
#:
#: Aktienzahlen sind Stichtags- beziehungsweise Durchschnittsgroessen einer
#: Periode: vier Quartale zu addieren wuerde die vierfache Aktienzahl ergeben.
#: **EPS gehoert hier NICHT hinein** - der Gewinn je Aktie ist eine
#: Flussgroesse und wird ueber die vier Quartale summiert, sonst stuende ein
#: Quartalsgewinn als Jahreswert im UI.
_STOCK_LINE_ITEMS: Final[frozenset[str]] = frozenset(
    {
        LineItem.SHARES_BASIC.value,
        LineItem.SHARES_DILUTED.value,
    }
)


def normalize_statement(statement: FinancialStatement) -> FinancialStatement:
    """Mappt die Rohzeilen eines Statement auf kanonische Bezeichner.

    Mehrere Rohnamen koennen auf dieselbe Position zeigen. Es gewinnt der in
    :data:`_INCOME_MAP` und Geschwistern zuerst gelistete Name, sofern er im
    Payload vorhanden ist und dort nicht ausschliesslich Fehlwerte traegt.

    Nicht gemappte Zeilen bleiben unter ihrem Originalnamen erhalten.

    Args:
        statement: Statement mit Provider-Rohnamen im Index.

    Returns:
        Neues Statement mit kanonischen Bezeichnern und gefuellter
        ``unmapped_line_items``-Liste. Das Original bleibt unveraendert.
    """
    mapping = _MAPS[statement.statement_type]
    raw = statement.data

    canonical: dict[str, pd.Series] = {}
    consumed: set[str] = set()

    # Praeferenzreihenfolge je Zielposition beibehalten.
    for raw_name, item in mapping.items():
        if raw_name not in raw.index:
            continue
        if item.value in canonical:
            continue
        series = raw.loc[raw_name]
        if isinstance(series, pd.DataFrame):
            # Doppelter Rohname - nicht eindeutig, deshalb uebergehen.
            _LOG.debug("Rohname %s ist im Payload mehrfach vorhanden", raw_name)
            continue
        if series.isna().all():
            # Position existiert, ist aber leer: kein besserer Kandidat als
            # ein spaeterer Alias mit echten Werten.
            continue
        canonical[item.value] = series
        consumed.add(raw_name)

    unmapped = [str(name) for name in raw.index if name not in consumed]
    passthrough = {str(name): raw.loc[name] for name in unmapped}

    combined = pd.DataFrame({**canonical, **passthrough}).T
    combined.index.name = STATEMENT_INDEX_NAME
    combined.columns = raw.columns

    return FinancialStatement(
        ticker=statement.ticker,
        statement_type=statement.statement_type,
        period=statement.period,
        currency=statement.currency,
        fiscal_year_end_month=statement.fiscal_year_end_month
        or infer_fiscal_year_end_month(statement),
        data=combined,
        unmapped_line_items=unmapped,
        provenance=statement.provenance,
    )


def infer_fiscal_year_end_month(statement: FinancialStatement) -> int | None:
    """Leitet den Monat des Geschaeftsjahresendes aus den Perioden ab.

    Nur fuer Jahresabschluesse belastbar. Apple etwa schliesst Ende September
    ab, nicht im Dezember.

    Args:
        statement: Statement mit Periodenende-Zeitstempeln als Spalten.

    Returns:
        Monat von 1 bis 12, oder None wenn keine Perioden vorliegen oder die
        Basis nicht jaehrlich ist.
    """
    if statement.period is not Period.ANNUAL or statement.data.empty:
        return None
    periods = statement.periods
    if not periods:
        return None
    # Der haeufigste Monat gewinnt, damit ein Rumpfgeschaeftsjahr nicht stoert.
    months = [p.month for p in periods]
    return max(set(months), key=months.count)


def normalize_bundle(bundle: StatementBundle) -> StatementBundle:
    """Normalisiert alle drei Statements eines Buendels.

    Args:
        bundle: Buendel mit Provider-Rohnamen.

    Returns:
        Neues Buendel mit kanonischen Bezeichnern.
    """
    return StatementBundle(
        ticker=bundle.ticker,
        period=bundle.period,
        income=normalize_statement(bundle.income) if bundle.income else None,
        balance=normalize_statement(bundle.balance) if bundle.balance else None,
        cashflow=normalize_statement(bundle.cashflow) if bundle.cashflow else None,
        currency=bundle.currency,
        provenance=bundle.provenance,
    )


def compute_ttm(statement: FinancialStatement) -> FinancialStatement:
    """Bildet Trailing-Twelve-Months aus einem Quartals-Statement.

    Die Unterscheidung ist wesentlich und keine Feinheit:

    - **Flussgroessen** (Umsatz, Gewinn, Cashflow) werden ueber die letzten
      vier Quartale **summiert**.
    - **Bestandsgroessen** (alle Bilanzpositionen, ausserdem EPS und
      Aktienzahlen) sind Stichtagswerte. Fuer sie gilt der **juengste** Wert;
      eine Summe waere fachlich falsch.

    Args:
        statement: Statement auf Quartalsbasis.

    Returns:
        Statement mit ``period=Period.TTM`` und genau einer Spalte, dem
        juengsten Periodenende.

    Raises:
        AnalyticsError: Wenn die Basis nicht quartalsweise ist.
        InsufficientDataError: Wenn weniger als vier Quartale vorliegen.
    """
    from btlite.core.errors import InsufficientDataError

    if statement.period is not Period.QUARTERLY:
        msg = "TTM requires a quarterly basis"
        raise AnalyticsError(msg, detail=f"period={statement.period}")

    n_periods = len(statement.data.columns)
    if n_periods < TTM_QUARTERS:
        msg = f"TTM requires {TTM_QUARTERS} quarters, {n_periods} available"
        raise InsufficientDataError(msg, required=TTM_QUARTERS, available=n_periods)

    window = statement.data.iloc[:, -TTM_QUARTERS:]
    is_stock_statement = statement.statement_type in _STOCK_STATEMENTS

    values: dict[str, float | None] = {}
    for name in window.index:
        row = window.loc[name]
        if isinstance(row, pd.DataFrame):
            continue
        treat_as_stock = is_stock_statement or str(name) in _STOCK_LINE_ITEMS
        if treat_as_stock:
            latest = row.iloc[-1]
            values[str(name)] = None if pd.isna(latest) else float(latest)
        else:
            # Eine Summe mit Luecken waere irrefuehrend: sie sieht wie ein
            # vollstaendiges Jahr aus, deckt aber weniger ab.
            values[str(name)] = None if row.isna().any() else float(row.sum())

    latest_period = window.columns[-1]
    frame = pd.DataFrame({latest_period: pd.Series(values, dtype="float64")})
    frame.index.name = STATEMENT_INDEX_NAME

    return FinancialStatement(
        ticker=statement.ticker,
        statement_type=statement.statement_type,
        period=Period.TTM,
        currency=statement.currency,
        fiscal_year_end_month=statement.fiscal_year_end_month,
        data=frame,
        unmapped_line_items=list(statement.unmapped_line_items),
        provenance=_ttm_provenance(statement),
    )


def _ttm_provenance(statement: FinancialStatement) -> Provenance:
    """Baut den Herkunftsnachweis eines abgeleiteten TTM-Statement.

    Args:
        statement: Quell-Statement auf Quartalsbasis.

    Returns:
        Provenance mit ``source=COMPUTED``, damit im UI sichtbar bleibt, dass
        der Wert gerechnet und nicht berichtet ist.
    """
    src = statement.provenance
    return Provenance(
        source=DataSource.COMPUTED,
        fetched_at=src.fetched_at,
        from_cache=src.from_cache,
        delay_minutes=src.delay_minutes,
        note=f"TTM from {TTM_QUARTERS} quarters (basis: {src.source.value})",
    )


def compute_ttm_bundle(bundle: StatementBundle) -> StatementBundle:
    """Bildet TTM fuer ein ganzes Buendel.

    Die Bilanz behaelt Stichtagscharakter: ihre Werte werden nicht summiert,
    sondern auf den juengsten Stichtag reduziert.

    Args:
        bundle: Buendel auf Quartalsbasis.

    Returns:
        Buendel mit ``period=Period.TTM``.

    Raises:
        AnalyticsError: Wenn die Basis nicht quartalsweise ist.
    """
    return StatementBundle(
        ticker=bundle.ticker,
        period=Period.TTM,
        income=compute_ttm(bundle.income) if bundle.income else None,
        balance=compute_ttm(bundle.balance) if bundle.balance else None,
        cashflow=compute_ttm(bundle.cashflow) if bundle.cashflow else None,
        currency=bundle.currency,
        provenance=bundle.provenance,
    )


def get_value(
    bundle: StatementBundle,
    item: LineItem,
    *,
    period_index: int = -1,
) -> float | None:
    """Liest eine Position aus dem passenden Statement des Buendels.

    Sucht die Position in allen drei Statements, damit Aufrufer nicht wissen
    muessen, wo eine Groesse gefuehrt wird.

    Args:
        bundle: Normalisiertes Buendel.
        item: Kanonischer Bezeichner.
        period_index: Spaltenindex, ``-1`` ist die juengste Periode.

    Returns:
        Betrag in der Berichtswaehrung, oder None wenn die Position fehlt
        oder keinen belastbaren Wert traegt. Nie ein stiller Nullwert.
    """
    for statement in (bundle.income, bundle.balance, bundle.cashflow):
        if statement is None or statement.data.empty:
            continue
        if item.value not in statement.data.index:
            continue
        row = statement.data.loc[item.value]
        if isinstance(row, pd.DataFrame):
            continue
        try:
            value = row.iloc[period_index]
        except IndexError:
            continue
        if pd.isna(value):
            continue
        return float(value)
    return None


def statement_currency(bundle: StatementBundle) -> Currency:
    """Liefert die Berichtswaehrung eines Buendels.

    Args:
        bundle: Normalisiertes Buendel.

    Returns:
        Die Waehrung des Buendels.
    """
    return bundle.currency


# --------------------------------------------------------------------------
# Trendrechnung: Veraenderung je Position und Periode
#
# Die Ergebnistypen stehen bewusst hier und nicht in models/fundamentals.py:
# sie sind das Rueckgabeformat genau dieser Funktionen und werden von keinem
# Provider gefuellt. Der Vertrag aus models/ bleibt damit auf das beschraenkt,
# was zwischen Schichten wandert.
# --------------------------------------------------------------------------

#: Begruendung, wenn Basis und Folgewert das Vorzeichen wechseln.
SIGN_CHANGE_REASON: Final[str] = "sign change"

#: Begruendung, wenn die Basisperiode null ist.
ZERO_BASE_REASON: Final[str] = "base value is 0"

#: Begruendung, wenn es keine Vorperiode gibt.
NO_PRIOR_REASON: Final[str] = "no prior period"

#: Begruendung, wenn einer der beiden Werte fehlt.
MISSING_VALUE_REASON: Final[str] = "value missing"


class TrendPoint(BTModel):
    """Veraenderung einer Position von einer Periode zur naechsten.

    ``change_pct is None`` bedeutet: als Prozentsatz nicht interpretierbar.
    :attr:`reason` nennt dann warum. Das UI rendert das als "n/a", genau wie
    bei :class:`~btlite.models.fundamentals.Ratio`.

    Die absolute Veraenderung ist auch dann definiert, wenn der Prozentsatz
    es nicht ist - etwa bei einem Vorzeichenwechsel.

    Einheiten:
        - ``current``, ``previous``, ``change_abs``: Betraege in der
          Berichtswaehrung des Statement.
        - ``change_pct``: dezimal, 0.12 bedeutet 12 Prozent.
    """

    line_item: str = Field(description="Kanonischer Bezeichner oder Rohname der Position.")
    period_end: date = Field(description="Periodenende der betrachteten Periode.")
    previous_period_end: date | None = Field(
        default=None, description="Periodenende der Vergleichsperiode."
    )
    change_abs: float | None = Field(
        default=None, description="Absolute Veraenderung, current minus previous."
    )
    change_pct: float | None = Field(
        default=None,
        description="Relative Veraenderung, dezimal. None wenn nicht interpretierbar.",
    )
    reason: str | None = Field(
        default=None, description="Begruendung, wenn ``change_pct`` None ist."
    )

    @property
    def is_available(self) -> bool:
        """Ob eine belastbare Prozentangabe vorliegt.

        Returns:
            True wenn ``change_pct`` gefuellt ist.
        """
        return self.change_pct is not None


class StatementTrend(BTModel):
    """Trendrechnung ueber alle Positionen eines Statement.

    Der Payload spiegelt die Form des Quell-Statement: Index sind die
    Positionsbezeichner, Spalten die Periodenenden. Es gibt drei Tabellen:
    absolute Veraenderung, relative Veraenderung und Begruendung. Die
    aelteste Periode hat keine Vorperiode und traegt deshalb durchgaengig
    Fehlwerte.

    Einheiten: ``change_pct`` dezimal, ``change_abs`` in ``currency``.
    """

    ticker: Ticker = Field(description="Kanonisches Symbol.")
    statement_type: StatementType = Field(description="Art des Statement.")
    period: Period = Field(description="Jahres- oder Quartalsbasis.")
    currency: Currency = Field(default=Currency.USD, description="Berichtswaehrung.")
    change_abs: DataFrameField = Field(description="Absolute Veraenderung je Position/Periode.")
    change_pct: DataFrameField = Field(
        description="Relative Veraenderung, dezimal. NaN wo nicht interpretierbar."
    )
    reasons: DataFrameField = Field(
        description="Begruendung der Fehlwerte, leerer String wo ein Wert vorliegt."
    )
    provenance: Provenance = Field(description="Herkunft der Berechnungsbasis.")

    def point(self, item: LineItem | str, *, period_index: int = -1) -> TrendPoint | None:
        """Liest einen einzelnen Trendpunkt.

        Args:
            item: Kanonischer Bezeichner oder Rohname.
            period_index: Spaltenindex, ``-1`` ist die juengste Periode.

        Returns:
            Der Trendpunkt, oder None wenn die Position oder die Periode
            nicht existiert.
        """
        key = item.value if isinstance(item, LineItem) else item
        if self.change_abs.empty or key not in self.change_abs.index:
            return None
        columns = cast("pd.DatetimeIndex", self.change_abs.columns)
        n_periods = len(columns)
        position = period_index + n_periods if period_index < 0 else period_index
        if not 0 <= position < n_periods:
            return None

        abs_row = self.change_abs.loc[key]
        pct_row = self.change_pct.loc[key]
        reason_row = self.reasons.loc[key]
        if isinstance(abs_row, pd.DataFrame):
            # Doppelter Positionsbezeichner - nicht eindeutig zuordenbar.
            return None

        abs_value = abs_row.iloc[position]
        pct_value = pct_row.iloc[position]
        reason = str(reason_row.iloc[position]) or None

        return TrendPoint(
            line_item=key,
            period_end=columns[position].date(),
            previous_period_end=columns[position - 1].date() if position > 0 else None,
            change_abs=None if pd.isna(abs_value) else float(abs_value),
            change_pct=None if pd.isna(pct_value) else float(pct_value),
            reason=reason,
        )


def _pct_change(current: float | None, previous: float | None) -> tuple[float | None, str | None]:
    """Berechnet eine relative Veraenderung oder begruendet ihr Fehlen.

    Ein Vorzeichenwechsel ist bewusst **nicht** als Prozentsatz darstellbar.
    Von -100 auf +50 sind keine "150 Prozent Wachstum": der Quotient misst
    eine Groesse relativ zu einer Basis, deren Richtung sich umgekehrt hat.
    Die uebliche Formel liefert dort eine Zahl, die das Gegenteil dessen
    suggeriert, was passiert ist - ein Verlust, der zum Gewinn wird, erschiene
    je nach Formel als negatives Wachstum. Deshalb None mit Begruendung.

    Dasselbe bei einer Basis von 0: der Sprung von 0 auf irgendetwas ist
    unendlich, nicht ein grosser Prozentsatz.

    Der Nenner ist der **Betrag** der Basis. Damit bedeutet ein positiver
    Wert auch bei negativer Basis eine Verbesserung und nicht das Gegenteil.

    Args:
        current: Wert der betrachteten Periode.
        previous: Wert der Vorperiode.

    Returns:
        Tupel aus relativer Veraenderung (dezimal) und Begruendung. Genau
        eines der beiden ist gefuellt.
    """
    if current is None or previous is None:
        return None, MISSING_VALUE_REASON
    if previous == 0:
        return None, ZERO_BASE_REASON
    if current != 0 and (current < 0) != (previous < 0):
        return None, SIGN_CHANGE_REASON
    return (current - previous) / abs(previous), None


def compute_trend(statement: FinancialStatement) -> StatementTrend:
    """Berechnet die Veraenderung je Position und Periode.

    Zu jedem Wert eines Statement wird die Veraenderung gegenueber der
    unmittelbaren Vorperiode gebildet. Die Spalten des Quell-Statement sind
    aufsteigend sortiert, die Vorperiode ist also die Spalte links daneben.

    Die absolute Veraenderung ist immer definiert, sobald beide Werte
    vorliegen. Die relative nicht: bei Vorzeichenwechsel und bei Basis 0
    bleibt sie ``NaN``, die Begruendung steht in
    :attr:`StatementTrend.reasons`.

    Args:
        statement: Normalisiertes Statement. Die Normalisierung ist keine
            Voraussetzung der Rechnung, macht die Ergebniszeilen aber
            lesbar.

    Returns:
        Trendtabellen in der Form des Quell-Statement. ``change_pct`` ist
        dezimal (0.12 = 12 Prozent), ``change_abs`` in der Berichtswaehrung.
    """
    data = statement.data
    n_periods = len(data.columns)
    nan = float("nan")

    abs_rows: list[list[float]] = []
    pct_rows: list[list[float]] = []
    reason_rows: list[list[str]] = []

    for name in data.index:
        row = data.loc[name]
        if isinstance(row, pd.DataFrame):
            # Doppelter Positionsbezeichner - nicht eindeutig zuordenbar.
            abs_rows.append([nan] * n_periods)
            pct_rows.append([nan] * n_periods)
            reason_rows.append([MISSING_VALUE_REASON] * n_periods)
            continue

        abs_row: list[float] = []
        pct_row: list[float] = []
        reason_row: list[str] = []
        previous: float | None = None

        for position in range(n_periods):
            raw = row.iloc[position]
            current = None if pd.isna(raw) else float(raw)
            if position == 0:
                # Die aelteste Periode hat keine Basis, an der sie zu messen
                # waere. Das ist kein Fehler, nur eine fehlende Vorperiode.
                abs_row.append(nan)
                pct_row.append(nan)
                reason_row.append(NO_PRIOR_REASON)
            else:
                # Die absolute Veraenderung ist immer definiert, sobald beide
                # Werte vorliegen - auch ueber einen Vorzeichenwechsel hinweg.
                if current is None or previous is None:
                    abs_row.append(nan)
                else:
                    abs_row.append(current - previous)
                pct, reason = _pct_change(current, previous)
                pct_row.append(nan if pct is None else pct)
                reason_row.append("" if reason is None else reason)
            previous = current

        abs_rows.append(abs_row)
        pct_rows.append(pct_row)
        reason_rows.append(reason_row)

    index = data.index.copy()
    index.name = STATEMENT_INDEX_NAME
    columns = data.columns.copy()

    def _frame(rows: list[list[float]] | list[list[str]], dtype: str) -> pd.DataFrame:
        """Baut eine Ergebnistabelle, auch wenn es keine Zeile gibt.

        Args:
            rows: Zeilenweise Werte.
            dtype: Zieltyp der Spalten.

        Returns:
            DataFrame in der Form des Quell-Statement.
        """
        if not rows:
            return pd.DataFrame(index=index, columns=columns, dtype=dtype)
        return pd.DataFrame(rows, index=index, columns=columns, dtype=dtype)

    abs_frame = _frame(abs_rows, "float64")
    pct_frame = _frame(pct_rows, "float64")
    reason_frame = _frame(reason_rows, "object")

    return StatementTrend(
        ticker=statement.ticker,
        statement_type=statement.statement_type,
        period=statement.period,
        currency=statement.currency,
        change_abs=abs_frame,
        change_pct=pct_frame,
        reasons=reason_frame,
        provenance=_trend_provenance(statement),
    )


def _trend_provenance(statement: FinancialStatement) -> Provenance:
    """Baut den Herkunftsnachweis einer abgeleiteten Trendtabelle.

    Args:
        statement: Quell-Statement.

    Returns:
        Provenance mit ``source=COMPUTED``, damit im UI sichtbar bleibt, dass
        die Veraenderung gerechnet und nicht berichtet ist.
    """
    src = statement.provenance
    return Provenance(
        source=DataSource.COMPUTED,
        fetched_at=src.fetched_at,
        from_cache=src.from_cache,
        delay_minutes=src.delay_minutes,
        note=f"Period-over-period change (basis: {src.source.value})",
    )


def _is_normalized(bundle: StatementBundle) -> bool:
    """Prueft, ob ein Buendel bereits durch die Normalisierung lief.

    :func:`normalize_statement` fuellt ``unmapped_line_items`` mit den
    Rohnamen, die es keiner kanonischen Position zuordnen konnte. Ein rohes
    Statement hat diese Liste leer, weil sie erst dort entsteht. Zusaetzlich
    genuegt ein kanonischer Bezeichner im Index als Beleg - ein Provider
    liefert solche Namen nie.

    Args:
        bundle: Zu pruefendes Buendel.

    Returns:
        True, wenn mindestens ein Statement Spuren der Normalisierung traegt.
    """
    canonical = {item.value for item in LineItem}
    for statement in (bundle.income, bundle.balance, bundle.cashflow):
        if statement is None:
            continue
        if statement.unmapped_line_items:
            return True
        if any(str(name) in canonical for name in statement.data.index):
            return True
    return False


def compute_trend_bundle(bundle: StatementBundle) -> dict[StatementType, StatementTrend]:
    """Berechnet die Trendtabellen aller Statements eines Buendels.

    Ein Buendel mit Provider-Rohnamen wird vorher normalisiert. Ohne das
    traegt das Ergebnis Zeilennamen wie ``TotalRevenue``, waehrend der Rest
    des Projekts ``revenue`` erwartet - die Trendwerte waeren rechnerisch
    richtig, aber fuer die Oberflaeche nicht auffindbar. Bereits
    normalisierte Buendel werden nicht erneut normalisiert: der zweite Lauf
    wuerde die nicht zugeordneten Rohzeilen verwerfen.

    Args:
        bundle: Buendel, roh oder normalisiert.

    Returns:
        Abbildung Statement-Art auf Trendtabelle, mit kanonischen
        Bezeichnern im Index. Fehlende Statements erscheinen nicht als
        Schluessel - ein leerer Eintrag waere nicht von einem echten
        Nullwert zu unterscheiden.
    """
    source = bundle if _is_normalized(bundle) else normalize_bundle(bundle)
    out: dict[StatementType, StatementTrend] = {}
    for statement in (source.income, source.balance, source.cashflow):
        if statement is None:
            continue
        out[statement.statement_type] = compute_trend(statement)
    return out


def compute_cagr(
    statement: FinancialStatement,
    item: LineItem | str,
    *,
    periods_per_year: float | None = None,
) -> tuple[float | None, str | None]:
    """Berechnet die mehrperiodige Wachstumsrate (CAGR) einer Position.

    Die CAGR setzt die juengste Periode ins Verhaeltnis zur aeltesten
    verfuegbaren und annualisiert ueber die dazwischenliegenden Perioden.

    Sie ist nur definiert, wenn beide Werte **positiv** sind. Eine negative
    Basis macht die Wurzel im Reellen undefiniert, ein Vorzeichenwechsel
    waere aus demselben Grund wie in :func:`_pct_change` nicht
    interpretierbar.

    Args:
        statement: Statement mit mindestens zwei belegten Perioden.
        item: Kanonischer Bezeichner oder Rohname der Position.
        periods_per_year: Perioden pro Jahr. None leitet aus
            ``statement.period`` ab: 4 bei Quartalsbasis, sonst 1.

    Returns:
        Tupel aus annualisierter Wachstumsrate (dezimal, 0.08 = 8 Prozent)
        und Begruendung. Genau eines der beiden ist gefuellt.
    """
    key = item.value if isinstance(item, LineItem) else item
    data = statement.data
    if data.empty or key not in data.index:
        return None, MISSING_VALUE_REASON

    row = data.loc[key]
    if isinstance(row, pd.DataFrame):
        return None, MISSING_VALUE_REASON

    series = row.dropna()
    if len(series) < 2:
        return None, NO_PRIOR_REASON

    start = float(series.iloc[0])
    end = float(series.iloc[-1])
    if start == 0:
        return None, ZERO_BASE_REASON
    if start < 0 or end < 0:
        return None, SIGN_CHANGE_REASON

    per_year = periods_per_year if periods_per_year is not None else _periods_per_year(statement)
    n_years = (len(series) - 1) / per_year
    if n_years <= 0:
        return None, NO_PRIOR_REASON

    return (end / start) ** (1.0 / n_years) - 1.0, None


def _periods_per_year(statement: FinancialStatement) -> float:
    """Leitet die Anzahl Berichtsperioden pro Jahr ab.

    Args:
        statement: Statement mit gesetzter Periodenbasis.

    Returns:
        4.0 bei Quartalsbasis, sonst 1.0.
    """
    return float(TTM_QUARTERS) if statement.period is Period.QUARTERLY else 1.0
