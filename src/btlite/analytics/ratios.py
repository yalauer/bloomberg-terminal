"""Kennzahlen-Engine.

Berechnet aus einem normalisierten :class:`StatementBundle` einen
:class:`RatioSet` mit sechs Kategorien.

**Harte Regel des Projekts:** Ein nicht berechenbarer Wert ist ``None``, nie
``0.0`` und nie stilles ``NaN``. Jede nicht berechenbare Kennzahl nennt in
:attr:`Ratio.inputs_missing`, welche Eingangsposition gefehlt hat. Das UI
rendert das als "n/a". Eine still zu Null gewordene Kennzahl wuerde jeden
Peer-Vergleich verfaelschen, der sie mittelt.

Einheiten:
    - ``"percent"``: dezimal, 0.31 bedeutet 31 Prozent.
    - ``"ratio"``: dimensionsloses Verhaeltnis.
    - ``"x"``: Vielfaches, etwa KGV oder Zinsdeckung.
    - ``"days"``: Tage.

Bewertungskennzahlen brauchen Preis und Marktkapitalisierung. Diese werden
**injiziert** - dieses Modul holt keine Daten (Architekturregel 2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from btlite.analytics.statements import get_value
from btlite.core.logging import get_logger
from btlite.models.common import DataSource, Provenance
from btlite.models.fundamentals import (
    LineItem,
    Ratio,
    RatioCategory,
    RatioSet,
)

if TYPE_CHECKING:
    from btlite.models.fundamentals import StatementBundle

_LOG = get_logger(__name__)

#: Kalendertage im Jahr fuer die Umschlagskennzahlen (DSO, DIO, DPO, CCC).
DAYS_PER_YEAR: Final[int] = 365


class _Builder:
    """Sammelt Kennzahlen und haelt die Fehlwert-Begruendungen fest.

    Interner Helfer. Er kapselt die immer gleiche Abfolge: Eingangswerte
    lesen, auf Vollstaendigkeit und Nullteiler pruefen, Ergebnis oder
    Begruendung ablegen.
    """

    def __init__(self, bundle: StatementBundle) -> None:
        """Initialisiert den Builder.

        Args:
            bundle: Normalisiertes Statement-Buendel.
        """
        self._bundle = bundle
        self.ratios: dict[str, Ratio] = {}

    def value(self, item: LineItem, *, period_index: int = -1) -> float | None:
        """Liest eine Position aus dem Buendel.

        Args:
            item: Kanonischer Bezeichner.
            period_index: Spaltenindex, ``-1`` ist die juengste Periode.

        Returns:
            Betrag, oder None wenn nicht vorhanden.
        """
        return get_value(self._bundle, item, period_index=period_index)

    def add(
        self,
        key: str,
        label: str,
        category: RatioCategory,
        *,
        numerator: tuple[LineItem, float | None],
        denominator: tuple[LineItem, float | None],
        unit: str = "ratio",
        formula: str,
        scale: float = 1.0,
    ) -> None:
        """Legt eine Quotienten-Kennzahl ab.

        Args:
            key: Stabiler Bezeichner.
            label: Anzeigename.
            category: Fachliche Gruppe.
            numerator: Position und Wert des Zaehlers.
            denominator: Position und Wert des Nenners.
            unit: Einheit des Ergebnisses.
            formula: Formel als Klartext.
            scale: Faktor auf das Ergebnis.
        """
        num_item, num = numerator
        den_item, den = denominator

        missing: list[str] = []
        if num is None:
            missing.append(num_item.value)
        if den is None:
            missing.append(den_item.value)

        computed: float | None = None
        if num is not None and den is not None:
            if den == 0:
                # Kein inf, keine Null: der Wert ist fachlich undefiniert.
                missing.append(f"{den_item.value} is 0")
            else:
                computed = (num / den) * scale

        self.ratios[key] = Ratio(
            key=key,
            label=label,
            category=category,
            value=computed,
            unit=unit,
            inputs_missing=missing,
            formula=formula,
        )

    def add_direct(
        self,
        key: str,
        label: str,
        category: RatioCategory,
        *,
        value: float | None,
        missing: list[str] | None = None,
        unit: str = "ratio",
        formula: str,
    ) -> None:
        """Legt eine bereits berechnete Kennzahl ab.

        Args:
            key: Stabiler Bezeichner.
            label: Anzeigename.
            category: Fachliche Gruppe.
            value: Ergebnis, oder None.
            missing: Fehlende Eingangspositionen.
            unit: Einheit des Ergebnisses.
            formula: Formel als Klartext.
        """
        self.ratios[key] = Ratio(
            key=key,
            label=label,
            category=category,
            value=value,
            unit=unit,
            inputs_missing=missing or [],
            formula=formula,
        )


def _safe_div(num: float | None, den: float | None) -> float | None:
    """Dividiert, sofern beide Werte vorliegen und der Nenner nicht Null ist.

    Args:
        num: Zaehler.
        den: Nenner.

    Returns:
        Quotient, oder None bei fehlendem Wert oder Nullteiler.
    """
    if num is None or den is None or den == 0:
        return None
    return num / den


def _liquidity(b: _Builder) -> None:
    """Berechnet die Liquiditaetskennzahlen.

    Args:
        b: Der sammelnde Builder.
    """
    ca = b.value(LineItem.CURRENT_ASSETS)
    cl = b.value(LineItem.CURRENT_LIABILITIES)
    cash = b.value(LineItem.CASH_AND_EQUIVALENTS)
    inv = b.value(LineItem.INVENTORY)

    b.add(
        "current_ratio",
        "Current Ratio",
        RatioCategory.LIQUIDITY,
        numerator=(LineItem.CURRENT_ASSETS, ca),
        denominator=(LineItem.CURRENT_LIABILITIES, cl),
        formula="Current assets / current liabilities",
    )

    quick_assets = None if ca is None else ca - (inv or 0.0)
    b.add_direct(
        "quick_ratio",
        "Quick Ratio",
        RatioCategory.LIQUIDITY,
        value=_safe_div(quick_assets, cl),
        missing=[
            i.value
            for i, v in ((LineItem.CURRENT_ASSETS, ca), (LineItem.CURRENT_LIABILITIES, cl))
            if v is None
        ],
        formula="(Current assets - inventories) / current liabilities",
    )

    b.add(
        "cash_ratio",
        "Cash Ratio",
        RatioCategory.LIQUIDITY,
        numerator=(LineItem.CASH_AND_EQUIVALENTS, cash),
        denominator=(LineItem.CURRENT_LIABILITIES, cl),
        formula="Cash and equivalents / current liabilities",
    )


def _leverage(b: _Builder) -> None:
    """Berechnet die Verschuldungskennzahlen.

    Args:
        b: Der sammelnde Builder.
    """
    debt = b.value(LineItem.TOTAL_DEBT)
    equity = b.value(LineItem.TOTAL_EQUITY)
    assets = b.value(LineItem.TOTAL_ASSETS)
    ebit = b.value(LineItem.EBIT) or b.value(LineItem.OPERATING_INCOME)
    interest = b.value(LineItem.INTEREST_EXPENSE)
    ebitda = b.value(LineItem.EBITDA)
    cash = b.value(LineItem.CASH_AND_EQUIVALENTS)

    b.add(
        "debt_to_equity",
        "Debt / Equity",
        RatioCategory.LEVERAGE,
        numerator=(LineItem.TOTAL_DEBT, debt),
        denominator=(LineItem.TOTAL_EQUITY, equity),
        formula="Total debt / shareholders equity",
    )
    b.add(
        "debt_to_assets",
        "Debt / Assets",
        RatioCategory.LEVERAGE,
        numerator=(LineItem.TOTAL_DEBT, debt),
        denominator=(LineItem.TOTAL_ASSETS, assets),
        formula="Total debt / total assets",
    )

    # Zinsaufwand ist in der GuV positiv gefuehrt (Vorzeichenkonvention).
    b.add_direct(
        "interest_coverage",
        "Interest Coverage",
        RatioCategory.LEVERAGE,
        value=_safe_div(ebit, abs(interest) if interest is not None else None),
        missing=[
            i.value
            for i, v in ((LineItem.EBIT, ebit), (LineItem.INTEREST_EXPENSE, interest))
            if v is None
        ],
        unit="x",
        formula="EBIT / interest expense",
    )

    net_debt = None if debt is None else debt - (cash or 0.0)
    b.add_direct(
        "net_debt_to_ebitda",
        "Net Debt / EBITDA",
        RatioCategory.LEVERAGE,
        value=_safe_div(net_debt, ebitda),
        missing=[
            i.value
            for i, v in ((LineItem.TOTAL_DEBT, debt), (LineItem.EBITDA, ebitda))
            if v is None
        ],
        unit="x",
        formula="(Total debt - cash and equivalents) / EBITDA",
    )


def _profitability(b: _Builder) -> None:
    """Berechnet die Rentabilitaetskennzahlen.

    Margen und Renditen sind dezimal (0.31 = 31 Prozent).

    Args:
        b: Der sammelnde Builder.
    """
    revenue = b.value(LineItem.REVENUE)
    gross = b.value(LineItem.GROSS_PROFIT)
    op_income = b.value(LineItem.OPERATING_INCOME)
    net_income = b.value(LineItem.NET_INCOME)
    equity = b.value(LineItem.TOTAL_EQUITY)
    assets = b.value(LineItem.TOTAL_ASSETS)
    debt = b.value(LineItem.TOTAL_DEBT)
    tax = b.value(LineItem.TAX_PROVISION)
    pretax = b.value(LineItem.PRETAX_INCOME)
    ebit = b.value(LineItem.EBIT) or op_income

    for key, label, num_item, num in (
        ("gross_margin", "Gross Margin", LineItem.GROSS_PROFIT, gross),
        ("operating_margin", "Operating Margin", LineItem.OPERATING_INCOME, op_income),
        ("net_margin", "Net Margin", LineItem.NET_INCOME, net_income),
    ):
        b.add(
            key,
            label,
            RatioCategory.PROFITABILITY,
            numerator=(num_item, num),
            denominator=(LineItem.REVENUE, revenue),
            unit="percent",
            formula=f"{num_item.value} / revenue",
        )

    b.add(
        "roe",
        "Return on Equity",
        RatioCategory.PROFITABILITY,
        numerator=(LineItem.NET_INCOME, net_income),
        denominator=(LineItem.TOTAL_EQUITY, equity),
        unit="percent",
        formula="Net income / shareholders equity",
    )
    b.add(
        "roa",
        "Return on Assets",
        RatioCategory.PROFITABILITY,
        numerator=(LineItem.NET_INCOME, net_income),
        denominator=(LineItem.TOTAL_ASSETS, assets),
        unit="percent",
        formula="Net income / total assets",
    )

    # ROIC = NOPAT / investiertes Kapital, NOPAT = EBIT * (1 - Steuerquote).
    tax_rate = _safe_div(tax, pretax)
    invested = None if equity is None else equity + (debt or 0.0)
    nopat = None if ebit is None or tax_rate is None else ebit * (1.0 - tax_rate)
    roic_missing = [
        i.value
        for i, v in (
            (LineItem.EBIT, ebit),
            (LineItem.TAX_PROVISION, tax),
            (LineItem.TOTAL_EQUITY, equity),
        )
        if v is None
    ]
    b.add_direct(
        "roic",
        "Return on Invested Capital",
        RatioCategory.PROFITABILITY,
        value=_safe_div(nopat, invested),
        missing=roic_missing,
        unit="percent",
        formula="EBIT * (1 - tax rate) / (shareholders equity + total debt)",
    )


def _efficiency(b: _Builder) -> None:
    """Berechnet die Effizienz- und Umschlagskennzahlen.

    Args:
        b: Der sammelnde Builder.
    """
    revenue = b.value(LineItem.REVENUE)
    cogs = b.value(LineItem.COST_OF_REVENUE)
    assets = b.value(LineItem.TOTAL_ASSETS)
    inv = b.value(LineItem.INVENTORY)
    recv = b.value(LineItem.RECEIVABLES)
    pay = b.value(LineItem.PAYABLES)

    b.add(
        "asset_turnover",
        "Asset Turnover",
        RatioCategory.EFFICIENCY,
        numerator=(LineItem.REVENUE, revenue),
        denominator=(LineItem.TOTAL_ASSETS, assets),
        unit="x",
        formula="Revenue / total assets",
    )
    b.add(
        "inventory_turnover",
        "Inventory Turnover",
        RatioCategory.EFFICIENCY,
        numerator=(LineItem.COST_OF_REVENUE, cogs),
        denominator=(LineItem.INVENTORY, inv),
        unit="x",
        formula="Cost of goods sold / inventories",
    )
    b.add(
        "receivables_turnover",
        "Receivables Turnover",
        RatioCategory.EFFICIENCY,
        numerator=(LineItem.REVENUE, revenue),
        denominator=(LineItem.RECEIVABLES, recv),
        unit="x",
        formula="Revenue / receivables",
    )

    dso = _safe_div(recv, revenue)
    dio = _safe_div(inv, cogs)
    dpo = _safe_div(pay, cogs)

    b.add_direct(
        "dso",
        "Days Sales Outstanding",
        RatioCategory.EFFICIENCY,
        value=None if dso is None else dso * DAYS_PER_YEAR,
        missing=[
            i.value
            for i, v in ((LineItem.RECEIVABLES, recv), (LineItem.REVENUE, revenue))
            if v is None
        ],
        unit="days",
        formula="Receivables / revenue * 365",
    )
    b.add_direct(
        "dio",
        "Days Inventory Outstanding",
        RatioCategory.EFFICIENCY,
        value=None if dio is None else dio * DAYS_PER_YEAR,
        missing=[
            i.value
            for i, v in ((LineItem.INVENTORY, inv), (LineItem.COST_OF_REVENUE, cogs))
            if v is None
        ],
        unit="days",
        formula="Inventories / cost of goods sold * 365",
    )
    b.add_direct(
        "dpo",
        "Days Payables Outstanding",
        RatioCategory.EFFICIENCY,
        value=None if dpo is None else dpo * DAYS_PER_YEAR,
        missing=[
            i.value
            for i, v in ((LineItem.PAYABLES, pay), (LineItem.COST_OF_REVENUE, cogs))
            if v is None
        ],
        unit="days",
        formula="Payables / cost of goods sold * 365",
    )

    # dso, dio und dpo sind hier noch dimensionslose Anteile eines Jahres.
    # Der Faktor DAYS_PER_YEAR wird genau einmal angewandt, auf die Summe.
    ccc = None
    ccc_missing: list[str] = []
    if dso is not None and dio is not None and dpo is not None:
        ccc = (dso + dio - dpo) * DAYS_PER_YEAR
    else:
        ccc_missing = [
            name for name, value in (("dso", dso), ("dio", dio), ("dpo", dpo)) if value is None
        ]
    b.add_direct(
        "ccc",
        "Cash Conversion Cycle",
        RatioCategory.EFFICIENCY,
        value=ccc,
        missing=ccc_missing,
        unit="days",
        formula="DSO + DIO - DPO",
    )


def _valuation(
    b: _Builder,
    *,
    price: float | None,
    market_cap: float | None,
    dividend_per_share: float | None,
) -> None:
    """Berechnet die Bewertungskennzahlen.

    Preis und Marktkapitalisierung werden injiziert, nicht selbst geholt.

    Args:
        b: Der sammelnde Builder.
        price: Aktueller Kurs je Aktie in Berichtswaehrung.
        market_cap: Marktkapitalisierung, absolut.
        dividend_per_share: Dividende je Aktie im Jahr.
    """
    eps = b.value(LineItem.EPS_DILUTED) or b.value(LineItem.EPS_BASIC)
    equity = b.value(LineItem.TOTAL_EQUITY)
    shares = b.value(LineItem.SHARES_DILUTED) or b.value(LineItem.SHARES_BASIC)
    revenue = b.value(LineItem.REVENUE)
    ebitda = b.value(LineItem.EBITDA)
    debt = b.value(LineItem.TOTAL_DEBT)
    cash = b.value(LineItem.CASH_AND_EQUIVALENTS)
    ocf = b.value(LineItem.OPERATING_CASHFLOW)
    capex = b.value(LineItem.CAPEX)

    b.add_direct(
        "pe",
        "Price / Earnings",
        RatioCategory.VALUATION,
        value=_safe_div(price, eps),
        missing=[m for m, v in (("price", price), ("eps_diluted", eps)) if v is None],
        unit="x",
        formula="Price / earnings per share",
    )

    book_per_share = _safe_div(equity, shares)
    b.add_direct(
        "pb",
        "Price / Book",
        RatioCategory.VALUATION,
        value=_safe_div(price, book_per_share),
        missing=[
            m
            for m, v in (("price", price), ("total_equity", equity), ("shares_diluted", shares))
            if v is None
        ],
        unit="x",
        formula="Price / book value per share",
    )
    b.add_direct(
        "ps",
        "Price / Sales",
        RatioCategory.VALUATION,
        value=_safe_div(market_cap, revenue),
        missing=[m for m, v in (("market_cap", market_cap), ("revenue", revenue)) if v is None],
        unit="x",
        formula="Market cap / revenue",
    )

    enterprise_value = None if market_cap is None else market_cap + (debt or 0.0) - (cash or 0.0)
    b.add_direct(
        "ev_to_ebitda",
        "EV / EBITDA",
        RatioCategory.VALUATION,
        value=_safe_div(enterprise_value, ebitda),
        missing=[m for m, v in (("market_cap", market_cap), ("ebitda", ebitda)) if v is None],
        unit="x",
        formula="(Market cap + total debt - cash and equivalents) / EBITDA",
    )

    # CapEx ist negativ gespeichert, deshalb addieren (Vorzeichenkonvention).
    fcf = None if ocf is None else ocf + (capex or 0.0)
    b.add_direct(
        "fcf_yield",
        "Free Cashflow Yield",
        RatioCategory.VALUATION,
        value=_safe_div(fcf, market_cap),
        missing=[
            m for m, v in (("operating_cashflow", ocf), ("market_cap", market_cap)) if v is None
        ],
        unit="percent",
        formula="(Operating cash flow + CapEx) / market cap",
    )
    b.add_direct(
        "dividend_yield",
        "Dividend Yield",
        RatioCategory.VALUATION,
        value=_safe_div(dividend_per_share, price),
        missing=[
            m
            for m, v in (("dividend_per_share", dividend_per_share), ("price", price))
            if v is None
        ],
        unit="percent",
        formula="Dividend per share / price",
    )


def _growth(b: _Builder) -> None:
    """Berechnet die Wachstumskennzahlen.

    Braucht mindestens zwei Perioden. Bei nur einer Periode sind alle Werte
    None mit entsprechender Begruendung.

    Args:
        b: Der sammelnde Builder.
    """
    income = b._bundle.income
    n_periods = 0 if income is None or income.data.empty else len(income.data.columns)

    def growth(item: LineItem, key: str, label: str, lookback: int = 1) -> None:
        """Legt eine Wachstumsrate ab."""
        if n_periods <= lookback:
            b.add_direct(
                key,
                label,
                RatioCategory.GROWTH,
                value=None,
                missing=[f"at least {lookback + 1} periods"],
                unit="percent",
                formula=f"{item.value}: change over {lookback} period(s)",
            )
            return
        now = b.value(item)
        then = b.value(item, period_index=-1 - lookback)
        missing = [item.value] if now is None or then is None else []
        value: float | None = None
        if not missing and then is not None and then != 0:
            assert now is not None
            # Bei negativer Basis ist eine Wachstumsrate nicht interpretierbar.
            if then < 0:
                missing = [f"{item.value} base is negative"]
            else:
                value = (now - then) / then
        elif not missing:
            missing = [f"{item.value} base is 0"]
        b.add_direct(
            key,
            label,
            RatioCategory.GROWTH,
            value=value,
            missing=missing,
            unit="percent",
            formula=f"({item.value}[t] - {item.value}[t-{lookback}]) / {item.value}[t-{lookback}]",
        )

    growth(LineItem.REVENUE, "revenue_growth_yoy", "Revenue Growth (YoY)")
    growth(LineItem.NET_INCOME, "earnings_growth_yoy", "Earnings Growth (YoY)")

    # CAGR ueber drei Perioden.
    lookback = 3
    if n_periods > lookback:
        now = b.value(LineItem.REVENUE)
        then = b.value(LineItem.REVENUE, period_index=-1 - lookback)
        value = None
        missing: list[str] = []
        if now is None or then is None:
            missing = [LineItem.REVENUE.value]
        elif then <= 0:
            missing = ["revenue base is not positive"]
        else:
            value = (now / then) ** (1.0 / lookback) - 1.0
        b.add_direct(
            "revenue_cagr_3y",
            "Revenue CAGR (3y)",
            RatioCategory.GROWTH,
            value=value,
            missing=missing,
            unit="percent",
            formula="(Revenue[t] / Revenue[t-3])^(1/3) - 1",
        )
    else:
        b.add_direct(
            "revenue_cagr_3y",
            "Revenue CAGR (3y)",
            RatioCategory.GROWTH,
            value=None,
            missing=[f"at least {lookback + 1} periods"],
            unit="percent",
            formula="(Revenue[t] / Revenue[t-3])^(1/3) - 1",
        )

    growth(LineItem.OPERATING_CASHFLOW, "operating_cashflow_growth", "Operating Cashflow Growth")


def compute_ratios(
    bundle: StatementBundle,
    *,
    price: float | None = None,
    market_cap: float | None = None,
    dividend_per_share: float | None = None,
) -> RatioSet:
    """Berechnet alle Kennzahlen eines Statement-Buendels.

    Das Buendel muss **normalisiert** sein, also durch
    :func:`~btlite.analytics.statements.normalize_bundle` gelaufen sein.

    Args:
        bundle: Normalisiertes Statement-Buendel.
        price: Aktueller Kurs je Aktie. Ohne ihn bleiben KGV, KBV und
            Dividendenrendite ``None``.
        market_cap: Marktkapitalisierung, absolut.
        dividend_per_share: Jaehrliche Dividende je Aktie.

    Returns:
        Kennzahlensatz. Nicht berechenbare Werte sind ``None`` und nennen die
        fehlenden Eingangspositionen.
    """
    b = _Builder(bundle)
    _liquidity(b)
    _leverage(b)
    _profitability(b)
    _efficiency(b)
    _valuation(b, price=price, market_cap=market_cap, dividend_per_share=dividend_per_share)
    _growth(b)

    period_end = None
    for statement in (bundle.income, bundle.balance, bundle.cashflow):
        if statement is not None and statement.periods:
            period_end = statement.periods[-1]
            break

    n_available = sum(1 for r in b.ratios.values() if r.is_available)
    _LOG.debug(
        "Kennzahlen fuer %s: %d von %d berechenbar", bundle.ticker, n_available, len(b.ratios)
    )

    return RatioSet(
        ticker=bundle.ticker,
        period=bundle.period,
        period_end=period_end,
        currency=bundle.currency,
        ratios=b.ratios,
        provenance=Provenance(
            source=DataSource.COMPUTED,
            fetched_at=bundle.provenance.fetched_at,
            from_cache=bundle.provenance.from_cache,
            delay_minutes=bundle.provenance.delay_minutes,
            note=f"Computed from statements (basis: {bundle.provenance.source.value})",
        ),
    )


def compare_peers(
    subject: RatioSet,
    peers: list[RatioSet],
    *,
    ratio_keys: list[str] | None = None,
) -> tuple[object, list[str]]:
    """Baut eine Peer-Vergleichsmatrix.

    Args:
        subject: Kennzahlen des betrachteten Titels.
        peers: Kennzahlen der Vergleichsgruppe.
        ratio_keys: Auswahl der Kennzahlen. None nimmt alle des Subjekts.

    Returns:
        Tupel aus :class:`PeerComparison` und einer Liste von Warnungen.
        Die Warnungen nennen etwa abweichende Geschaeftsjahre oder
        Waehrungen - beides macht einen Vergleich nur eingeschraenkt
        periodengenau und wird nicht verschwiegen.
    """
    import pandas as pd

    from btlite.models.fundamentals import PeerComparison

    keys = ratio_keys or sorted(subject.ratios)
    all_sets = [subject, *peers]

    rows: dict[str, dict[str, float | None]] = {}
    for rs in all_sets:
        rows[rs.ticker] = {k: rs.value_of(k) for k in keys}

    warnings: list[str] = []

    currencies = {rs.currency for rs in all_sets}
    if len(currencies) > 1:
        names = ", ".join(sorted(c.value for c in currencies))
        warnings.append(
            f"Different reporting currencies ({names}). Currency-denominated "
            "ratios are not directly comparable."
        )

    period_ends = {rs.period_end.month for rs in all_sets if rs.period_end is not None}
    if len(period_ends) > 1:
        months = ", ".join(str(m) for m in sorted(period_ends))
        warnings.append(
            f"Differing fiscal year ends (months {months}). The comparison "
            "is only partly period-aligned."
        )

    periods = {rs.period for rs in all_sets}
    if len(periods) > 1:
        warnings.append("Mixed period basis (annual/quarterly/ttm) in the comparison.")

    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame = frame.reindex(columns=keys)

    comparison = PeerComparison(
        subject=subject.ticker,
        peers=[p.ticker for p in peers],
        period=subject.period,
        data=frame,
        provenance=subject.provenance,
    )
    return comparison, warnings
