"""Portfolio: Optimierung, Attribution, Rebalancing.

Rechenschicht ohne Netzwerkzugriff. Kurse werden injiziert
(Architekturregel 2).

Module:
    optimizer: Wrapper um PyPortfolioOpt, alle Zielfunktionen aus
        :class:`~btlite.models.portfolio.OptimizationObjective`, plus
        Effizienzgrenze, Korrelationsmatrix und Discrete Allocation.
    attribution: Zerlegung der Portfoliorendite in Positionsbeitraege,
        optional gegen eine Benchmark.
    rebalance: Drift-Berechnung und Order-**Vorschlaege**. Das Projekt hat
        bewusst keine Broker-Anbindung und fuehrt keine Orders aus.
"""
