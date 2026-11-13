"""Screening: lokale Filter-DSL, Orchestrierung, Presets.

Drei Bausteine:

* :mod:`btlite.screening.filters` - reine Auswertung von
  :class:`~btlite.models.screening.FilterGroup` gegen einen DataFrame.
  Die NaN-Semantik ist dort verbindlich festgelegt: nur ``IS_NULL`` trifft
  fehlende Werte, ``NEQ`` und ``NOT_IN`` schliessen sie aus.
* :mod:`btlite.screening.engine` - setzt Remote-Filter, lokale Praedikate,
  Sortierung und Kuerzung zusammen. Der Provider wird injiziert
  (Architekturregel 2), die Engine holt nie selbst Daten.
* :mod:`btlite.screening.presets` - die vier Presets Value, Quality,
  Momentum und Dividend.
"""
