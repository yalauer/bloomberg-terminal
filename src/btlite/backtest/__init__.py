"""Backtesting: Engine-Protocol, Adapter, Strategien, Metriken.

Die Schicht ist zweistufig getrennt (siehe ``docs/ARCHITECTURE.md``):

1. Eine :class:`~btlite.backtest.base.Strategy` erzeugt aus Kursdaten einen
   :class:`~btlite.models.backtest.SignalSet`. Sie kennt keine Engine, keine
   Kosten und kein Kapital.
2. Eine :class:`~btlite.backtest.base.BacktestEngine` fuehrt die Signale aus
   und liefert ein :class:`~btlite.models.backtest.BacktestResult`.

Module dieser Schicht:

* ``base.py`` - die Protocols und die Strategie-Basisklasse.
* ``metrics.py`` - engine-unabhaengige Kennzahlen. **Beide** Engines rufen
  dieses Modul auf, damit ihre Zahlen vergleichbar sind und nicht von der
  Konvention der jeweiligen Bibliothek abhaengen.
* ``vbt_engine.py`` - die primaere Engine auf Basis von ``vectorbt``.
* ``bt_engine.py`` - optionaler Zweit-Adapter auf Basis von ``backtrader``.
* ``registry.py`` - Nachschlagewerk der verfuegbaren Strategien fuer das UI.
* ``strategies/`` - die vier Referenzstrategien.

Beide Engines importieren ihre Bibliothek **lazy** und melden ueber
``is_available()``, ob sie nutzbar sind. Das UI bietet nur an, was wirklich
installiert ist.

Look-ahead-Bias ist an zwei Stellen ausgeschlossen: die Strategien nutzen
ausschliesslich kausale Fenster, und die Engine verschiebt die Signale um
``SignalSet.execution_lag`` Bars, bevor sie Orders daraus macht.
"""
