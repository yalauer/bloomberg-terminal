"""Exception-Hierarchie fuer BT-Lite.

Jede Schicht wirft ihren eigenen Typ. Das UI faengt ausschliesslich
``BTLiteError`` und rendert ``user_message`` in ein Fehlerbanner - rohe
Provider-Exceptions duerfen die ``data/``-Schicht niemals verlassen.
"""

from __future__ import annotations


class BTLiteError(Exception):
    """Basisklasse aller BT-Lite-Fehler.

    Attributes:
        user_message: Fuer Endnutzer lesbare Meldung (UI-Banner).
        detail: Technischer Kontext fuer das Log, nie im UI.
    """

    def __init__(self, user_message: str, detail: str | None = None) -> None:
        """Initialisiert den Fehler.

        Args:
            user_message: Kurze, nicht-technische Beschreibung.
            detail: Optionaler technischer Zusatzkontext.
        """
        super().__init__(user_message if detail is None else f"{user_message} ({detail})")
        self.user_message = user_message
        self.detail = detail


# ----------------------------------------------------------------- data ----
class DataError(BTLiteError):
    """Oberklasse fuer alle Fehler der Datenschicht."""


class ProviderUnavailableError(DataError):
    """Die Quelle ist nicht erreichbar (Netzwerk, DNS, 5xx)."""


class EmptyUniverseError(DataError):
    """Kein Titel des Universums lieferte Daten.

    Kein Fehler des Nutzers und keiner der Filter: die Quelle hat fuer
    jedes einzelne Symbol nichts geliefert - unerreichbar, oder ihr
    Rate-Limit hat den Lauf gestoppt.

    Eigener Typ, damit die Oberflaeche ihn von einem echten Fehler
    unterscheiden kann. Ohne ihn lief der Fall in die Filter-Pruefung und
    meldete "Field 'pe' does not exist in the universe" - technisch
    richtig und fuer den Nutzer unbrauchbar.
    """


class RateLimitError(DataError):
    """Die Quelle hat gedrosselt (429 / Bot-Erkennung).

    Wird bewusst nach oben gereicht und **nicht** umgangen.

    Attributes:
        retry_after_s: Wartezeit in Sekunden laut Quelle, falls angegeben.
    """

    def __init__(
        self,
        user_message: str,
        detail: str | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        """Initialisiert den Rate-Limit-Fehler.

        Args:
            user_message: Kurze, nicht-technische Beschreibung.
            detail: Optionaler technischer Zusatzkontext.
            retry_after_s: Empfohlene Wartezeit in Sekunden, falls bekannt.
        """
        super().__init__(user_message, detail)
        self.retry_after_s = retry_after_s


class SymbolNotFoundError(DataError):
    """Das angefragte Symbol existiert bei der Quelle nicht."""


class DataValidationError(DataError):
    """Die Antwort der Quelle passt nicht auf den Vertrag aus ``models/``."""


class InsufficientDataError(DataError):
    """Zu wenige Datenpunkte fuer die angeforderte Berechnung.

    Attributes:
        required: Mindestanzahl benoetigter Beobachtungen.
        available: Tatsaechlich vorhandene Anzahl.
    """

    def __init__(
        self,
        user_message: str,
        detail: str | None = None,
        required: int | None = None,
        available: int | None = None,
    ) -> None:
        """Initialisiert den Fehler.

        Args:
            user_message: Kurze, nicht-technische Beschreibung.
            detail: Optionaler technischer Zusatzkontext.
            required: Mindestanzahl benoetigter Beobachtungen.
            available: Tatsaechlich vorhandene Anzahl.
        """
        super().__init__(user_message, detail)
        self.required = required
        self.available = available


# ------------------------------------------------------------ analytics ----
class AnalyticsError(BTLiteError):
    """Fehler in Statement-Normalisierung, Ratios, Technicals oder Risk."""


class MetricNotComputableError(AnalyticsError):
    """Eine Kennzahl ist aus den vorliegenden Daten nicht berechenbar.

    Wird nur geworfen, wenn das Fehlen ein echter Fehler ist. Regulaer
    fehlende Werte sind ``None`` im Modell und erscheinen im UI als "n/a".
    """


# ------------------------------------------------------------ screening ----
class ScreeningError(BTLiteError):
    """Fehler in Filter-Mapping oder lokaler Filter-Engine."""


class UnknownFilterError(ScreeningError):
    """Ein Filter-Schluessel ist der Engine unbekannt."""


# ------------------------------------------------------------- backtest ----
class BacktestError(BTLiteError):
    """Fehler in Engine, Strategie oder Metrik-Berechnung."""


class StrategyConfigError(BacktestError):
    """Strategie-Parameter sind unzulaessig oder widerspruechlich."""


class EngineUnavailableError(BacktestError):
    """Die gewaehlte Backtest-Engine ist nicht installiert.

    ``vectorbt`` und ``backtrader`` sind optionale Extras.
    """


# ------------------------------------------------------------ portfolio ----
class PortfolioError(BTLiteError):
    """Fehler in Optimierung, Allokation oder Attribution."""


class OptimizationFailedError(PortfolioError):
    """Der Solver ist nicht konvergiert oder das Problem ist unzulaessig."""


# ---------------------------------------------------------------- config ----
class ConfigError(BTLiteError):
    """Ungueltige oder fehlende Konfiguration."""
