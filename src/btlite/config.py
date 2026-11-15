"""Zentrale Konfiguration via pydantic-settings.

Alle Werte sind ueber Umgebungsvariablen mit Praefix ``BTLITE_`` oder eine
``.env``-Datei im Repo-Root ueberschreibbar. Beispiel::

    BTLITE_DATA_QUOTE_POLL_INTERVAL_S=30
    BTLITE_UI_PORT=8060

Es gibt genau eine Instanz, erreichbar ueber :func:`get_settings`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class CacheSettings(BaseSettings):
    """Caching von HTTP-Antworten und DataFrames."""

    model_config = SettingsConfigDict(env_prefix="BTLITE_CACHE_", extra="ignore")

    dir: Path = Field(
        default=_REPO_ROOT / ".cache",
        description="Wurzelverzeichnis fuer requests-cache und diskcache.",
    )
    enabled: bool = Field(default=True, description="Globaler Cache-Schalter.")
    http_ttl_s: int = Field(
        default=900,
        ge=0,
        description="TTL fuer HTTP-Antworten in Sekunden (Default 15 min = Quote-Delay).",
    )
    prices_ttl_s: int = Field(
        default=12 * 3600,
        ge=0,
        description="TTL fuer Tages-OHLCV in Sekunden (Default 12 h).",
    )
    intraday_ttl_s: int = Field(
        default=60,
        ge=0,
        description=(
            "TTL fuer Intraday-Kerzen (1m bis 1h) in Sekunden. Kurz, weil eine "
            "1-Minuten-Kerze sonst eine Stunde lang dieselbe bliebe (Default 60 s)."
        ),
    )
    statements_ttl_s: int = Field(
        default=86_400,
        ge=0,
        description="TTL fuer Financial Statements in Sekunden (Default 24 h).",
    )
    composition_ttl_s: int = Field(
        default=24 * 3600, ge=0, description="TTL fuer ETF-Bestaende in Sekunden."
    )
    screener_ttl_s: int = Field(
        default=1800, ge=0, description="TTL fuer Screener-Ergebnisse in Sekunden."
    )
    fx_historical_ttl_s: int = Field(
        default=30 * 24 * 3600,
        ge=0,
        description=(
            "TTL fuer Wechselkurse abgeschlossener Handelstage in Sekunden. Lang, weil "
            "ein historischer Kurs sich nicht mehr aendert (Default 30 Tage)."
        ),
    )
    earnings_ttl_s: int = Field(
        default=24 * 3600,
        ge=0,
        description=(
            "TTL fuer Quartalszahlen-Termine in Sekunden. Ein Termin bewegt sich "
            "selten; gespeichert wird der Rohbefund, der Vergangenheitsabgleich "
            "laeuft bei jedem Lesen neu (Default 24 h)."
        ),
    )
    info_ttl_s: int = Field(
        default=4 * 3600,
        ge=0,
        description=(
            "TTL fuer Stammdaten aus Ticker.info in Sekunden (Name, Sektor, Dividende, "
            "Marktkapitalisierung). 4 h: die Marktkapitalisierung ist hoechstens einen "
            "halben Handelstag alt; der Kurs selbst kommt weiter live (Default 4 h)."
        ),
    )
    fx_latest_ttl_s: int = Field(
        default=900,
        ge=0,
        description=(
            "TTL fuer den aktuellsten Wechselkurs in Sekunden. Kurz, weil er sich noch "
            "bewegt (Default 15 min, wie der Quote-Delay)."
        ),
    )
    size_limit_bytes: int = Field(
        default=2_000_000_000, gt=0, description="Obergrenze des diskcache in Bytes."
    )


class RateLimitSettings(BaseSettings):
    """Konservatives Throttling pro Datenquelle.

    Bewusst defensiv gewaehlt: Ziel ist private Nutzung ohne Last fuer die
    Quellen, nicht maximaler Durchsatz.
    """

    model_config = SettingsConfigDict(env_prefix="BTLITE_RATELIMIT_", extra="ignore")

    yfinance_rps: float = Field(
        default=2.0, gt=0, description="Erlaubte Requests pro Sekunde gegen yfinance."
    )
    yfinance_burst: int = Field(default=5, ge=1, description="Token-Bucket-Kapazitaet yfinance.")
    finviz_rps: float = Field(
        default=0.5,
        gt=0,
        description="Requests pro Sekunde gegen Finviz (bewusst niedrig, Scraping).",
    )
    finviz_burst: int = Field(default=2, ge=1, description="Token-Bucket-Kapazitaet Finviz.")
    ishares_rps: float = Field(default=1.0, gt=0, description="Requests pro Sekunde an iShares.")
    ishares_burst: int = Field(default=2, ge=1, description="Token-Bucket-Kapazitaet iShares.")
    eodhd_rps: float = Field(default=1.0, gt=0, description="Requests pro Sekunde an EODHD.")
    eodhd_burst: int = Field(default=2, ge=1, description="Token-Bucket-Kapazitaet EODHD.")
    max_retries: int = Field(default=3, ge=0, description="Retry-Versuche bei transienten Fehlern.")
    backoff_base_s: float = Field(
        default=1.0, gt=0, description="Basis des exponentiellen Backoff in Sekunden."
    )
    backoff_max_s: float = Field(
        default=60.0, gt=0, description="Obergrenze einer einzelnen Wartezeit in Sekunden."
    )
    backoff_jitter: float = Field(
        default=0.25, ge=0, le=1, description="Relativer Jitter-Anteil auf die Wartezeit."
    )


class DataSettings(BaseSettings):
    """Verhalten der Datenschicht."""

    model_config = SettingsConfigDict(env_prefix="BTLITE_DATA_", extra="ignore")
    eodhd_api_token: str | None = Field(default=None, repr=False,
        description="Optionaler EODHD-Schluessel fuer ETF-Bestaende ohne Emittentenfeed.")

    quote_poll_interval_s: float = Field(
        default=15.0,
        ge=1.0,
        description="Poll-Intervall des PollingQuoteStream in Sekunden.",
    )
    quote_batch_size: int = Field(
        default=50, ge=1, description="Maximale Ticker pro Batch-Quote-Request."
    )
    screening_workers: int = Field(
        default=4,
        ge=1,
        le=8,
        description=(
            "Maximale parallele Titelabrufe beim Live-Screening. Der gemeinsame "
            "Token-Bucket begrenzt weiterhin die Request-Rate."
        ),
    )
    quote_delay_notice_min: int = Field(
        default=15,
        ge=0,
        description="Im UI angezeigter Delay-Hinweis in Minuten. yfinance ist NICHT Echtzeit.",
    )
    auto_adjust_prices: bool = Field(
        default=True,
        description=(
            "Split- und dividendenbereinigte Kurse verwenden. Semantik ist in "
            "docs/DATA_SOURCES.md dokumentiert."
        ),
    )
    request_timeout_s: float = Field(
        default=30.0, gt=0, description="HTTP-Timeout pro Request in Sekunden."
    )
    offline: bool = Field(
        default=True,
        description=(
            "Wenn True, sind alle Netzwerkzugriffe verboten und es bedienen die "
            "eingefrorenen Fixtures. Default ist True, damit die App ohne Netz "
            "startet und sofort etwas zeigt. Live-Daten per BTLITE_DATA_OFFLINE=0."
        ),
    )
    fixtures_dir: Path = Field(
        default=_REPO_ROOT / "tests" / "fixtures",
        description="Quelle der eingefrorenen API-Antworten fuer den Offline-Modus.",
    )


class AnalyticsSettings(BaseSettings):
    """Konventionen der Rechenschicht."""

    model_config = SettingsConfigDict(env_prefix="BTLITE_ANALYTICS_", extra="ignore")

    trading_days_per_year: int = Field(
        default=252, gt=0, description="Annualisierungsfaktor fuer Tagesdaten."
    )
    risk_free_rate: float = Field(
        default=0.02,
        description="Jaehrlicher risikofreier Zins, dezimal (0.02 = 2 Prozent).",
    )
    benchmark_ticker: str = Field(
        default="SPY", description="Default-Benchmark fuer Beta, Alpha und Tracking Error."
    )
    var_confidence: float = Field(
        default=0.95, gt=0, lt=1, description="Konfidenzniveau fuer VaR und CVaR, dezimal."
    )
    min_observations: int = Field(
        default=30,
        ge=2,
        description="Mindestanzahl Beobachtungen, sonst InsufficientDataError.",
    )


class BacktestSettings(BaseSettings):
    """Defaults fuer Kosten und Ausfuehrung im Backtest."""

    model_config = SettingsConfigDict(env_prefix="BTLITE_BACKTEST_", extra="ignore")

    init_cash: float = Field(default=100_000.0, gt=0, description="Startkapital in Kontowaehrung.")
    fees: float = Field(
        default=0.0005, ge=0, description="Kommission je Trade, dezimal (0.0005 = 5 bps)."
    )
    slippage: float = Field(
        default=0.0005, ge=0, description="Slippage je Trade, dezimal (0.0005 = 5 bps)."
    )
    engine: Literal["vectorbt", "backtrader"] = Field(
        default="vectorbt", description="Default-Engine."
    )


class PortfolioSettings(BaseSettings):
    """Defaults der Optimierung."""

    model_config = SettingsConfigDict(env_prefix="BTLITE_PORTFOLIO_", extra="ignore")

    allow_shorting: bool = Field(
        default=False, description="Erlaubt negative Gewichte in der Optimierung."
    )
    max_weight: float = Field(
        default=1.0, gt=0, le=1, description="Obergrenze je Position, dezimal."
    )
    frontier_points: int = Field(default=50, ge=5, description="Stuetzstellen der Effizienzgrenze.")
    holdings_file: Path = Field(
        default=_REPO_ROOT / "data" / "holdings.json",
        description=(
            "JSON-Datei mit gespeicherten Depots und Watchlists. Liegt im "
            "Projektordner, damit das Werkzeug ausserhalb seines Verzeichnisses "
            "nichts ablegt: wer den Ordner kopiert, hat den Bestand dabei, und "
            "wer ihn loescht, laesst nichts zurueck. ``data/`` ist in "
            ".gitignore - Depotbestand ist Nutzerdatum, kein Projektinhalt, und "
            "aus einer Git-Historie bekaeme man ihn nur mit Aufwand wieder "
            "heraus. Ueberschreibbar per BTLITE_PORTFOLIO_HOLDINGS_FILE."
        ),
    )


class UISettings(BaseSettings):
    """Dash-Shell und Darstellung."""

    model_config = SettingsConfigDict(env_prefix="BTLITE_UI_", extra="ignore")

    host: str = Field(default="127.0.0.1", description="Bind-Adresse. Bewusst nur lokal.")
    port: int = Field(default=8050, ge=1024, le=65_535, description="TCP-Port der Dash-App.")
    debug: bool = Field(default=False, description="Dash-Debugmodus mit Hot Reload.")
    display_timezone: str = Field(
        default="Europe/Berlin",
        description="Zeitzone NUR fuer die Anzeige. Intern ist alles UTC.",
    )
    refresh_interval_ms: int = Field(
        default=15_000, ge=1000, description="Intervall des UI-Quote-Refresh in Millisekunden."
    )
    default_tickers: list[str] = Field(
        default_factory=lambda: ["AAPL", "MSFT", "NVDA"],
        description="Vorbelegung der Kommandozeile beim Start.",
    )


class Settings(BaseSettings):
    """Wurzel der Konfiguration."""

    model_config = SettingsConfigDict(
        env_prefix="BTLITE_",
        env_file=_REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    repo_root: Path = Field(default=_REPO_ROOT, description="Wurzel des Repositories.")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Globaler Log-Level."
    )
    log_json: bool = Field(default=False, description="Strukturierte JSON-Logs statt Klartext.")

    cache: CacheSettings = Field(default_factory=CacheSettings)
    ratelimit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    analytics: AnalyticsSettings = Field(default_factory=AnalyticsSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    portfolio: PortfolioSettings = Field(default_factory=PortfolioSettings)
    ui: UISettings = Field(default_factory=UISettings)

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper(cls, v: str) -> str:
        """Akzeptiert Log-Level unabhaengig von der Schreibweise."""
        return v.upper() if isinstance(v, str) else v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Liefert die prozessweite Settings-Instanz.

    Returns:
        Die gecachte :class:`Settings`-Instanz.
    """
    return Settings()
