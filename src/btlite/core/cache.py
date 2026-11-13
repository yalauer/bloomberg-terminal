"""Zweistufiges Caching fuer die Datenschicht.

Zwei getrennte Ebenen, weil sie unterschiedliche Dinge speichern:

1. **HTTP-Ebene** (:func:`make_cached_session`, ``requests-cache``): rohe
   Antworten der Quellen. Verhindert, dass ein wiederholter Abruf desselben
   Symbols erneut ins Netz geht.
2. **DataFrame-Ebene** (:class:`DataFrameCache`, ``diskcache`` + parquet):
   fertig normalisierte Payloads. Spart neben dem Netzwerk auch das
   Parsen und Normalisieren.

Beide Ebenen respektieren :attr:`btlite.config.CacheSettings.enabled`. Ist
der Cache aus, verhaelt sich :class:`DataFrameCache` wie ein Nullobjekt: jedes
``get`` ist ein Miss, jedes ``set`` ein No-op.

Im Offline-Modus (:attr:`btlite.config.DataSettings.offline`) geht ueber diese
Schicht nie ein Request ins Netz - :func:`make_cached_session` gibt dann eine
Session zurueck, die ausschliesslich aus dem Cache bedient.

TTL-Einheiten: alle ``ttl_s``-Parameter sind Sekunden.
"""

from __future__ import annotations

import hashlib
import io
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from btlite.config import CacheSettings, get_settings
from btlite.core.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd
    import requests

_LOG = get_logger(__name__)

#: Unterverzeichnis des diskcache unterhalb von ``CacheSettings.dir``.
_DF_SUBDIR: Final[str] = "dataframes"

#: Dateiname (ohne Endung) der requests-cache-Datenbank.
_HTTP_DB_NAME: Final[str] = "http"

#: Unterverzeichnis fuer den yfinance-eigenen Cache unterhalb von
#: ``CacheSettings.dir``.
_YF_SUBDIR: Final[str] = "yfinance"

#: Merker, damit die Umlenkung nur einmal je Prozess laeuft.
_yf_cache_redirected = False


def cache_key(*parts: object) -> str:
    """Baut einen stabilen Cache-Schluessel aus beliebigen Bestandteilen.

    Die Bestandteile werden mit ``|`` verbunden. Wird der Schluessel zu lang
    fuer einen Dateinamen, ersetzt ihn ein SHA-256-Digest mit lesbarem Praefix.

    Args:
        *parts: Bestandteile des Schluessels, etwa Quelle, Ticker, Intervall.

    Returns:
        Schluessel als String, hoechstens 200 Zeichen lang.
    """
    raw = "|".join(str(p) for p in parts)
    if len(raw) <= 200:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"{raw[:120]}|sha256:{digest}"


class DataFrameCache:
    """Persistenter DataFrame-Cache auf Basis von ``diskcache`` und parquet.

    DataFrames werden als parquet-Bytes abgelegt. Das haelt Dtypes, den
    tz-aware ``DatetimeIndex`` und die Spaltennamen verlustfrei - anders als
    CSV, das Zeitzonen und Dtypes stillschweigend verliert.

    Ist ``settings.enabled`` False, ist die Instanz ein Nullobjekt: kein
    Verzeichnis wird angelegt, jedes ``get_df`` liefert None.
    """

    def __init__(self, settings: CacheSettings | None = None) -> None:
        """Initialisiert den Cache.

        Args:
            settings: Cache-Settings. ``None`` nimmt die globalen.
        """
        self._settings = settings if settings is not None else get_settings().cache
        self._lock = threading.Lock()
        self._cache: Any | None = None

    @property
    def settings(self) -> CacheSettings:
        """Die zugrunde liegenden Cache-Settings.

        Returns:
            Die Settings-Instanz.
        """
        return self._settings

    @property
    def enabled(self) -> bool:
        """Ob der Cache aktiv ist.

        Returns:
            True wenn Lesen und Schreiben wirksam sind.
        """
        return self._settings.enabled

    @property
    def directory(self) -> Path:
        """Verzeichnis des diskcache.

        Returns:
            Pfad zum DataFrame-Cache-Verzeichnis.
        """
        return self._settings.dir / _DF_SUBDIR

    def _store(self) -> Any | None:
        """Oeffnet den diskcache lazy und liefert ihn zurueck.

        Returns:
            Die ``diskcache.Cache``-Instanz, oder None wenn der Cache aus ist
            oder sich nicht oeffnen laesst.
        """
        if not self._settings.enabled:
            return None
        with self._lock:
            if self._cache is None:
                import diskcache

                self.directory.mkdir(parents=True, exist_ok=True)
                self._cache = diskcache.Cache(
                    str(self.directory),
                    size_limit=self._settings.size_limit_bytes,
                )
            return self._cache

    def get_df(self, key: str) -> pd.DataFrame | None:
        """Liest einen DataFrame aus dem Cache.

        Args:
            key: Cache-Schluessel, idealerweise aus :func:`cache_key`.

        Returns:
            Der gecachte DataFrame, oder None bei Miss, abgelaufener TTL oder
            deaktiviertem Cache.
        """
        store = self._store()
        if store is None:
            return None
        try:
            payload = store.get(key)
        except Exception as exc:  # pragma: no cover - defekte Cache-Datei
            _LOG.warning("DataFrame-Cache nicht lesbar (key=%s): %s", key, exc)
            return None
        if payload is None:
            return None

        import pandas as pd

        try:
            return pd.read_parquet(io.BytesIO(payload))
        except Exception as exc:  # pragma: no cover - korrupter Eintrag
            _LOG.warning("Cache-Eintrag nicht deserialisierbar (key=%s): %s", key, exc)
            return None

    def set_df(self, key: str, df: pd.DataFrame, ttl_s: float | None = None) -> bool:
        """Legt einen DataFrame im Cache ab.

        Args:
            key: Cache-Schluessel, idealerweise aus :func:`cache_key`.
            df: Abzulegender DataFrame.
            ttl_s: Lebensdauer in Sekunden. ``None`` bedeutet unbegrenzt.

        Returns:
            True wenn der Eintrag geschrieben wurde, sonst False.
        """
        store = self._store()
        if store is None:
            return False
        buffer = io.BytesIO()
        try:
            df.to_parquet(buffer, engine="pyarrow", compression="snappy")
        except Exception as exc:
            # Nicht serialisierbare Payloads (etwa Objekt-Spalten mit
            # gemischten Typen) sind kein Grund, den Abruf scheitern zu
            # lassen - der Cache ist eine Optimierung, keine Pflicht.
            _LOG.warning("DataFrame nicht als parquet serialisierbar (key=%s): %s", key, exc)
            return False
        try:
            result = store.set(key, buffer.getvalue(), expire=ttl_s)
        except Exception as exc:  # pragma: no cover - volles Laufwerk
            _LOG.warning("DataFrame-Cache nicht schreibbar (key=%s): %s", key, exc)
            return False
        return bool(result)

    def cached_df(
        self,
        key: str,
        ttl_s: float | None,
        loader: Callable[[], pd.DataFrame],
    ) -> tuple[pd.DataFrame, bool]:
        """Liefert einen DataFrame aus dem Cache oder laedt ihn nach.

        Args:
            key: Cache-Schluessel, idealerweise aus :func:`cache_key`.
            ttl_s: Lebensdauer in Sekunden fuer einen frisch geladenen Eintrag.
            loader: Funktion, die den DataFrame bei einem Miss beschafft. Wird
                bei einem Hit nicht aufgerufen.

        Returns:
            Tupel aus DataFrame und einem Flag, das angibt, ob der Wert aus
            dem Cache kam. Das Flag fuellt ``Provenance.from_cache``.
        """
        hit = self.get_df(key)
        if hit is not None:
            _LOG.debug("DataFrame-Cache Hit: %s", key)
            return hit, True
        _LOG.debug("DataFrame-Cache Miss: %s", key)
        df = loader()
        self.set_df(key, df, ttl_s)
        return df, False

    def delete(self, key: str) -> bool:
        """Entfernt einen Eintrag aus dem Cache.

        Args:
            key: Cache-Schluessel.

        Returns:
            True wenn ein Eintrag entfernt wurde.
        """
        store = self._store()
        if store is None:
            return False
        return bool(store.delete(key))

    def clear(self) -> None:
        """Leert den gesamten DataFrame-Cache."""
        store = self._store()
        if store is not None:
            store.clear()

    def close(self) -> None:
        """Schliesst den zugrunde liegenden diskcache."""
        with self._lock:
            if self._cache is not None:
                self._cache.close()
                self._cache = None

    def __enter__(self) -> DataFrameCache:
        """Betritt den Kontext.

        Returns:
            Der Cache selbst.
        """
        return self

    def __exit__(self, *exc: object) -> None:
        """Schliesst den Cache beim Verlassen des Kontexts."""
        self.close()


def make_cached_session(
    settings: CacheSettings | None = None,
    *,
    offline: bool = False,
    expire_after_s: float | None = None,
) -> requests.Session:
    """Baut eine HTTP-Session mit transparentem Antwort-Cache.

    Args:
        settings: Cache-Settings. ``None`` nimmt die globalen.
        offline: Wenn True, bedient die Session ausschliesslich aus dem
            Cache. Ein Miss fuehrt dann zu einem Fehler, nie zu einem
            Netzwerkzugriff.
        expire_after_s: TTL der Antworten in Sekunden. ``None`` nimmt
            ``CacheSettings.http_ttl_s``.

    Returns:
        Eine ``requests``-kompatible Session. Bei deaktiviertem Cache und
        ``offline=False`` eine gewoehnliche :class:`requests.Session`.

    Raises:
        RuntimeError: Wenn ``offline=True`` und der Cache deaktiviert ist -
            eine Kombination, die nur zu Fehlern fuehren koennte.
    """
    cfg = settings if settings is not None else get_settings().cache

    if not cfg.enabled:
        if offline:
            msg = "Offline mode requires an active HTTP cache (BTLITE_CACHE_ENABLED=true)"
            raise RuntimeError(msg)
        import requests

        return requests.Session()

    import requests_cache

    cfg.dir.mkdir(parents=True, exist_ok=True)
    ttl_s = cfg.http_ttl_s if expire_after_s is None else expire_after_s
    session = requests_cache.CachedSession(
        cache_name=str(cfg.dir / _HTTP_DB_NAME),
        backend="sqlite",
        expire_after=ttl_s,
        allowable_codes=(200,),
        stale_if_error=True,
    )
    if offline:
        # only_if_cached: ein Miss liefert 504 statt eines Netzwerkzugriffs.
        session.settings.only_if_cached = True
    return session


def redirect_yfinance_cache(settings: CacheSettings | None = None) -> Path:
    """Verlegt den yfinance-eigenen Cache in den Projektordner.

    yfinance legt unabhaengig von unserer Cache-Schicht eigene SQLite-Dateien
    an - Zeitzonen je Ticker, Cookies und ISIN-Zuordnungen. Der Default ist
    ein Nutzerverzeichnis ausserhalb des Projekts (unter Windows
    ``%LOCALAPPDATA%`` / ``py-yfinance``). Das widerspricht der Vorgabe, dass das
    Werkzeug nichts ausserhalb seines eigenen Ordners ablegt: wer das Projekt
    loescht, soll nichts zuruecklassen.

    ``yfinance.set_tz_cache_location`` ruft intern ``set_cache_location`` und
    deckt damit alle drei Caches ab, nicht nur den Zeitzonen-Cache - am Paket
    geprueft, nicht angenommen.

    Der Aufruf ist idempotent und laeuft nur einmal je Prozess.

    Args:
        settings: Cache-Konfiguration. None nimmt die globalen Settings.

    Returns:
        Das gesetzte Verzeichnis.
    """
    global _yf_cache_redirected

    cfg = settings if settings is not None else get_settings().cache
    target = cfg.dir / _YF_SUBDIR
    if _yf_cache_redirected:
        return target

    target.mkdir(parents=True, exist_ok=True)
    import yfinance

    yfinance.set_tz_cache_location(str(target))
    _yf_cache_redirected = True
    _LOG.debug("yfinance-Cache umgelenkt nach %s", target)
    return target
