"""Prueft, dass das Werkzeug nichts ausserhalb des Projektordners ablegt.

Der Nutzer hat das ausdruecklich verlangt: "es soll nichts auf C: abgelegt
werden, es soll alles innerhalb des project ordners bleiben." Die Vorgabe ist
nicht nur Ordnungsliebe - wer den Ordner kopiert, soll seinen Bestand dabei
haben, und wer ihn loescht, soll nichts zuruecklassen.

Diese Tests sind Regressionsschutz. Ein neuer Default, der wieder ins
Nutzerverzeichnis zeigt, faellt hier auf und nicht erst, wenn jemand sein
Depot nach einem Umzug vermisst.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from btlite.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Frische Settings mit allen Defaults."""
    return Settings()


def _is_inside(path: Path, root: Path) -> bool:
    """Sagt, ob ein Pfad unterhalb einer Wurzel liegt.

    Args:
        path: Zu pruefender Pfad.
        root: Erwartete Wurzel.

    Returns:
        True wenn ``path`` gleich ``root`` ist oder darunter liegt.
    """
    resolved = Path(path).resolve()
    root = Path(root).resolve()
    return resolved == root or root in resolved.parents


class TestConfiguredPaths:
    """Jeder konfigurierte Pfad liegt im Projekt."""

    def test_holdings_file_is_in_the_project(self, settings: Settings) -> None:
        """Der Depotbestand ist der wichtigste Fall: er ist nicht ersetzbar."""
        assert _is_inside(settings.portfolio.holdings_file, settings.repo_root)

    def test_holdings_file_is_not_in_the_old_user_directory(self, settings: Settings) -> None:
        """Frueher lag die Datei unter ``~/.btlite`` - genau das soll weg.

        Bewusst NICHT "liegt nicht unter Path.home()" geprueft: ein
        Projektordner darf sehr wohl im Heimatverzeichnis liegen, etwa unter
        OneDrive. Entscheidend ist, dass die Ablage im Projekt liegt und
        nicht in einem versteckten Nebenort.
        """
        assert not _is_inside(settings.portfolio.holdings_file, Path.home() / ".btlite")

    def test_cache_dir_is_in_the_project(self, settings: Settings) -> None:
        assert _is_inside(settings.cache.dir, settings.repo_root)

    def test_fixtures_dir_is_in_the_project(self, settings: Settings) -> None:
        assert _is_inside(settings.data.fixtures_dir, settings.repo_root)

    def test_every_path_default_stays_inside(self, settings: Settings) -> None:
        """Faengt auch Pfade, die spaeter dazukommen.

        Die Einzeltests oben benennen die bekannten Faelle; dieser hier
        greift, wenn jemand ein neues Path-Feld mit einem Default ausserhalb
        des Projekts ergaenzt.
        """
        root = Path(settings.repo_root).resolve()
        outside: list[str] = []
        for group_name in ("cache", "data", "analytics", "backtest", "portfolio", "ui"):
            group = getattr(settings, group_name)
            for field_name in type(group).model_fields:
                value = getattr(group, field_name)
                if isinstance(value, Path) and not _is_inside(value, root):
                    outside.append(f"{group_name}.{field_name} = {value}")
        assert not outside, f"Pfade ausserhalb des Projekts: {outside}"


class TestHoldingsLocation:
    """Der Bestand ist Nutzerdatum und gehoert nicht in die Versionierung."""

    def test_data_directory_is_git_ignored(self, settings: Settings) -> None:
        """``/data/`` steht in .gitignore - verankert am Projektstamm.

        Ohne den Eintrag waere jede Kursaenderung ein Commit-Kandidat, und
        ein spaeter geteiltes Repository enthielte die Vermoegenswerte des
        Nutzers dauerhaft.

        Verankert, weil ein nacktes ``data/`` jedes Verzeichnis dieses
        Namens trifft - auch die Datenschicht ``src/btlite/data``. Genau so
        stand es zuerst da: neue Dateien der Datenschicht waeren still
        unversioniert geblieben.
        """
        gitignore = Path(settings.repo_root) / ".gitignore"
        if not gitignore.exists():
            pytest.skip("Kein .gitignore vorhanden")
        patterns = {
            line.strip().rstrip("/")
            for line in gitignore.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
        holdings_dir = Path(settings.portfolio.holdings_file).parent.name
        assert f"/{holdings_dir}" in patterns, f"'/{holdings_dir}/' fehlt in .gitignore"
        assert holdings_dir not in patterns, (
            f"Nacktes {holdings_dir!r} trifft auch src/btlite/{holdings_dir}"
        )

    def test_the_override_still_works(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Wer den Pfad bewusst verlegt, darf das weiterhin.

        Die Tests selbst haengen daran: sie schreiben nach ``tmp_path`` und
        wuerden sonst in den echten Bestand des Nutzers greifen.
        """
        target = tmp_path / "woanders" / "holdings.json"
        monkeypatch.setenv("BTLITE_PORTFOLIO_HOLDINGS_FILE", str(target))
        assert Settings().portfolio.holdings_file == target


class TestYFinanceCache:
    """yfinance legt eigene Dateien an - auch die gehoeren ins Projekt."""

    def test_redirect_targets_the_project_cache(self, tmp_path: Path) -> None:
        """Die Umlenkung zeigt unter ``CacheSettings.dir``."""
        from btlite.config import CacheSettings
        from btlite.core import cache as cache_module

        cfg = CacheSettings(dir=tmp_path / "cache")
        # Der Merker ist prozessweit; fuer den Test zuruecksetzen, sonst
        # meldet ein frueherer Aufruf das Ergebnis eines anderen Pfades.
        monkey = cache_module._yf_cache_redirected
        cache_module._yf_cache_redirected = False
        try:
            target = cache_module.redirect_yfinance_cache(cfg)
        finally:
            cache_module._yf_cache_redirected = monkey
        assert _is_inside(target, tmp_path)
        assert target.exists(), "Das Verzeichnis muss angelegt sein"

    def test_the_redirect_is_called_where_yfinance_is_imported(self) -> None:
        """Jede Importstelle muss die Umlenkung ausloesen.

        Sonst schreibt der erste Abruf wieder ins Nutzerverzeichnis - der
        Default greift, bevor irgendjemand ihn korrigiert hat.
        """
        import inspect

        from btlite.data import rates, yfinance_provider

        for module in (rates, yfinance_provider):
            source = inspect.getsource(module)
            imports = source.count("import yfinance as yf")
            # Mit Klammer gezaehlt, damit die Importzeile der Funktion
            # selbst nicht mitzaehlt - sie ruft nichts auf.
            redirects = source.count("redirect_yfinance_cache(")
            assert imports > 0, f"{module.__name__}: erwartet yfinance zu importieren"
            assert redirects >= imports, (
                f"{module.__name__}: {imports} yfinance-Importe, aber nur {redirects} Umlenkungen"
            )
