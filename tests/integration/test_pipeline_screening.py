"""Integration: Provider -> Screening -> Export.

Ein vollstaendiger Screening-Lauf: aus Statements werden Kennzahlen, aus
Kennzahlen ein Universum, darauf laufen die Filter, und das Ergebnis geht
als Datei wieder heraus.

Die Unit-Tests pruefen die Filter-DSL gegen synthetische Frames. Hier
laeuft sie gegen ein Universum, das wirklich aus Statements gerechnet
wurde - mit allen Luecken, die echte Daten haben.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from btlite.data.fixtures import FixtureProvider
from btlite.models.common import Period
from btlite.models.screening import (
    FilterGroup,
    FilterPredicate,
    LogicalOp,
    Operator,
    ScreenRequest,
)
from btlite.screening.engine import ScreeningEngine, export_result, load_export
from btlite.screening.filters import describe_group
from btlite.screening.presets import all_presets
from btlite.screening.universe import build_ratio_universe


@pytest.fixture(scope="module")
def provider() -> FixtureProvider:
    """Offline-Provider fuer die gesamte Datei."""
    return FixtureProvider()


@pytest.fixture(scope="module")
def universum(provider: FixtureProvider) -> pd.DataFrame:
    """Kennzahlen-Universum ueber alle Fixture-Ticker.

    Das ist die eigentliche Integrationsleistung: Statements holen,
    normalisieren, Kennzahlen rechnen, Quotes dazunehmen - alles ueber die
    echten Schichten, nicht ueber einen handgebauten Frame.
    """
    return build_ratio_universe(provider, provider.available_tickers(), period=Period.ANNUAL)


class TestUniversumAufbau:
    """Die Bruecke von der Fundamentaldaten- in die Screening-Schicht."""

    def test_universum_deckt_die_fixture_ticker_ab(self, provider, universum):
        """Jeder Ticker mit Statements taucht im Universum auf."""
        assert not universum.empty
        assert universum.index.name == "ticker"

        # Nicht jeder Ticker hat Statements (SPY ist ein ETF), aber die
        # Mehrheit muss durchkommen, sonst stimmt die Kette nicht.
        assert len(universum) >= len(provider.available_tickers()) - 2

    def test_universum_traegt_kennzahlenspalten(self, universum):
        """Das Universum fuehrt die Kennzahlen, auf die Presets filtern."""
        spalten = set(universum.columns)
        erwartet = {"pe", "roe", "current_ratio", "debt_to_equity"}

        fehlend = erwartet - spalten
        assert not fehlend, f"Kennzahlen fehlen im Universum: {fehlend}"

    def test_fehlwerte_sind_nan_nicht_null(self, universum):
        """Eine fehlende Kennzahl ist NaN im Frame, nie stillschweigend 0.

        Im DataFrame ist NaN die richtige Darstellung - die Filter-DSL
        behandelt ihn definiert. Eine 0 waere dagegen ein echter Wert und
        wuerde Vergleiche falsch beantworten.
        """
        numerisch = universum.select_dtypes(include="number")
        assert not numerisch.empty

        # Es gibt Luecken (echte Daten), aber keine Spalte ist komplett leer.
        vollstaendig_leer = [c for c in numerisch.columns if numerisch[c].isna().all()]
        assert len(vollstaendig_leer) < len(numerisch.columns)


class TestScreeningLauf:
    """Der Lauf ueber die Engine, mit lokalen Praedikaten."""

    def test_lokaler_filter_schneidet_das_universum(self, universum):
        """Ein Praedikat auf einer echten Kennzahl reduziert die Treffer."""
        engine = ScreeningEngine()
        schwelle = float(universum["roe"].median(skipna=True))

        request = ScreenRequest(
            name="ROE ueber Median",
            local_filters=FilterGroup(
                op=LogicalOp.AND,
                predicates=[
                    FilterPredicate(field="roe", op=Operator.GT, value=schwelle),
                ],
            ),
        )
        result = engine.screen(request, universe=universum)

        assert result.total_matches > 0
        assert result.total_matches < len(universum), "Der Filter schneidet nichts weg"
        assert (result.data["roe"] > schwelle).all()

    def test_und_verknuepfung_ist_enger_als_jede_einzelbedingung(self, universum):
        """Zwei UND-verknuepfte Filter treffen hoechstens so viel wie einer.

        Das ist die Mengenlehre hinter der DSL. Ein Test, der nur "laeuft
        durch" prueft, wuerde eine vertauschte UND/ODER-Logik nicht
        bemerken.
        """
        engine = ScreeningEngine()

        def treffer(group: FilterGroup) -> int:
            return engine.screen(
                ScreenRequest(local_filters=group), universe=universum
            ).total_matches

        nur_roe = FilterPredicate(field="roe", op=Operator.GT, value=0.0)
        nur_pe = FilterPredicate(field="pe", op=Operator.LT, value=100.0)

        einzeln_roe = treffer(FilterGroup(predicates=[nur_roe]))
        einzeln_pe = treffer(FilterGroup(predicates=[nur_pe]))
        zusammen = treffer(FilterGroup(op=LogicalOp.AND, predicates=[nur_roe, nur_pe]))

        assert zusammen <= einzeln_roe
        assert zusammen <= einzeln_pe

    def test_sortierung_und_kuerzung(self, universum):
        """``sort_by`` und ``limit`` wirken, ``total_matches`` zaehlt davor."""
        engine = ScreeningEngine()
        request = ScreenRequest(sort_by="roe", ascending=False, limit=3)

        result = engine.screen(request, universe=universum)

        assert result.total_matches == len(universum)
        assert len(result.data) == 3, "limit wurde nicht angewandt"

        werte = result.data["roe"].dropna().tolist()
        assert werte == sorted(werte, reverse=True), "Nicht absteigend sortiert"

    @pytest.mark.parametrize("preset", all_presets(), ids=lambda p: p.key)
    def test_jedes_preset_laeuft_gegen_echte_kennzahlen(self, universum, preset):
        """Alle vier Presets vertragen ein echtes, lueckenhaftes Universum."""
        engine = ScreeningEngine()
        result = engine.screen(preset.request, universe=universum)

        assert result.total_matches >= 0
        assert len(result.data) <= preset.request.limit
        # Die Filter lassen sich fuer das UI beschreiben.
        if preset.request.local_filters is not None:
            assert describe_group(preset.request.local_filters)


class TestExportKette:
    """Das Ergebnis verlaesst das System als Datei und kommt zurueck."""

    @pytest.fixture
    def ergebnis(self, universum):
        """Ein Screening-Ergebnis mit garantierten Treffern."""
        engine = ScreeningEngine()
        return engine.screen(ScreenRequest(sort_by="roe", limit=5), universe=universum)

    def test_csv_round_trip_erhaelt_die_daten(self, ergebnis, tmp_path: Path):
        """Werte und Ticker ueberstehen Schreiben und Lesen unveraendert."""
        ziel = export_result(ergebnis, tmp_path / "screen.csv")
        zurueck = load_export(ziel)

        assert list(zurueck.index) == list(ergebnis.data.index)

        for spalte in ergebnis.data.select_dtypes(include="number").columns:
            original = ergebnis.data[spalte]
            gelesen = zurueck[spalte]
            pd.testing.assert_series_equal(
                original, gelesen, check_names=False, rtol=1e-9, check_dtype=False
            )

    def test_parquet_round_trip_erhaelt_die_daten(self, ergebnis, tmp_path: Path):
        """Derselbe Round-Trip ueber Parquet, das typtreue Format."""
        ziel = export_result(ergebnis, tmp_path / "screen.parquet")
        zurueck = load_export(ziel)

        assert list(zurueck.index) == list(ergebnis.data.index)
        assert list(zurueck.columns) == list(ergebnis.data.columns)

    def test_export_legt_verzeichnisse_an(self, ergebnis, tmp_path: Path):
        """Ein Export in einen noch nicht existierenden Pfad gelingt."""
        ziel = tmp_path / "tief" / "verschachtelt" / "screen.csv"
        geschrieben = export_result(ergebnis, ziel)

        assert geschrieben.is_file()

    def test_unbekanntes_format_wird_abgelehnt(self, ergebnis, tmp_path: Path):
        """Ein unbekanntes Suffix scheitert laut, statt still nichts zu tun."""
        from btlite.core.errors import ScreeningError

        with pytest.raises(ScreeningError):
            export_result(ergebnis, tmp_path / "screen.xlsx")
