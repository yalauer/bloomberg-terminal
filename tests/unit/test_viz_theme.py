"""Tests des zentralen Themes.

Prueft die Garantien, auf die alle Chart-Factories bauen: registriertes
Template, schwarzer Grund, Monospace, farbenblindsichere Richtungsfarben und
die Hinweis-Figure statt eines leeren Rahmens.
"""

from __future__ import annotations

import math
import re

import plotly.graph_objects as go
import plotly.io as pio
import pytest

from btlite.viz import theme


class TestTemplate:
    def test_template_is_registered(self):
        assert theme.TEMPLATE_NAME in pio.templates

    def test_register_is_idempotent(self):
        first = theme.register_template()
        second = theme.register_template()
        assert first == second == theme.TEMPLATE_NAME
        assert theme.TEMPLATE_NAME in pio.templates

    def test_import_does_not_hijack_global_default(self):
        # Das Modul darf den globalen Default nicht umbiegen, sonst
        # veraendert ein Import fremde Figures.
        assert pio.templates.default != theme.TEMPLATE_NAME

    def test_surface_is_black(self):
        template = pio.templates[theme.TEMPLATE_NAME]
        assert template.layout.plot_bgcolor == theme.SURFACE
        assert template.layout.paper_bgcolor == theme.PAPER

    def test_font_is_monospace(self):
        template = pio.templates[theme.TEMPLATE_NAME]
        assert "mono" in template.layout.font.family.lower()

    def test_colorway_matches_categorical(self):
        template = pio.templates[theme.TEMPLATE_NAME]
        assert tuple(template.layout.colorway) == theme.CATEGORICAL

    def test_apply_theme_sets_template_and_title(self):
        fig = go.Figure()
        theme.apply_theme(fig, title="TEST")
        assert fig.layout.template.layout.plot_bgcolor == theme.SURFACE
        assert fig.layout.title.text == "TEST"


class TestColors:
    def test_up_and_down_are_the_chosen_pair(self):
        """Steigend gruen, fallend rot - aber gemessen, nicht gegriffen.

        Die Farben stehen hier als Literal, damit ein spaeterer Griff zu
        einem "schoeneren" Gruen nicht unbemerkt die Messung unterlaeuft,
        die :func:`test_red_green_pair_is_measured_not_assumed` absichert.
        """
        assert theme.UP.upper() == "#00E676"
        assert theme.DOWN.upper() == "#E03131"

    def test_up_down_separable_under_colour_blindness(self):
        """Nachgemessen: das Paar muss auch farbenblind trennbar sein.

        Delta E in OKLab mal 100, CVD-Simulation nach
        Machado-Oliveira-Fernandes 2009 bei Severity 1.0. Der Zielwert der
        Methode ist 8.

        Dass ein Gruen/Rot-Paar diese Huerde nimmt, ist nicht
        selbstverstaendlich - ein typisches faellt bei 6.5 durch. Getragen
        wird sie hier vom Helligkeitsunterschied: helles Gruen, dunkles Rot.
        """
        for kind in ("protan", "deutan"):
            assert _delta_e(theme.UP, theme.DOWN, kind) >= 8.0

    def test_fill_colours_match_their_solid_colour(self):
        """Die halbtransparenten Toene sind dieselbe Farbe, nur gedaempft.

        Ohne diese Pruefung koennte ein Farbwechsel die Volltonfarbe
        umstellen und die Fuellung stehen lassen: Kerzenraender gruen,
        Kerzenkoerper blau. Das faellt in keinem Kontrasttest auf, weil
        beide Werte fuer sich genommen gueltig sind.
        """
        for solid, fill in ((theme.UP, theme.UP_FILL), (theme.DOWN, theme.DOWN_FILL)):
            expected = tuple(int(solid.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
            found = tuple(int(x) for x in re.findall(r"\d+", fill)[:3])
            assert found == expected, f"{fill} gehoert nicht zu {solid}"

    def test_fill_colours_are_translucent(self):
        """Eine Fuellung ohne Alpha waere keine Fuellung."""
        for fill in (theme.UP_FILL, theme.DOWN_FILL):
            alpha = float(fill.rstrip(")").split(",")[-1])
            assert 0.0 < alpha < 1.0, f"{fill} ist nicht halbtransparent"

    def test_direction_colors_readable_on_black(self):
        for color in (theme.UP, theme.DOWN, theme.AMBER, theme.CYAN, theme.INK):
            assert _contrast(color, theme.SURFACE) >= 3.0

    def test_categorical_has_eight_fixed_slots(self):
        assert len(theme.CATEGORICAL) == 8
        assert len(set(theme.CATEGORICAL)) == 8

    def test_series_color_follows_fixed_order(self):
        assert theme.series_color(0) == theme.CATEGORICAL[0]
        assert theme.series_color(2) == theme.CATEGORICAL[2]

    def test_series_color_wraps_instead_of_inventing_hues(self):
        assert theme.series_color(8) == theme.CATEGORICAL[0]

    def test_scatter_cap_slots_separable_for_all_pairs(self):
        """Streudiagramme: alle Paare der ersten drei Slots muessen trennen."""
        slots = theme.CATEGORICAL[: theme.CATEGORICAL_SCATTER_CAP]
        for i in range(len(slots)):
            for j in range(i + 1, len(slots)):
                for kind in ("protan", "deutan"):
                    assert _delta_e(slots[i], slots[j], kind) >= 8.0

    def test_sequential_scale_is_monotone_in_lightness(self):
        lightness = [_oklab(c)[0] for c in theme.SEQUENTIAL]
        assert lightness == sorted(lightness)

    def test_diverging_scale_has_neutral_midpoint(self):
        scale = theme.diverging_scale()
        assert scale[0][0] == 0.0
        assert scale[-1][0] == 1.0
        midpoint = theme.DIVERGING[len(theme.DIVERGING) // 2]
        # Der Mittelpunkt muss grau sein, sonst liest sich "keine
        # Veraenderung" als eigene Kategorie.
        _, chroma = _oklch(midpoint)
        assert chroma < 0.02


class TestDivergingPolarity:
    """Die Rampe fuer Korrelation und Monatsrenditen.

    Sie traegt dieselbe Aussage wie die Kerzen - positiv gegen negativ -
    und muss deshalb dieselben Farben und dieselbe Trennschaerfe haben.
    """

    def test_the_poles_are_the_direction_colours(self):
        """Ein positiver Monat hat die Farbe einer steigenden Kerze.

        Ohne diese Bindung koennten beide Enden auseinanderlaufen, und
        dieselbe Aussage traege im selben Terminal zwei Farben.
        """
        assert theme.UP in theme.DIVERGING, "Das Gruen der Rampe ist nicht theme.UP"
        red_side = theme.DIVERGING[: len(theme.DIVERGING) // 2]
        green_side = theme.DIVERGING[len(theme.DIVERGING) // 2 + 1 :]
        # Die rote Seite liegt nahe an DOWN, die gruene enthaelt UP selbst.
        assert min(_delta_e(c, theme.DOWN) for c in red_side) < 8.0
        assert theme.UP in green_side

    def test_lightness_rises_from_red_to_green(self):
        """Der eigentliche Mechanismus: Helligkeit statt Farbton.

        Zwei gleich dunkle Enden - dunkles Rot gegen dunkles Gruen -
        faellen bei Farbenblindheit zusammen; gemessen lag eine solche
        Variante bei 6.2 und damit unter der Schwelle. Steigt die
        Helligkeit durchgehend, bleibt die Richtung auch ohne
        Farbwahrnehmung ablesbar.
        """
        half = len(theme.DIVERGING) // 2
        red_side = [_oklab(c)[0] for c in theme.DIVERGING[:half]]
        green_side = [_oklab(c)[0] for c in theme.DIVERGING[half + 1 :]]
        assert red_side == sorted(red_side), "Rote Seite nicht monoton"
        assert green_side == sorted(green_side), "Gruene Seite nicht monoton"
        assert min(green_side) > max(red_side), (
            "Die gruene Seite muss durchgehend heller sein als die rote"
        )

    def test_the_poles_stay_separable_under_colour_blindness(self):
        """Die beiden Enden duerfen nicht zusammenfallen."""
        for kind in ("protan", "deutan"):
            assert _delta_e(theme.DIVERGING[0], theme.DIVERGING[-1], kind) >= 8.0

    def test_neighbouring_steps_are_distinguishable(self):
        """Sonst liest sich die Rampe als Stufe, wo keine ist."""
        for first, second in zip(theme.DIVERGING, theme.DIVERGING[1:], strict=False):
            for kind in ("protan", "deutan"):
                assert _delta_e(first, second, kind) >= 4.0, f"{first} -> {second}"

    def test_the_ramp_is_symmetric_in_length(self):
        """Gleich viele Stufen je Seite, sonst liegt die Null nicht mittig."""
        assert len(theme.DIVERGING) % 2 == 1
        half = len(theme.DIVERGING) // 2
        assert half == len(theme.DIVERGING) - half - 1


class TestScales:
    def test_sequential_scale_positions_span_unit_interval(self):
        scale = theme.sequential_scale()
        assert scale[0][0] == 0.0
        assert scale[-1][0] == 1.0
        assert all(0.0 <= pos <= 1.0 for pos, _ in scale)

    def test_scales_are_ordered(self):
        for scale in (theme.sequential_scale(), theme.diverging_scale()):
            positions = [pos for pos, _ in scale]
            assert positions == sorted(positions)


class TestFormatValue:
    def test_none_renders_as_na_not_zero(self):
        assert theme.format_value(None) == theme.NA_TEXT
        assert "0" not in theme.format_value(None)

    def test_nan_renders_as_na(self):
        assert theme.format_value(float("nan")) == theme.NA_TEXT

    def test_zero_is_not_confused_with_missing(self):
        assert theme.format_value(0.0) != theme.NA_TEXT

    def test_formats_with_unit(self):
        assert theme.format_value(1234.5, unit="USD") == "1,234.50 USD"


class TestEmptyFigure:
    def test_returns_figure(self):
        assert isinstance(theme.empty_figure(), go.Figure)

    def test_has_centered_hint_instead_of_empty_frame(self):
        fig = theme.empty_figure("Keine Daten")
        assert len(fig.data) == 0
        assert len(fig.layout.annotations) == 1
        annotation = fig.layout.annotations[0]
        assert annotation.text == "Keine Daten"
        assert annotation.x == 0.5
        assert annotation.y == 0.5

    def test_hides_axes(self):
        fig = theme.empty_figure()
        assert fig.layout.xaxis.visible is False
        assert fig.layout.yaxis.visible is False

    def test_applies_theme(self):
        fig = theme.empty_figure()
        assert fig.layout.template.layout.plot_bgcolor == theme.SURFACE

    def test_keeps_title(self):
        fig = theme.empty_figure("Nichts", title="AAPL")
        assert fig.layout.title.text == "AAPL"


class TestHelpers:
    def test_hover_template_carries_unit(self):
        result = theme.hover_template(label="Close", value_format=".2f", unit="USD")
        assert "Close" in result
        assert "USD" in result
        assert "<extra></extra>" in result

    def test_crosshair_enables_spikes(self):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[1, 2]))
        theme.add_crosshair(fig)
        assert fig.layout.hovermode == "x unified"
        assert fig.layout.xaxis.showspikes is True

    def test_range_selector_has_all_six_buttons(self):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[1, 2]))
        theme.add_range_selector(fig)
        labels = [b.label for b in fig.layout.xaxis.rangeselector.buttons]
        assert labels == ["1M", "3M", "6M", "1Y", "5Y", "MAX"]

    def test_style_axes_sets_titles_and_log(self):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[1, 2]))
        theme.style_axes(fig, x_title="Datum", y_title="Preis", log_y=True)
        assert fig.layout.xaxis.title.text == "Datum"
        assert fig.layout.yaxis.title.text == "Preis"
        assert fig.layout.yaxis.type == "log"


# --------------------------------------------------------------------------
# Farbmathematik fuer die Tests - bewusst hier und nicht in viz/,
# damit die Produktionsschicht keine Rechenlogik traegt.
# --------------------------------------------------------------------------

_MACHADO = {
    "protan": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deutan": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
}


def _linear(hex_color: str) -> tuple[float, float, float]:
    raw = hex_color.lstrip("#")
    channels = [int(raw[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    out = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return out[0], out[1], out[2]


def _oklab_from_linear(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    r, g, b = rgb
    l_ = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m_ = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s_ = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def _oklab(hex_color: str) -> tuple[float, float, float]:
    return _oklab_from_linear(_linear(hex_color))


def _oklch(hex_color: str) -> tuple[float, float]:
    lightness, a, b = _oklab(hex_color)
    return lightness, math.hypot(a, b)


def _simulate(hex_color: str, kind: str) -> tuple[float, float, float]:
    r, g, b = _linear(hex_color)
    matrix = _MACHADO[kind]
    out = [min(1.0, max(0.0, row[0] * r + row[1] * g + row[2] * b)) for row in matrix]
    return out[0], out[1], out[2]


def _delta_e(first: str, second: str, kind: str | None = None) -> float:
    a = _oklab_from_linear(_simulate(first, kind) if kind else _linear(first))
    b = _oklab_from_linear(_simulate(second, kind) if kind else _linear(second))
    return 100 * math.dist(a, b)


def _relative_luminance(hex_color: str) -> float:
    r, g, b = _linear(hex_color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(first: str, second: str) -> float:
    high, low = sorted([_relative_luminance(first), _relative_luminance(second)], reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_red_green_pair_is_measured_not_assumed():
    """Belegt, dass die Farbwahl eine Messung war und keine Gewohnheit.

    Ein *typisches* Gruen/Rot faellt durch das Gate - genau deshalb galt
    die Kodierung lange als unbrauchbar. Das hier gewaehlte Paar besteht
    es trotzdem, weil Helligkeit statt Farbton die Unterscheidung traegt.

    Der Test haelt beide Aussagen zusammen: faellt die zweite, war der
    letzte Farbwechsel ein Rueckschritt und faellt hier auf.
    """
    typical_green, typical_red = "#199E70", "#E66767"
    worst = min(_delta_e(typical_green, typical_red, k) for k in ("protan", "deutan"))
    assert worst < 8.0, "Die Referenz muss durchfallen, sonst prueft der Vergleich nichts"

    chosen = min(_delta_e(theme.UP, theme.DOWN, k) for k in ("protan", "deutan"))
    assert chosen >= 8.0, f"Das gewaehlte Paar faellt durch das Gate: {chosen:.1f}"
    assert chosen > 2 * worst, (
        f"Der Abstand zur untauglichen Referenz ist zu knapp: {chosen:.1f} gegen {worst:.1f}"
    )


def test_up_is_the_lighter_of_the_two():
    """Die Helligkeit traegt die Unterscheidung, nicht der Farbton.

    Das ist der Grund, warum dieses Gruen/Rot funktioniert und ein
    dunkleres Gruen nicht. Waehlt jemand spaeter einen satteren Ton,
    faellt genau diese Eigenschaft als erstes weg.
    """
    up_lightness = _oklab(theme.UP)[0]
    down_lightness = _oklab(theme.DOWN)[0]
    assert up_lightness > down_lightness + 0.2, (
        f"Zu wenig Helligkeitsabstand: {up_lightness:.2f} gegen {down_lightness:.2f}"
    )


def test_a_darker_green_would_fail():
    """Gegenprobe: ein naheliegendes, satteres Gruen besteht das Gate nicht.

    ``#16A34A`` ist ein verbreitetes "Boersengruen" und kommt gegen
    dasselbe Rot nur auf 4.5 - es waere genau der Griff, den dieses Paar
    vermeidet. Haelt fest, dass die Wahl eng ist: ohne diesen Test saehe
    die bestandene Pruefung nach Zufall aus.

    ``#1DB954`` liegt mit 8.4 knapp darueber und steht deshalb nicht hier;
    ein Test, der zu viel behauptet, faellt beim naechsten Messwert um.
    """
    tempting = "#16A34A"
    worst = min(_delta_e(tempting, theme.DOWN, k) for k in ("protan", "deutan"))
    assert worst < 8.0, f"{tempting} besteht wider Erwarten: {worst:.1f}"

    chosen = min(_delta_e(theme.UP, theme.DOWN, k) for k in ("protan", "deutan"))
    assert chosen > worst * 2, "Der Vorsprung des gewaehlten Gruens ist zu klein"


@pytest.mark.parametrize("color", theme.CATEGORICAL)
def test_categorical_colors_have_enough_chroma(color):
    """Unterhalb der Chroma-Schwelle liest sich eine Farbe als Grau."""
    _, chroma = _oklch(color)
    assert chroma >= 0.10
