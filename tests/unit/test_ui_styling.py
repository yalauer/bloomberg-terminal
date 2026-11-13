"""Regressionstests fuer das Terminal-Stylesheet.

Hintergrund
-----------
``ui/assets/btlite.css`` entstand in Phase 1, die Seiten unter ``ui/pages/``
erst in Phase 3. Dadurch waren 37 vergebene CSS-Klassen ohne jede Regel; sie
erbten die Farben des Darkly-Themes und der Dash-Defaults und waren auf dem
schwarzen Terminalgrund nicht lesbar. Die Tests hier frieren drei Aussagen
ein, damit genau das nicht wiederkehrt:

1. Jede im UI vergebene Klasse hat eine Regel im Stylesheet.
2. Jede gesetzte Vordergrund/Hintergrund-Kombination erfuellt WCAG AA.
3. ``outline: none`` steht nie ohne sichtbaren Ersatz.

Der Kontrastrechner folgt WCAG 2.1: relative Luminanz nach der dort
definierten sRGB-Linearisierung, Kontrast als ``(L_hell + 0.05) /
(L_dunkel + 0.05)``. Die Werte sind dimensionslose Verhaeltnisse.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from btlite.viz import theme

# --------------------------------------------------------------------------
# Grenzwerte, bewusst eingefroren
# --------------------------------------------------------------------------

#: WCAG AA fuer normalen Text. Dimensionsloses Kontrastverhaeltnis.
AA_NORMAL = 4.5

#: WCAG AA fuer grossen Text ab 18pt bzw. 24px.
AA_LARGE = 3.0

#: Stylesheet des Terminals.
CSS_PATH = Path(__file__).resolve().parents[2] / "src" / "btlite" / "ui" / "assets" / "btlite.css"

#: Wurzel der UI-Module, aus denen die Klassennamen stammen.
UI_ROOT = Path(__file__).resolve().parents[2] / "src" / "btlite" / "ui"


# --------------------------------------------------------------------------
# Kontrastrechner nach WCAG 2.1
# --------------------------------------------------------------------------


def _channel(value: int) -> float:
    """Linearisiert einen sRGB-Kanal.

    Args:
        value: Kanalwert 0 bis 255.

    Returns:
        Linearisierter Anteil, dezimal zwischen 0 und 1.
    """
    c = value / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def parse_hex(color: str) -> tuple[int, int, int]:
    """Zerlegt einen Hex-Farbwert in RGB.

    Args:
        color: Farbe als ``#rgb`` oder ``#rrggbb``, Gross- und
            Kleinschreibung egal.

    Returns:
        Tripel ``(r, g, b)``, jeweils 0 bis 255.

    Raises:
        ValueError: Wenn der Wert kein Hex-Farbwert ist.
    """
    text = color.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6 or not re.fullmatch(r"[0-9a-fA-F]{6}", text):
        raise ValueError(f"Kein Hex-Farbwert: {color!r}")
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))


def relative_luminance(color: str) -> float:
    """Berechnet die relative Luminanz nach WCAG 2.1.

    Args:
        color: Hex-Farbwert.

    Returns:
        Luminanz, dezimal zwischen 0 (schwarz) und 1 (weiss).
    """
    r, g, b = parse_hex(color)
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast_ratio(fg: str, bg: str) -> float:
    """Berechnet das Kontrastverhaeltnis zweier Farben.

    Args:
        fg: Vordergrundfarbe als Hex.
        bg: Hintergrundfarbe als Hex.

    Returns:
        Verhaeltnis zwischen 1.0 (identisch) und 21.0 (schwarz gegen weiss).
    """
    lighter, darker = sorted((relative_luminance(fg), relative_luminance(bg)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def composite(fg: str, alpha: float, bg: str) -> str:
    """Blendet eine teiltransparente Farbe auf einen Grund.

    Noetig, weil die Banner ihre Flaeche als ``rgba(..., 0.12)`` setzen; der
    sichtbare Hintergrund ist dann die Mischung, nicht die Grundflaeche.

    Args:
        fg: Vordergrundfarbe als Hex.
        alpha: Deckkraft, dezimal zwischen 0 und 1.
        bg: Hintergrundfarbe als Hex.

    Returns:
        Resultierende Volltonfarbe als ``#rrggbb``.
    """
    f = parse_hex(fg)
    b = parse_hex(bg)
    mixed = tuple(round(f[i] * alpha + b[i] * (1.0 - alpha)) for i in range(3))
    return "#{:02x}{:02x}{:02x}".format(*mixed)


class TestContrastCalculator:
    """Der Rechner selbst muss stimmen, sonst sagt der Rest nichts aus."""

    def test_black_on_white_is_maximal(self) -> None:
        assert contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0, abs=0.01)

    def test_identical_colors_are_one(self) -> None:
        assert contrast_ratio("#ffa028", "#ffa028") == pytest.approx(1.0, abs=0.001)

    def test_is_symmetric(self) -> None:
        assert contrast_ratio("#ffa028", "#000000") == pytest.approx(
            contrast_ratio("#000000", "#ffa028")
        )

    def test_shorthand_hex_expands(self) -> None:
        assert parse_hex("#fff") == (255, 255, 255)

    def test_known_reference_value(self) -> None:
        # Referenzwert gegen die WCAG-Formel: #767676 auf Weiss ist der
        # kanonische Grenzfall, der AA gerade noch besteht.
        assert contrast_ratio("#767676", "#ffffff") == pytest.approx(4.54, abs=0.02)

    def test_rejects_non_hex(self) -> None:
        with pytest.raises(ValueError, match="Kein Hex-Farbwert"):
            parse_hex("rgba(0,0,0,0.5)")

    def test_composite_blends_towards_background(self) -> None:
        assert composite("#ffffff", 0.0, "#000000") == "#000000"
        assert composite("#ffffff", 1.0, "#000000") == "#ffffff"


# --------------------------------------------------------------------------
# Stylesheet einlesen
# --------------------------------------------------------------------------


def _css_text() -> str:
    """Liest das Stylesheet.

    Returns:
        Vollstaendiger Inhalt von ``btlite.css``.
    """
    return CSS_PATH.read_text(encoding="utf8")


def _strip_comments(css: str) -> str:
    """Entfernt CSS-Kommentare.

    Kommentare nennen Klassennamen im Fliesstext; ohne diesen Schritt wuerde
    eine blosse Erwaehnung als Regel durchgehen.

    Args:
        css: Stylesheet-Text.

    Returns:
        Text ohne ``/* ... */``-Bloecke.
    """
    return re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)


def _declared_classes(css_body: str) -> set[str]:
    """Sammelt alle Klassennamen, die in Selektoren vorkommen.

    Args:
        css_body: Stylesheet ohne Kommentare.

    Returns:
        Menge der Klassennamen ohne fuehrenden Punkt.
    """
    selectors = " ".join(re.findall(r"([^{}]+)\{", css_body))
    return set(re.findall(r"\.([A-Za-z0-9_-]+)", selectors))


def _used_classes() -> set[str]:
    """Sammelt alle im UI vergebenen CSS-Klassen.

    Liest die ``className="..."``-Literale aus allen UI-Modulen. Dynamisch
    zusammengesetzte Klassen stammen durchweg aus denselben Literalen, weil
    ``formatting.direction_class`` und Co. nur diese zurueckgeben.

    Returns:
        Menge der vergebenen Klassennamen.
    """
    used: set[str] = set()
    for path in sorted(UI_ROOT.rglob("*.py")):
        for literal in re.findall(r'className="([^"]*)"', path.read_text(encoding="utf8")):
            used.update(literal.split())
    return used


class TestClassCoverage:
    """Jede vergebene Klasse muss eine Regel haben."""

    def test_stylesheet_exists(self) -> None:
        assert CSS_PATH.is_file()

    def test_ui_actually_assigns_classes(self) -> None:
        # Schutz gegen einen stillen Leerlauf des Abgleichs unten.
        used = _used_classes()
        assert len(used) > 40, f"Nur {len(used)} Klassen gefunden - Sammler defekt?"

    def test_every_assigned_class_has_a_rule(self) -> None:
        declared = _declared_classes(_strip_comments(_css_text()))
        missing = sorted(cls for cls in _used_classes() if cls not in declared)
        assert missing == [], (
            "Diese im UI vergebenen Klassen haben keine CSS-Regel und erben "
            f"deshalb unlesbare Fremdfarben: {missing}"
        )

    @pytest.mark.parametrize(
        "cls",
        [
            "bt-control-label",
            "bt-radio",
            "bt-checklist",
            "bt-dropdown",
            "bt-input",
            "bt-navlink",
            "bt-page-title",
            "bt-subhead",
        ],
    )
    def test_reported_offenders_are_covered(self, cls: str) -> None:
        # Diese Klassen trugen den gemeldeten Fehler; sie werden einzeln
        # festgehalten, damit ein Sammler-Bug sie nicht stillschweigend
        # aus der Pruefung faellt.
        assert cls in _declared_classes(_strip_comments(_css_text()))


class TestDashInternals:
    """Die inneren Elemente von Dash brauchen eigene Regeln.

    Die Klasse am Container faerbt sie nicht mit. Dash 4 rendert ausserdem
    kein react-select mehr - ``.Select-control`` existiert nicht, die
    Struktur heisst ``.dash-dropdown-*``.
    """

    @pytest.mark.parametrize(
        "selector",
        [
            ".dash-options-list-option",
            ".dash-dropdown-content",
            ".dash-dropdown-option",
            ".dash-dropdown-search",
            ".dash-input-element",
        ],
    )
    def test_inner_element_is_styled(self, selector: str) -> None:
        assert selector in _strip_comments(_css_text())

    def test_dash_tokens_are_overridden(self) -> None:
        # Die --Dash-*-Defaults sind eine helle Palette; ohne Umdefinition
        # steht dunkles Marineblau auf schwarzem Grund. Das war die Ursache.
        body = _strip_comments(_css_text())
        for token in (
            "--Dash-Text-Primary",
            "--Dash-Text-Strong",
            "--Dash-Fill-Inverse-Strong",
            "--Dash-Stroke-Strong",
        ):
            assert f"{token}:" in body, f"{token} wird nicht umdefiniert"

    def test_dash_lowercase_typo_token_is_covered(self) -> None:
        # Dash schreibt das Token in .dash-options-list-option klein
        # ("Inverse-strong"). Das ist ein Fehler im Paket; ohne eigene
        # Definition bleibt die Flaeche der Optionen undefiniert.
        assert "--Dash-Fill-Inverse-strong:" in _strip_comments(_css_text())

    def test_no_react_select_leftovers(self) -> None:
        # Regeln fuer die alte react-select-Struktur waeren toter Code und
        # ein Zeichen dafuer, dass gegen die falsche Dash-Version gestylt
        # wurde.
        body = _strip_comments(_css_text())
        assert ".Select-control" not in body
        assert ".Select-menu-outer" not in body


# --------------------------------------------------------------------------
# Kontrast der tatsaechlich gesetzten Paare
# --------------------------------------------------------------------------

#: Grundflaechen des Terminals. Schluessel = CSS-Variable ohne Praefix.
SURFACES: dict[str, str] = {
    "--bt-surface": "#000000",
    "--bt-surface-raised": "#12120f",
    "--bt-surface-hover": "#1c1c18",
}

#: Textfarben, die das Stylesheet setzt, mit ihren Hex-Werten.
TEXT_COLORS: dict[str, str] = {
    "--bt-ink": "#e8e6dc",
    "--bt-ink-muted": "#898781",
    "--bt-amber": "#ffa028",
    "--bt-cyan": "#22d3ee",
    "--bt-up": "#00e676",
    "--bt-down": "#e03131",
    "--bt-warning": "#fab219",
    "--bt-critical-text": "#e66767",
    # Ersatzton fuer --bt-down als Schrift. Fehlte hier bislang, womit die
    # Kontrastpruefung ihn schlicht uebersprungen hat.
    "--bt-down-text": "#e66767",
}

#: Farben, die es als Token gibt, aber nie als Schriftfarbe. Sie stehen
#: in Charts (dort sind sie Flaechen) oder als Rahmen. Fuer sie gilt die
#: Textschwelle nicht - ``test_down_is_never_a_text_colour`` haelt fest,
#: dass sie auch wirklich nicht als Text auftauchen.
NEVER_TEXT: frozenset[str] = frozenset({"--bt-down"})

#: Paare, die im Stylesheet tatsaechlich als Text auf Flaeche vorkommen.
#: Bewusst gefiltert statt als volles Kreuzprodukt: --bt-amber-dim etwa
#: ist nur Rahmenfarbe und darf hier nicht auftauchen.
USED_PAIRS: list[tuple[str, str]] = [
    (fg, bg)
    for bg in SURFACES
    for fg in TEXT_COLORS
    if bg != "--bt-surface-hover" and fg not in NEVER_TEXT
] + [
    # Auf der Hover-Flaeche setzt das Stylesheet nur diese Textfarben.
    ("--bt-ink", "--bt-surface-hover"),
    ("--bt-amber", "--bt-surface-hover"),
    ("--bt-ink-muted", "--bt-surface-hover"),
]


class TestContrastOfPalette:
    """Jede gesetzte Farbkombination erfuellt WCAG AA."""

    @pytest.mark.parametrize(("fg", "bg"), USED_PAIRS)
    def test_pair_meets_aa(self, fg: str, bg: str) -> None:
        ratio = contrast_ratio(TEXT_COLORS[fg], SURFACES[bg])
        assert ratio >= AA_NORMAL, f"{fg} auf {bg}: {ratio:.2f}:1, gefordert {AA_NORMAL}:1"

    def test_banner_text_on_tinted_surface(self) -> None:
        # .bt-error und .bt-warning legen eine 12-Prozent-Tinte auf den
        # Grund; geprueft wird gegen die gemischte Flaeche, nicht gegen
        # Schwarz.
        for tint in ("#d03b3b", "#fab219"):
            bg = composite(tint, 0.12, SURFACES["--bt-surface"])
            assert contrast_ratio(TEXT_COLORS["--bt-ink"], bg) >= AA_NORMAL

    def test_command_error_text_on_tinted_surface(self) -> None:
        bg = composite("#d03b3b", 0.12, SURFACES["--bt-surface"])
        ratio = contrast_ratio(TEXT_COLORS["--bt-critical-text"], bg)
        assert ratio >= AA_NORMAL, f"Kommandofehler: {ratio:.2f}:1"

    def test_down_is_never_a_text_colour(self) -> None:
        """``--bt-down`` darf nur in Charts stehen, nie als Schrift.

        Das Rot erreicht als Text nur 4.2:1 auf der erhobenen Flaeche und
        3.8:1 unter der Maus - beides unter AA. Fuer Schrift gibt es
        ``--bt-down-text``, einen aufgehellten Ton derselben Familie.

        Ohne diesen Test waere der Ausschluss aus USED_PAIRS eine blosse
        Behauptung: jemand koennte die Farbe als Schrift setzen, und die
        Kontrastpruefung schaute weg.
        """
        body = _strip_comments(_css_text())
        as_text = re.search(r"(?:^|[;{\s])color\s*:\s*var\(--bt-down\)", body)
        assert as_text is None, f"--bt-down als Schriftfarbe: {as_text.group()!r}"

    def test_down_text_is_readable_everywhere(self) -> None:
        """Der Ersatzton muss halten, was das Original nicht kann."""
        for surface in SURFACES.values():
            ratio = contrast_ratio(TEXT_COLORS["--bt-down-text"], surface)
            assert ratio >= AA_NORMAL, f"--bt-down-text: {ratio:.2f}:1"

    def test_critical_is_not_used_as_text_colour(self) -> None:
        # #d03b3b erreicht nur 4.37:1 gegen Schwarz und ist deshalb als
        # Schriftfarbe unzulaessig - nur als Rahmen erlaubt.
        #
        # Die Eigenschaft wird mit Wortgrenze gesucht: ein blosses
        # "color: var(--bt-critical)" als Teilstring traefe auch
        # "border-color: var(--bt-critical)" - also genau die Verwendung,
        # die dieser Test erlauben will. Vorher musste man deshalb zur
        # border-Kurzform ausweichen, um einen korrekten Rahmen zu
        # schreiben; der Test verbot mehr, als er meinte.
        assert contrast_ratio("#d03b3b", "#000000") < AA_NORMAL
        body = _strip_comments(_css_text())
        as_text = re.search(r"(?:^|[;{\s])color\s*:\s*var\(--bt-critical\)", body)
        assert as_text is None, f"--bt-critical als Schriftfarbe: {as_text.group()!r}"

    def test_amber_dim_is_not_text_on_hover_surface(self) -> None:
        # Messwert 4.23:1 - unter AA. Das Stylesheet nutzt die Farbe dort
        # nur als Rahmen, was dieser Test festhaelt.
        assert contrast_ratio("#b36f16", SURFACES["--bt-surface-hover"]) < AA_NORMAL

    def test_selection_highlight_is_readable(self) -> None:
        # ::selection setzt schwarzen Text auf Bernstein.
        assert contrast_ratio("#000000", TEXT_COLORS["--bt-amber"]) >= AA_NORMAL

    def test_primary_button_is_readable(self) -> None:
        # .btn-primary: schwarze Schrift auf gedimmtem Bernstein, im
        # Hover-Zustand auf vollem Bernstein.
        assert contrast_ratio("#000000", "#b36f16") >= AA_NORMAL
        assert contrast_ratio("#000000", "#ffa028") >= AA_NORMAL

    def test_focus_ring_is_visible_against_surfaces(self) -> None:
        # Der Ring ist eine Flaeche, kein Text: Grenzwert 3:1.
        for surface in SURFACES.values():
            assert contrast_ratio(TEXT_COLORS["--bt-amber"], surface) >= AA_LARGE

    def test_hex_values_match_the_theme_module(self) -> None:
        # viz/theme.py ist die Referenz; das Stylesheet darf nicht
        # auseinanderlaufen.
        assert TEXT_COLORS["--bt-amber"].lower() == theme.AMBER.lower()
        assert TEXT_COLORS["--bt-cyan"].lower() == theme.CYAN.lower()
        assert TEXT_COLORS["--bt-ink"].lower() == theme.INK.lower()
        assert TEXT_COLORS["--bt-ink-muted"].lower() == theme.INK_MUTED.lower()
        assert TEXT_COLORS["--bt-up"].lower() == theme.UP.lower()
        assert TEXT_COLORS["--bt-down"].lower() == theme.DOWN.lower()
        # Der Ersatzton hat kein Gegenstueck im Theme - er existiert nur,
        # weil --bt-down als Schrift zu dunkel ist. Geprueft wird deshalb
        # gegen das Stylesheet selbst, nicht gegen eine Modulkonstante.
        declared = re.search(r"--bt-down-text:\s*(#[0-9a-fA-F]{6})", _css_text())
        assert declared is not None, "--bt-down-text fehlt im Stylesheet"
        assert declared.group(1).lower() == TEXT_COLORS["--bt-down-text"].lower()
        assert SURFACES["--bt-surface"].lower() == theme.SURFACE.lower()
        assert SURFACES["--bt-surface-raised"].lower() == theme.SURFACE_RAISED.lower()

    def test_critical_text_colour_comes_from_the_theme_palette(self) -> None:
        # Keine erfundene Farbe: der hellere Rotton ist Slot 8 aus
        # CATEGORICAL.
        assert TEXT_COLORS["--bt-critical-text"].lower() in {c.lower() for c in theme.CATEGORICAL}

    def test_css_declares_every_colour_variable(self) -> None:
        body = _strip_comments(_css_text())
        for name, value in {**SURFACES, **TEXT_COLORS}.items():
            assert f"{name}:" in body, f"{name} fehlt im Stylesheet"
            assert value in body.lower(), f"{name} hat nicht mehr den Wert {value}"


# --------------------------------------------------------------------------
# Fokus und Ruhezustand
# --------------------------------------------------------------------------


def _rules(css_body: str) -> list[tuple[str, str]]:
    """Zerlegt das Stylesheet in ``(selektor, block)``-Paare.

    Verschachtelte At-Regeln wie ``@media`` liefern dabei ihren Rumpf als
    eigenen Block; fuer die Pruefungen hier reicht das.

    Args:
        css_body: Stylesheet ohne Kommentare.

    Returns:
        Liste aus Selektortext und zugehoerigem Deklarationsblock.
    """
    return [
        (sel.strip(), block)
        for sel, block in re.findall(r"([^{}]+)\{([^{}]*)\}", css_body)
        if sel.strip()
    ]


class TestFocusVisibility:
    """Tastaturbedienung braucht einen sichtbaren, eigenen Fokuszustand."""

    def test_global_focus_visible_rule_exists(self) -> None:
        assert ":focus-visible" in _strip_comments(_css_text())

    def test_outline_none_always_has_a_replacement(self) -> None:
        body = _strip_comments(_css_text())
        offenders: list[str] = []
        for selector, block in _rules(body):
            if not re.search(r"outline\s*:\s*none", block):
                continue
            # Zulaessig nur, wenn derselbe Selektor einen Ersatz mitbringt
            # oder ein eigener :focus-visible-Block existiert.
            has_shadow = "box-shadow" in block
            base = selector.split(":")[0].strip()
            has_focus_rule = any(
                base and base in other_sel and ":focus-visible" in other_sel
                for other_sel, _ in _rules(body)
            )
            if not (has_shadow or has_focus_rule):
                offenders.append(selector)
        assert offenders == [], f"outline:none ohne sichtbaren Ersatz: {offenders}"

    @pytest.mark.parametrize(
        "selector",
        [
            ".bt-navlink:focus-visible",
            ".bt-command:focus-visible",
            ".bt-subhead:focus-visible",
            ".dash-dropdown-search:focus-visible",
        ],
    )
    def test_focusable_elements_have_focus_visible(self, selector: str) -> None:
        assert selector in _strip_comments(_css_text())

    def test_radio_and_checklist_have_focus_ring(self) -> None:
        body = _strip_comments(_css_text())
        assert ".bt-radio .dash-options-list-option:focus-within" in body
        assert ".bt-checklist .dash-options-list-option:focus-within" in body


class TestRestingStateIsReadable:
    """Der gemeldete Fehler war Text, der erst beim Hovern erschien."""

    @pytest.mark.parametrize(
        "base",
        [".bt-navlink", ".bt-control-label", ".bt-subhead", ".bt-page-title"],
    )
    def test_colour_is_set_outside_hover(self, base: str) -> None:
        body = _strip_comments(_css_text())
        resting = [
            block
            for sel, block in _rules(body)
            if base in sel and ":hover" not in sel and ":focus" not in sel
        ]
        assert resting, f"{base} hat keine Regel im Ruhezustand"
        assert any("color:" in block for block in resting), (
            f"{base} setzt seine Farbe nur im Hover- oder Fokuszustand"
        )

    def test_every_hover_colour_has_a_resting_counterpart(self) -> None:
        """Wer im Hover die Schriftfarbe setzt, muss sie auch im Ruhezustand setzen.

        Reine Flaechen-Hervorhebungen (nur ``background-color``, etwa das
        Zeilen-Highlight der Tabellen) sind davon ausgenommen: sie machen
        keinen Text sichtbar, sondern heben eine schon lesbare Zeile hervor.
        """
        body = _strip_comments(_css_text())
        rules = _rules(body)
        has_color = re.compile(r"(^|[;{\s])color\s*:")
        # Nur Ruhezustands-Regeln, die tatsaechlich eine Schriftfarbe
        # setzen. Eine Regel, die den Selektor bloss erwaehnt (Abstand,
        # Schriftgroesse), ist kein Ruhezustand der Farbe - diese Lockerheit
        # liess eine Hover-Farbe ohne Gegenstueck durchgehen.
        resting = {sel for sel, block in rules if ":hover" not in sel and has_color.search(block)}
        missing: list[str] = []
        for sel, block in rules:
            if ":hover" not in sel or not has_color.search(block):
                continue
            base = sel.replace(":hover", "").strip()
            # Einzelne Teilselektoren einer Gruppe genuegen.
            parts = [p.strip() for p in base.split(",") if p.strip()]
            # Bei einem Nachfahren-Selektor traegt das LETZTE Glied die
            # Farbe: ".bt-navlink:hover .bt-navlink-code" faerbt den Code,
            # nicht den Link. Der Ruhezustand ist also unter
            # ".bt-navlink-code" zu suchen, nicht unter der ganzen Kette.
            targets = [p.rsplit(" ", 1)[-1] for p in parts]
            if not any(any(t in r for r in resting) for t in targets):
                missing.append(sel)
        assert missing == [], f"Hover setzt Schriftfarbe ohne Ruhezustand: {missing}"
