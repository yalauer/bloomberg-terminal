"""Querschnittstests der viz-Schicht.

Friert Architekturregel 3 ein: ``viz/`` gibt Figures zurueck, ruft nie
``.show()`` auf und enthaelt keine Datenbeschaffung.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import plotly.graph_objects as go
import pytest

from btlite.viz import backtest, frontier, fundamentals, price, risk, theme

VIZ_MODULES = (theme, price, fundamentals, risk, frontier, backtest)
VIZ_DIR = Path(theme.__file__).parent


def _source_files() -> list[Path]:
    return sorted(VIZ_DIR.glob("*.py"))


def test_viz_package_has_expected_modules():
    names = {p.stem for p in _source_files()}
    assert {"theme", "price", "fundamentals", "risk", "frontier", "backtest"} <= names


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_never_calls_show(path):
    """Architekturregel 3: keine Factory zeigt eine Figure selbst an."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"show", "write_html", "write_image"}, (
                f"{path.name}: verbotener Aufruf .{node.func.attr}()"
            )


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_no_network_or_data_layer_imports(path):
    """Architekturregel 2 und 3: viz/ holt keine Daten."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = {"requests", "urllib", "httpx", "yfinance", "finvizfinance", "socket"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in forbidden, f"{path.name}: importiert {alias.name}"
                assert not alias.name.startswith("btlite.data"), (
                    f"{path.name}: importiert die Datenschicht"
                )
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            assert root not in forbidden, f"{path.name}: importiert {node.module}"
            assert not node.module.startswith("btlite.data"), (
                f"{path.name}: importiert die Datenschicht"
            )


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_source_is_ascii_only(path):
    """Bestandskonvention: Umlaute werden als ue/ae/oe geschrieben."""
    text = path.read_text(encoding="utf-8")
    offenders = sorted({c for c in text if ord(c) > 127})
    assert not offenders, f"{path.name}: nicht-ASCII {offenders}"


def _public_functions(module):
    return [
        (name, obj)
        for name, obj in inspect.getmembers(module, inspect.isfunction)
        if not name.startswith("_") and obj.__module__ == module.__name__
    ]


@pytest.mark.parametrize("module", VIZ_MODULES, ids=lambda m: m.__name__)
def test_public_functions_have_docstrings(module):
    for name, func in _public_functions(module):
        assert func.__doc__, f"{module.__name__}.{name} ohne Docstring"


@pytest.mark.parametrize("module", VIZ_MODULES, ids=lambda m: m.__name__)
def test_public_functions_are_annotated(module):
    for name, func in _public_functions(module):
        signature = inspect.signature(func)
        assert signature.return_annotation is not inspect.Signature.empty, (
            f"{module.__name__}.{name} ohne Rueckgabe-Annotation"
        )
        for param in signature.parameters.values():
            assert param.annotation is not inspect.Signature.empty, (
                f"{module.__name__}.{name}: Parameter {param.name} ohne Annotation"
            )


@pytest.mark.parametrize("module", VIZ_MODULES, ids=lambda m: m.__name__)
def test_documented_returns_mention_figure(module):
    """Jede Factory dokumentiert, dass sie eine Figure liefert."""
    for name, func in _public_functions(module):
        if inspect.signature(func).return_annotation in (go.Figure, "go.Figure"):
            assert "Returns:" in (func.__doc__ or ""), (
                f"{module.__name__}.{name} ohne Returns-Abschnitt"
            )
