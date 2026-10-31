"""
Unit and integration tests for the React Bits PatternWaves background component integration.
"""

import os
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

from ui.gradient_ascii_hero import (
    get_hero_bundle_html,
    render_gradient_ascii_hero,
    DIST_DIR,
    INDEX_HTML_PATH,
)
from ui.components import render_header, render_static_header

APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_PY_PATH = os.path.join(APP_ROOT, "app.py")


def test_gradient_ascii_bundle_exists():
    """Validates that the production bundle has been built and exists at expected path."""
    assert os.path.isdir(DIST_DIR), f"Dist directory not found at {DIST_DIR}"
    assert os.path.isfile(INDEX_HTML_PATH), f"Bundle index.html not found at {INDEX_HTML_PATH}"
    size = os.path.getsize(INDEX_HTML_PATH)
    assert size > 50_000, f"Bundle size {size} bytes is suspiciously small"


def test_get_hero_bundle_html_content():
    """Validates that bundle HTML content is successfully retrieved and cached."""
    html_content = get_hero_bundle_html()
    assert html_content is not None
    assert "<!DOCTYPE html>" in html_content or "<!doctype html>" in html_content
    assert "outer-pattern-waves-container" in html_content or "pattern-waves" in html_content


def test_pattern_waves_source_props_exact_match():
    """Validates that frontend source enforces the exact user-specified PatternWaves props."""
    jsx_path = os.path.join(APP_ROOT, "frontend", "gradient_ascii", "src", "App.jsx")
    assert os.path.isfile(jsx_path), "App.jsx source missing"

    with open(jsx_path, "r", encoding="utf-8") as f:
        src = f.read()

    # Exact component
    assert "PatternWaves" in src
    # Exact preset
    assert 'preset="silk"' in src or "preset={'silk'}" in src
    # Exact color
    assert 'color="#ffffff"' in src or "color={'#ffffff'}" in src
    # Exact backgroundColor
    assert 'backgroundColor="#000000"' in src or "backgroundColor={'#000000'}" in src
    # Exact fade
    assert 'fade="edges"' in src or "fade={'edges'}" in src
    # Exact interactive
    assert "interactive" in src
    # Exact cursorSize and cursorStrength
    assert "cursorSize={50}" in src
    assert "cursorStrength={0.6}" in src


def test_pattern_waves_css_layout_properties():
    """Validates that container CSS matches layout, overflow, and fixed background specifications."""
    css_path = os.path.join(APP_ROOT, "frontend", "gradient_ascii", "src", "App.css")
    assert os.path.isfile(css_path), "App.css source missing"

    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()

    assert "position: fixed;" in css
    assert "overflow: hidden;" in css
    assert "width: 100vw;" in css
    assert "height: 100vh;" in css


def test_gradient_ascii_fallback_when_dist_missing():
    """Validates that missing bundle triggers graceful fallback without raising exceptions."""
    with patch("ui.gradient_ascii_hero.get_hero_bundle_html", return_value=None):
        rendered = render_gradient_ascii_hero()
        assert rendered is False


def test_render_header_fallback_invoked_when_missing():
    """Validates that render_header seamlessly calls render_static_header if hero returns False."""
    with patch("ui.components.render_gradient_ascii_hero", return_value=False):
        with patch("ui.components.render_static_header") as mock_static:
            render_header()
            mock_static.assert_called_once()


def test_app_test_runs_with_gradient_ascii_hero():
    """Verifies that the full Streamlit app initializes cleanly with the hero component."""
    at = AppTest.from_file(APP_PY_PATH)
    at.run(timeout=15)
    assert not at.exception, f"App execution raised exception: {at.exception}"
