"""
Unit tests for SpecularButton component integration.
"""

import os
import pytest
from unittest.mock import patch
from ui.specular_button import (
    get_specular_bundle_html,
    render_specular_analyze_button,
    SPECULAR_HTML_PATH,
)


def test_specular_bundle_exists_and_loads():
    """Validates that the compiled specular.html bundle exists and contains valid HTML."""
    assert os.path.isfile(SPECULAR_HTML_PATH), f"Bundle not found at {SPECULAR_HTML_PATH}"
    html = get_specular_bundle_html()
    assert html is not None
    assert len(html) > 1000
    assert "<!DOCTYPE html>" in html
    assert "Analyze Audio" in html or "specular-button" in html


def test_specular_button_disabled_injection():
    """Validates that the disabled configuration is properly injected into the HTML."""
    with patch("streamlit.components.v1.html") as mock_html:
        success = render_specular_analyze_button(disabled=True, height=64)
        assert success is True
        mock_html.assert_called_once()
        rendered_content = mock_html.call_args[0][0]
        assert "window.__BUTTON_DISABLED__ = true;" in rendered_content

    with patch("streamlit.components.v1.html") as mock_html:
        success = render_specular_analyze_button(disabled=False, height=64)
        assert success is True
        mock_html.assert_called_once()
        rendered_content = mock_html.call_args[0][0]
        assert "window.__BUTTON_DISABLED__ = false;" in rendered_content


def test_specular_button_fallback_on_missing_bundle():
    """Validates that render_specular_analyze_button returns False when bundle is missing."""
    with patch("ui.specular_button.get_specular_bundle_html", return_value=None):
        success = render_specular_analyze_button()
        assert success is False
