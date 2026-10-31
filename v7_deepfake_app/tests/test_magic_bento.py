"""
Unit tests for MagicBento forensic component integration.
"""

import os
import pytest
from unittest.mock import patch
from ui.magic_bento import (
    get_bento_bundle_html,
    render_forensic_magic_bento,
    BENTO_HTML_PATH,
)
from ui.components import render_methodology_bento_section


def test_bento_bundle_exists_and_loads():
    """Validates that the compiled bento.html bundle exists and contains valid HTML."""
    assert os.path.isfile(BENTO_HTML_PATH), f"Bundle not found at {BENTO_HTML_PATH}"
    html = get_bento_bundle_html()
    assert html is not None
    assert len(html) > 1000
    assert "<!DOCTYPE html>" in html


def test_bento_bundle_forensic_content_integrity():
    """Validates that MagicBento bundle contains verified forensic content and no SaaS fakes."""
    html = get_bento_bundle_html()
    assert html is not None

    # Must contain verified signal processing and LCNN descriptions
    assert "16 kHz" in html
    assert "FrFT" in html
    assert "0.4820" in html
    assert "LCNN" in html or "Max-Feature-Map" in html

    # Must NOT contain fake generic SaaS demo metrics
    assert "$48.2k" not in html
    assert "+18%" not in html
    assert "MRR" not in html


def test_bento_rendering_success():
    """Validates that render_forensic_magic_bento invokes components.html cleanly."""
    with patch("streamlit.components.v1.html") as mock_html:
        success = render_forensic_magic_bento(height=280)
        assert success is True
        mock_html.assert_called_once()
        assert mock_html.call_args[1]["height"] == 280


def test_render_methodology_bento_section_executes():
    """Validates that the full section header and bento component execute cleanly."""
    with patch("streamlit.markdown") as mock_md, patch("ui.magic_bento.render_forensic_magic_bento") as mock_bento:
        render_methodology_bento_section()
        mock_md.assert_called_once()
        mock_bento.assert_called_once()
