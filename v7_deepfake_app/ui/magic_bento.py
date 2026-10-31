"""
MagicBento Integration for V7 Audio Authenticity Analyzer.

Renders the official React Bits MagicBento component showcasing the 4-stage
scientific audio forensics pipeline with authentic signal processing descriptions,
interactive SVG scrubbers, and zero generic SaaS metrics.
"""

import os
import logging
from typing import Optional
import streamlit as st
import streamlit.components.v1 as components

logger = logging.getLogger(__name__)

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
BENTO_HTML_PATH = os.path.join(_APP_ROOT, "frontend", "gradient_ascii", "dist", "bento.html")

_CACHED_BENTO_HTML: Optional[str] = None


def get_bento_bundle_html() -> Optional[str]:
    """Reads and caches the self-contained single-file HTML bundle for MagicBento.

    Returns:
        The HTML bundle content as a string, or None if the bundle does not exist.
    """
    global _CACHED_BENTO_HTML
    if _CACHED_BENTO_HTML is not None:
        return _CACHED_BENTO_HTML

    if not os.path.isfile(BENTO_HTML_PATH):
        logger.debug(f"MagicBento bundle not found at {BENTO_HTML_PATH}")
        return None

    try:
        with open(BENTO_HTML_PATH, "r", encoding="utf-8") as f:
            _CACHED_BENTO_HTML = f.read()
        return _CACHED_BENTO_HTML
    except Exception as exc:
        logger.warning(f"Failed to read MagicBento bundle: {exc}")
        return None


def render_forensic_magic_bento(height: int = 315) -> bool:
    """Renders the authentic forensic methodology MagicBento section.

    Args:
        height: The base height in pixels for the container iframe.

    Returns:
        True if MagicBento was rendered successfully, False otherwise.
    """
    raw_html = get_bento_bundle_html()
    if not raw_html:
        return False

    try:
        components.html(raw_html, height=height, scrolling=False)
        return True
    except Exception as exc:
        logger.warning(f"Failed to render MagicBento component: {exc}")
        return False
