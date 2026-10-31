"""
SpecularButton Integration for V7 Audio Authenticity Analyzer.

Renders the official React Bits SpecularButton component with WebGL2 specular shine
reflections, interactive mouse proximity tracking, keyboard accessibility, and direct
event delegation to the Streamlit inference execution workflow.
"""

import os
import logging
from typing import Optional
import streamlit as st
import streamlit.components.v1 as components

logger = logging.getLogger(__name__)

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
SPECULAR_HTML_PATH = os.path.join(_APP_ROOT, "frontend", "gradient_ascii", "dist", "specular.html")

_CACHED_SPECULAR_HTML: Optional[str] = None


def get_specular_bundle_html() -> Optional[str]:
    """Reads and caches the self-contained single-file HTML bundle for SpecularButton.

    Returns:
        The HTML bundle content as a string, or None if the bundle does not exist.
    """
    global _CACHED_SPECULAR_HTML
    if _CACHED_SPECULAR_HTML is not None:
        return _CACHED_SPECULAR_HTML

    if not os.path.isfile(SPECULAR_HTML_PATH):
        logger.debug(f"SpecularButton bundle not found at {SPECULAR_HTML_PATH}")
        return None

    try:
        with open(SPECULAR_HTML_PATH, "r", encoding="utf-8") as f:
            _CACHED_SPECULAR_HTML = f.read()
        return _CACHED_SPECULAR_HTML
    except Exception as exc:
        logger.warning(f"Failed to read SpecularButton bundle: {exc}")
        return None


def render_specular_analyze_button(disabled: bool = False, height: int = 68) -> bool:
    """Renders the React Bits SpecularButton as the primary action button.

    Args:
        disabled: Whether the button is in a disabled state (e.g. while processing or without audio).
        height: The iframe height in pixels.

    Returns:
        True if the button was rendered successfully, False if falling back to native Streamlit button.
    """
    raw_html = get_specular_bundle_html()
    if not raw_html:
        return False

    # Inject runtime configuration (disabled state) into the bundle
    config_script = f"<script>window.__BUTTON_DISABLED__ = {'true' if disabled else 'false'};</script>"
    injected_html = raw_html.replace("<head>", f"<head>{config_script}", 1)

    try:
        components.html(injected_html, height=height, scrolling=False)
        return True
    except Exception as exc:
        logger.warning(f"Failed to render SpecularButton component: {exc}")
        return False
