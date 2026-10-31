"""
PatternWaves Outer Background Integration for V7 Audio Authenticity Analyzer.

Integrates the official React Bits PatternWaves WebGL2 component as a
full-bleed outer page background behind all application cards, controls,
and content, with pointer event forwarding for responsive wave ripples and
zero interference with application inputs.
"""

import os
import logging
from typing import Optional
import streamlit as st
import streamlit.components.v1 as components

logger = logging.getLogger(__name__)

# Resolve frontend distribution directory and HTML entry point
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
DIST_DIR = os.path.join(_APP_ROOT, "frontend", "gradient_ascii", "dist")
INDEX_HTML_PATH = os.path.join(DIST_DIR, "index.html")

_CACHED_HTML: Optional[str] = None
_CACHED_MTIME: float = 0.0


def get_hero_bundle_html() -> Optional[str]:
    """Reads and returns the self-contained single-file HTML bundle.

    Reloads if the file on disk has changed.

    Returns:
        The HTML bundle content as a string, or None if the bundle does not exist.
    """
    global _CACHED_HTML, _CACHED_MTIME

    if not os.path.isfile(INDEX_HTML_PATH):
        logger.debug(f"PatternWaves bundle not found at {INDEX_HTML_PATH}")
        return None

    try:
        current_mtime = os.path.getmtime(INDEX_HTML_PATH)
        if _CACHED_HTML is not None and current_mtime == _CACHED_MTIME:
            return _CACHED_HTML

        with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
            _CACHED_HTML = f.read()
        _CACHED_MTIME = current_mtime
        return _CACHED_HTML
    except Exception as exc:
        logger.warning(f"Failed to read PatternWaves bundle: {exc}")
        return None


def render_pattern_waves_outer_background() -> bool:
    """Renders the official React Bits PatternWaves as the outer page background.

    Injects the compiled WebGL2 PatternWaves component into a fixed, full-viewport
    frame positioned at z-index: -1 behind all Streamlit elements. Pointer events
    are forwarded from the parent document to trigger dynamic ripples and trails
    without capturing or blocking any clicks on Streamlit buttons or uploaders.

    Returns:
        True if the outer background was rendered successfully, False otherwise.
    """
    html_content = get_hero_bundle_html()
    if not html_content:
        return False

    script_wrapper = f"""
    <div id="pattern-waves-outer-wrapper">
        {html_content}
    </div>
    <script>
    (function() {{
        try {{
            var ifr = window.frameElement;
            if (ifr) {{
                ifr.style.position = 'fixed';
                ifr.style.top = '0';
                ifr.style.left = '0';
                ifr.style.width = '100vw';
                ifr.style.height = '100vh';
                ifr.style.zIndex = '-1';
                ifr.style.border = 'none';
                ifr.style.margin = '0';
                ifr.style.padding = '0';
                ifr.style.pointerEvents = 'none';
                ifr.title = 'pattern_waves_outer_bg';

                var parentElem = ifr.closest('.element-container') || ifr.parentElement;
                if (parentElem) {{
                    parentElem.style.position = 'fixed';
                    parentElem.style.top = '0';
                    parentElem.style.left = '0';
                    parentElem.style.width = '0';
                    parentElem.style.height = '0';
                    parentElem.style.margin = '0';
                    parentElem.style.padding = '0';
                    parentElem.style.overflow = 'visible';
                    parentElem.style.zIndex = '-1';
                }}
            }}
        }} catch(e) {{
            console.warn('Outer background iframe positioning error:', e);
        }}
    }})();
    </script>
    """

    try:
        components.html(script_wrapper, height=0, scrolling=False)
        return True
    except Exception as exc:
        logger.warning(f"Failed to render PatternWaves outer background: {exc}")
        return False


def render_gradient_ascii_outer_background() -> bool:
    """Alias for render_pattern_waves_outer_background."""
    return render_pattern_waves_outer_background()


def render_gradient_ascii_hero(height: int = 600) -> bool:
    """Backward-compatible alias for render_pattern_waves_outer_background."""
    return render_pattern_waves_outer_background()
