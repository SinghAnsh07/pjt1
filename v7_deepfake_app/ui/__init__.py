"""
UI package for the V7 Audio Deepfake Detection System.
"""

from ui.styles import get_custom_css
from ui.components import (
    render_header,
    render_ffmpeg_missing_notice,
    render_metadata_card,
    render_single_result,
    render_long_recording_result,
    render_technical_details,
    render_limitations_and_privacy,
)

__all__ = [
    "get_custom_css",
    "render_header",
    "render_ffmpeg_missing_notice",
    "render_metadata_card",
    "render_single_result",
    "render_long_recording_result",
    "render_technical_details",
    "render_limitations_and_privacy",
]
