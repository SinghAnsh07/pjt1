"""
Streamlit Application Entry Point for the V7 Audio Authenticity Analyzer.

Provides a clean, professional, non-technical web interface for audio upload,
metadata inspection, neural deepfake screening, and temporal timeline evaluation
using the verified V7 engine.
"""

import os
import sys
import tempfile
import streamlit as st
import numpy as np

# Ensure application root is on sys.path
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from core.inference import V7InferenceEngine
from core.audio_loader import load_audio
from core.preprocessor import preprocess_waveform
from core.windowing import generate_windows, analyze_long_recording
from core.video_processor import discover_ffmpeg
from core.config import (
    TARGET_SAMPLE_RATE,
    WINDOW_DURATION_SECONDS,
    DEFAULT_HOP_SECONDS,
    MAX_FILE_SIZE_BYTES,
)
from core.exceptions import DeepfakeAppError
from ui.styles import get_custom_css
from ui.components import (
    render_header,
    render_ffmpeg_missing_notice,
    render_metadata_card,
    render_results_summary,
    render_single_result,
    render_long_recording_result,
    render_technical_details,
    render_signal_insights,
    render_export_section,
    render_methodology_bento_section,
    render_limitations_and_privacy,
)
from ui.specular_button import render_specular_analyze_button


@st.cache_resource(show_spinner="Initializing V7 LCNN-2D Detection Engine...")
def get_cached_engine() -> V7InferenceEngine:
    """Loads and caches the V7 detection model in evaluation mode."""
    ckpt_path = os.path.abspath(os.path.join(APP_ROOT, "..", "v7", "best_lcnn2d_v7.pt"))
    return V7InferenceEngine(checkpoint_path=ckpt_path, device="cpu")


def main() -> None:
    st.set_page_config(
        page_title="V7 Audio Analyzer",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Inject visual styling
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    # 1. Header and System Notices
    render_header()

    has_ffmpeg, _ = discover_ffmpeg()
    if not has_ffmpeg:
        render_ffmpeg_missing_notice()

    # Load pre-trained model engine
    try:
        engine = get_cached_engine()
    except Exception as e:
        st.error(f"❌ Failed to initialize model backend: {str(e)}")
        st.stop()

    # 2. File Uploader Section
    uploaded_file = st.file_uploader(
        label="Choose an audio recording",
        type=["wav", "flac", "mp3", "ogg", "aiff", "aif", "m4a", "aac"],
        help="Supported formats: WAV, FLAC, MP3, OGG, AIFF, M4A, AAC (up to 100 MB).",
        accept_multiple_files=False,
    )

    # 3. Session State & File Invalidation
    current_file_id = None
    if uploaded_file is not None:
        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.get("active_file_id") != current_file_id:
        st.session_state["active_file_id"] = current_file_id
        st.session_state["loaded_audio"] = None
        st.session_state["analysis_result"] = None
        st.session_state["is_long_recording"] = False
        st.session_state["decoding_error"] = None
        st.session_state["export_cache_key"] = None
        st.session_state["cached_pdf_bytes"] = None
        st.session_state["cached_json_data"] = None
        st.session_state["cached_report_id"] = None

    if uploaded_file is None:
        st.info("👈 Select or drop an audio file above to begin screening.")
        render_methodology_bento_section()
        render_limitations_and_privacy()
        return

    # 4. Defensive Audio Decoding (Only on file selection change)
    if st.session_state.get("loaded_audio") is None and st.session_state.get("decoding_error") is None:
        with tempfile.TemporaryDirectory(prefix="v7_ui_upload_") as td:
            temp_path = os.path.join(td, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                loaded = load_audio(temp_path, max_size_bytes=MAX_FILE_SIZE_BYTES)
                st.session_state["loaded_audio"] = loaded
            except DeepfakeAppError as e:
                st.session_state["decoding_error"] = str(e)
            except Exception as e:
                st.session_state["decoding_error"] = f"An unexpected error occurred during audio loading: {str(e)}"

    if st.session_state.get("decoding_error"):
        st.error(f"⚠️ {st.session_state['decoding_error']}")
        render_limitations_and_privacy()
        return

    loaded = st.session_state["loaded_audio"]
    if loaded is None:
        return

    # 5. Display Audio Player & Inspected Metadata
    st.audio(uploaded_file)

    is_long = loaded.duration_seconds > WINDOW_DURATION_SECONDS
    num_windows = 1
    if is_long:
        target_total_samples = int(round(loaded.duration_seconds * TARGET_SAMPLE_RATE))
        windows, _ = generate_windows(
            target_total_samples,
            sample_rate=TARGET_SAMPLE_RATE,
        )
        num_windows = len(windows)

    render_metadata_card(loaded, num_windows=num_windows)

    # 6. Action Controls
    col_btn, col_reset = st.columns([3, 1])
    with col_btn:
        specular_mounted = render_specular_analyze_button(disabled=False)
        if specular_mounted:
            st.markdown(
                """
                <style>
                div[data-testid="stColumn"]:first-child div.stButton {
                    height: 0px !important;
                    margin: 0 !important;
                    padding: 0 !important;
                    overflow: hidden !important;
                }
                div[data-testid="stColumn"]:first-child div.stButton button {
                    opacity: 0 !important;
                    position: absolute !important;
                    width: 0px !important;
                    height: 0px !important;
                    margin: 0 !important;
                    padding: 0 !important;
                    overflow: hidden !important;
                    pointer-events: none !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            analyze_clicked = st.button("Analyze Audio", type="primary", use_container_width=True, key="specular_trigger_btn")
        else:
            analyze_clicked = st.button("Analyze Audio", type="primary", use_container_width=True)
    with col_reset:
        st.markdown('<div style="height: 14px;"></div>', unsafe_allow_html=True)
        if st.button("Reset", use_container_width=True):
            st.session_state["active_file_id"] = None
            st.session_state["loaded_audio"] = None
            st.session_state["analysis_result"] = None
            st.session_state["is_long_recording"] = False
            st.session_state["export_cache_key"] = None
            st.session_state["cached_pdf_bytes"] = None
            st.session_state["cached_json_data"] = None
            st.session_state["cached_report_id"] = None
            st.rerun()

    # 7. Analysis Execution
    if analyze_clicked:
        with st.spinner("Analyzing acoustic features and evaluating model similarity..."):
            try:
                if not is_long:
                    # Single 4.0s window inference
                    res = engine.predict_waveform(
                        loaded.waveform,
                        sample_rate=loaded.sample_rate,
                        filename=uploaded_file.name,
                    )
                    st.session_state["analysis_result"] = res
                    st.session_state["is_long_recording"] = False
                else:
                    # Multi-window long recording timeline analysis
                    full_tensor, _ = preprocess_waveform(
                        loaded.waveform,
                        sample_rate=loaded.sample_rate,
                        target_samples=None,
                    )
                    audio_16k_mono = full_tensor.squeeze(0).numpy()
                    long_res = analyze_long_recording(
                        audio_16k_mono=audio_16k_mono,
                        engine=engine,
                        hop_seconds=DEFAULT_HOP_SECONDS,
                        filename=uploaded_file.name,
                    )
                    st.session_state["analysis_result"] = long_res
                    st.session_state["is_long_recording"] = True

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.session_state["analysis_result"] = None
                st.session_state["export_cache_key"] = None
                st.session_state["cached_pdf_bytes"] = None
                st.session_state["cached_json_data"] = None
                st.session_state["cached_report_id"] = None

    # 8. Render Results
    analysis_res = st.session_state.get("analysis_result")
    if analysis_res is not None:
        render_results_summary(loaded=loaded, result=analysis_res)

        if not st.session_state.get("is_long_recording", False):
            render_single_result(analysis_res)
            render_technical_details(
                metadata=engine.metadata,
                single_res=analysis_res,
            )
        else:
            render_long_recording_result(analysis_res)
            render_technical_details(
                metadata=engine.metadata,
                long_res=analysis_res,
            )

        # 9. Audio Signal Insights (Waveform, Spectrogram, Timeline, Neural Features)
        render_signal_insights(
            loaded=loaded,
            result=analysis_res,
            engine=engine,
        )

        # 10. Downloadable Analysis Reports (PDF & JSON)
        render_export_section(
            loaded=loaded,
            result=analysis_res,
            engine=engine,
        )

        # 10b. Forensic Methodology & Detection Pipeline
        render_methodology_bento_section()

    # 11. Limitations and Local Privacy Footer
    render_limitations_and_privacy()


if __name__ == "__main__":
    main()
