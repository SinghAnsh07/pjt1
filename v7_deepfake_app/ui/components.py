"""
Presentation and visual layout components for the V7 Audio Authenticity Analyzer.

Implements clean, accessible Streamlit UI elements with plain-language explanations,
restrained color semantics, dynamic audio exploration, and production-quality styling.
Does not fabricate probabilities, confidence percentages, or legal authenticity claims.
"""

import streamlit as st
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any, Union

from core.schemas import (
    LoadedAudio,
    InferenceResult,
    LongRecordingInferenceResult,
    DecisionLabel,
    ModelMetadata,
)
from core.config import (
    WINDOW_DURATION_SECONDS,
    DEFAULT_HOP_SECONDS,
    WINDOW_SAMPLES,
    TARGET_SAMPLE_RATE,
)
from core.preprocessor import preprocess_waveform
from ui.plots import (
    create_waveform_figure,
    create_spectrogram_figure,
    create_timeline_figure,
    create_feature_views_figure,
)
from reporting.report_schema import build_report_payload
from reporting.json_export import export_report_to_json
from reporting.pdf_export import export_report_to_pdf
from ui.gradient_ascii_hero import render_gradient_ascii_hero


def render_static_header() -> None:
    """Renders the static product header, badges, and introductory explanation."""
    st.markdown(
        """
        <div class="app-header">
            <div class="app-header-top">
                <div class="app-title-group">
                    <div class="app-icon">🎙️</div>
                    <div>
                        <h1 class="app-title">V7 Audio Analyzer</h1>
                        <div class="app-subtitle">Audio authenticity screening</div>
                    </div>
                </div>
                <div class="header-badges">
                    <span class="badge-pill">
                        <span class="badge-local-dot"></span>
                        Processed on this device
                    </span>
                    <span class="badge-pill">Model: V7 LCNN-2D</span>
                </div>
            </div>
        </div>
        <div class="intro-card">
            <div class="intro-headline">Understand what an audio recording may reveal.</div>
            <div class="intro-text">
                Analyze audio for acoustic patterns associated with genuine or AI-generated speech.
                Results are indicative, not definitive proof.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Renders the product header, badges, and introductory explanation.

    Mounts the interactive React Bits GradientASCII WebGL2 outer background
    behind all application elements if the compiled bundle is present, followed
    by the foreground brand header and introductory guidance card.
    """
    render_gradient_ascii_hero()
    render_static_header()


def render_ffmpeg_missing_notice() -> None:
    """Renders a contextual callout about optional FFmpeg dependency for video."""
    st.info(
        "ℹ️ **Supported Uploads:** WAV, FLAC, MP3, OGG, AIFF, and M4A audio files are supported. "
        "Video files (.mp4, .mov) require external FFmpeg, which is currently unavailable on this system."
    )


def render_metadata_card(loaded: LoadedAudio, num_windows: int = 1) -> None:
    """Displays verified file and container metadata before or after inference."""
    duration_str = f"{loaded.duration_seconds:.2f} s"
    ch_str = "Mono (1 channel)" if loaded.channels == 1 else f"Stereo ({loaded.channels} channels)"

    if loaded.duration_seconds <= WINDOW_DURATION_SECONDS:
        analysis_type = "Single 4.0s Window"
        if loaded.duration_seconds < WINDOW_DURATION_SECONDS:
            analysis_type += " (Wrap-Padded)"
    else:
        analysis_type = f"Sliding Timeline ({num_windows} windows, {DEFAULT_HOP_SECONDS:.1f}s hop)"

    st.markdown(
        f"""
        <div class="metadata-card">
            <div class="metadata-title">Audio File Inspection</div>
            <div class="metadata-grid">
                <div>
                    <div class="metadata-item-label">Container Format</div>
                    <div class="metadata-item-value">{loaded.format_name} ({loaded.subtype})</div>
                </div>
                <div>
                    <div class="metadata-item-label">Recorded Duration</div>
                    <div class="metadata-item-value">{duration_str}</div>
                </div>
                <div>
                    <div class="metadata-item-label">Original Sample Rate</div>
                    <div class="metadata-item-value">{loaded.sample_rate:,} Hz</div>
                </div>
                <div>
                    <div class="metadata-item-label">Channel Layout</div>
                    <div class="metadata-item-value">{ch_str}</div>
                </div>
                <div>
                    <div class="metadata-item-label">Target Sample Rate</div>
                    <div class="metadata-item-value">16,000 Hz</div>
                </div>
                <div>
                    <div class="metadata-item-label">Analysis Segmentation</div>
                    <div class="metadata-item-value">{analysis_type}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_single_result(result: InferenceResult) -> None:
    """Displays verdict and plain-language explanation for a single-window inference result."""
    is_real = result.decision.prediction == DecisionLabel.REAL

    if is_real:
        card_class = "verdict-card verdict-real"
        title_class = "verdict-title-real"
        heading_text = "Patterns more consistent with genuine speech"
        status_label = "Likely Genuine Audio"
        pill_class = "pill-real"
        desc_text = "The recording's audio patterns are closer to those the model learned from genuine speech."
        caveat_text = (
            "This is an automated screening result, not proof that the recording is authentic. "
            "Edited or AI-generated audio may still receive this result."
        )
    else:
        card_class = "verdict-card verdict-fake"
        title_class = "verdict-title-fake"
        heading_text = "Unusual speech patterns detected"
        status_label = "Potential AI-Generated or Manipulated Audio"
        pill_class = "pill-fake"
        desc_text = "The recording's audio patterns differ from those the model learned from genuine speech."
        caveat_text = (
            "This does not prove that the recording was AI-generated. "
            "Noise, compression, recording equipment, or other factors may affect the result."
        )

    # Defensive formatting for model_score and threshold
    score_raw = getattr(result, "model_score", None)
    if isinstance(score_raw, (int, float, np.floating, np.integer)):
        score_val = f"{float(score_raw):+.4f}"
    else:
        score_val = str(score_raw) if score_raw is not None else "N/A"

    th_raw = getattr(result.decision, "threshold", 0.4820)
    if isinstance(th_raw, (int, float, np.floating, np.integer)):
        thresh_val = f"{float(th_raw):.4f}"
    else:
        thresh_val = str(th_raw)

    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="verdict-header-row">
                <h3 class="{title_class}">{heading_text}</h3>
                <span class="verdict-status-pill {pill_class}">{status_label}</span>
            </div>
            <p class="verdict-desc">{desc_text}</p>
            <div class="score-strip">
                <div class="score-item">
                    <span class="score-label">Model similarity score</span>
                    <span class="score-val">{score_val}</span>
                </div>
                <div class="score-item">
                    <span class="score-label">Candidate threshold</span>
                    <span class="score-val">{thresh_val}</span>
                </div>
                <div class="score-explanation">
                    A technical score describing how closely the recording matches patterns learned from genuine speech.
                </div>
            </div>
            <div class="verdict-disclaimer">
                <strong>Important:</strong> {caveat_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_long_recording_result(result: LongRecordingInferenceResult) -> None:
    """Displays multi-window timeline breakdown with plain-language explanations."""
    total_w = result.total_windows
    real_w = result.summary_real_windows
    fake_w = result.summary_fake_windows
    flagged_pct = (fake_w / total_w) * 100.0 if total_w > 0 else 0.0

    if fake_w == 0:
        banner_class = "verdict-card verdict-real"
        title_class = "verdict-title-real"
        heading_text = "Patterns more consistent with genuine speech across all segments"
        status_label = "Likely Genuine Audio"
        pill_class = "pill-real"
        desc = (
            f"All {total_w} analyzed segments across the {result.covered_duration_seconds:.1f}s recording "
            f"matched patterns learned from genuine speech."
        )
        caveat = (
            "This is an automated screening result, not proof that the recording is authentic. "
            "Edited or AI-generated audio may still receive this result."
        )
    elif real_w == 0:
        banner_class = "verdict-card verdict-fake"
        title_class = "verdict-title-fake"
        heading_text = "Unusual speech patterns detected across all segments"
        status_label = "Potential AI-Generated or Manipulated Audio"
        pill_class = "pill-fake"
        desc = (
            f"All {total_w} analyzed segments across the {result.covered_duration_seconds:.1f}s recording "
            f"differed from patterns the model learned from genuine speech."
        )
        caveat = (
            "This does not prove that the recording was AI-generated. "
            "Noise, compression, recording equipment, or other factors may affect the result."
        )
    else:
        banner_class = "verdict-card verdict-fake"
        title_class = "verdict-title-fake"
        heading_text = "Mixed speech patterns detected across recording"
        status_label = "Mixed Patterns"
        pill_class = "pill-fake"
        desc = (
            f"Screening results vary across segments: {real_w} segment(s) aligned with genuine speech patterns, "
            f"and {fake_w} segment(s) showed unusual patterns. Natural speech dynamics, background noise, "
            f"or compression artifacts warrant careful contextual review."
        )
        caveat = (
            "Screening results vary across the recording. A single segment with unusual patterns does not prove "
            "the entire recording is AI-generated, nor does a genuine segment guarantee authenticity."
        )

    st.markdown(
        f"""
        <div class="{banner_class}">
            <div class="verdict-header-row">
                <h3 class="{title_class}">{heading_text}</h3>
                <span class="verdict-status-pill {pill_class}">{status_label}</span>
            </div>
            <p class="verdict-desc">{desc}</p>
            <div class="score-strip">
                <div class="score-item">
                    <span class="score-label">Mean similarity score</span>
                    <span class="score-val">{result.mean_score:+.4f}</span>
                </div>
                <div class="score-item">
                    <span class="score-label">Score range</span>
                    <span class="score-val">[{result.min_score:+.4f}, {result.max_score:+.4f}]</span>
                </div>
                <div class="score-explanation">
                    Average model similarity score across all analyzed segments. It reflects pattern similarity, not a probability.
                </div>
            </div>
            <div class="verdict-disclaimer">
                <strong>Important:</strong> {caveat}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Window metrics summary (exact labels for regression test compatibility)
    st.markdown("##### ⏱️ Timeline Segmentation Breakdown")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Windows", total_w)
    with col2:
        st.metric("Likely Genuine", real_w)
    with col3:
        st.metric("Potential Fake", f"{fake_w} ({flagged_pct:.1f}%)")
    with col4:
        st.metric("Timeline Coverage", f"{result.covered_duration_seconds:.1f}s (100%)")

    st.caption(
        f"ℹ️ **Metric Note:** 'Potential Fake' shows the percentage of analyzed 4.0s windows assigned to the "
        f"unusual-patterns category ({fake_w}/{total_w} windows). Because windows overlap by 75% "
        f"({DEFAULT_HOP_SECONDS:.1f}s hop), this reflects window count distribution, not a percentage of unique "
        f"audio duration or a probability that the recording is fake."
    )

    # Table of individual windows
    table_data = []
    for w in result.windows:
        status_label = "Likely Genuine" if w.decision.prediction == DecisionLabel.REAL else "Unusual Patterns"
        table_data.append({
            "Window": f"#{w.window.window_index + 1}",
            "Time Interval": f"{w.window.start_time_seconds:.1f}s – {w.window.end_time_seconds:.1f}s",
            "Screening Assessment": status_label,
            "Latency": f"{w.timing_ms:.1f} ms",
        })

    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True,
    )


def render_results_summary(
    loaded: LoadedAudio,
    result: Union[InferenceResult, LongRecordingInferenceResult],
) -> None:
    """Renders a clean, compact grid summarizing the recording and analysis results."""
    is_long = isinstance(result, LongRecordingInferenceResult)
    num_windows = result.total_windows if is_long else 1

    if is_long:
        if result.summary_fake_windows == 0:
            outcome = "Genuine Patterns"
        elif result.summary_real_windows == 0:
            outcome = "Unusual Patterns"
        else:
            outcome = "Mixed Patterns"
        score_display = f"{result.mean_score:+.4f} (mean)"
        timing_display = f"{result.total_processing_time_ms:.1f} ms"
    else:
        is_real = result.decision.prediction == DecisionLabel.REAL
        outcome = "Genuine Patterns" if is_real else "Unusual Patterns"
        score_display = f"{result.model_score:+.4f}"
        timing_display = f"{result.timing.total_ms:.1f} ms"

    format_desc = getattr(loaded, "format_name", "Audio")

    st.markdown(
        f"""
        <div class="results-summary-card">
            <div class="summary-grid">
                <div>
                    <div class="summary-cell-label">Recording</div>
                    <div class="summary-cell-value">{format_desc} ({loaded.duration_seconds:.2f}s)</div>
                </div>
                <div>
                    <div class="summary-cell-label">Segments</div>
                    <div class="summary-cell-value">{num_windows} {'window' if num_windows == 1 else 'windows'}</div>
                </div>
                <div>
                    <div class="summary-cell-label">Screening Finding</div>
                    <div class="summary-cell-value">{outcome}</div>
                </div>
                <div>
                    <div class="summary-cell-label">Similarity Score</div>
                    <div class="summary-cell-value">{score_display}</div>
                </div>
                <div>
                    <div class="summary-cell-label">Analysis Time</div>
                    <div class="summary-cell-value">{timing_display}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_technical_details(
    metadata: Optional[ModelMetadata] = None,
    single_res: Optional[InferenceResult] = None,
    long_res: Optional[LongRecordingInferenceResult] = None,
) -> None:
    """Renders the collapsed Technical Details expander with factual backend metrics."""
    with st.expander("🔍 How this result was produced (Technical Details)", expanded=False):
        st.markdown("**Model Specifications**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("- **Model Architecture:** 2D Light CNN (LCNN-2D)")
            st.markdown("- **Feature Representation:** 3-Channel 2D Spatiotemporal Tensor")
            st.markdown("  - Channel 0: ST-FrFT (α = 0.50, chirp angle 45°)")
            st.markdown("  - Channel 1: ST-FrFT (α = 0.85, chirp angle 76.5°)")
            st.markdown("  - Channel 2: Continuous Morlet Wavelet Scalogram (64 scales)")
            st.markdown("- **Embedding Dimension:** 128-D")
        with col2:
            st.markdown("- **Total Parameters:** 170,336 (Backbone: 170,208 | Centroid: 128)")
            st.markdown("- **Scoring Method:** One-Class Softmax Cosine Similarity")
            st.markdown("- **Score Interpretation:** S(x) ∈ [-1.0, +1.0]; Higher = More Bona Fide")
            st.markdown("- **Candidate Decision Threshold:** `0.4820` (Derived from ASVspoof 2019 LA benchmark)")
            st.markdown("- **Threshold Status:** Candidate operating point (uncalibrated on in-the-wild audio)")

        st.markdown("---")
        st.markdown("**Processing & Timing Breakdown**")
        if single_res:
            t = single_res.timing
            st.markdown(
                f"- Preprocessing: `{t.preprocessing_ms:.1f} ms` | "
                f"Feature Extraction: `{t.feature_extraction_ms:.1f} ms` | "
                f"Neural Forward: `{t.neural_forward_ms:.1f} ms` | "
                f"**Total:** `{t.total_ms:.1f} ms`"
            )
            st.markdown(f"- Input Tensor Shape: `{single_res.feature_tensor_shape}`")
        elif long_res:
            flagged_pct = (long_res.summary_fake_windows / long_res.total_windows) * 100.0 if long_res.total_windows > 0 else 0.0
            st.markdown(
                f"- **Total Multi-Window Processing Time:** `{long_res.total_processing_time_ms:.1f} ms` "
                f"across {long_res.total_windows} windows."
            )
            st.markdown(
                f"- **Mean cosine similarity score across analyzed windows:** `{long_res.mean_score:+.4f}`"
            )
            st.markdown(
                f"- **Minimum observed cosine similarity across analyzed windows:** `{long_res.min_score:+.4f}` "
                f"(identifies lowest score among processed windows; not standalone proof of splicing or manipulation)"
            )
            st.markdown(
                f"- **Maximum observed cosine similarity across analyzed windows:** `{long_res.max_score:+.4f}`"
            )
            st.markdown(
                f"- **Percentage of analyzed windows assigned potential-manipulation category:** `{flagged_pct:.1f}%` "
                f"({long_res.summary_fake_windows}/{long_res.total_windows} windows)"
            )


def render_signal_insights(
    loaded: LoadedAudio,
    result: Union[InferenceResult, LongRecordingInferenceResult],
    engine: Any,
) -> None:
    """
    Renders the Audio Signal Insights section with tabbed Waveform, Spectrogram,
    Timeline, and Neural Feature Views.
    """
    is_long = isinstance(result, LongRecordingInferenceResult)

    if is_long:
        overall_status = "Genuine Patterns" if result.summary_fake_windows == 0 else ("Unusual Patterns" if result.summary_real_windows == 0 else "Mixed Patterns")
        score_val = f"{result.mean_score:+.4f} (mean)"
    else:
        overall_status = "Genuine Patterns" if result.decision.prediction == DecisionLabel.REAL else "Unusual Patterns"
        score_val = f"{result.model_score:+.4f}"

    st.markdown(
        f"""
        <div class="insights-header">
            <div class="insights-title">Explore the recording</div>
            <div class="insights-subtitle">
                Review the audio waveform, frequency patterns, and changes across the recording.
            </div>
            <div class="insights-context-pill">
                <span>🎵 <strong>Duration:</strong> {loaded.duration_seconds:.2f}s</span>
                <span>•</span>
                <span>📊 <strong>Overall:</strong> {overall_status}</span>
                <span>•</span>
                <span>⚡ <strong>Score:</strong> {score_val}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_wave, tab_spec, tab_time, tab_feat = st.tabs([
        "🌊 Waveform",
        "🌈 Spectrogram",
        "⏱️ Analysis Timeline",
        "🧬 Advanced Model Views",
    ])

    # Precompute 16 kHz mono waveform for tabs that benefit from it
    audio_16k_mono = None
    try:
        if is_long:
            full_tensor, _ = preprocess_waveform(
                loaded.waveform,
                sample_rate=loaded.sample_rate,
                target_samples=None,
            )
            audio_16k_mono = full_tensor.squeeze(0).numpy()
        else:
            prep_tensor, _ = preprocess_waveform(
                loaded.waveform,
                sample_rate=loaded.sample_rate,
                target_samples=WINDOW_SAMPLES,
            )
            audio_16k_mono = prep_tensor.squeeze(0).numpy()
    except Exception:
        audio_16k_mono = None

    # -------------------------------------------------------------------------
    # Tab 1: Waveform
    # -------------------------------------------------------------------------
    with tab_wave:
        col_sel, _ = st.columns([2, 2])
        with col_sel:
            wave_view = st.radio(
                "Waveform Signal View:",
                options=["Original Decoded Signal", "Prepared 16 kHz Mono Signal"],
                horizontal=True,
                key="wave_view_selector",
            )

        try:
            if wave_view == "Prepared 16 kHz Mono Signal" and audio_16k_mono is not None:
                plot_audio = audio_16k_mono
                plot_sr = TARGET_SAMPLE_RATE
                subtitle = f"Prepared Signal at 16,000 Hz Mono ({len(plot_audio):,} samples)"
            else:
                plot_audio = loaded.waveform
                plot_sr = loaded.sample_rate
                ch_text = "Mono" if loaded.channels == 1 else f"{loaded.channels}-ch"
                subtitle = f"Original Decoded Signal ({loaded.sample_rate:,} Hz, {ch_text}, {loaded.frames:,} frames)"

            fig_wave = create_waveform_figure(
                audio=plot_audio,
                sample_rate=plot_sr,
                title="Audio Waveform (Amplitude vs. Time)",
                subtitle=subtitle,
            )
            st.pyplot(fig_wave)
            plt.close(fig_wave)

            st.caption(
                "ℹ️ **Waveform Interpretation:** Shows how the audio signal's amplitude changes over time. "
                "Useful for inspecting pauses, loudness transitions, and recording dynamics. "
                "Peak-envelope decimation preserves acoustic transients without altering inference data."
            )
        except Exception as e:
            st.warning(f"⚠️ Unable to render waveform visualization: {str(e)}")

    # -------------------------------------------------------------------------
    # Tab 2: Spectrogram
    # -------------------------------------------------------------------------
    with tab_spec:
        try:
            spec_audio = audio_16k_mono if audio_16k_mono is not None else loaded.waveform
            spec_sr = TARGET_SAMPLE_RATE if audio_16k_mono is not None else loaded.sample_rate

            fig_spec = create_spectrogram_figure(
                audio=spec_audio,
                sample_rate=spec_sr,
                title=f"Audio Spectrogram (0 – {spec_sr / 2000.0:.1f} kHz, Power in dB)",
            )
            st.pyplot(fig_spec)
            plt.close(fig_spec)

            st.caption(
                "ℹ️ **Spectrogram Interpretation:** A spectrogram shows how the recording's sound energy changes across "
                "frequencies over time. Similar spectral patterns can occur in both genuine and synthetic speech, "
                "so this plot illustrates acoustic frequency structure rather than proving manipulation."
            )
        except Exception as e:
            st.warning(f"⚠️ Unable to render spectrogram visualization: {str(e)}")

    # -------------------------------------------------------------------------
    # Tab 3: Analysis Timeline
    # -------------------------------------------------------------------------
    with tab_time:
        if is_long:
            try:
                thresh = getattr(engine.decision_engine, "threshold", 0.4820)
                fig_time = create_timeline_figure(
                    long_result=result,
                    candidate_threshold=thresh,
                )
                st.pyplot(fig_time)
                plt.close(fig_time)

                st.caption(
                    "ℹ️ **Timeline Interpretation:** Shows the model's cosine similarity score for each analyzed "
                    "4.0s window across the recording (1.0s hop). Scores indicate similarity to the model's bona fide "
                    "reference representation; they are not probabilities or percentages. The dashed amber line marks "
                    f"the candidate threshold (τ* = {thresh:.4f}, unvalidated)."
                )

                # Interactive Window Inspector
                st.markdown("##### 🔍 Inspect Individual Segment")
                selected_idx = st.slider(
                    "Select segment index to inspect:",
                    min_value=1,
                    max_value=len(result.windows),
                    value=1,
                    step=1,
                    key="timeline_window_slider",
                ) - 1

                inspect_win = result.windows[selected_idx]
                col_w1, col_w2, col_w3 = st.columns(3)
                with col_w1:
                    st.metric(
                        f"Segment #{selected_idx + 1} Interval",
                        f"{inspect_win.window.start_time_seconds:.1f}s – {inspect_win.window.end_time_seconds:.1f}s",
                    )
                with col_w2:
                    st.metric(
                        "Segment Score",
                        f"{inspect_win.model_score:+.4f}",
                    )
                with col_w3:
                    seg_label = "Genuine Patterns" if inspect_win.decision.prediction == DecisionLabel.REAL else "Unusual Patterns"
                    st.metric(
                        "Assessment",
                        seg_label,
                    )

            except Exception as e:
                st.warning(f"⚠️ Unable to render timeline visualization: {str(e)}")
        else:
            st.info(
                f"ℹ️ **Single 4.0s Window Analysis:** This recording ({loaded.duration_seconds:.2f}s) was evaluated as "
                "a single standardized 4.0s window. Multi-window sliding timeline charts apply to extended recordings "
                "exceeding 4.0 seconds."
            )
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Model Similarity Score", f"{result.model_score:+.4f}")
            with col_s2:
                st.metric("Candidate Threshold", f"{result.decision.threshold:.4f}")
            with col_s3:
                st.metric("Margin to Threshold", f"{result.decision.margin_to_threshold:+.4f}")

    # -------------------------------------------------------------------------
    # Tab 4: Advanced Model Views (ST-FrFT + Wavelet Scalogram)
    # -------------------------------------------------------------------------
    with tab_feat:
        st.markdown("**Internal Neural Representations**")
        st.caption(
            "These visualizations show internal representations used by the model. "
            "They are not direct explanations of why a recording received its result."
        )
        try:
            if is_long and audio_16k_mono is not None:
                win_idx = st.selectbox(
                    "Select Timeline Window to Inspect Neural Representation:",
                    range(len(result.windows)),
                    format_func=lambda i: (
                        f"Window #{i + 1} ({result.windows[i].window.start_time_seconds:.1f}s – "
                        f"{result.windows[i].window.end_time_seconds:.1f}s) | Score: "
                        f"{result.windows[i].model_score:+.4f} | {result.windows[i].decision.prediction.value}"
                    ),
                    key="feature_window_selector",
                )
                sel_win = result.windows[win_idx].window
                if sel_win.is_padded:
                    slice_raw = np.pad(audio_16k_mono, (0, WINDOW_SAMPLES - len(audio_16k_mono)), mode="wrap")
                else:
                    slice_raw = audio_16k_mono[sel_win.start_sample : sel_win.end_sample]
                slice_tensor = torch.from_numpy(slice_raw.astype(np.float32)).unsqueeze(0).to(engine.device)
                window_label = f"Window #{win_idx + 1} ({sel_win.start_time_seconds:.1f}s – {sel_win.end_time_seconds:.1f}s)"
            else:
                prep_tensor, _ = preprocess_waveform(
                    loaded.waveform,
                    sample_rate=loaded.sample_rate,
                    target_samples=WINDOW_SAMPLES,
                )
                slice_tensor = prep_tensor.to(engine.device)
                window_label = f"Standard 4.0s Window ({loaded.duration_seconds:.2f}s audio)"

            with torch.inference_mode():
                feat_tensor = engine.model.extractor(slice_tensor).detach().cpu()

            fig_feat = create_feature_views_figure(feat_tensor, window_label=window_label)
            st.pyplot(fig_feat)
            plt.close(fig_feat)

            st.caption(
                "ℹ️ **Feature Representation Note:** Displays the 3-channel 2D spatiotemporal representations generated "
                "by the model's feature extraction layer (ST-FrFT with α=0.50 and α=0.85, and Continuous Morlet Wavelet "
                "Scalogram). These represent transform-domain representations processed by the LCNN-2D convolutional "
                "backbone, not causal explanations of authenticity or deepfake fabrication."
            )
        except Exception as e:
            st.warning(f"⚠️ Unable to render neural feature representations: {str(e)}")


def render_export_section(
    loaded: LoadedAudio,
    result: Union[InferenceResult, LongRecordingInferenceResult],
    engine: Any,
) -> None:
    """
    Renders the downloadable PDF report and machine-readable JSON export controls.
    Compiles documentation in-memory on demand without storing data on disk.
    """
    st.markdown(
        """
        <div class="export-card">
            <div class="export-title">Export Analysis Documentation</div>
            <div class="export-desc">
                Download comprehensive documentation of this screening run in publication-quality PDF
                or machine-readable JSON format.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        # Check if reports have already been compiled for this specific analysis result
        export_cache_key = f"{id(result)}_{id(loaded)}"
        if (
            st.session_state.get("export_cache_key") == export_cache_key
            and st.session_state.get("cached_pdf_bytes") is not None
            and st.session_state.get("cached_json_data") is not None
            and st.session_state.get("cached_report_id") is not None
        ):
            pdf_bytes = st.session_state["cached_pdf_bytes"]
            json_data = st.session_state["cached_json_data"]
            report_id = st.session_state["cached_report_id"]
        else:
            # 1. Build canonical validated report payload
            candidate_th = getattr(engine.decision_engine, "threshold", 0.4820)
            engine_meta = getattr(engine, "metadata", None)
            payload = build_report_payload(
                loaded=loaded,
                result=result,
                engine_metadata=engine_meta,
                candidate_threshold=candidate_th,
            )

            # 2. Serialize JSON string
            json_data = export_report_to_json(payload)

            # 3. Generate in-memory PDF with signal visualizations
            is_long = isinstance(result, LongRecordingInferenceResult)
            fig_wave = None
            fig_spec = None
            fig_time = None
            fig_feat = None

            try:
                # Prepare 16k mono array for PDF figures
                if is_long:
                    full_tensor, _ = preprocess_waveform(
                        loaded.waveform,
                        sample_rate=loaded.sample_rate,
                        target_samples=None,
                    )
                    audio_16k_mono = full_tensor.squeeze(0).numpy()
                else:
                    prep_tensor, _ = preprocess_waveform(
                        loaded.waveform,
                        sample_rate=loaded.sample_rate,
                        target_samples=WINDOW_SAMPLES,
                    )
                    audio_16k_mono = prep_tensor.squeeze(0).numpy()

                fig_wave = create_waveform_figure(
                    audio=audio_16k_mono,
                    sample_rate=TARGET_SAMPLE_RATE,
                    title="Audio Waveform (Normalized Amplitude vs. Time)",
                    subtitle=f"Prepared 16,000 Hz Mono Signal ({len(audio_16k_mono):,} samples)",
                )
                fig_spec = create_spectrogram_figure(
                    audio=audio_16k_mono,
                    sample_rate=TARGET_SAMPLE_RATE,
                    title="Audio Spectrogram (Power Spectral Density in dB)",
                )

                if is_long:
                    fig_time = create_timeline_figure(
                        long_result=result,
                        candidate_threshold=candidate_th,
                    )

                # Generate feature view for window 0
                with torch.inference_mode():
                    if is_long:
                        slice_0 = (
                            audio_16k_mono[result.windows[0].window.start_sample : result.windows[0].window.end_sample]
                            if not result.windows[0].window.is_padded
                            else np.pad(audio_16k_mono, (0, WINDOW_SAMPLES - len(audio_16k_mono)), mode="wrap")
                        )
                        slice_t = torch.from_numpy(slice_0.astype(np.float32)).unsqueeze(0).to(engine.device)
                    else:
                        slice_t = prep_tensor.to(engine.device)
                    feat_t = engine.model.extractor(slice_t).detach().cpu()

                fig_feat = create_feature_views_figure(
                    spec_2d_tensor=feat_t,
                    window_label="Window #1 (0.0s – 4.0s @ 16 kHz)",
                )

                pdf_bytes = export_report_to_pdf(
                    payload=payload,
                    waveform_fig=fig_wave,
                    spectrogram_fig=fig_spec,
                    timeline_fig=fig_time,
                    feature_views_fig=fig_feat,
                )

            finally:
                # Explicit figure handle cleanup
                if fig_wave:
                    plt.close(fig_wave)
                if fig_spec:
                    plt.close(fig_spec)
                if fig_time:
                    plt.close(fig_time)
                if fig_feat:
                    plt.close(fig_feat)

                report_id = payload.metadata.report_id
                st.session_state["export_cache_key"] = export_cache_key
                st.session_state["cached_pdf_bytes"] = pdf_bytes
                st.session_state["cached_json_data"] = json_data
                st.session_state["cached_report_id"] = report_id

        # 4. Render Download Action Buttons
        col_pdf, col_json = st.columns(2)
        with col_pdf:
            st.download_button(
                label="📄 Download PDF Analysis Report",
                data=pdf_bytes,
                file_name=f"v7_report_{report_id}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        with col_json:
            st.download_button(
                label="📊 Download JSON Analysis Data",
                data=json_data,
                file_name=f"v7_report_{report_id}.json",
                mime="application/json",
                use_container_width=True,
            )

        st.caption(
            "🔒 **Local Privacy Guarantee:** Export documents are rendered in-memory on demand. "
            "No report files, waveforms, or metadata are stored on disk or transmitted across network connections."
        )

    except Exception as e:
        st.error(f"⚠️ Unable to compile analysis export: {str(e)}")


def render_limitations_and_privacy() -> None:
    """Renders the transparent limitations and local privacy disclosure."""
    st.markdown(
        """
        <div class="info-dual-grid">
            <div class="info-card">
                <div class="info-card-header">
                    <span>🔒</span> Local Processing
                </div>
                <p class="info-card-text">
                    Your recording is processed locally by this application.
                    The app is designed to avoid sending audio to external services.
                    All neural inference and feature extraction execute directly on your machine.
                </p>
            </div>
            <div class="info-card">
                <div class="info-card-header">
                    <span>⚠️</span> Important Limitations
                </div>
                <p class="info-card-text">
                    This tool provides an automated screening result, not definitive proof of authenticity.
                    Noise, compression, recording conditions, and unfamiliar speech-generation methods
                    may affect the result. The candidate threshold (0.4820) is unvalidated for general real-world use.
                </p>
            </div>
        </div>
        <div class="footer-note">
            <strong>🔒 Local Privacy Guarantee:</strong> All audio decoding and neural network inference execute locally on your machine.
            No waveforms, audio files, or metadata are transmitted to external servers or cloud services.<br><br>
            <strong>⚠️ Limitations & Ethical Use:</strong> This application is a machine-learning screening tool designed for research evaluation.
            The V7 model was trained on the ASVspoof 2019 dataset. Acoustic variations caused by strong compression (e.g. messaging apps),
            background noise, heavy reverberation, or microphone anomalies may affect scores.
            Its output should not be treated as definitive legal proof of authenticity or synthetic fabrication.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_methodology_bento_section() -> None:
    """Renders the interactive 4-stage forensic methodology section via MagicBento."""
    st.markdown(
        """
        <div class="methodology-header">
            <h3 class="methodology-title">
                <span>🔬</span> What the Analysis Examines: Forensic Methodology & Architecture
            </h3>
            <p class="methodology-subtitle">
                Explore the verified signal processing and deep neural representations behind the V7 LCNN-2D authenticity pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    from ui.magic_bento import render_forensic_magic_bento
    render_forensic_magic_bento()

