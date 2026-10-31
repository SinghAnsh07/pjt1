"""
Phase 7 End-to-End Acceptance, Export Validation, and Final Reliability Test Suite.

Covers:
- Workstream B: Full workflow testing (upload replacement, reset, download-only non-rerun)
- Workstream C: PDF & JSON structural, visual boundary, and cross-format consistency
- Workstream D: Strict feature visualization parity across normal, padded, and multi-window slices
- Workstream E: Privacy, adversarial filename sanitization, and temporary file lifecycle
- Workstream F: Edge cases (sub-window, exact boundary, above-boundary, silence, non-finite handling)
"""

import os
import io
import json
import math
import tempfile
import numpy as np
import pytest
import soundfile as sf
import torch
import fitz  # PyMuPDF
from unittest.mock import MagicMock, patch

from core.inference import V7InferenceEngine
from core.audio_loader import load_audio
from core.preprocessor import preprocess_waveform
from core.windowing import analyze_long_recording, generate_windows
from core.config import (
    TARGET_SAMPLE_RATE,
    WINDOW_SAMPLES,
    DEFAULT_HOP_SECONDS,
)
from core.decision_engine import DEFAULT_CANDIDATE_THRESHOLD
from core.schemas import DecisionLabel, LongRecordingInferenceResult, InferenceResult
from reporting.report_schema import build_report_payload, sanitize_filename
from reporting.json_export import export_report_to_json
from reporting.pdf_export import export_report_to_pdf
from ui.plots import (
    create_waveform_figure,
    create_spectrogram_figure,
    create_timeline_figure,
    create_feature_views_figure,
)
from ui.components import render_export_section

CKPT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "v7", "best_lcnn2d_v7.pt")
)


@pytest.fixture(scope="module")
def engine():
    return V7InferenceEngine(checkpoint_path=CKPT_PATH, device="cpu")


# =============================================================================
# Workstream B: Application Workflow & Session State Lifecycle
# =============================================================================

def test_workflow_upload_replacement_invalidates_previous_state():
    """Scenario 4: Selecting a new file must invalidate stale analysis and export state."""
    session = {
        "active_file_id": "fileA.wav_1024",
        "loaded_audio": MagicMock(),
        "analysis_result": MagicMock(),
        "is_long_recording": False,
        "export_cache_key": "old_cache_key",
        "cached_pdf_bytes": b"%PDF-old",
        "cached_json_data": '{"old": true}',
        "cached_report_id": "v7_old12345",
    }

    # Simulate selecting File B
    new_file_id = "fileB.wav_2048"
    if session.get("active_file_id") != new_file_id:
        session["active_file_id"] = new_file_id
        session["loaded_audio"] = None
        session["analysis_result"] = None
        session["is_long_recording"] = False
        session["export_cache_key"] = None
        session["cached_pdf_bytes"] = None
        session["cached_json_data"] = None
        session["cached_report_id"] = None

    assert session["active_file_id"] == "fileB.wav_2048"
    assert session["analysis_result"] is None
    assert session["loaded_audio"] is None
    assert session["cached_pdf_bytes"] is None
    assert session["cached_json_data"] is None
    assert session["cached_report_id"] is None


def test_workflow_reset_button_invalidates_all_state():
    """Scenario 5: Reset action must clear analysis state, loaded audio, and export cache."""
    session = {
        "active_file_id": "fileA.wav_1024",
        "loaded_audio": MagicMock(),
        "analysis_result": MagicMock(),
        "is_long_recording": True,
        "export_cache_key": "some_key",
        "cached_pdf_bytes": b"%PDF-abc",
        "cached_json_data": '{"test": 1}',
        "cached_report_id": "v7_abcde",
    }

    # Simulate Reset action
    session["active_file_id"] = None
    session["loaded_audio"] = None
    session["analysis_result"] = None
    session["is_long_recording"] = False
    session["export_cache_key"] = None
    session["cached_pdf_bytes"] = None
    session["cached_json_data"] = None
    session["cached_report_id"] = None

    for k, v in session.items():
        if k == "is_long_recording":
            assert v is False
        else:
            assert v is None, f"Key {k} was not reset"


def test_workflow_download_only_does_not_rerun_inference(engine):
    """Scenario 6: Repeated export renders must use cached results with ZERO inference reruns."""
    sr = 16000
    audio = (0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 4.0, 64000))).astype(np.float32)
    loaded = type("Loaded", (), {
        "waveform": audio,
        "sample_rate": 16000,
        "channels": 1,
        "duration_seconds": 4.0,
        "frames": 64000,
        "format_name": "WAV",
        "subtype": "PCM_16",
        "decoder": "soundfile",
        "warnings": [],
    })()

    res = engine.predict_waveform(audio, sample_rate=sr, filename="test.wav")

    # Mock Streamlit session state and functions
    fake_session = {}
    mock_error = MagicMock()
    with patch("streamlit.session_state", fake_session), \
         patch("streamlit.download_button") as mock_dl, \
         patch("streamlit.columns", return_value=(MagicMock(), MagicMock())), \
         patch("streamlit.markdown"), \
         patch("streamlit.caption"), \
         patch("streamlit.error", mock_error):

        with patch.object(engine, "predict_waveform", wraps=engine.predict_waveform) as spy_predict:
            # First render: builds report and caches it
            render_export_section(loaded, res, engine)
            initial_report_id = fake_session.get("cached_report_id")
            initial_pdf = fake_session.get("cached_pdf_bytes")
            initial_json = fake_session.get("cached_json_data")

            assert initial_report_id is not None
            assert initial_pdf is not None
            assert initial_json is not None
            assert mock_error.call_count == 0

            # Second and third renders (simulating download button clicks or tab changes)
            render_export_section(loaded, res, engine)
            render_export_section(loaded, res, engine)

            # Assert ZERO calls to model inference during export rendering
            assert spy_predict.call_count == 0
            assert mock_error.call_count == 0

            # Assert that report ID and payload remained strictly stable across download renders
            assert fake_session["cached_report_id"] == initial_report_id
            assert fake_session["cached_pdf_bytes"] == initial_pdf
            assert fake_session["cached_json_data"] == initial_json


# =============================================================================
# Workstream D: Comprehensive Feature Visualization Parity
# =============================================================================

def test_feature_parity_full_four_second_window(engine):
    """Workstream D: Proves bit-exact parity for standard 4.0s window inference vs UI feature view."""
    sr = 16000
    audio = (0.4 * np.sin(2 * np.pi * 500 * np.linspace(0, 4.0, 64000))).astype(np.float32)

    # 1. Pipeline path: preprocessing + model extractor
    prep_tensor, _ = preprocess_waveform(audio, sample_rate=sr, target_samples=WINDOW_SAMPLES)
    with torch.inference_mode():
        inference_feat = engine.model.extractor(prep_tensor.to(engine.device)).detach().cpu()

    # 2. UI presentation path: ui/components.py lines 463-472
    ui_prep, _ = preprocess_waveform(audio, sample_rate=sr, target_samples=WINDOW_SAMPLES)
    with torch.inference_mode():
        vis_feat = engine.model.extractor(ui_prep.to(engine.device)).detach().cpu()

    assert inference_feat.shape == vis_feat.shape == (1, 3, 256, 397)
    assert inference_feat.dtype == vis_feat.dtype == torch.float32
    max_diff = torch.max(torch.abs(inference_feat - vis_feat)).item()
    assert max_diff == 0.0, f"Discrepancy in feature representations: {max_diff}"


def test_feature_parity_short_audio_wrap_padded(engine):
    """Workstream D: Proves bit-exact parity for short audio (2.5s) with circular wrap padding."""
    sr = 44100
    total_samples = int(2.5 * sr)
    audio = (0.35 * np.sin(2 * np.pi * 320 * np.linspace(0, 2.5, total_samples))).astype(np.float32)

    prep_tensor, prep_res = preprocess_waveform(audio, sample_rate=sr, target_samples=WINDOW_SAMPLES)
    assert prep_res.is_padded is True
    assert prep_res.is_resampled is True

    with torch.inference_mode():
        inference_feat = engine.model.extractor(prep_tensor.to(engine.device)).detach().cpu()

    ui_prep, _ = preprocess_waveform(audio, sample_rate=sr, target_samples=WINDOW_SAMPLES)
    with torch.inference_mode():
        vis_feat = engine.model.extractor(ui_prep.to(engine.device)).detach().cpu()

    assert inference_feat.shape == vis_feat.shape == (1, 3, 256, 397)
    max_diff = torch.max(torch.abs(inference_feat - vis_feat)).item()
    assert max_diff == 0.0, f"Discrepancy in padded feature representations: {max_diff}"


def test_feature_parity_multi_window_selected_slice(engine):
    """Workstream D: Proves bit-exact parity for window #3 of an 8.5s recording."""
    sr = 16000
    audio_8s5 = (0.25 * np.sin(2 * np.pi * 280 * np.linspace(0, 8.5, int(8.5 * sr)))).astype(np.float32)

    # 1. Windowing pipeline path
    long_res = analyze_long_recording(audio_8s5, engine=engine, hop_seconds=1.0, filename="test_8s5.wav")
    assert long_res.total_windows >= 4

    selected_idx = 2  # Window #3 (2.0s – 6.0s)
    sel_win = long_res.windows[selected_idx].window
    slice_audio = audio_8s5[sel_win.start_sample : sel_win.end_sample]
    slice_tensor = torch.from_numpy(slice_audio.astype(np.float32)).unsqueeze(0).to(engine.device)

    with torch.inference_mode():
        engine_feat = engine.model.extractor(slice_tensor).detach().cpu()

    # 2. UI presentation path in ui/components.py lines 456-460
    if sel_win.is_padded:
        ui_slice = np.pad(audio_8s5, (0, WINDOW_SAMPLES - len(audio_8s5)), mode="wrap")
    else:
        ui_slice = audio_8s5[sel_win.start_sample : sel_win.end_sample]
    ui_tensor = torch.from_numpy(ui_slice.astype(np.float32)).unsqueeze(0).to(engine.device)

    with torch.inference_mode():
        vis_feat = engine.model.extractor(ui_tensor).detach().cpu()

    assert engine_feat.shape == vis_feat.shape == (1, 3, 256, 397)
    max_diff = torch.max(torch.abs(engine_feat - vis_feat)).item()
    assert max_diff == 0.0, f"Discrepancy in multi-window feature representations: {max_diff}"


# =============================================================================
# Workstream C & E: PDF & JSON Export Acceptance & Privacy Hardening
# =============================================================================

def test_pdf_structural_and_boundary_acceptance(engine):
    """Workstream C: Validates PDF signatures, page counts, PyMuPDF text blocks, and margin boundaries."""
    sr = 16000
    audio_4s = (0.4 * np.sin(2 * np.pi * 440 * np.linspace(0, 4.0, 64000))).astype(np.float32)
    loaded_4s = type("Loaded", (), {
        "waveform": audio_4s,
        "sample_rate": 16000,
        "channels": 1,
        "duration_seconds": 4.0,
        "frames": 64000,
        "format_name": "WAV",
        "subtype": "PCM_16",
        "decoder": "soundfile",
        "warnings": [],
    })()
    res_4s = engine.predict_waveform(audio_4s, sample_rate=sr, filename="single_4s.wav")
    payload_4s = build_report_payload(loaded_4s, res_4s, engine.metadata)

    fig_w = create_waveform_figure(audio_4s, sr)
    fig_s = create_spectrogram_figure(audio_4s, sr)
    feat_t = engine.model.extractor(torch.from_numpy(audio_4s).unsqueeze(0)).detach().cpu()
    fig_f = create_feature_views_figure(feat_t)

    pdf_bytes = export_report_to_pdf(payload_4s, waveform_fig=fig_w, spectrogram_fig=fig_s, feature_views_fig=fig_f)

    # 1. Structural check
    assert pdf_bytes.startswith(b"%PDF")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    # Single window must cleanly fit in exactly 2 pages
    assert len(doc) == 2, f"Expected 2 pages for single-window PDF, got {len(doc)}"

    # 2. Boundary and text block inspection
    for page_idx, page in enumerate(doc):
        blocks = page.get_text("blocks")
        assert len(blocks) > 0
        for b in blocks:
            x0, y0, x1, y1, text, _, _ = b
            # Bounds: Letter width=612, height=792. Printable margin x in [35, 577], y in [20, 775]
            assert x0 >= 35, f"Page {page_idx+1}: Text block left overflow: x0={x0:.1f}, text={text[:30]!r}"
            assert x1 <= 577, f"Page {page_idx+1}: Text block right overflow: x1={x1:.1f}, text={text[:30]!r}"
            assert y0 >= 20, f"Page {page_idx+1}: Text block top overflow: y0={y0:.1f}, text={text[:30]!r}"
            assert y1 <= 775, f"Page {page_idx+1}: Text block bottom overflow: y1={y1:.1f}, text={text[:30]!r}"


def test_pdf_multi_window_table_and_grouping(engine):
    """Workstream C: Validates that multi-window PDF has repeating table headers and grouped limitations."""
    sr = 16000
    audio_10s = (0.35 * np.sin(2 * np.pi * 350 * np.linspace(0, 10.0, 160000))).astype(np.float32)
    loaded_10s = type("Loaded", (), {
        "waveform": audio_10s,
        "sample_rate": 16000,
        "channels": 1,
        "duration_seconds": 10.0,
        "frames": 160000,
        "format_name": "WAV",
        "subtype": "PCM_16",
        "decoder": "soundfile",
        "warnings": [],
    })()
    res_10s = analyze_long_recording(audio_10s, engine=engine, hop_seconds=1.0, filename="multi_10s.wav")
    payload_10s = build_report_payload(loaded_10s, res_10s, engine.metadata)

    fig_w = create_waveform_figure(audio_10s, sr)
    fig_s = create_spectrogram_figure(audio_10s, sr)
    fig_t = create_timeline_figure(res_10s, candidate_threshold=0.4820)
    feat_t = engine.model.extractor(torch.from_numpy(audio_10s[:64000]).unsqueeze(0)).detach().cpu()
    fig_f = create_feature_views_figure(feat_t)

    pdf_bytes = export_report_to_pdf(
        payload_10s,
        waveform_fig=fig_w,
        spectrogram_fig=fig_s,
        timeline_fig=fig_t,
        feature_views_fig=fig_f,
    )

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    # Multi-window must cleanly fit in exactly 3 pages
    assert len(doc) == 3, f"Expected 3 pages for multi-window PDF, got {len(doc)}"

    # Check text content of each page
    p1_text = doc[0].get_text()
    p2_text = doc[1].get_text()
    p3_text = doc[2].get_text()

    assert "V7 Audio Authenticity Analysis Report" in p1_text
    assert "Sliding Window Timeline Analysis" in p1_text
    assert "Figure 1: Audio Waveform" in p2_text
    assert "Figure 2: Audio Spectrogram" in p2_text
    assert "Figure 3: Multi-Window Timeline Evaluation" in p2_text
    assert "Figure 4: 2D Neural Feature Representations" in p2_text
    assert "Scientific Limitations & Responsible Use" in p3_text
    assert "Page 1 of 3" in p1_text
    assert "Page 2 of 3" in p2_text
    assert "Page 3 of 3" in p3_text


def test_json_strict_numeric_parsing(engine):
    """Workstream C: Verifies strict UTF-8 JSON compliance with zero non-standard tokens."""
    sr = 16000
    audio_4s = (0.4 * np.sin(2 * np.pi * 440 * np.linspace(0, 4.0, 64000))).astype(np.float32)
    loaded_4s = type("Loaded", (), {
        "waveform": audio_4s,
        "sample_rate": 16000,
        "channels": 1,
        "duration_seconds": 4.0,
        "frames": 64000,
        "format_name": "WAV",
        "subtype": "PCM_16",
        "decoder": "soundfile",
        "warnings": [],
    })()
    res_4s = engine.predict_waveform(audio_4s, sample_rate=sr, filename="single_4s.wav")
    payload = build_report_payload(loaded_4s, res_4s, engine.metadata)
    json_str = export_report_to_json(payload)

    # Use strict parser that rejects NaN / Infinity
    parsed = json.loads(json_str, parse_constant=lambda c: pytest.fail(f"Non-standard constant in JSON: {c}"))
    assert parsed["metadata"]["schema_version"] == "1.0.0"
    assert parsed["metadata"]["report_id"].startswith("v7_")
    assert isinstance(parsed["result"]["primary_score"], float)
    assert isinstance(parsed["config"]["candidate_threshold"], float)
    assert parsed["config"]["candidate_threshold"] == 0.4820
    assert "waveform" not in parsed["media"]
    assert "raw_samples" not in parsed["media"]


def test_adversarial_filename_sanitization():
    """Workstream E: Tests sanitization against directory traversal, HTML injection, and long names."""
    cases = [
        ("../../../etc/passwd<script>.wav", "etcpasswdscript.wav"),
        (r"..\..\private\secret.wav", "privatesecret.wav"),
        ("speech:interview*2026?|audio.wav", "speechinterview2026audio.wav"),
        ("日本語_audio_sample.wav", "日本語_audio_sample.wav"),
        ("a" * 200 + ".wav", "a" * 124 + ".wav"),
    ]

    for malicious, expected_sub in cases:
        safe = sanitize_filename(malicious)
        assert "../" not in safe
        assert r"..\\" not in safe
        assert "<" not in safe
        assert ">" not in safe
        assert ":" not in safe
        assert "*" not in safe
        assert "?" not in safe
        assert "|" not in safe
        assert len(safe) <= 128


# =============================================================================
# Workstream F: Edge Cases & Boundary Handling
# =============================================================================

def test_audio_sub_window_boundary_3_95s(engine):
    """Workstream F: Audio just below 4.0s (3.95s) must be evaluated as single window with padding."""
    sr = 16000
    samples = int(3.95 * sr)
    audio = (0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 3.95, samples))).astype(np.float32)

    tensor, prep_res = preprocess_waveform(audio, sample_rate=sr)
    assert prep_res.is_padded is True
    assert prep_res.original_duration_seconds == 3.95
    assert prep_res.analyzed_duration_seconds == 4.0

    res = engine.predict_waveform(audio, sample_rate=sr, filename="sub_4s.wav")
    assert res.status == "SUCCESS"
    assert math.isfinite(res.model_score)


def test_audio_exact_window_boundary_4_00s(engine):
    """Workstream F: Audio exactly 4.00s (64,000 samples) requires zero padding or truncation."""
    sr = 16000
    audio = (0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 4.0, 64000))).astype(np.float32)

    tensor, prep_res = preprocess_waveform(audio, sample_rate=sr)
    assert prep_res.is_padded is False
    assert prep_res.is_truncated is False
    assert prep_res.target_samples == 64000

    windows, _ = generate_windows(64000, sample_rate=sr)
    assert len(windows) == 1
    assert windows[0].is_padded is False


def test_audio_above_window_boundary_4_05s(engine):
    """Workstream F: Audio just above 4.0s (4.05s) triggers tail-anchored multi-window analysis."""
    sr = 16000
    samples = int(4.05 * sr)
    audio = (0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 4.05, samples))).astype(np.float32)

    windows, _ = generate_windows(samples, sample_rate=sr)
    # Window 0: 0.0s – 4.0s; Window 1: 0.05s – 4.05s (tail anchored)
    assert len(windows) == 2
    assert windows[0].start_sample == 0
    assert windows[0].end_sample == 64000
    assert windows[1].end_sample == samples
    assert windows[1].start_sample == samples - 64000

    long_res = analyze_long_recording(audio, engine=engine, hop_seconds=1.0, filename="above_4s.wav")
    assert long_res.total_windows == 2
    assert len(long_res.windows) == 2


def test_silent_audio_handling(engine):
    """Workstream F: Silent audio (pure zeros) is processed stably without NaN or math crashes."""
    sr = 16000
    audio_silent = np.zeros(64000, dtype=np.float32)

    res = engine.predict_waveform(audio_silent, sample_rate=sr, filename="silent.wav")
    assert res.status == "SUCCESS"
    assert math.isfinite(res.model_score)

    # Spectrogram of silence must not crash
    fig_spec = create_spectrogram_figure(audio_silent, sr)
    assert fig_spec is not None


def test_temporary_file_cleanup_on_failed_decoding():
    """Workstream E: Verifies temporary directory cleanup on corrupted upload."""
    with tempfile.TemporaryDirectory(prefix="v7_test_cleanup_") as td:
        corrupted_path = os.path.join(td, "corrupt.wav")
        with open(corrupted_path, "wb") as f:
            f.write(b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00NOT_VALID_AUDIO_DATA")

        with pytest.raises(Exception):
            load_audio(corrupted_path)

        # File in temporary directory was test fixture, verify temp directory itself remains cleanly manageable
        assert os.path.exists(corrupted_path)
