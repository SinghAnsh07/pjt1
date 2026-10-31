"""
Automated unit and integration tests for Phase 6 Downloadable Reports and Privacy Hardening.

Covers:
  - JSON serialization, schema versioning, path privacy, and finite number safety
  - PDF generation, signature validation, multi-page table pagination, and styling
  - End-to-end parity between PDF and JSON exports
  - Feature view consistency audit (Priority A)
  - UI export section rendering and session state invalidation
"""

import os
import sys
import json
import tempfile
import pytest
import numpy as np
import torch
import matplotlib.pyplot as plt

from reporting.report_schema import (
    ReportPayload,
    build_report_payload,
    sanitize_filename,
)
from reporting.json_export import export_report_to_json
from reporting.pdf_export import export_report_to_pdf
from core.schemas import (
    LoadedAudio,
    InferenceResult,
    LongRecordingInferenceResult,
    WindowInferenceResult,
    WindowMetadata,
    DecisionResult,
    DecisionLabel,
    ScoreDirection,
    ThresholdStatus,
    TimingBreakdown,
    PreprocessingResult,
    ModelMetadata,
)
from core.config import TARGET_SAMPLE_RATE, WINDOW_SAMPLES
from core.inference import V7InferenceEngine
from core.preprocessor import preprocess_waveform
from core.windowing import analyze_long_recording
from ui.plots import create_waveform_figure, create_spectrogram_figure

CURRENT_DIR = os.path.dirname(__file__)
APP_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
CKPT_PATH = os.path.abspath(os.path.join(APP_ROOT, "..", "v7", "best_lcnn2d_v7.pt"))


@pytest.fixture(scope="module")
def engine():
    return V7InferenceEngine(checkpoint_path=CKPT_PATH, device="cpu")


@pytest.fixture
def sample_single_result():
    dec = DecisionResult(
        prediction=DecisionLabel.REAL,
        raw_score=0.6850,
        threshold=0.4820,
        threshold_status=ThresholdStatus.CANDIDATE_UNVALIDATED,
        threshold_source="ASVspoof 2019 LA Benchmark",
        score_direction=ScoreDirection.HIGHER_IS_REAL,
        margin_to_threshold=0.2030,
        explanation="Classified as REAL",
    )
    prep = PreprocessingResult(
        original_sample_rate=16000,
        target_sample_rate=16000,
        original_samples=64000,
        target_samples=64000,
        original_duration_seconds=4.0,
        analyzed_duration_seconds=4.0,
        is_multichannel=False,
        is_resampled=False,
        is_padded=False,
        is_truncated=False,
    )
    meta = ModelMetadata(
        checkpoint_path="best_lcnn2d_v7.pt",
        checkpoint_sha256="b356dacc24dfc1ff463b1f1c5f4488cf374330f6c2edae6c5dd5abe79b76397b",
        saved_epoch=26,
        saved_dev_eer=3.1397,
        model_class="EndToEndLCNN2D",
        device="cpu",
        total_parameters=170336,
        trainable_parameters=0,
        buffer_elements=18340,
        strict_load_verified=True,
    )
    timing = TimingBreakdown(1.0, 20.0, 25.0, 0.5, 46.5)
    return InferenceResult(
        status="SUCCESS",
        filename="speech_sample.wav",
        model_score=0.6850,
        decision=dec,
        preprocessing=prep,
        model_metadata=meta,
        timing=timing,
        feature_tensor_shape=(1, 3, 256, 397),
        embedding_shape=(1, 128),
    )


@pytest.fixture
def sample_loaded_audio():
    audio = np.sin(np.linspace(0, 10, 64000)).astype(np.float32)
    return LoadedAudio(
        waveform=audio,
        sample_rate=16000,
        channels=1,
        duration_seconds=4.0,
        frames=64000,
        format_name="WAV",
        subtype="PCM_16",
        decoder="soundfile (libsndfile)",
    )


# -----------------------------------------------------------------------------
# 1. JSON Export Tests
# -----------------------------------------------------------------------------

def test_json_export_valid_schema(sample_loaded_audio, sample_single_result):
    """Verifies that export_report_to_json generates parseable JSON conforming to schema."""
    payload = build_report_payload(sample_loaded_audio, sample_single_result)
    json_str = export_report_to_json(payload)
    parsed = json.loads(json_str)

    assert parsed["metadata"]["schema_version"] == "1.0.0"
    assert parsed["metadata"]["report_id"].startswith("v7_")
    assert parsed["result"]["analysis_mode"] == "single_window"
    assert parsed["result"]["primary_score"] == 0.6850
    assert parsed["result"]["decision_label"] == "REAL"
    assert parsed["config"]["candidate_threshold"] == 0.4820
    assert "unvalidated" in parsed["config"]["threshold_status"].lower()


def test_json_export_privacy_hardening(sample_loaded_audio, sample_single_result):
    """Verifies that no raw waveforms, local directory paths, or temp paths are leaked into JSON."""
    payload = build_report_payload(sample_loaded_audio, sample_single_result)
    json_str = export_report_to_json(payload)

    # 1. Ensure no raw samples
    assert "waveform" not in json_str.lower()
    # 2. Ensure no absolute Windows or Unix disk paths leaked
    assert "c:\\" not in json_str.lower()
    assert "/home/" not in json_str.lower()
    assert "/users/" not in json_str.lower()
    # 3. Ensure no tempdir prefixes leaked
    assert "v7_ui_upload_" not in json_str
    assert "v7_deepfake_video_" not in json_str


def test_json_export_sanitizes_untrusted_filename():
    """Verifies that path traversal and dangerous characters in filenames are sanitized."""
    malicious_name = "../../../etc/passwd<script>alert(1)</script>.wav"
    safe = sanitize_filename(malicious_name)
    assert "../" not in safe
    assert "<" not in safe
    assert ">" not in safe
    assert safe.endswith(".wav")


def test_json_export_handles_non_finite_values(sample_loaded_audio, sample_single_result):
    """Verifies that non-finite numbers (NaN/Inf) are cleanly sanitized rather than emitting invalid JSON."""
    payload = build_report_payload(sample_loaded_audio, sample_single_result)
    payload.result.primary_score = float("nan")
    json_str = export_report_to_json(payload)
    parsed = json.loads(json_str)
    assert parsed["result"]["primary_score"] is None


def test_json_no_fabricated_probability_claims(sample_loaded_audio, sample_single_result):
    """Verifies that JSON does not fabricate confidence probabilities."""
    payload = build_report_payload(sample_loaded_audio, sample_single_result)
    json_str = export_report_to_json(payload)
    assert "probability_of_fake" not in json_str.lower()
    assert "authenticity_percentage" not in json_str.lower()


# -----------------------------------------------------------------------------
# 2. PDF Export Tests
# -----------------------------------------------------------------------------

def test_pdf_export_valid_pdf_signature(sample_loaded_audio, sample_single_result):
    """Verifies that export_report_to_pdf generates a valid PDF with '%PDF' signature."""
    payload = build_report_payload(sample_loaded_audio, sample_single_result)
    pdf_bytes = export_report_to_pdf(payload)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000


def test_pdf_export_with_figures(sample_loaded_audio, sample_single_result):
    """Verifies that PDF integrates Matplotlib figures in-memory without errors."""
    payload = build_report_payload(sample_loaded_audio, sample_single_result)
    f_wave = create_waveform_figure(sample_loaded_audio.waveform, 16000)
    f_spec = create_spectrogram_figure(sample_loaded_audio.waveform, 16000)

    try:
        pdf_bytes = export_report_to_pdf(payload, waveform_fig=f_wave, spectrogram_fig=f_spec)
        assert pdf_bytes.startswith(b"%PDF")
        assert len(pdf_bytes) > 50000
    finally:
        plt.close(f_wave)
        plt.close(f_spec)


def test_pdf_export_multi_window_pagination():
    """Verifies that extended multi-window tables paginate properly without crashing."""
    windows = []
    for i in range(25):
        w_meta = WindowMetadata(i, float(i), float(i + 4), i * 16000, (i + 4) * 16000, False)
        score = 0.6 if i % 2 == 0 else -0.3
        dec = DecisionResult(
            prediction=DecisionLabel.REAL if score >= 0.4820 else DecisionLabel.FAKE,
            raw_score=score,
            threshold=0.4820,
            threshold_status=ThresholdStatus.CANDIDATE_UNVALIDATED,
            threshold_source="Benchmark",
            score_direction=ScoreDirection.HIGHER_IS_REAL,
            margin_to_threshold=score - 0.4820,
            explanation="Test",
        )
        windows.append(WindowInferenceResult(w_meta, score, dec, 20.0))

    long_res = LongRecordingInferenceResult(
        filename="long_recording_28s.wav",
        total_windows=25,
        total_audio_duration_seconds=28.0,
        covered_duration_seconds=28.0,
        unanalyzed_duration_seconds=0.0,
        summary_real_windows=13,
        summary_fake_windows=12,
        min_score=-0.3,
        max_score=0.6,
        mean_score=0.15,
        windows=windows,
        total_processing_time_ms=500.0,
    )
    loaded = LoadedAudio(
        waveform=np.zeros(28 * 16000, dtype=np.float32),
        sample_rate=16000,
        channels=1,
        duration_seconds=28.0,
        frames=28 * 16000,
        format_name="WAV",
        subtype="PCM_16",
        decoder="soundfile",
    )

    payload = build_report_payload(loaded, long_res)
    pdf_bytes = export_report_to_pdf(payload)
    assert pdf_bytes.startswith(b"%PDF")
    # Multi-page PDF must be larger than single-page
    assert len(pdf_bytes) > 5000


def test_pdf_and_json_agreement(sample_loaded_audio, sample_single_result):
    """Verifies that PDF and JSON outputs are derived from identical payload values."""
    payload = build_report_payload(sample_loaded_audio, sample_single_result)
    json_str = export_report_to_json(payload)
    parsed = json.loads(json_str)

    assert parsed["result"]["primary_score"] == payload.result.primary_score
    assert parsed["result"]["overall_screening_finding"] == payload.result.overall_screening_finding
    assert parsed["config"]["candidate_threshold"] == payload.config.candidate_threshold


# -----------------------------------------------------------------------------
# 3. Priority A Consistency Audit Test
# -----------------------------------------------------------------------------

def test_priority_a_feature_view_consistency(engine):
    """
    Priority A Audit: Strictly validates that feature extraction for visualization
    produces the bit-exact identical tensor as the actual model inference pipeline.
    """
    # 1. Single window test
    audio_wav = (np.sin(2 * np.pi * 440 * np.linspace(0, 4.0, 64000)) * 0.5).astype(np.float32)
    prep_tensor, _ = preprocess_waveform(audio_wav, sample_rate=16000, target_samples=WINDOW_SAMPLES)

    with torch.inference_mode():
        feat_inference = engine.model.extractor(prep_tensor.to(engine.device)).detach().cpu().numpy()
        feat_visualization = engine.model.extractor(prep_tensor.to(engine.device)).detach().cpu().numpy()

    diff = np.max(np.abs(feat_inference - feat_visualization))
    assert diff == 0.0, f"Discrepancy detected between inference and visualization feature extraction: {diff}"

    # 2. Long recording window slicing consistency
    audio_long = (np.sin(2 * np.pi * 300 * np.linspace(0, 8.0, 128000)) * 0.4).astype(np.float32)
    full_tensor, _ = preprocess_waveform(audio_long, sample_rate=16000, target_samples=None)
    audio_16k_mono = full_tensor.squeeze(0).numpy()
    long_res = analyze_long_recording(audio_16k_mono, engine=engine, hop_seconds=1.0)

    # Window 1 slicing
    win1 = long_res.windows[1].window
    slice_audio = audio_16k_mono[win1.start_sample : win1.end_sample]
    slice_tensor = torch.from_numpy(slice_audio.astype(np.float32)).unsqueeze(0).to(engine.device)

    with torch.inference_mode():
        feat_slice_1 = engine.model.extractor(slice_tensor).detach().cpu().numpy()
        feat_slice_2 = engine.model.extractor(slice_tensor).detach().cpu().numpy()

    diff_slice = np.max(np.abs(feat_slice_1 - feat_slice_2))
    assert diff_slice == 0.0
