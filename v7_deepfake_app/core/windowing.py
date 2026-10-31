"""
Long-Recording Window Generation and Temporal Inference Module.

Provides deterministic sliding window segmentation and per-window deepfake
evaluation for audio files exceeding the 4.0s model window, with 100% temporal
coverage and transparent descriptive summarization.
"""

import time
import numpy as np
import torch
from typing import List, Optional, Tuple

from core.schemas import (
    WindowMetadata,
    WindowInferenceResult,
    LongRecordingInferenceResult,
    DecisionLabel,
)
from core.inference import V7InferenceEngine
from core.config import (
    TARGET_SAMPLE_RATE,
    WINDOW_SAMPLES,
    WINDOW_DURATION_SECONDS,
    DEFAULT_HOP_SECONDS,
    DEFAULT_HOP_SAMPLES,
    MAX_WINDOWS_COUNT,
)
from core.exceptions import WindowingError, InvalidWaveformError


def generate_windows(
    total_samples: int,
    sample_rate: int = TARGET_SAMPLE_RATE,
    window_samples: int = WINDOW_SAMPLES,
    hop_samples: int = DEFAULT_HOP_SAMPLES,
    max_windows: int = MAX_WINDOWS_COUNT,
) -> Tuple[List[WindowMetadata], List[str]]:
    """
    Computes exact, non-overlapping or overlapping window boundaries across an audio file.

    Policy:
      1. Short Audio (< 64,000 samples): Exactly 1 window marked is_padded=True.
      2. Regular Windows: Slices start at 0, hop, 2*hop... while start + window <= total.
      3. Tail Alignment: If regular windows do not reach total_samples, anchors a final
         window at [total_samples - window_samples : total_samples] to ensure 100% coverage,
         skipping if it perfectly coincides with the previous window.
      4. Limits: Enforces max_windows safeguard.

    Args:
        total_samples: Length of the 1D waveform in samples
        sample_rate: Sampling frequency (default: 16000 Hz)
        window_samples: Window length (default: 64000 samples / 4.0s)
        hop_samples: Stride length (default: 16000 samples / 1.0s)
        max_windows: Upper threshold on generated windows

    Returns:
        Tuple of (List[WindowMetadata], List[str] warnings)
    """
    if total_samples <= 0:
        raise WindowingError(f"Cannot generate windows for {total_samples} samples.")

    warnings: List[str] = []
    windows: List[WindowMetadata] = []

    # Case 1: Short audio (< 4.0s)
    if total_samples < window_samples:
        dur = float(total_samples) / float(sample_rate)
        windows.append(
            WindowMetadata(
                window_index=0,
                start_time_seconds=0.0,
                end_time_seconds=round(dur, 4),
                start_sample=0,
                end_sample=total_samples,
                is_padded=True,
            )
        )
        warnings.append(
            f"Recording duration ({dur:.2f}s) is shorter than model window (4.0s); single padded window generated."
        )
        return windows, warnings

    # Case 2: Standard sliding window progression
    current_start = 0
    idx = 0

    while current_start + window_samples <= total_samples:
        current_end = current_start + window_samples
        windows.append(
            WindowMetadata(
                window_index=idx,
                start_time_seconds=round(current_start / sample_rate, 4),
                end_time_seconds=round(current_end / sample_rate, 4),
                start_sample=current_start,
                end_sample=current_end,
                is_padded=False,
            )
        )
        idx += 1
        current_start += hop_samples

        if len(windows) >= max_windows:
            warnings.append(
                f"Window count reached maximum configured limit ({max_windows}); subsequent audio was omitted."
            )
            return windows, warnings

    # Case 3: Tail anchoring for 100% coverage
    last_end = windows[-1].end_sample if windows else 0
    if last_end < total_samples and len(windows) < max_windows:
        tail_start = total_samples - window_samples
        # Avoid duplicate if the last window already had this exact start
        if tail_start != windows[-1].start_sample:
            windows.append(
                WindowMetadata(
                    window_index=len(windows),
                    start_time_seconds=round(tail_start / sample_rate, 4),
                    end_time_seconds=round(total_samples / sample_rate, 4),
                    start_sample=tail_start,
                    end_sample=total_samples,
                    is_padded=False,
                )
            )

    return windows, warnings


def analyze_long_recording(
    audio_16k_mono: np.ndarray,
    engine: V7InferenceEngine,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    max_windows: int = MAX_WINDOWS_COUNT,
    filename: Optional[str] = None,
) -> LongRecordingInferenceResult:
    """
    Executes multi-window deepfake analysis across an entire 16 kHz mono audio recording.

    Args:
        audio_16k_mono: 1D floating-point NumPy array sampled at 16 kHz
        engine: Initialized, preloaded V7InferenceEngine instance
        hop_seconds: Hop step between windows in seconds (default: 1.0s)
        max_windows: Maximum number of windows to analyze
        filename: Optional filename identifier

    Returns:
        LongRecordingInferenceResult: Comprehensive timeline result
    """
    t_start = time.perf_counter()

    if not isinstance(audio_16k_mono, np.ndarray) or audio_16k_mono.ndim != 1:
        raise InvalidWaveformError("analyze_long_recording expects a 1D mono NumPy array.")

    total_samples = len(audio_16k_mono)
    total_duration = float(total_samples) / float(TARGET_SAMPLE_RATE)
    hop_samples = int(round(hop_seconds * TARGET_SAMPLE_RATE))

    windows, warnings = generate_windows(
        total_samples=total_samples,
        sample_rate=TARGET_SAMPLE_RATE,
        window_samples=WINDOW_SAMPLES,
        hop_samples=hop_samples,
        max_windows=max_windows,
    )

    window_results: List[WindowInferenceResult] = []
    scores: List[float] = []

    # Run inference across windows sequentially reusing the loaded model
    for win in windows:
        t_win_start = time.perf_counter()

        if win.is_padded:
            pad_amount = WINDOW_SAMPLES - total_samples
            slice_audio = np.pad(audio_16k_mono, (0, pad_amount), mode="wrap")
        else:
            slice_audio = audio_16k_mono[win.start_sample : win.end_sample]

        # Convert to model input tensor [1, 64000]
        slice_tensor = torch.from_numpy(slice_audio.astype(np.float32)).unsqueeze(0).to(engine.device)

        with torch.inference_mode():
            spec_2d = engine.model.extractor(slice_tensor)
            emb = engine.model.backbone(spec_2d)
            raw_score = float(engine.model.oc_loss.get_score(emb)[0].item())

        scores.append(raw_score)
        dec_result = engine.decision_engine.decide(raw_score)
        win_duration_ms = (time.perf_counter() - t_win_start) * 1000.0

        window_results.append(
            WindowInferenceResult(
                window=win,
                model_score=round(raw_score, 4),
                decision=dec_result,
                timing_ms=round(win_duration_ms, 2),
                warnings=dec_result.warnings,
            )
        )

    t_total_ms = (time.perf_counter() - t_start) * 1000.0

    # Descriptive summary computation (Transparent; no fake probability)
    real_count = sum(1 for w in window_results if w.decision.prediction == DecisionLabel.REAL)
    fake_count = sum(1 for w in window_results if w.decision.prediction == DecisionLabel.FAKE)
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0
    mean_score = float(np.mean(scores)) if scores else 0.0

    covered_duration = max(w.window.end_time_seconds for w in window_results) if window_results else 0.0
    unanalyzed_duration = max(0.0, total_duration - covered_duration)

    flagged_pct = (fake_count / len(window_results)) * 100.0 if window_results else 0.0
    warnings.append(
        f"Timeline summary: {real_count}/{len(window_results)} windows classified Real, "
        f"{fake_count}/{len(window_results)} windows classified Fake ({flagged_pct:.1f}% flagged). "
        f"Note: 4.0s sliding windows overlap by 75% ({hop_seconds:.1f}s hop). Window scores reflect local segment "
        f"similarity to the bona fide centroid and do not constitute standalone proof of splicing or manipulation."
    )

    return LongRecordingInferenceResult(
        filename=filename,
        total_windows=len(window_results),
        total_audio_duration_seconds=round(total_duration, 4),
        covered_duration_seconds=round(covered_duration, 4),
        unanalyzed_duration_seconds=round(unanalyzed_duration, 4),
        summary_real_windows=real_count,
        summary_fake_windows=fake_count,
        min_score=round(min_score, 4),
        max_score=round(max_score, 4),
        mean_score=round(mean_score, 4),
        windows=window_results,
        total_processing_time_ms=round(t_total_ms, 2),
        system_warnings=warnings,
    )
