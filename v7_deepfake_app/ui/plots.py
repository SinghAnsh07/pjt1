"""
Defensive plotting and signal visualization module for V7 Audio Authenticity Analyzer.

Provides clean, headless Matplotlib figure generators for:
  1. Audio Waveform (with transient-preserving min/max envelope decimation)
  2. Audio Spectrogram (via scipy.signal.spectrogram in dB scale)
  3. Analysis Timeline (per-window cosine similarity scores vs. candidate threshold)
  4. Neural Feature Views (2D ST-FrFT and Wavelet Scalogram representations)

All functions are pure, non-mutating, deterministic, and safe against silent/NaN inputs.
"""

import math
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
import scipy.signal as signal
import torch

# Ensure headless Agg backend for thread safety and zero GUI dependencies
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from core.schemas import (
    LongRecordingInferenceResult,
    DecisionLabel,
)
from core.decision_engine import DEFAULT_CANDIDATE_THRESHOLD
from core.config import (
    TARGET_SAMPLE_RATE,
    WINDOW_SAMPLES,
)

# Shared forensic dark-theme color palette
THEME_BG = "#0C0F14"         # Deep workspace background
PANEL_BG = "#121720"         # Primary surface
GRID_COLOR = "#293241"       # Border / grid line
TEXT_MAIN = "#F3F5F8"        # Slate 50
TEXT_MUTED = "#A6B0C0"       # Slate 400
ACCENT_BLUE = "#8B9CFF"      # Primary accent (indigo-blue)
ACCENT_TEAL = "#65D6C4"      # Secondary accent (teal)
ACCENT_GREEN = "#4CC9A6"     # Positive result (soft emerald)
ACCENT_AMBER = "#E8B86D"     # Caution / threshold (warm amber)
ACCENT_RED = "#F07878"       # Divergent anomaly (coral red)


def apply_dark_theme(fig: plt.Figure, ax: plt.Axes) -> None:
    """Applies cohesive dark slate styling to Matplotlib figures and axes."""
    fig.patch.set_facecolor(THEME_BG)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.xaxis.label.set_color(TEXT_MUTED)
    ax.yaxis.label.set_color(TEXT_MUTED)
    ax.title.set_color(TEXT_MAIN)
    ax.grid(True, linestyle="--", linewidth=0.5, color=GRID_COLOR, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def compute_waveform_envelope(
    audio: np.ndarray,
    max_points: int = 4000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes a deterministic min/max envelope representation of an audio signal.

    Preserves transient speech spikes and loud bursts without aliasing or smoothing,
    preventing DOM bloat when rendering long recordings.

    Args:
        audio: 1D NumPy array of audio samples
        max_points: Maximum number of temporal envelope bins to generate

    Returns:
        Tuple of (time_coordinates, min_envelope, max_envelope)
    """
    if audio.ndim != 1:
        raise ValueError(f"Expected 1D audio array, got shape {audio.shape}")

    total_samples = len(audio)
    if total_samples == 0:
        return np.array([0.0]), np.array([0.0]), np.array([0.0])

    if total_samples <= max_points:
        # For short signals, min and max envelope are identical to the signal
        t = np.arange(total_samples, dtype=np.float32)
        return t, audio.copy(), audio.copy()

    # Block decimation into max_points intervals
    block_size = int(math.ceil(total_samples / max_points))
    num_blocks = int(math.ceil(total_samples / block_size))
    pad_len = num_blocks * block_size - total_samples

    if pad_len > 0:
        padded = np.pad(audio, (0, pad_len), mode="edge")
    else:
        padded = audio

    blocks = padded.reshape(num_blocks, block_size)
    mins = np.min(blocks, axis=1)
    maxs = np.max(blocks, axis=1)
    t = np.arange(num_blocks, dtype=np.float32) * block_size

    return t, mins, maxs


def create_waveform_figure(
    audio: np.ndarray,
    sample_rate: int,
    title: str = "Audio Waveform",
    subtitle: Optional[str] = None,
    max_display_points: int = 4000,
) -> plt.Figure:
    """
    Generates a dark-themed Matplotlib waveform plot.

    Args:
        audio: 1D or 2D audio array (if 2D, downmixes to mono non-mutatingly for display)
        sample_rate: Audio sampling frequency in Hz
        title: Main chart title
        subtitle: Optional descriptive subtitle
        max_display_points: Upper limit on rendered envelope points

    Returns:
        matplotlib.figure.Figure (Caller is responsible for plt.close)
    """
    # Non-mutating mono conversion for multichannel display
    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio_1d = np.mean(audio, axis=0)
        else:
            audio_1d = np.mean(audio, axis=1)
    elif audio.ndim == 1:
        audio_1d = audio
    else:
        audio_1d = audio.flatten()

    # Ensure finite samples
    if not np.all(np.isfinite(audio_1d)):
        audio_1d = np.nan_to_num(audio_1d, nan=0.0, posinf=1.0, neginf=-1.0)

    total_samples = len(audio_1d)
    duration = float(total_samples) / float(max(1, sample_rate))

    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=100)
    apply_dark_theme(fig, ax)

    if total_samples <= max_display_points:
        t = np.linspace(0, duration, total_samples, endpoint=False)
        ax.plot(t, audio_1d, color=ACCENT_BLUE, linewidth=0.8, alpha=0.9, label="Signal Amplitude")
    else:
        t_indices, mins, maxs = compute_waveform_envelope(audio_1d, max_points=max_display_points)
        t = t_indices / float(sample_rate)
        ax.fill_between(t, mins, maxs, color=ACCENT_BLUE, alpha=0.6, label="Peak Envelope (Decimated)")
        # Subtle center line
        ax.plot(t, (mins + maxs) / 2.0, color=ACCENT_TEAL, linewidth=0.5, alpha=0.7)

    # Format axis limits and labels
    max_amp = float(np.max(np.abs(audio_1d))) if len(audio_1d) > 0 else 0.0
    y_bound = max(0.1, max_amp * 1.08)
    ax.set_ylim(-y_bound, y_bound)
    ax.set_xlim(0, max(0.01, duration))

    ax.set_xlabel("Time (seconds)", fontsize=9.5)
    ax.set_ylabel("Amplitude", fontsize=9.5)

    full_title = title
    if subtitle:
        full_title += f"\n{subtitle}"
    ax.set_title(full_title, fontsize=11, fontweight=600, pad=8)
    ax.legend(loc="upper right", facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_MUTED, fontsize=8)

    fig.tight_layout()
    return fig


def create_spectrogram_figure(
    audio: np.ndarray,
    sample_rate: int,
    title: str = "Audio Spectrogram",
    max_time_bins: int = 1000,
) -> plt.Figure:
    """
    Computes and plots a conventional audio spectrogram in dB scale.

    Args:
        audio: 1D or 2D audio array
        sample_rate: Sampling frequency in Hz
        title: Main chart title
        max_time_bins: Upper limit on rendered temporal slices

    Returns:
        matplotlib.figure.Figure (Caller is responsible for plt.close)
    """
    # Non-mutating mono conversion
    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio_1d = np.mean(audio, axis=0)
        else:
            audio_1d = np.mean(audio, axis=1)
    elif audio.ndim == 1:
        audio_1d = audio
    else:
        audio_1d = audio.flatten()

    if not np.all(np.isfinite(audio_1d)):
        audio_1d = np.nan_to_num(audio_1d, nan=0.0, posinf=1.0, neginf=-1.0)

    total_samples = len(audio_1d)
    if total_samples == 0:
        audio_1d = np.zeros(256, dtype=np.float32)

    # Adaptive FFT parameter selection based on signal length
    nperseg = min(512, len(audio_1d))
    if nperseg < 64:
        nperseg = 64
        audio_1d = np.pad(audio_1d, (0, 64 - len(audio_1d)), mode="constant")

    noverlap = nperseg // 2

    # Decimate long recordings to keep max_time_bins bounded
    expected_bins = (len(audio_1d) - noverlap) // (nperseg - noverlap)
    if expected_bins > max_time_bins:
        hop = int(math.ceil((len(audio_1d) - nperseg) / max_time_bins))
        noverlap = max(0, nperseg - hop)

    freqs, times, Sxx = signal.spectrogram(
        audio_1d,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
    )

    # Convert to power in dB with numerical stability epsilon
    eps = 1e-10
    Sxx_db = 10.0 * np.log10(np.maximum(Sxx, eps))

    vmax = float(np.max(Sxx_db))
    vmin = max(vmax - 80.0, float(np.min(Sxx_db)))

    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=100)
    apply_dark_theme(fig, ax)

    pcm = ax.pcolormesh(
        times,
        freqs / 1000.0,  # Convert Hz to kHz
        Sxx_db,
        shading="gouraud",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel("Time (seconds)", fontsize=9.5)
    ax.set_ylabel("Frequency (kHz)", fontsize=9.5)
    ax.set_title(title, fontsize=11, fontweight=600, pad=8)
    ax.set_ylim(0, sample_rate / 2000.0)

    # Colorbar
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label("Power Spectral Density (dB)", color=TEXT_MUTED, fontsize=8.5)
    cbar.ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    cbar.outline.set_edgecolor(GRID_COLOR)

    fig.tight_layout()
    return fig


def create_timeline_figure(
    long_result: LongRecordingInferenceResult,
    candidate_threshold: float = DEFAULT_CANDIDATE_THRESHOLD,
) -> plt.Figure:
    """
    Plots per-window cosine similarity scores across the multi-window timeline.

    Args:
        long_result: Structured result container from analyze_long_recording
        candidate_threshold: Decision threshold value (default: 0.4820)

    Returns:
        matplotlib.figure.Figure (Caller is responsible for plt.close)
    """
    windows = long_result.windows
    if not windows:
        # Fallback figure for empty window results
        fig, ax = plt.subplots(figsize=(9, 3.2), dpi=100)
        apply_dark_theme(fig, ax)
        ax.text(0.5, 0.5, "No window results available to plot.", color=TEXT_MUTED, ha="center", va="center")
        fig.tight_layout()
        return fig

    midpoints = np.array([(w.window.start_time_seconds + w.window.end_time_seconds) / 2.0 for w in windows])
    scores = np.array([w.model_score for w in windows])
    is_real = np.array([w.decision.prediction == DecisionLabel.REAL for w in windows])

    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=100)
    apply_dark_theme(fig, ax)

    # Shaded decision regions
    duration = long_result.covered_duration_seconds
    ax.axhspan(candidate_threshold, 1.05, color=ACCENT_GREEN, alpha=0.08, label="Likely Genuine Region (Score ≥ τ*)")
    ax.axhspan(-1.05, candidate_threshold, color=ACCENT_RED, alpha=0.08, label="Potential Spoof Region (Score < τ*)")

    # Candidate decision threshold horizontal reference line
    ax.axhline(
        candidate_threshold,
        color=ACCENT_AMBER,
        linestyle="--",
        linewidth=1.6,
        label=f"Candidate Threshold (τ* = {candidate_threshold:.4f}, Unvalidated)",
        zorder=2,
    )

    # Trajectory connecting line
    ax.plot(midpoints, scores, color=TEXT_MUTED, linestyle="-", linewidth=1.2, alpha=0.6, zorder=3)

    # Discrete window markers
    if np.any(is_real):
        ax.scatter(
            midpoints[is_real],
            scores[is_real],
            color=ACCENT_GREEN,
            s=45,
            edgecolors=THEME_BG,
            linewidth=1.2,
            zorder=4,
            label="Likely Genuine Window",
        )
    if np.any(~is_real):
        ax.scatter(
            midpoints[~is_real],
            scores[~is_real],
            color=ACCENT_RED,
            s=45,
            marker="s",
            edgecolors=THEME_BG,
            linewidth=1.2,
            zorder=4,
            label="Potential Spoof Window",
        )

    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(0, max(duration, midpoints[-1] + 1.0))
    ax.set_xlabel("Timeline Position — Window Midpoint (seconds)", fontsize=9.5)
    ax.set_ylabel("Cosine Similarity S(x)", fontsize=9.5)
    ax.set_title(
        f"Sliding Window Timeline Evaluation ({len(windows)} windows, 1.0s hop)",
        fontsize=11,
        fontweight=600,
        pad=8,
    )

    ax.legend(
        loc="lower left",
        facecolor=PANEL_BG,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_MUTED,
        fontsize=7.8,
        framealpha=0.9,
    )

    fig.tight_layout()
    return fig


def create_feature_views_figure(
    spec_2d_tensor: Union[np.ndarray, torch.Tensor],
    window_label: str = "Analysis Window (4.0s @ 16 kHz)",
) -> plt.Figure:
    """
    Renders the 3-channel 2D spatiotemporal representations processed by the LCNN-2D backbone:
      - Channel 0: ST-FrFT (alpha = 0.50, chirp angle 45°)
      - Channel 1: ST-FrFT (alpha = 0.85, chirp angle 76.5°)
      - Channel 2: Continuous Morlet Wavelet Scalogram (64 scales)

    Args:
        spec_2d_tensor: Tensor of shape (3, 256, 397) or (1, 3, 256, 397)
        window_label: Context label describing the displayed window

    Returns:
        matplotlib.figure.Figure (Caller is responsible for plt.close)
    """
    if isinstance(spec_2d_tensor, torch.Tensor):
        tensor_np = spec_2d_tensor.detach().cpu().numpy()
    elif isinstance(spec_2d_tensor, np.ndarray):
        tensor_np = spec_2d_tensor
    else:
        raise ValueError(f"Expected torch.Tensor or numpy.ndarray, got {type(spec_2d_tensor).__name__}")

    if tensor_np.ndim == 4:
        tensor_np = tensor_np[0]

    if tensor_np.shape[0] != 3 or tensor_np.ndim != 3:
        raise ValueError(f"Expected shape (3, H, W) or (1, 3, H, W), got {tensor_np.shape}")

    channel_titles = [
        "Channel 0: ST-FrFT (α = 0.50, 45° chirp)",
        "Channel 1: ST-FrFT (α = 0.85, 76.5° chirp)",
        "Channel 2: Morlet Wavelet Scalogram",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), dpi=100)
    fig.patch.set_facecolor(THEME_BG)

    for i in range(3):
        ax = axes[i]
        apply_dark_theme(fig, ax)
        ch_data = tensor_np[i]

        im = ax.imshow(
            ch_data,
            aspect="auto",
            origin="lower",
            cmap="viridis" if i < 2 else "plasma",
        )
        ax.set_title(channel_titles[i], fontsize=9.5, fontweight=600, pad=6)
        ax.set_xlabel("Time Frames (0–397)", fontsize=8.5)
        if i == 0:
            ax.set_ylabel("Transform Bins (0–255)", fontsize=8.5)
        else:
            ax.set_ylabel("")

        cbar = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.046)
        cbar.ax.tick_params(colors=TEXT_MUTED, labelsize=7)
        cbar.outline.set_edgecolor(GRID_COLOR)

    fig.suptitle(f"Neural Feature Representations — {window_label}", fontsize=11, fontweight=600, color=TEXT_MAIN, y=0.98)
    fig.tight_layout()
    return fig
