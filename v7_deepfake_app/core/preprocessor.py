"""
Waveform Preprocessor for the V7 Audio Deepfake Detection System.

Handles waveform validation, multichannel downmixing, polyphase resampling to 16 kHz,
and length normalization (circular wrap-padding for short audio, single-window truncation
for long audio) strictly adhering to the verified V7 training protocol.
"""

import math
import numpy as np
import scipy.signal as signal
import torch
from typing import Tuple, Union, Optional

from core.schemas import PreprocessingResult
from core.exceptions import (
    InvalidWaveformError,
    InvalidSampleRateError,
    PreprocessingError,
)

TARGET_SAMPLE_RATE: int = 16000
TARGET_SAMPLES: int = 64000  # 4.0 seconds at 16 kHz


def preprocess_waveform(
    audio: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    target_samples: Optional[int] = TARGET_SAMPLES,
    channel_layout: Optional[str] = None,
) -> Tuple[torch.Tensor, PreprocessingResult]:
    """
    Standardizes input audio waveform to the fixed-length tensor expected by the V7 model.

    Processing Pipeline:
      1. Validation: Ensures audio is non-empty, finite (no NaN/Inf), and 1D or 2D.
      2. Multichannel Downmix: Averages channels to mono according to explicit contract or unambiguous shape.
         - 'frames_first' / 'channels_last': shape is (frames, channels), downmixes across axis=1.
         - 'channels_first' / 'frames_last': shape is (channels, frames), downmixes across axis=0.
         - None (auto): auto-detects when one dimension is <= 8 channels and the other > 8 frames;
           raises InvalidWaveformError if dimensions are ambiguous.
      3. Polyphase Resampling: Converts arbitrary sample rates to 16 kHz using scipy.signal.resample_poly.
      4. Length Normalization:
         - Short audio (< 4.0s): Circular wrap-padding (mode='wrap') matching training protocol.
         - Long audio (> 4.0s): Truncates to initial 4.0s window with documented warning.
      5. Output Formatting: Converts to torch.FloatTensor with shape [1, 64000].

    Args:
        audio: Input audio data as NumPy array or PyTorch tensor
        sample_rate: Sampling frequency of the input audio in Hz
        target_sample_rate: Expected model sampling rate (default: 16000 Hz)
        target_samples: Expected sample length (default: 64000 samples)
        channel_layout: Optional channel layout override ('frames_first', 'channels_first')

    Returns:
        Tuple of (preprocessed_tensor [1, target_samples], PreprocessingResult)

    Raises:
        InvalidWaveformError: If input is empty, has >2 dimensions, contains NaN/Inf, or has ambiguous shape
        InvalidSampleRateError: If sample rate is non-positive or invalid
        PreprocessingError: If resampling, downmixing, or length normalization fails
    """
    warnings = []

    # 1. Sample rate validation
    if not isinstance(sample_rate, (int, float)) or sample_rate <= 0 or not math.isfinite(sample_rate):
        raise InvalidSampleRateError(
            f"Invalid sample rate: {sample_rate}. Sample rate must be a positive finite number."
        )
    sample_rate = int(round(sample_rate))

    # 2. Type conversion and dimension inspection
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().cpu().numpy()
    elif isinstance(audio, np.ndarray):
        audio_np = audio
    else:
        raise InvalidWaveformError(
            f"Unsupported audio type: {type(audio).__name__}. Expected numpy.ndarray or torch.Tensor."
        )

    if audio_np.size == 0:
        raise InvalidWaveformError("Input audio array is empty (0 samples).")

    if not np.all(np.isfinite(audio_np)):
        raise InvalidWaveformError("Audio contains non-finite values (NaN or Inf).")

    if audio_np.ndim > 2:
        raise InvalidWaveformError(
            f"Audio has {audio_np.ndim} dimensions. Expected 1D (mono) or 2D (multichannel)."
        )

    # 3. Multichannel downmixing to mono
    is_multichannel = False
    if audio_np.ndim == 2:
        is_multichannel = True
        num_rows, num_cols = audio_np.shape

        if channel_layout in ["frames_first", "channels_last"]:
            # Explicit contract: (frames, channels), channels is axis 1
            if num_cols > 8 and num_cols > num_rows:
                warnings.append(f"Unusual channel count ({num_cols}) for frames_first layout.")
            audio_np = np.mean(audio_np, axis=1)
        elif channel_layout in ["channels_first", "frames_last"]:
            # Explicit contract: (channels, frames), channels is axis 0
            if num_rows > 8 and num_rows > num_cols:
                warnings.append(f"Unusual channel count ({num_rows}) for channels_first layout.")
            audio_np = np.mean(audio_np, axis=0)
        elif channel_layout is None:
            # Auto-detection with defensive ambiguity check
            if num_cols <= 8 and num_rows > 8:
                # Canonical decoder format: (frames, channels)
                audio_np = np.mean(audio_np, axis=1)
            elif num_rows <= 8 and num_cols > 8:
                # Librosa / PyTorch format: (channels, frames)
                audio_np = np.mean(audio_np, axis=0)
            else:
                raise InvalidWaveformError(
                    f"Ambiguous 2D audio dimensions {audio_np.shape}. Cannot auto-detect channel axis. "
                    "Please specify channel_layout='frames_first' for (frames, channels) or "
                    "channel_layout='channels_first' for (channels, frames)."
                )
        else:
            raise InvalidWaveformError(
                f"Unknown channel_layout '{channel_layout}'. Supported: 'frames_first', 'channels_first', or None."
            )

        # Defensive check: ensure audio_np is now 1D and has valid frame count
        if audio_np.ndim != 1:
            raise PreprocessingError(f"Downmixing failed to produce 1D waveform: shape is {audio_np.shape}")
        warnings.append("Multichannel audio detected; downmixed to mono via channel averaging.")

    original_samples = len(audio_np)
    original_duration = float(original_samples) / float(sample_rate)

    # 4. Polyphase anti-aliased resampling to target_sample_rate
    is_resampled = False
    if sample_rate != target_sample_rate:
        try:
            gcd = math.gcd(sample_rate, target_sample_rate)
            up = target_sample_rate // gcd
            down = sample_rate // gcd
            audio_np = signal.resample_poly(audio_np, up, down)
            is_resampled = True
            warnings.append(
                f"Audio resampled from {sample_rate} Hz to {target_sample_rate} Hz via polyphase filtering."
            )
        except Exception as e:
            raise PreprocessingError(f"Polyphase resampling failed: {str(e)}") from e

    current_samples = len(audio_np)

    # 5. Length normalization
    is_padded = False
    is_truncated = False

    if target_samples is not None:
        if current_samples < target_samples:
            pad_amount = target_samples - current_samples
            # Circular wrap padding strictly matching training protocol (mode='wrap')
            audio_np = np.pad(audio_np, (0, pad_amount), mode="wrap")
            is_padded = True
            warnings.append(
                f"Input audio ({original_duration:.2f}s) is shorter than {target_samples / target_sample_rate:.1f}s window; "
                f"circular wrap padding applied. Detection confidence on short audio may be lower than full-length recordings."
            )
        elif current_samples > target_samples:
            # Single-window policy: evaluate the first 4.0 seconds (64,000 samples)
            audio_np = audio_np[:target_samples]
            is_truncated = True
            warnings.append(
                f"Input audio ({original_duration:.2f}s) exceeds {target_samples / target_sample_rate:.1f}s window; "
                f"truncated to initial {target_samples / target_sample_rate:.1f}s under single-window policy. Audio beyond 4.0s was omitted."
            )
        analyzed_samples = target_samples
        analyzed_duration = float(target_samples) / float(target_sample_rate)
    else:
        analyzed_samples = current_samples
        analyzed_duration = float(current_samples) / float(target_sample_rate)

    # 6. Formatting and finite validation
    audio_np = audio_np.astype(np.float32)
    if not np.all(np.isfinite(audio_np)):
        raise PreprocessingError("Waveform contains non-finite values after length normalization.")

    # 7. Construct PyTorch tensor [1, analyzed_samples]
    tensor = torch.from_numpy(audio_np).unsqueeze(0)  # Shape: [1, N]

    result_meta = PreprocessingResult(
        original_sample_rate=sample_rate,
        target_sample_rate=target_sample_rate,
        original_samples=original_samples,
        target_samples=analyzed_samples,
        original_duration_seconds=round(original_duration, 4),
        analyzed_duration_seconds=round(analyzed_duration, 4),
        is_multichannel=is_multichannel,
        is_resampled=is_resampled,
        is_padded=is_padded,
        is_truncated=is_truncated,
        warnings=warnings,
    )

    return tensor, result_meta
