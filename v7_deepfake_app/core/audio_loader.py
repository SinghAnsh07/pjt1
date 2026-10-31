"""
Audio Loading and Decoding Module for the V7 Audio Deepfake System.

Provides robust, defensive audio decoding using soundfile/libsndfile, PyAV,
and FFmpeg fallback, validating file existence, size, header integrity, sample
rate, and numerical sanity.
"""

import os
import subprocess
import tempfile
from typing import Optional, List, Tuple
import numpy as np
import soundfile as sf

from core.schemas import LoadedAudio
from core.config import (
    MAX_FILE_SIZE_BYTES,
    MAX_AUDIO_DURATION_SECONDS,
    MIN_AUDIO_DURATION_SECONDS,
    NATIVE_AUDIO_EXTENSIONS,
    FFMPEG_REQUIRED_AUDIO_EXTENSIONS,
)
from core.exceptions import (
    AudioLoadError,
    UnsupportedAudioFormatError,
    CorruptAudioFileError,
    AudioFileTooLargeError,
    AudioTooLongError,
    EmptyAudioFileError,
    InvalidWaveformError,
    InvalidSampleRateError,
)


def _decode_audio_with_av(
    file_path: str,
    max_duration_seconds: float = MAX_AUDIO_DURATION_SECONDS,
) -> LoadedAudio:
    """
    Decodes containerized audio formats (e.g. M4A, AAC, ALAC) into canonical LoadedAudio
    using PyAV (libavformat / libavcodec).
    """
    import av

    warnings: List[str] = []
    try:
        container = av.open(file_path)
    except Exception as e:
        raise CorruptAudioFileError(f"Failed to open audio container: {str(e)}") from e

    with container:
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is None:
            raise CorruptAudioFileError(f"No audio stream found in container: {file_path}")

        codec_ctx = audio_stream.codec_context
        sample_rate = codec_ctx.sample_rate if codec_ctx else None
        if sample_rate is None or sample_rate <= 0:
            raise InvalidSampleRateError(f"Audio file has invalid sample rate: {file_path}")

        # Stream duration check if metadata is present
        if audio_stream.duration is not None and audio_stream.time_base is not None:
            est_duration = float(audio_stream.duration * audio_stream.time_base)
            if est_duration > max_duration_seconds:
                raise AudioTooLongError(
                    f"Audio duration ({est_duration:.1f}s) exceeds maximum allowed duration "
                    f"({max_duration_seconds:.1f}s): {file_path}"
                )

        chunks: List[np.ndarray] = []
        total_frames = 0

        try:
            for frame in container.decode(audio_stream):
                # frame.to_ndarray() returns (channels, samples)
                arr = frame.to_ndarray()
                if np.issubdtype(arr.dtype, np.integer):
                    max_val = float(np.iinfo(arr.dtype).max)
                    arr = arr.astype(np.float32) / max_val
                elif arr.dtype != np.float32:
                    arr = arr.astype(np.float32)

                chunks.append(arr)
                total_frames += arr.shape[1]

                current_duration = float(total_frames) / float(sample_rate)
                if current_duration > max_duration_seconds:
                    raise AudioTooLongError(
                        f"Audio duration ({current_duration:.1f}s) exceeds maximum allowed duration "
                        f"({max_duration_seconds:.1f}s): {file_path}"
                    )
        except AudioTooLongError:
            raise
        except Exception as e:
            raise CorruptAudioFileError(f"Failed to decode audio frames: {str(e)}") from e

    if not chunks or total_frames == 0:
        raise EmptyAudioFileError(f"Audio file contains 0 decoded frames: {file_path}")

    waveform_2d = np.concatenate(chunks, axis=1)  # shape (channels, total_frames)
    channels = waveform_2d.shape[0]
    frames = waveform_2d.shape[1]

    if channels == 1:
        waveform = waveform_2d[0]  # shape (total_frames,)
    else:
        waveform = waveform_2d.T  # shape (total_frames, channels)

    duration = float(frames) / float(sample_rate)
    if duration > max_duration_seconds:
        raise AudioTooLongError(
            f"Audio duration ({duration:.1f}s) exceeds maximum allowed duration ({max_duration_seconds:.1f}s): {file_path}"
        )
    if duration < MIN_AUDIO_DURATION_SECONDS:
        warnings.append(f"Audio duration ({duration:.3f}s) is extremely short (< {MIN_AUDIO_DURATION_SECONDS}s).")

    if not np.all(np.isfinite(waveform)):
        raise InvalidWaveformError("Decoded audio contains non-finite samples (NaN or Inf).")

    codec_name = getattr(codec_ctx, "name", "aac")
    subtype = codec_name.upper() if codec_name else "AAC"
    ext = os.path.splitext(file_path)[1].replace(".", "").upper()

    return LoadedAudio(
        waveform=waveform,
        sample_rate=sample_rate,
        channels=channels,
        duration_seconds=duration,
        frames=frames,
        format_name=ext if ext else "M4A",
        subtype=subtype,
        decoder="av",
        warnings=warnings,
    )


def _decode_audio_with_ffmpeg(
    file_path: str,
    ffmpeg_bin: str,
    max_duration_seconds: float = MAX_AUDIO_DURATION_SECONDS,
) -> LoadedAudio:
    """
    Decodes audio by transcoding to temporary uncompressed WAV using external FFmpeg.
    """
    with tempfile.TemporaryDirectory(prefix="v7_ffmpeg_audio_") as td:
        out_wav = os.path.join(td, "decoded.wav")
        try:
            res = subprocess.run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-nostdin",
                    "-loglevel", "error",
                    "-i", file_path,
                    "-vn",
                    "-c:a", "pcm_s16le",
                    out_wav,
                ],
                capture_output=True,
                timeout=30.0,
                shell=False,
            )
        except subprocess.TimeoutExpired as e:
            raise CorruptAudioFileError(f"FFmpeg transcoding timed out on: {file_path}") from e
        except Exception as e:
            raise CorruptAudioFileError(f"FFmpeg invocation failed: {str(e)}") from e

        if res.returncode != 0:
            err_msg = res.stderr.decode("utf-8", errors="ignore").strip()
            raise CorruptAudioFileError(f"FFmpeg failed to transcode audio: {err_msg}")

        if not os.path.exists(out_wav) or os.path.getsize(out_wav) == 0:
            raise EmptyAudioFileError(f"Transcoded audio stream is empty: {file_path}")

        loaded = load_audio(out_wav, max_duration_seconds=max_duration_seconds)
        ext = os.path.splitext(file_path)[1].replace(".", "").upper()

        return LoadedAudio(
            waveform=loaded.waveform,
            sample_rate=loaded.sample_rate,
            channels=loaded.channels,
            duration_seconds=loaded.duration_seconds,
            frames=loaded.frames,
            format_name=ext if ext else "M4A",
            subtype="AAC",
            decoder="ffmpeg",
            warnings=loaded.warnings,
        )


def _decode_container_audio(
    file_path: str,
    ext: str,
    max_duration_seconds: float = MAX_AUDIO_DURATION_SECONDS,
) -> LoadedAudio:
    """
    Attempts to decode container audio (e.g. M4A/AAC) using PyAV or FFmpeg.
    """
    # 1. Try PyAV first
    try:
        import av  # noqa: F401
        return _decode_audio_with_av(file_path, max_duration_seconds=max_duration_seconds)
    except ImportError:
        pass
    except AudioLoadError:
        raise
    except Exception as e:
        raise CorruptAudioFileError(f"Failed to decode '{ext}' audio with PyAV: {str(e)}") from e

    # 2. Try FFmpeg fallback
    try:
        from core.video_processor import discover_ffmpeg
        is_avail, ffmpeg_bin = discover_ffmpeg()
        if is_avail and ffmpeg_bin:
            return _decode_audio_with_ffmpeg(
                file_path, ffmpeg_bin=ffmpeg_bin, max_duration_seconds=max_duration_seconds
            )
    except Exception:
        pass

    # 3. Neither is available
    raise UnsupportedAudioFormatError(
        f"Audio format '{ext}' requires 'av' (PyAV) or external FFmpeg, neither of which is currently "
        f"available in the environment. Please install the 'av' package or add FFmpeg to PATH."
    )


def load_audio(
    file_path: str,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
    max_duration_seconds: float = MAX_AUDIO_DURATION_SECONDS,
) -> LoadedAudio:
    """
    Decodes an audio file from disk into a floating-point NumPy waveform with metadata.

    Canonical output layout:
      - Mono: 1D array shaped (frames,)
      - Multichannel: 2D array shaped (frames, channels)

    Args:
        file_path: Absolute or relative filesystem path to the audio file
        max_size_bytes: Configured maximum file size in bytes
        max_duration_seconds: Configured maximum duration limit in seconds

    Returns:
        LoadedAudio: Structured container with waveform array and container metadata

    Raises:
        AudioLoadError / FileNotFoundError: If file does not exist or cannot be read
        AudioFileTooLargeError: If file size exceeds max_size_bytes
        UnsupportedAudioFormatError: If file format is not supported or lacks decoder
        CorruptAudioFileError: If file cannot be decoded
        EmptyAudioFileError: If decoded audio contains 0 frames
        AudioTooLongError: If audio duration exceeds max_duration_seconds
        InvalidWaveformError: If decoded waveform contains non-finite samples
    """
    warnings: List[str] = []

    # 1. Existence and Readability Validation
    if not os.path.exists(file_path):
        raise AudioLoadError(f"Audio file not found at: {file_path}")
    if not os.path.isfile(file_path):
        raise AudioLoadError(f"Audio path is not a regular file: {file_path}")
    if not os.access(file_path, os.R_OK):
        raise AudioLoadError(f"Audio file exists but is not readable: {file_path}")

    # 2. File Size Inspection (before reading entire payload into memory)
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise EmptyAudioFileError(f"Audio file is completely empty (0 bytes): {file_path}")
    if file_size > max_size_bytes:
        raise AudioFileTooLargeError(
            f"File size ({file_size / (1024*1024):.2f} MB) exceeds maximum allowed limit "
            f"({max_size_bytes / (1024*1024):.1f} MB): {file_path}"
        )

    # 3. Extension and Format Inspection
    ext = os.path.splitext(file_path)[1].lower()
    if ext in FFMPEG_REQUIRED_AUDIO_EXTENSIONS:
        raise UnsupportedAudioFormatError(
            f"Audio format '{ext}' requires external FFmpeg transcoding, which is not natively "
            f"supported by libsndfile. Please convert to WAV, FLAC, MP3, OGG, or M4A."
        )

    # 4. Container formats (.m4a, .aac) bypass libsndfile and go to container decoder
    if ext in {".m4a", ".aac"}:
        return _decode_container_audio(file_path, ext, max_duration_seconds=max_duration_seconds)

    # 5. Native decoding with libsndfile via soundfile
    try:
        info = sf.info(file_path)
    except Exception as e:
        # Check if container decoder can salvage this file
        try:
            return _decode_container_audio(file_path, ext, max_duration_seconds=max_duration_seconds)
        except Exception:
            pass
        raise CorruptAudioFileError(f"Failed to read audio file headers (file may be corrupted or invalid format): {str(e)}") from e

    if info.frames == 0:
        raise EmptyAudioFileError(f"Audio file contains 0 frames: {file_path}")

    if info.samplerate <= 0:
        raise InvalidSampleRateError(f"Audio file has invalid sample rate ({info.samplerate} Hz): {file_path}")

    duration = float(info.frames) / float(info.samplerate)
    if duration > max_duration_seconds:
        raise AudioTooLongError(
            f"Audio duration ({duration:.1f}s) exceeds maximum allowed duration ({max_duration_seconds:.1f}s): {file_path}"
        )
    if duration < MIN_AUDIO_DURATION_SECONDS:
        warnings.append(f"Audio duration ({duration:.3f}s) is extremely short (< {MIN_AUDIO_DURATION_SECONDS}s).")

    try:
        waveform, sample_rate = sf.read(file_path, always_2d=False, dtype="float32")
    except Exception as e:
        raise CorruptAudioFileError(f"Failed to decode audio frames: {str(e)}") from e

    # 6. Sanity Validation on Decoded Array
    if waveform.size == 0:
        raise EmptyAudioFileError(f"Decoded waveform is empty: {file_path}")
    if not np.all(np.isfinite(waveform)):
        raise InvalidWaveformError("Decoded audio contains non-finite samples (NaN or Inf).")

    # 7. Channel Layout Verification
    if waveform.ndim == 1:
        channels = 1
        frames = len(waveform)
    elif waveform.ndim == 2:
        frames = waveform.shape[0]
        channels = waveform.shape[1]
    else:
        raise InvalidWaveformError(f"Decoded waveform has unexpected dimensions {waveform.ndim}")

    return LoadedAudio(
        waveform=waveform,
        sample_rate=sample_rate,
        channels=channels,
        duration_seconds=duration,
        frames=frames,
        format_name=getattr(info, "format", ext.replace(".", "").upper()),
        subtype=getattr(info, "subtype", "UNKNOWN"),
        decoder="soundfile",
        warnings=warnings,
    )
