"""
Automated unit tests for Audio Loading and Decoding Module.
"""

import os
import tempfile
import numpy as np
import pytest
import soundfile as sf

from core.audio_loader import load_audio
from core.schemas import LoadedAudio
from core.exceptions import (
    AudioLoadError,
    UnsupportedAudioFormatError,
    CorruptAudioFileError,
    AudioFileTooLargeError,
    AudioTooLongError,
    EmptyAudioFileError,
)


@pytest.fixture
def temp_audio_dir():
    with tempfile.TemporaryDirectory() as td:
        yield td


def test_load_valid_wav_mono(temp_audio_dir):
    """Verifies that a standard mono WAV file decodes into 1D array with exact frames."""
    path = os.path.join(temp_audio_dir, "mono.wav")
    sr = 16000
    data = (np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr)) * 0.5).astype(np.float32)
    sf.write(path, data, sr)

    loaded = load_audio(path)
    assert isinstance(loaded, LoadedAudio)
    assert loaded.sample_rate == 16000
    assert loaded.channels == 1
    assert loaded.frames == 16000
    assert abs(loaded.duration_seconds - 1.0) < 1e-4
    assert loaded.waveform.ndim == 1
    assert loaded.format_name == "WAV"


def test_load_valid_wav_stereo(temp_audio_dir):
    """Verifies that a stereo WAV file decodes into canonical (frames, channels) layout."""
    path = os.path.join(temp_audio_dir, "stereo.wav")
    sr = 44100
    frames = 44100 * 2
    stereo_data = np.stack([np.ones(frames) * 0.3, np.ones(frames) * 0.7], axis=1).astype(np.float32)
    sf.write(path, stereo_data, sr)

    loaded = load_audio(path)
    assert loaded.channels == 2
    assert loaded.frames == frames
    assert loaded.waveform.shape == (frames, 2)
    assert abs(loaded.duration_seconds - 2.0) < 1e-4


def test_load_flac_format(temp_audio_dir):
    """Verifies decoding of native FLAC lossless container."""
    path = os.path.join(temp_audio_dir, "test.flac")
    sr = 16000
    data = (np.sin(2 * np.pi * 300 * np.linspace(0, 0.5, 8000)) * 0.5).astype(np.float32)
    sf.write(path, data, sr)

    loaded = load_audio(path)
    assert loaded.format_name == "FLAC"
    assert loaded.frames == 8000


def test_load_mp3_format(temp_audio_dir):
    """Verifies decoding of native MP3 container."""
    path = os.path.join(temp_audio_dir, "test.mp3")
    sr = 16000
    data = (np.sin(2 * np.pi * 300 * np.linspace(0, 1.0, sr)) * 0.5).astype(np.float32)
    sf.write(path, data, sr, format="MP3")

    loaded = load_audio(path)
    assert loaded.format_name in ["MP3", "MPEG"]
    assert loaded.channels == 1


def test_load_ogg_format(temp_audio_dir):
    """Verifies decoding of native OGG container."""
    path = os.path.join(temp_audio_dir, "test.ogg")
    sr = 16000
    data = (np.sin(2 * np.pi * 500 * np.linspace(0, 0.5, 8000)) * 0.5).astype(np.float32)
    sf.write(path, data, sr)

    loaded = load_audio(path)
    assert loaded.format_name == "OGG"


def test_unsupported_ffmpeg_extension_wma(temp_audio_dir):
    """Verifies that WMA files produce an informative UnsupportedAudioFormatError."""
    path = os.path.join(temp_audio_dir, "test.wma")
    with open(path, "wb") as f:
        f.write(b"dummy wma content")

    with pytest.raises(UnsupportedAudioFormatError) as exc_info:
        load_audio(path)
    assert "requires external FFmpeg" in str(exc_info.value)


def test_corrupt_m4a_file_raises_error(temp_audio_dir):
    """Verifies that an unparseable M4A file raises CorruptAudioFileError (or UnsupportedAudioFormatError if decoder missing)."""
    path = os.path.join(temp_audio_dir, "corrupt.m4a")
    with open(path, "wb") as f:
        f.write(b"not an mp4/m4a container")

    with pytest.raises((CorruptAudioFileError, UnsupportedAudioFormatError)):
        load_audio(path)


def test_load_m4a_format():
    """Verifies decoding of native M4A/AAC container using PyAV."""
    fixture_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "test_artifacts",
        "fixtures",
        "sample_test.m4a",
    )
    if not os.path.exists(fixture_path):
        pytest.skip("sample_test.m4a fixture not found")

    loaded = load_audio(fixture_path)
    assert loaded.format_name == "M4A"
    assert loaded.subtype == "AAC"
    assert loaded.decoder in ("av", "ffmpeg")
    assert loaded.sample_rate == 16000
    assert loaded.duration_seconds > 0
    assert loaded.waveform is not None
    assert np.all(np.isfinite(loaded.waveform))


def test_missing_file_raises_error():
    """Verifies that missing files raise AudioLoadError."""
    with pytest.raises(AudioLoadError):
        load_audio("C:/nonexistent_audio_path_xyz.wav")


def test_empty_file_raises_error(temp_audio_dir):
    """Verifies that 0-byte audio files raise EmptyAudioFileError."""
    path = os.path.join(temp_audio_dir, "empty.wav")
    with open(path, "wb") as f:
        pass  # 0 bytes

    with pytest.raises(EmptyAudioFileError):
        load_audio(path)


def test_corrupted_file_raises_error(temp_audio_dir):
    """Verifies that corrupted files raise CorruptAudioFileError."""
    path = os.path.join(temp_audio_dir, "corrupt.wav")
    with open(path, "wb") as f:
        f.write(b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00corrupted garbage data")

    with pytest.raises(CorruptAudioFileError):
        load_audio(path)


def test_file_too_large_rejection(temp_audio_dir):
    """Verifies that files exceeding max_size_bytes are rejected before reading."""
    path = os.path.join(temp_audio_dir, "large.wav")
    sr = 16000
    data = np.zeros(16000, dtype=np.float32)
    sf.write(path, data, sr)

    # Set threshold lower than file size
    with pytest.raises(AudioFileTooLargeError):
        load_audio(path, max_size_bytes=100)


def test_duration_too_long_rejection(temp_audio_dir):
    """Verifies that audio exceeding max_duration_seconds is rejected."""
    path = os.path.join(temp_audio_dir, "toolong.wav")
    sr = 16000
    data = np.zeros(16000 * 5, dtype=np.float32)  # 5.0 seconds
    sf.write(path, data, sr)

    with pytest.raises(AudioTooLongError):
        load_audio(path, max_duration_seconds=3.0)
