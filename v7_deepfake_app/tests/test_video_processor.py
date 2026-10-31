"""
Automated unit tests for Video-to-Audio Processing Module.
"""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from core.video_processor import (
    discover_ffmpeg,
    extract_audio_from_video,
    cleanup_temp_dir,
)
from core.exceptions import (
    VideoProcessingError,
    FFmpegNotFoundError,
    FFmpegExecutionError,
    NoAudioStreamError,
)


def test_discover_ffmpeg_status():
    """Verifies FFmpeg discovery behavior on the local machine."""
    is_avail, path = discover_ffmpeg()
    # On this audited Windows environment, ffmpeg is confirmed absent from PATH
    if not is_avail:
        assert path is None
    else:
        assert isinstance(path, str)
        assert os.path.exists(path)


def test_video_extraction_fails_when_ffmpeg_missing():
    """Verifies that attempting video extraction without FFmpeg raises informative FFmpegNotFoundError."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"dummy video data")
        temp_video = f.name

    try:
        with patch("core.video_processor.discover_ffmpeg", return_value=(False, None)):
            with pytest.raises(FFmpegNotFoundError) as exc_info:
                extract_audio_from_video(temp_video)
            assert "Video analysis requires FFmpeg" in str(exc_info.value)
    finally:
        if os.path.exists(temp_video):
            os.remove(temp_video)


def test_video_unsupported_extension():
    """Verifies that unsupported video formats raise VideoProcessingError."""
    with tempfile.NamedTemporaryFile(suffix=".flv", delete=False) as f:
        f.write(b"dummy")
        temp_video = f.name

    try:
        with pytest.raises(VideoProcessingError):
            extract_audio_from_video(temp_video)
    finally:
        if os.path.exists(temp_video):
            os.remove(temp_video)


def test_mocked_ffmpeg_successful_extraction():
    """Mocks FFmpeg subprocess execution to verify argument assembly, output validation, and cleanup."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"dummy mp4 container")
        temp_video = f.name

    def fake_subprocess_run(cmd, **kwargs):
        # cmd: [ffmpeg, -y, -v, error, -i, input, -vn, -acodec, pcm_s16le, -ar, 16000, -ac, 1, out_wav]
        out_wav = cmd[-1]
        with open(out_wav, "wb") as f_out:
            f_out.write(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00fake audio payload")
        mock_res = MagicMock()
        mock_res.returncode = 0
        return mock_res

    try:
        with patch("core.video_processor.discover_ffmpeg", return_value=(True, "fake_ffmpeg")):
            with patch("subprocess.run", side_effect=fake_subprocess_run):
                result = extract_audio_from_video(temp_video)
                assert result.has_audio_stream is True
                assert os.path.exists(result.extracted_audio_path)
                assert "v7_deepfake_video_" in result.temp_dir

                # Verify cleanup removes directory and WAV file
                cleanup_temp_dir(result.temp_dir)
                assert not os.path.exists(result.temp_dir)
    finally:
        if os.path.exists(temp_video):
            os.remove(temp_video)


def test_mocked_ffmpeg_no_audio_stream():
    """Mocks a video that contains no audio streams and verifies NoAudioStreamError is raised."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"dummy video data")
        temp_video = f.name

    mock_res = MagicMock()
    mock_res.returncode = 1
    mock_res.stderr = "Output file does not contain any stream"

    try:
        with patch("core.video_processor.discover_ffmpeg", return_value=(True, "fake_ffmpeg")):
            with patch("subprocess.run", return_value=mock_res):
                with pytest.raises(NoAudioStreamError):
                    extract_audio_from_video(temp_video)
    finally:
        if os.path.exists(temp_video):
            os.remove(temp_video)
