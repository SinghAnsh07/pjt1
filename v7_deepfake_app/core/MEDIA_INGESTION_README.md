# Media Ingestion, Video Extraction, and Long-Recording Analysis (`core/`)

## 1. Overview and Scope

This module establishes the comprehensive media ingestion and preprocessing layer for the V7 Audio Deepfake Detection System. It bridges raw media files (audio files, video containers, and long recordings) to the verified `V7InferenceEngine` without modifying model weights or creating misleading confidence percentages.

---

## 2. Supported Audio & Video Formats

### 2.1 Natively Supported Audio Formats (Tested & Verified via `libsndfile` / `soundfile 0.14.0`)
The following formats are decoded directly into floating-point arrays in memory without external CLI dependencies:
- **WAV:** 16-bit, 24-bit, 32-bit float, PCM, $\mu$-law, A-law
- **FLAC:** Lossless Free Lossless Audio Codec (16-bit, 24-bit)
- **MP3:** MPEG-1/2 Audio Layer III
- **OGG:** OGG Vorbis
- **AIFF / AIF:** Apple Audio Interchange File Format

### 2.2 Formats Requiring External FFmpeg
The following formats cannot be decoded by `libsndfile` directly and require system FFmpeg:
- **Audio:** `.m4a`, `.aac`, `.wma`, `.opus`
- **Video Containers:** `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`

> [!IMPORTANT]
> **FFmpeg Host Status:** FFmpeg is currently **not present** on the host Windows `PATH` (`WinError 2`). Video container uploads and M4A/AAC decoding fail gracefully with an explicit, actionable error:
> *"Video analysis requires FFmpeg, which is not available in this environment. You can upload an audio file directly instead."*

---

## 3. Waveform Array Layout and Stereo Downmixing

### 3.1 Canonical Layout Contract
When audio is decoded via `core.audio_loader.load_audio`:
- **Mono Audio:** 1-dimensional array of shape `(frames,)`.
- **Multichannel Audio:** 2-dimensional array of shape `(frames, channels)` where `channels <= 8` and `frames > channels`.

### 3.2 Stereo Downmix Fix & Verification
In Phase 2 review, a critical risk was flagged where averaging along the time axis could accidentally collapse frames into channels (e.g. producing 2 samples instead of 64,000).

`core.preprocessor.preprocess_waveform` enforces an explicit contract:
1. **Canonical `frames_first` (`channels_last`):**
   Input shape `(frames, channels)`. Downmixing performs `np.mean(audio, axis=1)`. The output has exactly `frames` samples.
2. **Alternate `channels_first` (`frames_last`):**
   Input shape `(channels, frames)`. Downmixing performs `np.mean(audio, axis=0)`. The output has exactly `frames` samples.
3. **Ambiguity Prevention:**
   If both dimensions are $\le 8$ or both $> 8$ (e.g. square matrices $4 \times 4$), the preprocessor rejects the input with an `InvalidWaveformError`, requiring the caller to specify `channel_layout` explicitly.
4. **Resampling Order:**
   Downmixing to mono occurs **before** polyphase resampling. This reduces resampling computation from $N$ channels to 1 channel and guarantees mathematical consistency.

---

## 4. Waveform Conditioning Contract

1. **Anti-Aliased Polyphase Resampling:**
   Converts any sampling rate $f_s$ to 16,000 Hz using `scipy.signal.resample_poly` with rational integer factors:
   $$g = \gcd(f_s, 16000), \quad \text{up} = \frac{16000}{g}, \quad \text{down} = \frac{f_s}{g}$$
2. **Length Normalization (Target = 64,000 samples / 4.0s @ 16 kHz):**
   - **Short Audio (< 4.0s):** Circular wrap padding (`mode='wrap'`) strictly matching the ASVspoof 2019 training protocol.
   - **Single-Window Long Audio (> 4.0s):** Truncates to the first 64,000 samples with a documented warning.
   - **Multi-Window Long Audio:** Handled by the windowing module described below.

---

## 5. Long-Recording Windowing Policy

For recordings exceeding 4.0 seconds, `core.windowing` provides continuous timeline segmentation:

### 5.1 Window Generation Algorithm
- **Window Size:** 64,000 samples (4.0 seconds at 16 kHz).
- **Hop Size:** 16,000 samples (1.0 second at 16 kHz, 75% overlap).
- **Tail-Anchoring Policy (100% Coverage):**
  Regular windows advance by `hop_samples`. If the remaining audio at the end is smaller than 4.0s, a final window is anchored at `[total_samples - 64000 : total_samples]`. If this window coincides with the previous window, it is omitted to prevent duplicates.
  **This guarantees 100% of audio samples from 0.0s to the final frame are evaluated without out-of-bounds indexing.**
- **Maximum Limit:** Enforces `MAX_WINDOWS_COUNT = 120` to prevent memory or CPU exhaustion.

### 5.2 Window-Level Scoring & Result Interpretation
- Each window is evaluated independently using the preloaded `V7InferenceEngine` (no checkpoint reloading).
- **Descriptive Timeline Summary:**
  - Reports count of windows classified as Real ($S \ge 0.4820$) vs. Fake ($S < 0.4820$).
  - Reports minimum score, maximum score, and mean score.
  - Reports exact start and end timestamps for each window.
- **Biometric Integrity Rule:** The system does **NOT** average scores to compute a fabricated "whole-recording probability". Natural speech exhibits acoustic variance across phonetic transitions; individual window scores reflect local cosine distance from the bona fide centroid.

---

## 6. Resource Limits and Operational Safeguards

Configured in `core.config`:

| Parameter | Value | Operational Purpose |
| :--- | :---: | :--- |
| `MAX_FILE_SIZE_BYTES` | 100 MB | Rejects oversized files before reading into RAM |
| `MAX_AUDIO_DURATION_SECONDS` | 600.0s (10 min) | Prevents runaway inference on massive files |
| `MAX_VIDEO_DURATION_SECONDS` | 600.0s (10 min) | Bounds video demuxing processing time |
| `MAX_WINDOWS_COUNT` | 120 windows | Caps long-recording analysis to 120 windows (~2 min timeline) |
| `FFMPEG_TIMEOUT_SECONDS` | 60.0s | Terminates hanging FFmpeg subprocesses |
| `MIN_AUDIO_DURATION_SECONDS` | 0.1s | Flags extreme micro-recordings |

---

## 7. Temporary File Management & Privacy

- Video audio extraction generates unique temporary directories using `tempfile.mkdtemp(prefix="v7_deepfake_video_")`.
- `core.media_ingestion.cleanup_prepared_media()` or the `media_context` context manager guarantees deletion of all extracted audio and temporary directories upon completion or failure.
- **Privacy:** Media files are processed purely locally. No audio, video, or metadata is transmitted externally.

---

## 8. Verification and Test Suite

### Running the Full Test Suite:
```powershell
python -m pytest "C:\Users\ANSH SINGH\Desktop\capstone1\v7_deepfake_app\tests" -v
```

### Verified Test Breakdown (62 Passed, 0 Failed):
- `tests/test_audio_loader.py` (11 tests): WAV, FLAC, MP3, OGG, M4A rejection, corrupt headers, file limits.
- `tests/test_video_processor.py` (4 tests): FFmpeg discovery, missing FFmpeg error, mocked extraction and cleanup.
- `tests/test_media_ingestion.py` (4 tests): Audio preparation, video rejection, context manager cleanup.
- `tests/test_windowing.py` (6 tests): Window generation, short padding, 10s timeline, tail anchoring, end-to-end inference.
- `tests/test_preprocessor.py` (15 tests): Canonical frames-first stereo, channels-first stereo, 5.1 surround, ambiguous rejection, polyphase resampling.
- `tests/test_decision_engine.py` (7 tests): Threshold boundary logic, score clamping, no percentage confidences.
- `tests/test_feature_extractor.py` (5 tests): Tensor shapes `[B, 3, 256, 397]`, channels dictionary, determinism.
- `tests/test_inference.py` (4 tests): End-to-end inference, error handling.
- `tests/test_model_loading.py` (5 tests): Strict key matching, parameter freezing, device binding.
- `tests/test_numerical_parity.py` (1 test): Bit-exact parity regression against original research code.
