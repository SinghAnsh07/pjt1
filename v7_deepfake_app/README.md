# V7 Audio Authenticity Analyzer

A local, privacy-preserving desktop application for audio deepfake screening and forensic acoustic analysis powered by the trained **V7 2D Fractional Fourier Transform (FrFT) + Wavelet Packet + Light CNN (LCNN-2D)** model.

---

## 1. Project Overview

The **V7 Audio Authenticity Analyzer** evaluates speech recordings to assist forensic researchers and audio engineers in detecting synthetic voice cloning, text-to-speech (TTS), and voice conversion artifacts.

### Key Capabilities
- **Local Processing Guarantee:** All audio decoding, preprocessing, neural network inference, visualization generation, and document export run 100% locally on your machine. No waveforms, metadata, or user files are ever transmitted over external networks.
- **Biometric Screening Focus:** Formulated as a research screening tool adhering to ASVspoof 2019 logical access evaluation conventions.
- **Continuous Cosine Metric:** Scores represent geometric cosine similarity $S(x) \in [-1.0, +1.0]$ relative to the model's bona fide speech centroid. Higher values reflect genuine acoustic properties, while lower values suggest synthetic or manipulated speech patterns.
- **Sliding-Window Timeline:** Automatically segments extended recordings ($> 4.0\text{ s}$) into overlapping 4.0-second analysis windows with tail anchoring for complete temporal coverage.
- **Acoustic & Neural Visualizations:** Inspect peak-preserving waveform envelopes, calibrated power spectral density spectrograms (dB), temporal score trajectories, and the exact 3-channel 2D spatiotemporal tensors evaluated by the neural network.
- **Publication-Quality Exports:** Download self-contained multi-page PDF forensic reports and standardized machine-readable JSON payloads (`schema_version: "1.0.0"`).

---

## 2. System Requirements

### Host Environment
- **Operating System:** Windows 10/11 (64-bit), Linux (x86_64), or macOS. Tested on Windows 11 Enterprise AMD64.
- **Python Version:** Python 3.10, 3.11, or 3.12 (tested on Python 3.12.6 64-bit).
- **Disk Space:** ~500 MB for Python environment and dependencies; model checkpoint requires ~770 KB.
- **RAM:** Minimum 4 GB; 8 GB recommended for multi-window recordings.

### Software Dependencies
- **Core Runtime:** `torch>=2.0.0` (standard CPU/GPU PyPI package), `numpy>=1.24.0,<2.0.0`, `scipy>=1.11.0`, `soundfile>=0.12.0`, `streamlit>=1.30.0`, `matplotlib>=3.8.0`, `reportlab>=4.0.0`.
- **Optional External Dependency (FFmpeg):** Required only if decoding `.m4a` / `.aac` audio files or demuxing audio tracks from video containers (`.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`). When FFmpeg is not installed, the application runs in audio-only mode supporting native `.wav`, `.flac`, `.mp3`, `.ogg`, and `.aiff` files.

---

## 3. Installation Guide

### Step 1: Open Terminal in Application Directory
Open PowerShell or Command Prompt:

```powershell
cd "C:\Users\ANSH SINGH\Desktop\capstone1\v7_deepfake_app"
```

### Step 2: Create and Activate a Virtual Environment

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt (CMD):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### Step 3: Install Core Dependencies
Install verified PyPI packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

*(Optional) To run developer tests and visual audit tools, also install `requirements-dev.txt`:*
```bash
pip install -r requirements-dev.txt
```

### Step 4: Verify Model Checkpoint Integrity
The pre-trained model weights are expected at `../v7/best_lcnn2d_v7.pt`. Run the standalone verification utility:

```bash
python scripts/verify_checkpoint.py
```

Expected output:
```
========================================================================
  V7 MODEL CHECKPOINT INTEGRITY VERIFICATION
========================================================================
Target Checkpoint : C:\Users\ANSH SINGH\Desktop\capstone1\v7\best_lcnn2d_v7.pt
File Size         : 770,309 bytes
Calculated SHA-256: b356dacc24dfc1ff463b1f1c5f4488cf374330f6c2edae6c5dd5abe79b76397b
Expected SHA-256  : b356dacc24dfc1ff463b1f1c5f4488cf374330f6c2edae6c5dd5abe79b76397b

>>> VERIFICATION SUCCESS: Checkpoint integrity confirmed. <<<
========================================================================
```

---

## 4. Launching the Application

### Option A: Using the Windows Batch Launcher (Recommended for Windows)
Double-click `run_app.bat` or run it from any terminal:

```cmd
run_app.bat
```

The launcher will automatically:
1. Detect local virtual environments (`.venv` or `venv`) or fall back to system Python.
2. Validate required package dependencies.
3. Verify the model checkpoint cryptographic hash.
4. Launch the Streamlit interface and open your default browser.

### Option B: Manual Command-Line Launch
With your virtual environment activated:

```bash
python -m streamlit run app.py
```

The interface will initialize and display the local URL:
```
Local URL: http://localhost:8501
```

To stop the application, return to the terminal and press `Ctrl+C`.

---

## 5. User Workflow & Features

```
 [Upload Audio] ➔ [Inspect Metadata] ➔ [Analyze Audio] ➔ [Examine Findings & Tabs] ➔ [Download PDF/JSON]
```

### 1. Uploading Audio
- Select or drag-and-drop a file into the upload zone.
- **Supported Native Formats:** WAV (PCM 16/24/32-bit, float), FLAC, MP3, OGG (Vorbis), AIFF.
- **Safety Limits:** Maximum file size: 100 MB; Maximum duration: 10 minutes (600s); Minimum duration: 0.1s.
- Upon upload, an interactive audio player and an Ingestion Metadata card will appear displaying duration, original sampling rate, channel layout, container type, and estimated analysis windows.

### 2. Executing Analysis
- Click **"Analyze Audio"** to start the evaluation.
- The pipeline polyphase-resamples the audio to 16,000 Hz, averages multichannel stereo signals to mono, normalizes duration, computes FrFT and Morlet wavelet transforms, and executes the LCNN-2D classifier.

### 3. Understanding Plain-Language Results
- **Plain-Language Verdict:**
  - When matching genuine speech patterns: displays **"Patterns more consistent with genuine speech"** (Likely Genuine Audio).
  - When exhibiting divergent patterns: displays **"Unusual speech patterns detected"** (Potential AI-Generated or Manipulated Audio).
  - Explicit prominent caveats accompany both verdicts explaining that results reflect learned acoustic similarity, not legal or definitive forensic proof.
- **Results Summary Grid:** A compact card showing recording duration, number of analyzed segments, screening finding, model similarity score, and measured analysis latency.
- **Model Similarity Score:** Continuous similarity metric in range $[-1.0, +1.0]$. The interface intentionally **does not display fabricated percentage probabilities** because the model output is geometric distance, not a calibrated probability.
- **Candidate Threshold:** Displays the reference decision boundary ($\tau^* = 0.4820$) derived from the ASVspoof 2019 Logical Access benchmark (labeled unvalidated on in-the-wild audio).
- **Multi-Window Timeline Breakdown:** For recordings $> 4.0\text{ s}$, summarizes window counts (Total, Likely Genuine, Potential Fake), mean/range scores, and timeline coverage.
- **Technical Transparency:** An expandable panel (*"How this result was produced"*) reveals full LCNN-2D model parameters, 3-channel spatiotemporal tensor dimensions, and millisecond timing breakdowns.

### 4. Audio Signal Insights (Visual Workspace)
- **🌊 Waveform:** Interactive switch between the original decoded waveform and the prepared 16 kHz mono waveform. Employs peak-preserving envelope decimation to prevent browser DOM lag.
- **🌈 Spectrogram:** Calibrated time-frequency power spectral density with logarithmic decibel (dB) scale and 80 dB dynamic range clamping.
- **⏱️ Analysis Timeline:** For long recordings, plots per-window cosine scores against time with an explicit reference line at candidate threshold ($\tau^* = 0.4820$) and includes an interactive segment inspector slider.
- **🧬 Advanced Model Views:** Clearly labeled research exploration tab inspecting the 3-channel 2D spatiotemporal representations $(3 \times 256 \times 397)$ processed by the LCNN-2D convolutional backbone:
  - Channel 0: Short-Time Fractional Fourier Transform (ST-FrFT $\alpha=0.50$, chirp angle $45^\circ$)
  - Channel 1: ST-FrFT ($\alpha=0.85$, chirp angle $76.5^\circ$)
  - Channel 2: Continuous Morlet Wavelet Scalogram (64 scales / 256 frequency bins)

### 5. Report Downloads
- **📄 Download PDF Analysis Report:** Generates an in-memory, publication-quality multi-page PDF document featuring executive summary cards, metadata, paginated multi-window tables (`repeatRows=1`), embedded high-resolution figures, and explicit forensic disclosures. Single-window reports fit in 2 pages; multi-window reports fit in 3 pages.
- **📊 Download JSON Analysis Data:** Generates machine-readable UTF-8 JSON conforming to versioned schema `1.0.0`. Recursively sanitizes non-finite values (`NaN`, `Infinity` mapped to `null`).
- **Privacy Hardening:** All local absolute paths (`C:\...`, `/home/`) and internal temporary directory prefixes are completely scrubbed from exports. Raw audio sample arrays are omitted.
- **Session Caching:** Report artifacts are cached in session state upon initial generation; clicking download buttons does not re-invoke model inference.

### 6. Reset & Replacement
- Clicking **"Reset"** or uploading a different audio file immediately wipes analysis results, cached report payloads, and selected-window state, preventing stale findings from persisting.

---

## 6. Project Architecture

```
v7_deepfake_app/
├── app.py                      # Primary Streamlit application entry point
├── run_app.bat                 # One-click Windows application launcher
├── requirements.txt            # Core runtime dependencies
├── requirements-dev.txt        # Optional developer & test dependencies
├── README.md                   # Comprehensive user guide and documentation
├── CHANGELOG.md                # Phase-by-phase development history
├── RELEASE_MANIFEST.md         # Release manifest & system specifications
├── FINAL_HANDOVER_CHECKLIST.md # Final quality assurance & handover checklist
├── PHASE9_FINAL_AUDIT_REPORT.md# Complete Phase 9 independent audit report
├── PHASE10_FINAL_HANDOVER_REPORT.md # Final project handover report
├── core/                       # Machine learning & signal processing core
│   ├── config.py               # Operational constants, sampling rates & thresholds
│   ├── exceptions.py           # Domain exception taxonomy
│   ├── schemas.py              # Strongly typed dataclasses
│   ├── audio_loader.py         # Defensive audio decoding via SoundFile
│   ├── preprocessor.py         # Polyphase resampling, mono downmixing & wrap-padding
│   ├── feature_extractor.py    # Differentiable 2D ST-FrFT & Morlet wavelet extractor
│   ├── model.py                # Isolated PyTorch LCNN-2D neural architecture with MFM
│   ├── checkpoint_loader.py    # Secure LCNN-2D weights loader (weights_only=True)
│   ├── decision_engine.py      # ASVspoof candidate threshold evaluator
│   ├── inference.py            # Unified V7InferenceEngine
│   ├── video_processor.py      # Optional FFmpeg video demuxing wrapper
│   ├── media_ingestion.py      # Media routing & temporary-file lifecycle
│   └── windowing.py            # Multi-window sliding analysis & tail anchoring
├── ui/                         # Streamlit presentation & styling layer
│   ├── styles.py               # Dark navy/slate CSS stylesheet
│   ├── plots.py                # Headless Matplotlib signal & timeline figure generators
│   └── components.py           # Modular UI widgets, cards, tabs, and export controls
├── reporting/                  # Export & document generation layer
│   ├── __init__.py             # Public export interfaces
│   ├── report_schema.py        # Canonical ReportPayload & filename sanitization
│   ├── json_export.py          # Standards-compliant UTF-8 JSON serializer
│   └── pdf_export.py           # Multi-page ReportLab PDF generator with NumberedCanvas
├── scripts/
│   ├── verify_checkpoint.py    # Standalone SHA-256 cryptographic validator
│   ├── verify_inference.py     # Backend parity smoke test & numerical check
│   └── visual_acceptance_audit.py # Headless Selenium visual screenshot capturer
└── tests/                      # Automated test suite (107 passing tests)
    ├── conftest.py             # Shared fixtures (synthetic waveforms, dummy models)
    ├── test_audio_loader.py    # Audio decoding & format boundary tests
    ├── test_decision_engine.py
    ├── test_feature_extractor.py
    ├── test_inference.py
    ├── test_media_ingestion.py
    ├── test_model_loading.py
    ├── test_numerical_parity.py
    ├── test_phase7_acceptance.py # End-to-end acceptance, PDF inspection & parity tests
    ├── test_preprocessor.py
    ├── test_reporting.py       # JSON schema, privacy, PDF rendering & feature parity tests
    ├── test_ui.py              # Streamlit UI AppTest & component tests
    ├── test_video_processor.py
    ├── test_visualization.py   # Waveform, spectrogram, timeline & feature view tests
    └── test_windowing.py
```

---

## 7. Running Verification & Smoke Tests

### Backend Parity Smoke Test
```bash
python scripts/verify_inference.py
```
Validates model loading in strict eval mode, runs 4 deterministic synthetic test cases, checks timing breakdowns, and verifies bit-exact numerical reproducibility ($\Delta = 0.00000000$).

### Full Automated Test Suite (107 Tests)
With developer dependencies installed (`pip install -r requirements-dev.txt`):

```bash
python -m pytest tests -v
```

All 107 tests across ingestion, windowing, inference, numerical parity, UI state, visualizations, reporting, and acceptance must pass.

> [!NOTE]
> Synthetic audio fixtures (`test_artifacts/fixtures/`) validate correct software pipeline execution, numerical reproducibility, and exception handling; they do **not** evaluate model classification accuracy on unconstrained human speech.

---

## 8. Troubleshooting

| Issue / Error | Probable Cause | Action / Resolution |
| :--- | :--- | :--- |
| **`ImportError: No module named 'torch'`** | Virtual environment not activated or dependencies missing. | Run `.\.venv\Scripts\Activate.ps1` followed by `pip install -r requirements.txt`. |
| **`Checkpoint verification failed`** | Model checkpoint is missing, corrupted, or altered. | Verify that `best_lcnn2d_v7.pt` exists in `../v7/` and its SHA-256 is `b356dacc...`. |
| **`Unsupported format (.m4a / video)`** | FFmpeg is not installed on the system PATH. | Use standard audio formats (`.wav`, `.flac`, `.mp3`, `.ogg`, `.aiff`) or install FFmpeg to enable `.m4a` and video extraction. |
| **`File too large error`** | Uploaded file exceeds 100 MB. | Compress the audio or trim the recording below 100 MB before uploading. |
| **`Port 8501 is already in use`** | Another Streamlit instance is running. | Specify an alternate port: `python -m streamlit run app.py --server.port=8502`. |
| **`Streamlit rerun slow during export`** | Export cache invalidated. | Reports are cached after initial generation. Allow 1–2 seconds for initial PDF compilation; subsequent downloads are instantaneous. |

---

## 9. Scientific & Forensic Limitations

1. **Screening Nature:** This application is a machine-learning research screening tool. Its classification output **does not constitute definitive legal or forensic proof** of authenticity or voice manipulation.
2. **Unvalidated Candidate Threshold:** The decision threshold ($\tau^* = 0.4820$) was empirically derived from the ASVspoof 2019 Logical Access evaluation partition. It has **not been independently calibrated** on unconstrained, in-the-wild audio (e.g. phone recordings, messaging app compression, social media audio).
3. **Software Correctness vs. Real-World Accuracy:** Passing all 107 automated software and numerical tests proves that the codebase executes reliably and deterministically; it **does not demonstrate accuracy on wild voice clones**.
4. **Descriptive Role of Visualizations:** Acoustic amplitude traces, spectrograms, and 2D neural feature views illustrate signal energy and mathematical representations; they are descriptive tools, not causal explanations of authenticity.
5. **Absence of Calibrated Probabilities:** The model outputs continuous One-Class Cosine similarity scores $S(x) \in [-1.0, +1.0]$, representing geometric distances to the genuine reference centroid. Scores cannot be interpreted as calibrated posterior probabilities or percentage confidences.
6. **No Forensic Certification:** The system must not be used as the sole basis for legal, judicial, or punitive determinations.

---

## 10. Release & Audit Documentation

For detailed technical specifications, quality assurance records, and audit history, refer to:
- [`RELEASE_MANIFEST.md`](RELEASE_MANIFEST.md): Release manifest, environment specifications, and runtime file inventory.
- [`FINAL_HANDOVER_CHECKLIST.md`](FINAL_HANDOVER_CHECKLIST.md): Itemized verification checklist across installation, runtime, integrity, and documentation.
- [`PHASE10_FINAL_HANDOVER_REPORT.md`](PHASE10_FINAL_HANDOVER_REPORT.md): Final software handover report and release verdict.
- [`PHASE9_FINAL_AUDIT_REPORT.md`](PHASE9_FINAL_AUDIT_REPORT.md): Independent release audit, reproducibility verification, and defect closure.
- [`CHANGELOG.md`](CHANGELOG.md): Chronological history of all development phases and bug fixes.
