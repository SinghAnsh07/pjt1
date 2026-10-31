# RELEASE MANIFEST: V7 Audio Authenticity Analyzer

**Release Name:** V7 Audio Authenticity Analyzer (Local Research Release)  
**Release Date:** 2026-10-20  
**Release Status:** Validated & Accepted for Local Demonstration and Research Use  
**Release Purpose:** Offline biometric deepfake screening, temporal timeline analysis, and acoustic forensic evaluation.  

---

## 1. System Requirements & Test Environment

- **Tested Operating System:** Windows 11 Enterprise (64-bit AMD64) (compatible with Linux x86_64 and macOS)
- **Tested Python Version:** Python 3.12.6 (compatible with Python 3.10+)
- **Hardware Profile:** Tested on CPU (supports CUDA via PyTorch when available)
- **Memory Footprint:** Minimum 4 GB RAM (8 GB recommended for multi-window audio)
- **Network Requirement:** 0 KB (100% offline, local-only execution; zero cloud APIs or telemetry)

---

## 2. Model Checkpoint Specifications

- **Checkpoint Relative Path:** `../v7/best_lcnn2d_v7.pt`
- **Checkpoint Absolute Path:** `C:\Users\ANSH SINGH\Desktop\capstone1\v7\best_lcnn2d_v7.pt`
- **File Size:** 770,309 bytes
- **Cryptographic SHA-256 Digest:** `b356dacc24dfc1ff463b1f1c5f4488cf374330f6c2edae6c5dd5abe79b76397b`
- **Neural Architecture:** 2D Fractional Fourier Transform + Morlet Wavelet + Light CNN (LCNN-2D) with Max-Feature-Map (MFM) activations and One-Class Softmax (OC-Softmax)
- **Parameter Statistics:** 170,336 total parameters; 0 trainable parameters in eval mode; 18,340 registered buffers

---

## 3. Installation & Launch Procedure

### 3.1 Dependency Installation
- **Core Runtime Packages:**
  ```bash
  pip install -r requirements.txt
  ```
  *Packages:* `torch==2.9.1`, `numpy==1.26.4`, `scipy==1.15.3`, `soundfile==0.14.0`, `streamlit==1.52.2`, `matplotlib==3.9.0`, `reportlab==5.0.1`.

- **Developer & Test Packages (Optional):**
  ```bash
  pip install -r requirements-dev.txt
  ```
  *Packages:* `pytest==6.2.5`, `pymupdf==1.24.1` (plus all runtime requirements).

### 3.2 Launch Methods
- **Recommended Windows Launch (One-Click):**
  Execute `run_app.bat` or double-click in Windows Explorer:
  ```cmd
  run_app.bat
  ```
- **Manual Launch:**
  ```bash
  python -m streamlit run app.py
  ```
  *Access URL:* `http://localhost:8501`

---

## 4. Operational Verification Commands

| Step | Objective | Command | Success Indicator |
| :--- | :--- | :--- | :--- |
| **1** | Checkpoint Cryptographic Integrity | `python scripts/verify_checkpoint.py` | Exit code 0, SHA-256 match confirmed |
| **2** | Backend Parity Smoke Test | `python scripts/verify_inference.py` | Exit code 0, 5/5 checks passed, $\Delta = 0.00000000$ |
| **3** | Full Automated Test Suite | `python -m pytest tests -v` | Exit code 0, 107 passed, 0 failed |

---

## 5. Media Format Compatibility

### 5.1 Native Formats (Zero External Dependencies)
Decoded directly using `soundfile` / `libsndfile`:
- `.wav` (PCM 16/24/32-bit, IEEE float 32-bit)
- `.flac` (Lossless Audio Codec)
- `.mp3` (MPEG Layer 3)
- `.ogg` (Ogg Vorbis)
- `.aiff` / `.aif` (Audio Interchange File Format)

### 5.2 Optional Formats (Requires Host FFmpeg)
When `ffmpeg` is available on the host `PATH`:
- Compressed Audio: `.m4a`, `.aac`, `.wma`, `.opus`
- Video Demuxing: `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`

*When FFmpeg is absent, the application gracefully operates in native-only audio mode with clear user guidance.*

---

## 6. Inventory of Core Application Runtime Files

```
v7_deepfake_app/
├── app.py                      # Primary Streamlit application entry point
├── run_app.bat                 # Windows batch launcher (venv + preflight + launch)
├── requirements.txt            # Pinned runtime dependencies
├── requirements-dev.txt        # Pinned developer & test dependencies
├── README.md                   # Comprehensive user guide and documentation
├── CHANGELOG.md                # Phase-by-phase development history
├── RELEASE_MANIFEST.md         # This manifest
├── FINAL_HANDOVER_CHECKLIST.md # Final quality assurance and handover checklist
├── PHASE9_FINAL_AUDIT_REPORT.md# Complete Phase 9 independent audit report
├── PHASE10_FINAL_HANDOVER_REPORT.md # Final project handover report
├── core/
│   ├── __init__.py             # Core package initialization
│   ├── audio_loader.py         # Defensive audio decoding & limit enforcement
│   ├── checkpoint_loader.py    # Strict model loading with weights_only=True
│   ├── config.py               # Application constants, sample rates & thresholds
│   ├── decision_engine.py      # Conservative ASVspoof threshold evaluator
│   ├── exceptions.py           # Domain-specific exception hierarchy
│   ├── feature_extractor.py    # Differentiable 2D ST-FrFT & Morlet wavelet extractor
│   ├── inference.py            # Unified V7InferenceEngine pipeline
│   ├── media_ingestion.py      # Unified media ingestion & context management
│   ├── model.py                # Pure PyTorch LCNN-2D architecture with MFM
│   ├── preprocessor.py         # Polyphase resampler, downmixer & wrap-padder
│   ├── schemas.py              # Typed dataclasses for results and metadata
│   ├── video_processor.py      # FFmpeg video demuxing sub-pipeline
│   └── windowing.py            # Sliding window generator & tail anchoring
├── reporting/
│   ├── __init__.py             # Public export interfaces
│   ├── json_export.py          # Standards-compliant UTF-8 JSON serializer
│   ├── pdf_export.py           # Publication-quality ReportLab PDF generator
│   └── report_schema.py        # Canonical ReportPayload & filename sanitizer
├── ui/
│   ├── __init__.py             # UI package initialization
│   ├── components.py           # Streamlit cards, metric widgets & cached download buttons
│   ├── plots.py                # Peak-preserving waveforms, spectrograms & timelines
│   └── styles.py               # Dark navy stylesheet (CSS)
└── scripts/
    ├── verify_checkpoint.py    # Standalone cryptographic SHA-256 verifier
    └── verify_inference.py     # Deterministic 5-case inference smoke test
```

---

## 7. Known Scientific & Operational Limitations

1. **Biometric Screening Classification:** The application provides automated biometric screening, **not legal or forensic proof**.
2. **Unvalidated Candidate Threshold:** The decision threshold ($\tau^* = 0.4820$) is an experimental operating point derived from the ASVspoof 2019 Logical Access evaluation partition. It is explicitly labeled **`CANDIDATE_UNVALIDATED`** and has not been calibrated on out-of-domain in-the-wild audio.
3. **Software Correctness vs. Model Accuracy:** 107 passing automated tests demonstrate software correctness and numerical determinism; they do **not** validate detection accuracy on novel, wild AI speech.
4. **Non-Probabilistic Scores:** One-Class Cosine scores $S(x) \in [-1.0, +1.0]$ are geometric similarity metrics relative to the bona fide centroid, not posterior probabilities or percentage confidences.
5. **Descriptive Visualizations:** Waveform envelopes, calibrated spectrograms, and 2D neural feature representations describe acoustic energy and network inputs; they do not causalize or certify authenticity.
