# FRFT–Wavelet Spatiotemporal Feature Fusion and 2D Light CNN Architecture for Robust Audio Deepfake Detection

This repository contains the complete experimental research code, trained PyTorch model checkpoints, forensic diagnostic benchmarks, capstone project reports, publication manuscripts, and interactive web application for:

> **FRFT–Wavelet Spatiotemporal Feature Fusion and 2D Light CNN Architecture for Robust Audio Deepfake Detection**  
> **Authors:** Advit Bhutani, Ansh Singh, and Dr. Pattabiraman V  
> *School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology, Chennai, India*

---

## 🏆 Key Research Highlights (Version 7 Champion System)

- **Development EER:** **3.14%** (3.1397% at Epoch 17)
- **Evaluation EER:** **9.25%** (9.2454% continuous linear interpolation; **9.26%** discrete ROC)
- **Evaluation ROC-AUC:** **0.9496** (0.949559)
- **Evaluation Accuracy / Balanced Accuracy:** **90.75%** (64,651 / 71,237 correct across 13 unseen zero-day attacks)
- **Trainable Parameters:** **170,208** (~680 KB, a **70.47% parameter reduction** compared to earlier 1D baselines)
- **Model Checkpoint Footprint:** **770,309 bytes** on disk (`v7/best_lcnn2d_v7.pt`)
- **Forensic Front-End:** Native GPU-vectorized 3-Channel Spatiotemporal Tensor ($\mathbf{T}_{\text{2D}} \in \mathbb{R}^{3 \times 256 \times 397}$) combining multi-angle Short-Time Fractional Fourier Transform (ST-FrFT $\alpha \in \{0.50, 0.85\}$) and Continuous Morlet Wavelet Scalograms ($s \in [1, 64]$).

---

## 📁 Repository Structure

```
pjt1/
├── paper/                                  # Capstone project reports and research manuscripts
│   ├── Ansh_Singh_Capstone_Project_Report_Final.pdf   # 73-page individual capstone report (PDF)
│   ├── Ansh_Singh_Capstone_Project_Report_Final.docx  # 73-page individual capstone report (DOCX)
│   ├── research_paper_final.pdf                       # Publication research paper
│   └── figures/                                       # High-resolution publication figures & diagrams
│
├── v7_deepfake_app/                        # Full interactive forensic web application
│   ├── app.py                              # Streamlit application entry point
│   ├── run_app.bat                         # Quick launcher script for Windows
│   ├── requirements.txt                    # App runtime dependencies
│   ├── core/                               # Core audio ingestion, feature extraction & inference engine
│   ├── frontend/                           # Precompiled React/Vite aesthetic components
│   ├── reporting/                          # PDF and JSON export generation modules
│   ├── tests/                              # Comprehensive unit and integration test suite
│   ├── ui/                                 # Streamlit UI layouts, charts, and styling
│   └── UI_REDESIGN_SCREENSHOTS/            # Verification screenshots of the application
│
├── notebooks/                              # Experimental research & training notebooks
│   ├── v1_baseline_frft_448d.ipynb         # 448-D statistical moment baseline
│   ├── v2_overfitting_mixup_focal_996d.ipynb# Mixup & focal loss regularization
│   ├── v3_champion_1d_lcnn_704d.ipynb      # Champion 1D LCNN baseline (10.67% EER, 576k params)
│   ├── v3_champion_1d_lcnn_draft.ipynb     # 1D LCNN architecture exploration
│   ├── v4_se_lcnn_attention_ensemble.ipynb # SE-LCNN attention experiments
│   ├── v4_se_lcnn_fast.ipynb               # Fast SE-LCNN pipeline
│   ├── v5_oc_softmax_metric_learning.ipynb # 1D One-Class Softmax baseline (10.46% EER)
│   ├── v5_oc_softmax_metric_learning.py    # Standalone OC-Softmax script
│   ├── v6_2d_spectrogram_feasibility.ipynb # 2D Conv spatiotemporal feasibility study
│   └── v7_champion_2d_spatiotemporal_lcnn.ipynb # Champion 2D Spatiotemporal LCNN (9.25% EER)
│
├── v7/                                     # Champion Version 7 model artifacts
│   ├── best_lcnn2d_v7.pt                   # PyTorch checkpoint (770 KB, 170,208 params)
│   ├── v7_2d_frft_wavelet_lcnn.py          # Standalone training and evaluation script
│   ├── v7_results.json                     # Complete metrics (EER, ROC-AUC, thresholds, per-attack)
│   └── v7_results.png                      # 4-panel evaluation dashboard (ROC, scores, loss, history)
│
├── v1/ to v6/                              # Iterative experimental stages & baselines
│   ├── v1/                                 # V1 baseline notebooks and curves
│   ├── v2/                                 # V2 regularization notebooks
│   ├── v3/                                 # V3 champion 1D baseline model checkpoint (2.3 MB)
│   ├── v4/                                 # V4 SE-LCNN weights archive and logs
│   ├── v5/                                 # V5 OC-Softmax baseline metrics and plots
│   └── v6/                                 # V6 2D feasibility study notebook
│
├── results_v1/                             # Baseline artifacts & notebooks
│   ├── best_lcnn_v3.pt                     # 1D baseline model weights
│   ├── capproject_v1.ipynb                 # Initial baseline notebook
│   └── ablation_results.png                # Ablation charts
│
├── v7-capstone.ipynb                       # Root synchronized Version 7 notebook
├── README.md                               # Project documentation
└── .gitignore                              # Git exclusion rules
```

---

## 🔬 Experimental Evolution

| Version | Architecture | Front-End Features | Trainable Params | Dev EER | Eval EER | Eval ROC-AUC |
|:-------:|:-------------|:-------------------|:----------------:|:-------:|:--------:|:------------:|
| **V1**  | MLP Baseline | 448-D Statistical Moments | 128k | 18.40% | 24.10% | 0.8120 |
| **V2**  | MLP + Mixup + Focal | 996-D Multi-Order Moments | 256k | 14.20% | 19.80% | 0.8650 |
| **V3**  | 1D Light CNN (LCNN-1D) | 704-D Statistical Moments | 576k | 4.82% | 10.67% | 0.9312 |
| **V4**  | SE-LCNN-1D (Squeeze-and-Excitation) | 704-D Moments + Attention | 612k | 4.65% | 10.52% | 0.9340 |
| **V5**  | 1D LCNN + One-Class Softmax (OC-Softmax) | 704-D Statistical Moments | 576k | 4.51% | 10.46% | 0.9358 |
| **V6**  | 2D Prototype CNN | STFT Power Spectrogram | 340k | 5.10% | 12.30% | 0.9180 |
| **V7 (Champion)** | **2D Light CNN (LCNN-2D) + MFM** | **3-Channel ST-FrFT + Wavelet Tensor** | **170k** | **3.14%** | **9.25%** | **0.9496** |

---

## 🚀 Quick Start Guide

### 1. Launch the Deepfake Detection Web Application

The interactive web application provides a complete dashboard for audio screening, multi-window timeline analysis, forensic spectrogram views, and PDF/JSON export generation:

```powershell
cd v7_deepfake_app

# Create and activate a Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install application dependencies
pip install -r requirements.txt

# Launch the Streamlit application
streamlit run app.py
```

Alternatively, run `run_app.bat` on Windows.

---

### 2. Train or Evaluate the Champion V7 Model

To run the standalone Version 7 2D Spatiotemporal LCNN training and evaluation pipeline:

```powershell
python v7\v7_2d_frft_wavelet_lcnn.py
```

---

### 3. Explore Jupyter Research Notebooks

Launch Jupyter to explore the step-by-step experimental notebooks:

```powershell
jupyter notebook notebooks/
```

- **Champion 2D System:** [`notebooks/v7_champion_2d_spatiotemporal_lcnn.ipynb`](notebooks/v7_champion_2d_spatiotemporal_lcnn.ipynb)
- **Champion 1D Baseline:** [`notebooks/v3_champion_1d_lcnn_704d.ipynb`](notebooks/v3_champion_1d_lcnn_704d.ipynb)
- **Metric Learning (OC-Softmax):** [`notebooks/v5_oc_softmax_metric_learning.ipynb`](notebooks/v5_oc_softmax_metric_learning.ipynb)

---

## 📄 Publications & Project Reports

The complete capstone reports and research publication manuscripts are available in [`paper/`](paper/):
- **Final Capstone Project Report (73 Pages, PDF):** [`paper/Ansh_Singh_Capstone_Project_Report_Final.pdf`](paper/Ansh_Singh_Capstone_Project_Report_Final.pdf)
- **Final Capstone Project Report (DOCX):** [`paper/Ansh_Singh_Capstone_Project_Report_Final.docx`](paper/Ansh_Singh_Capstone_Project_Report_Final.docx)
- **Publication Research Paper:** [`paper/research_paper_final.pdf`](paper/research_paper_final.pdf)

---

## 📜 Citation

If you use this codebase or research artifacts, please cite:

```bibtex
@article{bhutani2026frft,
  title={FRFT--Wavelet Spatiotemporal Feature Fusion and 2D Light CNN Architecture for Robust Audio Deepfake Detection},
  author={Bhutani, Advit and Singh, Ansh and Pattabiraman, V.},
  journal={School of Computer Science and Engineering, Vellore Institute of Technology, Chennai},
  year={2026}
}
```
