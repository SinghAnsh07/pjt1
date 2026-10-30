# FRFT–Wavelet Spatiotemporal Feature Fusion and 2D Light CNN Architecture for Robust Audio Deepfake Detection

This repository contains the complete experimental code, trained model checkpoints, diagnostic benchmarks, capstone project reports, and publication manuscripts for:

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
- **Checkpoint Footprint:** **770,309 bytes** on disk
- **Forensic Front-End:** Native GPU-vectorized 3-Channel Spatiotemporal Tensor ($\mathbf{T}_{\text{2D}} \in \mathbb{R}^{3 \times 256 \times 397}$) combining multi-angle Short-Time Fractional Fourier Transform (ST-FrFT $\alpha \in \{0.50, 0.85\}$) and Continuous Wavelet Scalograms ($s \in [1, 64]$).

---

## 📁 Repository Structure

```
capstone1/
├── paper/                                  # Research manuscripts, reports, and publication assets
│   ├── version7_single_sided/              # Finalized single-column research paper (12 pages)
│   │   ├── research_paper_v7_single_sided.pdf  # Publication PDF (12 pages, single-column)
│   │   ├── research_paper_v7_single_sided.docx # Formatted Word document (single-column)
│   │   ├── research_paper_v7_single_sided.tex  # XeTeX / Tectonic LaTeX source
│   │   ├── references_v7.bib                   # 25 verified peer-reviewed citations
│   │   ├── figures/                            # High-resolution figures (Fig. 1 to Fig. 5)
│   │   └── version7_scientific_audit.md        # Scientific reproducibility and audit report
│   ├── revised_final_report_submission/    # Submission-ready final capstone project reports
│   │   ├── Ansh_Singh_Capstone_Project_Report_Final.pdf   # 73-page individual report (PDF)
│   │   └── Ansh_Singh_Capstone_Project_Report_Final.docx  # 73-page individual report (DOCX)
│   ├── Ansh_Singh_Capstone_Project_Report_Final.pdf       # Synchronized root individual report (73 pages)
│   ├── Ansh_Singh_Capstone_Project_Report_Final.docx      # Synchronized root individual report (DOCX)
│   ├── Capstone_Project_Report_Final.pdf                  # Complete joint capstone report (75 pages)
│   ├── Capstone_Project_Report_Final.docx                 # Complete joint capstone report (DOCX)
│   ├── build_expanded_report.py                           # Multi-pass calibrated report compiler
│   ├── expanded_report_content.py                         # Report narrative and OMML equation module
│   └── figures/                                           # Report figures and diagnostic charts
│
├── teammate_guide/                         # 60/40 Collaborative work division documentation
│   ├── Project_Report_Guide_Teammate.pdf   # Formatted guide for teammate's independent report (PDF)
│   ├── Project_Report_Guide_Teammate.docx  # Editable guide for teammate's independent report (DOCX)
│   └── generate_teammate_guide.py          # Python automation generator for the teammate guide
│
├── notebooks/                              # Jupyter experimental research & training notebooks
│   ├── v3.ipynb                            # Champion 1D LCNN baseline (10.67% EER, 576k params)
│   ├── v4_fast.ipynb / v4_improved.ipynb   # SE-LCNN-1D attention experiments
│   ├── v5_oc_softmax.ipynb                 # 1D One-Class Softmax baseline (10.46% EER)
│   ├── v6-capstone.ipynb                   # 2D Conv spatiotemporal feasibility study
│   ├── v7_2d_frft_wavelet_lcnn.ipynb       # Champion 2D Spatiotemporal LCNN (9.25% EER, 170k params)
│   ├── v8_2d_frft_wavelet_ensemble.ipynb   # 2D Ensemble model exploration
│   └── capstoneproject2.ipynb              # Baseline exploratory notebooks
│
├── v7/                                     # Champion Version 7 model artifacts
│   ├── best_lcnn2d_v7.pt                   # PyTorch checkpoint (770 KB, 170,208 params)
│   ├── v7_2d_frft_wavelet_lcnn.py          # Standalone training and evaluation script
│   ├── v7_results.json                     # Complete metrics (EER, ROC-AUC, threshold, per-attack)
│   └── v7_results.png                      # 4-panel evaluation dashboard (ROC, scores, loss, history)
│
├── v4/ / v5/ / v6/ / v8/                   # Experimental iterations & diagnostic checkpoints
│   ├── v4/                                 # SE-LCNN-1D weights and parameter saturation analysis
│   ├── v5/                                 # 1D OC-Softmax weights and 1D moment ceiling analysis
│   ├── v6/                                 # 2D prototype spatiotemporal study
│   └── v8/                                 # 2D multi-seed ensemble evaluation
│
├── results_v1/                             # Initial baselines (V1–V3)
│   ├── best_lcnn_v3.pt                     # Champion 1D baseline model weights
│   └── features_v2/                        # Precomputed 704-D statistical moment feature arrays
│
├── scripts/                                # Project build, generation, and automation scripts
│   ├── paper_build/                        # LaTeX & Word builders for the research paper
│   │   ├── generate_v7_tex.py              # Generates research_paper_v7_single_sided.tex
│   │   └── build_v7_docx.py                # Generates research_paper_v7_single_sided.docx
│   ├── report_build/                       # Historical report assembly and content scripts
│   │   ├── rebuild_full_report.py          # Full report builder
│   │   ├── generate_report_figures.py      # Generates publication & report figures
│   │   └── update_report_content.py        # Content module updater
│   ├── notebook_generators/                # Reproducible notebook generation scripts
│   │   ├── generate_v6_nb.py               # Generates v6-capstone.ipynb
│   │   ├── generate_v7_nb.py               # Generates v7_2d_frft_wavelet_lcnn.ipynb
│   │   └── generate_v8_nb.py               # Generates v8_2d_frft_wavelet_ensemble.ipynb
│   └── scratch_tests/                      # Unit test scripts (OMML, tables, bullets, TOC)
│
├── reference_docs/                         # Project documentation, reference papers, and templates
│   ├── templates/                          # Official institutional report templates
│   │   └── BCE497J PROJECT REPORT TEMPLATE_V1.docx
│   ├── PROJECT_DOCUMENTATION.md            # Comprehensive system documentation
│   ├── Capstone_Project_Comprehensive_Documentation.docx
│   └── master_prompt_extracted.txt         # Version 7 scientific specification prompt
│
├── audits/                                 # Scientific reproducibility and originality audits
│   ├── research_audit.md                   # Systematic audit of all experimental claims
│   ├── originality_audit.md                # Plagiarism and originality audit report
│   ├── research_revision_log.md            # Chronological revision history
│   └── figure_placement_guide.md           # Layout and figure placement rules
│
├── slides/                                 # Defense presentation decks & slide automation
│   ├── output_slides/                      # Rendered presentation slides (PNG)
│   └── update_slides_with_chart.py         # Slide deck chart generator
│
├── tools/                                  # External tool binaries
│   └── tectonic.exe                        # Self-contained XeTeX/LaTeX compiler executable
│
└── backups/                                # Preserved timestamped project backups
```

---

## 🛠️ Quick Execution Guide

### 1. Compile the Research Paper (Single-Column LaTeX PDF, 12 Pages)
```powershell
cd paper\version7_single_sided
..\..\tools\tectonic.exe research_paper_v7_single_sided.tex
```

### 2. Regenerate the Research Paper Word Document (.docx)
```powershell
python scripts\paper_build\build_v7_docx.py
```

### 3. Build the Expanded Capstone Project Report (73 Pages)
```powershell
python paper\build_expanded_report.py
```

### 4. Execute Champion Model Training (Version 7)
```powershell
python v7\v7_2d_frft_wavelet_lcnn.py
```
