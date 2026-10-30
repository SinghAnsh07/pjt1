# Manuscript Revision Log and Change Audit

**Project Title:** FRFT–Wavelet Feature Fusion for Audio Deepfake Detection Using a Lightweight Competitive Neural Network  
**Date of Revision:** October 10, 2026  
**Auditor / Lead Editor:** Automated Machine-Learning and Signal-Processing Experimental Audit System  
**Target Submission Format:** IEEE Two-Column Conference Style Manuscript  
**Affected Files:**
- `research_paper.tex`
- `research_paper_draft.docx`
- `research_audit.md`
- `figure_placement_guide.md`
- `fig1_framework.png`
- `build_docx.py`

---

## 1. Summary of Major Methodological and Scientific Corrections

| Revision ID | Manuscript Section | Previous / Problematic State | Corrected State | Scientific & Evidentiary Rationale |
| :---: | :--- | :--- | :--- | :--- |
| **REV-01** | Title, Abstract, Section I, III, IV | Described classifier as a "Light Convolutional Neural Network (LCNN)" with "4 convolutional blocks". | Re-classified as a **Max-Feature-Map Deep Competitive Neural Network (MFM-DNN)** adapting the LCNN paradigm to fixed 704-D feature vectors. | PyTorch inspection of Cell 10 and `best_lcnn_v3.pt` state dict confirms all blocks use `nn.Linear` layers with MFM activations, BatchNorm1d, Dropout, and residual skip connections, rather than 1D/2D convolutions. |
| **REV-02** | Abstract, Section V-C, Conclusion | Claimed an "identical ablation protocol" between the champion system and single-lens baselines. | Acknowledged that sub-ablations were trained for **25 epochs** under `CosineAnnealingLR`, while the champion fused system underwent **40 epochs** under `CosineAnnealingWarmRestarts`. | Notebook Cell 13 code proves differing epoch counts and schedulers. Described the 2.61% EER delta as an indicative/exploratory comparison rather than strictly controlled causal proof. |
| **REV-03** | Abstract, Section IV-C, Section V-B | Emphasized 85.61% raw accuracy as strong standalone proof of system efficacy. | Elevated **Balanced Accuracy (90.26%)**, Sensitivity (96.13%), and Specificity (84.40%) to primary status alongside EER (10.67%). | Evaluation set exhibits an 89.68% majority spoof prevalence. A naive "all-spoof" baseline gets 89.68% accuracy with 0% genuine recall; raw accuracy alone is misleading. |
| **REV-04** | Section III-C, III-D, Section V-D | Claimed FrFT "directly captures complex phase information and boundary phase jumps". | Clarified that FrFT features are extracted from the **magnitude spectrum** $\|X_\alpha[k]\|$, capturing **energy redistribution across oblique chirp projection axes**. | Code inspection confirms `mag = np.abs(transformed)` is chunked; raw complex phase angles are not preserved in the feature vector. |
| **REV-05** | Section III-D | Stated that WPD sub-bands represent "strictly monotonic 0–8 kHz frequency bands (250 Hz per band)". | Clarified that WPD level-5 decomposition uses PyWavelets **natural dyadic tree order** (`'natural'`), which exhibits recursive dyadic branching rather than monotonic frequency sorting. | Traced `wp.get_level(level, 'natural')` in Cell 7. Noted that dyadic order captures multi-resolution wave packets across the entire Nyquist band. |
| **REV-06** | Section IV-C, Section V-B, Table V | Table V listed an unverified EER threshold of $\tau^* = 0.584$. | Corrected EER threshold to $\tau^* = \mathbf{0.8362}$ on $P(\text{bonafide})$, supported by both nearest ROC point and linear interpolation. | Independent Python recomputation confirmed nearest discrete threshold is 0.836186 and interpolated threshold is 0.836169 (both rounding to 0.8362), achieving FAR = FRR = 10.67%. |
| **REV-07** | Section IV-B, Table II | Table II contained slight arithmetic discrepancies in layer parameter decompositions. | Fixed decomposition formulas: Block 1 = 361,472; Block 2 = 132,096; Block 3 = 66,048; Block 4 = 16,640; Head = 130; exactly summing to **576,386 trainable parameters**. | Enumerated exact linear weights ($W$), biases ($b$), and BatchNorm learnable affine parameters ($\gamma, \beta$) across state dict tensors. |
| **REV-08** | Section V-B, Table V | Confusion matrix was described in prose without explicit orientation or rate definitions. | Formatted confusion matrix at $\tau = 0.50$ explicitly: TN = 53,919, FP = 9,963, FN = 285, TP = 7,070, with rows as true class and columns as predicted class. | Verified that FAR is 15.60% and FRR is 3.87%. Added security discussion explaining why 15.60% FAR requires threshold tuning for high-security applications. |
| **REV-09** | Section V-E, Table IV | Described Version 3 feature reduction (996 to 704 features) as a blind pre-registered choice. | Transparently disclosed that discarding LFCC and Spectral features was informed by the post-hoc diagnostic of Version 2's catastrophic overfitting (Dev EER 0.05% vs Eval EER 13.55%). | Enhances scientific integrity by acknowledging the iterative development history of the capstone project. |
| **REV-10** | Section VI (Limitations) | Generic boilerplate limitations. | Formulated 7 substantive, technically grounded limitations: lack of cross-dataset codec testing, fixed heuristic FrFT/WPD parameters, ablation epoch differences, 15.60% FAR at default threshold, post-hoc feature selection history, dense vs convolutional trade-offs, and feature extraction computational cost. | Directly addresses reviewer inquiries regarding real-world biometric deployment and experimental rigor. |
| **REV-11** | References & LaTeX Citations | 24 out of 25 bib entries cited in LaTeX; `rawnet_jung2019` was unreferenced. | Actively integrated citation to `rawnet_jung2019` in Section II-E alongside `rawnet2_tak2021`. All 25 references in `references.bib` verified. | Ensured 100% active utilization of the verified bibliography with zero missing or orphaned keys. |
| **REV-12** | Figure Dependencies & Paths | Figures used subfolder paths (`results_v1/...`) which caused compiler failures in environments requiring flat paths. | Copied genuine project figures (`training_curves.png`, `audio_visualization.png`, `ablation_results.png`) to root directory, set `\graphicspath{{./}{results_v1/}}`, and verified all paths. | Guarantees build compatibility across local TeX engines, Overleaf, and IEEE PDF eXpress. |

---

## 2. Section-by-Section Change Audit

### Title and Author Block
- Retained requested working title: `FRFT--Wavelet Feature Fusion for Audio Deepfake Detection Using a Lightweight CNN`.
- Preserved institutional affiliations and emails for authors Advit Bhutani, Ansh Singh, and supervisor Dr. Pattabiraman V.

### Abstract
- Integrated Balanced Accuracy (**90.26%**) alongside raw accuracy (**85.61%**) and EER (**10.67%**).
- Qualified classifier as an **MFM-based Deep Competitive Neural Network (MFM-DNN)** adapting the LCNN Max-Feature-Map principle to 704-D acoustic feature vectors.
- Replaced claim of an "identical ablation protocol" with an accurate description of exploratory single-lens comparisons.

### Section I: Introduction
- Added explicit mathematical definitions of the Heisenberg-Gabor resolution constraint and high-frequency psychophysical compression.
- Itemized five verified scientific contributions, emphasizing parameter efficiency (576,386 params, $\sim$2.3 MB) and balanced biometric evaluation.

### Section II: Related Work
- Structured survey into five subsections: Conventional representations, FrFT, WPD, Lightweight classifiers, and Multi-lens fusion.
- Full-page literature table (Table VI) contrasting 10 representative published systems on ASVspoof 2019 LA.
- Added active citation to RawNet [12] alongside RawNet2 [13].

### Section III: Proposed Methodology
- **Preprocessing:** Documented 16 kHz sampling, mono averaging, and 6.0 s duration (96,000 samples) with front-truncation and zero-padding.
- **FrFT Formulation:** Explicitly stated discrete chirp-premultiplication implementation and clarified that descriptors are extracted over the magnitude distribution $|X_\alpha[k]|$.
- **WPD Formulation:** Defined dyadic decomposition and natural tree order across 32 sub-bands.
- **Statistical Descriptors:** Formulated all 8 descriptors with probability mass normalization.
- **Classifier Architecture:** Detailed layer-by-layer MFM-DNN structure with input dropout (0.10) and residual Block 2 skip connection.
- **Training Algorithm:** Formalized complete pseudocode in Algorithm 1, detailing class weights ($[0.557, 4.919]$), label smoothing ($\epsilon=0.10$), and CosineAnnealingWarmRestarts.

### Section IV: Experimental Setup
- **Dataset Table (Table I):** Formally detailed Train (25,380), Dev (24,844), and Eval (71,237) partitions, clearly delineating known attacks (A01–A06) from 13 unseen algorithms (A07–A19).
- **Architecture Table (Table II):** Exact parameter decompositions matching PyTorch state dict.
- **Metric Definitions:** Formalized EER, FAR, FRR, sensitivity, specificity, and balanced accuracy.

### Section V: Results and Discussion
- **Training Curves (Fig. 3):** Verified loss descent ($0.90 \to 0.51$) and Dev EER convergence (6.00% at Epoch 38).
- **Evaluation Performance (Table V):** Fully verified confusion matrix and class report at threshold 0.50.
- **Security Discussion:** Analyzed biometric trade-offs between default threshold ($\tau = 0.50$, $\text{FAR} = 15.60\%$) and EER threshold ($\tau^* = 0.8362$, $\text{FAR} = 10.67\%$).
- **Ablation Benchmarks (Table III & Fig. 4):** Reported verified metrics with honest disclosure of 25 vs 40 training epochs.
- **Signal Visualizations (Fig. 2 & Fig. 5):** Multi-panel time, FFT, and FrFT comparisons.
- **Version Evolution (Table IV):** Detailed progression from v1 (proof of concept) to v2 (overfitting failure) to v3 (champion system).

### Section VI: Limitations and Threats to Validity
- Comprehensive, non-boilerplate discussion of 7 real technical constraints.

### Section VII: Conclusion and Future Work
- Summarized verified findings with conservative scientific phrasing; outlined future work including differentiable FrFT angles and cross-codec evaluation.

### References
- 25 verified BibTeX entries with DOIs, clean author lists, and standard IEEE formatting.

---

## 3. Two-Column Formatting & Compilation Audit

| Revision ID | Deliverable | Previous State | Two-Column Corrected State | Verification Method |
| :---: | :--- | :--- | :--- | :--- |
| **REV-13** | `research_paper.tex` & Layout | Single-column tables/floats causing horizontal overflow warnings; missing `algorithm.sty`. | Added `algorithm.sty`, configured `table*` for wide tables (Dataset composition, Version evolution), sized single-column tables to fit 3.4 in column width (`\scriptsize`, tight padding), and verified all float placements. | Recompiled via Tectonic with zero fatal errors. |
| **REV-14** | `research_paper_final.pdf` | Uncompiled LaTeX source due to missing local TeX distribution. | Deployed standalone modern XeTeX engine `tectonic` (v0.17.0), successfully compiled `research_paper.tex` to `research_paper_final.pdf` (12 pages, 1.19 MB). High-resolution page rasterization (`fitz`) visually confirmed two-column balance, typography, and crisp 300 DPI figures. | Visual inspection of rendered PNG pages 1 through 12. |
| **REV-15** | `research_paper_draft.docx` | Single-column linear Word document. | Implemented genuine OpenXML multi-column section architecture (`build_docx.py`): Section 1 full-width header (Title, Authors) followed by continuous section break into 2 columns (`w:space="360"`). Continuous section breaks seamlessly alternate between two-column text flow and full-width float sections for wide tables/figures. | Verified 14 OpenXML sections and table alignments via `python-docx`. |

---

## 4. Originality Audit & Similarity Reduction (October 10, 2026)

| Revision ID | Scope / Target | Identified Problem / Unoriginal Phrasing | Corrective Originality Action | Verification Status |
| :---: | :--- | :--- | :--- | :--- |
| **REV-16** | Abstract & Section I | Canned survey clichés ("The rapid proliferation of deep generative acoustic models...", "democratized the generation of high-fidelity artificial voices", "theoretical curiosity into an acute security threat"). | Rebuilt from first principles of acoustic phonation: contrasted continuous vocal-tract aerodynamic resonance with discrete neural vocoder reconstruction mechanisms (transposed convolution upsampling, periodic source filters, codebooks); formalized Heisenberg-Gabor uncertainty constraint ($\Delta t \cdot \Delta f \ge \frac{1}{4\pi}$) and psychoacoustic masking compression. | Preserved all 15 empirical metrics and parameters with 100% precision. |
| **REV-17** | Section II & Table I | Generic literature survey paragraphs and repetitive table rows. | Re-analyzed prior work across 5 thematic axes; updated Table I with differentiated analytical columns focusing on architectural gaps and contrasting lightweight MFM-DNN against 300M+ parameter foundation models. | All 25 citations verified against authoritative databases; 0 missing, 0 uncited. |
| **REV-18** | LaTeX Typography & Equations | Minor overfull `\hbox` warnings on Eq. 3 kernel, Eq. 5 discrete FrFT, and Table III width in two-column format. | Reformatted Eq. 3 with compact exponential notation, split discrete FrFT across two lines using `\begin{split}`, wrapped Table III in `\resizebox{\columnwidth}{!}{...}`, and added `\usepackage{ragged2e}` with `>{\RaggedRight}` columns in tables. | Recompiled via Tectonic: **0 errors, 0 overfull boxes**. |
| **REV-19** | `research_paper_draft.docx` | Word manuscript contained older draft prose and mismatched table numbers. | Synchronized `build_docx.py` to match `research_paper.tex` word-for-word, standardized table numbering (Tables I–VI), and regenerated `research_paper_draft.docx` (~1.19 MB). | Verified OpenXML 2-column sections and table parity. |
| **REV-20** | Originality Audit Deliverable | Required formal audit report detailing rewrites, bibliography matrix, and evidence preservation. | Authored `originality_audit.md` covering section-by-section rewrite rationales, complete 25-entry bibliographic verification ledger, empirical preservation checklist, and explicit institutional checker status disclosure. | **SIMILARITY METRIC: NOT VERIFIED** (disclosed honestly due to lack of local institutional checker API). |
| **REV-21** | Author Block Layout | Unbroken `\textit{School of Computer Science and Engineering}` in `\IEEEauthorblockA` exceeded total line textwidth (700 pt > 516 pt), forcing Dr. Pattabiraman V into an asymmetrical second row on the left with an empty right column. | Balanced affiliation lines with explicit breaks (`\textit{School of Computer Science} \\ \textit{and Engineering}`), reducing column width and enabling all three authors to align side-by-side on a single horizontal row across 3 balanced columns (`y = 150.26` to `220.21`). | Visually verified in `author_fixed.png` and `rendered_pdf_pages/page_01.png`. |
| **REV-22** | Table I Literature Summary | Final row previously listed "Ours" and abbreviated last names ("Bhutani, Singh, & Pattabiraman") in comparison matrix. | Replaced "Ours" with "Proposed" in the Reference column. | Synchronized across deliverables. |
| **REV-23** | Table I Author Full Names | User requested explicitly displaying full author names rather than abbreviations or generic placeholders. | Formatted Author column as "Advit Bhutani, Ansh Singh, & Pattabiraman V" with non-breaking space and adjusted column widths (`p{3.4cm}`) to format across two balanced lines with 0 overfull boxes. | Synchronized across `research_paper.tex`, `research_paper_final.pdf`, and `research_paper_draft.docx`. |





