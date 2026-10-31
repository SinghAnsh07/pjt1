"""
PDF export module for the V7 Audio Authenticity Analyzer.

Generates a professional, multi-page, publication-quality forensic PDF report
from the canonical ReportPayload using ReportLab. Incorporates verified media
metadata, analysis configuration, score breakdowns, paginated timeline tables,
embedded signal visualizations, and explicit scientific disclaimers.
"""

import io
import math
from typing import Optional, List, Dict, Any
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    PageBreak,
    HRFlowable,
)
from reportlab.pdfgen import canvas

from reporting.report_schema import ReportPayload


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass ReportLab canvas that dynamically computes and renders
    total page counts ('Page X of Y') and professional running footers.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Dict[str, Any]] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count: int) -> None:
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.6)

        # Running Footer
        self.line(40, 36, 612 - 40, 36)
        self.drawString(40, 24, "V7 Audio Authenticity Analyzer • Forensic Screening Report")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 40, 24, page_str)
        self.restoreState()


def _fig_to_reportlab_image(
    fig: Optional[plt.Figure],
    width_pt: float = 510,
    height_pt: float = 180,
) -> Optional[Image]:
    """Converts an in-memory Matplotlib figure to a ReportLab Flowable Image."""
    if fig is None:
        return None
    try:
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=180,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        buf.seek(0)
        return Image(buf, width=width_pt, height=height_pt)
    except Exception:
        return None


def export_report_to_pdf(
    payload: ReportPayload,
    waveform_fig: Optional[plt.Figure] = None,
    spectrogram_fig: Optional[plt.Figure] = None,
    timeline_fig: Optional[plt.Figure] = None,
    feature_views_fig: Optional[plt.Figure] = None,
) -> bytes:
    """
    Renders the canonical ReportPayload into a multi-page PDF document.

    Args:
        payload: Validated ReportPayload instance
        waveform_fig: Optional Matplotlib figure of audio waveform
        spectrogram_fig: Optional Matplotlib figure of audio spectrogram
        timeline_fig: Optional Matplotlib figure of multi-window timeline
        feature_views_fig: Optional Matplotlib figure of neural feature views

    Returns:
        bytes: Complete binary PDF document
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=50,
    )

    styles = getSampleStyleSheet()

    # Custom Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748B"),
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#1E293B"),
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#1E293B"),
    )
    caption_style = ParagraphStyle(
        "FigCaption",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#64748B"),
        spaceBefore=4,
        spaceAfter=8,
    )

    story = []

    # -------------------------------------------------------------------------
    # 1. Header Block
    # -------------------------------------------------------------------------
    story.append(Paragraph("V7 Audio Authenticity Analysis Report", title_style))
    story.append(Paragraph("Automated Biometric Screening & Acoustic Forensic Evaluation", subtitle_style))
    story.append(Spacer(1, 8))

    meta = payload.metadata
    media = payload.media
    res = payload.result
    cfg = payload.config

    header_table_data = [
        [
            Paragraph("<b>Report Identifier:</b>", table_cell),
            Paragraph(escape(meta.report_id), table_cell_bold),
            Paragraph("<b>Generated (UTC):</b>", table_cell),
            Paragraph(escape(meta.generated_at_utc), table_cell),
        ],
        [
            Paragraph("<b>Model Architecture:</b>", table_cell),
            Paragraph(escape(meta.model_name), table_cell),
            Paragraph("<b>Checkpoint SHA-256:</b>", table_cell),
            Paragraph(f"<font size=6.5>{escape(meta.checkpoint_sha256[:16])}...</font>", table_cell),
        ],
    ]
    header_table = Table(header_table_data, colWidths=[110, 160, 100, 162])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # 2. Forensic Screening Verdict Card
    # -------------------------------------------------------------------------
    is_real = ("Genuine" in res.overall_screening_finding)
    card_bg = colors.HexColor("#ECFDF5") if is_real else colors.HexColor("#FEF2F2")
    card_border = colors.HexColor("#10B981") if is_real else colors.HexColor("#EF4444")
    card_text_color = "#047857" if is_real else "#B91C1C"

    finding_p = Paragraph(
        f"<b><font size=12 color='{card_text_color}'>{escape(res.overall_screening_finding)}</font></b>",
        styles["Normal"],
    )
    score_p = Paragraph(
        f"<b>Primary Model Cosine Score S(x):</b> <font color='{card_text_color}'><b>{res.primary_score:+.4f}</b></font> "
        f"&nbsp;&bull;&nbsp; <b>Candidate Threshold (tau*):</b> <b>{cfg.candidate_threshold:.4f}</b> (Unvalidated)",
        body_style,
    )
    verdict_card_data = [[finding_p], [score_p]]
    verdict_table = Table(verdict_card_data, colWidths=[532])
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), card_bg),
        ("BOX", (0, 0), (-1, -1), 1.2, card_border),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # 3. Input Media & Analysis Configuration Tables (Side-by-side)
    # -------------------------------------------------------------------------
    story.append(Paragraph("1. Ingestion Metadata & Operational Configuration", h2_style))

    media_data = [
        [Paragraph("<b>Filename</b>", table_cell_bold), Paragraph(escape(media.filename), table_cell)],
        [Paragraph("<b>Recorded Duration</b>", table_cell_bold), Paragraph(f"{media.duration_seconds:.2f} s", table_cell)],
        [Paragraph("<b>Original Sample Rate</b>", table_cell_bold), Paragraph(f"{media.sample_rate_hz:,} Hz", table_cell)],
        [Paragraph("<b>Channel Layout</b>", table_cell_bold), Paragraph("Mono (1 ch)" if media.channels == 1 else f"Stereo ({media.channels} ch)", table_cell)],
        [Paragraph("<b>Container / Codec</b>", table_cell_bold), Paragraph(f"{escape(media.format_name)} ({escape(media.subtype)})", table_cell)],
        [Paragraph("<b>Conditioning</b>", table_cell_bold), Paragraph(f"Resample: {media.is_resampled} | Downmix: {media.is_downmixed} | Pad: {media.is_padded}", table_cell)],
    ]
    media_table = Table(media_data, colWidths=[105, 155])
    media_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]))

    config_data = [
        [Paragraph("<b>Target Model Rate</b>", table_cell_bold), Paragraph(f"{cfg.target_sample_rate_hz:,} Hz", table_cell)],
        [Paragraph("<b>Window Length</b>", table_cell_bold), Paragraph(f"{cfg.window_duration_seconds:.1f} s (64,000 samples)", table_cell)],
        [Paragraph("<b>Hop Stride</b>", table_cell_bold), Paragraph(f"{cfg.hop_duration_seconds:.1f} s (75% overlap)" if cfg.hop_duration_seconds else "Single 4.0s Window", table_cell)],
        [Paragraph("<b>Scoring Metric</b>", table_cell_bold), Paragraph("One-Class Cosine S(x) in [-1, 1]", table_cell)],
        [Paragraph("<b>Candidate Threshold</b>", table_cell_bold), Paragraph(f"tau* = {cfg.candidate_threshold:.4f} (ASVspoof 2019)", table_cell)],
        [Paragraph("<b>Validation Status</b>", table_cell_bold), Paragraph("Unvalidated on in-the-wild audio", table_cell)],
    ]
    config_table = Table(config_data, colWidths=[105, 155])
    config_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]))

    side_by_side = Table([[media_table, config_table]], colWidths=[266, 266])
    side_by_side.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(side_by_side)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # 4. Multi-Window Timeline Breakdown (if multi-window recording)
    # -------------------------------------------------------------------------
    if res.analysis_mode == "multi_window_timeline" and res.windows:
        story.append(Paragraph("2. Sliding Window Timeline Analysis", h2_style))
        summary_line = (
            f"<b>Total Windows:</b> {res.total_windows} &nbsp;&bull;&nbsp; "
            f"<b>Likely Genuine:</b> {res.likely_genuine_windows} &nbsp;&bull;&nbsp; "
            f"<b>Potential Spoof:</b> {res.potential_fake_windows} ({res.flagged_window_percentage:.1f}%) &nbsp;&bull;&nbsp; "
            f"<b>Mean Score:</b> {res.mean_window_score:+.4f} &nbsp;&bull;&nbsp; "
            f"<b>Min Score:</b> {res.min_window_score:+.4f}"
        )
        story.append(Paragraph(summary_line, body_style))
        story.append(Spacer(1, 6))

        # Build table rows with repeated headers
        table_rows = [
            [
                Paragraph("<b>Window</b>", table_cell_bold),
                Paragraph("<b>Time Interval</b>", table_cell_bold),
                Paragraph("<b>Model Score S(x)</b>", table_cell_bold),
                Paragraph("<b>Screening Verdict</b>", table_cell_bold),
                Paragraph("<b>Threshold Margin</b>", table_cell_bold),
            ]
        ]
        for w in res.windows:
            w_is_real = (w.prediction == "REAL")
            v_color = "#047857" if w_is_real else "#B91C1C"
            v_label = "Likely Genuine" if w_is_real else "Potential Spoof"
            table_rows.append([
                Paragraph(f"#{w.window_index + 1}", table_cell),
                Paragraph(f"{w.start_seconds:.1f}s – {w.end_seconds:.1f}s", table_cell),
                Paragraph(f"{w.score:+.4f}", table_cell_bold),
                Paragraph(f"<font color='{v_color}'><b>{v_label}</b></font>", table_cell),
                Paragraph(f"{w.margin:+.4f}", table_cell),
            ])

        timeline_table = Table(table_rows, colWidths=[65, 115, 110, 130, 112], repeatRows=1)
        t_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ]
        # Alternating row background
        for i in range(1, len(table_rows)):
            if i % 2 == 0:
                t_style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F8FAFC")))
        timeline_table.setStyle(TableStyle(t_style))
        story.append(timeline_table)
        story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # 5. Acoustic Signal Visualizations
    # -------------------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("3. Acoustic Signal & Neural Feature Visualizations", h2_style))

    # Waveform Plot
    img_wave = _fig_to_reportlab_image(waveform_fig, width_pt=532, height_pt=118)
    if img_wave:
        story.append(img_wave)
        story.append(Paragraph(
            "<b>Figure 1: Audio Waveform</b> &mdash; Displays amplitude variations over time. "
            "Peak-envelope decimation preserves acoustic transients without altering model input data.",
            caption_style,
        ))

    # Spectrogram Plot
    img_spec = _fig_to_reportlab_image(spectrogram_fig, width_pt=532, height_pt=118)
    if img_spec:
        story.append(img_spec)
        story.append(Paragraph(
            "<b>Figure 2: Audio Spectrogram</b> &mdash; Displays time-frequency power spectral density (dB scale). "
            "Harmonic formants occur in both natural and synthesized speech and do not independently establish authenticity.",
            caption_style,
        ))

    # Timeline Chart (if multi-window)
    if timeline_fig:
        img_time = _fig_to_reportlab_image(timeline_fig, width_pt=532, height_pt=118)
        if img_time:
            story.append(img_time)
            story.append(Paragraph(
                "<b>Figure 3: Multi-Window Timeline Evaluation</b> &mdash; Plots per-window cosine similarity S(x) against "
                "temporal window midpoints. Dashed amber line indicates candidate decision threshold (tau* = 0.4820).",
                caption_style,
            ))

    # Neural Feature Views (ST-FrFT + Wavelet)
    if feature_views_fig:
        img_feat = _fig_to_reportlab_image(feature_views_fig, width_pt=532, height_pt=118)
        if img_feat:
            story.append(img_feat)
            story.append(Paragraph(
                "<b>Figure 4: 2D Neural Feature Representations</b> &mdash; Displays the 3-channel spatiotemporal tensor "
                "(ST-FrFT alpha=0.50, ST-FrFT alpha=0.85, and Continuous Morlet Wavelet Scalogram) processed by the LCNN-2D backbone.",
                caption_style,
            ))

    story.append(Spacer(1, 8))

    # -------------------------------------------------------------------------
    # 6. Technical Limitations & Disclaimers
    # -------------------------------------------------------------------------
    limitations_story = [Paragraph("4. Scientific Limitations & Responsible Use", h2_style)]
    for limit in payload.limitations:
        bullet_p = Paragraph(f"&bull;&nbsp;&nbsp;{escape(limit)}", body_style)
        limitations_story.append(bullet_p)
        limitations_story.append(Spacer(1, 3))

    story.append(KeepTogether(limitations_story))

    doc.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()
