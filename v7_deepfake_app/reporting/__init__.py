"""
Reporting package for the V7 Audio Authenticity Analyzer.

Provides validated report payload construction, standardized UTF-8 JSON export,
and professional multi-page PDF generation using ReportLab.
"""

from reporting.report_schema import (
    ReportPayload,
    ReportMetadata,
    ReportMediaInfo,
    ReportAnalysisConfig,
    ReportResultSummary,
    ReportWindowItem,
    build_report_payload,
)
from reporting.json_export import export_report_to_json
from reporting.pdf_export import export_report_to_pdf

__all__ = [
    "ReportPayload",
    "ReportMetadata",
    "ReportMediaInfo",
    "ReportAnalysisConfig",
    "ReportResultSummary",
    "ReportWindowItem",
    "build_report_payload",
    "export_report_to_json",
    "export_report_to_pdf",
]
