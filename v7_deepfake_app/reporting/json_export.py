"""
JSON export module for the V7 Audio Authenticity Analyzer.

Serializes the canonical ReportPayload into a strictly compliant, versioned,
UTF-8 JSON document without embedding raw waveforms, internal server paths,
or non-standard numeric tokens (NaN, Inf).
"""

import json
import math
from typing import Any
from reporting.report_schema import ReportPayload


def _sanitize_json_values(obj: Any) -> Any:
    """
    Recursively inspects data structures to ensure finite floating-point values
    and valid standard JSON types.
    """
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {str(k): _sanitize_json_values(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_json_values(item) for item in obj]
    return obj


def export_report_to_json(payload: ReportPayload, indent: int = 2) -> str:
    """
    Serializes a validated ReportPayload into a formatted JSON string.

    Args:
        payload: Validated ReportPayload instance
        indent: Indentation spaces for human readability

    Returns:
        str: UTF-8 compliant JSON document string

    Raises:
        ValueError: If payload is invalid or cannot be serialized
    """
    if not isinstance(payload, ReportPayload):
        raise ValueError(f"Expected ReportPayload instance, got {type(payload).__name__}")

    raw_dict = payload.to_dict()
    clean_dict = _sanitize_json_values(raw_dict)

    try:
        return json.dumps(clean_dict, indent=indent, ensure_ascii=False)
    except Exception as e:
        raise ValueError(f"Failed to serialize report payload to JSON: {str(e)}") from e
