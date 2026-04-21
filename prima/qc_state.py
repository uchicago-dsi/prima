"""Canonical QC state helpers.

QC review state is stored as a single JSON object mapping exam_id -> record:

    {
      "<exam_id>": {
        "status": "good" | "auto_excluded" | null,
        "annotations": ["vertical line (detector artifact)", ...],
        "annotation_meta": {
          "vertical line (detector artifact)": {
            "source": "human" | "auto",
            "origin_run_id": "2026-04-03_qwen_v1",
            "model": "Qwen3.5-397B-A17B-FP8",
            "score": 0.91
          }
        }
      }
    }

The helpers in this module normalize that structure for both producers
(`qc_gallery.py`) and consumers (`analyze_mirai.py`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

DEFAULT_ANNOTATION_TAGS = [
    "vertical line (detector artifact)",
    "horizontal line (detector artifact)",
]

LEGACY_ANNOTATION_TAG_ALIASES = {
    "detector artifact - vertical line": "vertical line (detector artifact)",
    "detector artifact - horizontal line": "horizontal line (detector artifact)",
    "horizontal line (detector artifact": "horizontal line (detector artifact)",
}

VALID_ANNOTATION_SOURCES = {"human", "auto"}


def canonical_annotation_tag(raw_tag: Any) -> str:
    """Normalize a single annotation tag to its canonical label."""
    tag = str(raw_tag).strip()
    if not tag:
        return ""
    return LEGACY_ANNOTATION_TAG_ALIASES.get(tag, tag)


def normalize_annotation_tags(raw_tags: Any) -> list[str]:
    """Normalize and deduplicate annotation tags while preserving order."""
    if not isinstance(raw_tags, list):
        return []

    normalized: list[str] = []
    for raw_tag in raw_tags:
        tag = canonical_annotation_tag(raw_tag)
        if tag and tag not in normalized:
            normalized.append(tag)
    return normalized


def normalize_annotation_tag_catalog(raw_tags: Any) -> list[str]:
    """Normalize the available tag catalog while keeping defaults first."""
    normalized = normalize_annotation_tags(raw_tags)
    extras = [tag for tag in normalized if tag not in DEFAULT_ANNOTATION_TAGS]
    return DEFAULT_ANNOTATION_TAGS + extras


def normalize_qc_status(raw_status: Any) -> str | None:
    """Normalize a QC status string or return None for missing/blank values."""
    if raw_status is None:
        return None
    status = str(raw_status).strip().lower()
    return status or None


def normalize_annotation_source(raw_source: Any) -> str:
    """Normalize annotation provenance source, defaulting to 'human'."""
    source = str(raw_source).strip().lower() if raw_source is not None else "human"
    if source not in VALID_ANNOTATION_SOURCES:
        return "human"
    return source


def _normalize_optional_text(raw_value: Any) -> str | None:
    """Normalize optional string metadata fields."""
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    return value or None


def _normalize_optional_score(raw_value: Any) -> float | None:
    """Normalize an optional finite numeric score."""
    if raw_value is None or raw_value == "":
        return None
    try:
        score = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return score


def normalize_annotation_meta_entry(raw_entry: Any) -> dict[str, Any]:
    """Normalize one accepted annotation's provenance metadata."""
    entry = raw_entry if isinstance(raw_entry, dict) else {}
    normalized = {"source": normalize_annotation_source(entry.get("source"))}

    origin_run_id = _normalize_optional_text(entry.get("origin_run_id"))
    if origin_run_id is not None:
        normalized["origin_run_id"] = origin_run_id

    model = _normalize_optional_text(entry.get("model"))
    if model is not None:
        normalized["model"] = model

    score = _normalize_optional_score(entry.get("score"))
    if score is not None:
        normalized["score"] = score

    if bool(entry.get("legacy_suspect_default_injection")):
        normalized["legacy_suspect_default_injection"] = True

    return normalized


def normalize_annotation_meta(
    raw_meta: Any,
    annotations: list[str],
) -> dict[str, dict[str, Any]]:
    """Normalize per-tag provenance metadata for accepted annotations."""
    meta_map = raw_meta if isinstance(raw_meta, dict) else {}
    normalized: dict[str, dict[str, Any]] = {}
    for tag in annotations:
        normalized[tag] = normalize_annotation_meta_entry(meta_map.get(tag))
    return normalized


def normalize_qc_state_record(raw_record: Any) -> dict[str, Any]:
    """Normalize one exam's QC record to the canonical schema."""
    if not isinstance(raw_record, dict):
        raise ValueError("each QC exam entry must be a JSON object")

    status = normalize_qc_status(raw_record.get("status"))
    annotations = normalize_annotation_tags(raw_record.get("annotations", []))
    annotation_meta = normalize_annotation_meta(
        raw_record.get("annotation_meta"), annotations
    )

    # "good" means reviewed with no findings, so findings win if both are present.
    if status == "good" and annotations:
        status = None

    return {
        "status": status,
        "annotations": annotations,
        "annotation_meta": annotation_meta,
    }


def normalize_qc_state(payload: Any) -> dict[str, dict[str, Any]]:
    """Normalize a raw QC state payload and drop empty exam records."""
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("QC state file must contain a JSON object")

    normalized: dict[str, dict[str, Any]] = {}
    for raw_exam_id, raw_record in payload.items():
        exam_id = str(raw_exam_id)
        record = normalize_qc_state_record(raw_record)
        if record["status"] is None and not record["annotations"]:
            continue
        normalized[exam_id] = record
    return normalized


def load_qc_state(
    path: Path | None,
    *,
    persist_normalized: bool = False,
) -> dict[str, dict[str, Any]]:
    """Load a canonical QC state file."""
    if path is None or not path.exists():
        return {}

    with open(path) as f:
        payload = json.load(f)

    normalized = normalize_qc_state(payload)
    if persist_normalized and normalized != payload:
        save_qc_state(path, normalized)
    return normalized


def save_qc_state(path: Path, qc_state: Any) -> dict[str, dict[str, Any]]:
    """Normalize and persist QC state, returning the normalized payload."""
    normalized = normalize_qc_state(qc_state)
    stable_payload = {exam_id: normalized[exam_id] for exam_id in sorted(normalized)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stable_payload, f, indent=2)
    return stable_payload


def qc_state_to_status_map(qc_state: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Extract exam_id -> status for exams with a stored status."""
    return {
        str(exam_id): str(record["status"])
        for exam_id, record in qc_state.items()
        if normalize_qc_status(record.get("status")) is not None
    }


def qc_state_to_annotations_map(
    qc_state: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    """Extract exam_id -> annotation list for exams with findings."""
    return {
        str(exam_id): list(record["annotations"])
        for exam_id, record in qc_state.items()
        if normalize_annotation_tags(record.get("annotations"))
    }


def qc_state_to_annotation_meta_map(
    qc_state: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Extract exam_id -> tag -> annotation metadata for accepted findings."""
    meta_by_exam: dict[str, dict[str, dict[str, Any]]] = {}
    for exam_id, record in qc_state.items():
        annotations = normalize_annotation_tags(record.get("annotations"))
        if not annotations:
            continue
        meta_by_exam[str(exam_id)] = normalize_annotation_meta(
            record.get("annotation_meta"), annotations
        )
    return meta_by_exam


def merge_qc_state(
    *,
    status_map: dict[str, Any] | None = None,
    annotations_map: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build canonical QC state from split legacy maps."""
    status_map = status_map or {}
    annotations_map = annotations_map or {}

    all_exam_ids = {
        str(exam_id) for exam_id in list(status_map.keys()) + list(annotations_map.keys())
    }
    merged: dict[str, dict[str, Any]] = {}
    for exam_id in sorted(all_exam_ids):
        status = normalize_qc_status(status_map.get(exam_id))
        annotations = normalize_annotation_tags(annotations_map.get(exam_id, []))
        if status == "good" and annotations:
            status = None
        if status is None and not annotations:
            continue
        merged[exam_id] = {
            "status": status,
            "annotations": annotations,
            "annotation_meta": normalize_annotation_meta({}, annotations),
        }
    return merged
