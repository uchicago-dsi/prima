"""Canonical state and manifest helpers for binary view-level QC."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

VIEW_QC_SCHEMA_VERSION = 1
VIEW_QC_TARGET = "vertical line (detector artifact)"
VIEW_LABEL_PASS = "pass"
VIEW_LABEL_VERTICAL_LINE = "vertical_line"
VALID_VIEW_LABELS = {VIEW_LABEL_PASS, VIEW_LABEL_VERTICAL_LINE}

VIEW_MANIFEST_REQUIRED_COLUMNS = {
    "view_id",
    "image_path",
    "laterality",
    "view",
    "review_order",
}


def normalize_view_id(value: object) -> str:
    """Return a SHA-256 view key or fail loudly."""
    view_id = str(value).strip().lower()
    if len(view_id) != 64 or any(char not in "0123456789abcdef" for char in view_id):
        raise ValueError("view_id must be a 64-character SHA-256 digest")
    return view_id


def validate_view_manifest_columns(columns: Iterable[str], context: str) -> None:
    """Require the current view-review manifest schema."""
    available = set(columns)
    missing = sorted(VIEW_MANIFEST_REQUIRED_COLUMNS - available)
    if missing:
        raise ValueError(
            f"{context} is missing required view-QC columns: {', '.join(missing)}"
        )


def empty_view_qc_state() -> dict[str, Any]:
    """Build a new empty state for the frozen binary target."""
    return {
        "schema_version": VIEW_QC_SCHEMA_VERSION,
        "target": VIEW_QC_TARGET,
        "labels": {},
    }


def normalize_view_qc_state(payload: Any) -> dict[str, Any]:
    """Validate and normalize the current view-level state schema."""
    if not isinstance(payload, dict):
        raise ValueError("view QC state must be a JSON object")
    if payload.get("schema_version") != VIEW_QC_SCHEMA_VERSION:
        raise ValueError(
            "view QC state schema is unsupported; start a fresh view-level review"
        )
    if payload.get("target") != VIEW_QC_TARGET:
        raise ValueError("view QC state target does not match the frozen target")
    raw_labels = payload.get("labels")
    if not isinstance(raw_labels, dict):
        raise ValueError("view QC state labels must be a JSON object")

    labels: dict[str, dict[str, str]] = {}
    for raw_view_id, raw_record in raw_labels.items():
        view_id = normalize_view_id(raw_view_id)
        if not isinstance(raw_record, dict):
            raise ValueError("each view QC label must be a JSON object")
        label = str(raw_record.get("label", "")).strip()
        if label not in VALID_VIEW_LABELS:
            raise ValueError(f"invalid view QC label for {view_id[:12]}")
        source = str(raw_record.get("source", "")).strip()
        if source != "human":
            raise ValueError("view QC reference labels must have source='human'")
        updated_at = str(raw_record.get("updated_at", "")).strip()
        if not updated_at:
            raise ValueError("view QC label is missing updated_at")
        labels[view_id] = {
            "label": label,
            "source": "human",
            "updated_at": updated_at,
        }

    return {
        "schema_version": VIEW_QC_SCHEMA_VERSION,
        "target": VIEW_QC_TARGET,
        "labels": labels,
    }


def load_view_qc_state(path: Path) -> dict[str, Any]:
    """Load a view-level QC state, returning an empty state when absent."""
    path = Path(path)
    if not path.exists():
        return empty_view_qc_state()
    with path.open() as handle:
        payload = json.load(handle)
    return normalize_view_qc_state(payload)


def save_view_qc_state(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically write a normalized view-level state with restricted permissions."""
    path = Path(path)
    normalized = normalize_view_qc_state(dict(state))
    normalized["labels"] = {
        view_id: normalized["labels"][view_id]
        for view_id in sorted(normalized["labels"])
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(normalized, handle, indent=2)
            handle.write("\n")
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
        os.chmod(path, 0o600)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return normalized


def set_view_label(
    state: Mapping[str, Any], view_id: object, label: str | None
) -> dict[str, Any]:
    """Set or clear one human view label and return normalized state."""
    normalized = normalize_view_qc_state(dict(state))
    key = normalize_view_id(view_id)
    labels = dict(normalized["labels"])
    if label is None:
        labels.pop(key, None)
    else:
        label = str(label).strip()
        if label not in VALID_VIEW_LABELS:
            raise ValueError(f"unsupported view QC label: {label!r}")
        labels[key] = {
            "label": label,
            "source": "human",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    normalized["labels"] = labels
    return normalize_view_qc_state(normalized)


def summarize_view_qc_state(
    state: Mapping[str, Any], manifest_view_ids: Iterable[object]
) -> dict[str, int]:
    """Return one-denominator progress counts for a review manifest."""
    normalized = normalize_view_qc_state(dict(state))
    manifest_ids = [normalize_view_id(value) for value in manifest_view_ids]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("view QC manifest contains duplicate view_id values")
    manifest_set = set(manifest_ids)
    foreign = set(normalized["labels"]) - manifest_set
    if foreign:
        raise ValueError("view QC state contains labels outside this manifest")
    labels = normalized["labels"]
    passed = sum(record["label"] == VIEW_LABEL_PASS for record in labels.values())
    vertical = sum(
        record["label"] == VIEW_LABEL_VERTICAL_LINE for record in labels.values()
    )
    reviewed = passed + vertical
    total = len(manifest_ids)
    return {
        "total": total,
        "reviewed": reviewed,
        "remaining": total - reviewed,
        "pass": passed,
        "vertical_line": vertical,
    }


def render_dicom_view_png(dataset: Any, output_path: Path, max_pixels: int) -> None:
    """Render one DICOM view to an 8-bit PNG with deterministic min-max display."""
    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive")
    pixels = dataset.pixel_array.astype(np.float32, copy=False)
    if pixels.ndim != 2:
        raise ValueError("view QC requires a single-frame two-dimensional DICOM")
    slope = float(dataset.get("RescaleSlope", 1.0))
    intercept = float(dataset.get("RescaleIntercept", 0.0))
    pixels = pixels * slope + intercept
    vmin = float(np.min(pixels))
    vmax = float(np.max(pixels))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        raise ValueError("DICOM has a degenerate pixel range")
    pixels = (pixels - vmin) * (255.0 / (vmax - vmin))
    if (
        str(dataset.get("PhotometricInterpretation", "")).strip().upper()
        == "MONOCHROME1"
    ):
        pixels = 255.0 - pixels
    image = Image.fromarray(np.rint(pixels).astype(np.uint8), mode="L")
    if image.width * image.height > max_pixels:
        scale = sqrt(max_pixels / float(image.width * image.height))
        size = (
            max(1, int(image.width * scale)),
            max(1, int(image.height * scale)),
        )
        image = image.resize(size, Image.LANCZOS)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
