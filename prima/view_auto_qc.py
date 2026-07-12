"""Normalized model suggestion runs for individual mammography views."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from prima.auto_qc import normalize_auto_suggestion_entry, utc_now_iso
from prima.view_qc import normalize_view_id, normalize_view_qc_target

VIEW_AUTO_QC_SCHEMA_VERSION = 2
VIEW_AUTO_QC_PROMPT_VERSION = "single_target_view_v1"
VIEW_CONFIDENCE_LEVELS = ("low", "medium", "high")


def normalize_view_suggestion_record(raw_record: Any, *, target: str) -> dict[str, Any]:
    """Normalize one model record for a frozen view-level target."""
    if not isinstance(raw_record, dict):
        raise ValueError("each view suggestion record must be a JSON object")
    raw_suggestions = raw_record.get("suggestions", [])
    if not isinstance(raw_suggestions, list):
        raise ValueError("view suggestions must be a JSON list")
    suggestions = []
    for raw_suggestion in raw_suggestions:
        normalized = normalize_auto_suggestion_entry(
            raw_suggestion, allowed_tags={target}
        )
        if normalized is not None:
            suggestions.append(normalized)
    if len(suggestions) > 1:
        raise ValueError("single-target view QC may contain at most one suggestion")

    image_path = str(raw_record.get("image_path", "")).strip()
    if not image_path:
        raise ValueError("view suggestion record is missing image_path")
    record = {
        "image_path": image_path,
        "suggestions": suggestions,
    }
    debug_dump_file = str(raw_record.get("debug_dump_file", "")).strip()
    if debug_dump_file:
        record["debug_dump_file"] = debug_dump_file
    return record


def view_suggestion_is_target_present(
    record: Mapping[str, Any], *, target: str, minimum_confidence: str
) -> bool:
    """Return whether the model marks the target present at the threshold."""
    minimum_confidence = str(minimum_confidence).strip().lower()
    if minimum_confidence not in VIEW_CONFIDENCE_LEVELS:
        raise ValueError(
            f"unsupported minimum suggestion confidence: {minimum_confidence!r}"
        )
    normalized = normalize_view_suggestion_record(
        dict(record), target=normalize_view_qc_target(target)
    )
    if not normalized["suggestions"]:
        return False
    confidence = normalized["suggestions"][0].get("confidence")
    if confidence not in VIEW_CONFIDENCE_LEVELS:
        raise ValueError("model suggestion is missing a valid confidence")
    ranks = {level: index for index, level in enumerate(VIEW_CONFIDENCE_LEVELS)}
    return ranks[confidence] >= ranks[minimum_confidence]


def normalize_view_auto_run(payload: Any) -> dict[str, Any]:
    """Validate the current view-level auto-QC run schema."""
    if not isinstance(payload, dict):
        raise ValueError("view auto-QC run must be a JSON object")
    if payload.get("schema_version") != VIEW_AUTO_QC_SCHEMA_VERSION:
        raise ValueError("unsupported view auto-QC schema")
    if payload.get("input_level") != "view":
        raise ValueError("view auto-QC run must have input_level='view'")
    target = normalize_view_qc_target(payload.get("target"))
    if payload.get("prompt_version") != VIEW_AUTO_QC_PROMPT_VERSION:
        raise ValueError("view auto-QC prompt version does not match current code")
    suggestions_raw = payload.get("view_suggestions")
    if not isinstance(suggestions_raw, dict):
        raise ValueError("view_suggestions must be a JSON object")
    view_suggestions = {
        normalize_view_id(view_id): normalize_view_suggestion_record(
            record, target=target
        )
        for view_id, record in suggestions_raw.items()
    }
    inference_settings = payload.get("inference_settings")
    if not isinstance(inference_settings, dict):
        raise ValueError("inference_settings must be a JSON object")
    inference_settings = json.loads(json.dumps(inference_settings, sort_keys=True))
    return {
        "schema_version": VIEW_AUTO_QC_SCHEMA_VERSION,
        "input_level": "view",
        "target": target,
        "run_id": str(payload.get("run_id", "")).strip(),
        "model": str(payload.get("model", "")).strip(),
        "backend": str(payload.get("backend", "")).strip(),
        "created_at": str(payload.get("created_at", "")).strip(),
        "prompt_version": VIEW_AUTO_QC_PROMPT_VERSION,
        "prompt_mode": str(payload.get("prompt_mode", "")).strip(),
        "prompt_variant": str(payload.get("prompt_variant", "")).strip(),
        "inference_settings": inference_settings,
        "view_suggestions": view_suggestions,
    }


def new_view_auto_run(
    *,
    target: str,
    model: str,
    prompt_variant: str,
    inference_settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Create an empty frozen run payload."""
    created_at = utc_now_iso()
    return normalize_view_auto_run(
        {
            "schema_version": VIEW_AUTO_QC_SCHEMA_VERSION,
            "input_level": "view",
            "target": normalize_view_qc_target(target),
            "run_id": f"{created_at.replace(':', '').replace('+00:00', 'Z')}_view_qc",
            "model": model,
            "backend": "vllm_local",
            "created_at": created_at,
            "prompt_version": VIEW_AUTO_QC_PROMPT_VERSION,
            "prompt_mode": "marker_classifier",
            "prompt_variant": prompt_variant,
            "inference_settings": dict(inference_settings),
            "view_suggestions": {},
        }
    )


def load_view_auto_run(path: Path) -> dict[str, Any]:
    """Load a normalized view auto-QC run."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open() as handle:
        return normalize_view_auto_run(json.load(handle))


def save_view_auto_run(path: Path, payload: Any) -> dict[str, Any]:
    """Atomically save a normalized view auto-QC run."""
    path = Path(path)
    normalized = normalize_view_auto_run(payload)
    normalized["view_suggestions"] = {
        view_id: normalized["view_suggestions"][view_id]
        for view_id in sorted(normalized["view_suggestions"])
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(normalized, handle, indent=2)
            handle.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()
    return normalized


def require_compatible_view_auto_run(
    existing: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    """Reject a resume target produced by another model or prompt."""
    fields = (
        "schema_version",
        "input_level",
        "target",
        "model",
        "backend",
        "prompt_version",
        "prompt_mode",
        "prompt_variant",
        "inference_settings",
    )
    mismatches = [
        field for field in fields if existing.get(field) != current.get(field)
    ]
    if mismatches:
        raise ValueError(
            "refusing to resume incompatible view auto-QC run: " + ", ".join(mismatches)
        )
