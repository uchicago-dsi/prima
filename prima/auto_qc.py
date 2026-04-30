"""Helpers for model-produced QC suggestion runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prima.qc_state import normalize_annotation_tag_catalog, normalize_annotation_tags

AUTO_QC_PROMPT_VERSION = "qc_multilabel_v1"


def utc_now_iso() -> str:
    """Return a stable UTC timestamp string for run metadata."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_auto_suggestion_entry(
    raw_entry: Any,
    *,
    allowed_tags: set[str] | None = None,
) -> dict[str, Any] | None:
    """Normalize a single tag suggestion record."""
    if isinstance(raw_entry, str):
        tag = raw_entry
        payload: dict[str, Any] = {}
    elif isinstance(raw_entry, dict):
        tag = raw_entry.get("tag")
        payload = raw_entry
    else:
        return None

    normalized_tags = normalize_annotation_tags([tag])
    if not normalized_tags:
        return None
    normalized_tag = normalized_tags[0]
    if allowed_tags is not None and normalized_tag not in allowed_tags:
        return None

    normalized = {"tag": normalized_tag}

    raw_score = payload.get("score")
    if raw_score is not None and raw_score != "":
        try:
            normalized["score"] = float(raw_score)
        except (TypeError, ValueError):
            pass

    rationale = str(payload.get("rationale", "")).strip() if payload else ""
    if rationale:
        normalized["rationale"] = rationale
    confidence = str(payload.get("confidence", "")).strip().lower() if payload else ""
    if confidence in {"high", "medium", "low"}:
        normalized["confidence"] = confidence
    review = payload.get("review") if payload else None
    if isinstance(review, bool):
        normalized["review"] = review

    return normalized


def normalize_exam_suggestion_record(
    raw_record: Any,
    *,
    tag_catalog: list[str],
) -> dict[str, Any]:
    """Normalize one exam's model suggestions."""
    if not isinstance(raw_record, dict):
        raise ValueError("each auto-QC exam entry must be a JSON object")

    allowed_tags = set(tag_catalog) if tag_catalog else None
    suggestions_raw = raw_record.get("suggestions", [])
    if not isinstance(suggestions_raw, list):
        suggestions_raw = []

    suggestions: list[dict[str, Any]] = []
    seen_tags: set[str] = set()
    for raw_entry in suggestions_raw:
        normalized = normalize_auto_suggestion_entry(
            raw_entry, allowed_tags=allowed_tags
        )
        if not normalized:
            continue
        tag = normalized["tag"]
        if tag in seen_tags:
            continue
        seen_tags.add(tag)
        suggestions.append(normalized)

    image_path = str(raw_record.get("image_path", "")).strip()
    model = str(raw_record.get("model", "")).strip()
    normalized_record = {
        "image_path": image_path,
        "model": model or None,
        "suggestions": suggestions,
    }
    few_shot_example_exam_ids = raw_record.get("few_shot_example_exam_ids", [])
    if isinstance(few_shot_example_exam_ids, list):
        normalized_ids = []
        for exam_id in few_shot_example_exam_ids:
            exam_id_str = str(exam_id).strip()
            if exam_id_str and exam_id_str not in normalized_ids:
                normalized_ids.append(exam_id_str)
        normalized_record["few_shot_example_exam_ids"] = normalized_ids
    prompt_mode = str(raw_record.get("prompt_mode", "")).strip()
    if prompt_mode:
        normalized_record["prompt_mode"] = prompt_mode
    prompt_variant = str(raw_record.get("prompt_variant", "")).strip()
    if prompt_variant:
        normalized_record["prompt_variant"] = prompt_variant
    debug_dump_file = str(raw_record.get("debug_dump_file", "")).strip()
    if debug_dump_file:
        normalized_record["debug_dump_file"] = debug_dump_file
    return normalized_record


def normalize_auto_run(payload: Any) -> dict[str, Any]:
    """Normalize a full auto-QC run payload."""
    if not isinstance(payload, dict):
        raise ValueError("auto-QC run file must contain a JSON object")

    tag_catalog = normalize_annotation_tag_catalog(payload.get("tag_catalog", []))
    exam_suggestions_raw = payload.get("exam_suggestions", {})
    if exam_suggestions_raw is None:
        exam_suggestions_raw = {}
    if not isinstance(exam_suggestions_raw, dict):
        raise ValueError("exam_suggestions must be a JSON object")

    exam_suggestions = {
        str(exam_id): normalize_exam_suggestion_record(
            raw_record, tag_catalog=tag_catalog
        )
        for exam_id, raw_record in exam_suggestions_raw.items()
    }
    model = str(payload.get("model", "")).strip()
    backend = str(payload.get("backend", "")).strip()
    run_id = str(payload.get("run_id", "")).strip()
    created_at = str(payload.get("created_at", "")).strip() or utc_now_iso()
    prompt_version = (
        str(payload.get("prompt_version", "")).strip() or AUTO_QC_PROMPT_VERSION
    )
    prompt_mode = str(payload.get("prompt_mode", "")).strip() or "tagger_json"
    prompt_variant = str(payload.get("prompt_variant", "")).strip() or "baseline"

    return {
        "run_id": run_id or created_at.replace(":", "").replace("+00:00", "Z"),
        "model": model,
        "backend": backend,
        "created_at": created_at,
        "prompt_version": prompt_version,
        "prompt_mode": prompt_mode,
        "prompt_variant": prompt_variant,
        "tag_catalog": tag_catalog,
        "exam_suggestions": exam_suggestions,
    }


def load_auto_run(
    path: Path | None, *, persist_normalized: bool = False
) -> dict[str, Any]:
    """Load a normalized auto-QC run file."""
    if path is None or not path.exists():
        return {}

    with open(path) as f:
        payload = json.load(f)

    normalized = normalize_auto_run(payload)
    if persist_normalized and normalized != payload:
        save_auto_run(path, normalized)
    return normalized


def save_auto_run(path: Path, payload: Any) -> dict[str, Any]:
    """Normalize and save an auto-QC run file."""
    normalized = normalize_auto_run(payload)
    stable_exam_suggestions = {
        exam_id: normalized["exam_suggestions"][exam_id]
        for exam_id in sorted(normalized["exam_suggestions"])
    }
    stable_payload = {
        **normalized,
        "exam_suggestions": stable_exam_suggestions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stable_payload, f, indent=2)
    return stable_payload


def auto_run_to_tag_map(auto_run: dict[str, Any]) -> dict[str, list[str]]:
    """Extract exam_id -> suggested tag list."""
    exam_suggestions = auto_run.get("exam_suggestions", {})
    if not isinstance(exam_suggestions, dict):
        return {}
    return {
        str(exam_id): [entry["tag"] for entry in record.get("suggestions", [])]
        for exam_id, record in exam_suggestions.items()
        if record.get("suggestions")
    }


def compute_exam_level_tag_metrics(
    *,
    exam_ids: set[str],
    gt_by_exam: dict[str, set[str]],
    pred_by_exam: dict[str, set[str]],
    tag_catalog: list[str],
) -> list[dict[str, Any]]:
    """Compute TP/FP/FN/TN summary per tag over a fixed exam universe."""
    metrics: list[dict[str, Any]] = []
    sorted_tags = sorted(
        {
            *tag_catalog,
            *(tag for tags in gt_by_exam.values() for tag in tags),
            *(tag for tags in pred_by_exam.values() for tag in tags),
        }
    )
    for tag in sorted_tags:
        tp = fp = fn = tn = 0
        for exam_id in exam_ids:
            gt = tag in gt_by_exam.get(exam_id, set())
            pred = tag in pred_by_exam.get(exam_id, set())
            if pred and gt:
                tp += 1
            elif pred and not gt:
                fp += 1
            elif not pred and gt:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        metrics.append(
            {
                "tag": tag,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
            }
        )
    return sorted(metrics, key=lambda row: (-row["tp"], -row["fp"], row["tag"]))
