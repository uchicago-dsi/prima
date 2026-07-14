"""Validated, ordered visual exemplars for single-target view auto-QC."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from prima.view_qc import (
    VIEW_LABEL_ABSENT,
    VIEW_LABEL_PRESENT,
    normalize_view_id,
    normalize_view_qc_target,
    validate_rendered_view_png,
)

MAX_VIEW_FEW_SHOT_EXAMPLES = 15
VIEW_FEW_SHOT_REQUIRED_COLUMNS = {
    "view_id",
    "image_path",
    "target",
    "label",
    "exemplar_order",
    "role",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one required file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_view_few_shot_manifest(
    path: Path,
    *,
    target: str,
    excluded_view_ids: Iterable[object] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a fixed, balanced exemplar bank and its frozen run provenance."""
    path = Path(path).resolve()
    target = normalize_view_qc_target(target)
    if not path.is_file():
        raise FileNotFoundError(f"view few-shot manifest not found: {path}")
    frame = pd.read_parquet(path)
    missing = sorted(VIEW_FEW_SHOT_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            "view few-shot manifest is missing columns: " + ", ".join(missing)
        )
    if not 2 <= len(frame) <= MAX_VIEW_FEW_SHOT_EXAMPLES:
        raise ValueError(
            "view few-shot manifest must contain between 2 and "
            f"{MAX_VIEW_FEW_SHOT_EXAMPLES} examples"
        )

    frame = frame.copy()
    frame["view_id"] = frame["view_id"].map(normalize_view_id)
    if frame["view_id"].duplicated().any():
        raise ValueError("view few-shot manifest contains duplicate view IDs")
    normalized_targets = frame["target"].map(normalize_view_qc_target)
    if set(normalized_targets) != {target}:
        raise ValueError("view few-shot manifest target does not match the run target")
    frame["target"] = normalized_targets

    labels = frame["label"].astype(str).str.strip().str.lower()
    allowed_labels = {VIEW_LABEL_PRESENT, VIEW_LABEL_ABSENT}
    if not set(labels).issubset(allowed_labels):
        raise ValueError("view few-shot labels must be present or absent")
    if set(labels) != allowed_labels:
        raise ValueError(
            "view few-shot manifest requires positive and negative examples"
        )
    frame["label"] = labels

    orders = pd.to_numeric(frame["exemplar_order"], errors="raise")
    if any(float(value) != int(value) for value in orders):
        raise ValueError("view few-shot exemplar_order values must be integers")
    frame["exemplar_order"] = orders.astype(int)
    expected_orders = list(range(1, len(frame) + 1))
    if sorted(frame["exemplar_order"].tolist()) != expected_orders:
        raise ValueError("view few-shot exemplar_order must be contiguous from 1")
    if frame["role"].isna().any() or not frame["role"].astype(str).str.strip().all():
        raise ValueError("view few-shot examples require nonempty roles")
    frame["role"] = frame["role"].astype(str).str.strip()
    if frame["role"].str.contains(r"[\r\n]", regex=True).any():
        raise ValueError("view few-shot roles must be single-line evidence phrases")

    excluded = {normalize_view_id(value) for value in excluded_view_ids}
    if set(frame["view_id"]) & excluded:
        raise ValueError("view few-shot examples overlap the scored manifest")

    root = path.parent
    exemplars: list[dict[str, Any]] = []
    provenance_examples: list[dict[str, Any]] = []
    for row in frame.sort_values("exemplar_order", kind="stable").to_dict("records"):
        view_id = row["view_id"]
        relative = Path(str(row["image_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("view few-shot image_path must be a safe relative path")
        expected = Path("images") / f"{view_id}.png"
        if relative != expected:
            raise ValueError("view few-shot image_path is not canonical")
        image_path = (root / relative).resolve()
        try:
            image_path.relative_to(root)
        except ValueError as error:
            raise ValueError("view few-shot image escapes its manifest root") from error
        validate_rendered_view_png(image_path, max_pixels=2_097_152)
        annotations = [target] if row["label"] == VIEW_LABEL_PRESENT else []
        exemplars.append(
            {
                "exam_id": view_id,
                "view_id": view_id,
                "image_path": str(image_path),
                "annotations": annotations,
                "few_shot_order": int(row["exemplar_order"]),
                "role": row["role"],
            }
        )
        provenance_examples.append(
            {
                "view_id": view_id,
                "label": row["label"],
                "exemplar_order": int(row["exemplar_order"]),
                "role": row["role"],
                "image_sha256": sha256_file(image_path),
            }
        )
    metadata = {
        "few_shot_manifest_sha256": sha256_file(path),
        "few_shot_example_count": len(exemplars),
        "few_shot_examples": provenance_examples,
    }
    return exemplars, metadata
