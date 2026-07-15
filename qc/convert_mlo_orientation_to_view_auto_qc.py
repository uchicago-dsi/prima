#!/usr/bin/env python3
"""Convert paired MLO-orientation logits into a complete view auto-QC run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from prima.view_auto_qc import new_view_auto_run, save_view_auto_run
from prima.view_few_shot import sha256_file
from prima.view_qc import (
    normalize_view_id,
    normalize_view_qc_target,
    validate_view_manifest_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--orientation-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-orientation-result-sha256", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    return parser.parse_args()


def _require_hash(path: Path, expected: str, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{description} SHA-256 mismatch: expected={expected} found={actual}"
        )


def run_from_args(args: argparse.Namespace) -> dict[str, object]:
    manifest_path = args.manifest.resolve()
    orientation_path = args.orientation_result.resolve()
    output_path = args.output.resolve()
    target = normalize_view_qc_target(args.target)
    _require_hash(
        manifest_path, args.expected_manifest_sha256, "whole-exam audit manifest"
    )
    _require_hash(
        orientation_path,
        args.expected_orientation_result_sha256,
        "MLO orientation result",
    )
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite orientation view run: {output_path}"
        )

    manifest = pd.read_parquet(manifest_path).copy()
    validate_view_manifest_columns(manifest.columns, str(manifest_path))
    manifest["view_id"] = manifest["view_id"].map(normalize_view_id)
    if manifest.empty or manifest["view_id"].duplicated().any():
        raise ValueError("whole-exam audit manifest has duplicate view IDs")
    orientation = json.loads(orientation_path.read_text())
    if orientation.get("adapter_sha256") != args.expected_adapter_sha256:
        raise RuntimeError("orientation result adapter SHA-256 mismatch")
    predictions = orientation.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("orientation result has no predictions")
    frame = pd.DataFrame.from_records(predictions)
    required = {
        "source_view_id",
        "split",
        "rotation_degrees_clockwise",
        "paired_logit_contrast",
        "paired_predicted_label",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"orientation predictions missing columns: {missing}")
    if set(frame["split"]) != {"audit"}:
        raise ValueError("orientation result must contain only the audit split")
    frame["source_view_id"] = frame["source_view_id"].map(normalize_view_id)
    if frame[["source_view_id", "rotation_degrees_clockwise"]].duplicated().any():
        raise ValueError("orientation result contains duplicate source variants")
    for source_view_id, pair in frame.groupby("source_view_id"):
        if len(pair) != 2 or set(pair["rotation_degrees_clockwise"]) != {0, 180}:
            raise ValueError("orientation result does not contain exact rotation pairs")
        if set(pair["paired_predicted_label"]) != {"UPRIGHT", "INVERTED"}:
            raise ValueError("orientation pair is tied or lacks opposite predictions")
        contrasts = {
            int(row.rotation_degrees_clockwise): float(row.paired_logit_contrast)
            for row in pair.itertuples(index=False)
        }
        if contrasts[0] != -contrasts[180] or contrasts[0] == 0:
            raise ValueError("orientation paired contrasts are not exact opposites")
    originals = frame[frame["rotation_degrees_clockwise"].eq(0)].copy()
    manifest_mlo = set(manifest.loc[manifest["view"].eq("MLO"), "view_id"])
    if set(originals["source_view_id"]) != manifest_mlo:
        raise ValueError("orientation result does not cover every audit MLO exactly")
    positive_ids = set(
        originals.loc[
            originals["paired_predicted_label"].eq("INVERTED"), "source_view_id"
        ]
    )
    contrast_by_view = {
        str(row.source_view_id): float(row.paired_logit_contrast)
        for row in originals.itertuples(index=False)
    }

    run = new_view_auto_run(
        target=target,
        model=(
            f"{orientation['model_repo_id']}@{orientation['model_revision']}+"
            f"LoRA[{orientation['adapter_sha256'][:12]}]"
        ),
        prompt_variant="smolvlm_mlo_orientation_paired_logits_v1",
        inference_settings={
            "combination_rule": "MLO paired-logit orientation rejection; CC passthrough",
            "paired_rule": orientation["paired_logit_contrast"]["method"],
            "threshold": 0.0,
            "orientation_result_file": str(orientation_path),
            "orientation_result_sha256": sha256_file(orientation_path),
            "orientation_manifest": orientation["manifest"],
            "orientation_manifest_sha256": orientation["manifest_sha256"],
            "adapter_sha256": orientation["adapter_sha256"],
            "adapter_config_sha256": orientation["adapter_config_sha256"],
            "whole_exam_manifest_sha256": sha256_file(manifest_path),
            "mlo_views": len(originals),
            "orientation_positive_views": len(positive_ids),
        },
    )
    run["backend"] = "derived_paired_vlm_logits"
    run["prompt_mode"] = "paired_orientation_classifier"
    run["view_suggestions"] = {
        str(row.view_id): {
            "image_path": str(row.image_path),
            "suggestions": (
                [
                    {
                        "tag": target,
                        "confidence": "high",
                        "rationale": (
                            "MLO superior-inferior orientation is inverted by frozen "
                            "zero-threshold paired-logit contrast "
                            f"({contrast_by_view[str(row.view_id)]:.6g})"
                        ),
                    }
                ]
                if str(row.view_id) in positive_ids
                else []
            ),
        }
        for row in manifest.itertuples(index=False)
    }
    saved = save_view_auto_run(output_path, run)
    print(
        "MLO orientation view auto-QC ready: "
        f"views={len(saved['view_suggestions'])} mlo={len(originals)} "
        f"positive={len(positive_ids)} output_sha256={sha256_file(output_path)}"
    )
    return saved


def main() -> int:
    run_from_args(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
