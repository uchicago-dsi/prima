#!/usr/bin/env python3
"""Rescore frozen MLO-orientation predictions after label-only adjudication."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys

import pandas as pd

from prima.view_few_shot import sha256_file
from prima.view_landmark_grid import resolve_relative_image
from qc.train_mlo_orientation_lora import _json_safe, _score_by_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-provenance-sha256", required=True)
    return parser.parse_args()


def _require_hash(path: Path, expected: str, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{description} SHA-256 mismatch: expected={expected} found={actual}"
        )


def _read_manifest(path: Path, expected_sha256: str) -> pd.DataFrame:
    _require_hash(path, expected_sha256, "adjudicated orientation manifest")
    frame = pd.read_parquet(path)
    required = {
        "sample_id",
        "source_view_id",
        "source_key",
        "split",
        "rotation_degrees_clockwise",
        "expected_label",
        "model_image_path",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"adjudicated manifest missing columns: {missing}")
    if frame.empty or frame["sample_id"].duplicated().any():
        raise ValueError("adjudicated manifest is empty or has duplicate samples")
    return frame.set_index("sample_id", verify_integrity=True)


def run_from_args(args: argparse.Namespace) -> dict[str, object]:
    result_path = args.result.resolve()
    manifest_path = args.manifest.resolve()
    provenance_path = args.dataset_provenance.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite adjudicated result: {output_path}"
        )
    _require_hash(result_path, args.expected_result_sha256, "frozen result")
    _require_hash(
        provenance_path,
        args.expected_provenance_sha256,
        "adjudicated dataset provenance",
    )
    result = json.loads(result_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    labeling = provenance.get("orientation_labeling", {})
    if (
        provenance.get("manifest_sha256") != args.expected_manifest_sha256
        or labeling.get("verified_sop_instance_uid") is not True
        or labeling.get("challenge_labels_match_frozen_human_labels") is not True
    ):
        raise RuntimeError("adjudicated dataset provenance is incomplete")
    predictions = result.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("frozen result has no predictions")

    old_manifest_path = Path(str(result.get("manifest", ""))).resolve()
    old_manifest_sha256 = str(result.get("manifest_sha256", ""))
    _require_hash(old_manifest_path, old_manifest_sha256, "frozen result manifest")
    old_manifest = pd.read_parquet(old_manifest_path).set_index(
        "sample_id", verify_integrity=True
    )
    new_manifest = _read_manifest(manifest_path, args.expected_manifest_sha256)
    sample_ids = [str(record.get("sample_id", "")) for record in predictions]
    if len(sample_ids) != len(set(sample_ids)) or not set(sample_ids) <= set(
        new_manifest.index
    ):
        raise ValueError("prediction sample IDs do not map one-to-one to the manifest")
    if not set(sample_ids) <= set(old_manifest.index):
        raise ValueError("prediction sample IDs are missing from the frozen manifest")

    stable_fields = ("source_view_id", "split", "rotation_degrees_clockwise")
    rescored: list[dict[str, object]] = []
    changed_source_keys: set[str] = set()
    changed_rows_by_split: dict[str, int] = {}
    for prediction in predictions:
        sample_id = str(prediction["sample_id"])
        old_row = old_manifest.loc[sample_id]
        new_row = new_manifest.loc[sample_id]
        for field in stable_fields:
            if str(old_row[field]) != str(new_row[field]) or str(
                prediction[field]
            ) != str(new_row[field]):
                raise RuntimeError(f"label adjudication changed stable field {field}")
        if str(prediction["expected_label"]) != str(old_row["expected_label"]):
            raise RuntimeError("frozen prediction label disagrees with its manifest")
        if str(old_row["expected_label"]) != str(new_row["expected_label"]):
            changed_source_keys.add(str(new_row["source_key"]))
            split = str(new_row["split"])
            changed_rows_by_split[split] = changed_rows_by_split.get(split, 0) + 1
        old_image, _ = resolve_relative_image(
            old_manifest_path.parent,
            old_row["model_image_path"],
            description="frozen model_image_path",
        )
        new_image, _ = resolve_relative_image(
            manifest_path.parent,
            new_row["model_image_path"],
            description="adjudicated model_image_path",
        )
        if sha256_file(old_image) != sha256_file(new_image):
            raise RuntimeError("label adjudication changed a frozen model image")
        rescored.append({**prediction, "expected_label": new_row["expected_label"]})

    paired_records = [
        {**record, "predicted_label": record["paired_predicted_label"]}
        for record in rescored
    ]
    payload = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "scope": (
            "metric-only label adjudication of frozen predictions; no model "
            "inference, threshold fitting, or pixel changes"
        ),
        "source_result": str(result_path),
        "source_result_sha256": sha256_file(result_path),
        "source_manifest_sha256": old_manifest_sha256,
        "adjudicated_manifest": str(manifest_path),
        "adjudicated_manifest_sha256": sha256_file(manifest_path),
        "adjudicated_provenance_sha256": sha256_file(provenance_path),
        "orientation_labeling": labeling,
        "evaluated_rows": len(rescored),
        "verified_pixel_identical_rows": len(rescored),
        "changed_label_rows_by_split": changed_rows_by_split,
        "changed_source_pairs": len(changed_source_keys),
        "original_scores": result.get("scores"),
        "scores": _score_by_split(rescored),
        "original_paired_scores": result.get("paired_logit_contrast", {}).get("scores"),
        "paired_logit_contrast": {
            "method": result.get("paired_logit_contrast", {}).get("method"),
            "scores": _score_by_split(paired_records),
        },
        "predictions": rescored,
    }
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n")
    os.chmod(output_path, 0o600)
    print(
        "Adjudicated MLO orientation result ready: "
        f"rows={len(rescored)} changed_pairs={len(changed_source_keys)} "
        f"output_sha256={sha256_file(output_path)}"
    )
    return payload


def main() -> int:
    run_from_args(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
