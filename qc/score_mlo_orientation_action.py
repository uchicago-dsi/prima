#!/usr/bin/env python3
"""Score a frozen confidence-aware MLO orientation action on one holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from prima.view_qc import normalize_view_id

REQUIRED_PREDICTION_COLUMNS = {
    "source_view_id",
    "split",
    "rotation_degrees_clockwise",
    "expected_label",
    "paired_logit_contrast",
    "paired_predicted_label",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orientation-result", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-orientation-result-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, description: str) -> str:
    actual = sha256_file(path)
    if actual != str(expected).strip().lower():
        raise RuntimeError(f"{description} SHA-256 mismatch")
    return actual


def require_fraction(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"protocol gate {name} must be finite")
    fraction = float(value)
    if not 0 <= fraction <= 1:
        raise ValueError(f"protocol gate {name} must lie in [0, 1]")
    return fraction


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("orientation action score has an empty reference class")
    return numerator / denominator


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    result_path = args.orientation_result.resolve()
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    for path in (result_path, protocol_path):
        if not path.is_file():
            raise FileNotFoundError(f"orientation action input not found: {path}")
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite orientation action score: {output_path}"
        )
    result_sha256 = require_hash(
        result_path,
        args.expected_orientation_result_sha256,
        "orientation result",
    )
    protocol_sha256 = require_hash(
        protocol_path, args.expected_protocol_sha256, "orientation protocol"
    )
    result = json.loads(result_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema_version") != 1:
        raise ValueError("orientation action protocol must use schema_version 1")
    if protocol.get("status") != "frozen_before_orientation_inference":
        raise ValueError("orientation action protocol was not frozen before inference")
    if result.get("manifest_sha256") != protocol.get(
        "orientation_bank_manifest_sha256"
    ):
        raise RuntimeError("orientation result does not match the frozen holdout bank")
    if result.get("adapter_sha256") != protocol.get("orientation_adapter_sha256"):
        raise RuntimeError("orientation result does not use the frozen adapter")
    if result.get("model_repo_id") != protocol.get("orientation_model_repo_id"):
        raise RuntimeError("orientation result model repository mismatch")
    if result.get("model_revision") != protocol.get("orientation_model_revision"):
        raise RuntimeError("orientation result model revision mismatch")
    floor = protocol.get("minimum_inversion_contrast")
    if not isinstance(floor, (int, float)) or not math.isfinite(float(floor)):
        raise ValueError("orientation protocol has no finite confidence floor")
    floor = float(floor)
    if floor <= 0:
        raise ValueError("orientation protocol confidence floor must be positive")
    gate = protocol.get("gate")
    if not isinstance(gate, dict):
        raise ValueError("orientation protocol has no gate")
    minimum_sensitivity = require_fraction(
        gate.get("minimum_natural_inversion_sensitivity"),
        "minimum_natural_inversion_sensitivity",
    )
    minimum_specificity = require_fraction(
        gate.get("minimum_natural_upright_specificity"),
        "minimum_natural_upright_specificity",
    )
    minimum_pair_accuracy = require_fraction(
        gate.get("minimum_pair_accuracy"), "minimum_pair_accuracy"
    )

    predictions = result.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("orientation result has no predictions")
    frame = pd.DataFrame(predictions)
    missing = sorted(REQUIRED_PREDICTION_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            "orientation predictions are missing columns: " + ", ".join(missing)
        )
    if set(frame["split"]) != {"audit"}:
        raise ValueError("orientation holdout result must contain only the audit split")
    frame["source_view_id"] = frame["source_view_id"].map(normalize_view_id)
    if frame[["source_view_id", "rotation_degrees_clockwise"]].duplicated().any():
        raise ValueError("orientation result contains duplicate source variants")
    valid_labels = {"UPRIGHT", "INVERTED"}
    if not set(frame["expected_label"]).issubset(valid_labels):
        raise ValueError("orientation result contains invalid expected labels")
    if not set(frame["paired_predicted_label"]).issubset(valid_labels):
        raise ValueError("orientation result contains invalid paired predictions")

    pair_rows: list[dict[str, Any]] = []
    for source_view_id, pair in frame.groupby("source_view_id", sort=True):
        if len(pair) != 2 or set(pair["rotation_degrees_clockwise"]) != {0, 180}:
            raise ValueError("orientation result does not contain exact rotation pairs")
        by_rotation = pair.set_index("rotation_degrees_clockwise")
        original = by_rotation.loc[0]
        partner = by_rotation.loc[180]
        original_contrast = float(original["paired_logit_contrast"])
        partner_contrast = float(partner["paired_logit_contrast"])
        if not math.isfinite(original_contrast) or not math.isfinite(partner_contrast):
            raise ValueError("orientation result contains a nonfinite paired contrast")
        if original_contrast != -partner_contrast or original_contrast == 0:
            raise ValueError("orientation paired contrasts are not exact opposites")
        if original["expected_label"] == partner["expected_label"]:
            raise ValueError("orientation pair expected labels are not opposites")
        pair_rows.append(
            {
                "source_view_id": source_view_id,
                "expected_label": str(original["expected_label"]),
                "paired_logit_contrast": original_contrast,
                "action_reject": original_contrast <= -floor,
                "raw_pair_exact": bool(
                    (pair["expected_label"] == pair["paired_predicted_label"]).all()
                ),
            }
        )
    pairs = pd.DataFrame(pair_rows)
    actual_inverted = pairs["expected_label"].eq("INVERTED")
    predicted_inverted = pairs["action_reject"].astype(bool)
    true_positive = int((actual_inverted & predicted_inverted).sum())
    false_negative = int((actual_inverted & ~predicted_inverted).sum())
    false_positive = int((~actual_inverted & predicted_inverted).sum())
    true_negative = int((~actual_inverted & ~predicted_inverted).sum())
    sensitivity = safe_rate(true_positive, true_positive + false_negative)
    specificity = safe_rate(true_negative, true_negative + false_positive)
    exact_pairs = int(pairs["raw_pair_exact"].sum())
    pair_accuracy = safe_rate(exact_pairs, len(pairs))
    checks = {
        "natural_inversion_sensitivity": sensitivity >= minimum_sensitivity,
        "natural_upright_specificity": specificity >= minimum_specificity,
        "raw_pair_accuracy": pair_accuracy >= minimum_pair_accuracy,
    }
    payload = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": "passes_frozen_orientation_gate"
        if all(checks.values())
        else "fails_frozen_orientation_gate",
        "orientation_result": str(result_path),
        "orientation_result_sha256": result_sha256,
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "model_repo_id": result["model_repo_id"],
        "model_revision": result["model_revision"],
        "adapter_sha256": result["adapter_sha256"],
        "minimum_inversion_contrast": floor,
        "sources": int(len(pairs)),
        "reference_counts": {
            str(key): int(value)
            for key, value in pairs["expected_label"].value_counts().items()
        },
        "action_confusion": {
            "true_positive": true_positive,
            "false_negative": false_negative,
            "false_positive": false_positive,
            "true_negative": true_negative,
        },
        "natural_inversion_sensitivity": sensitivity,
        "natural_upright_specificity": specificity,
        "raw_exact_pairs": exact_pairs,
        "raw_pair_accuracy": pair_accuracy,
        "false_negative_view_ids": pairs.loc[
            actual_inverted & ~predicted_inverted, "source_view_id"
        ].tolist(),
        "false_positive_view_ids": pairs.loc[
            ~actual_inverted & predicted_inverted, "source_view_id"
        ].tolist(),
        "gate": gate,
        "checks": checks,
        "passes_frozen_orientation_gate": all(checks.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    os.chmod(output_path, 0o600)
    return payload


def main() -> int:
    args = parse_args()
    result = run_from_args(args)
    print(
        "MLO orientation action scored: "
        f"status={result['status']} sources={result['sources']}"
    )
    print(f"output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
