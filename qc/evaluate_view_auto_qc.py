#!/usr/bin/env python3
"""Evaluate a frozen view-level auto-QC run against complete blinded labels."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd

from prima.view_auto_qc import (
    VIEW_CONFIDENCE_LEVELS,
    load_view_auto_run,
    view_suggestion_is_target_present,
)
from prima.view_qc import (
    VIEW_LABEL_PRESENT,
    load_view_qc_state,
    normalize_view_id,
    summarize_view_qc_state,
    validate_view_manifest_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--run-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--minimum-present-confidence",
        choices=VIEW_CONFIDENCE_LEVELS,
        required=True,
    )
    return parser.parse_args()


def wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return center - half_width, center + half_width


def confusion_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    human = rows["human_target_present"].astype(bool)
    model = rows["model_target_present"].astype(bool)
    tp = int((human & model).sum())
    fn = int((human & ~model).sum())
    fp = int((~human & model).sum())
    tn = int((~human & ~model).sum())

    def metric(successes: int, total: int) -> dict[str, Any]:
        low, high = wilson_interval(successes, total)
        return {
            "value": None if total == 0 else successes / total,
            "numerator": successes,
            "denominator": total,
            "wilson_95_low": low,
            "wilson_95_high": high,
        }

    return {
        "n": int(len(rows)),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "sensitivity": metric(tp, tp + fn),
        "specificity": metric(tn, tn + fp),
        "precision": metric(tp, tp + fp),
        "negative_predictive_value": metric(tn, tn + fn),
    }


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    state_path = args.state.resolve()
    run_path = args.run_file.resolve()
    out_dir = args.out_dir.resolve()
    for path in (manifest_path, state_path, run_path):
        if not path.is_file():
            raise FileNotFoundError(f"required view QC input not found: {path}")
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {out_dir}")

    manifest = pd.read_parquet(manifest_path)
    validate_view_manifest_columns(manifest.columns, str(manifest_path))
    if "stratum" not in manifest.columns:
        raise ValueError("view QC manifest is missing stratum")
    manifest = manifest.copy()
    manifest["view_id"] = manifest["view_id"].map(normalize_view_id)
    state = load_view_qc_state(state_path)
    progress = summarize_view_qc_state(state, manifest["view_id"])
    if progress["remaining"] != 0:
        raise RuntimeError(
            "view QC evaluation requires complete blinded labels; "
            f"{progress['remaining']} remain"
        )
    if progress["low_confidence"]:
        raise RuntimeError(
            "view QC evaluation requires adjudicated labels; "
            f"{progress['low_confidence']} remain low confidence"
        )
    run = load_view_auto_run(run_path)
    if run["target"] != state["target"]:
        raise RuntimeError("human state and model run use different targets")
    run_ids = set(run["view_suggestions"])
    manifest_ids = set(manifest["view_id"])
    if run_ids != manifest_ids:
        raise RuntimeError(
            "view auto-QC coverage must exactly match the evaluation manifest"
        )

    labels = state["labels"]
    predictions = run["view_suggestions"]
    manifest["human_target_present"] = manifest["view_id"].map(
        lambda view_id: labels[view_id]["label"] == VIEW_LABEL_PRESENT
    )
    manifest["model_target_present"] = manifest["view_id"].map(
        lambda view_id: view_suggestion_is_target_present(
            predictions[view_id],
            target=state["target"],
            minimum_confidence=args.minimum_present_confidence,
        )
    )
    manifest["agreement"] = (
        manifest["human_target_present"] == manifest["model_target_present"]
    )

    metrics = {
        "target": state["target"],
        "minimum_present_confidence": args.minimum_present_confidence,
        "manifest_rows": int(len(manifest)),
        "overall": confusion_metrics(manifest),
        "by_stratum": {
            str(stratum): confusion_metrics(rows)
            for stratum, rows in manifest.groupby("stratum", sort=True)
        },
    }
    disagreements = manifest[~manifest["agreement"]].copy()
    out_dir.mkdir(parents=True, mode=0o700)
    metrics_path = out_dir / "metrics.json"
    disagreements_path = out_dir / "disagreements.parquet"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    disagreements.to_parquet(disagreements_path, index=False)
    os.chmod(metrics_path, 0o600)
    os.chmod(disagreements_path, 0o600)
    overall = metrics["overall"]
    print(
        "view auto-QC evaluation complete: "
        f"n={overall['n']} tp={overall['tp']} fn={overall['fn']} "
        f"fp={overall['fp']} tn={overall['tn']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
