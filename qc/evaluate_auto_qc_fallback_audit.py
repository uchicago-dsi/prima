#!/usr/bin/env python3
"""Evaluate a complete blinded audit at both view and fallback-decision level."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from prima.view_auto_qc import load_view_auto_run
from prima.view_qc import (
    VIEW_LABEL_VERTICAL_LINE,
    load_view_qc_state,
    normalize_view_id,
    summarize_view_qc_state,
    validate_view_manifest_columns,
)
from qc.evaluate_view_auto_qc import confusion_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--hidden-model-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def evaluate_group(rows: pd.DataFrame) -> tuple[bool, str]:
    strata = set(rows["stratum"].astype(str))
    if len(strata) != 1:
        raise ValueError("one audit group contains multiple strata")
    stratum = next(iter(strata))
    rows = rows.sort_values("selection_rank", kind="stable")
    human = rows["human_positive"].astype(bool).tolist()
    if stratum == "alternate_pass":
        correct = len(human) >= 2 and all(human[:-1]) and not human[-1]
        rule = "all rejected predecessors positive and accepted alternate negative"
    elif stratum == "no_passing_candidate":
        correct = bool(human) and all(human)
        rule = "every exhausted candidate positive"
    elif stratum == "original_pass_control":
        correct = len(human) == 1 and not human[0]
        rule = "original model-pass control negative"
    else:
        raise ValueError(f"unsupported fallback audit stratum: {stratum!r}")
    return correct, rule


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    state_path = args.state.resolve()
    run_path = args.hidden_model_run.resolve()
    out_dir = args.out_dir.resolve()
    for path in (manifest_path, state_path, run_path):
        if not path.is_file():
            raise FileNotFoundError(f"required fallback audit input not found: {path}")
    if out_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite fallback audit evaluation: {out_dir}"
        )

    manifest = pd.read_parquet(manifest_path)
    validate_view_manifest_columns(manifest.columns, str(manifest_path))
    required = {"stratum", "audit_group_id", "selection_rank"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(
            "fallback audit manifest is missing columns: " + ", ".join(missing)
        )
    manifest = manifest.copy()
    manifest["view_id"] = manifest["view_id"].map(normalize_view_id)
    state = load_view_qc_state(state_path)
    progress = summarize_view_qc_state(state, manifest["view_id"])
    if progress["remaining"]:
        raise RuntimeError(
            f"fallback audit evaluation requires complete labels; {progress['remaining']} remain"
        )
    run = load_view_auto_run(run_path)
    if set(run["view_suggestions"]) != set(manifest["view_id"]):
        raise RuntimeError("hidden audit model run does not exactly cover the manifest")

    labels = state["labels"]
    manifest["human_positive"] = manifest["view_id"].map(
        lambda view_id: labels[view_id]["label"] == VIEW_LABEL_VERTICAL_LINE
    )
    manifest["model_positive"] = manifest["view_id"].map(
        lambda view_id: bool(run["view_suggestions"][view_id]["suggestions"])
    )
    manifest["agreement"] = manifest["human_positive"] == manifest["model_positive"]

    group_rows = []
    for group_id, rows in manifest.groupby("audit_group_id", sort=True):
        correct, rule = evaluate_group(rows)
        group_rows.append(
            {
                "audit_group_id": str(group_id),
                "stratum": str(rows.iloc[0]["stratum"]),
                "view_count": int(len(rows)),
                "decision_correct": bool(correct),
                "decision_rule": rule,
            }
        )
    groups = pd.DataFrame(group_rows)
    group_summary = {
        str(stratum): {
            "groups": int(len(rows)),
            "correct": int(rows["decision_correct"].sum()),
            "incorrect": int((~rows["decision_correct"]).sum()),
        }
        for stratum, rows in groups.groupby("stratum", sort=True)
    }
    metrics = {
        "views": confusion_metrics(manifest),
        "views_by_stratum": {
            str(stratum): confusion_metrics(rows)
            for stratum, rows in manifest.groupby("stratum", sort=True)
        },
        "fallback_groups_by_stratum": group_summary,
    }
    out_dir.mkdir(parents=True, mode=0o700)
    metrics_path = out_dir / "metrics.json"
    view_failures_path = out_dir / "view_disagreements.parquet"
    group_failures_path = out_dir / "decision_failures.parquet"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    manifest[~manifest["agreement"]].to_parquet(view_failures_path, index=False)
    groups[~groups["decision_correct"]].to_parquet(group_failures_path, index=False)
    for path in (metrics_path, view_failures_path, group_failures_path):
        os.chmod(path, 0o600)
    print(
        f"fallback audit evaluated: views={len(manifest)} groups={len(groups)} "
        f"view_disagreements={int((~manifest['agreement']).sum())} "
        f"decision_failures={int((~groups['decision_correct']).sum())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
