#!/usr/bin/env python3
"""Evaluate a complete blinded audit at both view and fallback-decision level."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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
from qc.evaluate_view_auto_qc import confusion_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--hidden-model-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--minimum-present-confidence",
        choices=VIEW_CONFIDENCE_LEVELS,
        required=True,
    )
    return parser.parse_args()


def evaluate_group(rows: pd.DataFrame) -> dict[str, object]:
    """Evaluate model-driven selection over one complete candidate sequence."""
    strata = set(rows["stratum"].astype(str))
    if len(strata) != 1:
        raise ValueError("one audit group contains multiple strata")
    rows = rows.sort_values("selection_rank", kind="stable")
    ranks = rows["selection_rank"].astype(int).tolist()
    if ranks != list(range(1, len(rows) + 1)):
        raise ValueError("audit group has a non-contiguous candidate sequence")
    counts = set(rows["candidate_count"].astype(int))
    if counts != {len(rows)}:
        raise ValueError("audit group does not include its complete candidate sequence")

    accepted = rows[~rows["model_target_present"].astype(bool)]
    if accepted.empty:
        safe = bool(rows["human_target_present"].astype(bool).all())
        exact = safe
        outcome = "safe_exhaustion" if safe else "false_exhaustion"
        selected_rank = None
    else:
        selected = accepted.iloc[0]
        selected_rank = int(selected["selection_rank"])
        selected_target_present = bool(selected["human_target_present"])
        predecessors = rows[rows["selection_rank"].astype(int) < selected_rank]
        safe = not selected_target_present
        exact = safe and bool(predecessors["human_target_present"].astype(bool).all())
        outcome = (
            "selected_target_present"
            if selected_target_present
            else "selected_target_absent"
        )
    return {
        "decision_safe": bool(safe),
        "decision_exact": bool(exact),
        "decision_outcome": outcome,
        "selected_candidate_rank": selected_rank,
    }


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
    required = {"stratum", "audit_group_id", "selection_rank", "candidate_count"}
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
    if progress["low_confidence"]:
        raise RuntimeError(
            "fallback audit evaluation requires adjudicated labels; "
            f"{progress['low_confidence']} remain low confidence"
        )
    run = load_view_auto_run(run_path)
    if run["target"] != state["target"]:
        raise RuntimeError("human state and model run use different targets")
    if set(run["view_suggestions"]) != set(manifest["view_id"]):
        raise RuntimeError("hidden audit model run does not exactly cover the manifest")

    labels = state["labels"]
    manifest["human_target_present"] = manifest["view_id"].map(
        lambda view_id: labels[view_id]["label"] == VIEW_LABEL_PRESENT
    )
    manifest["model_target_present"] = manifest["view_id"].map(
        lambda view_id: view_suggestion_is_target_present(
            run["view_suggestions"][view_id],
            target=state["target"],
            minimum_confidence=args.minimum_present_confidence,
        )
    )
    manifest["agreement"] = (
        manifest["human_target_present"] == manifest["model_target_present"]
    )

    group_rows = []
    for group_id, rows in manifest.groupby("audit_group_id", sort=True):
        result = evaluate_group(rows)
        group_rows.append(
            {
                "audit_group_id": str(group_id),
                "stratum": str(rows.iloc[0]["stratum"]),
                "view_count": int(len(rows)),
                **result,
            }
        )
    groups = pd.DataFrame(group_rows)
    group_summary = {
        str(stratum): {
            "groups": int(len(rows)),
            "safe": int(rows["decision_safe"].sum()),
            "unsafe": int((~rows["decision_safe"]).sum()),
            "exact": int(rows["decision_exact"].sum()),
            "inexact": int((~rows["decision_exact"]).sum()),
            "selected_target_absent": int(
                (rows["decision_outcome"] == "selected_target_absent").sum()
            ),
            "selected_target_present": int(
                (rows["decision_outcome"] == "selected_target_present").sum()
            ),
            "safe_exhaustion": int(
                (rows["decision_outcome"] == "safe_exhaustion").sum()
            ),
            "false_exhaustion": int(
                (rows["decision_outcome"] == "false_exhaustion").sum()
            ),
        }
        for stratum, rows in groups.groupby("stratum", sort=True)
    }
    metrics = {
        "target": state["target"],
        "minimum_present_confidence": args.minimum_present_confidence,
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
    group_inexact_path = out_dir / "decision_inexact.parquet"
    group_decisions_path = out_dir / "group_decisions.parquet"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    manifest[~manifest["agreement"]].to_parquet(view_failures_path, index=False)
    groups[~groups["decision_safe"]].to_parquet(group_failures_path, index=False)
    groups[~groups["decision_exact"]].to_parquet(group_inexact_path, index=False)
    groups.to_parquet(group_decisions_path, index=False)
    for path in (
        metrics_path,
        view_failures_path,
        group_failures_path,
        group_inexact_path,
        group_decisions_path,
    ):
        os.chmod(path, 0o600)
    print(
        f"fallback audit evaluated: views={len(manifest)} groups={len(groups)} "
        f"view_disagreements={int((~manifest['agreement']).sum())} "
        f"unsafe_decisions={int((~groups['decision_safe']).sum())} "
        f"inexact_decisions={int((~groups['decision_exact']).sum())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
