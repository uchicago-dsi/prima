#!/usr/bin/env python3
"""Evaluate view, exact-slot, and whole-exam behavior of hybrid Mirai QC."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from prima.view_auto_qc import load_view_auto_run, view_suggestion_is_target_present
from prima.view_qc import (
    VIEW_LABEL_PRESENT,
    load_view_qc_state,
    normalize_view_id,
    summarize_view_qc_state,
)
from qc.evaluate_auto_qc_fallback_audit import evaluate_group
from qc.evaluate_view_auto_qc import confusion_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--eligibility-audit", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--system-run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def evaluate_exam(rows: pd.DataFrame) -> dict[str, object]:
    """Summarize four exact-slot decisions as one exam-level disposition."""
    if len(rows) != 4:
        raise ValueError("whole-exam evaluation requires exactly four slot decisions")
    accepted = bool(rows["selected_candidate_rank"].notna().all())
    unsafe_selected = bool(
        (rows["decision_outcome"] == "selected_target_present").any()
    )
    false_exhaustion = bool((rows["decision_outcome"] == "false_exhaustion").any())
    if accepted and unsafe_selected:
        outcome = "unsafe_accepted"
    elif accepted:
        outcome = "complete_safe_accepted"
    elif false_exhaustion:
        outcome = "false_rejected"
    else:
        outcome = "safe_routed_out"
    return {
        "exam_accepted": accepted,
        "exam_safe": not unsafe_selected,
        "exam_outcome": outcome,
    }


def main() -> int:
    args = parse_args()
    group_path = args.group_manifest.resolve()
    eligibility_path = args.eligibility_audit.resolve()
    state_path = args.state.resolve()
    run_path = args.system_run.resolve()
    protocol_path = args.protocol.resolve()
    out_dir = args.out_dir.resolve()
    for path in (group_path, eligibility_path, state_path, run_path, protocol_path):
        if not path.is_file():
            raise FileNotFoundError(f"whole-exam evaluation input not found: {path}")
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite whole-exam evaluation: {out_dir}")

    group = pd.read_parquet(group_path).copy()
    required = {
        "view_id",
        "audit_exam_id",
        "audit_group_id",
        "selection_rank",
        "candidate_count",
        "sampling_stratum",
    }
    missing = sorted(required - set(group.columns))
    if missing:
        raise ValueError("group manifest is missing columns: " + ", ".join(missing))
    group["view_id"] = group["view_id"].map(normalize_view_id)
    if group["view_id"].duplicated().any():
        raise ValueError("group manifest contains duplicate view IDs")

    state = load_view_qc_state(state_path)
    progress = summarize_view_qc_state(state, group["view_id"])
    if progress["remaining"]:
        raise RuntimeError(
            f"whole-exam evaluation requires complete labels; {progress['remaining']} remain"
        )
    if progress["low_confidence"]:
        raise RuntimeError(
            "whole-exam evaluation requires adjudicated labels; "
            f"{progress['low_confidence']} remain low confidence"
        )
    eligibility = pd.read_parquet(eligibility_path).copy()
    eligibility["view_id"] = eligibility["view_id"].map(normalize_view_id)
    if set(eligibility["view_id"]) != set(group["view_id"]):
        raise ValueError("eligibility audit does not cover the group manifest")
    group = group.merge(
        eligibility[["view_id", "is_standard_mirai_view", "exclusion_reasons"]],
        on="view_id",
        how="left",
        validate="one_to_one",
    )

    run = load_view_auto_run(run_path)
    if run["target"] != state["target"]:
        raise ValueError("human state and system target disagree")
    if set(run["view_suggestions"]) != set(group["view_id"]):
        raise ValueError("system run does not cover the group manifest")
    labels = state["labels"]
    group["human_visual_target_present"] = group["view_id"].map(
        lambda value: labels[value]["label"] == VIEW_LABEL_PRESENT
    )
    group["deterministic_target_present"] = ~group["is_standard_mirai_view"].astype(
        bool
    )
    group["human_target_present"] = (
        group["human_visual_target_present"] | group["deterministic_target_present"]
    )
    group["model_target_present"] = group["view_id"].map(
        lambda value: view_suggestion_is_target_present(
            run["view_suggestions"][value],
            target=run["target"],
            minimum_confidence="high",
        )
    )
    group["agreement"] = group["human_target_present"] == group["model_target_present"]
    group["stratum"] = group["sampling_stratum"]

    decision_rows = []
    for group_id, rows in group.groupby("audit_group_id", sort=True):
        result = evaluate_group(rows)
        first = rows.iloc[0]
        decision_rows.append(
            {
                "audit_exam_id": str(first["audit_exam_id"]),
                "audit_group_id": str(group_id),
                "sampling_stratum": str(first["sampling_stratum"]),
                "laterality": str(first["laterality"]),
                "view": str(first["view"]),
                **result,
            }
        )
    decisions = pd.DataFrame(decision_rows)
    exam_rows = []
    for exam_id, rows in decisions.groupby("audit_exam_id", sort=True):
        result = evaluate_exam(rows)
        exam_rows.append(
            {
                "audit_exam_id": str(exam_id),
                "sampling_stratum": str(rows.iloc[0]["sampling_stratum"]),
                **result,
            }
        )
    exams = pd.DataFrame(exam_rows)

    protocol = json.loads(protocol_path.read_text())
    registered = protocol["gate"]
    view_metrics = confusion_metrics(group)
    unsafe_selected_slots = int(
        (decisions["decision_outcome"] == "selected_target_present").sum()
    )
    false_exhausted_slots = int(
        (decisions["decision_outcome"] == "false_exhaustion").sum()
    )
    unsafe_accepted_exams = int((exams["exam_outcome"] == "unsafe_accepted").sum())
    gate = {
        "registered_gate": registered,
        "observed": {
            "view_sensitivity": view_metrics["sensitivity"]["value"],
            "view_specificity": view_metrics["specificity"]["value"],
            "unsafe_selected_slots": unsafe_selected_slots,
            "false_exhausted_slots": false_exhausted_slots,
            "unsafe_accepted_exams": unsafe_accepted_exams,
        },
    }
    gate["passes"] = bool(
        gate["observed"]["view_sensitivity"] >= registered["minimum_view_sensitivity"]
        and gate["observed"]["view_specificity"]
        >= registered["minimum_view_specificity"]
        and unsafe_selected_slots <= registered["maximum_unsafe_selected_slots"]
        and false_exhausted_slots <= registered["maximum_false_exhausted_slots"]
        and unsafe_accepted_exams <= registered["maximum_unsafe_accepted_exams"]
    )
    metrics = {
        "target": state["target"],
        "reference_rule": protocol["reference"],
        "views": view_metrics,
        "views_by_sampling_stratum": {
            str(stratum): confusion_metrics(rows)
            for stratum, rows in group.groupby("sampling_stratum", sort=True)
        },
        "slot_outcomes": decisions["decision_outcome"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "exam_outcomes": exams["exam_outcome"].value_counts().sort_index().to_dict(),
    }

    out_dir.mkdir(parents=True, mode=0o700)
    outputs = {
        "metrics.json": metrics,
        "gate.json": gate,
    }
    for filename, payload in outputs.items():
        path = out_dir / filename
        path.write_text(json.dumps(payload, indent=2) + "\n")
        os.chmod(path, 0o600)
    for filename, table in (
        ("view_disagreements.parquet", group[~group["agreement"]]),
        ("slot_decisions.parquet", decisions),
        ("exam_decisions.parquet", exams),
    ):
        path = out_dir / filename
        table.to_parquet(path, index=False)
        os.chmod(path, 0o600)
    print(
        f"whole-exam Mirai QC evaluated: views={len(group)} slots={len(decisions)} "
        f"exams={len(exams)} gate_passes={gate['passes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
