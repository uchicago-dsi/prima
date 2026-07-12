#!/usr/bin/env python3
"""Apply full model view labels to exact-slot ranked fallback candidates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from prima.dicom_source import (
    SOURCE_COLUMNS,
    require_source_columns,
    require_valid_sources,
)
from prima.view_auto_qc import (
    VIEW_CONFIDENCE_LEVELS,
    load_view_auto_run,
    view_suggestion_meets_confidence,
)
from prima.view_fallback import (
    choose_exact_slot_views_from_outcomes,
    validate_candidate_table,
)
from prima.view_qc import normalize_view_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--view-auto-run", type=Path, required=True)
    parser.add_argument("--render-complete", type=Path, required=True)
    parser.add_argument(
        "--minimum-reject-confidence",
        choices=VIEW_CONFIDENCE_LEVELS,
        required=True,
    )
    parser.add_argument("--decisions-output", type=Path, required=True)
    parser.add_argument("--selected-views-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates_path = args.candidates.resolve()
    run_path = args.view_auto_run.resolve()
    render_complete_path = args.render_complete.resolve()
    decisions_path = args.decisions_output.resolve()
    selected_path = args.selected_views_output.resolve()
    for path in (candidates_path, run_path, render_complete_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"required auto-QC fallback input not found: {path}"
            )
    for path in (decisions_path, selected_path):
        if path.exists():
            raise FileExistsError(
                f"refusing to overwrite auto-QC fallback output: {path}"
            )

    candidates = pd.read_parquet(candidates_path)
    validate_candidate_table(candidates, str(candidates_path))
    require_source_columns(candidates.columns, str(candidates_path))
    require_valid_sources(
        candidates[list(SOURCE_COLUMNS)].to_dict("records"), str(candidates_path)
    )
    candidates = candidates.copy()
    candidates["view_id"] = candidates["sha256"].map(normalize_view_id)
    run = load_view_auto_run(run_path)
    expected = set(candidates["view_id"])
    render_complete = json.loads(render_complete_path.read_text())
    render_failures = {
        normalize_view_id(view_id)
        for view_id in render_complete.get("failed_view_ids", [])
    }
    model_ids = set(run["view_suggestions"])
    if model_ids & render_failures or model_ids | render_failures != expected:
        raise RuntimeError(
            "model results plus deterministic render failures must cover every candidate: "
            f"expected={len(expected)} model={len(model_ids)} "
            f"render_failures={len(render_failures)}"
        )
    model_rejected = {
        view_id
        for view_id, record in run["view_suggestions"].items()
        if view_suggestion_meets_confidence(
            record, minimum_confidence=args.minimum_reject_confidence
        )
    }
    decisions = choose_exact_slot_views_from_outcomes(
        candidates,
        passing_view_ids=model_ids - model_rejected,
        rejected_view_ids=model_rejected | render_failures,
        context=str(candidates_path),
    )
    decisions["minimum_reject_confidence"] = args.minimum_reject_confidence
    resolved = decisions[decisions["selected_view_id"].notna()][
        [
            "exam_id",
            "laterality",
            "view",
            "selected_view_id",
            "fallback_status",
            "minimum_reject_confidence",
        ]
    ]
    selected = resolved.merge(
        candidates,
        left_on=["exam_id", "laterality", "view", "selected_view_id"],
        right_on=["exam_id", "laterality", "view", "view_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(selected) != len(resolved):
        raise RuntimeError("resolved fallback decisions did not map back to candidates")
    selected = selected.sort_values(
        ["exam_id", "laterality", "view"], kind="stable"
    ).reset_index(drop=True)
    decisions = decisions.sort_values(
        ["exam_id", "laterality", "view"], kind="stable"
    ).reset_index(drop=True)

    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_parquet(decisions_path, index=False)
    selected.to_parquet(selected_path, index=False)
    os.chmod(decisions_path, 0o600)
    os.chmod(selected_path, 0o600)
    counts = decisions["fallback_status"].value_counts().sort_index().to_dict()
    print(f"wrote {len(decisions):,} exact-slot model decisions")
    print(f"minimum reject confidence: {args.minimum_reject_confidence}")
    print(
        "status counts: " + ", ".join(f"{key}={value}" for key, value in counts.items())
    )
    print(f"wrote {len(selected):,} resolved selected view rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
