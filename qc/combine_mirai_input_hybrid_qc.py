#!/usr/bin/env python3
"""Combine modular visual QC with deterministic DICOM view eligibility."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd

from prima.view_auto_qc import (
    VIEW_CONFIDENCE_LEVELS,
    load_view_auto_run,
    new_view_auto_run,
    save_view_auto_run,
    view_suggestion_is_target_present,
)
from prima.view_few_shot import sha256_file
from prima.view_qc import normalize_view_id, normalize_view_qc_target
from qc.combine_view_auto_qc_runs import combine_view_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--eligibility-audit", type=Path, required=True)
    parser.add_argument("--run-file", type=Path, action="append", required=True)
    parser.add_argument("--visual-output", type=Path, required=True)
    parser.add_argument("--system-output", type=Path, required=True)
    parser.add_argument("--decisions-output", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--minimum-present-confidence",
        choices=VIEW_CONFIDENCE_LEVELS,
        required=True,
    )
    parser.add_argument(
        "--allow-repeated-run-targets",
        action="store_true",
        help="allow multiple provenance-distinct visual runs for the same target",
    )
    parser.add_argument(
        "--allow-single-run",
        action="store_true",
        help="allow one visual run to pass through the provenance combiner",
    )
    return parser.parse_args()


def build_hybrid_run(
    *,
    visual_run_path: Path,
    eligibility_path: Path,
    output_path: Path,
    target: str,
    minimum_confidence: str,
) -> dict[str, Any]:
    """Create the deterministic-DICOM OR modular-visual system run."""
    visual_run_path = Path(visual_run_path).resolve()
    eligibility_path = Path(eligibility_path).resolve()
    output_path = Path(output_path).resolve()
    target = normalize_view_qc_target(target)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite hybrid run: {output_path}")
    if not eligibility_path.is_file():
        raise FileNotFoundError(f"eligibility audit not found: {eligibility_path}")
    visual = load_view_auto_run(visual_run_path)
    if not visual:
        raise FileNotFoundError(f"visual run not found: {visual_run_path}")
    if visual["target"] != target:
        raise ValueError("visual run and hybrid target disagree")

    eligibility = pd.read_parquet(eligibility_path)
    required = {"view_id", "is_standard_mirai_view", "exclusion_reasons"}
    missing = sorted(required - set(eligibility.columns))
    if missing:
        raise ValueError("eligibility audit is missing columns: " + ", ".join(missing))
    eligibility = eligibility.copy()
    eligibility["view_id"] = eligibility["view_id"].map(normalize_view_id)
    if eligibility["view_id"].duplicated().any():
        raise ValueError("eligibility audit contains duplicate view IDs")
    if set(eligibility["view_id"]) != set(visual["view_suggestions"]):
        raise ValueError("eligibility and visual runs cover different views")
    eligibility = eligibility.set_index("view_id")

    hybrid = new_view_auto_run(
        target=target,
        model="deterministic_dicom_or[" + visual["model"] + "]",
        prompt_variant="deterministic_dicom_or_modular_visual_v1",
        inference_settings={
            "combination_rule": "logical_or",
            "minimum_visual_confidence": minimum_confidence,
            "visual_run_file": str(visual_run_path),
            "visual_run_sha256": sha256_file(visual_run_path),
            "eligibility_audit_file": str(eligibility_path),
            "eligibility_audit_sha256": sha256_file(eligibility_path),
            "deterministic_rule": "nonstandard_mirai_view_reasons is nonempty",
        },
    )
    hybrid["backend"] = "derived_dicom_or_visual"
    hybrid["prompt_mode"] = "derived_dicom_or_visual"
    records: dict[str, dict[str, Any]] = {}
    for view_id, visual_record in visual["view_suggestions"].items():
        visual_positive = view_suggestion_is_target_present(
            visual_record,
            target=target,
            minimum_confidence=minimum_confidence,
        )
        deterministic_positive = not bool(
            eligibility.at[view_id, "is_standard_mirai_view"]
        )
        reasons = []
        if deterministic_positive:
            reason = str(eligibility.at[view_id, "exclusion_reasons"]).strip()
            reasons.append("DICOM nonstandard" + (f": {reason}" if reason else ""))
        if visual_positive:
            visual_reason = visual_record["suggestions"][0].get("rationale", "")
            reasons.append(
                "visual modular" + (f": {visual_reason}" if visual_reason else "")
            )
        suggestions = []
        if reasons:
            suggestions.append(
                {
                    "tag": target,
                    "confidence": "high",
                    "rationale": "; ".join(reasons),
                }
            )
        records[view_id] = {
            "image_path": visual_record["image_path"],
            "suggestions": suggestions,
        }
    hybrid["view_suggestions"] = records
    return save_view_auto_run(output_path, hybrid)


def build_fallback_decisions(
    group_manifest: pd.DataFrame,
    *,
    target_present_ids: set[str],
) -> pd.DataFrame:
    """Select the first system-usable candidate within each exact slot."""
    required = {
        "view_id",
        "audit_exam_id",
        "audit_group_id",
        "laterality",
        "view",
        "selection_rank",
        "candidate_count",
        "sampling_stratum",
    }
    missing = sorted(required - set(group_manifest.columns))
    if missing:
        raise ValueError("group manifest is missing columns: " + ", ".join(missing))
    work = group_manifest.copy()
    work["view_id"] = work["view_id"].map(normalize_view_id)
    if work["view_id"].duplicated().any():
        raise ValueError("group manifest contains duplicate view IDs")
    target_present = {normalize_view_id(value) for value in target_present_ids}
    if not target_present.issubset(set(work["view_id"])):
        raise ValueError("hybrid positives include views outside the group manifest")
    work["system_target_present"] = work["view_id"].isin(target_present)

    decisions: list[dict[str, object]] = []
    for group_id, rows in work.groupby("audit_group_id", sort=True):
        rows = rows.sort_values("selection_rank", kind="stable")
        ranks = rows["selection_rank"].astype(int).tolist()
        if ranks != list(range(1, len(rows) + 1)):
            raise ValueError("one fallback group has non-contiguous candidate ranks")
        if set(rows["candidate_count"].astype(int)) != {len(rows)}:
            raise ValueError("one fallback group has an invalid candidate count")
        accepted = rows[~rows["system_target_present"].astype(bool)]
        selected = None if accepted.empty else accepted.iloc[0]
        if selected is None:
            status = "no_target_absent_candidate"
        elif int(selected["selection_rank"]) == 1:
            status = "original_target_absent"
        else:
            status = "alternate_target_absent"
        first = rows.iloc[0]
        decisions.append(
            {
                "audit_exam_id": str(first["audit_exam_id"]),
                "audit_group_id": str(group_id),
                "laterality": str(first["laterality"]),
                "view": str(first["view"]),
                "sampling_stratum": str(first["sampling_stratum"]),
                "original_view_id": str(first["view_id"]),
                "selected_view_id": None
                if selected is None
                else str(selected["view_id"]),
                "selected_candidate_rank": None
                if selected is None
                else int(selected["selection_rank"]),
                "fallback_status": status,
                "candidate_count": int(len(rows)),
            }
        )
    return pd.DataFrame(decisions)


def main() -> int:
    args = parse_args()
    visual_path = args.visual_output.resolve()
    system_path = args.system_output.resolve()
    decisions_path = args.decisions_output.resolve()
    if decisions_path.exists():
        raise FileExistsError(
            f"refusing to overwrite fallback decisions: {decisions_path}"
        )
    combine_view_runs(
        manifest_path=args.manifest,
        run_paths=args.run_file,
        output_path=visual_path,
        target=args.target,
        minimum_confidence=args.minimum_present_confidence,
        allow_repeated_targets=args.allow_repeated_run_targets,
        allow_single_run=args.allow_single_run,
    )
    system = build_hybrid_run(
        visual_run_path=visual_path,
        eligibility_path=args.eligibility_audit,
        output_path=system_path,
        target=args.target,
        minimum_confidence=args.minimum_present_confidence,
    )
    positives = {
        view_id
        for view_id, record in system["view_suggestions"].items()
        if view_suggestion_is_target_present(
            record,
            target=system["target"],
            minimum_confidence="high",
        )
    }
    group_manifest = pd.read_parquet(args.group_manifest.resolve())
    decisions = build_fallback_decisions(group_manifest, target_present_ids=positives)
    decisions_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    decisions.to_parquet(decisions_path, index=False)
    os.chmod(decisions_path, 0o600)
    counts = decisions["fallback_status"].value_counts().sort_index().to_dict()
    print(
        f"hybrid Mirai QC combined: views={len(system['view_suggestions'])} "
        f"system_positive={len(positives)} slots={len(decisions)} statuses={counts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
