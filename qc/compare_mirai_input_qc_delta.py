#!/usr/bin/env python3
"""Compare one additive Mirai-input QC arm with its retained baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from prima.view_qc import normalize_view_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-evaluation", type=Path, required=True)
    parser.add_argument("--candidate-evaluation", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "panel manifest containing view_id and review_order; required when "
            "the arm registers exact review orders that must be recovered"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"QC delta input not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"QC delta input is not an object: {path}")
    return payload


def _error_sets(path: Path) -> tuple[set[str], set[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"QC disagreement table not found: {path}")
    frame = pd.read_parquet(path)
    required = {"view_id", "human_target_present", "model_target_present"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "QC disagreement table is missing columns: " + ", ".join(missing)
        )
    frame = frame.copy()
    frame["view_id"] = frame["view_id"].map(normalize_view_id)
    if frame["view_id"].duplicated().any():
        raise ValueError("QC disagreement table contains duplicate view IDs")
    human = frame["human_target_present"].astype(bool)
    model = frame["model_target_present"].astype(bool)
    false_negatives = set(frame.loc[human & ~model, "view_id"])
    false_positives = set(frame.loc[~human & model, "view_id"])
    if len(frame) != len(false_negatives) + len(false_positives):
        raise ValueError("QC disagreement table contains an agreement row")
    return false_negatives, false_positives


def _observed_gate(evaluation_dir: Path) -> dict[str, int | float]:
    gate = _load_json(evaluation_dir / "gate.json")
    observed = gate.get("observed")
    if not isinstance(observed, dict):
        raise ValueError("QC gate lacks observed readouts")
    required = {
        "view_sensitivity",
        "view_specificity",
        "unsafe_selected_slots",
        "false_exhausted_slots",
        "unsafe_accepted_exams",
    }
    missing = sorted(required - set(observed))
    if missing:
        raise ValueError("QC gate lacks readouts: " + ", ".join(missing))
    return {key: observed[key] for key in sorted(required)}


def _review_order_map(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"QC panel manifest not found: {path}")
    frame = pd.read_parquet(path)
    required = {"view_id", "review_order"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("QC panel manifest is missing columns: " + ", ".join(missing))
    frame = frame.copy()
    frame["view_id"] = frame["view_id"].map(normalize_view_id)
    if frame["view_id"].duplicated().any():
        raise ValueError("QC panel manifest contains duplicate view IDs")
    review_order = pd.to_numeric(frame["review_order"], errors="raise")
    if review_order.isna().any() or (review_order % 1 != 0).any():
        raise ValueError("QC panel manifest review orders must be integers")
    frame["review_order"] = review_order.astype(int)
    if frame["review_order"].duplicated().any():
        raise ValueError("QC panel manifest contains duplicate review orders")
    return dict(zip(frame["view_id"], frame["review_order"]))


def _mapped_review_orders(
    view_ids: set[str], review_order_by_view_id: dict[str, int] | None
) -> list[int]:
    if review_order_by_view_id is None:
        return []
    missing = sorted(view_ids - set(review_order_by_view_id))
    if missing:
        raise ValueError(
            "QC panel manifest does not map every changed view to a review order"
        )
    return sorted(review_order_by_view_id[view_id] for view_id in view_ids)


def build_comparison(
    *,
    baseline_evaluation: Path,
    candidate_evaluation: Path,
    protocol: dict[str, Any],
    arm: str,
    review_order_by_view_id: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build a hypothesis-scoped additive-arm comparison."""
    arms = protocol.get("arms")
    if not isinstance(arms, dict) or arm not in arms:
        raise ValueError(f"QC protocol does not define arm: {arm}")
    arm_protocol = arms[arm]
    shared = protocol.get("shared_component_rules")
    if not isinstance(arm_protocol, dict) or not isinstance(shared, dict):
        raise ValueError("QC protocol lacks component rules")

    baseline_metrics = _load_json(baseline_evaluation / "metrics.json")
    candidate_metrics = _load_json(candidate_evaluation / "metrics.json")
    if baseline_metrics.get("target") != candidate_metrics.get("target"):
        raise ValueError("baseline and candidate targets disagree")
    baseline_views = baseline_metrics.get("views")
    candidate_views = candidate_metrics.get("views")
    if not isinstance(baseline_views, dict) or not isinstance(candidate_views, dict):
        raise ValueError("QC metrics lack view readouts")
    if baseline_views.get("n") != candidate_views.get("n"):
        raise ValueError("baseline and candidate view counts disagree")

    baseline_fn, baseline_fp = _error_sets(
        baseline_evaluation / "view_disagreements.parquet"
    )
    candidate_fn, candidate_fp = _error_sets(
        candidate_evaluation / "view_disagreements.parquet"
    )
    recovered_fn = baseline_fn - candidate_fn
    introduced_fn = candidate_fn - baseline_fn
    introduced_fp = candidate_fp - baseline_fp
    resolved_fp = baseline_fp - candidate_fp
    if introduced_fn or resolved_fp:
        raise ValueError(
            "candidate is not a monotonic logical-OR addition to the baseline"
        )

    baseline_gate = _observed_gate(baseline_evaluation)
    candidate_gate = _observed_gate(candidate_evaluation)
    operational_deltas = {
        key: int(candidate_gate[key]) - int(baseline_gate[key])
        for key in (
            "unsafe_selected_slots",
            "false_exhausted_slots",
            "unsafe_accepted_exams",
        )
    }
    minimum_recovered = int(arm_protocol["minimum_recovered_baseline_false_negatives"])
    maximum_introduced_fp = int(arm_protocol["maximum_introduced_false_positives"])
    required_recovered_orders_raw = arm_protocol.get(
        "required_recovered_review_orders", []
    )
    if not isinstance(required_recovered_orders_raw, list):
        raise ValueError("required recovered review orders must be a list")
    required_recovered_orders = sorted(
        {int(value) for value in required_recovered_orders_raw}
    )
    if len(required_recovered_orders) != len(required_recovered_orders_raw):
        raise ValueError("required recovered review orders must be unique")
    if any(value <= 0 for value in required_recovered_orders):
        raise ValueError("required recovered review orders must be positive")
    if required_recovered_orders and review_order_by_view_id is None:
        raise ValueError(
            "arm requires exact recovered review orders but no manifest was provided"
        )
    recovered_review_orders = _mapped_review_orders(
        recovered_fn, review_order_by_view_id
    )
    introduced_fp_review_orders = _mapped_review_orders(
        introduced_fp, review_order_by_view_id
    )
    recovers_required_orders = set(required_recovered_orders).issubset(
        recovered_review_orders
    )
    passes = bool(
        len(recovered_fn) >= minimum_recovered
        and len(introduced_fp) <= maximum_introduced_fp
        and recovers_required_orders
        and operational_deltas["unsafe_selected_slots"]
        <= int(shared["maximum_increase_in_unsafe_selected_slots"])
        and operational_deltas["false_exhausted_slots"]
        <= int(shared["maximum_increase_in_false_exhausted_slots"])
        and operational_deltas["unsafe_accepted_exams"]
        <= int(shared["maximum_increase_in_unsafe_accepted_exams"])
    )
    if passes:
        component_disposition = "retain_for_combined_development_candidate"
        falsified = []
    else:
        component_disposition = "reject_frozen_delta"
        falsified = [
            f"{arm_protocol['target']} as a useful additive component under its frozen prompt and registered development gate"
        ]
    return {
        "schema_version": 2,
        "arm": arm,
        "target": arm_protocol["target"],
        "passes_incremental_gate": passes,
        "baseline": {
            "disposition": "active_and_retained",
            "reason": (
                "an additive development arm cannot supersede the baseline; "
                "a combined candidate must still beat it on a new whole-exam panel"
            ),
            "view_counts": {
                key: int(baseline_views[key]) for key in ("tp", "fn", "fp", "tn")
            },
            "operational_readouts": baseline_gate,
        },
        "candidate": {
            "component_disposition": component_disposition,
            "view_counts": {
                key: int(candidate_views[key]) for key in ("tp", "fn", "fp", "tn")
            },
            "operational_readouts": candidate_gate,
        },
        "delta": {
            "recovered_baseline_false_negatives": len(recovered_fn),
            "recovered_baseline_false_negative_review_orders": (
                recovered_review_orders
            ),
            "introduced_false_positives": len(introduced_fp),
            "introduced_false_positive_review_orders": introduced_fp_review_orders,
            "remaining_false_negatives": len(candidate_fn),
            **operational_deltas,
        },
        "registered_incremental_gate": {
            "minimum_recovered_baseline_false_negatives": minimum_recovered,
            "required_recovered_review_orders": required_recovered_orders,
            "maximum_introduced_false_positives": maximum_introduced_fp,
            "maximum_increase_in_unsafe_selected_slots": int(
                shared["maximum_increase_in_unsafe_selected_slots"]
            ),
            "maximum_increase_in_false_exhausted_slots": int(
                shared["maximum_increase_in_false_exhausted_slots"]
            ),
            "maximum_increase_in_unsafe_accepted_exams": int(
                shared["maximum_increase_in_unsafe_accepted_exams"]
            ),
        },
        "falsification_scope": {
            "falsified": falsified,
            "not_falsified": [
                "consolidated deterministic DICOM eligibility",
                "the frozen modular visual components",
                "the retained deterministic-plus-modular baseline",
            ],
        },
    }


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite QC delta comparison: {output}")
    protocol = _load_json(args.protocol.resolve())
    comparison = build_comparison(
        baseline_evaluation=args.baseline_evaluation.resolve(),
        candidate_evaluation=args.candidate_evaluation.resolve(),
        protocol=protocol,
        arm=args.arm,
        review_order_by_view_id=(
            _review_order_map(args.manifest.resolve()) if args.manifest else None
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(comparison, indent=2) + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, output)
        os.chmod(output, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        "Mirai QC delta compared: "
        f"arm={args.arm} passes={comparison['passes_incremental_gate']} "
        f"recovered_fn={comparison['delta']['recovered_baseline_false_negatives']} "
        f"introduced_fp={comparison['delta']['introduced_false_positives']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
