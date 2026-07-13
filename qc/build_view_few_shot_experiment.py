#!/usr/bin/env python3
"""Build a fixed view-level exemplar bank and matched development subset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import sys

import pandas as pd

from prima.view_auto_qc import load_view_auto_run, save_view_auto_run
from prima.view_few_shot import MAX_VIEW_FEW_SHOT_EXAMPLES, sha256_file
from prima.view_qc import (
    VIEW_LABEL_ABSENT,
    VIEW_LABEL_PRESENT,
    load_view_qc_events,
    load_view_qc_state,
    normalize_view_qc_target,
    save_view_qc_state,
    summarize_view_qc_state,
    validate_view_manifest_columns,
    validate_view_qc_campaign_state,
)

SAFE_EXPERIMENT_MANIFEST_COLUMNS = (
    "view_id",
    "image_path",
    "laterality",
    "view",
    "review_order",
    "stratum",
)


def parse_example(value: str) -> tuple[int, str]:
    """Parse REVIEW_ORDER=ROLE without embedding target-specific vocabulary."""
    raw_order, separator, raw_role = str(value).partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("--example must use REVIEW_ORDER=ROLE")
    try:
        review_order = int(raw_order)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "example review order must be an integer"
        ) from error
    role = raw_role.strip()
    if review_order <= 0 or not role:
        raise argparse.ArgumentTypeError("example requires a positive order and role")
    return review_order, role


def parse_example_label(value: str) -> tuple[int, str]:
    """Parse REVIEW_ORDER=LABEL for an explicitly adjudicated component target."""
    raw_order, separator, raw_label = str(value).partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("--example-label must use REVIEW_ORDER=LABEL")
    try:
        review_order = int(raw_order)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "example-label review order must be an integer"
        ) from error
    label = raw_label.strip().lower()
    if review_order <= 0 or label not in {
        VIEW_LABEL_PRESENT,
        VIEW_LABEL_ABSENT,
    }:
        raise argparse.ArgumentTypeError(
            "example-label requires a positive order and present or absent"
        )
    return review_order, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--operational-target",
        help=(
            "completed campaign/baseline target when --target is one component "
            "of a broader operational union"
        ),
    )
    parser.add_argument(
        "--example",
        type=parse_example,
        action="append",
        required=True,
        help="repeat REVIEW_ORDER=ROLE in the intended prompt order",
    )
    parser.add_argument(
        "--example-label",
        type=parse_example_label,
        action="append",
        help=(
            "repeat REVIEW_ORDER=present|absent for component-target examples; "
            "required exactly once per --example with --operational-target"
        ),
    )
    return parser.parse_args()


def _copy_image(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError("selected view image is missing")
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o600)


def run_from_args(args: argparse.Namespace) -> dict[str, int]:
    """Create one immutable-input mechanism experiment without source identifiers."""
    campaign_dir = args.campaign_dir.resolve()
    baseline_path = args.baseline_run.resolve()
    out_dir = args.out_dir.resolve()
    target = normalize_view_qc_target(args.target)
    raw_operational_target = getattr(args, "operational_target", None)
    operational_target = (
        normalize_view_qc_target(raw_operational_target)
        if raw_operational_target is not None
        else None
    )
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite few-shot experiment: {out_dir}")
    manifest_path = campaign_dir / "manifest.parquet"
    state_path = campaign_dir / "view_qc_state.json"
    events_path = campaign_dir / "view_qc_events.jsonl"
    for path in (manifest_path, state_path, events_path, baseline_path):
        if not path.is_file():
            raise FileNotFoundError(f"few-shot source input not found: {path}")

    manifest = pd.read_parquet(manifest_path)
    validate_view_manifest_columns(manifest.columns, str(manifest_path))
    state = load_view_qc_state(state_path)
    events = load_view_qc_events(events_path)
    validate_view_qc_campaign_state(state, events, manifest["view_id"])
    summary = summarize_view_qc_state(state, manifest["view_id"])
    if summary["remaining"] or summary["uncertain"]:
        raise ValueError("few-shot source campaign must be complete with binary labels")
    source_target = operational_target or target
    if state["target"] != source_target:
        raise ValueError(
            "few-shot operational target does not match the human campaign"
        )
    baseline = load_view_auto_run(baseline_path)
    if baseline["target"] != source_target:
        raise ValueError("few-shot operational target does not match the baseline run")
    if set(baseline["view_suggestions"]) != set(manifest["view_id"]):
        raise ValueError("baseline run coverage does not match the campaign")

    examples = list(args.example)
    if not 2 <= len(examples) <= MAX_VIEW_FEW_SHOT_EXAMPLES:
        raise ValueError(
            "few-shot experiment requires between 2 and "
            f"{MAX_VIEW_FEW_SHOT_EXAMPLES} examples"
        )
    review_orders = [order for order, _role in examples]
    if len(review_orders) != len(set(review_orders)):
        raise ValueError("few-shot example review orders must be unique")
    if manifest["review_order"].duplicated().any():
        raise ValueError("source review manifest contains duplicate review_order")
    by_order = manifest.set_index("review_order", drop=False)
    missing_orders = sorted(set(review_orders) - set(by_order.index))
    if missing_orders:
        raise ValueError("few-shot example review order is outside the campaign")

    explicit_label_pairs = list(getattr(args, "example_label", None) or [])
    explicit_labels: dict[int, str] = {}
    for review_order, label in explicit_label_pairs:
        if review_order in explicit_labels:
            raise ValueError("few-shot example-label review orders must be unique")
        explicit_labels[review_order] = label
    if operational_target is None and explicit_labels:
        raise ValueError("--example-label requires --operational-target")
    if operational_target is not None and set(explicit_labels) != set(review_orders):
        raise ValueError(
            "component few-shot experiments require one --example-label per --example"
        )

    selected_rows = []
    labels = state["labels"]
    for exemplar_order, (review_order, role) in enumerate(examples, start=1):
        row = by_order.loc[review_order].to_dict()
        label = (
            explicit_labels[review_order]
            if operational_target is not None
            else labels[row["view_id"]]["label"]
        )
        if label not in {VIEW_LABEL_PRESENT, VIEW_LABEL_ABSENT}:
            raise ValueError("few-shot examples must have adjudicated binary labels")
        selected_rows.append(
            {
                **row,
                "target": target,
                "label": label,
                "exemplar_order": exemplar_order,
                "role": role,
                "source_review_order": int(review_order),
            }
        )
    selected = pd.DataFrame(selected_rows)
    if set(selected["label"]) != {VIEW_LABEL_PRESENT, VIEW_LABEL_ABSENT}:
        raise ValueError("few-shot examples must include positive and negative labels")

    selected_ids = set(selected["view_id"])
    safe_columns = [
        column for column in SAFE_EXPERIMENT_MANIFEST_COLUMNS if column in manifest
    ]
    evaluation = manifest.loc[
        ~manifest["view_id"].isin(selected_ids), safe_columns
    ].copy()
    if len(evaluation) + len(selected) != len(manifest):
        raise RuntimeError("few-shot development split does not cover the source panel")

    exemplar_dir = out_dir / "exemplars"
    evaluation_dir = out_dir / "evaluation_inputs"
    for directory in (
        out_dir,
        exemplar_dir,
        exemplar_dir / "images",
        evaluation_dir,
        evaluation_dir / "images",
    ):
        directory.mkdir(parents=True, mode=0o700)
        os.chmod(directory, 0o700)

    exemplar_records = []
    for row in selected.sort_values("exemplar_order").to_dict("records"):
        image_name = f"{row['view_id']}.png"
        _copy_image(
            campaign_dir / row["image_path"], exemplar_dir / "images" / image_name
        )
        exemplar_records.append(
            {
                "view_id": row["view_id"],
                "image_path": f"images/{image_name}",
                "target": target,
                "label": row["label"],
                "exemplar_order": int(row["exemplar_order"]),
                "role": row["role"],
                "source_review_order": int(row["source_review_order"]),
            }
        )
    exemplar_manifest = pd.DataFrame(exemplar_records)
    exemplar_manifest_path = exemplar_dir / "manifest.parquet"
    exemplar_manifest.to_parquet(exemplar_manifest_path, index=False)

    evaluation_records = []
    for row in evaluation.sort_values("review_order", kind="stable").to_dict("records"):
        image_name = f"{row['view_id']}.png"
        _copy_image(
            campaign_dir / row["image_path"], evaluation_dir / "images" / image_name
        )
        evaluation_records.append({**row, "image_path": f"images/{image_name}"})
    evaluation_manifest = pd.DataFrame(evaluation_records)
    evaluation_manifest_path = evaluation_dir / "manifest.parquet"
    evaluation_manifest.to_parquet(evaluation_manifest_path, index=False)

    evaluation_state = {
        **state,
        "labels": {
            view_id: labels[view_id] for view_id in evaluation_manifest["view_id"]
        },
    }
    evaluation_state_path = evaluation_dir / "view_qc_state.json"
    save_view_qc_state(evaluation_state_path, evaluation_state)
    baseline_subset = {
        **baseline,
        "view_suggestions": {
            view_id: baseline["view_suggestions"][view_id]
            for view_id in evaluation_manifest["view_id"]
        },
    }
    baseline_subset_path = evaluation_dir / "baseline_run.json"
    save_view_auto_run(baseline_subset_path, baseline_subset)

    command = shlex.join([sys.executable, *sys.argv])
    provenance = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "target": target,
        "operational_target": source_target,
        "example_label_source": (
            "explicit_component_adjudication"
            if operational_target is not None
            else "campaign_state"
        ),
        "source_rows": len(manifest),
        "exemplar_rows": len(exemplar_manifest),
        "evaluation_rows": len(evaluation_manifest),
        "source_digests": {
            "manifest": sha256_file(manifest_path),
            "state": sha256_file(state_path),
            "events": sha256_file(events_path),
            "baseline_run": sha256_file(baseline_path),
        },
        "examples": exemplar_manifest[
            ["source_review_order", "label", "exemplar_order", "role"]
        ].to_dict("records"),
    }
    provenance_path = out_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    readme_path = out_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# View-level few-shot mechanism experiment",
                "",
                f"- target: `{target}`",
                f"- operational target: `{source_target}`",
                "- exemplar labels: "
                + (
                    "explicit component adjudication"
                    if operational_target is not None
                    else "completed campaign state"
                ),
                f"- exemplars: `{len(exemplar_manifest)}`",
                f"- matched evaluation views: `{len(evaluation_manifest)}`",
                "- exemplar and evaluation view IDs are disjoint",
                "- all labels and model outputs come from the completed development panel",
                "- this experiment does not read or modify the blinded holdout",
                "",
                f"Exact producer command: `{command}`",
                "",
            ]
        )
    )
    for path in (
        exemplar_manifest_path,
        evaluation_manifest_path,
        evaluation_state_path,
        baseline_subset_path,
        provenance_path,
        readme_path,
    ):
        os.chmod(path, 0o600)
    return {
        "examples": len(exemplar_manifest),
        "evaluation": len(evaluation_manifest),
    }


def main() -> int:
    result = run_from_args(parse_args())
    print(
        "view few-shot experiment ready: "
        f"examples={result['examples']} evaluation={result['evaluation']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
