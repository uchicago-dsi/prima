#!/usr/bin/env python3
# ruff: noqa: E402
"""Build an enriched fixed exam list for auto-QC prompt diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from auto_annotate_qc import expected_montage_path, load_views_df  # noqa: E402
from prima.qc_state import (
    load_qc_state,
    normalize_annotation_source,
    qc_state_to_annotation_meta_map,
    qc_state_to_annotations_map,
)  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic enriched auto-QC test set with target-tag "
            "positives, hard negatives, and optional near-confuser positives."
        )
    )
    parser.add_argument("--qc-file", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--out-exam-list", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument(
        "--tag",
        default="vertical line (detector artifact)",
        help="Target tag to enrich for.",
    )
    parser.add_argument(
        "--confuser-tag",
        default="horizontal line (detector artifact)",
        help="Near-confuser tag to include among negatives when available.",
    )
    parser.add_argument("--positive-count", type=int, default=30)
    parser.add_argument("--negative-count", type=int, default=30)
    parser.add_argument("--confuser-count", type=int, default=10)
    parser.add_argument(
        "--positive-source",
        choices=["any", "human", "auto"],
        default="human",
        help="Required source for target positives based on annotation_meta.",
    )
    return parser.parse_args()


def has_source(
    meta_by_exam: dict[str, dict[str, dict[str, object]]],
    *,
    exam_id: str,
    tag: str,
    source: str,
) -> bool:
    if source == "any":
        return True
    meta = meta_by_exam.get(exam_id, {}).get(tag, {})
    return normalize_annotation_source(meta.get("source")) == source


def take(records: list[str], count: int, used: set[str]) -> list[str]:
    selected: list[str] = []
    for exam_id in records:
        if exam_id in used:
            continue
        selected.append(exam_id)
        used.add(exam_id)
        if len(selected) >= count:
            break
    return selected


def build_targeted_montage_index(
    *,
    views_path: Path,
    export_dir: Path,
    exam_ids: set[str],
) -> dict[str, dict[str, str]]:
    """Build exam_id -> montage metadata only for reviewed candidate exams."""
    views_df = load_views_df(views_path)
    views_df = views_df[views_df["exam_id"].isin(exam_ids)]
    index: dict[str, dict[str, str]] = {}
    for exam_id, exam_views in views_df.groupby("exam_id", sort=True):
        row = exam_views.iloc[0]
        patient_id = str(row["patient_id"])
        accession = str(row["accession_number"])
        montage_path = expected_montage_path(
            export_dir=export_dir,
            patient_id=patient_id,
            accession=accession,
            exam_id=str(exam_id),
        )
        if not montage_path.exists():
            continue
        index[str(exam_id)] = {
            "exam_id": str(exam_id),
            "patient_id": patient_id,
            "accession_number": accession,
            "image_path": str(montage_path),
        }
    return index


def main() -> int:
    args = parse_args()
    if args.positive_count < 0 or args.negative_count < 0 or args.confuser_count < 0:
        raise ValueError("requested counts must be non-negative")

    qc_state = load_qc_state(args.qc_file.resolve())
    annotations_by_exam = {
        exam_id: set(tags)
        for exam_id, tags in qc_state_to_annotations_map(qc_state).items()
    }
    meta_by_exam = qc_state_to_annotation_meta_map(qc_state)
    montage_index = build_targeted_montage_index(
        views_path=args.views.resolve(),
        export_dir=args.export_dir.resolve(),
        exam_ids=set(qc_state),
    )
    montage_exam_ids = set(montage_index)

    positives = sorted(
        exam_id
        for exam_id, tags in annotations_by_exam.items()
        if exam_id in montage_exam_ids
        and args.tag in tags
        and has_source(
            meta_by_exam,
            exam_id=exam_id,
            tag=args.tag,
            source=args.positive_source,
        )
    )
    confusers = sorted(
        exam_id
        for exam_id, tags in annotations_by_exam.items()
        if exam_id in montage_exam_ids
        and args.tag not in tags
        and args.confuser_tag in tags
    )
    negatives = sorted(
        exam_id
        for exam_id in qc_state
        if exam_id in montage_exam_ids
        and args.tag not in annotations_by_exam.get(exam_id, set())
        and args.confuser_tag not in annotations_by_exam.get(exam_id, set())
    )

    used: set[str] = set()
    selected_positive = take(positives, args.positive_count, used)
    selected_confuser = take(confusers, args.confuser_count, used)
    selected_negative = take(negatives, args.negative_count, used)
    selected_exam_ids = selected_positive + selected_confuser + selected_negative
    if not selected_exam_ids:
        raise RuntimeError("no exams selected for the diagnostic set")

    args.out_exam_list.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_exam_list.write_text("\n".join(selected_exam_ids) + "\n")

    rows: list[dict[str, object]] = []
    cohort_by_exam = {
        **{exam_id: "target_positive" for exam_id in selected_positive},
        **{exam_id: "confuser_negative" for exam_id in selected_confuser},
        **{exam_id: "plain_negative" for exam_id in selected_negative},
    }
    for exam_id in selected_exam_ids:
        tags = annotations_by_exam.get(exam_id, set())
        rows.append(
            {
                "exam_id": exam_id,
                "cohort": cohort_by_exam[exam_id],
                "truth_tag": int(args.tag in tags),
                "truth_confuser_tag": int(args.confuser_tag in tags),
                "annotation_count": len(tags),
                "image_path": montage_index[exam_id]["image_path"],
            }
        )

    with args.out_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "tag": args.tag,
        "confuser_tag": args.confuser_tag,
        "requested": {
            "positive": args.positive_count,
            "confuser": args.confuser_count,
            "negative": args.negative_count,
        },
        "selected": {
            "positive": len(selected_positive),
            "confuser": len(selected_confuser),
            "negative": len(selected_negative),
            "total": len(selected_exam_ids),
        },
        "available_with_montage": {
            "positive": len(positives),
            "confuser": len(confusers),
            "negative": len(negatives),
        },
    }
    (args.out_manifest.with_suffix(".summary.json")).write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
