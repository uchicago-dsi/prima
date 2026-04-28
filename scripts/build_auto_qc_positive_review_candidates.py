#!/usr/bin/env python3
# ruff: noqa: E402
"""Build a review queue of candidate positives missing from canonical QC GT."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from auto_annotate_qc import expected_montage_path, load_views_df  # noqa: E402
from prima.qc_state import load_qc_state, qc_state_to_annotations_map  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find candidate target-tag positives from a non-canonical source "
            "that are not yet positives in the canonical QC state."
        )
    )
    parser.add_argument("--canonical-qc-file", type=Path, required=True)
    parser.add_argument("--candidate-qc-file", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--out-exam-list", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument(
        "--tag",
        default="vertical line (detector artifact)",
        help="Candidate positive tag to review.",
    )
    parser.add_argument("--max-candidates", type=int, default=None)
    return parser.parse_args()


def montage_paths_for_exams(
    *,
    views_path: Path,
    export_dir: Path,
    exam_ids: set[str],
) -> dict[str, Path]:
    views_df = load_views_df(views_path)
    views_df = views_df[views_df["exam_id"].isin(exam_ids)]
    paths: dict[str, Path] = {}
    for exam_id, exam_views in views_df.groupby("exam_id", sort=True):
        row = exam_views.iloc[0]
        montage_path = expected_montage_path(
            export_dir=export_dir,
            patient_id=str(row["patient_id"]),
            accession=str(row["accession_number"]),
            exam_id=str(exam_id),
        )
        if montage_path.exists():
            paths[str(exam_id)] = montage_path
    return paths


def main() -> int:
    args = parse_args()
    if args.max_candidates is not None and args.max_candidates <= 0:
        raise ValueError("--max-candidates must be positive when provided")

    canonical = {
        exam_id: set(tags)
        for exam_id, tags in qc_state_to_annotations_map(
            load_qc_state(args.canonical_qc_file.resolve())
        ).items()
    }
    candidate = {
        exam_id: set(tags)
        for exam_id, tags in qc_state_to_annotations_map(
            load_qc_state(args.candidate_qc_file.resolve())
        ).items()
    }
    candidate_exam_ids = sorted(
        exam_id
        for exam_id, tags in candidate.items()
        if args.tag in tags and args.tag not in canonical.get(exam_id, set())
    )
    if args.max_candidates is not None:
        candidate_exam_ids = candidate_exam_ids[: args.max_candidates]

    montage_paths = montage_paths_for_exams(
        views_path=args.views.resolve(),
        export_dir=args.export_dir.resolve(),
        exam_ids=set(candidate_exam_ids),
    )
    rows = [
        {
            "exam_id": exam_id,
            "candidate_tag": args.tag,
            "canonical_has_tag": 0,
            "candidate_source": str(args.candidate_qc_file.resolve()),
            "image_path": str(montage_paths[exam_id]),
            "review_decision": "",
            "review_notes": "",
        }
        for exam_id in candidate_exam_ids
        if exam_id in montage_paths
    ]
    if not rows:
        raise RuntimeError("no candidate positives with cached montages found")

    args.out_exam_list.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_exam_list.write_text("\n".join(str(row["exam_id"]) for row in rows) + "\n")
    with args.out_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"candidate positives with cached montages: {len(rows)} (tag={args.tag!r})")
    print(f"exam list: {args.out_exam_list}")
    print(f"manifest:  {args.out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
