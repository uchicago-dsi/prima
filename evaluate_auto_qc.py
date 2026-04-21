#!/usr/bin/env python3
"""Compare a saved auto-QC run against accepted GT annotations."""

from __future__ import annotations

import argparse
from pathlib import Path

from prima.auto_qc import auto_run_to_tag_map, compute_exam_level_tag_metrics, load_auto_run
from prima.qc_state import load_qc_state, qc_state_to_annotations_map


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate an auto-QC suggestion run against accepted GT annotations."
    )
    parser.add_argument(
        "--qc-file",
        type=Path,
        required=True,
        help="Unified QC state JSON with accepted annotations",
    )
    parser.add_argument(
        "--run-file",
        type=Path,
        required=True,
        help="Auto-QC run JSON produced by auto_annotate_qc.py",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=1,
        help="Only print tags with at least this many GT positives or predicted positives (default: 1)",
    )
    args = parser.parse_args()

    qc_state = load_qc_state(args.qc_file.resolve())
    auto_run = load_auto_run(args.run_file.resolve())

    gt_by_exam = {
        exam_id: set(tags)
        for exam_id, tags in qc_state_to_annotations_map(qc_state).items()
    }
    pred_by_exam = {
        exam_id: set(tags)
        for exam_id, tags in auto_run_to_tag_map(auto_run).items()
    }
    scored_exam_ids = {
        str(exam_id) for exam_id in auto_run.get("exam_suggestions", {}).keys()
    }
    exam_ids = scored_exam_ids or (set(gt_by_exam) | set(pred_by_exam))
    if not exam_ids:
        raise RuntimeError("no overlapping GT or predicted exams found")

    rows = compute_exam_level_tag_metrics(
        exam_ids=exam_ids,
        gt_by_exam=gt_by_exam,
        pred_by_exam=pred_by_exam,
        tag_catalog=list(auto_run.get("tag_catalog", [])),
    )

    print(f"run_id: {auto_run.get('run_id', '(unknown)')}")
    print(f"model:  {auto_run.get('model', '(unknown)')}")
    print(f"exams:   {len(exam_ids):,}")
    print()
    print(
        f"{'tag':40} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>6} {'prec':>8} {'recall':>8}"
    )
    print("-" * 82)
    for row in rows:
        support = row["tp"] + row["fp"] + row["fn"]
        if support < args.min_support:
            continue
        precision = "n/a" if row["precision"] is None else f"{row['precision'] * 100:6.1f}%"
        recall = "n/a" if row["recall"] is None else f"{row['recall'] * 100:6.1f}%"
        print(
            f"{row['tag'][:40]:40} {row['tp']:4d} {row['fp']:4d} {row['fn']:4d} {row['tn']:6d} {precision:>8} {recall:>8}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
