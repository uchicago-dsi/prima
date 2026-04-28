#!/usr/bin/env python3
"""Summarize target-tag metrics for a prompt ablation run set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from prima.auto_qc import (
    auto_run_to_tag_map,
    compute_exam_level_tag_metrics,
    load_auto_run,
)
from prima.qc_state import load_qc_state, qc_state_to_annotations_map


def parse_labeled_run(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw)
        return path.stem, path
    label, path_text = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"empty label in run spec: {raw}")
    return label, Path(path_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare prompt-ablation auto-QC runs on one target tag."
    )
    parser.add_argument("--qc-file", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--run-file",
        action="append",
        required=True,
        help="Run file path, or label=/path/run.json. Repeat once per arm.",
    )
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--target-recall", type=float, default=0.70)
    parser.add_argument(
        "--max-fp-rate",
        type=float,
        default=0.25,
        help="Maximum FP / GT-negative rate considered tolerable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    qc_state = load_qc_state(args.qc_file.resolve())
    gt_by_exam = {
        exam_id: set(tags)
        for exam_id, tags in qc_state_to_annotations_map(qc_state).items()
    }

    rows: list[dict[str, object]] = []
    for raw_run in args.run_file:
        label, run_path = parse_labeled_run(raw_run)
        auto_run = load_auto_run(run_path.resolve())
        pred_by_exam = {
            exam_id: set(tags)
            for exam_id, tags in auto_run_to_tag_map(auto_run).items()
        }
        exam_ids = {
            str(exam_id) for exam_id in auto_run.get("exam_suggestions", {}).keys()
        }
        if not exam_ids:
            raise RuntimeError(f"run has no scored exams: {run_path}")
        metrics = compute_exam_level_tag_metrics(
            exam_ids=exam_ids,
            gt_by_exam=gt_by_exam,
            pred_by_exam=pred_by_exam,
            tag_catalog=list(auto_run.get("tag_catalog", [])),
        )
        tag_row = next((row for row in metrics if row["tag"] == args.tag), None)
        if tag_row is None:
            raise RuntimeError(f"tag {args.tag!r} not found in metrics for {run_path}")

        tp = int(tag_row["tp"])
        fp = int(tag_row["fp"])
        fn = int(tag_row["fn"])
        tn = int(tag_row["tn"])
        recall = tag_row["recall"]
        precision = tag_row["precision"]
        fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
        passes = (
            recall is not None
            and recall >= args.target_recall
            and fp_rate <= args.max_fp_rate
        )
        rows.append(
            {
                "label": label,
                "run_file": str(run_path.resolve()),
                "exams": len(exam_ids),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": "" if precision is None else f"{precision:.6f}",
                "recall": "" if recall is None else f"{recall:.6f}",
                "fp_rate": f"{fp_rate:.6f}",
                "passes_rule": int(passes),
                "prompt_mode": auto_run.get("prompt_mode", ""),
                "prompt_variant": auto_run.get("prompt_variant", ""),
            }
        )

    fieldnames = list(rows[0].keys())
    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(
        f"{'label':18} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} {'recall':>8} {'fp_rate':>8} {'pass':>5}"
    )
    print("-" * 68)
    for row in rows:
        recall = row["recall"] or "n/a"
        print(
            f"{str(row['label'])[:18]:18} {row['tp']:4d} {row['fp']:4d} {row['fn']:4d} {row['tn']:4d} {str(recall):>8} {row['fp_rate']:>8} {row['passes_rule']:>5}"
        )

    passing = [row for row in rows if row["passes_rule"]]
    if passing:
        print(f"decision: first passing arm is {passing[0]['label']}")
    else:
        print("decision: no arm met the recall/FP rule")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
