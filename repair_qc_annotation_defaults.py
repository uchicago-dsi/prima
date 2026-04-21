#!/usr/bin/env python3
"""Repair legacy QC annotations affected by default-tag injection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prima.qc_state import (
    DEFAULT_ANNOTATION_TAGS,
    load_qc_state,
    normalize_annotation_meta,
    normalize_annotation_tags,
    save_qc_state,
)


def repair_qc_state_defaults(qc_state: dict[str, dict]) -> tuple[dict[str, dict], dict[str, object]]:
    """Drop obviously injected default tags and flag ambiguous legacy cases."""
    default_tags = set(DEFAULT_ANNOTATION_TAGS)
    repaired: dict[str, dict] = {}
    ambiguous_exam_ids: list[str] = []
    dropped_default_count = 0
    repaired_exam_count = 0

    for exam_id, record in qc_state.items():
        annotations = normalize_annotation_tags(record.get("annotations", []))
        if not annotations:
            repaired[exam_id] = record
            continue

        extras = [tag for tag in annotations if tag not in default_tags]
        annotation_meta = normalize_annotation_meta(
            record.get("annotation_meta"), annotations
        )

        if extras:
            new_annotations = extras
            dropped_default_count += len([tag for tag in annotations if tag in default_tags])
            repaired_exam_count += 1
            repaired[exam_id] = {
                **record,
                "annotations": new_annotations,
                "annotation_meta": normalize_annotation_meta(annotation_meta, new_annotations),
            }
            continue

        if set(annotations) == default_tags:
            ambiguous_exam_ids.append(str(exam_id))
            for tag in annotations:
                entry = dict(annotation_meta.get(tag, {"source": "human"}))
                entry["legacy_suspect_default_injection"] = True
                annotation_meta[tag] = entry

        repaired[exam_id] = {
            **record,
            "annotations": annotations,
            "annotation_meta": normalize_annotation_meta(annotation_meta, annotations),
        }

    summary = {
        "exam_records": len(qc_state),
        "repaired_exam_count": repaired_exam_count,
        "dropped_default_count": dropped_default_count,
        "ambiguous_default_only_exam_count": len(ambiguous_exam_ids),
        "ambiguous_default_only_exam_ids": ambiguous_exam_ids,
    }
    return repaired, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair legacy QC records where default detector-artifact tags were injected into every annotation."
    )
    parser.add_argument(
        "--qc-file",
        type=Path,
        required=True,
        help="Unified QC state JSON to repair",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the repaired QC state JSON",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write a JSON repair report",
    )
    args = parser.parse_args()

    qc_state = load_qc_state(args.qc_file.resolve())
    repaired, summary = repair_qc_state_defaults(qc_state)
    save_qc_state(args.output.resolve(), repaired)

    if args.report is not None:
        args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
        with open(args.report.resolve(), "w") as f:
            json.dump(summary, f, indent=2)

    print(f"wrote repaired QC state to {args.output.resolve()}")
    print(f"  repaired exams: {summary['repaired_exam_count']}")
    print(f"  dropped injected defaults: {summary['dropped_default_count']}")
    print(
        f"  ambiguous defaults-only exams kept/flagged: {summary['ambiguous_default_only_exam_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
