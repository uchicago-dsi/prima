#!/usr/bin/env python3
"""Migrate split QC JSON files into the canonical unified QC state file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prima.qc_state import merge_qc_state, save_qc_state


def _load_json_object(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate legacy qc_status.json + annotations.json into qc_state.json."
    )
    parser.add_argument(
        "--qc-file",
        type=Path,
        required=True,
        help="Legacy qc_status.json path",
    )
    parser.add_argument(
        "--annotations-file",
        type=Path,
        default=None,
        help="Legacy annotations.json path (defaults to qc_file sibling annotations.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output unified QC state path (defaults to qc_file sibling qc_state.json)",
    )
    args = parser.parse_args()

    qc_file = args.qc_file.resolve()
    annotations_file = (
        args.annotations_file.resolve()
        if args.annotations_file is not None
        else qc_file.parent / "annotations.json"
    )
    output = (
        args.output.resolve()
        if args.output is not None
        else qc_file.parent / "qc_state.json"
    )

    status_map = _load_json_object(qc_file)
    annotations_map = _load_json_object(annotations_file)
    qc_state = merge_qc_state(
        status_map=status_map,
        annotations_map=annotations_map,
    )
    qc_state = save_qc_state(output, qc_state)

    status_count = sum(1 for record in qc_state.values() if record["status"] is not None)
    annotation_count = sum(1 for record in qc_state.values() if record["annotations"])

    print(f"wrote {len(qc_state):,} exam records to {output}")
    print(f"  exams with status: {status_count:,}")
    print(f"  exams with annotations: {annotation_count:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
