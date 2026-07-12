#!/usr/bin/env python3
"""Select target-absent mammography views within exact L/R CC/MLO slots."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from prima.dicom_source import (
    SOURCE_COLUMNS,
    require_source_columns,
    require_valid_sources,
)
from prima.view_fallback import choose_exact_slot_views
from prima.view_qc import load_view_qc_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply human view-level QC labels to ranked exact-slot candidates."
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--view-qc-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates_path = args.candidates.resolve()
    state_path = args.view_qc_state.resolve()
    output_path = args.output.resolve()
    if not candidates_path.is_file():
        raise FileNotFoundError(f"candidate table not found: {candidates_path}")
    if not state_path.is_file():
        raise FileNotFoundError(f"view QC state not found: {state_path}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite fallback output: {output_path}")

    candidates = pd.read_parquet(candidates_path)
    require_source_columns(candidates.columns, str(candidates_path))
    require_valid_sources(
        candidates[list(SOURCE_COLUMNS)].to_dict("records"), str(candidates_path)
    )
    state = load_view_qc_state(state_path)
    labels = {view_id: record["label"] for view_id, record in state["labels"].items()}
    selections = choose_exact_slot_views(
        candidates, labels, context=str(candidates_path)
    )
    selections["qc_target"] = state["target"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selections.to_parquet(output_path, index=False)
    os.chmod(output_path, 0o600)
    counts = selections["fallback_status"].value_counts().to_dict()
    print(f"wrote {len(selections):,} exact-slot decisions")
    print(
        "status counts: " + ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
