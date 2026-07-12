#!/usr/bin/env python3
"""Align rebuilt view candidates to authoritative production selections."""

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
from prima.view_fallback import align_candidates_to_selected_views


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--selected-views", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates_path = args.candidates.resolve()
    selected_path = args.selected_views.resolve()
    output_path = args.output.resolve()
    for path in (candidates_path, selected_path):
        if not path.is_file():
            raise FileNotFoundError(f"required view table not found: {path}")
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite aligned candidates: {output_path}"
        )

    candidates = pd.read_parquet(candidates_path)
    selected = pd.read_parquet(selected_path)
    for frame, path in ((candidates, candidates_path), (selected, selected_path)):
        require_source_columns(frame.columns, str(path))
        require_valid_sources(frame[list(SOURCE_COLUMNS)].to_dict("records"), str(path))
    aligned = align_candidates_to_selected_views(
        candidates, selected, context=str(candidates_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_parquet(output_path, index=False)
    os.chmod(output_path, 0o600)
    rank_comparison = candidates[
        ["exam_id", "sop_instance_uid", "selection_rank"]
    ].merge(
        aligned[["exam_id", "sop_instance_uid", "selection_rank"]],
        on=["exam_id", "sop_instance_uid"],
        how="inner",
        validate="one_to_one",
        suffixes=("_before", "_after"),
    )
    if len(rank_comparison) != len(candidates):
        raise RuntimeError("candidate alignment did not preserve every source row")
    changed_rank_one = int(
        (
            rank_comparison["selection_rank_before"].astype(int)
            != rank_comparison["selection_rank_after"].astype(int)
        ).sum()
    )
    print(f"aligned candidate rows: {len(aligned):,}")
    print(f"rows with changed rank: {changed_rank_one:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
