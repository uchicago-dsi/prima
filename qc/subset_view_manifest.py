#!/usr/bin/env python3
"""Build a deterministic, stratum-balanced subset of a view QC manifest."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from prima.view_qc import validate_view_manifest_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--per-stratum", type=int)
    selection.add_argument(
        "--review-order",
        type=int,
        action="append",
        help="repeat to preserve an explicit source review-order challenge",
    )
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    output_path = args.output.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"view manifest not found: {manifest_path}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite view subset: {output_path}")
    if output_path.parent != manifest_path.parent:
        raise ValueError("view subset must remain beside its source manifest")
    if args.per_stratum is not None and args.per_stratum <= 0:
        raise ValueError("--per-stratum must be positive")

    manifest = pd.read_parquet(manifest_path)
    validate_view_manifest_columns(manifest.columns, str(manifest_path))
    if args.review_order is not None:
        requested = list(args.review_order)
        if any(value <= 0 for value in requested):
            raise ValueError("--review-order values must be positive")
        if len(requested) != len(set(requested)):
            raise ValueError("--review-order values must be unique")
        if manifest["review_order"].duplicated().any():
            raise ValueError("source manifest has duplicate review_order values")
        by_order = manifest.set_index("review_order", drop=False)
        missing = sorted(set(requested) - set(by_order.index))
        if missing:
            raise ValueError("requested review order is outside the source manifest")
        subset = by_order.loc[requested].reset_index(drop=True)
        strategy = "explicit_review_order"
    else:
        if "stratum" not in manifest.columns:
            raise ValueError("view manifest is missing stratum")
        subset = (
            manifest.groupby("stratum", group_keys=False, sort=True)
            .sample(n=args.per_stratum, random_state=args.seed)
            .sort_values(["stratum", "view_id"], kind="stable")
            .reset_index(drop=True)
        )
        subset["review_order"] = range(1, len(subset) + 1)
        strategy = "balanced_stratum_sample"
    subset.to_parquet(output_path, index=False)
    os.chmod(output_path, 0o600)
    print(f"wrote view subset: rows={len(subset)} strategy={strategy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
