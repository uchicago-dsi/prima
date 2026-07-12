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
    parser.add_argument("--per-stratum", type=int, default=1)
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
    if args.per_stratum <= 0:
        raise ValueError("--per-stratum must be positive")

    manifest = pd.read_parquet(manifest_path)
    validate_view_manifest_columns(manifest.columns, str(manifest_path))
    if "stratum" not in manifest.columns:
        raise ValueError("view manifest is missing stratum")
    subset = (
        manifest.groupby("stratum", group_keys=False, sort=True)
        .sample(n=args.per_stratum, random_state=args.seed)
        .sort_values(["stratum", "view_id"], kind="stable")
        .reset_index(drop=True)
    )
    subset["review_order"] = range(1, len(subset) + 1)
    subset.to_parquet(output_path, index=False)
    os.chmod(output_path, 0o600)
    print(
        f"wrote balanced view subset: rows={len(subset)} "
        f"strata={subset['stratum'].nunique()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
