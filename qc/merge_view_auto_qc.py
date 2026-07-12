#!/usr/bin/env python3
"""Merge disjoint view auto-QC shard runs with exact manifest coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from prima.view_auto_qc import (
    load_view_auto_run,
    require_compatible_view_auto_run,
    save_view_auto_run,
)
from prima.view_qc import normalize_view_id, validate_view_manifest_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def merge_view_runs(
    manifest_path: Path, run_paths: list[Path], output_path: Path
) -> dict[str, object]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite merged run: {output_path}")
    manifest = pd.read_parquet(manifest_path)
    validate_view_manifest_columns(manifest.columns, str(manifest_path))
    manifest = manifest.copy()
    manifest["view_id"] = manifest["view_id"].map(normalize_view_id)
    if manifest["view_id"].duplicated().any():
        raise ValueError("full inference manifest contains duplicate view IDs")
    expected_paths = dict(zip(manifest["view_id"], manifest["image_path"]))
    if not run_paths:
        raise ValueError("no view auto-QC shard runs were found")

    base: dict[str, object] | None = None
    combined: dict[str, object] = {}
    created_at: list[str] = []
    for run_path in sorted(run_paths):
        run = load_view_auto_run(run_path)
        if not run:
            raise ValueError(f"empty view auto-QC shard run: {run_path}")
        if base is None:
            base = run
        else:
            require_compatible_view_auto_run(base, run)
        overlap = set(combined) & set(run["view_suggestions"])
        if overlap:
            raise RuntimeError("view auto-QC shards contain overlapping view IDs")
        for view_id, record in run["view_suggestions"].items():
            if view_id not in expected_paths:
                raise RuntimeError("view auto-QC shard contains a foreign view ID")
            if record["image_path"] != str(expected_paths[view_id]):
                raise RuntimeError(
                    "view auto-QC shard image path disagrees with manifest"
                )
            combined[view_id] = record
        created_at.append(str(run["created_at"]))

    expected = set(expected_paths)
    actual = set(combined)
    if actual != expected:
        raise RuntimeError(
            "merged view auto-QC coverage is incomplete: "
            f"expected={len(expected)} actual={len(actual)} missing={len(expected - actual)}"
        )
    assert base is not None
    merged = {
        **base,
        "run_id": min(created_at).replace(":", "").replace("+00:00", "Z")
        + "_view_qc_merged",
        "created_at": min(created_at),
        "view_suggestions": combined,
    }
    return save_view_auto_run(output_path, merged)


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    run_dir = args.run_dir.resolve()
    output_path = args.output.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"full inference manifest not found: {manifest_path}")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"view auto-QC run directory not found: {run_dir}")
    campaign_path = manifest_path.parent / "campaign.json"
    run_paths = sorted(run_dir.glob("shard_*.json"))
    if campaign_path.is_file():
        campaign = json.loads(campaign_path.read_text())
        expected_shards = int(campaign["inference_shards"])
        if len(run_paths) != expected_shards:
            raise RuntimeError(
                f"expected {expected_shards} inference shard runs, found {len(run_paths)}"
            )
    merged = merge_view_runs(manifest_path, run_paths, output_path)
    suggested = sum(
        bool(record["suggestions"]) for record in merged["view_suggestions"].values()
    )
    print(
        f"merged view auto-QC: shards={len(run_paths)} "
        f"scored={len(merged['view_suggestions'])} suggested={suggested}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
