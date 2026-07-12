#!/usr/bin/env python3
"""Validate every rendered image in a prepared view-candidate campaign."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from prima.view_qc import normalize_view_id, validate_rendered_view_png


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", type=Path, required=True)
    return parser.parse_args()


def run_from_args(args: argparse.Namespace) -> int:
    campaign_dir = args.campaign_dir.resolve()
    metadata_path = campaign_dir / "campaign.json"
    manifest_path = campaign_dir / "manifest.parquet"
    output_path = campaign_dir / "render_complete.json"
    if output_path.exists():
        raise FileExistsError("render campaign has already been validated")
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("prepared render campaign inputs are incomplete")
    metadata = json.loads(metadata_path.read_text())
    manifest = pd.read_parquet(manifest_path)
    manifest["view_id"] = manifest["view_id"].map(normalize_view_id)
    if len(manifest) != int(metadata["candidate_rows"]):
        raise RuntimeError("manifest row count disagrees with campaign metadata")
    marker_paths = sorted((campaign_dir / "render_markers").glob("shard_*.json"))
    if len(marker_paths) != int(metadata["render_shards"]):
        raise RuntimeError("not every render shard has a completion marker")
    markers = [json.loads(path.read_text()) for path in marker_paths]
    marker_indices = {int(marker["shard_index"]) for marker in markers}
    if marker_indices != set(range(int(metadata["render_shards"]))):
        raise RuntimeError("render completion markers do not cover every shard index")
    failure_records = [
        failure for marker in markers for failure in marker.get("failures", [])
    ]
    failed_view_ids = [normalize_view_id(row["view_id"]) for row in failure_records]
    if len(failed_view_ids) != len(set(failed_view_ids)):
        raise RuntimeError("one view appears in multiple render failure records")
    failed = set(failed_view_ids)

    max_pixels = int(metadata["max_render_pixels"])
    seen: set[str] = set()
    maximum_pixels = 0
    for row in tqdm(manifest.to_dict("records"), desc="validating rendered views"):
        view_id = normalize_view_id(row["view_id"])
        if view_id in seen:
            raise ValueError("render manifest contains duplicate view IDs")
        seen.add(view_id)
        relative = Path(str(row["image_path"]))
        if relative != Path("images") / f"{view_id}.png":
            raise ValueError("render manifest contains a noncanonical image path")
        if view_id in failed:
            if (campaign_dir / relative).exists():
                raise RuntimeError(
                    "failed render view unexpectedly has an installed image"
                )
            continue
        width, height = validate_rendered_view_png(
            campaign_dir / relative, max_pixels=max_pixels
        )
        maximum_pixels = max(maximum_pixels, width * height)

    actual_images = {path.stem for path in (campaign_dir / "images").glob("*.png")}
    validated = seen - failed
    if actual_images != validated:
        raise RuntimeError(
            "rendered image inventory does not exactly match the manifest"
        )
    if not failed.issubset(seen):
        raise RuntimeError("render failures contain views outside the manifest")

    inference_manifest = manifest[~manifest["view_id"].isin(failed)].copy()
    inference_manifest["review_order"] = range(1, len(inference_manifest) + 1)
    atomic_parquet(campaign_dir / "inference_manifest.parquet", inference_manifest)
    inference_counts = []
    inference_shards = int(metadata["inference_shards"])
    for shard_index in range(inference_shards):
        shard = inference_manifest.iloc[shard_index::inference_shards].copy()
        shard["review_order"] = range(1, len(shard) + 1)
        atomic_parquet(
            campaign_dir / f"manifest_shard_{shard_index:03d}.parquet", shard
        )
        inference_counts.append(int(len(shard)))
    payload = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "candidate_views": len(seen),
        "validated_images": len(validated),
        "failed_views": len(failed),
        "failed_view_ids": sorted(failed),
        "inference_shard_rows": inference_counts,
        "maximum_pixels": maximum_pixels,
    }
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, output_path)
        os.chmod(output_path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        f"render campaign validated: images={len(validated)} "
        f"failed={len(failed)} max_pixels={maximum_pixels}"
    )
    return 0


def main() -> int:
    return run_from_args(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
