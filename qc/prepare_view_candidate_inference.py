#!/usr/bin/env python3
"""Prepare restricted render and inference shards for every view candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from prima.dicom_source import SOURCE_ARCHIVE_COLUMN, SOURCE_COLUMNS
from prima.dicom_source import require_source_columns, require_valid_sources
from prima.view_fallback import validate_candidate_table
from prima.view_qc import normalize_view_id

CAMPAIGN_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare all exact-slot candidates for sharded view auto-QC."
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--render-shards", type=int, default=16)
    parser.add_argument("--inference-shards", type=int, default=16)
    parser.add_argument("--max-render-pixels", type=int, default=2_000_000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_render_shard(archive_relpath: str, num_shards: int) -> int:
    digest = hashlib.sha256(archive_relpath.encode()).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def write_restricted_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    candidates_path = args.candidates.resolve()
    out_dir = args.out_dir.resolve()
    if not candidates_path.is_file():
        raise FileNotFoundError(f"candidate table not found: {candidates_path}")
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite campaign directory: {out_dir}")
    if args.render_shards <= 0 or args.inference_shards <= 0:
        raise ValueError("render and inference shard counts must be positive")
    if args.max_render_pixels <= 0:
        raise ValueError("--max-render-pixels must be positive")

    candidates = pd.read_parquet(candidates_path)
    validate_candidate_table(candidates, str(candidates_path))
    require_source_columns(candidates.columns, str(candidates_path))
    require_valid_sources(
        candidates[list(SOURCE_COLUMNS)].to_dict("records"), str(candidates_path)
    )
    candidates = candidates.copy()
    candidates["view_id"] = candidates["sha256"].map(normalize_view_id)
    if candidates["view_id"].duplicated().any():
        raise ValueError("candidate table contains duplicate SHA-256 view IDs")
    candidates["image_path"] = candidates["view_id"].map(
        lambda view_id: f"images/{view_id}.png"
    )
    candidates["render_shard"] = candidates[SOURCE_ARCHIVE_COLUMN].map(
        lambda value: stable_render_shard(str(value), args.render_shards)
    )
    candidates = candidates.sort_values("view_id", kind="stable").reset_index(drop=True)
    candidates["review_order"] = range(1, len(candidates) + 1)
    candidates["inference_shard"] = candidates.index % args.inference_shards

    out_dir.mkdir(parents=True, mode=0o700)
    (out_dir / "images").mkdir(mode=0o700)
    (out_dir / "render_markers").mkdir(mode=0o700)

    source_columns = [
        *SOURCE_COLUMNS,
        "view_id",
        "image_path",
        "laterality",
        "view",
        "selection_rank",
        "is_selected",
        "render_shard",
    ]
    render_sources = candidates[source_columns]
    source_path = out_dir / "render_sources.parquet"
    render_sources.to_parquet(source_path, index=False)
    os.chmod(source_path, 0o600)

    manifest_columns = [
        "view_id",
        "image_path",
        "laterality",
        "view",
        "review_order",
    ]
    manifest_path = out_dir / "manifest.parquet"
    candidates[manifest_columns].to_parquet(manifest_path, index=False)
    os.chmod(manifest_path, 0o600)

    inference_counts: list[int] = []
    for shard_index in range(args.inference_shards):
        shard = candidates[candidates["inference_shard"] == shard_index][
            manifest_columns
        ].copy()
        shard["review_order"] = range(1, len(shard) + 1)
        shard_path = out_dir / f"manifest_shard_{shard_index:03d}.parquet"
        shard.to_parquet(shard_path, index=False)
        os.chmod(shard_path, 0o600)
        inference_counts.append(int(len(shard)))

    render_counts = [
        int((candidates["render_shard"] == shard_index).sum())
        for shard_index in range(args.render_shards)
    ]
    metadata = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "candidate_table": str(candidates_path),
        "candidate_table_sha256": sha256_file(candidates_path),
        "candidate_rows": int(len(candidates)),
        "selected_rows": int(candidates["is_selected"].astype(bool).sum()),
        "archive_count": int(candidates[SOURCE_ARCHIVE_COLUMN].nunique()),
        "render_shards": int(args.render_shards),
        "render_shard_rows": render_counts,
        "inference_shards": int(args.inference_shards),
        "inference_shard_rows": inference_counts,
        "max_render_pixels": int(args.max_render_pixels),
    }
    write_restricted_json(out_dir / "campaign.json", metadata)
    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Full view-candidate auto-QC campaign",
                "",
                "This restricted scratch campaign renders every exact-slot candidate,",
                "scores each image with the frozen view-level model, and never mutates",
                "the production source-of-truth tables.",
                "",
                f"- candidates: `{candidates_path}`",
                f"- candidate rows: `{len(candidates)}`",
                f"- selected rank-1 rows: `{metadata['selected_rows']}`",
                f"- render shards: `{args.render_shards}`",
                f"- inference shards: `{args.inference_shards}`",
                f"- maximum rendered pixels: `{args.max_render_pixels}`",
                f"- input SHA-256: `{metadata['candidate_table_sha256']}`",
                "",
                "The campaign directory and all source-linked artifacts are restricted",
                "to the owner. Images are source-verified against SOP identity and",
                "SHA-256 before rendering.",
                "",
            ]
        )
    )
    os.chmod(readme, 0o600)
    print(
        f"view candidate campaign prepared: rows={len(candidates)} "
        f"render_shards={args.render_shards} inference_shards={args.inference_shards}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
