#!/usr/bin/env python3
"""Render one restart-safe shard of a prepared view-candidate campaign."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from prima.view_render import render_source_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--temp-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def write_marker(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_from_args(args: argparse.Namespace) -> int:
    os.umask(0o077)
    campaign_dir = args.campaign_dir.resolve()
    raw_root = args.raw_root.resolve()
    metadata_path = campaign_dir / "campaign.json"
    source_path = campaign_dir / "render_sources.parquet"
    if not metadata_path.is_file() or not source_path.is_file():
        raise FileNotFoundError("prepared render campaign inputs are incomplete")
    metadata = json.loads(metadata_path.read_text())
    num_shards = int(metadata["render_shards"])
    if not 0 <= args.shard_index < num_shards:
        raise ValueError(f"--shard-index must be between 0 and {num_shards - 1}")
    max_pixels = int(metadata["max_render_pixels"])
    marker_path = campaign_dir / "render_markers" / f"shard_{args.shard_index:03d}.json"
    if marker_path.exists() and not args.resume:
        raise FileExistsError("render shard already has a completion marker")

    sources = pd.read_parquet(source_path)
    if "render_shard" not in sources.columns:
        raise ValueError("render source table is missing render_shard")
    shard = sources[sources["render_shard"] == args.shard_index].copy()
    expected = int(metadata["render_shard_rows"][args.shard_index])
    if len(shard) != expected or expected <= 0:
        raise RuntimeError("render shard row count disagrees with campaign metadata")
    print(
        f"render shard starting: shard={args.shard_index}/{num_shards} rows={len(shard)}"
    )
    counts = render_source_rows(
        shard,
        raw_root=raw_root,
        out_dir=campaign_dir,
        max_pixels=max_pixels,
        resume=args.resume,
        allow_pixel_failures=True,
        temp_root=args.temp_root,
        progress_desc=f"render shard {args.shard_index:03d}",
    )
    write_marker(
        marker_path,
        {
            "schema_version": 1,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "shard_index": int(args.shard_index),
            **counts,
        },
    )
    print(
        f"render shard complete: shard={args.shard_index} "
        f"rendered={counts['rendered']} reused={counts['reused']} "
        f"failed={counts['failed']}"
    )
    return 0


def main() -> int:
    return run_from_args(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
