#!/usr/bin/env python3
"""Audit a deidentified view manifest against source DICOM Mirai view rules."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from prima.dicom_source import SOURCE_COLUMNS, read_dicom_source, require_source_columns
from prima.view_qc import normalize_view_id, validate_view_manifest_columns
from prima.view_selection import (
    nonstandard_mirai_view_reasons,
    view_modifier_code_meanings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def audit_source(
    row: dict[str, Any], raw_root: Path, temp_root: Path
) -> dict[str, Any]:
    """Read one source header and return only deidentified eligibility fields."""
    dataset = read_dicom_source(
        row,
        raw_root,
        stop_before_pixels=True,
        verify_sha256=False,
        temp_root=temp_root,
    )
    reasons = nonstandard_mirai_view_reasons(dataset)
    return {
        "view_id": normalize_view_id(row["sha256"]),
        "is_standard_mirai_view": not reasons,
        "view_position": str(dataset.get("ViewPosition", "") or "").strip(),
        "view_modifiers": " | ".join(view_modifier_code_meanings(dataset)),
        "partial_view": str(dataset.get("PartialView", "") or "").strip(),
        "exclusion_reasons": " | ".join(reasons),
    }


def restricted_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_from_args(args: argparse.Namespace) -> pd.DataFrame:
    manifest_path = args.manifest.resolve()
    candidates_path = args.candidates.resolve()
    raw_root = args.raw_root.resolve()
    output_path = args.output.resolve()
    summary_path = args.summary.resolve()
    temp_root = args.temp_root.resolve()
    for path in (manifest_path, candidates_path):
        if not path.is_file():
            raise FileNotFoundError(f"eligibility audit input not found: {path}")
    for path in (raw_root, temp_root):
        if not path.is_dir():
            raise FileNotFoundError(f"eligibility audit directory not found: {path}")
    for path in (output_path, summary_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite eligibility audit: {path}")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    manifest = pd.read_parquet(manifest_path)
    validate_view_manifest_columns(manifest.columns, str(manifest_path))
    manifest = manifest.copy()
    manifest["view_id"] = manifest["view_id"].map(normalize_view_id)
    if manifest["view_id"].duplicated().any():
        raise ValueError("eligibility audit manifest contains duplicate view IDs")

    candidates = pd.read_parquet(candidates_path)
    require_source_columns(candidates.columns, str(candidates_path))
    candidates = candidates[[*SOURCE_COLUMNS]].copy()
    candidates["view_id"] = candidates["sha256"].map(normalize_view_id)
    if candidates["view_id"].duplicated().any():
        raise ValueError("candidate table contains duplicate view IDs")
    sources = manifest[["view_id"]].merge(
        candidates,
        on="view_id",
        how="left",
        validate="one_to_one",
    )
    if sources[list(SOURCE_COLUMNS)].isna().any().any():
        raise RuntimeError("eligibility audit manifest contains an unknown source")

    rows = sources.to_dict("records")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(
            executor.map(
                lambda row: audit_source(row, raw_root, temp_root),
                rows,
            )
        )
    audit = pd.DataFrame(results).sort_values("view_id", kind="stable")
    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    audit.to_parquet(output_path, index=False)
    os.chmod(output_path, 0o600)

    excluded = audit[~audit["is_standard_mirai_view"].astype(bool)]
    reason_counts = Counter(excluded["exclusion_reasons"].astype(str))
    restricted_json(
        summary_path,
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join([sys.executable, *sys.argv]),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "manifest": str(manifest_path),
            "candidates": str(candidates_path),
            "raw_root": str(raw_root),
            "views": int(len(audit)),
            "standard_views": int(audit["is_standard_mirai_view"].sum()),
            "excluded_views": int(len(excluded)),
            "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        },
    )
    return audit


def main() -> int:
    args = parse_args()
    audit = run_from_args(args)
    standard = int(audit["is_standard_mirai_view"].sum())
    print(
        f"view eligibility audited: views={len(audit)} "
        f"standard={standard} excluded={len(audit) - standard}"
    )
    print(f"output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
