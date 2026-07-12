#!/usr/bin/env python3
# ruff: noqa: E402
"""Audit durable DICOM lineage without printing source identifiers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pydicom

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prima.dicom_source import (
    DicomSource,
    SOURCE_ARCHIVE_COLUMN,
    SOURCE_COLUMNS,
    materialize_dicom_sources,
    require_source_columns,
    require_valid_sources,
    validate_materialized_source,
)

EXPECTED_CANONICAL_VIEWS = {("L", "CC"), ("R", "CC"), ("L", "MLO"), ("R", "MLO")}


def audit_table(
    views: pd.DataFrame,
    raw_root: Path,
    *,
    verify_count: int,
    require_canonical_quads: bool,
) -> dict[str, int]:
    require_source_columns(views.columns, "views table")
    require_valid_sources(views[list(SOURCE_COLUMNS)].to_dict("records"), "views table")

    if views.duplicated(
        subset=["source_archive_relpath", "source_archive_member"]
    ).any():
        raise RuntimeError("views table assigns one source member more than once")

    if require_canonical_quads:
        required = {"exam_id", "laterality", "view"}
        missing = required - set(views.columns)
        if missing:
            raise RuntimeError(
                "canonical-quad audit is missing columns: " + ", ".join(sorted(missing))
            )
        invalid_exams = 0
        for _, exam_rows in views.groupby("exam_id", sort=False):
            found = set(zip(exam_rows["laterality"], exam_rows["view"]))
            if len(exam_rows) != 4 or found != EXPECTED_CANONICAL_VIEWS:
                invalid_exams += 1
        if invalid_exams:
            raise RuntimeError(
                f"{invalid_exams} exams do not contain exactly one canonical quad"
            )

    sources = [DicomSource.from_row(row) for _, row in views.iterrows()]
    unpacked_rows = 0
    archived_rows = 0
    missing_rows = 0
    for source in sources:
        if source.unpacked_path(raw_root).is_file():
            unpacked_rows += 1
        elif source.archive_path(raw_root).is_file():
            archived_rows += 1
        else:
            missing_rows += 1
    if missing_rows:
        raise FileNotFoundError(
            f"{missing_rows} DICOM rows have neither an unpacked member nor an archive"
        )

    if verify_count < 0 or verify_count > len(sources):
        verify_count = len(sources)
    sampled_sources = sorted(sources, key=lambda source: source.source_id)[
        :verify_count
    ]
    verified_rows = 0
    sources_by_archive: dict[str, list[DicomSource]] = {}
    for source in sampled_sources:
        sources_by_archive.setdefault(source.archive_relpath.as_posix(), []).append(
            source
        )
    for archive_sources in sources_by_archive.values():
        with materialize_dicom_sources(archive_sources, raw_root) as paths:
            for source in archive_sources:
                path = paths[source.archive_member.as_posix()]
                dataset = pydicom.dcmread(
                    str(path), force=True, stop_before_pixels=True
                )
                validate_materialized_source(
                    source,
                    path,
                    dataset,
                    verify_sha256=True,
                )
                verified_rows += 1

    return {
        "rows": len(views),
        "exams": views["exam_id"].nunique() if "exam_id" in views else 0,
        "archives": views[SOURCE_ARCHIVE_COLUMN].nunique(),
        "unpacked_rows": unpacked_rows,
        "archived_rows": archived_rows,
        "verified_rows": verified_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit archive/member DICOM lineage without printing identifiers."
    )
    parser.add_argument("--raw", type=Path, required=True, help="Raw DICOM root")
    parser.add_argument(
        "--views",
        type=Path,
        default=None,
        help="Views parquet (default: <raw>/sot/views.parquet)",
    )
    parser.add_argument(
        "--verify-count",
        type=int,
        default=100,
        help="Deterministic number of rows to verify by SOP UID and SHA-256; -1 verifies all",
    )
    parser.add_argument(
        "--allow-noncanonical",
        action="store_true",
        help="Do not require exactly one L/R CC/MLO quad per exam",
    )
    args = parser.parse_args()

    if args.verify_count < -1:
        parser.error("--verify-count must be -1 or nonnegative")
    raw_root = args.raw.resolve()
    views_path = args.views.resolve() if args.views else raw_root / "sot/views.parquet"
    if not views_path.is_file():
        raise FileNotFoundError(f"views table not found: {views_path}")

    result = audit_table(
        pd.read_parquet(views_path),
        raw_root,
        verify_count=args.verify_count,
        require_canonical_quads=not args.allow_noncanonical,
    )
    print(
        "lineage audit passed: "
        f"rows={result['rows']:,}, exams={result['exams']:,}, "
        f"archives={result['archives']:,}, unpacked_rows={result['unpacked_rows']:,}, "
        f"archived_rows={result['archived_rows']:,}, "
        f"verified_rows={result['verified_rows']:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
