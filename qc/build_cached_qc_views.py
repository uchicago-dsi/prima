#!/usr/bin/env python3
"""Build a cached-montage views table from authoritative DICOM source rows.

This is intended for preprocessed-only QC runs where the combined 4-view
montages already exist under:

    <export_dir>/success/<patient_id>/<accession_number>/COMBINED_four_views_<exam_id>.png

Synthetic source rows are forbidden: every cached montage must retain its SOP,
SHA-256, archive, and archive-member lineage from an authoritative views table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prima.dicom_source import (
    SOURCE_COLUMNS,
    require_source_columns,
    require_valid_sources,
)

PNG_PREFIX = "COMBINED_four_views_"
EXPECTED_VIEWS = {("L", "CC"), ("R", "CC"), ("L", "MLO"), ("R", "MLO")}


def build_cached_views_dataframe(
    export_dir: Path, source_views_path: Path
) -> pd.DataFrame:
    success_dir = export_dir / "success"
    if not success_dir.exists():
        raise FileNotFoundError(f"success directory not found: {success_dir}")

    source_views = pd.read_parquet(source_views_path)
    require_source_columns(source_views.columns, str(source_views_path))
    require_valid_sources(
        source_views[list(SOURCE_COLUMNS)].to_dict("records"), str(source_views_path)
    )
    source_views = source_views.assign(
        exam_id=source_views["exam_id"].astype(str),
        patient_id=source_views["patient_id"].astype(str),
    )

    rows: list[dict[str, object]] = []
    missing_sources = 0
    invalid_quads = 0
    for png_path in sorted(success_dir.glob("*/*/COMBINED_four_views_*.png")):
        accession = png_path.parent.name
        patient_id = png_path.parent.parent.name
        filename = png_path.name
        if not filename.startswith(PNG_PREFIX):
            continue
        exam_id = filename[len(PNG_PREFIX) : -len(".png")]

        exam_rows = source_views[source_views["exam_id"] == str(exam_id)]
        if exam_rows.empty:
            missing_sources += 1
            continue
        found = set(zip(exam_rows["laterality"], exam_rows["view"]))
        if len(exam_rows) != 4 or found != EXPECTED_VIEWS:
            invalid_quads += 1
            continue
        if set(exam_rows["patient_id"]) != {str(patient_id)}:
            raise RuntimeError(
                "cached montage and authoritative patient mapping disagree"
            )

        for source_row in exam_rows.to_dict("records"):
            source_row["accession_number"] = str(accession)
            source_row["cached_png_path"] = str(png_path.relative_to(export_dir))
            rows.append(source_row)

    if missing_sources or invalid_quads:
        raise RuntimeError(
            "cached montage lineage is incomplete: "
            f"{missing_sources} exams lack source rows and "
            f"{invalid_quads} exams lack an exact canonical quad"
        )

    if not rows:
        raise RuntimeError(f"no montage PNGs found under {success_dir}")

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a source-linked views.parquet for cached QC montages."
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        required=True,
        help="QC export directory containing success/<patient>/<accession>/*.png",
    )
    parser.add_argument(
        "--source-views",
        type=Path,
        required=True,
        help="Authoritative views.parquet with archive/member DICOM lineage",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the source-linked cached views parquet",
    )
    args = parser.parse_args()

    df = build_cached_views_dataframe(
        args.export_dir.resolve(), args.source_views.resolve()
    )
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(
        f"wrote {len(df):,} rows for {df['exam_id'].nunique():,} exams to {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
