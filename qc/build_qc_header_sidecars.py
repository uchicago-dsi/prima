#!/usr/bin/env python3
"""Build source-linked QC sidecars from authoritative SoT tables.

The cached montage inventory determines which exams are included. View and tag
metadata are subset from the rebuilt SoT rather than re-read from DICOM headers,
so QC uses the exact same source identity and metadata as preprocessing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from prima.dicom_source import (
    SOURCE_COLUMNS,
    require_source_columns,
    require_valid_sources,
)

PNG_PREFIX = "COMBINED_four_views_"
EXPECTED_VIEWS = {("L", "CC"), ("L", "MLO"), ("R", "CC"), ("R", "MLO")}
QC_TAG_COLUMNS = [
    "sop_instance_uid",
    "AcquisitionDeviceProcessingCode",
    "DetectorType",
]


def iter_exported_exams(export_dir: Path) -> Iterable[dict[str, str]]:
    success_dir = export_dir / "success"
    if not success_dir.exists():
        raise FileNotFoundError(f"success directory not found: {success_dir}")

    for png_path in sorted(success_dir.glob("*/*/COMBINED_four_views_*.png")):
        filename = png_path.name
        if not filename.startswith(PNG_PREFIX):
            continue
        yield {
            "patient_id": str(png_path.parent.parent.name),
            "accession_number": str(png_path.parent.name),
            "exam_id": str(filename[len(PNG_PREFIX) : -len(".png")]),
        }


def build_sidecars(
    export_dir: Path,
    views_path: Path,
    tags_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    export_rows = list(iter_exported_exams(export_dir))
    if not export_rows:
        raise RuntimeError(f"no montage PNGs found under {export_dir / 'success'}")

    cached_exams = pd.DataFrame(export_rows)
    if cached_exams["exam_id"].duplicated().any():
        raise RuntimeError("a cached exam maps to more than one montage")

    source_views = pd.read_parquet(views_path)
    require_source_columns(source_views.columns, str(views_path))
    require_valid_sources(
        source_views[list(SOURCE_COLUMNS)].to_dict("records"), str(views_path)
    )
    source_views = source_views.assign(
        patient_id=source_views["patient_id"].astype(str),
        exam_id=source_views["exam_id"].astype(str),
    )

    cached_exam_ids = set(cached_exams["exam_id"])
    selected_views = source_views[source_views["exam_id"].isin(cached_exam_ids)].copy()
    found_exam_ids = set(selected_views["exam_id"])
    missing_exams = cached_exam_ids - found_exam_ids
    if missing_exams:
        raise RuntimeError(
            f"{len(missing_exams)} cached montage exams lack authoritative source rows"
        )

    invalid_quads = 0
    for _, exam_rows in selected_views.groupby("exam_id", sort=False):
        found = set(zip(exam_rows["laterality"], exam_rows["view"]))
        if len(exam_rows) != 4 or found != EXPECTED_VIEWS:
            invalid_quads += 1
    if invalid_quads:
        raise RuntimeError(
            f"{invalid_quads} cached montage exams lack one exact canonical quad"
        )

    cached_identity = cached_exams.set_index("exam_id")
    expected_patients = selected_views["exam_id"].map(cached_identity["patient_id"])
    if not selected_views["patient_id"].eq(expected_patients).all():
        raise RuntimeError("cached montage and authoritative patient mappings disagree")
    selected_views["accession_number"] = selected_views["exam_id"].map(
        cached_identity["accession_number"]
    )

    source_tags = pd.read_parquet(tags_path)
    missing_tag_columns = set(QC_TAG_COLUMNS) - set(source_tags.columns)
    if missing_tag_columns:
        raise RuntimeError(
            "authoritative tags table is missing columns: "
            + ", ".join(sorted(missing_tag_columns))
        )
    source_tags = source_tags[QC_TAG_COLUMNS].drop_duplicates(
        subset=["sop_instance_uid"]
    )
    selected_sops = set(selected_views["sop_instance_uid"])
    selected_tags = source_tags[
        source_tags["sop_instance_uid"].isin(selected_sops)
    ].copy()
    missing_sops = selected_sops - set(selected_tags["sop_instance_uid"])
    if missing_sops:
        raise RuntimeError(
            f"{len(missing_sops)} selected SOPs lack authoritative tag rows"
        )

    selected_views = selected_views.sort_values(
        ["exam_id", "laterality", "view"], kind="stable"
    ).reset_index(drop=True)
    selected_tags = selected_tags.sort_values(
        "sop_instance_uid", kind="stable"
    ).reset_index(drop=True)
    return selected_views, selected_tags


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build source-linked QC views/tags from authoritative SoT tables."
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        required=True,
        help="QC export directory containing success/<patient>/<accession>/*.png",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Raw DICOM root containing sot/views.parquet and sot/dicom_tags.parquet",
    )
    parser.add_argument(
        "--views",
        type=Path,
        default=None,
        help="Authoritative views.parquet (default: <raw-dir>/sot/views.parquet)",
    )
    parser.add_argument(
        "--tags",
        type=Path,
        default=None,
        help="Authoritative dicom_tags.parquet (default: <raw-dir>/sot/dicom_tags.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Sidecar output directory (default: --export-dir)",
    )
    args = parser.parse_args()

    export_dir = args.export_dir.resolve()
    raw_dir = args.raw_dir.resolve()
    views_path = (
        args.views.resolve() if args.views else raw_dir / "sot" / "views.parquet"
    )
    tags_path = (
        args.tags.resolve() if args.tags else raw_dir / "sot" / "dicom_tags.parquet"
    )
    output_dir = args.output_dir.resolve() if args.output_dir else export_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    views_df, tags_df = build_sidecars(export_dir, views_path, tags_path)
    views_output = output_dir / "views_for_qc.parquet"
    tags_output = output_dir / "dicom_tags.parquet"
    views_df.to_parquet(views_output, index=False)
    tags_df.to_parquet(tags_output, index=False)

    print(
        f"wrote {len(views_df):,} selected source-linked views from "
        f"{views_df['exam_id'].nunique():,} exams"
    )
    print(f"wrote {len(tags_df):,} authoritative QC tag rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
