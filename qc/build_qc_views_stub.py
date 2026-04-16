#!/usr/bin/env python3
"""Build a minimal views.parquet from cached QC montage PNGs.

This is intended for preprocessed-only QC runs where the combined 4-view
montages already exist under:

    <export_dir>/success/<patient_id>/<accession_number>/COMBINED_four_views_<exam_id>.png

The generated parquet contains four synthetic rows per exam so qc/qc_gallery.py
can group by exam and display the expected view count without requiring raw
DICOM mounts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

VIEW_ORDER = [
    ("L", "CC"),
    ("R", "CC"),
    ("L", "MLO"),
    ("R", "MLO"),
]
PNG_PREFIX = "COMBINED_four_views_"


def build_stub_dataframe(export_dir: Path) -> pd.DataFrame:
    success_dir = export_dir / "success"
    if not success_dir.exists():
        raise FileNotFoundError(f"success directory not found: {success_dir}")

    rows: list[dict[str, object]] = []
    for png_path in sorted(success_dir.glob("*/*/COMBINED_four_views_*.png")):
        accession = png_path.parent.name
        patient_id = png_path.parent.parent.name
        filename = png_path.name
        if not filename.startswith(PNG_PREFIX):
            continue
        exam_id = filename[len(PNG_PREFIX) : -len(".png")]

        for laterality, view in VIEW_ORDER:
            rows.append(
                {
                    "patient_id": str(patient_id),
                    "exam_id": str(exam_id),
                    "accession_number": str(accession),
                    "laterality": laterality,
                    "view": view,
                    "dicom_path": "",
                    "sop_instance_uid": f"stub://{exam_id}/{laterality}_{view}",
                    "has_implant": False,
                    "cached_png_path": str(png_path.relative_to(export_dir)),
                }
            )

    if not rows:
        raise RuntimeError(f"no montage PNGs found under {success_dir}")

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a minimal views.parquet for cached QC montage exports."
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        required=True,
        help="QC export directory containing success/<patient>/<accession>/*.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the stub parquet",
    )
    args = parser.parse_args()

    df = build_stub_dataframe(args.export_dir.resolve())
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(
        f"wrote {len(df):,} rows for {df['exam_id'].nunique():,} exams to {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
