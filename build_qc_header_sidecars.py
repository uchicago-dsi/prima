#!/usr/bin/env python3
"""Build minimal QC sidecar parquets from cached montages and DICOM headers.

This is a cheap alternative to running the full preprocessing pipeline when the
combined four-view montage PNGs already exist. It scans only the exams present
in:

    <export_dir>/success/<patient_id>/<accession_number>/COMBINED_four_views_<exam_id>.png

DICOM paths are resolved via:
  - --views: use views.parquet dicom_path column (recommended; works with any raw layout)
  - else: walk <raw_dir>/<patient_id>/<exam_id>/**/*.dcm (requires flat patient/exam_id layout)

Outputs:
  - <export_dir>/views_for_qc.parquet with one selected row per canonical view
  - <export_dir>/dicom_tags.parquet containing only the columns needed by
    prima.qc_filters today

The resulting tables are sufficient for preprocessed-only qc_gallery runs and
preserve SOP Instance UIDs so the duplicate-SOP auto-filter still works.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import pydicom

from preprocess import (
    get_tag,
    has_implant,
    infer_view_fields,
    is_for_presentation,
    is_marked_up,
)
from prima.view_selection import (
    estimate_magnification_factor,
    estimate_pixel_spacing_mm,
    view_selection_key,
)

PNG_PREFIX = "COMBINED_four_views_"


def iter_exported_exams(export_dir: Path) -> Iterable[dict[str, str]]:
    success_dir = export_dir / "success"
    if not success_dir.exists():
        raise FileNotFoundError(f"success directory not found: {success_dir}")

    for png_path in sorted(success_dir.glob("*/*/COMBINED_four_views_*.png")):
        patient_id = png_path.parent.parent.name
        accession_number = png_path.parent.name
        filename = png_path.name
        if not filename.startswith(PNG_PREFIX):
            continue
        exam_id = filename[len(PNG_PREFIX) : -len(".png")]
        yield {
            "patient_id": str(patient_id),
            "accession_number": str(accession_number),
            "exam_id": str(exam_id),
            "png_path": str(png_path),
        }


def read_dicom_header(path: Path) -> pydicom.dataset.FileDataset:
    return pydicom.dcmread(str(path), force=True, stop_before_pixels=True)


def build_sidecars(
    export_dir: Path,
    raw_dir: Path,
    views_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    export_rows = list(iter_exported_exams(export_dir))
    if not export_rows:
        raise RuntimeError(f"no montage PNGs found under {export_dir / 'success'}")

    exam_to_dicom_paths: dict[str, list[Path]] = {}
    if views_path and views_path.exists():
        views_df = pd.read_parquet(views_path)
        views_df["exam_id"] = views_df["exam_id"].astype(str)
        for exam_id, grp in views_df.groupby("exam_id"):
            paths = [Path(p) for p in grp["dicom_path"].dropna().unique() if p]
            exam_to_dicom_paths[str(exam_id)] = paths

    view_rows: list[dict[str, object]] = []
    tag_rows: list[dict[str, object]] = []
    missing_exam_dirs: list[str] = []
    skipped_exams: list[str] = []

    for exam_info in export_rows:
        patient_id = exam_info["patient_id"]
        exam_id = exam_info["exam_id"]
        accession_number = exam_info["accession_number"]

        if exam_to_dicom_paths:
            dicom_paths = exam_to_dicom_paths.get(exam_id, [])
            if not dicom_paths:
                missing_exam_dirs.append(
                    f"{raw_dir}/{patient_id}/{exam_id} (no rows in views)"
                )
                continue
            dicom_iter = iter(dicom_paths)
        else:
            exam_path = raw_dir / patient_id / exam_id
            if not exam_path.exists():
                missing_exam_dirs.append(str(exam_path))
                continue
            dicom_iter = (
                Path(root) / filename
                for root, _dirs, files in os.walk(exam_path)
                for filename in files
                if filename.endswith(".dcm")
            )

        candidate_rows: list[dict[str, object]] = []
        candidate_tags: dict[str, dict[str, object]] = {}

        for dicom_path in dicom_iter:
            try:
                ds = read_dicom_header(dicom_path)
                laterality, view = infer_view_fields(ds)
                if not is_for_presentation(ds):
                    continue
                sop_instance_uid = str(ds.get("SOPInstanceUID", "")).strip()
                if not sop_instance_uid:
                    continue

                candidate_rows.append(
                    {
                        "patient_id": patient_id,
                        "exam_id": exam_id,
                        "sop_instance_uid": sop_instance_uid,
                        "laterality": laterality,
                        "view": view,
                        "dicom_path": str(dicom_path.resolve()),
                        "rows": int(ds.get("Rows", 0) or 0),
                        "cols": int(ds.get("Columns", 0) or 0),
                        "photometric_interpretation": str(
                            ds.get("PhotometricInterpretation", "")
                        ),
                        "bits_stored": int(
                            ds.get("BitsStored", ds.get("BitsAllocated", 16)) or 16
                        ),
                        "acquisition_time": get_tag(ds, (0x0008, 0x0032), ""),
                        "is_marked_up": is_marked_up(ds),
                        "has_implant": has_implant(ds),
                        "for_presentation": True,
                        "sha256": "",
                        "device_manufacturer": str(ds.get("Manufacturer", "")),
                        "device_model": str(ds.get("ManufacturerModelName", "")),
                        "study_date": get_tag(ds, (0x0008, 0x0020), ""),
                        "accession_number": accession_number
                        or get_tag(ds, (0x0008, 0x0050), ""),
                        "estimated_magnification_factor": estimate_magnification_factor(
                            ds.get("EstimatedRadiographicMagnificationFactor")
                        ),
                        "pixel_spacing_mm": estimate_pixel_spacing_mm(
                            ds.get("PixelSpacing")
                        ),
                    }
                )
                candidate_tags[sop_instance_uid] = {
                    "sop_instance_uid": sop_instance_uid,
                    "AcquisitionDeviceProcessingCode": str(
                        ds.get("AcquisitionDeviceProcessingCode", "")
                    ),
                    "DetectorType": str(ds.get("DetectorType", "")),
                }
            except Exception:
                continue

        if not candidate_rows:
            skipped_exams.append(exam_id)
            continue

        exam_df = pd.DataFrame(candidate_rows)
        exam_df["_view_selection_key"] = [
            view_selection_key(
                for_presentation=bool(for_presentation),
                estimated_magnification_factor=mag,
                pixel_spacing_mm=pixel_spacing_mm,
                dicom_path=dicom_path,
            )
            for for_presentation, mag, pixel_spacing_mm, dicom_path in zip(
                exam_df["for_presentation"],
                exam_df["estimated_magnification_factor"],
                exam_df["pixel_spacing_mm"],
                exam_df["dicom_path"],
            )
        ]
        exam_df = (
            exam_df.sort_values(
                by=["exam_id", "laterality", "view", "_view_selection_key"],
                kind="stable",
            )
            .drop_duplicates(subset=["exam_id", "laterality", "view"], keep="first")
            .drop(columns=["_view_selection_key"])
        )

        expected = {("L", "CC"), ("L", "MLO"), ("R", "CC"), ("R", "MLO")}
        found = {(r["laterality"], r["view"]) for _, r in exam_df.iterrows()}
        if found != expected:
            skipped_exams.append(exam_id)
            continue

        view_rows.extend(exam_df.to_dict("records"))
        for sop_instance_uid in exam_df["sop_instance_uid"]:
            tag_rows.append(candidate_tags[str(sop_instance_uid)])

    if missing_exam_dirs:
        sample = ", ".join(missing_exam_dirs[:5])
        raise FileNotFoundError(
            f"{len(missing_exam_dirs)} exam directories missing under {raw_dir}; sample: {sample}"
        )

    if not view_rows:
        raise RuntimeError(
            "no QC sidecars could be built from the supplied export/raw dirs"
        )

    views_df = pd.DataFrame(view_rows).reset_index(drop=True)
    tags_df = (
        pd.DataFrame(tag_rows)
        .drop_duplicates(subset=["sop_instance_uid"])
        .reset_index(drop=True)
    )

    if skipped_exams:
        print(
            f"warning: skipped {len(skipped_exams)} exams without a full presentation quad "
            f"(sample: {', '.join(skipped_exams[:5])})"
        )

    return views_df, tags_df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build minimal QC views/tags parquets from cached montages and DICOM headers."
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
        help="Raw DICOM root (used for fallback path layout or to locate sot/views.parquet)",
    )
    parser.add_argument(
        "--views",
        type=Path,
        default=None,
        help="views.parquet with dicom_path column (default: <raw-dir>/sot/views.parquet)",
    )
    args = parser.parse_args()

    export_dir = args.export_dir.resolve()
    raw_dir = args.raw_dir.resolve()
    views_path = (
        args.views.resolve() if args.views else raw_dir / "sot" / "views.parquet"
    )
    views_df, tags_df = build_sidecars(
        export_dir=export_dir,
        raw_dir=raw_dir,
        views_path=views_path,
    )

    views_output = export_dir / "views_for_qc.parquet"
    tags_output = export_dir / "dicom_tags.parquet"
    views_df.to_parquet(views_output, index=False)
    tags_df.to_parquet(tags_output, index=False)

    print(
        f"wrote {len(views_df):,} selected views from {views_df['exam_id'].nunique():,} exams "
        f"to {views_output}"
    )
    print(
        f"wrote {len(tags_df):,} minimal DICOM tag rows with columns "
        f"{list(tags_df.columns)} to {tags_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
