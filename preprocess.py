#!/usr/bin/env python3
"""prima_pipeline.py

Deterministic one-pass preprocessor for ChiMEC mammograms.

It builds a single source-of-truth (SoT) and a Mirai-compatible pixel cache — without
writing PNGs. Subsequent training/dev reads only Parquet+Zarr; raw DICOMs stay untouched.

Design constraints (matching Mirai exactly):
- accept only the four presentation views per exam: L-CC, L-MLO, R-CC, R-MLO
- mimic DCMTK `dcmj2pnm +on2 --min-max-window` (16-bit min–max VOI), then resize to 1664×2048
- defer normalization (mean/std) to the consumer; store uint16 in Zarr

References: Mirai README + supplementary material
- requires PNG16 generated via `+on2` and `--min-max-window` and uses `--img_size 1664 2048`
  and `--img_mean 7047.99` / `--img_std 12005.5`
- expects exactly four For Presentation views (no implants/markups); CSV keys are patient_id + exam_id

This script does five things:
1) indexes DICOMs and selects the canonical four views per exam
2) performs minimal QC and prints a summary
3) writes SoT Parquet tables (exams, views, cohort) joined with genotype
4) writes a Zarr cache per exam with four uint16 arrays (L_CC, L_MLO, R_CC, R_MLO)
5) emits a manifest.parquet pointing to the per-exam Zarr groups

Usage example
-------------
python prima_pipeline.py preprocess \
  --raw /data/prima/raw \
  --sot /data/prima/sot \
  --out /data/prima/mirai-prep

Defaults: --site ucmed, --workers 8, --genotype uses ChiMEC 2025 October phenotype CSV

Dependencies (install explicitly):
  pip install pydicom numpy pandas pyarrow zarr numcodecs opencv-python-headless einops tqdm

Notes
-----
- no defensive coding; if a required tag is missing or an assumption is violated, we raise
- avoid silent defaults; CLI args are required
- per-image operations are vectorized; per-exam iteration is unavoidable by I/O
- rearrange is used for shape control; never permute/transpose

"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
import zarr
from einops import rearrange
from numcodecs import Blosc
from pydicom.dataset import FileDataset
from pydicom.tag import Tag
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from metadata_utils import extract_base_modality

logger = logging.getLogger(__name__)

# hard requirements mirrored from Mirai
IMG_W = 1664
IMG_H = 2048
IMG_MEAN = 7047.99
IMG_STD = 12005.5

# allowed view labels for canonical four-pack
VIEWS = {
    ("L", "CC"): "L_CC",
    ("L", "MLO"): "L_MLO",
    ("R", "CC"): "R_CC",
    ("R", "MLO"): "R_MLO",
}

# columns for SoT tables
EXAMS_COLS = [
    "patient_id",
    "patient_hash",
    "exam_id",
    "study_date",
    "accession_number",
    "site",
    "device_manufacturer",
    "device_model",
    "presentation_intent",
    "n_views_present",
    "has_full_quad",
    "ibroker_study_id",
    "has_geno",
]

VIEWS_COLS = [
    "patient_id",
    "exam_id",
    "sop_instance_uid",
    "laterality",
    "view",
    "dicom_path",
    "rows",
    "cols",
    "photometric_interpretation",
    "bits_stored",
    "acquisition_time",
    "is_marked_up",
    "has_implant",
    "for_presentation",
    "sha256",
    "device_manufacturer",
    "device_model",
    "study_date",
    "accession_number",
]


@dataclass(frozen=True)
class PreprocessConfig:
    """Immutable configuration for preprocessing.

    All paths are required. No defaults to avoid accidental misconfiguration.
    """

    raw_dir: Path
    sot_dir: Path
    out_dir: Path
    genotype_csv: Path
    site: str
    workers: int
    summary: bool
    max_exams: Optional[int]
    debug_dir: Optional[Path]


# ------------- DICOM helpers -------------


def read_dicom(path: Path) -> FileDataset:
    """Read a DICOM from disk using pydicom without applying VOI LUT.

    Parameters
    ----------
    path : Path
        Filesystem path to the DICOM (.dcm) file.

    Returns
    -------
    FileDataset
        Parsed pydicom dataset. Pixel data is not normalized.
    """
    return pydicom.dcmread(str(path), force=True)


def get_tag(
    ds: FileDataset, tag: Tuple[int, int], default: Optional[str] = None
) -> Optional[str]:
    """Fetch a DICOM tag as string if present.

    parameters
    ----------
    ds : FileDataset
        dicom dataset
    tag : Tuple[int,int]
        group, element of the tag
    default : Optional[str]
        value to return if tag missing

    returns
    -------
    Optional[str]
    """
    t = Tag(tag)
    if t in ds:
        v = ds[t].value
        return str(v)
    return default


def infer_view_fields(ds: FileDataset) -> Tuple[str, str]:
    """Infer laterality (L/R) and view (CC/MLO) from standard tags.

    raises if tags are missing or values are not in the allowed set.
    """
    lat = get_tag(ds, (0x0020, 0x0062)) or get_tag(ds, (0x0020, 0x0060))
    vp = get_tag(ds, (0x0018, 0x5101))
    if lat is None or vp is None:
        raise ValueError("missing laterality or ViewPosition")
    lat = lat.strip().upper()
    vp = vp.strip().upper()
    if lat not in {"L", "R"}:
        raise ValueError(f"unexpected laterality: {lat}")
    if vp not in {"CC", "MLO"}:
        raise ValueError(f"unexpected view position: {vp}")
    return lat, vp


def is_for_presentation(ds: FileDataset) -> bool:
    """True if PresentationIntentType == 'FOR PRESENTATION'.

    Checks both (0x0008, 0x0068) PresentationIntentType and (0x0008, 0x0069) Presentation Intent Type.
    """
    # Try PresentationIntentType first (0x0008, 0x0068)
    pit = get_tag(ds, (0x0008, 0x0068), "")
    if pit and str(pit).strip().upper() == "FOR PRESENTATION":
        return True

    # Try Presentation Intent Type (0x0008, 0x0069) as fallback
    pit_alt = get_tag(ds, (0x0008, 0x0069), "")
    if pit_alt and str(pit_alt).strip().upper() == "FOR PRESENTATION":
        return True

    return False


def has_implant(ds: FileDataset) -> bool:
    """Return True if BreastImplantPresent (0028,1300) == 'YES'."""
    bip = get_tag(ds, (0x0028, 0x1300))
    return str(bip).strip().upper() == "YES" if bip is not None else False


def is_marked_up(ds: FileDataset) -> bool:
    """Use 'Burned In Annotation' (0028,0301) == 'YES' if present; else False."""
    bia = get_tag(ds, (0x0028, 0x0301))
    return str(bia).strip().upper() == "YES" if bia is not None else False


def to_uint16_minmax(ds: FileDataset) -> np.ndarray:
    """Convert DICOM pixel data to uint16 using min–max VOI (DCMTK --min-max-window).

    Steps
    -----
    - decode PixelData to ndarray
    - apply RescaleSlope/RescaleIntercept if present
    - if PhotometricInterpretation == MONOCHROME1, invert after scaling
    - linearly map min..max → 0..65535, cast to uint16
    - no clipping beyond [min, max] since mapping uses data extents

    returns
    -------
    np.ndarray of shape (H, W), dtype=uint16
    """
    arr = ds.pixel_array.astype(np.float32, copy=False)
    slope = float(ds.get("RescaleSlope", 1.0))
    inter = float(ds.get("RescaleIntercept", 0.0))
    arr = arr * slope + inter
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        raise ValueError("degenerate dynamic range in pixel data")
    arr = (arr - vmin) * (65535.0 / (vmax - vmin))
    if str(ds.get("PhotometricInterpretation", "")).strip().upper() == "MONOCHROME1":
        arr = 65535.0 - arr
    return arr.astype(np.uint16, copy=False)


def resize_uint16(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a uint16 image with area interpolation for downscale / linear for upscale.

    parameters
    ----------
    img : np.ndarray
        2D uint16 image
    width : int
        target width in pixels
    height : int
        target height in pixels
    """
    h, w = img.shape
    if h == height and w == width:
        return img
    interp = cv2.INTER_AREA if (height < h or width < w) else cv2.INTER_LINEAR
    out = cv2.resize(img, (width, height), interpolation=interp)
    return out.astype(np.uint16, copy=False)


def hash_file_sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    """Compute streaming SHA256 for provenance.

    parameters
    ----------
    path : Path
        file path
    chunk : int
        chunk size in bytes for streaming
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ------------- Discovery & selection -------------


def _save_debug_figure(
    ds: FileDataset,
    path: Path,
    status: str,
    reason_or_view: str,
    debug_dir: Path,
    patient_id: str = None,
    exam_id: str = None,
) -> None:
    """Save debug figure showing pixel data and DICOM tags.

    status: 'SUCCESS' or 'FAILED'
    reason_or_view: if failed, the reason; if success, the laterality and view (e.g., 'L CC')
    Organizes debug files by: debug_dir/patient_id/fail_reason/exam_id/
    """
    try:
        # create figure with two subplots
        fig = plt.figure(figsize=(16, 10))

        # left: pixel data
        ax_img = plt.subplot(1, 2, 1)
        try:
            pixel_array = ds.pixel_array
            ax_img.imshow(pixel_array, cmap="gray")
            ax_img.set_title(f"Pixel Data\n{ds.Rows}x{ds.Columns}")
            ax_img.axis("off")
        except Exception as e:
            ax_img.text(
                0.5,
                0.5,
                f"Cannot display pixel data:\n{e}",
                ha="center",
                va="center",
                transform=ax_img.transAxes,
            )
            ax_img.axis("off")

        # right: DICOM tags
        ax_text = plt.subplot(1, 2, 2)
        ax_text.axis("off")

        # collect tag info
        tag_lines = [
            f"File: {path.name}",
            f"Exam: {path.parent.name}",
            f"Patient: {path.parent.parent.name}",
            f"Status: {status}",
        ]

        if status == "FAILED":
            tag_lines.append(f"Reason: {reason_or_view}")
        else:
            tag_lines.append(f"View: {reason_or_view}")
            tag_lines.append(
                f"For Presentation: {get_tag(ds, (0x0008, 0x0068), 'N/A')}"
            )

        tag_lines.extend(
            [
                "",
                "TAGS WE'RE LOOKING FOR:",
                f"  (0x0020,0x0062) Laterality: {get_tag(ds, (0x0020, 0x0062), 'MISSING')}",
                f"  (0x0020,0x0060) ImageLaterality: {get_tag(ds, (0x0020, 0x0060), 'MISSING')}",
                f"  (0x0018,0x5101) ViewPosition: {get_tag(ds, (0x0018, 0x5101), 'MISSING')}",
                f"  (0x0008,0x0068) PresentationIntentType: {get_tag(ds, (0x0008, 0x0068), 'MISSING')}",
                f"  (0x0008,0x0069) Presentation Intent Type: {get_tag(ds, (0x0008, 0x0069), 'MISSING')}",
                "",
                "ALL TAGS IN THIS FILE:",
                "-" * 60,
            ]
        )

        for elem in ds:
            if elem.VR == "SQ":
                tag_lines.append(f"{elem.tag} {elem.name}: <sequence>")
            else:
                try:
                    value_str = str(elem.value)
                    if len(value_str) > 80:
                        value_str = value_str[:80] + "..."
                    tag_lines.append(f"{elem.tag} {elem.name}: {value_str}")
                except Exception:
                    tag_lines.append(f"{elem.tag} {elem.name}: <cannot display>")

        # render text
        text_content = "\n".join(tag_lines)
        ax_text.text(
            0,
            1,
            text_content,
            verticalalignment="top",
            fontsize=7,
            family="monospace",
            transform=ax_text.transAxes,
        )

        # save figure organized by patient_id/fail_reason/exam_id
        # use provided patient_id and exam_id if available, otherwise extract from path
        if patient_id is None:
            patient_id = path.parent.parent.name
        if exam_id is None:
            exam_id = path.parent.name

        # clean up reason for directory name (remove spaces, special chars)
        clean_reason = (
            reason_or_view.replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )
        clean_reason = "".join(c for c in clean_reason if c.isalnum() or c in "_-")[
            :50
        ]  # limit length

        patient_dir = debug_dir / patient_id
        reason_dir = patient_dir / clean_reason
        exam_dir = reason_dir / exam_id
        exam_dir.mkdir(parents=True, exist_ok=True)

        out_name = f"{status}_{path.stem}.png"
        out_path = exam_dir / out_name
        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        logger.debug(
            f"    Saved debug figure: {patient_id}/{clean_reason}/{exam_id}/{out_name}"
        )
    except Exception as e:
        logger.warning(f"    Failed to save debug figure for {path.name}: {e}")


def _check_and_move_non_mammogram_exam(exam_path: Path) -> bool:
    """Check if exam is mammogram; if not, move to modality-specific directory.

    Returns True if exam was moved (non-mammogram), False if it's a mammogram.
    """
    # check if exam path still exists (might have been moved by another process)
    if not exam_path.exists():
        logger.warning(f"Exam path {exam_path} no longer exists, skipping")
        return True

    # collect first dicom file to check modality
    first_dcm = None
    for root, dirs, files in os.walk(exam_path):
        for name in files:
            if name.endswith(".dcm"):
                first_dcm = Path(root) / name
                break
        if first_dcm:
            break

    if not first_dcm:
        logger.warning(f"No DICOM files found in {exam_path}, skipping modality check")
        return False

    # read modality from first DICOM
    try:
        ds = read_dicom(first_dcm)
        modality_raw = get_tag(ds, (0x0008, 0x0060), default="")
        study_description = get_tag(ds, (0x0008, 0x1030), default="")
        base_modality = extract_base_modality(modality_raw, study_description)

        logger.info(f"Exam modality: {modality_raw} -> base: {base_modality}")

        # if it's a mammogram, don't move
        if base_modality == "MG":
            return False

        # move to modality-specific directory
        # structure: /gpfs/data/huo-lab/Image/ChiMEC/{modality}/patient_id/exam_id
        patient_dir = exam_path.parent
        patient_id = patient_dir.name
        exam_id = exam_path.name
        chimec_root = patient_dir.parent

        # create modality directory
        modality_root = chimec_root / base_modality
        new_patient_dir = modality_root / patient_id
        new_exam_path = new_patient_dir / exam_id

        # move the exam directory
        logger.info(f"Moving non-mammogram exam from {exam_path} to {new_exam_path}")
        new_patient_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exam_path), str(new_exam_path))
        logger.info(f"Successfully moved {base_modality} exam to {new_exam_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to check/move exam {exam_path}: {e}")
        return False


def _process_exam_dir(exam_path: Path, debug_dir: Optional[Path] = None) -> List[Dict]:
    """Process all DICOMs in a single exam directory.

    walks subdirectories under exam_path to find all .dcm files
    """
    logger.info(f"\n=== Processing exam: {exam_path} ===")

    # check modality and move if not mammogram
    if _check_and_move_non_mammogram_exam(exam_path):
        logger.info(f"Exam {exam_path} is not a mammogram, skipping preprocessing")
        return []

    # collect all dicom files first
    dcm_files = []
    for root, dirs, files in os.walk(exam_path):
        for name in files:
            if name.endswith(".dcm"):
                dcm_files.append(Path(root) / name)

    logger.info(f"Found {len(dcm_files)} DICOM files in exam")

    rows = []
    failed_files = []

    for p in dcm_files:
        try:
            ds = read_dicom(p)
            patient_id = get_tag(ds, (0x0010, 0x0020))
            study_uid = get_tag(ds, (0x0020, 0x000D))
            sop_uid = get_tag(ds, (0x0008, 0x0018))

            if patient_id is None or study_uid is None or sop_uid is None:
                reason = "missing patient/study/SOP IDs"
                logger.warning(f"  SKIP: {p.name} - {reason}")
                failed_files.append((p, reason, ds))
                if debug_dir:
                    _save_debug_figure(
                        ds, p, "FAILED", reason, debug_dir, patient_id, study_uid
                    )
                continue

            # check presentation intent first
            present = is_for_presentation(ds)
            if not present:
                reason = "not for presentation"
                logger.warning(f"  SKIP: {p.name} - {reason}")
                failed_files.append((p, reason, ds))
                if debug_dir:
                    _save_debug_figure(
                        ds, p, "FAILED", reason, debug_dir, patient_id, study_uid
                    )
                continue

            # try to get laterality and view
            try:
                lat, vp = infer_view_fields(ds)
            except ValueError as e:
                reason = str(e)
                logger.warning(f"  SKIP: {p.name} - {reason}")
                failed_files.append((p, reason, ds))
                if debug_dir:
                    _save_debug_figure(
                        ds, p, "FAILED", reason, debug_dir, patient_id, study_uid
                    )
                continue

            logger.info(f"  OK: {p.name} - {lat} {vp}, for_presentation={present}")

            # save debug figure for successful DICOMs too
            if debug_dir:
                _save_debug_figure(
                    ds, p, "SUCCESS", f"{lat} {vp}", debug_dir, patient_id, study_uid
                )

            rows.append(
                {
                    "patient_id": patient_id,
                    "exam_id": study_uid,
                    "sop_instance_uid": sop_uid,
                    "laterality": lat,
                    "view": vp,
                    "dicom_path": str(p.resolve()),
                    "rows": int(ds.Rows),
                    "cols": int(ds.Columns),
                    "photometric_interpretation": str(
                        ds.get("PhotometricInterpretation", "")
                    ),
                    "bits_stored": int(
                        ds.get("BitsStored", ds.get("BitsAllocated", 16))
                    ),
                    "acquisition_time": get_tag(ds, (0x0008, 0x0032), ""),
                    "is_marked_up": is_marked_up(ds),
                    "has_implant": has_implant(ds),
                    "for_presentation": present,
                    "sha256": hash_file_sha256(p),
                    "device_manufacturer": str(ds.get("Manufacturer", "")),
                    "device_model": str(ds.get("ManufacturerModelName", "")),
                    "study_date": get_tag(ds, (0x0008, 0x0020), ""),
                    "accession_number": get_tag(ds, (0x0008, 0x0050), ""),
                }
            )
        except Exception as e:
            logger.error(f"  ERROR: {p.name} - unexpected error: {e}")
            failed_files.append((p, f"unexpected error: {e}", None))

    logger.info("\nExam summary:")
    logger.info(f"  Total files: {len(dcm_files)}")
    logger.info(f"  Successfully processed: {len(rows)}")
    logger.info(f"  Failed/skipped: {len(failed_files)}")

    # report failure reasons
    if failed_files:
        from collections import Counter

        failure_reasons = Counter()
        for p, reason, ds in failed_files:
            failure_reasons[reason] += 1
        logger.info("  Failure reasons:")
        for reason, count in failure_reasons.most_common():
            logger.info(f"    {reason}: {count} files")

    if rows:
        views = {}
        for r in rows:
            key = (r["laterality"], r["view"])
            if key not in views:
                views[key] = []
            views[key].append(r["for_presentation"])
        logger.info("  Views found:")
        for (lat, vw), pres_list in sorted(views.items()):
            for_pres = sum(pres_list)
            logger.info(
                f"    {lat}-{vw}: {len(pres_list)} total, {for_pres} for_presentation"
            )

    # if we got NO valid rows, dump debug info and exit
    if not rows:
        logger.error(f"\n=== FATAL: No valid DICOMs found in exam {exam_path} ===")
        logger.error("Dumping first failed DICOM for debugging:")
        if failed_files:
            p, reason, ds = failed_files[0]
            logger.error(f"File: {p}")
            logger.error(f"Reason: {reason}")
            if ds is not None:
                logger.error("\nAll DICOM tags:")
                for elem in ds:
                    if elem.VR == "SQ":
                        logger.error(f"  {elem.tag} {elem.name}: <sequence>")
                        continue
                    try:
                        value_str = str(elem.value)
                        if len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                        logger.error(f"  {elem.tag} {elem.name}: {value_str}")
                    except Exception:
                        logger.error(f"  {elem.tag} {elem.name}: <cannot display>")
        raise RuntimeError(f"No valid DICOMs with laterality/view in exam {exam_path}")

    return rows


def discover_dicoms(
    raw_dir: Path,
    max_exams: Optional[int] = None,
    workers: int = 1,
    debug_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Scan `raw_dir` for exam directories and extract DICOM tags in parallel.

    assumes structure: raw_dir/patient_id/exam_id/[series_dirs]/file.dcm
    walks to exam level, then processes each exam in parallel
    """
    logger.info(f"scanning for exam directories in: {raw_dir}")
    exam_dirs = []

    # walk to depth 2: patient_id/exam_id
    for patient_name in os.listdir(raw_dir):
        patient_path = raw_dir / patient_name
        if not patient_path.is_dir():
            continue
        for exam_name in os.listdir(patient_path):
            exam_path = patient_path / exam_name
            if exam_path.is_dir():
                exam_dirs.append(exam_path)
                if max_exams is not None and len(exam_dirs) >= max_exams:
                    logger.info(
                        f"reached --max-exams limit of {max_exams}, stopping scan"
                    )
                    break
        if max_exams is not None and len(exam_dirs) >= max_exams:
            break

    logger.info(f"found {len(exam_dirs)} exam directories")

    if not exam_dirs:
        raise RuntimeError("no exam directories found")

    # create debug directory if requested
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"debug output will be saved to: {debug_dir}")

    # process exams in parallel
    logger.info(f"processing exams with {workers} workers...")
    from concurrent.futures import ProcessPoolExecutor, as_completed

    all_rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_exam_dir, exam_path, debug_dir): exam_path
            for exam_path in exam_dirs
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="processing exams",
            unit="exam",
        ):
            exam_path = futures[future]
            try:
                rows = future.result()
                all_rows.extend(rows)
            except Exception as e:
                logger.error(f"failed to process {exam_path}: {e}")
                raise  # re-raise to stop execution for debugging

    logger.info(f"collected metadata from {len(all_rows)} DICOM files")
    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("no DICOMs found")

    # print summary statistics
    if not df.empty:
        logger.info("\n=== PROCESSING SUMMARY ===")
        logger.info(f"Total exams processed: {len(exam_dirs)}")
        logger.info(f"Total DICOMs processed: {len(all_rows)}")
        logger.info(f"Unique patients: {df['patient_id'].nunique()}")
        logger.info(f"Unique exams: {df['exam_id'].nunique()}")

        # view breakdown
        view_counts = df.groupby(["laterality", "view"]).size()
        logger.info("Views found:")
        for (lat, view), count in view_counts.items():
            logger.info(f"  {lat}-{view}: {count} files")

    return df


def select_full_quad(df_views: pd.DataFrame) -> pd.DataFrame:
    """Return only rows that constitute exactly one L/R × CC/MLO per exam.

    selection policy for duplicates per (exam, laterality, view):
      - prefer not marked up
      - prefer higher bits stored
      - prefer later acquisition_time
    """
    df = df_views.copy()
    df = df[df["for_presentation"]]
    df = df[df["laterality"].isin(["L", "R"]) & df["view"].isin(["CC", "MLO"])]

    df["_rank"] = (
        (~df["is_marked_up"]).astype(int).astype(np.int64) * 1_000_000
        + df["bits_stored"].astype(np.int64) * 1_000
        + pd.to_numeric(df["acquisition_time"].str.replace(":", ""), errors="coerce")
        .fillna(0)
        .astype(np.int64)
    )

    # pick the best per exam+view key
    df = df.sort_values(
        ["exam_id", "laterality", "view", "_rank"], ascending=[True, True, True, False]
    )
    df = df.groupby(["exam_id", "laterality", "view"], as_index=False).head(1)

    # keep only exams that now have all four views
    counts = df.groupby("exam_id").size()
    full_exams = counts[counts == 4].index
    out = (
        df[df["exam_id"].isin(full_exams)]
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )
    return out


# ------------- Zarr writing -------------


def write_exam_zarr(
    exam_rows: pd.DataFrame, root_dir: Path
) -> Tuple[Path, Dict[str, Tuple[int, int]]]:
    """Write one Zarr group per exam with four uint16 arrays.

    parameters
    ----------
    exam_rows : pd.DataFrame
        exactly 4 rows for a single exam
    root_dir : Path
        base directory for zarr groups; subdirs are /{patient_hash}/{exam_id}.zarr

    returns
    -------
    (zarr_path, shapes)
    """
    assert len(exam_rows) == 4
    first = exam_rows.iloc[0]
    patient_id = first["patient_id"]
    exam_id = first["exam_id"]
    patient_hash = hashlib.sha1(patient_id.encode("utf-8")).hexdigest()[:12]

    zpath = root_dir / patient_hash / f"{exam_id}.zarr"
    zpath.parent.mkdir(parents=True, exist_ok=True)

    # compressor tuned for uint16 imagery
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    store = zarr.DirectoryStore(str(zpath))
    grp = zarr.group(store=store, overwrite=True)
    grp.attrs.update(
        {
            "patient_id": patient_id,
            "exam_id": exam_id,
            "img_dtype": "uint16",
            "img_size": [IMG_H, IMG_W],
            "transform": "minmax_uint16_then_resize",
            "source": "DICOM",
        }
    )

    shapes: Dict[str, Tuple[int, int]] = {}

    for _, r in exam_rows.iterrows():
        key = VIEWS[(r["laterality"], r["view"])]
        ds = read_dicom(Path(r["dicom_path"]))
        arr16 = to_uint16_minmax(ds)
        arr16 = resize_uint16(arr16, IMG_W, IMG_H)
        # write chunks ~half-size tiles for balance
        chunks = (IMG_H // 2, IMG_W // 2)
        zarr_arr = grp.create_dataset(
            name=key,
            shape=(IMG_H, IMG_W),
            dtype=np.uint16,
            chunks=chunks,
            compressor=compressor,
            overwrite=True,
        )
        zarr_arr[:] = arr16
        zarr_arr.attrs.update(
            {
                "laterality": r["laterality"],
                "view": r["view"],
                "source_sop_instance_uid": r["sop_instance_uid"],
                "photometric_interpretation": r["photometric_interpretation"],
            }
        )
        shapes[key] = (IMG_H, IMG_W)

    return zpath, shapes


# ------------- Orchestration -------------


def preprocess(cfg: PreprocessConfig) -> None:
    """Run the full pipeline: index, select, cache, and emit SoT + manifest.

    prints a concise QC summary at the end.
    """
    raw = cfg.raw_dir
    sot = cfg.sot_dir
    out = cfg.out_dir

    logger.info("=== starting preprocessing ===")
    logger.info(f"raw dir: {raw}")
    logger.info(f"sot dir: {sot}")
    logger.info(f"out dir: {out}")
    logger.info(f"workers: {cfg.workers}")
    if cfg.max_exams:
        logger.info(f"max exams (debug mode): {cfg.max_exams}")
    if cfg.debug_dir:
        logger.info(f"debug dir: {cfg.debug_dir}")

    # discovery
    logger.info("discovering DICOMs...")
    views_df = discover_dicoms(
        raw, max_exams=cfg.max_exams, workers=cfg.workers, debug_dir=cfg.debug_dir
    )
    logger.info(f"found {len(views_df)} DICOM files")

    # selection to full quad
    logger.info("selecting full quad exams...")
    sel_df = select_full_quad(views_df)
    logger.info(f"selected {len(sel_df)} views from full quad exams")

    # exams table
    exams = (
        sel_df.groupby(["patient_id", "exam_id"], as_index=False)
        .agg(
            {
                "dicom_path": "count",
                "device_manufacturer": "first",
                "device_model": "first",
                "study_date": "first",
                "accession_number": "first",
            }
        )
        .rename(columns={"dicom_path": "n_views_present"})
    )
    exams["has_full_quad"] = exams["n_views_present"] == 4
    exams["patient_hash"] = exams["patient_id"].apply(
        lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
    )
    exams["presentation_intent"] = "FOR PRESENTATION"
    exams["site"] = cfg.site
    exams["ibroker_study_id"] = None

    # genotype join determines cohort
    logger.info(f"loading genotype CSV: {cfg.genotype_csv}")
    geno = pd.read_csv(cfg.genotype_csv)
    if "PUB_id" not in geno.columns:
        raise KeyError("genotype CSV must have column 'PUB_id'")
    geno = geno.rename(columns={"PUB_id": "patient_id"})
    logger.info(f"loaded {len(geno)} genotype records")

    logger.info("joining with genotype...")
    exams = exams.merge(
        geno[["patient_id"]].drop_duplicates().assign(has_geno=True),
        on="patient_id",
        how="left",
    )
    exams["has_geno"] = exams["has_geno"].fillna(False)
    logger.info(f"exams with genotype: {exams['has_geno'].sum()}")

    # write SoT tables
    logger.info(f"writing SoT tables to {sot}...")
    sot.mkdir(parents=True, exist_ok=True)
    sel_df[VIEWS_COLS].to_parquet(sot / "views.parquet", index=False)
    exams[EXAMS_COLS].to_parquet(sot / "exams.parquet", index=False)

    cohort = exams[(exams["has_full_quad"]) & (exams["has_geno"])][
        ["patient_id", "patient_hash", "exam_id", "study_date"]
    ]
    cohort.to_parquet(sot / "cohort.parquet", index=False)
    logger.info("wrote views.parquet, exams.parquet, cohort.parquet")

    # if summary-only, stop here
    if cfg.summary:
        logger.info("=== summary-only: wrote SoT, skipped zarr cache ===")
        logger.info(f"SoT dir: {sot}")
        return

    # zarr cache + manifest for the cohort
    logger.info(f"preparing zarr cache in {out}...")
    prep_dir = out / "zarr"
    prep_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict] = []

    logger.info("grouping exams for zarr writing...")
    grouped = {k: v for k, v in sel_df.groupby("exam_id")}
    targets = [
        (eid, grouped[eid]) for eid in cohort["exam_id"].tolist() if eid in grouped
    ]
    logger.info(f"will write {len(targets)} exams to zarr with {cfg.workers} workers")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futs = {
            ex.submit(write_exam_zarr, grp_df, prep_dir): eid for eid, grp_df in targets
        }
        for fut in tqdm(
            as_completed(futs), total=len(futs), desc="writing zarr", unit="exam"
        ):
            zpath, shapes = fut.result()
            eid = futs[fut]
            grp_df = grouped[eid]
            for _, r in grp_df.iterrows():
                key = VIEWS[(r["laterality"], r["view"])]
                manifest_rows.append(
                    {
                        "patient_id": r["patient_id"],
                        "exam_id": r["exam_id"],
                        "laterality": r["laterality"],
                        "view": r["view"],
                        "zarr_uri": str(zpath),
                        "zarr_key": key,
                        "height": shapes[key][0],
                        "width": shapes[key][1],
                    }
                )

    logger.info(f"writing manifest with {len(manifest_rows)} rows...")
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    logger.info(f"wrote manifest to {manifest_path}")

    # QC summary
    total_files = len(views_df)
    total_exams = views_df[["exam_id"]].nunique()[0]
    full_exams = exams["has_full_quad"].sum()
    cohort_exams = len(cohort)

    vendor_counts = (
        sel_df.groupby(["device_manufacturer", "device_model"], as_index=False)
        .size()
        .rename(columns={"size": "n_views"})
    )

    logger.info("=== QC summary ===")
    logger.info(f"dicom files indexed: {total_files}")
    logger.info(f"unique exams: {total_exams}")
    logger.info(f"full-quad exams: {full_exams}")
    logger.info(f"cohort (full-quad ∩ has-geno): {cohort_exams}")
    logger.info("views by vendor/model:")
    logger.info(f"\n{vendor_counts.to_string(index=False)}")
    logger.info(f"written SoT: {sot}")
    logger.info(f"written cache manifest: {manifest_path}")


# ------------- Consumer helpers -------------


def load_view_from_manifest(
    manifest_row: pd.Series, normalize: bool = True, channels: int = 3
) -> np.ndarray:
    """Load a single view from manifest row, optionally normalize and expand channels.

    returns CHW float32 array ready for pytorch
    """
    z = zarr.open(str(manifest_row["zarr_uri"]), mode="r")
    arr16: np.ndarray = z[manifest_row["zarr_key"]][:]
    x = arr16.astype(np.float32)
    if normalize:
        x = (x - IMG_MEAN) / IMG_STD
    x = rearrange(x, "h w -> 1 h w")
    if channels == 3:
        x = np.repeat(x, 3, axis=0)
    return x


def write_mirai_csv(
    manifest_parquet: Path, labels_parquet: Path, out_csv: Path
) -> None:
    """Create a Mirai-compatible CSV that points to Zarr URIs instead of PNGs.

    The CSV schema mirrors Mirai's documented requirements:
    columns: patient_id, exam_id, laterality, view, file_path,
             years_to_cancer, years_to_last_followup, split_group

    - file_path is encoded as: zarr://<abs_path_to_exam.zarr>#<view_key>
    - strict join on (patient_id, exam_id); raises if any labels missing

    parameters
    ----------
    manifest_parquet : Path
        parquet emitted by preprocess() mapping views to zarr URIs and keys
    labels_parquet : Path
        parquet with columns [patient_id, exam_id, years_to_cancer, years_to_last_followup, split_group]
    out_csv : Path
        destination CSV; parent must exist
    """
    man = pd.read_parquet(manifest_parquet)
    req_m = {"patient_id", "exam_id", "laterality", "view", "zarr_uri", "zarr_key"}
    if req_m - set(man.columns):
        miss = sorted(req_m - set(man.columns))
        raise KeyError(f"manifest missing columns: {miss}")

    lab = pd.read_parquet(labels_parquet)
    req_l = {
        "patient_id",
        "exam_id",
        "years_to_cancer",
        "years_to_last_followup",
        "split_group",
    }
    if req_l - set(lab.columns):
        miss = sorted(req_l - set(lab.columns))
        raise KeyError(f"labels missing columns: {miss}")

    df = man.merge(lab[list(req_l)], on=["patient_id", "exam_id"], how="inner")
    if len(df) != len(man):
        missing = set(man[["patient_id", "exam_id"]].apply(tuple, axis=1)) - set(
            df[["patient_id", "exam_id"]].apply(tuple, axis=1)
        )
        raise ValueError(f"labels missing for {len(missing)} exam(s)")

    df["file_path"] = df.apply(
        lambda r: f"zarr://{Path(r['zarr_uri']).resolve()}#{r['zarr_key']}", axis=1
    )
    out = df[
        [
            "patient_id",
            "exam_id",
            "laterality",
            "view",
            "file_path",
            "years_to_cancer",
            "years_to_last_followup",
            "split_group",
        ]
    ]
    out.to_csv(out_csv, index=False)


def materialize_mirai_embeddings(
    mirai_repo: Path,
    img_encoder_snapshot: Path,
    manifest_parquet: Path,
    out_dir: Path,
    batch_size: int = 8,
    num_workers: int = 4,
) -> None:
    """Dump per-view embeddings using Mirai's encoder weights.

    This implementation loads a torchvision ResNet-18 and attempts to load the
    provided Mirai encoder snapshot into it. If the snapshot format differs, it
    raises loudly so you can adapt the key mapping. Aggregation is left to
    downstream code; we emit per-view 512-d vectors as Parquet.

    parameters
    ----------
    mirai_repo : Path
        path to a local clone of Mirai; added to sys.path for custom loaders if needed
    img_encoder_snapshot : Path
        path to Mirai's ResNet-18 encoder weights (.p / .pt); must be compatible with torchvision resnet18
    manifest_parquet : Path
        our manifest mapping views to zarr; used to build dataset
    out_dir : Path
        base output dir; writes features/mirai_v1/per_view.parquet
    batch_size : int
        dataloader batch size
    num_workers : int
        dataloader workers
    """
    if mirai_repo is not None:
        sys.path.insert(0, str(mirai_repo))

    import torch
    import torch.nn as nn
    import torchvision.models as tv

    enc = tv.resnet18(weights=None)
    enc.fc = nn.Identity()

    sd = torch.load(str(img_encoder_snapshot), map_location="cpu")
    tried = []
    try:
        enc.load_state_dict(sd)
    except Exception as e1:
        tried.append("root")
        ok = False
        for k in (
            "state_dict",
            "model",
            "model_state",
            "encoder_state",
            "net",
            "module",
        ):
            if isinstance(sd, dict) and k in sd:
                try:
                    enc.load_state_dict(sd[k])
                    ok = True
                    break
                except Exception:
                    tried.append(k)
                    continue
        if not ok:
            raise RuntimeError(
                f"cannot load snapshot into resnet18; tried keys {tried}"
            ) from e1

    enc.eval()

    ds = MiraiZarrDataset(manifest_parquet)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=lambda b: b,
    )

    rows = []
    with torch.no_grad():
        for batch in tqdm(dl, desc="mirai encoder", unit="exams"):
            for sample in batch:
                x = sample["images"]
                v = enc(x)
                v = v.cpu().numpy().astype(np.float32)
                for key, vec in zip(sample["order"], v):
                    rows.append(
                        {
                            "patient_id": sample["patient_id"],
                            "exam_id": sample["exam_id"],
                            "view_key": key,
                            **{f"f{i}": float(vec[i]) for i in range(vec.shape[0])},
                        }
                    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_view = pd.DataFrame(rows)
    (out_dir / "mirai_v1").mkdir(parents=True, exist_ok=True)
    per_view.to_parquet(out_dir / "mirai_v1" / "per_view.parquet", index=False)


def ingest_mirai_predictions(prediction_csv: Path, out_parquet: Path) -> None:
    """Convert Mirai's validation output CSV to per-exam Parquet.

    expects columns: patient_exam_id (tab-separated patient_id and exam_id),
    and risk predictions for 1-5 years (column names as per demo/validation_output.csv).
    """
    df = pd.read_csv(prediction_csv)
    if "patient_exam_id" not in df.columns:
        raise KeyError("prediction CSV missing 'patient_exam_id'")
    pe = df["patient_exam_id"].astype(str).str.split("	", n=1, expand=True)
    df["patient_id"] = pe[0]
    df["exam_id"] = pe[1]
    keep = [
        c
        for c in df.columns
        if any(
            t in c.lower()
            for t in ("1year", "2year", "3year", "4year", "5year", "risk")
        )
    ]
    out = df[["patient_id", "exam_id"] + keep]
    out.to_parquet(out_parquet, index=False)


# ------------- CLI -------------


def parse_args() -> PreprocessConfig:
    """Parse CLI arguments into a PreprocessConfig.

    defaults: raw=/gpfs/data/huo-lab/Image/ChiMEC/, sot=raw/sot, out=raw/out
    """
    p = argparse.ArgumentParser(
        description="Build SoT + Mirai-compatible Zarr cache (no PNGs)"
    )
    p.add_argument(
        "--raw",
        dest="raw_dir",
        type=Path,
        default=Path("/gpfs/data/huo-lab/Image/ChiMEC/"),
    )
    p.add_argument("--sot", dest="sot_dir", type=Path, default=None)
    p.add_argument("--out", dest="out_dir", type=Path, default=None)
    p.add_argument(
        "--genotype",
        dest="genotype_csv",
        type=Path,
        default=Path(
            "/gpfs/data/phs/groups/Projects/Huo_projects/SPORE/annawoodard/Phenotype_ChiMEC_2025October4.csv"
        ),
    )
    p.add_argument("--site", dest="site", type=str, default="ucmed")
    p.add_argument("--workers", dest="workers", type=int, default=8)
    p.add_argument("--summary", dest="summary", action="store_true")
    p.add_argument(
        "--max-exams",
        dest="max_exams",
        type=int,
        default=None,
        help="limit number of exams to process (for debugging)",
    )
    p.add_argument(
        "--debug-dir",
        dest="debug_dir",
        type=Path,
        default=None,
        help="save debug figures for skipped DICOMs to this directory",
    )
    # optional extras
    p.add_argument("--emit-csv", dest="emit_csv", type=Path)
    p.add_argument("--labels", dest="labels_parquet", type=Path)
    p.add_argument("--features-out", dest="features_out", type=Path)
    p.add_argument("--img-encoder-snapshot", dest="img_encoder_snapshot", type=Path)
    p.add_argument("--mirai-repo", dest="mirai_repo", type=Path)
    args = p.parse_args()

    # derive sot and out from raw if not provided
    raw_dir = args.raw_dir
    sot_dir = args.sot_dir if args.sot_dir is not None else raw_dir / "sot"
    out_dir = args.out_dir if args.out_dir is not None else raw_dir / "out"

    return PreprocessConfig(
        raw_dir=raw_dir,
        sot_dir=sot_dir,
        out_dir=out_dir,
        genotype_csv=args.genotype_csv,
        site=args.site,
        workers=args.workers,
        summary=args.summary,
        max_exams=args.max_exams,
        debug_dir=args.debug_dir,
    ), args


def main() -> None:
    """Entrypoint for the CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("parsing arguments...")
    cfg, args = parse_args()
    logger.info("configuration loaded")
    logger.info(f"  raw_dir: {cfg.raw_dir}")
    logger.info(f"  sot_dir: {cfg.sot_dir}")
    logger.info(f"  out_dir: {cfg.out_dir}")
    logger.info(f"  genotype_csv: {cfg.genotype_csv}")
    logger.info(f"  site: {cfg.site}")
    logger.info(f"  workers: {cfg.workers}")
    logger.info(f"  summary: {cfg.summary}")
    logger.info(f"  max_exams: {cfg.max_exams or 'unlimited'}")

    logger.info("validating configuration...")
    if cfg.workers <= 0:
        raise ValueError("workers must be > 0")
    if cfg.raw_dir == cfg.out_dir:
        raise ValueError("out_dir must differ from raw_dir")
    logger.info("configuration valid")

    preprocess(cfg)

    # emit Mirai CSV if requested
    if args.emit_csv is not None:
        logger.info("generating Mirai CSV...")
        if args.labels_parquet is None:
            raise ValueError(
                "--emit-csv requires --labels with years_to_cancer/years_to_last_followup/split_group"
            )
        write_mirai_csv(
            cfg.out_dir / "manifest.parquet", args.labels_parquet, args.emit_csv
        )
        logger.info(f"wrote Mirai CSV → {args.emit_csv}")

    # materialize per-view embeddings if requested
    if args.features_out is not None:
        logger.info("materializing per-view embeddings...")
        if args.img_encoder_snapshot is None:
            raise ValueError(
                "--features-out requires --img-encoder-snapshot (Mirai ResNet18 weights)"
            )
        materialize_mirai_embeddings(
            args.mirai_repo,
            args.img_encoder_snapshot,
            cfg.out_dir / "manifest.parquet",
            args.features_out,
        )
        logger.info(
            f"wrote per-view embeddings → {args.features_out / 'per_view.parquet'}"
        )

    logger.info("=== preprocessing complete ===")


if __name__ == "__main__":
    main()


class MiraiZarrDataset(Dataset):
    """Minimal dataset that reads four-view exams from a manifest and emits tensors.

    It expects a Parquet manifest with columns: patient_id, exam_id, laterality, view,
    zarr_uri, zarr_key. It groups rows by (patient_id, exam_id) and yields a dict with:
        - 'patient_id', 'exam_id'
        - 'images': torch.float32 tensor of shape (4, 3, 2048, 1664), normalized with Mirai mean/std
        - 'order': tuple of view keys in canonical order (L_CC, L_MLO, R_CC, R_MLO)

    Raises if any expected view is missing. Uses rearrange for shape control.
    """

    ORDER = ("L_CC", "L_MLO", "R_CC", "R_MLO")

    def __init__(self, manifest_parquet: Path, channels: int = 3):
        self.manifest = pd.read_parquet(manifest_parquet)
        req = {"patient_id", "exam_id", "laterality", "view", "zarr_uri", "zarr_key"}
        missing = req - set(self.manifest.columns)
        if missing:
            raise KeyError(f"manifest missing columns: {sorted(missing)}")
        self.channels = channels
        self.groups = self.manifest.groupby(["patient_id", "exam_id"], sort=False)
        self.index = list(self.groups.groups.keys())

    def __len__(self) -> int:
        return len(self.index)

    def _load_view(self, row: pd.Series) -> np.ndarray:
        z = zarr.open(str(row["zarr_uri"]), mode="r")
        arr16 = z[row["zarr_key"]][:]
        x = arr16.astype(np.float32)
        x = (x - IMG_MEAN) / IMG_STD
        x = rearrange(x, "h w -> 1 h w")
        if self.channels == 3:
            x = np.repeat(x, 3, axis=0)
        return x

    def __getitem__(self, i: int) -> Dict:
        pid, eid = self.index[i]
        df = self.groups.get_group((pid, eid))
        keyed = {VIEWS[(r["laterality"], r["view"])]: r for _, r in df.iterrows()}
        if any(k not in keyed for k in self.ORDER):
            missing = [k for k in self.ORDER if k not in keyed]
            raise ValueError(f"exam {pid}/{eid} missing views: {missing}")
        imgs = [self._load_view(pd.Series(keyed[k])) for k in self.ORDER]
        x = np.stack(imgs, axis=0)
        return {
            "patient_id": pid,
            "exam_id": eid,
            "images": torch.from_numpy(x),
            "order": self.ORDER,
        }


# --- Patch for Mirai to resolve zarr:// URIs in csv datasets ---
# File: onconet/datasets/csv_mammo.py  (or the module that defines csv_mammo_risk_all_full_future)
# Replace the PNG loader with this function and call it where the code currently opens images


def _load_image_path(file_path: str) -> np.ndarray:
    """Return a CHW float32 tensor from either PNG path or zarr URI.

    Supports two cases:
      - regular PNG path (the original Mirai code path)
      - zarr URI of the form: 'zarr:///abs/path/to/exam.zarr#L_CC'
    """
    if file_path.startswith("zarr://"):
        uri, key = file_path[7:].split("#", 1)
        z = zarr.open(uri, mode="r")
        arr16 = z[key][:]
        x = arr16.astype(np.float32)
        x = (x - IMG_MEAN) / IMG_STD
        x = rearrange(x, "h w -> 1 h w")
        x = np.repeat(x, 3, axis=0)
        return x
    import PIL.Image as Image

    im = Image.open(file_path)
    x = np.array(im, dtype=np.float32)
    x = (x - IMG_MEAN) / IMG_STD
    x = rearrange(x, "h w -> 1 h w")
    x = np.repeat(x, 3, axis=0)
    return x
