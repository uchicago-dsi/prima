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
3) writes SoT Parquet tables (exams, views, cohort) for all full-quad exams
4) writes a Zarr cache per exam with four uint16 arrays (L_CC, L_MLO, R_CC, R_MLO)
5) emits a manifest.parquet pointing to the per-exam Zarr groups

Supports incremental processing: use --incremental to only process new exams
and append to existing SoT tables. Genotype filtering is done downstream.

Usage example
-------------
python prima_pipeline.py preprocess \
  --raw /data/prima/raw \
  --sot /data/prima/sot \
  --out /data/prima/mirai-prep

Defaults: --site ucmed, --workers 8

Dependencies (install explicitly):
  pip install pydicom numpy pandas pyarrow zarr numcodecs opencv-python-headless einops tqdm

Notes
-----
- no defensive coding; if a required tag is missing or an assumption is violated, we raise
- avoid silent defaults; CLI args are required
- per-image operations are vectorized; per-exam iteration is unavoidable by I/O
- rearrange is used for shape control; never permute/transpose

"""

import argparse
import hashlib
import json
import logging
import os
import signal
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import psutil

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
import zarr
from einops import rearrange
from numcodecs import Blosc
from pydicom import config
from pydicom.dataset import FileDataset
from pydicom.pixel_data_handlers import gdcm_handler, numpy_handler, pillow_handler

try:
    from pydicom.pixel_data_handlers import pylibjpeg_handler
except ImportError:
    pylibjpeg_handler = None  # type: ignore[assignment]
from pydicom.tag import Tag
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# prefer pylibjpeg for JPEG 2000 (gdcm 3.x not available on conda-forge)
config.pixel_data_handlers = [
    handler
    for handler in [
        pylibjpeg_handler,
        pillow_handler,
        gdcm_handler,
        numpy_handler,
    ]
    if handler is not None
]

# suppress pydicom VR UI validation warnings for non-standard UIDs
warnings.filterwarnings("ignore", message=".*Invalid value for VR UI.*", append=True)
# also set pydicom logging level to suppress warnings at the source
logging.getLogger("pydicom.valuerep").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """handle SIGINT/SIGTERM gracefully"""
    global shutdown_requested
    logger.info(f"received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True


# register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def log_memory_usage(context: str = "") -> None:
    """Log current memory usage for debugging."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logger.debug(f"Memory usage {context}: {memory_mb:.1f} MB")
    except Exception:
        pass  # don't fail if psutil isn't available


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB, return 0 if unavailable."""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


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

# DICOM tags are stored in wide format (one column per tag keyword)
# No fixed column list needed - dynamically determined from data


class PreprocessConfig:
    """Immutable configuration for preprocessing.

    All paths are required. No defaults to avoid accidental misconfiguration.
    """

    def __init__(
        self,
        raw_dir,
        sot_dir,
        out_dir,
        site,
        workers,
        summary,
        max_exams,
        debug_dir,
        chunk_size=100,
        no_resume=False,
        incremental=False,
    ):
        self.raw_dir = raw_dir
        self.sot_dir = sot_dir
        self.out_dir = out_dir
        self.site = site
        self.workers = workers
        self.summary = summary
        self.max_exams = max_exams
        self.debug_dir = debug_dir
        self.chunk_size = chunk_size
        self.no_resume = no_resume
        self.incremental = incremental


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


def extract_all_tags(ds: FileDataset, sop_instance_uid: str) -> Dict[str, str]:
    """Extract all DICOM tags as a dictionary for wide-format table.

    parameters
    ----------
    ds : FileDataset
        dicom dataset
    sop_instance_uid : str
        unique identifier to link tags to the view

    returns
    -------
    Dict[str, str]
        dictionary with sop_instance_uid and all tag keywords as columns
    """
    tag_dict = {"sop_instance_uid": sop_instance_uid}

    for tag in ds.dir():
        try:
            element = ds[tag]
            tag_keyword = element.keyword if hasattr(element, "keyword") else ""

            # skip if no keyword (can't use as column name)
            if not tag_keyword:
                continue

            # convert value to string, handling special cases
            if hasattr(element, "value"):
                if element.VR == "SQ":
                    value_str = f"<Sequence with {len(element.value)} items>"
                elif element.VR in ["OB", "OW", "OF", "OD"]:
                    value_str = f"<Binary data: {len(element.value)} bytes>"
                else:
                    value_str = str(element.value)
            else:
                value_str = ""

            tag_dict[tag_keyword] = value_str
        except Exception:
            # skip problematic tags silently
            pass

    return tag_dict


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
    accession_number: str = None,
) -> None:
    """Save debug figure showing pixel data and DICOM tags.

    status: 'SUCCESS' or 'FAILED'
    reason_or_view: if failed, the reason; if success, the laterality and view (e.g., 'L CC')
    input data: /patient_id/accession_number/exam_id/
    Organizes debug files by:
      - SUCCESS: debug_dir/success/patient_id/accession_number/
      - FAILED: debug_dir/failed/patient_id/fail_reason/exam_id/
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
        # use provided patient_id, exam_id, and accession_number if available, otherwise extract from path
        if patient_id is None:
            patient_id = path.parent.parent.name
        if exam_id is None:
            exam_id = path.parent.name
        if accession_number is None:
            accession_number = get_tag(ds, (0x0008, 0x0050), "unknown_accession")

        if status == "SUCCESS":
            # for successful files, organize by success/patient_id/accession_number
            success_dir = debug_dir / "success"
            patient_dir = success_dir / patient_id
            accession_dir = patient_dir / accession_number
            accession_dir.mkdir(parents=True, exist_ok=True)
            exam_dir = accession_dir
        else:
            # for failed files, organize by failed/patient_id/fail_reason/exam_id
            # clean up reason for directory name (remove spaces, special chars, fix double underscores)
            clean_reason = (
                reason_or_view.replace(" ", "_")
                .replace("/", "_")
                .replace("\\", "_")
                .replace(":", "_")
            )
            clean_reason = "".join(c for c in clean_reason if c.isalnum() or c in "_-")[
                :50
            ]  # limit length
            # fix multiple underscores in a row
            import re

            clean_reason = re.sub(r"_+", "_", clean_reason)
            # remove leading/trailing underscores
            clean_reason = clean_reason.strip("_")

            failed_dir = debug_dir / "failed"
            patient_dir = failed_dir / patient_id
            reason_dir = patient_dir / clean_reason
            exam_dir = reason_dir / exam_id
            exam_dir.mkdir(parents=True, exist_ok=True)

        out_name = f"{status}_{path.stem}.png"
        out_path = exam_dir / out_name
        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        if status == "SUCCESS":
            logger.debug(
                f"    Saved debug figure: success/{patient_id}/{accession_number}/{out_name}"
            )
        else:
            logger.debug(
                f"    Saved debug figure: failed/{patient_id}/{clean_reason}/{exam_id}/{out_name}"
            )
    except Exception as e:
        logger.warning(f"    Failed to save debug figure for {path.name}: {e}")


def _process_exam_dir(
    exam_path: Path, debug_dir: Optional[Path] = None
) -> Tuple[List[Dict], List[Dict]]:
    """Process all DICOMs in a single exam directory.

    walks subdirectories under exam_path to find all .dcm files
    """
    try:
        log_memory_usage(f"start_exam_{exam_path.name}")
        if debug_dir:
            logger.info(f"\n=== Processing exam: {exam_path} ===")

        # check if exam path still exists (race condition protection)
        if not exam_path.exists():
            logger.warning(f"Exam path {exam_path} no longer exists, skipping")
            return {
                "rows": [],
                "tag_rows": [],
                "exam_status": "path_not_found",
                "total_files": 0,
                "failed_files": 0,
                "total_dicoms": 0,
                "valid_dicoms": 0,
                "for_presentation_dicoms": 0,
                "valid_views": 0,
                "has_four_views": False,
            }
    except Exception as e:
        logger.error(f"Error in exam setup for {exam_path}: {e}")
        return {
            "rows": [],
            "tag_rows": [],
            "exam_status": "setup_error",
            "total_files": 0,
            "failed_files": 0,
            "total_dicoms": 0,
            "valid_dicoms": 0,
            "for_presentation_dicoms": 0,
            "valid_views": 0,
            "has_four_views": False,
        }

    try:
        # collect all dicom files first
        dcm_files = []
        for root, dirs, files in os.walk(exam_path):
            for name in files:
                if name.endswith(".dcm"):
                    dcm_files.append(Path(root) / name)

        if debug_dir:
            logger.info(f"Found {len(dcm_files)} DICOM files in exam")

        rows = []
        tag_rows = []
        failed_files = []

        for p in dcm_files:
            try:
                ds = read_dicom(p)
                patient_id = get_tag(ds, (0x0010, 0x0020))
                study_uid = get_tag(ds, (0x0020, 0x000D))
                sop_uid = get_tag(ds, (0x0008, 0x0018))
                accession_number = get_tag(ds, (0x0008, 0x0050))

                if patient_id is None or study_uid is None or sop_uid is None:
                    reason = "missing patient/study/SOP IDs"
                    if debug_dir:
                        logger.warning(f"  SKIP: {p.name} - {reason}")
                    failed_files.append((p, reason, ds))
                    if debug_dir:
                        _save_debug_figure(
                            ds,
                            p,
                            "FAILED",
                            reason,
                            debug_dir,
                            patient_id,
                            study_uid,
                            accession_number,
                        )
                    continue

                # check presentation intent first
                present = is_for_presentation(ds)
                if not present:
                    reason = "not for presentation"
                    if debug_dir:
                        logger.warning(f"  SKIP: {p.name} - {reason}")
                    failed_files.append((p, reason, ds))
                    if debug_dir:
                        _save_debug_figure(
                            ds,
                            p,
                            "FAILED",
                            reason,
                            debug_dir,
                            patient_id,
                            study_uid,
                            accession_number,
                        )
                    continue

                # try to get laterality and view
                try:
                    lat, vp = infer_view_fields(ds)
                except ValueError as e:
                    reason = str(e)
                    if debug_dir:
                        logger.warning(f"  SKIP: {p.name} - {reason}")
                    failed_files.append((p, reason, ds))
                    if debug_dir:
                        _save_debug_figure(
                            ds,
                            p,
                            "FAILED",
                            reason,
                            debug_dir,
                            patient_id,
                            study_uid,
                            accession_number,
                        )
                    continue

                if debug_dir:
                    logger.info(
                        f"  OK: {p.name} - {lat} {vp}, for_presentation={present}"
                    )

                # save debug figure for successful DICOMs too
                if debug_dir:
                    _save_debug_figure(
                        ds,
                        p,
                        "SUCCESS",
                        f"{lat} {vp}",
                        debug_dir,
                        patient_id,
                        study_uid,
                        accession_number,
                    )

                # prepare base row data
                row_data = {
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

                rows.append(row_data)

                # extract all DICOM tags for this image (wide format: one dict per image)
                tag_rows.append(extract_all_tags(ds, sop_uid))
            except Exception as e:
                if debug_dir:
                    logger.error(f"  ERROR: {p.name} - unexpected error: {e}")
                failed_files.append((p, f"unexpected error: {e}", None))

        if debug_dir:
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

        if rows and debug_dir:
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

        # if we got NO valid rows, log the issue but don't raise an error
        if not rows and debug_dir:
            logger.warning(f"\n=== NO VALID DICOMs found in exam {exam_path} ===")
            logger.warning("Dumping first failed DICOM for debugging:")
            if failed_files:
                p, reason, ds = failed_files[0]
                logger.warning(f"File: {p}")
                logger.warning(f"Reason: {reason}")
                if ds is not None:
                    logger.warning("\nAll DICOM tags:")
                    for elem in ds:
                        if elem.VR == "SQ":
                            logger.warning(f"  {elem.tag} {elem.name}: <sequence>")
                            continue
                        try:
                            value_str = str(elem.value)
                            if len(value_str) > 100:
                                value_str = value_str[:100] + "..."
                            logger.warning(f"  {elem.tag} {elem.name}: {value_str}")
                        except Exception:
                            logger.warning(
                                f"  {elem.tag} {elem.name}: <cannot display>"
                            )

            # return empty rows and exam status
            return {
                "rows": [],
                "tag_rows": [],
                "exam_status": "no_valid_dicoms",
                "total_files": len(dcm_files),
                "failed_files": len(failed_files),
                "total_dicoms": len(dcm_files),
                "valid_dicoms": 0,
                "for_presentation_dicoms": 0,
                "valid_views": 0,
                "has_four_views": False,
            }

        # check if exam has all four required views (L-CC, L-MLO, R-CC, R-MLO)
        view_keys = set((r["laterality"], r["view"]) for r in rows)
        required_views = {("L", "CC"), ("L", "MLO"), ("R", "CC"), ("R", "MLO")}
        has_four_views = len(view_keys & required_views) == 4

        # return successful exam data
        log_memory_usage(f"end_exam_{exam_path.name}")
        return {
            "rows": rows,
            "tag_rows": tag_rows,
            "exam_status": "success",
            "total_files": len(dcm_files),
            "failed_files": len(failed_files),
            "total_dicoms": len(dcm_files),
            "valid_dicoms": len(rows),
            "for_presentation_dicoms": sum(r["for_presentation"] for r in rows),
            "valid_views": len(view_keys),
            "has_four_views": has_four_views,
        }
    except Exception as e:
        logger.error(f"Unexpected error processing exam {exam_path}: {e}")
        log_memory_usage(f"error_exam_{exam_path.name}")
        return {
            "rows": [],
            "tag_rows": [],
            "exam_status": "processing_error",
            "total_files": 0,
            "failed_files": 0,
            "total_dicoms": 0,
            "valid_dicoms": 0,
            "for_presentation_dicoms": 0,
            "valid_views": 0,
            "has_four_views": False,
        }


def list_checkpoints(
    checkpoint_dir: Path = Path("data/discovery_checkpoints"),
) -> List[Dict]:
    """list all available checkpoints with metadata"""
    if not checkpoint_dir.exists():
        logger.info("no checkpoint directory found")
        return []

    checkpoints = []
    for checkpoint_file in checkpoint_dir.glob("discovery_*.json"):
        try:
            with open(checkpoint_file, "r") as f:
                data = json.load(f)

            checkpoint_info = {
                "file": checkpoint_file.name,
                "path": str(checkpoint_file),
                "processed_exams": len(data.get("processed_exams", [])),
                "total_exams": data.get("total_exams", 0),
                "progress_pct": (
                    len(data.get("processed_exams", []))
                    / max(data.get("total_exams", 1), 1)
                )
                * 100,
                "timestamp": data.get("timestamp", 0),
                "total_rows": len(data.get("all_rows", [])),
            }
            checkpoints.append(checkpoint_info)
        except Exception as e:
            logger.warning(f"failed to read checkpoint {checkpoint_file}: {e}")

    return sorted(checkpoints, key=lambda x: x["timestamp"], reverse=True)


def show_checkpoint_status(
    checkpoint_dir: Path = Path("data/discovery_checkpoints"),
) -> None:
    """show status of all checkpoints"""
    checkpoints = list_checkpoints(checkpoint_dir)

    if not checkpoints:
        print("No checkpoints found")
        return

    print(f"\nFound {len(checkpoints)} checkpoint(s) in {checkpoint_dir}")
    print("-" * 80)

    for i, cp in enumerate(checkpoints, 1):
        timestamp_str = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(cp["timestamp"])
        )
        print(f"{i}. {cp['file']}")
        print(
            f"   Progress: {cp['processed_exams']}/{cp['total_exams']} exams ({cp['progress_pct']:.1f}%)"
        )
        print(f"   Rows collected: {cp['total_rows']:,}")
        print(f"   Last updated: {timestamp_str}")
        print()


def monitor_progress(
    checkpoint_dir: Path = Path("data/discovery_checkpoints"), interval: int = 30
) -> None:
    """monitor preprocessing progress by watching checkpoints"""
    logger.info(f"monitoring progress in {checkpoint_dir} (interval: {interval}s)")
    logger.info("press Ctrl+C to stop monitoring")

    last_checkpoint = None

    try:
        while not shutdown_requested:
            checkpoints = list_checkpoints(checkpoint_dir)

            if checkpoints:
                latest = checkpoints[0]  # sorted by timestamp desc

                if last_checkpoint is None:
                    last_checkpoint = latest
                    logger.info(
                        f"monitoring started - {latest['processed_exams']}/{latest['total_exams']} exams processed"
                    )
                else:
                    # check for progress
                    if latest["processed_exams"] > last_checkpoint["processed_exams"]:
                        rate = (
                            latest["processed_exams"]
                            - last_checkpoint["processed_exams"]
                        ) / interval
                        eta_seconds = (
                            latest["total_exams"] - latest["processed_exams"]
                        ) / max(rate, 0.001)
                        eta_hours = eta_seconds / 3600

                        logger.info(
                            f"progress: {latest['processed_exams']}/{latest['total_exams']} exams "
                            f"({latest['progress_pct']:.1f}%) - "
                            f"rate: {rate:.1f} exams/sec - "
                            f"ETA: {eta_hours:.1f}h"
                        )

                        last_checkpoint = latest
                    elif latest["timestamp"] > last_checkpoint["timestamp"]:
                        # checkpoint updated but no progress (might be error handling)
                        logger.info(
                            f"checkpoint updated - {latest['processed_exams']}/{latest['total_exams']} exams processed"
                        )
                        last_checkpoint = latest
            else:
                if last_checkpoint is None:
                    logger.info("waiting for checkpoint to appear...")
                else:
                    logger.warning("no checkpoints found - processing may have failed")

            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("monitoring stopped by user")

    logger.info("monitoring ended")


def resume_from_checkpoint(
    checkpoint_file: Path,
    raw_dir: Path,
    max_exams: Optional[int] = None,
    workers: int = 4,
) -> None:
    """resume preprocessing from a specific checkpoint"""
    if not checkpoint_file.exists():
        logger.error(f"checkpoint file not found: {checkpoint_file}")
        return

    logger.info(f"resuming from checkpoint: {checkpoint_file}")

    try:
        views_df, tags_df = discover_dicoms(
            raw_dir=raw_dir,
            max_exams=max_exams,
            workers=workers,
            debug_dir=None,
            chunk_size=100,
            resume_from_checkpoint=True,
        )
        logger.info(
            f"resumed processing completed, collected {len(views_df)} views and {len(tags_df)} tag records"
        )
    except Exception as e:
        logger.error(f"failed to resume processing: {e}")
        raise


def cleanup_old_checkpoints(checkpoint_dir: Path, max_age_days: int = 7) -> None:
    """remove checkpoint files older than max_age_days"""
    cutoff_time = time.time() - (max_age_days * 24 * 3600)

    for checkpoint_file in checkpoint_dir.glob("discovery_*.json"):
        try:
            if checkpoint_file.stat().st_mtime < cutoff_time:
                checkpoint_file.unlink()
                logger.info(f"removed old checkpoint: {checkpoint_file.name}")
        except Exception as e:
            logger.warning(
                f"failed to remove old checkpoint {checkpoint_file.name}: {e}"
            )


def clean_checkpoints(
    checkpoint_dir: Path = Path("data/discovery_checkpoints"),
    max_age_days: int = 7,
    keep_latest: bool = True,
) -> None:
    """clean old checkpoints"""
    if not checkpoint_dir.exists():
        logger.info("no checkpoint directory found")
        return

    cutoff_time = time.time() - (max_age_days * 24 * 3600)
    checkpoints = list_checkpoints(checkpoint_dir)

    if not checkpoints:
        logger.info("no checkpoints to clean")
        return

    # keep the latest checkpoint if requested
    if keep_latest and checkpoints:
        checkpoints = checkpoints[1:]

    removed_count = 0
    for cp in checkpoints:
        if cp["timestamp"] < cutoff_time:
            try:
                Path(cp["path"]).unlink()
                logger.info(f"removed old checkpoint: {cp['file']}")
                removed_count += 1
            except Exception as e:
                logger.warning(f"failed to remove {cp['file']}: {e}")

    logger.info(f"cleaned {removed_count} old checkpoints")


def _combine_staging_files(staging_dir: Path, pattern: str) -> pd.DataFrame:
    """Combine multiple parquet files matching a pattern.

    For tags files with varying schemas, this ensures all chunks have the same columns.
    """
    import glob

    files = sorted(glob.glob(str(staging_dir / pattern)))
    if not files:
        return pd.DataFrame()

    # for tags files, collect all unique columns first to handle schema evolution
    if "tags_chunk" in pattern:
        # read all files and collect all columns
        all_columns = set()
        all_columns.add("sop_instance_uid")  # ensure this is always first

        for f in files:
            df = pd.read_parquet(f)
            all_columns.update(df.columns)

        # sort columns for consistency (sop_instance_uid first, rest alphabetical)
        sorted_cols = ["sop_instance_uid"] + sorted(
            [c for c in all_columns if c != "sop_instance_uid"]
        )

        # read each file and ensure it has all columns
        dfs = []
        for f in files:
            df = pd.read_parquet(f)
            # find missing columns
            missing_cols = [col for col in sorted_cols if col not in df.columns]
            # add all missing columns at once to avoid fragmentation
            if missing_cols:
                missing_df = pd.DataFrame(
                    {col: None for col in missing_cols}, index=df.index
                )
                df = pd.concat([df, missing_df], axis=1)
            # reorder columns
            df = df[sorted_cols]
            dfs.append(df)
    else:
        # for non-tags files, simple concat is fine
        dfs = [pd.read_parquet(f) for f in files]

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def discover_dicoms(
    raw_dir: Path,
    max_exams: Optional[int] = None,
    workers: int = 1,
    debug_dir: Optional[Path] = None,
    chunk_size: int = 100,
    resume_from_checkpoint: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Scan `raw_dir` for exam directories and extract DICOM tags in parallel.

    assumes structure: raw_dir/patient_id/exam_id/[series_dirs]/file.dcm
    walks to exam level, then processes each exam in parallel with checkpointing

    Args:
        raw_dir: Root directory containing patient/exam structure
        max_exams: Maximum number of exams to process (for debugging)
        workers: Number of parallel workers
        debug_dir: Directory for debug output
        chunk_size: Number of exams to process in each chunk
        resume_from_checkpoint: Whether to resume from existing checkpoints
    """
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # setup checkpoint directory
    checkpoint_dir = Path("data/discovery_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # cleanup old checkpoints
    cleanup_old_checkpoints(checkpoint_dir)

    # generate checkpoint filename based on raw_dir and parameters
    raw_dir_hash = hashlib.md5(str(raw_dir).encode()).hexdigest()[:8]
    checkpoint_file = (
        checkpoint_dir / f"discovery_{raw_dir_hash}_{max_exams or 'all'}.json"
    )

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

    # load or initialize checkpoint state
    # use a staging directory for incremental parquet writes
    staging_dir = (
        checkpoint_dir / f"staging_{hashlib.md5(str(raw_dir).encode()).hexdigest()[:8]}"
    )
    staging_dir.mkdir(parents=True, exist_ok=True)

    processed_exams = set()
    chunk_counter = 0  # track which chunk we're on for file naming
    exam_stats = {
        "total_exams": len(exam_dirs),
        "exams_with_valid_dicoms": 0,
        "exams_with_four_views": 0,
        "total_dicoms": 0,
        "valid_dicoms": 0,
        "for_presentation_dicoms": 0,
        "failed_dicoms": 0,
        "failed_exams": 0,
    }

    if resume_from_checkpoint and checkpoint_file.exists():
        logger.info(f"loading checkpoint from {checkpoint_file}")
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)
            processed_exams = set(checkpoint_data.get("processed_exams", []))
            chunk_counter = checkpoint_data.get("chunk_counter", 0)
            # load previous stats but keep total_exams current
            saved_stats = checkpoint_data.get("exam_stats", {})
            for key in exam_stats:
                if key in saved_stats and key != "total_exams":
                    exam_stats[key] = saved_stats[key]
            # total_exams should always reflect the full set
            exam_stats["total_exams"] = len(exam_dirs)
            logger.info(
                f"resuming from checkpoint: {len(processed_exams)} exams already processed, at chunk {chunk_counter}"
            )
        except Exception as e:
            logger.warning(f"failed to load checkpoint: {e}, starting fresh")

    # filter out already processed exams
    remaining_exams = [exam for exam in exam_dirs if str(exam) not in processed_exams]
    logger.info(
        f"processing {len(remaining_exams)} remaining exams (skipping {len(processed_exams)} already processed)"
    )

    if not remaining_exams:
        logger.info("all exams already processed, loading from staging files")
        # combine all staging files
        views_df = _combine_staging_files(staging_dir, "views_chunk_*.parquet")
        tags_df = _combine_staging_files(staging_dir, "tags_chunk_*.parquet")
        return views_df, tags_df

    # process exams in chunks
    logger.info(f"processing exams with {workers} workers in chunks of {chunk_size}...")

    def save_checkpoint():
        """save current progress to checkpoint file"""
        checkpoint_data = {
            "processed_exams": list(processed_exams),
            "chunk_counter": chunk_counter,
            "exam_stats": exam_stats,
            "timestamp": time.time(),
            "total_exams": len(exam_dirs),
        }
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

    num_chunks = (len(remaining_exams) + chunk_size - 1) // chunk_size
    chunk_ranges = range(0, len(remaining_exams), chunk_size)

    for chunk_start in tqdm(
        chunk_ranges,
        desc="processing chunks",
        unit="chunk",
        total=num_chunks,
        position=0,
    ):
        # check for shutdown request
        if shutdown_requested:
            logger.info(
                "shutdown requested, saving checkpoint and exiting gracefully..."
            )
            save_checkpoint()
            views_df = _combine_staging_files(staging_dir, "views_chunk_*.parquet")
            tags_df = _combine_staging_files(staging_dir, "tags_chunk_*.parquet")
            return views_df, tags_df

        chunk_end = min(chunk_start + chunk_size, len(remaining_exams))
        chunk_exams = remaining_exams[chunk_start:chunk_end]

        logger.info(
            f"processing chunk {chunk_start // chunk_size + 1}/{num_chunks}: exams {chunk_start + 1}-{chunk_end} of {len(remaining_exams)}"
        )

        chunk_start_mem = get_memory_usage_mb()
        chunk_max_mem = chunk_start_mem

        # accumulate rows for this chunk only
        chunk_rows = []
        chunk_tag_rows = []

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_exam_dir, exam_path, debug_dir): exam_path
                for exam_path in chunk_exams
            }

            # process chunk with progress bar
            for future in tqdm(
                as_completed(futures, timeout=600),  # 10 minute timeout per chunk
                total=len(futures),
                desc=f"chunk {chunk_start // chunk_size + 1}/{num_chunks}",
                unit="exam",
                position=1,
                leave=False,
            ):
                exam_path = futures[future]
                try:
                    result = future.result(timeout=120)  # 2 minute timeout per exam

                    # handle case where result might be None
                    if result is None:
                        logger.error(f"got None result for {exam_path}")
                        exam_stats["failed_exams"] += 1
                        continue

                    chunk_rows.extend(result["rows"])
                    chunk_tag_rows.extend(result["tag_rows"])
                    exam_stats["total_dicoms"] += result["total_dicoms"]
                    exam_stats["valid_dicoms"] += result["valid_dicoms"]
                    exam_stats["for_presentation_dicoms"] += result[
                        "for_presentation_dicoms"
                    ]
                    exam_stats["failed_dicoms"] += result["failed_files"]

                    if result["exam_status"] == "success":
                        exam_stats["exams_with_valid_dicoms"] += 1
                        if result["has_four_views"]:
                            exam_stats["exams_with_four_views"] += 1
                    else:
                        exam_stats["failed_exams"] += 1

                    # mark as processed
                    processed_exams.add(str(exam_path))

                    # track peak memory
                    current_mem = get_memory_usage_mb()
                    chunk_max_mem = max(chunk_max_mem, current_mem)

                except TimeoutError:
                    logger.error(f"timeout processing {exam_path}")
                    exam_stats["failed_exams"] += 1
                    processed_exams.add(
                        str(exam_path)
                    )  # mark as processed to avoid retry
                except Exception as e:
                    logger.error(f"failed to process {exam_path}: {e}")
                    exam_stats["failed_exams"] += 1
                    processed_exams.add(
                        str(exam_path)
                    )  # mark as processed to avoid retry

        # write chunk data to staging files
        if chunk_rows:
            chunk_views_df = pd.DataFrame(chunk_rows)
            chunk_tags_df = pd.DataFrame(chunk_tag_rows)

            views_file = staging_dir / f"views_chunk_{chunk_counter:04d}.parquet"
            tags_file = staging_dir / f"tags_chunk_{chunk_counter:04d}.parquet"

            chunk_views_df.to_parquet(views_file, index=False)
            chunk_tags_df.to_parquet(tags_file, index=False)

            logger.info(
                f"wrote chunk {chunk_counter} to staging: {len(chunk_rows)} views, {len(chunk_tag_rows)} tags"
            )

        chunk_counter += 1

        # save checkpoint after each chunk
        save_checkpoint()
        chunk_end_mem = get_memory_usage_mb()
        mem_delta = chunk_end_mem - chunk_start_mem
        logger.info(
            f"chunk completed, checkpoint saved | "
            f"memory: start={chunk_start_mem:.0f}MB, peak={chunk_max_mem:.0f}MB, "
            f"end={chunk_end_mem:.0f}MB, delta={mem_delta:+.0f}MB"
        )

    # final checkpoint save
    save_checkpoint()
    logger.info("all processing completed. final checkpoint saved.")

    # combine all staging files
    logger.info("combining staging files...")
    views_df = _combine_staging_files(staging_dir, "views_chunk_*.parquet")
    tags_df = _combine_staging_files(staging_dir, "tags_chunk_*.parquet")

    logger.info(
        f"loaded {len(views_df)} views and {len(tags_df)} DICOM tag records from staging"
    )

    # print summary statistics
    logger.info("\n=== PROCESSING SUMMARY ===")
    logger.info(f"Total exams processed: {exam_stats['total_exams']}")
    logger.info(f"Total views (DICOM files) processed: {exam_stats['total_dicoms']}")

    if not views_df.empty:
        logger.info(f"Unique patients: {views_df['patient_id'].nunique()}")
        logger.info(f"Unique exams with valid views: {views_df['exam_id'].nunique()}")

    logger.info("\n=== FILTERING FLOW ===")
    logger.info(f"1. Total exams: {exam_stats['total_exams']}")
    logger.info(
        f"2. Exams with valid views: {exam_stats['exams_with_valid_dicoms']} (lost {exam_stats['total_exams'] - exam_stats['exams_with_valid_dicoms']} exams)"
    )
    logger.info(
        f"3. Exams with all four required views (L-CC, L-MLO, R-CC, R-MLO): {exam_stats['exams_with_four_views']} (lost {exam_stats['exams_with_valid_dicoms'] - exam_stats['exams_with_four_views']} exams)"
    )

    logger.info("\n=== VIEW-LEVEL STATISTICS ===")
    logger.info(f"Total views processed: {exam_stats['total_dicoms']}")
    logger.info(f"  - Valid views (passed all checks): {exam_stats['valid_dicoms']}")
    logger.info(f"  - For presentation views: {exam_stats['for_presentation_dicoms']}")
    logger.info(f"  - Failed views: {exam_stats['failed_dicoms']}")

    if not views_df.empty:
        # view breakdown by type
        view_counts = views_df.groupby(["laterality", "view"]).size()
        logger.info("\nViews by type:")
        for (lat, view), count in view_counts.items():
            logger.info(f"  {lat}-{view}: {count} views")
    else:
        logger.warning("No valid views found in any exam!")

    if not tags_df.empty:
        num_tag_cols = len(tags_df.columns) - 1  # exclude sop_instance_uid
        logger.info(
            f"\nDICOM tags: {num_tag_cols} unique tags for {len(tags_df)} images (wide format)"
        )

    return views_df, tags_df


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

    if cfg.incremental and (sot / "views.parquet").exists():
        logger.info("incremental mode: loading existing views...")
        existing_views = pd.read_parquet(sot / "views.parquet")
        existing_exam_ids = set(existing_views["exam_id"].unique())
        logger.info(f"found {len(existing_exam_ids)} existing exams")

        # discover new DICOMs
        views_df, tags_df = discover_dicoms(
            raw,
            max_exams=cfg.max_exams,
            workers=cfg.workers,
            debug_dir=cfg.debug_dir,
            chunk_size=cfg.chunk_size,
            resume_from_checkpoint=not cfg.no_resume,
        )

        # filter out already processed exams
        new_views = views_df[~views_df["exam_id"].isin(existing_exam_ids)]
        logger.info(
            f"found {len(new_views)} new DICOM files from {new_views['exam_id'].nunique()} new exams"
        )

        if len(new_views) == 0:
            logger.info("no new exams found, nothing to process")
            return

        views_df = new_views
        # filter tags to match filtered views
        new_sop_uids = set(new_views["sop_instance_uid"])
        tags_df = tags_df[tags_df["sop_instance_uid"].isin(new_sop_uids)]
    else:
        views_df, tags_df = discover_dicoms(
            raw,
            max_exams=cfg.max_exams,
            workers=cfg.workers,
            debug_dir=cfg.debug_dir,
            chunk_size=cfg.chunk_size,
            resume_from_checkpoint=not cfg.no_resume,
        )
        logger.info(f"found {len(views_df)} DICOM files")

    # selection to full quad
    logger.info("selecting full quad exams...")
    sel_df = select_full_quad(views_df)
    logger.info(f"selected {len(sel_df)} views from full quad exams")

    # filter tags to match selected views only
    selected_sop_uids = set(sel_df["sop_instance_uid"])
    tags_df = tags_df[tags_df["sop_instance_uid"].isin(selected_sop_uids)]
    logger.info(f"retained {len(tags_df)} DICOM tag records for selected views")

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

    # write SoT tables
    logger.info(f"writing SoT tables to {sot}...")
    sot.mkdir(parents=True, exist_ok=True)

    if cfg.incremental and (sot / "views.parquet").exists():
        # append to existing tables
        logger.info("appending to existing SoT tables...")
        existing_views = pd.read_parquet(sot / "views.parquet")
        existing_exams = pd.read_parquet(sot / "exams.parquet")

        # combine new and existing data
        combined_views = pd.concat(
            [existing_views, sel_df[VIEWS_COLS]], ignore_index=True
        )
        combined_exams = pd.concat(
            [existing_exams, exams[EXAMS_COLS]], ignore_index=True
        )

        # remove duplicates (in case of re-processing)
        combined_views = combined_views.drop_duplicates(
            subset=["exam_id", "sop_instance_uid"]
        )
        combined_exams = combined_exams.drop_duplicates(
            subset=["patient_id", "exam_id"]
        )

        combined_views.to_parquet(sot / "views.parquet", index=False)
        combined_exams.to_parquet(sot / "exams.parquet", index=False)

        # append tags
        if (sot / "dicom_tags.parquet").exists():
            existing_tags = pd.read_parquet(sot / "dicom_tags.parquet")
            combined_tags = pd.concat([existing_tags, tags_df], ignore_index=True)
            combined_tags = combined_tags.drop_duplicates(subset=["sop_instance_uid"])
            combined_tags.to_parquet(sot / "dicom_tags.parquet", index=False)
        else:
            tags_df.to_parquet(sot / "dicom_tags.parquet", index=False)

        # update cohort with all full-quad exams
        cohort = combined_exams[combined_exams["has_full_quad"]][
            ["patient_id", "patient_hash", "exam_id", "study_date"]
        ]
        cohort.to_parquet(sot / "cohort.parquet", index=False)
        logger.info(
            "updated views.parquet, exams.parquet, dicom_tags.parquet, cohort.parquet"
        )
    else:
        # write new tables
        sel_df[VIEWS_COLS].to_parquet(sot / "views.parquet", index=False)
        exams[EXAMS_COLS].to_parquet(sot / "exams.parquet", index=False)
        tags_df.to_parquet(sot / "dicom_tags.parquet", index=False)

        # create cohort of all full-quad exams (genotype filtering done downstream)
        cohort = exams[exams["has_full_quad"]][
            ["patient_id", "patient_hash", "exam_id", "study_date"]
        ]
        cohort.to_parquet(sot / "cohort.parquet", index=False)
        logger.info(
            "wrote views.parquet, exams.parquet, dicom_tags.parquet, cohort.parquet"
        )

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
    failed_exams: List[Dict] = []

    logger.info("grouping exams for zarr writing...")
    grouped = {k: v for k, v in sel_df.groupby("exam_id")}
    targets = [
        (eid, grouped[eid]) for eid in cohort["exam_id"].tolist() if eid in grouped
    ]
    logger.info(f"will write {len(targets)} exams to zarr with {cfg.workers} workers")

    # load existing manifest if incremental
    existing_manifest = None
    if cfg.incremental and (out / "manifest.parquet").exists():
        existing_manifest = pd.read_parquet(out / "manifest.parquet")
        logger.info(f"loaded existing manifest with {len(existing_manifest)} entries")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futs = {
            ex.submit(write_exam_zarr, grp_df, prep_dir): eid for eid, grp_df in targets
        }
        try:
            for fut in tqdm(
                as_completed(futs), total=len(futs), desc="writing zarr", unit="exam"
            ):
                if shutdown_requested:
                    logger.info(
                        "shutdown requested during zarr writing, cancelling remaining tasks..."
                    )
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise KeyboardInterrupt("user requested shutdown")

                eid = futs[fut]
                try:
                    zpath, shapes = fut.result()
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
                except Exception as e:
                    grp_df = grouped[eid]
                    first = grp_df.iloc[0]
                    patient_id = first["patient_id"]
                    error_type = type(e).__name__
                    error_msg = str(e)
                    failed_exams.append(
                        {
                            "patient_id": patient_id,
                            "exam_id": eid,
                            "error_type": error_type,
                            "error_message": error_msg,
                        }
                    )
                    logger.warning(
                        f"failed to convert exam {eid} (patient {patient_id}): {error_type}: {error_msg}"
                    )
        except KeyboardInterrupt:
            logger.info("cancelling zarr writing...")
            ex.shutdown(wait=False, cancel_futures=True)
            raise

    if shutdown_requested:
        logger.warning("shutdown requested, skipping manifest write")
        logger.info(f"partial zarr cache written to {prep_dir}")
        return

    logger.info(f"writing manifest with {len(manifest_rows)} rows...")
    manifest = pd.DataFrame(manifest_rows)

    if cfg.incremental and existing_manifest is not None:
        # combine with existing manifest
        combined_manifest = pd.concat([existing_manifest, manifest], ignore_index=True)
        # remove duplicates
        combined_manifest = combined_manifest.drop_duplicates(
            subset=["patient_id", "exam_id", "laterality", "view"]
        )
        manifest = combined_manifest
        logger.info(f"combined manifest now has {len(manifest)} total entries")

    manifest_path = out / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    logger.info(f"wrote manifest to {manifest_path}")

    # write failure log if any conversions failed
    if failed_exams:
        failures_path = out / "conversion_failures.parquet"
        failures_df = pd.DataFrame(failed_exams)
        failures_df.to_parquet(failures_path, index=False)
        logger.warning(
            f"wrote {len(failed_exams)} conversion failures to {failures_path}"
        )
    else:
        logger.info("no conversion failures")

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
    logger.info(f"cohort (full-quad exams): {cohort_exams}")
    logger.info(f"successful conversions: {len(manifest_rows) // 4}")
    logger.info(f"failed conversions: {len(failed_exams)}")
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


def parse_args():
    """Parse CLI arguments and handle different commands."""
    parser = argparse.ArgumentParser(description="prima preprocessing pipeline")
    subparsers = parser.add_subparsers(dest="command", help="available commands")

    # preprocess command
    p = subparsers.add_parser("preprocess", help="run full preprocessing pipeline")
    p.add_argument(
        "--raw",
        dest="raw_dir",
        type=Path,
        default=Path("/gpfs/data/huo-lab/Image/ChiMEC/MG"),
    )
    p.add_argument("--sot", dest="sot_dir", type=Path, default=None)
    p.add_argument("--out", dest="out_dir", type=Path, default=None)
    p.add_argument("--site", dest="site", type=str, default="ucmed")
    p.add_argument("--workers", dest="workers", type=int, default=32)
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
    p.add_argument(
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=100,
        help="number of exams to process in each chunk (default: 100)",
    )
    p.add_argument(
        "--no-resume",
        dest="no_resume",
        action="store_true",
        help="don't resume from checkpoint, start fresh",
    )
    p.add_argument(
        "--incremental",
        dest="incremental",
        action="store_true",
        help="incremental mode: only process new exams, append to existing SoT tables",
    )
    p.add_argument(
        "--emit-csv",
        dest="emit_csv",
        action="store_true",
        default=True,
        help="emit CSV outputs (default: True)",
    )

    # optional extras
    p.add_argument("--labels", dest="labels_parquet", type=Path)
    p.add_argument("--features-out", dest="features_out", type=Path)
    p.add_argument("--img-encoder-snapshot", dest="img_encoder_snapshot", type=Path)
    p.add_argument("--mirai-repo", dest="mirai_repo", type=Path)

    # checkpoint management commands
    cp = subparsers.add_parser("checkpoint", help="manage checkpoints")
    cp.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("data/discovery_checkpoints"),
        help="directory containing checkpoints",
    )
    cp_subparsers = cp.add_subparsers(dest="cp_command", help="checkpoint commands")

    cp_subparsers.add_parser("list", help="list all checkpoints")
    cp_subparsers.add_parser("status", help="show checkpoint status")

    clean_parser = cp_subparsers.add_parser("clean", help="clean old checkpoints")
    clean_parser.add_argument(
        "--max-age-days",
        type=int,
        default=7,
        help="remove checkpoints older than N days",
    )
    clean_parser.add_argument(
        "--keep-latest",
        action="store_true",
        default=True,
        help="keep the latest checkpoint",
    )

    resume_parser = cp_subparsers.add_parser("resume", help="resume from checkpoint")
    resume_parser.add_argument(
        "checkpoint_file", type=Path, help="checkpoint file to resume from"
    )
    resume_parser.add_argument(
        "--raw-dir", type=Path, required=True, help="raw data directory"
    )
    resume_parser.add_argument("--max-exams", type=int, help="limit number of exams")
    resume_parser.add_argument(
        "--workers", type=int, default=4, help="number of workers"
    )

    # monitor command
    monitor_parser = subparsers.add_parser(
        "monitor", help="monitor preprocessing progress"
    )
    monitor_parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("data/discovery_checkpoints"),
        help="directory containing checkpoints",
    )
    monitor_parser.add_argument(
        "--interval", type=int, default=30, help="monitoring interval in seconds"
    )

    args = parser.parse_args()

    if args.command == "preprocess":
        # derive sot and out from raw if not provided
        raw_dir = args.raw_dir
        sot_dir = args.sot_dir if args.sot_dir is not None else raw_dir / "sot"
        out_dir = args.out_dir if args.out_dir is not None else raw_dir / "out"

        cfg = PreprocessConfig(
            raw_dir=raw_dir,
            sot_dir=sot_dir,
            out_dir=out_dir,
            site=args.site,
            workers=args.workers,
            summary=args.summary,
            max_exams=args.max_exams,
            debug_dir=args.debug_dir,
            chunk_size=args.chunk_size,
            no_resume=args.no_resume,
            incremental=args.incremental,
        )
        return cfg, args
    else:
        return None, args


def main() -> None:
    """Entrypoint for the CLI."""
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    log_datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt=log_datefmt)

    logger.info("parsing arguments...")
    cfg, args = parse_args()

    if cfg is not None:
        log_dir = cfg.out_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = log_dir / f"{args.command}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=log_datefmt))
        logging.getLogger().addHandler(file_handler)

        logger.info(f"Log file: {log_file}")

    if args.command == "preprocess":
        logger.info("configuration loaded")
        logger.info(f"  raw_dir: {cfg.raw_dir}")
        logger.info(f"  sot_dir: {cfg.sot_dir}")
        logger.info(f"  out_dir: {cfg.out_dir}")
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

    elif args.command == "checkpoint":
        if args.cp_command == "list":
            checkpoints = list_checkpoints(args.checkpoint_dir)
            if checkpoints:
                for cp in checkpoints:
                    print(
                        f"{cp['file']}: {cp['processed_exams']}/{cp['total_exams']} exams"
                    )
            else:
                print("No checkpoints found")

        elif args.cp_command == "status":
            show_checkpoint_status(args.checkpoint_dir)

        elif args.cp_command == "clean":
            clean_checkpoints(args.checkpoint_dir, args.max_age_days, args.keep_latest)

        elif args.cp_command == "resume":
            resume_from_checkpoint(
                args.checkpoint_file, args.raw_dir, args.max_exams, args.workers
            )

        else:
            print("Available checkpoint commands: list, status, clean, resume")

    elif args.command == "monitor":
        monitor_progress(args.checkpoint_dir, args.interval)

    else:
        print("Available commands: preprocess, checkpoint, monitor")
        print("Use --help for more information")


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
