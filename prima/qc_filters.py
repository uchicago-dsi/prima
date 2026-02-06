"""qc_filters.py

Canonical auto-filter definitions for QC exclusion.

Each filter maps a name to a set of exam IDs that should be excluded.
All consumers (qc_gallery, analyze_mirai, qc_cutflow) import from here
so filter logic is defined exactly once.

Available filters (configure in data/qc_auto_exclude.json):
    gems_ffdm_tc1       AcquisitionDeviceProcessingCode starts with GEMS_
    has_implant         has_implant flag from preprocessing
    scanned_film        DetectorType == FILM
    duplicate_sop_uid   re-archived duplicates sharing SOP Instance UIDs
                        within a patient (keeps first exam_id alphabetically)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

SUPPORTED_AUTO_FILTERS = {
    "gems_ffdm_tc1",
    "has_implant",
    "scanned_film",
    "duplicate_sop_uid",
}

DEFAULT_AUTO_EXCLUDE_CONFIG = Path("data/qc_auto_exclude.json")


def load_auto_filter_names(
    config_path: Optional[Path] = None,
    explicit_filters: Optional[list[str]] = None,
) -> list[str]:
    """load ordered list of auto-filter names from config or explicit list.

    Parameters
    ----------
    config_path : Path | None
        path to qc_auto_exclude.json; defaults to data/qc_auto_exclude.json
    explicit_filters : list[str] | None
        if provided, use these filter names instead of reading from config
    """
    if explicit_filters is not None:
        candidates = list(explicit_filters)
    else:
        if config_path is None:
            config_path = DEFAULT_AUTO_EXCLUDE_CONFIG
        if not config_path.exists():
            return []
        with open(config_path) as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return []
        candidates = payload.get("filters", [])

    filters: list[str] = []
    for raw in candidates:
        name = str(raw).strip().lower()
        if not name or name in filters:
            continue
        if name not in SUPPORTED_AUTO_FILTERS:
            print(f"[warn] ignoring unsupported auto filter '{name}'")
            continue
        filters.append(name)
    return filters


def compute_auto_filter_sets(
    views_path: Path,
    tags_path: Path,
    filter_names: Optional[list[str]] = None,
) -> dict[str, set[str]]:
    """compute exam-id sets matched by each auto filter.

    Parameters
    ----------
    views_path : Path
        path to views.parquet
    tags_path : Path
        path to dicom_tags.parquet
    filter_names : list[str] | None
        which filters to compute; if None, computes all supported filters

    Returns
    -------
    dict[str, set[str]]
        mapping filter name -> set of matching exam_id strings
    """
    if not views_path.exists() or not tags_path.exists():
        return {}

    views = pd.read_parquet(views_path)
    views["exam_id"] = views["exam_id"].astype(str)
    tags = pd.read_parquet(tags_path)
    merged = views.merge(tags, on="sop_instance_uid", how="left")
    merged["exam_id"] = merged["exam_id"].astype(str)

    if filter_names is None:
        filter_names = sorted(SUPPORTED_AUTO_FILTERS)

    result: dict[str, set[str]] = {}

    for name in filter_names:
        if name not in SUPPORTED_AUTO_FILTERS:
            continue

        exam_ids: set[str] = set()

        if (
            name == "gems_ffdm_tc1"
            and "AcquisitionDeviceProcessingCode" in merged.columns
        ):
            matches = merged[
                merged["AcquisitionDeviceProcessingCode"]
                .astype(str)
                .str.startswith("GEMS_", na=False)
            ]["exam_id"]
            exam_ids.update(matches.unique())

        elif name == "has_implant" and "has_implant" in views.columns:
            matches = views[views["has_implant"]]["exam_id"]
            exam_ids.update(matches.unique())

        elif name == "scanned_film" and "DetectorType" in merged.columns:
            matches = merged[merged["DetectorType"].astype(str) == "FILM"]["exam_id"]
            exam_ids.update(matches.unique())

        elif name == "duplicate_sop_uid":
            dup_counts = views.groupby(["patient_id", "sop_instance_uid"])[
                "exam_id"
            ].nunique()
            shared = dup_counts[dup_counts > 1].reset_index()
            if len(shared) > 0:
                detail = views.merge(
                    shared[["patient_id", "sop_instance_uid"]],
                    on=["patient_id", "sop_instance_uid"],
                )
                for _, grp in detail.groupby("patient_id"):
                    eids = sorted(grp["exam_id"].astype(str).unique())
                    exam_ids.update(eids[1:])

        if exam_ids:
            result[name] = exam_ids

    return result
