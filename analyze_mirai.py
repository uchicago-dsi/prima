#!/usr/bin/env python3
"""analyze_mirai_output.py

Quick analysis for Mirai's validation outputs.

Reads Mirai's prediction CSV and metadata CSV, then computes per-horizon
summary metrics, survival metrics, and stratified performance tables.

Notes
-----
- this is *not* Uno's time-dependent C-index; it's a pragmatic ROC on a binary label per horizon
- labels are computed as:
    case_h  = years_to_cancer <= h
    ctrl_h  = (years_to_cancer  > h) & (years_to_last_followup >= h)
    include = case_h | ctrl_h
  rows with insufficient follow-up are excluded from AUC_h
- prediction columns are auto-detected via regex; override with config `colmap` if needed

Usage
-----
python analyze_mirai.py /path/to/analyze_mirai_config.yaml

The config controls:
- input/output paths
- split/column mapping/k-fold options
- QC status filters (e.g., include only "good" or exclude "bad")
- automatic QC server filters (GEMS, implants, scanned film, etc.)
- annotation-based filtering (include/exclude by tags)

The script writes:
- analysis outputs
- a copy of the input config
- a resolved config (with defaults and absolute paths)

By default, config `out_dir` is expected to contain `validation_output.csv`
and `mirai_manifest.csv` unless `pred_csv` / `meta_csv` are set explicitly.

Requirements
------------
  pip install pandas numpy scikit-learn pyarrow omegaconf

"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prima.qc_filters import compute_auto_filter_sets

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sksurv.metrics import (
    brier_score,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv

PHENOTYPE_CSV = Path(
    "/gpfs/data/phs/groups/Projects/Huo_projects/SPORE/annawoodard/Phenotype_ChiMEC_2025Oct4.csv"
)
PATIENT_KEY_CSV = Path("/gpfs/data/huo-lab/Image/ChiMEC/study-16352a.csv")
PHENOTYPE_COLUMNS = [
    "CaseControl",
    "RaceEthnic",
    "dob",
    "grade",
    "grade_new",
    "subtype",
    "subtype_new",
    "ER",
    "ER_new",
    "PR",
    "PR_new",
    "HER2",
    "HER2_new",
]
POSITIVE_MARKERS = {"pos", "positive", "1", "true", "yes", "y"}
NEGATIVE_MARKERS = {"neg", "negative", "0", "false", "no", "n"}
OMOLEYE_TARGET_HORIZONS = [1, 2, 3, 4, 5]

LEGACY_ANNOTATION_TAG_ALIASES = {
    "detector artifact - vertical line": "vertical line (detector artifact)",
    "detector artifact - horizontal line": "horizontal line (detector artifact)",
    "horizontal line (detector artifact": "horizontal line (detector artifact)",
}


@dataclass(frozen=True)
class QCFilterConfig:
    """QC-driven filtering configuration loaded from JSON config."""

    qc_file: Path | None
    include_statuses: tuple[str, ...] | None
    exclude_statuses: tuple[str, ...]
    annotations_file: Path | None
    annotation_include_any: tuple[str, ...]
    annotation_include_all: tuple[str, ...]
    annotation_exclude_any: tuple[str, ...]
    annotation_exclude_all: tuple[str, ...]
    enable_auto_filters: bool
    auto_filter_config: Path | None
    auto_filters: tuple[str, ...] | None


@dataclass(frozen=True)
class Config:
    """immutable configuration for analysis"""

    pred_csv: Path
    meta_csv: Path
    out_json: Path | None
    split: str | None
    colmap: dict[int, str] | None
    cindex_col: str | None
    kfold: int | None
    auc_mode: str
    random_exam_seed: int
    qc_filters: QCFilterConfig
    config_path: Path
    raw_config: dict[str, Any]
    resolved_config: dict[str, Any]


@dataclass(frozen=True)
class ExamData:
    """container for merged prediction/metadata tables"""

    merged_views: pd.DataFrame
    exam_level: pd.DataFrame
    exam_level_unfiltered: pd.DataFrame
    meta_all: pd.DataFrame
    meta_eval: pd.DataFrame
    pred_cols: dict[int, str]
    n_pred_rows: int
    filter_summary: dict[str, Any]
    exam_pairs: frozenset[tuple[str, str]]


def _parse_colmap(colmap_obj: Any) -> dict[int, str] | None:
    """parse colmap object like {"1": "pred_1year"} into {1: "pred_1year"}."""
    if colmap_obj is None:
        return None
    if not isinstance(colmap_obj, dict):
        raise ValueError("config key 'colmap' must be a JSON object")
    out: dict[int, str] = {}
    for key, value in colmap_obj.items():
        out[int(key)] = str(value)
    return dict(sorted(out.items()))


def _normalize_string_tuple(
    values: Any, *, lowercase: bool = False
) -> tuple[str, ...] | None:
    """normalize config lists like ['bad', 'review'] to a deduplicated tuple."""
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        raise ValueError("expected list/tuple")

    normalized: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        if lowercase:
            item = item.lower()
        if item not in normalized:
            normalized.append(item)
    return tuple(normalized)


def _resolve_config_path(value: Any, config_dir: Path) -> Path | None:
    """resolve config path values relative to the config file directory."""
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    return path


def _canonical_annotation_tag(tag: str) -> str:
    """normalize annotation tags to canonical labels used by QC server."""
    stripped = str(tag).strip()
    if not stripped:
        return ""
    canonical = LEGACY_ANNOTATION_TAG_ALIASES.get(stripped, stripped)
    return canonical.lower()


def _assign_race_category(series: pd.Series | None) -> pd.Series:
    """map free-text race/ethnicity labels to mutually exclusive buckets"""

    if series is None:
        return pd.Series(dtype=object)
    race_norm = (
        series.astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    category = pd.Series("Unknown", index=series.index, dtype=object)

    non_hispanic = race_norm.str.contains(
        r"non[\s-]?hispanic", case=False, na=False, regex=True
    )
    is_hispanic = (
        race_norm.str.contains("hispanic", case=False, na=False, regex=True)
        & ~non_hispanic
    )

    mask_black = race_norm.str.contains(
        "black|african american", case=False, na=False, regex=True
    )
    mask_asian = race_norm.str.contains(
        "asian|pacific islander", case=False, na=False, regex=True
    )
    mask_alaska = race_norm.str.contains(
        "alaska native|american indian", case=False, na=False, regex=True
    )
    mask_white = race_norm.str.contains("white", case=False, na=False, regex=True)

    category[mask_black] = "African American"
    category[mask_asian] = "Asian or Pacific Islander"
    category[mask_alaska] = "Alaska Native"
    category[is_hispanic] = "Hispanic"
    category[mask_white & ~is_hispanic] = "White"

    empty = race_norm.isna() | (race_norm == "")
    category[empty] = "Unknown"
    return category


def _marker_is_positive(value: object) -> bool | None:
    """normalize marker status strings like 'Pos', 'Neg', 0/1"""

    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (int, float)):
        if value > 0:
            return True
        if value == 0:
            return False
        return None
    val = str(value).strip().lower()
    if val in ("", "nan", "na", "none"):
        return None
    if val in POSITIVE_MARKERS:
        return True
    if val in NEGATIVE_MARKERS:
        return False
    return None


def _infer_hr_status(row: pd.Series) -> str | None:
    """derive HR+/HR- status from ER/PR markers"""

    status_flags: list[bool] = []
    for col in ("ER_new", "ER", "PR_new", "PR"):
        if col in row:
            marker = _marker_is_positive(row[col])
            if marker is not None:
                status_flags.append(marker)
    if not status_flags:
        return None
    if any(status_flags):
        return "HR+"
    return "HR-"


def _normalize_subtype_string(value: object) -> str | None:
    """normalize free-text subtype strings to HR+/HER2- style"""

    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip().upper()
    if text in ("", "NAN", "NA", "NONE"):
        return None

    hr = None
    if "HR+" in text:
        hr = "HR+"
    elif "HR-" in text:
        hr = "HR-"
    elif "ER+" in text or "PR+" in text:
        hr = "HR+"
    elif "ER-" in text and "PR-" in text:
        hr = "HR-"

    her2 = None
    if "HER2+" in text:
        her2 = "HER2+"
    elif "HER2-" in text:
        her2 = "HER2-"

    if hr and her2:
        return f"{hr}/{her2}"
    return None


def _infer_receptor_subtype(row: pd.Series) -> str | None:
    """derive receptor subtype, preferring explicit subtype columns"""

    for col in ("subtype_new", "subtype"):
        if col in row:
            subtype = _normalize_subtype_string(row[col])
            if subtype:
                return subtype

    hr_status = row.get("hr_status") or _infer_hr_status(row)

    her2_marker = None
    for col in ("HER2_new", "HER2"):
        if col in row:
            marker = _marker_is_positive(row[col])
            if marker is not None:
                her2_marker = marker
                break

    if hr_status and her2_marker is not None:
        return f"{hr_status}/{'HER2+' if her2_marker else 'HER2-'}"
    return None


def _extract_grade_value(value: object) -> int | None:
    """extract numeric grade from raw phenotype strings"""

    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (int, float)):
        grade_int = int(value)
        if grade_int in (1, 2, 3):
            return grade_int
    text = str(value)
    match = re.search(r"[123]", text)
    if match:
        return int(match.group(0))
    return None


def _grade_category_from_row(row: pd.Series) -> str | None:
    """map numeric grade to low/intermediate/high"""

    grade_value = None
    for col in ("grade_new", "grade"):
        if col in row:
            grade_value = _extract_grade_value(row[col])
            if grade_value is not None:
                break
    if grade_value == 1:
        return "Low grade"
    if grade_value == 2:
        return "Intermediate grade"
    if grade_value == 3:
        return "High grade"
    return None


def _format_elapsed(elapsed: float) -> str:
    """format elapsed time in seconds to human-readable string"""
    if elapsed < 60:
        return f"{elapsed:.2f}s"
    elif elapsed < 3600:
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        return f"{minutes}m {seconds:.2f}s"
    else:
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60
        return f"{hours}h {minutes}m {seconds:.2f}s"


def _load_enriched_manifest(cfg: Config) -> pd.DataFrame:
    """load mirai manifest merged with phenotype fields and derived attributes"""

    meta = (
        pd.read_csv(cfg.meta_csv)
        if cfg.meta_csv.suffix.lower() == ".csv"
        else pd.read_parquet(cfg.meta_csv)
    )
    if cfg.split is not None and "split_group" in meta.columns:
        meta = meta[
            meta["split_group"].astype(str).str.lower() == cfg.split.lower()
        ].copy()

    meta = meta.assign(
        patient_id=meta["patient_id"].astype(str),
        exam_id=meta["exam_id"].astype(str),
    )

    if not PHENOTYPE_CSV.exists():
        raise FileNotFoundError(f"phenotype CSV not found: {PHENOTYPE_CSV}")
    if not PATIENT_KEY_CSV.exists():
        raise FileNotFoundError(f"patient key CSV not found: {PATIENT_KEY_CSV}")

    phenotype = pd.read_csv(PHENOTYPE_CSV)
    phenotype["MRN"] = phenotype["MRN"].astype(str).str.strip().str.zfill(8)
    key = pd.read_csv(PATIENT_KEY_CSV)
    key["MRN"] = key["MRN"].astype(str).str.strip().str.zfill(8)
    key["AnonymousID"] = key["AnonymousID"].astype(str).str.strip()

    phenotype = phenotype.merge(
        key[["MRN", "AnonymousID"]],
        on="MRN",
        how="left",
    )
    phenotype = phenotype.rename(columns={"AnonymousID": "patient_id"})
    keep_cols = ["patient_id"] + [
        col for col in PHENOTYPE_COLUMNS if col in phenotype.columns
    ]
    phenotype_subset = phenotype[keep_cols].copy()

    meta = meta.merge(phenotype_subset, on="patient_id", how="left")

    meta["race_category"] = (
        _assign_race_category(meta["RaceEthnic"])
        if "RaceEthnic" in meta.columns
        else "Unknown"
    )

    exams_parquet = cfg.meta_csv.parent.parent / "sot" / "exams.parquet"
    if exams_parquet.exists():
        exams = pd.read_parquet(exams_parquet)
        exams = exams.assign(
            patient_id=exams["patient_id"].astype(str),
            exam_id=exams["exam_id"].astype(str),
        )
        meta = meta.merge(
            exams[["patient_id", "exam_id", "study_date"]],
            on=["patient_id", "exam_id"],
            how="left",
        )

    if "dob" in meta.columns:
        dob_parsed = pd.to_datetime(meta["dob"], format="%d%b%Y", errors="coerce")
        study_date_parsed = pd.to_datetime(meta.get("study_date"), errors="coerce")
        meta["age_at_exam"] = (study_date_parsed - dob_parsed).dt.days / 365.25
    else:
        meta["age_at_exam"] = np.nan

    meta["hr_status"] = meta.apply(_infer_hr_status, axis=1)
    meta["receptor_subtype"] = meta.apply(_infer_receptor_subtype, axis=1)
    meta["tumor_grade_group"] = meta.apply(_grade_category_from_row, axis=1)

    return meta


def _load_auto_filter_names(qc_cfg: QCFilterConfig) -> list[str]:
    """load ordered list of auto-filter names from config or qc_auto_exclude.json."""
    if not qc_cfg.enable_auto_filters:
        return []

    from prima.qc_filters import load_auto_filter_names

    return load_auto_filter_names(
        config_path=qc_cfg.auto_filter_config,
        explicit_filters=list(qc_cfg.auto_filters) if qc_cfg.auto_filters else None,
    )


def _load_qc_status_map(qc_file: Path | None) -> dict[str, str]:
    """load qc_status.json as exam_id -> normalized status."""
    if qc_file is None or not qc_file.exists():
        return {}
    with open(qc_file) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"QC file must contain a JSON object: {qc_file}")

    out: dict[str, str] = {}
    for exam_id, status in payload.items():
        out[str(exam_id)] = str(status).strip().lower()
    return out


def _normalize_annotation_filter(values: tuple[str, ...] | None) -> tuple[str, ...]:
    """normalize annotation filter tag list to canonical lowercase tags."""
    if values is None:
        return tuple()
    normalized: list[str] = []
    for value in values:
        tag = _canonical_annotation_tag(value)
        if tag and tag not in normalized:
            normalized.append(tag)
    return tuple(normalized)


def _load_annotations_map(annotations_file: Path | None) -> dict[str, set[str]]:
    """load annotations.json as exam_id -> normalized tag set."""
    if annotations_file is None or not annotations_file.exists():
        return {}
    with open(annotations_file) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(
            f"annotations file must contain a JSON object: {annotations_file}"
        )

    out: dict[str, set[str]] = {}
    for exam_id, tags in payload.items():
        if not isinstance(tags, list):
            continue
        normalized_tags = {
            _canonical_annotation_tag(tag)
            for tag in tags
            if _canonical_annotation_tag(tag)
        }
        if normalized_tags:
            out[str(exam_id)] = normalized_tags
    return out


def _build_exam_filter(
    exam_ids: set[str], cfg: Config, views_path: Path, tags_path: Path
) -> tuple[set[str], dict[str, Any]]:
    """build and apply QC/auto/annotation filters on a set of exam IDs."""
    active_exam_ids = set(str(eid) for eid in exam_ids)
    summary: dict[str, Any] = {"initial_exams": len(active_exam_ids), "steps": []}

    def apply_exclude(name: str, candidates: set[str]) -> None:
        before = len(active_exam_ids)
        active_exam_ids.difference_update(candidates)
        excluded = before - len(active_exam_ids)
        summary["steps"].append(
            {
                "mode": "exclude",
                "name": name,
                "candidate_exams": len(candidates),
                "excluded": excluded,
                "remaining": len(active_exam_ids),
            }
        )

    def apply_include(name: str, candidates: set[str]) -> None:
        before = len(active_exam_ids)
        active_exam_ids.intersection_update(candidates)
        excluded = before - len(active_exam_ids)
        summary["steps"].append(
            {
                "mode": "include",
                "name": name,
                "candidate_exams": len(candidates),
                "excluded": excluded,
                "remaining": len(active_exam_ids),
            }
        )

    qc_cfg = cfg.qc_filters

    auto_filter_names = _load_auto_filter_names(qc_cfg)
    if auto_filter_names:
        auto_by_filter = compute_auto_filter_sets(
            views_path, tags_path, filter_names=auto_filter_names
        )
        for filter_name in auto_filter_names:
            apply_exclude(
                f"auto_filter:{filter_name}", auto_by_filter.get(filter_name, set())
            )

    qc_status_map = _load_qc_status_map(qc_cfg.qc_file)
    if qc_cfg.include_statuses is not None:
        include_set = set(qc_cfg.include_statuses)
        included_exam_ids = {
            eid for eid, status in qc_status_map.items() if status in include_set
        }
        apply_include(
            f"qc_status:include={','.join(sorted(include_set))}", included_exam_ids
        )
    if qc_cfg.exclude_statuses:
        exclude_set = set(qc_cfg.exclude_statuses)
        excluded_exam_ids = {
            eid for eid, status in qc_status_map.items() if status in exclude_set
        }
        apply_exclude(
            f"qc_status:exclude={','.join(sorted(exclude_set))}", excluded_exam_ids
        )

    annotation_map = _load_annotations_map(qc_cfg.annotations_file)
    include_any = set(qc_cfg.annotation_include_any)
    include_all = set(qc_cfg.annotation_include_all)
    exclude_any = set(qc_cfg.annotation_exclude_any)
    exclude_all = set(qc_cfg.annotation_exclude_all)

    if include_any:
        include_any_ids = {
            eid for eid, tags in annotation_map.items() if tags & include_any
        }
        apply_include(
            f"annotations:include_any={','.join(sorted(include_any))}", include_any_ids
        )
    if include_all:
        include_all_ids = {
            eid for eid, tags in annotation_map.items() if include_all.issubset(tags)
        }
        apply_include(
            f"annotations:include_all={','.join(sorted(include_all))}", include_all_ids
        )
    if exclude_any:
        exclude_any_ids = {
            eid for eid, tags in annotation_map.items() if tags & exclude_any
        }
        apply_exclude(
            f"annotations:exclude_any={','.join(sorted(exclude_any))}", exclude_any_ids
        )
    if exclude_all:
        exclude_all_ids = {
            eid for eid, tags in annotation_map.items() if exclude_all.issubset(tags)
        }
        apply_exclude(
            f"annotations:exclude_all={','.join(sorted(exclude_all))}", exclude_all_ids
        )

    summary["final_exams"] = len(active_exam_ids)
    summary["total_excluded"] = summary["initial_exams"] - summary["final_exams"]
    return active_exam_ids, summary


def _build_filter_cutflow_df(
    df_exam: pd.DataFrame, cfg: Config, views_path: Path, tags_path: Path
) -> pd.DataFrame:
    """Compute exam/patient cutflow under current QC+annotation config."""
    exam_to_patient = (
        df_exam[["exam_id", "patient_id"]]
        .drop_duplicates(subset=["exam_id"])
        .assign(
            exam_id=lambda d: d["exam_id"].astype(str),
            patient_id=lambda d: d["patient_id"].astype(str),
        )
        .set_index("exam_id")["patient_id"]
        .to_dict()
    )
    active_exam_ids = set(exam_to_patient.keys())
    initial_patients = len(set(exam_to_patient.values()))
    rows: list[dict[str, Any]] = [
        {
            "step": 0,
            "mode": "start",
            "name": "start",
            "candidate_exams": len(active_exam_ids),
            "candidate_patients": initial_patients,
            "excluded_exams": 0,
            "excluded_patients": 0,
            "remaining_exams": len(active_exam_ids),
            "remaining_patients": initial_patients,
        }
    ]

    def _apply(mode: str, name: str, candidates: set[str]) -> None:
        nonlocal active_exam_ids
        before_exams = len(active_exam_ids)
        before_patients = len({exam_to_patient[e] for e in active_exam_ids})
        if mode == "exclude":
            active_exam_ids = active_exam_ids - set(candidates)
        else:
            active_exam_ids = active_exam_ids & set(candidates)
        after_exams = len(active_exam_ids)
        after_patients = len({exam_to_patient[e] for e in active_exam_ids})
        candidate_patients = len(
            {exam_to_patient[e] for e in candidates if e in exam_to_patient}
        )
        rows.append(
            {
                "step": len(rows),
                "mode": mode,
                "name": name,
                "candidate_exams": len(candidates),
                "candidate_patients": candidate_patients,
                "excluded_exams": before_exams - after_exams,
                "excluded_patients": before_patients - after_patients,
                "remaining_exams": after_exams,
                "remaining_patients": after_patients,
            }
        )

    qc_cfg = cfg.qc_filters
    auto_filter_names = _load_auto_filter_names(qc_cfg)
    if auto_filter_names:
        auto_by_filter = compute_auto_filter_sets(
            views_path, tags_path, filter_names=auto_filter_names
        )
        for filter_name in auto_filter_names:
            _apply(
                "exclude",
                f"auto_filter:{filter_name}",
                auto_by_filter.get(filter_name, set()),
            )

    qc_status_map = _load_qc_status_map(qc_cfg.qc_file)
    if qc_cfg.include_statuses is not None:
        include_set = set(qc_cfg.include_statuses)
        included_exam_ids = {
            eid for eid, status in qc_status_map.items() if status in include_set
        }
        _apply(
            "include",
            f"qc_status:include={','.join(sorted(include_set))}",
            included_exam_ids,
        )
    if qc_cfg.exclude_statuses:
        exclude_set = set(qc_cfg.exclude_statuses)
        excluded_exam_ids = {
            eid for eid, status in qc_status_map.items() if status in exclude_set
        }
        _apply(
            "exclude",
            f"qc_status:exclude={','.join(sorted(exclude_set))}",
            excluded_exam_ids,
        )

    annotation_map = _load_annotations_map(qc_cfg.annotations_file)
    include_any = set(qc_cfg.annotation_include_any)
    include_all = set(qc_cfg.annotation_include_all)
    exclude_any = set(qc_cfg.annotation_exclude_any)
    exclude_all = set(qc_cfg.annotation_exclude_all)
    if include_any:
        include_any_ids = {
            eid for eid, tags in annotation_map.items() if tags & include_any
        }
        _apply(
            "include",
            f"annotations:include_any={','.join(sorted(include_any))}",
            include_any_ids,
        )
    if include_all:
        include_all_ids = {
            eid for eid, tags in annotation_map.items() if include_all.issubset(tags)
        }
        _apply(
            "include",
            f"annotations:include_all={','.join(sorted(include_all))}",
            include_all_ids,
        )
    if exclude_any:
        exclude_any_ids = {
            eid for eid, tags in annotation_map.items() if tags & exclude_any
        }
        _apply(
            "exclude",
            f"annotations:exclude_any={','.join(sorted(exclude_any))}",
            exclude_any_ids,
        )
    if exclude_all:
        exclude_all_ids = {
            eid for eid, tags in annotation_map.items() if exclude_all.issubset(tags)
        }
        _apply(
            "exclude",
            f"annotations:exclude_all={','.join(sorted(exclude_all))}",
            exclude_all_ids,
        )

    return pd.DataFrame(rows)


def _load_exam_data(cfg: Config) -> ExamData:
    """load predictions merged with metadata, aggregated per exam"""

    pred = pd.read_csv(cfg.pred_csv)
    if {"patient_id", "exam_id"}.issubset(pred.columns):
        pid = pred["patient_id"].astype(str)
        eid = pred["exam_id"].astype(str)
    elif "patient_exam_id" in pred.columns:
        pid, eid = _split_patient_exam_id(pred["patient_exam_id"])
    else:
        raise KeyError(
            "predictions must have 'patient_exam_id' or both 'patient_id' and 'exam_id'"
        )
    pred = pred.assign(patient_id=pid, exam_id=eid)

    meta_all = (
        pd.read_csv(cfg.meta_csv)
        if cfg.meta_csv.suffix.lower() == ".csv"
        else pd.read_parquet(cfg.meta_csv)
    )
    req = {"patient_id", "exam_id", "years_to_cancer", "years_to_last_followup"}
    if not req.issubset(set(meta_all.columns)):
        missing = sorted(req - set(meta_all.columns))
        raise KeyError(f"metadata missing columns: {missing}")
    meta_all = meta_all.assign(
        patient_id=meta_all["patient_id"].astype(str),
        exam_id=meta_all["exam_id"].astype(str),
    )

    if cfg.split is not None and "split_group" in meta_all.columns:
        meta_eval = meta_all[
            meta_all["split_group"].astype(str).str.lower() == cfg.split.lower()
        ].copy()
    else:
        meta_eval = meta_all.copy()

    cols_to_merge = list(req)
    if "split_group" in meta_eval.columns and "split_group" not in cols_to_merge:
        cols_to_merge.append("split_group")

    df = pred.merge(
        meta_eval[cols_to_merge],
        on=["patient_id", "exam_id"],
        how="inner",
    )

    pred_cols = cfg.colmap or _detect_pred_cols(df)
    agg_dict = {col: "mean" for col in pred_cols.values()}
    agg_dict["years_to_cancer"] = "first"
    agg_dict["years_to_last_followup"] = "first"
    if "split_group" in df.columns:
        agg_dict["split_group"] = "first"

    df_exam = df.groupby(["patient_id", "exam_id"], as_index=False).agg(agg_dict)
    df_exam_unfiltered = df_exam.copy()

    # QC/auto/annotation filtering
    views_path = cfg.pred_csv.parent.parent / "sot" / "views.parquet"
    tags_path = cfg.pred_csv.parent.parent / "sot" / "dicom_tags.parquet"
    base_exam_ids = set(df_exam["exam_id"].astype(str).unique())
    kept_exam_ids, filter_summary = _build_exam_filter(
        base_exam_ids, cfg, views_path, tags_path
    )

    auto_steps = [
        step
        for step in filter_summary["steps"]
        if step["name"].startswith("auto_filter:")
    ]
    if auto_steps:
        print("Auto-filter cutflow:")
        for idx, step in enumerate(auto_steps, start=1):
            filter_name = step["name"].split(":", 1)[1]
            print(
                f"  {idx}. {filter_name}: matched {step['candidate_exams']} exams, "
                f"excluded {step['excluded']} new, remaining {step['remaining']}"
            )
        auto_total_excluded = sum(step["excluded"] for step in auto_steps)
        print(f"  total newly excluded by auto filters: {auto_total_excluded}")
        print()

    for step in filter_summary["steps"]:
        if step["name"].startswith("auto_filter:"):
            continue
        if step["excluded"] > 0 or step["mode"] == "include":
            print(
                f"Filter {step['name']}: excluded {step['excluded']} exams "
                f"(remaining {step['remaining']})"
            )
    if filter_summary["total_excluded"] > 0:
        print(
            f"Total filtered exams: {filter_summary['total_excluded']} "
            f"(from {filter_summary['initial_exams']} to {filter_summary['final_exams']})"
        )
        print()

    df_exam = df_exam[df_exam["exam_id"].astype(str).isin(kept_exam_ids)].copy()
    df = df[df["exam_id"].astype(str).isin(kept_exam_ids)].copy()
    exam_pairs = frozenset(
        zip(df_exam["patient_id"].astype(str), df_exam["exam_id"].astype(str))
    )

    return ExamData(
        merged_views=df,
        exam_level=df_exam,
        exam_level_unfiltered=df_exam_unfiltered,
        meta_all=meta_all,
        meta_eval=meta_eval,
        pred_cols=pred_cols,
        n_pred_rows=len(pred),
        filter_summary=filter_summary,
        exam_pairs=exam_pairs,
    )


def _split_patient_exam_id(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """split 'patient_exam_id' formatted as '<patient_id>\t<exam_id>' into two series"""
    pe = series.astype(str).str.split("\t", n=1, expand=True)
    if pe.shape[1] != 2:
        raise ValueError("patient_exam_id not formatted as 'patient\texam'")
    return pe[0], pe[1]


def _detect_pred_cols(df: pd.DataFrame) -> dict[int, str]:
    """auto-detect prediction columns keyed by horizon in years

    looks for names with horizon numbers next to 'year'/'yr', optionally prefixed with 'risk'/'pred'/'prob'
    also handles patterns like '1year', '2year', etc directly
    """
    cols = {}
    for c in df.columns:
        cl = c.lower()
        # regexes to catch '1year', '1_year', '1_year_risk', 'year1', '1yr', 'yr5', 'risk_5_year', 'pred_1year', etc
        # [\s_] matches whitespace or underscore
        m = re.search(
            r"(?:(\d+)[\s_]*year|year[\s_]*(\d+)|(\d+)[\s_]*yr|yr[\s_]*(\d+))", cl
        )
        if m:
            h = int(next(g for g in m.groups() if g))
            # skip if it's clearly not a prediction column (e.g., 'years_to_cancer')
            if any(
                skip in cl for skip in ("years_to", "year_to", "followup", "follow_up")
            ):
                continue
            cols[h] = c
    if not cols:
        # print available columns for debugging
        numeric_cols = [
            c
            for c in df.columns
            if df[c].dtype in (np.float64, np.float32, np.int64, np.int32)
        ]
        raise KeyError(
            f'could not auto-detect prediction columns; set config \'colmap\' (e.g., {{"1": "pred_1year"}})\n'
            f"Available numeric columns: {numeric_cols}"
        )
    return dict(sorted(cols.items()))


def _horizon_auc_arrays(
    df_exam: pd.DataFrame, horizon: int, score_col: str, cfg: Config
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, int]:
    """Prepare y/s/weights/patient_id arrays for horizon-specific AUC."""
    tmp = df_exam[
        ["patient_id", "years_to_cancer", "years_to_last_followup", score_col]
    ].copy()
    tmp = tmp.rename(columns={score_col: "score"})
    ytc = pd.to_numeric(tmp["years_to_cancer"], errors="coerce").to_numpy()
    ylf = pd.to_numeric(tmp["years_to_last_followup"], errors="coerce").to_numpy()
    case = np.less_equal(ytc, horizon)
    ctrl = np.logical_and(np.greater(ytc, horizon), np.greater_equal(ylf, horizon))
    include = np.logical_or(case, ctrl)
    excluded = int((~include).sum())

    eligible = tmp.loc[include].copy()
    eligible["case"] = case[include].astype(np.int32)

    if cfg.auc_mode == "random_exam":
        eligible = (
            eligible.groupby("patient_id", group_keys=False)
            .sample(n=1, random_state=cfg.random_exam_seed + int(horizon))
            .reset_index(drop=True)
        )
        weights = None
    else:
        # each patient contributes total weight 1 across all their eligible exams
        n_exam_per_patient = eligible.groupby("patient_id")["patient_id"].transform(
            "size"
        )
        weights = (1.0 / n_exam_per_patient.astype(float)).to_numpy()

    y = eligible["case"].astype(np.int32).to_numpy()
    s = pd.to_numeric(eligible["score"], errors="coerce").to_numpy()
    patient_ids = eligible["patient_id"].astype(str).to_numpy()
    return y, s, weights, patient_ids, excluded


def _auc_point_estimate(
    y_true: np.ndarray, y_score: np.ndarray, sample_weight: np.ndarray | None = None
) -> float:
    """Compute AUC point estimate with optional sample weights."""
    if len(y_true) == 0 or not (y_true.any() and (~y_true.astype(bool)).any()):
        return float("nan")
    return float(roc_auc_score(y_true, y_score, sample_weight=sample_weight))


def summarize(cfg: Config) -> tuple[pd.DataFrame, ExamData]:
    """compute per-horizon summary: cases, controls, excluded, auc, prevalence"""

    exam_data = _load_exam_data(cfg)
    df = exam_data.merged_views
    df_exam = exam_data.exam_level
    meta_all = exam_data.meta_all
    meta_eval = exam_data.meta_eval
    pred_cols = exam_data.pred_cols

    # print summary statistics
    print("=== Data Summary ===")
    print(f"Predictions CSV rows: {exam_data.n_pred_rows:,}")
    print(f"Metadata CSV rows (total): {len(meta_all):,}")
    print(f"Metadata CSV rows (evaluation set): {len(meta_eval):,}")
    print(f"Rows after merge: {len(df):,}")
    print(f"Unique patients: {df['patient_id'].nunique():,}")
    unique_exams = df[["patient_id", "exam_id"]].drop_duplicates().shape[0]
    print(f"Unique exams: {unique_exams:,}")
    if len(df) > unique_exams:
        print("  (multiple rows per exam - likely multiple views)")
        views_per_exam = df.groupby(["patient_id", "exam_id"]).size()
        print(
            f"  Views per exam: min={views_per_exam.min()}, max={views_per_exam.max()}, mean={views_per_exam.mean():.2f}"
        )
    print(f"AUC mode: {cfg.auc_mode}")
    print()

    rows = []
    for h, col in pred_cols.items():
        y, s, w, _pids, excluded = _horizon_auc_arrays(df_exam, h, col, cfg)
        auc = _auc_point_estimate(y, s, sample_weight=w)
        rows.append(
            {
                "horizon_years": h,
                "n": int(len(y)),
                "cases": int(y.sum()),
                "controls": int((1 - y).sum()),
                "excluded": excluded,
                "prevalence": float(y.sum() / max(1, len(y))),
                "auc": auc,
                "auc_mode": cfg.auc_mode,
            }
        )

    out = pd.DataFrame(rows).sort_values("horizon_years").reset_index(drop=True)
    return out, exam_data


def _load_config(config_path: Path) -> Config:
    """load OmegaConf config and return normalized Config dataclass."""
    config_path = config_path.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    cfg_omega = OmegaConf.load(config_path)
    raw_config = OmegaConf.to_container(cfg_omega, resolve=False)
    resolved_input = OmegaConf.to_container(cfg_omega, resolve=True)
    if not isinstance(raw_config, dict) or not isinstance(resolved_input, dict):
        raise ValueError("config root must be a mapping/object")

    config_dir = config_path.parent
    out_dir = _resolve_config_path(resolved_input.get("out_dir"), config_dir)
    if out_dir is None:
        raise ValueError("config key 'out_dir' is required")

    pred_csv = _resolve_config_path(resolved_input.get("pred_csv"), config_dir)
    if pred_csv is None:
        pred_csv = out_dir / "validation_output.csv"
    meta_csv = _resolve_config_path(resolved_input.get("meta_csv"), config_dir)
    if meta_csv is None:
        meta_csv = out_dir / "mirai_manifest.csv"
    out_json = _resolve_config_path(resolved_input.get("out_json"), config_dir)
    if out_json is None:
        out_json = out_dir / "summary.json"

    split_value = resolved_input.get("split", "test")
    split = None if split_value is None else str(split_value)
    colmap = _parse_colmap(resolved_input.get("colmap"))
    cindex_col_value = resolved_input.get("cindex_col")
    cindex_col = None if cindex_col_value is None else str(cindex_col_value)
    kfold_value = resolved_input.get("kfold")
    kfold = None if kfold_value is None else int(kfold_value)
    if kfold is not None and kfold < 2:
        raise ValueError("config key 'kfold' must be >= 2 when provided")
    auc_mode = str(resolved_input.get("auc_mode", "patient_weighted")).strip().lower()
    if auc_mode not in {"patient_weighted", "random_exam"}:
        raise ValueError(
            "config key 'auc_mode' must be 'patient_weighted' or 'random_exam'"
        )
    random_exam_seed = int(resolved_input.get("random_exam_seed", 42))

    if not pred_csv.exists():
        raise FileNotFoundError(f"prediction CSV not found: {pred_csv}")
    if not meta_csv.exists():
        raise FileNotFoundError(f"metadata CSV not found: {meta_csv}")

    qc_cfg_raw = resolved_input.get("qc_filters")
    if qc_cfg_raw is None:
        qc_cfg_raw = {}
    if not isinstance(qc_cfg_raw, dict):
        raise ValueError("config key 'qc_filters' must be a mapping/object")

    include_statuses = _normalize_string_tuple(
        qc_cfg_raw.get("include_statuses"), lowercase=True
    )
    if include_statuses is not None and len(include_statuses) == 0:
        include_statuses = None
    exclude_statuses = _normalize_string_tuple(
        qc_cfg_raw.get("exclude_statuses", ["bad", "review"]), lowercase=True
    )
    if exclude_statuses is None:
        exclude_statuses = tuple()

    qc_file = _resolve_config_path(qc_cfg_raw.get("qc_file"), config_dir)
    annotations_file = _resolve_config_path(
        qc_cfg_raw.get("annotations_file"), config_dir
    )
    if annotations_file is None and qc_file is not None:
        annotations_file = qc_file.parent / "annotations.json"

    include_any = _normalize_annotation_filter(
        _normalize_string_tuple(qc_cfg_raw.get("annotation_include_any"))
    )
    include_all = _normalize_annotation_filter(
        _normalize_string_tuple(qc_cfg_raw.get("annotation_include_all"))
    )
    exclude_any = _normalize_annotation_filter(
        _normalize_string_tuple(qc_cfg_raw.get("annotation_exclude_any"))
    )
    exclude_all = _normalize_annotation_filter(
        _normalize_string_tuple(qc_cfg_raw.get("annotation_exclude_all"))
    )

    auto_filter_config = _resolve_config_path(
        qc_cfg_raw.get("auto_filter_config"), config_dir
    )
    if auto_filter_config is None:
        auto_filter_config = (Path.cwd() / "data" / "qc_auto_exclude.json").resolve()

    auto_filters = _normalize_string_tuple(
        qc_cfg_raw.get("auto_filters"), lowercase=True
    )
    enable_auto_filters = bool(qc_cfg_raw.get("enable_auto_filters", True))

    qc_filters = QCFilterConfig(
        qc_file=qc_file,
        include_statuses=include_statuses,
        exclude_statuses=exclude_statuses,
        annotations_file=annotations_file,
        annotation_include_any=include_any,
        annotation_include_all=include_all,
        annotation_exclude_any=exclude_any,
        annotation_exclude_all=exclude_all,
        enable_auto_filters=enable_auto_filters,
        auto_filter_config=auto_filter_config,
        auto_filters=auto_filters,
    )

    resolved_config: dict[str, Any] = {
        "out_dir": str(out_dir),
        "pred_csv": str(pred_csv),
        "meta_csv": str(meta_csv),
        "out_json": str(out_json),
        "split": split,
        "colmap": {str(k): v for k, v in colmap.items()} if colmap else None,
        "cindex_col": cindex_col,
        "kfold": kfold,
        "auc_mode": auc_mode,
        "random_exam_seed": random_exam_seed,
        "qc_filters": {
            "qc_file": str(qc_file) if qc_file else None,
            "include_statuses": list(include_statuses)
            if include_statuses is not None
            else None,
            "exclude_statuses": list(exclude_statuses),
            "annotations_file": str(annotations_file) if annotations_file else None,
            "annotation_include_any": list(include_any),
            "annotation_include_all": list(include_all),
            "annotation_exclude_any": list(exclude_any),
            "annotation_exclude_all": list(exclude_all),
            "enable_auto_filters": enable_auto_filters,
            "auto_filter_config": str(auto_filter_config)
            if auto_filter_config
            else None,
            "auto_filters": list(auto_filters) if auto_filters is not None else None,
        },
    }

    return Config(
        pred_csv=pred_csv,
        meta_csv=meta_csv,
        out_json=out_json,
        split=split,
        colmap=colmap,
        cindex_col=cindex_col,
        kfold=kfold,
        auc_mode=auc_mode,
        random_exam_seed=random_exam_seed,
        qc_filters=qc_filters,
        config_path=config_path,
        raw_config=raw_config,
        resolved_config=resolved_config,
    )


def parse_args() -> Config:
    """parse CLI args (config-only)."""
    p = argparse.ArgumentParser(
        description="Analyze Mirai outputs using OmegaConf config"
    )
    p.add_argument(
        "config",
        type=Path,
        help="OmegaConf YAML/JSON config path",
    )
    args = p.parse_args()
    return _load_config(args.config)


def plot_hiddens_pca(
    cfg: Config,
    exam_data: ExamData,
    hiddens_path: Path,
    color_cols: list[str] | None = None,
    n_components: int = 2,
) -> None:
    """PCA of Mirai hidden vectors, colored by phenotype labels.

    Aggregates view-level hiddens to exam level (mean), merges with phenotype,
    runs PCA, and saves scatter plots colored by each color_col.
    """
    data = np.load(hiddens_path, allow_pickle=True)
    hiddens = data["hiddens"]
    paths = data["paths"].tolist()

    meta = exam_data.meta_all
    if "file_path" not in meta.columns:
        print("[warn] manifest has no file_path; cannot map hiddens to exams")
        return
    # build file_path -> (patient_id, exam_id); also exam_id -> patient_id for fallback
    path_to_exam = (
        meta[["file_path", "patient_id", "exam_id"]].drop_duplicates().astype(str)
    )
    path_lookup = path_to_exam.set_index("file_path")
    exam_to_patient = (
        meta[["patient_id", "exam_id"]]
        .drop_duplicates()
        .astype(str)
        .set_index("exam_id")
    )

    def _path_to_exam_id(path_str: str) -> str | None:
        """extract exam_id from zarr path like .../EXAM_ID.zarr#VIEW"""
        if ".zarr#" in path_str:
            base = path_str.split("#")[0]
            return base.split("/")[-1].replace(".zarr", "")
        return None

    rows = []
    for i, path in enumerate(paths):
        path_str = str(path).strip()
        matched = False
        for view_path in path_str.split("\t"):
            view_path = view_path.strip()
            if not view_path:
                continue
            if view_path in path_lookup.index:
                row = path_lookup.loc[view_path]
                rows.append(
                    {
                        "patient_id": row["patient_id"],
                        "exam_id": row["exam_id"],
                        "idx": i,
                    }
                )
                matched = True
                break
        if not matched:
            first_view = path_str.split("\t")[0].strip() if path_str else ""
            exam_id = _path_to_exam_id(first_view or path_str)
            if exam_id and exam_id in exam_to_patient.index:
                pid = exam_to_patient.loc[exam_id, "patient_id"]
                rows.append({"patient_id": pid, "exam_id": exam_id, "idx": i})
    if not rows:
        print("[warn] no hiddens paths matched manifest file_path or exam_id")
        return
    path_df = pd.DataFrame(rows)

    def mean_hidden(idx_series):
        return hiddens[idx_series.values].mean(axis=0)

    exam_vecs = (
        path_df.groupby(["patient_id", "exam_id"])["idx"]
        .apply(mean_hidden)
        .reset_index()
    )
    exam_vecs.columns = ["patient_id", "exam_id", "vec"]
    X = np.stack(exam_vecs["vec"].values)
    if X.ndim == 3:
        X = X.mean(axis=1)
    elif X.ndim != 2:
        raise ValueError(
            f"Expected 2D or 3D hiddens after aggregation, got shape {X.shape}"
        )
    exam_keys = exam_vecs[["patient_id", "exam_id"]].copy()

    allowed = exam_data.exam_pairs
    mask = exam_keys.apply(
        lambda r: (str(r["patient_id"]), str(r["exam_id"])) in allowed, axis=1
    )
    exam_keys = exam_keys[mask].copy()
    X = X[mask]

    enriched = _load_enriched_manifest(cfg)
    enriched = enriched.sort_values(["patient_id", "exam_id"]).drop_duplicates(
        subset=["patient_id", "exam_id"], keep="first"
    )
    merged = exam_keys.merge(
        enriched,
        on=["patient_id", "exam_id"],
        how="left",
    )

    X_scaled = StandardScaler().fit_transform(X)
    n_pc = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_pc)
    pc = pca.fit_transform(X_scaled)
    merged["PC1"] = pc[:, 0]
    merged["PC2"] = pc[:, 1] if n_pc > 1 else np.nan

    if color_cols is None:
        color_cols = ["subtype", "receptor_subtype", "CaseControl", "race_category"]
    color_cols = [c for c in color_cols if c in merged.columns]

    out_dir = cfg.pred_csv.parent
    for col in color_cols:
        plot_df = merged.dropna(subset=[col])
        if len(plot_df) < 2:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        for label, grp in plot_df.groupby(col):
            ax.scatter(
                grp["PC1"],
                grp["PC2"],
                label=str(label)[:30],
                alpha=0.6,
                s=12,
            )
        ax.set_xlabel("PC1 ({:.1f}%)".format(pca.explained_variance_ratio_[0] * 100))
        ax.set_ylabel(
            "PC2 ({:.1f}%)".format(pca.explained_variance_ratio_[1] * 100)
            if n_pc > 1
            else ""
        )
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        out_path = out_dir / f"hiddens_pca_{col}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


def _build_surv_arrays(meta: pd.DataFrame) -> np.ndarray:
    """return Surv array from meta with columns years_to_cancer and years_to_last_followup

    event = cancer observed before or at last follow-up; time = min(ytc, ylf)
    """
    ytc = meta["years_to_cancer"].astype(float).to_numpy()
    ylf = meta["years_to_last_followup"].astype(float).to_numpy()
    event = np.isfinite(ytc) & (ytc <= ylf)
    time = np.where(event, ytc, ylf)
    return Surv.from_arrays(event=event.astype(bool), time=time.astype(float))


def survival_metrics(cfg: Config, exam_data: ExamData) -> dict:
    """compute censoring-adjusted metrics: Uno's C (IPCW), time-dependent AUC, IBS

    uses all available data to estimate censoring via Kaplan–Meier for IPCW
    """
    df_exam = exam_data.exam_level
    pred_cols = exam_data.pred_cols

    meta_all_exam = exam_data.meta_all.groupby(
        ["patient_id", "exam_id"], as_index=False
    ).agg({"years_to_cancer": "first", "years_to_last_followup": "first"})

    # use all available data for IPCW censoring estimation
    y_train = _build_surv_arrays(meta_all_exam)
    y_test = _build_surv_arrays(df_exam)

    # prediction columns / times
    times = np.array(sorted(pred_cols.keys()), dtype=float)

    # matrix of risk estimates shape (n_test, n_times)
    risk = np.stack(
        [df_exam[pred_cols[h]].astype(float).to_numpy() for h in times], axis=1
    )

    # time-dependent AUC (Uno 2007) and weighted mean AUC (Lambert & Chevret 2014)
    auc_t, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk, times)

    # Uno's IPCW C-index at tau = max(times); pick a single risk column
    if cfg.cindex_col is not None:
        risk_for_c = df_exam[cfg.cindex_col].astype(float).to_numpy()
    else:
        risk_for_c = df_exam[pred_cols[int(times.max())]].astype(float).to_numpy()
    c_ipcw, n_conc, n_disc, n_trisk, n_ttime = concordance_index_ipcw(
        y_train, y_test, risk_for_c, tau=float(times.max())
    )

    # Brier score & IBS require survival probabilities at times
    surv_prob = 1.0 - risk
    _, bs_t = brier_score(y_train, y_test, surv_prob, times)
    ibs = integrated_brier_score(y_train, y_test, surv_prob, times)

    return {
        "times": times.tolist(),
        "auc_t": auc_t.tolist(),
        "mean_auc": float(mean_auc),
        "uno_c": float(c_ipcw),
        "pairs": {
            "concordant": int(n_conc),
            "discordant": int(n_disc),
            "tied_risk": int(n_trisk),
            "tied_time": int(n_ttime),
        },
        "brier_t": bs_t.tolist(),
        "ibs": float(ibs),
    }


def kfold_survival_metrics(cfg: Config, exam_data: ExamData) -> dict:
    """k-fold CV for survival metrics to assess IPCW sensitivity

    splits at patient level (not exam level) to avoid leakage. For each fold:
    - train fold estimates IPCW censoring distribution
    - test fold evaluates metrics using that IPCW estimate
    """
    df_exam = exam_data.exam_level
    pred_cols = exam_data.pred_cols

    # get unique patients for splitting
    patients = df_exam["patient_id"].unique()
    kf = KFold(n_splits=cfg.kfold, shuffle=True, random_state=42)

    # prediction columns / times
    times = np.array(sorted(pred_cols.keys()), dtype=float)

    fold_results = []
    for fold_idx, (train_pat_idx, test_pat_idx) in enumerate(kf.split(patients)):
        train_patients = patients[train_pat_idx]
        test_patients = patients[test_pat_idx]

        df_train = df_exam[df_exam["patient_id"].isin(train_patients)].copy()
        df_test = df_exam[df_exam["patient_id"].isin(test_patients)].copy()

        if len(df_train) == 0 or len(df_test) == 0:
            continue

        y_train = _build_surv_arrays(df_train)
        y_test = _build_surv_arrays(df_test)

        # risk matrix for test set
        risk = np.stack(
            [df_test[pred_cols[h]].astype(float).to_numpy() for h in times], axis=1
        )

        # time-dependent AUC
        auc_t, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk, times)

        # Uno's C-index
        if cfg.cindex_col is not None:
            risk_for_c = df_test[cfg.cindex_col].astype(float).to_numpy()
        else:
            risk_for_c = df_test[pred_cols[int(times.max())]].astype(float).to_numpy()
        c_ipcw, n_conc, n_disc, n_trisk, n_ttime = concordance_index_ipcw(
            y_train, y_test, risk_for_c, tau=float(times.max())
        )

        # Brier score & IBS
        surv_prob = 1.0 - risk
        _, bs_t = brier_score(y_train, y_test, surv_prob, times)
        ibs = integrated_brier_score(y_train, y_test, surv_prob, times)

        fold_results.append(
            {
                "fold": fold_idx,
                "n_train": len(df_train),
                "n_test": len(df_test),
                "auc_t": auc_t.tolist(),
                "mean_auc": float(mean_auc),
                "uno_c": float(c_ipcw),
                "brier_t": bs_t.tolist(),
                "ibs": float(ibs),
            }
        )

    # aggregate across folds
    n_folds = len(fold_results)
    if n_folds == 0:
        raise ValueError("no valid folds; check data splitting")

    mean_aucs = np.array([r["mean_auc"] for r in fold_results])
    uno_cs = np.array([r["uno_c"] for r in fold_results])
    ibss = np.array([r["ibs"] for r in fold_results])
    auc_ts = np.array([r["auc_t"] for r in fold_results])

    return {
        "k": cfg.kfold,
        "n_folds": n_folds,
        "times": times.tolist(),
        "mean_auc": {
            "mean": float(mean_aucs.mean()),
            "std": float(mean_aucs.std()),
            "values": mean_aucs.tolist(),
        },
        "uno_c": {
            "mean": float(uno_cs.mean()),
            "std": float(uno_cs.std()),
            "values": uno_cs.tolist(),
        },
        "ibs": {
            "mean": float(ibss.mean()),
            "std": float(ibss.std()),
            "values": ibss.tolist(),
        },
        "auc_t": {
            "mean": auc_ts.mean(axis=0).tolist(),
            "std": auc_ts.std(axis=0).tolist(),
        },
        "folds": fold_results,
    }


def plot_exam_time_histogram(meta_exam: pd.DataFrame, out_path: Path) -> None:
    """plot histogram of exam times (study_date) for cases vs controls"""
    if "study_date" not in meta_exam.columns:
        return

    ytc = pd.to_numeric(meta_exam["years_to_cancer"], errors="coerce")
    is_case = (ytc <= 5) & ytc.notna()
    is_control = (ytc > 5) | ytc.isna()

    study_dates = pd.to_datetime(meta_exam["study_date"], errors="coerce")
    case_dates = study_dates[is_case].dropna()
    control_dates = study_dates[is_control].dropna()

    if len(case_dates) == 0 and len(control_dates) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    date_list = []
    if len(case_dates) > 0:
        date_list.append(case_dates)
    if len(control_dates) > 0:
        date_list.append(control_dates)
    all_dates = pd.concat(date_list)
    bins = pd.date_range(
        start=all_dates.min(),
        end=all_dates.max(),
        freq="YS",
    )

    if len(case_dates) > 0:
        ax.hist(
            case_dates,
            bins=bins,
            histtype="step",
            linewidth=2,
            label=f"cases (n={len(case_dates):,})",
            color="#E64B35",
        )
    if len(control_dates) > 0:
        ax.hist(
            control_dates,
            bins=bins,
            histtype="step",
            linewidth=2,
            label=f"controls (n={len(control_dates):,})",
            color="#4DBBD5",
        )

    ax.set_xlabel("exam date")
    ax.set_ylabel("count")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _compute_binary_metrics_at_threshold(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float
) -> dict[str, float]:
    """compute sensitivity, specificity, PPV, NPV at a given threshold"""
    y_pred_binary = (y_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
    }


def _find_optimal_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """find optimal threshold using Youden's index"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    youden = tpr - fpr
    optimal_idx = np.argmax(youden)
    return thresholds[optimal_idx]


def _delong_auc_ci(
    y_true: np.ndarray, y_pred: np.ndarray, confidence: float = 0.95
) -> tuple[float, float, float]:
    """compute AUC with 95% CI using DeLong's non-parametric method

    returns (auc, lower_bound, upper_bound)
    """
    from scipy import stats

    if len(y_true) == 0 or not (y_true.any() and (~y_true.astype(bool)).any()):
        return float("nan"), float("nan"), float("nan")

    auc = roc_auc_score(y_true, y_pred)

    # DeLong variance estimation
    # separate cases and controls
    case_idx = np.where(y_true == 1)[0]
    ctrl_idx = np.where(y_true == 0)[0]

    if len(case_idx) == 0 or len(ctrl_idx) == 0:
        return auc, float("nan"), float("nan")

    n_case = len(case_idx)
    n_ctrl = len(ctrl_idx)

    # compute V10 and V01 (DeLong et al. 1988) - vectorized version
    case_scores = y_pred[case_idx]
    ctrl_scores = y_pred[ctrl_idx]

    # Vectorized computation: compare all case scores with all control scores at once
    # Shape: (n_case, n_ctrl)
    comparison_matrix = case_scores[:, np.newaxis] - ctrl_scores[np.newaxis, :]

    # V10: for each case, compute mean indicator (ctrl < case) + 0.5 * (ctrl == case)
    less_than = (comparison_matrix > 0).astype(float)
    equal_to = (comparison_matrix == 0).astype(float) * 0.5
    v10 = np.mean(less_than + equal_to, axis=1)

    # V01: for each control, compute mean indicator (case > ctrl) + 0.5 * (case == ctrl)
    # Note: (case > ctrl) is same as (ctrl < case), so we can reuse
    v01 = np.mean(less_than.T + equal_to.T, axis=1)

    # variance of AUC
    s10 = np.var(v10, ddof=1) if len(v10) > 1 else 0.0
    s01 = np.var(v01, ddof=1) if len(v01) > 1 else 0.0
    var_auc = s10 / n_case + s01 / n_ctrl

    if var_auc <= 0:
        return auc, float("nan"), float("nan")

    # standard error
    se_auc = np.sqrt(var_auc)

    # 95% CI using normal approximation
    z = stats.norm.ppf((1 + confidence) / 2)
    lower = auc - z * se_auc
    upper = auc + z * se_auc

    # clip to [0, 1]
    lower = max(0.0, lower)
    upper = min(1.0, upper)

    return auc, lower, upper


def _bootstrap_auc_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
    group_ids: np.ndarray | None = None,
    n_bootstrap: int = 300,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """compute AUC with 95% CI using bootstrap resampling

    used for shared-control designs (receptor subtype, tumor grade)
    returns (auc, lower_bound, upper_bound)
    """
    if len(y_true) == 0 or not (y_true.any() and (~y_true.astype(bool)).any()):
        return float("nan"), float("nan"), float("nan")

    bootstrap_start = time.time()
    auc_observed = roc_auc_score(y_true, y_pred, sample_weight=sample_weight)
    n = len(y_true)
    rng = np.random.RandomState(random_state)

    if group_ids is not None:
        group_ids = np.asarray(group_ids).astype(str)
        unique_groups = np.unique(group_ids)
        group_to_indices = {g: np.where(group_ids == g)[0] for g in unique_groups}

        def _single_bootstrap(_i):
            boot_groups = rng.choice(
                unique_groups, size=len(unique_groups), replace=True
            )
            idx = np.concatenate([group_to_indices[g] for g in boot_groups])
            y_boot = y_true[idx]
            s_boot = y_pred[idx]
            w_boot = sample_weight[idx] if sample_weight is not None else None
            if y_boot.any() and (~y_boot.astype(bool)).any():
                try:
                    return roc_auc_score(y_boot, s_boot, sample_weight=w_boot)
                except Exception:
                    return None
            return None

    else:
        bootstrap_indices = rng.choice(n, size=(n_bootstrap, n), replace=True)

        def _single_bootstrap(i):
            idx = bootstrap_indices[i]
            y_boot = y_true[idx]
            s_boot = y_pred[idx]
            w_boot = sample_weight[idx] if sample_weight is not None else None
            if y_boot.any() and (~y_boot.astype(bool)).any():
                try:
                    return roc_auc_score(y_boot, s_boot, sample_weight=w_boot)
                except Exception:
                    return None
            return None

    # try parallelization with joblib using multiprocessing (not threading - GIL prevents speedup)
    try:
        from joblib import Parallel, delayed

        if group_ids is not None:
            # avoid parallel RNG coupling for grouped bootstrap
            bootstrap_aucs = [_single_bootstrap(i) for i in range(n_bootstrap)]
        else:
            # use multiprocessing backend for CPU-bound work
            bootstrap_aucs = Parallel(n_jobs=-1, backend="loky", prefer="processes")(
                delayed(_single_bootstrap)(i) for i in range(n_bootstrap)
            )
        bootstrap_aucs = [x for x in bootstrap_aucs if x is not None]
    except ImportError:
        # fallback to sequential if joblib not available
        bootstrap_aucs = []
        for i in range(n_bootstrap):
            result = _single_bootstrap(i)
            if result is not None:
                bootstrap_aucs.append(result)

    if len(bootstrap_aucs) == 0:
        return auc_observed, float("nan"), float("nan")

    # compute percentiles
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_aucs, 100 * alpha / 2)
    upper = np.percentile(bootstrap_aucs, 100 * (1 - alpha / 2))

    bootstrap_time = time.time() - bootstrap_start
    if bootstrap_time > 5.0:  # log if bootstrap takes more than 5 seconds
        print(
            f"    Bootstrap AUC CI (n={n}, iterations={n_bootstrap}): {_format_elapsed(bootstrap_time)}"
        )

    return auc_observed, lower, upper


def _harrell_cindex_ci(
    subset: pd.DataFrame,
    pred_cols: dict[int, str],
    target_horizons: list[int],
    cfg: Config,
    confidence: float = 0.95,
    compute_ci: bool = True,
) -> tuple[float, float, float]:
    """compute Harrell C-index with 95% CI using robust variance estimation

    uses bootstrap for variance estimation
    returns (c_index, lower_bound, upper_bound)
    if compute_ci=False, only returns point estimate (much faster)
    """
    if subset is None or len(subset) < 2:
        return float("nan"), float("nan"), float("nan")

    try:
        y_surv = _build_surv_arrays(subset)
        max_h = max(target_horizons)
        risk_col = cfg.cindex_col or pred_cols[max_h]
        risk = subset[risk_col].astype(float).to_numpy()

        # compute observed C-index
        c_index, *_ = concordance_index_ipcw(y_surv, y_surv, risk, tau=float(max_h))
        c_index = float(c_index)

        if not compute_ci:
            return c_index, float("nan"), float("nan")

        # bootstrap for CI (reduced iterations and optimized)
        bootstrap_start = time.time()
        rng = np.random.RandomState(42)
        n = len(subset)
        n_bootstrap = 300  # reduced from 500 for speed

        # pre-generate all bootstrap indices
        bootstrap_indices = rng.choice(n, size=(n_bootstrap, n), replace=True)

        # pre-extract risk values as numpy array for faster access
        risk_array = subset[risk_col].astype(float).to_numpy()

        # pre-extract survival data columns for faster access
        ytc_array = pd.to_numeric(subset["years_to_cancer"], errors="coerce").to_numpy()
        ylf_array = pd.to_numeric(
            subset["years_to_last_followup"], errors="coerce"
        ).to_numpy()

        # helper function for single bootstrap iteration
        def _single_bootstrap_c(i):
            indices = bootstrap_indices[i]
            if len(indices) < 2:
                return None
            try:
                # build survival arrays directly from arrays (faster than dataframe)
                ytc_boot = ytc_array[indices]
                ylf_boot = ylf_array[indices]
                event_boot = np.isfinite(ytc_boot) & (ytc_boot <= ylf_boot)
                time_boot = np.where(event_boot, ytc_boot, ylf_boot)
                y_surv_boot = Surv.from_arrays(
                    event=event_boot.astype(bool), time=time_boot.astype(float)
                )
                risk_boot = risk_array[indices]
                c_boot, *_ = concordance_index_ipcw(
                    y_surv_boot, y_surv_boot, risk_boot, tau=float(max_h)
                )
                return float(c_boot)
            except Exception:
                return None

        # try parallelization with joblib using multiprocessing (not threading - GIL prevents speedup)
        try:
            from joblib import Parallel, delayed

            # use multiprocessing backend for CPU-bound work
            bootstrap_cs = Parallel(n_jobs=-1, backend="loky", prefer="processes")(
                delayed(_single_bootstrap_c)(i) for i in range(n_bootstrap)
            )
            bootstrap_cs = [x for x in bootstrap_cs if x is not None]
        except ImportError:
            # fallback to sequential if joblib not available
            bootstrap_cs = []
            for i in range(n_bootstrap):
                result = _single_bootstrap_c(i)
                if result is not None:
                    bootstrap_cs.append(result)

        if len(bootstrap_cs) == 0:
            return c_index, float("nan"), float("nan")

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_cs, 100 * alpha / 2)
        upper = np.percentile(bootstrap_cs, 100 * (1 - alpha / 2))

        bootstrap_time = time.time() - bootstrap_start
        if bootstrap_time > 5.0:  # log if bootstrap takes more than 5 seconds
            print(
                f"    Bootstrap Harrell C-index CI (n={n}, iterations={n_bootstrap}): {_format_elapsed(bootstrap_time)}"
            )

        return c_index, lower, upper
    except Exception:
        return float("nan"), float("nan"), float("nan")


def compute_model_performance_metrics(
    cfg: Config,
    per_horizon_df: pd.DataFrame,
    surv_metrics: dict | None,
    exam_data: ExamData,
    meta_exam: pd.DataFrame | None,
) -> pd.DataFrame:
    """construct Omoleye-style model performance table"""

    if meta_exam is None:
        raise RuntimeError(
            "Omoleye-style performance table requires phenotype-enriched metadata"
        )

    func_start = time.time()

    pred_cols = exam_data.pred_cols
    available_horizons = sorted(pred_cols.keys())
    target_horizons = [
        h for h in available_horizons if h in OMOLEYE_TARGET_HORIZONS
    ] or available_horizons

    horizon_labels = [f"year {int(h)}" for h in target_horizons]
    harrell_label = "harrell c-index"

    # enrich exam-level predictions with phenotype metadata
    meta_cols = [
        "patient_id",
        "exam_id",
        "race_category",
        "age_at_exam",
        "receptor_subtype",
        "hr_status",
        "tumor_grade_group",
    ]
    merge_cols = [col for col in meta_cols if col in meta_exam.columns]
    df_exam = exam_data.exam_level.merge(
        meta_exam[merge_cols].drop_duplicates(),
        on=["patient_id", "exam_id"],
        how="left",
    )
    df_all = df_exam.copy()

    ytc_all = pd.to_numeric(df_all["years_to_cancer"], errors="coerce")
    df_filtered = df_all[ytc_all.isna() | (ytc_all >= 0.5)].copy()
    df_filtered["_case_5y"] = pd.to_numeric(
        df_filtered["years_to_cancer"], errors="coerce"
    ).notna() & (pd.to_numeric(df_filtered["years_to_cancer"], errors="coerce") <= 5)

    def _compute_auc_dict(
        subset: pd.DataFrame, use_bootstrap: bool = False
    ) -> dict[int, tuple[float, float, float]]:
        """compute AUC with CIs for each horizon

        returns dict mapping horizon -> (auc, lower_ci, upper_ci)
        uses DeLong by default, bootstrap if use_bootstrap=True
        """
        if subset.empty:
            return {
                h: (float("nan"), float("nan"), float("nan")) for h in target_horizons
            }
        result: dict[int, tuple[float, float, float]] = {}
        for h in target_horizons:
            y, s, w, pids, _excluded = _horizon_auc_arrays(subset, h, pred_cols[h], cfg)
            if len(y) == 0:
                result[h] = (float("nan"), float("nan"), float("nan"))
                continue
            if y.any() and (~y.astype(bool)).any():
                if use_bootstrap or cfg.auc_mode == "patient_weighted":
                    auc, lower, upper = _bootstrap_auc_ci(
                        y,
                        s,
                        sample_weight=w,
                        group_ids=pids if cfg.auc_mode == "patient_weighted" else None,
                    )
                else:
                    auc, lower, upper = _delong_auc_ci(y, s)
                result[h] = (auc, lower, upper)
            else:
                result[h] = (float("nan"), float("nan"), float("nan"))
        return result

    def _compute_harrell(
        subset: pd.DataFrame, compute_ci: bool = False
    ) -> tuple[float, float, float]:
        """compute Harrell C-index with CI

        returns (c_index, lower_ci, upper_ci)
        if compute_ci=False, only returns point estimate (much faster)
        """
        if subset is None or len(subset) < 2:
            return (float("nan"), float("nan"), float("nan"))
        try:
            return _harrell_cindex_ci(
                subset, pred_cols, target_horizons, cfg, compute_ci=compute_ci
            )
        except Exception:
            return (float("nan"), float("nan"), float("nan"))

    def _extract_overall_harrell() -> tuple[float, float, float] | None:
        """extract overall Harrell C-index with CI from survival metrics"""
        if surv_metrics is None:
            return None
        uno = surv_metrics.get("uno_c")
        if isinstance(uno, dict):
            mean_val = uno.get("mean")
            std_val = uno.get("std", 0.0)
            if mean_val is None:
                return None
            # approximate CI using std (for k-fold) or bootstrap if available
            from scipy import stats

            z = stats.norm.ppf(0.975)
            lower = mean_val - z * std_val
            upper = mean_val + z * std_val
            return (mean_val, lower, upper)
        if uno is None:
            return None
        # single value - would need bootstrap, but return without CI for now
        return (uno, float("nan"), float("nan"))

    def _add_row(
        rows: list[dict],
        category: str,
        label: str,
        subset: pd.DataFrame,
        precomputed_aucs: dict[int, tuple[float, float, float]] | None = None,
        harrell_override: tuple[float, float, float] | None = None,
        use_bootstrap: bool = False,
    ) -> None:
        """add a row to the performance metrics table with CIs

        precomputed_aucs: dict mapping horizon -> (auc, lower_ci, upper_ci)
        harrell_override: (c_index, lower_ci, upper_ci) or None
        use_bootstrap: if True, use bootstrap for AUC CIs (for shared-control designs)
        """
        if subset is None or subset.empty:
            return
        row_start = time.time()
        row_label = f"{category}: {label}"

        aucs = precomputed_aucs or _compute_auc_dict(
            subset, use_bootstrap=use_bootstrap
        )
        auc_time = time.time() - row_start
        if auc_time > 1.0:  # only log if it takes more than 1 second
            print(
                f"  [{_format_elapsed(time.time() - func_start)}] Computed AUCs for {row_label} ({_format_elapsed(auc_time)})"
            )

        row = {
            "category": category.lower(),
            "group": label.lower(),
            "n": int(len(subset)),
        }
        for h, col_label in zip(target_horizons, horizon_labels):
            auc_val, lower_ci, upper_ci = aucs.get(
                h, (float("nan"), float("nan"), float("nan"))
            )
            row[col_label] = auc_val
            row[f"{col_label}_lower"] = lower_ci
            row[f"{col_label}_upper"] = upper_ci

        harrell_start = time.time()
        # if harrell_override provided but doesn't have CI, compute CI anyway
        if harrell_override is not None:
            harrell_val, harrell_lower, harrell_upper = harrell_override
            # check if CI is missing (NaN)
            if (
                harrell_lower is None
                or pd.isna(harrell_lower)
                or harrell_upper is None
                or pd.isna(harrell_upper)
            ):
                # compute CI for the override value
                computed_harrell = _compute_harrell(subset, compute_ci=True)
                harrell_val, harrell_lower, harrell_upper = (
                    harrell_val,  # use override value
                    computed_harrell[1],  # use computed CI
                    computed_harrell[2],
                )
        else:
            harrell_val, harrell_lower, harrell_upper = _compute_harrell(
                subset, compute_ci=True
            )
        harrell_time = time.time() - harrell_start
        if harrell_time > 1.0:  # only log if it takes more than 1 second
            print(
                f"  [{_format_elapsed(time.time() - func_start)}] Computed Harrell C-index for {row_label} ({_format_elapsed(harrell_time)})"
            )

        row[harrell_label] = harrell_val
        row[f"{harrell_label}_lower"] = harrell_lower
        row[f"{harrell_label}_upper"] = harrell_upper
        rows.append(row)

    rows: list[dict] = []

    # All examinations (unfiltered) uses per-horizon summary to keep numbers aligned
    # need to compute CIs for precomputed AUCs
    summary_lookup: dict[int, tuple[float, float, float]] | None = None
    if not per_horizon_df.empty:
        summary_start = time.time()
        print(
            f"  [{_format_elapsed(time.time() - func_start)}] Computing summary lookup CIs..."
        )
        summary_lookup = {}
        for _, r in per_horizon_df.iterrows():
            h = int(r["horizon_years"])
            if h in target_horizons:
                y_all, s_all, w_all, pids_all, _excluded = _horizon_auc_arrays(
                    df_all, h, pred_cols[h], cfg
                )
                if len(y_all) > 0:
                    if y_all.any() and (~y_all.astype(bool)).any():
                        if cfg.auc_mode == "patient_weighted":
                            auc, lower, upper = _bootstrap_auc_ci(
                                y_all,
                                s_all,
                                sample_weight=w_all,
                                group_ids=pids_all,
                            )
                        else:
                            auc, lower, upper = _delong_auc_ci(y_all, s_all)
                        summary_lookup[h] = (auc, lower, upper)
                    else:
                        summary_lookup[h] = (float("nan"), float("nan"), float("nan"))
                else:
                    summary_lookup[h] = (float("nan"), float("nan"), float("nan"))
        summary_time = time.time() - summary_start
        print(
            f"  [{_format_elapsed(time.time() - func_start)}] Completed summary lookup ({_format_elapsed(summary_time)})"
        )

    overall_harrell = _extract_overall_harrell()
    print(
        f"  [{_format_elapsed(time.time() - func_start)}] Adding row: all examinations"
    )
    _add_row(
        rows,
        "all examinations",
        "all examinations",
        df_all,
        precomputed_aucs=summary_lookup if summary_lookup else None,
        harrell_override=overall_harrell,
        use_bootstrap=False,
    )

    print(
        f"  [{_format_elapsed(time.time() - func_start)}] Adding row: all examinations (ttc ≥ 6 mo)"
    )
    _add_row(
        rows,
        "all examinations",
        "all examinations (ttc ≥ 6 mo)",
        df_filtered,
        use_bootstrap=False,
    )

    # Self-reported race (filtered)
    for race_label in ("African American", "White"):
        print(
            f"  [{_format_elapsed(time.time() - func_start)}] Adding row: self-reported race - {race_label.lower()}"
        )
        subset = df_filtered[df_filtered["race_category"].astype(str) == race_label]
        _add_row(
            rows, "self-reported race", race_label.lower(), subset, use_bootstrap=False
        )

    # Age groups (filtered)
    age_bins = [
        ("<50", None, 50),
        ("50-60", 50, 60),
        ("60-70", 60, 70),
        ("70-90", 70, 90),
    ]
    age_values = pd.to_numeric(df_filtered["age_at_exam"], errors="coerce")
    df_filtered = df_filtered.assign(_age_at_exam=age_values)
    for label, low, high in age_bins:
        print(
            f"  [{_format_elapsed(time.time() - func_start)}] Adding row: age group - {label.lower()}"
        )
        mask = pd.Series(True, index=df_filtered.index)
        if low is not None:
            mask &= df_filtered["_age_at_exam"] >= low
        if high is not None:
            mask &= df_filtered["_age_at_exam"] < high
        subset = df_filtered[mask & df_filtered["_age_at_exam"].notna()]
        _add_row(rows, "age group", label.lower(), subset, use_bootstrap=False)

    # Helper for case-restricted subsets with common controls
    def _subset_cases_with_controls(
        df: pd.DataFrame, case_condition: pd.Series
    ) -> pd.DataFrame | None:
        if "_case_5y" not in df.columns:
            return None
        case_mask = df["_case_5y"] & case_condition
        if not case_mask.any():
            return None
        keep_mask = (~df["_case_5y"]) | case_mask
        return df[keep_mask]

    # Receptor subtype (shared controls - use bootstrap)
    for subtype in ("HR+/HER2-", "HR+/HER2+", "HR-/HER2+", "HR-/HER2-"):
        print(
            f"  [{_format_elapsed(time.time() - func_start)}] Adding row: receptor subtype - {subtype.lower()}"
        )
        subset = _subset_cases_with_controls(
            df_filtered,
            df_filtered["receptor_subtype"].astype(str) == subtype,
        )
        if subset is not None:
            _add_row(
                rows,
                "receptor subtype (cases ≤5y; common controls)",
                subtype.lower(),
                subset,
                use_bootstrap=True,
            )

    for hr_group in ("HR+", "HR-"):
        print(
            f"  [{_format_elapsed(time.time() - func_start)}] Adding row: receptor subtype - {hr_group.lower()}"
        )
        subset = _subset_cases_with_controls(
            df_filtered, df_filtered["hr_status"].astype(str) == hr_group
        )
        if subset is not None:
            _add_row(
                rows,
                "receptor subtype (cases ≤5y; common controls)",
                hr_group.lower(),
                subset,
                use_bootstrap=True,
            )

    # Tumor grade (shared controls - use bootstrap)
    for grade_group in ("Low grade", "Intermediate grade", "High grade"):
        print(
            f"  [{_format_elapsed(time.time() - func_start)}] Adding row: tumor grade - {grade_group.lower()}"
        )
        subset = _subset_cases_with_controls(
            df_filtered,
            df_filtered["tumor_grade_group"].astype(str) == grade_group,
        )
        if subset is not None:
            _add_row(
                rows,
                "tumor grade (cases ≤5y; common controls)",
                grade_group.lower(),
                subset,
                use_bootstrap=True,
            )

    if not rows:
        raise RuntimeError("no rows generated for performance metrics table")

    df_result = pd.DataFrame(rows)
    total_time = time.time() - func_start
    print(
        f"  [{_format_elapsed(time.time() - func_start)}] Completed model performance metrics computation ({_format_elapsed(total_time)})"
    )
    return df_result


def compute_patient_examination_characteristics(
    cfg: Config,
    allowed_exam_pairs: frozenset[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """compute patient- and examination-level characteristics table

    computes statistics from mirai_manifest.csv merged with phenotype data
    """
    meta = _load_enriched_manifest(cfg)
    if allowed_exam_pairs is not None:
        allowed_df = pd.DataFrame(
            list(allowed_exam_pairs), columns=["patient_id", "exam_id"]
        )
        before = len(meta)
        meta = meta.merge(allowed_df, on=["patient_id", "exam_id"], how="inner")
        print(
            f"Applied analysis exam-pair filter to characteristics: {before - len(meta)} rows excluded"
        )

    print("\n=== Date Source Debugging ===")
    print(f"Exams in manifest (enriched): {len(meta):,}")
    if "study_date" in meta.columns:
        print(
            f"Exams with study_date after enrichment: {meta['study_date'].notna().sum():,}"
        )
        print(f"Exams missing study_date: {meta['study_date'].isna().sum():,}")
        if "age_at_exam" in meta.columns:
            print(
                f"Exams with age_at_exam calculated: {meta['age_at_exam'].notna().sum():,}"
            )
    else:
        print("study_date not available after enrichment")

    # aggregate per exam BEFORE filtering to check date coverage
    meta_exam_before_filter = meta.groupby(
        ["patient_id", "exam_id"], as_index=False
    ).first()
    print("\nBefore 6-month filter (aggregated per exam):")
    print(f"  Unique exams: {len(meta_exam_before_filter):,}")
    if "study_date" in meta_exam_before_filter.columns:
        print(
            f"  Exams with study_date: {meta_exam_before_filter['study_date'].notna().sum():,}"
        )
        print(
            f"  Exams missing study_date: {meta_exam_before_filter['study_date'].isna().sum():,}"
        )
        # check which exams are missing dates
        missing_dates = meta_exam_before_filter[
            meta_exam_before_filter["study_date"].isna()
        ]
        if len(missing_dates) > 0:
            print("  Sample of exams missing dates (first 5):")
            print(missing_dates[["patient_id", "exam_id", "years_to_cancer"]].head())

    # filter to 6-month time-to-cancer filter (exclude cases with TTC < 6 months)
    ytc = pd.to_numeric(meta["years_to_cancer"], errors="coerce")
    meta_filtered = meta[ytc.isna() | (ytc >= 0.5)].copy()

    print("\nAfter 6-month filter:")
    print(f"  Exams in meta_filtered: {len(meta_filtered):,}")
    if "study_date" in meta_filtered.columns:
        print(f"  Exams with study_date: {meta_filtered['study_date'].notna().sum():,}")
        print(
            f"  Exams missing study_date: {meta_filtered['study_date'].isna().sum():,}"
        )

    # aggregate per exam
    # use 'first' for study_date to preserve dates (assuming all views for same exam have same date)
    meta_exam = meta.groupby(["patient_id", "exam_id"], as_index=False).first()
    meta_filtered_exam = meta_filtered.groupby(
        ["patient_id", "exam_id"], as_index=False
    ).first()

    print("\nAfter aggregation per exam:")
    print(f"  Unique exams in meta_exam: {len(meta_exam):,}")
    print(f"  Unique exams in meta_filtered_exam: {len(meta_filtered_exam):,}")
    if "study_date" in meta_filtered_exam.columns:
        print(
            f"  Exams with study_date in meta_filtered_exam: {meta_filtered_exam['study_date'].notna().sum():,}"
        )
        print(
            f"  Exams missing study_date in meta_filtered_exam: {meta_filtered_exam['study_date'].isna().sum():,}"
        )

        # check if dates are being lost during aggregation
        if "study_date" in meta_filtered.columns:
            # count exams that have at least one row with study_date before aggregation
            exams_with_date_before = (
                meta_filtered.groupby(["patient_id", "exam_id"])["study_date"]
                .apply(lambda x: x.notna().any())
                .sum()
            )
            print(
                f"  Exams with at least one row having study_date before aggregation: {exams_with_date_before:,}"
            )
            print(
                f"  Exams losing study_date during aggregation: {exams_with_date_before - meta_filtered_exam['study_date'].notna().sum():,}"
            )

    # identify case examinations (cancer within 5 years)
    ytc_exam = pd.to_numeric(meta_exam["years_to_cancer"], errors="coerce")
    ytc_filtered = pd.to_numeric(meta_filtered_exam["years_to_cancer"], errors="coerce")
    case_exam = meta_exam[ytc_exam <= 5].copy()
    case_filtered = meta_filtered_exam[ytc_filtered <= 5].copy()

    rows = []

    # Total individuals and examinations
    rows.append(
        (
            "Total",
            f"{meta_exam['patient_id'].nunique()}",
            f"{len(meta_exam)}",
            f"{len(case_exam)}",
        )
    )
    rows.append(
        (
            "6-month time-to-cancer filter",
            f"{meta_filtered_exam['patient_id'].nunique()}",
            f"{len(meta_filtered_exam)}",
            f"{len(case_filtered)}",
        )
    )

    # Age at examination
    if "age_at_exam" in meta_filtered_exam.columns:
        age_filtered = pd.to_numeric(meta_filtered_exam["age_at_exam"], errors="coerce")
        age_case_filtered = pd.to_numeric(case_filtered["age_at_exam"], errors="coerce")

        rows.append(("Age at examination (y)", "", "", ""))
        for label, low, high in [
            ("<50", 0, 50),
            ("50–60", 50, 60),
            ("60–70", 60, 70),
            ("70–90", 70, 90),
        ]:
            mask = (age_filtered >= low) & (age_filtered < high)
            n_total = mask.sum()
            pct_total = (
                f"{n_total}/{len(meta_filtered_exam)} ({100 * n_total / len(meta_filtered_exam):.1f})"
                if len(meta_filtered_exam) > 0
                else "0/0 (0.0)"
            )
            if len(age_case_filtered) > 0:
                mask_case = (age_case_filtered >= low) & (age_case_filtered < high)
                n_case = mask_case.sum()
                pct_case = f"{n_case}/{len(case_filtered)} ({100 * n_case / len(case_filtered):.1f})"
            else:
                pct_case = "0/0 (0.0)"
            rows.append((f"  {label}", "", pct_total, pct_case))

    # Race/ethnicity - categories already derived during enrichment
    if "race_category" in meta_filtered_exam.columns:
        rows.append(("Self-reported race and ethnicity", "", "", ""))
        race_category = meta_filtered_exam["race_category"]
        race_category_case = case_filtered["race_category"]

        # count by category
        for table_label in [
            "African American",
            "White",
            "Asian or Pacific Islander",
            "Hispanic",
            "Alaska Native",
        ]:
            mask = race_category == table_label
            n_pat = meta_filtered_exam[mask]["patient_id"].nunique()
            n_exam = mask.sum()
            pct_pat = (
                f"{n_pat}/{meta_filtered_exam['patient_id'].nunique()} ({100 * n_pat / meta_filtered_exam['patient_id'].nunique():.1f})"
                if meta_filtered_exam["patient_id"].nunique() > 0
                else "0/0 (0.0)"
            )
            pct_exam = (
                f"{n_exam}/{len(meta_filtered_exam)} ({100 * n_exam / len(meta_filtered_exam):.1f})"
                if len(meta_filtered_exam) > 0
                else "0/0 (0.0)"
            )
            if len(case_filtered) > 0:
                mask_case = race_category_case == table_label
                n_case = mask_case.sum()
                pct_case = f"{n_case}/{len(case_filtered)} ({100 * n_case / len(case_filtered):.1f})"
            else:
                pct_case = "0/0 (0.0)"
            rows.append((f"  {table_label}", pct_pat, pct_exam, pct_case))

        # handle unknown/missing
        mask_unknown = race_category == "Unknown"
        if mask_unknown.any():
            n_pat = meta_filtered_exam[mask_unknown]["patient_id"].nunique()
            n_exam = mask_unknown.sum()
            pct_pat = (
                f"{n_pat}/{meta_filtered_exam['patient_id'].nunique()} ({100 * n_pat / meta_filtered_exam['patient_id'].nunique():.1f})"
                if meta_filtered_exam["patient_id"].nunique() > 0
                else "0/0 (0.0)"
            )
            pct_exam = (
                f"{n_exam}/{len(meta_filtered_exam)} ({100 * n_exam / len(meta_filtered_exam):.1f})"
                if len(meta_filtered_exam) > 0
                else "0/0 (0.0)"
            )
            if len(case_filtered) > 0:
                mask_case = race_category_case == "Unknown"
                n_case = mask_case.sum()
                pct_case = f"{n_case}/{len(case_filtered)} ({100 * n_case / len(case_filtered):.1f})"
            else:
                pct_case = "0/0 (0.0)"
            rows.append(("  Unknown or missing data", pct_pat, pct_exam, pct_case))

    # Breast density - skip for now (not in phenotype CSV)
    # BI-RADS assessment - skip for now (not in phenotype CSV)

    df = pd.DataFrame(
        rows,
        columns=["Characteristic", "Individuals", "Examinations", "Case Examinations*"],
    )
    return df, meta_exam, meta_filtered_exam


def main() -> None:
    """entrypoint"""
    start_time = time.time()
    cfg = parse_args()
    last_checkpoint = start_time

    def checkpoint(msg: str):
        nonlocal last_checkpoint
        now = time.time()
        elapsed = now - last_checkpoint
        total = now - start_time
        print(f"[{_format_elapsed(total)}] {msg} ({_format_elapsed(elapsed)})")
        last_checkpoint = now

    checkpoint("Starting analysis")
    cutflow_csv_path: Path | None = None
    cutflow_json_path: Path | None = None
    resolved_cfg_path: Path | None = None
    cutflow_df_for_html: pd.DataFrame | None = None

    if cfg.out_json is not None:
        cfg.out_json.parent.mkdir(parents=True, exist_ok=True)
        (cfg.out_json.parent / "analyze_mirai_config.source.yaml").write_text(
            cfg.config_path.read_text()
        )
        (cfg.out_json.parent / "analyze_mirai_config.input.yaml").write_text(
            OmegaConf.to_yaml(OmegaConf.create(cfg.raw_config), resolve=False)
        )
        resolved_cfg_path = cfg.out_json.parent / "analyze_mirai_config.resolved.yaml"
        resolved_cfg_path.write_text(
            OmegaConf.to_yaml(OmegaConf.create(cfg.resolved_config), resolve=True)
        )
        print(f"Saved analysis config snapshots to {cfg.out_json.parent}")

    checkpoint("Computing per-horizon summary...")
    per_horizon_df, exam_data = summarize(cfg)
    checkpoint("Computed per-horizon summary")
    if cfg.out_json is not None:
        try:
            views_path = cfg.pred_csv.parent.parent / "sot" / "views.parquet"
            tags_path = cfg.pred_csv.parent.parent / "sot" / "dicom_tags.parquet"
            cutflow_df = _build_filter_cutflow_df(
                exam_data.exam_level_unfiltered, cfg, views_path, tags_path
            )
            cutflow_df_for_html = cutflow_df.copy()
            cutflow_csv_path = cfg.out_json.parent / "filter_cutflow.csv"
            cutflow_json_path = cfg.out_json.parent / "filter_cutflow.json"
            cutflow_df.to_csv(cutflow_csv_path, index=False)
            cutflow_json_path.write_text(
                json.dumps(cutflow_df.to_dict(orient="records"), indent=2)
            )
            print(f"Filter cutflow saved to {cutflow_csv_path}")
        except Exception as e:
            print(f"[warn] Could not write filter cutflow: {e}")
    print("\n=== Per-Horizon AUC ===")
    cols = ["horizon_years", "n", "cases", "controls", "excluded", "prevalence", "auc"]
    print(
        per_horizon_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    # compute patient and examination characteristics (using the same filtered exam pairs)
    try:
        checkpoint("Loading patient/examination characteristics...")
        characteristics_df, meta_exam, meta_filtered_exam = (
            compute_patient_examination_characteristics(
                cfg, allowed_exam_pairs=exam_data.exam_pairs
            )
        )
        checkpoint("Computed patient/examination characteristics")
        print("\n=== Patient- and Examination-level Characteristics ===")
        print(characteristics_df.to_string(index=False))
        if cfg.out_json is not None:
            characteristics_path = (
                cfg.out_json.parent / "patient_examination_characteristics.csv"
            )
            characteristics_df.to_csv(characteristics_path, index=False)
            print(
                f"\nPatient examination characteristics saved to {characteristics_path}"
            )

            # plot exam time histogram (using all exams after QC-driven filtering)
            plot_path = cfg.out_json.parent / "exam_time_histogram.png"
            plot_exam_time_histogram(meta_exam, plot_path)
            print(f"Exam time histogram saved to {plot_path}")
    except Exception as e:
        print(f"\n[warn] Could not compute patient examination characteristics: {e}")
        meta_exam = None

    surv = None
    if cfg.kfold is not None:
        try:
            checkpoint(f"Computing {cfg.kfold}-fold CV survival metrics...")
            surv = kfold_survival_metrics(cfg, exam_data)
            checkpoint("Computed k-fold survival metrics")
            print("\n=== k-fold CV survival metrics (IPCW) ===")
            print(f"k={surv['k']}, n_folds={surv['n_folds']}")
            print("times:", ", ".join(str(int(t)) for t in surv["times"]))
            print(
                "auc(t) mean:",
                ", ".join(f"{a:.4f}" for a in surv["auc_t"]["mean"]),
            )
            print(
                "auc(t) std:",
                ", ".join(f"{a:.4f}" for a in surv["auc_t"]["std"]),
            )
            print(
                f"mean auc: {surv['mean_auc']['mean']:.4f} ± {surv['mean_auc']['std']:.4f}"
            )
            print(
                f"uno c (tau=max time): {surv['uno_c']['mean']:.4f} ± {surv['uno_c']['std']:.4f}"
            )
            print(f"ibs: {surv['ibs']['mean']:.4f} ± {surv['ibs']['std']:.4f}")
        except Exception as e:
            print("\n[warn] k-fold survival metrics unavailable:", str(e))
    else:
        try:
            checkpoint("Computing survival metrics (IPCW)...")
            surv = survival_metrics(cfg, exam_data)
            checkpoint("Computed survival metrics")
            print("\n=== scikit-survival (IPCW) ===")
            print("times:", ", ".join(str(int(t)) for t in surv["times"]))
            print("auc(t):", ", ".join(f"{a:.4f}" for a in surv["auc_t"]))
            print(f"mean auc: {surv['mean_auc']:.4f}")
            print(f"uno c (tau=max time): {surv['uno_c']:.4f}")
            print("ibs:", f"{surv['ibs']:.4f}")
        except Exception as e:
            # allow quick use even if scikit-survival isn't installed or inputs are incomplete
            print("\n[warn] survival metrics unavailable:", str(e))

    hiddens_path = cfg.pred_csv.parent / "hiddens.npz"
    if hiddens_path.exists():
        try:
            checkpoint("Plotting PCA of Mirai hiddens...")
            plot_hiddens_pca(cfg, exam_data, hiddens_path)
            checkpoint("Saved hiddens PCA plots")
        except Exception as e:
            print(f"\n[warn] hiddens PCA failed: {e}")

    # compute model performance metrics
    try:
        checkpoint("Computing model performance metrics...")
        performance_df = compute_model_performance_metrics(
            cfg, per_horizon_df, surv, exam_data, meta_exam
        )
        checkpoint("Computed model performance metrics")

        # extract year columns and sort by numeric value
        year_cols_list = [
            c
            for c in performance_df.columns
            if c.startswith("year")
            and not c.endswith("_lower")
            and not c.endswith("_upper")
        ]
        year_cols_list.sort(
            key=lambda x: int(x.split()[-1]) if x.split()[-1].isdigit() else 0
        )

        # calculate column widths based on content
        max_category_width = (
            max(len(str(cat)) for cat in performance_df["category"]) + 1
        )  # +1 for padding
        max_category_width = max(max_category_width, len("category"))
        max_group_width = (
            max(len(str(group)) for group in performance_df["group"]) + 1
        )  # +1 for padding
        max_group_width = max(max_group_width, len("group"))

        # calculate width needed for year columns (value + CI on same row)
        def get_year_col_width(col):
            max_width = len(col)
            for _, row in performance_df.iterrows():
                val = row.get(col, float("nan"))
                lower = row.get(f"{col}_lower", float("nan"))
                upper = row.get(f"{col}_upper", float("nan"))
                val_str = f"{val:.2f}" if val is not None and not pd.isna(val) else ""
                ci_str = (
                    f" ({lower:.2f}, {upper:.2f})"
                    if (
                        lower is not None
                        and not pd.isna(lower)
                        and upper is not None
                        and not pd.isna(upper)
                    )
                    else ""
                )
                combined = val_str + ci_str
                max_width = max(max_width, len(combined))
            return max_width + 1

        year_col_widths = {col: get_year_col_width(col) for col in year_cols_list}

        # calculate width for harrell c-index column
        harrell_col = "harrell c-index"
        harrell_width = len(harrell_col)
        for _, row in performance_df.iterrows():
            val = row.get(harrell_col, float("nan"))
            lower = row.get(f"{harrell_col}_lower", float("nan"))
            upper = row.get(f"{harrell_col}_upper", float("nan"))
            val_str = f"{val:.2f}" if val is not None and not pd.isna(val) else ""
            ci_str = (
                f" ({lower:.2f}, {upper:.2f})"
                if (
                    lower is not None
                    and not pd.isna(lower)
                    and upper is not None
                    and not pd.isna(upper)
                )
                else ""
            )
            combined = val_str + ci_str
            harrell_width = max(harrell_width, len(combined))
        harrell_width += 1

        def format_value_with_ci(val, lower, upper):
            """format value with CI on same row"""
            val_str = f"{val:.2f}" if val is not None and not pd.isna(val) else ""
            ci_str = (
                f" ({lower:.2f}, {upper:.2f})"
                if (
                    lower is not None
                    and not pd.isna(lower)
                    and upper is not None
                    and not pd.isna(upper)
                )
                else ""
            )
            return val_str + ci_str

        def format_row(row):
            """format row with values and CIs on same line"""
            category_display = str(row["category"]).ljust(max_category_width)
            group = str(row["group"]).ljust(max_group_width)
            n = str(int(row["n"])).rjust(6)
            year_vals = []
            for col in year_cols_list:
                val = row.get(col, float("nan"))
                lower = row.get(f"{col}_lower", float("nan"))
                upper = row.get(f"{col}_upper", float("nan"))
                formatted = format_value_with_ci(val, lower, upper)
                year_vals.append(formatted.ljust(year_col_widths[col]))
            harrell_val = row.get(harrell_col, float("nan"))
            harrell_lower = row.get(f"{harrell_col}_lower", float("nan"))
            harrell_upper = row.get(f"{harrell_col}_upper", float("nan"))
            harrell_formatted = format_value_with_ci(
                harrell_val, harrell_lower, harrell_upper
            )
            harrell = (
                harrell_formatted.ljust(harrell_width)
                if harrell_formatted
                else "".ljust(harrell_width)
            )
            return f"{category_display} {group} {n} {' '.join(year_vals)} {harrell}"

        # build formatted table
        header_parts = [
            "category".ljust(max_category_width),
            "group".ljust(max_group_width),
            "n".rjust(6),
        ]
        year_headers = [c.ljust(year_col_widths[c]) for c in year_cols_list]
        header_parts.extend(year_headers)
        header_parts.append(harrell_col.ljust(harrell_width))
        header = " ".join(header_parts)
        separator = "-" * len(header)

        lines = [header, separator]
        for _, row in performance_df.iterrows():
            lines.append(format_row(row))

        # add notes below table
        notes = []
        # explain count issue for receptor subtype and tumor grade
        receptor_rows = performance_df[
            performance_df["category"].str.contains("Receptor subtype", na=False)
        ]
        grade_rows = performance_df[
            performance_df["category"].str.contains("Tumor grade", na=False)
        ]
        if len(receptor_rows) > 0 or len(grade_rows) > 0:
            notes.append("")
            notes.append(
                "Note: Counts for receptor subtype and tumor grade categories sum to more than"
            )
            notes.append(
                "the total because each category includes all controls plus only the cases"
            )
            notes.append(
                "matching that specific subtype/grade. This allows fair comparison across"
            )
            notes.append("subtypes/grades using the same control set.")

        formatted_table = "\n".join(lines)
        if notes:
            formatted_table += "\n\n" + "\n".join(notes)

        print(formatted_table)
        if cfg.out_json is not None:
            performance_csv_path = cfg.out_json.parent / "model_performance_metrics.csv"
            performance_df.to_csv(performance_csv_path, index=False)
            print(f"\nModel performance metrics saved to {performance_csv_path}")

            performance_txt_path = cfg.out_json.parent / "model_performance_metrics.txt"
            performance_txt_path.write_text(formatted_table)
            print(f"Model performance metrics saved to {performance_txt_path}")

            # generate HTML version with alternating row colors
            html_lines = [
                "<!DOCTYPE html>",
                "<html>",
                "<head>",
                "<meta charset='utf-8'>",
                "<title>discriminatory performance of mirai in chimec with stratified analysis</title>",
                "<style>",
                "  body { font-family: monospace; margin: 20px; }",
                "  table { border-collapse: collapse; width: auto; min-width: 100%; }",
                "  th { background-color: #e0e0e0; padding: 8px; text-align: left; border-bottom: 2px solid #333; white-space: nowrap; }",
                "  td { padding: 6px 8px; white-space: nowrap; }",
                "  pre { white-space: pre-wrap; background-color: #f6f8fa; padding: 10px; border: 1px solid #ddd; border-radius: 6px; }",
                "  .row-group-0 { background-color: #ffffff; }",
                "  .row-group-1 { background-color: #d4e4f7; }",
                "  tr td { transition: background-color 120ms ease-in-out; }",
                "  tr.row-group-0:hover td, tr.row-group-1:hover td { background-color: #fff3bf; }",
                "</style>",
                "</head>",
                "<body>",
                "<div style='margin-bottom: 16px;'>",
                "<strong>report artifacts:</strong>",
            ]
            if cutflow_csv_path is not None:
                html_lines.append(
                    f"<div>filter cutflow csv: <code>{cutflow_csv_path}</code></div>"
                )
            if cutflow_json_path is not None:
                html_lines.append(
                    f"<div>filter cutflow json: <code>{cutflow_json_path}</code></div>"
                )
            if resolved_cfg_path is not None:
                html_lines.append(
                    f"<div>resolved config: <code>{resolved_cfg_path}</code></div>"
                )
            html_lines.extend(
                [
                    "</div>",
                    "<table>",
                ]
            )

            # header row
            html_lines.append("  <tr>")
            html_lines.append("    <th>category</th>")
            html_lines.append("    <th>group</th>")
            html_lines.append("    <th>n</th>")
            for col in year_cols_list:
                html_lines.append(f"    <th>{col}</th>")
            html_lines.append(f"    <th>{harrell_col}</th>")
            html_lines.append("  </tr>")

            # data rows - alternate colors by category group
            group_idx = 0
            for _, row in performance_df.iterrows():
                # row with values and CIs on same line
                group_class = f"row-group-{group_idx % 2}"
                html_lines.append(f"  <tr class='{group_class}'>")
                html_lines.append(f"    <td>{row['category']}</td>")
                html_lines.append(f"    <td>{row['group']}</td>")
                html_lines.append(f"    <td>{int(row['n']):,}</td>")
                for col in year_cols_list:
                    val = row.get(col, float("nan"))
                    lower = row.get(f"{col}_lower", float("nan"))
                    upper = row.get(f"{col}_upper", float("nan"))
                    val_ci_str = format_value_with_ci(val, lower, upper)
                    html_lines.append(f"    <td>{val_ci_str}</td>")
                harrell_val = row.get(harrell_col, float("nan"))
                harrell_lower = row.get(f"{harrell_col}_lower", float("nan"))
                harrell_upper = row.get(f"{harrell_col}_upper", float("nan"))
                harrell_str = format_value_with_ci(
                    harrell_val, harrell_lower, harrell_upper
                )
                html_lines.append(f"    <td>{harrell_str}</td>")
                html_lines.append("  </tr>")

                # increment group index for next category
                group_idx += 1

            html_lines.append("</table>")

            # add notes if any
            if notes:
                html_lines.append("<div style='margin-top: 20px;'>")
                for note in notes:
                    if note:
                        html_lines.append(f"<p>{note}</p>")
                html_lines.append("</div>")

            # embedded cutflow table for environments where local file links don't open
            if cutflow_df_for_html is not None and not cutflow_df_for_html.empty:
                html_lines.append("<details style='margin-top: 20px;'>")
                html_lines.append(
                    "<summary><strong>filter cutflow (embedded)</strong></summary>"
                )
                html_lines.append("<table style='margin-top: 10px;'>")
                html_lines.append("  <tr>")
                for col in cutflow_df_for_html.columns:
                    html_lines.append(f"    <th>{col}</th>")
                html_lines.append("  </tr>")
                for _, row in cutflow_df_for_html.iterrows():
                    html_lines.append("  <tr>")
                    for col in cutflow_df_for_html.columns:
                        html_lines.append(f"    <td>{row[col]}</td>")
                    html_lines.append("  </tr>")
                html_lines.append("</table>")
                html_lines.append("</details>")

            # embedded resolved config text
            if resolved_cfg_path is not None and resolved_cfg_path.exists():
                html_lines.append("<details style='margin-top: 20px;'>")
                html_lines.append(
                    "<summary><strong>resolved config (embedded)</strong></summary>"
                )
                html_lines.append("<pre>")
                html_lines.append(resolved_cfg_path.read_text())
                html_lines.append("</pre>")
                html_lines.append("</details>")

            html_lines.append("</body>")
            html_lines.append("</html>")

            html_content = "\n".join(html_lines)
            performance_html_path = (
                cfg.out_json.parent / "model_performance_metrics.html"
            )
            performance_html_path.write_text(html_content)
            print(f"Model performance metrics saved to {performance_html_path}")
    except Exception as e:
        print(f"\n[warn] Could not compute model performance metrics: {e}")

    if cfg.out_json is not None:
        payload = per_horizon_df.to_dict(orient="records")
        out = {
            "per_horizon": payload,
            "filter_summary": exam_data.filter_summary,
            "resolved_config": cfg.resolved_config,
        }
        if surv is not None:
            out["survival_metrics"] = surv
        cfg.out_json.parent.mkdir(parents=True, exist_ok=True)
        cfg.out_json.write_text(json.dumps(out, indent=2))

    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    if hours > 0:
        print(f"\n=== Total runtime: {hours}h {minutes}m {seconds:.2f}s ===")
    elif minutes > 0:
        print(f"\n=== Total runtime: {minutes}m {seconds:.2f}s ===")
    else:
        print(f"\n=== Total runtime: {seconds:.2f}s ===")


if __name__ == "__main__":
    main()
