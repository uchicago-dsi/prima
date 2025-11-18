#!/usr/bin/env python3
"""analyze_mirai_output.py

Quick analysis for Mirai's validation outputs.

Reads Mirai's prediction CSV (e.g., demo/validation_output.csv) and the
metadata CSV you used for validation (with years_to_cancer and years_to_last_followup),
and prints a simple summary by horizon: cases, controls, excluded (insufficient follow-up),
AUC (naive ROC restricted to exams with ≥h years of follow-up or an event ≤h).

Notes
-----
- this is *not* Uno's time-dependent C-index; it's a pragmatic ROC on a binary label per horizon
- labels are computed as:
    case_h  = years_to_cancer <= h
    ctrl_h  = (years_to_cancer  > h) & (years_to_last_followup >= h)
    include = case_h | ctrl_h
  rows with insufficient follow-up are excluded from AUC_h
- prediction columns are auto-detected via regex; override with --map if needed

Usage
-----
python analyze_mirai.py \
  --out-dir /prima/mirai-prep \
  --out /prima/mirai-prep/summary.json

Expects validation_output.csv and mirai_manifest.csv in --out-dir.

Requirements
------------
  pip install pandas numpy scikit-learn pyarrow

"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
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


@dataclass(frozen=True)
class ExamData:
    """container for merged prediction/metadata tables"""

    merged_views: pd.DataFrame
    exam_level: pd.DataFrame
    meta_all: pd.DataFrame
    meta_eval: pd.DataFrame
    pred_cols: dict[int, str]
    n_pred_rows: int


def _parse_map(map_kv: list[str]) -> dict[int, str]:
    """parse --map entries like 1:pred_1year,5:pred_5yr into {1: 'pred_1year', 5: 'pred_5yr'}"""
    m: dict[int, str] = {}
    for kv in map_kv:
        k, v = kv.split(":", 1)
        m[int(k)] = v
    return m


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

    return ExamData(
        merged_views=df,
        exam_level=df_exam,
        meta_all=meta_all,
        meta_eval=meta_eval,
        pred_cols=pred_cols,
        n_pred_rows=len(pred),
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
            f"could not auto-detect prediction columns; pass --map 1:col,...\n"
            f"Available numeric columns: {numeric_cols}"
        )
    return dict(sorted(cols.items()))


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
    print()

    # vectorized label tensors (now per exam)
    ytc = df_exam["years_to_cancer"].astype(float).to_numpy()
    ylf = df_exam["years_to_last_followup"].astype(float).to_numpy()

    rows = []
    for h, col in pred_cols.items():
        scores = df_exam[col].astype(float).to_numpy()
        case = ytc <= h
        ctrl = (ytc > h) & (ylf >= h)
        include = case | ctrl
        y = case[include].astype(np.int32)
        s = scores[include]
        auc = float("nan")
        if y.any() and (~y.astype(bool)).any():
            auc = float(roc_auc_score(y, s))
        rows.append(
            {
                "horizon_years": h,
                "n": int(include.sum()),
                "cases": int(case.sum()),
                "controls": int(ctrl.sum()),
                "excluded": int((~include).sum()),
                "prevalence": float(case.sum() / max(1, (case | ctrl).sum())),
                "auc": auc,
            }
        )

    out = pd.DataFrame(rows).sort_values("horizon_years").reset_index(drop=True)
    return out, exam_data


def parse_args() -> Config:
    """parse CLI args"""
    p = argparse.ArgumentParser(
        description="Summarize Mirai validation_output.csv with simple per-horizon AUC and survival metrics"
    )
    p.add_argument(
        "--out-dir",
        dest="out_dir",
        type=Path,
        required=True,
        help="directory containing validation_output.csv and mirai_manifest.csv",
    )
    p.add_argument("--out", dest="out_json", type=Path)
    p.add_argument("--split", dest="split", default="test", type=str)
    p.add_argument(
        "--map",
        dest="map_kv",
        nargs="*",
        help="override horizon→column mapping: e.g., 1:pred_1year 5:pred_5year",
    )
    p.add_argument(
        "--cindex-col",
        dest="cindex_col",
        type=str,
        help="column to use for Uno's C-index (default: max-horizon risk)",
    )
    p.add_argument(
        "--kfold",
        dest="kfold",
        type=int,
        help="k-fold CV for IPCW sensitivity analysis (e.g., 5); splits at patient level",
    )
    args = p.parse_args()
    colmap = _parse_map(args.map_kv) if args.map_kv else None

    # construct paths from output directory
    pred_csv = args.out_dir / "validation_output.csv"
    meta_csv = args.out_dir / "mirai_manifest.csv"
    out_json = args.out_json or (args.out_dir / "summary.json")

    return Config(
        pred_csv=pred_csv,
        meta_csv=meta_csv,
        out_json=out_json,
        split=args.split,
        colmap=colmap,
        cindex_col=args.cindex_col,
        kfold=args.kfold,
    )


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

    pred_cols = exam_data.pred_cols
    available_horizons = sorted(pred_cols.keys())
    target_horizons = [
        h for h in available_horizons if h in OMOLEYE_TARGET_HORIZONS
    ] or available_horizons

    horizon_labels = [f"Year {int(h)}" for h in target_horizons]
    harrell_label = "Harrell C-index"

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

    def _compute_auc_dict(subset: pd.DataFrame) -> dict[int, float]:
        if subset.empty:
            return {h: float("nan") for h in target_horizons}
        ytc = pd.to_numeric(subset["years_to_cancer"], errors="coerce").to_numpy()
        ylf = pd.to_numeric(
            subset["years_to_last_followup"], errors="coerce"
        ).to_numpy()
        result: dict[int, float] = {}
        for h in target_horizons:
            scores = subset[pred_cols[h]].astype(float).to_numpy()
            case = np.less_equal(ytc, h)
            ctrl = np.logical_and(np.greater(ytc, h), np.greater_equal(ylf, h))
            include = np.logical_or(case, ctrl)
            if not include.any():
                result[h] = float("nan")
                continue
            y = case[include].astype(np.int32)
            s = scores[include]
            if y.any() and (~y.astype(bool)).any():
                result[h] = float(roc_auc_score(y, s))
            else:
                result[h] = float("nan")
        return result

    def _compute_harrell(subset: pd.DataFrame) -> float:
        if subset is None or len(subset) < 2:
            return float("nan")
        try:
            y_surv = _build_surv_arrays(subset)
            max_h = max(target_horizons)
            risk_col = cfg.cindex_col or pred_cols[max_h]
            risk = subset[risk_col].astype(float).to_numpy()
            c_index, *_ = concordance_index_ipcw(y_surv, y_surv, risk, tau=float(max_h))
            return float(c_index)
        except Exception:
            return float("nan")

    def _extract_overall_harrell() -> float | None:
        if surv_metrics is None:
            return None
        uno = surv_metrics.get("uno_c")
        if isinstance(uno, dict):
            return uno.get("mean")
        return uno

    def _add_row(
        rows: list[dict],
        category: str,
        label: str,
        subset: pd.DataFrame,
        precomputed_aucs: dict[int, float] | None = None,
        harrell_override: float | None = None,
    ) -> None:
        if subset is None or subset.empty:
            return
        aucs = precomputed_aucs or _compute_auc_dict(subset)
        row = {
            "category": category,
            "group": f"{label} (n = {len(subset):,})",
            "n": int(len(subset)),
        }
        for h, col_label in zip(target_horizons, horizon_labels):
            row[col_label] = aucs.get(h, float("nan"))
        row[harrell_label] = (
            harrell_override
            if harrell_override is not None
            else _compute_harrell(subset)
        )
        rows.append(row)

    rows: list[dict] = []

    # All examinations (unfiltered) uses per-horizon summary to keep numbers aligned
    summary_lookup = {
        int(r["horizon_years"]): r["auc"]
        for _, r in per_horizon_df.iterrows()
        if int(r["horizon_years"]) in target_horizons
    }
    overall_harrell = _extract_overall_harrell()
    _add_row(
        rows,
        "All examinations",
        "All examinations",
        df_all,
        precomputed_aucs=summary_lookup if summary_lookup else None,
        harrell_override=overall_harrell,
    )

    _add_row(
        rows,
        "All examinations",
        "All examinations (TTC ≥ 6 mo)",
        df_filtered,
    )

    # Self-reported race (filtered)
    for race_label in ("African American", "White"):
        subset = df_filtered[df_filtered["race_category"].astype(str) == race_label]
        _add_row(rows, "Self-reported race", race_label, subset)

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
        mask = pd.Series(True, index=df_filtered.index)
        if low is not None:
            mask &= df_filtered["_age_at_exam"] >= low
        if high is not None:
            mask &= df_filtered["_age_at_exam"] < high
        subset = df_filtered[mask & df_filtered["_age_at_exam"].notna()]
        _add_row(rows, "Age group", label, subset)

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

    # Receptor subtype
    for subtype in ("HR+/HER2-", "HR+/HER2+", "HR-/HER2+", "HR-/HER2-"):
        subset = _subset_cases_with_controls(
            df_filtered,
            df_filtered["receptor_subtype"].astype(str) == subtype,
        )
        if subset is not None:
            _add_row(
                rows,
                "Receptor subtype (cases ≤5y; common controls)",
                subtype,
                subset,
            )

    for hr_group in ("HR+", "HR-"):
        subset = _subset_cases_with_controls(
            df_filtered, df_filtered["hr_status"].astype(str) == hr_group
        )
        if subset is not None:
            _add_row(
                rows,
                "Receptor subtype (cases ≤5y; common controls)",
                hr_group,
                subset,
            )

    # Tumor grade
    for grade_group in ("Low grade", "Intermediate grade", "High grade"):
        subset = _subset_cases_with_controls(
            df_filtered,
            df_filtered["tumor_grade_group"].astype(str) == grade_group,
        )
        if subset is not None:
            _add_row(
                rows,
                "Tumor grade (cases ≤5y; common controls)",
                grade_group,
                subset,
            )

    if not rows:
        raise RuntimeError("no rows generated for performance metrics table")

    df_result = pd.DataFrame(rows)
    return df_result


def compute_patient_examination_characteristics(
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """compute patient- and examination-level characteristics table

    computes statistics from mirai_manifest.csv merged with phenotype data
    """
    meta = _load_enriched_manifest(cfg)

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
    cfg = parse_args()

    # compute patient and examination characteristics
    try:
        characteristics_df, meta_exam, meta_filtered_exam = (
            compute_patient_examination_characteristics(cfg)
        )
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

            # plot exam time histogram (using all exams, not just filtered)
            plot_path = cfg.out_json.parent / "exam_time_histogram.png"
            plot_exam_time_histogram(meta_exam, plot_path)
            print(f"Exam time histogram saved to {plot_path}")
    except Exception as e:
        print(f"\n[warn] Could not compute patient examination characteristics: {e}")
        meta_exam = None

    per_horizon_df, exam_data = summarize(cfg)
    print("\n=== Per-Horizon AUC ===")
    cols = ["horizon_years", "n", "cases", "controls", "excluded", "prevalence", "auc"]
    print(
        per_horizon_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    surv = None
    if cfg.kfold is not None:
        try:
            surv = kfold_survival_metrics(cfg, exam_data)
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
            surv = survival_metrics(cfg, exam_data)
            print("\n=== scikit-survival (IPCW) ===")
            print("times:", ", ".join(str(int(t)) for t in surv["times"]))
            print("auc(t):", ", ".join(f"{a:.4f}" for a in surv["auc_t"]))
            print(f"mean auc: {surv['mean_auc']:.4f}")
            print(f"uno c (tau=max time): {surv['uno_c']:.4f}")
            print("ibs:", f"{surv['ibs']:.4f}")
        except Exception as e:
            # allow quick use even if scikit-survival isn't installed or inputs are incomplete
            print("\n[warn] survival metrics unavailable:", str(e))

    # compute model performance metrics
    try:
        performance_df = compute_model_performance_metrics(
            cfg, per_horizon_df, surv, exam_data, meta_exam
        )
        print("\n=== Model Performance Metrics ===")

        # format numeric columns
        def format_metric(x):
            return f"{x:.4f}" if x is not None and not pd.isna(x) else ""

        float_cols = performance_df.select_dtypes(include="float").columns
        format_dict = {col: format_metric for col in float_cols}
        print(performance_df.to_string(index=False, formatters=format_dict))
        if cfg.out_json is not None:
            performance_path = cfg.out_json.parent / "model_performance_metrics.csv"
            performance_df.to_csv(performance_path, index=False)
            print(f"\nModel performance metrics saved to {performance_path}")
    except Exception as e:
        print(f"\n[warn] Could not compute model performance metrics: {e}")

    if cfg.out_json is not None:
        payload = per_horizon_df.to_dict(orient="records")
        out = {"per_horizon": payload}
        if surv is not None:
            out["survival_metrics"] = surv
        cfg.out_json.parent.mkdir(parents=True, exist_ok=True)
        cfg.out_json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
