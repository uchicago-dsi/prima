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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sksurv.metrics import (
    brier_score,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv


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


def _parse_map(map_kv: list[str]) -> dict[int, str]:
    """parse --map entries like 1:pred_1year,5:pred_5yr into {1: 'pred_1year', 5: 'pred_5yr'}"""
    m: dict[int, str] = {}
    for kv in map_kv:
        k, v = kv.split(":", 1)
        m[int(k)] = v
    return m


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


def summarize(cfg: Config) -> pd.DataFrame:
    """compute per-horizon summary: cases, controls, excluded, auc, prevalence

    returns a dataframe indexed by horizon with columns [n, cases, controls, excluded, auc]
    """
    pred = pd.read_csv(cfg.pred_csv)

    # patient/exam ids from predictions
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

    # load metadata with labels
    meta = (
        pd.read_csv(cfg.meta_csv)
        if cfg.meta_csv.suffix.lower() == ".csv"
        else pd.read_parquet(cfg.meta_csv)
    )
    req = {"patient_id", "exam_id", "years_to_cancer", "years_to_last_followup"}
    if not req.issubset(set(meta.columns)):
        missing = sorted(req - set(meta.columns))
        raise KeyError(f"metadata missing columns: {missing}")
    if cfg.split is not None and "split_group" in meta.columns:
        meta_eval = meta[
            meta["split_group"].astype(str).str.lower() == cfg.split.lower()
        ].copy()
    else:
        meta_eval = meta.copy()

    # ensure merge keys are strings in both dataframes
    meta_eval = meta_eval.assign(
        patient_id=meta_eval["patient_id"].astype(str),
        exam_id=meta_eval["exam_id"].astype(str),
    )

    # join predictions to metadata (evaluation set)
    cols_to_merge = list(req)
    if "split_group" in meta_eval.columns:
        cols_to_merge.append("split_group")
    df = pred.merge(
        meta_eval[cols_to_merge],
        on=["patient_id", "exam_id"],
        how="inner",
    )

    # print summary statistics
    print("=== Data Summary ===")
    print(f"Predictions CSV rows: {len(pred):,}")
    print(f"Metadata CSV rows (total): {len(meta):,}")
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

    # decide prediction columns
    pred_cols = cfg.colmap or _detect_pred_cols(df)

    # aggregate per exam: Mirai predictions are per-view but should be evaluated per-exam
    # take mean of predictions across views for each exam
    # labels (years_to_cancer, years_to_last_followup) should be identical across views, take first
    agg_dict = {col: "mean" for col in pred_cols.values()}
    agg_dict["years_to_cancer"] = "first"
    agg_dict["years_to_last_followup"] = "first"
    if "split_group" in df.columns:
        agg_dict["split_group"] = "first"

    df_exam = df.groupby(["patient_id", "exam_id"], as_index=False).agg(agg_dict)

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
    return out


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


def survival_metrics(cfg: Config) -> dict:
    """compute censoring-adjusted metrics: Uno's C (IPCW), time-dependent AUC, IBS

    uses all available data to estimate censoring via Kaplan–Meier for IPCW
    """
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

    # evaluation set
    if cfg.split is not None and "split_group" in meta_all.columns:
        meta_eval = meta_all[
            meta_all["split_group"].astype(str).str.lower() == cfg.split.lower()
        ].copy()
    else:
        meta_eval = meta_all.copy()

    # ensure merge keys are strings
    meta_eval = meta_eval.assign(
        patient_id=meta_eval["patient_id"].astype(str),
        exam_id=meta_eval["exam_id"].astype(str),
    )
    meta_all = meta_all.assign(
        patient_id=meta_all["patient_id"].astype(str),
        exam_id=meta_all["exam_id"].astype(str),
    )

    dfe = pred.merge(meta_eval[list(req)], on=["patient_id", "exam_id"], how="inner")

    # aggregate per exam: Mirai predictions are per-view but should be evaluated per-exam
    pred_cols = cfg.colmap or _detect_pred_cols(dfe)
    agg_dict = {col: "mean" for col in pred_cols.values()}
    agg_dict["years_to_cancer"] = "first"
    agg_dict["years_to_last_followup"] = "first"
    dfe_exam = dfe.groupby(["patient_id", "exam_id"], as_index=False).agg(agg_dict)

    # aggregate meta_all per exam for IPCW estimation
    meta_all_exam = meta_all.groupby(["patient_id", "exam_id"], as_index=False).agg(
        {"years_to_cancer": "first", "years_to_last_followup": "first"}
    )

    # use all available data for IPCW censoring estimation
    y_train = _build_surv_arrays(meta_all_exam)
    y_test = _build_surv_arrays(dfe_exam)

    # prediction columns / times
    times = np.array(sorted(pred_cols.keys()), dtype=float)

    # matrix of risk estimates shape (n_test, n_times)
    risk = np.stack(
        [dfe_exam[pred_cols[h]].astype(float).to_numpy() for h in times], axis=1
    )

    # time-dependent AUC (Uno 2007) and weighted mean AUC (Lambert & Chevret 2014)
    auc_t, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk, times)

    # Uno's IPCW C-index at tau = max(times); pick a single risk column
    if cfg.cindex_col is not None:
        risk_for_c = dfe_exam[cfg.cindex_col].astype(float).to_numpy()
    else:
        risk_for_c = dfe_exam[pred_cols[int(times.max())]].astype(float).to_numpy()
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


def kfold_survival_metrics(cfg: Config) -> dict:
    """k-fold CV for survival metrics to assess IPCW sensitivity

    splits at patient level (not exam level) to avoid leakage. For each fold:
    - train fold estimates IPCW censoring distribution
    - test fold evaluates metrics using that IPCW estimate
    """
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

    # filter to evaluation set if split specified
    if cfg.split is not None and "split_group" in meta_all.columns:
        meta_eval = meta_all[
            meta_all["split_group"].astype(str).str.lower() == cfg.split.lower()
        ].copy()
    else:
        meta_eval = meta_all.copy()

    # ensure merge keys are strings
    meta_eval = meta_eval.assign(
        patient_id=meta_eval["patient_id"].astype(str),
        exam_id=meta_eval["exam_id"].astype(str),
    )

    # join predictions
    df = pred.merge(meta_eval[list(req)], on=["patient_id", "exam_id"], how="inner")

    # aggregate per exam: Mirai predictions are per-view but should be evaluated per-exam
    pred_cols = cfg.colmap or _detect_pred_cols(df)
    agg_dict = {col: "mean" for col in pred_cols.values()}
    agg_dict["years_to_cancer"] = "first"
    agg_dict["years_to_last_followup"] = "first"
    df_exam = df.groupby(["patient_id", "exam_id"], as_index=False).agg(agg_dict)

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


def reproduce_table2(
    cfg: Config, per_horizon_df: pd.DataFrame, surv_metrics: dict | None
) -> pd.DataFrame:
    """reproduce Table 2 from Omoleye et al: Model Performance Metrics

    combines per-horizon AUC and survival metrics into a single table
    """
    rows = []
    for _, row in per_horizon_df.iterrows():
        h = int(row["horizon_years"])
        rows.append(
            {
                "horizon_years": h,
                "n": int(row["n"]),
                "cases": int(row["cases"]),
                "controls": int(row["controls"]),
                "auc": row["auc"] if not pd.isna(row["auc"]) else None,
            }
        )

    df = pd.DataFrame(rows).sort_values("horizon_years").reset_index(drop=True)

    # add survival metrics if available
    if surv_metrics is not None:
        # handle k-fold vs single metrics
        if "mean_auc" in surv_metrics and isinstance(surv_metrics["mean_auc"], dict):
            # k-fold format
            mean_auc_val = surv_metrics["mean_auc"].get("mean")
            uno_c_val = (
                surv_metrics["uno_c"].get("mean") if "uno_c" in surv_metrics else None
            )
            ibs_val = surv_metrics["ibs"].get("mean") if "ibs" in surv_metrics else None
        else:
            # single metrics format
            mean_auc_val = surv_metrics.get("mean_auc")
            uno_c_val = surv_metrics.get("uno_c")
            ibs_val = surv_metrics.get("ibs")

        # add mean AUC and C-index as summary rows
        df.loc[len(df)] = {
            "horizon_years": "mean",
            "n": "",
            "cases": "",
            "controls": "",
            "auc": mean_auc_val,
        }
        df.loc[len(df)] = {
            "horizon_years": "Uno's C-index",
            "n": "",
            "cases": "",
            "controls": "",
            "auc": uno_c_val,
        }
        df.loc[len(df)] = {
            "horizon_years": "IBS",
            "n": "",
            "cases": "",
            "controls": "",
            "auc": ibs_val,
        }

    return df


def reproduce_table1(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """reproduce Table 1 from Omoleye et al: Patient- and Examination-level Characteristics

    computes statistics from mirai_manifest.csv merged with phenotype data
    """
    # load manifest
    meta = (
        pd.read_csv(cfg.meta_csv)
        if cfg.meta_csv.suffix.lower() == ".csv"
        else pd.read_parquet(cfg.meta_csv)
    )

    # align table with the requested evaluation split to avoid counting exams twice
    if cfg.split is not None and "split_group" in meta.columns:
        meta = meta[
            meta["split_group"].astype(str).str.lower() == cfg.split.lower()
        ].copy()

    # load phenotype CSV
    phenotype_csv = Path(
        "/gpfs/data/phs/groups/Projects/Huo_projects/SPORE/annawoodard/Phenotype_ChiMEC_2025Oct4.csv"
    )
    phenotype = pd.read_csv(phenotype_csv)
    phenotype["MRN"] = phenotype["MRN"].astype(str).str.strip().str.zfill(8)

    # load key file to map MRN → AnonymousID (patient_id)
    key_file = Path("/gpfs/data/huo-lab/Image/ChiMEC/study-16352a.csv")
    key = pd.read_csv(key_file)
    key["MRN"] = key["MRN"].astype(str).str.strip().str.zfill(8)
    key["AnonymousID"] = key["AnonymousID"].astype(str).str.strip()

    # merge phenotype with key to get patient_id
    phenotype = phenotype.merge(key[["MRN", "AnonymousID"]], on="MRN", how="left")
    phenotype = phenotype.rename(columns={"AnonymousID": "patient_id"})

    # merge phenotype with manifest
    meta = meta.assign(
        patient_id=meta["patient_id"].astype(str),
        exam_id=meta["exam_id"].astype(str),
    )
    meta = meta.merge(
        phenotype[["patient_id", "RaceEthnic", "dob"]],
        on="patient_id",
        how="left",
    )

    # try to get exam dates from exams parquet if available
    # study_date comes from DICOM tag (0x0008, 0x0020) extracted during preprocessing
    # dates may be missing if: (1) exam not in exams.parquet, (2) DICOM missing study_date tag,
    # or (3) patient_id/exam_id mismatch between manifest and exams.parquet
    exams_parquet = cfg.meta_csv.parent.parent / "sot" / "exams.parquet"
    print("\n=== Date Source Debugging ===")
    print(f"Exams in manifest before merge: {len(meta):,}")
    if exams_parquet.exists():
        exams = pd.read_parquet(exams_parquet)
        exams = exams.assign(
            patient_id=exams["patient_id"].astype(str),
            exam_id=exams["exam_id"].astype(str),
        )
        print(f"Exams in exams.parquet: {len(exams):,}")
        print(
            f"Exams in exams.parquet with study_date: {exams['study_date'].notna().sum():,}"
        )

        # check for matching keys
        meta_keys = set(zip(meta["patient_id"], meta["exam_id"]))
        exams_keys = set(zip(exams["patient_id"], exams["exam_id"]))
        matching_keys = meta_keys & exams_keys
        print(f"Exams with matching (patient_id, exam_id): {len(matching_keys):,}")

        meta = meta.merge(
            exams[["patient_id", "exam_id", "study_date"]],
            on=["patient_id", "exam_id"],
            how="left",
        )
        print(f"Exams in manifest after merge: {len(meta):,}")
        print(
            f"Exams with study_date after merge: {meta['study_date'].notna().sum():,}"
        )
        print(f"Exams missing study_date: {meta['study_date'].isna().sum():,}")

        # calculate age at exam from dob and study_date
        if "dob" in meta.columns and "study_date" in meta.columns:
            dob_parsed = pd.to_datetime(meta["dob"], format="%d%b%Y", errors="coerce")
            study_date_parsed = pd.to_datetime(meta["study_date"], errors="coerce")
            age_at_exam = (study_date_parsed - dob_parsed).dt.days / 365.25
            meta["age_at_exam"] = age_at_exam
            print(
                f"Exams with age_at_exam calculated: {meta['age_at_exam'].notna().sum():,}"
            )
    else:
        print(f"exams.parquet not found at {exams_parquet}")

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

    # Race/ethnicity - assign mutually exclusive categories
    if "RaceEthnic" in meta_filtered_exam.columns:
        rows.append(("Self-reported race and ethnicity", "", "", ""))

        def _assign_race_category(series: pd.Series) -> pd.Series:
            """map free-text race/ethnicity labels to mutually exclusive buckets"""
            race_str = series.astype(str).str.lower()
            race_norm = race_str.str.replace(r"\s+", " ", regex=True).str.strip()
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
            mask_white = race_norm.str.contains(
                "white", case=False, na=False, regex=True
            )

            category[mask_black] = "African American"
            category[mask_asian] = "Asian or Pacific Islander"
            category[mask_alaska] = "Alaska Native"
            category[is_hispanic] = "Hispanic"
            category[mask_white & ~is_hispanic] = "White"

            empty = race_norm.isna() | (race_norm == "")
            category[empty] = "Unknown"
            return category

        race_category = _assign_race_category(meta_filtered_exam["RaceEthnic"])
        race_category_case = _assign_race_category(case_filtered["RaceEthnic"])

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

    # reproduce Table 1
    try:
        table1, meta_exam, meta_filtered_exam = reproduce_table1(cfg)
        print("\n=== Table 1: Patient- and Examination-level Characteristics ===")
        print(table1.to_string(index=False))
        if cfg.out_json is not None:
            table1_path = (
                cfg.out_json.parent / "patient_examination_characteristics.csv"
            )
            table1.to_csv(table1_path, index=False)
            print(f"\nTable 1 saved to {table1_path}")

            # plot exam time histogram (using all exams, not just filtered)
            plot_path = cfg.out_json.parent / "exam_time_histogram.png"
            plot_exam_time_histogram(meta_exam, plot_path)
            print(f"Exam time histogram saved to {plot_path}")
    except Exception as e:
        print(f"\n[warn] Could not reproduce Table 1: {e}")
        meta_exam = None

    df = summarize(cfg)
    print("\n=== Per-Horizon AUC ===")
    cols = ["horizon_years", "n", "cases", "controls", "excluded", "prevalence", "auc"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    surv = None
    if cfg.kfold is not None:
        try:
            surv = kfold_survival_metrics(cfg)
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
            surv = survival_metrics(cfg)
            print("\n=== scikit-survival (IPCW) ===")
            print("times:", ", ".join(str(int(t)) for t in surv["times"]))
            print("auc(t):", ", ".join(f"{a:.4f}" for a in surv["auc_t"]))
            print(f"mean auc: {surv['mean_auc']:.4f}")
            print(f"uno c (tau=max time): {surv['uno_c']:.4f}")
            print("ibs:", f"{surv['ibs']:.4f}")
        except Exception as e:
            # allow quick use even if scikit-survival isn't installed or inputs are incomplete
            print("\n[warn] survival metrics unavailable:", str(e))

    # reproduce Table 2
    try:
        table2 = reproduce_table2(cfg, df, surv)
        print("\n=== Table 2: Model Performance Metrics ===")
        print(
            table2.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
                if x is not None and not pd.isna(x)
                else "",
            )
        )
        if cfg.out_json is not None:
            table2_path = cfg.out_json.parent / "model_performance_metrics.csv"
            table2.to_csv(table2_path, index=False)
            print(f"\nTable 2 saved to {table2_path}")
    except Exception as e:
        print(f"\n[warn] Could not reproduce Table 2: {e}")

    if cfg.out_json is not None:
        payload = df.to_dict(orient="records")
        out = {"per_horizon": payload}
        if surv is not None:
            out["survival_metrics"] = surv
        cfg.out_json.parent.mkdir(parents=True, exist_ok=True)
        cfg.out_json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
