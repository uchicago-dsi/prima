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

    # decide prediction columns
    pred_cols = cfg.colmap or _detect_pred_cols(df)

    # vectorized label tensors
    ytc = df["years_to_cancer"].astype(float).to_numpy()
    ylf = df["years_to_last_followup"].astype(float).to_numpy()

    rows = []
    for h, col in pred_cols.items():
        scores = df[col].astype(float).to_numpy()
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

    # use all available data for IPCW censoring estimation
    y_train = _build_surv_arrays(meta_all)
    y_test = _build_surv_arrays(dfe)

    # prediction columns / times
    pred_cols = cfg.colmap or _detect_pred_cols(pd.concat([dfe, pred], axis=1))
    times = np.array(sorted(pred_cols.keys()), dtype=float)

    # matrix of risk estimates shape (n_test, n_times)
    risk = np.stack([dfe[pred_cols[h]].astype(float).to_numpy() for h in times], axis=1)

    # time-dependent AUC (Uno 2007) and weighted mean AUC (Lambert & Chevret 2014)
    auc_t, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk, times)

    # Uno's IPCW C-index at tau = max(times); pick a single risk column
    if cfg.cindex_col is not None:
        risk_for_c = dfe[cfg.cindex_col].astype(float).to_numpy()
    else:
        risk_for_c = dfe[pred_cols[int(times.max())]].astype(float).to_numpy()
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

    # get unique patients for splitting
    patients = df["patient_id"].unique()
    kf = KFold(n_splits=cfg.kfold, shuffle=True, random_state=42)

    # detect prediction columns once
    pred_cols = cfg.colmap or _detect_pred_cols(df)
    times = np.array(sorted(pred_cols.keys()), dtype=float)

    fold_results = []
    for fold_idx, (train_pat_idx, test_pat_idx) in enumerate(kf.split(patients)):
        train_patients = patients[train_pat_idx]
        test_patients = patients[test_pat_idx]

        df_train = df[df["patient_id"].isin(train_patients)].copy()
        df_test = df[df["patient_id"].isin(test_patients)].copy()

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


def main() -> None:
    """entrypoint"""
    cfg = parse_args()
    df = summarize(cfg)
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

    if cfg.out_json is not None:
        payload = df.to_dict(orient="records")
        out = {"per_horizon": payload}
        if surv is not None:
            out["survival_metrics"] = surv
        cfg.out_json.parent.mkdir(parents=True, exist_ok=True)
        cfg.out_json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
