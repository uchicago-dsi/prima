#!/usr/bin/env python3
"""MVP baseline: combine Mirai risk with PRS using patient-level CV.

This script is intentionally hardcoded (no CLI): paths and settings are defined
at module scope for repeatable reruns.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------
# hardcoded inputs and outputs
# ----------------------------
MIRAI_OUT_DIR = Path("/gpfs/data/huo-lab/Image/ChiMEC/MG/out")
MIRAI_PRED_CSV = MIRAI_OUT_DIR / "validation_output.csv"
MIRAI_META_CSV = MIRAI_OUT_DIR / "mirai_manifest.csv"

PRS_SCORES_PATH = Path(
    "/gpfs/data/huo-lab/AABCG/Yijia.Sun/LynnSage/01_PRS_Scores_testing_validation_combined/all_combined/ALL_MODELS_combined.cPRS.cOR"
)
PRS_IID_MRN_MAP_PATH = Path(
    "/gpfs/data/huo-lab/AABCG/Yijia.Sun/LynnSage/AABCG_ChiMEC_IDs/aabcg_chimec_MRN.csv"
)
STUDY_KEY_PATH = Path("/gpfs/data/huo-lab/Image/ChiMEC/study-16352a.csv")

PRS_SCORE_COLUMN = "cPRS_ERNEG_XANCESTRY_FSS_0.05"
HORIZONS = (2, 5)
N_FOLDS = 5
RANDOM_SEED = 42

OUTPUT_DIR = MIRAI_OUT_DIR / "prs_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MERGED_CSV = OUTPUT_DIR / "mirai_prs_exam_level.csv"
OOF_PRED_CSV = OUTPUT_DIR / "cv_oof_predictions.csv"
METRICS_CSV = OUTPUT_DIR / "cv_metrics.csv"
METRICS_JSON = OUTPUT_DIR / "cv_metrics.json"


def _split_patient_exam_id(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split patient_exam_id into patient_id and exam_id."""
    split = series.astype(str).str.split("\t", n=1, expand=True)
    return split[0].astype(str), split[1].astype(str)


def _normalize_mrn(series: pd.Series) -> pd.Series:
    """Normalize MRN values to zero-padded digit strings."""
    return (
        series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(8)
    )


def load_mirai_exam_level() -> pd.DataFrame:
    """Load Mirai predictions and aggregate to one row per exam."""
    pred = pd.read_csv(MIRAI_PRED_CSV)
    patient_id, exam_id = _split_patient_exam_id(pred["patient_exam_id"])
    pred = pred.assign(patient_id=patient_id, exam_id=exam_id)

    meta = pd.read_csv(MIRAI_META_CSV).assign(
        patient_id=lambda d: d["patient_id"].astype(str),
        exam_id=lambda d: d["exam_id"].astype(str),
    )

    labels = meta[
        ["patient_id", "exam_id", "years_to_cancer", "years_to_last_followup"]
    ].drop_duplicates()

    merged = pred.merge(labels, on=["patient_id", "exam_id"], how="inner")
    risk_cols = [f"{h}_year_risk" for h in HORIZONS]
    exam = (
        merged.groupby(["patient_id", "exam_id"], as_index=False)
        .agg(
            {
                **{col: "mean" for col in risk_cols},
                "years_to_cancer": "first",
                "years_to_last_followup": "first",
            }
        )
        .reset_index(drop=True)
    )
    return exam


def load_prs_patient_level() -> pd.DataFrame:
    """Load PRS scores and map IID -> MRN -> AnonymousID (patient_id)."""
    prs = pd.read_csv(PRS_SCORES_PATH, sep="\t")
    iid_mrn = pd.read_csv(PRS_IID_MRN_MAP_PATH)
    key = pd.read_csv(STUDY_KEY_PATH)

    iid_mrn = iid_mrn.assign(
        IID=lambda d: d["IID"].astype(str).str.strip(),
        MRN=lambda d: _normalize_mrn(d["MRN"]),
    )
    key = key.assign(
        MRN=lambda d: _normalize_mrn(d["MRN"]),
        AnonymousID=lambda d: d["AnonymousID"].astype(str).str.strip(),
    )

    prs_patient = (
        prs.assign(IID=lambda d: d["IID"].astype(str).str.strip())[
            ["IID", PRS_SCORE_COLUMN]
        ]
        .dropna(subset=[PRS_SCORE_COLUMN])
        .drop_duplicates(subset=["IID"])
        .merge(iid_mrn[["IID", "MRN"]].drop_duplicates(subset=["IID"]), on="IID")
        .merge(key[["MRN", "AnonymousID"]].drop_duplicates(subset=["MRN"]), on="MRN")
        .rename(columns={"AnonymousID": "patient_id", PRS_SCORE_COLUMN: "prs_score"})
    )
    return prs_patient[["patient_id", "prs_score"]].drop_duplicates(
        subset=["patient_id"]
    )


def build_analysis_table() -> pd.DataFrame:
    """Build exam-level table with Mirai risks + PRS + label times."""
    exam = load_mirai_exam_level()
    prs_patient = load_prs_patient_level()
    df = exam.merge(prs_patient, on="patient_id", how="inner")
    return df


def build_horizon_label(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Build binary label at fixed horizon using case/control eligibility."""
    ytc = df["years_to_cancer"].astype(float)
    ylf = df["years_to_last_followup"].astype(float)

    is_case = ytc <= float(horizon)
    is_control = (ytc > float(horizon)) & (ylf >= float(horizon))
    eligible = is_case | is_control

    out = df.loc[eligible].copy()
    out["label"] = is_case.loc[eligible].astype(int).to_numpy()
    return out


def cv_auc(
    df: pd.DataFrame, feature_cols: list[str], horizon: int, model_name: str
) -> tuple[dict, pd.DataFrame]:
    """Run patient-level CV and return metrics + OOF predictions."""
    work = df.dropna(subset=feature_cols + ["label"]).copy()
    if work.empty:
        raise ValueError(
            f"No samples after dropping NaNs for features={feature_cols} at horizon={horizon}"
        )
    X = work[feature_cols].to_numpy()
    y = work["label"].to_numpy()
    groups = work["patient_id"].to_numpy()

    n_groups = int(work["patient_id"].nunique())
    n_case_groups = int(
        work.loc[work["label"] == 1, "patient_id"].astype(str).nunique()
    )
    n_ctrl_groups = int(
        work.loc[work["label"] == 0, "patient_id"].astype(str).nunique()
    )
    n_splits = min(N_FOLDS, n_groups, n_case_groups, n_ctrl_groups)
    if n_splits < 2:
        raise ValueError(
            f"Not enough patient groups for CV at horizon={horizon}, model={model_name}: "
            f"n_groups={n_groups}, n_case_groups={n_case_groups}, n_ctrl_groups={n_ctrl_groups}"
        )

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

    oof_pred = np.full(len(work), np.nan, dtype=float)
    fold_rows: list[dict] = []

    for fold_idx, (tr, te) in enumerate(cv.split(X, y, groups), start=1):
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict_proba(X[te])[:, 1]
        oof_pred[te] = pred
        fold_rows.append(
            {
                "horizon_years": horizon,
                "model": model_name,
                "fold": fold_idx,
                "n_test": int(len(te)),
                "auc": float(roc_auc_score(y[te], pred)),
            }
        )

    overall_auc = float(roc_auc_score(y, oof_pred))
    metrics = {
        "horizon_years": horizon,
        "model": model_name,
        "n_exams": int(len(work)),
        "n_patients": int(work["patient_id"].nunique()),
        "n_cases": int(y.sum()),
        "n_controls": int((1 - y).sum()),
        "auc_mean_fold": float(np.mean([r["auc"] for r in fold_rows])),
        "auc_std_fold": float(np.std([r["auc"] for r in fold_rows], ddof=1)),
        "auc_oof": overall_auc,
    }

    oof = work[["patient_id", "exam_id"]].copy()
    oof["horizon_years"] = horizon
    oof["model"] = model_name
    oof["label"] = y
    oof["pred"] = oof_pred
    return metrics, oof


def run() -> None:
    """Execute end-to-end MVP baseline pipeline."""
    df = build_analysis_table()
    if df.empty:
        exam = load_mirai_exam_level()
        prs_patient = load_prs_patient_level()
        raise ValueError(
            "Mirai/PRS merge is empty. "
            f"Mirai exams={len(exam):,} ({exam['patient_id'].nunique():,} patients), "
            f"PRS mapped patients={prs_patient['patient_id'].nunique():,}. "
            "Check ID mapping inputs."
        )
    df.to_csv(MERGED_CSV, index=False)

    all_metrics: list[dict] = []
    all_oof: list[pd.DataFrame] = []

    for horizon in HORIZONS:
        labeled = build_horizon_label(df, horizon)
        mirai_col = f"{horizon}_year_risk"

        configs = [
            ("mirai_only", [mirai_col]),
            ("prs_only", ["prs_score"]),
            ("mirai_plus_prs", [mirai_col, "prs_score"]),
        ]
        for model_name, feature_cols in configs:
            metrics, oof = cv_auc(labeled, feature_cols, horizon, model_name)
            all_metrics.append(metrics)
            all_oof.append(oof)

    metrics_df = pd.DataFrame(all_metrics).sort_values(["horizon_years", "model"])
    oof_df = pd.concat(all_oof, axis=0, ignore_index=True)

    metrics_df.to_csv(METRICS_CSV, index=False)
    oof_df.to_csv(OOF_PRED_CSV, index=False)
    METRICS_JSON.write_text(json.dumps(metrics_df.to_dict(orient="records"), indent=2))

    print(f"Wrote merged exam-level table: {MERGED_CSV}")
    print(f"Wrote CV metrics: {METRICS_CSV}")
    print(f"Wrote CV metrics JSON: {METRICS_JSON}")
    print(f"Wrote OOF predictions: {OOF_PRED_CSV}")
    print()
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    run()
