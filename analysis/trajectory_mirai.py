#!/usr/bin/env python3
"""Analyze longitudinal Mirai score trajectories toward diagnosis."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_INPUT = Path(
    "/gpfs/data/huo-lab/Image/ChiMEC/MG/out/validation_output_clinical.csv"
)
DEFAULT_OUT_DIR = Path("/gpfs/data/huo-lab/Image/ChiMEC/MG/out/trajectory_analysis")
DEFAULT_SCORE = "1_year_risk"
DEFAULT_BINS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, np.inf]
SENSITIVITY_WINDOWS_DAYS = [0, 90, 180]
HORIZON_SCORES = ["1_year_risk", "2_year_risk", "5_year_risk"]
AGE_BINS = [0.0, 50.0, 60.0, 70.0, 80.0, np.inf]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--score-col", type=str, default=DEFAULT_SCORE)
    p.add_argument(
        "--max-spaghetti-patients",
        type=int,
        default=40,
        help="Number of patients to show in the spaghetti plot.",
    )
    return p.parse_args()


def _parse_study_date(series: pd.Series) -> pd.Series:
    x = series.astype(str).str.strip()
    ymd = pd.to_datetime(x, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(x[ymd.isna()], errors="coerce")
    ymd = ymd.copy()
    ymd.loc[ymd.isna()] = fallback
    return ymd


def _first_available_date(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for col in cols:
        if col not in df.columns:
            continue
        if col.endswith("_parsed"):
            parsed = pd.to_datetime(df[col], errors="coerce")
        elif col == "study_date":
            parsed = _parse_study_date(df[col])
        else:
            parsed = pd.to_datetime(df[col], errors="coerce")
        out = out.fillna(parsed)
    return out


def _is_case(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.contains("case", na=False)


def build_exam_table(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    req = {"patient_id", "exam_id", "study_date", "years_to_cancer", score_col}
    missing = sorted(req - set(df.columns))
    if missing:
        raise KeyError(f"input missing columns: {missing}")

    work = df.copy()
    work["patient_id"] = work["patient_id"].astype(str).str.strip()
    work["exam_id"] = work["exam_id"].astype(str).str.strip()
    work["study_date_ts"] = _first_available_date(work, ["study_date"])
    work["dx_date_ts"] = _first_available_date(
        work,
        [
            "date_diagnosis_parsed",
            "datedx_new_parsed",
            "datedx_parsed",
            "date_diagnosis",
            "datedx_new",
            "datedx",
        ],
    )
    work["is_case_patient"] = _is_case(
        work.get("CaseControl", pd.Series("", index=work.index))
    )
    work["score"] = pd.to_numeric(work[score_col], errors="coerce")
    work["years_to_cancer"] = pd.to_numeric(work["years_to_cancer"], errors="coerce")
    work["years_to_last_followup"] = pd.to_numeric(
        work.get("years_to_last_followup", pd.Series(np.nan, index=work.index)),
        errors="coerce",
    )

    # Prefer raw dates when present; fall back to years_to_cancer.
    date_based = (work["dx_date_ts"] - work["study_date_ts"]).dt.days / 365.25
    work["time_to_dx_years"] = date_based
    missing_time = work["time_to_dx_years"].isna()
    work.loc[missing_time, "time_to_dx_years"] = work.loc[
        missing_time, "years_to_cancer"
    ]

    work["is_prediagnostic"] = work["time_to_dx_years"].notna() & (
        work["time_to_dx_years"] >= 0
    )
    work = work.dropna(subset=["patient_id", "exam_id", "score", "study_date_ts"])
    work = work.drop_duplicates(subset=["patient_id", "exam_id"]).copy()

    work = work.sort_values(["patient_id", "study_date_ts", "exam_id"]).reset_index(
        drop=True
    )
    work["exam_order"] = work.groupby("patient_id").cumcount() + 1
    work["n_exams_per_patient"] = work.groupby("patient_id")["exam_id"].transform(
        "size"
    )
    first_exam = work.groupby("patient_id")["study_date_ts"].transform("min")
    last_exam = work.groupby("patient_id")["study_date_ts"].transform("max")
    work["time_from_first_exam_years"] = (
        work["study_date_ts"] - first_exam
    ).dt.days / 365.25
    work["time_to_last_exam_years"] = (
        last_exam - work["study_date_ts"]
    ).dt.days / 365.25
    return work


def build_case_table(exams: pd.DataFrame) -> pd.DataFrame:
    has_case_timing = exams["dx_date_ts"].notna() | (exams["years_to_cancer"] < 100)
    case_df = exams[
        exams["is_case_patient"] & exams["is_prediagnostic"] & has_case_timing
    ].copy()
    case_df = case_df.sort_values(
        ["patient_id", "study_date_ts", "exam_id"]
    ).reset_index(drop=True)
    case_df["n_case_exams_per_patient"] = case_df.groupby("patient_id")[
        "exam_id"
    ].transform("size")
    span = case_df.groupby("patient_id")["time_to_dx_years"].agg(["min", "max"])
    span["trajectory_span_years"] = span["max"] - span["min"]
    case_df = case_df.merge(
        span[["trajectory_span_years"]],
        left_on="patient_id",
        right_index=True,
        how="left",
    )
    return case_df


def build_control_table(exams: pd.DataFrame) -> pd.DataFrame:
    control_df = exams[~exams["is_case_patient"]].copy()
    control_df = control_df.sort_values(
        ["patient_id", "study_date_ts", "exam_id"]
    ).reset_index(drop=True)
    control_df["n_control_exams_per_patient"] = control_df.groupby("patient_id")[
        "exam_id"
    ].transform("size")
    return control_df


def filter_case_table(case_df: pd.DataFrame, min_days_to_dx: int) -> pd.DataFrame:
    if min_days_to_dx <= 0:
        return case_df.copy()
    min_years = min_days_to_dx / 365.25
    return case_df[case_df["time_to_dx_years"] >= min_years].copy()


def summarize_cohort(case_df: pd.DataFrame) -> pd.DataFrame:
    by_patient = (
        case_df.groupby("patient_id")
        .agg(
            n_exams=("exam_id", "size"),
            earliest_time_to_dx=("time_to_dx_years", "max"),
            latest_time_to_dx=("time_to_dx_years", "min"),
            trajectory_span_years=("trajectory_span_years", "first"),
        )
        .reset_index()
    )

    def _q(s: pd.Series, q: float) -> float:
        return float(s.quantile(q)) if len(s) else float("nan")

    counts = by_patient["n_exams"].value_counts().to_dict()
    summary = pd.DataFrame(
        [
            {
                "metric": "n_case_patients",
                "value": int(by_patient["patient_id"].nunique()),
            },
            {"metric": "n_prediagnostic_case_exams", "value": int(len(case_df))},
            {
                "metric": "median_exams_per_patient",
                "value": float(by_patient["n_exams"].median()),
            },
            {
                "metric": "iqr_exams_per_patient_low",
                "value": _q(by_patient["n_exams"], 0.25),
            },
            {
                "metric": "iqr_exams_per_patient_high",
                "value": _q(by_patient["n_exams"], 0.75),
            },
            {
                "metric": "max_exams_per_patient",
                "value": int(by_patient["n_exams"].max()) if len(by_patient) else 0,
            },
            {
                "metric": "median_earliest_predx_span_years",
                "value": float(by_patient["earliest_time_to_dx"].median())
                if len(by_patient)
                else float("nan"),
            },
            {
                "metric": "iqr_earliest_predx_span_years_low",
                "value": _q(by_patient["earliest_time_to_dx"], 0.25),
            },
            {
                "metric": "iqr_earliest_predx_span_years_high",
                "value": _q(by_patient["earliest_time_to_dx"], 0.75),
            },
            {"metric": "patients_with_1_exam", "value": int(counts.get(1, 0))},
            {"metric": "patients_with_2_exams", "value": int(counts.get(2, 0))},
            {"metric": "patients_with_3_exams", "value": int(counts.get(3, 0))},
            {
                "metric": "patients_with_4plus_exams",
                "value": int(sum(v for k, v in counts.items() if k >= 4)),
            },
        ]
    )
    return summary


def _bin_label(left: float, right: float) -> str:
    if np.isinf(right):
        return f"({left}, inf]"
    return f"({left}, {right}]"


def _assign_bins(series: pd.Series, bins: list[float]) -> pd.Series:
    labels = [_bin_label(left, right) for left, right in zip(bins[:-1], bins[1:])]
    return pd.cut(series, bins=bins, labels=labels, right=True, include_lowest=False)


def build_binned_trajectory(
    df: pd.DataFrame, bins: list[float], time_col: str
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (df[time_col] > left) & (df[time_col] <= right)
        chunk = df.loc[mask].copy()
        if chunk.empty:
            rows.append(
                {
                    "time_bin": _bin_label(left, right),
                    "bin_left": left,
                    "bin_right": right,
                    "n_patients": 0,
                    "n_exams": 0,
                    "weighted_mean_score": np.nan,
                    "weighted_se": np.nan,
                    "weighted_ci_lower": np.nan,
                    "weighted_ci_upper": np.nan,
                }
            )
            continue

        per_patient = chunk.groupby("patient_id").agg(
            patient_weighted_score=(
                "score",
                lambda s: np.nan if len(s) == 0 else float(np.mean(s)),
            ),
        )
        # Patient-weighted mean equals mean of within-bin patient means.
        patient_scores = per_patient["patient_weighted_score"].dropna()
        mean = float(patient_scores.mean())
        se = (
            float(patient_scores.std(ddof=1) / np.sqrt(len(patient_scores)))
            if len(patient_scores) > 1
            else np.nan
        )
        rows.append(
            {
                "time_bin": _bin_label(left, right),
                "bin_left": left,
                "bin_right": right,
                "n_patients": int(chunk["patient_id"].nunique()),
                "n_exams": int(len(chunk)),
                "weighted_mean_score": mean,
                "weighted_se": se,
                "weighted_ci_lower": mean - 1.96 * se if pd.notna(se) else np.nan,
                "weighted_ci_upper": mean + 1.96 * se if pd.notna(se) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_age_standardized_binned_trajectory(
    df: pd.DataFrame,
    bins: list[float],
    time_col: str,
    age_col: str,
    target_age_weights: pd.Series | None = None,
    target_age_weights_by_time: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    work = df.dropna(subset=[time_col, age_col, "score"]).copy()
    if work.empty:
        return pd.DataFrame()
    work["time_bin"] = _assign_bins(work[time_col], bins)
    work["age_bin"] = _assign_bins(work[age_col], AGE_BINS)
    work = work.dropna(subset=["time_bin", "age_bin"]).copy()
    rows: list[dict[str, float | int | str]] = []

    for left, right in zip(bins[:-1], bins[1:]):
        time_bin = _bin_label(left, right)
        chunk = work[work["time_bin"] == time_bin].copy()
        if chunk.empty:
            rows.append(
                {
                    "time_bin": time_bin,
                    "bin_left": left,
                    "bin_right": right,
                    "n_patients": 0,
                    "n_exams": 0,
                    "age_standardized_mean_score": np.nan,
                    "age_standardized_se": np.nan,
                    "age_standardized_ci_lower": np.nan,
                    "age_standardized_ci_upper": np.nan,
                    "age_standardized_n_age_bins": 0,
                }
            )
            continue

        if target_age_weights_by_time is not None:
            weights = target_age_weights_by_time.get(time_bin)
        else:
            weights = target_age_weights
        if weights is None or len(weights) == 0:
            rows.append(
                {
                    "time_bin": time_bin,
                    "bin_left": left,
                    "bin_right": right,
                    "n_patients": int(chunk["patient_id"].nunique()),
                    "n_exams": int(len(chunk)),
                    "age_standardized_mean_score": np.nan,
                    "age_standardized_se": np.nan,
                    "age_standardized_ci_lower": np.nan,
                    "age_standardized_ci_upper": np.nan,
                    "age_standardized_n_age_bins": 0,
                }
            )
            continue

        per_patient = (
            chunk.groupby(["age_bin", "patient_id"], observed=False)["score"]
            .mean()
            .reset_index()
        )
        age_stats = (
            per_patient.groupby("age_bin", observed=False)["score"]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "age_mean", "std": "age_sd", "count": "age_n"})
        )
        common_bins = [
            age_bin for age_bin in weights.index if age_bin in age_stats.index
        ]
        if not common_bins:
            rows.append(
                {
                    "time_bin": time_bin,
                    "bin_left": left,
                    "bin_right": right,
                    "n_patients": int(chunk["patient_id"].nunique()),
                    "n_exams": int(len(chunk)),
                    "age_standardized_mean_score": np.nan,
                    "age_standardized_se": np.nan,
                    "age_standardized_ci_lower": np.nan,
                    "age_standardized_ci_upper": np.nan,
                    "age_standardized_n_age_bins": 0,
                }
            )
            continue

        w = weights.loc[common_bins].astype(float)
        w = w / w.sum()
        age_means = age_stats.loc[common_bins, "age_mean"].to_numpy(dtype=float)
        age_ses = []
        for age_bin in common_bins:
            sd = age_stats.loc[age_bin, "age_sd"]
            n = age_stats.loc[age_bin, "age_n"]
            if pd.isna(sd) or n <= 1:
                age_ses.append(0.0)
            else:
                age_ses.append(float(sd / np.sqrt(n)))
        age_ses = np.asarray(age_ses, dtype=float)
        mean = float(np.sum(age_means * w.to_numpy()))
        se = float(np.sqrt(np.sum(np.square(w.to_numpy()) * np.square(age_ses))))
        rows.append(
            {
                "time_bin": time_bin,
                "bin_left": left,
                "bin_right": right,
                "n_patients": int(chunk["patient_id"].nunique()),
                "n_exams": int(len(chunk)),
                "age_standardized_mean_score": mean,
                "age_standardized_se": se,
                "age_standardized_ci_lower": mean - 1.96 * se
                if pd.notna(se)
                else np.nan,
                "age_standardized_ci_upper": mean + 1.96 * se
                if pd.notna(se)
                else np.nan,
                "age_standardized_n_age_bins": int(len(common_bins)),
            }
        )
    return pd.DataFrame(rows)


def compute_age_weights(df: pd.DataFrame, age_col: str) -> pd.Series:
    work = df.dropna(subset=[age_col]).copy()
    if work.empty:
        return pd.Series(dtype=float)
    age_bin = _assign_bins(work[age_col], AGE_BINS)
    counts = age_bin.value_counts(normalize=True, sort=False)
    counts = counts[counts > 0]
    return counts.astype(float)


def compute_age_weights_by_time_bin(
    df: pd.DataFrame, bins: list[float], time_col: str, age_col: str
) -> dict[str, pd.Series]:
    work = df.dropna(subset=[time_col, age_col]).copy()
    if work.empty:
        return {}
    work["time_bin"] = _assign_bins(work[time_col], bins)
    work["age_bin"] = _assign_bins(work[age_col], AGE_BINS)
    work = work.dropna(subset=["time_bin", "age_bin"]).copy()
    out: dict[str, pd.Series] = {}
    for time_bin, grp in work.groupby("time_bin", observed=False):
        if pd.isna(time_bin) or grp.empty:
            continue
        counts = grp["age_bin"].value_counts(normalize=True, sort=False)
        counts = counts[counts > 0].astype(float)
        out[str(time_bin)] = counts
    return out


def build_horizon_trajectories(
    df: pd.DataFrame, score_cols: list[str], bins: list[float], time_col: str
) -> pd.DataFrame:
    outputs = []
    base = df.copy()
    for score_col in score_cols:
        if score_col not in base.columns:
            continue
        tmp = base.copy()
        tmp["score"] = pd.to_numeric(tmp[score_col], errors="coerce")
        tmp = tmp.dropna(subset=["score"])
        traj = build_binned_trajectory(tmp, bins=bins, time_col=time_col)
        traj["score_col"] = score_col
        outputs.append(traj)
    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True)


def build_patient_slopes(
    df: pd.DataFrame,
    time_col: str = "time_to_dx_years",
    slope_col: str = "slope_score_per_year_to_dx",
) -> pd.DataFrame:
    rows = []
    for patient_id, grp in df.groupby("patient_id"):
        patient_exam_count = int(len(grp))
        grp = grp.dropna(subset=[time_col, "score"]).copy()
        if len(grp) < 2:
            continue
        grp = (
            grp.groupby(time_col, as_index=False)
            .agg(
                score=("score", "mean"),
                n_exams_at_time=("exam_id", "size"),
            )
            .sort_values(time_col)
        )
        if len(grp) < 2:
            continue
        x = grp[time_col].to_numpy(dtype=float)
        y = grp["score"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        rows.append(
            {
                "patient_id": patient_id,
                "n_exams": patient_exam_count,
                "n_timepoints": int(len(grp)),
                "time_min": float(np.min(x)),
                "time_max": float(np.max(x)),
                "trajectory_span_years": float(np.max(x) - np.min(x)),
                slope_col: float(slope),
                "intercept": float(intercept),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(slope_col).reset_index(drop=True)


def fit_patient_fixed_effects(
    df: pd.DataFrame, time_col: str = "time_to_dx_years"
) -> dict[str, float | int]:
    work = df.dropna(subset=["patient_id", time_col, "score"]).copy()
    if work.empty:
        return {
            "n_patients": 0,
            "n_exams": 0,
            "beta_time": np.nan,
            "se_cluster": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        }

    work["x_dm"] = work[time_col] - work.groupby("patient_id")[time_col].transform(
        "mean"
    )
    work["y_dm"] = work["score"] - work.groupby("patient_id")["score"].transform("mean")
    work = work[work["x_dm"] != 0].copy()
    if work.empty:
        return {
            "n_patients": int(df["patient_id"].nunique()),
            "n_exams": int(len(df)),
            "beta_time": np.nan,
            "se_cluster": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        }

    x = work["x_dm"].to_numpy(dtype=float)
    y = work["y_dm"].to_numpy(dtype=float)
    xtx = float(np.dot(x, x))
    beta = float(np.dot(x, y) / xtx)
    resid = y - beta * x

    cluster_scores = []
    for _, grp in work.assign(resid=resid).groupby("patient_id"):
        xg = grp["x_dm"].to_numpy(dtype=float)
        eg = grp["resid"].to_numpy(dtype=float)
        cluster_scores.append(float(np.dot(xg, eg)))

    g = len(cluster_scores)
    n = len(work)
    if g > 1 and n > 1:
        meat = float(np.sum(np.square(cluster_scores)))
        vcov = (1.0 / xtx) ** 2 * meat
        vcov *= (g / (g - 1)) * ((n - 1) / max(n - 1, 1))
        se = float(np.sqrt(vcov))
    else:
        se = np.nan

    return {
        "n_patients": int(work["patient_id"].nunique()),
        "n_exams": int(len(df)),
        "beta_time": beta,
        "se_cluster": se,
        "ci_lower": beta - 1.96 * se if pd.notna(se) else np.nan,
        "ci_upper": beta + 1.96 * se if pd.notna(se) else np.nan,
    }


def build_sensitivity_summary(case_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for min_days in SENSITIVITY_WINDOWS_DAYS:
        subset = filter_case_table(case_df, min_days)
        slopes = build_patient_slopes(subset)
        trend = fit_patient_fixed_effects(subset)
        rows.append(
            {
                "min_days_to_dx": min_days,
                "n_case_patients": int(subset["patient_id"].nunique()),
                "n_case_exams": int(len(subset)),
                "n_slope_patients": int(len(slopes)),
                "fraction_negative_slopes": float(
                    (slopes["slope_score_per_year_to_dx"] < 0).mean()
                )
                if len(slopes)
                else np.nan,
                "median_slope": float(slopes["slope_score_per_year_to_dx"].median())
                if len(slopes)
                else np.nan,
                "fe_beta_time_to_dx": trend["beta_time"],
                "fe_se_cluster": trend["se_cluster"],
                "fe_ci_lower": trend["ci_lower"],
                "fe_ci_upper": trend["ci_upper"],
            }
        )
    return pd.DataFrame(rows)


def build_control_trend_summary(
    control_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    slopes = build_patient_slopes(
        control_df,
        time_col="time_to_last_exam_years",
        slope_col="slope_score_per_year_to_last_exam",
    )
    trend = fit_patient_fixed_effects(control_df, time_col="time_to_last_exam_years")
    summary = pd.DataFrame(
        [
            {
                "n_control_patients": int(control_df["patient_id"].nunique()),
                "n_control_exams": int(len(control_df)),
                "n_slope_patients": int(len(slopes)),
                "fraction_negative_slopes": float(
                    (slopes["slope_score_per_year_to_last_exam"] < 0).mean()
                )
                if len(slopes)
                else np.nan,
                "median_slope": float(
                    slopes["slope_score_per_year_to_last_exam"].median()
                )
                if len(slopes)
                else np.nan,
                "fe_beta_time_to_last_exam": trend["beta_time"],
                "fe_se_cluster": trend["se_cluster"],
                "fe_ci_lower": trend["ci_lower"],
                "fe_ci_upper": trend["ci_upper"],
            }
        ]
    )
    return slopes, summary


def bootstrap_case_trajectory(
    case_df: pd.DataFrame,
    bins: list[float],
    n_boot: int = 40,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_ids = case_df["patient_id"].dropna().astype(str).unique()
    rng = np.random.default_rng(seed)
    boot_rows: list[dict[str, float | int | str]] = []

    grouped = {
        pid: grp.copy()
        for pid, grp in case_df.groupby(case_df["patient_id"].astype(str))
    }
    for boot_idx in range(n_boot):
        sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        parts = []
        for draw_idx, pid in enumerate(sampled):
            grp = grouped[pid].copy()
            grp["patient_id"] = f"{pid}__boot{boot_idx}_{draw_idx}"
            parts.append(grp)
        sample_df = pd.concat(parts, ignore_index=True)
        binned = build_binned_trajectory(
            sample_df, bins=bins, time_col="time_to_dx_years"
        )
        trend = fit_patient_fixed_effects(sample_df, time_col="time_to_dx_years")
        for _, row in binned.iterrows():
            boot_rows.append(
                {
                    "bootstrap_idx": boot_idx,
                    "time_bin": row["time_bin"],
                    "weighted_mean_score": row["weighted_mean_score"],
                    "fe_beta_time_to_dx": trend["beta_time"],
                }
            )

    boot_df = pd.DataFrame(boot_rows)
    summary = boot_df.groupby("time_bin", as_index=False).agg(
        bootstrap_mean=("weighted_mean_score", "mean"),
        bootstrap_ci_lower=(
            "weighted_mean_score",
            lambda s: float(np.quantile(s.dropna(), 0.025))
            if s.notna().any()
            else np.nan,
        ),
        bootstrap_ci_upper=(
            "weighted_mean_score",
            lambda s: float(np.quantile(s.dropna(), 0.975))
            if s.notna().any()
            else np.nan,
        ),
    )
    beta_series = (
        boot_df[["bootstrap_idx", "fe_beta_time_to_dx"]]
        .drop_duplicates()["fe_beta_time_to_dx"]
        .dropna()
    )
    beta_summary = pd.DataFrame(
        [
            {
                "n_boot": n_boot,
                "bootstrap_beta_mean": float(beta_series.mean())
                if len(beta_series)
                else np.nan,
                "bootstrap_beta_ci_lower": float(np.quantile(beta_series, 0.025))
                if len(beta_series)
                else np.nan,
                "bootstrap_beta_ci_upper": float(np.quantile(beta_series, 0.975))
                if len(beta_series)
                else np.nan,
            }
        ]
    )
    return summary, beta_summary


def plot_binned_trajectory(df: pd.DataFrame, out_path: Path, score_col: str) -> None:
    valid = df.dropna(subset=["weighted_mean_score"]).copy()
    if valid.empty:
        return
    valid = valid.sort_values("bin_left", ascending=False).reset_index(drop=True)
    x = np.arange(len(valid))
    y = valid["weighted_mean_score"].to_numpy()
    lower = valid["weighted_ci_lower"].to_numpy()
    upper = valid["weighted_ci_upper"].to_numpy()
    yerr = np.vstack([y - lower, upper - y])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=2, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(valid["time_bin"], rotation=30, ha="right")
    ax.set_xlabel("Aligned time bin (farther from diagnosis to closer to diagnosis)")
    ax.set_ylabel(score_col)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_case_control_trajectory(
    case_df: pd.DataFrame,
    control_df: pd.DataFrame,
    out_path: Path,
    score_col: str,
    case_label: str = "Cases aligned to diagnosis",
    control_label: str = "Controls aligned to last exam",
    value_col: str = "weighted_mean_score",
    case_lower_col: str | None = "weighted_ci_lower",
    case_upper_col: str | None = "weighted_ci_upper",
    control_lower_col: str | None = "weighted_ci_lower",
    control_upper_col: str | None = "weighted_ci_upper",
) -> None:
    case_valid = case_df.dropna(subset=[value_col]).copy()
    control_valid = control_df.dropna(subset=[value_col]).copy()
    if case_valid.empty or control_valid.empty:
        return

    case_cols = ["time_bin", value_col]
    control_cols = ["time_bin", value_col]
    if (
        case_lower_col
        and case_upper_col
        and case_lower_col in case_valid.columns
        and case_upper_col in case_valid.columns
    ):
        case_cols.extend([case_lower_col, case_upper_col])
    if (
        control_lower_col
        and control_upper_col
        and control_lower_col in control_valid.columns
        and control_upper_col in control_valid.columns
    ):
        control_cols.extend([control_lower_col, control_upper_col])

    merged = case_valid[case_cols].merge(
        control_valid[control_cols],
        on="time_bin",
        suffixes=("_case", "_control"),
        how="inner",
    )
    if merged.empty:
        return
    merged["bin_left"] = merged["time_bin"].map(
        dict(zip(case_df["time_bin"], case_df["bin_left"]))
    )
    merged = merged.sort_values("bin_left", ascending=False).reset_index(drop=True)
    x = np.arange(len(merged))
    fig, ax = plt.subplots(figsize=(9, 5))
    case_y = merged[f"{value_col}_case"].to_numpy()
    control_y = merged[f"{value_col}_control"].to_numpy()
    if (
        case_lower_col
        and case_upper_col
        and f"{case_lower_col}_case" in merged.columns
        and f"{case_upper_col}_case" in merged.columns
    ):
        case_lower = merged[f"{case_lower_col}_case"].to_numpy()
        case_upper = merged[f"{case_upper_col}_case"].to_numpy()
        case_yerr = np.vstack([case_y - case_lower, case_upper - case_y])
        ax.errorbar(
            x,
            case_y,
            yerr=case_yerr,
            marker="o",
            linewidth=2,
            capsize=4,
            label=case_label,
        )
    else:
        ax.plot(x, case_y, marker="o", linewidth=2, label=case_label)

    if (
        control_lower_col
        and control_upper_col
        and f"{control_lower_col}_control" in merged.columns
        and f"{control_upper_col}_control" in merged.columns
    ):
        control_lower = merged[f"{control_lower_col}_control"].to_numpy()
        control_upper = merged[f"{control_upper_col}_control"].to_numpy()
        control_yerr = np.vstack([control_y - control_lower, control_upper - control_y])
        ax.errorbar(
            x,
            control_y,
            yerr=control_yerr,
            marker="o",
            linewidth=2,
            capsize=4,
            label=control_label,
        )
    else:
        ax.plot(x, control_y, marker="o", linewidth=2, label=control_label)

    ax.set_xticks(x)
    ax.set_xticklabels(merged["time_bin"], rotation=30, ha="right")
    ax.set_xlabel("Aligned time bin (farther from anchor to closer to anchor)")
    ax.set_ylabel(score_col)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_horizon_trajectories(df: pd.DataFrame, out_path: Path) -> None:
    valid = df.dropna(subset=["weighted_mean_score"]).copy()
    if valid.empty:
        return
    score_cols = [c for c in HORIZON_SCORES if c in valid["score_col"].unique()]
    if not score_cols:
        return

    fig, axes = plt.subplots(len(score_cols), 1, figsize=(9, 10), sharex=True)
    if len(score_cols) == 1:
        axes = [axes]
    labels = None
    for ax, score_col in zip(axes, score_cols):
        chunk = valid[valid["score_col"] == score_col].copy()
        chunk = chunk.sort_values("bin_left", ascending=False).reset_index(drop=True)
        x = np.arange(len(chunk))
        y = chunk["weighted_mean_score"].to_numpy()
        lower = chunk["weighted_ci_lower"].to_numpy()
        upper = chunk["weighted_ci_upper"].to_numpy()
        yerr = np.vstack([y - lower, upper - y])
        ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=2, capsize=4)
        ax.set_ylabel(score_col)
        ax.margins(y=0.15)
        labels = chunk["time_bin"].tolist()
    axes[-1].set_xticks(np.arange(len(labels)))
    axes[-1].set_xticklabels(labels, rotation=30, ha="right")
    axes[-1].set_xlabel(
        "Time to diagnosis bin (farther from diagnosis to closer to diagnosis)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _binary_auc(case_scores: pd.Series, control_scores: pd.Series) -> float:
    case = pd.to_numeric(case_scores, errors="coerce").dropna()
    control = pd.to_numeric(control_scores, errors="coerce").dropna()
    if len(case) == 0 or len(control) == 0:
        return np.nan
    all_scores = pd.concat(
        [
            pd.DataFrame({"score": case, "label": 1}),
            pd.DataFrame({"score": control, "label": 0}),
        ],
        ignore_index=True,
    )
    ranks = all_scores["score"].rank(method="average")
    n_pos = float((all_scores["label"] == 1).sum())
    n_neg = float((all_scores["label"] == 0).sum())
    rank_sum_pos = float(ranks[all_scores["label"] == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def build_auc_cohort(
    exams: pd.DataFrame,
    *,
    horizon_years: int,
    score_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = exams.copy()
    work["score_for_auc"] = pd.to_numeric(work[score_col], errors="coerce")
    work["years_to_cancer"] = pd.to_numeric(work["years_to_cancer"], errors="coerce")
    work["years_to_last_followup"] = pd.to_numeric(
        work["years_to_last_followup"], errors="coerce"
    )
    work = work.dropna(
        subset=["score_for_auc", "years_to_cancer", "years_to_last_followup"]
    )

    is_case = (work["years_to_cancer"] >= 0) & (work["years_to_cancer"] < horizon_years)
    is_control = (work["years_to_cancer"] >= horizon_years) & (
        work["years_to_last_followup"] >= horizon_years
    )
    case_df = work.loc[is_case].copy()
    control_df = work.loc[is_control].copy()
    return case_df, control_df


def plot_score_distributions(
    case_df: pd.DataFrame,
    control_df: pd.DataFrame,
    out_path: Path,
    score_col: str,
) -> dict[str, float]:
    case_scores = pd.to_numeric(case_df["score"], errors="coerce").dropna()
    control_scores = pd.to_numeric(control_df["score"], errors="coerce").dropna()
    if case_scores.empty or control_scores.empty:
        return {
            "auc": np.nan,
            "case_mean": np.nan,
            "control_mean": np.nan,
            "case_median": np.nan,
            "control_median": np.nan,
        }

    auc = _binary_auc(case_scores, control_scores)
    x_min = float(min(case_scores.min(), control_scores.min()))
    x_max = float(max(case_scores.max(), control_scores.max()))
    # Scores are strictly positive and strongly right-skewed, so log-spaced bins
    # make the bulk near 1e-3 visible without letting the high-score tail dominate.
    bins = np.geomspace(x_min, x_max, 40)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ax = axes[0]
    ax.hist(
        control_scores,
        bins=bins,
        weights=np.ones(len(control_scores)) / len(control_scores),
        alpha=0.45,
        color="#7fb3ff",
        label="Controls",
    )
    ax.hist(
        case_scores,
        bins=bins,
        weights=np.ones(len(case_scores)) / len(case_scores),
        alpha=0.45,
        color="#ff9a76",
        label="Cases",
    )
    ax.set_xscale("log")
    ax.set_xlabel(score_col)
    ax.set_ylabel("Fraction of exams")
    ax.legend(frameon=False)

    ax = axes[1]
    case_sorted = np.sort(case_scores.to_numpy(dtype=float))
    control_sorted = np.sort(control_scores.to_numpy(dtype=float))
    ax.plot(
        control_sorted,
        np.arange(1, len(control_sorted) + 1) / len(control_sorted),
        color="#7fb3ff",
        linewidth=2,
        label="Controls",
    )
    ax.plot(
        case_sorted,
        np.arange(1, len(case_sorted) + 1) / len(case_sorted),
        color="#ff9a76",
        linewidth=2,
        label="Cases",
    )
    ax.set_xscale("log")
    ax.set_xlabel(score_col)
    ax.set_ylabel("Cumulative proportion")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {
        "auc": float(auc),
        "case_mean": float(case_scores.mean()),
        "control_mean": float(control_scores.mean()),
        "case_median": float(case_scores.median()),
        "control_median": float(control_scores.median()),
    }


def plot_auc_cohort_distributions(
    case_df: pd.DataFrame,
    control_df: pd.DataFrame,
    out_path: Path,
    score_col: str,
) -> dict[str, float]:
    case_scores = pd.to_numeric(case_df["score_for_auc"], errors="coerce").dropna()
    control_scores = pd.to_numeric(
        control_df["score_for_auc"], errors="coerce"
    ).dropna()
    if case_scores.empty or control_scores.empty:
        return {
            "auc": np.nan,
            "case_mean": np.nan,
            "control_mean": np.nan,
            "case_median": np.nan,
            "control_median": np.nan,
            "n_cases": 0,
            "n_controls": 0,
        }

    auc = _binary_auc(case_scores, control_scores)
    x_min = float(min(case_scores.min(), control_scores.min()))
    x_max = float(max(case_scores.max(), control_scores.max()))
    bins = np.geomspace(x_min, x_max, 40)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ax = axes[0]
    ax.hist(
        control_scores,
        bins=bins,
        weights=np.ones(len(control_scores)) / len(control_scores),
        alpha=0.45,
        color="#7fb3ff",
        label="Eligible controls",
    )
    ax.hist(
        case_scores,
        bins=bins,
        weights=np.ones(len(case_scores)) / len(case_scores),
        alpha=0.45,
        color="#ff9a76",
        label="Eligible cases",
    )
    ax.set_xscale("log")
    ax.set_xlabel(score_col)
    ax.set_ylabel("Fraction of exams")
    ax.legend(frameon=False)

    ax = axes[1]
    case_sorted = np.sort(case_scores.to_numpy(dtype=float))
    control_sorted = np.sort(control_scores.to_numpy(dtype=float))
    ax.plot(
        control_sorted,
        np.arange(1, len(control_sorted) + 1) / len(control_sorted),
        color="#7fb3ff",
        linewidth=2,
        label="Eligible controls",
    )
    ax.plot(
        case_sorted,
        np.arange(1, len(case_sorted) + 1) / len(case_sorted),
        color="#ff9a76",
        linewidth=2,
        label="Eligible cases",
    )
    ax.set_xscale("log")
    ax.set_xlabel(score_col)
    ax.set_ylabel("Cumulative proportion")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {
        "auc": float(auc),
        "case_mean": float(case_scores.mean()),
        "control_mean": float(control_scores.mean()),
        "case_median": float(case_scores.median()),
        "control_median": float(control_scores.median()),
        "n_cases": int(len(case_scores)),
        "n_controls": int(len(control_scores)),
    }


def plot_slope_histogram(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["slope_score_per_year_to_dx"], bins=30, edgecolor="black", alpha=0.8)
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Per-patient slope of score vs time-to-diagnosis")
    ax.set_ylabel("Patients")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_spaghetti(
    case_df: pd.DataFrame, out_path: Path, max_patients: int, score_col: str
) -> None:
    eligible = case_df.groupby("patient_id").filter(lambda g: len(g) >= 3).copy()
    if eligible.empty:
        return
    patient_order = (
        eligible.groupby("patient_id")["trajectory_span_years"]
        .first()
        .sort_values(ascending=False)
        .index[:max_patients]
    )
    sample = eligible[eligible["patient_id"].isin(patient_order)].copy()

    fig, ax = plt.subplots(figsize=(9, 6))
    for patient_id, grp in sample.groupby("patient_id"):
        grp = grp.sort_values("time_to_dx_years")
        ax.plot(
            grp["time_to_dx_years"],
            grp["score"],
            marker="o",
            linewidth=1,
            alpha=0.5,
        )
    ax.set_xlabel("Time to diagnosis (years)")
    ax.set_ylabel(score_col)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_slope_summary(
    slopes: pd.DataFrame,
    out_path: Path,
    slope_col: str = "slope_score_per_year_to_dx",
) -> None:
    if slopes.empty:
        out_path.write_text("No patients with at least 2 pre-diagnostic exams.\n")
        return
    frac_negative = float((slopes[slope_col] < 0).mean())
    lines = [
        f"n_patients={len(slopes)}",
        f"median_slope={slopes[slope_col].median():.6f}",
        f"iqr_slope_low={slopes[slope_col].quantile(0.25):.6f}",
        f"iqr_slope_high={slopes[slope_col].quantile(0.75):.6f}",
        f"fraction_negative_slopes={frac_negative:.6f}",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def write_fixed_effects_summary(
    result: dict[str, float | int],
    out_path: Path,
    min_days_to_dx: int = 0,
    beta_label: str = "beta_time_to_dx",
) -> None:
    lines = [
        f"min_days_to_dx={min_days_to_dx}",
        f"n_patients={result['n_patients']}",
        f"n_exams={result['n_exams']}",
        f"{beta_label}={result['beta_time']:.6f}"
        if pd.notna(result["beta_time"])
        else f"{beta_label}=nan",
        f"se_cluster={result['se_cluster']:.6f}"
        if pd.notna(result["se_cluster"])
        else "se_cluster=nan",
        f"ci_lower={result['ci_lower']:.6f}"
        if pd.notna(result["ci_lower"])
        else "ci_lower=nan",
        f"ci_upper={result['ci_upper']:.6f}"
        if pd.notna(result["ci_upper"])
        else "ci_upper=nan",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def _fmt(x: float | int | str) -> str:
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return "NA"
    xf = float(x)
    if np.isinf(xf):
        return "inf" if xf > 0 else "-inf"
    if np.isclose(xf, round(xf)):
        return str(int(round(xf)))
    if abs(xf) >= 1:
        return f"{xf:.4f}"
    return f"{xf:.6f}"


def _embed_image(path: Path, alt: str) -> str:
    if not path.exists():
        return f'<p class="muted">Missing image: {path.name}</p>'
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{alt}">'


def _parse_summary_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().strip().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def write_html_report(out_dir: Path) -> None:
    cohort = pd.read_csv(out_dir / "trajectory_cohort_summary.csv")
    case_bins = pd.read_csv(out_dir / "case_binned_trajectory.csv")
    control_bins = pd.read_csv(out_dir / "control_binned_trajectory.csv")
    control_age = pd.read_csv(out_dir / "control_binned_trajectory_age_matched.csv")
    horizons = pd.read_csv(out_dir / "trajectory_by_horizon.csv")
    sensitivity = pd.read_csv(out_dir / "trajectory_sensitivity_summary.csv")
    control_trend = pd.read_csv(out_dir / "control_trend_summary.csv")
    bootstrap = pd.read_csv(out_dir / "case_bootstrap_trend_summary.csv")
    bootstrap_bins = pd.read_csv(out_dir / "case_bootstrap_binned_trajectory.csv")
    control_exam_table = pd.read_csv(out_dir / "trajectory_control_exam_table.csv")
    case_exam_table = pd.read_csv(out_dir / "trajectory_case_exam_table.csv")
    auc_case = pd.read_csv(out_dir / "auc_cohort_case_1yr.csv")
    auc_control = pd.read_csv(out_dir / "auc_cohort_control_1yr.csv")

    cohort_map = dict(zip(cohort["metric"], cohort["value"]))
    case_fe_map = _parse_summary_kv(out_dir / "case_fixed_effects_trend_summary.txt")
    control_fe_map = _parse_summary_kv(
        out_dir / "control_fixed_effects_trend_summary.txt"
    )
    case_fe = [f"{k}={v}" for k, v in case_fe_map.items()]
    control_fe = [f"{k}={v}" for k, v in control_fe_map.items()]
    control_patients = int(control_exam_table["patient_id"].nunique())
    control_exams = int(len(control_exam_table))
    case_traj_img = _embed_image(out_dir / "case_trajectory_1yr.png", "Case trajectory")
    horizon_img = _embed_image(
        out_dir / "trajectory_by_horizon.png", "Trajectory by horizon"
    )
    case_vs_control_img = _embed_image(
        out_dir / "case_vs_control_trajectory_1yr.png", "Case versus control trajectory"
    )
    case_vs_age_matched_img = _embed_image(
        out_dir / "case_vs_age_matched_control_trajectory_1yr.png",
        "Case versus age matched control trajectory",
    )
    slope_hist_img = _embed_image(
        out_dir / "case_slope_histogram.png", "Case slope histogram"
    )
    spaghetti_img = _embed_image(
        out_dir / "case_spaghetti_sample.png", "Case spaghetti plot"
    )
    score_dist_img = _embed_image(
        out_dir / "case_control_score_distribution_1yr.png",
        "Case versus control score distribution",
    )
    auc_score_dist_img = _embed_image(
        out_dir / "auc_cohort_score_distribution_1yr.png",
        "AUC-cohort case versus control score distribution",
    )
    score_summary = {
        "auc": _binary_auc(case_exam_table["score"], control_exam_table["score"]),
        "case_mean": float(
            pd.to_numeric(case_exam_table["score"], errors="coerce").mean()
        ),
        "control_mean": float(
            pd.to_numeric(control_exam_table["score"], errors="coerce").mean()
        ),
        "case_median": float(
            pd.to_numeric(case_exam_table["score"], errors="coerce").median()
        ),
        "control_median": float(
            pd.to_numeric(control_exam_table["score"], errors="coerce").median()
        ),
    }
    auc_score_summary = {
        "auc": _binary_auc(auc_case["score_for_auc"], auc_control["score_for_auc"]),
        "case_mean": float(
            pd.to_numeric(auc_case["score_for_auc"], errors="coerce").mean()
        ),
        "control_mean": float(
            pd.to_numeric(auc_control["score_for_auc"], errors="coerce").mean()
        ),
        "case_median": float(
            pd.to_numeric(auc_case["score_for_auc"], errors="coerce").median()
        ),
        "control_median": float(
            pd.to_numeric(auc_control["score_for_auc"], errors="coerce").median()
        ),
        "n_cases": int(len(auc_case)),
        "n_controls": int(len(auc_control)),
    }

    def table_from_df(df: pd.DataFrame) -> str:
        header = "".join(f"<th>{c}</th>" for c in df.columns)
        rows = []
        for _, row in df.iterrows():
            rows.append(
                "<tr>" + "".join(f"<td>{_fmt(v)}</td>" for v in row.tolist()) + "</tr>"
            )
        return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Mirai Trajectory Analysis</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{
      margin: 0;
      font-family: Georgia, 'Times New Roman', serif;
      color: #e8edf2;
      background:
        radial-gradient(circle at top left, rgba(84, 126, 166, 0.20), transparent 28%),
        radial-gradient(circle at top right, rgba(181, 118, 74, 0.12), transparent 22%),
        linear-gradient(180deg, #0d1117 0%, #151b23 100%);
      line-height: 1.6;
    }}
    .page {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 38px 26px 56px;
    }}
    .hero {{
      border: 1px solid #2e3945;
      border-radius: 22px;
      background: rgba(16, 22, 29, 0.84);
      box-shadow: 0 24px 70px rgba(0, 0, 0, 0.28);
      padding: 28px 30px 24px;
      margin-bottom: 28px;
      backdrop-filter: blur(8px);
    }}
    .hero h1 {{
      font-family: Helvetica, Arial, sans-serif;
      font-size: 42px;
      line-height: 1.02;
      letter-spacing: -0.03em;
      margin: 0 0 14px;
      color: #f7fafc;
    }}
    .hero p {{
      margin: 0;
      max-width: 760px;
      font-size: 18px;
      color: #d1d9e0;
    }}
    .callout-grid {{
      display: flex;
      flex-direction: column;
      gap: 14px;
      margin-top: 22px;
    }}
    .callout-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 14px;
    }}
    .callout {{
      border: 1px solid #334150;
      border-radius: 16px;
      padding: 14px 16px;
      background: rgba(24, 31, 39, 0.92);
    }}
    .callout .label {{
      font-family: Helvetica, Arial, sans-serif;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.10em;
      color: #8fbbe2;
      margin-bottom: 6px;
    }}
    .callout .value {{
      font-family: Helvetica, Arial, sans-serif;
      font-size: 26px;
      color: #f4f7fa;
    }}
    .section {{
      margin: 20px 0;
      padding: 24px 26px;
      border-radius: 18px;
      border: 1px solid #2b3642;
      background: rgba(18, 24, 31, 0.88);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
    }}
    h2, h3 {{
      font-family: Helvetica, Arial, sans-serif;
      color: #f4f7fa;
      margin-top: 0;
    }}
    h2 {{
      font-size: 25px;
      margin-bottom: 10px;
    }}
    h3 {{
      font-size: 18px;
      margin-bottom: 8px;
    }}
    p.lead {{
      font-size: 17px;
      color: #d5dde5;
    }}
    .muted {{ color: #b6c0c8; }}
    a {{ color: #8fc8ff; }}
    ul, ol {{ padding-left: 22px; }}
    li {{ margin: 6px 0; }}
    img {{
      max-width: 100%;
      border: 1px solid #3a4653;
      border-radius: 14px;
      background: #ffffff;
      box-shadow: 0 10px 24px rgba(0, 0, 0, 0.22);
      margin: 12px 0 22px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 12px 0 24px;
      font-size: 14px;
      background: #171d24;
      border-radius: 12px;
      overflow: hidden;
    }}
    th, td {{
      border: 1px solid #33404d;
      padding: 7px 9px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #222b35;
      color: #f4f7fa;
    }}
    td {{ color: #e8edf2; }}
    code {{
      background: #222b35;
      color: #f4f7fa;
      padding: 1px 5px;
      border-radius: 5px;
    }}
    pre {{
      background: #151c23;
      border: 1px solid #33404d;
      border-radius: 12px;
      color: #dbe3ea;
      padding: 12px 14px;
      overflow-x: auto;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Mirai Trajectory Analysis</h1>
      <p>This report focuses on a longitudinal question that AUC does not answer: whether Mirai scores behave coherently within patients over time, not just whether cases rank above controls cross-sectionally.</p>
      <div class="callout-grid">
        <div class="callout-row">
          <div class="callout">
            <div class="label">Case Patients</div>
            <div class="value">{_fmt(cohort_map.get("n_case_patients"))}</div>
          </div>
          <div class="callout">
            <div class="label">Case Exams</div>
            <div class="value">{_fmt(cohort_map.get("n_prediagnostic_case_exams"))}</div>
          </div>
          <div class="callout">
            <div class="label">Case FE Trend</div>
            <div class="value">{_fmt(float(case_fe_map["beta_time_to_dx"])) if "beta_time_to_dx" in case_fe_map else "NA"}</div>
          </div>
        </div>
        <div class="callout-row">
          <div class="callout">
            <div class="label">Control Patients</div>
            <div class="value">{_fmt(control_patients)}</div>
          </div>
          <div class="callout">
            <div class="label">Control Exams</div>
            <div class="value">{_fmt(control_exams)}</div>
          </div>
          <div class="callout">
            <div class="label">Control FE Trend</div>
            <div class="value">{_fmt(float(control_fe_map["beta_time_to_last_exam"])) if "beta_time_to_last_exam" in control_fe_map else "NA"}</div>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Cohort Definition</h2>
      <p class="lead">Cases in this analysis were defined by a sequence of explicit filters, not just by the phenotype label.</p>
      <ul>
        <li>Start from the exam-level clinical Mirai table.</li>
        <li>Keep only rows with non-missing <code>patient_id</code>, <code>exam_id</code>, score, and study date.</li>
        <li>Collapse to one row per exam.</li>
        <li>Define candidate cases as rows with <code>CaseControl</code> containing <code>Case</code>.</li>
        <li>Keep only pre-diagnostic exams: <code>time_to_dx_years &gt;= 0</code>, using diagnosis date when available and otherwise falling back to <code>years_to_cancer</code>.</li>
        <li>Exclude nominal case exams with neither a diagnosis date nor finite <code>years_to_cancer &lt; 100</code>. This removes the unresolved phenotype rows that are labeled case but have no usable timing information.</li>
      </ul>
      <p>After these cuts, the primary case cohort contains {_fmt(cohort_map.get("n_case_patients"))} patients and {_fmt(cohort_map.get("n_prediagnostic_case_exams"))} pre-diagnostic exams. Median exams per case patient: {_fmt(cohort_map.get("median_exams_per_patient"))}.</p>
    </section>

    <section class="section">
      <h2>How The Plots Are Constructed</h2>
      <ul>
        <li>Start from the enriched exam-level Mirai table, with one row per exam after collapsing the raw prediction output.</li>
        <li>For cases, define time using diagnosis date when available: <code>time_to_dx_years = (dx_date - study_date) / 365.25</code>. If diagnosis date is missing but <code>years_to_cancer</code> is finite, use that as a fallback.</li>
        <li>For controls, use <code>time_to_last_exam_years</code>, so the control plots are aligned to the patient’s last observed exam.</li>
        <li>Bin exams into six intervals: <code>&gt;5y</code>, <code>3-5y</code>, <code>2-3y</code>, <code>1-2y</code>, <code>0.5-1y</code>, and <code>&lt;0.5y</code> from the anchor.</li>
        <li>Within each bin, compute each patient’s mean score first, then average those patient means. This makes the trajectory patient-weighted rather than exam-weighted, so heavily imaged patients do not dominate the plot.</li>
        <li>The x-axis is ordered from farther from the anchor to closer to the anchor, so moving to the right means moving toward diagnosis for cases or toward last observed exam for controls.</li>
      </ul>
      <h3>Error Bars</h3>
      <p>The error bars in the binned plots are simple 95% Wald intervals around the patient-weighted mean in each bin: mean ± 1.96 × standard error, where the standard error is computed from the distribution of patient-level bin means. They are not bootstrap intervals.</p>
      <p>The bootstrap results are reported separately in the tables and are used as a sensitivity check for the overall case trend.</p>
    </section>

    <section class="section">
      <h2>Why Trajectory Analysis Is Distinct From AUC</h2>
      <p class="lead">AUC measures ranking across people. Trajectory analysis measures temporal coherence within a person.</p>
      <ul>
        <li>A model can have good AUC because it separates higher-risk from lower-risk patients in the population, even if an individual patient’s score is flat or noisy over time.</li>
        <li>This analysis asks a different question: in future cases, do scores actually rise as diagnosis approaches?</li>
        <li>That matters if the score is going to be interpreted as something that should evolve meaningfully across repeated screening exams.</li>
      </ul>
    </section>

    <section class="section">
      <h2>What We Learned</h2>
      <ul>
        <li>Within future cases, Mirai scores rise toward diagnosis on average.</li>
        <li>The case-only trend survives exclusion of exams within 90 and 180 days of diagnosis.</li>
        <li>The patient bootstrap agrees with the fixed-effects estimate, so the case result is not being carried by a small number of patients.</li>
        <li>The same directional pattern appears across 1-year, 2-year, and 5-year risk horizons.</li>
      </ul>
      <pre>{chr(10).join(case_fe)}</pre>
      <h3>Case Trajectory Plot: Methods</h3>
      <p>The case trajectory plot uses the primary case cohort only. Exams are binned by time to diagnosis, ordered from farther from diagnosis on the left to closer to diagnosis on the right. Within each bin, each patient contributes their mean score in that bin, and the plotted point is the mean across patients. Error bars are 95% Wald intervals computed from the patient-level bin means: mean ± 1.96 × SE.</p>
      {case_traj_img}
    </section>

    <section class="section">
      <h2>Why We Should Be Careful Not To Overinterpret It</h2>
      <p class="lead">The main caution is that controls also show a strong rise toward their last observed exam, and age matching does not remove that drift.</p>
      <ul>
        <li>This means the case-side rise is real, but it is not yet evidence of a cancer-specific temporal signature.</li>
        <li>The current control anchor may still reflect background drift, surveillance intensity, aging, or end-of-follow-up effects.</li>
        <li>So the defensible claim is: Mirai shows a longitudinal rise in cases, but the present control analysis does not establish specificity to cancer.</li>
      </ul>
      <pre>{chr(10).join(control_fe)}</pre>
      {table_from_df(control_trend)}
      <h3>Case vs Control Plot: Methods</h3>
      <p>This plot compares two different alignments. Cases are aligned to diagnosis using <code>time_to_dx_years</code>. Controls are aligned to their last observed exam using <code>time_to_last_exam_years</code>. Both sides use patient-weighted bin means. The raw control plot includes error bars because the raw control summary table includes bin-level Wald intervals computed the same way as for cases.</p>
      {case_vs_control_img}
      <h3>Case vs Age-Matched Control Plot: Methods</h3>
      <p>This plot uses the same case trajectory, but the control side is reweighted within each aligned time bin so the control age distribution matches the case age distribution in that same bin. This is meant to reduce simple age-composition differences. The error bars are approximate 95% Wald intervals for the weighted age-standardized mean, computed by combining age-bin-specific standard errors under a fixed-weight approximation.</p>
      {case_vs_age_matched_img}
      <h3>Score Distribution Plot: Methods</h3>
      <p>This figure ignores the time alignment entirely and instead compares the raw score distributions of the case and control trajectory cohorts. The left panel is an overlaid density histogram and the right panel is an empirical cumulative distribution function. This is useful because it answers a different question from the aligned-bin plots: whether the overall score distribution for cases is shifted upward relative to controls, which is the same direction summarized by the binary AUC.</p>
      <p>For the current trajectory cohorts, the trajectory-cohort AUC is {_fmt(score_summary["auc"])}, the case mean score is {_fmt(score_summary["case_mean"])}, the control mean score is {_fmt(score_summary["control_mean"])}, the case median is {_fmt(score_summary["case_median"])}, and the control median is {_fmt(score_summary["control_median"])}.</p>
      {score_dist_img}
      <h3>Horizon-Matched AUC Cohort Plot: Methods</h3>
      <p>This figure uses the actual binary AUC cohort definition for <code>1_year_risk</code>, rather than the longitudinal trajectory cohort. Eligible cases are exams with <code>0 &lt;= years_to_cancer &lt; 1</code>. Eligible controls are exams with <code>years_to_cancer &gt;= 1</code> and <code>years_to_last_followup &gt;= 1</code>. This is the relevant distribution comparison if the question is how the reported 1-year AUC can be greater than 0.5 even when the trajectory-cohort distributions appear reversed.</p>
      <p>For the 1-year AUC cohort, there are {_fmt(auc_score_summary["n_cases"])} eligible case exams and {_fmt(auc_score_summary["n_controls"])} eligible control exams. The AUC-cohort AUC is {_fmt(auc_score_summary["auc"])}, the case mean score is {_fmt(auc_score_summary["case_mean"])}, the control mean score is {_fmt(auc_score_summary["control_mean"])}, the case median is {_fmt(auc_score_summary["case_median"])}, and the control median is {_fmt(auc_score_summary["control_median"])}.</p>
      <p>This distinction is the key interpretive point: the longitudinal trajectory cohort and the horizon-matched AUC cohort are different subsets answering different questions. The trajectory plots ask whether scores rise within repeatedly imaged patients as the anchor approaches. The AUC cohort asks whether, at a given prediction horizon, eligible case exams tend to score above eligible control exams. Those can disagree without any plotting bug.</p>
      {auc_score_dist_img}
    </section>

    <section class="section">
      <h2>Horizon Comparison</h2>
      <h3>Methods</h3>
      <p>For the horizon comparison, the same case cohort and the same time bins are reused, but the plot is repeated separately for <code>1_year_risk</code>, <code>2_year_risk</code>, and <code>5_year_risk</code>. The panels are stacked vertically with separate y-scales so the shorter-horizon trajectory is visible rather than visually flattened by the larger 5-year risk values. Error bars are the same bin-level 95% Wald intervals used in the main case plot.</p>
      {horizon_img}
    </section>

    <section class="section">
      <h2>Technical Summary</h2>
      <p class="muted">The figures above are the main results. Additional plots below show within-patient heterogeneity rather than new primary estimands.</p>
      <h3>Slope Histogram: Methods</h3>
      <p>For each patient with at least two distinct time points, we fit a simple line of score versus aligned time and record the slope. Negative slopes for cases mean scores rise as diagnosis approaches.</p>
      {slope_hist_img}
      <h3>Spaghetti Plot: Methods</h3>
      <p>The spaghetti plot shows a subset of patients with at least three exams, chosen from those with the longest spans. It is meant to visualize heterogeneity, not to provide a pooled estimate.</p>
      {spaghetti_img}
    </section>

    <section class="section">
      <h2>Bottom Line</h2>
      <p>Trajectory analysis adds information that AUC cannot provide. It tells us that Mirai scores rise longitudinally within future cases. That is useful and reassuring. But because a similar rise appears in controls under the current alignment, this analysis does not yet justify a strong claim that the temporal pattern is specific to cancer.</p>
      <h3>Why The Distribution Changes After Age Matching</h3>
      <p>Age matching changes the control curve because it changes how much weight each control age stratum receives within a time bin. In the raw control plot, the mean reflects the natural age mix of the controls who happen to appear in that bin. In the age-matched plot, the control age strata are reweighted to mimic the case age distribution in the corresponding case bin. If risk scores differ by age, the weighted mean will shift even though the underlying control exams are the same.</p>
      <p>So a difference between the raw and age-matched control curves means that age composition was part of the original control pattern. It does not by itself prove that age fully explains the control drift, but it tells us age structure matters.</p>
      <h3>Possible Explanations For The Control-Side Rise</h3>
      <ul>
        <li><strong>End-of-follow-up anchoring artifact:</strong> every control is forced to have a final exam at time zero by construction. That creates an artificial countdown axis even though there is no biologic event like diagnosis at the endpoint. If the kinds of exams that happen near the end of observed follow-up differ from earlier exams, the plot can show a rise toward the anchor even when nothing cancer-specific is happening.</li>
        <li><strong>Aging:</strong> risk may increase over time even in controls because patients are older at later exams.</li>
        <li><strong>Surveillance intensity:</strong> patients who continue imaging through a final observed exam may differ systematically from patients with sparse follow-up.</li>
        <li><strong>Clinical enrichment near the last exam:</strong> the final observed exam in a control may still be associated with symptoms, callbacks, short-interval follow-up, or other reasons that raise risk scores without eventual cancer in the available follow-up window.</li>
        <li><strong>Residual label error or incomplete follow-up:</strong> some apparent controls near their last exam may later develop cancer outside the observed follow-up window.</li>
        <li><strong>Model sensitivity to temporal covariates:</strong> device changes, calendar drift, acquisition differences, or correlated age-related imaging changes may all produce longitudinal score drift unrelated to cancer.</li>
      </ul>
    </section>

    <section class="section">
      <h2>Appendix</h2>
      <h3>Cohort Table</h3>
      {table_from_df(cohort)}
      <h3>Case Binned Means</h3>
      {table_from_df(case_bins)}
      <h3>Bootstrap Trend Summary</h3>
      {table_from_df(bootstrap)}
      <h3>Bootstrap Bin Summary</h3>
      {table_from_df(bootstrap_bins)}
      <h3>Sensitivity Windows</h3>
      {table_from_df(sensitivity)}
      <h3>Raw Control Binned Means</h3>
      {table_from_df(control_bins)}
      <h3>Age-Matched Control Binned Means</h3>
      {table_from_df(control_age)}
      <h3>Trajectory By Horizon</h3>
      {table_from_df(horizons)}
    </section>
  </div>
</body>
</html>
"""
    (out_dir / "trajectory_analysis_report.html").write_text(html)


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False)
    exams = build_exam_table(df, score_col=args.score_col)
    case_df = build_case_table(exams)
    control_df = build_control_table(exams)

    exams.to_csv(args.out_dir / "trajectory_exam_level_table.csv", index=False)
    case_df.to_csv(args.out_dir / "trajectory_case_exam_table.csv", index=False)
    control_df.to_csv(args.out_dir / "trajectory_control_exam_table.csv", index=False)

    cohort_summary = summarize_cohort(case_df)
    cohort_summary.to_csv(args.out_dir / "trajectory_cohort_summary.csv", index=False)

    binned = build_binned_trajectory(
        case_df, bins=DEFAULT_BINS, time_col="time_to_dx_years"
    )
    binned.to_csv(args.out_dir / "case_binned_trajectory.csv", index=False)
    control_binned = build_binned_trajectory(
        control_df, bins=DEFAULT_BINS, time_col="time_to_last_exam_years"
    )
    control_binned.to_csv(args.out_dir / "control_binned_trajectory.csv", index=False)
    horizon_binned = build_horizon_trajectories(
        case_df,
        score_cols=HORIZON_SCORES,
        bins=DEFAULT_BINS,
        time_col="time_to_dx_years",
    )
    horizon_binned.to_csv(args.out_dir / "trajectory_by_horizon.csv", index=False)
    case_age_weights = compute_age_weights(case_df, age_col="age_at_exam_years")
    case_age_std = build_age_standardized_binned_trajectory(
        case_df,
        bins=DEFAULT_BINS,
        time_col="time_to_dx_years",
        age_col="age_at_exam_years",
        target_age_weights=case_age_weights,
    )
    case_age_std.to_csv(
        args.out_dir / "case_binned_trajectory_age_standardized.csv", index=False
    )
    control_age_matched = build_age_standardized_binned_trajectory(
        control_df,
        bins=DEFAULT_BINS,
        time_col="time_to_last_exam_years",
        age_col="age_at_exam_years",
        target_age_weights_by_time=compute_age_weights_by_time_bin(
            case_df,
            bins=DEFAULT_BINS,
            time_col="time_to_dx_years",
            age_col="age_at_exam_years",
        ),
    )
    control_age_matched.to_csv(
        args.out_dir / "control_binned_trajectory_age_matched.csv", index=False
    )
    auc_case_1yr, auc_control_1yr = build_auc_cohort(
        exams, horizon_years=1, score_col=args.score_col
    )
    auc_case_1yr.to_csv(args.out_dir / "auc_cohort_case_1yr.csv", index=False)
    auc_control_1yr.to_csv(args.out_dir / "auc_cohort_control_1yr.csv", index=False)

    slopes = build_patient_slopes(case_df)
    slopes.to_csv(args.out_dir / "case_patient_slopes.csv", index=False)
    write_slope_summary(slopes, args.out_dir / "case_patient_slope_summary.txt")
    fixed_effects = fit_patient_fixed_effects(case_df)
    write_fixed_effects_summary(
        fixed_effects, args.out_dir / "case_fixed_effects_trend_summary.txt"
    )
    sensitivity = build_sensitivity_summary(case_df)
    sensitivity.to_csv(args.out_dir / "trajectory_sensitivity_summary.csv", index=False)
    control_slopes, control_trend_summary = build_control_trend_summary(control_df)
    control_slopes.to_csv(args.out_dir / "control_patient_slopes.csv", index=False)
    control_trend_summary.to_csv(
        args.out_dir / "control_trend_summary.csv", index=False
    )
    write_slope_summary(
        control_slopes,
        args.out_dir / "control_patient_slope_summary.txt",
        slope_col="slope_score_per_year_to_last_exam",
    )
    control_fe = fit_patient_fixed_effects(
        control_df, time_col="time_to_last_exam_years"
    )
    write_fixed_effects_summary(
        control_fe,
        args.out_dir / "control_fixed_effects_trend_summary.txt",
        beta_label="beta_time_to_last_exam",
    )
    bootstrap_bins, bootstrap_beta = bootstrap_case_trajectory(
        case_df, bins=DEFAULT_BINS
    )
    bootstrap_bins.to_csv(
        args.out_dir / "case_bootstrap_binned_trajectory.csv", index=False
    )
    bootstrap_beta.to_csv(
        args.out_dir / "case_bootstrap_trend_summary.csv", index=False
    )

    plot_binned_trajectory(
        binned, args.out_dir / "case_trajectory_1yr.png", args.score_col
    )
    plot_case_control_trajectory(
        binned,
        control_binned,
        args.out_dir / "case_vs_control_trajectory_1yr.png",
        args.score_col,
    )
    plot_horizon_trajectories(
        horizon_binned, args.out_dir / "trajectory_by_horizon.png"
    )
    plot_case_control_trajectory(
        case_age_std,
        control_age_matched,
        args.out_dir / "case_vs_age_matched_control_trajectory_1yr.png",
        args.score_col,
        control_label="Age-matched controls aligned to last exam",
        value_col="age_standardized_mean_score",
        case_lower_col="age_standardized_ci_lower",
        case_upper_col="age_standardized_ci_upper",
        control_lower_col="age_standardized_ci_lower",
        control_upper_col="age_standardized_ci_upper",
    )
    plot_score_distributions(
        case_df,
        control_df,
        args.out_dir / "case_control_score_distribution_1yr.png",
        args.score_col,
    )
    plot_auc_cohort_distributions(
        auc_case_1yr,
        auc_control_1yr,
        args.out_dir / "auc_cohort_score_distribution_1yr.png",
        args.score_col,
    )
    plot_slope_histogram(slopes, args.out_dir / "case_slope_histogram.png")
    plot_spaghetti(
        case_df,
        args.out_dir / "case_spaghetti_sample.png",
        max_patients=args.max_spaghetti_patients,
        score_col=args.score_col,
    )

    summary_payload = {
        "input": str(args.input),
        "score_col": args.score_col,
        "n_exam_rows": int(len(exams)),
        "n_case_exam_rows": int(len(case_df)),
        "n_case_patients": int(case_df["patient_id"].nunique()) if len(case_df) else 0,
        "n_control_exam_rows": int(len(control_df)),
        "n_control_patients": int(control_df["patient_id"].nunique())
        if len(control_df)
        else 0,
        "n_slope_patients": int(len(slopes)),
        "fixed_effects_beta_time_to_dx": fixed_effects["beta_time"],
        "control_fixed_effects_beta_time_to_last_exam": control_fe["beta_time"],
    }
    (args.out_dir / "trajectory_run_summary.json").write_text(
        json.dumps(summary_payload, indent=2)
    )
    write_html_report(args.out_dir)
    print(f"Wrote trajectory analysis outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
