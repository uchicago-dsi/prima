from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def auc(case_scores: pd.Series, control_scores: pd.Series) -> float:
    case = pd.to_numeric(case_scores, errors="coerce").dropna()
    control = pd.to_numeric(control_scores, errors="coerce").dropna()
    all_scores = pd.concat(
        [
            pd.DataFrame({"s": case, "y": 1}),
            pd.DataFrame({"s": control, "y": 0}),
        ],
        ignore_index=True,
    )
    ranks = all_scores["s"].rank(method="average")
    n1 = (all_scores["y"] == 1).sum()
    n0 = (all_scores["y"] == 0).sum()
    rank_sum = ranks[all_scores["y"] == 1].sum()
    return float((rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_preds(path: Path) -> pd.DataFrame:
    pred = pd.read_csv(path, sep=None, engine="python")
    parts = pred["patient_exam_id"].astype(str).str.split("\t", n=1, expand=True)
    pred["patient_id"] = parts[0].astype(str)
    pred["exam_id"] = parts[1].astype(str)
    return pred


def summarize_condition(
    label: str, pred_path: Path, meta_exam: pd.DataFrame
) -> tuple[list[dict], pd.DataFrame]:
    pred = load_preds(pred_path)
    df = pred.merge(meta_exam, on=["patient_id", "exam_id"])

    rows = []
    for h in [1, 2, 3, 4, 5]:
        risk_col = f"{h}_year_risk"
        cases = df[(df["years_to_cancer"] >= 0) & (df["years_to_cancer"] < h)][risk_col]
        ctrls = df[(df["years_to_cancer"] >= h) & (df["years_to_last_followup"] >= h)][
            risk_col
        ]
        rows.append(
            {
                "condition": label,
                "horizon_years": h,
                "auc": auc(cases, ctrls),
                "n_case": len(cases),
                "n_control": len(ctrls),
                "case_mean": float(cases.mean()),
                "control_mean": float(ctrls.mean()),
            }
        )
    return rows, df


def add_pairwise_correlations(
    condition_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    merged = None
    for label, df in condition_frames.items():
        keep = ["patient_id", "exam_id"] + [f"{h}_year_risk" for h in [1, 2, 3, 4, 5]]
        tmp = df[keep].copy()
        tmp = tmp.rename(
            columns={
                f"{h}_year_risk": f"{label}__{h}_year_risk" for h in [1, 2, 3, 4, 5]
            }
        )
        merged = (
            tmp if merged is None else merged.merge(tmp, on=["patient_id", "exam_id"])
        )

    rows = []
    labels = list(condition_frames.keys())
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            row = {"condition_a": a, "condition_b": b}
            for h in [1, 2, 3, 4, 5]:
                ca = f"{a}__{h}_year_risk"
                cb = f"{b}__{h}_year_risk"
                row[f"corr_{h}y"] = float(merged[[ca, cb]].corr().iloc[0, 1])
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta",
        default="/gpfs/data/huo-lab/Image/ChiMEC/MG/out/mirai_shards/chunk_000.csv",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--condition",
        action="append",
        nargs=2,
        metavar=("LABEL", "CSV"),
        required=True,
        help="Repeatable pair of condition label and prediction csv path",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.meta, dtype={"patient_id": str, "exam_id": str})
    meta_exam = meta[
        ["patient_id", "exam_id", "years_to_cancer", "years_to_last_followup"]
    ].drop_duplicates()

    summary_rows = []
    condition_frames: dict[str, pd.DataFrame] = {}
    for label, csv_path in args.condition:
        rows, df = summarize_condition(label, Path(csv_path), meta_exam)
        summary_rows.extend(rows)
        condition_frames[label] = df

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "condition_auc_summary.csv", index=False)

    corr_df = add_pairwise_correlations(condition_frames)
    corr_df.to_csv(out_dir / "condition_risk_correlations.csv", index=False)

    print(out_dir / "condition_auc_summary.csv")
    print(out_dir / "condition_risk_correlations.csv")


if __name__ == "__main__":
    main()
