from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def split_patient_exam_id(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = series.astype(str).str.split("\t", n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError(
            "patient_exam_id must be formatted as '<patient_id>\\t<exam_id>'"
        )
    return parts[0].astype(str).str.strip(), parts[1].astype(str).str.strip()


def old_exam_table(old_input: pd.DataFrame) -> pd.DataFrame:
    exams = (
        old_input.groupby(["patient_id", "exam_id"], as_index=False)
        .agg(
            years_to_cancer=("years_to_cancer", "first"),
            years_to_last_followup=("years_to_last_followup", "first"),
            split_group=("split_group", "first"),
            n_views=("view", "count"),
        )
        .copy()
    )
    exams["patient_id"] = exams["patient_id"].astype(str)
    exams["accession_number"] = exams["exam_id"].astype(str).str.split(".", n=1).str[1]
    exams["old_key"] = exams["patient_id"] + "." + exams["accession_number"]
    return exams


def old_view_table(old_input: pd.DataFrame) -> pd.DataFrame:
    df = old_input.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["accession_number"] = df["exam_id"].astype(str).str.split(".", n=1).str[1]
    df["old_key"] = df["patient_id"] + "." + df["accession_number"]
    df["legacy_sop_instance_uid"] = (
        df["file_path"].astype(str).map(lambda s: Path(s).stem.split("_")[2])
    )
    return df


def current_exam_table(
    exams: pd.DataFrame, manifest: pd.DataFrame, pred_clin: pd.DataFrame
) -> pd.DataFrame:
    exams = exams.copy()
    exams["patient_id"] = exams["patient_id"].astype(str).str.strip()
    exams["exam_id"] = exams["exam_id"].astype(str).str.strip()
    exams["accession_number"] = exams["accession_number"].astype(str).str.strip()
    exams["old_key"] = exams["patient_id"] + "." + exams["accession_number"]

    manifest = manifest.copy()
    manifest["patient_id"] = manifest["patient_id"].astype(str).str.strip()
    manifest["exam_id"] = manifest["exam_id"].astype(str).str.strip()
    man_exam = manifest.groupby(["patient_id", "exam_id"], as_index=False).agg(
        years_to_cancer_manifest=("years_to_cancer", "first"),
        years_to_last_followup_manifest=("years_to_last_followup", "first"),
        split_group_manifest=("split_group", "first"),
        n_manifest_views=("view", "count"),
    )
    current = exams.merge(man_exam, on=["patient_id", "exam_id"], how="left")

    pred_keys = (
        pred_clin[["patient_id", "exam_id", "accession_number"]]
        .drop_duplicates()
        .assign(
            patient_id=lambda x: x["patient_id"].astype(str).str.strip(),
            exam_id=lambda x: x["exam_id"].astype(str).str.strip(),
            accession_number=lambda x: x["accession_number"].astype(str).str.strip(),
        )
    )
    pred_keys["old_key"] = pred_keys["patient_id"] + "." + pred_keys["accession_number"]
    current = current.merge(
        pred_keys.assign(in_pred=True)[["patient_id", "exam_id", "old_key", "in_pred"]],
        on=["patient_id", "exam_id", "old_key"],
        how="left",
    )
    current["in_pred"] = current["in_pred"].fillna(False)
    return current


def current_view_table(
    views: pd.DataFrame, exams: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    views = views.copy()
    views["patient_id"] = views["patient_id"].astype(str).str.strip()
    views["exam_id"] = views["exam_id"].astype(str).str.strip()
    views["accession_number"] = views["accession_number"].astype(str).str.strip()
    views["old_key"] = views["patient_id"] + "." + views["accession_number"]

    zarr_map = manifest[
        ["patient_id", "exam_id", "laterality", "view", "file_path"]
    ].copy()
    zarr_map["patient_id"] = zarr_map["patient_id"].astype(str).str.strip()
    zarr_map["exam_id"] = zarr_map["exam_id"].astype(str).str.strip()
    zarr_map = zarr_map.rename(columns={"file_path": "current_file_path"})

    merged = views.merge(
        zarr_map, on=["patient_id", "exam_id", "laterality", "view"], how="left"
    )
    return merged


def classify_missing(
    old_exams: pd.DataFrame, old_output: pd.DataFrame, current_exams: pd.DataFrame
) -> pd.DataFrame:
    old_pred = old_output[["old_key"]].drop_duplicates().assign(in_old_output=True)
    old_all = old_exams.merge(old_pred, on="old_key", how="left")
    old_all = old_all[old_all["in_old_output"].fillna(False)].copy()

    cur_by_key = current_exams.copy()
    key_counts = cur_by_key.groupby("old_key", as_index=False).agg(
        n_current_exam_ids=("exam_id", "nunique"),
        n_manifest_exam_ids=(
            "years_to_cancer_manifest",
            lambda s: int(s.notna().sum()),
        ),
        n_pred_exam_ids=("in_pred", lambda s: int(s.fillna(False).sum())),
        has_any_manifest=("years_to_cancer_manifest", lambda s: bool(s.notna().any())),
        has_any_pred=("in_pred", lambda s: bool(s.fillna(False).any())),
    )
    out = old_all.merge(key_counts, on="old_key", how="left")
    out["n_current_exam_ids"] = out["n_current_exam_ids"].fillna(0).astype(int)
    out["n_manifest_exam_ids"] = out["n_manifest_exam_ids"].fillna(0).astype(int)
    out["n_pred_exam_ids"] = out["n_pred_exam_ids"].fillna(0).astype(int)
    out["has_any_manifest"] = out["has_any_manifest"].fillna(False)
    out["has_any_pred"] = out["has_any_pred"].fillna(False)

    def reason(row: pd.Series) -> str:
        if row["has_any_pred"]:
            return "present_in_current_predictions"
        if row["n_current_exam_ids"] == 0:
            return "no_current_exam_with_patient_accession"
        if row["n_current_exam_ids"] > 1:
            return "multiple_current_exam_ids_same_patient_accession"
        if row["has_any_manifest"]:
            return "in_manifest_but_filtered_before_prediction"
        return "in_exams_but_not_manifest"

    out["reason"] = out.apply(reason, axis=1)
    return out


def build_view_identity(
    old_views: pd.DataFrame, current_views: pd.DataFrame
) -> pd.DataFrame:
    shared_exam_keys = sorted(
        set(old_views["old_key"].unique()) & set(current_views["old_key"].unique())
    )
    old_shared = old_views[old_views["old_key"].isin(shared_exam_keys)].copy()
    cur_shared = current_views[current_views["old_key"].isin(shared_exam_keys)].copy()
    merged = old_shared.merge(
        cur_shared[
            [
                "old_key",
                "patient_id",
                "accession_number",
                "exam_id",
                "laterality",
                "view",
                "sop_instance_uid",
                "source_archive_relpath",
                "source_archive_member",
                "sha256",
                "current_file_path",
            ]
        ].rename(columns={"exam_id": "current_exam_id"}),
        on=["old_key", "patient_id", "accession_number", "laterality", "view"],
        how="left",
    )
    merged["sop_match"] = (
        merged["legacy_sop_instance_uid"] == merged["sop_instance_uid"]
    )
    return merged


def choose_one_current_view(view_identity: pd.DataFrame) -> pd.DataFrame:
    cols = ["old_key", "patient_id", "exam_id", "laterality", "view", "file_path"]
    legacy_meta = (
        view_identity[cols]
        .drop_duplicates(
            subset=["patient_id", "exam_id", "laterality", "view", "file_path"]
        )
        .copy()
    )

    one_to_one = view_identity.groupby(
        ["old_key", "patient_id", "exam_id", "laterality", "view"], as_index=False
    ).agg(
        n_current_matches=("current_file_path", lambda s: int(s.notna().sum())),
        exact_sop_match=("sop_match", lambda s: bool(s.fillna(False).any())),
    )
    chosen = (
        view_identity.sort_values(
            ["sop_match", "current_file_path"], ascending=[False, True]
        )
        .drop_duplicates(
            subset=["old_key", "patient_id", "exam_id", "laterality", "view"],
            keep="first",
        )
        .copy()
    )
    chosen = chosen.merge(
        one_to_one,
        on=["old_key", "patient_id", "exam_id", "laterality", "view"],
        how="left",
    )
    current_meta = chosen[
        [
            "patient_id",
            "exam_id",
            "laterality",
            "view",
            "current_file_path",
            "n_current_matches",
            "exact_sop_match",
        ]
    ].rename(columns={"current_file_path": "file_path"})
    return legacy_meta, current_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-input", required=True, type=Path)
    parser.add_argument("--old-output", required=True, type=Path)
    parser.add_argument("--current-exams", required=True, type=Path)
    parser.add_argument("--current-views", required=True, type=Path)
    parser.add_argument("--current-manifest", required=True, type=Path)
    parser.add_argument("--current-pred-clin", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    old_input = pd.read_csv(args.old_input, dtype=str)
    old_input["years_to_cancer"] = pd.to_numeric(
        old_input["years_to_cancer"], errors="coerce"
    )
    old_input["years_to_last_followup"] = pd.to_numeric(
        old_input["years_to_last_followup"], errors="coerce"
    )
    old_in_views = old_view_table(old_input)
    old_in_exams = old_exam_table(old_input)

    old_output = pd.read_csv(args.old_output)
    old_pid, old_eid = split_patient_exam_id(old_output["patient_exam_id"])
    old_output = old_output.assign(patient_id=old_pid, exam_id=old_eid)
    old_output["accession_number"] = (
        old_output["exam_id"].astype(str).str.split(".", n=1).str[1]
    )
    old_output["old_key"] = (
        old_output["patient_id"] + "." + old_output["accession_number"]
    )

    current_exams = pd.read_parquet(args.current_exams)
    current_views = pd.read_parquet(args.current_views)
    current_manifest = pd.read_csv(args.current_manifest)
    current_pred_clin = pd.read_csv(args.current_pred_clin, low_memory=False)

    cur_exam = current_exam_table(current_exams, current_manifest, current_pred_clin)
    cur_view = current_view_table(current_views, current_exams, current_manifest)

    missing = classify_missing(old_in_exams, old_output, cur_exam)
    missing_only = missing[missing["reason"] != "present_in_current_predictions"].copy()
    view_identity = build_view_identity(old_in_views, cur_view)

    old_in_views.to_csv(out_dir / "omoleye_yrcut_input_views.csv", index=False)
    old_in_exams.to_csv(out_dir / "omoleye_yrcut_input_exams.csv", index=False)
    old_output.to_csv(out_dir / "omoleye_yrcut_output_exams.csv", index=False)
    missing.to_csv(out_dir / "omoleye_yrcut_exam_membership.csv", index=False)
    missing_only.to_csv(out_dir / "omoleye_yrcut_missing_exams.csv", index=False)
    view_identity.to_csv(out_dir / "omoleye_yrcut_view_identity.csv", index=False)
    legacy_meta, current_meta = choose_one_current_view(view_identity)
    one_to_one = current_meta[
        [
            "patient_id",
            "exam_id",
            "laterality",
            "view",
            "n_current_matches",
            "exact_sop_match",
        ]
    ].drop_duplicates()
    legacy_meta.to_csv(out_dir / "tensor_audit_legacy_meta.csv", index=False)
    current_meta.to_csv(out_dir / "tensor_audit_current_meta.csv", index=False)

    reason_counts = (
        missing_only["reason"]
        .value_counts()
        .rename_axis("reason")
        .reset_index(name="n")
    )
    reason_counts.to_csv(
        out_dir / "omoleye_yrcut_missing_reason_counts.csv", index=False
    )

    summary = {
        "old_input_exams": int(old_in_exams.shape[0]),
        "old_output_exams": int(old_output["old_key"].nunique()),
        "current_exam_keys": int(cur_exam["old_key"].nunique()),
        "overlap_predicted_exam_keys": int(
            missing[missing["reason"] == "present_in_current_predictions"][
                "old_key"
            ].nunique()
        ),
        "missing_exam_keys": int(missing_only["old_key"].nunique()),
        "missing_reason_counts": reason_counts.to_dict(orient="records"),
        "view_identity": {
            "n_view_rows_compared": int(view_identity.shape[0]),
            "n_exact_sop_matches": int(view_identity["sop_match"].fillna(False).sum()),
            "exact_sop_match_fraction": float(
                view_identity["sop_match"].fillna(False).mean()
            ),
            "n_missing_current_view_match": int(
                view_identity["sop_instance_uid"].isna().sum()
            ),
            "n_unique_view_positions": int(
                view_identity[["old_key", "laterality", "view"]]
                .drop_duplicates()
                .shape[0]
            ),
            "n_unique_view_positions_with_any_exact_sop_match": int(
                one_to_one["exact_sop_match"].fillna(False).sum()
            ),
            "unique_view_position_exact_sop_match_fraction": float(
                one_to_one["exact_sop_match"].fillna(False).mean()
            ),
            "n_unique_view_positions_with_multiple_current_matches": int(
                (one_to_one["n_current_matches"] > 1).sum()
            ),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(out_dir / "summary.json")
    print(out_dir / "omoleye_yrcut_missing_reason_counts.csv")
    print(out_dir / "omoleye_yrcut_view_identity.csv")


if __name__ == "__main__":
    main()
