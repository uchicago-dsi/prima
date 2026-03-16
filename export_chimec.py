#!/usr/bin/env python
"""Export script for ChiMEC dataset."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prima.export_utils import (
    MERGE_KEY_COLUMNS,
    PASSWORD,
    USERNAME,
    audit_remote_export_status,
    execute_downloads,
    get_base_modality,
    identify_download_targets,
    import_scrape_ibroker,
    parse_wait_interval,
    save_current_state,
)
from prima.filesystem_utils import (
    build_chimec_disk_fingerprints,
    check_disk_for_downloads,
    reconcile_disk_ibroker_accessions,
)
from prima.ibroker_refresh import add_ibroker_state_columns, refresh_metadata_snapshot

# ChiMEC dataset configuration
CHIMEC_PATIENTS_FILE = "/gpfs/data/phs/groups/Projects/Huo_projects/SPORE/annawoodard/List_ChiMEC_priority_2025July30.csv"
CHIMEC_PCR_PATIENTS_FILE = "/gpfs/data/phs/groups/Projects/Huo_projects/SPORE/annawoodard/Phenotype_ChiMEC_2026Jan25.csv"
CHIMEC_KEY_FILE = "/gpfs/data/huo-lab/Image/ChiMEC/study-16352a.csv"
CHIMEC_METADATA_FILE = "data/imaging_metadata.csv"
CHIMEC_EXPORT_STATE_FILE = Path("data/export_state_chimec.csv")
CHIMEC_FINGERPRINT_DIR = Path("fingerprints/chimec")
CHIMEC_FINGERPRINT_CACHE = CHIMEC_FINGERPRINT_DIR / "disk_fingerprints.json"
CHIMEC_BASE_DOWNLOAD_DIR = "/gpfs/data/huo-lab/Image/ChiMEC/"
CHIMEC_IBROKER_STUDY_NUMBER = "16352A"
CHIMEC_MODALITY = "MG"  # ChiMEC fingerprint/reconciliation are MG-only
COHORT_PRIORITY = "priority"
COHORT_PCR = "pcr"
COHORT_CONFIGS = {
    COHORT_PRIORITY: {
        "default_patients_file": CHIMEC_PATIENTS_FILE,
        "required_non_null_column": None,
    },
    COHORT_PCR: {
        "default_patients_file": CHIMEC_PCR_PATIENTS_FILE,
        "required_non_null_column": "pcr",
    },
}


REQUESTED_OUTCOME_VALUES = {
    "request submitted",
    "requested",
    "start cmove",
    "study retrieved",
    "audit: available",
}

REQUESTED_STATUS_VALUES = {
    "request submitted",
    "requested",
    "start cmove",
    "study retrieved",
}


def _load_patients_for_cohort(cohort: str, patients_file: str | None) -> pd.DataFrame:
    """Load patient cohort table for export selection."""
    if cohort not in COHORT_CONFIGS:
        raise ValueError(
            f"Unsupported cohort '{cohort}'. Expected one of {COHORT_CONFIGS}"
        )

    cohort_config = COHORT_CONFIGS[cohort]
    resolved_patients_file = Path(
        patients_file or cohort_config["default_patients_file"]
    )
    patients = pd.read_csv(resolved_patients_file, low_memory=False)
    print(
        f"Loaded {len(patients):,} rows from patient info file: {resolved_patients_file}"
    )

    required_non_null_column = cohort_config["required_non_null_column"]
    if required_non_null_column is not None:
        normalized_cols = {
            str(col).strip().lower().replace("_", "").replace(" ", ""): col
            for col in patients.columns
        }
        required_key = (
            required_non_null_column.strip().lower().replace("_", "").replace(" ", "")
        )
        resolved_required_col = normalized_cols.get(required_key)
        if resolved_required_col is None:
            raise KeyError(
                f"Required cohort column '{required_non_null_column}' is missing from "
                f"{resolved_patients_file}"
            )
        before_filter = len(patients)
        patients = patients[patients[resolved_required_col].notna()].copy()
        print(
            f"Filtered cohort to {len(patients):,}/{before_filter:,} rows with non-null "
            f"'{resolved_required_col}'."
        )

    if "MRN" not in patients.columns:
        raise KeyError(
            f"MRN column is required in patient file {resolved_patients_file}"
        )
    patients["MRN"] = pd.to_numeric(patients["MRN"], errors="coerce")
    patients = patients[patients["MRN"].notna()].copy()
    duplicate_mrn_mask = patients["MRN"].duplicated(keep=False)
    if duplicate_mrn_mask.any():
        duplicate_count = duplicate_mrn_mask.sum()
        raise ValueError(
            f"Patient cohort has {duplicate_count:,} duplicate MRN rows after filtering. "
            "Cohort must be one row per MRN."
        )

    return patients


def load_chimec_data(cohort: str, patients_file: str | None):
    """Load and merge ChiMEC dataset for a specific cohort."""
    try:
        patients = _load_patients_for_cohort(cohort, patients_file)
        key = pd.read_csv(CHIMEC_KEY_FILE)
        print(f"Loaded {len(key):,} rows from study_id-MRN key file.")
        metadata = pd.read_csv(CHIMEC_METADATA_FILE, low_memory=False)
        print(f"Loaded {len(metadata):,} exam records from raw metadata file.")

        # A small fix: The Accession number is often what we care about, not the old exam_id
        # Let's rename exam_id to Accession if Accession is missing.
        if "exam_id" in metadata.columns and "Accession" in metadata.columns:
            metadata["Accession"] = metadata["Accession"].fillna(metadata["exam_id"])

        metadata["base_modality"] = metadata["Modality"].apply(get_base_modality)
        print("  - Added 'base_modality' column.")

        metadata["Study DateTime"] = pd.to_datetime(
            metadata["Study DateTime"], errors="coerce"
        )

        if "Exported On" in metadata.columns:
            metadata["Exported On"] = pd.to_datetime(
                metadata["Exported On"], errors="coerce"
            )

        metadata = add_ibroker_state_columns(metadata)
        metadata["download_attempt_outcome"] = pd.NA
        metadata["download_attempt_outcome"] = metadata[
            "download_attempt_outcome"
        ].astype("string")
        metadata["export_requested_on"] = pd.NaT

        # load previous export state from separate file if it exists
        if CHIMEC_EXPORT_STATE_FILE.exists():
            export_state = pd.read_csv(CHIMEC_EXPORT_STATE_FILE)
            print(f"Loaded {len(export_state):,} rows from export state file.")
            export_state["Study DateTime"] = pd.to_datetime(
                export_state["Study DateTime"], errors="coerce"
            )
            export_state["export_requested_on"] = pd.to_datetime(
                export_state["export_requested_on"], errors="coerce"
            )
            export_state["is_exported"] = (
                export_state["is_exported"].fillna(False).astype(bool)
            )
            if "is_requested" in export_state.columns:
                export_state["is_requested"] = (
                    export_state["is_requested"].fillna(False).astype(bool)
                )
            export_state["download_attempt_outcome"] = export_state[
                "download_attempt_outcome"
            ].astype("string")
            # merge export state back into metadata on the key columns
            metadata = metadata.merge(
                export_state,
                on=MERGE_KEY_COLUMNS,
                how="left",
                suffixes=("", "_state"),
            )
            # prefer values from export state file
            for col in [
                "is_exported",
                "is_requested",
                "download_attempt_outcome",
                "export_requested_on",
            ]:
                state_col = f"{col}_state"
                if state_col in metadata.columns:
                    mask = metadata[state_col].notna()
                    if col in {"is_exported", "is_requested"}:
                        metadata[col] = metadata[col].astype("boolean")
                        metadata[state_col] = metadata[state_col].astype("boolean")
                        metadata.loc[mask, col] = metadata.loc[mask, state_col].astype(
                            "boolean"
                        )
                        metadata[col] = metadata[col].fillna(False).astype(bool)
                    else:
                        metadata.loc[mask, col] = metadata.loc[mask, state_col]
                    metadata.drop(columns=[state_col], inplace=True)
    except FileNotFoundError as e:
        print(f"ERROR: Input file not found - {e}", file=sys.stderr)
        sys.exit(1)

    print("\nStep 1.1: Merging metadata with key file...")
    db = pd.merge(metadata, key, left_on="study_id", right_on="AnonymousID", how="left")
    db["MRN"] = pd.to_numeric(db["MRN"], errors="coerce")

    print("\nStep 1.2: Merging result with patient info file...")
    patients["_in_patient_file"] = True
    db = pd.merge(
        db, patients.drop(columns=["status"], errors="ignore"), on="MRN", how="left"
    )

    print("\nStep 1.3: Cleaning dates and deriving Case/Control status...")
    db["Study DateTime"] = pd.to_datetime(db["Study DateTime"], errors="coerce")
    normalized_db_cols = {
        str(col).strip().lower().replace("_", "").replace(" ", ""): col
        for col in db.columns
    }
    if "datedxindex" not in normalized_db_cols:
        diagnosis_fallback_cols = ["datedx", "datediagnosis", "datedxnew"]
        resolved_diag_col = None
        for candidate in diagnosis_fallback_cols:
            if candidate in normalized_db_cols:
                resolved_diag_col = normalized_db_cols[candidate]
                break
        if resolved_diag_col is None:
            db["DatedxIndex"] = pd.NaT
            print("  - No diagnosis date column found in cohort table; using all NaT.")
        else:
            db["DatedxIndex"] = db[resolved_diag_col]
            print(
                f"  - Using diagnosis date column '{resolved_diag_col}' as DatedxIndex."
            )

    db["DatedxIndex"] = pd.to_datetime(
        db["DatedxIndex"], errors="coerce", dayfirst=True
    )

    in_patient_mask = db["_in_patient_file"].astype("boolean").fillna(False)
    db["in_selected_cohort"] = in_patient_mask.to_numpy(dtype=bool, na_value=False)
    case_control_source = None
    for candidate in ["casecontrol", "case_control_status"]:
        if candidate in normalized_db_cols:
            case_control_source = normalized_db_cols[candidate]
            break

    if case_control_source is not None:
        case_control_raw = (
            db[case_control_source].fillna("").astype(str).str.strip().str.lower()
        )
        is_case = case_control_raw.str.contains("case", na=False)
        is_control = case_control_raw.str.contains("control", na=False)
        db["case_control_status"] = np.select(
            [is_case, is_control], ["Case", "Control"], default="Unknown"
        )
        missing_status_mask = db["case_control_status"] == "Unknown"
        db.loc[
            missing_status_mask & db["DatedxIndex"].notna(), "case_control_status"
        ] = "Case"
        db.loc[
            missing_status_mask
            & db["DatedxIndex"].isna()
            & in_patient_mask.to_numpy(dtype=bool, na_value=False),
            "case_control_status",
        ] = "Control"
        print(
            f"  - Derived Case/Control status using '{case_control_source}' with diagnosis-date fallback:"
        )
    else:
        conditions = [
            db["DatedxIndex"].notna(),
            in_patient_mask.to_numpy(dtype=bool, na_value=False),
        ]
        choices = ["Case", "Control"]
        db["case_control_status"] = np.select(conditions, choices, default="Unknown")
        print("  - Derived Case/Control status based on DatedxIndex:")
    print(
        db["case_control_status"]
        .value_counts(dropna=False)
        .rename_axis("case_control_status")
        .to_string()
    )

    db.drop(columns=["_in_patient_file"], inplace=True)
    db.dropna(subset=["study_id", "Study DateTime", "StudyDescription"], inplace=True)

    # Prefer rows with accessions when deduplicating exam metadata
    if "Accession" in db.columns:
        db["_has_accession"] = db["Accession"].notna().astype(int)
    else:
        db["_has_accession"] = 0

    before_exam_dedup = len(db)
    db = db.sort_values(
        MERGE_KEY_COLUMNS + ["_has_accession"], ascending=[True, True, True, False]
    ).drop_duplicates(subset=MERGE_KEY_COLUMNS, keep="first")
    db.drop(columns=["_has_accession"], inplace=True)
    if len(db) != before_exam_dedup:
        print(
            "  - Deduplicated exam rows on "
            f"{MERGE_KEY_COLUMNS}: {before_exam_dedup:,} → {len(db):,}"
        )

    has_accession = db["Accession"].notna()
    db.loc[has_accession, "is_exported"] = True
    db.loc[has_accession, "is_requested"] = False

    db["is_exported"] = db["is_exported"].astype("boolean").fillna(False).astype(bool)
    if "is_requested" not in db.columns:
        db["is_requested"] = False
    db["is_requested"] = db["is_requested"].astype("boolean").fillna(False).astype(bool)

    outcome_requested = (
        db["download_attempt_outcome"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(REQUESTED_OUTCOME_VALUES)
    )
    status_requested = (
        db.get("Status", pd.Series("", index=db.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(REQUESTED_STATUS_VALUES)
    )
    requested_by_timestamp = db["export_requested_on"].notna()
    db["is_requested"] = (~db["is_exported"]) & (
        db["is_requested"]
        | outcome_requested
        | status_requested
        | requested_by_timestamp
    )

    modality_counts = (
        db.loc[db["is_exported"], "base_modality"].fillna("<missing>").value_counts()
    )
    print(
        "  - Studies already exported (have Accession) by modality:\n"
        + (modality_counts.to_string() if not modality_counts.empty else "<none>")
    )

    # Initialize columns that may not exist yet
    if "download_attempt_outcome" not in db.columns:
        db["download_attempt_outcome"] = pd.NA
    db["download_attempt_outcome"] = db["download_attempt_outcome"].astype("string")

    if "export_requested_on" not in db.columns:
        db["export_requested_on"] = pd.NaT
    db["export_requested_on"] = pd.to_datetime(
        db["export_requested_on"], errors="coerce"
    )

    db["is_exported"] = db["is_exported"].fillna(False).astype(bool)
    if "is_on_disk" not in db.columns:
        db["is_on_disk"] = False

    disk_counts = (
        db.loc[db["is_on_disk"], "base_modality"].fillna("<missing>").value_counts()
    )

    print(f"\nMaster database created with {len(db):,} total exam records.")
    if not disk_counts.empty:
        print(
            "  - Currently on disk (by base_modality incl. <missing>):\n"
            + disk_counts.to_string()
        )
    return db


def _normalize_args_for_cohort(args: argparse.Namespace) -> None:
    """Apply cohort-specific defaults that prevent accidental over-filtering."""
    if args.cohort != COHORT_PCR:
        return

    requested_modality = args.modality.upper()
    if requested_modality == "MG":
        args.modality = "MR"
        print("PCR cohort selected: overriding modality from MG to MR.")
    else:
        args.modality = requested_modality

    if not args.no_genotyping_filter:
        print("PCR cohort selected: disabling genotyping filter.")
    args.no_genotyping_filter = True


def _should_filter_by_genotyping(args: argparse.Namespace) -> bool:
    """Return whether genotyping-based filtering should be applied."""
    return (args.cohort != COHORT_PCR) and (not args.no_genotyping_filter)


def _print_modality_scope_note(args: argparse.Namespace) -> None:
    """Clarify that downstream target selection is modality-scoped."""
    print(
        f"Note: downstream candidate filtering/export is scoped to modality={args.modality.upper()} "
        f"for cohort={args.cohort}."
    )


def _save_pcr_ibroker_time_from_dx_histogram(
    cohort_scoped: pd.DataFrame, output_prefix: str
) -> None:
    """Save histogram and binned counts for MR exam timing relative to diagnosis."""
    has_dx_mask = cohort_scoped["DatedxIndex"].notna()
    has_exam_mask = cohort_scoped["Study DateTime"].notna()
    analyzable = cohort_scoped.loc[has_dx_mask & has_exam_mask].copy()
    if analyzable.empty:
        print(
            "  - Skipping histogram save: no exams with both Study DateTime and DatedxIndex."
        )
        return

    delta_days = (
        analyzable["Study DateTime"] - analyzable["DatedxIndex"]
    ).dt.total_seconds() / 86400.0
    output_base = Path(output_prefix)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    csv_path = output_base.with_suffix(".csv")

    bin_width_days = 60
    min_edge = int(np.floor(delta_days.min() / bin_width_days) * bin_width_days)
    max_edge = int(np.ceil(delta_days.max() / bin_width_days) * bin_width_days)
    if min_edge == max_edge:
        max_edge = min_edge + bin_width_days
    bins = np.arange(min_edge, max_edge + bin_width_days, bin_width_days)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(delta_days, bins=bins, edgecolor="black", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("days from diagnosis (negative = before dx, positive = after dx)")
    ax.set_ylabel("number of MR exams in iBroker")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    bin_index = pd.cut(delta_days, bins=bins, include_lowest=True)
    bin_counts = bin_index.value_counts(sort=False)
    hist_df = pd.DataFrame(
        {
            "bin_start_day": bins[:-1],
            "bin_end_day": bins[1:],
            "exam_count": bin_counts.to_numpy(),
        }
    )
    hist_df.to_csv(csv_path, index=False)

    print(
        f"  - Saved iBroker time-from-dx histogram for selected cohort MR exams: {png_path}"
    )
    print(f"  - Saved iBroker time-from-dx binned counts CSV: {csv_path}")


def _print_pcr_prefilter_summary(
    db: pd.DataFrame,
    modality: str,
    pre_dx_days: int,
    post_dx_days: int,
    histogram_prefix: str,
) -> None:
    """Print cohort-constrained PCR counts before generic target filtering."""
    modality_upper = modality.upper()
    modality_mask = db["base_modality"] == modality_upper
    in_cohort_mask = (
        db.get("in_selected_cohort", pd.Series(False, index=db.index))
        .fillna(False)
        .astype(bool)
    )
    scoped = db[modality_mask].copy()
    cohort_scoped = db[modality_mask & in_cohort_mask].copy()

    print("\n--- PCR cohort prefilter summary ---")
    print(
        f"  {modality_upper} exams in metadata:          {len(scoped):,} "
        f"({scoped['study_id'].nunique():,} patients)"
    )
    print(
        f"  {modality_upper} exams in selected cohort:   {len(cohort_scoped):,} "
        f"({cohort_scoped['study_id'].nunique():,} patients)"
    )
    if cohort_scoped.empty:
        return

    case_series = (
        cohort_scoped.get(
            "case_control_status", pd.Series("Unknown", index=cohort_scoped.index)
        )
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    case_mask = case_series == "case"
    has_dx_mask = cohort_scoped["DatedxIndex"].notna()
    has_exam_mask = cohort_scoped["Study DateTime"].notna()
    analyzable_mask = case_mask & has_dx_mask & has_exam_mask

    analyzable = cohort_scoped.loc[analyzable_mask].copy()
    delta_days = (
        analyzable["Study DateTime"] - analyzable["DatedxIndex"]
    ).dt.total_seconds() / 86400.0
    pre_treatment_mask = delta_days <= 0
    in_window_mask = (delta_days >= -pre_dx_days) & (delta_days <= post_dx_days)

    print("  iBroker-only attrition (ignoring on-disk/export/requested):")
    print(
        f"    case-labeled MR exams in selected cohort:  {case_mask.sum():,} "
        f"({cohort_scoped.loc[case_mask, 'study_id'].nunique():,} patients)"
    )
    print(
        f"    with dx + exam date available:            {analyzable_mask.sum():,} "
        f"({cohort_scoped.loc[analyzable_mask, 'study_id'].nunique():,} patients)"
    )
    print(
        f"    pre-treatment MR exams (<=0 days):        {pre_treatment_mask.sum():,} "
        f"({analyzable.loc[pre_treatment_mask, 'study_id'].nunique():,} patients)"
    )
    print(
        f"    in window [−{pre_dx_days}, +{post_dx_days}] days:      {in_window_mask.sum():,} "
        f"({analyzable.loc[in_window_mask, 'study_id'].nunique():,} patients)"
    )

    is_on_disk = cohort_scoped.get(
        "is_on_disk", pd.Series(False, index=cohort_scoped.index)
    )
    is_exported = cohort_scoped.get(
        "is_exported", pd.Series(False, index=cohort_scoped.index)
    )
    is_requested = cohort_scoped.get(
        "is_requested", pd.Series(False, index=cohort_scoped.index)
    )
    is_on_disk = is_on_disk.fillna(False).astype(bool)
    is_exported = is_exported.fillna(False).astype(bool)
    is_requested = is_requested.fillna(False).astype(bool)
    pending = (~is_on_disk) & (~is_exported) & (~is_requested)
    print(
        f"  selected cohort pending status filters:      {pending.sum():,} "
        f"({cohort_scoped.loc[pending, 'study_id'].nunique():,} patients)"
    )
    _save_pcr_ibroker_time_from_dx_histogram(cohort_scoped, histogram_prefix)


def _print_pcr_window_summary(
    targets: pd.DataFrame, pre_dx_days: int, post_dx_days: int
) -> None:
    """Print PCR candidate MRI counts relative to diagnosis date."""
    if targets.empty:
        print("\n--- PCR dx-window summary ---")
        print("No candidates available for PCR dx-window analysis.")
        return

    case_series = (
        targets.get("case_control_status", pd.Series("Unknown", index=targets.index))
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    case_counts = case_series.value_counts(dropna=False)
    case_mask = case_series == "case"
    in_cohort_mask = (
        targets.get("in_selected_cohort", pd.Series(False, index=targets.index))
        .fillna(False)
        .astype(bool)
    )
    has_dx_mask = targets["DatedxIndex"].notna()
    has_exam_mask = targets["Study DateTime"].notna()
    analyzable_mask = case_mask & has_dx_mask & has_exam_mask

    analyzable = targets.loc[analyzable_mask].copy()
    if analyzable.empty:
        print("\n--- PCR dx-window summary ---")
        print(
            "No analyzable case MR exams with both Study DateTime and DatedxIndex after base filtering."
        )
        return

    delta_days = (
        analyzable["Study DateTime"] - analyzable["DatedxIndex"]
    ).dt.total_seconds() / 86400.0
    pre_window_mask = (delta_days >= -pre_dx_days) & (delta_days <= 0)
    around_dx_mask = (delta_days >= -pre_dx_days) & (delta_days <= post_dx_days)
    early_post_mask = (delta_days > 0) & (delta_days <= post_dx_days)
    late_post_mask = delta_days > post_dx_days
    older_pre_mask = delta_days < -pre_dx_days
    in_window_mask = (delta_days >= -pre_dx_days) & (delta_days <= post_dx_days)

    non_case_mask = ~case_mask
    case_missing_dx_mask = case_mask & (~has_dx_mask)
    case_missing_exam_mask = case_mask & has_dx_mask & (~has_exam_mask)

    print("\n--- PCR dx-window summary (after base target filters) ---")
    print(
        f"  candidates entering dx-window step: {len(targets):,} exams "
        f"({targets['study_id'].nunique():,} patients)"
    )
    print(
        f"  in selected PCR cohort file:         {in_cohort_mask.sum():,} exams "
        f"({targets.loc[in_cohort_mask, 'study_id'].nunique():,} patients)"
    )
    print("  case_control_status composition:")
    print(
        f"    case: {int(case_counts.get('case', 0)):,}, "
        f"control: {int(case_counts.get('control', 0)):,}, "
        f"unknown/other: {len(targets) - int(case_counts.get('case', 0)) - int(case_counts.get('control', 0)):,}"
    )
    print("  attrition from candidate MR exams:")
    print(
        f"    excluded as non-case:                  {non_case_mask.sum():,} exams "
        f"({targets.loc[non_case_mask, 'study_id'].nunique():,} patients)"
    )
    print(
        f"    excluded as outside selected cohort:   {(~in_cohort_mask).sum():,} exams "
        f"({targets.loc[~in_cohort_mask, 'study_id'].nunique():,} patients)"
    )
    print(
        f"    excluded case exams missing dx date:   {case_missing_dx_mask.sum():,} exams "
        f"({targets.loc[case_missing_dx_mask, 'study_id'].nunique():,} patients)"
    )
    print(
        f"    excluded case exams missing exam date: {case_missing_exam_mask.sum():,} exams "
        f"({targets.loc[case_missing_exam_mask, 'study_id'].nunique():,} patients)"
    )
    print(
        f"  case exams with dx + exam date:      {len(analyzable):,} exams "
        f"({analyzable['study_id'].nunique():,} patients)"
    )
    print(
        f"  pre-treatment window [-{pre_dx_days}, 0] days: {pre_window_mask.sum():,} exams "
        f"({analyzable.loc[pre_window_mask, 'study_id'].nunique():,} patients)"
    )
    print(
        f"  around-dx window [-{pre_dx_days}, +{post_dx_days}] days: {around_dx_mask.sum():,} exams "
        f"({analyzable.loc[around_dx_mask, 'study_id'].nunique():,} patients)"
    )
    print(
        f"  early post-dx (0, +{post_dx_days}] days: {early_post_mask.sum():,} exams "
        f"({analyzable.loc[early_post_mask, 'study_id'].nunique():,} patients)"
    )
    print(
        f"  older pre-dx (<-{pre_dx_days} days):   {older_pre_mask.sum():,} exams "
        f"({analyzable.loc[older_pre_mask, 'study_id'].nunique():,} patients)"
    )
    print(
        f"  late post-dx (>{post_dx_days} days):   {late_post_mask.sum():,} exams "
        f"({analyzable.loc[late_post_mask, 'study_id'].nunique():,} patients)"
    )
    print(
        f"  final in-window cases kept for export: {in_window_mask.sum():,} exams "
        f"({analyzable.loc[in_window_mask, 'study_id'].nunique():,} patients)"
    )

    hist_bins = [
        -np.inf,
        -365,
        -180,
        -120,
        -90,
        -60,
        -30,
        -14,
        -7,
        0,
        7,
        14,
        30,
        60,
        90,
        120,
        180,
        365,
        np.inf,
    ]
    hist_labels = [
        "<=-365",
        "(-365,-180]",
        "(-180,-120]",
        "(-120,-90]",
        "(-90,-60]",
        "(-60,-30]",
        "(-30,-14]",
        "(-14,-7]",
        "(-7,0]",
        "(0,7]",
        "(7,14]",
        "(14,30]",
        "(30,60]",
        "(60,90]",
        "(90,120]",
        "(120,180]",
        "(180,365]",
        ">365",
    ]
    binned = pd.cut(delta_days, bins=hist_bins, labels=hist_labels, include_lowest=True)
    hist_counts = binned.value_counts(sort=False)
    hist_patients = (
        analyzable.assign(_bin=binned)
        .groupby("_bin", observed=True)["study_id"]
        .nunique()
        .reindex(hist_labels, fill_value=0)
    )
    max_count = int(hist_counts.max()) if len(hist_counts) > 0 else 0
    scale = max(1, max_count // 40) if max_count > 0 else 1

    print("\n  histogram: case MRI exams by days from dx")
    print("    bin_days_from_dx      exams   patients   bar")
    for label in hist_labels:
        count = int(hist_counts.get(label, 0))
        patient_count = int(hist_patients.get(label, 0))
        bar = "#" * (count // scale) if count > 0 else ""
        print(f"    {label:>16}  {count:>8,}  {patient_count:>9,}   {bar}")

    pre_dx_only = delta_days <= 0
    pre_dx_df = analyzable.loc[pre_dx_only].copy()
    pre_dx_delta = delta_days[pre_dx_only]
    cumulative_thresholds = [14, 30, 60, 90, 120, 180, 270, 365, 540, 730]
    print("\n  cumulative pre-dx yields (windows [−N, 0] days)")
    print("    pre_dx_days<=N        exams   patients")
    for n_days in cumulative_thresholds:
        within = pre_dx_delta >= -n_days
        exams_n = int(within.sum())
        patients_n = int(pre_dx_df.loc[within, "study_id"].nunique())
        print(f"    {n_days:>12}  {exams_n:>8,}  {patients_n:>9,}")


def _filter_pcr_targets_by_dx_window(
    targets: pd.DataFrame, pre_dx_days: int, post_dx_days: int
) -> pd.DataFrame:
    """Restrict PCR export targets to case MR exams near diagnosis."""
    if targets.empty:
        return targets

    case_series = (
        targets.get("case_control_status", pd.Series("Unknown", index=targets.index))
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    case_mask = case_series == "case"
    in_cohort_mask = (
        targets.get("in_selected_cohort", pd.Series(False, index=targets.index))
        .fillna(False)
        .astype(bool)
    )
    has_dx_mask = targets["DatedxIndex"].notna()
    has_exam_mask = targets["Study DateTime"].notna()
    delta_days = (
        targets["Study DateTime"] - targets["DatedxIndex"]
    ).dt.total_seconds() / 86400.0
    window_mask = (delta_days >= -pre_dx_days) & (delta_days <= post_dx_days)
    keep_mask = in_cohort_mask & case_mask & has_dx_mask & has_exam_mask & window_mask
    filtered = targets.loc[keep_mask].copy()
    print(
        "PCR export filter applied: "
        f"{len(targets):,} -> {len(filtered):,} exams "
        f"({targets['study_id'].nunique():,} -> {filtered['study_id'].nunique():,} patients) "
        f"using selected cohort + case status + dx window [-{pre_dx_days}, +{post_dx_days}] days."
    )
    return filtered


def apply_disk_status_for_modality(db: pd.DataFrame, modality: str) -> pd.DataFrame:
    """Apply disk status check only for the requested modality subset."""
    modality_upper = modality.upper()
    modality_dir = Path(CHIMEC_BASE_DOWNLOAD_DIR) / modality_upper
    if modality_dir.exists() and modality_dir.is_dir():
        disk_check_dir = str(modality_dir)
    else:
        disk_check_dir = CHIMEC_BASE_DOWNLOAD_DIR

    modality_mask = db["base_modality"] == modality_upper
    subset = db[modality_mask].copy()
    if subset.empty:
        db["is_on_disk"] = False
        return db

    subset = check_disk_for_downloads(
        subset,
        basedir=disk_check_dir,
        fingerprint_cache=CHIMEC_FINGERPRINT_CACHE,
    )
    db["is_on_disk"] = False
    db.loc[subset.index, "is_on_disk"] = subset["is_on_disk"].fillna(False).astype(bool)
    return db


def get_modality_disk_dir(modality: str) -> str:
    """Resolve modality-specific disk directory, falling back to dataset root."""
    modality_upper = modality.upper()
    modality_dir = Path(CHIMEC_BASE_DOWNLOAD_DIR) / modality_upper
    if modality_dir.exists() and modality_dir.is_dir():
        return str(modality_dir)
    return CHIMEC_BASE_DOWNLOAD_DIR


def run_reconciliation_if_enabled(args, db: pd.DataFrame, cycle_number: int) -> None:
    """Run disk-vs-iBroker reconciliation and print ambiguity-focused summary."""
    if not args.reconcile_disk_ibroker:
        return
    output_path = Path(args.reconcile_output_csv)
    if args.max_cycles != 1 and not args.status_only:
        output_path = output_path.with_name(
            f"{output_path.stem}_cycle{cycle_number}{output_path.suffix}"
        )
    summary = reconcile_disk_ibroker_accessions(
        db,
        basedir=get_modality_disk_dir(CHIMEC_MODALITY),
        modality=CHIMEC_MODALITY,
        output_csv=output_path,
        fingerprint_cache=CHIMEC_FINGERPRINT_CACHE
        if CHIMEC_FINGERPRINT_CACHE.exists()
        else None,
        key_file=CHIMEC_KEY_FILE,
    )
    print("\n=== disk-vs-ibroker reconciliation (MG) ===\n")
    print(
        "Disk:  {:>8,} exams, {:>6,} study IDs".format(
            summary["disk_total"], summary["study_ids_on_disk"]
        )
    )
    if summary.get("unique_study_uids", 0) > 0:
        dup = summary.get("total_duplicate_exams", 0)
        uid_dup = summary.get("study_uids_with_duplicates", 0)
        print(
            "       {:>8,} unique StudyInstanceUIDs, {:>6,} duplicate exams "
            "({} study UIDs with >1 copy)".format(
                summary["unique_study_uids"], dup, uid_dup
            )
        )
        dupe_bytes = summary.get("duplicate_disk_usage_bytes", 0)
        if dupe_bytes > 0:
            dupe_gb = dupe_bytes / (1024**3)
            print("       disk usage of dupes: {:.2f} GB".format(dupe_gb))
    print("iBroker: {:>7,} exams with accession\n".format(summary["ib_total"]))
    print("Match breakdown:")
    print("  exact match: {:>8,}".format(summary["exact_match"]))
    print(
        "  accession-changed (unambiguous): {:>8,}".format(
            summary["accession_changed_unambiguous"]
        )
    )
    print(
        "  accession-changed (ambiguous):   {:>8,}".format(
            summary["accession_changed_ambiguous"]
        )
    )
    print("  disk-only:                       {:>8,}\n".format(summary["disk_only"]))
    if summary.get("key_study_ids_count", 0) > 0:
        in_key = summary.get("study_ids_on_disk_not_in_ibroker_in_key", 0)
        not_in_key = summary.get("study_ids_on_disk_not_in_ibroker_not_in_key", 0)
        print("Key file overlap:")
        print(
            "  study IDs in key:        {:>8,}".format(summary["key_study_ids_count"])
        )
        print(
            "  study IDs on disk:       {:>8,} (in key: {:>6,}, not in key: {:>4,})".format(
                summary["study_ids_on_disk"],
                summary["disk_study_ids_in_key"],
                summary["disk_study_ids_not_in_key"],
            )
        )
        print(
            "  study IDs on disk not in iBroker: {:>6,}, (in key: {:>6,}, not in key: {:>4,})".format(
                summary["study_ids_on_disk_not_in_ibroker"], in_key, not_in_key
            )
        )
        print(
            "  study IDs in key not on disk:    {:>8,}\n".format(
                summary["key_study_ids_not_on_disk"]
            )
        )
    print(f"Wrote: {output_path}")
    if output_path.exists():
        reconciled_df = pd.read_csv(output_path)
        drift_df = reconciled_df[
            reconciled_df["match_type"].isin(
                ["accession_changed_unambiguous", "accession_changed_ambiguous"]
            )
        ].copy()
        if not drift_df.empty:
            print(
                "sample accession-drift candidates (study_id, disk_date, disk_accession -> ib_accession):"
            )
            preview_cols = [
                "study_id",
                "disk_date",
                "disk_accession",
                "ib_accession_norm",
                "match_type",
            ]
            for _, row in drift_df[preview_cols].head(10).iterrows():
                print(
                    "  "
                    f"{row['study_id']}, {row['disk_date']}, "
                    f"{row['disk_accession']} -> {row.get('ib_accession_norm', '<ambiguous>')} "
                    f"({row['match_type']})"
                )


def run_export_cycle(args, cycle_number: int):
    """Run a single export pass and return summary stats.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    cycle_number : int
        Current cycle number
    """
    cycle_banner = (
        f"\n=== Export cycle {cycle_number} started at "
        f"{datetime.now():%Y-%m-%d %H:%M:%S} (ChiMEC, cohort={args.cohort}) ==="
    )
    print(cycle_banner)

    refresh_summary = getattr(args, "_last_refresh_summary", None)
    if args.refresh_metadata and not getattr(args, "_refresh_finished", False):
        _, refresh_summary = refresh_chimec_metadata(
            args.refresh_workers,
            args.refresh_mode,
            args.refresh_checkpoint_batch_size,
            args.refresh_max_new_batches_per_cycle,
        )
        args._last_refresh_summary = refresh_summary
        if (
            args.refresh_max_new_batches_per_cycle > 0
            and cycle_number == 1
            and args.refresh_mode == "fresh"
        ):
            args.refresh_mode = "resume"
            print("Refresh mode switched to resume for subsequent cycles.")
        if not refresh_summary["is_complete"]:
            print(
                "Metadata refresh is still in progress: "
                f"{refresh_summary['completed_batches']}/{refresh_summary['total_batches']} batches complete."
            )
        elif args.refresh_max_new_batches_per_cycle > 0:
            args._refresh_finished = True
            print(
                "Metadata refresh reached full coverage; skipping refresh in future cycles."
            )
    elif args.refresh_metadata and getattr(args, "_refresh_finished", False):
        print("Metadata refresh already complete in this run; skipping refresh step.")

    db = load_chimec_data(cohort=args.cohort, patients_file=args.patients_file)
    db = apply_disk_status_for_modality(db, args.modality)
    _print_modality_scope_note(args)
    run_reconciliation_if_enabled(args, db, cycle_number)

    db_for_targeting = db
    if args.cohort == COHORT_PCR:
        _print_pcr_prefilter_summary(
            db,
            args.modality,
            pre_dx_days=args.pcr_pre_dx_window_days,
            post_dx_days=args.pcr_post_dx_window_days,
            histogram_prefix=args.pcr_histogram_prefix,
        )
        in_cohort_mask = (
            db.get("in_selected_cohort", pd.Series(False, index=db.index))
            .fillna(False)
            .astype(bool)
        )
        db_for_targeting = db[in_cohort_mask].copy()
        print(
            f"PCR targeting dataset restricted to selected cohort: {len(db_for_targeting):,} exams "
            f"({db_for_targeting['study_id'].nunique():,} patients)."
        )

    filter_by_genotyping = _should_filter_by_genotyping(args)
    targets = identify_download_targets(
        db_for_targeting,
        filter_by_genotyping=filter_by_genotyping,
        modality=args.modality.upper(),
        base_download_dir=CHIMEC_BASE_DOWNLOAD_DIR,
        dataset="chimec",
    )
    if args.cohort == COHORT_PCR:
        _print_pcr_window_summary(
            targets,
            pre_dx_days=args.pcr_pre_dx_window_days,
            post_dx_days=args.pcr_post_dx_window_days,
        )
        targets = _filter_pcr_targets_by_dx_window(
            targets,
            pre_dx_days=args.pcr_pre_dx_window_days,
            post_dx_days=args.pcr_post_dx_window_days,
        )

    db["is_target"] = False
    db.loc[targets.index, "is_target"] = True

    # Only initialize download_attempt_outcome for NEW targets, preserve previous outcomes
    new_targets_mask = db["is_target"] & db["download_attempt_outcome"].isna()
    db.loc[new_targets_mask, "download_attempt_outcome"] = pd.NA
    db.loc[new_targets_mask, "export_requested_on"] = pd.NaT

    outcomes = {}
    already_exported_count = 0
    successfully_exported_count = 0

    if targets.empty:
        print("\n⚠ No new exams to download based on the current criteria.")
        print("   All exams may already be exported, on disk, or filtered out.")
        refresh_summary = getattr(args, "_last_refresh_summary", None)
        if refresh_summary is not None and not bool(refresh_summary["is_complete"]):
            print(
                "   Note: metadata refresh is partial "
                f"({refresh_summary['completed_batches']}/{refresh_summary['total_batches']} batches). "
                "Zero targets can occur before full coverage is reached."
            )
    else:
        print(
            f"\n--- Download targets: {len(targets):,} exams ({targets['study_id'].nunique():,} patients) ---"
        )

        proceed = "y" if args.auto_confirm else None
        if proceed == "y":
            print(
                f"Auto-confirm enabled; attempting to export up to {args.batch_size} exams."
            )
        else:
            proceed = (
                input(
                    f"\nProceed with attempting to export up to {args.batch_size} exams? (y/n): "
                )
                .lower()
                .strip()
            )

        effective_batch_size = args.batch_size
        if args.max_exports_per_hour > 0:
            effective_batch_size = min(args.batch_size, int(args._export_tokens))
            print(
                f"Rate limit budget currently allows {effective_batch_size} exports this cycle "
                f"(tokens={args._export_tokens:.1f}/{args.max_exports_per_hour:.1f})."
            )

        if proceed == "y" and effective_batch_size > 0:
            outcomes, already_exported_count, successfully_exported_count = (
                execute_downloads(
                    targets,
                    effective_batch_size,
                    full_db=db,
                    export_state_file=CHIMEC_EXPORT_STATE_FILE,
                    merge_key_columns=MERGE_KEY_COLUMNS,
                )
            )

            if outcomes:
                for index, outcome in outcomes.items():
                    db.loc[index, "download_attempt_outcome"] = outcome
                    if "Status" in db.columns:
                        db.loc[index, "Status"] = outcome
                    if outcome == "Request Submitted":
                        db.loc[index, "export_requested_on"] = pd.Timestamp.now()
                        db.loc[index, "is_requested"] = True
                    elif outcome == "Already Exported":
                        db.loc[index, "is_exported"] = True
                        db.loc[index, "is_requested"] = False
                        if "Exported On" in db.columns and pd.isna(
                            db.loc[index, "Exported On"]
                        ):
                            db.loc[index, "Exported On"] = pd.Timestamp.now()

            print("\n--- Final Cycle Summary ---")
            if successfully_exported_count > 0:
                print(
                    f"✓ Successfully submitted export requests: {successfully_exported_count} exam(s)"
                )
            else:
                print("✗ No export requests were submitted: 0 exam(s)")
            if already_exported_count > 0:
                print(
                    f"  Discovered to be already exported: {already_exported_count} exam(s)"
                )
            print(f"  Total exams processed: {len(outcomes)} exam(s)")
            if args.max_exports_per_hour > 0:
                args._export_tokens -= successfully_exported_count
        elif proceed == "y":
            print("✗ Cycle skipped - rate limit budget currently allows 0 exports.")
        else:
            print("✗ Cycle cancelled by user - no exports were requested.")

    # Save final state
    save_current_state(db, CHIMEC_EXPORT_STATE_FILE, MERGE_KEY_COLUMNS)
    print(f"\nCycle complete. Export state written to '{CHIMEC_EXPORT_STATE_FILE}'")

    return {
        "submitted": successfully_exported_count,
        "already_exported": already_exported_count,
        "processed": len(outcomes),
        "targets_considered": len(targets),
        "target_indices": targets.index.tolist(),
    }


def refresh_export_status(
    args,
    cycle_number: int,
    *,
    target_indices: list[int] | None = None,
):
    """Optionally reconcile export status during the wait window.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    cycle_number : int
        Current cycle number
    target_indices : list[int] | None
        Indices of target exams to prioritize
    """
    print(
        f"\n=== Refresh cycle {cycle_number}: auditing export status before next run (ChiMEC) ==="
    )

    refresh_db = load_chimec_data(cohort=args.cohort, patients_file=args.patients_file)
    refresh_db = apply_disk_status_for_modality(refresh_db, args.modality)

    modality = args.modality.upper()
    base_modality = refresh_db.get("base_modality")
    if base_modality is None:
        base_modality = pd.Series(pd.NA, index=refresh_db.index)

    is_on_disk_series = refresh_db.get("is_on_disk")
    if is_on_disk_series is None:
        is_on_disk_series = pd.Series(False, index=refresh_db.index)
    else:
        is_on_disk_series = is_on_disk_series.fillna(False)

    is_exported_series = refresh_db.get("is_exported")
    if is_exported_series is None:
        is_exported_series = pd.Series(False, index=refresh_db.index)
    else:
        is_exported_series = is_exported_series.fillna(False)

    candidates = refresh_db[
        (base_modality == modality) & (~is_on_disk_series) & (~is_exported_series)
    ].copy()

    if candidates.empty:
        print("No pending exams require status reconciliation.")
        return

    max_to_audit = args.refresh_limit if args.refresh_limit > 0 else None
    priority_indices = set(target_indices or [])
    if priority_indices:
        priority_mask = candidates.index.isin(priority_indices)
        candidates["__priority"] = priority_mask.astype(int)
    else:
        candidates["__priority"] = 0

    subset = candidates.sort_values(
        by=["__priority", "Study DateTime", "study_id"],
        ascending=[False, True, True],
    )
    subset = subset.drop(columns="__priority")
    candidates = candidates.drop(columns="__priority")

    if max_to_audit is not None:
        subset = subset.head(max_to_audit)

    priority_count = (
        subset.index.isin(priority_indices).sum() if priority_indices else 0
    )

    print(
        f"Auditing {len(subset)} exam(s) out of {len(candidates)} pending for modality {modality}."
    )
    if priority_indices:
        print(f"  - Priority exams (from current target list): {priority_count}")

    audit_stats = audit_remote_export_status(
        subset,
        full_db=refresh_db,
        max_exams=max_to_audit,
        export_state_file=CHIMEC_EXPORT_STATE_FILE,
        merge_key_columns=MERGE_KEY_COLUMNS,
    )

    print(
        "Audit summary: "
        f"checked {audit_stats['audited']} exams — "
        f"marked {audit_stats['marked_exported']} as exported, "
        f"{audit_stats['still_available']} still available."
    )

    save_current_state(refresh_db, CHIMEC_EXPORT_STATE_FILE, MERGE_KEY_COLUMNS)
    print(f"Audit state persisted to '{CHIMEC_EXPORT_STATE_FILE}'.")


def refresh_chimec_metadata(
    max_workers: int,
    refresh_mode: str,
    checkpoint_batch_size: int,
    max_new_batches: int,
) -> tuple[pd.DataFrame, dict[str, int | bool]]:
    """Refresh ChiMEC metadata with explicit fresh/resume mode."""
    study_ids = pd.read_csv(CHIMEC_KEY_FILE)["AnonymousID"].astype(str).tolist()
    print(
        f"\nRefreshing ChiMEC metadata from iBroker for {len(study_ids):,} study IDs "
        f"(mode={refresh_mode}, workers={max_workers}, batch_size={checkpoint_batch_size})..."
    )
    refresh_limit = max_new_batches if max_new_batches > 0 else None
    refreshed, summary = refresh_metadata_snapshot(
        study_ids,
        CHIMEC_METADATA_FILE,
        study_number=CHIMEC_IBROKER_STUDY_NUMBER,
        max_workers=max_workers,
        refresh_mode=refresh_mode,
        checkpoint_batch_size=checkpoint_batch_size,
        max_new_batches=refresh_limit,
    )
    print(
        f"Metadata refresh complete: wrote {len(refreshed):,} rows to "
        f"'{CHIMEC_METADATA_FILE}'."
    )
    if not summary["is_complete"]:
        print(
            "Refresh checkpoint saved for resume: "
            f"{summary['completed_batches']}/{summary['total_batches']} batches complete."
        )
    return refreshed, summary


def main():
    parser = argparse.ArgumentParser(
        description="Identify and download ChiMEC imaging exams from iBroker."
    )
    parser.add_argument(
        "--no-genotyping-filter",
        action="store_true",
        help="Include patients without genotyping data.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Max number of exams to request per cycle.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="MG",
        help="Base modality to filter for (e.g., MG, MR, CT).",
    )
    parser.add_argument(
        "--cohort",
        type=str,
        choices=[COHORT_PRIORITY, COHORT_PCR],
        default=COHORT_PRIORITY,
        help=(
            "Patient cohort used to select ChiMEC patients: "
            "'priority' uses the historical priority list, "
            "'pcr' uses phenotype rows with non-null pcr."
        ),
    )
    parser.add_argument(
        "--patients-file",
        type=str,
        default=None,
        help=(
            "Override cohort patient CSV path. Defaults are cohort-specific: "
            f"{COHORT_PRIORITY}={CHIMEC_PATIENTS_FILE}, "
            f"{COHORT_PCR}={CHIMEC_PCR_PATIENTS_FILE}."
        ),
    )
    parser.add_argument(
        "--loop-wait",
        type=parse_wait_interval,
        default="1h",
        help=(
            "Seconds to wait between cycles. Accepts plain seconds (e.g. 3600) or "
            "values with units like 60m or 1h. Set to 0 to run a single cycle."
        ),
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help=(
            "Number of cycles to run. Use 0 for unlimited cycles when --loop-wait > 0."
        ),
    )
    parser.add_argument(
        "--auto-confirm",
        action="store_true",
        default=True,
        help="Proceed without interactive confirmation prompts.",
    )
    parser.add_argument(
        "--refresh-export-status",
        action="store_true",
        help=(
            "During the wait window, audit pending exams against iBroker to mark "
            "newly exported exams."
        ),
    )
    parser.add_argument(
        "--refresh-limit",
        type=int,
        default=0,
        help="Max number of exams to audit during each refresh cycle (0 means all).",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Show export status summary and exit without exporting anything.",
    )
    parser.add_argument(
        "--refresh-metadata",
        action="store_true",
        help=(
            "Query iBroker for all ChiMEC study IDs and overwrite "
            "data/imaging_metadata.csv before proceeding."
        ),
    )
    parser.add_argument(
        "--refresh-workers",
        type=int,
        default=4,
        help="Number of parallel iBroker workers for --refresh-metadata.",
    )
    parser.add_argument(
        "--refresh-mode",
        type=str,
        choices=["fresh", "resume"],
        default="fresh",
        help=(
            "Mode for metadata refresh: fresh deletes refresh checkpoints and starts "
            "from scratch; resume continues an interrupted refresh."
        ),
    )
    parser.add_argument(
        "--refresh-checkpoint-batch-size",
        type=int,
        default=500,
        help=(
            "Study IDs per checkpoint batch for --refresh-metadata. "
            "Smaller values checkpoint more frequently."
        ),
    )
    parser.add_argument(
        "--refresh-max-new-batches-per-cycle",
        type=int,
        default=0,
        help=(
            "When >0, process at most this many new refresh batches per cycle. "
            "Use with --refresh-mode resume to interleave refresh with exports."
        ),
    )
    parser.add_argument(
        "--max-exports-per-hour",
        type=float,
        default=0,
        help=(
            "Rate limit for submitted export requests. "
            "Set to 0 to disable rate limiting."
        ),
    )
    parser.add_argument(
        "--build-fingerprints",
        action="store_true",
        help=(
            "Build light fingerprints (study_id, study_date, study_uid from DICOM) "
            "in fingerprints/chimec/ for disk-vs-iBroker cross-check. "
            "Dates are always from DICOM metadata, never from filename."
        ),
    )
    parser.add_argument(
        "--reconcile-disk-ibroker",
        action="store_true",
        help=(
            "Run disk-vs-iBroker accession reconciliation and report "
            "accession-drift/ambiguity buckets."
        ),
    )
    parser.add_argument(
        "--reconcile-output-csv",
        type=str,
        default="data/chimec_disk_ibroker_reconciliation.csv",
        help="Output CSV path for disk-vs-iBroker reconciliation details.",
    )
    parser.add_argument(
        "--pcr-pre-dx-window-days",
        type=int,
        default=90,
        help=(
            "Used when --cohort pcr: keep case MR exams this many days before diagnosis "
            "through the post-dx window."
        ),
    )
    parser.add_argument(
        "--pcr-post-dx-window-days",
        type=int,
        default=0,
        help=(
            "Used when --cohort pcr: include case MR exams up to this many days after diagnosis. "
            "Default 0 keeps pre-treatment only."
        ),
    )
    parser.add_argument(
        "--pcr-histogram-prefix",
        type=str,
        default="data/pcr_ibroker_time_from_dx",
        help=(
            "Used when --cohort pcr: output prefix for iBroker time-from-dx histogram artifacts "
            "(writes <prefix>.png and <prefix>.csv)."
        ),
    )

    args = parser.parse_args()
    _normalize_args_for_cohort(args)

    # build-fingerprints: scan disk, read DICOM metadata, write to fingerprints/chimec/
    if args.build_fingerprints:
        build_chimec_disk_fingerprints(
            basedir=CHIMEC_BASE_DOWNLOAD_DIR,
            output_dir=CHIMEC_FINGERPRINT_DIR,
            modality=CHIMEC_MODALITY,
        )
        if not args.status_only and not args.reconcile_disk_ibroker:
            print(
                "Fingerprints built. Run with --status-only --reconcile-disk-ibroker "
                "to cross-check with iBroker dates."
            )
            sys.exit(0)

    # status-only mode doesn't need credentials (uses existing metadata file)
    if args.status_only:
        print("Running in --status-only mode (no exports will be performed)\n")
        if args.refresh_metadata:
            if not all([USERNAME, PASSWORD]):
                print(
                    "ERROR: IBROKER_USERNAME and IBROKER_PASSWORD must be set for --refresh-metadata.",
                    file=sys.stderr,
                )
                sys.exit(1)
            refresh_chimec_metadata(
                args.refresh_workers,
                args.refresh_mode,
                args.refresh_checkpoint_batch_size,
                args.refresh_max_new_batches_per_cycle,
            )
        db = load_chimec_data(cohort=args.cohort, patients_file=args.patients_file)
        db = apply_disk_status_for_modality(db, args.modality)
        _print_modality_scope_note(args)
        db_for_targeting = db
        if args.cohort == COHORT_PCR:
            _print_pcr_prefilter_summary(
                db,
                args.modality,
                pre_dx_days=args.pcr_pre_dx_window_days,
                post_dx_days=args.pcr_post_dx_window_days,
                histogram_prefix=args.pcr_histogram_prefix,
            )
            in_cohort_mask = (
                db.get("in_selected_cohort", pd.Series(False, index=db.index))
                .fillna(False)
                .astype(bool)
            )
            db_for_targeting = db[in_cohort_mask].copy()
            print(
                f"PCR targeting dataset restricted to selected cohort: {len(db_for_targeting):,} exams "
                f"({db_for_targeting['study_id'].nunique():,} patients)."
            )
        filter_by_genotyping = _should_filter_by_genotyping(args)
        targets = identify_download_targets(
            db_for_targeting,
            filter_by_genotyping=filter_by_genotyping,
            modality=args.modality.upper(),
            base_download_dir=CHIMEC_BASE_DOWNLOAD_DIR,
            dataset="chimec",
        )
        if args.cohort == COHORT_PCR:
            _print_pcr_window_summary(
                targets,
                pre_dx_days=args.pcr_pre_dx_window_days,
                post_dx_days=args.pcr_post_dx_window_days,
            )
            _filter_pcr_targets_by_dx_window(
                targets,
                pre_dx_days=args.pcr_pre_dx_window_days,
                post_dx_days=args.pcr_post_dx_window_days,
            )
        run_reconciliation_if_enabled(args, db, cycle_number=1)
        sys.exit(0)

    if not all([USERNAME, PASSWORD]):
        print(
            "ERROR: IBROKER_USERNAME and IBROKER_PASSWORD must be set.", file=sys.stderr
        )
        sys.exit(1)

    # import iBroker browser/http helpers only when exports are needed
    import_scrape_ibroker()

    cycles_run = 0
    args._export_tokens = 0.0 if args.max_exports_per_hour > 0 else 0.0
    args._refresh_finished = False
    last_rate_update = time.monotonic()
    last_target_indices: list[int] | None = None
    try:
        while True:
            if args.max_exports_per_hour > 0:
                now = time.monotonic()
                elapsed_seconds = now - last_rate_update
                last_rate_update = now
                refill = args.max_exports_per_hour * (elapsed_seconds / 3600.0)
                args._export_tokens = min(
                    float(args.max_exports_per_hour), args._export_tokens + refill
                )
            cycles_run += 1
            cycle_result = run_export_cycle(args, cycles_run)
            last_target_indices = cycle_result.get("target_indices")

            max_cycles = args.max_cycles
            if max_cycles > 0 and cycles_run >= max_cycles:
                break

            wait_seconds = args.loop_wait
            if wait_seconds <= 0:
                break

            if args.refresh_export_status:
                try:
                    refresh_export_status(
                        args,
                        cycles_run,
                        target_indices=last_target_indices,
                    )
                except Exception as exc:
                    print(
                        f"\nWARNING: Refresh step failed with error: {exc}. Continuing to wait."
                    )

            print(
                f"\nWaiting {wait_seconds:.1f} seconds before starting the next cycle..."
            )
            try:
                time.sleep(wait_seconds)
            except KeyboardInterrupt:
                print("\nLoop interrupted during wait; exiting.")
                break
    except KeyboardInterrupt:
        print("\nLoop interrupted; exiting.")

    if cycles_run:
        print(f"\nRan {cycles_run} cycle{'s' if cycles_run != 1 else ''}.")


if __name__ == "__main__":
    main()
