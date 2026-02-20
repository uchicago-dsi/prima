#!/usr/bin/env python
"""Export script for ChiMEC dataset."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

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
CHIMEC_KEY_FILE = "/gpfs/data/huo-lab/Image/ChiMEC/study-16352a.csv"
CHIMEC_METADATA_FILE = "data/imaging_metadata.csv"
CHIMEC_EXPORT_STATE_FILE = Path("data/export_state_chimec.csv")
CHIMEC_FINGERPRINT_DIR = Path("fingerprints/chimec")
CHIMEC_FINGERPRINT_CACHE = CHIMEC_FINGERPRINT_DIR / "disk_fingerprints.json"
CHIMEC_BASE_DOWNLOAD_DIR = "/gpfs/data/huo-lab/Image/ChiMEC/"
CHIMEC_IBROKER_STUDY_NUMBER = "16352A"
CHIMEC_MODALITY = "MG"  # ChiMEC fingerprint/reconciliation are MG-only


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


def load_chimec_data():
    """Load and merge ChiMEC dataset."""
    try:
        patients = pd.read_csv(CHIMEC_PATIENTS_FILE)
        print(f"Loaded {len(patients):,} rows from patient info file.")
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
    db["DatedxIndex"] = pd.to_datetime(
        db["DatedxIndex"], errors="coerce", dayfirst=True
    )

    in_patient_mask = db["_in_patient_file"].astype("boolean").fillna(False)
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


def print_cycle_status_snapshot(
    db: pd.DataFrame,
    modality: str,
    refresh_summary: dict[str, int | bool] | None,
) -> None:
    """Print a concise per-cycle status snapshot for export decisions."""
    modality_mask = db["base_modality"] == modality
    subset = db[modality_mask]
    total = len(subset)
    on_disk = subset["is_on_disk"].fillna(False).astype(bool).sum()
    exported_not_on_disk = (
        subset["is_exported"].fillna(False).astype(bool)
        & ~subset["is_on_disk"].fillna(False).astype(bool)
    ).sum()
    requested_not_on_disk = (
        subset["is_requested"].fillna(False).astype(bool)
        & ~subset["is_exported"].fillna(False).astype(bool)
        & ~subset["is_on_disk"].fillna(False).astype(bool)
    ).sum()
    remaining = (
        ~subset["is_exported"].fillna(False).astype(bool)
        & ~subset["is_requested"].fillna(False).astype(bool)
        & ~subset["is_on_disk"].fillna(False).astype(bool)
    ).sum()
    print("\n=== Cycle status snapshot ===")
    print(f"modality: {modality}")
    print(f"known exams in metadata: {total:,}")
    print(f"on disk: {on_disk:,}")
    print(f"exported not on disk: {exported_not_on_disk:,}")
    print(f"requested not on disk: {requested_not_on_disk:,}")
    print(f"eligible to request now: {remaining:,}")
    if refresh_summary is not None:
        if bool(refresh_summary["is_complete"]):
            print("metadata refresh: complete")
        else:
            print(
                "metadata refresh: in progress "
                f"({refresh_summary['completed_batches']}/{refresh_summary['total_batches']} batches)"
            )


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
    )
    print("\n=== disk-vs-ibroker reconciliation ===")
    print(f"modality: {CHIMEC_MODALITY}")
    print(f"disk exams scanned: {summary['disk_total']:,}")
    print(f"ibroker exams with accession: {summary['ib_total']:,}")
    print(f"exact patient+accession matches: {summary['exact_match']:,}")
    print(
        "accession-changed candidates (unambiguous date): "
        f"{summary['accession_changed_unambiguous']:,}"
    )
    print(
        "accession-changed candidates (ambiguous date): "
        f"{summary['accession_changed_ambiguous']:,}"
    )
    print(f"disk-only after reconciliation: {summary['disk_only']:,}")
    print(f"disk rows without date suffix: {summary['disk_no_date']:,}")
    print(f"disk rows with date suffix: {summary['disk_with_date']:,}")
    print(
        "disk rows where date was inferred from DICOM metadata: "
        f"{summary['disk_dates_inferred_from_dicom']:,}"
    )
    print(f"disk .tar.xz entries: {summary['disk_tar_xz_entries']:,}")
    print(f"ibroker (study_id, study_date) pairs: {summary['ib_patient_date_pairs']:,}")
    print(f"disk (study_id, study_date) pairs: {summary['disk_patient_date_pairs']:,}")
    print(
        f"shared (study_id, study_date) pairs: {summary['shared_patient_date_pairs']:,}"
    )
    print(
        "disk rows whose (study_id, study_date) exists in iBroker: "
        f"{summary['disk_rows_with_shared_patient_date']:,}"
    )
    print(
        "ibroker (study_id, study_date) pairs with multiple exams: "
        f"{summary['ib_multi_patient_date']:,}"
    )
    print(
        "disk (study_id, study_date) pairs with multiple exams: "
        f"{summary['disk_multi_patient_date']:,}"
    )
    print(
        "disk-only rows (with date) whose pair exists in any iBroker row: "
        f"{summary['disk_only_with_date_pair_in_any_ibroker']:,}"
    )
    print(
        "disk-only rows (with date) whose study_id is absent from iBroker: "
        f"{summary['disk_only_with_date_study_id_not_in_ibroker']:,}"
    )
    print(
        "disk-only rows (with date) whose study_id exists but date does not: "
        f"{summary['disk_only_with_date_study_id_in_ibroker_but_date_not']:,}"
    )
    print(
        "study IDs on disk but not in iBroker: "
        f"{summary['study_ids_on_disk_not_in_ibroker']:,}"
    )
    print(f"wrote reconciliation CSV: {output_path}")
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
        f"{datetime.now():%Y-%m-%d %H:%M:%S} (ChiMEC) ==="
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

    db = load_chimec_data()
    db = apply_disk_status_for_modality(db, args.modality)
    print_cycle_status_snapshot(db, args.modality.upper(), refresh_summary)
    run_reconciliation_if_enabled(args, db, cycle_number)

    filter_by_genotyping = not args.no_genotyping_filter
    targets = identify_download_targets(
        db,
        filter_by_genotyping=filter_by_genotyping,
        modality=args.modality.upper(),
        base_download_dir=CHIMEC_BASE_DOWNLOAD_DIR,
        dataset="chimec",
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
        print("\n--- Summary of Exams to Download ---")
        print(f"Total potential targets: {len(targets)}")
        print(f"Unique patients to process: {targets['study_id'].nunique()}")

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

    refresh_db = load_chimec_data()
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

    args = parser.parse_args()

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
        db = load_chimec_data()
        db = apply_disk_status_for_modality(db, args.modality)
        filter_by_genotyping = not args.no_genotyping_filter
        identify_download_targets(
            db,
            filter_by_genotyping=filter_by_genotyping,
            modality=args.modality.upper(),
            base_download_dir=CHIMEC_BASE_DOWNLOAD_DIR,
            dataset="chimec",
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
