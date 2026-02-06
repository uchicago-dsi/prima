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

from export_utils import (
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
from filesystem_utils import update_metadata_with_disk_status_by_date

# ChiMEC dataset configuration
CHIMEC_PATIENTS_FILE = "/gpfs/data/phs/groups/Projects/Huo_projects/SPORE/annawoodard/List_ChiMEC_priority_2025July30.csv"
CHIMEC_KEY_FILE = "/gpfs/data/huo-lab/Image/ChiMEC/study-16352a.csv"
CHIMEC_METADATA_FILE = "data/imaging_metadata.csv"
CHIMEC_EXPORT_STATE_FILE = Path("data/export_state_chimec.csv")
CHIMEC_FINGERPRINT_CACHE = Path("data/destination_fingerprints_chimec.json")
CHIMEC_BASE_DOWNLOAD_DIR = "/gpfs/data/huo-lab/Image/ChiMEC/"


def load_chimec_data():
    """Load and merge ChiMEC dataset."""
    try:
        patients = pd.read_csv(CHIMEC_PATIENTS_FILE)
        print(f"Loaded {len(patients):,} rows from patient info file.")
        key = pd.read_csv(CHIMEC_KEY_FILE)
        print(f"Loaded {len(key):,} rows from study_id-MRN key file.")
        metadata = pd.read_csv(CHIMEC_METADATA_FILE)
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

        # initialize export state columns from scraper metadata where available
        if "is_exported" not in metadata.columns:
            metadata["is_exported"] = False

        status_series = metadata.get("Status")
        if status_series is not None:
            exported_status_mask = (
                status_series.fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"already exported", "exported"})
            )
            metadata.loc[exported_status_mask, "is_exported"] = True

        if "Exported On" in metadata.columns:
            exported_on_mask = metadata["Exported On"].notna()
            metadata.loc[exported_on_mask, "is_exported"] = True

        metadata["is_exported"] = metadata["is_exported"].fillna(False).astype(bool)
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
                "download_attempt_outcome",
                "export_requested_on",
            ]:
                state_col = f"{col}_state"
                if state_col in metadata.columns:
                    mask = metadata[state_col].notna()
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

    db["is_exported"] = db["is_exported"].fillna(False).astype(bool)

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

    db = load_chimec_data()
    db = update_metadata_with_disk_status_by_date(
        db, conservative=True, fingerprint_cache=CHIMEC_FINGERPRINT_CACHE
    )

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

        if proceed == "y":
            outcomes, already_exported_count, successfully_exported_count = (
                execute_downloads(
                    targets,
                    args.batch_size,
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
                    elif outcome == "Already Exported":
                        db.loc[index, "is_exported"] = True
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
    refresh_db = update_metadata_with_disk_status_by_date(
        refresh_db, conservative=True, fingerprint_cache=CHIMEC_FINGERPRINT_CACHE
    )

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

    args = parser.parse_args()

    # status-only mode doesn't need credentials (uses existing metadata file)
    if args.status_only:
        print("Running in --status-only mode (no exports will be performed)\n")
        db = load_chimec_data()
        db = update_metadata_with_disk_status_by_date(
            db, conservative=True, fingerprint_cache=CHIMEC_FINGERPRINT_CACHE
        )
        filter_by_genotyping = not args.no_genotyping_filter
        identify_download_targets(
            db,
            filter_by_genotyping=filter_by_genotyping,
            modality=args.modality.upper(),
            base_download_dir=CHIMEC_BASE_DOWNLOAD_DIR,
            dataset="chimec",
        )
        sys.exit(0)

    if not all([USERNAME, PASSWORD]):
        print(
            "ERROR: IBROKER_USERNAME and IBROKER_PASSWORD must be set.", file=sys.stderr
        )
        sys.exit(1)

    # import scrape_ibroker now that we know we need it (triggers credential check)
    import_scrape_ibroker()

    cycles_run = 0
    last_target_indices: list[int] | None = None
    try:
        while True:
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
