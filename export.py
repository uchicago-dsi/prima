#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from filesystem_utils import check_disk_for_downloads

# Assuming scrape_ibroker.py is in the same directory or accessible
from scrape_ibroker import login, make_driver, wait_aspnet_idle

# --- Configuration ---
CHIMEC_PATIENTS_FILE = "/gpfs/data/phs/groups/Projects/Huo_projects/SPORE/annawoodard/List_ChiMEC_priority_2025July30.csv"
KEY_FILE = "/gpfs/data/huo-lab/Image/ChiMEC/study-16352a.csv"
METADATA_FILE = "data/imaging_metadata.csv"
BASE_DOWNLOAD_DIR = "/gpfs/data/huo-lab/Image/ChiMEC/"
METADATA_EXTRA_COLUMNS = [
    "is_exported",
    "download_attempt_outcome",
    "export_requested_on",
]


METADATA_BASE_COLUMNS: list[str] = []

MERGE_KEY_COLUMNS = ["study_id", "Study DateTime", "StudyDescription"]

USERNAME = os.getenv("IBROKER_USERNAME")
PASSWORD = os.getenv("IBROKER_PASSWORD")


def _parse_wait_interval(value: str) -> float:
    """Parse an interval string into seconds.

    Accepts plain seconds (e.g., ``3600``), or values with a unit suffix:
    ``Xs`` for seconds, ``Xm`` for minutes, ``Xh`` for hours.``X`` may be a
    float. Raises ``argparse.ArgumentTypeError`` for invalid inputs.
    """

    normalized = value.strip().lower()
    if not normalized:
        raise argparse.ArgumentTypeError("wait interval must be non-empty")

    unit_multipliers = {"s": 1, "m": 60, "h": 3600}

    suffix = normalized[-1]
    if suffix in unit_multipliers:
        number_portion = normalized[:-1]
        if not number_portion:
            raise argparse.ArgumentTypeError(
                "wait interval must include a numeric component"
            )
        try:
            magnitude = float(number_portion)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"could not parse wait interval '{value}'"
            ) from exc
        return magnitude * unit_multipliers[suffix]

    try:
        return float(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"could not parse wait interval '{value}'"
        ) from exc


def get_base_modality(modality):
    if pd.isna(modality):
        return None
    modality_str = str(modality).upper()
    if "MG" in modality_str:
        return "MG"
    if "MR" in modality_str:
        return "MR"
    if "CT" in modality_str:
        return "CT"
    if "US" in modality_str:
        return "US"
    if "CR" in modality_str:
        return "CR"
    if "DX" in modality_str:
        return "DX"
    if "NM" in modality_str:
        return "NM"
    if "PT" in modality_str:
        return "PT"
    return modality_str.split("/")[0]


def _atomic_write_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    """Write a CSV via a temp file and replace atomically."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.name}", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as tmp_file:
            df.to_csv(tmp_file, **kwargs)
        os.replace(tmp_path, output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def save_current_state(db: pd.DataFrame):
    """Persist export state back into the metadata file atomically."""

    if not METADATA_BASE_COLUMNS:
        raise RuntimeError(
            "Metadata columns are unknown; load_and_merge_data must run before saving."
        )

    working = db.copy()
    for column in METADATA_EXTRA_COLUMNS:
        if column not in working.columns:
            working[column] = pd.NA

    persist_columns: list[str] = []
    seen = set()
    for column in METADATA_BASE_COLUMNS + METADATA_EXTRA_COLUMNS:
        if column in working.columns and column not in seen:
            persist_columns.append(column)
            seen.add(column)

    metadata_subset = working[persist_columns].copy()

    key_cols = [col for col in MERGE_KEY_COLUMNS if col in metadata_subset.columns]
    if key_cols:
        before = len(metadata_subset)
        metadata_subset = (
            metadata_subset.sort_values(key_cols)
            .drop_duplicates(subset=key_cols, keep="last")
            .reset_index(drop=True)
        )
        if len(metadata_subset) != before:
            print(
                "  - save_current_state: collapsed "
                f"{before:,} rows to {len(metadata_subset):,}"
            )

    if "is_target" in metadata_subset.columns:
        metadata_subset = metadata_subset.drop(columns=["is_target"])

    _atomic_write_csv(
        metadata_subset,
        METADATA_FILE,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )


def load_and_merge_data():
    print("--- Phase 1: Loading and Merging Data ---")
    try:
        patients = pd.read_csv(CHIMEC_PATIENTS_FILE)
        print(f"Loaded {len(patients):,} rows from patient info file.")
        key = pd.read_csv(KEY_FILE)
        print(f"Loaded {len(key):,} rows from study_id-MRN key file.")
        metadata = pd.read_csv(METADATA_FILE)
        print(f"Loaded {len(metadata):,} exam records from raw metadata file.")

        global METADATA_BASE_COLUMNS
        METADATA_BASE_COLUMNS = list(metadata.columns)

        # A small fix: The Accession number is often what we care about, not the old exam_id
        # Let's rename exam_id to Accession if Accession is missing.
        if "exam_id" in metadata.columns and "Accession" in metadata.columns:
            metadata["Accession"] = metadata["Accession"].fillna(metadata["exam_id"])

        metadata["base_modality"] = metadata["Modality"].apply(get_base_modality)
        print("  - Added 'base_modality' column.")

        if "Exported On" in metadata.columns:
            metadata["Exported On"] = pd.to_datetime(
                metadata["Exported On"], errors="coerce"
            )

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

        if "download_attempt_outcome" not in metadata.columns:
            metadata["download_attempt_outcome"] = pd.NA
        metadata["download_attempt_outcome"] = metadata["download_attempt_outcome"].astype(
            "string"
        )

        if "export_requested_on" not in metadata.columns:
            metadata["export_requested_on"] = pd.NaT
        metadata["export_requested_on"] = pd.to_datetime(
            metadata["export_requested_on"], errors="coerce"
        )
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
        "  - Export flags (by base_modality incl. <missing>):\n"
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


def identify_download_targets(
    df: pd.DataFrame, filter_by_genotyping: bool, modality: str
):
    print("\n--- Phase 3: Identifying Target Exams for Download ---")
    df["rejection_reason"] = ""
    if "is_on_disk" in df.columns:
        df["is_on_disk"] = df["is_on_disk"].fillna(False).astype(bool)
    else:
        df["is_on_disk"] = False

    if "is_exported" in df.columns:
        df["is_exported"] = df["is_exported"].fillna(False).astype(bool)
    else:
        df["is_exported"] = False
    print(f"Initial pool: {len(df):,} exams")

    modality_mask = df["base_modality"] == modality
    df.loc[~modality_mask, "rejection_reason"] = f"Wrong modality (not {modality})"
    targets = df[modality_mask].copy()
    print(f"  - Kept {len(targets):,} exams after filtering for modality '{modality}'.")

    mask_on_disk = targets["is_on_disk"]
    targets.loc[mask_on_disk, "rejection_reason"] = "Already on disk"
    targets = targets[~mask_on_disk]
    print(f"  - Rejected {mask_on_disk.sum():,} because they are already on disk.")

    # We still check if it's exported, but now it's a secondary check.
    # An exam could be exported but the sync failed, so it's not on disk.
    # This includes exams with Accession numbers AND exams discovered as already exported in previous runs
    already_exported_mask = targets["is_exported"]
    exported_missing_disk = targets[already_exported_mask]
    if not exported_missing_disk.empty:
        print("  - Sample of exported-but-not-on-disk exams:")
        debug_columns = [
            "study_id",
            "Accession",
            "Study DateTime",
            "StudyDescription",
            "download_attempt_outcome",
            "Exported On",
        ]
        available_debug_columns = [
            column for column in debug_columns if column in exported_missing_disk.columns
        ]
        for _, row in exported_missing_disk.head(10).iterrows():
            details = []
            for column in available_debug_columns:
                value = row[column]
                if pd.isna(value):
                    value_repr = "<missing>"
                else:
                    value_repr = str(value)
                details.append(f"{column}={value_repr}")
            print("    " + ", ".join(details))
    targets.loc[already_exported_mask, "rejection_reason"] = (
        "Already exported (but not found on disk - possible sync issue)"
    )
    targets = targets[~already_exported_mask]
    print(
        f"  - Rejected {already_exported_mask.sum():,} exams already exported (from previous runs or iBroker's 'Exported' list)."
    )

    if filter_by_genotyping:
        mask_no_chip = targets["chip"].isna()
        targets.loc[mask_no_chip, "rejection_reason"] = "No genotyping data"
        targets = targets[~mask_no_chip]
        print(f"  - Rejected {mask_no_chip.sum():,} due to missing genotyping data.")

    case_mask = targets["case_control_status"] == "Case"
    bad_case_date_mask = case_mask & (
        targets["Study DateTime"] > targets["DatedxIndex"]
    )
    targets.loc[bad_case_date_mask, "rejection_reason"] = "Exam is after diagnosis date"
    targets = targets[~bad_case_date_mask]
    print(
        f"  - Rejected {bad_case_date_mask.sum():,} 'Case' exams that occurred after diagnosis."
    )

    print(f"  = {len(targets):,} final potential target exams identified.")
    return targets.sort_values(by=["study_id", "Study DateTime"])


def execute_downloads(
    targets_df: pd.DataFrame, batch_size: int, full_db: pd.DataFrame = None
):
    """Verifies exam availability live and requests exports."""
    if targets_df.empty:
        return {}, 0, 0

    print(
        f"\n--- Phase 4: Executing Downloads (target: {batch_size} successful exports) ---"
    )
    driver = None
    outcomes = {}
    already_exported_counter = 0
    successfully_exported_counter = 0

    # Process patient by patient from the sorted target list
    unique_patients = targets_df["study_id"].unique()

    try:
        driver = make_driver()
        login(driver, USERNAME, PASSWORD)

        for study_id in tqdm(unique_patients, desc="Processing Patients"):
            if successfully_exported_counter >= batch_size:
                print(f"\nReached target of {batch_size} successful exports. Stopping.")
                break

            patient_exams = targets_df[targets_df["study_id"] == study_id]
            print(
                f"\nProcessing patient {study_id} (has {len(patient_exams)} target exams)..."
            )

            try:
                driver.find_element(by="name", value="tbxAssignedID").clear()
                driver.find_element(by="name", value="tbxAssignedID").send_keys(
                    str(int(study_id))
                )
                driver.find_element(by="name", value="btnFetch").click()
                wait_aspnet_idle(driver)
            except Exception as e:
                print(f"ERROR navigating for study_id {study_id}. Skipping. Error: {e}")
                for index, _ in patient_exams.iterrows():
                    outcomes[index] = "Navigation Error"
                continue

            available_on_page = {}
            page_rows = driver.find_elements(
                by="xpath",
                value="//table[@id='TabContainer1_tabPanel1_gv1']//tr[position()>1]",
            )
            for row in page_rows:
                try:
                    cells = row.find_elements(by="tag name", value="td")
                    row_date = pd.to_datetime(cells[2].text).date()
                    row_desc = cells[3].text.strip()
                    checkbox = row.find_element(
                        by="xpath", value=".//input[@type='checkbox']"
                    )
                    available_on_page[(row_date, row_desc)] = checkbox
                except Exception:
                    pass

            requested_this_patient = False
            requested_indices = []
            for index, target_exam in patient_exams.iterrows():
                if successfully_exported_counter >= batch_size:
                    outcomes[index] = "Skipped - Batch full"
                    continue

                target_key = (
                    target_exam["Study DateTime"].date(),
                    target_exam["StudyDescription"],
                )

                if target_key in available_on_page:
                    print(
                        f"  - Found available exam from {target_key[0]}. Selecting checkbox."
                    )
                    available_on_page[target_key].click()
                    outcomes[index] = "Request Submitted"
                    successfully_exported_counter += 1
                    requested_this_patient = True
                    requested_indices.append(index)
                    if full_db is not None:
                        full_db.loc[index, "download_attempt_outcome"] = (
                            "Request Submitted"
                        )
                        if "Status" in full_db.columns:
                            full_db.loc[index, "Status"] = "Request Submitted"
                else:
                    print(
                        f"  - INFO: Exam from {target_key[0]} is no longer available (already exported)."
                    )
                    outcomes[index] = "Already Exported"
                    already_exported_counter += 1

                    # Mark as exported in the full database if provided
                    if full_db is not None:
                        full_db.loc[index, "is_exported"] = True
                        full_db.loc[index, "download_attempt_outcome"] = (
                            "Already Exported"
                        )
                        if "Status" in full_db.columns:
                            full_db.loc[index, "Status"] = "Already Exported"
                        if "Exported On" in full_db.columns:
                            if pd.isna(full_db.loc[index, "Exported On"]):
                                full_db.loc[index, "Exported On"] = pd.Timestamp.now()

            if requested_this_patient:
                print("  Submitting export request for selected exams...")
                driver.find_element(by="name", value="btnExport").click()
                wait_aspnet_idle(driver)
                print("  Export request submitted.")
                if full_db is not None and requested_indices:
                    submission_ts = pd.Timestamp.now()
                    for idx in requested_indices:
                        full_db.loc[idx, "export_requested_on"] = submission_ts

            # Save state periodically after each patient to preserve progress
            # This includes discoveries of already exported exams
            if full_db is not None:
                save_current_state(full_db)
                print("  - State saved to disk.")
    finally:
        if driver:
            print("Closing webdriver session.")
            driver.quit()

    return outcomes, already_exported_counter, successfully_exported_counter


def run_export_cycle(args, cycle_number: int):
    """Run a single export pass and return summary stats."""

    cycle_banner = (
        f"\n=== Export cycle {cycle_number} started at "
        f"{datetime.now():%Y-%m-%d %H:%M:%S} ==="
    )
    print(cycle_banner)

    db = load_and_merge_data()
    db = check_disk_for_downloads(db, BASE_DOWNLOAD_DIR)

    targets = identify_download_targets(
        db,
        filter_by_genotyping=not args.no_genotyping_filter,
        modality=args.modality.upper(),
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
        print("\nNo new exams to download based on the current criteria.")
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
                execute_downloads(targets, args.batch_size, full_db=db)
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

            print("\n--- Cycle Summary ---")
            print(
                f"Successfully submitted export requests for: {successfully_exported_count} exams."
            )
            print(
                f"Discovered to be already exported:        {already_exported_count} exams."
            )
            print(f"Total exams processed:                    {len(outcomes)} exams.")
        else:
            print("Cycle cancelled by user response.")

    # Save final state
    save_current_state(db)
    print(
        f"\nCycle complete. Metadata updates have been written to '{METADATA_FILE}'"
    )

    return {
        "submitted": successfully_exported_count,
        "already_exported": already_exported_count,
        "processed": len(outcomes),
        "targets_considered": len(targets),
    }


def audit_remote_export_status(
    audit_df: pd.DataFrame,
    full_db: pd.DataFrame | None = None,
    max_exams: int | None = None,
) -> dict[str, int]:
    """Verify availability of pending exams without requesting exports."""

    if audit_df.empty:
        return {"audited": 0, "marked_exported": 0, "still_available": 0}

    driver = None
    audited = 0
    marked_exported = 0
    still_available = 0

    unique_patients = audit_df["study_id"].unique()

    try:
        driver = make_driver()
        login(driver, USERNAME, PASSWORD)

        for study_id in tqdm(unique_patients, desc="Auditing Patients"):
            patient_exams = audit_df[audit_df["study_id"] == study_id]

            try:
                driver.find_element(by="name", value="tbxAssignedID").clear()
                driver.find_element(by="name", value="tbxAssignedID").send_keys(
                    str(int(study_id))
                )
                driver.find_element(by="name", value="btnFetch").click()
                wait_aspnet_idle(driver)
            except Exception as exc:
                print(
                    f"ERROR navigating during audit for study_id {study_id}. Skipping. Error: {exc}"
                )
                continue

            available_on_page = {}
            page_rows = driver.find_elements(
                by="xpath",
                value="//table[@id='TabContainer1_tabPanel1_gv1']//tr[position()>1]",
            )
            for row in page_rows:
                try:
                    cells = row.find_elements(by="tag name", value="td")
                    row_date = pd.to_datetime(cells[2].text).date()
                    row_desc = cells[3].text.strip()
                    available_on_page[(row_date, row_desc)] = True
                except Exception:
                    continue

            for index, pending_exam in patient_exams.iterrows():
                if max_exams is not None and audited >= max_exams:
                    break

                study_datetime = pending_exam.get("Study DateTime")
                study_desc = pending_exam.get("StudyDescription")
                if pd.isna(study_datetime) or pd.isna(study_desc):
                    continue

                target_key = (pd.to_datetime(study_datetime).date(), study_desc)

                audited += 1
                if target_key in available_on_page:
                    still_available += 1
                    if full_db is not None:
                        full_db.loc[index, "download_attempt_outcome"] = "Audit: Available"
                else:
                    marked_exported += 1
                    if full_db is not None:
                        full_db.loc[index, "is_exported"] = True
                        full_db.loc[index, "download_attempt_outcome"] = (
                            "Audit: Already Exported"
                        )
                        if "Status" in full_db.columns:
                            full_db.loc[index, "Status"] = "Already Exported"
                        if "Exported On" in full_db.columns and pd.isna(
                            full_db.loc[index, "Exported On"]
                        ):
                            full_db.loc[index, "Exported On"] = pd.Timestamp.now()

            if max_exams is not None and audited >= max_exams:
                break
    finally:
        if driver:
            print("Closing webdriver session from audit.")
            driver.quit()

    return {
        "audited": audited,
        "marked_exported": marked_exported,
        "still_available": still_available,
    }


def refresh_export_status(args, cycle_number: int):
    """Optionally reconcile export status during the wait window."""

    print(
        f"\n=== Refresh cycle {cycle_number}: auditing export status before next run ==="
    )

    refresh_db = load_and_merge_data()
    refresh_db = check_disk_for_downloads(refresh_db, BASE_DOWNLOAD_DIR)

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
        (base_modality == modality)
        & (~is_on_disk_series)
        & (~is_exported_series)
    ].copy()

    if candidates.empty:
        print("No pending exams require status reconciliation.")
        return

    max_to_audit = args.refresh_limit if args.refresh_limit > 0 else None
    if max_to_audit is not None:
        subset = candidates.sort_values(by=["Study DateTime", "study_id"]).head(
            max_to_audit
        )
    else:
        subset = candidates.sort_values(by=["Study DateTime", "study_id"])

    print(
        f"Auditing {len(subset)} exam(s) out of {len(candidates)} pending for modality {modality}."
    )

    audit_stats = audit_remote_export_status(subset, full_db=refresh_db, max_exams=max_to_audit)

    print(
        "Audit summary: "
        f"checked {audit_stats['audited']} exams — "
        f"marked {audit_stats['marked_exported']} as exported, "
        f"{audit_stats['still_available']} still available."
    )

    save_current_state(refresh_db)
    print(
        f"Audit metadata persisted to '{METADATA_FILE}'."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Identify and download ChiMEC imaging exams."
    )
    parser.add_argument(
        "--no-genotyping-filter",
        action="store_true",
        help="Include patients without genotyping data.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
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
        type=_parse_wait_interval,
        default=0.0,
        help=(
            "Seconds to wait between cycles. Accepts plain seconds (e.g. 3600) or "
            "values with units like 60m or 1h. Set to 0 to run a single cycle."
        ),
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=1,
        help=(
            "Number of cycles to run. Use 0 for unlimited cycles when --loop-wait > 0."
        ),
    )
    parser.add_argument(
        "--auto-confirm",
        action="store_true",
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

    args = parser.parse_args()

    if not all([USERNAME, PASSWORD]):
        print(
            "ERROR: IBROKER_USERNAME and IBROKER_PASSWORD must be set.", file=sys.stderr
        )
        sys.exit(1)

    cycles_run = 0
    try:
        while True:
            cycles_run += 1
            run_export_cycle(args, cycles_run)

            max_cycles = args.max_cycles
            if max_cycles > 0 and cycles_run >= max_cycles:
                break

            wait_seconds = args.loop_wait
            if wait_seconds <= 0:
                break

            if args.refresh_export_status:
                try:
                    refresh_export_status(args, cycles_run)
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
        print(
            f"\nRan {cycles_run} cycle{'s' if cycles_run != 1 else ''}."
        )


if __name__ == "__main__":
    main()
