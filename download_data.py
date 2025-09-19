#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys
from datetime import datetime

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
OUTPUT_DATABASE_FILE = "data/imaging_database_with_status.csv"

USERNAME = os.getenv("IBROKER_USERNAME")
PASSWORD = os.getenv("IBROKER_PASSWORD")


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


def load_and_merge_data():
    print("--- Phase 1: Loading and Merging Data ---")
    try:
        patients = pd.read_csv(CHIMEC_PATIENTS_FILE)
        print(f"Loaded {len(patients):,} rows from patient info file.")
        key = pd.read_csv(KEY_FILE)
        print(f"Loaded {len(key):,} rows from study_id-MRN key file.")
        metadata = pd.read_csv(METADATA_FILE)
        print(f"Loaded {len(metadata):,} exam records from raw metadata file.")

        # A small fix: The Accession number is often what we care about, not the old exam_id
        # Let's rename exam_id to Accession if Accession is missing.
        if "exam_id" in metadata.columns and "Accession" in metadata.columns:
            metadata["Accession"] = metadata["Accession"].fillna(metadata["exam_id"])

        metadata["base_modality"] = metadata["Modality"].apply(get_base_modality)
        print("  - Added 'base_modality' column.")
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

    conditions = [db["DatedxIndex"].notna(), db["_in_patient_file"].fillna(False)]
    choices = ["Case", "Control"]
    db["case_control_status"] = np.select(conditions, choices, default="Unknown")
    print("  - Derived Case/Control status based on DatedxIndex:")
    print(db["case_control_status"].value_counts().to_string())

    db.drop(columns=["_in_patient_file"], inplace=True)
    db.dropna(subset=["study_id", "Study DateTime", "StudyDescription"], inplace=True)

    # This column will be populated by check_disk_for_downloads
    db["is_exported"] = db["Accession"].notna()
    print(
        f"Marked {db['is_exported'].sum():,} exams as 'is_exported' based on having an Accession number."
    )

    print(f"\nMaster database created with {len(db):,} total exam records.")
    return db


def identify_download_targets(
    df: pd.DataFrame, filter_by_genotyping: bool, modality: str
):
    print("\n--- Phase 3: Identifying Target Exams for Download ---")
    df["rejection_reason"] = ""
    print(f"Initial pool: {len(df):,} exams")

    modality_mask = df["base_modality"] == modality
    df.loc[~modality_mask, "rejection_reason"] = f"Wrong modality (not {modality})"
    targets = df[modality_mask].copy()
    print(f"  - Kept {len(targets):,} exams after filtering for modality '{modality}'.")

    # NEW: Check against the on-disk status first
    mask_on_disk = targets["is_on_disk"]
    targets.loc[mask_on_disk, "rejection_reason"] = "Already on disk"
    targets = targets[~mask_on_disk]
    print(f"  - Rejected {mask_on_disk.sum():,} because they are already on disk.")

    # We still check if it's exported, but now it's a secondary check.
    # An exam could be exported but the sync failed, so it's not on disk.
    has_accession_mask = targets["is_exported"]
    targets.loc[has_accession_mask, "rejection_reason"] = (
        "Already exported (but not found on disk - possible sync issue)"
    )
    targets = targets[~has_accession_mask]
    print(
        f"  - Rejected {has_accession_mask.sum():,} exams already in iBroker's 'Exported' list (but not on disk)."
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


def execute_downloads(targets_df: pd.DataFrame, batch_size: int):
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
                else:
                    print(
                        f"  - INFO: Exam from {target_key[0]} is no longer available (already exported)."
                    )
                    outcomes[index] = "Already Exported"
                    already_exported_counter += 1

            if requested_this_patient:
                print("  Submitting export request for selected exams...")
                driver.find_element(by="name", value="btnExport").click()
                wait_aspnet_idle(driver)
                print("  Export request submitted.")
    finally:
        if driver:
            print("Closing webdriver session.")
            driver.quit()

    return outcomes, already_exported_counter, successfully_exported_counter


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
        help="Max number of exams to request in one run.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="MG",
        help="Base modality to filter for (e.g., MG, MR, CT).",
    )
    args = parser.parse_args()

    if not all([USERNAME, PASSWORD]):
        print(
            "ERROR: IBROKER_USERNAME and IBROKER_PASSWORD must be set.", file=sys.stderr
        )
        sys.exit(1)

    db = load_and_merge_data()
    db = check_disk_for_downloads(db, BASE_DOWNLOAD_DIR)

    targets = identify_download_targets(
        db,
        filter_by_genotyping=not args.no_genotyping_filter,
        modality=args.modality.upper(),
    )

    db["is_target"] = False
    db.loc[targets.index, "is_target"] = True
    db["download_attempt_outcome"] = pd.NA
    db["export_requested_on"] = pd.NaT

    if targets.empty:
        print("\nNo new exams to download based on the current criteria.")
    else:
        print("\n--- Summary of Exams to Download ---")
        print(f"Total potential targets: {len(targets)}")
        print(f"Unique patients to process: {targets['study_id'].nunique()}")

        proceed = (
            input(
                f"\nProceed with attempting to export up to {args.batch_size} exams? (y/n): "
            )
            .lower()
            .strip()
        )
        if proceed == "y":
            outcomes, already_exported_count, successfully_exported_count = (
                execute_downloads(targets, args.batch_size)
            )

            if outcomes:
                for index, outcome in outcomes.items():
                    db.loc[index, "download_attempt_outcome"] = outcome
                    if outcome == "Request Submitted":
                        db.loc[index, "export_requested_on"] = datetime.now()

            print("\n--- Run Summary ---")
            print(
                f"Successfully submitted export requests for: {successfully_exported_count} exams."
            )
            print(
                f"Discovered to be already exported:        {already_exported_count} exams."
            )
            print(f"Total exams processed:                    {len(outcomes)} exams.")
        else:
            print("Download cancelled by user.")

    final_cols = [
        "study_id",
        "MRN",
        "case_control_status",
        "chip",
        "base_modality",
        "Study DateTime",
        "DatedxIndex",
        "StudyDescription",
        "Accession",
        "is_exported",  # New name: True if it has an Accession
        "is_on_disk",  # New column: True if found on filesystem
        "is_target",
        "download_attempt_outcome",
        "export_requested_on",
        "rejection_reason",
        "Modality",
    ]
    db_final = db[[col for col in final_cols if col in db.columns]].copy()

    db_final.to_csv(OUTPUT_DATABASE_FILE, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(
        f"\nProcess complete. A full snapshot of this run has been saved to '{OUTPUT_DATABASE_FILE}'"
    )


if __name__ == "__main__":
    main()
