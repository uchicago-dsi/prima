#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

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

    conditions = [
        db["DatedxIndex"].notna(),
        db["_in_patient_file"] == True,
    ]
    choices = ["Case", "Control"]
    db["case_control_status"] = np.select(conditions, choices, default="Unknown")
    print("  - Derived Case/Control status based on DatedxIndex:")
    print(db["case_control_status"].value_counts().to_string())

    db.drop(columns=["_in_patient_file"], inplace=True)
    db.dropna(subset=["study_id", "Study DateTime", "StudyDescription"], inplace=True)

    print(f"\nMaster database created with {len(db):,} total exam records.")
    return db


def check_disk_for_downloads(df: pd.DataFrame, basedir: str):
    print(f"\n--- Phase 2: Auditing Filesystem for Downloads in {basedir} ---")
    df["is_downloaded"] = False
    exams_with_accession = df[df["Accession"].notna()]
    if exams_with_accession.empty:
        print(
            "No exams with known Accession Numbers in metadata. Cannot pre-emptively check download status."
        )
        return df
    found_accessions = set()
    if os.path.isdir(basedir):
        all_patient_dirs = [d for d in os.scandir(basedir) if d.is_dir()]
        for entry in tqdm(all_patient_dirs, desc="Scanning patient dirs"):
            for sub_entry in os.scandir(entry.path):
                accession = re.split(r"[-.]", sub_entry.name)[0]
                if accession:
                    found_accessions.add(accession)
    else:
        print(f"WARNING: Base directory not found: {basedir}.", file=sys.stderr)
    print(
        f"Found {len(found_accessions):,} unique downloaded accession numbers on disk."
    )
    downloaded_mask = df["Accession"].isin(found_accessions)
    df.loc[downloaded_mask, "is_downloaded"] = True
    print(
        f"{df['is_downloaded'].sum():,} exams in the database are marked as downloaded."
    )
    return df


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

    mask_downloaded = targets["is_downloaded"] == True
    targets.loc[mask_downloaded, "rejection_reason"] = "Already on disk"
    targets = targets[~mask_downloaded]
    print(f"  - Rejected {mask_downloaded.sum():,} because they are on disk.")

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

    has_accession_mask = targets["Accession"].notna()
    targets.loc[has_accession_mask, "rejection_reason"] = (
        "Already has Accession (in exported list)"
    )
    targets = targets[~has_accession_mask]
    print(
        f"  - Rejected {has_accession_mask.sum():,} exams already in iBroker's 'Exported' list."
    )

    print(f"  = {len(targets):,} final potential target exams identified.")
    return targets.sort_values(by=["study_id", "Study DateTime"])


def execute_downloads(targets_df: pd.DataFrame, batch_size: int):
    """Verifies exam availability live and requests exports."""
    if targets_df.empty:
        return {}, 0

    batch_targets = targets_df.head(batch_size).copy()
    print(
        f"\n--- Phase 4: Executing Downloads for a Batch of {len(batch_targets)} Exams ---"
    )
    driver = None
    # This will map the DataFrame index to the outcome
    outcomes = {}
    already_exported_counter = 0

    try:
        driver = make_driver()
        login(driver, USERNAME, PASSWORD)

        for study_id, patient_exams in tqdm(
            batch_targets.groupby("study_id"), desc="Processing Patients"
        ):
            print(f"\nProcessing patient {study_id} ({len(patient_exams)} exams)...")
            try:
                driver.find_element(by="name", value="tbxAssignedID").clear()
                driver.find_element(by="name", value="tbxAssignedID").send_keys(
                    str(int(study_id))
                )
                driver.find_element(by="name", value="btnFetch").click()
                wait_aspnet_idle(driver)
            except Exception as e:
                print(f"ERROR navigating for study_id {study_id}. Skipping. Error: {e}")
                continue

            # Build a map of what's ACTUALLY available on the live page
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
            # Now, check our targets against the live reality
            for index, target_exam in patient_exams.iterrows():
                target_key = (
                    target_exam["Study DateTime"].date(),
                    target_exam["StudyDescription"],
                )

                if target_key in available_on_page:
                    # It's available! Click it.
                    print(
                        f"  - Found available exam from {target_key[0]}. Selecting checkbox."
                    )
                    available_on_page[target_key].click()
                    outcomes[index] = "Request Submitted"
                    requested_this_patient = True
                else:
                    # It's NOT available - must have been exported already.
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

    return outcomes, already_exported_counter


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
        default=20,
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
        print(f"First 5 patients: {targets['study_id'].unique().tolist()[:5]}")

        proceed = (
            input(
                f"\nProceed with attempting to request the first batch of up to {args.batch_size} exams? (y/n): "
            )
            .lower()
            .strip()
        )
        if proceed == "y":
            outcomes, already_exported_count = execute_downloads(
                targets, args.batch_size
            )

            submitted_count = 0
            if outcomes:
                for index, outcome in outcomes.items():
                    db.loc[index, "download_attempt_outcome"] = outcome
                    if outcome == "Request Submitted":
                        db.loc[index, "export_requested_on"] = datetime.now()
                        submitted_count += 1

            print("\n--- Run Summary ---")
            print(
                f"Successfully submitted export requests for: {submitted_count} exams."
            )
            print(
                f"Discovered to be already exported:        {already_exported_count} exams."
            )
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
        "is_downloaded",
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
