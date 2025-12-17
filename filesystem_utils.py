#!/usr/bin/env python3
"""
Shared filesystem utilities for ChiMEC imaging data management.
Contains functions for scanning and inventorying downloaded imaging data.
"""

import os
from pathlib import Path
from typing import Dict, Set

import pandas as pd
from tqdm.auto import tqdm


KNOWN_MODALITY_CODES = {
    "CR",
    "CT",
    "DX",
    "ES",
    "MG",
    "MR",
    "NM",
    "PT",
    "PX",
    "RF",
    "RG",
    "US",
    "XA",
    "XC",
}


def build_disk_inventory(basedir: str) -> Dict[str, Set[str]]:
    """
    performs a one-time, efficient scan of the download directory to build an in-memory
    inventory of all existing exams and provides a detailed summary.

    handles two directory structures:
    - flat: basedir/{patient_id}/{accession}/
    - modality-grouped: basedir/{modality}/{patient_id}/{accession}/

    handles various naming conventions (e.g., ACCESSION, ACCESSION-DATE, ACCESSION.tar.xz).

    Parameters
    ----------
    basedir : str
        the root directory where patient data is stored.

    Returns
    -------
    Dict[str, Set[str]]
        a dictionary mapping {patient_id: {set_of_accession_numbers}}.
    """
    print("--- pre-scanning all patient directories to build file inventory ---")
    inventory = {}
    total_exam_count = 0

    if not Path(basedir).is_dir():
        print(f"WARNING: base download directory not found: {basedir}")
        print("inventory will be empty.")
        return inventory

    try:
        top_level_dirs = [d for d in os.scandir(basedir) if d.is_dir()]
    except PermissionError:
        print(f"WARNING: permission denied to read directory: {basedir}")
        return inventory

    # check if top-level contains modality subdirs or patient dirs directly
    modality_dirs = [d for d in top_level_dirs if d.name in KNOWN_MODALITY_CODES]
    patient_dirs_at_root = [
        d for d in top_level_dirs if d.name not in KNOWN_MODALITY_CODES
    ]

    # collect all patient directories to scan
    all_patient_dirs = []

    if modality_dirs:
        print(
            f"  - detected modality subdirectories: {[d.name for d in modality_dirs]}"
        )
        for mod_dir in modality_dirs:
            try:
                patient_dirs_in_mod = [
                    d for d in os.scandir(mod_dir.path) if d.is_dir()
                ]
                all_patient_dirs.extend(patient_dirs_in_mod)
            except PermissionError:
                continue

    # also include any patient dirs at root level (handles mixed structure)
    all_patient_dirs.extend(patient_dirs_at_root)

    print(f"  - found {len(all_patient_dirs):,} patient directories to scan")

    for entry in tqdm(all_patient_dirs, desc="building file inventory"):
        patient_id = entry.name
        found_accessions = set()
        try:
            # scan for subdirectories and files (accession numbers)
            for item in os.scandir(entry.path):
                accession_number = None

                if item.is_dir():
                    # directory format: "2O42657" or "2O42657-2015-12-08"
                    # extract accession number (part before first dash)
                    accession_number = item.name.split("-")[0]
                elif item.is_file() and item.name.endswith(".tar.xz"):
                    # compressed file format: "2O42660.tar.xz"
                    # extract accession number (part before .tar.xz)
                    accession_number = item.name.replace(".tar.xz", "")

                if accession_number:
                    found_accessions.add(accession_number)

            if found_accessions:
                # merge with existing inventory for this patient (may appear in multiple modalities)
                if patient_id in inventory:
                    inventory[patient_id].update(found_accessions)
                else:
                    inventory[patient_id] = found_accessions
                total_exam_count += len(found_accessions)
        except (FileNotFoundError, PermissionError):
            continue

    # detailed summary print statements
    patient_count = len(inventory)
    print("\n--- inventory summary ---")
    print(f"  - scanned {len(all_patient_dirs):,} patient directories.")
    print(f"  - found downloaded data for {patient_count:,} unique patients.")
    print(f"  - inventoried a total of {total_exam_count:,} unique exams on disk.")
    print("-------------------------\n")
    return inventory


def update_metadata_with_disk_status(
    metadata_df: pd.DataFrame, basedir: str
) -> pd.DataFrame:
    """
    update metadata with current filesystem status by scanning disk inventory

    Parameters
    ----------
    metadata_df : pd.DataFrame
        the metadata dataframe to update
    basedir : str
        the base directory where patient data is stored

    Returns
    -------
    pd.DataFrame
        updated metadata with corrected is_on_disk column
    """
    print("\n=== UPDATING METADATA WITH CURRENT DISK STATUS ===")

    # build current disk inventory
    disk_inventory = build_disk_inventory(basedir)

    # create corrected is_on_disk column
    metadata_df["is_on_disk_corrected"] = False

    # filter for rows that can be checked (have Accession and study_id)
    checkable_mask = metadata_df["Accession"].notna() & metadata_df["study_id"].notna()
    checkable_df = metadata_df[checkable_mask].copy()

    if checkable_df.empty:
        print(
            "no exams with accession numbers found in metadata. cannot check filesystem."
        )
        return metadata_df

    def check_row_on_disk(row):
        """check if a specific exam row is found on disk"""
        try:
            patient_id_str = str(int(row["study_id"]))
            accession_str = str(row["Accession"])
            patient_inventory = disk_inventory.get(patient_id_str, set())
            return accession_str in patient_inventory
        except (ValueError, TypeError):
            return False

    print("matching metadata against current file inventory...")
    # apply check to rows that can be checked
    on_disk_mask = checkable_df.apply(check_row_on_disk, axis=1)

    # update the corrected column for matching rows
    metadata_df.loc[on_disk_mask[on_disk_mask].index, "is_on_disk_corrected"] = True

    # summary of changes
    total_checkable = len(checkable_df)
    found_on_disk = on_disk_mask.sum()

    print(f"  - checked {total_checkable:,} exams with accession numbers")
    print(f"  - found {found_on_disk:,} exams currently on disk")

    if "is_on_disk" in metadata_df.columns:
        # compare with existing is_on_disk column
        old_on_disk = metadata_df["is_on_disk"].sum()
        print(f"  - previous is_on_disk count: {old_on_disk:,}")
        print(f"  - updated is_on_disk count: {found_on_disk:,}")
        print(f"  - difference: {found_on_disk - old_on_disk:+,}")

    print("updated metadata with current disk status")
    return metadata_df


def check_disk_for_downloads(df: pd.DataFrame, basedir: str) -> pd.DataFrame:
    """
    audits the filesystem to find which exams have been downloaded.
    handles various naming conventions (e.g., ACCESSION, ACCESSION-DATE, ACCESSION.tar.xz).

    this is an alias for update_metadata_with_disk_status for backward compatibility
    with download_data.py
    """
    print(f"\n--- Phase 2: Auditing Filesystem for Downloads in {basedir} ---")

    # initialize the new column with False
    df["is_on_disk"] = False

    # filter for rows that *could* be checked (have an Accession and study_id)
    checkable_df = df.dropna(subset=["Accession", "study_id"]).copy()
    if checkable_df.empty:
        print(
            "No exams with Accession Numbers found in metadata. Cannot check filesystem."
        )
        return df

    # build inventory using shared function
    inventory = build_disk_inventory(basedir)

    # logic to check against the inventory
    def check_row(row):
        # ensure study_id is a clean string without decimals for dict lookup
        patient_id = str(int(row["study_id"]))
        accession = str(row["Accession"])

        # check if we have any inventory for this patient, and if the accession is in their set
        if patient_id in inventory and accession in inventory[patient_id]:
            return True
        return False

    print("Matching metadata against file inventory...")
    # apply the check only to the relevant subset of the DataFrame
    on_disk_mask = checkable_df.apply(check_row, axis=1)

    # update the original DataFrame using the index from our checkable subset
    df.loc[on_disk_mask[on_disk_mask].index, "is_on_disk"] = True

    on_disk_count = df["is_on_disk"].sum()
    print(f"  - Marked {on_disk_count:,} exams in the database as 'is_on_disk'.")

    if on_disk_count > 0:
        on_disk_with_accession = df[df["is_on_disk"]]["Accession"].notna().sum()
        print(
            f"  - Exams on disk with Accession numbers: {on_disk_with_accession:,}/{on_disk_count:,} ({on_disk_with_accession / on_disk_count * 100:.1f}%)"
        )

    return df
