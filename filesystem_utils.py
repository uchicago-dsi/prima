#!/usr/bin/env python3
"""
Shared filesystem utilities for ChiMEC imaging data management.
Contains functions for scanning and inventorying downloaded imaging data.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd
from tqdm.auto import tqdm

# default fingerprint cache location
DEFAULT_FINGERPRINT_CACHE = Path("data/destination_fingerprints.json")


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


def load_disk_dates_from_fingerprints(
    fingerprint_cache: Path = DEFAULT_FINGERPRINT_CACHE,
) -> Dict[str, Set[str]]:
    """
    load fingerprint cache and extract StudyDates for each patient.

    Parameters
    ----------
    fingerprint_cache : Path
        path to the fingerprint JSON cache file

    Returns
    -------
    Dict[str, Set[str]]
        mapping of patient_id → {set of study dates as YYYY-MM-DD}
    """
    if not fingerprint_cache.exists():
        print(f"WARNING: fingerprint cache not found: {fingerprint_cache}")
        return {}

    print(f"Loading disk dates from fingerprint cache: {fingerprint_cache}")
    with open(fingerprint_cache) as f:
        raw_data = json.load(f)

    patient_dates = {}
    exams_without_dates = 0

    for patient_id, exams in raw_data.items():
        dates = set()
        for exam_name, data in exams.items():
            # format: (uid, hashes, study_date, study_time)
            uid, hashes, study_date, study_time = data
            if study_date and len(study_date) >= 8:
                # normalize DICOM date (YYYYMMDD) to ISO (YYYY-MM-DD)
                date_str = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:8]}"
                dates.add(date_str)
            else:
                exams_without_dates += 1

        if dates:
            patient_dates[patient_id] = dates

    total_dates = sum(len(d) for d in patient_dates.values())
    print(
        f"  - loaded {len(patient_dates):,} patients with {total_dates:,} unique exam dates"
    )
    if exams_without_dates > 0:
        print(f"  - exams without StudyDate: {exams_without_dates:,}")

    return patient_dates


def update_metadata_with_disk_status_by_date(
    metadata_df: pd.DataFrame,
    fingerprint_cache: Path = DEFAULT_FINGERPRINT_CACHE,
    conservative: bool = False,
) -> pd.DataFrame:
    """
    update metadata with disk status using StudyDate matching from fingerprint cache.

    this is more robust than accession-only matching because:
    - works even if accession is missing from metadata
    - handles format differences in accession numbers

    Parameters
    ----------
    metadata_df : pd.DataFrame
        the metadata dataframe to update (must have study_id and Study DateTime columns)
    fingerprint_cache : Path
        path to the fingerprint JSON cache file
    conservative : bool
        if True, only mark is_on_disk=True when there's a 1:1 match (one ibroker row
        and one disk exam for that patient/date). use this for export decisions to
        avoid skipping exams we might not actually have. default False.

    Returns
    -------
    pd.DataFrame
        updated metadata with is_on_disk column
    """
    print("\n=== UPDATING METADATA WITH DISK STATUS (StudyDate matching) ===")
    if conservative:
        print("  (conservative mode: only 1:1 matches count as on disk)")

    # handle empty DataFrames
    if len(metadata_df) == 0:
        print("  WARNING: metadata DataFrame is empty, skipping disk status update")
        metadata_df["is_on_disk"] = False
        return metadata_df

    # load disk dates from fingerprint cache
    disk_dates = load_disk_dates_from_fingerprints(fingerprint_cache)

    if not disk_dates:
        print("WARNING: no disk dates loaded, cannot update metadata")
        metadata_df["is_on_disk"] = False
        return metadata_df

    # load raw fingerprints to count exams per (patient_id, date)
    if fingerprint_cache.exists():
        with open(fingerprint_cache) as f:
            raw_fingerprints = json.load(f)
        total_disk_exams = sum(len(exams) for exams in raw_fingerprints.values())

        # count exams per (patient_id, date) on disk
        disk_exam_counts = {}  # (patient_id, date) -> count of exams
        for patient_id, exams in raw_fingerprints.items():
            for exam_name, data in exams.items():
                uid, hashes, study_date, study_time = data
                if study_date and len(study_date) >= 8:
                    date_str = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:8]}"
                    key = (patient_id, date_str)
                    disk_exam_counts[key] = disk_exam_counts.get(key, 0) + 1
    else:
        total_disk_exams = 0
        disk_exam_counts = {}

    disk_exam_keys = set(disk_exam_counts.keys())

    # ensure Study DateTime is datetime
    if "Study DateTime" not in metadata_df.columns:
        print("ERROR: metadata missing 'Study DateTime' column")
        metadata_df["is_on_disk"] = False
        return metadata_df

    metadata_df["Study DateTime"] = pd.to_datetime(
        metadata_df["Study DateTime"], errors="coerce"
    )

    # extract date string for matching
    metadata_df["_study_date_str"] = metadata_df["Study DateTime"].dt.strftime(
        "%Y-%m-%d"
    )

    # count ibroker rows per (patient_id, date)
    ibroker_exam_counts = {}
    for _, row in metadata_df.iterrows():
        try:
            patient_id = str(int(row["study_id"]))
            date_str = row["_study_date_str"]
            if date_str and not pd.isna(date_str):
                key = (patient_id, date_str)
                ibroker_exam_counts[key] = ibroker_exam_counts.get(key, 0) + 1
        except (ValueError, TypeError):
            pass

    ibroker_exam_keys = set(ibroker_exam_counts.keys())

    def check_on_disk(row):
        """check if exam is on disk by matching (patient_id, study_date)"""
        try:
            patient_id = str(int(row["study_id"]))
            date_str = row["_study_date_str"]
            if pd.isna(date_str) or not date_str:
                return False
            key = (patient_id, date_str)
            if key not in disk_exam_counts:
                return False
            if conservative:
                # only count as on disk if 1:1 match (no ambiguity)
                return (
                    disk_exam_counts[key] == 1 and ibroker_exam_counts.get(key, 0) == 1
                )
            return True
        except (ValueError, TypeError):
            return False

    print("matching metadata against disk dates...")
    if len(metadata_df) == 0:
        metadata_df["is_on_disk"] = pd.Series([], dtype=bool, index=metadata_df.index)
    else:
        is_on_disk_series = metadata_df.apply(check_on_disk, axis=1)
        metadata_df["is_on_disk"] = is_on_disk_series

    # find disk exams not in ibroker
    disk_only_keys = disk_exam_keys - ibroker_exam_keys
    disk_only_patients = set(k[0] for k in disk_only_keys)

    # find matched keys
    matched_keys = disk_exam_keys & ibroker_exam_keys

    # count ambiguous matches (multiple exams on same date)
    ambiguous_keys = {
        k
        for k in matched_keys
        if disk_exam_counts.get(k, 0) > 1 or ibroker_exam_counts.get(k, 0) > 1
    }

    # cleanup temp column
    metadata_df.drop(columns=["_study_date_str"], inplace=True)

    on_disk_count = metadata_df["is_on_disk"].sum()
    total_count = len(metadata_df)

    print("\n  SUMMARY:")
    print(f"  - unique (patient, date) pairs in ibroker: {len(ibroker_exam_keys):,}")
    print(f"  - unique (patient, date) pairs on disk:    {len(disk_exam_keys):,}")
    print(f"  - matched (patient, date) pairs:           {len(matched_keys):,}")
    print(f"  - ambiguous matches (>1 exam same date):   {len(ambiguous_keys):,}")
    print(
        f"  - disk-only (patient, date) pairs:         {len(disk_only_keys):,} ({len(disk_only_patients):,} patients)"
    )
    print("")
    print(
        f"  - ibroker rows marked is_on_disk=True: {on_disk_count:,} / {total_count:,}"
    )
    print(f"  - total individual exams on disk:      {total_disk_exams:,}")

    return metadata_df


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


def check_disk_for_downloads(
    df: pd.DataFrame,
    basedir: str,
    fingerprint_cache: Optional[Path] = None,
) -> pd.DataFrame:
    """
    audits the filesystem to find which exams have been downloaded.

    uses StudyDate matching from fingerprint cache (preferred) or falls back to
    accession-based matching if fingerprint cache is not available.

    Parameters
    ----------
    df : pd.DataFrame
        metadata dataframe with study_id, Study DateTime, and optionally Accession
    basedir : str
        base directory where patient data is stored (used for accession fallback)
    fingerprint_cache : Path, optional
        path to fingerprint cache. if None, uses DEFAULT_FINGERPRINT_CACHE.
        if that doesn't exist, falls back to accession-based matching.

    Returns
    -------
    pd.DataFrame
        dataframe with is_on_disk column updated
    """
    print("\n--- Phase 2: Auditing Filesystem for Downloads ---")

    # try date-based matching first (preferred)
    cache_path = fingerprint_cache or DEFAULT_FINGERPRINT_CACHE
    if cache_path.exists():
        print(f"Using StudyDate-based matching from: {cache_path}")
        return update_metadata_with_disk_status_by_date(df, cache_path)

    # fallback to accession-based matching
    print("Fingerprint cache not found, falling back to accession-based matching")
    print(f"Scanning directory: {basedir}")

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
