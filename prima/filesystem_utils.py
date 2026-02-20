#!/usr/bin/env python3
"""
Shared filesystem utilities for ChiMEC imaging data management.
Contains functions for scanning and inventorying downloaded imaging data.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd
import pydicom
from tqdm.auto import tqdm

# suppress pydicom VR UI validation warnings for non-standard UIDs
warnings.filterwarnings("ignore", message=".*Invalid value for VR UI.*", append=True)

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

    # handle None fingerprint_cache (e.g., when dataset-specific cache doesn't exist)
    if fingerprint_cache is None:
        print("  Note: No fingerprint cache provided, skipping disk status check")
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


def _normalize_accession(accession: str) -> str:
    """Normalize accession string for robust matching."""
    return re.sub(r"[^A-Za-z0-9]", "", str(accession).upper())


def _extract_date_from_entry_name(entry_name: str) -> str | None:
    """Extract YYYY-MM-DD date suffix from disk entry name when present."""
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})$", entry_name)
    if match is None:
        return None
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"


def _read_study_date_from_exam_dir(exam_dir: Path) -> str | None:
    """Read StudyDate from the first parseable DICOM file in an exam directory."""
    if not exam_dir.exists() or not exam_dir.is_dir():
        return None
    try:
        for candidate in exam_dir.rglob("*"):
            try:
                if not candidate.is_file():
                    continue
            except OSError:
                continue
            try:
                dcm = pydicom.dcmread(str(candidate), stop_before_pixels=True)
                study_date = str(dcm.get("StudyDate", "")).strip()
                if len(study_date) >= 8 and study_date[:8].isdigit():
                    return f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:8]}"
            except Exception:
                continue
    except OSError:
        return None
    return None


def _read_light_fingerprint_from_exam_dir(exam_dir: Path) -> dict | None:
    """
    Read StudyInstanceUID, StudyDate, StudyTime from the first DICOM in an exam dir.
    Never uses filename; always from DICOM metadata.
    Returns dict with keys: study_uid, study_date (YYYYMMDD), study_time, or None if unreadable.
    """
    if not exam_dir.exists() or not exam_dir.is_dir():
        return None
    try:
        for candidate in exam_dir.rglob("*"):
            try:
                if not candidate.is_file():
                    continue
            except OSError:
                continue
            try:
                dcm = pydicom.dcmread(str(candidate), stop_before_pixels=True)
                study_uid = str(dcm.get("StudyInstanceUID", "")).strip()
                if not study_uid:
                    continue
                study_date = str(dcm.get("StudyDate", "")).strip()
                study_date = (
                    study_date[:8]
                    if len(study_date) >= 8 and study_date[:8].isdigit()
                    else None
                )
                study_time = (
                    str(dcm.get("StudyTime", "")).strip()
                    if dcm.get("StudyTime")
                    else None
                )
                return {
                    "study_uid": study_uid,
                    "study_date": study_date,
                    "study_time": study_time,
                }
            except Exception:
                continue
    except OSError:
        return None
    return None


def build_chimec_disk_fingerprints(
    basedir: str,
    output_dir: Path | str = "fingerprints/chimec",
    modality: str = "MG",
) -> tuple[Path, Path]:
    """
    Build light fingerprints for ChiMEC disk exams: study_id, study_date, study_uid.
    Always reads from DICOM metadata; never uses filename for date.

    Parameters
    ----------
    basedir : str
        Base directory (e.g. /gpfs/data/huo-lab/Image/ChiMEC/MG or
        /gpfs/data/huo-lab/Image/ChiMEC for modality subdirs)
    output_dir : Path | str
        Directory for output files (default: fingerprints/chimec)
    modality : str
        Modality code (e.g. MG). If basedir contains modality subdirs, scans that subdir.

    Returns
    -------
    tuple[Path, Path]
        (json_path, csv_path)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "disk_fingerprints.json"
    csv_path = output_path / "disk_fingerprints.csv"

    base_path = Path(basedir)
    if not base_path.exists():
        raise RuntimeError(f"disk path does not exist: {basedir}")

    modality_upper = modality.upper()
    modality_subdir = base_path / modality_upper
    if modality_subdir.exists() and modality_subdir.is_dir():
        scan_root = modality_subdir
    else:
        scan_root = base_path

    patient_dirs = [d for d in os.scandir(scan_root) if d.is_dir() and d.name.isdigit()]
    print(
        f"Building ChiMEC fingerprints from {scan_root} ({len(patient_dirs):,} patient dirs)"
    )

    disk_inventory: Dict[str, Dict[str, tuple]] = {}
    csv_rows: list[dict] = []

    for patient_dir in tqdm(patient_dirs, desc="fingerprinting"):
        patient_id = patient_dir.name
        exams: Dict[str, tuple] = {}

        try:
            for item in os.scandir(patient_dir.path):
                entry_name = item.name
                accession = None
                if item.is_dir():
                    accession = item.name.split("-")[0]
                    exam_path = Path(item.path)
                elif item.is_file() and item.name.endswith(".tar.xz"):
                    accession = item.name[:-7]
                    continue
                else:
                    continue
                if not accession:
                    continue

                fp = _read_light_fingerprint_from_exam_dir(exam_path)
                if fp is None:
                    continue

                study_uid = fp["study_uid"]
                study_date = fp["study_date"]
                study_time = fp.get("study_time")

                exams[entry_name] = (study_uid, [], study_date, study_time)

                study_date_iso = study_date
                if study_date and len(study_date) >= 8:
                    study_date_iso = (
                        f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:8]}"
                    )
                csv_rows.append(
                    {
                        "patient_id": patient_id,
                        "entry_name": entry_name,
                        "study_uid": study_uid,
                        "study_date": study_date,
                        "study_date_iso": study_date_iso or "",
                        "study_time": study_time or "",
                        "accession": accession,
                    }
                )
        except (PermissionError, FileNotFoundError, OSError):
            continue

        if exams:
            disk_inventory[patient_id] = exams

    with open(json_path, "w") as f:
        json.dump(disk_inventory, f, indent=2)

    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(csv_path, index=False)

    total_exams = sum(len(e) for e in disk_inventory.values())
    print(
        f"  Wrote {json_path} ({len(disk_inventory):,} patients, {total_exams:,} exams) "
        f"and {csv_path}"
    )
    return json_path, csv_path


def reconcile_disk_ibroker_accessions(
    metadata_df: pd.DataFrame,
    basedir: str,
    *,
    modality: str = "MG",
    output_csv: Path | None = None,
    fingerprint_cache: Path | None = None,
    key_file: Path | str | None = None,
) -> dict[str, int]:
    """Reconcile disk exams vs iBroker metadata and classify mismatch reasons.

    When fingerprint_cache is provided and exists, disk dates are taken from DICOM
    metadata in the fingerprint (never from filename). Otherwise falls back to
    scanning disk and inferring date from filename or DICOM.

    When key_file is provided (MRN-to-study_id map with AnonymousID column),
    computes overlap between disk study IDs and key study IDs.
    """
    modality_upper = modality.upper()
    subset_all = metadata_df[metadata_df["base_modality"] == modality_upper].copy()
    subset_all["study_id"] = subset_all["study_id"].astype("string")
    subset_all["Study DateTime"] = pd.to_datetime(
        subset_all["Study DateTime"], errors="coerce"
    )
    subset_all["study_date"] = subset_all["Study DateTime"].dt.strftime("%Y-%m-%d")
    ib_all_ids = set(subset_all["study_id"].dropna().astype(str))
    ib_all_date_pairs = set(
        zip(
            subset_all["study_id"].dropna().astype(str),
            subset_all["study_date"].fillna("").astype(str),
        )
    )

    subset = subset_all.copy()
    subset["ib_accession"] = subset["Accession"].astype("string")
    subset = subset.dropna(subset=["study_id", "ib_accession"])
    subset["ib_accession_norm"] = subset["ib_accession"].map(_normalize_accession)

    ib_key_set = set(zip(subset["study_id"], subset["ib_accession_norm"]))
    ib_date_groups = (
        subset.groupby(["study_id", "study_date"], dropna=True)["ib_accession_norm"]
        .agg(list)
        .to_dict()
    )
    ib_date_pairs = set(ib_date_groups.keys())
    ib_date_multiplicity = (
        subset.groupby(["study_id", "study_date"], dropna=True)["ib_accession_norm"]
        .size()
        .to_dict()
    )

    disk_records = []
    inferred_dates_from_dicom = 0
    tar_xz_entries = 0
    partial_output_csv = None
    if output_csv is not None:
        partial_output_csv = output_csv.with_name(
            f"{output_csv.stem}_partial{output_csv.suffix}"
        )

    if fingerprint_cache is not None and Path(fingerprint_cache).exists():
        print(
            f"Using fingerprint cache for disk dates (DICOM-only): {fingerprint_cache}"
        )
        with open(fingerprint_cache) as f:
            raw = json.load(f)
        for patient_id, exams in tqdm(
            raw.items(), desc="reconciling from fingerprints"
        ):
            for entry_name, data in exams.items():
                uid, hashes, study_date, study_time = data
                accession = entry_name.split("-")[0]
                accession_norm = _normalize_accession(accession)
                disk_date = None
                if study_date and len(study_date) >= 8:
                    disk_date = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:8]}"
                    inferred_dates_from_dicom += 1
                disk_records.append(
                    {
                        "study_id": patient_id,
                        "disk_accession": accession,
                        "disk_accession_norm": accession_norm,
                        "disk_date": disk_date,
                        "disk_entry_name": entry_name,
                        "study_uid": uid,
                    }
                )
    else:
        base_path = Path(basedir)
        if not base_path.exists():
            raise RuntimeError(f"disk path does not exist: {basedir}")

        patient_dirs = [
            d for d in os.scandir(base_path) if d.is_dir() and d.name.isdigit()
        ]
        for patient_idx, patient_dir in enumerate(
            tqdm(patient_dirs, desc="reconciling disk exams"), start=1
        ):
            patient_id = patient_dir.name
            try:
                for item in os.scandir(patient_dir.path):
                    accession = None
                    if item.is_dir():
                        accession = item.name.split("-")[0]
                        entry_name = item.name
                        disk_date = _extract_date_from_entry_name(entry_name)
                        if disk_date is None:
                            disk_date = _read_study_date_from_exam_dir(Path(item.path))
                            if disk_date is not None:
                                inferred_dates_from_dicom += 1
                    elif item.is_file() and item.name.endswith(".tar.xz"):
                        accession = item.name[:-7]
                        entry_name = item.name
                        disk_date = _extract_date_from_entry_name(entry_name)
                        tar_xz_entries += 1
                    else:
                        continue
                    if not accession:
                        continue
                    accession_norm = _normalize_accession(accession)
                    disk_records.append(
                        {
                            "study_id": patient_id,
                            "disk_accession": accession,
                            "disk_accession_norm": accession_norm,
                            "disk_date": disk_date,
                            "disk_entry_name": entry_name,
                            "study_uid": None,
                        }
                    )
            except (PermissionError, FileNotFoundError, OSError):
                continue

            if partial_output_csv is not None and patient_idx % 250 == 0:
                partial_df = pd.DataFrame(disk_records)
                partial_output_csv.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".tmp",
                    prefix=f".{partial_output_csv.name}.",
                    dir=partial_output_csv.parent,
                    delete=False,
                    encoding="utf-8",
                    newline="",
                ) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                    partial_df.to_csv(tmp_file, index=False)
                os.replace(tmp_path, partial_output_csv)

    disk_df = pd.DataFrame(disk_records)
    if disk_df.empty:
        return {
            "disk_total": 0,
            "ib_total": len(subset),
            "exact_match": 0,
            "accession_changed_unambiguous": 0,
            "accession_changed_ambiguous": 0,
            "disk_only": 0,
            "disk_no_date": 0,
            "disk_with_date": 0,
            "disk_tar_xz_entries": 0,
            "ib_multi_patient_date": 0,
            "disk_multi_patient_date": 0,
            "ib_patient_date_pairs": len(ib_date_pairs),
            "disk_patient_date_pairs": 0,
            "shared_patient_date_pairs": 0,
            "disk_rows_with_shared_patient_date": 0,
            "disk_dates_inferred_from_dicom": 0,
            "disk_only_with_date_pair_in_any_ibroker": 0,
            "disk_only_with_date_study_id_not_in_ibroker": 0,
            "disk_only_with_date_study_id_in_ibroker_but_date_not": 0,
            "study_ids_on_disk_not_in_ibroker": 0,
            "key_study_ids_count": 0,
            "disk_study_ids_in_key": 0,
            "disk_study_ids_not_in_key": 0,
            "key_study_ids_not_on_disk": 0,
        }

    disk_study_ids = set(disk_df["study_id"].astype(str))
    study_ids_on_disk_not_in_ibroker = len(disk_study_ids - ib_all_ids)

    key_study_ids: Set[str] = set()
    if key_file is not None:
        key_path = Path(key_file)
        if key_path.exists():
            key_df = pd.read_csv(key_path)
            if "AnonymousID" in key_df.columns:
                key_study_ids = set(key_df["AnonymousID"].dropna().astype(str))

    disk_df["match_type"] = "disk_only"
    disk_df["ib_accession_norm"] = pd.NA
    disk_df["ib_patient_date_count"] = 0
    disk_df["disk_patient_date_count"] = 0

    exact_mask = [
        (sid, acc_norm) in ib_key_set
        for sid, acc_norm in zip(disk_df["study_id"], disk_df["disk_accession_norm"])
    ]
    disk_df.loc[exact_mask, "match_type"] = "exact_patient_accession"

    unresolved = disk_df["match_type"] == "disk_only"
    unresolved_with_date = unresolved & disk_df["disk_date"].notna()
    for idx, row in disk_df[unresolved_with_date].iterrows():
        key = (row["study_id"], row["disk_date"])
        ib_accessions = ib_date_groups.get(key)
        if not ib_accessions:
            continue
        date_count = len(ib_accessions)
        disk_df.loc[idx, "ib_patient_date_count"] = date_count
        if date_count == 1:
            disk_df.loc[idx, "match_type"] = "accession_changed_unambiguous"
            disk_df.loc[idx, "ib_accession_norm"] = ib_accessions[0]
        else:
            disk_df.loc[idx, "match_type"] = "accession_changed_ambiguous"

    disk_known_date = disk_df[disk_df["disk_date"].notna()].copy()
    disk_date_pair_set = set(
        zip(disk_known_date["study_id"], disk_known_date["disk_date"])
    )
    shared_date_pairs = disk_date_pair_set & ib_date_pairs
    disk_rows_with_shared_date = int(
        (
            disk_df["disk_date"].notna()
            & [
                (sid, d) in shared_date_pairs
                for sid, d in zip(disk_df["study_id"], disk_df["disk_date"])
            ]
        ).sum()
    )
    if not disk_known_date.empty:
        disk_date_mult = (
            disk_known_date.groupby(["study_id", "disk_date"])["disk_accession_norm"]
            .size()
            .to_dict()
        )
        for idx, row in disk_known_date.iterrows():
            disk_df.loc[idx, "disk_patient_date_count"] = disk_date_mult.get(
                (row["study_id"], row["disk_date"]), 0
            )
    else:
        disk_date_mult = {}

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        disk_df.to_csv(output_csv, index=False)
        if partial_output_csv is not None and partial_output_csv.exists():
            partial_output_csv.unlink()

    summary = {
        "disk_total": len(disk_df),
        "ib_total": len(subset),
        "exact_match": int((disk_df["match_type"] == "exact_patient_accession").sum()),
        "accession_changed_unambiguous": int(
            (disk_df["match_type"] == "accession_changed_unambiguous").sum()
        ),
        "accession_changed_ambiguous": int(
            (disk_df["match_type"] == "accession_changed_ambiguous").sum()
        ),
        "disk_only": int((disk_df["match_type"] == "disk_only").sum()),
        "disk_no_date": int(disk_df["disk_date"].isna().sum()),
        "disk_with_date": int(disk_df["disk_date"].notna().sum()),
        "ib_patient_date_pairs": int(len(ib_date_pairs)),
        "disk_patient_date_pairs": int(len(disk_date_pair_set)),
        "shared_patient_date_pairs": int(len(shared_date_pairs)),
        "disk_rows_with_shared_patient_date": int(disk_rows_with_shared_date),
        "ib_multi_patient_date": int(
            sum(1 for count in ib_date_multiplicity.values() if count > 1)
        ),
        "disk_multi_patient_date": int(
            sum(1 for count in disk_date_mult.values() if count > 1)
        ),
        "disk_dates_inferred_from_dicom": int(inferred_dates_from_dicom),
        "disk_tar_xz_entries": int(tar_xz_entries),
    }
    disk_only_df = disk_df[disk_df["match_type"] == "disk_only"].copy()
    disk_only_with_date = disk_only_df[disk_only_df["disk_date"].notna()].copy()
    if disk_only_with_date.empty:
        summary["disk_only_with_date_pair_in_any_ibroker"] = 0
        summary["disk_only_with_date_study_id_not_in_ibroker"] = 0
        summary["disk_only_with_date_study_id_in_ibroker_but_date_not"] = 0
    else:
        pair_in_any = [
            (sid, date) in ib_all_date_pairs
            for sid, date in zip(
                disk_only_with_date["study_id"].astype(str),
                disk_only_with_date["disk_date"].astype(str),
            )
        ]
        sid_in_any = disk_only_with_date["study_id"].astype(str).isin(ib_all_ids)
        pair_in_any_series = pd.Series(pair_in_any, index=disk_only_with_date.index)
        summary["disk_only_with_date_pair_in_any_ibroker"] = int(
            pair_in_any_series.sum()
        )
        summary["disk_only_with_date_study_id_not_in_ibroker"] = int(
            (~sid_in_any).sum()
        )
        summary["disk_only_with_date_study_id_in_ibroker_but_date_not"] = int(
            (sid_in_any & ~pair_in_any_series).sum()
        )
    summary["study_ids_on_disk_not_in_ibroker"] = int(study_ids_on_disk_not_in_ibroker)
    if key_study_ids:
        summary["key_study_ids_count"] = int(len(key_study_ids))
        summary["disk_study_ids_in_key"] = int(len(disk_study_ids & key_study_ids))
        summary["disk_study_ids_not_in_key"] = int(len(disk_study_ids - key_study_ids))
        summary["key_study_ids_not_on_disk"] = int(len(key_study_ids - disk_study_ids))
    else:
        summary["key_study_ids_count"] = 0
        summary["disk_study_ids_in_key"] = 0
        summary["disk_study_ids_not_in_key"] = 0
        summary["key_study_ids_not_on_disk"] = 0
    return summary
