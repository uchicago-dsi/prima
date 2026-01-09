#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from filesystem_utils import update_metadata_with_disk_status_by_date

# scrape_ibroker imports are deferred to avoid credential check at import time
# when running in --status-only mode
login = None
make_driver = None
wait_aspnet_idle = None
bootstrap_http_session_from_driver = None
http_get_root = None
post_link_event = None
post_fetch_grid = None
parse_all_tables_from_page = None


def _import_scrape_ibroker():
    """Import scrape_ibroker functions (triggers credential check)."""
    global login, make_driver, wait_aspnet_idle
    global bootstrap_http_session_from_driver, http_get_root
    global post_link_event, post_fetch_grid, parse_all_tables_from_page
    from scrape_ibroker import bootstrap_http_session_from_driver as _bootstrap
    from scrape_ibroker import http_get_root as _http_get_root
    from scrape_ibroker import login as _login
    from scrape_ibroker import make_driver as _make_driver
    from scrape_ibroker import parse_all_tables_from_page as _parse_all_tables
    from scrape_ibroker import post_fetch_grid as _post_fetch_grid
    from scrape_ibroker import post_link_event as _post_link_event
    from scrape_ibroker import wait_aspnet_idle as _wait_aspnet_idle

    login = _login
    make_driver = _make_driver
    wait_aspnet_idle = _wait_aspnet_idle
    bootstrap_http_session_from_driver = _bootstrap
    http_get_root = _http_get_root
    post_link_event = _post_link_event
    post_fetch_grid = _post_fetch_grid
    parse_all_tables_from_page = _parse_all_tables


# --- Configuration ---
# ChiMEC dataset configuration
CHIMEC_PATIENTS_FILE = "/gpfs/data/phs/groups/Projects/Huo_projects/SPORE/annawoodard/List_ChiMEC_priority_2025July30.csv"
CHIMEC_KEY_FILE = "/gpfs/data/huo-lab/Image/ChiMEC/study-16352a.csv"
CHIMEC_METADATA_FILE = "data/imaging_metadata.csv"
CHIMEC_EXPORT_STATE_FILE = "data/export_state.csv"
CHIMEC_BASE_DOWNLOAD_DIR = "/gpfs/data/huo-lab/Image/ChiMEC/"

# MRI1.0 dataset configuration
MRI1_STUDY_IDS_FILE = (
    "/gpfs/data/karczmar-lab/CAPS/MRI1.0/MRI1.0_AnonymousIDs_hiro.xlsx"
)
MRI1_METADATA_FILE = "data/imaging_metadata_mri1.csv"
MRI1_EXPORT_STATE_FILE = "data/export_state_mri1.csv"
MRI1_BASE_DOWNLOAD_DIR = "/gpfs/data/karczmar-lab/CAPS/MRI1.0/"

# Legacy constants for backward compatibility (default to ChiMEC)
KEY_FILE = CHIMEC_KEY_FILE
METADATA_FILE = CHIMEC_METADATA_FILE
EXPORT_STATE_FILE = CHIMEC_EXPORT_STATE_FILE
BASE_DOWNLOAD_DIR = CHIMEC_BASE_DOWNLOAD_DIR

MERGE_KEY_COLUMNS = ["study_id", "Study DateTime", "StudyDescription"]
EXPORT_STATE_COLUMNS = MERGE_KEY_COLUMNS + [
    "is_exported",
    "download_attempt_outcome",
    "export_requested_on",
]

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


def analyze_export_timeline(missing_exams_df: pd.DataFrame) -> None:
    """Analyze timeline patterns for exported-but-not-on-disk exams."""

    print("\n--- Timeline Analysis for Exported-but-Not-on-Disk Exams ---")

    current_time = pd.Timestamp.now()
    bins = [0, 1, 3, 7, 14, 30, 90, float("inf")]

    # Analyze export_requested_on timeline
    if "export_requested_on" in missing_exams_df.columns:
        requested_times = pd.to_datetime(
            missing_exams_df["export_requested_on"], errors="coerce"
        )
        valid_requested = requested_times.dropna()

        if not valid_requested.empty:
            time_since_request = current_time - valid_requested
            days_since = time_since_request.dt.total_seconds() / (24 * 3600)

            print(f"\nTime since export request (n={len(valid_requested)}):")
            print("  (When THIS script requested exports via web interface)")

            for _i, (bin_min, bin_max) in enumerate(zip(bins[:-1], bins[1:])):
                if bin_max == float("inf"):
                    count = (days_since >= bin_min).sum()
                    label = f">{bin_min} days"
                else:
                    count = ((days_since >= bin_min) & (days_since < bin_max)).sum()
                    label = f"{bin_min}-{bin_max} days"
                print(f"  {label}: {count} exams")

    # Analyze Exported On timeline
    if "Exported On" in missing_exams_df.columns:
        exported_times = pd.to_datetime(
            missing_exams_df["Exported On"], errors="coerce"
        )
        valid_exported = exported_times.dropna()

        if not valid_exported.empty:
            time_since_export = current_time - valid_exported
            days_since = time_since_export.dt.total_seconds() / (24 * 3600)

            print(f"\nTime since export completion (n={len(valid_exported)}):")
            print(
                "  (When exams were marked as exported in the system, from any source)"
            )
            for _i, (bin_min, bin_max) in enumerate(zip(bins[:-1], bins[1:])):
                if bin_max == float("inf"):
                    count = (days_since >= bin_min).sum()
                    label = f">{bin_min} days"
                else:
                    count = ((days_since >= bin_min) & (days_since < bin_max)).sum()
                    label = f"{bin_min}-{bin_max} days"
                print(f"  {label}: {count} exams")


def investigate_file_system(
    missing_exams_df: pd.DataFrame, base_download_dir: str
) -> pd.DataFrame:
    """Investigate file system for missing exams and return enhanced dataframe."""

    print("\n--- File System Investigation ---")

    enhanced_df = missing_exams_df.copy()

    # Add expected file paths
    enhanced_df["expected_base_path"] = enhanced_df["study_id"].apply(
        lambda x: Path(base_download_dir) / str(int(x)) if pd.notna(x) else None
    )

    # Check what actually exists
    path_exists = []
    files_found_at_root = []
    total_files_recursive = []
    subdirs_found = []
    dir_contents = []
    accession_found_on_disk = []

    for _idx, row in enhanced_df.iterrows():
        base_path = row["expected_base_path"]
        expected_accession = (
            str(row.get("Accession", "")).strip()
            if pd.notna(row.get("Accession"))
            else None
        )

        if base_path and base_path.exists():
            path_exists.append(True)
            try:
                contents = list(base_path.iterdir())
                files_only = [f for f in contents if f.is_file()]
                dirs_only = [f for f in contents if f.is_dir()]
                files_found_at_root.append(len(files_only))
                subdirs_found.append(len(dirs_only))

                # Count total files recursively
                total_recursive = sum(1 for _ in base_path.rglob("*") if _.is_file())
                total_files_recursive.append(total_recursive)

                # Check if expected accession matches any subdirectory
                found_accession = False
                if expected_accession and dirs_only:
                    for d in dirs_only:
                        # Extract accession from directory name (before first dash)
                        dir_accession = d.name.split("-")[0]
                        if dir_accession == expected_accession:
                            found_accession = True
                            break
                accession_found_on_disk.append(found_accession)

                # Show both files and directories
                file_names = [f.name for f in files_only[:3]]
                dir_names = [f.name for f in dirs_only[:3]]

                content_parts = []
                if file_names:
                    content_parts.append(f"FILES: {', '.join(file_names)}")
                if dir_names:
                    content_parts.append(f"DIRS: {', '.join(dir_names)}")
                if not content_parts:
                    content_parts.append("EMPTY")

                dir_contents.append("; ".join(content_parts))
            except Exception as e:
                files_found_at_root.append(0)
                total_files_recursive.append(0)
                subdirs_found.append(0)
                accession_found_on_disk.append(False)
                dir_contents.append(f"Error: {e}")
        else:
            path_exists.append(False)
            files_found_at_root.append(0)
            total_files_recursive.append(0)
            subdirs_found.append(0)
            accession_found_on_disk.append(False)
            dir_contents.append("Path does not exist")

    enhanced_df["path_exists"] = path_exists
    enhanced_df["files_at_root"] = files_found_at_root
    enhanced_df["total_files_recursive"] = total_files_recursive
    enhanced_df["subdirs_count"] = subdirs_found
    enhanced_df["accession_dir_exists"] = accession_found_on_disk
    enhanced_df["directory_contents"] = dir_contents

    # Summary statistics
    existing_paths = sum(path_exists)
    total_files_at_root = sum(files_found_at_root)
    total_files_all = sum(total_files_recursive)

    has_accession = enhanced_df["Accession"].notna()
    accession_match = enhanced_df["accession_dir_exists"]
    has_subdirs = enhanced_df["subdirs_count"] > 0
    path_exists_mask = enhanced_df["path_exists"]

    print(
        f"Study directories that exist on disk: {existing_paths}/{len(missing_exams_df)} ({existing_paths / len(missing_exams_df) * 100:.1f}%)"
    )
    print(f"  └─ Total files recursively: {total_files_all:,}")
    print(f"  └─ Files at root level only: {total_files_at_root}")

    print("\nWhy are these 'missing'?")
    print(
        f"  └─ Missing accession number in DB: {(~has_accession & path_exists_mask).sum()}"
    )
    print(
        f"  └─ Accession in DB but directory doesn't match: {(has_accession & ~accession_match & has_subdirs).sum()}"
    )
    print(
        f"  └─ Directory exists but is empty: {(path_exists_mask & (enhanced_df['total_files_recursive'] == 0)).sum()}"
    )
    print(f"  └─ Accession directory found on disk: {accession_match.sum()}")

    # Show examples of accession mismatches
    mismatch_mask = has_accession & ~accession_match & has_subdirs & path_exists_mask
    if mismatch_mask.sum() > 0:
        print(f"\nAccession MISMATCH examples ({mismatch_mask.sum()} total):")
        print("  DB has accession, but no matching subdirectory found")
        for _idx, row in enhanced_df[mismatch_mask].head(3).iterrows():
            study_id = row.get("study_id", "unknown")
            expected = row.get("Accession", "N/A")
            contents = row.get("directory_contents", "unknown")
            total = row.get("total_files_recursive", 0)
            print(
                f"  {study_id}: expected accession={expected}, found {contents}, {total} files total"
            )
        if mismatch_mask.sum() > 3:
            print(f"  ... and {mismatch_mask.sum() - 3} more")

    # Show examples where accession matches
    if accession_match.sum() > 0:
        print(
            f"\nAccession MATCHES but still marked missing ({accession_match.sum()} total):"
        )
        print("  This suggests disk status check is working correctly")
        print(
            "  but these exams might have OTHER issues (wrong modality, wrong study, etc.)"
        )
        for _idx, row in enhanced_df[accession_match].head(3).iterrows():
            study_id = row.get("study_id", "unknown")
            accession = row.get("Accession", "N/A")
            total = row.get("total_files_recursive", 0)
            print(f"  {study_id}: accession={accession}, {total} files found")
        if accession_match.sum() > 3:
            print(f"  ... and {accession_match.sum() - 3} more")

    # Show examples of missing accession in DB
    missing_acc_mask = ~has_accession & path_exists_mask
    if missing_acc_mask.sum() > 0:
        print(f"\nMissing accession in DB ({missing_acc_mask.sum()} total):")
        print("  These cannot be verified because DB has no accession number")
        for _idx, row in enhanced_df[missing_acc_mask].head(3).iterrows():
            study_id = row.get("study_id", "unknown")
            contents = row.get("directory_contents", "unknown")
            total = row.get("total_files_recursive", 0)
            print(f"  {study_id}: {contents}, {total} files total")
        if missing_acc_mask.sum() > 3:
            print(f"  ... and {missing_acc_mask.sum() - 3} more")

    return enhanced_df


def save_missing_exams_debug_csv(
    missing_exams_df: pd.DataFrame, base_download_dir: str, dataset: str = "chimec"
) -> None:
    """Save comprehensive debug information for missing exams to CSV.

    Parameters
    ----------
    missing_exams_df : pd.DataFrame
        DataFrame with missing exams to debug
    base_download_dir : str
        Base download directory for the dataset
    dataset : str
        Dataset name: "chimec" or "mri1.0" (used for filename)
    """

    print("\n--- Saving Missing Exams Debug CSV ---")

    # Create enhanced dataframe with file system investigation
    enhanced_df = investigate_file_system(missing_exams_df, base_download_dir)

    # Add timeline information
    current_time = pd.Timestamp.now()

    if "export_requested_on" in enhanced_df.columns:
        requested_times = pd.to_datetime(
            enhanced_df["export_requested_on"], errors="coerce"
        )
        enhanced_df["days_since_export_request"] = (
            current_time - requested_times
        ).dt.total_seconds() / (24 * 3600)

    if "Exported On" in enhanced_df.columns:
        exported_times = pd.to_datetime(enhanced_df["Exported On"], errors="coerce")
        enhanced_df["days_since_export_completion"] = (
            current_time - exported_times
        ).dt.total_seconds() / (24 * 3600)

    # Remove MRN and other sensitive columns
    columns_to_exclude = ["MRN", "AnonymousID"]
    debug_columns = [
        col for col in enhanced_df.columns if col not in columns_to_exclude
    ]
    debug_df = enhanced_df[debug_columns].copy()

    # Save to CSV with dataset-specific filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_suffix = "_mri1" if dataset == "mri1.0" else ""
    debug_filename = f"data/missing_exams_debug{dataset_suffix}_{timestamp}.csv"
    Path("data").mkdir(exist_ok=True)

    debug_df.to_csv(debug_filename, index=False)
    print(f"Debug information saved to: {debug_filename}")
    print(f"Columns included: {len(debug_columns)}")
    print(f"Exams analyzed: {len(debug_df)}")


def print_export_history_summary(missing_exams_df: pd.DataFrame) -> None:
    """Print summary of export attempt history for missing exams."""

    print("\n--- Export History Analysis ---")

    if "download_attempt_outcome" in missing_exams_df.columns:
        outcome_counts = missing_exams_df["download_attempt_outcome"].value_counts(
            dropna=False
        )
        print("Download attempt outcomes:")
        for outcome, count in outcome_counts.items():
            if pd.isna(outcome):
                print(f"  <NA>: {count}")
            elif outcome == "Already Exported":
                print(f"  {outcome}: {count} (discovered during main export loop)")
            elif outcome == "Audit: Already Exported":
                print(f"  {outcome}: {count} (discovered during audit process)")
            else:
                print(f"  {outcome}: {count}")

    if "Status" in missing_exams_df.columns:
        status_counts = missing_exams_df["Status"].value_counts(dropna=False)
        print("\nStatus distribution:")
        for status, count in status_counts.items():
            print(f"  {status}: {count}")

    # Check for patterns in accession numbers
    if "Accession" in missing_exams_df.columns:
        has_accession = missing_exams_df["Accession"].notna().sum()
        print(
            f"\nExams with Accession numbers: {has_accession}/{len(missing_exams_df)} ({has_accession / len(missing_exams_df) * 100:.1f}%)"
        )


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


def save_current_state(db: pd.DataFrame, dataset: str = "chimec"):
    """Persist export state to a separate file (not the scraper's metadata file).

    Writes only the merge keys and export-specific columns to dataset-specific export state file.
    This avoids conflicts with scrape_ibroker.py which owns METADATA_FILE.

    Parameters
    ----------
    db : pd.DataFrame
        Database with export state
    dataset : str
        Dataset name: "chimec" or "mri1.0"
    """
    export_state_file = (
        MRI1_EXPORT_STATE_FILE if dataset == "mri1.0" else CHIMEC_EXPORT_STATE_FILE
    )

    available_cols = [c for c in EXPORT_STATE_COLUMNS if c in db.columns]
    export_state = db[available_cols].copy()

    # only keep rows that have meaningful export state
    has_state = (
        export_state["is_exported"].fillna(False)
        | export_state["download_attempt_outcome"].notna()
        | export_state["export_requested_on"].notna()
    )
    export_state = export_state[has_state]

    key_cols = [c for c in MERGE_KEY_COLUMNS if c in export_state.columns]
    if key_cols:
        export_state = (
            export_state.sort_values(key_cols)
            .drop_duplicates(subset=key_cols, keep="last")
            .reset_index(drop=True)
        )

    _atomic_write_csv(
        export_state,
        export_state_file,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )


def load_and_merge_data(
    dataset: str = "chimec",
    max_exams: int | None = None,
    submit_exports: bool = False,
    batch_size: int | None = None,
    full_db: pd.DataFrame | None = None,
):
    """Load and merge data for the specified dataset.

    Parameters
    ----------
    dataset : str
        Dataset name: "chimec" or "mri1.0"
    max_exams : int | None
        For MRI1.0: stop querying once we have this many MR+BREAST exams.
        None means query all study IDs.
    submit_exports : bool
        For MRI1.0: if True, submit export requests during query phase.
    batch_size : int | None
        For MRI1.0: max export requests to submit (only if submit_exports=True).
    full_db : pd.DataFrame | None
        For MRI1.0: database to update with export outcomes (only if submit_exports=True).

    Returns
    -------
    pd.DataFrame | tuple[pd.DataFrame, dict]
        For ChiMEC: Merged database with all exam records.
        For MRI1.0: (Merged database, export_stats dict) if submit_exports=True,
        else just the merged database.
    """
    print(f"--- Phase 1: Loading and Merging Data (dataset: {dataset}) ---")

    if dataset == "mri1.0":
        result = _load_mri1_data(
            max_exams=max_exams,
            submit_exports=submit_exports,
            batch_size=batch_size,
            full_db=full_db,
        )
        # Always returns tuple, but for backward compatibility, unpack if not submitting exports
        if submit_exports:
            return result  # (db, export_stats)
        else:
            return result[0]  # Just db
    else:
        return _load_chimec_data()


def _load_chimec_data():
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
        if Path(CHIMEC_EXPORT_STATE_FILE).exists():
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
                    if col == "is_exported":
                        mask = mask | metadata[state_col].fillna(False).astype(bool)
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


def _load_mri1_data(
    max_exams: int | None = None,
    submit_exports: bool = False,
    batch_size: int | None = None,
    full_db: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Load MRI1.0 dataset by querying iBroker directly for MR exams with BREAST in description.

    Parameters
    ----------
    max_exams : int | None
        Stop querying once we have this many MR+BREAST exams ready to export
        (not on disk, not already exported). None means query all study IDs.
    submit_exports : bool
        If True, submit export requests as we find ready-to-export exams using Selenium.
        Requires driver to remain open.
    batch_size : int | None
        Maximum number of export requests to submit (only used if submit_exports=True).
    full_db : pd.DataFrame | None
        Database to update with export outcomes (only used if submit_exports=True).

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Metadata DataFrame and export statistics dict with keys:
        - 'submitted': number of export requests submitted
        - 'already_exported': number discovered to be already exported
        - 'processed': total exams processed for export
    """
    # Import scrape functions (requires credentials)
    _import_scrape_ibroker()

    # Load fingerprint cache for disk status checking (MRI1.0 specific only)
    # Don't use ChiMEC cache - patient IDs don't match between datasets
    import json

    mri1_fingerprint_cache = Path("data/destination_fingerprints_mri1.json")
    disk_exam_counts = {}

    if mri1_fingerprint_cache.exists():
        print(f"Loading MRI1.0 disk fingerprint cache: {mri1_fingerprint_cache}")
        # Load raw fingerprints for counting exams per (patient_id, date)
        with open(mri1_fingerprint_cache) as f:
            raw_fingerprints = json.load(f)
        for patient_id, exams in raw_fingerprints.items():
            for exam_name, data in exams.items():
                uid, hashes, study_date, study_time = data
                if study_date and len(study_date) >= 8:
                    date_str = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:8]}"
                    key = (patient_id, date_str)
                    disk_exam_counts[key] = disk_exam_counts.get(key, 0) + 1
        print(
            f"Loaded {len(disk_exam_counts):,} (patient_id, date) pairs from fingerprint cache."
        )
    else:
        print(
            f"Note: No MRI1.0 fingerprint cache found at {mri1_fingerprint_cache}. "
            "Disk status checking will be skipped (all exams assumed not on disk)."
        )

    try:
        # Load study IDs from Excel file
        study_ids_df = pd.read_excel(MRI1_STUDY_IDS_FILE)
        study_ids = study_ids_df["AnonymousID"].astype(str).tolist()
        print(f"Loaded {len(study_ids):,} study IDs from MRI1.0 file.")

        # Query iBroker directly for each study ID
        if submit_exports:
            print("Querying iBroker and submitting export requests for MR exams...")
        else:
            print("Querying iBroker for MR exams...")
        driver = None
        session = None
        export_stats = {"submitted": 0, "already_exported": 0, "processed": 0}
        try:
            print("Starting browser driver...")
            driver = make_driver()
            print("Logging into iBroker...")
            try:
                login(driver, USERNAME, PASSWORD)
            except Exception as login_error:
                # Save page HTML for debugging
                try:
                    page_source = driver.page_source
                    Path("data").mkdir(exist_ok=True)
                    debug_file = Path("data/debug_login_failure.html")
                    with open(debug_file, "w", encoding="utf-8") as f:
                        f.write(page_source)
                    print(
                        f"Saved page HTML to {debug_file} for debugging",
                        file=sys.stderr,
                    )
                    # Check what's actually on the page
                    if "tbxUsername" in page_source or "tbxPassword" in page_source:
                        print(
                            "Page still shows login form - login may have failed",
                            file=sys.stderr,
                        )
                    elif "lbUser" in page_source:
                        print(
                            "Page contains 'lbUser' element but Selenium couldn't find it - timing issue?",
                            file=sys.stderr,
                        )
                except Exception as debug_error:
                    print(f"Could not save debug HTML: {debug_error}", file=sys.stderr)
                raise login_error
            print("Login successful. Bootstrapping HTTP session...")
            session = bootstrap_http_session_from_driver(driver)
            print("HTTP session created.")
        except Exception as e:
            print(f"ERROR: Failed to login to iBroker: {e}", file=sys.stderr)
            print("This may be due to:", file=sys.stderr)
            print("  - Network/VPN connectivity issues", file=sys.stderr)
            print("  - Invalid credentials", file=sys.stderr)
            print("  - iBroker server issues", file=sys.stderr)
            print(
                "  - Check data/debug_login_failure.html if it was created",
                file=sys.stderr,
            )
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass
            raise

        if session is None:
            raise RuntimeError("Failed to create HTTP session - cannot query iBroker")

        # Initialize HTTP session state
        print("Initializing iBroker session state...")
        try:
            page_html, state = http_get_root(session)
            page_html, state = post_link_event(session, state, "lbAll")
        except Exception as e:
            print(f"ERROR: Failed to initialize iBroker session: {e}", file=sys.stderr)
            raise

        # Initialize export state columns if submitting exports
        if submit_exports:
            # These will be added to each df as we process them
            pass

        # Query study IDs incrementally and stop when we have enough ready-to-export exams
        # This avoids querying all study IDs when we only need a small batch
        all_exams = []
        ready_to_export_count = 0

        query_desc = "Querying iBroker"
        if submit_exports:
            query_desc = "Querying iBroker and submitting exports"
        if max_exams is not None:
            query_desc = f"{query_desc} (target: {max_exams} ready-to-export exams)"
            # Create progress bar with target as total
            pbar = tqdm(total=max_exams, desc=query_desc, unit="exam")
        else:
            # No target, show progress through study IDs
            pbar = tqdm(study_ids, desc=query_desc)

        def check_exam_on_disk(study_id: str, study_datetime: pd.Timestamp) -> bool:
            """Check if an exam is on disk using fingerprint cache."""
            try:
                patient_id = str(int(study_id))
                if pd.isna(study_datetime):
                    return False
                date_str = study_datetime.strftime("%Y-%m-%d")
                key = (patient_id, date_str)
                return key in disk_exam_counts
            except (ValueError, TypeError):
                return False

        def submit_exports_for_patient(
            driver, study_id: str, ready_exams_df: pd.DataFrame
        ) -> tuple[int, int, list]:
            """Submit export requests for ready-to-export exams using Selenium.

            Returns
            -------
            tuple[int, int, list]
                (submitted_count, already_exported_count, submitted_exam_keys)
                submitted_exam_keys is a list of (Study DateTime date, StudyDescription) tuples
            """
            submitted = 0
            already_exported = 0
            submitted_keys = []

            try:
                # Navigate to patient's page
                driver.find_element(by="name", value="tbxAssignedID").clear()
                driver.find_element(by="name", value="tbxAssignedID").send_keys(
                    str(int(study_id))
                )
                driver.find_element(by="name", value="btnFetch").click()
                wait_aspnet_idle(driver)

                # Get available exams on the page
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

                # Match ready exams with available exams on page
                requested_any = False
                for _, exam_row in ready_exams_df.iterrows():
                    if (
                        batch_size is not None
                        and export_stats["submitted"] >= batch_size
                    ):
                        break

                    target_key = (
                        exam_row["Study DateTime"].date(),
                        exam_row["StudyDescription"],
                    )

                    if target_key in available_on_page:
                        print(
                            f"  - Found available exam from {target_key[0]}. Selecting checkbox."
                        )
                        available_on_page[target_key].click()
                        submitted += 1
                        export_stats["submitted"] += 1
                        export_stats["processed"] += 1
                        requested_any = True
                        submitted_keys.append(target_key)
                    else:
                        print(
                            f"  - INFO: Exam from {target_key[0]} is no longer available (already exported)."
                        )
                        already_exported += 1
                        export_stats["already_exported"] += 1
                        export_stats["processed"] += 1

                # Submit export request if any exams were selected
                if requested_any:
                    print(f"  Submitting export request for {submitted} exam(s)...")
                    driver.find_element(by="name", value="btnExport").click()
                    wait_aspnet_idle(driver)
                    print(f"  ✓ Export request submitted for {submitted} exam(s).")

            except Exception as e:
                print(f"ERROR submitting exports for study_id {study_id}: {e}")
                # Continue processing other patients

            return submitted, already_exported, submitted_keys

        for study_id in study_ids:
            # Check if we've reached batch limit
            if (
                submit_exports
                and batch_size is not None
                and export_stats["submitted"] >= batch_size
            ):
                pbar.close()
                print(
                    f"\nReached batch limit of {batch_size} export requests. Stopping."
                )
                break

            try:
                page_html, state = post_fetch_grid(session, state, str(study_id))
                df = parse_all_tables_from_page(page_html)

                if not df.empty:
                    df["study_id"] = study_id
                    # Filter for MR+BREAST immediately
                    df["base_modality"] = df["Modality"].apply(get_base_modality)
                    df = df[df["base_modality"] == "MR"]

                    if "StudyDescription" in df.columns:
                        breast_mask = df["StudyDescription"].str.contains(
                            "BREAST", case=False, na=False
                        )
                        df = df[breast_mask]

                    if not df.empty:
                        # Parse Study DateTime
                        df["Study DateTime"] = pd.to_datetime(
                            df["Study DateTime"], errors="coerce"
                        )

                        # Initialize export state columns if submitting exports
                        if submit_exports:
                            if "download_attempt_outcome" not in df.columns:
                                df["download_attempt_outcome"] = pd.NA
                            df["download_attempt_outcome"] = df[
                                "download_attempt_outcome"
                            ].astype("string")
                            if "export_requested_on" not in df.columns:
                                df["export_requested_on"] = pd.NaT
                            if "Status" not in df.columns:
                                df["Status"] = pd.NA

                        # Check disk status and export status incrementally
                        df["is_on_disk"] = df.apply(
                            lambda row: check_exam_on_disk(
                                row["study_id"], row["Study DateTime"]
                            ),
                            axis=1,
                        )

                        # Check if already exported (has Accession or Status indicates exported)
                        if "Accession" in df.columns:
                            has_accession = df["Accession"].notna()
                        else:
                            has_accession = pd.Series([False] * len(df), index=df.index)

                        if "Status" in df.columns:
                            status_exported = (
                                df["Status"]
                                .fillna("")
                                .astype(str)
                                .str.strip()
                                .str.lower()
                                .isin({"already exported", "exported"})
                            )
                        else:
                            status_exported = pd.Series(
                                [False] * len(df), index=df.index
                            )

                        df["is_exported"] = has_accession | status_exported

                        # Count ready-to-export exams (not on disk, not exported)
                        ready_mask = (~df["is_on_disk"]) & (~df["is_exported"])
                        ready_count = ready_mask.sum()
                        ready_to_export_count += ready_count

                        # Submit exports if requested
                        if submit_exports and ready_count > 0:
                            ready_exams = df[ready_mask]
                            _, _, submitted_keys = submit_exports_for_patient(
                                driver, study_id, ready_exams
                            )
                            # Update df with export outcomes
                            submission_ts = pd.Timestamp.now()
                            for target_key in submitted_keys:
                                mask = (
                                    df["Study DateTime"].dt.date == target_key[0]
                                ) & (df["StudyDescription"] == target_key[1])
                                if mask.any():
                                    df.loc[mask, "download_attempt_outcome"] = (
                                        "Request Submitted"
                                    )
                                    df.loc[mask, "export_requested_on"] = submission_ts
                                    if "Status" in df.columns:
                                        df.loc[mask, "Status"] = "Request Submitted"

                        all_exams.append(df)

                        # Update progress bar
                        if max_exams is not None:
                            # Update to show current count (capped at total)
                            pbar.n = min(ready_to_export_count, max_exams)
                            pbar.refresh()

                        if max_exams is not None and ready_to_export_count >= max_exams:
                            pbar.close()
                            print(
                                f"\nReached target of {max_exams} ready-to-export exams "
                                f"(found {ready_to_export_count} total). "
                                f"Stopped querying after processing {len(all_exams)} study IDs."
                            )
                            break
            except Exception as e:
                print(f"Warning: Failed to query study_id {study_id}: {e}")
                continue

        # Close progress bar if still open
        if max_exams is not None:
            pbar.close()

        # Close driver if we're not submitting exports (or if we're done submitting)
        if not submit_exports and driver:
            try:
                print("Closing browser driver...")
                driver.quit()
            except Exception:
                pass

        if not all_exams:
            metadata = pd.DataFrame(
                columns=[
                    "Modality",
                    "Study DateTime",
                    "StudyDescription",
                    "Status",
                    "Accession",
                    "Exported On",
                    "study_id",
                ]
            )
        else:
            metadata = pd.concat(all_exams, ignore_index=True)

        print(f"Retrieved {len(metadata):,} total exam records from iBroker.")

        # Filter for MR modality
        metadata["base_modality"] = metadata["Modality"].apply(get_base_modality)
        metadata = metadata[metadata["base_modality"] == "MR"]
        print(f"Filtered to {len(metadata):,} MR exams.")

        # Filter for BREAST in study description
        if "StudyDescription" in metadata.columns:
            breast_mask = metadata["StudyDescription"].str.contains(
                "BREAST", case=False, na=False
            )
            metadata = metadata[breast_mask]
            print(
                f"Filtered to {len(metadata):,} MR exams with 'BREAST' in description."
            )

        # Deduplicate by (study_id, Study DateTime, StudyDescription) to avoid counting timepoints as separate exams
        # This matches the MERGE_KEY_COLUMNS used elsewhere
        before_dedup = len(metadata)
        metadata = metadata.sort_values(
            MERGE_KEY_COLUMNS, ascending=[True, True, True]
        ).drop_duplicates(subset=MERGE_KEY_COLUMNS, keep="first")
        if len(metadata) != before_dedup:
            print(
                f"Deduplicated exam rows on {MERGE_KEY_COLUMNS}: "
                f"{before_dedup:,} → {len(metadata):,}"
            )

        # A small fix: The Accession number is often what we care about, not the old exam_id
        if "exam_id" in metadata.columns and "Accession" in metadata.columns:
            metadata["Accession"] = metadata["Accession"].fillna(metadata["exam_id"])

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
        if Path(MRI1_EXPORT_STATE_FILE).exists():
            export_state = pd.read_csv(MRI1_EXPORT_STATE_FILE)
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
                    if col == "is_exported":
                        mask = mask | metadata[state_col].fillna(False).astype(bool)
                    metadata.loc[mask, col] = metadata.loc[mask, state_col]
                    metadata.drop(columns=[state_col], inplace=True)
    except FileNotFoundError as e:
        print(f"ERROR: Input file not found - {e}", file=sys.stderr)
        sys.exit(1)

    db = metadata.copy()
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

    # Close driver if we were submitting exports
    if submit_exports and driver:
        try:
            print("Closing browser driver after export submission...")
            driver.quit()
        except Exception:
            pass

    return db, export_stats


def identify_download_targets(
    df: pd.DataFrame, filter_by_genotyping: bool, modality: str, dataset: str = "chimec"
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

    # filter to modality first for summary
    modality_mask = df["base_modality"] == modality
    modality_df = df[modality_mask].copy()

    # comprehensive export status summary
    print(f"\n{'=' * 60}")
    print(f"EXPORT STATUS SUMMARY FOR {modality}")
    print(f"{'=' * 60}")
    total_modality = len(modality_df)
    on_disk = modality_df["is_on_disk"].sum()
    exported_not_on_disk = (
        (modality_df["is_exported"]) & (~modality_df["is_on_disk"])
    ).sum()
    not_exported = ((~modality_df["is_exported"]) & (~modality_df["is_on_disk"])).sum()

    print(f"  Total {modality} exams in iBroker:     {total_modality:>8,}")
    print(f"  Already on disk (done):              {on_disk:>8,}")
    print(f"  Exported but not on disk (sync?):    {exported_not_on_disk:>8,}")
    print(f"  Not yet exported (REMAINING):        {not_exported:>8,}")

    # check phenotype coverage for remaining
    if "chip" in modality_df.columns:
        remaining_mask = (~modality_df["is_on_disk"]) & (~modality_df["is_exported"])
        remaining_df = modality_df[remaining_mask]
        with_genotype = remaining_df["chip"].notna().sum()
        print(f"\n  Of the {not_exported:,} remaining to export:")
        print(f"    - with genotype data: {with_genotype:,}")
        print(f"    - without genotype:   {not_exported - with_genotype:,}")

    print(f"{'=' * 60}\n")

    print(f"Initial pool: {len(df):,} exams")
    df.loc[~modality_mask, "rejection_reason"] = f"Wrong modality (not {modality})"
    targets = modality_df.copy()
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
    print(
        f"  - Rejected {already_exported_mask.sum():,} exams because already exported, but not on disk."
    )
    if not exported_missing_disk.empty:
        print("  - Found exported-but-not-on-disk exams!")
        print(f"  - Total missing exams: {len(exported_missing_disk)}")

        # Run comprehensive debugging analysis
        analyze_export_timeline(exported_missing_disk)
        print_export_history_summary(exported_missing_disk)
        base_download_dir = (
            MRI1_BASE_DOWNLOAD_DIR if dataset == "mri1.0" else CHIMEC_BASE_DOWNLOAD_DIR
        )
        save_missing_exams_debug_csv(
            exported_missing_disk, base_download_dir, dataset=dataset
        )

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
            column
            for column in debug_columns
            if column in exported_missing_disk.columns
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

    # Skip genomics and case/control filtering for MRI1.0 dataset
    if dataset != "mri1.0":
        if filter_by_genotyping:
            mask_no_chip = targets["chip"].isna()
            targets.loc[mask_no_chip, "rejection_reason"] = "No genotyping data"
            targets = targets[~mask_no_chip]
            print(
                f"  - Rejected {mask_no_chip.sum():,} due to missing genotyping data."
            )

        case_mask = targets["case_control_status"] == "Case"
        bad_case_date_mask = case_mask & (
            targets["Study DateTime"] > targets["DatedxIndex"]
        )
        targets.loc[bad_case_date_mask, "rejection_reason"] = (
            "Exam is after diagnosis date"
        )
        targets = targets[~bad_case_date_mask]
        print(
            f"  - Rejected {bad_case_date_mask.sum():,} 'Case' exams that occurred after diagnosis."
        )

    print(f"  = {len(targets):,} final potential target exams identified.")
    return targets.sort_values(by=["study_id", "Study DateTime"])


def _populate_mri1_fingerprint_cache(cache_path: Path, base_dir: str) -> Path | None:
    """Populate MRI1.0 fingerprint cache by running fingerprinter on the MRI1.0 directory.

    Handles both flat structure (base_dir/patient_id/exam_dir/) and
    modality-grouped structure (base_dir/MR/patient_id/exam_dir/).

    Parameters
    ----------
    cache_path : Path
        Path where the fingerprint cache should be created
    base_dir : str
        Base directory for MRI1.0 data (e.g., MRI1_BASE_DOWNLOAD_DIR)

    Returns
    -------
    Path | None
        Path to the created cache file, or None if creation failed
    """
    fingerprinter_script = Path(__file__).parent / "fingerprinter.py"
    log_file = Path("data") / "fingerprinter_mri1.log"

    if not fingerprinter_script.exists():
        print(f"  ERROR: fingerprinter.py not found at {fingerprinter_script}")
        return None

    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"  WARNING: MRI1.0 base directory does not exist: {base_path}")
        print("  Skipping fingerprint cache creation.")
        return None

    # Check if structure is modality-grouped (base_dir/MR/patient_id/) or flat (base_dir/patient_id/)
    # Fingerprinter expects root_dir/patient_id/exam_dir/, so point it at the right level
    fingerprint_root = base_path
    mr_subdir = base_path / "MR"
    if mr_subdir.exists() and mr_subdir.is_dir():
        # Modality-grouped structure: use MR subdirectory
        fingerprint_root = mr_subdir
        print(f"  Detected modality-grouped structure, using: {fingerprint_root}")
    else:
        # Flat structure: use base directory directly
        print(f"  Detected flat structure, using: {fingerprint_root}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Running fingerprinter on {fingerprint_root}")
    print(f"  Output cache: {cache_path}")
    print(f"  Log file: {log_file}")

    cmd = [
        sys.executable,
        "-u",
        str(fingerprinter_script),
        str(fingerprint_root),
        str(cache_path),
        "--parallel-jobs",
        "4",  # Conservative parallel jobs for MRI1.0
    ]

    print(f"  Executing: {' '.join(cmd)}")
    print(f"  (Also logging to: {log_file})")
    try:
        # Stream stdout to both terminal and log file
        with open(log_file, "w") as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
            )

            # Timeout handler
            timeout_occurred = threading.Event()

            def timeout_handler():
                time.sleep(3600)  # 1 hour timeout
                if process.poll() is None:  # Process still running
                    timeout_occurred.set()
                    process.kill()

            timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
            timeout_thread.start()

            # Read and print lines in real-time
            for line in process.stdout:
                if timeout_occurred.is_set():
                    raise subprocess.TimeoutExpired(cmd, 3600)
                print(f"  {line.rstrip()}")
                log_f.write(line)
                log_f.flush()

            process.wait()
            result = type("obj", (object,), {"returncode": process.returncode})()

        if result.returncode != 0:
            print(
                f"  WARNING: Fingerprinter failed with return code {result.returncode}"
            )
            print(f"  Check log file: {log_file}")
            return None

        if not cache_path.exists():
            print(f"  WARNING: Cache file was not created: {cache_path}")
            return None

        size_mb = cache_path.stat().st_size / 1_000_000
        print(f"  Created cache file: {cache_path} ({size_mb:.2f} MB)")
        return cache_path
    except subprocess.TimeoutExpired:
        print("  ERROR: Fingerprinter timed out after 1 hour")
        return None
    except Exception as e:
        print(f"  ERROR: Failed to run fingerprinter: {e}")
        return None


def execute_downloads(
    targets_df: pd.DataFrame,
    batch_size: int,
    full_db: pd.DataFrame = None,
    dataset: str = "chimec",
):
    """Verifies exam availability live and requests exports.

    Exports are chunked - stops after batch_size successful export requests.
    This prevents overwhelming the iBroker system with too many simultaneous requests.
    """
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
                    # Print all metadata for this exam
                    print("    Exam metadata:")
                    for col in full_db.columns if full_db is not None else []:
                        val = target_exam.get(col, "N/A")
                        if pd.notna(val) and str(val).strip():
                            print(f"      {col}: {val}")
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
                print(
                    f"  Submitting export request for {len(requested_indices)} exam(s)..."
                )
                driver.find_element(by="name", value="btnExport").click()
                wait_aspnet_idle(driver)
                print(
                    f"  ✓ Export request submitted for {len(requested_indices)} exam(s)."
                )
                if full_db is not None and requested_indices:
                    submission_ts = pd.Timestamp.now()
                    for idx in requested_indices:
                        full_db.loc[idx, "export_requested_on"] = submission_ts

            # Save state periodically after each patient to preserve progress
            # This includes discoveries of already exported exams
            if full_db is not None:
                save_current_state(full_db, dataset=dataset)
                print("  - State saved to disk.")
    finally:
        if driver:
            print("Closing webdriver session.")
            driver.quit()

    print("\n--- Export Execution Summary ---")
    print(
        f"Successfully submitted export requests: {successfully_exported_counter} exam(s)"
    )
    print(f"Discovered to be already exported:      {already_exported_counter} exam(s)")
    print(f"Total exams processed:                   {len(outcomes)} exam(s)")
    if successfully_exported_counter == 0 and already_exported_counter == 0:
        print(
            "⚠ No exports were requested - all exams may have been unavailable or skipped."
        )

    return outcomes, already_exported_counter, successfully_exported_counter


def run_export_cycle(args, cycle_number: int, dataset: str = "chimec"):
    """Run a single export pass and return summary stats.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    cycle_number : int
        Current cycle number
    dataset : str
        Dataset name: "chimec" or "mri1.0"
    """

    cycle_banner = (
        f"\n=== Export cycle {cycle_number} started at "
        f"{datetime.now():%Y-%m-%d %H:%M:%S} (dataset: {dataset}) ==="
    )
    print(cycle_banner)

    # For MRI1.0, submit exports during query phase to avoid double-pass
    # For ChiMEC, query first then submit exports separately
    max_exams_to_query = None
    export_stats = {}
    if dataset == "mri1.0":
        max_exams_to_query = args.batch_size
        # Submit exports during query phase for MRI1.0
        result = load_and_merge_data(
            dataset=dataset,
            max_exams=max_exams_to_query,
            submit_exports=True,
            batch_size=args.batch_size,
            full_db=None,
        )
        db, export_stats = result
    else:
        db = load_and_merge_data(dataset=dataset, max_exams=max_exams_to_query)

    # conservative=True: only mark is_on_disk if 1:1 match, otherwise re-export to be safe
    # For MRI1.0, use MRI1.0-specific fingerprint cache if it exists, or populate it if missing
    fingerprint_cache = None
    if dataset == "mri1.0":
        mri1_cache = Path("data/destination_fingerprints_mri1.json")
        if mri1_cache.exists():
            fingerprint_cache = mri1_cache
            print(f"Using MRI1.0-specific fingerprint cache: {fingerprint_cache}")
        else:
            print(f"MRI1.0 fingerprint cache not found at {mri1_cache}")
            print("Populating fingerprint cache by scanning MRI1.0 directory...")
            fingerprint_cache = _populate_mri1_fingerprint_cache(
                mri1_cache, MRI1_BASE_DOWNLOAD_DIR
            )
            if fingerprint_cache:
                print(f"✓ Created MRI1.0 fingerprint cache: {fingerprint_cache}")
    db = update_metadata_with_disk_status_by_date(
        db, conservative=True, fingerprint_cache=fingerprint_cache
    )

    # For MRI1.0, skip genotyping filter (it doesn't apply)
    filter_by_genotyping = (
        not args.no_genotyping_filter if dataset != "mri1.0" else False
    )
    targets = identify_download_targets(
        db,
        filter_by_genotyping=filter_by_genotyping,
        modality=args.modality.upper(),
        dataset=dataset,
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

    # For MRI1.0, exports were already submitted during query phase
    if dataset == "mri1.0" and export_stats:
        print("\n--- Exports Already Submitted During Query Phase ---")
        successfully_exported_count = export_stats.get("submitted", 0)
        already_exported_count = export_stats.get("already_exported", 0)
        print(
            f"✓ Successfully submitted export requests: {successfully_exported_count} exam(s)"
        )
        print(f"  Discovered to be already exported: {already_exported_count} exam(s)")
        print(f"  Total exams processed: {export_stats.get('processed', 0)} exam(s)")
    elif targets.empty:
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
                execute_downloads(targets, args.batch_size, full_db=db, dataset=dataset)
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
    save_current_state(db, dataset=dataset)
    export_state_file = (
        MRI1_EXPORT_STATE_FILE if dataset == "mri1.0" else CHIMEC_EXPORT_STATE_FILE
    )
    print(f"\nCycle complete. Export state written to '{export_state_file}'")

    return {
        "submitted": successfully_exported_count,
        "already_exported": already_exported_count,
        "processed": len(outcomes),
        "targets_considered": len(targets),
        "target_indices": targets.index.tolist(),
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

        patients_processed = 0
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
                        full_db.loc[index, "download_attempt_outcome"] = (
                            "Audit: Available"
                        )
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

            patients_processed += 1

            # Save state every 20 patients to preserve audit progress
            if full_db is not None and patients_processed % 20 == 0:
                # Note: dataset parameter would need to be passed here if we want dataset-specific state
                # For now, using default (chimec) since audit_remote_export_status doesn't know dataset
                save_current_state(full_db, dataset="chimec")
                print(
                    f"  - Audit state saved after {patients_processed} patients (patient {study_id})."
                )

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


def refresh_export_status(
    args,
    cycle_number: int,
    *,
    target_indices: list[int] | None = None,
    dataset: str = "chimec",
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
    dataset : str
        Dataset name: "chimec" or "mri1.0"
    """

    print(
        f"\n=== Refresh cycle {cycle_number}: auditing export status before next run (dataset: {dataset}) ==="
    )

    # In refresh mode, query all exams (no limit) to get accurate status
    refresh_db = load_and_merge_data(dataset=dataset, max_exams=None)
    # conservative=True: only mark is_on_disk if 1:1 match, otherwise re-export to be safe
    # For MRI1.0, use MRI1.0-specific fingerprint cache if it exists
    fingerprint_cache = None
    if dataset == "mri1.0":
        mri1_cache = Path("data/destination_fingerprints_mri1.json")
        if mri1_cache.exists():
            fingerprint_cache = mri1_cache
    refresh_db = update_metadata_with_disk_status_by_date(
        refresh_db, conservative=True, fingerprint_cache=fingerprint_cache
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
        subset, full_db=refresh_db, max_exams=max_to_audit
    )

    print(
        "Audit summary: "
        f"checked {audit_stats['audited']} exams — "
        f"marked {audit_stats['marked_exported']} as exported, "
        f"{audit_stats['still_available']} still available."
    )

    save_current_state(refresh_db, dataset=dataset)
    export_state_file = (
        MRI1_EXPORT_STATE_FILE if dataset == "mri1.0" else CHIMEC_EXPORT_STATE_FILE
    )
    print(f"Audit state persisted to '{export_state_file}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Identify and download imaging exams from iBroker."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["chimec", "mri1.0"],
        default="chimec",
        help="Dataset to process: 'chimec' (default) or 'mri1.0'.",
    )
    parser.add_argument(
        "--no-genotyping-filter",
        action="store_true",
        help="Include patients without genotyping data (ChiMEC only).",
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
        type=_parse_wait_interval,
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

    dataset = args.dataset.lower()

    # MRI1.0 dataset requires credentials even for status-only mode (needs to query iBroker)
    if dataset == "mri1.0" and not all([USERNAME, PASSWORD]):
        print(
            "ERROR: IBROKER_USERNAME and IBROKER_PASSWORD must be set for MRI1.0 dataset.",
            file=sys.stderr,
        )
        sys.exit(1)

    # status-only mode doesn't need credentials for ChiMEC (uses existing metadata file)
    if args.status_only:
        print(
            f"Running in --status-only mode (no exports will be performed, dataset: {dataset})\n"
        )
        # In status-only mode, query all exams (no limit)
        db = load_and_merge_data(dataset=dataset, max_exams=None)
        db = update_metadata_with_disk_status_by_date(db, conservative=True)
        filter_by_genotyping = (
            not args.no_genotyping_filter if dataset != "mri1.0" else False
        )
        identify_download_targets(
            db,
            filter_by_genotyping=filter_by_genotyping,
            modality=args.modality.upper(),
            dataset=dataset,
        )
        sys.exit(0)

    if not all([USERNAME, PASSWORD]):
        print(
            "ERROR: IBROKER_USERNAME and IBROKER_PASSWORD must be set.", file=sys.stderr
        )
        sys.exit(1)

    # import scrape_ibroker now that we know we need it (triggers credential check)
    _import_scrape_ibroker()

    cycles_run = 0
    last_target_indices: list[int] | None = None
    try:
        while True:
            cycles_run += 1
            cycle_result = run_export_cycle(args, cycles_run, dataset=dataset)
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
                        dataset=dataset,
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
