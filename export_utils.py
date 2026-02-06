#!/usr/bin/env python
"""Shared utilities for export scripts."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

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


def import_scrape_ibroker():
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


# Shared constants
MERGE_KEY_COLUMNS = ["study_id", "Study DateTime", "StudyDescription"]
EXPORT_STATE_COLUMNS = MERGE_KEY_COLUMNS + [
    "is_exported",
    "download_attempt_outcome",
    "export_requested_on",
]

USERNAME = os.getenv("IBROKER_USERNAME")
PASSWORD = os.getenv("IBROKER_PASSWORD")


def parse_wait_interval(value: str) -> float:
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
    """Extract base modality from modality string."""
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


def atomic_write_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
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


def save_current_state(
    db: pd.DataFrame, export_state_file: Path, merge_key_columns: list[str]
) -> None:
    """Persist export state to a separate file.

    Writes only the merge keys and export-specific columns to export state file.

    Parameters
    ----------
    db : pd.DataFrame
        Database with export state
    export_state_file : Path
        Path to export state file
    merge_key_columns : list[str]
        Columns to use for merging (e.g., MERGE_KEY_COLUMNS)
    """
    export_state_columns = merge_key_columns + [
        "is_exported",
        "download_attempt_outcome",
        "export_requested_on",
    ]

    available_cols = [c for c in export_state_columns if c in db.columns]
    export_state = db[available_cols].copy()

    # only keep rows that have meaningful export state
    has_state = (
        export_state["is_exported"].fillna(False)
        | export_state["download_attempt_outcome"].notna()
        | export_state["export_requested_on"].notna()
    )
    export_state = export_state[has_state]

    key_cols = [c for c in merge_key_columns if c in export_state.columns]
    if key_cols:
        export_state = (
            export_state.sort_values(key_cols)
            .drop_duplicates(subset=key_cols, keep="last")
            .reset_index(drop=True)
        )

    atomic_write_csv(
        export_state,
        export_state_file,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )


def identify_download_targets(
    df: pd.DataFrame,
    filter_by_genotyping: bool,
    modality: str,
    base_download_dir: str,
    dataset: str = "chimec",
    verbose: bool = True,
):
    """Identify target exams for download.

    Parameters
    ----------
    df : pd.DataFrame
        Database with exam records
    filter_by_genotyping : bool
        Whether to filter by genotyping data (ChiMEC only)
    modality : str
        Base modality to filter for
    base_download_dir : str
        Base download directory for the dataset
    dataset : str
        Dataset name: "chimec" or "mri1.0"
    verbose : bool
        Whether to print detailed summary (default: True)

    Returns
    -------
    pd.DataFrame
        Filtered targets ready for export
    """
    if verbose:
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
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"EXPORT STATUS SUMMARY FOR {modality}")
        print(f"{'=' * 60}")
    total_modality = len(modality_df)
    on_disk = modality_df["is_on_disk"].sum()
    exported_not_on_disk = (
        (modality_df["is_exported"]) & (~modality_df["is_on_disk"])
    ).sum()
    not_exported = ((~modality_df["is_exported"]) & (~modality_df["is_on_disk"])).sum()

    if verbose:
        print(f"  Total {modality} exams in iBroker:     {total_modality:>8,}")
        print(f"  Already on disk (done):              {on_disk:>8,}")
        print(f"  Exported but not on disk (sync?):    {exported_not_on_disk:>8,}")
        print(f"  Not yet exported (REMAINING):        {not_exported:>8,}")

        # check phenotype coverage for remaining
        if "chip" in modality_df.columns:
            remaining_mask = (~modality_df["is_on_disk"]) & (
                ~modality_df["is_exported"]
            )
            remaining_df = modality_df[remaining_mask]
            with_genotype = remaining_df["chip"].notna().sum()
            print(f"\n  Of the {not_exported:,} remaining to export:")
            print(f"    - with genotype data: {with_genotype:,}")
            print(f"    - without genotype:   {not_exported - with_genotype:,}")

        print(f"{'=' * 60}\n")

    if verbose:
        print(f"Initial pool: {len(df):,} exams")
    df.loc[~modality_mask, "rejection_reason"] = f"Wrong modality (not {modality})"
    targets = modality_df.copy()
    if verbose:
        print(
            f"  - Kept {len(targets):,} exams after filtering for modality '{modality}'."
        )

    mask_on_disk = targets["is_on_disk"]
    targets.loc[mask_on_disk, "rejection_reason"] = "Already on disk"
    targets = targets[~mask_on_disk]
    if verbose:
        print(f"  - Rejected {mask_on_disk.sum():,} because they are already on disk.")

    # We still check if it's exported, but now it's a secondary check.
    # An exam could be exported but the sync failed, so it's not on disk.
    # This includes exams with Accession numbers AND exams discovered as already exported in previous runs
    already_exported_mask = targets["is_exported"]
    exported_missing_disk = targets[already_exported_mask]
    if verbose:
        print(
            f"  - Rejected {already_exported_mask.sum():,} exams because already exported, but not on disk."
        )
    if not exported_missing_disk.empty and verbose:
        print("  - Found exported-but-not-on-disk exams!")
        print(f"  - Total missing exams: {len(exported_missing_disk)}")

        # Run comprehensive debugging analysis
        analyze_export_timeline(exported_missing_disk)
        print_export_history_summary(exported_missing_disk)
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
            if verbose:
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
        if verbose:
            print(
                f"  - Rejected {bad_case_date_mask.sum():,} 'Case' exams that occurred after diagnosis."
            )

    if verbose:
        print(f"  = {len(targets):,} final potential target exams identified.")
    return targets.sort_values(by=["study_id", "Study DateTime"])


def execute_downloads(
    targets_df: pd.DataFrame,
    batch_size: int,
    full_db: pd.DataFrame | None = None,
    export_state_file: Path | None = None,
    merge_key_columns: list[str] | None = None,
) -> tuple[dict, int, int]:
    """Verifies exam availability live and requests exports.

    Exports are chunked - stops after batch_size successful export requests.
    This prevents overwhelming the iBroker system with too many simultaneous requests.

    Parameters
    ----------
    targets_df : pd.DataFrame
        Target exams to export
    batch_size : int
        Maximum number of export requests to submit
    full_db : pd.DataFrame | None
        Full database to update with export outcomes
    export_state_file : Path | None
        Path to export state file for saving progress
    merge_key_columns : list[str] | None
        Columns to use for merging (e.g., MERGE_KEY_COLUMNS)

    Returns
    -------
    tuple[dict, int, int]
        (outcomes dict, already_exported_count, successfully_exported_count)
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
        try:
            login(driver, USERNAME, PASSWORD)
        except Exception as login_error:
            # save page HTML for debugging
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
                # check what's actually on the page
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
            if full_db is not None and export_state_file and merge_key_columns:
                save_current_state(full_db, export_state_file, merge_key_columns)
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


def audit_remote_export_status(
    audit_df: pd.DataFrame,
    full_db: pd.DataFrame | None = None,
    max_exams: int | None = None,
    export_state_file: Path | None = None,
    merge_key_columns: list[str] | None = None,
) -> dict[str, int]:
    """Verify availability of pending exams without requesting exports.

    Parameters
    ----------
    audit_df : pd.DataFrame
        Exams to audit
    full_db : pd.DataFrame | None
        Full database to update with audit results
    max_exams : int | None
        Maximum number of exams to audit
    export_state_file : Path | None
        Path to export state file for saving progress
    merge_key_columns : list[str] | None
        Columns to use for merging

    Returns
    -------
    dict[str, int]
        Audit statistics
    """

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
            if (
                full_db is not None
                and export_state_file
                and merge_key_columns
                and patients_processed % 20 == 0
            ):
                save_current_state(full_db, export_state_file, merge_key_columns)
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
