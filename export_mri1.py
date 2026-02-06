#!/usr/bin/env python
"""Export script for MRI1.0 dataset."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import pydicom
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm.auto import tqdm

from prima.export_utils import (
    MERGE_KEY_COLUMNS,
    PASSWORD,
    USERNAME,
    audit_remote_export_status,
    get_base_modality,
    identify_download_targets,
    import_scrape_ibroker,
    parse_wait_interval,
)
from prima.filesystem_utils import update_metadata_with_disk_status_by_date

# MRI1.0 dataset configuration
MRI1_STUDY_IDS_FILE = (
    "/gpfs/data/karczmar-lab/CAPS/MRI1.0/MRI1.0_AnonymousIDs_hiro.xlsx"
)
MRI1_BASE_DOWNLOAD_DIR = "/gpfs/data/karczmar-lab/CAPS/MRI1.0/"
MRI1_STATUS_FILE = Path(MRI1_BASE_DOWNLOAD_DIR) / "export_status.csv"

warnings.filterwarnings("ignore", message=".*Invalid value for VR UI.*")


def build_mri1_disk_cache_simple(cache_path: Path, base_dir: str) -> Path | None:
    """Build MRI1.0 disk cache by traversing directories and reading DICOM StudyDate.

    Much faster than full fingerprinting - just reads one DICOM file per exam
    to extract StudyDate, without hashing files.

    Handles both flat structure (base_dir/patient_id/exam_dir/) and
    modality-grouped structure (base_dir/MR/patient_id/exam_dir/).

    Parameters
    ----------
    cache_path : Path
        Path where the cache should be created
    base_dir : str
        Base directory for MRI1.0 data (e.g., MRI1_BASE_DOWNLOAD_DIR)

    Returns
    -------
    Path | None
        Path to the created cache file, or None if creation failed
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"  WARNING: MRI1.0 base directory does not exist: {base_path}")
        return None

    # Check if structure is modality-grouped (base_dir/MR/patient_id/) or flat (base_dir/patient_id/)
    scan_root = base_path
    mr_subdir = base_path / "MR"
    if mr_subdir.exists() and mr_subdir.is_dir():
        scan_root = mr_subdir
        print(f"  Detected modality-grouped structure, using: {scan_root}")
    else:
        print(f"  Detected flat structure, using: {scan_root}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Scanning {scan_root} for exam directories...")
    disk_inventory = {}

    # Traverse patient directories
    patient_dirs = [d for d in scan_root.iterdir() if d.is_dir() and d.name.isdigit()]
    print(f"  Found {len(patient_dirs):,} patient directories")

    for patient_dir in tqdm(patient_dirs, desc="Scanning patients"):
        patient_id = patient_dir.name
        exams = {}

        # Traverse exam directories
        exam_dirs = [d for d in patient_dir.iterdir() if d.is_dir()]
        for exam_dir in exam_dirs:
            exam_name = exam_dir.name

            # Find first DICOM file to read StudyDate
            study_date = None
            study_time = None
            study_uid = None
            study_description = None

            # Look for DICOM files (common extensions)
            dicom_files = list(exam_dir.rglob("*.dcm")) + list(exam_dir.rglob("*.DCM"))
            if not dicom_files:
                # Try any file - might be DICOM without extension
                all_files = [f for f in exam_dir.rglob("*") if f.is_file()][:10]
                dicom_files = all_files

            for dicom_file in dicom_files[:1]:  # Just read first file
                try:
                    dcm = pydicom.dcmread(dicom_file, stop_before_pixels=True)
                    if not study_uid:
                        study_uid = str(dcm.get("StudyInstanceUID", ""))
                    if not study_date and hasattr(dcm, "StudyDate") and dcm.StudyDate:
                        study_date = str(dcm.StudyDate)
                    if not study_time and hasattr(dcm, "StudyTime") and dcm.StudyTime:
                        study_time = str(dcm.StudyTime)
                    if (
                        not study_description
                        and hasattr(dcm, "StudyDescription")
                        and dcm.StudyDescription
                    ):
                        study_description = str(dcm.StudyDescription)
                    break  # Got what we need
                except Exception:
                    continue

            # Print exam info
            date_str = (
                f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:8]}"
                if study_date and len(study_date) >= 8
                else "unknown date"
            )
            desc_str = study_description or "no description"
            print(f"    {patient_id}/{exam_name}: {date_str} - {desc_str}")

            # Store exam info (format matches fingerprint cache: uid, hashes, study_date, study_time)
            # Use empty list for hashes since we're not hashing
            exams[exam_name] = (study_uid or "", [], study_date, study_time)

        if exams:
            disk_inventory[patient_id] = exams

    # Write cache file (same format as fingerprint cache for compatibility)
    print(f"  Writing cache to {cache_path}")
    with open(cache_path, "w") as f:
        json.dump(disk_inventory, f, indent=2)

    total_exams = sum(len(exams) for exams in disk_inventory.values())
    size_mb = cache_path.stat().st_size / 1_000_000
    print(
        f"  ✓ Created cache: {len(disk_inventory):,} patients, {total_exams:,} exams ({size_mb:.2f} MB)"
    )
    return cache_path


def _calculate_export_requested(status_df: pd.DataFrame) -> pd.Series:
    """Calculate export_requested from other columns.

    Logic: if on disk, export_requested=True (can't be on disk without requesting)
           if has Accession, export_requested=True (exported in iBroker)
           if Status indicates exported, export_requested=True
           otherwise export_requested=False
    """
    export_requested = (
        status_df.get("is_on_disk", False).fillna(False).astype(bool).copy()
    )

    # Also mark as requested if has Accession (exported in iBroker)
    if "Accession" in status_df.columns:
        has_accession = status_df["Accession"].notna()
        export_requested = export_requested | has_accession

    # Also mark as requested if Status indicates exported
    if "Status" in status_df.columns:
        status_exported = (
            status_df["Status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"already exported", "exported", "completed"})
        )
        export_requested = export_requested | status_exported

    return export_requested


def load_mri1_data(
    max_exams: int | None = None,
    submit_exports: bool = False,
    batch_size: int | None = None,
    include_biopsy: bool = False,
) -> tuple[pd.DataFrame, dict, set[str]]:
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
    include_biopsy : bool
        If False (default), exclude exams with "biopsy" in study description.
        If True, include all exams regardless of biopsy status.

    Returns
    -------
    tuple[pd.DataFrame, dict, set[str]]
        Metadata DataFrame, export statistics dict, and set of queried study IDs.
        Export statistics dict has keys:
        - 'submitted': number of export requests submitted
        - 'already_exported': number discovered to be already exported
        - 'processed': total exams processed for export
    """
    # Import scrape functions (must be done after import_scrape_ibroker is called)
    from prima.export_utils import (
        bootstrap_http_session_from_driver,
        http_get_root,
        login,
        make_driver,
        parse_all_tables_from_page,
        post_fetch_grid,
        post_link_event,
        wait_aspnet_idle,
    )

    # Load fingerprint cache for disk status checking
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

        # Prioritize study IDs that are more likely to have unexported exams
        # Use status CSV if it exists to prioritize
        status_file = Path(MRI1_BASE_DOWNLOAD_DIR) / "export_status.csv"
        if status_file.exists():
            try:
                status_df = pd.read_csv(status_file)
                status_df["study_id"] = status_df["study_id"].astype(str)

                # Calculate export_requested from other columns
                export_requested = _calculate_export_requested(status_df)

                # Find study IDs with unexported exams
                unexported_mask = (~export_requested) & (
                    ~status_df.get("is_on_disk", False).fillna(False)
                )
                study_ids_with_unexported = set(
                    status_df.loc[unexported_mask, "study_id"].unique()
                )

                # Find study IDs where ALL exams are exported (skip these)
                study_id_groups = status_df.groupby("study_id")
                all_exported_study_ids = set()
                for sid, group in study_id_groups:
                    # Check if all exams for this study_id are exported
                    group_export_requested = _calculate_export_requested(group)
                    all_exported = (
                        group_export_requested.all()
                        | group.get("is_on_disk", False).fillna(False).all()
                    )
                    if all_exported and len(group) > 0:
                        all_exported_study_ids.add(sid)

                # Find study IDs that are not in status CSV at all (never queried)
                all_status_study_ids = set(status_df["study_id"].unique())
                never_queried = set(study_ids) - all_status_study_ids

                # Filter out study IDs where all exams are already exported
                study_ids = [
                    sid for sid in study_ids if sid not in all_exported_study_ids
                ]

                # Prioritize: never queried first, then those with unexported exams
                priority_ids = list(never_queried) + list(study_ids_with_unexported)
                remaining_ids = [sid for sid in study_ids if sid not in priority_ids]

                # Reorder: priority IDs first, then remaining
                study_ids = priority_ids + remaining_ids

                print(
                    f"Prioritized {len(priority_ids):,} study IDs "
                    f"({len(never_queried):,} never queried, "
                    f"{len(study_ids_with_unexported):,} with unexported exams). "
                    f"Skipped {len(all_exported_study_ids):,} study IDs with all exams exported."
                )
            except Exception as e:
                print(f"Warning: Could not prioritize study IDs from status CSV: {e}")
                print("Using original order.")

        # Query iBroker directly for each study ID
        if submit_exports:
            print("Querying iBroker and submitting export requests for MR exams...")
        else:
            print("Querying iBroker for MR exams...")
        driver = None
        session = None
        export_stats = {"submitted": 0, "already_exported": 0, "processed": 0}
        try:
            # Initialize driver and session
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

        # Query study IDs incrementally and stop when we have enough ready-to-export exams
        # This avoids querying all study IDs when we only need a small batch
        all_exams = []
        ready_to_export_count = 0

        query_desc = "Querying iBroker"
        if submit_exports:
            query_desc = "Querying iBroker and submitting exports"
        if max_exams is not None:
            query_desc = f"{query_desc} (target: {max_exams} ready-to-export exams)"

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
                # Wait for any modal from previous exports to close before navigating
                # The modal (mdlPopup_backgroundElement) can block btnFetch
                try:
                    WebDriverWait(driver, 5).until(
                        EC.invisibility_of_element_located(
                            (By.ID, "mdlPopup_backgroundElement")
                        )
                    )
                except Exception:
                    # If modal doesn't exist or timeout, continue anyway
                    pass

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
                            f"  - Found available exam from {target_key[0]} ({target_key[1]}). Selecting checkbox."
                        )
                        available_on_page[target_key].click()
                        submitted += 1
                        export_stats["submitted"] += 1
                        export_stats["processed"] += 1
                        requested_any = True
                        submitted_keys.append(target_key)
                    else:
                        print(
                            f"  - INFO: Exam from {target_key[0]} ({target_key[1]}) is no longer available (already exported)."
                        )
                        already_exported += 1
                        export_stats["already_exported"] += 1
                        export_stats["processed"] += 1

                # Submit export request if any exams were selected
                if requested_any:
                    print(f"  Submitting export request for {submitted} exam(s)...")
                    driver.find_element(by="name", value="btnExport").click()
                    wait_aspnet_idle(driver)

                    # Wait for modal popup to close before proceeding to next patient
                    # The modal (mdlPopup_backgroundElement) can block btnFetch for the next patient
                    try:
                        # Wait for modal background to disappear (or not be present)
                        WebDriverWait(driver, 10).until(
                            EC.invisibility_of_element_located(
                                (By.ID, "mdlPopup_backgroundElement")
                            )
                        )
                    except Exception:
                        # If modal doesn't exist or timeout, continue anyway
                        # It might have already closed or not appeared
                        pass

                    print(f"  ✓ Export request submitted for {submitted} exam(s).")

            except Exception as e:
                print(f"ERROR submitting exports for study_id {study_id}: {e}")
                # Continue processing other patients

            return submitted, already_exported, submitted_keys

        # Track exams for incremental status CSV updates
        status_update_chunk_size = 5  # Update status CSV every N study IDs
        exams_for_status_update = []
        status_file_path = Path(MRI1_BASE_DOWNLOAD_DIR) / "export_status.csv"
        queried_study_ids_set: set[str] = set()

        for idx, study_id in enumerate(study_ids, 1):
            # Check if we've reached batch limit
            if (
                submit_exports
                and batch_size is not None
                and export_stats["submitted"] >= batch_size
            ):
                print(
                    f"\nReached batch limit of {batch_size} export requests. Stopping."
                )
                # Update status CSV with any remaining exams before breaking
                if exams_for_status_update:
                    combined_chunk = pd.concat(
                        exams_for_status_update, ignore_index=True
                    )
                    update_status_csv(
                        combined_chunk,
                        status_file_path,
                        fingerprint_cache=mri1_fingerprint_cache,
                    )
                    exams_for_status_update = []
                break

            print(
                f"\n[{idx}/{len(study_ids)}] Querying study_id {study_id}...",
                end=" ",
                flush=True,
            )
            try:
                page_html, state = post_fetch_grid(session, state, str(study_id))
                df = parse_all_tables_from_page(page_html)
                queried_study_ids_set.add(
                    study_id
                )  # Track that we queried this study ID
                print(f"Found {len(df)} total exams", end="", flush=True)

                if not df.empty:
                    df["study_id"] = str(study_id)  # Ensure string type
                    # Filter for MR+BREAST immediately
                    df["base_modality"] = df["Modality"].apply(get_base_modality)
                    df = df[df["base_modality"] == "MR"]
                    print(f", {len(df)} MR exams", end="", flush=True)

                    if "StudyDescription" in df.columns:
                        breast_mask = df["StudyDescription"].str.contains(
                            "BREAST", case=False, na=False
                        )
                        df = df[breast_mask]
                        print(f", {len(df)} BREAST exams", end="", flush=True)

                        # Filter out biopsy exams unless include_biopsy is True
                        if not include_biopsy:
                            biopsy_mask = df["StudyDescription"].str.contains(
                                "biopsy", case=False, na=False
                            )
                            df = df[~biopsy_mask]
                            print(f", {len(df)} non-biopsy exams", end="", flush=True)

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

                        # Check disk status using fingerprint cache
                        def check_exam_on_disk(
                            study_id: str, study_datetime: pd.Timestamp
                        ) -> bool:
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
                        print(
                            f" → {ready_count} ready-to-export "
                            f"(on_disk: {(~df['is_on_disk']).sum()}, "
                            f"exported: {df['is_exported'].sum()})",
                            flush=True,
                        )

                        # Submit exports if requested
                        if submit_exports and ready_count > 0:
                            ready_exams = df[ready_mask]
                            print(
                                f"  Submitting exports for {len(ready_exams)} exam(s)...",
                                flush=True,
                            )
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

                        # Add to status update chunk (even if already exported)
                        exams_for_status_update.append(df)

                        # Update status CSV incrementally every chunk_size study IDs
                        if len(exams_for_status_update) >= status_update_chunk_size:
                            combined_chunk = pd.concat(
                                exams_for_status_update, ignore_index=True
                            )
                            update_status_csv(
                                combined_chunk,
                                status_file_path,
                                fingerprint_cache=mri1_fingerprint_cache,
                            )
                            exams_for_status_update = []
                            print(
                                f"  [Status CSV updated after {idx} study IDs]",
                                flush=True,
                            )

                        if max_exams is not None and ready_to_export_count >= max_exams:
                            print(
                                f"\n✓ Reached target of {max_exams} ready-to-export exams "
                                f"(found {ready_to_export_count} total). "
                                f"Stopped querying after processing {idx} study IDs."
                            )
                            # Update status CSV with any remaining exams before breaking
                            if exams_for_status_update:
                                combined_chunk = pd.concat(
                                    exams_for_status_update, ignore_index=True
                                )
                                update_status_csv(
                                    combined_chunk,
                                    status_file_path,
                                    fingerprint_cache=mri1_fingerprint_cache,
                                )
                                exams_for_status_update = []
                            break
                    else:
                        print(" → No MR+BREAST exams", flush=True)
                else:
                    print(" → No exams found", flush=True)
            except Exception as e:
                print(f" → ERROR: {e}", flush=True)
                continue

        # Update status CSV with any remaining exams at the end
        if exams_for_status_update:
            combined_chunk = pd.concat(exams_for_status_update, ignore_index=True)
            update_status_csv(
                combined_chunk,
                status_file_path,
                fingerprint_cache=mri1_fingerprint_cache,
            )
            print(
                f"  [Final status CSV update with {len(combined_chunk)} exams]",
                flush=True,
            )

        # Progress tracking complete

        # Close driver if we're not submitting exports
        # (if submitting exports, driver is closed after all processing)
        if not submit_exports and driver:
            try:
                print("Closing browser driver...")
                driver.quit()
            except Exception:
                pass
        elif submit_exports and driver:
            # Close driver after submitting exports
            try:
                print("Closing browser driver after export submission...")
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

            # Filter out biopsy exams unless include_biopsy is True
            if not include_biopsy:
                before_biopsy_filter = len(metadata)
                biopsy_mask = metadata["StudyDescription"].str.contains(
                    "biopsy", case=False, na=False
                )
                metadata = metadata[~biopsy_mask]
                excluded_count = biopsy_mask.sum()
                if excluded_count > 0:
                    print(
                        f"Excluded {excluded_count:,} exams with 'biopsy' in description "
                        f"({before_biopsy_filter:,} → {len(metadata):,})."
                    )

        # Deduplicate by (study_id, Study DateTime, StudyDescription)
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

        # initialize export state columns
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
    # Driver is defined in the inner try block, so we need to handle cleanup there
    # For now, driver cleanup happens in the loop above

    return db, export_stats, queried_study_ids_set


def update_status_csv(
    exams_df: pd.DataFrame,
    status_file: Path,
    fingerprint_cache: Path | None = None,
) -> None:
    """Update status CSV with current exam data.

    Merges new/updated rows, preserves existing data.
    Only includes MR+BREAST exams (filtered before calling).
    Updates rows if newer data is available (based on last_updated timestamp).

    Parameters
    ----------
    exams_df : pd.DataFrame
        DataFrame with exam data from iBroker query
    status_file : Path
        Path to status CSV file
    fingerprint_cache : Path | None
        Path to fingerprint cache for disk status checking
    """
    if exams_df.empty:
        return

    # Ensure we only have MR+BREAST exams
    if "base_modality" in exams_df.columns:
        exams_df = exams_df[exams_df["base_modality"] == "MR"].copy()
    if "StudyDescription" in exams_df.columns:
        breast_mask = exams_df["StudyDescription"].str.contains(
            "BREAST", case=False, na=False
        )
        exams_df = exams_df[breast_mask]

    if exams_df.empty:
        return

    # Prepare status DataFrame with required columns
    status_cols = [
        "study_id",
        "Accession",
        "Study DateTime",
        "StudyDescription",
        "Status",  # Raw status from iBroker (e.g., "completed", "exported", etc.)
        "Exported On",
    ]

    # Build status DataFrame
    status_df = pd.DataFrame()
    status_df["study_id"] = exams_df["study_id"].astype(str)
    status_df["Accession"] = exams_df.get("Accession", pd.NA)
    status_df["Study DateTime"] = pd.to_datetime(
        exams_df["Study DateTime"], errors="coerce"
    )
    status_df["StudyDescription"] = exams_df.get("StudyDescription", "")
    status_df["Status"] = exams_df.get("Status", pd.NA)
    status_df["Exported On"] = pd.to_datetime(
        exams_df.get("Exported On", pd.NaT), errors="coerce"
    )

    # Check disk status using fingerprint cache
    # Suppress verbose output - we show full comparison in print_status_summary()
    if fingerprint_cache and fingerprint_cache.exists():
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            status_df = update_metadata_with_disk_status_by_date(
                status_df,
                conservative=True,
                fingerprint_cache=fingerprint_cache,
            )
    else:
        status_df["is_on_disk"] = False

    if "is_on_disk" not in status_df.columns:
        status_df["is_on_disk"] = False

    # Load existing status CSV if it exists
    if status_file.exists():
        existing_status = pd.read_csv(status_file)
        existing_status["study_id"] = existing_status["study_id"].astype(str)
        existing_status["Study DateTime"] = pd.to_datetime(
            existing_status["Study DateTime"], errors="coerce"
        )

        # Merge: update existing rows, add new rows
        merged = existing_status.merge(
            status_df,
            on=["study_id", "Study DateTime", "StudyDescription"],
            how="outer",
            suffixes=("_old", "_new"),
        )

        # For each column, prefer new value (latest query takes precedence)
        for col in status_cols + ["is_on_disk"]:
            old_col = f"{col}_old" if col in existing_status.columns else None
            new_col = f"{col}_new"

            if old_col and old_col in merged.columns and new_col in merged.columns:
                # Prefer new value if available, otherwise keep old
                mask_has_new = merged[new_col].notna()
                merged[col] = merged[old_col]
                merged.loc[mask_has_new, col] = merged.loc[mask_has_new, new_col]
            elif new_col in merged.columns:
                merged[col] = merged[new_col]
            elif old_col and old_col in merged.columns:
                merged[col] = merged[old_col]

        # Clean up merge columns
        drop_cols = [
            c for c in merged.columns if c.endswith("_old") or c.endswith("_new")
        ]
        merged = merged.drop(columns=drop_cols)

        final_status = merged
    else:
        final_status = status_df

    # Ensure all required columns exist
    required_cols = [
        "study_id",
        "Accession",
        "Study DateTime",
        "StudyDescription",
        "Status",
        "Exported On",
        "is_on_disk",
    ]
    for col in required_cols:
        if col not in final_status.columns:
            if col == "Exported On":
                final_status[col] = pd.NaT
            else:
                final_status[col] = pd.NA

    # Sort by study_id and Study DateTime
    final_status = final_status.sort_values(
        by=["study_id", "Study DateTime", "StudyDescription"]
    )

    # Write to CSV
    status_file.parent.mkdir(parents=True, exist_ok=True)
    final_status.to_csv(status_file, index=False)

    print(f"Updated status CSV: {len(final_status):,} exams ({status_file})")


def print_status_summary(
    status_file: Path, fingerprint_cache: Path | None = None
) -> None:
    """Print summary statistics from status CSV, including disk status comparison.

    Parameters
    ----------
    status_file : Path
        Path to status CSV file
    fingerprint_cache : Path | None
        Optional path to fingerprint cache for disk status comparison
    """
    if not status_file.exists():
        print("Status CSV does not exist yet.")
        return

    status_df = pd.read_csv(status_file)
    status_df["Study DateTime"] = pd.to_datetime(
        status_df["Study DateTime"], errors="coerce"
    )
    status_df["is_on_disk"] = (
        status_df.get("is_on_disk", False).fillna(False).astype(bool)
    )
    status_df["Exported On"] = pd.to_datetime(
        status_df.get("Exported On", pd.NaT), errors="coerce"
    )

    # Calculate export_requested from other columns
    export_requested = _calculate_export_requested(status_df)

    total = len(status_df)
    on_disk = status_df["is_on_disk"].sum()
    exported_not_on_disk = (
        status_df["Exported On"].notna() & (~status_df["is_on_disk"])
    ).sum()
    export_requested_pending = (
        export_requested & status_df["Exported On"].isna() & (~status_df["is_on_disk"])
    ).sum()
    not_exported = ((~export_requested) & (~status_df["is_on_disk"])).sum()

    print("\n--- Full Status CSV Summary (All Known Exams) ---")
    print(f"Total MR+BREAST exams in status CSV: {total:,}")
    print(f"  - On disk: {on_disk:,} ({on_disk / total * 100:.1f}%)")
    print(
        f"  - Exported (not on disk): {exported_not_on_disk:,} ({exported_not_on_disk / total * 100:.1f}%)"
    )
    print(
        f"  - Export requested (pending): {export_requested_pending:,} ({export_requested_pending / total * 100:.1f}%)"
    )
    print(f"  - Not yet exported: {not_exported:,} ({not_exported / total * 100:.1f}%)")

    # Show disk status comparison if fingerprint cache is available
    if fingerprint_cache is not None and fingerprint_cache.exists():
        from filesystem_utils import load_disk_dates_from_fingerprints

        disk_dates = load_disk_dates_from_fingerprints(fingerprint_cache)
        if disk_dates:
            # Count unique (patient, date) pairs on disk
            total_disk_pairs = sum(len(dates) for dates in disk_dates.values())
            total_disk_patients = len(disk_dates)

            # Count unique (patient, date) pairs in status CSV
            status_df["_study_date_str"] = status_df["Study DateTime"].dt.strftime(
                "%Y%m%d"
            )
            status_pairs = set(
                (str(int(row["study_id"])), row["_study_date_str"])
                for _, row in status_df.iterrows()
                if pd.notna(row["Study DateTime"]) and pd.notna(row["study_id"])
            )
            status_df.drop(columns=["_study_date_str"], inplace=True)

            matched_pairs = sum(
                1
                for pid, date_str in status_pairs
                if pid in disk_dates and date_str in disk_dates[pid]
            )

            print("\n--- Disk Status Comparison (Full Picture) ---")
            print(
                f"  - Unique (patient, date) pairs in status CSV: {len(status_pairs):,}"
            )
            print(
                f"  - Unique (patient, date) pairs on disk: {total_disk_pairs:,} ({total_disk_patients:,} patients)"
            )
            print(f"  - Matched pairs: {matched_pairs:,}")
            print(f"  - Disk-only pairs: {total_disk_pairs - matched_pairs:,}")


def query_and_update_status_during_wait(
    study_ids: list[str],
    already_queried: set[str],
    status_file: Path,
    fingerprint_cache: Path | None,
    wait_seconds: float,
    include_biopsy: bool = False,
) -> set[str]:
    """Query remaining study IDs and update status CSV during wait period.

    Parameters
    ----------
    study_ids : list[str]
        All study IDs to query
    already_queried : set[str]
        Study IDs already queried in this session
    status_file : Path
        Path to status CSV file
    fingerprint_cache : Path | None
        Path to fingerprint cache for disk status
    wait_seconds : float
        Maximum seconds to spend querying (should be less than wait period)
    include_biopsy : bool
        Whether to include biopsy exams

    Returns
    -------
    set[str]
        Set of newly queried study IDs
    """
    from prima.export_utils import (
        bootstrap_http_session_from_driver,
        http_get_root,
        login,
        make_driver,
        parse_all_tables_from_page,
        post_fetch_grid,
        post_link_event,
    )

    remaining_ids = [sid for sid in study_ids if sid not in already_queried]
    if not remaining_ids:
        return set()

    print(
        f"\nQuerying {len(remaining_ids):,} remaining study IDs to update status CSV..."
    )

    driver = None
    session = None
    newly_queried = set()
    start_time = time.time()

    try:
        driver = make_driver()
        login(driver, USERNAME, PASSWORD)
        session = bootstrap_http_session_from_driver(driver)

        # Initialize HTTP session state
        page_html, state = http_get_root(session)
        page_html, state = post_link_event(session, state, "lbAll")

        all_exams = []
        status_update_chunk_size = 10  # Update status CSV every N study IDs
        exams_for_status_update = []

        # Track stats for less verbose output
        no_exams_count = 0
        no_breast_count = 0
        last_summary_idx = 0
        batch_size = 20  # Print summary every N study IDs

        for idx, study_id in enumerate(remaining_ids, 1):
            # Check if we've used up our time budget
            elapsed = time.time() - start_time
            if elapsed >= wait_seconds * 0.9:  # Use 90% of wait time for querying
                print(f"\nTime budget reached ({elapsed:.1f}s). Stopping query.")
                break

            # Print summary every batch_size study IDs
            if idx % batch_size == 1 and idx > 1:
                summary_parts = []
                if no_exams_count > 0:
                    summary_parts.append(f"{no_exams_count} with no exams")
                if no_breast_count > 0:
                    summary_parts.append(f"{no_breast_count} with no BREAST exams")
                if summary_parts:
                    print(
                        f"  [{last_summary_idx}-{idx - 1}] Summary: {', '.join(summary_parts)}",
                        flush=True,
                    )
                no_exams_count = 0
                no_breast_count = 0
                last_summary_idx = idx

            # Only print individual progress every 10 study IDs or when we find BREAST exams
            verbose_progress = idx % 10 == 1 or idx == len(remaining_ids)
            if verbose_progress:
                print(
                    f"[{idx}/{len(remaining_ids)}] Querying study_id {study_id}...",
                    end=" ",
                    flush=True,
                )

            try:
                page_html, state = post_fetch_grid(session, state, str(study_id))
                df = parse_all_tables_from_page(page_html)
                # Track that we queried this study ID regardless of results
                newly_queried.add(study_id)

                if not df.empty:
                    df["study_id"] = str(study_id)
                    df["base_modality"] = df["Modality"].apply(get_base_modality)
                    df = df[df["base_modality"] == "MR"]

                    if "StudyDescription" in df.columns:
                        breast_mask = df["StudyDescription"].str.contains(
                            "BREAST", case=False, na=False
                        )
                        df = df[breast_mask]

                        if not include_biopsy:
                            biopsy_mask = df["StudyDescription"].str.contains(
                                "biopsy", case=False, na=False
                            )
                            df = df[~biopsy_mask]

                    if not df.empty:
                        df["Study DateTime"] = pd.to_datetime(
                            df["Study DateTime"], errors="coerce"
                        )
                        all_exams.append(df)
                        exams_for_status_update.append(df)
                        # Always print when we find BREAST exams
                        print(f"Found {len(df)} BREAST exam(s) ✓", flush=True)

                        # Update status CSV incrementally every chunk_size study IDs
                        if len(exams_for_status_update) >= status_update_chunk_size:
                            combined_chunk = pd.concat(
                                exams_for_status_update, ignore_index=True
                            )
                            update_status_csv(
                                combined_chunk, status_file, fingerprint_cache
                            )
                            exams_for_status_update = []
                            print(
                                f"  [Status CSV updated after {idx} study IDs]",
                                flush=True,
                            )
                            # Print periodic status summary every 50 study IDs
                            if idx % 50 == 0:
                                print("\n--- Periodic Status Summary ---", flush=True)
                                print_status_summary(status_file, fingerprint_cache)
                    else:
                        no_breast_count += 1
                        if verbose_progress:
                            print("No MR+BREAST exams", flush=True)
                else:
                    no_exams_count += 1
                    if verbose_progress:
                        print("No exams found", flush=True)
            except Exception as e:
                print(f" → ERROR: {e}", flush=True)
                continue

        # Print final summary
        if no_exams_count > 0 or no_breast_count > 0:
            summary_parts = []
            if no_exams_count > 0:
                summary_parts.append(f"{no_exams_count} with no exams")
            if no_breast_count > 0:
                summary_parts.append(f"{no_breast_count} with no BREAST exams")
            if summary_parts:
                print(
                    f"  [{last_summary_idx}-{len(remaining_ids)}] Summary: {', '.join(summary_parts)}",
                    flush=True,
                )

        # Update status CSV with any remaining exams
        if exams_for_status_update:
            combined_chunk = pd.concat(exams_for_status_update, ignore_index=True)
            update_status_csv(combined_chunk, status_file, fingerprint_cache)
            print(
                f"  [Final status CSV update with {len(combined_chunk):,} exams from chunk]",
                flush=True,
            )
            # Print final status summary
            print("\n--- Final Status Summary ---", flush=True)
            print_status_summary(status_file, fingerprint_cache)

    except Exception as e:
        print(f"Error during status update query: {e}")
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

    return newly_queried


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
        f"{datetime.now():%Y-%m-%d %H:%M:%S} (MRI1.0) ==="
    )
    print(cycle_banner)

    # Submit exports during query phase to avoid double-pass
    max_exams_to_query = args.batch_size
    result = load_mri1_data(
        max_exams=max_exams_to_query,
        submit_exports=True,
        batch_size=args.batch_size,
        include_biopsy=args.include_biopsy,
    )
    db, export_stats, cycle_queried_study_ids = result

    # Rebuild fingerprint cache at start of each cycle to include newly downloaded exams
    fingerprint_cache = None
    mri1_cache = Path("data/destination_fingerprints_mri1.json")
    print("Rebuilding MRI1.0 fingerprint cache to reflect current disk state...")
    fingerprint_cache = build_mri1_disk_cache_simple(mri1_cache, MRI1_BASE_DOWNLOAD_DIR)
    if fingerprint_cache:
        print(f"✓ Updated MRI1.0 fingerprint cache: {fingerprint_cache}")
    else:
        print("WARNING: Failed to build fingerprint cache")
    # Suppress verbose disk status summary - we show full comparison in print_status_summary()
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        db = update_metadata_with_disk_status_by_date(
            db, conservative=True, fingerprint_cache=fingerprint_cache
        )

    # Update status CSV with all exams from this cycle first
    update_status_csv(db, MRI1_STATUS_FILE, fingerprint_cache=fingerprint_cache)

    # Print cycle-specific summary (what happened THIS cycle)
    successfully_exported_count = export_stats.get("submitted", 0)
    already_exported_count = export_stats.get("already_exported", 0)
    processed_count = export_stats.get("processed", 0)

    print("\n--- Cycle Summary (This Cycle Only) ---")
    print(f"Study IDs queried this cycle: {len(cycle_queried_study_ids):,}")
    print(f"Exams found this cycle: {len(db):,}")
    print(
        f"  - Successfully submitted export requests: {successfully_exported_count} exam(s)"
    )
    print(f"  - Discovered to be already exported: {already_exported_count} exam(s)")
    print(f"  - Total exams processed: {processed_count} exam(s)")

    # Print full status CSV summary (complete picture of everything we know)
    print_status_summary(MRI1_STATUS_FILE, fingerprint_cache=fingerprint_cache)

    # Identify targets for next cycle (suppress verbose output since we show full status CSV summary above)
    targets = identify_download_targets(
        db,
        filter_by_genotyping=False,
        modality="MR",
        base_download_dir=MRI1_BASE_DOWNLOAD_DIR,
        dataset="mri1.0",
        verbose=False,  # Suppress cycle-specific summary - we show full status CSV summary instead
    )

    db["is_target"] = False
    db.loc[targets.index, "is_target"] = True

    # Only initialize download_attempt_outcome for NEW targets, preserve previous outcomes
    new_targets_mask = db["is_target"] & db["download_attempt_outcome"].isna()
    db.loc[new_targets_mask, "download_attempt_outcome"] = pd.NA
    db.loc[new_targets_mask, "export_requested_on"] = pd.NaT

    return {
        "submitted": successfully_exported_count,
        "already_exported": already_exported_count,
        "processed": export_stats.get("processed", 0),
        "targets_considered": len(targets),
        "target_indices": targets.index.tolist(),
        "queried_study_ids": cycle_queried_study_ids,
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
        f"\n=== Refresh cycle {cycle_number}: auditing export status before next run (MRI1.0) ==="
    )

    refresh_db, _, _ = load_mri1_data(
        max_exams=None, submit_exports=False, include_biopsy=args.include_biopsy
    )

    # Rebuild fingerprint cache to reflect current disk state
    fingerprint_cache = None
    mri1_cache = Path("data/destination_fingerprints_mri1.json")
    print("Rebuilding MRI1.0 fingerprint cache to reflect current disk state...")
    fingerprint_cache = build_mri1_disk_cache_simple(mri1_cache, MRI1_BASE_DOWNLOAD_DIR)
    if fingerprint_cache:
        print(f"✓ Updated MRI1.0 fingerprint cache: {fingerprint_cache}")
    else:
        print("WARNING: Failed to build fingerprint cache")
    # Suppress verbose disk status summary - we show full comparison in print_status_summary()
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        refresh_db = update_metadata_with_disk_status_by_date(
            refresh_db, conservative=True, fingerprint_cache=fingerprint_cache
        )

    # Modality is always MR for MRI1.0
    modality = "MR"
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
        export_state_file=None,  # Status CSV is now the source of truth
        merge_key_columns=MERGE_KEY_COLUMNS,
    )

    print(
        "Audit summary: "
        f"checked {audit_stats['audited']} exams — "
        f"marked {audit_stats['marked_exported']} as exported, "
        f"{audit_stats['still_available']} still available."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Identify and download MRI1.0 imaging exams from iBroker."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Max number of exams to request per cycle.",
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
        "--include-biopsy",
        action="store_true",
        help="Include exams with 'biopsy' in study description (default: exclude biopsy exams).",
    )

    args = parser.parse_args()

    # MRI1.0 dataset requires credentials even for status-only mode (needs to query iBroker)
    if not all([USERNAME, PASSWORD]):
        print(
            "ERROR: IBROKER_USERNAME and IBROKER_PASSWORD must be set for MRI1.0 dataset.",
            file=sys.stderr,
        )
        sys.exit(1)

    # import scrape_ibroker now that we know we need it (triggers credential check)
    import_scrape_ibroker()

    # status-only mode: do full refresh of all study IDs
    if args.status_only:
        print("Running in --status-only mode (no exports will be performed)\n")
        print("Querying all study IDs to refresh status CSV...")

        # Load all study IDs
        study_ids_df = pd.read_excel(MRI1_STUDY_IDS_FILE)
        study_ids = study_ids_df["AnonymousID"].astype(str).tolist()

        # Rebuild fingerprint cache to reflect current disk state
        print("Rebuilding MRI1.0 fingerprint cache to reflect current disk state...")
        mri1_cache = Path("data/destination_fingerprints_mri1.json")
        fingerprint_cache = build_mri1_disk_cache_simple(
            mri1_cache, MRI1_BASE_DOWNLOAD_DIR
        )
        if fingerprint_cache:
            print(f"✓ Updated MRI1.0 fingerprint cache: {fingerprint_cache}")
        else:
            print("WARNING: Failed to build fingerprint cache")

        newly_queried = query_and_update_status_during_wait(
            study_ids=study_ids,
            already_queried=set(),
            status_file=MRI1_STATUS_FILE,
            fingerprint_cache=fingerprint_cache,
            wait_seconds=float("inf"),  # No time limit for status-only mode
            include_biopsy=args.include_biopsy,
        )

        print_status_summary(MRI1_STATUS_FILE, fingerprint_cache=fingerprint_cache)
        sys.exit(0)

    # Load all study IDs for status tracking
    study_ids_df = pd.read_excel(MRI1_STUDY_IDS_FILE)
    all_study_ids = study_ids_df["AnonymousID"].astype(str).tolist()
    queried_study_ids: set[str] = set()

    cycles_run = 0
    last_target_indices: list[int] | None = None
    try:
        while True:
            cycles_run += 1
            cycle_result = run_export_cycle(args, cycles_run)
            last_target_indices = cycle_result.get("target_indices")

            # Track which study IDs were queried in this cycle
            cycle_queried = cycle_result.get("queried_study_ids", set())
            queried_study_ids.update(cycle_queried)
            if cycle_queried:
                print(
                    f"Cycle queried {len(cycle_queried):,} study IDs. Total queried: {len(queried_study_ids):,} / {len(all_study_ids):,}"
                )

            max_cycles = args.max_cycles
            if max_cycles > 0 and cycles_run >= max_cycles:
                break

            wait_seconds = args.loop_wait
            if wait_seconds <= 0:
                break

            # During wait period, query remaining study IDs and update status CSV
            if len(queried_study_ids) < len(all_study_ids):
                mri1_cache = Path("data/destination_fingerprints_mri1.json")
                fingerprint_cache = mri1_cache if mri1_cache.exists() else None

                try:
                    newly_queried = query_and_update_status_during_wait(
                        study_ids=all_study_ids,
                        already_queried=queried_study_ids,
                        status_file=MRI1_STATUS_FILE,
                        fingerprint_cache=fingerprint_cache,
                        wait_seconds=wait_seconds,
                        include_biopsy=args.include_biopsy,
                    )
                    queried_study_ids.update(newly_queried)
                    print(
                        f"Total study IDs queried: {len(queried_study_ids):,} / {len(all_study_ids):,}"
                    )
                except Exception as exc:
                    print(
                        f"\nWARNING: Status update during wait failed with error: {exc}. Continuing."
                    )

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

            # Calculate remaining wait time (if any)
            # query_and_update_status_during_wait uses up to 90% of wait_seconds
            remaining_wait = max(0, wait_seconds * 0.1)
            if remaining_wait > 0:
                print(
                    f"\nWaiting {remaining_wait:.1f} seconds before starting the next cycle..."
                )
                try:
                    time.sleep(remaining_wait)
                except KeyboardInterrupt:
                    print("\nLoop interrupted during wait; exiting.")
                    break
    except KeyboardInterrupt:
        print("\nLoop interrupted; exiting.")

    if cycles_run:
        print(f"\nRan {cycles_run} cycle{'s' if cycles_run != 1 else ''}.")


if __name__ == "__main__":
    main()
