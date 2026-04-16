#!/usr/bin/env python
# ruff: noqa: E402
"""Deduplicate exams on disk, preferring versions with iBroker accession numbers."""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from time import monotonic
from typing import Dict, List, Set, Tuple

import pandas as pd
import pydicom
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prima.fingerprint_utils import ExamFingerprint

# suppress pydicom VR UI validation warnings for non-standard UIDs
warnings.filterwarnings("ignore", message=".*Invalid value for VR UI.*", append=True)
logging.getLogger("pydicom.valuerep").setLevel(logging.ERROR)

# --- CONFIGURATION ---
DST_ROOT = Path("/gpfs/data/huo-lab/Image/ChiMEC/MG")
REPO_PATH = PROJECT_ROOT
CACHE_FILE_REL_PATH = Path("data/destination_fingerprints.json")
LOCAL_CACHE_FILE = REPO_PATH / CACHE_FILE_REL_PATH
IBROKER_METADATA_FILE = REPO_PATH / "data" / "imaging_metadata.csv"
FINGERPRINTER_PARALLEL_JOBS = 8
# --- END CONFIGURATION ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("deduplicate_disk.log"), logging.StreamHandler()],
)


def extract_accession_from_exam_name(exam_name: str) -> str:
    """Extract accession number from exam directory name.

    Exam names can be:
    - "2O42657" (just accession)
    - "2O42657-2015-12-08" (accession-date)

    Returns the accession number (part before first dash).
    """
    return exam_name.split("-")[0]


def load_ibroker_accessions(metadata_path: Path) -> Dict[str, Set[str]]:
    """Load iBroker metadata and build mapping of patient_id -> set of accession numbers.

    Parameters
    ----------
    metadata_path : Path
        path to iBroker metadata CSV

    Returns
    -------
    Dict[str, Set[str]]
        mapping of patient_id (as string) -> {set of accession numbers}
    """
    if not metadata_path.exists():
        logging.warning(f"iBroker metadata not found: {metadata_path}")
        logging.warning("Will deduplicate without preferring iBroker accession numbers")
        return {}

    logging.info(f"Loading iBroker metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # ensure patient_id is string
    if "study_id" in df.columns:
        df["study_id"] = df["study_id"].astype(str)
    else:
        logging.warning("No 'study_id' column in iBroker metadata")
        return {}

    # extract accession numbers
    if "Accession" not in df.columns:
        logging.warning("No 'Accession' column in iBroker metadata")
        return {}

    accessions = defaultdict(set)
    for _, row in df.iterrows():
        patient_id = str(row["study_id"])
        accession = row.get("Accession")
        if pd.notna(accession) and accession:
            accessions[patient_id].add(str(accession).strip())

    total_accessions = sum(len(accs) for accs in accessions.values())
    logging.info(
        f"Loaded {len(accessions):,} patients with {total_accessions:,} accession numbers from iBroker"
    )

    return dict(accessions)


def update_destination_inventory(parallel_jobs: int):
    """Run fingerprinter locally on destination to update inventory cache.

    Parameters
    ----------
    parallel_jobs : int
        number of parallel workers for fingerprinting
    """
    logging.info("--- updating destination inventory ---")

    fingerprinter_script = REPO_PATH / "ops" / "fingerprinter.py"
    cache_file = LOCAL_CACHE_FILE
    log_file = REPO_PATH / "data" / "fingerprinter.log"

    if not fingerprinter_script.exists():
        logging.error(f"Fingerprinter script not found: {fingerprinter_script}")
        raise FileNotFoundError(
            f"Fingerprinter script not found: {fingerprinter_script}"
        )

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Running fingerprinter on {DST_ROOT}")
    logging.info(f"Output cache: {cache_file}")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Parallel jobs: {parallel_jobs}")

    cmd = [
        sys.executable,
        "-u",
        str(fingerprinter_script),
        str(DST_ROOT),
        str(cache_file),
        "--parallel-jobs",
        str(parallel_jobs),
    ]

    logging.info(f"Executing: {' '.join(cmd)}")
    with open(log_file, "w") as log_f:
        result = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if result.returncode != 0:
        logging.error(f"Fingerprinter failed with return code {result.returncode}")
        logging.error(f"Check log file: {log_file}")
        raise RuntimeError(f"Fingerprinter failed with return code {result.returncode}")

    if not cache_file.exists():
        logging.error(f"Cache file was not created: {cache_file}")
        raise RuntimeError(f"Cache file was not created: {cache_file}")

    size_mb = cache_file.stat().st_size / 1_000_000
    logging.info(f"Created cache file: {cache_file} ({size_mb:.2f} MB)")

    with open(cache_file) as f:
        data = json.load(f)
    n_patients = len(data)
    n_exams = sum(len(v) for v in data.values())
    logging.info(f"Inventory snapshot: {n_patients:,} patients, {n_exams:,} exams")


def load_destination_inventory() -> Dict[str, Dict[ExamFingerprint, str]]:
    """Load destination inventory from cache file."""
    if not LOCAL_CACHE_FILE.exists():
        logging.error(f"Local cache file not found: {LOCAL_CACHE_FILE}")
        raise FileNotFoundError(f"Local cache file not found: {LOCAL_CACHE_FILE}")

    logging.info(f"Loading destination inventory from {LOCAL_CACHE_FILE}...")
    t0 = monotonic()
    file_size_mb = LOCAL_CACHE_FILE.stat().st_size / 1_000_000
    logging.info(f"Cache file size: {file_size_mb:.1f} MB")

    with open(LOCAL_CACHE_FILE) as f:
        logging.info("Parsing JSON...")
        json_start = monotonic()
        raw_data = json.load(f)
        json_time = monotonic() - json_start
        logging.info(f"JSON parsed in {json_time:.1f}s")

    logging.info("Building inventory structure...")
    build_start = monotonic()
    inventory = {}
    for patient_id, exams in raw_data.items():
        patient_inv = {}
        for name, data in exams.items():
            uid, hashes, study_date, study_time = data
            fp = ExamFingerprint(uid, frozenset(hashes), study_date, study_time)
            patient_inv[fp] = name
        inventory[patient_id] = patient_inv
    build_time = monotonic() - build_start
    total_time = monotonic() - t0
    logging.info(f"Inventory structure built in {build_time:.1f}s")
    logging.info(f"Destination inventory loaded in {total_time:.1f}s total")
    return inventory


def get_sop_instance_uids(exam_path: Path) -> Set[str]:
    """Extract all SOPInstanceUIDs from an exam directory.

    Parameters
    ----------
    exam_path : Path
        path to exam directory

    Returns
    -------
    Set[str]
        set of SOPInstanceUIDs found in the exam
    """
    sop_uids = set()
    for dcm_file in exam_path.rglob("*.dcm"):
        try:
            dcm = pydicom.dcmread(dcm_file, stop_before_pixels=True)
            if hasattr(dcm, "SOPInstanceUID") and dcm.SOPInstanceUID:
                sop_uids.add(str(dcm.SOPInstanceUID))
        except (pydicom.errors.InvalidDicomError, Exception):
            continue
    return sop_uids


def find_duplicate_groups(
    patient_id: str,
    patient_inventory: Dict[ExamFingerprint, str],
) -> List[Set[str]]:
    """Find groups of duplicate exams for a patient based on StudyInstanceUID and SOPInstanceUID set.

    Two exams are considered duplicates if they have:
    - Same StudyInstanceUID
    - Same set of SOPInstanceUIDs (same images)

    Parameters
    ----------
    patient_id : str
        patient ID
    patient_inventory : Dict[ExamFingerprint, str]
        fingerprint -> exam name mapping

    Returns
    -------
    List[Set[str]]
        list of exam name sets for duplicate groups
    """
    # group by StudyInstanceUID - exams with the same UID are potential duplicates
    uid_to_exams: Dict[str, List[str]] = defaultdict(list)
    for fp, name in patient_inventory.items():
        uid_to_exams[fp.study_uid].append(name)

    duplicate_groups = []
    for study_uid, exam_names in uid_to_exams.items():
        if len(exam_names) < 2:
            continue

        # group by SOPInstanceUID set
        sop_set_to_exams: Dict[frozenset, List[str]] = defaultdict(list)
        for exam_name in exam_names:
            exam_path = DST_ROOT / patient_id / exam_name
            if not exam_path.exists():
                continue
            sop_uids = get_sop_instance_uids(exam_path)
            if sop_uids:
                sop_set_to_exams[frozenset(sop_uids)].append(exam_name)

        # find groups with same StudyInstanceUID and same SOPInstanceUID set
        for sop_set, exams in sop_set_to_exams.items():
            if len(exams) >= 2:
                duplicate_groups.append(set(exams))

    return duplicate_groups


def choose_preferred_exam(
    exam_names: Set[str],
    patient_id: str,
    ibroker_accessions: Dict[str, Set[str]],
) -> str:
    """Choose which exam to keep from a duplicate group.

    Prefers exams with accession numbers from iBroker metadata.
    If multiple exams have iBroker accessions, prefers lexicographically greater (more recent).
    If none have iBroker accessions, falls back to lexicographically greater.

    Parameters
    ----------
    exam_names : Set[str]
        set of exam directory names that are duplicates
    patient_id : str
        patient ID
    ibroker_accessions : Dict[str, Set[str]]
        mapping of patient_id -> set of accession numbers from iBroker

    Returns
    -------
    str
        exam name to keep
    """
    patient_accessions = ibroker_accessions.get(patient_id, set())

    # extract accession numbers from exam names
    exam_to_accession = {
        name: extract_accession_from_exam_name(name) for name in exam_names
    }

    # find exams that match iBroker accessions
    ibroker_matches = [
        name for name, acc in exam_to_accession.items() if acc in patient_accessions
    ]

    if ibroker_matches:
        # prefer iBroker matches, choose lexicographically greatest (most recent)
        return max(ibroker_matches)

    # no iBroker matches, fall back to lexicographically greatest
    return max(exam_names)


def deduplicate_patient(
    patient_id: str,
    patient_inventory: Dict[ExamFingerprint, str],
    ibroker_accessions: Dict[str, Set[str]],
    output_dir: Path,
) -> Tuple[int, int]:
    """Find and remove duplicate exams for a single patient.

    Moves duplicates to output_dir, preserving patient/exam structure.
    Duplicates are identified by matching StudyInstanceUID and SOPInstanceUID set.

    Parameters
    ----------
    patient_id : str
        patient ID
    patient_inventory : Dict[ExamFingerprint, str]
        fingerprint -> exam name mapping for this patient
    ibroker_accessions : Dict[str, Set[str]]
        mapping of patient_id -> set of accession numbers from iBroker
    output_dir : Path
        directory to move duplicates to

    Returns
    -------
    Tuple[int, int]
        (duplicates_found, duplicates_removed)
    """
    duplicate_groups = find_duplicate_groups(patient_id, patient_inventory)
    if not duplicate_groups:
        return 0, 0

    duplicates_found = sum(len(g) for g in duplicate_groups)
    duplicates_removed = 0

    for group in duplicate_groups:
        if len(group) < 2:
            continue

        preferred = choose_preferred_exam(group, patient_id, ibroker_accessions)
        to_remove = [n for n in group if n != preferred]

        logging.info(
            f"Patient {patient_id}: found {len(group)} duplicate exams (same StudyInstanceUID and SOPInstanceUIDs): {sorted(group)}"
        )
        logging.info(f"  Keeping: {preferred}")
        logging.info(f"  Removing: {to_remove}")

        for exam_name in to_remove:
            exam_path = DST_ROOT / patient_id / exam_name
            if not exam_path.exists():
                logging.warning(f"  Exam path does not exist: {exam_path}")
                continue

            # move to output directory, preserving patient/exam structure
            staging_path = output_dir / patient_id / exam_name
            try:
                staging_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(exam_path), str(staging_path))
                logging.info(f"  Moved to output dir: {exam_path} -> {staging_path}")
                duplicates_removed += 1
            except Exception as e:
                logging.error(f"  Failed to move {exam_path} to output dir: {e}")

    return duplicates_found, duplicates_removed


def main(
    update_fingerprints: bool,
    fingerprint_parallel_jobs: int,
):
    """Main entry point for deduplication."""
    logging.info("=== STARTING DEDUPLICATION ===")

    # output directory is always within the root directory being deduplicated
    output_dir = DST_ROOT / "duplicates"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Duplicates will be moved to: {output_dir}")
    logging.info(
        "Deduplication criteria: same StudyInstanceUID and same set of SOPInstanceUIDs"
    )

    if update_fingerprints:
        logging.info("Step 1: Updating destination inventory...")
        update_destination_inventory(fingerprint_parallel_jobs)
    else:
        logging.info(
            "Skipping fingerprint update (use --update-fingerprints to enable)"
        )

    logging.info("Step 2: Loading iBroker accession numbers...")
    ibroker_accessions = load_ibroker_accessions(IBROKER_METADATA_FILE)

    logging.info("Step 3: Loading destination inventory...")
    inv_load_start = monotonic()
    dest_inventory = load_destination_inventory()
    inv_load_time = monotonic() - inv_load_start
    if not dest_inventory:
        logging.warning("No destination inventory loaded. Exiting.")
        return

    inv_patients = len(dest_inventory)
    inv_exams = sum(len(exams) for exams in dest_inventory.values())
    logging.info(
        f"Destination inventory summary: {inv_patients:,} patients, {inv_exams:,} exams"
    )
    logging.info(f"Inventory loading took {inv_load_time:.1f}s")

    logging.info("Step 4: Scanning destination for patient directories...")
    scan_start = monotonic()
    patient_dirs = [d for d in DST_ROOT.iterdir() if d.is_dir() and d.name.isdigit()]
    scan_time = monotonic() - scan_start
    logging.info(f"Found {len(patient_dirs)} patient directories in {scan_time:.1f}s")

    logging.info("Step 5: Finding and removing duplicates...")
    total_duplicates_found = 0
    total_duplicates_removed = 0
    patients_with_duplicates = 0

    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        patient_id = patient_dir.name
        patient_inventory = dest_inventory.get(patient_id, {})
        if not patient_inventory:
            continue

        found, removed = deduplicate_patient(
            patient_id,
            patient_inventory,
            ibroker_accessions,
            output_dir,
        )
        if found > 0:
            patients_with_duplicates += 1
            total_duplicates_found += found
            total_duplicates_removed += removed

    logging.info("=== DEDUPLICATION COMPLETE ===")
    logging.info(f"Patients with duplicates: {patients_with_duplicates}")
    logging.info(f"Total duplicate exams found: {total_duplicates_found}")
    logging.info(
        f"Total duplicate exams moved to output dir: {total_duplicates_removed}"
    )
    logging.info(f"Duplicates moved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find and remove duplicate exams in destination directory, "
        "preferring versions with iBroker accession numbers. "
        "Duplicates are identified by matching StudyInstanceUID and SOPInstanceUID set. "
        "Duplicates are moved to 'duplicates' subdirectory within the root directory."
    )
    parser.add_argument(
        "--update-fingerprints",
        action="store_true",
        help="Update fingerprint inventory before deduplication (slow).",
    )
    parser.add_argument(
        "--fingerprint-parallel-jobs",
        type=int,
        default=FINGERPRINTER_PARALLEL_JOBS,
        help="Number of parallel jobs for fingerprinting.",
    )
    args = parser.parse_args()

    main(
        update_fingerprints=args.update_fingerprints,
        fingerprint_parallel_jobs=args.fingerprint_parallel_jobs,
    )
