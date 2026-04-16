#!/usr/bin/env python
# ruff: noqa: E402

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
STAGING_DIR = DST_ROOT / "_synced_and_queued_for_deletion"
REPO_PATH = PROJECT_ROOT
CACHE_FILE_REL_PATH = Path("data/destination_fingerprints.json")
LOCAL_CACHE_FILE = REPO_PATH / CACHE_FILE_REL_PATH
FINGERPRINTER_PARALLEL_JOBS = 8
# --- END CONFIGURATION ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("deduplicate.log"), logging.StreamHandler()],
)


def _compare_accession_numbers(name1: str, name2: str) -> str:
    """
    compare two exam names and return the one with the most recent accession number.

    assumes exam names contain accession numbers and later accession numbers are
    lexicographically greater (or contain higher numeric values).
    """
    return name1 if name1 > name2 else name2


def update_destination_inventory(parallel_jobs: int):
    """
    run fingerprinter locally on destination to update inventory cache

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
    """load destination inventory from cache file"""
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


def find_duplicate_groups(
    patient_inventory: Dict[ExamFingerprint, str],
) -> List[Set[str]]:
    """
    find groups of duplicate exams for a patient based on UID.

    returns list of sets, where each set contains exam names that are duplicates
    """
    # group by UID - exams with the same UID are duplicates
    uid_to_names: Dict[str, Set[str]] = defaultdict(set)
    for fp, name in patient_inventory.items():
        uid_to_names[fp.study_uid].add(name)

    # return groups with more than one exam (duplicates)
    return [names for names in uid_to_names.values() if len(names) > 1]


def deduplicate_patient(
    patient_id: str,
    patient_inventory: Dict[ExamFingerprint, str],
    dry_run: bool,
    delete: bool,
) -> Tuple[int, int]:
    """
    find and remove duplicate exams for a single patient.

    by default, moves duplicates to STAGING_DIR for manual review.
    if delete=True, permanently removes them instead.

    returns (duplicates_found, duplicates_removed)
    """
    duplicate_groups = find_duplicate_groups(patient_inventory)
    if not duplicate_groups:
        return 0, 0

    duplicates_found = sum(len(g) for g in duplicate_groups)
    duplicates_removed = 0

    for group in duplicate_groups:
        if len(group) < 2:
            continue

        exam_names = sorted(group)
        # find the one with the most recent accession number
        preferred = exam_names[0]
        for name in exam_names[1:]:
            preferred = _compare_accession_numbers(preferred, name)

        to_remove = [n for n in exam_names if n != preferred]

        logging.info(
            f"Patient {patient_id}: found {len(exam_names)} duplicate exams: {exam_names}"
        )
        logging.info(f"  Keeping: {preferred}")
        logging.info(f"  Removing: {to_remove}")

        for exam_name in to_remove:
            exam_path = DST_ROOT / patient_id / exam_name
            if not exam_path.exists():
                logging.warning(f"  Exam path does not exist: {exam_path}")
                continue

            if dry_run:
                action = "delete" if delete else f"move to {STAGING_DIR}"
                logging.info(f"  [DRY RUN] Would {action}: {exam_path}")
            elif delete:
                try:
                    shutil.rmtree(exam_path)
                    logging.info(f"  Deleted: {exam_path}")
                    duplicates_removed += 1
                except Exception as e:
                    logging.error(f"  Failed to delete {exam_path}: {e}")
            else:
                # move to staging directory, preserving patient/exam structure
                staging_path = STAGING_DIR / patient_id / exam_name
                try:
                    staging_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(exam_path), str(staging_path))
                    logging.info(f"  Moved to staging: {exam_path} -> {staging_path}")
                    duplicates_removed += 1
                except Exception as e:
                    logging.error(f"  Failed to move {exam_path} to staging: {e}")

    return duplicates_found, duplicates_removed


def main(
    dry_run: bool,
    delete: bool,
    update_fingerprints: bool,
    fingerprint_parallel_jobs: int,
):
    """main entry point for deduplication"""
    logging.info("=== STARTING DEDUPLICATION ===")
    if delete:
        logging.warning("DELETE MODE: duplicates will be permanently removed!")
    else:
        logging.info(f"STAGING MODE: duplicates will be moved to {STAGING_DIR}")

    if update_fingerprints:
        logging.info("Step 1: Updating destination inventory...")
        update_destination_inventory(fingerprint_parallel_jobs)
    else:
        logging.info(
            "Skipping fingerprint update (use --update-fingerprints to enable)"
        )

    logging.info("Step 2: Loading destination inventory...")
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

    logging.info("Step 3: Scanning destination for patient directories...")
    scan_start = monotonic()
    patient_dirs = [d for d in DST_ROOT.iterdir() if d.is_dir() and d.name.isdigit()]
    scan_time = monotonic() - scan_start
    logging.info(f"Found {len(patient_dirs)} patient directories in {scan_time:.1f}s")

    logging.info("Step 4: Finding and removing duplicates...")
    total_duplicates_found = 0
    total_duplicates_removed = 0
    patients_with_duplicates = 0

    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        patient_id = patient_dir.name
        patient_inventory = dest_inventory.get(patient_id, {})
        if not patient_inventory:
            continue

        found, removed = deduplicate_patient(
            patient_id, patient_inventory, dry_run, delete
        )
        if found > 0:
            patients_with_duplicates += 1
            total_duplicates_found += found
            total_duplicates_removed += removed

    logging.info("=== DEDUPLICATION COMPLETE ===")
    logging.info(f"Patients with duplicates: {patients_with_duplicates}")
    logging.info(f"Total duplicate exams found: {total_duplicates_found}")
    action = "deleted" if delete else "moved to staging"
    logging.info(f"Total duplicate exams {action}: {total_duplicates_removed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find and remove duplicate exams in destination directory."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate all operations; no files will be moved or removed.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Permanently delete duplicates instead of moving to staging directory.",
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
        dry_run=args.dry_run,
        delete=args.delete,
        update_fingerprints=args.update_fingerprints,
        fingerprint_parallel_jobs=args.fingerprint_parallel_jobs,
    )
