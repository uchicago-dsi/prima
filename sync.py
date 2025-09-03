#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import logging
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

from tqdm import tqdm

# Import the shared logic
from fingerprint_utils import ExamFingerprint, create_exam_fingerprint

# --- CONFIGURATION ---
SRC_ROOT = Path("/Volumes/16352A")
DST_SSH_TARGET = "annawoodard@cri-ksysappdsp3.cri.uchicago.edu"
DST_ROOT_REMOTE = Path("/gpfs/data/huo-lab/Image/ChiMEC")

# --- GIT-AWARE PATH CONFIGURATION ---
# The path to your Git repository ON THE REMOTE SERVER
REMOTE_REPO_PATH = Path(
    "/gpfs/data/huo-lab/Image/annawoodard/prima"
)

# --- LOCAL AND REMOTE CACHE PATHS ---
# Relative path for the cache file INSIDE the repo
CACHE_FILE_REL_PATH = Path("data/destination_fingerprints.json")
# Full path to the local cache file
LOCAL_CACHE_FILE = Path(__file__).resolve().parent / CACHE_FILE_REL_PATH

# Path to the delete queue on the source share
DELETE_QUEUE_DIR = SRC_ROOT / "_synced_and_queued_for_deletion"
PARALLEL_JOBS = 8
# --- END CONFIGURATION ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("sync.log"),
        logging.StreamHandler()
    ]
)


def run_ssh_command(command: list, check: bool = True):
    full_command = ["ssh", DST_SSH_TARGET] + command
    try:
        process = subprocess.run(
            full_command, check=check, capture_output=True, text=True
        )
        return process.stdout, process.stderr
    except subprocess.CalledProcessError as e:
        logging.error(f"SSH command failed: {' '.join(e.cmd)}")
        logging.error(f"Stderr: {e.stderr.strip()}")
        raise


def update_remote_inventory():
    logging.info("--- Updating and Running Remote Inventory Script ---")

    # 1. Ensure remote repository is up-to-date
    logging.info(
        f"Running 'git pull' in remote directory: {REMOTE_REPO_PATH}"
    )
    # Note: This assumes you have passwordless SSH or an agent set up.
    # The 'cd' and 'git pull' are combined to ensure it runs in the
    # right place.
    git_command = f"cd {REMOTE_REPO_PATH} && git pull"
    stdout, stderr = run_ssh_command([git_command])
    if "Already up to date." not in stdout and stdout:
        logging.info(f"Git pull output:\n{stdout.strip()}")
    if stderr:
        logging.warning(f"Git pull stderr:\n{stderr.strip()}")

    # 2. Define remote script and cache paths based on the repo path
    remote_script = REMOTE_REPO_PATH / "fingerprinter.py"
    remote_cache = REMOTE_REPO_PATH / CACHE_FILE_REL_PATH

    # 3. Run the remote script
    logging.info(
        "Running fingerprinter on remote server (this may take a while)..."
    )
    cmd = [
        "python", str(remote_script),
        str(DST_ROOT_REMOTE), str(remote_cache),
        "--parallel-jobs", str(PARALLEL_JOBS)
    ]
    _, stderr = run_ssh_command(cmd)
    if stderr:
        logging.info(f"Remote script output:\n{stderr.strip()}")

    # 4. Download the inventory
    logging.info("Downloading remote inventory cache...")
    LOCAL_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "scp", f"{DST_SSH_TARGET}:{remote_cache}", str(LOCAL_CACHE_FILE)
    ], check=True)
    logging.info("Remote inventory updated successfully.")


def load_destination_inventory() -> Dict[str, Dict[ExamFingerprint, str]]:
    if not LOCAL_CACHE_FILE.exists():
        logging.error(f"Local cache file not found: {LOCAL_CACHE_FILE}")
        logging.error("Run with --update-inventory to generate it.")
        return {}

    logging.info(f"Loading destination inventory from {LOCAL_CACHE_FILE}...")
    with open(LOCAL_CACHE_FILE, 'r') as f:
        raw_data = json.load(f)
        inventory = {}
        for patient_id, exams in raw_data.items():
            inventory[patient_id] = {
                ExamFingerprint(uid, frozenset(hashes)): name
                for name, (uid, hashes) in exams.items()
            }
    logging.info("Destination inventory loaded.")
    return inventory


def rsync_exam_remote(src: Path, patient_id: str, exam_name: str):
    dest_path_str = (
        f"{DST_SSH_TARGET}:{DST_ROOT_REMOTE / patient_id / exam_name}"
    )
    logging.info(f"RSYNC: {src} -> {dest_path_str}")
    run_ssh_command([
        "mkdir", "-p", str(DST_ROOT_REMOTE / patient_id)
    ])
    command = [
        "rsync", "-aH", "--partial", "--info=stats2,progress2",
        f"{src}/", dest_path_str
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    logging.info(f"RSYNC successful for {src.name}")


def main(update_inventory: bool, dry_run: bool):
    if dry_run:
        logging.info(
            "--- DRY RUN MODE ENABLED: "
            "No files will be moved or transferred. ---"
        )
    if update_inventory:
        update_remote_inventory()

    dest_inventory = load_destination_inventory()
    if not dest_inventory:
        return

    logging.info(f"Discovering source exams in {SRC_ROOT}...")
    source_exams = [
        p for p_dir in SRC_ROOT.iterdir()
        if p_dir.is_dir() and not p_dir.name.startswith('_')
        for p in p_dir.iterdir() if p.is_dir()
    ]
    logging.info(f"Found {len(source_exams)} source exams to process.")

    if not dry_run:
        DELETE_QUEUE_DIR.mkdir(exist_ok=True)

    stats = {"renamed": 0, "transferred": 0, "skipped": 0, "failed": 0}

    with ProcessPoolExecutor(max_workers=PARALLEL_JOBS) as executor:
        futures = {
            executor.submit(create_exam_fingerprint, path): path
            for path in source_exams
        }
        for future in tqdm(
            as_completed(futures), total=len(futures),
            desc="Processing Source Exams"
        ):
            src_path = futures[future]
            try:
                src_fingerprint = future.result()
                if not src_fingerprint or not src_fingerprint.is_valid():
                    logging.warning(
                        f"SKIPPED: Invalid fingerprint for {src_path}"
                    )
                    stats["skipped"] += 1
                    continue

                patient_id, new_exam_name = (
                    src_path.parent.name, src_path.name
                )
                match_name = dest_inventory.get(
                    patient_id, {}
                ).get(src_fingerprint)
                delete_queue_path = (
                    DELETE_QUEUE_DIR / patient_id / new_exam_name
                )

                if match_name:
                    if match_name != new_exam_name:
                        logging.info(
                            f"MATCH: {new_exam_name} is identical to "
                            f"remote {match_name}."
                        )
                        old_rem, new_rem = (
                            DST_ROOT_REMOTE/patient_id/match_name,
                            DST_ROOT_REMOTE/patient_id/new_exam_name
                        )
                        if dry_run:
                            logging.info(
                                f"[DRY RUN] WOULD RENAME REMOTE: "
                                f"{old_rem} -> {new_rem}"
                            )
                        else:
                            run_ssh_command(
                                ["mv", str(old_rem), str(new_rem)]
                            )
                            logging.info("RENAME successful on remote.")
                    else:
                        logging.info(
                            f"MATCH: {new_exam_name} already exists "
                            "and is up to date."
                        )
                    stats["renamed"] += 1
                else:
                    logging.info(
                        f"NEW: {new_exam_name} not found. Transferring."
                    )
                    if dry_run:
                        logging.info(
                            f"[DRY RUN] WOULD TRANSFER: {src_path} -> "
                            f"{DST_ROOT_REMOTE / patient_id / new_exam_name}"
                        )
                    else:
                        rsync_exam_remote(src_path, patient_id, new_exam_name)
                    stats["transferred"] += 1

                if dry_run:
                    logging.info(
                        f"[DRY RUN] WOULD QUEUE FOR DELETE: "
                        f"{src_path} -> {delete_queue_path}"
                    )
                else:
                    delete_queue_path.parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    shutil.move(str(src_path), str(delete_queue_path))

            except Exception as e:
                logging.error(f"FATAL ERROR processing {src_path}: {e}")
                stats["failed"] += 1

    logging.info("--- Sync Complete ---")
    if dry_run:
        logging.info("--- (DRY RUN MODE) ---")
    logging.info(f"Renamed/Matched: {stats['renamed']}")
    logging.info(f"Transferred New: {stats['transferred']}")
    logging.info(f"Skipped (bad source): {stats['skipped']}")
    logging.info(f"Failed: {stats['failed']}")
    if not dry_run:
        logging.info(
            f"Processed source exams moved to: {DELETE_QUEUE_DIR}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Intelligently sync DICOM exams over SSH."
    )
    parser.add_argument(
        "--update-inventory", action="store_true",
        help="Run the fingerprinter on the remote server to update "
        "the destination inventory cache."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate all operations; no files will be moved or "
             "transferred."
    )
    args = parser.parse_args()
    main(update_inventory=args.update_inventory, dry_run=args.dry_run)
