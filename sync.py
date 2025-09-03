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
    """
    Executes a command on the remote server via SSH. This is a non-interactive,
    'fire-and-forget' execution that waits for completion.
    """
    MICROMAMBA_EXECUTABLE = (
        "/gpfs/data/huo-lab/Image/annawoodard/bin/micromamba"
    )
    MAMBA_ROOT_PREFIX = (
        "/gpfs/data/huo-lab/Image/annawoodard/micromamba"
    )
    init_command = f"export MAMBA_ROOT_PREFIX='{MAMBA_ROOT_PREFIX}'"
    shell_hook_command = (
        f'eval "$({MICROMAMBA_EXECUTABLE} shell hook -s posix)"'
    )
    mamba_activate_command = 'micromamba activate prima'
    final_command_str = " ".join(command)
    remote_command_wrapper = (
        f"set -e && "
        f"{init_command} && "
        f"{shell_hook_command} && "
        f"{mamba_activate_command} && "
        f"{final_command_str}"
    )

    # NO '-t' flag. This is a non-interactive execution.
    full_ssh_command = [
        "ssh", DST_SSH_TARGET, "bash", "-c", remote_command_wrapper
    ]
    try:
        process = subprocess.run(
            full_ssh_command, check=check, capture_output=True, text=True,
        )
        # Log the final stderr at the end for debugging purposes.
        if process.stderr:
            logging.debug(f"Remote stderr:\n{process.stderr.strip()}")
        return process.stdout, process.stderr
    except subprocess.CalledProcessError as e:
        logging.error(
            f"SSH command failed. Full command executed on remote: "
            f"'{remote_command_wrapper}'"
        )
        logging.error(f"  Return Code: {e.returncode}")
        logging.error(f"  --- Remote Stdout --- \n{e.stdout.strip()}")
        logging.error(f"  --- Remote Stderr --- \n{e.stderr.strip()}")
        raise


def update_remote_inventory():
    """
    Ensures the remote fingerprinter is up-to-date, runs it on the server,
    and downloads the resulting inventory. Progress should be monitored by
    tailing the log file on the remote server.
    """
    logging.info("--- Updating and Running Remote Inventory Script ---")
    remote_script = REMOTE_REPO_PATH / "fingerprinter.py"
    remote_cache = REMOTE_REPO_PATH / CACHE_FILE_REL_PATH
    remote_log_file = REMOTE_REPO_PATH / "data/fingerprinter.log"

    logging.info(f"Running 'git pull' in remote directory: {REMOTE_REPO_PATH}")
    git_command = f"cd {REMOTE_REPO_PATH} && git pull"
    run_ssh_command([git_command])

    logging.info(
        "Starting remote fingerprinting process... This will take a long time."
    )
    print("\n" + "="*70)
    print("  MONITORING INSTRUCTIONS:")
    print("  In a separate terminal, run the following command:")
    print(f"  ssh {DST_SSH_TARGET} 'tail -f {remote_log_file}'")
    print("="*70 + "\n")
    # Run the python script and redirect all its output (stdout and stderr)
    # to the log file. The 'nohup' command and '&' are not strictly necessary
    # here since the Python script will block until the SSH command completes,
    # but it's good practice for long-running jobs.
    python_command = (
        f"python -u {remote_script} "  # '-u' for unbuffered output
        f"{DST_ROOT_REMOTE} {remote_cache} "
        f"--parallel-jobs {PARALLEL_JOBS} "
        f"> {remote_log_file} 2>&1"  # Redirect stdout and stderr to log
    )

    try:
        run_ssh_command([python_command])
        logging.info("Remote fingerprinting completed successfully.")
    except subprocess.CalledProcessError:
        logging.error("Remote fingerprinting failed!")
        logging.error(
            f"Check the detailed log on the server for errors: "
            f"{remote_log_file}"
        )
        raise

    logging.info("Downloading remote inventory cache...")
    LOCAL_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["scp", f"{DST_SSH_TARGET}:{remote_cache}", str(LOCAL_CACHE_FILE)],
        check=True
    )
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
