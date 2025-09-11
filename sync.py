#!/usr/bin/env python

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
REMOTE_REPO_PATH = Path("/gpfs/data/huo-lab/Image/annawoodard/prima")

# --- LOCAL AND REMOTE CACHE PATHS ---
# Relative path for the cache file INSIDE the repo
CACHE_FILE_REL_PATH = Path("data/destination_fingerprints.json")
# Full path to the local cache file
LOCAL_CACHE_FILE = Path(__file__).resolve().parent / CACHE_FILE_REL_PATH

# Path to the delete queue on the source share
DELETE_QUEUE_DIR = SRC_ROOT / "_synced_and_queued_for_deletion"
PARALLEL_JOBS = 8
REMOTE_PARALLEL_JOBS = 4  # NEW: For REMOTE processing. Start with 4, maybe try 2.
# --- END CONFIGURATION ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("sync.log"), logging.StreamHandler()],
)


def run_ssh_command(command_str: str, check: bool = True):
    """
    execute a complete command string on the remote via SSH.

    important: ssh joins arguments with spaces before handing them to the remote shell.
    to ensure the entire payload is seen as the single argument to `bash -c`, we wrap
    it in single quotes and escape any embedded single quotes.
    """
    # wrap as one argument to bash -c; escape any single quotes inside
    quoted_cmd = "'" + command_str.replace("'", "'\"'\"'") + "'"
    full_ssh_command = ["ssh", DST_SSH_TARGET, "bash", "-c", quoted_cmd]

    try:
        process = subprocess.run(
            full_ssh_command,
            check=check,
            capture_output=True,
            text=True,
        )
        if process.stderr:
            logging.debug(f"Remote stderr:\n{process.stderr.strip()}")
        return process.stdout, process.stderr

    except subprocess.CalledProcessError as e:
        logging.error(
            f"SSH command failed. Full command executed on remote: {quoted_cmd}"
        )
        logging.error(f"  Return Code: {e.returncode}")
        logging.error(f"  --- Remote Stdout --- \n{(e.stdout or '').strip()}")
        logging.error(f"  --- Remote Stderr --- \n{(e.stderr or '').strip()}")
        raise


def update_remote_inventory():
    """
    print instructions to run the remote fingerprinter manually, then ask the user to confirm
    completion before fetching the resulting cache file via scp; fails loudly on errors
    """
    logging.info("--- remote inventory: manual run, interactive fetch ---")

    remote_script = REMOTE_REPO_PATH / "fingerprinter.py"
    remote_cache = REMOTE_REPO_PATH / CACHE_FILE_REL_PATH
    remote_log_file = REMOTE_REPO_PATH / "data/fingerprinter.log"

    MICROMAMBA_EXECUTABLE = "/gpfs/data/huo-lab/Image/annawoodard/bin/micromamba"
    MAMBA_ROOT_PREFIX = "/gpfs/data/huo-lab/Image/annawoodard/micromamba"

    print("\n" + "=" * 78)
    print("RUN THESE ON THE REMOTE HOST")
    print("-" * 78)
    print(f"ssh {DST_SSH_TARGET}")
    print(f"cd {REMOTE_REPO_PATH} && git pull --ff-only")
    print(f"export MAMBA_ROOT_PREFIX='{MAMBA_ROOT_PREFIX}'")
    print(f'eval "$({MICROMAMBA_EXECUTABLE} shell hook -s posix)"')
    print("micromamba activate prima")
    print(
        f"python -u {remote_script} {DST_ROOT_REMOTE} {remote_cache} "
        f"--parallel-jobs {REMOTE_PARALLEL_JOBS} > {remote_log_file} 2>&1 &"
    )
    print(f"tail -f {remote_log_file}")
    print("=" * 78)
    print("when the remote run completes and the cache exists at:")
    print(f"  {remote_cache}")
    print("type 'y' to fetch it now; anything else to skip.")
    print("=" * 78 + "\n")

    answer = (
        input("has the remote fingerprinting finished? fetch cache now [y/N]: ")
        .strip()
        .lower()
    )
    if answer not in {"y", "yes"}:
        logging.info("skipping cache fetch (user did not confirm completion)")
        return

    logging.info("attempting to fetch remote inventory cache via scp...")
    LOCAL_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    # pull the cache; if it's not there yet, this will raise and stop the run
    subprocess.run(
        ["scp", f"{DST_SSH_TARGET}:{remote_cache}", str(LOCAL_CACHE_FILE)],
        check=True,
        capture_output=True,
        text=True,
    )

    size_mb = LOCAL_CACHE_FILE.stat().st_size / 1_000_000
    logging.info(f"downloaded {LOCAL_CACHE_FILE} ({size_mb:.2f} MB)")

    # quick sanity summary; will raise if the file is not valid JSON
    with open(LOCAL_CACHE_FILE) as f:
        data = json.load(f)
    n_patients = len(data)
    n_exams = sum(len(v) for v in data.values())
    logging.info(f"inventory snapshot: {n_patients:,} patients, {n_exams:,} exams")


def load_destination_inventory() -> Dict[str, Dict[ExamFingerprint, str]]:
    if not LOCAL_CACHE_FILE.exists():
        logging.error(f"Local cache file not found: {LOCAL_CACHE_FILE}")
        logging.error("Run with --update-inventory to generate it.")
        return {}

    logging.info(f"Loading destination inventory from {LOCAL_CACHE_FILE}...")
    with open(LOCAL_CACHE_FILE) as f:
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
    """rsync an exam directory to the remote host; returns (net_bytes, logical_bytes, seconds) for throughput accounting"""
    import re
    from time import monotonic

    dest_parent = DST_ROOT_REMOTE / patient_id
    dest_dir = DST_ROOT_REMOTE / patient_id / exam_name
    dest_path_str = f"{DST_SSH_TARGET}:{dest_dir}"
    logging.info(f"RSYNC: {src} -> {dest_path_str}")

    # create parent dir remotely, then exec rsync; avoids brittle ssh 'bash -c' quoting
    rsync_remote_prefix = f"mkdir -p {dest_parent} && rsync"

    command = [
        "rsync",
        "-aH",
        "--partial",
        "--info=stats2,progress2",
        "--stats",
        "--rsync-path",
        rsync_remote_prefix,
        f"{src}/",
        dest_path_str,
    ]

    t0 = monotonic()
    proc = subprocess.run(command, check=True, capture_output=True, text=True)
    dt = monotonic() - t0

    # combine both streams; different rsync builds print stats on either
    combined = f"{proc.stdout or ''}\n{proc.stderr or ''}"

    # parse network bytes actually moved: "sent X bytes  received Y bytes"
    sent_bytes = 0
    recv_bytes = 0
    m_sent = re.search(r"sent\s+([\d,]+)\s+bytes", combined, flags=re.I)
    m_recv = re.search(r"received\s+([\d,]+)\s+bytes", combined, flags=re.I)
    if m_sent:
        sent_bytes = int(m_sent.group(1).replace(",", ""))
    if m_recv:
        recv_bytes = int(m_recv.group(1).replace(",", ""))
    net_bytes = sent_bytes + recv_bytes

    # parse logical payload size (size of files considered / written)
    logical_bytes = 0
    for pat in (
        r"Total transferred file size:\s*([\d,]+)",
        r"Total file size:\s*([\d,]+)",
        r"total size is\s*([\d,]+)",
    ):
        m = re.search(pat, combined, flags=re.I)
        if m:
            logical_bytes = int(m.group(1).replace(",", ""))
            break

    # if we couldn't get a logical size, fall back to net bytes
    if logical_bytes == 0:
        logical_bytes = net_bytes

    mb_log = logical_bytes / 1_000_000
    mb_net = net_bytes / 1_000_000
    rate_net = (net_bytes / dt / 1_000_000) if dt > 0 else 0.0
    logging.info(
        f"RSYNC successful for {src.name} [logical {mb_log:.1f} MB, net {mb_net:.1f} MB in {dt:.1f}s, {rate_net:.2f} MB/s net]"
    )
    return net_bytes, logical_bytes, dt


def main(update_inventory: bool, dry_run: bool):
    """drive end-to-end sync of source exams to remote, with explicit remote before/after and verification"""
    from time import monotonic

    if dry_run:
        logging.info(
            "--- DRY RUN MODE ENABLED: No files will be moved or transferred. ---"
        )

    if update_inventory:
        update_remote_inventory()
        return

    dest_inventory = load_destination_inventory()
    if not dest_inventory:
        return

    # quick inventory summary
    inv_patients = len(dest_inventory)
    inv_exams = sum(len(exams) for exams in dest_inventory.values())
    logging.info(
        f"Destination inventory summary: {inv_patients:,} patients, {inv_exams:,} exams"
    )

    logging.info(f"Discovering source exams in {SRC_ROOT}...")
    source_exams = [
        p
        for p_dir in SRC_ROOT.iterdir()
        if p_dir.is_dir() and not p_dir.name.startswith("_")
        for p in p_dir.iterdir()
        if p.is_dir()
    ]
    logging.info(f"Found {len(source_exams)} source exams to process.")

    if not dry_run:
        DELETE_QUEUE_DIR.mkdir(exist_ok=True)

    stats = {
        "renamed": 0,
        "transferred": 0,
        "skipped": 0,
        "failed": 0,
        "remote_renamed": 0,
    }

    # heartbeat setup (emits even if no futures finish)
    HEARTBEAT_SEC = 60
    t_start = monotonic()
    last_hb = t_start
    bytes_net_total = 0
    bytes_log_total = 0
    transfers_done = 0
    processed = 0  # exams we fully handled (any outcome)

    logging.info("Processing source exams in parallel...")

    with ProcessPoolExecutor(max_workers=PARALLEL_JOBS) as executor:
        futures = {
            executor.submit(create_exam_fingerprint, path): path
            for path in source_exams
        }
        pending = set(futures.keys())
        total = len(pending)

        pbar = tqdm(total=total, desc="Processing Source Exams (Parallel)")

        while pending:
            try:
                for future in as_completed(pending, timeout=HEARTBEAT_SEC):
                    pending.remove(future)
                    src_path = futures[future]

                    try:
                        src_fingerprint, reason = future.result()

                        if not src_fingerprint or not src_fingerprint.is_valid():
                            logging.warning(f"SKIPPED: {src_path} - Reason: {reason}")
                            stats["skipped"] += 1
                            processed += 1
                        else:
                            patient_id, new_exam_name = (
                                src_path.parent.name,
                                src_path.name,
                            )
                            match_name = dest_inventory.get(patient_id, {}).get(
                                src_fingerprint
                            )
                            delete_queue_path = (
                                DELETE_QUEUE_DIR / patient_id / new_exam_name
                            )

                            if match_name:
                                old_rem = DST_ROOT_REMOTE / patient_id / match_name
                                new_rem = DST_ROOT_REMOTE / patient_id / new_exam_name

                                if match_name != new_exam_name:
                                    # show BEFORE
                                    if not dry_run:
                                        pre = subprocess.run(
                                            [
                                                "ssh",
                                                DST_SSH_TARGET,
                                                "ls",
                                                "-ld",
                                                str(old_rem),
                                            ],
                                            capture_output=True,
                                            text=True,
                                        )
                                        if pre.returncode == 0:
                                            logging.info(
                                                f"REMOTE BEFORE: {pre.stdout.strip()}"
                                            )
                                        else:
                                            logging.warning(
                                                f"REMOTE BEFORE missing: {old_rem} ({pre.stderr.strip()})"
                                            )

                                    logging.info(
                                        f"MATCH (remote found): fingerprint equals remote '{match_name}' "
                                        f"→ renaming REMOTE to '{new_exam_name}' and queuing LOCAL for deletion"
                                    )

                                    if dry_run:
                                        logging.info(
                                            f"[DRY RUN] WOULD RENAME REMOTE: {old_rem} -> {new_rem}"
                                        )
                                    else:
                                        subprocess.run(
                                            [
                                                "ssh",
                                                DST_SSH_TARGET,
                                                "mv",
                                                str(old_rem),
                                                str(new_rem),
                                            ],
                                            check=True,
                                            capture_output=True,
                                            text=True,
                                        )
                                        # show AFTER and verify
                                        post = subprocess.run(
                                            [
                                                "ssh",
                                                DST_SSH_TARGET,
                                                "ls",
                                                "-ld",
                                                str(new_rem),
                                            ],
                                            capture_output=True,
                                            text=True,
                                        )
                                        if post.returncode != 0:
                                            logging.error(
                                                f"REMOTE VERIFY FAILED after rename: {new_rem} "
                                                f"({post.stderr.strip()})"
                                            )
                                            raise RuntimeError(
                                                "remote rename verify failed"
                                            )
                                        logging.info(
                                            f"REMOTE AFTER:  {post.stdout.strip()}"
                                        )

                                        gone = subprocess.run(
                                            [
                                                "ssh",
                                                DST_SSH_TARGET,
                                                "ls",
                                                "-ld",
                                                str(old_rem),
                                            ],
                                            capture_output=True,
                                            text=True,
                                        )
                                        if gone.returncode == 0:
                                            logging.warning(
                                                f"REMOTE old path still visible after mv: {gone.stdout.strip()}"
                                            )

                                        stats["remote_renamed"] += 1
                                    stats["renamed"] += 1
                                    processed += 1
                                else:
                                    # names already match; verify remote exists before deleting local
                                    final_rem = new_rem
                                    if dry_run:
                                        logging.info(
                                            f"MATCH (remote found): {new_exam_name} already exists on remote; "
                                            f"[DRY RUN] WOULD VERIFY and queue LOCAL for deletion"
                                        )
                                    else:
                                        ver = subprocess.run(
                                            [
                                                "ssh",
                                                DST_SSH_TARGET,
                                                "ls",
                                                "-ld",
                                                str(final_rem),
                                            ],
                                            capture_output=True,
                                            text=True,
                                        )
                                        if ver.returncode != 0:
                                            logging.error(
                                                f"REMOTE VERIFY FAILED (no rename): expected {final_rem} "
                                                f"({ver.stderr.strip()})"
                                            )
                                            raise RuntimeError(
                                                "remote verify failed (no-rename match)"
                                            )
                                        logging.info(
                                            f"REMOTE PRESENT: {ver.stdout.strip()}"
                                        )

                                    logging.info(
                                        f"MATCH (remote found): {new_exam_name} already exists on remote; "
                                        f"queuing LOCAL for deletion"
                                    )
                                    stats["renamed"] += 1
                                    processed += 1
                            else:
                                logging.info(
                                    f"NEW: {new_exam_name} not found on remote. Transferring."
                                )
                                if dry_run:
                                    logging.info(
                                        f"[DRY RUN] WOULD TRANSFER: {src_path} -> "
                                        f"{DST_ROOT_REMOTE / patient_id / new_exam_name}"
                                    )
                                    stats["transferred"] += 1
                                    processed += 1
                                else:
                                    b_net, b_log, _ = rsync_exam_remote(
                                        src_path, patient_id, new_exam_name
                                    )
                                    bytes_net_total += b_net
                                    bytes_log_total += b_log
                                    transfers_done += 1
                                    stats["transferred"] += 1
                                    processed += 1

                            # move source into delete queue only after remote verified/renamed/transferred
                            if dry_run:
                                logging.info(
                                    f"[DRY RUN] WOULD QUEUE FOR DELETE (LOCAL): {src_path} -> {delete_queue_path}"
                                )
                            else:
                                delete_queue_path.parent.mkdir(
                                    parents=True, exist_ok=True
                                )
                                shutil.move(str(src_path), str(delete_queue_path))
                                logging.info(
                                    f"QUEUED FOR DELETE (LOCAL): moved {src_path} -> {delete_queue_path}"
                                )

                    except Exception:
                        logging.error(
                            f"FATAL ERROR processing {src_path}", exc_info=True
                        )
                        stats["failed"] += 1
                        processed += 1

                    pbar.update(1)

                # heartbeat (also when we had completions)
                now = monotonic()
                if now - last_hb >= HEARTBEAT_SEC:
                    elapsed = now - t_start
                    mb_log_total = bytes_log_total / 1_000_000
                    mb_net_total = bytes_net_total / 1_000_000
                    mbps_net = (
                        (bytes_net_total / elapsed / 1_000_000) if elapsed > 0 else 0.0
                    )
                    finger_done = total - len(pending)
                    backlog = max(0, finger_done - processed)
                    inflight = min(PARALLEL_JOBS, len(pending))

                    logging.info(
                        f"[HEARTBEAT] processed {processed}/{total} | "
                        f"{transfers_done} transferred (logical {mb_log_total:.1f} MB, net {mb_net_total:.1f} MB), "
                        f"{stats['remote_renamed']} remote-renamed | "
                        f"overall {(processed / elapsed * 3600.0) if elapsed > 0 else 0.0:.2f} exams/hr, "
                        f"{mbps_net:.2f} MB/s net | "
                        f"fingerprinting done {finger_done}/{total}, backlog {backlog}, active {inflight}"
                    )
                    last_hb = now

            except Exception as e:
                # timeout -> emit heartbeat anyway
                if e.__class__.__name__ == "TimeoutError":
                    now = monotonic()
                    elapsed = now - t_start
                    finger_done = total - len(pending)
                    backlog = max(0, finger_done - processed)
                    inflight = min(PARALLEL_JOBS, len(pending))
                    mbps_net = (
                        (bytes_net_total / elapsed / 1_000_000) if elapsed > 0 else 0.0
                    )
                    logging.info(
                        f"[HEARTBEAT] processed {processed}/{total} | "
                        f"{transfers_done} transferred (logical {bytes_log_total / 1_000_000:.1f} MB, "
                        f"net {bytes_net_total / 1_000_000:.1f} MB), "
                        f"{stats['remote_renamed']} remote-renamed | "
                        f"overall {(processed / elapsed * 3600.0) if elapsed > 0 else 0.0:.2f} exams/hr, "
                        f"{mbps_net:.2f} MB/s net | "
                        f"fingerprinting done {finger_done}/{total}, backlog {backlog}, active {inflight}"
                    )
                    last_hb = now
                    continue
                raise

        pbar.close()

    logging.info("--- Sync Complete ---")
    if dry_run:
        logging.info("--- (DRY RUN MODE) ---")
    logging.info(
        f"Renamed/Matched: {stats['renamed']}  (remote-renamed: {stats['remote_renamed']})"
    )
    logging.info(f"Transferred New: {stats['transferred']}")
    logging.info(f"Skipped (bad source): {stats['skipped']}")
    logging.info(f"Failed: {stats['failed']}")
    if not dry_run:
        logging.info(f"Processed source exams moved to: {DELETE_QUEUE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Intelligently sync DICOM exams over SSH."
    )
    parser.add_argument(
        "--update-inventory",
        action="store_true",
        help="Run the fingerprinter on the remote server to update "
        "the destination inventory cache.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate all operations; no files will be moved or transferred.",
    )
    args = parser.parse_args()
    main(update_inventory=args.update_inventory, dry_run=args.dry_run)
