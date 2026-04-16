#!/usr/bin/env python
# ruff: noqa: E402

import argparse
import json
import logging
import shutil
import signal
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prima.fingerprint_utils import ExamFingerprint

# suppress pydicom VR UI validation warnings for non-standard UIDs
warnings.filterwarnings("ignore", message=".*Invalid value for VR UI.*", append=True)
# also set pydicom logging level to suppress warnings at the source
logging.getLogger("pydicom.valuerep").setLevel(logging.ERROR)

# --- CONFIGURATION ---
SRC_ROOT = Path("/Volumes/16352A")
DST_SSH_TARGET = "annawoodard@cri-ksysappdsp3.cri.uchicago.edu"
DST_ROOT_REMOTE = Path("/gpfs/data/huo-lab/Image/ChiMEC/MG")

# --- GIT-AWARE PATH CONFIGURATION ---
# The path to your Git repository ON THE REMOTE SERVER
REMOTE_REPO_PATH = Path("/gpfs/data/huo-lab/Image/annawoodard/prima")

# --- LOCAL AND REMOTE CACHE PATHS ---
# Relative path for the cache file INSIDE the repo
CACHE_FILE_REL_PATH = Path("data/destination_fingerprints.json")
# Full path to the local cache file
LOCAL_CACHE_FILE = PROJECT_ROOT / CACHE_FILE_REL_PATH

# Path to the delete queue on the source share
DELETE_QUEUE_DIR = SRC_ROOT / "_synced_and_queued_for_deletion"
PARALLEL_JOBS = 1
REMOTE_PARALLEL_JOBS = 4  # For REMOTE processing. Start with 4, maybe try 2.
# File stability check: only process exam dirs where most recent file is older than this
STABILITY_THRESHOLD_SEC = 600
# Auto-restart configuration
RESTART_DELAY_SEC = 120  # Wait 60 seconds between restarts
# Cap how many exam directories we fully handle in a single run.
# Set to None to disable the cap.
MAX_EXAMS_PER_RUN: int | None = 50
# --- END CONFIGURATION ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("sync.log"), logging.StreamHandler()],
)

# Global flag for graceful shutdown
shutdown_requested = False


def _log_remote_debug(ver_target: Path, ssh_opts: list[str]) -> None:
    """Log remote directory state to help debug verification failures."""

    remote_parent = ver_target.parent
    try:
        list_cmd = subprocess.run(
            ["ssh"] + ssh_opts + [DST_SSH_TARGET, "ls", "-l", str(remote_parent)],
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.error(
            "Diagnostic ssh listing for %s failed to launch: %s",
            remote_parent,
            exc,
        )
        return

    if list_cmd.returncode == 0:
        listing = list_cmd.stdout.strip()
        if listing:
            lines = listing.splitlines()
            if len(lines) > 20:
                logging.error(
                    "Remote directory %s contents (first 20 of %s entries):",
                    remote_parent,
                    len(lines),
                )
                lines = lines[:20]
            else:
                logging.error("Remote directory %s contents:", remote_parent)
            for line in lines:
                logging.error("    %s", line)
        else:
            logging.error(
                "Remote directory %s is empty (ls returned no entries).", remote_parent
            )
    else:
        logging.error(
            "Unable to list remote directory %s (rc=%s): %s",
            remote_parent,
            list_cmd.returncode,
            list_cmd.stderr.strip(),
        )


def signal_handler(signum, frame):
    """handle SIGINT and SIGTERM for graceful shutdown"""
    global shutdown_requested
    logging.info(f"Received signal {signum}. Shutting down gracefully...")
    shutdown_requested = True


def setup_signal_handlers():
    """setup signal handlers for graceful shutdown"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def _read_uid_quick(exam_path: Path):
    """
    return (study_uid, files_touched) by scanning for the first readable dicom header.
    uses pydicom with stop_before_pixels to avoid payload reads.
    """
    import pydicom

    # suppress pydicom VR UI validation warnings for non-standard UIDs
    warnings.filterwarnings(
        "ignore", message=".*Invalid value for VR UI.*", append=True
    )
    # also set pydicom logging level to suppress warnings at the source
    import logging

    logging.getLogger("pydicom.valuerep").setLevel(logging.ERROR)

    touched = 0
    for p in exam_path.rglob("*"):
        if not p.is_file() or p.name.startswith("."):
            continue
        touched += 1
        try:
            dcm = pydicom.dcmread(p, stop_before_pixels=True)
            uid = str(getattr(dcm, "StudyInstanceUID", "")).strip()
            if uid:
                return uid, touched
        except pydicom.errors.InvalidDicomError:
            continue
        except Exception:
            # let it fail loudly elsewhere; UID fast path is best-effort
            continue
    return None, touched


def _early_hash_match(
    src_exam: Path,
    patient_inventory: Dict[ExamFingerprint, str],
    candidate_names: set[str] | None = None,
    min_confirm: int = 5,
):
    """
    hash-only-as-needed discriminator against remote hashes for one patient.

    Parameters
    ----------
    src_exam : Path
        local exam directory to scan/hash
    patient_inventory : Dict[ExamFingerprint, str]
        remote inventory for this patient (fingerprint -> exam_name)
    candidate_names : Optional[set[str]]
        if provided, restrict matching to this subset of remote exam names
    min_confirm : int
        minimum distinct hash confirmations required once a single candidate remains

    Returns
    -------
    tuple[str|None, dict]
        (matched_remote_name or None, stats dict with keys:
         files_hashed, bytes_hashed, confirms_for_<name>, elapsed_s)
    """
    from time import perf_counter

    from fingerprint_utils import hash_file

    # build per-patient maps
    name_to_hashes = {
        name: fp.file_hashes
        for fp, name in patient_inventory.items()
        if (candidate_names is None or name in candidate_names)
    }
    # invert: hash -> {names}
    hash_to_names = {}
    for name, hs in name_to_hashes.items():
        for h in hs:
            s = hash_to_names.get(h)
            if s is None:
                hash_to_names[h] = {name}
            else:
                s.add(name)

    candidates = set(name_to_hashes.keys())
    confirms = dict.fromkeys(candidates, 0)

    files_hashed = 0
    bytes_hashed = 0
    t0 = perf_counter()

    # stream files; intersect candidates as we go
    for p in src_exam.rglob("*"):
        if not p.is_file() or p.name.startswith("."):
            continue
        h = hash_file(p)
        files_hashed += 1
        try:
            bytes_hashed += p.stat().st_size
        except Exception:
            pass

        possible = hash_to_names.get(h)
        if not possible:
            # no remote exam contains this file hash; this exam is new
            candidates.clear()
            break

        # shrink candidate set and update confirmations
        candidates &= possible
        for n in list(possible):
            if n in confirms:
                confirms[n] += 1

        # early exit: unique candidate with enough confirmations
        if len(candidates) == 1:
            (only,) = tuple(candidates)
            if confirms[only] >= min_confirm or confirms[only] == len(
                name_to_hashes[only]
            ):
                elapsed = perf_counter() - t0
                return only, {
                    "files_hashed": files_hashed,
                    "bytes_hashed": bytes_hashed,
                    "elapsed_s": elapsed,
                    f"confirms_for_{only}": confirms[only],
                }

    elapsed = perf_counter() - t0
    if len(candidates) == 1:
        (only,) = tuple(candidates)
        return only, {
            "files_hashed": files_hashed,
            "bytes_hashed": bytes_hashed,
            "elapsed_s": elapsed,
            f"confirms_for_{only}": confirms[only],
        }
    return None, {
        "files_hashed": files_hashed,
        "bytes_hashed": bytes_hashed,
        "elapsed_s": elapsed,
    }


def _share_usage_gb():
    """return (used_gb, free_gb, total_gb) for the SRC_ROOT mount via filesystem accounting (fast, no traversal)"""
    usage = shutil.disk_usage(SRC_ROOT)
    return (
        usage.used / 1_000_000_000,
        usage.free / 1_000_000_000,
        usage.total / 1_000_000_000,
    )


def run_ssh_command(command_str: str, check: bool = True):
    """
    execute a complete command string on the remote via SSH with connection multiplexing.

    important: ssh joins arguments with spaces before handing them to the remote shell.
    to ensure the entire payload is seen as the single argument to `bash -c`, we wrap
    it in single quotes and escape any embedded single quotes.
    """
    # wrap as one argument to bash -c; escape any single quotes inside
    quoted_cmd = "'" + command_str.replace("'", "'\"'\"'") + "'"

    # use SSH multiplexing for better performance with multiple connections
    ssh_opts = [
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPath=~/.ssh/control-%h-%p-%r",
        "-o",
        "ControlPersist=300",
    ]
    full_ssh_command = ["ssh"] + ssh_opts + [DST_SSH_TARGET, "bash", "-c", quoted_cmd]

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


def run_ssh_batch_commands(commands: list[str], check: bool = True):
    """
    execute multiple commands in a single SSH session for better efficiency

    returns list of (stdout, stderr) tuples for each command
    """
    if not commands:
        return []

    # combine commands with proper separation and error handling
    batch_script = []
    for i, cmd in enumerate(commands):
        # add command with exit code capture
        batch_script.append(f"echo '=== COMMAND {i} START ==='")
        batch_script.append(f"({cmd})")
        batch_script.append(f"echo '=== COMMAND {i} EXIT $? ==='")

    combined_cmd = " && ".join(batch_script)

    ssh_opts = [
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPath=~/.ssh/control-%h-%p-%r",
        "-o",
        "ControlPersist=300",
    ]
    full_ssh_command = ["ssh"] + ssh_opts + [DST_SSH_TARGET, "bash", "-c", combined_cmd]

    try:
        process = subprocess.run(
            full_ssh_command,
            check=check,
            capture_output=True,
            text=True,
        )

        # parse the combined output back into individual results
        output_lines = process.stdout.split("\n")
        results = []
        current_output = []

        for line in output_lines:
            if line.startswith("=== COMMAND ") and " START ===" in line:
                current_output = []
            elif line.startswith("=== COMMAND ") and " EXIT " in line:
                exit_code = int(line.split()[-2])
                results.append(
                    (
                        "\n".join(current_output),
                        process.stderr if exit_code != 0 else "",
                    )
                )
                current_output = []
            else:
                current_output.append(line)

        return results

    except subprocess.CalledProcessError as e:
        logging.error(f"SSH batch command failed: {e}")
        raise


def update_remote_inventory():
    """
    print instructions to run the remote fingerprinter manually, then ask the user to confirm
    completion before fetching the resulting cache file via scp; fails loudly on errors
    """
    logging.info("--- remote inventory: manual run, interactive fetch ---")

    remote_script = REMOTE_REPO_PATH / "ops" / "fingerprinter.py"
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


def _is_exam_stable(exam_path: Path, threshold_sec: int) -> tuple[bool, float]:
    """
    check if exam directory is stable (no files modified within threshold_sec)

    returns (is_stable, age_of_most_recent_file_sec)
    """
    import time

    now = time.time()
    most_recent_age = 0.0

    for p in exam_path.rglob("*"):
        if not p.is_file() or p.name.startswith("."):
            continue
        try:
            st = p.stat()
            file_age = now - max(st.st_mtime, st.st_ctime)
            most_recent_age = max(most_recent_age, file_age)
        except Exception:
            # if we can't stat a file, assume it's unstable to be safe
            return False, 0.0

    is_stable = most_recent_age >= threshold_sec
    return is_stable, most_recent_age


def _is_logically_empty(patient_dir: Path):
    """
    return (is_empty, n_subdirs, n_files_non_dot)

    emptiness ignores dotfiles like .DS_Store; only checks immediate children
    """
    n_dirs = 0
    n_files = 0
    for entry in patient_dir.iterdir():
        try:
            if entry.is_dir():
                n_dirs += 1
            elif entry.is_file():
                name = entry.name
                if name.startswith(".") or name.lower() in {".ds_store", "thumbs.db"}:
                    continue
                n_files += 1
        except Exception:
            # if we can't stat it, treat as non-empty to be safe
            n_files += 1
    return (n_dirs == 0 and n_files == 0), n_dirs, n_files


def _prune_empty_patients(
    patient_dirs: list[Path], min_age_sec: int, dry_run: bool
) -> int:
    """
    remove totally empty patient-level dirs older than min_age_sec; returns count pruned
    """
    import time

    now = time.time()
    pruned = 0
    recent_empty = 0
    nonempty = 0

    for d in patient_dirs:
        is_empty, _, _ = _is_logically_empty(d)
        if not is_empty:
            nonempty += 1
            continue

        try:
            st = d.stat()
            age_sec = now - max(st.st_mtime, st.st_ctime)
        except Exception as e:
            logging.warning(f"[PRUNE] unable to stat {d}: {e}")
            continue

        if age_sec < min_age_sec:
            recent_empty += 1
            continue

        if dry_run:
            logging.info(
                f"[PRUNE DRY RUN] would remove empty patient dir {d} (age {age_sec / 3600:.1f}h)"
            )
        else:
            try:
                d.rmdir()
                pruned += 1
                logging.info(
                    f"[PRUNE] removed empty patient dir {d} (age {age_sec / 3600:.1f}h)"
                )
            except Exception as e:
                logging.warning(f"[PRUNE] failed to remove {d}: {e}")

    logging.info(
        f"[PRUNE SUMMARY] checked {len(patient_dirs)} patient roots: "
        f"pruned {pruned}, recent-empty {recent_empty}, nonempty {nonempty}"
    )
    return pruned


def _summarize_unknown_patients(unknown_dirs: list[Path]):
    """
    log a one-line summary for each unknown patient root to explain what they are
    """
    import time

    if not unknown_dirs:
        return
    now = time.time()
    for d in unknown_dirs:
        try:
            entries = list(d.iterdir())
        except Exception as e:
            logging.warning(f"[UNKNOWN] {d.name}: unable to list ({e})")
            continue

        subdirs = [e.name for e in entries if e.is_dir()]
        files = [e.name for e in entries if e.is_file()]
        is_empty = len(subdirs) == 0 and all(
            fn.startswith(".") or fn.lower() == ".ds_store" for fn in files
        )
        try:
            age_h = (now - max(d.stat().st_mtime, d.stat().st_ctime)) / 3600.0
        except Exception:
            age_h = float("nan")

        sample = ", ".join(subdirs[:5]) if subdirs else ""
        logging.info(
            f"[UNKNOWN] patient {d.name}: {'EMPTY' if is_empty else 'HAS DATA'}; "
            f"subdirs={len(subdirs)} files={len(files)}; age={age_h:.1f}h; "
            f"sample_subdirs=[{sample}]"
        )


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
        "-e",
        (
            "ssh -o ControlMaster=auto "
            "-o ControlPath=~/.ssh/control-%h-%p-%r "
            "-o ControlPersist=300"
        ),
        f"{src}/",
        dest_path_str,
    ]

    t0 = monotonic()
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    combined_lines: list[str] = []
    last_progress_line = ""
    last_heartbeat = t0

    assert proc.stdout is not None  # silences type checkers

    def should_log(msg: str) -> bool:
        if "to-chk=" in msg:
            return True
        lowered = msg.lower()
        if "error" in lowered:
            return True
        prefixes = (
            "number of ",
            "total transferred file size",
            "total file size",
            "total bytes sent",
            "total bytes received",
            "sent ",
            "speedup is ",
        )
        return lowered.startswith(prefixes)

    for line in proc.stdout:
        combined_lines.append(line)
        msg = line.rstrip()
        if msg and should_log(msg):
            logging.info(f"[RSYNC] {msg}")
        if "to-chk=" in msg:
            last_progress_line = msg

        now = monotonic()
        if now - last_heartbeat >= 60:
            elapsed = now - t0
            hb = f"[RSYNC HEARTBEAT] {src.name}: elapsed {elapsed:.0f}s" + (
                f" | progress: {last_progress_line}" if last_progress_line else ""
            )
            logging.info(hb)
            last_heartbeat = now

    proc.wait()
    dt = monotonic() - t0
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, command)

    combined = "".join(combined_lines)

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


def process_patient_exams(
    patient_id: str,
    exam_paths: list[Path],
    patient_inventory: Dict[ExamFingerprint, str],
    dry_run: bool,
    immediate_delete: bool = False,
) -> Tuple[str, Dict[str, int], int, int, int]:
    """
    process all exams for a single patient to minimize serialization overhead

    returns (patient_id, combined_stats_dict, total_net_bytes, total_logical_bytes, total_queued_for_delete)
    """
    import time

    process_start = time.time()

    # initialize combined stats for this patient
    combined_stats = {
        "renamed": 0,
        "transferred": 0,
        "skipped": 0,
        "failed": 0,
        "remote_renamed": 0,
        "uid_fastpath_hits": 0,
        "early_hash_hits": 0,
        "full_hash_fallbacks": 0,
    }

    total_net_bytes = 0
    total_logical_bytes = 0
    total_queued = 0

    serialization_time = time.time() - process_start
    if serialization_time > 0.1:  # only log if significant
        logging.info(
            f"[TIMING] Patient {patient_id}: worker startup took {serialization_time:.2f}s"
        )

    for src_path in exam_paths:
        try:
            exam_name, exam_stats, net_bytes, logical_bytes, queued = (
                process_single_exam_internal(
                    src_path, patient_id, patient_inventory, dry_run, immediate_delete
                )
            )

            # accumulate stats
            for key, value in exam_stats.items():
                combined_stats[key] += value

            total_net_bytes += net_bytes
            total_logical_bytes += logical_bytes
            total_queued += queued

        except Exception as e:
            logging.error(f"FATAL ERROR processing {src_path}: {e}", exc_info=True)
            combined_stats["failed"] += 1

    return (
        patient_id,
        combined_stats,
        total_net_bytes,
        total_logical_bytes,
        total_queued,
    )


def process_single_exam_internal(
    src_path: Path,
    patient_id: str,
    patient_inventory: Dict[ExamFingerprint, str],
    dry_run: bool,
    immediate_delete: bool = False,
) -> Tuple[str, Dict[str, int], int, int, int]:
    """
    internal function to process a single exam (moved from process_single_exam)
    now takes patient_id and patient_inventory instead of full dest_inventory

    returns (exam_name, stats_dict, net_bytes, logical_bytes, queued_for_delete)
    """
    exam_name = src_path.name

    # initialize stats for this exam
    exam_stats = {
        "renamed": 0,
        "transferred": 0,
        "skipped": 0,
        "failed": 0,
        "remote_renamed": 0,
        "uid_fastpath_hits": 0,
        "early_hash_hits": 0,
        "full_hash_fallbacks": 0,
    }

    # ---------- stability check ----------
    is_stable, most_recent_age = _is_exam_stable(src_path, STABILITY_THRESHOLD_SEC)
    if not is_stable:
        logging.info(
            f"SKIP UNSTABLE: {exam_name} "
            f"(most recent file {most_recent_age / 60:.1f}min old, "
            f"need {STABILITY_THRESHOLD_SEC / 60:.1f}min)"
        )
        exam_stats["skipped"] += 1
        return exam_name, exam_stats, 0, 0, 0

    # ---------- UID-first fast path ----------
    uid, touched = _read_uid_quick(src_path)
    if uid:
        uid_matches = [
            (fp, name) for fp, name in patient_inventory.items() if fp.study_uid == uid
        ]
        if len(uid_matches) == 1:
            fp, match_name = uid_matches[0]
            old_rem = DST_ROOT_REMOTE / patient_id / match_name
            new_rem = DST_ROOT_REMOTE / patient_id / exam_name

            if dry_run:
                logging.info(
                    f"UID HIT ({uid}) after touching {touched} files: remote has '{match_name}'. "
                    f"[DRY RUN] WOULD "
                    + (
                        "rename remote to new name"
                        if match_name != exam_name
                        else "queue local for deletion"
                    )
                )
            else:
                ver_target = old_rem if match_name != exam_name else new_rem
                ssh_opts = [
                    "-o",
                    "ControlMaster=auto",
                    "-o",
                    "ControlPath=~/.ssh/control-%h-%p-%r",
                    "-o",
                    "ControlPersist=300",
                ]
                ver = subprocess.run(
                    ["ssh"] + ssh_opts + [DST_SSH_TARGET, "ls", "-ld", str(ver_target)],
                    capture_output=True,
                    text=True,
                )
                if ver.returncode != 0:
                    logging.error(
                        "Remote verify failed (UID fastpath) for patient=%s exam=%s uid=%s",
                        patient_id,
                        exam_name,
                        uid,
                    )
                    logging.error("Local source path: %s", src_path)
                    logging.error(
                        "ssh stderr: %s",
                        ver.stderr.strip() or "<empty>",
                    )
                    _log_remote_debug(ver_target, ssh_opts)
                    raise RuntimeError(
                        f"remote verify failed for {ver_target}: {ver.stderr.strip()}"
                    )

            if match_name != exam_name:
                logging.info(
                    f"UID FASTPATH: {exam_name} == remote '{match_name}' "
                    f"→ renaming REMOTE to '{exam_name}' (no hashing)"
                )
                if not dry_run:
                    ssh_opts = [
                        "-o",
                        "ControlMaster=auto",
                        "-o",
                        "ControlPath=~/.ssh/control-%h-%p-%r",
                        "-o",
                        "ControlPersist=300",
                    ]
                    subprocess.run(
                        ["ssh"]
                        + ssh_opts
                        + [DST_SSH_TARGET, "mv", str(old_rem), str(new_rem)],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
            else:
                logging.info(
                    f"UID FASTPATH: {exam_name} already present on remote (verified) – queue local for deletion"
                )
            exam_stats["remote_renamed"] += int(match_name != exam_name)
            exam_stats["renamed"] += 1
            exam_stats["uid_fastpath_hits"] += 1

            if dry_run:
                action = "DELETE" if immediate_delete else "QUEUE FOR DELETE"
                logging.info(f"[DRY RUN] WOULD {action} (LOCAL): {src_path}")
            else:
                if immediate_delete:
                    shutil.rmtree(str(src_path))
                    logging.info(f"DELETED (LOCAL): {src_path}")
                else:
                    dst = DELETE_QUEUE_DIR / patient_id / exam_name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src_path), str(dst))
                    logging.info(
                        f"QUEUED FOR DELETE (LOCAL): moved {src_path} -> {dst}"
                    )

            return exam_name, exam_stats, 0, 0, 1

    # ---------- UID miss/ambiguous: early-exit hashing ----------
    if patient_inventory:
        candidate_set = None
        if uid:
            candidate_set = {
                name for fp, name in patient_inventory.items() if fp.study_uid == uid
            }
            if len(candidate_set) <= 1:
                candidate_set = None

        matched, ev = _early_hash_match(
            src_exam=src_path,
            patient_inventory=patient_inventory,
            candidate_names=candidate_set,
            min_confirm=5,
        )
        if matched:
            old_rem = DST_ROOT_REMOTE / patient_id / matched
            new_rem = DST_ROOT_REMOTE / patient_id / exam_name
            if matched != exam_name:
                logging.info(
                    f"EARLY-HASH MATCH: {exam_name} == remote '{matched}' "
                    f"(confirmed {ev.get(f'confirms_for_{matched}', 0)} hashes, "
                    f"{ev['files_hashed']} files, {ev['bytes_hashed'] / 1e6:.1f} MB hashed in {ev['elapsed_s']:.1f}s) "
                    f"→ renaming REMOTE to '{exam_name}'"
                )
                if not dry_run:
                    ssh_opts = [
                        "-o",
                        "ControlMaster=auto",
                        "-o",
                        "ControlPath=~/.ssh/control-%h-%p-%r",
                        "-o",
                        "ControlPersist=300",
                    ]
                    subprocess.run(
                        ["ssh"]
                        + ssh_opts
                        + [DST_SSH_TARGET, "mv", str(old_rem), str(new_rem)],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
            else:
                logging.info(
                    f"EARLY-HASH MATCH: {exam_name} already present on remote "
                    f"(confirmed {ev.get(f'confirms_for_{matched}', 0)} hashes, "
                    f"{ev['files_hashed']} files, {ev['bytes_hashed'] / 1e6:.1f} MB hashed in {ev['elapsed_s']:.1f}s) "
                    f"→ queue LOCAL for deletion"
                )
            exam_stats["remote_renamed"] += int(matched != exam_name)
            exam_stats["renamed"] += 1
            exam_stats["early_hash_hits"] += 1

            if dry_run:
                action = "DELETE" if immediate_delete else "QUEUE FOR DELETE"
                logging.info(f"[DRY RUN] WOULD {action} (LOCAL): {src_path}")
            else:
                if immediate_delete:
                    shutil.rmtree(str(src_path))
                    logging.info(f"DELETED (LOCAL): {src_path}")
                else:
                    dst = DELETE_QUEUE_DIR / patient_id / exam_name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src_path), str(dst))
                    logging.info(
                        f"QUEUED FOR DELETE (LOCAL): moved {src_path} -> {dst}"
                    )

            return exam_name, exam_stats, 0, 0, 1

    # ---------- Not found: transfer ----------
    logging.info(f"NEW: {exam_name} not found on remote. Transferring.")
    if dry_run:
        logging.info(
            f"[DRY RUN] WOULD TRANSFER: {src_path} -> "
            f"{DST_ROOT_REMOTE / patient_id / exam_name}"
        )
        exam_stats["transferred"] += 1
        net_bytes, logical_bytes = 0, 0
    else:
        net_bytes, logical_bytes, _ = rsync_exam_remote(src_path, patient_id, exam_name)
        exam_stats["transferred"] += 1

    # queue local after transfer
    if dry_run:
        action = "DELETE" if immediate_delete else "QUEUE FOR DELETE"
        logging.info(f"[DRY RUN] WOULD {action} (LOCAL): {src_path}")
    else:
        if immediate_delete:
            shutil.rmtree(str(src_path))
            logging.info(f"DELETED (LOCAL): {src_path}")
        else:
            dst = DELETE_QUEUE_DIR / patient_id / exam_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst))
            logging.info(f"QUEUED FOR DELETE (LOCAL): moved {src_path} -> {dst}")

    return exam_name, exam_stats, net_bytes, logical_bytes, 1


def process_uid_fastpath_phase(
    exam_paths: list[Path],
    dest_inventory: Dict[str, Dict[ExamFingerprint, str]],
    dry_run: bool,
    immediate_delete: bool = False,
    max_exams: int | None = None,
) -> Tuple[list[Path], Dict[str, int], int, list[Path]]:
    """
    First phase: process exam directories for UID fastpath matches.

    Parameters
    ----------
    max_exams : Optional[int]
        Cap on how many exams to touch during this run. Remaining directories are
        returned so they can be deferred to the next restart.

    Returns
    -------
    tuple
        (exams_requiring_transfer, stats_dict, processed_count, deferred_dirs)
    """
    import shutil  # Import at function level to avoid scoping issues

    remaining_exams: list[Path] = []
    deferred_exams: list[Path] = []
    stats = {
        "renamed": 0,
        "skipped": 0,
        "failed": 0,
        "remote_renamed": 0,
        "uid_fastpath_hits": 0,
    }
    processed = 0

    for idx, src_path in enumerate(exam_paths):
        if max_exams is not None and processed >= max_exams:
            deferred_exams.extend(exam_paths[idx:])
            break

        processed += 1
        try:
            patient_id = src_path.parent.name
            exam_name = src_path.name
            patient_inventory = dest_inventory.get(patient_id, {})

            # stability check
            is_stable, most_recent_age = _is_exam_stable(
                src_path, STABILITY_THRESHOLD_SEC
            )
            if not is_stable:
                logging.info(
                    f"SKIP UNSTABLE: {exam_name} "
                    f"(most recent file {most_recent_age / 60:.1f}min old, "
                    f"need {STABILITY_THRESHOLD_SEC / 60:.1f}min)"
                )
                stats["skipped"] += 1
                continue

            # UID fastpath check
            uid, touched = _read_uid_quick(src_path)
            if uid and patient_inventory:
                uid_matches = [
                    (fp, name)
                    for fp, name in patient_inventory.items()
                    if fp.study_uid == uid
                ]
                if len(uid_matches) == 1:
                    fp, match_name = uid_matches[0]
                    old_rem = DST_ROOT_REMOTE / patient_id / match_name
                    new_rem = DST_ROOT_REMOTE / patient_id / exam_name

                    # verify remote exists
                    if not dry_run:
                        ver_target = old_rem if match_name != exam_name else new_rem
                        ssh_opts = [
                            "-o",
                            "ControlMaster=auto",
                            "-o",
                            "ControlPath=~/.ssh/control-%h-%p-%r",
                            "-o",
                            "ControlPersist=300",
                        ]
                        ver = subprocess.run(
                            ["ssh"]
                            + ssh_opts
                            + [DST_SSH_TARGET, "ls", "-ld", str(ver_target)],
                            capture_output=True,
                            text=True,
                        )
                        if ver.returncode != 0:
                            logging.warning(
                                f"UID match but remote verify failed for {ver_target}, will transfer"
                            )
                            logging.warning(
                                "Local source path: %s | patient=%s exam=%s uid=%s",
                                src_path,
                                patient_id,
                                exam_name,
                                uid,
                            )
                            logging.warning(
                                "ssh stderr: %s",
                                ver.stderr.strip() or "<empty>",
                            )
                            _log_remote_debug(ver_target, ssh_opts)
                            remaining_exams.append(src_path)
                            continue

                    # handle rename or already exists
                    if match_name != exam_name:
                        logging.info(
                            f"UID FASTPATH: {exam_name} == remote '{match_name}' "
                            f"→ renaming REMOTE to '{exam_name}' (no hashing)"
                        )
                        if not dry_run:
                            ssh_opts = [
                                "-o",
                                "ControlMaster=auto",
                                "-o",
                                "ControlPath=~/.ssh/control-%h-%p-%r",
                                "-o",
                                "ControlPersist=300",
                            ]
                            subprocess.run(
                                ["ssh"]
                                + ssh_opts
                                + [DST_SSH_TARGET, "mv", str(old_rem), str(new_rem)],
                                check=True,
                                capture_output=True,
                                text=True,
                            )
                        stats["remote_renamed"] += 1
                    else:
                        logging.info(
                            f"UID FASTPATH: {exam_name} already present on remote (verified) – queue local for deletion"
                        )

                    stats["renamed"] += 1
                    stats["uid_fastpath_hits"] += 1

                    # handle local deletion/queueing
                    if dry_run:
                        action = "DELETE" if immediate_delete else "QUEUE FOR DELETE"
                        logging.info(f"[DRY RUN] WOULD {action} (LOCAL): {src_path}")
                    else:
                        if immediate_delete:
                            shutil.rmtree(str(src_path))
                            logging.info(f"DELETED (LOCAL): {src_path}")
                        else:
                            dst = DELETE_QUEUE_DIR / patient_id / exam_name
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(src_path), str(dst))
                            logging.info(
                                f"QUEUED FOR DELETE (LOCAL): moved {src_path} -> {dst}"
                            )

                    continue

            # no UID fastpath match, needs transfer
            remaining_exams.append(src_path)

        except Exception as e:
            logging.error(
                f"FATAL ERROR in UID fastpath for {src_path}: {e}", exc_info=True
            )
            stats["failed"] += 1
            remaining_exams.append(src_path)  # still try to transfer

    return remaining_exams, stats, processed, deferred_exams


def process_single_exam(
    src_path: Path,
    dest_inventory: Dict[str, Dict[ExamFingerprint, str]],
    dry_run: bool,
    immediate_delete: bool = False,
) -> Tuple[str, Dict[str, int], int, int, int]:
    """
    legacy wrapper for backward compatibility - delegates to internal function

    returns (exam_name, stats_dict, net_bytes, logical_bytes, queued_for_delete)
    """
    try:
        patient_id = src_path.parent.name
        patient_inventory = dest_inventory.get(patient_id, {})
        return process_single_exam_internal(
            src_path, patient_id, patient_inventory, dry_run, immediate_delete
        )
    except Exception:
        logging.error(f"FATAL ERROR processing {src_path}", exc_info=True)
        exam_stats = {
            "renamed": 0,
            "transferred": 0,
            "skipped": 0,
            "failed": 1,
            "remote_renamed": 0,
            "uid_fastpath_hits": 0,
            "early_hash_hits": 0,
            "full_hash_fallbacks": 0,
        }
        return src_path.name, exam_stats, 0, 0, 0


def run_single_sync(
    update_inventory: bool, dry_run: bool, immediate_delete: bool = False
):
    """
    drive end-to-end sync with:
      - numeric-only patient discovery
      - UID-first and early-hash match (as you already have)
      - pruning of old empty patient roots to shrink future scans
      - unknown-patient summaries so you know what those are
    """
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

    inv_patients = len(dest_inventory)
    inv_exams = sum(len(exams) for exams in dest_inventory.values())
    logging.info(
        f"Destination inventory summary: {inv_patients:,} patients, {inv_exams:,} exams"
    )

    # ----- discover source set (numeric patient dirs only) -----
    logging.info(f"Scanning top-level in {SRC_ROOT} for patient directories...")
    top_dirs = [d for d in SRC_ROOT.iterdir() if d.is_dir()]
    patient_dirs = [d for d in top_dirs if d.name.isdigit()]
    ignored_dirs = [
        d for d in top_dirs if not d.name.isdigit() and not d.name.startswith("_")
    ]
    if ignored_dirs:
        sample = ", ".join(sorted(p.name for p in ignored_dirs[:10]))
        logging.info(
            f"Ignoring {len(ignored_dirs)} non-patient root dirs: [{sample}...]"
        )

    known_pat = [d for d in patient_dirs if d.name in dest_inventory]
    unknown_pat = [d for d in patient_dirs if d.name not in dest_inventory]
    if unknown_pat:
        sample_u = ", ".join(sorted(p.name for p in unknown_pat[:10]))
        logging.info(
            f"Found {len(patient_dirs)} patient roots; {len(known_pat)} known in cache, "
            f"{len(unknown_pat)} unknown. Sample unknown: [{sample_u}...]"
        )
        _summarize_unknown_patients(unknown_pat)
    else:
        logging.info(f"Found {len(patient_dirs)} patient roots; all present in cache.")

    # ----- prune old-empty patient dirs BEFORE building the exam list -----
    PRUNE_AGE_SEC = 6 * 3600  # delete totally empty patient dirs older than 6h
    _prune_empty_patients(patient_dirs, PRUNE_AGE_SEC, dry_run=dry_run)

    # rebuild patient_dirs after pruning in case some were removed
    patient_dirs = [d for d in SRC_ROOT.iterdir() if d.is_dir() and d.name.isdigit()]

    # ----- collect all exams for two-phase processing -----
    all_exams = []
    for p_dir in patient_dirs:
        exam_paths = [p for p in p_dir.iterdir() if p.is_dir()]
        all_exams.extend(exam_paths)

    total_exams = len(all_exams)
    logging.info(f"Found {total_exams} source exams to process.")
    max_exams_this_run = (
        MAX_EXAMS_PER_RUN
        if (MAX_EXAMS_PER_RUN is None or MAX_EXAMS_PER_RUN > 0)
        else None
    )

    # ===== PHASE 1: UID FASTPATH (Sequential, Fast) =====
    logging.info("=== PHASE 1: UID FASTPATH PROCESSING ===")
    logging.info("Processing all exams for UID matches first (fast, no SSH transfers)")

    phase1_start = monotonic()
    transfer_candidates, phase1_stats, phase1_processed, deferred_exams = (
        process_uid_fastpath_phase(
            all_exams,
            dest_inventory,
            dry_run,
            immediate_delete,
        )
    )
    phase1_time = monotonic() - phase1_start

    logging.info(f"Phase 1 complete in {phase1_time:.1f}s:")
    logging.info(f"  - UID fastpath hits: {phase1_stats['uid_fastpath_hits']}")
    logging.info(f"  - Remote renamed: {phase1_stats['remote_renamed']}")
    logging.info(f"  - Skipped (unstable): {phase1_stats['skipped']}")
    logging.info(f"  - Failed: {phase1_stats['failed']}")
    total_transfer_candidates = len(transfer_candidates)
    logging.info(f"  - Remaining for transfer: {total_transfer_candidates}")
    logging.info(f"  - Exams touched this run: {phase1_processed}")

    remaining_exams = transfer_candidates
    cap_deferrals = 0
    if (
        max_exams_this_run is not None
        and total_transfer_candidates > max_exams_this_run
    ):
        remaining_exams = transfer_candidates[:max_exams_this_run]
        deferred_chunk = transfer_candidates[max_exams_this_run:]
        cap_deferrals = len(deferred_chunk)
        deferred_exams.extend(deferred_chunk)
        logging.info(
            f"  - Transfer cap active ({max_exams_this_run}); scheduling {len(remaining_exams)} now, deferring {cap_deferrals} for the next pass."
        )

    if deferred_exams:
        logging.info(
            f"Deferred {len(deferred_exams)} exams from earlier stage for later processing."
            if cap_deferrals == 0
            else f"Deferred {len(deferred_exams)} exams for later processing."
        )

    if not remaining_exams:
        if deferred_exams:
            logging.info(
                "Run cap reached with no transfers left in quota; will resume on next restart."
            )
        else:
            logging.info("All exams processed via UID fastpath! No transfers needed.")
        return

    # ===== PHASE 2: TRANSFERS AND HASH MATCHING (Parallel) =====
    logging.info("=== PHASE 2: TRANSFER PROCESSING ===")
    logging.info(
        f"Processing {len(remaining_exams)} exams requiring hash matching or transfer"
    )

    phase2_stats = {
        "renamed": 0,
        "transferred": 0,
        "skipped": 0,
        "failed": 0,
        "remote_renamed": 0,
        "uid_fastpath_hits": 0,
        "early_hash_hits": 0,
        "full_hash_fallbacks": 0,
    }

    HEARTBEAT_SEC = 60
    phase2_start = monotonic()
    last_hb = phase2_start
    bytes_net_total = 0
    bytes_log_total = 0
    transfers_done = 0
    phase2_completed = 0
    queued_local_for_delete = 0

    initial_completed = max(0, phase1_processed - total_transfer_candidates)
    total_known = total_exams

    used_gb, free_gb, total_gb = _share_usage_gb()
    logging.info(
        f"[USAGE mount] {SRC_ROOT}: used {used_gb:.1f} GB / total {total_gb:.1f} GB (free {free_gb:.1f} GB)"
    )

    pbar = tqdm(
        total=total_known,
        initial=initial_completed,
        desc="Processing Exams",
    )

    with ProcessPoolExecutor(max_workers=PARALLEL_JOBS) as executor:
        # submit individual exam processing tasks (back to exam-level for better load balancing)
        future_to_exam = {
            executor.submit(
                process_single_exam,
                src_path,
                dest_inventory,
                dry_run,
                immediate_delete,
            ): src_path
            for src_path in remaining_exams
        }

        # process results as they complete, maintaining heartbeat
        for future in as_completed(future_to_exam):
            src_path = future_to_exam[future]
            try:
                exam_name, exam_stats, net_bytes, logical_bytes, queued = (
                    future.result()
                )

                # update global stats
                for key, value in exam_stats.items():
                    phase2_stats[key] += value

                bytes_net_total += net_bytes
                bytes_log_total += logical_bytes
                if exam_stats["transferred"] > 0:
                    transfers_done += 1
                queued_local_for_delete += queued
                phase2_completed += 1
                pbar.update(1)

                # heartbeat check
                now = monotonic()
                if now - last_hb >= HEARTBEAT_SEC:
                    elapsed = now - phase2_start
                    used_gb, free_gb, total_gb = _share_usage_gb()
                    logging.info(
                        f"[HEARTBEAT] processed {initial_completed + phase2_completed}/{total_known} | "
                        f"{transfers_done} transferred (logical {bytes_log_total / 1e6:.1f} MB, net {bytes_net_total / 1e6:.1f} MB) | "
                        f"{phase2_stats['remote_renamed']} remote-renamed | "
                        f"early_hash {phase2_stats['early_hash_hits']}, "
                        f"full_hash {phase2_stats['full_hash_fallbacks']} | "
                        f"rate {((initial_completed + phase2_completed) / elapsed * 3600.0) if elapsed > 0 else 0.0:.2f} exams/hr | "
                        f"share {used_gb:.1f}/{total_gb:.1f} GB used (free {free_gb:.1f} GB)"
                    )
                    last_hb = now

            except Exception as e:
                logging.error(f"FATAL ERROR processing {src_path}: {e}", exc_info=True)
                phase2_stats["failed"] += 1
                phase2_completed += 1
                pbar.update(1)

    pbar.close()

    # combine stats from both phases
    combined_stats = {
        "renamed": phase1_stats["renamed"] + phase2_stats["renamed"],
        "transferred": phase2_stats["transferred"],
        "skipped": phase1_stats["skipped"] + phase2_stats["skipped"],
        "failed": phase1_stats["failed"] + phase2_stats["failed"],
        "remote_renamed": phase1_stats["remote_renamed"]
        + phase2_stats["remote_renamed"],
        "uid_fastpath_hits": phase1_stats["uid_fastpath_hits"],
        "early_hash_hits": phase2_stats["early_hash_hits"],
        "full_hash_fallbacks": phase2_stats["full_hash_fallbacks"],
    }

    total_elapsed = monotonic() - phase1_start
    phase2_time = monotonic() - phase2_start
    total_completed_this_run = initial_completed + phase2_completed

    logging.info("=== SYNC COMPLETE ===")
    logging.info(
        f"Total time: {total_elapsed / 60:.1f} minutes (Phase 1: {phase1_time:.1f}s, Phase 2: {phase2_time:.1f}s)"
    )
    if total_elapsed > 0 and total_completed_this_run > 0:
        logging.info(
            f"Overall rate (this run): {total_completed_this_run / total_elapsed * 3600:.1f} exams/hour"
        )
    logging.info(
        f"Processed {total_completed_this_run}/{total_exams} known exams this pass."
    )

    logging.info("PHASE 1 (UID Fastpath):")
    logging.info(f"  - UID fastpath hits: {phase1_stats['uid_fastpath_hits']}")
    logging.info(f"  - Remote renamed: {phase1_stats['remote_renamed']}")
    logging.info(f"  - Skipped (unstable): {phase1_stats['skipped']}")

    logging.info("PHASE 2 (Transfers/Hash Matching):")
    logging.info(f"  - Transferred new: {phase2_stats['transferred']}")
    logging.info(f"  - Early hash matches: {phase2_stats['early_hash_hits']}")
    logging.info(f"  - Remote renamed: {phase2_stats['remote_renamed']}")
    logging.info(f"  - Failed: {phase2_stats['failed']}")

    logging.info("TOTALS:")
    logging.info(f"  - Total renamed/matched: {combined_stats['renamed']}")
    logging.info(f"  - Total transferred: {combined_stats['transferred']}")
    logging.info(f"  - Total failed: {combined_stats['failed']}")
    logging.info(
        f"  - Total queued/deleted locally: {queued_local_for_delete + phase1_stats['renamed']}"
    )
    if deferred_exams:
        logging.info(f"  - Deferred for next run (due to cap): {len(deferred_exams)}")

    used_gb, free_gb, total_gb = _share_usage_gb()
    logging.info(
        f"[FINAL USAGE] {SRC_ROOT}: used {used_gb:.1f} GB / total {total_gb:.1f} GB (free {free_gb:.1f} GB)"
    )


def run_with_auto_restart(
    update_inventory: bool, dry_run: bool, immediate_delete: bool = False
):
    """run sync with automatic restart functionality"""
    setup_signal_handlers()

    restart_count = 0
    while not shutdown_requested:
        restart_count += 1

        if restart_count > 1:
            logging.info(f"=== AUTO-RESTART #{restart_count} ===")
        else:
            logging.info("=== STARTING SYNC ===")

        try:
            run_single_sync(update_inventory, dry_run, immediate_delete)

            if shutdown_requested:
                logging.info("Shutdown requested during sync. Exiting.")
                break

            logging.info(
                f"Sync completed successfully. Waiting "
                f"{RESTART_DELAY_SEC} seconds before restart..."
            )

            # wait for restart delay with periodic checks for shutdown
            for _ in range(RESTART_DELAY_SEC):
                if shutdown_requested:
                    logging.info("Shutdown requested during restart delay. Exiting.")
                    return
                time.sleep(1)

        except KeyboardInterrupt:
            logging.info("Received KeyboardInterrupt. Shutting down gracefully...")
            break
        except Exception as e:
            mount_missing = isinstance(e, FileNotFoundError) and (
                getattr(e, "filename", None) == str(SRC_ROOT)
            )
            if mount_missing:
                logging.error(
                    f"Source root {SRC_ROOT} is unavailable; stopping auto-restart.",
                    exc_info=True,
                )
                logging.info(
                    "Auto-restart halted. Remount the source share and restart the sync manually."
                )
                return

            logging.error(f"Sync failed with error: {e}", exc_info=True)
            if shutdown_requested:
                logging.info("Shutdown requested after error. Exiting.")
                break

            logging.info(
                f"Waiting {RESTART_DELAY_SEC} seconds before restart after error..."
            )
            for _ in range(RESTART_DELAY_SEC):
                if shutdown_requested:
                    logging.info(
                        "Shutdown requested during error restart delay. Exiting."
                    )
                    return
                time.sleep(1)


def main(
    update_inventory: bool,
    dry_run: bool,
    immediate_delete: bool = False,
    auto_restart: bool = False,
):
    """main entry point with optional auto-restart"""
    if auto_restart:
        run_with_auto_restart(update_inventory, dry_run, immediate_delete)
    else:
        run_single_sync(update_inventory, dry_run, immediate_delete)


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
    parser.add_argument(
        "--no-immediate-delete",
        action="store_true",
        help="Move local files to delete queue instead of immediately deleting them.",
    )
    parser.add_argument(
        "--no-auto-restart",
        action="store_true",
        help="Run sync once and exit instead of automatically restarting.",
    )
    args = parser.parse_args()
    main(
        update_inventory=args.update_inventory,
        dry_run=args.dry_run,
        immediate_delete=not args.no_immediate_delete,
        auto_restart=not args.no_auto_restart,
    )
