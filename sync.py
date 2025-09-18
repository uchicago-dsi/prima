#!/usr/bin/env python

import argparse
import json
import logging
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple

from tqdm import tqdm

# Import the shared logic
from fingerprint_utils import ExamFingerprint

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
PARALLEL_JOBS = 4
REMOTE_PARALLEL_JOBS = 4  # NEW: For REMOTE processing. Start with 4, maybe try 2.
# File stability check: only process exam dirs where most recent file is older than this
STABILITY_THRESHOLD_SEC = 300  # 5 minutes
# --- END CONFIGURATION ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("sync.log"), logging.StreamHandler()],
)


def _read_uid_quick(exam_path: Path):
    """
    return (study_uid, files_touched) by scanning for the first readable dicom header.
    uses pydicom with stop_before_pixels to avoid payload reads.
    """
    import pydicom

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


def process_single_exam(
    src_path: Path,
    dest_inventory: Dict[str, Dict[ExamFingerprint, str]],
    dry_run: bool,
) -> Tuple[str, Dict[str, int], int, int, int]:
    """
    process a single exam directory and return results for heartbeat tracking

    returns (exam_name, stats_dict, net_bytes, logical_bytes, queued_for_delete)
    """
    try:
        patient_id, exam_name = src_path.parent.name, src_path.name
        dest_patient = dest_inventory.get(patient_id, {})

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
                (fp, name) for fp, name in dest_patient.items() if fp.study_uid == uid
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
                    ver = subprocess.run(
                        ["ssh", DST_SSH_TARGET, "ls", "-ld", str(ver_target)],
                        capture_output=True,
                        text=True,
                    )
                    if ver.returncode != 0:
                        raise RuntimeError(
                            f"remote verify failed for {ver_target}: {ver.stderr.strip()}"
                        )

                if match_name != exam_name:
                    logging.info(
                        f"UID FASTPATH: {exam_name} == remote '{match_name}' "
                        f"→ renaming REMOTE to '{exam_name}' (no hashing)"
                    )
                    if not dry_run:
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
                else:
                    logging.info(
                        f"UID FASTPATH: {exam_name} already present on remote (verified) – queue local for deletion"
                    )
                exam_stats["remote_renamed"] += int(match_name != exam_name)
                exam_stats["renamed"] += 1
                exam_stats["uid_fastpath_hits"] += 1

                if dry_run:
                    logging.info(
                        f"[DRY RUN] WOULD QUEUE FOR DELETE (LOCAL): {src_path} -> "
                        f"{DELETE_QUEUE_DIR / patient_id / exam_name}"
                    )
                else:
                    dst = DELETE_QUEUE_DIR / patient_id / exam_name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src_path), str(dst))
                    logging.info(
                        f"QUEUED FOR DELETE (LOCAL): moved {src_path} -> {dst}"
                    )

                return exam_name, exam_stats, 0, 0, 1

        # ---------- UID miss/ambiguous: early-exit hashing ----------
        if dest_patient:
            candidate_set = None
            if uid:
                candidate_set = {
                    name for fp, name in dest_patient.items() if fp.study_uid == uid
                }
                if len(candidate_set) <= 1:
                    candidate_set = None

            matched, ev = _early_hash_match(
                src_exam=src_path,
                patient_inventory=dest_patient,
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
                    logging.info(
                        f"[DRY RUN] WOULD QUEUE FOR DELETE (LOCAL): {src_path} -> "
                        f"{DELETE_QUEUE_DIR / patient_id / exam_name}"
                    )
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
            net_bytes, logical_bytes, _ = rsync_exam_remote(
                src_path, patient_id, exam_name
            )
            exam_stats["transferred"] += 1

        # queue local after transfer
        if dry_run:
            logging.info(
                f"[DRY RUN] WOULD QUEUE FOR DELETE (LOCAL): {src_path} -> "
                f"{DELETE_QUEUE_DIR / patient_id / exam_name}"
            )
        else:
            dst = DELETE_QUEUE_DIR / patient_id / exam_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst))
            logging.info(f"QUEUED FOR DELETE (LOCAL): moved {src_path} -> {dst}")

        return exam_name, exam_stats, net_bytes, logical_bytes, 1

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


def main(update_inventory: bool, dry_run: bool):
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

    # exams: only under numeric patients
    source_exams = [p for p_dir in patient_dirs for p in p_dir.iterdir() if p.is_dir()]
    logging.info(f"Found {len(source_exams)} source exams to process.")

    # --- the rest of your main stays as in your current implementation ---
    # below is your existing loop with UID fast path, early-hash, rsync + heartbeats

    stats = {
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
    t_start = monotonic()
    last_hb = t_start
    bytes_net_total = 0
    bytes_log_total = 0
    transfers_done = 0
    processed = 0
    queued_local_for_delete = 0

    used_gb, free_gb, total_gb = _share_usage_gb()
    logging.info(
        f"[USAGE mount] {SRC_ROOT}: used {used_gb:.1f} GB / total {total_gb:.1f} GB (free {free_gb:.1f} GB)"
    )

    pbar = tqdm(total=len(source_exams), desc="Processing Source Exams")

    with ProcessPoolExecutor(max_workers=PARALLEL_JOBS) as executor:
        # submit all exam processing tasks
        future_to_exam = {
            executor.submit(
                process_single_exam, src_path, dest_inventory, dry_run
            ): src_path
            for src_path in source_exams
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
                    stats[key] += value

                bytes_net_total += net_bytes
                bytes_log_total += logical_bytes
                if exam_stats["transferred"] > 0:
                    transfers_done += 1
                queued_local_for_delete += queued
                processed += 1
                pbar.update(1)

                # heartbeat check
                now = monotonic()
                if now - last_hb >= HEARTBEAT_SEC:
                    elapsed = now - t_start
                    used_gb, free_gb, total_gb = _share_usage_gb()
                    logging.info(
                        f"[HEARTBEAT] processed {processed}/{len(source_exams)} | "
                        f"{transfers_done} transferred (logical {bytes_log_total / 1e6:.1f} MB, net {bytes_net_total / 1e6:.1f} MB) | "
                        f"{stats['remote_renamed']} remote-renamed | "
                        f"uid_fastpath {stats['uid_fastpath_hits']}, early_hash {stats['early_hash_hits']}, "
                        f"full_hash {stats['full_hash_fallbacks']} | "
                        f"overall {(processed / elapsed * 3600.0) if elapsed > 0 else 0.0:.2f} exams/hr | "
                        f"share {used_gb:.1f}/{total_gb:.1f} GB used (free {free_gb:.1f} GB)"
                    )
                    last_hb = now

            except Exception as e:
                logging.error(f"FATAL ERROR processing {src_path}: {e}", exc_info=True)
                stats["failed"] += 1
                processed += 1
                pbar.update(1)

    pbar.close()

    logging.info("--- Sync Complete ---")
    logging.info(
        f"Renamed/Matched: {stats['renamed']}  (remote-renamed: {stats['remote_renamed']}, "
        f"uid_fastpath: {stats['uid_fastpath_hits']}, early_hash: {stats['early_hash_hits']}, "
        f"full_hash_fallbacks: {stats['full_hash_fallbacks']})"
    )
    logging.info(f"Transferred New: {stats['transferred']}")
    logging.info(f"Skipped (bad source): {stats['skipped']}")
    logging.info(f"Failed: {stats['failed']}")
    logging.info(f"Queued locally for deletion: {queued_local_for_delete}")
    used_gb, free_gb, total_gb = _share_usage_gb()
    logging.info(
        f"[USAGE mount FINAL] {SRC_ROOT}: used {used_gb:.1f} GB / total {total_gb:.1f} GB (free {free_gb:.1f} GB)"
    )


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
