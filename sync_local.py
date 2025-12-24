#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import stat
import subprocess
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import monotonic
from typing import Tuple

# suppress pydicom VR UI validation warnings for non-standard UIDs
warnings.filterwarnings("ignore", message=".*Invalid value for VR UI.*", append=True)
logging.getLogger("pydicom.valuerep").setLevel(logging.ERROR)

# --- CONFIGURATION ---
# SRC_ROOT = Path("/mnt/uchad_samba/16352A/")
# DST_ROOT = Path("/gpfs/data/huo-lab/Image/ChiMEC/MG")
SRC_ROOT = Path("/mnt/uchad_samba/13073B")
DST_ROOT = Path("/gpfs/data/karczmar-lab/CAPS/MRI1.0")
DELETE_QUEUE_DIR = SRC_ROOT / "_synced_and_queued_for_deletion"
STABILITY_THRESHOLD_SEC = 600
RESTART_DELAY_SEC = 120
EMPTY_DIR_MIN_AGE_SEC = 3600  # only delete empty patient dirs older than 1 hour
# --- END CONFIGURATION ---

PROGRESS_BAR_WIDTH = 28
DEFAULT_WORKERS = 8


def _format_interval(seconds: float) -> str:
    """format seconds as HH:MM:SS or MM:SS"""
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _format_progress_bar(fraction: float, width: int = PROGRESS_BAR_WIDTH) -> str:
    """format a progress bar like [####------]"""
    frac = max(0.0, min(1.0, float(fraction)))
    filled = int(round(frac * width))
    filled = min(filled, width)
    empty = max(width - filled, 0)
    return "[" + ("#" * filled) + ("-" * empty) + "]"


def _format_size(num_bytes: float | int) -> str:
    """format bytes as human-readable size"""
    if num_bytes >= 1_000_000_000_000:
        return f"{num_bytes / 1_000_000_000_000:.2f} TB"
    if num_bytes >= 1_000_000_000:
        return f"{num_bytes / 1_000_000_000:.2f} GB"
    if num_bytes >= 1_000_000:
        return f"{num_bytes / 1_000_000:.1f} MB"
    return f"{num_bytes / 1_000:.1f} KB"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("sync.log"), logging.StreamHandler()],
)

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """handle SIGINT and SIGTERM for graceful shutdown"""
    global shutdown_requested
    logging.info(f"Received signal {signum}. Shutting down gracefully...")
    shutdown_requested = True


def setup_signal_handlers():
    """setup signal handlers for graceful shutdown"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def _share_usage_gb():
    """return (used_gb, free_gb, total_gb) for the SRC_ROOT mount"""
    usage = shutil.disk_usage(SRC_ROOT)
    return (
        usage.used / 1_000_000_000,
        usage.free / 1_000_000_000,
        usage.total / 1_000_000_000,
    )


def _is_exam_stable(exam_path: Path, threshold_sec: int) -> Tuple[bool, float]:
    """
    check if exam directory is stable (no files modified within threshold_sec)

    returns (is_stable, age_of_most_recent_file_sec)
    """
    now = time.time()
    most_recent_age = float("inf")

    for p in exam_path.rglob("*"):
        if not p.is_file() or p.name.startswith("."):
            continue
        try:
            st = p.stat()
            file_age = now - max(st.st_mtime, st.st_ctime)
            most_recent_age = min(most_recent_age, file_age)
        except Exception:
            return False, 0.0

    is_stable = most_recent_age >= threshold_sec
    return is_stable, most_recent_age


def _calculate_dir_size(path: Path) -> int:
    """calculate total size of directory in bytes"""
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return total


def _cleanup_empty_patient_dirs(dry_run: bool) -> int:
    """
    remove empty patient directories from SRC_ROOT if older than EMPTY_DIR_MIN_AGE_SEC

    returns number of directories removed (or would be removed in dry_run mode)
    """
    now = time.time()
    removed = 0

    for d in SRC_ROOT.iterdir():
        if not d.is_dir() or not d.name.isdigit():
            continue

        # check if empty (no files or subdirectories)
        try:
            contents = list(d.iterdir())
        except PermissionError:
            continue

        if contents:
            continue

        # check age via mtime
        try:
            dir_age = now - d.stat().st_mtime
        except Exception:
            continue

        if dir_age < EMPTY_DIR_MIN_AGE_SEC:
            logging.debug(
                f"SKIP EMPTY DIR: {d.name} too recent ({dir_age / 60:.1f}min old, "
                f"need {EMPTY_DIR_MIN_AGE_SEC / 60:.1f}min)"
            )
            continue

        if dry_run:
            logging.info(f"[DRY RUN] WOULD REMOVE EMPTY DIR: {d}")
        else:
            try:
                d.rmdir()
                logging.info(f"REMOVED EMPTY DIR: {d}")
            except OSError as e:
                logging.warning(f"Failed to remove empty dir {d}: {e}")
                continue

        removed += 1

    return removed


def _robust_rmtree(path: Path):
    """
    remove directory tree robustly, handling permission errors.
    raises RuntimeError if directory still exists after deletion attempt.
    """
    # collect errors during rmtree
    rmtree_errors: list[tuple[str, str]] = []

    def onerror(func, fpath, exc_info):
        """try to fix permissions and retry; log failures"""
        try:
            os.chmod(fpath, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        except Exception:
            pass
        try:
            func(fpath)
        except Exception as e:
            rmtree_errors.append((fpath, str(e)))

    try:
        shutil.rmtree(str(path), onerror=onerror)
    except PermissionError as e:
        logging.warning(
            f"Permission error deleting {path}: {e}. Attempting to fix permissions..."
        )
        # try to make everything writable and retry
        for root, dirs, files in os.walk(path):
            for d in dirs:
                try:
                    os.chmod(
                        os.path.join(root, d),
                        stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC,
                    )
                except Exception:
                    pass
            for f in files:
                try:
                    os.chmod(os.path.join(root, f), stat.S_IWRITE | stat.S_IREAD)
                except Exception:
                    pass
        # retry deletion
        rmtree_errors.clear()
        shutil.rmtree(str(path), onerror=onerror)

    # verify deletion actually happened
    if path.exists():
        remaining = list(path.rglob("*"))[:10]
        error_sample = rmtree_errors[:5] if rmtree_errors else []
        raise RuntimeError(
            f"Failed to delete {path}: directory still exists. "
            f"Errors: {error_sample}. "
            f"Remaining files (sample): {[str(p) for p in remaining]}"
        )


def copy_exam_local(
    src: Path, patient_id: str, exam_name: str
) -> Tuple[int, float, int]:
    """
    copy an exam directory locally with benchmarking

    returns (logical_bytes, seconds, rsync_exit_code)
    uses rsync for efficient local copying with progress tracking
    """
    import re

    dest_parent = DST_ROOT / patient_id
    dest_dir = DST_ROOT / patient_id / exam_name

    logging.info(f"COPY: {src} -> {dest_dir}")

    dest_parent.mkdir(parents=True, exist_ok=True)

    logical_bytes_src = _calculate_dir_size(src)

    command = [
        "rsync",
        "-aH",
        "--info=stats2,progress2",
        "--stats",
        f"{src}/",
        str(dest_dir),
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

    assert proc.stdout is not None

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
            "sent ",
            "speedup is ",
        )
        return lowered.startswith(prefixes)

    try:
        for line in proc.stdout:
            if shutdown_requested:
                logging.info("Shutdown requested. Terminating rsync...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise KeyboardInterrupt("Shutdown requested during transfer")

            combined_lines.append(line)
            msg = line.rstrip()
            if msg and should_log(msg):
                logging.info(f"[COPY] {msg}")
            if "to-chk=" in msg:
                last_progress_line = msg

            now = monotonic()
            if now - last_heartbeat >= 60:
                elapsed = now - t0
                hb = f"[COPY HEARTBEAT] {src.name}: elapsed {elapsed:.0f}s" + (
                    f" | progress: {last_progress_line}" if last_progress_line else ""
                )
                logging.info(hb)
                last_heartbeat = now
    except KeyboardInterrupt:
        logging.info("Interrupted during transfer. Terminating rsync...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise

    proc.wait()
    dt = monotonic() - t0

    if shutdown_requested:
        logging.info("Shutdown requested. Exiting...")
        raise KeyboardInterrupt("Shutdown requested")

    rsync_exit_code = proc.returncode
    if rsync_exit_code != 0:
        if rsync_exit_code == 20:
            raise KeyboardInterrupt("rsync was interrupted")
        # exit codes 23/24 are partial transfers - some files couldn't be transferred
        # this is usually harmless (files in use, vanished, etc.) and rsync will retry next run
        if rsync_exit_code in (23, 24):
            logging.warning(
                f"rsync returned code {rsync_exit_code} (partial transfer) for {src.name}. "
                "Some files may not have been transferred. Continuing..."
            )
        else:
            raise subprocess.CalledProcessError(rsync_exit_code, command)

    combined = "".join(combined_lines)

    logical_bytes = logical_bytes_src
    for pat in (
        r"Total transferred file size:\s*([\d,]+)",
        r"Total file size:\s*([\d,]+)",
        r"total size is\s*([\d,]+)",
    ):
        m = re.search(pat, combined, flags=re.I)
        if m:
            logical_bytes = int(m.group(1).replace(",", ""))
            break

    if not dest_dir.exists():
        raise RuntimeError(f"Destination directory was not created: {dest_dir}")

    mb_log = logical_bytes / 1_000_000
    rate_mb_s = (logical_bytes / dt / 1_000_000) if dt > 0 else 0.0
    status = "with warnings" if rsync_exit_code in (23, 24) else "successful"
    logging.info(
        f"COPY {status} for {src.name} [logical {mb_log:.1f} MB in {dt:.1f}s, {rate_mb_s:.2f} MB/s]"
    )
    return logical_bytes, dt, rsync_exit_code


def process_single_exam(
    src_path: Path,
    patient_id: str,
    dry_run: bool,
    immediate_delete: bool,
    deletion_log: list[tuple[str, str]],
) -> Tuple[int, float, int]:
    """
    process a single exam: copy from source to destination.

    returns (logical_bytes, transfer_time_seconds, rsync_exit_code)
    rsync_exit_code is 0 for success, 23/24 for partial transfer warnings
    """
    exam_name = src_path.name

    # stability check
    is_stable, most_recent_age = _is_exam_stable(src_path, STABILITY_THRESHOLD_SEC)
    if not is_stable:
        logging.info(
            f"SKIP UNSTABLE: {exam_name} "
            f"(most recent file {most_recent_age / 60:.1f}min old, "
            f"need {STABILITY_THRESHOLD_SEC / 60:.1f}min)"
        )
        return 0, 0.0, 0

    # check if already exists at destination
    dest_exam = DST_ROOT / patient_id / exam_name
    if dest_exam.exists():
        logging.info(f"SKIP EXISTS: {exam_name} already exists at destination")
        # still queue/delete source since it's already synced
        if dry_run:
            action = "DELETE" if immediate_delete else "QUEUE FOR DELETE"
            logging.info(f"[DRY RUN] WOULD {action} (LOCAL): {src_path}")
            deletion_log.append((str(src_path), f"dry-run {action.lower()} (exists)"))
        else:
            if immediate_delete:
                try:
                    _robust_rmtree(src_path)
                    logging.info(f"DELETED (LOCAL): {src_path}")
                    deletion_log.append((str(src_path), "deleted (exists at dst)"))
                except Exception as e:
                    logging.error(f"Failed to delete {src_path}: {e}")
                    deletion_log.append(
                        (str(src_path), f"failed delete (exists at dst): {e}")
                    )
                    raise
            else:
                dst = DELETE_QUEUE_DIR / patient_id / exam_name
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(src_path), str(dst))
                    logging.info(
                        f"QUEUED FOR DELETE (LOCAL): moved {src_path} -> {dst}"
                    )
                    deletion_log.append((str(src_path), f"queued (exists) -> {dst}"))
                except PermissionError:
                    # if move fails due to permissions, try making writable first
                    logging.warning(
                        f"Permission error moving {src_path}, attempting to fix permissions..."
                    )
                    for root, dirs, files in os.walk(src_path):
                        for d in dirs:
                            try:
                                os.chmod(
                                    os.path.join(root, d),
                                    stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC,
                                )
                            except Exception:
                                pass
                        for f in files:
                            try:
                                os.chmod(
                                    os.path.join(root, f), stat.S_IWRITE | stat.S_IREAD
                                )
                            except Exception:
                                pass
                    shutil.move(str(src_path), str(dst))
                    logging.info(
                        f"QUEUED FOR DELETE (LOCAL): moved {src_path} -> {dst}"
                    )
                    deletion_log.append((str(src_path), f"queued (exists) -> {dst}"))
        return 0, 0.0, 0

    # copy exam
    logging.info(f"NEW: {exam_name} not found at destination. Copying.")
    if dry_run:
        logging.info(
            f"[DRY RUN] WOULD COPY: {src_path} -> {DST_ROOT / patient_id / exam_name}"
        )
        logical_bytes = 0
        transfer_time = 0.0
        rsync_exit_code = 0
    else:
        logical_bytes, transfer_time, rsync_exit_code = copy_exam_local(
            src_path, patient_id, exam_name
        )

    # queue/delete source after transfer
    if dry_run:
        action = "DELETE" if immediate_delete else "QUEUE FOR DELETE"
        logging.info(f"[DRY RUN] WOULD {action} (LOCAL): {src_path}")
        deletion_log.append((str(src_path), f"dry-run {action.lower()}"))
    else:
        if immediate_delete:
            try:
                _robust_rmtree(src_path)
                logging.info(f"DELETED (LOCAL): {src_path}")
                deletion_log.append((str(src_path), "deleted after transfer"))
            except Exception as e:
                msg = f"failed delete after transfer: {e}"
                logging.error(f"Failed to delete {src_path}: {e}")
                deletion_log.append((str(src_path), msg))
                raise
        else:
            dst = DELETE_QUEUE_DIR / patient_id / exam_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(src_path), str(dst))
                logging.info(f"QUEUED FOR DELETE (LOCAL): moved {src_path} -> {dst}")
                deletion_log.append((str(src_path), f"queued for delete -> {dst}"))
            except PermissionError:
                # if move fails due to permissions, try making writable first
                logging.warning(
                    f"Permission error moving {src_path}, attempting to fix permissions..."
                )
                for root, dirs, files in os.walk(src_path):
                    for d in dirs:
                        try:
                            os.chmod(
                                os.path.join(root, d),
                                stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC,
                            )
                        except Exception:
                            pass
                    for f in files:
                        try:
                            os.chmod(
                                os.path.join(root, f), stat.S_IWRITE | stat.S_IREAD
                            )
                        except Exception:
                            pass
                shutil.move(str(src_path), str(dst))
                logging.info(f"QUEUED FOR DELETE (LOCAL): moved {src_path} -> {dst}")
                deletion_log.append((str(src_path), f"queued for delete -> {dst}"))

    return logical_bytes, transfer_time, rsync_exit_code


def _collect_pending_exams(
    dry_run: bool,
) -> tuple[list[tuple[Path, str, int]], int, int]:
    """
    scan source directory and collect exams pending transfer

    returns (pending_exams, skipped_unstable, skipped_exists) where pending_exams
    is a list of (src_path, patient_id, size_bytes) tuples
    """
    logging.info("Step 1: Scanning source directory...")
    scan_start = monotonic()
    logging.info(f"Scanning top-level in {SRC_ROOT} for patient directories...")
    top_dirs = [d for d in SRC_ROOT.iterdir() if d.is_dir()]
    patient_dirs = [d for d in top_dirs if d.name.isdigit()]
    scan_time = monotonic() - scan_start
    logging.info(f"Found {len(patient_dirs)} patient directories in {scan_time:.1f}s")

    ignored_dirs = [
        d for d in top_dirs if not d.name.isdigit() and not d.name.startswith("_")
    ]
    if ignored_dirs:
        sample = ", ".join(sorted(p.name for p in ignored_dirs[:10]))
        logging.info(
            f"Ignoring {len(ignored_dirs)} non-patient root dirs: [{sample}...]"
        )

    # collect all exams
    logging.info("Step 2: Collecting and filtering exam directories...")
    collect_start = monotonic()
    all_exams: list[tuple[Path, str]] = []
    for idx, p_dir in enumerate(patient_dirs):
        if idx % 100 == 0 and idx > 0:
            logging.info(f"  Scanned {idx}/{len(patient_dirs)} patient directories...")
        try:
            for exam_path in p_dir.iterdir():
                if exam_path.is_dir():
                    all_exams.append((exam_path, p_dir.name))
        except Exception as e:
            logging.warning(f"Failed to list exams in {p_dir}: {e}")

    logging.info(
        f"Found {len(all_exams)} total exams. Filtering with {DEFAULT_WORKERS} workers..."
    )

    # filter to pending exams (stable and not already at destination)
    pending: list[tuple[Path, str, int]] = []
    skipped_unstable = 0
    skipped_exists = 0
    filter_start = monotonic()
    last_filter_log = filter_start
    lock = threading.Lock()
    completed_filter = 0

    def filter_exam(
        item: tuple[Path, str],
    ) -> tuple[str, Path, str, int, float] | None:
        """
        check if exam should be transferred; returns (status, src, patient_id, size, age)
        status is 'pending', 'exists', or 'unstable'
        """
        src_path, patient_id = item

        # stability check first (slow - walks SMB, but we need size anyway)
        is_stable, most_recent_age = _is_exam_stable(src_path, STABILITY_THRESHOLD_SEC)
        if not is_stable:
            return ("unstable", src_path, patient_id, 0, most_recent_age)

        # compute source size (slow - walks SMB)
        src_size = _calculate_dir_size(src_path)

        # check if destination exists and has same size (fast local check)
        dest_exam = DST_ROOT / patient_id / src_path.name
        if dest_exam.exists():
            dest_size = _calculate_dir_size(dest_exam)
            if dest_size == src_size:
                return ("exists", src_path, patient_id, 0, 0.0)
            # destination exists but size differs - needs sync (partial transfer)
            # return the difference as size estimate for progress tracking
            return ("pending", src_path, patient_id, max(0, src_size - dest_size), 0.0)

        return ("pending", src_path, patient_id, src_size, 0.0)

    with ThreadPoolExecutor(max_workers=DEFAULT_WORKERS) as executor:
        futures = {executor.submit(filter_exam, item): item for item in all_exams}

        for future in as_completed(futures):
            if shutdown_requested:
                executor.shutdown(wait=False)
                break

            result = future.result()
            if result is None:
                continue

            status, src_path, patient_id, size_bytes, most_recent_age = result

            with lock:
                completed_filter += 1

                if status == "exists":
                    if dry_run:
                        logging.info(
                            f"[DRY RUN] SKIP EXISTS: {src_path.name} already at destination (size matches)"
                        )
                    skipped_exists += 1
                elif status == "unstable":
                    if dry_run:
                        logging.info(
                            f"[DRY RUN] SKIP UNSTABLE: {src_path.name} "
                            f"(most recent file {most_recent_age / 60:.1f}min old)"
                        )
                    skipped_unstable += 1
                else:  # pending
                    pending.append((src_path, patient_id, size_bytes))
                    dest_exam = DST_ROOT / patient_id / src_path.name
                    if dry_run and dest_exam.exists():
                        logging.info(
                            f"[DRY RUN] RESUME PARTIAL: {src_path.name} "
                            f"(~{_format_size(size_bytes)} remaining)"
                        )

                # progress logging every 10 seconds
                now = monotonic()
                if now - last_filter_log >= 10.0:
                    elapsed = now - filter_start
                    rate = completed_filter / elapsed if elapsed > 0 else 0
                    remaining = len(all_exams) - completed_filter
                    eta = remaining / rate if rate > 0 else 0
                    logging.info(
                        f"  Filtering {completed_filter}/{len(all_exams)} exams... "
                        f"({skipped_exists} exist, {skipped_unstable} unstable, {len(pending)} pending) "
                        f"ETA≈{_format_interval(eta)}"
                    )
                    last_filter_log = now

    collect_time = monotonic() - collect_start
    total_pending_bytes = sum(size for _, _, size in pending)
    logging.info(
        f"Filtering took {collect_time:.1f}s. "
        f"Pending: {len(pending)} exams ({_format_size(total_pending_bytes)}), "
        f"skipped: {skipped_unstable} unstable, {skipped_exists} already exist"
    )

    return pending, skipped_unstable, skipped_exists


def run_single_sync(
    dry_run: bool, immediate_delete: bool, workers: int = DEFAULT_WORKERS
):
    """run a single sync pass with parallel transfers"""
    global shutdown_requested
    setup_signal_handlers()

    if dry_run:
        logging.info(
            "--- DRY RUN MODE ENABLED: No files will be moved or transferred. ---"
        )

    logging.info("=== STARTING SYNC ===")

    used_gb, free_gb, total_gb = _share_usage_gb()
    logging.info(
        f"[USAGE mount] {SRC_ROOT}: used {used_gb:.1f} GB / total {total_gb:.1f} GB (free {free_gb:.1f} GB)"
    )

    # phase 1: collect pending exams with sizes
    pending_exams, skipped_unstable, skipped_exists = _collect_pending_exams(dry_run)

    if not pending_exams:
        logging.info("No exams pending transfer. Nothing to do.")
        _cleanup_empty_patient_dirs(dry_run)
        return

    if shutdown_requested:
        logging.info("Shutdown requested during scan. Exiting.")
        return

    total_bytes = sum(size for _, _, size in pending_exams)
    total_exams = len(pending_exams)
    logging.info(
        f"=== STARTING PARALLEL TRANSFER: {total_exams} exams, "
        f"{_format_size(total_bytes)}, {workers} workers ==="
    )

    # phase 2: parallel transfer with progress tracking
    sync_start = monotonic()
    PROGRESS_INTERVAL_SEC = 30.0
    RATE_WINDOW_SEC = 3600.0  # 1 hour sliding window for rate calculation

    # thread-safe counters
    lock = threading.Lock()
    completed_bytes = 0
    completed_exams = 0
    transfers_done = 0
    transfer_time_total = 0.0
    deletion_log: list[tuple[str, str]] = []
    partial_transfers: list[tuple[str, int]] = []  # (exam_path, rsync_exit_code)
    first_error: Exception | None = None
    progress_stop = threading.Event()
    # sliding window: list of (timestamp, cumulative_bytes) for rate calculation
    progress_samples: list[tuple[float, int]] = [(sync_start, 0)]

    def process_exam_wrapper(
        item: tuple[Path, str, int],
    ) -> tuple[Path, int, float, int, list[tuple[str, str]]]:
        """wrapper for thread pool - returns (path, bytes, time, rsync_exit_code, deletion_entries)"""
        src_path, patient_id, expected_size = item
        local_deletion_log: list[tuple[str, str]] = []

        if shutdown_requested:
            return src_path, 0, 0.0, 0, local_deletion_log

        logical_bytes, transfer_time, rsync_exit_code = process_single_exam(
            src_path, patient_id, dry_run, immediate_delete, local_deletion_log
        )
        return (
            src_path,
            logical_bytes,
            transfer_time,
            rsync_exit_code,
            local_deletion_log,
        )

    def log_progress() -> None:
        """log progress bar - call with lock held, uses 5-min sliding window for rate"""
        now = monotonic()
        elapsed = now - sync_start

        # record sample for sliding window
        progress_samples.append((now, completed_bytes))

        # prune old samples outside the window
        cutoff = now - RATE_WINDOW_SEC
        while len(progress_samples) > 1 and progress_samples[0][0] < cutoff:
            progress_samples.pop(0)

        # calculate rate from sliding window
        if len(progress_samples) >= 2:
            oldest_time, oldest_bytes = progress_samples[0]
            window_elapsed = now - oldest_time
            window_bytes = completed_bytes - oldest_bytes
            rate_mb_s = (
                window_bytes / window_elapsed / 1_000_000 if window_elapsed > 0 else 0.0
            )
        else:
            rate_mb_s = completed_bytes / elapsed / 1_000_000 if elapsed > 0 else 0.0

        if total_bytes > 0:
            fraction = completed_bytes / total_bytes
            pct = min(100.0, fraction * 100.0)
            bar = _format_progress_bar(fraction)
            remaining_bytes = total_bytes - completed_bytes
            if rate_mb_s > 0:
                eta_sec = remaining_bytes / (rate_mb_s * 1_000_000)
                eta = _format_interval(eta_sec)
            else:
                eta = "??:??"
            in_flight = min(workers, total_exams - completed_exams)
            logging.info(
                f"[PROGRESS] {bar} {pct:5.1f}%  "
                f"{_format_size(completed_bytes)} / {_format_size(total_bytes)}  "
                f"elapsed={_format_interval(elapsed)} ETA≈{eta}  "
                f"rate={rate_mb_s:.1f} MB/s ({completed_exams}/{total_exams} done, {in_flight} in-flight)"
            )
        else:
            fraction = completed_exams / total_exams if total_exams > 0 else 0
            bar = _format_progress_bar(fraction)
            logging.info(
                f"[PROGRESS] {bar} {completed_exams}/{total_exams} exams  "
                f"elapsed={_format_interval(elapsed)}"
            )

    def progress_thread_fn() -> None:
        """background thread that logs progress every PROGRESS_INTERVAL_SEC"""
        while not progress_stop.wait(timeout=PROGRESS_INTERVAL_SEC):
            if shutdown_requested:
                break
            with lock:
                log_progress()

    if dry_run:
        # in dry-run, just log what would happen (already done in _collect_pending_exams)
        for src_path, patient_id, size_bytes in pending_exams:
            logging.info(
                f"[DRY RUN] WOULD COPY: {src_path} -> "
                f"{DST_ROOT / patient_id / src_path.name} ({_format_size(size_bytes)})"
            )
            action = "DELETE" if immediate_delete else "QUEUE FOR DELETE"
            logging.info(f"[DRY RUN] WOULD {action} (LOCAL): {src_path}")
            deletion_log.append((str(src_path), f"dry-run {action.lower()}"))
            completed_bytes += size_bytes
            completed_exams += 1
        transfers_done = total_exams
    else:
        # start background progress thread
        progress_thread = threading.Thread(target=progress_thread_fn, daemon=True)
        progress_thread.start()

        # actual parallel transfer
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_exam_wrapper, item): item
                for item in pending_exams
            }

            for future in as_completed(futures):
                if first_error is not None:
                    # already failed, just drain remaining futures
                    continue

                item = futures[future]
                src_path, patient_id, expected_size = item

                try:
                    (
                        _,
                        logical_bytes,
                        transfer_time,
                        rsync_exit_code,
                        local_deletions,
                    ) = future.result()

                    with lock:
                        completed_bytes += expected_size
                        completed_exams += 1
                        if logical_bytes > 0:
                            transfers_done += 1
                            transfer_time_total += transfer_time
                        if rsync_exit_code in (23, 24):
                            partial_transfers.append((str(src_path), rsync_exit_code))
                        deletion_log.extend(local_deletions)
                        log_progress()

                except KeyboardInterrupt:
                    logging.info("KeyboardInterrupt received. Cancelling remaining...")
                    shutdown_requested = True
                    executor.shutdown(wait=False)
                    raise

                except Exception as e:
                    logging.error(
                        f"FATAL ERROR processing {src_path}: {e}", exc_info=True
                    )
                    first_error = e
                    shutdown_requested = True
                    executor.shutdown(wait=False)

        # stop progress thread
        progress_stop.set()
        progress_thread.join(timeout=1.0)

        if first_error is not None:
            logging.error("Exiting immediately on error (fail-fast mode).")
            raise first_error

    # log final progress
    with lock:
        log_progress()

    total_elapsed = monotonic() - sync_start

    logging.info("=== SYNC COMPLETE ===")
    logging.info(f"Total time: {total_elapsed / 60:.1f} minutes")
    if total_elapsed > 0 and completed_exams > 0:
        logging.info(
            f"Overall rate: {completed_exams / total_elapsed * 3600:.1f} exams/hour"
        )
    logging.info(f"Processed {completed_exams}/{total_exams} exams.")

    logging.info("SUMMARY:")
    logging.info(f"  - Transferred: {transfers_done}")
    logging.info(f"  - Skipped (unstable): {skipped_unstable}")
    logging.info(f"  - Skipped (already exists): {skipped_exists}")
    logging.info(f"  - Total bytes transferred: {_format_size(completed_bytes)}")

    if transfers_done > 0 and not dry_run:
        overall_avg = (
            completed_bytes / total_elapsed / 1_000_000 if total_elapsed > 0 else 0.0
        )
        transfer_only_avg = (
            completed_bytes / transfer_time_total / 1_000_000
            if transfer_time_total > 0
            else 0.0
        )
        logging.info("  - Transfer throughput:")
        logging.info(
            f"    * Overall average: {overall_avg:.2f} MB/s (includes all processing)"
        )
        logging.info(
            f"    * Transfer-only average: {transfer_only_avg:.2f} MB/s ({transfer_time_total:.1f}s transfer time)"
        )
        if transfer_time_total > 0:
            transfer_pct = (
                (transfer_time_total / total_elapsed * 100)
                if total_elapsed > 0
                else 0.0
            )
            logging.info(
                f"    * Transfer time: {transfer_time_total:.1f}s ({transfer_pct:.1f}% of total)"
            )

    deleted = sum(1 for _, status in deletion_log if status.startswith("deleted"))
    queued = sum(1 for _, status in deletion_log if status.startswith("queued"))
    dry_runs = sum(1 for _, status in deletion_log if status.startswith("dry-run"))
    failures = [(p, s) for p, s in deletion_log if s.startswith("failed")]
    logging.info("DELETE SUMMARY:")
    logging.info(f"  - Deleted: {deleted}")
    logging.info(f"  - Queued: {queued}")
    logging.info(f"  - Dry-run actions: {dry_runs}")
    logging.info(f"  - Failures: {len(failures)}")
    if failures:
        max_show = 10
        for path, status in failures[:max_show]:
            logging.info(f"    * {path} -> {status}")
        if len(failures) > max_show:
            logging.info(f"    ... plus {len(failures) - max_show} more")

    if partial_transfers:
        logging.warning(f"PARTIAL TRANSFERS ({len(partial_transfers)} exams):")
        logging.warning(
            "  These exams had rsync warnings (code 23/24). Some files may not have transferred."
        )
        logging.warning("  Re-run sync to retry, or manually verify these exams:")
        for exam_path, exit_code in partial_transfers:
            logging.warning(f"    * {exam_path} (rsync exit code {exit_code})")

    used_gb, free_gb, total_gb = _share_usage_gb()
    logging.info(
        f"[FINAL USAGE] {SRC_ROOT}: used {used_gb:.1f} GB / total {total_gb:.1f} GB (free {free_gb:.1f} GB)"
    )

    # clean up empty patient directories
    logging.info("Cleaning up empty patient directories...")
    empty_removed = _cleanup_empty_patient_dirs(dry_run)
    if empty_removed > 0:
        action = "would remove" if dry_run else "removed"
        logging.info(f"  - Empty patient directories {action}: {empty_removed}")

    # clean up rsync partial directories from destination
    logging.info("Cleaning up rsync partial directories...")
    partial_removed = 0
    partial_bytes = 0
    for partial_dir in DST_ROOT.rglob(".rsync-partial"):
        if partial_dir.is_dir():
            dir_size = _calculate_dir_size(partial_dir)
            if dry_run:
                logging.info(
                    f"[DRY RUN] WOULD REMOVE: {partial_dir} ({_format_size(dir_size)})"
                )
            else:
                try:
                    shutil.rmtree(partial_dir)
                    logging.info(f"REMOVED: {partial_dir} ({_format_size(dir_size)})")
                except Exception as e:
                    logging.warning(f"Failed to remove {partial_dir}: {e}")
                    continue
            partial_removed += 1
            partial_bytes += dir_size
    if partial_removed > 0:
        action = "would remove" if dry_run else "removed"
        logging.info(
            f"  - Rsync partial directories {action}: {partial_removed} "
            f"({_format_size(partial_bytes)})"
        )


def run_with_auto_restart(
    dry_run: bool, immediate_delete: bool, workers: int = DEFAULT_WORKERS
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
            run_single_sync(dry_run, immediate_delete, workers)

            if shutdown_requested:
                logging.info("Shutdown requested during sync. Exiting.")
                break

            logging.info(
                f"Sync completed successfully. Waiting "
                f"{RESTART_DELAY_SEC} seconds before restart..."
            )

            for _ in range(RESTART_DELAY_SEC):
                if shutdown_requested:
                    logging.info("Shutdown requested during restart delay. Exiting.")
                    return
                time.sleep(1)

        except KeyboardInterrupt:
            logging.info("Received KeyboardInterrupt. Shutting down gracefully...")
            break
        except Exception as e:
            logging.error(f"Sync failed with error: {e}", exc_info=True)
            logging.error("Exiting immediately on error (fail-fast mode).")
            raise


def main(
    dry_run: bool,
    immediate_delete: bool,
    auto_restart: bool,
    workers: int = DEFAULT_WORKERS,
):
    """main entry point with optional auto-restart"""
    if auto_restart:
        run_with_auto_restart(dry_run, immediate_delete, workers)
    else:
        run_single_sync(dry_run, immediate_delete, workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple sync script to copy DICOM exams from source to destination."
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
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel rsync processes (default: {DEFAULT_WORKERS}).",
    )
    args = parser.parse_args()
    main(
        dry_run=args.dry_run,
        immediate_delete=not args.no_immediate_delete,
        auto_restart=not args.no_auto_restart,
        workers=args.workers,
    )
