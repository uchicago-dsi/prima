#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import stat
import subprocess
import time
import warnings
from pathlib import Path
from time import monotonic
from typing import Tuple

from tqdm import tqdm

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


def copy_exam_local(src: Path, patient_id: str, exam_name: str) -> Tuple[int, float]:
    """
    copy an exam directory locally with benchmarking; returns (logical_bytes, seconds)

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

    if proc.returncode != 0:
        if proc.returncode == 20:
            raise KeyboardInterrupt("rsync was interrupted")
        raise subprocess.CalledProcessError(proc.returncode, command)

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
    logging.info(
        f"COPY successful for {src.name} [logical {mb_log:.1f} MB in {dt:.1f}s, {rate_mb_s:.2f} MB/s]"
    )
    return logical_bytes, dt


def process_single_exam(
    src_path: Path,
    patient_id: str,
    dry_run: bool,
    immediate_delete: bool,
    deletion_log: list[tuple[str, str]],
) -> Tuple[int, float]:
    """
    process a single exam: copy from source to destination.

    returns (logical_bytes, transfer_time_seconds)
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
        return 0, 0.0

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
        return 0, 0.0

    # copy exam
    logging.info(f"NEW: {exam_name} not found at destination. Copying.")
    if dry_run:
        logging.info(
            f"[DRY RUN] WOULD COPY: {src_path} -> {DST_ROOT / patient_id / exam_name}"
        )
        logical_bytes = 0
        transfer_time = 0.0
    else:
        logical_bytes, transfer_time = copy_exam_local(src_path, patient_id, exam_name)

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

    return logical_bytes, transfer_time


def run_single_sync(dry_run: bool, immediate_delete: bool):
    """run a single sync pass"""
    setup_signal_handlers()

    if dry_run:
        logging.info(
            "--- DRY RUN MODE ENABLED: No files will be moved or transferred. ---"
        )

    logging.info("=== STARTING SYNC ===")

    # discover source exams
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
    logging.info("Step 2: Collecting all exam directories...")
    collect_start = monotonic()
    all_exams = []
    for idx, p_dir in enumerate(patient_dirs):
        if idx % 100 == 0 and idx > 0:
            logging.info(
                f"  Processed {idx}/{len(patient_dirs)} patient directories..."
            )
        try:
            exam_paths = [p for p in p_dir.iterdir() if p.is_dir()]
            all_exams.extend(exam_paths)
        except Exception as e:
            logging.warning(f"Failed to list exams in {p_dir}: {e}")

    collect_time = monotonic() - collect_start
    total_exams = len(all_exams)
    logging.info(f"Exam collection took {collect_time:.1f}s")
    logging.info(f"Found {total_exams} source exams to process.")

    # process exams
    logging.info("=== PROCESSING EXAMS ===")

    sync_start = monotonic()
    last_hb = sync_start
    bytes_log_total = 0
    transfers_done = 0
    transfer_time_total = 0.0
    completed = 0
    skipped = 0
    HEARTBEAT_SEC = 60
    deletion_log: list[tuple[str, str]] = []

    used_gb, free_gb, total_gb = _share_usage_gb()
    logging.info(
        f"[USAGE mount] {SRC_ROOT}: used {used_gb:.1f} GB / total {total_gb:.1f} GB (free {free_gb:.1f} GB)"
    )

    pbar = tqdm(total=total_exams, desc="Processing Exams")

    for src_path in all_exams:
        if shutdown_requested:
            logging.info("Shutdown requested. Exiting gracefully...")
            break

        try:
            patient_id = src_path.parent.name
            logical_bytes, transfer_time = process_single_exam(
                src_path, patient_id, dry_run, immediate_delete, deletion_log
            )

            bytes_log_total += logical_bytes
            if logical_bytes > 0:
                transfers_done += 1
                transfer_time_total += transfer_time
            else:
                skipped += 1
            completed += 1
            pbar.update(1)

            # heartbeat
            now = monotonic()
            if now - last_hb >= HEARTBEAT_SEC:
                elapsed = now - sync_start
                used_gb, free_gb, total_gb = _share_usage_gb()

                avg_rate = (
                    (bytes_log_total / elapsed / 1_000_000) if elapsed > 0 else 0.0
                )
                transfer_avg_rate = (
                    (bytes_log_total / transfer_time_total / 1_000_000)
                    if transfer_time_total > 0
                    else 0.0
                )

                logging.info(
                    f"[HEARTBEAT] processed {completed}/{total_exams} | "
                    f"{transfers_done} transferred ({bytes_log_total / 1e6:.1f} MB) | "
                    f"{skipped} skipped | "
                    f"throughput: {avg_rate:.2f} MB/s overall, "
                    f"{transfer_avg_rate:.2f} MB/s transfer-only | "
                    f"rate {(completed / elapsed * 3600.0) if elapsed > 0 else 0.0:.1f} exams/hr | "
                    f"share {used_gb:.1f}/{total_gb:.1f} GB used"
                )
                last_hb = now

        except KeyboardInterrupt:
            logging.info("Received KeyboardInterrupt. Exiting...")
            raise
        except Exception as e:
            logging.error(f"FATAL ERROR processing {src_path}: {e}", exc_info=True)
            logging.error("Exiting immediately on error.")
            raise

    pbar.close()

    total_elapsed = monotonic() - sync_start

    logging.info("=== SYNC COMPLETE ===")
    logging.info(f"Total time: {total_elapsed / 60:.1f} minutes")
    if total_elapsed > 0 and completed > 0:
        logging.info(f"Overall rate: {completed / total_elapsed * 3600:.1f} exams/hour")
    logging.info(f"Processed {completed}/{total_exams} exams.")

    logging.info("SUMMARY:")
    logging.info(f"  - Transferred: {transfers_done}")
    logging.info(f"  - Skipped (unstable/exists): {skipped}")
    logging.info(
        f"  - Total bytes transferred: {bytes_log_total / 1e6:.1f} MB ({bytes_log_total / 1e9:.2f} GB)"
    )

    if transfers_done > 0:
        overall_avg = (
            bytes_log_total / total_elapsed / 1_000_000 if total_elapsed > 0 else 0.0
        )
        transfer_only_avg = (
            bytes_log_total / transfer_time_total / 1_000_000
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


def run_with_auto_restart(dry_run: bool, immediate_delete: bool):
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
            run_single_sync(dry_run, immediate_delete)

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


def main(dry_run: bool, immediate_delete: bool, auto_restart: bool):
    """main entry point with optional auto-restart"""
    if auto_restart:
        run_with_auto_restart(dry_run, immediate_delete)
    else:
        run_single_sync(dry_run, immediate_delete)


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
    args = parser.parse_args()
    main(
        dry_run=args.dry_run,
        immediate_delete=not args.no_immediate_delete,
        auto_restart=not args.no_auto_restart,
    )
