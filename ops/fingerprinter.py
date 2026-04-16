#!/usr/bin/env python
# ruff: noqa: E402
# THIS SCRIPT RUNS ON THE REMOTE (HUO-LAB) SERVER

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import psutil
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the shared logic
from prima.fingerprint_utils import create_exam_fingerprint

# --- NEW: Checkpoint Configuration ---
CHECKPOINT_DIR = Path("data/fingerprint_checkpoints")


def log_memory_usage(stage: str):
    """Log current memory usage for debugging."""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    logging.info(f"[{stage}] Memory usage: {memory_mb:.1f} MB")


def load_checkpoint_file(checkpoint_file: Path):
    """Load a single checkpoint file and return patient data."""
    try:
        with open(checkpoint_file) as f:
            checkpoint_data = json.load(f)

        # Handle both old format (direct inventory) and new format (with metadata)
        if isinstance(checkpoint_data, dict) and "inventory" in checkpoint_data:
            # New format with metadata
            patient_data = checkpoint_data["inventory"]
        else:
            # Old format - direct inventory
            patient_data = checkpoint_data

        return checkpoint_file.stem, patient_data
    except Exception as e:
        logging.error(f"Failed to load checkpoint {checkpoint_file}: {e}")
        return None, None


def fingerprint_all_exams_for_patient(patient_dir: Path):
    """
    Worker function to process all exams for a single patient.
    This is a better unit of work for checkpointing.
    """
    patient_id = patient_dir.name
    patient_inventory = {}
    failure_reasons = {}

    exam_dirs = [d for d in patient_dir.iterdir() if d.is_dir()]
    for exam_path in exam_dirs:
        try:
            fingerprint, reason = create_exam_fingerprint(exam_path)
            if fingerprint and fingerprint.is_valid():
                exam_name = exam_path.name
                # store: (study_uid, file_hashes, study_date, study_time)
                result = (
                    fingerprint.study_uid,
                    sorted(fingerprint.file_hashes),
                    fingerprint.study_date,
                    fingerprint.study_time,
                )
                patient_inventory[exam_name] = result
            else:
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        except Exception as e:
            reason = f"Worker exception on {exam_path.name}: {e}"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    return patient_id, patient_inventory, failure_reasons


def consolidate_checkpoints(final_output_file: Path, parallel_jobs: int = 4):
    """Reads all patient checkpoint files and combines them into one final inventory using parallel processing."""
    start_time = time.time()
    logging.info(f"Consolidating checkpoints from {CHECKPOINT_DIR}...")
    log_memory_usage("consolidation_start")

    # only consolidate numeric patient ids; ignore stray files like 'transferred.json' or '2O*.json'
    all_checkpoint_files = list(CHECKPOINT_DIR.glob("*.json"))
    checkpoint_files = [p for p in all_checkpoint_files if p.stem.isdigit()]
    skipped = [p for p in all_checkpoint_files if not p.stem.isdigit()]
    logging.info(
        f"Found {len(checkpoint_files)} numeric checkpoint files to consolidate"
    )
    if skipped:
        preview = ", ".join(sorted(p.stem for p in skipped)[:10])
        logging.info(
            f"Ignoring {len(skipped)} non-patient checkpoint(s): [{preview}...]"
        )

    # sample accessibility check on a few files
    accessible_files = []
    for checkpoint_file in checkpoint_files[:10]:
        try:
            if checkpoint_file.stat().st_size > 0:
                accessible_files.append(checkpoint_file)
        except Exception as e:
            logging.warning(f"Cannot access {checkpoint_file}: {e}")

    logging.info(
        f"Sample check: {len(accessible_files)}/{min(10, len(checkpoint_files))} files are accessible"
    )

    if not checkpoint_files:
        logging.warning("No checkpoint files found!")
        return {}

    # Try batch processing first (most reliable), then parallel, then sequential
    try:
        logging.info("Attempting batch consolidation (most reliable)...")
        full_inventory = _consolidate_batch(checkpoint_files, batch_size=50)
        logging.info("Batch consolidation completed successfully")
    except Exception as e:
        logging.warning(
            f"Batch consolidation failed: {e}. Trying parallel processing..."
        )
        try:
            full_inventory = _consolidate_parallel(checkpoint_files, parallel_jobs)
            logging.info("Parallel consolidation completed successfully")
        except Exception as e2:
            logging.warning(
                f"Parallel consolidation failed: {e2}. Falling back to sequential processing."
            )
            full_inventory = _consolidate_sequential(checkpoint_files)

    log_memory_usage("before_json_write")

    # Write the consolidated inventory
    write_start = time.time()
    logging.info(f"Writing final inventory to {final_output_file}...")
    final_output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(final_output_file, "w") as f:
        json.dump(full_inventory, f, indent=None, separators=(",", ":"))

    write_time = time.time() - write_start
    total_time = time.time() - start_time

    logging.info(f"Final inventory written successfully in {write_time:.1f} seconds")
    logging.info(f"Total consolidation time: {total_time:.1f} seconds")
    log_memory_usage("consolidation_complete")

    return full_inventory


def _consolidate_parallel(checkpoint_files, parallel_jobs):
    """Parallel consolidation implementation."""
    full_inventory = {}
    failed_files = []

    logging.info(
        f"Starting parallel consolidation with {len(checkpoint_files)} files using {parallel_jobs} workers"
    )
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=parallel_jobs) as executor:
        # Submit all checkpoint files for parallel loading
        future_to_file = {
            executor.submit(load_checkpoint_file, checkpoint_file): checkpoint_file
            for checkpoint_file in checkpoint_files
        }

        logging.info(f"Submitted {len(future_to_file)} tasks to executor")

        # Process results as they complete
        with tqdm(total=len(checkpoint_files), desc="Loading checkpoints") as pbar:
            last_heartbeat = time.time()
            for future in as_completed(
                future_to_file, timeout=300
            ):  # 5 minute timeout per future
                checkpoint_file = future_to_file[future]
                try:
                    patient_id, patient_data = future.result()
                    if patient_id is not None and patient_data is not None:
                        full_inventory[patient_id] = patient_data
                    else:
                        failed_files.append(checkpoint_file)
                except Exception as e:
                    logging.error(f"Failed to process {checkpoint_file}: {e}")
                    failed_files.append(checkpoint_file)

                pbar.update(1)

                # Heartbeat every 30 seconds
                current_time = time.time()
                if current_time - last_heartbeat > 30:
                    elapsed = current_time - start_time
                    logging.info(
                        f"HEARTBEAT: Processed {pbar.n}/{len(checkpoint_files)} files in {elapsed:.1f}s"
                    )
                    last_heartbeat = current_time

                # Log progress every 100 files
                if pbar.n % 100 == 0:
                    log_memory_usage(f"loaded_{pbar.n}_files")

    if failed_files:
        logging.warning(f"Failed to load {len(failed_files)} checkpoint files")

    return full_inventory


def _consolidate_sequential(checkpoint_files):
    """Sequential consolidation implementation as fallback."""
    full_inventory = {}
    failed_files = []

    logging.info(
        f"Starting sequential consolidation with {len(checkpoint_files)} files"
    )
    start_time = time.time()
    last_heartbeat = start_time

    for i, checkpoint_file in enumerate(
        tqdm(checkpoint_files, desc="Loading checkpoints (sequential)")
    ):
        try:
            patient_id, patient_data = load_checkpoint_file(checkpoint_file)
            if patient_id is not None and patient_data is not None:
                full_inventory[patient_id] = patient_data
            else:
                failed_files.append(checkpoint_file)
        except Exception as e:
            logging.error(f"Failed to process {checkpoint_file}: {e}")
            failed_files.append(checkpoint_file)

        # Heartbeat every 30 seconds
        current_time = time.time()
        if current_time - last_heartbeat > 30:
            elapsed = current_time - start_time
            logging.info(
                f"HEARTBEAT: Processed {i + 1}/{len(checkpoint_files)} files in {elapsed:.1f}s"
            )
            last_heartbeat = current_time

    if failed_files:
        logging.warning(f"Failed to load {len(failed_files)} checkpoint files")

    return full_inventory


def _consolidate_batch(checkpoint_files, batch_size=100):
    """Batch consolidation implementation - processes files in small batches."""
    full_inventory = {}
    failed_files = []

    logging.info(
        f"Starting batch consolidation with {len(checkpoint_files)} files in batches of {batch_size}"
    )
    start_time = time.time()

    for i in range(0, len(checkpoint_files), batch_size):
        batch = checkpoint_files[i : i + batch_size]
        batch_start = time.time()

        logging.info(
            f"Processing batch {i // batch_size + 1}/{(len(checkpoint_files) + batch_size - 1) // batch_size} ({len(batch)} files)"
        )

        for checkpoint_file in batch:
            try:
                patient_id, patient_data = load_checkpoint_file(checkpoint_file)
                if patient_id is not None and patient_data is not None:
                    full_inventory[patient_id] = patient_data
                else:
                    failed_files.append(checkpoint_file)
            except Exception as e:
                logging.error(f"Failed to process {checkpoint_file}: {e}")
                failed_files.append(checkpoint_file)

        batch_time = time.time() - batch_start
        elapsed = time.time() - start_time
        logging.info(f"Batch completed in {batch_time:.1f}s (total: {elapsed:.1f}s)")
        log_memory_usage(f"batch_{i // batch_size + 1}_complete")

    if failed_files:
        logging.warning(f"Failed to load {len(failed_files)} checkpoint files")

    return full_inventory


def main(root_dir: Path, output_file: Path, parallel_jobs: int):
    start_time = time.time()
    logging.info(f"Starting remote fingerprinting of {root_dir}")
    log_memory_usage("main_start")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # only treat numeric directories as patients; skip staging/trash like 'transferred' or '2O*'
    all_patient_dirs_raw = [d for d in root_dir.iterdir() if d.is_dir()]
    patient_dirs = sorted(
        [d for d in all_patient_dirs_raw if d.name.isdigit()], key=lambda p: p.name
    )
    ignored = sorted([d.name for d in all_patient_dirs_raw if not d.name.isdigit()])
    if ignored:
        preview = ", ".join(ignored[:10])
        logging.info(f"Ignoring {len(ignored)} non-patient root dirs: [{preview}...]")

    completed_patient_ids = {
        p.stem for p in CHECKPOINT_DIR.glob("*.json") if p.stem.isdigit()
    }
    patients_to_process = [
        p for p in patient_dirs if p.name not in completed_patient_ids
    ]

    logging.info(f"Found {len(patient_dirs)} total patients in source directory.")
    logging.info(f"Found {len(completed_patient_ids)} already completed checkpoints.")

    if patients_to_process:
        logging.info(f"Patients to process: {[p.name for p in patients_to_process]}")
    else:
        logging.info("All patients already fingerprinted. Proceeding to consolidation.")

    if patients_to_process:
        logging.info(
            f"Resuming... {len(patients_to_process)} patients remaining to be processed."
        )

        overall_failure_reasons = {}

        logging.info(f"Using {parallel_jobs} parallel workers")
        log_memory_usage("before_processing")

        with ProcessPoolExecutor(max_workers=parallel_jobs) as executor:
            # submit all tasks upfront - executor manages the queue
            futures = {
                executor.submit(fingerprint_all_exams_for_patient, path): path
                for path in patients_to_process
            }
            logging.info(f"Submitted {len(futures)} tasks to executor")

            # process results as they complete (streaming)
            with tqdm(
                total=len(patients_to_process), desc="Fingerprinting Patients"
            ) as pbar:
                for future in as_completed(futures):
                    try:
                        patient_id, patient_inventory, failure_reasons = future.result()

                        checkpoint_data = {
                            "inventory": patient_inventory,
                            "failure_reasons": failure_reasons,
                            "processed_at": time.time(),
                        }

                        with open(CHECKPOINT_DIR / f"{patient_id}.json", "w") as f:
                            json.dump(checkpoint_data, f)

                        logging.info(
                            f"Saved checkpoint for patient {patient_id} "
                            f"({len(patient_inventory)} exams, {sum(failure_reasons.values())} failures)"
                        )

                        for reason, count in failure_reasons.items():
                            overall_failure_reasons[reason] = (
                                overall_failure_reasons.get(reason, 0) + count
                            )

                    except Exception as e:
                        path = futures[future]
                        logging.error(
                            f"A master process future failed for patient {path.name}: {e}"
                        )

                    pbar.update(1)

    # Consolidation and summary
    processing_time = time.time() - start_time
    logging.info(f"Patient processing completed in {processing_time:.1f} seconds")
    log_memory_usage("before_consolidation")

    consolidation_start = time.time()
    logging.info("Starting consolidation phase...")

    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Consolidation timed out after 30 minutes")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(1800)

    try:
        full_inventory = consolidate_checkpoints(output_file, parallel_jobs)
        signal.alarm(0)
    except TimeoutError:
        logging.error("Consolidation timed out after 30 minutes!")
        logging.error(
            "This suggests the parallel processing is hanging. Check system resources."
        )
        return
    except Exception as e:
        signal.alarm(0)
        logging.error(f"Consolidation failed: {e}")
        return

    consolidation_time = time.time() - consolidation_start

    total_exams_fingerprinted = sum(len(v) for v in full_inventory.values())
    total_time = time.time() - start_time

    logging.info("\n" + "=" * 50)
    logging.info("--- Fingerprinting Run Complete ---")
    logging.info(f"Total patients in final inventory: {len(full_inventory):,}")
    logging.info(f"Total exams in final inventory:  {total_exams_fingerprinted:,}")
    logging.info(f"Processing time: {processing_time:.1f} seconds")
    logging.info(f"Consolidation time: {consolidation_time:.1f} seconds")
    logging.info(f"Total runtime: {total_time:.1f} seconds")
    if "overall_failure_reasons" in locals() and overall_failure_reasons:
        logging.info("--- Failure Summary During Run ---")
        for reason, count in sorted(
            overall_failure_reasons.items(), key=lambda item: item[1], reverse=True
        ):
            logging.info(f"  - [{count:,}]: {reason}")
    logging.info("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scans and fingerprints a DICOM directory structure with checkpointing."
    )
    parser.add_argument("root_dir", type=Path, help="The root directory to scan.")
    parser.add_argument(
        "output_file", type=Path, help="The final JSON file to write the inventory to."
    )
    parser.add_argument(
        "-p",
        "--parallel-jobs",
        type=int,
        default=8,
        help="Number of parallel processes.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )
    main(args.root_dir, args.output_file, args.parallel_jobs)
