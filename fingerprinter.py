#!/usr/bin/env python
# coding: utf-8
# THIS SCRIPT RUNS ON THE REMOTE (HUO-LAB) SERVER

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# Import the shared logic
from fingerprint_utils import create_exam_fingerprint

# --- NEW: Checkpoint Configuration ---
CHECKPOINT_DIR = Path("data/fingerprint_checkpoints")


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
                result = (fingerprint.study_uid, sorted(list(fingerprint.file_hashes)))
                patient_inventory[exam_name] = result
            else:
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        except Exception as e:
            reason = f"Worker exception on {exam_path.name}: {e}"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    return patient_id, patient_inventory, failure_reasons


def consolidate_checkpoints(final_output_file: Path):
    """Reads all patient checkpoint files and combines them into one final inventory."""
    logging.info(f"Consolidating checkpoints from {CHECKPOINT_DIR}...")
    full_inventory = {}
    checkpoint_files = list(CHECKPOINT_DIR.glob("*.json"))

    for checkpoint_file in tqdm(checkpoint_files, desc="Consolidating"):
        patient_id = checkpoint_file.stem
        with open(checkpoint_file, "r") as f:
            patient_data = json.load(f)
        full_inventory[patient_id] = patient_data

    logging.info(f"Consolidated data for {len(full_inventory)} patients.")
    logging.info(f"Writing final inventory to {final_output_file}...")
    final_output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(final_output_file, "w") as f:
        json.dump(full_inventory, f)
    logging.info("Final inventory written successfully.")
    return full_inventory


def main(root_dir: Path, output_file: Path, parallel_jobs: int):
    logging.info(f"Starting remote fingerprinting of {root_dir}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    all_patient_dirs = [d for d in root_dir.iterdir() if d.is_dir()]

    completed_patient_ids = {p.stem for p in CHECKPOINT_DIR.glob("*.json")}
    patients_to_process = [
        p for p in all_patient_dirs if p.name not in completed_patient_ids
    ]

    logging.info(f"Found {len(all_patient_dirs)} total patients in source directory.")
    logging.info(f"Found {len(completed_patient_ids)} already completed checkpoints.")

    if not patients_to_process:
        logging.info("All patients already fingerprinted. Proceeding to consolidation.")
    else:
        logging.info(
            f"Resuming... {len(patients_to_process)} patients remaining to be processed."
        )

        # Process patients in batches to reduce main process overhead.
        # Chunk size is a multiple of parallel jobs to keep all workers busy.
        chunk_size = parallel_jobs * 20  # e.g., 8 jobs * 20 = 160 patients per batch
        chunk_size = 1
        overall_failure_reasons = {}

        with ProcessPoolExecutor(max_workers=parallel_jobs) as executor:
            # Wrap the main loop in a tqdm bar that tracks patients, not chunks
            with tqdm(
                total=len(patients_to_process), desc="Fingerprinting Patients"
            ) as pbar:
                for i in range(0, len(patients_to_process), chunk_size):
                    chunk = patients_to_process[i : i + chunk_size]

                    futures = {
                        executor.submit(fingerprint_all_exams_for_patient, path): path
                        for path in chunk
                    }

                    for future in as_completed(futures):
                        try:
                            patient_id, patient_inventory, failure_reasons = (
                                future.result()
                            )

                            if patient_inventory:
                                with open(
                                    CHECKPOINT_DIR / f"{patient_id}.json", "w"
                                ) as f:
                                    json.dump(patient_inventory, f)

                            for reason, count in failure_reasons.items():
                                overall_failure_reasons[reason] = (
                                    overall_failure_reasons.get(reason, 0) + count
                                )

                        except Exception as e:
                            path = futures[future]
                            logging.error(
                                f"A master process future failed for patient {path.name}: {e}"
                            )

                        # Update the single, overarching progress bar
                        pbar.update(1)

    # Consolidation and summary remain the same
    full_inventory = consolidate_checkpoints(output_file)
    total_exams_fingerprinted = sum(len(v) for v in full_inventory.values())

    logging.info("\n" + "=" * 50)
    logging.info("--- Fingerprinting Run Complete ---")
    logging.info(f"Total patients in final inventory: {len(full_inventory):,}")
    logging.info(f"Total exams in final inventory:  {total_exams_fingerprinted:,}")
    # Note: Failure summary is now only for the current run, not accumulated.
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
