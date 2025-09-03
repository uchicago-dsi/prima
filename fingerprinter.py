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


def main(root_dir: Path, output_file: Path, parallel_jobs: int):
    logging.info(f"Starting remote fingerprinting of {root_dir}")
    # The inventory will map: {patient_id: {exam_name: [study_uid, [hashes]]}}
    inventory = {}

    # Discover all exam directories to be processed
    all_exam_paths = [
        exam_path
        for patient_dir in tqdm(
            list(root_dir.iterdir()), desc="Discovering remote exams"
        )
        if patient_dir.is_dir()
        for exam_path in patient_dir.iterdir()
        if exam_path.is_dir()
    ]
    if not all_exam_paths:
        logging.warning("No exam directories found to process.")

    # Process all exams in parallel
    with ProcessPoolExecutor(max_workers=parallel_jobs) as executor:
        futures = {
            executor.submit(create_exam_fingerprint, path): path
            for path in all_exam_paths
        }
        for future in tqdm(
            as_completed(futures), total=len(futures),
            desc="Fingerprinting Remotely"
        ):
            path = futures[future]
            try:
                fingerprint = future.result()
                if fingerprint and fingerprint.is_valid():
                    patient_id = path.parent.name
                    exam_name = path.name
                    # Store as a list for JSON compatibility
                    result = (
                        fingerprint.study_uid,
                        sorted(list(fingerprint.file_hashes))
                    )

                    if patient_id not in inventory:
                        inventory[patient_id] = {}
                    inventory[patient_id][exam_name] = result
            except Exception as e:
                logging.error(f"Failed to fingerprint {path}: {e}")

    logging.info(
        f"Fingerprinting complete. Saving inventory to {output_file}"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(inventory, f)
    logging.info("Inventory saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scans and fingerprints a DICOM directory structure."
    )
    parser.add_argument(
        "root_dir", type=Path, help="The root directory to scan."
    )
    parser.add_argument(
        "output_file", type=Path,
        help="The JSON file to write the inventory to."
    )
    parser.add_argument(
        "-p", "--parallel-jobs", type=int, default=8,
        help="Number of parallel processes."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(message)s",
        stream=sys.stderr
    )
    main(args.root_dir, args.output_file, args.parallel_jobs)
