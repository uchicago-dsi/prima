#!/usr/bin/env python3
"""Verify that recovery was successful.

This script compares the current directory with the snapshot to ensure
all exams have been recovered successfully.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


def is_patient_dir(name: str) -> bool:
    """Check if directory name looks like a patient ID (starts with digit)."""
    return name and name[0].isdigit()


def count_dcm_files(directory: Path) -> int:
    """Count DICOM files in a directory."""
    count = 0
    for root, dirs, files in os.walk(directory):
        count += sum(1 for f in files if f.endswith(".dcm"))
    return count


def compare_directories(snapshot_dir: Path, current_dir: Path):
    """Compare snapshot and current directories to find missing exams."""

    missing_exams = []
    mismatched_exams = []

    logger.info("Scanning snapshot for patient directories...")

    # Find all patient directories in snapshot
    patient_dirs = []
    for item in snapshot_dir.iterdir():
        if item.is_dir() and is_patient_dir(item.name):
            patient_dirs.append(item)

    logger.info(f"Found {len(patient_dirs)} patient directories in snapshot")

    # Check each exam
    logger.info("Verifying exams...")
    for patient_dir in tqdm(patient_dirs, desc="Verifying patients", unit="patient"):
        patient_id = patient_dir.name
        current_patient_dir = current_dir / patient_id

        for exam_dir in patient_dir.iterdir():
            if not exam_dir.is_dir():
                continue

            exam_id = exam_dir.name
            current_exam_dir = current_patient_dir / exam_id

            # Check if exam exists
            if not current_exam_dir.exists():
                missing_exams.append((patient_id, exam_id))
                continue

            # Check if DICOM counts match
            snapshot_count = count_dcm_files(exam_dir)
            current_count = count_dcm_files(current_exam_dir)

            if snapshot_count != current_count:
                mismatched_exams.append(
                    {
                        "patient_id": patient_id,
                        "exam_id": exam_id,
                        "snapshot_count": snapshot_count,
                        "current_count": current_count,
                    }
                )

    return missing_exams, mismatched_exams


def main():
    """Main verification function."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Verify recovery from snapshot")

    parser.add_argument(
        "--current-dir",
        type=Path,
        default=Path("/gpfs/data/huo-lab/Image/ChiMEC/"),
        help="Current data directory",
    )

    parser.add_argument(
        "--snapshot-date",
        type=str,
        default="2025.10.12-03.04.09",
        help="Snapshot date in format YYYY.MM.DD-HH.MM.SS",
    )

    args = parser.parse_args()

    # Construct snapshot path
    snapshot_dir = Path(
        f"/gpfs/data/huo-lab/.snapshots/@GMT-{args.snapshot_date}/Image/ChiMEC/"
    )

    # Validate directories
    if not args.current_dir.exists():
        logger.error(f"Current directory does not exist: {args.current_dir}")
        sys.exit(1)

    if not snapshot_dir.exists():
        logger.error(f"Snapshot directory does not exist: {snapshot_dir}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("RECOVERY VERIFICATION")
    logger.info("=" * 80)
    logger.info(f"Current directory: {args.current_dir}")
    logger.info(f"Snapshot directory: {snapshot_dir}")
    logger.info("=" * 80)

    # Compare directories
    missing_exams, mismatched_exams = compare_directories(
        snapshot_dir, args.current_dir
    )

    # Report results
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION RESULTS")
    logger.info("=" * 80)

    if not missing_exams and not mismatched_exams:
        logger.info("✓ SUCCESS: All exams recovered successfully!")
        logger.info(
            "✓ All DICOM file counts match between snapshot and current directory"
        )
        return

    if missing_exams:
        logger.error(f"\n✗ MISSING EXAMS: {len(missing_exams)} exams are still missing")
        logger.error("\nFirst 10 missing exams:")
        for patient_id, exam_id in missing_exams[:10]:
            logger.error(f"  Patient {patient_id}, Exam {exam_id}")

        if len(missing_exams) > 10:
            logger.error(f"  ... and {len(missing_exams) - 10} more")

    if mismatched_exams:
        logger.error(
            f"\n✗ MISMATCHED EXAMS: {len(mismatched_exams)} exams have different DICOM counts"
        )
        logger.error("\nFirst 10 mismatched exams:")
        for exam in mismatched_exams[:10]:
            logger.error(
                f"  Patient {exam['patient_id']}, Exam {exam['exam_id']}: "
                f"snapshot={exam['snapshot_count']}, current={exam['current_count']}"
            )

        if len(mismatched_exams) > 10:
            logger.error(f"  ... and {len(mismatched_exams) - 10} more")

    logger.info(
        "\nRecommendation: Re-run ops/recover_from_snapshot.py to fix missing/mismatched exams"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
