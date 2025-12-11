#!/usr/bin/env python3
"""Verify that missing exams from patient directories are in modality directories.

This script checks that the exams missing from patient directories (compared to snapshot)
are the same exams that were moved to CR, MR, US, and Other directories.
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


def is_patient_dir(name: str) -> bool:
    """Check if directory name looks like a patient ID (starts with digit)."""
    return name and name[0].isdigit()


def get_exam_key(patient_id: str, exam_id: str) -> str:
    """Create a unique key for an exam."""
    return f"{patient_id}/{exam_id}"


def find_missing_from_patient_dirs(snapshot_dir: Path, current_dir: Path):
    """Find exams that exist in snapshot but are missing from patient directories.

    Returns a dict mapping exam_key -> snapshot_path.
    """
    missing_exams = {}

    logger.info("Scanning snapshot for patient directories...")

    # Find all patient directories in snapshot
    patient_dirs = []
    for item in snapshot_dir.iterdir():
        if item.is_dir() and is_patient_dir(item.name):
            patient_dirs.append(item)

    logger.info(f"Found {len(patient_dirs)} patient directories in snapshot")

    # Check each exam
    logger.info("Finding missing exams from patient directories...")
    for patient_dir in tqdm(patient_dirs, desc="Scanning patients", unit="patient"):
        patient_id = patient_dir.name
        current_patient_dir = current_dir / patient_id

        for exam_dir in patient_dir.iterdir():
            if not exam_dir.is_dir():
                continue

            exam_id = exam_dir.name
            current_exam_dir = current_patient_dir / exam_id

            # Check if exam is missing from patient directory
            if not current_exam_dir.exists():
                exam_key = get_exam_key(patient_id, exam_id)
                missing_exams[exam_key] = exam_dir

    return missing_exams


def find_exams_in_modality_dirs(current_dir: Path, modalities=None):
    """Find all exams in modality-specific directories.

    Returns a dict mapping exam_key -> modality_path.
    """
    if modalities is None:
        modalities = ["CR", "MR", "US", "Other"]

    modality_exams = {}

    logger.info("Scanning modality directories...")

    for modality in modalities:
        modality_dir = current_dir / modality

        if not modality_dir.exists():
            logger.warning(f"Modality directory does not exist: {modality_dir}")
            continue

        logger.info(f"Scanning {modality} directory...")

        # Find all patient directories in modality dir
        for patient_dir in tqdm(
            list(modality_dir.iterdir()), desc=f"Scanning {modality}", unit="patient"
        ):
            if not patient_dir.is_dir():
                continue

            patient_id = patient_dir.name

            # Find all exam directories
            for exam_dir in patient_dir.iterdir():
                if not exam_dir.is_dir():
                    continue

                exam_id = exam_dir.name
                exam_key = get_exam_key(patient_id, exam_id)

                if exam_key in modality_exams:
                    logger.warning(
                        f"Duplicate exam found: {exam_key} in both "
                        f"{modality_exams[exam_key].parent.parent.name} and {modality}"
                    )
                else:
                    modality_exams[exam_key] = exam_dir

    return modality_exams


def main():
    """Main verification function."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Verify that missing exams are in modality directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script verifies that the exams missing from patient directories
(compared to snapshot) are the same exams that were moved to the
modality-specific directories (CR, MR, US, Other).

Examples:
  python3 verify_moved_exams.py
  python3 verify_moved_exams.py --snapshot-date 2025.10.11-03.04.09
        """,
    )

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

    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["CR", "MR", "US", "Other"],
        help="Modality directories to check (default: CR MR US Other)",
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
    logger.info("VERIFY MOVED EXAMS")
    logger.info("=" * 80)
    logger.info(f"Current directory: {args.current_dir}")
    logger.info(f"Snapshot directory: {snapshot_dir}")
    logger.info(f"Modality directories to check: {', '.join(args.modalities)}")
    logger.info("=" * 80)

    # Find missing exams from patient directories
    missing_from_patient = find_missing_from_patient_dirs(
        snapshot_dir, args.current_dir
    )

    logger.info(
        f"\nFound {len(missing_from_patient)} exams missing from patient directories"
    )

    # Find exams in modality directories
    modality_exams = find_exams_in_modality_dirs(args.current_dir, args.modalities)

    logger.info(f"Found {len(modality_exams)} exams in modality directories")

    # Compare the two sets
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 80)

    missing_keys = set(missing_from_patient.keys())
    modality_keys = set(modality_exams.keys())

    # Exams that are missing from patient dirs AND in modality dirs (expected)
    in_both = missing_keys & modality_keys

    # Exams missing from patient dirs but NOT in modality dirs (unexpected!)
    missing_not_in_modality = missing_keys - modality_keys

    # Exams in modality dirs but NOT missing from patient dirs (duplicates!)
    in_modality_not_missing = modality_keys - missing_keys

    # Print results
    logger.info(f"\n✓ Exams correctly moved to modality directories: {len(in_both)}")

    if missing_not_in_modality:
        logger.error(
            f"\n✗ Exams missing from patient dirs but NOT in modality dirs: "
            f"{len(missing_not_in_modality)}"
        )
        logger.error("These exams may be LOST! First 20:")
        for exam_key in sorted(missing_not_in_modality)[:20]:
            logger.error(f"  {exam_key}")

        if len(missing_not_in_modality) > 20:
            logger.error(f"  ... and {len(missing_not_in_modality) - 20} more")
    else:
        logger.info(
            "✓ No exams are lost - all missing exams are in modality directories"
        )

    if in_modality_not_missing:
        logger.warning(
            f"\n⚠ Exams in modality dirs that are NOT missing from patient dirs: "
            f"{len(in_modality_not_missing)}"
        )
        logger.warning("These are DUPLICATES! First 20:")
        for exam_key in sorted(in_modality_not_missing)[:20]:
            modality_path = modality_exams[exam_key]
            modality_name = modality_path.parent.parent.name
            logger.warning(f"  {exam_key} (in {modality_name})")

        if len(in_modality_not_missing) > 20:
            logger.warning(f"  ... and {len(in_modality_not_missing) - 20} more")
    else:
        logger.info(
            "✓ No duplicates - all exams in modality dirs are missing from patient dirs"
        )

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(
        f"Total exams missing from patient directories: {len(missing_from_patient)}"
    )
    logger.info(f"Total exams in modality directories: {len(modality_exams)}")
    logger.info(f"Exams correctly moved (in both): {len(in_both)}")
    logger.info(f"Exams potentially LOST: {len(missing_not_in_modality)}")
    logger.info(f"Duplicate exams: {len(in_modality_not_missing)}")

    # Breakdown by modality
    logger.info("\nBreakdown by modality:")
    modality_counts = defaultdict(int)
    for exam_key in modality_keys:
        modality_path = modality_exams[exam_key]
        modality_name = modality_path.parent.parent.name
        modality_counts[modality_name] += 1

    for modality, count in sorted(modality_counts.items()):
        logger.info(f"  {modality}: {count} exams")

    # Final verdict
    logger.info("\n" + "=" * 80)
    logger.info("VERDICT")
    logger.info("=" * 80)

    if missing_not_in_modality:
        logger.error(
            f"✗ CRITICAL: {len(missing_not_in_modality)} exams are MISSING and not in modality directories!"
        )
        logger.error(
            "These exams need to be recovered from snapshot using recover_from_snapshot.py"
        )
        sys.exit(1)
    elif len(in_both) == len(missing_from_patient):
        logger.info(
            "✓ SUCCESS: All missing exams are accounted for in modality directories"
        )
        logger.info("✓ It is safe to delete the modality directories after recovery")

        if in_modality_not_missing:
            logger.warning(
                f"\n⚠ Note: {len(in_modality_not_missing)} duplicate exams exist in modality directories"
            )
            logger.warning(
                "These should be removed after verifying patient directories are complete"
            )
    else:
        logger.warning("⚠ Partial match - some exams may need investigation")
        sys.exit(1)


if __name__ == "__main__":
    main()
