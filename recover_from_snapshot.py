#!/usr/bin/env python3
"""Recover missing exams from snapshot.

This script safely recovers any exams that exist in the snapshot but are missing
from the current directory. It uses atomic copy operations to ensure no data loss
even if the script is interrupted.

The script:
1. Finds all patient directories (starting with digit) in snapshot
2. Checks each exam (accession number) subdirectory
3. If exam exists in snapshot but not in current dir, copies it
4. Uses atomic copy (copy + verify + no delete) to ensure safety
5. Handles interruptions gracefully - can be re-run safely
"""

import argparse
import logging
import os
import shutil
import sys
from collections import defaultdict
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


def verify_copy(source: Path, dest: Path) -> bool:
    """Verify that a copy was successful by comparing DICOM file counts."""
    source_count = count_dcm_files(source)
    dest_count = count_dcm_files(dest)

    if source_count != dest_count:
        logger.error(
            f"Copy verification failed: {source_count} vs {dest_count} DICOM files"
        )
        return False

    return True


def safe_copy_exam(source_exam: Path, dest_exam: Path, dry_run: bool = False) -> bool:
    """Safely copy an exam directory with verification.

    Uses atomic copy: copy to temp location, verify, then rename.
    This ensures we never have partial/corrupted data even if interrupted.

    Returns True if copy was successful (or would be in dry-run mode).
    """
    if dry_run:
        # Don't log individual exams in dry-run mode to avoid overwhelming output
        return True

    # Check if destination already exists
    if dest_exam.exists():
        logger.info(f"Destination already exists, skipping: {dest_exam}")
        return True

    # Create parent directory
    dest_exam.parent.mkdir(parents=True, exist_ok=True)

    # Use a temporary name for atomic copy
    temp_dest = dest_exam.parent / f".tmp_{dest_exam.name}"

    try:
        # Clean up any existing temp directory from previous interrupted run
        if temp_dest.exists():
            logger.info(f"Cleaning up previous temp directory: {temp_dest}")
            shutil.rmtree(str(temp_dest))

        # Copy to temporary location
        logger.info(f"Copying {source_exam.name} to temporary location...")
        shutil.copytree(str(source_exam), str(temp_dest))

        # Verify the copy
        logger.info(f"Verifying copy of {source_exam.name}...")
        if not verify_copy(source_exam, temp_dest):
            logger.error(f"Copy verification failed for {source_exam.name}")
            shutil.rmtree(str(temp_dest), ignore_errors=True)
            return False

        # Rename to final destination (atomic operation)
        logger.info(f"Moving to final destination: {dest_exam}")
        temp_dest.rename(dest_exam)

        logger.info(f"Successfully recovered: {dest_exam}")
        return True

    except Exception as e:
        logger.error(f"Failed to copy {source_exam}: {e}")
        # Clean up temp directory on failure
        if temp_dest.exists():
            shutil.rmtree(str(temp_dest), ignore_errors=True)
        return False


def get_exam_key(patient_id: str, exam_id: str) -> str:
    """Create a unique key for an exam."""
    return f"{patient_id}/{exam_id}"


def find_missing_exams(snapshot_dir: Path, current_dir: Path):
    """Find exams that exist in snapshot but are missing from current directory.

    Returns a dict mapping patient_id -> list of missing exam paths in snapshot.
    """
    missing_exams = defaultdict(list)

    logger.info("Scanning snapshot for patient directories...")

    # Find all patient directories in snapshot
    patient_dirs = []
    for item in snapshot_dir.iterdir():
        if item.is_dir() and is_patient_dir(item.name):
            patient_dirs.append(item)

    logger.info(f"Found {len(patient_dirs)} patient directories in snapshot")

    # For each patient, check which exams are missing
    logger.info("Checking for missing exams...")
    for patient_dir in tqdm(patient_dirs, desc="Scanning patients", unit="patient"):
        patient_id = patient_dir.name
        current_patient_dir = current_dir / patient_id

        # Get all exam directories in snapshot
        for exam_dir in patient_dir.iterdir():
            if not exam_dir.is_dir():
                continue

            exam_id = exam_dir.name
            current_exam_dir = current_patient_dir / exam_id

            # Check if this exam is missing from current directory
            if not current_exam_dir.exists():
                missing_exams[patient_id].append(exam_dir)

    return missing_exams


def find_exams_in_modality_dirs(current_dir: Path, modalities=None):
    """Find all exams in modality-specific directories.

    Returns a dict mapping exam_key -> modality_name.
    """
    if modalities is None:
        modalities = ["CR", "MR", "US", "Other"]

    modality_exams = {}

    logger.info("\nVerifying exams in modality directories...")

    for modality in modalities:
        modality_dir = current_dir / modality

        if not modality_dir.exists():
            continue

        # Find all patient directories in modality dir
        for patient_dir in modality_dir.iterdir():
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
                        f"{modality_exams[exam_key]} and {modality}"
                    )
                else:
                    modality_exams[exam_key] = modality

    return modality_exams


def verify_against_modality_dirs(missing_exams_dict, current_dir: Path):
    """Verify that missing exams match what's in modality directories.

    Returns True if verification passes, False otherwise.
    """
    # Convert missing exams to set of exam keys
    missing_keys = set()
    for patient_id, exam_list in missing_exams_dict.items():
        for exam_path in exam_list:
            exam_id = exam_path.name
            exam_key = get_exam_key(patient_id, exam_id)
            missing_keys.add(exam_key)

    # Find exams in modality directories
    modality_exams = find_exams_in_modality_dirs(current_dir)
    modality_keys = set(modality_exams.keys())

    # Compare the sets
    in_both = missing_keys & modality_keys
    missing_not_in_modality = missing_keys - modality_keys
    in_modality_not_missing = modality_keys - missing_keys

    # Report results
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION AGAINST MODALITY DIRECTORIES")
    logger.info("=" * 80)
    logger.info(f"Exams to recover: {len(missing_keys)}")
    logger.info(f"Exams in modality directories: {len(modality_keys)}")
    logger.info(f"Exams in both (correctly moved): {len(in_both)}")

    # Breakdown by modality
    if modality_exams:
        logger.info("\nBreakdown by modality:")
        modality_counts = defaultdict(int)
        for exam_key in modality_keys:
            modality_counts[modality_exams[exam_key]] += 1

        for modality, count in sorted(modality_counts.items()):
            logger.info(f"  {modality}: {count} exams")

    verification_passed = True

    if missing_not_in_modality:
        logger.error(
            f"\n✗ WARNING: {len(missing_not_in_modality)} exams are in snapshot but NOT in patient dirs OR modality dirs!"
        )
        logger.error("These exams exist in the snapshot but are missing from BOTH:")
        logger.error("  1. Patient directories (e.g., /ChiMEC/12345678/2O12345)")
        logger.error("  2. Modality directories (e.g., /ChiMEC/CR/, /ChiMEC/MR/, etc.)")
        logger.error("\nThese exams appear to be LOST and need recovery. First 10:")
        for exam_key in sorted(missing_not_in_modality)[:10]:
            logger.error(f"  {exam_key}")

        if len(missing_not_in_modality) > 10:
            logger.error(f"  ... and {len(missing_not_in_modality) - 10} more")

        verification_passed = False

    if in_modality_not_missing:
        logger.warning(
            f"\n⚠ WARNING: {len(in_modality_not_missing)} exams in modality directories are NOT missing from patient dirs!"
        )
        logger.warning("These are duplicates that exist in both locations. First 10:")
        for exam_key in sorted(in_modality_not_missing)[:10]:
            modality = modality_exams[exam_key]
            logger.warning(f"  {exam_key} (in {modality})")

        if len(in_modality_not_missing) > 10:
            logger.warning(f"  ... and {len(in_modality_not_missing) - 10} more")

        verification_passed = False

    if verification_passed and len(in_both) == len(missing_keys):
        logger.info(
            "\n✓ VERIFICATION PASSED: All exams to recover are accounted for in modality directories"
        )
        logger.info(
            "✓ After recovery, it will be safe to delete the modality directories"
        )

    return verification_passed


def main():
    """Main recovery function."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Recover missing exams from snapshot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be recovered
  python3 recover_from_snapshot.py --dry-run
  
  # Actually perform the recovery
  python3 recover_from_snapshot.py
  
  # Use custom snapshot date
  python3 recover_from_snapshot.py --snapshot-date 2025.10.11-03.04.09
        """,
    )

    parser.add_argument(
        "--current-dir",
        type=Path,
        default=Path("/gpfs/data/huo-lab/Image/ChiMEC/"),
        help="Current data directory (default: /gpfs/data/huo-lab/Image/ChiMEC/)",
    )

    parser.add_argument(
        "--snapshot-date",
        type=str,
        default="2025.10.12-03.04.09",
        help="Snapshot date in format YYYY.MM.DD-HH.MM.SS (default: 2025.10.12-03.04.09)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be recovered without actually copying",
    )

    parser.add_argument(
        "--max-exams",
        type=int,
        help="Maximum number of exams to recover (for testing)",
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
        logger.error("Available snapshots:")
        snapshots_root = Path("/gpfs/data/huo-lab/.snapshots/")
        if snapshots_root.exists():
            for snap in sorted(snapshots_root.iterdir()):
                if snap.is_dir():
                    logger.error(f"  {snap.name}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("EXAM RECOVERY FROM SNAPSHOT")
    logger.info("=" * 80)
    logger.info(f"Current directory: {args.current_dir}")
    logger.info(f"Snapshot directory: {snapshot_dir}")

    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be copied")

    logger.info("=" * 80)

    # Find missing exams
    missing_exams = find_missing_exams(snapshot_dir, args.current_dir)

    # Calculate statistics
    total_missing = sum(len(exams) for exams in missing_exams.values())
    total_patients = len(missing_exams)

    logger.info("\n" + "=" * 80)
    logger.info("MISSING EXAMS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total patients with missing exams: {total_patients}")
    logger.info(f"Total missing exams: {total_missing}")

    if total_missing == 0:
        logger.info("\nNo missing exams found! All data is present.")
        return

    # Show breakdown by patient
    logger.info("\nMissing exams by patient:")
    for patient_id in sorted(missing_exams.keys())[:20]:  # Show first 20
        count = len(missing_exams[patient_id])
        logger.info(f"  Patient {patient_id}: {count} missing exams")

    if len(missing_exams) > 20:
        logger.info(f"  ... and {len(missing_exams) - 20} more patients")

    # Verify against modality directories
    verify_against_modality_dirs(missing_exams, args.current_dir)

    # Apply max-exams limit if specified
    if args.max_exams:
        logger.info(f"\nLimiting recovery to {args.max_exams} exams (--max-exams)")

    # Ask for confirmation if not dry-run
    if not args.dry_run:
        logger.info("\n" + "=" * 80)
        response = input(f"Proceed with recovering {total_missing} exams? [y/N]: ")
        if response.lower() != "y":
            logger.info("Recovery cancelled by user")
            return

    # Recover missing exams
    logger.info("\n" + "=" * 80)
    logger.info("RECOVERING MISSING EXAMS")
    logger.info("=" * 80)

    success_count = 0
    failure_count = 0
    exam_count = 0

    # Flatten the list of exams to recover
    exams_to_recover = []
    for patient_id, exam_list in missing_exams.items():
        for exam_path in exam_list:
            exams_to_recover.append((patient_id, exam_path))
            if args.max_exams and len(exams_to_recover) >= args.max_exams:
                break
        if args.max_exams and len(exams_to_recover) >= args.max_exams:
            break

    # Process each exam with progress bar
    for patient_id, snapshot_exam_path in tqdm(
        exams_to_recover, desc="Recovering exams", unit="exam"
    ):
        exam_id = snapshot_exam_path.name
        dest_exam_path = args.current_dir / patient_id / exam_id

        if safe_copy_exam(snapshot_exam_path, dest_exam_path, args.dry_run):
            success_count += 1
        else:
            failure_count += 1

        exam_count += 1

    # Final summary
    logger.info("\n" + "=" * 80)
    if args.dry_run:
        logger.info("DRY RUN COMPLETE")
    else:
        logger.info("RECOVERY COMPLETE")
    logger.info("=" * 80)

    if args.dry_run:
        logger.info(f"DRY RUN: Would have recovered {success_count} exams")
    else:
        logger.info(f"Successfully recovered: {success_count} exams")
        logger.info(f"Failed to recover: {failure_count} exams")

    if failure_count > 0:
        logger.warning("\nSome exams failed to recover. Check the log for details.")
        logger.warning("You can safely re-run this script to retry failed exams.")

    logger.info("\nNext steps:")
    logger.info("1. Review the recovered exams")
    logger.info("2. After confirming recovery is complete, you can manually delete:")
    logger.info("   - /gpfs/data/huo-lab/Image/ChiMEC/CR/")
    logger.info("   - /gpfs/data/huo-lab/Image/ChiMEC/MR/")
    logger.info("   - /gpfs/data/huo-lab/Image/ChiMEC/US/")
    logger.info("   - /gpfs/data/huo-lab/Image/ChiMEC/Other/")
    logger.info(
        "   (These directories contain duplicated data from the move operations)"
    )


if __name__ == "__main__":
    main()
