#!/usr/bin/env python3
"""reorganize_by_modality.py

Dedicated script for reorganizing DICOM data by modality.
This script moves non-mammogram exams from the root directory structure
into modality-specific subdirectories (US/, CT/, MR/, etc.).

The script is designed to be resilient to interruptions and includes
a dry-run mode to preview changes before execution.

Usage:
    python reorganize_by_modality.py --raw /path/to/data --dry-run
    python reorganize_by_modality.py --raw /path/to/data --resume
"""

import argparse
import logging
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import pydicom
from pydicom.dataset import FileDataset
from pydicom.tag import Tag

from metadata_utils import extract_base_modality

# suppress pydicom VR UI validation warnings for non-standard UIDs
warnings.filterwarnings("ignore", message=".*Invalid value for VR UI.*", append=True)
# also set pydicom logging level to suppress warnings at the source
logging.getLogger("pydicom.valuerep").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


def get_tag(
    ds: FileDataset, tag: Tuple[int, int], default: Optional[str] = None
) -> Optional[str]:
    """Fetch a DICOM tag as string if present."""
    t = Tag(tag)
    if t in ds:
        v = ds[t].value
        return str(v)
    return default


def read_dicom(path: Path) -> FileDataset:
    """Read a DICOM from disk using pydicom."""
    return pydicom.dcmread(str(path), force=True)


def find_first_dicom(exam_path: Path) -> Optional[Path]:
    """Find the first DICOM file in an exam directory."""
    for root, _dirs, files in os.walk(exam_path):
        for name in files:
            if name.endswith(".dcm"):
                return Path(root) / name
    return None


def analyze_exam_modality(exam_path: Path) -> Tuple[str, str, str]:
    """Analyze an exam to determine its modality.

    Returns:
        (modality_raw, base_modality, study_description)
    """
    first_dcm = find_first_dicom(exam_path)
    if not first_dcm:
        return "", "Other", ""

    try:
        ds = read_dicom(first_dcm)
        modality_raw = get_tag(ds, (0x0008, 0x0060), default="")
        study_description = get_tag(ds, (0x0008, 0x1030), default="")
        base_modality = extract_base_modality(modality_raw, study_description)
        return modality_raw, base_modality, study_description
    except Exception as e:
        logger.warning(f"Failed to read DICOM {first_dcm}: {e}")
        return "", "Other", ""


def get_exam_directories(raw_dir: Path) -> List[Path]:
    """Get all exam directories from the raw directory structure.

    Assumes structure: raw_dir/patient_id/exam_id/
    """
    exam_dirs = []

    for patient_name in os.listdir(raw_dir):
        patient_path = raw_dir / patient_name
        if not patient_path.is_dir():
            continue

        # Skip modality directories (US/, CT/, MR/, etc.)
        if patient_name in ["US", "CT", "MR", "CR", "DX", "MG", "NM", "PT", "Other"]:
            continue

        for exam_name in os.listdir(patient_path):
            exam_path = patient_path / exam_name
            if exam_path.is_dir():
                exam_dirs.append(exam_path)

    return exam_dirs


def save_progress_state(
    exam_paths: List[Path], processed_count: int, output_file: Path
) -> None:
    """Save progress state to resume interrupted runs."""
    state = {
        "total_exams": len(exam_paths),
        "processed_count": processed_count,
        "remaining_exams": [str(p) for p in exam_paths[processed_count:]],
    }

    import json

    with open(output_file, "w") as f:
        json.dump(state, f, indent=2)


def load_progress_state(state_file: Path) -> Tuple[int, List[Path]]:
    """Load progress state from a previous run."""
    if not state_file.exists():
        return 0, []

    import json

    with open(state_file) as f:
        state = json.load(f)

    remaining_exams = [Path(p) for p in state.get("remaining_exams", [])]
    processed_count = state.get("processed_count", 0)

    return processed_count, remaining_exams


def find_empty_patient_directories(raw_dir: Path) -> List[Path]:
    """Find patient ID directories that are empty after reorganization.

    Returns directories that start with a number (patient IDs) and are empty
    or contain only dotfiles (temporary files).
    """
    empty_dirs = []

    for item in os.listdir(raw_dir):
        item_path = raw_dir / item

        # Skip non-directories
        if not item_path.is_dir():
            continue

        # Skip modality directories
        if item in ["US", "CT", "MR", "CR", "DX", "MG", "NM", "PT", "Other"]:
            continue

        # Only consider directories that start with a number (patient IDs)
        if not item[0].isdigit():
            continue

        # Check if directory is empty or contains only dotfiles
        try:
            contents = list(item_path.iterdir())
            if not contents:
                # Truly empty directory
                empty_dirs.append(item_path)
            elif all(item.name.startswith(".") for item in contents):
                # Directory contains only dotfiles - consider it empty
                empty_dirs.append(item_path)
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not check directory {item_path}: {e}")

    return empty_dirs


def cleanup_empty_patient_directories(raw_dir: Path, dry_run: bool = False) -> int:
    """Remove empty patient ID directories.

    Returns the number of directories removed (or would be removed in dry run).
    """
    empty_dirs = find_empty_patient_directories(raw_dir)

    if not empty_dirs:
        logger.info("No empty patient directories found")
        return 0

    logger.info(f"Found {len(empty_dirs)} empty patient directories")

    for empty_dir in empty_dirs:
        if dry_run:
            logger.info(f"Would remove empty patient directory: {empty_dir}")
        else:
            try:
                logger.info(f"Removing empty patient directory: {empty_dir}")
                # Use shutil.rmtree to handle directories with dotfiles
                shutil.rmtree(str(empty_dir))
            except (OSError, PermissionError) as e:
                logger.warning(f"Failed to remove directory {empty_dir}: {e}")

    return len(empty_dirs)


def move_exam_to_modality_directory(
    exam_path: Path, base_modality: str, dry_run: bool = False
) -> bool:
    """Move an exam to its appropriate modality directory using atomic move operation.

    Uses shutil.move() which performs a metadata-only operation on the same filesystem,
    making it much faster than copy+delete operations.

    Returns:
        True if exam was moved (or would be moved in dry run), False if move failed
    """
    # Move all exams to their appropriate modality directories (including MG for symmetry)

    patient_dir = exam_path.parent
    patient_id = patient_dir.name
    exam_id = exam_path.name
    chimec_root = patient_dir.parent

    # Create modality directory structure
    modality_root = chimec_root / base_modality
    new_patient_dir = modality_root / patient_id
    new_exam_path = new_patient_dir / exam_id

    if dry_run:
        return True

    # Check if destination already exists
    if new_exam_path.exists():
        logger.info(f"Destination {new_exam_path} already exists, skipping move")
        return True

    try:
        # Create parent directories
        new_patient_dir.mkdir(parents=True, exist_ok=True)

        # Use atomic move operation (metadata-only on same filesystem)
        logger.info(f"Moving {base_modality} exam from {exam_path} to {new_exam_path}")
        shutil.move(str(exam_path), str(new_exam_path))
        logger.info(f"Successfully moved {base_modality} exam to {new_exam_path}")

        return True

    except (OSError, FileExistsError) as e:
        logger.warning(f"Move failed for {exam_path}: {e}")
        # Clean up partial copy if it exists
        if new_exam_path.exists():
            logger.info(f"Cleaning up partial copy at {new_exam_path}")
            shutil.rmtree(str(new_exam_path), ignore_errors=True)
        return True  # Consider it handled to avoid reprocessing


def reorganize_by_modality(
    raw_dir: Path,
    dry_run: bool = False,
    resume: bool = False,
    max_exams: Optional[int] = None,
) -> None:
    """Reorganize DICOM data by modality.

    Moves non-mammogram exams from root structure into modality-specific directories.
    """
    logger.info("=== Starting data reorganization by modality ===")
    logger.info(f"Raw directory: {raw_dir}")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"Resume: {resume}")

    state_file = raw_dir / ".modality_reorganization_state.json"
    processed_count = 0
    exam_paths = []

    if resume and state_file.exists():
        logger.info("Resuming from previous run...")
        processed_count, remaining_exams = load_progress_state(state_file)
        exam_paths = remaining_exams
        logger.info(
            f"Resuming from exam {processed_count}, {len(exam_paths)} exams remaining"
        )
    else:
        logger.info("Scanning for exam directories...")
        exam_paths = get_exam_directories(raw_dir)
        logger.info(f"Found {len(exam_paths)} exam directories")

    if max_exams:
        exam_paths = exam_paths[:max_exams]
        logger.info(f"Limited to {max_exams} exams for processing")

    # Statistics tracking
    stats = {
        "total_exams": len(exam_paths),
        "processed": 0,
        "mammograms": 0,
        "moved_by_modality": {},
        "failed": 0,
        "other_categorized": 0,
    }

    # Process each exam
    for i, exam_path in enumerate(exam_paths):
        if not exam_path.exists():
            logger.warning(f"Exam path {exam_path} no longer exists, skipping")
            continue

        try:
            modality_raw, base_modality, study_description = analyze_exam_modality(
                exam_path
            )

            if dry_run:
                # Show details for all exams to verify common paths are working
                logger.info(f"Exam {i + 1}/{len(exam_paths)}: {exam_path.name}")
                logger.info(f"  Modality: {modality_raw}")
                logger.info(f"  Base Modality: {base_modality}")
                logger.info(f"  Study Description: {study_description}")
                logger.info("")
            else:
                logger.info(
                    f"Processing exam {i + 1}/{len(exam_paths)}: {exam_path.name} -> {base_modality}"
                )

            if base_modality == "Other":
                stats["other_categorized"] += 1

            # Move all exams to their modality directories (including MG for symmetry)
            success = move_exam_to_modality_directory(exam_path, base_modality, dry_run)
            if success:
                stats["moved_by_modality"][base_modality] = (
                    stats["moved_by_modality"].get(base_modality, 0) + 1
                )
                if base_modality == "MG":
                    stats["mammograms"] += 1
            else:
                stats["failed"] += 1

            stats["processed"] += 1

            # Save progress every 100 exams
            if not dry_run and (i + 1) % 100 == 0:
                save_progress_state(exam_paths, i + 1, state_file)
                logger.info(
                    f"Progress saved: {i + 1}/{len(exam_paths)} exams processed"
                )

        except Exception as e:
            logger.error(f"Error processing exam {exam_path}: {e}")
            stats["failed"] += 1

    # Clean up state file on successful completion
    if not dry_run and stats["failed"] == 0:
        if state_file.exists():
            state_file.unlink()
            logger.info("Progress state file cleaned up")

    # Clean up empty patient directories
    logger.info("\n=== Cleaning up empty patient directories ===")
    removed_count = cleanup_empty_patient_directories(raw_dir, dry_run)
    if dry_run:
        logger.info(f"Would remove {removed_count} empty patient directories")
    else:
        logger.info(f"Removed {removed_count} empty patient directories")

    # Print final statistics
    logger.info("\n=== Reorganization Summary ===")
    logger.info(f"Total exams processed: {stats['processed']}")
    logger.info(f"Failed exams: {stats['failed']}")

    if stats["moved_by_modality"]:
        logger.info("Exams moved by modality:")
        for modality, count in sorted(stats["moved_by_modality"].items()):
            logger.info(f"  {modality}: {count}")

    if dry_run:
        logger.info(f"\nExams categorized as 'Other': {stats['other_categorized']}")
        logger.info("\nThis was a dry run - no files were actually moved")
        logger.info("Run without --dry-run to perform the actual reorganization")
    else:
        logger.info("\nData reorganization completed successfully")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reorganize DICOM data by modality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be moved and cleaned up
  python reorganize_by_modality.py --raw /gpfs/data/huo-lab/Image/ChiMEC/ --dry-run

  # Actually perform the reorganization and cleanup empty dirs
  python reorganize_by_modality.py --raw /gpfs/data/huo-lab/Image/ChiMEC/

  # Resume interrupted run
  python reorganize_by_modality.py --raw /gpfs/data/huo-lab/Image/ChiMEC/ --resume

  # Process only first 50 exams (for testing)
  python reorganize_by_modality.py --raw /gpfs/data/huo-lab/Image/ChiMEC/ --max-exams 50
        """,
    )

    parser.add_argument(
        "--raw",
        dest="raw_dir",
        type=Path,
        required=True,
        help="Path to the raw data directory containing patient/exam structure",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without actually moving files",
    )

    parser.add_argument(
        "--resume", action="store_true", help="Resume from a previous interrupted run"
    )

    parser.add_argument(
        "--max-exams",
        type=int,
        default=None,
        help="Limit number of exams to process (for testing)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Validate arguments
    if not args.raw_dir.exists():
        logger.error(f"Raw directory does not exist: {args.raw_dir}")
        sys.exit(1)

    if not args.raw_dir.is_dir():
        logger.error(f"Raw path is not a directory: {args.raw_dir}")
        sys.exit(1)

    try:
        reorganize_by_modality(
            raw_dir=args.raw_dir,
            dry_run=args.dry_run,
            resume=args.resume,
            max_exams=args.max_exams,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
