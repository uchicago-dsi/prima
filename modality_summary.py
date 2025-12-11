#!/usr/bin/env python3
"""Quick summary of modality distribution across a directory structure.

This script provides a fast overview of modality categorization
across multiple exams without detailed file-by-file analysis.
"""

import argparse
import logging
import os
import sys
import warnings
from collections import Counter
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import pydicom

    from metadata_utils import extract_base_modality
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure pydicom is installed: pip install pydicom")
    sys.exit(1)

# suppress pydicom VR UI validation warnings for non-standard UIDs
warnings.filterwarnings("ignore", message=".*Invalid value for VR UI.*", append=True)
# also set pydicom logging level to suppress warnings at the source
logging.getLogger("pydicom.valuerep").setLevel(logging.ERROR)


def get_tag(ds, tag, default=None):
    """Fetch a DICOM tag as string if present."""
    try:
        if tag in ds:
            return str(ds[tag].value)
    except Exception:
        pass
    return default


def quick_modality_check(dcm_path: Path):
    """Quickly check modality of a single DICOM file."""
    try:
        ds = pydicom.dcmread(str(dcm_path), force=True)
        modality_raw = get_tag(ds, (0x0008, 0x0060), "")
        study_description = get_tag(ds, (0x0008, 0x1030), "")
        base_modality = extract_base_modality(modality_raw, study_description)

        return {
            "modality_raw": modality_raw,
            "study_description": study_description,
            "base_modality": base_modality,
        }
    except Exception:
        return None


def scan_directory(root_dir: Path, max_files: int = 1000):
    """Scan directory for DICOM files and collect modality info."""
    raw_modalities = Counter()
    base_modalities = Counter()
    other_examples = []
    total_files = 0
    total_exams = 0
    processed_exams = set()

    logger = logging.getLogger(__name__)

    for root, dirs, files in os.walk(root_dir):
        # Skip if we've hit the file limit
        if total_files >= max_files:
            logger.info(f"Reached file limit of {max_files}, stopping scan")
            break

        # Check if this looks like an exam directory (contains .dcm files)
        dcm_files = [f for f in files if f.endswith(".dcm")]
        if not dcm_files:
            continue

        # Get exam path (assuming structure: root/patient/exam/files)
        exam_path = Path(root)
        exam_key = str(exam_path)  # Use full path as exam identifier

        if exam_key not in processed_exams:
            total_exams += 1
            processed_exams.add(exam_key)

        # Sample a few files from each exam
        sample_files = dcm_files[:3]  # Take up to 3 files per exam

        for dcm_file in sample_files:
            if total_files >= max_files:
                break

            dcm_path = Path(root) / dcm_file
            result = quick_modality_check(dcm_path)

            if result:
                total_files += 1
                raw_modalities[result["modality_raw"]] += 1
                base_modalities[result["base_modality"]] += 1

                # Collect examples of "Other" modality
                if result["base_modality"] == "Other" and len(other_examples) < 20:
                    other_examples.append(
                        {
                            "file": dcm_path.name,
                            "exam": exam_path.name,
                            "modality_raw": result["modality_raw"],
                            "study_description": result["study_description"],
                        }
                    )

    return {
        "raw_modalities": raw_modalities,
        "base_modalities": base_modalities,
        "other_examples": other_examples,
        "total_files": total_files,
        "total_exams": total_exams,
    }


def main():
    """Main function."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Quick modality summary")
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Directory to scan (can be patient dir, exam dir, or root dir)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=1000,
        help="Maximum number of DICOM files to analyze (default: 1000)",
    )

    args = parser.parse_args()

    if not args.dir.exists():
        print(f"Error: Directory {args.dir} does not exist")
        sys.exit(1)

    logger = logging.getLogger(__name__)
    logger.info(f"Scanning directory: {args.dir}")
    logger.info(f"Max files to analyze: {args.max_files}")

    results = scan_directory(args.dir, args.max_files)

    print("\n" + "=" * 60)
    print("MODALITY SUMMARY")
    print("=" * 60)
    print(f"Total files analyzed: {results['total_files']}")
    print(f"Total exams sampled: {results['total_exams']}")

    print("\nBase modality distribution:")
    for modality, count in results["base_modalities"].most_common():
        percentage = (count / results["total_files"]) * 100
        print(f"  {modality}: {count} files ({percentage:.1f}%)")

    print("\nRaw modality distribution:")
    for modality, count in results["raw_modalities"].most_common():
        percentage = (count / results["total_files"]) * 100
        print(f"  '{modality}': {count} files ({percentage:.1f}%)")

    # Show examples of "Other" modality
    if results["other_examples"]:
        print("\n" + "=" * 60)
        print("EXAMPLES OF 'OTHER' MODALITY")
        print("=" * 60)
        for example in results["other_examples"]:
            print(f"\nFile: {example['file']} (Exam: {example['exam']})")
            print(f"  Raw Modality: '{example['modality_raw']}'")
            if example["study_description"]:
                desc = example["study_description"][:50]
                if len(example["study_description"]) > 50:
                    desc += "..."
                print(f"  Study Description: '{desc}'")

    other_count = results["base_modalities"].get("Other", 0)
    if other_count > 0:
        other_percentage = (other_count / results["total_files"]) * 100
        print(
            f"\nNote: {other_count} files ({other_percentage:.1f}%) categorized as 'Other'"
        )
        print("Use investigate_modality.py for detailed analysis of specific exams")


if __name__ == "__main__":
    main()
