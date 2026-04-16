#!/usr/bin/env python
"""
Deduplicate ChiMEC exams on disk by StudyInstanceUID.
Keeps the newest exam (by study_date, study_time); moves older copies to staging.

Uses light fingerprints (StudyInstanceUID, file_count, study_date) - no file hashes.
Accession numbers may differ between copies; same StudyInstanceUID = same exam.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# --- CONFIG ---
CHIMEC_MG_ROOT = Path("/gpfs/data/huo-lab/Image/ChiMEC/MG")
FINGERPRINT_CSV = Path("fingerprints/chimec/disk_fingerprints.csv")
DEDUP_STAGING_DIR = CHIMEC_MG_ROOT / "_dedup_staging"
# --- END CONFIG ---


def find_duplicate_groups(fp_df: pd.DataFrame) -> list[pd.DataFrame]:
    """Group by study_uid; return DataFrames for groups with >1 exam."""
    groups = []
    for study_uid, grp in fp_df.groupby("study_uid"):
        if len(grp) < 2:
            continue
        groups.append(grp)
    return groups


def choose_newest(grp: pd.DataFrame) -> tuple[int, list[int]]:
    """
    Pick the row to keep (newest by study_date, study_time).
    Returns (index_to_keep, indices_to_move).
    """
    sorted_grp = grp.sort_values(
        by=["study_date_iso", "study_time"],
        ascending=[False, False],
        na_position="first",
    )
    keep_idx = sorted_grp.index[0]
    move_indices = sorted_grp.index[1:].tolist()
    return int(keep_idx), move_indices


def run_deduplication(
    fingerprint_csv: Path,
    root_dir: Path,
    staging_dir: Path,
    dry_run: bool = True,
) -> tuple[int, int]:
    """
    Find duplicates by StudyInstanceUID, keep oldest, move rest to staging.
    Returns (exams_moved, groups_processed).
    """
    if not fingerprint_csv.exists():
        raise FileNotFoundError(
            f"Fingerprint CSV not found: {fingerprint_csv}. "
            "Run: python exports/export_chimec.py --build-fingerprints"
        )

    fp_df = pd.read_csv(fingerprint_csv)
    fp_df["study_date_iso"] = fp_df.get("study_date_iso", "").fillna("")
    fp_df["study_time"] = fp_df.get("study_time", "").fillna("")

    groups = find_duplicate_groups(fp_df)
    if not groups:
        print("No duplicate groups found (no study_uid with >1 exam).")
        return 0, 0

    total_to_move = sum(len(g) - 1 for g in groups)
    print(f"Found {len(groups):,} duplicate groups ({total_to_move:,} exams to move)")
    if dry_run:
        print("(dry-run; use --execute to move)")
        return 0, len(groups)

    staging_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for grp in tqdm(groups, desc="deduplicating"):
        keep_idx, move_indices = choose_newest(grp)
        for idx in move_indices:
            row = grp.loc[idx]
            patient_id = str(row["patient_id"])
            entry_name = str(row["entry_name"])
            exam_path = root_dir / patient_id / entry_name
            if not exam_path.exists():
                continue
            dest = staging_dir / patient_id / entry_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(exam_path), str(dest))
                moved += 1
            except Exception as e:
                print(f"  Failed to move {exam_path}: {e}")

    return moved, len(groups)


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate ChiMEC exams by StudyInstanceUID. "
        "Keeps newest exam (by study_date); moves older copies to staging. "
        "Uses light fingerprints (no file hashes)."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move duplicates (default: dry-run only)",
    )
    parser.add_argument(
        "--fingerprint-csv",
        type=Path,
        default=FINGERPRINT_CSV,
        help=f"Path to fingerprint CSV (default: {FINGERPRINT_CSV})",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=CHIMEC_MG_ROOT,
        help=f"ChiMEC MG root (default: {CHIMEC_MG_ROOT})",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=DEDUP_STAGING_DIR,
        help=f"Staging directory for duplicates (default: {DEDUP_STAGING_DIR})",
    )
    args = parser.parse_args()

    fp_csv = args.fingerprint_csv
    if not fp_csv.is_absolute():
        fp_csv = Path(__file__).resolve().parent / fp_csv

    moved, groups = run_deduplication(
        fingerprint_csv=fp_csv,
        root_dir=args.root,
        staging_dir=args.staging_dir,
        dry_run=not args.execute,
    )
    if moved > 0:
        print(f"Moved {moved:,} duplicate exams to {args.staging_dir}")
    if groups > 0 and args.execute and moved == 0:
        print("(No exams moved - paths may not exist or move failed)")


if __name__ == "__main__":
    main()
