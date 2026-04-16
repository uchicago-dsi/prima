#!/usr/bin/env python
"""
Reconcile patient IDs between disk and iBroker metadata.

Uses StudyDateTime "signatures" to match patients - if a patient on disk
has exams with dates that match an iBroker patient, they're the same person.

ALWAYS run with --dry-run first to see planned changes.
"""

import argparse
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("reconcile.log"), logging.StreamHandler()],
)

# --- CONFIGURATION ---
FINGERPRINT_CACHE = Path("data/destination_fingerprints.json")
IBROKER_METADATA = Path("data/imaging_metadata.csv")
QUARANTINE_DIR = Path("/gpfs/data/huo-lab/Image/ChiMEC/_quarantine_orphan_patients")
# --- END CONFIGURATION ---


def normalize_dicom_date(date_str: Optional[str]) -> Optional[str]:
    """normalize DICOM date (YYYYMMDD) to ISO format (YYYY-MM-DD)"""
    if not date_str or len(date_str) < 8:
        return None
    try:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    except Exception:
        return None


def normalize_datetime_to_date(dt) -> Optional[str]:
    """normalize pandas datetime to ISO date string"""
    if pd.isna(dt):
        return None
    try:
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def load_ibroker_signatures(metadata_path: Path) -> Dict[str, Set[str]]:
    """
    load iBroker metadata and build patient date signatures.

    Parameters
    ----------
    metadata_path : Path
        path to iBroker metadata CSV

    Returns
    -------
    Dict[str, Set[str]]
        mapping of ibroker_patient_id → {set of study dates as YYYY-MM-DD}
    """
    logging.info(f"Loading iBroker metadata from {metadata_path}...")

    if not metadata_path.exists():
        raise FileNotFoundError(f"iBroker metadata not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    df["Study DateTime"] = pd.to_datetime(df["Study DateTime"], errors="coerce")

    signatures = defaultdict(set)
    for _, row in df.iterrows():
        patient_id = str(int(row["study_id"])) if pd.notna(row["study_id"]) else None
        date_str = normalize_datetime_to_date(row["Study DateTime"])
        if patient_id and date_str:
            signatures[patient_id].add(date_str)

    logging.info(f"Loaded {len(signatures):,} patients from iBroker metadata")
    total_dates = sum(len(dates) for dates in signatures.values())
    logging.info(f"Total exam dates in iBroker: {total_dates:,}")

    return dict(signatures)


def load_disk_signatures(fingerprint_cache: Path) -> Dict[str, Dict[str, Set[str]]]:
    """
    load disk fingerprint cache and build patient date signatures.

    Returns
    -------
    Dict[str, Dict[str, Set[str]]]
        mapping of disk_patient_id → {
            "dates": {set of study dates as YYYY-MM-DD},
            "exams": {exam_name: date_str}
        }
    """
    logging.info(f"Loading disk fingerprints from {fingerprint_cache}...")

    if not fingerprint_cache.exists():
        raise FileNotFoundError(f"Fingerprint cache not found: {fingerprint_cache}")

    with open(fingerprint_cache) as f:
        raw_data = json.load(f)

    signatures = {}
    exams_without_dates = 0

    for patient_id, exams in tqdm(raw_data.items(), desc="Building disk signatures"):
        dates = set()
        exam_dates = {}

        for exam_name, data in exams.items():
            uid, hashes, study_date, study_time = data
            date_str = normalize_dicom_date(study_date)
            if date_str:
                dates.add(date_str)
                exam_dates[exam_name] = date_str
            else:
                exams_without_dates += 1

        if dates:
            signatures[patient_id] = {"dates": dates, "exam_dates": exam_dates}

    logging.info(f"Loaded {len(signatures):,} patients from disk with date info")
    total_dates = sum(len(s["dates"]) for s in signatures.values())
    logging.info(f"Total exam dates on disk: {total_dates:,}")
    if exams_without_dates > 0:
        logging.warning(
            f"Exams without StudyDate (old fingerprint format): {exams_without_dates:,}"
        )

    return signatures


def find_patient_matches(
    disk_signatures: Dict[str, Dict[str, Set[str]]],
    ibroker_signatures: Dict[str, Set[str]],
    min_match_ratio: float = 0.5,
    min_matches: int = 1,
) -> List[Dict]:
    """
    find matching patients between disk and iBroker based on exam date overlap.

    Parameters
    ----------
    disk_signatures : Dict
        disk patient signatures
    ibroker_signatures : Dict
        iBroker patient signatures
    min_match_ratio : float
        minimum ratio of disk dates that must match iBroker (0.5 = 50%)
    min_matches : int
        minimum number of matching dates required

    Returns
    -------
    List[Dict]
        list of match records with confidence scores
    """
    logging.info("Finding patient matches by date signature...")

    matches = []

    for disk_id, disk_data in tqdm(disk_signatures.items(), desc="Matching patients"):
        disk_dates = disk_data["dates"]

        # skip if disk patient is already in iBroker with same ID
        if disk_id in ibroker_signatures:
            ibroker_dates = ibroker_signatures[disk_id]
            overlap = disk_dates & ibroker_dates
            if len(overlap) >= min_matches:
                matches.append(
                    {
                        "disk_patient_id": disk_id,
                        "ibroker_patient_id": disk_id,
                        "action": "KEEP",
                        "disk_dates": len(disk_dates),
                        "ibroker_dates": len(ibroker_dates),
                        "matching_dates": len(overlap),
                        "match_ratio": len(overlap) / len(disk_dates)
                        if disk_dates
                        else 0,
                        "confidence": "ID_MATCH",
                    }
                )
                continue

        # search for best matching iBroker patient
        best_match = None
        best_overlap = 0
        best_ratio = 0

        for ibroker_id, ibroker_dates in ibroker_signatures.items():
            overlap = disk_dates & ibroker_dates
            if len(overlap) > best_overlap:
                best_overlap = len(overlap)
                best_ratio = len(overlap) / len(disk_dates) if disk_dates else 0
                best_match = ibroker_id

        if best_match and best_overlap >= min_matches and best_ratio >= min_match_ratio:
            # determine confidence level
            if best_ratio >= 0.9 and best_overlap >= 3:
                confidence = "HIGH"
            elif best_ratio >= 0.7 and best_overlap >= 2:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            matches.append(
                {
                    "disk_patient_id": disk_id,
                    "ibroker_patient_id": best_match,
                    "action": "RENAME",
                    "disk_dates": len(disk_dates),
                    "ibroker_dates": len(ibroker_signatures[best_match]),
                    "matching_dates": best_overlap,
                    "match_ratio": best_ratio,
                    "confidence": confidence,
                }
            )
        else:
            # no match found - orphan patient
            matches.append(
                {
                    "disk_patient_id": disk_id,
                    "ibroker_patient_id": None,
                    "action": "ORPHAN",
                    "disk_dates": len(disk_dates),
                    "ibroker_dates": 0,
                    "matching_dates": best_overlap if best_match else 0,
                    "match_ratio": best_ratio if best_match else 0,
                    "confidence": "NONE",
                }
            )

    return matches


def generate_reconciliation_plan(
    matches: List[Dict],
    base_dir: Path,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    generate a reconciliation plan from matches.

    Returns
    -------
    Tuple of (renames, keeps, orphans)
    """
    renames = []
    keeps = []
    orphans = []

    for match in matches:
        if match["action"] == "KEEP":
            keeps.append(match)
        elif match["action"] == "RENAME":
            src = base_dir / match["disk_patient_id"]
            dst = base_dir / match["ibroker_patient_id"]
            match["src_path"] = str(src)
            match["dst_path"] = str(dst)
            match["src_exists"] = src.exists()
            match["dst_exists"] = dst.exists()
            renames.append(match)
        elif match["action"] == "ORPHAN":
            src = base_dir / match["disk_patient_id"]
            match["src_path"] = str(src)
            match["src_exists"] = src.exists()
            orphans.append(match)

    return renames, keeps, orphans


def print_reconciliation_summary(
    renames: List[Dict],
    keeps: List[Dict],
    orphans: List[Dict],
):
    """print a summary of the reconciliation plan"""
    print("\n" + "=" * 80)
    print("RECONCILIATION PLAN SUMMARY")
    print("=" * 80)

    print(f"\nPatients already correct (KEEP): {len(keeps):,}")
    print(f"Patients to rename (RENAME):     {len(renames):,}")
    print(f"Orphan patients (no match):      {len(orphans):,}")

    # breakdown by confidence
    if renames:
        print("\n--- RENAMES BY CONFIDENCE ---")
        confidence_counts = defaultdict(int)
        for r in renames:
            confidence_counts[r["confidence"]] += 1
        for conf in ["HIGH", "MEDIUM", "LOW"]:
            if conf in confidence_counts:
                print(f"  {conf}: {confidence_counts[conf]:,}")

    # show sample renames
    if renames:
        print("\n--- SAMPLE RENAMES (first 10) ---")
        for r in sorted(renames, key=lambda x: -x["match_ratio"])[:10]:
            print(f"  {r['disk_patient_id']} → {r['ibroker_patient_id']}")
            print(
                f"    dates: {r['matching_dates']}/{r['disk_dates']} match ({r['match_ratio']:.0%})"
            )
            print(f"    confidence: {r['confidence']}")
            if r.get("dst_exists"):
                print("    WARNING: destination already exists!")

    # show orphans
    if orphans:
        print(f"\n--- ORPHAN PATIENTS (first 10 of {len(orphans)}) ---")
        for o in orphans[:10]:
            print(
                f"  {o['disk_patient_id']}: {o['disk_dates']} exams, best match ratio: {o['match_ratio']:.0%}"
            )

    print("\n" + "=" * 80)


def execute_renames(
    renames: List[Dict],
    base_dir: Path,
    dry_run: bool = True,
    confidence_filter: Optional[str] = None,
) -> Tuple[int, int, int]:
    """
    execute the rename operations.

    Parameters
    ----------
    renames : List[Dict]
        list of rename operations
    base_dir : Path
        base directory containing patient folders
    dry_run : bool
        if True, only print what would be done
    confidence_filter : str, optional
        only execute renames with this confidence level or higher
        ("HIGH", "MEDIUM", "LOW")

    Returns
    -------
    Tuple of (successful, skipped, failed)
    """
    confidence_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

    if confidence_filter:
        min_confidence = confidence_order.get(confidence_filter, 0)
        renames = [
            r
            for r in renames
            if confidence_order.get(r["confidence"], 0) >= min_confidence
        ]
        logging.info(
            f"Filtered to {len(renames)} renames with confidence >= {confidence_filter}"
        )

    successful = 0
    skipped = 0
    failed = 0

    for rename in tqdm(renames, desc="Executing renames"):
        src = Path(rename["src_path"])
        dst = Path(rename["dst_path"])

        if not src.exists():
            logging.warning(f"Source does not exist: {src}")
            skipped += 1
            continue

        if dst.exists():
            # destination exists - need to merge
            logging.warning(f"Destination exists, merging: {src} → {dst}")
            if dry_run:
                logging.info(f"[DRY RUN] Would merge {src} into {dst}")
                successful += 1
            else:
                try:
                    # move contents of src into dst
                    for item in src.iterdir():
                        item_dst = dst / item.name
                        if item_dst.exists():
                            logging.warning(f"  Skipping existing: {item.name}")
                            continue
                        shutil.move(str(item), str(item_dst))
                        logging.info(f"  Moved: {item.name}")
                    # remove empty src directory
                    if not any(src.iterdir()):
                        src.rmdir()
                        logging.info(f"  Removed empty source: {src}")
                    successful += 1
                except Exception as e:
                    logging.error(f"Failed to merge {src} → {dst}: {e}")
                    failed += 1
        else:
            if dry_run:
                logging.info(f"[DRY RUN] Would rename {src} → {dst}")
                successful += 1
            else:
                try:
                    src.rename(dst)
                    logging.info(f"Renamed: {src} → {dst}")
                    successful += 1
                except Exception as e:
                    logging.error(f"Failed to rename {src} → {dst}: {e}")
                    failed += 1

    return successful, skipped, failed


def quarantine_orphans(
    orphans: List[Dict],
    base_dir: Path,
    quarantine_dir: Path,
    dry_run: bool = True,
) -> Tuple[int, int, int]:
    """
    move orphan patients to quarantine directory.

    Returns
    -------
    Tuple of (successful, skipped, failed)
    """
    successful = 0
    skipped = 0
    failed = 0

    for orphan in tqdm(orphans, desc="Quarantining orphans"):
        src = Path(orphan["src_path"])

        if not src.exists():
            logging.warning(f"Source does not exist: {src}")
            skipped += 1
            continue

        dst = quarantine_dir / src.name

        if dry_run:
            logging.info(f"[DRY RUN] Would quarantine {src} → {dst}")
            successful += 1
        else:
            try:
                quarantine_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                logging.info(f"Quarantined: {src} → {dst}")
                successful += 1
            except Exception as e:
                logging.error(f"Failed to quarantine {src}: {e}")
                failed += 1

    return successful, skipped, failed


def save_plan_to_csv(
    renames: List[Dict],
    keeps: List[Dict],
    orphans: List[Dict],
    output_path: Path,
):
    """save the full reconciliation plan to CSV for review"""
    all_records = []

    for r in keeps:
        all_records.append({**r, "action": "KEEP"})
    for r in renames:
        all_records.append({**r, "action": "RENAME"})
    for r in orphans:
        all_records.append({**r, "action": "ORPHAN"})

    df = pd.DataFrame(all_records)
    df.to_csv(output_path, index=False)
    logging.info(f"Saved reconciliation plan to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Reconcile patient IDs between disk and iBroker metadata."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/gpfs/data/huo-lab/Image/ChiMEC/MG"),
        help="Base directory containing patient folders",
    )
    parser.add_argument(
        "--fingerprint-cache",
        type=Path,
        default=FINGERPRINT_CACHE,
        help="Path to fingerprint cache JSON",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=IBROKER_METADATA,
        help="Path to iBroker metadata CSV",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Only print what would be done (default: True)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the changes (disables dry-run)",
    )
    parser.add_argument(
        "--confidence",
        choices=["HIGH", "MEDIUM", "LOW"],
        default="HIGH",
        help="Minimum confidence level for renames (default: HIGH)",
    )
    parser.add_argument(
        "--quarantine-orphans",
        action="store_true",
        help="Move orphan patients to quarantine directory",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=2,
        help="Minimum number of matching dates required (default: 2)",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.5,
        help="Minimum ratio of dates that must match (default: 0.5)",
    )
    parser.add_argument(
        "--output-plan",
        type=Path,
        default=Path("data/reconciliation_plan.csv"),
        help="Output CSV file for reconciliation plan",
    )

    args = parser.parse_args()

    # dry_run is True unless --execute is specified
    dry_run = not args.execute

    if dry_run:
        logging.info("=" * 60)
        logging.info("DRY RUN MODE - no changes will be made")
        logging.info("=" * 60)
    else:
        logging.warning("=" * 60)
        logging.warning("EXECUTE MODE - changes WILL be made!")
        logging.warning("=" * 60)

    # Phase 1: Load signatures
    logging.info("Phase 1: Loading signatures...")
    ibroker_signatures = load_ibroker_signatures(args.metadata)
    disk_signatures = load_disk_signatures(args.fingerprint_cache)

    # Phase 2: Find matches
    logging.info("Phase 2: Finding patient matches...")
    matches = find_patient_matches(
        disk_signatures,
        ibroker_signatures,
        min_match_ratio=args.min_ratio,
        min_matches=args.min_matches,
    )

    # Phase 3: Generate plan
    logging.info("Phase 3: Generating reconciliation plan...")
    renames, keeps, orphans = generate_reconciliation_plan(matches, args.base_dir)

    # Save plan to CSV
    save_plan_to_csv(renames, keeps, orphans, args.output_plan)

    # Print summary
    print_reconciliation_summary(renames, keeps, orphans)

    # Phase 4: Execute (if not dry-run or if --execute)
    if renames:
        logging.info(f"Phase 4: Executing renames (confidence >= {args.confidence})...")
        successful, skipped, failed = execute_renames(
            renames,
            args.base_dir,
            dry_run=dry_run,
            confidence_filter=args.confidence,
        )
        logging.info(
            f"Renames: {successful} successful, {skipped} skipped, {failed} failed"
        )

    if args.quarantine_orphans and orphans:
        logging.info("Phase 5: Quarantining orphans...")
        successful, skipped, failed = quarantine_orphans(
            orphans,
            args.base_dir,
            QUARANTINE_DIR,
            dry_run=dry_run,
        )
        logging.info(
            f"Quarantine: {successful} successful, {skipped} skipped, {failed} failed"
        )

    logging.info("Reconciliation complete.")


if __name__ == "__main__":
    main()
