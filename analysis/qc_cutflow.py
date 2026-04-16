#!/usr/bin/env python3
# ruff: noqa: E402
"""qc_cutflow.py

Generate cutflow analysis showing how many exams are filtered by each criterion.
Compares automatic filters against manual QC decisions to validate filter effectiveness.

Usage:
    python analysis/qc_cutflow.py --qc-file data/qc_status.json
"""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def generate_cutflow(
    views_path: Path,
    tags_path: Path,
    qc_file: Path,
    output_file: Path,
) -> None:
    """generate cutflow showing exam filtering at each stage

    Args:
        views_path: path to views.parquet
        tags_path: path to dicom_tags.parquet
        qc_file: path to qc_status.json
        output_file: where to save cutflow results
    """
    print("Loading data...")
    views_df = pd.read_parquet(views_path)

    total_exams = views_df["exam_id"].nunique()

    # load QC data if available
    qc_data = {}
    if qc_file.exists():
        with open(qc_file) as f:
            qc_data = json.load(f)

    # compute all auto-filter sets
    from prima.qc_filters import compute_auto_filter_sets

    filter_sets = compute_auto_filter_sets(views_path, tags_path)

    # track which exams are excluded by each filter
    cutflow = []
    excluded_exams = set()

    cutflow.append(
        {
            "step": "0. Total exams",
            "filter": "None",
            "exams_excluded_this_step": 0,
            "cumulative_excluded": 0,
            "exams_remaining": total_exams,
            "percent_remaining": 100.0,
        }
    )

    filter_steps = [
        ("1. Implants", "has_implant == True", "has_implant"),
        ("2. Scanned Film", "DetectorType == FILM", "scanned_film"),
        (
            "3. GEMS Processing",
            "AcquisitionDeviceProcessingCode starts with GEMS_",
            "gems_ffdm_tc1",
        ),
        (
            "4. Duplicate Exams",
            "Shared SOP Instance UIDs within patient",
            "duplicate_sop_uid",
        ),
    ]

    for step_name, filter_desc, filter_key in filter_steps:
        if filter_key not in filter_sets:
            continue
        matched = filter_sets[filter_key]
        new_excluded = matched - excluded_exams
        excluded_exams.update(matched)
        cutflow.append(
            {
                "step": step_name,
                "filter": filter_desc,
                "exams_excluded_this_step": len(new_excluded),
                "cumulative_excluded": len(excluded_exams),
                "exams_remaining": total_exams - len(excluded_exams),
                "percent_remaining": (total_exams - len(excluded_exams))
                / total_exams
                * 100,
            }
        )

    # Create DataFrame
    cutflow_df = pd.DataFrame(cutflow)

    # Print cutflow
    print("\n" + "=" * 80)
    print("CUTFLOW: Automatic Filter Coverage")
    print("=" * 80)
    print()
    print(cutflow_df.to_string(index=False))
    print()

    # If we have QC data, analyze coverage
    if qc_data:
        bad_exams = {eid for eid, status in qc_data.items() if status == "bad"}
        good_exams = {eid for eid, status in qc_data.items() if status == "good"}

        print("=" * 80)
        print(f"MANUAL QC COVERAGE (based on {len(qc_data)} QC'd exams)")
        print("=" * 80)
        print()

        # For each filter, check how many manually-marked bad exams it would catch
        print("Filter effectiveness on manually marked BAD exams:")
        print("-" * 80)

        total_caught = set()
        for filter_name, filter_exams in filter_sets.items():
            caught_bad = bad_exams & filter_exams
            caught_good = good_exams & filter_exams
            total_caught.update(caught_bad)

            if len(caught_bad) > 0 or len(caught_good) > 0:
                precision = (
                    len(caught_bad) / (len(caught_bad) + len(caught_good))
                    if (len(caught_bad) + len(caught_good)) > 0
                    else 0
                )
                recall = len(caught_bad) / len(bad_exams) if len(bad_exams) > 0 else 0

                print(f"\n{filter_name}:")
                print(
                    f"  Catches {len(caught_bad)} / {len(bad_exams)} manually marked bad ({recall:.1%} recall)"
                )
                print(f"  False positives: {len(caught_good)} good exams")
                print(
                    f"  Precision: {precision:.1%} (of exams caught by filter, {precision:.1%} are truly bad)"
                )

                if precision >= 0.95:
                    print("  ✅ HIGH CONFIDENCE - safe to auto-exclude")
                elif precision >= 0.8:
                    print("  ⚠️  GOOD - consider auto-excluding after validation")
                elif precision >= 0.5:
                    print("  ⚠️  MIXED - needs more review")
                else:
                    print("  ❌ LOW PRECISION - do not auto-exclude")

        # Summary
        uncaught_bad = bad_exams - total_caught

        print("\n" + "=" * 80)
        print("SUMMARY:")
        print(f"  Total manually marked bad: {len(bad_exams)}")
        print(
            f"  Caught by at least one filter: {len(total_caught)} ({len(total_caught) / len(bad_exams):.1%})"
        )
        print(
            f"  Still need manual QC: {len(uncaught_bad)} ({len(uncaught_bad) / len(bad_exams):.1%})"
        )
        print("=" * 80)

        # List filters by effectiveness
        print("\nFILTERS BY EFFECTIVENESS (bad exams caught):")
        filter_effectiveness = []
        for filter_name, filter_exams in filter_sets.items():
            caught_bad = bad_exams & filter_exams
            caught_good = good_exams & filter_exams
            if len(caught_bad) > 0:
                precision = (
                    len(caught_bad) / (len(caught_bad) + len(caught_good))
                    if (len(caught_bad) + len(caught_good)) > 0
                    else 0
                )
                filter_effectiveness.append(
                    {
                        "filter": filter_name,
                        "bad_caught": len(caught_bad),
                        "good_caught": len(caught_good),
                        "precision": precision,
                    }
                )

        eff_df = pd.DataFrame(filter_effectiveness).sort_values(
            "bad_caught", ascending=False
        )
        print(eff_df.to_string(index=False))

    # Save to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cutflow_df.to_csv(output_file, index=False)
    print(f"\n\nCutflow saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate cutflow analysis for QC filtering"
    )
    parser.add_argument(
        "--views",
        type=Path,
        default=Path("/gpfs/data/huo-lab/Image/ChiMEC/MG/sot/views.parquet"),
        help="path to views.parquet",
    )
    parser.add_argument(
        "--tags",
        type=Path,
        default=Path("/gpfs/data/huo-lab/Image/ChiMEC/MG/sot/dicom_tags.parquet"),
        help="path to dicom_tags.parquet",
    )
    parser.add_argument(
        "--qc-file",
        type=Path,
        default=Path("data/qc_status.json"),
        help="path to QC status file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/qc_cutflow.csv"),
        help="output file for cutflow results",
    )

    args = parser.parse_args()

    generate_cutflow(
        views_path=args.views,
        tags_path=args.tags,
        qc_file=args.qc_file,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
