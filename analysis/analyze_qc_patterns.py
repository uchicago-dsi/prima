#!/usr/bin/env python3
"""analyze_qc_patterns.py

Analyze DICOM tags of QC'd exams to find patterns that distinguish bad from good.
Run this after each QC batch to discover filters for the next batch.

Usage:
    python analysis/analyze_qc_patterns.py --qc-file data/qc_status.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_qc_data(qc_file: Path) -> dict:
    """load QC status mapping"""
    if not qc_file.exists():
        return {}
    with open(qc_file) as f:
        return json.load(f)


def analyze_patterns(
    views_path: Path,
    tags_path: Path,
    qc_file: Path,
    min_bad_examples: int = 5,
) -> None:
    """find DICOM tag patterns that distinguish bad from good exams

    Args:
        views_path: path to views.parquet
        tags_path: path to dicom_tags.parquet
        qc_file: path to qc_status.json
        min_bad_examples: minimum number of bad examples to consider a pattern
    """
    print("Loading data...")
    views_df = pd.read_parquet(views_path)
    tags_df = pd.read_parquet(tags_path)
    qc_data = load_qc_data(qc_file)

    if not qc_data:
        print("No QC data found. Run QC first!")
        return

    # merge QC status with views and tags
    views_df["qc_status"] = views_df["exam_id"].map(qc_data)
    merged = views_df.merge(tags_df, on="sop_instance_uid", how="left")

    # filter to QC'd exams only
    qc_merged = merged[merged["qc_status"].notna()].copy()

    print("\nQC Summary:")
    print(f"  Total exams QC'd: {len(qc_data)}")
    print(f"  Good: {sum(1 for s in qc_data.values() if s == 'good')}")
    print(f"  Bad: {sum(1 for s in qc_data.values() if s == 'bad')}")
    print(f"  Review: {sum(1 for s in qc_data.values() if s == 'review')}")

    bad_views = qc_merged[qc_merged["qc_status"] == "bad"]
    good_views = qc_merged[qc_merged["qc_status"] == "good"]

    if len(bad_views) < min_bad_examples:
        print(f"\nNeed at least {min_bad_examples} bad examples. Keep QC'ing!")
        return

    print(f"\n{'=' * 60}")
    print("PATTERN ANALYSIS")
    print(f"{'=' * 60}")

    # analyze numeric columns
    print("\n1. NUMERIC DICOM TAG PATTERNS:")
    print("-" * 60)

    numeric_cols = tags_df.select_dtypes(include=[np.number]).columns
    patterns = []

    for col in numeric_cols:
        if col == "sop_instance_uid":
            continue

        bad_vals = pd.to_numeric(bad_views[col], errors="coerce").dropna()
        good_vals = pd.to_numeric(good_views[col], errors="coerce").dropna()

        if len(bad_vals) < min_bad_examples or len(good_vals) < 10:
            continue

        # statistical test
        try:
            statistic, pvalue = stats.mannwhitneyu(
                bad_vals, good_vals, alternative="two-sided"
            )

            if pvalue < 0.05:
                bad_median = bad_vals.median()
                good_median = good_vals.median()
                bad_range = (bad_vals.min(), bad_vals.max())
                good_range = (good_vals.min(), good_vals.max())

                patterns.append(
                    {
                        "tag": col,
                        "pvalue": pvalue,
                        "bad_median": bad_median,
                        "good_median": good_median,
                        "bad_range": bad_range,
                        "good_range": good_range,
                    }
                )
        except Exception:
            continue

    # sort by significance
    patterns.sort(key=lambda x: x["pvalue"])

    for i, p in enumerate(patterns[:10], 1):
        print(f"\n{i}. {p['tag']}")
        print(f"   p-value: {p['pvalue']:.4f}")
        print(f"   Bad:  median={p['bad_median']:.2f}, range={p['bad_range']}")
        print(f"   Good: median={p['good_median']:.2f}, range={p['good_range']}")

        # suggest filter
        if p["bad_median"] < p["good_median"]:
            threshold = p["bad_range"][1]
            print(f"   → FILTER: Exclude if {p['tag']} < {threshold:.2f}")
        else:
            threshold = p["bad_range"][0]
            print(f"   → FILTER: Exclude if {p['tag']} > {threshold:.2f}")

    # analyze categorical/string columns
    print("\n\n2. CATEGORICAL DICOM TAG PATTERNS:")
    print("-" * 60)

    string_cols = [c for c in tags_df.columns if c not in numeric_cols]

    for col in string_cols:
        if col == "sop_instance_uid":
            continue

        bad_vals = bad_views[col].dropna()
        good_vals = good_views[col].dropna()

        if len(bad_vals) < min_bad_examples or len(good_vals) < 10:
            continue

        # find values that appear mostly in bad exams
        bad_value_counts = bad_vals.value_counts()
        good_value_counts = good_vals.value_counts()

        for val, bad_count in bad_value_counts.items():
            good_count = good_value_counts.get(val, 0)

            # if this value appears in >50% of bad exams but <10% of good exams
            bad_frac = bad_count / len(bad_vals)
            good_frac = good_count / len(good_vals) if len(good_vals) > 0 else 0

            if bad_frac > 0.3 and good_frac < 0.1:
                print(f"\n{col} = '{val}'")
                print(f"   Bad exams: {bad_count}/{len(bad_vals)} ({bad_frac:.1%})")
                print(f"   Good exams: {good_count}/{len(good_vals)} ({good_frac:.1%})")
                print(f"   → FILTER: Exclude if {col} == '{val}'")

    # analyze image dimensions
    print("\n\n3. IMAGE DIMENSION PATTERNS:")
    print("-" * 60)

    for dim_col in ["rows", "cols"]:
        if dim_col in views_df.columns:
            bad_vals = bad_views[dim_col].dropna()
            good_vals = good_views[dim_col].dropna()

            print(f"\n{dim_col.upper()}:")
            print(
                f"   Bad:  {bad_vals.min()}-{bad_vals.max()} (median: {bad_vals.median()})"
            )
            print(
                f"   Good: {good_vals.min()}-{good_vals.max()} (median: {good_vals.median()})"
            )

            # check for outliers
            bad_outliers = bad_vals[
                (bad_vals < good_vals.quantile(0.01))
                | (bad_vals > good_vals.quantile(0.99))
            ]
            if len(bad_outliers) >= min_bad_examples:
                print("   → FILTER: Exclude unusual dimensions")

    # check device models
    print("\n\n4. DEVICE MODEL PATTERNS:")
    print("-" * 60)

    bad_devices = bad_views.groupby("device_model").size()
    good_devices = good_views.groupby("device_model").size()

    all_devices = set(bad_devices.index) | set(good_devices.index)

    for device in all_devices:
        bad_count = bad_devices.get(device, 0)
        good_count = good_devices.get(device, 0)

        if bad_count + good_count < min_bad_examples:
            continue

        bad_rate = bad_count / (bad_count + good_count)

        if bad_rate > 0.5:
            print(f"\n{device}")
            print(f"   Bad: {bad_count}, Good: {good_count}")
            print(f"   Bad rate: {bad_rate:.1%}")
            if bad_rate > 0.8:
                print("   → FILTER: Consider excluding this device model")

    # check specific known problematic patterns
    print("\n\n5. KNOWN PROBLEMATIC PATTERNS:")
    print("-" * 60)

    # check for positioning artifacts
    if "PositionerPrimaryAngle" in merged.columns:
        bad_angles = bad_views["PositionerPrimaryAngle"].dropna()
        good_angles = good_views["PositionerPrimaryAngle"].dropna()

        bad_negative = (pd.to_numeric(bad_angles, errors="coerce") < 0).sum()
        good_negative = (pd.to_numeric(good_angles, errors="coerce") < 0).sum()

        if bad_negative > 0:
            print("\nPositionerPrimaryAngle < 0:")
            print(
                f"   Bad: {bad_negative}/{len(bad_angles)} ({bad_negative / len(bad_angles):.1%})"
            )
            print(
                f"   Good: {good_negative}/{len(good_angles)} ({good_negative / len(good_angles):.1%})"
            )
            if bad_negative > good_negative * 2:
                print("   → FILTER: Exclude if PositionerPrimaryAngle < 0")

    # check compression force
    if "CompressionForce" in merged.columns:
        bad_comp = pd.to_numeric(
            bad_views["CompressionForce"], errors="coerce"
        ).dropna()
        good_comp = pd.to_numeric(
            good_views["CompressionForce"], errors="coerce"
        ).dropna()

        bad_zero = (bad_comp == 0).sum()
        good_zero = (good_comp == 0).sum()

        if bad_zero > 0:
            print("\nCompressionForce == 0:")
            print(
                f"   Bad: {bad_zero}/{len(bad_comp)} ({bad_zero / len(bad_comp):.1%})"
            )
            print(
                f"   Good: {good_zero}/{len(good_comp)} ({good_zero / len(good_comp):.1%})"
            )
            if bad_zero > min_bad_examples and bad_zero / len(bad_comp) > 0.3:
                print("   → FILTER: Exclude if CompressionForce == 0")

    # check for implants
    if "has_implant" in views_df.columns:
        bad_implant = bad_views["has_implant"].sum()
        good_implant = good_views["has_implant"].sum()

        if bad_implant > 0 or good_implant > 0:
            print("\nhas_implant == True:")
            print(
                f"   Bad: {bad_implant}/{len(bad_views)} ({bad_implant / len(bad_views):.1%})"
            )
            print(
                f"   Good: {good_implant}/{len(good_views)} ({good_implant / len(good_views):.1%})"
            )
            print(
                "   → RECOMMENDATION: Exclude implants (Mirai not validated on implants)"
            )

    # check for scanned film (secondary capture)
    if "SOPClassUID" in merged.columns:
        # Secondary Capture Image Storage = 1.2.840.10008.5.1.4.1.1.7
        bad_sc = (
            bad_views["SOPClassUID"]
            .str.contains("1.2.840.10008.5.1.4.1.1.7", na=False)
            .sum()
        )
        good_sc = (
            good_views["SOPClassUID"]
            .str.contains("1.2.840.10008.5.1.4.1.1.7", na=False)
            .sum()
        )

        if bad_sc > 0 or good_sc > 0:
            print("\nScanned Film (Secondary Capture):")
            print(f"   Bad: {bad_sc}/{len(bad_views)} ({bad_sc / len(bad_views):.1%})")
            print(
                f"   Good: {good_sc}/{len(good_views)} ({good_sc / len(good_views):.1%})"
            )
            print("   → FILTER: Exclude scanned film (Mirai trained on digital only)")

    # check filter type
    if "FilterType" in merged.columns:
        bad_filter = bad_views["FilterType"].value_counts()
        good_filter = good_views["FilterType"].value_counts()

        for filt_type in bad_filter.index:
            bad_count = bad_filter.get(filt_type, 0)
            good_count = good_filter.get(filt_type, 0)

            if bad_count >= min_bad_examples:
                bad_frac = bad_count / len(bad_views)
                good_frac = good_count / len(good_views) if len(good_views) > 0 else 0

                if bad_frac > 0.3 and good_frac < 0.1:
                    print(f"\nFilterType == '{filt_type}':")
                    print(f"   Bad: {bad_count}/{len(bad_views)} ({bad_frac:.1%})")
                    print(f"   Good: {good_count}/{len(good_views)} ({good_frac:.1%})")
                    print(f"   → FILTER: Consider excluding FilterType='{filt_type}'")

    print(f"\n{'=' * 60}")
    print("NEXT STEPS:")
    print("1. Test hypothesis with test_positioning_filters.py")
    print("2. Add validated filters to pipelines/preprocess.py")
    print("3. Re-run QC with filters to process next batch")
    print("4. Repeat until most exams are auto-filtered")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze QC patterns to discover filters"
    )
    parser.add_argument(
        "--qc-file",
        type=Path,
        default=Path("data/qc_status.json"),
        help="path to QC status file",
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
        "--min-bad-examples",
        type=int,
        default=5,
        help="minimum number of bad examples to consider a pattern",
    )

    args = parser.parse_args()

    analyze_patterns(
        views_path=args.views,
        tags_path=args.tags,
        qc_file=args.qc_file,
        min_bad_examples=args.min_bad_examples,
    )


if __name__ == "__main__":
    main()
