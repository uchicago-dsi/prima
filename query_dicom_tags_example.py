#!/usr/bin/env python3
"""Examples of how to query the DICOM tags table.

The preprocessing pipeline saves all DICOM tags to sot/dicom_tags.parquet
in wide format (one row per image, one column per tag).

Schema:
    - sop_instance_uid: unique identifier linking to views table
    - <tag_keyword>: one column per DICOM tag (e.g., "Modality", "KVP", "ViewPosition")

All values are stored as strings. Cast to numeric as needed for filtering.
"""

from pathlib import Path

import pandas as pd


def example_queries(sot_dir: Path = Path("sot")):
    """demonstrate common queries on the DICOM tags table (wide format)"""

    # load tables
    tags_df = pd.read_parquet(sot_dir / "dicom_tags.parquet")
    views_df = pd.read_parquet(sot_dir / "views.parquet")

    num_tags = len(tags_df.columns) - 1  # exclude sop_instance_uid
    print(f"Loaded {len(tags_df):,} images with {num_tags} DICOM tag columns\n")

    # example 1: filter by numeric tag value (much simpler in wide format!)
    print("=" * 60)
    print("Example 1: Filter by ExposureTime > 100")
    print("=" * 60)
    if "ExposureTime" in tags_df.columns:
        tags_df["ExposureTime_numeric"] = pd.to_numeric(
            tags_df["ExposureTime"], errors="coerce"
        )
        high_exposure = tags_df[tags_df["ExposureTime_numeric"] > 100]
        print(f"Found {len(high_exposure)} images with ExposureTime > 100")
        if len(high_exposure) > 0:
            print(f"Example values: {high_exposure['ExposureTime'].head(3).tolist()}\n")
    else:
        print("ExposureTime column not found\n")

    # example 2: get all tags for a specific image
    print("=" * 60)
    print("Example 2: Get all tags for one image")
    print("=" * 60)
    sample_sop = tags_df["sop_instance_uid"].iloc[0]
    image_row = tags_df[tags_df["sop_instance_uid"] == sample_sop].iloc[0]
    # show non-null tags
    non_null_tags = {
        k: v for k, v in image_row.items() if pd.notna(v) and k != "sop_instance_uid"
    }
    print(f"Image {sample_sop} has {len(non_null_tags)} non-null tags:")
    for i, (tag, value) in enumerate(list(non_null_tags.items())[:10]):
        print(f"  {tag}: {value}")
    if len(non_null_tags) > 10:
        print(f"  ... and {len(non_null_tags) - 10} more\n")
    else:
        print()

    # example 3: select multiple tags (already in wide format!)
    print("=" * 60)
    print("Example 3: Select specific tags and merge with views")
    print("=" * 60)
    tags_of_interest = ["ViewPosition", "PatientAge", "KVP", "ExposureTime"]
    available_tags = [t for t in tags_of_interest if t in tags_df.columns]

    if available_tags:
        subset = tags_df[["sop_instance_uid"] + available_tags]
        # merge with views to get laterality and view
        result = subset.merge(
            views_df[["sop_instance_uid", "laterality", "view"]],
            on="sop_instance_uid",
            how="left",
        )
        print(f"Selected {len(available_tags)} tags for {len(result)} images:")
        print(result.head().to_string(index=False))
        print()
    else:
        print("None of the requested tags found\n")

    # example 4: find images missing a specific tag
    print("=" * 60)
    print("Example 4: Find images missing PatientAge tag")
    print("=" * 60)
    if "PatientAge" in tags_df.columns:
        missing_age = tags_df["PatientAge"].isna().sum()
        print(f"{missing_age} out of {len(tags_df)} images are missing PatientAge\n")
    else:
        print("PatientAge column not present in any image\n")

    # example 5: tag completeness analysis
    print("=" * 60)
    print("Example 5: Tag completeness (% of images with each tag)")
    print("=" * 60)
    completeness = {}
    for col in tags_df.columns:
        if col != "sop_instance_uid":
            pct = (tags_df[col].notna().sum() / len(tags_df)) * 100
            completeness[col] = pct

    completeness_sorted = sorted(completeness.items(), key=lambda x: x[1], reverse=True)
    print("Top 15 most complete tags:")
    for tag, pct in completeness_sorted[:15]:
        print(f"  {tag}: {pct:.1f}%")
    print()

    # example 6: value distribution for a categorical tag
    print("=" * 60)
    print("Example 6: Value distribution for ViewPosition")
    print("=" * 60)
    if "ViewPosition" in tags_df.columns:
        view_pos = tags_df["ViewPosition"].value_counts()
        print(view_pos.to_string())
    else:
        print("ViewPosition column not found")


if __name__ == "__main__":
    import sys

    sot_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sot")

    if not (sot_dir / "dicom_tags.parquet").exists():
        print(f"Error: {sot_dir / 'dicom_tags.parquet'} not found")
        print("Run preprocessing first to generate the tags table")
        sys.exit(1)

    example_queries(sot_dir)
