#!/usr/bin/env python3
"""check_vertical_artifacts.py

Quick script to check how many exams have vertical line artifacts.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm


def has_vertical_line_artifact(pixel_array: np.ndarray, threshold: float = 4.0):
    """detect vertical line artifacts by checking column-wise intensity variance

    Returns:
        tuple of (has_artifact: bool, max_z_score: float)
    """
    # compute mean intensity for each column
    col_means = pixel_array.mean(axis=0)

    # compute z-scores for each column
    col_mean_std = col_means.std()
    col_mean_avg = col_means.mean()

    if col_mean_std == 0:
        return False, 0.0

    z_scores = np.abs(col_means - col_mean_avg) / col_mean_std
    max_z = z_scores.max()

    # flag if any column is anomalous
    anomalous = z_scores > threshold

    if anomalous.sum() > 0:
        # check for consecutive anomalous columns (line vs noise)
        indices = np.where(anomalous)[0]
        if len(indices) > 1:
            consecutive = np.diff(indices) == 1
            if consecutive.any():
                return True, max_z

    return False, max_z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("/gpfs/data/huo-lab/Image/ChiMEC/MG"),
        help="raw data directory",
    )
    parser.add_argument(
        "--n-samples", type=int, default=200, help="number of exams to check"
    )
    parser.add_argument(
        "--threshold", type=float, default=4.0, help="z-score threshold"
    )
    args = parser.parse_args()

    print("Loading views...")
    views = pd.read_parquet(args.raw / "sot" / "views.parquet")
    views = views[views["for_presentation"]]

    # sample exams
    exam_ids = views["exam_id"].unique()
    sampled_exams = np.random.choice(
        exam_ids, size=min(args.n_samples, len(exam_ids)), replace=False
    )

    print(f"Checking {len(sampled_exams)} exams for vertical artifacts...")

    flagged_exams = []
    max_z_scores = []

    for exam_id in tqdm(sampled_exams):
        exam_views = views[views["exam_id"] == exam_id]

        has_artifact = False
        exam_max_z = 0.0

        for _, row in exam_views.iterrows():
            try:
                path = args.raw / row["dicom_path"]
                ds = pydicom.dcmread(str(path))
                pixels = ds.pixel_array.astype(np.float32)

                artifact, max_z = has_vertical_line_artifact(pixels, args.threshold)
                exam_max_z = max(exam_max_z, max_z)

                if artifact:
                    has_artifact = True
                    break
            except Exception as e:
                print(f"Error loading {row['dicom_path']}: {e}")

        max_z_scores.append(exam_max_z)
        if has_artifact:
            flagged_exams.append(exam_id)

    print()
    print("=" * 60)
    print(f"Results (threshold={args.threshold}):")
    print(f"  Exams checked: {len(sampled_exams)}")
    print(
        f"  Exams with vertical artifacts: {len(flagged_exams)} ({len(flagged_exams) / len(sampled_exams) * 100:.1f}%)"
    )
    print(f"  Max z-score across all exams: {max(max_z_scores):.2f}")
    print()
    print("Sample flagged exams:")
    for exam_id in flagged_exams[:5]:
        print(f"  {exam_id}")

    if len(flagged_exams) > 0:
        print()
        print("To review flagged exams, create a file with exam IDs and use:")
        print(
            "  python qc/qc_gallery.py --serve --exam-list flagged_vertical_artifacts.txt"
        )


if __name__ == "__main__":
    main()
