#!/usr/bin/env python3
"""test_vertical_line_inpainting.py

Test script for detecting and inpainting vertical line artifacts.
Creates before/after visualizations to validate the approach.

Usage:
    python experiments/test_vertical_line_inpainting.py --n-samples 10
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom


def detect_vertical_lines(pixel_array: np.ndarray, threshold: float = 4.0):
    """detect vertical line artifacts and return their column positions

    Args:
        pixel_array: 2D image array
        threshold: z-score threshold for detecting anomalous columns

    Returns:
        list of (start_col, end_col, center_col, z_score) tuples
    """
    # compute mean intensity for each column
    col_means = pixel_array.mean(axis=0)

    # compute z-scores
    col_mean_std = col_means.std()
    col_mean_avg = col_means.mean()

    if col_mean_std == 0:
        return []

    z_scores = np.abs(col_means - col_mean_avg) / col_mean_std

    # flag anomalous columns
    anomalous = z_scores > threshold

    # find consecutive runs (actual lines vs scattered noise)
    line_positions = []
    i = 0
    while i < len(anomalous):
        if anomalous[i]:
            # found start of a line
            start = i
            while i < len(anomalous) and anomalous[i]:
                i += 1
            end = i

            # require at least 2 consecutive columns to be a line (not single-pixel noise)
            if end - start >= 1:
                line_col = (start + end) // 2
                line_positions.append((start, end, line_col, z_scores[line_col]))
        else:
            i += 1

    return line_positions


def inpaint_vertical_lines(
    pixel_array: np.ndarray, line_positions: list, method: str = "interpolate"
):
    """inpaint vertical line artifacts

    Args:
        pixel_array: 2D image array
        line_positions: list of (start_col, end_col, center_col, z_score) tuples
        method: 'interpolate' or 'median'

    Returns:
        inpainted array
    """
    result = pixel_array.copy()

    for start, end, center, z_score in line_positions:
        # define inpainting region (add small margin)
        left = max(0, start)
        right = min(pixel_array.shape[1], end)

        # define reference regions on either side
        margin = 10
        left_ref_start = max(0, left - margin * 2)
        left_ref_end = max(0, left - 1)
        right_ref_start = min(pixel_array.shape[1], right + 1)
        right_ref_end = min(pixel_array.shape[1], right + margin * 2)

        if method == "interpolate":
            # get reference columns
            if left_ref_end > left_ref_start:
                left_ref = pixel_array[:, left_ref_start:left_ref_end].mean(axis=1)
            else:
                left_ref = None

            if right_ref_end > right_ref_start:
                right_ref = pixel_array[:, right_ref_start:right_ref_end].mean(axis=1)
            else:
                right_ref = None

            # inpaint each column in the artifact region
            for col in range(left, right):
                if left_ref is not None and right_ref is not None:
                    # linear interpolation between left and right
                    alpha = (col - left) / max(1, right - left)
                    result[:, col] = (1 - alpha) * left_ref + alpha * right_ref
                elif left_ref is not None:
                    result[:, col] = left_ref
                elif right_ref is not None:
                    result[:, col] = right_ref

        elif method == "median":
            # use median filter along rows to fill in the artifact
            for col in range(left, right):
                # for each row, replace with median of neighboring columns
                for row in range(pixel_array.shape[0]):
                    neighborhood = []
                    if left_ref_end > left_ref_start:
                        neighborhood.extend(
                            pixel_array[row, left_ref_start:left_ref_end]
                        )
                    if right_ref_end > right_ref_start:
                        neighborhood.extend(
                            pixel_array[row, right_ref_start:right_ref_end]
                        )
                    if neighborhood:
                        result[row, col] = np.median(neighborhood)

    return result


def process_exam(exam_id: str, views_df: pd.DataFrame, raw_dir: Path, output_dir: Path):
    """process one exam: detect artifacts, inpaint, visualize"""
    exam_views = views_df[views_df["exam_id"] == exam_id]

    if len(exam_views) == 0:
        return None

    # process each view
    results = []

    for _, row in exam_views.iterrows():
        try:
            path = raw_dir / row["dicom_path"]
            ds = pydicom.dcmread(str(path))
            pixels = ds.pixel_array.astype(np.float32)

            # detect lines
            lines = detect_vertical_lines(pixels, threshold=4.0)

            if len(lines) > 0:
                # inpaint
                inpainted = inpaint_vertical_lines(pixels, lines, method="interpolate")

                results.append(
                    {
                        "view": f"{row['laterality']}_{row['view']}",
                        "original": pixels,
                        "inpainted": inpainted,
                        "lines": lines,
                    }
                )
        except Exception as e:
            print(f"Error processing {row['dicom_path']}: {e}")

    if len(results) == 0:
        return None

    # create visualization
    fig = plt.figure(figsize=(20, 5 * len(results)))

    for idx, result in enumerate(results):
        # original
        ax = plt.subplot(len(results), 3, idx * 3 + 1)
        ax.imshow(result["original"], cmap="gray")
        lines_str = ", ".join(
            [f"col {c} (z={z:.1f})" for _, _, c, z in result["lines"]]
        )
        ax.set_title(f"{result['view']} - Original\nDetected: {lines_str}", fontsize=10)
        ax.axis("off")

        # inpainted
        ax = plt.subplot(len(results), 3, idx * 3 + 2)
        ax.imshow(result["inpainted"], cmap="gray")
        ax.set_title(f"{result['view']} - Inpainted", fontsize=10)
        ax.axis("off")

        # difference (amplified)
        ax = plt.subplot(len(results), 3, idx * 3 + 3)
        diff = np.abs(result["original"] - result["inpainted"])
        ax.imshow(diff, cmap="hot", vmax=diff.mean() + 3 * diff.std())
        ax.set_title(f"{result['view']} - Difference (amplified)", fontsize=10)
        ax.axis("off")

    plt.tight_layout()

    # save
    output_path = output_dir / f"{exam_id}_inpaint_test.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return {
        "exam_id": exam_id,
        "n_views_with_artifacts": len(results),
        "output_path": output_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test vertical line inpainting with before/after visualization"
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("/gpfs/data/huo-lab/Image/ChiMEC/MG"),
        help="raw data directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("inpaint_test_output"),
        help="output directory for test figures",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="number of exams to test",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=4.0,
        help="z-score threshold for line detection",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading views...")
    views = pd.read_parquet(args.raw / "sot" / "views.parquet")
    views = views[views["for_presentation"]]

    # find exams with artifacts
    print(f"Searching for {args.n_samples} exams with vertical artifacts...")
    print("(will stop after finding the requested number)")
    print()
    exam_ids = views["exam_id"].unique()

    tested_exams = []
    processed = 0
    checked = 0

    for exam_id in exam_ids:
        if processed >= args.n_samples:
            break

        checked += 1
        if checked % 50 == 0:
            print(f"  Checked {checked} exams, found {processed} with artifacts...")

        exam_views = views[views["exam_id"] == exam_id]

        # quick check if this exam has artifacts
        has_artifact = False
        for _, row in exam_views.iterrows():
            try:
                path = args.raw / row["dicom_path"]
                ds = pydicom.dcmread(str(path))
                pixels = ds.pixel_array.astype(np.float32)

                lines = detect_vertical_lines(pixels, threshold=args.threshold)
                if len(lines) > 0:
                    has_artifact = True
                    break
            except Exception:
                continue

        if has_artifact:
            print(f"  ✓ Found artifact in {exam_id}, processing...")
            result = process_exam(exam_id, views, args.raw, args.output)
            if result:
                tested_exams.append(result)
                processed += 1
                print(
                    f"    [{processed}/{args.n_samples}] Saved: {result['output_path'].name}"
                )

    print()
    print("=" * 60)
    print("INPAINTING TEST COMPLETE")
    print("=" * 60)
    print(f"Exams checked: {checked}")
    print(
        f"Exams with artifacts: {len(tested_exams)} ({len(tested_exams) / checked * 100:.1f}%)"
    )
    print(f"Output directory: {args.output}")
    print()

    if len(tested_exams) == 0:
        print("⚠️  No exams with artifacts found!")
        print(f"   Checked {checked} exams with threshold={args.threshold}")
        print("   Try lowering --threshold or the detection logic needs adjustment")
        return 1

    print("Review the before/after images to validate inpainting quality:")
    for result in tested_exams:
        print(f"  {result['output_path']}")
    print()
    print("If inpainting looks good:")
    print("  1. Integrate into pipelines/preprocess.py (apply before writing to Zarr)")
    print("  2. Delete existing Zarr cache and zarr manifest")
    print("  3. Re-run preprocessing with inpainting enabled")
    print("  4. Re-run Mirai inference")

    return 0


if __name__ == "__main__":
    main()
