#!/usr/bin/env python3
"""test_inpainting_methods.py

Test different inpainting methods for vertical line artifacts.
Compares multiple approaches side-by-side.

Usage:
    python experiments/test_inpainting_methods.py --exam-id <exam_id>
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from scipy import ndimage


def detect_vertical_line_simple(pixel_array: np.ndarray, threshold: float = 4.0):
    """detect vertical lines by column mean anomalies"""
    col_means = pixel_array.mean(axis=0)
    col_mean_std = col_means.std()
    col_mean_avg = col_means.mean()

    if col_mean_std == 0:
        return []

    z_scores = np.abs(col_means - col_mean_avg) / col_mean_std
    anomalous = z_scores > threshold

    # find consecutive runs
    line_positions = []
    i = 0
    while i < len(anomalous):
        if anomalous[i]:
            start = i
            while i < len(anomalous) and anomalous[i]:
                i += 1
            end = i
            if end - start >= 1:
                line_col = (start + end) // 2
                line_positions.append((start, end, line_col, z_scores[line_col]))
        else:
            i += 1

    return line_positions


def inpaint_method_1_wide_interpolation(pixel_array: np.ndarray, start: int, end: int):
    """method 1: interpolate using wide reference regions (50 pixels each side)"""
    result = pixel_array.copy()

    margin = 50
    left_ref_start = max(0, start - margin)
    left_ref_end = start
    right_ref_start = end
    right_ref_end = min(pixel_array.shape[1], end + margin)

    # get reference regions
    if left_ref_end > left_ref_start and right_ref_end > right_ref_start:
        left_ref = pixel_array[:, left_ref_start:left_ref_end].mean(axis=1)
        right_ref = pixel_array[:, right_ref_start:right_ref_end].mean(axis=1)

        # linear interpolation
        for col in range(start, end):
            alpha = (col - start) / max(1, end - start)
            result[:, col] = (1 - alpha) * left_ref + alpha * right_ref

    return result


def inpaint_method_2_median_filter(pixel_array: np.ndarray, start: int, end: int):
    """method 2: median filter across neighboring columns"""
    result = pixel_array.copy()

    # create a median filter mask
    for col in range(start, end):
        for row in range(pixel_array.shape[0]):
            # sample from columns on either side
            left_start = max(0, col - 30)
            left_end = start
            right_start = end
            right_end = min(pixel_array.shape[1], col + 30)

            neighborhood = []
            if left_end > left_start:
                neighborhood.extend(pixel_array[row, left_start:left_end])
            if right_end > right_start:
                neighborhood.extend(pixel_array[row, right_start:right_end])

            if len(neighborhood) > 0:
                result[row, col] = np.median(neighborhood)

    return result


def inpaint_method_3_morphological(pixel_array: np.ndarray, start: int, end: int):
    """method 3: use morphological closing to fill the line"""
    result = pixel_array.copy()

    # create mask for artifact region
    mask = np.zeros_like(pixel_array, dtype=bool)
    mask[:, start:end] = True

    # use median filter only on masked region
    filtered = ndimage.median_filter(
        pixel_array, size=(3, 11)
    )  # tall kernel to smooth vertically

    # replace artifact region with filtered version
    result[mask] = filtered[mask]

    return result


def inpaint_method_4_gaussian_blur(pixel_array: np.ndarray, start: int, end: int):
    """method 4: gaussian blur with edge preservation"""
    result = pixel_array.copy()

    # extract region with margins
    margin = 20
    left = max(0, start - margin)
    right = min(pixel_array.shape[1], end + margin)

    region = pixel_array[:, left:right].copy()

    # apply gaussian blur to entire region
    blurred = ndimage.gaussian_filter(region, sigma=(2, 5))  # more blur horizontally

    # blend back only the artifact columns
    artifact_start = start - left
    artifact_end = end - left

    # use blurred version for artifact, but taper at edges
    for col in range(artifact_start, artifact_end):
        # taper weight from edges to center
        dist_from_edge = min(col - artifact_start, artifact_end - col)
        blend_weight = min(1.0, dist_from_edge / 5.0)
        result[:, left + col] = (
            blend_weight * blurred[:, col] + (1 - blend_weight) * region[:, col]
        )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exam-id", required=True, help="exam ID to test")
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("/gpfs/data/huo-lab/Image/ChiMEC/MG"),
        help="raw data directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("inpaint_method_comparison.png"),
        help="output comparison figure",
    )
    parser.add_argument("--threshold", type=float, default=4.0)
    args = parser.parse_args()

    print(f"Loading exam {args.exam_id}...")
    views = pd.read_parquet(args.raw / "sot" / "views.parquet")
    exam_views = views[views["exam_id"] == args.exam_id]

    if len(exam_views) == 0:
        print(f"ERROR: Exam {args.exam_id} not found")
        return 1

    # process each view
    view_results = []

    for _, row in exam_views.iterrows():
        try:
            path = args.raw / row["dicom_path"]
            ds = pydicom.dcmread(str(path))
            pixels = ds.pixel_array.astype(np.float32)

            lines = detect_vertical_line_simple(pixels, threshold=args.threshold)

            if len(lines) > 0:
                print(
                    f"  {row['laterality']}_{row['view']}: found {len(lines)} line(s)"
                )
                for start, end, center, z in lines:
                    print(f"    columns {start}-{end} (center={center}, z={z:.2f})")

                view_results.append(
                    {
                        "view": f"{row['laterality']}_{row['view']}",
                        "original": pixels,
                        "lines": lines,
                    }
                )
        except Exception as e:
            print(f"Error loading {row['dicom_path']}: {e}")

    if len(view_results) == 0:
        print("No artifacts detected in this exam")
        return 0

    # test all methods on each view
    print("\nTesting inpainting methods...")

    fig, axes = plt.subplots(len(view_results), 6, figsize=(30, 5 * len(view_results)))
    if len(view_results) == 1:
        axes = axes.reshape(1, -1)

    for view_idx, view_data in enumerate(view_results):
        original = view_data["original"]
        lines = view_data["lines"]

        # for simplicity, inpaint all detected lines
        # (in practice, might need to handle multiple lines differently)
        start, end, center, z = lines[0]  # use first line

        # original
        axes[view_idx, 0].imshow(original, cmap="gray")
        axes[view_idx, 0].set_title(
            f"{view_data['view']}\nOriginal (col {center}, z={z:.1f})"
        )
        axes[view_idx, 0].axis("off")

        # method 1: wide interpolation
        inpainted_1 = inpaint_method_1_wide_interpolation(original, start, end)
        axes[view_idx, 1].imshow(inpainted_1, cmap="gray")
        axes[view_idx, 1].set_title("Wide Interpolation\n(50px margins)")
        axes[view_idx, 1].axis("off")

        # method 2: median filter
        inpainted_2 = inpaint_method_2_median_filter(original, start, end)
        axes[view_idx, 2].imshow(inpainted_2, cmap="gray")
        axes[view_idx, 2].set_title("Median Filter\n(30px neighbors)")
        axes[view_idx, 2].axis("off")

        # method 3: morphological
        inpainted_3 = inpaint_method_3_morphological(original, start, end)
        axes[view_idx, 3].imshow(inpainted_3, cmap="gray")
        axes[view_idx, 3].set_title("Morphological\n(3x11 median)")
        axes[view_idx, 3].axis("off")

        # method 4: gaussian blur
        inpainted_4 = inpaint_method_4_gaussian_blur(original, start, end)
        axes[view_idx, 4].imshow(inpainted_4, cmap="gray")
        axes[view_idx, 4].set_title("Gaussian Blur\n(sigma=2,5)")
        axes[view_idx, 4].axis("off")

        # zoom in on artifact region
        row_center = original.shape[0] // 2
        row_window = 200
        col_window = 100
        zoom_rows = slice(row_center - row_window, row_center + row_window)
        zoom_cols = slice(
            max(0, center - col_window), min(original.shape[1], center + col_window)
        )

        axes[view_idx, 5].imshow(original[zoom_rows, zoom_cols], cmap="gray")
        axes[view_idx, 5].axvline(
            center - zoom_cols.start, color="red", linewidth=2, alpha=0.7
        )
        axes[view_idx, 5].set_title("Zoomed Original\n(red line = artifact)")
        axes[view_idx, 5].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison to: {args.output}")
    print()
    print("Review the figure and choose the best method:")
    print("  1. Wide Interpolation - uses 50px margins on each side")
    print("  2. Median Filter - pixel-by-pixel median from neighbors")
    print("  3. Morphological - median filter with tall kernel")
    print("  4. Gaussian Blur - smooth with edge-aware blending")
    print()
    print("Look for:")
    print("  ✓ Line is removed or significantly reduced")
    print("  ✓ No new artifacts introduced")
    print("  ✓ Breast tissue detail preserved")

    return 0


if __name__ == "__main__":
    exit(main())
