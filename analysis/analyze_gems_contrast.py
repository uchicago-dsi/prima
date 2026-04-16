#!/usr/bin/env python3
"""analyze_gems_contrast.py

Compare GEMS vs non-GEMS image statistics to determine if preprocessing adjustments
could fix contrast differences.

Usage:
    python analyze_gems_contrast.py --n-samples 100
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm


def read_dicom_pixels(path: Path) -> np.ndarray:
    """read DICOM and return pixel array"""
    ds = pydicom.dcmread(str(path), force=True)
    return ds.pixel_array.astype(np.float32)


def compute_stats(pixel_arrays):
    """compute aggregate statistics across multiple images"""
    all_pixels = np.concatenate([arr.ravel() for arr in pixel_arrays])
    return {
        "mean": np.mean(all_pixels),
        "std": np.std(all_pixels),
        "median": np.median(all_pixels),
        "p01": np.percentile(all_pixels, 1),
        "p99": np.percentile(all_pixels, 99),
        "min": np.min(all_pixels),
        "max": np.max(all_pixels),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GEMS vs non-GEMS image contrast differences"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="number of images to sample from each group (default: 100)",
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
        default=Path("plots/gems_contrast_analysis.png"),
        help="output plot path",
    )
    args = parser.parse_args()

    print("Loading views and tags...")
    views = pd.read_parquet(args.raw / "sot" / "views.parquet")
    tags = pd.read_parquet(args.raw / "sot" / "dicom_tags.parquet")

    # merge to get processing codes
    merged = views.merge(tags, on="sop_instance_uid", how="left")

    # separate GEMS and non-GEMS
    if "AcquisitionDeviceProcessingCode" not in merged.columns:
        print("ERROR: AcquisitionDeviceProcessingCode column not found in tags")
        return 1

    gems_mask = merged["AcquisitionDeviceProcessingCode"].str.startswith(
        "GEMS_", na=False
    )
    gems_views = merged[gems_mask]
    non_gems_views = merged[~gems_mask & merged["for_presentation"]]

    print(f"GEMS views: {len(gems_views):,}")
    print(f"Non-GEMS views: {len(non_gems_views):,}")

    # sample views
    n_samples = min(args.n_samples, len(gems_views), len(non_gems_views))
    gems_sample = gems_views.sample(n=n_samples, random_state=42)
    non_gems_sample = non_gems_views.sample(n=n_samples, random_state=42)

    print(f"\nSampling {n_samples} images from each group...")

    # load pixel data
    print("Loading GEMS images...")
    gems_pixels = []
    for _, row in tqdm(gems_sample.iterrows(), total=len(gems_sample)):
        path = args.raw / row["dicom_path"]
        try:
            pixels = read_dicom_pixels(path)
            gems_pixels.append(pixels)
        except Exception as e:
            print(f"  Failed to load {path}: {e}")

    print("Loading non-GEMS images...")
    non_gems_pixels = []
    for _, row in tqdm(non_gems_sample.iterrows(), total=len(non_gems_sample)):
        path = args.raw / row["dicom_path"]
        try:
            pixels = read_dicom_pixels(path)
            non_gems_pixels.append(pixels)
        except Exception as e:
            print(f"  Failed to load {path}: {e}")

    print("\nSuccessfully loaded:")
    print(f"  GEMS: {len(gems_pixels)} images")
    print(f"  Non-GEMS: {len(non_gems_pixels)} images")

    # compute statistics
    print("\nComputing statistics...")
    gems_stats = compute_stats(gems_pixels)
    non_gems_stats = compute_stats(non_gems_pixels)

    print("\n" + "=" * 60)
    print("STATISTICS COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<15} {'GEMS':>12} {'Non-GEMS':>12} {'Ratio':>10}")
    print("-" * 60)
    for key in ["mean", "std", "median", "p01", "p99", "min", "max"]:
        gems_val = gems_stats[key]
        non_gems_val = non_gems_stats[key]
        ratio = gems_val / non_gems_val if non_gems_val != 0 else float("inf")
        print(f"{key:<15} {gems_val:>12.2f} {non_gems_val:>12.2f} {ratio:>10.3f}")

    # check Mirai normalization
    MIRAI_MEAN = 7047.99
    MIRAI_STD = 12005.5

    print("\n" + "=" * 60)
    print("MIRAI NORMALIZATION COMPARISON")
    print("=" * 60)
    print(f"Current Mirai constants: mean={MIRAI_MEAN:.2f}, std={MIRAI_STD:.2f}")
    print()
    print("GEMS deviation from Mirai:")
    print(
        f"  Mean offset: {gems_stats['mean'] - MIRAI_MEAN:.2f} ({(gems_stats['mean'] - MIRAI_MEAN) / MIRAI_MEAN * 100:.1f}%)"
    )
    print(
        f"  Std ratio: {gems_stats['std'] / MIRAI_STD:.3f} ({(gems_stats['std'] / MIRAI_STD - 1) * 100:.1f}% difference)"
    )
    print()
    print("Non-GEMS deviation from Mirai:")
    print(
        f"  Mean offset: {non_gems_stats['mean'] - MIRAI_MEAN:.2f} ({(non_gems_stats['mean'] - MIRAI_MEAN) / MIRAI_MEAN * 100:.1f}%)"
    )
    print(
        f"  Std ratio: {non_gems_stats['std'] / MIRAI_STD:.3f} ({(non_gems_stats['std'] / MIRAI_STD - 1) * 100:.1f}% difference)"
    )

    # suggest GEMS-specific constants if difference is large
    mean_diff_pct = (
        abs(gems_stats["mean"] - non_gems_stats["mean"]) / non_gems_stats["mean"] * 100
    )
    std_diff_pct = (
        abs(gems_stats["std"] - non_gems_stats["std"]) / non_gems_stats["std"] * 100
    )

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    if mean_diff_pct > 10 or std_diff_pct > 10:
        print(
            f"⚠️  LARGE DIFFERENCE: mean differs by {mean_diff_pct:.1f}%, std by {std_diff_pct:.1f}%"
        )
        print()
        print("Suggested GEMS-specific normalization constants:")
        print(f"  GEMS_IMG_MEAN = {gems_stats['mean']:.2f}")
        print(f"  GEMS_IMG_STD = {gems_stats['std']:.2f}")
        print()
        print(
            "To use: modify pipelines/preprocess.py to detect GEMS and apply different normalization"
        )
    else:
        print(
            f"✓ Small difference: mean differs by {mean_diff_pct:.1f}%, std by {std_diff_pct:.1f}%"
        )
        print("Current Mirai normalization should work for GEMS.")
        print("Performance issues likely due to other factors (not just contrast).")

    # create visualization
    print(f"\nGenerating plots to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # plot 1: histogram comparison
    ax = axes[0, 0]
    gems_flat = np.concatenate([arr.ravel() for arr in gems_pixels[:10]])
    non_gems_flat = np.concatenate([arr.ravel() for arr in non_gems_pixels[:10]])
    ax.hist(
        gems_flat,
        bins=100,
        alpha=0.5,
        label="GEMS",
        density=True,
        range=(0, 30000),
    )
    ax.hist(
        non_gems_flat,
        bins=100,
        alpha=0.5,
        label="Non-GEMS",
        density=True,
        range=(0, 30000),
    )
    ax.axvline(MIRAI_MEAN, color="red", linestyle="--", label="Mirai mean", alpha=0.7)
    ax.set_xlabel("pixel value")
    ax.set_ylabel("density")
    ax.set_title("raw pixel distributions (10 images each)")
    ax.legend()

    # plot 2: example GEMS image
    ax = axes[0, 1]
    ax.imshow(gems_pixels[0], cmap="gray")
    ax.set_title(f"GEMS example\nmean={gems_pixels[0].mean():.0f}")
    ax.axis("off")

    # plot 3: example non-GEMS image
    ax = axes[0, 2]
    ax.imshow(non_gems_pixels[0], cmap="gray")
    ax.set_title(f"non-GEMS example\nmean={non_gems_pixels[0].mean():.0f}")
    ax.axis("off")

    # plot 4: mean comparison
    ax = axes[1, 0]
    gems_means = [arr.mean() for arr in gems_pixels]
    non_gems_means = [arr.mean() for arr in non_gems_pixels]
    ax.boxplot(
        [gems_means, non_gems_means],
        labels=["GEMS", "Non-GEMS"],
        showmeans=True,
    )
    ax.axhline(MIRAI_MEAN, color="red", linestyle="--", label="Mirai mean", alpha=0.7)
    ax.set_ylabel("mean pixel value")
    ax.set_title("mean pixel values per image")
    ax.legend()

    # plot 5: std comparison
    ax = axes[1, 1]
    gems_stds = [arr.std() for arr in gems_pixels]
    non_gems_stds = [arr.std() for arr in non_gems_pixels]
    ax.boxplot(
        [gems_stds, non_gems_stds],
        labels=["GEMS", "Non-GEMS"],
        showmeans=True,
    )
    ax.axhline(MIRAI_STD, color="red", linestyle="--", label="Mirai std", alpha=0.7)
    ax.set_ylabel("std pixel value")
    ax.set_title("std pixel values per image")
    ax.legend()

    # plot 6: processing code breakdown
    ax = axes[1, 2]
    gems_codes = gems_views["AcquisitionDeviceProcessingCode"].value_counts()
    ax.barh(range(len(gems_codes)), gems_codes.values)
    ax.set_yticks(range(len(gems_codes)))
    ax.set_yticklabels(gems_codes.index)
    ax.set_xlabel("number of views")
    ax.set_title("GEMS processing codes")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
