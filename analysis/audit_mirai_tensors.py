from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "vendor" / "mirai"))

from onconet.transformers.basic import ToTensor  # noqa: E402
from onconet.transformers.image import Align_To_Left, Scale_2d  # noqa: E402
from onconet.transformers.tensor import (  # noqa: E402
    Force_Num_Chan_Tensor_2d,
    Normalize_Tensor_2d,
)
from onconet.utils.zarr_image_loader import open_image_mono16_any  # noqa: E402


def load_meta(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    key_cols = ["patient_id", "exam_id", "laterality", "view", "file_path"]
    missing = set(key_cols) - set(df.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {sorted(missing)}")
    return df[key_cols].copy()


def make_transformers() -> tuple[
    Scale_2d, Align_To_Left, ToTensor, Force_Num_Chan_Tensor_2d, Normalize_Tensor_2d
]:
    args = SimpleNamespace(
        img_size=(1664, 2048),
        img_mean=[7047.99],
        img_std=[12005.5],
        num_chan=3,
    )
    return (
        Scale_2d(args, {}),
        Align_To_Left(args, {}),
        ToTensor(),
        Force_Num_Chan_Tensor_2d(args, {}),
        Normalize_Tensor_2d(args, {}),
    )


TRANSFORMS = make_transformers()


def apply_current_pipeline(
    image: Image.Image,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, str, str]:
    scale, align, to_tensor, force_chan, normalize = TRANSFORMS
    scaled = scale(image)
    aligned = align(scaled, {})
    raw_tensor = to_tensor(aligned)
    tensor = force_chan(raw_tensor)
    tensor = normalize(tensor)
    arr = np.array(aligned, dtype=np.uint16)
    return arr, raw_tensor, tensor, scaled.mode, aligned.mode


def corr(a: np.ndarray, b: np.ndarray, max_points: int = 4096) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if len(a) > max_points:
        idx = np.linspace(0, len(a) - 1, num=max_points, dtype=int)
        a = a[idx]
        b = b[idx]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def summarize_pair(
    patient_id: str,
    exam_id: str,
    laterality: str,
    view: str,
    zarr_path: str,
    png_path: str,
) -> dict:
    zarr_img = open_image_mono16_any(zarr_path)
    png_img = open_image_mono16_any(png_path)

    zarr_raw = np.array(zarr_img, dtype=np.uint16)
    png_raw = np.array(png_img, dtype=np.uint16)
    raw_same_shape = zarr_raw.shape == png_raw.shape
    if raw_same_shape:
        raw_diff = zarr_raw.astype(np.int64) - png_raw.astype(np.int64)
        raw_mae = float(np.abs(raw_diff).mean())
        raw_max_abs = int(np.abs(raw_diff).max())
        raw_equal_fraction = float((raw_diff == 0).mean())
        raw_corr = corr(zarr_raw, png_raw)
    else:
        raw_mae = float("nan")
        raw_max_abs = -1
        raw_equal_fraction = float("nan")
        raw_corr = float("nan")

    zarr_post, zarr_raw_tensor, zarr_tensor, zarr_scaled_mode, zarr_aligned_mode = (
        apply_current_pipeline(zarr_img)
    )
    png_post, png_raw_tensor, png_tensor, png_scaled_mode, png_aligned_mode = (
        apply_current_pipeline(png_img)
    )
    post_diff = zarr_post.astype(np.int64) - png_post.astype(np.int64)
    raw_tensor_diff = (zarr_raw_tensor - png_raw_tensor).cpu().numpy()
    tensor_diff = (zarr_tensor - png_tensor).cpu().numpy()

    return {
        "patient_id": patient_id,
        "exam_id": exam_id,
        "laterality": laterality,
        "view": view,
        "zarr_open_mode": zarr_img.mode,
        "png_open_mode": png_img.mode,
        "zarr_scaled_mode": zarr_scaled_mode,
        "png_scaled_mode": png_scaled_mode,
        "zarr_aligned_mode": zarr_aligned_mode,
        "png_aligned_mode": png_aligned_mode,
        "raw_same_shape": raw_same_shape,
        "zarr_raw_shape": list(zarr_raw.shape),
        "png_raw_shape": list(png_raw.shape),
        "raw_mae": raw_mae,
        "raw_max_abs": raw_max_abs,
        "raw_equal_fraction": raw_equal_fraction,
        "raw_corr": raw_corr,
        "post_mae": float(np.abs(post_diff).mean()),
        "post_max_abs": int(np.abs(post_diff).max()),
        "post_equal_fraction": float((post_diff == 0).mean()),
        "post_corr": corr(zarr_post, png_post),
        "raw_tensor_mae": float(np.abs(raw_tensor_diff).mean()),
        "raw_tensor_max_abs": float(np.abs(raw_tensor_diff).max()),
        "raw_tensor_equal_fraction": float((raw_tensor_diff == 0).mean()),
        "raw_tensor_corr": corr(
            zarr_raw_tensor.cpu().numpy(), png_raw_tensor.cpu().numpy()
        ),
        "tensor_mae": float(np.abs(tensor_diff).mean()),
        "tensor_max_abs": float(np.abs(tensor_diff).max()),
        "tensor_equal_fraction": float((tensor_diff == 0).mean()),
        "tensor_corr": corr(zarr_tensor.cpu().numpy(), png_tensor.cpu().numpy()),
        "zarr_raw_tensor_min": float(zarr_raw_tensor.min()),
        "zarr_raw_tensor_max": float(zarr_raw_tensor.max()),
        "png_raw_tensor_min": float(png_raw_tensor.min()),
        "png_raw_tensor_max": float(png_raw_tensor.max()),
        "zarr_raw_min": int(zarr_raw.min()),
        "zarr_raw_max": int(zarr_raw.max()),
        "png_raw_min": int(png_raw.min()),
        "png_raw_max": int(png_raw.max()),
    }


def write_examples(df: pd.DataFrame, out_dir: Path) -> None:
    top = df.sort_values(["tensor_mae", "raw_mae"], ascending=False).head(12)
    top.to_csv(out_dir / "worst_view_pairs.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-meta", required=True, type=Path)
    parser.add_argument("--legacy-meta", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    current = load_meta(args.current_meta).rename(
        columns={"file_path": "current_file_path"}
    )
    legacy = load_meta(args.legacy_meta).rename(
        columns={"file_path": "legacy_file_path"}
    )

    merged = current.merge(
        legacy,
        on=["patient_id", "exam_id", "laterality", "view"],
        how="inner",
        validate="one_to_one",
    )
    if args.limit is not None:
        merged = merged.head(args.limit)

    rows = [
        summarize_pair(
            row.patient_id,
            row.exam_id,
            row.laterality,
            row.view,
            row.current_file_path,
            row.legacy_file_path,
        )
        for row in merged.itertuples(index=False)
    ]
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "view_tensor_comparison.csv", index=False)
    write_examples(detail, out_dir)

    summary = {
        "n_matched_views": int(len(detail)),
        "n_unique_exams": int(
            detail[["patient_id", "exam_id"]].drop_duplicates().shape[0]
        ),
        "overall": {
            "raw_mae_mean": float(detail["raw_mae"].mean()),
            "raw_equal_fraction_mean": float(detail["raw_equal_fraction"].mean()),
            "raw_corr_mean": float(detail["raw_corr"].mean()),
            "post_mae_mean": float(detail["post_mae"].mean()),
            "post_equal_fraction_mean": float(detail["post_equal_fraction"].mean()),
            "post_corr_mean": float(detail["post_corr"].mean()),
            "raw_tensor_mae_mean": float(detail["raw_tensor_mae"].mean()),
            "raw_tensor_equal_fraction_mean": float(
                detail["raw_tensor_equal_fraction"].mean()
            ),
            "raw_tensor_corr_mean": float(detail["raw_tensor_corr"].mean()),
            "tensor_mae_mean": float(detail["tensor_mae"].mean()),
            "tensor_equal_fraction_mean": float(detail["tensor_equal_fraction"].mean()),
            "tensor_corr_mean": float(detail["tensor_corr"].mean()),
        },
        "mode_counts": detail[
            [
                "zarr_open_mode",
                "png_open_mode",
                "zarr_scaled_mode",
                "png_scaled_mode",
                "zarr_aligned_mode",
                "png_aligned_mode",
            ]
        ]
        .value_counts()
        .reset_index(name="count")
        .to_dict(orient="records"),
        "by_view": detail.groupby(["laterality", "view"])[
            [
                "raw_mae",
                "raw_equal_fraction",
                "raw_corr",
                "post_mae",
                "post_equal_fraction",
                "post_corr",
                "raw_tensor_mae",
                "raw_tensor_equal_fraction",
                "raw_tensor_corr",
                "tensor_mae",
                "tensor_equal_fraction",
                "tensor_corr",
            ]
        ]
        .mean()
        .reset_index()
        .to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(out_dir / "summary.json")
    print(out_dir / "view_tensor_comparison.csv")


if __name__ == "__main__":
    main()
