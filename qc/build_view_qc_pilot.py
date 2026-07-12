#!/usr/bin/env python3
"""Build a fresh blinded view-level vertical-line QC pilot."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from prima.dicom_source import (
    SOURCE_COLUMNS,
    require_source_columns,
    require_valid_sources,
)
from prima.view_render import render_source_rows
from prima.view_qc import empty_view_qc_state, save_view_qc_state

VIEW_ORDER = [("L", "CC"), ("R", "CC"), ("L", "MLO"), ("R", "MLO")]
VERTICAL_DETECTOR_SEAM_TARGET = "vertical detector seam"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score cached montage panels for enrichment, sample individual views, "
            "and render source-linked PNGs for blinded human review."
        )
    )
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--total-views", type=int, default=160)
    parser.add_argument("--heuristic-views", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--max-render-pixels", type=int, default=2_000_000)
    return parser.parse_args()


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0 or window % 2 == 0:
        raise ValueError("rolling-median window must be a positive odd integer")
    if values.size < window:
        return np.full_like(values, float(np.median(values)), dtype=np.float32)
    pad = window // 2
    padded = np.pad(values.astype(np.float32), pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(windows, axis=1).astype(np.float32)


def longest_true_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def score_panel(panel: Image.Image) -> dict[str, float | int]:
    """Return a cheap vertical-profile enrichment score for one montage panel."""
    array = np.asarray(panel.convert("L"), dtype=np.float32) / 255.0
    height, width = array.shape
    row0 = max(0, int(height * 0.10))
    row1 = min(height, int(height * 0.96))
    col0 = max(0, int(width * 0.06))
    col1 = min(width, int(width * 0.94))
    crop = array[row0:row1, col0:col1]
    if crop.size == 0 or crop.shape[1] < 48:
        return {
            "heuristic_score": 0.0,
            "max_column_z": 0.0,
            "max_run_width": 0,
        }

    profile = np.median(crop, axis=0).astype(np.float32)
    baseline_window = min(41, profile.size // 2 * 2 - 1)
    mad_window = min(81, profile.size // 2 * 2 - 1)
    if baseline_window < 3 or mad_window < 3:
        return {
            "heuristic_score": 0.0,
            "max_column_z": 0.0,
            "max_run_width": 0,
        }
    baseline = rolling_median(profile, baseline_window)
    residual = np.abs(profile - baseline)
    local_mad = rolling_median(residual, mad_window)
    global_mad = float(np.median(residual)) + 1e-6
    z_score = residual / np.maximum(local_mad, global_mad * 0.35)
    maximum = float(np.max(z_score))
    line_mask = z_score >= max(4.0, maximum * 0.60)
    run_width = longest_true_run(line_mask)
    narrowness = 1.0 / (1.0 + max(0, run_width - 3) / 5.0)
    peak = int(np.argmax(z_score))
    contrast = float(residual[peak])
    return {
        "heuristic_score": maximum * narrowness * (1.0 + min(contrast, 0.25)),
        "max_column_z": maximum,
        "max_run_width": int(run_width),
    }


def expected_montage_path(export_dir: Path, row: pd.Series) -> Path:
    return (
        export_dir
        / "success"
        / str(row["patient_id"])
        / str(row["accession_number"])
        / f"COMBINED_four_views_{row['exam_id']}.png"
    )


def score_cached_views(views: pd.DataFrame, export_dir: Path) -> pd.DataFrame:
    """Score each canonical view by its deterministic panel in the cached montage."""
    score_rows: list[dict[str, Any]] = []
    for _exam_id, exam_views in tqdm(
        views.groupby("exam_id", sort=True), desc="scoring cached view panels"
    ):
        slots = {
            (str(row["laterality"]), str(row["view"])): row
            for row in exam_views.to_dict("records")
        }
        if set(slots) != set(VIEW_ORDER) or len(exam_views) != 4:
            raise ValueError("view QC source must contain one exact canonical quad")
        first = exam_views.iloc[0]
        montage_path = expected_montage_path(export_dir, first)
        if not montage_path.is_file():
            raise FileNotFoundError("canonical view is missing its cached montage")
        with Image.open(montage_path) as montage:
            width, height = montage.size
            for panel_index, slot in enumerate(VIEW_ORDER):
                left = round(panel_index * width / 4)
                right = round((panel_index + 1) * width / 4)
                panel = montage.crop((left, 0, right, height))
                score_rows.append({**slots[slot], **score_panel(panel)})
    scores = pd.DataFrame(score_rows)
    if len(scores) != len(views):
        raise RuntimeError("cached panel scoring did not preserve every source view")
    return scores


def sample_views(
    scores: pd.DataFrame, total_views: int, heuristic_views: int, seed: int
) -> pd.DataFrame:
    """Sample high-score and random views with at most one view per exam."""
    if total_views <= 0 or heuristic_views <= 0:
        raise ValueError("view counts must be positive")
    if heuristic_views >= total_views:
        raise ValueError("--heuristic-views must be smaller than --total-views")
    if total_views > scores["exam_id"].nunique():
        raise ValueError("not enough independent exams for the requested pilot")

    heuristic_rows: list[dict[str, Any]] = []
    used_exams: set[str] = set()
    sorted_scores = scores.sort_values(
        ["heuristic_score", "max_column_z", "sha256"],
        ascending=[False, False, True],
        kind="stable",
    )
    for row in sorted_scores.to_dict("records"):
        exam_id = str(row["exam_id"])
        if exam_id in used_exams:
            continue
        heuristic_rows.append({**row, "stratum": "heuristic_top"})
        used_exams.add(exam_id)
        if len(heuristic_rows) == heuristic_views:
            break
    if len(heuristic_rows) != heuristic_views:
        raise RuntimeError("could not construct the heuristic view stratum")

    rng = random.Random(seed)
    random_pool = [
        row
        for row in scores.to_dict("records")
        if str(row["exam_id"]) not in used_exams
    ]
    rng.shuffle(random_pool)
    random_rows: list[dict[str, Any]] = []
    for row in random_pool:
        exam_id = str(row["exam_id"])
        if exam_id in used_exams:
            continue
        random_rows.append({**row, "stratum": "random"})
        used_exams.add(exam_id)
        if len(random_rows) == total_views - heuristic_views:
            break
    if len(heuristic_rows) + len(random_rows) != total_views:
        raise RuntimeError("could not construct the random view stratum")

    selected = heuristic_rows + random_rows
    rng.shuffle(selected)
    for review_order, row in enumerate(selected, start=1):
        row["review_order"] = review_order
        row["view_id"] = str(row["sha256"]).lower()
        row["image_path"] = f"images/{row['view_id']}.png"
    return pd.DataFrame(selected)


def render_selected_views(
    manifest: pd.DataFrame,
    raw_root: Path,
    out_dir: Path,
    max_pixels: int,
    *,
    overwrite: bool = False,
) -> None:
    (out_dir / "images").mkdir(parents=True, exist_ok=overwrite)
    render_source_rows(
        manifest,
        raw_root=raw_root,
        out_dir=out_dir,
        max_pixels=max_pixels,
        resume=overwrite,
    )


def write_provenance(
    out_dir: Path,
    *,
    views_path: Path,
    export_dir: Path,
    total_views: int,
    heuristic_views: int,
    seed: int,
    max_render_pixels: int,
) -> None:
    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# View-level vertical-line QC pilot",
                "",
                f"- target: `{VERTICAL_DETECTOR_SEAM_TARGET}`",
                f"- authoritative selected views: `{views_path}`",
                f"- cached montage export: `{export_dir}`",
                f"- total views: `{total_views}`",
                f"- heuristic-enriched views: `{heuristic_views}`",
                f"- random views: `{total_views - heuristic_views}`",
                f"- seed: `{seed}`",
                f"- maximum rendered pixels: `{max_render_pixels}`",
                "- one sampled view per exam",
                "- labels start empty and are not copied from exam-level review",
                "- manifest order is randomized and the gallery does not expose strata",
                "",
                "The abandoned exam-level state is historical only and is not converted",
                "into view labels.",
                "",
                "## Decision rule",
                "",
                "Continue view-level model development only if sensitivity and",
                "specificity are each at least 0.90 and disagreement review does not",
                "show a repeated missed morphology. This pilot never authorizes an",
                "automatic deployment-grade target-present/target-absent gate.",
                "",
                "Review command:",
                "",
                "```bash",
                "micromamba run -p /gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima \\",
                "  python qc/view_qc_gallery.py \\",
                "  --manifest qc_redo/review_batches/vertical_line_view_review/manifest.parquet \\",
                "  --state qc_redo/review_batches/vertical_line_view_review/view_qc_state.json \\",
                "  --port 8767",
                "```",
                "",
            ]
        )
    )


def main() -> int:
    args = parse_args()
    views_path = args.views.resolve()
    export_dir = args.export_dir.resolve()
    raw_root = args.raw_root.resolve()
    out_dir = args.out_dir.resolve()
    if not views_path.is_file():
        raise FileNotFoundError(f"views parquet not found: {views_path}")
    if not export_dir.is_dir():
        raise FileNotFoundError(f"QC export directory not found: {export_dir}")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw DICOM root not found: {raw_root}")
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite view QC batch: {out_dir}")
    out_dir.mkdir(parents=True, mode=0o700)

    views = pd.read_parquet(views_path)
    require_source_columns(views.columns, str(views_path))
    require_valid_sources(
        views[list(SOURCE_COLUMNS)].to_dict("records"), str(views_path)
    )
    required = {
        "patient_id",
        "exam_id",
        "accession_number",
        "laterality",
        "view",
        "sha256",
    }
    missing = sorted(required - set(views.columns))
    if missing:
        raise ValueError(f"views parquet is missing columns: {', '.join(missing)}")
    if views["sha256"].duplicated().any():
        raise ValueError("selected views contain duplicate SHA-256 identifiers")

    scores = score_cached_views(views, export_dir)
    manifest = sample_views(
        scores,
        total_views=args.total_views,
        heuristic_views=args.heuristic_views,
        seed=args.seed,
    )
    render_selected_views(
        manifest,
        raw_root=raw_root,
        out_dir=out_dir,
        max_pixels=args.max_render_pixels,
    )

    scores_path = out_dir / "heuristic_scores.parquet"
    manifest_path = out_dir / "manifest.parquet"
    state_path = out_dir / "view_qc_state.json"
    scores.to_parquet(scores_path, index=False)
    manifest.to_parquet(manifest_path, index=False)
    os.chmod(scores_path, 0o600)
    os.chmod(manifest_path, 0o600)
    save_view_qc_state(state_path, empty_view_qc_state(VERTICAL_DETECTOR_SEAM_TARGET))
    write_provenance(
        out_dir,
        views_path=views_path,
        export_dir=export_dir,
        total_views=args.total_views,
        heuristic_views=args.heuristic_views,
        seed=args.seed,
        max_render_pixels=args.max_render_pixels,
    )
    print(
        f"view QC pilot ready: total={len(manifest)} "
        f"heuristic={sum(manifest['stratum'] == 'heuristic_top')} "
        f"random={sum(manifest['stratum'] == 'random')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
