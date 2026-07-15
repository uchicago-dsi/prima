#!/usr/bin/env python3
"""Build blinded original/180-degree single-image MLO landmark grids."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

import pandas as pd

from prima.view_few_shot import sha256_file
from prima.view_landmark_grid import (
    LANDMARK_GRID_BANDS,
    LANDMARK_GRID_CANVAS_SIZE,
    LANDMARK_GRID_IMAGE_BOUNDS,
    render_landmark_grid_png,
    resolve_relative_image,
)
from prima.view_qc import normalize_view_id, validate_view_manifest_columns

REPRESENTATION_VERSION = "mlo-landmark-grid-v1"
ROTATIONS_DEGREES_CLOCKWISE = (0, 180)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--review-order",
        type=int,
        action="append",
        required=True,
        help="source review order to include; repeat for each MLO view",
    )
    parser.add_argument("--max-source-pixels", type=int, default=2_097_152)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")
    os.chmod(path, 0o600)


def _synthetic_view_id(source_view_id: str, rotation: int) -> str:
    identity = (
        f"{source_view_id}|{REPRESENTATION_VERSION}|"
        f"rotation_degrees_clockwise={rotation}"
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def run_from_args(args: argparse.Namespace) -> dict[str, object]:
    manifest_path = args.manifest.resolve()
    output_manifest_path = args.output_manifest.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"view manifest not found: {manifest_path}")
    if output_manifest_path.parent != manifest_path.parent:
        raise ValueError("landmark-grid output manifest must remain beside its source")
    if args.max_source_pixels <= 0:
        raise ValueError("--max-source-pixels must be positive")
    review_orders = [int(value) for value in args.review_order]
    if any(value <= 0 for value in review_orders):
        raise ValueError("--review-order values must be positive")
    if len(review_orders) != len(set(review_orders)):
        raise ValueError("--review-order values must be unique")

    asset_dir = output_manifest_path.with_suffix("")
    image_dir = asset_dir / "images"
    provenance_path = output_manifest_path.with_suffix(".provenance.json")
    readme_path = output_manifest_path.with_suffix(".README.md")
    for path in (output_manifest_path, asset_dir, provenance_path, readme_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite landmark-grid output: {path}")

    manifest = pd.read_parquet(manifest_path)
    validate_view_manifest_columns(manifest.columns, str(manifest_path))
    manifest = manifest.sort_values("review_order", kind="stable").copy()
    manifest["view_id"] = manifest["view_id"].map(normalize_view_id)
    if manifest.empty:
        raise ValueError("landmark-grid source manifest is empty")
    if manifest["view_id"].duplicated().any():
        raise ValueError("landmark-grid source manifest contains duplicate view IDs")
    if manifest["review_order"].duplicated().any():
        raise ValueError(
            "landmark-grid source manifest contains duplicate review_order"
        )
    missing_orders = sorted(set(review_orders) - set(manifest["review_order"]))
    if missing_orders:
        raise ValueError("--review-order is outside the source manifest")
    selected = manifest[manifest["review_order"].isin(review_orders)].copy()
    selected = selected.sort_values("review_order", kind="stable")
    if len(selected) != len(review_orders):
        raise ValueError("landmark-grid source selection is incomplete")
    if not selected["view"].eq("MLO").all():
        raise ValueError("landmark-grid source selection must contain only MLO views")

    asset_dir.mkdir(mode=0o700)
    image_dir.mkdir(mode=0o700)
    root = manifest_path.parent.resolve()
    records: list[dict[str, object]] = []
    ordered_image_digests: list[str] = []
    output_order = 0
    for row in selected.to_dict("records"):
        source_view_id = normalize_view_id(row["view_id"])
        source_path, canonical_relative = resolve_relative_image(
            root, row["image_path"], description="canonical image_path"
        )
        for rotation in ROTATIONS_DEGREES_CLOCKWISE:
            output_order += 1
            view_id = _synthetic_view_id(source_view_id, rotation)
            output_path = image_dir / f"{view_id}.png"
            render_landmark_grid_png(
                source_path,
                output_path,
                rotation_degrees_clockwise=rotation,
                max_source_pixels=args.max_source_pixels,
            )
            records.append(
                {
                    "view_id": view_id,
                    "image_path": canonical_relative,
                    "laterality": row["laterality"],
                    "view": row["view"],
                    "review_order": output_order,
                    "source_view_id": source_view_id,
                    "source_review_order": int(row["review_order"]),
                    "rotation_degrees_clockwise": rotation,
                    "model_image_path": output_path.relative_to(root).as_posix(),
                }
            )
            ordered_image_digests.append(sha256_file(output_path))

    output = pd.DataFrame.from_records(records)
    if output["view_id"].duplicated().any():
        raise ValueError("landmark-grid synthetic view IDs are not unique")
    output.to_parquet(output_manifest_path, index=False)
    os.chmod(output_manifest_path, 0o600)

    bank_digest = hashlib.sha256("".join(ordered_image_digests).encode()).hexdigest()
    command = shlex.join([sys.executable, *sys.argv])
    provenance = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "representation_version": REPRESENTATION_VERSION,
        "source_manifest_sha256": sha256_file(manifest_path),
        "output_manifest_sha256": sha256_file(output_manifest_path),
        "ordered_image_bank_sha256": bank_digest,
        "rows": len(output),
        "source_review_orders": sorted(review_orders),
        "rotations_degrees_clockwise": list(ROTATIONS_DEGREES_CLOCKWISE),
        "synthetic_view_id_formula": (
            "sha256(source_view_id|mlo-landmark-grid-v1|"
            "rotation_degrees_clockwise=DEGREES)"
        ),
        "canonical_image_column": "image_path",
        "model_image_column": "model_image_path",
        "source_view_id_column": "source_view_id",
        "source_review_order_column": "source_review_order",
        "canvas_size": list(LANDMARK_GRID_CANVAS_SIZE),
        "image_bounds": list(LANDMARK_GRID_IMAGE_BOUNDS),
        "horizontal_bands": LANDMARK_GRID_BANDS,
        "band_coordinates": "1=top through 4=bottom in fixed display coordinates",
        "anatomy_crop": "substantial_foreground_box before rotation",
        "task_semantics": "neutral anatomy localization; no QC or orientation label",
    }
    _write_json(provenance_path, provenance)
    readme_path.write_text(
        "\n".join(
            [
                "# Single-image MLO landmark grids",
                "",
                f"- rows: `{len(output)}`",
                f"- source review orders: `{sorted(review_orders)}`",
                "- variants per source: `0 and 180 degrees clockwise`",
                "- canonical target column: `image_path`",
                "- model input column: `model_image_path`",
                f"- canvas: `{LANDMARK_GRID_CANVAS_SIZE[0]}x{LANDMARK_GRID_CANVAS_SIZE[1]}`",
                f"- ordered image-bank SHA-256: `{bank_digest}`",
                "",
                "Each source MLO is shown alone under four fixed horizontal",
                "coordinate bands. The original and 180-degree variants receive",
                "opaque synthetic view IDs while retaining source-view and",
                "canonical-image lineage. The panel does not expose the variant",
                "rotation or any QC/orientation label to the model.",
                "",
                f"Exact producer command: `{command}`",
                "",
            ]
        )
    )
    os.chmod(readme_path, 0o600)
    return {"rows": len(output), "bank_digest": bank_digest}


def main() -> int:
    result = run_from_args(parse_args())
    print(
        "landmark-grid inputs ready: "
        f"rows={result['rows']} bank_sha256={result['bank_digest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
