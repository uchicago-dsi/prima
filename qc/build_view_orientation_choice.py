#!/usr/bin/env python3
"""Build equal-scale orientation comparison panels for view-level auto-QC."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from prima.view_few_shot import sha256_file
from prima.view_multiscale import substantial_foreground_box
from prima.view_qc import (
    normalize_view_id,
    validate_rendered_view_png,
    validate_view_manifest_columns,
)

FONT_PATH = Path("/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf")
FOUR_WAY_CANVAS_SIZE = (1440, 1440)
PAIRWISE_CANVAS_SIZE = (1440, 720)
FOUR_WAY_PANEL_BOUNDS = (
    (12, 78, 716, 756),
    (724, 78, 1428, 756),
    (12, 764, 716, 1432),
    (724, 764, 1428, 1432),
)
PAIRWISE_PANEL_BOUNDS = (
    (12, 78, 716, 712),
    (724, 78, 1428, 712),
)
ORIENTATIONS = (
    ("A", "CURRENT DISPLAY", None),
    ("B", "90 DEGREES CLOCKWISE", Image.ROTATE_270),
    ("C", "180 DEGREES", Image.ROTATE_180),
    ("D", "90 DEGREES COUNTERCLOCKWISE", Image.ROTATE_90),
)
TRANSPOSE_BY_CLOCKWISE_DEGREES = {
    90: Image.ROTATE_270,
    180: Image.ROTATE_180,
    270: Image.ROTATE_90,
}


def parse_current_rotation(value: str) -> tuple[int, int]:
    """Parse REVIEW_ORDER=DEGREES for a synthetic current display."""
    raw_order, separator, raw_degrees = str(value).partition("=")
    if not separator:
        raise argparse.ArgumentTypeError(
            "--current-rotation must use REVIEW_ORDER=DEGREES"
        )
    try:
        review_order = int(raw_order)
        degrees = int(raw_degrees)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "current rotation order and degrees must be integers"
        ) from error
    if review_order <= 0 or degrees not in TRANSPOSE_BY_CLOCKWISE_DEGREES:
        raise argparse.ArgumentTypeError(
            "current rotation requires a positive order and 90, 180, or 270 degrees"
        )
    return review_order, degrees


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--input-image-column",
        default="image_path",
        help="manifest image column to rotate; image_path remains canonical",
    )
    parser.add_argument(
        "--comparison-rotation-degrees-clockwise",
        type=int,
        choices=sorted(TRANSPOSE_BY_CLOCKWISE_DEGREES),
        default=None,
        help=(
            "render only A=current and B at this rotation using equal-scale "
            "square image boxes; omit for the four-way panel"
        ),
    )
    parser.add_argument(
        "--current-rotation",
        type=parse_current_rotation,
        action="append",
        default=[],
        help=(
            "repeat REVIEW_ORDER=DEGREES to rotate candidate A before building "
            "its comparison candidates"
        ),
    )
    parser.add_argument("--max-source-pixels", type=int, default=2_097_152)
    return parser.parse_args()


def _font(size: int) -> ImageFont.FreeTypeFont:
    if not FONT_PATH.is_file():
        raise FileNotFoundError(f"orientation-choice font not found: {FONT_PATH}")
    return ImageFont.truetype(str(FONT_PATH), size=size)


def _resolve_relative_image(
    root: Path, raw_path: object, *, description: str
) -> tuple[Path, str]:
    relative = Path(str(raw_path))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{description} must be a safe relative path")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{description} escapes the manifest root") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found")
    return resolved, relative.as_posix()


def _draw_panel(
    canvas: Image.Image,
    *,
    image: Image.Image,
    bounds: tuple[int, int, int, int],
    label: str,
    is_current: bool,
) -> None:
    left, top, right, bottom = bounds
    label_height = 42
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(bounds, outline=255 if is_current else 150, width=4)
    draw.text((left + 10, top + 6), label, fill=255, font=_font(23))
    available_width = right - left - 12
    available_height = bottom - top - label_height - 6
    side = min(available_width, available_height)
    image_left = left + (right - left - side) // 2
    image_top = top + label_height + (available_height - side) // 2
    image_box = (image_left, image_top, image_left + side, image_top + side)
    box_size = (side, side)
    contained = ImageOps.contain(image, box_size, Image.LANCZOS)
    x = image_box[0] + (box_size[0] - contained.width) // 2
    y = image_box[1] + (box_size[1] - contained.height) // 2
    canvas.paste(contained, (x, y))


def render_orientation_choice_png(
    source_path: Path,
    output_path: Path,
    *,
    declared_slot: str,
    max_source_pixels: int,
    comparison_rotation_degrees_clockwise: int | None,
    current_rotation_degrees_clockwise: int = 0,
) -> None:
    """Render current display beside deterministic orientation candidates."""
    if current_rotation_degrees_clockwise not in {0, *TRANSPOSE_BY_CLOCKWISE_DEGREES}:
        raise ValueError("current-display rotation must be 0, 90, 180, or 270")
    validate_rendered_view_png(source_path, max_pixels=max_source_pixels)
    with Image.open(source_path) as source:
        source.load()
        grayscale = source.convert("L")
    anatomy = grayscale.crop(substantial_foreground_box(grayscale))
    if anatomy.width < 2 or anatomy.height < 2:
        raise ValueError("orientation-choice anatomy crop is empty")
    if current_rotation_degrees_clockwise:
        anatomy = anatomy.transpose(
            TRANSPOSE_BY_CLOCKWISE_DEGREES[current_rotation_degrees_clockwise]
        )

    if comparison_rotation_degrees_clockwise is None:
        canvas_size = FOUR_WAY_CANVAS_SIZE
        panel_bounds = FOUR_WAY_PANEL_BOUNDS
        orientations = ORIENTATIONS
        heading = f"SAME TARGET IN FOUR ORIENTATIONS | DECLARED SLOT: {declared_slot}"
        subheading = "A IS THE CURRENT DISPLAY; B-D ARE COMPARISON ROTATIONS"
    else:
        canvas_size = PAIRWISE_CANVAS_SIZE
        panel_bounds = PAIRWISE_PANEL_BOUNDS
        orientations = (
            ("A", "CURRENT DISPLAY", None),
            (
                "B",
                f"{comparison_rotation_degrees_clockwise} DEGREES CLOCKWISE",
                TRANSPOSE_BY_CLOCKWISE_DEGREES[comparison_rotation_degrees_clockwise],
            ),
        )
        heading = f"SAME TARGET AT EQUAL SCALE | DECLARED SLOT: {declared_slot}"
        subheading = "A IS CURRENT; B IS THE ONLY COMPARISON ROTATION"

    canvas = Image.new("L", canvas_size, color=0)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (18, 10),
        heading,
        fill=255,
        font=_font(27),
    )
    draw.text(
        (18, 46),
        subheading,
        fill=190,
        font=_font(20),
    )
    for (candidate, description, transpose), bounds in zip(orientations, panel_bounds):
        image = anatomy if transpose is None else anatomy.transpose(transpose)
        _draw_panel(
            canvas,
            image=image,
            bounds=bounds,
            label=f"{candidate}: {description}",
            is_current=candidate == "A",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        canvas.save(temporary_path, format="PNG", optimize=True)
        validate_rendered_view_png(
            temporary_path, max_pixels=canvas_size[0] * canvas_size[1]
        )
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, output_path)
        os.chmod(output_path, 0o600)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")
    os.chmod(path, 0o600)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.resolve()
    output_manifest_path = args.output_manifest.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"orientation-choice manifest not found: {manifest_path}"
        )
    if output_manifest_path.parent != manifest_path.parent:
        raise ValueError(
            "orientation-choice output manifest must share the input manifest directory"
        )
    if args.max_source_pixels <= 0:
        raise ValueError("--max-source-pixels must be positive")
    current_rotation_pairs = list(args.current_rotation)
    current_rotation_orders = [order for order, _degrees in current_rotation_pairs]
    if len(current_rotation_orders) != len(set(current_rotation_orders)):
        raise ValueError("--current-rotation review orders must be unique")
    if current_rotation_pairs and args.comparison_rotation_degrees_clockwise is None:
        raise ValueError("--current-rotation requires a pairwise --comparison-rotation")
    current_rotation_by_order = dict(current_rotation_pairs)
    image_column = str(args.input_image_column).strip()
    if not image_column:
        raise ValueError("--input-image-column must be nonempty")

    asset_dir = output_manifest_path.with_suffix("")
    provenance_path = output_manifest_path.with_suffix(".provenance.json")
    readme_path = output_manifest_path.with_suffix(".README.md")
    for path in (output_manifest_path, asset_dir, provenance_path, readme_path):
        if path.exists():
            raise FileExistsError(
                f"refusing to overwrite orientation-choice output: {path}"
            )

    manifest = pd.read_parquet(manifest_path)
    validate_view_manifest_columns(manifest.columns, str(manifest_path))
    if image_column not in manifest:
        raise ValueError(
            f"orientation-choice manifest lacks image column: {image_column}"
        )
    manifest = manifest.sort_values("review_order", kind="stable").copy()
    manifest["view_id"] = manifest["view_id"].map(normalize_view_id)
    if manifest.empty:
        raise ValueError("orientation-choice manifest is empty")
    if manifest["view_id"].duplicated().any():
        raise ValueError("orientation-choice manifest contains duplicate view IDs")
    if manifest["review_order"].duplicated().any():
        raise ValueError("orientation-choice manifest contains duplicate review_order")
    missing_rotation_orders = sorted(
        set(current_rotation_orders) - set(manifest["review_order"])
    )
    if missing_rotation_orders:
        raise ValueError(
            "--current-rotation review order is outside the source manifest"
        )
    if not manifest["laterality"].isin({"L", "R"}).all():
        raise ValueError("orientation-choice manifest has non-L/R laterality")
    if not manifest["view"].isin({"CC", "MLO"}).all():
        raise ValueError("orientation-choice manifest has non-CC/MLO projection")

    root = manifest_path.parent.resolve()
    image_dir = asset_dir / "images"
    image_dir.mkdir(parents=True, mode=0o700)
    model_paths: dict[str, str] = {}
    for row in manifest.to_dict("records"):
        _canonical_path, _canonical_relative = _resolve_relative_image(
            root, row["image_path"], description="canonical image_path"
        )
        source_path, _source_relative = _resolve_relative_image(
            root, row[image_column], description=f"input image column {image_column}"
        )
        output_path = image_dir / f"{row['view_id']}.png"
        render_orientation_choice_png(
            source_path,
            output_path,
            declared_slot=f"{row['laterality']}{row['view']}",
            max_source_pixels=args.max_source_pixels,
            comparison_rotation_degrees_clockwise=(
                args.comparison_rotation_degrees_clockwise
            ),
            current_rotation_degrees_clockwise=current_rotation_by_order.get(
                int(row["review_order"]), 0
            ),
        )
        model_paths[row["view_id"]] = output_path.relative_to(root).as_posix()

    output = manifest.copy()
    output["model_image_path"] = output["view_id"].map(model_paths)
    output.to_parquet(output_manifest_path, index=False)
    os.chmod(output_manifest_path, 0o600)

    ordered_panel_digests = [
        sha256_file(image_dir / f"{view_id}.png") for view_id in output["view_id"]
    ]
    panel_bank_digest = hashlib.sha256(
        "".join(ordered_panel_digests).encode()
    ).hexdigest()
    command = shlex.join([sys.executable, *sys.argv])
    if args.comparison_rotation_degrees_clockwise is None:
        candidate_rotations = {"A": 0, "B": 90, "C": 180, "D": 270}
        canvas_size = FOUR_WAY_CANVAS_SIZE
        layout = "four_way_equal_scale"
    else:
        candidate_rotations = {
            "A": 0,
            "B": int(args.comparison_rotation_degrees_clockwise),
        }
        canvas_size = PAIRWISE_CANVAS_SIZE
        layout = "pairwise_equal_scale"
    provenance = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "source_manifest_sha256": sha256_file(manifest_path),
        "output_manifest_sha256": sha256_file(output_manifest_path),
        "ordered_panel_bank_sha256": panel_bank_digest,
        "rows": int(len(output)),
        "canonical_image_column": "image_path",
        "input_image_column": image_column,
        "model_image_column": "model_image_path",
        "current_display_candidate": "A",
        "synthetic_current_display_rotations_degrees_clockwise": {
            str(order): current_rotation_by_order[order]
            for order in sorted(current_rotation_by_order)
        },
        "candidate_rotations_degrees_clockwise": candidate_rotations,
        "anatomy_crop": "substantial_foreground_box",
        "layout": layout,
        "candidate_image_boxes": "equal square bounds",
        "canvas_size": list(canvas_size),
    }
    _write_json(provenance_path, provenance)
    readme_path.write_text(
        "\n".join(
            [
                "# Orientation-comparison view inputs",
                "",
                f"- rows: `{len(output)}`",
                "- canonical target column: `image_path`",
                f"- source image column: `{image_column}`",
                "- model input column: `model_image_path`",
                "- candidate A: current display",
                "- synthetic candidate-A rotations by review order: "
                f"`{dict(sorted(current_rotation_by_order.items()))}`",
                f"- layout: `{layout}`",
                f"- candidate rotations clockwise: `{candidate_rotations}`",
                f"- ordered panel-bank SHA-256: `{panel_bank_digest}`",
                "",
                "Each panel crops to substantial breast foreground before applying",
                "deterministic rotations. Equal square image boxes prevent rotated",
                "candidates from gaining effective scale merely because their aspect",
                "ratio matches the outer panel. The canonical target is unchanged.",
                "",
                f"Exact producer command: `{command}`",
                "",
            ]
        )
    )
    os.chmod(readme_path, 0o600)
    return {
        "rows": int(len(output)),
        "ordered_panel_bank_sha256": panel_bank_digest,
    }


def main() -> int:
    result = run_from_args(parse_args())
    print(
        "orientation-choice inputs ready: "
        f"rows={result['rows']} "
        f"bank_sha256={result['ordered_panel_bank_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
