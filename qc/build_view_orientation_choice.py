#!/usr/bin/env python3
"""Build four-orientation comparison panels for view-level auto-QC."""

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
CANVAS_SIZE = (1600, 1200)
CANVAS_MAX_PIXELS = CANVAS_SIZE[0] * CANVAS_SIZE[1]
PANEL_BOUNDS = (
    (12, 78, 796, 632),
    (804, 78, 1588, 632),
    (12, 640, 796, 1192),
    (804, 640, 1588, 1192),
)
ORIENTATIONS = (
    ("A", "CURRENT DISPLAY", None),
    ("B", "90 DEGREES CLOCKWISE", Image.ROTATE_270),
    ("C", "180 DEGREES", Image.ROTATE_180),
    ("D", "90 DEGREES COUNTERCLOCKWISE", Image.ROTATE_90),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--input-image-column",
        default="image_path",
        help="manifest image column to rotate; image_path remains canonical",
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
) -> None:
    left, top, right, bottom = bounds
    label_height = 42
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(bounds, outline=255 if label.startswith("A") else 150, width=4)
    draw.text((left + 10, top + 6), label, fill=255, font=_font(23))
    image_box = (left + 6, top + label_height, right - 6, bottom - 6)
    box_size = (image_box[2] - image_box[0], image_box[3] - image_box[1])
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
) -> None:
    """Render current, clockwise, inverted, and counterclockwise candidates."""
    validate_rendered_view_png(source_path, max_pixels=max_source_pixels)
    with Image.open(source_path) as source:
        source.load()
        grayscale = source.convert("L")
    anatomy = grayscale.crop(substantial_foreground_box(grayscale))
    if anatomy.width < 2 or anatomy.height < 2:
        raise ValueError("orientation-choice anatomy crop is empty")

    canvas = Image.new("L", CANVAS_SIZE, color=0)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (18, 10),
        f"SAME TARGET IN FOUR ORIENTATIONS | DECLARED SLOT: {declared_slot}",
        fill=255,
        font=_font(27),
    )
    draw.text(
        (18, 46),
        "A IS THE CURRENT DISPLAY; B-D ARE COMPARISON ROTATIONS",
        fill=190,
        font=_font(20),
    )
    for (candidate, description, transpose), bounds in zip(ORIENTATIONS, PANEL_BOUNDS):
        image = anatomy if transpose is None else anatomy.transpose(transpose)
        _draw_panel(
            canvas,
            image=image,
            bounds=bounds,
            label=f"{candidate}: {description}",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        canvas.save(temporary_path, format="PNG", optimize=True)
        validate_rendered_view_png(temporary_path, max_pixels=CANVAS_MAX_PIXELS)
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
        "candidate_rotations_degrees_clockwise": {
            "A": 0,
            "B": 90,
            "C": 180,
            "D": 270,
        },
        "anatomy_crop": "substantial_foreground_box",
        "canvas_size": list(CANVAS_SIZE),
    }
    _write_json(provenance_path, provenance)
    readme_path.write_text(
        "\n".join(
            [
                "# Four-orientation view inputs",
                "",
                f"- rows: `{len(output)}`",
                "- canonical target column: `image_path`",
                f"- source image column: `{image_column}`",
                "- model input column: `model_image_path`",
                "- candidate A: current display",
                "- candidates B/C/D: 90/180/270 degrees clockwise",
                f"- ordered panel-bank SHA-256: `{panel_bank_digest}`",
                "",
                "Each panel crops to substantial breast foreground before applying",
                "the four deterministic rotations, suppressing most isolated",
                "burned-in text as an orientation shortcut. The canonical target",
                "is unchanged.",
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
