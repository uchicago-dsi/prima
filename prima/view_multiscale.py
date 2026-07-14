"""Multiscale visual inputs for single-view QC models."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from prima.view_qc import validate_rendered_view_png

MULTISCALE_CANVAS_SIZE = (1600, 1200)
MULTISCALE_CANVAS_MAX_PIXELS = MULTISCALE_CANVAS_SIZE[0] * MULTISCALE_CANVAS_SIZE[1]
FONT_PATH = Path("/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf")


def _font(size: int) -> ImageFont.FreeTypeFont:
    if not FONT_PATH.is_file():
        raise FileNotFoundError(f"multiscale view font not found: {FONT_PATH}")
    return ImageFont.truetype(str(FONT_PATH), size=size)


def _draw_panel(
    canvas: Image.Image,
    *,
    image: Image.Image,
    bounds: tuple[int, int, int, int],
    label: str,
    border_value: int,
) -> None:
    left, top, right, bottom = bounds
    if right <= left or bottom <= top:
        raise ValueError("multiscale view panel has invalid bounds")
    label_height = 38
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(bounds, outline=border_value, width=3)
    draw.text((left + 8, top + 5), label, fill=255, font=_font(22))
    image_box = (left + 5, top + label_height, right - 5, bottom - 5)
    box_size = (
        image_box[2] - image_box[0],
        image_box[3] - image_box[1],
    )
    contained = ImageOps.contain(image, box_size, Image.LANCZOS)
    x = image_box[0] + (box_size[0] - contained.width) // 2
    y = image_box[1] + (box_size[1] - contained.height) // 2
    canvas.paste(contained, (x, y))


def substantial_foreground_box(image: Image.Image) -> tuple[int, int, int, int]:
    """Bound substantial non-background content while ignoring isolated text."""
    pixels = np.asarray(image)
    foreground = pixels > 4
    column_minimum = max(8, round(image.height * 0.01))
    row_minimum = max(8, round(image.width * 0.01))
    columns = np.flatnonzero(foreground.sum(axis=0) >= column_minimum)
    rows = np.flatnonzero(foreground.sum(axis=1) >= row_minimum)
    if not len(columns) or not len(rows):
        return (0, 0, image.width, image.height)
    margin_x = max(2, round(image.width * 0.03))
    margin_y = max(2, round(image.height * 0.03))
    return (
        max(0, int(columns[0]) - margin_x),
        max(0, int(rows[0]) - margin_y),
        min(image.width, int(columns[-1]) + 1 + margin_x),
        min(image.height, int(rows[-1]) + 1 + margin_y),
    )


def overlapping_detail_crops(
    detail_box: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]]:
    """Return four overlapping bands along the foreground's long axis."""
    left, top, right, bottom = detail_box
    width = right - left
    height = bottom - top
    if width < 2 or height < 2:
        raise ValueError("multiscale source image is too small")
    starts = (0.0, 0.2, 0.4, 0.6)
    if height >= width:
        return [
            (
                left,
                top + round(height * start),
                right,
                top + round(height * (start + 0.4)),
            )
            for start in starts
        ]
    return [
        (
            left + round(width * start),
            top,
            left + round(width * (start + 0.4)),
            bottom,
        )
        for start in starts
    ]


def render_multiscale_view_png(
    source_path: Path,
    output_path: Path,
    *,
    max_source_pixels: int = 2_097_152,
) -> None:
    """Render one full view beside four enlarged overlapping details."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    validate_rendered_view_png(source_path, max_pixels=max_source_pixels)
    with Image.open(source_path) as source:
        source.load()
        image = source.convert("L")

    canvas = Image.new("L", MULTISCALE_CANVAS_SIZE, color=0)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (18, 10),
        "ONE TARGET VIEW: FULL + 4 OVERLAPPING ENLARGED DETAILS",
        fill=255,
        font=_font(27),
    )
    _draw_panel(
        canvas,
        image=image,
        bounds=(12, 52, 496, 1188),
        label="FULL TARGET VIEW",
        border_value=255,
    )

    crops = overlapping_detail_crops(substantial_foreground_box(image))
    detail_bounds = (
        (508, 52, 1048, 615),
        (1058, 52, 1598, 615),
        (508, 625, 1048, 1188),
        (1058, 625, 1598, 1188),
    )
    for index, (crop_box, panel_bounds) in enumerate(
        zip(crops, detail_bounds), start=1
    ):
        _draw_panel(
            canvas,
            image=image.crop(crop_box),
            bounds=panel_bounds,
            label=f"ENLARGED DETAIL {index}",
            border_value=150,
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
            temporary_path, max_pixels=MULTISCALE_CANVAS_MAX_PIXELS
        )
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, output_path)
        os.chmod(output_path, 0o600)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
