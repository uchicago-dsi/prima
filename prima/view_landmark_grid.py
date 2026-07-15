"""Neutral single-image coordinate grids for mammography landmark probes."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from PIL import Image, ImageDraw, ImageFont, ImageOps

from prima.view_multiscale import substantial_foreground_box
from prima.view_qc import validate_rendered_view_png

LANDMARK_GRID_CANVAS_SIZE = (1024, 1100)
LANDMARK_GRID_IMAGE_BOUNDS = (64, 160, 960, 1056)
LANDMARK_GRID_BANDS = 4
FONT_PATH = Path("/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf")


def _font(size: int) -> ImageFont.FreeTypeFont:
    if not FONT_PATH.is_file():
        raise FileNotFoundError(f"landmark-grid font not found: {FONT_PATH}")
    return ImageFont.truetype(str(FONT_PATH), size=size)


def resolve_relative_image(
    root: Path, raw_path: object, *, description: str
) -> tuple[Path, str]:
    """Resolve one safe image path beneath a manifest root."""
    root = Path(root).resolve()
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


def render_landmark_grid_png(
    source_path: Path,
    output_path: Path,
    *,
    rotation_degrees_clockwise: int,
    max_source_pixels: int = 2_097_152,
) -> None:
    """Render one anatomy crop under four fixed top-to-bottom coordinate bands."""
    if rotation_degrees_clockwise not in {0, 180}:
        raise ValueError("landmark-grid rotation must be 0 or 180 degrees")
    source_path = Path(source_path)
    output_path = Path(output_path)
    validate_rendered_view_png(source_path, max_pixels=max_source_pixels)
    with Image.open(source_path) as source:
        source.load()
        grayscale = source.convert("L")
    anatomy = grayscale.crop(substantial_foreground_box(grayscale))
    if anatomy.width < 2 or anatomy.height < 2:
        raise ValueError("landmark-grid anatomy crop is empty")
    if rotation_degrees_clockwise == 180:
        anatomy = anatomy.transpose(Image.ROTATE_180)

    canvas = Image.new("L", LANDMARK_GRID_CANVAS_SIZE, color=0)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (18, 12),
        "ONE MLO IMAGE | LOCATE ANATOMY ONLY",
        fill=255,
        font=_font(30),
    )
    draw.text(
        (18, 60),
        "HORIZONTAL COORDINATES: BAND 1=TOP, 2, 3, BAND 4=BOTTOM",
        fill=205,
        font=_font(20),
    )
    draw.text(
        (18, 102),
        "REPORT THE BAND CONTAINING THE LANDMARK CENTER",
        fill=170,
        font=_font(18),
    )

    left, top, right, bottom = LANDMARK_GRID_IMAGE_BOUNDS
    box_size = (right - left, bottom - top)
    contained = ImageOps.contain(anatomy, box_size, Image.LANCZOS)
    x = left + (box_size[0] - contained.width) // 2
    y = top + (box_size[1] - contained.height) // 2
    canvas.paste(contained, (x, y))
    draw.rectangle(LANDMARK_GRID_IMAGE_BOUNDS, outline=220, width=3)

    band_height = (bottom - top) // LANDMARK_GRID_BANDS
    for band in range(1, LANDMARK_GRID_BANDS + 1):
        band_top = top + (band - 1) * band_height
        band_bottom = top + band * band_height
        if band < LANDMARK_GRID_BANDS:
            draw.line((left, band_bottom, right, band_bottom), fill=115, width=2)
        label_y = band_top + (band_bottom - band_top - 30) // 2
        draw.text((13, label_y), f"B{band}", fill=255, font=_font(24))
        draw.text((969, label_y), f"{band}", fill=255, font=_font(24))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        canvas.save(temporary_path, format="PNG", optimize=True)
        validate_rendered_view_png(
            temporary_path,
            max_pixels=LANDMARK_GRID_CANVAS_SIZE[0] * LANDMARK_GRID_CANVAS_SIZE[1],
        )
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, output_path)
        os.chmod(output_path, 0o600)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
