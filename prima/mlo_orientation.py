"""Synthetic MLO orientation inputs and exact-label scoring."""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

from prima.view_multiscale import substantial_foreground_box
from prima.view_qc import validate_rendered_view_png

ORIENTATION_CANVAS_SIZE = (896, 896)
ORIENTATION_LABELS = ("UPRIGHT", "INVERTED")
ORIENTATION_PROMPT = (
    "This is one MLO mammogram image. Classify only its superior-to-inferior "
    "anatomical display orientation. In an UPRIGHT MLO, the axillary and upper "
    "pectoral region is toward the top and the lower breast/inframammary region "
    "is toward the bottom. An INVERTED image has those directions reversed by "
    "an approximately 180-degree whole-image rotation. Ignore laterality, "
    "pathology, exposure, devices, and positioning quality. Answer with exactly "
    "UPRIGHT or INVERTED and no other text."
)
REPRESENTATION_VERSION = "mlo-orientation-anatomy-crop-v3"


def suppress_isolated_annotations(image: Image.Image) -> Image.Image:
    """Zero small bright components outside the substantial anatomy support."""
    pixels = np.asarray(image)
    background_threshold = max(1.0, float(np.percentile(pixels, 10)) + 1.0)
    foreground = pixels > background_threshold
    components, count = ndimage.label(foreground, structure=np.ones((3, 3)))
    if count == 0:
        return image.copy()
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    largest = int(sizes.max())
    minimum_size = max(64, round(largest * 0.01))
    substantial_ids = np.flatnonzero(sizes >= minimum_size)
    support = np.isin(components, substantial_ids)
    support = ndimage.binary_fill_holes(support)
    dilation_iterations = max(3, round(min(image.size) * 0.015))
    support = ndimage.binary_dilation(support, iterations=dilation_iterations)
    isolated_bright_pixels = foreground & ~support
    isolated_bright_pixels = ndimage.binary_dilation(
        isolated_bright_pixels, iterations=1
    )
    cleaned = pixels.copy()
    cleaned[isolated_bright_pixels] = 0
    return Image.fromarray(cleaned)


def render_orientation_png(
    source_path: Path,
    output_path: Path,
    *,
    rotation_degrees_clockwise: int,
    max_source_pixels: int = 2_097_152,
) -> None:
    """Crop substantial anatomy, rotate it, and center it on a square canvas."""
    if rotation_degrees_clockwise not in {0, 180}:
        raise ValueError("MLO orientation rotation must be 0 or 180 degrees")
    source_path = Path(source_path)
    output_path = Path(output_path)
    validate_rendered_view_png(source_path, max_pixels=max_source_pixels)
    with Image.open(source_path) as source:
        source.load()
        grayscale = source.convert("L")
    anatomy = grayscale.crop(substantial_foreground_box(grayscale))
    anatomy = suppress_isolated_annotations(anatomy)
    if anatomy.width < 2 or anatomy.height < 2:
        raise ValueError("MLO orientation anatomy crop is empty")
    canvas = Image.new("L", ORIENTATION_CANVAS_SIZE, color=0)
    contained = ImageOps.contain(anatomy, ORIENTATION_CANVAS_SIZE, Image.LANCZOS)
    x = (canvas.width - contained.width) // 2
    y = (canvas.height - contained.height) // 2
    canvas.paste(contained, (x, y))
    if rotation_degrees_clockwise == 180:
        canvas = canvas.transpose(Image.ROTATE_180)

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
            max_pixels=ORIENTATION_CANVAS_SIZE[0] * ORIENTATION_CANVAS_SIZE[1],
        )
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, output_path)
        os.chmod(output_path, 0o600)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def parse_orientation_label(text: object) -> str | None:
    """Return one unambiguous orientation label from generated text."""
    found = set(re.findall(r"\b(?:UPRIGHT|INVERTED)\b", str(text).upper()))
    if len(found) != 1:
        return None
    return found.pop()


def score_orientation_predictions(
    expected: list[str], predicted: list[str | None]
) -> dict[str, object]:
    """Score synthetic inversion as the positive class."""
    if len(expected) != len(predicted):
        raise ValueError("orientation prediction lengths differ")
    if any(label not in ORIENTATION_LABELS for label in expected):
        raise ValueError("orientation expected labels are invalid")
    tp = sum(e == "INVERTED" and p == e for e, p in zip(expected, predicted))
    fn = sum(e == "INVERTED" and p != e for e, p in zip(expected, predicted))
    tn = sum(e == "UPRIGHT" and p == e for e, p in zip(expected, predicted))
    fp = sum(e == "UPRIGHT" and p != e for e, p in zip(expected, predicted))
    return {
        "rows": len(expected),
        "exact": tp + tn,
        "accuracy": (tp + tn) / len(expected) if expected else None,
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "inversion_sensitivity": tp / (tp + fn) if tp + fn else None,
        "upright_specificity": tn / (tn + fp) if tn + fp else None,
    }
