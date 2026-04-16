"""Shared mammography view selection utilities for preprocess and QC."""

from pathlib import Path
from typing import Optional, Tuple, Union

from pydicom.dataset import FileDataset


def to_float(value: object) -> Optional[float]:
    """Convert scalar or multivalue DICOM field to float when possible."""
    if value is None:
        return None
    try:
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return None
            return float(value[0])
        return float(value)
    except Exception:
        return None


def estimate_pixel_spacing_mm(value: object) -> Optional[float]:
    """Extract first pixel spacing value (mm) from DICOM PixelSpacing-like values."""
    return to_float(value)


def estimate_magnification_factor(value: object) -> Optional[float]:
    """Extract estimated radiographic magnification factor from DICOM value."""
    return to_float(value)


def view_selection_key(
    *,
    for_presentation: bool,
    estimated_magnification_factor: object,
    pixel_spacing_mm: object,
    dicom_path: Union[str, Path],
) -> Tuple[float, int, float, str]:
    """Return deterministic sort key for choosing a canonical image per (lat, view)."""
    mag = estimate_magnification_factor(estimated_magnification_factor)
    if mag is None:
        mag = 1.0
    px = estimate_pixel_spacing_mm(pixel_spacing_mm)
    if px is None:
        px = 0.0
    return (abs(mag - 1.0), 0 if for_presentation else 1, -px, str(dicom_path))


def view_selection_key_from_dataset(
    ds: FileDataset, dicom_path: Union[str, Path]
) -> Tuple[float, int, float, str]:
    """Build a canonical view-selection key directly from a parsed DICOM dataset."""
    presentation_intent = (
        str(ds.get("PresentationIntentType", "") or "").strip().upper()
    )
    return view_selection_key(
        for_presentation=(presentation_intent == "FOR PRESENTATION"),
        estimated_magnification_factor=ds.get(
            "EstimatedRadiographicMagnificationFactor"
        ),
        pixel_spacing_mm=ds.get("PixelSpacing"),
        dicom_path=dicom_path,
    )
