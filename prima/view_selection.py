"""Shared heuristics for choosing one canonical image per view slot."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _coerce_positive_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _first_spacing_value(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        for item in raw:
            value = _coerce_positive_float(item)
            if value is not None:
                return value
        return None
    if hasattr(raw, "__len__") and not isinstance(raw, (str, bytes)):
        try:
            for item in raw:
                value = _coerce_positive_float(item)
                if value is not None:
                    return value
            return None
        except TypeError:
            pass
    return _coerce_positive_float(raw)


def estimate_pixel_spacing_mm(ds: Any) -> float | None:
    """Best-effort mm/pixel estimate from common mammography DICOM tags."""

    for attr in (
        "PixelSpacing",
        "ImagerPixelSpacing",
        "DetectorElementSpacing",
        "NominalScannedPixelSpacing",
    ):
        value = _first_spacing_value(getattr(ds, attr, None))
        if value is not None:
            return value
    return None


def estimate_magnification_factor(
    ds: Any,
    pixel_spacing_mm: float | None = None,
) -> float | None:
    """Estimate magnification factor, preferring explicit DICOM metadata."""

    explicit = _coerce_positive_float(
        getattr(ds, "EstimatedRadiographicMagnificationFactor", None)
    )
    if explicit is not None:
        return explicit

    imager_spacing = _first_spacing_value(getattr(ds, "ImagerPixelSpacing", None))
    if pixel_spacing_mm is not None and imager_spacing is not None:
        ratio = _coerce_positive_float(imager_spacing / pixel_spacing_mm)
        if ratio is not None:
            return ratio

    dsd = _coerce_positive_float(getattr(ds, "DistanceSourceToDetector", None))
    dsp = _coerce_positive_float(getattr(ds, "DistanceSourceToPatient", None))
    if dsd is not None and dsp is not None:
        ratio = _coerce_positive_float(dsd / dsp)
        if ratio is not None:
            return ratio

    return None


def view_selection_key(
    *,
    for_presentation: bool,
    estimated_magnification_factor: float | None,
    pixel_spacing_mm: float | None,
    dicom_path: str | Path | None,
) -> tuple[int, float, float, str]:
    """Stable ordering for duplicate (exam, laterality, view) candidates.

    Lower is better:
    1. Prefer presentation images over processing images.
    2. Prefer views closest to standard scale (magnification ~= 1).
    3. Prefer smaller pixel spacing when otherwise tied.
    4. Use path as a deterministic final tiebreaker.
    """

    presentation_rank = 0 if for_presentation else 1
    magnification_rank = (
        abs(estimated_magnification_factor - 1.0)
        if estimated_magnification_factor is not None
        else 999.0
    )
    spacing_rank = pixel_spacing_mm if pixel_spacing_mm is not None else 999.0
    path_rank = str(dicom_path or "")
    return (presentation_rank, magnification_rank, spacing_rank, path_rank)


def view_selection_key_from_dataset(
    ds: Any,
    dicom_path: str | Path | None,
) -> tuple[int, float, float, str]:
    """Dataset-backed wrapper used by QC and preprocessing code."""

    presentation_intent = str(getattr(ds, "PresentationIntentType", "")).upper()
    for_presentation = presentation_intent != "FOR PROCESSING"
    pixel_spacing_mm = estimate_pixel_spacing_mm(ds)
    magnification = estimate_magnification_factor(ds, pixel_spacing_mm)
    return view_selection_key(
        for_presentation=for_presentation,
        estimated_magnification_factor=magnification,
        pixel_spacing_mm=pixel_spacing_mm,
        dicom_path=dicom_path,
    )
