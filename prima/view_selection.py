"""Shared mammography view selection utilities for preprocess and QC."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

from pydicom.dataset import FileDataset


def mammography_laterality(ds: FileDataset) -> str:
    """Return normalized image laterality, preferring ImageLaterality."""
    return (
        str(ds.get("ImageLaterality", ds.get("Laterality", "")) or "").strip().upper()
    )


def presentation_intent_type(ds: FileDataset) -> str:
    """Return normalized DICOM PresentationIntentType."""
    return str(ds.get("PresentationIntentType", "") or "").strip().upper()


def is_for_presentation(ds: FileDataset) -> bool:
    """Return whether the standard DICOM intent is FOR PRESENTATION."""
    return presentation_intent_type(ds) == "FOR PRESENTATION"


def burned_in_annotation_value(ds: FileDataset) -> str:
    """Return normalized BurnedInAnnotation metadata without interpreting it."""
    return str(ds.get("BurnedInAnnotation", "") or "").strip().upper()


def has_overlay_data(ds: FileDataset) -> bool:
    """Return whether any standard repeating overlay group contains pixel data."""
    return any(
        0x6000 <= int(tag.group) <= 0x601E
        and int(tag.group) % 2 == 0
        and int(tag.element) == 0x3000
        for tag in ds.keys()
    )


def view_modifier_code_meanings(ds: FileDataset) -> tuple[str, ...]:
    """Return explicit mammography view modifiers from either DICOM location."""
    modifier_items = list(ds.get("ViewModifierCodeSequence", []) or [])
    for view_item in ds.get("ViewCodeSequence", []) or []:
        modifier_items.extend(view_item.get("ViewModifierCodeSequence", []) or [])

    meanings: list[str] = []
    for item in modifier_items:
        meaning = str(item.get("CodeMeaning", "") or "").strip()
        code_value = str(item.get("CodeValue", "") or "").strip()
        label = meaning or code_value or "unspecified view modifier"
        if label not in meanings:
            meanings.append(label)
    return tuple(meanings)


def mirai_source_eligibility_reasons(ds: FileDataset) -> tuple[str, ...]:
    """Explain why a source DICOM cannot fill a standard Mirai input slot."""
    reasons: list[str] = []
    laterality = mammography_laterality(ds)
    if laterality not in {"L", "R"}:
        reasons.append(f"unsupported laterality {laterality or '<missing>'}")
    view_position = str(ds.get("ViewPosition", "") or "").strip().upper()
    if view_position not in {"CC", "MLO"}:
        reasons.append(f"unsupported ViewPosition {view_position or '<missing>'}")
    presentation_intent = presentation_intent_type(ds)
    if presentation_intent != "FOR PRESENTATION":
        reasons.append(
            "PresentationIntentType is " + (presentation_intent or "<missing>")
        )
    if str(ds.get("PartialView", "") or "").strip().upper() == "YES":
        reasons.append("PartialView is YES")
    modifiers = view_modifier_code_meanings(ds)
    if modifiers:
        reasons.append("view modifier: " + ", ".join(modifiers))
    return tuple(reasons)


def is_mirai_source_eligible(ds: FileDataset) -> bool:
    """Return whether a source DICOM can fill a standard Mirai input slot."""
    return not mirai_source_eligibility_reasons(ds)


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
    source_key: Union[str, Path],
) -> Tuple[float, int, float, str]:
    """Return deterministic sort key for choosing a canonical image per (lat, view)."""
    mag = estimate_magnification_factor(estimated_magnification_factor)
    if mag is None:
        mag = 1.0
    px = estimate_pixel_spacing_mm(pixel_spacing_mm)
    if px is None:
        px = 0.0
    return (abs(mag - 1.0), 0 if for_presentation else 1, -px, str(source_key))


def view_selection_key_from_dataset(
    ds: FileDataset, source_key: Union[str, Path]
) -> Tuple[float, int, float, str]:
    """Build a canonical view-selection key directly from a parsed DICOM dataset."""
    reasons = mirai_source_eligibility_reasons(ds)
    if reasons:
        raise ValueError("cannot select Mirai-ineligible source: " + "; ".join(reasons))
    return view_selection_key(
        for_presentation=True,
        estimated_magnification_factor=ds.get(
            "EstimatedRadiographicMagnificationFactor"
        ),
        pixel_spacing_mm=ds.get("PixelSpacing"),
        source_key=source_key,
    )
