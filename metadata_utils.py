"""Shared utilities for DICOM metadata extraction and parsing."""

import math


def _is_null(value):
    """Check if a value is null/missing (None, NaN, empty string)."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def extract_base_modality(modality: str, study_description: str = None) -> str:
    """Extract the base imaging modality from a modality string.

    Ignores derived/secondary DICOM object classes (PR, SR, KO, OT, SC, DOC, XC, CADSR, REG, RTSTRUCT, etc.)
    If modality is missing, try to extract from study description.

    Parameters
    ----------
    modality : str
        modality string (e.g., "CT/PR/SR")
    study_description : str, optional
        study description to parse if modality is missing

    Returns
    -------
    str
        base modality (e.g., "CT"), or "Other" if no base modality is found
    """
    # if modality is missing, try to extract from study description
    if _is_null(modality):
        if study_description is not None and not _is_null(study_description):
            desc_upper = str(study_description).upper()
            # mammography patterns
            if any(keyword in desc_upper for keyword in ["MAM", "MAMM", "BREAST"]):
                return "MG"
            # MRI patterns
            elif desc_upper.startswith("MRI"):
                return "MR"
            # CT patterns
            elif desc_upper.startswith("CT"):
                return "CT"
            # ultrasound patterns
            elif desc_upper.startswith("US"):
                return "US"
            # nuclear medicine patterns
            elif desc_upper.startswith("NM"):
                return "NM"
            # x-ray patterns
            elif desc_upper.startswith("XR"):
                return "CR"  # treat XR as CR
        return "Other"

    # set of actual acquisition modalities
    base_modalities = {
        "CR",
        "DX",
        "MG",
        "US",
        "CT",
        "MR",
        "NM",
        "PT",
        "XA",
        "RF",
        "ES",
        "XC",
        "PX",
        "RG",
    }

    tokens = str(modality).split("/")
    for token in tokens:
        if token in base_modalities:
            return token
    return "Other"
