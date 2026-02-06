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
            # mammography patterns - more comprehensive matching
            if any(
                keyword in desc_upper
                for keyword in [
                    "MAM",
                    "MAMM",
                    "BREAST",
                    "SENO",
                    "MAMMOGRAPHY",
                    "MAMMO",
                    "BI-RADS",
                    "SCREENING MAMMO",
                    "DIAGNOSTIC MAMMO",
                ]
            ):
                return "MG"
            # MRI patterns - more comprehensive matching
            elif any(
                keyword in desc_upper
                for keyword in ["MRI", "MAGNETIC RESONANCE", "MR ", "MR/", "MR-"]
            ):
                return "MR"
            # CT patterns
            elif any(
                keyword in desc_upper for keyword in ["CT", "COMPUTED TOMOGRAPHY"]
            ):
                return "CT"
            # ultrasound patterns
            elif any(
                keyword in desc_upper
                for keyword in ["US", "ULTRASOUND", "SONO", "SONOGRAM"]
            ):
                return "US"
            # nuclear medicine patterns
            elif any(keyword in desc_upper for keyword in ["NM", "NUCLEAR"]):
                return "NM"
            # x-ray patterns
            elif any(
                keyword in desc_upper
                for keyword in ["XR", "X-RAY", "CHEST XR", "CHEST X-RAY"]
            ):
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

    # set of secondary/derived DICOM object types that should be ignored
    # and the actual modality determined from study description
    secondary_object_types = {
        "PR",
        "SR",
        "KO",
        "OT",
        "SC",
        "DOC",
        "XC",
        "CADSR",
        "REG",
        "RTSTRUCT",
        "AU",  # Audio - often misused for other modalities
    }

    # first check if the modality string contains any base modality
    modality_upper = str(modality).upper()

    # special handling for MG - check for mammography patterns in modality string
    if any(keyword in modality_upper for keyword in ["MG", "MAM", "MAMM", "BREAST"]):
        return "MG"

    # special handling for MR - check for MRI patterns in modality string
    if any(keyword in modality_upper for keyword in ["MR", "MRI"]):
        return "MR"

    # check for other modalities in the string
    for base_mod in [
        "CT",
        "US",
        "NM",
        "PT",
        "CR",
        "DX",
        "XA",
        "RF",
        "ES",
        "XC",
        "PX",
        "RG",
    ]:
        if base_mod in modality_upper:
            return base_mod

    # special case: if modality is a secondary object type (like PR - Presentation State),
    # try to determine the actual imaging modality from study description
    if (
        modality_upper in secondary_object_types
        and study_description is not None
        and not _is_null(study_description)
    ):
        desc_upper = str(study_description).upper()

        # MRI patterns - check first since it's more specific (e.g., "MRI BREAST" should be MR, not MG)
        if any(
            keyword in desc_upper
            for keyword in ["MRI", "MAGNETIC RESONANCE", "MR ", "MR/", "MR-"]
        ):
            return "MR"
        # mammography patterns - more specific matching to avoid conflicts
        elif any(
            keyword in desc_upper
            for keyword in [
                "MAM",
                "MAMM",
                "MAMMOGRAPHY",
                "MAMMO",
                "BI-RADS",
                "SCREENING MAMMO",
                "DIAGNOSTIC MAMMO",
                "SENO",
            ]
        ):
            return "MG"
        # breast-specific patterns (but only if no other modality is indicated)
        elif any(
            keyword in desc_upper
            for keyword in [
                "BREAST",
                "BRST",
            ]
        ) and not any(
            other_modality in desc_upper
            for other_modality in [
                "MRI",
                "MAGNETIC RESONANCE",
                "MR ",
                "MR/",
                "MR-",
                "CT",
                "COMPUTED TOMOGRAPHY",
                "US",
                "ULTRASOUND",
                "SONO",
                "SONOGRAM",
                "NM",
                "NUCLEAR",
                "XR",
                "X-RAY",
                "CHEST XR",
                "CHEST X-RAY",
            ]
        ):
            return "MG"
        # MG modality code (but only if no other modality is indicated)
        elif "MG" in desc_upper and not any(
            other_modality in desc_upper
            for other_modality in [
                "MRI",
                "MAGNETIC RESONANCE",
                "MR ",
                "MR/",
                "MR-",
                "CT",
                "COMPUTED TOMOGRAPHY",
                "US",
                "ULTRASOUND",
                "SONO",
                "SONOGRAM",
                "NM",
                "NUCLEAR",
                "XR",
                "X-RAY",
                "CHEST XR",
                "CHEST X-RAY",
            ]
        ):
            return "MG"
        # CT patterns
        elif any(keyword in desc_upper for keyword in ["CT", "COMPUTED TOMOGRAPHY"]):
            return "CT"
        # ultrasound patterns
        elif any(
            keyword in desc_upper
            for keyword in ["US", "ULTRASOUND", "SONO", "SONOGRAM"]
        ):
            return "US"
        # nuclear medicine patterns
        elif any(keyword in desc_upper for keyword in ["NM", "NUCLEAR"]):
            return "NM"
        # x-ray patterns
        elif any(
            keyword in desc_upper
            for keyword in ["XR", "X-RAY", "CHEST XR", "CHEST X-RAY"]
        ):
            return "CR"  # treat XR as CR

    # fallback to token-based parsing
    tokens = str(modality).split("/")
    for token in tokens:
        if token in base_modalities:
            return token
    return "Other"
