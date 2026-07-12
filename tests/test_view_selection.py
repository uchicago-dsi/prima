from __future__ import annotations

import json
from pathlib import Path

from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian
import pytest

from pipelines.preprocess import (
    _process_exam_dir,
    infer_view_fields,
    infer_view_identity,
)
from prima.view_selection import (
    is_standard_mirai_view,
    nonstandard_mirai_view_reasons,
    view_modifier_code_meanings,
    view_selection_key_from_dataset,
)


def standard_view(view_position: str = "CC") -> Dataset:
    dataset = Dataset()
    dataset.ImageLaterality = "L"
    dataset.ViewPosition = view_position
    dataset.PresentationIntentType = "FOR PRESENTATION"
    dataset.EstimatedRadiographicMagnificationFactor = 1.0
    dataset.PixelSpacing = [0.07, 0.07]
    return dataset


def implant_displaced_view() -> Dataset:
    dataset = standard_view()
    view_code = Dataset()
    modifier = Dataset()
    modifier.CodeValue = "R-102D5"
    modifier.CodeMeaning = "Implant Displaced"
    view_code.ViewModifierCodeSequence = Sequence([modifier])
    dataset.ViewCodeSequence = Sequence([view_code])
    return dataset


def test_standard_cc_and_mlo_views_are_allowed() -> None:
    for view_position in ("CC", "MLO"):
        dataset = standard_view(view_position)
        assert is_standard_mirai_view(dataset)
        assert nonstandard_mirai_view_reasons(dataset) == ()
        assert infer_view_fields(dataset) == ("L", view_position)


def test_implant_displaced_modifier_is_not_a_standard_mirai_view() -> None:
    dataset = implant_displaced_view()
    assert view_modifier_code_meanings(dataset) == ("Implant Displaced",)
    assert not is_standard_mirai_view(dataset)
    assert nonstandard_mirai_view_reasons(dataset) == (
        "view modifier: Implant Displaced",
    )
    with pytest.raises(ValueError, match="Implant Displaced"):
        infer_view_fields(dataset)
    with pytest.raises(ValueError, match="non-standard Mirai view"):
        view_selection_key_from_dataset(dataset, "source.dcm")


def test_partial_view_is_not_a_standard_mirai_view() -> None:
    dataset = standard_view("MLO")
    dataset.PartialView = "YES"
    assert not is_standard_mirai_view(dataset)
    assert nonstandard_mirai_view_reasons(dataset) == ("PartialView is YES",)
    with pytest.raises(ValueError, match="PartialView is YES"):
        infer_view_fields(dataset)


def test_top_level_view_modifier_is_detected() -> None:
    dataset = standard_view()
    modifier = Dataset()
    modifier.CodeValue = "R-102D2"
    modifier.CodeMeaning = "Rolled Medial"
    dataset.ViewModifierCodeSequence = Sequence([modifier])
    assert view_modifier_code_meanings(dataset) == ("Rolled Medial",)
    assert not is_standard_mirai_view(dataset)


def test_nonstandard_view_identity_is_read_before_eligibility_filter() -> None:
    dataset = standard_view("ML")
    assert infer_view_identity(dataset) == ("L", "ML")
    with pytest.raises(ValueError, match="unsupported ViewPosition ML"):
        infer_view_fields(dataset)


def test_nonstandard_diagnostic_view_is_persisted_with_source_and_reason(
    tmp_path: Path,
) -> None:
    exam_dir = tmp_path / "patient" / "exam"
    exam_dir.mkdir(parents=True)
    dicom_path = exam_dir / "spot.dcm"

    dataset = implant_displaced_view()
    dataset.PatientID = "TEST_PATIENT"
    dataset.StudyInstanceUID = "1.2.826.0.1.3680043.10.999.1"
    dataset.SOPInstanceUID = "1.2.826.0.1.3680043.10.999.2"
    dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.1.2"
    dataset.file_meta = Dataset()
    dataset.file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
    dataset.file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
    dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset.Rows = 2
    dataset.Columns = 2
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelData = b"\x00\x00" * 4
    dataset.save_as(dicom_path, enforce_file_format=True)

    result = _process_exam_dir(
        exam_dir,
        Path("patient/exam.tar.zst"),
    )

    assert result["rows"] == []
    assert result["exam_status"] == "no_standard_dicoms"
    assert result["excluded_nonstandard_dicoms"] == 1
    assert len(result["exclusion_rows"]) == 1
    exclusion = result["exclusion_rows"][0]
    assert exclusion["source_archive_relpath"] == "patient/exam.tar.zst"
    assert exclusion["source_archive_member"] == "exam/spot.dcm"
    assert json.loads(exclusion["view_modifiers"]) == ["Implant Displaced"]
    assert json.loads(exclusion["exclusion_reasons"]) == [
        "view modifier: Implant Displaced"
    ]
