from __future__ import annotations

from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
import pytest

from pipelines.preprocess import infer_view_fields
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
