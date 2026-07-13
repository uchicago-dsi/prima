from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
import pytest

from prima.dicom_source import SOURCE_ARCHIVE_COLUMN
from qc.build_same_exam_context_views import (
    CANVAS_SIZE,
    compose_context_image,
    same_exam_choices,
    select_context_rows,
)


def view_id(index: int) -> str:
    return f"{index:064x}"


def test_context_selection_prefers_slot_diversity_and_selected_views() -> None:
    choices = pd.DataFrame(
        [
            {
                "view_id": view_id(1),
                "laterality": "L",
                "view": "CC",
                "source_kind": "candidate",
                "is_selected": True,
                "selection_rank": 1,
            },
            {
                "view_id": view_id(2),
                "laterality": "L",
                "view": "CC",
                "source_kind": "exclusion",
                "is_selected": False,
                "selection_rank": None,
            },
            {
                "view_id": view_id(3),
                "laterality": "L",
                "view": "MLO",
                "source_kind": "candidate",
                "is_selected": True,
                "selection_rank": 1,
            },
            {
                "view_id": view_id(4),
                "laterality": "R",
                "view": "CC",
                "source_kind": "candidate",
                "is_selected": True,
                "selection_rank": 1,
            },
            {
                "view_id": view_id(5),
                "laterality": "R",
                "view": "MLO",
                "source_kind": "candidate",
                "is_selected": True,
                "selection_rank": 1,
            },
        ]
    )

    selected = select_context_rows(
        choices,
        target_view_id=view_id(0),
        target_laterality="L",
        target_view="CC",
        maximum=3,
    )

    assert selected["view_id"].tolist() == [view_id(1), view_id(3), view_id(4)]
    assert selected["context_order"].tolist() == [1, 2, 3]


def test_context_pool_requires_exact_exam_and_physical_archive() -> None:
    pool = pd.DataFrame(
        [
            {"exam_id": "exam-a", SOURCE_ARCHIVE_COLUMN: "archive-a", "row": 1},
            {"exam_id": "exam-a", SOURCE_ARCHIVE_COLUMN: "archive-b", "row": 2},
            {"exam_id": "exam-b", SOURCE_ARCHIVE_COLUMN: "archive-a", "row": 3},
        ]
    )

    selected = same_exam_choices(
        pool,
        exam_id="exam-a",
        source_archive="archive-a",
    )

    assert selected["row"].tolist() == [1]


def test_context_composite_is_labeled_grayscale_and_bounded(tmp_path: Path) -> None:
    target = tmp_path / "target.png"
    references = [tmp_path / f"reference_{index}.png" for index in range(3)]
    Image.new("L", (600, 900), color=80).save(target)
    for index, path in enumerate(references, start=1):
        Image.new("L", (500, 400), color=80 + index * 30).save(path)
    output = tmp_path / "context.png"

    compose_context_image(
        target_path=target,
        context_paths=references,
        output_path=output,
    )

    with Image.open(output) as image:
        image.load()
        assert image.mode == "L"
        assert image.size == CANVAS_SIZE
        assert image.getbbox() is not None
    assert output.stat().st_mode & 0o777 == 0o600


def test_context_composite_rejects_more_than_three_references(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.png"
    Image.new("L", (8, 8)).save(target)
    references = []
    for index in range(4):
        path = tmp_path / f"reference_{index}.png"
        Image.new("L", (8, 8)).save(path)
        references.append(path)

    with pytest.raises(ValueError, match="at most three"):
        compose_context_image(
            target_path=target,
            context_paths=references,
            output_path=tmp_path / "context.png",
        )
