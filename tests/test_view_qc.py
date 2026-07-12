from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from prima.view_fallback import (
    align_candidates_to_selected_views,
    choose_exact_slot_views,
)
from prima.view_qc import (
    VIEW_LABEL_PASS,
    VIEW_LABEL_VERTICAL_LINE,
    empty_view_qc_state,
    load_view_qc_state,
    render_dicom_view_png,
    save_view_qc_state,
    set_view_label,
    summarize_view_qc_state,
)
from qc.build_view_qc_pilot import sample_views
from qc.view_qc_gallery import load_review_items


def view_id(index: int) -> str:
    return f"{index:064x}"


def test_view_qc_state_uses_one_manifest_denominator(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = empty_view_qc_state()
    state = set_view_label(state, view_id(1), VIEW_LABEL_PASS)
    state = set_view_label(state, view_id(2), VIEW_LABEL_VERTICAL_LINE)
    save_view_qc_state(path, state)

    loaded = load_view_qc_state(path)
    assert summarize_view_qc_state(loaded, [view_id(1), view_id(2), view_id(3)]) == {
        "total": 3,
        "reviewed": 2,
        "remaining": 1,
        "pass": 1,
        "vertical_line": 1,
    }
    assert path.stat().st_mode & 0o777 == 0o600


def test_gallery_exposes_no_exam_or_patient_identifiers(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("L", (8, 8)).save(image_dir / f"{view_id(1)}.png")
    manifest = pd.DataFrame(
        [
            {
                "view_id": view_id(1),
                "image_path": f"images/{view_id(1)}.png",
                "laterality": "L",
                "view": "CC",
                "review_order": 1,
                "patient_id": "hidden",
                "exam_id": "hidden",
                "stratum": "hidden",
            }
        ]
    )
    manifest_path = tmp_path / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)

    items, images = load_review_items(manifest_path)
    assert set(items[0]) == {
        "view_id",
        "laterality",
        "view",
        "review_order",
        "image_url",
    }
    assert images[view_id(1)].is_file()


def test_view_sampling_is_disjoint_and_one_per_exam() -> None:
    rows = []
    for exam_index in range(10):
        for slot_index, (laterality, view) in enumerate(
            [("L", "CC"), ("R", "CC"), ("L", "MLO"), ("R", "MLO")]
        ):
            rows.append(
                {
                    "exam_id": f"exam-{exam_index}",
                    "laterality": laterality,
                    "view": view,
                    "sha256": view_id(exam_index * 4 + slot_index + 1),
                    "heuristic_score": float(exam_index * 4 + slot_index),
                    "max_column_z": float(slot_index),
                }
            )
    selected = sample_views(pd.DataFrame(rows), 8, 4, seed=7)
    assert len(selected) == 8
    assert selected["exam_id"].nunique() == 8
    assert set(selected["stratum"]) == {"heuristic_top", "random"}
    assert sorted(selected["review_order"]) == list(range(1, 9))
    assert (selected["view_id"] == selected["sha256"]).all()


class FakeDataset(dict):
    def __init__(self, pixels: np.ndarray) -> None:
        super().__init__(PhotometricInterpretation="MONOCHROME2")
        self.pixel_array = pixels


def test_render_dicom_view_png_respects_pixel_budget(tmp_path: Path) -> None:
    pixels = np.arange(200 * 100, dtype=np.uint16).reshape(200, 100)
    output = tmp_path / "view.png"
    render_dicom_view_png(FakeDataset(pixels), output, max_pixels=5_000)
    with Image.open(output) as image:
        assert image.width * image.height <= 5_000
        assert image.mode == "L"


def test_fallback_never_crosses_exact_view_slots() -> None:
    candidates = pd.DataFrame(
        [
            {
                "exam_id": "exam-1",
                "laterality": "L",
                "view": "CC",
                "sop_instance_uid": "sop-1",
                "sha256": view_id(1),
                "selection_rank": 1,
                "is_selected": True,
            },
            {
                "exam_id": "exam-1",
                "laterality": "L",
                "view": "CC",
                "sop_instance_uid": "sop-2",
                "sha256": view_id(2),
                "selection_rank": 2,
                "is_selected": False,
            },
            {
                "exam_id": "exam-1",
                "laterality": "R",
                "view": "CC",
                "sop_instance_uid": "sop-3",
                "sha256": view_id(3),
                "selection_rank": 1,
                "is_selected": True,
            },
        ]
    )
    decisions = choose_exact_slot_views(
        candidates,
        {
            view_id(1): VIEW_LABEL_VERTICAL_LINE,
            view_id(2): VIEW_LABEL_PASS,
            view_id(3): VIEW_LABEL_PASS,
        },
    ).set_index(["laterality", "view"])

    left = decisions.loc[("L", "CC")]
    right = decisions.loc[("R", "CC")]
    assert left["fallback_status"] == "alternate_pass"
    assert left["selected_view_id"] == view_id(2)
    assert left["selected_candidate_rank"] == 2
    assert right["fallback_status"] == "original_pass"
    assert right["selected_view_id"] == view_id(3)


def test_candidate_alignment_promotes_authoritative_source() -> None:
    candidates = pd.DataFrame(
        [
            {
                "exam_id": "exam-1",
                "laterality": "L",
                "view": "CC",
                "sop_instance_uid": "sop-1",
                "sha256": view_id(1),
                "selection_rank": 1,
                "is_selected": True,
            },
            {
                "exam_id": "exam-1",
                "laterality": "L",
                "view": "CC",
                "sop_instance_uid": "sop-2",
                "sha256": view_id(2),
                "selection_rank": 2,
                "is_selected": False,
            },
        ]
    )
    selected = candidates.iloc[[1]].drop(columns=["selection_rank", "is_selected"])
    aligned = align_candidates_to_selected_views(candidates, selected)
    promoted = aligned[aligned["is_selected"]].iloc[0]
    demoted = aligned[~aligned["is_selected"]].iloc[0]
    assert promoted["sha256"] == view_id(2)
    assert promoted["selection_rank"] == 1
    assert demoted["sha256"] == view_id(1)
    assert demoted["selection_rank"] == 2
