from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

import auto_annotate_qc
from prima.view_auto_qc import (
    load_view_auto_run,
    new_view_auto_run,
    require_compatible_view_auto_run,
    save_view_auto_run,
    view_suggestion_meets_confidence,
)
from qc.run_view_auto_qc import load_view_records
from qc import evaluate_view_auto_qc
from prima.view_qc import (
    VIEW_LABEL_PASS,
    empty_view_qc_state,
    save_view_qc_state,
    set_view_label,
)


def view_id(index: int) -> str:
    return f"{index:064x}"


def test_view_prompt_removes_montage_cross_view_rule() -> None:
    prompt = auto_annotate_qc.build_target_prompt_text(
        prompt_mode="marker_classifier",
        tag_catalog=["vertical line (detector artifact)"],
        few_shot_examples=[],
        probe_tag="vertical line (detector artifact)",
        prompt_variant="confidence_specificity",
        input_level="view",
    )
    assert "Target view only" in prompt
    assert "this mammogram" in prompt
    assert "across views" not in prompt
    system = auto_annotate_qc.build_system_prompt(
        "marker_classifier", input_level="view"
    )
    assert "single target view" in system
    assert "four-view" not in system


def test_detector_boundary_prompt_defines_positive_and_negative_morphologies() -> None:
    prompt = auto_annotate_qc.build_target_prompt_text(
        prompt_mode="marker_classifier",
        tag_catalog=["vertical line (detector artifact)"],
        few_shot_examples=[],
        probe_tag="vertical line (detector artifact)",
        prompt_variant="detector_boundary_v2",
        input_level="view",
    )
    assert "detector-panel boundary" in prompt
    assert "repeated bright curved bands" in prompt
    assert "surgical scar/incision marker" in prompt
    assert "image-frame/crop borders" in prompt


def test_view_suggestion_confidence_threshold_is_explicit() -> None:
    record = {
        "image_path": "images/view.png",
        "suggestions": [
            {
                "tag": "vertical line (detector artifact)",
                "confidence": "medium",
            }
        ],
    }
    assert view_suggestion_meets_confidence(record, minimum_confidence="medium")
    assert not view_suggestion_meets_confidence(record, minimum_confidence="high")
    record["suggestions"][0].pop("confidence")
    with pytest.raises(ValueError, match="missing a valid confidence"):
        view_suggestion_meets_confidence(record, minimum_confidence="high")


def test_view_run_round_trip_and_resume_guard(tmp_path: Path) -> None:
    payload = new_view_auto_run(
        model="model@revision",
        prompt_variant="confidence_specificity",
        inference_settings={"temperature": 0.0},
    )
    payload["view_suggestions"] = {
        view_id(1): {
            "image_path": f"images/{view_id(1)}.png",
            "suggestions": [],
        }
    }
    path = tmp_path / "run.json"
    saved = save_view_auto_run(path, payload)
    assert load_view_auto_run(path) == saved
    assert path.stat().st_mode & 0o777 == 0o600

    changed = new_view_auto_run(
        model="another-model@revision",
        prompt_variant="confidence_specificity",
        inference_settings={"temperature": 0.0},
    )
    with pytest.raises(ValueError, match="model"):
        require_compatible_view_auto_run(saved, changed)


def test_view_manifest_loader_uses_relative_images(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_name = f"{view_id(1)}.png"
    Image.new("L", (8, 8)).save(image_dir / image_name)
    manifest = pd.DataFrame(
        [
            {
                "view_id": view_id(1),
                "image_path": f"images/{image_name}",
                "laterality": "R",
                "view": "MLO",
                "review_order": 1,
            }
        ]
    )
    manifest_path = tmp_path / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    records = load_view_records(manifest_path)
    assert records == [
        {
            "view_id": view_id(1),
            "image_path": str((image_dir / image_name).resolve()),
            "saved_image_path": f"images/{image_name}",
        }
    ]


def test_evaluator_refuses_partial_blinded_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = pd.DataFrame(
        [
            {
                "view_id": view_id(index),
                "image_path": f"images/{view_id(index)}.png",
                "laterality": "L",
                "view": "CC",
                "review_order": index,
                "stratum": "random",
            }
            for index in (1, 2)
        ]
    )
    manifest_path = tmp_path / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    state = set_view_label(empty_view_qc_state(), view_id(1), VIEW_LABEL_PASS)
    state_path = tmp_path / "state.json"
    save_view_qc_state(state_path, state)
    run = new_view_auto_run(
        model="model@revision",
        prompt_variant="confidence_specificity",
        inference_settings={"temperature": 0.0},
    )
    run["view_suggestions"] = {
        view_id(index): {
            "image_path": f"images/{view_id(index)}.png",
            "suggestions": [],
        }
        for index in (1, 2)
    }
    run_path = tmp_path / "run.json"
    save_view_auto_run(run_path, run)
    monkeypatch.setattr(
        evaluate_view_auto_qc,
        "parse_args",
        lambda: SimpleNamespace(
            manifest=manifest_path,
            state=state_path,
            run_file=run_path,
            out_dir=tmp_path / "evaluation",
            minimum_reject_confidence="high",
        ),
    )
    with pytest.raises(RuntimeError, match="1 remain"):
        evaluate_view_auto_qc.main()
