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
    view_suggestion_is_target_present,
)
from qc.run_view_auto_qc import load_target_prompt, load_view_records
from qc import evaluate_view_auto_qc
from prima.view_qc import (
    VIEW_LABEL_ABSENT,
    VIEW_LABEL_UNCERTAIN,
    empty_view_qc_state,
    save_view_qc_state,
    set_view_label,
)

TARGET = "test artifact"


def view_id(index: int) -> str:
    return f"{index:064x}"


def test_shared_view_prompt_is_target_agnostic() -> None:
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
    assert "straight, narrow" not in prompt
    assert "compression hardware" not in prompt
    system = auto_annotate_qc.build_system_prompt(
        "marker_classifier", input_level="view"
    )
    assert "single target view" in system
    assert "four-view" not in system

    generic = auto_annotate_qc.build_target_prompt_text(
        prompt_mode="marker_classifier",
        tag_catalog=["breast implant visible"],
        few_shot_examples=[],
        probe_tag="breast implant visible",
        prompt_variant="confidence_specificity",
        input_level="view",
    )
    assert "vertical line detector artifact" not in generic
    assert "named target only" in generic


def test_target_prompt_file_owns_artifact_specific_definition() -> None:
    prompt = load_target_prompt(
        Path("qc/targets/vertical_detector_seam_v1.txt"),
        target="vertical detector seam",
    )
    assert "detector-panel boundary" in prompt
    assert "repeated bright curved bands" in prompt
    assert "surgical scar/incision marker" in prompt
    assert "image-frame/crop borders" in prompt


def test_target_prompt_must_name_the_run_target(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text(
        "Target: different finding\n"
        "EVIDENCE: visible evidence\n"
        "ANSWER: YES or NO\n"
        "CONFIDENCE: high, medium, or low\n"
        "REVIEW: YES or NO\n"
    )
    with pytest.raises(ValueError, match="does not name --target"):
        load_target_prompt(prompt_path, target=TARGET)


def test_view_suggestion_confidence_threshold_is_explicit() -> None:
    record = {
        "image_path": "images/view.png",
        "suggestions": [
            {
                "tag": TARGET,
                "confidence": "medium",
            }
        ],
    }
    assert view_suggestion_is_target_present(
        record, target=TARGET, minimum_confidence="medium"
    )
    assert not view_suggestion_is_target_present(
        record, target=TARGET, minimum_confidence="high"
    )
    record["suggestions"][0].pop("confidence")
    with pytest.raises(ValueError, match="missing a valid confidence"):
        view_suggestion_is_target_present(
            record, target=TARGET, minimum_confidence="high"
        )


def test_view_run_round_trip_and_resume_guard(tmp_path: Path) -> None:
    payload = new_view_auto_run(
        target=TARGET,
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
        target=TARGET,
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


def test_vllm_view_examples_use_fixed_order_and_view_wording(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    target = tmp_path / "target.png"
    for path in (first, second, target):
        Image.new("L", (8, 8)).save(path)
    annotator = auto_annotate_qc.VLLMVisionAnnotator.__new__(
        auto_annotate_qc.VLLMVisionAnnotator
    )
    annotator.few_shot_exemplar_pool = [
        {
            "exam_id": "second",
            "image_path": str(second),
            "annotations": [],
            "few_shot_order": 2,
        },
        {
            "exam_id": "first",
            "image_path": str(first),
            "annotations": [TARGET],
            "few_shot_order": 1,
        },
    ]
    annotator.few_shot_examples = 2
    annotator.prompt_mode = "marker_classifier"
    annotator.probe_tag = TARGET
    annotator.input_level = "view"
    annotator.text_only_prompt = None

    selected = annotator._select_few_shot_examples("target")
    assert [record["exam_id"] for record in selected] == ["first", "second"]
    messages = annotator._build_messages(
        image_path=target,
        target_prompt_text="classify the target view",
        few_shot_examples=selected,
    )
    exemplar_text = messages[1]["content"][1]["text"]
    assert "for this view" in exemplar_text
    assert "montage" not in exemplar_text


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
    state = set_view_label(empty_view_qc_state(TARGET), view_id(1), VIEW_LABEL_ABSENT)
    state_path = tmp_path / "state.json"
    save_view_qc_state(state_path, state)
    run = new_view_auto_run(
        target=TARGET,
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
            minimum_present_confidence="high",
        ),
    )
    with pytest.raises(RuntimeError, match="1 remain"):
        evaluate_view_auto_qc.main()

    state = set_view_label(state, view_id(2), VIEW_LABEL_UNCERTAIN)
    save_view_qc_state(state_path, state)
    with pytest.raises(RuntimeError, match="1 remain uncertain"):
        evaluate_view_auto_qc.main()
