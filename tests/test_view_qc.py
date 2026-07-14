from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from prima.view_fallback import (
    align_candidates_to_selected_views,
    choose_exact_slot_views,
)
from prima.view_qc import (
    VIEW_LABEL_ABSENT,
    VIEW_LABEL_PRESENT,
    default_view_qc_events_path,
    empty_view_qc_state,
    initialize_view_qc_event_log,
    load_view_qc_events,
    load_view_qc_state,
    reconcile_view_qc_campaign_state,
    record_view_qc_label,
    render_dicom_view_png,
    replay_view_qc_events,
    save_view_qc_state,
    set_view_label,
    summarize_view_qc_state,
    validate_rendered_view_png,
)
from qc.build_view_qc_pilot import sample_views
from qc.build_ranked_view_qc_pilot import run_from_args as build_ranked_pilot
from qc.init_view_qc_review import initialize_review
from qc.view_qc_gallery import (
    DEFAULT_NEGATIVE_LABEL,
    DEFAULT_NEGATIVE_SHORTCUT,
    DEFAULT_POSITIVE_LABEL,
    DEFAULT_POSITIVE_SHORTCUT,
    DEFAULT_REVIEW_INSTRUCTION,
    DEFAULT_REVIEW_PORT,
    HTML,
    load_review_rubric,
    load_review_items,
    normalize_shortcut,
    normalize_ui_text,
)


def view_id(index: int) -> str:
    return f"{index:064x}"


def test_view_qc_state_uses_one_manifest_denominator(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = empty_view_qc_state("test artifact")
    state = set_view_label(state, view_id(1), VIEW_LABEL_ABSENT)
    state = set_view_label(state, view_id(2), VIEW_LABEL_PRESENT)
    state = set_view_label(
        state,
        view_id(3),
        VIEW_LABEL_ABSENT,
        low_confidence=True,
    )
    save_view_qc_state(path, state)

    loaded = load_view_qc_state(path)
    assert summarize_view_qc_state(
        loaded, [view_id(1), view_id(2), view_id(3), view_id(4)]
    ) == {
        "total": 4,
        "reviewed": 3,
        "remaining": 1,
        "present": 1,
        "absent": 2,
        "low_confidence": 1,
    }
    assert path.stat().st_mode & 0o777 == 0o600


def test_view_qc_target_is_required() -> None:
    with pytest.raises(ValueError, match="must be a string"):
        empty_view_qc_state(None)
    with pytest.raises(ValueError, match="cannot be empty"):
        empty_view_qc_state("  ")


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


def test_generic_review_initializer_binds_target_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("L", (8, 8)).save(image_dir / f"{view_id(1)}.png")
    manifest_path = tmp_path / "manifest.parquet"
    pd.DataFrame(
        [
            {
                "view_id": view_id(1),
                "image_path": f"images/{view_id(1)}.png",
                "laterality": "L",
                "view": "CC",
                "review_order": 1,
            }
        ]
    ).to_parquet(manifest_path, index=False)
    state_path = tmp_path / "state.json"

    assert initialize_review(manifest_path, state_path, "  test   artifact ") == 1
    assert load_view_qc_state(state_path) == empty_view_qc_state("test artifact")
    events_path = default_view_qc_events_path(state_path)
    assert events_path.is_file()
    assert events_path.stat().st_mode & 0o777 == 0o600
    assert load_view_qc_events(events_path) == []
    with pytest.raises(FileExistsError, match="refusing to replace"):
        initialize_review(manifest_path, state_path, "another artifact")


def test_view_qc_event_log_is_append_only_hash_chained_and_replayable(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "view_qc_state.json"
    events_path = default_view_qc_events_path(state_path)
    manifest_ids = [view_id(1), view_id(2)]
    state = save_view_qc_state(state_path, empty_view_qc_state("test artifact"))
    initialize_view_qc_event_log(events_path, state)

    state = record_view_qc_label(
        state_path=state_path,
        events_path=events_path,
        manifest_view_ids=manifest_ids,
        view_id=view_id(1),
        label=VIEW_LABEL_PRESENT,
        low_confidence=False,
        reviewer="reviewer-1",
    )
    state = record_view_qc_label(
        state_path=state_path,
        events_path=events_path,
        manifest_view_ids=manifest_ids,
        view_id=view_id(1),
        label=VIEW_LABEL_ABSENT,
        low_confidence=True,
        reviewer="reviewer-1",
    )
    state = record_view_qc_label(
        state_path=state_path,
        events_path=events_path,
        manifest_view_ids=manifest_ids,
        view_id=view_id(1),
        label=None,
        low_confidence=None,
        reviewer="reviewer-1",
    )

    events = load_view_qc_events(events_path)
    assert len(events) == 3
    assert [event["previous_label"] for event in events] == [
        None,
        VIEW_LABEL_PRESENT,
        VIEW_LABEL_ABSENT,
    ]
    assert [event["label"] for event in events] == [
        VIEW_LABEL_PRESENT,
        VIEW_LABEL_ABSENT,
        None,
    ]
    assert [event["previous_low_confidence"] for event in events] == [
        None,
        False,
        True,
    ]
    assert [event["low_confidence"] for event in events] == [
        False,
        True,
        None,
    ]
    assert all(event["reviewer"] == "reviewer-1" for event in events)
    assert replay_view_qc_events("test artifact", events) == state
    assert state == empty_view_qc_state("test artifact")
    assert events_path.stat().st_mode & 0o777 == 0o600


def test_view_qc_event_log_detects_tampering(tmp_path: Path) -> None:
    state_path = tmp_path / "view_qc_state.json"
    events_path = default_view_qc_events_path(state_path)
    state = save_view_qc_state(state_path, empty_view_qc_state("test artifact"))
    initialize_view_qc_event_log(events_path, state)
    record_view_qc_label(
        state_path=state_path,
        events_path=events_path,
        manifest_view_ids=[view_id(1)],
        view_id=view_id(1),
        label=VIEW_LABEL_PRESENT,
        low_confidence=False,
        reviewer="reviewer-1",
    )
    text = events_path.read_text().replace('"label":"present"', '"label":"absent"')
    events_path.write_text(text)

    with pytest.raises(ValueError, match="event hash is invalid"):
        load_view_qc_events(events_path)


def test_imported_state_is_explicit_in_event_history(tmp_path: Path) -> None:
    state = set_view_label(
        empty_view_qc_state("test artifact"),
        view_id(1),
        VIEW_LABEL_PRESENT,
        low_confidence=True,
    )
    events_path = tmp_path / "events.jsonl"
    events = initialize_view_qc_event_log(
        events_path, state, import_reviewer="system:test-import"
    )

    assert len(events) == 1
    assert events[0]["event_type"] == "state_import"
    assert events[0]["reviewer"] == "system:test-import"
    assert replay_view_qc_events("test artifact", events) == state


def test_event_replay_recovers_state_after_interrupted_projection_write(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "view_qc_state.json"
    events_path = default_view_qc_events_path(state_path)
    manifest_ids = [view_id(1)]
    empty = save_view_qc_state(state_path, empty_view_qc_state("test artifact"))
    initialize_view_qc_event_log(events_path, empty)
    committed = record_view_qc_label(
        state_path=state_path,
        events_path=events_path,
        manifest_view_ids=manifest_ids,
        view_id=view_id(1),
        label=VIEW_LABEL_PRESENT,
        low_confidence=False,
        reviewer="reviewer-1",
    )

    save_view_qc_state(state_path, empty)
    recovered = reconcile_view_qc_campaign_state(
        state_path, load_view_qc_events(events_path), manifest_ids
    )

    assert recovered == committed
    assert load_view_qc_state(state_path) == committed


def test_event_replay_refuses_divergent_state(tmp_path: Path) -> None:
    state_path = tmp_path / "view_qc_state.json"
    events_path = default_view_qc_events_path(state_path)
    manifest_ids = [view_id(1)]
    empty = save_view_qc_state(state_path, empty_view_qc_state("test artifact"))
    initialize_view_qc_event_log(events_path, empty)
    record_view_qc_label(
        state_path=state_path,
        events_path=events_path,
        manifest_view_ids=manifest_ids,
        view_id=view_id(1),
        label=VIEW_LABEL_PRESENT,
        low_confidence=False,
        reviewer="reviewer-1",
    )
    divergent = set_view_label(empty, view_id(1), VIEW_LABEL_ABSENT)
    save_view_qc_state(state_path, divergent)

    with pytest.raises(ValueError, match="state diverges"):
        reconcile_view_qc_campaign_state(
            state_path, load_view_qc_events(events_path), manifest_ids
        )


def test_gallery_has_explicit_completion_state() -> None:
    assert "Review complete" in HTML
    assert "| COMPLETE" in HTML
    assert 'id="end-marker"' in HTML
    assert "✓ End of batch" in HTML
    assert "End reached" not in HTML
    assert 'id="review-unsure"' in HTML
    assert "Review low confidence (" in HTML
    assert "function startUnsureReview()" in HTML
    assert "low-confidence review pass complete" in HTML


def test_gallery_keeps_stable_port_and_smaller_image() -> None:
    assert DEFAULT_REVIEW_PORT == 8767
    assert "max-width: 70vw; max-height: 65vh" in HTML
    assert HTML.index('id="controls"') < HTML.index("<main>")
    assert "height: calc(100vh" not in HTML
    assert "overflow: hidden" not in HTML
    assert "overflow-y: auto" in HTML
    assert "caret-color: transparent" in HTML
    assert 'id="save-status"' in HTML
    assert "event.preventDefault()" in HTML
    assert "'Saved: ' + negativeLabel.toLowerCase()" in HTML
    assert "'Saved: ' + positiveLabel.toLowerCase()" in HTML
    assert "Not present [n]" in HTML
    assert "Present [y]" in HTML
    assert "Low confidence: off [u]" in HTML
    assert 'id="rubric-panel"' in HTML
    assert "reviewRubric = config.review_rubric" in HTML
    assert "record?.low_confidence === true" in HTML
    assert "pendingLowConfidence" in HTML
    assert "Target: " in HTML
    assert "fetch('/api/config')" in HTML
    assert "negativeLabel = config.negative_label" in HTML
    assert "positiveLabel = config.positive_label" in HTML
    assert "negativeShortcut = config.negative_shortcut" in HTML
    assert "positiveShortcut = config.positive_shortcut" in HTML
    assert "reviewInstruction = config.review_instruction" in HTML


def test_gallery_supports_generic_binary_decision_labels() -> None:
    assert DEFAULT_NEGATIVE_LABEL == "Not present"
    assert DEFAULT_POSITIVE_LABEL == "Present"
    assert DEFAULT_NEGATIVE_SHORTCUT == "n"
    assert DEFAULT_POSITIVE_SHORTCUT == "y"
    assert DEFAULT_REVIEW_INSTRUCTION.startswith("Decide only whether")
    assert (
        normalize_ui_text("  Use   for Mirai ", field="label", max_length=80)
        == "Use for Mirai"
    )
    with pytest.raises(ValueError, match="cannot be empty"):
        normalize_ui_text("  ", field="label", max_length=80)
    assert normalize_shortcut(" P ", field="shortcut") == "p"
    with pytest.raises(ValueError, match="one letter or number"):
        normalize_shortcut("!", field="shortcut")


def test_gallery_loads_multiline_review_rubric(tmp_path: Path) -> None:
    path = tmp_path / "rubric.txt"
    path.write_text("Use:\n- ordinary image\n\nDo not use:\n- film\n")
    assert load_review_rubric(path) == ("Use:\n- ordinary image\n\nDo not use:\n- film")
    assert load_review_rubric(None) == ""


def test_gallery_tracks_session_annotation_rate_after_first_save() -> None:
    assert 'id="session-rate"' in HTML
    assert 'id="reset-session"' in HTML
    assert "Rate and ETA start with your first saved annotation." in HTML
    assert "function recordSessionAnnotation(" in HTML
    assert "recordSessionAnnotation(item.view_id, label, actionStartedAtMs);" in HTML
    assert "sessionAnnotatedViewIds = new Set()" in HTML
    assert "sessionAnnotatedViewIds.add(viewId)" in HTML
    assert "sessionStorage.setItem(" in HTML
    assert "sessionStorage.getItem(" in HTML
    assert "unique view" in HTML
    assert "views/min" in HTML
    assert "ETA measuring" in HTML
    assert "ETA done" in HTML
    assert "summary.remaining / ratePerMinute" in HTML
    assert "Reset timer" in HTML


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


def test_ranked_pilot_is_target_bound_and_deidentified(tmp_path: Path) -> None:
    candidates = []
    score_rows = []
    rendered_rows = []
    rendered_root = tmp_path / "rendered"
    rendered_images = rendered_root / "images"
    rendered_images.mkdir(parents=True)
    slots = [("L", "CC"), ("R", "CC"), ("L", "MLO"), ("R", "MLO")]
    for index, (laterality, view) in enumerate(slots, start=1):
        identifier = view_id(index)
        candidates.append(
            {
                "exam_id": f"exam-{index}",
                "laterality": laterality,
                "view": view,
                "sop_instance_uid": f"sop-{index}",
                "sha256": identifier,
                "selection_rank": 1,
                "is_selected": True,
            }
        )
        score_rows.append(
            {
                "exam_id": f"exam-{index}",
                "sop_instance_uid": f"sop-{index}",
                "score_a": float(index),
                "score_b": float(index * 2),
                "load_error": None,
                "laterality": "score-table-value-must-not-shadow-candidate",
                "view": "score-table-value-must-not-shadow-candidate",
            }
        )
        Image.new("L", (8, 8), color=index).save(rendered_images / f"{identifier}.png")
        rendered_rows.append(
            {
                "view_id": identifier,
                "image_path": f"images/{identifier}.png",
                "laterality": laterality,
                "view": view,
                "review_order": index,
            }
        )
    candidates_path = tmp_path / "candidates.parquet"
    pd.DataFrame(candidates).to_parquet(candidates_path, index=False)
    scores_path = tmp_path / "scores.csv"
    pd.DataFrame(score_rows).to_csv(scores_path, index=False)
    rendered_manifest = rendered_root / "manifest.parquet"
    pd.DataFrame(rendered_rows).to_parquet(rendered_manifest, index=False)
    out_dir = tmp_path / "pilot"

    manifest = build_ranked_pilot(
        SimpleNamespace(
            scores=scores_path,
            score_columns="score_a,score_b",
            score_top_k=1,
            candidates=candidates_path,
            rendered_manifest=rendered_manifest,
            out_dir=out_dir,
            target="test artifact",
            enriched_count=1,
            random_count=1,
            seed=7,
            max_render_pixels=100,
        )
    )

    assert set(manifest["stratum"]) == {"score_enriched", "random_control"}
    assert set(manifest.columns) == {
        "view_id",
        "image_path",
        "laterality",
        "view",
        "review_order",
        "stratum",
    }
    assert load_view_qc_state(out_dir / "view_qc_state.json")["target"] == (
        "test artifact"
    )
    assert all((out_dir / path).is_file() for path in manifest["image_path"])


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
    assert output.stat().st_mode & 0o777 == 0o600
    assert validate_rendered_view_png(output, max_pixels=5_000)


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
            view_id(1): VIEW_LABEL_PRESENT,
            view_id(2): VIEW_LABEL_ABSENT,
            view_id(3): VIEW_LABEL_ABSENT,
        },
    ).set_index(["laterality", "view"])

    left = decisions.loc[("L", "CC")]
    right = decisions.loc[("R", "CC")]
    assert left["fallback_status"] == "alternate_target_absent"
    assert left["selected_view_id"] == view_id(2)
    assert left["selected_candidate_rank"] == 2
    assert right["fallback_status"] == "original_target_absent"
    assert right["selected_view_id"] == view_id(3)


def test_unreviewed_target_keeps_slot_unresolved() -> None:
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
            }
        ]
    )
    decision = choose_exact_slot_views(candidates, {}).iloc[0]
    assert decision["fallback_status"] == "unresolved_candidates"
    assert pd.isna(decision["selected_view_id"])
    assert decision["reviewed_candidate_count"] == 0


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
