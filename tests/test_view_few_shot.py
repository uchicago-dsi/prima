from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

from prima.view_auto_qc import (
    load_view_auto_run,
    new_view_auto_run,
    save_view_auto_run,
)
from prima.view_few_shot import load_view_few_shot_manifest, sha256_file
from prima.view_qc import (
    VIEW_LABEL_ABSENT,
    VIEW_LABEL_PRESENT,
    empty_view_qc_state,
    initialize_view_qc_event_log,
    load_view_qc_state,
    save_view_qc_state,
    set_view_label,
)
from qc.build_view_few_shot_experiment import run_from_args

TARGET = "test artifact"


def view_id(index: int) -> str:
    return f"{index:064x}"


def build_completed_campaign(root: Path) -> tuple[Path, Path]:
    images = root / "images"
    images.mkdir(parents=True)
    rows = []
    state = empty_view_qc_state(TARGET)
    labels = [
        VIEW_LABEL_PRESENT,
        VIEW_LABEL_ABSENT,
        VIEW_LABEL_PRESENT,
        VIEW_LABEL_ABSENT,
    ]
    for order, label in enumerate(labels, start=1):
        identifier = view_id(order)
        image_path = images / f"{identifier}.png"
        Image.new("L", (8, 8), color=order).save(image_path)
        rows.append(
            {
                "view_id": identifier,
                "image_path": f"images/{identifier}.png",
                "laterality": "L" if order % 2 else "R",
                "view": "CC" if order % 2 else "MLO",
                "review_order": order,
                "stratum": "development",
                "source_identifier": f"restricted-{order}",
            }
        )
        state = set_view_label(state, identifier, label)
    pd.DataFrame(rows).to_parquet(root / "manifest.parquet", index=False)
    state_path = root / "view_qc_state.json"
    saved_state = save_view_qc_state(state_path, state)
    initialize_view_qc_event_log(root / "view_qc_events.jsonl", saved_state)

    run = new_view_auto_run(
        target=TARGET,
        model="model@revision",
        prompt_variant="confidence_specificity",
        inference_settings={"temperature": 0.0},
    )
    run["view_suggestions"] = {
        row["view_id"]: {"image_path": row["image_path"], "suggestions": []}
        for row in rows
    }
    run_path = root / "baseline_run.json"
    save_view_auto_run(run_path, run)
    return state_path, run_path


def test_build_and_load_ordered_balanced_view_exemplars(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    _state_path, run_path = build_completed_campaign(campaign)
    out_dir = tmp_path / "experiment"

    result = run_from_args(
        SimpleNamespace(
            campaign_dir=campaign,
            baseline_run=run_path,
            out_dir=out_dir,
            target=TARGET,
            example=[(1, "positive morphology"), (2, "negative look-alike")],
        )
    )

    assert result == {"examples": 2, "evaluation": 2}
    exemplars, metadata = load_view_few_shot_manifest(
        out_dir / "exemplars" / "manifest.parquet",
        target=TARGET,
    )
    assert [record["few_shot_order"] for record in exemplars] == [1, 2]
    assert exemplars[0]["annotations"] == [TARGET]
    assert exemplars[1]["annotations"] == []
    assert metadata["few_shot_example_count"] == 2
    assert metadata["few_shot_examples"][0]["image_sha256"] == sha256_file(
        out_dir / "exemplars" / "images" / f"{view_id(1)}.png"
    )

    evaluation = pd.read_parquet(out_dir / "evaluation_inputs" / "manifest.parquet")
    assert set(evaluation["view_id"]) == {view_id(3), view_id(4)}
    assert "source_identifier" not in evaluation
    evaluation_state = load_view_qc_state(
        out_dir / "evaluation_inputs" / "view_qc_state.json"
    )
    assert set(evaluation_state["labels"]) == {view_id(3), view_id(4)}
    baseline = load_view_auto_run(out_dir / "evaluation_inputs" / "baseline_run.json")
    assert set(baseline["view_suggestions"]) == {view_id(3), view_id(4)}
    with pytest.raises(ValueError, match="overlap"):
        load_view_few_shot_manifest(
            out_dir / "exemplars" / "manifest.parquet",
            target=TARGET,
            excluded_view_ids=[view_id(1)],
        )


def test_experiment_requires_a_completed_binary_campaign(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    state_path, run_path = build_completed_campaign(campaign)
    state = set_view_label(load_view_qc_state(state_path), view_id(4), None)
    save_view_qc_state(state_path, state)
    events_path = campaign / "view_qc_events.jsonl"
    events_path.unlink()
    initialize_view_qc_event_log(events_path, state)

    with pytest.raises(ValueError, match="must be complete with binary labels"):
        run_from_args(
            SimpleNamespace(
                campaign_dir=campaign,
                baseline_run=run_path,
                out_dir=tmp_path / "experiment",
                target=TARGET,
                example=[(1, "positive"), (2, "negative")],
            )
        )
