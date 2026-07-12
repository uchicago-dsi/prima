from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from prima.view_auto_qc import new_view_auto_run, save_view_auto_run
from qc.merge_view_auto_qc import merge_view_runs
from qc.prepare_view_candidate_inference import stable_render_shard
from qc.validate_view_candidate_render import run_from_args as validate_render
from qc.build_auto_qc_fallback_audit import audit_group_id
from qc.evaluate_auto_qc_fallback_audit import evaluate_group
from submit_view_auto_qc_campaign import parse_partition_plan


def view_id(index: int) -> str:
    return f"{index:064x}"


def test_render_sharding_is_stable() -> None:
    assert stable_render_shard("source/exam.tar.zst", 16) == stable_render_shard(
        "source/exam.tar.zst", 16
    )
    assert 0 <= stable_render_shard("source/exam.tar.zst", 16) < 16


def test_partition_plan_expands_exact_lane_counts() -> None:
    assert parse_partition_plan("siweiq:2,catherineq:1") == [
        "siweiq",
        "siweiq",
        "catherineq",
    ]


def test_fallback_audit_group_ids_are_stable_and_deidentified() -> None:
    first = audit_group_id("exam-1", "L", "CC")
    assert first == audit_group_id("exam-1", "L", "CC")
    assert len(first) == 64
    assert "exam-1" not in first


def test_fallback_audit_scores_ranked_replacement_sequence() -> None:
    rows = pd.DataFrame(
        {
            "stratum": ["alternate_pass", "alternate_pass", "alternate_pass"],
            "selection_rank": [1, 2, 3],
            "human_positive": [True, True, False],
        }
    )
    assert evaluate_group(rows)[0]
    rows.loc[1, "human_positive"] = False
    assert not evaluate_group(rows)[0]


def test_merge_view_runs_requires_exact_disjoint_coverage(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        [
            {
                "view_id": view_id(index),
                "image_path": f"images/{view_id(index)}.png",
                "laterality": "L",
                "view": "CC",
                "review_order": index,
            }
            for index in (1, 2)
        ]
    )
    manifest_path = tmp_path / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    base = new_view_auto_run(
        model="model@revision",
        prompt_variant="confidence_specificity",
        inference_settings={"frozen": True},
    )
    run_paths = []
    for index in (1, 2):
        payload = {
            **base,
            "run_id": f"shard-{index}",
            "view_suggestions": {
                view_id(index): {
                    "image_path": f"images/{view_id(index)}.png",
                    "suggestions": [],
                }
            },
        }
        run_path = tmp_path / f"shard_{index:03d}.json"
        save_view_auto_run(run_path, payload)
        run_paths.append(run_path)

    output = tmp_path / "merged.json"
    merged = merge_view_runs(manifest_path, run_paths, output)
    assert set(merged["view_suggestions"]) == {view_id(1), view_id(2)}
    assert output.stat().st_mode & 0o777 == 0o600


def test_render_validation_excludes_explicit_decode_failure(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "render_markers").mkdir()
    manifest = pd.DataFrame(
        [
            {
                "view_id": view_id(index),
                "image_path": f"images/{view_id(index)}.png",
                "laterality": "L",
                "view": "CC",
                "review_order": index,
            }
            for index in (1, 2)
        ]
    )
    manifest.to_parquet(tmp_path / "manifest.parquet", index=False)
    (tmp_path / "campaign.json").write_text(
        json.dumps(
            {
                "candidate_rows": 2,
                "render_shards": 1,
                "inference_shards": 1,
                "max_render_pixels": 100,
            }
        )
    )
    (tmp_path / "render_markers" / "shard_000.json").write_text(
        json.dumps(
            {
                "shard_index": 0,
                "failures": [{"view_id": view_id(2), "error_type": "ValueError"}],
            }
        )
    )
    Image.new("L", (8, 8)).save(tmp_path / "images" / f"{view_id(1)}.png")

    assert validate_render(argparse.Namespace(campaign_dir=tmp_path)) == 0
    inference = pd.read_parquet(tmp_path / "inference_manifest.parquet")
    complete = json.loads((tmp_path / "render_complete.json").read_text())
    assert inference["view_id"].tolist() == [view_id(1)]
    assert complete["failed_view_ids"] == [view_id(2)]
