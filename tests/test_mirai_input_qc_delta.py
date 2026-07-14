from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from qc.compare_mirai_input_qc_delta import build_comparison


def _write_evaluation(
    root: Path,
    *,
    false_negatives: list[str],
    false_positives: list[str],
    unsafe_slots: int,
    false_exhaustions: int,
    unsafe_exams: int,
) -> None:
    root.mkdir()
    tp = 10 - len(false_negatives)
    tn = 10 - len(false_positives)
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "target": "target",
                "views": {
                    "n": 20,
                    "tp": tp,
                    "fn": len(false_negatives),
                    "fp": len(false_positives),
                    "tn": tn,
                },
            }
        )
    )
    (root / "gate.json").write_text(
        json.dumps(
            {
                "observed": {
                    "view_sensitivity": tp / 10,
                    "view_specificity": tn / 10,
                    "unsafe_selected_slots": unsafe_slots,
                    "false_exhausted_slots": false_exhaustions,
                    "unsafe_accepted_exams": unsafe_exams,
                }
            }
        )
    )
    rows = [
        {
            "view_id": value * 64,
            "human_target_present": True,
            "model_target_present": False,
        }
        for value in false_negatives
    ]
    rows.extend(
        {
            "view_id": value * 64,
            "human_target_present": False,
            "model_target_present": True,
        }
        for value in false_positives
    )
    pd.DataFrame(
        rows,
        columns=["view_id", "human_target_present", "model_target_present"],
    ).to_parquet(root / "view_disagreements.parquet", index=False)


def _protocol() -> dict[str, object]:
    return {
        "arms": {
            "arm": {
                "target": "narrow target",
                "minimum_recovered_baseline_false_negatives": 1,
                "maximum_introduced_false_positives": 1,
            }
        },
        "shared_component_rules": {
            "maximum_increase_in_unsafe_selected_slots": 0,
            "maximum_increase_in_false_exhausted_slots": 0,
            "maximum_increase_in_unsafe_accepted_exams": 0,
        },
    }


def test_useful_delta_retains_baseline_and_component(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_evaluation(
        baseline,
        false_negatives=["1", "2"],
        false_positives=["a"],
        unsafe_slots=2,
        false_exhaustions=1,
        unsafe_exams=1,
    )
    _write_evaluation(
        candidate,
        false_negatives=["2"],
        false_positives=["a", "b"],
        unsafe_slots=1,
        false_exhaustions=1,
        unsafe_exams=0,
    )
    result = build_comparison(
        baseline_evaluation=baseline,
        candidate_evaluation=candidate,
        protocol=_protocol(),
        arm="arm",
    )
    assert result["passes_incremental_gate"]
    assert result["baseline"]["disposition"] == "active_and_retained"
    assert (
        result["candidate"]["component_disposition"]
        == "retain_for_combined_development_candidate"
    )


def test_failed_delta_rejects_only_component(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_evaluation(
        baseline,
        false_negatives=["1"],
        false_positives=[],
        unsafe_slots=1,
        false_exhaustions=0,
        unsafe_exams=1,
    )
    _write_evaluation(
        candidate,
        false_negatives=["1"],
        false_positives=["a", "b"],
        unsafe_slots=1,
        false_exhaustions=1,
        unsafe_exams=1,
    )
    result = build_comparison(
        baseline_evaluation=baseline,
        candidate_evaluation=candidate,
        protocol=_protocol(),
        arm="arm",
    )
    assert not result["passes_incremental_gate"]
    assert result["baseline"]["disposition"] == "active_and_retained"
    assert result["candidate"]["component_disposition"] == "reject_frozen_delta"
    assert (
        "the retained deterministic-plus-modular baseline"
        in result["falsification_scope"]["not_falsified"]
    )


def test_exact_required_recovery_passes_only_for_registered_view(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_evaluation(
        baseline,
        false_negatives=["1", "2"],
        false_positives=[],
        unsafe_slots=2,
        false_exhaustions=0,
        unsafe_exams=2,
    )
    _write_evaluation(
        candidate,
        false_negatives=["2"],
        false_positives=[],
        unsafe_slots=1,
        false_exhaustions=0,
        unsafe_exams=1,
    )
    protocol = _protocol()
    protocol["arms"]["arm"]["required_recovered_review_orders"] = [177]
    result = build_comparison(
        baseline_evaluation=baseline,
        candidate_evaluation=candidate,
        protocol=protocol,
        arm="arm",
        review_order_by_view_id={"1" * 64: 177, "2" * 64: 24},
    )
    assert result["passes_incremental_gate"]
    assert result["delta"]["recovered_baseline_false_negative_review_orders"] == [177]


def test_wrong_recovery_fails_exact_required_recovery_gate(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_evaluation(
        baseline,
        false_negatives=["1", "2"],
        false_positives=[],
        unsafe_slots=2,
        false_exhaustions=0,
        unsafe_exams=2,
    )
    _write_evaluation(
        candidate,
        false_negatives=["1"],
        false_positives=[],
        unsafe_slots=1,
        false_exhaustions=0,
        unsafe_exams=1,
    )
    protocol = _protocol()
    protocol["arms"]["arm"]["required_recovered_review_orders"] = [177]
    result = build_comparison(
        baseline_evaluation=baseline,
        candidate_evaluation=candidate,
        protocol=protocol,
        arm="arm",
        review_order_by_view_id={"1" * 64: 177, "2" * 64: 24},
    )
    assert not result["passes_incremental_gate"]
    assert result["delta"]["recovered_baseline_false_negative_review_orders"] == [24]


def test_exact_recovery_gate_requires_manifest_mapping(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_evaluation(
        baseline,
        false_negatives=["1"],
        false_positives=[],
        unsafe_slots=1,
        false_exhaustions=0,
        unsafe_exams=1,
    )
    _write_evaluation(
        candidate,
        false_negatives=[],
        false_positives=[],
        unsafe_slots=0,
        false_exhaustions=0,
        unsafe_exams=0,
    )
    protocol = _protocol()
    protocol["arms"]["arm"]["required_recovered_review_orders"] = [177]
    try:
        build_comparison(
            baseline_evaluation=baseline,
            candidate_evaluation=candidate,
            protocol=protocol,
            arm="arm",
        )
    except ValueError as error:
        assert "no manifest was provided" in str(error)
    else:
        raise AssertionError("exact recovery gate accepted a missing manifest")
