from __future__ import annotations

import pandas as pd

from qc.combine_mirai_input_hybrid_qc import build_fallback_decisions
from qc.evaluate_mirai_input_whole_exam_audit import evaluate_exam


def view_id(index: int) -> str:
    return f"{index:064x}"


def test_hybrid_fallback_stays_within_exact_slot() -> None:
    rows = pd.DataFrame(
        {
            "view_id": [view_id(1), view_id(2), view_id(3)],
            "audit_exam_id": ["exam"] * 3,
            "audit_group_id": ["slot"] * 3,
            "laterality": ["L"] * 3,
            "view": ["CC"] * 3,
            "selection_rank": [1, 2, 3],
            "candidate_count": [3, 3, 3],
            "sampling_stratum": ["challenge"] * 3,
        }
    )
    decisions = build_fallback_decisions(rows, target_present_ids={view_id(1)})
    decision = decisions.iloc[0]
    assert decision["fallback_status"] == "alternate_target_absent"
    assert decision["selected_view_id"] == view_id(2)
    assert decision["selected_candidate_rank"] == 2


def test_hybrid_fallback_records_exhaustion() -> None:
    rows = pd.DataFrame(
        {
            "view_id": [view_id(1), view_id(2)],
            "audit_exam_id": ["exam"] * 2,
            "audit_group_id": ["slot"] * 2,
            "laterality": ["R"] * 2,
            "view": ["MLO"] * 2,
            "selection_rank": [1, 2],
            "candidate_count": [2, 2],
            "sampling_stratum": ["challenge"] * 2,
        }
    )
    decisions = build_fallback_decisions(
        rows, target_present_ids={view_id(1), view_id(2)}
    )
    assert decisions.iloc[0]["fallback_status"] == "no_target_absent_candidate"
    assert pd.isna(decisions.iloc[0]["selected_view_id"])


def test_whole_exam_outcomes_distinguish_safe_route_from_false_rejection() -> None:
    safe = pd.DataFrame(
        {
            "selected_candidate_rank": [1, 1, 1, None],
            "decision_outcome": [
                "selected_target_absent",
                "selected_target_absent",
                "selected_target_absent",
                "safe_exhaustion",
            ],
        }
    )
    assert evaluate_exam(safe)["exam_outcome"] == "safe_routed_out"
    unsafe = safe.copy()
    unsafe.loc[3, "decision_outcome"] = "false_exhaustion"
    assert evaluate_exam(unsafe)["exam_outcome"] == "false_rejected"


def test_whole_exam_outcome_detects_unsafe_acceptance() -> None:
    rows = pd.DataFrame(
        {
            "selected_candidate_rank": [1, 1, 2, 1],
            "decision_outcome": [
                "selected_target_absent",
                "selected_target_absent",
                "selected_target_present",
                "selected_target_absent",
            ],
        }
    )
    result = evaluate_exam(rows)
    assert result["exam_accepted"]
    assert not result["exam_safe"]
    assert result["exam_outcome"] == "unsafe_accepted"
