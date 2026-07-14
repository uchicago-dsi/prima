from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from qc.build_mirai_input_whole_exam_audit import (
    audit_exam_id,
    audit_group_id,
    load_prior_exclusions,
    sample_audit_exams,
)


def test_whole_exam_audit_ids_are_stable_and_deidentified() -> None:
    assert audit_group_id("exam-1", "L", "CC") == audit_group_id("exam-1", "L", "CC")
    assert audit_exam_id("patient-1", "exam-1") == audit_exam_id("patient-1", "exam-1")
    assert "exam-1" not in audit_group_id("exam-1", "L", "CC")
    assert "patient-1" not in audit_exam_id("patient-1", "exam-1")


def test_prior_exclusions_include_whole_patients(tmp_path: Path) -> None:
    path = tmp_path / "source.parquet"
    pd.DataFrame(
        {
            "patient_id": ["patient-1", "patient-1"],
            "exam_id": ["exam-1", "exam-1"],
            "view_id": ["1" * 64, "2" * 64],
        }
    ).to_parquet(path, index=False)
    patients, exams, views = load_prior_exclusions([path])
    assert patients == {"patient-1"}
    assert exams == {"exam-1"}
    assert views == {"1" * 64, "2" * 64}


def test_sample_audit_exams_is_patient_disjoint() -> None:
    rows = []
    for index in range(20):
        rows.append(
            {
                "patient_id": f"patient-{index // 2}",
                "exam_id": f"exam-{index}",
                "candidate_count": 5 if index % 2 else 4,
                "seam_fallback_challenge": index < 4,
                "rank1_film_positive": 4 <= index < 8,
                "rank1_has_implant": 8 <= index < 12,
                "all_seam_original_absent": index >= 4,
            }
        )
    summary = pd.DataFrame(rows)
    counts = {
        "seam_fallback_challenge": 1,
        "film_exam_exclusion": 1,
        "implant_exam_exclusion": 1,
        "multi_candidate_control": 1,
        "single_candidate_control": 1,
    }
    sampled = sample_audit_exams(summary, counts=counts, seed=7)
    assert len(sampled) == 5
    assert sampled["patient_id"].nunique() == 5
    assert sampled["exam_id"].nunique() == 5
    assert sampled["sampling_stratum"].value_counts().eq(1).all()


def test_sample_audit_exams_fails_when_stratum_is_too_small() -> None:
    summary = pd.DataFrame(
        {
            "patient_id": ["patient-1"],
            "exam_id": ["exam-1"],
            "candidate_count": [4],
            "seam_fallback_challenge": [True],
            "rank1_film_positive": [False],
            "rank1_has_implant": [False],
            "all_seam_original_absent": [False],
        }
    )
    counts = {
        "seam_fallback_challenge": 2,
        "film_exam_exclusion": 0,
        "implant_exam_exclusion": 0,
        "multi_candidate_control": 0,
        "single_candidate_control": 0,
    }
    with pytest.raises(ValueError, match="only 1 patient-disjoint exams"):
        sample_audit_exams(summary, counts=counts, seed=7)
