from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qc.run_mirai_embedding_qc import (
    choose_c,
    confusion_counts,
    cross_validated_classifier,
    rates_from_counts,
    require_patient_exam_disjoint,
    select_sensitivity_threshold,
)


def test_sensitivity_threshold_is_largest_inclusive_cutoff() -> None:
    scores = [0.95, 0.8, 0.6, 0.4, 0.9, 0.7]
    labels = [1, 1, 1, 1, 0, 0]
    threshold = select_sensitivity_threshold(scores, labels, minimum_sensitivity=0.75)
    assert threshold == pytest.approx(0.6)
    counts = confusion_counts(labels, scores, threshold)
    assert counts == {
        "true_positive": 3,
        "false_negative": 1,
        "false_positive": 2,
        "true_negative": 0,
    }
    assert rates_from_counts(counts)["sensitivity"] == pytest.approx(0.75)


def test_sensitivity_threshold_keeps_positive_ties() -> None:
    threshold = select_sensitivity_threshold(
        [0.8, 0.5, 0.5, 0.1], [1, 1, 1, 0], minimum_sensitivity=2 / 3
    )
    assert threshold == pytest.approx(0.5)
    assert (
        confusion_counts([1, 1, 1, 0], [0.8, 0.5, 0.5, 0.1], threshold)["true_positive"]
        == 3
    )


def test_choose_c_uses_smallest_c_for_auc_tie() -> None:
    assert choose_c(
        [
            {"c": 1.0, "mean_roc_auc": 0.8},
            {"c": 0.1, "mean_roc_auc": 0.8},
            {"c": 0.01, "mean_roc_auc": 0.7},
        ]
    ) == pytest.approx(0.1)


def test_cross_validated_classifier_is_deterministic() -> None:
    generator = np.random.default_rng(9)
    labels = np.asarray([0] * 20 + [1] * 20)
    features = generator.normal(size=(40, 8))
    features[:, 0] += labels * 1.5
    first = cross_validated_classifier(
        features, labels, c_grid=[0.01, 0.1], folds=4, seed=3
    )
    second = cross_validated_classifier(
        features, labels, c_grid=[0.01, 0.1], folds=4, seed=3
    )
    assert first[0] == second[0]
    np.testing.assert_allclose(first[1], second[1])
    assert first[2] == second[2]


def test_patient_exam_disjointness_fails_on_shared_patient() -> None:
    development = pd.DataFrame({"patient_id": ["one"], "exam_id": ["development"]})
    audit = pd.DataFrame({"patient_id": ["one"], "exam_id": ["audit"]})
    with pytest.raises(ValueError, match="share patients"):
        require_patient_exam_disjoint(development, audit)


def test_patient_exam_disjointness_reports_only_counts() -> None:
    development = pd.DataFrame({"patient_id": ["one", "two"], "exam_id": ["a", "b"]})
    audit = pd.DataFrame({"patient_id": ["three"], "exam_id": ["c"]})
    assert require_patient_exam_disjoint(development, audit) == {
        "development_patients": 2,
        "development_exams": 2,
        "audit_patients": 1,
        "audit_exams": 1,
    }
