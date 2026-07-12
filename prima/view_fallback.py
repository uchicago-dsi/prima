"""Exact-slot mammography view fallback after view-level QC."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from prima.view_qc import VIEW_LABEL_PASS, VALID_VIEW_LABELS, normalize_view_id

REQUIRED_CANDIDATE_COLUMNS = {
    "exam_id",
    "laterality",
    "view",
    "sop_instance_uid",
    "sha256",
    "selection_rank",
    "is_selected",
}

REQUIRED_SELECTED_COLUMNS = {
    "exam_id",
    "laterality",
    "view",
    "sop_instance_uid",
    "sha256",
}


def validate_candidate_table(candidates: pd.DataFrame, context: str) -> None:
    """Validate exact-slot candidate ranking invariants."""
    missing = sorted(REQUIRED_CANDIDATE_COLUMNS - set(candidates.columns))
    if missing:
        raise ValueError(f"{context} is missing columns: {', '.join(missing)}")
    if candidates.empty:
        raise ValueError(f"{context} is empty")
    if not candidates["laterality"].isin(["L", "R"]).all():
        raise ValueError(f"{context} contains invalid laterality")
    if not candidates["view"].isin(["CC", "MLO"]).all():
        raise ValueError(f"{context} contains invalid view")
    if candidates.duplicated(["exam_id", "sop_instance_uid"]).any():
        raise ValueError(f"{context} contains duplicate exam/SOP rows")
    ranks = pd.to_numeric(candidates["selection_rank"], errors="raise").astype(int)
    if (ranks <= 0).any():
        raise ValueError(f"{context} contains non-positive selection ranks")

    work = candidates.assign(selection_rank=ranks)
    for _slot, rows in work.groupby(["exam_id", "laterality", "view"], sort=False):
        expected = list(range(1, len(rows) + 1))
        if sorted(rows["selection_rank"].tolist()) != expected:
            raise ValueError(f"{context} has a non-contiguous slot ranking")
        selected = rows[rows["is_selected"].astype(bool)]
        if len(selected) != 1 or int(selected.iloc[0]["selection_rank"]) != 1:
            raise ValueError(
                f"{context} must have exactly one rank-1 selected candidate per slot"
            )


def normalize_view_labels(labels: Mapping[object, object]) -> dict[str, str]:
    """Normalize a view_id -> binary QC label mapping."""
    normalized: dict[str, str] = {}
    for raw_view_id, raw_label in labels.items():
        view_id = normalize_view_id(raw_view_id)
        label = str(raw_label).strip()
        if label not in VALID_VIEW_LABELS:
            raise ValueError(f"unsupported QC label for view {view_id[:12]}")
        normalized[view_id] = label
    return normalized


def align_candidates_to_selected_views(
    candidates: pd.DataFrame,
    selected_views: pd.DataFrame,
    *,
    context: str = "view_candidates.parquet",
) -> pd.DataFrame:
    """Re-rank candidates so the authoritative production selection is rank 1."""
    validate_candidate_table(candidates, context)
    missing = sorted(REQUIRED_SELECTED_COLUMNS - set(selected_views.columns))
    if missing:
        raise ValueError(
            "authoritative selected views are missing columns: " + ", ".join(missing)
        )
    if selected_views.empty:
        raise ValueError("authoritative selected views are empty")
    slot_columns = ["exam_id", "laterality", "view"]
    if selected_views.duplicated(slot_columns).any():
        raise ValueError("authoritative selected views contain duplicate slots")

    selected = selected_views[slot_columns + ["sop_instance_uid", "sha256"]].copy()
    selected = selected.rename(
        columns={
            "sop_instance_uid": "authoritative_sop_instance_uid",
            "sha256": "authoritative_sha256",
        }
    )
    work = candidates.merge(
        selected, on=slot_columns, how="left", validate="many_to_one"
    )
    if work["authoritative_sha256"].isna().any():
        raise ValueError("candidate table contains slots outside authoritative views")
    authoritative_slots = set(map(tuple, selected[slot_columns].to_numpy()))
    candidate_slots = set(map(tuple, work[slot_columns].drop_duplicates().to_numpy()))
    if authoritative_slots != candidate_slots:
        raise ValueError(
            "authoritative views and candidate table cover different slots"
        )

    source_match = (
        work["sha256"].astype(str).str.lower()
        == work["authoritative_sha256"].astype(str).str.lower()
    ) & (
        work["sop_instance_uid"].astype(str)
        == work["authoritative_sop_instance_uid"].astype(str)
    )
    matches_per_slot = source_match.groupby(
        [work[column] for column in slot_columns], sort=False
    ).sum()
    if not (matches_per_slot == 1).all():
        raise ValueError(
            "every authoritative selected source must occur exactly once among candidates"
        )

    work["_authoritative_order"] = (~source_match).astype(int)
    work["_prior_rank"] = pd.to_numeric(work["selection_rank"], errors="raise").astype(
        int
    )
    work = work.sort_values(
        slot_columns + ["_authoritative_order", "_prior_rank"], kind="stable"
    )
    work["selection_rank"] = (
        work.groupby(slot_columns, sort=False).cumcount().add(1).astype(int)
    )
    work["is_selected"] = source_match.loc[work.index].astype(bool)
    aligned = work.drop(
        columns=[
            "authoritative_sop_instance_uid",
            "authoritative_sha256",
            "_authoritative_order",
            "_prior_rank",
        ]
    ).reset_index(drop=True)
    validate_candidate_table(aligned, "aligned view candidates")
    return aligned


def choose_exact_slot_views(
    candidates: pd.DataFrame,
    labels: Mapping[object, object],
    *,
    context: str = "view_candidates.parquet",
) -> pd.DataFrame:
    """Choose the first explicitly passing candidate within every exact slot.

    An unreviewed original or alternate never counts as a pass. The function
    records unresolved and exhausted slots rather than substituting another
    laterality or projection.
    """
    validate_candidate_table(candidates, context)
    normalized_labels = normalize_view_labels(labels)
    work = candidates.copy()
    work["view_id"] = work["sha256"].map(normalize_view_id)
    work["selection_rank"] = pd.to_numeric(
        work["selection_rank"], errors="raise"
    ).astype(int)
    work["qc_label"] = work["view_id"].map(normalized_labels)

    result_rows: list[dict[str, Any]] = []
    for (exam_id, laterality, view), rows in work.groupby(
        ["exam_id", "laterality", "view"], sort=True
    ):
        rows = rows.sort_values("selection_rank", kind="stable")
        original = rows.iloc[0]
        passing = rows[rows["qc_label"] == VIEW_LABEL_PASS]
        reviewed_count = int(rows["qc_label"].notna().sum())
        if original["qc_label"] == VIEW_LABEL_PASS:
            chosen = original
            status = "original_pass"
        elif not passing.empty:
            chosen = passing.iloc[0]
            status = "alternate_pass"
        else:
            chosen = None
            status = (
                "no_passing_candidate"
                if reviewed_count == len(rows)
                else "unresolved_candidates"
            )

        result_rows.append(
            {
                "exam_id": str(exam_id),
                "laterality": str(laterality),
                "view": str(view),
                "original_view_id": str(original["view_id"]),
                "original_sop_instance_uid": str(original["sop_instance_uid"]),
                "selected_view_id": None if chosen is None else str(chosen["view_id"]),
                "selected_sop_instance_uid": None
                if chosen is None
                else str(chosen["sop_instance_uid"]),
                "selected_candidate_rank": None
                if chosen is None
                else int(chosen["selection_rank"]),
                "fallback_status": status,
                "candidate_count": int(len(rows)),
                "reviewed_candidate_count": reviewed_count,
            }
        )
    return pd.DataFrame(result_rows)
