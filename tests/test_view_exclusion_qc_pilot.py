from __future__ import annotations

from pathlib import Path

import pandas as pd

from qc.build_view_exclusion_qc_pilot import (
    classify_exclusion_pool,
    load_source_table,
    sample_candidate_sources,
    select_rendered_panel,
)


def source_rows(prefix: str, count: int, **values: object) -> pd.DataFrame:
    rows = []
    for index in range(count):
        rows.append(
            {
                "patient_id": f"TEST_{prefix}_{index}",
                "exam_id": f"EXAM_{prefix}_{index}",
                "sop_instance_uid": f"SOP_{prefix}_{index}",
                "sha256": f"{index + 1:064x}",
                "view_id": f"{index + 1:064x}",
                "laterality": "L" if index % 2 == 0 else "R",
                "view": "CC" if index % 2 == 0 else "MLO",
                "view_modifiers": "[]",
                "partial_view_description": "",
                "paddle_description": "",
                "exclusion_reasons": "[]",
                **values,
            }
        )
    return pd.DataFrame(rows)


def test_exclusion_enrichment_uses_text_but_keeps_only_allowed_angles() -> None:
    rows = source_rows("POOL", 3)
    rows.loc[0, "view_modifiers"] = '["Spot Compression"]'
    rows.loc[1, "paddle_description"] = "magnification paddle"
    rows.loc[2, "view"] = "ML"
    rows.loc[2, "view_modifiers"] = '["Spot Compression"]'

    classified = classify_exclusion_pool(rows, r"spot|magnif")

    assert len(classified) == 2
    assert classified["metadata_enriched"].tolist() == [True, True]
    assert set(classified["view"]) == {"CC", "MLO"}


def test_source_pool_preserves_reused_exam_and_sop_identifiers(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        [
            {
                "patient_id": "TEST_A",
                "exam_id": "SAME_EXAM",
                "sop_instance_uid": "SAME_SOP",
                "sha256": "1" * 64,
                "laterality": "L",
                "view": "CC",
                "source_archive_relpath": "patient_a/exam_a.tar.zst",
                "source_archive_member": "exam_a/one.dcm",
            },
            {
                "patient_id": "TEST_B",
                "exam_id": "SAME_EXAM",
                "sop_instance_uid": "SAME_SOP",
                "sha256": "2" * 64,
                "laterality": "R",
                "view": "MLO",
                "source_archive_relpath": "patient_b/exam_b.tar.zst",
                "source_archive_member": "exam_b/two.dcm",
            },
        ]
    )
    path = tmp_path / "sources.parquet"
    rows.to_parquet(path, index=False)

    loaded = load_source_table(
        path,
        required={
            "patient_id",
            "exam_id",
            "sop_instance_uid",
            "sha256",
            "laterality",
            "view",
        },
    )

    assert len(loaded) == 2
    assert loaded["view_id"].tolist() == ["1" * 64, "2" * 64]


def test_sampling_and_render_reserves_preserve_strata_and_exam_disjointness() -> None:
    enriched = source_rows("ENRICHED", 5)
    other = source_rows("OTHER", 5)
    other["sha256"] = other["sha256"].map(lambda value: "a" + value[1:])
    other["view_id"] = other["sha256"]
    standard = source_rows("STANDARD", 5)
    standard["sha256"] = standard["sha256"].map(lambda value: "b" + value[1:])
    standard["view_id"] = standard["sha256"]
    counts = {
        "metadata_enriched_exclusion": 2,
        "other_diagnostic_exclusion": 2,
        "standard_view_control": 2,
    }

    candidates = sample_candidate_sources(
        enriched,
        other,
        standard,
        requested_counts=counts,
        reserve_per_stratum=1,
        seed=7,
    )
    failed = {
        str(
            candidates[candidates["stratum"] == "metadata_enriched_exclusion"]
            .sort_values("sampling_priority")
            .iloc[0]["view_id"]
        )
    }
    panel = select_rendered_panel(
        candidates,
        failed_view_ids=failed,
        requested_counts=counts,
        seed=7,
    )

    assert len(candidates) == 9
    assert candidates["exam_id"].nunique() == 9
    assert len(panel) == 6
    assert panel["exam_id"].nunique() == 6
    assert panel["stratum"].value_counts().to_dict() == counts
    assert sorted(panel["review_order"]) == list(range(1, 7))
