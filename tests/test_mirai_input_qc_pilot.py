from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from PIL import Image

from prima.view_qc import load_view_qc_events, load_view_qc_state
from qc.build_mirai_input_qc_pilot import DEFAULT_TARGET, run_from_args


def _view_id(index: int) -> str:
    return f"{index:064x}"


def _source_row(
    root: Path,
    index: int,
    *,
    laterality: str,
    view: str,
    has_implant: bool = False,
) -> dict[str, object]:
    exam_id = f"exam-{index}"
    archive_relpath = f"patient-{index}/{exam_id}.tar.zst"
    archive = root / archive_relpath
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.touch()
    return {
        "patient_id": f"patient-{index}",
        "exam_id": exam_id,
        "sop_instance_uid": f"1.2.840.{index}",
        "laterality": laterality,
        "view": view,
        "source_archive_relpath": archive_relpath,
        "source_archive_member": f"{exam_id}/image.dcm",
        "sha256": _view_id(index),
        "has_implant": has_implant,
        "for_presentation": True,
        "is_marked_up": False,
    }


def _write_campaign(
    root: Path,
    name: str,
    rows: list[dict[str, object]],
    labels: dict[str, str],
    *,
    source_manifest: bool = False,
) -> None:
    campaign = root / name
    images = campaign / "images"
    images.mkdir(parents=True)
    browser_rows = []
    for order, row in enumerate(rows, start=1):
        view_id = str(row["sha256"])
        Image.new("L", (8, 8), color=order).save(images / f"{view_id}.png")
        browser_rows.append(
            {
                "view_id": view_id,
                "image_path": f"images/{view_id}.png",
                "laterality": row["laterality"],
                "view": row["view"],
                "review_order": order,
                "stratum": row["reference_stratum"],
            }
        )
    pd.DataFrame(browser_rows).to_parquet(campaign / "manifest.parquet", index=False)
    if source_manifest:
        source = pd.DataFrame(rows).drop(columns=["reference_stratum"])
        source["view_id"] = source["sha256"]
        source.to_parquet(campaign / "source_manifest.parquet", index=False)
    state_labels = {
        view_id: {
            "label": label,
            "source": "human",
            "updated_at": "2026-07-13T00:00:00+00:00",
        }
        for view_id, label in labels.items()
    }
    (campaign / "view_qc_state.json").write_text(
        json.dumps(
            {"schema_version": 2, "target": f"target-{name}", "labels": state_labels}
        )
    )


def test_binary_mirai_pilot_is_blinded_patient_disjoint_and_source_linked(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archive"
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    slots = [
        ("L", "CC"),
        ("R", "CC"),
        ("L", "MLO"),
        ("R", "MLO"),
        ("L", "CC"),
        ("R", "CC"),
        ("L", "MLO"),
    ]
    rows = [
        _source_row(raw_root, index, laterality=lat, view=view)
        for index, (lat, view) in enumerate(slots, start=1)
    ]
    rows[2]["has_implant"] = True
    candidates_path = tmp_path / "candidates.parquet"
    pd.DataFrame(rows).to_parquet(candidates_path, index=False)

    seam = [{**rows[0], "reference_stratum": "seam"}]
    film = [{**rows[1], "reference_stratum": "film"}]
    implant = [
        {**rows[2], "reference_stratum": "implant"},
        {**rows[6], "reference_stratum": "control"},
    ]
    diagnostic = [
        {**rows[3], "reference_stratum": "metadata_enriched_exclusion"},
        {**rows[4], "reference_stratum": "other_diagnostic_exclusion"},
        {**rows[5], "reference_stratum": "standard_view_control"},
    ]
    _write_campaign(
        archive_root,
        "vertical_detector_seam_reference",
        seam,
        {_view_id(1): "vertical_line"},
    )
    _write_campaign(
        archive_root,
        "digitized_hard_copy_film_reference",
        film,
        {_view_id(2): "present"},
    )
    _write_campaign(
        archive_root,
        "visible_breast_implant_reference",
        implant,
        {_view_id(3): "present", _view_id(7): "absent"},
    )
    _write_campaign(
        archive_root,
        "spot_compression_magnification_holdout_reference",
        diagnostic,
        {_view_id(4): "present", _view_id(5): "absent", _view_id(6): "absent"},
        source_manifest=True,
    )

    out_dir = tmp_path / "pilot"
    manifest = run_from_args(
        SimpleNamespace(
            annotation_archive_root=archive_root,
            candidates=candidates_path,
            raw_root=raw_root,
            out_dir=out_dir,
            target=DEFAULT_TARGET,
            failure_count=1,
            diagnostic_control_count=1,
            standard_extra_count=1,
            seed=17,
            max_render_pixels=100,
        )
    )

    assert len(manifest) == 7
    assert set(manifest.columns) == {
        "view_id",
        "image_path",
        "laterality",
        "view",
        "review_order",
        "stratum",
    }
    assert not {"patient_id", "exam_id", "sop_instance_uid"} & set(manifest.columns)
    assert set(manifest["laterality"]) <= {"L", "R"}
    assert set(manifest["view"]) <= {"CC", "MLO"}
    assert all((out_dir / value).is_file() for value in manifest["image_path"])

    source = pd.read_parquet(out_dir / "source_manifest.parquet")
    assert source["patient_id"].nunique() == 7
    assert source["exam_id"].nunique() == 7
    assert source["view_id"].nunique() == 7
    assert source["source_archive_relpath"].notna().all()
    assert source["source_archive_member"].notna().all()
    assert load_view_qc_state(out_dir / "view_qc_state.json")["labels"] == {}
    assert load_view_qc_events(out_dir / "view_qc_events.jsonl") == []
    assert (out_dir / "source_manifest.parquet").stat().st_mode & 0o777 == 0o600
    assert "Use for Mirai" in (out_dir / "README.md").read_text()
    assert "Do not use" in (out_dir / "README.md").read_text()
