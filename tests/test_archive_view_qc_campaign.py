from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from PIL import Image
import pytest

from prima.view_qc import (
    VIEW_QC_SCHEMA_VERSION,
    VIEW_LABEL_ABSENT,
    VIEW_LABEL_PRESENT,
    default_view_qc_events_path,
    empty_view_qc_state,
    initialize_view_qc_event_log,
    load_view_qc_state,
    save_view_qc_state,
    set_view_label,
)
from qc.archive_view_qc_campaign import create_archive, verify_archive


def view_id(index: int) -> str:
    return f"{index:064x}"


def make_campaign(root: Path, *, legacy: bool = False, complete: bool = True) -> Path:
    images = root / "images"
    images.mkdir(parents=True)
    rows = []
    for index, (laterality, view) in enumerate([("L", "CC"), ("R", "MLO")], start=1):
        Image.new("L", (8, 8), color=index).save(images / f"{view_id(index)}.png")
        rows.append(
            {
                "view_id": view_id(index),
                "image_path": f"images/{view_id(index)}.png",
                "laterality": laterality,
                "view": view,
                "review_order": index,
            }
        )
    pd.DataFrame(rows).to_parquet(root / "manifest.parquet", index=False)
    if legacy:
        labels = {
            view_id(1): {
                "label": "vertical_line",
                "source": "human",
                "updated_at": "2026-07-12T12:00:00+00:00",
            },
            view_id(2): {
                "label": "pass",
                "source": "human",
                "updated_at": "2026-07-12T12:01:00+00:00",
            },
        }
        (root / "view_qc_state.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "target": "vertical line (detector artifact)",
                    "labels": labels,
                }
            )
            + "\n"
        )
    else:
        state = set_view_label(
            empty_view_qc_state("test artifact"), view_id(1), VIEW_LABEL_PRESENT
        )
        if complete:
            state = set_view_label(state, view_id(2), VIEW_LABEL_ABSENT)
        save_view_qc_state(root / "view_qc_state.json", state)
    (root / "evaluation").mkdir()
    (root / "evaluation" / "metrics.json").write_text('{"complete": true}\n')
    return root


def archive_args(source: Path, archive_root: Path, name: str) -> SimpleNamespace:
    return SimpleNamespace(
        source_dir=source,
        archive_root=archive_root,
        campaign_name=name,
        disposition="canonical-reference",
        notes="Test completed campaign.",
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_completed_campaign_archive_is_restricted_and_verifiable(
    tmp_path: Path, legacy: bool
) -> None:
    source = make_campaign(tmp_path / "source", legacy=legacy)
    archive = create_archive(
        archive_args(source, tmp_path / "archives", f"campaign_{int(legacy)}")
    )

    metadata = verify_archive(archive)
    assert metadata["manifest_rows"] == 2
    assert metadata["human_labels"] == 2
    assert metadata["state_schema_version"] == (1 if legacy else VIEW_QC_SCHEMA_VERSION)
    assert metadata["source_file_count"] == 5
    assert metadata["event_history"] == (
        "not-applicable-legacy-state" if legacy else "unavailable-pre-audit"
    )
    assert all(
        path.stat().st_mode & 0o777 == 0o400
        for path in archive.rglob("*")
        if path.is_file()
    )
    assert all(
        path.stat().st_mode & 0o777 == 0o500
        for path in [archive, *archive.rglob("*")]
        if path.is_dir()
    )


def test_campaign_archive_detects_tampering(tmp_path: Path) -> None:
    source = make_campaign(tmp_path / "source")
    archive = create_archive(
        archive_args(source, tmp_path / "archives", "test_campaign")
    )
    image = next((archive / "images").glob("*.png"))
    image.chmod(0o600)
    image.write_bytes(image.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="file size|checksum"):
        verify_archive(archive)


def test_campaign_archive_verifies_available_event_history(tmp_path: Path) -> None:
    source = make_campaign(tmp_path / "source")
    state_path = source / "view_qc_state.json"
    initialize_view_qc_event_log(
        default_view_qc_events_path(state_path), load_view_qc_state(state_path)
    )

    archive = create_archive(
        archive_args(source, tmp_path / "archives", "audited_campaign")
    )
    metadata = verify_archive(archive)

    assert metadata["event_history"] == "verified"
    assert metadata["event_count"] == 2


def test_campaign_archive_requires_complete_labels(tmp_path: Path) -> None:
    source = make_campaign(tmp_path / "source", complete=False)

    with pytest.raises(ValueError, match="completely annotated"):
        create_archive(archive_args(source, tmp_path / "archives", "incomplete"))


def test_campaign_archive_requires_low_confidence_adjudication(
    tmp_path: Path,
) -> None:
    source = make_campaign(tmp_path / "source")
    state_path = source / "view_qc_state.json"
    state = load_view_qc_state(state_path)
    state = set_view_label(
        state,
        view_id(1),
        VIEW_LABEL_PRESENT,
        low_confidence=True,
    )
    save_view_qc_state(state_path, state)

    with pytest.raises(ValueError, match="must be adjudicated"):
        create_archive(archive_args(source, tmp_path / "archives", "low_confidence"))


def test_campaign_archive_materializes_only_internal_file_symlinks(
    tmp_path: Path,
) -> None:
    source = make_campaign(tmp_path / "source")
    link = source / "evaluation" / "metrics_link.json"
    link.symlink_to(source / "evaluation" / "metrics.json")

    archive = create_archive(
        archive_args(source, tmp_path / "archives", "linked_campaign")
    )
    metadata = verify_archive(archive)

    relative = "evaluation/metrics_link.json"
    assert not (archive / relative).is_symlink()
    assert metadata["source_files"][relative]["source_type"] == "materialized_symlink"


def test_campaign_archive_refuses_external_symlink(tmp_path: Path) -> None:
    source = make_campaign(tmp_path / "source")
    external = tmp_path / "external.txt"
    external.write_text("outside\n")
    (source / "external_link.txt").symlink_to(external)

    with pytest.raises(ValueError, match="external or non-file"):
        create_archive(archive_args(source, tmp_path / "archives", "external_campaign"))
