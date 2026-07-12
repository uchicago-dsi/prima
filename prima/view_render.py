"""Restricted, source-verified rendering for view-level auto-QC."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pydicom
from tqdm import tqdm

from prima.dicom_source import (
    SOURCE_ARCHIVE_COLUMN,
    SOURCE_COLUMNS,
    DicomSource,
    materialize_dicom_sources,
    require_source_columns,
    require_valid_sources,
    validate_materialized_source,
)
from prima.view_qc import (
    normalize_view_id,
    render_dicom_view_png,
    validate_rendered_view_png,
)

RENDER_REQUIRED_COLUMNS = {
    *SOURCE_COLUMNS,
    "view_id",
    "image_path",
}


def _resolve_render_path(out_dir: Path, row: dict[str, Any]) -> Path:
    view_id = normalize_view_id(row["view_id"])
    if normalize_view_id(row["sha256"]) != view_id:
        raise ValueError("render row view_id must equal its source SHA-256")
    relative = Path(str(row["image_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("render image_path must be a safe relative path")
    expected = Path("images") / f"{view_id}.png"
    if relative != expected:
        raise ValueError("render image_path does not match the canonical view path")
    resolved = (Path(out_dir).resolve() / relative).resolve()
    try:
        resolved.relative_to(Path(out_dir).resolve())
    except ValueError as error:
        raise ValueError("render image path escapes the campaign directory") from error
    return resolved


def render_source_rows(
    rows: pd.DataFrame,
    *,
    raw_root: Path,
    out_dir: Path,
    max_pixels: int,
    resume: bool = False,
    allow_pixel_failures: bool = False,
    temp_root: Path | None = None,
    progress_desc: str = "rendering source views",
) -> dict[str, Any]:
    """Render source-linked rows after exact SOP and SHA-256 verification."""
    missing = sorted(RENDER_REQUIRED_COLUMNS - set(rows.columns))
    if missing:
        raise ValueError("render rows are missing columns: " + ", ".join(missing))
    if rows.empty:
        raise ValueError("render rows are empty")
    require_source_columns(rows.columns, "view render rows")
    require_valid_sources(
        rows[list(SOURCE_COLUMNS)].to_dict("records"), "view render rows"
    )
    if rows["view_id"].duplicated().any():
        raise ValueError("render rows contain duplicate view IDs")

    raw_root = Path(raw_root).resolve()
    out_dir = Path(out_dir).resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw DICOM root not found: {raw_root}")
    if not out_dir.is_dir():
        raise FileNotFoundError(f"render campaign directory not found: {out_dir}")
    if temp_root is not None:
        temp_root = Path(temp_root).resolve()
        if not temp_root.is_dir():
            raise FileNotFoundError(
                f"render temporary directory not found: {temp_root}"
            )

    rendered = 0
    reused = 0
    failures: list[dict[str, str]] = []
    archive_groups = rows.groupby(SOURCE_ARCHIVE_COLUMN, sort=True)
    for _archive, archive_rows in tqdm(
        archive_groups,
        total=rows[SOURCE_ARCHIVE_COLUMN].nunique(),
        desc=progress_desc,
    ):
        pending: list[dict[str, Any]] = []
        for row in archive_rows.to_dict("records"):
            output_path = _resolve_render_path(out_dir, row)
            if output_path.exists():
                if not resume:
                    raise FileExistsError("refusing to overwrite a rendered view")
                validate_rendered_view_png(output_path, max_pixels=max_pixels)
                reused += 1
            else:
                pending.append(row)
        if not pending:
            continue

        sources = [DicomSource.from_row(row) for row in pending]
        with materialize_dicom_sources(
            sources, raw_root, temp_root=temp_root
        ) as materialized:
            for row, source in zip(pending, sources):
                source_path = materialized[source.archive_member.as_posix()]
                dataset = pydicom.dcmread(str(source_path), force=True)
                validate_materialized_source(
                    source, source_path, dataset, verify_sha256=True
                )
                output_path = _resolve_render_path(out_dir, row)
                try:
                    render_dicom_view_png(dataset, output_path, max_pixels=max_pixels)
                except (AttributeError, NotImplementedError, ValueError) as error:
                    if not allow_pixel_failures:
                        raise
                    failures.append(
                        {
                            "view_id": normalize_view_id(row["view_id"]),
                            "error_type": type(error).__name__,
                        }
                    )
                else:
                    rendered += 1

    if rendered + reused + len(failures) != len(rows):
        raise RuntimeError("render accounting did not cover every source row")
    return {
        "total": int(len(rows)),
        "rendered": rendered,
        "reused": reused,
        "failed": len(failures),
        "failures": failures,
    }
