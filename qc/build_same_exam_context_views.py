#!/usr/bin/env python3
"""Build target-preserving same-exam context composites for view auto-QC."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from prima.dicom_source import (
    SOURCE_ARCHIVE_COLUMN,
    SOURCE_COLUMNS,
    require_source_columns,
    require_valid_sources,
)
from prima.view_few_shot import sha256_file
from prima.view_qc import (
    normalize_view_id,
    validate_rendered_view_png,
    validate_view_manifest_columns,
)
from prima.view_render import render_source_rows

ALLOWED_LATERALITIES = {"L", "R"}
ALLOWED_VIEWS = {"CC", "MLO"}
FONT_PATH = Path("/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf")
CANVAS_SIZE = (1600, 1200)
CANVAS_MAX_PIXELS = CANVAS_SIZE[0] * CANVAS_SIZE[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--max-context-views", type=int, default=3)
    parser.add_argument("--max-source-pixels", type=int, default=2_000_000)
    return parser.parse_args()


def _load_source_pool(path: Path, *, source_kind: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"same-exam context source pool not found: {path}")
    frame = pd.read_parquet(path)
    required = {"exam_id", "laterality", "view", "sha256", *SOURCE_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"same-exam context source pool is missing columns: {', '.join(missing)}"
        )
    require_source_columns(frame.columns, str(path))
    frame = frame.copy()
    frame["view_id"] = frame["sha256"].map(normalize_view_id)
    frame["source_kind"] = source_kind
    if "is_selected" not in frame:
        frame["is_selected"] = False
    if "selection_rank" not in frame:
        frame["selection_rank"] = pd.NA
    return frame


def _sort_context_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["_selected_order"] = ~ranked["is_selected"].fillna(False).astype(bool)
    ranked["_source_order"] = (ranked["source_kind"] != "candidate").astype(int)
    ranked["_rank_order"] = pd.to_numeric(
        ranked["selection_rank"], errors="coerce"
    ).fillna(1_000_000)
    return ranked.sort_values(
        ["_selected_order", "_source_order", "_rank_order", "view_id"],
        kind="stable",
    )


def select_context_rows(
    choices: pd.DataFrame,
    *,
    target_view_id: str,
    target_laterality: str,
    target_view: str,
    maximum: int,
) -> pd.DataFrame:
    """Select diverse same-exam references in a fixed slot-aware order."""
    if maximum <= 0:
        raise ValueError("maximum same-exam context views must be positive")
    target_view_id = normalize_view_id(target_view_id)
    pool = choices[
        choices["laterality"].isin(ALLOWED_LATERALITIES)
        & choices["view"].isin(ALLOWED_VIEWS)
        & (choices["view_id"] != target_view_id)
    ].copy()
    if pool.empty:
        return pool
    pool = _sort_context_candidates(pool).drop_duplicates("view_id", keep="first")
    opposite_laterality = "R" if target_laterality == "L" else "L"
    opposite_view = "MLO" if target_view == "CC" else "CC"
    slots = (
        (target_laterality, target_view),
        (target_laterality, opposite_view),
        (opposite_laterality, target_view),
        (opposite_laterality, opposite_view),
    )
    selected_indices: list[Any] = []
    selected_ids: set[str] = set()
    for laterality, view in slots:
        candidates = pool[
            (pool["laterality"] == laterality)
            & (pool["view"] == view)
            & ~pool["view_id"].isin(selected_ids)
        ]
        if candidates.empty:
            continue
        chosen = candidates.iloc[0]
        selected_indices.append(chosen.name)
        selected_ids.add(str(chosen["view_id"]))
        if len(selected_indices) == maximum:
            break
    if len(selected_indices) < maximum:
        for row in pool[~pool["view_id"].isin(selected_ids)].itertuples():
            selected_indices.append(row.Index)
            selected_ids.add(str(row.view_id))
            if len(selected_indices) == maximum:
                break
    selected = pool.loc[selected_indices].copy()
    selected["context_order"] = range(1, len(selected) + 1)
    return selected


def same_exam_choices(
    pool: pd.DataFrame, *, exam_id: object, source_archive: object
) -> pd.DataFrame:
    """Return references from the exact physical exam archive."""
    return pool[
        (pool["exam_id"].astype(str) == str(exam_id))
        & (pool[SOURCE_ARCHIVE_COLUMN].astype(str) == str(source_archive))
    ]


def _font(size: int) -> ImageFont.FreeTypeFont:
    if not FONT_PATH.is_file():
        raise FileNotFoundError(f"same-exam context font not found: {FONT_PATH}")
    return ImageFont.truetype(str(FONT_PATH), size=size)


def _draw_panel(
    canvas: Image.Image,
    *,
    source_path: Path,
    bounds: tuple[int, int, int, int],
    label: str,
    border_value: int,
) -> None:
    left, top, right, bottom = bounds
    if right <= left or bottom <= top:
        raise ValueError("same-exam context panel has invalid bounds")
    label_height = 42
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(bounds, outline=border_value, width=4)
    draw.text((left + 10, top + 5), label, fill=255, font=_font(24))
    image_box = (left + 6, top + label_height, right - 6, bottom - 6)
    box_width = image_box[2] - image_box[0]
    box_height = image_box[3] - image_box[1]
    with Image.open(source_path) as source:
        source.load()
        contained = ImageOps.contain(
            source.convert("L"), (box_width, box_height), Image.LANCZOS
        )
    x = image_box[0] + (box_width - contained.width) // 2
    y = image_box[1] + (box_height - contained.height) // 2
    canvas.paste(contained, (x, y))


def compose_context_image(
    *, target_path: Path, context_paths: list[Path], output_path: Path
) -> None:
    """Render one target-dominant grayscale composite with labeled references."""
    if len(context_paths) > 3:
        raise ValueError(
            "same-exam context composite supports at most three references"
        )
    canvas = Image.new("L", CANVAS_SIZE, color=0)
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 11), "CLASSIFY TARGET VIEW ONLY", fill=255, font=_font(30))
    _draw_panel(
        canvas,
        source_path=target_path,
        bounds=(16, 56, 998, 1184),
        label="TARGET VIEW",
        border_value=255,
    )
    reference_left, reference_right = 1014, 1584
    reference_top, reference_bottom = 56, 1184
    if context_paths:
        gap = 10
        height = (
            reference_bottom - reference_top - gap * (len(context_paths) - 1)
        ) // len(context_paths)
        for index, path in enumerate(context_paths, start=1):
            top = reference_top + (index - 1) * (height + gap)
            bottom = reference_bottom if index == len(context_paths) else top + height
            _draw_panel(
                canvas,
                source_path=path,
                bounds=(reference_left, top, reference_right, bottom),
                label=f"SAME-EXAM REFERENCE {index}",
                border_value=150,
            )
    else:
        draw.rectangle(
            (reference_left, reference_top, reference_right, reference_bottom),
            outline=100,
            width=4,
        )
        draw.multiline_text(
            (reference_left + 32, 520),
            "NO SAME-EXAM\nREFERENCE AVAILABLE",
            fill=180,
            font=_font(26),
            spacing=12,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        canvas.save(temporary_path, format="PNG", optimize=True)
        validate_rendered_view_png(temporary_path, max_pixels=CANVAS_MAX_PIXELS)
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, output_path)
        os.chmod(output_path, 0o600)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")
    os.chmod(path, 0o600)


def run_from_args(args: argparse.Namespace) -> dict[str, int]:
    manifest_path = args.manifest.resolve()
    source_manifest_path = args.source_manifest.resolve()
    exclusions_path = args.exclusions.resolve()
    candidates_path = args.candidates.resolve()
    raw_root = args.raw_root.resolve()
    output_manifest_path = args.output_manifest.resolve()
    temp_root = args.temp_root.resolve()
    for path in (
        manifest_path,
        source_manifest_path,
        exclusions_path,
        candidates_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"same-exam context input not found: {path}")
    for path in (raw_root, temp_root):
        if not path.is_dir():
            raise FileNotFoundError(f"same-exam context directory not found: {path}")
    if output_manifest_path.parent != manifest_path.parent:
        raise ValueError(
            "same-exam output manifest must share the target manifest directory"
        )
    if not 1 <= args.max_context_views <= 3:
        raise ValueError("--max-context-views must be between 1 and 3")
    if args.max_source_pixels <= 0:
        raise ValueError("--max-source-pixels must be positive")
    asset_dir = output_manifest_path.with_suffix("")
    provenance_path = output_manifest_path.with_suffix(".provenance.json")
    source_map_path = output_manifest_path.with_suffix(".sources.parquet")
    readme_path = output_manifest_path.with_suffix(".README.md")
    for path in (
        output_manifest_path,
        asset_dir,
        provenance_path,
        source_map_path,
        readme_path,
    ):
        if path.exists():
            raise FileExistsError(
                f"refusing to overwrite same-exam context output: {path}"
            )

    manifest = pd.read_parquet(manifest_path)
    validate_view_manifest_columns(manifest.columns, str(manifest_path))
    manifest = manifest.sort_values("review_order", kind="stable").copy()
    manifest["view_id"] = manifest["view_id"].map(normalize_view_id)
    if manifest["view_id"].duplicated().any():
        raise ValueError("same-exam target manifest contains duplicate view IDs")
    source_manifest = pd.read_parquet(source_manifest_path)
    required_source = {
        "view_id",
        "exam_id",
        "laterality",
        "view",
        SOURCE_ARCHIVE_COLUMN,
    }
    missing_source = sorted(required_source - set(source_manifest.columns))
    if missing_source:
        raise ValueError(
            "same-exam target source manifest is missing columns: "
            + ", ".join(missing_source)
        )
    source_manifest = source_manifest.copy()
    source_manifest["view_id"] = source_manifest["view_id"].map(normalize_view_id)
    source_manifest = source_manifest[
        source_manifest["view_id"].isin(set(manifest["view_id"]))
    ]
    if (
        len(source_manifest) != len(manifest)
        or source_manifest["view_id"].duplicated().any()
    ):
        raise ValueError(
            "same-exam source manifest must map every target view exactly once"
        )
    targets = manifest.merge(
        source_manifest[
            ["view_id", "exam_id", "laterality", "view", SOURCE_ARCHIVE_COLUMN]
        ],
        on="view_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_source"),
    )
    for column in ("laterality", "view"):
        source_column = f"{column}_source"
        if (
            source_column in targets
            and not (
                targets[column].astype(str) == targets[source_column].astype(str)
            ).all()
        ):
            raise ValueError(
                "same-exam source and target manifests disagree on view metadata"
            )

    exclusions = _load_source_pool(exclusions_path, source_kind="exclusion")
    candidates = _load_source_pool(candidates_path, source_kind="candidate")
    target_groups = set(
        zip(
            targets["exam_id"].astype(str),
            targets[SOURCE_ARCHIVE_COLUMN].astype(str),
        )
    )
    pool = pd.concat([candidates, exclusions], ignore_index=True, sort=False)
    pool_groups = pd.Series(
        list(
            zip(
                pool["exam_id"].astype(str),
                pool[SOURCE_ARCHIVE_COLUMN].astype(str),
            )
        ),
        index=pool.index,
    )
    pool = pool[pool_groups.isin(target_groups)].copy()

    selections: list[pd.DataFrame] = []
    context_counts: dict[str, int] = {}
    for target in targets.to_dict("records"):
        choices = same_exam_choices(
            pool,
            exam_id=target["exam_id"],
            source_archive=target[SOURCE_ARCHIVE_COLUMN],
        )
        selected = select_context_rows(
            choices,
            target_view_id=target["view_id"],
            target_laterality=str(target["laterality"]),
            target_view=str(target["view"]),
            maximum=args.max_context_views,
        )
        context_counts[target["view_id"]] = len(selected)
        if selected.empty:
            continue
        selected = selected.copy()
        selected["target_view_id"] = target["view_id"]
        selections.append(selected)
    if not selections:
        raise RuntimeError("same-exam context selection found no reference views")
    source_map = pd.concat(selections, ignore_index=True, sort=False)
    render_rows = source_map.drop_duplicates("view_id", keep="first").copy()
    require_valid_sources(
        render_rows[list(SOURCE_COLUMNS)].to_dict("records"),
        "same-exam context selections",
    )

    asset_dir.mkdir(mode=0o700)
    source_asset_dir = asset_dir / "sources"
    context_image_dir = asset_dir / "images"
    source_asset_dir.mkdir(mode=0o700)
    context_image_dir.mkdir(mode=0o700)
    render_rows["image_path"] = render_rows["view_id"].map(
        lambda value: f"images/{value}.png"
    )
    render_result = render_source_rows(
        render_rows,
        raw_root=raw_root,
        out_dir=source_asset_dir,
        max_pixels=args.max_source_pixels,
        temp_root=temp_root,
        progress_desc="rendering same-exam context",
    )
    if render_result["failed"]:
        raise RuntimeError("same-exam context rendering had pixel failures")

    source_map = source_map.sort_values(
        ["target_view_id", "context_order"], kind="stable"
    )
    model_paths: dict[str, str] = {}
    root = manifest_path.parent.resolve()
    for target in targets.to_dict("records"):
        relative_target = Path(str(target["image_path"]))
        if relative_target.is_absolute() or ".." in relative_target.parts:
            raise ValueError("same-exam target image path must be safe and relative")
        target_path = (root / relative_target).resolve()
        try:
            target_path.relative_to(root)
        except ValueError as error:
            raise ValueError("same-exam target image escapes manifest root") from error
        validate_rendered_view_png(target_path, max_pixels=args.max_source_pixels)
        context_rows = source_map[
            source_map["target_view_id"] == target["view_id"]
        ].sort_values("context_order", kind="stable")
        context_paths = [
            source_asset_dir / "images" / f"{view_id}.png"
            for view_id in context_rows["view_id"]
        ]
        output_path = context_image_dir / f"{target['view_id']}.png"
        compose_context_image(
            target_path=target_path,
            context_paths=context_paths,
            output_path=output_path,
        )
        model_paths[target["view_id"]] = output_path.relative_to(root).as_posix()

    output = manifest.copy()
    output["model_image_path"] = output["view_id"].map(model_paths)
    output["context_count"] = output["view_id"].map(context_counts).astype(int)
    output.to_parquet(output_manifest_path, index=False)
    os.chmod(output_manifest_path, 0o600)
    safe_source_columns = [
        "target_view_id",
        "context_order",
        "view_id",
        "exam_id",
        "laterality",
        "view",
        "source_kind",
        *SOURCE_COLUMNS,
    ]
    source_map[
        [column for column in safe_source_columns if column in source_map]
    ].to_parquet(source_map_path, index=False)
    os.chmod(source_map_path, 0o600)

    ordered_composite_digests = [
        sha256_file(context_image_dir / f"{view_id}.png")
        for view_id in output["view_id"]
    ]
    bank_digest = hashlib.sha256(
        "".join(ordered_composite_digests).encode()
    ).hexdigest()
    count_distribution = dict(sorted(Counter(context_counts.values()).items()))
    command = shlex.join([sys.executable, *sys.argv])
    provenance = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "source_digests": {
            "manifest": sha256_file(manifest_path),
            "source_manifest": sha256_file(source_manifest_path),
            "exclusions": sha256_file(exclusions_path),
            "candidates": sha256_file(candidates_path),
        },
        "output_manifest_sha256": sha256_file(output_manifest_path),
        "ordered_context_image_bank_sha256": bank_digest,
        "target_rows": int(len(output)),
        "targets_with_context": int(
            sum(value > 0 for value in context_counts.values())
        ),
        "context_count_distribution": {
            str(key): int(value) for key, value in count_distribution.items()
        },
        "max_context_views": int(args.max_context_views),
        "selection_rule": (
            "same slot, same-side opposite projection, opposite-side same "
            "projection, opposite-side opposite projection; selected standard "
            "candidates precede exclusions within each slot"
        ),
        "canonical_image_column": "image_path",
        "model_image_column": "model_image_path",
        "canvas_size": list(CANVAS_SIZE),
    }
    _write_json(provenance_path, provenance)
    readme_path.write_text(
        "\n".join(
            [
                "# Same-exam context view inputs",
                "",
                f"- target rows: `{len(output)}`",
                "- canonical target column: `image_path`",
                "- model input column: `model_image_path`",
                f"- targets with context: `{provenance['targets_with_context']}`",
                f"- context-count distribution: `{count_distribution}`",
                f"- maximum references per target: `{args.max_context_views}`",
                f"- ordered composite-bank SHA-256: `{bank_digest}`",
                "",
                "Each composite labels one large TARGET VIEW and up to three",
                "same-exam references. The canonical target path remains unchanged",
                "for evaluation and logical-OR lineage.",
                "",
                f"Exact producer command: `{command}`",
                "",
            ]
        )
    )
    os.chmod(readme_path, 0o600)
    return {
        "targets": int(len(output)),
        "with_context": int(provenance["targets_with_context"]),
        "rendered_sources": int(render_result["rendered"]),
    }


def main() -> int:
    result = run_from_args(parse_args())
    print(
        "same-exam context inputs ready: "
        f"targets={result['targets']} with_context={result['with_context']} "
        f"rendered_sources={result['rendered_sources']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
