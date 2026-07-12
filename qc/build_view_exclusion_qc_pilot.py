#!/usr/bin/env python3
"""Build a blinded visual-QC pilot from diagnostic exclusions and controls."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from prima.dicom_source import (
    SOURCE_ARCHIVE_COLUMN,
    SOURCE_COLUMNS,
    SOURCE_MEMBER_COLUMN,
    require_source_columns,
    require_valid_sources,
)
from prima.view_qc import (
    empty_view_qc_state,
    normalize_view_id,
    normalize_view_qc_target,
    save_view_qc_state,
    validate_view_manifest_columns,
)
from prima.view_render import render_source_rows

ALLOWED_VIEW_POSITIONS = {"CC", "MLO"}
ENRICHMENT_TEXT_COLUMNS = (
    "view_modifiers",
    "partial_view_description",
    "paddle_description",
    "exclusion_reasons",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--standard-views", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--enrichment-regex", required=True)
    parser.add_argument("--enriched-count", type=int, default=50)
    parser.add_argument("--other-exclusion-count", type=int, default=20)
    parser.add_argument("--standard-count", type=int, default=50)
    parser.add_argument("--reserve-per-stratum", type=int, default=10)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--max-render-pixels", type=int, default=2_000_000)
    parser.add_argument("--temp-root", type=Path, default=None)
    return parser.parse_args()


def load_source_table(path: Path, *, required: set[str]) -> pd.DataFrame:
    """Load a source-linked parquet table and enforce its current schema."""
    if not path.is_file():
        raise FileNotFoundError(f"source table not found: {path}")
    table = pd.read_parquet(path)
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    require_source_columns(table.columns, str(path))
    require_valid_sources(table[list(SOURCE_COLUMNS)].to_dict("records"), str(path))
    if table.duplicated([SOURCE_ARCHIVE_COLUMN, SOURCE_MEMBER_COLUMN]).any():
        raise ValueError(f"{path} contains duplicate physical DICOM sources")
    table = table.copy()
    table["view_id"] = table["sha256"].map(normalize_view_id)
    return table


def load_excluded_view_ids(paths: list[Path]) -> set[str]:
    """Load exact view IDs from prior deidentified review manifests."""
    excluded: set[str] = set()
    for raw_path in paths:
        path = raw_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"excluded review manifest not found: {path}")
        manifest = pd.read_parquet(path)
        validate_view_manifest_columns(manifest.columns, str(path))
        ids = {normalize_view_id(value) for value in manifest["view_id"]}
        excluded.update(ids)
    return excluded


def classify_exclusion_pool(
    exclusions: pd.DataFrame, enrichment_regex: str
) -> pd.DataFrame:
    """Classify exclusions for sampling; metadata remains enrichment, not truth."""
    try:
        pattern = re.compile(enrichment_regex, flags=re.IGNORECASE)
    except re.error as error:
        raise ValueError(f"invalid --enrichment-regex: {error}") from error
    pool = exclusions[
        exclusions["laterality"].isin(["L", "R"])
        & exclusions["view"].isin(ALLOWED_VIEW_POSITIONS)
    ].copy()
    text = pd.Series("", index=pool.index, dtype="string")
    for column in ENRICHMENT_TEXT_COLUMNS:
        text = text.str.cat(pool[column].fillna("").astype(str), sep=" ")
    pool["metadata_enriched"] = text.map(lambda value: bool(pattern.search(value)))
    return pool


def _sample_unique_exams(
    pool: pd.DataFrame,
    *,
    count: int,
    stratum: str,
    used_exams: set[str],
    used_views: set[str],
    seed: int,
) -> pd.DataFrame:
    """Sample deterministically with no repeated exam or image across strata."""
    shuffled = pool.sort_values("view_id", kind="stable").sample(
        frac=1.0, random_state=seed
    )
    rows: list[dict[str, object]] = []
    for row in shuffled.to_dict("records"):
        exam_id = str(row["exam_id"])
        view_id = str(row["view_id"])
        if exam_id in used_exams or view_id in used_views:
            continue
        used_exams.add(exam_id)
        used_views.add(view_id)
        row["stratum"] = stratum
        row["sampling_priority"] = len(rows) + 1
        rows.append(row)
        if len(rows) == count:
            break
    if len(rows) != count:
        raise ValueError(
            f"only {len(rows)} independent render candidates are available for "
            f"{stratum}; requested {count}"
        )
    return pd.DataFrame(rows)


def sample_candidate_sources(
    enriched: pd.DataFrame,
    other_exclusions: pd.DataFrame,
    standard: pd.DataFrame,
    *,
    requested_counts: dict[str, int],
    reserve_per_stratum: int,
    seed: int,
) -> pd.DataFrame:
    """Sample requested views plus render reserves from three independent strata."""
    pools = {
        "metadata_enriched_exclusion": enriched,
        "other_diagnostic_exclusion": other_exclusions,
        "standard_view_control": standard,
    }
    used_exams: set[str] = set()
    used_views: set[str] = set()
    sampled: list[pd.DataFrame] = []
    for offset, (stratum, pool) in enumerate(pools.items()):
        sampled.append(
            _sample_unique_exams(
                pool,
                count=requested_counts[stratum] + reserve_per_stratum,
                stratum=stratum,
                used_exams=used_exams,
                used_views=used_views,
                seed=seed + offset,
            )
        )
    candidates = pd.concat(sampled, ignore_index=True)
    candidates["image_path"] = candidates["view_id"].map(
        lambda value: f"images/{value}.png"
    )
    if (
        candidates["exam_id"].duplicated().any()
        or candidates["view_id"].duplicated().any()
    ):
        raise RuntimeError("candidate sampling violated exam/view disjointness")
    return candidates


def select_rendered_panel(
    candidates: pd.DataFrame,
    *,
    failed_view_ids: set[str],
    requested_counts: dict[str, int],
    seed: int,
) -> pd.DataFrame:
    """Use render reserves without changing frozen stratum counts."""
    successful = candidates[~candidates["view_id"].isin(failed_view_ids)].copy()
    selected: list[pd.DataFrame] = []
    for stratum, count in requested_counts.items():
        rows = successful[successful["stratum"] == stratum].sort_values(
            "sampling_priority", kind="stable"
        )
        if len(rows) < count:
            raise RuntimeError(
                f"only {len(rows)} {stratum} views rendered; {count} are required"
            )
        selected.append(rows.head(count))
    panel = pd.concat(selected, ignore_index=True)
    panel = panel.sample(frac=1.0, random_state=seed + 100).reset_index(drop=True)
    panel["review_order"] = range(1, len(panel) + 1)
    return panel


def write_outputs(
    out_dir: Path,
    panel: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    target: str,
    metadata: dict[str, object],
) -> None:
    """Write a browser-safe manifest and restricted source/provenance artifacts."""
    selected_ids = set(panel["view_id"])
    for view_id in set(candidates["view_id"]) - selected_ids:
        image_path = out_dir / "images" / f"{view_id}.png"
        if image_path.exists():
            image_path.unlink()
    for view_id in selected_ids:
        image_path = out_dir / "images" / f"{view_id}.png"
        if not image_path.is_file():
            raise FileNotFoundError("selected review image is missing after rendering")
        os.chmod(image_path, 0o600)

    safe_columns = [
        "view_id",
        "image_path",
        "laterality",
        "view",
        "review_order",
        "stratum",
    ]
    manifest = panel[safe_columns].sort_values("review_order", kind="stable")
    source_manifest = panel.drop(columns=["sampling_priority"]).sort_values(
        "review_order", kind="stable"
    )
    manifest_path = out_dir / "manifest.parquet"
    source_path = out_dir / "source_manifest.parquet"
    metadata_path = out_dir / "sampling_metadata.json"
    state_path = out_dir / "view_qc_state.json"
    manifest.to_parquet(manifest_path, index=False)
    source_manifest.to_parquet(source_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    save_view_qc_state(state_path, empty_view_qc_state(target))
    for path in (manifest_path, source_path, metadata_path):
        os.chmod(path, 0o600)

    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Blinded diagnostic-view QC pilot",
                "",
                f"- target: `{target}`",
                f"- total views: `{len(manifest)}`",
                f"- strata: `{metadata['selected_counts']}`",
                f"- seed: `{metadata['seed']}`",
                "- one view per exam across all strata",
                "- only L/R CC/MLO images are included",
                "- browser manifest contains no patient, exam, SOP, or source identifiers",
                "- DICOM metadata is used only for enrichment, never as reference truth",
                "- model output must remain hidden until human review is complete",
                "",
                "## Decision rule",
                "",
                "Do not estimate sensitivity if the completed panel contains fewer than",
                "20 human-positive views; redesign enrichment instead. Otherwise report",
                "the frozen model confusion matrix overall and specificity separately for",
                "other diagnostic exclusions and standard controls. Metadata agreement is",
                "a challenger analysis, not model accuracy.",
                "",
                f"Exact producer command: `{metadata['command']}`",
                "",
            ]
        )
    )
    os.chmod(readme, 0o600)


def run_from_args(args: argparse.Namespace) -> pd.DataFrame:
    """Build the panel and return its deidentified review manifest."""
    exclusions_path = args.exclusions.resolve()
    standard_path = args.standard_views.resolve()
    raw_root = args.raw_root.resolve()
    out_dir = args.out_dir.resolve()
    target = normalize_view_qc_target(args.target)
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite view QC pilot: {out_dir}")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw DICOM root not found: {raw_root}")
    counts = {
        "metadata_enriched_exclusion": int(args.enriched_count),
        "other_diagnostic_exclusion": int(args.other_exclusion_count),
        "standard_view_control": int(args.standard_count),
    }
    if any(value <= 0 for value in counts.values()):
        raise ValueError("all stratum counts must be positive")
    if args.reserve_per_stratum < 0 or args.max_render_pixels <= 0:
        raise ValueError("render reserve must be nonnegative and max pixels positive")

    exclusion_required = {
        "patient_id",
        "exam_id",
        "sop_instance_uid",
        "sha256",
        "laterality",
        "view",
        *ENRICHMENT_TEXT_COLUMNS,
    }
    standard_required = {
        "patient_id",
        "exam_id",
        "sop_instance_uid",
        "sha256",
        "laterality",
        "view",
    }
    exclusions = load_source_table(exclusions_path, required=exclusion_required)
    standard = load_source_table(standard_path, required=standard_required)
    prior_ids = load_excluded_view_ids(args.exclude_manifest)
    exclusions = exclusions[~exclusions["view_id"].isin(prior_ids)].copy()
    standard = standard[~standard["view_id"].isin(prior_ids)].copy()
    exclusions = classify_exclusion_pool(exclusions, args.enrichment_regex)
    standard = standard[
        standard["laterality"].isin(["L", "R"])
        & standard["view"].isin(ALLOWED_VIEW_POSITIONS)
    ].copy()
    enriched = exclusions[exclusions["metadata_enriched"]].copy()
    other = exclusions[~exclusions["metadata_enriched"]].copy()

    candidates = sample_candidate_sources(
        enriched,
        other,
        standard,
        requested_counts=counts,
        reserve_per_stratum=args.reserve_per_stratum,
        seed=args.seed,
    )
    out_dir.mkdir(parents=True, mode=0o700)
    (out_dir / "images").mkdir(mode=0o700)
    temp_root = args.temp_root.resolve() if args.temp_root else None
    render_result = render_source_rows(
        candidates,
        raw_root=raw_root,
        out_dir=out_dir,
        max_pixels=args.max_render_pixels,
        allow_pixel_failures=True,
        temp_root=temp_root,
        progress_desc="rendering blinded diagnostic-view candidates",
    )
    failed_ids = {
        normalize_view_id(record["view_id"]) for record in render_result["failures"]
    }
    panel = select_rendered_panel(
        candidates,
        failed_view_ids=failed_ids,
        requested_counts=counts,
        seed=args.seed,
    )
    metadata: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "target": target,
        "exclusions": str(exclusions_path),
        "standard_views": str(standard_path),
        "enrichment_regex": args.enrichment_regex,
        "metadata_is_reference_truth": False,
        "allowed_view_positions": sorted(ALLOWED_VIEW_POSITIONS),
        "input_pool_counts": {
            "metadata_enriched_exclusion": int(len(enriched)),
            "other_diagnostic_exclusion": int(len(other)),
            "standard_view_control": int(len(standard)),
        },
        "selected_counts": counts,
        "reserve_per_stratum": int(args.reserve_per_stratum),
        "render_failures": int(render_result["failed"]),
        "prior_view_ids_excluded": int(len(prior_ids)),
        "seed": int(args.seed),
        "max_render_pixels": int(args.max_render_pixels),
    }
    write_outputs(out_dir, panel, candidates, target=target, metadata=metadata)
    return panel[
        ["view_id", "image_path", "laterality", "view", "review_order", "stratum"]
    ]


def main() -> int:
    args = parse_args()
    manifest = run_from_args(args)
    counts = manifest["stratum"].value_counts().sort_index().to_dict()
    print(f"view QC pilot ready: target={normalize_view_qc_target(args.target)!r}")
    print(f"views={len(manifest)} strata={counts}")
    print(f"output: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
