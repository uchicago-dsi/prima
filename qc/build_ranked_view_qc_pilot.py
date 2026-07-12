#!/usr/bin/env python3
"""Build a blinded single-target pilot from ranked and random rendered views."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from prima.view_fallback import validate_candidate_table
from prima.view_qc import (
    empty_view_qc_state,
    normalize_view_id,
    normalize_view_qc_target,
    save_view_qc_state,
    validate_rendered_view_png,
    validate_view_manifest_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--score-columns", required=True)
    parser.add_argument("--score-top-k", type=int, default=5)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--rendered-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--enriched-count", type=int, default=60)
    parser.add_argument("--random-count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--max-render-pixels", type=int, default=2_000_000)
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    """Load one CSV or Parquet table without format guessing."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("score table must be CSV or Parquet")


def parse_score_columns(value: str) -> list[str]:
    """Parse a nonempty comma-separated list without duplicate columns."""
    columns = [column.strip() for column in value.split(",") if column.strip()]
    if not columns:
        raise ValueError("--score-columns must name at least one column")
    if len(columns) != len(set(columns)):
        raise ValueError("--score-columns contains duplicates")
    return columns


def safe_rendered_path(manifest_path: Path, relative_value: object) -> Path:
    """Resolve a rendered image while keeping it inside its manifest root."""
    root = manifest_path.parent.resolve()
    relative = Path(str(relative_value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("rendered image path must be safe and relative")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("rendered image escapes its manifest root") from error
    if not path.is_file():
        raise FileNotFoundError(f"rendered image not found: {path}")
    return path


def run_from_args(args: argparse.Namespace) -> pd.DataFrame:
    """Build and return the deidentified review manifest."""
    scores_path = args.scores.resolve()
    candidates_path = args.candidates.resolve()
    rendered_manifest_path = args.rendered_manifest.resolve()
    out_dir = args.out_dir.resolve()
    target = normalize_view_qc_target(args.target)
    score_columns = parse_score_columns(args.score_columns)
    for path in (scores_path, candidates_path, rendered_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"pilot input not found: {path}")
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite view QC pilot: {out_dir}")
    if args.enriched_count <= 0 or args.random_count <= 0:
        raise ValueError("pilot stratum counts must be positive")
    if args.score_top_k <= 0:
        raise ValueError("--score-top-k must be positive")
    if args.max_render_pixels <= 0:
        raise ValueError("--max-render-pixels must be positive")

    scores = load_table(scores_path)
    required_scores = {"exam_id", "sop_instance_uid", *score_columns}
    missing_scores = sorted(required_scores - set(scores.columns))
    if missing_scores:
        raise ValueError("score table is missing columns: " + ", ".join(missing_scores))
    if scores.duplicated(["exam_id", "sop_instance_uid"]).any():
        raise ValueError("score table contains duplicate exam/source rows")
    if "load_error" in scores.columns:
        load_error = scores["load_error"]
        scores = scores[
            load_error.isna() | load_error.astype(str).str.strip().eq("")
        ].copy()
    scores = scores[["exam_id", "sop_instance_uid", *score_columns]].copy()

    candidates = pd.read_parquet(candidates_path)
    validate_candidate_table(candidates, str(candidates_path))
    selected = candidates[candidates["is_selected"].astype(bool)][
        ["exam_id", "sop_instance_uid", "sha256", "laterality", "view"]
    ].copy()
    selected["view_id"] = selected["sha256"].map(normalize_view_id)
    matched = scores.merge(
        selected,
        on=["exam_id", "sop_instance_uid"],
        how="inner",
        validate="one_to_one",
    )
    if matched.empty:
        raise RuntimeError("score table has no selected views in the candidate table")

    values = (
        matched[score_columns]
        .apply(pd.to_numeric, errors="coerce")
        .abs()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(upper=20.0)
    )
    top_k = min(int(args.score_top_k), len(score_columns))
    matched["enrichment_score"] = np.sort(values.to_numpy(), axis=1)[:, -top_k:].mean(
        axis=1
    )

    ranked = (
        matched.sort_values(
            ["enrichment_score", "view_id"],
            ascending=[False, True],
            kind="stable",
        )
        .drop_duplicates("exam_id", keep="first")
        .reset_index(drop=True)
    )
    if len(ranked) < args.enriched_count:
        raise ValueError(
            f"only {len(ranked)} distinct exams are eligible for enriched sampling"
        )
    enriched = ranked.head(args.enriched_count).copy()
    enriched["stratum"] = "score_enriched"

    remaining = matched[~matched["exam_id"].isin(enriched["exam_id"])].copy()
    random_candidates = (
        remaining.sample(frac=1.0, random_state=args.seed)
        .drop_duplicates("exam_id", keep="first")
        .reset_index(drop=True)
    )
    if len(random_candidates) < args.random_count:
        raise ValueError(
            f"only {len(random_candidates)} distinct exams remain for random sampling"
        )
    controls = random_candidates.sample(
        n=args.random_count, random_state=args.seed + 1
    ).copy()
    controls["stratum"] = "random_control"

    sampled = pd.concat([enriched, controls], ignore_index=True)
    if sampled["exam_id"].duplicated().any() or sampled["view_id"].duplicated().any():
        raise RuntimeError("pilot sampling did not preserve exam and view disjointness")
    sampled = sampled.sample(frac=1.0, random_state=args.seed + 2).reset_index(
        drop=True
    )
    sampled["review_order"] = range(1, len(sampled) + 1)

    rendered = pd.read_parquet(rendered_manifest_path)
    validate_view_manifest_columns(rendered.columns, str(rendered_manifest_path))
    rendered = rendered.copy()
    rendered["view_id"] = rendered["view_id"].map(normalize_view_id)
    if rendered["view_id"].duplicated().any():
        raise ValueError("rendered manifest contains duplicate view IDs")
    rendered_lookup = rendered.set_index("view_id")
    missing_rendered = set(sampled["view_id"]) - set(rendered_lookup.index)
    if missing_rendered:
        raise RuntimeError(
            f"rendered manifest is missing {len(missing_rendered)} sampled views"
        )

    out_dir.mkdir(parents=True, mode=0o700)
    image_dir = out_dir / "images"
    image_dir.mkdir(mode=0o700)
    records: list[dict[str, object]] = []
    for row in sampled.to_dict("records"):
        view_id = str(row["view_id"])
        rendered_row = rendered_lookup.loc[view_id]
        if str(rendered_row["laterality"]) != str(row["laterality"]) or str(
            rendered_row["view"]
        ) != str(row["view"]):
            raise RuntimeError("sampled and rendered view metadata disagree")
        source = safe_rendered_path(rendered_manifest_path, rendered_row["image_path"])
        validate_rendered_view_png(source, max_pixels=args.max_render_pixels)
        destination = image_dir / f"{view_id}.png"
        os.link(source, destination)
        os.chmod(destination, 0o600)
        records.append(
            {
                "view_id": view_id,
                "image_path": f"images/{view_id}.png",
                "laterality": str(row["laterality"]),
                "view": str(row["view"]),
                "review_order": int(row["review_order"]),
                "stratum": str(row["stratum"]),
            }
        )

    manifest = pd.DataFrame(records).sort_values("review_order", kind="stable")
    manifest_path = out_dir / "manifest.parquet"
    state_path = out_dir / "view_qc_state.json"
    metadata_path = out_dir / "sampling_metadata.json"
    manifest.to_parquet(manifest_path, index=False)
    os.chmod(manifest_path, 0o600)
    save_view_qc_state(state_path, empty_view_qc_state(target))
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "target": target,
        "scores": str(scores_path),
        "score_columns": score_columns,
        "score_top_k": top_k,
        "candidates": str(candidates_path),
        "rendered_manifest": str(rendered_manifest_path),
        "matched_views": int(len(matched)),
        "matched_exams": int(matched["exam_id"].nunique()),
        "enriched_count": int(args.enriched_count),
        "random_count": int(args.random_count),
        "seed": int(args.seed),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    os.chmod(metadata_path, 0o600)
    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Blinded ranked view-QC pilot",
                "",
                f"- target: `{target}`",
                f"- total views: `{len(manifest)}`",
                f"- score-enriched views: `{args.enriched_count}`",
                f"- random controls: `{args.random_count}`",
                f"- seed: `{args.seed}`",
                "",
                "Sampling uses only the requested numeric ranking columns and a random",
                "control stratum. It does not use QC tags as labels or eligibility filters.",
                "The browser manifest contains no exam, patient, or source identifiers.",
                "Model outputs must remain hidden until human review is complete.",
                "",
                f"Exact producer command: `{metadata['command']}`",
                "",
            ]
        )
    )
    os.chmod(readme, 0o600)
    return manifest


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
