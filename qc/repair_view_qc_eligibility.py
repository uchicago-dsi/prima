#!/usr/bin/env python3
"""Replace non-standard pilot views while preserving eligible labels and scores."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from prima.dicom_source import SOURCE_COLUMNS, require_source_columns
from prima.view_auto_qc import load_view_auto_run, save_view_auto_run
from prima.view_fallback import validate_candidate_table
from prima.view_qc import (
    default_view_qc_events_path,
    initialize_view_qc_event_log,
    load_view_qc_state,
    normalize_view_id,
    save_view_qc_state,
    validate_rendered_view_png,
    validate_view_manifest_columns,
)
from qc.audit_view_eligibility import audit_source, restricted_json
from qc.build_ranked_view_qc_pilot import safe_rendered_path

SLOT_ORDER = {("L", "CC"): 0, ("L", "MLO"): 1, ("R", "CC"): 2, ("R", "MLO"): 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--model-run", type=Path, required=True)
    parser.add_argument("--eligibility-audit", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--rendered-manifest", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-render-pixels", type=int, default=2_000_000)
    parser.add_argument("--reserve-match-columns", default="")
    parser.add_argument("--reserve-exams", type=int, default=100)
    return parser.parse_args()


def parse_columns(value: str) -> list[str]:
    columns = [column.strip() for column in value.split(",") if column.strip()]
    if len(columns) != len(set(columns)):
        raise ValueError("reserve match columns contain duplicates")
    return columns


def audit_candidates(
    candidates: pd.DataFrame,
    raw_root: Path,
    temp_root: Path,
    workers: int,
) -> pd.DataFrame:
    source_rows = candidates[["view_id", *SOURCE_COLUMNS]].to_dict("records")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return pd.DataFrame(
            list(
                executor.map(
                    lambda row: audit_source(row, raw_root, temp_root),
                    source_rows,
                )
            )
        )


def sort_replacement_choices(choices: pd.DataFrame, seed: pd.Series) -> pd.DataFrame:
    choices = choices.copy()
    choices["_same_slot"] = (
        (choices["laterality"] != seed["laterality"])
        | (choices["view"] != seed["view"])
    ).astype(int)
    choices["_slot_order"] = [
        SLOT_ORDER[(str(lat), str(view))]
        for lat, view in zip(choices["laterality"], choices["view"])
    ]
    return choices.sort_values(
        ["_same_slot", "selection_rank", "_slot_order", "view_id"],
        kind="stable",
    )


def run_from_args(args: argparse.Namespace) -> pd.DataFrame:
    manifest_path = args.manifest.resolve()
    state_path = args.state.resolve()
    model_run_path = args.model_run.resolve()
    audit_path = args.eligibility_audit.resolve()
    candidates_path = args.candidates.resolve()
    rendered_manifest_path = args.rendered_manifest.resolve()
    exclude_manifest_paths = [path.resolve() for path in args.exclude_manifest]
    raw_root = args.raw_root.resolve()
    out_dir = args.out_dir.resolve()
    temp_root = args.temp_root.resolve()
    for path in (
        manifest_path,
        state_path,
        model_run_path,
        audit_path,
        candidates_path,
        rendered_manifest_path,
        *exclude_manifest_paths,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"eligibility repair input not found: {path}")
    for path in (raw_root, temp_root):
        if not path.is_dir():
            raise FileNotFoundError(f"eligibility repair directory not found: {path}")
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite eligibility repair: {out_dir}")
    if args.workers <= 0 or args.max_render_pixels <= 0 or args.reserve_exams <= 0:
        raise ValueError("workers and maximum render pixels must be positive")
    reserve_match_columns = parse_columns(args.reserve_match_columns)

    manifest = pd.read_parquet(manifest_path)
    validate_view_manifest_columns(manifest.columns, str(manifest_path))
    manifest = manifest.copy()
    manifest["view_id"] = manifest["view_id"].map(normalize_view_id)
    if manifest["view_id"].duplicated().any():
        raise ValueError("eligibility repair manifest contains duplicate view IDs")

    state = load_view_qc_state(state_path)
    model_run = load_view_auto_run(model_run_path)
    if not model_run:
        raise ValueError("eligibility repair model run is empty")
    if model_run["target"] != state["target"]:
        raise ValueError("eligibility repair state and model run targets disagree")
    manifest_ids = set(manifest["view_id"])
    if set(state["labels"]) != manifest_ids:
        raise ValueError("eligibility repair requires a completely labeled state")
    if set(model_run["view_suggestions"]) != manifest_ids:
        raise ValueError("eligibility repair requires complete model coverage")

    audit = pd.read_parquet(audit_path)
    required_audit = {"view_id", "is_mirai_source_eligible"}
    missing_audit = sorted(required_audit - set(audit.columns))
    if missing_audit:
        raise ValueError(
            "eligibility audit is missing columns: " + ", ".join(missing_audit)
        )
    audit = audit.copy()
    audit["view_id"] = audit["view_id"].map(normalize_view_id)
    if set(audit["view_id"]) != manifest_ids:
        raise ValueError("eligibility audit does not exactly cover the pilot manifest")
    excluded_ids = set(
        audit.loc[~audit["is_mirai_source_eligible"].astype(bool), "view_id"]
    )
    if not excluded_ids:
        raise ValueError("eligibility repair found no non-standard views to replace")
    kept_ids = manifest_ids - excluded_ids

    candidates = pd.read_parquet(candidates_path)
    validate_candidate_table(candidates, str(candidates_path))
    require_source_columns(candidates.columns, str(candidates_path))
    candidates = candidates.copy()
    candidates["view_id"] = candidates["sha256"].map(normalize_view_id)
    if candidates["view_id"].duplicated().any():
        raise ValueError("candidate table contains duplicate view IDs")
    missing_match_columns = sorted(set(reserve_match_columns) - set(candidates.columns))
    if missing_match_columns:
        raise ValueError(
            "candidate table is missing reserve match columns: "
            + ", ".join(missing_match_columns)
        )
    seeds = candidates[candidates["view_id"].isin(excluded_ids)].copy()
    if len(seeds) != len(excluded_ids):
        raise RuntimeError("not every excluded view maps to one candidate source")
    if seeds["exam_id"].duplicated().any():
        raise ValueError("eligibility repair expects one pilot view per exam")

    rendered = pd.read_parquet(rendered_manifest_path)
    validate_view_manifest_columns(rendered.columns, str(rendered_manifest_path))
    rendered = rendered.copy()
    rendered["view_id"] = rendered["view_id"].map(normalize_view_id)
    if rendered["view_id"].duplicated().any():
        raise ValueError("rendered manifest contains duplicate view IDs")
    rendered_ids = set(rendered["view_id"])
    excluded_replacement_ids: set[str] = set()
    for path in exclude_manifest_paths:
        excluded_manifest = pd.read_parquet(path)
        validate_view_manifest_columns(excluded_manifest.columns, str(path))
        excluded_replacement_ids.update(
            excluded_manifest["view_id"].map(normalize_view_id)
        )

    replacement_pool = candidates[
        candidates["exam_id"].isin(set(seeds["exam_id"]))
        & ~candidates["view_id"].isin(manifest_ids)
        & ~candidates["view_id"].isin(excluded_replacement_ids)
        & candidates["view_id"].isin(rendered_ids)
    ].copy()
    if replacement_pool.empty:
        raise RuntimeError(
            "no rendered same-exam candidates can replace excluded views"
        )
    replacement_audit = audit_candidates(
        replacement_pool, raw_root, temp_root, args.workers
    )
    replacement_pool = replacement_pool.merge(
        replacement_audit[["view_id", "is_mirai_source_eligible"]],
        on="view_id",
        how="left",
        validate="one_to_one",
    )
    replacement_pool = replacement_pool[
        replacement_pool["is_mirai_source_eligible"].astype(bool)
    ].copy()

    seed_lookup = seeds.set_index("exam_id")
    chosen_rows: list[pd.Series] = []
    missing_seeds: list[pd.Series] = []
    for exam_id, seed in seed_lookup.iterrows():
        choices = replacement_pool[replacement_pool["exam_id"] == exam_id].copy()
        if choices.empty:
            missing_seeds.append(seed)
            continue
        choices = sort_replacement_choices(choices, seed)
        choice = choices.iloc[0].copy()
        choice["_replaces_seed_view_id"] = str(seed["view_id"])
        chosen_rows.append(choice)

    reserve_replacements = 0
    if missing_seeds:
        original_exam_ids = set(
            candidates[candidates["view_id"].isin(manifest_ids)]["exam_id"]
        )
        reserve_audits: list[pd.DataFrame] = []
        for seed in missing_seeds:
            reserve_pool = candidates[
                ~candidates["exam_id"].isin(original_exam_ids)
                & candidates["view_id"].isin(rendered_ids)
                & ~candidates["view_id"].isin(excluded_replacement_ids)
            ].copy()
            for column in reserve_match_columns:
                reserve_pool = reserve_pool[reserve_pool[column] == seed[column]]
            if reserve_pool.empty:
                raise RuntimeError("no reserve exams match the excluded pilot stratum")
            reserve_exam_order = (
                reserve_pool.groupby("exam_id", sort=False)["view_id"]
                .min()
                .sort_values(kind="stable")
                .head(args.reserve_exams)
                .index
            )
            reserve_pool = reserve_pool[
                reserve_pool["exam_id"].isin(set(reserve_exam_order))
            ].copy()
            reserve_audit = audit_candidates(
                reserve_pool, raw_root, temp_root, args.workers
            )
            reserve_audits.append(reserve_audit)
            reserve_pool = reserve_pool.merge(
                reserve_audit[["view_id", "is_mirai_source_eligible"]],
                on="view_id",
                how="left",
                validate="one_to_one",
            )
            reserve_pool = reserve_pool[
                reserve_pool["is_mirai_source_eligible"].astype(bool)
            ].copy()
            if reserve_pool.empty:
                raise RuntimeError("matching reserve exams contain no standard views")
            reserve_pool = sort_replacement_choices(reserve_pool, seed)
            reserve_pool = reserve_pool.sort_values(
                ["_same_slot", "selection_rank", "_slot_order", "view_id"],
                kind="stable",
            )
            choice = reserve_pool.iloc[0].copy()
            choice["_replaces_seed_view_id"] = str(seed["view_id"])
            chosen_rows.append(choice)
            original_exam_ids.add(choice["exam_id"])
            reserve_replacements += 1
        replacement_audit = pd.concat(
            [replacement_audit, *reserve_audits], ignore_index=True
        ).drop_duplicates("view_id", keep="first")
    chosen = pd.DataFrame(chosen_rows)
    if len(chosen) != len(excluded_ids) or chosen["view_id"].duplicated().any():
        raise RuntimeError(
            "eligibility repair did not choose one unique replacement per view"
        )

    original_by_id = manifest.set_index("view_id")
    replacement_records: list[dict[str, Any]] = []
    for row in chosen.to_dict("records"):
        original = original_by_id.loc[row["_replaces_seed_view_id"]]
        record = original.to_dict()
        record.update(
            {
                "view_id": str(row["view_id"]),
                "image_path": f"images/{row['view_id']}.png",
                "laterality": str(row["laterality"]),
                "view": str(row["view"]),
            }
        )
        replacement_records.append(record)

    kept = manifest[manifest["view_id"].isin(kept_ids)].copy()
    repaired = pd.concat(
        [kept, pd.DataFrame(replacement_records)], ignore_index=True
    ).sort_values("review_order", kind="stable")
    if len(repaired) != len(manifest) or repaired["view_id"].duplicated().any():
        raise RuntimeError("eligibility repair changed the pilot denominator")
    validate_view_manifest_columns(repaired.columns, "repaired view QC manifest")

    out_dir.mkdir(parents=True, mode=0o700)
    image_dir = out_dir / "images"
    image_dir.mkdir(mode=0o700)
    original_lookup = manifest.set_index("view_id")
    rendered_lookup = rendered.set_index("view_id")
    for view_id in repaired["view_id"]:
        if view_id in kept_ids:
            source = safe_rendered_path(
                manifest_path, original_lookup.loc[view_id, "image_path"]
            )
        else:
            source = safe_rendered_path(
                rendered_manifest_path, rendered_lookup.loc[view_id, "image_path"]
            )
        validate_rendered_view_png(source, max_pixels=args.max_render_pixels)
        destination = image_dir / f"{view_id}.png"
        os.link(source, destination)
        os.chmod(destination, 0o600)

    repaired_path = out_dir / "manifest.parquet"
    repaired.to_parquet(repaired_path, index=False)
    os.chmod(repaired_path, 0o600)
    repaired_state = {
        **state,
        "labels": {
            view_id: record
            for view_id, record in state["labels"].items()
            if view_id in kept_ids
        },
    }
    repaired_state_path = out_dir / "view_qc_state.json"
    saved_repaired_state = save_view_qc_state(repaired_state_path, repaired_state)
    initialize_view_qc_event_log(
        default_view_qc_events_path(repaired_state_path),
        saved_repaired_state,
        import_reviewer="system:eligibility-repair",
    )
    repaired_run = {
        **model_run,
        "view_suggestions": {
            view_id: record
            for view_id, record in model_run["view_suggestions"].items()
            if view_id in kept_ids
        },
    }
    save_view_auto_run(out_dir / "model_run.json", repaired_run)

    kept_audit = audit[audit["view_id"].isin(kept_ids)].copy()
    chosen_audit = replacement_audit[
        replacement_audit["view_id"].isin(set(chosen["view_id"]))
    ].copy()
    final_audit = pd.concat([kept_audit, chosen_audit], ignore_index=True)
    if (
        len(final_audit) != len(repaired)
        or not final_audit["is_mirai_source_eligible"].astype(bool).all()
    ):
        raise RuntimeError("repaired pilot eligibility validation failed")
    final_audit.to_parquet(out_dir / "eligibility_audit.parquet", index=False)
    os.chmod(out_dir / "eligibility_audit.parquet", 0o600)

    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "target": state["target"],
        "original_views": int(len(manifest)),
        "preserved_views": int(len(kept)),
        "replaced_views": int(len(chosen)),
        "preserved_human_labels": int(len(repaired_state["labels"])),
        "preserved_model_scores": int(len(repaired_run["view_suggestions"])),
        "reserve_exam_replacements": int(reserve_replacements),
        "explicitly_excluded_replacement_views": int(
            len(excluded_replacement_ids - manifest_ids)
        ),
    }
    restricted_json(out_dir / "repair_metadata.json", metadata)
    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Mirai-eligible repaired view-QC pilot",
                "",
                f"- target: `{state['target']}`",
                f"- views: `{len(repaired)}`",
                f"- preserved labeled views: `{len(kept)}`",
                f"- standard-view replacements awaiting review: `{len(chosen)}`",
                "",
                "Every view was verified from its original DICOM header as an",
                "unmodified, non-partial CC or MLO view. Eligible human labels and",
                "frozen model scores were preserved by SHA-256 view identity.",
                "Replacement views remain unlabeled and unscored until their separate",
                "model pass finishes.",
                "",
                f"Exact producer command: `{metadata['command']}`",
                "",
            ]
        )
    )
    os.chmod(readme, 0o600)
    return repaired


def main() -> int:
    args = parse_args()
    repaired = run_from_args(args)
    print(
        f"view QC eligibility repaired: views={len(repaired)} "
        f"reviewed={len(load_view_qc_state(args.out_dir / 'view_qc_state.json')['labels'])}"
    )
    print(f"output: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
