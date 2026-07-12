#!/usr/bin/env python3
"""Build a blinded human audit of model-driven exact-slot fallback decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import pandas as pd

from prima.view_auto_qc import load_view_auto_run, save_view_auto_run
from prima.view_fallback import validate_candidate_table
from prima.view_qc import (
    empty_view_qc_state,
    normalize_view_id,
    save_view_qc_state,
    validate_rendered_view_png,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--view-auto-run", type=Path, required=True)
    parser.add_argument("--render-complete", type=Path, required=True)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="Prior review manifest whose exact-slot groups must not be resampled",
    )
    parser.add_argument("--alternate-slots", type=int, default=50)
    parser.add_argument("--no-pass-slots", type=int, default=25)
    parser.add_argument("--original-pass-views", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--max-render-pixels", type=int, default=2_000_000)
    return parser.parse_args()


def sample_rows(rows: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count < 0:
        raise ValueError("audit sample sizes cannot be negative")
    if count == 0:
        return rows.iloc[0:0].copy()
    if len(rows) < count:
        raise ValueError(
            f"audit requested {count} rows but only {len(rows)} are eligible"
        )
    return rows.sample(n=count, random_state=seed).copy()


def audit_group_id(exam_id: object, laterality: object, view: object) -> str:
    payload = f"{exam_id}|{laterality}|{view}".encode()
    return hashlib.sha256(payload).hexdigest()


def load_excluded_audit_group_ids(paths: list[Path]) -> set[str]:
    """Load deidentified exact-slot group IDs from prior review manifests."""
    excluded: set[str] = set()
    for raw_path in paths:
        path = raw_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"excluded audit manifest not found: {path}")
        manifest = pd.read_parquet(path)
        if manifest.empty:
            raise ValueError(f"excluded audit manifest is empty: {path}")
        if "audit_group_id" in manifest.columns:
            if manifest["audit_group_id"].isna().any():
                raise ValueError(f"excluded audit manifest has null group IDs: {path}")
            values = manifest["audit_group_id"].astype(str).str.strip()
            if (values == "").any():
                raise ValueError(f"excluded audit manifest has blank group IDs: {path}")
            excluded.update(values)
            continue
        required = {"exam_id", "laterality", "view"}
        missing = sorted(required - set(manifest.columns))
        if missing:
            raise ValueError(
                f"excluded audit manifest cannot identify exact-slot groups: {path}; "
                f"missing {', '.join(missing)}"
            )
        if manifest[list(required)].isna().any(axis=None):
            raise ValueError(f"excluded audit manifest has null slot fields: {path}")
        if (
            manifest[list(required)]
            .astype(str)
            .apply(lambda values: values.str.strip().eq(""))
            .any(axis=None)
        ):
            raise ValueError(f"excluded audit manifest has blank slot fields: {path}")
        excluded.update(
            audit_group_id(exam_id, laterality, view)
            for exam_id, laterality, view in manifest[
                ["exam_id", "laterality", "view"]
            ].itertuples(index=False, name=None)
        )
    return excluded


def main() -> int:
    args = parse_args()
    candidates_path = args.candidates.resolve()
    decisions_path = args.decisions.resolve()
    run_path = args.view_auto_run.resolve()
    render_complete_path = args.render_complete.resolve()
    campaign_dir = args.campaign_dir.resolve()
    out_dir = args.out_dir.resolve()
    excluded_group_ids = load_excluded_audit_group_ids(args.exclude_manifest)
    for path in (candidates_path, decisions_path, run_path, render_complete_path):
        if not path.is_file():
            raise FileNotFoundError(f"required fallback audit input not found: {path}")
    if not (campaign_dir / "render_complete.json").is_file():
        raise FileNotFoundError("full candidate render campaign is not validated")
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite fallback audit: {out_dir}")
    if args.max_render_pixels <= 0:
        raise ValueError("--max-render-pixels must be positive")

    candidates = pd.read_parquet(candidates_path)
    validate_candidate_table(candidates, str(candidates_path))
    candidates = candidates.copy()
    candidates["view_id"] = candidates["sha256"].map(normalize_view_id)
    decisions = pd.read_parquet(decisions_path)
    required_decisions = {
        "exam_id",
        "laterality",
        "view",
        "original_view_id",
        "selected_view_id",
        "selected_candidate_rank",
        "fallback_status",
    }
    missing = sorted(required_decisions - set(decisions.columns))
    if missing:
        raise ValueError(
            "fallback decisions are missing columns: " + ", ".join(missing)
        )
    run = load_view_auto_run(run_path)
    render_complete = json.loads(render_complete_path.read_text())
    render_failures = {
        normalize_view_id(view_id)
        for view_id in render_complete.get("failed_view_ids", [])
    }
    model_ids = set(run["view_suggestions"])
    if model_ids & render_failures or model_ids | render_failures != set(
        candidates["view_id"]
    ):
        raise RuntimeError(
            "model results plus render failures do not cover the candidate inventory"
        )

    failed_group_keys = set(
        map(
            tuple,
            candidates[candidates["view_id"].isin(render_failures)][
                ["exam_id", "laterality", "view"]
            ].to_numpy(),
        )
    )
    decisions["_group_key"] = list(
        map(
            tuple,
            decisions[["exam_id", "laterality", "view"]].to_numpy(),
        )
    )
    decisions["_audit_group_id"] = [
        audit_group_id(exam_id, laterality, view)
        for exam_id, laterality, view in decisions[
            ["exam_id", "laterality", "view"]
        ].itertuples(index=False, name=None)
    ]
    visually_auditable = decisions[
        ~decisions["_group_key"].isin(failed_group_keys)
        & ~decisions["_audit_group_id"].isin(excluded_group_ids)
    ]

    alternate = sample_rows(
        visually_auditable[visually_auditable["fallback_status"] == "alternate_pass"],
        args.alternate_slots,
        args.seed,
    )
    no_pass = sample_rows(
        visually_auditable[
            visually_auditable["fallback_status"] == "no_passing_candidate"
        ],
        args.no_pass_slots,
        args.seed + 1,
    )
    controls = sample_rows(
        visually_auditable[visually_auditable["fallback_status"] == "original_pass"],
        args.original_pass_views,
        args.seed + 2,
    )

    keyed_candidates = {
        (str(exam_id), str(laterality), str(view)): rows.sort_values(
            "selection_rank", kind="stable"
        )
        for (exam_id, laterality, view), rows in candidates.groupby(
            ["exam_id", "laterality", "view"], sort=False
        )
    }
    audit_records: list[dict[str, object]] = []
    for stratum, sampled in (
        ("alternate_pass", alternate),
        ("no_passing_candidate", no_pass),
        ("original_pass_control", controls),
    ):
        for decision in sampled.to_dict("records"):
            key = (
                str(decision["exam_id"]),
                str(decision["laterality"]),
                str(decision["view"]),
            )
            rows = keyed_candidates[key]
            group_id = audit_group_id(*key)
            for row in rows.to_dict("records"):
                audit_records.append(
                    {
                        "view_id": str(row["view_id"]),
                        "laterality": str(row["laterality"]),
                        "view": str(row["view"]),
                        "selection_rank": int(row["selection_rank"]),
                        "candidate_count": int(len(rows)),
                        "audit_group_id": group_id,
                        "stratum": stratum,
                    }
                )
    audit = pd.DataFrame(audit_records)
    if audit.empty:
        raise RuntimeError("fallback audit sampling produced no views")
    if audit["view_id"].duplicated().any():
        raise RuntimeError("fallback audit contains duplicate view IDs")
    records = audit.to_dict("records")
    random.Random(args.seed).shuffle(records)
    for review_order, record in enumerate(records, start=1):
        record["review_order"] = review_order
        record["image_path"] = f"images/{record['view_id']}.png"
    manifest = pd.DataFrame(records)

    out_dir.mkdir(parents=True, mode=0o700)
    image_dir = out_dir / "images"
    image_dir.mkdir(mode=0o700)
    for record in manifest.to_dict("records"):
        source = campaign_dir / "images" / f"{record['view_id']}.png"
        validate_rendered_view_png(source, max_pixels=args.max_render_pixels)
        destination = image_dir / source.name
        shutil.copyfile(source, destination)
        os.chmod(destination, 0o600)

    manifest_path = out_dir / "manifest.parquet"
    state_path = out_dir / "view_qc_state.json"
    hidden_run_path = out_dir / "hidden_model_run.json"
    manifest.to_parquet(manifest_path, index=False)
    os.chmod(manifest_path, 0o600)
    save_view_qc_state(state_path, empty_view_qc_state())
    subset_run = {
        **run,
        "run_id": str(run["run_id"]) + "_fallback_audit",
        "view_suggestions": {
            view_id: run["view_suggestions"][view_id] for view_id in manifest["view_id"]
        },
    }
    save_view_auto_run(hidden_run_path, subset_run)
    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Blinded model-fallback audit",
                "",
                f"- total audit views: `{len(manifest)}`",
                f"- sampled alternate-pass slots: `{len(alternate)}`",
                f"- sampled no-passing-candidate slots: `{len(no_pass)}`",
                f"- sampled original-pass controls: `{len(controls)}`",
                f"- deterministic render-failure slots excluded: `{len(failed_group_keys)}`",
                f"- prior exact-slot groups excluded: `{len(excluded_group_ids)}`",
                f"- seed: `{args.seed}`",
                "",
                "The gallery exposes only individual views. Sampling strata, candidate",
                "ranks, group hashes, and the hidden model run remain server-side.",
                "Every candidate in each sampled exact-slot group is included so selection",
                "and exhaustion decisions can be evaluated without an unseen tail.",
                "",
            ]
        )
    )
    os.chmod(readme, 0o600)
    counts = manifest["stratum"].value_counts().sort_index().to_dict()
    print(f"fallback audit ready: views={len(manifest)}")
    print(
        "stratum views: " + ", ".join(f"{key}={value}" for key, value in counts.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
