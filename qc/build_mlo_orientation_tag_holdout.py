#!/usr/bin/env python3
"""Build a matched prospective holdout of natural MLO display orientations."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from prima.dicom_source import (
    SOURCE_COLUMNS,
    require_source_columns,
    require_valid_sources,
)
from prima.mlo_orientation import mlo_orientation_label_from_column_direction
from prima.view_fallback import validate_candidate_table
from prima.view_qc import normalize_view_id, validate_rendered_view_png
from qc.build_mirai_input_whole_exam_audit import (
    DEFAULT_TARGET,
    load_frozen_candidate_spec,
    load_prior_exclusions,
)

MATCH_COLUMNS = (
    "laterality",
    "device_manufacturer",
    "device_model",
    "has_implant",
)
REFERENCE_LABELS = {"UPRIGHT", "INVERTED"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--dicom-tags", type=Path, required=True)
    parser.add_argument("--render-campaign", type=Path, required=True)
    parser.add_argument("--candidate-spec", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-source-manifest", type=Path, action="append", default=[]
    )
    parser.add_argument("--inverted-sources", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--max-render-pixels", type=int, default=2_000_000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sample_key(seed: int, view_id: object, namespace: str) -> str:
    payload = f"{seed}|{namespace}|{normalize_view_id(view_id)}".encode()
    return hashlib.sha256(payload).hexdigest()


def orientation_label_from_cached_tag(value: object) -> str:
    """Parse the serialized PatientOrientation representation in dicom_tags."""
    if not isinstance(value, str) or not value.strip():
        return "UNKNOWN"
    text = value.strip()
    try:
        directions = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return "UNKNOWN"
    if not isinstance(directions, (list, tuple)) or len(directions) < 2:
        return "UNKNOWN"
    return mlo_orientation_label_from_column_direction(directions[1])


def restricted_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")
    os.chmod(path, 0o600)


def normalize_match_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in ("laterality", "device_manufacturer", "device_model"):
        if normalized[column].isna().any():
            raise ValueError(f"orientation matching column contains nulls: {column}")
        normalized[column] = normalized[column].astype(str).str.strip()
        if normalized[column].eq("").any():
            raise ValueError(f"orientation matching column contains blanks: {column}")
    if normalized["has_implant"].isna().any():
        raise ValueError("orientation matching column contains nulls: has_implant")
    normalized["has_implant"] = normalized["has_implant"].astype(bool)
    return normalized


def complete_match_metadata(frame: pd.DataFrame) -> pd.Series:
    complete = frame["has_implant"].notna()
    for column in ("laterality", "device_manufacturer", "device_model"):
        complete &= frame[column].notna() & frame[column].astype(str).str.strip().ne("")
    return complete


def choose_unique_sources(
    candidates: pd.DataFrame,
    *,
    count: int,
    used_patients: set[str],
    used_exams: set[str],
) -> pd.DataFrame:
    chosen: list[pd.Series] = []
    for _, row in candidates.sort_values(
        ["sample_key", "view_id"], kind="stable"
    ).iterrows():
        patient = str(row["patient_id"])
        exam = str(row["exam_id"])
        if patient in used_patients or exam in used_exams:
            continue
        chosen.append(row)
        used_patients.add(patient)
        used_exams.add(exam)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise RuntimeError(
            f"could select only {len(chosen)} of {count} patient/exam-unique sources"
        )
    return pd.DataFrame(chosen).reset_index(drop=True)


def choose_matched_controls(
    upright: pd.DataFrame,
    inverted: pd.DataFrame,
    *,
    used_patients: set[str],
    used_exams: set[str],
) -> pd.DataFrame:
    controls: list[pd.DataFrame] = []
    strata = inverted.groupby(list(MATCH_COLUMNS), dropna=False, sort=True).size()
    for raw_key, required in strata.items():
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        mask = pd.Series(True, index=upright.index)
        for column, value in zip(MATCH_COLUMNS, key):
            mask &= upright[column].eq(value)
        selected = choose_unique_sources(
            upright.loc[mask],
            count=int(required),
            used_patients=used_patients,
            used_exams=used_exams,
        )
        controls.append(selected)
    matched = pd.concat(controls, ignore_index=True)
    if len(matched) != len(inverted):
        raise RuntimeError("upright matching did not preserve one-to-one class balance")
    return matched


def load_orientation_tags(path: Path) -> pd.DataFrame:
    tags = pd.read_parquet(path, columns=["sop_instance_uid", "PatientOrientation"])
    if tags["sop_instance_uid"].isna().any():
        raise ValueError("DICOM tag table contains null SOP Instance UIDs")
    tags["sop_instance_uid"] = tags["sop_instance_uid"].astype(str)
    conflict_counts = tags.groupby("sop_instance_uid", sort=False)[
        "PatientOrientation"
    ].nunique(dropna=False)
    if conflict_counts.gt(1).any():
        raise ValueError("DICOM tag table has conflicting PatientOrientation values")
    tags = tags.drop_duplicates("sop_instance_uid").copy()
    tags["reference_label"] = tags["PatientOrientation"].map(
        orientation_label_from_cached_tag
    )
    return tags[["sop_instance_uid", "reference_label"]]


def run_from_args(args: argparse.Namespace) -> pd.DataFrame:
    builder_path = Path(__file__).resolve()
    candidates_path = args.candidates.resolve()
    tags_path = args.dicom_tags.resolve()
    render_campaign = args.render_campaign.resolve()
    candidate_spec_path = args.candidate_spec.resolve()
    out_dir = args.out_dir.resolve()
    campaign_path = render_campaign / "campaign.json"
    render_manifest_path = render_campaign / "manifest.parquet"
    render_complete_path = render_campaign / "render_complete.json"
    for path in (
        candidates_path,
        tags_path,
        campaign_path,
        render_manifest_path,
        render_complete_path,
        candidate_spec_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"orientation holdout input not found: {path}")
    if not (render_campaign / "images").is_dir():
        raise FileNotFoundError("orientation render campaign has no images directory")
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite orientation holdout: {out_dir}")
    count = int(args.inverted_sources)
    if count <= 0:
        raise ValueError("--inverted-sources must be positive")
    if int(args.max_render_pixels) <= 0:
        raise ValueError("--max-render-pixels must be positive")
    candidate_spec = load_frozen_candidate_spec(
        candidate_spec_path, target=DEFAULT_TARGET
    )
    orientation_candidate = candidate_spec.get("smolvlm_mlo_orientation_layer")
    if not isinstance(orientation_candidate, dict):
        raise ValueError("candidate specification has no SmolVLM orientation layer")
    minimum_inversion_contrast = orientation_candidate.get("minimum_inversion_contrast")
    if (
        not isinstance(minimum_inversion_contrast, (int, float))
        or float(minimum_inversion_contrast) <= 0
    ):
        raise ValueError("candidate specification has no positive orientation floor")
    adapter_sha256 = str(orientation_candidate.get("adapter_sha256", ""))
    if len(adapter_sha256) != 64:
        raise ValueError(
            "candidate specification has no valid orientation adapter hash"
        )

    candidates = pd.read_parquet(candidates_path)
    validate_candidate_table(candidates, str(candidates_path))
    require_source_columns(candidates.columns, str(candidates_path))
    require_valid_sources(
        candidates[list(SOURCE_COLUMNS)].to_dict("records"), str(candidates_path)
    )
    required_columns = {
        "patient_id",
        "exam_id",
        "view",
        "laterality",
        "device_manufacturer",
        "device_model",
        "has_implant",
    }
    missing = sorted(required_columns - set(candidates.columns))
    if missing:
        raise ValueError("candidate table is missing columns: " + ", ".join(missing))
    candidates = candidates.copy()
    candidates["view_id"] = candidates["sha256"].map(normalize_view_id)
    if candidates["view_id"].duplicated().any():
        raise ValueError("candidate table contains duplicate view IDs")

    campaign = json.loads(campaign_path.read_text())
    if campaign.get("candidate_table_sha256") != sha256_file(candidates_path):
        raise RuntimeError("render campaign and candidate table hashes disagree")
    rendered = pd.read_parquet(render_manifest_path)
    rendered_ids = set(rendered["view_id"].map(normalize_view_id))
    if rendered_ids != set(candidates["view_id"]):
        raise RuntimeError("render campaign does not cover the candidate table")
    render_complete = json.loads(render_complete_path.read_text())
    failed_ids = {
        normalize_view_id(value) for value in render_complete.get("failed_view_ids", [])
    }

    prior_patients, prior_exams, prior_views = load_prior_exclusions(
        args.exclude_source_manifest
    )
    eligible = candidates[
        candidates["view"].eq("MLO")
        & ~candidates["patient_id"].astype(str).isin(prior_patients)
        & ~candidates["exam_id"].astype(str).isin(prior_exams)
        & ~candidates["view_id"].isin(prior_views | failed_ids)
    ].copy()
    complete_match = complete_match_metadata(eligible)
    incomplete_match_sources_excluded = int((~complete_match).sum())
    eligible = eligible[complete_match].copy()
    eligible = normalize_match_columns(eligible)
    orientation_tags = load_orientation_tags(tags_path)
    eligible = eligible.merge(
        orientation_tags,
        on="sop_instance_uid",
        how="left",
        validate="many_to_one",
    )
    eligible = eligible[eligible["reference_label"].isin(REFERENCE_LABELS)].copy()
    eligible["sample_key"] = eligible["view_id"].map(
        lambda value: stable_sample_key(int(args.seed), value, "natural-orientation")
    )

    used_patients: set[str] = set()
    used_exams: set[str] = set()
    inverted = choose_unique_sources(
        eligible[eligible["reference_label"].eq("INVERTED")],
        count=count,
        used_patients=used_patients,
        used_exams=used_exams,
    )
    upright_pool = eligible[eligible["reference_label"].eq("UPRIGHT")].copy()
    upright_pool["sample_key"] = upright_pool["view_id"].map(
        lambda value: stable_sample_key(int(args.seed), value, "upright-control")
    )
    upright = choose_matched_controls(
        upright_pool,
        inverted,
        used_patients=used_patients,
        used_exams=used_exams,
    )
    selected = pd.concat([inverted, upright], ignore_index=True)
    selected["review_key"] = selected["view_id"].map(
        lambda value: stable_sample_key(int(args.seed), value, "review-order")
    )
    selected = selected.sort_values(
        ["review_key", "view_id"], kind="stable"
    ).reset_index(drop=True)
    selected["review_order"] = range(1, len(selected) + 1)
    selected["image_path"] = selected["view_id"].map(
        lambda value: f"images/{value}.png"
    )
    if selected["patient_id"].astype(str).nunique() != len(selected):
        raise RuntimeError("orientation holdout is not patient-disjoint by source")
    if selected["exam_id"].astype(str).nunique() != len(selected):
        raise RuntimeError("orientation holdout is not exam-disjoint by source")

    out_dir.mkdir(parents=True, mode=0o700)
    image_dir = out_dir / "images"
    image_dir.mkdir(mode=0o700)
    for view_id in selected["view_id"]:
        source = render_campaign / "images" / f"{view_id}.png"
        validate_rendered_view_png(source, max_pixels=int(args.max_render_pixels))
        destination = image_dir / source.name
        shutil.copyfile(source, destination)
        os.chmod(destination, 0o600)

    manifest = selected[
        ["view_id", "image_path", "laterality", "view", "review_order"]
    ].copy()
    source_manifest = selected.drop(columns=["sample_key", "review_key"]).copy()
    for path, table in (
        (out_dir / "manifest.parquet", manifest),
        (out_dir / "source_manifest.parquet", source_manifest),
    ):
        table.to_parquet(path, index=False)
        os.chmod(path, 0o600)

    match_counts = (
        selected.groupby(["reference_label", *MATCH_COLUMNS], dropna=False)
        .size()
        .reset_index(name="sources")
    )
    match_summary = [
        {
            "reference_label": str(row.reference_label),
            "laterality": str(row.laterality),
            "device_manufacturer": str(row.device_manufacturer),
            "device_model": str(row.device_model),
            "has_implant": bool(row.has_implant),
            "sources": int(row.sources),
        }
        for row in match_counts.itertuples(index=False)
    ]
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "builder": str(builder_path),
        "builder_sha256": sha256_file(builder_path),
        "seed": int(args.seed),
        "candidates": str(candidates_path),
        "candidates_sha256": sha256_file(candidates_path),
        "dicom_tags": str(tags_path),
        "dicom_tags_sha256": sha256_file(tags_path),
        "render_campaign": str(render_campaign),
        "render_campaign_sha256": sha256_file(campaign_path),
        "candidate_spec": str(candidate_spec_path),
        "candidate_spec_sha256": sha256_file(candidate_spec_path),
        "candidate_name": candidate_spec["candidate_name"],
        "prior_source_manifests": [
            str(path.resolve()) for path in args.exclude_source_manifest
        ],
        "prior_patients_excluded": len(prior_patients),
        "prior_exams_excluded": len(prior_exams),
        "prior_views_excluded": len(prior_views),
        "incomplete_match_sources_excluded": incomplete_match_sources_excluded,
        "eligible_tag_labeled_mlo_sources": int(len(eligible)),
        "natural_inverted_pool_sources": int(
            eligible["reference_label"].eq("INVERTED").sum()
        ),
        "sources": int(len(selected)),
        "patients": int(selected["patient_id"].astype(str).nunique()),
        "exams": int(selected["exam_id"].astype(str).nunique()),
        "reference_counts": {
            str(key): int(value)
            for key, value in selected["reference_label"].value_counts().items()
        },
        "exact_match_columns": list(MATCH_COLUMNS),
        "exact_match_counts": match_summary,
        "cached_tags_are_sampling_labels_pending_raw_dicom_verification": True,
    }
    restricted_json(out_dir / "sampling_metadata.json", metadata)
    protocol = {
        "schema_version": 1,
        "status": "frozen_before_orientation_inference",
        "decision": "prospectively evaluate the frozen paired SmolVLM orientation action on a balanced set of natural DICOM-labeled inverted MLO sources and exact matched upright controls",
        "candidate_name": candidate_spec["candidate_name"],
        "candidate_spec": str(candidate_spec_path),
        "candidate_spec_sha256": sha256_file(candidate_spec_path),
        "sampling": {
            "natural_inverted_sources": count,
            "natural_upright_sources": count,
            "patient_and_exam_disjoint": True,
            "exact_match_columns": list(MATCH_COLUMNS),
            "cached_tag_role": "discovery and deterministic sampling only; every selected label must be verified from its durable raw DICOM before inference",
        },
        "manifest": str((out_dir / "manifest.parquet").resolve()),
        "manifest_sha256": sha256_file(out_dir / "manifest.parquet"),
        "source_manifest": str((out_dir / "source_manifest.parquet").resolve()),
        "source_manifest_sha256": sha256_file(out_dir / "source_manifest.parquet"),
        "gate": {
            "minimum_natural_inversion_sensitivity": 0.95,
            "minimum_natural_upright_specificity": 0.90,
            "minimum_pair_accuracy": 0.95,
        },
        "orientation_model_repo_id": orientation_candidate.get("model_repo_id"),
        "orientation_model_revision": orientation_candidate.get("model_revision"),
        "orientation_adapter_sha256": adapter_sha256,
        "minimum_inversion_contrast": float(minimum_inversion_contrast),
        "action_rule": "reject an original MLO only when paired_logit_contrast is at most the negative frozen minimum inversion contrast; otherwise abstain",
        "stop_rule": "run the frozen adapter and action floor once after raw-DICOM label verification; do not tune on this holdout",
    }
    restricted_json(out_dir / "protocol.json", protocol)
    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Prospective natural-MLO orientation holdout",
                "",
                f"- sources: `{len(selected)}` (`{count}` inverted, `{count}` upright)",
                f"- patients/exams: `{metadata['patients']}` / `{metadata['exams']}`",
                f"- exact matching: `{', '.join(MATCH_COLUMNS)}`",
                f"- seed: `{args.seed}`",
                f"- frozen candidate: `{candidate_spec['candidate_name']}`",
                "",
                "Cached PatientOrientation values are used only to discover and sample a",
                "balanced natural-orientation challenge. The durable raw DICOM tag for",
                "every selected source must agree before any model inference. Model weights,",
                "paired transform, contrast floor, gates, and the one-shot stop rule are",
                "frozen in `protocol.json`.",
                "",
            ]
        )
    )
    os.chmod(readme, 0o600)
    return manifest


def main() -> int:
    args = parse_args()
    manifest = run_from_args(args)
    print(f"natural MLO orientation holdout ready: sources={len(manifest)}")
    print(f"output: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
