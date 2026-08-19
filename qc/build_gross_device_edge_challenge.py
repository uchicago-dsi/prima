#!/usr/bin/env python3
"""Prepare and assemble a disjoint prompt-only gross-device challenge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from prima.dicom_source import (
    SOURCE_COLUMNS,
    SOURCE_ARCHIVE_COLUMN,
    SOURCE_MEMBER_COLUMN,
    require_source_columns,
    require_valid_sources,
)
from prima.view_auto_qc import (
    load_view_auto_run,
    save_view_auto_run,
    view_suggestion_is_target_present,
)
from prima.view_fallback import validate_candidate_table
from prima.view_qc import (
    default_view_qc_events_path,
    empty_view_qc_state,
    initialize_view_qc_event_log,
    normalize_view_id,
    save_view_qc_state,
)

TARGET = "visible gross implanted or procedural device"
DEFAULT_SEED = 20260724
DEFAULT_QUOTAS = {
    "candidate_only_edge_high": 40,
    "candidate_only_other_high": 20,
    "both_high": 30,
    "baseline_only_high": 30,
    "either_lower_confidence": 20,
    "both_negative_random": 20,
}
STRATUM_ORDER = tuple(DEFAULT_QUOTAS)
EDGE_WORDS = re.compile(
    r"\b(?:border|corner|crop|cropped|edge|partial|partly|superior)\b",
    flags=re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def require_columns(frame: pd.DataFrame, columns: set[str], context: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing columns: {', '.join(missing)}")


def load_prior_reference_universe(
    paths: list[Path],
) -> tuple[set[str], set[str], set[str], list[dict[str, Any]]]:
    patients: set[str] = set()
    exams: set[str] = set()
    views: set[str] = set()
    provenance: list[dict[str, Any]] = []
    for raw_path in paths:
        path = raw_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"prior-reference source table not found: {path}")
        frame = pd.read_parquet(path)
        require_columns(frame, {"patient_id", "exam_id", "view_id"}, str(path))
        patients.update(frame["patient_id"].astype(str))
        exams.update(frame["exam_id"].astype(str))
        views.update(frame["view_id"].map(normalize_view_id))
        provenance.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": int(len(frame)),
            }
        )
    return patients, exams, views, provenance


def load_model_spec(path: Path, model_key: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "models" in payload:
        payload = payload["models"]
    if isinstance(payload, list):
        matches = [row for row in payload if row.get("key") == model_key]
        if len(matches) != 1:
            raise ValueError(f"model registry does not contain one {model_key!r}")
        return matches[0]
    if isinstance(payload, dict) and model_key in payload:
        value = payload[model_key]
        if not isinstance(value, dict):
            raise ValueError("model registry entry must be an object")
        return {"key": model_key, **value}
    raise ValueError(f"model registry does not contain {model_key!r}")


def prepare(args: argparse.Namespace) -> int:
    candidates_path = args.candidates.resolve()
    rendered_campaign = args.rendered_campaign.resolve()
    out_dir = args.out_dir.resolve()
    spec_path = args.spec_output.resolve()
    baseline_prompt = args.baseline_prompt.resolve()
    candidate_prompt = args.candidate_prompt.resolve()
    registry_path = args.model_registry.resolve()
    for path in (
        candidates_path,
        rendered_campaign / "render_sources.parquet",
        rendered_campaign / "campaign.json",
        baseline_prompt,
        candidate_prompt,
        registry_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required challenge input not found: {path}")
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite mining campaign: {out_dir}")
    if spec_path.exists():
        raise FileExistsError(f"refusing to overwrite challenge spec: {spec_path}")
    if args.inference_shards <= 0:
        raise ValueError("--inference-shards must be positive")

    candidates = pd.read_parquet(candidates_path)
    validate_candidate_table(candidates, str(candidates_path))
    require_source_columns(candidates.columns, str(candidates_path))
    require_valid_sources(
        candidates[list(SOURCE_COLUMNS)].to_dict("records"), str(candidates_path)
    )
    candidates = candidates.copy()
    candidates["patient_id"] = candidates["patient_id"].astype(str)
    candidates["exam_id"] = candidates["exam_id"].astype(str)
    candidates["view_id"] = candidates["sha256"].map(normalize_view_id)
    if candidates["view_id"].duplicated().any():
        raise ValueError("candidate table contains duplicate view identifiers")

    prior_patients, prior_exams, prior_views, prior_provenance = (
        load_prior_reference_universe(args.prior_reference_sources)
    )
    excluded = (
        candidates["patient_id"].isin(prior_patients)
        | candidates["exam_id"].isin(prior_exams)
        | candidates["view_id"].isin(prior_views)
    )
    mining = candidates.loc[~excluded].copy()
    if mining.empty:
        raise RuntimeError("patient/exam-disjoint mining population is empty")
    if (
        mining["patient_id"].isin(prior_patients).any()
        or mining["exam_id"].isin(prior_exams).any()
        or mining["view_id"].isin(prior_views).any()
    ):
        raise RuntimeError("prior-reference exclusion did not produce disjoint data")

    rendered_sources_path = rendered_campaign / "render_sources.parquet"
    rendered_sources = pd.read_parquet(rendered_sources_path)
    require_columns(
        rendered_sources,
        {
            "view_id",
            SOURCE_ARCHIVE_COLUMN,
            SOURCE_MEMBER_COLUMN,
            "sop_instance_uid",
            "sha256",
        },
        str(rendered_sources_path),
    )
    rendered_sources = rendered_sources.copy()
    rendered_sources["view_id"] = rendered_sources["view_id"].map(normalize_view_id)
    rendered_lookup = rendered_sources.set_index("view_id", verify_integrity=True)
    missing_render_sources = set(mining["view_id"]) - set(rendered_lookup.index)
    if missing_render_sources:
        raise RuntimeError(
            "rebuilt candidates are not a subset of the validated rendered campaign"
        )
    source_key = [SOURCE_ARCHIVE_COLUMN, SOURCE_MEMBER_COLUMN, "sop_instance_uid"]
    expected_sources = mining.set_index("view_id")[source_key].astype(str).sort_index()
    actual_sources = (
        rendered_lookup.loc[expected_sources.index, source_key].astype(str).sort_index()
    )
    if not expected_sources.equals(actual_sources):
        raise RuntimeError("rendered image lineage disagrees with rebuilt candidates")

    out_dir.mkdir(parents=True, mode=0o700)
    image_dir = out_dir / "images"
    image_dir.mkdir(mode=0o700)
    for view_id in mining["view_id"].sort_values():
        source = rendered_campaign / "images" / f"{view_id}.png"
        destination = image_dir / f"{view_id}.png"
        if not source.is_file():
            raise FileNotFoundError("validated rendered candidate image is missing")
        os.link(source, destination)

    mining = mining.sort_values("view_id", kind="stable").reset_index(drop=True)
    mining["image_path"] = mining["view_id"].map(
        lambda view_id: f"images/{view_id}.png"
    )
    mining["review_order"] = range(1, len(mining) + 1)
    mining["inference_shard"] = mining.index % args.inference_shards
    source_manifest = out_dir / "source_manifest.parquet"
    mining.to_parquet(source_manifest, index=False)
    os.chmod(source_manifest, 0o600)

    manifest_columns = [
        "view_id",
        "image_path",
        "laterality",
        "view",
        "review_order",
    ]
    manifest = mining[manifest_columns].copy()
    manifest_path = out_dir / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    os.chmod(manifest_path, 0o600)
    shard_counts: list[int] = []
    for shard_index in range(args.inference_shards):
        shard = mining.loc[
            mining["inference_shard"].eq(shard_index), manifest_columns
        ].copy()
        shard["review_order"] = range(1, len(shard) + 1)
        shard_path = out_dir / f"manifest_shard_{shard_index:03d}.parquet"
        shard.to_parquet(shard_path, index=False)
        os.chmod(shard_path, 0o600)
        shard_counts.append(int(len(shard)))

    model_spec = load_model_spec(registry_path, args.model_key)
    campaign = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "gross-device prompt-only edge-crop challenge mining",
        "candidate_table": str(candidates_path),
        "candidate_table_sha256": sha256_file(candidates_path),
        "source_rows": int(len(candidates)),
        "excluded_prior_reference_rows": int(excluded.sum()),
        "mining_rows": int(len(mining)),
        "mining_patients": int(mining["patient_id"].nunique()),
        "mining_exams": int(mining["exam_id"].nunique()),
        "selected_rank_one_rows": int(mining["is_selected"].astype(bool).sum()),
        "inference_shards": int(args.inference_shards),
        "inference_shard_rows": shard_counts,
        "manifest_sha256": sha256_file(manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest),
        "rendered_campaign": str(rendered_campaign),
        "rendered_campaign_sha256": sha256_file(rendered_campaign / "campaign.json"),
        "rendered_source_sha256": sha256_file(rendered_sources_path),
        "prior_reference_sources": prior_provenance,
        "disjointness": {
            "prior_patients": len(prior_patients),
            "prior_exams": len(prior_exams),
            "prior_views": len(prior_views),
            "overlapping_patients": 0,
            "overlapping_exams": 0,
            "overlapping_views": 0,
        },
    }
    write_json(out_dir / "campaign.json", campaign)

    script_path = Path(__file__).resolve()
    spec = {
        "schema_version": 1,
        "status": "frozen_before_inference_and_human_reference_review",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "decision": (
            "whether one zero-shot prompt clarification recovers partially cropped "
            "gross devices without materially increasing gross-device hard-negative "
            "false positives"
        ),
        "target": TARGET,
        "nearest_baseline": {
            "name": "visible_gross_implanted_or_procedural_device_v1",
            "prompt": str(baseline_prompt),
            "prompt_sha256": sha256_file(baseline_prompt),
            "threshold": "high confidence",
            "disposition": (
                "immutable comparator; the prompt-only model family remains active "
                "until a prospective candidate beats it"
            ),
        },
        "candidate": {
            "name": "visible_gross_implanted_or_procedural_device_v2_edge_crops",
            "prompt": str(candidate_prompt),
            "prompt_sha256": sha256_file(candidate_prompt),
            "threshold": "high confidence",
            "single_changed_delta": (
                "explicitly inspect every border and count a recognizable partially "
                "cropped device housing while rejecting unstructured bright corners, "
                "markers, detector or paddle edges, and anatomy"
            ),
        },
        "unchanged_inference": {
            "model_key": args.model_key,
            "model_registry": str(registry_path),
            "model_registry_sha256": sha256_file(registry_path),
            "model_spec": model_spec,
            "temperature": 0,
            "thinking_disabled": True,
            "max_new_tokens": 128,
            "few_shot_examples": 0,
            "embeddings": False,
        },
        "inputs": campaign,
        "minimal_matrix": [
            "A: unchanged v1 zero-shot prompt at high confidence",
            "B: v2 edge-crop zero-shot prompt at high confidence",
        ],
        "blinding": (
            "models score the full disjoint mining population before human review; "
            "the final gallery exposes neither arm, prediction, rationale, nor stratum"
        ),
        "panel_selection": {
            "seed": int(args.seed),
            "target_rows": int(sum(DEFAULT_QUOTAS.values())),
            "primary_quotas": DEFAULT_QUOTAS,
            "one_view_per_patient_and_exam": True,
            "edge_enrichment": (
                "candidate-only high-confidence rationale contains a frozen edge "
                "lexicon: border, corner, crop, cropped, edge, partial, partly, superior"
            ),
            "shortfall_rule": (
                "after taking each registered quota in order, fill any remaining "
                "positions from unused views in the same fixed stratum order; retain "
                "the view's actual stratum and never use human labels for selection"
            ),
        },
        "readouts": [
            "paired high-confidence sensitivity and specificity against the same labels",
            "candidate-only edge-enriched positive yield",
            "false positives by registered hard-negative stratum",
        ],
        "success_rule": (
            "after all low-confidence labels are adjudicated, candidate sensitivity "
            "must be at least 0.95 and not below baseline; candidate specificity must "
            "be at least 0.90 and may add at most one false positive versus baseline; "
            "at least three human-positive candidate-only edge-enriched views and ten "
            "human-positive views overall are required for an adequate mechanism check"
        ),
        "insufficient_reference_rule": (
            "if positive-yield requirements are not met, expand once from still-disjoint "
            "unused mining views using these frozen prompts and selection rules"
        ),
        "stop_rule": (
            "score this panel once; do not tune either prompt, threshold, examples, or "
            "selection after opening its labels"
        ),
        "if_positive": (
            "carry the candidate unchanged into one new patient/exam-disjoint "
            "whole-system audit"
        ),
        "if_negative": (
            "retain v1 and falsify only this edge-crop wording; do not revive the "
            "external-marker veto or an embedding classifier"
        ),
        "source_code": {
            "builder": str(script_path),
            "builder_sha256": sha256_file(script_path),
        },
    }
    spec_path.parent.mkdir(parents=True, mode=0o700)
    write_json(spec_path, spec)
    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Gross-device edge-crop challenge mining population",
                "",
                "This restricted campaign contains every current deterministic-eligible",
                "view candidate after patient-, exam-, and view-level exclusion of all",
                "prior reference sources. PNGs are hard links to the previously rendered",
                "and source-verified current-policy candidate campaign.",
                "",
                f"- candidate rows before reference exclusion: `{len(candidates)}`",
                f"- mining rows: `{len(mining)}`",
                f"- mining patients/exams: `{mining['patient_id'].nunique()}` / "
                f"`{mining['exam_id'].nunique()}`",
                f"- inference shards per prompt arm: `{args.inference_shards}`",
                f"- manifest SHA-256: `{campaign['manifest_sha256']}`",
                f"- frozen specification: `{spec_path}`",
                "",
                "The source manifest contains restricted identifiers and must not be",
                "committed or printed. Model outputs are candidate-mining signals, not",
                "reference labels.",
                "",
            ]
        )
    )
    os.chmod(readme, 0o600)
    print(
        "gross-device mining population ready: "
        f"rows={len(mining)} exams={mining['exam_id'].nunique()} "
        f"shards={args.inference_shards}"
    )
    return 0


def suggestion_details(record: dict[str, Any]) -> tuple[bool, str, str]:
    suggestions = record["suggestions"]
    if not suggestions:
        return False, "", ""
    suggestion = suggestions[0]
    return (
        True,
        str(suggestion.get("confidence", "")).strip().lower(),
        str(suggestion.get("rationale", "")).strip(),
    )


def subset_run(
    run: dict[str, Any], view_ids: set[str], output_path: Path
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite model subset: {output_path}")
    return save_view_auto_run(
        output_path,
        {
            **run,
            "view_suggestions": {
                view_id: run["view_suggestions"][view_id]
                for view_id in sorted(view_ids)
            },
        },
    )


def assemble(args: argparse.Namespace) -> int:
    mining_dir = args.mining_dir.resolve()
    source_path = mining_dir / "source_manifest.parquet"
    manifest_path = mining_dir / "manifest.parquet"
    spec_path = args.spec.resolve()
    baseline_path = args.baseline_run.resolve()
    candidate_path = args.candidate_run.resolve()
    out_dir = args.out_dir.resolve()
    model_subset_dir = args.model_subset_dir.resolve()
    for path in (
        source_path,
        manifest_path,
        mining_dir / "campaign.json",
        spec_path,
        baseline_path,
        candidate_path,
        args.human_rubric.resolve(),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"required challenge assembly input missing: {path}"
            )
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite challenge panel: {out_dir}")
    if model_subset_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite challenge model subsets: {model_subset_dir}"
        )

    spec = json.loads(spec_path.read_text())
    if spec.get("status") != "frozen_before_inference_and_human_reference_review":
        raise ValueError("challenge specification is not frozen")
    manifest = pd.read_parquet(manifest_path)
    source = pd.read_parquet(source_path)
    require_columns(
        source,
        {"view_id", "patient_id", "exam_id", "image_path", "laterality", "view"},
        str(source_path),
    )
    source = source.copy()
    source["view_id"] = source["view_id"].map(normalize_view_id)
    source["patient_id"] = source["patient_id"].astype(str)
    source["exam_id"] = source["exam_id"].astype(str)
    expected_ids = set(manifest["view_id"].map(normalize_view_id))
    if set(source["view_id"]) != expected_ids:
        raise RuntimeError("mining source and inference manifests disagree")

    baseline = load_view_auto_run(baseline_path)
    candidate = load_view_auto_run(candidate_path)
    for name, run in (("baseline", baseline), ("candidate", candidate)):
        if run.get("target") != TARGET:
            raise RuntimeError(f"{name} run target disagrees with challenge")
        if set(run["view_suggestions"]) != expected_ids:
            raise RuntimeError(f"{name} run does not exactly cover mining manifest")
    if (
        baseline["inference_settings"].get("target_prompt_sha256")
        != spec["nearest_baseline"]["prompt_sha256"]
    ):
        raise RuntimeError("baseline run prompt differs from frozen specification")
    if (
        candidate["inference_settings"].get("target_prompt_sha256")
        != spec["candidate"]["prompt_sha256"]
    ):
        raise RuntimeError("candidate run prompt differs from frozen specification")
    unchanged_fields = ("model_key", "model_revision", "temperature", "max_new_tokens")
    if any(
        baseline["inference_settings"].get(field)
        != candidate["inference_settings"].get(field)
        for field in unchanged_fields
    ):
        raise RuntimeError("prompt arms differ in a frozen inference setting")

    classifications: list[dict[str, Any]] = []
    for view_id in sorted(expected_ids):
        baseline_record = baseline["view_suggestions"][view_id]
        candidate_record = candidate["view_suggestions"][view_id]
        baseline_high = view_suggestion_is_target_present(
            baseline_record, target=TARGET, minimum_confidence="high"
        )
        candidate_high = view_suggestion_is_target_present(
            candidate_record, target=TARGET, minimum_confidence="high"
        )
        baseline_any, baseline_confidence, _baseline_rationale = suggestion_details(
            baseline_record
        )
        candidate_any, candidate_confidence, candidate_rationale = suggestion_details(
            candidate_record
        )
        if (
            candidate_high
            and not baseline_high
            and EDGE_WORDS.search(candidate_rationale)
        ):
            stratum = "candidate_only_edge_high"
        elif candidate_high and not baseline_high:
            stratum = "candidate_only_other_high"
        elif candidate_high and baseline_high:
            stratum = "both_high"
        elif baseline_high and not candidate_high:
            stratum = "baseline_only_high"
        elif baseline_any or candidate_any:
            stratum = "either_lower_confidence"
        else:
            stratum = "both_negative_random"
        classifications.append(
            {
                "view_id": view_id,
                "stratum": stratum,
                "baseline_high": baseline_high,
                "candidate_high": candidate_high,
                "baseline_confidence": baseline_confidence,
                "candidate_confidence": candidate_confidence,
            }
        )
    classified = source.merge(
        pd.DataFrame(classifications),
        on="view_id",
        how="inner",
        validate="one_to_one",
    )

    selected_indices: list[int] = []
    used_patients: set[str] = set()
    used_exams: set[str] = set()

    def take_rows(frame: pd.DataFrame, limit: int, namespace: str) -> int:
        taken = 0
        ranked = frame.assign(
            _rank=frame["view_id"].map(
                lambda value: stable_rank(args.seed, namespace, value)
            )
        ).sort_values("_rank", kind="stable")
        for index, row in ranked.iterrows():
            if index in selected_indices:
                continue
            patient_id = str(row["patient_id"])
            exam_id = str(row["exam_id"])
            if patient_id in used_patients or exam_id in used_exams:
                continue
            selected_indices.append(index)
            used_patients.add(patient_id)
            used_exams.add(exam_id)
            taken += 1
            if taken == limit:
                break
        return taken

    primary_counts: dict[str, int] = {}
    for stratum in STRATUM_ORDER:
        primary_counts[stratum] = take_rows(
            classified[classified["stratum"].eq(stratum)],
            DEFAULT_QUOTAS[stratum],
            f"primary:{stratum}",
        )
    target_rows = sum(DEFAULT_QUOTAS.values())
    remaining = target_rows - len(selected_indices)
    fill_counts = {stratum: 0 for stratum in STRATUM_ORDER}
    if remaining > 0:
        for stratum in STRATUM_ORDER:
            if remaining == 0:
                break
            added = take_rows(
                classified[classified["stratum"].eq(stratum)],
                remaining,
                f"fill:{stratum}",
            )
            fill_counts[stratum] += added
            remaining -= added
    if remaining:
        raise RuntimeError(
            "could not build the registered patient/exam-disjoint panel size"
        )

    selected = classified.loc[selected_indices].copy()
    selected["_review_rank"] = selected["view_id"].map(
        lambda value: stable_rank(args.seed, "review", value)
    )
    selected = selected.sort_values("_review_rank", kind="stable").reset_index(
        drop=True
    )
    selected["review_order"] = range(1, len(selected) + 1)
    selected["image_path"] = selected["view_id"].map(
        lambda view_id: f"images/{view_id}.png"
    )
    if (
        selected["patient_id"].duplicated().any()
        or selected["exam_id"].duplicated().any()
    ):
        raise RuntimeError("assembled challenge is not patient/exam disjoint")

    out_dir.mkdir(parents=True, mode=0o700)
    image_dir = out_dir / "images"
    image_dir.mkdir(mode=0o700)
    for view_id in selected["view_id"]:
        source_image = mining_dir / "images" / f"{view_id}.png"
        destination = image_dir / f"{view_id}.png"
        shutil.copy2(source_image, destination)
        os.chmod(destination, 0o600)

    browser_columns = [
        "view_id",
        "image_path",
        "laterality",
        "view",
        "review_order",
        "stratum",
    ]
    panel_manifest = out_dir / "manifest.parquet"
    selected[browser_columns].to_parquet(panel_manifest, index=False)
    os.chmod(panel_manifest, 0o600)
    panel_sources = out_dir / "source_manifest.parquet"
    selected.drop(columns=["_review_rank"]).to_parquet(panel_sources, index=False)
    os.chmod(panel_sources, 0o600)

    state_path = out_dir / "view_qc_state.json"
    state = save_view_qc_state(state_path, empty_view_qc_state(TARGET))
    initialize_view_qc_event_log(default_view_qc_events_path(state_path), state)
    shutil.copy2(args.human_rubric.resolve(), out_dir / "human_rubric.txt")
    os.chmod(out_dir / "human_rubric.txt", 0o600)

    model_subset_dir.mkdir(parents=True, mode=0o700)
    selected_ids = set(selected["view_id"])
    baseline_subset_path = model_subset_dir / "baseline_run.json"
    candidate_subset_path = model_subset_dir / "candidate_run.json"
    subset_run(baseline, selected_ids, baseline_subset_path)
    subset_run(candidate, selected_ids, candidate_subset_path)

    actual_counts = (
        selected["stratum"].value_counts().reindex(STRATUM_ORDER, fill_value=0)
    )
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "specification": str(spec_path),
        "specification_sha256": sha256_file(spec_path),
        "mining_manifest": str(manifest_path),
        "mining_manifest_sha256": sha256_file(manifest_path),
        "baseline_run": str(baseline_path),
        "baseline_run_sha256": sha256_file(baseline_path),
        "candidate_run": str(candidate_path),
        "candidate_run_sha256": sha256_file(candidate_path),
        "panel_rows": int(len(selected)),
        "panel_patients": int(selected["patient_id"].nunique()),
        "panel_exams": int(selected["exam_id"].nunique()),
        "registered_primary_counts": primary_counts,
        "registered_fill_counts": fill_counts,
        "actual_stratum_counts": {
            str(key): int(value) for key, value in actual_counts.items()
        },
        "manifest_sha256": sha256_file(panel_manifest),
        "source_manifest_sha256": sha256_file(panel_sources),
        "baseline_subset_sha256": sha256_file(baseline_subset_path),
        "candidate_subset_sha256": sha256_file(candidate_subset_path),
        "model_outputs_exposed_to_reviewer": False,
    }
    write_json(out_dir / "sampling_metadata.json", metadata)
    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Gross-device edge-crop prompt challenge",
                "",
                f"- target: `{TARGET}`",
                f"- views/patients/exams: `{len(selected)}` / "
                f"`{selected['patient_id'].nunique()}` / "
                f"`{selected['exam_id'].nunique()}`",
                f"- manifest SHA-256: `{metadata['manifest_sha256']}`",
                f"- frozen specification: `{spec_path}`",
                "- every patient, exam, and view is disjoint from prior references",
                "- one view per patient and exam",
                "- model arm, prediction, rationale, and sampling stratum are hidden",
                "- Low confidence is independent of the binary target decision",
                "",
                "Server command:",
                "",
                "```bash",
                "/gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima/bin/python \\",
                "  qc/view_qc_gallery.py \\",
                f"  --manifest {panel_manifest} \\",
                f"  --state {state_path} \\",
                "  --reviewer annawoodard \\",
                "  --host 127.0.0.1 \\",
                "  --port 8767 \\",
                "  --negative-label 'No gross device' \\",
                "  --positive-label 'Gross device present' \\",
                "  --negative-shortcut n \\",
                "  --positive-shortcut y \\",
                "  --review-instruction 'Judge only whether the target gross device is visibly present; ignore every other finding.' \\",
                f"  --review-rubric-file {out_dir / 'human_rubric.txt'}",
                "```",
                "",
            ]
        )
    )
    os.chmod(readme, 0o600)
    print(
        "gross-device challenge ready: "
        f"rows={len(selected)} strata="
        + ",".join(f"{key}={int(actual_counts[key])}" for key in STRATUM_ORDER)
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--candidates", type=Path, required=True)
    prepare_parser.add_argument("--rendered-campaign", type=Path, required=True)
    prepare_parser.add_argument(
        "--prior-reference-sources",
        type=Path,
        action="append",
        required=True,
    )
    prepare_parser.add_argument("--out-dir", type=Path, required=True)
    prepare_parser.add_argument("--spec-output", type=Path, required=True)
    prepare_parser.add_argument("--baseline-prompt", type=Path, required=True)
    prepare_parser.add_argument("--candidate-prompt", type=Path, required=True)
    prepare_parser.add_argument(
        "--model-registry",
        type=Path,
        default=Path("qc/auto_qc_models.json"),
    )
    prepare_parser.add_argument("--model-key", default="qwen35_27b_fp8")
    prepare_parser.add_argument("--inference-shards", type=int, default=4)
    prepare_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare_parser.set_defaults(handler=prepare)

    assemble_parser = subparsers.add_parser("assemble")
    assemble_parser.add_argument("--mining-dir", type=Path, required=True)
    assemble_parser.add_argument("--spec", type=Path, required=True)
    assemble_parser.add_argument("--baseline-run", type=Path, required=True)
    assemble_parser.add_argument("--candidate-run", type=Path, required=True)
    assemble_parser.add_argument("--out-dir", type=Path, required=True)
    assemble_parser.add_argument("--model-subset-dir", type=Path, required=True)
    assemble_parser.add_argument("--human-rubric", type=Path, required=True)
    assemble_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    assemble_parser.set_defaults(handler=assemble)
    return parser


def main() -> int:
    os.umask(0o077)
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
