#!/usr/bin/env python3
"""Build a fresh patient-disjoint whole-exam audit of Mirai view fallback."""

from __future__ import annotations

import argparse
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
import pydicom

from prima.dicom_source import (
    SOURCE_ARCHIVE_COLUMN,
    SOURCE_COLUMNS,
    DicomSource,
    materialize_dicom_sources,
    require_source_columns,
    require_valid_sources,
    validate_materialized_source,
)
from prima.view_auto_qc import load_view_auto_run, view_suggestion_is_target_present
from prima.view_fallback import validate_candidate_table
from prima.view_qc import (
    default_view_qc_events_path,
    empty_view_qc_state,
    initialize_view_qc_event_log,
    normalize_view_id,
    normalize_view_qc_target,
    save_view_qc_state,
    validate_rendered_view_png,
)
from prima.view_selection import (
    nonstandard_mirai_view_reasons,
    view_modifier_code_meanings,
)

DEFAULT_TARGET = (
    "view requiring exclusion from standard Mirai input under visual rubric v2"
)
REQUIRED_SEAM_DECISION_COLUMNS = {
    "exam_id",
    "laterality",
    "view",
    "original_view_id",
    "fallback_status",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--render-campaign", type=Path, required=True)
    parser.add_argument("--seam-decisions", type=Path, required=True)
    parser.add_argument("--film-run", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-source-manifest", type=Path, action="append", default=[]
    )
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--seam-fallback-exams", type=int, default=8)
    parser.add_argument("--film-exclusion-exams", type=int, default=2)
    parser.add_argument("--implant-exclusion-exams", type=int, default=8)
    parser.add_argument("--multi-candidate-control-exams", type=int, default=7)
    parser.add_argument("--single-candidate-control-exams", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--max-render-pixels", type=int, default=2_000_000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_group_id(exam_id: object, laterality: object, view: object) -> str:
    payload = f"{exam_id}|{laterality}|{view}".encode()
    return hashlib.sha256(payload).hexdigest()


def audit_exam_id(patient_id: object, exam_id: object) -> str:
    payload = f"{patient_id}|{exam_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def load_prior_exclusions(paths: list[Path]) -> tuple[set[str], set[str], set[str]]:
    """Return patient, exam, and exact-view exclusions from prior source panels."""
    patients: set[str] = set()
    exams: set[str] = set()
    views: set[str] = set()
    for raw_path in paths:
        path = raw_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"prior source manifest not found: {path}")
        table = pd.read_parquet(path)
        required = {"patient_id", "exam_id", "view_id"}
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(
                f"prior source manifest is missing columns: {path}: "
                + ", ".join(missing)
            )
        if table[list(required)].isna().any().any():
            raise ValueError(f"prior source manifest contains null identifiers: {path}")
        patients.update(table["patient_id"].astype(str))
        exams.update(table["exam_id"].astype(str))
        views.update(table["view_id"].map(normalize_view_id))
    return patients, exams, views


def _sample_unique_patient_exams(
    pool: pd.DataFrame,
    *,
    count: int,
    stratum: str,
    used_patients: set[str],
    used_exams: set[str],
    seed: int,
) -> pd.DataFrame:
    if count < 0:
        raise ValueError("audit stratum counts cannot be negative")
    if count == 0:
        return pool.iloc[0:0].assign(sampling_stratum=pd.Series(dtype="string"))
    shuffled = pool.sort_values("exam_id", kind="stable").sample(
        frac=1.0, random_state=seed
    )
    rows: list[dict[str, object]] = []
    for row in shuffled.to_dict("records"):
        patient_id = str(row["patient_id"])
        exam_id = str(row["exam_id"])
        if patient_id in used_patients or exam_id in used_exams:
            continue
        used_patients.add(patient_id)
        used_exams.add(exam_id)
        row["sampling_stratum"] = stratum
        rows.append(row)
        if len(rows) == count:
            break
    if len(rows) != count:
        raise ValueError(
            f"only {len(rows)} patient-disjoint exams are available for {stratum}; "
            f"requested {count}"
        )
    return pd.DataFrame(rows)


def sample_audit_exams(
    exam_summary: pd.DataFrame,
    *,
    counts: dict[str, int],
    seed: int,
) -> pd.DataFrame:
    """Sample mutually exclusive challenge and control strata by patient."""
    predicates = {
        "seam_fallback_challenge": exam_summary["seam_fallback_challenge"],
        "film_exam_exclusion": exam_summary["rank1_film_positive"],
        "implant_exam_exclusion": exam_summary["rank1_has_implant"],
        "multi_candidate_control": (
            (exam_summary["candidate_count"] > 4)
            & ~exam_summary["seam_fallback_challenge"]
            & ~exam_summary["rank1_film_positive"]
            & ~exam_summary["rank1_has_implant"]
            & exam_summary["all_seam_original_absent"]
        ),
        "single_candidate_control": (
            (exam_summary["candidate_count"] == 4)
            & ~exam_summary["rank1_film_positive"]
            & ~exam_summary["rank1_has_implant"]
            & exam_summary["all_seam_original_absent"]
        ),
    }
    if set(counts) != set(predicates):
        raise ValueError("audit stratum counts do not match the registered strata")
    used_patients: set[str] = set()
    used_exams: set[str] = set()
    sampled: list[pd.DataFrame] = []
    for offset, stratum in enumerate(predicates):
        pool = exam_summary[predicates[stratum]].copy()
        sampled.append(
            _sample_unique_patient_exams(
                pool,
                count=counts[stratum],
                stratum=stratum,
                used_patients=used_patients,
                used_exams=used_exams,
                seed=seed + offset,
            )
        )
    result = pd.concat(sampled, ignore_index=True)
    if result["patient_id"].duplicated().any() or result["exam_id"].duplicated().any():
        raise RuntimeError("whole-exam audit sampling violated patient disjointness")
    return result


def build_exam_summary(
    candidates: pd.DataFrame,
    seam_decisions: pd.DataFrame,
    film_positive_ids: set[str],
) -> pd.DataFrame:
    """Build one row per eligible four-slot exam for registered sampling."""
    work = candidates.copy()
    work["slot"] = work["laterality"].astype(str) + "_" + work["view"].astype(str)
    slot_counts = work.groupby("exam_id", sort=False)["slot"].nunique()
    work = work[work["exam_id"].isin(slot_counts[slot_counts == 4].index)].copy()
    rank1 = work[work["selection_rank"].astype(int) == 1].copy()
    rank1["film_positive"] = rank1["view_id"].isin(film_positive_ids)

    seam = seam_decisions.copy()
    seam["exam_id"] = seam["exam_id"].astype(str)
    seam_challenge = set(
        seam.loc[seam["fallback_status"] == "alternate_target_absent", "exam_id"]
    )
    seam_clean = (
        seam.groupby("exam_id", sort=False)["fallback_status"]
        .apply(lambda values: bool((values == "original_target_absent").all()))
        .to_dict()
    )

    grouped = work.groupby("exam_id", sort=False)
    summary = grouped.agg(
        patient_id=("patient_id", "first"),
        candidate_count=("view_id", "size"),
        slot_count=("slot", "nunique"),
    ).reset_index()
    if grouped["patient_id"].nunique().ne(1).any():
        raise ValueError("one exam maps to multiple patients")
    summary["seam_fallback_challenge"] = (
        summary["exam_id"].astype(str).isin(seam_challenge)
    )
    summary["all_seam_original_absent"] = (
        summary["exam_id"]
        .astype(str)
        .map(lambda value: bool(seam_clean.get(value, False)))
    )
    summary = summary.merge(
        rank1.groupby("exam_id", sort=False).agg(
            rank1_film_positive=("film_positive", "any"),
            rank1_has_implant=("has_implant", "any"),
        ),
        on="exam_id",
        how="left",
        validate="one_to_one",
    )
    if not summary["slot_count"].eq(4).all():
        raise RuntimeError("eligible exam summary contains a non-quad exam")
    return summary


def audit_selected_sources(
    selected: pd.DataFrame, *, raw_root: Path, temp_root: Path
) -> pd.DataFrame:
    """Read each selected source header, extracting no patient identifiers."""
    rows: list[dict[str, object]] = []
    for _archive, group in selected.groupby(SOURCE_ARCHIVE_COLUMN, sort=True):
        records = group.to_dict("records")
        sources = [DicomSource.from_row(record) for record in records]
        with materialize_dicom_sources(sources, raw_root, temp_root=temp_root) as paths:
            for record, source in zip(records, sources):
                path = paths[source.archive_member.as_posix()]
                dataset = pydicom.dcmread(
                    str(path), force=True, stop_before_pixels=True
                )
                validate_materialized_source(source, path, dataset)
                reasons = nonstandard_mirai_view_reasons(dataset)
                rows.append(
                    {
                        "view_id": normalize_view_id(record["sha256"]),
                        "is_standard_mirai_view": not reasons,
                        "view_position": str(
                            dataset.get("ViewPosition", "") or ""
                        ).strip(),
                        "view_modifiers": " | ".join(
                            view_modifier_code_meanings(dataset)
                        ),
                        "partial_view": str(
                            dataset.get("PartialView", "") or ""
                        ).strip(),
                        "exclusion_reasons": " | ".join(reasons),
                    }
                )
    audit = pd.DataFrame(rows).sort_values("view_id", kind="stable")
    if len(audit) != len(selected) or audit["view_id"].duplicated().any():
        raise RuntimeError("DICOM eligibility audit does not exactly cover the panel")
    return audit


def _restricted_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_from_args(args: argparse.Namespace) -> pd.DataFrame:
    candidates_path = args.candidates.resolve()
    render_campaign = args.render_campaign.resolve()
    seam_path = args.seam_decisions.resolve()
    film_path = args.film_run.resolve()
    raw_root = args.raw_root.resolve()
    temp_root = args.temp_root.resolve()
    out_dir = args.out_dir.resolve()
    target = normalize_view_qc_target(args.target)
    for path in (
        candidates_path,
        seam_path,
        film_path,
        render_campaign / "campaign.json",
        render_campaign / "render_complete.json",
        render_campaign / "manifest.parquet",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"whole-exam audit input not found: {path}")
    for path in (raw_root, temp_root, render_campaign / "images"):
        if not path.is_dir():
            raise FileNotFoundError(f"whole-exam audit directory not found: {path}")
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite whole-exam audit: {out_dir}")
    if args.max_render_pixels <= 0:
        raise ValueError("--max-render-pixels must be positive")

    counts = {
        "seam_fallback_challenge": int(args.seam_fallback_exams),
        "film_exam_exclusion": int(args.film_exclusion_exams),
        "implant_exam_exclusion": int(args.implant_exclusion_exams),
        "multi_candidate_control": int(args.multi_candidate_control_exams),
        "single_candidate_control": int(args.single_candidate_control_exams),
    }
    if any(value < 0 for value in counts.values()) or not sum(counts.values()):
        raise ValueError("audit requires nonnegative counts and at least one exam")

    candidates = pd.read_parquet(candidates_path)
    validate_candidate_table(candidates, str(candidates_path))
    require_source_columns(candidates.columns, str(candidates_path))
    require_valid_sources(
        candidates[list(SOURCE_COLUMNS)].to_dict("records"), str(candidates_path)
    )
    required = {"patient_id", "has_implant"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError("candidate table is missing columns: " + ", ".join(missing))
    candidates = candidates.copy()
    candidates["view_id"] = candidates["sha256"].map(normalize_view_id)
    if candidates["view_id"].duplicated().any():
        raise ValueError("candidate table contains duplicate view IDs")

    campaign = json.loads((render_campaign / "campaign.json").read_text())
    if campaign.get("candidate_table_sha256") != sha256_file(candidates_path):
        raise RuntimeError("render campaign and candidate table hashes disagree")
    rendered_manifest = pd.read_parquet(render_campaign / "manifest.parquet")
    rendered_ids = set(rendered_manifest["view_id"].map(normalize_view_id))
    render_complete = json.loads((render_campaign / "render_complete.json").read_text())
    failed_ids = {
        normalize_view_id(value) for value in render_complete.get("failed_view_ids", [])
    }
    if rendered_ids != set(candidates["view_id"]):
        raise RuntimeError("render campaign does not cover the candidate table")

    prior_patients, prior_exams, prior_views = load_prior_exclusions(
        args.exclude_source_manifest
    )
    candidates = candidates[
        ~candidates["patient_id"].astype(str).isin(prior_patients)
        & ~candidates["exam_id"].astype(str).isin(prior_exams)
        & ~candidates["view_id"].isin(prior_views)
    ].copy()
    failed_exams = set(
        candidates.loc[candidates["view_id"].isin(failed_ids), "exam_id"]
    )
    candidates = candidates[~candidates["exam_id"].isin(failed_exams)].copy()

    seam = pd.read_parquet(seam_path)
    seam_missing = sorted(REQUIRED_SEAM_DECISION_COLUMNS - set(seam.columns))
    if seam_missing:
        raise ValueError(
            "seam decisions are missing columns: " + ", ".join(seam_missing)
        )
    seam = seam.copy()
    seam["exam_id"] = seam["exam_id"].astype(str)
    seam["original_view_id"] = seam["original_view_id"].map(normalize_view_id)

    film_run = load_view_auto_run(film_path)
    film_positive = {
        view_id
        for view_id, record in film_run["view_suggestions"].items()
        if view_suggestion_is_target_present(
            record,
            target=film_run["target"],
            minimum_confidence="high",
        )
    }
    if not set(candidates["view_id"]).issubset(
        set(film_run["view_suggestions"]) | failed_ids
    ):
        raise RuntimeError("film sampling run does not cover eligible candidates")

    summary = build_exam_summary(candidates, seam, film_positive)
    sampled = sample_audit_exams(summary, counts=counts, seed=int(args.seed))
    stratum_by_exam = sampled.set_index("exam_id")["sampling_stratum"].to_dict()
    selected = candidates[candidates["exam_id"].isin(sampled["exam_id"])].copy()
    selected["sampling_stratum"] = selected["exam_id"].map(stratum_by_exam)
    selected["audit_exam_id"] = [
        audit_exam_id(patient_id, exam_id)
        for patient_id, exam_id in selected[["patient_id", "exam_id"]].itertuples(
            index=False, name=None
        )
    ]
    selected["audit_group_id"] = [
        audit_group_id(exam_id, laterality, view)
        for exam_id, laterality, view in selected[
            ["exam_id", "laterality", "view"]
        ].itertuples(index=False, name=None)
    ]
    selected["candidate_count"] = selected.groupby(
        ["exam_id", "laterality", "view"], sort=False
    )["view_id"].transform("size")
    selected = selected.sort_values(
        ["audit_exam_id", "laterality", "view", "selection_rank"], kind="stable"
    ).reset_index(drop=True)

    out_dir.mkdir(parents=True, mode=0o700)
    image_dir = out_dir / "images"
    image_dir.mkdir(mode=0o700)
    for view_id in selected["view_id"]:
        source = render_campaign / "images" / f"{view_id}.png"
        validate_rendered_view_png(source, max_pixels=args.max_render_pixels)
        destination = image_dir / source.name
        shutil.copyfile(source, destination)
        os.chmod(destination, 0o600)

    eligibility = audit_selected_sources(
        selected, raw_root=raw_root, temp_root=temp_root
    )
    records = selected.to_dict("records")
    order = pd.Series(range(len(records))).sample(
        frac=1.0, random_state=int(args.seed) + 100
    )
    shuffled = [records[index] for index in order]
    for review_order, record in enumerate(shuffled, start=1):
        record["review_order"] = review_order
        record["image_path"] = f"images/{record['view_id']}.png"
    review = pd.DataFrame(shuffled)

    manifest = review[
        ["view_id", "image_path", "laterality", "view", "review_order"]
    ].copy()
    group_manifest = review[
        [
            "view_id",
            "audit_exam_id",
            "audit_group_id",
            "laterality",
            "view",
            "selection_rank",
            "candidate_count",
            "is_selected",
            "sampling_stratum",
            "review_order",
        ]
    ].copy()
    source_manifest = review.copy()
    for path, table in (
        (out_dir / "manifest.parquet", manifest),
        (out_dir / "group_manifest.parquet", group_manifest),
        (out_dir / "source_manifest.parquet", source_manifest),
        (out_dir / "eligibility_audit.parquet", eligibility),
    ):
        table.to_parquet(path, index=False)
        os.chmod(path, 0o600)

    state_path = out_dir / "view_qc_state.json"
    state = save_view_qc_state(state_path, empty_view_qc_state(target))
    initialize_view_qc_event_log(default_view_qc_events_path(state_path), state)

    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "target": target,
        "seed": int(args.seed),
        "candidate_table": str(candidates_path),
        "candidate_table_sha256": sha256_file(candidates_path),
        "render_campaign": str(render_campaign),
        "render_campaign_sha256": sha256_file(render_campaign / "campaign.json"),
        "seam_decisions": str(seam_path),
        "seam_decisions_sha256": sha256_file(seam_path),
        "film_run": str(film_path),
        "film_run_sha256": sha256_file(film_path),
        "prior_source_manifests": [
            str(path.resolve()) for path in args.exclude_source_manifest
        ],
        "prior_patients_excluded": len(prior_patients),
        "prior_exams_excluded": len(prior_exams),
        "prior_views_excluded": len(prior_views),
        "render_failure_exams_excluded": len(failed_exams),
        "registered_exam_counts": counts,
        "observed_exam_counts": {
            str(key): int(value)
            for key, value in sampled["sampling_stratum"]
            .value_counts()
            .sort_index()
            .items()
        },
        "exams": int(sampled["exam_id"].nunique()),
        "patients": int(sampled["patient_id"].nunique()),
        "exact_slots": int(group_manifest["audit_group_id"].nunique()),
        "candidate_views": int(len(manifest)),
        "nonstandard_dicom_candidates": int(
            (~eligibility["is_standard_mirai_view"].astype(bool)).sum()
        ),
        "max_render_pixels": int(args.max_render_pixels),
        "sampling_outputs_are_not_reference_labels": True,
    }
    _restricted_json(out_dir / "sampling_metadata.json", metadata)
    protocol = {
        "schema_version": 1,
        "status": "frozen_before_modular_inference_and_reference_review",
        "target": target,
        "decision": "deterministic DICOM exclusion OR any high-confidence frozen visual component",
        "reference": "deterministic DICOM exclusion OR completed residual visual human/agent label",
        "component_protocol": "qc_redo/auto_qc_development/mirai_input_modular_five_family/protocol.json",
        "component_protocol_sha256": sha256_file(
            Path(
                "qc_redo/auto_qc_development/mirai_input_modular_five_family/protocol.json"
            )
        ),
        "gate": {
            "minimum_view_sensitivity": 0.95,
            "minimum_view_specificity": 0.90,
            "maximum_unsafe_selected_slots": 0,
            "maximum_false_exhausted_slots": 1,
            "maximum_unsafe_accepted_exams": 0,
        },
        "stop_rule": "score this one audit once; do not tune prompts or threshold on it",
    }
    _restricted_json(out_dir / "protocol.json", protocol)

    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Fresh whole-exam Mirai-input fallback audit",
                "",
                f"- target: `{target}`",
                f"- patients/exams: `{metadata['patients']}` / `{metadata['exams']}`",
                f"- exact slots/candidate views: `{metadata['exact_slots']}` / `{metadata['candidate_views']}`",
                f"- registered exam strata: `{counts}`",
                f"- deterministic nonstandard candidates: `{metadata['nonstandard_dicom_candidates']}`",
                f"- seed: `{args.seed}`",
                "- every selected exam has all four L/R CC/MLO slots",
                "- every candidate sequence is complete and renderable",
                "- patients, exams, and views from prior source-linked panels are excluded",
                "- browser manifest contains no patient, exam, SOP, or source identifiers",
                "",
                "The seam and film runs are used only to enrich challenge strata. They",
                "are not reference labels. The registered system combines deterministic",
                "DICOM acquisition eligibility with the unchanged six-call modular visual",
                "arm. The original candidate rank is never allowed to cross laterality or",
                "projection. Score once after the residual visual labels are complete.",
                "",
            ]
        )
    )
    os.chmod(readme, 0o600)
    return manifest


def main() -> int:
    args = parse_args()
    manifest = run_from_args(args)
    print(f"whole-exam fallback audit ready: views={len(manifest)}")
    print(f"output: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
