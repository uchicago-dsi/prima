#!/usr/bin/env python3
"""Build a blinded binary visual-usability pilot from canonical QC archives."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from prima.dicom_source import (
    SOURCE_ARCHIVE_COLUMN,
    SOURCE_COLUMNS,
    SOURCE_MEMBER_COLUMN,
    DicomSource,
    DicomSourceError,
    require_source_columns,
    require_valid_sources,
)
from prima.view_qc import (
    default_view_qc_events_path,
    empty_view_qc_state,
    initialize_view_qc_event_log,
    normalize_view_id,
    normalize_view_qc_target,
    save_view_qc_state,
    validate_rendered_view_png,
    validate_view_manifest_columns,
)

DEFAULT_TARGET = (
    "view requiring exclusion from standard Mirai input under visual rubric v2"
)
DEFAULT_NEGATIVE_LABEL = "Use for Mirai"
DEFAULT_POSITIVE_LABEL = "Do not use"
DEFAULT_REVIEW_INSTRUCTION = (
    "Choose Use or Do not use for this exact image using the full rubric below. "
    "Low confidence is a separate flag and never replaces the binary decision."
)
ALLOWED_LATERALITIES = {"L", "R"}
ALLOWED_VIEWS = {"CC", "MLO"}

SEAM_CAMPAIGN = "vertical_detector_seam_reference"
FILM_CAMPAIGN = "digitized_hard_copy_film_reference"
IMPLANT_CAMPAIGN = "visible_breast_implant_reference"
DIAGNOSTIC_CAMPAIGN = "spot_compression_magnification_holdout_reference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-archive-root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--failure-count", type=int, default=12)
    parser.add_argument("--diagnostic-control-count", type=int, default=46)
    parser.add_argument("--standard-extra-count", type=int, default=14)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--max-render-pixels", type=int, default=2_000_000)
    return parser.parse_args()


def _load_reference_labels(path: Path) -> dict[str, str]:
    """Read labels from one immutable archive, including its declared legacy state."""
    if not path.is_file():
        raise FileNotFoundError(f"reference state not found: {path}")
    payload = json.loads(path.read_text())
    raw_labels = payload.get("labels")
    if not isinstance(raw_labels, dict):
        raise ValueError(f"reference state has invalid labels: {path}")
    labels: dict[str, str] = {}
    for raw_view_id, raw_record in raw_labels.items():
        view_id = normalize_view_id(raw_view_id)
        label = raw_record.get("label") if isinstance(raw_record, dict) else raw_record
        label = str(label).strip()
        if not label:
            raise ValueError("reference archive contains an empty label")
        labels[view_id] = label
    return labels


def _safe_reference_image(campaign_dir: Path, raw_value: object) -> Path:
    root = campaign_dir.resolve()
    relative = Path(str(raw_value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("reference image path must be safe and relative")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("reference image escapes its campaign archive") from error
    if not path.is_file():
        raise FileNotFoundError("reference campaign image is missing")
    return path


def _load_candidate_sources(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"view candidate table not found: {path}")
    candidates = pd.read_parquet(path)
    required = {
        "patient_id",
        "exam_id",
        "laterality",
        "view",
        "has_implant",
        "for_presentation",
        "is_marked_up",
        *SOURCE_COLUMNS,
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError("view candidate table is missing: " + ", ".join(missing))
    require_source_columns(candidates.columns, str(path))
    candidates = candidates.copy()
    candidates["view_id"] = candidates["sha256"].map(normalize_view_id)
    if candidates["view_id"].duplicated().any():
        raise ValueError("view candidate table contains duplicate SHA-256 identities")
    return candidates


def _source_table_for_campaign(
    campaign_dir: Path, candidates: pd.DataFrame
) -> pd.DataFrame:
    source_path = campaign_dir / "source_manifest.parquet"
    if source_path.is_file():
        source = pd.read_parquet(source_path)
        require_source_columns(source.columns, str(source_path))
        if "view_id" not in source.columns:
            source = source.copy()
            source["view_id"] = source["sha256"].map(normalize_view_id)
        else:
            source = source.copy()
            source["view_id"] = source["view_id"].map(normalize_view_id)
        if source["view_id"].duplicated().any():
            raise ValueError("reference source manifest contains duplicate views")
        return source
    return candidates


def load_reference_campaign(
    archive_root: Path,
    campaign: str,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Load one browser archive, its labels, and durable original-DICOM lineage."""
    campaign_dir = archive_root / campaign
    if not campaign_dir.is_dir():
        raise FileNotFoundError(f"reference campaign not found: {campaign_dir}")
    manifest_path = campaign_dir / "manifest.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"reference manifest not found: {manifest_path}")
    browser = pd.read_parquet(manifest_path)
    validate_view_manifest_columns(browser.columns, str(manifest_path))
    if "stratum" not in browser.columns:
        raise ValueError("reference manifest is missing its sampling stratum")
    browser = browser.copy()
    browser["view_id"] = browser["view_id"].map(normalize_view_id)
    if browser["view_id"].duplicated().any():
        raise ValueError("reference manifest contains duplicate views")
    browser["reference_image_path"] = [
        str(_safe_reference_image(campaign_dir, value))
        for value in browser["image_path"]
    ]
    browser = browser[
        [
            "view_id",
            "laterality",
            "view",
            "stratum",
            "reference_image_path",
        ]
    ].rename(
        columns={
            "laterality": "browser_laterality",
            "view": "browser_view",
            "stratum": "reference_stratum",
        }
    )

    source = _source_table_for_campaign(campaign_dir, candidates)
    required_source = {
        "patient_id",
        "exam_id",
        "laterality",
        "view",
        "has_implant",
        "for_presentation",
        "is_marked_up",
        *SOURCE_COLUMNS,
        "view_id",
    }
    missing = sorted(required_source - set(source.columns))
    if missing:
        raise ValueError(
            f"reference source for {campaign} is missing: " + ", ".join(missing)
        )
    source = source[list(required_source)].copy()
    source = source[source["view_id"].isin(browser["view_id"])].copy()
    if source["view_id"].duplicated().any():
        raise ValueError("reference source contains duplicate selected views")
    merged = browser.merge(source, on="view_id", how="left", validate="one_to_one")
    if merged[list(SOURCE_COLUMNS)].isna().any().any():
        raise RuntimeError("reference archive cannot map every view to original DICOM")
    require_valid_sources(merged[list(SOURCE_COLUMNS)].to_dict("records"), campaign)
    if (
        not (
            merged["browser_laterality"].astype(str) == merged["laterality"].astype(str)
        ).all()
        or not (merged["browser_view"].astype(str) == merged["view"].astype(str)).all()
    ):
        raise RuntimeError("browser and original source view metadata disagree")
    if not merged["laterality"].isin(ALLOWED_LATERALITIES).all():
        raise ValueError("reference campaign contains a disallowed laterality")
    if not merged["view"].isin(ALLOWED_VIEWS).all():
        raise ValueError("reference campaign contains a disallowed projection")

    labels = _load_reference_labels(campaign_dir / "view_qc_state.json")
    merged["reference_label"] = merged["view_id"].map(labels)
    if merged["reference_label"].isna().any():
        raise ValueError("reference campaign is not completely labeled")
    merged["reference_campaign"] = campaign
    return merged.drop(columns=["browser_laterality", "browser_view"])


def _require_boolean_column(frame: pd.DataFrame, column: str) -> pd.Series:
    series = frame[column]
    if not pd.api.types.is_bool_dtype(series.dtype) or series.isna().any():
        raise ValueError(f"{column} must contain only nonmissing booleans")
    return series.astype(bool)


def _standard_control_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        ~_require_boolean_column(frame, "has_implant")
        & _require_boolean_column(frame, "for_presentation")
        & ~_require_boolean_column(frame, "is_marked_up")
    )


def _sample_unique_patients(
    pool: pd.DataFrame,
    *,
    count: int,
    stratum: str,
    seed: int,
    used_patients: set[str],
    used_exams: set[str],
    used_views: set[str],
) -> pd.DataFrame:
    shuffled = pool.sort_values("view_id", kind="stable").sample(
        frac=1.0, random_state=seed
    )
    rows: list[dict[str, Any]] = []
    for row in shuffled.to_dict("records"):
        patient_id = str(row["patient_id"])
        exam_id = str(row["exam_id"])
        view_id = str(row["view_id"])
        if (
            patient_id in used_patients
            or exam_id in used_exams
            or view_id in used_views
        ):
            continue
        used_patients.add(patient_id)
        used_exams.add(exam_id)
        used_views.add(view_id)
        row["sampling_stratum"] = stratum
        rows.append(row)
        if len(rows) == count:
            break
    if len(rows) != count:
        raise ValueError(
            f"only {len(rows)} patient-independent views are available for "
            f"{stratum}; requested {count}"
        )
    return pd.DataFrame(rows)


def sample_panel(
    campaigns: dict[str, pd.DataFrame],
    *,
    failure_count: int,
    diagnostic_control_count: int,
    standard_extra_count: int,
    seed: int,
) -> pd.DataFrame:
    """Sample a patient-disjoint mixed panel without copying prior labels."""
    seam = campaigns[SEAM_CAMPAIGN]
    film = campaigns[FILM_CAMPAIGN]
    implant = campaigns[IMPLANT_CAMPAIGN]
    diagnostic = campaigns[DIAGNOSTIC_CAMPAIGN]
    specs = [
        (
            "vertical_seam_enriched",
            seam[seam["reference_label"].eq("vertical_line")],
            failure_count,
        ),
        (
            "digitized_film_enriched",
            film[film["reference_label"].eq("present")],
            failure_count,
        ),
        (
            "visible_implant_enriched",
            implant[implant["reference_label"].eq("present")],
            failure_count,
        ),
        (
            "special_diagnostic_enriched",
            diagnostic[diagnostic["reference_label"].eq("present")],
            failure_count,
        ),
        (
            "other_diagnostic_exclusion",
            diagnostic[
                diagnostic["reference_stratum"].eq("other_diagnostic_exclusion")
                & diagnostic["reference_label"].eq("absent")
            ],
            failure_count,
        ),
        (
            "standard_control_primary",
            diagnostic[
                diagnostic["reference_stratum"].eq("standard_view_control")
                & diagnostic["reference_label"].eq("absent")
                & _standard_control_mask(diagnostic)
            ],
            diagnostic_control_count,
        ),
        (
            "standard_control_extra",
            implant[
                implant["reference_label"].eq("absent")
                & _standard_control_mask(implant)
            ],
            standard_extra_count,
        ),
    ]
    used_patients: set[str] = set()
    used_exams: set[str] = set()
    used_views: set[str] = set()
    sampled = [
        _sample_unique_patients(
            pool,
            count=count,
            stratum=stratum,
            seed=seed + offset,
            used_patients=used_patients,
            used_exams=used_exams,
            used_views=used_views,
        )
        for offset, (stratum, pool, count) in enumerate(specs)
    ]
    panel = pd.concat(sampled, ignore_index=True)
    panel = panel.sample(frac=1.0, random_state=seed + 100).reset_index(drop=True)
    panel["review_order"] = range(1, len(panel) + 1)
    if (
        panel["patient_id"].astype(str).duplicated().any()
        or panel["exam_id"].astype(str).duplicated().any()
        or panel["view_id"].duplicated().any()
    ):
        raise RuntimeError("binary QC sampling violated patient/exam/view independence")
    return panel


def _require_available_sources(panel: pd.DataFrame, raw_root: Path) -> None:
    for row in panel.to_dict("records"):
        source = DicomSource.from_row(row)
        if (
            not source.unpacked_path(raw_root).is_file()
            and not source.archive_path(raw_root).is_file()
        ):
            raise DicomSourceError(
                f"DICOM source {source.source_id} has neither unpacked data nor archive"
            )


def _write_outputs(
    out_dir: Path,
    panel: pd.DataFrame,
    *,
    target: str,
    max_render_pixels: int,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, mode=0o700)
    image_dir = out_dir / "images"
    image_dir.mkdir(mode=0o700)
    browser_rows: list[dict[str, Any]] = []
    for row in panel.to_dict("records"):
        view_id = normalize_view_id(row["view_id"])
        source_image = Path(str(row["reference_image_path"])).resolve()
        validate_rendered_view_png(source_image, max_pixels=max_render_pixels)
        destination = image_dir / f"{view_id}.png"
        os.link(source_image, destination)
        os.chmod(destination, 0o600)
        browser_rows.append(
            {
                "view_id": view_id,
                "image_path": f"images/{view_id}.png",
                "laterality": str(row["laterality"]),
                "view": str(row["view"]),
                "review_order": int(row["review_order"]),
                "stratum": str(row["sampling_stratum"]),
            }
        )
    manifest = pd.DataFrame(browser_rows).sort_values("review_order", kind="stable")
    manifest_path = out_dir / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)

    source_columns = [
        "patient_id",
        "exam_id",
        "sop_instance_uid",
        "laterality",
        "view",
        SOURCE_ARCHIVE_COLUMN,
        SOURCE_MEMBER_COLUMN,
        "sha256",
        "view_id",
        "has_implant",
        "for_presentation",
        "is_marked_up",
        "sampling_stratum",
        "reference_campaign",
        "reference_label",
        "reference_stratum",
        "review_order",
    ]
    source_manifest = panel[source_columns].copy()
    source_manifest["image_path"] = source_manifest["view_id"].map(
        lambda value: f"images/{value}.png"
    )
    source_manifest = source_manifest.sort_values("review_order", kind="stable")
    source_path = out_dir / "source_manifest.parquet"
    source_manifest.to_parquet(source_path, index=False)

    state_path = out_dir / "view_qc_state.json"
    state = save_view_qc_state(state_path, empty_view_qc_state(target))
    initialize_view_qc_event_log(default_view_qc_events_path(state_path), state)
    metadata_path = out_dir / "sampling_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    for path in (manifest_path, source_path, metadata_path):
        os.chmod(path, 0o600)

    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Blinded binary Mirai-input visual-QC pilot",
                "",
                f"- target: `{target}`",
                f"- total views: `{len(manifest)}`",
                f"- strata: `{metadata['selected_counts']}`",
                f"- seed: `{metadata['seed']}`",
                "- every view comes from a different patient and exam",
                "- only L/R CC/MLO projections are included",
                "- no prior component label is copied into the binary human state",
                "- browser manifest contains no patient, exam, SOP, or source identifiers",
                "- mode-600 source_manifest.parquet maps every view back to its original DICOM archive and member",
                "",
                "## Human decision mapping",
                "",
                f"- `{DEFAULT_NEGATIVE_LABEL}` -> internal `absent` (exclusion target absent)",
                f"- `{DEFAULT_POSITIVE_LABEL}` -> internal `present` (exclusion target present)",
                "- `Low confidence` -> independent boolean flag on either decision",
                "",
                "The prior labels and strata are enrichment only, not global usability truth.",
                "The new binary labels are authoritative for this target.",
                "",
                "## Frozen experiment",
                "",
                "Score exactly one zero-shot Qwen3.5-27B run with",
                "`qc/targets/mirai_input_view_exclusion_v2.txt`. Do not add examples or",
                "revise the prompt after inspecting these human labels. Report overall",
                "failure sensitivity, usable-view specificity, and failure sensitivity",
                "for each enrichment stratum.",
                "",
                f"Exact producer command: `{metadata['command']}`",
                "",
                "## Original-DICOM source validation",
                "",
                "```bash",
                "/gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima/bin/python \\",
                "  qc/audit_view_eligibility.py \\",
                f"  --manifest {out_dir}/manifest.parquet \\",
                f"  --candidates {out_dir}/source_manifest.parquet \\",
                f"  --raw-root {metadata['raw_root']} \\",
                f"  --output {out_dir}/source_eligibility_audit.parquet \\",
                f"  --summary {out_dir}/source_eligibility_audit_summary.json \\",
                "  --temp-root /scratch/annawoodard/tmp \\",
                "  --workers 16",
                "```",
                "",
                "## Annotation runtime",
                "",
                "```bash",
                "/gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima/bin/python \\",
                "  qc/view_qc_gallery.py \\",
                f"  --manifest {out_dir}/manifest.parquet \\",
                f"  --state {out_dir}/view_qc_state.json \\",
                f"  --events {out_dir}/view_qc_events.jsonl \\",
                "  --reviewer annawoodard \\",
                f"  --negative-label {shlex.quote(DEFAULT_NEGATIVE_LABEL)} \\",
                f"  --positive-label {shlex.quote(DEFAULT_POSITIVE_LABEL)} \\",
                "  --negative-shortcut p \\",
                "  --positive-shortcut f \\",
                f"  --review-instruction {shlex.quote(DEFAULT_REVIEW_INSTRUCTION)} \\",
                "  --review-rubric-file qc/targets/mirai_input_view_exclusion_human_v2.txt \\",
                "  --host 127.0.0.1 \\",
                "  --port 8767",
                "```",
                "",
                "Browser address through the existing tunnel: `http://localhost:8767/`.",
                "",
            ]
        )
    )
    os.chmod(readme, 0o600)
    return manifest


def run_from_args(args: argparse.Namespace) -> pd.DataFrame:
    archive_root = args.annotation_archive_root.resolve()
    candidates_path = args.candidates.resolve()
    raw_root = args.raw_root.resolve()
    out_dir = args.out_dir.resolve()
    target = normalize_view_qc_target(args.target)
    for path in (archive_root, raw_root):
        if not path.is_dir():
            raise FileNotFoundError(f"required directory not found: {path}")
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite binary QC pilot: {out_dir}")
    counts = {
        "vertical_seam_enriched": int(args.failure_count),
        "digitized_film_enriched": int(args.failure_count),
        "visible_implant_enriched": int(args.failure_count),
        "special_diagnostic_enriched": int(args.failure_count),
        "other_diagnostic_exclusion": int(args.failure_count),
        "standard_control_primary": int(args.diagnostic_control_count),
        "standard_control_extra": int(args.standard_extra_count),
    }
    if any(count <= 0 for count in counts.values()):
        raise ValueError("all binary pilot stratum counts must be positive")
    if args.max_render_pixels <= 0:
        raise ValueError("--max-render-pixels must be positive")

    candidates = _load_candidate_sources(candidates_path)
    campaigns = {
        campaign: load_reference_campaign(archive_root, campaign, candidates)
        for campaign in (
            SEAM_CAMPAIGN,
            FILM_CAMPAIGN,
            IMPLANT_CAMPAIGN,
            DIAGNOSTIC_CAMPAIGN,
        )
    }
    panel = sample_panel(
        campaigns,
        failure_count=args.failure_count,
        diagnostic_control_count=args.diagnostic_control_count,
        standard_extra_count=args.standard_extra_count,
        seed=args.seed,
    )
    _require_available_sources(panel, raw_root)
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "target": target,
        "annotation_archive_root": str(archive_root),
        "candidate_table": str(candidates_path),
        "raw_root": str(raw_root),
        "reference_campaigns": sorted(campaigns),
        "reference_labels_are_global_truth": False,
        "selected_counts": counts,
        "unique_patients": int(panel["patient_id"].astype(str).nunique()),
        "unique_exams": int(panel["exam_id"].astype(str).nunique()),
        "unique_views": int(panel["view_id"].nunique()),
        "allowed_lateralities": sorted(ALLOWED_LATERALITIES),
        "allowed_views": sorted(ALLOWED_VIEWS),
        "seed": int(args.seed),
        "max_render_pixels": int(args.max_render_pixels),
    }
    return _write_outputs(
        out_dir,
        panel,
        target=target,
        max_render_pixels=args.max_render_pixels,
        metadata=metadata,
    )


def main() -> int:
    args = parse_args()
    manifest = run_from_args(args)
    counts = manifest["stratum"].value_counts().sort_index().to_dict()
    print(f"binary Mirai-input QC pilot ready: views={len(manifest)}")
    print(f"strata={counts}")
    print(f"output: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
