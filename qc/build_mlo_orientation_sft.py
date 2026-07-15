#!/usr/bin/env python3
"""Build a patient-disjoint synthetic MLO orientation adapter dataset."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from prima.dicom_source import SOURCE_COLUMNS, read_dicom_source, require_source_columns
from prima.mlo_orientation import (
    DICOM_PATIENT_ORIENTATION_STANDARD,
    ORIENTATION_CANVAS_SIZE,
    ORIENTATION_PROMPT,
    REPRESENTATION_VERSION,
    mlo_orientation_label_from_column_direction,
    patient_orientation_directions,
    render_orientation_png,
)
from prima.view_few_shot import sha256_file
from prima.view_landmark_grid import resolve_relative_image
from prima.view_qc import normalize_view_id, validate_view_manifest_columns

FONT_PATH = Path("/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf")
POOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path)
    parser.add_argument("--max-dicom-workers", type=int, default=4)
    parser.add_argument("--max-source-pixels", type=int, default=2_097_152)
    return parser.parse_args()


def _read_json(path: Path, *, description: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return payload


def _resolve_input(raw_path: object, *, description: str) -> Path:
    path = Path(str(raw_path)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def _read_manifest(path: Path, *, description: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    validate_view_manifest_columns(frame.columns, str(path))
    frame = frame.copy()
    frame["view_id"] = frame["view_id"].map(normalize_view_id)
    if frame.empty:
        raise ValueError(f"{description} is empty")
    if frame["view_id"].duplicated().any():
        raise ValueError(f"{description} contains duplicate view IDs")
    return frame


def _read_state(
    path: Path, *, accepted_labels: set[str], description: str
) -> dict[str, str]:
    payload = _read_json(path, description=description)
    if payload.get("schema_version") not in {1, 2, 3} or not isinstance(
        payload.get("labels"), dict
    ):
        raise ValueError(f"{description} has an unsupported schema")
    labels: dict[str, str] = {}
    for raw_view_id, record in payload["labels"].items():
        view_id = normalize_view_id(raw_view_id)
        if not isinstance(record, dict):
            raise ValueError(f"{description} contains a malformed label record")
        label = record.get("label")
        if label not in accepted_labels:
            continue
        if bool(record.get("low_confidence", False)):
            continue
        labels[view_id] = str(label)
    if not labels:
        raise ValueError(f"{description} has no accepted confident labels")
    return labels


def _join_lineage(
    manifest: pd.DataFrame, source: pd.DataFrame, *, description: str
) -> pd.DataFrame:
    lineage_columns = [
        "patient_id",
        "exam_id",
        "view_id",
        "laterality",
        "view",
        *SOURCE_COLUMNS,
    ]
    missing = sorted(set(lineage_columns) - set(source.columns))
    if missing:
        raise ValueError(f"{description} source manifest missing columns: {missing}")
    require_source_columns(source.columns, f"{description} source manifest")
    lineage = source[lineage_columns].copy()
    lineage["view_id"] = lineage["view_id"].map(normalize_view_id)
    if lineage["view_id"].duplicated().any():
        raise ValueError(f"{description} source manifest has duplicate view IDs")
    merged = manifest.merge(
        lineage,
        on=["view_id", "laterality", "view"],
        how="left",
        validate="one_to_one",
    )
    if merged[["patient_id", "exam_id"]].isna().any().any():
        raise ValueError(f"{description} lineage join is incomplete")
    return merged


def _hash_key(salt: str, *parts: object) -> str:
    value = "|".join([salt, *(str(part) for part in parts)])
    return hashlib.sha256(value.encode()).hexdigest()


def _load_pool(
    spec: dict[str, object], *, pool_order: int, split_salt: str
) -> tuple[pd.DataFrame, dict[str, Path], dict[str, int]]:
    name = str(spec.get("name", ""))
    if not POOL_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"invalid source-pool name: {name!r}")
    accepted = spec.get("accepted_labels")
    if (
        not isinstance(accepted, list)
        or not accepted
        or not all(isinstance(value, str) and value for value in accepted)
    ):
        raise ValueError(f"source pool {name} must define accepted_labels")
    quota = spec.get("new_quota_by_laterality")
    if not isinstance(quota, dict) or set(quota) != {"L", "R"}:
        raise ValueError(f"source pool {name} must define L/R new_quota_by_laterality")
    quotas = {side: int(quota[side]) for side in ("L", "R")}
    if any(value < 0 for value in quotas.values()):
        raise ValueError(f"source pool {name} quotas must be nonnegative")

    manifest_path = _resolve_input(
        spec.get("manifest"), description=f"{name} review manifest"
    )
    state_path = _resolve_input(spec.get("state"), description=f"{name} state")
    manifest = _read_manifest(manifest_path, description=f"{name} review manifest")
    if spec.get("source_manifest") is None:
        source_path = manifest_path
        source = manifest
        required = {"patient_id", "exam_id"}
        missing = sorted(required - set(source.columns))
        if missing:
            raise ValueError(f"{name} manifest missing lineage columns: {missing}")
        require_source_columns(source.columns, f"{name} manifest")
        joined = source.copy()
    else:
        source_path = _resolve_input(
            spec.get("source_manifest"), description=f"{name} source manifest"
        )
        source = pd.read_parquet(source_path)
        joined = _join_lineage(manifest, source, description=name)
    labels = _read_state(
        state_path,
        accepted_labels=set(accepted),
        description=f"{name} human state",
    )
    joined["human_source_label"] = joined["view_id"].map(labels)
    joined = joined[
        joined["view"].eq("MLO")
        & joined["laterality"].isin(["L", "R"])
        & joined["human_source_label"].notna()
    ].copy()
    excluded_orders = spec.get("exclude_review_orders", [])
    if not isinstance(excluded_orders, list) or not all(
        isinstance(value, int) and value > 0 for value in excluded_orders
    ):
        raise ValueError(f"source pool {name} has invalid exclude_review_orders")
    if excluded_orders:
        joined = joined[~joined["review_order"].isin(excluded_orders)].copy()
    if joined.empty:
        raise ValueError(f"source pool {name} has no eligible MLO views")
    joined["source_pool"] = name
    joined["source_pool_order"] = pool_order
    joined["source_manifest_root"] = str(manifest_path.parent)
    joined["selection_key"] = joined["view_id"].map(
        lambda value: _hash_key(split_salt, "select", value)
    )
    paths = {
        f"pool_{name}_manifest": manifest_path,
        f"pool_{name}_source_manifest": source_path,
        f"pool_{name}_state": state_path,
    }
    return joined, paths, quotas


def _load_audit(
    spec: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, Path], list[int], list[int]]:
    manifest_path = _resolve_input(spec.get("manifest"), description="audit manifest")
    source_path = _resolve_input(
        spec.get("source_manifest"), description="audit source manifest"
    )
    manifest = _read_manifest(manifest_path, description="audit manifest")
    audit = _join_lineage(manifest, pd.read_parquet(source_path), description="audit")
    upright = spec.get("upright_review_orders")
    inverted = spec.get("inverted_review_orders")
    if not isinstance(upright, list) or not isinstance(inverted, list):
        raise ValueError("audit must define upright/inverted review-order lists")
    upright_orders = [int(value) for value in upright]
    inverted_orders = [int(value) for value in inverted]
    all_orders = upright_orders + inverted_orders
    if (
        not upright_orders
        or not inverted_orders
        or any(value <= 0 for value in all_orders)
        or len(all_orders) != len(set(all_orders))
    ):
        raise ValueError(
            "audit challenge review orders must be unique positive integers"
        )
    challenge = audit[audit["review_order"].isin(all_orders)]
    if len(challenge) != len(all_orders):
        raise ValueError("audit challenge review-order selection is incomplete")
    if not challenge["view"].eq("MLO").all():
        raise ValueError("audit challenge selection must contain only MLO views")
    return (
        audit,
        {
            "audit_manifest": manifest_path,
            "audit_source_manifest": source_path,
        },
        upright_orders,
        inverted_orders,
    )


def _read_prior_dataset(
    path: Path,
) -> tuple[set[str], dict[tuple[str, int], Path]]:
    frame = pd.read_parquet(path)
    required = {
        "model_image_path",
        "source_view_id",
        "split",
        "rotation_degrees_clockwise",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"prior dataset manifest missing columns: {missing}")
    originals = frame[frame["rotation_degrees_clockwise"].eq(0)].copy()
    if originals["source_view_id"].duplicated().any():
        raise ValueError("prior dataset manifest has duplicate source views")
    prior = originals[originals["split"].isin(["train", "validation"])]
    if prior.empty:
        raise ValueError("prior dataset contains no development sources")
    prior_source_ids = {normalize_view_id(value) for value in prior["source_view_id"]}
    image_paths: dict[tuple[str, int], Path] = {}
    for row in frame.to_dict("records"):
        view_id = normalize_view_id(row["source_view_id"])
        rotation = int(row["rotation_degrees_clockwise"])
        if rotation not in {0, 180}:
            raise ValueError("prior dataset contains an invalid rotation")
        image_path, _ = resolve_relative_image(
            path.parent,
            row["model_image_path"],
            description="prior model_image_path",
        )
        key = (view_id, rotation)
        if key in image_paths:
            raise ValueError("prior dataset contains a duplicate source/rotation")
        image_paths[key] = image_path
    return prior_source_ids, image_paths


def _select_sources(
    pools: pd.DataFrame,
    *,
    quotas: dict[str, dict[str, int]],
    prior_source_ids: set[str],
    prior_source_pool: str,
    audit: pd.DataFrame,
    validation_per_pool_laterality: int,
    split_salt: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if validation_per_pool_laterality <= 0:
        raise ValueError("new_validation_per_pool_laterality must be positive")
    pools = pools.copy()
    pools["is_prior_source"] = pools["view_id"].isin(prior_source_ids)
    if prior_source_pool != "*":
        pools["is_prior_source"] &= pools["source_pool"].eq(prior_source_pool)

    audit_patients = set(audit["patient_id"])
    audit_exams = set(audit["exam_id"])
    audit_views = set(audit["view_id"])
    pools["audit_overlap"] = (
        pools["patient_id"].isin(audit_patients)
        | pools["exam_id"].isin(audit_exams)
        | pools["view_id"].isin(audit_views)
    )
    if pools[pools["is_prior_source"]]["audit_overlap"].any():
        raise ValueError("a prior development source overlaps the audit pool")
    audit_removed = (
        pools[pools["audit_overlap"]]
        .groupby("source_pool")["view_id"]
        .nunique()
        .astype(int)
        .to_dict()
    )
    pools = pools[~pools["audit_overlap"]].copy()
    located_prior = set(pools.loc[pools["is_prior_source"], "view_id"])
    if located_prior != prior_source_ids:
        raise ValueError("not every prior development source was found in source pools")

    pools = pools.sort_values(
        ["is_prior_source", "source_pool_order", "selection_key"],
        ascending=[False, True, True],
        kind="stable",
    )
    before = len(pools)
    pools = pools.drop_duplicates("view_id", keep="first")
    pools = pools.drop_duplicates("patient_id", keep="first")
    pools = pools.drop_duplicates("exam_id", keep="first")
    deduplicated = before - len(pools)
    prior = pools[pools["is_prior_source"]].copy()
    if len(prior) != len(prior_source_ids):
        raise ValueError("patient/exam deduplication removed a prior source")
    if prior["patient_id"].duplicated().any() or prior["exam_id"].duplicated().any():
        raise ValueError("prior sources are not patient/exam disjoint")
    prior["split"] = "train"

    new_pieces: list[pd.DataFrame] = []
    available = pools[~pools["is_prior_source"]].copy()
    for pool_name, pool_quotas in quotas.items():
        for laterality in ("L", "R"):
            quota = pool_quotas[laterality]
            if quota == 0:
                continue
            side = available[
                available["source_pool"].eq(pool_name)
                & available["laterality"].eq(laterality)
            ].sort_values("selection_key", kind="stable")
            if len(side) < quota:
                raise ValueError(
                    f"source pool {pool_name} has only {len(side)} eligible "
                    f"{laterality} sources for quota {quota}"
                )
            selected = side.iloc[:quota].copy()
            if len(selected) < validation_per_pool_laterality:
                raise ValueError(
                    f"source pool {pool_name} {laterality} quota is smaller than "
                    "the validation allocation"
                )
            selected["validation_key"] = selected["view_id"].map(
                lambda value: _hash_key(split_salt, "validation", value)
            )
            selected = selected.sort_values("validation_key", kind="stable")
            selected["split"] = "train"
            selected.iloc[
                :validation_per_pool_laterality,
                selected.columns.get_loc("split"),
            ] = "validation"
            new_pieces.append(selected)
    new = pd.concat(new_pieces, ignore_index=True)
    selected = pd.concat([prior, new], ignore_index=True)
    if selected["patient_id"].duplicated().any():
        raise ValueError("selected development sources are not patient-disjoint")
    if selected["exam_id"].duplicated().any():
        raise ValueError("selected development sources are not exam-disjoint")
    if selected["view_id"].duplicated().any():
        raise ValueError("selected development sources are not view-disjoint")
    selected["source_key"] = [
        f"source_{index:03d}" for index in range(1, len(selected) + 1)
    ]
    diagnostics = {
        "audit_overlap_views_removed_by_pool": audit_removed,
        "cross_pool_or_patient_exam_duplicates_removed": int(deduplicated),
        "prior_training_sources": int(len(prior)),
        "new_sources": int(len(new)),
        "new_validation_sources": int(new["split"].eq("validation").sum()),
    }
    return selected, diagnostics


def _variant_label(original_label: str, rotation: int) -> str:
    if original_label not in {"UPRIGHT", "INVERTED"}:
        raise ValueError("invalid original orientation label")
    if rotation == 0:
        return original_label
    return "INVERTED" if original_label == "UPRIGHT" else "UPRIGHT"


def _assign_dicom_orientation_labels(
    sources: pd.DataFrame,
    *,
    raw_root: Path,
    temp_root: Path | None,
    max_workers: int,
) -> pd.DataFrame:
    require_source_columns(sources.columns, "selected MLO orientation sources")

    def inspect(record: dict[str, object]) -> tuple[str, str, str]:
        dataset = read_dicom_source(
            record,
            raw_root,
            stop_before_pixels=True,
            verify_sha256=False,
            temp_root=temp_root,
        )
        row_direction, column_direction = patient_orientation_directions(
            dataset.get("PatientOrientation")
        )
        label = mlo_orientation_label_from_column_direction(column_direction)
        if label not in {"UPRIGHT", "INVERTED"}:
            raise ValueError(
                "selected MLO source has no usable PatientOrientation column direction"
            )
        return row_direction, column_direction, label

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        orientations = list(executor.map(inspect, sources.to_dict("records")))
    labeled = sources.copy()
    labeled["patient_orientation_row_direction"] = [value[0] for value in orientations]
    labeled["patient_orientation_column_direction"] = [
        value[1] for value in orientations
    ]
    labeled["original_label"] = [value[2] for value in orientations]
    challenge = labeled[labeled["split"].eq("challenge")]
    if not challenge["frozen_original_label"].eq(challenge["original_label"]).all():
        raise ValueError(
            "DICOM PatientOrientation labels disagree with the frozen challenge labels"
        )
    return labeled


def _sample_id(source_view_id: str, rotation: int) -> str:
    value = f"{source_view_id}|{REPRESENTATION_VERSION}|rotation={rotation}"
    return hashlib.sha256(value.encode()).hexdigest()


def _contact_sheet(rows: pd.DataFrame, output_path: Path) -> None:
    originals = rows[rows["rotation_degrees_clockwise"].eq(0)].copy()
    originals = originals.sort_values("source_key", kind="stable")
    columns = min(4, len(originals))
    thumb_size = (300, 300)
    label_height = 54
    rows_count = (len(originals) + columns - 1) // columns
    canvas = Image.new(
        "L", (columns * thumb_size[0], rows_count * (thumb_size[1] + label_height)), 0
    )
    if not FONT_PATH.is_file():
        raise FileNotFoundError(f"contact-sheet font not found: {FONT_PATH}")
    font = ImageFont.truetype(str(FONT_PATH), size=15)
    draw = ImageDraw.Draw(canvas)
    for index, record in enumerate(originals.to_dict("records")):
        path = output_path.parents[1] / record["model_image_path"]
        with Image.open(path) as source:
            source.load()
            thumbnail = ImageOps.contain(source.convert("L"), thumb_size, Image.LANCZOS)
        column = index % columns
        row = index // columns
        x = column * thumb_size[0] + (thumb_size[0] - thumbnail.width) // 2
        y = row * (thumb_size[1] + label_height)
        canvas.paste(thumbnail, (x, y))
        label = (
            f"{record['source_key']} {record['laterality']} "
            f"{record['human_source_label']} {record['split']}"
        )
        draw.text((column * thumb_size[0] + 4, y + thumb_size[1] + 8), label, 255, font)
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    os.chmod(output_path, 0o600)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")
    os.chmod(path, 0o600)


def run_from_args(args: argparse.Namespace) -> dict[str, object]:
    config_path = args.config.resolve()
    config = _read_json(config_path, description="orientation source config")
    if config.get("schema_version") != 1:
        raise ValueError("orientation source config must use schema version 1")
    split_salt = str(config.get("split_salt", ""))
    if not split_salt:
        raise ValueError("orientation source config must define split_salt")
    output_dir = args.output_dir.resolve()
    raw_root = args.raw_root.resolve()
    temp_root = args.temp_root.resolve() if args.temp_root else None
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite MLO orientation dataset: {output_dir}"
        )
    if args.max_source_pixels <= 0:
        raise ValueError("--max-source-pixels must be positive")
    if args.max_dicom_workers <= 0:
        raise ValueError("--max-dicom-workers must be positive")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw DICOM root not found: {raw_root}")
    if temp_root is not None and not temp_root.is_dir():
        raise FileNotFoundError(f"temporary root not found: {temp_root}")
    pool_specs = config.get("source_pools")
    if not isinstance(pool_specs, list) or not pool_specs:
        raise ValueError("orientation source config must define source_pools")

    pools: list[pd.DataFrame] = []
    paths: dict[str, Path] = {"config": config_path}
    quotas: dict[str, dict[str, int]] = {}
    for pool_order, raw_spec in enumerate(pool_specs):
        if not isinstance(raw_spec, dict):
            raise ValueError("every source-pool spec must be an object")
        frame, pool_paths, pool_quotas = _load_pool(
            raw_spec, pool_order=pool_order, split_salt=split_salt
        )
        name = str(raw_spec["name"])
        if name in quotas:
            raise ValueError(f"duplicate source-pool name: {name}")
        pools.append(frame)
        paths.update(pool_paths)
        quotas[name] = pool_quotas
    combined_pools = pd.concat(pools, ignore_index=True)

    prior_manifest_path = _resolve_input(
        config.get("prior_dataset_manifest"),
        description="prior orientation dataset manifest",
    )
    paths["prior_dataset_manifest"] = prior_manifest_path
    prior_source_ids, prior_image_paths = _read_prior_dataset(prior_manifest_path)
    prior_source_pool = str(config.get("prior_source_pool", ""))
    if prior_source_pool != "*" and prior_source_pool not in quotas:
        raise ValueError(
            "prior_source_pool must name one configured source pool or use '*'"
        )
    if not any(value > 0 for quota in quotas.values() for value in quota.values()):
        raise ValueError("at least one new source quota must be positive")
    audit_spec = config.get("audit")
    if not isinstance(audit_spec, dict):
        raise ValueError("orientation source config must define audit")
    audit, audit_paths, upright_orders, inverted_orders = _load_audit(audit_spec)
    paths.update(audit_paths)
    validation_per = int(config.get("new_validation_per_pool_laterality", 0))
    development, selection_diagnostics = _select_sources(
        combined_pools,
        quotas=quotas,
        prior_source_ids=prior_source_ids,
        prior_source_pool=prior_source_pool,
        audit=audit,
        validation_per_pool_laterality=validation_per,
        split_salt=split_salt,
    )

    all_orders = upright_orders + inverted_orders
    challenge = audit[audit["review_order"].isin(all_orders)].copy()
    challenge["split"] = "challenge"
    challenge["source_pool"] = "audit"
    challenge["human_source_label"] = "frozen_challenge"
    challenge["is_prior_source"] = False
    challenge["frozen_original_label"] = challenge["review_order"].map(
        {
            **{value: "UPRIGHT" for value in upright_orders},
            **{value: "INVERTED" for value in inverted_orders},
        }
    )
    challenge["source_manifest_root"] = str(paths["audit_manifest"].parent)
    challenge["source_key"] = [
        f"challenge_{index:02d}" for index in range(1, len(challenge) + 1)
    ]
    development_patients = set(development["patient_id"])
    development_exams = set(development["exam_id"])
    development_views = set(development["view_id"])
    if development_patients & set(challenge["patient_id"]):
        raise ValueError("development and challenge patients overlap")
    if development_exams & set(challenge["exam_id"]):
        raise ValueError("development and challenge exams overlap")
    if development_views & set(challenge["view_id"]):
        raise ValueError("development and challenge views overlap")
    sources = pd.concat([development, challenge], ignore_index=True)
    if sources["view_id"].duplicated().any():
        raise ValueError("source views are not unique")
    split_order = pd.Categorical(
        sources["split"], categories=["train", "validation", "challenge"], ordered=True
    )
    sources = sources.assign(_split_order=split_order).sort_values(
        ["_split_order", "source_pool", "source_key"], kind="stable"
    )
    sources = _assign_dicom_orientation_labels(
        sources,
        raw_root=raw_root,
        temp_root=temp_root,
        max_workers=args.max_dicom_workers,
    )

    output_dir.mkdir(mode=0o700, parents=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(mode=0o700)
    records: list[dict[str, object]] = []
    image_digests: list[str] = []
    frozen_prior_images = 0
    for source in sources.to_dict("records"):
        source_view_id = normalize_view_id(source["view_id"])
        source_path, canonical_relative = resolve_relative_image(
            Path(source["source_manifest_root"]),
            source["image_path"],
            description=f"{source['source_pool']} canonical image_path",
        )
        for rotation in (0, 180):
            sample_id = _sample_id(source_view_id, rotation)
            output_path = image_dir / f"{sample_id}.png"
            prior_image_path = prior_image_paths.get((source_view_id, rotation))
            if prior_image_path is None:
                render_orientation_png(
                    source_path,
                    output_path,
                    rotation_degrees_clockwise=rotation,
                    max_source_pixels=args.max_source_pixels,
                )
                image_origin = "rendered_current"
            else:
                shutil.copyfile(prior_image_path, output_path)
                os.chmod(output_path, 0o600)
                image_origin = "frozen_prior"
                frozen_prior_images += 1
            records.append(
                {
                    "sample_id": sample_id,
                    "source_view_id": source_view_id,
                    "source_key": source["source_key"],
                    "source_pool": source["source_pool"],
                    "source_review_order": int(source["review_order"]),
                    "human_source_label": source["human_source_label"],
                    "is_prior_source": bool(source["is_prior_source"]),
                    "laterality": source["laterality"],
                    "view": source["view"],
                    "split": source["split"],
                    "rotation_degrees_clockwise": rotation,
                    "original_label": source["original_label"],
                    "orientation_label_source": "DICOM PatientOrientation",
                    "patient_orientation_row_direction": source[
                        "patient_orientation_row_direction"
                    ],
                    "patient_orientation_column_direction": source[
                        "patient_orientation_column_direction"
                    ],
                    "expected_label": _variant_label(
                        source["original_label"], rotation
                    ),
                    "image_origin": image_origin,
                    "canonical_image_path": canonical_relative,
                    "model_image_path": output_path.relative_to(output_dir).as_posix(),
                }
            )
            image_digests.append(sha256_file(output_path))
    manifest = pd.DataFrame.from_records(records)
    if manifest["sample_id"].duplicated().any():
        raise ValueError("synthetic MLO orientation sample IDs are not unique")
    manifest_path = output_dir / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    os.chmod(manifest_path, 0o600)

    contact_dir = output_dir / "contact_sheets"
    contact_sheet_paths: dict[str, Path] = {}
    for pool_name in [*quotas, "prior", "audit"]:
        originals = manifest[manifest["rotation_degrees_clockwise"].eq(0)]
        if pool_name == "prior":
            subset = originals[originals["is_prior_source"]]
        elif pool_name == "audit":
            subset = originals[originals["split"].eq("challenge")]
        else:
            subset = originals[
                originals["source_pool"].eq(pool_name) & ~originals["is_prior_source"]
            ]
        if subset.empty:
            continue
        path = contact_dir / f"{pool_name}.png"
        _contact_sheet(subset, path)
        contact_sheet_paths[pool_name] = path

    bank_digest = hashlib.sha256("".join(image_digests).encode()).hexdigest()
    command = shlex.join([sys.executable, *sys.argv])
    source_counts = (
        manifest[manifest["rotation_degrees_clockwise"].eq(0)]
        .groupby(["split", "source_pool", "laterality"])
        .size()
    )
    provenance = {
        "schema_version": 3,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "representation_version": REPRESENTATION_VERSION,
        "orientation_prompt": ORIENTATION_PROMPT,
        "orientation_labeling": {
            "method": (
                "derive each original source label from the second value of "
                "DICOM PatientOrientation; principal F is UPRIGHT and principal "
                "H is INVERTED; the 180-degree variant receives the opposite label"
            ),
            "dicom_standard": DICOM_PATIENT_ORIENTATION_STANDARD,
            "raw_root": str(raw_root),
            "verified_sop_instance_uid": True,
            "verified_dicom_sha256": False,
            "challenge_labels_match_frozen_human_labels": True,
            "original_source_labels_by_split": {
                f"{split}|{label}": int(value)
                for (split, label), value in manifest[
                    manifest["rotation_degrees_clockwise"].eq(0)
                ]
                .groupby(["split", "original_label"])
                .size()
                .items()
            },
        },
        "input_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "split_salt": split_salt,
        "split_method": (
            "all prior development sources moved to train; deterministic SHA-256 "
            "selection within configured pool/laterality quotas; first configured "
            "number by independent validation hash assigned to validation"
        ),
        "new_validation_per_pool_laterality": validation_per,
        "selection_diagnostics": selection_diagnostics,
        "challenge_upright_review_orders": sorted(upright_orders),
        "challenge_inverted_review_orders": sorted(inverted_orders),
        "source_counts": {
            f"{split}|{pool}|{side}": int(value)
            for (split, pool, side), value in source_counts.items()
        },
        "unique_sources_by_split": {
            split: int(group["source_view_id"].nunique())
            for split, group in manifest.groupby("split")
        },
        "row_counts_by_split_and_label": {
            f"{split}|{label}": int(value)
            for (split, label), value in manifest.groupby(["split", "expected_label"])
            .size()
            .items()
        },
        "rows": len(manifest),
        "frozen_prior_images_reused": frozen_prior_images,
        "new_images_rendered": len(manifest) - frozen_prior_images,
        "manifest_sha256": sha256_file(manifest_path),
        "ordered_image_bank_sha256": bank_digest,
        "contact_sheet_sha256": {
            name: sha256_file(path) for name, path in contact_sheet_paths.items()
        },
        "canvas_size": list(ORIENTATION_CANVAS_SIZE),
        "anatomy_crop": "substantial_foreground_box before rotation",
        "annotation_suppression": (
            "retain connected foreground components at least 1% the size of the "
            "largest component, dilate their support by 1.5% of image size, and "
            "zero only isolated bright pixels outside that support"
        ),
        "training_exclusions": (
            "all audit patients/exams/views, low-confidence labels, configured "
            "source exclusions, and duplicate patients/exams/views"
        ),
    }
    provenance_path = output_dir / "provenance.json"
    _write_json(provenance_path, provenance)
    readme = [
        "# Synthetic MLO orientation adapter dataset",
        "",
        f"- rows: `{len(manifest)}`",
        f"- train sources: `{provenance['unique_sources_by_split']['train']}`",
        f"- validation sources: `{provenance['unique_sources_by_split']['validation']}`",
        f"- challenge sources: `{provenance['unique_sources_by_split']['challenge']}`",
        f"- manifest SHA-256: `{provenance['manifest_sha256']}`",
        f"- ordered image-bank SHA-256: `{bank_digest}`",
        "",
        "Each source contributes an original and exact 180-degree anatomy-crop",
        "variant. Sources are selected and split by patient/exam before variants",
        "are created. Original labels come from verified DICOM PatientOrientation",
        "metadata; model inputs remain pixels only. The frozen audit challenge",
        "cannot enter adapter training.",
        "Review every per-pool contact sheet before starting model training.",
        "",
        f"Exact producer command: `{command}`",
        "",
    ]
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(readme))
    os.chmod(readme_path, 0o600)
    return {
        "rows": len(manifest),
        "manifest_sha256": provenance["manifest_sha256"],
        "bank_sha256": bank_digest,
    }


def main() -> int:
    result = run_from_args(parse_args())
    print(
        "MLO orientation adapter inputs ready: "
        f"rows={result['rows']} manifest_sha256={result['manifest_sha256']} "
        f"bank_sha256={result['bank_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
