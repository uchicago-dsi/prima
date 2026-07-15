#!/usr/bin/env python3
"""Audit MLO source orientation from durable DICOM PatientOrientation tags."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys

import pandas as pd

from prima.dicom_source import read_dicom_source, require_source_columns
from prima.mlo_orientation import (
    DICOM_PATIENT_ORIENTATION_STANDARD,
    mlo_orientation_label_from_column_direction,
    patient_orientation_directions,
)
from prima.view_few_shot import sha256_file
from prima.view_qc import normalize_view_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "challenge"),
        action="append",
        required=True,
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--temp-root", type=Path)
    return parser.parse_args()


def _resolve_input(raw_path: object, *, description: str) -> Path:
    path = Path(str(raw_path)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def _read_source(path: Path, *, description: str) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    require_source_columns(frame.columns, description)
    required = {"view_id", "laterality", "view"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{description} missing columns: {missing}")
    frame["view_id"] = frame["view_id"].map(normalize_view_id)
    if frame.empty or frame["view_id"].duplicated().any():
        raise ValueError(f"{description} is empty or has duplicate view IDs")
    return frame


def _load_source_tables(
    config: dict[str, object],
) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    pool_specs = config.get("source_pools")
    audit_spec = config.get("audit")
    if not isinstance(pool_specs, list) or not isinstance(audit_spec, dict):
        raise ValueError("source config must define source_pools and audit")
    tables: dict[str, pd.DataFrame] = {}
    paths: dict[str, Path] = {}
    for raw_spec in pool_specs:
        if not isinstance(raw_spec, dict):
            raise ValueError("source-pool spec must be an object")
        name = str(raw_spec.get("name", ""))
        if not name or name in tables:
            raise ValueError(f"invalid or duplicate source-pool name: {name!r}")
        raw_source_path = raw_spec.get("source_manifest", raw_spec.get("manifest"))
        source_path = _resolve_input(
            raw_source_path, description=f"{name} source manifest"
        )
        tables[name] = _read_source(source_path, description=name)
        paths[f"pool_{name}_source_manifest"] = source_path
    audit_path = _resolve_input(
        audit_spec.get("source_manifest"), description="audit source manifest"
    )
    tables["audit"] = _read_source(audit_path, description="audit")
    paths["audit_source_manifest"] = audit_path
    return tables, paths


def run_from_args(args: argparse.Namespace) -> dict[str, object]:
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive")
    splits = list(dict.fromkeys(args.split))
    if len(splits) != len(args.split):
        raise ValueError("orientation audit splits must be unique")
    manifest_path = args.manifest.resolve()
    config_path = args.source_config.resolve()
    raw_root = args.raw_root.resolve()
    output_path = args.output.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"orientation manifest not found: {manifest_path}")
    if sha256_file(manifest_path) != args.expected_manifest_sha256:
        raise RuntimeError("orientation manifest SHA-256 mismatch")
    if not config_path.is_file():
        raise FileNotFoundError(f"source config not found: {config_path}")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw DICOM root not found: {raw_root}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite orientation audit: {output_path}")
    temp_root = args.temp_root.resolve() if args.temp_root else None
    if temp_root is not None and not temp_root.is_dir():
        raise FileNotFoundError(f"temporary root not found: {temp_root}")

    config = json.loads(config_path.read_text())
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("source config must use schema version 1")
    source_tables, source_paths = _load_source_tables(config)
    manifest = pd.read_parquet(manifest_path)
    required = {
        "source_view_id",
        "source_key",
        "source_pool",
        "source_review_order",
        "laterality",
        "view",
        "split",
        "rotation_degrees_clockwise",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"orientation manifest missing columns: {missing}")
    originals = manifest[
        manifest["split"].isin(splits) & manifest["rotation_degrees_clockwise"].eq(0)
    ].copy()
    if originals.empty or set(originals["split"]) != set(splits):
        raise ValueError("requested orientation audit splits are incomplete")
    if originals["source_key"].duplicated().any():
        raise ValueError("orientation audit source keys are not unique")

    tasks: list[dict[str, object]] = []
    for record in originals.to_dict("records"):
        pool = str(record["source_pool"])
        if pool not in source_tables:
            raise ValueError(f"orientation source pool has no lineage table: {pool}")
        source_view_id = normalize_view_id(record["source_view_id"])
        matches = source_tables[pool][source_tables[pool]["view_id"].eq(source_view_id)]
        if len(matches) != 1:
            raise ValueError("orientation source lineage lookup is not one-to-one")
        tasks.append({"manifest": record, "source": matches.iloc[0].to_dict()})

    def inspect(task: dict[str, object]) -> dict[str, object]:
        record = task["manifest"]
        source = task["source"]
        dataset = read_dicom_source(
            source,
            raw_root,
            stop_before_pixels=True,
            verify_sha256=False,
            temp_root=temp_root,
        )
        row_direction, column_direction = patient_orientation_directions(
            dataset.get("PatientOrientation")
        )
        return {
            "source_key": record["source_key"],
            "source_pool": record["source_pool"],
            "source_review_order": int(record["source_review_order"]),
            "split": record["split"],
            "laterality": record["laterality"],
            "view": record["view"],
            "row_direction": row_direction,
            "column_direction": column_direction,
            "column_principal_direction": column_direction[:1],
            "dicom_original_label": mlo_orientation_label_from_column_direction(
                column_direction
            ),
            "field_of_view_rotation": str(dataset.get("FieldOfViewRotation", "")),
            "field_of_view_horizontal_flip": str(
                dataset.get("FieldOfViewHorizontalFlip", "")
            ),
        }

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        records = list(executor.map(inspect, tasks))
    records = sorted(records, key=lambda record: str(record["source_key"]))
    result_frame = pd.DataFrame.from_records(records)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "dicom_standard": DICOM_PATIENT_ORIENTATION_STANDARD,
        "interpretation": (
            "PatientOrientation second value is the positive column direction "
            "from image top to bottom; principal F means superior anatomy is at "
            "the top (UPRIGHT), while principal H means superior anatomy is at "
            "the bottom (INVERTED)"
        ),
        "manifest_sha256": sha256_file(manifest_path),
        "source_config_sha256": sha256_file(config_path),
        "source_manifest_sha256": {
            name: sha256_file(path) for name, path in source_paths.items()
        },
        "raw_root": str(raw_root),
        "verified_sop_instance_uid": True,
        "verified_dicom_sha256": False,
        "splits": splits,
        "sources": len(records),
        "label_counts": {
            str(label): int(count)
            for label, count in result_frame["dicom_original_label"]
            .value_counts(dropna=False)
            .items()
        },
        "column_direction_counts": {
            str(direction): int(count)
            for direction, count in result_frame["column_direction"]
            .value_counts(dropna=False)
            .items()
        },
        "records": records,
    }
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    os.chmod(output_path, 0o600)
    print(
        "MLO DICOM patient-orientation audit ready: "
        f"sources={len(records)} labels={payload['label_counts']} "
        f"output_sha256={sha256_file(output_path)}"
    )
    return payload


def main() -> int:
    run_from_args(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
