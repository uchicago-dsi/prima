#!/usr/bin/env python3
"""Build exact original/180-degree MLO pairs for an evaluation manifest."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

import pandas as pd

from prima.dicom_source import SOURCE_COLUMNS, read_dicom_source, require_source_columns
from prima.mlo_orientation import (
    DICOM_PATIENT_ORIENTATION_STANDARD,
    REPRESENTATION_VERSION,
    mlo_orientation_label_from_column_direction,
    patient_orientation_directions,
    render_orientation_png,
)
from prima.view_few_shot import sha256_file
from prima.view_landmark_grid import resolve_relative_image
from prima.view_qc import normalize_view_id, validate_view_manifest_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--temp-root", type=Path)
    parser.add_argument("--max-dicom-workers", type=int, default=4)
    parser.add_argument("--max-source-pixels", type=int, default=2_097_152)
    return parser.parse_args()


def _sample_id(source_view_id: str, rotation: int) -> str:
    value = f"{source_view_id}|{REPRESENTATION_VERSION}|rotation={rotation}"
    return hashlib.sha256(value.encode()).hexdigest()


def _variant_label(original_label: str, rotation: int) -> str:
    if rotation == 0:
        return original_label
    return "INVERTED" if original_label == "UPRIGHT" else "UPRIGHT"


def run_from_args(args: argparse.Namespace) -> dict[str, object]:
    manifest_path = args.manifest.resolve()
    source_manifest_path = args.source_manifest.resolve()
    raw_root = args.raw_root.resolve()
    output_dir = args.output_dir.resolve()
    temp_root = args.temp_root.resolve() if args.temp_root else None
    if args.max_dicom_workers <= 0:
        raise ValueError("--max-dicom-workers must be positive")
    if args.max_source_pixels <= 0:
        raise ValueError("--max-source-pixels must be positive")
    for path, expected, description in (
        (manifest_path, args.expected_manifest_sha256, "evaluation manifest"),
        (
            source_manifest_path,
            args.expected_source_manifest_sha256,
            "evaluation source manifest",
        ),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"{description} SHA-256 mismatch: expected={expected} found={actual}"
            )
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw DICOM root not found: {raw_root}")
    if temp_root is not None and not temp_root.is_dir():
        raise FileNotFoundError(f"temporary root not found: {temp_root}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite orientation bank: {output_dir}")

    manifest = pd.read_parquet(manifest_path).copy()
    validate_view_manifest_columns(manifest.columns, str(manifest_path))
    required = {"review_order"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"evaluation manifest missing columns: {missing}")
    manifest["view_id"] = manifest["view_id"].map(normalize_view_id)
    if manifest.empty or manifest["view_id"].duplicated().any():
        raise ValueError("evaluation manifest is empty or has duplicate view IDs")
    source = pd.read_parquet(source_manifest_path).copy()
    require_source_columns(source.columns, str(source_manifest_path))
    source_required = {"view_id", "laterality", "view", *SOURCE_COLUMNS}
    missing = sorted(source_required - set(source.columns))
    if missing:
        raise ValueError(f"evaluation source manifest missing columns: {missing}")
    source["view_id"] = source["view_id"].map(normalize_view_id)
    if source.empty or source["view_id"].duplicated().any():
        raise ValueError("evaluation source manifest has duplicate view IDs")
    lineage_columns = ["view_id", "laterality", "view", *SOURCE_COLUMNS]
    mlo = manifest[
        manifest["view"].eq("MLO") & manifest["laterality"].isin(["L", "R"])
    ].copy()
    mlo = mlo.merge(
        source[lineage_columns],
        on=["view_id", "laterality", "view"],
        how="left",
        validate="one_to_one",
    )
    if mlo.empty or mlo[list(SOURCE_COLUMNS)].isna().any().any():
        raise ValueError("MLO evaluation lineage is empty or incomplete")
    mlo = mlo.sort_values("review_order", kind="stable").reset_index(drop=True)
    mlo["source_key"] = [f"audit_mlo_{index:03d}" for index in range(1, len(mlo) + 1)]

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
                "MLO evaluation source has no usable PatientOrientation column direction"
            )
        return row_direction, column_direction, label

    with ThreadPoolExecutor(max_workers=args.max_dicom_workers) as executor:
        orientations = list(executor.map(inspect, mlo.to_dict("records")))
    mlo["patient_orientation_row_direction"] = [value[0] for value in orientations]
    mlo["patient_orientation_column_direction"] = [value[1] for value in orientations]
    mlo["original_label"] = [value[2] for value in orientations]

    output_dir.mkdir(mode=0o700, parents=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(mode=0o700)
    records: list[dict[str, object]] = []
    image_digests: list[str] = []
    for source_record in mlo.to_dict("records"):
        source_view_id = normalize_view_id(source_record["view_id"])
        source_path, canonical_relative = resolve_relative_image(
            manifest_path.parent,
            source_record["image_path"],
            description="evaluation image_path",
        )
        for rotation in (0, 180):
            sample_id = _sample_id(source_view_id, rotation)
            output_path = image_dir / f"{sample_id}.png"
            render_orientation_png(
                source_path,
                output_path,
                rotation_degrees_clockwise=rotation,
                max_source_pixels=args.max_source_pixels,
            )
            records.append(
                {
                    "sample_id": sample_id,
                    "source_view_id": source_view_id,
                    "source_key": source_record["source_key"],
                    "source_pool": "whole_exam_audit",
                    "source_review_order": int(source_record["review_order"]),
                    "laterality": source_record["laterality"],
                    "view": source_record["view"],
                    "split": "audit",
                    "rotation_degrees_clockwise": rotation,
                    "original_label": source_record["original_label"],
                    "orientation_label_source": "DICOM PatientOrientation",
                    "patient_orientation_row_direction": source_record[
                        "patient_orientation_row_direction"
                    ],
                    "patient_orientation_column_direction": source_record[
                        "patient_orientation_column_direction"
                    ],
                    "expected_label": _variant_label(
                        source_record["original_label"], rotation
                    ),
                    "canonical_image_path": canonical_relative,
                    "model_image_path": output_path.relative_to(output_dir).as_posix(),
                }
            )
            image_digests.append(sha256_file(output_path))
    output_manifest = pd.DataFrame.from_records(records)
    if output_manifest["sample_id"].duplicated().any():
        raise ValueError("orientation evaluation sample IDs are not unique")
    output_manifest_path = output_dir / "manifest.parquet"
    output_manifest.to_parquet(output_manifest_path, index=False)
    os.chmod(output_manifest_path, 0o600)
    bank_sha256 = hashlib.sha256("".join(image_digests).encode()).hexdigest()
    originals = output_manifest[output_manifest["rotation_degrees_clockwise"].eq(0)]
    provenance = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "input_manifest_sha256": sha256_file(manifest_path),
        "input_source_manifest_sha256": sha256_file(source_manifest_path),
        "raw_root": str(raw_root),
        "selection": "every L/R MLO view in the input evaluation manifest",
        "orientation_labeling": {
            "method": (
                "DICOM PatientOrientation second value; principal F is UPRIGHT "
                "and principal H is INVERTED"
            ),
            "dicom_standard": DICOM_PATIENT_ORIENTATION_STANDARD,
            "verified_sop_instance_uid": True,
            "verified_dicom_sha256": False,
        },
        "sources": len(originals),
        "rows": len(output_manifest),
        "original_label_counts": {
            str(label): int(count)
            for label, count in originals["original_label"].value_counts().items()
        },
        "manifest_sha256": sha256_file(output_manifest_path),
        "ordered_image_bank_sha256": bank_sha256,
        "representation_version": REPRESENTATION_VERSION,
    }
    provenance_path = output_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    os.chmod(provenance_path, 0o600)
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# MLO orientation evaluation bank",
                "",
                f"- sources: `{provenance['sources']}`",
                f"- rows: `{provenance['rows']}`",
                f"- manifest SHA-256: `{provenance['manifest_sha256']}`",
                f"- ordered image-bank SHA-256: `{bank_sha256}`",
                "",
                "Every eligible MLO contributes its displayed image and exact",
                "180-degree partner. DICOM metadata supplies evaluation labels only;",
                "the model inputs are PNG pixels.",
                "",
                f"Exact producer command: `{provenance['command']}`",
                "",
            ]
        )
    )
    os.chmod(readme_path, 0o600)
    print(
        "MLO orientation evaluation bank ready: "
        f"sources={provenance['sources']} rows={provenance['rows']} "
        f"labels={provenance['original_label_counts']} "
        f"manifest_sha256={provenance['manifest_sha256']}"
    )
    return provenance


def main() -> int:
    run_from_args(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
