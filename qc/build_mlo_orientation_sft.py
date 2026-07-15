#!/usr/bin/env python3
"""Build an exam-disjoint synthetic MLO orientation adapter dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from prima.mlo_orientation import (
    ORIENTATION_CANVAS_SIZE,
    ORIENTATION_PROMPT,
    REPRESENTATION_VERSION,
    render_orientation_png,
)
from prima.view_few_shot import sha256_file
from prima.view_landmark_grid import resolve_relative_image
from prima.view_qc import normalize_view_id, validate_view_manifest_columns

SPLIT_SALT = "mlo-orientation-sft-split-v1"
FONT_PATH = Path("/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--development-source-manifest", type=Path, required=True)
    parser.add_argument("--development-state", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--audit-source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-per-laterality", type=int, default=4)
    parser.add_argument(
        "--challenge-upright-review-order", type=int, action="append", required=True
    )
    parser.add_argument(
        "--challenge-inverted-review-order", type=int, action="append", required=True
    )
    parser.add_argument("--max-source-pixels", type=int, default=2_097_152)
    return parser.parse_args()


def _read_manifest(path: Path, *, description: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    frame = pd.read_parquet(path)
    validate_view_manifest_columns(frame.columns, str(path))
    frame = frame.copy()
    frame["view_id"] = frame["view_id"].map(normalize_view_id)
    if frame.empty:
        raise ValueError(f"{description} is empty")
    if frame["view_id"].duplicated().any():
        raise ValueError(f"{description} contains duplicate view IDs")
    return frame


def _read_state(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"development human state not found: {path}")
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 3 or not isinstance(
        payload.get("labels"), dict
    ):
        raise ValueError("development human state must use schema version 3")
    labels: dict[str, str] = {}
    for raw_view_id, record in payload["labels"].items():
        view_id = normalize_view_id(raw_view_id)
        label = record.get("label") if isinstance(record, dict) else None
        if label not in {"present", "absent"}:
            raise ValueError("development human state contains an invalid label")
        labels[view_id] = label
    return labels


def _join_lineage(
    manifest: pd.DataFrame, source: pd.DataFrame, *, description: str
) -> pd.DataFrame:
    required = {"patient_id", "exam_id", "view_id", "laterality", "view"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"{description} source manifest missing columns: {missing}")
    lineage = source[list(required)].copy()
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


def _split_development(candidates: pd.DataFrame, per_laterality: int) -> pd.DataFrame:
    if per_laterality <= 0:
        raise ValueError("--validation-per-laterality must be positive")
    if candidates["patient_id"].duplicated().any():
        raise ValueError(
            "usable development MLO pool must contain one source per patient"
        )
    if candidates["exam_id"].duplicated().any():
        raise ValueError("usable development MLO pool must contain one source per exam")
    pieces = []
    for laterality in ("L", "R"):
        side = candidates[candidates["laterality"].eq(laterality)].copy()
        if len(side) <= per_laterality:
            raise ValueError(
                f"insufficient usable development {laterality} MLO sources"
            )
        side["split_key"] = side["view_id"].map(
            lambda value: hashlib.sha256(f"{SPLIT_SALT}|{value}".encode()).hexdigest()
        )
        side = side.sort_values("split_key", kind="stable")
        side["split"] = "train"
        side.iloc[:per_laterality, side.columns.get_loc("split")] = "validation"
        pieces.append(side)
    result = pd.concat(pieces, ignore_index=True)
    if set(result["split"]) != {"train", "validation"}:
        raise ValueError("development split is incomplete")
    return result.sort_values(["split", "split_key"], kind="stable")


def _variant_label(original_label: str, rotation: int) -> str:
    if original_label not in {"UPRIGHT", "INVERTED"}:
        raise ValueError("invalid original orientation label")
    if rotation == 0:
        return original_label
    return "INVERTED" if original_label == "UPRIGHT" else "UPRIGHT"


def _sample_id(source_view_id: str, rotation: int) -> str:
    value = f"{source_view_id}|{REPRESENTATION_VERSION}|rotation={rotation}"
    return hashlib.sha256(value.encode()).hexdigest()


def _contact_sheet(rows: pd.DataFrame, output_path: Path) -> None:
    originals = rows[rows["rotation_degrees_clockwise"].eq(0)].copy()
    originals = originals.sort_values(["split", "source_review_order"], kind="stable")
    columns = 5
    thumb_size = (210, 210)
    label_height = 42
    rows_count = (len(originals) + columns - 1) // columns
    canvas = Image.new(
        "L", (columns * thumb_size[0], rows_count * (thumb_size[1] + label_height)), 0
    )
    if not FONT_PATH.is_file():
        raise FileNotFoundError(f"contact-sheet font not found: {FONT_PATH}")
    font = ImageFont.truetype(str(FONT_PATH), size=18)
    draw = ImageDraw.Draw(canvas)
    for index, record in enumerate(originals.to_dict("records")):
        path = output_path.parent / record["model_image_path"]
        with Image.open(path) as source:
            source.load()
            thumbnail = ImageOps.contain(source.convert("L"), thumb_size, Image.LANCZOS)
        column = index % columns
        row = index // columns
        x = column * thumb_size[0] + (thumb_size[0] - thumbnail.width) // 2
        y = row * (thumb_size[1] + label_height)
        canvas.paste(thumbnail, (x, y))
        label = f"{record['split']} #{int(record['source_review_order']):03d}"
        draw.text((column * thumb_size[0] + 4, y + thumb_size[1] + 8), label, 255, font)
    canvas.save(output_path, format="PNG", optimize=True)
    os.chmod(output_path, 0o600)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")
    os.chmod(path, 0o600)


def run_from_args(args: argparse.Namespace) -> dict[str, object]:
    paths = {
        "development_manifest": args.development_manifest.resolve(),
        "development_source_manifest": args.development_source_manifest.resolve(),
        "development_state": args.development_state.resolve(),
        "audit_manifest": args.audit_manifest.resolve(),
        "audit_source_manifest": args.audit_source_manifest.resolve(),
    }
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite MLO orientation dataset: {output_dir}"
        )
    if args.max_source_pixels <= 0:
        raise ValueError("--max-source-pixels must be positive")

    development_manifest = _read_manifest(
        paths["development_manifest"], description="development manifest"
    )
    development_source = pd.read_parquet(paths["development_source_manifest"])
    development = _join_lineage(
        development_manifest, development_source, description="development"
    )
    labels = _read_state(paths["development_state"])
    development["human_label"] = development["view_id"].map(labels)
    if development["human_label"].isna().any():
        raise ValueError("development manifest has unlabeled views")

    audit_manifest = _read_manifest(
        paths["audit_manifest"], description="audit manifest"
    )
    audit_source = pd.read_parquet(paths["audit_source_manifest"])
    audit = _join_lineage(audit_manifest, audit_source, description="audit")
    overlap = {
        "patients": len(set(development["patient_id"]) & set(audit["patient_id"])),
        "exams": len(set(development["exam_id"]) & set(audit["exam_id"])),
        "views": len(set(development["view_id"]) & set(audit["view_id"])),
    }
    if any(overlap.values()):
        raise ValueError("development and audit pools are not disjoint")

    candidates = development[
        development["view"].eq("MLO") & development["human_label"].eq("absent")
    ].copy()
    candidates = _split_development(candidates, args.validation_per_laterality)
    candidates["source_pool"] = "development"
    candidates["original_label"] = "UPRIGHT"

    upright_orders = [int(value) for value in args.challenge_upright_review_order]
    inverted_orders = [int(value) for value in args.challenge_inverted_review_order]
    all_orders = upright_orders + inverted_orders
    if any(value <= 0 for value in all_orders) or len(all_orders) != len(
        set(all_orders)
    ):
        raise ValueError("challenge review orders must be unique positive integers")
    challenge = audit[audit["review_order"].isin(all_orders)].copy()
    if len(challenge) != len(all_orders):
        raise ValueError("challenge review-order selection is incomplete")
    if not challenge["view"].eq("MLO").all():
        raise ValueError("challenge selection must contain only MLO views")
    challenge["split"] = "challenge"
    challenge["source_pool"] = "audit"
    challenge["original_label"] = challenge["review_order"].map(
        {
            **{value: "UPRIGHT" for value in upright_orders},
            **{value: "INVERTED" for value in inverted_orders},
        }
    )
    sources = pd.concat([candidates, challenge], ignore_index=True)

    output_dir.mkdir(mode=0o700, parents=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(mode=0o700)
    records: list[dict[str, object]] = []
    image_digests: list[str] = []
    roots = {
        "development": paths["development_manifest"].parent,
        "audit": paths["audit_manifest"].parent,
    }
    for source in sources.to_dict("records"):
        source_view_id = normalize_view_id(source["view_id"])
        source_path, canonical_relative = resolve_relative_image(
            roots[source["source_pool"]],
            source["image_path"],
            description=f"{source['source_pool']} canonical image_path",
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
                    "source_pool": source["source_pool"],
                    "source_review_order": int(source["review_order"]),
                    "laterality": source["laterality"],
                    "view": source["view"],
                    "split": source["split"],
                    "rotation_degrees_clockwise": rotation,
                    "expected_label": _variant_label(
                        source["original_label"], rotation
                    ),
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
    contact_sheet_path = output_dir / "source_contact_sheet.png"
    _contact_sheet(manifest, contact_sheet_path)

    bank_digest = hashlib.sha256("".join(image_digests).encode()).hexdigest()
    command = shlex.join([sys.executable, *sys.argv])
    source_counts = (
        manifest.groupby(["split", "expected_label"]).size().unstack(fill_value=0)
    )
    provenance = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "representation_version": REPRESENTATION_VERSION,
        "orientation_prompt": ORIENTATION_PROMPT,
        "input_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "development_audit_overlap": overlap,
        "development_selection": "MLO plus completed human label absent",
        "split_method": (
            "one source per patient/exam; SHA-256 source-view ordering within "
            "laterality; first N per laterality assigned to validation"
        ),
        "split_salt": SPLIT_SALT,
        "validation_per_laterality": args.validation_per_laterality,
        "challenge_upright_review_orders": sorted(upright_orders),
        "challenge_inverted_review_orders": sorted(inverted_orders),
        "source_counts": {
            split: int(group["source_view_id"].nunique())
            for split, group in manifest.groupby("split")
        },
        "row_counts": {
            split: {label: int(value) for label, value in row.items()}
            for split, row in source_counts.to_dict("index").items()
        },
        "rows": len(manifest),
        "manifest_sha256": sha256_file(manifest_path),
        "ordered_image_bank_sha256": bank_digest,
        "contact_sheet_sha256": sha256_file(contact_sheet_path),
        "canvas_size": list(ORIENTATION_CANVAS_SIZE),
        "anatomy_crop": "substantial_foreground_box before rotation",
        "annotation_suppression": (
            "retain connected foreground components at least 1% the size of the "
            "largest component, dilate their support by 1.5% of image size, and "
            "zero only isolated bright pixels outside that support"
        ),
        "training_exclusions": (
            "all audit patients/exams/views and every human development failure"
        ),
    }
    provenance_path = output_dir / "provenance.json"
    _write_json(provenance_path, provenance)
    readme = [
        "# Synthetic MLO orientation adapter dataset",
        "",
        f"- rows: `{len(manifest)}`",
        f"- train sources: `{provenance['source_counts']['train']}`",
        f"- validation sources: `{provenance['source_counts']['validation']}`",
        f"- challenge sources: `{provenance['source_counts']['challenge']}`",
        f"- manifest SHA-256: `{provenance['manifest_sha256']}`",
        f"- ordered image-bank SHA-256: `{bank_digest}`",
        "",
        "Each source contributes an original and exact 180-degree anatomy-crop",
        "variant. Development sources are usable human-reviewed MLO views and",
        "are split by patient/exam before variants are created. Audit challenge",
        "sources are evaluation-only and cannot enter adapter training.",
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
