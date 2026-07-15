#!/usr/bin/env python3
"""Materialize the exact positive subset required by a candidate-only veto."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from prima.view_auto_qc import (
    VIEW_CONFIDENCE_LEVELS,
    load_view_auto_run,
    view_suggestion_is_target_present,
)
from prima.view_few_shot import sha256_file
from prima.view_qc import (
    normalize_view_id,
    normalize_view_qc_target,
    validate_view_manifest_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--minimum-candidate-confidence",
        choices=VIEW_CONFIDENCE_LEVELS,
        required=True,
    )
    parser.add_argument(
        "--model-image-column",
        default="image_path",
        help="source column containing the representation used by both observers",
    )
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-candidate-run-sha256", required=True)
    return parser.parse_args()


def _require_hash(path: Path, expected: str, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{description} SHA-256 mismatch: expected={expected} found={actual}"
        )


def _safe_source_path(root: Path, value: object, description: str) -> tuple[Path, Path]:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{description} must be a safe relative path")
    source = (root / relative).resolve()
    try:
        source.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{description} escapes the source manifest root") from error
    if not source.is_file():
        raise FileNotFoundError(f"{description} not found: {source}")
    return source, relative


def _ordered_image_bank_sha256(paths: list[Path]) -> str:
    digests = [sha256_file(path) for path in paths]
    return hashlib.sha256("".join(digests).encode()).hexdigest()


def main() -> int:
    args = parse_args()
    source_manifest = args.source_manifest.resolve()
    candidate_run_path = args.candidate_run.resolve()
    output_manifest = args.output_manifest.resolve()
    output_root = output_manifest.parent
    target = normalize_view_qc_target(args.target)
    model_image_column = str(args.model_image_column).strip()
    if not model_image_column:
        raise ValueError("--model-image-column must be nonempty")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite veto input root: {output_root}")
    if output_root == source_manifest.parent:
        raise ValueError("veto inputs must remain outside the source review directory")

    _require_hash(
        source_manifest,
        args.expected_source_manifest_sha256,
        "source view manifest",
    )
    _require_hash(
        candidate_run_path,
        args.expected_candidate_run_sha256,
        "candidate view auto-QC run",
    )
    source = pd.read_parquet(source_manifest).copy()
    validate_view_manifest_columns(source.columns, str(source_manifest))
    if model_image_column not in source.columns:
        raise ValueError(
            f"source manifest lacks model image column: {model_image_column}"
        )
    source["view_id"] = source["view_id"].map(normalize_view_id)
    if source.empty or source["view_id"].duplicated().any():
        raise ValueError("source manifest is empty or contains duplicate view IDs")

    candidate = load_view_auto_run(candidate_run_path)
    if not candidate:
        raise ValueError("candidate run is empty")
    if candidate["target"] != target:
        raise ValueError("candidate run target does not match --target")
    if set(candidate["view_suggestions"]) != set(source["view_id"]):
        raise ValueError("candidate run does not exactly cover the source manifest")
    source_paths = source.set_index("view_id")["image_path"].astype(str).to_dict()
    for view_id, record in candidate["view_suggestions"].items():
        if record["image_path"] != source_paths[view_id]:
            raise ValueError("candidate canonical image path differs from the source")

    settings = candidate["inference_settings"]
    candidate_model_column = settings.get("model_image_column", "image_path")
    if candidate_model_column != model_image_column:
        raise ValueError("candidate and veto model image columns differ")
    if model_image_column != "image_path":
        if settings.get("model_input_manifest_sha256") != sha256_file(source_manifest):
            raise ValueError("candidate model-input manifest hash is not the source")

    positive_ids = {
        view_id
        for view_id, record in candidate["view_suggestions"].items()
        if view_suggestion_is_target_present(
            record,
            target=target,
            minimum_confidence=args.minimum_candidate_confidence,
        )
    }
    if not positive_ids:
        raise ValueError("candidate run has no positives at the requested confidence")
    subset = source[source["view_id"].isin(positive_ids)].copy()
    subset = subset.sort_values("review_order", kind="stable").reset_index(drop=True)
    if set(subset["view_id"]) != positive_ids:
        raise RuntimeError("positive subset construction lost candidate views")

    output_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary_root = output_root.with_name(f".{output_root.name}.building")
    if temporary_root.exists():
        raise FileExistsError(f"stale veto-input build root exists: {temporary_root}")
    temporary_root.mkdir(mode=0o700)
    try:
        materialized: dict[Path, Path] = {}
        ordered_model_paths: list[Path] = []
        for row in subset.to_dict("records"):
            for column in dict.fromkeys(("image_path", model_image_column)):
                source_path, relative = _safe_source_path(
                    source_manifest.parent,
                    row[column],
                    f"{column} image",
                )
                destination = temporary_root / relative
                if relative not in materialized:
                    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    shutil.copyfile(source_path, destination)
                    os.chmod(destination, 0o600)
                    if sha256_file(destination) != sha256_file(source_path):
                        raise RuntimeError(
                            "materialized veto image differs from source"
                        )
                    materialized[relative] = destination
            ordered_model_paths.append(materialized[Path(str(row[model_image_column]))])

        temporary_manifest = temporary_root / output_manifest.name
        subset.to_parquet(temporary_manifest, index=False)
        os.chmod(temporary_manifest, 0o600)
        provenance = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "command": shlex.join([sys.executable, *sys.argv]),
            "producer": str(Path(__file__).resolve()),
            "producer_sha256": sha256_file(Path(__file__).resolve()),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": sha256_file(source_manifest),
            "candidate_run": str(candidate_run_path),
            "candidate_run_sha256": sha256_file(candidate_run_path),
            "candidate_target": target,
            "minimum_candidate_confidence": args.minimum_candidate_confidence,
            "model_image_column": model_image_column,
            "source_rows": int(len(source)),
            "selected_rows": int(len(subset)),
            "materialized_files": int(len(materialized)),
            "ordered_model_image_bank_sha256": _ordered_image_bank_sha256(
                ordered_model_paths
            ),
            "output_manifest": str(output_manifest),
            "output_manifest_sha256": sha256_file(temporary_manifest),
        }
        provenance_path = temporary_manifest.with_suffix(".provenance.json")
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
        os.chmod(provenance_path, 0o600)
        os.replace(temporary_root, output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    print(
        "veto input manifest ready: "
        f"source={len(source)} selected={len(subset)} "
        f"manifest_sha256={sha256_file(output_manifest)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
