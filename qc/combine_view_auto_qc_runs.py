#!/usr/bin/env python3
"""Combine complete single-target view runs with a frozen logical OR."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from prima.view_auto_qc import (
    VIEW_CONFIDENCE_LEVELS,
    load_view_auto_run,
    new_view_auto_run,
    save_view_auto_run,
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-file", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--minimum-present-confidence",
        choices=VIEW_CONFIDENCE_LEVELS,
        required=True,
    )
    return parser.parse_args()


def _component_provenance(path: Path, run: dict[str, Any]) -> dict[str, Any]:
    settings = run["inference_settings"]
    return {
        "run_file": str(path),
        "run_sha256": sha256_file(path),
        "target": run["target"],
        "run_id": run["run_id"],
        "model": run["model"],
        "backend": run["backend"],
        "created_at": run["created_at"],
        "prompt_version": run["prompt_version"],
        "prompt_variant": run["prompt_variant"],
        "target_prompt_sha256": settings.get("target_prompt_sha256"),
        "few_shot_manifest_sha256": settings.get("few_shot_manifest_sha256"),
        "few_shot_example_count": settings.get("few_shot_example_count", 0),
        "model_image_column": settings.get("model_image_column", "image_path"),
        "model_input_manifest_sha256": settings.get("model_input_manifest_sha256"),
    }


def combine_view_runs(
    *,
    manifest_path: Path,
    run_paths: list[Path],
    output_path: Path,
    target: str,
    minimum_confidence: str,
) -> dict[str, Any]:
    """Create one provenance-rich target run from component target decisions."""
    manifest_path = Path(manifest_path).resolve()
    output_path = Path(output_path).resolve()
    target = normalize_view_qc_target(target)
    minimum_confidence = str(minimum_confidence).strip().lower()
    if minimum_confidence not in VIEW_CONFIDENCE_LEVELS:
        raise ValueError("unsupported minimum component confidence")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"view manifest not found: {manifest_path}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite combined run: {output_path}")
    resolved_runs = [Path(path).resolve() for path in run_paths]
    if len(resolved_runs) < 2:
        raise ValueError("logical-OR combination requires at least two runs")
    if len(resolved_runs) != len(set(resolved_runs)):
        raise ValueError("component run paths must be unique")

    manifest = pd.read_parquet(manifest_path)
    validate_view_manifest_columns(manifest.columns, str(manifest_path))
    manifest = manifest.copy()
    manifest["view_id"] = manifest["view_id"].map(normalize_view_id)
    if manifest["view_id"].duplicated().any():
        raise ValueError("view manifest contains duplicate view IDs")
    expected_paths = {
        str(row.view_id): str(row.image_path)
        for row in manifest.itertuples(index=False)
    }

    components: list[tuple[Path, dict[str, Any]]] = []
    for path in resolved_runs:
        if not path.is_file():
            raise FileNotFoundError(f"component run not found: {path}")
        run = load_view_auto_run(path)
        if not run:
            raise ValueError(f"component run is empty: {path}")
        if set(run["view_suggestions"]) != set(expected_paths):
            raise ValueError("component run coverage does not match the manifest")
        for view_id, record in run["view_suggestions"].items():
            if record["image_path"] != expected_paths[view_id]:
                raise ValueError("component image path does not match the manifest")
        components.append((path, run))
    component_targets = [run["target"] for _path, run in components]
    if len(component_targets) != len(set(component_targets)):
        raise ValueError("component runs must have distinct targets")

    models = sorted({run["model"] for _path, run in components})
    combined = new_view_auto_run(
        target=target,
        model="logical_or[" + ",".join(models) + "]",
        prompt_variant="logical_or_v1",
        inference_settings={
            "combination_rule": "logical_or",
            "minimum_component_confidence": minimum_confidence,
            "components": [
                _component_provenance(path, run) for path, run in components
            ],
        },
    )
    combined["backend"] = "derived_logical_or"
    combined["prompt_mode"] = "derived_logical_or"
    ranks = {level: index for index, level in enumerate(VIEW_CONFIDENCE_LEVELS)}
    records: dict[str, dict[str, Any]] = {}
    for view_id, image_path in expected_paths.items():
        positive_components: list[tuple[str, str]] = []
        for _path, run in components:
            record = run["view_suggestions"][view_id]
            if view_suggestion_is_target_present(
                record,
                target=run["target"],
                minimum_confidence=minimum_confidence,
            ):
                confidence = record["suggestions"][0]["confidence"]
                positive_components.append((run["target"], confidence))
        suggestions = []
        if positive_components:
            confidence = max(
                (value for _component, value in positive_components),
                key=ranks.__getitem__,
            )
            suggestions.append(
                {
                    "tag": target,
                    "confidence": confidence,
                    "rationale": "positive component: "
                    + ", ".join(name for name, _value in positive_components),
                }
            )
        records[view_id] = {"image_path": image_path, "suggestions": suggestions}
    combined["view_suggestions"] = records
    return save_view_auto_run(output_path, combined)


def main() -> int:
    args = parse_args()
    combined = combine_view_runs(
        manifest_path=args.manifest,
        run_paths=args.run_file,
        output_path=args.output,
        target=args.target,
        minimum_confidence=args.minimum_present_confidence,
    )
    print(
        "combined view auto-QC complete: "
        f"components={len(args.run_file)} scored={len(combined['view_suggestions'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
