#!/usr/bin/env python3
"""Gate a complete sensitive view-QC run with an exact candidate-only veto run."""

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
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--veto-manifest", type=Path, required=True)
    parser.add_argument("--veto-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--minimum-candidate-confidence",
        choices=VIEW_CONFIDENCE_LEVELS,
        required=True,
    )
    parser.add_argument(
        "--minimum-veto-confidence",
        choices=VIEW_CONFIDENCE_LEVELS,
        required=True,
    )
    return parser.parse_args()


def _load_manifest_paths(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"view manifest not found: {path}")
    frame = pd.read_parquet(path)
    validate_view_manifest_columns(frame.columns, str(path))
    if frame.empty:
        raise ValueError(f"view manifest is empty: {path}")
    frame = frame.copy()
    frame["view_id"] = frame["view_id"].map(normalize_view_id)
    if frame["view_id"].duplicated().any():
        raise ValueError(f"view manifest contains duplicate view IDs: {path}")
    return {
        str(row.view_id): str(row.image_path) for row in frame.itertuples(index=False)
    }


def _run_provenance(path: Path, run: dict[str, Any]) -> dict[str, Any]:
    settings = run["inference_settings"]
    return {
        "run_file": str(path),
        "run_sha256": sha256_file(path),
        "target": run["target"],
        "run_id": run["run_id"],
        "model": run["model"],
        "backend": run["backend"],
        "created_at": run["created_at"],
        "prompt_variant": run["prompt_variant"],
        "target_prompt_sha256": settings.get("target_prompt_sha256"),
        "few_shot_manifest_sha256": settings.get("few_shot_manifest_sha256"),
        "model_input_manifest_sha256": settings.get("model_input_manifest_sha256"),
    }


def gate_view_run(
    *,
    manifest_path: Path,
    candidate_run_path: Path,
    veto_manifest_path: Path,
    veto_run_path: Path,
    output_path: Path,
    target: str,
    minimum_candidate_confidence: str,
    minimum_veto_confidence: str,
) -> dict[str, Any]:
    """Keep candidate positives unless the candidate-only run gives a strong veto."""
    manifest_path = Path(manifest_path).resolve()
    candidate_run_path = Path(candidate_run_path).resolve()
    veto_manifest_path = Path(veto_manifest_path).resolve()
    veto_run_path = Path(veto_run_path).resolve()
    output_path = Path(output_path).resolve()
    target = normalize_view_qc_target(target)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite gated run: {output_path}")

    expected_paths = _load_manifest_paths(manifest_path)
    veto_paths = _load_manifest_paths(veto_manifest_path)
    candidate = load_view_auto_run(candidate_run_path)
    veto = load_view_auto_run(veto_run_path)
    if not candidate or not veto:
        raise ValueError("candidate and veto runs must both be complete run files")
    if candidate["target"] != target:
        raise ValueError("candidate target does not match --target")
    if veto["target"] == target:
        raise ValueError("veto run must use a distinct target")
    if set(candidate["view_suggestions"]) != set(expected_paths):
        raise ValueError("candidate run does not exactly cover the full manifest")
    if set(veto["view_suggestions"]) != set(veto_paths):
        raise ValueError("veto run does not exactly cover the veto manifest")
    for view_id, image_path in expected_paths.items():
        if candidate["view_suggestions"][view_id]["image_path"] != image_path:
            raise ValueError("candidate image path does not match the full manifest")
    for view_id, image_path in veto_paths.items():
        if view_id not in expected_paths or expected_paths[view_id] != image_path:
            raise ValueError(
                "veto manifest is not a source-linked full-manifest subset"
            )
        if veto["view_suggestions"][view_id]["image_path"] != image_path:
            raise ValueError("veto image path does not match the veto manifest")

    candidate_positive_ids = {
        view_id
        for view_id, record in candidate["view_suggestions"].items()
        if view_suggestion_is_target_present(
            record,
            target=target,
            minimum_confidence=minimum_candidate_confidence,
        )
    }
    if set(veto_paths) != candidate_positive_ids:
        raise ValueError(
            "veto manifest must exactly cover every thresholded candidate positive"
        )
    veto_positive_ids = {
        view_id
        for view_id, record in veto["view_suggestions"].items()
        if view_suggestion_is_target_present(
            record,
            target=veto["target"],
            minimum_confidence=minimum_veto_confidence,
        )
    }

    gated = new_view_auto_run(
        target=target,
        model=f"veto_gate[{candidate['model']}|{veto['model']}]",
        prompt_variant="candidate_and_not_veto_v1",
        inference_settings={
            "combination_rule": "candidate_and_not_veto",
            "minimum_candidate_confidence": minimum_candidate_confidence,
            "minimum_veto_confidence": minimum_veto_confidence,
            "candidate_positive_count": len(candidate_positive_ids),
            "veto_positive_count": len(veto_positive_ids),
            "full_manifest_sha256": sha256_file(manifest_path),
            "veto_manifest_sha256": sha256_file(veto_manifest_path),
            "candidate": _run_provenance(candidate_run_path, candidate),
            "veto": _run_provenance(veto_run_path, veto),
        },
    )
    gated["backend"] = "derived_veto_gate"
    gated["prompt_mode"] = "derived_veto_gate"
    records: dict[str, dict[str, Any]] = {}
    for view_id, image_path in expected_paths.items():
        suggestions: list[dict[str, str]] = []
        if view_id in candidate_positive_ids and view_id not in veto_positive_ids:
            candidate_suggestion = candidate["view_suggestions"][view_id][
                "suggestions"
            ][0]
            suggestions.append(
                {
                    "tag": target,
                    "confidence": candidate_suggestion["confidence"],
                    "rationale": (
                        "sensitive candidate retained after no high-confidence "
                        f"veto for {veto['target']}"
                    ),
                }
            )
        records[view_id] = {"image_path": image_path, "suggestions": suggestions}
    gated["view_suggestions"] = records
    return save_view_auto_run(output_path, gated)


def main() -> int:
    args = parse_args()
    result = gate_view_run(
        manifest_path=args.manifest,
        candidate_run_path=args.candidate_run,
        veto_manifest_path=args.veto_manifest,
        veto_run_path=args.veto_run,
        output_path=args.output,
        target=args.target,
        minimum_candidate_confidence=args.minimum_candidate_confidence,
        minimum_veto_confidence=args.minimum_veto_confidence,
    )
    print(
        "gated view auto-QC complete: "
        f"scored={len(result['view_suggestions'])} "
        f"candidates={result['inference_settings']['candidate_positive_count']} "
        f"vetoes={result['inference_settings']['veto_positive_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
