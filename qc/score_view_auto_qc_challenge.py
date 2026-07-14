#!/usr/bin/env python3
"""Score an exact view challenge and require independent-run repeatability."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from prima.view_auto_qc import load_view_auto_run, view_suggestion_is_target_present
from prima.view_qc import normalize_view_id, normalize_view_qc_target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-file", type=Path, action="append", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_protocol_contract(protocol: Any, *, arm_name: str) -> dict[str, Any]:
    if not isinstance(protocol, dict):
        raise ValueError("challenge protocol must be a JSON object")
    repeatability = protocol.get("repeatability")
    if not isinstance(repeatability, dict):
        raise ValueError("challenge protocol requires a repeatability object")
    required_runs = repeatability.get("required_independent_runs")
    if not isinstance(required_runs, int) or required_runs < 2:
        raise ValueError("repeatability requires at least two independent runs")
    minimum_confidence = repeatability.get("minimum_confidence")
    if minimum_confidence not in {"low", "medium", "high"}:
        raise ValueError("repeatability requires an explicit minimum_confidence")

    model = protocol.get("model")
    if not isinstance(model, dict):
        raise ValueError("challenge protocol requires a model object")
    for field in ("model_key", "revision"):
        if not isinstance(model.get(field), str) or not model[field].strip():
            raise ValueError(f"challenge protocol model requires {field}")
    if not isinstance(model.get("serve_extra_args"), list):
        raise ValueError("challenge protocol model requires serve_extra_args")
    if not isinstance(model.get("request_chat_template_kwargs"), dict):
        raise ValueError(
            "challenge protocol model requires request_chat_template_kwargs"
        )
    runtime_versions = model.get("runtime_versions")
    if not isinstance(runtime_versions, dict) or not runtime_versions:
        raise ValueError("challenge protocol model requires runtime_versions")

    input_spec = protocol.get("input")
    if not isinstance(input_spec, dict):
        raise ValueError("challenge protocol requires an input object")
    for field in (
        "challenge_manifest_sha256",
        "rows",
        "model_image_column",
    ):
        if field not in input_spec:
            raise ValueError(f"challenge protocol input requires {field}")
    if not isinstance(input_spec["rows"], int) or input_spec["rows"] <= 0:
        raise ValueError("challenge protocol input rows must be positive")

    arms = protocol.get("arms")
    if not isinstance(arms, dict) or arm_name not in arms:
        raise ValueError(f"challenge protocol has no arm {arm_name!r}")
    arm = arms[arm_name]
    if not isinstance(arm, dict):
        raise ValueError(f"challenge arm {arm_name!r} must be an object")
    for field in (
        "target",
        "target_prompt_sha256",
        "required_component_positive_review_orders",
        "required_component_negative_review_orders",
    ):
        if field not in arm:
            raise ValueError(f"challenge arm {arm_name!r} requires {field}")
    return {
        "required_runs": required_runs,
        "minimum_confidence": minimum_confidence,
        "model": model,
        "input": input_spec,
        "arm": arm,
    }


def main() -> int:
    args = build_parser().parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite challenge score: {output}")

    manifest_path = args.manifest.resolve()
    manifest = pd.read_parquet(manifest_path)
    required_columns = {"review_order", "view_id"}
    missing_columns = sorted(required_columns - set(manifest.columns))
    if missing_columns:
        raise ValueError(f"challenge manifest lacks columns: {missing_columns}")
    if manifest.empty:
        raise ValueError("challenge manifest is empty")
    if manifest["review_order"].duplicated().any():
        raise ValueError("challenge manifest has duplicate review orders")
    normalized_ids = manifest["view_id"].map(normalize_view_id)
    if normalized_ids.duplicated().any():
        raise ValueError("challenge manifest has duplicate view IDs")
    order_by_view = dict(zip(normalized_ids, manifest["review_order"].astype(int)))

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    contract = require_protocol_contract(protocol, arm_name=args.arm)
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != contract["input"]["challenge_manifest_sha256"]:
        raise ValueError("challenge manifest hash does not match protocol")
    if len(manifest) != contract["input"]["rows"]:
        raise ValueError("challenge manifest row count does not match protocol")
    arm = contract["arm"]
    target = normalize_view_qc_target(arm["target"])
    required_positive = sorted(
        {int(value) for value in arm["required_component_positive_review_orders"]}
    )
    required_negative = sorted(
        {int(value) for value in arm["required_component_negative_review_orders"]}
    )
    if set(required_positive) & set(required_negative):
        raise ValueError("challenge positive and negative orders overlap")
    unknown_orders = sorted(
        (set(required_positive) | set(required_negative)) - set(order_by_view.values())
    )
    if unknown_orders:
        raise ValueError(f"challenge orders are absent from manifest: {unknown_orders}")

    run_paths = [path.resolve() for path in args.run_file]
    if len(run_paths) != contract["required_runs"]:
        raise ValueError(
            f"protocol requires {contract['required_runs']} independent runs, "
            f"received {len(run_paths)}"
        )
    if len(run_paths) != len(set(run_paths)):
        raise ValueError("independent challenge run paths must be unique")

    run_results = []
    decision_vectors: list[dict[int, bool]] = []
    run_ids: set[str] = set()
    expected_settings: dict[str, Any] | None = None
    for run_path in run_paths:
        run = load_view_auto_run(run_path)
        if run["target"] != target:
            raise ValueError(f"run target does not match challenge arm: {run_path}")
        if set(run["view_suggestions"]) != set(order_by_view):
            raise ValueError(
                f"run does not exactly cover challenge manifest: {run_path}"
            )
        settings = run["inference_settings"]
        for field in ("model_key", "model_revision"):
            expected = contract["model"][
                "revision" if field == "model_revision" else field
            ]
            if settings.get(field) != expected:
                raise ValueError(
                    f"run {field} does not match protocol model: {run_path}"
                )
        if settings.get("serve_extra_args") != contract["model"]["serve_extra_args"]:
            raise ValueError(f"run serve_extra_args do not match protocol: {run_path}")
        if (
            settings.get("request_chat_template_kwargs")
            != contract["model"]["request_chat_template_kwargs"]
        ):
            raise ValueError(
                f"run request_chat_template_kwargs do not match protocol: {run_path}"
            )
        if settings.get("runtime_versions") != contract["model"]["runtime_versions"]:
            raise ValueError(f"run runtime versions do not match protocol: {run_path}")
        if settings.get("temperature") != 0.0:
            raise ValueError(f"run temperature is not frozen at zero: {run_path}")
        if settings.get("thinking_disabled") is not True:
            raise ValueError(f"run thinking mode does not match protocol: {run_path}")
        if settings.get("target_prompt_sha256") != arm["target_prompt_sha256"]:
            raise ValueError(f"run target prompt does not match protocol: {run_path}")
        model_image_column = contract["input"]["model_image_column"]
        if model_image_column != "image_path":
            if settings.get("model_image_column") != model_image_column:
                raise ValueError(
                    f"run model image column does not match protocol: {run_path}"
                )
            if settings.get("model_input_manifest_sha256") != manifest_sha256:
                raise ValueError(
                    f"run input manifest does not match protocol: {run_path}"
                )
        comparable_settings = {
            "model": run["model"],
            "backend": run["backend"],
            "prompt_version": run["prompt_version"],
            "prompt_mode": run["prompt_mode"],
            "prompt_variant": run["prompt_variant"],
            "inference_settings": settings,
        }
        if expected_settings is None:
            expected_settings = comparable_settings
        elif comparable_settings != expected_settings:
            raise ValueError(
                "independent runs do not share identical inference settings"
            )
        if not run["run_id"] or run["run_id"] in run_ids:
            raise ValueError("independent runs require unique nonempty run IDs")
        run_ids.add(run["run_id"])

        decisions = {
            order_by_view[view_id]: view_suggestion_is_target_present(
                record,
                target=target,
                minimum_confidence=contract["minimum_confidence"],
            )
            for view_id, record in run["view_suggestions"].items()
        }
        decision_vectors.append(decisions)
        positive_orders = sorted(
            order for order, present in decisions.items() if present
        )
        missing_positive = sorted(set(required_positive) - set(positive_orders))
        violated_negative = sorted(set(required_negative) & set(positive_orders))
        run_results.append(
            {
                "run_file": str(run_path),
                "run_sha256": sha256_file(run_path),
                "run_id": run["run_id"],
                "observed_positive_review_orders": positive_orders,
                "missing_required_positive_review_orders": missing_positive,
                "violated_required_negative_review_orders": violated_negative,
                "passes_exact_component_challenge": not missing_positive
                and not violated_negative,
            }
        )

    unstable_orders = sorted(
        order
        for order in order_by_view.values()
        if len({decisions[order] for decisions in decision_vectors}) > 1
    )
    passes_repeatability = not unstable_orders
    passes_each_run = all(
        result["passes_exact_component_challenge"] for result in run_results
    )
    result = {
        "schema_version": 2,
        "arm": args.arm,
        "target": target,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "minimum_confidence": contract["minimum_confidence"],
        "required_independent_runs": contract["required_runs"],
        "required_positive_review_orders": required_positive,
        "required_negative_review_orders": required_negative,
        "unstable_review_orders": unstable_orders,
        "passes_repeatability": passes_repeatability,
        "passes_each_run_component_challenge": passes_each_run,
        "passes_exact_component_challenge": passes_repeatability and passes_each_run,
        "runs": run_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        os.chmod(temporary, 0o600)
        temporary.replace(output)
        os.chmod(output, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        "view challenge scored: "
        f"arm={args.arm} passes={result['passes_exact_component_challenge']} "
        f"repeatable={passes_repeatability} unstable={len(unstable_orders)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
