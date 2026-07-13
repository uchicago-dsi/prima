#!/usr/bin/env python3
"""Run frozen single-target inference over individual mammography views."""

from __future__ import annotations

import argparse
import hashlib
import os
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from auto_annotate_qc import (
    DEFAULT_MODELS_DIR,
    DEFAULT_VLLM_MODEL_REGISTRY,
    PROMPT_VARIANTS,
    VLLMVisionAnnotator,
    install_shutdown_handlers,
    shutdown_requested,
)
from prima.view_auto_qc import (
    load_view_auto_run,
    new_view_auto_run,
    require_compatible_view_auto_run,
    save_view_auto_run,
)
from prima.view_few_shot import load_view_few_shot_manifest, sha256_file
from prima.view_qc import (
    normalize_view_id,
    normalize_view_qc_target,
    validate_view_manifest_columns,
)
from prima.vllm_server import (
    find_available_loopback_port,
    resolve_model_path,
    select_model_spec,
    validate_vllm_runtime,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a source-linked individual-view QC manifest with local vLLM."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--model-image-column",
        default="image_path",
        help=(
            "manifest column containing the image sent to the model; image_path "
            "always remains the canonical target-view path saved in the run"
        ),
    )
    parser.add_argument("--run-file", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--target-prompt-file", type=Path, required=True)
    parser.add_argument("--few-shot-manifest", type=Path, default=None)
    parser.add_argument("--model-key", default="qwen35_27b_fp8")
    parser.add_argument(
        "--model-registry", type=Path, default=DEFAULT_VLLM_MODEL_REGISTRY
    )
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--vllm-port", type=int, default=0)
    parser.add_argument("--vllm-server-log", type=Path, default=None)
    parser.add_argument("--startup-timeout-seconds", type=int, default=1800)
    parser.add_argument("--request-timeout-seconds", type=int, default=600)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--prompt-variant",
        choices=PROMPT_VARIANTS,
        default="confidence_specificity",
    )
    parser.add_argument("--debug-dump-dir", type=Path, default=None)
    parser.add_argument("--force-rescore", action="store_true")
    parser.add_argument("--expected-gpus", type=int, default=None)
    return parser


def load_target_prompt(path: Path, *, target: str) -> str:
    """Load one explicit target prompt and validate its output contract."""
    target = normalize_view_qc_target(target)
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"target prompt file not found: {path}")
    prompt = path.read_text().strip()
    if not prompt:
        raise ValueError("target prompt file is empty")
    if len(prompt) > 20_000:
        raise ValueError("target prompt file exceeds 20,000 characters")
    required_lines = ("EVIDENCE:", "ANSWER:", "CONFIDENCE:", "REVIEW:")
    missing = [line for line in required_lines if line not in prompt]
    if missing:
        raise ValueError(
            "target prompt is missing output fields: " + ", ".join(missing)
        )
    normalized_prompt = " ".join(prompt.split()).casefold()
    if target.casefold() not in normalized_prompt:
        raise ValueError("target prompt does not name --target exactly")
    return prompt


def _resolve_manifest_image(
    *, root: Path, raw_path: object, description: str
) -> tuple[Path, str]:
    relative = Path(str(raw_path))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"view auto-QC {description} must be a safe relative path")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"view auto-QC {description} escapes its manifest root"
        ) from error
    if not resolved.is_file():
        raise FileNotFoundError(f"view auto-QC {description} is missing")
    return resolved, relative.as_posix()


def load_view_records(
    manifest_path: Path, *, model_image_column: str = "image_path"
) -> list[dict[str, str]]:
    manifest = pd.read_parquet(manifest_path)
    validate_view_manifest_columns(manifest.columns, str(manifest_path))
    if manifest.empty:
        raise ValueError("view auto-QC manifest is empty")
    model_image_column = str(model_image_column).strip()
    if not model_image_column:
        raise ValueError("--model-image-column must be nonempty")
    if model_image_column not in manifest.columns:
        raise ValueError(
            f"view auto-QC manifest lacks model image column: {model_image_column}"
        )
    manifest = manifest.sort_values("review_order", kind="stable")
    root = manifest_path.parent.resolve()
    records: list[dict[str, str]] = []
    for row in manifest.to_dict("records"):
        view_id = normalize_view_id(row["view_id"])
        _canonical_path, saved_image_path = _resolve_manifest_image(
            root=root,
            raw_path=row["image_path"],
            description="canonical image_path",
        )
        model_path, saved_model_path = _resolve_manifest_image(
            root=root,
            raw_path=row[model_image_column],
            description=f"model image column {model_image_column}",
        )
        record = {
            "view_id": view_id,
            "image_path": str(model_path),
            "saved_image_path": saved_image_path,
        }
        if model_image_column != "image_path":
            record["saved_model_image_path"] = saved_model_path
        records.append(record)
    if len(records) != len({record["view_id"] for record in records}):
        raise ValueError("view auto-QC manifest contains duplicate view IDs")
    return records


def build_inference_settings(
    args: argparse.Namespace,
    model_spec: Any,
    target_prompt: str,
    few_shot_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = {
        "model_key": model_spec.key,
        "model_revision": model_spec.revision,
        "runtime_versions": {
            package: metadata.version(package)
            for package in ("dsi-local-llms", "openai", "vllm")
        },
        "serve_environment": dict(model_spec.environment),
        "serve_extra_args": list(model_spec.extra_args),
        "request_timeout_seconds": int(args.request_timeout_seconds),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": 0.0,
        "thinking_disabled": True,
        "target_prompt_sha256": hashlib.sha256(target_prompt.encode()).hexdigest(),
        "target_prompt_text": target_prompt,
    }
    if few_shot_metadata:
        settings.update(few_shot_metadata)
    model_image_column = str(getattr(args, "model_image_column", "image_path")).strip()
    if model_image_column != "image_path":
        settings.update(
            {
                "model_image_column": model_image_column,
                "model_input_manifest_sha256": sha256_file(args.manifest),
            }
        )
    return settings


def run_from_args(args: argparse.Namespace) -> int:
    install_shutdown_handlers()
    validate_vllm_runtime()
    manifest_path = args.manifest.resolve()
    run_file = args.run_file.resolve()
    target = normalize_view_qc_target(args.target)
    target_prompt = load_target_prompt(args.target_prompt_file, target=target)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"view manifest not found: {manifest_path}")
    if not 0 <= args.vllm_port <= 65535:
        raise ValueError("--vllm-port must be between 0 and 65535")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    model_spec = select_model_spec(args.model_registry.resolve(), args.model_key)
    model_path = resolve_model_path(model_spec, args.models_dir.resolve())
    if (
        args.expected_gpus is not None
        and args.expected_gpus != model_spec.tensor_parallel_size
    ):
        raise ValueError(
            f"model requires {model_spec.tensor_parallel_size} GPUs, "
            f"but --expected-gpus={args.expected_gpus}"
        )
    records = load_view_records(
        manifest_path, model_image_column=args.model_image_column
    )
    if args.few_shot_manifest is None:
        few_shot_exemplars: list[dict[str, Any]] = []
        few_shot_metadata: dict[str, Any] = {}
    else:
        few_shot_exemplars, few_shot_metadata = load_view_few_shot_manifest(
            args.few_shot_manifest,
            target=target,
            excluded_view_ids=(record["view_id"] for record in records),
        )
    model_label = f"{model_spec.served_model_name}@{model_spec.revision[:12]}"
    current = new_view_auto_run(
        target=target,
        model=model_label,
        prompt_variant=args.prompt_variant,
        inference_settings=build_inference_settings(
            args,
            model_spec,
            target_prompt,
            few_shot_metadata=few_shot_metadata,
        ),
    )
    existing = load_view_auto_run(run_file)
    if existing and not args.force_rescore:
        require_compatible_view_auto_run(existing, current)
        payload = existing
    else:
        payload = save_view_auto_run(run_file, current)

    if args.vllm_server_log is not None:
        server_log = args.vllm_server_log.resolve()
    else:
        process_label = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
        server_log = run_file.with_name(f"{run_file.stem}.vllm_{process_label}.log")
    debug_dir = args.debug_dump_dir.resolve() if args.debug_dump_dir else None
    annotator = VLLMVisionAnnotator(
        model_spec=model_spec,
        model_path=model_path,
        port=args.vllm_port or find_available_loopback_port(),
        server_log_path=server_log,
        startup_timeout_seconds=args.startup_timeout_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
        max_new_tokens=args.max_new_tokens,
        few_shot_examples=len(few_shot_exemplars),
        few_shot_exemplar_pool=few_shot_exemplars,
        prompt_mode="marker_classifier",
        prompt_variant=args.prompt_variant,
        probe_tag=target,
        target_prompt_override=target_prompt,
        text_only_prompt=None,
        disable_thinking=True,
        debug_dump_dir=debug_dir,
        input_level="view",
    )
    interrupted = False
    try:
        view_suggestions = dict(payload["view_suggestions"])
        for record in tqdm(records, desc="view auto-QC"):
            if shutdown_requested():
                interrupted = True
                break
            view_id = record["view_id"]
            if view_id in view_suggestions and not args.force_rescore:
                continue
            result = annotator.annotate(
                exam_id=view_id,
                image_path=Path(record["image_path"]),
                tag_catalog=[target],
            )
            view_suggestions[view_id] = {
                "image_path": record["saved_image_path"],
                "suggestions": result["suggestions"],
                "debug_dump_file": result["debug_dump_file"],
            }
            payload = {**payload, "view_suggestions": view_suggestions}
            payload = save_view_auto_run(run_file, payload)
        payload = save_view_auto_run(run_file, payload)
    finally:
        annotator.close()

    print(f"view auto-QC complete: scored={len(payload['view_suggestions'])}")
    return 130 if interrupted else 0


def main() -> int:
    return run_from_args(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
