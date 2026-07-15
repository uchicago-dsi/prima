#!/usr/bin/env python3
"""Score one frozen MLO orientation adapter on selected manifest splits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from prima.view_few_shot import sha256_file
from qc.train_mlo_orientation_lora import (
    SUPPORTED_MODEL_ARCHITECTURES,
    _generate_predictions,
    _json_safe,
    _load_manifest,
    _score_by_split,
    _validate_model_snapshot,
    _validate_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--expected-adapter-config-sha256", required=True)
    parser.add_argument("--expected-model-repo-id", required=True)
    parser.add_argument("--expected-model-revision", required=True)
    parser.add_argument(
        "--expected-model-architecture",
        choices=SUPPORTED_MODEL_ARCHITECTURES,
        required=True,
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "challenge"),
        action="append",
        required=True,
    )
    parser.add_argument("--precision", choices=("bf16", "fp16"), default="fp16")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    return parser.parse_args()


def run_from_args(args: argparse.Namespace) -> dict[str, object]:
    runtime_versions = _validate_runtime()
    if args.max_new_tokens != 1:
        raise ValueError("orientation evaluation requires --max-new-tokens=1")
    splits = list(dict.fromkeys(args.split))
    if len(splits) != len(args.split):
        raise ValueError("orientation evaluation splits must be unique")

    manifest_path = args.manifest.resolve()
    model_path = args.model_path.resolve()
    adapter_path = args.adapter_path.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite adapter evaluation: {output_path}"
        )
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"orientation adapter not found: {adapter_path}")
    adapter_weights_path = adapter_path / "adapter_model.safetensors"
    adapter_config_path = adapter_path / "adapter_config.json"
    for path, expected, description in (
        (adapter_weights_path, args.expected_adapter_sha256, "adapter weights"),
        (
            adapter_config_path,
            args.expected_adapter_config_sha256,
            "adapter config",
        ),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"{description} SHA-256 mismatch: expected={expected} found={actual}"
            )

    manifest = _load_manifest(manifest_path, args.expected_manifest_sha256)
    evaluation_frame = manifest[manifest["split"].isin(splits)].copy()
    if evaluation_frame.empty or set(evaluation_frame["split"]) != set(splits):
        raise ValueError("requested orientation evaluation splits are incomplete")
    model_provenance = _validate_model_snapshot(
        model_path,
        args.expected_model_repo_id,
        args.expected_model_revision,
        args.expected_model_architecture,
    )

    import torch
    from peft import PeftModel
    from transformers import (
        AutoProcessor,
        Idefics3ForConditionalGeneration,
        Qwen2VLForConditionalGeneration,
    )

    if torch.__version__ != "2.0.1+cu118":
        raise RuntimeError(
            "MLO orientation evaluation requires torch 2.0.1+cu118 in prima; "
            f"found {torch.__version__}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "MLO orientation evaluation requires exactly one visible GPU"
        )
    if torch.cuda.get_device_properties(0).total_memory < 100 * 1024**3:
        raise RuntimeError("MLO orientation evaluation requires a 100+ GiB GPU")

    processor_kwargs: dict[str, object] = {"local_files_only": True}
    if args.expected_model_architecture == "Qwen2VLForConditionalGeneration":
        processor_kwargs.update(min_pixels=896 * 896, max_pixels=896 * 896)
    processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
    model_classes = {
        "Idefics3ForConditionalGeneration": Idefics3ForConditionalGeneration,
        "Qwen2VLForConditionalGeneration": Qwen2VLForConditionalGeneration,
    }
    torch_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    base_model = (
        model_classes[args.expected_model_architecture]
        .from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        .to("cuda")
    )
    model = PeftModel.from_pretrained(
        base_model, adapter_path, is_trainable=False, local_files_only=True
    )
    model.config.use_cache = True
    predictions = _generate_predictions(
        model=model,
        processor=processor,
        frame=evaluation_frame,
        manifest_root=manifest_path.parent,
        max_new_tokens=args.max_new_tokens,
    )
    result = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "evaluated_splits": splits,
        "model_repo_id": model_provenance["repo_id"],
        "model_revision": model_provenance["revision"],
        "model_architecture": args.expected_model_architecture,
        "adapter_path": str(adapter_path),
        "adapter_sha256": sha256_file(adapter_weights_path),
        "adapter_config_sha256": sha256_file(adapter_config_path),
        "precision": args.precision,
        "runtime_versions": {
            **runtime_versions,
            "torch": torch.__version__,
            "python": os.sys.version.split()[0],
        },
        "scores": _score_by_split(predictions),
        "predictions": predictions,
    }
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(result), indent=2) + "\n")
    os.chmod(output_path, 0o600)
    print(f"MLO orientation adapter evaluation complete: output={output_path}")
    return result


def main() -> int:
    run_from_args(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
