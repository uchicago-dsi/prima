#!/usr/bin/env python3
"""Train and score a Qwen2-VL LoRA on synthetic MLO orientation pairs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
from typing import Any

import pandas as pd
from PIL import Image

from prima.mlo_orientation import (
    ORIENTATION_LABELS,
    ORIENTATION_PROMPT,
    parse_orientation_label,
    score_orientation_predictions,
)
from prima.view_few_shot import sha256_file
from prima.view_landmark_grid import resolve_relative_image

REQUIRED_RUNTIME = {
    "transformers": "4.46.3",
    "accelerate": "1.0.1",
    "peft": "0.13.2",
}
SPLITS = ("train", "validation", "challenge")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-model-revision", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def _validate_runtime() -> dict[str, str]:
    versions = {name: importlib.metadata.version(name) for name in REQUIRED_RUNTIME}
    mismatches = {
        name: {"expected": expected, "found": versions[name]}
        for name, expected in REQUIRED_RUNTIME.items()
        if versions[name] != expected
    }
    if mismatches:
        raise RuntimeError(f"MLO orientation training runtime mismatch: {mismatches}")
    return versions


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if args.lora_rank <= 0 or args.lora_alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("--lora-dropout must be in [0, 1)")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")


def _load_manifest(path: Path, expected_sha256: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"MLO orientation manifest not found: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "MLO orientation manifest SHA-256 mismatch: "
            f"expected={expected_sha256} found={actual_sha256}"
        )
    manifest = pd.read_parquet(path)
    required = {
        "sample_id",
        "source_view_id",
        "source_review_order",
        "laterality",
        "view",
        "split",
        "rotation_degrees_clockwise",
        "expected_label",
        "model_image_path",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"MLO orientation manifest missing columns: {missing}")
    if manifest.empty or manifest["sample_id"].duplicated().any():
        raise ValueError("MLO orientation manifest is empty or has duplicate samples")
    if set(manifest["split"]) != set(SPLITS):
        raise ValueError(
            "MLO orientation manifest must contain all three frozen splits"
        )
    if not manifest["view"].eq("MLO").all():
        raise ValueError("MLO orientation manifest contains a non-MLO view")
    if not manifest["expected_label"].isin(ORIENTATION_LABELS).all():
        raise ValueError("MLO orientation manifest contains invalid labels")
    if not manifest["rotation_degrees_clockwise"].isin({0, 180}).all():
        raise ValueError("MLO orientation manifest contains invalid rotations")
    source_sets = {
        split: set(group["source_view_id"])
        for split, group in manifest.groupby("split")
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if source_sets[left] & source_sets[right]:
                raise ValueError(
                    f"MLO orientation source leaks between {left} and {right}"
                )
    for (_, _), group in manifest.groupby(["split", "source_view_id"]):
        if set(group["rotation_degrees_clockwise"]) != {0, 180} or len(group) != 2:
            raise ValueError(
                "each MLO orientation source must have one exact rotation pair"
            )
        if set(group["expected_label"]) != set(ORIENTATION_LABELS):
            raise ValueError("each MLO orientation pair must contain opposite labels")
    root = path.parent
    for relative in manifest["model_image_path"]:
        resolve_relative_image(root, relative, description="model_image_path")
    return manifest.sort_values(
        ["split", "source_view_id", "rotation_degrees_clockwise"]
    )


def _validate_model_snapshot(
    model_path: Path, expected_revision: str
) -> dict[str, Any]:
    if not model_path.is_dir():
        raise FileNotFoundError(f"Qwen2-VL model snapshot not found: {model_path}")
    provenance_path = model_path / "prima_snapshot.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"model snapshot provenance not found: {provenance_path}"
        )
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("repo_id") != "Qwen/Qwen2-VL-2B-Instruct":
        raise RuntimeError("model snapshot is not Qwen/Qwen2-VL-2B-Instruct")
    if provenance.get("revision") != expected_revision:
        raise RuntimeError(
            "model revision mismatch: "
            f"expected={expected_revision} found={provenance.get('revision')}"
        )
    return provenance


class OrientationDataset:
    """Minimal dataframe-backed dataset for Transformers Trainer."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.records = frame.to_dict("records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.records[index]


def _messages(answer: str | None = None) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": ORIENTATION_PROMPT},
            ],
        }
    ]
    if answer is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        )
    return messages


class OrientationCollator:
    """Encode one image conversation and mask every non-answer token."""

    def __init__(self, processor: Any, manifest_root: Path) -> None:
        self.processor = processor
        self.manifest_root = manifest_root

    def __call__(self, features: list[dict[str, object]]) -> dict[str, Any]:
        if len(features) != 1:
            raise ValueError("MLO orientation training requires batch size one")
        record = features[0]
        image_path, _ = resolve_relative_image(
            self.manifest_root,
            record["model_image_path"],
            description="model_image_path",
        )
        with Image.open(image_path) as source:
            source.load()
            image = source.convert("RGB")
        prompt_text = self.processor.apply_chat_template(
            _messages(), tokenize=False, add_generation_prompt=True
        )
        full_text = self.processor.apply_chat_template(
            _messages(str(record["expected_label"])),
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_inputs = self.processor(
            text=[prompt_text], images=[image], return_tensors="pt"
        )
        encoded = self.processor(text=[full_text], images=[image], return_tensors="pt")
        prompt_length = int(prompt_inputs["input_ids"].shape[1])
        full_ids = encoded["input_ids"]
        if full_ids.shape[1] <= prompt_length:
            raise RuntimeError("assistant answer produced no supervised tokens")
        if not full_ids[:, :prompt_length].equal(prompt_inputs["input_ids"]):
            raise RuntimeError(
                "prompt is not an exact prefix of the supervised example"
            )
        labels = full_ids.clone()
        labels[:, :prompt_length] = -100
        encoded["labels"] = labels
        return dict(encoded)


def _generate_predictions(
    *,
    model: Any,
    processor: Any,
    frame: pd.DataFrame,
    manifest_root: Path,
    max_new_tokens: int,
) -> list[dict[str, object]]:
    import torch

    model.eval()
    records: list[dict[str, object]] = []
    prompt_text = processor.apply_chat_template(
        _messages(), tokenize=False, add_generation_prompt=True
    )
    for row in frame.to_dict("records"):
        image_path, _ = resolve_relative_image(
            manifest_root, row["model_image_path"], description="model_image_path"
        )
        with Image.open(image_path) as source:
            source.load()
            image = source.convert("RGB")
        inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
        inputs = {name: value.to(model.device) for name, value in inputs.items()}
        input_length = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        generated_text = processor.batch_decode(
            generated[:, input_length:], skip_special_tokens=True
        )[0].strip()
        records.append(
            {
                "sample_id": row["sample_id"],
                "source_view_id": row["source_view_id"],
                "source_review_order": int(row["source_review_order"]),
                "split": row["split"],
                "rotation_degrees_clockwise": int(row["rotation_degrees_clockwise"]),
                "expected_label": row["expected_label"],
                "generated_text": generated_text,
                "predicted_label": parse_orientation_label(generated_text),
            }
        )
    return records


def _score_by_split(predictions: list[dict[str, object]]) -> dict[str, object]:
    frame = pd.DataFrame.from_records(predictions)
    results: dict[str, object] = {}
    for split in sorted(frame["split"].unique()):
        subset = frame[frame["split"].eq(split)].copy()
        score = score_orientation_predictions(
            subset["expected_label"].tolist(), subset["predicted_label"].tolist()
        )
        pair_results = []
        for source_view_id, pair in subset.groupby("source_view_id"):
            pair_results.append(
                {
                    "source_view_id": source_view_id,
                    "exact": bool(
                        pair["expected_label"].eq(pair["predicted_label"]).all()
                    ),
                }
            )
        exact_pairs = sum(item["exact"] for item in pair_results)
        score["source_pairs"] = len(pair_results)
        score["exact_pairs"] = exact_pairs
        score["pair_accuracy"] = exact_pairs / len(pair_results)
        results[split] = score
    return results


def _gate(base: dict[str, object], adapted: dict[str, object]) -> dict[str, object]:
    validation = adapted["validation"]
    challenge = adapted["challenge"]
    checks = {
        "validation_inversion_sensitivity_at_least_0_875": (
            validation["inversion_sensitivity"] >= 0.875
        ),
        "validation_upright_specificity_at_least_0_875": (
            validation["upright_specificity"] >= 0.875
        ),
        "validation_exact_pairs_at_least_7_of_8": (
            validation["exact_pairs"] >= 7 and validation["source_pairs"] == 8
        ),
        "challenge_all_six_labels_exact": (
            challenge["exact"] == 6 and challenge["rows"] == 6
        ),
        "challenge_all_three_pairs_exact": (
            challenge["exact_pairs"] == 3 and challenge["source_pairs"] == 3
        ),
        "adapter_not_worse_than_base_validation": (
            validation["exact"] >= base["validation"]["exact"]
        ),
    }
    return {"checks": checks, "passes": all(checks.values())}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def run_from_args(args: argparse.Namespace) -> dict[str, object]:
    _validate_args(args)
    runtime_versions = _validate_runtime()
    manifest_path = args.manifest.resolve()
    model_path = args.model_path.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite LoRA output: {output_dir}")
    manifest = _load_manifest(manifest_path, args.expected_manifest_sha256)
    model_provenance = _validate_model_snapshot(
        model_path, args.expected_model_revision
    )

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoProcessor,
        Qwen2VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )

    if torch.__version__ != "2.0.1+cu118":
        raise RuntimeError(
            "MLO orientation training requires torch 2.0.1+cu118 in prima; "
            f"found {torch.__version__}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("MLO orientation training requires exactly one visible GPU")
    if torch.cuda.get_device_properties(0).total_memory < 100 * 1024**3:
        raise RuntimeError("MLO orientation training requires a 100+ GiB GPU")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output_dir.mkdir(mode=0o700, parents=True)
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        min_pixels=896 * 896,
        max_pixels=896 * 896,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda")
    evaluation_frame = manifest[manifest["split"].isin({"validation", "challenge"})]
    base_predictions = _generate_predictions(
        model=model,
        processor=processor,
        frame=evaluation_frame,
        manifest_root=manifest_path.parent,
        max_new_tokens=args.max_new_tokens,
    )
    base_scores = _score_by_split(base_predictions)

    model.config.use_cache = False
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    train_frame = manifest[manifest["split"].eq("train")]
    validation_frame = manifest[manifest["split"].eq("validation")]
    collator = OrientationCollator(processor, manifest_path.parent)
    training_arguments = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_strategy="steps",
        logging_steps=5,
        evaluation_strategy="epoch",
        save_strategy="no",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to=[],
        dataloader_num_workers=0,
        optim="adamw_torch",
        max_grad_norm=1.0,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=OrientationDataset(train_frame),
        eval_dataset=OrientationDataset(validation_frame),
        data_collator=collator,
        tokenizer=processor,
    )
    train_result = trainer.train()
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    processor.save_pretrained(adapter_dir)

    model.config.use_cache = True
    adapted_predictions = _generate_predictions(
        model=model,
        processor=processor,
        frame=manifest,
        manifest_root=manifest_path.parent,
        max_new_tokens=args.max_new_tokens,
    )
    adapted_scores = _score_by_split(adapted_predictions)
    gate = _gate(base_scores, adapted_scores)
    result = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "decision": (
            "whether a small open-weight VLM adapter learns exam-disjoint MLO "
            "orientation and fixes the frozen audit inversion challenge"
        ),
        "status": "passes_mechanism_gate" if gate["passes"] else "fails_mechanism_gate",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "model_path": str(model_path),
        "model_repo_id": model_provenance["repo_id"],
        "model_revision": model_provenance["revision"],
        "runtime_versions": {
            **runtime_versions,
            "torch": torch.__version__,
            "python": os.sys.version.split()[0],
        },
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "seed": args.seed,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": "all-linear including vision and language linear layers",
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "trainable_fraction": trainable_parameters / total_parameters,
            "train_metrics": _json_safe(train_result.metrics),
            "log_history": _json_safe(trainer.state.log_history),
        },
        "base_scores": base_scores,
        "adapted_scores": adapted_scores,
        "mechanism_gate": gate,
        "base_predictions": base_predictions,
        "adapted_predictions": adapted_predictions,
        "interpretation_scope": (
            "mechanism pilot only; cannot replace or prospectively validate the "
            "retained whole-exam QC baseline"
        ),
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(_json_safe(result), indent=2) + "\n")
    os.chmod(result_path, 0o600)
    print(
        f"MLO orientation LoRA complete: status={result['status']} result={result_path}"
    )
    return result


def main() -> int:
    run_from_args(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
