#!/usr/bin/env python3
"""Run a local multimodal model over cached QC montage PNGs and save suggestions."""

from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter
from collections import defaultdict
from contextlib import contextmanager
import fcntl
from importlib import metadata
import json
import logging
import os
import re
import signal
import socket
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from prima.auto_qc import (
    AUTO_QC_PROMPT_VERSION,
    load_auto_run,
    save_auto_run,
    utc_now_iso,
)
from prima.qc_state import (
    DEFAULT_ANNOTATION_TAGS,
    load_qc_state,
    normalize_annotation_tag_catalog,
    normalize_annotation_tags,
    qc_state_to_annotations_map,
)
from tqdm import tqdm

PNG_PREFIX = "COMBINED_four_views_"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_VLLM_MODEL_REGISTRY = REPO_ROOT / "qc" / "auto_qc_models.json"
DEFAULT_MODELS_DIR = Path("/gpfs/data/huo-lab/Image/annawoodard/models")

_SHUTDOWN_REQUESTED = False

TRANSFORMERS_FP8_KERNEL_REVISIONS = {
    "finegrained-fp8": "061130fedf845f320c56de4425f7404f6512c87e",
    "deep-gemm": "9590415046fa95a187af7ea03391d4782047170d",
}


def request_shutdown(signum: int, _frame: Any) -> None:
    """Record a scheduler/user shutdown request without interrupting CUDA work."""
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    logger.warning("received signal %s; stopping after the current exam", signum)


def install_shutdown_handlers() -> None:
    """Install lightweight handlers so opportunistic jobs can checkpoint cleanly."""
    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)


def shutdown_requested() -> bool:
    """Return whether the current process should stop after a safe checkpoint."""
    return _SHUTDOWN_REQUESTED


def pin_transformers_fp8_kernel_revisions() -> None:
    """Avoid Hub version lookups for cached FP8 kernels on compute nodes."""
    try:
        from transformers.integrations import hub_kernels
    except Exception as exc:  # pragma: no cover - depends on optional imports
        logger.debug("could not inspect Transformers hub kernel mapping: %s", exc)
        return

    mapping = getattr(hub_kernels, "_HUB_KERNEL_MAPPING", {})
    module_mapping = getattr(hub_kernels, "_KERNEL_MODULE_MAPPING", {})
    changed: list[str] = []
    for kernel_name, revision in TRANSFORMERS_FP8_KERNEL_REVISIONS.items():
        entry = mapping.get(kernel_name)
        if not isinstance(entry, dict):
            continue
        if entry.get("revision") == revision and "version" not in entry:
            continue
        entry["revision"] = revision
        entry.pop("version", None)
        module_mapping.pop(kernel_name, None)
        changed.append(f"{kernel_name}@{revision[:12]}")
    if changed:
        logger.info(
            "pinned Transformers FP8 hub kernel revisions for compute-node cache use: %s",
            ", ".join(changed),
        )


def patch_torch_cuda_get_device_properties_default(torch_module: Any) -> None:
    """Handle Transformers FP8 code that calls get_device_properties with no device."""
    original_get_device_properties = torch_module.cuda.get_device_properties
    if getattr(
        original_get_device_properties,
        "_prima_default_device_patch",
        False,
    ):
        return
    try:
        original_get_device_properties()
        return
    except TypeError as exc:
        if "missing 1 required positional argument" not in str(exc):
            raise

    def patched_get_device_properties(device: Any = None) -> Any:
        if device is None:
            device = torch_module.cuda.current_device()
        return original_get_device_properties(device)

    patched_get_device_properties._prima_default_device_patch = True
    torch_module.cuda.get_device_properties = patched_get_device_properties
    logger.info(
        "patched torch.cuda.get_device_properties default device for Transformers FP8 compatibility"
    )


def load_tag_catalog(tags_file: Path | None) -> list[str]:
    """Load the allowed annotation tag vocabulary for model suggestions."""
    if tags_file is None:
        return list(DEFAULT_ANNOTATION_TAGS)
    if not tags_file.exists():
        raise FileNotFoundError(f"annotation tags file not found: {tags_file}")
    with open(tags_file) as f:
        payload = json.load(f)
    return normalize_annotation_tag_catalog(payload)


def resolve_tags_file(
    *,
    requested_tags_file: Path | None,
    run_file: Path,
    qc_file: Path | None,
) -> Path | None:
    """Pick the tag catalog path with sane local defaults."""
    if requested_tags_file is not None:
        return requested_tags_file.resolve()

    candidate_paths: list[Path] = []
    if qc_file is not None:
        candidate_paths.append(qc_file.parent / "annotation_tags.json")
    candidate_paths.append(run_file.parent / "annotation_tags.json")

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate.resolve()
    return None


def load_completed_annotations(qc_file: Path | None) -> set[str]:
    """Load exam IDs that already have accepted annotations."""
    if qc_file is None or not qc_file.exists():
        return set()
    qc_state = load_qc_state(qc_file)
    return set(qc_state_to_annotations_map(qc_state))


def load_views_df(views_path: Path) -> pd.DataFrame:
    """Load and normalize the QC views parquet once."""
    views_df = pd.read_parquet(views_path)
    views_df = views_df.assign(
        exam_id=views_df["exam_id"].astype(str),
        patient_id=views_df["patient_id"].astype(str),
    )
    if "accession_number" not in views_df.columns:
        raise ValueError(f"views parquet is missing accession_number: {views_path}")
    views_df["accession_number"] = views_df["accession_number"].astype(str)
    return views_df


def expected_montage_path(
    export_dir: Path, patient_id: str, accession: str, exam_id: str
) -> Path:
    """Resolve the cached 4-view montage path for an exam."""
    return (
        export_dir
        / "success"
        / str(patient_id)
        / str(accession)
        / f"{PNG_PREFIX}{exam_id}.png"
    )


def build_exam_montage_index(
    views_df: pd.DataFrame,
    *,
    export_dir: Path,
) -> dict[str, dict[str, str]]:
    """Build exam_id -> montage metadata for cached QC montages."""
    index: dict[str, dict[str, str]] = {}
    for source_exam_id in views_df["exam_id"].drop_duplicates():
        exam_views = views_df[views_df["exam_id"] == source_exam_id]
        if exam_views.empty:
            continue
        row = exam_views.iloc[0]
        patient_id_str = str(row["patient_id"])
        accession_str = str(row["accession_number"])
        montage_path = expected_montage_path(
            export_dir=export_dir,
            patient_id=patient_id_str,
            accession=accession_str,
            exam_id=str(source_exam_id),
        )
        if not montage_path.exists():
            continue
        index[str(source_exam_id)] = {
            "exam_id": str(source_exam_id),
            "patient_id": patient_id_str,
            "accession_number": accession_str,
            "image_path": str(montage_path),
        }
    return index


def clean_exemplar_annotations(
    annotations: list[str],
    *,
    preserve_tags: set[str] | None = None,
) -> list[str]:
    """Repair obvious legacy default-tag contamination for few-shot exemplars only."""
    normalized = normalize_annotation_tags(annotations)
    default_tags = set(DEFAULT_ANNOTATION_TAGS)
    preserve_tags = preserve_tags or set()
    kept_default_tags = {tag for tag in normalized if tag in preserve_tags}
    extras = [tag for tag in normalized if tag not in default_tags]
    if extras or kept_default_tags:
        return [
            tag
            for tag in normalized
            if tag in kept_default_tags or tag not in default_tags
        ]
    if set(normalized) == default_tags:
        return []
    return normalized


def load_few_shot_exemplars(
    *,
    qc_file: Path | None,
    views_df: pd.DataFrame,
    export_dir: Path,
    preserve_tags: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load a reliable few-shot exemplar pool from accepted human GT."""
    if qc_file is None or not qc_file.exists():
        return []

    qc_state = load_qc_state(qc_file)
    annotations_map = qc_state_to_annotations_map(qc_state)
    montage_index = build_exam_montage_index(views_df, export_dir=export_dir)

    exemplars: list[dict[str, Any]] = []
    skipped_missing_montage = 0
    skipped_ambiguous_defaults = 0
    for exam_id, raw_annotations in annotations_map.items():
        cleaned_annotations = clean_exemplar_annotations(
            list(raw_annotations),
            preserve_tags=preserve_tags,
        )
        if not cleaned_annotations:
            skipped_ambiguous_defaults += 1
            continue
        montage_record = montage_index.get(str(exam_id))
        if montage_record is None:
            skipped_missing_montage += 1
            continue
        exemplars.append(
            {
                **montage_record,
                "annotations": cleaned_annotations,
            }
        )

    logger.info(
        "few-shot exemplar pool: %d usable exams from %s (skipped_missing_montage=%d skipped_ambiguous_defaults=%d)",
        len(exemplars),
        qc_file,
        skipped_missing_montage,
        skipped_ambiguous_defaults,
    )
    return exemplars


def select_few_shot_examples(
    *,
    exemplar_pool: list[dict[str, Any]],
    max_examples: int,
    exclude_exam_id: str | None = None,
) -> list[dict[str, Any]]:
    """Choose a small, tag-diverse exemplar set with rare tags prioritized."""
    if max_examples <= 0:
        return []

    candidates = [
        record
        for record in exemplar_pool
        if str(record["exam_id"]) != str(exclude_exam_id)
    ]
    if not candidates:
        return []

    tag_support = Counter(
        tag for record in candidates for tag in record.get("annotations", [])
    )
    selected: list[dict[str, Any]] = []
    covered_tags: set[str] = set()
    remaining = list(candidates)

    while remaining and len(selected) < max_examples:
        best_idx = None
        best_score: tuple[float, int, int, str] | None = None
        for idx, record in enumerate(remaining):
            annotations = list(record.get("annotations", []))
            new_tags = [tag for tag in annotations if tag not in covered_tags]
            if new_tags:
                rarity_score = sum(1.0 / tag_support[tag] for tag in new_tags)
            else:
                rarity_score = sum(0.25 / tag_support[tag] for tag in annotations)
            score = (
                rarity_score,
                len(new_tags),
                -len(annotations),
                str(record["exam_id"]),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        covered_tags.update(chosen.get("annotations", []))

    return selected


def select_probe_few_shot_examples(
    *,
    exemplar_pool: list[dict[str, Any]],
    max_examples: int,
    probe_tag: str,
    exclude_exam_id: str | None = None,
) -> list[dict[str, Any]]:
    """Choose exemplars for one target tag, prioritizing target positives."""
    if max_examples <= 0:
        return []

    candidates = [
        record
        for record in exemplar_pool
        if str(record["exam_id"]) != str(exclude_exam_id)
    ]
    target_positive = [
        record
        for record in candidates
        if probe_tag in set(record.get("annotations", []))
    ]
    selected = sorted(target_positive, key=lambda record: str(record["exam_id"]))[
        :max_examples
    ]
    if len(selected) >= max_examples:
        return selected

    selected_ids = {str(record["exam_id"]) for record in selected}
    filler_pool = [
        record for record in candidates if str(record["exam_id"]) not in selected_ids
    ]
    selected.extend(
        select_few_shot_examples(
            exemplar_pool=filler_pool,
            max_examples=max_examples - len(selected),
            exclude_exam_id=None,
        )
    )
    return selected


def load_exam_records(
    *,
    views_df: pd.DataFrame,
    export_dir: Path,
    exam_list_path: Path | None,
    exam_id: str | None,
    max_exams: int | None,
    skip_existing_annotations: bool,
    qc_file: Path | None,
) -> list[dict[str, str]]:
    """Build one cached-montage record per exam."""
    if exam_id is not None:
        views_df = views_df[views_df["exam_id"] == str(exam_id)]

    if exam_list_path is not None:
        with open(exam_list_path) as f:
            selected_exam_ids = {
                line.strip()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            }
        views_df = views_df[views_df["exam_id"].isin(selected_exam_ids)]

    existing_annotated_exams = (
        load_completed_annotations(qc_file) if skip_existing_annotations else set()
    )

    records: list[dict[str, str]] = []
    for source_exam_id in views_df["exam_id"].drop_duplicates():
        exam_views = views_df[views_df["exam_id"] == source_exam_id]
        if exam_views.empty:
            continue
        row = exam_views.iloc[0]
        patient_id_str = str(row["patient_id"])
        accession_str = str(row["accession_number"])
        montage_path = expected_montage_path(
            export_dir=export_dir,
            patient_id=patient_id_str,
            accession=accession_str,
            exam_id=str(source_exam_id),
        )
        if not montage_path.exists():
            logger.warning("missing cached montage; skipping one exam")
            continue
        if (
            skip_existing_annotations
            and str(source_exam_id) in existing_annotated_exams
        ):
            continue
        records.append(
            {
                "exam_id": str(source_exam_id),
                "patient_id": patient_id_str,
                "accession_number": accession_str,
                "image_path": str(montage_path),
            }
        )
        if max_exams is not None and len(records) >= max_exams:
            break

    return records


def build_inference_settings(
    *,
    args: argparse.Namespace,
    model_spec: Any = None,
) -> dict[str, Any]:
    """Capture every invocation setting that can change saved predictions."""
    settings: dict[str, Any] = {
        "few_shot_examples": int(args.few_shot_examples),
        "few_shot_qc_file": (
            str(args.few_shot_qc_file.resolve())
            if args.few_shot_qc_file
            else (str(args.qc_file.resolve()) if args.qc_file else None)
        )
        if args.few_shot_examples > 0
        else None,
        "max_new_tokens": int(args.max_new_tokens),
        "target_prompt_override": args.target_prompt_override.strip()
        if args.target_prompt_override
        else None,
        "text_only_prompt": args.text_only_prompt.strip()
        if args.text_only_prompt
        else None,
        "thinking_disabled": True
        if args.backend == "vllm"
        else bool(args.disable_thinking),
    }
    if args.backend == "vllm":
        if model_spec is None:
            raise ValueError("vLLM inference settings require a resolved model spec")
        settings.update(
            {
                "model_key": model_spec.key,
                "model_revision": model_spec.revision,
                "request_timeout_seconds": int(args.vllm_request_timeout_seconds),
                "runtime_versions": {
                    package: metadata.version(package)
                    for package in ("dsi-local-llms", "openai", "vllm")
                },
                "serve_environment": dict(model_spec.environment),
                "serve_extra_args": list(model_spec.extra_args),
                "structured_output": {
                    "tagger_json": "json_schema",
                    "binary_tag_probe": "choice",
                }.get(args.prompt_mode),
                "temperature": 0.0,
            }
        )
    return settings


def build_run_payload(
    *,
    run_id: str,
    model_label: str,
    args: argparse.Namespace,
    tag_catalog: list[str],
    model_spec: Any = None,
    created_at: str | None = None,
    exam_suggestions: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the normalized auto-QC run payload for this invocation."""
    payload: dict[str, Any] = {
        "run_id": run_id,
        "model": model_label,
        "backend": f"{args.backend}_local",
        "created_at": created_at or utc_now_iso(),
        "prompt_version": AUTO_QC_PROMPT_VERSION,
        "prompt_mode": args.prompt_mode,
        "prompt_variant": args.prompt_variant,
        "inference_settings": build_inference_settings(
            args=args,
            model_spec=model_spec,
        ),
        "tag_catalog": tag_catalog,
        "exam_suggestions": exam_suggestions or {},
    }
    if args.probe_tag:
        payload["probe_tag"] = args.probe_tag
    return payload


def validate_auto_run_compatible(
    *,
    existing: dict[str, Any],
    current: dict[str, Any],
    run_file: Path,
) -> None:
    """Fail early when a resume target belongs to a different run setup."""
    existing_suggestions = existing.get("exam_suggestions", {})
    if not existing_suggestions:
        return

    mismatches: list[str] = []
    for field in (
        "model",
        "backend",
        "prompt_version",
        "prompt_mode",
        "prompt_variant",
    ):
        if str(existing.get(field, "")) != str(current.get(field, "")):
            mismatches.append(
                f"{field}: existing={existing.get(field)!r} current={current.get(field)!r}"
            )
    if existing.get("probe_tag") and current.get("probe_tag"):
        if str(existing["probe_tag"]) != str(current["probe_tag"]):
            mismatches.append(
                f"probe_tag: existing={existing.get('probe_tag')!r} current={current.get('probe_tag')!r}"
            )
    if list(existing.get("tag_catalog", [])) != list(current.get("tag_catalog", [])):
        mismatches.append("tag_catalog differs")
    if existing.get("inference_settings", {}) != current.get("inference_settings", {}):
        mismatches.append("inference_settings differ")

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            f"refusing to resume incompatible auto-QC run file {run_file}: "
            f"{mismatch_text}. Use a new --run-file or --force-rescore."
        )


@contextmanager
def locked_auto_run(run_file: Path) -> Any:
    """Take an advisory lock around run-file load/merge/save operations."""
    lock_path = run_file.with_suffix(f"{run_file.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)


def save_run_progress(
    *,
    run_file: Path,
    payload: dict[str, Any],
    merge_existing: bool = True,
) -> dict[str, Any]:
    """Save progress, merging concurrent completed exams under a file lock."""
    with locked_auto_run(run_file):
        if merge_existing and run_file.exists():
            existing = load_auto_run(run_file)
            if existing:
                validate_auto_run_compatible(
                    existing=existing,
                    current=payload,
                    run_file=run_file,
                )
                merged_exam_suggestions = dict(existing.get("exam_suggestions", {}))
                merged_exam_suggestions.update(payload.get("exam_suggestions", {}))
                payload = {
                    **payload,
                    "run_id": existing.get("run_id") or payload.get("run_id"),
                    "created_at": existing.get("created_at")
                    or payload.get("created_at"),
                    "exam_suggestions": merged_exam_suggestions,
                }
                if not payload.get("probe_tag") and existing.get("probe_tag"):
                    payload["probe_tag"] = existing["probe_tag"]
        return save_auto_run(run_file, payload)


def load_or_initialize_run_payload(
    *,
    run_file: Path,
    model_label: str,
    args: argparse.Namespace,
    tag_catalog: list[str],
    model_spec: Any = None,
) -> dict[str, Any]:
    """Load a compatible run file or initialize a new empty one."""
    run_id = f"{utc_now_iso().replace(':', '').replace('+00:00', 'Z')}_{model_label}"
    current = build_run_payload(
        run_id=run_id,
        model_label=model_label,
        args=args,
        tag_catalog=tag_catalog,
        model_spec=model_spec,
    )
    if args.force_rescore or not run_file.exists():
        return save_run_progress(
            run_file=run_file,
            payload=current,
            merge_existing=False,
        )

    existing = load_auto_run(run_file, persist_normalized=True)
    if not existing:
        return save_run_progress(
            run_file=run_file,
            payload=current,
            merge_existing=False,
        )
    if not existing.get("exam_suggestions"):
        current["run_id"] = existing.get("run_id") or current["run_id"]
        current["created_at"] = existing.get("created_at") or current["created_at"]
        return save_run_progress(
            run_file=run_file,
            payload=current,
            merge_existing=False,
        )
    validate_auto_run_compatible(existing=existing, current=current, run_file=run_file)
    if current.get("probe_tag") and not existing.get("probe_tag"):
        existing["probe_tag"] = current["probe_tag"]
    return save_run_progress(
        run_file=run_file,
        payload=existing,
        merge_existing=False,
    )


def make_exam_suggestion_record(
    *,
    record: dict[str, str],
    result: dict[str, Any],
    model_label: str,
) -> dict[str, Any]:
    """Convert one annotator result to the saved run-record schema."""
    return {
        "image_path": str(record["image_path"]),
        "model": model_label,
        "few_shot_example_exam_ids": result["few_shot_example_exam_ids"],
        "suggestions": result["suggestions"],
        "prompt_mode": result["prompt_mode"],
        "prompt_variant": result["prompt_variant"],
        "debug_dump_file": result["debug_dump_file"],
    }


def score_exam_records(
    *,
    exam_records: list[dict[str, str]],
    annotator: "LocalVisionAnnotator",
    tag_catalog: list[str],
    model_label: str,
    run_file: Path,
    payload: dict[str, Any],
    force_rescore: bool,
    desc: str,
) -> tuple[dict[str, Any], dict[str, int | bool]]:
    """Score records with per-exam checkpointing and resume skips."""
    stats: dict[str, int | bool] = {
        "scored": 0,
        "skipped_existing": 0,
        "interrupted": False,
    }
    exam_suggestions = dict(payload.get("exam_suggestions", {}))
    for record in tqdm(exam_records, desc=desc):
        if shutdown_requested():
            stats["interrupted"] = True
            break
        exam_id = str(record["exam_id"])
        if not force_rescore and exam_id in exam_suggestions:
            stats["skipped_existing"] = int(stats["skipped_existing"]) + 1
            continue

        image_path = Path(record["image_path"])
        result = annotator.annotate(
            exam_id=exam_id,
            image_path=image_path,
            tag_catalog=tag_catalog,
        )
        exam_suggestions[exam_id] = make_exam_suggestion_record(
            record=record,
            result=result,
            model_label=model_label,
        )
        payload = {**payload, "exam_suggestions": exam_suggestions}
        saved = save_run_progress(run_file=run_file, payload=payload)
        payload = saved
        exam_suggestions = dict(saved.get("exam_suggestions", {}))
        stats["scored"] = int(stats["scored"]) + 1

    saved = save_run_progress(run_file=run_file, payload=payload)
    return saved, stats


def pilot_queue_dirs(queue_dir: Path) -> dict[str, Path]:
    """Return and create the standard pilot queue directories."""
    dirs = {name: queue_dir / name for name in ("pending", "running", "done", "failed")}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def read_exam_ids_file(path: Path) -> list[str]:
    """Read unique exam IDs from a plain text file."""
    exam_ids: list[str] = []
    seen: set[str] = set()
    with open(path) as handle:
        for line in handle:
            exam_id = line.strip()
            if not exam_id or exam_id.startswith("#") or exam_id in seen:
                continue
            seen.add(exam_id)
            exam_ids.append(exam_id)
    return exam_ids


def load_pilot_task_exam_ids(task_path: Path) -> list[str]:
    """Load exam IDs from a JSON or text pilot task."""
    if task_path.suffix.lower() != ".json":
        return read_exam_ids_file(task_path)

    with open(task_path) as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        raw_exam_ids = payload
    elif isinstance(payload, dict):
        if "exam_ids" in payload:
            raw_exam_ids = payload["exam_ids"]
        elif "exam_id" in payload:
            raw_exam_ids = [payload["exam_id"]]
        elif "exam_list" in payload:
            exam_list_path = Path(str(payload["exam_list"]))
            if not exam_list_path.is_absolute():
                exam_list_path = task_path.parent / exam_list_path
            return read_exam_ids_file(exam_list_path)
        else:
            raise ValueError(
                f"pilot task {task_path} must contain exam_ids, exam_id, or exam_list"
            )
    else:
        raise ValueError(f"unsupported pilot task payload in {task_path}")

    exam_ids: list[str] = []
    seen: set[str] = set()
    for raw_exam_id in raw_exam_ids:
        exam_id = str(raw_exam_id).strip()
        if exam_id and exam_id not in seen:
            seen.add(exam_id)
            exam_ids.append(exam_id)
    return exam_ids


def claim_next_pilot_task(queue_dirs: dict[str, Path]) -> Path | None:
    """Atomically claim one pending pilot task."""
    for task_path in sorted(queue_dirs["pending"].iterdir()):
        if not task_path.is_file() or task_path.name.startswith("."):
            continue
        claimed_path = queue_dirs["running"] / task_path.name
        try:
            task_path.replace(claimed_path)
            return claimed_path
        except FileNotFoundError:
            continue
        except OSError:
            logger.exception("failed to claim a pilot task")
            continue
    return None


def move_task_with_collision(task_path: Path, destination_dir: Path) -> Path:
    """Move a task to a terminal directory without overwriting old attempts."""
    destination = destination_dir / task_path.name
    if destination.exists():
        timestamp = utc_now_iso().replace(":", "").replace("+00:00", "Z")
        destination = (
            destination_dir / f"{task_path.stem}.{timestamp}{task_path.suffix}"
        )
    task_path.replace(destination)
    return destination


def write_pilot_status(
    *,
    task_path: Path,
    status: str,
    pilot_id: str,
    payload: dict[str, Any],
) -> None:
    """Write a small sidecar status for a pilot task transition."""
    status_path = task_path.with_suffix(f"{task_path.suffix}.status.json")
    status_payload = {
        "task_file": str(task_path),
        "status": status,
        "pilot_id": pilot_id,
        "updated_at": utc_now_iso(),
        **payload,
    }
    with open(status_path, "w") as handle:
        json.dump(status_payload, handle, indent=2)
        handle.write("\n")


def requeue_stale_running_tasks(
    *,
    queue_dirs: dict[str, Path],
    stale_seconds: int | None,
) -> int:
    """Return stale running tasks to pending so a restarted pilot can resume."""
    if stale_seconds is None or stale_seconds <= 0:
        return 0
    now = time.time()
    requeued = 0
    for task_path in sorted(queue_dirs["running"].iterdir()):
        if not task_path.is_file() or task_path.name.startswith("."):
            continue
        age_seconds = now - task_path.stat().st_mtime
        if age_seconds < stale_seconds:
            continue
        destination = queue_dirs["pending"] / task_path.name
        if destination.exists():
            timestamp = utc_now_iso().replace(":", "").replace("+00:00", "Z")
            destination = (
                queue_dirs["pending"]
                / f"{task_path.stem}.requeued_{timestamp}{task_path.suffix}"
            )
        task_path.replace(destination)
        requeued += 1
        logger.warning(
            "requeued one stale pilot task after %.0f seconds",
            age_seconds,
        )
    return requeued


def records_for_exam_ids(
    *,
    exam_ids: list[str],
    exam_index: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """Resolve pilot task exam IDs to cached-montage records."""
    records: list[dict[str, str]] = []
    missing: list[str] = []
    for exam_id in exam_ids:
        record = exam_index.get(str(exam_id))
        if record is None:
            missing.append(str(exam_id))
            continue
        records.append(record)
    if missing:
        logger.warning(
            "pilot task references %d exams without cached montages",
            len(missing),
        )
    return records


def default_pilot_id() -> str:
    """Build a stable-enough identifier for pilot logs and status sidecars."""
    job_id = os.environ.get("SLURM_JOB_ID", "no_slurm_job")
    raw = f"{socket.gethostname()}_{job_id}_{os.getpid()}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def run_pilot_loop(
    *,
    args: argparse.Namespace,
    views_df: pd.DataFrame,
    export_dir: Path,
    annotator: "LocalVisionAnnotator",
    tag_catalog: list[str],
    model_label: str,
    run_file: Path,
    payload: dict[str, Any],
) -> int:
    """Keep a loaded model alive and process file-queue tasks."""
    queue_dir = args.pilot_queue_dir.resolve()
    queue_dirs = pilot_queue_dirs(queue_dir)
    pilot_id = args.pilot_id or default_pilot_id()
    stop_file = args.pilot_stop_file.resolve() if args.pilot_stop_file else None
    poll_seconds = max(1, int(args.pilot_poll_seconds))
    idle_timeout_seconds = int(args.pilot_idle_timeout_seconds)
    max_tasks = args.pilot_max_tasks
    max_exams = args.pilot_max_exams
    exam_index = build_exam_montage_index(views_df, export_dir=export_dir)
    logger.info(
        "starting auto-QC pilot %s queue=%s indexed_montages=%d",
        pilot_id,
        queue_dir,
        len(exam_index),
    )

    tasks_completed = 0
    exams_scored = 0
    idle_since: float | None = None
    while True:
        if shutdown_requested():
            logger.warning("pilot %s stopping because shutdown was requested", pilot_id)
            break
        if stop_file is not None and stop_file.exists():
            logger.info(
                "pilot %s stopping because stop file exists: %s", pilot_id, stop_file
            )
            break
        if max_tasks is not None and tasks_completed >= max_tasks:
            logger.info("pilot %s reached --pilot-max-tasks=%d", pilot_id, max_tasks)
            break
        if max_exams is not None and exams_scored >= max_exams:
            logger.info("pilot %s reached --pilot-max-exams=%d", pilot_id, max_exams)
            break

        requeue_stale_running_tasks(
            queue_dirs=queue_dirs,
            stale_seconds=args.pilot_reclaim_stale_seconds,
        )
        task_path = claim_next_pilot_task(queue_dirs)
        if task_path is None:
            if idle_since is None:
                idle_since = time.monotonic()
                logger.info("pilot %s is idle; waiting for new tasks", pilot_id)
            idle_seconds = time.monotonic() - idle_since
            if idle_timeout_seconds > 0 and idle_seconds >= idle_timeout_seconds:
                logger.info(
                    "pilot %s exiting after %.0f idle seconds",
                    pilot_id,
                    idle_seconds,
                )
                break
            time.sleep(poll_seconds)
            continue

        idle_since = None
        task_started_at = utc_now_iso()
        try:
            task_exam_ids = load_pilot_task_exam_ids(task_path)
            task_records = records_for_exam_ids(
                exam_ids=task_exam_ids,
                exam_index=exam_index,
            )
            if max_exams is not None:
                remaining_exam_budget = max(0, max_exams - exams_scored)
                task_records = task_records[:remaining_exam_budget]
            logger.info(
                "pilot %s claimed a task with %d requested exams, %d cached records",
                pilot_id,
                len(task_exam_ids),
                len(task_records),
            )
            task_path.touch()
            payload, stats = score_exam_records(
                exam_records=task_records,
                annotator=annotator,
                tag_catalog=tag_catalog,
                model_label=model_label,
                run_file=run_file,
                payload=payload,
                force_rescore=args.force_rescore,
                desc=f"pilot:{task_path.stem}",
            )
            task_path.touch()
            exams_scored += int(stats["scored"])
            if stats["interrupted"] or shutdown_requested():
                requeued_path = move_task_with_collision(
                    task_path, queue_dirs["pending"]
                )
                write_pilot_status(
                    task_path=requeued_path,
                    status="requeued_after_shutdown",
                    pilot_id=pilot_id,
                    payload={
                        "started_at": task_started_at,
                        "stats": stats,
                        "run_file": str(run_file),
                    },
                )
                break

            completed_path = move_task_with_collision(task_path, queue_dirs["done"])
            tasks_completed += 1
            write_pilot_status(
                task_path=completed_path,
                status="done",
                pilot_id=pilot_id,
                payload={
                    "started_at": task_started_at,
                    "stats": stats,
                    "run_file": str(run_file),
                },
            )
        except Exception as exc:
            logger.exception("pilot %s failed a task", pilot_id)
            failed_path = move_task_with_collision(task_path, queue_dirs["failed"])
            write_pilot_status(
                task_path=failed_path,
                status="failed",
                pilot_id=pilot_id,
                payload={
                    "started_at": task_started_at,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "run_file": str(run_file),
                },
            )
            if not args.pilot_continue_on_task_error:
                return 1

    saved = save_run_progress(run_file=run_file, payload=payload)
    suggested_exam_count = sum(
        1 for record in saved["exam_suggestions"].values() if record["suggestions"]
    )
    total_suggestions = sum(
        len(record["suggestions"]) for record in saved["exam_suggestions"].values()
    )
    print(f"pilot wrote auto-QC run to {run_file}")
    print(f"  tasks completed this process: {tasks_completed:,}")
    print(f"  exams scored this process: {exams_scored:,}")
    print(f"  total exams in run file: {len(saved['exam_suggestions']):,}")
    print(f"  exams with >=1 suggestion: {suggested_exam_count:,}")
    print(f"  total suggested tags: {total_suggestions:,}")
    return 0


def build_system_prompt(prompt_mode: str, input_level: str = "exam") -> str:
    """System prompt for the multimodal QC model."""
    if input_level == "exam":
        image_context = (
            "You are given cached four-view mammography montages for one labeled "
            "reference set and one target exam. Only output tags that are visually "
            "evident in the target montage itself."
        )
    elif input_level == "view":
        image_context = (
            "You are given individual mammography views. Only output tags that are "
            "visually evident in the single target view itself."
        )
    else:
        raise ValueError(f"unsupported input level: {input_level}")
    base = (
        "You are a strict mammography QC tagger. "
        f"{image_context} "
        "Use labeled references only as visual examples of the accepted QC tags. "
        "Do not mention patient identity or transcribe visible identifiers."
    )
    mode_rules = {
        "tagger_json": " Return only the requested JSON object.",
        "binary_tag_probe": " Return only yes or no.",
        "marker_classifier": " Follow the requested labeled-line response format exactly.",
        "what_is_this": " Answer in concise plain language without JSON.",
    }
    try:
        return base + mode_rules[prompt_mode]
    except KeyError as exc:
        raise ValueError(f"unsupported prompt mode: {prompt_mode}") from exc


def build_debug_describe_prompt(
    *,
    few_shot_examples: list[dict[str, Any]],
    input_level: str = "exam",
) -> str:
    """Simple freeform debug prompt to test whether the model sees the montage at all."""
    example_text = ""
    if few_shot_examples:
        example_text = (
            f"You are also given {len(few_shot_examples)} labeled reference examples before the target exam. "
            "Use them only as loose visual context.\n\n"
        )
    target = "target exam" if input_level == "exam" else "target view"
    return (
        f"{example_text}"
        f"This is the {target}.\n"
        "What is this image? Identify the modality and layout first.\n"
        "Then describe what is visually present in plain language, including any obvious markers, clips, implants, truncation, inversion, low contrast, lines, calcifications, or other unusual findings.\n"
        "Be concrete and visual. Do not output JSON."
    )


def build_binary_probe_prompt(
    *,
    probe_tag: str,
    few_shot_examples: list[dict[str, Any]],
    input_level: str = "exam",
) -> str:
    """Binary yes/no prompt for one QC tag."""
    example_text = ""
    if few_shot_examples:
        example_text = (
            f"You are also given {len(few_shot_examples)} labeled reference examples before the target exam. "
            "Use them only as loose visual context.\n\n"
        )
    target = "exam" if input_level == "exam" else "view"
    image_type = "mammography montage" if input_level == "exam" else "mammogram"
    return (
        f"{example_text}"
        f"Target {target} only: is the QC tag '{probe_tag}' visually present in this {image_type}?\n"
        "Answer only yes or no.\n"
    )


def build_marker_classifier_prompt(
    *,
    probe_tag: str,
    few_shot_examples: list[dict[str, Any]],
    prompt_variant: str,
    input_level: str = "exam",
) -> str:
    """Short classifier prompt that asks for evidence plus an explicit answer line."""
    example_text = ""
    if few_shot_examples:
        example_text = (
            f"You are also given {len(few_shot_examples)} labeled reference examples before the target exam. "
            "Use them only as loose visual context.\n\n"
        )
    target_description = (
        "a skin marker or BB marker"
        if probe_tag.strip().lower() == "bb"
        else f"the QC tag '{probe_tag}'"
    )
    target = "exam" if input_level == "exam" else "view"
    image_type = "mammography montage" if input_level == "exam" else "mammogram"
    if prompt_variant == "confidence_specificity":
        return (
            f"{example_text}"
            f"Target {target} only: decide whether {target_description} is visually present anywhere in this {image_type}.\n"
            "Use direct visible evidence of the named target only. Do not infer it "
            "from text labels, metadata, patient age, or unrelated findings. "
            "Answer NO when the named target is not visibly present.\n"
            "Use CONFIDENCE: high only when the visual evidence is unmistakable; use medium or low for borderline appearances.\n"
            "Answer in exactly four lines and nothing else:\n"
            "EVIDENCE: <one short visual phrase, or none>\n"
            "ANSWER: YES or ANSWER: NO\n"
            "CONFIDENCE: high, medium, or low\n"
            "REVIEW: YES or REVIEW: NO\n"
        )
    return (
        f"{example_text}"
        f"Target {target} only: decide whether {target_description} is visually present anywhere in this {image_type}.\n"
        "Answer in exactly two lines and nothing else:\n"
        "EVIDENCE: <one short visual phrase, or none>\n"
        "ANSWER: YES or ANSWER: NO\n"
    )


def build_binary_probe_choice_variants(processor: Any) -> list[dict[str, Any]]:
    """Tokenize a small forced-choice vocabulary for binary probe decoding."""
    tokenizer = getattr(processor, "tokenizer", processor)
    raw_variants = [
        ("yes", True),
        ("Yes", True),
        (" yes", True),
        (" Yes", True),
        ("no", False),
        ("No", False),
        (" no", False),
        (" No", False),
    ]
    variants: list[dict[str, Any]] = []
    seen_token_sequences: set[tuple[int, ...]] = set()
    for text, present in raw_variants:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            raise RuntimeError(
                f"binary probe choice {text!r} tokenized to an empty sequence"
            )
        token_key = tuple(int(token_id) for token_id in token_ids)
        if token_key in seen_token_sequences:
            continue
        seen_token_sequences.add(token_key)
        variants.append(
            {
                "text": text,
                "present": present,
                "token_ids": list(token_key),
            }
        )
    return variants


def build_binary_probe_prefix_allowed_tokens_fn(
    *,
    prompt_length: int,
    choice_variants: list[dict[str, Any]],
    eos_token_ids: list[int],
) -> Any:
    """Constrain decoding to an exact binary-probe choice string plus EOS."""
    choice_token_sequences = [
        [int(token_id) for token_id in variant["token_ids"]]
        for variant in choice_variants
    ]
    first_choice_tokens = sorted({token_ids[0] for token_ids in choice_token_sequences})
    if not first_choice_tokens:
        raise RuntimeError("binary probe decoding has no candidate first tokens")

    def _prefix_allowed_tokens_fn(batch_id: int, input_ids: Any) -> list[int]:
        del batch_id
        generated_prefix = [
            int(token_id) for token_id in input_ids[prompt_length:].tolist()
        ]
        allowed_tokens: set[int] = set()
        matched_complete_choice = False
        for token_ids in choice_token_sequences:
            if len(generated_prefix) > len(token_ids):
                continue
            if token_ids[: len(generated_prefix)] != generated_prefix:
                continue
            if len(generated_prefix) == len(token_ids):
                matched_complete_choice = True
            else:
                allowed_tokens.add(int(token_ids[len(generated_prefix)]))
        if matched_complete_choice:
            return list(eos_token_ids)
        if allowed_tokens:
            return sorted(allowed_tokens)
        return first_choice_tokens

    return _prefix_allowed_tokens_fn


PROMPT_VARIANTS = (
    "baseline",
    "recall_tilted",
    "confidence_specificity",
)


def build_recall_tilted_rule(probe_tag: str | None = None) -> str:
    """Instruction for generic high-recall target triage."""
    target_text = f"the target '{probe_tag}'" if probe_tag else "an allowed QC target"
    return (
        f"- High-recall target triage: mark {target_text} present whenever direct visual evidence is consistent with it, even if subtle.\n"
        "- Do not use unrelated findings, text labels, metadata, or patient characteristics as evidence.\n"
        f"- If direct visual evidence is ambiguous but consistent with {target_text}, prefer marking it present and explain the visible evidence briefly.\n"
    )


def build_user_prompt(
    tag_catalog: list[str],
    *,
    few_shot_examples: list[dict[str, Any]],
    prompt_variant: str,
) -> str:
    """User prompt for constrained multi-label prediction."""
    tags_text = "\n".join(f"- {tag}" for tag in tag_catalog)
    recall_rule = (
        build_recall_tilted_rule() if prompt_variant == "recall_tilted" else ""
    )
    example_text = ""
    if few_shot_examples:
        example_text = (
            f"You are also given {len(few_shot_examples)} labeled reference examples before the target exam. "
            "Study how the accepted QC tags correspond to the montage appearance, but do not copy labels unless they are visually present in the target.\n\n"
        )
    return (
        f"{example_text}"
        "Review the final target four-view mammography QC montage and identify which tags from the allowed list are present.\n\n"
        "Allowed tags:\n"
        f"{tags_text}\n\n"
        "Rules:\n"
        "- Only use tags from the allowed list.\n"
        "- If none are clearly present, return an empty suggestions list.\n"
        "- Use the labeled references only as visual guidance.\n"
        "- Keep rationale short and visual.\n"
        f"{recall_rule}"
        "- Return valid JSON only with this schema:\n"
        '{"suggestions":[{"tag":"<allowed tag>","score":0.0,"rationale":"short visual reason"}]}\n'
    )


def build_few_shot_assistant_payload(
    annotations: list[str],
    *,
    prompt_mode: str,
    probe_tag: str | None,
    evidence: str | None = None,
) -> str:
    """Render a human-labeled exemplar in the active target output format."""
    evidence = evidence.strip() if evidence else "accepted human QC label"
    if prompt_mode == "tagger_json":
        payload = {
            "suggestions": [
                {
                    "tag": tag,
                    "score": 1.0,
                    "rationale": evidence,
                }
                for tag in annotations
            ]
        }
        return json.dumps(payload, separators=(",", ":"))
    if prompt_mode in {"binary_tag_probe", "marker_classifier"}:
        if not probe_tag:
            raise ValueError(f"{prompt_mode} few-shot output requires probe_tag")
        present = probe_tag in annotations
        if prompt_mode == "binary_tag_probe":
            return "yes" if present else "no"
        return (
            f"EVIDENCE: {evidence}\n"
            f"ANSWER: {'YES' if present else 'NO'}\n"
            "CONFIDENCE: high\n"
            "REVIEW: NO"
        )
    if prompt_mode == "what_is_this":
        label_text = ", ".join(annotations) if annotations else "none"
        return f"Human-reviewed QC labels: {label_text}."
    raise ValueError(f"unsupported prompt mode: {prompt_mode}")


def build_few_shot_user_prompt(
    *,
    exemplar_index: int,
    reference_type: str,
    target_prompt_text: str,
) -> str:
    """Apply the target task itself to a labeled visual reference."""
    return (
        f"Labeled reference example {exemplar_index}. Apply the same QC target "
        f"definition and decision rules below to this {reference_type}. The "
        "accepted answer follows.\n\n"
        f"{target_prompt_text}"
    )


def build_target_prompt_text(
    *,
    prompt_mode: str,
    tag_catalog: list[str],
    few_shot_examples: list[dict[str, Any]],
    probe_tag: str | None,
    prompt_variant: str,
    input_level: str = "exam",
) -> str:
    """Build the target prompt text for the chosen debug or tagging mode."""
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"unsupported prompt variant: {prompt_variant}")
    if prompt_mode == "tagger_json":
        return build_user_prompt(
            tag_catalog,
            few_shot_examples=few_shot_examples,
            prompt_variant=prompt_variant,
        )
    if prompt_mode == "binary_tag_probe":
        if not probe_tag:
            raise ValueError("binary_tag_probe requires a probe_tag")
        return build_binary_probe_prompt(
            probe_tag=probe_tag,
            few_shot_examples=few_shot_examples,
            input_level=input_level,
        )
    if prompt_mode == "what_is_this":
        return build_debug_describe_prompt(
            few_shot_examples=few_shot_examples,
            input_level=input_level,
        )
    if prompt_mode == "marker_classifier":
        if not probe_tag:
            raise ValueError("marker_classifier requires a probe_tag")
        prompt = build_marker_classifier_prompt(
            probe_tag=probe_tag,
            few_shot_examples=few_shot_examples,
            prompt_variant=prompt_variant,
            input_level=input_level,
        )
        if prompt_variant == "recall_tilted":
            prompt += "\nAdditional decision rule:\n"
            prompt += build_recall_tilted_rule(probe_tag)
        return prompt
    raise ValueError(f"unsupported prompt mode: {prompt_mode}")


def extract_json_payload(text: str) -> Any:
    """Extract the first JSON object or array from a model response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    candidates = [stripped]
    match = re.search(r"(\{.*\}|\[.*\])", stripped, flags=re.DOTALL)
    if match:
        candidates.append(match.group(1))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        try:
            payload = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            continue
        if isinstance(payload, (dict, list)):
            return payload
    raise ValueError("model response did not contain valid JSON")


def normalize_model_suggestions(
    payload: Any, allowed_tags: set[str]
) -> list[dict[str, Any]]:
    """Normalize a model JSON payload into the run-file suggestion schema."""
    if isinstance(payload, list):
        raw_suggestions = payload
    elif isinstance(payload, dict):
        raw_suggestions = payload.get("suggestions", [])
    else:
        raw_suggestions = []

    normalized: list[dict[str, Any]] = []
    seen_tags: set[str] = set()
    for entry in raw_suggestions:
        if isinstance(entry, str):
            tag = entry.strip()
            score = None
            rationale = None
        elif isinstance(entry, dict):
            tag = str(entry.get("tag", "")).strip()
            score = entry.get("score")
            rationale = str(entry.get("rationale", "")).strip() or None
        else:
            continue
        if not tag or tag not in allowed_tags or tag in seen_tags:
            continue
        suggestion: dict[str, Any] = {"tag": tag}
        if score is not None:
            try:
                suggestion["score"] = float(score)
            except (TypeError, ValueError):
                pass
        if rationale:
            suggestion["rationale"] = rationale
        seen_tags.add(tag)
        normalized.append(suggestion)
    return normalized


def normalize_binary_probe_response(
    payload: Any,
    *,
    probe_tag: str,
    allowed_tags: set[str],
) -> list[dict[str, Any]]:
    """Convert a binary probe payload into the normal suggestion schema."""
    if probe_tag not in allowed_tags:
        raise ValueError(f"probe tag is not in allowed tag catalog: {probe_tag}")
    if not isinstance(payload, dict):
        return []

    present = payload.get("present")
    if not isinstance(present, bool) or not present:
        return []

    suggestion: dict[str, Any] = {"tag": probe_tag}
    score = payload.get("score")
    if score is not None:
        try:
            suggestion["score"] = float(score)
        except (TypeError, ValueError):
            pass
    confidence = str(payload.get("confidence", "")).strip().lower()
    if confidence in {"high", "medium", "low"}:
        suggestion["confidence"] = confidence
        if "score" not in suggestion:
            suggestion["score"] = {
                "high": 0.95,
                "medium": 0.66,
                "low": 0.33,
            }[confidence]
    review = payload.get("review")
    if isinstance(review, bool):
        suggestion["review"] = review
    rationale = str(payload.get("rationale", "")).strip()
    if rationale:
        suggestion["rationale"] = rationale
    return [suggestion]


def coerce_binary_probe_payload(text: str) -> dict[str, Any] | None:
    """Best-effort parse for binary probe outputs that are not strict JSON."""
    stripped = text.strip()
    if not stripped:
        return None

    lowered = stripped.lower()
    present_match = re.search(r"\bpresent\b\s*[:=]\s*(true|false|yes|no)\b", lowered)
    if present_match is not None:
        present = present_match.group(1) in {"true", "yes"}
    elif re.search(r"\b(yes|true)\b", lowered) and not re.search(
        r"\b(no|false)\b", lowered
    ):
        present = True
    elif re.search(r"\b(no|false)\b", lowered) and not re.search(
        r"\b(yes|true)\b", lowered
    ):
        present = False
    else:
        return None

    payload: dict[str, Any] = {"present": present}
    score_match = re.search(r"\bscore\b\s*[:=]\s*(-?\d+(?:\.\d+)?)", stripped, re.I)
    if score_match is not None:
        try:
            payload["score"] = float(score_match.group(1))
        except ValueError:
            pass
    rationale_match = re.search(r"\brationale\b\s*[:=]\s*(.+)", stripped, re.I | re.S)
    if rationale_match is not None:
        rationale = rationale_match.group(1).strip()
        if rationale:
            payload["rationale"] = rationale
    return payload


def coerce_marker_classifier_payload(text: str) -> dict[str, Any] | None:
    """Parse a short evidence-plus-answer classifier response."""
    stripped = text.strip()
    if not stripped:
        return None

    answer_match = re.search(r"\bANSWER\s*:\s*(YES|NO)\b", stripped, re.I)
    if answer_match is not None:
        present = answer_match.group(1).upper() == "YES"
    else:
        final_match = re.search(r"\bFINAL\s*:\s*(YES|NO)\b", stripped, re.I)
        if final_match is not None:
            present = final_match.group(1).upper() == "YES"
        elif re.fullmatch(r"(?i)\s*(yes|no)\s*", stripped):
            present = stripped.strip().upper() == "YES"
        else:
            return None

    payload: dict[str, Any] = {"present": present}
    evidence_match = re.search(r"\bEVIDENCE\s*:\s*(.+)", stripped, re.I)
    if evidence_match is not None:
        rationale = evidence_match.group(1).strip()
        if rationale and rationale.lower() != "none":
            payload["rationale"] = rationale
    confidence_match = re.search(
        r"\bCONFIDENCE\s*:\s*(high|medium|low)\b", stripped, re.I
    )
    if confidence_match is not None:
        payload["confidence"] = confidence_match.group(1).lower()
    review_match = re.search(r"\bREVIEW\s*:\s*(YES|NO)\b", stripped, re.I)
    if review_match is not None:
        payload["review"] = review_match.group(1).upper() == "YES"
    return payload


def infer_model_label(model_path: Path) -> str:
    """Use the local checkpoint directory name as the run model label."""
    return model_path.name


def image_data_url(image_path: Path) -> str:
    """Encode a local QC image for the loopback OpenAI-compatible API."""
    media_types = {
        ".jpeg": "image/jpeg",
        ".jpg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    if not image_path.is_file():
        raise FileNotFoundError("QC montage file is missing")
    try:
        media_type = media_types[image_path.suffix.lower()]
    except KeyError as exc:
        raise ValueError(
            f"unsupported QC image type for vLLM: {image_path.suffix!r}"
        ) from exc
    try:
        image_bytes = image_path.read_bytes()
    except OSError:
        raise RuntimeError("failed to read QC montage") from None
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def sanitized_vllm_request_error(exc: Exception) -> RuntimeError:
    """Remove request bodies and image data from client-facing API failures."""
    status_code = getattr(exc, "status_code", None)
    status_text = f", status={int(status_code)}" if isinstance(status_code, int) else ""
    return RuntimeError(
        f"vLLM request failed ({type(exc).__name__}{status_text}); "
        "inspect the request-logging-disabled server log"
    )


def tagger_json_response_format(tag_catalog: list[str]) -> dict[str, Any]:
    """Return the strict structured-output schema for multi-label tagging."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "prima_auto_qc_suggestions",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "suggestions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tag": {"type": "string", "enum": tag_catalog},
                                "score": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                                "rationale": {"type": "string"},
                            },
                            "required": ["tag", "score", "rationale"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["suggestions"],
                "additionalProperties": False,
            },
        },
    }


def build_max_memory_map(
    *,
    num_visible_gpus: int,
    max_memory_per_gpu: str | None,
    cpu_max_memory: str | None,
) -> dict[int | str, str] | None:
    """Build a Hugging Face max_memory map for sharded loading."""
    if max_memory_per_gpu is None and cpu_max_memory is None:
        return None
    memory_map: dict[int | str, str] = {}
    if max_memory_per_gpu is not None:
        for gpu_idx in range(num_visible_gpus):
            memory_map[gpu_idx] = max_memory_per_gpu
    if cpu_max_memory is not None:
        memory_map["cpu"] = cpu_max_memory
    return memory_map


def preflight_transformers_checkpoint(model_path: Path) -> None:
    """Fail fast if the installed Transformers build cannot parse the checkpoint."""
    import transformers
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=False)
    except ValueError as exc:
        raise RuntimeError(
            "installed transformers build cannot load this checkpoint config. "
            f"model_path={model_path} transformers_version={transformers.__version__}. "
            "Install a newer source build of transformers in the prima env."
        ) from exc

    logger.info(
        "checkpoint preflight ok: transformers=%s config=%s model_type=%s",
        transformers.__version__,
        type(config).__name__,
        getattr(config, "model_type", None),
    )


def checkpoint_quant_method(model_path: Path) -> str | None:
    """Read the checkpoint quantization method from config, if present."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=False)
    quantization_config = getattr(config, "quantization_config", None) or {}
    quant_method = quantization_config.get("quant_method")
    if quant_method is None:
        return None
    return str(quant_method).strip().lower()


def checkpoint_modules_to_not_convert(model_path: Path) -> set[str]:
    """Read AWQ modules that the checkpoint says must stay unquantized."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=False)
    quantization_config = getattr(config, "quantization_config", None) or {}
    raw_modules = quantization_config.get("modules_to_not_convert") or []
    return {
        str(module_name).strip()
        for module_name in raw_modules
        if str(module_name).strip()
    }


def checkpoint_weight_names(model_path: Path) -> set[str] | None:
    """Load checkpoint tensor names from the safetensors index, if present."""
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        return None
    with open(index_path) as f:
        payload = json.load(f)
    raw_weight_map = payload.get("weight_map")
    if not isinstance(raw_weight_map, dict):
        raise ValueError(
            f"checkpoint safetensors index is missing weight_map: {index_path}"
        )
    return {str(name) for name in raw_weight_map}


def load_prima_repair_manifest(model_path: Path) -> dict[str, Any] | None:
    """Load an optional PRIMA repair manifest that wraps a base checkpoint."""
    manifest_path = model_path / "repair_manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"repair manifest must be a JSON object: {manifest_path}")
    return payload


def resolve_model_source_and_repair_cache(
    model_path: Path,
) -> tuple[Path, Path | None, dict[str, Any] | None]:
    """Resolve a user-facing model path into its base checkpoint and repair cache."""
    manifest = load_prima_repair_manifest(model_path)
    if manifest is None:
        return model_path, None, None

    base_model_path = manifest.get("base_model_path")
    if not isinstance(base_model_path, str) or not base_model_path.strip():
        raise ValueError(
            f"repair manifest is missing a valid base_model_path: {model_path / 'repair_manifest.json'}"
        )
    repair_cache_dir = manifest.get("repair_cache_dir")
    resolved_cache_dir = None
    if repair_cache_dir is not None:
        if not isinstance(repair_cache_dir, str) or not repair_cache_dir.strip():
            raise ValueError(
                f"repair manifest has invalid repair_cache_dir: {model_path / 'repair_manifest.json'}"
            )
        resolved_cache_dir = Path(repair_cache_dir).expanduser()
        if not resolved_cache_dir.is_absolute():
            resolved_cache_dir = (model_path / resolved_cache_dir).resolve()
        else:
            resolved_cache_dir = resolved_cache_dir.resolve()

    resolved_base_model_path = Path(base_model_path).expanduser()
    if not resolved_base_model_path.is_absolute():
        resolved_base_model_path = (model_path / resolved_base_model_path).resolve()
    else:
        resolved_base_model_path = resolved_base_model_path.resolve()
    logger.info(
        "resolved repaired model path %s -> base checkpoint %s with repair cache %s",
        model_path,
        resolved_base_model_path,
        resolved_cache_dir,
    )
    return resolved_base_model_path, resolved_cache_dir, manifest


def load_model_config(model_path: Path, *, trust_remote_code: bool) -> Any:
    """Load a checkpoint config and patch known schema gaps for supported models."""
    from transformers import AutoConfig

    def _env_flag(name: str) -> bool:
        value = os.environ.get(name, "").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _normalize_modules_to_not_convert(raw_modules: list[Any]) -> list[str]:
        normalized_modules: list[str] = []
        seen_modules: set[str] = set()
        for module_name in raw_modules:
            name = str(module_name).strip()
            if not name:
                continue
            candidates = [name]
            if name.startswith("model."):
                candidates.append(name.removeprefix("model."))
            else:
                candidates.append(f"model.{name}")
            for candidate in candidates:
                if candidate in seen_modules:
                    continue
                seen_modules.add(candidate)
                normalized_modules.append(candidate)
        return normalized_modules

    config = AutoConfig.from_pretrained(
        str(model_path),
        trust_remote_code=trust_remote_code,
    )
    text_config = getattr(config, "text_config", None)
    quant_method = checkpoint_quant_method(model_path)
    if (
        getattr(config, "model_type", None) == "qwen3_5_moe"
        and text_config is not None
        and not hasattr(text_config, "intermediate_size")
        and hasattr(text_config, "moe_intermediate_size")
    ):
        text_config.intermediate_size = text_config.moe_intermediate_size
        logger.info(
            "patched %s text_config.intermediate_size=%s from moe_intermediate_size",
            model_path,
            text_config.intermediate_size,
        )
    if (
        getattr(config, "model_type", None) == "qwen3_5_moe"
        and quant_method == "fp8"
        and _env_flag("PRIMA_QWEN35_FP8_PATCH_TEXT_MAPPING")
    ):
        from transformers.conversion_mapping import (
            get_checkpoint_conversion_mapping,
            register_checkpoint_conversion_mapping,
        )

        qwen2_moe_mapping = get_checkpoint_conversion_mapping("qwen2_moe")
        if qwen2_moe_mapping is None:
            raise RuntimeError(
                "transformers build is missing qwen2_moe checkpoint conversion mapping; "
                "cannot repair Qwen3.5 FP8 expert weight loading"
            )
        register_checkpoint_conversion_mapping(
            "qwen3_5_moe_text",
            qwen2_moe_mapping,
            overwrite=True,
        )
        logger.info(
            "patched qwen3_5_moe_text checkpoint conversion mapping to reuse qwen2_moe expert fusion without stripping the model.language_model prefix"
        )
    if (
        getattr(config, "model_type", None) == "qwen3_5_moe"
        and quant_method == "fp8"
        and text_config is not None
        and _env_flag("PRIMA_QWEN35_FP8_REMAP_EXPERTS")
    ):
        from transformers.conversion_mapping import (
            get_checkpoint_conversion_mapping,
            register_checkpoint_conversion_mapping,
        )

        qwen2_moe_mapping = get_checkpoint_conversion_mapping("qwen2_moe")
        if qwen2_moe_mapping is None:
            raise RuntimeError(
                "transformers build is missing qwen2_moe checkpoint conversion mapping; "
                "cannot repair Qwen3.5 FP8 expert weight loading"
            )
        register_checkpoint_conversion_mapping(
            "qwen3_5_moe",
            qwen2_moe_mapping,
            overwrite=True,
        )
        if getattr(text_config, "model_type", None) != "qwen3_5_moe":
            original_model_type = getattr(text_config, "model_type", None)
            text_config.model_type = "qwen3_5_moe"
            logger.info(
                "patched %s text_config.model_type from %s to %s so full-model Qwen3.5 FP8 expert fusion stays under model.language_model.*",
                model_path,
                original_model_type,
                text_config.model_type,
            )
    quantization_config = getattr(config, "quantization_config", None)
    if isinstance(quantization_config, dict):
        raw_modules = quantization_config.get("modules_to_not_convert") or []
        if _env_flag("PRIMA_AWQ_SKIP_LM_HEAD"):
            raw_modules = [*raw_modules, "lm_head"]
        normalized_modules = _normalize_modules_to_not_convert(raw_modules)
        if normalized_modules and normalized_modules != list(raw_modules):
            quantization_config["modules_to_not_convert"] = normalized_modules
            logger.info(
                "expanded %s modules_to_not_convert to %s",
                model_path,
                normalized_modules,
            )
    return config


def patch_qwen35_fp8_eager_experts() -> None:
    """Patch HF FP8 eager experts to tolerate tensor-parallel sentinel expert IDs."""
    import functools
    import torch
    import torch.nn.functional as F
    from transformers.integrations import finegrained_fp8

    if getattr(finegrained_fp8, "_prima_qwen35_fp8_eager_patch", False):
        return

    original_forward = finegrained_fp8.FP8Experts.forward

    @functools.wraps(original_forward)
    def patched_forward(self, hidden_states, top_k_index, top_k_weights):
        if not hasattr(self, "_prima_nonfinite_detail"):
            self._prima_nonfinite_detail = None
        layer_idx = getattr(self, "_prima_layer_idx", None)
        sync_linear = os.environ.get(
            "PRIMA_QWEN35_FP8_SYNC_LINEAR", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        force_bf16_experts = bool(
            getattr(self, "_prima_force_bf16_experts", False)
        ) or os.environ.get(
            "PRIMA_QWEN35_FP8_FORCE_BF16_EXPERTS", ""
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        def _maybe_sync_cuda(tensor: Any) -> None:
            if not sync_linear:
                return
            device = getattr(tensor, "device", None)
            if device is None or getattr(device, "type", None) != "cuda":
                return
            torch.cuda.synchronize(device=device)

        def _maybe_record_nonfinite(
            stage: str, tensor: Any, expert_idx: int | None = None
        ) -> None:
            if getattr(self, "_prima_nonfinite_detail", None) is not None:
                return
            summary = summarize_nonfinite_tensor(tensor)
            if summary is None:
                return
            detail = {
                "stage": stage,
                "layer_idx": int(layer_idx) if layer_idx is not None else None,
                "expert_idx": int(expert_idx) if expert_idx is not None else None,
                **summary,
            }
            self._prima_nonfinite_detail = detail
            logger.warning(
                "Qwen3.5 FP8 experts first non-finite tensor at layer=%s stage=%s expert=%s: %s",
                layer_idx,
                stage,
                expert_idx,
                detail,
            )

        def _maybe_log_router_indices_once() -> None:
            if getattr(self, "_prima_router_detail_logged", False):
                return
            if top_k_index.numel() == 0:
                self._prima_router_detail_logged = True
                return
            try:
                top_k_index_cpu = top_k_index.detach().to(
                    device="cpu", dtype=torch.int64
                )
                invalid_mask_cpu = (top_k_index_cpu < 0) | (
                    top_k_index_cpu >= self.num_experts
                )
                invalid_count_cpu = int(invalid_mask_cpu.sum().item())
                if invalid_count_cpu > 0:
                    invalid_values = (
                        top_k_index_cpu[invalid_mask_cpu].reshape(-1)[:16].tolist()
                    )
                    logger.warning(
                        "Qwen3.5 FP8 eager experts at layer=%s received invalid router indices "
                        "(min=%d max=%d num_experts=%d invalid=%d sample=%s); "
                        "masking those routes before one_hot",
                        layer_idx,
                        int(top_k_index_cpu.min().item()),
                        int(top_k_index_cpu.max().item()),
                        self.num_experts,
                        invalid_count_cpu,
                        invalid_values,
                    )
                self._prima_router_detail_logged = True
            except Exception as exc:
                logger.warning(
                    "Qwen3.5 FP8 eager experts at layer=%s could not snapshot router "
                    "indices on CPU before masking: %s",
                    layer_idx,
                    exc,
                )
                self._prima_router_detail_logged = True

        def _expert_linear(
            input_tensor: Any,
            *,
            weight: Any,
            weight_scale_inv: Any,
            activation_scale: Any = None,
        ) -> Any:
            if not force_bf16_experts:
                return self.linear(
                    input_tensor,
                    weight,
                    weight_scale_inv,
                    activation_scale=activation_scale,
                )

            if weight.element_size() > 1:
                bf16_weight = weight.to(
                    device=input_tensor.device, dtype=torch.bfloat16
                )
            else:
                bf16_weight = dequantize_fp8_weight_blocks(
                    weight,
                    weight_scale_inv,
                    block_size=self.block_size,
                ).to(device=input_tensor.device)
            bf16_input = input_tensor.to(dtype=torch.bfloat16)
            output = F.linear(bf16_input, bf16_weight, None)
            return output.to(dtype=input_tensor.dtype)

        _maybe_record_nonfinite("hidden_states_input", hidden_states)
        _maybe_record_nonfinite("routing_weights_input", top_k_weights)
        if top_k_index.numel() > 0:
            valid_expert_mask = (top_k_index >= 0) & (top_k_index < self.num_experts)
        else:
            valid_expert_mask = None

        # Mirror the upstream eager path, but select routed tokens on CPU so
        # corrupt/sentinel expert ids are skipped before any CUDA indexing kernel.
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)
        with torch.no_grad():
            top_k_index_cpu = top_k_index.detach().to(device="cpu", dtype=torch.int64)
            valid_expert_mask_cpu = (top_k_index_cpu >= 0) & (
                top_k_index_cpu < self.num_experts
            )
            if valid_expert_mask is not None:
                _maybe_log_router_indices_once()
            if valid_expert_mask_cpu.any():
                expert_hit = torch.unique(
                    top_k_index_cpu[valid_expert_mask_cpu]
                ).tolist()
            else:
                expert_hit = []

        for expert_idx in expert_hit:
            expert_routes_cpu = valid_expert_mask_cpu & (top_k_index_cpu == expert_idx)
            token_idx_cpu, top_k_pos_cpu = torch.where(expert_routes_cpu)
            if (token_idx_cpu >= hidden_states.shape[0]).any() or (
                top_k_pos_cpu >= top_k_weights.shape[1]
            ).any():
                raise RuntimeError(
                    "invalid FP8 expert route indices: "
                    f"hidden_states={tuple(hidden_states.shape)} "
                    f"top_k_weights={tuple(top_k_weights.shape)} "
                    f"max_token={int(token_idx_cpu.max())} "
                    f"max_top_k_pos={int(top_k_pos_cpu.max())}"
                )
            token_idx = token_idx_cpu.to(
                device=hidden_states.device, dtype=torch.long, non_blocking=True
            )
            top_k_pos = top_k_pos_cpu.to(
                device=hidden_states.device, dtype=torch.long, non_blocking=True
            )
            current_state = hidden_states[token_idx]
            gate_up_act_scale = (
                self.gate_up_proj_activation_scale[expert_idx]
                if self.activation_scheme == "static"
                else None
            )
            proj_out = _expert_linear(
                current_state,
                weight=(
                    self.gate_up_proj[expert_idx]
                    if self.has_gate
                    else self.up_proj[expert_idx]
                ),
                weight_scale_inv=(
                    self.gate_up_proj_scale_inv[expert_idx]
                    if self.has_gate
                    else self.up_proj_scale_inv[expert_idx]
                ),
                activation_scale=gate_up_act_scale,
            )
            _maybe_sync_cuda(proj_out)
            _maybe_record_nonfinite("gate_up_linear", proj_out, int(expert_idx))
            proj_out = (
                self._apply_gate(proj_out) if self.has_gate else self.act_fn(proj_out)
            )
            _maybe_record_nonfinite("gate_activation", proj_out, int(expert_idx))
            down_act_scale = (
                self.down_proj_activation_scale[expert_idx]
                if self.activation_scheme == "static"
                else None
            )
            proj_out = _expert_linear(
                proj_out,
                weight=self.down_proj[expert_idx],
                weight_scale_inv=self.down_proj_scale_inv[expert_idx],
                activation_scale=down_act_scale,
            )
            _maybe_sync_cuda(proj_out)
            _maybe_record_nonfinite("down_linear", proj_out, int(expert_idx))
            routing_weights = top_k_weights[token_idx, top_k_pos, None]
            _maybe_record_nonfinite(
                "routing_weights_selected", routing_weights, int(expert_idx)
            )
            weighted_out = proj_out * routing_weights.to(proj_out.dtype)
            _maybe_record_nonfinite("weighted_output", weighted_out, int(expert_idx))
            final_hidden_states.index_add_(
                0, token_idx, weighted_out.to(final_hidden_states.dtype)
            )
            _maybe_sync_cuda(final_hidden_states)
        _maybe_record_nonfinite("accumulated_output", final_hidden_states)
        return final_hidden_states.to(hidden_states.dtype)

    finegrained_fp8.FP8Experts.forward = patched_forward
    finegrained_fp8._prima_qwen35_fp8_eager_patch = True


def prepare_awq_backend(model_path: Path) -> str:
    """Import a compatible AWQ backend eagerly before meta-device loading."""
    requested_backend = os.environ.get("PRIMA_AWQ_BACKEND", "").strip().lower()
    if requested_backend == "autoawq":
        try:
            import awq

            version = getattr(awq, "__version__", "unknown")
            logger.info(
                "using requested AWQ backend autoawq %s for %s",
                version,
                model_path,
            )
            return "autoawq"
        except Exception as awq_exc:
            raise RuntimeError(
                f"requested PRIMA_AWQ_BACKEND=autoawq but import failed with {type(awq_exc).__name__}: {awq_exc}"
            ) from awq_exc

    try:
        import gptqmodel

        version = getattr(gptqmodel, "__version__", "unknown")
        logger.info("using AWQ backend gptqmodel %s", version)
        return "gptqmodel"
    except Exception as gptq_exc:
        try:
            import awq
        except Exception as awq_exc:
            raise RuntimeError(
                "no compatible AWQ backend is importable in the active environment. "
                f"gptqmodel failed with {type(gptq_exc).__name__}: {gptq_exc}; "
                f"autoawq failed with {type(awq_exc).__name__}: {awq_exc}"
            ) from awq_exc

        version = getattr(awq, "__version__", "unknown")
        logger.info(
            "using fallback AWQ backend autoawq %s after gptqmodel import failed with %s: %s",
            version,
            type(gptq_exc).__name__,
            gptq_exc,
        )
        return "autoawq"


def move_inputs_to_model_device(
    inputs: dict[str, Any], model: Any, *, dtype: Any | None = None
) -> dict[str, Any]:
    """Move tensor inputs onto the model's first device and compute dtype."""
    model_device = getattr(model, "device", None)
    if model_device is None:
        return inputs
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            moved_value = value.to(model_device)
            if (
                dtype is not None
                and hasattr(moved_value, "is_floating_point")
                and moved_value.is_floating_point()
            ):
                moved_value = moved_value.to(dtype=dtype)
            moved[key] = moved_value
        else:
            moved[key] = value
    return moved


def iter_nested_tensors(value: Any, *, prefix: str) -> list[tuple[str, Any]]:
    """Collect nested tensors under tuples/lists/dicts for debug inspection."""
    items: list[tuple[str, Any]] = []
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        items.append((prefix, value))
        return items
    if isinstance(value, tuple | list):
        for idx, item in enumerate(value):
            items.extend(iter_nested_tensors(item, prefix=f"{prefix}[{idx}]"))
        return items
    if isinstance(value, dict):
        for key, item in value.items():
            items.extend(iter_nested_tensors(item, prefix=f"{prefix}.{key}"))
        return items
    return items


def summarize_nonfinite_tensor(tensor: Any) -> dict[str, Any] | None:
    """Return a compact summary when a floating tensor contains NaN/Inf."""
    if not (
        hasattr(tensor, "is_floating_point")
        and tensor.is_floating_point()
        and hasattr(tensor, "numel")
    ):
        return None
    if tensor.numel() <= 0:
        return None

    finite_mask = tensor.isfinite()
    if bool(finite_mask.all().item()):
        return None

    nan_count = int(tensor.isnan().sum().item())
    posinf_count = int(tensor.isposinf().sum().item())
    neginf_count = int(tensor.isneginf().sum().item())
    finite_count = int(finite_mask.sum().item())
    finite_min: float | None = None
    finite_max: float | None = None
    if finite_count > 0:
        finite_values = tensor[finite_mask].float()
        finite_min = float(finite_values.min().item())
        finite_max = float(finite_values.max().item())
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
        "finite_count": finite_count,
        "nan_count": nan_count,
        "posinf_count": posinf_count,
        "neginf_count": neginf_count,
        "finite_min": finite_min,
        "finite_max": finite_max,
    }


class NonFiniteActivationTracer:
    """Record the first traced module whose floating activations become non-finite."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.handles: list[Any] = []
        self.instrumented_modules: list[str] = []
        self.first_nonfinite: dict[str, Any] | None = None

    @staticmethod
    def should_trace_module(name: str) -> bool:
        if not name:
            return False
        if name in {
            "model.visual",
            "model.language_model",
            "model.language_model.norm",
            "lm_head",
        }:
            return True
        if re.fullmatch(r"model\.language_model\.layers\.\d+", name):
            return True
        return name.endswith(
            (
                ".input_layernorm",
                ".post_attention_layernorm",
                ".self_attn",
                ".linear_attn",
                ".mlp",
                ".mlp.gate",
                ".mlp.experts",
                ".mlp.shared_expert",
                ".mlp.shared_expert_gate",
            )
        )

    def start(self) -> NonFiniteActivationTracer:
        for name, module in self.model.named_modules():
            if not self.should_trace_module(name):
                continue
            self.instrumented_modules.append(name)
            self.handles.append(module.register_forward_hook(self._build_hook(name)))
        return self

    def close(self) -> None:
        while self.handles:
            self.handles.pop().remove()

    def _capture_nonfinite(
        self, *, module_name: str, module: Any, side: str, payload: Any
    ) -> None:
        if self.first_nonfinite is not None:
            return
        for tensor_path, tensor in iter_nested_tensors(payload, prefix=side):
            summary = summarize_nonfinite_tensor(tensor)
            if summary is None:
                continue
            self.first_nonfinite = {
                "module_name": module_name,
                "module_class": type(module).__name__,
                "tensor_path": tensor_path,
                "side": side,
                **summary,
            }
            return

    def _build_hook(self, module_name: str):
        def _hook(module: Any, args: tuple[Any, ...], output: Any) -> None:
            if self.first_nonfinite is not None:
                return
            self._capture_nonfinite(
                module_name=module_name,
                module=module,
                side="input",
                payload=args,
            )
            if self.first_nonfinite is not None:
                return
            self._capture_nonfinite(
                module_name=module_name,
                module=module,
                side="output",
                payload=output,
            )

        return _hook

    def summary(self) -> dict[str, Any]:
        return {
            "instrumented_module_count": len(self.instrumented_modules),
            "instrumented_modules_preview": self.instrumented_modules[:32],
            "first_nonfinite": self.first_nonfinite,
        }


def collect_fp8_expert_nonfinite_detail(model: Any) -> dict[str, Any] | None:
    """Read the first stage-level FP8 expert non-finite record, if one was captured."""
    for module_name, module in model.named_modules():
        detail = getattr(module, "_prima_nonfinite_detail", None)
        if detail is None:
            continue
        return {
            "module_name": module_name,
            "module_class": type(module).__name__,
            **detail,
        }
    return None


def _decode_token_piece(tokenizer: Any, token_id: int) -> str | None:
    """Best-effort decode for one token id without failing the main path."""
    if tokenizer is None:
        return None
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        try:
            return str(tokenizer.convert_ids_to_tokens(int(token_id)))
        except Exception:
            pass
    if hasattr(tokenizer, "decode"):
        try:
            return str(
                tokenizer.decode(
                    [int(token_id)],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
        except Exception:
            pass
    return None


def build_generation_debug_payload(
    *,
    processor: Any,
    model: Any,
    prompt_length: int,
    new_token_ids: Any,
    generation_scores: list[Any],
    quant_method: str | None,
    compute_dtype: Any,
    awq_backend: str | None,
    nonfinite_trace: dict[str, Any] | None,
) -> dict[str, Any]:
    """Collect compact generation diagnostics for debugging gibberish outputs."""
    tokenizer = getattr(processor, "tokenizer", None)
    generated_token_ids = [int(token_id) for token_id in new_token_ids[0].tolist()]
    generated_token_pieces = [
        _decode_token_piece(tokenizer, token_id)
        for token_id in generated_token_ids[:64]
    ]
    first_token_topk: list[dict[str, Any]] = []
    if generation_scores:
        first_score = generation_scores[0][0].detach().float().cpu()
        top_k = min(10, int(first_score.shape[-1]))
        top_values, top_indices = first_score.topk(top_k)
        for token_id, logit in zip(
            top_indices.tolist(), top_values.tolist(), strict=True
        ):
            first_token_topk.append(
                {
                    "token_id": int(token_id),
                    "token_piece": _decode_token_piece(tokenizer, int(token_id)),
                    "logit": float(logit),
                }
            )

    lm_head = getattr(model, "lm_head", None)
    lm_head_weight = getattr(lm_head, "weight", None)
    visual_module = getattr(model, "visual", None)
    if visual_module is None:
        visual_module = getattr(getattr(model, "model", None), "visual", None)
    return {
        "prompt_token_count": int(prompt_length),
        "generated_token_count": len(generated_token_ids),
        "generated_token_ids": generated_token_ids,
        "generated_token_pieces_preview": generated_token_pieces,
        "first_generated_token_topk": first_token_topk,
        "quant_method": quant_method,
        "compute_dtype": str(compute_dtype) if compute_dtype is not None else None,
        "awq_backend": awq_backend,
        "lm_head_module_class": type(lm_head).__name__ if lm_head is not None else None,
        "lm_head_weight_dtype": (
            str(lm_head_weight.dtype) if lm_head_weight is not None else None
        ),
        "visual_module_class": (
            type(visual_module).__name__ if visual_module is not None else None
        ),
        "nonfinite_trace": nonfinite_trace,
        "fp8_expert_nonfinite": collect_fp8_expert_nonfinite_detail(model),
    }


def align_lm_head_input_dtype(model: Any) -> Any | None:
    """Cast hidden states to the lm_head weight dtype just before projection."""
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None or not hasattr(lm_head, "register_forward_pre_hook"):
        return None
    weight = getattr(lm_head, "weight", None)
    target_dtype = getattr(weight, "dtype", None)
    if target_dtype is None:
        return None

    def _cast_hidden_states(module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if not args:
            return args
        hidden_states = args[0]
        if (
            hasattr(hidden_states, "is_floating_point")
            and hidden_states.is_floating_point()
            and getattr(hidden_states, "dtype", None) != target_dtype
        ):
            hidden_states = hidden_states.to(dtype=target_dtype)
        return (hidden_states, *args[1:])

    return lm_head.register_forward_pre_hook(_cast_hidden_states)


def iter_named_model_tensors(model: Any) -> list[str]:
    """Collect parameter and buffer names without duplicating shared tensors."""
    seen: set[str] = set()
    names: list[str] = []
    for name, _ in model.named_parameters(recurse=True):
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    for name, _ in model.named_buffers(recurse=True):
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def is_quantized_awq_tensor_name(name: str) -> bool:
    """Identify AWQ tensor suffixes materialized by the runtime."""
    return (
        name.endswith(".qweight")
        or name.endswith(".qzeros")
        or name.endswith(".scales")
    )


def validate_awq_visual_load_layout(model_path: Path, model: Any) -> None:
    """Fail fast if an AWQ VL checkpoint excluded `visual` but runtime quantized it."""
    if checkpoint_quant_method(model_path) != "awq":
        return
    if "visual" not in checkpoint_modules_to_not_convert(model_path):
        return

    checkpoint_names = checkpoint_weight_names(model_path)
    if checkpoint_names is None:
        logger.warning(
            "awq visual load preflight skipped: missing model.safetensors.index.json for %s",
            model_path,
        )
        return

    checkpoint_has_plain_visual = any(
        name.startswith("visual.") for name in checkpoint_names
    )
    checkpoint_has_quantized_visual = any(
        name.startswith("model.visual.") and is_quantized_awq_tensor_name(name)
        for name in checkpoint_names
    )
    if not checkpoint_has_plain_visual or checkpoint_has_quantized_visual:
        return

    runtime_tensor_names = iter_named_model_tensors(model)
    runtime_quantized_visual = sorted(
        name
        for name in runtime_tensor_names
        if name.startswith("model.visual.") and is_quantized_awq_tensor_name(name)
    )
    if not runtime_quantized_visual:
        return

    example_names = ", ".join(runtime_quantized_visual[:3])
    raise RuntimeError(
        "AWQ visual tower load mismatch: checkpoint keeps `visual` unquantized "
        f"but runtime requested quantized visual tensors ({example_names}). "
        "This Transformers/AWQ stack is not compatible with this Qwen2.5-VL "
        "checkpoint; use an AutoAWQ-compatible Transformers stack or a non-AWQ checkpoint."
    )


def parse_layer_index_spec(raw_spec: str | None, *, num_layers: int) -> list[int]:
    """Parse a comma-separated list of layer indices or ranges."""
    if raw_spec is None:
        return []
    spec = raw_spec.strip().lower()
    if not spec or spec in {"false", "off", "none"}:
        return []
    if spec in {"1", "true", "on", "all"}:
        return list(range(num_layers))

    selected: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"invalid descending layer range: {token}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))

    invalid = sorted(index for index in selected if index < 0 or index >= num_layers)
    if invalid:
        raise ValueError(
            f"layer indices out of range for {num_layers} layers: {invalid}"
        )
    return sorted(selected)


def qwen35_fp8_repair_cache_file(
    repair_cache_dir: Path,
    *,
    layer_idx: int,
) -> Path:
    """Path for one cached repaired FP8 MoE layer."""
    return repair_cache_dir / f"qwen35_fp8_layer_{layer_idx:02d}.safetensors"


def build_qwen35_fp8_repair_cache(
    *,
    model_path: Path,
    repair_cache_dir: Path,
    layer_spec: str | None = "all",
    dequant_down_proj_spec: str | None = "all",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build cached repaired FP8 MoE tensors directly from checkpoint shards."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch

    preflight_transformers_checkpoint(model_path)
    model_config = load_model_config(model_path, trust_remote_code=False)
    if getattr(model_config, "model_type", None) != "qwen3_5_moe":
        raise RuntimeError(
            f"Qwen3.5 FP8 repair cache builder only supports qwen3_5_moe checkpoints: {model_path}"
        )
    if checkpoint_quant_method(model_path) != "fp8":
        raise RuntimeError(
            f"Qwen3.5 repair cache builder expected an FP8 checkpoint: {model_path}"
        )

    text_config = getattr(model_config, "text_config", None)
    if text_config is None:
        raise RuntimeError(
            f"Qwen3.5 FP8 repair cache builder expected text_config on {model_path}"
        )
    num_layers = int(getattr(text_config, "num_hidden_layers"))
    num_experts = int(getattr(text_config, "num_experts"))
    layer_indices = parse_layer_index_spec(layer_spec, num_layers=num_layers)
    if not layer_indices:
        raise ValueError(
            f"repair cache builder did not select any layers from layer_spec={layer_spec!r}"
        )
    dequant_down_proj_layers = set(
        parse_layer_index_spec(dequant_down_proj_spec, num_layers=num_layers)
    )

    quantization_config = getattr(model_config, "quantization_config", None)
    if isinstance(quantization_config, dict):
        raw_block_size = quantization_config.get("weight_block_size", [128, 128])
    else:
        raw_block_size = getattr(
            quantization_config,
            "weight_block_size",
            [128, 128],
        )
    block_size = tuple(int(dim) for dim in raw_block_size)

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise RuntimeError(
            f"missing safetensors index for Qwen3.5 FP8 repair cache build: {model_path}"
        )
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    repair_cache_dir.mkdir(parents=True, exist_ok=True)
    name_pattern = re.compile(
        r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
        r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale_inv)$"
    )
    written_layers: list[int] = []
    skipped_layers: list[int] = []
    for layer_idx in layer_indices:
        layer_cache_file = qwen35_fp8_repair_cache_file(
            repair_cache_dir,
            layer_idx=layer_idx,
        )
        if layer_cache_file.exists() and not overwrite:
            skipped_layers.append(layer_idx)
            logger.info(
                "skipping existing Qwen3.5 FP8 repair cache for layer %d at %s",
                layer_idx,
                layer_cache_file,
            )
            continue

        shard_to_names: dict[str, list[str]] = defaultdict(list)
        for expert_idx in range(num_experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for kind in ("weight", "weight_scale_inv"):
                    tensor_name = (
                        f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_idx}."
                        f"{proj}.{kind}"
                    )
                    shard_name = weight_map.get(tensor_name)
                    if shard_name is None:
                        raise RuntimeError(
                            f"missing checkpoint tensor required for expert repair: {tensor_name}"
                        )
                    shard_to_names[shard_name].append(tensor_name)

        gate_proj_weights: dict[int, Any] = {}
        up_proj_weights: dict[int, Any] = {}
        down_proj_weights: dict[int, Any] = {}
        gate_proj_scales: dict[int, Any] = {}
        up_proj_scales: dict[int, Any] = {}
        down_proj_scales: dict[int, Any] = {}
        for shard_name, tensor_names in shard_to_names.items():
            shard_path = model_path / shard_name
            with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
                for tensor_name in tensor_names:
                    match = name_pattern.match(tensor_name)
                    if match is None:
                        raise RuntimeError(
                            f"unexpected Qwen3.5 expert tensor name during repair: {tensor_name}"
                        )
                    expert_idx = int(match.group("expert"))
                    proj = str(match.group("proj"))
                    kind = str(match.group("kind"))
                    tensor = sf.get_tensor(tensor_name)
                    if kind == "weight":
                        if proj == "gate_proj":
                            gate_proj_weights[expert_idx] = tensor
                        elif proj == "up_proj":
                            up_proj_weights[expert_idx] = tensor
                        else:
                            down_proj_weights[expert_idx] = tensor
                    else:
                        if proj == "gate_proj":
                            gate_proj_scales[expert_idx] = tensor
                        elif proj == "up_proj":
                            up_proj_scales[expert_idx] = tensor
                        else:
                            down_proj_scales[expert_idx] = tensor

        missing_gate = sorted(
            expert_idx
            for expert_idx in range(num_experts)
            if expert_idx not in gate_proj_weights
            or expert_idx not in up_proj_weights
            or expert_idx not in gate_proj_scales
            or expert_idx not in up_proj_scales
        )
        missing_down = sorted(
            expert_idx
            for expert_idx in range(num_experts)
            if expert_idx not in down_proj_weights or expert_idx not in down_proj_scales
        )
        if missing_gate or missing_down:
            raise RuntimeError(
                f"incomplete Qwen3.5 FP8 repair tensors for layer {layer_idx}: "
                f"missing_gate={missing_gate} missing_down={missing_down}"
            )

        fused_gate_up = torch.stack(
            [
                torch.cat(
                    [
                        gate_proj_weights[expert_idx],
                        up_proj_weights[expert_idx],
                    ],
                    dim=0,
                )
                for expert_idx in range(num_experts)
            ],
            dim=0,
        ).contiguous()
        fused_gate_up_scale = torch.stack(
            [
                torch.cat(
                    [
                        gate_proj_scales[expert_idx],
                        up_proj_scales[expert_idx],
                    ],
                    dim=0,
                )
                for expert_idx in range(num_experts)
            ],
            dim=0,
        ).contiguous()

        cache_payload: dict[str, Any] = {
            "gate_up_proj": fused_gate_up,
            "gate_up_proj_scale_inv": fused_gate_up_scale,
        }
        if layer_idx in dequant_down_proj_layers:
            cache_payload["down_proj_bf16"] = torch.stack(
                [
                    dequantize_fp8_weight_blocks(
                        down_proj_weights[expert_idx],
                        down_proj_scales[expert_idx],
                        block_size=block_size,
                    )
                    for expert_idx in range(num_experts)
                ],
                dim=0,
            ).contiguous()
        else:
            cache_payload["down_proj"] = torch.stack(
                [down_proj_weights[expert_idx] for expert_idx in range(num_experts)],
                dim=0,
            ).contiguous()
            cache_payload["down_proj_scale_inv"] = torch.stack(
                [down_proj_scales[expert_idx] for expert_idx in range(num_experts)],
                dim=0,
            ).contiguous()

        save_file(cache_payload, str(layer_cache_file))
        written_layers.append(layer_idx)
        logger.info(
            "built Qwen3.5 FP8 repair cache for layer %d at %s",
            layer_idx,
            layer_cache_file,
        )

    return {
        "base_model_path": str(model_path),
        "repair_cache_dir": str(repair_cache_dir),
        "layer_spec": layer_spec,
        "dequant_down_proj_spec": dequant_down_proj_spec,
        "written_layers": written_layers,
        "skipped_existing_layers": skipped_layers,
        "num_layers": num_layers,
        "num_experts": num_experts,
    }


def dequantize_fp8_weight_blocks(
    quantized: Any,
    weight_scale_inv: Any,
    *,
    block_size: tuple[int, int],
) -> Any:
    """Dequantize a 2D fine-grained FP8 weight using its stored inverse scales."""
    import torch

    rows, cols = quantized.shape[-2:]
    block_m, block_n = block_size
    if rows % block_m != 0 or cols % block_n != 0:
        raise ValueError(
            f"FP8 weight shape {(rows, cols)} is not divisible by block size {block_size}"
        )
    reshaped = quantized.to(weight_scale_inv.dtype).reshape(
        rows // block_m,
        block_m,
        cols // block_n,
        block_n,
    )
    expanded_scales = weight_scale_inv.reshape(rows // block_m, cols // block_n)
    expanded_scales = expanded_scales.unsqueeze(-1).unsqueeze(1)
    dequantized = reshaped * expanded_scales
    return dequantized.reshape(rows, cols).to(torch.bfloat16)


def maybe_repair_qwen35_fp8_experts(
    *,
    model_path: Path,
    model: Any,
    repair_cache_dir: Path | None = None,
    repair_manifest: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Patch missing fused FP8 expert tensors from per-expert checkpoint weights."""
    manifest_layer_spec = None
    manifest_dequant_spec = None
    manifest_force_bf16_experts: bool | None = None
    if repair_manifest is not None:
        raw_manifest_layer_spec = repair_manifest.get("repair_layer_spec")
        if raw_manifest_layer_spec is not None:
            manifest_layer_spec = str(raw_manifest_layer_spec).strip() or None
        raw_manifest_dequant_spec = repair_manifest.get("dequant_down_proj_spec")
        if raw_manifest_dequant_spec is not None:
            manifest_dequant_spec = str(raw_manifest_dequant_spec).strip() or None
        raw_manifest_force_bf16_experts = repair_manifest.get("force_bf16_experts")
        if raw_manifest_force_bf16_experts is not None:
            manifest_force_bf16_experts = bool(raw_manifest_force_bf16_experts)

    env_layer_spec = os.environ.get("PRIMA_QWEN35_FP8_REPAIR_LAYERS")
    layer_spec = env_layer_spec or manifest_layer_spec
    if layer_spec is None:
        return None

    raw_env_force_bf16_experts = os.environ.get("PRIMA_QWEN35_FP8_FORCE_BF16_EXPERTS")
    if raw_env_force_bf16_experts is not None and raw_env_force_bf16_experts.strip():
        force_bf16_experts = raw_env_force_bf16_experts.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        force_bf16_experts_source = "env"
    else:
        force_bf16_experts = bool(manifest_force_bf16_experts)
        force_bf16_experts_source = "manifest"

    from safetensors import safe_open
    import torch

    text_model = getattr(getattr(model, "model", None), "language_model", None)
    if text_model is None or not hasattr(text_model, "layers"):
        raise RuntimeError(
            "Qwen3.5 FP8 expert repair expected model.model.language_model.layers"
        )

    layer_indices = parse_layer_index_spec(
        layer_spec,
        num_layers=len(text_model.layers),
    )
    dequant_down_proj_layers = set(
        parse_layer_index_spec(
            os.environ.get("PRIMA_QWEN35_FP8_DEQUANT_DOWN_PROJ_LAYERS")
            or manifest_dequant_spec,
            num_layers=len(text_model.layers),
        )
    )
    if not layer_indices:
        logger.info(
            "Qwen3.5 FP8 expert repair disabled by PRIMA_QWEN35_FP8_REPAIR_LAYERS=%s",
            layer_spec,
        )
        return {
            "enabled": False,
            "requested": layer_spec,
            "requested_source": "env" if env_layer_spec is not None else "manifest",
            "repaired_layers": [],
            "down_proj_dequantized_layers": [],
        }

    checkpoint_names = checkpoint_weight_names(model_path)
    if checkpoint_names is None:
        raise RuntimeError(
            f"missing safetensors index for Qwen3.5 FP8 expert repair: {model_path}"
        )
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    name_pattern = re.compile(
        r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
        r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale_inv)$"
    )

    repaired_layers: list[int] = []
    cached_layers_used: list[int] = []
    cached_layers_written: list[int] = []
    if repair_cache_dir is not None:
        repair_cache_dir.mkdir(parents=True, exist_ok=True)
    for layer_idx in layer_indices:
        experts = text_model.layers[layer_idx].mlp.experts
        setattr(experts, "_prima_layer_idx", int(layer_idx))
        setattr(experts, "_prima_force_bf16_experts", force_bf16_experts)
        gate_up_target = getattr(experts, "gate_up_proj", None)
        gate_up_scale_target = getattr(experts, "gate_up_proj_scale_inv", None)
        down_target = getattr(experts, "down_proj", None)
        down_scale_target = getattr(experts, "down_proj_scale_inv", None)
        if (
            gate_up_target is None
            or gate_up_scale_target is None
            or down_target is None
            or down_scale_target is None
        ):
            raise RuntimeError(
                f"layer {layer_idx} experts module does not expose fused FP8 expert tensors"
            )

        num_experts = int(getattr(experts, "num_experts"))
        quantization_config = getattr(model.config, "quantization_config", None)
        if isinstance(quantization_config, dict):
            raw_block_size = quantization_config.get("weight_block_size", [128, 128])
        else:
            raw_block_size = getattr(
                quantization_config,
                "weight_block_size",
                [128, 128],
            )
        block_size = tuple(int(dim) for dim in raw_block_size)
        dequantize_down_proj = layer_idx in dequant_down_proj_layers
        layer_cache_file = (
            qwen35_fp8_repair_cache_file(repair_cache_dir, layer_idx=layer_idx)
            if repair_cache_dir is not None
            else None
        )
        if layer_cache_file is not None and layer_cache_file.exists():
            with safe_open(str(layer_cache_file), framework="pt", device="cpu") as sf:
                gate_up_cached = sf.get_tensor("gate_up_proj")
                gate_up_scale_cached = sf.get_tensor("gate_up_proj_scale_inv")
                down_proj_bf16_cached = (
                    sf.get_tensor("down_proj_bf16")
                    if "down_proj_bf16" in sf.keys()
                    else None
                )
                down_proj_cached = (
                    sf.get_tensor("down_proj") if "down_proj" in sf.keys() else None
                )
                down_proj_scale_cached = (
                    sf.get_tensor("down_proj_scale_inv")
                    if "down_proj_scale_inv" in sf.keys()
                    else None
                )
            with torch.no_grad():
                gate_up_target.copy_(
                    gate_up_cached.to(
                        device=gate_up_target.device,
                        dtype=gate_up_target.dtype,
                    )
                )
                gate_up_scale_target.copy_(
                    gate_up_scale_cached.to(
                        device=gate_up_scale_target.device,
                        dtype=gate_up_scale_target.dtype,
                    )
                )
                if down_proj_bf16_cached is not None:
                    down_target.data = down_proj_bf16_cached.to(
                        device=down_target.device,
                        dtype=torch.bfloat16,
                    )
                elif (
                    down_proj_cached is not None and down_proj_scale_cached is not None
                ):
                    down_target.copy_(
                        down_proj_cached.to(
                            device=down_target.device,
                            dtype=down_target.dtype,
                        )
                    )
                    down_scale_target.copy_(
                        down_proj_scale_cached.to(
                            device=down_scale_target.device,
                            dtype=down_scale_target.dtype,
                        )
                    )
                else:
                    raise RuntimeError(
                        f"repair cache file is missing down_proj tensors: {layer_cache_file}"
                    )
            repaired_layers.append(layer_idx)
            cached_layers_used.append(layer_idx)
            logger.info(
                "loaded cached Qwen3.5 FP8 repaired experts for layer %d from %s",
                layer_idx,
                layer_cache_file,
            )
            continue

        down_proj_bf16 = None
        if dequantize_down_proj:
            down_proj_bf16 = down_target.detach().to(dtype=torch.bfloat16).clone()
        shard_to_names: dict[str, list[str]] = defaultdict(list)
        for expert_idx in range(num_experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for kind in ("weight", "weight_scale_inv"):
                    tensor_name = (
                        f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_idx}."
                        f"{proj}.{kind}"
                    )
                    if tensor_name not in checkpoint_names:
                        raise RuntimeError(
                            f"missing checkpoint tensor required for expert repair: {tensor_name}"
                        )
                    shard_to_names[weight_map[tensor_name]].append(tensor_name)

        pending_weights: dict[int, dict[str, Any]] = defaultdict(dict)
        pending_scales: dict[int, dict[str, Any]] = defaultdict(dict)
        pending_down_weights: dict[int, Any] = {}
        pending_down_scales: dict[int, Any] = {}
        for shard_name, tensor_names in shard_to_names.items():
            shard_path = model_path / shard_name
            with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
                for tensor_name in tensor_names:
                    match = name_pattern.match(tensor_name)
                    if match is None:
                        raise RuntimeError(
                            f"unexpected Qwen3.5 expert tensor name during repair: {tensor_name}"
                        )
                    expert_idx = int(match.group("expert"))
                    proj = str(match.group("proj"))
                    kind = str(match.group("kind"))
                    tensor = sf.get_tensor(tensor_name)

                    if kind == "weight":
                        if proj == "down_proj":
                            if dequantize_down_proj:
                                pending_down_weights[expert_idx] = tensor
                                if expert_idx in pending_down_scales:
                                    dequantized = dequantize_fp8_weight_blocks(
                                        pending_down_weights.pop(expert_idx),
                                        pending_down_scales.pop(expert_idx),
                                        block_size=block_size,
                                    )
                                    down_proj_bf16[expert_idx].copy_(
                                        dequantized.to(device=down_target.device)
                                    )
                            else:
                                with torch.no_grad():
                                    down_target[expert_idx].copy_(
                                        tensor.to(
                                            device=down_target.device,
                                            dtype=down_target.dtype,
                                        )
                                    )
                        else:
                            pending_weights[expert_idx][proj] = tensor
                            if {"gate_proj", "up_proj"} <= pending_weights[
                                expert_idx
                            ].keys():
                                fused = torch.cat(
                                    [
                                        pending_weights[expert_idx]["gate_proj"],
                                        pending_weights[expert_idx]["up_proj"],
                                    ],
                                    dim=0,
                                )
                                with torch.no_grad():
                                    gate_up_target[expert_idx].copy_(
                                        fused.to(
                                            device=gate_up_target.device,
                                            dtype=gate_up_target.dtype,
                                        )
                                    )
                                del pending_weights[expert_idx]
                    else:
                        if proj == "down_proj":
                            if dequantize_down_proj:
                                pending_down_scales[expert_idx] = tensor
                                if expert_idx in pending_down_weights:
                                    dequantized = dequantize_fp8_weight_blocks(
                                        pending_down_weights.pop(expert_idx),
                                        pending_down_scales.pop(expert_idx),
                                        block_size=block_size,
                                    )
                                    down_proj_bf16[expert_idx].copy_(
                                        dequantized.to(device=down_target.device)
                                    )
                            else:
                                with torch.no_grad():
                                    down_scale_target[expert_idx].copy_(
                                        tensor.to(
                                            device=down_scale_target.device,
                                            dtype=down_scale_target.dtype,
                                        )
                                    )
                        else:
                            pending_scales[expert_idx][proj] = tensor
                            if {"gate_proj", "up_proj"} <= pending_scales[
                                expert_idx
                            ].keys():
                                fused_scale = torch.cat(
                                    [
                                        pending_scales[expert_idx]["gate_proj"],
                                        pending_scales[expert_idx]["up_proj"],
                                    ],
                                    dim=0,
                                )
                                with torch.no_grad():
                                    gate_up_scale_target[expert_idx].copy_(
                                        fused_scale.to(
                                            device=gate_up_scale_target.device,
                                            dtype=gate_up_scale_target.dtype,
                                        )
                                    )
                                del pending_scales[expert_idx]

        if pending_weights or pending_scales:
            raise RuntimeError(
                f"incomplete Qwen3.5 FP8 expert repair state for layer {layer_idx}: "
                f"weights_pending={sorted(pending_weights)} scales_pending={sorted(pending_scales)}"
            )
        if pending_down_weights or pending_down_scales:
            raise RuntimeError(
                f"incomplete Qwen3.5 down_proj repair state for layer {layer_idx}: "
                f"down_weights_pending={sorted(pending_down_weights)} "
                f"down_scales_pending={sorted(pending_down_scales)}"
            )
        if dequantize_down_proj and down_proj_bf16 is not None:
            with torch.no_grad():
                down_target.data = down_proj_bf16.to(device=down_target.device)
        repaired_layers.append(layer_idx)
        logger.info(
            "repaired Qwen3.5 FP8 fused expert tensors for layer %d from checkpoint %s",
            layer_idx,
            model_path,
        )
        if dequantize_down_proj:
            logger.info(
                "dequantized Qwen3.5 FP8 expert down_proj tensors to bf16 for layer %d",
                layer_idx,
            )
        if layer_cache_file is not None:
            from safetensors.torch import save_file

            cache_payload: dict[str, Any] = {
                "gate_up_proj": gate_up_target.detach().cpu(),
                "gate_up_proj_scale_inv": gate_up_scale_target.detach().cpu(),
            }
            if dequantize_down_proj:
                cache_payload["down_proj_bf16"] = down_target.detach().cpu()
            else:
                cache_payload["down_proj"] = down_target.detach().cpu()
                cache_payload["down_proj_scale_inv"] = down_scale_target.detach().cpu()
            save_file(cache_payload, str(layer_cache_file))
            cached_layers_written.append(layer_idx)
            logger.info(
                "wrote cached Qwen3.5 FP8 repaired experts for layer %d to %s",
                layer_idx,
                layer_cache_file,
            )

    return {
        "enabled": True,
        "requested": layer_spec,
        "requested_source": "env" if env_layer_spec is not None else "manifest",
        "repaired_layers": repaired_layers,
        "down_proj_dequantized_layers": sorted(
            dequant_down_proj_layers & set(repaired_layers)
        ),
        "repair_cache_dir": str(repair_cache_dir)
        if repair_cache_dir is not None
        else None,
        "cached_layers_used": cached_layers_used,
        "cached_layers_written": cached_layers_written,
        "force_bf16_experts": force_bf16_experts,
        "force_bf16_experts_source": force_bf16_experts_source,
    }


def load_model_with_optional_allocator_warmup_skip(
    auto_vision_model: Any,
    *,
    model_path: Path,
    model_config: Any,
    load_kwargs: dict[str, Any],
) -> Any:
    """Load a model, skipping allocator warmup for known-bad MoE load paths."""
    model_type = getattr(model_config, "model_type", None)
    if model_type != "qwen3_5_moe":
        return auto_vision_model.from_pretrained(
            str(model_path),
            config=model_config,
            **load_kwargs,
        )

    import transformers.modeling_utils as modeling_utils

    original_caching_allocator_warmup = modeling_utils.caching_allocator_warmup

    def _skip_caching_allocator_warmup(*args: Any, **kwargs: Any) -> None:
        return None

    modeling_utils.caching_allocator_warmup = _skip_caching_allocator_warmup
    logger.info(
        "temporarily disabled transformers caching_allocator_warmup for %s to avoid loader-time OOM",
        model_path,
    )
    try:
        return auto_vision_model.from_pretrained(
            str(model_path),
            config=model_config,
            **load_kwargs,
        )
    finally:
        modeling_utils.caching_allocator_warmup = original_caching_allocator_warmup


class LocalVisionAnnotator:
    """Direct local inference over the checkpoint using Transformers."""

    def __init__(
        self,
        *,
        model_path: Path,
        expected_gpus: int | None,
        max_memory_per_gpu: str | None,
        cpu_max_memory: str | None,
        max_new_tokens: int,
        few_shot_examples: int,
        few_shot_exemplar_pool: list[dict[str, Any]],
        prompt_mode: str,
        prompt_variant: str,
        probe_tag: str | None,
        target_prompt_override: str | None,
        text_only_prompt: str | None,
        disable_thinking: bool,
        debug_dump_dir: Path | None,
        trust_remote_code: bool,
    ) -> None:
        self.requested_model_path = model_path
        (
            self.model_path,
            self.repair_cache_dir,
            self.repair_manifest,
        ) = resolve_model_source_and_repair_cache(model_path)
        self.max_new_tokens = max_new_tokens
        self.few_shot_examples = few_shot_examples
        self.few_shot_exemplar_pool = list(few_shot_exemplar_pool)
        self.prompt_mode = prompt_mode
        if prompt_variant not in PROMPT_VARIANTS:
            raise ValueError(f"unsupported prompt variant: {prompt_variant}")
        self.prompt_variant = prompt_variant
        self.probe_tag = probe_tag
        self.target_prompt_override = (
            target_prompt_override.strip() if target_prompt_override else None
        )
        self.text_only_prompt = text_only_prompt.strip() if text_only_prompt else None
        self.disable_thinking = disable_thinking
        self.debug_dump_dir = debug_dump_dir
        if self.debug_dump_dir is not None:
            self.debug_dump_dir.mkdir(parents=True, exist_ok=True)
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as auto_vision_model
        except ImportError:  # pragma: no cover - depends on transformers build
            from transformers import AutoModelForVision2Seq as auto_vision_model

        pin_transformers_fp8_kernel_revisions()
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        visible_gpus = torch.cuda.device_count()
        if expected_gpus is not None and visible_gpus != expected_gpus:
            raise RuntimeError(
                f"expected {expected_gpus} visible GPUs, found {visible_gpus}"
            )
        if visible_gpus <= 0:
            raise RuntimeError("no CUDA devices visible; run this on a GPU node")
        patch_torch_cuda_get_device_properties_default(torch)

        max_memory = build_max_memory_map(
            num_visible_gpus=visible_gpus,
            max_memory_per_gpu=max_memory_per_gpu,
            cpu_max_memory=cpu_max_memory,
        )

        preflight_transformers_checkpoint(self.model_path)
        model_config = load_model_config(
            self.model_path,
            trust_remote_code=trust_remote_code,
        )

        logger.info(
            "loading processor from %s (requested path %s)",
            self.model_path,
            self.requested_model_path,
        )
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            trust_remote_code=trust_remote_code,
        )
        if not hasattr(self.processor, "apply_chat_template"):
            raise RuntimeError(
                f"processor at {self.model_path} does not support apply_chat_template"
            )

        logger.info(
            "loading model from %s across %d visible GPUs (requested path %s)",
            self.model_path,
            visible_gpus,
            self.requested_model_path,
        )
        quant_method = checkpoint_quant_method(self.model_path)
        self.quant_method = quant_method
        self.awq_backend: str | None = None
        load_dtype: Any = "auto"
        if (
            quant_method == "fp8"
            and getattr(model_config, "model_type", None) == "qwen3_5_moe"
        ):
            patch_qwen35_fp8_eager_experts()
        if quant_method == "awq":
            # transformers replaces AWQ linear modules under torch.device("meta");
            # importing the selected backend for the first time inside that context crashes.
            self.awq_backend = prepare_awq_backend(self.model_path)
            load_dtype = torch.float16
            logger.info(
                "forcing AWQ checkpoint %s to torch.float16 using backend %s to avoid mixed bfloat16/float16 generation",
                self.model_path,
                self.awq_backend,
            )

        self.model = load_model_with_optional_allocator_warmup_skip(
            auto_vision_model,
            model_path=self.model_path,
            model_config=model_config,
            load_kwargs={
                "torch_dtype": load_dtype,
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                "trust_remote_code": trust_remote_code,
                "max_memory": max_memory,
            },
        )
        if (
            quant_method == "fp8"
            and getattr(model_config, "model_type", None) == "qwen3_5_moe"
        ):
            if not hasattr(self.model, "set_experts_implementation"):
                raise RuntimeError(
                    "Qwen3.5 FP8 model does not expose set_experts_implementation; "
                    "cannot force the safer MoE dispatch path"
                )
            self.model.set_experts_implementation("eager")
            logger.info(
                "forcing Qwen3.5 FP8 experts_implementation=eager for %s to avoid Triton MoE decode crashes on sharded GPU inference",
                self.model_path,
            )
        self.fp8_expert_repair_summary: dict[str, Any] | None = None
        if (
            quant_method == "fp8"
            and getattr(model_config, "model_type", None) == "qwen3_5_moe"
        ):
            self.fp8_expert_repair_summary = maybe_repair_qwen35_fp8_experts(
                model_path=self.model_path,
                model=self.model,
                repair_cache_dir=self.repair_cache_dir,
                repair_manifest=self.repair_manifest,
            )
        self.compute_dtype = (
            torch.float16
            if quant_method == "awq"
            else getattr(self.model, "dtype", None)
        )
        self.lm_head_dtype_hook = None
        if quant_method == "awq":
            if hasattr(self.model.config, "torch_dtype"):
                self.model.config.torch_dtype = torch.float16
            text_config = getattr(self.model.config, "text_config", None)
            if text_config is not None and hasattr(text_config, "torch_dtype"):
                text_config.torch_dtype = torch.float16
            self.lm_head_dtype_hook = align_lm_head_input_dtype(self.model)
            validate_awq_visual_load_layout(self.model_path, self.model)
        generation_config = getattr(self.model, "generation_config", None)
        if (
            generation_config is not None
            and getattr(generation_config, "temperature", None) is not None
        ):
            generation_config.temperature = None
        logger.info(
            "model ready: device=%s dtype=%s experts_implementation=%s",
            getattr(self.model, "device", None),
            getattr(self.model, "dtype", None),
            getattr(self.model.config, "_experts_implementation", None),
        )
        self.model.eval()

    def annotate(
        self,
        *,
        exam_id: str,
        image_path: Path,
        tag_catalog: list[str],
    ) -> dict[str, Any]:
        """Run one montage through the local model."""
        import torch
        from qwen_vl_utils import process_vision_info

        if self.prompt_mode == "marker_classifier" and self.probe_tag:
            few_shot_examples = select_probe_few_shot_examples(
                exemplar_pool=self.few_shot_exemplar_pool,
                max_examples=self.few_shot_examples,
                probe_tag=str(self.probe_tag),
                exclude_exam_id=exam_id,
            )
        else:
            few_shot_examples = select_few_shot_examples(
                exemplar_pool=self.few_shot_exemplar_pool,
                max_examples=self.few_shot_examples,
                exclude_exam_id=exam_id,
            )
        if self.text_only_prompt is not None:
            few_shot_examples = []
        target_prompt_text = build_target_prompt_text(
            prompt_mode=self.prompt_mode,
            tag_catalog=tag_catalog,
            few_shot_examples=few_shot_examples,
            probe_tag=self.probe_tag,
            prompt_variant=self.prompt_variant,
        )
        if self.target_prompt_override is not None:
            target_prompt_text = self.target_prompt_override

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": build_system_prompt(self.prompt_mode)}
                ],
            }
        ]
        for idx, exemplar in enumerate(few_shot_examples, start=1):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": Path(exemplar["image_path"]).resolve().as_uri(),
                        },
                        {
                            "type": "text",
                            "text": build_few_shot_user_prompt(
                                exemplar_index=idx,
                                reference_type="montage",
                                target_prompt_text=target_prompt_text,
                            ),
                        },
                    ],
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": build_few_shot_assistant_payload(
                                list(exemplar["annotations"]),
                                prompt_mode=self.prompt_mode,
                                probe_tag=self.probe_tag,
                                evidence=exemplar.get("role"),
                            ),
                        }
                    ],
                }
            )
        if self.text_only_prompt is not None:
            target_prompt_text = self.text_only_prompt
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": target_prompt_text,
                        }
                    ],
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path.resolve().as_uri()},
                        {
                            "type": "text",
                            "text": target_prompt_text,
                        },
                    ],
                }
            )

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=not self.disable_thinking,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        processor_inputs: dict[str, Any] = {
            "text": [prompt],
            "padding": True,
            "return_tensors": "pt",
        }
        if image_inputs is not None:
            processor_inputs["images"] = image_inputs
        if video_inputs is not None:
            processor_inputs["videos"] = video_inputs
        inputs = self.processor(**processor_inputs)
        inputs = move_inputs_to_model_device(
            inputs,
            self.model,
            dtype=self.compute_dtype,
        )
        prompt_length = inputs["input_ids"].shape[1]

        generation_scores: list[Any] = []
        nonfinite_trace_summary: dict[str, Any] | None = None
        nonfinite_tracer: NonFiniteActivationTracer | None = None
        if self.debug_dump_dir is not None and self.quant_method == "fp8":
            nonfinite_tracer = NonFiniteActivationTracer(self.model).start()
        with torch.inference_mode():
            try:
                generation_kwargs = {
                    "do_sample": False,
                    "max_new_tokens": self.max_new_tokens,
                }
                if self.prompt_mode == "binary_tag_probe":
                    tokenizer = getattr(self.processor, "tokenizer", self.processor)
                    eos_token_id = getattr(tokenizer, "eos_token_id", None)
                    if eos_token_id is None:
                        raise RuntimeError(
                            "binary probe constrained decoding requires eos_token_id"
                        )
                    if isinstance(eos_token_id, list):
                        eos_token_ids = [int(token_id) for token_id in eos_token_id]
                    else:
                        eos_token_ids = [int(eos_token_id)]
                    choice_variants = build_binary_probe_choice_variants(self.processor)
                    generation_kwargs["max_new_tokens"] = min(
                        generation_kwargs["max_new_tokens"],
                        max(len(variant["token_ids"]) for variant in choice_variants)
                        + 1,
                    )
                    generation_kwargs["prefix_allowed_tokens_fn"] = (
                        build_binary_probe_prefix_allowed_tokens_fn(
                            prompt_length=prompt_length,
                            choice_variants=choice_variants,
                            eos_token_ids=eos_token_ids,
                        )
                    )
                if self.debug_dump_dir is not None:
                    generation_kwargs.update(
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                    generation_output = self.model.generate(
                        **inputs,
                        **generation_kwargs,
                    )
                    generated_ids = generation_output.sequences
                    generation_scores = list(generation_output.scores or [])
                else:
                    generated_ids = self.model.generate(
                        **inputs,
                        **generation_kwargs,
                    )
            finally:
                if nonfinite_tracer is not None:
                    nonfinite_trace_summary = nonfinite_tracer.summary()
                    nonfinite_tracer.close()

        new_token_ids = generated_ids[:, prompt_length:]
        response_text = self.processor.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        suggestions: list[dict[str, Any]] = []
        response_payload: Any = None
        parse_error: str | None = None
        if self.prompt_mode == "tagger_json":
            try:
                response_payload = extract_json_payload(response_text)
            except ValueError as exc:
                parse_error = str(exc)
                logger.warning(
                    "model response was not valid JSON; treating as no suggestions"
                )
            suggestions = normalize_model_suggestions(
                response_payload, set(tag_catalog)
            )
        elif self.prompt_mode == "binary_tag_probe":
            try:
                response_payload = extract_json_payload(response_text)
            except ValueError as exc:
                parse_error = str(exc)
                response_payload = coerce_binary_probe_payload(response_text)
                if response_payload is None:
                    logger.warning(
                        "binary probe response was not parseable; treating as not present"
                    )
            suggestions = normalize_binary_probe_response(
                response_payload,
                probe_tag=str(self.probe_tag),
                allowed_tags=set(tag_catalog),
            )
        elif self.prompt_mode == "marker_classifier":
            response_payload = coerce_marker_classifier_payload(response_text)
            if response_payload is None:
                logger.warning(
                    "marker classifier response was not parseable; treating as not present"
                )
            suggestions = normalize_binary_probe_response(
                response_payload,
                probe_tag=str(self.probe_tag),
                allowed_tags=set(tag_catalog),
            )

        debug_dump_file = None
        if self.debug_dump_dir is not None:
            debug_payload = {
                "exam_id": str(exam_id),
                "image_file": image_path.name,
                "prompt_mode": self.prompt_mode,
                "prompt_variant": self.prompt_variant,
                "probe_tag": self.probe_tag,
                "target_prompt_override": self.target_prompt_override,
                "text_only_prompt": self.text_only_prompt,
                "disable_thinking": self.disable_thinking,
                "few_shot_example_exam_ids": [
                    str(example["exam_id"]) for example in few_shot_examples
                ],
                "few_shot_examples": [
                    {
                        "exam_id": str(example["exam_id"]),
                        "annotations": list(example["annotations"]),
                    }
                    for example in few_shot_examples
                ],
                "tag_catalog": list(tag_catalog),
                "target_prompt_text": target_prompt_text,
                "raw_response_text": response_text,
                "parsed_response_payload": response_payload,
                "parse_error": parse_error,
                "normalized_suggestions": suggestions,
                "generation_debug": build_generation_debug_payload(
                    processor=self.processor,
                    model=self.model,
                    prompt_length=prompt_length,
                    new_token_ids=new_token_ids,
                    generation_scores=generation_scores,
                    quant_method=self.quant_method,
                    compute_dtype=self.compute_dtype,
                    awq_backend=self.awq_backend,
                    nonfinite_trace=nonfinite_trace_summary,
                ),
                "fp8_expert_repair": self.fp8_expert_repair_summary,
            }
            debug_dump_file = f"{exam_id}.json"
            with open(self.debug_dump_dir / debug_dump_file, "w") as f:
                json.dump(debug_payload, f, indent=2)

        return {
            "suggestions": suggestions,
            "few_shot_example_exam_ids": [
                str(example["exam_id"]) for example in few_shot_examples
            ],
            "debug_dump_file": debug_dump_file,
            "prompt_mode": self.prompt_mode,
            "prompt_variant": self.prompt_variant,
        }


class VLLMVisionAnnotator:
    """Multimodal QC inference through a managed local vLLM service."""

    def __init__(
        self,
        *,
        model_spec: Any,
        model_path: Path,
        port: int,
        server_log_path: Path,
        startup_timeout_seconds: int,
        request_timeout_seconds: int,
        max_new_tokens: int,
        few_shot_examples: int,
        few_shot_exemplar_pool: list[dict[str, Any]],
        prompt_mode: str,
        prompt_variant: str,
        probe_tag: str | None,
        target_prompt_override: str | None,
        text_only_prompt: str | None,
        disable_thinking: bool,
        debug_dump_dir: Path | None,
        input_level: str = "exam",
    ) -> None:
        from openai import DefaultHttpxClient, OpenAI
        import torch

        from prima.vllm_server import ManagedVLLMServer

        if request_timeout_seconds <= 0:
            raise ValueError("vLLM request timeout must be positive")
        self.model_spec = model_spec
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.few_shot_examples = few_shot_examples
        self.few_shot_exemplar_pool = list(few_shot_exemplar_pool)
        self.prompt_mode = prompt_mode
        if prompt_variant not in PROMPT_VARIANTS:
            raise ValueError(f"unsupported prompt variant: {prompt_variant}")
        self.prompt_variant = prompt_variant
        self.probe_tag = probe_tag
        self.target_prompt_override = (
            target_prompt_override.strip() if target_prompt_override else None
        )
        self.text_only_prompt = text_only_prompt.strip() if text_only_prompt else None
        self.disable_thinking_requested = disable_thinking
        self.disable_thinking = True
        self.request_timeout_seconds = request_timeout_seconds
        self.debug_dump_dir = debug_dump_dir
        if input_level not in {"exam", "view"}:
            raise ValueError(f"unsupported input level: {input_level}")
        self.input_level = input_level
        if self.debug_dump_dir is not None:
            self.debug_dump_dir.mkdir(parents=True, exist_ok=True)
        visible_gpus = torch.cuda.device_count()
        if visible_gpus != model_spec.tensor_parallel_size:
            raise RuntimeError(
                f"vLLM model {model_spec.key!r} requires "
                f"{model_spec.tensor_parallel_size} visible GPUs, found {visible_gpus}"
            )

        self.server = ManagedVLLMServer(
            spec=model_spec,
            model_path=model_path,
            port=port,
            log_path=server_log_path,
            startup_timeout_seconds=startup_timeout_seconds,
        )
        self.client: Any = None
        try:
            self.server.start()
            self.client = OpenAI(
                base_url=self.server.base_url,
                api_key="not-needed",
                timeout=request_timeout_seconds,
                http_client=DefaultHttpxClient(trust_env=False),
            )
        except Exception:
            self.server.stop()
            raise

    def close(self) -> None:
        """Release the vLLM process and its GPU workers."""
        try:
            if self.client is not None:
                self.client.close()
                self.client = None
        finally:
            self.server.stop()

    def _select_few_shot_examples(self, exam_id: str) -> list[dict[str, Any]]:
        ordered = [
            example
            for example in self.few_shot_exemplar_pool
            if "few_shot_order" in example
        ]
        if ordered:
            if len(ordered) != len(self.few_shot_exemplar_pool):
                raise ValueError(
                    "few-shot exemplar pool mixes ordered and unordered records"
                )
            orders = [int(example["few_shot_order"]) for example in ordered]
            if len(orders) != len(set(orders)) or min(orders) <= 0:
                raise ValueError(
                    "ordered few-shot exemplars require unique positive order"
                )
            examples = [
                example
                for example in sorted(
                    ordered, key=lambda record: int(record["few_shot_order"])
                )
                if str(example["exam_id"]) != str(exam_id)
            ][: self.few_shot_examples]
        elif self.prompt_mode == "marker_classifier" and self.probe_tag:
            examples = select_probe_few_shot_examples(
                exemplar_pool=self.few_shot_exemplar_pool,
                max_examples=self.few_shot_examples,
                probe_tag=str(self.probe_tag),
                exclude_exam_id=exam_id,
            )
        else:
            examples = select_few_shot_examples(
                exemplar_pool=self.few_shot_exemplar_pool,
                max_examples=self.few_shot_examples,
                exclude_exam_id=exam_id,
            )
        return [] if self.text_only_prompt is not None else examples

    def _build_messages(
        self,
        *,
        image_path: Path,
        target_prompt_text: str,
        few_shot_examples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": build_system_prompt(
                    self.prompt_mode, input_level=self.input_level
                ),
            }
        ]
        reference_type = "view" if self.input_level == "view" else "montage"
        for idx, exemplar in enumerate(few_shot_examples, start=1):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url(Path(exemplar["image_path"]))
                            },
                        },
                        {
                            "type": "text",
                            "text": build_few_shot_user_prompt(
                                exemplar_index=idx,
                                reference_type=reference_type,
                                target_prompt_text=target_prompt_text,
                            ),
                        },
                    ],
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": build_few_shot_assistant_payload(
                        list(exemplar["annotations"]),
                        prompt_mode=self.prompt_mode,
                        probe_tag=self.probe_tag,
                        evidence=exemplar.get("role"),
                    ),
                }
            )
        if self.text_only_prompt is not None:
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": target_prompt_text}],
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url(image_path)},
                        },
                        {"type": "text", "text": target_prompt_text},
                    ],
                }
            )
        return messages

    def _parse_response(
        self,
        *,
        response_text: str,
        tag_catalog: list[str],
    ) -> tuple[Any, list[dict[str, Any]]]:
        if not response_text.strip():
            raise RuntimeError("vLLM returned empty response content")
        if self.prompt_mode == "tagger_json":
            response_payload = extract_json_payload(response_text)
            return response_payload, normalize_model_suggestions(
                response_payload, set(tag_catalog)
            )
        if self.prompt_mode == "binary_tag_probe":
            try:
                response_payload = extract_json_payload(response_text)
            except ValueError:
                response_payload = coerce_binary_probe_payload(response_text)
            if response_payload is None:
                raise ValueError("vLLM binary response was not parseable")
            return response_payload, normalize_binary_probe_response(
                response_payload,
                probe_tag=str(self.probe_tag),
                allowed_tags=set(tag_catalog),
            )
        if self.prompt_mode == "marker_classifier":
            response_payload = coerce_marker_classifier_payload(response_text)
            if response_payload is None:
                raise ValueError("vLLM marker response was not parseable")
            return response_payload, normalize_binary_probe_response(
                response_payload,
                probe_tag=str(self.probe_tag),
                allowed_tags=set(tag_catalog),
            )
        return None, []

    def annotate(
        self,
        *,
        exam_id: str,
        image_path: Path,
        tag_catalog: list[str],
    ) -> dict[str, Any]:
        """Send one target montage and optional references to local vLLM."""
        if self.client is None:
            raise RuntimeError("vLLM annotator is closed")
        few_shot_examples = self._select_few_shot_examples(exam_id)
        target_prompt_text = build_target_prompt_text(
            prompt_mode=self.prompt_mode,
            tag_catalog=tag_catalog,
            few_shot_examples=few_shot_examples,
            probe_tag=self.probe_tag,
            prompt_variant=self.prompt_variant,
            input_level=self.input_level,
        )
        if self.target_prompt_override is not None:
            target_prompt_text = self.target_prompt_override
        if self.text_only_prompt is not None:
            target_prompt_text = self.text_only_prompt
        messages = self._build_messages(
            image_path=image_path,
            target_prompt_text=target_prompt_text,
            few_shot_examples=few_shot_examples,
        )

        request: dict[str, Any] = {
            "model": self.model_spec.served_model_name,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_new_tokens,
            "timeout": self.request_timeout_seconds,
        }
        extra_body: dict[str, Any] = {}
        if self.disable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        if self.prompt_mode == "binary_tag_probe":
            extra_body["structured_outputs"] = {"choice": ["yes", "no"]}
        if extra_body:
            request["extra_body"] = extra_body
        if self.prompt_mode == "tagger_json":
            request["response_format"] = tagger_json_response_format(tag_catalog)
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            raise sanitized_vllm_request_error(exc) from None
        response_text = response.choices[0].message.content or ""
        response_payload, suggestions = self._parse_response(
            response_text=response_text,
            tag_catalog=tag_catalog,
        )

        debug_dump_file = None
        if self.debug_dump_dir is not None:
            usage = getattr(response, "usage", None)
            debug_payload = {
                "exam_id": str(exam_id),
                "image_file": image_path.name,
                "backend": "vllm_local",
                "input_level": self.input_level,
                "model_key": self.model_spec.key,
                "served_model_name": self.model_spec.served_model_name,
                "model_revision": self.model_spec.revision,
                "prompt_mode": self.prompt_mode,
                "prompt_variant": self.prompt_variant,
                "probe_tag": self.probe_tag,
                "target_prompt_override": self.target_prompt_override,
                "text_only_prompt": self.text_only_prompt,
                "disable_thinking_requested": self.disable_thinking_requested,
                "disable_thinking_effective": self.disable_thinking,
                "few_shot_example_exam_ids": [
                    str(example["exam_id"]) for example in few_shot_examples
                ],
                "few_shot_examples": [
                    {
                        "exam_id": str(example["exam_id"]),
                        "annotations": list(example["annotations"]),
                    }
                    for example in few_shot_examples
                ],
                "tag_catalog": list(tag_catalog),
                "target_prompt_text": target_prompt_text,
                "raw_response_text": response_text,
                "parsed_response_payload": response_payload,
                "normalized_suggestions": suggestions,
                "response_id": getattr(response, "id", None),
                "usage": usage.model_dump() if usage is not None else None,
            }
            debug_dump_file = f"{exam_id}.json"
            debug_path = self.debug_dump_dir / debug_dump_file
            temporary_path = debug_path.with_suffix(".json.tmp")
            try:
                temporary_path.write_text(json.dumps(debug_payload, indent=2) + "\n")
                temporary_path.replace(debug_path)
            except OSError:
                raise RuntimeError("failed to write vLLM debug record") from None

        return {
            "suggestions": suggestions,
            "few_shot_example_exam_ids": [
                str(example["exam_id"]) for example in few_shot_examples
            ],
            "debug_dump_file": debug_dump_file,
            "prompt_mode": self.prompt_mode,
            "prompt_variant": self.prompt_variant,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser so submitit can reuse the same arguments."""
    parser = argparse.ArgumentParser(
        description="Run a local multimodal model over cached mammography QC montages.",
        epilog=(
            "Run this directly from the pinned vLLM environment on a GPU node, "
            "or submit it with submit_auto_qc.py.\n\n"
            "Example using the configured 4xH200 Qwen3.5 service:\n"
            "  python auto_annotate_qc.py --views /path/views_for_qc.parquet "
            "--export-dir /path/qc_export --run-file /path/auto_qc_run.json "
            "--backend vllm --model-key qwen35_397b_fp8 --expected-gpus 4"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--views",
        type=Path,
        required=True,
        help="Path to views.parquet used for qc_gallery.py",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        required=True,
        help="QC export directory containing success/<patient>/<accession>/COMBINED_four_views_<exam>.png",
    )
    parser.add_argument(
        "--run-file",
        type=Path,
        required=True,
        help="Path to write the normalized auto-QC run JSON",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "transformers"],
        default="vllm",
        help="Inference backend (default: vllm)",
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default="qwen35_397b_fp8",
        help="Configured key in --model-registry for the vLLM backend",
    )
    parser.add_argument(
        "--model-registry",
        type=Path,
        default=DEFAULT_VLLM_MODEL_REGISTRY,
        help=f"vLLM model registry (default: {DEFAULT_VLLM_MODEL_REGISTRY})",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"Root containing local model snapshots (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local checkpoint directory (required only for --backend=transformers)",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=0,
        help="Loopback port for vLLM; 0 selects a free job-local port (default: 0)",
    )
    parser.add_argument(
        "--vllm-server-log",
        type=Path,
        default=None,
        help="vLLM server log; defaults beside --run-file with the Slurm job ID",
    )
    parser.add_argument(
        "--vllm-startup-timeout-seconds",
        type=int,
        default=1800,
        help="Maximum wait for the vLLM health endpoint (default: 1800)",
    )
    parser.add_argument(
        "--vllm-request-timeout-seconds",
        type=int,
        default=600,
        help="Per-exam OpenAI-compatible request timeout (default: 600)",
    )
    parser.add_argument(
        "--tags-file",
        type=Path,
        default=None,
        help="Path to annotation_tags.json. Defaults to <run-file dir>/annotation_tags.json if present, else default tags.",
    )
    parser.add_argument(
        "--qc-file",
        type=Path,
        default=None,
        help="Optional QC state file. Use with --skip-existing-annotations to avoid already accepted GT.",
    )
    parser.add_argument(
        "--skip-existing-annotations",
        action="store_true",
        help="Skip exams that already have accepted annotations in --qc-file",
    )
    parser.add_argument(
        "--exam",
        type=str,
        default=None,
        help="Restrict to one exam_id",
    )
    parser.add_argument(
        "--exam-list",
        type=Path,
        default=None,
        help="Optional text file of exam IDs to score",
    )
    parser.add_argument(
        "--max-exams",
        type=int,
        default=None,
        help="Maximum number of exams to score",
    )
    parser.add_argument(
        "--force-rescore",
        action="store_true",
        help="Overwrite any existing run file and rescore selected exams",
    )
    parser.add_argument(
        "--pilot-queue-dir",
        type=Path,
        default=None,
        help=(
            "Optional filesystem queue directory. When set, the model stays loaded "
            "and claims task files from <queue>/pending until idle, stopped, or preempted."
        ),
    )
    parser.add_argument(
        "--pilot-id",
        type=str,
        default=None,
        help="Optional pilot ID for queue status sidecars",
    )
    parser.add_argument(
        "--pilot-poll-seconds",
        type=int,
        default=30,
        help="Seconds to sleep between pilot queue polls when idle (default: 30)",
    )
    parser.add_argument(
        "--pilot-idle-timeout-seconds",
        type=int,
        default=1800,
        help="Exit after this many idle seconds in pilot mode; use 0 to wait until walltime/preemption",
    )
    parser.add_argument(
        "--pilot-stop-file",
        type=Path,
        default=None,
        help="Optional file path that asks a pilot to exit after the current task",
    )
    parser.add_argument(
        "--pilot-reclaim-stale-seconds",
        type=int,
        default=7200,
        help="Move running pilot tasks back to pending after this many stale seconds; use 0 to disable",
    )
    parser.add_argument(
        "--pilot-max-tasks",
        type=int,
        default=None,
        help="Optional maximum number of pilot tasks to complete before exiting",
    )
    parser.add_argument(
        "--pilot-max-exams",
        type=int,
        default=None,
        help="Optional maximum number of newly scored exams in pilot mode before exiting",
    )
    parser.add_argument(
        "--pilot-continue-on-task-error",
        action="store_true",
        help="Keep polling after a task failure instead of exiting the pilot with code 1",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens per exam (default: 512)",
    )
    parser.add_argument(
        "--few-shot-examples",
        type=int,
        default=0,
        help="Number of human-labeled reference montages to include before each target exam (default: 0)",
    )
    parser.add_argument(
        "--few-shot-qc-file",
        type=Path,
        default=None,
        help="QC state file to source few-shot exemplars from. Defaults to --qc-file.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=[
            "tagger_json",
            "what_is_this",
            "binary_tag_probe",
            "marker_classifier",
        ],
        default="tagger_json",
        help="Prompt style for the target exam. Use what_is_this for raw freeform debugging.",
    )
    parser.add_argument(
        "--prompt-variant",
        choices=PROMPT_VARIANTS,
        default="baseline",
        help="Prompt decision policy variant. Use recall_tilted for high-recall target triage.",
    )
    parser.add_argument(
        "--probe-tag",
        type=str,
        default=None,
        help="QC tag to test when --prompt-mode=binary_tag_probe",
    )
    parser.add_argument(
        "--target-prompt-override",
        type=str,
        default=None,
        help="Optional target prompt override that still passes the target image to the model.",
    )
    parser.add_argument(
        "--text-only-prompt",
        type=str,
        default=None,
        help="Optional text-only override prompt. When set, no image is passed to the model.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable thinking for Transformers; vLLM QC always disables it.",
    )
    parser.add_argument(
        "--debug-dump-dir",
        type=Path,
        default=None,
        help="Optional directory to write per-exam prompt/response debug JSON files.",
    )
    parser.add_argument(
        "--expected-gpus",
        type=int,
        default=None,
        help="Fail if the job does not see exactly this many GPUs",
    )
    parser.add_argument(
        "--max-memory-per-gpu",
        type=str,
        default=None,
        help='Optional Hugging Face max_memory GPU budget, e.g. "135GiB"',
    )
    parser.add_argument(
        "--cpu-max-memory",
        type=str,
        default=None,
        help='Optional Hugging Face CPU max_memory budget, e.g. "128GiB"',
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow remote model code if the installed transformers build lacks native support",
    )
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    """Execute local auto annotation from parsed arguments."""
    install_shutdown_handlers()
    export_dir = args.export_dir.resolve()
    views_path = args.views.resolve()
    run_file = args.run_file.resolve()
    model_path = args.model_path.resolve() if args.model_path else None
    qc_file = args.qc_file.resolve() if args.qc_file else None
    few_shot_qc_file = (
        args.few_shot_qc_file.resolve() if args.few_shot_qc_file else qc_file
    )
    exam_list_path = args.exam_list.resolve() if args.exam_list else None
    debug_dump_dir = args.debug_dump_dir.resolve() if args.debug_dump_dir else None

    tags_file = resolve_tags_file(
        requested_tags_file=args.tags_file,
        run_file=run_file,
        qc_file=qc_file,
    )

    if not export_dir.exists():
        raise FileNotFoundError(f"export dir not found: {export_dir}")
    if not views_path.exists():
        raise FileNotFoundError(f"views parquet not found: {views_path}")
    if args.skip_existing_annotations and qc_file is None:
        raise ValueError("--skip-existing-annotations requires --qc-file")
    if args.few_shot_examples < 0:
        raise ValueError("--few-shot-examples must be non-negative")
    if args.pilot_poll_seconds <= 0:
        raise ValueError("--pilot-poll-seconds must be positive")
    if args.pilot_max_tasks is not None and args.pilot_max_tasks <= 0:
        raise ValueError("--pilot-max-tasks must be positive")
    if args.pilot_max_exams is not None and args.pilot_max_exams <= 0:
        raise ValueError("--pilot-max-exams must be positive")
    if args.vllm_startup_timeout_seconds <= 0:
        raise ValueError("--vllm-startup-timeout-seconds must be positive")
    if args.vllm_request_timeout_seconds <= 0:
        raise ValueError("--vllm-request-timeout-seconds must be positive")
    if not 0 <= args.vllm_port <= 65535:
        raise ValueError("--vllm-port must be between 0 and 65535")

    model_spec: Any = None
    if args.backend == "vllm":
        from prima.vllm_server import (
            resolve_model_path,
            select_model_spec,
            validate_vllm_runtime,
        )

        if model_path is not None:
            raise ValueError(
                "--backend=vllm selects a pinned registry entry; do not pass --model-path"
            )

        model_registry = args.model_registry.resolve()
        models_dir = args.models_dir.resolve()
        validate_vllm_runtime()
        model_spec = select_model_spec(model_registry, args.model_key)
        model_path = resolve_model_path(model_spec, models_dir)
        if (
            args.expected_gpus is not None
            and args.expected_gpus != model_spec.tensor_parallel_size
        ):
            raise ValueError(
                f"model {model_spec.key!r} requires "
                f"{model_spec.tensor_parallel_size} GPUs, but --expected-gpus="
                f"{args.expected_gpus}"
            )
        revision_suffix = f"@{model_spec.revision[:12]}" if model_spec.revision else ""
        model_label = f"{model_spec.served_model_name}{revision_suffix}"
    else:
        if model_path is None:
            raise ValueError("--backend=transformers requires --model-path")
        if not model_path.exists():
            raise FileNotFoundError(f"model checkpoint not found: {model_path}")
        model_label = infer_model_label(model_path)

    tag_catalog = load_tag_catalog(tags_file)
    if args.prompt_mode in {"binary_tag_probe", "marker_classifier"}:
        if not args.probe_tag:
            raise ValueError(f"--prompt-mode={args.prompt_mode} requires --probe-tag")
        if args.probe_tag not in tag_catalog:
            raise ValueError(
                f"--probe-tag must be one of the allowed tags; got {args.probe_tag!r}"
            )
    views_df = load_views_df(views_path)
    exam_records: list[dict[str, str]] = []
    if args.pilot_queue_dir is None:
        exam_records = load_exam_records(
            views_df=views_df,
            export_dir=export_dir,
            exam_list_path=exam_list_path,
            exam_id=args.exam,
            max_exams=args.max_exams,
            skip_existing_annotations=args.skip_existing_annotations,
            qc_file=qc_file,
        )
        if not exam_records:
            raise RuntimeError("no exams matched the requested selection")

    exemplar_pool = load_few_shot_exemplars(
        qc_file=few_shot_qc_file,
        views_df=views_df,
        export_dir=export_dir,
        preserve_tags={args.probe_tag}
        if args.prompt_mode == "marker_classifier" and args.probe_tag
        else None,
    )
    if args.few_shot_examples > 0 and not exemplar_pool:
        logger.warning(
            "few-shot was requested but no usable exemplars were found in %s",
            few_shot_qc_file,
        )

    payload = load_or_initialize_run_payload(
        run_file=run_file,
        model_label=model_label,
        args=args,
        tag_catalog=tag_catalog,
        model_spec=model_spec,
    )
    if args.backend == "vllm":
        from prima.vllm_server import find_available_loopback_port

        assert model_spec is not None
        assert model_path is not None
        if args.vllm_server_log is not None:
            server_log_path = args.vllm_server_log.resolve()
        else:
            process_label = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
            server_log_path = run_file.with_name(
                f"{run_file.stem}.vllm_{process_label}.log"
            )
        annotator: Any = VLLMVisionAnnotator(
            model_spec=model_spec,
            model_path=model_path,
            port=args.vllm_port or find_available_loopback_port(),
            server_log_path=server_log_path,
            startup_timeout_seconds=args.vllm_startup_timeout_seconds,
            request_timeout_seconds=args.vllm_request_timeout_seconds,
            max_new_tokens=args.max_new_tokens,
            few_shot_examples=args.few_shot_examples,
            few_shot_exemplar_pool=exemplar_pool,
            prompt_mode=args.prompt_mode,
            prompt_variant=args.prompt_variant,
            probe_tag=args.probe_tag,
            target_prompt_override=args.target_prompt_override,
            text_only_prompt=args.text_only_prompt,
            disable_thinking=args.disable_thinking,
            debug_dump_dir=debug_dump_dir,
        )
    else:
        assert model_path is not None
        annotator = LocalVisionAnnotator(
            model_path=model_path,
            expected_gpus=args.expected_gpus,
            max_memory_per_gpu=args.max_memory_per_gpu,
            cpu_max_memory=args.cpu_max_memory,
            max_new_tokens=args.max_new_tokens,
            few_shot_examples=args.few_shot_examples,
            few_shot_exemplar_pool=exemplar_pool,
            prompt_mode=args.prompt_mode,
            prompt_variant=args.prompt_variant,
            probe_tag=args.probe_tag,
            target_prompt_override=args.target_prompt_override,
            text_only_prompt=args.text_only_prompt,
            disable_thinking=args.disable_thinking,
            debug_dump_dir=debug_dump_dir,
            trust_remote_code=args.trust_remote_code,
        )

    try:
        if args.pilot_queue_dir is not None:
            return run_pilot_loop(
                args=args,
                views_df=views_df,
                export_dir=export_dir,
                annotator=annotator,
                tag_catalog=tag_catalog,
                model_label=model_label,
                run_file=run_file,
                payload=payload,
            )

        saved, stats = score_exam_records(
            exam_records=exam_records,
            annotator=annotator,
            tag_catalog=tag_catalog,
            model_label=model_label,
            run_file=run_file,
            payload=payload,
            force_rescore=args.force_rescore,
            desc="auto-qc",
        )
        suggested_exam_count = sum(
            1 for record in saved["exam_suggestions"].values() if record["suggestions"]
        )
        total_suggestions = sum(
            len(record["suggestions"]) for record in saved["exam_suggestions"].values()
        )
        print(f"wrote auto-QC run to {run_file}")
        print(f"  exams newly scored: {int(stats['scored']):,}")
        print(f"  exams skipped from existing run: {int(stats['skipped_existing']):,}")
        print(f"  exams scored: {len(saved['exam_suggestions']):,}")
        print(f"  exams with >=1 suggestion: {suggested_exam_count:,}")
        print(f"  total suggested tags: {total_suggestions:,}")
        return 130 if stats["interrupted"] else 0
    finally:
        close_annotator = getattr(annotator, "close", None)
        if close_annotator is not None:
            close_annotator()


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
