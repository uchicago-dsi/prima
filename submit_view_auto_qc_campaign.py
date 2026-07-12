#!/usr/bin/env python3
"""Submit every prepared view auto-QC shard across opportunistic H200 lanes."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

import submitit

from auto_annotate_qc import DEFAULT_MODELS_DIR, DEFAULT_VLLM_MODEL_REGISTRY
from prima.vllm_server import (
    resolve_model_path,
    select_model_spec,
    validate_vllm_runtime,
)
from submit_auto_qc import slurm_gres_spec
from submit_view_auto_qc import ViewAutoQCJob

PARTITION_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def parse_partition_plan(value: str) -> list[str]:
    """Expand ``partition:count`` entries into one partition per shard."""
    partitions: list[str] = []
    for raw_entry in value.split(","):
        entry = raw_entry.strip()
        if not entry or ":" not in entry:
            raise ValueError(
                "partition plan must use comma-separated partition:count entries"
            )
        partition, raw_count = entry.rsplit(":", 1)
        partition = partition.strip()
        if not PARTITION_NAME.fullmatch(partition):
            raise ValueError(f"invalid partition name in plan: {partition!r}")
        try:
            count = int(raw_count)
        except ValueError as error:
            raise ValueError(f"invalid partition count: {raw_count!r}") from error
        if count <= 0:
            raise ValueError("partition counts must be positive")
        partitions.extend([partition] * count)
    return partitions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--name", default="view_auto_qc_candidates")
    parser.add_argument("--partition-plan", required=True)
    parser.add_argument("--qos", default="opportunistic")
    parser.add_argument("--gpuspec", default="nvidia_h200-141gb")
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--cpus-per-task", type=int, default=16)
    parser.add_argument("--mem-gb", type=int, default=128)
    parser.add_argument("--timeout-min", type=int, default=360)
    parser.add_argument("--dependency-job-id", default="")
    parser.add_argument("--model-key", default="qwen35_27b_fp8")
    parser.add_argument(
        "--model-registry", type=Path, default=DEFAULT_VLLM_MODEL_REGISTRY
    )
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--startup-timeout-seconds", type=int, default=1800)
    parser.add_argument("--request-timeout-seconds", type=int, default=600)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt-variant", default="confidence_specificity")
    parser.add_argument("--no-wait", action="store_true")
    return parser.parse_args()


def restricted_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    validate_vllm_runtime()
    campaign_dir = args.campaign_dir.resolve()
    run_dir = args.run_dir.resolve()
    log_dir = args.log_dir.resolve()
    metadata_path = campaign_dir / "campaign.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("prepared view-candidate campaign is missing metadata")
    render_complete_path = campaign_dir / "render_complete.json"
    if not args.dependency_job_id and not render_complete_path.is_file():
        raise FileNotFoundError(
            "render campaign is not validated; provide its validation dependency "
            "or wait for render_complete.json"
        )
    if args.ngpus <= 0 or args.cpus_per_task <= 0 or args.mem_gb <= 0:
        raise ValueError("inference resources must be positive")
    if args.timeout_min <= 0 or args.max_new_tokens <= 0:
        raise ValueError("inference time and token limits must be positive")
    if args.dependency_job_id and not args.dependency_job_id.isdigit():
        raise ValueError("--dependency-job-id must be a numeric Slurm job ID")

    metadata = json.loads(metadata_path.read_text())
    num_shards = int(metadata["inference_shards"])
    partitions = parse_partition_plan(args.partition_plan)
    if len(partitions) != num_shards:
        raise ValueError(
            f"partition plan provides {len(partitions)} lanes for {num_shards} shards"
        )
    manifests = [
        campaign_dir / f"manifest_shard_{shard_index:03d}.parquet"
        for shard_index in range(num_shards)
    ]
    missing_manifests = sum(not path.is_file() for path in manifests)
    if missing_manifests:
        raise FileNotFoundError(
            f"campaign is missing {missing_manifests} shard manifests"
        )

    model_registry = args.model_registry.resolve()
    models_dir = args.models_dir.resolve()
    model_spec = select_model_spec(model_registry, args.model_key)
    if model_spec.tensor_parallel_size != args.ngpus:
        raise ValueError(
            f"model requires {model_spec.tensor_parallel_size} GPUs, but --ngpus={args.ngpus}"
        )
    resolve_model_path(model_spec, models_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    submitit_logs = log_dir / f"{args.name}_{timestamp}" / "submitit_logs"
    submitit_logs.mkdir(parents=True, mode=0o700)
    submission_path = campaign_dir / f"inference_submission_{timestamp}.json"
    jobs = []
    for shard_index, (manifest, partition) in enumerate(zip(manifests, partitions)):
        executor = submitit.AutoExecutor(folder=str(submitit_logs))
        additional = {}
        if args.dependency_job_id:
            additional["dependency"] = f"afterok:{args.dependency_job_id}"
        executor.update_parameters(
            name=f"{args.name}_{shard_index:03d}",
            timeout_min=args.timeout_min,
            slurm_partition=partition,
            tasks_per_node=1,
            cpus_per_task=args.cpus_per_task,
            nodes=1,
            mem_gb=args.mem_gb,
            slurm_qos=args.qos,
            slurm_gres=slurm_gres_spec(args.gpuspec, args.ngpus),
            slurm_use_srun=False,
            slurm_additional_parameters=additional,
        )
        run_file = run_dir / f"shard_{shard_index:03d}.json"
        server_log = run_dir / f"shard_{shard_index:03d}.vllm.log"
        job_args = argparse.Namespace(
            manifest=manifest,
            run_file=run_file,
            model_key=args.model_key,
            model_registry=model_registry,
            models_dir=models_dir,
            vllm_port=0,
            vllm_server_log=server_log,
            startup_timeout_seconds=args.startup_timeout_seconds,
            request_timeout_seconds=args.request_timeout_seconds,
            max_new_tokens=args.max_new_tokens,
            prompt_variant=args.prompt_variant,
            debug_dump_dir=None,
            force_rescore=False,
            expected_gpus=args.ngpus,
        )
        jobs.append(executor.submit(ViewAutoQCJob(job_args)))

    payload = {
        "submitted_at": datetime.now().astimezone().isoformat(),
        "submitit_logs": str(submitit_logs),
        "run_dir": str(run_dir),
        "dependency_job_id": args.dependency_job_id or None,
        "qos": args.qos,
        "gpuspec": args.gpuspec,
        "model_key": args.model_key,
        "jobs": [
            {
                "shard_index": index,
                "partition": partitions[index],
                "job_id": job.job_id,
            }
            for index, job in enumerate(jobs)
        ],
    }
    restricted_json(submission_path, payload)
    print("submitted view auto-QC jobs: " + ",".join(job.job_id for job in jobs))
    print(f"logs: {submitit_logs}")
    print(f"runs: {run_dir}")
    if args.no_wait:
        return 0
    for job in jobs:
        job.result()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
