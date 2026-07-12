#!/usr/bin/env python3
"""Submit restart-safe CPU render shards and their validator to Slurm."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import submitit

from qc import render_view_candidates, validate_view_candidate_render


class ViewCandidateRenderJob:
    def __init__(self, job_args: argparse.Namespace) -> None:
        self.job_args = job_args

    def __call__(self) -> int:
        return render_view_candidates.run_from_args(self.job_args)


class ViewCandidateRenderValidationJob:
    def __init__(self, campaign_dir: Path) -> None:
        self.campaign_dir = campaign_dir

    def __call__(self) -> int:
        return validate_view_candidate_render.run_from_args(
            argparse.Namespace(campaign_dir=self.campaign_dir)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--name", default="view_candidate_render")
    parser.add_argument("--partition", default="tier1q")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem-gb", type=int, default=32)
    parser.add_argument("--timeout-min", type=int, default=360)
    parser.add_argument("--temp-root", type=Path, default=None)
    parser.add_argument("--shard-indices", nargs="*", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
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
    campaign_dir = args.campaign_dir.resolve()
    raw_root = args.raw_root.resolve()
    log_dir = args.log_dir.resolve()
    metadata_path = campaign_dir / "campaign.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("prepared view-candidate campaign is missing metadata")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw DICOM root not found: {raw_root}")
    if args.cpus_per_task <= 0 or args.mem_gb <= 0 or args.timeout_min <= 0:
        raise ValueError("render resources must be positive")
    temp_root = args.temp_root.resolve() if args.temp_root else None
    if temp_root is not None and not temp_root.is_dir():
        raise FileNotFoundError(f"render temporary directory not found: {temp_root}")

    metadata = json.loads(metadata_path.read_text())
    num_shards = int(metadata["render_shards"])
    if args.validate_only and args.shard_indices:
        raise ValueError("--validate-only cannot be combined with --shard-indices")
    shard_indices = (
        list(range(num_shards))
        if args.shard_indices is None
        else sorted(set(args.shard_indices))
    )
    if any(index < 0 or index >= num_shards for index in shard_indices):
        raise ValueError(f"render shard indices must be between 0 and {num_shards - 1}")
    if not args.validate_only and not shard_indices:
        raise ValueError("at least one render shard index is required")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = campaign_dir / f"render_submission_{timestamp}.json"
    run_root = log_dir / f"{args.name}_{timestamp}"
    submitit_logs = run_root / "submitit_logs"
    submitit_logs.mkdir(parents=True, mode=0o700)

    jobs = []
    executor = submitit.SlurmExecutor(folder=str(submitit_logs))
    for shard_index in [] if args.validate_only else shard_indices:
        executor.update_parameters(
            job_name=f"{args.name}_{shard_index:03d}",
            time=args.timeout_min,
            partition=args.partition,
            ntasks_per_node=1,
            cpus_per_task=args.cpus_per_task,
            nodes=1,
            mem=f"{args.mem_gb}G",
        )
        job_args = argparse.Namespace(
            campaign_dir=campaign_dir,
            raw_root=raw_root,
            shard_index=shard_index,
            temp_root=temp_root,
            resume=True,
        )
        jobs.append(executor.submit(ViewCandidateRenderJob(job_args)))

    submit_validator = args.validate_only or shard_indices == list(range(num_shards))
    validation_job = None
    if submit_validator:
        validation_parameters = {
            "job_name": f"{args.name}_validate",
            "time": 120,
            "partition": args.partition,
            "ntasks_per_node": 1,
            "cpus_per_task": 2,
            "nodes": 1,
            "mem": "16G",
        }
        if jobs:
            validation_parameters["dependency"] = "afterok:" + ":".join(
                job.job_id for job in jobs
            )
        validation_executor = submitit.SlurmExecutor(folder=str(submitit_logs))
        validation_executor.update_parameters(**validation_parameters)
        validation_job = validation_executor.submit(
            ViewCandidateRenderValidationJob(campaign_dir)
        )
    payload = {
        "submitted_at": datetime.now().astimezone().isoformat(),
        "run_root": str(run_root),
        "partition": args.partition,
        "render_job_ids": [job.job_id for job in jobs],
        "validation_job_id": None if validation_job is None else validation_job.job_id,
    }
    restricted_json(submission_path, payload)
    if jobs:
        print("submitted render jobs: " + ",".join(payload["render_job_ids"]))
    if validation_job is not None:
        print(f"submitted validation job: {validation_job.job_id}")
    print(f"logs: {submitit_logs}")
    if args.no_wait:
        return 0
    for job in jobs:
        job.result()
    if validation_job is not None:
        validation_job.result()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
