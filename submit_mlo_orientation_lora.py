#!/usr/bin/env python3
"""Submit one restartable MLO orientation LoRA pilot to Slurm."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path

import submitit

from qc import train_mlo_orientation_lora


class MLOOrientationJob:
    """Submitit entrypoint for one single-GPU adapter run."""

    def __init__(self, job_args: argparse.Namespace) -> None:
        self.job_args = job_args

    def __call__(self) -> int:
        train_mlo_orientation_lora.run_from_args(self.job_args)
        return 0


def build_submit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="mlo_orientation_lora")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--partition", default="zhoulabq,catherineq,siweiq")
    parser.add_argument("--qos", default="opportunistic")
    parser.add_argument("--gpuspec", default="nvidia_h200-141gb")
    parser.add_argument("--cpus-per-task", type=int, default=16)
    parser.add_argument("--mem-gb", type=int, default=128)
    parser.add_argument("--timeout-min", type=int, default=90)
    parser.add_argument("--no-wait", action="store_true")
    return parser


def parse_args() -> tuple[argparse.Namespace, argparse.Namespace]:
    submit_parser = build_submit_parser()
    submit_args, training_argv = submit_parser.parse_known_args()
    training_args = train_mlo_orientation_lora.build_arg_parser().parse_args(
        training_argv
    )
    return submit_args, training_args


def main() -> int:
    submit_args, training_args = parse_args()
    if submit_args.cpus_per_task <= 0 or submit_args.mem_gb <= 0:
        raise ValueError("CPU and memory requests must be positive")
    if submit_args.timeout_min <= 0:
        raise ValueError("--timeout-min must be positive")
    if submit_args.qos != "opportunistic":
        raise ValueError(
            "this submitter is restricted to the typed opportunistic route"
        )
    if submit_args.gpuspec != "nvidia_h200-141gb":
        raise ValueError("this pilot requires the typed nvidia_h200-141gb resource")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = submit_args.log_dir.resolve() / f"{submit_args.name}_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=False)
    log_dir = run_root / "submitit_logs"
    log_dir.mkdir(mode=0o700)

    job_args = copy.deepcopy(training_args)
    job_args.manifest = job_args.manifest.resolve()
    job_args.model_path = job_args.model_path.resolve()
    job_args.output_dir = job_args.output_dir.resolve()
    executor = submitit.AutoExecutor(folder=str(log_dir))
    executor.update_parameters(
        name=submit_args.name,
        timeout_min=submit_args.timeout_min,
        slurm_partition=submit_args.partition,
        slurm_qos=submit_args.qos,
        slurm_gres=f"gpu:{submit_args.gpuspec}:1",
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=submit_args.cpus_per_task,
        mem_gb=submit_args.mem_gb,
        slurm_use_srun=False,
    )
    job = executor.submit(MLOOrientationJob(job_args))
    print(f"Submitted MLO orientation LoRA job {job.job_id}")
    print(f"  logs: {log_dir}")
    print(f"  output: {job_args.output_dir}")
    if not submit_args.no_wait:
        print(f"  result: {job.result()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
