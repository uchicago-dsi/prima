#!/usr/bin/env python3
"""Submit individual-view auto-QC inference to Slurm."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime

import submitit

from prima.vllm_server import (
    resolve_model_path,
    select_model_spec,
    validate_vllm_runtime,
)
from qc import run_view_auto_qc
from submit_auto_qc import build_submit_parser, slurm_gres_spec


class ViewAutoQCJob:
    def __init__(self, job_args: argparse.Namespace) -> None:
        self.job_args = job_args

    def __call__(self) -> int:
        return run_view_auto_qc.run_from_args(self.job_args)


def parse_args() -> tuple[argparse.Namespace, argparse.Namespace]:
    submit_parser = build_submit_parser()
    submit_parser.description = "Submit individual-view auto-QC to Slurm."
    submit_parser.epilog = (
        "Additional arguments are parsed by qc/run_view_auto_qc.py. "
        "Pass --manifest and --run-file after the Slurm options."
    )
    submit_args, view_argv = submit_parser.parse_known_args()
    view_args = run_view_auto_qc.build_arg_parser().parse_args(view_argv)
    return submit_args, view_args


def main() -> int:
    submit_args, view_args = parse_args()
    if submit_args.ngpus <= 0:
        raise ValueError("--ngpus must be positive")
    view_args.target = run_view_auto_qc.normalize_view_qc_target(view_args.target)
    run_view_auto_qc.load_target_prompt(
        view_args.target_prompt_file, target=view_args.target
    )
    validate_vllm_runtime()
    spec = select_model_spec(view_args.model_registry.resolve(), view_args.model_key)
    if spec.tensor_parallel_size != submit_args.ngpus:
        raise ValueError(
            f"model requires {spec.tensor_parallel_size} GPUs, "
            f"but --ngpus={submit_args.ngpus}"
        )
    resolve_model_path(spec, view_args.models_dir.resolve())
    if view_args.expected_gpus is None:
        view_args.expected_gpus = submit_args.ngpus

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = submit_args.log_dir.resolve() / f"{submit_args.name}_{timestamp}"
    log_folder = run_root / "submitit_logs"
    log_folder.mkdir(parents=True, exist_ok=True)

    view_args = copy.deepcopy(view_args)
    view_args.manifest = view_args.manifest.resolve()
    view_args.run_file = view_args.run_file.resolve()
    view_args.target_prompt_file = view_args.target_prompt_file.resolve()
    view_args.model_registry = view_args.model_registry.resolve()
    view_args.models_dir = view_args.models_dir.resolve()
    if view_args.vllm_server_log:
        view_args.vllm_server_log = view_args.vllm_server_log.resolve()
    if view_args.debug_dump_dir:
        view_args.debug_dump_dir = view_args.debug_dump_dir.resolve()

    exclude = [str(node).strip() for node in submit_args.exclude if str(node).strip()]
    executor = submitit.AutoExecutor(folder=str(log_folder))
    executor.update_parameters(
        name=submit_args.name,
        timeout_min=submit_args.timeout_min,
        slurm_partition=submit_args.partition,
        tasks_per_node=1,
        cpus_per_task=submit_args.cpus_per_task,
        nodes=1,
        mem_gb=submit_args.mem_gb,
        slurm_constraint=submit_args.constraint or None,
        slurm_qos=submit_args.qos or None,
        slurm_comment=submit_args.comment or None,
        slurm_exclude=",".join(exclude) if exclude else None,
        slurm_gres=slurm_gres_spec(submit_args.gpuspec, submit_args.ngpus),
        slurm_use_srun=not submit_args.no_srun,
    )
    job = executor.submit(ViewAutoQCJob(view_args))
    print(f"Submitted view auto-QC job {job.job_id}")
    print(f"  logs: {log_folder}")
    print(f"  run file: {view_args.run_file}")
    if submit_args.no_wait:
        return 0
    result = job.result()
    print(f"Job {job.job_id} finished with exit code {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
