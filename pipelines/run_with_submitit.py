#!/usr/bin/env python3
"""Submitit launcher for cleandift-t (inspired by dynamicdino)."""

from __future__ import annotations

import argparse
import copy
import importlib
import os
from datetime import datetime
from pathlib import Path
from typing import List

import submitit
from cleandiftt.utils.config import load_config_structured
from cleandiftt.utils.summary import aggregate_folds


def _module_name_from_script(script: str) -> str:
    script_path = Path(script)
    if script_path.suffix == ".py":
        script_path = script_path.with_suffix("")
    return script_path.as_posix().replace("/", ".")


def _build_submitit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Submitit launcher", allow_abbrev=False)
    parser.add_argument("script", type=str, default="train.py", help="Training script")
    parser.add_argument("--name", type=str, default=None, help="Run name override")
    parser.add_argument(
        "--log_dir", type=str, default="runs", help="Base directory for logs"
    )
    parser.add_argument("--ngpus", type=int, default=4, help="GPUs per node")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--cpus_per_task", type=int, default=6)
    parser.add_argument("--mem_gb_per_gpu", type=int, default=60)
    parser.add_argument(
        "--timeout", type=int, default=12 * 60, help="Wall time minutes"
    )
    parser.add_argument("--partition", type=str, default="general")
    parser.add_argument("--constraint", type=str, default="")
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--gpuspec", type=str, default="", help="GPU type, e.g., a40")
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--fold", type=int, default=None, help="Single fold to run")
    parser.add_argument("--folds", type=str, default=None, help="Comma list or 'all'")
    parser.add_argument(
        "--local", action="store_true", help="Run locally with multiprocessing executor"
    )
    parser.add_argument(
        "--no_wait",
        action="store_true",
        help="Submit jobs and exit immediately (skip aggregation)",
    )
    return parser


def parse_args():
    submit_parser = _build_submitit_parser()
    submit_args, script_argv = submit_parser.parse_known_args()

    module_name = _module_name_from_script(submit_args.script)
    parent_module = importlib.import_module(module_name)
    if not hasattr(parent_module, "get_args_parser"):
        raise AttributeError(
            f"{submit_args.script} must define get_args_parser() for submitit integration"
        )
    script_parser = parent_module.get_args_parser()
    script_args = script_parser.parse_args(script_argv)
    return submit_args, script_args, module_name


def _parse_fold_selection(
    spec: str | None, single: int | None, total: int
) -> List[int]:
    if spec is None:
        return [single if single is not None else 0]
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(total))
    folds: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 0 or value >= total:
            raise ValueError(f"Fold {value} out of range [0,{total})")
        folds.append(value)
    return folds or [single if single is not None else 0]


def _get_init_file(run_root: Path) -> Path:
    init_dir = run_root / "ddp_init"
    init_dir.mkdir(parents=True, exist_ok=True)
    token = datetime.now().strftime("%H%M%S%f")
    init_file = init_dir / f"init_{token}"
    if init_file.exists():
        init_file.unlink()
    return init_file


class SubmititTrainer:
    def __init__(self, module_name: str, args_namespace):
        self.module_name = module_name
        self.args = args_namespace

    def __call__(self):
        parent_module = importlib.import_module(self.module_name)
        importlib.reload(parent_module)
        self._setup_env()
        parent_module.train(self.args)

    def _setup_env(self):
        job_env = submitit.JobEnvironment()
        self.args.output_dir = str(self.args.output_dir).replace(
            "%j", str(job_env.job_id)
        )
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        os.environ["LOCAL_RANK"] = str(job_env.local_rank)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["WORLD_SIZE"] = str(job_env.num_tasks)

    def checkpoint(self):
        run_root = Path(
            getattr(self.args, "run_root", Path(self.args.output_dir).parent)
        )
        self.args.dist_url = _get_init_file(run_root).as_uri()
        return submitit.helpers.DelayedSubmission(
            SubmititTrainer(self.module_name, self.args)
        )


def main():
    submit_args, script_args, module_name = parse_args()

    config = load_config_structured(script_args.config)
    total_folds = int(getattr(config.training, "num_folds", 1))
    config_stem = Path(script_args.config).stem
    run_name = submit_args.name or config_stem
    analysis_label = script_args.analysis_label or config.analysis_label or config_stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(submit_args.log_dir) / f"{run_name}_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    folds = _parse_fold_selection(submit_args.folds, submit_args.fold, total_folds)

    log_folder = run_root / "submitit_logs"
    log_folder.mkdir(parents=True, exist_ok=True)
    if submit_args.local:
        executor = submitit.AutoExecutor(folder=str(log_folder), cluster="local")
        executor.update_parameters(
            timeout_min=submit_args.timeout,
            gpus_per_node=submit_args.ngpus,
            tasks_per_node=submit_args.ngpus,
            cpus_per_task=submit_args.cpus_per_task,
        )
    else:
        executor = submitit.AutoExecutor(folder=str(log_folder))
        mem_gb = submit_args.mem_gb_per_gpu * submit_args.ngpus
        gres = (
            f"gpu:{submit_args.gpuspec}:{submit_args.ngpus}"
            if submit_args.gpuspec
            else f"gpu:{submit_args.ngpus}"
        )
        executor.update_parameters(
            timeout_min=submit_args.timeout,
            slurm_partition=submit_args.partition,
            gpus_per_node=submit_args.ngpus,
            tasks_per_node=submit_args.ngpus,
            cpus_per_task=submit_args.cpus_per_task,
            nodes=submit_args.nodes,
            mem_gb=mem_gb,
            slurm_constraint=submit_args.constraint or None,
            slurm_comment=submit_args.comment or None,
            slurm_exclude=submit_args.exclude or None,
            slurm_gres=gres,
        )

    jobs = []
    for fold in folds:
        job_args = copy.deepcopy(script_args)
        job_args.fold = fold
        job_args.run_name = run_name
        job_args.analysis_label = analysis_label
        job_args.output_dir = str(run_root / f"fold_{fold:02d}" / "%j")
        job_args.dist_url = _get_init_file(run_root).as_uri()
        job_args.run_root = str(run_root)
        job = SubmititTrainer(module_name, job_args)
        jobs.append(executor.submit(job))

    print(f"Submitted {len(jobs)} job(s) under {run_root}")
    for fold, job in zip(folds, jobs):
        print(f"  fold {fold}: job id {job.job_id}")

    if not submit_args.no_wait:
        print("Waiting for jobs to finish...")
        for job in jobs:
            job.result()
        summary = aggregate_folds(run_root)
        if summary:
            num_folds = summary.get("num_folds", len(folds))
            print(f"Cross-fold summary ({num_folds} folds):")
            for key, value in summary.items():
                if key == "num_folds":
                    continue
                print(f"  {key}: {value:.4f}")
        else:
            print("No fold summaries found yet; skipping aggregation.")


if __name__ == "__main__":
    main()
