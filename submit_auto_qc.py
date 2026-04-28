#!/usr/bin/env python3
"""Submit local auto-QC inference jobs to Slurm via submitit."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path

import submitit

import auto_annotate_qc


class AutoQCJob:
    """Submitit entrypoint for one-process multi-GPU auto-QC inference."""

    def __init__(self, job_args: argparse.Namespace) -> None:
        self.job_args = job_args

    def __call__(self) -> int:
        return auto_annotate_qc.run_from_args(self.job_args)


def build_submit_parser() -> argparse.ArgumentParser:
    """Build the Slurm submission parser."""
    parser = argparse.ArgumentParser(
        description="Submit a local multimodal auto-QC job with submitit.",
        epilog=(
            "Any additional arguments after the Slurm flags are passed through to "
            "auto_annotate_qc.py.\n\n"
            "Example (4xH200):\n"
            "  python submit_auto_qc.py --gpuspec h200 --ngpus 4 --no-wait "
            "--views /path/views_for_qc.parquet --export-dir /path/qc_export "
            "--run-file /path/auto_qc_run.json --model-path /path/Qwen3.5-397B-A17B-FP8\n\n"
            "Example (8xA100):\n"
            "  python submit_auto_qc.py --constraint a100 --gpuspec a100 --ngpus 8 --mem-gb 900 --no-wait "
            "--views /path/views_for_qc.parquet --export-dir /path/qc_export "
            "--run-file /path/auto_qc_run.json --model-path /path/Qwen3.5-397B-A17B-FP8"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--name",
        type=str,
        default="auto_qc",
        help="Run name prefix",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("runs"),
        help="Base directory for submitit logs",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="general",
        help="Slurm partition (default: general)",
    )
    parser.add_argument(
        "--qos",
        type=str,
        default="",
        help="Optional Slurm QOS, e.g. burst",
    )
    parser.add_argument(
        "--constraint",
        type=str,
        default="",
        help='Optional Slurm constraint, e.g. h200, a100, or "[h100|h200]"',
    )
    parser.add_argument(
        "--gpuspec",
        type=str,
        default="h200",
        help='GPU type in Slurm gres, e.g. h200 or a100. Use "any" for generic gpu:<n> requests.',
    )
    parser.add_argument(
        "--ngpus",
        type=int,
        default=4,
        help="GPUs on one node for the model job (default: 4)",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=16,
        help="CPUs for the single inference process (default: 16)",
    )
    parser.add_argument(
        "--mem-gb",
        type=int,
        default=720,
        help="Total host memory to request in GB (default: 720)",
    )
    parser.add_argument(
        "--timeout-min",
        type=int,
        default=240,
        help="Wall time in minutes (default: 240)",
    )
    parser.add_argument(
        "--comment",
        type=str,
        default="",
        help="Optional Slurm comment",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Optional Slurm node exclusions",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit and exit without waiting for completion",
    )
    return parser


def parse_args() -> tuple[argparse.Namespace, argparse.Namespace]:
    """Parse submit args plus the underlying auto-QC args."""
    submit_parser = build_submit_parser()
    submit_args, auto_qc_argv = submit_parser.parse_known_args()
    auto_qc_parser = auto_annotate_qc.build_arg_parser()
    auto_qc_args = auto_qc_parser.parse_args(auto_qc_argv)
    return submit_args, auto_qc_args


def default_max_memory_per_gpu(gpuspec: str) -> str | None:
    """Choose a conservative default max_memory budget per GPU type."""
    normalized = str(gpuspec).strip().lower()
    if normalized in {"any", "auto", "generic", "gpu"}:
        return "72GiB"
    if normalized == "h200":
        return "135GiB"
    if normalized in {"a100", "h100"}:
        return "72GiB"
    if normalized in {"a40", "l40s"}:
        return "44GiB"
    return None


def slurm_gres_spec(gpuspec: str, ngpus: int) -> str:
    """Build the Slurm gres string, supporting generic GPU requests."""
    normalized = str(gpuspec).strip().lower()
    if normalized in {"any", "auto", "generic", "gpu"}:
        return f"gpu:{ngpus}"
    return f"gpu:{gpuspec}:{ngpus}"


def main() -> int:
    submit_args, auto_qc_args = parse_args()

    if submit_args.ngpus <= 0:
        raise ValueError("--ngpus must be positive")
    if auto_qc_args.expected_gpus is None:
        auto_qc_args.expected_gpus = submit_args.ngpus
    if auto_qc_args.max_memory_per_gpu is None:
        auto_qc_args.max_memory_per_gpu = default_max_memory_per_gpu(
            submit_args.gpuspec
        )
    if auto_qc_args.cpu_max_memory is None:
        auto_qc_args.cpu_max_memory = "128GiB"
    exclude_nodes = [
        str(node).strip() for node in submit_args.exclude if str(node).strip()
    ]
    exclude_spec = ",".join(exclude_nodes) if exclude_nodes else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = submit_args.log_dir.resolve() / f"{submit_args.name}_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    log_folder = run_root / "submitit_logs"
    log_folder.mkdir(parents=True, exist_ok=True)

    auto_qc_args = copy.deepcopy(auto_qc_args)
    auto_qc_args.views = auto_qc_args.views.resolve()
    auto_qc_args.export_dir = auto_qc_args.export_dir.resolve()
    auto_qc_args.model_path = auto_qc_args.model_path.resolve()
    auto_qc_args.run_file = auto_qc_args.run_file.resolve()
    if auto_qc_args.qc_file:
        auto_qc_args.qc_file = auto_qc_args.qc_file.resolve()
    if auto_qc_args.tags_file:
        auto_qc_args.tags_file = auto_qc_args.tags_file.resolve()
    if auto_qc_args.exam_list:
        auto_qc_args.exam_list = auto_qc_args.exam_list.resolve()

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
        slurm_exclude=exclude_spec,
        slurm_gres=slurm_gres_spec(submit_args.gpuspec, submit_args.ngpus),
    )

    job = executor.submit(AutoQCJob(auto_qc_args))
    print(f"Submitted auto-QC job {job.job_id}")
    print(f"  logs: {log_folder}")
    print(f"  run file: {auto_qc_args.run_file}")
    resource_line = (
        f"  resources: partition={submit_args.partition} "
        f"gpus={submit_args.ngpus}x{submit_args.gpuspec}"
    )
    if submit_args.qos:
        resource_line += f" qos={submit_args.qos}"
    if submit_args.constraint:
        resource_line += f" constraint={submit_args.constraint}"
    print(resource_line)

    if submit_args.no_wait:
        return 0

    result = job.result()
    print(f"Job {job.job_id} finished with exit code {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
