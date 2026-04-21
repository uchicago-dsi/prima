#!/usr/bin/env python3
"""Submit Slurm debug-hold jobs as a dependency chain or a rolling pool."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def parse_walltime(walltime: str) -> int:
    parts = walltime.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"walltime must be HH:MM:SS, got {walltime!r}")
    hours, minutes, seconds = (int(part) for part in parts)
    total_seconds = hours * 3600 + minutes * 60 + seconds
    if total_seconds <= 0:
        raise ValueError("walltime must be positive")
    return total_seconds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit Slurm debug-hold jobs as a chain or rolling pool.",
    )
    parser.add_argument("--name", default="prima_397b_debug_hold")
    parser.add_argument(
        "--mode",
        choices=("rolling_pool", "chain"),
        default="rolling_pool",
        help=(
            "chain submits afterany dependencies; rolling_pool submits all jobs "
            "immediately and older running holds yield if a newer one starts."
        ),
    )
    parser.add_argument("--partition", default="general")
    parser.add_argument("--qos", default="normal")
    parser.add_argument("--constraint", default="h200")
    parser.add_argument("--gpuspec", default="h200")
    parser.add_argument("--ngpus", type=int, default=4)
    parser.add_argument("--cpus-per-task", type=int, default=16)
    parser.add_argument("--mem", default="720G")
    parser.add_argument("--walltime", default="12:00:00")
    parser.add_argument(
        "--reserve-minutes",
        type=int,
        default=20,
        help="Leave this many minutes unused before walltime.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=4,
        help="Number of chained hold jobs to submit.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/net/projects2/annawoodard/qc_redo/slurm_debug_holds"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.ngpus <= 0:
        raise ValueError("--ngpus must be positive")
    if args.reserve_minutes < 0:
        raise ValueError("--reserve-minutes must be non-negative")

    walltime_seconds = parse_walltime(args.walltime)
    sleep_seconds = walltime_seconds - args.reserve_minutes * 60
    if sleep_seconds <= 0:
        raise ValueError(
            "--reserve-minutes leaves no time to hold the allocation; "
            f"walltime={args.walltime} reserve_minutes={args.reserve_minutes}"
        )

    args.log_dir.mkdir(parents=True, exist_ok=True)

    runner_path = (Path(__file__).resolve().parent / "run_debug_hold.py").resolve()
    if not runner_path.exists():
        raise FileNotFoundError(f"missing runner script: {runner_path}")

    previous_job_id: str | None = None
    submitted_job_ids: list[str] = []
    group_prefix = f"{args.name}_"
    for idx in range(1, args.count + 1):
        job_name = f"{args.name}_{idx:02d}of{args.count}"
        wrap_cmd = " ".join(
            [
                shlex.quote(sys.executable),
                shlex.quote(str(runner_path)),
                f"--group-prefix={shlex.quote(group_prefix)}",
                f"--sleep-seconds={sleep_seconds}",
                "--poll-seconds=15",
                f"--hold-index={idx}",
                f"--hold-count={args.count}",
            ]
        )
        cmd = [
            "sbatch",
            "--parsable",
            f"--job-name={job_name}",
            f"--partition={args.partition}",
            f"--qos={args.qos}",
            f"--constraint={args.constraint}",
            f"--gres=gpu:{args.gpuspec}:{args.ngpus}",
            f"--cpus-per-task={args.cpus_per_task}",
            f"--mem={args.mem}",
            f"--time={args.walltime}",
            f"--output={args.log_dir}/%j.out",
            f"--error={args.log_dir}/%j.err",
            f"--wrap={wrap_cmd}",
        ]
        if args.mode == "chain" and previous_job_id is not None:
            cmd.append(f"--dependency=afterany:{previous_job_id}")
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        job_id = proc.stdout.strip().split(";")[0]
        submitted_job_ids.append(job_id)
        previous_job_id = job_id

    print(f"submitted {len(submitted_job_ids)} hold jobs in mode={args.mode}")
    print(f"sleep_seconds={sleep_seconds}")
    for idx, job_id in enumerate(submitted_job_ids, start=1):
        predecessor = submitted_job_ids[idx - 2] if args.mode == "chain" and idx > 1 else "-"
        print(f"{idx:02d} {job_id} afterany={predecessor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
