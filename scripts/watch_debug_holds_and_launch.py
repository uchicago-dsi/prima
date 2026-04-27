#!/usr/bin/env python3
"""Watch a sequence of debug holds and launch auto-QC retries inside them."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shlex
import subprocess
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PREFIX = Path("/net/projects2/annawoodard/micromamba/envs/prima")
ENV_PYTHON = ENV_PREFIX / "bin" / "python"
ISOLATED_RUNNER = PROJECT_ROOT / "scripts" / "run_auto_qc_isolated_caches.py"


def utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Watch pending/running debug hold jobs and launch auto_annotate_qc.py "
            "inside them with controlled cache isolation."
        )
    )
    parser.add_argument(
        "--job-id",
        action="append",
        required=True,
        help="Hold job id to watch. Repeat to define retry order.",
    )
    parser.add_argument(
        "--hf-cache-mode",
        action="append",
        required=True,
        help="HF cache mode per attempt: shared, copied, or empty.",
    )
    parser.add_argument(
        "--cache-root-base",
        type=Path,
        required=True,
        help="Base directory for per-attempt isolated caches.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        required=True,
        help="Watcher log file.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=30,
        help="Polling interval while waiting for the next hold.",
    )
    parser.add_argument(
        "--shared-hf-home",
        type=Path,
        default=Path.home() / ".cache" / "huggingface",
    )
    parser.add_argument(
        "--shared-hf-hub",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "hub",
    )
    parser.add_argument(
        "--copy-pattern",
        action="append",
        default=[],
        help="Glob to preseed into isolated HF cache when mode=copied.",
    )
    parser.add_argument(
        "--extra-env",
        action="append",
        default=[],
        help="Extra KEY=VALUE env var passed to the isolated-cache launcher.",
    )
    parser.add_argument(
        "auto_qc_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to auto_annotate_qc.py. Prefix with '--'.",
    )
    return parser


def strip_remainder_delimiter(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def append_log(log_file: Path, message: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a") as f:
        f.write(f"[{utc_now()}] {message}\n")


def query_hold_state(job_id: str) -> tuple[str, str]:
    proc = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T|%N"],
        capture_output=True,
        text=True,
        check=False,
    )
    line = proc.stdout.strip()
    if line:
        state, node = line.split("|", 1)
        return state, node
    proc = subprocess.run(
        ["sacct", "-n", "-P", "-j", job_id, "--format=State,NodeList"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        state, node = line.split("|", 1)
        return state, node
    return "UNKNOWN", ""


def wait_for_running_hold(job_id: str, poll_seconds: int, log_file: Path) -> tuple[str, str] | None:
    while True:
        state, node = query_hold_state(job_id)
        append_log(log_file, f"job_id={job_id} state={state} node={node or '-'}")
        if state == "RUNNING":
            return state, node
        if state.startswith(("CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "COMPLETED")):
            return None
        time.sleep(poll_seconds)


def build_srun_command(
    job_id: str,
    cache_root: Path,
    hf_cache_mode: str,
    shared_hf_home: Path,
    shared_hf_hub: Path,
    copy_patterns: list[str],
    extra_env: list[str],
    auto_qc_args: list[str],
) -> list[str]:
    rendered_auto_qc_args = [
        arg.format(
            job_id=job_id,
            hf_cache_mode=hf_cache_mode,
            cache_root=str(cache_root),
        )
        for arg in auto_qc_args
    ]
    runner_args = [
        str(ENV_PYTHON),
        str(ISOLATED_RUNNER),
        "--cache-root",
        str(cache_root),
        "--hf-cache-mode",
        hf_cache_mode,
        "--shared-hf-home",
        str(shared_hf_home),
        "--shared-hf-hub",
        str(shared_hf_hub),
        "--clear-cache-root",
        "--print-env",
    ]
    for pattern in copy_patterns:
        runner_args.extend(["--copy-pattern", pattern])
    for pair in extra_env:
        runner_args.extend(["--extra-env", pair])
    runner_args.append("--")
    runner_args.extend(rendered_auto_qc_args)
    return [
        "srun",
        "--jobid",
        job_id,
        "--overlap",
        "--ntasks",
        "1",
        *runner_args,
    ]


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    auto_qc_args = strip_remainder_delimiter(list(args.auto_qc_args))
    if not auto_qc_args:
        raise ValueError("must provide auto_annotate_qc.py arguments after '--'")
    if len(args.hf_cache_mode) != len(args.job_id):
        raise ValueError("--hf-cache-mode must be repeated exactly once per --job-id")
    if not ISOLATED_RUNNER.exists():
        raise FileNotFoundError(f"missing isolated-cache runner: {ISOLATED_RUNNER}")
    if not ENV_PYTHON.exists():
        raise FileNotFoundError(f"missing env python: {ENV_PYTHON}")

    append_log(
        args.log_file,
        "starting watcher "
        + " ".join(
            [
                f"job_ids={','.join(args.job_id)}",
                f"hf_modes={','.join(args.hf_cache_mode)}",
                f"cache_root_base={args.cache_root_base}",
            ]
        ),
    )

    for index, (job_id, hf_cache_mode) in enumerate(
        zip(args.job_id, args.hf_cache_mode, strict=True),
        start=1,
    ):
        result = wait_for_running_hold(
            job_id=job_id,
            poll_seconds=args.poll_seconds,
            log_file=args.log_file,
        )
        if result is None:
            append_log(args.log_file, f"job_id={job_id} became unavailable before launch; skipping")
            continue
        _, node = result
        cache_root = args.cache_root_base / f"{job_id}_{hf_cache_mode}"
        command = build_srun_command(
            job_id=job_id,
            cache_root=cache_root,
            hf_cache_mode=hf_cache_mode,
            shared_hf_home=args.shared_hf_home.resolve(),
            shared_hf_hub=args.shared_hf_hub.resolve(),
            copy_patterns=args.copy_pattern,
            extra_env=args.extra_env,
            auto_qc_args=auto_qc_args,
        )
        append_log(
            args.log_file,
            f"launching attempt={index} job_id={job_id} node={node} hf_cache_mode={hf_cache_mode} "
            + "command="
            + " ".join(shlex.quote(part) for part in command),
        )
        proc = subprocess.run(command, cwd=PROJECT_ROOT)
        append_log(
            args.log_file,
            f"attempt={index} job_id={job_id} hf_cache_mode={hf_cache_mode} exit_code={proc.returncode}",
        )
        if proc.returncode == 0:
            append_log(args.log_file, f"attempt={index} succeeded; stopping watcher")
            return 0

    append_log(args.log_file, "all watched holds exhausted without a successful run")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
