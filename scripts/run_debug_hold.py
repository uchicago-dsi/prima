#!/usr/bin/env python3
"""Run inside a Slurm allocation and yield to newer running holds."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hold a Slurm allocation unless a newer hold in the same group starts running.",
    )
    parser.add_argument("--group-prefix", required=True)
    parser.add_argument("--sleep-seconds", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--hold-index", type=int, default=0)
    parser.add_argument("--hold-count", type=int, default=0)
    return parser


def list_running_group_jobs(user: str, group_prefix: str) -> list[tuple[str, str, str]]:
    proc = subprocess.run(
        ["squeue", "-h", "-u", user, "-o", "%i|%j|%T|%S"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[tuple[str, str, str]] = []
    for line in proc.stdout.splitlines():
        job_id, job_name, state, start_time = line.split("|", 3)
        if not job_name.startswith(group_prefix):
            continue
        if state not in {"RUNNING", "COMPLETING"}:
            continue
        rows.append((job_id, job_name, start_time))
    return rows


def main() -> int:
    args = build_parser().parse_args()
    if args.sleep_seconds <= 0:
        raise ValueError("--sleep-seconds must be positive")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")

    job_id = os.environ["SLURM_JOB_ID"]
    user = os.environ.get("USER")
    if not user:
        raise RuntimeError("USER is not set")

    print(socket.gethostname(), flush=True)
    subprocess.run(["date"], check=False)
    subprocess.run(["nvidia-smi", "-L"], check=False)
    print(
        f"HOLD_INDEX={args.hold_index}/{args.hold_count} GROUP_PREFIX={args.group_prefix} "
        f"SLEEP_SECONDS={args.sleep_seconds} POLL_SECONDS={args.poll_seconds}",
        flush=True,
    )

    deadline = time.monotonic() + args.sleep_seconds
    last_status_log = 0.0
    cancelled_older_job_ids: set[str] = set()
    while True:
        now = time.monotonic()
        if now >= deadline:
            print(f"hold {job_id} reached planned sleep deadline; exiting", flush=True)
            return 0

        running_jobs = list_running_group_jobs(user=user, group_prefix=args.group_prefix)
        if running_jobs:
            newest_job_id, newest_job_name, newest_start_time = max(
                running_jobs,
                key=lambda row: (row[2], int(row[0])),
            )
            if newest_job_id != job_id:
                print(
                    f"newer running hold detected: job_id={newest_job_id} "
                    f"job_name={newest_job_name} start_time={newest_start_time}; "
                    f"yielding older hold {job_id}",
                    flush=True,
                )
                return 0
            older_job_ids = sorted(
                row[0] for row in running_jobs if row[0] != job_id and row[0] not in cancelled_older_job_ids
            )
            if older_job_ids:
                print(
                    f"newest running hold {job_id} cancelling older holds: "
                    + ",".join(older_job_ids),
                    flush=True,
                )
                subprocess.run(["scancel", *older_job_ids], check=False)
                cancelled_older_job_ids.update(older_job_ids)

        if now - last_status_log >= 600:
            remaining = int(deadline - now)
            print(f"hold {job_id} healthy; remaining_seconds={remaining}", flush=True)
            last_status_log = now

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
