#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pwd
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


DEFAULT_OUTPUT = "/var/spool/slurm/bootstrap_gpu_running.tsv"
DEFAULT_PARTITIONS = ("gpuq", "gpudev")
DEFAULT_BOOTSTRAP_QOS = "gpu_bootstrap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write cached GPU running counts for Slurm job_submit.lua. "
            "Output format: uid|running_gpu_gpus|bootstrap_inflight_gpus"
        )
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--partitions",
        nargs="+",
        default=list(DEFAULT_PARTITIONS),
        help="GPU partitions to count",
    )
    parser.add_argument(
        "--bootstrap-qos",
        default=DEFAULT_BOOTSTRAP_QOS,
        help="QoS name used for boosted bootstrap jobs",
    )
    return parser.parse_args()


def username_to_uid(username: str) -> int | None:
    try:
        return pwd.getpwnam(username).pw_uid
    except KeyError:
        print(f"warning: no passwd entry for username={username!r}", file=sys.stderr)
        return None


def run_squeue() -> list[str]:
    cmd = [
        "squeue",
        "-h",
        "-t",
        "PD,R,CG",
        "-o",
        "%u|%P|%T|%b|%q",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return [line for line in result.stdout.splitlines() if line.strip()]


def parse_gpu_count(gres: str) -> int:
    if not gres or gres in {"N/A", "(null)"}:
        return 0

    total = 0
    for chunk in gres.split(","):
        if "gpu" not in chunk:
            continue
        parts = chunk.split(":")
        if not parts:
            continue
        try:
            total += int(parts[-1])
        except ValueError:
            continue
    return total


def main() -> int:
    args = parse_args()
    partitions = set(args.partitions)

    running_gpu_gpus: Counter[int] = Counter()
    bootstrap_inflight_gpus: Counter[int] = Counter()

    for line in run_squeue():
        username, partition, state, gres, qos = line.split("|", 4)
        if partition not in partitions:
            continue
        gpu_count = parse_gpu_count(gres)
        if gpu_count <= 0:
            continue

        uid = username_to_uid(username)
        if uid is None:
            continue

        if state in {"RUNNING", "COMPLETING"}:
            running_gpu_gpus[uid] += gpu_count

        if qos == args.bootstrap_qos and state in {"PENDING", "RUNNING", "COMPLETING"}:
            bootstrap_inflight_gpus[uid] += gpu_count

    all_uids = sorted(set(running_gpu_gpus) | set(bootstrap_inflight_gpus))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w",
        dir=str(output_path.parent),
        prefix=output_path.name + ".",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        for uid in all_uids:
            tmp.write(
                f"{uid}|{running_gpu_gpus.get(uid, 0)}|{bootstrap_inflight_gpus.get(uid, 0)}\n"
            )

    os.chmod(tmp_path, 0o644)
    os.replace(tmp_path, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
