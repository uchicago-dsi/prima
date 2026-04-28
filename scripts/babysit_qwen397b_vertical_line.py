#!/usr/bin/env python3
"""Timed babysitter for the Qwen3.5 397B vertical-line experiment campaign."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import subprocess
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PREFIX = Path("/net/projects2/annawoodard/micromamba/envs/prima")
ENV_PYTHON = ENV_PREFIX / "bin" / "python"
SUBMIT_SCRIPT = PROJECT_ROOT / "submit_auto_qc.py"

DEFAULT_SUBMITIT_RUNS = Path("/net/projects2/annawoodard/qc_redo/submitit_runs")
DEFAULT_RUNS = Path("/net/projects2/annawoodard/qc_redo/auto_qc_runs")
DEFAULT_DEBUG = Path("/net/projects2/annawoodard/qc_redo/auto_qc_debug")
DEFAULT_LOG = Path("/net/projects2/annawoodard/qc_redo/interactive_debug/qwen397b_campaign_babysitter.log")
DEFAULT_STATE = Path("/net/projects2/annawoodard/qc_redo/interactive_debug/qwen397b_campaign_babysitter_state.json")

ACTIVE_STATES = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "SUSPENDED",
    "REQUEUED",
}
FAIL_STATES_PREFIXES = ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY")


@dataclass
class RunRecord:
    kind: str
    resource: str
    name: str
    run_root: Path
    log_dir: Path
    job_id: str
    state: str
    start: str | None
    end: str | None
    exit_code: str | None
    submit_time: datetime


def now_local() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_log(log_file: Path, message: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a") as f:
        f.write(f"[{now_local()}] {message}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot babysitter for the Qwen3.5 397B vertical-line campaign. "
            "Intended to be run from a timer."
        )
    )
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--submitit-runs-dir", type=Path, default=DEFAULT_SUBMITIT_RUNS)
    parser.add_argument("--auto-qc-runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--auto-qc-debug-dir", type=Path, default=DEFAULT_DEBUG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pending-fallback-hours", type=float, default=4.0)
    return parser.parse_args()


def load_state(state_file: Path) -> dict[str, Any]:
    if not state_file.exists():
        return {"history": []}
    with open(state_file) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid state file payload: {state_file}")
    payload.setdefault("history", [])
    return payload


def save_state(state_file: Path, payload: dict[str, Any]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(payload, f, indent=2)


def classify_failure(log_path: Path) -> str:
    if not log_path.exists():
        return "missing_log"
    text = log_path.read_text(errors="replace")
    if "Job has timed out" in text or "timed-out and not checkpointable" in text:
        return "timeout"
    if (
        "deep_gemm::DGException" in text
        or "device-side assert triggered" in text
        or "first non-finite tensor" in text
    ):
        return "fp8_runtime"
    if "No such file or directory" in text or "FileNotFoundError" in text:
        return "path_or_dependency"
    return "unknown"


def query_sacct_state(job_id: str) -> tuple[str, str | None, str | None, str | None]:
    proc = subprocess.run(
        [
            "sacct",
            "-j",
            job_id,
            "--format=State,Start,End,ExitCode",
            "-n",
            "-P",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        state, start, end, exit_code = line.split("|", 3)
        return state.strip(), start.strip() or None, end.strip() or None, exit_code.strip() or None
    return "UNKNOWN", None, None, None


def query_squeue_state(job_id: str) -> str | None:
    proc = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        capture_output=True,
        text=True,
        check=False,
    )
    state = proc.stdout.strip()
    return state or None


def effective_state(job_id: str) -> tuple[str, str | None, str | None, str | None]:
    state, start, end, exit_code = query_sacct_state(job_id)
    if state in {"UNKNOWN", ""}:
        queue_state = query_squeue_state(job_id)
        if queue_state:
            state = queue_state
    elif state == "PENDING":
        queue_state = query_squeue_state(job_id)
        if queue_state:
            state = queue_state
    return state, start, end, exit_code


def parse_submit_time(run_root: Path) -> datetime:
    match = re.search(r"_(\d{8}_\d{6})$", run_root.name)
    if match is None:
        return datetime.fromtimestamp(run_root.stat().st_mtime)
    return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")


def detect_kind_and_resource(name: str) -> tuple[str, str] | None:
    lowered = name.lower()
    if "qwen397b_fp8_vertical_line" not in lowered:
        return None
    if "oneexam" in lowered:
        kind = "oneexam"
    elif "reviewed33" in lowered:
        kind = "reviewed33"
    else:
        return None

    if "a100" in lowered:
        resource = "a100"
    else:
        resource = "h200"
    return kind, resource


def list_campaign_runs(submitit_runs_dir: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    for run_root in sorted(submitit_runs_dir.glob("auto_qc_qwen397b_fp8_vertical_line*")):
        submitit_logs = run_root / "submitit_logs"
        if not submitit_logs.exists():
            continue
        submitted = sorted(submitit_logs.glob("*_submitted.pkl"))
        if not submitted:
            continue
        job_id = submitted[0].name.split("_", 1)[0]
        detected = detect_kind_and_resource(run_root.name)
        if detected is None:
            continue
        kind, resource = detected
        state, start, end, exit_code = effective_state(job_id)
        records.append(
            RunRecord(
                kind=kind,
                resource=resource,
                name=run_root.name,
                run_root=run_root,
                log_dir=submitit_logs,
                job_id=job_id,
                state=state,
                start=start,
                end=end,
                exit_code=exit_code,
                submit_time=parse_submit_time(run_root),
            )
        )
    records.sort(key=lambda record: record.submit_time)
    return records


def latest(records: list[RunRecord], *, kind: str) -> RunRecord | None:
    candidates = [record for record in records if record.kind == kind]
    return candidates[-1] if candidates else None


def active(records: list[RunRecord], *, kind: str | None = None) -> list[RunRecord]:
    selected = records
    if kind is not None:
        selected = [record for record in records if record.kind == kind]
    return [record for record in selected if record.state in ACTIVE_STATES]


def completed(records: list[RunRecord], *, kind: str) -> list[RunRecord]:
    return [record for record in records if record.kind == kind and record.state == "COMPLETED"]


def failure_records(records: list[RunRecord], *, kind: str) -> list[RunRecord]:
    return [
        record
        for record in records
        if record.kind == kind and record.state.startswith(FAIL_STATES_PREFIXES)
    ]


def should_submit_a100_fallback(record: RunRecord, *, threshold_hours: float) -> bool:
    if record.resource != "h200":
        return False
    if record.state != "PENDING":
        return False
    age = datetime.now() - record.submit_time
    return age >= timedelta(hours=threshold_hours)


def build_oneexam_submit_cmd(*, resource: str, timeout_min: int) -> list[str]:
    timestamp = datetime.now().strftime("%Y%m%d")
    name = f"auto_qc_qwen397b_fp8_vertical_line_oneexam_bf16experts_{resource}_t{timeout_min}"
    run_file = (
        DEFAULT_RUNS
        / f"qwen397b_fp8_vertical_line_oneexam_bf16experts_{resource}_t{timeout_min}_{timestamp}.json"
    )
    debug_dir = (
        DEFAULT_DEBUG
        / f"qwen397b_fp8_vertical_line_oneexam_bf16experts_{resource}_t{timeout_min}_{timestamp}"
    )
    if resource == "a100":
        gpuspec = "a100"
        ngpus = "8"
        constraint = "a100"
        mem_gb = "900"
    else:
        gpuspec = "h200"
        ngpus = "4"
        constraint = "h200"
        mem_gb = "720"

    return [
        "bash",
        "-lc",
        " ".join(
            [
                "export PRIMA_QWEN35_FP8_FORCE_BF16_EXPERTS=1 CUDA_LAUNCH_BLOCKING=1 PYTORCH_SHOW_CPP_STACKTRACES=1;",
                f"micromamba run -p {ENV_PREFIX} python {SUBMIT_SCRIPT}",
                f"--name {name}",
                "--log-dir /net/projects2/annawoodard/qc_redo/submitit_runs",
                "--partition general",
                f"--constraint {constraint}",
                f"--gpuspec {gpuspec}",
                f"--ngpus {ngpus}",
                "--cpus-per-task 16",
                f"--mem-gb {mem_gb}",
                f"--timeout-min {timeout_min}",
                "--no-wait",
                "--views /net/projects2/annawoodard/qc_export/views_for_qc.parquet",
                "--export-dir /net/projects2/annawoodard/qc_export",
                f"--run-file {run_file}",
                "--model-path /net/projects2/annawoodard/models/Qwen3.5-397B-A17B-FP8-prima-repair",
                "--exam 1.2.124.113532.10.7.222.5.20101217.121444.14285505",
                "--few-shot-examples 0",
                "--prompt-mode marker_classifier",
                "--probe-tag 'vertical line (detector artifact)'",
                "--disable-thinking",
                f"--debug-dump-dir {debug_dir}",
                "--tags-file /net/projects2/annawoodard/qc_redo/annotation_tags.json",
            ]
        ),
    ]


def build_reviewed33_submit_cmd(*, resource: str, timeout_min: int) -> list[str]:
    timestamp = datetime.now().strftime("%Y%m%d")
    name = f"auto_qc_qwen397b_fp8_vertical_line_reviewed33_bf16experts_{resource}_t{timeout_min}"
    run_file = (
        DEFAULT_RUNS
        / f"qwen397b_fp8_vertical_line_reviewed33_bf16experts_{resource}_t{timeout_min}_{timestamp}.json"
    )
    debug_dir = (
        DEFAULT_DEBUG
        / f"qwen397b_fp8_vertical_line_reviewed33_bf16experts_{resource}_t{timeout_min}_{timestamp}"
    )
    if resource == "a100":
        gpuspec = "a100"
        ngpus = "8"
        constraint = "a100"
        mem_gb = "900"
    else:
        gpuspec = "h200"
        ngpus = "4"
        constraint = "h200"
        mem_gb = "720"

    return [
        "bash",
        "-lc",
        " ".join(
            [
                "export PRIMA_QWEN35_FP8_FORCE_BF16_EXPERTS=1 CUDA_LAUNCH_BLOCKING=1 PYTORCH_SHOW_CPP_STACKTRACES=1;",
                f"micromamba run -p {ENV_PREFIX} python {SUBMIT_SCRIPT}",
                f"--name {name}",
                "--log-dir /net/projects2/annawoodard/qc_redo/submitit_runs",
                "--partition general",
                f"--constraint {constraint}",
                f"--gpuspec {gpuspec}",
                f"--ngpus {ngpus}",
                "--cpus-per-task 16",
                f"--mem-gb {mem_gb}",
                f"--timeout-min {timeout_min}",
                "--no-wait",
                "--views /net/projects2/annawoodard/qc_export/views_for_qc.parquet",
                "--export-dir /net/projects2/annawoodard/qc_export",
                f"--run-file {run_file}",
                "--model-path /net/projects2/annawoodard/models/Qwen3.5-397B-A17B-FP8-prima-repair",
                "--exam-list /net/projects2/annawoodard/qc_redo/debug_exam_lists/vertical_line_reviewed33.txt",
                "--few-shot-examples 0",
                "--prompt-mode marker_classifier",
                "--probe-tag 'vertical line (detector artifact)'",
                "--disable-thinking",
                f"--debug-dump-dir {debug_dir}",
                "--tags-file /net/projects2/annawoodard/qc_redo/annotation_tags.json",
            ]
        ),
    ]


def run_submit(cmd: list[str], *, log_file: Path, dry_run: bool) -> None:
    append_log(log_file, "submit_cmd=" + " ".join(cmd))
    if dry_run:
        append_log(log_file, "dry_run=true; submission skipped")
        return
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    append_log(log_file, f"submit_returncode={proc.returncode}")
    if proc.stdout.strip():
        append_log(log_file, "submit_stdout=" + proc.stdout.strip().replace("\n", " | "))
    if proc.stderr.strip():
        append_log(log_file, "submit_stderr=" + proc.stderr.strip().replace("\n", " | "))


def decide_and_act(args: argparse.Namespace) -> dict[str, Any]:
    records = list_campaign_runs(args.submitit_runs_dir)
    summary: dict[str, Any] = {
        "checked_at": now_local(),
        "num_records": len(records),
        "active_oneexam": [record.job_id for record in active(records, kind="oneexam")],
        "active_reviewed33": [record.job_id for record in active(records, kind="reviewed33")],
        "action": "none",
    }
    latest_oneexam = latest(records, kind="oneexam")
    log_file = args.log_file

    if active(records, kind="reviewed33"):
        append_log(log_file, "reviewed33 already active; no action")
        summary["action"] = "wait_reviewed33_active"
        return summary

    if completed(records, kind="reviewed33"):
        append_log(log_file, "reviewed33 already completed; no action")
        summary["action"] = "done_reviewed33_completed"
        return summary

    active_oneexam = active(records, kind="oneexam")
    if active_oneexam:
        latest_active = active_oneexam[-1]
        append_log(
            log_file,
            f"oneexam active job_id={latest_active.job_id} state={latest_active.state} resource={latest_active.resource}",
        )
        summary["action"] = "wait_oneexam_active"
        if should_submit_a100_fallback(
            latest_active, threshold_hours=args.pending_fallback_hours
        ) and not any(
            record.resource == "a100" for record in active_oneexam
        ):
            append_log(
                log_file,
                f"h200 oneexam pending for >= {args.pending_fallback_hours}h; submitting a100 fallback",
            )
            run_submit(
                build_oneexam_submit_cmd(resource="a100", timeout_min=240),
                log_file=log_file,
                dry_run=args.dry_run,
            )
            summary["action"] = "submit_oneexam_a100_fallback"
        return summary

    if latest_oneexam is None:
        append_log(log_file, "no oneexam runs found; submitting initial h200 oneexam")
        run_submit(
            build_oneexam_submit_cmd(resource="h200", timeout_min=240),
            log_file=log_file,
            dry_run=args.dry_run,
        )
        summary["action"] = "submit_initial_oneexam_h200"
        return summary

    if latest_oneexam.state == "COMPLETED":
        chosen_resource = latest_oneexam.resource
        append_log(
            log_file,
            f"latest oneexam completed job_id={latest_oneexam.job_id}; submitting reviewed33 on {chosen_resource}",
        )
        run_submit(
            build_reviewed33_submit_cmd(resource=chosen_resource, timeout_min=240),
            log_file=log_file,
            dry_run=args.dry_run,
        )
        summary["action"] = f"submit_reviewed33_{chosen_resource}"
        return summary

    if latest_oneexam.state.startswith(FAIL_STATES_PREFIXES):
        log_path = latest_oneexam.log_dir / f"{latest_oneexam.job_id}_0_log.err"
        failure_kind = classify_failure(log_path)
        append_log(
            log_file,
            f"latest oneexam failed job_id={latest_oneexam.job_id} resource={latest_oneexam.resource} failure_kind={failure_kind}",
        )
        summary["latest_oneexam_failure_kind"] = failure_kind
        if failure_kind == "timeout":
            next_timeout = 240
            if "t240" in latest_oneexam.name or "retry240" in latest_oneexam.name:
                next_timeout = 360
            if "t360" in latest_oneexam.name or "retry360" in latest_oneexam.name:
                next_timeout = 480
            run_submit(
                build_oneexam_submit_cmd(
                    resource=latest_oneexam.resource, timeout_min=next_timeout
                ),
                log_file=log_file,
                dry_run=args.dry_run,
            )
            summary["action"] = f"resubmit_oneexam_timeout_{latest_oneexam.resource}_t{next_timeout}"
            return summary
        if failure_kind == "fp8_runtime":
            fallback_resource = "a100" if latest_oneexam.resource == "h200" else "h200"
            run_submit(
                build_oneexam_submit_cmd(resource=fallback_resource, timeout_min=240),
                log_file=log_file,
                dry_run=args.dry_run,
            )
            summary["action"] = f"resubmit_oneexam_fp8_runtime_{fallback_resource}"
            return summary
        run_submit(
            build_oneexam_submit_cmd(resource="h200", timeout_min=240),
            log_file=log_file,
            dry_run=args.dry_run,
        )
        summary["action"] = "resubmit_oneexam_unknown_h200"
        return summary

    append_log(log_file, f"no rule matched latest_oneexam_state={latest_oneexam.state}; no action")
    summary["action"] = f"no_rule_{latest_oneexam.state}"
    return summary


def main() -> int:
    args = parse_args()
    state = load_state(args.state_file)
    summary = decide_and_act(args)
    state["last_check"] = summary
    state.setdefault("history", []).append(summary)
    state["history"] = state["history"][-100:]
    save_state(args.state_file, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
