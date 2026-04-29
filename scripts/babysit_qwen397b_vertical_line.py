#!/usr/bin/env python3
"""Timed unattended babysitter for the PRIMA Qwen3.5 397B vertical-line campaign."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = (
    Path("/net/projects2/annawoodard/qc_redo/interactive_debug") / "qwen397b_babysitter"
)
DEFAULT_LOG_FILE = DEFAULT_OUTPUT_ROOT / "babysitter.log"
DEFAULT_STATE_FILE = DEFAULT_OUTPUT_ROOT / "state" / "state.json"
DEFAULT_LOCKFILE = (
    Path(os.environ.get("XDG_RUNTIME_DIR", str(DEFAULT_OUTPUT_ROOT / "state")))
    / "prima-qwen397b-babysitter.lock"
)
DEFAULT_SCHEMA = REPO_ROOT / "scripts" / "qwen397b_babysitter_output_schema.json"
DEFAULT_PROMPT = REPO_ROOT / "scripts" / "qwen397b_babysitter_prompt.txt"
DEFAULT_NOTEBOOK = (
    Path("/net/projects2/annawoodard/qc_redo/interactive_debug")
    / "fp8_debug_lab_notebook.md"
)
DEFAULT_SUBMITIT_RUNS = Path("/net/projects2/annawoodard/qc_redo/submitit_runs")
DEFAULT_RUN_NAME_REGEX = (
    r"(auto_qc_qwen397b_fp8_vertical_line.*bf16experts.*"
    r"|qwen397b_vertical_line_ablation.*)"
)
DEFAULT_FAMILY_LABEL = "qwen397b_vertical_line_bf16experts"
DEFAULT_CODEX_MODEL = "gpt-5.5"


def _detect_codex_bin() -> Path:
    candidates: list[Path] = []
    env_bin = os.environ.get("CODEX_BIN")
    if env_bin:
        candidates.append(Path(env_bin))
    candidates.append(Path("/net/projects/annawoodard/micromamba/envs/codex/bin/codex"))
    which = shutil.which("codex")
    if which:
        candidates.append(Path(which))
    for candidate in (
        Path("/home/annawoodard/.local/bin/codex"),
        Path("/usr/local/bin/codex"),
        Path("/opt/homebrew/bin/codex"),
    ):
        candidates.append(candidate)
    for candidate in candidates:
        try:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
        except OSError:
            continue
    return Path("codex")


DEFAULT_CODEX_BIN = _detect_codex_bin()
ACTIVE_STATES = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "SUSPENDED",
    "REQUEUED",
}


@dataclass
class RunRecord:
    run_name: str
    job_id: str
    job_state: str
    exit_code: str | None
    start: str | None
    end: str | None
    node_list: str | None
    submitit_run_root: str
    log_dir: str
    stdout_path: str | None
    stderr_path: str | None
    stdout_tail: list[str]
    stderr_tail: list[str]
    submit_time_local: str


def now_local() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_log(log_file: Path, message: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{now_local()}] {message}\n")


def _run(
    cmd: list[str],
    *,
    input_text: str | None = None,
    extra_path: Path | None = None,
) -> str:
    env = os.environ.copy()
    if extra_path is not None:
        env["PATH"] = f"{extra_path}:{env.get('PATH', '')}"
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        input=input_text,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed.stdout


def _tail_lines(path: Path | None, max_lines: int) -> list[str]:
    if path is None or not path.exists():
        return []
    try:
        out = _run(["tail", "-n", str(max_lines), str(path)])
    except Exception:
        return []
    return [line.rstrip("\n") for line in out.splitlines()]


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Timer-driven babysitter that selects the PRIMA Qwen3.5 397B vertical-line "
            "campaign by regex and invokes Codex non-interactively to make each decision."
        )
    )
    parser.add_argument("--family-label", default=DEFAULT_FAMILY_LABEL)
    parser.add_argument("--run-name-regex", default=DEFAULT_RUN_NAME_REGEX)
    parser.add_argument("--submitit-runs-dir", type=Path, default=DEFAULT_SUBMITIT_RUNS)
    parser.add_argument("--notebook-path", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--lockfile", type=Path, default=DEFAULT_LOCKFILE)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--prompt-template", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--codex-bin", type=Path, default=DEFAULT_CODEX_BIN)
    parser.add_argument("--model", default=DEFAULT_CODEX_MODEL)
    parser.add_argument("--tail-lines", type=int, default=80)
    parser.add_argument("--max-records", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def query_sacct_state(
    job_id: str,
) -> tuple[str, str | None, str | None, str | None, str | None]:
    proc = subprocess.run(
        [
            "sacct",
            "-j",
            job_id,
            "--format",
            "State,Start,End,ExitCode,NodeList",
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
        state, start, end, exit_code, node_list = line.split("|", 4)
        return (
            state.strip(),
            start.strip() or None,
            end.strip() or None,
            exit_code.strip() or None,
            node_list.strip() or None,
        )
    return "UNKNOWN", None, None, None, None


def query_squeue_state(job_id: str) -> str | None:
    proc = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        capture_output=True,
        text=True,
        check=False,
    )
    state = proc.stdout.strip()
    return state or None


def effective_state(
    job_id: str,
) -> tuple[str, str | None, str | None, str | None, str | None]:
    state, start, end, exit_code, node_list = query_sacct_state(job_id)
    queue_state = query_squeue_state(job_id)
    if queue_state:
        state = queue_state
    return state, start, end, exit_code, node_list


def parse_submit_time(run_root: Path) -> datetime:
    match = re.search(r"_(\d{8}_\d{6})$", run_root.name)
    if match is None:
        return datetime.fromtimestamp(run_root.stat().st_mtime)
    return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")


def log_paths_from_sacct(job_id: str) -> tuple[Path | None, Path | None]:
    proc = subprocess.run(
        [
            "sacct",
            "-j",
            job_id,
            "--format",
            "JobIDRaw,StdOut,StdErr",
            "-n",
            "-P",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_path = None
    stderr_path = None
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 3 or parts[0] != str(job_id):
            continue
        stdout_raw = parts[1].replace("%j", str(job_id)).replace("%A", str(job_id))
        stderr_raw = parts[2].replace("%j", str(job_id)).replace("%A", str(job_id))
        stdout_path = Path(stdout_raw) if stdout_raw else None
        stderr_path = Path(stderr_raw) if stderr_raw else None
        break
    return stdout_path, stderr_path


def list_campaign_runs(
    *,
    submitit_runs_dir: Path,
    run_name_regex: str,
    tail_lines: int,
    max_records: int,
) -> list[RunRecord]:
    pattern = re.compile(run_name_regex)
    records: list[RunRecord] = []
    for run_root in sorted(submitit_runs_dir.iterdir()):
        if not run_root.is_dir():
            continue
        if not pattern.search(run_root.name):
            continue
        log_dir = run_root / "submitit_logs"
        if not log_dir.exists():
            continue
        submitted = sorted(log_dir.glob("*_submitted.pkl"))
        if not submitted:
            continue
        job_id = submitted[0].name.split("_", 1)[0]
        state, start, end, exit_code, node_list = effective_state(job_id)
        stdout_path, stderr_path = log_paths_from_sacct(job_id)
        submit_time = parse_submit_time(run_root)
        records.append(
            RunRecord(
                run_name=run_root.name,
                job_id=job_id,
                job_state=state,
                exit_code=exit_code,
                start=start,
                end=end,
                node_list=node_list,
                submitit_run_root=str(run_root),
                log_dir=str(log_dir),
                stdout_path=str(stdout_path) if stdout_path else None,
                stderr_path=str(stderr_path) if stderr_path else None,
                stdout_tail=_tail_lines(stdout_path, tail_lines),
                stderr_tail=_tail_lines(stderr_path, tail_lines),
                submit_time_local=submit_time.astimezone().isoformat(
                    timespec="seconds"
                ),
            )
        )
    records.sort(key=lambda record: record.submit_time_local)
    return records[-max_records:]


def _invoke_codex(
    *,
    codex_bin: Path,
    schema_path: Path,
    prompt_text: str,
    output_file: Path,
    model: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    cmd = [
        str(codex_bin),
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        str(REPO_ROOT),
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_file),
    ]
    if model:
        cmd.extend(["-m", model])
    if dry_run:
        return {
            "campaign_status": "dry_run",
            "action_type": "dry_run",
            "did_act": False,
            "needs_human": False,
            "summary": "Dry run only; Codex not invoked.",
            "why": "Dry-run mode was explicitly requested for validation.",
            "failure_cause": "",
            "evidence": ["dry_run=true"],
            "files_changed": [],
            "new_job_ids": [],
            "notebook_updated": False,
            "goal_progress": "Validated the babysitter path without changing campaign state.",
            "next_check_hint": "none",
        }
    _run(cmd, input_text=prompt_text, extra_path=codex_bin.parent)
    raw = output_file.read_text(encoding="utf-8").strip()
    return json.loads(raw)


def build_pending_only_decision(active_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a deterministic decision for scheduler-only pending states."""
    job_summaries = [
        f"{record['job_id']} {record['run_name']} state={record['job_state']}"
        for record in active_records
    ]
    return {
        "campaign_status": "waiting",
        "action_type": "deterministic_pending_wait",
        "did_act": False,
        "needs_human": False,
        "summary": (
            "All matching active Qwen vertical-line jobs are pending in Slurm; "
            "skipped Codex agent invocation and will check again on the next timer tick."
        ),
        "why": (
            "Pending-only scheduler wait does not require model reasoning or code changes. "
            "Avoiding a Codex wake-up preserves quota while jobs wait for resources or priority."
        ),
        "failure_cause": "Scheduler wait only; no runtime logs or failures are available yet.",
        "evidence": job_summaries,
        "files_changed": [],
        "new_job_ids": [],
        "notebook_updated": False,
        "goal_progress": (
            "Preserved the queued jobs and avoided duplicate submissions while conserving agent quota."
        ),
        "next_check_hint": (
            "If any matching job changes from PENDING to RUNNING, inspect submitit logs. "
            "If all remain PENDING, continue deterministic waiting."
        ),
    }


def should_skip_codex_for_pending_only(active_records: list[dict[str, Any]]) -> bool:
    """Skip the expensive agent when every active matching job is only pending."""
    return bool(active_records) and all(
        str(record.get("job_state")) == "PENDING" for record in active_records
    )


def build_prompt(template: str, context: dict[str, Any]) -> str:
    return template.replace(
        "{{CAMPAIGN_CONTEXT_JSON}}", json.dumps(context, indent=2, sort_keys=True)
    )


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    state = _load_json(args.state_file, {"history": []})

    args.lockfile.parent.mkdir(parents=True, exist_ok=True)
    with args.lockfile.open("w") as lock_fp:
        try:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            append_log(args.log_file, "skipping run because babysitter lock is held")
            return 0

        records = list_campaign_runs(
            submitit_runs_dir=args.submitit_runs_dir,
            run_name_regex=args.run_name_regex,
            tail_lines=args.tail_lines,
            max_records=args.max_records,
        )
        active_records = [
            asdict(record) for record in records if record.job_state in ACTIVE_STATES
        ]
        terminal_records = [
            asdict(record)
            for record in records
            if record.job_state not in ACTIVE_STATES
        ]
        campaign_context = {
            "family_label": args.family_label,
            "goal": (
                "Keep the PRIMA Qwen3.5-397B vertical-line experiment moving with no dead time. "
                "User steers high-level direction; you own operational decisions, fixes, submissions, "
                "and documentation. Always make a decision."
            ),
            "repo_root": str(REPO_ROOT),
            "job_selection": {
                "run_name_regex": args.run_name_regex,
                "submitit_runs_dir": str(args.submitit_runs_dir),
            },
            "notebook_path": str(args.notebook_path),
            "log_file": str(args.log_file),
            "state_file": str(args.state_file),
            "current_time": now_local(),
            "active_runs": active_records,
            "recent_terminal_runs": terminal_records,
            "recent_decisions": state.get("history", [])[-8:],
        }

        decision_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        decision_file = args.output_root / "decisions" / f"{decision_time}.json"
        decision_file.parent.mkdir(parents=True, exist_ok=True)
        if should_skip_codex_for_pending_only(active_records) and not args.dry_run:
            result = build_pending_only_decision(active_records)
            _write_json(decision_file, result)
        else:
            prompt_template = args.prompt_template.read_text(encoding="utf-8")
            prompt = build_prompt(prompt_template, campaign_context)
            result = _invoke_codex(
                codex_bin=args.codex_bin,
                schema_path=args.schema,
                prompt_text=prompt,
                output_file=decision_file,
                model=args.model,
                dry_run=args.dry_run,
            )
        history_entry = {
            "decided_at": now_local(),
            "context_summary": {
                "num_records": len(records),
                "active_job_ids": [record["job_id"] for record in active_records],
                "terminal_job_ids": [record["job_id"] for record in terminal_records],
            },
            "decision": result,
        }
        state.setdefault("history", []).append(history_entry)
        state["history"] = state["history"][-100:]
        state["last_check"] = history_entry
        _write_json(args.state_file, state)
        append_log(
            args.log_file,
            "decision="
            + json.dumps(
                {
                    "campaign_status": result.get("campaign_status"),
                    "action_type": result.get("action_type"),
                    "summary": result.get("summary"),
                    "did_act": result.get("did_act"),
                    "new_job_ids": result.get("new_job_ids"),
                    "needs_human": result.get("needs_human"),
                },
                sort_keys=True,
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
