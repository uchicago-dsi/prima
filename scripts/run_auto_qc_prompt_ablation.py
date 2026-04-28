#!/usr/bin/env python3
"""Write or submit the fixed three-arm auto-QC prompt ablation."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Arm:
    name: str
    prompt_variant: str
    few_shot_examples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the baseline, few-shot, and recall-tilted auto-QC jobs for "
            "a fixed enriched exam list."
        )
    )
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--exam-list", type=Path, required=True)
    parser.add_argument("--qc-file", type=Path, required=True)
    parser.add_argument("--tags-file", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--debug-dir", type=Path, required=True)
    parser.add_argument("--submitit-log-dir", type=Path, required=True)
    parser.add_argument(
        "--tag",
        default="vertical line (detector artifact)",
        help="Target probe tag.",
    )
    parser.add_argument("--few-shot-examples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--name-prefix", default="qwen397b_vertical_line_ablation")
    parser.add_argument("--partition", default="general")
    parser.add_argument("--qos", default="")
    parser.add_argument("--constraint", default="h200")
    parser.add_argument("--gpuspec", default="h200")
    parser.add_argument("--ngpus", type=int, default=4)
    parser.add_argument("--cpus-per-task", type=int, default=16)
    parser.add_argument("--mem-gb", type=int, default=720)
    parser.add_argument("--timeout-min", type=int, default=240)
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit jobs. Default only writes commands.sh.",
    )
    return parser.parse_args()


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def build_command(args: argparse.Namespace, arm: Arm, stamp: str) -> list[str]:
    run_file = args.run_dir / f"{args.name_prefix}_{arm.name}_{stamp}.json"
    debug_dir = args.debug_dir / f"{args.name_prefix}_{arm.name}_{stamp}"
    command = [
        sys.executable,
        "submit_auto_qc.py",
        "--name",
        f"{args.name_prefix}_{arm.name}",
        "--log-dir",
        str(args.submitit_log_dir),
        "--partition",
        args.partition,
        "--constraint",
        args.constraint,
        "--gpuspec",
        args.gpuspec,
        "--ngpus",
        str(args.ngpus),
        "--cpus-per-task",
        str(args.cpus_per_task),
        "--mem-gb",
        str(args.mem_gb),
        "--timeout-min",
        str(args.timeout_min),
        "--no-wait",
    ]
    if args.qos:
        command.extend(["--qos", args.qos])
    command.extend(
        [
            "--views",
            str(args.views.resolve()),
            "--export-dir",
            str(args.export_dir.resolve()),
            "--exam-list",
            str(args.exam_list.resolve()),
            "--qc-file",
            str(args.qc_file.resolve()),
            "--few-shot-qc-file",
            str(args.qc_file.resolve()),
            "--tags-file",
            str(args.tags_file.resolve()),
            "--model-path",
            str(args.model_path.resolve()),
            "--run-file",
            str(run_file.resolve()),
            "--debug-dump-dir",
            str(debug_dir.resolve()),
            "--prompt-mode",
            "marker_classifier",
            "--probe-tag",
            args.tag,
            "--prompt-variant",
            arm.prompt_variant,
            "--few-shot-examples",
            str(arm.few_shot_examples),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--disable-thinking",
        ]
    )
    return command


def write_decision_rules(path: Path, args: argparse.Namespace) -> None:
    path.write_text(
        "\n".join(
            [
                "# Auto-QC Prompt Ablation Decision Rules",
                "",
                f"- Target tag: `{args.tag}`",
                "- Fixed exam universe: the supplied enriched exam list.",
                "- Primary readout: target-tag recall and FP burden.",
                "- Stop rule: choose the first arm with recall >= 0.70 and tolerable FP burden for human QC review.",
                "- Follow-up rule: if all arms remain near-zero recall, inspect FN debug dumps visually before changing model/runtime settings.",
                "- Pivot rule: if recall improves only with very high FP burden, build a second hard-negative set before broad rollout.",
                "",
            ]
        )
    )


def main() -> int:
    args = parse_args()
    if args.few_shot_examples <= 0:
        raise ValueError("--few-shot-examples must be positive for ablation arms")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arms = [
        Arm("baseline", "baseline", 0),
        Arm("fewshot", "baseline", args.few_shot_examples),
        Arm("recall_tilted", "recall_tilted", args.few_shot_examples),
    ]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.debug_dir.mkdir(parents=True, exist_ok=True)
    args.submitit_log_dir.mkdir(parents=True, exist_ok=True)

    commands = [build_command(args, arm, stamp) for arm in arms]
    commands_path = args.run_dir / f"{args.name_prefix}_{stamp}_commands.sh"
    commands_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "\n".join(shell_join(command) for command in commands)
        + "\n"
    )
    commands_path.chmod(0o755)
    write_decision_rules(
        args.run_dir / f"{args.name_prefix}_{stamp}_decision_rules.md",
        args,
    )

    if args.submit:
        for command in commands:
            subprocess.run(command, check=True)
    else:
        print(f"wrote commands: {commands_path}")
        print("dry run only; rerun with --submit to submit all three arms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
