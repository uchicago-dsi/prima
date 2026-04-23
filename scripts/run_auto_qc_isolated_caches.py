#!/usr/bin/env python3
"""Run auto_annotate_qc.py with isolated runtime caches."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTO_QC_PATH = PROJECT_ROOT / "auto_annotate_qc.py"

DEFAULT_SHARED_HF_HOME = Path.home() / ".cache" / "huggingface"
DEFAULT_SHARED_HF_HUB = DEFAULT_SHARED_HF_HOME / "hub"
DEFAULT_SHARED_KERNEL_PATTERNS = ("models--kernels-community--*",)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run auto_annotate_qc.py with isolated Triton/Torch caches and optional "
            "isolated Hugging Face kernel source caches."
        )
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="Per-run cache root for Triton/Torch/HF isolation.",
    )
    parser.add_argument(
        "--hf-cache-mode",
        choices=("shared", "copied", "empty"),
        default="shared",
        help=(
            "shared: keep using the shared HF hub cache while isolating runtime caches; "
            "copied: copy shared kernels-community entries into an isolated HF cache; "
            "empty: use an empty isolated HF cache."
        ),
    )
    parser.add_argument(
        "--shared-hf-home",
        type=Path,
        default=DEFAULT_SHARED_HF_HOME,
        help="Shared HF_HOME to preserve when --hf-cache-mode=shared.",
    )
    parser.add_argument(
        "--shared-hf-hub",
        type=Path,
        default=DEFAULT_SHARED_HF_HUB,
        help="Shared HF hub cache to preserve or copy from.",
    )
    parser.add_argument(
        "--copy-pattern",
        action="append",
        default=[],
        help=(
            "Glob under the shared HF hub cache to preseed into the isolated cache "
            "when --hf-cache-mode=copied. Can be repeated."
        ),
    )
    parser.add_argument(
        "--clear-cache-root",
        action="store_true",
        help="Delete --cache-root before launching.",
    )
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="Print the cache-related environment overrides before launching.",
    )
    parser.add_argument(
        "--extra-env",
        action="append",
        default=[],
        help="Extra KEY=VALUE environment variables for the launched process.",
    )
    parser.add_argument(
        "auto_qc_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to auto_annotate_qc.py. Prefix with '--'.",
    )
    return parser


def strip_remainder_delimiter(args: Sequence[str]) -> list[str]:
    cleaned = list(args)
    if cleaned and cleaned[0] == "--":
        cleaned = cleaned[1:]
    return cleaned


def copytree_overwrite(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def ensure_copied_hf_kernel_cache(
    shared_hf_hub: Path,
    isolated_hf_hub: Path,
    patterns: Sequence[str],
) -> list[str]:
    copied_entries: list[str] = []
    isolated_hf_hub.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for source in sorted(shared_hf_hub.glob(pattern)):
            if not source.exists():
                continue
            destination = isolated_hf_hub / source.name
            copytree_overwrite(source, destination)
            copied_entries.append(source.name)
    return copied_entries


def parse_extra_env(pairs: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--extra-env must be KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        if not key:
            raise ValueError(f"--extra-env must include a non-empty key, got {pair!r}")
        parsed[key] = value
    return parsed


def build_isolated_env(args: argparse.Namespace) -> tuple[dict[str, str], list[str]]:
    env = os.environ.copy()
    cache_root = args.cache_root.resolve()
    xdg_cache_home = cache_root / "xdg"
    torch_cache_root = cache_root / "torch"
    hf_home = cache_root / "hf"
    hf_hub = hf_home / "hub"
    triton_cache_dir = cache_root / "triton"
    torchinductor_cache_dir = cache_root / "torchinductor"
    pytorch_kernel_cache_path = torch_cache_root / "kernels"
    torch_extensions_dir = cache_root / "torch_extensions"

    for path in (
        xdg_cache_home,
        triton_cache_dir,
        torchinductor_cache_dir,
        pytorch_kernel_cache_path,
        torch_extensions_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    copied_entries: list[str] = []
    if args.hf_cache_mode == "shared":
        env["HF_HOME"] = str(args.shared_hf_home.resolve())
        env["HF_HUB_CACHE"] = str(args.shared_hf_hub.resolve())
        env["HUGGINGFACE_HUB_CACHE"] = str(args.shared_hf_hub.resolve())
    else:
        hf_home.mkdir(parents=True, exist_ok=True)
        hf_hub.mkdir(parents=True, exist_ok=True)
        env["HF_HOME"] = str(hf_home)
        env["HF_HUB_CACHE"] = str(hf_hub)
        env["HUGGINGFACE_HUB_CACHE"] = str(hf_hub)
        if args.hf_cache_mode == "copied":
            copy_patterns = args.copy_pattern or list(DEFAULT_SHARED_KERNEL_PATTERNS)
            copied_entries = ensure_copied_hf_kernel_cache(
                shared_hf_hub=args.shared_hf_hub.resolve(),
                isolated_hf_hub=hf_hub,
                patterns=copy_patterns,
            )

    env["XDG_CACHE_HOME"] = str(xdg_cache_home)
    env["TRITON_CACHE_DIR"] = str(triton_cache_dir)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(torchinductor_cache_dir)
    env["PYTORCH_KERNEL_CACHE_PATH"] = str(pytorch_kernel_cache_path)
    env["TORCH_EXTENSIONS_DIR"] = str(torch_extensions_dir)
    env.update(parse_extra_env(args.extra_env))
    return env, copied_entries


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    auto_qc_args = strip_remainder_delimiter(args.auto_qc_args)
    if not auto_qc_args:
        raise ValueError("must provide auto_annotate_qc.py arguments after '--'")
    if not AUTO_QC_PATH.exists():
        raise FileNotFoundError(f"missing auto_annotate_qc.py at {AUTO_QC_PATH}")

    cache_root = args.cache_root.resolve()
    if args.clear_cache_root and cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    env, copied_entries = build_isolated_env(args)
    command = [sys.executable, str(AUTO_QC_PATH), *auto_qc_args]

    print(f"cache_root={cache_root}")
    print(f"hf_cache_mode={args.hf_cache_mode}")
    if copied_entries:
        print("copied_hf_entries=" + ",".join(copied_entries))
    if args.print_env:
        for key in (
            "HF_HOME",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "XDG_CACHE_HOME",
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "PYTORCH_KERNEL_CACHE_PATH",
            "TORCH_EXTENSIONS_DIR",
        ):
            print(f"{key}={env[key]}")
        for pair in args.extra_env:
            key, _ = pair.split("=", 1)
            print(f"{key}={env[key]}")
    print("command=" + " ".join(command))
    proc = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
