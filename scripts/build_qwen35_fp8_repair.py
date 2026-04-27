#!/usr/bin/env python3
"""Build a reproducible PRIMA repair overlay for Qwen3.5-397B-A17B-FP8."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a PRIMA repair overlay for Qwen3.5-397B-A17B-FP8. "
            "The output directory contains a repair_manifest.json plus cached "
            "repaired expert tensors."
        )
    )
    parser.add_argument(
        "--base-model-path",
        type=Path,
        required=True,
        help="Base Qwen3.5-397B-A17B-FP8 checkpoint directory.",
    )
    parser.add_argument(
        "--output-model-path",
        type=Path,
        required=True,
        help="Output wrapper directory that will contain repair_manifest.json.",
    )
    parser.add_argument(
        "--repair-cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for repaired layer tensors. Defaults to <output-model-path>/cache.",
    )
    parser.add_argument(
        "--layer-spec",
        type=str,
        default="all",
        help='Layer selection for repair, e.g. "all", "0-15", or "0,4,8". Default: all.',
    )
    parser.add_argument(
        "--dequant-down-proj-spec",
        type=str,
        default="all",
        help='Layers whose expert down_proj weights should be cached in bf16. Default: all.',
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Rebuild cached repaired layers even if the cache files already exist.",
    )
    parser.add_argument(
        "--overwrite-manifest",
        action="store_true",
        help="Replace an existing repair_manifest.json in the output directory.",
    )
    parser.add_argument(
        "--force-bf16-experts",
        action="store_true",
        help=(
            "Record that expert matmuls should bypass the FP8 kernels and run "
            "through bf16 weights during inference."
        ),
    )
    return parser


def iso_utc_now() -> str:
    """Return a stable UTC timestamp for manifests."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def manifest_cache_path(output_model_path: Path, repair_cache_dir: Path) -> str:
    """Prefer a relative cache path when the cache lives under the wrapper dir."""
    try:
        return os.path.relpath(repair_cache_dir, start=output_model_path)
    except ValueError:
        return str(repair_cache_dir)


def main() -> int:
    from auto_annotate_qc import build_qwen35_fp8_repair_cache

    parser = build_parser()
    args = parser.parse_args()

    base_model_path = args.base_model_path.expanduser().resolve()
    output_model_path = args.output_model_path.expanduser().resolve()
    repair_cache_dir = (
        args.repair_cache_dir.expanduser().resolve()
        if args.repair_cache_dir is not None
        else output_model_path / "cache"
    )
    manifest_path = output_model_path / "repair_manifest.json"

    if not base_model_path.exists():
        raise FileNotFoundError(f"base model checkpoint not found: {base_model_path}")
    output_model_path.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and not args.overwrite_manifest:
        with open(manifest_path) as f:
            existing_manifest = json.load(f)
        if not isinstance(existing_manifest, dict):
            raise ValueError(f"repair manifest must be a JSON object: {manifest_path}")
        raise FileExistsError(
            f"repair manifest already exists at {manifest_path}; rerun with --overwrite-manifest to replace it"
        )

    build_summary = build_qwen35_fp8_repair_cache(
        model_path=base_model_path,
        repair_cache_dir=repair_cache_dir,
        layer_spec=args.layer_spec,
        dequant_down_proj_spec=args.dequant_down_proj_spec,
        overwrite=args.overwrite_cache,
    )

    manifest_payload: dict[str, Any] = {
        "format": "prima_qwen35_fp8_repair_manifest_v1",
        "created_at": iso_utc_now(),
        "base_model_path": os.path.relpath(base_model_path, start=output_model_path),
        "repair_cache_dir": manifest_cache_path(output_model_path, repair_cache_dir),
        "repair_layer_spec": args.layer_spec,
        "dequant_down_proj_spec": args.dequant_down_proj_spec,
        "force_bf16_experts": bool(args.force_bf16_experts),
        "build_summary": build_summary,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_payload, f, indent=2)

    print(f"Wrote repair manifest: {manifest_path}")
    print(f"Base model:            {base_model_path}")
    print(f"Repair cache:          {repair_cache_dir}")
    print(f"Layer spec:            {args.layer_spec}")
    print(f"Down-proj dequant:     {args.dequant_down_proj_spec}")
    print(f"Force bf16 experts:    {bool(args.force_bf16_experts)}")
    print(
        "Cache summary:         "
        f"written={len(build_summary['written_layers'])} "
        f"skipped={len(build_summary['skipped_existing_layers'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
