#!/usr/bin/env python
"""Download one pinned auto-QC model snapshot for offline vLLM serving."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prima.vllm_server import load_model_registry  # noqa: E402

DEFAULT_REGISTRY = ROOT / "qc" / "auto_qc_models.json"
DEFAULT_MODELS_DIR = Path("/gpfs/data/huo-lab/Image/annawoodard/models")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        help="Model key from qc/auto_qc_models.json (one model per invocation)",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help=f"Model registry (default: {DEFAULT_REGISTRY})",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"Destination root (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pinned download plan without downloading",
    )
    parser.add_argument(
        "--use-xet",
        action="store_true",
        help="Use Hugging Face Xet transport (disabled by default on this cluster)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    registry = load_model_registry(args.registry.resolve())
    try:
        spec = registry[args.model]
    except KeyError as exc:
        choices = ", ".join(sorted(registry))
        raise ValueError(
            f"unknown model key {args.model!r}; choose one of: {choices}"
        ) from exc
    if spec.revision is None:
        raise ValueError(
            f"refusing an unpinned model download for {spec.key!r}; add revision to the registry"
        )
    destination = (args.models_dir.resolve() / spec.directory_name).resolve()
    print(f"{spec.key}: {spec.repo_id}@{spec.revision} -> {destination}")
    if args.dry_run:
        return 0

    if not args.use_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import snapshot_download

    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=spec.repo_id,
        revision=spec.revision,
        local_dir=str(destination),
    )
    if not (destination / "config.json").is_file():
        raise RuntimeError(f"download completed without config.json: {destination}")
    provenance = {
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "xet_enabled": bool(args.use_xet),
        "command": " ".join(sys.argv),
    }
    provenance_path = destination / "prima_snapshot.json"
    temporary_path = provenance_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(provenance, indent=2) + "\n")
    temporary_path.replace(provenance_path)
    print(f"{spec.key}: verified local snapshot and wrote {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
