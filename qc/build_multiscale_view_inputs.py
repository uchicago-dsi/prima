#!/usr/bin/env python3
"""Build full-view plus overlapping-detail model inputs for view auto-QC."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

import pandas as pd

from prima.view_few_shot import sha256_file
from prima.view_multiscale import (
    MULTISCALE_CANVAS_SIZE,
    render_multiscale_view_png,
)
from prima.view_qc import normalize_view_id, validate_view_manifest_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--max-source-pixels", type=int, default=2_097_152)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")
    os.chmod(path, 0o600)


def _resolve_source(root: Path, raw_path: object) -> Path:
    relative = Path(str(raw_path))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("multiscale source image_path must be safe and relative")
    source = (root / relative).resolve()
    try:
        source.relative_to(root)
    except ValueError as error:
        raise ValueError("multiscale source image escapes its manifest root") from error
    if not source.is_file():
        raise FileNotFoundError("multiscale source image is missing")
    return source


def run_from_args(args: argparse.Namespace) -> dict[str, object]:
    manifest_path = args.manifest.resolve()
    output_manifest_path = args.output_manifest.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"view manifest not found: {manifest_path}")
    if output_manifest_path.parent != manifest_path.parent:
        raise ValueError("multiscale output manifest must remain beside its source")
    if args.max_source_pixels <= 0:
        raise ValueError("--max-source-pixels must be positive")

    asset_dir = output_manifest_path.with_suffix("")
    image_dir = asset_dir / "images"
    provenance_path = output_manifest_path.with_suffix(".provenance.json")
    readme_path = output_manifest_path.with_suffix(".README.md")
    for path in (
        output_manifest_path,
        asset_dir,
        provenance_path,
        readme_path,
    ):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite multiscale output: {path}")

    manifest = pd.read_parquet(manifest_path)
    validate_view_manifest_columns(manifest.columns, str(manifest_path))
    if manifest.empty:
        raise ValueError("multiscale source manifest is empty")
    manifest = manifest.sort_values("review_order", kind="stable").copy()
    manifest["view_id"] = manifest["view_id"].map(normalize_view_id)
    if manifest["view_id"].duplicated().any():
        raise ValueError("multiscale source manifest contains duplicate view IDs")

    asset_dir.mkdir(mode=0o700)
    image_dir.mkdir(mode=0o700)
    root = manifest_path.parent.resolve()
    model_paths: dict[str, str] = {}
    ordered_image_digests: list[str] = []
    for row in manifest.to_dict("records"):
        source_path = _resolve_source(root, row["image_path"])
        view_id = row["view_id"]
        output_path = image_dir / f"{view_id}.png"
        render_multiscale_view_png(
            source_path,
            output_path,
            max_source_pixels=args.max_source_pixels,
        )
        model_paths[view_id] = output_path.relative_to(root).as_posix()
        ordered_image_digests.append(sha256_file(output_path))

    output = manifest.copy()
    output["model_image_path"] = output["view_id"].map(model_paths)
    output.to_parquet(output_manifest_path, index=False)
    os.chmod(output_manifest_path, 0o600)

    bank_digest = hashlib.sha256("".join(ordered_image_digests).encode()).hexdigest()
    command = shlex.join([sys.executable, *sys.argv])
    provenance = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "source_manifest_sha256": sha256_file(manifest_path),
        "output_manifest_sha256": sha256_file(output_manifest_path),
        "ordered_multiscale_image_bank_sha256": bank_digest,
        "rows": len(output),
        "canonical_image_column": "image_path",
        "model_image_column": "model_image_path",
        "canvas_size": list(MULTISCALE_CANVAS_SIZE),
        "detail_layout": (
            "full target plus four 40%-length foreground bands with 50% "
            "overlap along the foreground long axis"
        ),
    }
    _write_json(provenance_path, provenance)
    readme_path.write_text(
        "\n".join(
            [
                "# Multiscale single-view model inputs",
                "",
                f"- rows: `{len(output)}`",
                "- canonical target column: `image_path`",
                "- model input column: `model_image_path`",
                f"- canvas: `{MULTISCALE_CANVAS_SIZE[0]}x{MULTISCALE_CANVAS_SIZE[1]}`",
                f"- ordered image-bank SHA-256: `{bank_digest}`",
                "",
                "Each model input contains one complete target view and four",
                "overlapping enlarged details from that same view. The canonical",
                "image path is unchanged for evaluation and source lineage.",
                "",
                f"Exact producer command: `{command}`",
                "",
            ]
        )
    )
    os.chmod(readme_path, 0o600)
    return {"rows": len(output), "bank_digest": bank_digest}


def main() -> int:
    result = run_from_args(parse_args())
    print(
        "multiscale view inputs ready: "
        f"rows={result['rows']} bank_sha256={result['bank_digest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
