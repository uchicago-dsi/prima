#!/usr/bin/env python
"""Download one pinned auto-QC model snapshot for offline vLLM serving."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
from fnmatch import fnmatch
import json
import os
from pathlib import Path
import subprocess
import sys
import time

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
    parser.add_argument(
        "--datamover-host",
        default="",
        help=(
            "Resolve pinned Hub URLs locally and transfer weight shards through "
            "this SSH host, which must see the same destination filesystem"
        ),
    )
    parser.add_argument(
        "--datamover-workers",
        type=int,
        default=4,
        help="Concurrent datamover weight transfers (default: 4)",
    )
    parser.add_argument(
        "--datamover-attempts",
        type=int,
        default=4,
        help="Fresh signed-URL attempts per weight file (default: 4)",
    )
    return parser


def _write_provenance(
    *, destination: Path, spec: object, transfer: dict[str, object]
) -> Path:
    provenance = {
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "weight_format": spec.weight_format,
        "download_ignore_patterns": list(spec.download_ignore_patterns),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "transfer": transfer,
        "command": " ".join(sys.argv),
    }
    provenance_path = destination / "prima_snapshot.json"
    temporary_path = provenance_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(provenance, indent=2) + "\n")
    temporary_path.replace(provenance_path)
    return provenance_path


def _safe_relative_hub_path(filename: str) -> Path:
    relative = Path(filename)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != filename
    ):
        raise ValueError(f"unsafe Hugging Face snapshot path: {filename!r}")
    return relative


def _remote_download_script(
    *, signed_url: str, target: Path, expected_size: int
) -> str:
    encoded_url = base64.b64encode(signed_url.encode()).decode()
    encoded_target = base64.b64encode(str(target).encode()).decode()
    return f"""set -euo pipefail
url=$(printf '%s' '{encoded_url}' | base64 -d)
target=$(printf '%s' '{encoded_target}' | base64 -d)
expected={expected_size}
part="${{target}}.part"
mkdir -p "$(dirname "${{target}}")"
if [[ -f "${{target}}" ]]; then
  actual=$(stat -c%s "${{target}}")
  [[ "${{actual}}" == "${{expected}}" ]] && exit 0
  echo "existing destination has wrong size" >&2
  exit 3
fi
if [[ -f "${{part}}" ]]; then
  actual=$(stat -c%s "${{part}}")
  if (( actual > expected )); then
    echo "partial destination exceeds expected size" >&2
    exit 4
  fi
fi
curl -L --fail --silent --show-error \
  --continue-at - \
  --retry 8 --retry-all-errors --retry-delay 10 \
  --speed-time 300 --speed-limit 10240 \
  -o "${{part}}" "${{url}}"
actual=$(stat -c%s "${{part}}")
if [[ "${{actual}}" != "${{expected}}" ]]; then
  echo "incomplete destination: ${{actual}} of ${{expected}} bytes" >&2
  exit 5
fi
mv "${{part}}" "${{target}}"
"""


def _download_weight_via_datamover(
    *,
    repo_id: str,
    revision: str,
    filename: str,
    destination: Path,
    host: str,
    attempts: int,
) -> tuple[str, int]:
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    relative = _safe_relative_hub_path(filename)
    target = destination / relative
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, attempts + 1):
        metadata = get_hf_file_metadata(
            hf_hub_url(repo_id, filename, revision=revision)
        )
        if metadata.commit_hash != revision:
            raise RuntimeError(
                f"resolved revision mismatch for {filename}: "
                f"expected {revision}, found {metadata.commit_hash}"
            )
        if metadata.size is None or metadata.size <= 0:
            raise RuntimeError(f"Hub metadata has no positive size for {filename}")
        script = _remote_download_script(
            signed_url=metadata.location,
            target=target,
            expected_size=metadata.size,
        )
        try:
            subprocess.run(
                [
                    "ssh",
                    "-q",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=30",
                    host,
                    "bash",
                    "-s",
                ],
                input=script,
                text=True,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as error:
            last_error = error
            if attempt == attempts:
                break
            time.sleep(min(60, 5 * attempt))
            continue
        actual_size = target.stat().st_size if target.is_file() else 0
        if actual_size != metadata.size:
            raise RuntimeError(
                f"datamover returned success but {filename} has size "
                f"{actual_size}, expected {metadata.size}"
            )
        return filename, actual_size
    assert last_error is not None
    detail = last_error.stderr.strip() or f"exit status {last_error.returncode}"
    raise RuntimeError(
        f"datamover failed for {filename} after {attempts} attempts: {detail}"
    ) from last_error


def _download_via_datamover(
    *, spec: object, destination: Path, host: str, workers: int, attempts: int
) -> dict[str, object]:
    from huggingface_hub import model_info, snapshot_download

    snapshot_download(
        repo_id=spec.repo_id,
        revision=spec.revision,
        local_dir=str(destination),
        ignore_patterns=["*.safetensors", "*.bin"],
    )
    info = model_info(spec.repo_id, revision=spec.revision, files_metadata=True)
    if info.sha != spec.revision:
        raise RuntimeError(
            f"model metadata revision mismatch: expected {spec.revision}, found {info.sha}"
        )
    weight_files = sorted(
        sibling.rfilename
        for sibling in info.siblings
        if sibling.rfilename.endswith((".safetensors", ".bin"))
        and not any(
            fnmatch(sibling.rfilename, pattern)
            for pattern in spec.download_ignore_patterns
        )
    )
    if not weight_files:
        raise RuntimeError(
            f"model snapshot has no supported weight files: {spec.repo_id}"
        )

    completed_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _download_weight_via_datamover,
                repo_id=spec.repo_id,
                revision=spec.revision,
                filename=filename,
                destination=destination,
                host=host,
                attempts=attempts,
            ): filename
            for filename in weight_files
        }
        for future in as_completed(futures):
            filename, size = future.result()
            completed_bytes += size
            print(f"downloaded {filename} ({size} bytes)", flush=True)
    return {
        "method": "datamover",
        "host": host,
        "workers": workers,
        "attempts_per_file": attempts,
        "weight_files": len(weight_files),
        "weight_bytes": completed_bytes,
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.datamover_workers <= 0:
        raise ValueError("--datamover-workers must be positive")
    if args.datamover_attempts <= 0:
        raise ValueError("--datamover-attempts must be positive")
    datamover_host = args.datamover_host.strip()
    if datamover_host and args.use_xet:
        raise ValueError("--datamover-host and --use-xet are mutually exclusive")
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
    route = f"datamover:{datamover_host}" if datamover_host else "direct"
    print(f"{spec.key}: {spec.repo_id}@{spec.revision} -> {destination} via {route}")
    print(
        f"  weight_format={spec.weight_format} "
        f"download_ignore_patterns={list(spec.download_ignore_patterns)}"
    )
    if args.dry_run:
        return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = destination / ".prima_download.lock"
    with lock_path.open("w") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another model download owns the destination lock: {lock_path}"
            ) from error
        if datamover_host:
            transfer = _download_via_datamover(
                spec=spec,
                destination=destination,
                host=datamover_host,
                workers=args.datamover_workers,
                attempts=args.datamover_attempts,
            )
        else:
            if not args.use_xet:
                os.environ["HF_HUB_DISABLE_XET"] = "1"
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=spec.repo_id,
                revision=spec.revision,
                local_dir=str(destination),
                ignore_patterns=list(spec.download_ignore_patterns) or None,
            )
            transfer = {"method": "direct", "xet_enabled": bool(args.use_xet)}
    if not (destination / "config.json").is_file():
        raise RuntimeError(f"download completed without config.json: {destination}")
    provenance_path = _write_provenance(
        destination=destination, spec=spec, transfer=transfer
    )
    print(f"{spec.key}: verified local snapshot and wrote {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
