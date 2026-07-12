#!/usr/bin/env python3
"""Create or verify an immutable, checksummed archive of a completed QC campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from prima.view_qc import (
    VIEW_QC_EVENT_FILENAME,
    load_view_qc_events,
    normalize_view_id,
    normalize_view_qc_target,
    validate_view_qc_campaign_state,
)

ARCHIVE_SCHEMA_VERSION = 1
ARCHIVE_METADATA_FILENAME = "archive_metadata.json"
ARCHIVE_README_FILENAME = "ARCHIVE_README.md"
ARCHIVE_DISPOSITIONS = {
    "canonical-reference",
    "supporting-adjudication",
    "invalidated",
    "superseded",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--source-dir", type=Path, required=True)
    create.add_argument("--archive-root", type=Path, required=True)
    create.add_argument("--campaign-name", required=True)
    create.add_argument(
        "--disposition", choices=sorted(ARCHIVE_DISPOSITIONS), required=True
    )
    create.add_argument("--notes", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--archive-dir", type=Path, required=True)
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_campaign_name(value: object) -> str:
    name = str(value).strip()
    if not name or len(name) > 100:
        raise ValueError("campaign name must contain 1-100 characters")
    if any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in name):
        raise ValueError("campaign name must use lowercase letters, digits, '_' or '-'")
    return name


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def load_completed_campaign(campaign_dir: Path) -> dict[str, Any]:
    """Validate the portable manifest/state contract without migrating its schema."""
    manifest_path = campaign_dir / "manifest.parquet"
    state_path = campaign_dir / "view_qc_state.json"
    image_dir = campaign_dir / "images"
    for path in (manifest_path, state_path, image_dir):
        if not path.exists():
            raise FileNotFoundError(f"completed campaign input not found: {path}")
    manifest = pd.read_parquet(manifest_path)
    required = {"view_id", "image_path", "laterality", "view", "review_order"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError("campaign manifest is missing columns: " + ", ".join(missing))
    view_ids = [normalize_view_id(value) for value in manifest["view_id"]]
    if len(view_ids) != len(set(view_ids)):
        raise ValueError("campaign manifest contains duplicate view IDs")
    state = json.loads(state_path.read_text())
    if not isinstance(state, dict) or set(state) != {
        "schema_version",
        "target",
        "labels",
    }:
        raise ValueError("campaign state has an unexpected structure")
    if state["schema_version"] not in {1, 2}:
        raise ValueError("campaign state schema is not archivable")
    target = normalize_view_qc_target(state["target"])
    if not isinstance(state["labels"], dict):
        raise ValueError("campaign labels must be a JSON object")
    label_ids = {normalize_view_id(value) for value in state["labels"]}
    if label_ids != set(view_ids):
        raise ValueError("only completely annotated campaigns may be archived")
    for view_id, record in state["labels"].items():
        if not isinstance(record, dict) or set(record) != {
            "label",
            "source",
            "updated_at",
        }:
            raise ValueError(f"campaign label record is invalid for {view_id[:12]}")
        if str(record["source"]).strip() != "human":
            raise ValueError("campaign archive accepts only human reference labels")
        if not str(record["label"]).strip() or not str(record["updated_at"]).strip():
            raise ValueError("campaign label record is incomplete")
    root = campaign_dir.resolve()
    image_paths: set[Path] = set()
    for relative_value in manifest["image_path"]:
        relative = Path(str(relative_value))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("campaign image path must be safe and relative")
        image_path = (root / relative).resolve()
        try:
            image_path.relative_to(root)
        except ValueError as error:
            raise ValueError("campaign image escapes its source directory") from error
        if not image_path.is_file():
            raise FileNotFoundError("campaign manifest references a missing image")
        image_paths.add(image_path)
    if len(image_paths) != len(view_ids):
        raise ValueError("campaign manifest does not map one image per view")
    events_path = campaign_dir / VIEW_QC_EVENT_FILENAME
    if events_path.exists():
        if state["schema_version"] != 2:
            raise ValueError("legacy campaign state cannot have current event history")
        events = load_view_qc_events(events_path)
        validate_view_qc_campaign_state(state, events, view_ids)
        event_history = "verified"
        event_count = len(events)
    elif state["schema_version"] == 2:
        event_history = "unavailable-pre-audit"
        event_count = 0
    else:
        event_history = "not-applicable-legacy-state"
        event_count = 0
    return {
        "target": target,
        "state_schema_version": int(state["schema_version"]),
        "manifest_rows": len(view_ids),
        "human_labels": len(label_ids),
        "event_history": event_history,
        "event_count": event_count,
    }


def source_files(source_dir: Path) -> list[Path]:
    files: list[Path] = []
    root = source_dir.resolve()
    for path in sorted(source_dir.rglob("*")):
        if path.is_symlink():
            target = path.resolve(strict=True)
            if not is_within(target, root) or not target.is_file():
                raise ValueError(
                    "campaign archive refuses external or non-file symbolic links"
                )
            files.append(path)
        if path.is_file():
            if not path.is_symlink():
                files.append(path)
        elif not path.is_dir():
            raise ValueError("campaign archive found an unsupported filesystem entry")
    if not files:
        raise ValueError("campaign archive source is empty")
    return files


def copy_restricted(source_dir: Path, archive_dir: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for source in source_files(source_dir):
        relative = source.relative_to(source_dir)
        destination = archive_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copyfile(source, destination)
        os.chmod(destination, 0o600)
        inventory[relative.as_posix()] = {
            "bytes": int(destination.stat().st_size),
            "sha256": sha256_file(destination),
            "source_type": ("materialized_symlink" if source.is_symlink() else "file"),
        }
    return inventory


def lock_archive(archive_dir: Path) -> None:
    """Make an archive owner-readable but not writable."""
    for path in archive_dir.rglob("*"):
        if path.is_file():
            os.chmod(path, 0o400)
    directories = [path for path in archive_dir.rglob("*") if path.is_dir()]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        os.chmod(path, 0o500)
    os.chmod(archive_dir, 0o500)


def remove_archive(archive_dir: Path) -> None:
    """Restore write permission only long enough to remove a failed new archive."""
    if not archive_dir.exists():
        return
    os.chmod(archive_dir, 0o700)
    for path in archive_dir.rglob("*"):
        if path.is_dir():
            os.chmod(path, 0o700)
        elif path.is_file():
            os.chmod(path, 0o600)
    shutil.rmtree(archive_dir)


def create_archive(args: argparse.Namespace) -> Path:
    source_dir = args.source_dir.resolve()
    archive_root = args.archive_root.resolve()
    campaign_name = normalize_campaign_name(args.campaign_name)
    disposition = str(args.disposition)
    notes = " ".join(str(args.notes).split())
    if not notes:
        raise ValueError("archive notes cannot be empty")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"campaign source directory not found: {source_dir}")
    archive_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(archive_root, 0o700)
    archive_dir = archive_root / campaign_name
    if archive_dir.exists():
        raise FileExistsError(f"refusing to replace campaign archive: {archive_dir}")
    if is_within(archive_root, source_dir):
        raise ValueError("campaign archive root cannot be inside its source")
    campaign = load_completed_campaign(source_dir)
    archive_dir.mkdir(mode=0o700)
    try:
        inventory = copy_restricted(source_dir, archive_dir)
        metadata = {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join([sys.executable, *sys.argv]),
            "campaign_name": campaign_name,
            "source_dir": str(source_dir),
            "disposition": disposition,
            "notes": notes,
            **campaign,
            "source_file_count": len(inventory),
            "source_bytes": sum(record["bytes"] for record in inventory.values()),
            "source_files": inventory,
        }
        metadata_path = archive_dir / ARCHIVE_METADATA_FILENAME
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        os.chmod(metadata_path, 0o600)
        readme_path = archive_dir / ARCHIVE_README_FILENAME
        readme_path.write_text(
            "\n".join(
                [
                    f"# Archived view-QC campaign: {campaign_name}",
                    "",
                    f"- target: `{campaign['target']}`",
                    f"- disposition: `{disposition}`",
                    f"- state schema: `{campaign['state_schema_version']}`",
                    f"- event history: `{campaign['event_history']}`",
                    f"- event count: `{campaign['event_count']}`",
                    f"- manifest rows / human labels: `{campaign['manifest_rows']}`",
                    f"- source files: `{len(inventory)}`",
                    f"- notes: {notes}",
                    "",
                    "This is an owner-readable, read-only snapshot. Checksums and exact",
                    f"source provenance are in `{ARCHIVE_METADATA_FILENAME}`. Legacy",
                    "state is preserved exactly and is not a runtime compatibility format.",
                    "",
                ]
            )
        )
        os.chmod(readme_path, 0o600)
        lock_archive(archive_dir)
        verify_archive(archive_dir)
        return archive_dir
    except Exception:
        remove_archive(archive_dir)
        raise


def verify_archive(archive_dir: Path) -> dict[str, Any]:
    archive_dir = Path(archive_dir).resolve()
    metadata_path = archive_dir / ARCHIVE_METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"campaign archive metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != ARCHIVE_SCHEMA_VERSION:
        raise ValueError("campaign archive schema is unsupported")
    inventory = metadata.get("source_files")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("campaign archive inventory is invalid")
    campaign = load_completed_campaign(archive_dir)
    for key, value in campaign.items():
        if metadata.get(key) != value:
            raise ValueError(f"campaign archive metadata disagrees on {key}")
    expected_paths = set(inventory)
    actual_paths = {
        path.relative_to(archive_dir).as_posix()
        for path in source_files(archive_dir)
        if path.name not in {ARCHIVE_METADATA_FILENAME, ARCHIVE_README_FILENAME}
    }
    if actual_paths != expected_paths:
        raise ValueError("campaign archive file inventory does not match metadata")
    for relative, record in inventory.items():
        path = archive_dir / relative
        if int(path.stat().st_size) != int(record["bytes"]):
            raise ValueError("campaign archive file size does not match metadata")
        if sha256_file(path) != str(record["sha256"]):
            raise ValueError("campaign archive checksum mismatch")
    for path in source_files(archive_dir):
        if path.stat().st_mode & 0o777 != 0o400:
            raise ValueError("campaign archive file is not read-only")
    directories = [
        archive_dir,
        *[path for path in archive_dir.rglob("*") if path.is_dir()],
    ]
    if any(path.stat().st_mode & 0o777 != 0o500 for path in directories):
        raise ValueError("campaign archive directory is not read-only")
    return metadata


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "create":
        archive_dir = create_archive(args)
        metadata = verify_archive(archive_dir)
        print(
            f"campaign archive ready: name={metadata['campaign_name']} "
            f"labels={metadata['human_labels']} files={metadata['source_file_count']}"
        )
        print(f"archive: {archive_dir}")
        return 0
    metadata = verify_archive(args.archive_dir)
    print(
        f"campaign archive verified: name={metadata['campaign_name']} "
        f"files={metadata['source_file_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
