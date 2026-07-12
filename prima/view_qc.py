"""Canonical state and manifest helpers for single-target view-level QC."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

VIEW_QC_SCHEMA_VERSION = 2
VIEW_QC_EVENT_SCHEMA_VERSION = 1
VIEW_QC_EVENT_FILENAME = "view_qc_events.jsonl"
VIEW_QC_EVENT_GENESIS_SHA256 = "0" * 64
VIEW_QC_EVENT_TYPES = {"label_change", "state_import"}
VIEW_LABEL_PRESENT = "present"
VIEW_LABEL_ABSENT = "absent"
VIEW_LABEL_UNCERTAIN = "uncertain"
VALID_VIEW_LABELS = {
    VIEW_LABEL_PRESENT,
    VIEW_LABEL_ABSENT,
    VIEW_LABEL_UNCERTAIN,
}

VIEW_MANIFEST_REQUIRED_COLUMNS = {
    "view_id",
    "image_path",
    "laterality",
    "view",
    "review_order",
}


def normalize_view_id(value: object) -> str:
    """Return a SHA-256 view key or fail loudly."""
    view_id = str(value).strip().lower()
    if len(view_id) != 64 or any(char not in "0123456789abcdef" for char in view_id):
        raise ValueError("view_id must be a 64-character SHA-256 digest")
    return view_id


def normalize_view_qc_target(value: object) -> str:
    """Return a concise nonempty visual QC target or fail loudly."""
    if not isinstance(value, str):
        raise ValueError("view QC target must be a string")
    target = " ".join(value.split())
    if not target:
        raise ValueError("view QC target cannot be empty")
    if len(target) > 200:
        raise ValueError("view QC target cannot exceed 200 characters")
    return target


def normalize_view_qc_reviewer(value: object) -> str:
    """Return a concise audit identity for the human or import process."""
    if not isinstance(value, str):
        raise ValueError("view QC reviewer must be a string")
    reviewer = " ".join(value.split())
    if not reviewer:
        raise ValueError("view QC reviewer cannot be empty")
    if len(reviewer) > 100:
        raise ValueError("view QC reviewer cannot exceed 100 characters")
    return reviewer


def default_view_qc_events_path(state_path: Path) -> Path:
    """Return the canonical append-only event path beside one state file."""
    return Path(state_path).with_name(VIEW_QC_EVENT_FILENAME)


def validate_view_manifest_columns(columns: Iterable[str], context: str) -> None:
    """Require the current view-review manifest schema."""
    available = set(columns)
    missing = sorted(VIEW_MANIFEST_REQUIRED_COLUMNS - available)
    if missing:
        raise ValueError(
            f"{context} is missing required view-QC columns: {', '.join(missing)}"
        )


def empty_view_qc_state(target: object) -> dict[str, Any]:
    """Build a new empty state for one frozen visual QC target."""
    return {
        "schema_version": VIEW_QC_SCHEMA_VERSION,
        "target": normalize_view_qc_target(target),
        "labels": {},
    }


def normalize_view_qc_state(payload: Any) -> dict[str, Any]:
    """Validate and normalize the current view-level state schema."""
    if not isinstance(payload, dict):
        raise ValueError("view QC state must be a JSON object")
    if payload.get("schema_version") != VIEW_QC_SCHEMA_VERSION:
        raise ValueError(
            "view QC state schema is unsupported; start a fresh view-level review"
        )
    target = normalize_view_qc_target(payload.get("target"))
    raw_labels = payload.get("labels")
    if not isinstance(raw_labels, dict):
        raise ValueError("view QC state labels must be a JSON object")

    labels: dict[str, dict[str, str]] = {}
    for raw_view_id, raw_record in raw_labels.items():
        view_id = normalize_view_id(raw_view_id)
        if not isinstance(raw_record, dict):
            raise ValueError("each view QC label must be a JSON object")
        label = str(raw_record.get("label", "")).strip()
        if label not in VALID_VIEW_LABELS:
            raise ValueError(f"invalid view QC label for {view_id[:12]}")
        source = str(raw_record.get("source", "")).strip()
        if source != "human":
            raise ValueError("view QC reference labels must have source='human'")
        updated_at = str(raw_record.get("updated_at", "")).strip()
        if not updated_at:
            raise ValueError("view QC label is missing updated_at")
        labels[view_id] = {
            "label": label,
            "source": "human",
            "updated_at": updated_at,
        }

    return {
        "schema_version": VIEW_QC_SCHEMA_VERSION,
        "target": target,
        "labels": labels,
    }


def load_view_qc_state(path: Path) -> dict[str, Any]:
    """Load a required single-target view-QC state."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"view QC state not found: {path}")
    with path.open() as handle:
        payload = json.load(handle)
    return normalize_view_qc_state(payload)


def save_view_qc_state(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically write a normalized view-level state with restricted permissions."""
    path = Path(path)
    normalized = normalize_view_qc_state(dict(state))
    normalized["labels"] = {
        view_id: normalized["labels"][view_id]
        for view_id in sorted(normalized["labels"])
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(normalized, handle, indent=2)
            handle.write("\n")
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
        os.chmod(path, 0o600)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return normalized


def set_view_label(
    state: Mapping[str, Any], view_id: object, label: str | None
) -> dict[str, Any]:
    """Set or clear one human view label and return normalized state."""
    normalized = normalize_view_qc_state(dict(state))
    key = normalize_view_id(view_id)
    labels = dict(normalized["labels"])
    if label is None:
        labels.pop(key, None)
    else:
        label = str(label).strip()
        if label not in VALID_VIEW_LABELS:
            raise ValueError(f"unsupported view QC label: {label!r}")
        labels[key] = {
            "label": label,
            "source": "human",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    normalized["labels"] = labels
    return normalize_view_qc_state(normalized)


def _require_utc_timestamp(value: object, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} cannot be empty")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return text


def _event_hash(payload: Mapping[str, Any]) -> str:
    hashed = {key: value for key, value in payload.items() if key != "event_sha256"}
    encoded = json.dumps(
        hashed, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def normalize_view_qc_event(
    payload: Any, *, expected_previous_sha256: str
) -> dict[str, Any]:
    """Validate one hash-chained append-only annotation event."""
    if not isinstance(payload, dict):
        raise ValueError("view QC event must be a JSON object")
    required = {
        "schema_version",
        "event_id",
        "event_type",
        "target",
        "view_id",
        "previous_label",
        "label",
        "source",
        "reviewer",
        "recorded_at",
        "previous_event_sha256",
        "event_sha256",
    }
    if set(payload) != required:
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise ValueError("invalid view QC event fields: " + "; ".join(details))
    if payload["schema_version"] != VIEW_QC_EVENT_SCHEMA_VERSION:
        raise ValueError("view QC event schema is unsupported")
    event_id = str(payload["event_id"]).strip().lower()
    if len(event_id) != 32 or any(char not in "0123456789abcdef" for char in event_id):
        raise ValueError("view QC event_id must be a 32-character hexadecimal value")
    event_type = str(payload["event_type"]).strip()
    if event_type not in VIEW_QC_EVENT_TYPES:
        raise ValueError(f"unsupported view QC event type: {event_type!r}")
    target = normalize_view_qc_target(payload["target"])
    view_id = normalize_view_id(payload["view_id"])
    previous_label = payload["previous_label"]
    if previous_label is not None and (
        not isinstance(previous_label, str) or previous_label not in VALID_VIEW_LABELS
    ):
        raise ValueError("view QC event has an invalid previous_label")
    label = payload["label"]
    if label is not None and (
        not isinstance(label, str) or label not in VALID_VIEW_LABELS
    ):
        raise ValueError("view QC event has an invalid label")
    if event_type == "state_import" and (previous_label is not None or label is None):
        raise ValueError("state_import must install one nonempty label")
    if str(payload["source"]).strip() != "human":
        raise ValueError("view QC events must have source='human'")
    reviewer = normalize_view_qc_reviewer(payload["reviewer"])
    recorded_at = _require_utc_timestamp(payload["recorded_at"], "recorded_at")
    previous_hash = str(payload["previous_event_sha256"]).strip().lower()
    if previous_hash != expected_previous_sha256:
        raise ValueError("view QC event hash chain is discontinuous")
    event_hash = str(payload["event_sha256"]).strip().lower()
    normalized = {
        "schema_version": VIEW_QC_EVENT_SCHEMA_VERSION,
        "event_id": event_id,
        "event_type": event_type,
        "target": target,
        "view_id": view_id,
        "previous_label": previous_label,
        "label": label,
        "source": "human",
        "reviewer": reviewer,
        "recorded_at": recorded_at,
        "previous_event_sha256": previous_hash,
        "event_sha256": event_hash,
    }
    if event_hash != _event_hash(normalized):
        raise ValueError("view QC event hash is invalid")
    return normalized


def new_view_qc_event(
    *,
    event_type: str,
    target: object,
    view_id: object,
    previous_label: str | None,
    label: str | None,
    reviewer: object,
    recorded_at: object,
    previous_event_sha256: str,
) -> dict[str, Any]:
    """Create one current-schema audit event with its chain hash."""
    payload = {
        "schema_version": VIEW_QC_EVENT_SCHEMA_VERSION,
        "event_id": uuid.uuid4().hex,
        "event_type": event_type,
        "target": normalize_view_qc_target(target),
        "view_id": normalize_view_id(view_id),
        "previous_label": previous_label,
        "label": label,
        "source": "human",
        "reviewer": normalize_view_qc_reviewer(reviewer),
        "recorded_at": _require_utc_timestamp(recorded_at, "recorded_at"),
        "previous_event_sha256": previous_event_sha256,
        "event_sha256": "",
    }
    payload["event_sha256"] = _event_hash(payload)
    return normalize_view_qc_event(
        payload, expected_previous_sha256=previous_event_sha256
    )


def load_view_qc_events(path: Path) -> list[dict[str, Any]]:
    """Load and verify an entire hash-chained annotation event log."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"view QC event log not found: {path}")
    events: list[dict[str, Any]] = []
    previous_hash = VIEW_QC_EVENT_GENESIS_SHA256
    event_ids: set[str] = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(
                    f"view QC event log has an empty line at {line_number}"
                )
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"view QC event log has invalid JSON at line {line_number}"
                ) from error
            event = normalize_view_qc_event(
                payload, expected_previous_sha256=previous_hash
            )
            if event["event_id"] in event_ids:
                raise ValueError("view QC event log contains a duplicate event_id")
            event_ids.add(event["event_id"])
            events.append(event)
            previous_hash = event["event_sha256"]
    return events


def replay_view_qc_events(
    target: object, events: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Replay verified events into the canonical current-state projection."""
    state = empty_view_qc_state(target)
    labels: dict[str, dict[str, str]] = {}
    for event in events:
        if normalize_view_qc_target(event["target"]) != state["target"]:
            raise ValueError("view QC event target does not match campaign target")
        view_id = normalize_view_id(event["view_id"])
        previous = labels.get(view_id, {}).get("label")
        if previous != event["previous_label"]:
            raise ValueError("view QC event previous_label does not match replay state")
        if event["label"] is None:
            labels.pop(view_id, None)
        else:
            labels[view_id] = {
                "label": event["label"],
                "source": "human",
                "updated_at": event["recorded_at"],
            }
    state["labels"] = labels
    return normalize_view_qc_state(state)


def _append_view_qc_event(path: Path, event: Mapping[str, Any]) -> None:
    line = json.dumps(dict(event), sort_keys=True, separators=(",", ":")) + "\n"
    encoded = line.encode()
    if len(encoded) > 4096:
        raise ValueError("view QC event exceeds the atomic append size limit")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        written = os.write(fd, encoded)
        if written != len(encoded):
            raise OSError("view QC event append was incomplete")
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(path, 0o600)


def initialize_view_qc_event_log(
    path: Path,
    state: Mapping[str, Any],
    *,
    import_reviewer: str = "system:state-import",
) -> list[dict[str, Any]]:
    """Create a new audit log, explicitly importing any preexisting labels."""
    path = Path(path)
    normalized = normalize_view_qc_state(dict(state))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    try:
        previous_hash = VIEW_QC_EVENT_GENESIS_SHA256
        for view_id, record in sorted(normalized["labels"].items()):
            event = new_view_qc_event(
                event_type="state_import",
                target=normalized["target"],
                view_id=view_id,
                previous_label=None,
                label=record["label"],
                reviewer=import_reviewer,
                recorded_at=record["updated_at"],
                previous_event_sha256=previous_hash,
            )
            _append_view_qc_event(path, event)
            previous_hash = event["event_sha256"]
        events = load_view_qc_events(path)
        if replay_view_qc_events(normalized["target"], events) != normalized:
            raise RuntimeError(
                "initialized event log does not replay to its source state"
            )
        return events
    except Exception:
        path.unlink(missing_ok=True)
        raise


def validate_view_qc_campaign_state(
    state: Mapping[str, Any],
    events: Iterable[Mapping[str, Any]],
    manifest_view_ids: Iterable[object],
) -> dict[str, Any]:
    """Require state, event replay, and manifest membership to agree exactly."""
    normalized = normalize_view_qc_state(dict(state))
    event_list = [dict(event) for event in events]
    replayed = replay_view_qc_events(normalized["target"], event_list)
    if replayed != normalized:
        raise ValueError("view QC state does not match its append-only event history")
    summarize_view_qc_state(normalized, manifest_view_ids)
    return normalized


def reconcile_view_qc_campaign_state(
    state_path: Path,
    events: Iterable[Mapping[str, Any]],
    manifest_view_ids: Iterable[object],
) -> dict[str, Any]:
    """Recover a state projection left one or more committed events behind."""
    state_path = Path(state_path)
    current = load_view_qc_state(state_path)
    event_list = [dict(event) for event in events]
    replayed = replay_view_qc_events(current["target"], event_list)
    if replayed == current:
        summarize_view_qc_state(current, manifest_view_ids)
        return current

    is_committed_prefix = any(
        replay_view_qc_events(current["target"], event_list[:prefix_length]) == current
        for prefix_length in range(len(event_list))
    )
    if not is_committed_prefix:
        raise ValueError("view QC state diverges from its append-only event history")
    summarize_view_qc_state(replayed, manifest_view_ids)
    return save_view_qc_state(state_path, replayed)


@contextmanager
def _view_qc_campaign_lock(events_path: Path):
    lock_path = Path(events_path).with_name(f".{Path(events_path).name}.lock")
    fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    os.chmod(lock_path, 0o600)


def record_view_qc_label(
    *,
    state_path: Path,
    events_path: Path,
    manifest_view_ids: Iterable[object],
    view_id: object,
    label: str | None,
    reviewer: object,
) -> dict[str, Any]:
    """Append one audit event, then atomically update the current-state projection."""
    state_path = Path(state_path)
    events_path = Path(events_path)
    manifest_ids = [normalize_view_id(value) for value in manifest_view_ids]
    key = normalize_view_id(view_id)
    if key not in set(manifest_ids):
        raise ValueError("view is outside the review manifest")
    if label is not None and label not in VALID_VIEW_LABELS:
        raise ValueError("invalid view label")
    reviewer = normalize_view_qc_reviewer(reviewer)
    with _view_qc_campaign_lock(events_path):
        events = load_view_qc_events(events_path)
        state = reconcile_view_qc_campaign_state(state_path, events, manifest_ids)
        previous_label = state["labels"].get(key, {}).get("label")
        new_state = set_view_label(state, key, label)
        if label is None:
            recorded_at = datetime.now(timezone.utc).isoformat()
        else:
            recorded_at = new_state["labels"][key]["updated_at"]
        previous_hash = (
            events[-1]["event_sha256"] if events else VIEW_QC_EVENT_GENESIS_SHA256
        )
        event = new_view_qc_event(
            event_type="label_change",
            target=state["target"],
            view_id=key,
            previous_label=previous_label,
            label=label,
            reviewer=reviewer,
            recorded_at=recorded_at,
            previous_event_sha256=previous_hash,
        )
        _append_view_qc_event(events_path, event)
        saved = save_view_qc_state(state_path, new_state)
        validate_view_qc_campaign_state(
            saved, load_view_qc_events(events_path), manifest_ids
        )
        return saved


def summarize_view_qc_state(
    state: Mapping[str, Any], manifest_view_ids: Iterable[object]
) -> dict[str, int]:
    """Return one-denominator progress counts for a review manifest."""
    normalized = normalize_view_qc_state(dict(state))
    manifest_ids = [normalize_view_id(value) for value in manifest_view_ids]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("view QC manifest contains duplicate view_id values")
    manifest_set = set(manifest_ids)
    foreign = set(normalized["labels"]) - manifest_set
    if foreign:
        raise ValueError("view QC state contains labels outside this manifest")
    labels = normalized["labels"]
    present = sum(record["label"] == VIEW_LABEL_PRESENT for record in labels.values())
    absent = sum(record["label"] == VIEW_LABEL_ABSENT for record in labels.values())
    uncertain = sum(
        record["label"] == VIEW_LABEL_UNCERTAIN for record in labels.values()
    )
    reviewed = present + absent + uncertain
    total = len(manifest_ids)
    return {
        "total": total,
        "reviewed": reviewed,
        "remaining": total - reviewed,
        "present": present,
        "absent": absent,
        "uncertain": uncertain,
    }


def render_dicom_view_png(dataset: Any, output_path: Path, max_pixels: int) -> None:
    """Render one DICOM view to an 8-bit PNG with deterministic min-max display."""
    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive")
    pixels = dataset.pixel_array.astype(np.float32, copy=False)
    if pixels.ndim != 2:
        raise ValueError("view QC requires a single-frame two-dimensional DICOM")
    slope = float(dataset.get("RescaleSlope", 1.0))
    intercept = float(dataset.get("RescaleIntercept", 0.0))
    pixels = pixels * slope + intercept
    vmin = float(np.min(pixels))
    vmax = float(np.max(pixels))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        raise ValueError("DICOM has a degenerate pixel range")
    pixels = (pixels - vmin) * (255.0 / (vmax - vmin))
    if (
        str(dataset.get("PhotometricInterpretation", "")).strip().upper()
        == "MONOCHROME1"
    ):
        pixels = 255.0 - pixels
    image = Image.fromarray(np.rint(pixels).astype(np.uint8), mode="L")
    if image.width * image.height > max_pixels:
        scale = sqrt(max_pixels / float(image.width * image.height))
        size = (
            max(1, int(image.width * scale)),
            max(1, int(image.height * scale)),
        )
        image = image.resize(size, Image.LANCZOS)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=str(output_path.parent)
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        image.save(temporary_path, format="PNG", optimize=True)
        validate_rendered_view_png(temporary_path, max_pixels=max_pixels)
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, output_path)
        os.chmod(output_path, 0o600)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def validate_rendered_view_png(path: Path, max_pixels: int) -> tuple[int, int]:
    """Validate one current-format rendered view and return its dimensions."""
    path = Path(path)
    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive")
    if not path.is_file():
        raise FileNotFoundError(f"rendered view is missing: {path}")
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG":
            raise ValueError("rendered view must be PNG")
        if image.mode != "L":
            raise ValueError("rendered view must be 8-bit grayscale")
        width, height = image.size
    if width <= 0 or height <= 0 or width * height > max_pixels:
        raise ValueError("rendered view violates the pixel budget")
    return width, height
