#!/usr/bin/env python3
"""Serve a blinded single-target reviewer for individual mammography views."""

from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from prima.view_qc import (
    VALID_VIEW_LABELS,
    default_view_qc_events_path,
    initialize_view_qc_event_log,
    load_view_qc_state,
    load_view_qc_events,
    normalize_view_id,
    normalize_view_qc_reviewer,
    reconcile_view_qc_campaign_state,
    record_view_qc_label,
    summarize_view_qc_state,
    validate_view_manifest_columns,
)

MAX_REQUEST_BYTES = 4096
DEFAULT_REVIEW_PORT = 8767


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a blinded single-target view-level QC review."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--events", type=Path, default=None)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_REVIEW_PORT)
    return parser.parse_args()


def load_review_items(
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    """Load a restricted manifest and return browser-safe item metadata."""
    manifest = pd.read_parquet(manifest_path)
    validate_view_manifest_columns(manifest.columns, str(manifest_path))
    if manifest.empty:
        raise ValueError("view QC manifest is empty")

    manifest = manifest.copy()
    manifest["view_id"] = manifest["view_id"].map(normalize_view_id)
    if manifest["view_id"].duplicated().any():
        raise ValueError("view QC manifest contains duplicate view_id values")
    manifest["review_order"] = pd.to_numeric(
        manifest["review_order"], errors="raise"
    ).astype(int)
    expected_order = list(range(1, len(manifest) + 1))
    if sorted(manifest["review_order"].tolist()) != expected_order:
        raise ValueError("review_order must contain each integer from 1 through N")
    if not manifest["laterality"].isin(["L", "R"]).all():
        raise ValueError("view QC manifest contains invalid laterality")
    if not manifest["view"].isin(["CC", "MLO"]).all():
        raise ValueError("view QC manifest contains invalid view")

    manifest_root = manifest_path.parent.resolve()
    items: list[dict[str, Any]] = []
    images: dict[str, Path] = {}
    for row in manifest.sort_values("review_order").to_dict("records"):
        relative = Path(str(row["image_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("view QC image_path must be a safe relative path")
        image_path = (manifest_root / relative).resolve()
        try:
            image_path.relative_to(manifest_root)
        except ValueError as error:
            raise ValueError("view QC image escapes the manifest directory") from error
        if not image_path.is_file():
            raise FileNotFoundError("view QC manifest references a missing image")
        if image_path.suffix.lower() != ".png":
            raise ValueError("view QC images must be PNG files")
        view_id = str(row["view_id"])
        images[view_id] = image_path
        items.append(
            {
                "view_id": view_id,
                "laterality": str(row["laterality"]),
                "view": str(row["view"]),
                "review_order": int(row["review_order"]),
                "image_url": f"/image/{view_id}",
            }
        )
    return items, images


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Single-target view QC</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { min-height: 100vh; margin: 0; overflow-y: auto; user-select: none; caret-color: transparent; background: #101214; color: #f0f2f4; }
    header { padding: 12px 18px; border-bottom: 1px solid #34383d; background: #171a1e; }
    #stats { font-variant-numeric: tabular-nums; font-weight: 650; }
    #context { color: #aeb6bf; margin-top: 5px; }
    #controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: center; padding: 10px 12px; border-bottom: 1px solid #34383d; background: #171a1e; }
    main { display: flex; align-items: flex-start; justify-content: center; padding: 16px; }
    img { display: block; width: auto; height: auto; max-width: 70vw; max-height: 65vh; object-fit: contain; background: black; }
    button { border: 1px solid #59616a; border-radius: 7px; padding: 10px 14px; color: white; background: #282d33; font-size: 16px; cursor: pointer; }
    button:hover { background: #343b43; }
    button.active { box-shadow: 0 0 0 3px #f4c542 inset; }
    #end-marker { border-left: 4px solid #66c58c; padding: 8px 12px; color: #b9e9ca; background: #193226; font-size: 16px; font-weight: 750; }
    #save-status { min-width: 145px; color: #9fd8b8; font-weight: 650; }
    #save-status.error { color: #ff9b9b; }
    .absent { background: #315d7d; }
    .present { background: #8a5b24; }
    .muted { color: #aeb6bf; }
  </style>
</head>
<body>
  <header>
    <div id="stats">Loading…</div>
    <div id="context">Loading target…</div>
  </header>
  <div id="controls">
    <button id="previous">← Previous</button>
    <button id="absent" class="absent">Not present [n]</button>
    <button id="present" class="present">Present [y]</button>
    <button id="uncertain">Unsure [u]</button>
    <button id="clear">Clear [x]</button>
    <button id="next">Next →</button>
    <span id="end-marker" role="status" hidden>✓ End of batch</span>
    <button id="pending">Next unreviewed</button>
    <button id="review-unsure">Review unsure (0)</button>
    <span id="save-status" role="status" aria-live="polite"></span>
  </div>
  <main><img id="image" alt="Mammography view"></main>
<script>
let items = [];
let labels = {};
let index = 0;
let target = '';
let saving = false;
let unsureReviewQueue = [];
let unsureReviewPosition = -1;

function unsureReviewActive() {
  return unsureReviewPosition >= 0 && unsureReviewQueue.length > 0;
}

function counts() {
  let absent = 0;
  let present = 0;
  let uncertain = 0;
  for (const item of items) {
    const label = labels[item.view_id]?.label;
    if (label === 'absent') absent += 1;
    if (label === 'present') present += 1;
    if (label === 'uncertain') uncertain += 1;
  }
  const reviewed = absent + present + uncertain;
  return {absent, present, uncertain, reviewed, remaining: items.length - reviewed};
}

function render() {
  const item = items[index];
  const summary = counts();
  if (!item) {
    document.getElementById('stats').textContent = 'No review items';
    return;
  }
  document.getElementById('image').src = item.image_url;
  const instruction = 'Target: ' + target + '. Decide only whether this target is present; ignore every other finding.';
  let context = item.laterality + ' ' + item.view + ' | ' + instruction;
  if (unsureReviewActive()) {
    context += ' Unsure review pass ' + (unsureReviewPosition + 1) + '/' + unsureReviewQueue.length + '.';
  } else if (summary.remaining === 0) {
    context = instruction + ' Review complete — all labels are saved.';
    context += summary.uncertain > 0
      ? ' Use Review unsure to revisit the unsure labels.'
      : ' Use Previous or the arrow keys to inspect them.';
  }
  document.getElementById('context').textContent = context;
  document.getElementById('stats').textContent =
    'position ' + (index + 1) + '/' + items.length +
    ' | reviewed ' + summary.reviewed + '/' + items.length +
    ' | remaining ' + summary.remaining +
    ' | absent ' + summary.absent +
    ' | present ' + summary.present +
    ' | unsure ' + summary.uncertain +
    (unsureReviewActive() ? ' | unsure pass ' + (unsureReviewPosition + 1) + '/' + unsureReviewQueue.length : '') +
    (summary.remaining === 0 ? ' | COMPLETE' : '');
  const active = labels[item.view_id]?.label;
  document.getElementById('absent').classList.toggle('active', active === 'absent');
  document.getElementById('present').classList.toggle('active', active === 'present');
  document.getElementById('uncertain').classList.toggle('active', active === 'uncertain');
  const reviewingUnsure = unsureReviewActive();
  document.getElementById('previous').disabled = reviewingUnsure
    ? unsureReviewPosition === 0
    : index === 0;
  const atEnd = reviewingUnsure
    ? unsureReviewPosition === unsureReviewQueue.length - 1
    : index === items.length - 1;
  const next = document.getElementById('next');
  const endMarker = document.getElementById('end-marker');
  next.hidden = atEnd;
  next.textContent = reviewingUnsure ? 'Next unsure →' : 'Next →';
  endMarker.hidden = !atEnd;
  endMarker.textContent = reviewingUnsure ? '✓ End of unsure review' : '✓ End of batch';
  const reviewUnsure = document.getElementById('review-unsure');
  reviewUnsure.disabled = !reviewingUnsure && summary.uncertain === 0;
  reviewUnsure.classList.toggle('active', reviewingUnsure);
  reviewUnsure.textContent = reviewingUnsure
    ? 'Exit unsure review (' + (unsureReviewPosition + 1) + '/' + unsureReviewQueue.length + ')'
    : 'Review unsure (' + summary.uncertain + ')';
}

async function setLabel(label) {
  if (saving) return;
  saving = true;
  showStatus('Saving…');
  const item = items[index];
  try {
    const response = await fetch('/api/label', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({view_id: item.view_id, label})
    });
    if (!response.ok) throw new Error(await response.text());
    const payload = await response.json();
    labels = payload.labels;
    const labelText = label === null ? 'Label cleared' :
      label === 'absent' ? 'Saved: not present' :
      label === 'present' ? 'Saved: present' : 'Saved: unsure';
    showStatus(labelText);
    if (label === null) {
      render();
    } else if (unsureReviewActive()) {
      advanceUnsureReview(labelText);
    } else if (counts().remaining > 0) {
      nextUnreviewed(index + 1);
    } else if (index < items.length - 1) {
      index += 1;
      render();
    } else {
      render();
    }
  } catch (error) {
    showStatus('Save failed: ' + error, true);
  } finally {
    saving = false;
  }
}

function showStatus(message, isError = false) {
  const status = document.getElementById('save-status');
  status.textContent = message;
  status.classList.toggle('error', isError);
}

function move(delta) {
  if (unsureReviewActive()) {
    unsureReviewPosition = Math.max(
      0,
      Math.min(unsureReviewQueue.length - 1, unsureReviewPosition + delta)
    );
    index = items.findIndex(item => item.view_id === unsureReviewQueue[unsureReviewPosition]);
    render();
    return;
  }
  index = Math.max(0, Math.min(items.length - 1, index + delta));
  render();
}

function exitUnsureReview() {
  unsureReviewQueue = [];
  unsureReviewPosition = -1;
}

function startUnsureReview() {
  unsureReviewQueue = items
    .filter(item => labels[item.view_id]?.label === 'uncertain')
    .map(item => item.view_id);
  if (unsureReviewQueue.length === 0) {
    unsureReviewPosition = -1;
    showStatus('No unsure labels to review');
    render();
    return;
  }
  unsureReviewPosition = 0;
  index = items.findIndex(item => item.view_id === unsureReviewQueue[0]);
  showStatus('Reviewing ' + unsureReviewQueue.length + ' unsure labels');
  render();
}

function advanceUnsureReview(labelText) {
  if (unsureReviewPosition < unsureReviewQueue.length - 1) {
    unsureReviewPosition += 1;
    index = items.findIndex(item => item.view_id === unsureReviewQueue[unsureReviewPosition]);
    render();
    return;
  }
  exitUnsureReview();
  showStatus(labelText + ' — unsure review pass complete');
  render();
}

function nextUnreviewed(start = 0) {
  for (let offset = 0; offset < items.length; offset += 1) {
    const candidate = (start + offset) % items.length;
    if (!labels[items[candidate].view_id]) {
      index = candidate;
      render();
      return;
    }
  }
  render();
}

document.getElementById('previous').onclick = () => move(-1);
document.getElementById('next').onclick = () => move(1);
document.getElementById('absent').onclick = () => setLabel('absent');
document.getElementById('present').onclick = () => setLabel('present');
document.getElementById('uncertain').onclick = () => setLabel('uncertain');
document.getElementById('clear').onclick = () => setLabel(null);
document.getElementById('pending').onclick = () => {
  exitUnsureReview();
  nextUnreviewed(index + 1);
};
document.getElementById('review-unsure').onclick = () => {
  if (unsureReviewActive()) {
    exitUnsureReview();
    render();
  } else {
    startUnsureReview();
  }
};
document.addEventListener('keydown', event => {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  const key = event.key.toLowerCase();
  if (!['arrowleft', 'arrowright', 'n', 'y', 'u', 'x'].includes(key)) return;
  event.preventDefault();
  if (key === 'arrowleft') move(-1);
  else if (key === 'arrowright') move(1);
  else if (key === 'n') setLabel('absent');
  else if (key === 'y') setLabel('present');
  else if (key === 'u') setLabel('uncertain');
  else if (key === 'x') setLabel(null);
});

Promise.all([
  fetch('/api/items').then(response => response.json()),
  fetch('/api/state').then(response => response.json())
]).then(([loadedItems, state]) => {
  items = loadedItems;
  labels = state.labels;
  target = state.target;
  nextUnreviewed(0);
}).catch(error => {
  document.getElementById('stats').textContent = 'Failed to load review: ' + error;
});
</script>
</body>
</html>
"""


class ReviewServer(ThreadingHTTPServer):
    items: list[dict[str, Any]]
    images: dict[str, Path]
    state_path: Path
    state_lock: threading.Lock


class ReviewHandler(BaseHTTPRequestHandler):
    server: ReviewServer

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, status: HTTPStatus) -> None:
        body = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            body = HTML.encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/items":
            self._send_json(self.server.items)
            return
        if path == "/api/state":
            with self.server.state_lock:
                events = load_view_qc_events(self.server.events_path)
                state = reconcile_view_qc_campaign_state(
                    self.server.state_path,
                    events,
                    [item["view_id"] for item in self.server.items],
                )
            self._send_json(state)
            return
        if path.startswith("/image/"):
            raw_view_id = path[len("/image/") :]
            try:
                view_id = normalize_view_id(raw_view_id)
            except ValueError:
                self._send_text("invalid image key", HTTPStatus.BAD_REQUEST)
                return
            image_path = self.server.images.get(view_id)
            if image_path is None:
                self._send_text("image not found", HTTPStatus.NOT_FOUND)
                return
            body = image_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "private, max-age=3600")
            self.end_headers()
            self.wfile.write(body)
            return
        self._send_text("not found", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/label":
            self._send_text("not found", HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            view_id = normalize_view_id(payload.get("view_id"))
            if view_id not in self.server.images:
                raise ValueError("view is outside the review manifest")
            label = payload.get("label")
            if label is not None and label not in VALID_VIEW_LABELS:
                raise ValueError("invalid view label")
            with self.server.state_lock:
                state = record_view_qc_label(
                    state_path=self.server.state_path,
                    events_path=self.server.events_path,
                    manifest_view_ids=[item["view_id"] for item in self.server.items],
                    view_id=view_id,
                    label=label,
                    reviewer=self.server.reviewer,
                )
            self._send_json(state)
        except (ValueError, json.JSONDecodeError) as error:
            self._send_text(str(error), HTTPStatus.BAD_REQUEST)


def main() -> int:
    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("view QC must bind to loopback; use an SSH tunnel remotely")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")

    manifest_path = args.manifest.resolve()
    state_path = args.state.resolve()
    events_path = (
        args.events.resolve()
        if args.events is not None
        else default_view_qc_events_path(state_path)
    )
    reviewer = normalize_view_qc_reviewer(args.reviewer)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"view QC manifest not found: {manifest_path}")
    items, images = load_review_items(manifest_path)
    state = load_view_qc_state(state_path)
    if not events_path.exists():
        if state["labels"]:
            raise FileNotFoundError(
                "nonempty view QC state has no event log; import it explicitly before review"
            )
        initialize_view_qc_event_log(events_path, state)
    events = load_view_qc_events(events_path)
    state = reconcile_view_qc_campaign_state(
        state_path, events, [item["view_id"] for item in items]
    )
    progress = summarize_view_qc_state(state, [item["view_id"] for item in items])

    server = ReviewServer((args.host, args.port), ReviewHandler)
    server.items = items
    server.images = images
    server.state_path = state_path
    server.events_path = events_path
    server.reviewer = reviewer
    server.state_lock = threading.Lock()
    print(
        "view QC ready: "
        f"total={progress['total']} reviewed={progress['reviewed']} "
        f"remaining={progress['remaining']}"
    )
    print(f"open through SSH tunnel: http://localhost:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
