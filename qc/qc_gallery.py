#!/usr/bin/env python3
# ruff: noqa: E402
"""qc_gallery.py

Interactive QC tool for reviewing and marking mammogram exams with persistent status tracking.

This script reads from views.parquet (output from preprocessing) and generates:
1. Combined 4-view figures (L_CC, L_MLO, R_CC, R_MLO) for each exam
2. Optional per-view debug figures with DICOM tag info
3. Interactive HTML gallery with button-based navigation, filtering, and QC marking

The gallery features:
- Single-image viewer with Previous/Next buttons (faster than scrolling)
- Keyboard navigation (left/right arrows, g for good, a to annotate, s for next, l to load more)
- Search/filter by patient ID, exam ID, or accession number
- QC button to mark exams as "good" (no findings)
- Independent annotation system: tag exams with categories (e.g., artifact types)
- QC status persistence: saves to JSON file, skips QC'd exams on next run
- Real-time QC statistics in the status bar

QC Workflow (fully keyboard-driven)
------------------------------------
1. Run script with --serve flag to generate gallery and start HTTP server
2. Open gallery in browser (with SSH port forwarding if remote)
3. Mark exams using keyboard - auto-saves to server after each action:
   - g for good (means reviewed with no issues)
   - a to annotate issues, then esc to close modal and move to next exam
   - in annotation modal, letter hotkeys toggle tags, type + enter to add new tag
   - arrow keys to navigate, s for next, l to load more
4. Re-run script to continue QC on remaining exams (by default, "good", annotated, and auto-excluded exams are skipped)
5. To re-visit annotated exams: use --qc-skip-status good auto_excluded

Data Format
-----------
Unified QC state stored in JSON (default: data/qc_state.json):
  - {exam_id: {"status": "good", "annotations": [], "annotation_meta": {}}}
  - {exam_id: {"status": null, "annotations": ["vertical line (detector artifact)"], "annotation_meta": {"vertical line (detector artifact)": {"source": "human"}}}}
Available annotation tags stored separately in data/annotation_tags.json
Optional model suggestion runs are loaded separately via --auto-run-file

Server mode (recommended for remote work):
  python qc/qc_gallery.py --serve --max-exams 100 --random
  Then forward port: ssh -L 5000:localhost:5000 user@remote
  Open: http://localhost:5000/

Local mode (no server, manual file moving):
  python qc/qc_gallery.py --max-exams 10
  Open qc_output/gallery.html in browser - local browser state is scoped by the QC state file path

Usage
-----
# start QC session with HTTP server (recommended for remote work)
python qc/qc_gallery.py --serve --max-exams 100 --random

# prioritize worst-performing exams for efficient QC (recommended!)
python qc/qc_gallery.py --serve --max-exams 100 --prioritize-errors \
  --pred-csv /path/to/validation_output.csv \
  --meta-csv /path/to/mirai_manifest.csv

# re-visit annotated exams
python qc/qc_gallery.py --serve --max-exams 100 --qc-skip-status good auto_excluded

# only show completely unmarked exams (skip all QC'd exams)
python qc/qc_gallery.py --serve --max-exams 100 --qc-skip-status good annotated auto_excluded

# QC without server (browser localStorage only)
python qc/qc_gallery.py --max-exams 10 --random

# QC specific patient with custom port
python qc/qc_gallery.py --serve --port 8080 --patient 12345

# use custom QC state file location
python qc/qc_gallery.py --serve --qc-file /path/to/my_qc_state.json

# review a saved model suggestion run against current GT
python qc/qc_gallery.py --serve --auto-run-file /path/to/auto_qc_run.json

Dependencies
------------
pydicom, pandas, matplotlib, tqdm
"""

import argparse
import gc
import json
import logging
import re
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Callable, Dict, Optional, Set, Tuple
from urllib.parse import parse_qs, urlparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from prima.auto_qc import (
    auto_run_to_tag_map,
    load_auto_run,
)
from prima.qc_state import (
    DEFAULT_ANNOTATION_TAGS,
    load_qc_state,
    normalize_annotation_meta,
    normalize_annotation_tag_catalog,
    normalize_annotation_tags,
    qc_state_to_annotations_map,
    qc_state_to_status_map,
    save_qc_state,
)
from pydicom.dataset import FileDataset
from pydicom.tag import Tag

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prima.view_selection import view_selection_key_from_dataset
from sklearn.metrics import roc_curve
from tqdm import tqdm

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# global variables for HTTP server
QC_FILE_PATH = None
ANNOTATION_TAGS_PATH = None  # list of available annotation tag strings
OUTPUT_DIR = None
VIEWS_PATH = None
TAGS_PATH = None
AUTO_RUN_PATH = None
LOAD_MORE_ARGS = None  # store args for dynamic loading
ENABLE_PRELOAD = False
_CACHED_FILTER_SETS = (
    None  # lazily computed auto-filter sets {filter_name: set(exam_id)}
)
_CACHED_AUTO_RUN = None

DEBUG_TRACKED_STATUSES = ("good", "auto_excluded")


def _normalize_saved_annotation_tags(tags):
    """Normalize saved annotation values without injecting defaults."""
    return normalize_annotation_tags(tags)


def _normalize_annotation_tag_catalog(tags):
    """Normalize the available tag catalog and keep defaults first."""
    return normalize_annotation_tag_catalog(tags)


def _load_qc_state(path: Optional[Path], persist_normalized: bool = False):
    """Load normalized QC state from disk."""
    return load_qc_state(path, persist_normalized=persist_normalized)


def _load_annotations_map(path: Optional[Path], persist_normalized: bool = False):
    """Load per-exam annotations derived from unified QC state."""
    qc_state = _load_qc_state(path, persist_normalized=persist_normalized)
    return qc_state_to_annotations_map(qc_state)


def _load_qc_status_map(path: Optional[Path], persist_normalized: bool = False):
    """Load exam_id -> status derived from unified QC state."""
    qc_state = _load_qc_state(path, persist_normalized=persist_normalized)
    return qc_state_to_status_map(qc_state)


def _load_auto_run(path: Optional[Path], persist_normalized: bool = False):
    """Load normalized auto-QC suggestion run metadata."""
    return load_auto_run(path, persist_normalized=persist_normalized)


def _blank_qc_record():
    """Create an empty QC record in the canonical schema."""
    return {"status": None, "annotations": [], "annotation_meta": {}}


def _slugify_filter_token(value: str) -> str:
    """Create a stable token for use in @filter expressions."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _status_counts(status_map: Dict[str, str]) -> Dict[str, int]:
    """Count known QC statuses and group unknown labels into 'other'."""
    counts = {status: 0 for status in DEBUG_TRACKED_STATUSES}
    other_count = 0
    for status in status_map.values():
        if status in counts:
            counts[status] += 1
        else:
            other_count += 1
    if other_count:
        counts["other"] = other_count
    return counts


def _format_status_counts(counts: Dict[str, int]) -> str:
    """Render status counts in a compact stable order for logs."""
    parts = [f"{status}={counts.get(status, 0)}" for status in DEBUG_TRACKED_STATUSES]
    if counts.get("other", 0):
        parts.append(f"other={counts['other']}")
    return ", ".join(parts)


class QCGalleryHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for serving gallery and handling QC saves."""

    def do_GET(self):
        """handle GET requests for static files and QC data."""
        parsed_path = urlparse(self.path)

        if parsed_path.path == "/load-qc":
            # return current QC data
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            if QC_FILE_PATH and QC_FILE_PATH.exists():
                qc_data = _load_qc_status_map(QC_FILE_PATH, persist_normalized=True)
                logger.info(
                    "QC DEBUG load-qc: file=%s, entries=%d, statuses=(%s)",
                    QC_FILE_PATH,
                    len(qc_data),
                    _format_status_counts(_status_counts(qc_data)),
                )
                self.wfile.write(json.dumps(qc_data).encode())
            else:
                logger.info(
                    "QC DEBUG load-qc: file missing (%s), returning empty map",
                    QC_FILE_PATH,
                )
                self.wfile.write(b"{}")
            return

        if parsed_path.path == "/load-annotation-tags":
            # return available annotation tag strings
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            if ANNOTATION_TAGS_PATH and ANNOTATION_TAGS_PATH.exists():
                with open(ANNOTATION_TAGS_PATH) as f:
                    tags = json.load(f)
                normalized_tags = _normalize_annotation_tag_catalog(tags)
                if normalized_tags != tags:
                    with open(ANNOTATION_TAGS_PATH, "w") as f:
                        json.dump(normalized_tags, f, indent=2)
                    tags = normalized_tags
                self.wfile.write(json.dumps(tags).encode())
            else:
                default_tags = DEFAULT_ANNOTATION_TAGS
                self.wfile.write(json.dumps(default_tags).encode())
            return

        if parsed_path.path == "/load-annotations":
            # return per-exam annotations (exam_id -> [tags])
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                annotations = _load_annotations_map(
                    QC_FILE_PATH, persist_normalized=True
                )
                self.wfile.write(json.dumps(annotations).encode())
            except Exception as e:
                logger.warning(f"failed to load annotations: {e}")
                self.wfile.write(b"{}")
            return

        if parsed_path.path == "/load-auto-suggestions":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                auto_run = _load_auto_run(AUTO_RUN_PATH, persist_normalized=True)
                self.wfile.write(json.dumps(auto_run).encode())
            except Exception as e:
                logger.warning(f"failed to load auto suggestions: {e}")
                self.wfile.write(b"{}")
            return

        if parsed_path.path == "/list-filters":
            # list auto-computed filters (from parquet data) plus any static .txt files
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            filters = []

            try:
                for name, filter_spec in self._get_filter_catalog().items():
                    exam_set = filter_spec["exam_ids"]
                    filters.append(
                        {
                            "name": filter_spec["label"] + f" ({len(exam_set)})",
                            "filename": name,
                            "token": filter_spec.get("token", name),
                            "aliases": filter_spec.get("aliases", []),
                            "kind": filter_spec.get("kind", "auto"),
                        }
                    )
            except Exception as e:
                logger.warning(f"failed to compute auto-filter sets: {e}")
            self.wfile.write(json.dumps(filters).encode())
            return

        if parsed_path.path.startswith("/load-filter/"):
            filter_name = parsed_path.path.split("/load-filter/", 1)[1]

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                fs = self._get_filter_sets()
                key = filter_name.replace(".txt", "")
                if key in fs:
                    self.wfile.write(json.dumps({"exam_ids": sorted(fs[key])}).encode())
                else:
                    self.wfile.write(
                        json.dumps({"error": f"Unknown filter: {filter_name}"}).encode()
                    )
            except Exception as e:
                logger.warning(
                    f"failed to compute auto-filter set for {filter_name}: {e}"
                )
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        if parsed_path.path == "/cutflow":
            # compute and return cutflow analysis
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                cutflow_data = self._compute_cutflow()
                self.wfile.write(json.dumps(cutflow_data).encode())
            except Exception as e:
                import traceback

                traceback.print_exc()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        if parsed_path.path == "/preload-more":
            # preload next batch in background (called at 80% progress)
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                if not ENABLE_PRELOAD:
                    self.wfile.write(
                        json.dumps(
                            {
                                "status": "disabled",
                                "message": "Preloading is disabled",
                            }
                        ).encode()
                    )
                    return

                if not LOAD_MORE_ARGS:
                    self.wfile.write(
                        json.dumps({"error": "Load more not configured"}).encode()
                    )
                    return

                # check if preload already in progress
                if (
                    hasattr(self.server, "_preload_in_progress")
                    and self.server._preload_in_progress
                ):
                    self.wfile.write(json.dumps({"status": "already_loading"}).encode())
                    return

                # keep batch size fixed across load-more cycles
                current_batch = LOAD_MORE_ARGS.get("max_exams") or 100
                next_batch = current_batch

                logger.info(
                    f"Preloading next batch with fixed batch size: {next_batch}"
                )

                # trigger gallery regeneration for the next fixed-size batch in background
                import threading

                def preload_regenerate():
                    try:
                        self.server._preload_in_progress = True
                        from copy import deepcopy

                        args = deepcopy(LOAD_MORE_ARGS)
                        args["max_exams"] = next_batch
                        args["exam_list_path"] = None  # ensure unfiltered
                        generate_gallery(**args)
                        logger.info(f"✓ Preload complete: {next_batch} exams ready")
                        self.server._preload_ready = True
                    except Exception as e:
                        logger.error(f"Failed to preload exams: {e}")
                        import traceback

                        traceback.print_exc()
                    finally:
                        self.server._preload_in_progress = False

                thread = threading.Thread(target=preload_regenerate)
                thread.daemon = True
                thread.start()

                self.wfile.write(
                    json.dumps(
                        {
                            "status": "preloading",
                            "message": f"Preloading {next_batch} exams in background...",
                        }
                    ).encode()
                )

            except Exception as e:
                logger.error(f"error preloading: {e}")
                import traceback

                traceback.print_exc()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        if parsed_path.path == "/regenerate":
            # refresh current batch gallery with latest QC state
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                if not LOAD_MORE_ARGS:
                    self.wfile.write(
                        json.dumps({"error": "Regenerate not configured"}).encode()
                    )
                    return

                # discard any stale preload
                self.server._preload_ready = False

                current_batch = LOAD_MORE_ARGS.get("max_exams") or 100
                logger.info(f"Refreshing current batch gallery: {current_batch} exams")

                import threading

                def regenerate():
                    try:
                        from copy import deepcopy

                        args = deepcopy(LOAD_MORE_ARGS)
                        args["exam_list_path"] = None  # ensure unfiltered
                        generate_gallery(**args)
                        logger.info(
                            f"Successfully refreshed gallery for {current_batch} exams"
                        )
                    except Exception as e:
                        logger.error(f"Failed to regenerate exams: {e}")
                        import traceback

                        traceback.print_exc()

                thread = threading.Thread(target=regenerate)
                thread.daemon = True
                thread.start()

                self.wfile.write(
                    json.dumps(
                        {
                            "status": "ok",
                            "message": (
                                f"Refreshing gallery for {current_batch} exams..."
                            ),
                            "reload_delay_ms": 3000,
                        }
                    ).encode()
                )
            except Exception as e:
                logger.error(f"error regenerating: {e}")
                import traceback

                traceback.print_exc()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        if parsed_path.path == "/load-more":
            # always regenerate fresh to pick up latest QC state
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                if not LOAD_MORE_ARGS:
                    self.wfile.write(
                        json.dumps({"error": "Load more not configured"}).encode()
                    )
                    return

                # discard any stale preload (QC state has changed since preload started)
                self.server._preload_ready = False

                current_batch = LOAD_MORE_ARGS.get("max_exams") or 100
                # allow client to override batch size via query param
                qs = parse_qs(parsed_path.query)
                if "batch_size" in qs:
                    try:
                        next_batch = max(1, int(qs["batch_size"][0]))
                    except (ValueError, IndexError):
                        next_batch = current_batch
                else:
                    next_batch = current_batch
                logger.info(
                    "QC DEBUG load-more: qc_file=%s, current_batch=%s, next_batch=%s, prioritize_errors=%s, random_sample=%s, qc_skip_status=%s",
                    QC_FILE_PATH,
                    current_batch,
                    next_batch,
                    LOAD_MORE_ARGS.get("prioritize_errors"),
                    LOAD_MORE_ARGS.get("random_sample"),
                    sorted(LOAD_MORE_ARGS.get("qc_skip_status") or []),
                )
                if QC_FILE_PATH and QC_FILE_PATH.exists():
                    try:
                        with open(QC_FILE_PATH) as f:
                            qc_snapshot = json.load(f)
                        if isinstance(qc_snapshot, dict):
                            qc_snapshot = {
                                str(exam_id): status
                                for exam_id, status in qc_snapshot.items()
                            }
                            logger.info(
                                "QC DEBUG load-more: qc file has %d entries (%s)",
                                len(qc_snapshot),
                                _format_status_counts(_status_counts(qc_snapshot)),
                            )
                        else:
                            logger.warning(
                                "QC DEBUG load-more: qc file is not a JSON object (%s)",
                                type(qc_snapshot).__name__,
                            )
                    except Exception as e:
                        logger.warning(
                            "QC DEBUG load-more: failed to read qc file: %s", e
                        )

                if (
                    hasattr(self.server, "_load_more_in_progress")
                    and self.server._load_more_in_progress
                ):
                    self.wfile.write(
                        json.dumps(
                            {
                                "status": "already_loading",
                                "message": "Load-more generation already in progress",
                                "next_batch": self.server._load_more_target_batch,
                            }
                        ).encode()
                    )
                    return

                self.server._load_more_in_progress = True
                self.server._load_more_ready = False
                self.server._load_more_error = None
                self.server._load_more_target_batch = next_batch
                self.server._load_more_progress = 0.0
                self.server._load_more_message = "Starting load-more..."

                logger.info(
                    f"Loading more exams (fresh): keeping fixed batch size at {next_batch}"
                )

                # trigger gallery regeneration for the next fixed-size batch
                import threading

                def regenerate():
                    def update_progress(progress: float, detail: str) -> None:
                        self.server._load_more_progress = progress
                        self.server._load_more_message = detail

                    try:
                        from copy import deepcopy

                        args = deepcopy(LOAD_MORE_ARGS)
                        args["max_exams"] = next_batch
                        args["exam_list_path"] = None  # ensure unfiltered
                        args["progress_callback"] = update_progress
                        generate_gallery(**args)
                        self.server._load_more_progress = 1.0
                        self.server._load_more_message = "Next batch ready."
                        LOAD_MORE_ARGS["max_exams"] = next_batch
                        self.server._load_more_ready = True
                        logger.info(f"Successfully generated {next_batch} exams")
                    except Exception as e:
                        self.server._load_more_error = str(e)
                        self.server._load_more_message = "Generation failed."
                        logger.error(f"Failed to generate more exams: {e}")
                        import traceback

                        traceback.print_exc()
                    finally:
                        self.server._load_more_in_progress = False

                thread = threading.Thread(target=regenerate)
                thread.daemon = True
                thread.start()

                self.wfile.write(
                    json.dumps(
                        {
                            "status": "ok",
                            "message": f"Generating {next_batch} exams...",
                            "next_batch": next_batch,
                        }
                    ).encode()
                )

            except Exception as e:
                self.server._load_more_in_progress = False
                self.server._load_more_error = str(e)
                logger.error(f"error loading more: {e}")
                import traceback

                traceback.print_exc()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        if parsed_path.path == "/load-filter-batch":
            # regenerate gallery with exams matching a specific filter
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                if not LOAD_MORE_ARGS:
                    self.wfile.write(
                        json.dumps({"error": "Load more not configured"}).encode()
                    )
                    return

                qs = parse_qs(parsed_path.query)
                filter_name = qs.get("filter", [None])[0]
                if not filter_name:
                    self.wfile.write(
                        json.dumps({"error": "Missing ?filter= parameter"}).encode()
                    )
                    return

                filter_catalog = self._get_filter_catalog()
                filter_spec = filter_catalog.get(filter_name)
                if filter_spec is None:
                    self.wfile.write(
                        json.dumps({"error": f"Unknown filter: {filter_name}"}).encode()
                    )
                    return

                filter_exam_ids = filter_spec["exam_ids"]
                filter_kind = filter_spec.get("kind", "auto")
                try:
                    requested_batch = qs.get("batch_size", [None])[0]
                    batch_size = (
                        max(1, int(requested_batch))
                        if requested_batch is not None
                        else len(filter_exam_ids)
                    )
                except (ValueError, TypeError):
                    batch_size = len(filter_exam_ids)
                batch_size = min(batch_size, len(filter_exam_ids))

                if (
                    hasattr(self.server, "_load_more_in_progress")
                    and self.server._load_more_in_progress
                ):
                    self.wfile.write(
                        json.dumps(
                            {
                                "status": "already_loading",
                                "message": "Generation already in progress",
                            }
                        ).encode()
                    )
                    return

                self.server._load_more_in_progress = True
                self.server._load_more_ready = False
                self.server._load_more_error = None
                self.server._load_more_target_batch = batch_size
                self.server._load_more_progress = 0.0
                self.server._load_more_message = "Starting filtered load..."

                logger.info(
                    f"Loading filter batch: filter={filter_name}, "
                    f"matching={len(filter_exam_ids)}, batch_size={batch_size}"
                )

                import tempfile
                import threading

                # write filter exam IDs to a temp file for generate_gallery
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, prefix="qc_filter_"
                )
                for eid in sorted(filter_exam_ids):
                    tmp.write(eid + "\n")
                tmp.close()
                tmp_path = Path(tmp.name)

                def regenerate():
                    def update_progress(progress: float, detail: str) -> None:
                        self.server._load_more_progress = progress
                        self.server._load_more_message = detail

                    try:
                        from copy import deepcopy

                        args = deepcopy(LOAD_MORE_ARGS)
                        args["max_exams"] = batch_size
                        args["exam_list_path"] = tmp_path
                        args["prioritize_errors"] = False
                        args["random_sample"] = False
                        args["progress_callback"] = update_progress
                        if str(filter_kind).startswith("annotation"):
                            requested_skip = args.get("qc_skip_status")
                            if requested_skip is None:
                                requested_skip = {"good", "annotated", "auto_excluded"}
                            else:
                                requested_skip = set(requested_skip)
                            args["qc_skip_status"] = {
                                status
                                for status in requested_skip
                                if status != "annotated"
                            }
                            logger.info(
                                "annotation filter batch requested; overriding qc_skip_status to %s",
                                sorted(args["qc_skip_status"]),
                            )
                        generate_gallery(**args)
                        self.server._load_more_progress = 1.0
                        self.server._load_more_message = "Filtered batch ready."
                        self.server._load_more_ready = True
                        logger.info(
                            f"Successfully generated filter batch: "
                            f"{filter_name} ({batch_size} exams)"
                        )
                    except Exception as e:
                        self.server._load_more_error = str(e)
                        self.server._load_more_message = "Generation failed."
                        logger.error(f"Failed to generate filter batch: {e}")
                        import traceback

                        traceback.print_exc()
                    finally:
                        self.server._load_more_in_progress = False
                        tmp_path.unlink(missing_ok=True)

                thread = threading.Thread(target=regenerate)
                thread.daemon = True
                thread.start()

                self.wfile.write(
                    json.dumps(
                        {
                            "status": "ok",
                            "message": f"Loading {batch_size} {filter_name} exams...",
                            "filter": filter_name,
                            "matching": len(filter_exam_ids),
                            "batch_size": batch_size,
                        }
                    ).encode()
                )

            except Exception as e:
                self.server._load_more_in_progress = False
                logger.error(f"error loading filter batch: {e}")
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        if parsed_path.path == "/load-more-status":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            status = {
                "in_progress": bool(
                    getattr(self.server, "_load_more_in_progress", False)
                ),
                "ready": bool(getattr(self.server, "_load_more_ready", False)),
                "error": getattr(self.server, "_load_more_error", None),
                "next_batch": getattr(self.server, "_load_more_target_batch", None),
                "progress": getattr(self.server, "_load_more_progress", None),
                "message": getattr(self.server, "_load_more_message", None),
            }
            self.wfile.write(json.dumps(status).encode())
            return

        # silently ignore favicon requests
        if parsed_path.path == "/favicon.ico":
            self.send_response(204)  # No Content
            self.end_headers()
            return

        # serve static files from output directory
        if parsed_path.path == "/" or parsed_path.path == "":
            self.path = "/gallery.html"

        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        """handle POST requests for saving QC data."""
        parsed_path = urlparse(self.path)

        if parsed_path.path == "/save-qc":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)

            try:
                qc_updates = json.loads(post_data.decode())
                if not isinstance(qc_updates, dict):
                    raise ValueError("QC payload must be a JSON object")
                qc_updates = {
                    str(exam_id): status for exam_id, status in qc_updates.items()
                }

                # save to file
                if QC_FILE_PATH:
                    replace_mode = parse_qs(parsed_path.query).get("replace", ["0"])[
                        0
                    ].lower() in {"1", "true", "yes"}
                    QC_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

                    qc_state = _load_qc_state(QC_FILE_PATH, persist_normalized=True)
                    if replace_mode:
                        kept_records = {
                            exam_id: qc_state.get(exam_id, _blank_qc_record())
                            for exam_id in qc_state
                        }
                        qc_state = {
                            exam_id: {
                                "status": qc_updates.get(exam_id),
                                "annotations": []
                                if qc_updates.get(exam_id) == "good"
                                else list(
                                    kept_records.get(exam_id, _blank_qc_record()).get(
                                        "annotations", []
                                    )
                                ),
                                "annotation_meta": {}
                                if qc_updates.get(exam_id) == "good"
                                else normalize_annotation_meta(
                                    kept_records.get(exam_id, _blank_qc_record()).get(
                                        "annotation_meta"
                                    ),
                                    normalize_annotation_tags(
                                        kept_records.get(
                                            exam_id, _blank_qc_record()
                                        ).get("annotations", [])
                                    ),
                                ),
                            }
                            for exam_id in set(qc_updates) | set(kept_records)
                        }
                    else:
                        for exam_id, status in qc_updates.items():
                            record = qc_state.get(exam_id, _blank_qc_record())
                            record["status"] = status
                            if status == "good":
                                record["annotations"] = []
                                record["annotation_meta"] = {}
                            qc_state[exam_id] = record

                    qc_state = save_qc_state(QC_FILE_PATH, qc_state)
                    qc_data = qc_state_to_status_map(qc_state)

                    logger.info(
                        f"saved QC data: {len(qc_updates)} updates ({len(qc_data)} total, replace={replace_mode})"
                    )
                    logger.info(
                        "QC DEBUG save-qc: file=%s, statuses=(%s)",
                        QC_FILE_PATH,
                        _format_status_counts(_status_counts(qc_data)),
                    )

                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {
                                "status": "ok",
                                "saved_updates": len(qc_updates),
                                "total_entries": len(qc_data),
                                "replace": replace_mode,
                            }
                        ).encode()
                    )
                else:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b'{"error": "QC file path not configured"}')
            except Exception as e:
                logger.error(f"error saving QC data: {e}")
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        elif parsed_path.path == "/save-annotation-tags":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)

            try:
                tags = json.loads(post_data.decode())

                if ANNOTATION_TAGS_PATH:
                    tags = _normalize_annotation_tag_catalog(tags)
                    ANNOTATION_TAGS_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with open(ANNOTATION_TAGS_PATH, "w") as f:
                        json.dump(tags, f, indent=2)

                    logger.info(f"saved annotation tags: {len(tags)} categories")

                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(b'{"status": "ok"}')
                else:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(
                        b'{"error": "Annotation tags path not configured"}'
                    )
            except Exception as e:
                logger.error(f"error saving annotation tags: {e}")
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        elif parsed_path.path == "/save-annotations":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)

            try:
                annotations = {
                    str(exam_id): _normalize_saved_annotation_tags(tags)
                    for exam_id, tags in json.loads(post_data.decode()).items()
                }

                if QC_FILE_PATH:
                    qc_state = _load_qc_state(QC_FILE_PATH, persist_normalized=True)
                    existing_statuses = qc_state_to_status_map(qc_state)
                    qc_state = {
                        exam_id: {
                            "status": existing_statuses.get(exam_id),
                            "annotations": annotations.get(exam_id, []),
                            "annotation_meta": normalize_annotation_meta(
                                qc_state.get(exam_id, _blank_qc_record()).get(
                                    "annotation_meta"
                                ),
                                annotations.get(exam_id, []),
                            ),
                        }
                        for exam_id in set(existing_statuses) | set(annotations)
                    }
                    save_qc_state(QC_FILE_PATH, qc_state)

                    logger.info(f"saved annotations: {len(annotations)} exams")

                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(b'{"status": "ok"}')
                else:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b'{"error": "QC file path not configured"}')
            except Exception as e:
                logger.error(f"error saving annotations: {e}")
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        elif parsed_path.path == "/accept-auto-suggestions":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)

            try:
                payload = json.loads(post_data.decode())
                exam_ids = payload.get("exam_ids", [])
                if not isinstance(exam_ids, list):
                    raise ValueError("exam_ids must be a JSON list")
                exam_ids = [str(exam_id) for exam_id in exam_ids]

                if QC_FILE_PATH is None:
                    raise ValueError("QC file path not configured")
                if AUTO_RUN_PATH is None:
                    raise ValueError("Auto suggestion run path not configured")

                qc_state = _load_qc_state(QC_FILE_PATH, persist_normalized=True)
                auto_run = _load_auto_run(AUTO_RUN_PATH, persist_normalized=True)
                if not auto_run:
                    raise ValueError(f"Auto suggestion run not found: {AUTO_RUN_PATH}")

                exam_suggestions = auto_run.get("exam_suggestions", {})
                total_added_tags = 0
                total_exam_updates = 0
                touched_exam_ids: list[str] = []
                for exam_id in exam_ids:
                    suggestion_record = exam_suggestions.get(exam_id)
                    if not suggestion_record:
                        continue

                    record = qc_state.get(exam_id, _blank_qc_record())
                    annotations = normalize_annotation_tags(
                        record.get("annotations", [])
                    )
                    annotation_meta = normalize_annotation_meta(
                        record.get("annotation_meta"), annotations
                    )
                    before_annotations = list(annotations)

                    for suggestion in suggestion_record.get("suggestions", []):
                        tag = suggestion["tag"]
                        if tag not in annotations:
                            annotations.append(tag)
                            total_added_tags += 1
                        existing_meta = annotation_meta.get(tag, {"source": "auto"})
                        if existing_meta.get("source") != "human":
                            new_meta = {"source": "auto"}
                            if auto_run.get("run_id"):
                                new_meta["origin_run_id"] = auto_run["run_id"]
                            model_name = (
                                suggestion_record.get("model")
                                or auto_run.get("model")
                                or None
                            )
                            if model_name:
                                new_meta["model"] = str(model_name)
                            if suggestion.get("score") is not None:
                                new_meta["score"] = suggestion.get("score")
                            annotation_meta[tag] = new_meta

                    if annotations != before_annotations:
                        total_exam_updates += 1
                        touched_exam_ids.append(exam_id)
                    record["annotations"] = annotations
                    record["annotation_meta"] = normalize_annotation_meta(
                        annotation_meta, annotations
                    )
                    if record.get("status") == "good" and annotations:
                        record["status"] = None
                    qc_state[exam_id] = record

                save_qc_state(QC_FILE_PATH, qc_state)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "status": "ok",
                            "updated_exams": total_exam_updates,
                            "added_tags": total_added_tags,
                            "exam_ids": touched_exam_ids,
                        }
                    ).encode()
                )
            except Exception as e:
                logger.error(f"error accepting auto suggestions: {e}")
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def _get_auto_filter_sets(self):
        """Compute or return cached auto-filter exam-id sets from parquet data."""
        global _CACHED_FILTER_SETS
        if _CACHED_FILTER_SETS is not None:
            return _CACHED_FILTER_SETS

        if not VIEWS_PATH or not TAGS_PATH:
            return {}

        from prima.qc_filters import compute_auto_filter_sets

        fs = compute_auto_filter_sets(VIEWS_PATH, TAGS_PATH)
        logger.info(
            "computed auto-filter sets: %s",
            ", ".join(f"{k}={len(v)}" for k, v in fs.items()),
        )
        _CACHED_FILTER_SETS = fs
        return fs

    def _get_auto_run(self):
        """Load or return cached model suggestion run metadata."""
        global _CACHED_AUTO_RUN
        if _CACHED_AUTO_RUN is not None:
            return _CACHED_AUTO_RUN

        if not AUTO_RUN_PATH:
            return {}

        auto_run = _load_auto_run(AUTO_RUN_PATH, persist_normalized=True)
        _CACHED_AUTO_RUN = auto_run
        return auto_run

    def _get_filter_catalog(self):
        """Build combined filter metadata for auto filters and saved annotations."""
        auto_sets = self._get_auto_filter_sets()
        catalog = {
            name: {
                "exam_ids": exam_set,
                "label": name.replace("_", " ").title(),
                "token": name,
                "aliases": [name],
                "kind": "auto",
            }
            for name, exam_set in auto_sets.items()
        }

        auto_run = self._get_auto_run()
        auto_tag_map = auto_run_to_tag_map(auto_run)
        if auto_tag_map:
            auto_exam_ids = set(auto_tag_map)
            catalog["auto_suggested"] = {
                "exam_ids": auto_exam_ids,
                "label": "Auto Suggestion: Any Tag",
                "token": "auto_suggested",
                "aliases": ["auto_suggested", "model_suggested"],
                "kind": "auto_suggestion_group",
            }

            auto_tag_to_exams: dict[str, set[str]] = {}
            for exam_id, tags in auto_tag_map.items():
                for tag in tags:
                    auto_tag_to_exams.setdefault(tag, set()).add(exam_id)

            used_filenames = set(catalog)
            used_tokens = {
                filter_spec["token"]
                for filter_spec in catalog.values()
                if filter_spec.get("token")
            }

            for tag, exam_ids in sorted(
                auto_tag_to_exams.items(),
                key=lambda item: (-len(item[1]), item[0].lower()),
            ):
                base_token = _slugify_filter_token(tag)
                if not base_token:
                    continue

                token = f"auto_{base_token}"
                if token in used_tokens:
                    suffix = 2
                    while f"{token}_{suffix}" in used_tokens:
                        suffix += 1
                    token = f"{token}_{suffix}"
                used_tokens.add(token)

                filename = f"auto_suggestion_{base_token}"
                if filename in used_filenames:
                    suffix = 2
                    while f"{filename}_{suffix}" in used_filenames:
                        suffix += 1
                    filename = f"{filename}_{suffix}"
                used_filenames.add(filename)

                catalog[filename] = {
                    "exam_ids": exam_ids,
                    "label": f"Auto Suggestion: {tag}",
                    "token": token,
                    "aliases": [token, filename],
                    "kind": "auto_suggestion",
                }

        annotations = _load_annotations_map(QC_FILE_PATH, persist_normalized=True)
        if not annotations:
            return catalog

        annotated_exams = set(annotations)
        if annotated_exams:
            catalog["manual_qc_annotated"] = {
                "exam_ids": annotated_exams,
                "label": "Annotation: Any Finding",
                "token": "annotated",
                "aliases": ["annotated", "manual_qc_annotated"],
                "kind": "annotation_group",
            }

        tag_to_exams = {}
        for exam_id, tags in annotations.items():
            for tag in tags:
                tag_to_exams.setdefault(tag, set()).add(exam_id)

        used_filenames = set(catalog)
        used_tokens = {
            filter_spec["token"]
            for filter_spec in catalog.values()
            if filter_spec.get("token")
        }
        for tag, exam_ids in sorted(
            tag_to_exams.items(), key=lambda item: (-len(item[1]), item[0].lower())
        ):
            base_token = _slugify_filter_token(tag)
            if not base_token:
                continue

            token = base_token
            if token in used_tokens:
                suffix = 2
                while f"{base_token}_{suffix}" in used_tokens:
                    suffix += 1
                token = f"{base_token}_{suffix}"
            used_tokens.add(token)

            filename = f"annotation_{token}"
            if filename in used_filenames:
                suffix = 2
                while f"{filename}_{suffix}" in used_filenames:
                    suffix += 1
                filename = f"{filename}_{suffix}"
            used_filenames.add(filename)

            aliases = [token, filename]
            catalog[filename] = {
                "exam_ids": exam_ids,
                "label": f"Annotation: {tag}",
                "token": token,
                "aliases": aliases,
                "kind": "annotation",
            }

        return catalog

    def _get_filter_sets(self):
        """Return combined filter exam-id sets keyed by filename."""
        return {
            name: filter_spec["exam_ids"]
            for name, filter_spec in self._get_filter_catalog().items()
        }

    def _compute_cutflow(self):
        """compute cutflow analysis on demand"""
        if not VIEWS_PATH or not TAGS_PATH:
            return {"error": "Views/tags paths not configured"}

        filter_sets = self._get_filter_sets()

        views_df = pd.read_parquet(VIEWS_PATH)
        views_df["exam_id"] = views_df["exam_id"].astype(str)
        total_exams = views_df["exam_id"].nunique()

        # load QC data
        qc_data = _load_qc_status_map(QC_FILE_PATH, persist_normalized=True)
        # load annotations (exam_id -> list of tags)
        annotations_data = _load_annotations_map(QC_FILE_PATH, persist_normalized=True)

        cutflow = []
        excluded_exams: set = set()

        cutflow.append(
            {
                "step": "0. Total",
                "filter": "All exams in dataset",
                "flagged_by_filter": 0,
                "exams_excluded_this_step": 0,
                "total_excluded": 0,
                "exams_remaining": total_exams,
                "percent_remaining": 100.0,
            }
        )

        # ordered filter steps
        filter_steps = [
            ("1. Implants", "has_implant == True", "has_implant"),
            ("2. Scanned Film", "DetectorType == FILM", "scanned_film"),
            ("3. GEMS", "All GE GEMS processing codes (GEMS_*)", "gems_ffdm_tc1"),
            (
                "4. Duplicate Exams",
                "Shared SOP Instance UIDs within patient",
                "duplicate_sop_uid",
            ),
        ]

        for step_name, filter_desc, filter_key in filter_steps:
            if filter_key not in filter_sets:
                continue
            matched = filter_sets[filter_key]
            new_excluded = matched - excluded_exams
            excluded_exams.update(matched)
            cutflow.append(
                {
                    "step": step_name,
                    "filter": filter_desc,
                    "flagged_by_filter": len(matched),
                    "exams_excluded_this_step": len(new_excluded),
                    "total_excluded": len(excluded_exams),
                    "exams_remaining": total_exams - len(excluded_exams),
                    "percent_remaining": (total_exams - len(excluded_exams))
                    / total_exams
                    * 100,
                }
            )

        # add manually annotated exams as final filter step
        annotated_exams = {
            eid
            for eid, tags in annotations_data.items()
            if isinstance(tags, list) and len(tags) > 0
        }
        if annotated_exams:
            new_excluded = annotated_exams - excluded_exams
            excluded_exams.update(annotated_exams)
            cutflow.append(
                {
                    "step": "5. Manual QC",
                    "filter": "Manually annotated with findings",
                    "flagged_by_filter": len(annotated_exams),
                    "exams_excluded_this_step": len(new_excluded),
                    "total_excluded": len(excluded_exams),
                    "exams_remaining": total_exams - len(excluded_exams),
                    "percent_remaining": (total_exams - len(excluded_exams))
                    / total_exams
                    * 100,
                }
            )

        # Compute filter effectiveness if we have manual annotation data.
        filter_effectiveness = None
        summary = None

        if annotations_data:
            annotated_exams = {
                eid
                for eid, tags in annotations_data.items()
                if isinstance(tags, list) and len(tags) > 0
            }
            good_exams = {eid for eid, status in qc_data.items() if status == "good"}
            good_exams -= annotated_exams

            effectiveness = []
            total_caught = set()

            for filter_name, filter_exams in filter_sets.items():
                caught_annotated = annotated_exams & filter_exams
                caught_good = good_exams & filter_exams
                total_caught.update(caught_annotated)

                if len(caught_annotated) > 0 or len(caught_good) > 0:
                    precision = len(caught_annotated) / (
                        len(caught_annotated) + len(caught_good)
                    )
                    effectiveness.append(
                        {
                            "filter": filter_name,
                            "annotated_caught": len(caught_annotated),
                            "good_caught": len(caught_good),
                            "precision": precision,
                        }
                    )

            filter_effectiveness = sorted(
                effectiveness, key=lambda x: x["annotated_caught"], reverse=True
            )
            summary = {
                "total_annotated": len(annotated_exams),
                "caught_by_filters": len(total_caught),
            }

        return {
            "total_exams": total_exams,
            "cutflow": cutflow,
            "filter_effectiveness": filter_effectiveness,
            "summary": summary,
        }

    def log_message(self, format, *args):
        """override to use our logger."""
        # skip logging favicon requests
        if "favicon.ico" in format % args:
            return
        logger.info(f"HTTP: {format % args}")


def start_qc_server(
    output_dir: Path,
    qc_file: Path,
    views_path: Path,
    tags_path: Path,
    auto_run_path: Optional[Path] = None,
    port: int = 5000,
    load_more_args: Optional[dict] = None,
) -> None:
    """start HTTP server to serve gallery and handle QC saves.

    parameters
    ----------
    output_dir : Path
        directory containing gallery.html and images
    qc_file : Path
        path where QC data should be saved
    views_path : Path
        path to views.parquet (for cutflow)
    tags_path : Path
        path to dicom_tags.parquet (for cutflow)
    port : int
        port to listen on
    load_more_args : Optional[dict]
        arguments for dynamically loading more exams
    """
    global \
        QC_FILE_PATH, \
        ANNOTATION_TAGS_PATH, \
        OUTPUT_DIR, \
        VIEWS_PATH, \
        TAGS_PATH, \
        AUTO_RUN_PATH, \
        LOAD_MORE_ARGS, \
        _CACHED_FILTER_SETS, \
        _CACHED_AUTO_RUN

    QC_FILE_PATH = qc_file.resolve()
    ANNOTATION_TAGS_PATH = QC_FILE_PATH.parent / "annotation_tags.json"
    OUTPUT_DIR = output_dir.resolve()
    VIEWS_PATH = views_path.resolve()
    TAGS_PATH = tags_path.resolve()
    AUTO_RUN_PATH = auto_run_path.resolve() if auto_run_path else None
    LOAD_MORE_ARGS = load_more_args
    _CACHED_FILTER_SETS = None
    _CACHED_AUTO_RUN = None

    # change to output directory so SimpleHTTPRequestHandler can serve files
    import os

    original_dir = os.getcwd()
    os.chdir(OUTPUT_DIR)

    server_address = ("", port)
    httpd = HTTPServer(server_address, QCGalleryHandler)
    httpd._load_more_in_progress = False
    httpd._load_more_ready = False
    httpd._load_more_error = None
    httpd._load_more_target_batch = None
    httpd._load_more_progress = None
    httpd._load_more_message = None

    import os
    import socket

    hostname = socket.gethostname()
    username = os.environ.get("USER", os.environ.get("USERNAME", "your_username"))

    logger.info("=" * 60)
    logger.info("QC SERVER STARTED")
    logger.info("=" * 60)
    logger.info(f"QC state file: {QC_FILE_PATH}")
    logger.info(f"Serving from: {OUTPUT_DIR}")
    logger.info("")
    logger.info("FOR REMOTE ACCESS:")
    logger.info("  1. In a NEW local terminal, run:")
    logger.info(f"     ssh -L {port}:localhost:{port} {username}@{hostname}")
    logger.info("  2. Open in your browser:")
    logger.info(f"     http://localhost:{port}/")
    logger.info("")
    logger.info("QC workflow:")
    logger.info("  • Mark exams with g (good) or use a to annotate findings")
    logger.info("  • Use dropdown to load filter lists (no restart needed!)")
    logger.info("  • QC data auto-saves to server on each click")
    logger.info("  • Press ctrl+c to stop server when done")
    logger.info("=" * 60)

    if not (OUTPUT_DIR / "gallery.html").exists():
        logger.warning(f"gallery.html not found in {OUTPUT_DIR}")
        logger.warning("Make sure you generated the gallery before starting server")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nShutting down server...")
        httpd.shutdown()
        os.chdir(original_dir)
        logger.info("Server stopped")


def get_tag(
    ds: FileDataset, tag: Tuple[int, int], default: Optional[str] = None
) -> Optional[str]:
    """fetch a DICOM tag as string if present.

    parameters
    ----------
    ds : FileDataset
        dicom dataset
    tag : Tuple[int,int]
        group, element of the tag
    default : Optional[str]
        value to return if tag missing

    returns
    -------
    Optional[str]
    """
    t = Tag(tag)
    if t in ds:
        val = ds[t].value
        return str(val) if val is not None else default
    return default


def _save_debug_figure(
    ds: FileDataset,
    path: Path,
    status: str,
    reason_or_view: str,
    debug_dir: Path,
    patient_id: str = None,
    exam_id: str = None,
    accession_number: str = None,
) -> None:
    """save debug figure showing pixel data and DICOM tags.

    status: 'SUCCESS' or 'FAILED'
    reason_or_view: if failed, the reason; if success, the laterality and view (e.g., 'L CC')
    organizes debug files by:
      - SUCCESS: debug_dir/success/patient_id/accession_number/
      - FAILED: debug_dir/failed/patient_id/fail_reason/exam_id/
    """
    try:
        # create figure with two subplots
        fig = plt.figure(figsize=(12, 8))

        # left: pixel data
        ax_img = plt.subplot(1, 2, 1)
        try:
            pixel_array = ds.pixel_array
            ax_img.imshow(pixel_array, cmap="gray")
            ax_img.set_title(f"Pixel Data\n{ds.Rows}x{ds.Columns}")
            ax_img.axis("off")
        except Exception as e:
            ax_img.text(
                0.5,
                0.5,
                f"Cannot display pixel data:\n{e}",
                ha="center",
                va="center",
                transform=ax_img.transAxes,
            )
            ax_img.axis("off")

        # right: DICOM tags
        ax_text = plt.subplot(1, 2, 2)
        ax_text.axis("off")

        # collect tag info
        tag_lines = [
            f"File: {path.name}",
            f"Exam: {path.parent.name}",
            f"Patient: {path.parent.parent.name}",
            f"Status: {status}",
        ]

        if status == "FAILED":
            tag_lines.append(f"Reason: {reason_or_view}")
        else:
            tag_lines.append(f"View: {reason_or_view}")
            tag_lines.append(
                f"For Presentation: {get_tag(ds, (0x0008, 0x0068), 'N/A')}"
            )

        tag_lines.extend(
            [
                "",
                "TAGS WE'RE LOOKING FOR:",
                f"  (0x0020,0x0062) Laterality: {get_tag(ds, (0x0020, 0x0062), 'MISSING')}",
                f"  (0x0020,0x0060) ImageLaterality: {get_tag(ds, (0x0020, 0x0060), 'MISSING')}",
                f"  (0x0018,0x5101) ViewPosition: {get_tag(ds, (0x0018, 0x5101), 'MISSING')}",
                f"  (0x0008,0x0068) PresentationIntentType: {get_tag(ds, (0x0008, 0x0068), 'MISSING')}",
                f"  (0x0008,0x0069) Presentation Intent Type: {get_tag(ds, (0x0008, 0x0069), 'MISSING')}",
                "",
                "ALL TAGS IN THIS FILE:",
                "-" * 60,
            ]
        )

        for elem in ds:
            if elem.VR == "SQ":
                tag_lines.append(f"{elem.tag} {elem.name}: <sequence>")
            else:
                try:
                    value_str = str(elem.value)
                    if len(value_str) > 80:
                        value_str = value_str[:80] + "..."
                    tag_lines.append(f"{elem.tag} {elem.name}: {value_str}")
                except Exception:
                    tag_lines.append(f"{elem.tag} {elem.name}: <cannot display>")

        # render text
        text_content = "\n".join(tag_lines)
        ax_text.text(
            0,
            1,
            text_content,
            verticalalignment="top",
            fontsize=7,
            family="monospace",
            transform=ax_text.transAxes,
        )

        # save figure organized by patient_id/fail_reason/exam_id
        if patient_id is None:
            patient_id = path.parent.parent.name
        if exam_id is None:
            exam_id = path.parent.name
        if accession_number is None:
            accession_number = get_tag(ds, (0x0008, 0x0050), "unknown_accession")

        if status == "SUCCESS":
            success_dir = debug_dir / "success"
            patient_dir = success_dir / patient_id
            accession_dir = patient_dir / accession_number
            accession_dir.mkdir(parents=True, exist_ok=True)
            exam_dir = accession_dir
        else:
            clean_reason = (
                reason_or_view.lower()
                .replace(" ", "_")
                .replace("/", "_")
                .replace("\\", "_")
                .replace("__", "_")
            )
            failed_dir = debug_dir / "failed"
            patient_dir = failed_dir / patient_id
            reason_dir = patient_dir / clean_reason
            exam_dir = reason_dir / exam_id
            exam_dir.mkdir(parents=True, exist_ok=True)

        view_label = (
            reason_or_view.replace(" ", "_") if status == "SUCCESS" else "failed"
        )
        out_path = exam_dir / f"{view_label}_{path.stem}.png"

        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        logger.debug(f"    Saved debug figure: {out_path.name}")

    except Exception as e:
        logger.warning(f"    Failed to save debug figure for {path.name}: {e}")


def _save_four_view_figure(
    view_data: Dict[str, tuple],
    debug_dir: Path,
    patient_id: str,
    accession_number: str,
    exam_id: str,
) -> Tuple[bool, bool]:
    """save a combined figure with all four views of an exam.

    Args:
        view_data: dict mapping view keys (L_CC, L_MLO, R_CC, R_MLO) to (dicom_dataset, path) tuples
        debug_dir: output directory
        patient_id: patient identifier
        accession_number: accession number
        exam_id: exam identifier

    Returns:
        (success: bool, was_cached: bool)
    """
    # construct output path first to check if it exists
    success_dir = debug_dir / "success"
    patient_dir = success_dir / patient_id
    accession_dir = patient_dir / accession_number
    out_path = accession_dir / f"COMBINED_four_views_{exam_id}.png"

    # if image already exists, skip regeneration
    if out_path.exists():
        logger.debug(f"    Using cached 4-view figure: {out_path.name}")
        return True, True

    try:
        fig = plt.figure(figsize=(20, 5))

        # canonical order
        view_order = [("L", "CC"), ("R", "CC"), ("L", "MLO"), ("R", "MLO")]
        view_labels = ["L CC", "R CC", "L MLO", "R MLO"]

        for idx, ((lat, view), label) in enumerate(zip(view_order, view_labels), 1):
            ax = plt.subplot(1, 4, idx)
            view_key = f"{lat}_{view}"

            if view_key in view_data:
                ds, _ = view_data[view_key]
                try:
                    pixel_array = ds.pixel_array
                    ax.imshow(pixel_array, cmap="gray")
                    ax.set_title(label, fontsize=14, fontweight="bold")
                except Exception as e:
                    ax.text(
                        0.5,
                        0.5,
                        f"Cannot display:\n{e}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"{label} (error)", fontsize=14)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Missing",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=16,
                )
                ax.set_title(f"{label} (missing)", fontsize=14)

            ax.axis("off")

        # save figure (path already computed above for caching check)
        accession_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=75, bbox_inches="tight")
        plt.close(fig)

        logger.debug(f"    Saved combined 4-view figure: {out_path.name}")
        return True, False

    except Exception as e:
        logger.warning(
            f"    Failed to save combined 4-view figure for exam {exam_id}: {e}"
        )
        return False, False


def _compute_exam_error_scores(
    views_df: pd.DataFrame,
    pred_csv: Optional[Path],
    meta_csv: Optional[Path],
    horizon: int = 5,
) -> pd.Series:
    """compute prediction error score for each exam to prioritize QC.

    Args:
        views_df: dataframe with exam_id column
        pred_csv: path to validation_output.csv with predictions
        meta_csv: path to mirai_manifest.csv with labels
        horizon: which prediction horizon to use for scoring (default: 5 years)

    Returns:
        series mapping exam_id to error score (higher = worse prediction)
    """
    if pred_csv is None or meta_csv is None:
        logger.warning("no predictions provided; cannot prioritize by error score")
        return pd.Series(dtype=float)

    if not pred_csv.exists() or not meta_csv.exists():
        logger.warning(f"prediction or metadata file not found: {pred_csv}, {meta_csv}")
        return pd.Series(dtype=float)

    try:
        # load predictions
        pred = pd.read_csv(pred_csv)
        if {"patient_id", "exam_id"}.issubset(pred.columns):
            pid = pred["patient_id"].astype(str)
            eid = pred["exam_id"].astype(str)
        elif "patient_exam_id" in pred.columns:
            parts = (
                pred["patient_exam_id"].astype(str).str.split("\t", n=1, expand=True)
            )
            pid = parts[0]
            eid = parts[1]
        else:
            logger.warning("predictions missing patient_id/exam_id columns")
            return pd.Series(dtype=float)

        pred = pred.assign(patient_id=pid, exam_id=eid)

        # load metadata
        meta = (
            pd.read_csv(meta_csv)
            if meta_csv.suffix.lower() == ".csv"
            else pd.read_parquet(meta_csv)
        )

        req = {"patient_id", "exam_id", "years_to_cancer", "years_to_last_followup"}
        if not req.issubset(set(meta.columns)):
            logger.warning(
                f"metadata missing required columns: {req - set(meta.columns)}"
            )
            return pd.Series(dtype=float)

        meta = meta.assign(
            patient_id=meta["patient_id"].astype(str),
            exam_id=meta["exam_id"].astype(str),
        )

        # merge predictions with metadata
        df = pred.merge(meta, on=["patient_id", "exam_id"], how="inner")

        # detect prediction column for this horizon
        pred_col = None
        for col in df.columns:
            if f"{horizon}" in col and "risk" in col.lower():
                pred_col = col
                break

        if pred_col is None:
            logger.warning(f"no prediction column found for horizon {horizon}")
            return pd.Series(dtype=float)

        # aggregate per exam (mean prediction across views)
        agg_dict = {
            pred_col: "mean",
            "years_to_cancer": "first",
            "years_to_last_followup": "first",
        }
        df_exam = df.groupby(["patient_id", "exam_id"], as_index=False).agg(agg_dict)

        # compute labels for this horizon
        ytc = df_exam["years_to_cancer"].astype(float).to_numpy()
        ylf = df_exam["years_to_last_followup"].astype(float).to_numpy()
        scores = df_exam[pred_col].astype(float).to_numpy()

        case = ytc <= horizon
        ctrl = (ytc > horizon) & (ylf >= horizon)
        include = case | ctrl

        if include.sum() < 10:
            logger.warning(f"insufficient samples with labels at horizon {horizon}")
            return pd.Series(dtype=float)

        y = case[include].astype(int)
        s = scores[include]
        exam_ids = df_exam["exam_id"].values[include]

        # compute error magnitude: for each exam, how far is prediction from true label?
        # higher error = worse prediction = higher priority for QC
        # for cases (y=1): error = 1 - score (missed cases have low scores)
        # for controls (y=0): error = score (false alarms have high scores)
        errors = np.where(y == 1, 1.0 - s, s)

        # additionally weight by distance from optimal threshold
        # find optimal threshold (Youden's J statistic)
        fpr, tpr, thresholds = roc_curve(y, s)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        # increase error for exams on the wrong side of threshold
        misclassified = (s >= optimal_threshold) != (y == 1)
        errors = np.where(misclassified, errors * 2.0, errors)

        error_series = pd.Series(errors, index=exam_ids)

        logger.info(
            f"computed error scores for {len(error_series)} exams at {horizon}y horizon"
        )
        logger.info(
            f"optimal threshold: {optimal_threshold:.3f}, "
            f"misclassified: {misclassified.sum()}/{len(misclassified)}"
        )

        return error_series

    except Exception as e:
        logger.error(f"error computing exam error scores: {e}")
        import traceback

        traceback.print_exc()
        return pd.Series(dtype=float)


# average bytes per combined 4-view PNG (from empirical sampling)
AVG_PNG_BYTES_PER_EXAM = int(183 * 1024)
GALLERY_HTML_OVERHEAD_BYTES = 5 * 1024 * 1024  # ~5 MB for large galleries


def estimate_qc_preprocess_disk(
    views_parquet: Path,
    raw_dir: Path,
    patient_id: Optional[str] = None,
    exam_id: Optional[str] = None,
    exam_list_path: Optional[Path] = None,
    per_view: bool = False,
) -> Tuple[int, int]:
    """Estimate disk usage for QC preprocessing (figures + gallery).

    Returns:
        (n_exams, estimated_bytes)
    """
    views_df = pd.read_parquet(views_parquet)
    views_df = views_df.assign(
        exam_id=views_df["exam_id"].astype(str),
        patient_id=views_df["patient_id"].astype(str),
    )
    if "accession_number" in views_df.columns:
        views_df["accession_number"] = views_df["accession_number"].astype(str)

    from prima.qc_filters import compute_auto_filter_sets, load_auto_filter_names

    auto_filter_names = load_auto_filter_names()
    if auto_filter_names:
        tags_path = raw_dir / "sot" / "dicom_tags.parquet"
        if tags_path.exists():
            filter_sets = compute_auto_filter_sets(
                views_parquet, tags_path, filter_names=auto_filter_names
            )
            auto_excluded = set()
            for fn in auto_filter_names:
                auto_excluded.update(filter_sets.get(fn, set()))
            views_df = views_df[~views_df["exam_id"].isin(auto_excluded)]

    if patient_id:
        views_df = views_df[views_df["patient_id"] == str(patient_id)]
    if exam_id:
        views_df = views_df[views_df["exam_id"] == str(exam_id)]
    if exam_list_path and exam_list_path.exists():
        with open(exam_list_path) as f:
            exam_list = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
        views_df = views_df[views_df["exam_id"].isin(exam_list)]

    n_exams = views_df["exam_id"].nunique()
    bytes_per_exam = AVG_PNG_BYTES_PER_EXAM
    if per_view:
        bytes_per_exam += 4 * AVG_PNG_BYTES_PER_EXAM  # 4 extra per-view figures
    estimated = n_exams * bytes_per_exam + GALLERY_HTML_OVERHEAD_BYTES
    return n_exams, estimated


def generate_gallery(
    views_parquet: Path,
    raw_dir: Path,
    output_dir: Path,
    max_exams: Optional[int] = None,
    random_sample: bool = False,
    patient_id: Optional[str] = None,
    exam_id: Optional[str] = None,
    exam_list_path: Optional[Path] = None,
    per_view: bool = False,
    no_gallery: bool = False,
    qc_file: Optional[Path] = None,
    auto_run_file: Optional[Path] = None,
    pred_csv: Optional[Path] = None,
    meta_csv: Optional[Path] = None,
    prioritize_errors: bool = False,
    horizon: int = 5,
    qc_skip_status: Optional[Set[str]] = None,
    serve: bool = False,
    preprocessed_only: bool = False,
    original_args: Optional[dict] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> None:
    """generate interactive HTML gallery from processed views.

    Args:
        views_parquet: path to views.parquet with dicom_path column
        raw_dir: root directory where DICOMs are stored
        output_dir: output directory for figures and gallery
        max_exams: limit number of exams to visualize
        random_sample: if True, sample randomly; else take first N (or by error if prioritize_errors)
        patient_id: filter to specific patient
        exam_id: filter to specific exam
        exam_list_path: path to text file with exam IDs (one per line) to review
        per_view: if True, also generate individual per-view debug figures
        no_gallery: if True, skip HTML gallery generation
        qc_file: path to unified QC state file; exams with status in qc_skip_status will be skipped
        auto_run_file: path to a saved model suggestion run to review/accept in the gallery
        pred_csv: path to validation_output.csv with predictions (for error prioritization)
        meta_csv: path to mirai_manifest.csv with labels (for error prioritization)
        prioritize_errors: if True, sort by prediction error (worst first)
        horizon: which prediction horizon to use for error scoring (default: 5 years)
        qc_skip_status: set of QC statuses to skip (default: {"good", "annotated", "auto_excluded"})
        serve: if True, suppress non-server instructions in logging
        preprocessed_only: if True, require cached montage PNGs and never fall back to raw DICOM loading
    """
    if qc_skip_status is None:
        qc_skip_status = {"good", "annotated", "auto_excluded"}

    def report_progress(progress: float, detail: str) -> None:
        if progress_callback is None:
            return
        bounded = max(0.0, min(1.0, float(progress)))
        try:
            progress_callback(bounded, detail)
        except Exception as e:
            logger.debug("progress callback failed: %s", e)

    report_progress(0.02, "Loading QC state...")
    logger.info(
        "QC DEBUG generate_gallery args: max_exams=%s, random_sample=%s, prioritize_errors=%s, qc_skip_status=%s, patient_id=%s, exam_id=%s, exam_list_path=%s, qc_file=%s, auto_run_file=%s, serve=%s, preprocessed_only=%s",
        max_exams,
        random_sample,
        prioritize_errors,
        sorted(qc_skip_status),
        patient_id,
        exam_id,
        exam_list_path,
        qc_file,
        auto_run_file,
        serve,
        preprocessed_only,
    )
    # load existing QC state if available
    qc_state = {}
    if qc_file and qc_file.exists():
        logger.info(f"loading QC state from {qc_file}")
        qc_state = _load_qc_state(qc_file, persist_normalized=True)
    qc_data = qc_state_to_status_map(qc_state)
    annotations_data = qc_state_to_annotations_map(qc_state)
    auto_run = {}
    auto_suggestion_data = {}
    if auto_run_file and auto_run_file.exists():
        logger.info(f"loading auto suggestion run from {auto_run_file}")
        auto_run = _load_auto_run(auto_run_file, persist_normalized=True)
        auto_suggestion_data = auto_run_to_tag_map(auto_run)
    if qc_data:
        logger.info(f"loaded QC status for {len(qc_data)} exams")
    if qc_data:
        logger.info(
            "QC DEBUG loaded QC map: %d entries (%s)",
            len(qc_data),
            _format_status_counts(_status_counts(qc_data)),
        )
    else:
        logger.info("QC DEBUG loaded QC map: empty")
    if annotations_data and qc_file:
        logger.info(
            "loaded annotation findings for %d exams from %s",
            len(annotations_data),
            qc_file,
        )
    if auto_suggestion_data and auto_run_file:
        logger.info(
            "loaded auto suggestions for %d exams from %s",
            len(auto_suggestion_data),
            auto_run_file,
        )

    logger.info(f"loading views from {views_parquet}")
    views_df = pd.read_parquet(views_parquet)
    views_df = views_df.assign(
        exam_id=views_df["exam_id"].astype(str),
        patient_id=views_df["patient_id"].astype(str),
    )
    if "accession_number" in views_df.columns:
        views_df["accession_number"] = views_df["accession_number"].astype(str)
    logger.info(
        f"loaded {len(views_df)} views from {views_df['exam_id'].nunique()} exams"
    )
    report_progress(0.16, "Loaded views metadata...")

    # apply auto-exclusions from config
    from prima.qc_filters import compute_auto_filter_sets, load_auto_filter_names

    auto_filter_names = load_auto_filter_names()
    if auto_filter_names:
        logger.info(f"applying auto-exclusions: {', '.join(auto_filter_names)}")
        tags_path = raw_dir / "sot" / "dicom_tags.parquet"
        if not tags_path.exists():
            logger.warning(
                "auto-filter config is enabled but dicom_tags.parquet is missing: %s",
                tags_path,
            )
            filter_sets = {}
        else:
            filter_sets = compute_auto_filter_sets(
                views_parquet, tags_path, filter_names=auto_filter_names
            )
        auto_excluded_exams: set = set()
        for filter_name in auto_filter_names:
            matched = filter_sets.get(filter_name, set())
            if matched:
                auto_excluded_exams.update(matched)
                logger.info(f"  {filter_name}: {len(matched)} exams")

        if auto_excluded_exams:
            before_count = views_df["exam_id"].nunique()
            views_df = views_df[~views_df["exam_id"].isin(auto_excluded_exams)]
            after_count = views_df["exam_id"].nunique()
            logger.info(
                f"auto-excluded {before_count - after_count} exams total via config filters"
            )

            # mark these in QC data as auto-excluded and persist
            new_auto = 0
            for eid in auto_excluded_exams:
                exam_key = str(eid)
                if exam_key not in qc_data:  # don't override manual QC
                    qc_data[exam_key] = "auto_excluded"
                    record = qc_state.get(exam_key, {"status": None, "annotations": []})
                    record["status"] = "auto_excluded"
                    qc_state[exam_key] = record
                    new_auto += 1
            if new_auto > 0 and qc_file:
                save_qc_state(qc_file, qc_state)
                logger.info(
                    f"persisted {new_auto} new auto_excluded entries to {qc_file}"
                )
    report_progress(0.28, "Applied exclusions and loaded annotations...")

    annotation_review_views_df = views_df[
        views_df["exam_id"].isin(annotations_data.keys())
    ].copy()
    auto_review_views_df = views_df[
        views_df["exam_id"].isin(auto_suggestion_data.keys())
    ].copy()

    # filter by patient/exam if specified
    if patient_id:
        views_df = views_df[views_df["patient_id"] == str(patient_id)]
        logger.info(
            f"filtered to patient {patient_id}: {len(views_df)} views from {views_df['exam_id'].nunique()} exams"
        )

    if exam_id:
        views_df = views_df[views_df["exam_id"] == str(exam_id)]
        logger.info(f"filtered to exam {exam_id}: {len(views_df)} views")

    # filter by exam list if provided
    if exam_list_path:
        if exam_list_path.exists():
            logger.info(f"loading exam list from {exam_list_path}")
            with open(exam_list_path) as f:
                # skip comment lines starting with #
                exam_list = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
            logger.info(f"loaded {len(exam_list)} exam IDs from list")
            views_df = views_df[views_df["exam_id"].isin(exam_list)]
            logger.info(
                f"filtered to {views_df['exam_id'].nunique()} exams from list: {len(views_df)} views"
            )
        else:
            logger.error(f"exam list file not found: {exam_list_path}")
            return

    # track total pool for QC metrics (before skipping completed exams)
    total_to_qc = views_df["exam_id"].nunique()
    eligible_exam_ids = set(views_df["exam_id"].unique())
    done_statuses = {"good"}
    done_exams = {eid for eid, status in qc_data.items() if status in done_statuses}
    annotated_exams = set(annotations_data.keys())
    done_exams |= annotated_exams
    done_count = len(done_exams & eligible_exam_ids)
    remaining_to_qc = max(total_to_qc - done_count, 0)
    logger.info(
        "QC DEBUG pool before skip: exams=%d, done_in_pool=%d, remaining_to_qc=%d",
        total_to_qc,
        done_count,
        remaining_to_qc,
    )

    # filter out exams with specified QC statuses
    if qc_skip_status:
        status_skip_statuses = {
            status for status in qc_skip_status if status != "annotated"
        }
        skip_exams = set()
        if qc_data and status_skip_statuses:
            skip_exams |= {
                eid for eid, status in qc_data.items() if status in status_skip_statuses
            }
        if "annotated" in qc_skip_status:
            skip_exams |= annotated_exams
        skip_in_pool = set(views_df["exam_id"].unique()) & skip_exams
        logger.info(
            "QC DEBUG skip filter: skip_status=%s, status_skip_entries=%d, annotated_skip_entries=%d, skip_in_current_pool=%d",
            sorted(qc_skip_status),
            len(
                {
                    eid
                    for eid, status in qc_data.items()
                    if status in status_skip_statuses
                }
            ),
            len(annotated_exams) if "annotated" in qc_skip_status else 0,
            len(skip_in_pool),
        )
        if skip_in_pool:
            logger.info(
                "QC DEBUG skip filter sample exam_ids: %s",
                sorted(skip_in_pool)[:10],
            )
        if skip_exams:
            before_count = views_df["exam_id"].nunique()
            views_df = views_df[~views_df["exam_id"].isin(skip_exams)]
            after_count = views_df["exam_id"].nunique()
            status_str = ", ".join(f"'{s}'" for s in sorted(qc_skip_status))
            logger.info(
                f"filtered out {before_count - after_count} exams based on skip buckets: {status_str}"
            )

    if len(views_df) == 0:
        logger.error("no views to visualize after filtering")
        return

    # compute error scores if requested
    error_scores = pd.Series(dtype=float)
    if prioritize_errors:
        logger.info(
            "computing prediction error scores to prioritize worst-performing exams..."
        )
        error_scores = _compute_exam_error_scores(
            views_df, pred_csv, meta_csv, horizon=horizon
        )
        if len(error_scores) > 0:
            # filter views to only exams with scores
            views_df = views_df[views_df["exam_id"].isin(error_scores.index)]
            logger.info(
                f"filtered to {views_df['exam_id'].nunique()} exams with error scores"
            )
            # also filter error_scores to only exams in views_df
            available_exam_ids = set(views_df["exam_id"].unique())
            error_scores = error_scores[error_scores.index.isin(available_exam_ids)]
            logger.info(
                f"error scores available for {len(error_scores)} exams in views dataset"
            )
    report_progress(0.4, "Selected exam pool...")

    # sample exams if requested
    selected_exam_ids = [str(eid) for eid in views_df["exam_id"].unique().tolist()]
    if max_exams:
        unique_exams = views_df["exam_id"].unique()
        selected_exams = unique_exams
        if len(unique_exams) > max_exams:
            if prioritize_errors and len(error_scores) > 0:
                # sort by error score (highest first), only from exams available in views
                sorted_exams = error_scores.sort_values(ascending=False).index.tolist()
                selected_exams = sorted_exams[:max_exams]
                logger.info(
                    f"selected top {max_exams} worst-performing exams from {len(unique_exams)} "
                    f"(mean error: {error_scores[selected_exams].mean():.3f})"
                )
            elif random_sample:
                selected_exams = (
                    pd.Series(unique_exams).sample(n=max_exams, random_state=42).values
                )
                logger.info(
                    f"randomly sampled {max_exams} exams from {len(unique_exams)}"
                )
            else:
                selected_exams = unique_exams[:max_exams]
                logger.info(f"taking first {max_exams} exams from {len(unique_exams)}")
            views_df = views_df[views_df["exam_id"].isin(selected_exams)]
        selected_exam_ids = [str(eid) for eid in selected_exams]
        logger.info(
            f"will visualize {len(views_df)} views from {views_df['exam_id'].nunique()} exams"
        )
    selected_counts = {status: 0 for status in DEBUG_TRACKED_STATUSES}
    selected_annotated = 0
    pending_count = 0
    for exam_key in selected_exam_ids:
        status = qc_data.get(str(exam_key), "")
        if status in selected_counts:
            selected_counts[status] += 1
        elif str(exam_key) in annotated_exams:
            selected_annotated += 1
        else:
            pending_count += 1
    logger.info(
        "QC DEBUG selected batch: exams=%d, statuses=(good=%d, annotated=%d, auto_excluded=%d, pending=%d)",
        len(selected_exam_ids),
        selected_counts["good"],
        selected_annotated,
        selected_counts["auto_excluded"],
        pending_count,
    )
    logger.info(
        "QC DEBUG selected batch exam_ids (first %d): %s",
        min(10, len(selected_exam_ids)),
        selected_exam_ids[:10],
    )
    report_progress(0.5, "Selected exams and checking cached montages...")

    # create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"output will be saved to: {output_dir}")

    def build_cached_exam_records(source_views_df: pd.DataFrame):
        """Build lightweight exam metadata for cached combined montages."""
        records = []
        success_dir = output_dir / "success"
        for source_exam_id in source_views_df["exam_id"].unique():
            exam_views = source_views_df[source_views_df["exam_id"] == source_exam_id]
            if len(exam_views) == 0:
                continue

            row = exam_views.iloc[0]
            source_patient_id = row["patient_id"]
            accession = row.get("accession_number", "unknown")
            img_path = (
                success_dir
                / source_patient_id
                / accession
                / f"COMBINED_four_views_{source_exam_id}.png"
            )
            if not img_path.exists():
                continue

            records.append(
                {
                    "path": img_path.relative_to(output_dir),
                    "patient_id": source_patient_id,
                    "exam_id": source_exam_id,
                    "accession": accession,
                    "num_views": len(exam_views),
                    "qc_status": get_qc_status(source_exam_id),
                }
            )
        return records

    # first, identify which exams already have cached figures
    logger.info("checking for cached figures...")
    cached_exams = set()
    exams_needing_generation = set()

    for exam_id in views_df["exam_id"].unique():
        exam_views = views_df[views_df["exam_id"] == exam_id]
        if len(exam_views) == 0:
            continue

        row = exam_views.iloc[0]
        patient_id = row["patient_id"]
        accession = row.get("accession_number", "unknown")

        # check if cached figure exists
        success_dir = output_dir / "success"
        cached_path = (
            success_dir / patient_id / accession / f"COMBINED_four_views_{exam_id}.png"
        )

        if cached_path.exists():
            cached_exams.add(exam_id)
        else:
            exams_needing_generation.add(exam_id)

    logger.info(
        f"found {len(cached_exams)} cached figures, need to generate {len(exams_needing_generation)}"
    )
    if len(exams_needing_generation) == 0:
        report_progress(0.78, "All montages already cached...")

    def get_qc_status(exam_id):
        """extract status string from qc_data"""
        return qc_data.get(str(exam_id), "")

    # process in batches to avoid OOM (DICOM pixel arrays are large)
    BATCH_SIZE = 100
    success_count = 0
    error_count = 0
    combined_images = []
    cached_count = len(cached_exams)
    generated_count = 0

    # add cached exams to gallery list
    for exam_id in cached_exams:
        exam_views = views_df[views_df["exam_id"] == exam_id]
        if len(exam_views) == 0:
            continue

        row = exam_views.iloc[0]
        patient_id = row["patient_id"]
        accession = row.get("accession_number", "unknown")

        success_dir = output_dir / "success"
        img_path = (
            success_dir / patient_id / accession / f"COMBINED_four_views_{exam_id}.png"
        )

        combined_images.append(
            {
                "path": img_path.relative_to(output_dir),
                "patient_id": patient_id,
                "exam_id": exam_id,
                "accession": accession,
                "num_views": len(exam_views),
                "qc_status": get_qc_status(exam_id),
            }
        )

    exams_needing_list = sorted(exams_needing_generation)
    if len(exams_needing_list) > 0:
        if preprocessed_only:
            sample_ids = ", ".join(exams_needing_list[:10])
            raise FileNotFoundError(
                "preprocessed-only mode cannot generate missing montages; "
                f"{len(exams_needing_list)} exams are missing cached PNGs in "
                f"{output_dir / 'success'} (sample exam_ids: {sample_ids})"
            )
        views_to_load = views_df[views_df["exam_id"].isin(exams_needing_generation)]
        n_batches = (len(exams_needing_list) + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(
            f"processing {len(exams_needing_list)} exams in {n_batches} batches of up to {BATCH_SIZE}"
        )
        report_progress(
            0.56, f"Generating {len(exams_needing_list)} missing montages..."
        )

        for batch_idx, batch_start in enumerate(
            tqdm(
                range(0, len(exams_needing_list), BATCH_SIZE),
                total=n_batches,
                desc="batches",
            )
        ):
            batch_exam_ids = set(
                exams_needing_list[batch_start : batch_start + BATCH_SIZE]
            )
            batch_views = views_to_load[views_to_load["exam_id"].isin(batch_exam_ids)]
            batch_cache = {}

            for idx, row in batch_views.iterrows():
                try:
                    dicom_path = Path(row["dicom_path"])
                    if not dicom_path.is_absolute():
                        dicom_path = raw_dir / dicom_path

                    if not dicom_path.exists():
                        logger.warning(f"DICOM not found: {dicom_path}")
                        error_count += 1
                        continue

                    ds = pydicom.dcmread(str(dicom_path))

                    if per_view:
                        view_label = f"{row['laterality']} {row['view']}"
                        _save_debug_figure(
                            ds,
                            dicom_path,
                            "SUCCESS",
                            view_label,
                            output_dir,
                            patient_id=row["patient_id"],
                            exam_id=row["exam_id"],
                            accession_number=row.get("accession_number", "unknown"),
                        )
                    success_count += 1

                    exam_key = row["exam_id"]
                    if exam_key not in batch_cache:
                        batch_cache[exam_key] = {
                            "patient_id": row["patient_id"],
                            "accession_number": row.get("accession_number", "unknown"),
                            "views": {},
                            "view_selection_keys": {},
                        }

                    view_key = f"{row['laterality']}_{row['view']}"
                    candidate_key = view_selection_key_from_dataset(ds, dicom_path)
                    current_key = batch_cache[exam_key]["view_selection_keys"].get(
                        view_key
                    )
                    if current_key is None or candidate_key < current_key:
                        batch_cache[exam_key]["view_selection_keys"][view_key] = (
                            candidate_key
                        )
                        batch_cache[exam_key]["views"][view_key] = (ds, dicom_path)

                except Exception as e:
                    logger.error(
                        f"error processing {row.get('dicom_path', 'unknown')}: {e}"
                    )
                    error_count += 1

            for exam_key, exam_data in batch_cache.items():
                try:
                    success, was_cached = _save_four_view_figure(
                        view_data=exam_data["views"],
                        debug_dir=output_dir,
                        patient_id=exam_data["patient_id"],
                        accession_number=exam_data["accession_number"],
                        exam_id=exam_key,
                    )

                    if success:
                        if was_cached:
                            cached_count += 1
                        else:
                            generated_count += 1

                        success_dir = output_dir / "success"
                        patient_dir = success_dir / exam_data["patient_id"]
                        accession_dir = patient_dir / exam_data["accession_number"]
                        img_path = accession_dir / f"COMBINED_four_views_{exam_key}.png"

                        combined_images.append(
                            {
                                "path": img_path.relative_to(output_dir),
                                "patient_id": exam_data["patient_id"],
                                "exam_id": exam_key,
                                "accession": exam_data["accession_number"],
                                "num_views": len(exam_data["views"]),
                                "qc_status": get_qc_status(exam_key),
                            }
                        )

                except Exception as e:
                    logger.error(
                        f"error creating combined figure for exam {exam_key}: {e}"
                    )

            del batch_cache
            gc.collect()
            report_progress(
                0.56 + 0.28 * ((batch_idx + 1) / max(n_batches, 1)),
                f"Generated montage batch {batch_idx + 1}/{n_batches}...",
            )
    else:
        logger.info("all figures cached, skipping DICOM loading")

    # compute total figures
    total_figures = cached_count + generated_count
    annotated_review_images = build_cached_exam_records(annotation_review_views_df)
    if annotated_review_images:
        logger.info(
            "annotation review pool ready: %d cached annotated exams",
            len(annotated_review_images),
        )
    auto_suggestion_review_images = build_cached_exam_records(auto_review_views_df)
    if auto_suggestion_review_images:
        logger.info(
            "auto suggestion review pool ready: %d cached suggested exams",
            len(auto_suggestion_review_images),
        )
    report_progress(0.9, "Building gallery HTML...")

    # generate HTML gallery
    if not no_gallery and combined_images:
        logger.info("generating HTML gallery...")
        gallery_path = output_dir / "gallery.html"

        # prepare QC state file path for saving
        qc_file_str = str(qc_file.resolve()) if qc_file else "data/qc_state.json"
        qc_storage_namespace = qc_file_str

        # construct load more command
        load_more_cmd = "python qc/qc_gallery.py --serve"

        if exam_list_path:
            # if loaded from exam list, suggest switching to prioritize-errors
            if pred_csv and meta_csv:
                next_batch = max_exams if max_exams else 100
                load_more_cmd += f" --max-exams {next_batch} --prioritize-errors"
                load_more_cmd += f" --pred-csv {pred_csv} --meta-csv {meta_csv}"
            else:
                load_more_cmd += f" --max-exams {max_exams if max_exams else 100}"
        else:
            # keep a fixed batch size across load-more cycles
            if max_exams:
                load_more_cmd += f" --max-exams {max_exams}"
            if prioritize_errors and pred_csv and meta_csv:
                load_more_cmd += (
                    f" --prioritize-errors --pred-csv {pred_csv} --meta-csv {meta_csv}"
                )

        if horizon != 5 and prioritize_errors:
            load_more_cmd += f" --horizon {horizon}"

        qc_skip_status_list = sorted(qc_skip_status) if qc_skip_status else []

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Mammogram QC - {total_figures} exams</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #1e1e1e;
            color: #d4d4d4;
            overflow: hidden;
        }}
        .container {{
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}
        .controls {{
            background-color: #252526;
            border-bottom: 1px solid #3e3e42;
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .stats {{
            color: #9cdcfe;
            font-size: 14px;
        }}
        .filter-controls {{
            flex-grow: 1;
            text-align: center;
            display: flex;
            gap: 10px;
            justify-content: center;
            align-items: center;
        }}
        .filter-controls input {{
            padding: 6px 12px;
            font-size: 13px;
            border: 1px solid #3e3e42;
            border-radius: 4px;
            background-color: #1e1e1e;
            color: #d4d4d4;
            width: 170px;
        }}
        .filter-controls input:focus {{
            outline: none;
            border-color: #4ec9b0;
        }}
        .filter-controls select {{
            padding: 6px 12px;
            font-size: 14px;
            border: 1px solid #3e3e42;
            border-radius: 4px;
            background-color: #1e1e1e;
            color: #d4d4d4;
            cursor: pointer;
        }}
        .filter-controls select:focus {{
            outline: none;
            border-color: #4ec9b0;
        }}
        .filter-controls button {{
            padding: 6px 12px;
            font-size: 13px;
            border: 1px solid #3e3e42;
            border-radius: 4px;
            background-color: #252526;
            color: #d4d4d4;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .filter-controls button:hover {{
            background-color: #4ec9b0;
            border-color: #4ec9b0;
            color: #1e1e1e;
        }}
        .nav-buttons {{
            display: flex;
            gap: 10px;
        }}
        .nav-buttons button {{
            padding: 8px 20px;
            font-size: 14px;
            border: 1px solid #3e3e42;
            border-radius: 4px;
            background-color: #252526;
            color: #d4d4d4;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .nav-buttons button:hover:not(:disabled) {{
            background-color: #4ec9b0;
            border-color: #4ec9b0;
            color: #1e1e1e;
        }}
        .nav-buttons button:disabled {{
            opacity: 0.3;
            cursor: not-allowed;
        }}
        .viewer {{
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
            overflow: auto;
        }}
        .exam-info {{
            margin-bottom: 15px;
            font-size: 14px;
            color: #9cdcfe;
            text-align: center;
        }}
        .exam-info strong {{
            color: #4ec9b0;
        }}
        .qc-controls {{
            margin-bottom: 15px;
            display: flex;
            gap: 10px;
            justify-content: center;
            align-items: center;
        }}
        .qc-button {{
            padding: 6px 16px;
            font-size: 13px;
            font-weight: bold;
            border: 2px solid #3e3e42;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .qc-button.good {{
            background-color: #1e3a1e;
            color: #6bcc6b;
            border-color: #4a8a4a;
        }}
        .qc-button.good:hover {{
            background-color: #2d5a2d;
            border-color: #6bcc6b;
        }}
        .qc-button.good.active {{
            background-color: #4a8a4a;
            border-color: #8fdf8f;
            box-shadow: 0 0 10px #6bcc6b;
        }}
        .qc-button.annotate {{
            background-color: #1e2e3a;
            color: #9cdcfe;
            border-color: #4a6a8a;
        }}
        .qc-button.annotate:hover {{
            background-color: #2d4a5a;
            border-color: #9cdcfe;
        }}
        .qc-button.annotate.has-tags {{
            background-color: #2a4a5a;
            border-color: #4ec9b0;
            color: #4ec9b0;
            box-shadow: 0 0 8px rgba(78, 201, 176, 0.3);
        }}
        .qc-button.skip {{
            background-color: #252526;
            color: #858585;
            border-color: #3e3e42;
        }}
        .qc-button.skip:hover {{
            background-color: #3e3e42;
            border-color: #858585;
        }}
        .qc-status-indicator {{
            padding: 4px 12px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 13px;
        }}
        .image-container {{
            max-width: 100%;
            max-height: calc(100vh - 180px);
            display: flex;
            justify-content: center;
        }}
        .image-container img {{
            max-width: 100%;
            max-height: 100%;
            border: 1px solid #3e3e42;
            border-radius: 4px;
            object-fit: contain;
        }}
        .info-button {{
            padding: 4px 8px;
            font-size: 11px;
            border: 1px solid #3e3e42;
            border-radius: 4px;
            background-color: #252526;
            color: #9cdcfe;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            flex-direction: column;
            align-items: center;
            line-height: 1.3;
        }}
        .info-button:hover {{
            background-color: #3e3e42;
        }}
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.8);
        }}
        .modal-content {{
            background-color: #252526;
            margin: 5% auto;
            padding: 20px;
            border: 1px solid #3e3e42;
            border-radius: 8px;
            width: 80%;
            max-width: 900px;
            max-height: 80vh;
            overflow-y: auto;
            color: #d4d4d4;
        }}
        .modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #3e3e42;
        }}
        .close {{
            color: #d4d4d4;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }}
        .close:hover {{
            color: #4ec9b0;
        }}
        .cutflow-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        .cutflow-table th {{
            background-color: #1e1e1e;
            padding: 8px;
            text-align: left;
            border: 1px solid #3e3e42;
            color: #4ec9b0;
        }}
        .cutflow-table td {{
            padding: 8px;
            border: 1px solid #3e3e42;
        }}
        .cutflow-table tr:hover {{
            background-color: #2d2d30;
        }}
        .completion-banner {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background-color: #252526;
            border: 2px solid #4ec9b0;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            z-index: 2000;
            text-align: center;
            min-width: 400px;
        }}
        .completion-banner h2 {{
            color: #4ec9b0;
            margin-top: 0;
        }}
        .completion-banner .stats-summary {{
            margin: 20px 0;
            padding: 15px;
            background-color: #1e1e1e;
            border-radius: 4px;
        }}
        .completion-banner .stats-summary div {{
            margin: 5px 0;
        }}
        .completion-banner button {{
            padding: 10px 20px;
            margin: 5px;
            font-size: 14px;
            border: 1px solid #3e3e42;
            border-radius: 4px;
            background-color: #4ec9b0;
            color: #1e1e1e;
            cursor: pointer;
            font-weight: bold;
        }}
        .completion-banner button:hover {{
            background-color: #6bdfcf;
        }}
        .progress-shell {{
            width: 100%;
            height: 10px;
            margin-top: 12px;
            border-radius: 999px;
            overflow: hidden;
            background: #1e1e1e;
            border: 1px solid #3e3e42;
        }}
        .progress-bar {{
            height: 100%;
            width: 38%;
            border-radius: 999px;
            background: linear-gradient(90deg, #2d4a44 0%, #4ec9b0 45%, #9cdcfe 100%);
            box-shadow: 0 0 12px rgba(78, 201, 176, 0.35);
        }}
        .progress-bar.indeterminate {{
            animation: qc-progress-slide 1.3s ease-in-out infinite;
        }}
        @keyframes qc-progress-slide {{
            0% {{ transform: translateX(-110%); }}
            100% {{ transform: translateX(290%); }}
        }}
    </style>
</head>
<body>
        <div class="container">
            <div class="controls">
            <div style="display: flex; gap: 8px; align-items: center;">
                <div class="stats" id="stats">
                    {total_figures} exams
                </div>
                <span class="stats">|</span>
                <div class="stats" id="manualQcIndicator">
                    qc'd so far: 0
                </div>
            </div>
            <div style="display: flex; gap: 5px;">
                <button class="info-button" onclick="showCutflow()"><span>📊</span><span>cutflow</span></button>
                <button id="autoReviewBtn" class="info-button" onclick="showAutoReview()" title="Compare loaded auto suggestions against accepted GT in this gallery pool" style="display: {"flex" if auto_suggestion_data else "none"};"><span>🤖</span><span>auto qc</span></button>
                <button class="info-button" onclick="resetAllQC()" title="Clear all saved QC statuses/annotations for this QC state file"><span>🗑</span><span>reset</span></button>
                <button class="info-button" onclick="refreshCurrentBatch()" title="Rebuild gallery from saved QC state (safe: does not delete QC/source data)"><span>🔁</span><span>refresh</span></button>
            </div>
            <div class="filter-controls">
                <input type="text" id="searchBox" placeholder="filter: text or @filter with &, |, ~" 
                       onkeyup="filterGallery()">
                <select id="filterSelect" onfocus="loadAvailableFilters()" onchange="useSelectedFilterToken()">
                    <option value="">load filter list...</option>
                </select>
                <button onclick="insertSelectedFilterToken()">insert filter</button>
                <button onclick="clearFilterExpression()">clear filter</button>
                <button id="acceptVisibleAutoBtn" onclick="acceptVisibleAutoSuggestions()" title="accept all model suggestions for currently visible exams" style="display: {"inline-block" if auto_suggestion_data else "none"};">accept visible auto</button>
                <button onclick="loadFilterBatch()" title="load a batch of exams matching the selected filter">load matching</button>
            </div>
            <div class="nav-buttons">
                <button id="prevBtn" onclick="navigate(-1)">← prev</button>
                <button id="nextBtn" onclick="navigate(1)">next →</button>
            </div>
        </div>
        <div class="viewer" id="viewer">
        </div>
    </div>
    
    <!-- Cutflow Modal -->
    <div id="cutflowModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2>QC Cutflow Analysis</h2>
                <span class="close" onclick="closeCutflow()">&times;</span>
            </div>
            <div id="cutflowContent">
                Loading...
            </div>
        </div>
    </div>

    <!-- Auto QC Review Modal -->
    <div id="autoReviewModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2>auto qc review</h2>
                <span class="close" onclick="closeAutoReview()">&times;</span>
            </div>
            <div id="autoReviewContent">
                Loading...
            </div>
        </div>
    </div>
    
    <!-- Annotation Modal -->
    <div id="annotationModal" class="modal">
        <div class="modal-content" style="max-width: 600px;">
            <div class="modal-header">
                <h2>annotate exam</h2>
                <span class="close" onclick="closeAnnotationModal()">&times;</span>
            </div>
            <div>
                <p style="color: #858585; font-size: 13px; margin-top: 0; margin-bottom: 15px;">
                    Press letter hotkeys to toggle tags, or type new tag and press enter. Press esc to save/close. Use next [s] when you're ready to advance.
                </p>
                <div id="annotationTagsList" style="margin-bottom: 20px;">
                </div>
                <div style="border-top: 1px solid #3e3e42; padding-top: 20px;">
                    <div style="display: flex; gap: 10px; align-items: center;">
                        <input type="text" id="newAnnotationInput" placeholder="add new tag and press enter..." 
                               style="flex: 1; padding: 8px 12px; font-size: 14px; border: 1px solid #3e3e42; 
                                      border-radius: 4px; background-color: #1e1e1e; color: #d4d4d4;">
                        <button onclick="addNewAnnotationTag()" 
                                style="padding: 8px 20px; font-size: 14px; border: 1px solid #3e3e42; 
                                       border-radius: 4px; background-color: #4ec9b0; color: #1e1e1e; 
                                       cursor: pointer; font-weight: bold;">
                            add
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Completion Banner -->
    <div id="completionBanner" style="display: none;" class="completion-banner">
        <h2>🎉 Batch Complete!</h2>
        <div class="stats-summary" id="completionStats">
        </div>
        <p style="color: #9cdcfe; margin: 15px 0;">
            You've reviewed all exams in this batch.
        </p>
        <div style="margin: 15px 0; display: flex; align-items: center; justify-content: center; gap: 10px;">
            <label for="nextBatchSize" style="color: #d4d4d4; font-size: 14px;">next batch:</label>
            <input type="number" id="nextBatchSize" value="{max_exams or 100}" min="1" style="width: 70px; padding: 8px; font-size: 14px; border: 1px solid #3e3e42; border-radius: 4px; background-color: #1e1e1e; color: #d4d4d4; text-align: center;">
            <button id="loadMoreBtn" onclick="loadMore()">🔄 Load More [l]</button>
            <button onclick="closeCompletion()">Continue Reviewing</button>
        </div>
        <div id="loadMoreStatus" style="display: none; margin-top: 20px; padding: 15px; background-color: #1e1e1e; border-radius: 4px;">
            <p style="color: #4ec9b0; font-weight: bold; margin: 0;">⏳ Generating more exams...</p>
            <p style="font-size: 12px; color: #858585; margin: 10px 0 0 0;">This may take a minute. Page will reload automatically.</p>
        </div>
    </div>
    
    <script>
        const allExams = [
"""

        for i, img_info in enumerate(combined_images):
            # convert path to posix format for web
            path_str = str(img_info["path"]).replace("\\", "/")
            patient_id_str = img_info["patient_id"]
            exam_id_str = img_info["exam_id"]
            accession_str = img_info["accession"]
            num_views_int = img_info["num_views"]
            qc_status_str = img_info["qc_status"]
            html_content += f"""            {{
                path: "{path_str}",
                patient_id: "{patient_id_str}",
                exam_id: "{exam_id_str}",
                accession: "{accession_str}",
                num_views: {num_views_int},
                qc_status: "{qc_status_str}"
            }}{"," if i < len(combined_images) - 1 else ""}
"""

        html_content += """
        ];

        let globalAnnotatedExams = [
"""

        for i, img_info in enumerate(annotated_review_images):
            path_str = str(img_info["path"]).replace("\\", "/")
            patient_id_str = img_info["patient_id"]
            exam_id_str = img_info["exam_id"]
            accession_str = img_info["accession"]
            num_views_int = img_info["num_views"]
            qc_status_str = img_info["qc_status"]
            html_content += f"""            {{
                path: "{path_str}",
                patient_id: "{patient_id_str}",
                exam_id: "{exam_id_str}",
                accession: "{accession_str}",
                num_views: {num_views_int},
                qc_status: "{qc_status_str}"
            }}{"," if i < len(annotated_review_images) - 1 else ""}
"""

        html_content += """
        ];

        let globalAutoSuggestedExams = [
"""

        for i, img_info in enumerate(auto_suggestion_review_images):
            path_str = str(img_info["path"]).replace("\\", "/")
            patient_id_str = img_info["patient_id"]
            exam_id_str = img_info["exam_id"]
            accession_str = img_info["accession"]
            num_views_int = img_info["num_views"]
            qc_status_str = img_info["qc_status"]
            html_content += f"""            {{
                path: "{path_str}",
                patient_id: "{patient_id_str}",
                exam_id: "{exam_id_str}",
                accession: "{accession_str}",
                num_views: {num_views_int},
                qc_status: "{qc_status_str}"
            }}{"," if i < len(auto_suggestion_review_images) - 1 else ""}
"""

        html_content += f"""
        ];

        const qcSkipStatus = {json.dumps(qc_skip_status_list)};
        const qcStorageNamespace = {json.dumps(qc_storage_namespace)};
        const qcStorageKey = 'qc_data::' + qcStorageNamespace;
        const annotationsStorageKey = 'annotations::' + qcStorageNamespace;
        const annotationTagsStorageKey = 'annotation_tags::' + qcStorageNamespace;
        const autoSuggestionRunPath = {json.dumps(str(auto_run_file.resolve()) if auto_run_file else "")};
        const enablePreload = {str(ENABLE_PRELOAD).lower()};
        const serverMode = {str(serve).lower()};
        const totalToQC = {total_to_qc};
        const remainingToQC = {remaining_to_qc};
        
        let filteredExams = [...allExams];
        let currentIndex = 0;
        let preloadTriggered = false; // track if we've started preloading next batch
        let availableFilters = [];
        let filtersLoaded = false;
        const filterTokenToFilename = {{}};
        const filterKindsByFilename = {{}};
        const filterSetsByFilename = {{}};
        const pendingFilterLoads = new Set();
        const unresolvedFilterWarnings = new Set();
        const sessionAnnotatedFilterFilename = 'session_annotated';
        
        // session tracking survives refreshes/window reopen for the same QC state file.
        const sessionStatsKey = 'qc_session_stats::' + qcStorageNamespace;
        const sessionStatsVersion = 2;
        const sessionResetAfterMs = 12 * 60 * 60 * 1000;
        
        function loadSessionStats() {{
            const fresh = {{
                version: sessionStatsVersion,
                startTimeMs: Date.now(),
                lastActivityMs: Date.now(),
                qcCount: 0,
                startingRemainingToQC: remainingToQC,
                totalToQC: totalToQC,
                annotatedExamIds: []
            }};
            const raw = localStorage.getItem(sessionStatsKey);
            if (!raw) {{
                return fresh;
            }}
            try {{
                const parsed = JSON.parse(raw);
                const validShape = (
                    parsed &&
                    typeof parsed === 'object' &&
                    Number.isFinite(parsed.startTimeMs) &&
                    Number.isFinite(parsed.qcCount) &&
                    Number.isFinite(parsed.startingRemainingToQC) &&
                    Number.isFinite(parsed.totalToQC)
                );
                if (!validShape) {{
                    return fresh;
                }}
                // reset session stats if we switched to a different QC pool
                if (parsed.totalToQC !== totalToQC) {{
                    return fresh;
                }}
                // reset if QC appears to have been globally reset mid-session
                if (remainingToQC > parsed.startingRemainingToQC) {{
                    return fresh;
                }}
                if (
                    Number.isFinite(parsed.lastActivityMs) &&
                    Date.now() - parsed.lastActivityMs > sessionResetAfterMs
                ) {{
                    return fresh;
                }}
                return {{
                    version: sessionStatsVersion,
                    startTimeMs: parsed.startTimeMs,
                    lastActivityMs: Number.isFinite(parsed.lastActivityMs)
                        ? parsed.lastActivityMs
                        : parsed.startTimeMs,
                    qcCount: parsed.qcCount,
                    startingRemainingToQC: parsed.startingRemainingToQC,
                    totalToQC: parsed.totalToQC,
                    annotatedExamIds: Array.isArray(parsed.annotatedExamIds)
                        ? [...new Set(parsed.annotatedExamIds.map(examId => String(examId)))]
                        : []
                }};
            }} catch (e) {{
                console.error('Failed to parse session stats:', e);
                return fresh;
            }}
        }}
        
        let sessionStats = loadSessionStats();
        let sessionStartTime = sessionStats.startTimeMs;
        let sessionQCCount = sessionStats.qcCount;
        let sessionStartRemainingToQC = sessionStats.startingRemainingToQC;
        
        // batch tracking: rate since this page load (not diluted by breaks)
        const batchStartTime = Date.now();
        let batchQCCount = 0;
        
        function saveSessionStats() {{
            sessionStats.lastActivityMs = Date.now();
            localStorage.setItem(sessionStatsKey, JSON.stringify(sessionStats));
        }}
        // ensure session state is materialized for this tab
        saveSessionStats();
        
        // track QC decisions (exam_id -> status)
        // server state is authoritative; localStorage only backfills missing entries
        let qcData = {{}};
        // annotation system: tags are independent of QC status
        let annotationTags = [];  // available tag strings
        let annotations = {{}};    // exam_id -> [tag1, tag2, ...]
        let autoSuggestionRun = {{}};
        let autoSuggestions = {{}};  // exam_id -> [{{tag, score?, rationale?}}, ...]
        let retainedAnnotatedExamIds = new Set();  // keep newly annotated exams visible in the current batch
        let annotationHotkeys = {{ keyToTag: {{}}, tagToKey: {{}} }};
        const validStatuses = new Set(['good', 'auto_excluded']);
        const doneStatuses = new Set(['good']);
        const skipAnnotatedExams = qcSkipStatus.includes('annotated');

        function getStatus(examId) {{
            return qcData[examId] || '';
        }}

        function getExamAnnotations(examId) {{
            const tags = annotations[examId];
            return Array.isArray(tags) ? tags : [];
        }}

        function getAutoSuggestions(examId) {{
            const entries = autoSuggestions[String(examId)];
            return Array.isArray(entries) ? entries : [];
        }}

        function hasAutoSuggestions(examId) {{
            return getAutoSuggestions(examId).length > 0;
        }}

        function hasAnnotationFindings(examId) {{
            return getExamAnnotations(examId).length > 0;
        }}

        function isDoneExam(examId) {{
            return doneStatuses.has(getStatus(examId)) || hasAnnotationFindings(examId);
        }}

        function shouldSkipExam(examId, options = {{}}) {{
            const examIdStr = String(examId);
            const includeAnnotated = options.includeAnnotated === true;
            if (qcSkipStatus.includes(getStatus(examIdStr))) {{
                return true;
            }}
            if (
                !includeAnnotated &&
                skipAnnotatedExams &&
                hasAnnotationFindings(examIdStr) &&
                !retainedAnnotatedExamIds.has(examIdStr)
            ) {{
                return true;
            }}
            return false;
        }}

        function updateDoneCounters(wasDone, willBeDone) {{
            if (!wasDone && willBeDone) {{
                sessionQCCount++;
                batchQCCount++;
                sessionStats.qcCount = sessionQCCount;
                saveSessionStats();
            }} else if (wasDone && !willBeDone && sessionQCCount > 0) {{
                sessionQCCount--;
                if (batchQCCount > 0) batchQCCount--;
                sessionStats.qcCount = sessionQCCount;
                saveSessionStats();
            }}
        }}

        function getManualQCCount() {{
            const doneExamIds = new Set();
            Object.keys(qcData).forEach(examId => {{
                if (doneStatuses.has(qcData[examId])) {{
                    doneExamIds.add(examId);
                }}
            }});
            Object.keys(annotations).forEach(examId => {{
                if (hasAnnotationFindings(examId)) {{
                    doneExamIds.add(examId);
                }}
            }});
            return doneExamIds.size;
        }}

        function updateManualQCIndicator() {{
            const el = document.getElementById('manualQcIndicator');
            if (!el) return;
            const manualCount = getManualQCCount();
            el.textContent = "qc'd so far: " + manualCount.toLocaleString();
        }}

        function getSessionAnnotatedExamIds() {{
            const raw = Array.isArray(sessionStats.annotatedExamIds)
                ? sessionStats.annotatedExamIds
                : [];
            return [...new Set(
                raw
                    .map(examId => String(examId))
                    .filter(examId => hasAnnotationFindings(examId))
            )];
        }}

        function buildClientOnlyFilters() {{
            const sessionAnnotatedExamIds = getSessionAnnotatedExamIds();
            const filters = [];
            if (sessionAnnotatedExamIds.length > 0) {{
                filters.push({{
                    name: 'This Session Annotated (' + sessionAnnotatedExamIds.length + ')',
                    filename: sessionAnnotatedFilterFilename,
                    token: sessionAnnotatedFilterFilename,
                    aliases: [
                        sessionAnnotatedFilterFilename,
                        'this_session_annotated'
                    ],
                    kind: 'annotation_session',
                    clientExamIds: sessionAnnotatedExamIds
                }});
            }}
            return filters;
        }}

        function updateSessionAnnotatedButton() {{
            const btn = document.getElementById('sessionAnnotatedBtn');
            if (!btn) return;
            const count = getSessionAnnotatedExamIds().length;
            btn.textContent = count > 0 ? 'this session (' + count + ')' : 'this session';
            btn.disabled = count === 0;
            btn.title = count > 0
                ? 'show only exams annotated in this browser session'
                : 'no exams annotated in this browser session yet';
        }}

        function syncSessionAnnotatedExam(examId) {{
            const examIdStr = String(examId);
            const current = new Set(
                Array.isArray(sessionStats.annotatedExamIds)
                    ? sessionStats.annotatedExamIds.map(existing => String(existing))
                    : []
            );
            if (hasAnnotationFindings(examIdStr)) {{
                current.add(examIdStr);
            }} else {{
                current.delete(examIdStr);
            }}
            sessionStats.annotatedExamIds = [...current].sort();
            saveSessionStats();
            updateSessionAnnotatedButton();
        }}
        
        // load statuses from server-rendered payload first
        allExams.forEach(exam => {{
            if (validStatuses.has(exam.qc_status)) {{
                qcData[exam.exam_id] = exam.qc_status;
            }}
        }});
        updateManualQCIndicator();

        // for --serve, hydrate full QC map from server to preserve status history
        if (serverMode) {{
            fetch('/load-qc')
                .then(response => response.json())
                .then(serverQCData => {{
                    const qcFromServer = (serverQCData && typeof serverQCData === 'object') ? serverQCData : {{}};
                    Object.keys(qcFromServer).forEach(examId => {{
                        const serverStatus = qcFromServer[examId];
                        if (validStatuses.has(serverStatus)) {{
                            qcData[examId] = serverStatus;
                        }}
                    }});
                    localStorage.setItem(qcStorageKey, JSON.stringify(qcData));
                    updateManualQCIndicator();
                    applyFilters();
                    console.log('loaded full QC map from server for', Object.keys(qcData).length, 'exams');
                }})
                .catch(error => {{
                    console.error('failed to load full QC map from server:', error);
                    localStorage.setItem(qcStorageKey, JSON.stringify(qcData));
                }});
        }} else {{
            // in local file mode, backfill from localStorage
            const savedQCData = localStorage.getItem(qcStorageKey);
            if (savedQCData) {{
                try {{
                    const parsed = JSON.parse(savedQCData);
                    allExams.forEach(exam => {{
                        const savedStatus = parsed[exam.exam_id];
                        if (!qcData[exam.exam_id] && validStatuses.has(savedStatus)) {{
                            qcData[exam.exam_id] = savedStatus;
                        }}
                    }});
                    updateManualQCIndicator();
                }} catch (e) {{
                    console.error('Failed to parse saved QC data:', e);
                }}
            }}
        }}

        const defaultAnnotationTags = [
            "vertical line (detector artifact)",
            "horizontal line (detector artifact)"
        ];
        const legacyAnnotationTagAliases = {{
            "detector artifact - vertical line": "vertical line (detector artifact)",
            "detector artifact - horizontal line": "horizontal line (detector artifact)",
            "horizontal line (detector artifact": "horizontal line (detector artifact)"
        }};

        function canonicalizeAnnotationTag(tag) {{
            const raw = String(tag || '').trim();
            if (raw === '') return '';
            return legacyAnnotationTagAliases[raw] || raw;
        }}

        function normalizeAnnotationValues(tags) {{
            const normalized = [];
            tags.forEach(tag => {{
                const canonical = canonicalizeAnnotationTag(tag);
                if (canonical === '') return;
                if (!normalized.includes(canonical)) {{
                    normalized.push(canonical);
                }}
            }});

            return normalized;
        }}

        function normalizeAnnotationTagCatalog(tags) {{
            const normalized = normalizeAnnotationValues(tags);
            const extras = normalized.filter(tag => !defaultAnnotationTags.includes(tag));
            return [...defaultAnnotationTags, ...extras];
        }}

        function normalizeAnnotationMap(rawMap) {{
            const normalized = {{}};
            if (!rawMap || typeof rawMap !== 'object') {{
                return normalized;
            }}
            Object.keys(rawMap).forEach(examId => {{
                const tags = normalizeAnnotationValues(Array.isArray(rawMap[examId]) ? rawMap[examId] : []);
                if (tags.length > 0) {{
                    normalized[String(examId)] = tags;
                }}
            }});
            return normalized;
        }}

        function normalizeAutoSuggestionMap(rawRun) {{
            const normalized = {{}};
            const examSuggestions = rawRun && typeof rawRun === 'object'
                ? rawRun.exam_suggestions
                : null;
            if (!examSuggestions || typeof examSuggestions !== 'object') {{
                return normalized;
            }}
            Object.keys(examSuggestions).forEach(examId => {{
                const record = examSuggestions[examId];
                const entries = Array.isArray(record && record.suggestions) ? record.suggestions : [];
                const deduped = [];
                const seenTags = new Set();
                entries.forEach(entry => {{
                    const canonical = canonicalizeAnnotationTag(entry && entry.tag);
                    if (!canonical || seenTags.has(canonical)) return;
                    seenTags.add(canonical);
                    const normalizedEntry = {{ tag: canonical }};
                    if (entry && Number.isFinite(Number(entry.score))) {{
                        normalizedEntry.score = Number(entry.score);
                    }}
                    if (entry && typeof entry.rationale === 'string' && entry.rationale.trim() !== '') {{
                        normalizedEntry.rationale = entry.rationale.trim();
                    }}
                    deduped.push(normalizedEntry);
                }});
                if (deduped.length > 0) {{
                    normalized[String(examId)] = deduped;
                }}
            }});
            return normalized;
        }}
        
        // load annotations from localStorage first (survives page refresh)
        const savedAnnotations = localStorage.getItem(annotationsStorageKey);
        if (savedAnnotations) {{
            try {{
                annotations = normalizeAnnotationMap(JSON.parse(savedAnnotations));
            }} catch (e) {{
                console.error('failed to parse saved annotations:', e);
            }}
        }}
        const savedAnnotationTags = localStorage.getItem(annotationTagsStorageKey);
        if (savedAnnotationTags) {{
            try {{
                annotationTags = normalizeAnnotationTagCatalog(JSON.parse(savedAnnotationTags));
            }} catch (e) {{
                console.error('failed to parse saved annotation tags:', e);
            }}
        }}
        updateSessionAnnotatedButton();
        
        // then merge with server data (server wins for tags, merge for annotations)
        fetch('/load-annotation-tags')
            .then(response => response.json())
            .then(tags => {{
                if (tags.length > 0) {{
                    // merge: keep any localStorage tags not on server, add server tags
                    annotationTags = normalizeAnnotationTagCatalog([...tags, ...annotationTags]);
                    localStorage.setItem(annotationTagsStorageKey, JSON.stringify(annotationTags));
                }}
                console.log('loaded annotation tags:', annotationTags);
            }})
            .catch(error => {{
                console.error('failed to load annotation tags from server:', error);
                if (annotationTags.length === 0) {{
                    annotationTags = [...defaultAnnotationTags];
                }}
            }});
        
        fetch('/load-annotations')
            .then(response => response.json())
            .then(data => {{
                // merge server annotations with localStorage
                annotations = {{
                    ...annotations,
                    ...normalizeAnnotationMap(data)
                }};
                console.log('loaded annotations for', Object.keys(annotations).length, 'exams');
                if (
                    (!Array.isArray(sessionStats.annotatedExamIds) || sessionStats.annotatedExamIds.length === 0) &&
                    allExams.some(exam => hasAnnotationFindings(exam.exam_id))
                ) {{
                    // bootstrap current-session recall for older browser state that only tracked counts
                    sessionStats.annotatedExamIds = allExams
                        .filter(exam => hasAnnotationFindings(exam.exam_id))
                        .map(exam => String(exam.exam_id));
                }} else {{
                    sessionStats.annotatedExamIds = getSessionAnnotatedExamIds();
                }}
                saveSessionStats();
                updateSessionAnnotatedButton();
                updateManualQCIndicator();
                applyFilters();
            }})
            .catch(error => {{
                console.error('failed to load annotations from server:', error);
            }});

        loadAutoSuggestionsFromServer();
        
        function saveAnnotationTags() {{
            localStorage.setItem(annotationTagsStorageKey, JSON.stringify(annotationTags));
            return fetch('/save-annotation-tags', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify(annotationTags)
            }}).catch(error => console.error('failed to save annotation tags:', error));
        }}
        
        function saveAnnotations() {{
            localStorage.setItem(annotationsStorageKey, JSON.stringify(annotations));
            return fetch('/save-annotations', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify(annotations)
            }})
                .then(response => {{
                    if (!response.ok) {{
                        throw new Error('HTTP ' + response.status);
                    }}
                    invalidateFilterCaches();
                    return loadAvailableFilters();
                }})
                .then(() => {{
                    applyFilters();
                }})
                .catch(error => console.error('failed to save annotations:', error));
        }}

        function syncQCFromServerMap(serverQCData) {{
            const qcFromServer = (serverQCData && typeof serverQCData === 'object') ? serverQCData : {{}};
            Object.keys(qcData).forEach(examId => {{
                delete qcData[examId];
            }});
            Object.keys(qcFromServer).forEach(examId => {{
                const serverStatus = qcFromServer[examId];
                if (validStatuses.has(serverStatus)) {{
                    qcData[examId] = serverStatus;
                }}
            }});
            localStorage.setItem(qcStorageKey, JSON.stringify(qcData));
        }}

        function updateAutoSuggestionButton() {{
            const btn = document.getElementById('acceptVisibleAutoBtn');
            if (!btn) return;
            const visibleSuggestedCount = filteredExams.filter(exam => hasAutoSuggestions(exam.exam_id)).length;
            btn.disabled = visibleSuggestedCount === 0;
            btn.textContent = visibleSuggestedCount > 0
                ? 'accept visible auto (' + visibleSuggestedCount + ')'
                : 'accept visible auto';
            btn.title = visibleSuggestedCount > 0
                ? 'accept all model suggestions for the currently visible exams'
                : 'no visible exams have model suggestions';
        }}

        function loadAutoSuggestionsFromServer() {{
            if (!autoSuggestionRunPath) {{
                autoSuggestionRun = {{}};
                autoSuggestions = {{}};
                updateAutoSuggestionButton();
                return Promise.resolve(autoSuggestions);
            }}

            return fetch('/load-auto-suggestions')
                .then(response => response.json())
                .then(data => {{
                    autoSuggestionRun = (data && typeof data === 'object') ? data : {{}};
                    autoSuggestions = normalizeAutoSuggestionMap(autoSuggestionRun);
                    invalidateFilterCaches();
                    const autoReviewBtn = document.getElementById('autoReviewBtn');
                    if (autoReviewBtn) {{
                        autoReviewBtn.style.display = Object.keys(autoSuggestions).length > 0 ? 'flex' : 'none';
                    }}
                    const acceptBtn = document.getElementById('acceptVisibleAutoBtn');
                    if (acceptBtn) {{
                        acceptBtn.style.display = Object.keys(autoSuggestions).length > 0 ? 'inline-block' : 'none';
                    }}
                    return loadAvailableFilters()
                        .catch(error => {{
                            console.error('failed to reload filters after loading auto suggestions:', error);
                        }})
                        .then(() => {{
                            updateAutoSuggestionButton();
                            applyFilters();
                            return autoSuggestions;
                        }});
                }})
                .catch(error => {{
                    console.error('failed to load auto suggestions from server:', error);
                    autoSuggestionRun = {{}};
                    autoSuggestions = {{}};
                    updateAutoSuggestionButton();
                    return autoSuggestions;
                }});
        }}

        function computeAutoComparisonRows(examPool) {{
            const exams = Array.isArray(examPool) ? examPool : [];
            const examIds = exams.map(exam => String(exam.exam_id));
            const tagSet = new Set(annotationTags);
            examIds.forEach(examId => {{
                getExamAnnotations(examId).forEach(tag => tagSet.add(tag));
                getAutoSuggestions(examId).forEach(entry => tagSet.add(entry.tag));
            }});

            const rows = [];
            [...tagSet].sort().forEach(tag => {{
                let tp = 0;
                let fp = 0;
                let fn = 0;
                let tn = 0;
                examIds.forEach(examId => {{
                    const hasGT = getExamAnnotations(examId).includes(tag);
                    const hasPred = getAutoSuggestions(examId).some(entry => entry.tag === tag);
                    if (hasGT && hasPred) tp++;
                    else if (!hasGT && hasPred) fp++;
                    else if (hasGT && !hasPred) fn++;
                    else tn++;
                }});
                if (tp === 0 && fp === 0 && fn === 0) {{
                    return;
                }}
                rows.push({{
                    tag,
                    tp,
                    fp,
                    fn,
                    tn,
                    precision: (tp + fp) > 0 ? tp / (tp + fp) : null,
                    recall: (tp + fn) > 0 ? tp / (tp + fn) : null
                }});
            }});

            return rows.sort((a, b) => (
                (b.tp - a.tp) ||
                (b.fp - a.fp) ||
                a.tag.localeCompare(b.tag)
            ));
        }}

        function formatMetric(value) {{
            return value === null ? 'n/a' : (value * 100).toFixed(1) + '%';
        }}

        function renderAutoReviewContent() {{
            const content = document.getElementById('autoReviewContent');
            if (!content) return;
            if (Object.keys(autoSuggestions).length === 0) {{
                content.innerHTML = '<p style="color: #9cdcfe;">No auto suggestion run is loaded.</p>';
                return;
            }}

            const rows = computeAutoComparisonRows(allExams);
            const suggestedVisible = allExams.filter(exam => hasAutoSuggestions(exam.exam_id)).length;
            let html = '';
            html += '<p><strong>run:</strong> ' + (autoSuggestionRun.run_id || '(unnamed)') + '</p>';
            html += '<p><strong>model:</strong> ' + (autoSuggestionRun.model || '(not recorded)') + '</p>';
            html += '<p><strong>pool:</strong> ' + allExams.length + ' loaded exams, ' + suggestedVisible + ' with model suggestions</p>';
            html += '<p style="color: #858585;">Metrics below are computed on the currently loaded gallery pool.</p>';
            if (rows.length === 0) {{
                html += '<p style="color: #9cdcfe;">No overlapping GT/predicted tags in the current pool yet.</p>';
                content.innerHTML = html;
                return;
            }}

            html += '<table class="cutflow-table"><thead><tr>';
            html += '<th>tag</th><th>TP</th><th>FP</th><th>FN</th><th>TN</th><th>precision</th><th>recall</th>';
            html += '</tr></thead><tbody>';
            rows.forEach(row => {{
                html += '<tr>';
                html += '<td>' + row.tag + '</td>';
                html += '<td style="color: #6bcc6b;">' + row.tp + '</td>';
                html += '<td style="color: ' + (row.fp > 0 ? '#e06b6b' : '#d4d4d4') + ';">' + row.fp + '</td>';
                html += '<td style="color: ' + (row.fn > 0 ? '#e5c07b' : '#d4d4d4') + ';">' + row.fn + '</td>';
                html += '<td>' + row.tn + '</td>';
                html += '<td>' + formatMetric(row.precision) + '</td>';
                html += '<td>' + formatMetric(row.recall) + '</td>';
                html += '</tr>';
            }});
            html += '</tbody></table>';
            content.innerHTML = html;
        }}

        function showAutoReview() {{
            const modal = document.getElementById('autoReviewModal');
            modal.style.display = 'block';
            renderAutoReviewContent();
        }}

        function closeAutoReview() {{
            document.getElementById('autoReviewModal').style.display = 'none';
        }}

        async function acceptVisibleAutoSuggestions() {{
            const visibleExamIds = filteredExams
                .map(exam => String(exam.exam_id))
                .filter(examId => hasAutoSuggestions(examId));
            if (visibleExamIds.length === 0) {{
                alert('No visible exams have auto suggestions.');
                return;
            }}

            const ok = confirm(
                'Accept model suggestions for ' + visibleExamIds.length + ' currently visible exams? ' +
                'This will add suggested tags into accepted annotations and preserve auto provenance.'
            );
            if (!ok) return;

            try {{
                const doneBefore = {{}};
                visibleExamIds.forEach(examId => {{
                    doneBefore[examId] = isDoneExam(examId);
                }});

                const response = await fetch('/accept-auto-suggestions', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ exam_ids: visibleExamIds }})
                }});
                const data = await response.json();
                if (!response.ok || data.error) {{
                    throw new Error(data.error || ('HTTP ' + response.status));
                }}

                const [serverAnnotations, serverQCData] = await Promise.all([
                    fetch('/load-annotations').then(resp => resp.json()),
                    fetch('/load-qc').then(resp => resp.json())
                ]);
                annotations = normalizeAnnotationMap(serverAnnotations);
                syncQCFromServerMap(serverQCData);
                retainedAnnotatedExamIds = new Set(
                    [...retainedAnnotatedExamIds, ...visibleExamIds]
                );
                visibleExamIds.forEach(examId => {{
                    updateDoneCounters(Boolean(doneBefore[examId]), isDoneExam(examId));
                    syncAnnotatedExamPoolForExam(examId);
                    syncSessionAnnotatedExam(examId);
                }});
                updateManualQCIndicator();
                renderAutoReviewContent();
                applyFilters();
                alert(
                    'Accepted auto suggestions for ' + data.updated_exams +
                    ' exams (' + data.added_tags + ' new tags).'
                );
            }} catch (error) {{
                console.error('failed to accept visible auto suggestions:', error);
                alert('Failed to accept auto suggestions: ' + (error.message || error));
            }}
        }}

        function getAnnotationFilterAliases(tag) {{
            const aliases = [];
            availableFilters.forEach(filter => {{
                if (!filter || filter.kind !== 'annotation') return;
                if (filter.name !== `Annotation: ${{tag}}`) return;
                if (filter.token) {{
                    aliases.push('@' + String(filter.token).replace(/^@+/, ''));
                }}
                if (filter.filename) {{
                    aliases.push('@' + String(filter.filename).replace(/^@+/, ''));
                }}
                if (Array.isArray(filter.aliases)) {{
                    filter.aliases.forEach(alias => {{
                        const normalized = String(alias || '').trim().replace(/^@+/, '');
                        if (normalized) {{
                            aliases.push('@' + normalized);
                        }}
                    }});
                }}
            }});

            if (aliases.length === 0) {{
                const fallbackToken = filterTokenFromName(tag);
                if (fallbackToken) {{
                    aliases.push('@' + fallbackToken);
                    aliases.push('@annotation_' + fallbackToken);
                }}
            }}

            return [...new Set(aliases)];
        }}

        function replaceRenamedAnnotationFilterTokens(oldTag, newTag) {{
            const input = document.getElementById('searchBox');
            if (!input) return;

            let updated = input.value;
            const oldAliases = getAnnotationFilterAliases(oldTag);
            const newAliases = getAnnotationFilterAliases(newTag);
            const replacement = newAliases[0] || ('@' + filterTokenFromName(newTag));

            oldAliases.forEach(alias => {{
                if (!alias) return;
                updated = updated.split(alias).join(replacement);
            }});

            if (updated !== input.value) {{
                input.value = updated;
            }}

            const select = document.getElementById('filterSelect');
            if (select) {{
                select.value = '';
            }}
        }}

        function buildAnnotationHotkeys(tags) {{
            const keyToTag = {{}};
            const tagToKey = {{}};
            const usedKeys = new Set();
            const fallbackPool = 'abcdefghijklmnopqrstuvwxyz0123456789';
            const preferredHotkeys = {{
                'vertical line (detector artifact)': 'v',
                'horizontal line (detector artifact)': 'h'
            }};

            Object.entries(preferredHotkeys).forEach(([tag, preferredKey]) => {{
                if (tags.includes(tag) && !usedKeys.has(preferredKey)) {{
                    usedKeys.add(preferredKey);
                    keyToTag[preferredKey] = tag;
                    tagToKey[tag] = preferredKey;
                }}
            }});

            tags.forEach(tag => {{
                if (tagToKey[tag]) return;
                const normalized = tag.toLowerCase().replace(/[^a-z0-9]/g, '');
                let selectedKey = '';

                for (const ch of normalized) {{
                    if (!usedKeys.has(ch)) {{
                        selectedKey = ch;
                        break;
                    }}
                }}

                if (!selectedKey) {{
                    for (const ch of fallbackPool) {{
                        if (!usedKeys.has(ch)) {{
                            selectedKey = ch;
                            break;
                        }}
                    }}
                }}

                if (selectedKey) {{
                    usedKeys.add(selectedKey);
                    keyToTag[selectedKey] = tag;
                    tagToKey[tag] = selectedKey;
                }}
            }});

            return {{ keyToTag, tagToKey }};
        }}
        
        function checkAndPreload() {{
            // check if we should start preloading next batch
            if (!enablePreload) return;
            if (preloadTriggered) return;
            
            // count how many exams in current batch have been QC'd
            let qcdCount = 0;
            allExams.forEach(exam => {{
                if (isDoneExam(exam.exam_id)) {{
                    qcdCount++;
                }}
            }});
            
            // trigger preload at 80% progress
            const progress = qcdCount / allExams.length;
            if (progress >= 0.80) {{
                preloadTriggered = true;
                console.log('🚀 Preloading next batch at ' + (progress * 100).toFixed(0) + '% progress...');
                
                fetch('/preload-more')
                    .then(response => response.json())
                    .then(data => {{
                        if (data.status === 'preloading') {{
                            console.log('✓ Next batch preloading in background');
                        }} else if (data.status === 'already_loading') {{
                            console.log('→ Preload already in progress');
                        }}
                    }})
                    .catch(error => {{
                        console.error('Preload failed:', error);
                    }});
            }}
        }}

        function saveQCToServer(options = {{}}) {{
            const replace = options.replace === true;
            localStorage.setItem(qcStorageKey, JSON.stringify(qcData));
            const saveUrl = replace ? '/save-qc?replace=1' : '/save-qc';

            return fetch(saveUrl, {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                }},
                body: JSON.stringify(qcData)
            }})
            .then(async response => {{
                let data = {{}};
                try {{
                    data = await response.json();
                }} catch (e) {{
                    data = {{}};
                }}
                if (!response.ok || data.error) {{
                    const message = data.error || ('HTTP ' + response.status);
                    throw new Error(message);
                }}
                console.log('QC data saved to server:', data);
                return data;
            }});
        }}
        
        function autoSaveQCData() {{
            // check if we should preload next batch
            checkAndPreload();

            saveQCToServer().catch(error => {{
                console.error('Failed to save to server:', error);
            }});
        }}
        
        function advanceAfterQCAction() {{
            if (currentIndex === filteredExams.length - 1) {{
                showCompletion();
            }} else {{
                navigate(1);
            }}
        }}

        function setQCStatus(status) {{
            if (filteredExams.length === 0) return;
            if (status !== 'good') return;
            const exam = filteredExams[currentIndex];
            const examAnnotations = getExamAnnotations(exam.exam_id);
            if (status === 'good' && examAnnotations.length > 0) {{
                const clearAnnotations = confirm(
                    'This exam has annotation tags. Marking it good means no findings and will remove those tags. Continue?'
                );
                if (!clearAnnotations) {{
                    return;
                }}
                delete annotations[exam.exam_id];
                retainedAnnotatedExamIds.delete(String(exam.exam_id));
                syncSessionAnnotatedExam(exam.exam_id);
                saveAnnotations();
            }}
            const wasDone = isDoneExam(exam.exam_id);
            qcData[exam.exam_id] = status;
            exam.qc_status = status;
            const willBeDone = isDoneExam(exam.exam_id);
            updateDoneCounters(wasDone, willBeDone);
            autoSaveQCData();
            updateManualQCIndicator();
            advanceAfterQCAction();
        }}
        
        function renderAnnotationTagsList() {{
            const tagsList = document.getElementById('annotationTagsList');
            if (!tagsList || filteredExams.length === 0) return;
            const exam = filteredExams[currentIndex];
            const examAnnotations = annotations[exam.exam_id] || [];
            
            tagsList.innerHTML = '';
            annotationTags.forEach(tag => {{
                const hotkey = annotationHotkeys.tagToKey[tag];
                const isActive = examAnnotations.includes(tag);
                const row = document.createElement('div');
                row.style.cssText = `
                    display: flex;
                    gap: 10px;
                    align-items: stretch;
                    margin-bottom: 10px;
                `;
                const btn = document.createElement('button');
                const hotkeyLabel = hotkey || '';
                if (hotkeyLabel) {{
                    btn.textContent = `[${{hotkeyLabel}}] ${{tag}}`;
                }} else {{
                    btn.textContent = tag;
                }}
                btn.dataset.tag = tag;
                btn.style.cssText = `
                    display: block;
                    flex: 1;
                    padding: 12px 16px;
                    font-size: 14px;
                    text-align: left;
                    border: 1px solid ${{isActive ? '#4ec9b0' : '#3e3e42'}};
                    border-radius: 4px;
                    background-color: ${{isActive ? '#2d4a44' : '#1e1e1e'}};
                    color: ${{isActive ? '#4ec9b0' : '#d4d4d4'}};
                    cursor: pointer;
                    transition: all 0.15s;
                `;
                if (isActive) {{
                    if (hotkeyLabel) {{
                        btn.textContent = `[${{hotkeyLabel}}] ✓ ${{tag}}`;
                    }} else {{
                        btn.textContent = `✓ ${{tag}}`;
                    }}
                }}
                btn.onclick = () => {{
                    toggleAnnotation(tag);
                }};

                const editBtn = document.createElement('button');
                editBtn.textContent = 'edit';
                editBtn.style.cssText = `
                    padding: 12px 14px;
                    font-size: 13px;
                    border: 1px solid #3e3e42;
                    border-radius: 4px;
                    background-color: #252526;
                    color: #d4d4d4;
                    cursor: pointer;
                    white-space: nowrap;
                `;
                editBtn.onclick = () => {{
                    renameAnnotationTag(tag);
                }};

                row.appendChild(btn);
                row.appendChild(editBtn);
                tagsList.appendChild(row);
            }});
        }}

        function showAnnotationModal() {{
            const modal = document.getElementById('annotationModal');
            annotationHotkeys = buildAnnotationHotkeys(annotationTags);
            renderAnnotationTagsList();
            modal.style.display = 'block';
        }}
        
        function closeAnnotationModal() {{
            const modal = document.getElementById('annotationModal');
            const input = document.getElementById('newAnnotationInput');
            modal.style.display = 'none';
            input.value = '';
        }}

        function closeAnnotationModalAndStay() {{
            closeAnnotationModal();
        }}
        
        function toggleAnnotation(tag) {{
            const exam = filteredExams[currentIndex];
            const examId = String(exam.exam_id);
            const wasDone = isDoneExam(exam.exam_id);
            if (!annotations[examId]) {{
                annotations[examId] = [];
            }}
            
            const idx = annotations[examId].indexOf(tag);
            if (idx >= 0) {{
                annotations[examId].splice(idx, 1);
                if (annotations[examId].length === 0) {{
                    delete annotations[examId];
                }}
            }} else {{
                annotations[examId].push(tag);
            }}
            if (hasAnnotationFindings(examId)) {{
                retainedAnnotatedExamIds.add(examId);
            }} else {{
                retainedAnnotatedExamIds.delete(examId);
            }}
            const willBeDone = isDoneExam(exam.exam_id);
            updateDoneCounters(wasDone, willBeDone);
            syncAnnotatedExamPoolForExam(exam.exam_id);
            syncSessionAnnotatedExam(exam.exam_id);
            
            saveAnnotations();
            updateManualQCIndicator();
            checkAndPreload();
            renderAnnotationTagsList();
            updateView();
        }}

        function renameAnnotationTag(oldTag) {{
            const proposed = prompt('rename annotation tag:', oldTag);
            if (proposed === null) return;

            const newTag = canonicalizeAnnotationTag(proposed);
            if (newTag === '') {{
                alert('annotation tag cannot be empty');
                return;
            }}
            if (newTag === oldTag) {{
                return;
            }}

            annotationTags = normalizeAnnotationTagCatalog(
                annotationTags.map(tag => (tag === oldTag ? newTag : tag))
            );

            Object.keys(annotations).forEach(examId => {{
                const currentTags = Array.isArray(annotations[examId]) ? annotations[examId] : [];
                const renamedTags = [];
                currentTags.forEach(tag => {{
                    const canonical = canonicalizeAnnotationTag(tag === oldTag ? newTag : tag);
                    if (canonical && !renamedTags.includes(canonical)) {{
                        renamedTags.push(canonical);
                    }}
                }});
                if (renamedTags.length > 0) {{
                    annotations[examId] = renamedTags;
                }} else {{
                    delete annotations[examId];
                }}
            }});

            replaceRenamedAnnotationFilterTokens(oldTag, newTag);
            annotationHotkeys = buildAnnotationHotkeys(annotationTags);
            saveAnnotationTags();
            saveAnnotations();
            updateManualQCIndicator();
            renderAnnotationTagsList();
            updateView();
        }}
        
        function addNewAnnotationTag() {{
            const input = document.getElementById('newAnnotationInput');
            const newTag = canonicalizeAnnotationTag(input.value);
            
            if (newTag === '') return;
            
            if (annotationTags.includes(newTag)) {{
                alert('this tag already exists');
                return;
            }}
            
            // add tag to available list and persist
            annotationTags.push(newTag);
            saveAnnotationTags();
            annotationHotkeys = buildAnnotationHotkeys(annotationTags);
            
            // toggle it on for current exam
            toggleAnnotation(newTag);
            input.value = '';
        }}
        
        // allow Enter key in annotation input
        document.addEventListener('DOMContentLoaded', () => {{
            const input = document.getElementById('newAnnotationInput');
            if (input) {{
                input.addEventListener('keypress', (e) => {{
                    if (e.key === 'Enter') {{
                        addNewAnnotationTag();
                    }}
                }});
            }}
        }});
        
        function nextExam() {{
            // advance to next exam without changing QC status
            advanceAfterQCAction();
        }}
        
        function showCompletion() {{
            const banner = document.getElementById('completionBanner');
            const statsDiv = document.getElementById('completionStats');
            
            // count QC statuses
            const qcCounts = {{good: 0, pending: 0}};
            allExams.forEach(exam => {{
                if (getStatus(exam.exam_id) === 'good') qcCounts.good++;
                else if (!hasAnnotationFindings(exam.exam_id)) qcCounts.pending++;
            }});
            const annotatedCount = allExams.filter(exam => {{
                return hasAnnotationFindings(exam.exam_id);
            }}).length;
            
            statsDiv.innerHTML = 
                '<div><strong>total exams in batch:</strong> ' + allExams.length + '</div>' +
                '<div style="color: #6bcc6b;"><strong>good:</strong> ' + qcCounts.good + '</div>' +
                '<div style="color: #4ec9b0;"><strong>annotated:</strong> ' + annotatedCount + '</div>' +
                '<div style="color: #9cdcfe;"><strong>pending:</strong> ' + qcCounts.pending + '</div>';
            
            banner.style.display = 'block';
        }}
        
        function closeCompletion() {{
            document.getElementById('completionBanner').style.display = 'none';
        }}

        function refreshCurrentBatch() {{
            const ok = confirm(
                'Rebuild the current gallery from saved QC status? This is safe: it does not delete QC records or source DICOM files.'
            );
            if (!ok) return;

            saveQCToServer()
                .then(() => fetch('/regenerate'))
                .then(response => response.json())
                .then(data => {{
                    if (data.error) {{
                        alert('Error refreshing batch: ' + data.error);
                        return;
                    }}
                    console.log('Batch refresh triggered:', data);
                    setTimeout(() => {{
                        location.reload();
                    }}, data.reload_delay_ms || 3000);
                }})
                .catch(error => {{
                    console.error('Failed to refresh batch:', error);
                    alert('Failed to refresh batch: ' + (error.message || error));
                }});
        }}

        function renderLoadingStatus(title, detail, progress = null) {{
            let progressHtml;
            if (Number.isFinite(progress)) {{
                const pct = Math.max(0, Math.min(100, Math.round(progress * 100)));
                progressHtml =
                    '<div class="progress-shell">' +
                        '<div class="progress-bar" style="width: ' + pct + '%;"></div>' +
                    '</div>' +
                    '<p style="font-size: 12px; color: #9cdcfe; margin: 8px 0 0 0;">' + pct + '%</p>';
            }} else {{
                progressHtml = '<div class="progress-shell"><div class="progress-bar indeterminate"></div></div>';
            }}
            return '<p style="color: #4ec9b0; font-weight: bold; margin: 0;">' + title + '</p>' +
                '<p style="font-size: 12px; color: #858585; margin: 10px 0 0 0;">' + detail + '</p>' +
                progressHtml;
        }}

        function renderReadyStatus(title, detail) {{
            return '<p style="color: #4ec9b0; font-weight: bold; margin: 0;">' + title + '</p>' +
                '<p style="font-size: 12px; color: #858585; margin: 10px 0 0 0;">' + detail + '</p>';
        }}

        function pollLoadMoreStatus(loadMoreBtn, statusDiv, attempt = 0) {{
            const maxAttempts = 600;  // ~15 minutes at 1.5s
            fetch('/load-more-status')
                .then(response => response.json())
                .then(status => {{
                    if (status.error) {{
                        alert('Failed to generate more exams: ' + status.error);
                        loadMoreBtn.disabled = false;
                        loadMoreBtn.textContent = '🔄 Load More Exams [l]';
                        statusDiv.style.display = 'none';
                        return;
                    }}

                    if (status.ready) {{
                        statusDiv.innerHTML = renderReadyStatus('✓ Next batch ready!', 'Reloading now...');
                        setTimeout(() => {{
                            location.reload();
                        }}, 200);
                        return;
                    }}

                    if (status.in_progress) {{
                        const target = status.next_batch ? status.next_batch.toLocaleString() : 'more';
                        statusDiv.innerHTML = renderLoadingStatus(
                            status.message || ('⏳ Generating ' + target + ' exams...'),
                            'This may take a minute. Page will reload when ready.',
                            status.progress
                        );
                    }}

                    if (attempt >= maxAttempts) {{
                        loadMoreBtn.disabled = false;
                        loadMoreBtn.textContent = '🔄 Load More Exams [l]';
                        statusDiv.innerHTML = renderReadyStatus(
                            'Still generating in background.',
                            'Wait a bit and click load more again to check status.'
                        );
                        return;
                    }}

                    setTimeout(() => {{
                        pollLoadMoreStatus(loadMoreBtn, statusDiv, attempt + 1);
                    }}, 1500);
                }})
                .catch(error => {{
                    console.error('Failed to check load-more status:', error);
                    if (attempt >= maxAttempts) {{
                        loadMoreBtn.disabled = false;
                        loadMoreBtn.textContent = '🔄 Load More Exams [l]';
                        statusDiv.style.display = 'none';
                        return;
                    }}
                    setTimeout(() => {{
                        pollLoadMoreStatus(loadMoreBtn, statusDiv, attempt + 1);
                    }}, 2000);
                }});
        }}
        
        async function loadMore() {{
            const loadMoreBtn = document.getElementById('loadMoreBtn');
            const statusDiv = document.getElementById('loadMoreStatus');

            if (loadMoreBtn.disabled) return;
            
            const batchInput = document.getElementById('nextBatchSize');
            const batchSize = batchInput ? parseInt(batchInput.value, 10) || {max_exams or 100} : {max_exams or 100};
            
            // disable button and show status
            loadMoreBtn.disabled = true;
            loadMoreBtn.textContent = '⏳ Loading...';
            statusDiv.style.display = 'block';
            statusDiv.innerHTML = renderLoadingStatus(
                '⏳ Saving latest QC decisions...',
                'Starting load-more right after save completes.'
            );

            try {{
                await saveQCToServer();

                const response = await fetch('/load-more?batch_size=' + batchSize);
                const data = await response.json();
                if (!response.ok || data.error) {{
                    throw new Error(data.error || ('HTTP ' + response.status));
                }}
                
                console.log('Load more triggered:', data);

                statusDiv.innerHTML = renderLoadingStatus(
                    '⏳ Generating more exams...',
                    'This may take a minute. Page will reload when ready.'
                );
                pollLoadMoreStatus(loadMoreBtn, statusDiv);
            }} catch (error) {{
                console.error('Failed to load more:', error);
                alert('Failed to load more exams: ' + (error.message || error));
                loadMoreBtn.disabled = false;
                loadMoreBtn.textContent = '🔄 Load More Exams [l]';
                statusDiv.style.display = 'none';
            }}
        }}
        
        async function loadFilterBatchForSelection(filterName, filterLabel) {{
            if (!filterName) {{
                alert('Select a filter from the dropdown first');
                return;
            }}

            // immediate visual feedback
            const btn = document.querySelector('[onclick="loadFilterBatch()"]');
            const origText = btn ? btn.textContent : '';
            if (btn) {{
                btn.disabled = true;
                btn.textContent = '⏳ Loading...';
            }}
            const stats = document.getElementById('stats');
            const origStats = stats ? stats.textContent : '';
            if (stats) stats.textContent = '⏳ Saving QC & requesting all ' + filterLabel + ' exams...';

            try {{
                await saveQCToServer();
                const response = await fetch('/load-filter-batch?filter=' + encodeURIComponent(filterName));
                const data = await response.json();
                if (!response.ok || data.error) {{
                    throw new Error(data.error || ('HTTP ' + response.status));
                }}
                console.log('Filter batch triggered:', data);

                if (stats) stats.textContent = '⏳ Generating all ' + filterLabel + ' exams... page will reload when ready';

                // reuse the load-more polling UI
                const loadMoreBtn = document.getElementById('loadMoreBtn');
                const statusDiv = document.getElementById('loadMoreStatus');
                if (loadMoreBtn) loadMoreBtn.disabled = true;
                if (statusDiv) {{
                    statusDiv.style.display = 'block';
                    statusDiv.innerHTML = renderLoadingStatus(
                        '⏳ ' + data.message,
                        'Page will reload when ready.'
                    );
                }}

                pollLoadMoreStatus(loadMoreBtn || document.createElement('button'), statusDiv || document.createElement('div'));
            }} catch (error) {{
                console.error('Failed to load filter batch:', error);
                alert('Failed to load filter batch: ' + (error.message || error));
                if (btn) {{ btn.disabled = false; btn.textContent = origText; }}
                if (stats) stats.textContent = origStats;
            }}
        }}

        async function loadFilterBatch() {{
            const select = document.getElementById('filterSelect');
            const filterName = select.value;
            if (!filterName) {{
                alert('Select a filter from the dropdown first');
                return;
            }}

            const selectedOption = select.options[select.selectedIndex];
            const filterLabel = selectedOption ? selectedOption.textContent : filterName.replace(/_/g, ' ');
            const filterKind =
                (selectedOption && selectedOption.dataset && selectedOption.dataset.kind) ||
                'auto';
            if (filterKind.startsWith('annotation')) {{
                await useSelectedFilterToken();
                return;
            }}
            await loadFilterBatchForSelection(filterName, filterLabel);
        }}

        async function useSessionAnnotatedFilter() {{
            const count = getSessionAnnotatedExamIds().length;
            if (count === 0) {{
                alert('No exams annotated in this browser session yet.');
                return;
            }}

            try {{
                await loadAvailableFilters();
                const select = document.getElementById('filterSelect');
                if (select) {{
                    select.value = sessionAnnotatedFilterFilename;
                }}
                replaceFilterBox('@' + sessionAnnotatedFilterFilename, {{ focus: false }});
                applyFilters();
            }} catch (error) {{
                alert('Failed to load session filter: ' + (error.message || error));
                console.error('Session filter load error:', error);
            }}
        }}

        function loadAvailableFilters() {{
            return fetch('/list-filters')
                .then(response => response.json())
                .then(filters => {{
                    const select = document.getElementById('filterSelect');
                    const serverFilters = Array.isArray(filters) ? filters : [];
                    const clientFilters = buildClientOnlyFilters();
                    availableFilters = [...serverFilters, ...clientFilters];
                    filtersLoaded = true;

                    // reset alias maps when reloading list
                    Object.keys(filterTokenToFilename).forEach(token => delete filterTokenToFilename[token]);
                    Object.keys(filterKindsByFilename).forEach(filename => delete filterKindsByFilename[filename]);
                    Object.keys(filterSetsByFilename).forEach(filename => delete filterSetsByFilename[filename]);

                    // keep the default option
                    select.innerHTML = '<option value="">Load filter list...</option>';
                    availableFilters.forEach(filter => {{
                        const option = document.createElement('option');
                        option.value = filter.filename;
                        option.textContent = filter.name;
                        option.dataset.token = filter.token || filterTokenFromFilename(filter.filename);
                        option.dataset.kind = filter.kind || 'auto';
                        filterKindsByFilename[filter.filename] = option.dataset.kind;
                        if (Array.isArray(filter.clientExamIds)) {{
                            filterSetsByFilename[filter.filename] = new Set(
                                filter.clientExamIds.map(examId => String(examId))
                            );
                        }}
                        select.appendChild(option);
                        registerFilterAliases(filter);
                    }});
                    updateSessionAnnotatedButton();
                    console.log(`Loaded ${{availableFilters.length}} filter options`);
                    if (document.getElementById('searchBox').value.trim() !== '') {{
                        applyFilters();
                    }}
                }})
                .catch(error => {{
                    console.error('Failed to load filter list:', error);
                    throw error;
                }});
        }}

        function normalizeFilterToken(token) {{
            return String(token || '').trim().toLowerCase();
        }}

        function filterTokenFromFilename(filename) {{
            return String(filename || '').replace(/\\.txt$/i, '').toLowerCase();
        }}

        function filterTokenFromName(name) {{
            return String(name || '')
                .trim()
                .toLowerCase()
                .replace(/[^a-z0-9]+/g, '_')
                .replace(/^_+|_+$/g, '');
        }}

        function registerFilterAliases(filter) {{
            const aliases = new Set();
            const stem = filterTokenFromFilename(filter.filename);
            if (stem) {{
                aliases.add(stem);
                aliases.add('@' + stem);
            }}
            aliases.add(normalizeFilterToken(filter.filename));
            const nameToken = filterTokenFromName(filter.name);
            if (nameToken) {{
                aliases.add(nameToken);
                aliases.add('@' + nameToken);
            }}
            if (Array.isArray(filter.aliases)) {{
                filter.aliases.forEach(alias => {{
                    const normalizedAlias = normalizeFilterToken(alias);
                    if (normalizedAlias) {{
                        aliases.add(normalizedAlias);
                        aliases.add('@' + normalizedAlias.replace(/^@+/, ''));
                    }}
                }});
            }}
            aliases.forEach(alias => {{
                const key = normalizeFilterToken(alias);
                if (key) {{
                    filterTokenToFilename[key] = filter.filename;
                }}
            }});
        }}

        function invalidateFilterCaches() {{
            availableFilters = [];
            filtersLoaded = false;
            Object.keys(filterTokenToFilename).forEach(token => delete filterTokenToFilename[token]);
            Object.keys(filterKindsByFilename).forEach(filename => delete filterKindsByFilename[filename]);
            Object.keys(filterSetsByFilename).forEach(filename => delete filterSetsByFilename[filename]);
            pendingFilterLoads.clear();
            unresolvedFilterWarnings.clear();
        }}

        function resolveFilterFilenameFromToken(token) {{
            const normalized = normalizeFilterToken(token);
            if (!normalized) return null;
            const direct = filterTokenToFilename[normalized];
            if (direct) return direct;
            if (normalized.startsWith('@')) {{
                return filterTokenToFilename[normalized.slice(1)] || null;
            }}
            return filterTokenToFilename['@' + normalized] || null;
        }}

        function isAnnotationFilterFilename(filename) {{
            return String(filterKindsByFilename[filename] || '').startsWith('annotation');
        }}

        function mergeExamPools(primaryExams, secondaryExams) {{
            const merged = [...primaryExams];
            const seen = new Set(primaryExams.map(exam => String(exam.exam_id)));
            secondaryExams.forEach(exam => {{
                const examId = String(exam.exam_id);
                if (!seen.has(examId)) {{
                    seen.add(examId);
                    merged.push(exam);
                }}
            }});
            return merged;
        }}

        function findKnownExamRecord(examId) {{
            const examIdStr = String(examId);
            const pools = [filteredExams, allExams, globalAnnotatedExams, globalAutoSuggestedExams];
            for (const pool of pools) {{
                const match = pool.find(exam => String(exam.exam_id) === examIdStr);
                if (match) {{
                    return {{ ...match }};
                }}
            }}
            return null;
        }}

        function syncAnnotatedExamPoolForExam(examId) {{
            const examIdStr = String(examId);
            const idx = globalAnnotatedExams.findIndex(exam => String(exam.exam_id) === examIdStr);
            if (hasAnnotationFindings(examIdStr)) {{
                const record = findKnownExamRecord(examIdStr);
                if (!record) {{
                    return;
                }}
                if (idx >= 0) {{
                    globalAnnotatedExams[idx] = record;
                }} else {{
                    globalAnnotatedExams.push(record);
                }}
            }} else if (idx >= 0) {{
                globalAnnotatedExams.splice(idx, 1);
            }}
        }}

        function filterUsesExpandedUniverse(filename) {{
            const kind = String(filterKindsByFilename[filename] || '');
            return kind.startsWith('annotation') || kind.startsWith('auto_suggestion');
        }}

        function expressionUsesExpandedUniverse(expression) {{
            const filterExpr = String(expression || '').trim();
            if (filterExpr === '' || !filterExpr.includes('@')) {{
                return false;
            }}
            try {{
                const tokens = tokenizeFilterExpression(filterExpr);
                return tokens.some(token => {{
                    if (token.type !== 'TERM') return false;
                    const filename = resolveFilterFilenameFromToken(token.value);
                    return Boolean(filename && filterUsesExpandedUniverse(filename));
                }});
            }} catch (error) {{
                return false;
            }}
        }}

        function getFilterSourceExams(filterExpr) {{
            if (!expressionUsesExpandedUniverse(filterExpr)) {{
                return [...allExams];
            }}
            return mergeExamPools(
                mergeExamPools(allExams, globalAnnotatedExams),
                globalAutoSuggestedExams
            );
        }}

        async function ensureFilterSetLoaded(filename) {{
            if (filterSetsByFilename[filename]) {{
                return filterSetsByFilename[filename];
            }}
            const response = await fetch('/load-filter/' + encodeURIComponent(filename));
            const data = await response.json();
            if (!response.ok || data.error) {{
                throw new Error(data.error || ('HTTP ' + response.status));
            }}
            const examIds = Array.isArray(data.exam_ids) ? data.exam_ids : [];
            const examSet = new Set(examIds.map(examId => String(examId)));
            filterSetsByFilename[filename] = examSet;
            return examSet;
        }}

        function ensureFilterSetLoadedAsync(filename) {{
            if (!filename || filterSetsByFilename[filename] || pendingFilterLoads.has(filename)) {{
                return;
            }}
            pendingFilterLoads.add(filename);
            ensureFilterSetLoaded(filename)
                .then(() => {{
                    unresolvedFilterWarnings.delete(filename);
                    applyFilters();
                }})
                .catch(error => {{
                    console.error('Failed to load filter set:', filename, error);
                }})
                .finally(() => {{
                    pendingFilterLoads.delete(filename);
                }});
        }}

        function appendTokenToFilterBox(token, options = {{}}) {{
            const input = document.getElementById('searchBox');
            const shouldFocus = options.focus !== false;
            const existing = input.value.trim();
            if (existing === '') {{
                input.value = token;
            }} else if (/[\s&|~(]$/.test(input.value)) {{
                input.value += token;
            }} else {{
                input.value += ' & ' + token;
            }}
            if (shouldFocus) {{
                input.focus();
            }} else {{
                input.blur();
            }}
        }}

        function replaceFilterBox(token, options = {{}}) {{
            const input = document.getElementById('searchBox');
            const shouldFocus = options.focus !== false;
            input.value = token;
            if (shouldFocus) {{
                input.focus();
            }} else {{
                input.blur();
            }}
        }}

        async function applySelectedFilterToken(options = {{}}) {{
            const select = document.getElementById('filterSelect');
            const filename = select.value;
            const appendMode = options.append === true;

            if (!filename) {{
                alert('Please select a filter from the dropdown');
                return;
            }}

            try {{
                const examSet = await ensureFilterSetLoaded(filename);
                const selectedOption = select.options[select.selectedIndex];
                const tokenStem =
                    (selectedOption && selectedOption.dataset && selectedOption.dataset.token) ||
                    filterTokenFromFilename(filename);
                const filterKind =
                    (selectedOption && selectedOption.dataset && selectedOption.dataset.kind) ||
                    'auto';
                const filterLabel =
                    (selectedOption && selectedOption.textContent) ||
                    filename.replace(/_/g, ' ');
                const token = '@' + tokenStem;
                if (appendMode) {{
                    appendTokenToFilterBox(token);
                }} else {{
                    replaceFilterBox(token, {{ focus: false }});
                }}
                applyFilters();
                if (filterKind.startsWith('annotation')) {{
                    const stats = document.getElementById('stats');
                    if (stats && examSet.size > 0) {{
                        const currentStats = stats.textContent;
                        if (!currentStats.includes('annotated universe')) {{
                            stats.textContent = currentStats + ' | annotated universe';
                        }}
                    }}
                }}
                console.log(`Loaded filter '${{filename}}' with ${{examSet.size}} exam IDs and inserted token '${{token}}'`);
            }} catch (error) {{
                alert('Failed to load filter: ' + (error.message || error));
                console.error('Filter load error:', error);
            }}
        }}

        async function insertSelectedFilterToken() {{
            await applySelectedFilterToken({{ append: true }});
        }}

        async function useSelectedFilterToken() {{
            await applySelectedFilterToken({{ append: false }});
        }}

        function clearFilterExpression() {{
            const input = document.getElementById('searchBox');
            const select = document.getElementById('filterSelect');
            input.value = '';
            select.value = '';
            applyFilters();
            input.focus();
            console.log('Cleared filter expression');
        }}

        function examMatchesTextTerm(exam, term) {{
            const q = term.toLowerCase();
            return (
                exam.patient_id.toLowerCase().includes(q) ||
                exam.exam_id.toLowerCase().includes(q) ||
                exam.accession.toLowerCase().includes(q)
            );
        }}

        function tokenizeFilterExpression(expression) {{
            const tokens = [];
            let i = 0;
            while (i < expression.length) {{
                const ch = expression[i];
                if (/\\s/.test(ch)) {{
                    i++;
                    continue;
                }}
                if (ch === '&' || ch === '|' || ch === '~' || ch === '(' || ch === ')') {{
                    tokens.push({{ type: ch }});
                    i++;
                    continue;
                }}
                if (ch === '"' || ch === "'") {{
                    const quote = ch;
                    i++;
                    const start = i;
                    while (i < expression.length && expression[i] !== quote) {{
                        i++;
                    }}
                    if (i >= expression.length) {{
                        throw new Error('Unterminated quoted term in filter expression');
                    }}
                    tokens.push({{ type: 'TERM', value: expression.slice(start, i) }});
                    i++;
                    continue;
                }}
                const start = i;
                while (
                    i < expression.length &&
                    !/\\s/.test(expression[i]) &&
                    !['&', '|', '~', '(', ')'].includes(expression[i])
                ) {{
                    i++;
                }}
                tokens.push({{ type: 'TERM', value: expression.slice(start, i) }});
            }}
            return tokens;
        }}

        function parseFilterExpression(expression) {{
            const tokens = tokenizeFilterExpression(expression);
            let index = 0;

            function peek() {{
                return tokens[index] || null;
            }}

            function consume(expectedType) {{
                const token = peek();
                if (!token || token.type !== expectedType) {{
                    throw new Error(`Expected '${{expectedType}}' in filter expression`);
                }}
                index++;
                return token;
            }}

            function parsePrimary() {{
                const token = peek();
                if (!token) {{
                    throw new Error('Unexpected end of filter expression');
                }}
                if (token.type === 'TERM') {{
                    index++;
                    return {{ type: 'TERM', value: token.value }};
                }}
                if (token.type === '(') {{
                    consume('(');
                    const expr = parseOr();
                    consume(')');
                    return expr;
                }}
                throw new Error(`Unexpected token '${{token.type}}' in filter expression`);
            }}

            function parseUnary() {{
                const token = peek();
                if (token && token.type === '~') {{
                    consume('~');
                    return {{ type: 'NOT', value: parseUnary() }};
                }}
                return parsePrimary();
            }}

            function parseAnd() {{
                let node = parseUnary();
                while (peek() && peek().type === '&') {{
                    consume('&');
                    node = {{ type: 'AND', left: node, right: parseUnary() }};
                }}
                return node;
            }}

            function parseOr() {{
                let node = parseAnd();
                while (peek() && peek().type === '|') {{
                    consume('|');
                    node = {{ type: 'OR', left: node, right: parseAnd() }};
                }}
                return node;
            }}

            const root = parseOr();
            if (index < tokens.length) {{
                throw new Error(`Unexpected token '${{tokens[index].type}}' in filter expression`);
            }}
            return root;
        }}

        function evalFilterNode(node, exam) {{
            if (node.type === 'TERM') {{
                const filename = resolveFilterFilenameFromToken(node.value);
                if (filename) {{
                    const filterSet = filterSetsByFilename[filename];
                    if (!filterSet) {{
                        ensureFilterSetLoadedAsync(filename);
                        if (!unresolvedFilterWarnings.has(filename)) {{
                            unresolvedFilterWarnings.add(filename);
                            console.warn(
                                `Filter token '${{node.value}}' maps to '${{filename}}' and is loading now.`
                            );
                        }}
                        return false;
                    }}
                    return filterSet.has(exam.exam_id);
                }}
                return examMatchesTextTerm(exam, node.value);
            }}
            if (node.type === 'NOT') {{
                return !evalFilterNode(node.value, exam);
            }}
            if (node.type === 'AND') {{
                return evalFilterNode(node.left, exam) && evalFilterNode(node.right, exam);
            }}
            if (node.type === 'OR') {{
                return evalFilterNode(node.left, exam) || evalFilterNode(node.right, exam);
            }}
            return false;
        }}

        function applyFilters() {{
            const filterExpr = document.getElementById('searchBox').value.trim();
            
            // remember current exam before filtering
            const currentExam = filteredExams.length > 0 ? filteredExams[currentIndex] : null;
            
            // annotation/auto-suggestion filters can search beyond the current batch
            const sourceExams = getFilterSourceExams(filterExpr);
            const includeAnnotatedInResults = expressionUsesExpandedUniverse(filterExpr);
            let exams = [...sourceExams];

            // apply expression filter (supports &, |, ~; plain text still works)
            if (filterExpr !== '') {{
                const hasOperators = /[&|~()]/.test(filterExpr);
                const hasFilterToken = filterExpr.includes('@');
                if (hasOperators || hasFilterToken) {{
                    try {{
                        const ast = parseFilterExpression(filterExpr);
                        exams = exams.filter(exam => evalFilterNode(ast, exam));
                    }} catch (error) {{
                        console.error('Invalid filter expression, falling back to plain text matching:', error);
                        exams = exams.filter(exam => examMatchesTextTerm(exam, filterExpr));
                    }}
                }} else {{
                    exams = exams.filter(exam => examMatchesTextTerm(exam, filterExpr));
                }}
            }}

            // skip already-reviewed exams (good / annotated / auto_excluded based on qcSkipStatus)
            exams = exams.filter(exam => {{
                return !shouldSkipExam(exam.exam_id, {{ includeAnnotated: includeAnnotatedInResults }});
            }});
            
            filteredExams = exams;
            
            // try to stay on same exam, or advance to next if current got filtered out
            if (currentExam) {{
                // check if current exam still in filtered list
                const sameExamIndex = filteredExams.findIndex(e => e.exam_id === currentExam.exam_id);
                if (sameExamIndex >= 0) {{
                    // current exam still visible, stay on it
                    currentIndex = sameExamIndex;
                }} else {{
                    // current exam filtered out, find next exam from original position
                    // look for the first exam in new filtered list that comes after the old current exam
                    const currentExamIdxInAll = sourceExams.findIndex(e => e.exam_id === currentExam.exam_id);
                    let foundNext = false;
                    for (let i = currentExamIdxInAll + 1; i < sourceExams.length; i++) {{
                        const nextExamIdx = filteredExams.findIndex(e => e.exam_id === sourceExams[i].exam_id);
                        if (nextExamIdx >= 0) {{
                            currentIndex = nextExamIdx;
                            foundNext = true;
                            break;
                        }}
                    }}
                    if (!foundNext) {{
                        // no exams after current position, go to first
                        currentIndex = 0;
                    }}
                }}
            }} else {{
                currentIndex = 0;
            }}
            
            // clamp to valid range
            if (currentIndex >= filteredExams.length) {{
                currentIndex = Math.max(0, filteredExams.length - 1);
            }}
            
            updateView();
        }}
        
        function filterGallery() {{
            applyFilters();
        }}
        
        function navigate(direction) {{
            currentIndex += direction;
            if (currentIndex < 0) currentIndex = 0;
            if (currentIndex >= filteredExams.length) currentIndex = filteredExams.length - 1;
            updateView();
        }}
        
        function updateView() {{
            const viewer = document.getElementById('viewer');
            const stats = document.getElementById('stats');
            const prevBtn = document.getElementById('prevBtn');
            const nextBtn = document.getElementById('nextBtn');
            
            if (filteredExams.length === 0) {{
                const filterExpr = document.getElementById('searchBox').value.trim();
                const hasActiveFilter = filterExpr !== '';
                const batchFullyCompleted = (
                    allExams.length > 0 &&
                    allExams.every(exam => shouldSkipExam(exam.exam_id))
                );
                const debugCounts = {{good: 0, annotated: 0, auto_excluded: 0, pending: 0}};
                allExams.forEach(exam => {{
                    const status = getStatus(exam.exam_id);
                    if (status === 'good') debugCounts.good++;
                    else if (status === 'auto_excluded') debugCounts.auto_excluded++;
                    else if (hasAnnotationFindings(exam.exam_id)) debugCounts.annotated++;
                    else debugCounts.pending++;
                }});
                const debugExamples = allExams.slice(0, 10).map(exam => {{
                    return {{ exam_id: exam.exam_id, status: getStatus(exam.exam_id) || '(pending)' }};
                }});
                console.log('QC DEBUG empty view state', {{
                    hasActiveFilter,
                    batchFullyCompleted,
                    qcSkipStatus,
                    totalAllExams: allExams.length,
                    debugCounts,
                    debugExamples
                }});
                
                if (batchFullyCompleted && !hasActiveFilter) {{
                    viewer.innerHTML = '<div style="color: #9cdcfe;">all exams in this batch are already QC\\'d</div>';
                    showCompletion();
                }} else {{
                    viewer.innerHTML = '<div style="color: #9cdcfe;">no matching exams (clear filters to see more)</div>';
                }}
                stats.textContent = '0 exams';
                prevBtn.disabled = true;
                nextBtn.disabled = true;
                updateAutoSuggestionButton();
                return;
            }}
            
            const exam = filteredExams[currentIndex];
            const currentStatus = getStatus(exam.exam_id);
            const totalRemaining = Math.max(remainingToQC - batchQCCount, 0);
            const queueRemaining = Math.max(filteredExams.length - currentIndex - 1, 0);
            const displayRemaining = totalRemaining > 0 ? totalRemaining : queueRemaining;
            
            // count QC statuses for exams in current gallery
            const qcCounts = {{good: 0, annotated: 0, pending: 0, auto_suggested: 0}};
            allExams.forEach(exam => {{
                const status = getStatus(exam.exam_id);
                if (status === 'good') qcCounts.good++;
                else if (hasAnnotationFindings(exam.exam_id)) qcCounts.annotated++;
                else qcCounts.pending++;
                if (hasAutoSuggestions(exam.exam_id)) qcCounts.auto_suggested++;
            }});
            
            const examAnnotations = getExamAnnotations(exam.exam_id);
            const hasAnnotations = examAnnotations.length > 0;
            const autoExamSuggestions = getAutoSuggestions(exam.exam_id);
            const hasAutoSuggestion = autoExamSuggestions.length > 0;
            
            let annotationPills = '';
            if (hasAnnotations) {{
                annotationPills = ' | ';
                examAnnotations.forEach(tag => {{
                    annotationPills += '<span style="display: inline-block; padding: 1px 8px; margin: 0 3px; ' +
                        'background-color: #1e1e1e; border: 1px solid #4ec9b0; border-radius: 12px; ' +
                        'font-size: 12px; color: #4ec9b0;">' + tag + '</span>';
                }});
            }}

            let autoSuggestionHtml = '';
            if (hasAutoSuggestion) {{
                autoSuggestionHtml = '<div style="margin-top: 8px; color: #e5c07b;">' +
                    '<strong>auto:</strong> ' +
                    autoExamSuggestions.map(entry => {{
                        const score = Number.isFinite(entry.score) ? ' (' + entry.score.toFixed(2) + ')' : '';
                        return '<span style="display: inline-block; padding: 1px 8px; margin: 0 3px; ' +
                            'background-color: #2a2110; border: 1px solid #e5c07b; border-radius: 12px; ' +
                            'font-size: 12px; color: #e5c07b;">' + entry.tag + score + '</span>';
                    }}).join('') +
                '</div>';
            }}
            
            viewer.innerHTML = 
                '<div class="exam-info">' +
                    '<strong>patient:</strong> ' + exam.patient_id + ' | ' +
                    '<strong>exam:</strong> ' + exam.exam_id + ' | ' +
                    '<strong>accession:</strong> ' + exam.accession +
                    annotationPills +
                    autoSuggestionHtml +
                '</div>' +
                '<div class="qc-controls">' +
                    '<button class="qc-button good' + (currentStatus === 'good' ? ' active' : '') + '" ' +
                        'onclick="setQCStatus(\\'good\\')" title="mark as good">✓ good [g]</button>' +
                    '<button class="qc-button annotate' + (hasAnnotations ? ' has-tags' : '') + '" ' +
                        'onclick="showAnnotationModal()" title="annotate this exam">🏷 annotate [a]</button>' +
                    '<button class="qc-button skip" ' +
                        'onclick="nextExam()" title="advance to next exam">⏭ next [s]</button>' +
                '</div>' +
                '<div class="image-container">' +
                    '<img src="' + exam.path + '" alt="Exam ' + exam.exam_id + '">' +
                '</div>';
            
            let statsText = (currentIndex + 1) + '/' + filteredExams.length;
            if (document.getElementById('searchBox').value.trim() !== '') {{
                statsText += ' (filtered)';
            }}
            // compute ETA using batch rate (since page load) and filtered remaining work
            const batchElapsedMin = (Date.now() - batchStartTime) / 60000;
            let rateText = '';
            if (batchQCCount > 0 && batchElapsedMin > 0.01) {{
                const rate = batchQCCount / batchElapsedMin;
                const remainingNow = displayRemaining;
                const etaMin = remainingNow / rate;
                let etaStr;
                if (etaMin < 1) etaStr = '<1 min';
                else if (etaMin < 60) etaStr = Math.round(etaMin) + ' min';
                else etaStr = (etaMin / 60).toFixed(1) + ' hr';
                rateText = ' | ' + rate.toFixed(1) + '/min, ETA ' + etaStr + ' (' + remainingNow + ' remaining)';
            }}
            
            statsText += ' | ' + displayRemaining + ' remaining | ' +
                         'qc: ' + qcCounts.good + ' good, ' + qcCounts.annotated + ' annotated, ' + 
                         qcCounts.pending + ' pending | auto: ' + qcCounts.auto_suggested + ' suggested' +
                         rateText;
            stats.textContent = statsText;
            
            prevBtn.disabled = currentIndex === 0;
            nextBtn.disabled = currentIndex === filteredExams.length - 1;
            updateAutoSuggestionButton();
        }}
        
        // keyboard navigation
        document.addEventListener('keydown', (e) => {{
            // check if annotation modal is open
            const annotationModal = document.getElementById('annotationModal');
            const modalIsOpen = annotationModal.style.display === 'block';

            // esc closes modal overlays
            if (e.key === 'Escape') {{
                if (modalIsOpen) {{
                    e.preventDefault();
                    closeAnnotationModalAndStay();
                    return;
                }}
                document.getElementById('cutflowModal').style.display = 'none';
                document.getElementById('autoReviewModal').style.display = 'none';
                document.getElementById('completionBanner').style.display = 'none';
                return;
            }}
            
            // if modal is open and user presses a tag hotkey (not in input), toggle that tag
            if (modalIsOpen && e.target.id !== 'newAnnotationInput') {{
                const key = e.key.toLowerCase();
                const taggedTag = annotationHotkeys.keyToTag[key];
                if (taggedTag) {{
                    e.preventDefault();
                    toggleAnnotation(taggedTag);
                    return;
                }}
            }}
            
            // ignore if typing in search box or annotation input
            if (e.target.id === 'searchBox' || e.target.id === 'newAnnotationInput') return;
            
            // don't process main shortcuts if modal is open
            if (modalIsOpen) return;

            const key = e.key.toLowerCase();
            
            if (e.key === 'ArrowLeft') {{
                navigate(-1);
            }} else if (e.key === 'ArrowRight') {{
                navigate(1);
            }} else if (key === 'g') {{
                setQCStatus('good');
            }} else if (key === 'a') {{
                showAnnotationModal();
            }} else if (key === 's') {{
                nextExam();
            }} else if (key === 'l') {{
                loadMore();
            }}
        }});
        
        function resetAllQC() {{
            const message =
                'This will clear ALL saved QC statuses and annotations in:\\n' +
                qcStorageNamespace +
                '\\n\\nThis includes previous sessions saved to this QC state file.\\n' +
                'Source DICOM files and generated images are NOT deleted.\\n\\n' +
                'Continue?';
            if (!confirm(message)) return;
            
            // clear JS state
            Object.keys(qcData).forEach(k => delete qcData[k]);
            Object.keys(annotations).forEach(k => delete annotations[k]);
            retainedAnnotatedExamIds = new Set();
            allExams.forEach(exam => {{ exam.qc_status = ''; }});
            updateManualQCIndicator();
            
            // clear localStorage
            localStorage.removeItem(qcStorageKey);
            localStorage.removeItem(annotationsStorageKey);
            localStorage.removeItem(annotationTagsStorageKey);
            // remove legacy keys to avoid stale state from older gallery builds
            localStorage.removeItem('qc_data');
            localStorage.removeItem('annotations');
            localStorage.removeItem('annotation_tags');
            localStorage.removeItem(sessionStatsKey);
            sessionStats = loadSessionStats();
            sessionStartTime = sessionStats.startTimeMs;
            sessionQCCount = sessionStats.qcCount;
            sessionStartRemainingToQC = sessionStats.startingRemainingToQC;
            saveSessionStats();
            updateSessionAnnotatedButton();
            
            // save empty data to server
            saveQCToServer({{ replace: true }}).catch(error => {{
                console.error('Failed to reset QC on server:', error);
            }});
            saveAnnotations();
            
            // re-render
            currentIndex = 0;
            updateView();
        }}
        
        function showCutflow() {{
            const modal = document.getElementById('cutflowModal');
            const content = document.getElementById('cutflowContent');
            
            modal.style.display = 'block';
            content.innerHTML = 'Loading cutflow data...';
            
            fetch('/cutflow')
                .then(response => response.json())
                .then(data => {{
                    if (data.error) {{
                        content.innerHTML = '<p style="color: #e06b6b;">Error: ' + data.error + '</p>';
                        return;
                    }}
                    
                    let html = '<h3>dataset overview</h3>';
                    html += '<p>total exams in dataset: <strong>' + data.total_exams.toLocaleString() + '</strong></p>';
                    html += '<p>currently loaded in gallery: <strong>' + allExams.length + '</strong></p>';
                    html += '<p>qc\\'d so far: <strong>' + getManualQCCount() + '</strong></p>';
                    html += '<hr style="border-color: #3e3e42; margin: 20px 0;">';
                    
                    html += '<h3>Automatic Filter Cutflow</h3>';
                    html += '<table class="cutflow-table"><thead><tr>';
                    html += '<th>Step</th><th>Filter</th><th>Total Flagged</th><th>New Excluded</th><th>Cumulative Excluded</th><th>Remaining</th><th>% Remaining</th>';
                    html += '</tr></thead><tbody>';
                    
                    data.cutflow.forEach(row => {{
                        html += '<tr>';
                        html += '<td>' + row.step + '</td>';
                        html += '<td>' + row.filter + '</td>';
                        html += '<td>' + (row.flagged_by_filter || 0).toLocaleString() + '</td>';
                        html += '<td>' + row.exams_excluded_this_step.toLocaleString() + '</td>';
                        html += '<td>' + row.total_excluded.toLocaleString() + '</td>';
                        html += '<td>' + row.exams_remaining.toLocaleString() + '</td>';
                        html += '<td>' + row.percent_remaining.toFixed(1) + '%</td>';
                        html += '</tr>';
                    }});
                    html += '</tbody></table>';
                    
                    if (data.filter_effectiveness) {{
                        html += '<hr style="border-color: #3e3e42; margin: 20px 0;">';
                        html += '<h3>Filter Effectiveness (Based on Manual Annotations)</h3>';
                        html += '<table class="cutflow-table"><thead><tr>';
                        html += '<th>Filter</th><th>Annotated Caught</th><th>Good Caught</th><th>Precision</th><th>Recommendation</th>';
                        html += '</tr></thead><tbody>';
                        
                        data.filter_effectiveness.forEach(f => {{
                            let rec = '';
                            if (f.precision >= 0.95) rec = '✅ AUTO-EXCLUDE';
                            else if (f.precision >= 0.8) rec = '⚠️ VALIDATE';
                            else if (f.precision >= 0.5) rec = '⚠️ MIXED';
                            else rec = '❌ DO NOT USE';
                            
                            html += '<tr>';
                            html += '<td>' + f.filter + '</td>';
                            html += '<td>' + f.annotated_caught + '</td>';
                            html += '<td style="color: ' + (f.good_caught > 0 ? '#e06b6b' : '#6bcc6b') + '">' + f.good_caught + '</td>';
                            html += '<td>' + (f.precision * 100).toFixed(1) + '%</td>';
                            html += '<td>' + rec + '</td>';
                            html += '</tr>';
                        }});
                        html += '</tbody></table>';
                        
                        html += '<p style="margin-top: 20px; color: #9cdcfe;">';
                        const totalAnnotated = data.summary.total_annotated || 0;
                        const caught = data.summary.caught_by_filters || 0;
                        const pct = totalAnnotated > 0 ? ((caught / totalAnnotated) * 100).toFixed(1) + '%' : 'n/a';
                        html += '<strong>Summary:</strong> ' + totalAnnotated + ' manually annotated exams, ';
                        html += caught + ' caught by filters ';
                        html += '(' + pct + ')';
                        html += '</p>';
                    }}
                    
                    content.innerHTML = html;
                }})
                .catch(error => {{
                    content.innerHTML = '<p style="color: #e06b6b;">Failed to load cutflow: ' + error + '</p>';
                    console.error('Cutflow error:', error);
                }});
        }}
        
        function closeCutflow() {{
            document.getElementById('cutflowModal').style.display = 'none';
        }}
        
        // close modals if clicking outside
        window.onclick = function(event) {{
            const cutflowModal = document.getElementById('cutflowModal');
            const autoReviewModal = document.getElementById('autoReviewModal');
            const completionBanner = document.getElementById('completionBanner');
            
            if (event.target === cutflowModal) {{
                cutflowModal.style.display = 'none';
            }}
            if (event.target === autoReviewModal) {{
                autoReviewModal.style.display = 'none';
            }}
            if (event.target === completionBanner) {{
                completionBanner.style.display = 'none';
            }}
        }}
        
        // initialize
        updateView();
        // show instructions in console
        console.log('QC data auto-saves to server on each button click');
        console.log('Server saves to: {qc_file_str}');
        console.log('Keyboard shortcuts: g=good, a=annotate (letter hotkeys toggle tags), s=next, l=load more, arrows=navigate');
        console.log('Backup: QC data also saved to browser localStorage (scoped by QC state file) - safe to refresh page');
        console.log('Dynamic filters: insert filter tokens from dropdown, then combine with &, |, ~ in the filter box');
        console.log('Cutflow: Click 📊 Cutflow button to see dataset statistics');
    </script>
</body>
</html>
"""

        with open(gallery_path, "w") as f:
            f.write(html_content)

        logger.info(f"HTML gallery saved to: {gallery_path}")
        report_progress(0.98, "Finalizing gallery...")

        if not serve:
            logger.info(f"Open in browser: file://{gallery_path.absolute()}")

        if qc_file:
            logger.info(f"QC state file: {qc_file.resolve()}")
            annotated_count = len(annotations_data)
            logger.info(
                f"QC status: {len([s for s in qc_data.values() if s == 'good'])} good, "
                f"{annotated_count} annotated"
            )

            if not serve:
                logger.info("=" * 60)
                logger.info(
                    "QC data saved to browser localStorage scoped by the QC state file"
                )
                logger.info(
                    "For auto-save to server, use --serve flag (recommended for remote work)"
                )
                logger.info("=" * 60)
                logger.info(
                    "Keyboard shortcuts: g=good, a=annotate (letter hotkeys), s=next, l=load more, arrow keys=navigate"
                )

    logger.info("gallery generation complete:")
    if per_view:
        logger.info(f"  individual view figures: {success_count}")
    logger.info(f"  combined 4-view figures: {total_figures} total")
    if cached_count > 0 or generated_count > 0:
        logger.info(f"    - reused from cache: {cached_count}")
        logger.info(f"    - newly generated: {generated_count}")
    logger.info(f"  errors: {error_count}")
    logger.info(f"  output directory: {output_dir}")
    report_progress(1.0, "Gallery ready.")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive QC tool for reviewing mammogram exams with persistent status tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # start QC session with HTTP server (recommended for remote work)
  python qc/qc_gallery.py --serve --max-exams 100 --random
  # then SSH port forward: ssh -L 5000:localhost:5000 user@host
  # open: http://localhost:5000/

  # prioritize worst-performing exams for QC (recommended!)
  python qc/qc_gallery.py --serve --max-exams 100 --prioritize-errors \\
    --pred-csv /path/to/validation_output.csv \\
    --meta-csv /path/to/mirai_manifest.csv

  # QC specific list of exams from filter test (streamlined!)
  python test_positioning_filters.py  # generates exam lists
  python qc/qc_gallery.py --serve --exam-list data/filter_tests/all_flagged_exams.txt

  # re-visit annotated exams
  python qc/qc_gallery.py --serve --max-exams 100 --qc-skip-status good auto_excluded

  # only show completely unmarked exams (skip all QC'd exams)
  python qc/qc_gallery.py --serve --max-exams 100 --qc-skip-status good annotated auto_excluded

  # QC without server (browser localStorage only)
  python qc/qc_gallery.py --max-exams 10 --random

  # preprocess all exams for local QC (full re-do, move output dir to local machine)
  python qc/qc_gallery.py --preprocess-all --output qc_export
  # estimate disk first:
  python qc/qc_gallery.py --estimate-only

  # use custom port
  python qc/qc_gallery.py --serve --port 8080 --patient 12345
  
  # use custom QC state file location
  python qc/qc_gallery.py --serve --qc-file /path/to/my_qc_state.json

  # load a saved model suggestion run for review / bulk acceptance
  python qc/qc_gallery.py --serve --auto-run-file /path/to/auto_qc_run.json

Server mode workflow:
  1. Run with --serve, forwards port 5000 to your local machine
  2. Open http://localhost:5000/ in browser
  3. Mark exams with g or annotate with a - auto-saves to server after each click
  4. ctrl+c to stop server when done
  5. Re-run to continue QC (by default, "good", annotated, and auto-excluded exams are skipped)
     To re-visit annotated findings: --qc-skip-status good auto_excluded

QC State File Format:
  JSON object mapping exam_id -> {"status": <string|null>, "annotations": <list>}
  Example: {"exam_id_1": {"status": "good", "annotations": []}}
        """,
    )

    parser.add_argument(
        "--views",
        type=Path,
        default=None,
        help="path to views.parquet (default: <raw>/sot/views.parquet)",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("/gpfs/data/huo-lab/Image/ChiMEC/MG"),
        help="root directory where raw DICOMs are stored (default: /gpfs/data/huo-lab/Image/ChiMEC/MG)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("qc_output"),
        help="output directory for figures and gallery (default: qc_output)",
    )
    parser.add_argument(
        "--max-exams",
        type=int,
        help="limit number of exams to review in this QC session",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="randomly sample exams (default: take first N, or worst-performing if --prioritize-errors)",
    )
    parser.add_argument(
        "--prioritize-errors",
        action="store_true",
        help="prioritize exams with worst prediction errors for QC (requires --pred-csv and --meta-csv)",
    )
    parser.add_argument(
        "--pred-csv",
        type=Path,
        help="path to validation_output.csv with predictions (for --prioritize-errors)",
    )
    parser.add_argument(
        "--meta-csv",
        type=Path,
        help="path to mirai_manifest.csv with metadata/labels (for --prioritize-errors)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="prediction horizon (years) to use for error scoring (default: 5)",
    )
    parser.add_argument(
        "--patient",
        type=str,
        help="filter to specific patient ID",
    )
    parser.add_argument(
        "--exam",
        type=str,
        help="filter to specific exam ID",
    )
    parser.add_argument(
        "--exam-list",
        type=Path,
        help="path to text file with exam IDs (one per line) to review",
    )
    parser.add_argument(
        "--per-view",
        action="store_true",
        help="also generate individual per-view debug figures",
    )
    parser.add_argument(
        "--no-gallery",
        action="store_true",
        help="skip HTML gallery generation",
    )
    parser.add_argument(
        "--qc-file",
        type=Path,
        default=Path("data/qc_state.json"),
        help="path to unified QC state file (default: data/qc_state.json)",
    )
    parser.add_argument(
        "--auto-run-file",
        type=Path,
        default=None,
        help="path to a saved auto-QC suggestion run JSON file",
    )
    parser.add_argument(
        "--qc-skip-status",
        nargs="*",
        default=["good", "annotated", "auto_excluded"],
        choices=["good", "annotated", "auto_excluded"],
        help="QC buckets to skip in future runs (default: good annotated auto_excluded). Use '--qc-skip-status good auto_excluded' to re-visit annotated findings",
    )
    parser.add_argument(
        "--preprocess-all",
        action="store_true",
        help="preprocess all exams that need QC (skip only auto_excluded). For full re-do before moving to local machine.",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="print disk estimate for preprocessing and exit",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="start HTTP server to serve gallery and handle QC saves (recommended for remote work)",
    )
    parser.add_argument(
        "--preprocessed-only",
        action="store_true",
        help="run from cached montage PNGs only; do not require raw DICOM mounts and fail if requested exams are missing cached figures",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="port for HTTP server (default: 5000; use with --serve)",
    )

    args = parser.parse_args()

    # derive views path from raw if not specified
    views_path = args.views
    if views_path is None:
        views_path = args.raw / "sot" / "views.parquet"
        logger.info(f"using default views path: {views_path}")

    # derive tags path (for cutflow)
    tags_path = args.raw / "sot" / "dicom_tags.parquet"

    # validate inputs
    if not views_path.exists():
        logger.error(f"views parquet not found: {views_path}")
        return 1

    if not args.raw.exists() and not args.preprocessed_only:
        logger.error(f"raw directory not found: {args.raw}")
        return 1
    if args.preprocessed_only and not args.raw.exists():
        logger.info(
            "running in preprocessed-only mode without raw DICOM access: %s",
            args.raw,
        )

    # validate prioritize-errors arguments
    if args.prioritize_errors:
        if args.pred_csv is None or args.meta_csv is None:
            logger.error("--prioritize-errors requires both --pred-csv and --meta-csv")
            return 1
        if not args.pred_csv.exists():
            logger.error(f"prediction CSV not found: {args.pred_csv}")
            return 1
        if not args.meta_csv.exists():
            logger.error(f"metadata CSV not found: {args.meta_csv}")
            return 1

    # resolve paths once before server changes cwd to output_dir
    resolved_raw_dir = args.raw.resolve()
    resolved_output_dir = args.output.resolve()
    resolved_qc_file = args.qc_file.resolve()
    resolved_auto_run_file = (
        args.auto_run_file.resolve() if args.auto_run_file is not None else None
    )
    resolved_views_path = views_path.resolve()
    resolved_tags_path = tags_path.resolve()
    resolved_exam_list = args.exam_list.resolve() if args.exam_list else None
    resolved_pred_csv = args.pred_csv.resolve() if args.pred_csv else None
    resolved_meta_csv = args.meta_csv.resolve() if args.meta_csv else None

    if resolved_auto_run_file is not None and not resolved_auto_run_file.exists():
        logger.error(f"auto suggestion run file not found: {resolved_auto_run_file}")
        return 1

    max_exams = args.max_exams
    if args.preprocess_all:
        qc_skip_status = {"auto_excluded"}
        max_exams = None
    else:
        qc_skip_status = set(args.qc_skip_status) if args.qc_skip_status else None

    if args.estimate_only:
        n_exams, estimated_bytes = estimate_qc_preprocess_disk(
            views_parquet=resolved_views_path,
            raw_dir=resolved_raw_dir,
            patient_id=args.patient,
            exam_id=args.exam,
            exam_list_path=resolved_exam_list,
            per_view=args.per_view,
        )
        gib = estimated_bytes / (1024**3)
        print(f"exams to preprocess: {n_exams}")
        print(f"estimated disk: {gib:.2f} GB ({estimated_bytes:,} bytes)")
        return 0

    # run gallery generation
    try:
        generate_gallery(
            views_parquet=resolved_views_path,
            raw_dir=resolved_raw_dir,
            output_dir=resolved_output_dir,
            max_exams=max_exams,
            random_sample=args.random,
            patient_id=args.patient,
            exam_id=args.exam,
            exam_list_path=resolved_exam_list,
            per_view=args.per_view,
            no_gallery=args.no_gallery,
            qc_file=resolved_qc_file,
            auto_run_file=resolved_auto_run_file,
            pred_csv=resolved_pred_csv,
            meta_csv=resolved_meta_csv,
            prioritize_errors=args.prioritize_errors,
            horizon=args.horizon,
            qc_skip_status=qc_skip_status,
            serve=args.serve,
            preprocessed_only=args.preprocessed_only,
            original_args=vars(args),
        )
    except Exception as e:
        logger.error("failed to generate gallery: %s", e)
        return 1

    # start HTTP server if requested
    if args.serve:
        logger.info("=" * 60)
        logger.info("Starting HTTP server for QC gallery...")
        # prepare args for load-more functionality
        load_more_args = {
            "views_parquet": resolved_views_path,
            "raw_dir": resolved_raw_dir,
            "output_dir": resolved_output_dir,
            "max_exams": args.max_exams,
            "random_sample": args.random,
            "patient_id": args.patient,
            "exam_id": args.exam,
            "exam_list_path": resolved_exam_list,
            "per_view": args.per_view,
            "no_gallery": args.no_gallery,
            "qc_file": resolved_qc_file,
            "auto_run_file": resolved_auto_run_file,
            "pred_csv": resolved_pred_csv,
            "meta_csv": resolved_meta_csv,
            "prioritize_errors": args.prioritize_errors,
            "horizon": args.horizon,
            "qc_skip_status": set(args.qc_skip_status) if args.qc_skip_status else None,
            "serve": True,
            "preprocessed_only": args.preprocessed_only,
            "original_args": None,
        }
        logger.info(
            "QC DEBUG server path binding: qc_file=%s, views=%s, output=%s",
            resolved_qc_file,
            resolved_views_path,
            resolved_output_dir,
        )
        start_qc_server(
            resolved_output_dir,
            resolved_qc_file,
            resolved_views_path,
            resolved_tags_path,
            resolved_auto_run_file,
            args.port,
            load_more_args,
        )
        # server runs until ctrl+c
        return 0

    return 0


if __name__ == "__main__":
    exit(main())
