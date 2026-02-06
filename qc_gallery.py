#!/usr/bin/env python3
"""qc_gallery.py

Interactive QC tool for reviewing and marking mammogram exams with persistent status tracking.

This script reads from views.parquet (output from preprocessing) and generates:
1. Combined 4-view figures (L_CC, L_MLO, R_CC, R_MLO) for each exam
2. Optional per-view debug figures with DICOM tag info
3. Interactive HTML gallery with button-based navigation, filtering, and QC marking

The gallery features:
- Single-image viewer with Previous/Next buttons (faster than scrolling)
- Keyboard navigation (left/right arrows, G/R/B for QC, A to annotate, S to skip)
- Search/filter by patient ID, exam ID, or accession number
- QC buttons to mark exams as "good", "needs review", or "bad"
- Independent annotation system: tag exams with categories (e.g., artifact types)
- QC status persistence: saves to JSON file, skips "good" exams on next run
- Real-time QC statistics in the status bar

QC Workflow (fully keyboard-driven)
------------------------------------
1. Run script with --serve flag to generate gallery and start HTTP server
2. Open gallery in browser (with SSH port forwarding if remote)
3. Mark exams using keyboard - auto-saves to server after each action:
   - G/R/B for good/review/bad (instant)
   - A to open annotation modal, 1-9 to toggle tags, type + Enter to add new tag
   - Arrow keys to navigate, S to skip
4. Re-run script to continue QC on remaining exams (by default, "good" and "bad" exams skipped)
5. To re-visit "bad" exams: use --qc-skip-status good

Data Format
-----------
QC status stored in JSON (default: data/qc_status.json):
  - {exam_id: "good"}, {exam_id: "review"}, or {exam_id: "bad"}
Annotations stored separately in data/annotations.json:
  - {exam_id: ["detector artifact - vertical line", ...]}
Available annotation tags stored in data/annotation_tags.json

Server mode (recommended for remote work):
  python qc_gallery.py --serve --max-exams 100 --random
  Then forward port: ssh -L 5000:localhost:5000 user@remote
  Open: http://localhost:5000/

Local mode (no server, manual file moving):
  python qc_gallery.py --max-exams 10
  Open qc_output/gallery.html in browser - downloads qc_status.json on each click

Usage
-----
# start QC session with HTTP server (recommended for remote work)
python qc_gallery.py --serve --max-exams 100 --random

# prioritize worst-performing exams for efficient QC (recommended!)
python qc_gallery.py --serve --max-exams 100 --prioritize-errors \
  --pred-csv /path/to/validation_output.csv \
  --meta-csv /path/to/mirai_manifest.csv

# re-visit exams previously marked as "bad"
python qc_gallery.py --serve --max-exams 100 --qc-skip-status good

# only show completely unmarked exams (skip all QC'd exams)
python qc_gallery.py --serve --max-exams 100 --qc-skip-status good bad review

# QC without server (downloads file on each click)
python qc_gallery.py --max-exams 10 --random

# QC specific patient with custom port
python qc_gallery.py --serve --port 8080 --patient 12345

# use custom QC file location
python qc_gallery.py --serve --qc-file /path/to/my_qc_status.json

Dependencies
------------
pydicom, pandas, matplotlib, tqdm
"""

import argparse
import json
import logging
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Dict, Optional, Set, Tuple
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset
from pydicom.tag import Tag
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
ANNOTATIONS_PATH = None  # exam_id -> [list of annotation tags]
ANNOTATION_TAGS_PATH = None  # list of available annotation tag strings
OUTPUT_DIR = None
FILTER_DIR = None
VIEWS_PATH = None
TAGS_PATH = None
LOAD_MORE_ARGS = None  # store args for dynamic loading


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
                with open(QC_FILE_PATH) as f:
                    qc_data = json.load(f)
                self.wfile.write(json.dumps(qc_data).encode())
            else:
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
                self.wfile.write(json.dumps(tags).encode())
            else:
                default_tags = ["detector artifact - vertical line"]
                self.wfile.write(json.dumps(default_tags).encode())
            return

        if parsed_path.path == "/load-annotations":
            # return per-exam annotations (exam_id -> [tags])
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            if ANNOTATIONS_PATH and ANNOTATIONS_PATH.exists():
                with open(ANNOTATIONS_PATH) as f:
                    annotations = json.load(f)
                self.wfile.write(json.dumps(annotations).encode())
            else:
                self.wfile.write(b"{}")
            return

        if parsed_path.path == "/list-filters":
            # list available filter files
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            filters = []
            if FILTER_DIR and FILTER_DIR.exists():
                for f in FILTER_DIR.glob("*.txt"):
                    filters.append(
                        {
                            "name": f.stem.replace("_", " ").title(),
                            "path": str(f),
                            "filename": f.name,
                        }
                    )
            self.wfile.write(json.dumps(filters).encode())
            return

        if parsed_path.path.startswith("/load-filter/"):
            # load specific filter file
            filename = parsed_path.path.split("/load-filter/", 1)[1]

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            if FILTER_DIR:
                filter_path = FILTER_DIR / filename

                if filter_path.exists():
                    with open(filter_path) as f:
                        exam_ids = [
                            line.strip()
                            for line in f
                            if line.strip() and not line.startswith("#")
                        ]
                    self.wfile.write(json.dumps({"exam_ids": exam_ids}).encode())
                else:
                    self.wfile.write(
                        json.dumps({"error": f"File not found: {filter_path}"}).encode()
                    )
            else:
                self.wfile.write(
                    json.dumps({"error": "Filter directory not configured"}).encode()
                )
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

                # double the batch size for next round
                current_batch = LOAD_MORE_ARGS.get("max_exams", 100)
                next_batch = current_batch * 2

                logger.info(
                    f"Preloading next batch: increasing from {current_batch} to {next_batch}"
                )

                # trigger gallery regeneration with larger batch in background
                import threading

                def preload_regenerate():
                    try:
                        self.server._preload_in_progress = True
                        from copy import deepcopy

                        args = deepcopy(LOAD_MORE_ARGS)
                        args["max_exams"] = next_batch
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
            # regenerate current batch with latest QC state
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

                current_batch = LOAD_MORE_ARGS.get("max_exams", 100)
                logger.info(f"Regenerating current batch: {current_batch} exams")

                import threading

                def regenerate():
                    try:
                        from copy import deepcopy

                        args = deepcopy(LOAD_MORE_ARGS)
                        generate_gallery(**args)
                        logger.info(f"Successfully regenerated {current_batch} exams")
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
                            "message": f"Regenerating {current_batch} exams...",
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

                current_batch = LOAD_MORE_ARGS.get("max_exams", 100)
                next_batch = current_batch * 2

                logger.info(
                    f"Loading more exams (fresh): increasing from {current_batch} to {next_batch}"
                )

                # trigger gallery regeneration with larger batch
                import threading

                def regenerate():
                    try:
                        from copy import deepcopy

                        args = deepcopy(LOAD_MORE_ARGS)
                        args["max_exams"] = next_batch
                        generate_gallery(**args)
                        logger.info(f"Successfully generated {next_batch} exams")
                    except Exception as e:
                        logger.error(f"Failed to generate more exams: {e}")
                        import traceback

                        traceback.print_exc()

                thread = threading.Thread(target=regenerate)
                thread.daemon = True
                thread.start()

                self.wfile.write(
                    json.dumps(
                        {
                            "status": "ok",
                            "message": f"Generating {next_batch} exams...",
                            "next_batch": next_batch,
                            "reload_delay_ms": 3000,  # wait 3s then reload
                        }
                    ).encode()
                )

            except Exception as e:
                logger.error(f"error loading more: {e}")
                import traceback

                traceback.print_exc()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
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
        if self.path == "/save-qc":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)

            try:
                qc_data = json.loads(post_data.decode())

                # save to file
                if QC_FILE_PATH:
                    QC_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with open(QC_FILE_PATH, "w") as f:
                        json.dump(qc_data, f, indent=2)

                    logger.info(f"saved QC data: {len(qc_data)} exams")

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
                logger.error(f"error saving QC data: {e}")
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        elif self.path == "/save-annotation-tags":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)

            try:
                tags = json.loads(post_data.decode())

                if ANNOTATION_TAGS_PATH:
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
        elif self.path == "/save-annotations":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)

            try:
                annotations = json.loads(post_data.decode())

                if ANNOTATIONS_PATH:
                    ANNOTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with open(ANNOTATIONS_PATH, "w") as f:
                        json.dump(annotations, f, indent=2)

                    logger.info(f"saved annotations: {len(annotations)} exams")

                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(b'{"status": "ok"}')
                else:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b'{"error": "Annotations path not configured"}')
            except Exception as e:
                logger.error(f"error saving annotations: {e}")
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def _compute_cutflow(self):
        """compute cutflow analysis on demand"""
        if not VIEWS_PATH or not TAGS_PATH:
            return {"error": "Views/tags paths not configured"}

        views_df = pd.read_parquet(VIEWS_PATH)
        tags_df = pd.read_parquet(TAGS_PATH)
        merged = views_df.merge(tags_df, on="sop_instance_uid", how="left")
        exam_df = merged.groupby("exam_id").first().reset_index()

        total_exams = len(exam_df)

        # load QC data
        qc_data = {}
        if QC_FILE_PATH and QC_FILE_PATH.exists():
            with open(QC_FILE_PATH) as f:
                qc_data = json.load(f)

        cutflow = []
        excluded_exams = set()
        filter_sets = {}

        # Starting point
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

        # Apply filters
        if "has_implant" in views_df.columns:
            implant_exams = set(views_df[views_df["has_implant"]]["exam_id"].unique())
            new_excluded = implant_exams - excluded_exams
            excluded_exams.update(implant_exams)
            filter_sets["has_implant"] = implant_exams
            cutflow.append(
                {
                    "step": "1. Implants",
                    "filter": "has_implant == True",
                    "flagged_by_filter": len(implant_exams),
                    "exams_excluded_this_step": len(new_excluded),
                    "total_excluded": len(excluded_exams),
                    "exams_remaining": total_exams - len(excluded_exams),
                    "percent_remaining": (total_exams - len(excluded_exams))
                    / total_exams
                    * 100,
                }
            )

        # scanned film: Secondary Capture OR R2 DigitalNow
        film_exams = set()
        if "SOPClassUID" in merged.columns:
            secondary = merged[
                merged["SOPClassUID"].str.contains(
                    "1.2.840.10008.5.1.4.1.1.7", na=False
                )
            ]["exam_id"].unique()
            film_exams.update(secondary)
        # R2 DigitalNow film digitizer
        r2_film = views_df[
            views_df["device_manufacturer"].str.contains(
                "R2 Technology", na=False, case=False
            )
            | views_df["device_model"].str.contains("DigitalNow", na=False, case=False)
        ]["exam_id"].unique()
        film_exams.update(r2_film)

        if len(film_exams) > 0:
            new_excluded = film_exams - excluded_exams
            excluded_exams.update(film_exams)
            filter_sets["scanned_film"] = film_exams
            cutflow.append(
                {
                    "step": "2. Scanned Film",
                    "filter": "Secondary Capture + R2 DigitalNow",
                    "flagged_by_filter": len(film_exams),
                    "exams_excluded_this_step": len(new_excluded),
                    "total_excluded": len(excluded_exams),
                    "exams_remaining": total_exams - len(excluded_exams),
                    "percent_remaining": (total_exams - len(excluded_exams))
                    / total_exams
                    * 100,
                }
            )

        if "AcquisitionDeviceProcessingCode" in merged.columns:
            gems_mask = merged["AcquisitionDeviceProcessingCode"].str.startswith(
                "GEMS_", na=False
            )
            gems_exams = set(merged[gems_mask]["exam_id"].unique())
            new_excluded = gems_exams - excluded_exams
            excluded_exams.update(gems_exams)
            filter_sets["gems_ffdm_tc1"] = gems_exams
            cutflow.append(
                {
                    "step": "3. GEMS",
                    "filter": "All GE GEMS processing codes (GEMS_*)",
                    "flagged_by_filter": len(gems_exams),
                    "exams_excluded_this_step": len(new_excluded),
                    "total_excluded": len(excluded_exams),
                    "exams_remaining": total_exams - len(excluded_exams),
                    "percent_remaining": (total_exams - len(excluded_exams))
                    / total_exams
                    * 100,
                }
            )

        # add manually marked bad exams as final filter step
        if qc_data:
            bad_exams = {eid for eid, status in qc_data.items() if status == "bad"}
            new_excluded = bad_exams - excluded_exams
            excluded_exams.update(bad_exams)
            filter_sets["manual_qc_bad"] = bad_exams
            cutflow.append(
                {
                    "step": "4. Manual QC",
                    "filter": "Manually marked as bad",
                    "flagged_by_filter": len(bad_exams),
                    "exams_excluded_this_step": len(new_excluded),
                    "total_excluded": len(excluded_exams),
                    "exams_remaining": total_exams - len(excluded_exams),
                    "percent_remaining": (total_exams - len(excluded_exams))
                    / total_exams
                    * 100,
                }
            )

        # Compute filter effectiveness if we have QC data
        filter_effectiveness = None
        summary = None

        if qc_data:
            bad_exams = {eid for eid, status in qc_data.items() if status == "bad"}
            good_exams = {eid for eid, status in qc_data.items() if status == "good"}

            effectiveness = []
            total_caught = set()

            for filter_name, filter_exams in filter_sets.items():
                caught_bad = bad_exams & filter_exams
                caught_good = good_exams & filter_exams
                total_caught.update(caught_bad)

                if len(caught_bad) > 0 or len(caught_good) > 0:
                    precision = len(caught_bad) / (len(caught_bad) + len(caught_good))
                    effectiveness.append(
                        {
                            "filter": filter_name,
                            "bad_caught": len(caught_bad),
                            "good_caught": len(caught_good),
                            "precision": precision,
                        }
                    )

            filter_effectiveness = sorted(
                effectiveness, key=lambda x: x["bad_caught"], reverse=True
            )
            summary = {
                "total_bad": len(bad_exams),
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
        ANNOTATIONS_PATH, \
        ANNOTATION_TAGS_PATH, \
        OUTPUT_DIR, \
        FILTER_DIR, \
        VIEWS_PATH, \
        TAGS_PATH, \
        LOAD_MORE_ARGS

    QC_FILE_PATH = qc_file.resolve()
    ANNOTATIONS_PATH = qc_file.parent / "annotations.json"
    ANNOTATION_TAGS_PATH = qc_file.parent / "annotation_tags.json"
    OUTPUT_DIR = output_dir.resolve()
    VIEWS_PATH = views_path.resolve()
    TAGS_PATH = tags_path.resolve()
    LOAD_MORE_ARGS = load_more_args
    # store filter directory as absolute path before changing directory
    FILTER_DIR = Path.cwd() / "data" / "filter_tests"

    # change to output directory so SimpleHTTPRequestHandler can serve files
    import os

    original_dir = os.getcwd()
    os.chdir(OUTPUT_DIR)

    server_address = ("", port)
    httpd = HTTPServer(server_address, QCGalleryHandler)

    import os
    import socket

    hostname = socket.gethostname()
    username = os.environ.get("USER", os.environ.get("USERNAME", "your_username"))

    logger.info("=" * 60)
    logger.info("QC SERVER STARTED")
    logger.info("=" * 60)
    logger.info(f"QC file: {QC_FILE_PATH}")
    logger.info(f"Serving from: {OUTPUT_DIR}")
    logger.info(f"Filter directory: {FILTER_DIR}")
    if FILTER_DIR.exists():
        filter_count = len(list(FILTER_DIR.glob("*.txt")))
        logger.info(f"  → {filter_count} filter lists available in dropdown")
    else:
        logger.info("  → No filter lists found (run test_positioning_filters.py)")
    logger.info("")
    logger.info("FOR REMOTE ACCESS:")
    logger.info("  1. In a NEW local terminal, run:")
    logger.info(f"     ssh -L {port}:localhost:{port} {username}@{hostname}")
    logger.info("  2. Open in your browser:")
    logger.info(f"     http://localhost:{port}/")
    logger.info("")
    logger.info("QC workflow:")
    logger.info("  • Mark exams with G/R/B keys or arrow keys for navigation")
    logger.info("  • Use dropdown to load filter lists (no restart needed!)")
    logger.info("  • QC data auto-saves to server on each click")
    logger.info("  • Press Ctrl+C to stop server when done")
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
    pred_csv: Optional[Path] = None,
    meta_csv: Optional[Path] = None,
    prioritize_errors: bool = False,
    horizon: int = 5,
    qc_skip_status: Optional[Set[str]] = None,
    serve: bool = False,
    original_args: Optional[dict] = None,
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
        qc_file: path to QC status file; exams with status in qc_skip_status will be skipped
        pred_csv: path to validation_output.csv with predictions (for error prioritization)
        meta_csv: path to mirai_manifest.csv with labels (for error prioritization)
        prioritize_errors: if True, sort by prediction error (worst first)
        horizon: which prediction horizon to use for error scoring (default: 5 years)
        qc_skip_status: set of QC statuses to skip (default: {"good", "bad"})
        serve: if True, suppress non-server instructions in logging
    """
    if qc_skip_status is None:
        qc_skip_status = {"good", "bad"}
    # load existing QC data if available
    qc_data = {}
    if qc_file and qc_file.exists():
        logger.info(f"loading QC data from {qc_file}")
        with open(qc_file) as f:
            qc_data = json.load(f)
        logger.info(f"loaded QC status for {len(qc_data)} exams")

    logger.info(f"loading views from {views_parquet}")
    views_df = pd.read_parquet(views_parquet)
    logger.info(
        f"loaded {len(views_df)} views from {views_df['exam_id'].nunique()} exams"
    )

    # apply auto-exclusions from config
    auto_exclude_config = Path("data/qc_auto_exclude.json")
    if auto_exclude_config.exists():
        with open(auto_exclude_config) as f:
            auto_config = json.load(f)

        auto_filters = auto_config.get("filters", [])
        if auto_filters:
            logger.info(f"applying auto-exclusions: {', '.join(auto_filters)}")

            # load tags to apply filters
            tags_df = pd.read_parquet(raw_dir / "sot" / "dicom_tags.parquet")
            merged = views_df.merge(tags_df, on="sop_instance_uid", how="left")

            auto_excluded_exams = set()
            for filter_name in auto_filters:
                if (
                    filter_name == "gems_ffdm_tc1"
                    and "AcquisitionDeviceProcessingCode" in merged.columns
                ):
                    gems_exams = merged[
                        merged["AcquisitionDeviceProcessingCode"].str.startswith(
                            "GEMS_", na=False
                        )
                    ]["exam_id"].unique()
                    auto_excluded_exams.update(gems_exams)
                    logger.info(f"  {filter_name}: {len(gems_exams)} exams")
                elif filter_name == "has_implant" and "has_implant" in views_df.columns:
                    implant_exams = views_df[views_df["has_implant"]][
                        "exam_id"
                    ].unique()
                    auto_excluded_exams.update(implant_exams)
                    logger.info(f"  {filter_name}: {len(implant_exams)} exams")
                elif filter_name == "scanned_film":
                    film_exams = set()
                    if "SOPClassUID" in merged.columns:
                        secondary = merged[
                            merged["SOPClassUID"].str.contains(
                                "1.2.840.10008.5.1.4.1.1.7", na=False
                            )
                        ]["exam_id"].unique()
                        film_exams.update(secondary)
                    # R2 DigitalNow film digitizer
                    r2_film = views_df[
                        views_df["device_manufacturer"].str.contains(
                            "R2 Technology", na=False, case=False
                        )
                        | views_df["device_model"].str.contains(
                            "DigitalNow", na=False, case=False
                        )
                    ]["exam_id"].unique()
                    film_exams.update(r2_film)
                    auto_excluded_exams.update(film_exams)
                    logger.info(f"  {filter_name}: {len(film_exams)} exams")
                elif (
                    filter_name == "negative_positioner_angle"
                    and "PositionerPrimaryAngle" in merged.columns
                ):
                    angle = pd.to_numeric(
                        merged["PositionerPrimaryAngle"], errors="coerce"
                    )
                    neg_exams = merged[angle < 0]["exam_id"].unique()
                    auto_excluded_exams.update(neg_exams)
                    logger.info(f"  {filter_name}: {len(neg_exams)} exams")
                elif (
                    filter_name == "zero_compression"
                    and "CompressionForce" in merged.columns
                ):
                    comp = pd.to_numeric(merged["CompressionForce"], errors="coerce")
                    zero_exams = merged[comp == 0]["exam_id"].unique()
                    auto_excluded_exams.update(zero_exams)
                    logger.info(f"  {filter_name}: {len(zero_exams)} exams")

            if auto_excluded_exams:
                before_count = views_df["exam_id"].nunique()
                views_df = views_df[~views_df["exam_id"].isin(auto_excluded_exams)]
                after_count = views_df["exam_id"].nunique()
                logger.info(
                    f"auto-excluded {before_count - after_count} exams total via config filters"
                )

                # mark these in QC data as auto-excluded (for tracking)
                for eid in auto_excluded_exams:
                    if eid not in qc_data:  # don't override manual QC
                        qc_data[eid] = "auto_excluded"

    # filter by patient/exam if specified
    if patient_id:
        views_df = views_df[views_df["patient_id"] == patient_id]
        logger.info(
            f"filtered to patient {patient_id}: {len(views_df)} views from {views_df['exam_id'].nunique()} exams"
        )

    if exam_id:
        views_df = views_df[views_df["exam_id"] == exam_id]
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
    done_statuses = {"good", "review", "bad"}
    done_exams = {eid for eid, status in qc_data.items() if status in done_statuses}
    done_count = len(done_exams & eligible_exam_ids)
    remaining_to_qc = max(total_to_qc - done_count, 0)

    # filter out exams with specified QC statuses
    if qc_data and qc_skip_status:
        skip_exams = {
            eid for eid, status in qc_data.items() if status in qc_skip_status
        }
        if skip_exams:
            before_count = views_df["exam_id"].nunique()
            views_df = views_df[~views_df["exam_id"].isin(skip_exams)]
            after_count = views_df["exam_id"].nunique()
            status_str = ", ".join(f"'{s}'" for s in sorted(qc_skip_status))
            logger.info(
                f"filtered out {before_count - after_count} exams marked as {status_str} in QC"
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

    # sample exams if requested
    if max_exams:
        unique_exams = views_df["exam_id"].unique()
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
        logger.info(
            f"will visualize {len(views_df)} views from {views_df['exam_id'].nunique()} exams"
        )

    # create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"output will be saved to: {output_dir}")

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

    # process each view and track by exam
    success_count = 0
    error_count = 0

    # dict to track loaded views per exam: {exam_id: {view_key: (ds, path), ...}}
    exam_view_cache = {}

    # only load DICOMs for exams that need new figures
    views_to_load = views_df[views_df["exam_id"].isin(exams_needing_generation)]

    if len(views_to_load) > 0:
        logger.info(f"loading DICOMs for {len(exams_needing_generation)} exams...")
        for idx, row in tqdm(
            views_to_load.iterrows(), total=len(views_to_load), desc="loading DICOMs"
        ):
            try:
                # construct full path to DICOM
                dicom_path = Path(row["dicom_path"])
                if not dicom_path.is_absolute():
                    dicom_path = raw_dir / dicom_path

                if not dicom_path.exists():
                    logger.warning(f"DICOM not found: {dicom_path}")
                    error_count += 1
                    continue

                # load DICOM
                ds = pydicom.dcmread(str(dicom_path))

                # generate individual debug figure only if requested
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

                # cache view for combined figure
                exam_key = row["exam_id"]
                if exam_key not in exam_view_cache:
                    exam_view_cache[exam_key] = {
                        "patient_id": row["patient_id"],
                        "accession_number": row.get("accession_number", "unknown"),
                        "views": {},
                    }

                view_key = f"{row['laterality']}_{row['view']}"
                exam_view_cache[exam_key]["views"][view_key] = (ds, dicom_path)

            except Exception as e:
                logger.error(
                    f"error processing {row.get('dicom_path', 'unknown')}: {e}"
                )
                error_count += 1
    else:
        logger.info("all figures cached, skipping DICOM loading")

    def get_qc_status(exam_id):
        """extract status string from qc_data"""
        return qc_data.get(exam_id, "")

    # generate combined 4-view figures for each exam (only those needing generation)
    logger.info("processing figures...")
    combined_images = []  # track for HTML gallery
    cached_count = len(cached_exams)
    generated_count = 0

    # first, add cached exams to gallery list
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

    # then, generate new figures for exams that need them
    for exam_key, exam_data in tqdm(exam_view_cache.items(), desc="generating figures"):
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
                    # shouldn't happen since we pre-filtered, but handle it
                    cached_count += 1
                else:
                    generated_count += 1

                # track for gallery only if save was successful
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
            logger.error(f"error creating combined figure for exam {exam_key}: {e}")

    # compute total figures
    total_figures = cached_count + generated_count

    # generate HTML gallery
    if not no_gallery and combined_images:
        logger.info("generating HTML gallery...")
        gallery_path = output_dir / "gallery.html"

        # prepare QC file path for saving
        qc_file_str = str(qc_file.resolve()) if qc_file else "data/qc_status.json"

        # construct load more command
        load_more_cmd = "python qc_gallery.py --serve"

        if exam_list_path:
            # if loaded from exam list, suggest switching to prioritize-errors
            if pred_csv and meta_csv:
                next_batch = max_exams if max_exams else 100
                load_more_cmd += f" --max-exams {next_batch} --prioritize-errors"
                load_more_cmd += f" --pred-csv {pred_csv} --meta-csv {meta_csv}"
            else:
                load_more_cmd += f" --max-exams {max_exams if max_exams else 100}"
        else:
            # if doing prioritize-errors, increase batch size
            if max_exams:
                load_more_cmd += f" --max-exams {max_exams * 2}"
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
            font-size: 14px;
            border: 1px solid #3e3e42;
            border-radius: 4px;
            background-color: #1e1e1e;
            color: #d4d4d4;
            width: 250px;
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
        .qc-button.review {{
            background-color: #3a3a1e;
            color: #f0d060;
            border-color: #8a8a4a;
        }}
        .qc-button.review:hover {{
            background-color: #5a5a2d;
            border-color: #f0d060;
        }}
        .qc-button.review.active {{
            background-color: #8a8a4a;
            border-color: #f0e080;
            box-shadow: 0 0 10px #f0d060;
        }}
        .qc-button.bad {{
            background-color: #3a1e1e;
            color: #e06b6b;
            border-color: #8a4a4a;
        }}
        .qc-button.bad:hover {{
            background-color: #5a2d2d;
            border-color: #e06b6b;
        }}
        .qc-button.bad.active {{
            background-color: #8a4a4a;
            border-color: #e08f8f;
            box-shadow: 0 0 10px #e06b6b;
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
            padding: 6px 12px;
            font-size: 13px;
            border: 1px solid #3e3e42;
            border-radius: 4px;
            background-color: #252526;
            color: #9cdcfe;
            cursor: pointer;
            transition: all 0.2s;
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
    </style>
</head>
<body>
    <div class="container">
        <div class="controls">
            <div class="stats" id="stats">
                {total_figures} exams
            </div>
            <div style="display: flex; gap: 5px;">
                <button class="info-button" onclick="showCutflow()">📊 Cutflow</button>
                <button class="info-button" onclick="resetAllQC()" title="Clear all QC data and annotations">🗑 Reset</button>
                <button class="info-button" onclick="forceRegenerate()" title="Rebuild current batch with latest QC state">🔁 Force Regenerate</button>
            </div>
            <div class="filter-controls">
                <input type="text" id="searchBox" placeholder="Filter by patient ID, exam ID, or accession..." 
                       onkeyup="filterGallery()">
                <select id="filterSelect" onfocus="loadAvailableFilters()">
                    <option value="">Load filter list...</option>
                </select>
                <button onclick="loadSelectedFilter()">Apply Filter</button>
                <button onclick="clearExamFilter()">Clear Filter</button>
                <label style="display: flex; align-items: center; gap: 5px; margin-left: 20px; cursor: pointer;">
                    <input type="checkbox" id="hideReviewCheckbox" onchange="applyFilters()" 
                           style="cursor: pointer; width: 16px; height: 16px;">
                    <span style="font-size: 13px;">Hide "review" exams</span>
                </label>
            </div>
            <div class="nav-buttons">
                <button id="prevBtn" onclick="navigate(-1)">← Previous</button>
                <button id="nextBtn" onclick="navigate(1)">Next →</button>
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
    
    <!-- Annotation Modal -->
    <div id="annotationModal" class="modal">
        <div class="modal-content" style="max-width: 600px;">
            <div class="modal-header">
                <h2>annotate exam</h2>
                <span class="close" onclick="closeAnnotationModal()">&times;</span>
            </div>
            <div>
                <p style="color: #858585; font-size: 13px; margin-top: 0; margin-bottom: 15px;">
                    Press 1-9 to toggle tags, or type new tag and press Enter. Press ESC when done.
                </p>
                <div id="annotationTagsList" style="margin-bottom: 20px;">
                </div>
                <div style="border-top: 1px solid #3e3e42; padding-top: 20px;">
                    <div style="display: flex; gap: 10px; align-items: center;">
                        <input type="text" id="newAnnotationInput" placeholder="add new tag and press Enter..." 
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
        <div style="margin: 15px 0;">
            <button id="loadMoreBtn" onclick="loadMore()">🔄 Load More Exams</button>
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

        html_content += f"""
        ];

        const qcSkipStatus = {json.dumps(qc_skip_status_list)};
        const totalToQC = {total_to_qc};
        const remainingToQC = {remaining_to_qc};
        
        let filteredExams = [...allExams];
        let currentIndex = 0;
        let activeExamFilter = null; // track active exam ID filter
        let preloadTriggered = false; // track if we've started preloading next batch
        
        // session rate tracking
        const sessionStartTime = Date.now();
        let sessionQCCount = 0;
        
        // track QC decisions (exam_id -> status)
        let qcData = {{}};
        
        // load from localStorage first (preserves work across page refreshes)
        const savedQCData = localStorage.getItem('qc_data');
        if (savedQCData) {{
            try {{
                qcData = JSON.parse(savedQCData);
            }} catch (e) {{
                console.error('Failed to parse saved QC data:', e);
            }}
        }}
        
        // merge with existing statuses from server
        allExams.forEach(exam => {{
            if (exam.qc_status && !qcData[exam.exam_id]) {{
                qcData[exam.exam_id] = exam.qc_status;
            }}
        }});

        // client-side skip of already QC'd exams (matches qcSkipStatus)
        if (qcSkipStatus.length > 0) {{
            filteredExams = filteredExams.filter(exam => {{
                return !qcSkipStatus.includes(qcData[exam.exam_id]);
            }});
        }}
        
        // annotation system: tags are independent of QC status
        let annotationTags = [];  // available tag strings
        let annotations = {{}};    // exam_id -> [tag1, tag2, ...]
        
        // load annotations from localStorage first (survives page refresh)
        const savedAnnotations = localStorage.getItem('annotations');
        if (savedAnnotations) {{
            try {{
                annotations = JSON.parse(savedAnnotations);
            }} catch (e) {{
                console.error('failed to parse saved annotations:', e);
            }}
        }}
        const savedAnnotationTags = localStorage.getItem('annotation_tags');
        if (savedAnnotationTags) {{
            try {{
                annotationTags = JSON.parse(savedAnnotationTags);
            }} catch (e) {{
                console.error('failed to parse saved annotation tags:', e);
            }}
        }}
        
        // then merge with server data (server wins for tags, merge for annotations)
        fetch('/load-annotation-tags')
            .then(response => response.json())
            .then(tags => {{
                if (tags.length > 0) {{
                    // merge: keep any localStorage tags not on server, add server tags
                    const merged = [...new Set([...annotationTags, ...tags])];
                    annotationTags = merged;
                }}
                console.log('loaded annotation tags:', annotationTags);
            }})
            .catch(error => {{
                console.error('failed to load annotation tags from server:', error);
                if (annotationTags.length === 0) {{
                    annotationTags = ["detector artifact - vertical line"];
                }}
            }});
        
        fetch('/load-annotations')
            .then(response => response.json())
            .then(data => {{
                // merge server annotations with localStorage
                Object.keys(data).forEach(examId => {{
                    annotations[examId] = data[examId];
                }});
                console.log('loaded annotations for', Object.keys(annotations).length, 'exams');
                updateView();
            }})
            .catch(error => {{
                console.error('failed to load annotations from server:', error);
            }});
        
        function saveAnnotationTags() {{
            localStorage.setItem('annotation_tags', JSON.stringify(annotationTags));
            fetch('/save-annotation-tags', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify(annotationTags)
            }}).catch(error => console.error('failed to save annotation tags:', error));
        }}
        
        function saveAnnotations() {{
            localStorage.setItem('annotations', JSON.stringify(annotations));
            fetch('/save-annotations', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify(annotations)
            }}).catch(error => console.error('failed to save annotations:', error));
        }}
        
        function checkAndPreload() {{
            // check if we should start preloading next batch
            if (preloadTriggered) return;
            
            // count how many exams in current batch have been QC'd
            let qcdCount = 0;
            allExams.forEach(exam => {{
                const status = getStatus(exam.exam_id);
                if (status && status !== 'auto_excluded') {{
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
        
        function autoSaveQCData() {{
            // save to localStorage as backup
            localStorage.setItem('qc_data', JSON.stringify(qcData));
            
            // check if we should preload next batch
            checkAndPreload();
            
            // try to save to server via POST
            fetch('/save-qc', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                }},
                body: JSON.stringify(qcData)
            }})
            .then(response => response.json())
            .then(data => {{
                console.log('QC data saved to server:', data);
            }})
            .catch(error => {{
                console.error('Failed to save to server:', error);
            }});
        }}
        
        function setQCStatus(status) {{
            const exam = filteredExams[currentIndex];
            qcData[exam.exam_id] = status;
            exam.qc_status = status;
            sessionQCCount++;
            autoSaveQCData();
            
            // check if this was the last exam
            if (currentIndex === filteredExams.length - 1) {{
                // show completion banner
                showCompletion();
            }} else {{
                // auto-advance to next exam
                navigate(1);
            }}
        }}
        
        function showAnnotationModal() {{
            const modal = document.getElementById('annotationModal');
            const tagsList = document.getElementById('annotationTagsList');
            const exam = filteredExams[currentIndex];
            const examAnnotations = annotations[exam.exam_id] || [];
            
            // clear and populate tag list
            tagsList.innerHTML = '';
            annotationTags.forEach((tag, index) => {{
                const hotkey = index + 1; // 1-9
                if (hotkey > 9) return;
                
                const isActive = examAnnotations.includes(tag);
                const btn = document.createElement('button');
                btn.textContent = `[${{hotkey}}] ${{tag}}`;
                btn.dataset.tag = tag;
                btn.style.cssText = `
                    display: block;
                    width: 100%;
                    padding: 12px 16px;
                    margin-bottom: 10px;
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
                    btn.textContent = `[${{hotkey}}] ✓ ${{tag}}`;
                }}
                btn.onclick = () => {{
                    toggleAnnotation(tag);
                }};
                tagsList.appendChild(btn);
            }});
            
            modal.style.display = 'block';
        }}
        
        function closeAnnotationModal() {{
            const modal = document.getElementById('annotationModal');
            const input = document.getElementById('newAnnotationInput');
            modal.style.display = 'none';
            input.value = '';
        }}
        
        function toggleAnnotation(tag) {{
            const exam = filteredExams[currentIndex];
            if (!annotations[exam.exam_id]) {{
                annotations[exam.exam_id] = [];
            }}
            
            const idx = annotations[exam.exam_id].indexOf(tag);
            if (idx >= 0) {{
                annotations[exam.exam_id].splice(idx, 1);
                if (annotations[exam.exam_id].length === 0) {{
                    delete annotations[exam.exam_id];
                }}
            }} else {{
                annotations[exam.exam_id].push(tag);
            }}
            
            saveAnnotations();
            closeAnnotationModal();
            updateView();
        }}
        
        function addNewAnnotationTag() {{
            const input = document.getElementById('newAnnotationInput');
            const newTag = input.value.trim();
            
            if (newTag === '') return;
            
            if (annotationTags.includes(newTag)) {{
                alert('this tag already exists');
                return;
            }}
            
            // add tag to available list and persist
            annotationTags.push(newTag);
            saveAnnotationTags();
            
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
        
        function getStatus(examId) {{
            return qcData[examId] || '';
        }}
        
        function skipExam() {{
            // skip without marking - just advance
            if (currentIndex === filteredExams.length - 1) {{
                showCompletion();
            }} else {{
                navigate(1);
            }}
        }}
        
        function showCompletion() {{
            const banner = document.getElementById('completionBanner');
            const statsDiv = document.getElementById('completionStats');
            
            // count QC statuses
            const qcCounts = {{good: 0, review: 0, bad: 0, pending: 0}};
            allExams.forEach(exam => {{
                const status = getStatus(exam.exam_id);
                if (status === 'good') qcCounts.good++;
                else if (status === 'review') qcCounts.review++;
                else if (status === 'bad') qcCounts.bad++;
                else qcCounts.pending++;
            }});
            
            statsDiv.innerHTML = 
                '<div><strong>Total exams in batch:</strong> ' + allExams.length + '</div>' +
                '<div style="color: #6bcc6b;"><strong>Good:</strong> ' + qcCounts.good + '</div>' +
                '<div style="color: #e06b6b;"><strong>Bad:</strong> ' + qcCounts.bad + '</div>' +
                '<div style="color: #f0d060;"><strong>Review:</strong> ' + qcCounts.review + '</div>' +
                '<div style="color: #9cdcfe;"><strong>Pending:</strong> ' + qcCounts.pending + '</div>';
            
            banner.style.display = 'block';
        }}
        
        function closeCompletion() {{
            document.getElementById('completionBanner').style.display = 'none';
        }}

        function forceRegenerate() {{
            fetch('/regenerate')
                .then(response => response.json())
                .then(data => {{
                    if (data.error) {{
                        alert('Error regenerating exams: ' + data.error);
                        return;
                    }}
                    console.log('Force regenerate triggered:', data);
                    setTimeout(() => {{
                        location.reload();
                    }}, data.reload_delay_ms || 3000);
                }})
                .catch(error => {{
                    console.error('Failed to regenerate:', error);
                    alert('Failed to regenerate: ' + error);
                }});
        }}
        
        function loadMore() {{
            const loadMoreBtn = document.getElementById('loadMoreBtn');
            const statusDiv = document.getElementById('loadMoreStatus');
            
            // disable button and show status
            loadMoreBtn.disabled = true;
            loadMoreBtn.textContent = '⏳ Loading...';
            statusDiv.style.display = 'block';
            
            fetch('/load-more')
                .then(response => response.json())
                .then(data => {{
                    if (data.error) {{
                        alert('Error loading more exams: ' + data.error);
                        loadMoreBtn.disabled = false;
                        loadMoreBtn.textContent = '🔄 Load More Exams';
                        statusDiv.style.display = 'none';
                        return;
                    }}
                    
                    console.log('Load more triggered:', data);
                    
                    // update status message based on whether preload was ready
                    if (data.reload_delay_ms < 1000) {{
                        statusDiv.innerHTML = '<p style="color: #4ec9b0; font-weight: bold; margin: 0;">✓ Next batch ready! (preloaded)</p>' +
                            '<p style="font-size: 12px; color: #858585; margin: 10px 0 0 0;">Reloading now...</p>';
                    }} else {{
                        statusDiv.innerHTML = '<p style="color: #4ec9b0; font-weight: bold; margin: 0;">⏳ Generating more exams...</p>' +
                            '<p style="font-size: 12px; color: #858585; margin: 10px 0 0 0;">This may take a minute. Page will reload automatically.</p>';
                    }}
                    
                    // wait for server to generate, then reload
                    setTimeout(() => {{
                        location.reload();
                    }}, data.reload_delay_ms || 3000);
                }})
                .catch(error => {{
                    console.error('Failed to load more:', error);
                    alert('Failed to load more exams: ' + error);
                    loadMoreBtn.disabled = false;
                    loadMoreBtn.textContent = '🔄 Load More Exams';
                    statusDiv.style.display = 'none';
                }});
        }}
        
        function loadAvailableFilters() {{
            fetch('/list-filters')
                .then(response => response.json())
                .then(filters => {{
                    const select = document.getElementById('filterSelect');
                    // keep the default option
                    select.innerHTML = '<option value="">Load filter list...</option>';
                    filters.forEach(filter => {{
                        const option = document.createElement('option');
                        option.value = filter.filename;
                        option.textContent = filter.name;
                        select.appendChild(option);
                    }});
                    console.log(`Loaded ${{filters.length}} filter options`);
                }})
                .catch(error => {{
                    console.error('Failed to load filter list:', error);
                }});
        }}
        
        function loadSelectedFilter() {{
            const select = document.getElementById('filterSelect');
            const filename = select.value;
            
            if (!filename) {{
                alert('Please select a filter from the dropdown');
                return;
            }}
            
            fetch('/load-filter/' + filename)
                .then(response => response.json())
                .then(data => {{
                    if (data.error) {{
                        alert('Error loading filter: ' + data.error);
                        return;
                    }}
                    
                    const examIds = new Set(data.exam_ids);
                    activeExamFilter = examIds;
                    applyFilters();
                    console.log(`Loaded filter with ${{examIds.size}} exam IDs`);
                }})
                .catch(error => {{
                    alert('Failed to load filter: ' + error);
                    console.error('Filter load error:', error);
                }});
        }}
        
        function clearExamFilter() {{
            activeExamFilter = null;
            document.getElementById('filterSelect').value = '';
            applyFilters();
            console.log('Cleared exam ID filter');
        }}
        
        function applyFilters() {{
            const searchTerm = document.getElementById('searchBox').value.toLowerCase();
            const hideReview = document.getElementById('hideReviewCheckbox').checked;
            
            // remember current exam before filtering
            const currentExam = filteredExams.length > 0 ? filteredExams[currentIndex] : null;
            
            // start with all exams
            let exams = [...allExams];
            
            // apply exam ID filter if active
            if (activeExamFilter) {{
                exams = exams.filter(exam => activeExamFilter.has(exam.exam_id));
            }}
            
            // apply text search filter
            if (searchTerm !== '') {{
                exams = exams.filter(exam => 
                    exam.patient_id.toLowerCase().includes(searchTerm) ||
                    exam.exam_id.toLowerCase().includes(searchTerm) ||
                    exam.accession.toLowerCase().includes(searchTerm)
                );
            }}
            
            // hide review exams if checkbox is checked
            if (hideReview) {{
                exams = exams.filter(exam => {{
                    const status = getStatus(exam.exam_id);
                    return status !== 'review';
                }});
            }}

            // skip already QC'd exams (matches qcSkipStatus)
            if (qcSkipStatus.length > 0) {{
                exams = exams.filter(exam => {{
                    return !qcSkipStatus.includes(getStatus(exam.exam_id));
                }});
            }}
            
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
                    const currentExamIdxInAll = allExams.findIndex(e => e.exam_id === currentExam.exam_id);
                    let foundNext = false;
                    for (let i = currentExamIdxInAll + 1; i < allExams.length; i++) {{
                        const nextExamIdx = filteredExams.findIndex(e => e.exam_id === allExams[i].exam_id);
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
                viewer.innerHTML = '<div style="color: #9cdcfe;">no matching exams</div>';
                stats.textContent = '0 exams';
                prevBtn.disabled = true;
                nextBtn.disabled = true;
                if (allExams.length > 0 && qcSkipStatus.length > 0) {{
                    showCompletion();
                }}
                return;
            }}
            
            const exam = filteredExams[currentIndex];
            const currentStatus = getStatus(exam.exam_id);
            
            // count QC statuses for exams in current gallery
            const qcCounts = {{good: 0, review: 0, bad: 0, pending: 0}};
            allExams.forEach(exam => {{
                const status = getStatus(exam.exam_id);
                if (status === 'good') qcCounts.good++;
                else if (status === 'review') qcCounts.review++;
                else if (status === 'bad') qcCounts.bad++;
                else qcCounts.pending++;
            }});
            
            const examAnnotations = annotations[exam.exam_id] || [];
            const hasAnnotations = examAnnotations.length > 0;
            
            let annotationPills = '';
            if (hasAnnotations) {{
                annotationPills = ' | ';
                examAnnotations.forEach(tag => {{
                    annotationPills += '<span style="display: inline-block; padding: 1px 8px; margin: 0 3px; ' +
                        'background-color: #1e1e1e; border: 1px solid #4ec9b0; border-radius: 12px; ' +
                        'font-size: 12px; color: #4ec9b0;">' + tag + '</span>';
                }});
            }}
            
            viewer.innerHTML = 
                '<div class="exam-info">' +
                    '<strong>Patient:</strong> ' + exam.patient_id + ' | ' +
                    '<strong>Exam:</strong> ' + exam.exam_id + ' | ' +
                    '<strong>Accession:</strong> ' + exam.accession +
                    annotationPills +
                '</div>' +
                '<div class="qc-controls">' +
                    '<button class="qc-button good' + (currentStatus === 'good' ? ' active' : '') + '" ' +
                        'onclick="setQCStatus(\\'good\\')" title="Mark as good">✓ Good [g]</button>' +
                    '<button class="qc-button review' + (currentStatus === 'review' ? ' active' : '') + '" ' +
                        'onclick="setQCStatus(\\'review\\')" title="Needs review">? Review [r]</button>' +
                    '<button class="qc-button bad' + (currentStatus === 'bad' ? ' active' : '') + '" ' +
                        'onclick="setQCStatus(\\'bad\\')" title="Mark as bad">✗ Bad [b]</button>' +
                    '<button class="qc-button annotate' + (hasAnnotations ? ' has-tags' : '') + '" ' +
                        'onclick="showAnnotationModal()" title="Annotate this exam">🏷 Annotate [a]</button>' +
                    '<button class="qc-button skip" ' +
                        'onclick="skipExam()" title="Skip without marking">⏭ Skip [s]</button>' +
                '</div>' +
                '<div class="image-container">' +
                    '<img src="' + exam.path + '" alt="Exam ' + exam.exam_id + '">' +
                '</div>';
            
            let statsText = (currentIndex + 1) + '/' + filteredExams.length;
            if (activeExamFilter) {{
                statsText += ' (filtered)';
            }}
            const hideReview = document.getElementById('hideReviewCheckbox').checked;
            if (hideReview && qcCounts.review > 0) {{
                statsText += ' (' + qcCounts.review + ' review hidden)';
            }}
            // compute session rate and ETA
            const elapsedMin = (Date.now() - sessionStartTime) / 60000;
            let rateText = '';
            if (sessionQCCount > 0 && elapsedMin > 0.01) {{
                const rate = sessionQCCount / elapsedMin;
                const remainingNow = Math.max(remainingToQC - sessionQCCount, 0);
                const etaMin = remainingNow / rate;
                let etaStr;
                if (etaMin < 1) etaStr = '<1 min';
                else if (etaMin < 60) etaStr = Math.round(etaMin) + ' min';
                else etaStr = (etaMin / 60).toFixed(1) + ' hr';
                rateText = ' | ' + rate.toFixed(1) + '/min, ETA ' + etaStr + ' (' + sessionQCCount + ' this session)';
            }}
            
            statsText += ' | ' + totalToQC + ' total | ' +
                         'QC: ' + qcCounts.good + ' good, ' + qcCounts.review + ' review, ' + 
                         qcCounts.bad + ' bad, ' + qcCounts.pending + ' pending' + rateText;
            stats.textContent = statsText;
            
            prevBtn.disabled = currentIndex === 0;
            nextBtn.disabled = currentIndex === filteredExams.length - 1;
        }}
        
        // keyboard navigation
        document.addEventListener('keydown', (e) => {{
            // ESC to close modals
            if (e.key === 'Escape') {{
                document.getElementById('cutflowModal').style.display = 'none';
                document.getElementById('completionBanner').style.display = 'none';
                closeAnnotationModal();
                return;
            }}
            
            // check if annotation modal is open
            const annotationModal = document.getElementById('annotationModal');
            const modalIsOpen = annotationModal.style.display === 'block';
            
            // if modal is open and user presses a number (not in input), toggle that tag
            if (modalIsOpen && e.target.id !== 'newAnnotationInput') {{
                const num = parseInt(e.key);
                if (num >= 1 && num <= 9 && num <= annotationTags.length) {{
                    toggleAnnotation(annotationTags[num - 1]);
                    return;
                }}
            }}
            
            // ignore if typing in search box or annotation input
            if (e.target.id === 'searchBox' || e.target.id === 'newAnnotationInput') return;
            
            // don't process main shortcuts if modal is open
            if (modalIsOpen) return;
            
            if (e.key === 'ArrowLeft') {{
                navigate(-1);
            }} else if (e.key === 'ArrowRight') {{
                navigate(1);
            }} else if (e.key === 'g') {{
                setQCStatus('good');
            }} else if (e.key === 'r') {{
                setQCStatus('review');
            }} else if (e.key === 'b') {{
                setQCStatus('bad');
            }} else if (e.key === 'a') {{
                showAnnotationModal();
            }} else if (e.key === 's') {{
                skipExam();
            }}
        }});
        
        function resetAllQC() {{
            if (!confirm('Clear ALL QC statuses and annotations? This cannot be undone.')) return;
            
            // clear JS state
            Object.keys(qcData).forEach(k => delete qcData[k]);
            Object.keys(annotations).forEach(k => delete annotations[k]);
            allExams.forEach(exam => {{ exam.qc_status = ''; }});
            
            // clear localStorage
            localStorage.removeItem('qc_data');
            localStorage.removeItem('annotations');
            localStorage.removeItem('annotation_tags');
            
            // save empty data to server
            autoSaveQCData();
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
                    
                    let html = '<h3>Dataset Overview</h3>';
                    html += '<p>Total exams in dataset: <strong>' + data.total_exams.toLocaleString() + '</strong></p>';
                    html += '<p>Currently loaded in gallery: <strong>' + allExams.length + '</strong></p>';
                    html += '<p>Manually QC\\'d so far: <strong>' + Object.keys(qcData).length + '</strong></p>';
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
                        html += '<h3>Filter Effectiveness (Based on Manual QC)</h3>';
                        html += '<table class="cutflow-table"><thead><tr>';
                        html += '<th>Filter</th><th>Bad Caught</th><th>Good Caught</th><th>Precision</th><th>Recommendation</th>';
                        html += '</tr></thead><tbody>';
                        
                        data.filter_effectiveness.forEach(f => {{
                            let rec = '';
                            if (f.precision >= 0.95) rec = '✅ AUTO-EXCLUDE';
                            else if (f.precision >= 0.8) rec = '⚠️ VALIDATE';
                            else if (f.precision >= 0.5) rec = '⚠️ MIXED';
                            else rec = '❌ DO NOT USE';
                            
                            html += '<tr>';
                            html += '<td>' + f.filter + '</td>';
                            html += '<td>' + f.bad_caught + '</td>';
                            html += '<td style="color: ' + (f.good_caught > 0 ? '#e06b6b' : '#6bcc6b') + '">' + f.good_caught + '</td>';
                            html += '<td>' + (f.precision * 100).toFixed(1) + '%</td>';
                            html += '<td>' + rec + '</td>';
                            html += '</tr>';
                        }});
                        html += '</tbody></table>';
                        
                        html += '<p style="margin-top: 20px; color: #9cdcfe;">';
                        html += '<strong>Summary:</strong> ' + data.summary.total_bad + ' manually marked bad, ';
                        html += data.summary.caught_by_filters + ' caught by filters ';
                        html += '(' + (data.summary.caught_by_filters / data.summary.total_bad * 100).toFixed(1) + '%)';
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
            const completionBanner = document.getElementById('completionBanner');
            
            if (event.target === cutflowModal) {{
                cutflowModal.style.display = 'none';
            }}
            if (event.target === completionBanner) {{
                completionBanner.style.display = 'none';
            }}
        }}
        
        // initialize
        updateView();
        loadAvailableFilters();
        
        // show instructions in console
        console.log('QC data auto-saves to server on each button click');
        console.log('Server saves to: {qc_file_str}');
        console.log('Keyboard shortcuts: g=good, r=review, b=bad, a=annotate (1-9 to toggle), arrows=navigate');
        console.log('Backup: QC data also saved to browser localStorage - safe to refresh page');
        console.log('Dynamic filters: Use dropdown to load filter lists without restarting server');
        console.log('Cutflow: Click 📊 Cutflow button to see dataset statistics');
    </script>
</body>
</html>
"""

        with open(gallery_path, "w") as f:
            f.write(html_content)

        logger.info(f"HTML gallery saved to: {gallery_path}")

        if not serve:
            logger.info(f"Open in browser: file://{gallery_path.absolute()}")

        if qc_file:
            logger.info(f"QC file: {qc_file.resolve()}")
            logger.info(
                f"QC status: {len([s for s in qc_data.values() if s == 'good'])} good, "
                f"{len([s for s in qc_data.values() if s == 'review'])} review, "
                f"{len([s for s in qc_data.values() if s == 'bad'])} bad"
            )

            if not serve:
                logger.info("=" * 60)
                logger.info(
                    "QC data saved to localStorage + downloads folder on each click"
                )
                logger.info(f"Move downloaded qc_status.json to: {qc_file.resolve()}")
                logger.info(
                    "For auto-save to server, use --serve flag (recommended for remote work)"
                )
                logger.info("=" * 60)
                logger.info(
                    "Keyboard shortcuts: G=Good, R=Review, B=Bad, A=Annotate (1-9 toggle), Arrow keys=Navigate"
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


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive QC tool for reviewing mammogram exams with persistent status tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # start QC session with HTTP server (recommended for remote work)
  python qc_gallery.py --serve --max-exams 100 --random
  # then SSH port forward: ssh -L 5000:localhost:5000 user@host
  # open: http://localhost:5000/

  # prioritize worst-performing exams for QC (recommended!)
  python qc_gallery.py --serve --max-exams 100 --prioritize-errors \\
    --pred-csv /path/to/validation_output.csv \\
    --meta-csv /path/to/mirai_manifest.csv

  # QC specific list of exams from filter test (streamlined!)
  python test_positioning_filters.py  # generates exam lists
  python qc_gallery.py --serve --exam-list data/filter_tests/all_flagged_exams.txt

  # re-visit exams previously marked as "bad"
  python qc_gallery.py --serve --max-exams 100 --qc-skip-status good

  # only show completely unmarked exams (skip all QC'd exams)
  python qc_gallery.py --serve --max-exams 100 --qc-skip-status good bad review

  # QC without server (downloads qc_status.json on each click)
  python qc_gallery.py --max-exams 10 --random

  # use custom port
  python qc_gallery.py --serve --port 8080 --patient 12345
  
  # use custom QC file location
  python qc_gallery.py --serve --qc-file /path/to/my_qc_status.json

Server mode workflow:
  1. Run with --serve, forwards port 5000 to your local machine
  2. Open http://localhost:5000/ in browser
  3. Mark exams with G/R/B keys - auto-saves to server after each click
  4. Ctrl+C to stop server when done
  5. Re-run to continue QC (by default, "good" and "bad" exams skipped)
     To re-visit bad exams: --qc-skip-status good

QC File Format:
  JSON file mapping exam_id to status: {"exam_id_1": "good", "exam_id_2": "review", ...}
  Valid statuses: "good", "review", "bad"
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
        default=Path("data/qc_status.json"),
        help="path to QC status file (default: data/qc_status.json)",
    )
    parser.add_argument(
        "--qc-skip-status",
        nargs="*",
        default=["good", "bad"],
        choices=["good", "bad", "review"],
        help="QC statuses to skip in future runs (default: good bad). Use '--qc-skip-status good' to re-visit bad exams",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="start HTTP server to serve gallery and handle QC saves (recommended for remote work)",
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

    if not args.raw.exists():
        logger.error(f"raw directory not found: {args.raw}")
        return 1

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

    # run gallery generation
    generate_gallery(
        views_parquet=views_path,
        raw_dir=args.raw,
        output_dir=args.output,
        max_exams=args.max_exams,
        random_sample=args.random,
        patient_id=args.patient,
        exam_id=args.exam,
        exam_list_path=args.exam_list,
        per_view=args.per_view,
        no_gallery=args.no_gallery,
        qc_file=args.qc_file,
        pred_csv=args.pred_csv,
        meta_csv=args.meta_csv,
        prioritize_errors=args.prioritize_errors,
        horizon=args.horizon,
        qc_skip_status=set(args.qc_skip_status) if args.qc_skip_status else None,
        serve=args.serve,
        original_args=vars(args),
    )

    # start HTTP server if requested
    if args.serve:
        logger.info("=" * 60)
        logger.info("Starting HTTP server for QC gallery...")
        # prepare args for load-more functionality
        load_more_args = {
            "views_parquet": views_path,
            "raw_dir": args.raw,
            "output_dir": args.output,
            "max_exams": args.max_exams,
            "random_sample": args.random,
            "patient_id": args.patient,
            "exam_id": args.exam,
            "exam_list_path": args.exam_list,
            "per_view": args.per_view,
            "no_gallery": args.no_gallery,
            "qc_file": args.qc_file,
            "pred_csv": args.pred_csv,
            "meta_csv": args.meta_csv,
            "prioritize_errors": args.prioritize_errors,
            "horizon": args.horizon,
            "qc_skip_status": set(args.qc_skip_status) if args.qc_skip_status else None,
            "serve": False,  # don't recurse when regenerating
            "original_args": None,
        }
        start_qc_server(
            args.output, args.qc_file, views_path, tags_path, args.port, load_more_args
        )
        # server runs until Ctrl+C
        return 0

    return 0


if __name__ == "__main__":
    exit(main())
