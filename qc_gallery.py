#!/usr/bin/env python3
"""qc_gallery.py

Interactive QC tool for reviewing and marking mammogram exams with persistent status tracking.

This script reads from views.parquet (output from preprocessing) and generates:
1. Combined 4-view figures (L_CC, L_MLO, R_CC, R_MLO) for each exam
2. Optional per-view debug figures with DICOM tag info
3. Interactive HTML gallery with button-based navigation, filtering, and QC marking

The gallery features:
- Single-image viewer with Previous/Next buttons (faster than scrolling)
- Keyboard navigation (left/right arrows, G/R/B for QC, S to save)
- Search/filter by patient ID, exam ID, or accession number
- QC buttons to mark exams as "good", "needs review", or "bad"
- QC status persistence: saves to JSON file, skips "good" exams on next run
- Real-time QC statistics in the status bar

QC Workflow
-----------
1. Run script with --serve flag to generate gallery and start HTTP server
2. Open gallery in browser (with SSH port forwarding if remote)
3. Mark exams using keyboard (G/R/B) or buttons - auto-saves to server after each click
4. Re-run script to continue QC on remaining exams (already-marked "good" exams skipped)

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
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset
from pydicom.tag import Tag
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
OUTPUT_DIR = None


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
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """override to use our logger."""
        # skip logging favicon requests
        if "favicon.ico" in format % args:
            return
        logger.info(f"HTTP: {format % args}")


def start_qc_server(output_dir: Path, qc_file: Path, port: int = 5000) -> None:
    """start HTTP server to serve gallery and handle QC saves.

    parameters
    ----------
    output_dir : Path
        directory containing gallery.html and images
    qc_file : Path
        path where QC data should be saved
    port : int
        port to listen on
    """
    global QC_FILE_PATH, OUTPUT_DIR

    QC_FILE_PATH = qc_file.resolve()
    OUTPUT_DIR = output_dir.resolve()

    # change to output directory so SimpleHTTPRequestHandler can serve files
    import os

    original_dir = os.getcwd()
    os.chdir(OUTPUT_DIR)

    server_address = ("", port)
    httpd = HTTPServer(server_address, QCGalleryHandler)

    logger.info(f"QC file: {QC_FILE_PATH}")
    logger.info(f"Serving from: {OUTPUT_DIR}")
    logger.info(f"Gallery URL: http://localhost:{port}/")
    logger.info("=" * 60)
    logger.info("QC workflow:")
    logger.info(f"  1. Open http://localhost:{port}/ in your browser")
    logger.info("  2. Mark exams with G/R/B keys or buttons")
    logger.info("  3. QC data auto-saves to server on each click")
    logger.info("  4. Press Ctrl+C to stop server when done")
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
) -> bool:
    """save a combined figure with all four views of an exam.

    Args:
        view_data: dict mapping view keys (L_CC, L_MLO, R_CC, R_MLO) to (dicom_dataset, path) tuples
        debug_dir: output directory
        patient_id: patient identifier
        accession_number: accession number
        exam_id: exam identifier

    Returns:
        True if successfully saved, False otherwise
    """
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

        # save figure
        success_dir = debug_dir / "success"
        patient_dir = success_dir / patient_id
        accession_dir = patient_dir / accession_number
        accession_dir.mkdir(parents=True, exist_ok=True)

        out_path = accession_dir / f"COMBINED_four_views_{exam_id}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=75, bbox_inches="tight")
        plt.close(fig)

        logger.debug(f"    Saved combined 4-view figure: {out_path.name}")
        return True

    except Exception as e:
        logger.warning(
            f"    Failed to save combined 4-view figure for exam {exam_id}: {e}"
        )
        return False


def generate_gallery(
    views_parquet: Path,
    raw_dir: Path,
    output_dir: Path,
    max_exams: Optional[int] = None,
    random_sample: bool = False,
    patient_id: Optional[str] = None,
    exam_id: Optional[str] = None,
    per_view: bool = False,
    no_gallery: bool = False,
    qc_file: Optional[Path] = None,
) -> None:
    """generate interactive HTML gallery from processed views.

    Args:
        views_parquet: path to views.parquet with dicom_path column
        raw_dir: root directory where DICOMs are stored
        output_dir: output directory for figures and gallery
        max_exams: limit number of exams to visualize
        random_sample: if True, sample randomly; else take first N
        patient_id: filter to specific patient
        exam_id: filter to specific exam
        per_view: if True, also generate individual per-view debug figures
        no_gallery: if True, skip HTML gallery generation
        qc_file: path to QC status file; exams marked "good" will be skipped
    """
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

    # filter by patient/exam if specified
    if patient_id:
        views_df = views_df[views_df["patient_id"] == patient_id]
        logger.info(
            f"filtered to patient {patient_id}: {len(views_df)} views from {views_df['exam_id'].nunique()} exams"
        )

    if exam_id:
        views_df = views_df[views_df["exam_id"] == exam_id]
        logger.info(f"filtered to exam {exam_id}: {len(views_df)} views")

    # filter out exams already marked as "good" in QC
    if qc_data:
        good_exams = {eid for eid, status in qc_data.items() if status == "good"}
        if good_exams:
            before_count = views_df["exam_id"].nunique()
            views_df = views_df[~views_df["exam_id"].isin(good_exams)]
            after_count = views_df["exam_id"].nunique()
            logger.info(
                f"filtered out {before_count - after_count} exams already marked 'good' in QC"
            )

    if len(views_df) == 0:
        logger.error("no views to visualize after filtering")
        return

    # sample exams if requested
    if max_exams:
        unique_exams = views_df["exam_id"].unique()
        if len(unique_exams) > max_exams:
            if random_sample:
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

    # process each view and track by exam
    success_count = 0
    error_count = 0
    combined_figure_count = 0

    # dict to track loaded views per exam: {exam_id: {view_key: (ds, path), ...}}
    exam_view_cache = {}

    for idx, row in tqdm(
        views_df.iterrows(), total=len(views_df), desc="loading DICOMs"
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
            logger.error(f"error processing {row.get('dicom_path', 'unknown')}: {e}")
            error_count += 1

    # generate combined 4-view figures for each exam
    logger.info("generating combined 4-view figures...")
    combined_images = []  # track for HTML gallery

    for exam_key, exam_data in tqdm(exam_view_cache.items(), desc="combined figures"):
        try:
            success = _save_four_view_figure(
                view_data=exam_data["views"],
                debug_dir=output_dir,
                patient_id=exam_data["patient_id"],
                accession_number=exam_data["accession_number"],
                exam_id=exam_key,
            )

            if success:
                combined_figure_count += 1

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
                        "qc_status": qc_data.get(exam_key, ""),
                    }
                )

        except Exception as e:
            logger.error(f"error creating combined figure for exam {exam_key}: {e}")

    # generate HTML gallery
    if not no_gallery and combined_images:
        logger.info("generating HTML gallery...")
        gallery_path = output_dir / "gallery.html"

        # prepare QC file path for saving
        qc_file_str = str(qc_file.resolve()) if qc_file else "data/qc_status.json"

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Mammogram QC - {combined_figure_count} exams</title>
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
        }}
        .filter-controls input {{
            padding: 6px 12px;
            font-size: 14px;
            border: 1px solid #3e3e42;
            border-radius: 4px;
            background-color: #1e1e1e;
            color: #d4d4d4;
            width: 300px;
        }}
        .filter-controls input:focus {{
            outline: none;
            border-color: #4ec9b0;
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
    </style>
</head>
<body>
    <div class="container">
        <div class="controls">
            <div class="stats" id="stats">
                {combined_figure_count} exams | {success_count} views loaded
            </div>
            <div class="filter-controls">
                <input type="text" id="searchBox" placeholder="Filter by patient ID, exam ID, or accession..." 
                       onkeyup="filterGallery()">
            </div>
            <div class="nav-buttons">
                <button id="prevBtn" onclick="navigate(-1)">← Previous</button>
                <button id="nextBtn" onclick="navigate(1)">Next →</button>
            </div>
        </div>
        <div class="viewer" id="viewer">
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
        
        let filteredExams = [...allExams];
        let currentIndex = 0;
        
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
        
        function autoSaveQCData() {{
            // save to localStorage as backup
            localStorage.setItem('qc_data', JSON.stringify(qcData));
            
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
            autoSaveQCData();
            
            // auto-advance to next exam (navigate will call updateView)
            navigate(1);
        }}
        
        function filterGallery() {{
            const searchTerm = document.getElementById('searchBox').value.toLowerCase();
            
            if (searchTerm === '') {{
                filteredExams = [...allExams];
            }} else {{
                filteredExams = allExams.filter(exam => 
                    exam.patient_id.toLowerCase().includes(searchTerm) ||
                    exam.exam_id.toLowerCase().includes(searchTerm) ||
                    exam.accession.toLowerCase().includes(searchTerm)
                );
            }}
            
            currentIndex = 0;
            updateView();
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
                return;
            }}
            
            const exam = filteredExams[currentIndex];
            const currentStatus = qcData[exam.exam_id] || '';
            
            // count QC statuses
            const qcCounts = {{good: 0, review: 0, bad: 0, pending: 0}};
            Object.values(qcData).forEach(status => {{
                if (status in qcCounts) qcCounts[status]++;
            }});
            qcCounts.pending = allExams.length - Object.keys(qcData).length;
            
            viewer.innerHTML = 
                '<div class="exam-info">' +
                    '<strong>Patient:</strong> ' + exam.patient_id + ' | ' +
                    '<strong>Exam:</strong> ' + exam.exam_id + ' | ' +
                    '<strong>Accession:</strong> ' + exam.accession + ' | ' +
                    '<strong>Views:</strong> ' + exam.num_views + '/4' +
                '</div>' +
                '<div class="qc-controls">' +
                    '<button class="qc-button good' + (currentStatus === 'good' ? ' active' : '') + '" ' +
                        'onclick="setQCStatus(\\'good\\')" title="Mark as good">✓ Good [g]</button>' +
                    '<button class="qc-button review' + (currentStatus === 'review' ? ' active' : '') + '" ' +
                        'onclick="setQCStatus(\\'review\\')" title="Needs review">? Review [r]</button>' +
                    '<button class="qc-button bad' + (currentStatus === 'bad' ? ' active' : '') + '" ' +
                        'onclick="setQCStatus(\\'bad\\')" title="Mark as bad">✗ Bad [b]</button>' +
                '</div>' +
                '<div class="image-container">' +
                    '<img src="' + exam.path + '" alt="Exam ' + exam.exam_id + '">' +
                '</div>';
            
            stats.textContent = (currentIndex + 1) + '/' + filteredExams.length + ' | ' + 
                                allExams.length + ' total | ' +
                                'QC: ' + qcCounts.good + ' good, ' + qcCounts.review + ' review, ' + 
                                qcCounts.bad + ' bad, ' + qcCounts.pending + ' pending';
            
            prevBtn.disabled = currentIndex === 0;
            nextBtn.disabled = currentIndex === filteredExams.length - 1;
        }}
        
        // keyboard navigation
        document.addEventListener('keydown', (e) => {{
            // ignore if typing in search box
            if (e.target.id === 'searchBox') return;
            
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
            }}
        }});
        
        // initialize
        updateView();
        
        // show instructions in console
        console.log('QC data auto-saves to server on each button click');
        console.log('Server saves to: {qc_file_str}');
        console.log('Keyboard shortcuts: g=good, r=review, b=bad, Arrow keys=navigate');
        console.log('Backup: QC data also saved to browser localStorage - safe to refresh page');
    </script>
</body>
</html>
"""

        with open(gallery_path, "w") as f:
            f.write(html_content)

        logger.info(f"HTML gallery saved to: {gallery_path}")
        logger.info(f"Open in browser: file://{gallery_path.absolute()}")

        if qc_file:
            logger.info(f"QC file: {qc_file.resolve()}")
            logger.info(
                f"QC status: {len([s for s in qc_data.values() if s == 'good'])} good, "
                f"{len([s for s in qc_data.values() if s == 'review'])} review, "
                f"{len([s for s in qc_data.values() if s == 'bad'])} bad"
            )
            logger.info("=" * 60)
            logger.info(
                "Without --serve: QC data saved to localStorage + downloads folder"
            )
            logger.info(f"Move downloaded qc_status.json to: {qc_file.resolve()}")
            logger.info(
                "With --serve: QC data auto-saves directly to server (recommended)"
            )
            logger.info("=" * 60)
            logger.info(
                "Keyboard shortcuts: G=Good, R=Review, B=Bad, Arrow keys=Navigate"
            )

    logger.info("gallery generation complete:")
    if per_view:
        logger.info(f"  individual view figures: {success_count}")
    logger.info(f"  combined 4-view figures: {combined_figure_count}")
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
  5. Re-run to continue QC (already-marked "good" exams skipped)

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
        help="randomly sample exams (default: take first N)",
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
        help="path to QC status file (default: data/qc_status.json); exams marked 'good' will be skipped",
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

    # validate inputs
    if not views_path.exists():
        logger.error(f"views parquet not found: {views_path}")
        return 1

    if not args.raw.exists():
        logger.error(f"raw directory not found: {args.raw}")
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
        per_view=args.per_view,
        no_gallery=args.no_gallery,
        qc_file=args.qc_file,
    )

    # start HTTP server if requested
    if args.serve:
        logger.info("=" * 60)
        logger.info("Starting HTTP server for QC gallery...")
        start_qc_server(args.output, args.qc_file, args.port)
        # server runs until Ctrl+C
        return 0

    return 0


if __name__ == "__main__":
    exit(main())
