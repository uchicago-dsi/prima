#!/usr/bin/env python3
"""Build a local review bundle for a single-tag auto-QC run."""

from __future__ import annotations

import argparse
import csv
import html
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from prima.auto_qc import auto_run_to_tag_map, load_auto_run
from prima.qc_state import load_qc_state, qc_state_to_annotations_map


@dataclass
class ReviewRow:
    review_id: str
    outcome: str
    exam_id: str
    truth: bool
    pred: bool
    pred_score: str
    pred_rationale: str
    image_path: Path
    copied_image_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a local FP/FN review sheet for a single tag."
    )
    parser.add_argument("--qc-file", type=Path, required=True)
    parser.add_argument("--run-file", type=Path, required=True)
    parser.add_argument("--tag", required=True, help="Tag to review, e.g. BB")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def slugify_tag(tag: str) -> str:
    slug = tag.lower().replace(" ", "_").replace("/", "_")
    slug = "".join(ch for ch in slug if ch.isalnum() or ch == "_")
    return slug or "tag"


def format_float(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def detect_score(suggestion: dict[str, object]) -> str:
    for key in ("score", "confidence", "probability"):
        if key in suggestion:
            return format_float(suggestion[key])
    return ""


def markdown_image_link(image_path: Path, label: str) -> str:
    return f"[{label}]({image_path})"


def build_rows(
    run: dict[str, object],
    gt_by_exam: dict[str, set[str]],
    pred_by_exam: dict[str, set[str]],
    tag: str,
) -> tuple[list[dict[str, object]], list[ReviewRow], dict[str, int]]:
    exam_suggestions = run.get("exam_suggestions", {})
    if not isinstance(exam_suggestions, dict):
        raise ValueError("run file has invalid exam_suggestions payload")

    all_rows: list[dict[str, object]] = []
    error_rows: list[ReviewRow] = []
    metrics = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    for index, exam_id in enumerate(sorted(exam_suggestions), start=1):
        record = exam_suggestions[exam_id]
        if not isinstance(record, dict):
            raise ValueError(f"invalid record for exam {exam_id}")

        truth = tag in gt_by_exam.get(exam_id, set())
        pred = tag in pred_by_exam.get(exam_id, set())

        if truth and pred:
            outcome = "TP"
        elif pred and not truth:
            outcome = "FP"
        elif truth and not pred:
            outcome = "FN"
        else:
            outcome = "TN"
        metrics[outcome] += 1

        suggestions = record.get("suggestions", [])
        matching_suggestion = None
        if isinstance(suggestions, list):
            for suggestion in suggestions:
                if isinstance(suggestion, dict) and suggestion.get("tag") == tag:
                    matching_suggestion = suggestion
                    break

        image_path = Path(str(record.get("image_path", ""))).resolve()
        if not image_path.exists():
            raise FileNotFoundError(
                f"missing montage image for {exam_id}: {image_path}"
            )

        pred_score = detect_score(matching_suggestion or {})
        pred_rationale = ""
        if matching_suggestion is not None:
            pred_rationale = str(matching_suggestion.get("rationale", "") or "")

        copied_image_name = ""
        review_id = ""
        if outcome in {"FP", "FN"}:
            review_id = f"{len(error_rows) + 1:03d}_{outcome}"
            copied_image_name = f"{review_id}.png"
            error_rows.append(
                ReviewRow(
                    review_id=review_id,
                    outcome=outcome,
                    exam_id=str(exam_id),
                    truth=truth,
                    pred=pred,
                    pred_score=pred_score,
                    pred_rationale=pred_rationale,
                    image_path=image_path,
                    copied_image_name=copied_image_name,
                )
            )

        all_rows.append(
            {
                "review_id": review_id,
                "outcome": outcome,
                "exam_id": str(exam_id),
                "truth_tag": int(truth),
                "pred_tag": int(pred),
                "pred_score": pred_score,
                "pred_rationale": pred_rationale,
                "image_path": str(image_path),
                "image_file": copied_image_name,
                "notes": "",
                "disposition": "",
            }
        )

    return all_rows, error_rows, metrics


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(error_rows: list[ReviewRow], image_dir: Path) -> str:
    lines = ["# Error Cases", ""]
    for row in error_rows:
        image_rel = f"images/{row.copied_image_name}"
        rationale = row.pred_rationale if row.pred_rationale else "(none)"
        lines.extend(
            [
                f"## {row.review_id} {row.outcome}",
                f"Exam: `{row.exam_id}`  ",
                f"Truth: `{int(row.truth)}` Pred: `{int(row.pred)}` Score: `{row.pred_score}`  ",
                f"Rationale: {rationale}  ",
                f"Image: [{row.copied_image_name}]({image_rel})",
                "",
            ]
        )
    return "\n".join(lines)


def build_html(error_rows: list[ReviewRow], image_dir: Path, embedded: bool) -> str:
    style = """
body { font-family: sans-serif; margin: 24px; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ccc; padding: 8px; vertical-align: top; text-align: left; }
th { background: #f5f5f5; position: sticky; top: 0; }
img.thumb { max-width: 420px; height: auto; display: block; }
code { font-family: monospace; }
"""
    rows_html: list[str] = []
    for row in error_rows:
        image_path = (image_dir / row.copied_image_name).resolve()
        image_rel = f"images/{row.copied_image_name}"
        if embedded:
            import base64

            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            image_cell = (
                f'<a href="{html.escape(image_rel)}">'
                f'<img class="thumb" src="data:image/png;base64,{encoded}" alt="{html.escape(row.review_id)}"></a>'
            )
        else:
            image_cell = f'<a href="{html.escape(image_rel)}">Open image</a>'

        rows_html.append(
            "<tr>"
            f"<td><code>{html.escape(row.review_id)}</code></td>"
            f"<td>{html.escape(row.outcome)}</td>"
            f"<td><code>{html.escape(row.exam_id)}</code></td>"
            f"<td>{int(row.truth)}</td>"
            f"<td>{int(row.pred)}</td>"
            f"<td>{html.escape(row.pred_score)}</td>"
            f"<td>{html.escape(row.pred_rationale)}</td>"
            f"<td>{image_cell}</td>"
            "</tr>"
        )

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{style}</style>"
        "</head><body>"
        "<table>"
        "<thead><tr><th>Review ID</th><th>Outcome</th><th>Exam</th><th>Truth</th><th>Pred</th><th>Score</th><th>Rationale</th><th>Image</th></tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table></body></html>"
    )


def write_readme(
    out_dir: Path,
    run_file: Path,
    qc_file: Path,
    tag: str,
    tag_slug: str,
    exam_count: int,
    metrics: dict[str, int],
) -> None:
    tp = metrics["TP"]
    fp = metrics["FP"]
    fn = metrics["FN"]
    tn = metrics["TN"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    readme = "\n".join(
        [
            f"# {tag} Review Sheet",
            "",
            f"- Created: {datetime.now().isoformat(timespec='seconds')}",
            f"- Run file: {run_file.name}",
            f"- QC file: {qc_file.name}",
            f"- Tag: {tag}",
            f"- Exams scored: {exam_count}",
            f"- TP: {tp}",
            f"- FP: {fp}",
            f"- FN: {fn}",
            f"- TN: {tn}",
            f"- Precision: {precision:.3f}",
            f"- Recall: {recall:.3f}",
            f"- Specificity: {specificity:.3f}",
            "",
            "Files:",
            f"- `{tag_slug}_review_open_links.html`: direct-link review page with one row per FP/FN case.",
            f"- `{tag_slug}_error_cases_links.md`: Markdown index with a clickable local image link per error case.",
            f"- `{tag_slug}_review.html`: embedded-thumbnail review page for preview compatibility.",
            f"- `{tag_slug}_error_cases.csv`: FP/FN cases only.",
            f"- `{tag_slug}_all_cases.csv`: all scored cases.",
            "- `images/`: scrubbed montage copies for FP/FN review only.",
            "",
        ]
    )
    (out_dir / "README.md").write_text(readme)


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    image_dir = out_dir / "images"
    tag_slug = slugify_tag(args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    qc_state = load_qc_state(args.qc_file.resolve())
    run = load_auto_run(args.run_file.resolve())

    gt_by_exam = {
        str(exam_id): set(tags)
        for exam_id, tags in qc_state_to_annotations_map(qc_state).items()
    }
    pred_by_exam = {
        str(exam_id): set(tags) for exam_id, tags in auto_run_to_tag_map(run).items()
    }

    all_rows, error_rows, metrics = build_rows(
        run=run,
        gt_by_exam=gt_by_exam,
        pred_by_exam=pred_by_exam,
        tag=args.tag,
    )

    for row in error_rows:
        shutil.copy2(row.image_path, image_dir / row.copied_image_name)

    error_csv_rows = [row for row in all_rows if row["outcome"] in {"FP", "FN"}]

    write_csv(out_dir / f"{tag_slug}_all_cases.csv", all_rows)
    if error_csv_rows:
        write_csv(out_dir / f"{tag_slug}_error_cases.csv", error_csv_rows)
    else:
        (out_dir / f"{tag_slug}_error_cases.csv").write_text(
            "review_id,outcome,exam_id,truth_tag,pred_tag,pred_score,pred_rationale,image_path,image_file,notes,disposition\n"
        )

    (out_dir / f"{tag_slug}_error_cases_links.md").write_text(
        build_markdown(error_rows, image_dir)
    )
    (out_dir / f"{tag_slug}_review_open_links.html").write_text(
        build_html(error_rows, image_dir, embedded=False)
    )
    (out_dir / f"{tag_slug}_review.html").write_text(
        build_html(error_rows, image_dir, embedded=True)
    )
    write_readme(
        out_dir=out_dir,
        run_file=args.run_file.resolve(),
        qc_file=args.qc_file.resolve(),
        tag=args.tag,
        tag_slug=tag_slug,
        exam_count=len(all_rows),
        metrics=metrics,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
