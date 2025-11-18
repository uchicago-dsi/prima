#!/usr/bin/env python3
"""Shard CSV and run mirai predictions in parallel GPU jobs."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import time
from pathlib import Path


def shard_csv(
    input_csv: Path,
    num_shards: int | None,
    max_samples_per_shard: int | None,
    work_dir: Path,
    debug_max_samples: int | None = None,
) -> list[Path]:
    """Split CSV into shards by exam (not by view), preserving header in each.

    Groups rows by (patient_id, exam_id) to ensure complete exams aren't split across shards.
    """
    shard_paths = []
    with open(input_csv, "r") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)

    total_rows = len(rows)
    original_total = total_rows

    # Group rows by (patient_id, exam_id) to count exams, not views
    exam_groups = {}
    for row in rows:
        exam_key = (row["patient_id"], row["exam_id"])
        if exam_key not in exam_groups:
            exam_groups[exam_key] = []
        exam_groups[exam_key].append(row)

    total_exams = len(exam_groups)
    exam_list = list(exam_groups.items())

    if debug_max_samples is not None and debug_max_samples > 0:
        exam_list = exam_list[:debug_max_samples]
        total_exams = len(exam_list)
        total_rows = sum(len(views) for _, views in exam_list)
        print(
            f"DEBUG MODE: Limited to {total_exams} exams ({total_rows} views) (from {original_total} views total)"
        )
    else:
        print(f"Found {total_exams} exams ({total_rows} views)")

    if num_shards is None:
        if max_samples_per_shard is None:
            raise ValueError(
                "Must specify either --num_shards or --max_samples_per_shard"
            )
        num_shards = (total_exams + max_samples_per_shard - 1) // max_samples_per_shard
        print(
            f"Calculated {num_shards} shards from {total_exams} exams with max {max_samples_per_shard} exams per shard"
        )

    exams_per_shard = total_exams // num_shards
    if exams_per_shard == 0:
        exams_per_shard = 1

    for i in range(num_shards):
        start_idx = i * exams_per_shard
        if i == num_shards - 1:
            end_idx = total_exams
        else:
            end_idx = (i + 1) * exams_per_shard

        shard_path = work_dir / f"chunk_{i:03d}.csv"
        shard_rows = []
        for exam_key, views in exam_list[start_idx:end_idx]:
            shard_rows.extend(views)

        with open(shard_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(shard_rows)
        shard_paths.append(shard_path)
        print(
            f"Created shard {i}: {shard_path} ({end_idx - start_idx} exams, {len(shard_rows)} views)"
        )

    return shard_paths


def create_sbatch_script(
    shard_idx: int,
    shard_csv: Path,
    output_csv: Path,
    base_args: list[str],
    work_dir: Path,
    partition: str,
    gpuspec: str,
    mem_gb: int,
    timeout_min: int,
) -> Path:
    """Generate sbatch script for a single shard."""
    script_path = work_dir / f"job_{shard_idx:03d}.sh"
    log_path = work_dir / f"job_{shard_idx:03d}.log"

    gres = f"gpu:{gpuspec}:1" if gpuspec else "gpu:1"

    script_content = f"""#!/bin/bash
#SBATCH --job-name=mirai_shard_{shard_idx:03d}
#SBATCH --output={log_path}
#SBATCH --error={log_path}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --mem={mem_gb}G
#SBATCH --time={timeout_min}
#SBATCH --cpus-per-task=6

eval "$(micromamba shell hook -s bash)"
micromamba activate prima
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd {shlex.quote(str(Path.cwd()))}
mkdir -p logs snapshot
# Remove logs/snapshot if it exists as a directory (should be a file)
rm -rf logs/snapshot
python vendor/mirai/scripts/main.py \\
"""
    for arg in base_args:
        script_content += f"  {shlex.quote(arg)} \\\n"

    script_content += f"  --metadata_path {shlex.quote(str(shard_csv))} \\\n"
    script_content += f"  --prediction_save_path {shlex.quote(str(output_csv))}\n"

    script_path.write_text(script_content)
    script_path.chmod(0o755)
    return script_path, log_path


def submit_jobs(script_paths: list[Path], log_paths: list[Path]) -> list[str]:
    """Submit all sbatch jobs and return job IDs."""
    job_ids = []
    for script_path, log_path in zip(script_paths, log_paths):
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        job_id = result.stdout.strip().split()[-1]
        job_ids.append(job_id)
        print(f"Submitted {script_path.name}: job {job_id}")
        print(f"  Script: {script_path}")
        print(f"  Logs (stdout/stderr): {log_path}")
    return job_ids


def check_job_status(job_ids: list[str]) -> dict[str, str]:
    """Check job status using sacct. Returns dict mapping job_id to state."""
    if not job_ids:
        return {}

    result = subprocess.run(
        ["sacct", "-j", ",".join(job_ids), "-n", "-P", "-o", "JobID,State,ExitCode"],
        capture_output=True,
        text=True,
    )

    status_map = {}
    if result.returncode == 0:
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.strip().split("|")
            if len(parts) >= 2:
                job_id = parts[0].split(".")[0]
                state = parts[1]
                exit_code = parts[2] if len(parts) >= 3 else "0:0"
                # Check if exit code indicates failure (format is "exit_code:signal")
                exit_parts = exit_code.split(":")
                if len(exit_parts) >= 1 and exit_parts[0] != "0":
                    # Job completed but with non-zero exit code = failed
                    status_map[job_id] = "FAILED"
                else:
                    status_map[job_id] = state

    return status_map


def wait_for_jobs(job_ids: list[str]) -> None:
    """Wait for all jobs to complete."""
    print(f"\nWaiting for {len(job_ids)} job(s) to complete...")
    while True:
        result = subprocess.run(
            ["squeue", "-j", ",".join(job_ids), "-h", "-o", "%i"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Warning: squeue returned error: {result.stderr}")
            time.sleep(30)
            continue

        running = [
            jid.strip() for jid in result.stdout.strip().split("\n") if jid.strip()
        ]
        if not running:
            break
        print(
            f"  {len(running)} job(s) still running: {', '.join(running[:5])}{'...' if len(running) > 5 else ''}"
        )
        time.sleep(30)
    print("All jobs completed")

    status_map = check_job_status(job_ids)
    failed_jobs = [
        jid
        for jid, state in status_map.items()
        if state not in ("COMPLETED", "RUNNING", "PENDING")
    ]
    if failed_jobs:
        print(f"\nWARNING: {len(failed_jobs)} job(s) failed or were cancelled:")
        for jid in failed_jobs:
            state = status_map.get(jid, "UNKNOWN")
            print(f"  Job {jid}: {state}")
    else:
        # Double-check by looking for missing output files
        print("All jobs completed (checking outputs...)")


def collate_results(
    output_csvs: list[Path],
    final_output: Path,
    job_ids: list[str] | None = None,
    log_paths: list[Path] | None = None,
) -> None:
    """Merge all shard outputs into final CSV."""
    if not output_csvs:
        raise ValueError("No output files to collate")

    print(f"\nCollating results from {len(output_csvs)} shard(s)...")
    missing_files = []
    successful_files = []

    with open(final_output, "w", newline="") as outfile:
        writer = None
        expected_header = None
        for i, csv_path in enumerate(output_csvs):
            if not csv_path.exists():
                missing_files.append((i, csv_path))
                if job_ids and log_paths and i < len(job_ids) and i < len(log_paths):
                    log_path = log_paths[i]
                    job_id = job_ids[i]
                    print(f"ERROR: {csv_path} does not exist (job {job_id})")
                    if log_path.exists():
                        print(f"  Check log file: {log_path}")
                        try:
                            log_content = log_path.read_text()
                            last_lines = log_content.split("\n")[-10:]
                            print("  Last 10 lines of log:")
                            for line in last_lines:
                                if line.strip():
                                    print(f"    {line}")

                            # Check for common errors and provide helpful messages
                            if (
                                "IsADirectoryError" in log_content
                                and "logs/snapshot" in log_content
                            ):
                                print(
                                    "  -> DIRECTORY CONFLICT: logs/snapshot exists as directory but should be a file"
                                )
                                print("     Remove the directory: rm -rf logs/snapshot")
                            elif (
                                "FileNotFoundError" in log_content
                                and "logs/snapshot" in log_content
                            ):
                                print(
                                    "  -> FIXED: Directory creation added to script (rerun jobs)"
                                )
                            elif (
                                "ValueError" in log_content
                                and "numeric" in log_content.lower()
                            ):
                                print(
                                    "  -> DATA ISSUE: CSV contains non-numeric values where numeric expected"
                                )
                                print(
                                    "     Check your CSV for missing values, date strings, or invalid data"
                                )
                            elif "lifelines" in log_content.lower():
                                print(
                                    "  -> DATA ISSUE: Survival analysis failed - check CSV data quality"
                                )
                        except Exception as e:
                            print(f"  Could not read log: {e}")
                    else:
                        print(f"  Log file also missing: {log_path}")
                else:
                    print(f"ERROR: {csv_path} does not exist")
                continue

            successful_files.append(csv_path)
            with open(csv_path, "r") as infile:
                reader = csv.reader(infile)
                header = next(reader)
                if writer is None:
                    writer = csv.writer(outfile)
                    writer.writerow(header)
                    expected_header = header
                elif header != expected_header:
                    print(f"Warning: header mismatch in {csv_path}")

                for row in reader:
                    writer.writerow(row)
            print(f"  Merged {csv_path}")

    if missing_files:
        print(f"\nWARNING: {len(missing_files)} output file(s) missing:")
        for i, csv_path in missing_files:
            print(f"  {csv_path}")
        if successful_files:
            print(
                f"\nCollated {len(successful_files)} successful shard(s) to {final_output}"
            )
        else:
            raise RuntimeError(
                f"No successful outputs found. All {len(missing_files)} job(s) failed."
            )
    else:
        print(f"Final output written to {final_output}")


def save_job_metadata(
    work_dir: Path,
    job_ids: list[str],
    output_csvs: list[Path],
    final_output: Path,
) -> Path:
    """Save job metadata for recovery."""
    metadata = {
        "job_ids": job_ids,
        "output_csvs": [str(p) for p in output_csvs],
        "final_output": str(final_output),
        "work_dir": str(work_dir),
    }
    metadata_path = work_dir / "job_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata_path


def load_job_metadata(metadata_path: Path) -> dict:
    """Load job metadata from recovery file."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    return json.loads(metadata_path.read_text())


def main():
    parser = argparse.ArgumentParser(
        description="Shard CSV and run mirai predictions in parallel",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--recover",
        type=Path,
        metavar="METADATA_JSON",
        help="Recover from previous run using metadata JSON file",
    )
    shard_group = parser.add_mutually_exclusive_group(required=False)
    shard_group.add_argument(
        "--num_shards",
        type=int,
        help="Number of shards to split CSV into",
    )
    shard_group.add_argument(
        "--max_samples_per_shard",
        type=int,
        help="Maximum number of samples per shard (number of shards will be calculated)",
    )
    parser.add_argument(
        "--metadata_path",
        type=Path,
        required=True,
        help="Input CSV metadata file",
    )
    parser.add_argument(
        "--prediction_save_path",
        type=Path,
        required=True,
        help="Final output CSV path",
    )
    parser.add_argument(
        "--work_dir",
        type=Path,
        default=None,
        help="Working directory for shards and scripts (default: temp dir)",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="gpuq",
        help="SLURM partition",
    )
    parser.add_argument(
        "--gpuspec",
        type=str,
        default="",
        help="GPU type specifier (e.g., a40)",
    )
    parser.add_argument(
        "--mem_gb",
        type=int,
        default=60,
        help="Memory per job (GB)",
    )
    parser.add_argument(
        "--timeout_min",
        type=int,
        default=720,
        help="Job timeout (minutes)",
    )
    parser.add_argument(
        "--no_wait",
        action="store_true",
        help="Submit jobs and exit immediately",
    )
    parser.add_argument(
        "--debug_max_samples",
        type=int,
        default=None,
        help="DEBUG MODE: Limit total number of samples processed (for testing)",
    )

    args, remaining = parser.parse_known_args()

    if args.recover:
        print(f"Recovering from {args.recover}...")
        metadata = load_job_metadata(args.recover)
        work_dir = Path(metadata["work_dir"])
        output_csvs = [Path(p) for p in metadata["output_csvs"]]
        final_output = Path(metadata["final_output"])
        job_ids = metadata["job_ids"]

        print(f"Found {len(job_ids)} job(s) in metadata")
        print(f"Work directory: {work_dir}")
        print(f"Final output: {final_output}")

        wait_for_jobs(job_ids)
        log_paths = [work_dir / f"job_{i:03d}.log" for i in range(len(output_csvs))]
        collate_results(output_csvs, final_output, job_ids, log_paths)
        print(f"\nDone! Results in {final_output}")
        return

    if not args.num_shards and not args.max_samples_per_shard:
        parser.error(
            "Must specify either --num_shards or --max_samples_per_shard (or use --recover)"
        )

    if not args.metadata_path.exists():
        parser.error(f"Metadata file not found: {args.metadata_path}")

    work_dir = args.work_dir
    if work_dir is None:
        work_dir = Path(args.prediction_save_path).parent / "mirai_shards"
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.num_shards:
        print(f"Sharding {args.metadata_path} into {args.num_shards} chunks...")
    else:
        print(
            f"Sharding {args.metadata_path} with max {args.max_samples_per_shard} samples per shard..."
        )
    shard_paths = shard_csv(
        args.metadata_path,
        args.num_shards,
        args.max_samples_per_shard,
        work_dir,
        args.debug_max_samples,
    )

    output_csvs = []
    script_paths = []
    log_paths = []
    for i, shard_path in enumerate(shard_paths):
        output_csv = work_dir / f"output_{i:03d}.csv"
        output_csvs.append(output_csv)
        script_path, log_path = create_sbatch_script(
            i,
            shard_path,
            output_csv,
            remaining,
            work_dir,
            args.partition,
            args.gpuspec,
            args.mem_gb,
            args.timeout_min,
        )
        script_paths.append(script_path)
        log_paths.append(log_path)

    job_ids = submit_jobs(script_paths, log_paths)
    print(f"\nSubmitted {len(job_ids)} job(s)")

    metadata_path = save_job_metadata(
        work_dir, job_ids, output_csvs, args.prediction_save_path
    )
    print(f"Saved job metadata to {metadata_path}")

    if args.no_wait:
        print(f"Jobs submitted. Results will be in {work_dir}")
        print(f"To recover later, run: python {__file__} --recover {metadata_path}")
        return

    wait_for_jobs(job_ids)
    collate_results(output_csvs, args.prediction_save_path, job_ids, log_paths)
    print(f"\nDone! Results in {args.prediction_save_path}")


if __name__ == "__main__":
    main()
