#!/usr/bin/env python3
# ruff: noqa: E402
"""Shard preprocessing across multiple SLURM jobs and merge results."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.preprocess import build_genotyped_exam_dir_filter

DEFAULT_FINGERPRINT = Path("fingerprints/chimec/disk_fingerprints.json")
DEFAULT_PATIENTS = Path(
    "/gpfs/data/phs/groups/Projects/Huo_projects/SPORE/annawoodard/List_ChiMEC_priority_2025July30.csv"
)
DEFAULT_KEY = Path("/gpfs/data/huo-lab/Image/ChiMEC/study-16352a.csv")


def partition_allowlist(allowlist: set, num_shards: int) -> list[list]:
    """Partition (patient_id, exam_dir_name) pairs into num_shards. Returns list of shard lists."""
    sorted_pairs = sorted(allowlist)
    shards = [[] for _ in range(num_shards)]
    for i, pair in enumerate(sorted_pairs):
        shards[i % num_shards].append(list(pair))
    return shards


def create_sbatch_script(
    shard_idx: int,
    shard_allowlist_path: Path,
    raw_dir: Path,
    sot_dir: Path,
    out_dir: Path,
    checkpoint_dir: Path,
    work_dir: Path,
    workers: int,
    labels_path: Path | None,
    partition: str,
    mem_gb: int,
    timeout_hours: int,
) -> tuple[Path, Path]:
    """Generate sbatch script for a single preprocess shard."""
    script_path = work_dir / f"job_{shard_idx:03d}.sh"
    log_path = work_dir / f"job_{shard_idx:03d}.log"

    shard_sot = sot_dir / f"shard_{shard_idx:03d}"
    shard_checkpoint = checkpoint_dir / f"shard_{shard_idx:03d}"
    shard_manifest = out_dir / f"manifest_shard_{shard_idx:03d}.parquet"

    cmd_parts = [
        "python",
        "-u",
        "pipelines/preprocess.py",
        "preprocess",
        "--raw",
        str(raw_dir),
        "--sot",
        str(shard_sot),
        "--out",
        str(out_dir),
        "--workers",
        str(workers),
        "--no-resume",
        "--shard-allowlist",
        str(shard_allowlist_path),
        "--checkpoint-dir",
        str(shard_checkpoint),
        "--manifest-output",
        str(shard_manifest),
    ]
    if labels_path:
        cmd_parts.extend(["--labels", str(labels_path)])

    script_content = f"""#!/bin/bash
#SBATCH --job-name=preproc_shard_{shard_idx:03d}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={workers}
#SBATCH --mem={mem_gb}G
#SBATCH --time={timeout_hours}:00:00
#SBATCH --output={log_path}
#SBATCH --error={log_path}

set -eo pipefail

cd {shlex.quote(str(Path.cwd()))}
eval "$(micromamba shell hook -s bash)"
micromamba activate prima
export PYTHONUNBUFFERED=1

echo "[shard {shard_idx}] starting at $(date)"
{" ".join(shlex.quote(p) for p in cmd_parts)}
echo "[shard {shard_idx}] done at $(date)"
"""

    script_path.write_text(script_content)
    script_path.chmod(0o755)
    return script_path, log_path


def submit_jobs(script_paths: list[Path]) -> list[str]:
    """Submit all sbatch jobs and return job IDs."""
    job_ids = []
    for script_path in script_paths:
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        job_id = result.stdout.strip().split()[-1]
        job_ids.append(job_id)
        print(f"Submitted {script_path.name}: job {job_id}")
    return job_ids


def wait_for_jobs(job_ids: list[str], poll_interval: int = 60) -> None:
    """Wait for all SLURM jobs to complete. Uses squeue (reliable for active jobs)."""
    if not job_ids:
        return
    job_set = set(job_ids)
    print(f"\nWaiting for {len(job_ids)} job(s)...")
    # Brief delay so squeue sees newly submitted jobs
    time.sleep(5)
    while True:
        result = subprocess.run(
            ["squeue", "-j", ",".join(job_ids), "-h", "-o", "%i"],
            capture_output=True,
            text=True,
        )
        still_running = set()
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                job_id = line.strip().split(".")[0]
                if job_id in job_set:
                    still_running.add(job_id)
        pending = len(still_running)
        if pending == 0:
            break
        print(f"  {pending} job(s) still running...")
        time.sleep(poll_interval)


def merge_shard_outputs(sot_dir: Path, out_dir: Path, num_shards: int) -> None:
    """Merge parquet files from sot/shard_*/ and manifest_shard_* into final outputs."""
    import pandas as pd

    views_dfs = []
    exams_dfs = []
    tags_dfs = []
    cohort_dfs = []
    manifest_dfs = []

    for i in range(num_shards):
        shard_dir = sot_dir / f"shard_{i:03d}"
        if shard_dir.exists():
            if (shard_dir / "views.parquet").exists():
                views_dfs.append(pd.read_parquet(shard_dir / "views.parquet"))
            if (shard_dir / "exams.parquet").exists():
                exams_dfs.append(pd.read_parquet(shard_dir / "exams.parquet"))
            if (shard_dir / "dicom_tags.parquet").exists():
                tags_dfs.append(pd.read_parquet(shard_dir / "dicom_tags.parquet"))
            if (shard_dir / "cohort.parquet").exists():
                cohort_dfs.append(pd.read_parquet(shard_dir / "cohort.parquet"))
        shard_manifest = out_dir / f"manifest_shard_{i:03d}.parquet"
        if shard_manifest.exists():
            manifest_dfs.append(pd.read_parquet(shard_manifest))

    sot_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if views_dfs:
        combined = pd.concat(views_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["exam_id", "sop_instance_uid"])
        combined.to_parquet(sot_dir / "views.parquet", index=False)
        print(f"  Merged views.parquet: {len(combined):,} rows")
    if exams_dfs:
        combined = pd.concat(exams_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["patient_id", "exam_id"])
        combined.to_parquet(sot_dir / "exams.parquet", index=False)
        print(f"  Merged exams.parquet: {len(combined):,} rows")
    if tags_dfs:
        combined = pd.concat(tags_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["sop_instance_uid"])
        combined.to_parquet(sot_dir / "dicom_tags.parquet", index=False)
        print(f"  Merged dicom_tags.parquet: {len(combined):,} rows")
    if cohort_dfs:
        combined = pd.concat(cohort_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["patient_id", "exam_id"])
        combined.to_parquet(sot_dir / "cohort.parquet", index=False)
        print(f"  Merged cohort.parquet: {len(combined):,} rows")
    if manifest_dfs:
        combined = pd.concat(manifest_dfs, ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["patient_id", "exam_id", "laterality", "view"]
        )
        combined.to_parquet(out_dir / "manifest.parquet", index=False)
        print(f"  Merged manifest.parquet: {len(combined):,} rows")
        for i in range(num_shards):
            p = out_dir / f"manifest_shard_{i:03d}.parquet"
            if p.exists():
                p.unlink()

    # Remove shard dirs
    import shutil

    for i in range(num_shards):
        shard_dir = sot_dir / f"shard_{i:03d}"
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
            print(f"  Removed {shard_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Shard preprocessing across multiple SLURM jobs",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--recover",
        type=Path,
        metavar="METADATA_JSON",
        help="Merge shard outputs from completed run (path to job_metadata.json)",
    )
    parser.add_argument("--raw", type=Path, default=None, help="Raw DICOM directory")
    parser.add_argument(
        "--sot", type=Path, default=None, help="SoT dir (default: <raw>/sot)"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Output dir (default: <raw>/out)"
    )
    parser.add_argument("--workers", type=int, default=32, help="Workers per shard")
    parser.add_argument(
        "--num_shards", type=int, required=True, help="Number of shards"
    )
    parser.add_argument(
        "--genotyped-only",
        action="store_true",
        help="Filter to patients with genomic data",
    )
    parser.add_argument("--fingerprint", type=Path, default=DEFAULT_FINGERPRINT)
    parser.add_argument("--patients", type=Path, default=DEFAULT_PATIENTS)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(
            "/gpfs/data/phs/groups/Projects/Huo_projects/SPORE/annawoodard/Phenotype_ChiMEC_2025Oct4.csv"
        ),
        help="Labels CSV for Mirai (emitted after merge; not passed to shards)",
    )
    parser.add_argument("--partition", type=str, default="tier1q")
    parser.add_argument("--mem-gb", type=int, default=64)
    parser.add_argument("--timeout-hours", type=int, default=24)
    parser.add_argument(
        "--no-wait", action="store_true", help="Submit and exit immediately"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Work dir for shards (default: <raw>/preprocess_shards)",
    )

    args = parser.parse_args()

    if args.recover:
        metadata_path = args.recover
        if not metadata_path.exists():
            print(f"ERROR: Metadata file not found: {metadata_path}", file=sys.stderr)
            sys.exit(1)
        metadata = json.loads(metadata_path.read_text())
        work_dir = Path(metadata["work_dir"])
        sot_dir = Path(metadata["sot_dir"])
        out_dir = Path(metadata["out_dir"])
        num_shards = metadata["num_shards"]
        raw_dir = sot_dir.parent
        print(f"Recovering: merging {num_shards} shards from {work_dir}")
        merge_shard_outputs(sot_dir, out_dir, num_shards)
        labels_path = args.labels or (
            Path(metadata["labels"]) if metadata.get("labels") else None
        )
        if labels_path and Path(labels_path).exists():
            print("\nGenerating Mirai CSV...")
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "pipelines/preprocess.py",
                    "emit-csv",
                    "--raw",
                    str(raw_dir),
                    "--sot",
                    str(sot_dir),
                    "--out",
                    str(out_dir),
                    "--labels",
                    str(labels_path),
                ],
                check=True,
                cwd=Path(__file__).resolve().parent,
            )
            print(f"  Wrote {out_dir / 'mirai_manifest.csv'}")
        print("Done.")
        return

    if not args.raw:
        parser.error("--raw is required (or use --recover)")
    raw_dir = args.raw
    sot_dir = args.sot or raw_dir / "sot"
    out_dir = args.out or raw_dir / "out"
    work_dir = args.work_dir or raw_dir / "preprocess_shards"
    checkpoint_base = Path("data/discovery_checkpoints")

    if not args.genotyped_only:
        parser.error("Sharded preprocessing requires --genotyped-only")

    allowlist = build_genotyped_exam_dir_filter(
        args.fingerprint,
        args.patients,
        args.key,
    )
    if allowlist is None:
        print(
            "ERROR: Could not build genotyped allowlist. Check fingerprint, patients, key.",
            file=sys.stderr,
        )
        sys.exit(1)

    shards = partition_allowlist(allowlist, args.num_shards)
    print(f"Partitioned {len(allowlist):,} exams into {args.num_shards} shards")
    for i, s in enumerate(shards):
        print(f"  Shard {i}: {len(s):,} exams")

    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_base.mkdir(parents=True, exist_ok=True)

    # Write shard allowlists
    script_paths = []
    log_paths = []
    for i, shard_pairs in enumerate(shards):
        if not shard_pairs:
            continue
        allowlist_path = work_dir / f"shard_{i:03d}_allowlist.json"
        with open(allowlist_path, "w") as f:
            json.dump(shard_pairs, f, indent=0)
        script_path, log_path = create_sbatch_script(
            shard_idx=i,
            shard_allowlist_path=allowlist_path,
            raw_dir=raw_dir,
            sot_dir=sot_dir,
            out_dir=out_dir,
            checkpoint_dir=checkpoint_base,
            work_dir=work_dir,
            workers=args.workers,
            labels_path=None,  # shards skip CSV emission; run after merge
            partition=args.partition,
            mem_gb=args.mem_gb,
            timeout_hours=args.timeout_hours,
        )
        script_paths.append(script_path)
        log_paths.append(log_path)

    job_ids = submit_jobs(script_paths)

    metadata = {
        "work_dir": str(work_dir),
        "job_ids": job_ids,
        "num_shards": args.num_shards,
        "sot_dir": str(sot_dir),
        "out_dir": str(out_dir),
        "labels": str(args.labels) if args.labels else None,
    }
    metadata_path = work_dir / "job_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"\nMetadata: {metadata_path}")

    if args.no_wait:
        print("\nJobs submitted. Run with --recover to merge after completion.")
        return

    wait_for_jobs(job_ids)
    print("\nMerging shard outputs...")
    merge_shard_outputs(sot_dir, out_dir, args.num_shards)
    # Emit Mirai CSV after merge (shards skip this; manifest.parquet exists now)
    if args.labels and args.labels.exists():
        print("\nGenerating Mirai CSV...")
        subprocess.run(
            [
                sys.executable,
                "-u",
                "pipelines/preprocess.py",
                "emit-csv",
                "--raw",
                str(raw_dir),
                "--sot",
                str(sot_dir),
                "--out",
                str(out_dir),
                "--labels",
                str(args.labels),
            ],
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        print(f"  Wrote {out_dir / 'mirai_manifest.csv'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
