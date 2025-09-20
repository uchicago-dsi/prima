#!/usr/bin/env python
# coding: utf-8
# SHARED LOGIC for exam fingerprinting.

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pydicom


@dataclass(frozen=True)
class ExamFingerprint:
    """A unique, content-based identifier for an exam."""

    study_uid: str
    file_hashes: frozenset[str]

    def is_valid(self) -> bool:
        return bool(self.study_uid and self.file_hashes)


def hash_file(filepath: Path) -> str:
    """Calculates the SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(8192), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except (IOError, PermissionError) as e:
        raise e


def create_exam_fingerprint(exam_path: Path) -> Tuple[Optional[ExamFingerprint], str]:
    """
    generate a fingerprint for an exam directory.

    instrumentation:
      - logs file count, approx bytes encountered, and timing breakdown (list vs hash vs dicom uid)
      - still searches recursively to handle nested layouts
    returns (ExamFingerprint|None, reason)
    """
    import logging
    import time

    if not exam_path.is_dir():
        return None, "Path is not a directory."

    t0 = time.perf_counter()

    # list recursively
    try:
        list_start = time.perf_counter()
        all_items = exam_path.rglob("*")
        files_to_process = [
            p for p in all_items if p.is_file() and not p.name.startswith(".")
        ]
        list_s = time.perf_counter() - list_start
    except (IOError, OSError) as e:
        return None, f"Failed to list directory contents: {e}"

    if not files_to_process:
        return (
            None,
            "No valid files found (directory appears to be empty or only contains dotfiles).",
        )

    approx_bytes = 0
    try:
        for p in files_to_process:
            try:
                approx_bytes += p.stat().st_size
            except Exception:
                pass
    except Exception:
        pass

    study_uid = None
    file_hashes = set()

    # hash + uid discovery
    hash_start = time.perf_counter()
    for fpath in files_to_process:
        try:
            file_hashes.add(hash_file(fpath))
        except (IOError, OSError) as e:
            return None, f"Failed to read/hash file {fpath.name}: {e}"

        if not study_uid:
            try:
                dcm = pydicom.dcmread(fpath, stop_before_pixels=True)
                study_uid = str(dcm.StudyInstanceUID)
            except pydicom.errors.InvalidDicomError:
                continue
    hash_s = time.perf_counter() - hash_start

    if not study_uid:
        total_s = time.perf_counter() - t0
        logging.info(
            f"[FPRINT] {exam_path.name}: files={len(files_to_process)}, bytes~{approx_bytes / 1e6:.1f} MB, "
            f"list={list_s:.2f}s, hash={hash_s:.2f}s, total={total_s:.2f}s -> no StudyInstanceUID"
        )
        return (
            None,
            f"Processed {len(file_hashes)} file(s), but none contained a valid DICOM StudyInstanceUID.",
        )

    fingerprint = ExamFingerprint(
        study_uid=study_uid, file_hashes=frozenset(file_hashes)
    )
    total_s = time.perf_counter() - t0
    logging.info(
        f"[FPRINT] {exam_path.name}: files={len(files_to_process)}, bytes~{approx_bytes / 1e6:.1f} MB, "
        f"list={list_s:.2f}s, hash={hash_s:.2f}s, total={total_s:.2f}s -> uid={study_uid}"
    )
    return fingerprint, "Success"
