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
    Generates a fingerprint for an exam directory. This version searches RECURSIVELY
    for files to handle nested directory structures. It also retains retries for
    flaky network filesystems.

    Returns a tuple: (ExamFingerprint | None, reason_string).
    """
    if not exam_path.is_dir():
        return None, "Path is not a directory."

    study_uid = None
    file_hashes = set()

    files_to_process = []
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # --- KEY CHANGE HERE ---
            # Use rglob('*') to find ALL files in ALL subdirectories.
            # This will correctly find the DICOM files in the nested folder.
            all_items = exam_path.rglob("*")
            files_to_process = [
                p for p in all_items if p.is_file() and not p.name.startswith(".")
            ]
            # --- End of Key Change ---

            if files_to_process:
                break

            # If still no files, it might be a true empty dir or a glitch. Retry.
            time.sleep(1 + attempt)
        except (IOError, OSError) as e:
            if attempt < max_retries - 1:
                time.sleep(1 + attempt)
                continue
            else:
                return (
                    None,
                    f"Failed to list directory contents after {max_retries} attempts: {e}",
                )

    if not files_to_process:
        return (
            None,
            "No valid files found (directory appears to be empty or only contains dotfiles).",
        )

    try:
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

        if not study_uid:
            return (
                None,
                f"Processed {len(file_hashes)} file(s), but none contained a valid DICOM StudyInstanceUID.",
            )

        fingerprint = ExamFingerprint(
            study_uid=study_uid, file_hashes=frozenset(file_hashes)
        )
        return fingerprint, "Success"

    except PermissionError as e:
        return None, f"Permission denied while accessing files: {e}"
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"
