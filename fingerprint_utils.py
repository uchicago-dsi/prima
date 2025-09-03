#!/usr/bin/env python
# coding: utf-8
# SHARED LOGIC for exam fingerprinting.

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
            # Read in chunks to handle large files efficiently
            for byte_block in iter(lambda: f.read(8192), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except (IOError, PermissionError) as e:
        # Log this or handle it as needed in the calling script
        raise e


def create_exam_fingerprint(exam_path: Path) -> Optional[ExamFingerprint]:
    """
    Generates a fingerprint for an exam directory by hashing its contents
    and reading the StudyInstanceUID from the first valid DICOM file.
    Returns None if the directory is invalid or a fingerprint cannot be
    created.
    """
    if not exam_path.is_dir():
        return None

    study_uid = None
    file_hashes = set()

    try:
        # Filter for files, ignore dotfiles (like .DS_Store)
        files_to_process = [
            p for p in exam_path.iterdir()
            if p.is_file() and not p.name.startswith('.')
        ]
        if not files_to_process:
            return None  # Skip empty or non-DICOM directories

        for fpath in files_to_process:
            file_hashes.add(hash_file(fpath))
            # Optimization: only try to read DICOM UID if we haven't found
            # it yet
            if not study_uid:
                try:
                    # stop_before_pixels is a crucial optimization
                    dcm = pydicom.dcmread(
                        fpath, stop_before_pixels=True
                    )
                    study_uid = str(dcm.StudyInstanceUID)
                except pydicom.errors.InvalidDicomError:
                    # This file is not a DICOM file, which is fine.
                    # Keep looking.
                    continue

        if not study_uid:
            # This case means we hashed files, but none were valid DICOMs
            # with a UID
            return None

        return ExamFingerprint(
            study_uid=study_uid, file_hashes=frozenset(file_hashes)
        )
    except Exception as e:
        # Catch any other unexpected errors during processing
        # The calling script will log this with the path context
        raise e
