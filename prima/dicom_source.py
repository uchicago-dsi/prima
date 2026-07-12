"""Durable source locators for DICOMs stored raw or in exam archives."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Mapping, Optional

import pydicom
from pydicom.dataset import FileDataset

ARCHIVE_SUFFIX = ".tar.zst"
SOURCE_ARCHIVE_COLUMN = "source_archive_relpath"
SOURCE_MEMBER_COLUMN = "source_archive_member"
SOURCE_COLUMNS = (
    SOURCE_ARCHIVE_COLUMN,
    SOURCE_MEMBER_COLUMN,
    "sop_instance_uid",
    "sha256",
)


class DicomSourceError(RuntimeError):
    """Raised when a durable DICOM source cannot be resolved exactly."""


def _relative_posix_path(value: object, field: str) -> PurePosixPath:
    text = str(value).strip()
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or path.as_posix() != text
    ):
        raise DicomSourceError(f"{field} must be a normalized relative path")
    return path


@dataclass(frozen=True)
class DicomSource:
    """A stable reference to one DICOM within an exam-level ``tar.zst``."""

    archive_relpath: PurePosixPath
    archive_member: PurePosixPath
    sop_instance_uid: str = ""
    sha256: str = ""

    def __post_init__(self) -> None:
        archive_text = self.archive_relpath.as_posix()
        if not archive_text.endswith(ARCHIVE_SUFFIX):
            raise DicomSourceError(
                f"{SOURCE_ARCHIVE_COLUMN} must end with {ARCHIVE_SUFFIX}"
            )
        exam_name = self.archive_relpath.name[: -len(ARCHIVE_SUFFIX)]
        if not self.archive_member.parts or self.archive_member.parts[0] != exam_name:
            raise DicomSourceError(
                f"{SOURCE_MEMBER_COLUMN} must begin with its archive exam directory"
            )
        if not self.sop_instance_uid or self.sop_instance_uid.lower() in {
            "nan",
            "none",
        }:
            raise DicomSourceError("sop_instance_uid is required")
        if not self.sha256:
            raise DicomSourceError("sha256 is required")
        if len(self.sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.sha256.lower()
        ):
            raise DicomSourceError("sha256 must be a 64-character hexadecimal digest")

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> "DicomSource":
        missing = [column for column in SOURCE_COLUMNS if column not in row]
        if missing:
            raise DicomSourceError(
                "DICOM source row is missing required columns: " + ", ".join(missing)
            )
        return cls(
            archive_relpath=_relative_posix_path(
                row[SOURCE_ARCHIVE_COLUMN], SOURCE_ARCHIVE_COLUMN
            ),
            archive_member=_relative_posix_path(
                row[SOURCE_MEMBER_COLUMN], SOURCE_MEMBER_COLUMN
            ),
            sop_instance_uid=str(row.get("sop_instance_uid", "") or "").strip(),
            sha256=str(row.get("sha256", "") or "").strip().lower(),
        )

    @property
    def source_id(self) -> str:
        """Return a non-identifying key suitable for logs and exceptions."""
        value = f"{self.archive_relpath}!{self.archive_member}".encode()
        return hashlib.sha256(value).hexdigest()[:12]

    def archive_path(self, raw_root: Path) -> Path:
        return Path(raw_root).joinpath(*self.archive_relpath.parts)

    def unpacked_path(self, raw_root: Path) -> Path:
        return Path(raw_root).joinpath(
            *self.archive_relpath.parent.parts,
            *self.archive_member.parts,
        )


def require_source_columns(columns: Iterable[str], context: str) -> None:
    """Fail when a table does not use the current durable source schema."""
    available = set(columns)
    missing = [column for column in SOURCE_COLUMNS if column not in available]
    if missing:
        raise DicomSourceError(
            f"{context} is missing durable DICOM source columns: {', '.join(missing)}; "
            "rebuild it with the current preprocessor"
        )
    if "dicom_path" in available:
        raise DicomSourceError(
            f"{context} still contains removed transient column dicom_path; rebuild it"
        )


def require_valid_sources(rows: Iterable[Mapping[str, object]], context: str) -> None:
    """Fail when any table row has an invalid or incomplete source locator."""
    for ordinal, row in enumerate(rows):
        try:
            DicomSource.from_row(row)
        except DicomSourceError as error:
            raise DicomSourceError(
                f"{context} contains an invalid DICOM source at row {ordinal}: {error}"
            ) from error


def source_path_in_materialization(source: DicomSource, root: Path) -> Path:
    return Path(root).joinpath(*source.archive_member.parts)


@contextmanager
def materialize_dicom_sources(
    sources: Iterable[DicomSource],
    raw_root: Path,
    temp_root: Optional[Path] = None,
) -> Iterator[dict[str, Path]]:
    """Yield paths for sources from one archive, extracting requested members once."""
    unique_sources = {source.archive_member.as_posix(): source for source in sources}
    if not unique_sources:
        raise DicomSourceError("cannot materialize an empty DICOM source collection")

    archive_relpaths = {source.archive_relpath for source in unique_sources.values()}
    if len(archive_relpaths) != 1:
        raise DicomSourceError("one materialization may reference only one archive")

    raw_root = Path(raw_root)
    unpacked = {
        member: source.unpacked_path(raw_root)
        for member, source in unique_sources.items()
    }
    if all(path.is_file() for path in unpacked.values()):
        yield unpacked
        return

    source = next(iter(unique_sources.values()))
    archive_path = source.archive_path(raw_root)
    if not archive_path.is_file():
        raise DicomSourceError(
            f"DICOM source {source.source_id} has neither unpacked files nor an archive"
        )

    if temp_root is not None:
        temp_root = Path(temp_root)
        if not temp_root.is_dir():
            raise DicomSourceError("DICOM materialization root does not exist")

    with tempfile.TemporaryDirectory(
        prefix="prima-dicom-", dir=str(temp_root) if temp_root else None
    ) as tmp_name:
        tmp_path = Path(tmp_name)
        members = sorted(unique_sources)
        result = subprocess.run(
            [
                "tar",
                "-I",
                "zstd",
                "-xf",
                str(archive_path),
                "-C",
                str(tmp_path),
                "--",
                *members,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise DicomSourceError(
                f"failed to extract DICOM source archive {source.source_id}"
            )

        materialized = {
            member: source_path_in_materialization(item, tmp_path)
            for member, item in unique_sources.items()
        }
        missing = [
            member for member, path in materialized.items() if not path.is_file()
        ]
        if missing:
            raise DicomSourceError(
                f"archive {source.source_id} is missing {len(missing)} requested members"
            )
        yield materialized


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_materialized_source(
    source: DicomSource,
    path: Path,
    dataset: FileDataset,
    verify_sha256: bool = False,
) -> None:
    """Verify that a materialized member is the DICOM named by its SoT row."""
    actual_sop_uid = str(dataset.get("SOPInstanceUID", "")).strip()
    if source.sop_instance_uid and actual_sop_uid != source.sop_instance_uid:
        raise DicomSourceError(
            f"DICOM source {source.source_id} has an unexpected SOP Instance UID"
        )
    if verify_sha256:
        if not source.sha256:
            raise DicomSourceError(
                f"DICOM source {source.source_id} has no SHA-256 for verification"
            )
        if _hash_file(path) != source.sha256:
            raise DicomSourceError(
                f"DICOM source {source.source_id} failed SHA-256 verification"
            )


def read_dicom_source(
    row: Mapping[str, object],
    raw_root: Path,
    *,
    stop_before_pixels: bool = False,
    verify_sha256: bool = False,
    temp_root: Optional[Path] = None,
) -> FileDataset:
    """Read and validate one DICOM from the current durable source schema."""
    source = DicomSource.from_row(row)
    with materialize_dicom_sources([source], raw_root, temp_root=temp_root) as paths:
        path = paths[source.archive_member.as_posix()]
        dataset = pydicom.dcmread(
            str(path), force=True, stop_before_pixels=stop_before_pixels
        )
        validate_materialized_source(
            source,
            path,
            dataset,
            verify_sha256=verify_sha256,
        )
        return dataset
