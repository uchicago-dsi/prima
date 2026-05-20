## Archive-Aware MG Preprocess Design

This note sketches how to make `pipelines/preprocess.py` work with a mixed raw/archive DICOM tree while extracting each archived exam only once per preprocess run.

### Goals

- Keep raw DICOM as the source material for occasional reprocessing and eventual public release.
- Allow a mixed tree where some exams are present as raw directories and others only as compressed archives.
- Default behavior after successful processing should be to archive/compress raw exams.
- Avoid a persistent state manifest if possible.
- Omit explicit `missing` handling for now.

### Current Dataflow

Today the preprocessing path is:

1. `discover_dicoms(raw_dir, ...)`
   - scans `raw_dir/patient_id/exam_id/`
   - runs `_process_exam_dir()` in parallel
   - emits:
     - `views_df`
     - `tags_df`
2. `select_full_quad(views_df)`
   - keeps only presentation 4-view exams
3. `preprocess()`
   - writes SoT tables:
     - `views.parquet`
     - `exams.parquet`
     - `dicom_tags.parquet`
     - `cohort.parquet`
4. `preprocess()`
   - groups selected views by `exam_id`
   - calls `write_exam_zarr()` for each selected exam
   - writes:
     - `out/zarr/...`
     - `manifest.parquet`

### Important Constraint

The current code reads raw exam directories in two logically separate phases:

- first for DICOM discovery / SoT building
- later again for zarr writing

That means a naive archive-aware wrapper would extract an archived exam twice:

1. once so `discover_dicoms()` can inspect the DICOM metadata
2. again so `write_exam_zarr()` can read pixels

If we want "extract once", we need to refactor the control flow.

### Recommended State Model

For now, use only:

- `raw`
- `archived`

State is computed on the fly from deterministic filesystem layout:

- raw exam directory exists:
  - `MG/<patient_id>/<exam_id>/`
- else archive exists:
  - `MG/<patient_id>/<exam_id>.tar.zst`

No persistent manifest is required for state.

### Why No Persistent Manifest

The user concern is valid: a persistent state manifest can drift out of sync with the filesystem.

For this workflow, state is simple and cheap to compute per exam:

- `raw` if the raw exam directory exists
- `archived` if the raw directory is absent but the archive file exists

The filesystem should remain the source of truth.

### What Still Needs Raw Materialization

Archive filenames are not enough for preprocessing.

The SoT tables need metadata from inside the DICOM files, including at least:

- laterality
- view
- SOP instance UID
- rows / cols
- photometric interpretation
- bits stored
- marked-up / implant / presentation flags
- manufacturer / model
- study date

And zarr writing needs pixel data.

So archived exams must still be materialized to a temporary raw directory before processing.

### Design Principle: Materialize Once Per Exam

For each exam selected for processing:

1. resolve state
2. if archived, extract to a staging directory
3. build SoT rows from that materialized directory
4. write zarr from that same materialized directory
5. archive/re-archive and clean up

That avoids double extraction.

### Proposed Refactor

#### 1. Introduce an exam record type

The pipeline should explicitly represent one discovered exam:

```python
@dataclass
class ExamRecord:
    patient_id: str
    exam_id: str
    raw_exam_dir: Path
    archive_path: Path
    state: Literal["raw", "archived"]
```

This object is enough to resolve where to read from and where to write the archive.

#### 2. Replace raw-dir-only discovery with mixed discovery

Current:

- `discover_dicoms()` enumerates only `patient/exam/` directories

Proposed:

- `discover_exam_records(raw_root)` enumerates:
  - raw exam directories: `patient/exam/`
  - archived exam files: `patient/exam.tar.zst`
- merges them by `(patient_id, exam_id)`
- prefers `raw` if both raw and archive happen to exist

This produces a list of `ExamRecord`s.

#### 3. Split `_process_exam_dir()` into metadata extraction from a materialized exam

Current worker API:

```python
_process_exam_dir(exam_path, debug_dir)
```

Proposed worker API:

```python
_process_materialized_exam_dir(materialized_exam_dir, patient_id, exam_id, debug_dir)
```

This keeps the DICOM parsing logic largely unchanged, but makes it independent of where the exam originally came from.

#### 4. Add materialization helpers

Add small helpers:

```python
def archive_path_for_exam(raw_exam_dir: Path) -> Path:
    ...

def materialize_exam(record: ExamRecord, staging_root: Path) -> tuple[Path, bool]:
    """Return (materialized_exam_dir, extracted_from_archive)."""

def archive_exam_dir(materialized_exam_dir: Path, archive_path: Path) -> None:
    ...

def finalize_exam_materialization(
    record: ExamRecord,
    materialized_exam_dir: Path,
    extracted_from_archive: bool,
    archive_after: bool = True,
) -> None:
    ...
```

Behavior:

- `raw`:
  - materialized dir is the raw exam dir itself
- `archived`:
  - extract archive into staging
  - materialized dir points to staged raw exam dir

Finalization:

- if source was `raw` and processing succeeded:
  - archive the raw exam dir to `exam.tar.zst`
  - remove the raw exam dir
- if source was `archived`:
  - re-archive only if needed
  - remove staged extraction

#### 5. Collapse discovery + zarr into one per-exam processing loop

This is the main structural change required for "extract once".

Instead of:

- pass A: scan all exams into `views_df`
- pass B: regroup selected exams and reread them for zarr

Use:

1. discover `ExamRecord`s
2. for each exam:
   - materialize once
   - extract metadata rows
3. after full-quad selection is known, for each selected exam:
   - reuse the same materialized directory if still available
   - or more realistically, perform metadata extraction and zarr writing in the same exam-level worker result

There are two implementation options.

### Option A: True one-pass per-exam processing

Per worker:

1. materialize exam
2. parse DICOM metadata for all views
3. decide whether exam is a valid full quad
4. if valid:
   - write zarr immediately
   - emit both metadata rows and manifest rows
5. archive / cleanup

Pros:

- archived exam extracted exactly once
- no second pixel read pass
- simplest runtime behavior

Cons:

- bigger refactor
- zarr writing happens before the global SoT tables are finalized
- less separation between "indexing" and "cache writing"

### Option B: Two-phase logic with retained staging

1. materialize all archived exams into staging
2. run discovery / selection
3. write zarr from staged/raw directories
4. cleanup/archive

Pros:

- smaller logical change to existing code

Cons:

- requires enough staging space for many exams at once
- less attractive operationally

### Recommendation

Use **Option A**.

It is the cleanest way to guarantee:

- archived exams are extracted once
- mixed raw/archive works naturally
- default archive-after-processing behavior is easy to enforce

### Concrete New Outputs Per Exam

Each exam worker should return a structured result like:

```python
{
    "exam_status": "success" | "failed",
    "rows": [...],          # SoT view rows
    "tag_rows": [...],      # dicom_tags rows
    "exam_row": {...},      # exams.parquet row
    "manifest_rows": [...], # 4 rows if zarr written
    "total_dicoms": int,
    "valid_dicoms": int,
    "for_presentation_dicoms": int,
    "failed_files": int,
    "has_four_views": bool,
}
```

Then `preprocess()` only has to concatenate results and write the parquet outputs.

### Archive Format

Prefer exam-level `tar.zst`:

- fast enough for frequent use
- much faster than `xz`
- good compression for large imaging trees

Recommended naming:

- raw:
  - `MG/<patient_id>/<exam_id>/`
- archive:
  - `MG/<patient_id>/<exam_id>.tar.zst`

### Logging / Provenance

Avoid a state manifest, but keep an append-only operational log, for example:

- exam processed
- source state (`raw` or `archived`)
- archive created or refreshed
- raw removed
- timestamp
- failure reason if any

This is not the source of truth; it is only provenance and recovery aid.

### Minimal First Implementation

1. add mixed discovery of raw dirs plus `*.tar.zst`
2. add per-exam materialization helpers
3. refactor exam worker so one materialized exam can produce:
   - metadata rows
   - zarr
4. default to archive-after-success
5. omit `missing` until a targeted rerun mode truly needs it

### What Not To Do

- do not encode SoT metadata in archive filenames
- do not require full-tree decompression before preprocess
- do not add a persistent mutable state manifest unless performance later proves on-the-fly discovery is too slow
