# Single-target view QC prompts

Each file defines one visual target for one model run. Shared QC code treats the
target only as present or absent; `uncertain` is reserved for human review and
never counts as a target-absent fallback candidate.

A target prompt must:

- ask about exactly one visible finding in one mammography view;
- define positive evidence and important negative look-alikes;
- say how borderline appearances should be handled;
- end with the exact `EVIDENCE`, `ANSWER`, `CONFIDENCE`, and `REVIEW` fields
  required by `qc/run_view_auto_qc.py`.

Pass the same concise target name to `--target` whenever the prompt is used.
Start a new human state and model run when either the target definition or
prompt changes. The run records the exact prompt text and its SHA-256 digest.

Initialize a fresh human review without changing a target-specific builder:

```bash
python qc/init_view_qc_review.py \
  --manifest /path/to/manifest.parquet \
  --state /path/to/view_qc_state.json \
  --target 'the visual finding being reviewed'
```

Initialization creates both `view_qc_state.json` and the append-only
`view_qc_events.jsonl`. Start the reviewer with a stable audit identity:

```bash
python qc/view_qc_gallery.py \
  --manifest /path/to/manifest.parquet \
  --state /path/to/view_qc_state.json \
  --reviewer annawoodard \
  --port 8767
```

Build a diagnostic-view target panel from the durable exclusion pool with
metadata-enriched cases, other diagnostic exclusions as hard negatives, and
standard-view controls:

```bash
python qc/build_view_exclusion_qc_pilot.py \
  --exclusions /path/to/sot/view_exclusions.parquet \
  --standard-views /path/to/sot/views.parquet \
  --raw-root /path/to/raw-dicoms \
  --out-dir /restricted/path/to/pilot \
  --target 'non-standard spot-compression or magnification view' \
  --enrichment-regex 'spot|magnif'
```

The browser reads the deidentified `manifest.parquet`. Exact DICOM lineage is
kept separately in mode-600 `source_manifest.parquet`; enrichment strata are
sampling aids and never reference labels.

Example:

```bash
python submit_view_auto_qc.py \
  --target 'vertical detector seam' \
  --target-prompt-file qc/targets/vertical_detector_seam_v1.txt \
  ...
```
