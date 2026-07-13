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

For a confirmatory panel, pass each prior browser manifest with
`--exclude-manifest` to prevent exact-view reuse and pass the restricted prior
`source_manifest.parquet` with `--exclude-source-manifest` to exclude every
image from those exams. Whole-exam exclusion prevents correlated views from the
same acquisition leaking between development and holdout panels.

Example:

```bash
python submit_view_auto_qc.py \
  --target 'vertical detector seam' \
  --target-prompt-file qc/targets/vertical_detector_seam_v1.txt \
  ...
```

## Fixed visual examples

Use a small ordered contrastive bank only after a completed development panel
shows that examples answer a specific model-error hypothesis. Build the bank
and a matched evaluation subset from explicit human review positions:

```bash
python qc/build_view_few_shot_experiment.py \
  --campaign-dir /restricted/path/to/completed_development_panel \
  --baseline-run /restricted/path/to/baseline_run.json \
  --out-dir /restricted/path/to/few_shot_experiment \
  --target 'the visual finding being reviewed' \
  --example '3=positive defining appearance' \
  --example '8=negative hard look-alike'
```

The resulting exemplar manifest is a fixed, contiguous order of 2--15
view-level examples and must include both `present` and `absent` labels. Each
row records `view_id`, a canonical relative PNG path, `target`, `label`,
`exemplar_order`, and a concise role. The loader rejects malformed images,
unsafe paths, duplicate examples, target mismatches, and overlap with the
scored manifest.

Pass the frozen bank to inference with `--few-shot-manifest`:

```bash
python submit_view_auto_qc.py \
  --manifest /restricted/path/to/evaluation/manifest.parquet \
  --run-file /restricted/path/to/model_run.json \
  --target 'the visual finding being reviewed' \
  --target-prompt-file qc/targets/example_target_v1.txt \
  --few-shot-manifest /restricted/path/to/exemplars/manifest.parquet \
  ...
```

Every request receives the examples in the manifest order. The saved run
records the exemplar manifest SHA-256 plus each example's image SHA-256, order,
label, and role so a result cannot silently resume with a different reference
bank. Keep example selection inside development data, then freeze it before
scoring an exam-disjoint blinded confirmation panel.
