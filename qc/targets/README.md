# Single-target view QC prompts

Each file defines one visual target for one model run. Shared QC code treats the
target only as present or absent. Human low confidence is an orthogonal boolean
flag on that binary decision; it is never a third target label and must be
adjudicated before evaluation or fallback selection.

A target prompt must:

- ask about exactly one visible finding in one mammography view;
- define positive evidence and important negative look-alikes;
- say how borderline appearances should be handled;
- end with the exact `EVIDENCE`, `ANSWER`, `CONFIDENCE`, and `REVIEW` fields
  required by `qc/run_view_auto_qc.py`.

An operational binary target may combine multiple visible disqualifiers only
when they all produce the same downstream action, such as excluding a view from
a standard Mirai input slot. Keep the output binary, freeze the full usability
rubric before review, and report performance within each sampled failure family
so pooled accuracy cannot hide a missed artifact class. DICOM lineage, decoding,
projection, and other deterministic checks remain outside the visual prompt.

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
  --review-rubric-file /path/to/human_rubric.txt \
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
bank. Marker-classifier references receive the same frozen target task as the
scored view, and the concise role is used as their accepted evidence phrase.
Keep example selection inside development data, then freeze it before scoring
an exam-disjoint blinded confirmation panel.

When the learned target is one component of a broader operational exclusion,
keep the component exemplar labels separate from the completed union campaign:

```bash
python qc/build_view_few_shot_experiment.py \
  --campaign-dir /restricted/path/to/completed_union_panel \
  --baseline-run /restricted/path/to/union_baseline.json \
  --out-dir /restricted/path/to/component_experiment \
  --target 'one visual component' \
  --operational-target 'the broader operational union' \
  --example '3=positive defining component appearance' \
  --example-label '3=present' \
  --example '8=negative hard look-alike from another component' \
  --example-label '8=absent'
```

This mode preserves the union labels and baseline for matched evaluation while
building an explicitly adjudicated component-target exemplar bank. Every
example needs exactly one explicit component label, and exemplars are excluded
from the scored manifest. Pass `--excluded-score-manifest` to additionally
require zero exemplar overlap at both view and audit-exam level. When synthetic
component labels are derived from otherwise usable source views, pass
`--require-operational-example-label absent` to enforce that source condition.

## Split targets

When one operational exclusion is the union of visually different findings,
score each component with its own single-target prompt and combine only the
frozen high-confidence decisions. The combiner requires identical manifest
coverage and image paths and records each source run and SHA-256:

```bash
python qc/combine_view_auto_qc_runs.py \
  --manifest /restricted/path/to/manifest.parquet \
  --run-file /restricted/path/to/component_a.json \
  --run-file /restricted/path/to/component_b.json \
  --output /restricted/path/to/combined.json \
  --target 'the operational union target' \
  --minimum-present-confidence high
```

The combined run is derived and contains no new model inference. Freeze the
component prompts, thresholds, and logical rule before scoring a blinded
panel.

## Same-exam context

When a completed development experiment shows that a target view is ambiguous
in isolation, build a target-preserving context manifest rather than replacing
the target path with a montage:

```bash
python qc/build_same_exam_context_views.py \
  --manifest /restricted/path/to/target_manifest.parquet \
  --source-manifest /restricted/path/to/target_sources.parquet \
  --exclusions /restricted/path/to/view_exclusions.parquet \
  --candidates /restricted/path/to/view_candidates.parquet \
  --raw-root /restricted/path/to/dicoms \
  --output-manifest /restricted/path/to/context_manifest.parquet \
  --temp-root /scratch/user/context-render
```

The output retains `image_path` as the canonical target view and adds
`model_image_path` for a labeled composite containing that target plus up to
three deterministic same-exam references. Run it with:

```bash
python submit_view_auto_qc.py \
  --manifest /restricted/path/to/context_manifest.parquet \
  --model-image-column model_image_path \
  ...
```

The run stores the context-manifest digest and model-image column while each
prediction remains keyed to the original target image. This allows evaluation
and logical-OR combination to enforce the same target-view lineage even when a
component model consumes additional visual evidence.

Prompts intended for these composites must explicitly assign the decision to
the large panel labeled `TARGET VIEW` and state that smaller same-exam references
are comparison aids that may themselves contain exclusions. Require target-
specific evidence so a finding visible only in a reference cannot reject the
canonical target. An unchanged single-image prompt may still be frozen as a
one-variable diagnostic arm, but do not assume it will interpret a multi-panel
input correctly or promote it without an explicit target-ownership check.
