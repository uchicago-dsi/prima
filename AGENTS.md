# Repository Guidelines

## Top Rules

- Use the `prima` micromamba env for all work except GPU model serving for
  auto-QC, which uses the isolated, pinned `prima-vllm` env from
  `env-vllm.yaml`. Do not install vLLM into the general-purpose `prima` env.
- Fail fast on bad paths, caches, or dependencies. Do not add backward-compatibility shims.
- If a maintained library already provides missing functionality, install it into the `prima` env instead of re-implementing that functionality locally.
- Never commit or print PHI.
- Do not write, expand, restore, or run automated tests or test scripts unless
  Anna explicitly asks. Validate with the cheapest direct check: compile/lint,
  config or dry-run validation, or one representative real-input smoke with
  artifact inspection. If safe validation appears to require a durable
  regression test, explain the specific risk and ask rather than creating one.
- When a runtime or launcher failure on the current path has a clear fix, apply the fix and retry or resubmit automatically before reporting back. Report the fix and the new job state after it has been attempted.

## Project Structure & Module Organization

CLI entrypoints live under `analysis/`, `exports/`, `ops/`, `pipelines/`, `qc/`, `examples/`, and `experiments/`. Shared helpers live in `prima/`; extend those modules instead of cloning code between scripts. Cache inventories sit under `data/`, figures under `plots/`, and external dependencies under `vendor/` (treat the Mirai submodule as read-only unless mirroring upstream).

## Shared Skills

- Shared skill library: `/home/annawoodard/.codex/skills/`
- Treat these as available tools, not always-on context. Use them when the task calls for them rather than preloading them.
- Common shared skills here: `$slurm`, `$job-babysitting`, `$sensemaking`, `$experiment-design`, `$lab-notebook`, `$rut-breaker`, `$handoff`, `$skeptical-labmate`, and `$bounded-auto-loop`.

## Long-Running Automation

- On Randi, access shared persistent priority/nonpreemptible H200 allocations only through `/gpfs/data/huo-lab/Image/annawoodard/hfdp/scripts/shared_h200_pool.sh`: inspect `status`, then use `run --project prima` with the lane, semantic task, broker-enforced maximum runtime, existing run root, and exact command. Never issue raw `srun --jobid` into these allocations; the broker owns atomic GPU auditing, claims, and cross-project fair sharing. When every shared lane is busy, follow the broker's profile-derived opportunistic route for restartable/checkpointed work; non-restartable work waits. Never leave the same task queued in the broker and opportunistic Slurm simultaneously.
- Prefer timer-driven babysitters over ad hoc tmux watchers for ongoing experiments.
- The control plane should be a fresh one-shot process on each tick, not a long-lived shell loop.
- Select jobs by campaign family (for example, regex over run roots or job names), not by hardcoded job IDs.
- Babysitters should make the best bounded decision they can on each tick, document it, and avoid dead time between retries or follow-ons.
- Decision records should include: what action was taken, why, evidence used, files or artifacts touched, and the next check condition.
- Keep durable automation conventions in `AGENTS.md`; keep live campaign state in the notebook or a handoff doc, not here.

## Baseline Retention And Scoped Falsification

- Keep the best valid baseline active until a prospectively defined successor
  beats it on the same decision readouts.
- Attribute failure only to the changed delta. Failure of an additive or rescue
  arm rejects the addition or combination, not unchanged baseline components.
- Before pivoting, record the baseline, delta, result, exact hypothesis
  falsified, hypotheses not falsified, and retained active path in the canonical
  notebook.
- Stop rules must state their exact scope. Do not retire a parent method family
  from failure of a broader prompt, extension, or challenger unless the parent
  itself was directly tested and failed.

## Lab Notebooks

- Use one canonical lab notebook per working environment, not one notebook per thread, conversation, campaign, or narrow experiment.
- On the DSI cluster, append experiment, Slurm, QC, and debugging entries to `logs/lab_notebook_dsi.md`.
- On Randi, append Prima experiment, Slurm, QC, and debugging entries to `logs/lab_notebook_randi.md`. Do not create DSI-local Randi campaign notebooks.
- Keep handoffs under `logs/` short and point back to the canonical notebook for durable chronology.

## Environment Setup

Use the `prima` micromamba environment for all operations:

```bash
eval "$(micromamba shell hook -s bash)"
micromamba activate prima
```

On this host, the named env entry can be stale. If `micromamba activate prima` fails because it looks under `/net/projects/annawoodard/micromamba/envs/prima`, use the explicit prefix instead:

```bash
eval "$(micromamba shell hook -s bash)"
micromamba activate /net/projects2/annawoodard/micromamba/envs/prima
```

For one-off commands, the equivalent non-interactive form is:

```bash
micromamba run -p /net/projects2/annawoodard/micromamba/envs/prima <command>
```

This environment has all required dependencies (torch, pydicom, zarr, pandas, etc.).

For vLLM-backed auto-QC only, create and use the isolated serving environment:

```bash
mkdir -p /scratch/annawoodard/tmp/prima-vllm-pip
mkdir -p /scratch/annawoodard/tmp/prima-vllm-runtime
TMPDIR=/scratch/annawoodard/tmp/prima-vllm-pip \
  micromamba create -y -f env-vllm.yaml
PYTHONNOUSERSITE=1 \
TMPDIR=/scratch/annawoodard/tmp/prima-vllm-pip \
UV_CACHE_DIR=/scratch/annawoodard/uv-cache/prima-vllm \
UV_LINK_MODE=copy \
  /gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima-vllm/bin/uv \
  pip install \
  --python /gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima-vllm/bin/python \
  --torch-backend=cu130 \
  vllm==0.24.0 openai==2.45.0 torch==2.11.0 torchvision==0.26.0 \
  torchaudio==2.11.0 numpy==2.3.5
TMPDIR=/scratch/annawoodard/tmp/prima-vllm-runtime \
micromamba run -p /gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima-vllm \
  python submit_auto_qc.py ...
```

The submitter deliberately fails before queueing if the active environment does
not contain the pinned vLLM and OpenAI client versions.
The executable scratch `TMPDIR` is required for both installation and serving
because `/tmp` is mounted `noexec`; Triton loads compiled runtime kernels from
that directory. The submitter fails before queueing if `TMPDIR` is unsuitable.
`env-vllm.yaml` also supplies the pinned CUDA 13.0 `nvcc` required by
DeepGEMM; do not rely on the compute image's `/usr/local/cuda` compiler. The
managed server sets `CUDA_HOME` to the active environment and puts FlashInfer's
JIT workspace under `TMPDIR`.
The explicit `cu130` backend is required because automatic backend detection on
a GPU-less login node installs CPU-only Torch.

## Build And Development Commands

Create the micromamba env once with `micromamba create -y -f env.yaml`, then `micromamba activate prima`, `pip install -e .`, `pip install -r requirements.txt`, and `pip install -r requirements-dev.txt` for linting/notebook extras. Scripts expose CLI help; run `python ops/fingerprinter.py --help` or `python ops/sync.py --dry-run` before touching production mounts.

Typical local validation after refactors:

```bash
python -m compileall prima analysis exports ops pipelines qc examples experiments
ruff format .
ruff check --fix .
```

## Data Sources & Pipeline

### Three Data Sources

1. **iBroker metadata** (`data/imaging_metadata.csv`) tracks exports from hospital PACS. Use it for sync and export tracking only, not for preprocessing or training.
2. **Disk DICOMs** (`/gpfs/data/huo-lab/Image/ChiMEC/MG/`) are the actual imaging data used by preprocessing.
3. **Phenotype labels** (`Phenotype_ChiMEC_*.csv`) provide case/control status and diagnosis dates for training labels.

### Key Insight

**Preprocessing depends on disk DICOMs plus phenotype labels, not on iBroker metadata.** Historical data on disk but missing from iBroker can still be used for training as long as patients have phenotype labels. The disk fingerprint cache in `data/destination_fingerprints.json` tracks what is actually on disk.

### Pipeline Flow

```text
Disk DICOMs → pipelines/preprocess.py → SoT tables (views.parquet, exams.parquet)
                           → Zarr cache + manifest.parquet
                           → emit-csv joins with phenotype → mirai_manifest.csv
                           → pipelines/run_mirai_sharded.py → predictions
```

### Running Analysis

```bash
# analyze metadata and show data coverage summary
python analysis/analyze_metadata.py --modality MG

# see DATA SOURCES SUMMARY for:
# - current training data (on disk with labels)
# - remaining to download (in iBroker but not on disk)
# - historical data (on disk but not in iBroker, most have labels)
```

## Cache And Data Discipline

- Persist only authoritative metadata. Update producers and consumers together when schemas change.
- When cache formats or preprocessing parameters change, rebuild caches end-to-end instead of supporting multiple formats.
- Keep cache JSON and exported logs under `data/` out of version control unless scrubbed.
- Drop legacy aliases instead of keeping passive support for multiple schemas.

## Coding Style & Naming Conventions

Keep configuration in module-level constants or argparse defaults. Do not scatter hidden fallbacks across call sites. Follow PEP 8 with 4-space indentation, snake_case functions, CamelCase classes, and ALL_CAPS constants. Prefer `pathlib.Path`, structured logging, and concise comments. Favor vectorized NumPy or PyTorch utilities for volume work.

## Handoffs

- `AGENTS.md` is for stable repo rules and operating conventions, not current run state.
- For restarts, write a short handoff under `logs/` with the goal, current state, evidence paths, open uncertainty, and the exact first re-entry check.
- Update the canonical notebook or experiment log before writing the handoff so a fresh agent can trust it.
- When writing a handoff, delete every stale handoff under `logs/` in the same
  change. Exactly one current handoff may exist. A superseded handoff is worse
  than none: a fresh agent cannot tell which of several files is live, and the
  durable chronology already lives in the canonical notebook.
- Name the handoff for its scope, not its date, so the current one is always
  found at the same path.

## Commit & Pull Request Guidelines

Use short imperative commit subjects and land on `main` unless coordination demands a PR. Always run `ruff format .` and `ruff check --fix .` before staging changes. Bundle related code, cache notes, and environment tweaks together. If a PR is opened, mirror the commit summary, list datasets exercised, attach relevant plots, and reference tracking issues.

## Reproducibility
Make outputs easy to recreate without relying on shell history or memory.

- When generating analysis artifacts or derived tables, write a short `README.md` or provenance text file in the output directory.
- Record the exact command, key input paths, output paths, date, and any important environment assumptions.
- If an output depends on multiple sequential commands, record them in order.
- When changing a producer script, regenerate downstream derived files that depend on it or note clearly that they are stale.
- Prefer deterministic scripts and explicit CLI arguments over one-off notebook state or ad hoc shell edits.
- Do not add date tags to filenames or directory names unless the date is truly part of the scientific meaning. For most outputs, dated suffixes become meaningless later and make stable paths harder to maintain.

## Key References

- Omoleye / Woodard / Huo Radiology: AI Mirai validation paper reference is recorded at:
  - [docs/papers/omoleye_2023_ryai_220299_reference.md](/gpfs/data/huo-lab/Image/annawoodard/prima/docs/papers/omoleye_2023_ryai_220299_reference.md)
- This note stores the exact citation, DOI, RSNA full-text URL, and the paper-era cohort/results used in CHiMEC Mirai debugging.

## No Backward Compatibility

**Do not add backward compatibility shims.** This is research code. When data formats change, delete old caches and regenerate rather than adding conditional logic to support multiple versions.

## Security & Data Handling

Never commit PHI or log it to stdout. Keep cache JSON and exported logs under `data/` out of version control unless scrubbed, and double-check destructive flags before touching hospital shares. Coordinate VPN, credential rotations, and mount path changes in lab channels so automation and sync jobs stay reproducible.

## Mirai Evaluation

- `analysis/analyze_mirai.py` should aggregate predictions per exam, not per view.
- Exam-level aggregation means:
  - mean predictions across views for each `(patient_id, exam_id)`
  - labels taken once per exam
  - AUC and survival metrics computed on exam-level rows

## analysis/analyze_mirai.py

`analysis/analyze_mirai.py` computes per-horizon AUC and survival metrics (Uno's C-index, time-dependent AUC, integrated Brier score) from Mirai validation outputs.

### Key Design: Per-Exam Aggregation

**Critical**: Mirai predictions must be evaluated per exam, not per view. The validation output CSV contains one row per view, but Mirai's risk predictions are intended to aggregate information across all views for a single exam-level prediction.

**Aggregation logic**:

- **Predictions**: mean across all views for each `(patient_id, exam_id)` pair
- **Labels**: `years_to_cancer` and `years_to_last_followup` are identical across views for the same exam, so take the first value
- **AUC calculation**: performed on exam-level aggregated predictions, not per-view

This ensures that:

1. Each exam contributes exactly one prediction to the AUC calculation.
2. Evaluation matches how Mirai is intended to be used clinically.
3. Multiple views per exam do not artificially inflate sample sizes.

### Functions

- **`summarize()`** computes per-horizon binary AUC. Aggregate predictions per exam before calculating metrics.
- **`survival_metrics()`** computes censoring-adjusted survival metrics using all available data for IPCW censoring estimation. Aggregate per exam here too.
- **`kfold_survival_metrics()`** runs K-fold cross-validation for IPCW sensitivity analysis. Split at patient level to avoid leakage and aggregate per exam within each fold.

### Input Format

Expect `validation_output.csv` and `mirai_manifest.csv` in the same directory (specified via `--out-dir`).

Predictions CSV requirements:

- `patient_id` and `exam_id` columns, or `patient_exam_id` with tab separator
- risk prediction columns auto-detected via regex such as `1_year_risk` and `2_year_risk`

Metadata CSV requirements:

- `patient_id`, `exam_id`, `years_to_cancer`, `years_to_last_followup`
- optional `split_group` for filtering evaluation sets

## Plotting Guidelines

Never add titles to plots unless the meaning is not obvious from the axis labels. Axis labels should be descriptive enough to convey the plot's purpose. Remove grid lines unless they significantly aid readability.
