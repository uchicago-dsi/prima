# Repository Guidelines

## Top Rules

- Use the `prima` micromamba env for all work.
- Fail fast on bad paths, caches, or dependencies. Do not add backward-compatibility shims.
- If a maintained library already provides missing functionality, install it into the `prima` env instead of re-implementing that functionality locally.
- Never commit or print PHI.
- When a runtime or launcher failure on the current path has a clear fix, apply the fix and retry or resubmit automatically before reporting back. Report the fix and the new job state after it has been attempted.

## Project Structure & Module Organization

CLI entrypoints live under `analysis/`, `exports/`, `ops/`, `pipelines/`, `qc/`, `examples/`, and `experiments/`. Shared helpers live in `prima/`; extend those modules instead of cloning code between scripts. Cache inventories sit under `data/`, figures under `plots/`, and external dependencies under `vendor/` (treat the Mirai submodule as read-only unless mirroring upstream).

## Shared Skills

- Shared skill library: `/home/annawoodard/.codex/skills/`
- Treat these as available tools, not always-on context. Use them when the task calls for them rather than preloading them.
- Common shared skills here: `$slurm`, `$job-babysitting`, `$sensemaking`, `$experiment-design`, `$lab-notebook`, `$rut-breaker`, `$handoff`, `$skeptical-labmate`, and `$bounded-auto-loop`.

## Long-Running Automation

- Prefer timer-driven babysitters over ad hoc tmux watchers for ongoing experiments.
- The control plane should be a fresh one-shot process on each tick, not a long-lived shell loop.
- Select jobs by campaign family (for example, regex over run roots or job names), not by hardcoded job IDs.
- Babysitters should make the best bounded decision they can on each tick, document it, and avoid dead time between retries or follow-ons.
- Decision records should include: what action was taken, why, evidence used, files or artifacts touched, and the next check condition.
- Keep durable automation conventions in `AGENTS.md`; keep live campaign state in the notebook or a handoff doc, not here.

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

## Build, Test, and Development Commands

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

## Testing Guidelines

There is no automated unit suite. Validate with targeted dry runs (for example, `python ops/fingerprinter.py --patients 1234 --max-workers 2`). When adjusting fingerprint rules, sync heuristics, or cache schemas, delete affected caches and regenerate them. Mixed-version caches are unsupported.

## Handoffs

- `AGENTS.md` is for stable repo rules and operating conventions, not current run state.
- For restarts, write a short handoff under `logs/` with the goal, current state, evidence paths, open uncertainty, and the exact first re-entry check.
- Update the canonical notebook or experiment log before writing the handoff so a fresh agent can trust it.

## Commit & Pull Request Guidelines

Use short imperative commit subjects and land on `main` unless coordination demands a PR. Always run `ruff format .` and `ruff check --fix .` before staging changes. Bundle related code, cache notes, and environment tweaks together. If a PR is opened, mirror the commit summary, list datasets exercised, attach relevant plots, and reference tracking issues.

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
