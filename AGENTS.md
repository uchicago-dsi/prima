# Repository Guidelines

## Project Structure & Module Organization
Root scripts (`fingerprinter.py`, `sync.py`, `export.py`, `download_data.py`) drive data movement and fingerprinting. Shared helpers (`fingerprint_utils.py`, `filesystem_utils.py`, `analyze_mirai.py`) centralize IO, hashing, and plotting—extend these instead of cloning code. The lightweight package stub lives in `prima/`; cache inventories sit under `data/`, figures under `plots/`, and external dependencies under `vendor/` (treat the Mirai submodule as read-only unless mirroring upstream).

## Build, Test, and Development Commands
Create the micromamba env once with `micromamba create -n prima python=3.11`, then `micromamba activate prima`, `pip install -e .`, `pip install -r requirements.txt`, and `pip install -r requirements-dev.txt` for linting/notebook extras. Scripts expose CLI help; run `python fingerprinter.py --help` or `python sync.py --dry-run` before touching production mounts. After refactors, execute `python -m compileall prima fingerprint_utils.py filesystem_utils.py` as a quick syntax check, then format with `ruff format .` followed by `ruff check --fix .`.

## Coding Style & Naming Conventions
Keep configuration in module-level constants or argparse defaults—no hidden fallbacks scattered across call sites. Follow PEP 8 with 4-space indentation, snake_case functions, CamelCase classes, and ALL_CAPS constants (see `sync.py`). Prefer `pathlib.Path`, structured logging, and concise comments; document tensor or array shapes only when needed. Favor vectorized NumPy/PyTorch utilities for volume work.

## Testing Guidelines
There is no automated unit suite. Validate with targeted dry runs (e.g., `python fingerprinter.py --patients 1234 --max-workers 2`). When adjusting fingerprint rules, sync heuristics, or cache schemas, delete the affected directories in `data/fingerprint_checkpoints/` and regenerate—mixed-version caches are unsupported. Capture manual validation steps in commit notes so others can replay them.

## Commit & Pull Request Guidelines
Use short imperative commit subjects (`add exported info to plotting`, `fastpath first syncing`) and land on `main` unless coordination demands a PR. Always run `ruff format .` and `ruff check --fix .` before staging changes. Bundle related code, cache notes, and environment tweaks together. If a PR is opened, mirror the commit summary, list datasets exercised, attach relevant plots, and reference Jira/GitHub tracking.

## Configuration & Cache Discipline
Persist only the authoritative metadata (`study_uid`, hashed file lists, cache manifests`) to JSON. Update producers and consumers together when adding keys; drop legacy aliases instead of keeping passive support. For new preprocessing parameters or remote paths, update the defining constants, rebuild caches end-to-end, and verify outputs via `sync.log` or plot diffs.

## No Backward Compatibility
**Do not add backward compatibility shims.** This is research code—simplicity is more valuable than compatibility. When data formats change (e.g., fingerprint cache schema), delete old caches and regenerate rather than adding conditional logic to handle multiple formats. The added complexity of backward compatibility is not worth it; we can always nuke and rerun.

## Security & Data Handling
Never commit PHI or log it to stdout. Keep cache JSON and exported logs under `data/` out of version control unless scrubbed, and double-check destructive flags before touching hospital shares. Coordinate VPN, credential rotations, and mount path changes in lab channels so automation and sync jobs stay reproducible.

## analyze_mirai.py: Mirai Prediction Analysis

`analyze_mirai.py` computes per-horizon AUC and survival metrics (Uno's C-index, time-dependent AUC, integrated Brier score) from Mirai's validation outputs.

### Key Design: Per-Exam Aggregation

**Critical**: Mirai predictions must be evaluated **per exam**, not per view. The validation output CSV contains one row per view (typically 4 views per exam: L CC, L MLO, R CC, R MLO), but Mirai's risk predictions are designed to aggregate information across all views for a single exam-level prediction.

**Aggregation logic**:
- **Predictions**: Mean across all views for each (patient_id, exam_id) pair
- **Labels**: `years_to_cancer` and `years_to_last_followup` are identical across views for the same exam, so we take the first value
- **AUC calculation**: Performed on exam-level aggregated predictions, not per-view

This ensures that:
1. Each exam contributes exactly one prediction to the AUC calculation
2. The evaluation matches how Mirai is intended to be used clinically (exam-level risk assessment)
3. Multiple views per exam don't artificially inflate sample sizes

### Functions

- **`summarize()`**: Computes per-horizon binary AUC (cases vs controls at each horizon). Aggregates predictions per exam before calculating metrics.
- **`survival_metrics()`**: Computes censoring-adjusted survival metrics using all available data for IPCW censoring estimation. Also aggregates per exam.
- **`kfold_survival_metrics()`**: K-fold cross-validation for IPCW sensitivity analysis. Splits at patient level (not exam level) to avoid leakage. Aggregates per exam within each fold.

### Input Format

Expects `validation_output.csv` and `mirai_manifest.csv` in the same directory (specified via `--out-dir`). The predictions CSV should have:
- `patient_id` and `exam_id` columns (or `patient_exam_id` with tab separator)
- Risk prediction columns auto-detected via regex (e.g., `1_year_risk`, `2_year_risk`, etc.)

The metadata CSV should have:
- `patient_id`, `exam_id`, `years_to_cancer`, `years_to_last_followup`
- Optional: `split_group` for filtering evaluation sets

## Plotting Guidelines

Never add titles to plots unless the meaning isn't obvious from the axis labels. Axis labels should be descriptive enough to convey the plot's purpose. Remove grid lines unless they significantly aid readability.
