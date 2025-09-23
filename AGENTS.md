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

## Security & Data Handling
Never commit PHI or log it to stdout. Keep cache JSON and exported logs under `data/` out of version control unless scrubbed, and double-check destructive flags before touching hospital shares. Coordinate VPN, credential rotations, and mount path changes in lab channels so automation and sync jobs stay reproducible.
