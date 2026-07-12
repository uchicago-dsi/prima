#!/usr/bin/env python3
"""Initialize a blinded single-target review from any valid view manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from prima.view_qc import (
    empty_view_qc_state,
    normalize_view_qc_target,
    save_view_qc_state,
)
from qc.view_qc_gallery import load_review_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--target", required=True)
    return parser.parse_args()


def initialize_review(manifest_path: Path, state_path: Path, target: str) -> int:
    """Validate a review manifest and create its target-bound empty state."""
    manifest_path = manifest_path.resolve()
    state_path = state_path.resolve()
    target = normalize_view_qc_target(target)
    if state_path.exists():
        raise FileExistsError(
            f"refusing to replace existing view QC state: {state_path}"
        )
    items, _images = load_review_items(manifest_path)
    save_view_qc_state(state_path, empty_view_qc_state(target))
    return len(items)


def main() -> int:
    args = parse_args()
    count = initialize_review(args.manifest, args.state, args.target)
    print(
        f"initialized {count} views for target {normalize_view_qc_target(args.target)!r}"
    )
    print(f"state: {args.state.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
