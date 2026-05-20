from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def audit_metadata(name: str, path: Path) -> dict:
    df = pd.read_csv(path, dtype={"patient_id": str, "exam_id": str})
    view_counts = (
        df.groupby(["patient_id", "exam_id"])
        .size()
        .value_counts()
        .sort_index()
        .to_dict()
    )
    by_view = (
        df.assign(
            view_name=df["laterality"].astype(str) + " " + df["view"].astype(str)
        )["view_name"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    sample_paths = df["file_path"].head(4).tolist()
    return {
        "name": name,
        "path": str(path),
        "rows": len(df),
        "unique_exams": df[["patient_id", "exam_id"]].drop_duplicates().shape[0],
        "unique_patients": df["patient_id"].nunique(),
        "columns": list(df.columns),
        "view_count_histogram": view_counts,
        "view_name_counts": by_view,
        "sample_paths": sample_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--metadata",
        action="append",
        nargs=2,
        metavar=("NAME", "CSV"),
        required=True,
        help="Repeatable pair of logical name and metadata csv path",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = [audit_metadata(name, Path(csv)) for name, csv in args.metadata]
    pd.DataFrame(rows).to_json(out, orient="records", indent=2)
    print(out)


if __name__ == "__main__":
    main()
