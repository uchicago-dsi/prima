#!/usr/bin/env python3
"""Join Mirai validation_output.csv to clinical columns without re-running inference.

Mirai writes one row per exam with patient_exam_id = "<patient_id>\\t<exam_id>" and
N_year_risk columns (see vendor/mirai/scripts/main.py). Sharded jobs collate into
the final CSV via run_mirai_sharded.collate_results.

This script merges:
  - exams.parquet (study_date, device_manufacturer, device_model)
  - mirai_manifest.csv (years_to_cancer, years_to_last_followup, split_group)
  - study-16352a.csv (AnonymousID -> MRN)
  - phenotype CSV (datedx, subtype, receptors, demographics, etc.)
  - optional CRDW last-contact extract (MRN, DATE_OF_LAST_CONTACT)

example::

    python enrich_mirai_validation_output.py \\
        --pred /gpfs/data/huo-lab/Image/ChiMEC/MG/out/validation_output.csv \\
        --out /gpfs/data/huo-lab/Image/ChiMEC/MG/out/validation_output_clinical.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PHENOTYPE = Path(
    "/gpfs/data/phs/groups/Projects/Huo_projects/SPORE/annawoodard/Phenotype_ChiMEC_2025Oct4.csv"
)
DEFAULT_KEY = Path("/gpfs/data/huo-lab/Image/ChiMEC/study-16352a.csv")
DEFAULT_LAST_CONTACT = Path(
    "/gpfs/data/phs/groups/Projects/Huo_projects/SPORE/CRDW_data/2025Oct23/dr_7934_pats.txt"
)


def _split_patient_exam_id(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """split patient_exam_id formatted as '<patient_id>\\t<exam_id>'."""
    parts = series.astype(str).str.split("\t", n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError("patient_exam_id not formatted as 'patient\\texam'")
    return parts[0].str.strip(), parts[1].str.strip()


def _norm_mrn(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.zfill(8)


def _parse_study_date(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip()
    ymd = pd.to_datetime(x, format="%Y%m%d", errors="coerce")
    rest = pd.to_datetime(x[ymd.isna()], errors="coerce")
    ymd = ymd.copy()
    ymd.loc[ymd.isna()] = rest
    return ymd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pred",
        type=Path,
        required=True,
        help="Mirai validation_output.csv",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="mirai_manifest.csv (default: <pred.parent>/mirai_manifest.csv)",
    )
    p.add_argument(
        "--exams",
        type=Path,
        default=None,
        help="sot/exams.parquet (default: <pred.parent.parent>/sot/exams.parquet)",
    )
    p.add_argument("--phenotype", type=Path, default=DEFAULT_PHENOTYPE)
    p.add_argument("--patient-key", type=Path, default=DEFAULT_KEY)
    p.add_argument(
        "--last-contact",
        type=Path,
        default=DEFAULT_LAST_CONTACT,
        help="pipe-separated file with MRN and DATE_OF_LAST_CONTACT; omit to skip",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    pred_path = args.pred.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else pred_path.parent / "mirai_manifest.csv"
    )
    exams_path = (
        args.exams.expanduser().resolve()
        if args.exams
        else pred_path.parent.parent / "sot" / "exams.parquet"
    )

    pred = pd.read_csv(pred_path)
    if "patient_exam_id" not in pred.columns:
        raise SystemExit(f"{pred_path} missing patient_exam_id")
    pid, eid = _split_patient_exam_id(pred["patient_exam_id"])
    pred = pred.assign(patient_id=pid, exam_id=eid)

    risk_cols = [c for c in pred.columns if c.endswith("_year_risk")]
    pred_exam = pred.groupby(["patient_id", "exam_id"], as_index=False).agg(
        {c: "mean" for c in risk_cols}
    )

    man = pd.read_csv(manifest_path)
    man = man.assign(
        patient_id=man["patient_id"].astype(str).str.strip(),
        exam_id=man["exam_id"].astype(str).str.strip(),
    )
    label_cols = [
        c
        for c in (
            "years_to_cancer",
            "years_to_last_followup",
            "split_group",
        )
        if c in man.columns
    ]
    if not label_cols:
        raise SystemExit(f"{manifest_path} missing label columns")
    labels = man[["patient_id", "exam_id", *label_cols]].drop_duplicates(
        subset=["patient_id", "exam_id"]
    )

    if not exams_path.exists():
        raise SystemExit(f"exams parquet not found: {exams_path}")
    exams = pd.read_parquet(exams_path)
    exams = exams.assign(
        patient_id=exams["patient_id"].astype(str).str.strip(),
        exam_id=exams["exam_id"].astype(str).str.strip(),
    )
    exam_cols = [
        c
        for c in (
            "study_date",
            "accession_number",
            "device_manufacturer",
            "device_model",
            "site",
            "n_views_present",
        )
        if c in exams.columns
    ]
    exams_sub = exams[["patient_id", "exam_id", *exam_cols]].drop_duplicates(
        subset=["patient_id", "exam_id"]
    )

    key = pd.read_csv(args.patient_key)
    if not {"MRN", "AnonymousID"}.issubset(key.columns):
        raise SystemExit(f"{args.patient_key} needs MRN and AnonymousID")
    key = key.assign(
        MRN=_norm_mrn(key["MRN"]),
        patient_id=key["AnonymousID"].astype(str).str.strip(),
    )[["patient_id", "MRN"]].drop_duplicates(subset=["patient_id"])

    pheno = pd.read_csv(args.phenotype, low_memory=False)
    if "MRN" not in pheno.columns:
        raise SystemExit(f"{args.phenotype} missing MRN")
    pheno = pheno.copy()
    pheno["MRN"] = _norm_mrn(pheno["MRN"])
    # one row per patient for stable joins
    pheno = pheno.drop_duplicates(subset=["MRN"], keep="first")

    out = pred_exam.merge(labels, on=["patient_id", "exam_id"], how="left")
    out = out.merge(exams_sub, on=["patient_id", "exam_id"], how="left")
    out = out.merge(key, on="patient_id", how="left")

    pheno_cols = [c for c in pheno.columns if c != "MRN"]
    out = out.merge(
        pheno[pheno_cols + ["MRN"]], on="MRN", how="left", suffixes=("", "_pheno")
    )

    study_ts = (
        _parse_study_date(out["study_date"]) if "study_date" in out.columns else None
    )
    if study_ts is not None and "dob" in out.columns:
        dob = pd.to_datetime(out["dob"], format="%d%b%Y", errors="coerce")
        out["age_at_exam_years"] = (study_ts - dob).dt.days / 365.25

    if args.last_contact and Path(args.last_contact).exists():
        lc = pd.read_csv(args.last_contact, sep="|", encoding="latin-1")
        if {"MRN", "DATE_OF_LAST_CONTACT"}.issubset(lc.columns):
            lc = lc.assign(MRN=lc["MRN"].astype(str).str.strip().str.zfill(8))
            lc["last_contact_date"] = pd.to_datetime(
                lc["DATE_OF_LAST_CONTACT"], errors="coerce"
            )
            out = out.merge(
                lc[["MRN", "last_contact_date"]].drop_duplicates(subset=["MRN"]),
                on="MRN",
                how="left",
            )

    if study_ts is not None and "years_to_last_followup" in out.columns:
        ylf = pd.to_numeric(out["years_to_last_followup"], errors="coerce")
        out["approx_followup_end_date"] = study_ts + pd.to_timedelta(
            np.floor(ylf * 365.25), unit="D"
        )

    # canonical cancer / control dates from phenotype when present
    for col in ("datedx", "datedx_new", "date_diagnosis"):
        if col in out.columns:
            out[f"{col}_parsed"] = pd.to_datetime(out[col], errors="coerce")

    front = ["patient_id", "exam_id", "MRN", *risk_cols]
    rest = [c for c in out.columns if c not in front]
    out = out[front + sorted(rest)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} exam rows to {args.out}")


if __name__ == "__main__":
    main()
