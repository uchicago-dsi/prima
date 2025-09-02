#!/usr/bin/env python3
"""
Script extracted from dev.ipynb
Performs ChiMEC patient data analysis and generates plots
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# argument parser
parser = argparse.ArgumentParser(
    description="Analyze imaging metadata for specified modality"
)
parser.add_argument(
    "--modality",
    type=str,
    default="MG",
    choices=[
        "CR",
        "DX",
        "MG",
        "US",
        "CT",
        "MR",
        "NM",
        "PT",
        "XA",
        "RF",
        "ES",
        "XC",
        "PX",
        "RG",
    ],
    help="Base modality to analyze (default: MG)",
)
parser.add_argument(
    "--dump-screening-patients",
    action="store_true",
    help="Dump CSV of patients with screening scans at least 3 months before diagnosis",
)
args = parser.parse_args()

SELECTED_MODALITY = args.modality

# create plots directory if it doesn't exist
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

print(f"=== LOADING DATA FOR MODALITY: {SELECTED_MODALITY} ===")
# load data
chimec_patients = pd.read_csv(
    "/gpfs/data/phs/groups/Projects/Huo_projects/SPORE/annawoodard/List_ChiMEC_priority_2025July30.csv"
)
key = pd.read_csv("/gpfs/data/huo-lab/Image/ChiMEC/study-16352a.csv")
metadata = pd.read_csv(
    "/gpfs/data/huo-lab/Image/annawoodard/prima/data/imaging_metadata.csv"
)
print(f"ChiMEC patients: {len(chimec_patients):,}")
print(f"Key table records: {len(key):,}")
print(f"Metadata records: {len(metadata):,}")


# helper function to extract base modality
def get_base_modality(modality):
    """
    extract base modality from full modality string
    examples: 'MG' -> 'MG', 'MR' -> 'MR', 'CT' -> 'CT', etc.
    """
    if pd.isna(modality):
        return None

    modality_str = str(modality).upper()

    # common base modalities
    if "MG" in modality_str:
        return "MG"
    elif "MR" in modality_str:
        return "MR"
    elif "CT" in modality_str:
        return "CT"
    elif "US" in modality_str:
        return "US"
    elif "CR" in modality_str:
        return "CR"
    elif "DX" in modality_str:
        return "DX"
    elif "NM" in modality_str:
        return "NM"
    elif "PT" in modality_str:
        return "PT"
    else:
        return modality_str  # return as is if no match found


# add base modality column to metadata
metadata["base_modality"] = metadata["Modality"].apply(get_base_modality)

# modality labels for plotting
BASE_MODALITY_LABELS = {
    "CR": "computed radiography",
    "DX": "digital radiography",
    "MG": "mammography",
    "US": "ultrasound",
    "CT": "ct",
    "MR": "mri",
    "NM": "nuclear medicine",
    "PT": "pet",
    "XA": "x-ray angiography",
    "RF": "fluoroscopy",
    "ES": "endoscopy",
    "XC": "external camera",
    "PX": "panoramic x-ray",
    "RG": "radiographic imaging",
}


# helper functions for analysis
def filter_patients_by_chip_status(data, chip_status="all"):
    """
    filter patients by chip status

    Parameters
    ----------
    data : pd.DataFrame
        dataframe with chip column
    chip_status : str
        "all" for all patients, "genotyped" for only patients with non-null chip status

    Returns
    -------
    pd.DataFrame
        filtered dataframe
    """
    if chip_status == "all":
        return data
    elif chip_status == "genotyped":
        return data[data["chip"].notna()]
    else:
        raise ValueError("chip_status must be 'all' or 'genotyped'")


def create_screening_mammograms_plot(
    case_screening_per_patient,
    case_before_dx_month_per_patient,
    control_screening_per_patient,
    case_screening_scans,
    case_before_dx_month_scans,
    control_screening_scans,
    chip_status,
    plots_dir,
    modality="MG",
):
    """create screening scans plot"""
    modality_label = BASE_MODALITY_LABELS.get(modality, modality.lower())
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(
        3, 3, figsize=(24, 18)
    )

    # cases - mammograms >3 months before diagnosis
    if len(case_screening_per_patient) > 0:
        ax1.hist(
            case_screening_per_patient.values,
            bins=range(1, case_screening_per_patient.max() + 2),
            alpha=0.7,
            edgecolor="black",
        )
        ax1.set_xticks(range(1, min(21, case_screening_per_patient.max() + 1)))
    else:
        ax1.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax1.transAxes)

    ax1.set_title(f"cases\n({modality_label} >3 months before diagnosis)")
    ax1.set_xlabel(f"number of screening {modality_label} scans per patient")
    ax1.set_ylabel("number of patients")
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # cases - mammograms before/including diagnosis month
    if len(case_before_dx_month_per_patient) > 0:
        ax2.hist(
            case_before_dx_month_per_patient.values,
            bins=range(1, case_before_dx_month_per_patient.max() + 2),
            alpha=0.7,
            edgecolor="black",
        )
        ax2.set_xticks(range(1, min(21, case_before_dx_month_per_patient.max() + 1)))
    else:
        ax2.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax2.transAxes)

    ax2.set_title(f"cases\n({modality_label} before/including dx month)")
    ax2.set_xlabel(f"number of {modality_label} scans per patient")
    ax2.set_ylabel("number of patients")
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # controls
    if len(control_screening_per_patient) > 0:
        ax3.hist(
            control_screening_per_patient.values,
            bins=range(1, control_screening_per_patient.max() + 2),
            alpha=0.7,
            edgecolor="black",
        )
        ax3.set_xticks(range(1, min(21, control_screening_per_patient.max() + 1)))
    else:
        ax3.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax3.transAxes)

    ax3.set_title(f"controls\n(all {modality_label})")
    ax3.set_xlabel(f"number of {modality_label} scans per patient")
    ax3.set_ylabel("number of patients")
    ax3.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # fourth panel - collapsed summary
    categories = [
        "cases\n(>3 months before dx)",
        "cases\n(before/including dx month)",
        f"controls\n(all {modality_label})",
    ]
    counts = [
        len(case_screening_per_patient),
        len(case_before_dx_month_per_patient),
        len(control_screening_per_patient),
    ]

    bars4 = ax4.bar(categories, counts, alpha=0.7, edgecolor="black")
    ax4.set_title(f"total patients with {modality_label}")
    ax4.set_ylabel("number of patients")
    ax4.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # add value labels on bars
    for bar, count in zip(bars4, counts):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01 * max(counts),
            f"{count}",
            ha="center",
            va="bottom",
        )

    # fifth panel - total scans by category
    categories_scans = [
        "cases\n(>3 months before dx)",
        "cases\n(before/including dx month)",
        f"controls\n(all {modality_label})",
    ]
    total_scans = [
        len(case_screening_scans),
        len(case_before_dx_month_scans),
        len(control_screening_scans),
    ]

    bars5 = ax5.bar(categories_scans, total_scans, alpha=0.7, edgecolor="black")
    ax5.set_title(f"total {modality_label} scans by category")
    ax5.set_ylabel(f"number of {modality_label} scans")
    ax5.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # add value labels on bars
    for bar, count in zip(bars5, total_scans):
        height = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01 * max(total_scans),
            f"{count}",
            ha="center",
            va="bottom",
        )

    # sixth panel - scan year distribution comparison
    if len(case_screening_scans) > 0 and len(control_screening_scans) > 0:
        # extract years from scan dates
        case_years = case_screening_scans["Study DateTime"].dt.year
        control_years = control_screening_scans["Study DateTime"].dt.year

        # create year bins
        min_year = min(case_years.min(), control_years.min())
        max_year = max(case_years.max(), control_years.max())
        year_bins = range(min_year, max_year + 2)

        # create overlapping histograms with outline style
        ax6.hist(
            case_years,
            bins=year_bins,
            histtype="step",
            label=f"cases (n={len(case_screening_scans)})",
            color="red",
            linewidth=2,
        )
        ax6.hist(
            control_years,
            bins=year_bins,
            histtype="step",
            label=f"controls (n={len(control_screening_scans)})",
            color="blue",
            linewidth=2,
        )

        ax6.set_title(f"screening {modality_label} years\n(cases vs controls)")
        ax6.set_xlabel(f"year of {modality_label}")
        ax6.set_ylabel(f"number of {modality_label} scans")
        ax6.legend()
        ax6.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax6.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # rotate x-axis labels if needed
        if max_year - min_year > 10:
            ax6.tick_params(axis="x", rotation=45)
    else:
        ax6.text(
            0.5,
            0.5,
            "insufficient data",
            ha="center",
            va="center",
            transform=ax6.transAxes,
        )
        ax6.set_title(f"screening {modality_label} years\n(cases vs controls)")
        ax6.set_xticks([])
        ax6.set_yticks([])

    # seventh panel - months between screening exam and diagnosis for cases
    if len(case_screening_scans) > 0:
        # calculate months between screening exam and diagnosis
        months_to_dx = (
            case_screening_scans["days_to_dx"] / 30.44
        )  # average days per month

        ax7.hist(months_to_dx, bins=30, alpha=0.7, edgecolor="black")
        ax7.set_title(f"cases\n(months from screening {modality_label} to diagnosis)")
        ax7.set_xlabel("months from screening to diagnosis")
        ax7.set_ylabel(f"number of {modality_label} scans")
        ax7.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    else:
        ax7.text(
            0.5,
            0.5,
            "no data",
            ha="center",
            va="center",
            transform=ax7.transAxes,
        )
        ax7.set_title(f"cases\n(months from screening {modality_label} to diagnosis)")
        ax7.set_xticks([])
        ax7.set_yticks([])

    # hide unused panels
    ax8.axis("off")
    ax9.axis("off")

    plt.tight_layout()
    plt.savefig(
        plots_dir / f"screening_{modality.lower()}_scans_per_patient_{chip_status}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_mg_mammograms_per_patient_plot(
    selected_modality_per_patient,
    chimec_with_study_id,
    selected_modality_scans,
    chip_status,
    plots_dir,
    modality="MG",
):
    """create selected modality scans per patient plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # get case and control patients
    selected_modality_with_case_control = selected_modality_scans.merge(
        chimec_with_study_id[["AnonymousID", "case_or_control"]],
        left_on="study_id",
        right_on="AnonymousID",
        how="inner",
    )

    # cases
    case_selected_modality_scans = selected_modality_with_case_control[
        selected_modality_with_case_control["case_or_control"] == "Case"
    ]
    case_selected_modality_per_patient = case_selected_modality_scans[
        "study_id"
    ].value_counts()

    modality_label = BASE_MODALITY_LABELS.get(modality, modality.lower())
    ax1.hist(
        case_selected_modality_per_patient.values,
        bins=range(1, max(case_selected_modality_per_patient.max(), 2) + 2),
        alpha=0.7,
        edgecolor="black",
    )
    ax1.set_title("cases")
    ax1.set_xlabel(f"number of {modality_label} scans per patient")
    ax1.set_ylabel("number of patients")
    ax1.set_xticks(range(1, min(21, case_selected_modality_per_patient.max() + 1)))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # controls
    control_selected_modality_scans = selected_modality_with_case_control[
        selected_modality_with_case_control["case_or_control"] == "Control"
    ]
    control_selected_modality_per_patient = control_selected_modality_scans[
        "study_id"
    ].value_counts()

    ax2.hist(
        control_selected_modality_per_patient.values,
        bins=range(1, max(control_selected_modality_per_patient.max(), 2) + 2),
        alpha=0.7,
        edgecolor="black",
    )
    ax2.set_title("controls")
    ax2.set_xlabel(f"number of {modality_label} scans per patient")
    ax2.set_ylabel("number of patients")
    ax2.set_xticks(range(1, min(21, control_selected_modality_per_patient.max() + 1)))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(
        plots_dir / f"{modality.lower()}_scans_per_patient_{chip_status}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_modality_distribution_plot(metadata, chip_status, plots_dir):
    """create modality distribution plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # number of exams by modality
    exam_counts = metadata["base_modality"].value_counts().sort_values(ascending=False)
    # map modality codes to readable labels
    exam_labels = [BASE_MODALITY_LABELS.get(mod, mod) for mod in exam_counts.index]
    ax1.bar(exam_labels, exam_counts.values)
    ax1.set_title("number of exams by modality")
    ax1.set_xlabel("modality")
    ax1.set_ylabel("number of exams")
    ax1.set_xticks(range(len(exam_labels)))
    ax1.set_xticklabels(exam_labels, rotation=45, ha="right")

    # patients with at least one exam in each modality
    patient_counts = (
        metadata.groupby("base_modality")["study_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    total_patients = metadata["study_id"].nunique()
    # map modality codes to readable labels
    patient_labels = [
        BASE_MODALITY_LABELS.get(mod, mod) for mod in patient_counts.index
    ]
    ax2.bar(patient_labels, patient_counts.values)
    ax2.set_title(
        "patients with ≥1 exam in each modality\n(sum > total patients due to multi-modality)"
    )
    ax2.set_xlabel("modality")
    ax2.set_ylabel("number of patients")
    ax2.set_xticks(range(len(patient_labels)))
    ax2.set_xticklabels(patient_labels, rotation=45, ha="right")
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(
        plots_dir / f"modality_distribution_{chip_status}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_time_analysis_plot(
    cases_selected_modality,
    controls_selected_modality,
    chip_status,
    plots_dir,
    modality="MG",
):
    """create time analysis plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    modality_label = BASE_MODALITY_LABELS.get(modality, modality.lower())
    # cases
    if len(cases_selected_modality) > 0:
        case_time_to_dx = cases_selected_modality["days_to_dx"].dropna()
        # convert days to years
        case_time_to_dx_years = case_time_to_dx / 365.25
        ax1.hist(case_time_to_dx_years, bins=50, alpha=0.7, edgecolor="black")
        ax1.set_title(f"cases\n(positive = {modality_label} before diagnosis)")
        ax1.set_xlabel(f"years from {modality_label} to diagnosis")
        ax1.set_ylabel(f"number of {modality_label} scans")
        ax1.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="diagnosis date")
        ax1.axvline(
            x=90 / 365.25,
            color="orange",
            linestyle="--",
            alpha=0.7,
            label="3 months before dx (screening threshold)",
        )
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title(f"cases\n(positive = {modality_label} before diagnosis)")
        ax1.set_xlabel(f"years from {modality_label} to diagnosis")
        ax1.set_ylabel(f"number of {modality_label} scans")

    # controls
    if len(controls_selected_modality) > 0:
        control_time_from_first = controls_selected_modality[
            "days_from_first_scan"
        ].dropna()
        # convert days to years
        control_time_from_first_years = control_time_from_first / 365.25
        ax2.hist(control_time_from_first_years, bins=50, alpha=0.7, edgecolor="black")
        ax2.axvline(
            x=0, color="red", linestyle="--", alpha=0.7, label=f"first {modality_label}"
        )
    else:
        ax2.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax2.transAxes)

    ax2.set_title(f"controls\n(time from first {modality_label})")
    ax2.set_xlabel(f"years from first {modality_label}")
    ax2.set_ylabel(f"number of {modality_label} scans")
    if len(controls_selected_modality) > 0:
        ax2.legend()

    plt.tight_layout()
    plt.savefig(
        plots_dir
        / f"time_analysis_{modality.lower()}_cases_vs_controls_{chip_status}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# find and save missing MRNs
missing_mrns = chimec_patients[~chimec_patients["MRN"].isin(key["MRN"])]["MRN"].tolist()
with open("missing_mrns.txt", "w") as f:
    for mrn in missing_mrns:
        f.write(f"{mrn}\n")
print(f"missing MRNs (not in key): {len(missing_mrns):,} (saved to missing_mrns.txt)")

# connect study_id in metadata to chip column in chimec_patients through key table
# step 1: merge chimec_patients with key table to get AnonymousID for each MRN
chimec_with_study_id = chimec_patients.merge(
    key[["MRN", "AnonymousID"]], on="MRN", how="left"
)

# rename status column to case_or_control
chimec_with_study_id = chimec_with_study_id.rename(
    columns={"status": "case_or_control"}
)

print("\n=== DATA LINKING SUMMARY ===")
print(f"ChiMEC patients after key merge: {chimec_with_study_id.shape[0]:,}")
print(f"patients with chip data: {chimec_with_study_id['chip'].notna().sum():,}")
print(
    f"patients with chip + study_id: {(chimec_with_study_id['chip'].notna() & chimec_with_study_id['AnonymousID'].notna()).sum():,}"
)

# identify chip patients with selected modality records
chip_patients_study_ids = chimec_with_study_id[chimec_with_study_id["chip"].notna()][
    "AnonymousID"
].dropna()
selected_modality_records = metadata[metadata["base_modality"] == SELECTED_MODALITY]
chip_patients_with_selected_modality = chip_patients_study_ids[
    chip_patients_study_ids.isin(selected_modality_records["study_id"])
]

print(
    f"chip patients with {SELECTED_MODALITY} records: {len(chip_patients_with_selected_modality):,} / {len(chip_patients_study_ids):,} ({len(chip_patients_with_selected_modality) / len(chip_patients_study_ids) * 100:.1f}%)"
)

# total cases vs controls breakdown (all ChiMEC patients)
total_cases_count = chimec_with_study_id[
    chimec_with_study_id["case_or_control"] == "Case"
].shape[0]
total_controls_count = chimec_with_study_id[
    chimec_with_study_id["case_or_control"] == "Control"
].shape[0]

# genotyped cases vs controls breakdown
chip_patients_with_status = chimec_with_study_id[chimec_with_study_id["chip"].notna()]
genotyped_cases_count = chip_patients_with_status[
    chip_patients_with_status["case_or_control"] == "Case"
].shape[0]
genotyped_controls_count = chip_patients_with_status[
    chip_patients_with_status["case_or_control"] == "Control"
].shape[0]

print("\n=== PATIENT SUMMARY ===")
print(
    f"total ChiMEC patients: {len(chimec_patients):,} (cases: {total_cases_count:,}, controls: {total_controls_count:,})"
)
print(
    f"genotyped patients: {len(chimec_with_study_id[chimec_with_study_id['chip'].notna()]):,} (cases: {genotyped_cases_count:,}, controls: {genotyped_controls_count:,})"
)
print(f"patients with imaging: {metadata['study_id'].nunique():,}")
print(f"genotyped with imaging: {len(chip_patients_study_ids):,}")
print(
    f"genotyped with {SELECTED_MODALITY} imaging: {len(chip_patients_with_selected_modality):,}"
)

# time-to-diagnosis analysis for cases
print("\n=== TIME-TO-DIAGNOSIS ANALYSIS ===")
cases_with_study_id = chimec_with_study_id[
    chimec_with_study_id["case_or_control"] == "Case"
].copy()

# convert DatedxIndex to datetime
cases_with_study_id["DatedxIndex"] = pd.to_datetime(
    cases_with_study_id["DatedxIndex"], format="%d/%m/%Y"
)

# merge with metadata to get imaging dates
cases_imaging = cases_with_study_id.merge(
    metadata, left_on="AnonymousID", right_on="study_id", how="inner"
)
cases_imaging = cases_imaging[cases_imaging["base_modality"] == SELECTED_MODALITY]

# ensure Study DateTime is datetime type
cases_imaging["Study DateTime"] = pd.to_datetime(cases_imaging["Study DateTime"])

# calculate time-to-diagnosis (days from imaging to diagnosis)
cases_imaging["time_to_diagnosis"] = (
    cases_imaging["DatedxIndex"] - cases_imaging["Study DateTime"]
).dt.days

# convert days to years (approximate, 365.25 days per year)
cases_imaging["time_to_diagnosis_years"] = cases_imaging["time_to_diagnosis"] / 365.25

print(f"total case imaging records: {len(cases_imaging):,}")
print(
    f"records with valid time-to-diagnosis: {cases_imaging['time_to_diagnosis'].notna().sum():,}"
)
print(
    f"range of time-to-diagnosis: {cases_imaging['time_to_diagnosis'].min():,} to {cases_imaging['time_to_diagnosis'].max():,} days"
)

# create histogram including positive times (mammos after dx), x axis in years
plt.figure(figsize=(6, 3))
time_to_dx_years = cases_imaging["time_to_diagnosis_years"].dropna()
plt.hist(time_to_dx_years, bins=50, alpha=0.7, edgecolor="black")
plt.title(
    "cases\n(positive = imaging before diagnosis, negative = imaging after diagnosis)"
)
plt.xlabel("time to diagnosis (years)")
plt.ylabel("number of exams")
plt.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="diagnosis date")
plt.legend()
plt.savefig(plots_dir / "time_to_diagnosis_years.png", dpi=300, bbox_inches="tight")
plt.close()

# summary stats (in years)
print("\ntime-to-diagnosis statistics (years):")
print(f"mean: {time_to_dx_years.mean():.2f} years")
print(f"median: {time_to_dx_years.median():.2f} years")
print(f"exams after diagnosis (negative values): {(time_to_dx_years < 0).sum():,}")
print(f"exams before diagnosis (positive values): {(time_to_dx_years > 0).sum():,}")

# check for extreme time-to-diagnosis values
print("\n=== TIME-TO-DIAGNOSIS VALIDATION ===")
extreme_negative = cases_imaging[
    cases_imaging["time_to_diagnosis"] < -3650
]  # >10 years before
extreme_positive = cases_imaging[
    cases_imaging["time_to_diagnosis"] > 3650
]  # >10 years after
print(f"records >10 years after diagnosis: {len(extreme_negative):,}")
print(f"records >10 years before diagnosis: {len(extreme_positive):,}")
print(
    f"time range: {cases_imaging['time_to_diagnosis'].min():.0f} to {cases_imaging['time_to_diagnosis'].max():.0f} days"
)
print(
    f"mean: {cases_imaging['time_to_diagnosis'].mean():.1f} days, median: {cases_imaging['time_to_diagnosis'].median():.1f} days"
)

if len(extreme_negative) > 0:
    min_record = extreme_negative.loc[extreme_negative["time_to_diagnosis"].idxmin()]
    print(
        f"most extreme case: {min_record['time_to_diagnosis']:.0f} days ({min_record['time_to_diagnosis'] / 365.25:.1f} years) after diagnosis"
    )


def extract_base_modality(modality: str) -> str:
    """
    extract the base imaging modality from a modality string,
    ignoring derived/secondary DICOM object classes (PR, SR, KO, OT, SC, DOC, XC, CADSR, REG, RTSTRUCT, etc.)

    Parameters
    ----------
    modality : str
        modality string (e.g., "CT/PR/SR")

    Returns
    -------
    str
        base modality (e.g., "CT"), or "Other" if no base modality is found
    """
    # set of actual acquisition modalities
    base_modalities = {
        "CR",
        "DX",
        "MG",
        "US",
        "CT",
        "MR",
        "NM",
        "PT",
        "XA",
        "RF",
        "ES",
        "XC",
        "PX",
        "RG",
    }

    tokens = modality.split("/")
    for token in tokens:
        if token in base_modalities:
            return token
    return "Other"


# add base modality column to metadata
metadata["base_modality"] = metadata["Modality"].apply(extract_base_modality)

print("\n=== BASE MODALITY ANALYSIS ===")
print("unique base modalities:")
print(metadata["base_modality"].value_counts())
print("\nexamples of modality -> base_modality mapping:")
sample_mapping = metadata[["Modality", "base_modality"]].drop_duplicates().head(20)
print(sample_mapping)

BASE_MODALITY_LABELS = {
    "CR": "computed radiography",
    "DX": "digital radiography",
    "MG": "mammography",
    "US": "ultrasound",
    "CT": "ct",
    "MR": "mri",
    "NM": "nuclear medicine",
    "PT": "pet",
    "XA": "x-ray angiography",
    "RF": "fluoroscopy",
    "ES": "endoscopy",
    "XC": "external camera",
    "PX": "panoramic x-ray",
    "RG": "radiographic imaging",
}

print("\n=== MODALITY DISTRIBUTION ===")
total_patients = metadata["study_id"].nunique()
patient_counts = (
    metadata.groupby("base_modality")["study_id"].nunique().sort_values(ascending=False)
)
print(
    f"patients with multi-modality imaging: sum ({patient_counts.sum():,}) > total ({total_patients:,})"
)
create_modality_distribution_plot(metadata, "all", plots_dir)

# selected modality analysis
print(f"\n=== {SELECTED_MODALITY} ANALYSIS ===")
# filter for only selected modality scans
selected_modality_scans = metadata[
    metadata["base_modality"] == SELECTED_MODALITY
].copy()

# additional filtering for MR modality - only include scans with "BREAST" in description
if SELECTED_MODALITY == "MR":
    breast_filter = selected_modality_scans["StudyDescription"].str.contains(
        "BREAST", case=False, na=False
    )
    selected_modality_scans = selected_modality_scans[breast_filter].copy()
    print("filtered MR scans to only include those with 'BREAST' in description")

selected_modality_per_patient = selected_modality_scans["study_id"].value_counts()
print(
    f"total {SELECTED_MODALITY} scans: {len(selected_modality_scans):,} (patients: {selected_modality_scans['study_id'].nunique():,})"
)
print(
    f"scans per patient - range: {selected_modality_per_patient.min()}-{selected_modality_per_patient.max()}, mean: {selected_modality_per_patient.mean():.1f}, median: {selected_modality_per_patient.median():.1f}"
)

# create selected modality scans per patient plot for all patients
create_mg_mammograms_per_patient_plot(
    selected_modality_per_patient,
    chimec_with_study_id,
    selected_modality_scans,
    "all",
    plots_dir,
    SELECTED_MODALITY,
)

print(f"patients with most scans: {selected_modality_per_patient.head(3).to_dict()}")

# screening analysis
print(f"\n=== SCREENING {SELECTED_MODALITY} ANALYSIS ===")
# first merge selected modality scans with patient data to get DatedxIndex
selected_modality_with_patient_data = selected_modality_scans.merge(
    chimec_with_study_id[["AnonymousID", "DatedxIndex", "case_or_control", "chip"]],
    left_on="study_id",
    right_on="AnonymousID",
    how="inner",
)

print(
    f"{SELECTED_MODALITY} scans with patient data: {len(selected_modality_with_patient_data)}"
)

# genotyping coverage in imaging population
total_imaging_patients = selected_modality_with_patient_data["study_id"].nunique()
genotyped_imaging = selected_modality_with_patient_data[
    selected_modality_with_patient_data["chip"].notna()
]["study_id"].nunique()
print(
    f"ChiMEC patients with {SELECTED_MODALITY} imaging: {total_imaging_patients:,} (genotyped: {genotyped_imaging:,}, {genotyped_imaging / total_imaging_patients * 100:.1f}%)"
)
if genotyped_imaging == total_imaging_patients:
    print(
        "all ChiMEC patients with imaging are genotyped (explains identical 'all' vs 'genotyped' plots)"
    )

# convert dates to datetime
selected_modality_with_patient_data["Study DateTime"] = pd.to_datetime(
    selected_modality_with_patient_data["Study DateTime"]
)
selected_modality_with_patient_data["DatedxIndex"] = pd.to_datetime(
    selected_modality_with_patient_data["DatedxIndex"], format="%d/%m/%Y"
)

# calculate time difference in days (positive = scan before diagnosis)
selected_modality_with_patient_data["days_to_dx"] = (
    selected_modality_with_patient_data["DatedxIndex"]
    - selected_modality_with_patient_data["Study DateTime"]
).dt.days

# separate cases and controls for screening analysis
cases_selected_modality = selected_modality_with_patient_data[
    selected_modality_with_patient_data["case_or_control"] == "Case"
].copy()
controls_selected_modality = selected_modality_with_patient_data[
    selected_modality_with_patient_data["case_or_control"] == "Control"
].copy()

# for cases: filter for screening scans (> 3 months = 90 days before diagnosis)
case_screening_scans = cases_selected_modality[
    cases_selected_modality["days_to_dx"] > 90
].copy()

# for cases: filter for scans anytime before and including diagnosis month
# convert to year-month for comparison
cases_selected_modality["scan_ym"] = cases_selected_modality[
    "Study DateTime"
].dt.to_period("M")
cases_selected_modality["diagnosis_ym"] = cases_selected_modality[
    "DatedxIndex"
].dt.to_period("M")
case_before_dx_month_scans = cases_selected_modality[
    cases_selected_modality["scan_ym"] <= cases_selected_modality["diagnosis_ym"]
].copy()

# for controls: all scans are considered (no diagnosis date filter)
control_screening_scans = controls_selected_modality.copy()

print(
    f"case screening scans (>90 days before dx): {len(case_screening_scans):,} ({case_screening_scans['study_id'].nunique() if len(case_screening_scans) > 0 else 0:,} patients)"
)

# check genotyping coverage in screening cases
if len(case_screening_scans) > 0:
    screening_patients_genotyped = case_screening_scans[
        case_screening_scans["chip"].notna()
    ]["study_id"].nunique()
    total_screening_patients = case_screening_scans["study_id"].nunique()
    print(
        f"  - screening case patients genotyped: {screening_patients_genotyped:,} / {total_screening_patients:,} ({screening_patients_genotyped / total_screening_patients * 100:.1f}%)"
    )

    # identify any non-genotyped screening patients
    non_genotyped_screening = case_screening_scans[case_screening_scans["chip"].isna()][
        "study_id"
    ].unique()
    if len(non_genotyped_screening) > 0:
        print(
            f"  - non-genotyped screening case patients: {list(non_genotyped_screening)}"
        )
    else:
        print("  - all screening case patients are genotyped")
print(
    f"case scans (before/including dx month): {len(case_before_dx_month_scans):,} ({case_before_dx_month_scans['study_id'].nunique() if len(case_before_dx_month_scans) > 0 else 0:,} patients)"
)
print(
    f"control scans (all): {len(control_screening_scans):,} ({control_screening_scans['study_id'].nunique() if len(control_screening_scans) > 0 else 0:,} patients)"
)

# count screening scans per patient for cases
case_screening_per_patient = (
    case_screening_scans["study_id"].value_counts()
    if len(case_screening_scans) > 0
    else pd.Series(dtype=int)
)

# count scans before/including diagnosis month per patient for cases
case_before_dx_month_per_patient = (
    case_before_dx_month_scans["study_id"].value_counts()
    if len(case_before_dx_month_scans) > 0
    else pd.Series(dtype=int)
)

# count all scans per patient for controls
control_screening_per_patient = (
    control_screening_scans["study_id"].value_counts()
    if len(control_screening_scans) > 0
    else pd.Series(dtype=int)
)

# scans per patient summary
if len(case_screening_per_patient) > 0:
    print(
        f"case screening per patient - range: {case_screening_per_patient.min()}-{case_screening_per_patient.max()}, mean: {case_screening_per_patient.mean():.1f}"
    )
if len(case_before_dx_month_per_patient) > 0:
    print(
        f"case before/including dx month per patient - range: {case_before_dx_month_per_patient.min()}-{case_before_dx_month_per_patient.max()}, mean: {case_before_dx_month_per_patient.mean():.1f}"
    )
if len(control_screening_per_patient) > 0:
    print(
        f"control per patient - range: {control_screening_per_patient.min()}-{control_screening_per_patient.max()}, mean: {control_screening_per_patient.mean():.1f}"
    )

# create screening scans plot for all patients
create_screening_mammograms_plot(
    case_screening_per_patient,
    case_before_dx_month_per_patient,
    control_screening_per_patient,
    case_screening_scans,
    case_before_dx_month_scans,
    control_screening_scans,
    "all",
    plots_dir,
    SELECTED_MODALITY,
)


# histogram of time to DatedxIndex for cases and time from first scan for controls
print("\n=== TIME ANALYSIS ===")

# separate cases and controls
cases_selected_modality_time = selected_modality_with_patient_data[
    selected_modality_with_patient_data["case_or_control"] == "Case"
].copy()
controls_selected_modality_time = selected_modality_with_patient_data[
    selected_modality_with_patient_data["case_or_control"] == "Control"
].copy()

# for controls, calculate time from first scan
if len(controls_selected_modality_time) > 0:
    controls_selected_modality_time = controls_selected_modality_time.sort_values(
        "Study DateTime"
    )
    first_scan_dates = controls_selected_modality_time.groupby("study_id")[
        "Study DateTime"
    ].first()
    controls_selected_modality_time = controls_selected_modality_time.merge(
        first_scan_dates.rename("first_scan_date"),
        left_on="study_id",
        right_index=True,
    )
    controls_selected_modality_time["days_from_first_scan"] = (
        controls_selected_modality_time["Study DateTime"]
        - controls_selected_modality_time["first_scan_date"]
    ).dt.days

# time analysis summary
if len(cases_selected_modality_time) > 0:
    case_time_to_dx = cases_selected_modality_time["days_to_dx"].dropna()
    print(
        f"case {SELECTED_MODALITY} scans: {len(case_time_to_dx):,} (before dx: {(case_time_to_dx > 0).sum():,}, after: {(case_time_to_dx < 0).sum():,}, on dx day: {(case_time_to_dx == 0).sum():,})"
    )
    print(
        f"case time to dx - range: {case_time_to_dx.min():.0f} to {case_time_to_dx.max():.0f} days, mean: {case_time_to_dx.mean():.0f}, median: {case_time_to_dx.median():.0f}"
    )

if len(controls_selected_modality_time) > 0:
    control_time_from_first = controls_selected_modality_time[
        "days_from_first_scan"
    ].dropna()
    print(
        f"control {SELECTED_MODALITY} scans: {len(control_time_from_first):,} with time from first scan"
    )
    print(
        f"control time from first - range: {control_time_from_first.min():.0f} to {control_time_from_first.max():.0f} days, mean: {control_time_from_first.mean():.0f}, median: {control_time_from_first.median():.0f}"
    )

# create time analysis plot for all patients
create_time_analysis_plot(
    cases_selected_modality_time,
    controls_selected_modality_time,
    "all",
    plots_dir,
    SELECTED_MODALITY,
)

# time period breakdown
if len(cases_selected_modality_time) > 0:
    case_time_to_dx = cases_selected_modality_time["days_to_dx"].dropna()
    screening = (case_time_to_dx > 90).sum()
    diagnostic = ((case_time_to_dx >= 0) & (case_time_to_dx <= 90)).sum()
    after = (case_time_to_dx < 0).sum()
    print(
        f"case time periods - screening (>90d): {screening:,}, diagnostic (0-90d): {diagnostic:,}, after dx: {after:,}"
    )

if len(controls_selected_modality_time) > 0:
    control_time_from_first = controls_selected_modality_time[
        "days_from_first_scan"
    ].dropna()
    first = (control_time_from_first == 0).sum()
    followup = (control_time_from_first > 0).sum()
    within_year = (
        (control_time_from_first >= 0) & (control_time_from_first <= 365)
    ).sum()
    print(
        f"control time periods - first: {first:,}, follow-up: {followup:,}, within 1yr: {within_year:,}"
    )

# create genotyped-only versions of all plots
print("\n=== CREATING GENOTYPED-ONLY VERSIONS ===")

# filter for only genotyped patients
genotyped_patients = filter_patients_by_chip_status(chimec_with_study_id, "genotyped")
genotyped_metadata = metadata[
    metadata["study_id"].isin(genotyped_patients["AnonymousID"])
]

print(
    f"genotyped patients: {len(genotyped_patients):,} (metadata records: {len(genotyped_metadata):,})"
)

# create genotyped-only modality distribution plot
create_modality_distribution_plot(genotyped_metadata, "genotyped", plots_dir)

# create genotyped-only selected modality scans per patient plot
genotyped_selected_modality_scans = genotyped_metadata[
    genotyped_metadata["base_modality"] == SELECTED_MODALITY
].copy()

# additional filtering for MR modality - only include scans with "BREAST" in description
if SELECTED_MODALITY == "MR":
    breast_filter = genotyped_selected_modality_scans["StudyDescription"].str.contains(
        "BREAST", case=False, na=False
    )
    genotyped_selected_modality_scans = genotyped_selected_modality_scans[
        breast_filter
    ].copy()
genotyped_selected_modality_per_patient = genotyped_selected_modality_scans[
    "study_id"
].value_counts()
create_mg_mammograms_per_patient_plot(
    genotyped_selected_modality_per_patient,
    genotyped_patients,
    genotyped_selected_modality_scans,
    "genotyped",
    plots_dir,
    SELECTED_MODALITY,
)

# create genotyped-only screening analysis
genotyped_selected_modality_with_patient_data = genotyped_selected_modality_scans.merge(
    genotyped_patients[["AnonymousID", "DatedxIndex", "case_or_control"]],
    left_on="study_id",
    right_on="AnonymousID",
    how="inner",
)

print(
    f"genotyped {SELECTED_MODALITY} scans with patient data: {len(genotyped_selected_modality_with_patient_data):,}"
)

# convert dates to datetime
genotyped_selected_modality_with_patient_data["Study DateTime"] = pd.to_datetime(
    genotyped_selected_modality_with_patient_data["Study DateTime"]
)
genotyped_selected_modality_with_patient_data["DatedxIndex"] = pd.to_datetime(
    genotyped_selected_modality_with_patient_data["DatedxIndex"], format="%d/%m/%Y"
)

# calculate time difference in days (positive = scan before diagnosis)
genotyped_selected_modality_with_patient_data["days_to_dx"] = (
    genotyped_selected_modality_with_patient_data["DatedxIndex"]
    - genotyped_selected_modality_with_patient_data["Study DateTime"]
).dt.days

# separate cases and controls for screening analysis
genotyped_cases_selected_modality = genotyped_selected_modality_with_patient_data[
    genotyped_selected_modality_with_patient_data["case_or_control"] == "Case"
].copy()
genotyped_controls_selected_modality = genotyped_selected_modality_with_patient_data[
    genotyped_selected_modality_with_patient_data["case_or_control"] == "Control"
].copy()

# for cases: filter for screening scans (> 3 months = 90 days before diagnosis)
genotyped_case_screening_scans = genotyped_cases_selected_modality[
    genotyped_cases_selected_modality["days_to_dx"] > 90
].copy()

# for cases: filter for scans anytime before and including diagnosis month
genotyped_cases_selected_modality["scan_ym"] = genotyped_cases_selected_modality[
    "Study DateTime"
].dt.to_period("M")
genotyped_cases_selected_modality["diagnosis_ym"] = genotyped_cases_selected_modality[
    "DatedxIndex"
].dt.to_period("M")
genotyped_case_before_dx_month_scans = genotyped_cases_selected_modality[
    genotyped_cases_selected_modality["scan_ym"]
    <= genotyped_cases_selected_modality["diagnosis_ym"]
].copy()

# for controls: all scans are considered (no diagnosis date filter)
genotyped_control_screening_scans = genotyped_controls_selected_modality.copy()

# count screening scans per patient for cases
genotyped_case_screening_per_patient = (
    genotyped_case_screening_scans["study_id"].value_counts()
    if len(genotyped_case_screening_scans) > 0
    else pd.Series(dtype=int)
)

# count scans before/including diagnosis month per patient for cases
genotyped_case_before_dx_month_per_patient = (
    genotyped_case_before_dx_month_scans["study_id"].value_counts()
    if len(genotyped_case_before_dx_month_scans) > 0
    else pd.Series(dtype=int)
)

# count all scans per patient for controls
genotyped_control_screening_per_patient = (
    genotyped_control_screening_scans["study_id"].value_counts()
    if len(genotyped_control_screening_scans) > 0
    else pd.Series(dtype=int)
)

# create genotyped-only screening scans plot
create_screening_mammograms_plot(
    genotyped_case_screening_per_patient,
    genotyped_case_before_dx_month_per_patient,
    genotyped_control_screening_per_patient,
    genotyped_case_screening_scans,
    genotyped_case_before_dx_month_scans,
    genotyped_control_screening_scans,
    "genotyped",
    plots_dir,
    SELECTED_MODALITY,
)

# create genotyped-only time analysis
genotyped_cases_selected_modality_for_time = (
    genotyped_selected_modality_with_patient_data[
        genotyped_selected_modality_with_patient_data["case_or_control"] == "Case"
    ].copy()
)
genotyped_controls_selected_modality_for_time = (
    genotyped_selected_modality_with_patient_data[
        genotyped_selected_modality_with_patient_data["case_or_control"] == "Control"
    ].copy()
)

# for controls, calculate time from first scan
if len(genotyped_controls_selected_modality_for_time) > 0:
    genotyped_controls_selected_modality_for_time = (
        genotyped_controls_selected_modality_for_time.sort_values("Study DateTime")
    )
    genotyped_first_scan_dates = genotyped_controls_selected_modality_for_time.groupby(
        "study_id"
    )["Study DateTime"].first()
    genotyped_controls_selected_modality_for_time = (
        genotyped_controls_selected_modality_for_time.merge(
            genotyped_first_scan_dates.rename("first_scan_date"),
            left_on="study_id",
            right_index=True,
        )
    )
    genotyped_controls_selected_modality_for_time["days_from_first_scan"] = (
        genotyped_controls_selected_modality_for_time["Study DateTime"]
        - genotyped_controls_selected_modality_for_time["first_scan_date"]
    ).dt.days

# create genotyped-only time analysis plot
create_time_analysis_plot(
    genotyped_cases_selected_modality_for_time,
    genotyped_controls_selected_modality_for_time,
    "genotyped",
    plots_dir,
    SELECTED_MODALITY,
)

print("\n=== ANALYSIS COMPLETE ===")
print(f"plots saved to: {plots_dir.absolute()}")
print("created 'all' and 'genotyped' versions of all plots")

# dump screening patients if requested
if args.dump_screening_patients:
    print(f"\n=== DUMPING SCREENING PATIENTS FOR {SELECTED_MODALITY} ===")

    # get case patients with screening scans (>90 days before diagnosis)
    screening_patients_data = case_screening_scans.copy()

    if len(screening_patients_data) > 0:
        # merge with chimec_patients to get MRN
        screening_patients_with_mrn = screening_patients_data.merge(
            key[["AnonymousID", "MRN"]],
            left_on="study_id",
            right_on="AnonymousID",
            how="left",
        )

        # create CSV with requested columns
        csv_data = screening_patients_with_mrn[
            ["MRN", "study_id", "DatedxIndex", "StudyDescription", "Study DateTime"]
        ].copy()

        # rename columns for clarity
        csv_data = csv_data.rename(
            columns={
                "study_id": "study_id",
                "DatedxIndex": "DX_date",
                "StudyDescription": "scan_description",
                "Study DateTime": "scan_date",
            }
        )

        # sort by patient MRN then scan date
        csv_data = csv_data.sort_values(["MRN", "scan_date"])

        # save CSV
        csv_filename = f"screening_patients_{SELECTED_MODALITY.lower()}.csv"
        csv_data.to_csv(csv_filename, index=False)

        # summary statistics
        unique_patients = csv_data["MRN"].nunique()
        total_scans = len(csv_data)

        print(
            f"patients with screening {SELECTED_MODALITY} scans (≥3 months before DX): {unique_patients:,}"
        )
        print(f"total screening {SELECTED_MODALITY} scans: {total_scans:,}")
        print(f"average scans per patient: {total_scans / unique_patients:.1f}")
        print(f"CSV saved to: {csv_filename}")

        # bin patients by scan date (every 2 years)
        print(f"\n--- {SELECTED_MODALITY} screening scans by time period ---")
        csv_data["scan_year"] = pd.to_datetime(csv_data["scan_date"]).dt.year
        min_year = csv_data["scan_year"].min()
        max_year = csv_data["scan_year"].max()

        # create 2-year bins
        bin_edges = list(
            range(min_year, max_year + 3, 2)
        )  # +3 to include the last year
        csv_data["year_bin"] = pd.cut(
            csv_data["scan_year"], bins=bin_edges, right=False, include_lowest=True
        )

        # count scans per bin
        scans_per_bin = csv_data["year_bin"].value_counts().sort_index()

        for bin_range, count in scans_per_bin.items():
            print(f"{bin_range}: {count:,} scans")

    else:
        print(
            f"no case patients found with screening {SELECTED_MODALITY} scans (≥3 months before diagnosis)"
        )
        print("CSV file not created")
