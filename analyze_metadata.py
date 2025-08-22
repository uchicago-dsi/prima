#!/usr/bin/env python3
"""
Script extracted from dev.ipynb
Performs ChiMEC patient data analysis and generates plots
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# create plots directory if it doesn't exist
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

print("=== LOADING DATA ===")
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
    case_screening_mammograms,
    case_before_dx_month_mammograms,
    control_screening_mammograms,
    chip_status,
    plots_dir,
):
    """create screening mammograms plot"""
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 12))

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

    ax1.set_title("cases\n(mammograms >3 months before diagnosis)")
    ax1.set_xlabel("number of screening mammograms per patient")
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

    ax2.set_title("cases\n(mammograms before/including dx month)")
    ax2.set_xlabel("number of mammograms per patient")
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

    ax3.set_title("controls\n(all mammograms)")
    ax3.set_xlabel("number of mammograms per patient")
    ax3.set_ylabel("number of patients")
    ax3.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # fourth panel - collapsed summary
    categories = [
        "cases\n(>3 months before dx)",
        "cases\n(before/including dx month)",
        "controls\n(all mammograms)",
    ]
    counts = [
        len(case_screening_per_patient),
        len(case_before_dx_month_per_patient),
        len(control_screening_per_patient),
    ]

    bars4 = ax4.bar(categories, counts, alpha=0.7, edgecolor="black")
    ax4.set_title("total patients with mammograms")
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

    # fifth panel - total mammograms by category
    categories_mammograms = [
        "cases\n(>3 months before dx)",
        "cases\n(before/including dx month)",
        "controls\n(all mammograms)",
    ]
    total_mammograms = [
        len(case_screening_mammograms),
        len(case_before_dx_month_mammograms),
        len(control_screening_mammograms),
    ]

    bars5 = ax5.bar(
        categories_mammograms, total_mammograms, alpha=0.7, edgecolor="black"
    )
    ax5.set_title("total mammograms by category")
    ax5.set_ylabel("number of mammograms")
    ax5.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # add value labels on bars
    for bar, count in zip(bars5, total_mammograms):
        height = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01 * max(total_mammograms),
            f"{count}",
            ha="center",
            va="bottom",
        )

    # sixth panel - mammogram year distribution comparison
    if len(case_screening_mammograms) > 0 and len(control_screening_mammograms) > 0:
        # extract years from mammogram dates
        case_years = case_screening_mammograms["Study DateTime"].dt.year
        control_years = control_screening_mammograms["Study DateTime"].dt.year

        # create year bins
        min_year = min(case_years.min(), control_years.min())
        max_year = max(case_years.max(), control_years.max())
        year_bins = range(min_year, max_year + 2)

        # create overlapping histograms with outline style
        ax6.hist(
            case_years,
            bins=year_bins,
            histtype="step",
            label=f"cases (n={len(case_screening_per_patient)})",
            color="red",
            linewidth=2,
        )
        ax6.hist(
            control_years,
            bins=year_bins,
            histtype="step",
            label=f"controls (n={len(control_screening_per_patient)})",
            color="blue",
            linewidth=2,
        )

        ax6.set_title("screening mammogram years\n(cases vs controls)")
        ax6.set_xlabel("year of mammogram")
        ax6.set_ylabel("number of mammograms")
        ax6.legend()
        ax6.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

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
        ax6.set_title("screening mammogram years\n(cases vs controls)")
        ax6.set_xticks([])
        ax6.set_yticks([])

    plt.tight_layout()
    plt.savefig(
        plots_dir / f"screening_mammograms_per_patient_{chip_status}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_mg_mammograms_per_patient_plot(
    mg_per_patient, chimec_with_study_id, mg_scans, chip_status, plots_dir
):
    """create MG mammograms per patient plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # get case and control patients
    mg_with_case_control = mg_scans.merge(
        chimec_with_study_id[["AnonymousID", "case_or_control"]],
        left_on="study_id",
        right_on="AnonymousID",
        how="inner",
    )

    # cases
    case_mg_scans = mg_with_case_control[
        mg_with_case_control["case_or_control"] == "Case"
    ]
    case_mg_per_patient = case_mg_scans["study_id"].value_counts()

    ax1.hist(
        case_mg_per_patient.values,
        bins=range(1, max(case_mg_per_patient.max(), 2) + 2),
        alpha=0.7,
        edgecolor="black",
    )
    ax1.set_title("cases")
    ax1.set_xlabel("number of mammograms per patient")
    ax1.set_ylabel("number of patients")
    ax1.set_xticks(range(1, min(21, case_mg_per_patient.max() + 1)))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # controls
    control_mg_scans = mg_with_case_control[
        mg_with_case_control["case_or_control"] == "Control"
    ]
    control_mg_per_patient = control_mg_scans["study_id"].value_counts()

    ax2.hist(
        control_mg_per_patient.values,
        bins=range(1, max(control_mg_per_patient.max(), 2) + 2),
        alpha=0.7,
        edgecolor="black",
    )
    ax2.set_title("controls")
    ax2.set_xlabel("number of mammograms per patient")
    ax2.set_ylabel("number of patients")
    ax2.set_xticks(range(1, min(21, control_mg_per_patient.max() + 1)))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(
        plots_dir / f"mg_mammograms_per_patient_{chip_status}.png",
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


def create_time_analysis_plot(cases_mg, controls_mg, chip_status, plots_dir):
    """create time analysis plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # cases
    if len(cases_mg) > 0:
        case_time_to_dx = cases_mg["days_to_dx"].dropna()
        # convert days to years
        case_time_to_dx_years = case_time_to_dx / 365.25
        ax1.hist(case_time_to_dx_years, bins=50, alpha=0.7, edgecolor="black")
        ax1.set_title("cases\n(positive = mammogram before diagnosis)")
        ax1.set_xlabel("years from mammogram to diagnosis")
        ax1.set_ylabel("number of mammograms")
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
        ax1.set_title("cases\n(positive = mammogram before diagnosis)")
        ax1.set_xlabel("years from mammogram to diagnosis")
        ax1.set_ylabel("number of mammograms")

    # controls
    if len(controls_mg) > 0:
        control_time_from_first = controls_mg["days_from_first_mammogram"].dropna()
        # convert days to years
        control_time_from_first_years = control_time_from_first / 365.25
        ax2.hist(control_time_from_first_years, bins=50, alpha=0.7, edgecolor="black")
        ax2.axvline(
            x=0, color="red", linestyle="--", alpha=0.7, label="first mammogram"
        )
    else:
        ax2.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax2.transAxes)

    ax2.set_title("controls\n(time from first mammogram)")
    ax2.set_xlabel("years from first mammogram")
    ax2.set_ylabel("number of mammograms")
    if len(controls_mg) > 0:
        ax2.legend()

    plt.tight_layout()
    plt.savefig(
        plots_dir / f"time_analysis_cases_vs_controls_{chip_status}.png",
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

# identify chip patients with MG records
chip_patients_study_ids = chimec_with_study_id[chimec_with_study_id["chip"].notna()][
    "AnonymousID"
].dropna()
mg_records = metadata[metadata["Modality"].str.contains("MG", na=False)]
chip_patients_with_mg = chip_patients_study_ids[
    chip_patients_study_ids.isin(mg_records["study_id"])
]

print(
    f"chip patients with MG records: {len(chip_patients_with_mg):,} / {len(chip_patients_study_ids):,} ({len(chip_patients_with_mg) / len(chip_patients_study_ids) * 100:.1f}%)"
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
print(f"genotyped with MG imaging: {len(chip_patients_with_mg):,}")

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
cases_imaging = cases_imaging[cases_imaging["base_modality"] == "MG"]

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

# MG mammogram analysis
print("\n=== MAMMOGRAM ANALYSIS ===")
# filter for only MG mammograms
mg_scans = metadata[metadata["base_modality"] == "MG"].copy()

mg_per_patient = mg_scans["study_id"].value_counts()
print(
    f"total MG scans: {len(mg_scans):,} (patients: {mg_scans['study_id'].nunique():,})"
)
print(
    f"mammograms per patient - range: {mg_per_patient.min()}-{mg_per_patient.max()}, mean: {mg_per_patient.mean():.1f}, median: {mg_per_patient.median():.1f}"
)

# create MG mammograms per patient plot for all patients
create_mg_mammograms_per_patient_plot(
    mg_per_patient, chimec_with_study_id, mg_scans, "all", plots_dir
)

print(f"patients with most mammograms: {mg_per_patient.head(3).to_dict()}")

# screening mammogram analysis
print("\n=== SCREENING MAMMOGRAM ANALYSIS ===")
# first merge MG scans with patient data to get DatedxIndex
mg_with_patient_data = mg_scans.merge(
    chimec_with_study_id[["AnonymousID", "DatedxIndex", "case_or_control", "chip"]],
    left_on="study_id",
    right_on="AnonymousID",
    how="inner",
)

print(f"MG scans with patient data: {len(mg_with_patient_data)}")

# genotyping coverage in imaging population
total_imaging_patients = mg_with_patient_data["study_id"].nunique()
genotyped_imaging = mg_with_patient_data[mg_with_patient_data["chip"].notna()][
    "study_id"
].nunique()
print(
    f"ChiMEC patients with MG imaging: {total_imaging_patients:,} (genotyped: {genotyped_imaging:,}, {genotyped_imaging / total_imaging_patients * 100:.1f}%)"
)
if genotyped_imaging == total_imaging_patients:
    print(
        "all ChiMEC patients with imaging are genotyped (explains identical 'all' vs 'genotyped' plots)"
    )

# convert dates to datetime
mg_with_patient_data["Study DateTime"] = pd.to_datetime(
    mg_with_patient_data["Study DateTime"]
)
mg_with_patient_data["DatedxIndex"] = pd.to_datetime(
    mg_with_patient_data["DatedxIndex"], format="%d/%m/%Y"
)

# calculate time difference in days (positive = mammogram before diagnosis)
mg_with_patient_data["days_to_dx"] = (
    mg_with_patient_data["DatedxIndex"] - mg_with_patient_data["Study DateTime"]
).dt.days

# separate cases and controls for screening analysis
cases_mg = mg_with_patient_data[
    mg_with_patient_data["case_or_control"] == "Case"
].copy()
controls_mg = mg_with_patient_data[
    mg_with_patient_data["case_or_control"] == "Control"
].copy()

# for cases: filter for screening mammograms (> 3 months = 90 days before diagnosis)
case_screening_mammograms = cases_mg[cases_mg["days_to_dx"] > 90].copy()

# for cases: filter for mammograms anytime before and including diagnosis month
# convert to year-month for comparison
cases_mg["mammogram_ym"] = cases_mg["Study DateTime"].dt.to_period("M")
cases_mg["diagnosis_ym"] = cases_mg["DatedxIndex"].dt.to_period("M")
case_before_dx_month_mammograms = cases_mg[
    cases_mg["mammogram_ym"] <= cases_mg["diagnosis_ym"]
].copy()

# for controls: all mammograms are considered (no diagnosis date filter)
control_screening_mammograms = controls_mg.copy()

print(
    f"case screening mammograms (>90 days before dx): {len(case_screening_mammograms):,} ({case_screening_mammograms['study_id'].nunique() if len(case_screening_mammograms) > 0 else 0:,} patients)"
)
print(
    f"case mammograms (before/including dx month): {len(case_before_dx_month_mammograms):,} ({case_before_dx_month_mammograms['study_id'].nunique() if len(case_before_dx_month_mammograms) > 0 else 0:,} patients)"
)
print(
    f"control mammograms (all): {len(control_screening_mammograms):,} ({control_screening_mammograms['study_id'].nunique() if len(control_screening_mammograms) > 0 else 0:,} patients)"
)

# count screening mammograms per patient for cases
case_screening_per_patient = (
    case_screening_mammograms["study_id"].value_counts()
    if len(case_screening_mammograms) > 0
    else pd.Series(dtype=int)
)

# count mammograms before/including diagnosis month per patient for cases
case_before_dx_month_per_patient = (
    case_before_dx_month_mammograms["study_id"].value_counts()
    if len(case_before_dx_month_mammograms) > 0
    else pd.Series(dtype=int)
)

# count all mammograms per patient for controls
control_screening_per_patient = (
    control_screening_mammograms["study_id"].value_counts()
    if len(control_screening_mammograms) > 0
    else pd.Series(dtype=int)
)

# mammograms per patient summary
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

# create screening mammograms plot for all patients
create_screening_mammograms_plot(
    case_screening_per_patient,
    case_before_dx_month_per_patient,
    control_screening_per_patient,
    case_screening_mammograms,
    case_before_dx_month_mammograms,
    control_screening_mammograms,
    "all",
    plots_dir,
)


# histogram of time to DatedxIndex for cases and time from first mammogram for controls
print("\n=== TIME ANALYSIS ===")

# separate cases and controls
cases_mg = mg_with_patient_data[
    mg_with_patient_data["case_or_control"] == "Case"
].copy()
controls_mg = mg_with_patient_data[
    mg_with_patient_data["case_or_control"] == "Control"
].copy()

# for controls, calculate time from first mammogram
if len(controls_mg) > 0:
    controls_mg = controls_mg.sort_values("Study DateTime")
    first_mammogram_dates = controls_mg.groupby("study_id")["Study DateTime"].first()
    controls_mg = controls_mg.merge(
        first_mammogram_dates.rename("first_mammogram_date"),
        left_on="study_id",
        right_index=True,
    )
    controls_mg["days_from_first_mammogram"] = (
        controls_mg["Study DateTime"] - controls_mg["first_mammogram_date"]
    ).dt.days

# time analysis summary
if len(cases_mg) > 0:
    case_time_to_dx = cases_mg["days_to_dx"].dropna()
    print(
        f"case MG scans: {len(case_time_to_dx):,} (before dx: {(case_time_to_dx > 0).sum():,}, after: {(case_time_to_dx < 0).sum():,}, on dx day: {(case_time_to_dx == 0).sum():,})"
    )
    print(
        f"case time to dx - range: {case_time_to_dx.min():.0f} to {case_time_to_dx.max():.0f} days, mean: {case_time_to_dx.mean():.0f}, median: {case_time_to_dx.median():.0f}"
    )

if len(controls_mg) > 0:
    control_time_from_first = controls_mg["days_from_first_mammogram"].dropna()
    print(
        f"control MG scans: {len(control_time_from_first):,} with time from first mammogram"
    )
    print(
        f"control time from first - range: {control_time_from_first.min():.0f} to {control_time_from_first.max():.0f} days, mean: {control_time_from_first.mean():.0f}, median: {control_time_from_first.median():.0f}"
    )

# create time analysis plot for all patients
create_time_analysis_plot(cases_mg, controls_mg, "all", plots_dir)

# time period breakdown
if len(cases_mg) > 0:
    case_time_to_dx = cases_mg["days_to_dx"].dropna()
    screening = (case_time_to_dx > 90).sum()
    diagnostic = ((case_time_to_dx >= 0) & (case_time_to_dx <= 90)).sum()
    after = (case_time_to_dx < 0).sum()
    print(
        f"case time periods - screening (>90d): {screening:,}, diagnostic (0-90d): {diagnostic:,}, after dx: {after:,}"
    )

if len(controls_mg) > 0:
    control_time_from_first = controls_mg["days_from_first_mammogram"].dropna()
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

# create genotyped-only MG mammograms per patient plot
genotyped_mg_scans = genotyped_metadata[
    genotyped_metadata["base_modality"] == "MG"
].copy()
genotyped_mg_per_patient = genotyped_mg_scans["study_id"].value_counts()
create_mg_mammograms_per_patient_plot(
    genotyped_mg_per_patient,
    genotyped_patients,
    genotyped_mg_scans,
    "genotyped",
    plots_dir,
)

# create genotyped-only screening mammograms analysis
genotyped_mg_with_patient_data = genotyped_mg_scans.merge(
    genotyped_patients[["AnonymousID", "DatedxIndex", "case_or_control"]],
    left_on="study_id",
    right_on="AnonymousID",
    how="inner",
)

print(f"genotyped MG scans with patient data: {len(genotyped_mg_with_patient_data):,}")

# convert dates to datetime
genotyped_mg_with_patient_data["Study DateTime"] = pd.to_datetime(
    genotyped_mg_with_patient_data["Study DateTime"]
)
genotyped_mg_with_patient_data["DatedxIndex"] = pd.to_datetime(
    genotyped_mg_with_patient_data["DatedxIndex"], format="%d/%m/%Y"
)

# calculate time difference in days (positive = mammogram before diagnosis)
genotyped_mg_with_patient_data["days_to_dx"] = (
    genotyped_mg_with_patient_data["DatedxIndex"]
    - genotyped_mg_with_patient_data["Study DateTime"]
).dt.days

# separate cases and controls for screening analysis
genotyped_cases_mg = genotyped_mg_with_patient_data[
    genotyped_mg_with_patient_data["case_or_control"] == "Case"
].copy()
genotyped_controls_mg = genotyped_mg_with_patient_data[
    genotyped_mg_with_patient_data["case_or_control"] == "Control"
].copy()

# for cases: filter for screening mammograms (> 3 months = 90 days before diagnosis)
genotyped_case_screening_mammograms = genotyped_cases_mg[
    genotyped_cases_mg["days_to_dx"] > 90
].copy()

# for cases: filter for mammograms anytime before and including diagnosis month
genotyped_cases_mg["mammogram_ym"] = genotyped_cases_mg["Study DateTime"].dt.to_period(
    "M"
)
genotyped_cases_mg["diagnosis_ym"] = genotyped_cases_mg["DatedxIndex"].dt.to_period("M")
genotyped_case_before_dx_month_mammograms = genotyped_cases_mg[
    genotyped_cases_mg["mammogram_ym"] <= genotyped_cases_mg["diagnosis_ym"]
].copy()

# for controls: all mammograms are considered (no diagnosis date filter)
genotyped_control_screening_mammograms = genotyped_controls_mg.copy()

# count screening mammograms per patient for cases
genotyped_case_screening_per_patient = (
    genotyped_case_screening_mammograms["study_id"].value_counts()
    if len(genotyped_case_screening_mammograms) > 0
    else pd.Series(dtype=int)
)

# count mammograms before/including diagnosis month per patient for cases
genotyped_case_before_dx_month_per_patient = (
    genotyped_case_before_dx_month_mammograms["study_id"].value_counts()
    if len(genotyped_case_before_dx_month_mammograms) > 0
    else pd.Series(dtype=int)
)

# count all mammograms per patient for controls
genotyped_control_screening_per_patient = (
    genotyped_control_screening_mammograms["study_id"].value_counts()
    if len(genotyped_control_screening_mammograms) > 0
    else pd.Series(dtype=int)
)

# create genotyped-only screening mammograms plot
create_screening_mammograms_plot(
    genotyped_case_screening_per_patient,
    genotyped_case_before_dx_month_per_patient,
    genotyped_control_screening_per_patient,
    genotyped_case_screening_mammograms,
    genotyped_case_before_dx_month_mammograms,
    genotyped_control_screening_mammograms,
    "genotyped",
    plots_dir,
)

# create genotyped-only time analysis
genotyped_cases_mg_for_time = genotyped_mg_with_patient_data[
    genotyped_mg_with_patient_data["case_or_control"] == "Case"
].copy()
genotyped_controls_mg_for_time = genotyped_mg_with_patient_data[
    genotyped_mg_with_patient_data["case_or_control"] == "Control"
].copy()

# for controls, calculate time from first mammogram
if len(genotyped_controls_mg_for_time) > 0:
    genotyped_controls_mg_for_time = genotyped_controls_mg_for_time.sort_values(
        "Study DateTime"
    )
    genotyped_first_mammogram_dates = genotyped_controls_mg_for_time.groupby(
        "study_id"
    )["Study DateTime"].first()
    genotyped_controls_mg_for_time = genotyped_controls_mg_for_time.merge(
        genotyped_first_mammogram_dates.rename("first_mammogram_date"),
        left_on="study_id",
        right_index=True,
    )
    genotyped_controls_mg_for_time["days_from_first_mammogram"] = (
        genotyped_controls_mg_for_time["Study DateTime"]
        - genotyped_controls_mg_for_time["first_mammogram_date"]
    ).dt.days

# create genotyped-only time analysis plot
create_time_analysis_plot(
    genotyped_cases_mg_for_time, genotyped_controls_mg_for_time, "genotyped", plots_dir
)

print("\n=== ANALYSIS COMPLETE ===")
print(f"plots saved to: {plots_dir.absolute()}")
print("created 'all' and 'genotyped' versions of all plots")
