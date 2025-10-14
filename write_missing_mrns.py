#!/usr/bin/env python3
"""
helper script to identify MRNs in chimec_patients that are not in the key table
and write them to missing_mrns.txt
"""

import pandas as pd


def main():
    # load data files
    chimec_patients = pd.read_csv(
        "/gpfs/data/phs/groups/Projects/Huo_projects/SPORE/annawoodard/List_ChiMEC_priority_2025July30.csv"
    )
    key = pd.read_csv("/gpfs/data/huo-lab/Image/ChiMEC/study-16352a.csv")

    # find MRNs in chimec_patients but not in key
    missing_mrns = chimec_patients[~chimec_patients["MRN"].isin(key["MRN"])][
        "MRN"
    ].tolist()
    print(f"Number of MRNs in chimec_patients but not in key: {len(missing_mrns)}")

    # save missing MRNs to text file
    with open("missing_mrns.txt", "w") as f:
        for mrn in missing_mrns:
            f.write(f"{mrn}\n")

    print(f"Saved {len(missing_mrns)} missing MRNs to missing_mrns.txt")


if __name__ == "__main__":
    main()
