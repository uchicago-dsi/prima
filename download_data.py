#!/usr/bin/env python
# coding: utf-8

import logging
import os

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

format_string = "%(asctime)s %(name)s:%(lineno)d [%(levelname)s]  %(message)s"
logging.basicConfig(filename="download.log", format=format_string, level=logging.DEBUG)


# replace with your username and password
username = "bar"
password = "foo"

# requires ssh keys are set up
remote_account = "annawoodard@gardner.cri.uchicago.edu"
remote_path = "/gpfs/data/huo-lab/Image/ChiMEC/"

# dx_time_after_exam_greater_than = "60 days"
# min_follow_up_time = "120 days"
dx_time_after_exam_greater_than = "0 days"
min_follow_up_time = "0 days"
max_pending_transfers = 600

original_metadata_path = "../data/interim/imaging_metadata.pkl"
metadata_with_download_updates_path = "../data/interim/downloaded_imaging_metadata.pkl"

if os.path.isfile(metadata_with_download_updates_path):
    metadata = pd.read_pickle(metadata_with_download_updates_path)
else:
    # created by the `exam_metadata.ipynb` noteboook
    metadata = pd.read_pickle(original_metadata_path)
    metadata["export_pending"] = False

print("loaded {} cases".format(metadata[metadata.case == True].study_id.nunique()))
print("loaded {} controls".format(metadata[metadata.case == False].study_id.nunique()))


def get_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("headless")
    driver = webdriver.Chrome(options=options)
    driver.get("http://cw2radiis03.uchad.uchospitals.edu/ibroker/")

    elem = driver.find_element_by_name("tbxUsername")
    elem.send_keys(username)
    elem.send_keys(Keys.RETURN)

    elem = driver.find_element_by_name("tbxPassword")
    elem.send_keys(password)
    elem.send_keys(Keys.RETURN)

    return driver


driver = get_driver()


def is_selected(metadata, date, case=True):
    # return True  # hack for more exams for pretraining
    if case is True:
        time_to_event = metadata[
            metadata["Study DateTime"] == date
        ].time_to_event.values[0]
        return time_to_event >= pd.Timedelta(dx_time_after_exam_greater_than)
    else:
        try:
            follow_up_time = metadata[
                metadata["Study DateTime"] == date
            ].follow_up_time.values[0]
            return follow_up_time >= pd.Timedelta(min_follow_up_time)
        except Exception as e:
            print(
                "encountered error while parsing last contact time after exam which took place on {}: {}".format(
                    date, e
                )
            )
            print(min_follow_up_time, date)
            return False


group = "control"
modality = "mammogram"
modality_tag = "MG"
# for group in ["control", "case"]:
for group in ["case", "control"]:
    if group == "case":
        cut = (metadata.case == True) & (
            metadata["time_to_event"] > pd.Timedelta(dx_time_after_exam_greater_than)
        )
    else:
        cut = (metadata.case == False) & (
            metadata["follow_up_time"] > pd.Timedelta(min_follow_up_time)
        )

    for modality, modality_tag in [("mammogram", "MG"), ("MRI", "MR")]:
        # for modality, modality_tag in [("MRI", "MR"), ("mammogram", "MG")]:
        if modality == "mammogram":
            cut = cut & (metadata["Modality"].str.contains("MG"))
        else:
            cut = (
                cut
                & (metadata["Modality"].str.contains("MR"))
                & (metadata["StudyDescription"].str.contains("BREAST"))
            )
        print(
            "starting {} {}s ({} unique women)".format(
                modality, group, len(metadata.loc[cut].study_id.unique())
            )
        )
        metadata.loc[~pd.isnull(metadata["exam_id"]), "export_pending"] = False
        pending_transfers = len(
            metadata.loc[cut & (metadata["export_pending"] == True)]
        )
        print(
            "there are {} remaining transfers; {} completed transfers; {} pending transfers".format(
                len(metadata.loc[cut & pd.isnull(metadata["exam_id"])]),
                len(metadata.loc[cut & ~pd.isnull(metadata["exam_id"])]),
                pending_transfers,
            )
        )

        while (
            len(
                metadata.loc[
                    cut
                    & (
                        pd.isnull(metadata["exam_id"])
                        | (metadata["export_pending"] == True)
                    )
                ]
            )
            > 0
        ):
            study_ids = (
                metadata.loc[
                    cut
                    & (
                        pd.isnull(metadata["exam_id"])
                        | (metadata["export_pending"] == True)
                    )
                ]
                .study_id.unique()
                .tolist()[::-1]
            )
            for study_id in study_ids:
                print("selected study id {}".format(study_id))
                print(
                    "missing exams: ",
                    metadata.loc[
                        cut
                        & (metadata.study_id == study_id)
                        & (pd.isnull(metadata["exam_id"]))
                    ][["Study DateTime"]].values,
                )
                print(
                    "incomplete transfers: ",
                    metadata.loc[
                        cut
                        & (metadata.study_id == study_id)
                        & (metadata["export_pending"] == True)
                    ][["Study DateTime"]].values,
                )
                try:
                    elem = driver.find_element_by_name("tbxAssignedID")
                except Exception as e:
                    logging.warning("catching exception: {}".format(e))
                    driver = get_driver()
                    elem = driver.find_element_by_name("tbxAssignedID")
                elem.clear()
                elem.send_keys(int(study_id))
                elem.send_keys(Keys.RETURN)

                # soup = BeautifulSoup(driver.page_source, "lxml")
                soup = BeautifulSoup(driver.page_source, "html")
                try:
                    table = soup.find_all("table")[3]
                except Exception:
                    print("could not find table for study id {}".format(study_id))
                    continue

                try:
                    df = pd.read_html(str(table), parse_dates=True, header=0)[0]
                    df["Study DateTime"] = pd.to_datetime(
                        df["Study DateTime"], infer_datetime_format=True
                    )
                except Exception:
                    print("could not parse table for study id {}".format(study_id))
                    continue

                print("processing table for study id {}".format(study_id))
                pending_transfers = len(
                    metadata.loc[cut & (metadata["export_pending"] == True)]
                )
                print(
                    "there are {} remaining transfers; {} completed transfers; {} pending transfers".format(
                        len(metadata.loc[cut & pd.isnull(metadata["exam_id"])]),
                        len(metadata.loc[cut & ~pd.isnull(metadata["exam_id"])]),
                        pending_transfers,
                    )
                )
                if len(table.find_all("tr")) == 0:
                    print("no rows found in table: {}".format(table))
                for index, row in enumerate(table.find_all("tr")):
                    pending_transfers = len(
                        metadata.loc[metadata["export_pending"] == True]
                    )
                    _, modality, date, description = row.find_all("td")
                    if modality_tag in modality.get_text():
                        title = row.get("title")
                        date = pd.to_datetime(date.text, infer_datetime_format=True)
                        try:
                            if not is_selected(metadata, date, case=(group == "case")):
                                try:
                                    print(
                                        "skipping unselected exam from {}".format(
                                            date.text
                                        )
                                    )
                                except AttributeError:
                                    try:
                                        print(
                                            "skipping unselected exam from {}".format(
                                                date
                                            )
                                        )
                                    except AttributeError:
                                        print(
                                            "skipping exam with unparseable date: {}".format(
                                                str(date)
                                            )
                                        )
                        except IndexError:
                            print(
                                "skipping exam from {} which produced an index error".format(
                                    date
                                )
                            )
                        else:
                            if title is not None and "Exported" in title:
                                if np.any(
                                    pd.isnull(
                                        metadata[
                                            metadata["Study DateTime"]
                                            == date.to_datetime64()
                                        ]["exam_id"]
                                    )
                                ):
                                    exam_id = title.split()[-1].split("\\")[-2]
                                    metadata.loc[
                                        metadata["Study DateTime"]
                                        == date.to_datetime64(),
                                        "exam_id",
                                    ] = exam_id
                                    metadata.loc[
                                        metadata["Study DateTime"]
                                        == date.to_datetime64(),
                                        "export_pending",
                                    ] = False
                                    print(
                                        "found exported exam from {} with id {}".format(
                                            date, exam_id
                                        )
                                    )
                            else:
                                if pending_transfers >= max_pending_transfers:
                                    continue
                                print(
                                    "exporting exam from {} for study id {}".format(
                                        date, study_id
                                    )
                                )
                                checkbox = driver.find_element_by_id(
                                    "TabContainer1_tabPanel1_gv1_ctl{}_chkExam".format(
                                        str(index + 1).zfill(2)
                                    )
                                )
                                checkbox.click()
                                export_button = driver.find_element_by_name("btnExport")
                                export_button.click()
                                logging.info(
                                    "exporting exam for study id {}".format(study_id)
                                )
                                metadata.loc[
                                    metadata["Study DateTime"] == date.to_datetime64(),
                                    "export_pending",
                                ] = True
                        metadata.to_pickle(
                            "../data/interim/downloaded_imaging_metadata.pkl"
                        )
