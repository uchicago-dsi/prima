#!/usr/bin/env python
# coding: utf-8

# # What does this script do?
# This script runs through a list of study ids, pulls down the imaging table produced in iBroker, concatenates the tables together, and saves them to an output csv file.
#
# # Installation
# To install the python requirements, run `pip install -r requirements.txt`
#
# Selenium additionally requires the installation of a driver for your browser (Chrome in this example). See here for details: https://selenium-python.readthedocs.io/installation.html

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm.auto import tqdm

# insert real values here; study id file should be a text file with a list of study ids, one per line
username = "bar"
password = "foo"
study_id_file = "../data/raw/mrn_to_study_id.csv"


study_ids = pd.read_csv(study_id_file, header=None, names=["id"])["id"].tolist()[::-1]

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


@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
def set_study_id(study_id):
    elem = driver.find_element_by_name("tbxAssignedID")
    elem.clear()
    elem.send_keys(study_id)
    elem.send_keys(Keys.RETURN)


result = None
# if os.path.isfile("../data/raw/imaging_metadata.csv"):
#     result = pd.read_csv("../data/raw/imaging_metadata.csv")
#     study_ids = set(study_ids) -         set(result[~pd.isnull(result["exam_id"])].study_id.tolist())
for study_id in tqdm(study_ids):

    try:
        set_study_id(study_id)
        soup = BeautifulSoup(driver.page_source, "lxml")
        table = soup.find_all("table")[3]

        df = pd.read_html(str(table), parse_dates=True, header=0)[0]
        df["Study DateTime"] = pd.to_datetime(df["Study DateTime"])
        df.drop(["Unnamed: 0"], inplace=True, axis=1)
        df["exam_id"] = np.nan
        for index, row in enumerate(table.find_all("tr")[1:]):
            _, _, date, _ = row.find_all("td")
            title = row.get("title")
            date = pd.to_datetime(date.text)
            if title is not None and "Exported" in title:
                exam_id = title.split()[-1].split("\\")[-2]
                df.loc[df["Study DateTime"] == date, "exam_id"] = exam_id
        df = df[df.Modality.str.contains("MG|MR")]
    except (KeyError, IndexError, NoSuchElementException) as e:
        d = {
            "Study DateTime": [np.nan],
            "StudyDescription": [np.nan],
            "Modality": [np.nan],
        }
        df = pd.DataFrame(data=d)
    df["study_id"] = study_id

    if result is None:
        result = df
    else:
        result = pd.concat([df, result])

    result.to_csv("../data/raw/imaging_metadata.csv", index=False)
