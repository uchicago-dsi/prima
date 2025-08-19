#!/usr/bin/env python
# coding: utf-8

# This script runs through a list of study ids, pulls down the imaging table produced in iBroker, concatenates the tables together, and saves them to an output csv file.
#
# # Installation
# To install the python requirements, run `pip install -r requirements.txt`
#
# Selenium additionally requires the installation of a driver for your browser (Chrome in this example). See here for details: https://selenium-python.readthedocs.io/installation.html

import os

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm.auto import tqdm

# insert real values here; study id file should be a text file with a list of study ids, one per line
username = "annawoodard@uchicago.edu"
password = "foo"
study_id_file = "/gpfs/data/huo-lab/Image/ChiMEC/study-16352a.csv"
output_file = "data/imaging_metadata.csv"


study_ids = pd.read_csv(study_id_file)["AnonymousID"].tolist()[::-1]


def make_driver():
    """return a headless firefox webdriver using conda-forge firefox+geckodriver

    assumes packages were installed in the active conda env
    """
    o = Options()
    o.add_argument("--headless")
    o.add_argument("--width=1600")
    o.add_argument("--height=1200")
    return webdriver.Firefox(options=o)


def login(driver, username: str, password: str) -> None:
    """log into ibroker and wait until the main form is ready

    waits for postbacks explicitly: the username submit causes a redraw, so we
    wait for that element to go stale, then reacquire the password field fresh
    """
    driver.get("http://cw2radiis03.uchad.uchospitals.edu/ibroker/")

    u = WebDriverWait(driver, 30).until(
        EC.visibility_of_element_located((By.NAME, "tbxUsername"))
    )
    u.clear()
    u.send_keys(username)
    u.send_keys(Keys.RETURN)

    # the username submit triggers a (partial) postback; wait for that element to be detached
    WebDriverWait(driver, 30).until(EC.staleness_of(u))
    wait_aspnet_idle(driver, 60)

    # reacquire password field from the new DOM; don't use 'clickable' here
    p = WebDriverWait(driver, 30).until(
        EC.visibility_of_element_located((By.NAME, "tbxPassword"))
    )
    p.clear()
    p.send_keys(password)
    p.send_keys(Keys.RETURN)

    # wait for the logged-in shell and assigned-id field
    wait_aspnet_idle(driver, 60)
    WebDriverWait(driver, 30).until(EC.visibility_of_element_located((By.ID, "lbUser")))
    WebDriverWait(driver, 30).until(
        EC.visibility_of_element_located((By.ID, "tbxAssignedID"))
    )


def wait_aspnet_idle(driver, timeout: int = 60) -> None:
    """block until document is complete and asp.net updatepanel is not mid-ajax

    uses PageRequestManager.get_isInAsyncPostBack when present
    """
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script(
            "var pm=(window.Sys&&Sys.WebForms&&Sys.WebForms.PageRequestManager.getInstance"
            "&&Sys.WebForms.PageRequestManager.getInstance())||null;"
            "return document.readyState==='complete' && !(pm && pm.get_isInAsyncPostBack());"
        )
    )


def set_study_id(study_id: str) -> None:
    """set the study id and fetch results then wait for the async update to settle"""
    field = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable((By.ID, "tbxAssignedID"))
    )
    field.clear()
    field.send_keys(study_id)
    driver.find_element(By.ID, "btnFetch").click()
    wait_aspnet_idle(driver, 60)


def find_results_table_element(driver):
    """locate the radiology results grid without table indexing

    prefers the grid under the radiology images tab with a header 'Study DateTime'
    falls back to the first table after the 'available studies to export' marker
    """
    wait_aspnet_idle(driver, 60)
    try:
        return WebDriverWait(driver, 20).until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    "//div[@id='TabContainer1_tabPanel1']"
                    "//table[.//th[contains(normalize-space(.),'Study DateTime')]]",
                )
            )
        )
    except TimeoutException:
        # fallback to the first table following the marker text
        return WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    "//span[contains(@class,'highlight') and "
                    "contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),"
                    "'available studies to export')]/following::table[1]",
                )
            )
        )


def parse_results_table_html(table_html: str) -> pd.DataFrame:
    """parse the ibroker results table into a dataframe with exam_id

    infers exam_id from each row's title when containing 'Exported'
    filters modalities to MG and MR
    """
    soup = BeautifulSoup(table_html, "lxml")

    # read the table header/body through pandas
    df = pd.read_html(str(soup), parse_dates=True, header=0)[0]
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["Study DateTime"] = pd.to_datetime(df["Study DateTime"], errors="coerce")

    # build an exam_id map from the row title attributes
    rows = soup.find_all("tr")[1:]
    tds = [r.find_all("td") for r in rows]
    dates = [
        pd.to_datetime(c[2].get_text(strip=True), errors="coerce")
        if len(c) >= 3
        else pd.NaT
        for c in tds
    ]
    titles = [r.get("title") or "" for r in rows]
    exam_ids = [
        t.split()[-1].split("\\")[-2] if "Exported" in t else np.nan for t in titles
    ]

    id_map = (
        pd.DataFrame({"Study DateTime": dates, "exam_id": exam_ids})
        .dropna(subset=["Study DateTime"])
        .drop_duplicates("Study DateTime")
    )
    out = df.merge(id_map, on="Study DateTime", how="left")
    out = out[out["Modality"].str.contains("MG|MR", na=False)]
    return out


driver = make_driver()
login(driver, username, password)

result = None
os.makedirs(os.path.dirname(output_file), exist_ok=True)

if os.path.isfile(output_file):
    result = pd.read_csv(output_file)
    study_ids = set(study_ids) - set(
        result[~pd.isnull(result["exam_id"])].study_id.tolist()
    )
for study_id in tqdm(study_ids):
    try:
        set_study_id(study_id)
        table_el = find_results_table_element(driver)
        df = parse_results_table_html(table_el.get_attribute("outerHTML"))
    except (TimeoutException, ValueError, KeyError, IndexError, NoSuchElementException):
        # no grid or no rows for this id
        df = pd.DataFrame(
            {
                "Study DateTime": [np.nan],
                "StudyDescription": [np.nan],
                "Modality": [np.nan],
            }
        )

    df["study_id"] = study_id

    if result is None:
        result = df
    else:
        # keep order stable and avoid index collisions
        result = pd.concat([result, df], ignore_index=True)

    result.to_csv(output_file, index=False)
