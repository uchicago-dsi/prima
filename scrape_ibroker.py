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
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm.auto import tqdm

# insert real values here; study id file should be a text file with a list of study ids, one per line
username = "annawoodard@uchicago.edu"
password = "16352a"
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


def activate_radiology_tab(driver) -> None:
    """ensure the 'radiology images' tab panel is visible

    clicks the tab header if the panel is still display:none; waits until it is displayed
    """
    panel = driver.find_element(By.ID, "TabContainer1_tabPanel1")
    if not panel.is_displayed():
        driver.find_element(By.ID, "__tab_TabContainer1_tabPanel1").click()
        WebDriverWait(driver, 30).until(
            lambda d: d.find_element(By.ID, "TabContainer1_tabPanel1").is_displayed()
        )


def select_all_modalities(driver) -> None:
    """select 'All' modalities so the results grid is populated

    calls the built-in __doPostBack via clicking the 'All' link
    """
    link = WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.ID, "lbAll")))
    link.click()
    wait_aspnet_idle(driver, 60)


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


def find_results_table_html(driver) -> str:
    """return outerHTML of the studies grid

    targets the GridView id 'TabContainer1_tabPanel1_gv1' directly, then falls back
    to a lightweight header-based search inside the radiology tab
    """
    activate_radiology_tab(driver)
    wait_aspnet_idle(driver, 60)

    # direct by id – fastest and most reliable on this WebForms page
    html = driver.execute_script(
        "var t=document.getElementById('TabContainer1_tabPanel1_gv1');"
        "return t ? t.outerHTML : null;"
    )
    if html:
        return html

    # lightweight fallback: pick the table whose first header row contains modality/studydt/studydescription
    js = r"""
    (function(){
      function norm(s){return (s||'').toLowerCase().replace(/\s+/g,' ').trim();}
      function score(tbl){
        var first = tbl.querySelector('tr'); if(!first) return 0;
        var cells = Array.from(first.children).map(td => norm(td.textContent));
        var s = 0;
        if (cells.some(t=>t.includes('modality'))) s += 3;
        if (cells.some(t=>t.includes('studydt')||t.includes('study date')||t.includes('study datetime'))) s += 3;
        if (cells.some(t=>t.includes('studydescription')||t.includes('study description'))) s += 2;
        return s;
      }
      var panel = document.getElementById('TabContainer1_tabPanel1') || document.body;
      var best=null, bestScore=0;
      for (var t of panel.querySelectorAll('table')) {
        var sc = score(t);
        if (sc > bestScore) { best = t; bestScore = sc; }
      }
      return best ? best.outerHTML : null;
    })();
    """
    html = driver.execute_script(js)
    if not html:
        # fail-fast with a forensics snapshot of what we actually saw
        panel_html = driver.execute_script(
            "var p=document.getElementById('TabContainer1_tabPanel1');"
            "return p ? p.outerHTML : document.body.outerHTML;"
        )
        with open("data/debug_last_panel.html", "w") as f:
            f.write(panel_html)
        raise TimeoutException(
            "studies grid not found (expected id 'TabContainer1_tabPanel1_gv1')"
        )
    return html


def parse_results_table_html(table_html: str) -> pd.DataFrame:
    """parse the studies grid into a dataframe

    normalizes 'StudyDT' → 'Study DateTime', drops the leading checkbox column,
    extracts exam_id when present (rare here), filters to MG/MR
    """
    df = pd.read_html(table_html, header=0)[0]  # pandas guarantees a list or raises
    df.columns = [c.strip() for c in df.columns]

    # drop the unnamed checkbox column if present
    for c in list(df.columns):
        if not c or c.lower().startswith("unnamed"):
            df = df.drop(columns=[c])

    # normalize column names we care about
    rename = {}
    for c in list(df.columns):
        lc = c.lower().replace(" ", "")
        if lc in {"studydt", "studydatetime", "studydate"}:
            rename[c] = "Study DateTime"
        elif lc.startswith("studydescription"):
            rename[c] = "StudyDescription"
        elif lc.startswith("modality"):
            rename[c] = "Modality"
    if rename:
        df = df.rename(columns=rename)

    required = {"Modality", "Study DateTime", "StudyDescription"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"grid missing expected columns: {missing}")

    df["Study DateTime"] = pd.to_datetime(df["Study DateTime"], errors="coerce")

    # optional exam_id inference: rows in this grid typically have no title/export info
    soup = BeautifulSoup(table_html, "lxml")
    rows = soup.find_all("tr")[1:]
    cells = [r.find_all("td") for r in rows]

    # find the dt column index from the header we just parsed to avoid guessing
    header_cells = [
        h.get_text(strip=True).lower().replace(" ", "")
        for h in soup.find("tr").find_all(["th", "td"])
    ]
    try:
        dt_idx = header_cells.index("studydt")
    except ValueError:
        # backup: look for 'study date' / 'study datetime'
        dt_idx = next(
            (
                i
                for i, t in enumerate(header_cells)
                if "study" in t and ("date" in t or "datetime" in t)
            ),
            2,
        )

    dates = [
        pd.to_datetime(c[dt_idx].get_text(strip=True), errors="coerce")
        if len(c) > dt_idx
        else pd.NaT
        for c in cells
    ]
    titles = [r.get("title") or "" for r in rows]
    exam_ids = [
        t.split()[-1].split("\\")[-2] if "exported" in t.lower() else np.nan
        for t in titles
    ]

    id_map = (
        pd.DataFrame({"Study DateTime": dates, "exam_id": exam_ids})
        .dropna(subset=["Study DateTime"])
        .drop_duplicates("Study DateTime")
    )
    out = df.merge(id_map, on="Study DateTime", how="left")

    # keep modalities you actually want
    out = out[out["Modality"].str.contains(r"\b(MG|MR)\b", na=False)]
    return out


driver = make_driver()
login(driver, username, password)

result = None
os.makedirs(os.path.dirname(output_file), exist_ok=True)

if os.path.isfile(output_file):
    result = pd.read_csv(output_file)
    study_ids = set(study_ids) - set(
        result[~pd.isnull(result["study_id"])].study_id.tolist()
    )
for study_id in tqdm(study_ids):
    set_study_id(study_id)
    activate_radiology_tab(driver)
    wait_aspnet_idle(driver, 60)

    table_html = find_results_table_html(driver)  # raises if missing
    df = parse_results_table_html(table_html)  # raises if schema changed
    df["study_id"] = study_id

    result = df if result is None else pd.concat([result, df], ignore_index=True)
    result.to_csv(output_file, index=False)
