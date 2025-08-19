#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
import pandas as pd
import requests
from lxml import html as lxml_html
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm.auto import tqdm

# --- required config: fail if you forget to set these ---
username = os.getenv("IBROKER_USERNAME")
password = os.getenv("IBROKER_PASSWORD")
study_id_file = "/gpfs/data/huo-lab/Image/ChiMEC/study-16352a.csv"
output_file = "data/imaging_metadata.csv"
BASE = "http://cw2radiis03.uchad.uchospitals.edu"

# read ids once
study_ids = pd.read_csv(study_id_file)["AnonymousID"].tolist()[::-1]


def make_driver():
    """return a headless firefox webdriver for login only

    blocks images to reduce bandwidth and layout work
    """
    o = Options()
    o.add_argument("--headless")
    o.add_argument("--width=1200")
    o.add_argument("--height=900")
    o.set_preference("permissions.default.image", 2)  # block images
    o.set_preference("browser.cache.disk.enable", False)
    o.set_preference("browser.cache.memory.enable", False)
    return webdriver.Firefox(options=o)


def wait_aspnet_idle(driver, timeout: int = 60) -> None:
    """block until document is complete and updatepanel is idle

    uses PageRequestManager.get_isInAsyncPostBack when present
    """
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script(
            "var pm=(window.Sys&&Sys.WebForms&&Sys.WebForms.PageRequestManager.getInstance"
            "&&Sys.WebForms.PageRequestManager.getInstance())||null;"
            "return document.readyState==='complete' && !(pm && pm.get_isInAsyncPostBack());"
        )
    )


def login(driver, username: str, password: str) -> None:
    """log into ibroker and wait until the shell is ready

    reacquires elements across postbacks to avoid stale references
    """
    driver.get(f"{BASE}/ibroker/")
    u = WebDriverWait(driver, 30).until(
        EC.visibility_of_element_located((By.NAME, "tbxUsername"))
    )
    u.clear()
    u.send_keys(username)
    u.send_keys(Keys.RETURN)
    WebDriverWait(driver, 30).until(EC.staleness_of(u))
    wait_aspnet_idle(driver, 60)

    p = WebDriverWait(driver, 30).until(
        EC.visibility_of_element_located((By.NAME, "tbxPassword"))
    )
    p.clear()
    p.send_keys(password)
    p.send_keys(Keys.RETURN)

    wait_aspnet_idle(driver, 60)
    WebDriverWait(driver, 30).until(EC.visibility_of_element_located((By.ID, "lbUser")))
    WebDriverWait(driver, 30).until(
        EC.visibility_of_element_located((By.ID, "tbxAssignedID"))
    )


def bootstrap_http_session_from_driver(driver) -> requests.Session:
    """return a requests.Session that reuses selenium's cookies and user-agent

    keeps server-side auth while avoiding webdriver overhead
    """
    s = requests.Session()
    for c in driver.get_cookies():
        s.cookies.set(c["name"], c["value"], domain=c.get("domain"), path=c.get("path"))
    ua = driver.execute_script("return navigator.userAgent;")
    s.headers.update({"User-Agent": ua})
    return s


def extract_webforms_state(page_html: str) -> dict:
    """extract viewstate/eventvalidation/clientstate for the next postback"""
    doc = lxml_html.fromstring(page_html)

    def val(i):
        v = doc.xpath(f"//input[@id='{i}']/@value")
        return v[0] if v else ""

    return {
        "__VIEWSTATE": val("__VIEWSTATE"),
        "__VIEWSTATEGENERATOR": val("__VIEWSTATEGENERATOR"),
        "__EVENTVALIDATION": val("__EVENTVALIDATION"),
        "TabContainer1_ClientState": val("TabContainer1_ClientState"),
        "__EVENTTARGET": "",
        "__EVENTARGUMENT": "",
        "__LASTFOCUS": "",
    }


def http_get_root(session: requests.Session) -> tuple[str, dict]:
    """GET the page to seed viewstate; error if not authenticated"""
    r = session.get(f"{BASE}/ibroker/iBroker.aspx", timeout=30)
    r.raise_for_status()
    html = r.text
    if ("tbxUsername" in html) and ("tbxPassword" in html):
        os.makedirs("data", exist_ok=True)
        with open("data/debug_login_page.html", "w") as f:
            f.write(html)
        raise RuntimeError(
            "http session is not authenticated (see data/debug_login_page.html)"
        )
    return html, extract_webforms_state(html)


def post_link_event(
    session: requests.Session, state: dict, target: str
) -> tuple[str, dict]:
    """fire LinkButton via __EVENTTARGET and return updated state"""
    data = dict(state)
    data["__EVENTTARGET"] = target
    data["__EVENTARGUMENT"] = ""
    r = session.post(f"{BASE}/ibroker/iBroker.aspx", data=data, timeout=30)
    r.raise_for_status()
    return r.text, extract_webforms_state(r.text)


def post_fetch_grid(
    session: requests.Session, state: dict, study_id: str
) -> tuple[str, dict]:
    """submit the 'Query Exams' action for study_id and return response + next state"""
    data = dict(state)
    data["tbxAssignedID"] = study_id
    data["btnFetch"] = "Query Exams"  # Button posts name/value, not __EVENTTARGET
    r = session.post(f"{BASE}/ibroker/iBroker.aspx", data=data, timeout=30)
    r.raise_for_status()
    return r.text, extract_webforms_state(r.text)


def extract_gv1_table_html(page_html: str) -> str:
    """return the gv1 studies grid html from a full page; write forensics on miss"""
    doc = lxml_html.fromstring(page_html)
    nodes = doc.xpath("//table[@id='TabContainer1_tabPanel1_gv1']")
    if not nodes:
        os.makedirs("data", exist_ok=True)
        with open("data/debug_last_page.html", "w") as f:
            f.write(page_html)
        raise ValueError("gv1 grid not found in response")
    return lxml_html.tostring(nodes[0], method="html", encoding="unicode")


def parse_results_table_html(table_html: str) -> pd.DataFrame:
    """fast parse of the gv1 grid into a dataframe

    handles three cases gracefully:
    - normal data rows under a header
    - explicit empty row (e.g., 'No record(s) found.')
    - header-only grid (ShowHeaderWhenEmpty) or shell with no rows

    returns an empty frame with canonical columns when the grid has no data
    """
    root = lxml_html.fromstring(table_html)

    # quick sentinel: many GridView configs render an EmptyRow with 'No record(s) found'
    table_text = " ".join(root.xpath(".//text()")).strip().lower()
    if "no record found" in table_text or "no records found" in table_text:
        return pd.DataFrame(
            columns=["Modality", "Study DateTime", "StudyDescription", "exam_id"]
        )

    # collect all rows; GridView often uses <td> even for the header
    trs = root.xpath(".//tr")
    if len(trs) <= 1:
        # header-only or truly empty grid → zero-row df with canonical schema
        return pd.DataFrame(
            columns=["Modality", "Study DateTime", "StudyDescription", "exam_id"]
        )

    # skip the header row; assume subsequent rows are data or an EmptyRow variant
    rows = trs[1:]

    # if every "data" row is actually a single colspan cell, treat as empty
    only_single_cell_rows = all(len(r.xpath("./td")) <= 1 for r in rows)
    if only_single_cell_rows:
        return pd.DataFrame(
            columns=["Modality", "Study DateTime", "StudyDescription", "exam_id"]
        )

    # extract columns by index; first <td> is the checkbox column
    modality = [r.xpath("normalize-space(td[2])") for r in rows]
    dt_str = [r.xpath("normalize-space(td[3])") for r in rows]
    desc = [r.xpath("normalize-space(td[4])") for r in rows]

    # vectorized datetime parse; unparsable → NaT
    dt = pd.to_datetime(pd.Series(dt_str, dtype="string"), errors="coerce")

    # derive exam_id from row title when it contains 'Exported ... \\EXAMID\\...'
    titles = np.array([r.get("title") or "" for r in rows], dtype=str)
    mask = np.char.find(np.char.lower(titles), "exported") >= 0
    exam = np.full(titles.shape, np.nan, dtype=object)
    if mask.any():
        extracted = np.array(
            [t.split()[-1].split("\\")[-2] for t in titles[mask]], dtype=object
        )
        exam[np.where(mask)[0]] = extracted

    df = pd.DataFrame(
        {
            "Modality": np.array(modality, dtype=object),
            "Study DateTime": dt,
            "StudyDescription": np.array(desc, dtype=object),
            "exam_id": exam,
        }
    )

    # keep MG/MR; if this empties the frame, that's still a valid 'no MG/MR for this id'
    # df = df[df["Modality"].str.contains(r"\b(MG|MR)\b", na=False)]
    return df


# --- run ---
os.makedirs(os.path.dirname(output_file), exist_ok=True)

output_exists = os.path.isfile(output_file)
# Initialize driver to None outside the try block
driver = None
try:
    print("Starting headless Firefox driver...")
    driver = make_driver()
    print("Driver started. Logging in...")
    login(driver, username, password)
    print("Login successful. Bootstrapping HTTP session...")

    session = bootstrap_http_session_from_driver(driver)
    print("HTTP session created. Browser is no longer needed.")

finally:
    # This block ALWAYS runs, ensuring the browser closes.
    # We put it here to close the browser as soon as its job is done,
    # or if any of the startup steps fail.
    if driver:
        print("Closing webdriver session...")
        driver.quit()
        print("Webdriver closed.")

# The rest of the script now runs without the webdriver, using the faster session.

# --- Main scraping logic using the HTTP session ---
try:
    # Seed state via GET, then ensure modalities are 'All' via LinkButton
    page_html, state = http_get_root(session)
    page_html, state = post_link_event(session, state, "lbAll")

    # Prepare for efficient, resumable scraping
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_exists = os.path.isfile(output_file)

    # Determine which study IDs still need to be processed
    if output_exists:
        print(f"Output file '{output_file}' found. Reading seen IDs to resume...")
        try:
            # Read only the necessary column to save memory
            seen_df = pd.read_csv(output_file, usecols=["study_id"])
            seen = set(seen_df["study_id"].astype(str))
            study_ids_to_process = [sid for sid in study_ids if str(sid) not in seen]
            print(
                f"{len(seen)} IDs already processed. {len(study_ids_to_process)} remaining."
            )
        except (pd.errors.EmptyDataError, KeyError):
            # Handle case where file exists but is empty or has no study_id column
            print("Output file is empty or invalid. Processing all IDs.")
            seen = set()
            study_ids_to_process = study_ids
    else:
        print(f"Output file '{output_file}' not found. Starting a new run.")
        study_ids_to_process = study_ids

    # Open the file in append mode ('a') for the duration of the loop
    with open(output_file, "a", newline="") as f:
        # Write the header row only if we are starting a new file
        if not output_exists or os.path.getsize(output_file) == 0:
            # Create a dummy empty dataframe just to write its headers
            header_df = pd.DataFrame(
                columns=[
                    "Modality",
                    "Study DateTime",
                    "StudyDescription",
                    "exam_id",
                    "study_id",
                ]
            )
            header_df.to_csv(f, index=False, header=True)

        # Main loop to process remaining IDs
        for study_id in tqdm(study_ids_to_process, desc="Scraping Study IDs"):
            try:
                page_html, state = post_fetch_grid(session, state, str(study_id))
                table_html = extract_gv1_table_html(page_html)
                df = parse_results_table_html(table_html)

                # If no relevant rows were found, skip to the next ID
                if df.empty:
                    continue

                df["study_id"] = study_id

                # Append the new data to the CSV file without writing the header again
                df.to_csv(f, index=False, header=False)

            except (requests.exceptions.RequestException, ValueError) as e:
                # Catch errors for a single ID without crashing the whole script
                print(f"\nWARNING: Failed to process study_id {study_id}. Error: {e}")
                print("Continuing to next ID...")
                # The state might be stale, so we re-fetch the root page to reset it
                page_html, state = http_get_root(session)
                continue  # Move to the next iteration of the loop

    print("Scraping complete.")

except Exception as e:
    # This will catch any unexpected errors during the session-based scraping
    print(f"\nAn unexpected error occurred during the scraping process: {e}")
    # You might want to log the error to a file here
    import traceback

    traceback.print_exc()
