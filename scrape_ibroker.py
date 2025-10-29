#!/usr/bin/env python

import os
import sys
import traceback

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

from filesystem_utils import build_disk_inventory

# --- required config: fail if you forget to set these ---
username = os.getenv("IBROKER_USERNAME")
password = os.getenv("IBROKER_PASSWORD")

BASE_DOWNLOAD_DIR = "/gpfs/data/huo-lab/Image/ChiMEC/MG"

# NEW: Fail fast if environment variables are not set
if not all([username, password]):
    print(
        "ERROR: IBROKER_USERNAME and IBROKER_PASSWORD environment variables must be set.",
        file=sys.stderr,
    )
    sys.exit(1)

study_id_file = "/gpfs/data/huo-lab/Image/ChiMEC/study-16352a.csv"
output_file = "data/imaging_metadata.csv"
BASE = "http://cw2radiis03.uchad.uchospitals.edu"

# read ids once
study_ids = pd.read_csv(study_id_file)["AnonymousID"].tolist()[::-1]


def make_driver():
    """return a headless firefox webdriver for login only"""
    o = Options()
    o.add_argument("--headless")
    o.add_argument("--width=1200")
    o.add_argument("--height=900")
    o.set_preference("permissions.default.image", 2)
    o.set_preference("browser.cache.disk.enable", False)
    o.set_preference("browser.cache.memory.enable", False)
    o.set_preference(
        "network.proxy.type", 0
    )  # force direct connections; ignore env proxy vars
    return webdriver.Firefox(options=o)


def wait_aspnet_idle(driver, timeout: int = 60) -> None:
    """block until document is complete and updatepanel is idle"""
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script(
            "var pm=(window.Sys&&Sys.WebForms&&Sys.WebForms.PageRequestManager.getInstance"
            "&&Sys.WebForms.PageRequestManager.getInstance())||null;"
            "return document.readyState==='complete' && !(pm && pm.get_isInAsyncPostBack());"
        )
    )


def login(driver, username: str, password: str) -> None:
    """log into ibroker and wait until the shell is ready"""
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
    """return a requests.Session that reuses selenium's cookies and user-agent"""
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
        " __EVENTARGUMENT": "",
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
    data["btnFetch"] = "Query Exams"
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
    """fast parse of the gv1 grid into a dataframe for ALL modalities"""
    root = lxml_html.fromstring(table_html)
    table_text = " ".join(root.xpath(".//text()")).strip().lower()
    if "no record found" in table_text or "no records found" in table_text:
        return pd.DataFrame(
            columns=["Modality", "Study DateTime", "StudyDescription", "exam_id"]
        )
    trs = root.xpath(".//tr")
    if len(trs) <= 1:
        return pd.DataFrame(
            columns=["Modality", "Study DateTime", "StudyDescription", "exam_id"]
        )
    rows = trs[1:]
    if all(len(r.xpath("./td")) <= 1 for r in rows):
        return pd.DataFrame(
            columns=["Modality", "Study DateTime", "StudyDescription", "exam_id"]
        )

    modality = [r.xpath("normalize-space(td[2])") for r in rows]
    dt_str = [r.xpath("normalize-space(td[3])") for r in rows]
    desc = [r.xpath("normalize-space(td[4])") for r in rows]
    dt = pd.to_datetime(pd.Series(dt_str, dtype="string"), errors="coerce")

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
    # Correctly not filtering by modality anymore
    return df


KNOWN_MODALITY_PREFIXES = [
    ("MRI", "MR"),
    ("MR ", "MR"),
    ("MAM", "MG"),
    ("MAM ", "MG"),
    ("CT", "CT"),
    ("XR", "CR"),
    ("US", "US"),
    ("NM", "NM"),
    ("PET", "PT"),
    ("PET/", "PT"),
]


def infer_modality_from_description(description: str | None) -> str | None:
    if not description or not isinstance(description, str):
        return None
    upper_desc = description.strip().upper()
    for prefix, modality in KNOWN_MODALITY_PREFIXES:
        if upper_desc.startswith(prefix):
            return modality
    if "BREAST" in upper_desc and "MAM" in upper_desc:
        return "MG"
    if "BREAST" in upper_desc and "MRI" in upper_desc:
        return "MR"
    if "BRAIN" in upper_desc and "MRI" in upper_desc:
        return "MR"
    return None


def parse_all_tables_from_page(page_html: str) -> pd.DataFrame:
    """
    Parses the full HTML page to find and extract data from both the "Exported"
    and "Available" studies tables, returning them as a single combined DataFrame.

    This version is robust to missing table IDs and missing tables entirely.
    """
    doc = lxml_html.fromstring(page_html)
    dfs_to_combine = []

    # --- Part 1: Parse the "Exported" table ---
    # NEW ROBUST XPATH: Find the table that contains a header with the text "Accession".
    # This is much more reliable than relying on an ID that may not exist.
    exported_table_nodes = doc.xpath(
        '//table[.//th[contains(text(), "Accession")] or .//td[contains(text(), "Accession")]]'
    )

    # Sometimes the header is not in a <th> but a <td>, so we check both.
    # The first table found is almost certainly the one we want.
    if not exported_table_nodes:
        # Fallback for older GridViews that might use different structures
        exported_table_nodes = doc.xpath(
            '//table[contains(.//tr[1]/td[4]/text(), "Accession")]'
        )

    if exported_table_nodes:
        root = exported_table_nodes[0]
        trs = root.xpath(".//tr")
        if len(trs) > 1 and "no record" not in root.text_content().lower():
            rows = trs[1:]
            # Columns: StudyDT, StudyDescription, Status, Accession, Exported On
            dt_str = [r.xpath("normalize-space(td[1])") for r in rows]
            desc = [r.xpath("normalize-space(td[2])") for r in rows]
            status = [r.xpath("normalize-space(td[3])") for r in rows]
            accession = [r.xpath("normalize-space(td[4])") for r in rows]
            exported_on_str = [r.xpath("normalize-space(td[5])") for r in rows]

            df_exported = pd.DataFrame(
                {
                    "Study DateTime": pd.to_datetime(
                        pd.Series(dt_str), errors="coerce"
                    ),
                    "StudyDescription": desc,
                    "Status": status,
                    "Accession": accession,
                    "Exported On": pd.to_datetime(
                        pd.Series(exported_on_str), errors="coerce"
                    ),
                }
            )
            df_exported["Modality"] = df_exported["StudyDescription"].map(
                infer_modality_from_description
            )
            dfs_to_combine.append(df_exported)

    # --- Part 2: Parse the "Available" table (gv1) if it exists ---
    # Your existing logic for this is fine as it seems to have a reliable ID.
    available_table_nodes = doc.xpath("//table[@id='TabContainer1_tabPanel1_gv1']")
    if available_table_nodes:
        root = available_table_nodes[0]
        trs = root.xpath(".//tr")
        if len(trs) > 1 and "no record" not in root.text_content().lower():
            rows = trs[1:]
            modality = [r.xpath("normalize-space(td[2])") for r in rows]
            dt_str = [r.xpath("normalize-space(td[3])") for r in rows]
            desc = [r.xpath("normalize-space(td[4])") for r in rows]

            df_available = pd.DataFrame(
                {
                    "Modality": modality,
                    "Study DateTime": pd.to_datetime(
                        pd.Series(dt_str), errors="coerce"
                    ),
                    "StudyDescription": desc,
                }
            )
            dfs_to_combine.append(df_available)

    # --- Part 3: Combine and return ---
    if not dfs_to_combine:
        return pd.DataFrame(
            columns=[
                "Modality",
                "Study DateTime",
                "StudyDescription",
                "Status",
                "Accession",
                "Exported On",
            ]
        )

    combined = pd.concat(dfs_to_combine, ignore_index=True, sort=False)

    required_cols = {
        "Modality": pd.Series(dtype="string"),
        "Study DateTime": pd.Series(dtype="datetime64[ns]"),
        "StudyDescription": pd.Series(dtype="string"),
        "Status": pd.Series(dtype="string"),
        "Accession": pd.Series(dtype="string"),
        "Exported On": pd.Series(dtype="datetime64[ns]"),
    }
    for col, template in required_cols.items():
        if col not in combined.columns:
            combined[col] = template

    combined["StudyDescription"] = combined["StudyDescription"].astype("string")
    combined["Accession"] = combined["Accession"].astype("string").replace({"": pd.NA})

    combined["Modality"] = combined["Modality"].astype("string").replace({"": pd.NA})
    missing_modality = combined["Modality"].isna()
    if missing_modality.any():
        combined.loc[missing_modality, "Modality"] = combined.loc[
            missing_modality, "StudyDescription"
        ].map(infer_modality_from_description)

    sort_cols = ["Study DateTime", "StudyDescription"]
    if combined["Accession"].notna().any():
        sort_cols.append("Accession")

    combined.sort_values(by=sort_cols, inplace=True)

    subset_cols = ["Study DateTime", "StudyDescription"]
    if combined["Accession"].notna().any():
        subset_cols.append("Accession")

    combined.drop_duplicates(subset=subset_cols, keep="first", inplace=True)

    return combined.reset_index(drop=True)


def main():
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
        if driver:
            print("Closing webdriver session...")
            driver.quit()
            print("Webdriver closed.")

    try:
        disk_inventory = build_disk_inventory(BASE_DOWNLOAD_DIR)

        page_html, state = http_get_root(session)
        page_html, state = post_link_event(session, state, "lbAll")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        output_exists = os.path.isfile(output_file)

        if output_exists and os.path.getsize(output_file) > 0:
            print(f"Output file '{output_file}' found. Reading seen IDs to resume...")
            try:
                seen_df = pd.read_csv(output_file, usecols=["study_id"])
                seen = set(seen_df["study_id"].astype(str))
                study_ids_to_process = [
                    sid for sid in study_ids if str(sid) not in seen
                ]
                print(
                    f"{len(seen)} IDs already processed. {len(study_ids_to_process)} remaining."
                )
            except (pd.errors.EmptyDataError, KeyError, ValueError):
                print("Output file is empty or invalid. Processing all IDs.")
                study_ids_to_process = study_ids
        else:
            print(
                f"Output file '{output_file}' not found or is empty. Starting a new run."
            )
            study_ids_to_process = study_ids

        all_columns = [
            "Modality",
            "Study DateTime",
            "StudyDescription",
            "Status",
            "Accession",
            "Exported On",
            "study_id",
            "is_on_disk",
        ]

        with open(output_file, "a", newline="") as f:
            if not output_exists or os.path.getsize(output_file) == 0:
                pd.DataFrame(columns=all_columns).to_csv(f, index=False, header=True)

            for study_id in tqdm(study_ids_to_process, desc="Scraping Study IDs"):
                try:
                    page_html, state = post_fetch_grid(session, state, str(study_id))
                    df = parse_all_tables_from_page(page_html)

                    if df.empty:
                        df = pd.DataFrame(columns=all_columns)

                    df["StudyDescription"] = df["StudyDescription"].astype("string")
                    df["Modality"] = df.get("Modality", pd.Series(dtype="string"))
                    df["Modality"] = df["Modality"].astype("string")
                    missing_modalities = df["Modality"].isna() | (df["Modality"] == "")
                    if missing_modalities.any():
                        df.loc[missing_modalities, "Modality"] = df.loc[
                            missing_modalities, "StudyDescription"
                        ].map(infer_modality_from_description)

                    df["study_id"] = study_id
                    df = df.reindex(columns=all_columns)

                    def check_row_on_disk(row):
                        accession_val = row.get("Accession")
                        if pd.notna(accession_val):
                            patient_id_str = str(int(row["study_id"]))
                            accession_str = str(accession_val)
                            patient_inventory = disk_inventory.get(
                                patient_id_str, set()
                            )
                            if accession_str in patient_inventory:
                                return True
                        return False

                    df["is_on_disk"] = False
                    on_disk_mask = df.apply(check_row_on_disk, axis=1)
                    df.loc[on_disk_mask, "is_on_disk"] = True

                    df.to_csv(f, index=False, header=False)
                    f.flush()

                except (requests.exceptions.RequestException, ValueError) as e:
                    print(
                        f"\nWARNING: Failed to process study_id {study_id}. Error: {e}"
                    )
                    print("Re-initializing session state and continuing to next ID...")
                    page_html, state = http_get_root(session)
                    continue

        print("Scraping complete.")

    except Exception as e:
        print(f"\nAn unexpected fatal error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
