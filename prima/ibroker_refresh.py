"""Shared iBroker querying and metadata refresh utilities."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from lxml import html as lxml_html
from tqdm.auto import tqdm

warnings.filterwarnings(
    "ignore",
    message=(
        "The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*"
    ),
    category=FutureWarning,
)

IBROKER_BASE_URL = "http://cw2radiis03.uchad.uchospitals.edu"

REQUIRED_METADATA_COLUMNS = [
    "Modality",
    "Study DateTime",
    "StudyDescription",
    "Status",
    "Accession",
    "Exported On",
    "study_id",
]

REQUESTED_STATUS_VALUES = {
    "requested",
    "request submitted",
    "start cmove",
    "study retrieved",
    "in progress",
    "queued",
}

EXPORTED_STATUS_VALUES = {
    "already exported",
    "exported",
    "completed",
}

FAILED_STATUS_VALUES = {
    "pacs error: unable to process",
    "no file exported, outside study or burn in phi",
    "failed",
    "error",
}

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


def make_driver():
    """Return a headless Firefox webdriver for iBroker login."""
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--width=1200")
    options.add_argument("--height=900")
    options.set_preference("permissions.default.image", 2)
    options.set_preference("browser.cache.disk.enable", False)
    options.set_preference("browser.cache.memory.enable", False)
    options.set_preference("network.proxy.type", 0)
    return webdriver.Firefox(options=options)


def wait_aspnet_idle(driver, timeout: int = 60) -> None:
    """Block until the ASP.NET page is fully idle."""
    from selenium.webdriver.support.ui import WebDriverWait

    WebDriverWait(driver, timeout).until(
        lambda drv: drv.execute_script(
            "var pm=(window.Sys&&Sys.WebForms&&Sys.WebForms.PageRequestManager.getInstance"
            "&&Sys.WebForms.PageRequestManager.getInstance())||null;"
            "return document.readyState==='complete' && !(pm && pm.get_isInAsyncPostBack());"
        )
    )


def login(driver, username: str, password: str) -> None:
    """Log into iBroker and wait for the search shell."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    driver.get(f"{IBROKER_BASE_URL}/ibroker/")
    username_input = WebDriverWait(driver, 30).until(
        EC.visibility_of_element_located((By.NAME, "tbxUsername"))
    )
    username_input.clear()
    username_input.send_keys(username)
    username_input.send_keys(Keys.RETURN)
    WebDriverWait(driver, 30).until(EC.staleness_of(username_input))
    wait_aspnet_idle(driver, 60)

    password_input = WebDriverWait(driver, 30).until(
        EC.visibility_of_element_located((By.NAME, "tbxPassword"))
    )
    password_input.clear()
    password_input.send_keys(password)
    password_input.send_keys(Keys.RETURN)
    wait_aspnet_idle(driver, 60)

    WebDriverWait(driver, 30).until(EC.visibility_of_element_located((By.ID, "lbUser")))
    WebDriverWait(driver, 30).until(
        EC.visibility_of_element_located((By.ID, "tbxAssignedID"))
    )


def bootstrap_http_session_from_driver(driver) -> requests.Session:
    """Create requests.Session from Selenium cookies and user-agent."""
    session = requests.Session()
    for cookie in driver.get_cookies():
        session.cookies.set(
            cookie["name"],
            cookie["value"],
            domain=cookie.get("domain"),
            path=cookie.get("path"),
        )
    session.headers.update(
        {"User-Agent": driver.execute_script("return navigator.userAgent;")}
    )
    return session


def extract_webforms_state(page_html: str) -> dict:
    """Extract all ASP.NET form fields needed for follow-up postbacks."""
    document = lxml_html.fromstring(page_html)
    state: dict[str, str] = {}

    for input_node in document.xpath("//input[@name]"):
        name = input_node.get("name", "")
        input_type = (input_node.get("type") or "text").lower()
        if input_type == "checkbox":
            if input_node.get("checked") is not None:
                state[name] = input_node.get("value", "on")
        elif input_type == "radio":
            if input_node.get("checked") is not None:
                state[name] = input_node.get("value", "")
        elif input_type != "submit":
            state[name] = input_node.get("value", "")

    for select_node in document.xpath("//select[@name]"):
        name = select_node.get("name", "")
        selected = select_node.xpath(".//option[@selected]/@value")
        state[name] = selected[0] if selected else ""

    for textarea_node in document.xpath("//textarea[@name]"):
        name = textarea_node.get("name", "")
        state[name] = textarea_node.text or ""

    state.setdefault("__EVENTTARGET", "")
    state.setdefault("__EVENTARGUMENT", "")
    state.setdefault("__LASTFOCUS", "")
    return state


def http_get_root(session: requests.Session) -> tuple[str, dict]:
    """Seed ASP.NET state by loading the iBroker root page."""
    response = session.get(f"{IBROKER_BASE_URL}/ibroker/iBroker.aspx", timeout=30)
    response.raise_for_status()
    page_html = response.text
    if "tbxUsername" in page_html and "tbxPassword" in page_html:
        raise RuntimeError("http session is not authenticated")
    return page_html, extract_webforms_state(page_html)


def post_link_event(
    session: requests.Session, state: dict, target: str
) -> tuple[str, dict]:
    """Fire a LinkButton postback and return next ASP.NET state."""
    payload = dict(state)
    payload["__EVENTTARGET"] = target
    payload["__EVENTARGUMENT"] = ""
    response = session.post(
        f"{IBROKER_BASE_URL}/ibroker/iBroker.aspx", data=payload, timeout=30
    )
    response.raise_for_status()
    return response.text, extract_webforms_state(response.text)


def post_fetch_grid(
    session: requests.Session,
    state: dict,
    study_id: str,
    study_number: str | None = None,
) -> tuple[str, dict]:
    """Submit Query Exams request for one study_id and return next state."""
    payload = dict(state)
    payload["tbxAssignedID"] = study_id
    payload["btnFetch"] = "Query Exams"
    if study_number is not None:
        payload["tbxStudyNumber"] = study_number
    response = session.post(
        f"{IBROKER_BASE_URL}/ibroker/iBroker.aspx", data=payload, timeout=30
    )
    response.raise_for_status()
    return response.text, extract_webforms_state(response.text)


def infer_modality_from_description(description: str | None) -> str | None:
    """Infer base modality from StudyDescription when modality is missing."""
    if not description or not isinstance(description, str):
        return None
    upper_desc = description.strip().upper()
    for prefix, modality in KNOWN_MODALITY_PREFIXES:
        if upper_desc.startswith(prefix):
            return modality
    if "BREAST" in upper_desc and "MAM" in upper_desc:
        return "MG"
    if "MRI" in upper_desc:
        return "MR"
    return None


def parse_all_tables_from_page(page_html: str) -> pd.DataFrame:
    """Parse exported and available studies tables from a full iBroker page."""
    document = lxml_html.fromstring(page_html)
    frames = []

    exported_nodes = document.xpath(
        '//table[.//th[contains(text(), "Accession")] or .//td[contains(text(), "Accession")]]'
    )
    if not exported_nodes:
        exported_nodes = document.xpath(
            '//table[contains(.//tr[1]/td[4]/text(), "Accession")]'
        )

    if exported_nodes:
        root = exported_nodes[0]
        rows = root.xpath(".//tr")[1:]
        if rows and "no record" not in root.text_content().lower():
            dt_str = [str(row.xpath("normalize-space(td[1])")) for row in rows]
            desc = [str(row.xpath("normalize-space(td[2])")) for row in rows]
            status = [str(row.xpath("normalize-space(td[3])")) for row in rows]
            accession = [str(row.xpath("normalize-space(td[4])")) for row in rows]
            exported_on_str = [str(row.xpath("normalize-space(td[5])")) for row in rows]
            exported = pd.DataFrame(
                {
                    "Study DateTime": pd.to_datetime(
                        pd.Series(dt_str, dtype="string"), errors="coerce"
                    ),
                    "StudyDescription": desc,
                    "Status": status,
                    "Accession": accession,
                    "Exported On": pd.to_datetime(
                        pd.Series(exported_on_str, dtype="string"), errors="coerce"
                    ),
                }
            )
            exported["Modality"] = exported["StudyDescription"].map(
                infer_modality_from_description
            )
            frames.append(exported)

    available_nodes = document.xpath("//table[@id='TabContainer1_tabPanel1_gv1']")
    if available_nodes:
        root = available_nodes[0]
        rows = root.xpath(".//tr")[1:]
        if rows and "no record" not in root.text_content().lower():
            modality = [str(row.xpath("normalize-space(td[2])")) for row in rows]
            dt_str = [str(row.xpath("normalize-space(td[3])")) for row in rows]
            desc = [str(row.xpath("normalize-space(td[4])")) for row in rows]
            available = pd.DataFrame(
                {
                    "Modality": modality,
                    "Study DateTime": pd.to_datetime(
                        pd.Series(dt_str, dtype="string"), errors="coerce"
                    ),
                    "StudyDescription": desc,
                }
            )
            frames.append(available)

    if not frames:
        return pd.DataFrame(columns=REQUIRED_METADATA_COLUMNS[:-1])

    combined = pd.concat(frames, ignore_index=True, sort=False)
    for col in REQUIRED_METADATA_COLUMNS[:-1]:
        if col not in combined.columns:
            combined[col] = pd.NA

    combined["StudyDescription"] = combined["StudyDescription"].astype("string")
    combined["Accession"] = combined["Accession"].astype("string").replace({"": pd.NA})
    combined["Modality"] = combined["Modality"].astype("string").replace({"": pd.NA})
    combined["Status"] = combined["Status"].astype("string").replace({"": pd.NA})
    combined["Exported On"] = pd.to_datetime(combined["Exported On"], errors="coerce")
    combined["Study DateTime"] = pd.to_datetime(
        combined["Study DateTime"], errors="coerce"
    )

    missing_modality = combined["Modality"].isna()
    if missing_modality.any():
        combined.loc[missing_modality, "Modality"] = combined.loc[
            missing_modality, "StudyDescription"
        ].map(infer_modality_from_description)

    combined = combined.sort_values(["Study DateTime", "StudyDescription", "Accession"])
    combined = combined.drop_duplicates(
        subset=["Study DateTime", "StudyDescription", "Accession"], keep="first"
    )
    return combined.reset_index(drop=True)


def add_ibroker_state_columns(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized iBroker state columns from raw status/accession/export info."""
    status_text = (
        metadata_df.get("Status", pd.Series("", index=metadata_df.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    status_lower = status_text.str.lower()

    has_accession = (
        metadata_df["Accession"].notna()
        if "Accession" in metadata_df.columns
        else pd.Series(False, index=metadata_df.index)
    )
    has_exported_on = pd.to_datetime(
        metadata_df.get("Exported On", pd.Series(pd.NaT, index=metadata_df.index)),
        errors="coerce",
    ).notna()

    is_exported = (
        has_accession | has_exported_on | status_lower.isin(EXPORTED_STATUS_VALUES)
    )
    is_requested = (~is_exported) & status_lower.isin(REQUESTED_STATUS_VALUES)
    is_failed = (
        (~is_exported) & (~is_requested) & status_lower.isin(FAILED_STATUS_VALUES)
    )
    is_available = (
        (~is_exported)
        & (~is_requested)
        & (~is_failed)
        & (status_lower.eq("") | status_lower.eq("<na>"))
    )

    metadata_df["ibroker_state"] = np.select(
        [is_exported, is_requested, is_failed, is_available],
        ["exported", "requested", "failed", "available"],
        default="unknown",
    )
    metadata_df["is_exported"] = is_exported.astype(bool)
    metadata_df["is_requested"] = is_requested.astype(bool)
    return metadata_df


def _build_id_batches(
    study_ids: list[str], batch_size: int
) -> list[tuple[int, list[str]]]:
    """Split study IDs into deterministic refresh batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [
        (batch_index, study_ids[start : start + batch_size])
        for batch_index, start in enumerate(range(0, len(study_ids), batch_size))
    ]


def _batch_file_name(batch_index: int) -> str:
    """Return stable file name for one refresh batch."""
    return f"batch_{batch_index:07d}.csv"


def _extract_batch_index(batch_file: Path) -> int:
    """Extract batch index from checkpoint batch filename."""
    match = re.fullmatch(r"batch_(\d{7})\.csv", batch_file.name)
    if match is None:
        raise RuntimeError(f"unexpected checkpoint batch filename: {batch_file.name}")
    return int(match.group(1))


def _write_checkpoint_batch(df: pd.DataFrame, batch_path: Path) -> None:
    """Atomically persist one checkpoint batch dataframe."""
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".tmp",
        prefix=f".{batch_path.name}.",
        dir=batch_path.parent,
        delete=False,
        encoding="utf-8",
        newline="",
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        df.to_csv(tmp_file, index=False)
    os.replace(tmp_path, batch_path)


def _query_batches_with_single_session(
    batch_specs: list[tuple[int, list[str]]],
    username: str,
    password: str,
    study_number: str | None,
    checkpoint_batches_dir: Path,
    on_batch_complete: Callable[[int, int], None] | None = None,
) -> None:
    """Query a set of batches in one iBroker session and checkpoint each batch."""
    driver = None
    try:
        driver = make_driver()
        login(driver, username, password)
        session = bootstrap_http_session_from_driver(driver)
        _, state = http_get_root(session)
        _, state = post_link_event(session, state, "lbAll")

        for batch_index, study_ids in batch_specs:
            batch_frames = []
            batch_error_count = 0
            for study_id in study_ids:
                try:
                    page_html, state = post_fetch_grid(
                        session,
                        state,
                        str(study_id),
                        study_number=study_number,
                    )
                    frame = parse_all_tables_from_page(page_html)
                    if frame.empty:
                        frame = pd.DataFrame(columns=REQUIRED_METADATA_COLUMNS)
                    frame["study_id"] = str(study_id)
                    batch_frames.append(
                        frame.reindex(columns=REQUIRED_METADATA_COLUMNS)
                    )
                except Exception as exc:
                    batch_error_count += 1
                    batch_frames.append(
                        pd.DataFrame(
                            [
                                {
                                    "Modality": pd.NA,
                                    "Study DateTime": pd.NaT,
                                    "StudyDescription": pd.NA,
                                    "Status": pd.NA,
                                    "Accession": pd.NA,
                                    "Exported On": pd.NaT,
                                    "study_id": str(study_id),
                                    "scrape_error": str(exc),
                                }
                            ]
                        )
                    )
            if batch_frames:
                batch_df = pd.concat(batch_frames, ignore_index=True, sort=False)
            else:
                batch_df = pd.DataFrame(
                    columns=REQUIRED_METADATA_COLUMNS + ["scrape_error"]
                )
            if "scrape_error" not in batch_df.columns:
                batch_df["scrape_error"] = pd.NA
            batch_path = checkpoint_batches_dir / _batch_file_name(batch_index)
            _write_checkpoint_batch(batch_df, batch_path)
            if on_batch_complete is not None:
                on_batch_complete(len(study_ids), batch_error_count)
    finally:
        if driver is not None:
            driver.quit()


def _count_errors_in_checkpoint_files(batch_files: list[Path]) -> int:
    """Count non-empty scrape_error values in checkpoint batch files."""
    total_errors = 0
    for batch_file in batch_files:
        df = pd.read_csv(batch_file)
        if "scrape_error" not in df.columns:
            continue
        error_series = df["scrape_error"]
        has_error = error_series.notna() & error_series.astype(str).str.strip().ne("")
        total_errors += int(has_error.sum())
    return total_errors


def _build_combined_from_checkpoint_files(batch_files: list[Path]) -> pd.DataFrame:
    """Load and combine checkpoint batch files into a normalized metadata frame."""
    if not batch_files:
        combined = pd.DataFrame(columns=REQUIRED_METADATA_COLUMNS + ["scrape_error"])
    else:
        batch_frames = [pd.read_csv(path) for path in batch_files]
        combined = pd.concat(batch_frames, ignore_index=True, sort=False)

    if "scrape_error" not in combined.columns:
        combined["scrape_error"] = pd.NA

    combined["Study DateTime"] = pd.to_datetime(
        combined["Study DateTime"], errors="coerce"
    )
    combined["Exported On"] = pd.to_datetime(combined["Exported On"], errors="coerce")
    combined["study_id"] = combined["study_id"].astype("string")
    combined["Accession"] = combined["Accession"].astype("string")
    combined["Status"] = combined["Status"].astype("string")
    combined["StudyDescription"] = combined["StudyDescription"].astype("string")
    combined["Modality"] = combined["Modality"].astype("string")

    combined = combined.sort_values(
        ["study_id", "Study DateTime", "StudyDescription", "Accession"],
        na_position="last",
    ).drop_duplicates(
        subset=["study_id", "Study DateTime", "StudyDescription", "Accession"],
        keep="first",
    )
    return combined


def _atomic_write_dataframe_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Write dataframe to CSV atomically."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".tmp",
        prefix=f".{output_path.name}.",
        dir=output_path.parent,
        delete=False,
        encoding="utf-8",
        newline="",
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        df.to_csv(tmp_file, index=False)
    os.replace(tmp_path, output_path)


def refresh_metadata_snapshot(
    study_ids: list[str],
    output_file: str | Path,
    *,
    study_number: str | None = None,
    max_workers: int = 1,
    refresh_mode: str = "fresh",
    checkpoint_batch_size: int = 500,
    checkpoint_dir: str | Path | None = None,
    show_progress: bool = True,
    max_new_batches: int | None = None,
    username: str | None = None,
    password: str | None = None,
) -> tuple[pd.DataFrame, dict[str, int | bool]]:
    """Refresh metadata with resumable checkpoints and overwrite/advance CSV snapshot."""
    resolved_username = username or os.getenv("IBROKER_USERNAME")
    resolved_password = password or os.getenv("IBROKER_PASSWORD")
    if not resolved_username or not resolved_password:
        raise RuntimeError("IBROKER_USERNAME and IBROKER_PASSWORD must be set")
    if refresh_mode not in {"fresh", "resume"}:
        raise RuntimeError("refresh_mode must be one of: fresh, resume")
    if max_workers <= 0:
        raise RuntimeError("max_workers must be > 0")
    if max_new_batches is not None and max_new_batches <= 0:
        raise RuntimeError("max_new_batches must be > 0 when provided")

    normalized_ids = [str(int(sid)) for sid in study_ids]
    all_batches = _build_id_batches(normalized_ids, checkpoint_batch_size)
    batch_size_map = {
        batch_index: len(batch_ids) for batch_index, batch_ids in all_batches
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir is None:
        checkpoint_root = output_path.parent / f".{output_path.name}.refresh"
    else:
        checkpoint_root = Path(checkpoint_dir)
    checkpoint_batches_dir = checkpoint_root / "batches"

    if refresh_mode == "fresh":
        if checkpoint_root.exists():
            shutil.rmtree(checkpoint_root)
        checkpoint_batches_dir.mkdir(parents=True, exist_ok=True)
    else:
        if not checkpoint_batches_dir.exists():
            raise RuntimeError(
                f"resume requested but checkpoint directory does not exist: {checkpoint_batches_dir}"
            )

    existing_batch_files = sorted(checkpoint_batches_dir.glob("batch_*.csv"))
    completed_batch_indices = {
        _extract_batch_index(path) for path in existing_batch_files
    }
    completed_study_ids = sum(batch_size_map[idx] for idx in completed_batch_indices)

    invalid_indices = [
        idx for idx in completed_batch_indices if idx >= len(all_batches)
    ]
    if invalid_indices:
        raise RuntimeError(
            "checkpoint contains batch files outside current study ID set; use --refresh-mode fresh"
        )

    pending_batches = [
        batch_spec
        for batch_spec in all_batches
        if batch_spec[0] not in completed_batch_indices
    ]
    batches_to_process = pending_batches
    if max_new_batches is not None:
        batches_to_process = pending_batches[:max_new_batches]

    if batches_to_process:
        worker_count = min(max_workers, len(batches_to_process))
        worker_batch_specs = [
            batches_to_process[i::worker_count] for i in range(worker_count)
        ]
        initial_error_count = _count_errors_in_checkpoint_files(existing_batch_files)
        total_study_ids = len(normalized_ids)
        progress_counters = {
            "processed_study_ids": completed_study_ids,
            "error_count": initial_error_count,
            "completed_batches": len(completed_batch_indices),
        }
        progress_bar = (
            tqdm(
                total=len(all_batches),
                initial=len(completed_batch_indices),
                desc="metadata refresh batches",
                unit="batch",
                mininterval=1.0,
            )
            if show_progress
            else None
        )
        progress_lock = threading.Lock()

        def _on_batch_complete(study_count: int, batch_error_count: int) -> None:
            if progress_bar is None:
                return
            with progress_lock:
                progress_counters["processed_study_ids"] += study_count
                progress_counters["error_count"] += batch_error_count
                progress_counters["completed_batches"] += 1
                progress_bar.update(1)
                remaining_ids = (
                    total_study_ids - progress_counters["processed_study_ids"]
                )
                progress_bar.set_postfix(
                    {
                        "remaining_ids": remaining_ids,
                        "errors": progress_counters["error_count"],
                    },
                    refresh=False,
                )
                if progress_counters["completed_batches"] % 5 == 0:
                    tqdm.write(
                        "refresh heartbeat: "
                        f"{progress_counters['processed_study_ids']:,}/{total_study_ids:,} study IDs done, "
                        f"{remaining_ids:,} remaining, "
                        f"scrape errors so far={progress_counters['error_count']:,}"
                    )

        try:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        _query_batches_with_single_session,
                        worker_specs,
                        resolved_username,
                        resolved_password,
                        study_number,
                        checkpoint_batches_dir,
                        _on_batch_complete,
                    )
                    for worker_specs in worker_batch_specs
                ]
                for future in futures:
                    future.result()
        finally:
            if progress_bar is not None:
                progress_bar.close()
    elif show_progress:
        tqdm(
            total=len(all_batches),
            initial=len(all_batches),
            desc="metadata refresh batches",
            unit="batch",
            mininterval=1.0,
        ).close()

    final_batch_files = sorted(checkpoint_batches_dir.glob("batch_*.csv"))
    final_indices = {_extract_batch_index(path) for path in final_batch_files}
    expected_indices = set(range(len(all_batches)))
    missing = sorted(expected_indices - final_indices)
    is_complete = len(missing) == 0

    if not is_complete and max_new_batches is None:
        raise RuntimeError(
            f"refresh incomplete; missing {len(missing)} checkpoint batch files"
        )

    combined = _build_combined_from_checkpoint_files(final_batch_files)
    _atomic_write_dataframe_csv(combined, output_path)
    if is_complete:
        shutil.rmtree(checkpoint_root)
    summary = {
        "is_complete": is_complete,
        "completed_batches": len(final_indices),
        "total_batches": len(all_batches),
        "remaining_batches": len(missing),
    }
    return combined.reset_index(drop=True), summary
