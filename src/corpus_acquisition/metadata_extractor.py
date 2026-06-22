"""Metadata extraction and persistence for the corpus acquisition stage.

This module turns raw crawl results (a newspaper, a date, and a set of
page URLs/downloaded files) into structured ``Page`` records, and persists
both:

- per-page metadata (``data/raw/metadata/raw_metadata.parquet``), used
  downstream by ``corpus_construction``.
- per-crawl logs (``data/raw/crawl_logs/<newspaper>_<date>.json``), used for
  auditing and for checkpointing/resuming interrupted crawls.

No new heavy dependency is required beyond ``pandas`` + ``pyarrow`` for the
parquet metadata store; everything else relies on the standard library.
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from src.schemas.page import Page

logger = logging.getLogger(__name__)


def build_page(
    newspaper: str,
    date,
    edition: str,
    page_number: int,
    page_url: str,
    image_url: str | None = None,
    image_path: str | None = None,
) -> Page:
    """Build a ``Page`` record from raw crawl values.

    ``date`` may be a ``datetime``/``date`` object or an already-formatted
    string; it is normalized to ``YYYY-MM-DD``.
    """
    if isinstance(date, datetime):
        date_str = date.strftime("%Y-%m-%d")
    elif hasattr(date, "isoformat"):
        date_str = date.isoformat()
    else:
        date_str = str(date)

    return Page(
        newspaper=newspaper,
        date=date_str,
        edition=edition,
        page_number=page_number,
        page_url=page_url,
        image_url=image_url,
        image_path=str(image_path) if image_path else None,
    )


def pages_to_records(pages: list[Page]) -> list[dict]:
    """Convert a list of ``Page`` dataclasses into plain dicts."""
    return [asdict(page) for page in pages]


def save_pages_metadata(
    pages: list[Page],
    path: str | Path = "data/raw/metadata/raw_metadata.parquet",
) -> Path:
    """Append page metadata to the shared parquet metadata store.

    Existing rows for the same (newspaper, date, page_number) are replaced
    rather than duplicated, so this function is safe to call repeatedly
    (e.g. once per crawled date, or on resumed/checkpointed runs).
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Saving metadata requires 'pandas' and 'pyarrow'. "
            "Install them with: pip install pandas pyarrow"
        ) from exc

    if not pages:
        logger.info("No pages to save; skipping metadata write.")
        return Path(path)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(pages_to_records(pages))
    key_cols = ["newspaper", "date", "page_number"]

    if output_path.exists():
        existing_df = pd.read_parquet(output_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
    else:
        combined = new_df

    combined = combined.sort_values(key_cols).reset_index(drop=True)
    combined.to_parquet(output_path, index=False)

    logger.info(
        "Saved metadata for %d page(s) -> %s (total rows: %d)",
        len(new_df),
        output_path,
        len(combined),
    )
    return output_path


def load_existing_page_numbers(
    newspaper: str,
    date,
    path: str | Path = "data/raw/metadata/raw_metadata.parquet",
) -> set[int]:
    """Return the set of page numbers already recorded for a given crawl.

    Used to support skipping already-downloaded pages on resumed runs.
    Returns an empty set if the metadata store doesn't exist yet or
    pandas/pyarrow aren't installed.
    """
    try:
        import pandas as pd
    except ImportError:
        return set()

    output_path = Path(path)
    if not output_path.exists():
        return set()

    date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)

    df = pd.read_parquet(output_path)
    mask = (df["newspaper"] == newspaper) & (df["date"] == date_str)
    return set(df.loc[mask, "page_number"].tolist())


def save_crawl_log(
    newspaper: str,
    date,
    status: str,
    pages_downloaded: int = 0,
    pages_expected: int | None = None,
    error: str | None = None,
    log_dir: str | Path = "data/raw/crawl_logs",
) -> Path:
    """Write a JSON log entry summarizing a single newspaper/date crawl.

    These logs back checkpointing: a workflow can inspect this directory
    before starting a crawl to see which (newspaper, date) pairs already
    completed successfully, and resume from where it left off.
    """
    date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_path = log_dir_path / f"{newspaper}_{date_str}.json"

    entry = {
        "newspaper": newspaper,
        "date": date_str,
        "status": status,
        "pages_downloaded": pages_downloaded,
        "pages_expected": pages_expected,
        "error": error,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)

    logger.info("Saved crawl log -> %s (status=%s)", log_path, status)
    return log_path


def is_already_crawled(
    newspaper: str,
    date,
    log_dir: str | Path = "data/raw/crawl_logs",
) -> bool:
    """Check the crawl log to see if this (newspaper, date) already succeeded.

    Used by the downloader to skip dates that were already fully crawled,
    enabling resumable runs.
    """
    date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
    log_path = Path(log_dir) / f"{newspaper}_{date_str}.json"

    if not log_path.exists():
        return False

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            entry = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    return entry.get("status") == "success"