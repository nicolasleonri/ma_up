"""High-level orchestrator for the corpus-acquisition crawl-and-download loop.

``CorpusDownloader`` ties together ``BrowserSession`` (login/navigation),
``ArchiveCrawler`` (per-date archive scraping), and ``metadata_extractor``
(persisting ``Page`` records + crawl logs) into a single entry point that
workflows (e.g. ``src/workflows/acquire_corpus.py``) can call with just a
newspaper name and a date range.

It addresses the TODOs left in ``acquire_corpus.py``:
- dynamic, config-driven output paths and expected page counts
- ``Page`` objects + persisted metadata instead of bare file writes
- skip-downloading pages that already exist on disk
- checkpointing via crawl logs, so interrupted runs can resume
- logging instead of bare ``print`` calls
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.corpus_acquisition import metadata_extractor as meta
from src.corpus_acquisition.crawler import ArchiveCrawler
from src.schemas.page import Page

logger = logging.getLogger(__name__)


class CorpusDownloader:
    """Crawl a newspaper's digital archive over a date range and persist results.

    Parameters
    ----------
    browser:
        An already-started ``BrowserSession`` (login/navigation to the
        newspaper's section is assumed to have happened already, since
        that flow is newspaper-portal-specific).
    crawler_cls:
        The crawler class to instantiate per newspaper. Defaults to
        ``ArchiveCrawler``; injectable for testing.
    output_root:
        Root directory under which page images are stored, as
        ``{output_root}/{newspaper}/{YYYY}/{MM}/{DD}/page_{n}.jpg``.
    metadata_path:
        Parquet file where page metadata is appended.
    log_dir:
        Directory where per-(newspaper, date) crawl logs are written.
    """

    def __init__(
        self,
        browser,
        crawler_cls=ArchiveCrawler,
        credentials: dict | None = None,
        output_root: str | Path = "data/raw/images",
        metadata_path: str | Path = "data/raw/metadata/raw_metadata.parquet",
        log_dir: str | Path = "data/raw/crawl_logs",
    ):
        self.browser = browser
        self.crawler_cls = crawler_cls
        self.credentials = credentials
        self.output_root = Path(output_root)
        self.metadata_path = Path(metadata_path)
        self.log_dir = Path(log_dir)

    def download_range(
        self,
        newspaper: str,
        config: dict,
        start_date: datetime,
        end_date: datetime | None = None,
        scale: int = 200,
        skip_existing: bool = True,
        resume: bool = True,
    ) -> list[Page]:
        """Crawl and download every date in ``[start_date, end_date]``.

        If ``end_date`` is omitted, only ``start_date`` is processed.
        Returns the full list of ``Page`` objects collected across all dates.
        """
        end_date = end_date or start_date
        all_pages: list[Page] = []

        current = start_date
        while current <= end_date:
            if resume and meta.is_already_crawled(newspaper, current, self.log_dir):
                logger.info(
                    "Skipping %s on %s: already crawled successfully.",
                    newspaper,
                    current.strftime("%Y-%m-%d"),
                )
            else:
                try:
                    pages = self.download_date(
                        newspaper=newspaper,
                        config=config,
                        date=current,
                        scale=scale,
                        skip_existing=skip_existing,
                    )
                    all_pages.extend(pages)
                except Exception as exc:
                    logger.error(
                        "Skipping %s on %s due to unrecoverable error: %s",
                        newspaper,
                        current.strftime("%Y-%m-%d"),
                        exc,
                    )
                    meta.save_crawl_log(
                        newspaper,
                        current,
                        status="failed",
                        error=str(exc),
                        log_dir=self.log_dir,
                    )

            current += timedelta(days=1)

        return all_pages

    def download_date(
        self,
        newspaper: str,
        config: dict,
        date: datetime,
        scale: int = 200,
        skip_existing: bool = True,
    ) -> list[Page]:
        """Crawl and download every page of a single newspaper issue.

        Persists metadata and a crawl log on completion (success or failure)
        so that ``download_range`` can resume later if interrupted.
        """
        crawler = self.crawler_cls(
            browser=self.browser,
            config=config,
            credentials=self.credentials,
        )
        expected_pages = int(config.get("length", 0)) or None
        date_str = date.strftime("%Y-%m-%d")

        expected_pages = int(config.get("length", 0)) or None
        date_str = date.strftime("%Y-%m-%d")

        if skip_existing and self._date_already_downloaded(newspaper, date, expected_pages):
            logger.info("All pages for %s on %s already exist on disk, skipping navigation.", newspaper, date_str)
            return []

        logger.info("Opening archive for %s on %s", newspaper, date_str)
        try:
            crawler.open_archive(date)
        except Exception as exc:
            screenshot_dir = Path("logs/screenshots")
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            screenshot_path = screenshot_dir / f"{newspaper}_{date:%Y-%m-%d}_{datetime.now():%H%M%S}.png"
            try:
                self.browser.page.screenshot(path=str(screenshot_path))
                logger.error("Navigation failed for %s on %s: %s (screenshot: %s)", newspaper, date_str, exc, screenshot_path)
            except Exception as shot_exc:
                logger.error("Navigation failed for %s on %s: %s (screenshot also failed: %s)", newspaper, date_str, exc, shot_exc)
            raise

        self.browser.wait(5)

        try:
            urls_by_page = self._collect_page_urls(crawler, expected_pages)
        except Exception as exc:
            logger.error("Failed to collect URLs for %s on %s: %s", newspaper, date_str, exc)
            meta.save_crawl_log(
                newspaper, date, status="failed", pages_expected=expected_pages, error=str(exc),
                log_dir=self.log_dir,
            )
            return []

        already_done = (
            meta.load_existing_page_numbers(newspaper, date, self.metadata_path)
            if skip_existing
            else set()
        )

        pages: list[Page] = []
        failures = 0

        for page_number in sorted(urls_by_page):
            output_path = self._build_output_path(newspaper, date, page_number)

            if skip_existing and (page_number in already_done or output_path.exists()):
                logger.info("Page %d already downloaded, skipping.", page_number)
                continue

            page_url = crawler.build_page_url(urls_by_page[page_number], scale=scale)
            logger.info("Downloading page %d from %s", page_number, page_url)

            try:
                crawler.download_page(page_url, page_number, path=str(output_path))
            except Exception as exc:
                logger.error("Failed to download page %d: %s", page_number, exc)
                failures += 1
                continue

            pages.append(
                meta.build_page(
                    newspaper=newspaper,
                    date=date,
                    edition=config.get("edition", "default"),
                    page_number=page_number,
                    page_url=page_url,
                    image_url=urls_by_page[page_number],
                    image_path=output_path,
                )
            )

        if pages:
            meta.save_pages_metadata(pages, self.metadata_path)

        status = "success" if failures == 0 and pages else ("partial" if pages else "failed")
        meta.save_crawl_log(
            newspaper,
            date,
            status=status,
            pages_downloaded=len(pages),
            pages_expected=expected_pages,
            error=None if status == "success" else f"{failures} page(s) failed to download",
            log_dir=self.log_dir,
        )

        return pages

    def _collect_page_urls(self, crawler: ArchiveCrawler, expected_pages: int | None) -> dict[int, str]:
        """Page through the archive viewer, collecting image URLs per page number.

        Mirrors the polling loop that used to live inline in
        ``acquire_corpus.py``: repeatedly read visible image URLs, advance
        the reader, and stop once the expected page count is reached (or
        once the reader stops producing new URLs, to avoid spinning forever
        when ``expected_pages`` is unknown or wrong).
        """
        seen: set[str] = set()
        stall_rounds = 0
        max_stall_rounds = 5

        while True:
            current = crawler.get_urls()
            before = len(seen)
            seen.update(current)

            if len(seen) > before:
                stall_rounds = 0
                logger.info("Collected %d%s pages", len(seen), f" / {expected_pages}" if expected_pages else "")
            else:
                stall_rounds += 1

            if expected_pages and len(seen) >= expected_pages:
                break
            if stall_rounds >= max_stall_rounds:
                logger.warning(
                    "No new pages found after %d attempts; stopping with %d page(s).",
                    stall_rounds,
                    len(seen),
                )
                break

            self.browser.locator("div[class='readingnav rn-right']").click()
            self.browser.wait(2)

        pages_by_number: dict[int, str] = {}
        for url in seen:
            page_number = int(parse_qs(urlparse(url).query)["page"][0])
            pages_by_number[page_number] = url

        return pages_by_number

    def _build_output_path(self, newspaper: str, date: datetime, page_number: int) -> Path:
        """Build the on-disk path for a page image: root/newspaper/Y/M/D/newspaper_date_page.jpg"""
        return (
            self.output_root
            / newspaper
            / f"{date:%Y}"
            / f"{date:%m}"
            / f"{date:%d}"
            / f"{newspaper}_{date:%Y-%m-%d}_{page_number}.jpg"
        )
    
    def _date_already_downloaded(self, newspaper: str, date: datetime, expected_pages: int | None) -> bool:
        """Check the output folder for an already-complete set of pages for this date.

        Used as a cheap pre-check before even opening the archive page, so a
        rerun doesn't navigate at all for dates that are already fully on disk.
        """
        date_dir = self.output_root / newspaper / f"{date:%Y}" / f"{date:%m}" / f"{date:%d}"
        if not date_dir.exists():
            return False

        existing = list(date_dir.glob(f"{newspaper}_{date:%Y-%m-%d}_page*.jpg"))
        if not existing:
            return False

        if expected_pages:
            return len(existing) >= expected_pages

        # No expected count to compare against; treat any existing files as "already done".
        return True