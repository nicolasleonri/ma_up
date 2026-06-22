import argparse
import logging
from datetime import datetime
from pathlib import Path

from src.corpus_acquisition.browser import BrowserSession
from src.corpus_acquisition.crawler import ArchiveCrawler
from src.corpus_acquisition.crawl_registry import NEWSPAPERS, PORTAL_LOGIN_URL
from src.utils.config_loader import load_yaml
from datetime import datetime
from src.corpus_acquisition.downloader import CorpusDownloader

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Acquire a newspaper corpus.")
    parser.add_argument(
        "--newspaper",
        required=True,
        choices=sorted(NEWSPAPERS.keys()),
        help="Newspaper key from crawl_registry.NEWSPAPERS.",
    )
    parser.add_argument(
        "--start-date",
        required=True,
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        help="First date to crawl, format YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        help="Last date to crawl (inclusive), format YYYY-MM-DD. Defaults to --start-date.",
    )
    parser.add_argument(
        "--credentials",
        default="config/corpus_acquisition/credentials.yaml",
        help="Path to the portal credentials YAML file.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing crawl logs and re-crawl every date, even if previously successful.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run the browser with a visible UI (useful for debugging).",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/corpus_acquisition",
        help="Directory where the run's log file is written.",
    )
    return parser.parse_args()

def login(browser: BrowserSession, credentials: dict, newspaper_config: dict) -> None:
    """Log in to the newspaper portal and open the requested newspaper section."""
    browser.goto(PORTAL_LOGIN_URL)
    browser.wait_for_selector("button[class='sc-dkzDqf xFTNr']")  # "Iniciar Sesión"
    browser.click("button[class='sc-dkzDqf xFTNr']")
    browser.wait(5)

    login_frame = browser.page.frames[1]
    login_frame.locator("input[name='email']").fill(credentials["username"])
    login_frame.locator("input[type='password']").fill(credentials["password"])
    login_frame.locator("button[type='submit']").click()
    browser.wait_for_selector("button[class='sc-dkzDqf kwFoJp']", timeout=30000)  # "Cerrar Sesión"
    browser.wait(5)

    browser.locator("div[class='sc-iIPllB bUkgIR']", has_text=newspaper_config.get("selector_text", "")).first.click()
    browser.wait(5)

    
def main():
    args = parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{args.newspaper}_{datetime.now():%Y%m%d_%H%M%S}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    newspaper_config = NEWSPAPERS[args.newspaper]

    browser = BrowserSession(headless=not args.headed)
    credentials = load_yaml(args.credentials)
    
    try:
        browser.start()
        logger.info("Logging in to portal...")

        login(browser, credentials, newspaper_config)

        logger.info("Login successful!")

        downloader = CorpusDownloader(browser=browser, credentials=credentials)

        pages = downloader.download_range(
            newspaper=args.newspaper,
            config=newspaper_config,
            start_date=args.start_date,
            end_date=args.end_date,
            resume=not args.no_resume,
        )

        logger.info("Finished. Downloaded %d page(s) across the requested date range.", len(pages))

        # # Then, crawls
        # crawler = ArchiveCrawler(
        #     browser=browser,
        #     config=NEWSPAPERS["el_comercio"], # TODO: Make newspaper configurable through command line arguments.
        #     credentials=credentials
        # )

        # date = datetime(2026, 6, 3) # TODO: Loop through years
        # crawler.open_archive(date)
        # browser.wait(5)

        # print("Collecting URLs...") #TODO: Replace print() with logging module
        # # TODO: Skip downloading files that already exist.

        # seen = set()
        # step = 0

        # while True:
        #     step += 1

        #     current = crawler.get_urls()
        #     old_size = len(seen)
        #     seen.update(current)
        #     new_urls = len(seen) - old_size

        #     if new_urls != 0:
        #         print(f"Collected {len(seen)} / 20 pages")

        #     if len(seen) >= 20: # TODO: Make this dynamic through config
        #         break

        #     browser.locator(
        #         "div[class='readingnav rn-right']"
        #     ).click()

        #     browser.wait(2)

        # pages = {}
        # for url in seen:
        #     page = int(
        #         parse_qs(
        #             urlparse(url).query
        #         )["page"][0]
        #     )
        #     pages[page] = url
        
        # for page in sorted(pages):
        #     url = crawler.build_page_url(
        #         pages[page],
        #         scale=200
        #     )

        #     print(f"Downloading page {page} from {url}...")

            # try:
            #     # crawler.download_page(url, page, path=f"data/raw/images/el_comercio/2026/03/03/name_date_{page}.jpg") # TODO: Make path dynamic through config
            #     # TODO: Store pages as Page objects
            #     # TODO: Save crawl metadata
            # except Exception as e:
            #     print(f"An error occurred while downloading page {page}: {e}")
    except Exception as exc:
        logger.error("An error occurred: %s", exc)
        # Crawl logs written by CorpusDownloader allow a re-run of this same
        # command to resume from the last successfully completed date.
    finally:
        browser.close()


if __name__ == "__main__":

    main()