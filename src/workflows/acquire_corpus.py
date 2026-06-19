from src.corpus_acquisition.browser import BrowserSession
from src.corpus_acquisition.crawler import ArchiveCrawler
from src.corpus_acquisition.crawl_registry import NEWSPAPERS
from src.utils.config_loader import load_yaml
from datetime import datetime
from urllib.parse import parse_qs, urlparse

def main():

    browser = BrowserSession(headless=True)
    credentials = load_yaml("config/corpus_acquisition/credentials.yaml")

    try:
        # First, logs in:
        browser.start()

        browser.goto("https://web.peruquiosco.pe/") # TODO: Move hardcoded URLs (https://web.peruquiosco.pe/) to crawl_registry.py
        browser.wait_for_selector("button[class='sc-dkzDqf xFTNr']") # Iniciar Sesión Button
        browser.click("button[class='sc-dkzDqf xFTNr']")
        browser.wait(5)

        login_frame = browser.page.frames[1]
        login_frame.locator( "input[name='email']" ).fill(credentials["username"]) 
        login_frame.locator( "input[type='password']" ).fill(credentials["password"]) 
        login_frame.locator("button[type='submit']").click()
        browser.wait_for_selector("button[class='sc-dkzDqf kwFoJp']", timeout=30000) # Cerrar Sesión Button
        browser.wait(5)

        browser.locator("div[class='sc-iIPllB bUkgIR']", has_text="El Comercio").first.click()
        browser.wait(5)

        # Then, crawls
        crawler = ArchiveCrawler(
            browser=browser,
            config=NEWSPAPERS["el_comercio"], # TODO: Make newspaper configurable through command line arguments.
            credentials=credentials
        )

        date = datetime(2026, 6, 3) # TODO: Loop through years
        crawler.open_archive(date)
        browser.wait(5)

        print("Collecting URLs...") #TODO: Replace print() with logging module
        # TODO: Skip downloading files that already exist.

        seen = set()
        step = 0

        while True:
            step += 1

            current = crawler.get_urls()
            old_size = len(seen)
            seen.update(current)
            new_urls = len(seen) - old_size

            if new_urls != 0:
                print(f"Collected {len(seen)} / 20 pages")

            if len(seen) >= 20: # TODO: Make this dynamic through config
                break

            browser.locator(
                "div[class='readingnav rn-right']"
            ).click()

            browser.wait(2)

        pages = {}
        for url in seen:
            page = int(
                parse_qs(
                    urlparse(url).query
                )["page"][0]
            )
            pages[page] = url
        
        for page in sorted(pages):
            url = crawler.build_page_url(
                pages[page],
                scale=200
            )

            print(f"Downloading page {page} from {url}...")

            try:
                crawler.download_page(url, page, path=f"data/raw/images/el_comercio/2026/03/03/name_date_{page}.jpg") # TODO: Make path dynamic through config
                # TODO: Store pages as Page objects
                # TODO: Save crawl metadata
            except Exception as e:
                print(f"An error occurred while downloading page {page}: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")
        # TODO: Add checkpointing so interrupted crawls can resume
    finally:
        browser.close()


if __name__ == "__main__":

    main()