from src.schemas.page import Page
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

class ArchiveCrawler:
    def __init__(self, browser, config, credentials):
        self.browser = browser
        self.config = config
        self.credentials = credentials

    def open_archive(self, date):
        url = f"{self.config['archive_url']}{date:%Y%m%d}"
        self.browser.goto(url)

    def get_urls(self):
        images = self.browser.page.locator("img")
        urls = set()

        for i in range(images.count()):
            src = images.nth(i).get_attribute("src")
            if src and "prcdn.co/img" in src:
                urls.add(src)

        return sorted(urls)

    def download_page(self, url, page_number, path):
        output_file = Path(path)
        output_dir = output_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading -> {output_file}")

        response = self.browser.page.request.get(url)

        if not response.ok:
            raise Exception(
                f"Failed to download page {page_number}: "
                f"{response.status}"
            )

        with open(output_file, "wb") as f:
            f.write(response.body())

        return output_file

    def build_page_url(self, base_url, scale=200):

        parsed = urlparse(base_url)

        qs = parse_qs(parsed.query)

        qs["scale"] = [str(scale)]

        new_query = urlencode(qs, doseq=True)

        return urlunparse(
            parsed._replace(query=new_query)
        )