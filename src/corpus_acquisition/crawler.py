from src.schemas.page import Page


class ArchiveCrawler:
    def __init__(self, browser, config, credentials):
        self.browser = browser
        self.config = config
        self.credentials = credentials

    def open_archive(self, date):
        url = f"{self.config['archive_url']}{date:%Y%m%d}"
        self.browser.goto(url)
    
    def download_page(self, page_number):
        output_dir = "../data/raw/images/el_comercio"
        path = f"{output_dir}/page_1.jpg" 
        browser.page.request.get(src).body()