from playwright.sync_api import sync_playwright

class BrowserSession:

    def __init__(self, headless=True):
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

    def start(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=self.headless
        )
        self.context = self.browser.new_context()
        self.page = self.context.new_page()


    def goto(self, url):
        self.page.goto(url)


    def fill(self, selector, value):
        self.page.locator(selector).fill(value)


    def click(self, selector):
        self.page.locator(selector).click()


    def locator(self, selector, **kwargs):
        return self.page.locator(selector, **kwargs)


    def screenshot(self, path):
        print(f"Saving screenshot: {path}")
        self.page.screenshot(path=path)

    def close(self):
        try:
            if self.context:
                self.context.close()
        except Exception as exc:
            print(f"Warning: failed to close context cleanly: {exc}")
        try:
            if self.playwright:
                self.playwright.stop()
        except Exception as exc:
            print(f"Warning: failed to stop playwright cleanly: {exc}")

    def wait(self, seconds):
        self.page.wait_for_timeout(seconds * 1000)

    def content(self):
        return self.page.content()

    def current_url(self):
        return self.page.url

    def wait_for_selector(
        self,
        selector,
        timeout=10000
    ):
        self.page.wait_for_selector(
            selector,
            timeout=timeout
        )

    def tab_and_enter(self):
        self.page.keyboard.press("Tab")
        self.page.keyboard.press("Enter")

    def enter(self):
        self.page.keyboard.press("Enter")

    def switch_to_new_page(self, timeout=10000):
        """Switch focus to the most recently opened tab."""
        self.page.wait_for_timeout(2000)  # brief wait for the new tab to open
        pages = self.context.pages
        if len(pages) > 1:
            self.page = pages[-1]  # switch to the last opened tab
            self.page.wait_for_load_state("load")
            return True
        return False
    
    def switch_to_first_page(self):
        """Switch focus back to the original/first tab."""
        pages = self.context.pages
        if pages:
            self.page = pages[0]
            self.page.wait_for_load_state("load")

    def debug_pages(self):
        """Log all currently open tabs and their URLs."""
        pages = self.context.pages
        for i, p in enumerate(pages):
            print(f"Tab {i}: {p.url}")
        return pages