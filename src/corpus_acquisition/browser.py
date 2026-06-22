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
        self.context.close()
        self.browser.close()
        self.playwright.stop()

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