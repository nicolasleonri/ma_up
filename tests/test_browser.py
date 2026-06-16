from src.corpus_acquisition.browser import BrowserSession


browser = BrowserSession(headless=True)

browser.start()

browser.goto("https://google.com")

print(browser.title())

browser.screenshot("google.png")

browser.close()