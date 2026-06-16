from src.corpus_acquisition.browser import BrowserSession
from src.corpus_acquisition.crawler import ArchiveCrawler
from src.corpus_acquisition.crawl_registry import NEWSPAPERS
from src.utils.config_loader import load_yaml
from datetime import datetime
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

def build_page_url(base_url, page, scale=200):
    parsed = urlparse(base_url)
    qs = parse_qs(parsed.query)
    qs["page"] = [str(page)]
    qs["scale"] = [str(scale)]
    new_query = urlencode(qs, doseq=True)
    return urlunparse(parsed._replace(query=new_query))

def main():

    browser = BrowserSession(headless=True)
    credentials = load_yaml("config/corpus_acquisition/credentials.yaml")

    try:
        # First, logs in:
        browser.start()

        browser.goto("https://web.peruquiosco.pe/")
        browser.wait_for_selector("button[class='sc-dkzDqf xFTNr']") # Iniciar Sesión Button
        browser.click("button[class='sc-dkzDqf xFTNr']")
        browser.wait(5)
        browser.screenshot("logsin.png")

        login_frame = browser.page.frames[1]
        login_frame.locator( "input[name='email']" ).fill(credentials["username"]) 
        login_frame.locator( "input[type='password']" ).fill(credentials["password"]) 
        login_frame.locator("button[type='submit']").click()
        browser.wait_for_selector("button[class='sc-dkzDqf kwFoJp']", timeout=30000) # Cerrar Sesión Button
        browser.wait(5)
        browser.screenshot("loggedin.png")

        browser.locator("div", has_text="El Comercio").click()
        browser.wait(5)
        # browser.screenshot("initialstate.png")

        # Then, crawls
        crawler = ArchiveCrawler(
            browser=browser,
            config=NEWSPAPERS["el_comercio"], #TODO: Make this dynamic through command line args
            credentials=credentials
        )

        date = datetime(2026, 6, 3) # TODO: Loop through years
        crawler.open_archive(date)
        browser.wait(5)
        browser.screenshot("gotthere.png")

        browser.locator("div[class='readingnav rn-right']").click()
        browser.wait(5)

        browser.screenshot("finalstate.png")

        # images = browser.page.locator("img")
        # base_file = None

        # for i in range(images.count()):
        #     src = images.nth(i).get_attribute("src")
        #     if src and "t.prcdn.co/img" in src:
        #         base_file = src
        #         break
        
        # url = build_page_url(base_file, 1, scale=200)
        # print(url)

        # browser.wait(5)

        

        # for i in range(images.count()):
        #     src = images.nth(i).get_attribute("src")
        #     print(src)
            # if src and "t.prcdn.co/img" in src:
            #     base_file = src
            #     break

        # browser.locator("div[class='readingnav rn-right']").click()
        

        # # for i in range(images.count()):
        # #     src = images.nth(i).get_attribute("src")
        # #     print(src)
        #     # if src and "t.prcdn.co/img" in src:
        #     #     base_file = src
        #     #     break

        # browser.locator("div[class='readingnav rn-right']").click()
        # browser.wait(5)

        # for i in range(images.count()):
        #     src = images.nth(i).get_attribute("src")
        #     print(src)
            # if src and "t.prcdn.co/img" in src:
            #     base_file = src
            #     break
        
        

        

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        browser.close()


if __name__ == "__main__":

    main()