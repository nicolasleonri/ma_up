from src.corpus_acquisition.browser import BrowserSession
from src.utils.config_loader import load_yaml

browser = BrowserSession(headless=True)
browser.start()

credentials = load_yaml("config/corpus_acquisition/credentials.yaml")
browser.goto("https://web.peruquiosco.pe/")

browser.wait_for_selector("button[class='sc-dkzDqf xFTNr']") # Iniciar Sesión Button
browser.click("button[class='sc-dkzDqf xFTNr']")

browser.wait(3)
login_frame = browser.page.frames[1]
login_frame.locator( "input[name='email']" ).fill(credentials["username"]) 
login_frame.locator( "input[type='password']" ).fill(credentials["password"]) 
login_frame.locator("button[type='submit']").click()

browser.wait_for_selector("button[class='sc-dkzDqf kwFoJp']", timeout=30000) # Cerrar Sesión Button
browser.wait(3)
browser.screenshot("after_login.png")

browser.close()