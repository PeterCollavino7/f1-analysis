"""
Keep the hosted app awake.

Streamlit Community Cloud puts an app to sleep after 12 hours without a
visit, and whoever arrives next -- someone following the link on Peter's
site, say -- meets a "Yes, get this app back up!" button and a minute of
waiting instead of the dashboard. A visit is all it takes to keep it up, so
.github/workflows/keep-awake.yml runs this every few hours: it opens the app
in headless Chrome, presses the wake-up button if the app did fall asleep,
and waits until the app itself has rendered, so the visit is a real session
and not just a fetch of the page around it.

Exits non-zero if the app never comes up, so a failed run shows in the
Actions tab.

Run with:  python keep_awake.py   (needs selenium and Chrome)
"""
import sys
import time

from selenium import webdriver
from selenium.webdriver.common.by import By

APP_URL = "https://pitwall-pc.streamlit.app/"


def app_rendered(driver):
    """True once the Streamlit app inside the host page has drawn itself."""
    for frame in driver.find_elements(By.CSS_SELECTOR, "iframe[src*='/~/+/']"):
        driver.switch_to.frame(frame)
        try:
            if driver.find_elements(By.CSS_SELECTOR, "[data-testid='stApp']"):
                return True
        finally:
            driver.switch_to.default_content()
    return False


def main():
    options = webdriver.ChromeOptions()
    for arg in ("--headless=new", "--no-sandbox", "--disable-dev-shm-usage", "--window-size=1400,900"):
        options.add_argument(arg)
    driver = webdriver.Chrome(options=options)
    try:
        driver.get(APP_URL)
        time.sleep(10)
        woke = False
        for button in driver.find_elements(By.TAG_NAME, "button"):
            if "get this app back up" in button.text.lower():
                button.click()
                woke = True
                break
        # A sleeping app takes a few minutes to boot; an awake one renders
        # in seconds.
        deadline = time.time() + (300 if woke else 90)
        up = False
        while time.time() < deadline:
            if app_rendered(driver):
                up = True
                break
            time.sleep(5)
        if up:
            time.sleep(20)  # stay a moment, as a visitor would
        print(("was asleep, woke it; " if woke else "") + ("app is up" if up else "app did not come up"))
        return 0 if up else 1
    finally:
        driver.quit()


if __name__ == "__main__":
    sys.exit(main())
