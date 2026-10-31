from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

opts = Options()
opts.add_argument('--headless=new')
opts.add_argument('--window-size=1440,900')
driver = webdriver.Chrome(options=opts)
driver.get('http://localhost:8501')
time.sleep(3)

fi = driver.find_element(By.CSS_SELECTOR, 'input[type="file"]')
fi.send_keys(r'C:\Users\ANSH SINGH\Desktop\capstone1\v7_deepfake_app\test_artifacts\fixtures\sample_single_4s.wav')
time.sleep(3)

print("Page title:", driver.title)

# Find iframes
iframes = driver.find_elements(By.TAG_NAME, "iframe")
print("Found iframes:", len(iframes))
for idx, ifr in enumerate(iframes):
    print(f"iframe {idx}: title={ifr.get_attribute('title')}, src={ifr.get_attribute('src')}")

# Switch to specular button iframe
for ifr in iframes:
    driver.switch_to.frame(ifr)
    sb = driver.find_elements(By.CLASS_NAME, "specular-button")
    if sb:
        print("Found specular button inside iframe:", sb[0].text)
        # Check parent document access
        has_parent_doc = driver.execute_script("return Boolean(window.parent && window.parent.document);")
        print("Has parent doc access:", has_parent_doc)
        parent_btns = driver.execute_script("""
            const btns = window.parent.document.querySelectorAll('button[kind="primary"], [data-testid="baseButton-primary"]');
            return Array.from(btns).map(b => ({text: b.innerText, class: b.className, disabled: b.disabled}));
        """)
        print("Parent primary buttons visible to iframe:", parent_btns)
    driver.switch_to.default_content()

driver.quit()
