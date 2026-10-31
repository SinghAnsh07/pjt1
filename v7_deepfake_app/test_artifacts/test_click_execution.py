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

# Switch to specular iframe and click
iframes = driver.find_elements(By.TAG_NAME, "iframe")
for ifr in iframes:
    driver.switch_to.frame(ifr)
    sb = driver.find_elements(By.CLASS_NAME, "specular-button")
    if sb:
        print("Clicking SpecularButton via Selenium...")
        sb[0].click()
        break
    driver.switch_to.default_content()

driver.switch_to.default_content()
print("Clicked. Waiting 6s for Streamlit execution...")
time.sleep(6)

# Check if results are rendered
body_text = driver.find_element(By.TAG_NAME, "body").text
print("Has 'Screening Verdict':", "Screening Verdict" in body_text)
print("Has 'Acoustic Signal Preprocessing':", "Acoustic Signal Preprocessing" in body_text or "Forensic Methodology" in body_text)
print("Has 'Candidate Decision Threshold':", "Candidate Decision Threshold" in body_text or "0.4820" in body_text)

driver.quit()
