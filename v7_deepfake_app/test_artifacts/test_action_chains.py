from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
import time

opts = Options()
opts.add_argument('--headless=new')
opts.add_argument('--window-size=1440,1200')
driver = webdriver.Chrome(options=opts)
driver.get('http://localhost:8501')
time.sleep(3)

fi = driver.find_element(By.CSS_SELECTOR, 'input[type="file"]')
fi.send_keys(r'C:\Users\ANSH SINGH\Desktop\capstone1\v7_deepfake_app\test_artifacts\fixtures\sample_single_4s.wav')
time.sleep(4)

print("Audio uploaded. Locating primary button...")
btn = driver.find_element(By.CSS_SELECTOR, 'button[kind="primary"]')
print("Found button:", btn.text, "tag:", btn.tag_name)

# Use Selenium ActionChains to perform a real user mouse click!
actions = ActionChains(driver)
actions.move_to_element(btn).click().perform()
print("ActionChains click executed! Waiting 6s for inference...")
time.sleep(6)

body = driver.find_element(By.TAG_NAME, "body").text
print("Has 'Screening Verdict':", "Screening Verdict" in body)
print("Has 'Candidate Decision Threshold':", "Candidate Decision Threshold" in body)

if "Screening Verdict" in body:
    print("SUCCESS: Streamlit inference triggered and executed!")
else:
    print("FAILURE: Inference not triggered.")
    print("Page excerpt:\n", body[:600])

driver.quit()
