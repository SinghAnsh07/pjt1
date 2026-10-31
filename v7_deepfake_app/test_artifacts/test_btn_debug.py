from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

opts = Options()
opts.add_argument('--headless=new')
opts.set_capability('goog:loggingPrefs', {'browser': 'ALL'})
driver = webdriver.Chrome(options=opts)
driver.get('http://localhost:8501')
time.sleep(3)

fi = driver.find_element(By.CSS_SELECTOR, 'input[type="file"]')
fi.send_keys(r'C:\Users\ANSH SINGH\Desktop\capstone1\v7_deepfake_app\test_artifacts\fixtures\sample_single_4s.wav')
time.sleep(3)

print("Browser logs before click:")
for log in driver.get_log('browser'):
    print(log)

# Let's inspect the button element
btn = driver.find_element(By.CSS_SELECTOR, 'button[kind="primary"]')
print("Button displayed:", btn.is_displayed())
print("Button enabled:", btn.is_enabled())
print("Button size:", btn.size)
print("Button location:", btn.location)

# Click using driver
btn.click()
time.sleep(5)

print("\nBrowser logs after click:")
for log in driver.get_log('browser'):
    print(log)

body_text = driver.find_element(By.TAG_NAME, "body").text
print("Has 'Screening Verdict':", "Screening Verdict" in body_text)
driver.quit()
