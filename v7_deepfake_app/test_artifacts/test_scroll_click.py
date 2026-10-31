from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

opts = Options()
opts.add_argument('--headless=new')
opts.add_argument('--window-size=1440,1200')
driver = webdriver.Chrome(options=opts)
driver.get('http://localhost:8501')
time.sleep(3)

fi = driver.find_element(By.CSS_SELECTOR, 'input[type="file"]')
fi.send_keys(r'C:\Users\ANSH SINGH\Desktop\capstone1\v7_deepfake_app\test_artifacts\fixtures\sample_single_4s.wav')
time.sleep(3)

btn = driver.find_element(By.CSS_SELECTOR, 'button[kind="primary"]')
driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", btn)
time.sleep(0.5)
driver.execute_script("arguments[0].click();", btn)

print("Clicked after scrollIntoView! Waiting 6s...")
time.sleep(6)

body_text = driver.find_element(By.TAG_NAME, "body").text
print("Has 'Screening Verdict':", "Screening Verdict" in body_text)
print("Has 'Authenticity Screening Results':", "Authenticity Screening Results" in body_text)
driver.quit()
