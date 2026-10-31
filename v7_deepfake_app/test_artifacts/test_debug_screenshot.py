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

btn = driver.find_element(By.CSS_SELECTOR, 'button[kind="primary"]')
driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", btn)
time.sleep(0.5)

# Click
btn.click()
time.sleep(2)
driver.save_screenshot(r'C:\Users\ANSH SINGH\Desktop\capstone1\v7_deepfake_app\UI_REDESIGN_SCREENSHOTS\debug_after_click_2s.png')
time.sleep(4)
driver.save_screenshot(r'C:\Users\ANSH SINGH\Desktop\capstone1\v7_deepfake_app\UI_REDESIGN_SCREENSHOTS\debug_after_click_6s.png')

print("Screenshots saved.")
driver.quit()
