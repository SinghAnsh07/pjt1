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

# Screenshot before click (file selected)
driver.execute_script("window.scrollTo(0, 180);")
time.sleep(1)
driver.save_screenshot(r'C:\Users\ANSH SINGH\Desktop\capstone1\v7_deepfake_app\UI_REDESIGN_SCREENSHOTS\verified_specular_button_clean.png')

# Find specular iframe and click inside it
iframes = driver.find_elements(By.TAG_NAME, "iframe")
for ifr in iframes:
    driver.switch_to.frame(ifr)
    sb = driver.find_elements(By.CLASS_NAME, "specular-button")
    if sb:
        print("Clicking SpecularButton inside iframe...")
        sb[0].click()
        break
    driver.switch_to.default_content()

driver.switch_to.default_content()
print("Waiting 6s for inference to execute...")
time.sleep(6)

driver.execute_script("window.scrollTo(0, 360);")
time.sleep(1)
driver.save_screenshot(r'C:\Users\ANSH SINGH\Desktop\capstone1\v7_deepfake_app\UI_REDESIGN_SCREENSHOTS\verified_results_clean.png')

# Scroll further to capture MagicBento section
driver.execute_script("window.scrollTo(0, 1400);")
time.sleep(1)
driver.save_screenshot(r'C:\Users\ANSH SINGH\Desktop\capstone1\v7_deepfake_app\UI_REDESIGN_SCREENSHOTS\verified_magic_bento_clean.png')

body = driver.find_element(By.TAG_NAME, "body").text
print("Has 'Patterns more consistent':", "Patterns more consistent" in body)
print("Screenshots captured successfully!")

driver.quit()
