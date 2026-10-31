from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import shutil

opts = Options()
opts.add_argument('--headless=new')
opts.add_argument('--window-size=1440,1200')
driver = webdriver.Chrome(options=opts)
driver.get('http://localhost:8501')
time.sleep(3)

# 1. Empty state MagicBento
print("Scrolling to empty state MagicBento...")
bento_elem = driver.find_element(By.XPATH, "//*[contains(., 'What the Analysis Examines')]")
driver.execute_script("arguments[0].scrollIntoView({block: 'start'});", bento_elem)
time.sleep(1.5)
driver.save_screenshot(r'C:\Users\ANSH SINGH\Desktop\capstone1\v7_deepfake_app\UI_REDESIGN_SCREENSHOTS\real_scroll_empty_bento.png')

# 2. Upload file
fi = driver.find_element(By.CSS_SELECTOR, 'input[type="file"]')
fi.send_keys(r'C:\Users\ANSH SINGH\Desktop\capstone1\v7_deepfake_app\test_artifacts\fixtures\sample_single_4s.wav')
time.sleep(3)

# 3. Click SpecularButton
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

# 4. Scroll to Verdict Card
verdict_elem = driver.find_element(By.XPATH, "//*[contains(text(), 'Patterns more consistent')]")
driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", verdict_elem)
time.sleep(1.5)
driver.save_screenshot(r'C:\Users\ANSH SINGH\Desktop\capstone1\v7_deepfake_app\UI_REDESIGN_SCREENSHOTS\real_scroll_verdict.png')

# 5. Scroll to Signal Insights / Spectrogram
spec_elem = driver.find_element(By.XPATH, "//*[contains(text(), 'Signal Insights')]")
driver.execute_script("arguments[0].scrollIntoView({block: 'start'});", spec_elem)
time.sleep(1.5)
driver.save_screenshot(r'C:\Users\ANSH SINGH\Desktop\capstone1\v7_deepfake_app\UI_REDESIGN_SCREENSHOTS\real_scroll_signal_insights.png')

# 6. Scroll to Export Documentation & Post-Analysis Bento
export_elem = driver.find_element(By.XPATH, "//*[contains(text(), 'Export Analysis Documentation')]")
driver.execute_script("arguments[0].scrollIntoView({block: 'start'});", export_elem)
time.sleep(1.5)
driver.save_screenshot(r'C:\Users\ANSH SINGH\Desktop\capstone1\v7_deepfake_app\UI_REDESIGN_SCREENSHOTS\real_scroll_export_and_bento.png')

# Copy to brain
for f in ['real_scroll_empty_bento.png', 'real_scroll_verdict.png', 'real_scroll_signal_insights.png', 'real_scroll_export_and_bento.png']:
    p1 = rf'C:\Users\ANSH SINGH\Desktop\capstone1\v7_deepfake_app\UI_REDESIGN_SCREENSHOTS\{f}'
    p2 = rf'C:\Users\ANSH SINGH\.gemini\antigravity\brain\be60057f-1574-49cf-8107-9c6970107535\UI_REDESIGN_SCREENSHOTS\{f}'
    shutil.copy2(p1, p2)

print("Real scroll screenshots captured and copied successfully!")
driver.quit()
