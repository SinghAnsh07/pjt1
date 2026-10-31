from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import shutil

opts = Options()
opts.add_argument('--headless=new')

# Tablet
opts.add_argument('--window-size=768,1024')
driver_tab = webdriver.Chrome(options=opts)
driver_tab.get('http://localhost:8501')
time.sleep(3)
driver_tab.save_screenshot(r'C:\Users\ANSH SINGH\Desktop\capstone1\v7_deepfake_app\UI_REDESIGN_SCREENSHOTS\final_tablet_empty.png')

bento_tab = driver_tab.find_element(By.XPATH, "//*[contains(., 'What the Analysis Examines')]")
driver_tab.execute_script("arguments[0].scrollIntoView({block: 'start'});", bento_tab)
time.sleep(1.5)
driver_tab.save_screenshot(r'C:\Users\ANSH SINGH\Desktop\capstone1\v7_deepfake_app\UI_REDESIGN_SCREENSHOTS\final_tablet_bento.png')
driver_tab.quit()

# Mobile
opts_mob = Options()
opts_mob.add_argument('--headless=new')
opts_mob.add_argument('--window-size=390,844')
driver_mob = webdriver.Chrome(options=opts_mob)
driver_mob.get('http://localhost:8501')
time.sleep(3)
driver_mob.save_screenshot(r'C:\Users\ANSH SINGH\Desktop\capstone1\v7_deepfake_app\UI_REDESIGN_SCREENSHOTS\final_mobile_empty.png')

bento_mob = driver_mob.find_element(By.XPATH, "//*[contains(., 'What the Analysis Examines')]")
driver_mob.execute_script("arguments[0].scrollIntoView({block: 'start'});", bento_mob)
time.sleep(1.5)
driver_mob.save_screenshot(r'C:\Users\ANSH SINGH\Desktop\capstone1\v7_deepfake_app\UI_REDESIGN_SCREENSHOTS\final_mobile_bento.png')
driver_mob.quit()

for f in ['final_tablet_empty.png', 'final_tablet_bento.png', 'final_mobile_empty.png', 'final_mobile_bento.png']:
    p1 = rf'C:\Users\ANSH SINGH\Desktop\capstone1\v7_deepfake_app\UI_REDESIGN_SCREENSHOTS\{f}'
    p2 = rf'C:\Users\ANSH SINGH\.gemini\antigravity\brain\be60057f-1574-49cf-8107-9c6970107535\UI_REDESIGN_SCREENSHOTS\{f}'
    shutil.copy2(p1, p2)

print("Tablet and Mobile screenshots captured and copied successfully!")
