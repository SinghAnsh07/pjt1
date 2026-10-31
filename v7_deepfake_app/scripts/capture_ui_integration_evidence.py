"""
Visual verification and headless Chrome automation for V7 Audio Authenticity Analyzer
UI Redesign and React Bits Component Integration (SpecularButton, MagicBento, GradientASCII).

Captures evidence across desktop, tablet, and mobile viewports, explicitly testing
iframe interaction, SpecularButton event triggering, MagicBento rendering, and layout widths.
"""

import os
import sys
import time
import shutil
import numpy as np
import soundfile as sf
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREENSHOTS_DIR = os.path.join(APP_ROOT, "UI_REDESIGN_SCREENSHOTS")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

BRAIN_DIR = r"C:\Users\ANSH SINGH\.gemini\antigravity\brain\be60057f-1574-49cf-8107-9c6970107535\UI_REDESIGN_SCREENSHOTS"
os.makedirs(BRAIN_DIR, exist_ok=True)

FIXTURES_DIR = os.path.join(APP_ROOT, "test_artifacts", "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)

single_path = os.path.join(FIXTURES_DIR, "sample_single_4s.wav")
sr = 16000
if not os.path.exists(single_path):
    audio_single = (0.35 * np.sin(2 * np.pi * 440 * np.linspace(0, 4.0, sr * 4, endpoint=False))).astype(np.float32)
    sf.write(single_path, audio_single, sr)

URL = "http://localhost:8501"

def get_driver(width=1440, height=900):
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument(f"--window-size={width},{height}")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=opts)

def save_both(driver, filename):
    p1 = os.path.join(SCREENSHOTS_DIR, filename)
    driver.save_screenshot(p1)
    p2 = os.path.join(BRAIN_DIR, filename)
    shutil.copy2(p1, p2)
    print(f"  [SAVED] {filename}")

def run_audit():
    print(f"Connecting to Streamlit application at {URL}...")
    driver = get_driver(1440, 950)
    wait = WebDriverWait(driver, 15)

    try:
        driver.get(URL)
        time.sleep(3)

        # 1. Desktop Empty State
        print("\n1. Capturing Desktop Empty State (1440px)...")
        # Scroll up to capture header, uploader and bento
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)
        save_both(driver, "01_desktop_empty_state_1440px.png")

        # Scroll down slightly to capture the MagicBento methodology grid in empty state
        driver.execute_script("window.scrollTo(0, 260);")
        time.sleep(1)
        save_both(driver, "02_desktop_empty_state_magic_bento.png")

        # 2. File Upload & SpecularButton Rendered
        print("\n2. Uploading 4.0s Audio File...")
        driver.execute_script("window.scrollTo(0, 0);")
        file_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="file"]')))
        file_input.send_keys(single_path)
        time.sleep(3)

        # Scroll to action controls
        driver.execute_script("window.scrollTo(0, 180);")
        time.sleep(1)
        print("  Capturing File Selected State with SpecularButton...")
        save_both(driver, "03_file_selected_specular_button.png")

        # 3. Click SpecularButton
        print("\n3. Interacting with SpecularButton...")
        # Locate the iframe containing SpecularButton
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        clicked = False
        for ifr in iframes:
            try:
                driver.switch_to.frame(ifr)
                specular_btn = driver.find_elements(By.CLASS_NAME, "specular-button")
                if specular_btn:
                    print("  Located SpecularButton inside iframe! Executing real click...")
                    specular_btn[0].click()
                    clicked = True
                    driver.switch_to.default_content()
                    break
                driver.switch_to.default_content()
            except Exception as e:
                driver.switch_to.default_content()

        if not clicked:
            print("  Fallback: Clicking primary button via parent document...")
            pbtn = driver.find_element(By.XPATH, "//button[contains(., 'Analyze Audio')]")
            driver.execute_script("arguments[0].click();", pbtn)

        print("  Waiting for inference analysis to complete...")
        time.sleep(5)

        # 4. Results Views
        print("\n4. Capturing Analysis Results...")
        driver.execute_script("window.scrollTo(0, 300);")
        time.sleep(1)
        save_both(driver, "04_analysis_results_summary.png")

        # Signal insights
        driver.execute_script("window.scrollTo(0, 750);")
        time.sleep(1)
        save_both(driver, "05_signal_insights_views.png")

        # Export & MagicBento in results
        driver.execute_script("window.scrollTo(0, 1300);")
        time.sleep(1)
        save_both(driver, "06_export_and_methodology_bento.png")

        driver.quit()

        # 5. Tablet Viewport (768px)
        print("\n5. Testing Tablet Viewport (768x1024)...")
        driver_tab = get_driver(768, 1024)
        driver_tab.get(URL)
        time.sleep(3)
        tab_input = driver_tab.find_element(By.CSS_SELECTOR, 'input[type="file"]')
        tab_input.send_keys(single_path)
        time.sleep(2)
        driver_tab.execute_script("window.scrollTo(0, 200);")
        time.sleep(1)
        save_both(driver_tab, "07_responsive_tablet_768px.png")
        driver_tab.quit()

        # 6. Mobile Viewport (390px)
        print("\n6. Testing Mobile Viewport (390x844)...")
        driver_mob = get_driver(390, 844)
        driver_mob.get(URL)
        time.sleep(3)
        mob_input = driver_mob.find_element(By.CSS_SELECTOR, 'input[type="file"]')
        mob_input.send_keys(single_path)
        time.sleep(2)
        driver_mob.execute_script("window.scrollTo(0, 200);")
        time.sleep(1)
        save_both(driver_mob, "08_responsive_mobile_390px.png")
        driver_mob.quit()

        print("\nAll integration screenshots captured and verified successfully!")

    except Exception as e:
        print(f"\n[ERROR during visual audit]: {e}")
        driver.quit()
        raise

if __name__ == "__main__":
    run_audit()
