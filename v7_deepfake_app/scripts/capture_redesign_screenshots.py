"""
Visual Audit and Screenshot Automation for V7 Audio Authenticity Analyzer Redesign.

Exercises the full user journey with Selenium headless Chrome across desktop,
tablet, and mobile viewports, saving high-resolution screenshots to:
    v7_deepfake_app/UI_REDESIGN_SCREENSHOTS/
"""

import os
import sys
import time
import subprocess
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

FIXTURES_DIR = os.path.join(APP_ROOT, "test_artifacts", "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)

single_path = os.path.join(FIXTURES_DIR, "sample_single_4s.wav")
multi_path = os.path.join(FIXTURES_DIR, "sample_multi_10s.wav")
corrupt_path = os.path.join(FIXTURES_DIR, "corrupt_file.wav")

sr = 16000
if not os.path.exists(single_path):
    audio_single = (0.35 * np.sin(2 * np.pi * 440 * np.linspace(0, 4.0, sr * 4, endpoint=False))).astype(np.float32)
    sf.write(single_path, audio_single, sr)

if not os.path.exists(multi_path):
    audio_multi = (0.35 * np.sin(2 * np.pi * 320 * np.linspace(0, 10.0, sr * 10, endpoint=False))).astype(np.float32)
    sf.write(multi_path, audio_multi, sr)

if not os.path.exists(corrupt_path):
    with open(corrupt_path, "wb") as f:
        f.write(b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00INVALID_DATA")

PORT = 8597
cmd = [
    sys.executable,
    "-m",
    "streamlit",
    "run",
    os.path.join(APP_ROOT, "app.py"),
    "--server.headless=true",
    f"--server.port={PORT}",
]

print(f"Starting Streamlit test instance on port {PORT}...")
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
time.sleep(5)

def get_driver(width=1440, height=900):
    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument(f"--window-size={width},{height}")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=opts)

def safe_click(driver, elem):
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", elem)
    time.sleep(0.5)
    driver.execute_script("arguments[0].click();", elem)
    time.sleep(1)

driver = get_driver(1440, 900)
wait = WebDriverWait(driver, 15)

try:
    url = f"http://localhost:{PORT}"
    print(f"Navigating to {url}...")
    driver.get(url)
    time.sleep(3)

    # 1. Initial Empty State
    print("Capturing 01: Initial Empty State...")
    driver.save_screenshot(os.path.join(SCREENSHOTS_DIR, "01_initial_empty_state.png"))

    # 2. File Selected Before Analysis
    print("Uploading single-window fixture...")
    file_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="file"]')))
    file_input.send_keys(single_path)
    time.sleep(3)

    print("Capturing 02: File Selected Before Analysis...")
    driver.save_screenshot(os.path.join(SCREENSHOTS_DIR, "02_file_selected_before_analysis.png"))

    # 3. Completed Single-Window Result
    print("Clicking Analyze Audio...")
    analyze_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Analyze Audio')]")))
    safe_click(driver, analyze_btn)
    time.sleep(4)

    print("Capturing 03: Completed Single-Window Result...")
    driver.execute_script("window.scrollTo(0, 320);")
    time.sleep(1)
    driver.save_screenshot(os.path.join(SCREENSHOTS_DIR, "03_single_window_result.png"))

    # 4. Expanded Technical Details
    print("Expanding Technical Details...")
    try:
        expander = driver.find_element(By.XPATH, "//details[contains(., 'How this result was produced')]")
        summary_elem = expander.find_element(By.TAG_NAME, "summary")
        safe_click(driver, summary_elem)
        time.sleep(1)
    except Exception as e:
        print(f"Note on expander: {e}")

    print("Capturing 04: Technical Details Expanded...")
    driver.save_screenshot(os.path.join(SCREENSHOTS_DIR, "04_technical_details_expanded.png"))

    # 5. Waveform View
    print("Navigating to Waveform View...")
    try:
        wave_tab = driver.find_element(By.XPATH, "//button[contains(., 'Waveform')]")
        safe_click(driver, wave_tab)
        time.sleep(1)
    except Exception as e:
        print(f"Note on wave tab: {e}")

    print("Capturing 05: Waveform Visualization...")
    driver.save_screenshot(os.path.join(SCREENSHOTS_DIR, "05_waveform_view.png"))

    # 6. Spectrogram View
    print("Navigating to Spectrogram View...")
    try:
        spec_tab = driver.find_element(By.XPATH, "//button[contains(., 'Spectrogram')]")
        safe_click(driver, spec_tab)
        time.sleep(1.5)
    except Exception as e:
        print(f"Note on spec tab: {e}")

    print("Capturing 06: Spectrogram Visualization...")
    driver.save_screenshot(os.path.join(SCREENSHOTS_DIR, "06_spectrogram_view.png"))

    # 7. Advanced Model Views
    print("Navigating to Advanced Model Views...")
    try:
        feat_tab = driver.find_element(By.XPATH, "//button[contains(., 'Advanced Model Views')]")
        safe_click(driver, feat_tab)
        time.sleep(2)
    except Exception as e:
        print(f"Note on feat tab: {e}")

    print("Capturing 07: Advanced Model Views...")
    driver.save_screenshot(os.path.join(SCREENSHOTS_DIR, "07_advanced_model_views.png"))

    # 8. Export Controls
    print("Navigating to Export Controls...")
    try:
        export_elem = driver.find_element(By.XPATH, "//*[contains(text(), 'Export Analysis Documentation')]")
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", export_elem)
        time.sleep(1)
    except Exception:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight - 600);")
        time.sleep(1)

    print("Capturing 08: Export Controls...")
    driver.save_screenshot(os.path.join(SCREENSHOTS_DIR, "08_export_controls.png"))

    # 9. Reset and Multi-Window Analysis
    print("Resetting and uploading multi-window fixture...")
    reset_btn = wait.until(EC.presence_of_element_located((By.XPATH, "//button[contains(., 'Reset')]")))
    safe_click(driver, reset_btn)
    time.sleep(2)

    file_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="file"]')))
    file_input.send_keys(multi_path)
    time.sleep(3)

    analyze_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Analyze Audio')]")))
    safe_click(driver, analyze_btn)
    time.sleep(6)

    print("Navigating to Timeline tab...")
    try:
        time_tab = driver.find_element(By.XPATH, "//button[contains(., 'Analysis Timeline')]")
        safe_click(driver, time_tab)
        time.sleep(1.5)
    except Exception as e:
        print(f"Note on time tab: {e}")

    print("Capturing 09: Multi-Window Result and Timeline...")
    driver.save_screenshot(os.path.join(SCREENSHOTS_DIR, "09_multi_window_result_and_timeline.png"))

    # 10. Error Handling (Corrupt File)
    print("Resetting and testing corrupt input rejection...")
    reset_btn = wait.until(EC.presence_of_element_located((By.XPATH, "//button[contains(., 'Reset')]")))
    safe_click(driver, reset_btn)
    time.sleep(2)

    file_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="file"]')))
    file_input.send_keys(corrupt_path)
    time.sleep(3)

    print("Capturing 10: Error Handling...")
    driver.save_screenshot(os.path.join(SCREENSHOTS_DIR, "10_error_handling_corrupt_file.png"))

    driver.quit()

    # 11. Responsive Tablet (768px)
    print("Testing Responsive Layout on Tablet (768x1024)...")
    driver_tablet = get_driver(768, 1024)
    driver_tablet.get(url)
    time.sleep(3)
    tab_input = driver_tablet.find_element(By.CSS_SELECTOR, 'input[type="file"]')
    tab_input.send_keys(single_path)
    time.sleep(2)
    tab_btn = driver_tablet.find_element(By.XPATH, "//button[contains(., 'Analyze Audio')]")
    safe_click(driver_tablet, tab_btn)
    time.sleep(5)
    driver_tablet.save_screenshot(os.path.join(SCREENSHOTS_DIR, "11_responsive_tablet_768px.png"))
    driver_tablet.quit()

    # 12. Responsive Mobile (390px)
    print("Testing Responsive Layout on Mobile (390x844)...")
    driver_mobile = get_driver(390, 844)
    driver_mobile.get(url)
    time.sleep(3)
    mob_input = driver_mobile.find_element(By.CSS_SELECTOR, 'input[type="file"]')
    mob_input.send_keys(single_path)
    time.sleep(2)
    mob_btn = driver_mobile.find_element(By.XPATH, "//button[contains(., 'Analyze Audio')]")
    safe_click(driver_mobile, mob_btn)
    time.sleep(5)
    driver_mobile.save_screenshot(os.path.join(SCREENSHOTS_DIR, "12_responsive_mobile_390px.png"))
    driver_mobile.quit()

    print(f"\nAll 12 redesign screenshots captured successfully in:\n  {SCREENSHOTS_DIR}")

finally:
    proc.terminate()
    proc.kill()
