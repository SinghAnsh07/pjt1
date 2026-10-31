"""
Visual Acceptance and UI Inspection Script for Phase 8.

Launches the application locally, drives headless Chrome via Selenium to exercise
the complete visual workflow, and captures high-resolution screenshots into
v7_deepfake_app/test_artifacts/screenshots/.

Screenshots captured:
1. initial_state.png
2. single_window_analyzed.png
3. signal_visualization_tabs.png
4. report_download_section.png
5. multi_window_analyzed.png
6. invalid_input_error_state.png
7. reset_cleared_state.png
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
ARTIFACTS_DIR = os.path.join(APP_ROOT, "test_artifacts", "screenshots")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Generate temporary audio fixtures for UI testing
FIXTURES_DIR = os.path.join(APP_ROOT, "test_artifacts", "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)

single_path = os.path.join(FIXTURES_DIR, "sample_single_4s.wav")
sr = 16000
audio_single = (0.35 * np.sin(2 * np.pi * 440 * np.linspace(0, 4.0, sr * 4, endpoint=False))).astype(np.float32)
sf.write(single_path, audio_single, sr)

multi_path = os.path.join(FIXTURES_DIR, "sample_multi_10s.wav")
audio_multi = (0.35 * np.sin(2 * np.pi * 320 * np.linspace(0, 10.0, sr * 10, endpoint=False))).astype(np.float32)
sf.write(multi_path, audio_multi, sr)

corrupt_path = os.path.join(FIXTURES_DIR, "corrupt_file.wav")
with open(corrupt_path, "wb") as f:
    f.write(b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00INVALID_DATA")

PORT = 8599
cmd = [
    sys.executable,
    "-m",
    "streamlit",
    "run",
    os.path.join(APP_ROOT, "app.py"),
    "--server.headless=true",
    f"--server.port={PORT}",
]

print(f"Launching Streamlit on port {PORT}...")
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
time.sleep(5)

options = Options()
options.add_argument("--headless")
options.add_argument("--window-size=1280,1024")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 15)

try:
    url = f"http://localhost:{PORT}"
    print(f"Navigating to {url}...")
    driver.get(url)
    time.sleep(3)

    # 1. Initial State
    print("Capturing 1: Initial State...")
    driver.save_screenshot(os.path.join(ARTIFACTS_DIR, "1_initial_state.png"))

    # Find file input
    file_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="file"]')))

    # 2. Upload Single Window
    print("Uploading single-window fixture...")
    file_input.send_keys(single_path)
    time.sleep(3)

    # Click Analyze
    print("Clicking Analyze Audio...")
    analyze_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Analyze Audio')]")))
    analyze_btn.click()
    time.sleep(5)

    print("Capturing 2: Single-Window Analyzed...")
    driver.save_screenshot(os.path.join(ARTIFACTS_DIR, "2_single_window_analyzed.png"))

    # Scroll to Signal Visualization Tabs
    print("Capturing 3: Signal Visualization Tabs...")
    driver.execute_script("window.scrollTo(0, 600);")
    time.sleep(1)
    driver.save_screenshot(os.path.join(ARTIFACTS_DIR, "3_signal_visualization_tabs.png"))

    # Scroll to Report Download Section
    print("Capturing 4: Report Download Section...")
    driver.execute_script("window.scrollTo(0, 1200);")
    time.sleep(1)
    driver.save_screenshot(os.path.join(ARTIFACTS_DIR, "4_report_download_section.png"))

    # 3. Test Reset Action
    print("Testing Reset action...")
    driver.execute_script("window.scrollTo(0, 200);")
    time.sleep(1)
    reset_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Reset')]")))
    reset_btn.click()
    time.sleep(3)

    print("Capturing 5: Reset Cleared State...")
    driver.save_screenshot(os.path.join(ARTIFACTS_DIR, "7_reset_cleared_state.png"))

    # 4. Upload Multi-Window
    print("Uploading multi-window fixture...")
    file_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="file"]')))
    file_input.send_keys(multi_path)
    time.sleep(3)

    analyze_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Analyze Audio')]")))
    analyze_btn.click()
    time.sleep(6)

    print("Capturing 6: Multi-Window Analyzed...")
    driver.execute_script("window.scrollTo(0, 300);")
    time.sleep(1)
    driver.save_screenshot(os.path.join(ARTIFACTS_DIR, "5_multi_window_analyzed.png"))

    # 5. Test Invalid File
    print("Testing invalid input error handling...")
    reset_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Reset')]")))
    reset_btn.click()
    time.sleep(3)

    file_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="file"]')))
    file_input.send_keys(corrupt_path)
    time.sleep(3)

    print("Capturing 7: Invalid Input Error State...")
    driver.save_screenshot(os.path.join(ARTIFACTS_DIR, "6_invalid_input_error_state.png"))

    print("\nVisual acceptance audit completed successfully! All 7 screenshots saved.")

finally:
    driver.quit()
    proc.terminate()
    proc.kill()
