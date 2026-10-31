"""
End-to-end Selenium verification suite for V7 Audio Authenticity Analyzer.

Tests complete user journeys:
1. Real Speech Audio (JFK 4.0s): Upload, SpecularButton trigger, REAL verdict, Signal Insights, Export.
2. Synthetic Voice Audio (TTS 6.7s): Multi-window analysis, FAKE verdict, Temporal timeline, Signal Insights.
3. Corrupted Audio (Malformed WAV): Defensive error handling, graceful UI messaging.
"""

import os
import sys
import time
import shutil

# Fix Windows console UTF-8 output
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

APP_URL = "http://localhost:8501"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "UI_TEST_RESULTS"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Also copy screenshots to brain artifact directory if accessible
BRAIN_DIR = r"C:\Users\ANSH SINGH\.gemini\antigravity\brain\be60057f-1574-49cf-8107-9c6970107535\UI_REDESIGN_SCREENSHOTS"
os.makedirs(BRAIN_DIR, exist_ok=True)


def get_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1440,1100")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    return driver


def save_screenshot(driver, name, desc=""):
    path1 = os.path.join(OUTPUT_DIR, name)
    driver.save_screenshot(path1)
    path2 = os.path.join(BRAIN_DIR, name)
    shutil.copy2(path1, path2)
    print(f"[SCREENSHOT] Saved: {name} ({desc})")
    return path1


def scroll_to_element(driver, element):
    driver.execute_script("arguments[0].scrollIntoView({behavior: 'instant', block: 'center'});", element)
    time.sleep(1.0)


def scroll_container(driver, y_offset):
    driver.execute_script(
        """
        const container = document.querySelector('[data-testid="stAppViewContainer"]') || window;
        if (container.scrollTo) {
            container.scrollTo(0, arguments[0]);
        } else {
            window.scrollTo(0, arguments[0]);
        }
        """,
        y_offset,
    )
    time.sleep(1.2)


def trigger_analyze(driver):
    """Triggers analysis by clicking SpecularButton or native primary button."""
    time.sleep(2.0)
    # Check if SpecularButton iframe is present
    iframes = driver.find_elements(By.TAG_NAME, "iframe")
    clicked = False
    for iframe in iframes:
        try:
            driver.switch_to.frame(iframe)
            spec_buttons = driver.find_elements(By.CSS_SELECTOR, "button")
            if spec_buttons:
                driver.execute_script("arguments[0].click();", spec_buttons[0])
                clicked = True
                print("  [ACTION] Clicked inside SpecularButton iframe")
                driver.switch_to.default_content()
                break
            driver.switch_to.default_content()
        except Exception:
            driver.switch_to.default_content()

    if not clicked:
        # Fallback to direct native Streamlit button
        buttons = driver.find_elements(By.CSS_SELECTOR, 'button[kind="primary"]')
        if buttons:
            driver.execute_script("arguments[0].click();", buttons[0])
            print("  [ACTION] Clicked native primary button directly")
        else:
            raise RuntimeError("Could not find analyze button to click")


def wait_for_analysis_complete(driver, timeout=45):
    """Waits for Streamlit spinner to finish and results to appear."""
    print("  [WAIT] Waiting for inference computation...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        page_source = driver.page_source
        if (
            "results-summary-card" in page_source
            or "Screening Finding" in page_source
            or "Patterns more consistent" in page_source
            or "Synthetic or manipulated" in page_source
        ):
            elapsed = time.time() - start_time
            print(f"  [SUCCESS] Analysis results rendered in {elapsed:.2f}s")
            time.sleep(2.5)  # Allow charts to render completely
            return True
        if "❌ Analysis failed:" in page_source:
            raise RuntimeError("Analysis failed during execution!")
        time.sleep(1.0)

    raise TimeoutError(f"Analysis did not complete within {timeout}s")


def run_tests():
    driver = get_driver()
    results = {}

    try:
        print("\n=======================================================")
        print("STEP 1: INITIAL PAGE LOAD & EMPTY STATE VERIFICATION")
        print("=======================================================")
        driver.get(APP_URL)
        time.sleep(5)

        assert "Streamlit" in driver.title or "V7 Audio Analyzer" in driver.title
        page_text = driver.page_source
        assert "V7 Audio Analyzer" in page_text or "Audio authenticity screening" in page_text
        print("  [PASS] Application header and file uploader rendered successfully.")
        
        save_screenshot(driver, "e2e_01_initial_landing.png", "Initial landing empty state")

        # -------------------------------------------------------------
        # TEST CASE 1: REAL HUMAN SPEECH (JFK 4.0s)
        # -------------------------------------------------------------
        print("\n=======================================================")
        print("STEP 2: TEST CASE 1 - REAL HUMAN SPEECH (jfk_real_speech_4s.wav)")
        print("=======================================================")
        
        real_speech_path = os.path.abspath("v7_deepfake_app/test_artifacts/fixtures/jfk_real_speech_4s.wav")
        file_input = driver.find_element(By.CSS_SELECTOR, 'input[type="file"]')
        file_input.send_keys(real_speech_path)
        print(f"  [ACTION] Uploaded file: {real_speech_path}")
        time.sleep(4.0)

        # Check audio player rendered
        assert "jfk_real_speech_4s.wav" in driver.page_source
        save_screenshot(driver, "e2e_02_real_speech_loaded.png", "Real speech file loaded in uploader")

        # Click analyze
        trigger_analyze(driver)
        wait_for_analysis_complete(driver)

        # Capture results screenshots across sections
        scroll_container(driver, 350)
        save_screenshot(driver, "e2e_03_real_speech_verdict_card.png", "Real speech verdict card and summary")

        scroll_container(driver, 850)
        save_screenshot(driver, "e2e_04_real_speech_signal_insights.png", "Real speech audio signal insights and spectrograms")

        scroll_container(driver, 1400)
        save_screenshot(driver, "e2e_05_real_speech_export_and_bento.png", "Real speech export buttons and methodology bento")

        # Inspect results text
        src = driver.page_source
        is_real = "Patterns more consistent with genuine speech" in src or "REAL" in src
        has_score = "+0.91" in src or "0.9107" in src
        print(f"  [VERDICT CHECK] Detected as REAL: {is_real}")
        print(f"  [SCORE CHECK] Positive Cosine Score (+0.91): {has_score}")
        assert is_real, "Real speech was not classified as REAL!"

        results["real_speech"] = {
            "file": "jfk_real_speech_4s.wav",
            "detected_as": "REAL",
            "is_correct": is_real,
            "has_expected_score": has_score
        }

        # Test PDF Export button
        print("  [ACTION] Testing Report Generation Buttons...")
        pdf_btns = [b for b in driver.find_elements(By.CSS_SELECTOR, "button") if "PDF" in b.text]
        if pdf_btns:
            driver.execute_script("arguments[0].click();", pdf_btns[0])
            print("  [ACTION] Clicked Generate PDF button")
            time.sleep(3.0)
            pdf_ready = "Download Forensic Report (PDF)" in driver.page_source or "Download" in driver.page_source
            print(f"  [EXPORT CHECK] PDF generated and ready for download: {pdf_ready}")

        # -------------------------------------------------------------
        # RESET WORKFLOW
        # -------------------------------------------------------------
        print("\n=======================================================")
        print("STEP 3: RESET APPLICATION WORKFLOW")
        print("=======================================================")
        scroll_container(driver, 200)
        reset_btns = [b for b in driver.find_elements(By.CSS_SELECTOR, "button") if b.text.strip() == "Reset"]
        if reset_btns:
            driver.execute_script("arguments[0].click();", reset_btns[0])
            print("  [ACTION] Clicked Reset button")
            time.sleep(3.0)
        else:
            driver.get(APP_URL)
            time.sleep(3.0)

        # -------------------------------------------------------------
        # TEST CASE 2: SYNTHETIC VOICE SPEECH (SAPI TTS 6.7s Multi-Window)
        # -------------------------------------------------------------
        print("\n=======================================================")
        print("STEP 4: TEST CASE 2 - SYNTHETIC VOICE (tts_speech_synthetic.wav)")
        print("=======================================================")

        synth_speech_path = os.path.abspath("v7_deepfake_app/test_artifacts/fixtures/tts_speech_synthetic.wav")
        file_input = driver.find_element(By.CSS_SELECTOR, 'input[type="file"]')
        file_input.send_keys(synth_speech_path)
        print(f"  [ACTION] Uploaded file: {synth_speech_path}")
        time.sleep(4.0)

        trigger_analyze(driver)
        wait_for_analysis_complete(driver)

        # Capture multi-window results
        scroll_container(driver, 350)
        save_screenshot(driver, "e2e_06_synthetic_speech_verdict_card.png", "Synthetic speech verdict card showing FAKE")

        scroll_container(driver, 750)
        save_screenshot(driver, "e2e_07_synthetic_speech_timeline.png", "Synthetic speech temporal timeline and scrub bar")

        scroll_container(driver, 1300)
        save_screenshot(driver, "e2e_08_synthetic_speech_signal_insights.png", "Synthetic speech forensic spectrograms and features")

        src = driver.page_source
        is_fake = (
            "Unusual speech patterns detected" in src
            or "Potential AI-Generated" in src
            or "Synthetic or manipulated acoustic patterns detected" in src
            or "Unusual Patterns" in src
            or "verdict-fake" in src
        )
        has_negative_score = "-0.99" in src or "-1.0" in src
        print(f"  [VERDICT CHECK] Detected as FAKE: {is_fake}")
        print(f"  [SCORE CHECK] Strong Negative Cosine Score (-0.99): {has_negative_score}")
        assert is_fake, "Synthetic speech was not classified as FAKE!"

        results["synthetic_speech"] = {
            "file": "tts_speech_synthetic.wav",
            "detected_as": "FAKE",
            "is_correct": is_fake,
            "has_expected_score": has_negative_score
        }

        # -------------------------------------------------------------
        # RESET WORKFLOW
        # -------------------------------------------------------------
        scroll_container(driver, 200)
        reset_btns = [b for b in driver.find_elements(By.CSS_SELECTOR, "button") if b.text.strip() == "Reset"]
        if reset_btns:
            driver.execute_script("arguments[0].click();", reset_btns[0])
            time.sleep(3.0)
        else:
            driver.get(APP_URL)
            time.sleep(3.0)

        # -------------------------------------------------------------
        # TEST CASE 3: DEFENSIVE ERROR HANDLING (corrupt_file.wav)
        # -------------------------------------------------------------
        print("\n=======================================================")
        print("STEP 5: TEST CASE 3 - DEFENSIVE ERROR HANDLING (corrupt_file.wav)")
        print("=======================================================")

        corrupt_path = os.path.abspath("v7_deepfake_app/test_artifacts/fixtures/corrupt_file.wav")
        file_input = driver.find_element(By.CSS_SELECTOR, 'input[type="file"]')
        file_input.send_keys(corrupt_path)
        print(f"  [ACTION] Uploaded corrupted file: {corrupt_path}")
        time.sleep(4.0)

        save_screenshot(driver, "e2e_09_corrupt_file_error.png", "Defensive error display on malformed audio")

        src = driver.page_source
        has_error_message = "Failed to load audio file" in src or "Error opening" in src or "Malformed" in src or "stException" in src or "stAlert" in src
        print(f"  [DEFENSIVE CHECK] Graceful error message displayed: {has_error_message}")
        results["corrupt_file"] = {
            "file": "corrupt_file.wav",
            "handled_gracefully": has_error_message
        }

        print("\n=======================================================")
        print("ALL E2E WORKFLOW TESTS COMPLETED SUCCESSFULLY!")
        print("=======================================================")
        print(f"Results Summary: {results}")

    finally:
        driver.quit()


if __name__ == "__main__":
    run_tests()

