from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

opts = Options()
opts.add_argument('--headless=new')
driver = webdriver.Chrome(options=opts)
driver.get('http://localhost:8501')
time.sleep(3)

fi = driver.find_element(By.CSS_SELECTOR, 'input[type="file"]')
fi.send_keys(r'C:\Users\ANSH SINGH\Desktop\capstone1\v7_deepfake_app\test_artifacts\fixtures\sample_single_4s.wav')
time.sleep(3)

print("Testing different dispatch event methods on the Streamlit button...")
res = driver.execute_script("""
    const btn = document.querySelector('button[kind="primary"], [data-testid="baseButton-primary"]');
    if (!btn) return "Button not found";

    // Method 1: pointerdown + pointerup + click
    btn.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, cancelable: true, view: window }));
    btn.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true, view: window }));
    btn.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, cancelable: true, view: window }));
    btn.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, cancelable: true, view: window }));
    btn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, view: window }));
    
    return "Events dispatched to " + btn.innerText;
""")
print(res)

print("Waiting 6s...")
time.sleep(6)

body_text = driver.find_element(By.TAG_NAME, "body").text
print("Has 'Screening Verdict':", "Screening Verdict" in body_text)
print("Has 'Authenticity Screening Results':", "Authenticity Screening Results" in body_text)

# Also test: What if Selenium clicks the parent button using driver.find_element(...).click()?
if "Screening Verdict" not in body_text:
    print("Testing Selenium native element.click()...")
    btn = driver.find_element(By.CSS_SELECTOR, 'button[kind="primary"], [data-testid="baseButton-primary"]')
    btn.click()
    time.sleep(6)
    body_text2 = driver.find_element(By.TAG_NAME, "body").text
    print("After Selenium .click() - Has 'Screening Verdict':", "Screening Verdict" in body_text2)

driver.quit()
