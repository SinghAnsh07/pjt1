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

print("Calling .click() directly on parent button via execute_script...")
res = driver.execute_script("""
    const btn = document.querySelector('button[kind="primary"], [data-testid="baseButton-primary"]');
    if (btn) {
        btn.click();
        return "Clicked button: " + btn.innerText;
    }
    return "Not found";
""")
print(res)

print("Waiting 6s...")
time.sleep(6)

body_text = driver.find_element(By.TAG_NAME, "body").text
print("Has 'Screening Verdict':", "Screening Verdict" in body_text)
print("Has 'Authenticity Screening Results':", "Authenticity Screening Results" in body_text)
driver.quit()
