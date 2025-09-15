from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import os
import time

def setup_driver():
    options = Options()
    options.add_argument("--window-size=1280,1024")
    return webdriver.Chrome(options=options)

def wait_page_ready(driver, timeout=20):
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )

def wait_element(driver, locator, timeout=20):
    return WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located(locator)
    )

def main():
    driver = setup_driver()
    wait = WebDriverWait(driver, 20)

    try:
        driver.get("http://127.0.0.1:5000/")
        wait_page_ready(driver)

        wait.until(EC.visibility_of_element_located((By.ID, "reviewForm")))

        jur = wait_element(driver, (By.CSS_SELECTOR, 'input[name="person_type"][value="jur"]'))
        driver.execute_script("arguments[0].scrollIntoView();", jur)
        driver.execute_script("arguments[0].click();", jur)

        wait.until(EC.visibility_of_element_located((By.ID, "jur_section")))

        driver.find_element(By.ID, "inn").send_keys("7707083893")
        fetch_btn = wait_element(driver, (By.ID, "fetchCompanyBtn"))
        driver.execute_script("arguments[0].scrollIntoView();", fetch_btn)
        driver.execute_script("arguments[0].click();", fetch_btn)

        wait.until(lambda d: d.find_element(By.ID, "company_name").get_attribute("value") != "")

        Select(driver.find_element(By.ID, "scenario")).select_by_value("other")
        wait.until(EC.visibility_of_element_located((By.ID, "custom_reason")))
        driver.find_element(By.ID, "custom_reason").send_keys("Причина тестового отзыва")

        for fid, val in [("seller_name","Иван П."), ("seller_city","Москва"), ("seller_phone","1234")]:
            el = driver.find_element(By.ID, fid); el.clear(); el.send_keys(val)

        for fid, val in [("buyer_name","Алексей К."), ("buyer_city","СПб"), ("buyer_phone","5678")]:
            el = driver.find_element(By.ID, fid); el.clear(); el.send_keys(val)

        driver.find_element(By.ID, "content").send_keys("Тестовый отзыв о сервисе.")
        driver.find_element(By.ID, "email").send_keys("test@example.com")
        driver.find_element(By.ID, "agree").click()

        test_file = os.path.abspath("test_document.pdf")
        assert os.path.exists(test_file), f"Файл не найден: {test_file}"
        driver.find_element(By.ID, "document").send_keys(test_file)
        Select(driver.find_element(By.ID, "document_type")).select_by_value("contract")

        submit = wait_element(driver, (By.ID, "submitBtn"))
        driver.execute_script("arguments[0].scrollIntoView();", submit)
        driver.execute_script("arguments[0].click();", submit)

        wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "#notification.show")))
        notif = driver.find_element(By.ID, "notification")
        text = notif.text.strip()
        assert "Отзыв успешно отправлен на модерацию" in text, f"Получили: «{text}»"

        print("✅ Тест пройден успешно")

    except Exception as e:
        print("❌ Ошибка при тестировании:", e)
        raise

    finally:
        time.sleep(1)
        driver.quit()

if __name__ == "__main__":
    main()
