# i-Scream 웹페이지 수학문제 크롤링
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys # 키보드 역할
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd


options = Options()
options.add_experimental_option("detach", True) # 백그라운드 실행

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)


driver.get('https://www.i-scream.co.kr/user/main/MainPage.do')
time.sleep(2)  # 페이지가 완전히 로딩될 때까지 대기
driver.maximize_window()

home = driver.find_element(By.XPATH, '//*[@id="loginForm"]/div[1]/button')
home.click()

id = driver.find_element(By.XPATH, '//*[@id="idValue"]')
id.send_keys('')

pw = driver.find_element(By.XPATH, '//*[@id="pwValue"]')
pw.send_keys('')

btn = driver.find_element(By.XPATH, '//*[@id="loginButton"]')
btn.click()

popup = driver.find_element(By.XPATH, '//*[@id="modifyForm"]/a')
popup.click()

eval = driver.find_element(By.XPATH, '//*[@id="SkipToGNB"]/ul/li[2]/a')
eval.click()

# # 5-1
# fifth = driver.find_element(By.XPATH, '//*[@id="SkipToLNB"]/ul/li[1]/ol/li[5]')
# fifth.click()

# # 5-2는 다음을 추가
# second = driver.find_element(By.XPATH, '//*[@id="SkipToLNB"]/ul/li[2]/ol/li[2]')
# second.click()

# 6-1
sixth = driver.find_element(By.XPATH, '//*[@id="SkipToLNB"]/ul/li[1]/ol/li[6]')
sixth.click()

# 6-2는 다음을 추가
second = driver.find_element(By.XPATH, '//*[@id="SkipToLNB"]/ul/li[2]/ol/li[2]')
second.click()

bank = driver.find_element(By.XPATH, '//*[@id="SkipToContents"]/main/p/a[1]')
bank.click()

driver.switch_to.window(driver.window_handles[-1])

driver.execute_script("document.eForm.submit()")


# a = driver.find_element(by=By.XPATH, value='//*[@id="frmExam"]/div/table/tbody/tr[8]/td[4]/a')
# a.click()

# 문제지 8개
for i in range(8, 0, -1):
    a = driver.find_element(By.XPATH, f'//*[@id="frmExam"]/div/table/tbody/tr[{i}]/td[4]/a')
    a.click()

    driver.switch_to.window(driver.window_handles[-1])

    iframe = driver.find_element(By.CSS_SELECTOR, "#mainarea")
    driver.switch_to.frame(iframe)
    driver.implicitly_wait(3)

    problem_selector  = "divquestion"

    problems = []

    # 최초에 한 번은 데이터를 수집
    for j in driver.find_elements(By.CLASS_NAME, problem_selector):
        problems.append(j.text)

    # 페이지 끝까지 스크롤하는 함수
    def scroll_to_bottom(driver):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # 페이지가 로딩될 때까지 잠시 대기

    while True:
        bh = driver.execute_script("return document.body.scrollHeight") # 브라우저 상의 처음 높이
        print(bh)
        time.sleep(4)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") # 스크롤 내리기
        time.sleep(2)
        ah = driver.execute_script("return document.body.scrollHeight")
        if ah == bh:
            break

        # 새로운 요소들을 수집
        for j in driver.find_elements(By.CLASS_NAME, problem_selector):
            problems.append(j.text)

    print(len(problems))

    # 데이터프레임 생성
    data = {'문제': problems}

    df= pd.DataFrame(data)
    file_index = 9 - i  # 파일 번호 계산
    df.to_csv(f'6-2-{file_index}_problems.csv', index=False)

    # 다음 문제 수집을 위해 이전 창으로 돌아가기
    driver.switch_to.window(driver.window_handles[1])

driver.quit()